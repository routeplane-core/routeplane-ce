//! Hermetic integration tests for POST /v1/audio/transcriptions (parity with
//! OpenAI's /v1/audio/transcriptions; OpenAI + Groq backed). Unlike the other
//! verticals the inbound body is `multipart/form-data` (binary audio), so these
//! tests drive the handler through a REAL axum router (the `Multipart` extractor
//! cannot be hand-constructed) with the same `auth_middleware` wiring as
//! `main.rs`. The only "network" is a localhost wiremock standing in for the
//! upstream. Covers:
//!   * multipart parse: `file` + `model` extracted; the upstream receives a
//!     multipart body at /v1/audio/transcriptions and `{text}` is mapped back.
//!   * Groq routing hits /openai/v1/audio/transcriptions (the base-path footgun).
//!   * missing `file` ⇒ clean 400 (no panic); missing `model` ⇒ clean 400.
//!   * routing to a provider without a first-party STT endpoint (cohere) ⇒ 422
//!     transcription_not_supported.
//!   * the route is auth-gated (no key ⇒ 401) via the real auth_middleware.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::response::Response;
use axum::routing::post;
use axum::Router;
use routeplane::audio_api::{transcriptions, translations};
use routeplane::auth::{auth_middleware, shared_auth_state, AuthState, SharedAuthState};
use routeplane::proxy::{AppState, ProviderRegistry};
use routeplane_adapters::cohere::CohereProvider;
use routeplane_adapters::groq::GroqProvider;
use routeplane_adapters::openai::OpenAIProvider;
use routeplane_adapters::Provider;
use routeplane_router::HealthTracker;
use std::collections::HashMap;
use std::sync::Arc;
use tower::ServiceExt;
use wiremock::matchers::{header_exists, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

// A single authed key whose provider_keys cover every provider these tests route
// to. The auth middleware injects the VirtualKey + TenantContext, exactly like
// the live gateway.
const KEYS: &str = r#"{"keys":[
    {"name":"k_acme","routeplane_key":"rp_acme","provider_keys":{"openai":"sk-openai","groq":"gsk-test","cohere":"sk-cohere"},"tenant_id":"t_acme","tier":"free"}
]}"#;

fn build_state(providers: ProviderRegistry) -> Arc<AppState> {
    Arc::new(AppState {
        health: HealthTracker::new(["openai", "groq", "cohere"]),
        ..AppState::for_tests(providers)
    })
}

fn auth() -> SharedAuthState {
    shared_auth_state(AuthState::load_from_json(KEYS, "test").expect("registry loads"))
}

fn authed_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/audio/transcriptions", post(transcriptions))
        .layer(axum::middleware::from_fn(auth_middleware))
        .layer(axum::Extension(auth()))
        .with_state(state)
}

/// Hand-build a `multipart/form-data` body with a fixed boundary. `fields` are
/// `(name, value)` text parts; `file` (when Some) is `(filename, bytes)`.
fn multipart_body(file: Option<(&str, &[u8])>, fields: &[(&str, &str)]) -> (String, Vec<u8>) {
    let boundary = "----routeplanetestboundary";
    let mut body: Vec<u8> = Vec::new();
    if let Some((filename, bytes)) = file {
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            format!(
                "Content-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\nContent-Type: audio/wav\r\n\r\n"
            )
            .as_bytes(),
        );
        body.extend_from_slice(bytes);
        body.extend_from_slice(b"\r\n");
    }
    for (name, value) in fields {
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
        );
        body.extend_from_slice(value.as_bytes());
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    (format!("multipart/form-data; boundary={boundary}"), body)
}

async fn send(
    router: Router,
    content_type: &str,
    body: Vec<u8>,
    api_key: Option<&str>,
) -> Response {
    let mut builder = Request::builder()
        .method("POST")
        .uri("/v1/audio/transcriptions")
        .header("content-type", content_type);
    if let Some(k) = api_key {
        builder = builder.header("x-routeplane-api-key", k);
    }
    router
        .oneshot(builder.body(Body::from(body)).expect("request builds"))
        .await
        .expect("router responds")
}

async fn body_json(resp: Response) -> serde_json::Value {
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .expect("body readable");
    serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null)
}

fn registry(name: &'static str, provider: Arc<dyn Provider>) -> ProviderRegistry {
    let mut providers: ProviderRegistry = HashMap::new();
    providers.insert(name, provider);
    providers
}

#[tokio::test]
async fn openai_transcription_parses_multipart_and_maps_text() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .and(header_exists("content-type")) // multipart upstream
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "text": "the transcribed words"
        })))
        .mount(&server)
        .await;
    let state = build_state(registry(
        "openai",
        Arc::new(OpenAIProvider::with_base_url(server.uri())),
    ));

    let (ct, body) = multipart_body(
        Some(("speech.wav", b"RIFFfake-wav")),
        &[("model", "whisper-1"), ("language", "en")],
    );
    let resp = send(authed_router(state), &ct, body, Some("rp_acme")).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;
    assert_eq!(v["text"], "the transcribed words");
}

#[tokio::test]
async fn groq_transcription_hits_openai_v1_audio_path() {
    let server = MockServer::start().await;
    // The Groq base-path footgun: the FULL path is /openai/v1/audio/transcriptions.
    Mock::given(method("POST"))
        .and(path("/openai/v1/audio/transcriptions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "text": "groq whisper transcript"
        })))
        .mount(&server)
        .await;
    let state = build_state(registry(
        "groq",
        Arc::new(GroqProvider::with_base_url(server.uri())),
    ));

    let (ct, body) = multipart_body(
        Some(("speech.m4a", b"fake-audio")),
        &[("model", "whisper-large-v3")],
    );
    // Route to groq via the provider header.
    let router = Router::new()
        .route("/v1/audio/transcriptions", post(transcriptions))
        .layer(axum::middleware::from_fn(auth_middleware))
        .layer(axum::Extension(auth()))
        .with_state(state);
    let req = Request::builder()
        .method("POST")
        .uri("/v1/audio/transcriptions")
        .header("content-type", ct)
        .header("x-routeplane-api-key", "rp_acme")
        .header("x-routeplane-provider", "groq")
        .body(Body::from(body))
        .expect("request builds");
    let resp = router.oneshot(req).await.expect("router responds");
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;
    assert_eq!(v["text"], "groq whisper transcript");
}

#[tokio::test]
async fn missing_file_is_clean_400_not_panic() {
    let state = build_state(registry(
        "openai",
        Arc::new(OpenAIProvider::with_base_url("http://127.0.0.1:9")),
    ));
    let (ct, body) = multipart_body(None, &[("model", "whisper-1")]);
    let resp = send(authed_router(state), &ct, body, Some("rp_acme")).await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let v = body_json(resp).await;
    assert_eq!(v["error"]["param"], "file");
}

#[tokio::test]
async fn missing_model_is_clean_400_not_panic() {
    let state = build_state(registry(
        "openai",
        Arc::new(OpenAIProvider::with_base_url("http://127.0.0.1:9")),
    ));
    let (ct, body) = multipart_body(Some(("speech.wav", b"bytes")), &[]);
    let resp = send(authed_router(state), &ct, body, Some("rp_acme")).await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let v = body_json(resp).await;
    assert_eq!(v["error"]["param"], "model");
}

#[tokio::test]
async fn routing_to_non_transcription_provider_is_422() {
    // Cohere has no first-party STT endpoint → trait default returns a typed 422
    // transcription_not_supported (never a panic).
    let state = build_state(registry(
        "cohere",
        Arc::new(CohereProvider::with_base_url("http://127.0.0.1:9")),
    ));
    let (ct, body) = multipart_body(Some(("speech.wav", b"bytes")), &[("model", "whisper-1")]);
    let router = Router::new()
        .route("/v1/audio/transcriptions", post(transcriptions))
        .layer(axum::middleware::from_fn(auth_middleware))
        .layer(axum::Extension(auth()))
        .with_state(state);
    let req = Request::builder()
        .method("POST")
        .uri("/v1/audio/transcriptions")
        .header("content-type", ct)
        .header("x-routeplane-api-key", "rp_acme")
        .header("x-routeplane-provider", "cohere")
        .body(Body::from(body))
        .expect("request builds");
    let resp = router.oneshot(req).await.expect("router responds");
    assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
    let v = body_json(resp).await;
    assert_eq!(v["error"]["code"], "transcription_not_supported");
}

#[tokio::test]
async fn oversized_audio_body_is_413() {
    use tower_http::limit::RequestBodyLimitLayer;
    // The audio route rides a route-specific RequestBodyLimitLayer in main.rs
    // (~26 MiB). Here we mount the route with a TINY cap (64 bytes) and send a
    // larger body to prove an oversized upload is rejected with 413 (Payload Too
    // Large) BEFORE the handler buffers it — never an OOM/panic.
    let state = build_state(registry(
        "openai",
        Arc::new(OpenAIProvider::with_base_url("http://127.0.0.1:9")),
    ));
    let router = Router::new()
        .route("/v1/audio/transcriptions", post(transcriptions))
        .layer(axum::middleware::from_fn(auth_middleware))
        .layer(axum::Extension(auth()))
        .layer(RequestBodyLimitLayer::new(64))
        .with_state(state);
    // A multipart body well over 64 bytes (the fake "audio" alone is larger).
    // A known Content-Length lets RequestBodyLimitLayer reject EARLY with 413
    // (before the handler buffers anything) — the production path for a real
    // oversized upload sent by an SDK with a Content-Length header.
    let big = vec![b'a'; 4096];
    let (ct, body) = multipart_body(Some(("speech.wav", &big)), &[("model", "whisper-1")]);
    let content_length = body.len();
    let req = Request::builder()
        .method("POST")
        .uri("/v1/audio/transcriptions")
        .header("content-type", ct)
        .header("content-length", content_length)
        .header("x-routeplane-api-key", "rp_acme")
        .body(Body::from(body))
        .expect("request builds");
    let resp = router.oneshot(req).await.expect("router responds");
    assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
}

#[tokio::test]
async fn unauthenticated_transcription_is_rejected_401() {
    let state = build_state(registry(
        "openai",
        Arc::new(OpenAIProvider::with_base_url("http://127.0.0.1:9")),
    ));
    let (ct, body) = multipart_body(Some(("speech.wav", b"bytes")), &[("model", "whisper-1")]);
    // No x-routeplane-api-key ⇒ 401 before the handler runs.
    let resp = send(authed_router(state), &ct, body, None).await;
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    let v = body_json(resp).await;
    assert_eq!(v["error"]["code"], "invalid_api_key");
}

// === /v1/audio/translations — the near-twin endpoint (speech → ENGLISH text) ===
//
// Mirrors the transcription suite above: the translations handler REUSES the same
// pipeline slice (multipart parse, residency posture, provider selection, attempt
// loop) via the shared core. These tests prove the route is wired, hits the
// upstream `/v1/audio/translations` path, maps `{text}`, never sends `language`,
// is auth-gated (401), validates inputs (400), and degrades cleanly (422).

fn translations_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/audio/translations", post(translations))
        .layer(axum::middleware::from_fn(auth_middleware))
        .layer(axum::Extension(auth()))
        .with_state(state)
}

async fn send_translations(
    router: Router,
    content_type: &str,
    body: Vec<u8>,
    api_key: Option<&str>,
    provider: Option<&str>,
) -> Response {
    let mut builder = Request::builder()
        .method("POST")
        .uri("/v1/audio/translations")
        .header("content-type", content_type);
    if let Some(k) = api_key {
        builder = builder.header("x-routeplane-api-key", k);
    }
    if let Some(p) = provider {
        builder = builder.header("x-routeplane-provider", p);
    }
    router
        .oneshot(builder.body(Body::from(body)).expect("request builds"))
        .await
        .expect("router responds")
}

#[tokio::test]
async fn openai_translation_parses_multipart_and_maps_text() {
    let server = MockServer::start().await;
    // Assert the upstream hits /v1/audio/TRANSLATIONS (not transcriptions).
    Mock::given(method("POST"))
        .and(path("/v1/audio/translations"))
        .and(header_exists("content-type")) // multipart upstream
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "text": "translated english words"
        })))
        .mount(&server)
        .await;
    let state = build_state(registry(
        "openai",
        Arc::new(OpenAIProvider::with_base_url(server.uri())),
    ));

    // A `language` field is supplied inbound but must NOT reach the upstream
    // (translations output is always English) — verified below from the recording.
    let (ct, body) = multipart_body(
        Some(("speech.wav", b"RIFFfake-wav")),
        &[("model", "whisper-1"), ("language", "fr")],
    );
    let resp =
        send_translations(translations_router(state), &ct, body, Some("rp_acme"), None).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;
    assert_eq!(v["text"], "translated english words");

    let received = &server.received_requests().await.unwrap()[0];
    let upstream_body = String::from_utf8_lossy(&received.body);
    assert!(upstream_body.contains("whisper-1"));
    assert!(
        !upstream_body.contains("name=\"language\""),
        "translations must NOT forward a `language` field upstream"
    );
}

#[tokio::test]
async fn groq_translation_hits_openai_v1_audio_translations_path() {
    let server = MockServer::start().await;
    // The Groq base-path footgun: FULL path is /openai/v1/audio/translations.
    Mock::given(method("POST"))
        .and(path("/openai/v1/audio/translations"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "text": "groq english translation"
        })))
        .mount(&server)
        .await;
    let state = build_state(registry(
        "groq",
        Arc::new(GroqProvider::with_base_url(server.uri())),
    ));

    let (ct, body) = multipart_body(
        Some(("speech.m4a", b"fake-audio")),
        &[("model", "whisper-large-v3")],
    );
    let resp = send_translations(
        translations_router(state),
        &ct,
        body,
        Some("rp_acme"),
        Some("groq"),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;
    assert_eq!(v["text"], "groq english translation");
}

#[tokio::test]
async fn translation_missing_file_is_clean_400_not_panic() {
    let state = build_state(registry(
        "openai",
        Arc::new(OpenAIProvider::with_base_url("http://127.0.0.1:9")),
    ));
    let (ct, body) = multipart_body(None, &[("model", "whisper-1")]);
    let resp =
        send_translations(translations_router(state), &ct, body, Some("rp_acme"), None).await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let v = body_json(resp).await;
    assert_eq!(v["error"]["param"], "file");
}

#[tokio::test]
async fn translation_missing_model_is_clean_400_not_panic() {
    let state = build_state(registry(
        "openai",
        Arc::new(OpenAIProvider::with_base_url("http://127.0.0.1:9")),
    ));
    let (ct, body) = multipart_body(Some(("speech.wav", b"bytes")), &[]);
    let resp =
        send_translations(translations_router(state), &ct, body, Some("rp_acme"), None).await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let v = body_json(resp).await;
    assert_eq!(v["error"]["param"], "model");
}

#[tokio::test]
async fn translation_to_non_translation_provider_is_422() {
    // Cohere has no first-party translation endpoint → trait default returns a
    // typed 422 translation_not_supported (never a panic).
    let state = build_state(registry(
        "cohere",
        Arc::new(CohereProvider::with_base_url("http://127.0.0.1:9")),
    ));
    let (ct, body) = multipart_body(Some(("speech.wav", b"bytes")), &[("model", "whisper-1")]);
    let resp = send_translations(
        translations_router(state),
        &ct,
        body,
        Some("rp_acme"),
        Some("cohere"),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
    let v = body_json(resp).await;
    assert_eq!(v["error"]["code"], "translation_not_supported");
}

#[tokio::test]
async fn unauthenticated_translation_is_rejected_401() {
    let state = build_state(registry(
        "openai",
        Arc::new(OpenAIProvider::with_base_url("http://127.0.0.1:9")),
    ));
    let (ct, body) = multipart_body(Some(("speech.wav", b"bytes")), &[("model", "whisper-1")]);
    // No x-routeplane-api-key ⇒ 401 before the handler runs.
    let resp = send_translations(translations_router(state), &ct, body, None, None).await;
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    let v = body_json(resp).await;
    assert_eq!(v["error"]["code"], "invalid_api_key");
}
