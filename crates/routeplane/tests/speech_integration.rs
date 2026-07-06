//! Hermetic integration tests for POST /v1/audio/speech (text-to-speech; parity
//! with OpenAI's /v1/audio/speech; OpenAI-backed). The handler is invoked
//! directly (no HTTP server, no auth round-trip — the extensions the auth
//! middleware would inject are passed explicitly, exactly like the images /
//! rerank / embeddings integration suites). The only "network" is a localhost
//! wiremock standing in for OpenAI. Covers:
//!   * OpenAI /v1/audio/speech request shape (input + voice + defaulted model)
//!     and the BINARY response: raw audio bytes returned with the right
//!     Content-Type (from the upstream header).
//!   * mask-before-egress: PII in the `input` is masked to `[EMAIL_MASKED]`
//!     BEFORE egress (the mock matches only on the masked text) — TTS does not
//!     become a PII-egress bypass (same posture as chat/images/rerank).
//!   * routing TTS to a provider without a first-party speech endpoint (cohere)
//!     → 422 speech_not_supported envelope.
//!   * empty input → clean 422, not a panic.
//!   * the route is auth-gated (no key ⇒ 401) via the real auth_middleware.

use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, Request, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Extension, Router};
use routeplane::audio_api::speech;
use routeplane::auth::{
    auth_middleware, shared_auth_state, AuthState, SharedAuthState, TenantContext, VirtualKey,
};
use routeplane::proxy::{AppState, ProviderRegistry};
use routeplane_adapters::cohere::CohereProvider;
use routeplane_adapters::openai::OpenAIProvider;
use routeplane_adapters::Provider;
use routeplane_entitlements::{CapabilitySet, Tier};
use routeplane_router::HealthTracker;
use routeplane_types::SpeechRequest;
use serde_json::json;
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;
use tower::ServiceExt;
use wiremock::matchers::{body_partial_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn build_state(providers: ProviderRegistry) -> Arc<AppState> {
    Arc::new(AppState {
        health: HealthTracker::new(["openai", "cohere"]),
        ..AppState::for_tests(providers)
    })
}

fn vk() -> VirtualKey {
    serde_json::from_value(json!({
        "name": "test-key",
        "routeplane_key": "rp_test",
        "provider_keys": { "openai": "sk-openai", "cohere": "sk-cohere" }
    }))
    .expect("virtual key deserializes")
}

fn ctx() -> TenantContext {
    TenantContext {
        tenant_id: "t_test".into(),
        tier: Tier::Free,
        capabilities: CapabilitySet::resolve(Tier::Free, &BTreeSet::new(), &BTreeSet::new()),
        compliance_frameworks: Vec::new(),
        compliance_mode: routeplane::auth::ComplianceMode::Strict,
    }
}

fn openai_registry(base_url: &str) -> ProviderRegistry {
    let mut providers: ProviderRegistry = HashMap::new();
    providers.insert(
        "openai",
        Arc::new(OpenAIProvider::with_base_url(base_url)) as Arc<dyn Provider>,
    );
    providers
}

fn payload(input: &str) -> SpeechRequest {
    SpeechRequest {
        model: None,
        input: input.into(),
        voice: "alloy".into(),
        response_format: None,
        speed: None,
        extra: Default::default(),
    }
}

async fn invoke(state: Arc<AppState>, headers: HeaderMap, payload: SpeechRequest) -> Response {
    speech(
        State(state),
        Extension(vk()),
        Extension(ctx()),
        headers,
        routeplane::api_error::OpenAiJson(payload),
    )
    .await
    .into_response()
}

async fn body_bytes(resp: Response) -> Vec<u8> {
    axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .expect("body readable")
        .to_vec()
}

async fn body_json(resp: Response) -> serde_json::Value {
    let bytes = body_bytes(resp).await;
    serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null)
}

#[tokio::test]
async fn openai_speech_returns_binary_audio_with_content_type_and_defaults_model() {
    let server = MockServer::start().await;
    let audio: Vec<u8> = vec![0xFF, 0xFB, 0x90, 0x00, 0xDE, 0xAD, 0xBE, 0xEF];
    // Caller omitted model ⇒ adapter defaults to gpt-4o-mini-tts. The upstream
    // returns RAW audio bytes (not JSON) with an audio/* Content-Type.
    Mock::given(method("POST"))
        .and(path("/v1/audio/speech"))
        .and(header("authorization", "Bearer sk-openai"))
        .and(body_partial_json(json!({
            "input": "Hello from Routeplane",
            "voice": "alloy",
            "model": "gpt-4o-mini-tts"
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "audio/mpeg")
                .set_body_bytes(audio.clone()),
        )
        .mount(&server)
        .await;
    let state = build_state(openai_registry(&server.uri()));

    let resp = invoke(state, HeaderMap::new(), payload("Hello from Routeplane")).await;
    assert_eq!(resp.status(), StatusCode::OK);
    // The response is BINARY audio with the per-format Content-Type, NOT JSON.
    assert_eq!(
        resp.headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok()),
        Some("audio/mpeg")
    );
    // Branding-load-bearing x-routeplane-provider header is echoed.
    assert_eq!(
        resp.headers()
            .get("x-routeplane-provider")
            .and_then(|v| v.to_str().ok()),
        Some("openai")
    );
    // Provenance trio: the correlation ids ride along, and trace-id/request-id
    // carry the SAME req_<uuid> value.
    let trace = resp
        .headers()
        .get("x-routeplane-trace-id")
        .and_then(|v| v.to_str().ok())
        .expect("x-routeplane-trace-id present on speech success");
    assert!(trace.starts_with("req_"), "trace id is req_<uuid>: {trace}");
    assert_eq!(
        resp.headers()
            .get("x-routeplane-request-id")
            .and_then(|v| v.to_str().ok()),
        Some(trace)
    );
    let bytes = body_bytes(resp).await;
    assert_eq!(bytes, audio);
}

#[tokio::test]
async fn speech_derives_content_type_from_requested_format() {
    let server = MockServer::start().await;
    // No content-type on the response → the handler returns the adapter-derived
    // type from response_format (wav ⇒ audio/wav).
    Mock::given(method("POST"))
        .and(path("/v1/audio/speech"))
        .and(body_partial_json(json!({
            "model": "tts-1",
            "voice": "nova",
            "response_format": "wav"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(vec![1u8, 2, 3]))
        .mount(&server)
        .await;
    let state = build_state(openai_registry(&server.uri()));

    let mut req = payload("read this aloud");
    req.model = Some("tts-1".into());
    req.voice = "nova".into();
    req.response_format = Some("wav".into());
    let resp = invoke(state, HeaderMap::new(), req).await;
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(
        resp.headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok()),
        Some("audio/wav")
    );
}

#[tokio::test]
async fn email_in_input_is_masked_before_egress() {
    let server = MockServer::start().await;
    // The mock ONLY matches when the input was masked to "[EMAIL_MASKED]"; had
    // masking not run, the body would not match and the call would not be 200.
    // Asserting 200 proves TTS does not become a PII-egress bypass.
    Mock::given(method("POST"))
        .and(path("/v1/audio/speech"))
        .and(body_partial_json(json!({ "input": "[EMAIL_MASKED]" })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "audio/mpeg")
                .set_body_bytes(vec![9u8, 9, 9]),
        )
        .expect(1)
        .mount(&server)
        .await;
    let state = build_state(openai_registry(&server.uri()));

    // A bare email so the whole input masks to "[EMAIL_MASKED]".
    let resp = invoke(state, HeaderMap::new(), payload("user@example.com")).await;
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "masked input must match the mock"
    );
}

#[tokio::test]
async fn routing_speech_to_non_speech_provider_is_422_not_supported() {
    // Cohere has no first-party TTS endpoint → the trait default returns a typed
    // 422 speech_not_supported (never a panic, never silent).
    let mut providers: ProviderRegistry = HashMap::new();
    providers.insert(
        "cohere",
        Arc::new(CohereProvider::with_base_url("http://127.0.0.1:9")) as Arc<dyn Provider>,
    );
    let state = build_state(providers);
    let mut headers = HeaderMap::new();
    headers.insert("x-routeplane-provider", HeaderValue::from_static("cohere"));

    let resp = invoke(state, headers, payload("hello")).await;
    assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
    let v = body_json(resp).await;
    assert_eq!(v["error"]["code"], "speech_not_supported");
    assert_eq!(v["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn empty_input_is_clean_422_not_panic() {
    // Empty input is rejected as a typed 422 before any network call — never a
    // panic.
    let state = build_state(openai_registry("http://127.0.0.1:9"));
    let resp = invoke(state, HeaderMap::new(), payload("   ")).await;
    assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
    let v = body_json(resp).await;
    assert_eq!(v["error"]["param"], "input");
}

// --- auth gate --------------------------------------------------------------

const KEYS: &str = r#"{"keys":[
    {"name":"k_acme","routeplane_key":"rp_acme","provider_keys":{"openai":"x"},"tenant_id":"t_acme","tier":"free"}
]}"#;

fn auth() -> SharedAuthState {
    shared_auth_state(AuthState::load_from_json(KEYS, "test").expect("registry loads"))
}

/// A minimal router mirroring `main.rs`'s authed wiring: `/v1/audio/speech`
/// behind the real `auth_middleware` + `SharedAuthState`. Used to assert the auth
/// gate rejects an unauthenticated caller with the standard 401 envelope BEFORE
/// the handler runs.
fn authed_router() -> Router {
    let state = build_state(openai_registry("http://127.0.0.1:9"));
    Router::new()
        .route("/v1/audio/speech", post(speech))
        .layer(axum::middleware::from_fn(auth_middleware))
        .layer(axum::Extension(auth()))
        .with_state(state)
}

#[tokio::test]
async fn unauthenticated_speech_is_rejected_401() {
    let resp = authed_router()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/audio/speech")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"input":"hi","voice":"alloy"}"#))
                .expect("request builds"),
        )
        .await
        .expect("router responds");
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    let v = body_json(resp).await;
    assert_eq!(v["error"]["code"], "invalid_api_key");
}
