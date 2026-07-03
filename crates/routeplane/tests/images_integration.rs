//! Hermetic integration tests for POST /v1/images/generations (parity with
//! OpenAI's /v1/images/generations; OpenAI-backed). The handler is invoked
//! directly (no HTTP server, no auth round-trip — the extensions the auth
//! middleware would inject are passed explicitly, exactly like the rerank /
//! embeddings integration suites). The only "network" is a localhost wiremock
//! standing in for OpenAI. Covers:
//!   * OpenAI /v1/images/generations request shape (prompt + defaulted model)
//!     and response mapping for BOTH the b64_json and url variants.
//!   * mask-before-egress: PII in the prompt is masked to `[EMAIL_MASKED]`
//!     BEFORE egress (the mock matches only on the masked text) — image-gen does
//!     not become a PII-egress bypass (same posture as chat/rerank).
//!   * routing image-gen to a provider without a first-party image endpoint
//!     (cohere) → 422 image_generation_not_supported envelope.
//!   * empty prompt → clean 422, not a panic.
//!   * the route is auth-gated (no key ⇒ 401) via the real auth_middleware.

use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, Request, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Extension, Router};
use routeplane::auth::{
    auth_middleware, shared_auth_state, AuthState, SharedAuthState, TenantContext, VirtualKey,
};
use routeplane::images_api::image_generation;
use routeplane::proxy::{AppState, ProviderRegistry};
use routeplane_adapters::cohere::CohereProvider;
use routeplane_adapters::openai::OpenAIProvider;
use routeplane_adapters::Provider;
use routeplane_entitlements::{CapabilitySet, Tier};
use routeplane_router::HealthTracker;
use routeplane_types::ImageGenerationRequest;
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

fn payload(prompt: &str) -> ImageGenerationRequest {
    ImageGenerationRequest {
        model: None,
        prompt: prompt.into(),
        n: None,
        size: None,
        quality: None,
        response_format: None,
        extra: Default::default(),
    }
}

async fn invoke(
    state: Arc<AppState>,
    headers: HeaderMap,
    payload: ImageGenerationRequest,
) -> Response {
    image_generation(
        State(state),
        Extension(vk()),
        Extension(ctx()),
        headers,
        routeplane::api_error::OpenAiJson(payload),
    )
    .await
    .into_response()
}

async fn body_json(resp: Response) -> serde_json::Value {
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .expect("body readable");
    serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null)
}

#[tokio::test]
async fn openai_image_returns_b64_data_and_defaults_model() {
    let server = MockServer::start().await;
    // Caller omitted model ⇒ adapter defaults to gpt-image-1; gpt-image-1 returns
    // b64_json + an optional top-level usage block.
    Mock::given(method("POST"))
        .and(path("/v1/images/generations"))
        .and(header("authorization", "Bearer sk-openai"))
        .and(body_partial_json(json!({
            "prompt": "a red panda in a forest",
            "model": "gpt-image-1"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "created": 1_700_000_000i64,
            "data": [ { "b64_json": "aGVsbG8=" } ],
            "usage": { "total_tokens": 42 }
        })))
        .mount(&server)
        .await;
    let state = build_state(openai_registry(&server.uri()));

    let resp = invoke(state, HeaderMap::new(), payload("a red panda in a forest")).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;
    assert_eq!(v["created"], 1_700_000_000i64);
    assert_eq!(v["data"][0]["b64_json"], "aGVsbG8=");
    assert!(v["data"][0].get("url").is_none());
    assert_eq!(v["usage"]["total_tokens"], 42);
}

#[tokio::test]
async fn openai_image_returns_url_variant() {
    let server = MockServer::start().await;
    // dall-e-3 can return a hosted url + revised_prompt.
    Mock::given(method("POST"))
        .and(path("/v1/images/generations"))
        .and(body_partial_json(json!({
            "model": "dall-e-3",
            "prompt": "a city skyline"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "created": 1_700_000_111i64,
            "data": [ {
                "url": "https://img.example/abc.png",
                "revised_prompt": "a vivid city skyline at dusk"
            } ]
        })))
        .mount(&server)
        .await;
    let state = build_state(openai_registry(&server.uri()));

    let mut req = payload("a city skyline");
    req.model = Some("dall-e-3".into());
    let resp = invoke(state, HeaderMap::new(), req).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;
    assert_eq!(v["data"][0]["url"], "https://img.example/abc.png");
    assert!(v["data"][0].get("b64_json").is_none());
    assert_eq!(
        v["data"][0]["revised_prompt"],
        "a vivid city skyline at dusk"
    );
}

#[tokio::test]
async fn email_in_prompt_is_masked_before_egress() {
    let server = MockServer::start().await;
    // The mock ONLY matches when the prompt was masked to "[EMAIL_MASKED]"; had
    // masking not run, the body would not match and the call would not be 200.
    // Asserting 200 proves image-gen does not become a PII-egress bypass.
    Mock::given(method("POST"))
        .and(path("/v1/images/generations"))
        .and(body_partial_json(json!({ "prompt": "[EMAIL_MASKED]" })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "created": 1_700_000_222i64,
            "data": [ { "b64_json": "eA==" } ]
        })))
        .expect(1)
        .mount(&server)
        .await;
    let state = build_state(openai_registry(&server.uri()));

    // A bare email so the whole prompt masks to "[EMAIL_MASKED]".
    let resp = invoke(state, HeaderMap::new(), payload("user@example.com")).await;
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "masked prompt must match the mock"
    );
}

#[tokio::test]
async fn routing_image_to_non_image_provider_is_422_not_supported() {
    // Cohere has no first-party image endpoint → the trait default returns a
    // typed 422 image_generation_not_supported (never a panic, never silent).
    let mut providers: ProviderRegistry = HashMap::new();
    providers.insert(
        "cohere",
        Arc::new(CohereProvider::with_base_url("http://127.0.0.1:9")) as Arc<dyn Provider>,
    );
    let state = build_state(providers);
    let mut headers = HeaderMap::new();
    headers.insert("x-routeplane-provider", HeaderValue::from_static("cohere"));

    let resp = invoke(state, headers, payload("a cat")).await;
    assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
    let v = body_json(resp).await;
    assert_eq!(v["error"]["code"], "image_generation_not_supported");
    assert_eq!(v["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn empty_prompt_is_clean_422_not_panic() {
    // Empty prompt is rejected as a typed 422 before any network call — never a
    // panic.
    let state = build_state(openai_registry("http://127.0.0.1:9"));
    let resp = invoke(state, HeaderMap::new(), payload("   ")).await;
    assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

// --- auth gate --------------------------------------------------------------

const KEYS: &str = r#"{"keys":[
    {"name":"k_acme","routeplane_key":"rp_acme","provider_keys":{"openai":"x"},"tenant_id":"t_acme","tier":"free"}
]}"#;

fn auth() -> SharedAuthState {
    shared_auth_state(AuthState::load_from_json(KEYS, "test").expect("registry loads"))
}

/// A minimal router mirroring `main.rs`'s authed wiring: `/v1/images/generations`
/// behind the real `auth_middleware` + `SharedAuthState`. Used to assert the auth
/// gate rejects an unauthenticated caller with the standard 401 envelope BEFORE
/// the handler runs.
fn authed_router() -> Router {
    let state = build_state(openai_registry("http://127.0.0.1:9"));
    Router::new()
        .route("/v1/images/generations", post(image_generation))
        .layer(axum::middleware::from_fn(auth_middleware))
        .layer(axum::Extension(auth()))
        .with_state(state)
}

#[tokio::test]
async fn unauthenticated_image_generation_is_rejected_401() {
    let resp = authed_router()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/images/generations")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"prompt":"a cat"}"#))
                .expect("request builds"),
        )
        .await
        .expect("router responds");
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    let v = body_json(resp).await;
    assert_eq!(v["error"]["code"], "invalid_api_key");
}
