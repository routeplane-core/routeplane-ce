//! Hermetic integration tests for POST /v1/embeddings (G2.7, PRD-011 §5).
//! The handler is invoked directly (no HTTP server, no auth round-trip — the
//! extensions the auth middleware would inject are passed explicitly, exactly
//! like the limits/guardrails integration suites). The only "network" is a
//! localhost wiremock standing in for OpenAI. Covers: OpenAI list shape + no
//! cache header (FR-1/FR-6), mask-before-embed (FR-5), residency 422 with a PII
//! input + IN and no resident provider (FR-5), and anthropic → 422
//! embeddings_not_supported envelope (FR-3).

use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Extension;
use routeplane::auth::{TenantContext, VirtualKey};
use routeplane::embeddings::embeddings;
use routeplane::proxy::{AppState, ProviderRegistry};
use routeplane_adapters::anthropic::AnthropicProvider;
use routeplane_adapters::openai::OpenAIProvider;
use routeplane_adapters::Provider;
use routeplane_entitlements::{CapabilitySet, Tier};
use routeplane_router::HealthTracker;
use routeplane_types::{EmbeddingInput, EmbeddingRequest};
use serde_json::json;
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn build_state(providers: ProviderRegistry) -> Arc<AppState> {
    Arc::new(AppState {
        health: HealthTracker::new(["openai", "anthropic"]),
        ..AppState::for_tests(providers)
    })
}

fn vk() -> VirtualKey {
    serde_json::from_value(json!({
        "name": "test-key",
        "routeplane_key": "rp_test",
        "provider_keys": { "openai": "test-api-key", "anthropic": "test-api-key" }
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

fn payload(input: EmbeddingInput) -> EmbeddingRequest {
    EmbeddingRequest {
        model: "text-embedding-3-small".into(),
        input,
        encoding_format: None,
        dimensions: None,
        user: None,
    }
}

async fn invoke(state: Arc<AppState>, headers: HeaderMap, payload: EmbeddingRequest) -> Response {
    embeddings(
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

async fn mount_embeddings_ok(server: &MockServer) {
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 4, "total_tokens": 4}
        })))
        .mount(server)
        .await;
}

#[tokio::test]
async fn empty_input_is_422_not_500() {
    // The empty-input guard fires at the route edge, before any provider call.
    let server = MockServer::start().await;
    let state = build_state(openai_registry(&server.uri()));
    let resp = invoke(
        state,
        HeaderMap::new(),
        payload(EmbeddingInput::Batch(vec![])),
    )
    .await;
    assert_eq!(
        resp.status(),
        StatusCode::UNPROCESSABLE_ENTITY,
        "empty input must be a 422 client error, not a 500"
    );
    let v = body_json(resp).await;
    assert_eq!(v["error"]["param"], "input");
}

#[tokio::test]
async fn openai_embeddings_returns_openai_shaped_list_with_no_cache_header() {
    let server = MockServer::start().await;
    mount_embeddings_ok(&server).await;
    let state = build_state(openai_registry(&server.uri()));

    let resp = invoke(
        state,
        HeaderMap::new(),
        payload(EmbeddingInput::Single("hello".into())),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    // FR-6: embeddings carry no cache header.
    assert!(resp.headers().get("x-routeplane-cache").is_none());

    let v = body_json(resp).await;
    assert_eq!(v["object"], "list");
    assert_eq!(v["data"].as_array().unwrap().len(), 1);
    assert_eq!(v["data"][0]["index"], 0);
    assert_eq!(v["usage"]["total_tokens"], 4);
}

#[tokio::test]
async fn embeddings_response_carries_provenance_headers() {
    let server = MockServer::start().await;
    mount_embeddings_ok(&server).await;
    let state = build_state(openai_registry(&server.uri()));

    let resp = invoke(
        state,
        HeaderMap::new(),
        payload(EmbeddingInput::Single("hello".into())),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let rid = resp
        .headers()
        .get("x-routeplane-request-id")
        .expect("x-routeplane-request-id header present on embeddings success");
    let rid_str = rid.to_str().expect("request-id is valid utf-8");
    assert!(
        rid_str.starts_with("req_"),
        "request-id must start with req_, got: {rid_str}"
    );
    // Provenance trio: trace-id carries the SAME value, and the serving
    // provider is echoed.
    assert_eq!(
        resp.headers()
            .get("x-routeplane-trace-id")
            .and_then(|v| v.to_str().ok()),
        Some(rid_str)
    );
    assert_eq!(
        resp.headers()
            .get("x-routeplane-provider")
            .and_then(|v| v.to_str().ok()),
        Some("openai")
    );
}

#[tokio::test]
async fn email_input_is_masked_before_embedding() {
    let server = MockServer::start().await;
    // The mock ONLY matches when the input was masked to exactly
    // "[EMAIL_MASKED]"; had masking not run, the body would not match and the
    // call would not be 200. Asserting 200 proves mask-then-embed (FR-5).
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .and(body_partial_json(json!({ "input": "[EMAIL_MASKED]" })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.0]}],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 1, "total_tokens": 1}
        })))
        .expect(1)
        .mount(&server)
        .await;
    let state = build_state(openai_registry(&server.uri()));

    // A bare email so that after masking the whole input is "[EMAIL_MASKED]".
    let resp = invoke(
        state,
        HeaderMap::new(),
        payload(EmbeddingInput::Single("user@example.com".into())),
    )
    .await;
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "masked input must match the mock"
    );
}

#[tokio::test]
async fn pii_input_with_required_region_and_no_resident_provider_is_422() {
    // openai is NOT resident in IN, so a PII input + IN residency has no eligible
    // provider → sovereign block 422 (no provider is ever dialed).
    let state = build_state(openai_registry("http://127.0.0.1:9"));
    let mut headers = HeaderMap::new();
    headers.insert("x-routeplane-residency", HeaderValue::from_static("IN"));

    let resp = invoke(
        state,
        headers,
        payload(EmbeddingInput::Single(
            "my email is user@example.com".into(),
        )),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn routing_embeddings_to_anthropic_is_422_not_supported() {
    let mut providers: ProviderRegistry = HashMap::new();
    providers.insert(
        "anthropic",
        Arc::new(AnthropicProvider::new()) as Arc<dyn Provider>,
    );
    let state = build_state(providers);
    let mut headers = HeaderMap::new();
    headers.insert(
        "x-routeplane-provider",
        HeaderValue::from_static("anthropic"),
    );

    let resp = invoke(
        state,
        headers,
        payload(EmbeddingInput::Single("hello".into())),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
    let v = body_json(resp).await;
    assert_eq!(v["error"]["code"], "embeddings_not_supported");
    assert_eq!(v["error"]["type"], "invalid_request_error");
}
