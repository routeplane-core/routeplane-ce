//! Hermetic integration tests for POST /v1/rerank (parity with LiteLLM's
//! /rerank; Cohere-backed). The handler is invoked directly (no HTTP server, no
//! auth round-trip — the extensions the auth middleware would inject are passed
//! explicitly, exactly like the embeddings integration suite). The only
//! "network" is a localhost wiremock standing in for Cohere. Covers:
//!   * Cohere /v2/rerank request shape (model+query+documents+top_n) + result
//!     mapping (index/score order) + search_units → usage.
//!   * mask-before-rerank: PII in a document is masked to `[EMAIL_MASKED]`
//!     BEFORE egress (the mock matches only on the masked text).
//!   * routing rerank to a non-rerank provider (openai) → 422
//!     rerank_not_supported envelope.
//!   * empty documents → clean 422, not a panic.
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
use routeplane::proxy::{AppState, ProviderRegistry};
use routeplane::rerank_api::rerank;
use routeplane_adapters::cohere::CohereProvider;
use routeplane_adapters::openai::OpenAIProvider;
use routeplane_adapters::Provider;
use routeplane_entitlements::{CapabilitySet, Tier};
use routeplane_router::HealthTracker;
use routeplane_types::RerankRequest;
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
        "provider_keys": { "cohere": "sk-cohere", "openai": "test-api-key" }
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

fn cohere_registry(base_url: &str) -> ProviderRegistry {
    let mut providers: ProviderRegistry = HashMap::new();
    providers.insert(
        "cohere",
        Arc::new(CohereProvider::with_base_url(base_url)) as Arc<dyn Provider>,
    );
    providers
}

fn payload(query: &str, documents: Vec<&str>, top_n: Option<u32>) -> RerankRequest {
    RerankRequest {
        model: "rerank-v3.5".into(),
        query: query.into(),
        documents: documents.into_iter().map(|s| s.to_string()).collect(),
        top_n,
        return_documents: false,
    }
}

async fn invoke(state: Arc<AppState>, headers: HeaderMap, payload: RerankRequest) -> Response {
    rerank(
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
async fn cohere_rerank_returns_ranked_results_and_usage() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v2/rerank"))
        .and(header("authorization", "Bearer sk-cohere"))
        .and(body_partial_json(json!({
            "model": "rerank-v3.5",
            "query": "capital of france",
            "documents": ["berlin", "paris", "tokyo"],
            "top_n": 2
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "rerank-1",
            "results": [
                {"index": 1, "relevance_score": 0.99},
                {"index": 0, "relevance_score": 0.05}
            ],
            "meta": {"billed_units": {"search_units": 1}}
        })))
        .mount(&server)
        .await;
    let state = build_state(cohere_registry(&server.uri()));

    let resp = invoke(
        state,
        HeaderMap::new(),
        payload(
            "capital of france",
            vec!["berlin", "paris", "tokyo"],
            Some(2),
        ),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    // Provenance trio: provider + trace/request correlation ids, trace-id and
    // request-id carrying the SAME req_<uuid> value.
    assert_eq!(
        resp.headers()
            .get("x-routeplane-provider")
            .and_then(|v| v.to_str().ok()),
        Some("cohere")
    );
    let trace = resp
        .headers()
        .get("x-routeplane-trace-id")
        .and_then(|v| v.to_str().ok())
        .expect("x-routeplane-trace-id present on rerank success")
        .to_string();
    assert!(trace.starts_with("req_"), "trace id is req_<uuid>: {trace}");
    assert_eq!(
        resp.headers()
            .get("x-routeplane-request-id")
            .and_then(|v| v.to_str().ok()),
        Some(trace.as_str())
    );
    let v = body_json(resp).await;
    assert_eq!(v["id"], "rerank-1");
    assert_eq!(v["model"], "rerank-v3.5");
    // Order preserved (relevance desc), index maps back to input.
    assert_eq!(v["results"][0]["index"], 1);
    assert_eq!(v["results"][0]["relevance_score"], 0.99);
    assert_eq!(v["results"][1]["index"], 0);
    assert_eq!(v["usage"]["search_units"], 1);
}

#[tokio::test]
async fn email_in_document_is_masked_before_egress() {
    let server = MockServer::start().await;
    // The mock ONLY matches when the document was masked to "[EMAIL_MASKED]";
    // had masking not run, the body would not match and the call would not be
    // 200. Asserting 200 proves rerank does not become a PII-egress bypass.
    Mock::given(method("POST"))
        .and(path("/v2/rerank"))
        .and(body_partial_json(json!({
            "documents": ["[EMAIL_MASKED]"]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "rerank-2",
            "results": [{"index": 0, "relevance_score": 0.5}],
            "meta": {"billed_units": {"search_units": 1}}
        })))
        .expect(1)
        .mount(&server)
        .await;
    let state = build_state(cohere_registry(&server.uri()));

    let resp = invoke(
        state,
        HeaderMap::new(),
        // A bare email so the whole document masks to "[EMAIL_MASKED]".
        payload("find the contact", vec!["user@example.com"], None),
    )
    .await;
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "masked document must match the mock"
    );
}

#[tokio::test]
async fn email_in_query_is_masked_before_egress() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v2/rerank"))
        .and(body_partial_json(json!({ "query": "[EMAIL_MASKED]" })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "rerank-3",
            "results": [{"index": 0, "relevance_score": 0.5}],
            "meta": {"billed_units": {"search_units": 1}}
        })))
        .expect(1)
        .mount(&server)
        .await;
    let state = build_state(cohere_registry(&server.uri()));

    let resp = invoke(
        state,
        HeaderMap::new(),
        payload("user@example.com", vec!["doc one"], None),
    )
    .await;
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "masked query must match the mock"
    );
}

#[tokio::test]
async fn routing_rerank_to_openai_is_422_not_supported() {
    // OpenAI has no first-party rerank endpoint → the trait default returns a
    // typed 422 rerank_not_supported (never a panic, never a silent drop).
    let mut providers: ProviderRegistry = HashMap::new();
    providers.insert(
        "openai",
        Arc::new(OpenAIProvider::with_base_url("http://127.0.0.1:9")) as Arc<dyn Provider>,
    );
    let state = build_state(providers);
    let mut headers = HeaderMap::new();
    headers.insert("x-routeplane-provider", HeaderValue::from_static("openai"));

    let resp = invoke(state, headers, payload("q", vec!["a", "b"], None)).await;
    assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
    let v = body_json(resp).await;
    assert_eq!(v["error"]["code"], "rerank_not_supported");
    assert_eq!(v["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn empty_documents_is_clean_422_not_panic() {
    // Empty document set is rejected as a typed 422 before any network call —
    // never a panic. cohere is the default provider; the adapter guards it.
    let state = build_state(cohere_registry("http://127.0.0.1:9"));
    let resp = invoke(state, HeaderMap::new(), payload("q", vec![], None)).await;
    assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

// --- auth gate --------------------------------------------------------------

const KEYS: &str = r#"{"keys":[
    {"name":"k_acme","routeplane_key":"rp_acme","provider_keys":{"cohere":"x"},"tenant_id":"t_acme","tier":"free"}
]}"#;

fn auth() -> SharedAuthState {
    shared_auth_state(AuthState::load_from_json(KEYS, "test").expect("registry loads"))
}

/// A minimal router mirroring `main.rs`'s authed wiring: `/v1/rerank` behind the
/// real `auth_middleware` + `SharedAuthState`. Used to assert the auth gate
/// rejects an unauthenticated caller with the standard 401 envelope BEFORE the
/// handler runs.
fn authed_router() -> Router {
    let state = build_state(cohere_registry("http://127.0.0.1:9"));
    Router::new()
        .route("/v1/rerank", post(rerank))
        .layer(axum::middleware::from_fn(auth_middleware))
        .layer(axum::Extension(auth()))
        .with_state(state)
}

#[tokio::test]
async fn unauthenticated_rerank_is_rejected_401() {
    let resp = authed_router()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/rerank")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"rerank-v3.5","query":"q","documents":["a"]}"#,
                ))
                .expect("request builds"),
        )
        .await
        .expect("router responds");
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    let v = body_json(resp).await;
    assert_eq!(v["error"]["code"], "invalid_api_key");
}
