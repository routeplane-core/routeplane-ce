//! Hermetic integration tests for POST /v1/messages — the NATIVE Anthropic
//! Messages API surface (parity with Portkey/LiteLLM). The handler funnels the
//! translated request through the SAME `chat_completions_core` pipeline as
//! `/v1/chat/completions`, so these tests exercise the full pipeline end-to-end
//! (residency classify-then-mask, guardrails, limits, routing, provider call)
//! against a localhost wiremock standing in for Anthropic's `/v1/messages`.
//!
//! Covers:
//!   * an Anthropic-native request (system + user text) → routed through the
//!     pipeline → an Anthropic-shaped response (msg id, content blocks,
//!     stop_reason, input/output tokens).
//!   * a base64 image block round-trips to the native Anthropic image block on
//!     the wire (inbound ContentPart::ImageUrl reconstruction).
//!   * `max_tokens` missing → a clean Anthropic-shaped 400 (not a serde envelope).
//!   * `stream:true` → a documented Anthropic-shaped 400.
//!   * PII in an Anthropic message is masked to `[EMAIL_MASKED]` BEFORE egress
//!     (the mock matches only the masked text — proves /v1/messages is not a
//!     PII-egress bypass and that classify-then-mask runs on the translated body).
//!   * the route is auth-gated (no key ⇒ 401) via the real auth_middleware.

use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, Request, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Extension, Router};
use routeplane::auth::{
    auth_middleware, shared_auth_state, AuthState, SharedAuthState, TenantContext,
    TenantGuardrails, VirtualKey,
};
use routeplane::messages_api::messages;
use routeplane::proxy::{AppState, ProviderRegistry};
use routeplane_adapters::anthropic::AnthropicProvider;
use routeplane_adapters::Provider;
use routeplane_entitlements::{CapabilitySet, Tier};
use routeplane_router::HealthTracker;
use serde_json::json;
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;
use tower::ServiceExt;
use wiremock::matchers::{body_partial_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn build_state(providers: ProviderRegistry) -> Arc<AppState> {
    Arc::new(AppState {
        health: HealthTracker::new(["anthropic", "openai"]),
        ..AppState::for_tests(providers)
    })
}

fn vk() -> VirtualKey {
    serde_json::from_value(json!({
        "name": "test-key",
        "routeplane_key": "rp_test",
        "provider_keys": { "anthropic": "ak-test", "openai": "sk-openai" }
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

fn anthropic_registry(base_url: &str) -> ProviderRegistry {
    let mut providers: ProviderRegistry = HashMap::new();
    providers.insert(
        "anthropic",
        Arc::new(AnthropicProvider::with_base_url(base_url)) as Arc<dyn Provider>,
    );
    providers
}

async fn invoke(state: Arc<AppState>, headers: HeaderMap, body: serde_json::Value) -> Response {
    let req = serde_json::from_value(body).expect("anthropic request deserializes");
    messages(
        State(state),
        Extension(vk()),
        Extension(ctx()),
        Extension(TenantGuardrails(None)),
        headers,
        routeplane::api_error::OpenAiJson(req),
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

/// The canonical Anthropic non-stream response the mock returns.
fn anthropic_ok_body() -> serde_json::Value {
    json!({
        "id": "msg_01abc",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-5-sonnet",
        "content": [{ "type": "text", "text": "the answer" }],
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": { "input_tokens": 12, "output_tokens": 7 }
    })
}

#[tokio::test]
async fn system_plus_user_routes_through_pipeline_to_anthropic_shape() {
    let server = MockServer::start().await;
    // The mock asserts the OUTBOUND native Anthropic wire: system lifted to the
    // top-level field, the user message a bare string, max_tokens threaded.
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("x-api-key", "ak-test"))
        .and(body_partial_json(json!({
            "model": "claude-3-5-sonnet",
            "max_tokens": 256,
            "system": "be terse",
            "messages": [{ "role": "user", "content": "hello" }]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(anthropic_ok_body()))
        .mount(&server)
        .await;
    let state = build_state(anthropic_registry(&server.uri()));

    let resp = invoke(
        state,
        HeaderMap::new(),
        json!({
            "model": "claude-3-5-sonnet",
            "max_tokens": 256,
            "system": "be terse",
            "messages": [{ "role": "user", "content": "hello" }]
        }),
    )
    .await;

    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;
    // Anthropic Messages response shape (NOT OpenAI chat.completion).
    assert_eq!(v["id"], "msg_01abc");
    assert_eq!(v["type"], "message");
    assert_eq!(v["role"], "assistant");
    assert_eq!(v["model"], "claude-3-5-sonnet");
    assert_eq!(v["content"][0]["type"], "text");
    assert_eq!(v["content"][0]["text"], "the answer");
    assert_eq!(v["stop_reason"], "end_turn");
    assert_eq!(v["stop_sequence"], serde_json::Value::Null);
    assert_eq!(v["usage"]["input_tokens"], 12);
    assert_eq!(v["usage"]["output_tokens"], 7);
}

#[tokio::test]
async fn base64_image_block_round_trips_to_native_anthropic_image() {
    let server = MockServer::start().await;
    // The mock matches only when the inbound image block was reconstructed to a
    // canonical data URL AND the Anthropic adapter re-emitted the native base64
    // image block on egress (the full inbound→canonical→native round-trip).
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(body_partial_json(json!({
            "messages": [{
                "role": "user",
                "content": [
                    { "type": "text", "text": "what is this" },
                    { "type": "image", "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "AAAA"
                    }}
                ]
            }]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(anthropic_ok_body()))
        .expect(1)
        .mount(&server)
        .await;
    let state = build_state(anthropic_registry(&server.uri()));

    let resp = invoke(
        state,
        HeaderMap::new(),
        json!({
            "model": "claude-3-5-sonnet",
            "max_tokens": 64,
            "messages": [{
                "role": "user",
                "content": [
                    { "type": "text", "text": "what is this" },
                    { "type": "image", "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "AAAA"
                    }}
                ]
            }]
        }),
    )
    .await;
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "the native image block must reach the wire"
    );
}

#[tokio::test]
async fn missing_max_tokens_is_anthropic_shaped_400() {
    let state = build_state(anthropic_registry("http://127.0.0.1:9"));
    let resp = invoke(
        state,
        HeaderMap::new(),
        json!({
            "model": "claude-3-5-sonnet",
            "messages": [{ "role": "user", "content": "hi" }]
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let v = body_json(resp).await;
    // Anthropic error shape: { type: "error", error: { type, message } }.
    assert_eq!(v["type"], "error");
    assert_eq!(v["error"]["type"], "invalid_request_error");
    assert!(v["error"]["message"]
        .as_str()
        .unwrap()
        .contains("max_tokens"));
}

#[tokio::test]
async fn stream_true_is_documented_anthropic_shaped_400() {
    let state = build_state(anthropic_registry("http://127.0.0.1:9"));
    let resp = invoke(
        state,
        HeaderMap::new(),
        json!({
            "model": "claude-3-5-sonnet",
            "max_tokens": 64,
            "stream": true,
            "messages": [{ "role": "user", "content": "hi" }]
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let v = body_json(resp).await;
    assert_eq!(v["type"], "error");
    assert!(v["error"]["message"]
        .as_str()
        .unwrap()
        .contains("/v1/chat/completions"));
}

#[tokio::test]
async fn email_in_message_is_masked_before_egress() {
    let server = MockServer::start().await;
    // The mock ONLY matches when the user message was masked to "[EMAIL_MASKED]"
    // before egress. Asserting 200 proves classify-then-mask ran on the
    // TRANSLATED Anthropic content — /v1/messages is not a PII-egress bypass.
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(body_partial_json(json!({
            "messages": [{ "role": "user", "content": "[EMAIL_MASKED]" }]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(anthropic_ok_body()))
        .expect(1)
        .mount(&server)
        .await;
    let state = build_state(anthropic_registry(&server.uri()));

    let resp = invoke(
        state,
        HeaderMap::new(),
        json!({
            "model": "claude-3-5-sonnet",
            "max_tokens": 64,
            "messages": [{ "role": "user", "content": "user@example.com" }]
        }),
    )
    .await;
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "masked message must match the mock"
    );
}

// --- auth gate --------------------------------------------------------------

const KEYS: &str = r#"{"keys":[
    {"name":"k_acme","routeplane_key":"rp_acme","provider_keys":{"anthropic":"x"},"tenant_id":"t_acme","tier":"free"}
]}"#;

fn auth() -> SharedAuthState {
    shared_auth_state(AuthState::load_from_json(KEYS, "test").expect("registry loads"))
}

/// A minimal router mirroring `main.rs`'s authed wiring: `/v1/messages` behind the
/// real `auth_middleware` + `SharedAuthState`. Asserts the auth gate rejects an
/// unauthenticated caller with the standard 401 envelope BEFORE the handler runs.
fn authed_router() -> Router {
    let state = build_state(anthropic_registry("http://127.0.0.1:9"));
    Router::new()
        .route("/v1/messages", post(messages))
        .layer(axum::middleware::from_fn(auth_middleware))
        .layer(axum::Extension(auth()))
        .with_state(state)
}

#[tokio::test]
async fn unauthenticated_messages_is_rejected_401() {
    let resp = authed_router()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"claude-3-5-sonnet","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}"#,
                ))
                .expect("request builds"),
        )
        .await
        .expect("router responds");
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    let v = body_json(resp).await;
    assert_eq!(v["error"]["code"], "invalid_api_key");
}

#[tokio::test]
async fn no_provider_header_defaults_to_anthropic() {
    // With NO x-routeplane-provider, /v1/messages must default to `anthropic`.
    // The registry holds ONLY anthropic; a successful 200 proves the default
    // chain selected it (an `openai` default would 500 — no openai in registry).
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(anthropic_ok_body()))
        .mount(&server)
        .await;
    let state = build_state(anthropic_registry(&server.uri()));

    let resp = invoke(
        state,
        HeaderMap::new(),
        json!({
            "model": "claude-3-5-sonnet",
            "max_tokens": 32,
            "messages": [{ "role": "user", "content": "hi" }]
        }),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
}
