//! Hermetic integration tests for budgets & rate limits (PRD-008 / ADR-023
//! Mode L). The handler is called directly; the only "network" is a localhost
//! wiremock standing in for OpenAI. Covers: rate breach → 429 + truthful
//! headers, advisory headers on success ONLY when configured, legacy keys
//! byte-identical (no headers), and budget debit across two calls → 402.

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Extension;
use routeplane::auth::{TenantContext, TenantGuardrails, VirtualKey};
use routeplane::proxy::{chat_completions, AppState, ProviderRegistry};
use routeplane_adapters::openai::OpenAIProvider;
use routeplane_adapters::Provider;
use routeplane_entitlements::{CapabilitySet, Tier};
use routeplane_limits::{KeyLimits, KeyLimitsInput, LimitRegistry};
use routeplane_router::HealthTracker;
use routeplane_types::{ChatCompletionRequest, Message};
use serde_json::json;
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const RP_KEY: &str = "rp_test";
const TENANT: &str = "t_test";

/// Build a registry where `rp_test` / `t_test` carry the given `limits` JSON.
fn registry(limits_json: serde_json::Value) -> LimitRegistry {
    let limits: KeyLimits = serde_json::from_value(limits_json).expect("KeyLimits deserializes");
    LimitRegistry::build(vec![KeyLimitsInput {
        routeplane_key: RP_KEY.into(),
        tenant_id: TENANT.into(),
        limits,
    }])
}

fn build_state(openai_base_url: &str, limits: LimitRegistry) -> Arc<AppState> {
    let mut providers: ProviderRegistry = HashMap::new();
    providers.insert(
        "openai",
        Arc::new(OpenAIProvider::with_base_url(openai_base_url)) as Arc<dyn Provider>,
    );
    Arc::new(AppState {
        health: HealthTracker::new(["openai"]),
        limits,
        ..AppState::for_tests(providers)
    })
}

fn vk() -> VirtualKey {
    serde_json::from_value(json!({
        "name": "test-key",
        "routeplane_key": RP_KEY,
        "provider_keys": { "openai": "test-api-key" }
    }))
    .expect("virtual key deserializes")
}

fn ctx() -> TenantContext {
    // Free tier on purpose: limits are CORE enforcement, NOT entitlement-gated in
    // this build — a Free tenant's platform/key caps must still bite.
    TenantContext {
        tenant_id: TENANT.into(),
        tier: Tier::Free,
        capabilities: CapabilitySet::resolve(Tier::Free, &BTreeSet::new(), &BTreeSet::new()),
        compliance_frameworks: Vec::new(),
        compliance_mode: routeplane::auth::ComplianceMode::Strict,
    }
}

fn payload() -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: "gpt-4o".into(),
        messages: vec![Message {
            role: "user".into(),
            content: "hello".into(),
            name: None,
            cache_control: None,
            tool_calls: None,
            tool_call_id: None,
        }],
        temperature: None,
        top_p: None,
        stream: None,
        max_tokens: None,
        stop: None,
        n: None,
        presence_penalty: None,
        frequency_penalty: None,
        user: None,
        tools: None,
        tool_choice: None,
        parallel_tool_calls: None,
        ..Default::default()
    }
}

async fn invoke(state: Arc<AppState>) -> Response {
    chat_completions(
        State(state),
        Extension(vk()),
        Extension(ctx()),
        Extension(TenantGuardrails(None)),
        HeaderMap::new(),
        routeplane::api_error::OpenAiJson(payload()),
    )
    .await
    .into_response()
}

async fn mount_openai_ok(server: &MockServer) {
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1_700_000_000,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": "hi there" },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12 }
        })))
        .mount(server)
        .await;
}

async fn body_json(resp: Response) -> serde_json::Value {
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .expect("body readable");
    serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null)
}

// --- breach → 429 with truthful headers -----------------------------------------

#[tokio::test]
async fn second_request_in_window_is_429_with_headers() {
    let server = MockServer::start().await;
    mount_openai_ok(&server).await;
    let state = build_state(
        &server.uri(),
        registry(json!({
            "key": { "policy_id": "pk", "rate": { "requests_per_min": 1 } }
        })),
    );

    // First request: admitted (consumes the single slot) → 200.
    let first = invoke(state.clone()).await;
    assert_eq!(first.status(), StatusCode::OK);

    // Second request in the same minute: rate-limited.
    let second = invoke(state.clone()).await;
    assert_eq!(second.status(), StatusCode::TOO_MANY_REQUESTS);
    let h = second.headers();
    assert_eq!(h.get("x-ratelimit-limit-requests").unwrap(), "1");
    assert_eq!(h.get("x-ratelimit-remaining-requests").unwrap(), "0");
    assert!(h.get("x-ratelimit-reset-requests").is_some());
    assert!(h.get("retry-after").is_some());
    assert_eq!(
        h.get("x-routeplane-limit-type").unwrap(),
        "rate_limit_requests"
    );
    assert_eq!(h.get("x-routeplane-limit-scope").unwrap(), "key");
    assert_eq!(h.get("x-routeplane-limit-policy").unwrap(), "pk");

    let v = body_json(second).await;
    assert_eq!(v["error"]["type"], "rate_limit_exceeded");
    assert_eq!(v["error"]["code"], "routeplane_rate_limit_exceeded");
}

// --- advisory headers on success when configured --------------------------------

#[tokio::test]
async fn success_carries_advisory_headers_when_limits_configured() {
    let server = MockServer::start().await;
    mount_openai_ok(&server).await;
    let state = build_state(
        &server.uri(),
        registry(json!({
            "key": { "rate": { "requests_per_min": 5, "tokens_per_min": 1000 } }
        })),
    );

    let resp = invoke(state).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let h = resp.headers();
    // One request consumed at admit → 4 remaining.
    assert_eq!(h.get("x-ratelimit-limit-requests").unwrap(), "5");
    assert_eq!(h.get("x-ratelimit-remaining-requests").unwrap(), "4");
    // 12 tokens settled-after → 988 remaining.
    assert_eq!(h.get("x-ratelimit-limit-tokens").unwrap(), "1000");
    assert_eq!(h.get("x-ratelimit-remaining-tokens").unwrap(), "988");
}

// --- legacy key (no limits) ⇒ byte-identical, no headers ------------------------

#[tokio::test]
async fn legacy_key_without_limits_emits_no_ratelimit_headers() {
    let server = MockServer::start().await;
    mount_openai_ok(&server).await;
    let state = build_state(&server.uri(), LimitRegistry::empty());

    let resp = invoke(state).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let h = resp.headers();
    assert!(h.get("x-ratelimit-limit-requests").is_none());
    assert!(h.get("x-ratelimit-remaining-requests").is_none());
    assert!(h.get("x-ratelimit-limit-tokens").is_none());
    assert!(h.get("x-routeplane-budget-remaining").is_none());
}

// --- budget debit across two calls → 402 ----------------------------------------

#[tokio::test]
async fn budget_is_402_after_first_call_debits_past_the_cap() {
    let server = MockServer::start().await;
    mount_openai_ok(&server).await;
    // A $0.000001 daily cap: the first call's settled cost blows past it, so the
    // SECOND call is rejected — proving settle-after debit carries across calls.
    let state = build_state(
        &server.uri(),
        registry(json!({
            "key": { "policy_id": "budget-pk", "budget": { "cost_micro_usd_daily": 1 } }
        })),
    );

    // Call 1: headroom (0 < 1) → admitted, then debits real cost on success.
    let first = invoke(state.clone()).await;
    assert_eq!(first.status(), StatusCode::OK);

    // Call 2: the debited spend now exceeds the cap → 402 before any provider call.
    let second = invoke(state.clone()).await;
    assert_eq!(second.status(), StatusCode::PAYMENT_REQUIRED);
    let h = second.headers();
    assert_eq!(h.get("x-routeplane-limit-type").unwrap(), "budget_cost");
    assert_eq!(h.get("x-routeplane-limit-scope").unwrap(), "key");
    assert_eq!(h.get("x-routeplane-limit-policy").unwrap(), "budget-pk");
    assert!(h.get("retry-after").is_some());

    let v = body_json(second).await;
    assert_eq!(v["error"]["type"], "insufficient_quota");
    assert_eq!(v["error"]["code"], "routeplane_budget_exceeded");
}

// --- per-tenant counter shared across keys; no cross-tenant bleed ----------------

#[tokio::test]
async fn tenant_scope_blocks_while_a_different_tenant_is_unaffected() {
    let server = MockServer::start().await;
    mount_openai_ok(&server).await;
    // Tenant-scope cap of 1 req/min, declared on rp_test / t_test.
    let state = build_state(
        &server.uri(),
        registry(json!({
            "tenant": { "policy_id": "ten", "rate": { "requests_per_min": 1 } }
        })),
    );

    // First request for t_test → 200 (consumes the tenant slot).
    assert_eq!(invoke(state.clone()).await.status(), StatusCode::OK);
    // Second request for the SAME tenant → 429 on the tenant scope.
    let denied = invoke(state.clone()).await;
    assert_eq!(denied.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        denied.headers().get("x-routeplane-limit-scope").unwrap(),
        "tenant"
    );

    // A DIFFERENT tenant (resolved against an empty registry slot) is unaffected:
    // build a separate state whose registry knows only a different tenant.
    let other_state = build_state(
        &server.uri(),
        LimitRegistry::build(vec![KeyLimitsInput {
            routeplane_key: "rp_other".into(),
            tenant_id: "t_other".into(),
            limits: serde_json::from_value(json!({
                "tenant": { "rate": { "requests_per_min": 1 } }
            }))
            .unwrap(),
        }]),
    );
    // rp_test/t_test resolves to UNLIMITED here (not in this registry) → always OK.
    assert_eq!(invoke(other_state.clone()).await.status(), StatusCode::OK);
    assert_eq!(invoke(other_state).await.status(), StatusCode::OK);
}

// --- error-path envelope (2026-06-12 dogfood: all-failed must be OpenAI-shaped,
// generic, and never leak provider/config detail) ---------------------------

#[tokio::test]
async fn all_providers_failed_returns_generic_openai_envelope_no_leak() {
    // Unreachable upstream, no mock mounted → the provider call fails → the
    // request exhausts its only candidate and hits the all-failed terminal.
    let state = build_state("http://127.0.0.1:9", LimitRegistry::empty());
    let resp = invoke(state).await;
    assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let v = body_json(resp).await;
    assert_eq!(v["error"]["code"], "upstream_error");
    assert_eq!(v["error"]["type"], "api_error");
    let msg = v["error"]["message"].as_str().unwrap().to_lowercase();
    // The dogfood saw "API key for openai not configured" leak — never again.
    assert!(!msg.contains("api key"), "leaked config: {msg}");
    assert!(!msg.contains("not configured"), "leaked config: {msg}");
    assert!(!msg.contains("127.0.0.1"), "leaked address: {msg}");
}
