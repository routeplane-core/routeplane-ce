//! ADR-087 multi-account per-provider key pool — integration tests.
//!
//! Drives the real `chat_completions` handler (no socket, no network) against
//! in-process stub providers that branch on the resolved `api_key`, with a virtual
//! key whose `provider_keys["openai"]` is a comma-pool. Proves the intra-pool
//! failover walk + per-key cooldown compose end-to-end.

use async_trait::async_trait;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::IntoResponse;
use axum::Extension;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use routeplane::auth::{TenantContext, TenantGuardrails, VirtualKey};
use routeplane::config::DeadlineConfig;
use routeplane::proxy::{chat_completions, AppState, ProviderRegistry};
use routeplane_adapters::{Provider, ProviderError};
use routeplane_router::HealthTracker;
use routeplane_types::{ChatCompletionRequest, ChatCompletionResponse, Message};

const TENANT: &str = "t_pool";

/// A stub that maps a set of `api_key`s to a 429 (RateLimited) and returns a canned
/// 200 for everything else — so it can model a rate-limited account vs a healthy one.
struct KeyedStub {
    rate_limited: Vec<&'static str>,
}

#[async_trait]
impl Provider for KeyedStub {
    fn name(&self) -> &'static str {
        "openai"
    }
    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
        api_key: String,
    ) -> Result<ChatCompletionResponse, ProviderError> {
        if self.rate_limited.iter().any(|k| *k == api_key) {
            return Err(ProviderError::RateLimited {
                provider: "openai".into(),
                retry_after: None,
                body: "rate limited".into(),
            });
        }
        let body = serde_json::json!({
            "id": "chatcmpl-stub", "object": "chat.completion", "created": 0,
            "model": request.model,
            "choices": [{ "index": 0,
                "message": { "role": "assistant", "content": "ok" },
                "finish_reason": "stop" }],
            "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 }
        });
        Ok(serde_json::from_value(body).expect("stub response deserializes"))
    }
}

/// A stub that 401s (Auth — dead key) every `api_key` NOT in `good`, and returns a
/// canned 200 for a good key. Models revoked/suspended accounts vs a healthy one —
/// and, because `Auth` `counts_as_health_failure`, exercises the shared-breaker
/// hazard (Findings 2/3) and the concatenated-pool bearer-key bug (Finding 1).
struct AuthStub {
    good: Vec<&'static str>,
}

#[async_trait]
impl Provider for AuthStub {
    fn name(&self) -> &'static str {
        "openai"
    }
    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
        api_key: String,
    ) -> Result<ChatCompletionResponse, ProviderError> {
        if !self.good.iter().any(|k| *k == api_key) {
            return Err(ProviderError::Auth {
                provider: "openai".into(),
                status: 401,
                body: "invalid api key".into(),
            });
        }
        let body = serde_json::json!({
            "id": "chatcmpl-stub", "object": "chat.completion", "created": 0,
            "model": request.model,
            "choices": [{ "index": 0,
                "message": { "role": "assistant", "content": "ok" },
                "finish_reason": "stop" }],
            "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 }
        });
        Ok(serde_json::from_value(body).expect("stub response deserializes"))
    }
}

fn state_with(rate_limited: Vec<&'static str>) -> Arc<AppState> {
    state_with_provider(Arc::new(KeyedStub { rate_limited }) as Arc<dyn Provider>)
}

fn state_with_auth(good: Vec<&'static str>) -> Arc<AppState> {
    state_with_provider(Arc::new(AuthStub { good }) as Arc<dyn Provider>)
}

fn state_with_provider(provider: Arc<dyn Provider>) -> Arc<AppState> {
    let mut providers: ProviderRegistry = HashMap::new();
    providers.insert("openai", provider);
    Arc::new(AppState {
        health: HealthTracker::new(["openai"]),
        deadline_config: DeadlineConfig {
            request_deadline: Duration::from_secs(30),
            per_attempt_timeout: Duration::from_secs(30),
        },
        ..AppState::for_tests(providers)
    })
}

/// Virtual key with a two-account pool for `openai` (keyA index 0, keyB index 1).
fn pool_vk() -> VirtualKey {
    serde_json::from_value(serde_json::json!({
        "name": "pool-key",
        "routeplane_key": "rp_pool",
        "tenant_id": TENANT,
        "provider_keys": { "openai": "keyA,keyB" }
    }))
    .expect("virtual key deserializes")
}

fn ctx() -> TenantContext {
    TenantContext {
        tenant_id: TENANT.to_string(),
        tier: routeplane_entitlements::Tier::Standard,
        capabilities: routeplane_entitlements::CapabilitySet::resolve(
            routeplane_entitlements::Tier::Standard,
            &Default::default(),
            &Default::default(),
        ),
        compliance_frameworks: Vec::new(),
        compliance_mode: routeplane::auth::ComplianceMode::Strict,
    }
}

/// A virtual key whose `openai` pool is the given comma-separated value.
fn pool_vk_with(pool: &str) -> VirtualKey {
    serde_json::from_value(serde_json::json!({
        "name": "pool-key",
        "routeplane_key": "rp_pool",
        "tenant_id": TENANT,
        "provider_keys": { "openai": pool }
    }))
    .expect("virtual key deserializes")
}

fn payload_stream(stream: bool) -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: "gpt-4o".to_string(),
        messages: vec![Message {
            role: "user".into(),
            content: "hi".into(),
            name: None,
            cache_control: None,
            tool_calls: None,
            tool_call_id: None,
            refusal: None,
            reasoning_content: None,
        }],
        stream: Some(stream),
        ..Default::default()
    }
}

fn payload() -> ChatCompletionRequest {
    payload_stream(false)
}

async fn drive(state: &Arc<AppState>) -> axum::response::Response {
    drive_vk(state, pool_vk(), payload()).await
}

async fn drive_vk(
    state: &Arc<AppState>,
    vk: VirtualKey,
    body: ChatCompletionRequest,
) -> axum::response::Response {
    chat_completions(
        State(state.clone()),
        Extension(vk),
        Extension(ctx()),
        Extension(TenantGuardrails(None)),
        HeaderMap::new(),
        routeplane::api_error::OpenAiJson(body),
    )
    .await
    .into_response()
}

#[tokio::test]
async fn one_rate_limited_key_fails_over_to_the_healthy_key() {
    // keyA is rate-limited, keyB is healthy ⇒ the request succeeds via the pool
    // regardless of which key the RNG tries first (order-independent).
    let state = state_with(vec!["keyA"]);
    let resp = drive(&state).await;
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "pool must route around a 429'd key"
    );
}

#[tokio::test]
async fn whole_pool_walked_and_each_failing_key_is_cooled() {
    // BOTH keys 429 ⇒ the intra-pool walk tries both (failover between them), cools
    // each, then exhausts the (single) provider. Order-independent, so no RNG flake.
    let state = state_with(vec!["keyA", "keyB"]);
    let resp = drive(&state).await;
    assert_ne!(
        resp.status(),
        StatusCode::OK,
        "all keys down ⇒ not a success"
    );
    // Both pool keys were tried and cooled down (per-key cooldown recorded).
    assert!(
        state.health.key_cooled_until(TENANT, "openai", 0) > 0,
        "pool key 0 must be cooled after its 429"
    );
    assert!(
        state.health.key_cooled_until(TENANT, "openai", 1) > 0,
        "pool key 1 must be cooled after its 429"
    );
    // A DIFFERENT tenant's same-index key is untouched (cross-tenant isolation).
    assert_eq!(state.health.key_cooled_until("t_other", "openai", 0), 0);
}

// --- Finding 1: streaming path is pool-aware (was single-key `resolve_api_key`) --

#[tokio::test]
async fn streaming_request_walks_the_key_pool() {
    // Finding 1: a `stream: true` request for a pooled provider used to resolve ONE
    // key via the legacy single-key `resolve_api_key`, which sent the whole
    // "keyA,keyB" string as the bearer key ⇒ provider 401 ⇒ every stream hard-failed
    // with a 500. keyA is a dead key (401), keyB is healthy: the pool-aware
    // establishment walk must fail over to keyB and serve the SSE (order-independent,
    // so no RNG flake). On the UNFIXED code this returns 500 (the concatenated
    // "keyA,keyB" is neither keyB nor any good key).
    let state = state_with_auth(vec!["keyB"]);
    let resp = drive_vk(&state, pool_vk_with("keyA,keyB"), payload_stream(true)).await;
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "streaming pool must fail over to the healthy key, not send the pool string whole"
    );
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .expect("stream body collects");
    let body = String::from_utf8(bytes.to_vec()).expect("SSE body is utf8");
    assert!(
        body.contains("data: [DONE]"),
        "a served stream terminates with data: [DONE] — got: {body}"
    );
    assert!(
        body.contains("\"content\":\"ok\""),
        "the healthy key (keyB) actually served the stubbed chunk — got: {body}"
    );
    // NB: whether the dead key (index 0) is cooled depends on which key the RNG
    // walked first (keyB first ⇒ keyA never reached), so that is intentionally NOT
    // asserted here — the order-independent guarantee is that the pool serves.
}

// --- Findings 2/3: per-key failures must NOT feed the shared provider breaker -----

#[tokio::test]
async fn dead_key_pool_does_not_open_the_shared_provider_breaker() {
    // Findings 2/3: the pool walk used to record a provider-level circuit-breaker
    // failure on EVERY per-key failure. A single request against a pool of >= 5 dead
    // keys therefore recorded >= 5 consecutive provider failures and opened the SHARED
    // per-provider breaker — downing the provider for EVERY tenant in the cell, the
    // exact cross-tenant hazard ADR-087 §4 claimed to prevent. Six dead keys (all
    // 401, non-retryable ⇒ exactly one attempt each) in ONE request:
    //   * UNFIXED: 6 consecutive provider failures ⇒ breaker OPEN ⇒ is_available == false.
    //   * FIXED:   the breaker is fed EXACTLY ONCE on pool exhaustion ⇒ still closed.
    // Order-independent (all keys identical class), so no RNG flake.
    let state = state_with_auth(vec![]); // nothing is a good key ⇒ all 401
    let resp = drive_vk(&state, pool_vk_with("d0,d1,d2,d3,d4,d5"), payload()).await;
    assert_ne!(
        resp.status(),
        StatusCode::OK,
        "a fully-dead pool cannot succeed"
    );
    assert!(
        state.health.is_available("openai"),
        "a single request through a dead key pool must NOT open the shared provider \
         breaker (ADR-087 §4: the breaker is fed once, on exhaustion — not per key)"
    );
    // Every pool key was still walked (per-key cooldown recorded for each) — the
    // breaker never tripped mid-walk to skip the tail of the pool.
    for idx in 0..6 {
        assert!(
            state.health.key_cooled_until(TENANT, "openai", idx) > 0,
            "dead pool key {idx} must be cooled after its 401 (walk reached it)"
        );
    }
}
