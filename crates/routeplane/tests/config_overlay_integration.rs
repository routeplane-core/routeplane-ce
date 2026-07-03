//! CP→DP model-enablement enforcement — integration tests ([ADR-063] / [PRD-039]).
//!
//! Drives the real `chat_completions` handler (no socket, no network) against an
//! in-process stub provider, with a `ConfigOverlay` INJECTED into `AppState`, and
//! asserts the four safety invariants of the first ADR-063 slice:
//!   1. Off by default / default-allow: an EMPTY overlay enforces nothing (200).
//!   2. A `(tenant, model)` marked `enabled = false` ⇒ 403 `model_disabled_for_tenant`.
//!   3. An allowed/absent model for the same tenant ⇒ proceeds (200).
//!   4. Tenant-scoped: a DIFFERENT tenant requesting the disabled model ⇒ allowed.
//!
//! The poller's network fetch is NOT exercised here (it is factored so the BUILDER
//! is unit-tested without network in `src/config_overlay.rs`); these tests inject
//! the overlay directly via the public `AppState` field + the builder.

use async_trait::async_trait;
use axum::body::to_bytes;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::IntoResponse;
use axum::Extension;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use routeplane::auth::{TenantContext, TenantGuardrails, VirtualKey};
use routeplane::config::DeadlineConfig;
use routeplane::config_overlay::{ConfigOverlay, CpModelConfig};
use routeplane::proxy::{chat_completions, AppState, ProviderRegistry};
use routeplane_adapters::{Provider, ProviderError};
use routeplane_entitlements::{CapabilitySet, Tier};
use routeplane_router::HealthTracker;
use routeplane_types::{ChatCompletionRequest, ChatCompletionResponse, Message};

const DISABLED_MODEL: &str = "blocked-model";
const ALLOWED_MODEL: &str = "gpt-4o";
const TENANT_A: &str = "tenant-a";
const TENANT_B: &str = "tenant-b";

/// In-process stub provider that returns a canned OpenAI-shaped response for ANY
/// model — so a 200 here proves the request reached dispatch (i.e. the overlay did
/// NOT block it). Registered under every provider name the requests route to.
struct StubProvider;

#[async_trait]
impl Provider for StubProvider {
    fn name(&self) -> &'static str {
        "openai"
    }
    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
        _api_key: String,
    ) -> Result<ChatCompletionResponse, ProviderError> {
        let body = serde_json::json!({
            "id": "chatcmpl-stub",
            "object": "chat.completion",
            "created": 0,
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": "ok" },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 }
        });
        Ok(serde_json::from_value(body).expect("stub response deserializes"))
    }
}

/// Build an `AppState` backed by the stub provider, with the given overlay injected.
fn state_with_overlay(overlay: ConfigOverlay) -> Arc<AppState> {
    let mut providers: ProviderRegistry = HashMap::new();
    providers.insert("openai", Arc::new(StubProvider) as Arc<dyn Provider>);
    Arc::new(AppState {
        health: HealthTracker::new(["openai"]),
        deadline_config: DeadlineConfig {
            request_deadline: Duration::from_secs(30),
            per_attempt_timeout: Duration::from_secs(30),
        },
        config_overlay: Arc::new(arc_swap::ArcSwap::from_pointee(overlay)),
        ..AppState::for_tests(providers)
    })
}

fn vk_for(tenant_id: &str) -> VirtualKey {
    serde_json::from_value(serde_json::json!({
        "name": "test-key",
        "routeplane_key": "rp_test",
        "tenant_id": tenant_id,
        "provider_keys": { "openai": "test-api-key" }
    }))
    .expect("virtual key deserializes")
}

fn ctx_for(tenant_id: &str) -> TenantContext {
    TenantContext {
        tenant_id: tenant_id.to_string(),
        tier: Tier::Standard,
        capabilities: CapabilitySet::resolve(
            Tier::Standard,
            &std::collections::BTreeSet::new(),
            &std::collections::BTreeSet::new(),
        ),
        compliance_frameworks: Vec::new(),
        compliance_mode: routeplane::auth::ComplianceMode::Strict,
    }
}

fn payload(model: &str) -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: model.into(),
        messages: vec![Message {
            role: "user".into(),
            content: "hello".into(),
            name: None,
            cache_control: None,
            tool_calls: None,
            tool_call_id: None,
        }],
        stream: Some(false),
        ..Default::default()
    }
}

/// Drive one buffered request for `(tenant, model)` and return the response.
async fn drive(state: &Arc<AppState>, tenant_id: &str, model: &str) -> axum::response::Response {
    chat_completions(
        State(state.clone()),
        Extension(vk_for(tenant_id)),
        Extension(ctx_for(tenant_id)),
        Extension(TenantGuardrails(None)),
        HeaderMap::new(),
        routeplane::api_error::OpenAiJson(payload(model)),
    )
    .await
    .into_response()
}

/// Overlay with `blocked-model` disabled for tenant-a only.
fn overlay_blocking_for_a() -> ConfigOverlay {
    ConfigOverlay::from_tenant_configs([(
        TENANT_A.to_string(),
        vec![
            CpModelConfig {
                model_id: DISABLED_MODEL.into(),
                enabled: false,
            },
            CpModelConfig {
                model_id: ALLOWED_MODEL.into(),
                enabled: true,
            },
        ],
    )])
}

async fn body_json(resp: axum::response::Response) -> serde_json::Value {
    let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

#[tokio::test]
async fn disabled_model_for_tenant_is_403() {
    let state = state_with_overlay(overlay_blocking_for_a());
    let resp = drive(&state, TENANT_A, DISABLED_MODEL).await;
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    let v = body_json(resp).await;
    assert_eq!(v["error"]["code"], "model_disabled_for_tenant");
    assert_eq!(v["error"]["param"], "model");
    assert!(v["error"]["message"]
        .as_str()
        .unwrap()
        .contains(DISABLED_MODEL));
}

#[tokio::test]
async fn allowed_model_for_same_tenant_proceeds() {
    let state = state_with_overlay(overlay_blocking_for_a());
    // Explicitly enabled in the overlay ⇒ proceeds to the stub provider (200).
    let resp = drive(&state, TENANT_A, ALLOWED_MODEL).await;
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn absent_model_for_known_tenant_proceeds() {
    let state = state_with_overlay(overlay_blocking_for_a());
    // No overlay entry for this model ⇒ default-allow ⇒ 200.
    let resp = drive(&state, TENANT_A, "claude-3-5-sonnet").await;
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn enforcement_is_tenant_scoped() {
    let state = state_with_overlay(overlay_blocking_for_a());
    // tenant-b has NO overlay entry for the model disabled for tenant-a ⇒ allowed.
    let resp = drive(&state, TENANT_B, DISABLED_MODEL).await;
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn empty_overlay_enforces_nothing() {
    // The off-by-default state: an empty overlay (the default when no poller runs).
    let state = state_with_overlay(ConfigOverlay::empty());
    // Even the "blocked" model proceeds — nothing is enforced.
    let resp = drive(&state, TENANT_A, DISABLED_MODEL).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let resp2 = drive(&state, TENANT_A, ALLOWED_MODEL).await;
    assert_eq!(resp2.status(), StatusCode::OK);
}
