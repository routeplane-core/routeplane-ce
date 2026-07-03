//! Model Catalog slug-gate ordering — integration tests (PRD-008 FR-3/FR-4).
//!
//! Regression guard for the disable-gate smuggling hole: a `@slug/model` address
//! resolves to a bare model id, and the CP-overlay `model_enabled` disable gate is
//! an EXACT-match lookup. If slug resolution ran AFTER the disable/compliance
//! gates, `@int/gpt-4o` would miss the `gpt-4o` disable row (None → default-allow),
//! get rewritten to `gpt-4o`, and dispatch — bypassing an operator's Console kill
//! switch. The fix resolves the slug FIRST, so both the CP-config disable gate and
//! the org compliance gate enforce on the RESOLVED model id (the ADR-086 §A4
//! anti-smuggling stance, applied to slugs). These tests drive the real
//! `chat_completions` handler (no socket, no network) against an in-process stub
//! provider with `Feature::ModelCatalog` active and a `ConfigOverlay` injected.

use async_trait::async_trait;
use axum::body::to_bytes;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::IntoResponse;
use axum::Extension;
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;
use std::time::Duration;

use routeplane::auth::{ComplianceMode, TenantContext, TenantGuardrails, VirtualKey};
use routeplane::config::DeadlineConfig;
use routeplane::config_overlay::{ConfigOverlay, CpModelConfig};
use routeplane::proxy::{chat_completions, AppState, ProviderRegistry};
use routeplane_adapters::{Provider, ProviderError};
use routeplane_entitlements::{CapabilitySet, Feature, Tier};
use routeplane_router::HealthTracker;
use routeplane_types::{ChatCompletionRequest, ChatCompletionResponse, Message};

const TENANT: &str = "t_cat";

/// In-process stub provider that returns a canned OpenAI-shaped response for ANY
/// model — so a 200 here proves the request reached dispatch (the gates did NOT
/// block it), and the echoed `model` proves the slug was rewritten to the bare id.
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

/// `AppState` backed by the stub provider, with the given overlay injected.
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

/// A key with a Model-Catalog integration (`@int/gpt-4o` → `gpt-4o`, `@ds/…` →
/// `deepseek-reasoner`) plus a bare-model allowlist (`gpt-4o-mini`, `gpt-4o`).
fn catalog_key() -> VirtualKey {
    serde_json::from_value(serde_json::json!({
        "name": "cat-key",
        "routeplane_key": "rp_cat",
        "tenant_id": TENANT,
        "provider_keys": { "openai": "test-api-key" },
        "provisioned_models": ["gpt-4o-mini", "gpt-4o"],
        "integrations": {
            "int": { "adapter": "openai", "models": ["gpt-4o"] },
            "ds": { "adapter": "openai", "models": ["deepseek-reasoner"] }
        }
    }))
    .expect("virtual key deserializes")
}

/// A `TenantContext` with `Feature::ModelCatalog` active (via the override set —
/// it is OFF in every tier baseline) and the given compliance posture.
fn ctx_with_catalog(frameworks: Vec<String>, mode: ComplianceMode) -> TenantContext {
    TenantContext {
        tenant_id: TENANT.to_string(),
        tier: Tier::Standard,
        capabilities: CapabilitySet::resolve(
            Tier::Standard,
            &BTreeSet::from([Feature::ModelCatalog]),
            &BTreeSet::new(),
        ),
        compliance_frameworks: frameworks,
        compliance_mode: mode,
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

async fn drive(state: &Arc<AppState>, ctx: TenantContext, model: &str) -> axum::response::Response {
    chat_completions(
        State(state.clone()),
        Extension(catalog_key()),
        Extension(ctx),
        Extension(TenantGuardrails(None)),
        HeaderMap::new(),
        routeplane::api_error::OpenAiJson(payload(model)),
    )
    .await
    .into_response()
}

async fn body_json(resp: axum::response::Response) -> serde_json::Value {
    let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

/// Overlay disabling `gpt-4o` for the tenant (an operator Console kill switch).
fn overlay_disabling_gpt4o() -> ConfigOverlay {
    ConfigOverlay::from_tenant_configs([(
        TENANT.to_string(),
        vec![CpModelConfig {
            model_id: "gpt-4o".into(),
            enabled: false,
        }],
    )])
}

/// THE FIX: a slug that resolves to a DISABLED model is rejected. Before the fix
/// the exact-match disable gate ran on `@int/gpt-4o` (miss → allow), then the
/// rewrite to `gpt-4o` dispatched past the kill switch. Now the slug resolves
/// first, so the gate sees `gpt-4o` and rejects.
#[tokio::test]
async fn slug_resolving_to_disabled_model_is_403() {
    let state = state_with_overlay(overlay_disabling_gpt4o());
    let ctx = ctx_with_catalog(Vec::new(), ComplianceMode::Strict);
    let resp = drive(&state, ctx, "@int/gpt-4o").await;
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    let v = body_json(resp).await;
    assert_eq!(v["error"]["code"], "model_disabled_for_tenant");
    // The 403 cites the RESOLVED bare model, not the `@slug/…` wrapper.
    assert!(v["error"]["message"].as_str().unwrap().contains("gpt-4o"));
    assert!(!v["error"]["message"].as_str().unwrap().contains('@'));
}

/// A slug resolving to an ENABLED, non-excluded model still proceeds — the fix
/// does not over-block. The stub echoes the rewritten bare model id.
#[tokio::test]
async fn slug_resolving_to_enabled_model_proceeds() {
    // Empty overlay ⇒ nothing disabled ⇒ default-allow.
    let state = state_with_overlay(ConfigOverlay::empty());
    let ctx = ctx_with_catalog(Vec::new(), ComplianceMode::Strict);
    let resp = drive(&state, ctx, "@int/gpt-4o").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;
    assert_eq!(
        v["model"], "gpt-4o",
        "slug should be rewritten to the bare id"
    );
}

/// The compliance gate also enforces on the resolved model: a slug that resolves
/// to a restriction-tagged model (`deepseek-reasoner` → DPDP/RBI/HIPAA) is blocked
/// in strict mode for a HIPAA tenant.
#[tokio::test]
async fn slug_resolving_to_compliance_excluded_model_is_403_strict() {
    let state = state_with_overlay(ConfigOverlay::empty());
    let ctx = ctx_with_catalog(vec!["HIPAA".to_string()], ComplianceMode::Strict);
    let resp = drive(&state, ctx, "@ds/deepseek-reasoner").await;
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    let v = body_json(resp).await;
    assert_eq!(v["error"]["code"], "model_compliance_excluded");
}

/// Regression guard: the BARE-model path still enforces the disable gate under
/// ModelCatalog — a provisioned-but-disabled model is rejected as before.
#[tokio::test]
async fn bare_provisioned_disabled_model_is_403() {
    let state = state_with_overlay(overlay_disabling_gpt4o());
    let ctx = ctx_with_catalog(Vec::new(), ComplianceMode::Strict);
    // `gpt-4o` is in the key's bare allowlist AND disabled in the overlay.
    let resp = drive(&state, ctx, "gpt-4o").await;
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    let v = body_json(resp).await;
    assert_eq!(v["error"]["code"], "model_disabled_for_tenant");
}

/// Documented side effect of resolve-first: a bare model that is BOTH not in the
/// key's provisioned allowlist AND disabled now returns `model_not_provisioned`
/// (the more fundamental default-deny) instead of `model_disabled_for_tenant`.
/// Both are 403; this pins the intended precedence.
#[tokio::test]
async fn bare_unprovisioned_and_disabled_model_is_not_provisioned() {
    // Disable a model that is NOT in the key's provisioned allowlist.
    let overlay = ConfigOverlay::from_tenant_configs([(
        TENANT.to_string(),
        vec![CpModelConfig {
            model_id: "claude-3-5-sonnet".into(),
            enabled: false,
        }],
    )]);
    let state = state_with_overlay(overlay);
    let ctx = ctx_with_catalog(Vec::new(), ComplianceMode::Strict);
    let resp = drive(&state, ctx, "claude-3-5-sonnet").await;
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    let v = body_json(resp).await;
    assert_eq!(v["error"]["code"], "model_not_provisioned");
}
