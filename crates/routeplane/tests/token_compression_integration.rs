//! Hermetic integration tests for RTK `tool_result` token-compression ([ADR-085],
//! slice 4). The handler is called directly; the only "network" is a localhost
//! wiremock standing in for OpenAI. These assert the load-bearing properties:
//!
//! - **ON** (Feature::TokenCompression granted): `tool`-role content is compressed
//!   in the body the PROVIDER receives (strictly shorter) — this FAILS without the
//!   proxy wiring, so it is the proof the feature is actually wired.
//! - **OFF** (default): the forwarded `tool` content is byte-identical to the input
//!   — zero-cost-when-off / additivity (the `ab_parity`/golden guard property).
//! - **Role gate**: only `tool`-role content is touched; a `user` message with the
//!   same payload is forwarded unchanged even when the feature is active.
//! - **Ordering**: PII masking runs BEFORE compression — a tool message carrying a
//!   phone number reaches the provider masked (never the raw value) AND compressed,
//!   proving compression operates on already-masked content (it cannot defeat DLP).

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::IntoResponse;
use axum::Extension;
use routeplane::auth::{TenantContext, TenantGuardrails, VirtualKey};
use routeplane::proxy::{chat_completions, AppState, ProviderRegistry};
use routeplane_adapters::openai::OpenAIProvider;
use routeplane_adapters::Provider;
use routeplane_entitlements::{CapabilitySet, Feature, Tier};
use routeplane_router::HealthTracker;
use routeplane_types::{ChatCompletionRequest, Message};
use serde_json::json;
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn build_state(openai_base_url: &str) -> Arc<AppState> {
    let mut providers: ProviderRegistry = HashMap::new();
    providers.insert(
        "openai",
        Arc::new(OpenAIProvider::with_base_url(openai_base_url)) as Arc<dyn Provider>,
    );
    Arc::new(AppState {
        health: HealthTracker::new(["openai"]),
        ..AppState::for_tests(providers)
    })
}

fn vk() -> VirtualKey {
    serde_json::from_value(json!({
        "name": "test-key",
        "routeplane_key": "rp_test",
        "provider_keys": { "openai": "test-api-key" }
    }))
    .expect("virtual key deserializes")
}

/// A Standard-tier tenant context. `token_compression` grants
/// `Feature::TokenCompression` via a per-tenant OVERRIDE (it is NOT in the
/// Standard baseline — paid tiers reach it opt-in only; the Free baseline
/// carries it per ADR-088 Bundle B, exercised by `free_ctx` below).
fn ctx(token_compression: bool) -> TenantContext {
    let overrides = if token_compression {
        BTreeSet::from([Feature::TokenCompression])
    } else {
        BTreeSet::new()
    };
    TenantContext {
        tenant_id: "t_test".into(),
        tier: Tier::Standard,
        capabilities: CapabilitySet::resolve(Tier::Standard, &overrides, &BTreeSet::new()),
        compliance_frameworks: Vec::new(),
        compliance_mode: routeplane::auth::ComplianceMode::Strict,
    }
}

/// A Free-tier tenant context with NO overrides — entitlement comes purely from
/// the tier baseline (ADR-088 Bundle B). `held_back` simulates the release
/// plane answering `released = false` (the Unleash `token_compression`
/// constraint OFF for `tier == free`, or `RP_ROLLOUT_HOLDBACKS`): auth folds
/// that answer into the holdback set exactly like this.
fn free_ctx(held_back: bool) -> TenantContext {
    let holdbacks = if held_back {
        BTreeSet::from([Feature::TokenCompression])
    } else {
        BTreeSet::new()
    };
    TenantContext {
        tenant_id: "t_free".into(),
        tier: Tier::Free,
        capabilities: CapabilitySet::resolve(Tier::Free, &BTreeSet::new(), &holdbacks),
        compliance_frameworks: Vec::new(),
        compliance_mode: routeplane::auth::ComplianceMode::Strict,
    }
}

/// A single-message request with the given role + content.
fn one_message(role: &str, content: &str) -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: "gpt-4o".into(),
        messages: vec![Message {
            role: role.into(),
            content: content.into(),
            name: None,
            cache_control: None,
            tool_calls: None,
            tool_call_id: (role == "tool").then(|| "call_1".to_string()),
        }],
        ..Default::default()
    }
}

/// A verbose, generically-compressible tool payload (200 numbered lines) — RTK's
/// smart-truncate filter keeps the first/last and omits the middle, so this shrinks.
fn big_log() -> String {
    let mut s = String::new();
    for i in 0..200 {
        s.push_str(&format!("log line number {i} doing some work\n"));
    }
    s
}

async fn call(
    state: Arc<AppState>,
    tenant: TenantContext,
    body: ChatCompletionRequest,
) -> StatusCode {
    let resp = chat_completions(
        State(state),
        Extension(vk()),
        Extension(tenant),
        Extension(TenantGuardrails(None)),
        HeaderMap::new(),
        routeplane::api_error::OpenAiJson(body),
    )
    .await
    .into_response();
    resp.status()
}

/// Mock OpenAI: 200 with a fixed assistant body (we only inspect what it RECEIVED).
async fn mount_openai(server: &MockServer) {
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1_700_000_000,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": "ok" },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12 }
        })))
        .mount(server)
        .await;
}

/// Read the `messages[0].content` the provider actually received.
async fn forwarded_content(server: &MockServer) -> String {
    let received = server.received_requests().await.unwrap();
    assert_eq!(received.len(), 1, "exactly one upstream call");
    let sent: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();
    sent["messages"][0]["content"].as_str().unwrap().to_string()
}

/// PROOF: with the feature active, a verbose `tool` message reaches the provider
/// COMPRESSED (strictly shorter). Fails without the proxy wiring.
#[tokio::test]
async fn compresses_tool_content_when_feature_active() {
    let server = MockServer::start().await;
    mount_openai(&server).await;
    let state = build_state(&server.uri());

    let original = big_log();
    let status = call(state, ctx(true), one_message("tool", &original)).await;
    assert_eq!(status, StatusCode::OK);

    let sent = forwarded_content(&server).await;
    assert!(
        sent.len() < original.len(),
        "tool content was not compressed: sent {} bytes vs original {} bytes",
        sent.len(),
        original.len()
    );
}

/// ADDITIVITY: with the feature OFF (default), the forwarded `tool` content is
/// byte-identical to the input — the zero-cost-when-off guarantee.
#[tokio::test]
async fn tool_content_byte_identical_when_feature_off() {
    let server = MockServer::start().await;
    mount_openai(&server).await;
    let state = build_state(&server.uri());

    let original = big_log();
    let status = call(state, ctx(false), one_message("tool", &original)).await;
    assert_eq!(status, StatusCode::OK);

    let sent = forwarded_content(&server).await;
    assert_eq!(
        sent, original,
        "feature OFF must forward the tool content unchanged"
    );
}

/// ROLE GATE: only `tool`-role content is compressed. A `user` message with the
/// same verbose payload is forwarded unchanged even when the feature is active.
#[tokio::test]
async fn only_tool_role_is_compressed() {
    let server = MockServer::start().await;
    mount_openai(&server).await;
    let state = build_state(&server.uri());

    let original = big_log();
    let status = call(state, ctx(true), one_message("user", &original)).await;
    assert_eq!(status, StatusCode::OK);

    let sent = forwarded_content(&server).await;
    assert_eq!(sent, original, "non-tool roles must not be compressed");
}

/// ORDERING (security): PII masking runs BEFORE compression. A phone number on the
/// FIRST line (which smart-truncate keeps) reaches the provider MASKED — never the
/// raw value — and the payload is still compressed. Proves compression operates on
/// already-masked content and cannot defeat DLP.
#[tokio::test]
async fn masking_runs_before_compression() {
    let server = MockServer::start().await;
    mount_openai(&server).await;
    let state = build_state(&server.uri());

    let original = format!("contact 415-555-2671 for access\n{}", big_log());
    let status = call(state, ctx(true), one_message("tool", &original)).await;
    assert_eq!(status, StatusCode::OK);

    let sent = forwarded_content(&server).await;
    assert!(
        !sent.contains("415-555-2671"),
        "raw PII reached the provider — masking did not run before compression: {sent}"
    );
    assert!(
        sent.len() < original.len(),
        "payload was not compressed: {} vs {}",
        sent.len(),
        original.len()
    );
}

/// ADR-088 Bundle B: a FREE-tier tenant with NO overrides is entitled via the
/// tier baseline alone — verbose `tool` content reaches the provider compressed.
/// This is the CE value proposition wired end-to-end (baseline → capability →
/// the proxy.rs gate).
#[tokio::test]
async fn free_tier_baseline_grants_compression() {
    let server = MockServer::start().await;
    mount_openai(&server).await;
    let state = build_state(&server.uri());

    let original = big_log();
    let status = call(state, free_ctx(false), one_message("tool", &original)).await;
    assert_eq!(status, StatusCode::OK);

    let sent = forwarded_content(&server).await;
    assert!(
        sent.len() < original.len(),
        "Free baseline did not grant compression: sent {} bytes vs original {} bytes",
        sent.len(),
        original.len()
    );
}

/// REVERSIBILITY (ADR-088 §2c): with the release plane answering
/// `released = false` for `token_compression` (the Unleash tier==free
/// constraint before the flip, or `RP_ROLLOUT_HOLDBACKS`), the entitled Free
/// tenant's tool content is forwarded byte-identical — the grant lands inert
/// until the deliberate release, and the kill switch restores exactly this.
#[tokio::test]
async fn free_tier_holdback_keeps_tool_content_byte_identical() {
    let server = MockServer::start().await;
    mount_openai(&server).await;
    let state = build_state(&server.uri());

    let original = big_log();
    let status = call(state, free_ctx(true), one_message("tool", &original)).await;
    assert_eq!(status, StatusCode::OK);

    let sent = forwarded_content(&server).await;
    assert_eq!(
        sent, original,
        "held-back token_compression must forward the tool content unchanged"
    );
}
