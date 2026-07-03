//! Shared harness for the handler-level integration tests: an in-process stub
//! provider (no network, no wiremock) behind the real `chat_completions`
//! handler, so tests measure/inspect pure gateway behavior.
//!
//! Each integration-test binary compiles this module independently, so any one
//! binary uses only a subset of the helpers — hence the file-wide
//! `allow(dead_code)`.

#![allow(dead_code)]

use async_trait::async_trait;
use axum::body::to_bytes;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::IntoResponse;
use axum::Extension;
use routeplane::auth::{TenantContext, TenantGuardrails, VirtualKey};
use routeplane::config::DeadlineConfig;
use routeplane::proxy::{chat_completions, AppState, ProviderRegistry};
use routeplane_adapters::{Provider, ProviderError};
use routeplane_entitlements::{CapabilitySet, Tier};
use routeplane_types::{ChatCompletionRequest, ChatCompletionResponse, Message};
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

pub const RP_KEY: &str = "rp_test";
pub const TENANT: &str = "t_test";

fn ok_chat_body() -> serde_json::Value {
    json!({
        "id": "chatcmpl-fixed",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "hello back"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
    })
}

fn vk() -> VirtualKey {
    serde_json::from_value(json!({
        "name": "test-key",
        "routeplane_key": RP_KEY,
        "provider_keys": {
            "openai": "test-api-key",
            "anthropic": "test-api-key",
            "gemini": "test-api-key"
        }
    }))
    .expect("virtual key deserializes")
}

fn ctx() -> TenantContext {
    TenantContext {
        tenant_id: TENANT.into(),
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

// ---------------------------------------------------------------------------
// Latency harness — an IN-PROCESS stub provider (no network) so the bench
// measures pure gateway/handler overhead, not localhost round-trips.
// ---------------------------------------------------------------------------

/// Returns a fixed canned response instantly. Registered as "openai".
struct StubProvider;

#[async_trait]
impl Provider for StubProvider {
    fn name(&self) -> &'static str {
        "openai"
    }
    async fn chat_completion(
        &self,
        _request: ChatCompletionRequest,
        _api_key: String,
    ) -> Result<ChatCompletionResponse, ProviderError> {
        Ok(serde_json::from_value(ok_chat_body()).expect("stub response deserializes"))
    }
}

/// AppState backed by the in-process stub (no wiremock, no network).
pub fn build_stub_state() -> Arc<AppState> {
    let mut providers: ProviderRegistry = HashMap::new();
    providers.insert("openai", Arc::new(StubProvider) as Arc<dyn Provider>);
    Arc::new(AppState {
        deadline_config: DeadlineConfig {
            request_deadline: Duration::from_secs(30),
            per_attempt_timeout: Duration::from_secs(30),
        },
        ..AppState::for_tests(providers)
    })
}

fn simple_payload() -> ChatCompletionRequest {
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
        stream: Some(false),
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

/// Drive one buffered request end-to-end and return the raw (undrained)
/// response so a test can inspect response headers (e.g. the additive
/// `x-routeplane-trace-id` the Feedback API references). Body is left for the
/// caller to drain.
pub async fn drive_buffered_resp(state: &Arc<AppState>) -> axum::response::Response {
    chat_completions(
        State(state.clone()),
        Extension(vk()),
        Extension(ctx()),
        Extension(TenantGuardrails(None)),
        HeaderMap::new(),
        routeplane::api_error::OpenAiJson(simple_payload()),
    )
    .await
    .into_response()
}

/// Drive one buffered request end-to-end through the handler (no network).
pub async fn drive_buffered(state: &Arc<AppState>) {
    let resp = drive_buffered_resp(state).await;
    let _ = to_bytes(resp.into_body(), usize::MAX).await;
}

/// p-quantile (0..=100) of a sorted-in-place duration slice (nearest-rank).
pub fn percentile(samples: &mut [Duration], p: f64) -> Duration {
    samples.sort_unstable();
    if samples.is_empty() {
        return Duration::ZERO;
    }
    let rank = ((p / 100.0) * (samples.len() as f64)).ceil() as usize;
    let idx = rank.saturating_sub(1).min(samples.len() - 1);
    samples[idx]
}
