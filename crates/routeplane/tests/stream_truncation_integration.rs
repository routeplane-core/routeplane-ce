//! Streaming truth: a mid-stream provider error or an idle upstream must NOT be
//! masked as a clean end. Before the fix, the SSE forward loop `break`-ed on
//! `Err`, then unconditionally yielded `data: [DONE]` and recorded a success
//! usage event (and an `Ok` audit outcome) — silent truncation, entered into the
//! record as a completed request.
//!
//! These hermetic tests drive `chat_completions` against scripted in-process
//! providers (the `stream_abort_integration.rs` pattern — no sockets) and prove:
//! a stream that errors mid-flight ends with an OpenAI-style `data: {"error":…}`
//! frame carrying `routeplane_stream_truncated`, emits NO `[DONE]`, and records
//! a usage event with `success == false` while keeping the real observed token
//! spend; a hung upstream is bounded by `ROUTEPLANE_STREAM_IDLE_TIMEOUT_MS`
//! (set to 200ms here — each tests/*.rs file is its own process, so the env is
//! test-local); and a cleanly-completed stream still ends with `[DONE]` and a
//! success event (no regression on the happy path).

use async_trait::async_trait;
use axum::body::to_bytes;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use axum::Extension;
use futures::StreamExt as _;
use routeplane::auth::{TenantContext, TenantGuardrails, VirtualKey};
use routeplane::observability::UsageEvent;
use routeplane::proxy::{chat_completions, AppState, ProviderRegistry};
use routeplane_adapters::{ChunkStream, Provider, ProviderError};
use routeplane_entitlements::{CapabilitySet, Tier};
use routeplane_router::HealthTracker;
use routeplane_types::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, Message,
};
use serde_json::json;
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;
use std::time::Duration;

const RP_KEY: &str = "rp_test";
const TENANT: &str = "t_test";

/// What the scripted stream does after its chunks are exhausted.
#[derive(Clone, Copy)]
enum Tail {
    /// End cleanly (provider stream naturally ends).
    Clean,
    /// Surface a mid-stream provider error.
    Error,
    /// Never produce another item (hung upstream — exercises the idle bound).
    Hang,
}

struct ScriptedStream {
    chunks: Vec<ChatCompletionChunk>,
    tail: Tail,
}

#[async_trait]
impl Provider for ScriptedStream {
    fn name(&self) -> &'static str {
        "openai"
    }
    async fn chat_completion(
        &self,
        _r: ChatCompletionRequest,
        _k: String,
    ) -> Result<ChatCompletionResponse, ProviderError> {
        unreachable!("these tests only stream")
    }
    async fn chat_completion_stream(
        &self,
        _r: ChatCompletionRequest,
        _k: String,
    ) -> Result<ChunkStream, ProviderError> {
        let head = futures::stream::iter(self.chunks.clone().into_iter().map(Ok));
        let s: ChunkStream = match self.tail {
            Tail::Clean => Box::pin(head),
            Tail::Error => Box::pin(head.chain(futures::stream::iter(std::iter::once(Err(
                ProviderError::Timeout {
                    provider: "openai".to_string(),
                    detail: "connection reset mid-body".to_string(),
                },
            ))))),
            Tail::Hang => Box::pin(head.chain(futures::stream::pending())),
        };
        Ok(s)
    }
}

fn chunk(content: &str, usage: Option<serde_json::Value>) -> ChatCompletionChunk {
    let mut v = json!({
        "id": "chatcmpl-stream",
        "object": "chat.completion.chunk",
        "created": 1_700_000_000u64,
        "model": "gpt-4o",
        "choices": [{ "index": 0, "delta": { "content": content } }],
    });
    if let Some(u) = usage {
        v["usage"] = u;
    }
    serde_json::from_value(v).expect("chunk deserializes")
}

fn build_state(chunks: Vec<ChatCompletionChunk>, tail: Tail) -> Arc<AppState> {
    let mut providers: ProviderRegistry = HashMap::new();
    providers.insert(
        "openai",
        Arc::new(ScriptedStream { chunks, tail }) as Arc<dyn Provider>,
    );
    Arc::new(AppState {
        health: HealthTracker::new(["openai"]),
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
    TenantContext {
        tenant_id: TENANT.into(),
        tier: Tier::Free,
        capabilities: CapabilitySet::resolve(Tier::Free, &BTreeSet::new(), &BTreeSet::new()),
        compliance_frameworks: Vec::new(),
        compliance_mode: routeplane::auth::ComplianceMode::Strict,
    }
}

fn request() -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: "gpt-4o".into(),
        messages: vec![Message {
            role: "user".into(),
            content: "hello".into(),
            name: None,
            cache_control: None,
            tool_calls: None,
            tool_call_id: None,
            refusal: None,
            reasoning_content: None,
        }],
        stream: Some(true),
        ..Default::default()
    }
}

async fn stream_body(state: &Arc<AppState>) -> String {
    let resp: Response = chat_completions(
        State(state.clone()),
        Extension(vk()),
        Extension(ctx()),
        Extension(TenantGuardrails(None)),
        HeaderMap::new(),
        routeplane::api_error::OpenAiJson(request()),
    )
    .await
    .into_response();
    assert_eq!(resp.status(), axum::http::StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("drain SSE body");
    String::from_utf8(bytes.to_vec()).expect("SSE body is utf-8")
}

async fn wait_for_event<F: Fn(&UsageEvent) -> bool>(state: &AppState, pred: F) -> UsageEvent {
    for _ in 0..300 {
        if let Some(e) = state
            .observability_engine
            .get_recent_events()
            .into_iter()
            .find(|e| pred(e))
        {
            return e;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("expected usage event did not arrive");
}

fn usage_json() -> serde_json::Value {
    json!({ "prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12 })
}

/// Mid-stream provider error → error terminal frame, no [DONE], failed event
/// that still carries the observed token spend.
#[tokio::test]
async fn midstream_error_yields_error_frame_not_done() {
    let state = build_state(
        vec![chunk("hel", None), chunk("lo", Some(usage_json()))],
        Tail::Error,
    );
    let body = stream_body(&state).await;

    assert!(
        !body.contains("data: [DONE]"),
        "a truncated stream must NOT be sealed with [DONE]; body:\n{body}"
    );
    assert!(
        body.contains("routeplane_stream_truncated"),
        "the terminal frame must carry the truncation code; body:\n{body}"
    );
    // The frame is an OpenAI-style error event on a data: line.
    let frame_line = body
        .lines()
        .rev()
        .find(|l| l.starts_with("data: "))
        .expect("a terminal data: line exists");
    let v: serde_json::Value =
        serde_json::from_str(frame_line.trim_start_matches("data: ")).expect("frame is JSON");
    assert_eq!(v["error"]["type"], "stream_error");

    let e = wait_for_event(&state, |e| !e.success).await;
    assert_eq!(e.total_tokens, 12, "observed spend is kept, not zeroed");
    assert!(
        e.error
            .as_deref()
            .unwrap_or("")
            .contains("stream truncated"),
        "event error names the truncation: {:?}",
        e.error
    );
}

/// Hung upstream → the ROUTEPLANE_STREAM_IDLE_TIMEOUT_MS bound fires (200ms
/// here) and the stream ends with the truncation frame instead of hanging.
#[tokio::test]
async fn idle_upstream_is_bounded_and_truncates() {
    std::env::set_var("ROUTEPLANE_STREAM_IDLE_TIMEOUT_MS", "200");
    let state = build_state(vec![chunk("partial", None)], Tail::Hang);
    let body = tokio::time::timeout(Duration::from_secs(10), stream_body(&state))
        .await
        .expect("the idle bound must terminate a hung upstream well before 10s");

    assert!(
        !body.contains("data: [DONE]"),
        "idle timeout is not a clean end"
    );
    assert!(body.contains("routeplane_stream_truncated"));
    let e = wait_for_event(&state, |e| !e.success).await;
    assert!(e
        .error
        .as_deref()
        .unwrap_or("")
        .contains("stream truncated"));
}

/// Happy path unchanged: clean provider end → [DONE], no error frame, success
/// event (regression guard for the fix itself).
#[tokio::test]
async fn clean_stream_still_ends_with_done() {
    let state = build_state(
        vec![chunk("hel", None), chunk("lo", Some(usage_json()))],
        Tail::Clean,
    );
    let body = stream_body(&state).await;

    assert!(
        body.contains("data: [DONE]"),
        "clean end keeps the [DONE] seal"
    );
    assert!(!body.contains("routeplane_stream_truncated"));
    let e = wait_for_event(&state, |e| e.success).await;
    assert_eq!(e.total_tokens, 12);
    assert!(e.error.is_none());
}
