//! Finding: a client that disconnects mid-stream skipped ALL post-stream
//! accounting — the budget `settle` (the only place a streamed request's
//! token/cost budget is debited), the usage event (observability/metrics), and
//! the sovereign-audit decision — because they live in the SSE generator body
//! AFTER the chunk loop and never run when the generator is cancelled at a
//! `yield`. These hermetic tests drive `chat_completions` directly against a
//! scripted in-process streaming provider (no socket) and prove two things: an
//! early client disconnect STILL settles the observed spend and emits the usage
//! event (the fail-safe `Drop` fires), and a normally-completed stream settles
//! EXACTLY ONCE (the guard is disarmed, so it never double-charges). Both read a
//! follow-up buffered request's advisory `x-ratelimit-remaining-tokens` header,
//! which reflects the shared token bucket after every settle.

use async_trait::async_trait;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Extension;
use futures::StreamExt;
use routeplane::auth::{TenantContext, TenantGuardrails, VirtualKey};
use routeplane::observability::UsageEvent;
use routeplane::proxy::{chat_completions, AppState, ProviderRegistry};
use routeplane_adapters::{ChunkStream, Provider, ProviderError};
use routeplane_entitlements::{CapabilitySet, Tier};
use routeplane_limits::{KeyLimits, KeyLimitsInput, LimitRegistry};
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

/// A provider that streams a fixed script of chunks and answers buffered calls
/// with a fixed response. No network — deterministic, in-process.
struct ScriptedStream {
    chunks: Vec<ChatCompletionChunk>,
    buffered: ChatCompletionResponse,
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
        Ok(self.buffered.clone())
    }
    async fn chat_completion_stream(
        &self,
        _r: ChatCompletionRequest,
        _k: String,
    ) -> Result<ChunkStream, ProviderError> {
        let chunks = self.chunks.clone();
        Ok(Box::pin(futures::stream::iter(chunks.into_iter().map(Ok))))
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

fn buffered_response() -> ChatCompletionResponse {
    serde_json::from_value(json!({
        "id": "chatcmpl-buffered",
        "object": "chat.completion",
        "created": 1_700_000_000u64,
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": "hi there" },
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12 }
    }))
    .expect("response deserializes")
}

fn build_state(chunks: Vec<ChatCompletionChunk>) -> Arc<AppState> {
    // Generous request cap, a 1000-token/min cap so the advisory
    // `x-ratelimit-remaining-tokens` header on a buffered call reflects settles.
    let limits: KeyLimits = serde_json::from_value(json!({
        "key": { "rate": { "requests_per_min": 100, "tokens_per_min": 1000 } }
    }))
    .expect("KeyLimits deserializes");
    let registry = LimitRegistry::build(vec![KeyLimitsInput {
        routeplane_key: RP_KEY.into(),
        tenant_id: TENANT.into(),
        limits,
    }]);
    let mut providers: ProviderRegistry = HashMap::new();
    providers.insert(
        "openai",
        Arc::new(ScriptedStream {
            chunks,
            buffered: buffered_response(),
        }) as Arc<dyn Provider>,
    );
    Arc::new(AppState {
        health: HealthTracker::new(["openai"]),
        limits: registry,
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

fn request(stream: bool) -> ChatCompletionRequest {
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
        stream: Some(stream),
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

async fn invoke(state: &Arc<AppState>, stream: bool) -> Response {
    chat_completions(
        State(state.clone()),
        Extension(vk()),
        Extension(ctx()),
        Extension(TenantGuardrails(None)),
        HeaderMap::new(),
        routeplane::api_error::OpenAiJson(request(stream)),
    )
    .await
    .into_response()
}

/// Poll the in-memory ring until `pred` matches (the observability pipeline is a
/// bounded channel + background drain, so arrival is asynchronous but local).
async fn wait_for_event<F: Fn(&UsageEvent) -> bool>(state: &AppState, pred: F) -> UsageEvent {
    for _ in 0..100 {
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

/// A usage of `{prompt:7, completion:5, total:12}` for a chunk's `usage` field.
fn usage_12() -> serde_json::Value {
    json!({ "prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12 })
}

// --- 1. early disconnect still settles + emits ----------------------------------

#[tokio::test]
async fn client_disconnect_midstream_settles_spend_and_emits_usage() {
    // Usage rides in the FIRST chunk so an early abort has a non-zero spend to
    // settle. A second chunk guarantees the generator is suspended MID-loop (not
    // at `[DONE]`) when we drop, so the fail-safe `Drop` — not the inline path —
    // does the accounting.
    let state = build_state(vec![
        chunk("Hello ", Some(usage_12())),
        chunk("world", None),
    ]);

    let resp = invoke(&state, true).await;
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(
        resp.headers().get("content-type").unwrap(),
        "text/event-stream"
    );

    // Read exactly ONE frame (the first chunk), then DROP the body — the client
    // hung up before the stream finished.
    let mut body = resp.into_body().into_data_stream();
    let first = body.next().await;
    assert!(first.is_some(), "the first SSE frame should be delivered");
    drop(body); // ← client disconnect: cancels the generator at its `yield`.

    // The fail-safe fired: a usage event landed (blind-spot closed) with the
    // observed 12 tokens. Without the fix NO event is emitted and this panics.
    let event = wait_for_event(&state, |e| e.provider == "openai" && e.success).await;
    assert_eq!(event.total_tokens, 12, "abort event carries observed usage");

    // And the spend was settled once: a buffered follow-up (settling its own 12)
    // sees 1000 - 12 (abort) - 12 (this call) = 976 remaining. Without the fix
    // the abort settled nothing → 1000 - 12 = 988.
    let follow = invoke(&state, false).await;
    assert_eq!(follow.status(), StatusCode::OK);
    assert_eq!(
        follow
            .headers()
            .get("x-ratelimit-remaining-tokens")
            .unwrap(),
        "976",
        "the aborted stream must have settled its observed spend exactly once"
    );
}

// --- 2. normal completion settles exactly once ----------------------------------

#[tokio::test]
async fn completed_stream_settles_exactly_once() {
    // Usage in the FINAL chunk (the realistic case). Drive the stream fully to
    // completion so the inline post-`[DONE]` accounting runs and the guard is
    // disarmed — it must NOT also fire from Drop (that would double-charge).
    let state = build_state(vec![
        chunk("Hello ", None),
        chunk("world", Some(usage_12())),
    ]);

    let resp = invoke(&state, true).await;
    assert_eq!(resp.status(), StatusCode::OK);

    // Drain to None — the final poll past `[DONE]` runs the inline settle.
    let mut body = resp.into_body().into_data_stream();
    while let Some(frame) = body.next().await {
        frame.expect("frame");
    }

    // A buffered follow-up (settling its own 12) sees 1000 - 12 (stream) - 12 =
    // 976. A DOUBLE settle (guard + inline) would leave 964 — this pins exactly-once.
    let follow = invoke(&state, false).await;
    assert_eq!(follow.status(), StatusCode::OK);
    assert_eq!(
        follow
            .headers()
            .get("x-ratelimit-remaining-tokens")
            .unwrap(),
        "976",
        "a completed stream settles exactly once (disarmed guard never double-charges)"
    );
}
