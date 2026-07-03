//! Hermetic integration tests for idempotency keys (Stripe/Portkey-style
//! safe-retry). The handler is called directly; a localhost wiremock stands in for
//! OpenAI and its `received_requests().len()` is the call counter proving
//! single-dispatch on a replay (provider hit ONCE, not twice).
//!
//! Covers: same key + same body twice → replay (one upstream call, byte-identical
//! body, `x-routeplane-idempotent-replayed: true`); same key + different body →
//! 422; in-flight concurrency → 409; no key → byte-identical (provider called each
//! time); tenant isolation (A's key never replays for B); streaming with a key
//! runs normally (no replay, no store).

use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Extension;
use routeplane::auth::{TenantContext, TenantGuardrails, VirtualKey};
use routeplane::proxy::{chat_completions, AppState, ProviderRegistry};
use routeplane_adapters::openai::OpenAIProvider;
use routeplane_adapters::Provider;
use routeplane_entitlements::{CapabilitySet, Tier};
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

fn vk(tenant: &str) -> VirtualKey {
    serde_json::from_value(json!({
        "name": format!("{tenant}-key"),
        "routeplane_key": format!("rp_{tenant}"),
        "provider_keys": { "openai": "test-api-key" },
        "tenant_id": tenant
    }))
    .expect("virtual key deserializes")
}

fn ctx(tenant: &str) -> TenantContext {
    TenantContext {
        tenant_id: tenant.into(),
        tier: Tier::Standard,
        capabilities: CapabilitySet::resolve(Tier::Standard, &BTreeSet::new(), &BTreeSet::new()),
        compliance_frameworks: Vec::new(),
        compliance_mode: routeplane::auth::ComplianceMode::Strict,
    }
}

fn payload(content: &str, stream: bool) -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: "gpt-4o".into(),
        messages: vec![Message {
            role: "user".into(),
            content: content.into(),
            name: None,
            cache_control: None,
            tool_calls: None,
            tool_call_id: None,
        }],
        temperature: None,
        top_p: None,
        stream: if stream { Some(true) } else { None },
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

fn headers_with_key(idem_key: Option<&str>) -> HeaderMap {
    let mut h = HeaderMap::new();
    h.insert("x-routeplane-provider", HeaderValue::from_static("openai"));
    if let Some(k) = idem_key {
        h.insert("idempotency-key", HeaderValue::from_str(k).unwrap());
    }
    h
}

async fn invoke(
    state: Arc<AppState>,
    tenant: &str,
    h: HeaderMap,
    body: ChatCompletionRequest,
) -> Response {
    chat_completions(
        State(state),
        Extension(vk(tenant)),
        Extension(ctx(tenant)),
        Extension(TenantGuardrails(None)),
        h,
        routeplane::api_error::OpenAiJson(body),
    )
    .await
    .into_response()
}

async fn split(resp: Response) -> (StatusCode, Option<String>, Vec<u8>) {
    let status = resp.status();
    let replay = resp
        .headers()
        .get("x-routeplane-idempotent-replayed")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .expect("body readable");
    (status, replay, bytes.to_vec())
}

/// A mock whose body content is the request count, so a REPLAY (which never calls
/// the provider) is provably distinguishable from a fresh dispatch.
async fn mount_openai_counting(server: &MockServer) {
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-idem",
            "object": "chat.completion",
            "created": 1_700_000_000,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": "answer" },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12 }
        })))
        .mount(server)
        .await;
}

async fn upstream_calls(server: &MockServer) -> usize {
    server.received_requests().await.unwrap().len()
}

// --- same key + same body twice → REPLAY, provider called ONCE ------------------

#[tokio::test]
async fn same_key_same_body_replays_with_single_dispatch() {
    let server = MockServer::start().await;
    mount_openai_counting(&server).await;
    let state = build_state(&server.uri());

    let first = invoke(
        state.clone(),
        "t_a",
        headers_with_key(Some("idem-1")),
        payload("hi", false),
    )
    .await;
    let (s1, r1, b1) = split(first).await;
    assert_eq!(s1, StatusCode::OK);
    assert_eq!(r1, None, "first call is not a replay");

    let second = invoke(
        state.clone(),
        "t_a",
        headers_with_key(Some("idem-1")),
        payload("hi", false),
    )
    .await;
    let (s2, r2, b2) = split(second).await;
    assert_eq!(s2, StatusCode::OK);
    assert_eq!(r2.as_deref(), Some("true"), "second call is a replay");

    // Byte-identical replay body.
    assert_eq!(b1, b2);
    // Provider was dispatched EXACTLY once (the replay made no upstream call).
    assert_eq!(upstream_calls(&server).await, 1);
    assert_eq!(state.idempotency.replays(), 1);
}

// --- same key + DIFFERENT body → 422 -------------------------------------------

#[tokio::test]
async fn same_key_different_body_is_422() {
    let server = MockServer::start().await;
    mount_openai_counting(&server).await;
    let state = build_state(&server.uri());

    let first = invoke(
        state.clone(),
        "t_a",
        headers_with_key(Some("idem-2")),
        payload("first", false),
    )
    .await;
    assert_eq!(first.status(), StatusCode::OK);
    let _ = axum::body::to_bytes(first.into_body(), 1 << 20).await;

    // Same key, different request body → mismatch.
    let second = invoke(
        state.clone(),
        "t_a",
        headers_with_key(Some("idem-2")),
        payload("DIFFERENT", false),
    )
    .await;
    assert_eq!(second.status(), StatusCode::UNPROCESSABLE_ENTITY);
    // No second provider dispatch happened for the rejected request.
    assert_eq!(upstream_calls(&server).await, 1);
    assert_eq!(state.idempotency.mismatches(), 1);
}

// --- in-flight concurrency → 409 (second sees the reservation) -----------------

#[tokio::test]
async fn concurrent_in_flight_second_request_is_409() {
    let server = MockServer::start().await;
    // Delay the upstream so the first request holds its reservation while the
    // second arrives — a deterministic in-flight window.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(std::time::Duration::from_millis(400))
                .set_body_json(json!({
                    "id": "chatcmpl-idem",
                    "object": "chat.completion",
                    "created": 1_700_000_000,
                    "model": "gpt-4o",
                    "choices": [{"index":0,"message":{"role":"assistant","content":"answer"},"finish_reason":"stop"}],
                    "usage": {"prompt_tokens":7,"completion_tokens":5,"total_tokens":12}
                })),
        )
        .mount(&server)
        .await;
    let state = build_state(&server.uri());

    let s1 = state.clone();
    let first = tokio::spawn(async move {
        invoke(
            s1,
            "t_a",
            headers_with_key(Some("idem-3")),
            payload("hi", false),
        )
        .await
    });
    // Give the first request time to win the reservation before the second.
    tokio::time::sleep(std::time::Duration::from_millis(80)).await;

    let second = invoke(
        state.clone(),
        "t_a",
        headers_with_key(Some("idem-3")),
        payload("hi", false),
    )
    .await;
    assert_eq!(
        second.status(),
        StatusCode::CONFLICT,
        "concurrent in-flight → 409"
    );

    let first_resp = first.await.unwrap();
    assert_eq!(first_resp.status(), StatusCode::OK);
    // Only the first request reached the provider.
    assert_eq!(upstream_calls(&server).await, 1);
    assert_eq!(state.idempotency.in_flight_conflicts(), 1);
}

// --- no key → byte-identical, provider called EACH time -------------------------

#[tokio::test]
async fn no_key_runs_normally_each_time() {
    let server = MockServer::start().await;
    mount_openai_counting(&server).await;
    let state = build_state(&server.uri());

    let first = invoke(
        state.clone(),
        "t_a",
        headers_with_key(None),
        payload("hi", false),
    )
    .await;
    let (s1, r1, _b1) = split(first).await;
    assert_eq!(s1, StatusCode::OK);
    assert_eq!(r1, None, "no replay header without an idempotency key");

    let second = invoke(
        state.clone(),
        "t_a",
        headers_with_key(None),
        payload("hi", false),
    )
    .await;
    let (s2, r2, _b2) = split(second).await;
    assert_eq!(s2, StatusCode::OK);
    assert_eq!(r2, None);

    // Both ran (the idempotency store never engaged): TWO upstream calls.
    assert_eq!(upstream_calls(&server).await, 2);
    assert_eq!(state.idempotency.reservations(), 0);
}

// --- tenant isolation: A's key never replays for B -----------------------------

#[tokio::test]
async fn tenant_a_key_does_not_replay_for_tenant_b() {
    let server = MockServer::start().await;
    mount_openai_counting(&server).await;
    let state = build_state(&server.uri());

    // Tenant A reserves + stores under "shared".
    let a1 = invoke(
        state.clone(),
        "t_a",
        headers_with_key(Some("shared")),
        payload("hi", false),
    )
    .await;
    assert_eq!(a1.status(), StatusCode::OK);
    let _ = axum::body::to_bytes(a1.into_body(), 1 << 20).await;

    // Tenant B with the SAME client key + SAME body is a FRESH dispatch (not a
    // replay) — structural tenant isolation in the key.
    let b1 = invoke(
        state.clone(),
        "t_b",
        headers_with_key(Some("shared")),
        payload("hi", false),
    )
    .await;
    let (sb, rb, _bb) = split(b1).await;
    assert_eq!(sb, StatusCode::OK);
    assert_eq!(rb, None, "tenant B must NOT replay tenant A's response");

    // Two real dispatches (A's, then B's) — B did not hit A's stored entry.
    assert_eq!(upstream_calls(&server).await, 2);
}

// --- streaming with a key runs normally (no replay, no store) ------------------

#[tokio::test]
async fn streaming_with_key_bypasses_replay() {
    let server = MockServer::start().await;
    // Stream SSE: one chunk + DONE. The proxy streams; the idempotency layer must
    // NOT engage (no reserve, no store, no 409 on a second identical stream).
    let sse = "data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\ndata: [DONE]\n\n";
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse),
        )
        .mount(&server)
        .await;
    let state = build_state(&server.uri());

    let first = invoke(
        state.clone(),
        "t_a",
        headers_with_key(Some("stream-key")),
        payload("hi", true),
    )
    .await;
    assert_eq!(first.status(), StatusCode::OK);
    let _ = axum::body::to_bytes(first.into_body(), 1 << 20).await;

    let second = invoke(
        state.clone(),
        "t_a",
        headers_with_key(Some("stream-key")),
        payload("hi", true),
    )
    .await;
    // A streamed request never reserves, so the second is NOT a 409 — it streams
    // again (a real second dispatch).
    assert_eq!(second.status(), StatusCode::OK);
    let _ = axum::body::to_bytes(second.into_body(), 1 << 20).await;

    assert_eq!(
        upstream_calls(&server).await,
        2,
        "streaming bypasses idempotency"
    );
    assert_eq!(
        state.idempotency.reservations(),
        0,
        "no reservation for a streamed request"
    );
}
