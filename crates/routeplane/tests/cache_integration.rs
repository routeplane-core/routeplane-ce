//! Hermetic integration tests for the rung-0 exact-match response cache
//! (G2.5: PRD-007 behavior, ADR-022 mechanics). The handler is called directly;
//! the only "network" is a localhost wiremock standing in for OpenAI — a HIT is
//! proven by the mock seeing NO additional request. Write-behind inserts are
//! made deterministic with the cache's writer barrier (`flush()`).
//!
//! Covers: AC-1 (hit = byte-identical body, zero upstream calls, header,
//! usage-event fields), AC-2 (key inclusion/exclusion), AC-3 (structural tenant
//! isolation), AC-4 (force_refresh overwrite), AC-5 (regulated bypass +
//! streaming write-bypass), AC-6 (fail-loud validation), AC-7 (no config ⇒ no
//! header), AC-8 posture (Free-tier semantic degradation).

use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Extension;
use routeplane::auth::{TenantContext, TenantGuardrails, VirtualKey};
use routeplane::observability::UsageEvent;
use routeplane::proxy::{chat_completions, AppState, ProviderRegistry};
use routeplane_adapters::openai::OpenAIProvider;
use routeplane_adapters::Provider;
use routeplane_entitlements::{CapabilitySet, Tier};
use routeplane_router::HealthTracker;
use routeplane_types::{ChatCompletionRequest, Message};
use serde_json::json;
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;
use std::time::Duration;
use wiremock::matchers::{body_string_contains, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const CACHE_CFG: &str =
    r#"{"routing":{"targets":[{"provider":"openai"}],"cache":{"mode":"simple"}}}"#;
const REFRESH_CFG: &str = r#"{"routing":{"targets":[{"provider":"openai"}],"cache":{"mode":"simple","force_refresh":true}}}"#;
const SEMANTIC_CFG: &str =
    r#"{"routing":{"targets":[{"provider":"openai"}],"cache":{"mode":"semantic"}}}"#;
const PLAIN_CFG: &str = r#"{"routing":{"targets":[{"provider":"openai"}]}}"#;

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

fn ctx(tenant: &str, tier: Tier) -> TenantContext {
    TenantContext {
        tenant_id: tenant.into(),
        tier,
        capabilities: CapabilitySet::resolve(tier, &BTreeSet::new(), &BTreeSet::new()),
        compliance_frameworks: Vec::new(),
        compliance_mode: routeplane::auth::ComplianceMode::Strict,
    }
}

fn payload(content: &str) -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: "gpt-4o".into(),
        messages: vec![Message {
            role: "user".into(),
            content: content.into(),
            name: None,
            cache_control: None,
            tool_calls: None,
            tool_call_id: None,
            refusal: None,
            reasoning_content: None,
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

fn headers(cfg: &str) -> HeaderMap {
    let mut h = HeaderMap::new();
    h.insert("x-routeplane-config", HeaderValue::from_str(cfg).unwrap());
    h
}

async fn invoke(
    state: Arc<AppState>,
    tenant: &str,
    tier: Tier,
    h: HeaderMap,
    body: ChatCompletionRequest,
) -> Response {
    chat_completions(
        State(state),
        Extension(vk(tenant)),
        Extension(ctx(tenant, tier)),
        Extension(TenantGuardrails(None)),
        h,
        routeplane::api_error::OpenAiJson(body),
    )
    .await
    .into_response()
}

/// (status, x-routeplane-cache header, body bytes) — header read BEFORE the
/// body is consumed.
async fn split(resp: Response) -> (StatusCode, Option<String>, Vec<u8>) {
    let status = resp.status();
    let cache_header = resp
        .headers()
        .get("x-routeplane-cache")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .expect("body readable");
    (status, cache_header, bytes.to_vec())
}

async fn mount_openai_ok(server: &MockServer, content: &str) {
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-cache-test",
            "object": "chat.completion",
            "created": 1_700_000_000,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": content },
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

/// Poll the observability ring (bounded local wait — same pattern as the
/// guardrails integration suite).
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

// --- AC-1: hit = byte-identical body, ZERO upstream calls, header + event -------

#[tokio::test]
async fn ac1_identical_request_hits_byte_identical_with_no_upstream_call() {
    let server = MockServer::start().await;
    mount_openai_ok(&server, "the cached answer").await;
    let state = build_state(&server.uri());

    let first = invoke(
        state.clone(),
        "t_a",
        Tier::Free,
        headers(CACHE_CFG),
        payload("hello cache"),
    )
    .await;
    let (s1, h1, b1) = split(first).await;
    assert_eq!(s1, StatusCode::OK);
    assert_eq!(h1.as_deref(), Some("miss"));

    // Make the write-behind insert visible deterministically.
    state.cache.flush();

    let second = invoke(
        state.clone(),
        "t_a",
        Tier::Free,
        headers(CACHE_CFG),
        payload("hello cache"),
    )
    .await;
    let (s2, h2, b2) = split(second).await;
    assert_eq!(s2, StatusCode::OK);
    assert_eq!(h2.as_deref(), Some("hit"));
    // NFR-6 / FR-9: byte-identical replay.
    assert_eq!(b1, b2);
    // The hit made NO provider call.
    assert_eq!(upstream_calls(&server).await, 1);

    // FR-16: the hit's usage event carries cache fields + the STORED usage.
    let event = wait_for_event(&state, |e| e.cache_hit == Some(true)).await;
    assert_eq!(event.cache_status.as_deref(), Some("hit"));
    assert_eq!(event.cache_namespace.as_deref(), Some("default"));
    assert_eq!(event.total_tokens, 12);
    assert_eq!(event.provider, "(cache)");
    assert!(event.success);
}

// --- AC-2: output-affecting fields miss; `user` is excluded from the key --------

#[tokio::test]
async fn ac2_param_changes_miss_but_user_change_still_hits() {
    let server = MockServer::start().await;
    mount_openai_ok(&server, "ok").await;
    let state = build_state(&server.uri());

    let base = payload("key exclusion proof");
    let _ = invoke(
        state.clone(),
        "t_a",
        Tier::Free,
        headers(CACHE_CFG),
        base.clone(),
    )
    .await;
    state.cache.flush();
    assert_eq!(upstream_calls(&server).await, 1);

    // `user` differs → still a HIT (excluded from the key, FR-5).
    let mut with_user = base.clone();
    with_user.user = Some("end-user-42".into());
    let (s, h, _) = split(
        invoke(
            state.clone(),
            "t_a",
            Tier::Free,
            headers(CACHE_CFG),
            with_user,
        )
        .await,
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(h.as_deref(), Some("hit"));
    assert_eq!(upstream_calls(&server).await, 1);

    // `temperature` differs → MISS (output-affecting, in the key).
    let mut with_temp = base.clone();
    with_temp.temperature = Some(0.9);
    let (s, h, _) = split(
        invoke(
            state.clone(),
            "t_a",
            Tier::Free,
            headers(CACHE_CFG),
            with_temp,
        )
        .await,
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(h.as_deref(), Some("miss"));
    assert_eq!(upstream_calls(&server).await, 2);
}

// --- AC-3: tenant isolation is structural — identical request, no cross-hit -----

#[tokio::test]
async fn ac3_tenant_a_entry_is_invisible_to_tenant_b() {
    let server = MockServer::start().await;
    mount_openai_ok(&server, "ok").await;
    let state = build_state(&server.uri());

    let _ = invoke(
        state.clone(),
        "t_a",
        Tier::Free,
        headers(CACHE_CFG),
        payload("identical request"),
    )
    .await;
    state.cache.flush();

    // Tenant B: identical request, identical (default) namespace string → MISS,
    // upstream called again. Isolation is structural (CacheKey field), not a
    // string-prefix convention.
    let (s, h, _) = split(
        invoke(
            state.clone(),
            "t_b",
            Tier::Free,
            headers(CACHE_CFG),
            payload("identical request"),
        )
        .await,
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(h.as_deref(), Some("miss"));
    assert_eq!(upstream_calls(&server).await, 2);
}

// --- AC-4: force_refresh executes upstream, OVERWRITES, reports `refreshed` -----

#[tokio::test]
async fn ac4_force_refresh_overwrites_the_entry() {
    let server = MockServer::start().await;
    // First upstream answer (consumed once), then a DIFFERENT answer — proving
    // the refresh stored the new body.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-v1", "object": "chat.completion", "created": 1_700_000_000,
            "model": "gpt-4o",
            "choices": [{ "index": 0, "message": { "role": "assistant", "content": "version one" }, "finish_reason": "stop" }],
            "usage": { "prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12 }
        })))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    mount_openai_ok(&server, "version two").await;
    let state = build_state(&server.uri());

    // Seed the entry with "version one".
    let (_, h1, b1) = split(
        invoke(
            state.clone(),
            "t_a",
            Tier::Free,
            headers(CACHE_CFG),
            payload("refresh me"),
        )
        .await,
    )
    .await;
    assert_eq!(h1.as_deref(), Some("miss"));
    state.cache.flush();

    // force_refresh: skips the lookup, calls upstream, overwrites (FR-9/FR-18).
    let (s2, h2, b2) = split(
        invoke(
            state.clone(),
            "t_a",
            Tier::Free,
            headers(REFRESH_CFG),
            payload("refresh me"),
        )
        .await,
    )
    .await;
    assert_eq!(s2, StatusCode::OK);
    assert_eq!(h2.as_deref(), Some("refreshed"));
    assert_eq!(upstream_calls(&server).await, 2);
    assert_ne!(b1, b2, "refresh must produce the new upstream body");
    state.cache.flush();

    // A normal lookup now serves the REFRESHED body.
    let (s3, h3, b3) = split(
        invoke(
            state.clone(),
            "t_a",
            Tier::Free,
            headers(CACHE_CFG),
            payload("refresh me"),
        )
        .await,
    )
    .await;
    assert_eq!(s3, StatusCode::OK);
    assert_eq!(h3.as_deref(), Some("hit"));
    assert_eq!(b3, b2, "hit serves the overwritten entry");
    assert_eq!(upstream_calls(&server).await, 2);
}

// --- AC-5a: classification-positive requests bypass — never read, never written --

#[tokio::test]
async fn ac5_regulated_request_bypasses_and_nothing_is_stored() {
    let server = MockServer::start().await;
    mount_openai_ok(&server, "ok").await;
    let state = build_state(&server.uri());

    // A PAN → classification-positive; NO residency header/default, so the
    // request still routes normally (no 422) — but the cache must never engage.
    let pii = "My PAN is ABCDE1234F, please summarize my account";
    let (s1, h1, _) = split(
        invoke(
            state.clone(),
            "t_a",
            Tier::Free,
            headers(CACHE_CFG),
            payload(pii),
        )
        .await,
    )
    .await;
    assert_eq!(s1, StatusCode::OK);
    assert_eq!(h1.as_deref(), Some("bypass"));
    state.cache.flush();

    // The identical regulated request goes upstream AGAIN (nothing was written,
    // nothing is looked up — FR-10.1 / §6.1).
    let (s2, h2, _) = split(
        invoke(
            state.clone(),
            "t_a",
            Tier::Free,
            headers(CACHE_CFG),
            payload(pii),
        )
        .await,
    )
    .await;
    assert_eq!(s2, StatusCode::OK);
    assert_eq!(h2.as_deref(), Some("bypass"));
    assert_eq!(upstream_calls(&server).await, 2);
}

// --- AC-5b: streaming requests bypass (write side never cached, v1) --------------

#[tokio::test]
async fn ac5_streaming_request_bypasses_and_writes_nothing() {
    let server = MockServer::start().await;
    // SSE mock for stream:true bodies (mounted FIRST; guarded by the body
    // containing "stream":true so the later JSON mock serves buffered calls).
    let sse_body = concat!(
        "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        "data: [DONE]\n\n"
    );
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_string_contains("\"stream\":true"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(sse_body.as_bytes().to_vec(), "text/event-stream"),
        )
        .mount(&server)
        .await;
    mount_openai_ok(&server, "buffered answer").await;
    let state = build_state(&server.uri());

    // Streaming request with a cache config → SSE + `bypass` (FR-10.2).
    let mut streamed = payload("stream me");
    streamed.stream = Some(true);
    let resp = invoke(
        state.clone(),
        "t_a",
        Tier::Free,
        headers(CACHE_CFG),
        streamed,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(
        resp.headers()
            .get("x-routeplane-cache")
            .and_then(|v| v.to_str().ok()),
        Some("bypass")
    );
    assert_eq!(
        resp.headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok()),
        Some("text/event-stream")
    );
    drop(resp);
    state.cache.flush();

    // The buffered twin (stream excluded from the key, FR-5) MISSES — proof the
    // stream wrote nothing.
    let (s, h, _) = split(
        invoke(
            state.clone(),
            "t_a",
            Tier::Free,
            headers(CACHE_CFG),
            payload("stream me"),
        )
        .await,
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(h.as_deref(), Some("miss"));
    assert_eq!(upstream_calls(&server).await, 2);
}

// --- AC-6: fail-loud validation → field-precise 400 ------------------------------

#[tokio::test]
async fn ac6_out_of_bounds_ttl_is_a_field_precise_400() {
    let state = build_state("http://127.0.0.1:9"); // never reached
    let bad = r#"{"routing":{"targets":[{"provider":"openai"}],"cache":{"mode":"simple","ttl_seconds":5}}}"#;
    let resp = invoke(
        state,
        "t_a",
        Tier::Free,
        headers(bad),
        payload("irrelevant"),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["error"]["code"], "invalid_config");
    assert_eq!(v["error"]["param"], "/cache/ttl_seconds");
}

// --- AC-7: no cache config ⇒ NO cache header (absence of signal, FR-2) -----------

#[tokio::test]
async fn ac7_no_cache_config_means_no_cache_header() {
    let server = MockServer::start().await;
    mount_openai_ok(&server, "ok").await;
    let state = build_state(&server.uri());

    // A routing config WITHOUT a cache object.
    let resp = invoke(
        state.clone(),
        "t_a",
        Tier::Free,
        headers(PLAIN_CFG),
        payload("hello"),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    assert!(resp.headers().get("x-routeplane-cache").is_none());
    assert!(resp.headers().get("x-routeplane-cache-degraded").is_none());

    // And no config header at all (pure legacy).
    let resp = invoke(
        state.clone(),
        "t_a",
        Tier::Free,
        HeaderMap::new(),
        payload("hello"),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    assert!(resp.headers().get("x-routeplane-cache").is_none());
}

// --- AC-8 posture: Free-tier semantic mode degrades to simple, LOUDLY ------------

#[tokio::test]
async fn ac8_free_tier_semantic_degrades_to_simple_with_explicit_header() {
    let server = MockServer::start().await;
    mount_openai_ok(&server, "ok").await;
    let state = build_state(&server.uri());

    let first = invoke(
        state.clone(),
        "t_free",
        Tier::Free,
        headers(SEMANTIC_CFG),
        payload("semantic degrade"),
    )
    .await;
    assert_eq!(first.status(), StatusCode::OK);
    assert_eq!(
        first
            .headers()
            .get("x-routeplane-cache-degraded")
            .and_then(|v| v.to_str().ok()),
        Some("semantic_requires_standard")
    );
    assert_eq!(
        first
            .headers()
            .get("x-routeplane-cache")
            .and_then(|v| v.to_str().ok()),
        Some("miss")
    );
    state.cache.flush();

    // Simple (exact) semantics still apply: the identical request HITS.
    let second = invoke(
        state.clone(),
        "t_free",
        Tier::Free,
        headers(SEMANTIC_CFG),
        payload("semantic degrade"),
    )
    .await;
    assert_eq!(
        second
            .headers()
            .get("x-routeplane-cache")
            .and_then(|v| v.to_str().ok()),
        Some("hit")
    );
    assert_eq!(upstream_calls(&server).await, 1);
}

// --- Rung-1 SEMANTIC cache (PRD-007 / ADR-022) ---------------------------------

/// A deterministic embeddings mock: every `/v1/embeddings` call returns the SAME
/// unit vector regardless of input text. That forces cosine similarity = 1.0
/// between any two requests sharing a `SemanticKey`, so a PARAPHRASE (different
/// exact-cache key, hence an exact miss) hits the semantic cache — proving the
/// semantic layer, not exact matching, served the response.
/// ENTERPRISE-ONLY helper (PRD-047): only the gated semantic-HIT tests use it.
#[cfg(feature = "enterprise")]
async fn mount_openai_embeddings_fixed(server: &MockServer) {
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "object": "list",
            "data": [{ "object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3, 0.4] }],
            "model": "text-embedding-3-small",
            "usage": { "prompt_tokens": 3, "total_tokens": 3 }
        })))
        .mount(server)
        .await;
}

async fn chat_calls(server: &MockServer) -> usize {
    server
        .received_requests()
        .await
        .unwrap()
        .iter()
        .filter(|r| r.url.path() == "/v1/chat/completions")
        .count()
}

// ENTERPRISE-ONLY (PRD-047 / ADR-088): a semantic HIT needs the real semantic
// cache, which the CE build compiles out (mode:"semantic" degrades to exact).
#[cfg(feature = "enterprise")]
#[tokio::test]
async fn semantic_paraphrase_hits_without_a_second_chat_call() {
    let server = MockServer::start().await;
    mount_openai_ok(&server, "the semantic answer").await;
    mount_openai_embeddings_fixed(&server).await;
    let state = build_state(&server.uri());

    // First request (Standard tier ⇒ SemanticCache entitled): exact + semantic
    // miss, one chat call, then a write-behind insert into both caches.
    let first = invoke(
        state.clone(),
        "t_sem",
        Tier::Standard,
        headers(SEMANTIC_CFG),
        payload("what is the capital of France"),
    )
    .await;
    let (s1, h1, b1) = split(first).await;
    assert_eq!(s1, StatusCode::OK);
    assert_eq!(h1.as_deref(), Some("miss"));
    // No degradation header for an entitled tenant.
    state.cache.flush();
    assert_eq!(chat_calls(&server).await, 1);

    // A PARAPHRASE: different text ⇒ different exact key ⇒ exact miss. The fixed
    // embedding makes the semantic similarity 1.0 ⇒ semantic HIT, byte-identical
    // body, and NO second chat call.
    let second = invoke(
        state.clone(),
        "t_sem",
        Tier::Standard,
        headers(SEMANTIC_CFG),
        payload("tell me France's capital city"),
    )
    .await;
    let (s2, h2, b2) = split(second).await;
    assert_eq!(s2, StatusCode::OK);
    assert_eq!(h2.as_deref(), Some("semantic-hit"));
    assert_eq!(
        b1, b2,
        "semantic hit replays the stored body byte-identically"
    );
    assert_eq!(
        chat_calls(&server).await,
        1,
        "semantic hit makes NO additional chat call"
    );

    // FR-16: the hit's usage event marks it a semantic-cache hit.
    let event = wait_for_event(&state, |e| e.provider == "(semantic-cache)").await;
    assert_eq!(event.cache_hit, Some(true));
    assert!(event.success);
}

// ENTERPRISE-ONLY (PRD-047): exercises real semantic-cache insert/lookup.
#[cfg(feature = "enterprise")]
#[tokio::test]
async fn semantic_tenant_isolation_is_structural() {
    let server = MockServer::start().await;
    mount_openai_ok(&server, "tenant a answer").await;
    mount_openai_embeddings_fixed(&server).await;
    let state = build_state(&server.uri());

    // tenant_a stores a semantic entry.
    let a = invoke(
        state.clone(),
        "tenant_a",
        Tier::Standard,
        headers(SEMANTIC_CFG),
        payload("shared meaning prompt"),
    )
    .await;
    assert_eq!(a.status(), StatusCode::OK);
    state.cache.flush();
    let after_a = chat_calls(&server).await;

    // tenant_b, identical text + identical fixed embedding, MUST miss (different
    // SemanticKey tenant component) ⇒ a fresh chat call.
    let b = invoke(
        state.clone(),
        "tenant_b",
        Tier::Standard,
        headers(SEMANTIC_CFG),
        payload("shared meaning prompt"),
    )
    .await;
    let (sb, hb, _) = split(b).await;
    assert_eq!(sb, StatusCode::OK);
    assert_eq!(
        hb.as_deref(),
        Some("miss"),
        "cross-tenant semantic hit is impossible by construction"
    );
    assert!(
        chat_calls(&server).await > after_a,
        "tenant_b made its own upstream call"
    );
}

#[tokio::test]
async fn semantic_off_when_no_embedding_provider_falls_back_cleanly() {
    // No `/v1/embeddings` mock is mounted, so the embedding attempt fails; the
    // request must still succeed via the chat path (semantic cache is an
    // optimization, never a dependency) with a normal exact `miss`.
    let server = MockServer::start().await;
    mount_openai_ok(&server, "no-embed answer").await;
    let state = build_state(&server.uri());

    let resp = invoke(
        state.clone(),
        "t_noembed",
        Tier::Standard,
        headers(SEMANTIC_CFG),
        payload("prompt without embeddings backend"),
    )
    .await;
    let (s, h, _) = split(resp).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(h.as_deref(), Some("miss"));
    assert_eq!(chat_calls(&server).await, 1);
}

// --- Client cache bypass: `x-routeplane-cache-control: no-store` (PARITY) --------

/// A `no-store` header forces a per-request bypass: no read, no write, and the
/// `bypass` status — even with a normal cache config and a warm entry.
#[tokio::test]
async fn no_store_header_bypasses_read_and_write() {
    let server = MockServer::start().await;
    mount_openai_ok(&server, "bypass answer").await;
    let state = build_state(&server.uri());

    // 1) Warm the cache with a normal cached request → miss + write.
    let (s1, h1, _) = split(
        invoke(
            state.clone(),
            "t_ns",
            Tier::Free,
            headers(CACHE_CFG),
            payload("no-store proof"),
        )
        .await,
    )
    .await;
    assert_eq!(s1, StatusCode::OK);
    assert_eq!(h1.as_deref(), Some("miss"));
    state.cache.flush();
    assert_eq!(upstream_calls(&server).await, 1);

    // 2) Same request WITH `no-store` → bypass status, and it does NOT read the
    //    warm entry (a fresh upstream call is made).
    let mut h = headers(CACHE_CFG);
    h.insert(
        "x-routeplane-cache-control",
        HeaderValue::from_static("No-Store"), // case-insensitive
    );
    let (s2, h2, _) = split(
        invoke(
            state.clone(),
            "t_ns",
            Tier::Free,
            h,
            payload("no-store proof"),
        )
        .await,
    )
    .await;
    assert_eq!(s2, StatusCode::OK);
    assert_eq!(h2.as_deref(), Some("bypass"));
    // The bypass forced an upstream call (no read) — 2 total now.
    assert_eq!(upstream_calls(&server).await, 2);
    state.cache.flush();

    // 3) A normal request (no header) still HITS the original warm entry: the
    //    bypass neither read nor wrote, so the cache is unchanged.
    let (s3, h3, _) = split(
        invoke(
            state.clone(),
            "t_ns",
            Tier::Free,
            headers(CACHE_CFG),
            payload("no-store proof"),
        )
        .await,
    )
    .await;
    assert_eq!(s3, StatusCode::OK);
    assert_eq!(h3.as_deref(), Some("hit"));
    assert_eq!(upstream_calls(&server).await, 2); // hit ⇒ still 2
}

// --- Cache purge via flush generations (PRD-007 FR-19) --------------------------

async fn call_purge(state: Arc<AppState>, tenant: &str, body: serde_json::Value) -> Response {
    let req: routeplane::cache_api::PurgeRequest =
        serde_json::from_value(body).expect("purge body deserializes");
    routeplane::cache_api::purge(
        State(state),
        Extension(ctx(tenant, Tier::Free)),
        Some(axum::Json(req)),
    )
    .await
    .into_response()
}

/// After a purge, a previously-cached key MISSES, and a fresh entry is stored
/// under the new generation (a subsequent identical request then HITS again).
#[tokio::test]
async fn purge_invalidates_then_repopulates() {
    let server = MockServer::start().await;
    mount_openai_ok(&server, "purge answer").await;
    let state = build_state(&server.uri());

    // Warm: miss + write, then a hit.
    let _ = invoke(
        state.clone(),
        "t_purge",
        Tier::Free,
        headers(CACHE_CFG),
        payload("purge me"),
    )
    .await;
    state.cache.flush();
    let (s, h, _) = split(
        invoke(
            state.clone(),
            "t_purge",
            Tier::Free,
            headers(CACHE_CFG),
            payload("purge me"),
        )
        .await,
    )
    .await;
    assert_eq!((s, h.as_deref()), (StatusCode::OK, Some("hit")));
    assert_eq!(upstream_calls(&server).await, 1);

    // Purge the "default" namespace (the CACHE_CFG namespace).
    let ack = call_purge(state.clone(), "t_purge", json!({ "namespace": "default" })).await;
    let (ps, _, pbody) = split(ack).await;
    assert_eq!(ps, StatusCode::OK);
    let pj: serde_json::Value = serde_json::from_slice(&pbody).unwrap();
    assert_eq!(pj["purged"], json!(true));
    assert_eq!(pj["tenant"], json!("t_purge"));
    assert_eq!(pj["namespace"], json!("default"));
    assert_eq!(pj["generation"], json!(1));

    // The SAME request now MISSES (new-generation key) → a fresh upstream call.
    let (s2, h2, _) = split(
        invoke(
            state.clone(),
            "t_purge",
            Tier::Free,
            headers(CACHE_CFG),
            payload("purge me"),
        )
        .await,
    )
    .await;
    assert_eq!((s2, h2.as_deref()), (StatusCode::OK, Some("miss")));
    assert_eq!(upstream_calls(&server).await, 2);

    // The fresh entry is stored under the new generation and now HITS.
    state.cache.flush();
    let (s3, h3, _) = split(
        invoke(
            state.clone(),
            "t_purge",
            Tier::Free,
            headers(CACHE_CFG),
            payload("purge me"),
        )
        .await,
    )
    .await;
    assert_eq!((s3, h3.as_deref()), (StatusCode::OK, Some("hit")));
    assert_eq!(upstream_calls(&server).await, 2);
}

/// Tenant A's purge does NOT affect tenant B's cached entries.
#[tokio::test]
async fn purge_is_tenant_isolated() {
    let server = MockServer::start().await;
    mount_openai_ok(&server, "iso answer").await;
    let state = build_state(&server.uri());

    // Warm BOTH tenants with the identical request (distinct entries by tenant).
    for t in ["t_a", "t_b"] {
        let _ = invoke(
            state.clone(),
            t,
            Tier::Free,
            headers(CACHE_CFG),
            payload("shared text"),
        )
        .await;
    }
    state.cache.flush();
    assert_eq!(upstream_calls(&server).await, 2);

    // Tenant A purges its "default" namespace.
    let _ = call_purge(state.clone(), "t_a", json!({ "namespace": "default" })).await;

    // Tenant B still HITS (unaffected) — no new upstream call.
    let (sb, hb, _) = split(
        invoke(
            state.clone(),
            "t_b",
            Tier::Free,
            headers(CACHE_CFG),
            payload("shared text"),
        )
        .await,
    )
    .await;
    assert_eq!((sb, hb.as_deref()), (StatusCode::OK, Some("hit")));
    assert_eq!(upstream_calls(&server).await, 2);

    // Tenant A MISSES (its generation was bumped) → a fresh upstream call.
    let (sa, ha, _) = split(
        invoke(
            state.clone(),
            "t_a",
            Tier::Free,
            headers(CACHE_CFG),
            payload("shared text"),
        )
        .await,
    )
    .await;
    assert_eq!((sa, ha.as_deref()), (StatusCode::OK, Some("miss")));
    assert_eq!(upstream_calls(&server).await, 3);
}

/// A no-namespace ("flush-all") purge invalidates a NAMESPACED entry. Before the
/// wildcard fold this was a silent no-op: the bump landed on the reserved "*"
/// scope, which the key-derivation read never consulted, so every entry kept
/// being served (a data-hygiene control reporting success while doing nothing).
/// Now the same request MISSES and re-hits upstream. The MISS cannot pass
/// vacuously — the upstream call count must increment.
#[tokio::test]
async fn flush_all_purge_invalidates_a_namespaced_entry() {
    let server = MockServer::start().await;
    mount_openai_ok(&server, "flush-all answer").await;
    let state = build_state(&server.uri());

    // Warm the "default" namespace (CACHE_CFG's directive default) → hit.
    let _ = invoke(
        state.clone(),
        "t_flushall",
        Tier::Free,
        headers(CACHE_CFG),
        payload("flush all"),
    )
    .await;
    state.cache.flush();
    let (s1, h1, _) = split(
        invoke(
            state.clone(),
            "t_flushall",
            Tier::Free,
            headers(CACHE_CFG),
            payload("flush all"),
        )
        .await,
    )
    .await;
    assert_eq!((s1, h1.as_deref()), (StatusCode::OK, Some("hit")));
    assert_eq!(upstream_calls(&server).await, 1);

    // Flush-all: purge with NO namespace (bumps the tenant-wide wildcard scope).
    // The ack reports a null namespace.
    let ack = call_purge(state.clone(), "t_flushall", json!({})).await;
    let (ps, _, pbody) = split(ack).await;
    assert_eq!(ps, StatusCode::OK);
    let pj: serde_json::Value = serde_json::from_slice(&pbody).unwrap();
    assert_eq!(pj["purged"], json!(true));
    assert_eq!(pj["namespace"], json!(null));

    // The SAME request now MISSES (the wildcard generation folds into the
    // "default" namespace's effective generation) → a fresh upstream call.
    let (s2, h2, _) = split(
        invoke(
            state.clone(),
            "t_flushall",
            Tier::Free,
            headers(CACHE_CFG),
            payload("flush all"),
        )
        .await,
    )
    .await;
    assert_eq!((s2, h2.as_deref()), (StatusCode::OK, Some("miss")));
    assert_eq!(upstream_calls(&server).await, 2);
}

/// FR-5 regression (Finding 1): the exact-match cache key must fold in the
/// output-affecting generation parameters threaded into ChatCompletionRequest
/// AFTER KeyView was first written (tool calling, structured outputs,
/// determinism). Two same-tenant/same-namespace requests with identical
/// messages/model but a different `response_format` MUST NOT collide — else
/// request B is served request A's cached body (a `{"type":"json_object"}`
/// request receiving a cached PLAIN-TEXT completion, breaking a JSON parser).
///
/// This is the end-to-end proof: before the KeyView fix the structured request
/// HIT (returning the plain-text body with ZERO extra upstream calls); now it
/// MISSES and makes its own upstream call. The MISS cannot pass vacuously — the
/// upstream call count must increment.
#[tokio::test]
async fn response_format_variant_does_not_collide_with_plain_cache_entry() {
    let server = MockServer::start().await;
    mount_openai_ok(&server, "plain text answer").await;
    let state = build_state(&server.uri());

    // Warm the "default" namespace with a PLAIN request (no response_format) → miss.
    let (s1, h1, _) = split(
        invoke(
            state.clone(),
            "t_rf",
            Tier::Free,
            headers(CACHE_CFG),
            payload("structured?"),
        )
        .await,
    )
    .await;
    assert_eq!((s1, h1.as_deref()), (StatusCode::OK, Some("miss")));
    state.cache.flush();

    // Sanity: an IDENTICAL plain request now HITS (the cache is live) with no
    // second upstream call.
    let (s2, h2, _) = split(
        invoke(
            state.clone(),
            "t_rf",
            Tier::Free,
            headers(CACHE_CFG),
            payload("structured?"),
        )
        .await,
    )
    .await;
    assert_eq!((s2, h2.as_deref()), (StatusCode::OK, Some("hit")));
    assert_eq!(upstream_calls(&server).await, 1);

    // Same messages/model/chain, differing ONLY in `response_format`. Distinct
    // output ⇒ distinct key ⇒ MISS (not a hit serving the plain-text body).
    let mut structured = payload("structured?");
    structured.response_format = Some(json!({"type": "json_object"}));
    let (s3, h3, _) = split(
        invoke(
            state.clone(),
            "t_rf",
            Tier::Free,
            headers(CACHE_CFG),
            structured,
        )
        .await,
    )
    .await;
    assert_eq!((s3, h3.as_deref()), (StatusCode::OK, Some("miss")));
    // The MISS drove a fresh upstream call (2 total) — proof it did not serve
    // the cached plain-text entry.
    assert_eq!(upstream_calls(&server).await, 2);
}

// --- Purge endpoint auth gating (router-level, real auth_middleware) -------------

const PURGE_KEYS: &str = r#"{"keys":[
    {"name":"k_acme","routeplane_key":"rp_acme","provider_keys":{"openai":"sk-openai"},"tenant_id":"t_acme","tier":"free"}
]}"#;

/// The purge route rides the same auth seam as the other /v1 routes: no
/// `x-routeplane-api-key` ⇒ 401 before the handler runs; a valid key ⇒ 200 ack
/// scoped to the authenticated tenant.
#[tokio::test]
async fn purge_endpoint_is_auth_gated_and_tenant_scoped() {
    use axum::body::Body;
    use axum::http::Request;
    use axum::routing::post;
    use axum::Router;
    use routeplane::auth::{auth_middleware, shared_auth_state, AuthState};
    use tower::ServiceExt;

    let state = build_state("http://127.0.0.1:9");
    let auth = shared_auth_state(AuthState::load_from_json(PURGE_KEYS, "test").expect("keys load"));
    let router = Router::new()
        .route("/v1/cache/purge", post(routeplane::cache_api::purge))
        .layer(axum::middleware::from_fn(auth_middleware))
        .layer(axum::Extension(auth))
        .with_state(state);

    // No key ⇒ 401.
    let unauth = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/cache/purge")
                .body(Body::from(r#"{"namespace":"default"}"#))
                .expect("request builds"),
        )
        .await
        .expect("router responds");
    assert_eq!(unauth.status(), StatusCode::UNAUTHORIZED);

    // Valid key ⇒ 200 ack, tenant resolved from the authenticated context.
    let authed = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/cache/purge")
                .header("x-routeplane-api-key", "rp_acme")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"namespace":"default"}"#))
                .expect("request builds"),
        )
        .await
        .expect("router responds");
    assert_eq!(authed.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(authed.into_body(), 1 << 20)
        .await
        .expect("body readable");
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["purged"], json!(true));
    assert_eq!(v["tenant"], json!("t_acme"));
    assert_eq!(v["namespace"], json!("default"));
    assert_eq!(v["generation"], json!(1));
}
