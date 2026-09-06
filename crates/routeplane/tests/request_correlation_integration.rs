//! The retained row joins the actual gateway response,
//! not the upstream completion, caller metadata, or distributed trace. Runs in
//! the CE build; all providers are in-process synthetic fixtures.

use async_trait::async_trait;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Extension;
use routeplane::auth::{shared_auth_state, AuthState, TenantContext, TenantGuardrails, VirtualKey};
use routeplane::logs_api::list_logs;
use routeplane::observability::{ObservabilityEngine, UsageEvent};
use routeplane::proxy::{chat_completions, AppState, ProviderRegistry};
use routeplane_adapters::{Provider, ProviderError};
use routeplane_entitlements::{CapabilitySet, Tier};
use routeplane_types::{ChatCompletionRequest, ChatCompletionResponse, TenantId};
use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Duration;

struct SyntheticProvider {
    name: &'static str,
    fail: bool,
}

#[async_trait]
impl Provider for SyntheticProvider {
    fn name(&self) -> &'static str {
        self.name
    }

    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
        _api_key: String,
    ) -> Result<ChatCompletionResponse, ProviderError> {
        if self.name == "azure_openai" {
            return Err(ProviderError::azure_deployment_unmapped(
                self.name,
                &request.model,
            ));
        }
        if self.fail {
            return Err(ProviderError::timeout(self.name, "synthetic timeout"));
        }
        Ok(serde_json::from_value(json!({
            "id": "chatcmpl-upstream-not-request-id", "object": "chat.completion",
            "created": 1, "model": "gpt-4o",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
        }))
        .expect("synthetic completion"))
    }

    async fn embeddings(
        &self,
        request: routeplane_types::EmbeddingRequest,
        _api_key: String,
    ) -> Result<routeplane_types::EmbeddingResponse, ProviderError> {
        if self.name == "azure_openai" {
            return Err(ProviderError::azure_deployment_unmapped(
                self.name,
                &request.model,
            ));
        }
        if self.fail {
            return Err(ProviderError::embeddings_not_supported(self.name));
        }
        Ok(serde_json::from_value(json!({
            "object": "list", "model": "text-embedding-3-small",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
            "usage": {"prompt_tokens": 2, "total_tokens": 2}
        }))
        .expect("synthetic embedding"))
    }
}

fn state() -> Arc<AppState> {
    let mut providers = ProviderRegistry::new();
    for (name, fail) in [
        ("openai", false),
        ("anthropic", true),
        ("azure_openai", true),
    ] {
        providers.insert(
            name,
            Arc::new(SyntheticProvider { name, fail }) as Arc<dyn Provider>,
        );
    }
    Arc::new(AppState {
        observability_engine: ObservabilityEngine::new(vec![
            TenantId::new("t_acme").expect("canonical fixture tenant"),
            TenantId::new("t_other").expect("canonical fixture tenant"),
        ]),
        cache: routeplane_cache::ExactCache::new(
            routeplane_cache::DEFAULT_BUDGET_BYTES,
            vec![TenantId::new("t_acme").expect("canonical fixture tenant")],
        ),
        cache_flush: routeplane_cache::FlushRegistry::new(vec![
            TenantId::new("t_acme").expect("canonical fixture tenant")
        ]),
        ..AppState::for_tests(providers)
    })
}

fn context(tenant: &str) -> TenantContext {
    TenantContext {
        tenant_id: tenant.into(),
        resource_tenant_id: Some(TenantId::new(tenant).expect("canonical fixture tenant")),
        tier: Tier::Free,
        capabilities: CapabilitySet::resolve(Tier::Free, &BTreeSet::new(), &BTreeSet::new()),
        compliance_frameworks: Vec::new(),
        compliance_mode: routeplane::auth::ComplianceMode::Strict,
    }
}

async fn invoke(
    state: Arc<AppState>,
    stream: bool,
    config: Option<&str>,
    refuse: bool,
) -> Response {
    let key: VirtualKey = serde_json::from_value(json!({
        "name": "shared-key-name", "tenant_id": "t_acme", "routeplane_key": "rp_test",
        "provider_keys": {"openai": "synthetic", "anthropic": "synthetic", "azure_openai": "synthetic"}
    }))
    .expect("synthetic key");
    let mut headers = HeaderMap::new();
    headers.insert("x-routeplane-provider", HeaderValue::from_static("openai"));
    headers.insert(
        "x-routeplane-request-id",
        HeaderValue::from_static("req_caller_spoof"),
    );
    headers.insert(
        "x-routeplane-trace-id",
        HeaderValue::from_static("req_trace_spoof"),
    );
    headers.insert(
        "traceparent",
        HeaderValue::from_static("00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"),
    );
    if let Some(config) = config {
        headers.insert(
            "x-routeplane-config",
            HeaderValue::from_str(config).unwrap(),
        );
    }
    if refuse {
        headers.insert("x-routeplane-residency", HeaderValue::from_static("IN"));
    }
    // Existing residency semantics lock a requested region only when the raw
    // input is classified as personal data. Use a reserved synthetic address.
    let content = if refuse {
        "synthetic.person@example.invalid"
    } else {
        "hello"
    };
    let body = serde_json::from_value(json!({
        "model": "gpt-4o", "messages": [{"role": "user", "content": content}],
        "stream": stream, "metadata": {"request_id": "req_metadata_spoof"}
    }))
    .expect("synthetic request");
    let tenant = context("t_acme");
    chat_completions(
        State(state),
        Extension(key),
        Extension(tenant),
        Extension(TenantGuardrails(None)),
        headers,
        routeplane::api_error::OpenAiJson(body),
    )
    .await
    .into_response()
}

async fn consume(response: Response) -> (StatusCode, String, Vec<u8>) {
    let status = response.status();
    let request_id = response.headers()["x-routeplane-request-id"]
        .to_str()
        .unwrap()
        .to_owned();
    assert_eq!(
        response.headers()["x-routeplane-trace-id"],
        request_id.as_str()
    );
    assert!(request_id.starts_with("req_"));
    assert_eq!(request_id.len(), 36);
    assert_ne!(request_id, "req_caller_spoof");
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
        .await
        .unwrap();
    (status, request_id, bytes.to_vec())
}

async fn logs(state: &Arc<AppState>, tenant: &str, expected: usize) -> Vec<Value> {
    let auth = shared_auth_state(
        AuthState::load_from_json(
            r#"{"keys":[
                {"name":"shared-key-name","tenant_id":"t_acme","routeplane_key":"rp_test","provider_keys":{"openai":"synthetic"},"tier":"free"},
                {"name":"other-key","tenant_id":"t_other","routeplane_key":"rp_other","provider_keys":{"openai":"synthetic"},"tier":"free"}
            ]}"#,
            "test",
        )
        .expect("synthetic registry"),
    );
    let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
    loop {
        let response = list_logs(
            State(state.clone()),
            Extension(auth.clone()),
            Extension(context(tenant)),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
            .await
            .unwrap();
        let value: Value = serde_json::from_slice(&bytes).unwrap();
        let rows = value["events"].as_array().unwrap().clone();
        if rows.len() == expected {
            return rows;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "unexpected rows: {rows:?}"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

#[tokio::test]
async fn buffered_and_sse_rows_match_response_not_caller_or_upstream_ids() {
    for stream in [false, true] {
        let state = state();
        let (status, request_id, body) =
            consume(invoke(state.clone(), stream, None, false).await).await;
        assert_eq!(status, StatusCode::OK);
        if stream {
            assert!(String::from_utf8_lossy(&body).contains("data: [DONE]"));
        } else {
            let body: Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(body["id"], "chatcmpl-upstream-not-request-id");
        }
        let rows = logs(&state, "t_acme", 1).await;
        assert_eq!(rows[0]["request_id"], request_id);
        assert!(rows[0]["id"].as_str().unwrap().starts_with("log_"));
        assert_eq!(rows[0]["outcome"], "success");
        assert!(!serde_json::to_string(&rows).unwrap().contains("spoof"));
        assert_eq!(logs(&state, "t_acme", 1).await, rows, "stable snapshot IDs");
        assert!(
            logs(&state, "t_other", 0).await.is_empty(),
            "B cannot read A"
        );
    }
}

#[tokio::test]
async fn fallback_and_terminal_failure_keep_every_existing_attempt_correlated() {
    for stream in [false, true] {
        for fallback in [false, true] {
            let state = state();
            let config = if fallback {
                r#"{"routing":{"targets":[{"provider":"anthropic"},{"provider":"openai"}]}}"#
            } else {
                r#"{"routing":{"targets":[{"provider":"anthropic"}]}}"#
            };
            let (status, id, _) =
                consume(invoke(state.clone(), stream, Some(config), false).await).await;
            assert_eq!(status.is_success(), fallback);
            let rows = logs(&state, "t_acme", if fallback { 2 } else { 1 }).await;
            for row in &rows {
                assert_eq!(row["request_id"], id);
            }
            assert!(rows.iter().any(|row| row["outcome"] == "error"));
            if fallback {
                assert_ne!(
                    rows[0]["id"], rows[1]["id"],
                    "attempts are not deduplicated"
                );
            }
        }
    }
}

#[tokio::test]
async fn cache_hit_has_its_own_response_identity() {
    let state = state();
    let config = r#"{"routing":{"targets":[{"provider":"openai"}],"cache":{"mode":"simple"}}}"#;
    let (_, first_id, _) = consume(invoke(state.clone(), false, Some(config), false).await).await;
    state.cache.flush();
    let response = invoke(state.clone(), false, Some(config), false).await;
    assert_eq!(response.headers()["x-routeplane-cache"], "hit");
    let (_, hit_id, _) = consume(response).await;
    assert_ne!(first_id, hit_id);
    let rows = logs(&state, "t_acme", 2).await;
    let hit = rows
        .iter()
        .find(|row| row["cache_status"] == "hit")
        .unwrap();
    assert_eq!(hit["request_id"], hit_id);
    assert!(rows.iter().any(|row| row["request_id"] == first_id));
}

#[tokio::test]
async fn azure_unmapped_buffered_and_streaming_refusals_keep_correlation() {
    for stream in [false, true] {
        let state = state();
        let config = r#"{"routing":{"targets":[{"provider":"azure_openai"}]}}"#;
        let response = invoke(state.clone(), stream, Some(config), false).await;
        assert!(!response.headers().contains_key("x-routeplane-provider"));
        let (status, id, body) = consume(response).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        let body: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["error"]["code"], "upstream_invalid_request");
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert!(body["error"]["param"].is_null());
        let rows = logs(&state, "t_acme", 1).await;
        assert_eq!(rows[0]["request_id"], id);
        assert_eq!(rows[0]["provider"], "azure_openai");
        assert_eq!(rows[0]["outcome"], "error");
        assert!(logs(&state, "t_other", 0).await.is_empty());
    }
}

#[tokio::test]
async fn embeddings_success_and_recorded_errors_correlate() {
    for provider in ["openai", "anthropic", "azure_openai"] {
        let state = state();
        let key = serde_json::from_value(json!({
            "name": "shared-key-name", "tenant_id": "t_acme", "routeplane_key": "rp_test",
            "provider_keys": {"openai": "synthetic", "anthropic": "synthetic", "azure_openai": "synthetic"}
        }))
        .unwrap();
        let mut headers = HeaderMap::new();
        headers.insert("x-routeplane-provider", HeaderValue::from_static(provider));
        let payload = serde_json::from_value(json!({
            "model": "text-embedding-3-small", "input": "hello"
        }))
        .unwrap();
        let response = routeplane::embeddings::embeddings(
            State(state.clone()),
            Extension(key),
            Extension(context("t_acme")),
            headers,
            routeplane::api_error::OpenAiJson(payload),
        )
        .await;
        if provider == "azure_openai" {
            assert!(!response.headers().contains_key("x-routeplane-provider"));
        }
        let (status, id, body) = consume(response).await;
        assert_eq!(status.is_success(), provider == "openai");
        if provider == "azure_openai" {
            // Preserve CE's existing all-failed envelope; correlation is additive.
            assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
            let body: Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(body["error"]["code"], "upstream_error");
            assert_eq!(body["error"]["type"], "api_error");
            assert!(body["error"]["param"].is_null());
        }
        let rows = logs(&state, "t_acme", 1).await;
        assert_eq!(rows[0]["request_id"], id);
        assert_eq!(rows[0]["provider"], provider);
        assert!(logs(&state, "t_other", 0).await.is_empty());
    }
}

#[tokio::test]
async fn recorded_residency_refusal_has_the_response_identity() {
    let state = state();
    let (status, id, _) = consume(invoke(state.clone(), false, None, true).await).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    let rows = logs(&state, "t_acme", 1).await;
    assert_eq!(rows[0]["request_id"], id);
    assert_eq!(rows[0]["outcome"], "blocked");
}

#[tokio::test]
async fn concurrent_requests_have_distinct_ids_and_unchanged_cardinality() {
    let state = state();
    let requests = (0..16).map(|_| {
        let state = state.clone();
        async move { consume(invoke(state, false, None, false).await).await.1 }
    });
    let ids: BTreeSet<_> = futures::future::join_all(requests)
        .await
        .into_iter()
        .collect();
    assert_eq!(ids.len(), 16);
    let rows = logs(&state, "t_acme", 16).await;
    let retained: BTreeSet<_> = rows
        .iter()
        .map(|row| row["request_id"].as_str().unwrap())
        .collect();
    assert_eq!(retained, ids.iter().map(String::as_str).collect());
}

#[tokio::test]
async fn correlation_and_same_key_name_never_select_another_tenants_rows() {
    let state = state();
    let (_, request_id, _) = consume(invoke(state.clone(), false, None, false).await).await;
    let a = logs(&state, "t_acme", 1).await;
    let tenant_b = TenantId::new("t_other").expect("canonical fixture tenant");
    state.observability_engine.record_usage(
        &tenant_b,
        UsageEvent::success(
            "t_other".into(),
            "shared-key-name".into(),
            "other-provider".into(),
            "other-model".into(),
            1,
            1,
            2,
            None,
            false,
        )
        // Even a colliding correlation value is NOT an authorization selector.
        .with_request_id(&request_id),
    );
    let b = logs(&state, "t_other", 1).await;
    assert_eq!(b[0]["provider"], "other-provider");
    assert_eq!(b[0]["request_id"], request_id);
    assert_ne!(a[0]["id"], b[0]["id"]);
    assert_eq!(logs(&state, "t_acme", 1).await, a);
}

#[tokio::test]
async fn legacy_missing_id_and_evicted_or_restarted_history_do_not_invent_joins() {
    let state = state();
    assert!(logs(&state, "t_acme", 0).await.is_empty());
    let event = UsageEvent::success(
        "t_acme".into(),
        "legacy".into(),
        "openai".into(),
        "gpt-4o".into(),
        1,
        1,
        2,
        None,
        false,
    );
    let encoded = serde_json::to_value(&event).unwrap();
    assert!(encoded.get("request_id").is_none());
    // Existing closed-vocabulary fields borrow &'static str, so the historical
    // wire fixture must be static rather than requesting DeserializeOwned.
    let historical: UsageEvent = serde_json::from_str(
        r#"{"timestamp":"2026-09-06T00:00:00Z","virtual_key_name":"legacy",
        "provider":"openai","model":"gpt-4o","prompt_tokens":1,
        "completion_tokens":1,"total_tokens":2,"sovereign_routed":false,"success":true}"#,
    )
    .unwrap();
    assert!(historical.request_id.is_none());
    let tenant = TenantId::new("t_acme").expect("canonical fixture tenant");
    state.observability_engine.record_usage(&tenant, historical);
    let rows = logs(&state, "t_acme", 1).await;
    assert!(rows[0].get("request_id").is_none());
    // Feed sequentially so this exercises ring eviction, not ingress overload.
    for i in 0..1001 {
        let request_id = format!("req_retained_{i}");
        state
            .observability_engine
            .record_usage(&tenant, event.clone().with_request_id(&request_id));
        let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
        while !state
            .observability_engine
            .get_recent_events()
            .iter()
            .any(|ev| ev.request_id.as_deref() == Some(request_id.as_str()))
        {
            assert!(tokio::time::Instant::now() < deadline);
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    }
    let events = state.observability_engine.get_recent_events();
    assert!(
        events.iter().all(|ev| ev.request_id.is_some()),
        "legacy event was evicted"
    );
    assert!(!events
        .iter()
        .any(|ev| ev.request_id.as_deref() == Some("req_retained_0")));
    let restarted = Arc::new(AppState {
        observability_engine: ObservabilityEngine::new(vec![tenant]),
        ..AppState::for_tests(ProviderRegistry::new())
    });
    assert!(
        logs(&restarted, "t_acme", 0).await.is_empty(),
        "RAM history is not durable"
    );
}
