//! Hermetic integration tests for POST /v1/feedback (PARITY: Portkey ships a
//! Feedback API; Helicone has feedback/scoring). The handler is invoked directly
//! (no HTTP server, no provider, no auth round-trip — the extensions the auth
//! middleware would inject are passed explicitly), exactly like the rerank /
//! moderations integration suites. Feedback never calls a provider, so there is
//! no wiremock here. Covers:
//!   * a valid submission → 200 ack `{ "status": "recorded", "trace_id": ... }`
//!     AND the feedback lands OFF-path in the in-memory observability ring as a
//!     synthetic `(feedback)` event carrying the value/weight/metadata-count.
//!   * value out of -10..=10 → 400 invalid_request_error.
//!   * weight out of 0.0..=1.0 (and NaN) → 400.
//!   * empty / oversized trace_id → 400.
//!   * metadata is bounded (too many keys / oversized value / nested → 400) and
//!     only a COUNT of keys is retained (never the raw values).
//!   * the route is auth-gated (no key ⇒ 401) via the real auth_middleware.

use axum::body::Body;
use axum::extract::State;
use axum::http::{Request, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Extension, Router};
use routeplane::auth::{
    auth_middleware, shared_auth_state, AuthState, SharedAuthState, TenantContext, VirtualKey,
};
use routeplane::feedback_api::feedback;
use routeplane::observability::UsageEvent;
use routeplane::proxy::AppState;
use routeplane_entitlements::{CapabilitySet, Tier};
use routeplane_router::HealthTracker;
use routeplane_types::FeedbackRequest;
use serde_json::json;
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;
use std::time::Duration;
use tower::ServiceExt;

fn build_state() -> Arc<AppState> {
    Arc::new(AppState {
        health: HealthTracker::new(["openai"]),
        ..AppState::for_tests(HashMap::new())
    })
}

fn vk() -> VirtualKey {
    serde_json::from_value(json!({
        "name": "test-key",
        "routeplane_key": "rp_test",
        "provider_keys": {}
    }))
    .expect("virtual key deserializes")
}

fn ctx() -> TenantContext {
    TenantContext {
        tenant_id: "t_test".into(),
        tier: Tier::Free,
        capabilities: CapabilitySet::resolve(Tier::Free, &BTreeSet::new(), &BTreeSet::new()),
        compliance_frameworks: Vec::new(),
        compliance_mode: routeplane::auth::ComplianceMode::Strict,
    }
}

async fn invoke(state: Arc<AppState>, payload: FeedbackRequest) -> Response {
    feedback(
        State(state),
        Extension(vk()),
        Extension(ctx()),
        routeplane::api_error::OpenAiJson(payload),
    )
    .await
    .into_response()
}

async fn body_json(resp: Response) -> serde_json::Value {
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .expect("body readable");
    serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null)
}

fn req(trace_id: &str, value: i8, weight: Option<f32>) -> FeedbackRequest {
    FeedbackRequest {
        trace_id: trace_id.to_string(),
        value,
        weight,
        metadata: None,
    }
}

/// Poll the observability ring (the recorder is async/non-blocking) on a small
/// real-time budget for the first `(feedback)` event.
async fn await_feedback_event(state: &Arc<AppState>) -> Option<UsageEvent> {
    for _ in 0..50 {
        if let Some(ev) = state
            .observability_engine
            .get_recent_events()
            .into_iter()
            .find(|e| e.provider == "(feedback)")
        {
            return Some(ev);
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    None
}

#[tokio::test]
async fn valid_feedback_is_recorded_and_acked() {
    let state = build_state();
    let resp = invoke(state.clone(), req("req_abc123", 8, Some(0.5))).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;
    assert_eq!(v["status"], "recorded");
    assert_eq!(v["trace_id"], "req_abc123");

    // OFF-path: it landed in the in-memory ring as a synthetic (feedback) event.
    let ev = await_feedback_event(&state)
        .await
        .expect("feedback event recorded in the ring");
    assert_eq!(ev.feedback_trace_id.as_deref(), Some("req_abc123"));
    assert_eq!(ev.feedback_value, Some(8));
    assert_eq!(ev.feedback_weight, Some(0.5));
    assert_eq!(ev.feedback_metadata_keys, Some(0));
    assert!(ev.success);
    assert_eq!(ev.total_tokens, 0);
    assert_eq!(ev.virtual_key_name, "test-key");
}

#[tokio::test]
async fn weight_defaults_to_one_when_omitted() {
    let state = build_state();
    let resp = invoke(state.clone(), req("req_def", -3, None)).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let ev = await_feedback_event(&state).await.expect("recorded");
    assert_eq!(ev.feedback_weight, Some(1.0));
    assert_eq!(ev.feedback_value, Some(-3));
}

#[tokio::test]
async fn value_out_of_range_is_400() {
    for bad in [11_i8, -11, 100, -128] {
        let state = build_state();
        let resp = invoke(state, req("req_x", bad, None)).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST, "value={bad}");
        let v = body_json(resp).await;
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert_eq!(v["error"]["param"], "value");
    }
}

#[tokio::test]
async fn weight_out_of_range_or_nan_is_400() {
    for bad in [1.5_f32, -0.1, f32::NAN, f32::INFINITY] {
        let state = build_state();
        let resp = invoke(state, req("req_x", 0, Some(bad))).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST, "weight={bad}");
        let v = body_json(resp).await;
        assert_eq!(v["error"]["param"], "weight");
    }
}

#[tokio::test]
async fn empty_trace_id_is_400() {
    let state = build_state();
    let resp = invoke(state, req("", 0, None)).await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let v = body_json(resp).await;
    assert_eq!(v["error"]["param"], "trace_id");

    // whitespace-only also trimmed to empty → 400
    let state = build_state();
    let resp = invoke(state, req("   ", 0, None)).await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn oversized_trace_id_is_400() {
    let state = build_state();
    let huge = "x".repeat(257);
    let resp = invoke(state, req(&huge, 0, None)).await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let v = body_json(resp).await;
    assert_eq!(v["error"]["param"], "trace_id");
}

#[tokio::test]
async fn bounded_metadata_records_only_a_key_count() {
    let state = build_state();
    let payload = FeedbackRequest {
        trace_id: "req_meta".into(),
        value: 5,
        weight: None,
        metadata: Some(json!({ "source": "eval-suite", "run": 42, "ok": true })),
    };
    let resp = invoke(state.clone(), payload).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let ev = await_feedback_event(&state).await.expect("recorded");
    // Only the COUNT of keys is retained — never the raw values.
    assert_eq!(ev.feedback_metadata_keys, Some(3));
    // No raw metadata values are present anywhere on the event.
    let serialized = serde_json::to_string(&ev).unwrap();
    assert!(!serialized.contains("eval-suite"));
}

#[tokio::test]
async fn unbounded_or_illshaped_metadata_is_400() {
    // too many keys
    let mut big = serde_json::Map::new();
    for i in 0..17 {
        big.insert(format!("k{i}"), json!(i));
    }
    let cases = vec![
        json!("not-an-object"),
        json!([1, 2, 3]),
        serde_json::Value::Object(big),
        json!({ "k": "x".repeat(257) }),
        json!({ "k": { "nested": 1 } }),
    ];
    for meta in cases {
        let state = build_state();
        let payload = FeedbackRequest {
            trace_id: "req_m".into(),
            value: 0,
            weight: None,
            metadata: Some(meta.clone()),
        };
        let resp = invoke(state, payload).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST, "metadata={meta}");
        let v = body_json(resp).await;
        assert_eq!(v["error"]["param"], "metadata");
    }
}

// --- auth gate --------------------------------------------------------------

const KEYS: &str = r#"{"keys":[
    {"name":"k_acme","routeplane_key":"rp_acme","provider_keys":{},"tenant_id":"t_acme","tier":"free"}
]}"#;

fn auth() -> SharedAuthState {
    shared_auth_state(AuthState::load_from_json(KEYS, "test").expect("registry loads"))
}

/// A minimal router mirroring `main.rs`'s authed wiring: `/v1/feedback` behind
/// the real `auth_middleware`. An unauthenticated caller is rejected with the
/// standard 401 envelope BEFORE the handler runs.
fn authed_router() -> Router {
    Router::new()
        .route("/v1/feedback", post(feedback))
        .layer(axum::middleware::from_fn(auth_middleware))
        .layer(axum::Extension(auth()))
        .with_state(build_state())
}

#[tokio::test]
async fn unauthenticated_feedback_is_rejected_401() {
    let resp = authed_router()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/feedback")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"trace_id":"req_x","value":5}"#))
                .expect("request builds"),
        )
        .await
        .expect("router responds");
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    let v = body_json(resp).await;
    assert_eq!(v["error"]["code"], "invalid_api_key");
}

#[tokio::test]
async fn authenticated_feedback_through_router_is_recorded() {
    let state = build_state();
    let router = Router::new()
        .route("/v1/feedback", post(feedback))
        .layer(axum::middleware::from_fn(auth_middleware))
        .layer(axum::Extension(auth()))
        .with_state(state.clone());
    let resp = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/feedback")
                .header("content-type", "application/json")
                .header("x-routeplane-api-key", "rp_acme")
                .body(Body::from(
                    r#"{"trace_id":"req_routed","value":10,"weight":0.25}"#,
                ))
                .expect("request builds"),
        )
        .await
        .expect("router responds");
    assert_eq!(resp.status(), StatusCode::OK);
    let ev = await_feedback_event(&state).await.expect("recorded");
    assert_eq!(ev.feedback_trace_id.as_deref(), Some("req_routed"));
    assert_eq!(ev.feedback_value, Some(10));
    assert_eq!(ev.feedback_weight, Some(0.25));
    // Tenant identified by key ownership.
    assert_eq!(ev.virtual_key_name, "k_acme");
}
