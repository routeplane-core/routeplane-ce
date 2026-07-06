//! Integration coverage for the read-only `GET /status` surface (the internal
//! status board's live signal). Exercises the real shaping logic
//! (`routeplane::status::status_snapshot_json`) against a stub `AppState`, and
//! the route + CORS wiring through Axum (mirroring main.rs's construction).

mod common;

use std::sync::Arc;

use axum::body::Body;
use axum::extract::State;
use axum::http::{Method, Request, StatusCode};
use axum::response::IntoResponse;
use axum::routing::get;
use axum::Router;
use tower::ServiceExt; // for `oneshot`
use tower_http::cors::{Any, CorsLayer};

use common::build_stub_state;
use routeplane::proxy::AppState;
use routeplane::status::status_snapshot_json;

async fn status_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    axum::Json(status_snapshot_json(
        &state.health,
        &state.cache,
        &state.observability_engine,
        0,
        &state.custom_providers.names(),
    ))
}

fn app(state: Arc<AppState>) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::GET, Method::OPTIONS]);
    Router::new()
        .route("/status", get(status_handler))
        .layer(cors)
        .with_state(state)
}

#[tokio::test]
async fn status_snapshot_has_expected_shape() {
    let state = build_stub_state();
    let v = status_snapshot_json(
        &state.health,
        &state.cache,
        &state.observability_engine,
        0,
        &state.custom_providers.names(),
    );

    for key in ["shed_total", "providers", "cache", "usage"] {
        assert!(v.get(key).is_some(), "missing top-level key `{key}`");
    }
    let provs = v["providers"].as_array().expect("providers is an array");
    assert_eq!(provs.len(), 1, "stub state registers exactly one provider");
    assert_eq!(provs[0]["provider"], "openai");
    assert_eq!(provs[0]["circuit"], "closed");
    assert!(
        provs[0]["latency_ewma_ms"].is_null(),
        "EWMA is null until the first sample"
    );
    assert_eq!(v["cache"]["hits"], 0);
    assert_eq!(v["cache"]["hit_rate"], 0.0);
    assert_eq!(v["usage"]["window"], 0);
    assert_eq!(v["usage"]["success_rate"], 0.0);
}

#[tokio::test]
async fn status_route_serves_json_with_cors() {
    // GET → 200 + JSON + permissive CORS origin.
    let resp = app(build_stub_state())
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/status")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(
        resp.headers()
            .get("content-type")
            .map(|h| h.to_str().unwrap()),
        Some("application/json")
    );
    assert_eq!(
        resp.headers()
            .get("access-control-allow-origin")
            .map(|h| h.to_str().unwrap()),
        Some("*")
    );
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(v.get("providers").is_some());

    // OPTIONS preflight → CORS headers present (the dashboard fetches cross-origin).
    let pre = app(build_stub_state())
        .oneshot(
            Request::builder()
                .method(Method::OPTIONS)
                .uri("/status")
                .header("origin", "https://status.routeplane.ai")
                .header("access-control-request-method", "GET")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(
        pre.headers().get("access-control-allow-origin").is_some(),
        "preflight must carry an access-control-allow-origin header"
    );
}
