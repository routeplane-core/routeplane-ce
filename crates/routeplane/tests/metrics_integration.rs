//! Integration coverage for the Prometheus `GET /metrics` surface.
//!
//! Verifies the unauthenticated operational endpoint (like `/healthz`) renders a
//! valid Prometheus text-exposition body with the right `Content-Type`, that the
//! expected metric families are present with `# HELP`/`# TYPE` headers and at
//! least one sample line each, that the legacy `shed_total` series is preserved,
//! and that label cardinality is bounded (no `model=` label; `provider` only ever
//! takes a value from the known bounded set, never a raw/sentinel string).
//!
//! The metrics table is a process-global static, so this test asserts the
//! STRUCTURE of the scrape (a stable, dashboard-friendly surface) rather than
//! exact counts (which the in-process classification unit test in `proxy.rs`
//! covers deterministically against a local table).

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::routing::get;
use axum::Router;
use tower::ServiceExt; // for `oneshot`

/// Mirror main.rs's `/metrics` handler wiring (unauthenticated, text exposition).
fn app() -> Router {
    Router::new().route(
        "/metrics",
        get(|| async {
            (
                [(
                    axum::http::header::CONTENT_TYPE,
                    "text/plain; version=0.0.4",
                )],
                // Real handler threads in the binary-level SHED_TOTAL; the lib
                // test passes 0 (the render logic is the same either way).
                routeplane::metrics::metrics().render(0),
            )
        }),
    )
}

#[tokio::test]
async fn metrics_endpoint_is_unauth_and_returns_prometheus_exposition() {
    // Pre-seed a few samples into the process-global table so the scrape is not
    // trivially all-zero (the table is shared, so we assert presence not exact
    // values).
    let m = routeplane::metrics::metrics();
    m.inc_request("openai", routeplane::metrics::Outcome::Success);
    m.observe_duration("openai", 130);
    m.add_tokens(10, 5);
    m.add_cost_micro_usd(1234);
    m.inc_cache(false, true);
    m.inc_provider_error("anthropic");
    m.inc_hedged_win();

    // No auth header — the endpoint must still serve (operational surface).
    let resp = app()
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .unwrap()
        .to_str()
        .unwrap();
    assert_eq!(ct, "text/plain; version=0.0.4");

    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .unwrap();
    let body = String::from_utf8(bytes.to_vec()).unwrap();

    // Every metric family carries HELP + TYPE + at least one sample line.
    for family in [
        "rp_requests_total",
        "rp_request_duration_ms",
        "rp_tokens_total",
        "rp_cost_micro_usd_total",
        "rp_cache_events_total",
        "rp_provider_errors_total",
        "rp_hedged_wins_total",
        "rp_shed_total",
        "shed_total", // legacy alias preserved
    ] {
        assert!(
            body.contains(&format!("# HELP {family} ")),
            "missing HELP for {family}"
        );
        assert!(
            body.contains(&format!("# TYPE {family} ")),
            "missing TYPE for {family}"
        );
        assert!(
            body.lines().any(|l| l.starts_with(family)),
            "missing a sample line for {family}"
        );
    }

    // Histogram has the standard _bucket / _sum / _count triad.
    assert!(body.contains("rp_request_duration_ms_bucket{"));
    assert!(body.contains("le=\"+Inf\""));
    assert!(body.contains("rp_request_duration_ms_sum{"));
    assert!(body.contains("rp_request_duration_ms_count{"));

    // The seeded samples are visible (>= because the static is shared).
    assert!(body.contains("rp_requests_total{provider=\"openai\",outcome=\"success\"}"));
    assert!(body.contains("rp_hedged_wins_total "));

    // Cardinality / privacy: NO model label anywhere, and the `provider` label
    // only ever takes a bounded value (no raw model, no sentinel string).
    assert!(!body.contains("model="));
    assert!(!body.contains("tenant"));
    assert!(!body.contains("(sovereign_block)"));
    assert!(!body.contains("(cache)"));
    for line in body.lines().filter(|l| l.contains("provider=\"")) {
        let v = line
            .split("provider=\"")
            .nth(1)
            .and_then(|s| s.split('"').next())
            .unwrap();
        assert!(
            [
                "openai",
                "anthropic",
                "gemini",
                "azure_openai",
                "mistral",
                "cohere",
                "bedrock",
                "groq",
                "deepseek",
                "self_hosted",
                "cache",
                "semantic_cache",
                "other",
            ]
            .contains(&v),
            "unexpected (unbounded) provider label value: {v}"
        );
    }
}
