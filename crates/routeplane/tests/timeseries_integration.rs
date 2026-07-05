//! Hermetic integration tests for the recent-window usage TIME-SERIES surface
//! (`GET /v1/finops/timeseries`). The handler is invoked directly (no HTTP server /
//! auth round-trip — the extensions the auth middleware would inject are passed
//! explicitly, exactly like the finops/prompts suites). The only state needed is an
//! `AppState` (for the observability ring) and a `SharedAuthState` registry (the
//! key→tenant ownership map the handler resolves the scope from).
//!
//! Covers: the SAME entitlement gate as `usage_export` (Free → 402
//! enterprise_only on CE; held-back Business → 403 feature_not_released; entitled
//! Business → 200); the bucketed shape for seeded events; key-ownership scoping (a
//! caller only sees its own keys' events); an empty ring → 200 with all-zero buckets
//! (honest); and the honest recent-window `note`.

use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::Response;
use axum::Extension;
use routeplane::auth::{shared_auth_state, AuthState, SharedAuthState, TenantContext};
use routeplane::finops_api::{usage_timeseries, TimeseriesQuery};
use routeplane::observability::UsageEvent;
use routeplane::proxy::{AppState, ProviderRegistry};
use routeplane_entitlements::{CapabilitySet, Feature, Tier};
use routeplane_limits::pricing::CostBreakdown;
use routeplane_router::HealthTracker;
use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

const KEYS: &str = r#"{"keys":[
    {"name":"k_acme_a","routeplane_key":"rp_acme_a","provider_keys":{"openai":"x"},"tenant_id":"t_acme","tier":"business"},
    {"name":"k_acme_b","routeplane_key":"rp_acme_b","provider_keys":{"openai":"x"},"tenant_id":"t_acme","tier":"business"},
    {"name":"k_other","routeplane_key":"rp_other","provider_keys":{"openai":"x"},"tenant_id":"t_other","tier":"business"}
]}"#;

fn build_state() -> Arc<AppState> {
    Arc::new(AppState {
        health: HealthTracker::new(["openai"]),
        ..AppState::for_tests(ProviderRegistry::new())
    })
}

fn auth() -> SharedAuthState {
    shared_auth_state(AuthState::load_from_json(KEYS, "test").expect("registry loads"))
}

fn ctx(tier: Tier, tenant: &str) -> TenantContext {
    TenantContext {
        tenant_id: tenant.into(),
        tier,
        capabilities: CapabilitySet::resolve(tier, &BTreeSet::new(), &BTreeSet::new()),
        compliance_frameworks: Vec::new(),
        compliance_mode: routeplane::auth::ComplianceMode::Strict,
    }
}

fn ctx_held(tier: Tier, tenant: &str) -> TenantContext {
    let holdbacks = BTreeSet::from([Feature::FinOpsExport]);
    TenantContext {
        tenant_id: tenant.into(),
        tier,
        capabilities: CapabilitySet::resolve(tier, &BTreeSet::new(), &holdbacks),
        compliance_frameworks: Vec::new(),
        compliance_mode: routeplane::auth::ComplianceMode::Strict,
    }
}

fn q(window_mins: Option<i64>, buckets: Option<usize>) -> Query<TimeseriesQuery> {
    Query(TimeseriesQuery {
        window_mins,
        buckets,
    })
}

async fn body_json(resp: Response) -> serde_json::Value {
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .expect("body readable");
    serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null)
}

/// Record an event and wait (bounded) until the async drain task has pushed it into
/// the ring — the same real-time-budget poll the finops/ab_parity harnesses use, so
/// the assertion never races the background writer.
async fn record_and_settle(state: &AppState, event: UsageEvent) {
    let before = state.observability_engine.get_recent_events().len();
    state.observability_engine.record_usage(event);
    let deadline = Instant::now() + Duration::from_secs(2);
    while state.observability_engine.get_recent_events().len() <= before {
        if Instant::now() > deadline {
            panic!("usage event never reached the ring");
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

fn cost(micro_usd: u64) -> CostBreakdown {
    CostBreakdown {
        micro_usd,
        currency: "USD".into(),
        minor_units: micro_usd / 10,
        region: None,
    }
}

// --- entitlement gate (must MATCH usage_export exactly) -----------------------

#[tokio::test]
async fn free_tenant_gets_402_enterprise_only() {
    // CE contract: not-entitled → the uniform 402 `enterprise_only` upsell
    // (matching /v1/finops/usage and the Enterprise stub routes).
    let resp = usage_timeseries(
        State(build_state()),
        Extension(auth()),
        Extension(ctx(Tier::Free, "t_acme")),
        q(None, None),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::PAYMENT_REQUIRED);
    let v = body_json(resp).await;
    assert_eq!(v["error"]["code"], "enterprise_only");
    assert!(v["error"]["message"]
        .as_str()
        .expect("message")
        .starts_with("/v1/finops/timeseries is a Routeplane Enterprise feature"));
}

#[tokio::test]
async fn held_back_business_tenant_gets_403_feature_not_released() {
    let resp = usage_timeseries(
        State(build_state()),
        Extension(auth()),
        Extension(ctx_held(Tier::Business, "t_acme")),
        q(None, None),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    assert_eq!(
        body_json(resp).await["error"]["code"],
        "feature_not_released"
    );
}

// --- empty ring → 200, all-zero buckets, honest note -------------------------

#[tokio::test]
async fn empty_ring_returns_200_with_zeroed_buckets_and_honest_note() {
    let resp = usage_timeseries(
        State(build_state()),
        Extension(auth()),
        Extension(ctx(Tier::Business, "t_acme")),
        q(Some(60), Some(12)),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;

    assert_eq!(v["tenant_id"], "t_acme");
    assert_eq!(v["window_mins"], 60);
    assert_eq!(v["total_events_in_window"], 0);
    let buckets = v["buckets"].as_array().expect("buckets present");
    assert_eq!(buckets.len(), 12); // every bucket emitted, even with no data…
    for b in buckets {
        // …but honestly zero — never fabricated traffic.
        assert_eq!(b["requests"], 0);
        assert_eq!(b["cost_micro_usd"], 0);
        assert_eq!(b["tokens"], 0);
        assert!(b["ts"].is_string()); // ISO8601 bucket start
    }
    // The honest recent-window note is present and does NOT claim durable history.
    let note = v["note"].as_str().expect("note present");
    assert!(note.contains("recent window") || note.contains("Recent activity"));
    assert!(note.contains("ring"));
}

// --- seeded events bucket correctly + tenant isolation -----------------------

#[tokio::test]
async fn buckets_reflect_seeded_events_and_exclude_other_tenants() {
    let state = build_state();
    // Seed via the real ingest path (record + settle on the async drain), so the
    // assertion never races the background writer. All timestamps default to
    // `Utc::now()`, so they land in the final (most-recent) bucket of the window.

    // Two priced successes + one failure on the tenant's OWN keys.
    record_and_settle(
        &state,
        UsageEvent::success(
            "k_acme_a".into(),
            "openai".into(),
            "gpt-4o".into(),
            10,
            5,
            15,
            None,
            false,
        )
        .with_cost(cost(2_000))
        .with_latency(100),
    )
    .await;
    record_and_settle(
        &state,
        UsageEvent::success(
            "k_acme_b".into(),
            "openai".into(),
            "gpt-4o".into(),
            20,
            10,
            30,
            None,
            false,
        )
        .with_cost(cost(3_000))
        .with_latency(300),
    )
    .await;
    record_and_settle(
        &state,
        UsageEvent::failure(
            "k_acme_a".into(),
            "openai".into(),
            "gpt-4o".into(),
            None,
            false,
            "boom".into(),
        ),
    )
    .await;
    // Another tenant's key — MUST be excluded (isolation).
    record_and_settle(
        &state,
        UsageEvent::success(
            "k_other".into(),
            "openai".into(),
            "gpt-4o".into(),
            99,
            99,
            198,
            None,
            false,
        )
        .with_cost(cost(9_999)),
    )
    .await;
    // A synthetic sentinel on an owned key — excluded (no real attempt).
    record_and_settle(
        &state,
        UsageEvent::sovereign_block("k_acme_a".into(), "m".into(), Some("IN".into())),
    )
    .await;

    let resp = usage_timeseries(
        State(state),
        Extension(auth()),
        Extension(ctx(Tier::Business, "t_acme")),
        q(Some(60), Some(30)),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;

    // 3 owned real attempts in the window (other-tenant + sentinel excluded).
    assert_eq!(v["total_events_in_window"], 3);

    // Sum across all buckets equals the owned, in-window aggregates.
    let buckets = v["buckets"].as_array().expect("buckets present");
    let sum = |field: &str| -> u64 { buckets.iter().map(|b| b[field].as_u64().unwrap_or(0)).sum() };
    assert_eq!(sum("requests"), 3);
    assert_eq!(sum("errors"), 1);
    assert_eq!(sum("tokens"), 15 + 30); // the failure carried 0 tokens
    assert_eq!(sum("cost_micro_usd"), 2_000 + 3_000); // failure + other-tenant excluded

    // The non-empty bucket carries the mean of its two timed samples (100, 300).
    let busy = buckets
        .iter()
        .find(|b| b["requests"].as_u64().unwrap_or(0) > 0)
        .expect("a non-empty bucket exists");
    assert_eq!(busy["avg_latency_ms"], 200);
}

// --- params are clamped server-side ------------------------------------------

#[tokio::test]
async fn out_of_range_params_are_clamped() {
    // window_mins above the 1440 cap and buckets above the 200 cap → clamped.
    let resp = usage_timeseries(
        State(build_state()),
        Extension(auth()),
        Extension(ctx(Tier::Business, "t_acme")),
        q(Some(99_999), Some(99_999)),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;
    assert_eq!(v["window_mins"], 1440); // clamped to MAX_WINDOW_MINS
    assert_eq!(v["buckets"].as_array().unwrap().len(), 200); // clamped to MAX_BUCKETS
}
