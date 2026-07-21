//! Hermetic integration tests for the recent-window CACHE-SAVINGS surface
//! (`GET /v1/finops/cache-savings`). The handler is invoked directly (no HTTP
//! server / auth round-trip — the extensions the auth middleware would inject are
//! passed explicitly, exactly like the finops/timeseries suites). The only state
//! needed is an `AppState` (for the observability ring) and a `SharedAuthState`
//! registry (the key→tenant ownership map the handler resolves the scope from).
//!
//! Covers: the SAME entitlement gate as `usage_timeseries` (every tier entitled
//! by baseline, so a held-back tenant → 403 feature_not_released; entitled
//! Business → 200); seeded cache-hit events sum to real saved cost/tokens + hit
//! count; key-ownership scoping (a caller only sees its OWN keys' hits); an empty
//! ring → 200 with honest zeroes; the honest recent-window + estimate `note`; and
//! the server-side window clamp.

use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::Response;
use axum::Extension;
use routeplane::auth::{shared_auth_state, AuthState, SharedAuthState, TenantContext};
use routeplane::finops_api::{cache_savings, CacheSavingsQuery};
use routeplane::observability::UsageEvent;
use routeplane::proxy::{AppState, ProviderRegistry};
use routeplane_entitlements::{CapabilitySet, Feature, Tier};
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

fn q(window_mins: Option<i64>) -> Query<CacheSavingsQuery> {
    Query(CacheSavingsQuery { window_mins })
}

async fn body_json(resp: Response) -> serde_json::Value {
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .expect("body readable");
    serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null)
}

/// Record an event and wait (bounded) until the async drain task has pushed it into
/// the ring — the same real-time-budget poll the finops/timeseries harnesses use, so
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

/// A served exact-cache hit on the `(cache)` sentinel — the exact shape `proxy.rs`
/// records via `with_cache_hit` (stored token counts preserved, savings attached).
fn cache_hit(key: &str, total_tokens: u32, saved_cost_micro_usd: u64) -> UsageEvent {
    UsageEvent::success(
        key.into(),
        "(cache)".into(),
        "gpt-4o".into(),
        total_tokens / 2,
        total_tokens - total_tokens / 2,
        total_tokens,
        None,
        false,
    )
    .with_cache_hit(Some("default".into()), saved_cost_micro_usd)
}

// --- entitlement gate (must MATCH usage_timeseries exactly) -------------------

#[tokio::test]
async fn free_tenant_held_back_gets_403_feature_not_released() {
    // Every tier grants finops_export by baseline now, so a Free tenant is
    // entitled; a rollout holdback is the only way it goes inactive →
    // feature_not_released (matching /v1/finops/usage exactly).
    let resp = cache_savings(
        State(build_state()),
        Extension(auth()),
        Extension(ctx_held(Tier::Free, "t_acme")),
        q(None),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    let v = body_json(resp).await;
    assert_eq!(v["error"]["code"], "feature_not_released");
}

#[tokio::test]
async fn held_back_business_tenant_gets_403_feature_not_released() {
    let resp = cache_savings(
        State(build_state()),
        Extension(auth()),
        Extension(ctx_held(Tier::Business, "t_acme")),
        q(None),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    assert_eq!(
        body_json(resp).await["error"]["code"],
        "feature_not_released"
    );
}

// --- empty ring → 200, honest zeroes, honest note ----------------------------

#[tokio::test]
async fn empty_ring_returns_200_with_honest_zeroes_and_note() {
    let resp = cache_savings(
        State(build_state()),
        Extension(auth()),
        Extension(ctx(Tier::Business, "t_acme")),
        q(Some(60)),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;

    assert_eq!(v["tenant_id"], "t_acme");
    assert_eq!(v["window_mins"], 60);
    assert_eq!(v["cache_hits"], 0);
    assert_eq!(v["cacheable_lookups"], 0);
    assert_eq!(v["saved_cost_micro_usd"], 0);
    assert_eq!(v["saved_tokens"], 0);
    // The honest recent-window + estimate note.
    let note = v["note"].as_str().expect("note present");
    assert!(note.contains("ring"));
    assert!(note.to_lowercase().contains("estimate"));
}

// --- seeded cache-hit events sum + tenant isolation --------------------------

#[tokio::test]
async fn seeded_hits_sum_saved_cost_tokens_and_count_excluding_other_tenants() {
    let state = build_state();

    // Two served cache hits on the tenant's OWN keys.
    record_and_settle(&state, cache_hit("k_acme_a", 12, 4_321)).await;
    record_and_settle(&state, cache_hit("k_acme_b", 30, 6_000)).await;

    // A cache MISS (participating, but not a hit) on an owned key.
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
        .with_cache_status(Some("miss"), Some("default".into())),
    )
    .await;

    // A plain non-cache success on an owned key — neither a hit nor a lookup.
    record_and_settle(
        &state,
        UsageEvent::success(
            "k_acme_a".into(),
            "openai".into(),
            "gpt-4o".into(),
            7,
            3,
            10,
            None,
            false,
        ),
    )
    .await;

    // Another tenant's cache hit — MUST be excluded (isolation).
    record_and_settle(&state, cache_hit("k_other", 99, 9_999)).await;

    let resp = cache_savings(
        State(state),
        Extension(auth()),
        Extension(ctx(Tier::Business, "t_acme")),
        q(Some(60)),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;

    // Two owned hits; the other tenant's hit excluded.
    assert_eq!(v["cache_hits"], 2);
    // Participating lookups: the two hits + the one miss (the plain success is not).
    assert_eq!(v["cacheable_lookups"], 3);
    // Saved cost sums the two owned hits' ESTIMATES (other tenant excluded).
    assert_eq!(v["saved_cost_micro_usd"], 4_321 + 6_000);
    // Saved tokens = the served responses' total_tokens not re-sent (12 + 30).
    assert_eq!(v["saved_tokens"], 12 + 30);
}

// --- window param clamped server-side ----------------------------------------

#[tokio::test]
async fn out_of_range_window_is_clamped() {
    let resp = cache_savings(
        State(build_state()),
        Extension(auth()),
        Extension(ctx(Tier::Business, "t_acme")),
        q(Some(99_999)),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;
    assert_eq!(v["window_mins"], 1440); // clamped to MAX_WINDOW_MINS
}
