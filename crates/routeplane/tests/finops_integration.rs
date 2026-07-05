//! Hermetic integration tests for the FinOps chargeback/showback export surface
//! (PRD-008 FR-24). The handler is invoked directly (no HTTP server / auth
//! round-trip — the extensions the auth middleware would inject are passed
//! explicitly, exactly like the prompts/embeddings suites). The only state needed
//! is an `AppState` (for the observability ring) and a `SharedAuthState` registry
//! (the key→tenant ownership map the handler resolves the scope from).
//!
//! Covers: the entitlement gate (Free → 402 enterprise_only on CE; held-back
//! Business → 403 feature_not_released; entitled Business → 200); the 200 envelope
//! shape (carries tenant_id + window/events_matched); and tenant isolation — a
//! seeded event on the tenant's own key is counted, and the report never reflects
//! another tenant's keys.

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::Response;
use axum::Extension;
use routeplane::auth::{shared_auth_state, AuthState, SharedAuthState, TenantContext};
use routeplane::finops_api::usage_export;
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

async fn body_json(resp: Response) -> serde_json::Value {
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .expect("body readable");
    serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null)
}

/// Record an event and wait (bounded) until the async drain task has pushed it
/// into the ring — the same real-time-budget poll the ab_parity harness uses, so
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

// --- entitlement gate ---------------------------------------------------------

#[tokio::test]
async fn free_tenant_gets_402_enterprise_only() {
    // CE contract: a NOT-ENTITLED tenant gets the uniform 402 `enterprise_only`
    // upsell (the same envelope as /v1/moderations and the /v1/mcp/* stubs),
    // replacing the old 403 `feature_not_entitled`.
    let resp = usage_export(
        State(build_state()),
        Extension(auth()),
        Extension(ctx(Tier::Free, "t_acme")),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::PAYMENT_REQUIRED);
    assert_eq!(
        resp.headers()
            .get("x-routeplane-upgrade")
            .and_then(|v| v.to_str().ok()),
        Some("https://routeplane.ai")
    );
    let v = body_json(resp).await;
    assert_eq!(v["error"]["code"], "enterprise_only");
    assert!(v["error"]["message"]
        .as_str()
        .expect("message")
        .starts_with("/v1/finops/usage is a Routeplane Enterprise feature"));
}

#[tokio::test]
async fn standard_tenant_gets_402_enterprise_only() {
    // Standard baseline does NOT include finops_export (Business+ only) — on
    // the CE build the not-entitled refusal is the uniform enterprise_only 402.
    let resp = usage_export(
        State(build_state()),
        Extension(auth()),
        Extension(ctx(Tier::Standard, "t_acme")),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::PAYMENT_REQUIRED);
    assert_eq!(body_json(resp).await["error"]["code"], "enterprise_only");
}

#[tokio::test]
async fn held_back_business_tenant_gets_403_feature_not_released() {
    // Business baseline grants finops_export, but a rollout holdback removes it →
    // entitled-by-baseline yet inactive → feature_not_released (distinct from
    // not-entitled).
    let resp = usage_export(
        State(build_state()),
        Extension(auth()),
        Extension(ctx_held(Tier::Business, "t_acme")),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    assert_eq!(
        body_json(resp).await["error"]["code"],
        "feature_not_released"
    );
}

// --- entitled happy path + envelope shape ------------------------------------

#[tokio::test]
async fn business_tenant_with_no_traffic_gets_200_empty_rollup() {
    let resp = usage_export(
        State(build_state()),
        Extension(auth()),
        Extension(ctx(Tier::Business, "t_acme")),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;
    assert_eq!(v["tenant_id"], "t_acme");
    assert_eq!(v["events_matched"], 0);
    assert_eq!(v["totals"]["requests"], 0);
    assert_eq!(v["totals"]["cost_micro_usd"], 0);
}

// --- tenant isolation end-to-end ---------------------------------------------

#[tokio::test]
async fn report_counts_own_key_and_excludes_other_tenants() {
    let state = build_state();

    // One priced success on the tenant's OWN key (k_acme_a)…
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
            true,
        )
        .with_cost(CostBreakdown {
            micro_usd: 2_500,
            currency: "USD".into(),
            minor_units: 250,
            region: None,
        }),
    )
    .await;
    // …and one on ANOTHER tenant's key (k_other) that must never appear.
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
            true,
        )
        .with_cost(CostBreakdown {
            micro_usd: 9_999,
            currency: "USD".into(),
            minor_units: 999,
            region: None,
        }),
    )
    .await;

    let resp = usage_export(
        State(state),
        Extension(auth()),
        Extension(ctx(Tier::Business, "t_acme")),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;

    assert_eq!(v["tenant_id"], "t_acme");
    assert_eq!(v["window"], 2); // both events are in the ring…
    assert_eq!(v["events_matched"], 1); // …but only the tenant's own counts.
    assert_eq!(v["totals"]["requests"], 1);
    assert_eq!(v["totals"]["total_tokens"], 15);
    assert_eq!(v["totals"]["cost_micro_usd"], 2_500);
    assert_eq!(v["totals"]["cost_by_currency"]["USD"], 250);
    assert_eq!(v["by_key"]["k_acme_a"]["requests"], 1);
    assert!(v["by_key"].get("k_other").is_none());
    assert_eq!(v["by_model"]["gpt-4o"]["total_tokens"], 15);
}

// --- multi-currency display selection surfaces in the export (PRD-015) --------

#[tokio::test]
async fn export_aggregates_multiple_display_currencies_including_jpy() {
    // Two priced events whose cost view was rendered in DIFFERENT display
    // currencies (the chosen-currency selection upstream produces a
    // `CostBreakdown` with that currency + minor units). The chargeback rollup
    // must sum each currency's integer minor units under its own key — proving the
    // multi-currency view (not USD-only) surfaces in the FinOps export.
    let state = build_state();

    // INR view (2dp paise): 1_300_000 micro-USD ≈ $1.30 → 13_000 * 8_300/1e6 …
    // here we assert the exact integer the converter produced via cost_breakdown.
    let inr = routeplane_limits::pricing::cost_breakdown_with(
        &state.fx_rates.load(),
        Some("INR"),
        "gpt-4o",
        Some("IN"),
        1000,
        1000,
    );
    assert_eq!(inr.currency, "INR");

    // JPY view (0dp): hot-swap a JPY-aware table in via the same JSON-merge path
    // the binary's env loader uses (no process env — env is global and would race
    // parallel tests). The minor units must be whole yen — NOT 100× a 2dp ccy.
    use routeplane_limits::fx::SharedFxRatesExt;
    let jpy_table = routeplane_limits::fx::FxRates::from_json_str(
        r#"{"rates":{"JPY":{"minor_per_usd":150,"exponent":0}},
            "region_currency":{"JP":"JPY"}}"#,
    );
    state.fx_rates.replace(jpy_table);

    let jpy = routeplane_limits::pricing::cost_breakdown_with(
        &state.fx_rates.load(),
        Some("JPY"),
        "gpt-4o",
        None,
        1000,
        1000,
    );
    assert_eq!(jpy.currency, "JPY");
    // 13_000 micro-USD * 150 / 1_000_000 = 1 yen (floor) — a 0dp result, not 100×.
    assert_eq!(jpy.minor_units, 13_000 * 150 / 1_000_000);

    record_and_settle(
        &state,
        UsageEvent::success(
            "k_acme_a".into(),
            "openai".into(),
            "gpt-4o".into(),
            1000,
            1000,
            2000,
            Some("IN".into()),
            true,
        )
        .with_cost(inr.clone()),
    )
    .await;
    record_and_settle(
        &state,
        UsageEvent::success(
            "k_acme_b".into(),
            "openai".into(),
            "gpt-4o".into(),
            1000,
            1000,
            2000,
            None,
            false,
        )
        .with_cost(jpy.clone()),
    )
    .await;

    let resp = usage_export(
        State(state),
        Extension(auth()),
        Extension(ctx(Tier::Business, "t_acme")),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v = body_json(resp).await;

    // Canonical USD truth sums both events regardless of display currency.
    assert_eq!(
        v["totals"]["cost_micro_usd"],
        (inr.micro_usd + jpy.micro_usd)
    );
    // Each display currency surfaces under its own key (multi-currency view).
    assert_eq!(v["totals"]["cost_by_currency"]["INR"], inr.minor_units);
    assert_eq!(v["totals"]["cost_by_currency"]["JPY"], jpy.minor_units);
}
