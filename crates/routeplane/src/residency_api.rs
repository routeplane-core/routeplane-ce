//! The read-only residency-observability surface (`GET /v1/residency/summary` +
//! `GET /v1/residency/ledger`).
//!
//! Two routes that return the authenticated tenant's residency decisions from the
//! in-memory observability ring (the last ~1000 `UsageEvent`s). They are the
//! residency-shaped twin of `logs_api` and follow the SAME tenant-isolation model
//! EXACTLY: the scope is the typed tenant authority resolved at authentication.
//! No client-supplied identifier or collidable display-key name selects the scope
//! (the ADR-023 bypass rule), so a tenant can only ever see its own decisions.
//!
//! Entitlement choice (documented deliberately): like `/v1/logs` (and UNLIKE
//! `finops_api`, gated on `Feature::FinOpsExport`), these are gated on **auth + key
//! ownership only** — no extra feature gate. Observability of one's OWN residency
//! decisions is a reasonable baseline for any authenticated tenant; it carries no
//! cross-tenant data and no raw content (the ledger is label-only — region,
//! outcome, model, key — never prompt/response bytes). Durable residency history
//! (a dated daily series, a per-request compliance framework) would live behind the
//! telemetry store + an entitlement when ADR-024 lands.
//!
//! This is a NEW module (the chat orchestrator in `proxy.rs` is UNTOUCHED). It is
//! read-only over the EXISTING ring (no new durable store, same posture as
//! `/v1/logs`, `/v1/finops/usage`, and `/metrics` — no ADR needed) and emits NO
//! usage event of its own.

use crate::auth::{SharedAuthState, TenantContext};
use crate::proxy::AppState;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::{Extension, Json};
use serde_json::json;
use std::sync::Arc;

/// Max rows returned in one `GET /v1/residency/ledger` read. Bounded so the
/// response (and the per-event projection work) is capped regardless of ring size;
/// the ring itself holds ≤1000 events. Matches `logs_api::LOGS_LIMIT`.
const RESIDENCY_LEDGER_LIMIT: usize = 200;

/// `GET /v1/residency/summary` — the tenant's residency-decision SUMMARY.
///
/// Read-only, tenant-isolated by typed auth authority, no extra entitlement gate.
/// Returns the `ResidencySummaryView` over the existing observability ring (counts,
/// convenience percentages, by-region, by-outcome). Carries NO daily `series` (the
/// ring is not dated history — honest-absent) and NO compliance framework.
pub async fn residency_summary(
    State(state): State<Arc<AppState>>,
    Extension(_auth_state): Extension<SharedAuthState>,
    Extension(tenant_ctx): Extension<TenantContext>,
) -> Response {
    // The summary is folded over the full window regardless of the ledger cap.
    let (summary, _rows) = state.observability_engine.residency_report(
        tenant_ctx.resource_tenant_id.as_ref(),
        RESIDENCY_LEDGER_LIMIT,
    );
    (StatusCode::OK, Json(summary)).into_response()
}

/// `GET /v1/residency/ledger` — the tenant's residency-decision ledger rows
/// (newest-first, label-only, capped at `RESIDENCY_LEDGER_LIMIT`).
///
/// Read-only, tenant-isolated by typed auth authority, no extra entitlement gate.
/// Returns `{ "entries": [ ResidencyLedgerRow, ... ] }` over the existing
/// observability ring. Each row carries decision labels only (region, outcome,
/// model, key) — never raw content. The compliance `framework` is honest-absent.
pub async fn residency_ledger(
    State(state): State<Arc<AppState>>,
    Extension(_auth_state): Extension<SharedAuthState>,
    Extension(tenant_ctx): Extension<TenantContext>,
) -> Response {
    let (_summary, rows) = state.observability_engine.residency_report(
        tenant_ctx.resource_tenant_id.as_ref(),
        RESIDENCY_LEDGER_LIMIT,
    );
    (StatusCode::OK, Json(json!({ "entries": rows }))).into_response()
}
