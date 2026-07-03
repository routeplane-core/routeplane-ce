//! The read-only recent-usage surface (`GET /analytics`).
//!
//! Returns the authenticated tenant's most-recent `UsageEvent`s from the in-memory
//! observability ring. It follows the SAME tenant-isolation model as `/v1/logs`,
//! `/v1/finops/*`, and `/v1/residency/*`: the scope is the set of virtual-key NAMES
//! the requesting tenant owns, resolved SERVER-SIDE from the `SharedAuthState`
//! snapshot (key ownership, ADR-023). No client-supplied identifier ever selects the
//! scope, so a tenant can only ever see its OWN events — never another tenant's
//! `virtual_key_name`, cost, `use_case` labels, or provider `error` strings (which
//! can echo prompt text).
//!
//! Unlike `/v1/logs` (which projects to the sanitized `LogRow`), `/analytics`
//! returns the full `UsageEvent` shape — the richer per-request view — but now scoped
//! to the caller. `/analytics/latency` returns only per-PROVIDER latency aggregates
//! (no per-tenant rows), so it needs no ownership scope and is wired separately.
//!
//! Entitlement: auth + key ownership only (no extra feature gate) — observability of
//! one's own recent requests is a baseline for any authenticated tenant.

use crate::auth::{SharedAuthState, TenantContext};
use crate::proxy::AppState;
use axum::extract::State;
use axum::response::IntoResponse;
use axum::{Extension, Json};
use std::collections::BTreeSet;
use std::sync::Arc;

/// `GET /analytics` — the tenant's own recent `UsageEvent`s, tenant-isolated by key
/// ownership. Read-only over the existing observability ring; no entitlement gate
/// beyond auth.
pub async fn analytics_events(
    State(state): State<Arc<AppState>>,
    Extension(auth_state): Extension<SharedAuthState>,
    Extension(tenant_ctx): Extension<TenantContext>,
) -> impl IntoResponse {
    // Resolve the virtual-key NAMES this tenant owns from the current registry
    // snapshot — identical to logs_api/finops_api. Tenant isolation is structural
    // (by key ownership), never by a client-supplied id (ADR-023).
    let snapshot = auth_state.load();
    let key_names: BTreeSet<String> = snapshot
        .keys
        .values()
        .filter(|vk| vk.resolved_tenant_id() == tenant_ctx.tenant_id)
        .map(|vk| vk.name.clone())
        .collect();

    Json(state.observability_engine.recent_events_owned(&key_names))
}
