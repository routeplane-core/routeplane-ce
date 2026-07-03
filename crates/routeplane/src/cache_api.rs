//! The cache-purge surface (PRD-007 FR-19 "flush generations" — PARITY with
//! Portkey/LiteLLM cache invalidation).
//!
//! One route, on the AUTHED router (so it inherits the standard `x-routeplane-
//! api-key` → 401 seam, exactly like the other `/v1/*` routes):
//!   * `POST /v1/cache/purge` — invalidate this tenant's exact-match cache for a
//!     specific `namespace`, or for ALL namespaces the tenant has ever purged.
//!
//! Tenant scope is taken from the authenticated `TenantContext` — never a client
//! header/body field — so a tenant can only ever purge ITS OWN entries
//! (structural cross-tenant isolation, the same posture as the cache key itself).
//!
//! Mechanics (ADR-022 incremental, no new ADR): a purge bumps the per-`(tenant,
//! namespace)` flush generation in the lock-free [`FlushRegistry`]. Subsequent
//! cacheable requests in that scope derive a new-generation key (a fresh miss);
//! the orphaned prior-generation entries age out via the existing TTL/FIFO
//! eviction. O(1), lock-free, no shard iteration, and the hot read path stays
//! wait-free. Per-replica, like the cache itself — multi-replica coordinated
//! purge is a documented follow-on (consistent with the per-replica cache
//! posture; no Redis here, a trigger-gated rung per ADR-022 §1).

use crate::auth::TenantContext;
use crate::proxy::AppState;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::{Extension, Json};
use routeplane_cache::WILDCARD_NAMESPACE;
use serde::Deserialize;
use serde_json::json;
use std::sync::Arc;

/// The (optional) JSON body for `POST /v1/cache/purge`. Absent body or absent
/// `namespace` ⇒ purge ALL of the tenant's namespaces (a wildcard generation
/// bump). The tenant is NEVER taken from the body — only from the authenticated
/// context — so cross-tenant purge is impossible by construction.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct PurgeRequest {
    /// A specific namespace to purge. `None`/absent ⇒ all namespaces.
    pub namespace: Option<String>,
}

/// `POST /v1/cache/purge` — bump the flush generation for this tenant's
/// `namespace` (or, when absent, the tenant-wide wildcard scope). Returns a
/// small JSON ack. Auth-gated by the authed router; tenant-scoped by the
/// authenticated context.
pub async fn purge(
    State(state): State<Arc<AppState>>,
    Extension(tenant_ctx): Extension<TenantContext>,
    // The body is optional: a bare `POST` with no/empty body purges all of the
    // tenant's namespaces. `Option<Json<..>>` tolerates an absent or empty body
    // without a 400 (parity with clients that send no body for a flush-all).
    body: Option<Json<PurgeRequest>>,
) -> Response {
    let req = body.map(|Json(b)| b).unwrap_or_default();

    // Tenant scope is structural: the registry keys on (tenant_id, namespace),
    // and we only ever pass THIS tenant's id, so a purge can never reach another
    // tenant's entries. The wildcard scope ("*") is namespace-disjoint from any
    // real namespace string and is folded into every namespace's effective
    // generation on the read path (`FlushRegistry::generation_effective`).
    let (scope_namespace, response_namespace): (&str, serde_json::Value) = match &req.namespace {
        Some(ns) if !ns.trim().is_empty() => (ns.as_str(), json!(ns)),
        // Flush-all: bump the tenant-wide wildcard ("*") scope. Because the read
        // path folds the wildcard generation into EVERY namespace's effective
        // generation, this genuinely invalidates all of the tenant's namespaces
        // at once — it is NOT a no-op that only a request naming "*" would ever
        // see. (Per-replica caveat, FR-19 follow-on: this clears THIS replica's
        // view; multi-replica coordinated purge remains a documented follow-on,
        // consistent with the per-replica cache posture.)
        _ => (WILDCARD_NAMESPACE, json!(null)),
    };

    let generation = state
        .cache_flush
        .bump(&tenant_ctx.tenant_id, scope_namespace);

    tracing::info!(
        "cache purge: tenant={} namespace={:?} new_generation={}",
        tenant_ctx.tenant_id,
        req.namespace,
        generation
    );

    (
        StatusCode::OK,
        Json(json!({
            "purged": true,
            "tenant": tenant_ctx.tenant_id,
            "namespace": response_namespace,
            "generation": generation,
        })),
    )
        .into_response()
}
