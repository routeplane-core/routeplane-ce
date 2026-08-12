//! The cache-purge surface (PRD-007 FR-19 "flush generations" — PARITY with
//! Portkey/LiteLLM cache invalidation).
//!
//! One route, on the AUTHED router (so it inherits the standard `x-routeplane-
//! api-key` → 401 seam, exactly like the other `/v1/*` routes):
//!   * `POST /v1/cache/purge` — invalidate this tenant's exact-match cache for a
//!     specific `namespace`, or for ALL namespaces the tenant has ever purged.
//!
//! Tenant scope is taken from the authenticated `VirtualKey`'s explicit,
//! validated `tenant_id` — never its legacy display-name fallback and never a
//! client header/body field — so a tenant can only purge ITS OWN entries.
//!
//! Mechanics (ADR-022): a purge bumps the per-`(tenant,
//! namespace)` flush generation in the lock-free [`FlushRegistry`]. Subsequent
//! cacheable requests in that scope derive a new-generation key (a fresh miss);
//! the orphaned prior-generation entries age out via the existing TTL/FIFO
//! eviction. O(1), lock-free, no shard iteration, and the hot read path stays
//! wait-free. Per-replica, like the cache itself — multi-replica coordinated
//! purge is a documented follow-on (consistent with the per-replica cache
//! posture; no Redis here, a trigger-gated rung per ADR-022 §1).

use crate::api_error::error_response;
use crate::auth::VirtualKey;
use crate::proxy::AppState;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::{Extension, Json};
use routeplane_cache::{FlushError, WILDCARD_NAMESPACE};
use routeplane_types::TenantId;
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
    Extension(virtual_key): Extension<VirtualKey>,
    // The body is optional: a bare `POST` with no/empty body purges all of the
    // tenant's namespaces. `Option<Json<..>>` tolerates an absent or empty body
    // without a 400 (parity with clients that send no body for a flush-all).
    body: Option<Json<PurgeRequest>>,
) -> Response {
    let req = body.map(|Json(b)| b).unwrap_or_default();

    let Some(tenant_id) = virtual_key
        .canonical_tenant_id()
        .and_then(|id| TenantId::new(id).ok())
        .filter(|tenant_id| state.cache.owns_tenant(tenant_id))
    else {
        return error_response(
            StatusCode::FORBIDDEN,
            "routeplane_cache_tenant_authority_missing",
            "Cache purge requires an explicit canonical tenant identity.",
            "permission_error",
            None,
        );
    };

    // Tenant scope is structural: the registry keys on (tenant_id, namespace),
    // and we only ever pass THIS tenant's id, so a purge can never reach another
    // tenant's entries. The wildcard scope ("*") is namespace-disjoint from any
    // real namespace string and is folded into every namespace's effective
    // generation on the read path (`FlushRegistry::generation_effective`).
    let (scope_namespace, response_namespace): (&str, serde_json::Value) = match &req.namespace {
        Some(ns) if routeplane_policy::is_valid_cache_namespace(ns) => (ns.as_str(), json!(ns)),
        Some(_) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "routeplane_invalid_cache_namespace",
                "Cache namespace must match [a-z0-9_-]{1,64}.",
                "invalid_request_error",
                Some("namespace"),
            );
        }
        // Flush-all: bump the tenant-wide wildcard ("*") scope. Because the read
        // path folds the wildcard generation into EVERY namespace's effective
        // generation, this genuinely invalidates all of the tenant's namespaces
        // at once — it is NOT a no-op that only a request naming "*" would ever
        // see. (Per-replica caveat, FR-19 follow-on: this clears THIS replica's
        // view; multi-replica coordinated purge remains a documented follow-on,
        // consistent with the per-replica cache posture.)
        _ => (WILDCARD_NAMESPACE, json!(null)),
    };

    let generation = match state.cache_flush.bump(&tenant_id, scope_namespace) {
        Ok(generation) => generation,
        Err(FlushError::InvalidNamespace) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "routeplane_invalid_cache_namespace",
                "Cache namespace must match [a-z0-9_-]{1,64}.",
                "invalid_request_error",
                Some("namespace"),
            );
        }
        Err(FlushError::ScopeLimit) => {
            return error_response(
                StatusCode::TOO_MANY_REQUESTS,
                "routeplane_cache_purge_scope_limit",
                "Cache purge scope limit reached for this tenant.",
                "rate_limit_error",
                Some("namespace"),
            );
        }
        Err(FlushError::UnknownTenant) => {
            return error_response(
                StatusCode::FORBIDDEN,
                "routeplane_cache_tenant_authority_missing",
                "Cache purge requires an explicit canonical tenant identity.",
                "permission_error",
                None,
            );
        }
    };

    tracing::info!(
        tenant = tenant_id.as_str(),
        flush_all = req.namespace.is_none(),
        new_generation = generation,
        "cache purge completed"
    );

    (
        StatusCode::OK,
        Json(json!({
            "purged": true,
            "tenant": tenant_id.as_str(),
            "namespace": response_namespace,
            "generation": generation,
        })),
    )
        .into_response()
}
