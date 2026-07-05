//! Read-only platform-status surface (`GET /status`) for the internal status
//! board. This module holds the PURE shaping logic — it takes the already-held
//! state references and produces a non-sensitive JSON snapshot. The thin Axum
//! handler + the CORS-scoped route live in `main.rs` (they pass `shed_total()`
//! in), so this stays free of any binary-only globals and is exercised directly
//! by the integration test against a stub `AppState`.
//!
//! Every read here is a lock-free atomic load or an off-hot-path snapshot — no
//! mutex on the request path, no `unwrap()`/panic. The output carries ONLY
//! aggregate operational state: no keys, tenant ids, request bodies, or PII.

use routeplane_cache::ExactCache;
use routeplane_router::{CircuitState, HealthTracker};
use serde_json::{json, Value};

use crate::observability::ObservabilityEngine;

fn circuit_str(state: CircuitState) -> &'static str {
    match state {
        CircuitState::Closed => "closed",
        CircuitState::HalfOpen => "half_open",
        CircuitState::Open => "open",
    }
}

/// Build the `/status` JSON snapshot from the live engines. `shed_total` is
/// passed in because the capacity-shed counter is a binary-level global.
/// `custom_providers` is the (sorted) runtime custom-provider name list —
/// appended to the provider list with an explicit `"custom": true` marker (they
/// are untracked by the boot-time health registry, which fails open for them,
/// so `circuit` is honestly reported `closed` with no latency sample).
pub fn status_snapshot_json(
    health: &HealthTracker,
    cache: &ExactCache,
    observability: &ObservabilityEngine,
    shed_total: u64,
    custom_providers: &[String],
) -> Value {
    let mut names = health.provider_names();
    names.sort_unstable();
    let mut providers: Vec<Value> = names
        .iter()
        .map(|p| {
            json!({
                "provider": p,
                "circuit": circuit_str(health.state(p)),
                "latency_ewma_ms": health.latency_ms(p), // null until first sample
            })
        })
        .collect();
    for name in custom_providers {
        providers.push(json!({
            "provider": name,
            "circuit": "closed", // untracked ⇒ always admitted (fail-open)
            "latency_ewma_ms": Value::Null,
            "custom": true,
        }));
    }

    let (entries, approx_bytes) = cache.stats_snapshot();
    let hits = cache.hits();
    let misses = cache.misses();
    let lookups = hits + misses;
    let hit_rate = if lookups > 0 {
        hits as f64 / lookups as f64
    } else {
        0.0
    };

    json!({
        "shed_total": shed_total,
        "providers": providers,
        "cache": {
            "hits": hits,
            "misses": misses,
            "hit_rate": hit_rate,
            "entries": entries,
            "approx_bytes": approx_bytes,
            "oversize_drops": cache.oversize_drops(),
            "write_drops": cache.write_drops(),
        },
        "usage": observability.usage_summary(),
    })
}
