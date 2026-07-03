//! CP→DP runtime config overlay — per-tenant model-enablement enforcement
//! ([ADR-063] / [PRD-039], first enforced config).
//!
//! The control plane is authoritative for per-tenant config (model enablement +
//! billing path, [ADR-035]), but historically that config was *inert* at the
//! gateway: a model disabled in the Console was not blocked at request time. This
//! module closes that gap with a **pull-based, lock-free overlay**:
//!
//! - A background poller ([`poller`]) fetches each known tenant's model-configs
//!   from the CP `GET /tenants/{id}/model-configs` endpoint off the hot path and
//!   atomically swaps an [`ArcSwap<ConfigOverlay>`] held on `AppState`.
//! - The request path reads the overlay snapshot wait-free
//!   ([`arc_swap::ArcSwap::load`]) and rejects ONLY a `(tenant, model)` pair that
//!   the overlay marks explicitly `enabled = false` (default-allow).
//!
//! ## Safety invariants (ADR-063 §4/§6)
//! - **Off by default.** The poller starts ONLY when `RP_CP_CONFIG_URL` is set;
//!   absent ⇒ no task, an EMPTY overlay, enforcement is a no-op ⇒ byte-identical
//!   to the boot-config gateway (the `ab_parity` golden guard stays green).
//! - **Default-allow.** No overlay entry for `(tenant, model)` ⇒ allowed (today's
//!   behavior). Only an explicit `enabled = false` rejects. An empty overlay
//!   allows everything.
//! - **Fail-open.** A failed/slow poll keeps the last-good overlay; a
//!   never-successful poll keeps the empty overlay ⇒ allow. A CP outage never
//!   rejects traffic nor crashes the gateway.
//! - **Lock-free.** Reads are a single `ArcSwap::load` + one `HashMap` lookup on a
//!   snapshot; no mutex, no network, no allocation on the request path.
//!
//! [ADR-063]: see docs/adr/063-cp-to-dp-runtime-config-distribution.md
//! [PRD-039]: see docs/product/prd/039-cp-to-dp-config-distribution.md
//! [ADR-035]: see docs/adr/035-catalog-enrichment-and-compliance-gating.md

use std::collections::HashMap;
use std::sync::Arc;

use arc_swap::ArcSwap;
use serde::Deserialize;

// The poller half of this module is enterprise-only (PRD-047 / ADR-088): the
// overlay TYPES below are always compiled (AppState holds one), but without a
// control plane there is nothing to poll, so the fetch/build/spawn functions —
// and this import — ride the `enterprise` feature with `cp_config` itself.
#[cfg(feature = "enterprise")]
use crate::cp_config::PollerConfig;

/// Hot-swappable handle to the current [`ConfigOverlay`], read wait-free on the
/// request path and swapped atomically by the poller. Mirrors the `SharedAuthState`
/// / FX-rate / policy-registry posture (`Arc<ArcSwap<_>>`): readers never lock.
pub type SharedConfigOverlay = Arc<ArcSwap<ConfigOverlay>>;

/// Build a shared overlay handle initialized to `empty` — the off-by-default
/// state. `AppState` always holds one of these; with the poller disabled it stays
/// empty for the process lifetime, so enforcement is a permanent no-op.
pub fn new_shared_empty() -> SharedConfigOverlay {
    Arc::new(ArcSwap::from_pointee(ConfigOverlay::empty()))
}

/// A single row of the CP `GET /tenants/{id}/model-configs` response. We only
/// read the two fields the data plane enforces (`model_id`, `enabled`); the wire
/// shape also carries `tenant_id`, `billing_path`, and `updated_at`, which
/// `serde` ignores here (extra fields are tolerated — forward-compatible).
#[derive(Debug, Clone, Deserialize)]
// CE: only the (gated) poller + tests read the rows; the type stays so the
// overlay builder keeps one shape on both variants.
#[cfg_attr(not(feature = "enterprise"), allow(dead_code))]
pub struct CpModelConfig {
    pub model_id: String,
    pub enabled: bool,
}

/// A lock-free snapshot of per-tenant enforcement config distributed from the
/// control plane. Today it carries only model enablement; per ADR-063 §7 it is a
/// typed struct so detectors/routing/limits become additional fields polled the
/// same way, each shipped behind the same gate.
///
/// The map is `tenant_id -> (model_id -> enabled)`. **Default is empty** ⇒ every
/// `model_enabled` returns `None` ⇒ default-allow (byte-identical to today).
#[derive(Debug, Clone, Default)]
pub struct ConfigOverlay {
    /// `tenant_id -> { model_id -> enabled }`. Absent tenant or absent model ⇒
    /// `None` from [`model_enabled`](Self::model_enabled) ⇒ allow.
    by_tenant: HashMap<String, HashMap<String, bool>>,
}

// CE: the builder/logging helpers are only called by the (gated) poller and
// the unit tests; `model_enabled` stays hot on both variants.
#[cfg_attr(not(feature = "enterprise"), allow(dead_code))]
impl ConfigOverlay {
    /// An empty overlay — the cold-start / fail-open default. Enforcement is a
    /// no-op against it (every lookup misses ⇒ allow).
    pub fn empty() -> Self {
        Self::default()
    }

    /// Build an overlay from per-tenant CP model-config rows. The input is a map
    /// of `tenant_id -> rows` (the poller fetches one tenant at a time and
    /// assembles this). On a duplicate `model_id` within a tenant the LAST row
    /// wins (the CP keeps one config per `model_id`, so duplicates are not
    /// expected; last-wins is a deterministic, allocation-free choice).
    pub fn from_tenant_configs<I, R>(tenants: I) -> Self
    where
        I: IntoIterator<Item = (String, R)>,
        R: IntoIterator<Item = CpModelConfig>,
    {
        let mut by_tenant: HashMap<String, HashMap<String, bool>> = HashMap::new();
        for (tenant_id, rows) in tenants {
            let entry = by_tenant.entry(tenant_id).or_default();
            for row in rows {
                entry.insert(row.model_id, row.enabled);
            }
        }
        Self { by_tenant }
    }

    /// Look up whether `model_id` is enabled for `tenant_id`.
    ///
    /// - `None` — no overlay entry for this `(tenant, model)` (unknown tenant or
    ///   unknown model). **Default-allow:** the caller proceeds as today.
    /// - `Some(true)` — explicitly enabled. Proceed.
    /// - `Some(false)` — explicitly disabled. The caller rejects with a typed
    ///   403 (`model_disabled_for_tenant`).
    ///
    /// This is the hot-path read: two `HashMap` lookups on a borrowed snapshot,
    /// no allocation. Against an empty overlay it is a single top-level miss.
    #[inline]
    pub fn model_enabled(&self, tenant_id: &str, model_id: &str) -> Option<bool> {
        self.by_tenant.get(tenant_id)?.get(model_id).copied()
    }

    /// Number of tenants carried — for startup/poll logging only (never the hot
    /// path).
    pub fn tenant_count(&self) -> usize {
        self.by_tenant.len()
    }

    /// Total `(tenant, model)` entries carried — for logging only.
    pub fn entry_count(&self) -> usize {
        self.by_tenant.values().map(HashMap::len).sum()
    }
}

/// Fetch one tenant's model-configs from the CP and parse the rows. Network +
/// JSON only — no overlay state, so it is trivially unit-testable against a mock
/// HTTP server, and the BUILDER ([`ConfigOverlay::from_tenant_configs`]) is tested
/// without any network at all. Returns an error string (logged, never surfaced to
/// a request) on any transport/parse failure so the caller can keep last-good.
#[cfg(feature = "enterprise")]
async fn fetch_tenant_configs(
    client: &reqwest::Client,
    cfg: &PollerConfig,
    tenant_id: &str,
) -> Result<Vec<CpModelConfig>, String> {
    let url = format!("{}/tenants/{}/model-configs", cfg.base_url, tenant_id);
    let mut req = client.get(&url).timeout(cfg.fetch_timeout);
    // ADR-066: resolve the service credential (managed-identity token in prod, a
    // gated static token in dev, or none for a local mock CP). A credential-
    // acquisition failure is treated as a failed fetch ⇒ keep last-good (fail-open).
    if let Some(token) = cfg.bearer_token().await? {
        req = req.bearer_auth(token);
    }
    let resp = req
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;
    let status = resp.status();
    if !status.is_success() {
        return Err(format!("CP returned status {status}"));
    }
    resp.json::<Vec<CpModelConfig>>()
        .await
        .map_err(|e| format!("decode failed: {e}"))
}

/// Build a fresh overlay by polling every tenant in `tenant_ids`. A per-tenant
/// fetch failure is logged and that tenant is simply omitted from THIS overlay
/// (default-allow for it) — the swap still happens with whatever succeeded. The
/// whole refresh never panics and never propagates an error to the request path.
#[cfg(feature = "enterprise")]
async fn build_overlay(
    client: &reqwest::Client,
    cfg: &PollerConfig,
    tenant_ids: &[String],
) -> ConfigOverlay {
    let mut tenants: Vec<(String, Vec<CpModelConfig>)> = Vec::with_capacity(tenant_ids.len());
    for tenant_id in tenant_ids {
        match fetch_tenant_configs(client, cfg, tenant_id).await {
            Ok(rows) => tenants.push((tenant_id.clone(), rows)),
            Err(e) => {
                tracing::warn!("cp-config poll: tenant {tenant_id} fetch failed ({e}); omitting (default-allow)");
            }
        }
    }
    ConfigOverlay::from_tenant_configs(tenants)
}

/// Spawn the background config-distribution poller (ADR-063 §2). Seeds once at
/// startup, then refreshes on the timer; each cycle atomically swaps `overlay`.
///
/// **Fail-open is structural:** the overlay is only ever REPLACED with a freshly
/// built one; a transport failure for a tenant omits that tenant (default-allow),
/// and if the very first poll yields nothing the overlay stays empty ⇒ allow. The
/// task is detached (like the Unleash poller, ADR-029): a transient CP outage just
/// means "no refresh", and the request path keeps reading the last-good snapshot.
/// `tenant_ids` is the distinct set known to `AuthState` at startup (ADR-063 §1).
#[cfg(feature = "enterprise")]
pub fn spawn_poller(
    cfg: PollerConfig,
    tenant_ids: Vec<String>,
    overlay: SharedConfigOverlay,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        // One reqwest client for the task's lifetime (connection reuse). A
        // build failure degrades to "no poller" rather than crashing the
        // gateway — fail-open: the overlay stays empty ⇒ allow.
        let client = match reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
        {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!(
                    "cp-config poll: failed to build HTTP client ({e}); enforcement stays inert (empty overlay)"
                );
                return;
            }
        };

        // Cold-start seed, then loop on the interval.
        let mut ticker = tokio::time::interval(cfg.interval);
        loop {
            ticker.tick().await;
            let fresh = build_overlay(&client, &cfg, &tenant_ids).await;
            tracing::debug!(
                "cp-config poll: refreshed overlay tenants={} entries={}",
                fresh.tenant_count(),
                fresh.entry_count()
            );
            overlay.store(Arc::new(fresh));
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sample of the CP `GET /tenants/{id}/model-configs` wire shape — the full
    /// shape (`tenant_id`, `billing_path`, `updated_at`) is present to prove the
    /// builder tolerates the extra fields it does not enforce.
    fn sample_rows() -> Vec<CpModelConfig> {
        serde_json::from_value(serde_json::json!([
            {
                "tenant_id": "tenant-a",
                "model_id": "blocked-model",
                "enabled": false,
                "billing_path": "byo",
                "updated_at": "2026-06-27T00:00:00Z"
            },
            {
                "tenant_id": "tenant-a",
                "model_id": "gpt-4o",
                "enabled": true,
                "billing_path": "credits",
                "updated_at": "2026-06-27T00:00:00Z"
            }
        ]))
        .expect("CP model-configs sample deserializes")
    }

    #[test]
    fn builder_maps_disabled_enabled_and_absent() {
        let overlay = ConfigOverlay::from_tenant_configs([("tenant-a".to_string(), sample_rows())]);

        // Explicitly disabled ⇒ Some(false) (the only reject condition).
        assert_eq!(
            overlay.model_enabled("tenant-a", "blocked-model"),
            Some(false)
        );
        // Explicitly enabled ⇒ Some(true).
        assert_eq!(overlay.model_enabled("tenant-a", "gpt-4o"), Some(true));
        // Absent model for a known tenant ⇒ None (default-allow).
        assert_eq!(overlay.model_enabled("tenant-a", "claude-3-5-sonnet"), None);
        // Unknown tenant ⇒ None (default-allow), even for a model another tenant
        // disabled (enforcement is tenant-scoped).
        assert_eq!(overlay.model_enabled("tenant-b", "blocked-model"), None);

        assert_eq!(overlay.tenant_count(), 1);
        assert_eq!(overlay.entry_count(), 2);
    }

    #[test]
    fn empty_overlay_allows_everything() {
        let overlay = ConfigOverlay::empty();
        // Every lookup misses ⇒ None ⇒ allow. This is the off-by-default /
        // fail-open posture: no enforcement against an empty overlay.
        assert_eq!(overlay.model_enabled("tenant-a", "blocked-model"), None);
        assert_eq!(overlay.model_enabled("any", "any"), None);
        assert_eq!(overlay.tenant_count(), 0);
        assert_eq!(overlay.entry_count(), 0);
    }

    #[test]
    fn last_row_wins_within_a_tenant() {
        let rows = vec![
            CpModelConfig {
                model_id: "m".into(),
                enabled: true,
            },
            CpModelConfig {
                model_id: "m".into(),
                enabled: false,
            },
        ];
        let overlay = ConfigOverlay::from_tenant_configs([("t".to_string(), rows)]);
        assert_eq!(overlay.model_enabled("t", "m"), Some(false));
    }

    #[test]
    fn tenants_are_isolated() {
        let a = vec![CpModelConfig {
            model_id: "shared".into(),
            enabled: false,
        }];
        let b = vec![CpModelConfig {
            model_id: "shared".into(),
            enabled: true,
        }];
        let overlay =
            ConfigOverlay::from_tenant_configs([("a".to_string(), a), ("b".to_string(), b)]);
        assert_eq!(overlay.model_enabled("a", "shared"), Some(false));
        assert_eq!(overlay.model_enabled("b", "shared"), Some(true));
    }
}
