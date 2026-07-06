use axum::extract::State;
use axum::http::{HeaderName, HeaderValue, Method};
use axum::{
    error_handling::HandleErrorLayer,
    http::StatusCode,
    middleware,
    response::IntoResponse,
    routing::{get, post},
    BoxError, Router,
};
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tower::{limit::ConcurrencyLimitLayer, load_shed::LoadShedLayer, ServiceBuilder};
use tower_http::cors::{AllowOrigin, CorsLayer};
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::set_header::SetResponseHeaderLayer;
use tower_http::timeout::TimeoutLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

/// App-wide CORS for the Routeplane Console (ADR-061).
///
/// The Console fetches with `credentials: "include"` (ADR-039: in prod it carries
/// the CP session cookie). The CORS spec FORBIDS `Access-Control-Allow-Origin: *`
/// on a credentialed response — the browser silently blocks reading it — so a
/// wildcard origin breaks every authed page in-browser (curl never enforces this,
/// which is why a `*` policy passed CLI checks but failed live). We therefore use
/// a specific-origin allow-list with `Access-Control-Allow-Credentials: true` and
/// an EXPLICIT allowed/exposed-header list (wildcards are likewise illegal
/// alongside credentials). The layer is applied OUTERMOST so the preflight
/// short-circuits before auth/reliability.
///
/// Origin policy is FAIL-CLOSED: `RP_CORS_ALLOWED_ORIGINS` (comma-separated)
/// pins the exact allowed origins; UNSET ⇒ no origin is ever allowed (the
/// browser blocks every cross-origin read — same-origin/curl/SDK traffic is
/// unaffected). The bundled Community Edition Console is served from THIS
/// origin (`RP_CONSOLE_DIR`, single image), so it never needs CORS and works
/// out of the box under the fail-closed default. Reflect-any-origin — which
/// pairs an arbitrary attacker origin with `allow-credentials: true` — is
/// available ONLY behind the explicit `RP_CORS_DEV_MODE=on` escape hatch, for
/// local dev (e.g. the Console's Vite dev server on its own port); NEVER set
/// it on an internet-facing deployment.
fn build_cors_layer() -> CorsLayer {
    let base = CorsLayer::new()
        .allow_methods([
            Method::GET,
            Method::POST,
            Method::PUT,
            Method::PATCH,
            Method::DELETE,
            Method::OPTIONS,
        ])
        // Explicit list (mandatory with credentials): every request header the
        // surface accepts — branded `x-routeplane-*` + the standard auth/body
        // ones. Each entry corresponds to a real request-header read in the
        // handlers (proxy.rs / prompts_api.rs / the idempotency layer).
        .allow_headers([
            HeaderName::from_static("content-type"),
            HeaderName::from_static("authorization"),
            HeaderName::from_static("idempotency-key"),
            HeaderName::from_static("x-routeplane-api-key"),
            HeaderName::from_static("x-routeplane-provider"),
            HeaderName::from_static("x-routeplane-residency"),
            HeaderName::from_static("x-routeplane-strategy"),
            HeaderName::from_static("x-routeplane-trace-id"),
            HeaderName::from_static("x-routeplane-config"),
            HeaderName::from_static("x-routeplane-timeout-ms"),
            HeaderName::from_static("x-routeplane-cache-control"),
            HeaderName::from_static("x-routeplane-cohort"),
            HeaderName::from_static("x-routeplane-output-mask"),
            HeaderName::from_static("x-routeplane-currency"),
            HeaderName::from_static("x-routeplane-metadata"),
            HeaderName::from_static("x-routeplane-pii-mode"),
            HeaderName::from_static("x-routeplane-use-case"),
            HeaderName::from_static("x-routeplane-idempotency-key"),
        ])
        // Response headers a browser client may READ (without this, CORS hides
        // everything but the safelist): the provenance trio, the cache/hedge/
        // replay/guardrail/shed markers, and the rate-limit + budget advisory
        // set. Each entry corresponds to a real response-header insert.
        .expose_headers([
            HeaderName::from_static("x-routeplane-provider"),
            HeaderName::from_static("x-routeplane-trace-id"),
            HeaderName::from_static("x-routeplane-request-id"),
            HeaderName::from_static("x-routeplane-cache"),
            HeaderName::from_static("x-routeplane-cache-degraded"),
            HeaderName::from_static("x-routeplane-hedged"),
            HeaderName::from_static("x-routeplane-idempotent-replayed"),
            HeaderName::from_static("x-routeplane-guardrails"),
            HeaderName::from_static("x-routeplane-shed"),
            HeaderName::from_static("x-routeplane-limit-type"),
            HeaderName::from_static("x-routeplane-limit-scope"),
            HeaderName::from_static("x-routeplane-limit-policy"),
            HeaderName::from_static("x-routeplane-budget-warning"),
            HeaderName::from_static("x-routeplane-budget-remaining"),
            HeaderName::from_static("x-routeplane-compliance-warning"),
            HeaderName::from_static("retry-after"),
            HeaderName::from_static("x-ratelimit-limit-requests"),
            HeaderName::from_static("x-ratelimit-remaining-requests"),
            HeaderName::from_static("x-ratelimit-reset-requests"),
            HeaderName::from_static("x-ratelimit-limit-tokens"),
            HeaderName::from_static("x-ratelimit-remaining-tokens"),
            HeaderName::from_static("x-ratelimit-reset-tokens"),
        ])
        .allow_credentials(true);
    let dev_mode = std::env::var("RP_CORS_DEV_MODE")
        .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "on" | "true" | "1"))
        .unwrap_or(false);
    match std::env::var("RP_CORS_ALLOWED_ORIGINS") {
        Ok(v) if !v.trim().is_empty() => {
            let origins: Vec<HeaderValue> = v
                .split(',')
                .filter_map(|s| s.trim().parse::<HeaderValue>().ok())
                .collect();
            tracing::info!("CORS: allow-list ({} origin(s))", origins.len());
            base.allow_origin(AllowOrigin::list(origins))
        }
        // Explicit dev escape hatch ONLY: reflect the caller's Origin
        // (credential-safe in form — a specific origin, never `*` — but it
        // allows ANY origin, so it is local/dev-only).
        _ if dev_mode => {
            tracing::warn!(
                "CORS: DEV reflect-any-origin (RP_CORS_DEV_MODE=on — never set on an internet-facing deployment)"
            );
            base.allow_origin(AllowOrigin::mirror_request())
        }
        // Fail-closed default: an empty allow-list emits no
        // `Access-Control-Allow-Origin`, so every cross-origin browser read is
        // blocked until origins are pinned via RP_CORS_ALLOWED_ORIGINS. The
        // bundled Console is same-origin and unaffected.
        _ => {
            tracing::info!(
                "CORS: closed (no cross-origin access; set RP_CORS_ALLOWED_ORIGINS to pin origins, or RP_CORS_DEV_MODE=on for local dev)"
            );
            base.allow_origin(AllowOrigin::list([]))
        }
    }
}

mod analytics_api;
mod api_error;
mod audio_api;
mod auth;
mod cache_api;
// CE slot types (PRD-047 / ADR-088): compiled ONLY under --no-default-features.
#[cfg(not(feature = "enterprise"))]
mod ce_stubs;
mod config;
mod config_overlay;
mod console_accounts;
mod console_api;
mod console_auth;
mod custom_providers;
mod embeddings;
mod feedback_api;
mod finops_api;
mod guardrails;
mod images_api;
mod ledger_sink;
mod logs_api;
mod messages_api;
mod metrics;
mod models_api;
mod observability;
mod otel;
mod prompts_api;
mod provenance;
mod providers_api;
mod proxy;
mod rerank_api;
mod residency_api;
mod status;

use crate::config::{CacheSettings, DeadlineConfig, GuardrailWebhookLimits, ServerLimits};
use crate::proxy::{build_provider_registry, chat_completions, AppState};

/// Process-global count of ingress requests SHED under capacity saturation
/// (ADR-025 §3). A platform SRE metric (not per-tenant), lock-free, read at
/// `GET /metrics`. Distinct from ADR-023 entitlement rate-limiting (429): a
/// shed is the server protecting its P99, not the client exceeding a quota.
static SHED_TOTAL: AtomicU64 = AtomicU64::new(0);

/// Read the cumulative load-shed count (for `/metrics` and tests).
pub fn shed_total() -> u64 {
    SHED_TOTAL.load(Ordering::Relaxed)
}
use routeplane_flags::UnleashFlags;
use routeplane_limits::{KeyLimitsInput, LimitRegistry};
use routeplane_residency::ResidencyEngine;
use routeplane_router::HealthTracker;
// CE compile-out seam (PRD-047 / ADR-088): the same name resolves to the real
// semantic cache under `enterprise` and to the inert stub otherwise.
#[cfg(not(feature = "enterprise"))]
use crate::ce_stubs::SemanticCache;
#[cfg(feature = "enterprise")]
use routeplane_semantic_cache::SemanticCache;
// Note: `Router` (above) is axum's; the routing crate's router is referenced
// fully-qualified below to avoid the name clash.
use crate::auth::{
    auth_failure_tracker_from_env, auth_middleware, global_holdbacks_from_env, shared_auth_state,
    AuthState,
};
use crate::guardrails::GuardrailEngine;
// Reversible PII tokenization (ADR-044) is a moat surface — the key holder rides
// the `enterprise` feature (it wraps the advanced crate's `tokenize::Tokenizer`).
#[cfg(feature = "enterprise")]
use crate::guardrails::TokenizerKey;
use crate::observability::ObservabilityEngine;

#[tokio::main]
async fn main() {
    // Load .env file if it exists
    dotenvy::dotenv().ok();

    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "routeplane=debug,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Gateway-level rollout holdbacks (branching-and-devex.md §6.4): the
    // operational `released(...)` half, parsed ONCE at startup from
    // RP_ROLLOUT_HOLDBACKS and subtracted for every tenant at resolution. An
    // unknown feature key refuses to start (fail-closed): a typo'd holdback
    // silently NOT holding back a half-finished feature is a release-control
    // failure.
    let global_holdbacks = match global_holdbacks_from_env() {
        Ok(set) => set,
        Err(e) => {
            tracing::error!("failed to load global rollout holdbacks: {e}");
            std::process::exit(1);
        }
    };
    tracing::info!(
        "global rollout holdbacks: count={} features={:?}",
        global_holdbacks.len(),
        global_holdbacks
            .iter()
            .map(|f| f.flag_key())
            .collect::<Vec<_>>(),
    );

    // Auth state, hot-swappable via ArcSwap (Task #7): readers never lock, and a
    // future control-plane key push can swap the whole registry atomically. The
    // global holdback set rides the same snapshot, so a future control-plane
    // push swaps registry + holdbacks together. Tenant-default Guardrails v2
    // specs (G2.6) are compiled inside load_from_file — an invalid spec refuses
    // to start (fail-closed, same doctrine as a typo'd holdback).
    // Key-registry source precedence: RP_KEYS_JSON env (raw JSON, or base64 of
    // it — ACA/K8s secret-friendly) > RP_KEYS_FILE path > ./configs/keys.json.
    // The env path exists because keys.json is gitignored (real key material):
    // a CI-built image carries no key file, so serverless deploys inject the
    // registry as configuration. Fail-closed semantics are identical on every
    // path: empty/invalid registry refuses startup.
    let load_result = match std::env::var("RP_KEYS_JSON") {
        Ok(raw) if !raw.trim().is_empty() => {
            use base64::Engine as _;
            let json = match base64::engine::general_purpose::STANDARD.decode(raw.trim()) {
                Ok(bytes) => String::from_utf8(bytes).unwrap_or(raw),
                Err(_) => raw, // not base64 -> treat as raw JSON
            };
            AuthState::load_from_json(&json, "env:RP_KEYS_JSON")
        }
        _ => {
            let path =
                std::env::var("RP_KEYS_FILE").unwrap_or_else(|_| "configs/keys.json".to_string());
            AuthState::load_from_file(&path)
        }
    };
    // Unleash flag source (ADR-029 G3 / PRD-013 FR-2 + FR-5), env-gated on
    // UNLEASH_API_URL. Absent ⇒ None ⇒ the auth path is byte-identical (no
    // dynamic holdbacks; the ab_parity golden guard stays green). The client is
    // built in string-feature mode and evaluates a LOCAL snapshot — no per-request
    // network. At startup (FR-5) we seed that snapshot (cold-start image-seed)
    // and spawn the background poller off the hot path, so the Unleash server can
    // scale to zero (ADR-029 D2). The client token comes from UNLEASH_CLIENT_TOKEN
    // (Key Vault → env via MSI in the deploy; PRD-014 FR-2) and is never logged.
    // A malformed config refuses to start — the same fail-on-bad-config doctrine
    // as the key registry / routing policies above.
    let unleash: Option<UnleashFlags> = match std::env::var("UNLEASH_API_URL") {
        Ok(url) if !url.trim().is_empty() => {
            let url = url.trim();
            let app_name =
                std::env::var("UNLEASH_APP_NAME").unwrap_or_else(|_| "routeplane".to_string());
            let instance_id = std::env::var("UNLEASH_INSTANCE_ID")
                .unwrap_or_else(|_| "routeplane-dataplane".to_string());
            let token = std::env::var("UNLEASH_CLIENT_TOKEN")
                .ok()
                .filter(|t| !t.trim().is_empty());
            let has_token = token.is_some();
            match UnleashFlags::new(url, &app_name, &instance_id, token) {
                Ok(flags) => {
                    // FR-5: cold-start seed THEN spawn the background poller. The
                    // seed lets the gateway serve a known toggle set immediately
                    // (before the first poll); the poller refreshes it off the hot
                    // path. The poller's JoinHandle is intentionally detached — the
                    // task runs for the process lifetime and a transient server
                    // outage only means "no refresh" (eval continues from the
                    // seed), the scale-to-zero design.
                    seed_unleash_snapshot(&flags);
                    let _poller = flags.spawn_refresh();
                    tracing::info!(
                        "unleash flag plane: ENABLED (url={url} app={app_name} instance={instance_id} client_token={})",
                        if has_token { "set" } else { "absent" }
                    );
                    Some(flags)
                }
                Err(e) => {
                    tracing::error!("failed to build the Unleash client: {e}");
                    std::process::exit(1);
                }
            }
        }
        _ => {
            tracing::info!(
                "unleash flag plane: disabled (set UNLEASH_API_URL to enable; auth resolves entitlements only)"
            );
            None
        }
    };

    let loaded_auth = match load_result {
        Ok(mut state) => {
            state.global_holdbacks = global_holdbacks;
            // Attach the (optional) Unleash flag source onto the same snapshot as
            // the static holdbacks (FR-2) — they compose together at auth.
            state.unleash = unleash;
            state
        }
        Err(e) => {
            // Refuse to start on a missing/invalid/empty key registry (Task #3b):
            // a gateway with no keys authenticates nobody and would 401 every
            // request — fail loud at boot instead of running broken.
            tracing::error!("failed to load key registry: {e}");
            std::process::exit(1);
        }
    };

    // Budgets & rate limits (PRD-008 / ADR-023 Mode L): build the counter
    // registry from the SAME loaded keys, before wrapping AuthState in its
    // ArcSwap handle. Keys with no `limits` field contribute nothing (unlimited
    // ⇒ byte-identical). Resolution is by routeplane_key (key scope) and
    // tenant_id (tenant scope, shared across the tenant's keys).
    // Sorted by routeplane_key so the registry's first-wins tenant-policy
    // registration is DETERMINISTIC across restarts (HashMap iteration order is
    // not) — security-review hardening.
    let mut limit_inputs: Vec<KeyLimitsInput> = loaded_auth
        .keys
        .values()
        .filter_map(|vk| {
            vk.limits.clone().map(|l| KeyLimitsInput {
                routeplane_key: vk.routeplane_key.clone(),
                tenant_id: vk.resolved_tenant_id(),
                limits: l,
            })
        })
        .collect();
    limit_inputs.sort_by(|a, b| a.routeplane_key.cmp(&b.routeplane_key));
    // FR-22 fail-closed validation (multi-currency budgets): a budget authored in
    // both micro-USD and a display currency, or an authored cap with a missing/zero
    // pinned FX rate, is rejected at BOOT — the same refuse-to-start doctrine as the
    // key registry. A silently-uncapped or deny-all $0 cap is a money bug; surface
    // it loud at startup, off the hot path (the registry builder itself falls back
    // to a deny-all $0 cap, but this pass guarantees a serving registry only ever
    // holds a validated budget config).
    for input in &limit_inputs {
        for (scope, policy) in [
            ("key", input.limits.key.as_ref()),
            ("tenant", input.limits.tenant.as_ref()),
        ] {
            if let Some(budget) = policy.and_then(|p| p.budget.as_ref()) {
                if let Err(e) = budget.resolved_cost_caps() {
                    tracing::error!(
                        "invalid budget config (key={} scope={scope}): {e}",
                        input.routeplane_key
                    );
                    std::process::exit(1);
                }
            }
        }
    }
    // Mode-L share-clamp (ADR-023 §6): each stateless replica enforces only
    // ceil(limit / max_replicas) so the fleet-wide cap is ~the configured value
    // instead of N× it under scale-out. Defaults to 1 (no clamp) when unset.
    let max_replicas: u32 = std::env::var("ROUTEPLANE_MAX_REPLICAS")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|n| *n >= 1)
        .unwrap_or(1);
    let limits = LimitRegistry::build_with_replicas(limit_inputs, max_replicas);
    let configured_limit_scopes = loaded_auth
        .keys
        .values()
        .filter(|vk| vk.limits.is_some())
        .count();
    tracing::info!(
        "budgets & rate limits: enforcement=local-only (ADR-023 Mode L) keys_with_policies={}",
        configured_limit_scopes
    );

    // ADR-064: the merge base for CP→DP rate-limit distribution — EVERY auth key +
    // its boot limits (so a tenant-scoped CP limit applies across all the tenant's
    // keys). Built now while `loaded_auth` is live; consumed by the gated
    // distributor spawned after `AppState` (it needs the `Arc<AppState>` handle).
    // ENTERPRISE-ONLY (PRD-047): CE has no control plane to distribute from.
    #[cfg(feature = "enterprise")]
    let limit_base_keys: Vec<crate::limit_distribution::BaseKey> = loaded_auth
        .keys
        .values()
        .map(|vk| crate::limit_distribution::BaseKey {
            routeplane_key: vk.routeplane_key.clone(),
            tenant_id: vk.resolved_tenant_id(),
            base: vk.limits.clone().unwrap_or_default(),
        })
        .collect();

    // CP→DP runtime config distribution (ADR-063 / PRD-039) — the per-tenant
    // model-enablement overlay + its background poller. The overlay handle is
    // ALWAYS built (empty); the poller starts ONLY when RP_CP_CONFIG_URL is set
    // (PollerConfig::from_env returns Some). Absent ⇒ no task, an empty overlay,
    // enforcement is a permanent no-op ⇒ the gateway is byte-identical to today
    // (the ab_parity golden guard runs with the env unset). The poller polls each
    // DISTINCT tenant_id known to AuthState (ADR-063 §1) — collected here from the
    // loaded keys BEFORE AuthState moves into its ArcSwap handle. Fail-open: a poll
    // failure keeps the last-good overlay (the task never panics, never rejects).
    let config_overlay = crate::config_overlay::new_shared_empty();
    // ENTERPRISE-ONLY (PRD-047): without the poller the overlay stays empty for
    // the process lifetime ⇒ enforcement is a permanent no-op on the CE build.
    #[cfg(feature = "enterprise")]
    match crate::cp_config::PollerConfig::from_env() {
        Some(cfg) => {
            let mut tenant_ids: Vec<String> = loaded_auth
                .keys
                .values()
                .map(|vk| vk.resolved_tenant_id())
                .collect();
            tenant_ids.sort();
            tenant_ids.dedup();
            // The auth mode (managed-identity / dev-token / none) is logged by
            // `PollerConfig::from_env` itself (ADR-066); the credential is never logged.
            tracing::info!(
                "cp-config distribution: ENABLED (poll_secs={} tenants={})",
                cfg.interval.as_secs(),
                tenant_ids.len(),
            );
            let _poller =
                crate::config_overlay::spawn_poller(cfg, tenant_ids, config_overlay.clone());
        }
        None => {
            tracing::info!(
                "cp-config distribution: disabled (set RP_CP_CONFIG_URL to enable; overlay stays empty ⇒ no model-enablement enforcement)"
            );
        }
    }

    let auth_state = shared_auth_state(loaded_auth);

    // ── CE Console email+password auth (console-session bridge) ─────────────
    // Accounts: RP_CONSOLE_ACCOUNTS_FILE > ./configs/console-accounts.json.
    // ABSENT/empty ⇒ start empty (open signup creates the first operator
    // account — approval/invite gating is Enterprise, not built here);
    // PRESENT-but-invalid ⇒ refuse start (holds password hashes — the
    // keys.json fail-closed doctrine). 0600 + gitignored + dockerignored.
    let console_accounts_path = std::env::var("RP_CONSOLE_ACCOUNTS_FILE")
        .unwrap_or_else(|_| "configs/console-accounts.json".to_string());
    let console_accounts = match crate::console_accounts::ConsoleAccountStore::load(
        std::path::PathBuf::from(&console_accounts_path),
    ) {
        Ok(store) => {
            tracing::info!(
                "console accounts: {} account(s) loaded from {console_accounts_path}",
                store.len()
            );
            Arc::new(store)
        }
        Err(e) => {
            tracing::error!("failed to load console accounts from {console_accounts_path}: {e}");
            std::process::exit(1);
        }
    };
    // Session-signing secret: RP_CONSOLE_SESSION_SECRET (stable across
    // restarts) or a per-boot random secret. NEVER logged. A CSPRNG failure
    // refuses to start (a guessable session secret is an auth bypass).
    let console_secret = match crate::console_auth::session_secret_from_env() {
        Ok((secret, generated)) => {
            if generated {
                tracing::warn!(
                    "RP_CONSOLE_SESSION_SECRET is not set; using a random per-boot secret — console sessions reset on every restart"
                );
            }
            secret
        }
        Err(e) => {
            tracing::error!("failed to initialise the console session secret: {e}");
            std::process::exit(1);
        }
    };
    // The gateway key a valid console session authorizes as (single-tenant
    // CE). RP_CONSOLE_KEY must name a REGISTERED key when set (fail-closed at
    // boot — a console silently mapping to nothing, or to a guessed key, is an
    // authz failure); default = the registry's only key, else the
    // lexicographically-first routeplane_key (deterministic across restarts —
    // HashMap iteration order is not). Only the key NAME is ever logged.
    let console_key = {
        let snapshot = auth_state.load();
        let resolved = match std::env::var("RP_CONSOLE_KEY") {
            Ok(k) if !k.trim().is_empty() => {
                let k = k.trim().to_string();
                if !snapshot.keys.contains_key(&k) {
                    tracing::error!(
                        "RP_CONSOLE_KEY does not match any key in the registry (refusing to start)"
                    );
                    std::process::exit(1);
                }
                k
            }
            _ => {
                let mut ids: Vec<&String> = snapshot.keys.keys().collect();
                ids.sort_unstable();
                match ids.first() {
                    Some(id) => {
                        if ids.len() > 1 {
                            tracing::info!(
                                "multiple gateway keys registered; console sessions default to the first (set RP_CONSOLE_KEY to choose)"
                            );
                        }
                        (*id).clone()
                    }
                    // Unreachable in practice (an empty registry refuses to
                    // start above) — but never panic on the boot path either.
                    None => {
                        tracing::error!("key registry is empty; cannot bind a console key");
                        std::process::exit(1);
                    }
                }
            }
        };
        let key_name = snapshot
            .keys
            .get(&resolved)
            .map(|vk| vk.name.clone())
            .unwrap_or_default();
        tracing::info!("console sessions authorize as key '{key_name}'");
        resolved
    };
    let console_bridge: crate::console_auth::SharedConsoleAuth =
        Arc::new(crate::console_auth::ConsoleAuthBridge::new(
            &console_secret,
            console_key,
            console_accounts.clone(),
        ));

    // Auth-failure rate limiting (security gap R0.2): throttle repeated failed
    // authentication per source IP with escalating backoff. Ship-dark — OFF
    // unless RP_AUTH_FAILURE_LIMIT is truthy, so the default auth path is
    // byte-identical (zero-cost when disabled). The tracker is a lock-free,
    // fixed-memory atomic-slot map (bounded; no per-IP growth) and rides the
    // authed router as an Extension, the same pattern as auth_state.
    let auth_failure_tracker = auth_failure_tracker_from_env();
    match auth_failure_tracker.as_ref() {
        Some(t) => {
            let c = t.config();
            tracing::info!(
                "auth-failure rate limiting: ENABLED (threshold={} window_ms={} backoff_base_ms={} backoff_cap_ms={} slots={})",
                c.threshold,
                c.window_ms,
                c.backoff_base_ms,
                c.backoff_cap_ms,
                c.slots,
            );
        }
        None => tracing::info!(
            "auth-failure rate limiting: disabled (set RP_AUTH_FAILURE_LIMIT=on to enable)"
        ),
    }

    // Saved routing-policy configs (G2.2 / ADR-021 §3): optional at this stage —
    // an absent configs/routing-policies.json simply means no cfg_ references
    // resolve (inline configs still work). A PRESENT-but-invalid file refuses
    // startup (fail-closed, same doctrine as the key registry).
    let policy_path = std::env::var("RP_ROUTING_POLICIES_FILE")
        .unwrap_or_else(|_| "configs/routing-policies.json".to_string());
    let policies = if std::path::Path::new(&policy_path).exists() {
        match routeplane_policy::load_registry_from_file(&policy_path) {
            Ok(reg) => {
                tracing::info!(
                    "loaded {} saved routing config(s) from {policy_path}",
                    reg.len()
                );
                routeplane_policy::new_shared_registry(reg)
            }
            Err(e) => {
                tracing::error!("failed to load routing policies: {e}");
                std::process::exit(1);
            }
        }
    } else {
        routeplane_policy::new_shared_registry(routeplane_policy::PolicyRegistry::new())
    };

    // Prompt registry (PRD-010 / G3.5): optional git-file snapshot, hot-swappable
    // via ArcSwap, served lock-free on the render path. Source precedence:
    // RP_PROMPTS_FILE > ./configs/prompts.json. ABSENT file ⇒ empty registry
    // (every /v1/prompts ref 404s; the endpoints still gate on the entitlement).
    // PRESENT-but-invalid ⇒ refuse startup (fail-closed, FR-11 — same doctrine as
    // the key registry and routing policies). The per-tenant cap is the tfvars/env
    // parameter RP_PROMPTS_MAX_PER_TENANT (default 1000).
    let prompt_bounds = routeplane_prompts::Bounds::from_env();
    let prompts_path =
        std::env::var("RP_PROMPTS_FILE").unwrap_or_else(|_| "configs/prompts.json".to_string());
    let prompts = if std::path::Path::new(&prompts_path).exists() {
        match routeplane_prompts::PromptRegistry::load_from_file(&prompts_path, &prompt_bounds) {
            Ok(reg) => {
                tracing::info!(
                    "loaded prompt registry from {prompts_path}: tenants={} prompts={}",
                    reg.tenant_count(),
                    reg.prompt_count()
                );
                routeplane_prompts::new_shared_registry(reg)
            }
            Err(e) => {
                tracing::error!("failed to load prompt registry: {e}");
                std::process::exit(1);
            }
        }
    } else {
        tracing::info!(
            "no prompt registry at {prompts_path}; /v1/prompts serves an empty registry"
        );
        routeplane_prompts::new_shared_registry(routeplane_prompts::PromptRegistry::empty())
    };

    // Agent identity registry (ADR-017) for the MCP gateway — per-agent scoped
    // grants + run-limits. RP_AGENTS_FILE > ./configs/agents.json; ABSENT ⇒
    // empty registry (every agent is unknown ⇒ default-deny). Malformed-if-
    // present ⇒ refuse start (an authz registry that silently fails open is a
    // security hole). The /v1/mcp/* routes additionally gate on the entitlement.
    // ENTERPRISE-ONLY region (PRD-047): the whole MCP bootstrap — agent registry,
    // manifest pins, run registry, egress resolver — is compiled out on CE.
    #[cfg(feature = "enterprise")]
    let agents_path =
        std::env::var("RP_AGENTS_FILE").unwrap_or_else(|_| "configs/agents.json".to_string());
    #[cfg(feature = "enterprise")]
    let agents = if std::path::Path::new(&agents_path).exists() {
        match std::fs::read_to_string(&agents_path)
            .map_err(|e| e.to_string())
            .and_then(|s| {
                routeplane_mcp::registry::AgentRegistry::from_json(&s).map_err(|e| e.to_string())
            }) {
            Ok(reg) => {
                tracing::info!(
                    "loaded agent registry from {agents_path}: agents={}",
                    reg.len()
                );
                std::sync::Arc::new(reg)
            }
            Err(e) => {
                tracing::error!("failed to load agent registry: {e}");
                std::process::exit(1);
            }
        }
    } else {
        tracing::info!("no agent registry at {agents_path}; MCP agents default-deny");
        std::sync::Arc::new(routeplane_mcp::registry::AgentRegistry::new())
    };

    // MCP server manifest pin registry (ADR-016 rug-pull protection). RP_MANIFESTS_FILE
    // > ./configs/manifests.json; ABSENT ⇒ empty (every server unpinned ⇒ no pin
    // enforcement; grants + egress still apply). Malformed-if-present ⇒ refuse start
    // (a pin registry that silently fails open is a security hole).
    #[cfg(feature = "enterprise")]
    let manifests_path =
        std::env::var("RP_MANIFESTS_FILE").unwrap_or_else(|_| "configs/manifests.json".to_string());
    #[cfg(feature = "enterprise")]
    let manifests = if std::path::Path::new(&manifests_path).exists() {
        match std::fs::read_to_string(&manifests_path)
            .map_err(|e| e.to_string())
            .and_then(|s| {
                routeplane_mcp::manifest::ManifestRegistry::from_json(&s).map_err(|e| e.to_string())
            }) {
            Ok(reg) => {
                tracing::info!(
                    "loaded manifest pin registry from {manifests_path}: pins={}",
                    reg.len()
                );
                std::sync::Arc::new(reg)
            }
            Err(e) => {
                tracing::error!("failed to load manifest pin registry: {e}");
                std::process::exit(1);
            }
        }
    } else {
        tracing::info!("no manifest pin registry at {manifests_path}; MCP servers unpinned");
        std::sync::Arc::new(routeplane_mcp::manifest::ManifestRegistry::new())
    };

    // Live agent-run state (ADR-017 per-run breakers). Runtime state, not config
    // — starts empty; runs are created lazily on first /v1/mcp/run/step.
    #[cfg(feature = "enterprise")]
    let run_registry = std::sync::Arc::new(routeplane_mcp::run::RunRegistry::new());

    // Binary-side DNS resolver for the MCP egress guard's resolve-then-pin leg
    // (ADR-016 §7): the `async` layer-3 SSRF defense that catches a DOMAIN host
    // fronting an internal/metadata IP. Injected into /v1/mcp/tool-call/authorize
    // as an Extension (tests swap in a hermetic stub).
    #[cfg(feature = "enterprise")]
    let egress_resolver: crate::mcp_api::SharedResolver =
        std::sync::Arc::new(crate::mcp_api::TokioResolver);

    // Sovereign audit ledger (PRD-001 / ADR-019) — ship-dark by default:
    // RP_AUDIT_LEDGER must be explicitly enabled, and writes are additionally
    // gated per tenant on the `audit_ledger` capability. When ENABLED but
    // misconfigured we refuse to start: an evidence ledger that silently is
    // not writing is worse than no ledger.
    #[cfg(feature = "enterprise")]
    let ledger = match routeplane_ledger::bootstrap::dataplane_ledger_from_env() {
        Ok(Some(handle)) => {
            tracing::info!("sovereign audit ledger: ENABLED");
            Some(handle)
        }
        Ok(None) => {
            tracing::info!("sovereign audit ledger: disabled (set RP_AUDIT_LEDGER=on to enable)");
            None
        }
        Err(e) => {
            tracing::error!("failed to initialise the sovereign audit ledger: {e}");
            std::process::exit(1);
        }
    };
    // CE build (PRD-047 / ADR-088): the ledger crate is absent; the slot type is
    // uninhabited so this binding is permanently `None` (same name — the
    // AppState literal below is identical on both variants).
    #[cfg(not(feature = "enterprise"))]
    let ledger: Option<crate::ledger_sink::LedgerHandle> = None;

    // PRD-009 / ADR-024 durable telemetry writer — ship-dark: `None` unless
    // RP_TELEMETRY_DURABLE is set. Gated per-tenant on `Feature::TelemetryDurable`
    // at the record site, so the default build emits zero durable telemetry and
    // is byte-identical (the free tier stays the in-memory ring at $0).
    #[cfg(feature = "enterprise")]
    let telemetry = routeplane_telemetry::bootstrap::dataplane_telemetry_from_env();
    #[cfg(feature = "enterprise")]
    if telemetry.is_some() {
        tracing::info!("durable telemetry: ENABLED");
    } else {
        tracing::info!("durable telemetry: disabled (set RP_TELEMETRY_DURABLE=on to enable)");
    }
    // CE build (PRD-047 / ADR-088): the telemetry crate is absent; the slot type
    // is uninhabited so this binding is permanently `None` (same name — the
    // AppState literal below is identical on both variants). Honest log: the
    // env flag does nothing here, so it is not suggested.
    #[cfg(not(feature = "enterprise"))]
    let telemetry: Option<crate::ce_stubs::TelemetryHandle> = None;
    #[cfg(not(feature = "enterprise"))]
    tracing::info!("durable telemetry: not included in this build (Community Edition)");

    // MCP agentic-security deepening engines (ADR-055) — anomaly detection,
    // sampling-attack defense, HITL approval queue, signed action receipts. All
    // OFF the synchronous chat path (driven only by the gated /v1/mcp/* surface).
    // The receipt issuer REUSES the platform's one signer seam (the same Key
    // Vault / test signer the audit ledger uses), so receipts and checkpoints
    // share key custody. A signer-build failure is non-fatal here: the receipt
    // routes degrade to `receipts_unavailable` rather than refusing to start (the
    // engines are off the request path; a missing receipt is a loggable integrity
    // gap, not a reason to take the gateway down). The DEFAULT run has no signer
    // configured → no receipts → no Key Vault needed (ship-dark).
    // ENTERPRISE-ONLY region (PRD-047): receipts + the agentic engines ride the
    // gated /v1/mcp/* surface, absent on CE.
    #[cfg(feature = "enterprise")]
    let receipt_signer = match routeplane_ledger::bootstrap::artifact_signer_from_env() {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(
                "MCP receipt signer unavailable ({e}); /v1/mcp/receipt/* will return receipts_unavailable"
            );
            None
        }
    };
    #[cfg(feature = "enterprise")]
    match receipt_signer.as_ref() {
        Some(s) => tracing::info!(
            "MCP agentic deepening: ENABLED (receipts signed via {} signer)",
            s.algorithm()
        ),
        None => tracing::info!(
            "MCP agentic deepening: ENABLED (receipts disabled — no signer configured)"
        ),
    }
    #[cfg(feature = "enterprise")]
    let mcp_agentic = Arc::new(crate::proxy::McpAgenticState::new(receipt_signer));

    let limits_server = ServerLimits::from_env();
    let deadline_config = DeadlineConfig::from_env();
    let webhook_limits = GuardrailWebhookLimits::from_env();
    // Exact-match cache (G2.5 / ADR-022 rung 0): always constructed — it is a
    // core capability with $0 standing cost. PARTICIPATION is per-request
    // opt-in via the routing config's `cache` object (PRD-007 FR-2); with no
    // config the cache is never read or written. Budget is the cell tfvars
    // parameter ROUTEPLANE_CACHE_BUDGET_BYTES (64 MiB pool-std default,
    // 16 MiB pool-free). Scale-to-zero resets it — accepted, documented.
    let cache_settings = CacheSettings::from_env();
    tracing::info!(
        "server limits: max_concurrency={} request_timeout={}ms max_body_bytes={} audio_max_body_bytes={} | deadline: request={}ms per_attempt={}ms | guardrail webhook: timeout={}ms max_bytes={} | exact cache: budget_bytes={}",
        limits_server.max_concurrency,
        limits_server.request_timeout.as_millis(),
        limits_server.max_body_bytes,
        limits_server.audio_max_body_bytes,
        deadline_config.request_deadline.as_millis(),
        deadline_config.per_attempt_timeout.as_millis(),
        webhook_limits.timeout.as_millis(),
        webhook_limits.max_response_bytes,
        cache_settings.budget_bytes,
    );

    // SIEM/warehouse export (R1.5 / ADR-054) — ship-dark by default. With no
    // RP_EXPORT_* sink configured this returns a DISABLED no-op handle (no task,
    // no channel, zero standing cost — the frugal scale-to-zero default). A
    // partial/invalid sink config (e.g. half a Sentinel triplet, an SSRF-unsafe
    // webhook) fails the boot LOUDLY, the same fail-on-bad-config doctrine as the
    // key registry / routing policies above: shipping audit-relevant telemetry
    // "nowhere" silently is worse than refusing to start.
    #[cfg(feature = "enterprise")]
    let export = match routeplane_export::exporter_from_env() {
        Ok(handle) => {
            if handle.is_enabled() {
                tracing::info!("SIEM/warehouse export: ENABLED");
            } else {
                tracing::info!(
                    "SIEM/warehouse export: disabled (set RP_EXPORT_* to enable a sink)"
                );
            }
            handle
        }
        Err(e) => {
            tracing::error!("failed to initialise SIEM/warehouse export: {e}");
            std::process::exit(1);
        }
    };
    // CE build (PRD-047 / ADR-088): the export crate is absent; the stub handle
    // is the permanently-disabled no-op (the enterprise ship-dark default).
    #[cfg(not(feature = "enterprise"))]
    let export = crate::ce_stubs::export_api::ExportHandle::disabled();

    // Off-path detector set (R1.1/R1.2 / ADR-053) — built once. DEFAULT build
    // (no ml-* cargo features, no RP_INJECTION/MODERATION_ONNX_* env) degrades to
    // the deterministic injection classifier and a no-op moderator, so the
    // adjudication is purely deterministic (µs) and off-path. A model-load error
    // degrades rather than refusing to start (an optional detector). Shared via
    // Arc; adjudication is gated at the call site on Feature::AdvancedGuardrails.
    // MOAT (ADR-088): the off-path pipeline rides `enterprise` — the CE build has
    // neither the `offpath_guard` module nor the AppState field.
    #[cfg(feature = "enterprise")]
    let offpath = Arc::new(crate::offpath_guard::OffpathDetectors::from_env());

    // Opt-in distributed (Redis) rate limiting (ADR-056 Mode D) — OFF by default.
    // On the DEFAULT build the field is the uninhabited `None` (no Redis pulled
    // in, byte-identical to Mode L). Under the `redis-limits` feature it is `Some`
    // ONLY when ROUTEPLANE_REDIS_URL is configured; `connect` is bounded so the
    // boot never hangs, and a connect FAILURE degrades to Mode L (the request
    // path must never depend on Redis being up). On any per-request Redis error
    // the limiter fails open to the local engine (see distributed.rs).
    let distributed_limiter = build_distributed_limiter().await;

    // Runtime custom-provider registry (CE operator surface): operator-defined
    // OpenAI-compatible endpoints, added over the authed /v1/providers API with
    // NO restart. Hot-swappable via ArcSwap (lock-free reads) + persisted to
    // configs/providers.json with 0600 perms (it holds upstream keys, exactly
    // like keys.json — both gitignored + dockerignored). Source precedence:
    // RP_PROVIDERS_FILE > ./configs/providers.json. ABSENT/empty file ⇒ start
    // empty (ship-dark, byte-identical); PRESENT-but-invalid ⇒ refuse start
    // (fail-closed, the keys.json doctrine).
    let providers_path =
        std::env::var("RP_PROVIDERS_FILE").unwrap_or_else(|_| "configs/providers.json".to_string());
    let custom_providers = match crate::custom_providers::CustomProviderStore::load(
        std::path::PathBuf::from(&providers_path),
    ) {
        Ok(store) => {
            tracing::info!(
                "custom provider registry: {} provider(s) loaded from {providers_path}",
                store.len()
            );
            Arc::new(store)
        }
        Err(e) => {
            tracing::error!("failed to load custom provider registry from {providers_path}: {e}");
            std::process::exit(1);
        }
    };

    let state = Arc::new(AppState {
        providers: build_provider_registry(),
        guardrail_engine: GuardrailEngine::new(),
        // ADR-044: reversible-tokenization key custody. `None` (ship-dark) unless
        // ROUTEPLANE_TOKENIZE_KEY[_HEX] is configured; then tokenize mode degrades
        // to masking. Built ONCE here — the AES key schedule is never per-request.
        // MOAT (ADR-088): reversible tokenization rides `enterprise`.
        #[cfg(feature = "enterprise")]
        tokenizer_key: TokenizerKey::from_env(),
        observability_engine: ObservabilityEngine::new(),
        residency_engine: ResidencyEngine::new(),
        health: HealthTracker::new([
            "openai",
            "anthropic",
            "gemini",
            "azure_openai",
            "mistral",
            "cohere",
            "bedrock",
            "groq",
            "deepseek",
            "together",
            "fireworks",
            "xai",
            "openrouter",
            "self_hosted",
        ]),
        router: routeplane_router::Router::with_defaults(),
        deadline_config,
        #[cfg(feature = "enterprise")]
        guardrail_webhooks: ReqwestWebhookClient::new(webhook_limits),
        limits,
        // FinOps FX rate table ([PRD-015] FR-2): ship-dark default (byte-identical
        // to the legacy placeholder) unless RP_FX_RATES_JSON/RP_FX_RATES_FILE is
        // set. $0 standing cost, no DB, hot-swappable without a restart.
        fx_rates: routeplane_limits::fx::shared_from_env(),
        ledger,
        telemetry,
        policies,
        cache: routeplane_cache::ExactCache::new(cache_settings.budget_bytes),
        // FR-19 cache-purge flush-generation registry: always constructed, $0
        // standing cost, empty (every scope at generation 0 ⇒ byte-identical
        // legacy key) until a tenant issues a purge.
        cache_flush: routeplane_cache::FlushRegistry::new(),
        // Idempotency-key store (Stripe/Portkey safe-retry): always constructed,
        // $0 standing cost, in-memory + per-replica (scale-to-zero resets it like
        // the cache). TTL overridable via RP_IDEMPOTENCY_TTL_SECONDS (default 24h);
        // participation is per-request opt-in via the `Idempotency-Key` header, so
        // a no-header request is byte-identical to today. Multi-replica coordinated
        // idempotency would be a Redis follow-on (needs an ADR).
        idempotency: idempotency_store_from_env(),
        // Rung-1 semantic cache (PRD-007 / ADR-022): always constructed, $0
        // standing cost; thresholds/capacity from env. Participation is the
        // per-request double gate (Feature::SemanticCache + CacheMode::Semantic).
        // On the CE build `SemanticCache` is the inert `ce_stubs` stand-in.
        semantic_cache: SemanticCache::from_env(),
        #[cfg(feature = "enterprise")]
        offpath,
        export,
        distributed_limiter,
        #[cfg(feature = "enterprise")]
        mcp_agentic,
        // CP→DP model-enablement overlay (ADR-063 / PRD-039). Always present;
        // initialized EMPTY (off by default). The poller below swaps it on a timer
        // ONLY when RP_CP_CONFIG_URL is set — absent ⇒ it stays empty for the
        // process lifetime ⇒ enforcement is a permanent no-op (parity-safe).
        config_overlay: config_overlay.clone(),
        // Runtime custom-provider registry (CE): loaded above; hot-swapped by
        // the /v1/providers handlers, read lock-free on the request path.
        custom_providers,
    });

    // ADR-064: CP→DP rate-limit distributor — spawned ONLY when RP_CP_CONFIG_URL is
    // set (the same gate as the model-enablement poller). Off ⇒ never runs ⇒ the
    // boot `keys.json` registry is untouched ⇒ byte-identical. When on, it polls CP
    // tenant rate-limits and budgets off the hot path and atomically swaps the (already
    // hot-swappable) LimitRegistry on a real config change (change-detected so
    // counters aren't reset every cycle). Needs the `Arc<AppState>` built above.
    // ENTERPRISE-ONLY (PRD-047): CE has no control plane to distribute from.
    #[cfg(feature = "enterprise")]
    if let Some(limit_cfg) = crate::cp_config::PollerConfig::from_env() {
        let mut tids: Vec<String> = limit_base_keys
            .iter()
            .map(|b| b.tenant_id.clone())
            .collect();
        tids.sort();
        tids.dedup();
        tracing::info!(
            "cp-limit distributor: ENABLED ({} tenants, every {:?}) — tenant-scoped rate limits + budgets",
            tids.len(),
            limit_cfg.interval,
        );
        crate::limit_distribution::spawn_limit_poller(
            limit_cfg,
            tids,
            limit_base_keys,
            state.clone(),
        );
    }

    // Cross-cutting reliability stack (Task #2), composed as Tower layers rather
    // than hand-rolled in the handler. ServiceBuilder applies layers
    // OUTERMOST-FIRST, so the order below is the request's path inward:
    //   1. RequestBodyLimitLayer — reject oversized bodies (413) before buffering.
    //   2. TimeoutLayer          — hard per-request wall-clock cap (408); a
    //      backstop above the provider-chain deadline covering the whole handler.
    //   3. HandleErrorLayer      — maps the LoadShed `Overloaded` error (a
    //      `BoxError`) into a clean HTTP 503 so a shed request is a proper
    //      response, not a dropped connection.
    //   4. LoadShedLayer         — when the concurrency limit is saturated, shed
    //      immediately (errors) instead of queueing unboundedly (backpressure,
    //      not buffering).
    //   5. ConcurrencyLimitLayer — bound in-flight requests; the limiter that
    //      LoadShed sheds against.
    let reliability = ServiceBuilder::new()
        .layer(RequestBodyLimitLayer::new(limits_server.max_body_bytes))
        .layer(TimeoutLayer::with_status_code(
            StatusCode::REQUEST_TIMEOUT,
            limits_server.request_timeout,
        ))
        .layer(HandleErrorLayer::new(handle_middleware_error))
        .layer(LoadShedLayer::new())
        .layer(ConcurrencyLimitLayer::new(limits_server.max_concurrency));

    // The audio-transcription route accepts binary audio uploads (OpenAI caps STT
    // at 25 MB), so the small global `max_body_bytes` (2 MiB, sized for chat JSON)
    // would 413 every real audio file. It rides its OWN reliability stack with a
    // LARGER `RequestBodyLimitLayer` (`RP_AUDIO_MAX_BODY_BYTES`, ~26 MiB) — the
    // global limit is unchanged for every other route. Same timeout / load-shed /
    // concurrency posture as the main stack (a separate `ServiceBuilder` because
    // a ServiceBuilder is move-consumed when applied).
    let audio_reliability = ServiceBuilder::new()
        .layer(RequestBodyLimitLayer::new(
            limits_server.audio_max_body_bytes,
        ))
        .layer(TimeoutLayer::with_status_code(
            StatusCode::REQUEST_TIMEOUT,
            limits_server.request_timeout,
        ))
        .layer(HandleErrorLayer::new(handle_middleware_error))
        .layer(LoadShedLayer::new())
        .layer(ConcurrencyLimitLayer::new(limits_server.max_concurrency));

    // Public routes — NO auth (liveness/readiness probes + root banner).
    // Health checks must never require a key, or ACA/Kubernetes probes fail.
    // The reliability stack deliberately does NOT wrap these: a load-shed must
    // never make /healthz fail and trigger a pod restart while shedding.
    // Whether the bundled Community Edition Console (static SPA) is served from
    // this origin — set by the single-image build (RP_CONSOLE_DIR → the built
    // assets). When enabled, the Console SPA owns "/" (served via the router
    // fallback below, where ServeDir returns index.html for "/"), so the
    // plain-text gateway banner is NOT mounted there. When disabled, "/" is the
    // banner and behavior is byte-identical to before.
    let console_dir: Option<String> = std::env::var("RP_CONSOLE_DIR")
        .ok()
        .filter(|d| !d.is_empty());

    let mut public = Router::new()
        .route("/healthz", get(|| async { "OK" }))
        // Platform SRE metrics (no auth, like /healthz) — the full Prometheus
        // text exposition surface (request counts/latency/tokens/cost/cache/
        // errors/hedged + the legacy capacity-shed count, ADR-025 §3). Every
        // counter is a lock-free atomic; the render allocates the body off the
        // hot path. Labels are bounded (provider only) and carry NO tenant, key,
        // or content — which is what makes leaving it unauthenticated safe.
        .route("/metrics", get(metrics_handler));
    if console_dir.is_none() {
        public = public.route(
            "/",
            get(|| async { "Routeplane AI Gateway (Rust) - Alpha" }),
        );
    }

    // Authenticated routes — require a valid x-routeplane-api-key. The auth
    // `from_fn` is preserved (Task #2) and runs inside the reliability stack.
    // The prompt registry rides an Extension layer (PRD-010 / G3.5) — the same
    // pattern as `auth_state`, keeping the proxy.rs orchestrator untouched; the
    // /v1/prompts handlers extract it via `Extension<SharedPromptRegistry>`.
    let authed = Router::new()
        .route("/analytics", get(crate::analytics_api::analytics_events))
        .route(
            "/analytics/latency",
            get(
                |axum::extract::State(state): axum::extract::State<Arc<AppState>>| async move {
                    axum::Json(state.observability_engine.latency_stats())
                },
            ),
        )
        .route("/v1/chat/completions", post(chat_completions))
        // Native Anthropic Messages API (PARITY: Portkey/LiteLLM expose Anthropic's
        // /v1/messages so an official Anthropic-SDK client can point base_url at the
        // gateway unchanged). A TRANSLATION surface — it translates the Anthropic
        // request to the canonical shape, funnels it through the SAME
        // `chat_completions_core` pipeline as /v1/chat/completions (classify-then-
        // mask, residency, guardrails, limits, routing, usage/ledger/export, cache
        // all apply), then translates the response back to the Anthropic shape.
        // Default provider `anthropic` (x-routeplane-provider overrides). Streaming
        // is rejected with a documented 400 this iteration (see messages_api.rs).
        .route("/v1/messages", post(crate::messages_api::messages))
        .route("/v1/embeddings", post(crate::embeddings::embeddings))
        // Portkey/Helicone-compatible Feedback API (PARITY: Portkey ships
        // POST /v1/feedback to attach a weighted quality score to a request
        // trace; pairs with the prompt A/B variant analytics). Authed +
        // tenant-scoped; records OFF the hot path into the in-memory
        // observability ring (no provider call, no DB). Clients reference the
        // `x-routeplane-trace-id` the proxy now emits on completions.
        .route("/v1/feedback", post(crate::feedback_api::feedback))
        // Cohere/LiteLLM-compatible reranking (PARITY: LiteLLM exposes /rerank;
        // core to RAG pipelines). Authed like the other /v1 routes; default
        // provider is `cohere` (OpenAI has no rerank endpoint).
        .route("/v1/rerank", post(crate::rerank_api::rerank))
        // FR-13 (PRD-011): /v1/responses is intentionally NOT supported. Return a
        // pointer-bearing 501 decline (not the bare unknown-route 404) so an SDK
        // gets a clean typed `endpoint_not_supported` error directing it to the
        // supported endpoint. Authed like the rest of /v1/* (no key → 401 first).
        .route(
            "/v1/responses",
            post(|| async {
                crate::api_error::endpoint_not_supported(
                    "The /v1/responses endpoint is not supported by this gateway. \
                     Use /v1/chat/completions instead.",
                )
            }),
        )
        // OpenAI-compatible image generation (PARITY: OpenAI exposes
        // /v1/images/generations; LiteLLM/Portkey proxy image generation). Authed
        // like the other /v1 routes; default provider is `openai` (gpt-image-1 /
        // dall-e-3). The prompt is user TEXT → PII-masked before egress (same
        // posture as chat/rerank, NOT moderations).
        .route(
            "/v1/images/generations",
            post(crate::images_api::image_generation),
        )
        // OpenAI-compatible text-to-speech (PARITY: OpenAI exposes
        // /v1/audio/speech; LiteLLM/Portkey proxy it). Completes the audio pair
        // with /v1/audio/transcriptions. The REQUEST is JSON (the `input` is
        // text, small), so it rides this NORMAL authed router under the standard
        // 2 MiB body cap — NOT the larger audio-upload router (that bounds the
        // request, and the TTS request is tiny; the binary RESPONSE size is not
        // governed by RequestBodyLimit). Default provider `openai`
        // (gpt-4o-mini-tts / tts-1). The `input` is user TEXT → PII-masked before
        // egress (classify-then-mask, same posture as chat/images/rerank). The
        // response is BINARY audio with a per-format Content-Type.
        .route("/v1/audio/speech", post(crate::audio_api::speech))
        // OpenAI-compatible model discovery (PARITY): every OpenAI-compatible
        // client (OpenAI SDK, LangChain, LlamaIndex) calls these to enumerate
        // models. Authed like the other /v1 routes; returns the full curated
        // catalog to any authed caller (no per-tenant allowlist today).
        .route("/v1/models", get(crate::models_api::list_models))
        .route("/v1/models/{id}", get(crate::models_api::retrieve_model))
        // Runtime custom-provider registry (CE): add/list/remove operator-
        // defined OpenAI-compatible providers with NO restart. POST upserts
        // (validate → persist to configs/providers.json → hot-swap the ArcSwap
        // snapshot), GET lists with the api_key MASKED (write-only secret),
        // DELETE removes (404 when absent). Authed like every /v1 route; NOT
        // entitlement-gated (a CE feature). Traffic to a custom provider rides
        // the SAME chat pipeline, so usage/logs/analytics/metrics/status all
        // record it under its registered name automatically.
        .route(
            "/v1/providers",
            post(crate::providers_api::upsert_provider).get(crate::providers_api::list_providers),
        )
        .route(
            "/v1/providers/{name}",
            axum::routing::delete(crate::providers_api::delete_provider),
        )
        // CE Console auth (SESSION-authed): the session's own identity +
        // logout (real revocation via a persisted per-account session-version
        // bump). Ride the SAME auth middleware — a console session is accepted
        // there via the console bridge; an rp_-key-authed request carries no
        // ConsoleSession extension, so these return a clean 401 for it.
        .route("/v1/console/me", get(crate::console_api::me))
        // Session-only own-key reveal (intentional: the operator copies their
        // gateway rp_ key into an SDK). Never reachable via rp_-key auth.
        .route("/v1/console/api-key", get(crate::console_api::api_key))
        .route("/v1/console/logout", post(crate::console_api::logout))
        // FinOps chargeback/showback export (PRD-008 FR-24): gated inside the
        // handler on Feature::FinOpsExport (403 when not entitled). Read-only,
        // tenant-isolated by key ownership; off the chat path.
        .route("/v1/finops/usage", get(crate::finops_api::usage_export))
        // Recent-window usage TIME-SERIES (powers the Console Overview/Usage trend
        // charts): same gating (Feature::FinOpsExport) + key-ownership scoping as
        // /v1/finops/usage. Read-only over the in-memory ring, bucketed by event
        // timestamp; HONEST — the ring is the recent window (~1000 events), not
        // durable history (the response `note` says so). No new store, no ADR.
        .route(
            "/v1/finops/timeseries",
            get(crate::finops_api::usage_timeseries),
        )
        // Recent-window CACHE-SAVINGS rollup (powers the Console Cache page's
        // "Cost saved" + "Tokens saved" StatCards): same gating
        // (Feature::FinOpsExport) + key-ownership scoping as the other finops
        // reads. Folds served cache-hit events ((cache)/(semantic-cache),
        // cache_hit=true) once off the hot path; sums the per-hit RECORDED ESTIMATE
        // (estimated_saved_cost_micro_usd) + tokens not re-sent. HONEST — the ring is
        // the recent window (~1000 events) and the cost is an estimate (the response
        // `note` says both); no cache hits → honest zeroes. No new store, no ADR.
        .route(
            "/v1/finops/cache-savings",
            get(crate::finops_api::cache_savings),
        )
        // Recent request logs (PRD-009 / observability v2): read-only over the
        // in-memory observability ring, tenant-isolated by key ownership (the same
        // model as /v1/finops/usage), authed-only (no extra entitlement gate — your
        // own request logs, like /analytics). Emits no usage event; no new store
        // (same posture as /v1/finops/usage + /metrics — no ADR needed).
        .route("/v1/logs", get(crate::logs_api::list_logs))
        // Residency observability (PRD-012 / sovereign routing): read-only over the
        // in-memory observability ring, tenant-isolated by key ownership (the same
        // model as /v1/logs + /v1/finops/usage), authed-only (no extra entitlement
        // gate — your OWN residency decisions, like /v1/logs). The summary is counts
        // + percentages + by-region/by-outcome (no dated `series` — the ring is not
        // history, honest-absent); the ledger is label-only rows (region/outcome/
        // model/key, no compliance framework, no raw content). Emits no usage event;
        // no new store (same posture as /v1/logs + /metrics — no ADR needed).
        .route(
            "/v1/residency/summary",
            get(crate::residency_api::residency_summary),
        )
        .route(
            "/v1/residency/ledger",
            get(crate::residency_api::residency_ledger),
        )
        // Cache purge (PRD-007 FR-19; PARITY with Portkey/LiteLLM cache
        // invalidation). Authed like the other /v1 routes; tenant-scoped by the
        // authenticated TenantContext (never a client-supplied tenant).
        .route("/v1/cache/purge", post(crate::cache_api::purge))
        .route(
            "/v1/prompts/{reference}",
            get(crate::prompts_api::get_prompt),
        )
        .route(
            "/v1/prompts/{reference}/render",
            post(crate::prompts_api::render_prompt),
        )
        .route(
            "/v1/prompts/{reference}/completions",
            post(crate::prompts_api::prompt_completions),
        );
    // Advanced-guardrails moat surfaces (ADR-088): the callable /v1/moderations
    // endpoint (built-in moderator + provider proxy; sends RAW input — moderation
    // must see raw content) and the read-only /v1/guardrails/outcomes detection
    // telemetry (tenant-isolated by key ownership, gated on
    // Feature::AdvancedGuardrails). ENTERPRISE-ONLY: on the CE build neither the
    // handler module nor the route exists, so both surfaces 404 by construction.
    #[cfg(feature = "enterprise")]
    let authed = authed
        .route("/v1/moderations", post(crate::moderations_api::moderations))
        .route(
            "/v1/guardrails/outcomes",
            get(crate::guardrails_api::outcomes),
        );
    // Agentic-security moat (P3 / ADR-016): the 13 /v1/mcp/* routes + their 4
    // registry extensions, mounted by the feature-gated helper below. Each
    // handler additionally gates on Feature::AgenticSecurity (404 when not
    // entitled). ENTERPRISE-ONLY (PRD-047 / ADR-088): on the CE build the
    // helper — and therefore every /v1/mcp/* route — does not exist, so the
    // surface 404s by construction.
    #[cfg(feature = "enterprise")]
    let authed = mount_mcp_routes(authed, agents, manifests, run_registry, egress_resolver);
    let authed = authed
        .layer(middleware::from_fn(auth_middleware))
        .layer(axum::Extension(auth_state.clone()))
        .layer(axum::Extension(prompts))
        // Console-session bridge (CE Console auth): read by auth_middleware as
        // the fallback credential seam — same injection pattern as auth_state.
        .layer(axum::Extension(console_bridge.clone()));
    // Auth-failure tracker extension (R0.2) — added ONLY when the feature is
    // enabled, so the disabled path inserts nothing into the request extensions
    // and `auth_middleware` skips the gate entirely (byte-identical). Applied as
    // an outer layer to the auth middleware so the handle is present in the
    // extensions when `auth_middleware` runs.
    let auth_failure_tracker_for_audio = auth_failure_tracker.clone();
    let authed = match auth_failure_tracker {
        Some(tracker) => authed.layer(axum::Extension(tracker)),
        None => authed,
    };
    // Audit-ledger handle for the auth seam (R0.3 — security-event logging).
    // ALWAYS injected (the handle is itself `Option`): the auth middleware
    // records auth-failure / throttle security events ONLY when the ledger is
    // enabled (Some), so a disabled ledger (ship-dark default = None) is a
    // zero-work no-op — byte-identical. Cloning `LedgerHandle` is a channel-
    // sender clone (cheap); it shares the single writer with `AppState.ledger`.
    let security_ledger: crate::auth::SharedLedgerHandle = Arc::new(state.ledger.clone());
    let authed = authed.layer(axum::Extension(security_ledger.clone()));
    let authed = authed.layer(reliability);

    // OpenAI-compatible speech-to-text + audio translation (PARITY: OpenAI exposes
    // /v1/audio/transcriptions and /v1/audio/translations; LiteLLM/Portkey proxy
    // audio). They live in their OWN router so they can ride a LARGER body limit
    // (`audio_reliability`, ~26 MiB) than the rest — audio uploads exceed the
    // 2 MiB global cap (sized for chat JSON). It re-applies the SAME auth seam
    // (auth_middleware + the same three
    // auth extensions: SharedAuthState, the optional SharedAuthFailureTracker, and
    // SharedLedgerHandle), so the 401/throttle posture is identical to the other
    // /v1 routes. Default provider `openai`; `groq` allowed via
    // x-routeplane-provider. Inbound body is multipart/form-data (binary audio);
    // audio is NOT text-maskable/classifiable (see audio_api.rs) — region
    // eligibility still applies. Merged into the app alongside `authed`.
    let audio_authed = Router::new()
        .route(
            "/v1/audio/transcriptions",
            post(crate::audio_api::transcriptions),
        )
        // OpenAI-compatible audio translation (PARITY: OpenAI exposes
        // /v1/audio/translations; LiteLLM/Portkey proxy it). The near-twin of
        // transcriptions — same multipart contract, output always English, no
        // `language` field. Rides this SAME audio router so it inherits the larger
        // body limit (binary audio uploads exceed the 2 MiB global cap) and the
        // identical auth seam. Default provider `openai`; `groq` allowed. Whisper
        // models (whisper-1, whisper-large-v3) support it.
        .route(
            "/v1/audio/translations",
            post(crate::audio_api::translations),
        )
        .layer(middleware::from_fn(auth_middleware))
        .layer(axum::Extension(auth_state))
        // Same console-session bridge as the main authed router, so a Console
        // session authorizes the audio surface identically.
        .layer(axum::Extension(console_bridge.clone()));
    let audio_authed = match auth_failure_tracker_for_audio {
        Some(tracker) => audio_authed.layer(axum::Extension(tracker)),
        None => audio_authed,
    };
    let audio_authed = audio_authed
        .layer(axum::Extension(security_ledger))
        .layer(audio_reliability);

    // Read-only platform-status surface (no auth, like /healthz) for the internal
    // status board. Carries ONLY non-sensitive aggregate operational state — no
    // keys, tenant ids, bodies, or PII. Like `public`, it is NOT wrapped by the
    // reliability stack — a load-shed must never make a probe fail. A status
    // board may fetch this cross-origin, and the surface is credential-free and
    // non-sensitive, so it keeps its own scoped allow-any CORS layer — the
    // app-wide layer is fail-closed by default and must stay that way for the
    // credentialed API surface.
    let status_routes = Router::new().route("/status", get(status)).layer(
        CorsLayer::new()
            .allow_origin(tower_http::cors::Any)
            .allow_methods([axum::http::Method::GET]),
    );

    // CE Console auth (PUBLIC): signup + login live OUTSIDE the auth layer —
    // they CREATE/EXCHANGE the credential. Both argon2-verify on the blocking
    // pool; login adds a fixed failure delay + dummy-verify (no enumeration /
    // timing oracle). Open signup is intentional for a self-hosted CE (the
    // operator bootstraps their own account); invite/approval gating is an
    // Enterprise concern. Bodies are bounded by axum's default 2 MiB limit,
    // and password length is capped before hashing (argon2-DoS guard).
    // Always-on per-source-IP throttle for the PUBLIC credential routes (the
    // only unauthenticated password endpoints). Dedicated instance, tuned looser
    // than the keyed default so a fumbling operator isn't locked out, but tight
    // enough to bound online brute force: 8 attempts / 5 min per IP, then
    // exponential backoff capped at 15 min. Lock-free (atomic slots).
    let console_throttle: crate::console_api::SharedConsoleThrottle =
        std::sync::Arc::new(routeplane_limits::auth_failures::AuthFailureTracker::new(
            routeplane_limits::auth_failures::AuthFailureConfig {
                threshold: 8,
                window_ms: 300_000,
                backoff_base_ms: 2_000,
                backoff_cap_ms: 900_000,
                slots: 4_096,
            },
        ));
    let console_public = Router::new()
        .route("/v1/console/signup", post(crate::console_api::signup))
        .route("/v1/console/login", post(crate::console_api::login))
        .layer(axum::Extension(console_bridge.clone()))
        .layer(axum::Extension(console_throttle));

    // Build our application. Security/hygiene response headers wrap the whole
    // app (public + authed + status) — 2026-06-12 dogfood found none were set.
    // Optionally serve the bundled Community Edition Console (a static SPA) from
    // this same origin — the single Docker image ships the gateway + Console
    // together. Enabled ONLY when `RP_CONSOLE_DIR` points at the built assets
    // (the image build sets it); unset ⇒ no static serving and default behavior
    // is byte-identical. Mounted as the router FALLBACK so it never shadows an API
    // route; unmatched non-API paths return `index.html` for SPA client-side
    // routing. It is public (outside the auth layer) so the app can load before
    // the operator enters a key — the SPA then authenticates its own API calls.
    let routed = public
        .merge(authed)
        .merge(audio_authed)
        .merge(status_routes)
        .merge(console_public);
    // Community Edition: the Enterprise-only surface (/v1/moderations,
    // /v1/guardrails/outcomes, /v1/mcp/*, the /v1/prompts collection) answers a
    // uniform 402 `enterprise_only` upsell instead of the accidental 404/405
    // (or, with the bundled Console, the SPA fallback's index.html). Mounted
    // OUTSIDE the auth layer — these paths required no key to observe before
    // (they were unmounted), and a caller must see `enterprise_only`, never a
    // 401 first (see api_error::mount_enterprise_only_stubs). On the enterprise
    // build this block does not exist and the real routes above are unchanged.
    #[cfg(not(feature = "enterprise"))]
    let routed = crate::api_error::mount_enterprise_only_stubs(routed);
    let routed = match &console_dir {
        Some(dir) => {
            let index = std::path::Path::new(dir).join("index.html");
            tracing::info!("Community Edition Console: serving static SPA from {dir}");
            routed.fallback_service(
                tower_http::services::ServeDir::new(dir)
                    .fallback(tower_http::services::ServeFile::new(index)),
            )
        }
        None => routed,
    };
    let app = with_security_headers(routed)
        // Cross-origin browser access for the Routeplane Console (ADR-061). The
        // Console is a separate static origin, so every authed call (which carries
        // `x-routeplane-api-key` / `Authorization`) makes the browser send a CORS
        // preflight `OPTIONS`. Without an app-wide CORS layer that preflight is
        // 401'd by auth (or 405'd where OPTIONS is unrouted) and the browser reports
        // a useless "Failed to fetch" — which is exactly what broke the dashboard's
        // live pages. Origins are FAIL-CLOSED: pinned via `RP_CORS_ALLOWED_ORIGINS`
        // (comma-separated), closed when unset (the bundled same-origin Console is
        // unaffected), reflect-any only behind RP_CORS_DEV_MODE=on (see
        // `build_cors_layer`). This layer is OUTERMOST so the preflight
        // short-circuits here before auth / reliability ever run.
        .layer(build_cors_layer())
        .with_state(state);

    // Run it
    let port = std::env::var("PORT").unwrap_or_else(|_| "8080".to_string());
    let addr: SocketAddr = format!("0.0.0.0:{port}").parse().expect("Invalid address");

    // PRD-009 FR-16: periodic OTLP metrics export (request/token/cost counters +
    // latency histograms under `routeplane.*`). No-op unless OTEL_EXPORT_ENABLED
    // + an OTLP endpoint are set — default off ⇒ byte-identical. Off the hot path.
    otel::spawn_metrics_exporter(shed_total);

    tracing::info!("listening on {}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    // `into_make_service_with_connect_info` surfaces the TCP peer address to
    // handlers via `ConnectInfo<SocketAddr>` — the non-spoofable key the console
    // credential-route throttle uses.
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await
    .unwrap();
}

/// Build the opt-in distributed (Redis) rate limiter (ADR-056 Mode D). Returns
/// the `AppState.distributed_limiter` handle. The DEFAULT build (no
/// `redis-limits` feature) returns `None` of an uninhabited type — no Redis
/// dependency, no env read, byte-identical to Mode L. With the feature on, it
/// activates ONLY when `ROUTEPLANE_REDIS_URL` is configured; `connect` is
/// bounded (boot never hangs) and a connect failure degrades to Mode L rather
/// than aborting startup (the request path must never hard-depend on Redis).
#[cfg(feature = "redis-limits")]
async fn build_distributed_limiter() -> crate::proxy::DistributedLimiterHandle {
    use routeplane_limits::distributed::{DistributedConfig, DistributedLimiter};
    let Some(cfg) = DistributedConfig::from_env() else {
        tracing::info!(
            "distributed rate limiting: disabled (set ROUTEPLANE_REDIS_URL to enable Mode D)"
        );
        return None;
    };
    match DistributedLimiter::connect(&cfg).await {
        Ok(limiter) => {
            tracing::info!(
                "distributed rate limiting: ENABLED (Mode D, fallback={:?})",
                limiter.fallback()
            );
            Some(limiter)
        }
        Err(e) => {
            tracing::warn!(
                "distributed rate limiting: Redis connect failed ({e}); degrading to local Mode L"
            );
            None
        }
    }
}

/// Default build: no Redis. Returns the uninhabited `None` handle — Mode L.
#[cfg(not(feature = "redis-limits"))]
async fn build_distributed_limiter() -> crate::proxy::DistributedLimiterHandle {
    None
}

/// Mount the agentic-security MCP surface (P3 / ADR-016 + ADR-055): the 13
/// `/v1/mcp/*` routes plus the four registry/resolver Extensions their handlers
/// extract. Every handler additionally gates on `Feature::AgenticSecurity`
/// (404 when not entitled) — this helper only exists on the enterprise build
/// (PRD-047 / ADR-088), so on the CE build the whole surface 404s by
/// construction with zero `routeplane-mcp` code in the binary. The Extensions
/// are applied INSIDE the auth middleware (they run after auth passes, before
/// the handlers extract them) — extension layers are order-insensitive as long
/// as they wrap the routes.
#[cfg(feature = "enterprise")]
fn mount_mcp_routes(
    authed: Router<Arc<crate::proxy::AppState>>,
    agents: Arc<routeplane_mcp::registry::AgentRegistry>,
    manifests: Arc<routeplane_mcp::manifest::ManifestRegistry>,
    run_registry: Arc<routeplane_mcp::run::RunRegistry>,
    egress_resolver: crate::mcp_api::SharedResolver,
) -> Router<Arc<crate::proxy::AppState>> {
    authed
        .route(
            "/v1/mcp/tool-result/inspect",
            post(crate::mcp_api::inspect_result),
        )
        .route(
            "/v1/mcp/tool-call/authorize",
            post(crate::mcp_api::authorize_tool_call),
        )
        .route("/v1/mcp/run/step", post(crate::mcp_api::run_step))
        // Ring-2 agentic deepening (ADR-055): sampling defense, HITL approvals,
        // signed receipts, anomaly operator surface. All gated inside the handler
        // on Feature::AgenticSecurity (404 when not entitled), off the chat path.
        .route(
            "/v1/mcp/sampling/evaluate",
            post(crate::mcp_api::sampling_evaluate),
        )
        .route("/v1/mcp/hitl/approve", post(crate::mcp_api::hitl_approve))
        .route("/v1/mcp/hitl/deny", post(crate::mcp_api::hitl_deny))
        .route("/v1/mcp/hitl/status/{id}", get(crate::mcp_api::hitl_status))
        .route(
            "/v1/mcp/hitl/pending",
            get(crate::mcp_api::hitl_list_pending),
        )
        .route("/v1/mcp/receipt/issue", post(crate::mcp_api::receipt_issue))
        .route(
            "/v1/mcp/receipt/verify",
            post(crate::mcp_api::receipt_verify),
        )
        .route(
            "/v1/mcp/anomaly/status/{agent_id}",
            get(crate::mcp_api::anomaly_status),
        )
        .route(
            "/v1/mcp/anomaly/clear",
            post(crate::mcp_api::anomaly_clear_quarantine),
        )
        .route(
            "/v1/mcp/security/events",
            get(crate::mcp_api::security_events),
        )
        .layer(axum::Extension(agents))
        .layer(axum::Extension(manifests))
        .layer(axum::Extension(run_registry))
        .layer(axum::Extension(egress_resolver))
}

/// Build the idempotency-key store. TTL is overridable via
/// `RP_IDEMPOTENCY_TTL_SECONDS` (default 24h, Stripe's window); any non-positive
/// or unparseable value falls back to the default. Always constructed ($0 standing
/// cost) — participation is per-request via the `Idempotency-Key` header.
fn idempotency_store_from_env() -> routeplane_cache::idempotency::IdempotencyStore {
    let ttl = std::env::var("RP_IDEMPOTENCY_TTL_SECONDS")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(routeplane_cache::idempotency::DEFAULT_IDEMPOTENCY_TTL_SECONDS);
    routeplane_cache::idempotency::IdempotencyStore::with_ttl(ttl)
}

/// Seed the Unleash cold-start snapshot (PRD-013 FR-5). With `UNLEASH_SEED_FILE`
/// set, load that baked client-features JSON (the ADR-025 image-seed) so the
/// gateway enforces a known toggle set from the very first request; otherwise
/// start from an empty (all-released) snapshot and let the poller populate it
/// within one interval. A seed problem is **non-fatal** — gating is fail-open (an
/// unknown flag resolves released, never a false holdback), so we log and fall
/// back to empty rather than refuse to start (unlike the security-control configs
/// in `main`, whose failure mode is the opposite).
fn seed_unleash_snapshot(flags: &UnleashFlags) {
    let seed_path = std::env::var("UNLEASH_SEED_FILE")
        .ok()
        .map(|p| p.trim().to_string())
        .filter(|p| !p.is_empty());
    match seed_path {
        Some(path) => match std::fs::read_to_string(&path) {
            Ok(json) => match flags.memoize_client_features_json(&json) {
                Ok(()) => tracing::info!("unleash: seeded cold-start snapshot from {path}"),
                Err(e) => {
                    tracing::warn!(
                        "unleash: seed file {path} failed to parse ({e}); starting from an empty all-released snapshot until the first poll"
                    );
                    seed_empty_unleash_snapshot(flags);
                }
            },
            Err(e) => {
                tracing::warn!(
                    "unleash: seed file {path} unreadable ({e}); starting from an empty snapshot"
                );
                seed_empty_unleash_snapshot(flags);
            }
        },
        None => {
            tracing::info!(
                "unleash: no UNLEASH_SEED_FILE; cold start from an empty snapshot (the poller populates it)"
            );
            seed_empty_unleash_snapshot(flags);
        }
    }
}

/// Initialise an empty (all-released) Unleash snapshot — the safe cold-start
/// default: an unknown flag resolves released, so an empty snapshot produces no
/// holdbacks until the first poll fetches the real toggle set.
fn seed_empty_unleash_snapshot(flags: &UnleashFlags) {
    if let Err(e) = flags.memoize_bools(&[]) {
        tracing::warn!("unleash: failed to initialise empty snapshot: {e}");
    }
}

/// Read-only `GET /status` handler — thin wrapper over the pure shaping logic in
/// `status::status_snapshot_json` (unit/integration-tested there). Reads only
/// lock-free atomics + off-hot-path snapshots; no `unwrap()`/panic. Passes the
/// binary-level capacity-shed counter in. No secrets, no PII in the output.
async fn status(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    axum::Json(status::status_snapshot_json(
        &state.health,
        &state.cache,
        &state.observability_engine,
        shed_total(),
        &state.custom_providers.names(),
    ))
}

/// `GET /metrics` — Prometheus text exposition (format `0.0.4`). Unauthenticated
/// (operational, like `/healthz`); reads only lock-free atomics, no per-tenant or
/// content data in any label. The body string is allocated here (off the hot
/// path), never under a lock. The `shed_total` counter lives at the binary level
/// (`SHED_TOTAL`), so it is threaded in alongside the request-path metrics table.
async fn metrics_handler() -> impl IntoResponse {
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4",
        )],
        metrics::metrics().render(shed_total()),
    )
}

/// Apply defense-in-depth response headers to every route (public + authed).
/// `if_not_present` so a handler that sets its own value (e.g. `content-type`,
/// or the SSE `x-routeplane-cache`) is never overwritten. HSTS is terminated at
/// the Azure Container Apps TLS edge and is intentionally not duplicated here.
fn with_security_headers<S>(router: Router<S>) -> Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    router
        .layer(SetResponseHeaderLayer::if_not_present(
            HeaderName::from_static("x-content-type-options"),
            HeaderValue::from_static("nosniff"),
        ))
        .layer(SetResponseHeaderLayer::if_not_present(
            HeaderName::from_static("x-frame-options"),
            HeaderValue::from_static("DENY"),
        ))
        .layer(SetResponseHeaderLayer::if_not_present(
            HeaderName::from_static("referrer-policy"),
            HeaderValue::from_static("no-referrer"),
        ))
        // Content-Security-Policy (R-CE-XSS MITIGATION): the Console stores its
        // session token in localStorage and `/v1/console/api-key` reveals the
        // gateway rp_ key to a session — so a script injection in the SPA is a
        // key-theft path. This CSP shrinks the injection surface: no external or
        // inline scripts (`script-src 'self'` — a Vite production build ships its
        // JS as same-origin module chunks, no inline `<script>`), no plugins,
        // no framing, and network egress limited to same-origin (the bundled
        // single image serves the API + SPA from one origin). It is a MITIGATION,
        // not elimination — an injected same-origin script could still read the
        // token; the durable fix is an httpOnly session cookie (tracked
        // follow-up). `style-src` keeps `'unsafe-inline'` because React inline
        // `style={{…}}` attributes and Tailwind's runtime styles need it; tighten
        // to a nonce later. VERIFY against the built SPA (browser console: zero
        // CSP violations) before relaxing or shipping a cross-origin API base.
        .layer(SetResponseHeaderLayer::if_not_present(
            HeaderName::from_static("content-security-policy"),
            HeaderValue::from_static(
                "default-src 'self'; \
                 script-src 'self'; \
                 style-src 'self' 'unsafe-inline'; \
                 img-src 'self' data:; \
                 font-src 'self' data:; \
                 connect-src 'self'; \
                 object-src 'none'; \
                 base-uri 'none'; \
                 frame-ancestors 'none'; \
                 form-action 'self'",
            ),
        ))
}

/// Map errors surfaced by the Tower reliability stack onto HTTP responses.
/// A `LoadShed` overload becomes 503 Service Unavailable (the standard
/// "try again later" signal, with a hint header); anything else is a 500.
async fn handle_middleware_error(err: BoxError) -> impl IntoResponse {
    if err.is::<tower::load_shed::error::Overloaded>() {
        // Capacity shed (ADR-025 §3): fast-fail BEFORE queueing to protect the
        // P99 tail. Status is 503 (Service Unavailable = server can't serve
        // now), NOT 429 — 429 means the *client* exceeded a quota (ADR-023's
        // job). The `x-routeplane-shed: capacity` header makes a capacity shed
        // unambiguously distinguishable from an entitlement 429. (ADR-025 §3
        // says "429"; this reconciles it to the HTTP-correct 503 + discriminator
        // — an ADR-025 amendment is recommended.)
        SHED_TOTAL.fetch_add(1, Ordering::Relaxed);
        (
            StatusCode::SERVICE_UNAVAILABLE,
            [("retry-after", "1"), ("x-routeplane-shed", "capacity")],
            "Service overloaded, retry shortly",
        )
            .into_response()
    } else {
        tracing::error!("unhandled middleware error: {err}");
        (StatusCode::INTERNAL_SERVER_ERROR, "Internal server error").into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::with_security_headers;
    use axum::body::Body;
    use axum::http::Request;
    use axum::{routing::get, Router};
    use tower::ServiceExt;

    #[tokio::test]
    async fn security_headers_present_on_every_response() {
        let app = with_security_headers(Router::new().route("/t", get(|| async { "ok" })));
        let resp = app
            .oneshot(Request::builder().uri("/t").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let h = resp.headers();
        assert_eq!(h.get("x-content-type-options").unwrap(), "nosniff");
        assert_eq!(h.get("x-frame-options").unwrap(), "DENY");
        assert_eq!(h.get("referrer-policy").unwrap(), "no-referrer");
    }

    // ADR-025 §3: under capacity saturation the gateway sheds FAST (before
    // queueing) to protect the P99 tail — 503 + `x-routeplane-shed: capacity`
    // + a bumped `shed_total` metric, while in-flight work still completes.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn capacity_shed_returns_503_with_discriminator_and_metric() {
        use super::{handle_middleware_error, shed_total};
        use axum::error_handling::HandleErrorLayer;
        use axum::http::StatusCode;
        use axum::response::Response;
        use axum::BoxError;
        use std::time::Duration;
        use tower::{
            limit::ConcurrencyLimitLayer, load_shed::LoadShedLayer, service_fn, ServiceBuilder,
        };

        let before = shed_total();

        // A slow inner service holds the single in-flight slot so a concurrent
        // request sheds instead of queueing. Built ONCE and cloned — tower's
        // ConcurrencyLimit shares its Arc<Semaphore> across clones (an axum
        // Router re-instantiates the layer per clone, so it would NOT share).
        // A generous in-handler hold (1s) so the single permit is reliably still
        // occupied when the concurrent request polls — even under heavy parallel
        // `cargo test` load — making the shed deterministic, not timing-flaky.
        let inner = service_fn(|_req: Request<Body>| async move {
            tokio::time::sleep(Duration::from_millis(1000)).await;
            Ok::<Response, BoxError>(Response::new(Body::from("ok")))
        });
        let svc = ServiceBuilder::new()
            .layer(HandleErrorLayer::new(handle_middleware_error))
            .layer(LoadShedLayer::new())
            .layer(ConcurrencyLimitLayer::new(1))
            .service(inner);

        let req = || Request::builder().uri("/").body(Body::empty()).unwrap();
        let (a, b) = tokio::join!(svc.clone().oneshot(req()), svc.clone().oneshot(req()));
        let (a, b) = (a.unwrap(), b.unwrap());

        let mut statuses = [a.status(), b.status()];
        statuses.sort();
        assert_eq!(
            statuses,
            [StatusCode::OK, StatusCode::SERVICE_UNAVAILABLE],
            "expected one 200 and one shed-503, got {statuses:?}"
        );

        let shed = if a.status() == StatusCode::SERVICE_UNAVAILABLE {
            &a
        } else {
            &b
        };
        assert_eq!(shed.headers().get("x-routeplane-shed").unwrap(), "capacity");
        assert!(shed.headers().get("retry-after").is_some());
        assert!(shed_total() > before, "shed_total must increment");
    }
}
