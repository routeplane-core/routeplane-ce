//! Unleash-backed [`FeatureProvider`] (ADR-029 G3) — the data-plane bridge from the
//! pure entitlement seam (`routeplane-entitlements`) to the self-hosted Unleash flag
//! plane.
//!
//! ## Fork A (ADR-029 Decision 3)
//! Unleash owns *release/rollout/ops* state (`released(feature, ctx)` — gradual
//! rollout, kill-switches, holdbacks); commercial *entitlement*
//! (`tier_baseline ∪ overrides`) stays contract/code-owned in
//! `routeplane-entitlements`. The two compose at auth as
//! `active = entitled ∧ released` — so this provider answers "is `flag` released for
//! this context?" and the auth layer feeds it into the holdback set, leaving the
//! lock-free `CapabilitySet::active` hot path untouched.
//!
//! ## In-process evaluation (no hot-path network call)
//! The `unleash-api-client` SDK memoizes the toggle spec and evaluates it **locally**
//! ([`UnleashFlags::resolve_bool`] is an in-memory lookup). A background poller
//! refreshes the snapshot off the hot path (wired in a follow-up); a bootstrap seed
//! ([`UnleashFlags::memoize`], reusing the ADR-025 image-seed) provides the
//! cold-start snapshot so the Unleash server can scale to zero. String-feature mode
//! maps each flag key 1:1 onto `routeplane_entitlements::Feature::flag_key()`.

use std::sync::Arc;

use routeplane_entitlements::{EvalContext, FeatureProvider};
use unleash_api_client::api::Feature as UnleashFeature;
use unleash_api_client::api::Strategy;
use unleash_api_client::client::{Client, ClientBuilder};
use unleash_api_client::context::Context;
use unleash_api_client::http::HttpClient;

/// String-feature mode still requires a feature-enum type parameter on the SDK
/// client; it is unused (we resolve by string key), so a one-variant placeholder
/// satisfies the `EnumArray` bound. Never constructed — hence `dead_code`.
#[allow(non_camel_case_types, dead_code)]
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize, enum_map::Enum)]
enum Flags {
    placeholder,
}

/// The SDK's [`HttpClient`] implemented over the **workspace's reqwest 0.13**
/// (ADR-029 G3 / PRD-013 FR-7), replacing the SDK's bundled reqwest 0.12 (its
/// `reqwest-client-rustls` feature). This collapses the two-reqwest tree to one
/// and reuses the gateway's own reqwest/TLS posture for the Unleash fetch — also
/// dropping the `webpki-roots`/CDLA license surface the SDK's rustls path pulled.
///
/// Orphan rules forbid `impl HttpClient for reqwest::Client` (both are foreign),
/// so we wrap it in a local newtype. The impl is a thin delegation — it mirrors
/// the SDK's own reqwest shim 1:1. The SDK constructs the client via
/// `C::default()` (in `http::HTTP::new`), so [`Default`] is required.
#[derive(Clone)]
struct ReqwestHttpClient(reqwest::Client);

impl Default for ReqwestHttpClient {
    /// The SDK constructs the client via `C::default()` (in `http::HTTP::new`).
    /// Harden the Unleash fetch here (PRD-014 FR-3 — poller SSRF/exposure):
    /// - an explicit **timeout** so a hung poll can't pin the background task
    ///   (generous enough for the flag plane's scale-to-zero cold start, ~20s);
    /// - **`redirect::none()`** so a `3xx` off the trusted, config-fixed
    ///   `UNLEASH_API_URL` is never followed to another host — no SSRF-via-redirect,
    ///   and a Cloudflare-Access `302` (once the admin surface is fronted, FR-1)
    ///   surfaces as an error rather than being chased to a login origin.
    ///
    /// Falls back to a plain client if the builder fails (TLS backend init) — this
    /// runs at startup/client-construction, never on a request thread.
    fn default() -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap_or_default();
        Self(client)
    }
}

#[async_trait::async_trait]
impl HttpClient for ReqwestHttpClient {
    type HeaderName = reqwest::header::HeaderName;
    type Error = reqwest::Error;
    type RequestBuilder = reqwest::RequestBuilder;

    fn build_header(name: &'static str) -> Result<Self::HeaderName, Self::Error> {
        Ok(reqwest::header::HeaderName::from_static(name))
    }

    fn get(&self, uri: &str) -> Self::RequestBuilder {
        self.0.get(uri)
    }

    fn post(&self, uri: &str) -> Self::RequestBuilder {
        self.0.post(uri)
    }

    fn header(
        builder: Self::RequestBuilder,
        key: &Self::HeaderName,
        value: &str,
    ) -> Self::RequestBuilder {
        builder.header(key.clone(), value)
    }

    async fn get_json<T: serde::de::DeserializeOwned>(
        req: Self::RequestBuilder,
    ) -> Result<T, Self::Error> {
        req.send().await?.json::<T>().await
    }

    async fn post_json<T: serde::Serialize + Sync>(
        req: Self::RequestBuilder,
        content: &T,
    ) -> Result<bool, Self::Error> {
        let res = req.json(content).send().await?;
        Ok(res.status().is_success())
    }
}

/// Unleash-backed flag source. Wraps the SDK client (which owns the memoized toggle
/// snapshot and evaluates locally). Cheap to clone (`Arc`); share one across the
/// request path.
#[derive(Clone)]
pub struct UnleashFlags {
    client: Arc<Client<Flags, ReqwestHttpClient>>,
}

impl std::fmt::Debug for UnleashFlags {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The SDK client wraps the memoized toggle snapshot + an HTTP client;
        // neither is `Debug` nor useful to render. A stable opaque marker lets
        // structs that embed `Option<UnleashFlags>` (e.g. the gateway's
        // `AuthState`) keep their derived `Debug`.
        f.debug_struct("UnleashFlags").finish_non_exhaustive()
    }
}

impl UnleashFlags {
    /// Build the client in string-feature mode. No network happens here or in
    /// [`resolve_bool`](Self::resolve_bool) — evaluation is local against the
    /// memoized snapshot. `register()`/`poll_for_updates()` (the refresh path) are
    /// wired in a follow-up; `api_url`/`secret` are stored for them.
    pub fn new(
        api_url: &str,
        app_name: &str,
        instance_id: &str,
        secret: Option<String>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let client = ClientBuilder::default()
            .enable_string_features()
            .into_client::<Flags, ReqwestHttpClient>(api_url, app_name, instance_id, secret)?;
        Ok(Self {
            client: Arc::new(client),
        })
    }

    /// Load the local snapshot from a fetched/baked toggle spec — the bootstrap
    /// seed (ADR-025 image-seed) at startup; the poller refreshes it thereafter.
    pub fn memoize(
        &self,
        features: Vec<UnleashFeature>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.client.memoize(features)?;
        Ok(())
    }

    /// Seed the local snapshot from a simple `(flag_key, released)` list — a
    /// convenience over [`memoize`](Self::memoize) for callers that don't need
    /// the full `ClientFeatures` wire shape: trivial bootstrap seeds (FR-5's
    /// "all released" cold-start default) and offline tests of the auth
    /// composition. Each pair becomes a toggle that is enabled iff `released`,
    /// carrying the built-in `default` strategy (always-on when enabled). A flag
    /// absent from the list stays unknown to the snapshot ⇒ `resolve_bool`
    /// returns the caller's default.
    pub fn memoize_bools(
        &self,
        flags: &[(&str, bool)],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let features = flags
            .iter()
            .map(|(name, released)| UnleashFeature {
                name: (*name).to_string(),
                description: None,
                enabled: *released,
                strategies: vec![Strategy {
                    name: "default".to_string(),
                    ..Default::default()
                }],
                variants: None,
                created_at: None,
            })
            .collect();
        self.memoize(features)
    }

    /// Seed the local snapshot from a baked Unleash **client-features** document
    /// — the `/api/client/features` JSON shape (`{ "version", "features": [...] }`),
    /// i.e. the ADR-025 cold-start image-seed. Lets the gateway serve a known
    /// toggle set immediately at startup, before the first poll, so the Unleash
    /// server can stay scaled to zero (ADR-029 D2). Parse errors surface to the
    /// caller, which decides the fallback (the gateway falls back to an empty
    /// all-released snapshot — gating is fail-open).
    pub fn memoize_client_features_json(
        &self,
        json: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let parsed: unleash_api_client::api::Features = serde_json::from_str(json)?;
        self.memoize(parsed.features)
    }

    /// Spawn the background refresh: register with the Unleash server, then poll for
    /// toggle-spec updates on the SDK's interval, memoizing each fetch into the
    /// local snapshot. Runs OFF the hot path — eval always reads the in-memory
    /// snapshot (the bootstrap seed until the first successful poll). Returns the
    /// task handle; abort it to stop. A transient server outage only means the
    /// snapshot isn't refreshed (eval continues from the last-known/seed) — the
    /// design that lets the Unleash server scale to zero (ADR-029 D2).
    pub fn spawn_refresh(&self) -> tokio::task::JoinHandle<()> {
        let client = Arc::clone(&self.client);
        tokio::spawn(async move {
            if let Err(e) = client.register().await {
                tracing::warn!(error = %e, "unleash: register failed; serving the seeded snapshot until the first successful poll");
            }
            // Loops on the interval until the task is aborted; resilient to a
            // server outage (just no refresh — eval keeps using the snapshot).
            client.poll_for_updates().await;
        })
    }
}

/// Map the entitlement seam's [`EvalContext`] onto the SDK's targeting context:
/// `targeting_key` → `user_id`, attributes → `properties` (region/tier/app for
/// context-aware rollout). Off the hot path (auth-time).
fn to_unleash_context(ctx: &EvalContext) -> Context {
    Context {
        user_id: ctx.targeting_key.clone(),
        properties: ctx
            .attributes
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect(),
        ..Default::default()
    }
}

impl FeatureProvider for UnleashFlags {
    /// Resolve `flag` against the local Unleash snapshot for `ctx`. Returns
    /// `default` when the flag is unknown to Unleash (`FLAG_NOT_FOUND → default`,
    /// matching `RouteplaneEntitlementProvider`). Lock-free, no network.
    fn resolve_bool(&self, flag: &str, default: bool, ctx: &EvalContext) -> bool {
        self.client
            .is_enabled_str(flag, Some(&to_unleash_context(ctx)), default)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A toggle definition as the Unleash server would serve it, built locally.
    fn feature(name: &str, enabled: bool) -> UnleashFeature {
        UnleashFeature {
            name: name.to_string(),
            description: None,
            enabled,
            // Built-in "default" strategy = always-on when the feature is enabled.
            strategies: vec![Strategy {
                name: "default".to_string(),
                ..Default::default()
            }],
            variants: None,
            created_at: None,
        }
    }

    #[test]
    fn resolves_from_local_snapshot_without_network() {
        let flags = UnleashFlags::new("http://127.0.0.1:0/api", "routeplane", "test", None)
            .expect("build client");
        flags
            .memoize(vec![
                feature("semantic_cache", true),
                feature("prompt_registry", false),
            ])
            .expect("memoize seed");

        let ctx = EvalContext::for_tenant("t_acme");
        // Known + released -> true (caller default ignored).
        assert!(flags.resolve_bool("semantic_cache", false, &ctx));
        // Known + held back -> false.
        assert!(!flags.resolve_bool("prompt_registry", true, &ctx));
        // Unknown flag -> caller's default (Unleash doesn't gate it).
        assert!(flags.resolve_bool("unknown_flag", true, &ctx));
        assert!(!flags.resolve_bool("unknown_flag", false, &ctx));
    }

    #[test]
    fn memoizes_from_client_features_json() {
        // The baked cold-start seed (ADR-025 image-seed): the /api/client/features
        // wire shape parses and seeds the snapshot.
        let flags = UnleashFlags::new("http://127.0.0.1:0/api", "routeplane", "test", None)
            .expect("build client");
        let json = r#"{
            "version": 2,
            "features": [
                {"name": "semantic_cache", "enabled": false, "strategies": [{"name": "default"}]},
                {"name": "prompt_registry", "enabled": true, "strategies": [{"name": "default"}]}
            ]
        }"#;
        flags
            .memoize_client_features_json(json)
            .expect("seed from client-features json");

        let ctx = EvalContext::for_tenant("t_acme");
        assert!(!flags.resolve_bool("semantic_cache", true, &ctx)); // disabled in seed
        assert!(flags.resolve_bool("prompt_registry", false, &ctx)); // enabled in seed
        assert!(flags.resolve_bool("not_in_seed", true, &ctx)); // unknown -> default
    }

    #[test]
    fn rejects_malformed_seed_json() {
        let flags = UnleashFlags::new("http://127.0.0.1:0/api", "routeplane", "test", None)
            .expect("build client");
        assert!(flags
            .memoize_client_features_json("{ not valid json")
            .is_err());
    }
}
