use arc_swap::ArcSwap;
use axum::{
    body::Body,
    http::{HeaderValue, Request, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use routeplane_entitlements::{
    CapabilitySet, EvalContext, Feature, FeatureProvider, TenantState, Tier,
};
use routeplane_flags::UnleashFlags;
// The tenant-default check ENGINE is a moat surface (ADR-088) — enterprise only.
// The raw `guardrails` JSON on a key (plain serde_json::Value) stays CE; only its
// COMPILATION into a `CompiledGuardrails` rides `enterprise`.
#[cfg(feature = "enterprise")]
use routeplane_guardrails_advanced::{CompiledGuardrails, ConfigSource};
use routeplane_limits::auth_failures::{AuthFailureConfig, AuthFailureTracker, AuthThrottle};
use routeplane_limits::{now_unix_ms, KeyLimits};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::fs;
use std::sync::Arc;
use subtle::ConstantTimeEq;

/// A tenant's durable, control-plane-owned key record.
///
/// The entitlement fields (`tenant_id`, `tier`, `capability_overrides`,
/// `rollout_holdbacks`) are the data-plane half of the [ADR-012] §3/§4
/// entitlement seam. Every one carries `#[serde(default)]`, so today's
/// `keys.json` — which has none of them — still deserialises: an absent record
/// resolves to `tenant_id = None` (the proxy falls back to `name`), `Tier::Free`,
/// and empty override/holdback sets, i.e. core-only. Backward-compatible by
/// construction (see [`branching-and-devex.md`] §6.4).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VirtualKey {
    pub name: String,
    pub routeplane_key: String,
    pub provider_keys: HashMap<String, String>,

    /// Stable tenant identity ([ADR-003]). Absent on legacy keys; the resolved
    /// [`TenantContext`] falls back to `name` so a tenant id always exists.
    #[serde(default)]
    pub tenant_id: Option<String>,

    /// Tenant lifecycle state ([ADR-050] D1/D6). The data plane fails closed on
    /// any non-`Active` tenant at key resolution ([`auth_middleware`]). Absent on
    /// legacy keys → [`TenantState::Active`] (backward-compatible by construction;
    /// no behaviour change for an existing `keys.json`).
    #[serde(default)]
    pub lifecycle_state: TenantState,

    /// Commercial plan; drives `tier_baseline`. Defaults to [`Tier::Free`].
    #[serde(default)]
    pub tier: Tier,

    /// Per-tenant additive grants (the `∪ per_tenant_overrides` term) — a
    /// custom-customer feature crate, or a Standard tenant's add-on.
    #[serde(default)]
    pub capability_overrides: BTreeSet<Feature>,

    /// Per-tenant operational subtractions (part of the `− rollout_holdbacks`
    /// term).
    ///
    /// Holdbacks are OPERATIONAL release state, not per-tenant durable data
    /// ([`branching-and-devex.md`] §6.4): the `released(...)` half lives at the
    /// gateway level — the `RP_ROLLOUT_HOLDBACKS` process config, parsed once at
    /// startup into [`AuthState::global_holdbacks`] and subtracted for EVERY
    /// tenant (the future control plane pushes the same set). This field is the
    /// rare per-tenant escape hatch (hold a feature back for exactly one
    /// tenant); it is UNIONED with the global set at resolution. It defaults
    /// empty, so it is inert today.
    #[serde(default)]
    pub rollout_holdbacks: BTreeSet<Feature>,

    /// Server-side DEFAULT data-residency region for this tenant/key (Task #6).
    ///
    /// When set, residency enforcement applies to this key's traffic by policy —
    /// the client's `x-routeplane-residency` header may NARROW it (request a
    /// different/stricter region) but can never DISABLE it (an absent or empty
    /// header does not turn enforcement off). This is the control-plane-owned
    /// guarantee: a tenant subject to DPDP can't opt out of region-locking by
    /// just omitting a header. Absent on legacy keys → no default policy.
    #[serde(default)]
    pub default_residency: Option<String>,

    /// Tenant-default Guardrails v2 spec (G2.6) — the raw `guardrails` JSON
    /// object (same shape as the `guardrails` key of `x-routeplane-config`).
    /// Compiled ONCE at startup into [`AuthState::compiled_guardrails`]; an
    /// invalid spec refuses to start (fail-closed, same doctrine as a typo'd
    /// holdback — a tenant's blocking guardrail silently not compiling is a
    /// security-control failure). This is the only config source allowed to
    /// declare `webhook` checks. Absent on legacy keys → no default checks.
    #[serde(default)]
    pub guardrails: Option<serde_json::Value>,

    /// Budgets & rate limits attached to this key (PRD-008 / ADR-023). The
    /// per-key policy and an optional per-tenant policy ride here as a
    /// serde-default field: ABSENT ⇒ unlimited, so a legacy key deserialises and
    /// behaves byte-identically (no counters, no `x-ratelimit-*` headers). At
    /// startup `main` folds every key's `limits` into the shared
    /// [`routeplane_limits::LimitRegistry`]; the proxy resolves per-request guards
    /// from it by `routeplane_key` + resolved tenant id.
    #[serde(default)]
    pub limits: Option<KeyLimits>,

    /// Org compliance frameworks this tenant operates under ([ADR-035] §4) — e.g.
    /// `["DPDP","HIPAA"]`. Drives the proxy-side default-deny compliance gate: a
    /// catalog model whose `compliance_restrictions` intersect this set is excluded
    /// (`strict`) or flagged (`warn`). Framework codes are the §5 registry
    /// identifiers (config strings, never user content). ABSENT/EMPTY on legacy
    /// keys ⇒ the gate is OFF ⇒ byte-identical (no lookup beyond an `is_empty()`).
    #[serde(default)]
    pub compliance_frameworks: Vec<String>,

    /// The enforcement posture for [`Self::compliance_frameworks`] ([ADR-035] §4):
    /// `strict` excludes/blocks (fail-closed, the safe default), `warn` routes but
    /// flags. Inert when `compliance_frameworks` is empty. Defaults to `Strict`
    /// (§6 fail-closed), but with no frameworks configured it never fires.
    #[serde(default)]
    pub compliance_mode: ComplianceMode,

    /// Model Catalog allowlist provisioned to this key (PRD-008 FR-1/FR-3) — the
    /// DP-side enforcement re-added atop #170's CP-side catalog object model
    /// (`routeplane_store::ProviderIntegration`/`ProvisioningGrant`). Serde-default
    /// empty ⇒ legacy keys deserialise unchanged; INERT unless the tenant carries
    /// [`Feature::ModelCatalog`] (off in every tier baseline). When active,
    /// enforcement is default-deny / fail-closed. See [`VirtualKey::is_model_provisioned`].
    #[serde(default)]
    pub provisioned_models: ProvisionedModels,

    /// The Model Catalog request-time projection (PRD-008 FR-4): the
    /// `@{slug}/{model}` integrations this key may address. Serde-default empty;
    /// INERT unless [`Feature::ModelCatalog`] is active. See [`VirtualKey::resolve_slug`].
    #[serde(default)]
    pub integrations: KeyIntegrations,
}

/// The Model Catalog integration projection on a [`VirtualKey`] (PRD-008 FR-4):
/// the `@{slug}/{model}` integrations the key may address, keyed by slug.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct KeyIntegrations(pub std::collections::BTreeMap<String, KeyIntegration>);

/// One addressable integration on a key (PRD-008 §4.2 projection): which
/// data-plane adapter it routes through, the credential's resident regions
/// (per-integration sovereign eligibility), and the model allowlist this key may
/// call through it. Carries NO secret — credential resolution rides
/// `provider_keys`/`env:` as today; this is policy only.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct KeyIntegration {
    /// The data-plane registry provider name this integration routes through.
    pub adapter: String,
    /// Resident regions of THIS integration's credential (sovereign eligibility).
    #[serde(default)]
    pub resident_regions: Vec<String>,
    /// The model ids this key may call through this integration (default-deny).
    #[serde(default)]
    pub models: BTreeSet<String>,
}

/// The outcome of resolving a `@{slug}/{model}` catalog address (PRD-008 FR-4).
/// Only ever produced when [`Feature::ModelCatalog`] is active.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SlugResolution<'a> {
    /// The request `model` carried no `@` sigil — fall back to the bare-model allowlist.
    NotASlug,
    /// A well-formed `@slug/model` resolved to a provisioned integration whose
    /// allowlist permits the model (carries the integration + bare model to rewrite).
    Resolved {
        integration: &'a KeyIntegration,
        model: String,
    },
    /// A `@…` request that is denied (malformed, unknown slug, or model excluded).
    Denied,
}

/// The Model Catalog allowlist object provisioned to a key (PRD-008 §4.2 / FR-1):
/// the set of model ids the key may call. A control-plane-owned config object
/// (not a wire shape); empty default = default-deny.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ProvisionedModels(pub BTreeSet<String>);

impl ProvisionedModels {
    /// Is `model` provisioned? An empty allowlist permits NOTHING — default-deny
    /// / fail-closed (PRD-008 FR-3).
    pub fn allows(&self, model: &str) -> bool {
        self.0.contains(model)
    }
}

impl VirtualKey {
    /// Model Catalog bare-model allowlist check (PRD-008 FR-3). The proxy calls it
    /// ONLY when [`Feature::ModelCatalog`] is active, so a key without that
    /// entitlement is byte-identical on the hot path.
    pub fn is_model_provisioned(&self, model: &str) -> bool {
        self.provisioned_models.allows(model)
    }

    /// Resolve a `@{slug}/{model}` catalog address (PRD-008 FR-4) against this
    /// key's [`KeyIntegrations`]. Pure; default-deny / fail-closed throughout. The
    /// model id is everything after the FIRST `/`, so ids with internal slashes
    /// (`@hf/meta-llama/Llama-3-70b`) survive intact.
    pub fn resolve_slug(&self, model: &str) -> SlugResolution<'_> {
        let Some(addr) = model.strip_prefix('@') else {
            return SlugResolution::NotASlug;
        };
        let Some((slug, bare)) = addr.split_once('/') else {
            return SlugResolution::Denied;
        };
        if slug.is_empty() || bare.is_empty() {
            return SlugResolution::Denied;
        }
        match self.integrations.0.get(slug) {
            Some(integration) if integration.models.contains(bare) => SlugResolution::Resolved {
                integration,
                model: bare.to_string(),
            },
            _ => SlugResolution::Denied,
        }
    }
}

/// Org compliance-gate enforcement posture ([ADR-035] §4/§6).
///
/// `Strict` is the fail-closed default (§6): for a tenant that has selected
/// frameworks, a restriction-tagged model is excluded, and a pinned request for
/// one is refused (`403 model_compliance_excluded`). `Warn` routes the request
/// but flags the decision (a ledger security event + a response header) so an org
/// can evaluate posture before enforcing. Empty `compliance_frameworks` makes the
/// mode inert regardless of value.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ComplianceMode {
    /// Exclude/block on a framework intersection (fail-closed; the §6 default).
    #[default]
    Strict,
    /// Route but flag the intersection (posture-evaluation mode).
    Warn,
}

impl VirtualKey {
    /// The tenant id, falling back to `name` when the record omits `tenant_id`
    /// (legacy keys). Never empty.
    pub fn resolved_tenant_id(&self) -> String {
        self.tenant_id.clone().unwrap_or_else(|| self.name.clone())
    }

    /// Resolve this key's [`CapabilitySet`]:
    /// `tier_baseline(tier) ∪ capability_overrides − (global_holdbacks ∪ rollout_holdbacks)`
    /// ([ADR-012] §3). The effective holdback set is the gateway-level
    /// operational holdbacks ([`AuthState::global_holdbacks`], subtracted for
    /// every tenant) unioned with this key's per-tenant escape hatch. Pure,
    /// allocation-light, no I/O.
    pub fn capability_set(&self, global_holdbacks: &BTreeSet<Feature>) -> CapabilitySet {
        if global_holdbacks.is_empty() {
            // Common case (no global holdbacks configured): skip the union
            // allocation entirely — identical cost to the per-key-only path.
            return CapabilitySet::resolve(
                self.tier,
                &self.capability_overrides,
                &self.rollout_holdbacks,
            );
        }
        let effective_holdbacks: BTreeSet<Feature> = self
            .rollout_holdbacks
            .union(global_holdbacks)
            .copied()
            .collect();
        CapabilitySet::resolve(self.tier, &self.capability_overrides, &effective_holdbacks)
    }

    /// Resolve the EFFECTIVE requested residency region (Task #6), combining the
    /// server-side default policy on this key with the client's
    /// `x-routeplane-residency` header.
    ///
    /// Rules:
    ///   * No default + no header  → `None` (no residency request).
    ///   * No default + header `H` → `H` (client opt-in, unchanged behaviour).
    ///   * Default `D` + no header → `D` (policy applies; client cannot disable
    ///     by omission).
    ///   * Default `D` + header `H` → `H` (the header NARROWS to the client's
    ///     requested region). The header can pick a different region but cannot
    ///     clear the default — an empty header is treated as absent, so `D` wins.
    ///
    /// Note this only computes the *requested* region; whether it is actually
    /// ENFORCED still depends on the presence of personal data (the residency
    /// engine's `required_region`). A default region therefore does not force
    /// region-locking on clean, PII-free traffic.
    pub fn effective_requested_region(&self, header: Option<&str>) -> Option<String> {
        let header = header.map(str::trim).filter(|h| !h.is_empty());
        match (self.default_residency.as_deref(), header) {
            (_, Some(h)) => Some(h.to_string()),
            (Some(d), None) => Some(d.to_string()),
            (None, None) => None,
        }
    }
}

/// The resolved per-request tenant context injected alongside [`VirtualKey`].
///
/// This is what `proxy.rs` and future Tower feature layers read to gate optional
/// features (`tenant_ctx.capabilities.active(Feature::X)`). It is cheap to clone
/// (an owned `String` + a `BTreeSet` of `Copy` enums) and immutable once built,
/// so it rides the hot path lock-free.
#[derive(Clone, Debug)]
pub struct TenantContext {
    pub tenant_id: String,
    pub tier: Tier,
    pub capabilities: CapabilitySet,
    /// Org compliance frameworks ([ADR-035] §4), cloned read-only from the
    /// resolved [`VirtualKey`]. Empty ⇒ the compliance gate is OFF (byte-identical).
    /// A small `Vec<String>` — the same cheap-clone posture as `tenant_id`.
    pub compliance_frameworks: Vec<String>,
    /// The compliance enforcement posture ([ADR-035] §4). `Copy`; inert when
    /// `compliance_frameworks` is empty.
    pub compliance_mode: ComplianceMode,
}

impl TenantContext {
    /// Build the context for a resolved virtual key, subtracting the
    /// gateway-level `global_holdbacks` alongside the key's own.
    pub fn from_virtual_key(key: &VirtualKey, global_holdbacks: &BTreeSet<Feature>) -> Self {
        Self {
            tenant_id: key.resolved_tenant_id(),
            tier: key.tier,
            capabilities: key.capability_set(global_holdbacks),
            compliance_frameworks: key.compliance_frameworks.clone(),
            compliance_mode: key.compliance_mode,
        }
    }
}

/// The tenant's STARTUP-COMPILED Guardrails v2 default, injected as a request
/// extension by [`auth_middleware`] (G2.6). Always inserted (as `None` when the
/// key has no spec) so the proxy's extractor never 500s. The `Arc` clone is the
/// only per-request cost — regexes were compiled once at load.
/// Enterprise build: carries the startup-compiled advanced check engine.
#[cfg(feature = "enterprise")]
#[derive(Clone, Debug)]
pub struct TenantGuardrails(pub Option<Arc<CompiledGuardrails>>);

/// CE build: an always-`None` stub (the ce_stubs precedent — cf. `LedgerHandle`).
/// The compiled check ENGINE is a moat surface (ADR-088), so it is never present
/// on CE; but this extension is inserted and THREADED through the shared proxy
/// pipeline (`/v1/chat/completions`, `/v1/messages`, `/v1/prompts/*` all extract
/// it), so the TYPE must exist in both builds to keep those signatures identical.
/// The inner is uninhabited (`Infallible`) — structurally always `None`.
#[cfg(not(feature = "enterprise"))]
#[derive(Clone, Debug)]
pub struct TenantGuardrails(pub Option<std::convert::Infallible>);

/// Build the OpenFeature-shaped [`EvalContext`] for a key's tenant (PRD-013
/// FR-4): `targeting_key = tenant_id` plus the non-personal `tier`/`region`
/// attributes Unleash strategies target on (context-aware gradual rollout,
/// per-tenant/per-region kill-switches). Carries NO personal data — `tenant_id`
/// is an opaque identifier and `region`/`tier` are non-personal, the DPDP
/// posture PRD-014 FR-4 asserts. Built only when Unleash is configured, at
/// auth-time, off the hot path.
fn eval_context_for(key: &VirtualKey) -> EvalContext {
    let mut ctx = EvalContext::for_tenant(key.resolved_tenant_id())
        .with_attribute("tier", tier_attr(key.tier));
    // The tenant's default residency is its targeting region; omit when unset so
    // a region-scoped strategy simply doesn't match (no spurious "" region).
    if let Some(region) = key
        .default_residency
        .as_deref()
        .map(str::trim)
        .filter(|r| !r.is_empty())
    {
        ctx = ctx.with_attribute("region", region);
    }
    ctx
}

/// The stable lowercase flag-targeting string for a [`Tier`], matching the serde
/// wire form (`free`/`standard`/`enterprise`). Exhaustive match: a new tier
/// forces an update here rather than silently targeting on a wrong string.
fn tier_attr(tier: Tier) -> &'static str {
    match tier {
        Tier::Free => "free",
        Tier::Standard => "standard",
        Tier::Business => "business",
        Tier::Enterprise => "enterprise",
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct KeyRegistry {
    pub keys: Vec<VirtualKey>,
}

/// Typed startup error for loading the key registry (Task #3b). Replaces the
/// previous `.unwrap()` on parse, which would panic the whole process on a
/// malformed `keys.json` with an opaque message. `thiserror` is intentionally
/// NOT pulled in for a single error type — a hand-rolled `Display`/`Error` keeps
/// the dependency surface (and the frugality bar) where it is.
#[derive(Debug)]
pub enum AuthLoadError {
    /// The registry file could not be read (missing/permission).
    Read {
        path: String,
        source: std::io::Error,
    },
    /// The registry file was not valid JSON / not the expected shape.
    Parse {
        path: String,
        source: serde_json::Error,
    },
    /// The registry parsed but contains no keys — a gateway with no keys
    /// authenticates nobody, so we refuse to start.
    Empty { path: String },
    /// `RP_ROLLOUT_HOLDBACKS` names a feature that does not exist. Fail-closed:
    /// a typo'd holdback silently NOT holding back a half-finished feature is a
    /// release-control failure, so we refuse to start instead of ignoring it.
    Holdback { key: String },
    /// `RP_ROLLOUT_HOLDBACKS` is set but not valid UTF-8. Same fail-closed
    /// rationale as [`AuthLoadError::Holdback`]: a mangled value silently
    /// disabling EVERY global holdback is a release-control failure, so we
    /// refuse to start instead of treating it as "no holdbacks".
    HoldbackNotUnicode { value: std::ffi::OsString },
    /// A key's tenant-default `guardrails` spec failed to parse/compile (G2.6).
    /// Fail-closed: a tenant's blocking guardrail silently not loading is a
    /// security-control failure, so we refuse to start. MOAT (ADR-088): the
    /// compile step is enterprise-only, so this variant is too.
    #[cfg(feature = "enterprise")]
    Guardrails { key_name: String, reason: String },
}

impl std::fmt::Display for AuthLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AuthLoadError::Read { path, source } => {
                write!(f, "cannot read key registry '{path}': {source}")
            }
            AuthLoadError::Parse { path, source } => {
                write!(f, "cannot parse key registry '{path}': {source}")
            }
            AuthLoadError::Empty { path } => {
                write!(
                    f,
                    "key registry '{path}' contains no keys (refusing to start)"
                )
            }
            AuthLoadError::Holdback { key } => {
                write!(
                    f,
                    "RP_ROLLOUT_HOLDBACKS contains unknown feature key '{key}' (refusing to start)"
                )
            }
            AuthLoadError::HoldbackNotUnicode { value } => {
                write!(
                    f,
                    "RP_ROLLOUT_HOLDBACKS is set but not valid UTF-8: {value:?} (refusing to start)"
                )
            }
            #[cfg(feature = "enterprise")]
            AuthLoadError::Guardrails { key_name, reason } => {
                write!(
                    f,
                    "key '{key_name}' has an invalid guardrails spec: {reason} (refusing to start)"
                )
            }
        }
    }
}

impl std::error::Error for AuthLoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            AuthLoadError::Read { source, .. } => Some(source),
            AuthLoadError::Parse { source, .. } => Some(source),
            AuthLoadError::Empty { .. } => None,
            AuthLoadError::Holdback { .. } => None,
            AuthLoadError::HoldbackNotUnicode { .. } => None,
            #[cfg(feature = "enterprise")]
            AuthLoadError::Guardrails { .. } => None,
        }
    }
}

/// The in-memory virtual-key registry.
///
/// Held behind an [`ArcSwap`] at the call site so a future control-plane key
/// push can hot-swap the whole registry atomically without locking readers
/// (Task #7). The map itself is immutable once built.
#[derive(Debug)]
pub struct AuthState {
    pub keys: HashMap<String, VirtualKey>,
    /// Gateway-level rollout holdbacks — the OPERATIONAL `released(...)` half
    /// ([`branching-and-devex.md`] §6.4), parsed ONCE at startup from
    /// `RP_ROLLOUT_HOLDBACKS` (see [`global_holdbacks_from_env`]) and subtracted
    /// for EVERY tenant at resolution. Riding the same [`ArcSwap`] handle means
    /// a future control-plane push hot-swaps holdbacks atomically with the
    /// registry — no new synchronization, readers stay wait-free.
    pub global_holdbacks: BTreeSet<Feature>,
    /// Tenant-default Guardrails v2 configs (G2.6), compiled ONCE at load and
    /// keyed by `routeplane_key`. Riding the same `ArcSwap` snapshot keeps a
    /// future control-plane push atomic with the registry. Empty when no key
    /// declares a `guardrails` spec (today's production registry). MOAT
    /// (ADR-088): the compiled check engine rides `enterprise` — the field is
    /// absent on the CE build (the raw `guardrails` JSON stays on the key, simply
    /// never compiled).
    #[cfg(feature = "enterprise")]
    pub compiled_guardrails: HashMap<String, Arc<CompiledGuardrails>>,

    /// Optional Unleash-backed flag source (ADR-029 G3 / PRD-013 fork A) — the
    /// DYNAMIC half of the `released(...)` set that complements the static
    /// [`AuthState::global_holdbacks`]. Co-located here (rather than in
    /// `AppState`) because holdback composition happens at `auth_middleware`,
    /// which reads this `ArcSwap` snapshot — so registry + static holdbacks +
    /// the live flag source all hot-swap atomically, and `proxy.rs` (which never
    /// needs the provider) stays untouched.
    ///
    /// `None` ⇒ no Unleash configured ⇒ the auth path is byte-identical to the
    /// pure entitlement resolution (the `ab_parity` golden guard stays green).
    /// `Some` ⇒ for each [`Feature::all`], a per-context `!released` answer is
    /// unioned into the holdbacks at auth-time — an in-memory snapshot lookup,
    /// no per-request network, the hot path's lock-free gate unchanged.
    /// Attached AFTER `load_*` (in `main`, like `global_holdbacks`); the loaders
    /// always default it to `None`. Env-gated on `UNLEASH_API_URL`.
    pub unleash: Option<UnleashFlags>,
}

impl AuthState {
    /// Load the registry from disk. Returns a typed error (Task #3b) on a
    /// missing/unreadable file, a parse failure, an empty registry, or an
    /// invalid tenant guardrails spec (G2.6) — the caller (`main`) logs and
    /// refuses to start rather than running a gateway that can authenticate no
    /// one or silently drops a tenant's security checks.
    pub fn load_from_file(path: &str) -> Result<Self, AuthLoadError> {
        let content = fs::read_to_string(path).map_err(|source| AuthLoadError::Read {
            path: path.to_string(),
            source,
        })?;
        Self::load_from_json(&content, path)
    }

    /// Load the registry from a JSON string. `origin` names the source in
    /// errors (a file path, or `env:RP_KEYS_JSON` for the serverless path where
    /// the registry is injected as configuration instead of baked into the
    /// image — keys.json is gitignored, so a CI-built image has no key file).
    pub fn load_from_json(content: &str, origin: &str) -> Result<Self, AuthLoadError> {
        let registry: KeyRegistry =
            serde_json::from_str(content).map_err(|source| AuthLoadError::Parse {
                path: origin.to_string(),
                source,
            })?;

        if registry.keys.is_empty() {
            return Err(AuthLoadError::Empty {
                path: origin.to_string(),
            });
        }

        let mut keys = HashMap::new();
        // MOAT (ADR-088): compiling tenant-default guardrail specs is an
        // enterprise-only surface. On the CE build the raw `guardrails` JSON on a
        // key is simply never compiled (the field is absent, the block gone).
        #[cfg(feature = "enterprise")]
        let mut compiled_guardrails = HashMap::new();
        for key in registry.keys {
            #[cfg(feature = "enterprise")]
            if let Some(spec) = &key.guardrails {
                // Tenant config is the trusted source: webhook checks allowed.
                match CompiledGuardrails::parse(spec, ConfigSource::Tenant) {
                    Ok(compiled) => {
                        compiled_guardrails.insert(key.routeplane_key.clone(), Arc::new(compiled));
                    }
                    Err(e) => {
                        return Err(AuthLoadError::Guardrails {
                            key_name: key.name.clone(),
                            reason: e.to_string(),
                        });
                    }
                }
            }
            keys.insert(key.routeplane_key.clone(), key);
        }

        Ok(Self {
            keys,
            global_holdbacks: BTreeSet::new(),
            #[cfg(feature = "enterprise")]
            compiled_guardrails,
            // Attached post-load in `main` (env-gated on UNLEASH_API_URL), the
            // same pattern as `global_holdbacks`. A freshly loaded registry has
            // no flag source until then ⇒ byte-identical resolution.
            unleash: None,
        })
    }

    /// Constant-time virtual-key lookup (Task #3c).
    ///
    /// We scan every registered key and compare with `subtle::ConstantTimeEq` so
    /// the comparison's running time does not depend on how many leading bytes of
    /// the presented key match a real one — closing the timing side-channel that
    /// a short-circuiting `HashMap::get` + `==` would leak. The registry is
    /// small (one entry per tenant key), so the full scan is negligible, and we
    /// deliberately do NOT early-return on the first match: `found` is folded in
    /// constant time across all entries.
    pub fn lookup_constant_time(&self, presented: &str) -> Option<VirtualKey> {
        let presented_bytes = presented.as_bytes();
        let mut found: Option<&VirtualKey> = None;
        for (stored, vk) in self.keys.iter() {
            // `ct_eq` requires equal-length slices to be meaningful; differing
            // lengths can't match, but we still run ct_eq on the common framing
            // to avoid leaking via length-dependent branching beyond the
            // unavoidable length check.
            let is_match: bool = stored.as_bytes().ct_eq(presented_bytes).into();
            if is_match {
                found = Some(vk);
            }
        }
        found.cloned()
    }

    /// Resolve a virtual key's [`TenantContext`] under this state's full holdback
    /// composition (ADR-029 fork A / PRD-013 FR-3):
    ///
    /// ```text
    /// active = tier_baseline ∪ overrides − (global_holdbacks ∪ unleash_!released ∪ per_key)
    /// ```
    ///
    /// - No flag source (`unleash = None`) ⇒ the static path: byte-identical to
    ///   pure entitlement resolution, no clone and no per-request eval (the
    ///   `ab_parity` golden guard stays green when `UNLEASH_API_URL` is unset).
    /// - Flag source present ⇒ for each [`Feature::all`], ask Unleash "released
    ///   for this context?" with `default = true`; a `false` answer (and ONLY an
    ///   explicit `false`) unions the feature into a copy of the static
    ///   holdbacks. Each `resolve_bool` is an in-memory snapshot lookup — no
    ///   network, no lock — and a flag unknown to Unleash defaults released, so
    ///   entitlement alone decides it (FLAG_NOT_FOUND → default).
    ///
    /// `CapabilitySet::active` (the hot-path gate) is unchanged either way — the
    /// composition happens once here, off the hot path. Split out from
    /// `auth_middleware` so the fork-A composition is unit-testable offline (a
    /// memoized Unleash snapshot, no server).
    pub fn resolve_tenant_context(&self, key: &VirtualKey) -> TenantContext {
        match self.unleash.as_ref() {
            None => TenantContext::from_virtual_key(key, &self.global_holdbacks),
            Some(unleash) => {
                let ctx = eval_context_for(key);
                let mut holdbacks = self.global_holdbacks.clone();
                for feature in Feature::all() {
                    if !unleash.resolve_bool(feature.flag_key(), true, &ctx) {
                        holdbacks.insert(feature);
                    }
                }
                TenantContext::from_virtual_key(key, &holdbacks)
            }
        }
    }
}

/// Hot-swappable handle around [`AuthState`] (Task #7). Cloning the handle is a
/// cheap `Arc` bump; `load()` is wait-free for readers on the hot path. A future
/// control-plane integration calls `store()` to swap the registry atomically.
pub type SharedAuthState = Arc<ArcSwap<AuthState>>;

/// Build a hot-swappable handle from an initial [`AuthState`].
pub fn shared_auth_state(state: AuthState) -> SharedAuthState {
    Arc::new(ArcSwap::from_pointee(state))
}

// ---------------------------------------------------------------------------
// Auth-failure rate limiting (security gap R0.2)
// ---------------------------------------------------------------------------

/// Shared handle for the per-source-IP failed-auth tracker (security gap R0.2).
///
/// Injected as a request extension exactly like [`SharedAuthState`]. It is an
/// `Option` at the call site (see [`auth_failure_tracker_from_env`]): `None` ⇒ the
/// feature is OFF and [`auth_middleware`] is byte-identical to today (zero-cost
/// when disabled). When `Some`, the middleware consults it BEFORE the
/// constant-time key lookup and records every auth failure against the source IP.
pub type SharedAuthFailureTracker = Arc<AuthFailureTracker>;

/// Optional audit-ledger handle for the AUTH seam (R0.3 — security-event
/// logging). Injected as a request extension exactly like
/// [`SharedAuthFailureTracker`]: the auth middleware records auth-failure /
/// brute-force-throttle security events into the sovereign audit ledger when
/// present. Absent (ship-dark default = audit ledger disabled) ⇒ the whole
/// emission is skipped ⇒ byte-identical to today. There is no tenant context at
/// auth time, so events are `_global`-scoped and gated on handle presence only
/// (see `ledger_sink::record_security_global`).
pub type SharedLedgerHandle = Arc<Option<crate::ledger_sink::LedgerHandle>>;

/// Default trusted client-IP header. Cloudflare is the locked edge (ADR-027), so
/// `CF-Connecting-IP` is the authoritative client IP; the ACA origin only ever
/// sees Cloudflare's connection. Overridable via `RP_AUTH_FAILURE_IP_HEADER` for
/// a different edge/proxy.
const DEFAULT_CLIENT_IP_HEADER: &str = "cf-connecting-ip";

/// Build the optional auth-failure tracker from the environment. The feature is
/// OFF unless `RP_AUTH_FAILURE_LIMIT` is set to a truthy value (`on`/`true`/`1`),
/// keeping the default behaviour byte-identical (fail-safe-by-omission, the same
/// ship-dark doctrine as the audit ledger). When ON, the tunables fall back to
/// the [`AuthFailureConfig`] defaults (5 failures / 60s window / 1s base backoff,
/// capped at 5min).
///
/// Knobs (all optional, sane defaults):
///   * `RP_AUTH_FAILURE_THRESHOLD`     — failures per window before throttling (5)
///   * `RP_AUTH_FAILURE_WINDOW_MS`     — sliding-window length in ms (60000)
///   * `RP_AUTH_FAILURE_BACKOFF_MS`    — base backoff in ms, doubled per overshoot (1000)
///   * `RP_AUTH_FAILURE_BACKOFF_CAP_MS`— max backoff / Retry-After in ms (300000)
///   * `RP_AUTH_FAILURE_SLOTS`         — fixed atomic-slot count, bounded memory (4096)
///   * `RP_AUTH_FAILURE_IP_HEADER`     — trusted client-IP header (cf-connecting-ip)
pub fn auth_failure_tracker_from_env() -> Option<SharedAuthFailureTracker> {
    let enabled = std::env::var("RP_AUTH_FAILURE_LIMIT")
        .ok()
        .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "on" | "true" | "1"))
        .unwrap_or(false);
    if !enabled {
        return None;
    }
    let d = AuthFailureConfig::default();
    let cfg = AuthFailureConfig {
        threshold: env_u64("RP_AUTH_FAILURE_THRESHOLD", d.threshold),
        window_ms: env_u64("RP_AUTH_FAILURE_WINDOW_MS", d.window_ms),
        backoff_base_ms: env_u64("RP_AUTH_FAILURE_BACKOFF_MS", d.backoff_base_ms),
        backoff_cap_ms: env_u64("RP_AUTH_FAILURE_BACKOFF_CAP_MS", d.backoff_cap_ms),
        slots: env_usize("RP_AUTH_FAILURE_SLOTS", d.slots),
    };
    Some(Arc::new(AuthFailureTracker::new(cfg)))
}

fn env_u64(var: &str, default: u64) -> u64 {
    std::env::var(var)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(default)
}

fn env_usize(var: &str, default: usize) -> usize {
    std::env::var(var)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(default)
}

/// The trusted client-IP header name (lowercased), read once per request from the
/// env. Behind the Cloudflare edge (ADR-027) the only trustworthy source IP is the
/// edge-injected forwarded header — the ACA origin's socket peer is always
/// Cloudflare, so a raw `ConnectInfo` peer would bucket the whole world together.
fn client_ip_header_name() -> String {
    std::env::var("RP_AUTH_FAILURE_IP_HEADER")
        .ok()
        .map(|v| v.trim().to_ascii_lowercase())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| DEFAULT_CLIENT_IP_HEADER.to_string())
}

/// Extract the source-IP throttling key from the request headers.
///
/// Precedence:
///   1. The configured trusted header (`CF-Connecting-IP` by default) — a single
///      authoritative IP injected by the edge.
///   2. `X-Forwarded-For` FIRST hop (the original client; later hops are proxies).
///   3. `"unknown"` — a single shared bucket. Fail-closed: an attacker who strips
///      forwarding headers is throttled as one aggregate source rather than
///      bypassing the limiter entirely (the locked Cloudflare origin always
///      injects the header, so this only fires on misconfiguration / direct hits).
fn source_ip_key<'h>(headers: &'h axum::http::HeaderMap, trusted_header: &str) -> &'h str {
    if let Some(v) = headers
        .get(trusted_header)
        .and_then(|h| h.to_str().ok())
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        return v;
    }
    if let Some(xff) = headers.get("x-forwarded-for").and_then(|h| h.to_str().ok()) {
        // XFF is `client, proxy1, proxy2`; the first hop is the original client.
        let first = xff.split(',').next().map(str::trim).unwrap_or("");
        if !first.is_empty() {
            return first;
        }
    }
    "unknown"
}

/// Build the 429 throttle response for a brute-force-limited source. OpenAI-shaped
/// envelope (so an SDK surfaces a clean error) + `Retry-After` and a
/// `x-routeplane-limit-type: auth_failure` discriminator. Mirrors the
/// `retry-after` header style of `proxy.rs`'s rate-limit builder WITHOUT importing
/// it (auth.rs must not depend on the proxy orchestrator). No key material, no
/// source IP echoed (C4 — same no-leak doctrine as the 401 path).
fn auth_throttled_response(retry_after_secs: u64) -> Response {
    let body = serde_json::json!({
        "error": {
            "message": "Too many failed authentication attempts. Retry after the indicated delay.",
            "type": "invalid_request_error",
            "param": serde_json::Value::Null,
            "code": "auth_failure_rate_limited",
        }
    });
    let mut resp = (StatusCode::TOO_MANY_REQUESTS, Json(body)).into_response();
    let h = resp.headers_mut();
    if let Ok(v) = HeaderValue::from_str(&retry_after_secs.to_string()) {
        h.insert("retry-after", v);
    }
    h.insert(
        "x-routeplane-limit-type",
        HeaderValue::from_static("auth_failure"),
    );
    resp
}

/// Read the gateway-level rollout holdback set from `RP_ROLLOUT_HOLDBACKS`
/// (comma-separated flag keys, e.g. `agentic_security,finops_export`), parsed
/// ONCE at startup — the interim `released(...)` half until the control plane
/// lands ([`branching-and-devex.md`] §6.4): one operational env-var flip holds a
/// feature back for every tenant, never N tenant key edits.
///
/// An absent or empty variable is an empty set (inert by default, mirroring the
/// serde-default pattern on the key record). An unknown feature key is
/// [`AuthLoadError::Holdback`]; a set-but-non-UTF-8 value is
/// [`AuthLoadError::HoldbackNotUnicode`] — in both cases the caller (`main`)
/// refuses to start. Only genuine absence is inert: mapping a mangled value to
/// "no holdbacks" would silently un-hold-back every globally held feature.
pub fn global_holdbacks_from_env() -> Result<BTreeSet<Feature>, AuthLoadError> {
    holdbacks_from_var(std::env::var("RP_ROLLOUT_HOLDBACKS"))
}

/// Pure half of [`global_holdbacks_from_env`]: maps the `std::env::var` result
/// to the holdback set. Split out so the `VarError` handling is unit-testable
/// without process-global env-var mutation (`set_var` is unsafe/racy under
/// concurrent test threads).
fn holdbacks_from_var(
    var: Result<String, std::env::VarError>,
) -> Result<BTreeSet<Feature>, AuthLoadError> {
    match var {
        Ok(raw) => parse_holdbacks(&raw),
        Err(std::env::VarError::NotPresent) => Ok(BTreeSet::new()),
        Err(std::env::VarError::NotUnicode(value)) => {
            Err(AuthLoadError::HoldbackNotUnicode { value })
        }
    }
}

/// Parse a comma-separated holdback list. Entries are trimmed; empty entries
/// are skipped; an unknown flag key fails closed with
/// [`AuthLoadError::Holdback`]. Split out from the env read so the parse rules
/// are unit-testable without process-global env-var races.
fn parse_holdbacks(raw: &str) -> Result<BTreeSet<Feature>, AuthLoadError> {
    let mut holdbacks = BTreeSet::new();
    for entry in raw.split(',') {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }
        match Feature::from_flag_key(entry) {
            Some(feature) => {
                holdbacks.insert(feature);
            }
            None => {
                return Err(AuthLoadError::Holdback {
                    key: entry.to_string(),
                });
            }
        }
    }
    Ok(holdbacks)
}

/// The header a presented gateway credential arrived on (ADR-041 observability
/// hook). Emitted as the `auth_source` field on a successful authentication so
/// we can measure the share of inbound auth that uses the OpenAI-SDK-compatible
/// `Authorization: Bearer` fallback vs the branded `x-routeplane-api-key`
/// primary. A pure tag — no behavioural effect; both sources resolve through the
/// identical lookup path (C2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AuthSource {
    /// The branded primary header `x-routeplane-api-key`.
    Native,
    /// The `Authorization: Bearer rp_…` SDK-compat fallback.
    Bearer,
}

impl AuthSource {
    fn as_str(self) -> &'static str {
        match self {
            AuthSource::Native => "native",
            AuthSource::Bearer => "bearer",
        }
    }
}

/// Extract the gateway-key candidate from the inbound `Authorization` header,
/// ADR-041 C2. We return a candidate **only** for `Bearer ` + an `rp_`-prefixed
/// value:
///   * the scheme is matched case-insensitively (`Bearer`/`bearer`/…) with a
///     single separating space, per RFC 7235;
///   * the token is trimmed;
///   * it must carry the `rp_` prefix — Routeplane's gateway-key brand.
///
/// A non-`rp_` Bearer (`sk-…`, `gsk_…`), a `Basic …` credential, or a bare token
/// is treated as **absent**: `None`. Such a value is never fed to the lookup and
/// is never guessed to be a provider key (no passthrough path exists — provider
/// keys are resolved server-side from the virtual key, `auth.rs` `provider_keys`).
/// The returned candidate flows into the SAME `lookup_constant_time` +
/// `resolve_tenant_context` path as the native header, preserving the
/// timing-side-channel property identically.
///
/// The `bool` return companion reports whether a NON-`rp_` Bearer was present —
/// the ADR-041 demand signal: an `sk-…` in `Authorization` is an attempted
/// provider-key passthrough. It drives a distinct `tracing::debug!` on failure
/// (still a 401), telling us whether a future passthrough mode is worth an ADR.
fn bearer_gateway_candidate(header: Option<&str>) -> (Option<&str>, bool) {
    let Some(raw) = header else {
        return (None, false);
    };
    // Split the scheme from the credential on the first space. Anything that is
    // not `Bearer <token>` (e.g. `Basic …`, a bare token, an empty value) is not
    // a Bearer credential at all → absent, not a non-`rp_`-Bearer signal.
    let Some((scheme, token)) = raw.split_once(' ') else {
        return (None, false);
    };
    if !scheme.eq_ignore_ascii_case("Bearer") {
        return (None, false);
    }
    let token = token.trim();
    if token.starts_with("rp_") {
        (Some(token), false)
    } else {
        // A Bearer credential that is present but not an `rp_` gateway key: an
        // attempted provider-key passthrough (`sk-…`) or a malformed/empty token.
        // Absent for auth, but flagged so the failure path can emit the signal.
        (None, true)
    }
}

/// Resolve the inbound gateway-key candidate, ADR-041 C1 precedence (fixed):
///   1. `x-routeplane-api-key` if present and NON-EMPTY after trim (a
///      whitespace-only native header is treated as absent so it cannot open a
///      smuggling seam, and falls through to the Bearer fallback);
///   2. else `Authorization: Bearer rp_…` (C2);
///   3. else `None` → 401.
///
/// The native header ALWAYS wins when present and non-empty; `Authorization` is
/// consulted ONLY as a fallback. We never "try both and accept whichever
/// validates" (a precedence-bypass / authz-confusion vector). The second tuple
/// element is the non-`rp_`-Bearer demand signal from [`bearer_gateway_candidate`].
fn resolve_gateway_credential<'h>(
    native: Option<&'h str>,
    authorization: Option<&'h str>,
) -> (Option<(&'h str, AuthSource)>, bool) {
    // Rule 1: native header wins when non-empty after trim. We pass the TRIMMED
    // value to the lookup so a header padded with surrounding whitespace cannot
    // diverge from the branded key (the constant-time lookup is exact-match).
    if let Some(native) = native {
        let trimmed = native.trim();
        if !trimmed.is_empty() {
            return (Some((trimmed, AuthSource::Native)), false);
        }
    }
    // Rule 2: Authorization: Bearer rp_… fallback (only when native is absent/empty).
    let (candidate, non_rp_bearer) = bearer_gateway_candidate(authorization);
    match candidate {
        Some(token) => (Some((token, AuthSource::Bearer)), non_rp_bearer),
        None => (None, non_rp_bearer),
    }
}

pub async fn auth_middleware(mut req: Request<Body>, next: Next) -> Result<Response, Response> {
    // The registry handle is the hot-swappable `SharedAuthState` (Task #7). We
    // `load()` the current pointer wait-free (no lock), so a concurrent
    // control-plane `store()` of a new registry never blocks a request.
    let shared = {
        let extensions = req.extensions();
        extensions
            .get::<SharedAuthState>()
            .cloned()
            .ok_or_else(crate::api_error::internal_error)?
    };
    let state = shared.load();

    // Auth-failure rate limiting (security gap R0.2). Optional, ship-dark: the
    // tracker rides the request extensions only when enabled, so an absent handle
    // ⇒ the whole block is skipped ⇒ byte-identical to today. When present we:
    //   1. resolve the source IP from the trusted edge header (Cloudflare),
    //   2. SHORT-CIRCUIT with 429 + Retry-After BEFORE the key lookup if the
    //      source is already over its failure threshold (saves the constant-time
    //      scan entirely for a known brute-forcer),
    //   3. record any genuine auth failure below against that source.
    // All operations are lock-free atomics with an injected clock (one
    // `now_unix_ms()` read), so the gate stays within the ADR-023 hot-path budget.
    let failure_tracker = req.extensions().get::<SharedAuthFailureTracker>().cloned();
    // R0.3: optional audit-ledger handle for security-event logging. Absent
    // extension OR an inner `None` (ship-dark default = ledger disabled) ⇒ a
    // zero-work no-op, byte-identical. We flatten to an owned `Option<LedgerHandle>`
    // so the `record_security_global` calls below pass `&Option<LedgerHandle>`.
    // A synthesized correlation id stands in for the request id (auth runs
    // before the proxy assigns one).
    let security_ledger: Option<crate::ledger_sink::LedgerHandle> = req
        .extensions()
        .get::<SharedLedgerHandle>()
        .and_then(|h| (**h).clone());
    let security_event_id = || format!("authsec_{}", uuid::Uuid::new_v4().simple());
    let throttle_source: Option<String> = failure_tracker.as_ref().map(|_| {
        let header = client_ip_header_name();
        source_ip_key(req.headers(), &header).to_string()
    });
    if let (Some(tracker), Some(source)) = (failure_tracker.as_ref(), throttle_source.as_deref()) {
        let now = now_unix_ms();
        if let AuthThrottle::Throttled { retry_after_secs } = tracker.check(source, now) {
            // No key material, no source IP echoed to the client (C4). The hashed
            // count is logged server-side only for an operator/security signal.
            tracing::warn!(
                retry_after_secs,
                "Auth-failure rate limit tripped: throttling repeated failed auth from source"
            );
            // R0.3: record the brute-force throttle trip. NO source IP, NO key
            // material — only the category + outcome + the Retry-After seconds as
            // an opaque count. `_global`-scoped (no tenant at auth time).
            crate::ledger_sink::record_security_global(&security_ledger, || {
                crate::ledger_sink::security_event(
                    &security_event_id(),
                    None,
                    crate::ledger_sink::SecurityCategory::AuthThrottle,
                    crate::ledger_sink::SecurityOutcome::Throttle,
                    Some(retry_after_secs),
                    None,
                )
            });
            return Err(auth_throttled_response(retry_after_secs));
        }
    }

    // Record one auth failure against the source IP (no-op when the feature is
    // off). Centralised so every fail-closed return path below records exactly
    // once. R0.3 SEAM: the post-increment count is also the hook for the
    // security-event ledger entry — one bounded `try_send`, no source IP / key
    // material, `_global`-scoped (no tenant resolved on a failed auth).
    let record_failure = |tracker: &Option<SharedAuthFailureTracker>| {
        let count = match (tracker.as_ref(), throttle_source.as_deref()) {
            (Some(t), Some(source)) => Some(t.record_failure(source, now_unix_ms())),
            _ => None,
        };
        crate::ledger_sink::record_security_global(&security_ledger, || {
            crate::ledger_sink::security_event(
                &security_event_id(),
                None,
                crate::ledger_sink::SecurityCategory::AuthFailure,
                crate::ledger_sink::SecurityOutcome::Deny,
                count,
                None,
            )
        });
    };

    // ADR-041 C1/C2: resolve the gateway credential with fixed precedence —
    // `x-routeplane-api-key` (branded primary) then `Authorization: Bearer rp_…`
    // (OpenAI-SDK-compat fallback). Header reads are two `headers.get` + a
    // prefix check; off the synchronous completion hot path, lock-free.
    let native = req
        .headers()
        .get("x-routeplane-api-key")
        .and_then(|header| header.to_str().ok());
    let authorization = req
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|header| header.to_str().ok());
    let (credential, non_rp_bearer) = resolve_gateway_credential(native, authorization);

    match credential {
        Some((key, source)) => {
            // Constant-time lookup (Task #3c): the comparison time does not
            // depend on how much of a presented key matches a real one. Both the
            // native and Bearer candidates flow through this SAME path (C2), so
            // the timing-side-channel and tenant/residency resolution are
            // identical regardless of which header carried the key.
            if let Some(virtual_key) = state.lookup_constant_time(key) {
                // [ADR-050] D6: fail closed on a non-Active tenant (suspended /
                // closed / pending). One comparison on a field already loaded
                // with the key — lock-free, no I/O, within the [ADR-023] hot-path
                // budget. Returns the SAME 401 invalid_api_key envelope as an
                // unknown key (C4 — no key material, no existence/state leak).
                if !virtual_key.lifecycle_state.serves_traffic() {
                    record_failure(&failure_tracker);
                    tracing::warn!("Unauthorized request: tenant not active");
                    return Err(crate::api_error::unauthorized());
                }
                // Resolve the tenant context (tenant_id + CapabilitySet) ONCE,
                // here at auth — the [ADR-012] §6.2 seam. tier_baseline ∪
                // overrides − holdbacks is a pure, allocation-light computation;
                // we do it on the auth path so the hot path only ever does a
                // lock-free set-membership test (`capabilities.active(..)`). The
                // fork-A Unleash composition (FR-3) folds in here too — see
                // [`AuthState::resolve_tenant_context`]. Both the registry and
                // the (static + dynamic) holdback sources ride the same loaded
                // snapshot, so resolution sees a consistent view.
                let tenant_ctx = state.resolve_tenant_context(&virtual_key);
                // Startup-compiled tenant guardrails (G2.6) ride the same
                // snapshot. The constant-time scan only matches on byte
                // equality, so indexing the side map by the PRESENTED (trimmed)
                // key is exact. Always inserted (None when unset) so the proxy's
                // extractor can never 500.
                // The tenant-guardrails extension is ALWAYS inserted (the proxy,
                // messages, and prompts handlers all extract it). On the CE build
                // the compiled check engine is a moat surface (ADR-088), so the
                // stub is structurally `None`; on enterprise it carries the
                // startup-compiled config for this key.
                #[cfg(feature = "enterprise")]
                let tenant_guardrails =
                    TenantGuardrails(state.compiled_guardrails.get(key).cloned());
                #[cfg(not(feature = "enterprise"))]
                let tenant_guardrails = TenantGuardrails(None);
                // ADR-041 observability hook: `auth_source` (native|bearer) is a
                // structured field — no metrics seam exists in auth.rs today, so
                // a tracing field is the cheap, no-standing-cost signal.
                tracing::debug!(
                    auth_source = source.as_str(),
                    "Authenticated key '{}' (tenant={} tier={:?} active_features={})",
                    virtual_key.name,
                    tenant_ctx.tenant_id,
                    tenant_ctx.tier,
                    tenant_ctx.capabilities.len(),
                );
                // Inject BOTH the existing VirtualKey (unchanged, for provider-
                // key resolution) AND the resolved TenantContext (the new seam).
                req.extensions_mut().insert(virtual_key);
                req.extensions_mut().insert(tenant_ctx);
                req.extensions_mut().insert(tenant_guardrails);
                Ok(next.run(req).await)
            } else {
                // Fail closed (C4): the OpenAI-shaped 401 envelope, no key
                // material logged (failure strings only).
                record_failure(&failure_tracker);
                tracing::warn!("Unauthorized request: Invalid API key");
                Err(crate::api_error::unauthorized())
            }
        }
        None => {
            // ADR-041 demand signal: a NON-`rp_` Bearer (an `sk-…` provider key
            // attempted as passthrough) is present but unusable. Distinct debug
            // event — never logs the value (C4) — so a climbing rate tells us
            // whether a future per-request passthrough mode is worth an ADR.
            // Still a 401.
            if non_rp_bearer {
                tracing::debug!(
                    bearer_non_rp = true,
                    "Unauthorized request: Authorization Bearer present but not an rp_ gateway key"
                );
            }
            record_failure(&failure_tracker);
            tracing::warn!("Unauthorized request: Missing or empty x-routeplane-api-key");
            Err(crate::api_error::unauthorized())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_key(json: &str) -> VirtualKey {
        serde_json::from_str(json).expect("key should deserialize")
    }

    // --- Model Catalog DP enforcement (PRD-008 FR-3/FR-4), re-added atop #170 ---
    fn catalog_key() -> VirtualKey {
        base_key(
            r#"{
                "name":"Catalog Key","routeplane_key":"rp_cat","provider_keys":{},
                "tenant_id":"t_cat","tier":"standard",
                "provisioned_models":["gpt-4o-mini"],
                "integrations":{
                    "azure-openai-in":{"adapter":"azure_openai","resident_regions":["IN"],"models":["gpt-4o"]},
                    "hf":{"adapter":"openai","models":["meta-llama/Llama-3-70b"]}
                }
            }"#,
        )
    }

    #[test]
    fn provisioned_allowlist_is_default_deny() {
        let key = catalog_key();
        assert!(key.is_model_provisioned("gpt-4o-mini"));
        assert!(!key.is_model_provisioned("gpt-4o")); // not in the bare allowlist
                                                      // A legacy key (empty allowlist) permits NOTHING under the catalog gate.
        assert!(
            !base_key(r#"{"name":"k","routeplane_key":"rp_l","provider_keys":{}}"#)
                .is_model_provisioned("gpt-4o")
        );
    }

    #[test]
    fn resolve_slug_routes_a_provisioned_address() {
        match catalog_key().resolve_slug("@azure-openai-in/gpt-4o") {
            SlugResolution::Resolved { integration, model } => {
                assert_eq!(integration.adapter, "azure_openai");
                assert_eq!(integration.resident_regions, vec!["IN".to_string()]);
                assert_eq!(model, "gpt-4o");
            }
            other => panic!("expected Resolved, got {other:?}"),
        }
    }

    #[test]
    fn resolve_slug_preserves_internal_slashes_in_model_id() {
        match catalog_key().resolve_slug("@hf/meta-llama/Llama-3-70b") {
            SlugResolution::Resolved { integration, model } => {
                assert_eq!(integration.adapter, "openai");
                assert_eq!(model, "meta-llama/Llama-3-70b");
            }
            other => panic!("expected Resolved, got {other:?}"),
        }
    }

    #[test]
    fn resolve_slug_is_default_deny_and_notaslug() {
        let key = catalog_key();
        assert_eq!(key.resolve_slug("gpt-4o-mini"), SlugResolution::NotASlug);
        assert_eq!(key.resolve_slug("@unknown/gpt-4o"), SlugResolution::Denied);
        assert_eq!(
            key.resolve_slug("@azure-openai-in/gpt-4-turbo"),
            SlugResolution::Denied
        );
        assert_eq!(key.resolve_slug("@azure-openai-in"), SlugResolution::Denied);
        assert_eq!(key.resolve_slug("@/gpt-4o"), SlugResolution::Denied);
    }

    fn vk(key: &str) -> VirtualKey {
        VirtualKey {
            name: "k".into(),
            routeplane_key: key.into(),
            provider_keys: HashMap::new(),
            tenant_id: None,
            lifecycle_state: TenantState::Active,
            tier: Tier::Free,
            capability_overrides: BTreeSet::new(),
            rollout_holdbacks: BTreeSet::new(),
            default_residency: None,
            guardrails: None,
            limits: None,
            compliance_frameworks: Vec::new(),
            compliance_mode: ComplianceMode::Strict,
            provisioned_models: Default::default(),
            integrations: Default::default(),
        }
    }

    fn auth_with(keys: &[&str]) -> AuthState {
        let mut map = HashMap::new();
        for k in keys {
            map.insert(k.to_string(), vk(k));
        }
        AuthState {
            keys: map,
            global_holdbacks: BTreeSet::new(),
            #[cfg(feature = "enterprise")]
            compiled_guardrails: HashMap::new(),
            unleash: None,
        }
    }

    // --- Task #3b: typed startup errors --------------------------------------

    #[test]
    fn missing_file_is_read_error() {
        let err = AuthState::load_from_file("/nonexistent/path/keys.json").unwrap_err();
        assert!(matches!(err, AuthLoadError::Read { .. }));
    }

    #[test]
    fn empty_registry_is_refused() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("rp_keys_empty_{}.json", std::process::id()));
        std::fs::write(&path, r#"{"keys": []}"#).unwrap();
        let err = AuthState::load_from_file(path.to_str().unwrap()).unwrap_err();
        assert!(matches!(err, AuthLoadError::Empty { .. }));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn malformed_json_is_parse_error() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("rp_keys_bad_{}.json", std::process::id()));
        std::fs::write(&path, "{ not json ").unwrap();
        let err = AuthState::load_from_file(path.to_str().unwrap()).unwrap_err();
        assert!(matches!(err, AuthLoadError::Parse { .. }));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn valid_registry_loads() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("rp_keys_ok_{}.json", std::process::id()));
        std::fs::write(
            &path,
            r#"{"keys":[{"name":"k","routeplane_key":"rp_x","provider_keys":{}}]}"#,
        )
        .unwrap();
        let state = AuthState::load_from_file(path.to_str().unwrap()).unwrap();
        assert!(state.lookup_constant_time("rp_x").is_some());
        #[cfg(feature = "enterprise")]
        assert!(state.compiled_guardrails.is_empty());
        let _ = std::fs::remove_file(&path);
    }

    // --- G2.6: tenant-default guardrails compile at load ----------------------

    // MOAT (ADR-088): tenant-default guardrail compilation is enterprise-only.
    #[cfg(feature = "enterprise")]
    #[test]
    fn tenant_guardrails_compile_at_load() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("rp_keys_guard_ok_{}.json", std::process::id()));
        std::fs::write(
            &path,
            r#"{"keys":[{
                "name":"g","routeplane_key":"rp_g","provider_keys":{},
                "guardrails":{
                    "before_request":[{"id":"b","type":"regex","pattern":"blocked"}],
                    "after_request":[{"id":"w","type":"webhook","action":"observe","url":"https://hooks.example.com/v"}]
                }
            }]}"#,
        )
        .unwrap();
        let state = AuthState::load_from_file(path.to_str().unwrap()).unwrap();
        let compiled = state.compiled_guardrails.get("rp_g").expect("compiled");
        assert!(compiled.has_checks(routeplane_guardrails::Hook::BeforeRequest));
        assert!(compiled.has_checks(routeplane_guardrails::Hook::AfterRequest));
        let _ = std::fs::remove_file(&path);
    }

    #[cfg(feature = "enterprise")]
    #[test]
    fn invalid_tenant_guardrails_refuse_start() {
        // Fail-closed: a tenant's blocking guardrail silently not compiling is
        // a security-control failure — same doctrine as a typo'd holdback.
        let dir = std::env::temp_dir();
        let path = dir.join(format!("rp_keys_guard_bad_{}.json", std::process::id()));
        std::fs::write(
            &path,
            r#"{"keys":[{
                "name":"Bad Guardrails Key","routeplane_key":"rp_bad","provider_keys":{},
                "guardrails":{"before_request":[{"type":"regex","pattern":"("}]}
            }]}"#,
        )
        .unwrap();
        let err = AuthState::load_from_file(path.to_str().unwrap()).unwrap_err();
        assert!(
            matches!(err, AuthLoadError::Guardrails { ref key_name, .. } if key_name == "Bad Guardrails Key")
        );
        let _ = std::fs::remove_file(&path);
    }

    // --- PRD-008: limits field is serde-default (legacy keys byte-identical) ---

    #[test]
    fn legacy_key_without_limits_resolves_to_none() {
        let key = base_key(r#"{"name":"k","routeplane_key":"rp_x","provider_keys":{}}"#);
        assert!(key.limits.is_none());
    }

    #[test]
    fn key_with_limits_deserializes() {
        let key = base_key(
            r#"{
                "name":"Budgeted","routeplane_key":"rp_lim","provider_keys":{},
                "tenant_id":"t_lim","tier":"standard",
                "limits":{
                    "key":{"rate":{"requests_per_min":60},"budget":{"cost_micro_usd_daily":5000000}},
                    "tenant":{"rate":{"requests_per_min":600}}
                }
            }"#,
        );
        let limits = key.limits.expect("limits present");
        assert!(limits.key.is_some());
        assert!(limits.tenant.is_some());
    }

    #[test]
    fn legacy_key_without_guardrails_resolves_to_none() {
        let key = base_key(r#"{"name":"k","routeplane_key":"rp_x","provider_keys":{}}"#);
        assert!(key.guardrails.is_none());
    }

    // --- Task #3c: constant-time lookup --------------------------------------

    #[test]
    fn constant_time_lookup_matches_exact_key_only() {
        let state = auth_with(&["rp_secret_abc", "rp_other_def"]);
        assert!(state.lookup_constant_time("rp_secret_abc").is_some());
        assert!(state.lookup_constant_time("rp_other_def").is_some());
        // Prefix of a real key must NOT match.
        assert!(state.lookup_constant_time("rp_secret_ab").is_none());
        // Wrong key must not match.
        assert!(state.lookup_constant_time("rp_nope").is_none());
        // Empty must not match.
        assert!(state.lookup_constant_time("").is_none());
    }

    // --- Task #6: server-side default residency, narrow-not-disable ----------

    #[test]
    fn default_residency_applies_without_header() {
        let mut k = vk("rp_x");
        k.default_residency = Some("IN".into());
        assert_eq!(k.effective_requested_region(None).as_deref(), Some("IN"));
        // Empty header does NOT disable the default.
        assert_eq!(
            k.effective_requested_region(Some("  ")).as_deref(),
            Some("IN")
        );
    }

    #[test]
    fn header_narrows_default_but_no_default_means_header_only() {
        let mut k = vk("rp_x");
        k.default_residency = Some("IN".into());
        // Header narrows to a different region.
        assert_eq!(
            k.effective_requested_region(Some("EU")).as_deref(),
            Some("EU")
        );

        let k2 = vk("rp_y"); // no default
        assert_eq!(k2.effective_requested_region(None), None);
        assert_eq!(
            k2.effective_requested_region(Some("US")).as_deref(),
            Some("US")
        );
    }

    #[test]
    fn legacy_key_without_entitlement_fields_loads_as_free_core_only() {
        // The exact shape of today's "Default Development Key" record — no
        // tenant_id/tier/overrides. Must still load (serde defaults) and resolve
        // to Free / core-only / tenant_id == name.
        let key = base_key(
            r#"{
                "name": "Default Development Key",
                "routeplane_key": "rp_dev_123456789",
                "provider_keys": { "openai": "env:OPENAI_API_KEY" }
            }"#,
        );
        assert_eq!(key.tier, Tier::Free);
        assert!(key.capability_overrides.is_empty());
        assert!(key.rollout_holdbacks.is_empty());

        let ctx = TenantContext::from_virtual_key(&key, &BTreeSet::new());
        assert_eq!(ctx.tenant_id, "Default Development Key"); // falls back to name
                                                              // The Free baseline carries exactly {RoutingPolicy (F13 core surface
                                                              // with a holdback kill switch), TokenCompression (ADR-088 Bundle B,
                                                              // release-plane gated at rollout)} — no other features.
        assert!(ctx.capabilities.active(Feature::RoutingPolicy));
        assert!(ctx.capabilities.active(Feature::TokenCompression));
        assert_eq!(ctx.capabilities.len(), 2);
        assert!(!ctx.capabilities.active(Feature::SemanticCache));
    }

    #[test]
    fn paid_key_resolves_tier_baseline_and_override() {
        let key = base_key(
            r#"{
                "name": "Paid Tier Key",
                "routeplane_key": "rp_paid_abc",
                "provider_keys": { "openai": "env:OPENAI_API_KEY" },
                "tenant_id": "t_paid",
                "tier": "standard",
                "capability_overrides": ["finops_export"]
            }"#,
        );
        let ctx = TenantContext::from_virtual_key(&key, &BTreeSet::new());
        assert_eq!(ctx.tenant_id, "t_paid");
        assert_eq!(ctx.tier, Tier::Standard);
        // Standard baseline:
        assert!(ctx.capabilities.active(Feature::SemanticCache));
        assert!(ctx.capabilities.active(Feature::AdvancedGuardrails));
        assert!(ctx.capabilities.active(Feature::PromptRegistry));
        // From the override (add-on):
        assert!(ctx.capabilities.active(Feature::FinOpsExport));
        // Not granted:
        assert!(!ctx.capabilities.active(Feature::AgenticSecurity));
    }

    #[test]
    fn holdback_on_key_subtracts_even_an_entitled_feature() {
        let key = base_key(
            r#"{
                "name": "Held Back Key",
                "routeplane_key": "rp_hold_xyz",
                "provider_keys": {},
                "tier": "enterprise",
                "rollout_holdbacks": ["agentic_security"]
            }"#,
        );
        let ctx = TenantContext::from_virtual_key(&key, &BTreeSet::new());
        // Enterprise grants agentic_security, but the holdback removes it.
        assert!(!ctx.capabilities.active(Feature::AgenticSecurity));
        // Other Enterprise features remain.
        assert!(ctx.capabilities.active(Feature::SemanticCache));
    }

    // --- Gateway-level rollout holdbacks (RP_ROLLOUT_HOLDBACKS) ----------------

    #[test]
    fn global_holdback_subtracts_a_tier_baseline_feature() {
        // Enterprise grants agentic_security; a GLOBAL holdback removes it for
        // this tenant with NO per-key holdback — the operational released(...)
        // half applies to every tenant.
        let key = base_key(
            r#"{
                "name": "Enterprise Key",
                "routeplane_key": "rp_ent_abc",
                "provider_keys": {},
                "tier": "enterprise"
            }"#,
        );
        assert!(key.rollout_holdbacks.is_empty());

        let global = BTreeSet::from([Feature::AgenticSecurity]);
        let ctx = TenantContext::from_virtual_key(&key, &global);
        assert!(!ctx.capabilities.active(Feature::AgenticSecurity));
        // Other Enterprise features remain.
        assert!(ctx.capabilities.active(Feature::SemanticCache));
        assert!(ctx.capabilities.active(Feature::FinOpsExport));
    }

    #[test]
    fn global_and_per_key_holdbacks_union_and_both_subtract() {
        let key = base_key(
            r#"{
                "name": "Enterprise Key",
                "routeplane_key": "rp_ent_xyz",
                "provider_keys": {},
                "tier": "enterprise",
                "rollout_holdbacks": ["finops_export"]
            }"#,
        );
        let global = BTreeSet::from([Feature::AgenticSecurity]);
        let ctx = TenantContext::from_virtual_key(&key, &global);
        // Subtracted by the global holdback:
        assert!(!ctx.capabilities.active(Feature::AgenticSecurity));
        // Subtracted by the per-key escape hatch (backward compat):
        assert!(!ctx.capabilities.active(Feature::FinOpsExport));
        // The rest of the Enterprise baseline remains.
        assert!(ctx.capabilities.active(Feature::SemanticCache));
        assert!(ctx.capabilities.active(Feature::AdvancedGuardrails));
        assert!(ctx.capabilities.active(Feature::PromptRegistry));
    }

    // --- FR-3/FR-4: Unleash fork-A holdback composition (offline) -------------

    /// Build an offline Unleash flag source seeded with a boolean snapshot. The
    /// localhost url is never dialed — eval reads the in-memory snapshot only (no
    /// `spawn_refresh` here), exactly the in-process path the gateway uses.
    fn unleash_with(flags: &[(&str, bool)]) -> UnleashFlags {
        let u = UnleashFlags::new("http://127.0.0.1:0/api", "rp-test", "offline", None)
            .expect("build offline unleash client");
        u.memoize_bools(flags).expect("seed snapshot");
        u
    }

    fn auth_state_with(
        key: VirtualKey,
        global: BTreeSet<Feature>,
        unleash: Option<UnleashFlags>,
    ) -> AuthState {
        let mut keys = HashMap::new();
        keys.insert(key.routeplane_key.clone(), key);
        AuthState {
            keys,
            global_holdbacks: global,
            #[cfg(feature = "enterprise")]
            compiled_guardrails: HashMap::new(),
            unleash,
        }
    }

    #[test]
    fn unleash_holdback_subtracts_a_released_false_feature() {
        // Unleash holds back semantic_cache (released=false); agentic_security is
        // NOT in the snapshot ⇒ unknown ⇒ released ⇒ entitlement alone decides.
        let key = base_key(
            r#"{"name":"E","routeplane_key":"rp_e","provider_keys":{},"tenant_id":"t_e","tier":"enterprise"}"#,
        );
        let state = auth_state_with(
            key.clone(),
            BTreeSet::new(),
            Some(unleash_with(&[("semantic_cache", false)])),
        );
        let ctx = state.resolve_tenant_context(&key);
        assert!(!ctx.capabilities.active(Feature::SemanticCache)); // held back by Unleash
        assert!(ctx.capabilities.active(Feature::AgenticSecurity)); // unknown → released
        assert!(ctx.capabilities.active(Feature::AdvancedGuardrails));
    }

    #[test]
    fn unleash_and_global_holdbacks_compose() {
        // active = baseline − (global ∪ unleash): semantic_cache from Unleash,
        // agentic_security from the static global set; both subtract.
        let key = base_key(
            r#"{"name":"E","routeplane_key":"rp_e2","provider_keys":{},"tier":"enterprise"}"#,
        );
        let state = auth_state_with(
            key.clone(),
            BTreeSet::from([Feature::AgenticSecurity]),
            Some(unleash_with(&[("semantic_cache", false)])),
        );
        let ctx = state.resolve_tenant_context(&key);
        assert!(!ctx.capabilities.active(Feature::SemanticCache)); // unleash
        assert!(!ctx.capabilities.active(Feature::AgenticSecurity)); // global
        assert!(ctx.capabilities.active(Feature::PromptRegistry)); // neither
        assert!(ctx.capabilities.active(Feature::FinOpsExport)); // neither
    }

    #[test]
    fn unleash_released_true_holds_back_nothing() {
        // Every flag explicitly released=true ⇒ no dynamic holdbacks ⇒ the full
        // Enterprise baseline survives.
        let key = base_key(
            r#"{"name":"E","routeplane_key":"rp_e3","provider_keys":{},"tier":"enterprise"}"#,
        );
        let state = auth_state_with(
            key.clone(),
            BTreeSet::new(),
            Some(unleash_with(&[
                ("semantic_cache", true),
                ("agentic_security", true),
            ])),
        );
        let ctx = state.resolve_tenant_context(&key);
        assert!(ctx.capabilities.active(Feature::SemanticCache));
        assert!(ctx.capabilities.active(Feature::AgenticSecurity));
    }

    #[test]
    fn no_unleash_resolves_pure_entitlement_byte_identical() {
        // None path == from_virtual_key with the static holdbacks only — the
        // dormant default that keeps ab_parity green.
        let key = base_key(
            r#"{"name":"S","routeplane_key":"rp_s","provider_keys":{},"tier":"standard"}"#,
        );
        let state = auth_state_with(key.clone(), BTreeSet::new(), None);
        let ctx = state.resolve_tenant_context(&key);
        let expect = TenantContext::from_virtual_key(&key, &BTreeSet::new());
        assert_eq!(ctx.capabilities, expect.capabilities);
        assert!(ctx.capabilities.active(Feature::SemanticCache));
    }

    #[test]
    fn eval_context_carries_tenant_tier_and_region() {
        let mut key = vk("rp_x");
        key.tenant_id = Some("t_acme".into());
        key.tier = Tier::Standard;
        key.default_residency = Some("IN".into());
        let ctx = eval_context_for(&key);
        assert_eq!(ctx.targeting_key.as_deref(), Some("t_acme"));
        assert_eq!(
            ctx.attributes.get("tier").map(String::as_str),
            Some("standard")
        );
        assert_eq!(ctx.attributes.get("region").map(String::as_str), Some("IN"));

        // No residency ⇒ the region attribute is omitted (not ""); tenant_id
        // falls back to the key name; Free tier targets "free".
        let key2 = vk("rp_y");
        let ctx2 = eval_context_for(&key2);
        assert_eq!(ctx2.targeting_key.as_deref(), Some("k"));
        assert_eq!(
            ctx2.attributes.get("tier").map(String::as_str),
            Some("free")
        );
        assert!(!ctx2.attributes.contains_key("region"));

        // DPDP (PRD-014 FR-4): the eval context carries ONLY non-personal
        // attributes — `tier` and an optional `region`. `targeting_key` is the
        // opaque `tenant_id`, not personal data. This exhaustive guard fails if
        // any new attribute key is ever threaded into the context, forcing a
        // reviewer to confirm it is non-personal before it can leave the process.
        for k in ctx.attributes.keys().chain(ctx2.attributes.keys()) {
            assert!(
                matches!(k.as_str(), "tier" | "region"),
                "unexpected eval-context attribute (possible DPDP/PII leak): {k}"
            );
        }
        assert_eq!(ctx.attributes.len(), 2); // tier + region
        assert_eq!(ctx2.attributes.len(), 1); // tier only
    }

    #[test]
    fn unknown_holdback_key_is_a_startup_error() {
        // Fail-closed: a typo'd holdback must refuse to start, never silently
        // ship the feature it was meant to hold back.
        let err = parse_holdbacks("semantic_cache,bogus_key").unwrap_err();
        assert!(matches!(err, AuthLoadError::Holdback { ref key } if key == "bogus_key"));
        // The error message names the variable and the offending key.
        let msg = err.to_string();
        assert!(msg.contains("RP_ROLLOUT_HOLDBACKS"));
        assert!(msg.contains("bogus_key"));
    }

    #[test]
    fn absent_or_blank_holdbacks_parse_to_empty_set() {
        // Inert by default: empty / whitespace / stray commas → empty set.
        assert!(parse_holdbacks("").unwrap().is_empty());
        assert!(parse_holdbacks("   ").unwrap().is_empty());
        assert!(parse_holdbacks(" , ,").unwrap().is_empty());
    }

    #[test]
    fn holdback_list_parses_with_trimming() {
        let set = parse_holdbacks(" advanced_guardrails , agentic_security ").unwrap();
        assert_eq!(
            set,
            BTreeSet::from([Feature::AdvancedGuardrails, Feature::AgenticSecurity])
        );
    }

    #[test]
    fn absent_holdbacks_var_is_inert() {
        // Only genuine absence maps to the empty set.
        let set = holdbacks_from_var(Err(std::env::VarError::NotPresent)).unwrap();
        assert!(set.is_empty());
    }

    #[test]
    fn present_holdbacks_var_delegates_to_the_parser() {
        let set = holdbacks_from_var(Ok("agentic_security".to_string())).unwrap();
        assert_eq!(set, BTreeSet::from([Feature::AgenticSecurity]));
    }

    // Exercises the pure function directly with the `VarError` that
    // `std::env::var` would produce for a non-UTF-8 value, rather than calling
    // `std::env::set_var` (unsafe in edition 2024 / racy across test threads).
    #[test]
    #[cfg(unix)]
    fn non_unicode_holdbacks_var_is_a_startup_error() {
        use std::os::unix::ffi::OsStrExt;

        // Fail-closed: a mangled RP_ROLLOUT_HOLDBACKS must refuse to start,
        // never silently disable every global holdback.
        let bad = std::ffi::OsStr::from_bytes(b"agentic_security\xff").to_os_string();
        let err = holdbacks_from_var(Err(std::env::VarError::NotUnicode(bad.clone()))).unwrap_err();
        assert!(matches!(err, AuthLoadError::HoldbackNotUnicode { ref value } if *value == bad));
        // The error message names the variable and the failure mode.
        let msg = err.to_string();
        assert!(msg.contains("RP_ROLLOUT_HOLDBACKS"));
        assert!(msg.contains("not valid UTF-8"));
    }
    // --- env/JSON registry loading (serverless deploys; keys.json is gitignored) ---

    #[test]
    fn load_from_json_parses_a_registry_string() {
        let state = AuthState::load_from_json(
            r#"{"keys":[{"name":"k","routeplane_key":"rp_env","provider_keys":{}}]}"#,
            "env:RP_KEYS_JSON",
        )
        .expect("valid registry loads");
        assert!(state.keys.contains_key("rp_env"));
    }

    #[test]
    fn load_from_json_rejects_empty_and_invalid_with_origin() {
        let err = AuthState::load_from_json(r#"{"keys":[]}"#, "env:RP_KEYS_JSON").unwrap_err();
        assert!(err.to_string().contains("env:RP_KEYS_JSON"));
        assert!(AuthState::load_from_json("{ not json", "env:RP_KEYS_JSON").is_err());
    }

    // --- ADR-041: Authorization: Bearer rp_… inbound fallback ------------------
    //
    // First, the pure credential-resolution helpers (C1 precedence + C2
    // disambiguation) unit-tested directly — no router, no async — so the
    // precedence/prefix rules are pinned independently of the wiring.

    #[test]
    fn c2_bearer_candidate_only_for_rp_prefixed_value() {
        // rp_ prefixed → candidate; case-insensitive scheme; token trimmed.
        assert_eq!(
            bearer_gateway_candidate(Some("Bearer rp_abc")),
            (Some("rp_abc"), false)
        );
        assert_eq!(
            bearer_gateway_candidate(Some("bearer rp_abc")),
            (Some("rp_abc"), false)
        );
        assert_eq!(
            bearer_gateway_candidate(Some("Bearer   rp_abc  ")),
            (Some("rp_abc"), false)
        );
        // Non-rp_ Bearer (provider key) → absent, but the demand-signal flag set.
        assert_eq!(
            bearer_gateway_candidate(Some("Bearer sk-live-xyz")),
            (None, true)
        );
        // Basic / bare token / empty → not a Bearer credential at all → absent,
        // NOT a non-rp_-Bearer signal.
        assert_eq!(
            bearer_gateway_candidate(Some("Basic dXNlcjpwYXNz")),
            (None, false)
        );
        assert_eq!(bearer_gateway_candidate(Some("rp_abc")), (None, false));
        assert_eq!(bearer_gateway_candidate(Some("")), (None, false));
        assert_eq!(bearer_gateway_candidate(None), (None, false));
        // Bearer with an empty token is a present-but-unusable Bearer → signal.
        assert_eq!(bearer_gateway_candidate(Some("Bearer ")), (None, true));
    }

    #[test]
    fn c1_native_header_always_wins_when_present_and_non_empty() {
        // Native present → native, even when a (disagreeing) Bearer is present.
        assert_eq!(
            resolve_gateway_credential(Some("rp_native"), Some("Bearer rp_other")),
            (Some(("rp_native", AuthSource::Native)), false)
        );
        // Native present, no Authorization → native.
        assert_eq!(
            resolve_gateway_credential(Some("rp_native"), None),
            (Some(("rp_native", AuthSource::Native)), false)
        );
        // Native trimmed before lookup (exact-match constant-time path).
        assert_eq!(
            resolve_gateway_credential(Some("  rp_native  "), None),
            (Some(("rp_native", AuthSource::Native)), false)
        );
    }

    #[test]
    fn c1_falls_through_to_bearer_only_when_native_absent_or_whitespace() {
        // No native → Bearer rp_ honoured.
        assert_eq!(
            resolve_gateway_credential(None, Some("Bearer rp_b")),
            (Some(("rp_b", AuthSource::Bearer)), false)
        );
        // Whitespace-only native is treated as ABSENT (C1) → Bearer honoured.
        assert_eq!(
            resolve_gateway_credential(Some("   "), Some("Bearer rp_b")),
            (Some(("rp_b", AuthSource::Bearer)), false)
        );
        // Neither usable → None; non-rp_ Bearer flag surfaces.
        assert_eq!(
            resolve_gateway_credential(None, Some("Bearer sk-live-x")),
            (None, true)
        );
        // Nothing at all → None, no signal.
        assert_eq!(resolve_gateway_credential(None, None), (None, false));
    }

    // --- ADR-041 C6.1–C6.6: end-to-end through auth_middleware -----------------
    //
    // Drive the REAL `auth_middleware` via a tiny router (the same `from_fn` +
    // `SharedAuthState` extension wiring as `main.rs`), with a leaf handler that
    // reflects the injected `VirtualKey`/`TenantContext` so we can assert both
    // the HTTP status (200/401) AND which tenant resolved.

    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use axum::routing::post;
    use axum::{middleware, Extension, Router};
    use tower::ServiceExt; // oneshot

    /// Leaf handler reached ONLY when auth succeeds. Echoes the resolved tenant
    /// id + the source-bearing key name so tests can assert the exact identity
    /// that resolved (not merely "a 200").
    async fn echo_identity(
        Extension(vk): Extension<VirtualKey>,
        Extension(ctx): Extension<TenantContext>,
    ) -> String {
        format!("name={};tenant={}", vk.name, ctx.tenant_id)
    }

    /// A registry with two distinct keys so "disagree" cases can prove WHICH key
    /// resolved. `rp_valid` → tenant `t_valid`; `rp_other` → tenant `t_other`.
    fn bearer_test_state() -> AuthState {
        let mk = |key: &str, tenant: &str| {
            let mut k = vk(key);
            k.name = format!("name-{tenant}");
            k.tenant_id = Some(tenant.to_string());
            k
        };
        let mut map = HashMap::new();
        map.insert("rp_valid".to_string(), mk("rp_valid", "t_valid"));
        map.insert("rp_other".to_string(), mk("rp_other", "t_other"));
        AuthState {
            keys: map,
            global_holdbacks: BTreeSet::new(),
            #[cfg(feature = "enterprise")]
            compiled_guardrails: HashMap::new(),
            unleash: None,
        }
    }

    fn bearer_test_app() -> Router {
        let auth_state = shared_auth_state(bearer_test_state());
        Router::new()
            .route("/v1/chat/completions", post(echo_identity))
            .layer(middleware::from_fn(auth_middleware))
            .layer(Extension(auth_state))
    }

    /// Build a POST request to the protected route with the given header pairs.
    fn req_with(headers: &[(&str, &str)]) -> Request<Body> {
        let mut b = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions");
        for (k, v) in headers {
            b = b.header(*k, *v);
        }
        b.body(Body::empty()).unwrap()
    }

    async fn body_string(resp: Response) -> String {
        let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
            .await
            .unwrap();
        String::from_utf8(bytes.to_vec()).unwrap()
    }

    // C6.1 — Bearer rp_valid, no native header → 200, resolves to the SAME
    // VirtualKey/TenantContext the native header would.
    #[tokio::test]
    async fn c6_1_bearer_only_authenticates_to_same_tenant() {
        let resp = bearer_test_app()
            .oneshot(req_with(&[("authorization", "Bearer rp_valid")]))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(body_string(resp).await, "name=name-t_valid;tenant=t_valid");

        // And it is byte-identical to the native-header resolution.
        let native = bearer_test_app()
            .oneshot(req_with(&[("x-routeplane-api-key", "rp_valid")]))
            .await
            .unwrap();
        assert_eq!(native.status(), StatusCode::OK);
        assert_eq!(
            body_string(native).await,
            "name=name-t_valid;tenant=t_valid"
        );
    }

    // C6.2 — both headers present and AGREE → 200.
    #[tokio::test]
    async fn c6_2_both_headers_agree_authenticates() {
        let resp = bearer_test_app()
            .oneshot(req_with(&[
                ("x-routeplane-api-key", "rp_valid"),
                ("authorization", "Bearer rp_valid"),
            ]))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(body_string(resp).await, "name=name-t_valid;tenant=t_valid");
    }

    // C6.3 — both present and DISAGREE → native wins (resolved tenant = native).
    #[tokio::test]
    async fn c6_3_disagreeing_headers_native_wins() {
        let resp = bearer_test_app()
            .oneshot(req_with(&[
                ("x-routeplane-api-key", "rp_valid"),
                ("authorization", "Bearer rp_other"),
            ]))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        // Native `rp_valid` → t_valid, NOT the Bearer's t_other.
        assert_eq!(body_string(resp).await, "name=name-t_valid;tenant=t_valid");
    }

    // C6.4 — non-rp_ Bearer (provider key), no native → 401, OpenAI-shaped body;
    // the value is never used as a provider key (it never reaches the handler,
    // and the constant-time lookup is never called with it — proven by the 401
    // and by the C2 unit test that returns it as `None`).
    #[tokio::test]
    async fn c6_4_non_rp_bearer_is_401_openai_shaped() {
        let resp = bearer_test_app()
            .oneshot(req_with(&[("authorization", "Bearer sk-live-secret")]))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        let v: serde_json::Value =
            serde_json::from_str(&body_string(resp).await).expect("OpenAI-shaped JSON body");
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert_eq!(v["error"]["code"], "invalid_api_key");
        // The provider-key value is never treated as a gateway key: a registry
        // lookup of the raw sk-… value must miss (it was never fed to lookup,
        // and even if it were it does not exist).
        assert!(bearer_test_state()
            .lookup_constant_time("sk-live-secret")
            .is_none());
    }

    // C6.5 — Basic / malformed / empty Bearer → 401.
    #[tokio::test]
    async fn c6_5_basic_malformed_empty_bearer_are_401() {
        for header in [
            ("authorization", "Basic dXNlcjpwYXNzd29yZA=="),
            ("authorization", "Bearer "),
            ("authorization", "Bearer"),
            ("authorization", "rp_valid"), // bare token, no scheme
            ("authorization", ""),
        ] {
            let resp = bearer_test_app()
                .oneshot(req_with(&[header]))
                .await
                .unwrap();
            assert_eq!(
                resp.status(),
                StatusCode::UNAUTHORIZED,
                "expected 401 for header {header:?}"
            );
        }
    }

    // C6.6 — whitespace-only native header + valid Bearer rp_… → empty native
    // treated as ABSENT → Bearer honoured.
    #[tokio::test]
    async fn c6_6_whitespace_native_falls_through_to_bearer() {
        let resp = bearer_test_app()
            .oneshot(req_with(&[
                ("x-routeplane-api-key", "   "),
                ("authorization", "Bearer rp_valid"),
            ]))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(body_string(resp).await, "name=name-t_valid;tenant=t_valid");
    }

    // --- ADR-050 D6: suspension enforced at key resolution, fail-closed --------

    /// A one-key app whose tenant sits in `state`.
    fn lifecycle_app(state: TenantState) -> Router {
        let mut k = vk("rp_x");
        k.tenant_id = Some("t_x".into());
        k.lifecycle_state = state;
        let mut map = HashMap::new();
        map.insert("rp_x".to_string(), k);
        let auth_state = shared_auth_state(AuthState {
            keys: map,
            global_holdbacks: BTreeSet::new(),
            #[cfg(feature = "enterprise")]
            compiled_guardrails: HashMap::new(),
            unleash: None,
        });
        Router::new()
            .route("/v1/chat/completions", post(echo_identity))
            .layer(middleware::from_fn(auth_middleware))
            .layer(Extension(auth_state))
    }

    // A non-Active tenant fails closed with the SAME 401 invalid_api_key envelope
    // as an unknown key — no state leak (C4). Proves "suspend → next call 401".
    #[tokio::test]
    async fn d6_non_active_tenant_fails_closed_401() {
        for state in [
            TenantState::Suspended,
            TenantState::Closed,
            TenantState::Pending,
        ] {
            let resp = lifecycle_app(state)
                .oneshot(req_with(&[("x-routeplane-api-key", "rp_x")]))
                .await
                .unwrap();
            assert_eq!(
                resp.status(),
                StatusCode::UNAUTHORIZED,
                "tenant in {state:?} must fail closed"
            );
            let v: serde_json::Value =
                serde_json::from_str(&body_string(resp).await).expect("OpenAI-shaped JSON body");
            assert_eq!(v["error"]["code"], "invalid_api_key");
        }
    }

    // An Active tenant authenticates as before.
    #[tokio::test]
    async fn d6_active_tenant_authenticates() {
        let resp = lifecycle_app(TenantState::Active)
            .oneshot(req_with(&[("x-routeplane-api-key", "rp_x")]))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(body_string(resp).await, "name=k;tenant=t_x");
    }

    // Backward-compat: a legacy key.json record with NO lifecycle_state field
    // deserialises to Active and serves traffic (no behaviour change).
    #[test]
    fn d6_legacy_key_without_lifecycle_state_defaults_active() {
        let k = base_key(r#"{"name":"k","routeplane_key":"rp_x","provider_keys":{}}"#);
        assert_eq!(k.lifecycle_state, TenantState::Active);
        assert!(k.lifecycle_state.serves_traffic());
    }

    // --- R0.2: auth-failure rate limiting -----------------------------------

    use axum::http::HeaderMap;

    fn headers(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut h = HeaderMap::new();
        for (k, v) in pairs {
            h.insert(
                axum::http::HeaderName::from_bytes(k.as_bytes()).unwrap(),
                HeaderValue::from_str(v).unwrap(),
            );
        }
        h
    }

    #[test]
    fn source_ip_prefers_trusted_header() {
        let h = headers(&[
            ("cf-connecting-ip", "203.0.113.7"),
            ("x-forwarded-for", "198.51.100.1, 10.0.0.1"),
        ]);
        assert_eq!(source_ip_key(&h, "cf-connecting-ip"), "203.0.113.7");
    }

    #[test]
    fn source_ip_falls_back_to_xff_first_hop() {
        let h = headers(&[("x-forwarded-for", "198.51.100.1, 10.0.0.1, 10.0.0.2")]);
        // Trusted header absent → XFF first hop (the original client).
        assert_eq!(source_ip_key(&h, "cf-connecting-ip"), "198.51.100.1");
    }

    #[test]
    fn source_ip_unknown_when_no_forwarding_headers() {
        let h = headers(&[("user-agent", "curl/8")]);
        // Fail-closed: no forwarding info → one shared "unknown" bucket, still
        // throttled as an aggregate rather than bypassing the limiter.
        assert_eq!(source_ip_key(&h, "cf-connecting-ip"), "unknown");
    }

    #[tokio::test]
    async fn throttled_response_is_429_with_retry_after_and_discriminator() {
        let resp = auth_throttled_response(42);
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(resp.headers().get("retry-after").unwrap(), "42");
        assert_eq!(
            resp.headers().get("x-routeplane-limit-type").unwrap(),
            "auth_failure"
        );
        let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["code"], "auth_failure_rate_limited");
        assert_eq!(v["error"]["type"], "invalid_request_error");
        // No source IP / key material leaked in the body.
        let msg = v["error"]["message"].as_str().unwrap();
        assert!(!msg.contains("rp_"));
    }

    /// Drive `auth_middleware` end-to-end through a minimal Router: a 200 handler
    /// guarded by `auth_middleware`, with the SharedAuthState and (optional)
    /// failure tracker injected as Extension layers exactly as `main.rs` wires
    /// them. Returns the final response (401/429/200).
    async fn run_auth(
        state: SharedAuthState,
        tracker: Option<SharedAuthFailureTracker>,
        api_key: Option<&str>,
        source_ip: &str,
    ) -> Response {
        use axum::routing::post;
        use axum::Router;
        use tower::ServiceExt;

        let mut router = Router::new()
            .route("/v1/chat/completions", post(|| async { "ok" }))
            .layer(axum::middleware::from_fn(auth_middleware))
            .layer(axum::Extension(state));
        if let Some(t) = tracker {
            router = router.layer(axum::Extension(t));
        }

        let mut builder = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("cf-connecting-ip", source_ip);
        if let Some(k) = api_key {
            builder = builder.header("x-routeplane-api-key", k);
        }
        let req = builder.body(Body::empty()).unwrap();
        router.oneshot(req).await.unwrap()
    }

    #[tokio::test]
    async fn middleware_throttles_after_threshold_failures() {
        let state = shared_auth_state(auth_with(&["rp_valid_key"]));
        let tracker: SharedAuthFailureTracker =
            Arc::new(AuthFailureTracker::new(AuthFailureConfig {
                threshold: 3,
                window_ms: 60_000,
                backoff_base_ms: 1_000,
                backoff_cap_ms: 300_000,
                slots: 64,
            }));
        let ip = "203.0.113.50";
        // 3 invalid-key attempts: each returns 401 and records a failure.
        for _ in 0..3 {
            let resp = run_auth(state.clone(), Some(tracker.clone()), Some("rp_wrong"), ip).await;
            assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        }
        // 4th attempt from the same source is short-circuited with 429 BEFORE the
        // lookup — even though we now present the VALID key, it is throttled.
        let resp = run_auth(
            state.clone(),
            Some(tracker.clone()),
            Some("rp_valid_key"),
            ip,
        )
        .await;
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        assert!(resp.headers().get("retry-after").is_some());
    }

    #[tokio::test]
    async fn middleware_does_not_throttle_a_different_source() {
        let state = shared_auth_state(auth_with(&["rp_valid_key"]));
        let tracker: SharedAuthFailureTracker =
            Arc::new(AuthFailureTracker::new(AuthFailureConfig {
                threshold: 2,
                window_ms: 60_000,
                backoff_base_ms: 1_000,
                backoff_cap_ms: 300_000,
                slots: 64,
            }));
        // Exhaust source A.
        for _ in 0..2 {
            let _ = run_auth(
                state.clone(),
                Some(tracker.clone()),
                Some("rp_wrong"),
                "1.1.1.1",
            )
            .await;
        }
        let a = run_auth(
            state.clone(),
            Some(tracker.clone()),
            Some("rp_wrong"),
            "1.1.1.1",
        )
        .await;
        assert_eq!(a.status(), StatusCode::TOO_MANY_REQUESTS);
        // Source B (different IP, valid key) is unaffected → authenticates (200).
        let b = run_auth(
            state.clone(),
            Some(tracker.clone()),
            Some("rp_valid_key"),
            "2.2.2.2",
        )
        .await;
        assert_eq!(b.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn disabled_tracker_is_byte_identical_no_throttle() {
        // No tracker extension ⇒ the gate is skipped entirely; repeated invalid
        // attempts keep returning 401 (never 429), byte-identical to today.
        let state = shared_auth_state(auth_with(&["rp_valid_key"]));
        for _ in 0..10 {
            let resp = run_auth(state.clone(), None, Some("rp_wrong"), "9.9.9.9").await;
            assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        }
        // A valid key still authenticates.
        let ok = run_auth(state.clone(), None, Some("rp_valid_key"), "9.9.9.9").await;
        assert_eq!(ok.status(), StatusCode::OK);
    }

    // --- R0.3: auth-failure security-event logging at the auth seam ------------

    // Enterprise-only: spins a REAL ledger writer (the CE build has no ledger
    // crate; its auth seam is the `record_security_global` no-op twin).
    #[cfg(feature = "enterprise")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn auth_failure_records_a_global_security_event_when_ledger_present() {
        use axum::routing::post;
        use axum::Router;
        use routeplane_ledger::{spawn_ledger, InMemoryLedgerStore, LedgerConfig, LedgerStore};
        use std::time::Duration;
        use tower::ServiceExt;

        let store = Arc::new(InMemoryLedgerStore::new());
        let cfg = LedgerConfig {
            wal_dir: std::env::temp_dir()
                .join(format!("rp_authsec_{}", std::process::id()))
                .join(uuid::Uuid::new_v4().simple().to_string()),
            drain_batch: 1,
            drain_interval: Duration::from_millis(10),
            ..Default::default()
        };
        let handle = spawn_ledger(cfg, store.clone(), None).unwrap();
        let security_ledger: SharedLedgerHandle = Arc::new(Some(handle));

        let state = shared_auth_state(auth_with(&["rp_valid_key"]));
        let router = Router::new()
            .route("/v1/chat/completions", post(|| async { "ok" }))
            .layer(axum::middleware::from_fn(auth_middleware))
            .layer(axum::Extension(state))
            .layer(axum::Extension(security_ledger));

        // One invalid-key attempt ⇒ 401 ⇒ a `_global`/`security` auth-failure event.
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("cf-connecting-ip", "203.0.113.99")
            .header("x-routeplane-api-key", "rp_wrong")
            .body(Body::empty())
            .unwrap();
        let resp = router.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

        let from = chrono::DateTime::parse_from_rfc3339("2000-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);
        let to = chrono::DateTime::parse_from_rfc3339("2100-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        loop {
            let got = store
                .read_entries("_global", "security", from, to)
                .await
                .unwrap();
            if let Some(e) = got.first() {
                assert_eq!(e.data_classes[0].as_str(), "auth_failure");
                // No key material / source IP anywhere in the entry.
                let json = serde_json::to_string(e).unwrap();
                assert!(!json.contains("rp_wrong"), "no key material in the ledger");
                assert!(!json.contains("203.0.113.99"), "no source IP in the ledger");
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "auth security event never landed"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
}
