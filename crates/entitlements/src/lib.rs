//! Routeplane entitlement seam — the data-plane half of [ADR-012] §3/§4.
//!
//! Customer variation (free/paid tiers, custom-customer features, dark-launch)
//! is expressed as a single resolved **`CapabilitySet`**, never a branch and
//! never a fork ([ADR-012] §3, [`branching-and-devex.md`] §6/§7). The set is
//! computed once at auth from the tenant's durable config:
//!
//! ```text
//! CapabilitySet = tier_baseline(tier) ∪ per_tenant_overrides − rollout_holdbacks
//! ```
//!
//! - `tier_baseline(tier)` — the commercial preset for a tier (the §7 table).
//! - `per_tenant_overrides` — additive per-tenant grants (e.g. a custom-customer
//!   feature crate shipped dark on trunk, flipped on for exactly one tenant).
//! - `rollout_holdbacks` — operational subtractions ("released for nobody yet"),
//!   so a half-finished feature can merge to trunk and ship inert.
//!
//! This realises the unified rule from [`branching-and-devex.md`] §5.2:
//! `active(feature) = entitled(tenant, feature) AND released(feature, …)`.
//! Entitlement is the tier-baseline-∪-overrides membership; the holdback set is
//! the `released` half subtracted out.
//!
//! ## Core vs optional
//! Core capabilities — auth, routing, basic guardrails — are **always-on** and
//! deliberately *not* representable as a [`Feature`]. They are never gated; only
//! the optional [ADR-004] features below are. An empty `CapabilitySet` therefore
//! means "core only", which is exactly the Free tier.
//!
//! ## Release-plane sequencing for baseline toggles (ADR-088 Bundle B / ADR-029)
//! `tier_baseline` edits are commercial-entitlement changes; they ship
//! **reversibly** by composing with the dynamic `released(...)` half that the
//! binary resolves at auth (`active = entitled ∧ released` — the Unleash flag
//! plane of ADR-029 lives in `crates/flags`; this crate stays pure/in-process).
//! The Unleash `EvalContext` carries a non-personal `tier` attribute, so
//! per-tier constraint targeting is available. The two moves:
//!
//! - **Grant** (e.g. `token_compression` → Free, ADR-088 Bundle B): land the
//!   baseline edit with the Unleash toggle constrained **OFF for
//!   `tier == free`** — behavior stays byte-identical until the deliberate
//!   constraint flip. Kill switch = flip it back, or the static
//!   `RP_ROLLOUT_HOLDBACKS=token_compression` env (parsed fail-closed at
//!   startup) for a whole-gateway hold.
//! - **Revocation** (planned: `routing_policy` out of Free): holdbacks can only
//!   subtract — the release plane cannot re-grant after a code removal — so
//!   revocations are a two-step. (1) KEEP the feature in the baseline and hold
//!   it back via an Unleash `routing_policy` constraint for `tier == free`
//!   contexts only (observable, instantly revertible); (2) once soaked, make
//!   the removal permanent in `tier_baseline` and retire the flag
//!   (flag-lifecycle hygiene per `crates/flag-governance`). Escape hatch either
//!   way: a per-tenant `capability_overrides` grant (the ∪ term) re-enables it
//!   for any grandfathered Free tenant. Step 1 is where we are today —
//!   `RoutingPolicy` is deliberately still in the Free baseline below.
//!
//! ## OpenFeature as the interface seam only
//! Per [ADR-012] §4 / [`branching-and-devex.md`] §6.3, call sites resolve
//! features through an **OpenFeature-shaped** [`FeatureProvider`] trait
//! (`resolve_bool(flag, default, ctx)`), backed by an **in-process**
//! [`RouteplaneEntitlementProvider`] over the same `CapabilitySet`. There is
//! **no external flag server** (no Unleash/flagd/GrowthBook): such a server is
//! always-on infrastructure (against the ~$1,000 credit, [ADR-001]) and an extra
//! network hop on a sub-10ms hot path (against [ADR-002]).
//!
//! We mirror OpenFeature's provider contract with a *small local trait* rather
//! than depending on the `open-feature` crate directly: as of this writing that
//! SDK is pre-1.0 and carries a "work-in-progress" badge (v0.3.0, March 2026),
//! so pulling it in would add breaking-change churn and hot-path allocation to
//! claim conformance we do not yet need. The trait below is deliberately the
//! same *shape* as OpenFeature's `FeatureProvider::resolve_bool_value` +
//! `EvaluationContext`, so adopting the real SDK later is a provider swap, not a
//! call-site rewrite — the same "neutral interface, frugal implementation"
//! discipline [ADR-007] applies to cloud and OTel.
//!
//! [ADR-001]: ../../../docs/adr/001-rust-data-plane-modular-architecture.md
//! [ADR-002]: ../../../docs/adr/002-modular-monolith-vs-microservices.md
//! [ADR-004]: ../../../docs/adr/004-feature-management-modules-and-runtime-flags.md
//! [ADR-007]: ../../../docs/adr/007-platform-engineering-charter-and-cloud-portability.md
//! [ADR-012]: ../../../docs/adr/012-trunk-based-development-and-entitlement-driven-delivery.md
//! [`branching-and-devex.md`]: ../../../docs/architecture/branching-and-devex.md

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::collections::BTreeSet;

/// An **optional** feature that may be gated per tenant ([ADR-004] examples).
///
/// Core capabilities (auth / routing / basic guardrails) are intentionally NOT
/// members here — they are always-on and never gated. Adding a variant here is
/// the act of making a new feature gateable; the `flag_key`/`from_flag_key`
/// mapping below is what wires it into the OpenFeature-shaped seam.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Feature {
    /// Semantic caching of completions.
    SemanticCache,
    /// Advanced guardrails (beyond the always-on basic PII masking).
    AdvancedGuardrails,
    /// Agentic-security controls (the MCP-gateway / agent-governance moat).
    AgenticSecurity,
    /// Versioned prompt registry.
    PromptRegistry,
    /// FinOps usage/cost export. NOTE: explicit rename — `snake_case` would
    /// derive `fin_ops_export`, but the §7 table / flag key is `finops_export`.
    #[serde(rename = "finops_export")]
    FinOpsExport,
    /// Config-driven routing policy (G2.2 / ADR-021). A CORE surface — included
    /// in EVERY tier baseline (incl. Free) — but modeled as a gateable feature so
    /// it carries a kill switch: a holdback removes it and the proxy ignores the
    /// `x-routeplane-config` routing section, falling back to the legacy
    /// `x-routeplane-provider`/`-strategy` header path. No branch, no fork.
    RoutingPolicy,
    /// Durable, hash-chained sovereign audit-ledger writes (PRD-001 FR-16,
    /// [ADR-019]). NOT in any tier baseline — ship-dark; granted per tenant via
    /// override only, so existing tiers are byte-identical with this merged.
    AuditLedger,
    /// Signed audit-artifact generation (PRD-002), gated at the control plane.
    /// Same ship-dark posture as `AuditLedger`.
    AuditArtifact,
    /// Model Catalog default-deny allowlist enforcement (PRD-008 FR-1/FR-3,
    /// [ADR-023]). NOT in any tier baseline — ship-dark; granted per tenant via
    /// override only, so existing tiers are byte-identical with this merged. When
    /// active, a request whose `model` is not in the key's provisioned allowlist
    /// is rejected 403 `model_not_provisioned` before any provider call or spend
    /// (fail-closed: an empty allowlist denies every model). When inactive (the
    /// default), the catalog check is skipped entirely — the hot path is
    /// byte-identical, so the `ab_parity`/`golden_snapshot` guards stay exact.
    ModelCatalog,
    /// RTK `tool_result` token-compression ([ADR-085]). In the **Free** tier
    /// baseline (the ADR-088 Bundle B / CE grant — see the module-doc
    /// "Release-plane sequencing" section: the grant ships behind an Unleash
    /// `token_compression` toggle constrained OFF for `tier == free` until the
    /// deliberate flip, so entitlement lands byte-identical). Paid tiers reach it
    /// via a per-tenant override (the ∪ term). When active, `tool`-role message
    /// content is deterministically compressed AFTER residency-classify and
    /// PII-masking, before forwarding (20–40% input-token savings on tool-heavy
    /// agentic workloads). When inactive the compression step is skipped entirely
    /// — the hot path is byte-identical, so the `ab_parity`/`golden_snapshot`
    /// guards stay exact.
    TokenCompression,
    /// Durable per-request telemetry — the rung-1 Observability v2 store + bus
    /// writer (PRD-009 FR-20, [ADR-024]). NOT in any tier baseline — ship-dark;
    /// granted per tenant via override only, so existing tiers are byte-identical
    /// with this merged. When active, the data plane ALSO emits each resolved
    /// request to a durable telemetry bus (off the hot path, drop-on-overflow);
    /// when inactive (the default) no telemetry writer is attached and behaviour
    /// is byte-identical — the free tier stays the in-memory VecDeque at $0
    /// (PRD-009 FR-8, binding).
    TelemetryDurable,
}

impl Feature {
    /// The stable, snake_case flag key used at OpenFeature call sites and in the
    /// §7 table (e.g. `semantic_cache`). This is the wire/identifier form; the
    /// enum is the typed form. Keep this in sync with `from_flag_key`.
    pub const fn flag_key(self) -> &'static str {
        match self {
            Feature::SemanticCache => "semantic_cache",
            Feature::AdvancedGuardrails => "advanced_guardrails",
            Feature::AgenticSecurity => "agentic_security",
            Feature::PromptRegistry => "prompt_registry",
            Feature::FinOpsExport => "finops_export",
            Feature::RoutingPolicy => "routing_policy",
            Feature::AuditLedger => "audit_ledger",
            Feature::AuditArtifact => "audit_artifact",
            Feature::ModelCatalog => "model_catalog",
            Feature::TokenCompression => "token_compression",
            Feature::TelemetryDurable => "telemetry_durable",
        }
    }

    /// Parse a flag key back into a typed [`Feature`]. Returns `None` for an
    /// unknown key — the provider treats an unknown flag as "not a Routeplane
    /// optional feature" and falls back to the caller-supplied default, matching
    /// OpenFeature's `FLAG_NOT_FOUND → default` semantics.
    pub fn from_flag_key(key: &str) -> Option<Feature> {
        match key {
            "semantic_cache" => Some(Feature::SemanticCache),
            "advanced_guardrails" => Some(Feature::AdvancedGuardrails),
            "agentic_security" => Some(Feature::AgenticSecurity),
            "prompt_registry" => Some(Feature::PromptRegistry),
            "finops_export" => Some(Feature::FinOpsExport),
            "routing_policy" => Some(Feature::RoutingPolicy),
            "audit_ledger" => Some(Feature::AuditLedger),
            "audit_artifact" => Some(Feature::AuditArtifact),
            "model_catalog" => Some(Feature::ModelCatalog),
            "token_compression" => Some(Feature::TokenCompression),
            "telemetry_durable" => Some(Feature::TelemetryDurable),
            _ => None,
        }
    }

    /// Every gateable [`Feature`], in declaration order — the stable enumerator
    /// the auth layer iterates to ask the flag plane "is each released for this
    /// context?" (ADR-029 G3, PRD-013 FR-1). It lets the Unleash composition at
    /// auth build the `!released` holdback set over the full feature space while
    /// the lock-free `CapabilitySet::active` hot path stays untouched.
    ///
    /// Adding a variant to [`Feature`] MUST add it here too — the same
    /// keep-in-sync discipline as `flag_key`/`from_flag_key` (the
    /// `all_is_complete_and_unique` test guards uniqueness + the flag-key round
    /// trip). Additive and pure; no behaviour change.
    pub const ALL: [Feature; 11] = [
        Feature::SemanticCache,
        Feature::AdvancedGuardrails,
        Feature::AgenticSecurity,
        Feature::PromptRegistry,
        Feature::FinOpsExport,
        Feature::RoutingPolicy,
        Feature::AuditLedger,
        Feature::AuditArtifact,
        Feature::ModelCatalog,
        Feature::TokenCompression,
        Feature::TelemetryDurable,
    ];

    /// Iterate every gateable [`Feature`] — a convenience over [`Feature::ALL`]
    /// for call sites that want an iterator (e.g. the auth-time holdback fold).
    pub fn all() -> impl Iterator<Item = Feature> {
        Self::ALL.into_iter()
    }
}

/// A commercial plan. `#[default]` is [`Tier::Free`] so a `keys.json` record
/// that omits `tier` (every record today) resolves to core-only — backward
/// compatible by construction.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    /// T0 — sponsored/free; core only, no optional features.
    #[default]
    Free,
    /// T1 — paid standard plan.
    Standard,
    /// T2 — paid business plan: Standard plus FinOps cost/usage export. The
    /// agentic-security moat stays Enterprise-exclusive (it is NOT in this
    /// baseline). Declared between `Standard` and `Enterprise` so the derived
    /// `Ord` reflects the commercial ladder (`Standard < Business < Enterprise`).
    Business,
    /// T3 — enterprise / custom plan (the per-customer override surface).
    Enterprise,
}

/// Tenant lifecycle state ([ADR-050] D1) — shared CP/DP vocabulary, alongside
/// [`Tier`]. The control plane owns transitions (`store` crate); the data plane
/// reads it at key resolution to fail closed on a non-`Active` tenant ([ADR-050]
/// D6). Lives here (not in `store`) so the hot-path binary reads it WITHOUT
/// depending on the persistence crate — like `Tier`, it is pure vocabulary with
/// no I/O.
///
/// State machine: `PENDING → ACTIVE → SUSPENDED → CLOSED` with `SUSPENDED →
/// ACTIVE` reinstate; `CLOSED` is terminal. The `snake_case` serde wire form
/// (`"pending"`/`"active"`/`"suspended"`/`"closed"`) is the seam contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TenantState {
    /// Self-serve signup recorded, not yet activated. No traffic served.
    Pending,
    /// Fully provisioned; the only state the data plane serves traffic for.
    /// Default so a legacy key/tenant record without the field is `Active`
    /// (backward-compatible by construction — no behaviour change).
    #[default]
    Active,
    /// Temporarily blocked (non-payment, abuse, operator hold). Reversible to
    /// `Active`; the data plane fails closed while suspended.
    Suspended,
    /// Terminal. No traffic, no transitions out.
    Closed,
}

impl TenantState {
    /// Whether a transition to `next` is permitted by the lifecycle state machine
    /// ([ADR-050] D1). `Closed` is terminal; same-state is a no-op (rejected so
    /// callers don't mistake an idempotent retry for a real transition).
    pub fn can_transition_to(self, next: TenantState) -> bool {
        use TenantState::*;
        matches!(
            (self, next),
            (Pending, Active)
                | (Pending, Closed)
                | (Active, Suspended)
                | (Active, Closed)
                | (Suspended, Active)
                | (Suspended, Closed)
        )
    }

    /// Whether the data plane should serve traffic for a tenant in this state.
    /// Only `Active` serves; everything else fails closed ([ADR-050] D6).
    pub fn serves_traffic(self) -> bool {
        matches!(self, TenantState::Active)
    }
}

/// The tier → optional-feature preset, mirroring the
/// [`branching-and-devex.md`] §7 table **exactly**:
///
/// | feature              | Free | Standard | Business | Enterprise |
/// |----------------------|:----:|:--------:|:--------:|:----------:|
/// | `semantic_cache`     |  ✗   |    ✓     |    ✓     |     ✓      |
/// | `advanced_guardrails`|  ✗   |    ✓     |    ✓     |     ✓      |
/// | `agentic_security`   |  ✗   |    ✗     |    ✗     |     ✓      |
/// | `prompt_registry`    |  ✗   |    ✓     |    ✓     |     ✓      |
/// | `finops_export`      |  ✗   |    ✗     |    ✓     |     ✓      |
/// | `token_compression`  |  ✓   |    ✗     |    ✗     |     ✗      |
///
/// Free = core plus `token_compression` (the ADR-088 Bundle B / CE grant —
/// deterministic RTK savings are part of the community-edition value; it lands
/// release-plane gated per the module-doc sequencing section, and paid tiers
/// reach it via a per-tenant override until their own baseline decision).
/// Standard adds semantic-cache, advanced-guardrails and prompt-registry.
/// Business is Standard **plus finops-export** (a mid-market cost/usage need).
/// Enterprise additionally adds agentic-security — the moat is
/// Enterprise-exclusive in the baseline. The `agentic_security` "add-on" column
/// for Standard/Business is *not* in their baseline — it is reached only via a
/// per-tenant override (the ∪ term), exactly as the table footnote describes.
pub fn tier_baseline(tier: Tier) -> BTreeSet<Feature> {
    match tier {
        // RoutingPolicy is a CORE surface in EVERY tier (incl. Free); it is a
        // Feature only so a holdback can kill-switch it (F13 / ADR-021 A1).
        // It stays in the Free baseline DELIBERATELY — step 1 of the ADR-088
        // revocation two-step (module doc): the removal happens first as an
        // Unleash per-tier holdback, only later here. TokenCompression is the
        // ADR-088 Bundle B grant, shipped behind the release plane (Unleash
        // `token_compression` constrained OFF for tier == free until the flip).
        Tier::Free => BTreeSet::from([Feature::RoutingPolicy, Feature::TokenCompression]),
        Tier::Standard => BTreeSet::from([
            Feature::RoutingPolicy,
            Feature::SemanticCache,
            Feature::AdvancedGuardrails,
            Feature::PromptRegistry,
        ]),
        // Business = Standard ∪ {finops_export}. Agentic-security stays
        // Enterprise-only (reachable on Business only via a per-tenant override).
        Tier::Business => BTreeSet::from([
            Feature::RoutingPolicy,
            Feature::SemanticCache,
            Feature::AdvancedGuardrails,
            Feature::PromptRegistry,
            Feature::FinOpsExport,
        ]),
        Tier::Enterprise => BTreeSet::from([
            Feature::RoutingPolicy,
            Feature::SemanticCache,
            Feature::AdvancedGuardrails,
            Feature::PromptRegistry,
            Feature::AgenticSecurity,
            Feature::FinOpsExport,
        ]),
    }
}

/// The resolved set of optional features active for a tenant on a request.
///
/// Constructed once at auth and carried (cheaply cloned) in request extensions
/// on the hot path, so it is intentionally allocation-light: a single
/// [`BTreeSet<Feature>`] of `Copy` C-like enums (no heap strings inside), and
/// the only operation on the hot path — [`CapabilitySet::active`] — is a
/// `log(n)`, lock-free set membership test. No mutex, no atomics needed (the set
/// is immutable once resolved).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CapabilitySet {
    active: BTreeSet<Feature>,
}

impl CapabilitySet {
    /// Resolve `CapabilitySet = tier_baseline(tier) ∪ overrides − holdbacks`
    /// ([ADR-012] §3, [`branching-and-devex.md`] §6.1).
    ///
    /// `overrides` are the tenant's durable additive grants (a custom-customer
    /// feature, or a Standard tenant's add-on). `holdbacks` are the operational
    /// `released = false` subtractions — a feature in this set is removed **even
    /// if the tier or an override would grant it**, which is precisely how a
    /// dark-launched feature stays inert for entitled tenants until canary.
    pub fn resolve(
        tier: Tier,
        overrides: &BTreeSet<Feature>,
        holdbacks: &BTreeSet<Feature>,
    ) -> Self {
        let mut active = tier_baseline(tier);
        active.extend(overrides.iter().copied()); // ∪ overrides
        for held in holdbacks {
            active.remove(held); // − holdbacks (after the ∪, so it wins)
        }
        Self { active }
    }

    /// Construct directly from an already-resolved set (test/control-plane use).
    pub fn from_set(active: BTreeSet<Feature>) -> Self {
        Self { active }
    }

    /// Is `feature` active for this request? The hot-path predicate every
    /// optional Tower layer / `proxy.rs` branch gates on. Lock-free, allocation-
    /// free, `O(log n)`.
    ///
    /// (A future evaluation-context argument — region/tenant for context-aware
    /// rollout — would thread through here; today membership is the whole
    /// answer because the holdback subtraction already happened at `resolve`.)
    pub fn active(&self, feature: Feature) -> bool {
        self.active.contains(&feature)
    }

    /// Iterate the active features (observability / debugging; not hot path).
    pub fn iter(&self) -> impl Iterator<Item = Feature> + '_ {
        self.active.iter().copied()
    }

    /// Number of active optional features.
    pub fn len(&self) -> usize {
        self.active.len()
    }

    /// True when no optional features are active (i.e. core-only / Free).
    pub fn is_empty(&self) -> bool {
        self.active.is_empty()
    }
}

/// An OpenFeature-shaped evaluation context: the targeting key plus arbitrary
/// string attributes (`tenant_id`, `region`, …). Mirrors OpenFeature's
/// `EvaluationContext` so call sites read identically to the SDK form.
///
/// Kept to owned `String`s for ergonomics off the hot path; the hot-path gate
/// in `proxy.rs` calls [`CapabilitySet::active`] directly and does not allocate
/// an `EvalContext` per request.
#[derive(Debug, Clone, Default)]
pub struct EvalContext {
    /// The targeting key (OpenFeature's `targeting_key`) — typically `tenant_id`.
    pub targeting_key: Option<String>,
    /// Free-form custom attributes (region, plan, …).
    pub attributes: BTreeMap<String, String>,
}

impl EvalContext {
    /// A context targeting a tenant.
    pub fn for_tenant(tenant_id: impl Into<String>) -> Self {
        Self {
            targeting_key: Some(tenant_id.into()),
            attributes: BTreeMap::new(),
        }
    }

    /// Builder-style attribute setter.
    pub fn with_attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.insert(key.into(), value.into());
        self
    }
}

/// OpenFeature-shaped provider contract — the interface seam ([ADR-012] §4).
///
/// This mirrors OpenFeature's `FeatureProvider::resolve_bool_value(flag, default,
/// ctx)`. Call sites depend on this trait, not on a concrete provider, so the
/// in-process implementation can be swapped for an external-server-backed one
/// later without touching a single call site. Contract for unknown flags:
/// return the supplied `default` (OpenFeature's `FLAG_NOT_FOUND → default`).
pub trait FeatureProvider: Send + Sync {
    /// Resolve a boolean flag. Returns `default` for an unknown flag key.
    fn resolve_bool(&self, flag: &str, default: bool, ctx: &EvalContext) -> bool;
}

/// In-process OpenFeature provider backed by a resolved [`CapabilitySet`]
/// ([ADR-012] §4, [`branching-and-devex.md`] §6.3). No network, no flag server.
///
/// It maps an OpenFeature flag-key string → typed [`Feature`] → the set's
/// [`CapabilitySet::active`] membership. An unknown flag key (one that is not a
/// Routeplane optional feature) returns the caller's `default`.
#[derive(Debug, Clone)]
pub struct RouteplaneEntitlementProvider {
    capabilities: CapabilitySet,
}

impl RouteplaneEntitlementProvider {
    pub fn new(capabilities: CapabilitySet) -> Self {
        Self { capabilities }
    }

    /// Borrow the underlying resolved set (for direct, allocation-free gating
    /// on the hot path when a call site already has typed `Feature`).
    pub fn capabilities(&self) -> &CapabilitySet {
        &self.capabilities
    }
}

impl FeatureProvider for RouteplaneEntitlementProvider {
    fn resolve_bool(&self, flag: &str, default: bool, _ctx: &EvalContext) -> bool {
        match Feature::from_flag_key(flag) {
            Some(feature) => self.capabilities.active(feature),
            // Unknown flag → not a Routeplane optional feature → default.
            None => default,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(features: impl IntoIterator<Item = Feature>) -> BTreeSet<Feature> {
        features.into_iter().collect()
    }

    // --- tier → baseline, mirroring the §7 table -------------------------------

    #[test]
    fn free_baseline_is_routing_policy_and_token_compression() {
        // Free baseline is exactly {RoutingPolicy, TokenCompression} — no other
        // optional features. RoutingPolicy stays per ADR-088 §2c step 1 (the
        // revocation happens as an Unleash per-tier holdback first, never as a
        // code removal without soak); TokenCompression is the ADR-088 Bundle B
        // CE grant, release-plane gated at rollout.
        assert_eq!(
            tier_baseline(Tier::Free),
            set([Feature::RoutingPolicy, Feature::TokenCompression])
        );
    }

    #[test]
    fn standard_baseline_matches_section_7_table() {
        let b = tier_baseline(Tier::Standard);
        assert_eq!(
            b,
            set([
                Feature::RoutingPolicy,
                Feature::SemanticCache,
                Feature::AdvancedGuardrails,
                Feature::PromptRegistry,
            ])
        );
        // Standard add-ons are NOT in the baseline (reached only via override).
        assert!(!b.contains(&Feature::AgenticSecurity));
        assert!(!b.contains(&Feature::FinOpsExport));
    }

    #[test]
    fn enterprise_baseline_matches_section_7_table() {
        let b = tier_baseline(Tier::Enterprise);
        assert_eq!(
            b,
            set([
                Feature::RoutingPolicy,
                Feature::SemanticCache,
                Feature::AdvancedGuardrails,
                Feature::PromptRegistry,
                Feature::AgenticSecurity,
                Feature::FinOpsExport,
            ])
        );
    }

    #[test]
    fn business_baseline_is_standard_plus_finops_export() {
        // Business = Standard ∪ {finops_export}; the agentic-security moat is
        // NOT in the baseline (Enterprise-exclusive, reachable on Business only
        // via a per-tenant override).
        let b = tier_baseline(Tier::Business);
        assert_eq!(
            b,
            set([
                Feature::RoutingPolicy,
                Feature::SemanticCache,
                Feature::AdvancedGuardrails,
                Feature::PromptRegistry,
                Feature::FinOpsExport,
            ])
        );
        assert!(!b.contains(&Feature::AgenticSecurity));
        // It is exactly the Standard baseline plus finops_export.
        let mut standard_plus = tier_baseline(Tier::Standard);
        standard_plus.insert(Feature::FinOpsExport);
        assert_eq!(b, standard_plus);
    }

    #[test]
    fn tier_ladder_orders_standard_business_enterprise() {
        // The derived Ord reflects the commercial ladder (declaration order).
        assert!(Tier::Free < Tier::Standard);
        assert!(Tier::Standard < Tier::Business);
        assert!(Tier::Business < Tier::Enterprise);
        // Wire form round-trips for the new variant.
        assert_eq!(
            serde_json::to_string(&Tier::Business).unwrap(),
            "\"business\""
        );
        assert_eq!(
            serde_json::from_str::<Tier>("\"business\"").unwrap(),
            Tier::Business
        );
    }

    #[test]
    fn tier_defaults_to_free() {
        assert_eq!(Tier::default(), Tier::Free);
    }

    // --- CapabilitySet::resolve (∪ overrides − holdbacks) ----------------------

    #[test]
    fn resolve_free_with_no_overrides_is_baseline_pair_only() {
        // Free resolves to exactly its baseline pair: {RoutingPolicy (F13 core
        // surface), TokenCompression (ADR-088 Bundle B)} — nothing else.
        let cs = CapabilitySet::resolve(Tier::Free, &BTreeSet::new(), &BTreeSet::new());
        assert!(cs.active(Feature::RoutingPolicy));
        assert!(cs.active(Feature::TokenCompression));
        assert_eq!(cs.len(), 2);
        assert!(!cs.active(Feature::SemanticCache));
    }

    #[test]
    fn override_unions_a_feature_onto_baseline() {
        // Free tenant granted a single feature via per-tenant override (custom
        // customer / add-on): ∪ adds it on top of the baseline pair.
        let overrides = set([Feature::AgenticSecurity]);
        let cs = CapabilitySet::resolve(Tier::Free, &overrides, &BTreeSet::new());
        assert!(cs.active(Feature::AgenticSecurity));
        // The Free baseline pair (RoutingPolicy + TokenCompression) remains, so
        // the override adds a 3rd.
        assert!(cs.active(Feature::RoutingPolicy));
        assert!(cs.active(Feature::TokenCompression));
        assert_eq!(cs.len(), 3);
    }

    #[test]
    fn standard_plus_addon_override() {
        // Standard + finops_export add-on (the §7 footnote case).
        let overrides = set([Feature::FinOpsExport]);
        let cs = CapabilitySet::resolve(Tier::Standard, &overrides, &BTreeSet::new());
        assert!(cs.active(Feature::SemanticCache)); // from baseline
        assert!(cs.active(Feature::FinOpsExport)); // from override
        assert!(!cs.active(Feature::AgenticSecurity)); // neither
    }

    #[test]
    fn holdback_subtracts_a_feature_the_tier_grants() {
        // Dark-launch: Enterprise grants advanced_guardrails, but a holdback
        // removes it even for an entitled tenant — active = entitled ∧ released
        // = true ∧ false = false.
        let holdbacks = set([Feature::AdvancedGuardrails]);
        let cs = CapabilitySet::resolve(Tier::Enterprise, &BTreeSet::new(), &holdbacks);
        assert!(!cs.active(Feature::AdvancedGuardrails));
        // Other Enterprise features remain active.
        assert!(cs.active(Feature::SemanticCache));
        assert!(cs.active(Feature::AgenticSecurity));
    }

    #[test]
    fn holdback_wins_over_override() {
        // A feature both granted by override AND held back is removed: the
        // subtraction runs after the union, so released=false wins.
        let overrides = set([Feature::SemanticCache]);
        let holdbacks = set([Feature::SemanticCache]);
        let cs = CapabilitySet::resolve(Tier::Free, &overrides, &holdbacks);
        assert!(!cs.active(Feature::SemanticCache));
        // The Free baseline pair remains (neither was held back).
        assert!(cs.active(Feature::RoutingPolicy));
        assert!(cs.active(Feature::TokenCompression));
        assert_eq!(cs.len(), 2);
    }

    // --- flag-key round trip ---------------------------------------------------

    #[test]
    fn serde_repr_matches_flag_key_for_every_feature() {
        // The JSON representation in keys.json and the OpenFeature flag key MUST
        // be identical, or a `capability_overrides` entry won't map to the same
        // feature the gate resolves. (Regression guard: snake_case derives
        // `fin_ops_export` for FinOpsExport — the explicit rename fixes it.)
        for f in [
            Feature::SemanticCache,
            Feature::AdvancedGuardrails,
            Feature::AgenticSecurity,
            Feature::PromptRegistry,
            Feature::FinOpsExport,
        ] {
            let json = serde_json::to_string(&f).expect("serialize");
            assert_eq!(json, format!("\"{}\"", f.flag_key()));
        }
    }

    #[test]
    fn flag_key_round_trips_for_every_feature() {
        for f in [
            Feature::SemanticCache,
            Feature::AdvancedGuardrails,
            Feature::AgenticSecurity,
            Feature::PromptRegistry,
            Feature::FinOpsExport,
        ] {
            assert_eq!(Feature::from_flag_key(f.flag_key()), Some(f));
        }
    }

    // --- FR-1: Feature::ALL / all() gateable-feature enumerator ----------------

    #[test]
    fn all_is_complete_and_unique() {
        // No duplicates: ALL lists each gateable feature exactly once.
        let unique: BTreeSet<Feature> = Feature::ALL.iter().copied().collect();
        assert_eq!(
            unique.len(),
            Feature::ALL.len(),
            "Feature::ALL contains a duplicate"
        );
        // Every member round-trips through its flag key — catches a stale ALL
        // entry whose variant was renamed or dropped from flag_key/from_flag_key.
        for f in Feature::all() {
            assert_eq!(Feature::from_flag_key(f.flag_key()), Some(f));
        }
        // ALL covers every variant the §7 table + the ship-dark set defines. If a
        // new Feature variant is added, flag_key()'s exhaustive match forces an
        // edit there; this list is the paired reminder to extend ALL.
        for expected in [
            Feature::SemanticCache,
            Feature::AdvancedGuardrails,
            Feature::AgenticSecurity,
            Feature::PromptRegistry,
            Feature::FinOpsExport,
            Feature::RoutingPolicy,
            Feature::AuditLedger,
            Feature::AuditArtifact,
            Feature::ModelCatalog,
        ] {
            assert!(
                unique.contains(&expected),
                "{expected:?} missing from Feature::ALL"
            );
        }
    }

    #[test]
    fn all_iterates_in_array_order() {
        assert_eq!(Feature::all().collect::<Vec<_>>(), Feature::ALL.to_vec());
    }

    // --- the OpenFeature-shaped provider --------------------------------------

    #[test]
    fn provider_resolves_known_flag_true_when_active() {
        let cs = CapabilitySet::resolve(Tier::Standard, &BTreeSet::new(), &BTreeSet::new());
        let provider = RouteplaneEntitlementProvider::new(cs);
        let ctx = EvalContext::for_tenant("t_acme");
        // semantic_cache is in the Standard baseline → true (default false ignored).
        assert!(provider.resolve_bool("semantic_cache", false, &ctx));
    }

    #[test]
    fn provider_resolves_known_flag_false_when_inactive() {
        // Free tenant: semantic_cache not entitled → false, even if default true.
        let cs = CapabilitySet::resolve(Tier::Free, &BTreeSet::new(), &BTreeSet::new());
        let provider = RouteplaneEntitlementProvider::new(cs);
        let ctx = EvalContext::for_tenant("t_free");
        assert!(!provider.resolve_bool("semantic_cache", true, &ctx));
    }

    #[test]
    fn provider_returns_default_for_unknown_flag() {
        let cs = CapabilitySet::resolve(Tier::Enterprise, &BTreeSet::new(), &BTreeSet::new());
        let provider = RouteplaneEntitlementProvider::new(cs);
        let ctx = EvalContext::default();
        // Unknown flag → caller's default verbatim, both polarities.
        assert!(provider.resolve_bool("nonexistent_flag", true, &ctx));
        assert!(!provider.resolve_bool("nonexistent_flag", false, &ctx));
    }

    #[test]
    fn eval_context_builder_sets_attributes() {
        let ctx = EvalContext::for_tenant("t_acme").with_attribute("region", "IN");
        assert_eq!(ctx.targeting_key.as_deref(), Some("t_acme"));
        assert_eq!(ctx.attributes.get("region").map(String::as_str), Some("IN"));
    }
    #[test]
    fn audit_features_ship_dark_in_no_tier_baseline() {
        // PRD-001 FR-16 / ADR-012: the sovereign-audit capabilities are NOT part
        // of any tier preset — merging them changes nothing for existing tiers.
        for tier in [Tier::Free, Tier::Standard, Tier::Business, Tier::Enterprise] {
            let b = tier_baseline(tier);
            assert!(!b.contains(&Feature::AuditLedger));
            assert!(!b.contains(&Feature::AuditArtifact));
        }
        // They are reachable ONLY via the per-tenant override (∪) term…
        let overrides = BTreeSet::from([Feature::AuditLedger, Feature::AuditArtifact]);
        let cs = CapabilitySet::resolve(Tier::Free, &overrides, &BTreeSet::new());
        assert!(cs.active(Feature::AuditLedger));
        assert!(cs.active(Feature::AuditArtifact));
        // …and a holdback still wins (dark-launch stays operable).
        let held = CapabilitySet::resolve(
            Tier::Free,
            &BTreeSet::from([Feature::AuditLedger]),
            &BTreeSet::from([Feature::AuditLedger]),
        );
        assert!(!held.active(Feature::AuditLedger));
    }

    #[test]
    fn audit_flag_keys_round_trip_and_match_serde() {
        for f in [Feature::AuditLedger, Feature::AuditArtifact] {
            assert_eq!(Feature::from_flag_key(f.flag_key()), Some(f));
            let json = serde_json::to_string(&f).unwrap();
            assert_eq!(json, format!("\"{}\"", f.flag_key()));
        }
    }

    // --- PRD-008 FR-1/FR-3: Model Catalog ships dark in no tier baseline -------

    #[test]
    fn model_catalog_ships_dark_in_no_tier_baseline() {
        // PRD-008 / ADR-012: the model-catalog default-deny capability is NOT part
        // of any tier preset — merging it changes nothing for existing tiers, so
        // the hot path stays byte-identical (the ab_parity/golden guards stay
        // green). It is reachable ONLY via the per-tenant override (∪) term.
        for tier in [Tier::Free, Tier::Standard, Tier::Business, Tier::Enterprise] {
            assert!(!tier_baseline(tier).contains(&Feature::ModelCatalog));
        }
        let overrides = BTreeSet::from([Feature::ModelCatalog]);
        let cs = CapabilitySet::resolve(Tier::Free, &overrides, &BTreeSet::new());
        assert!(cs.active(Feature::ModelCatalog));
        // …and a holdback still wins (dark-launch / kill switch stays operable).
        let held = CapabilitySet::resolve(
            Tier::Free,
            &BTreeSet::from([Feature::ModelCatalog]),
            &BTreeSet::from([Feature::ModelCatalog]),
        );
        assert!(!held.active(Feature::ModelCatalog));
    }

    #[test]
    fn model_catalog_flag_key_round_trips_and_matches_serde() {
        assert_eq!(Feature::ModelCatalog.flag_key(), "model_catalog");
        assert_eq!(
            Feature::from_flag_key("model_catalog"),
            Some(Feature::ModelCatalog)
        );
        // The natural snake_case derive is correct (no explicit rename needed).
        assert_eq!(
            serde_json::to_string(&Feature::ModelCatalog).unwrap(),
            "\"model_catalog\""
        );
    }

    // --- F13: RoutingPolicy = core surface in every tier + holdback kill switch -

    #[test]
    fn routing_policy_is_in_every_tier_baseline_with_kill_switch() {
        // Core surface: present in EVERY tier baseline (incl. Free). The Free
        // membership is ALSO ADR-088 §2c step 1 of the planned revocation: it
        // MUST stay in the baseline until the Unleash per-tier holdback
        // (tier == free) has soaked — this assertion is the guard against a
        // premature code removal.
        for tier in [Tier::Free, Tier::Standard, Tier::Business, Tier::Enterprise] {
            assert!(tier_baseline(tier).contains(&Feature::RoutingPolicy));
        }
        // A default Free tenant has it active…
        let on = CapabilitySet::resolve(Tier::Free, &BTreeSet::new(), &BTreeSet::new());
        assert!(on.active(Feature::RoutingPolicy));
        // …and a holdback (the kill switch) removes it even from the baseline.
        let killed =
            CapabilitySet::resolve(Tier::Free, &BTreeSet::new(), &set([Feature::RoutingPolicy]));
        assert!(!killed.active(Feature::RoutingPolicy));
        // Flag-key round trip for the new variant.
        assert_eq!(Feature::RoutingPolicy.flag_key(), "routing_policy");
        assert_eq!(
            Feature::from_flag_key("routing_policy"),
            Some(Feature::RoutingPolicy)
        );
    }

    // --- ADR-088 Bundle B: TokenCompression in the Free baseline, release-plane
    // --- gated (the holdback IS the Unleash `!released` answer at auth) --------

    #[test]
    fn token_compression_in_free_baseline_only_with_release_plane_kill_switch() {
        // The grant is Free-baseline-only; paid tiers reach it via override.
        assert!(tier_baseline(Tier::Free).contains(&Feature::TokenCompression));
        for tier in [Tier::Standard, Tier::Business, Tier::Enterprise] {
            assert!(!tier_baseline(tier).contains(&Feature::TokenCompression));
        }
        // A default Free tenant is entitled…
        let on = CapabilitySet::resolve(Tier::Free, &BTreeSet::new(), &BTreeSet::new());
        assert!(on.active(Feature::TokenCompression));
        // …but the release plane can hold it back (this is exactly how the
        // Unleash `token_compression` constraint — OFF for tier == free until
        // the deliberate flip — and the RP_ROLLOUT_HOLDBACKS escape hatch land
        // in the resolve expression): active = entitled ∧ released.
        let held = CapabilitySet::resolve(
            Tier::Free,
            &BTreeSet::new(),
            &set([Feature::TokenCompression]),
        );
        assert!(!held.active(Feature::TokenCompression));
        // The rest of the Free baseline is untouched by the holdback.
        assert!(held.active(Feature::RoutingPolicy));
        // Flag-key round trip (the Unleash toggle name + serde wire form).
        assert_eq!(Feature::TokenCompression.flag_key(), "token_compression");
        assert_eq!(
            Feature::from_flag_key("token_compression"),
            Some(Feature::TokenCompression)
        );
        assert_eq!(
            serde_json::to_string(&Feature::TokenCompression).unwrap(),
            "\"token_compression\""
        );
    }

    // --- ADR-050 D1/D6: tenant lifecycle state machine + serve-gate -----------

    #[test]
    fn tenant_state_transition_table_and_serve_gate() {
        use TenantState::*;

        // Default is Active (backward-compatible: an absent field is Active).
        assert_eq!(TenantState::default(), Active);

        // Only Active serves traffic (D6).
        assert!(Active.serves_traffic());
        for s in [Pending, Suspended, Closed] {
            assert!(!s.serves_traffic());
        }

        // Legal transitions (D1).
        for (from, to) in [
            (Pending, Active),
            (Pending, Closed),
            (Active, Suspended),
            (Active, Closed),
            (Suspended, Active),
            (Suspended, Closed),
        ] {
            assert!(
                from.can_transition_to(to),
                "{from:?} -> {to:?} should be legal"
            );
        }

        // Illegal: Closed is terminal, same-state is a no-op, and Pending can't
        // jump straight to Suspended.
        for (from, to) in [
            (Closed, Active),
            (Closed, Suspended),
            (Closed, Closed),
            (Active, Active),
            (Pending, Suspended),
            (Pending, Pending),
        ] {
            assert!(
                !from.can_transition_to(to),
                "{from:?} -> {to:?} should be illegal"
            );
        }

        // serde wire form is the snake_case seam contract.
        assert_eq!(serde_json::to_string(&Suspended).unwrap(), "\"suspended\"");
        assert_eq!(
            serde_json::from_str::<TenantState>("\"closed\"").unwrap(),
            Closed
        );
    }
}
