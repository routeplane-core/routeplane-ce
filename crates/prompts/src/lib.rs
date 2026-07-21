//! Routeplane prompt registry + renderer (PRD-010 / G3.5).
//!
//! A tenant-scoped registry of named prompts with immutable numbered versions,
//! movable labels (the promotion mechanism), declared variables, a logic-less
//! mustache-lite renderer (substitution, present/absent sections, partials), and
//! a lock-free `ArcSwap` snapshot loaded fail-closed from a git-versioned JSON
//! file. The crate is PURE: no network, no DB, no `routeplane_types` dependency.
//! The binding of a rendered template to a `ChatCompletionRequest` happens in the
//! binary (`prompts_api.rs`).
//!
//! See `CLAUDE.md` for the documented render semantics (FR-5/FR-6) and invariants.

use arc_swap::ArcSwap;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::sync::Arc;

/// Maximum partial include depth (FR-6 / FR-18). Root template is depth 0; each
/// `{{> ref}}` increments. A 4th level (depth would be 4) → `prompt_partial_cycle`.
pub const MAX_PARTIAL_DEPTH: usize = 3;

// --- FR-18 bounds ------------------------------------------------------------

/// Registry load bounds (FR-18). Enforced fail-closed at load (a CP `create`
/// would enforce the same at publish). The per-tenant cap is tfvars/env
/// configurable; the rest are fixed by the spec.
#[derive(Debug, Clone, Copy)]
pub struct Bounds {
    pub max_template_bytes: usize,
    pub max_variables: usize,
    pub max_labels: usize,
    pub max_prompts_per_tenant: usize,
}

impl Default for Bounds {
    fn default() -> Self {
        Self {
            max_template_bytes: 64 * 1024, // 64 KiB
            max_variables: 128,
            max_labels: 32,
            max_prompts_per_tenant: 1000,
        }
    }
}

impl Bounds {
    /// Defaults, with the per-tenant prompt cap overridable via
    /// `RP_PROMPTS_MAX_PER_TENANT` (a tfvars-delivered cell parameter). Other
    /// bounds are spec-fixed and not env-tunable.
    pub fn from_env() -> Self {
        let d = Self::default();
        let cap = std::env::var("RP_PROMPTS_MAX_PER_TENANT")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(d.max_prompts_per_tenant);
        Self {
            max_prompts_per_tenant: cap,
            ..d
        }
    }
}

// --- The object model (FR-1/FR-3) --------------------------------------------

/// A declared variable. `required` defaults to `true` (FR-3).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Variable {
    pub name: String,
    #[serde(default = "default_true")]
    pub required: bool,
}

fn default_true() -> bool {
    true
}

/// An immutable prompt version (FR-1). The compiled `nodes`/`declared` are
/// internal render state (not serialized); `template` is retained verbatim for
/// the FR-7 GET surface.
#[derive(Debug)]
pub struct Version {
    pub version: u32,
    pub template: String,
    pub variables: Vec<Variable>,
    pub default_model: Option<String>,
    pub default_params: Option<serde_json::Value>,
    nodes: Vec<Node>,
    declared: BTreeMap<String, bool>,
}

/// A registry prompt: id (`prompt_<slug>`), name, movable labels, immutable
/// versions keyed by number, and ([ADR-152] D2, rung-0 FR-2/FR-3) optional
/// first-class weighted VARIANTS — the prompt itself is the experiment.
#[derive(Debug)]
pub struct StoredPrompt {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub latest_version: u32,
    pub labels: BTreeMap<String, u32>,
    /// [ADR-152] D2: the prompt's weighted variant split. A `Vec`, not a map —
    /// declaration order is load-bearing: the FIRST declared variant is the
    /// control arm (served when no cohort is attributable). Empty (the
    /// default) ⇒ no split declared ⇒ resolution behaves exactly as before
    /// (byte-identical). Assignment is sticky by cohort and weight-STABLE
    /// ([ADR-152] D3; [`assign_variant`]).
    pub variants: Vec<Variant>,
    versions: BTreeMap<u32, Arc<Version>>,
}

/// One first-class variant ([ADR-152] D2, the FR-2 fold): a labelled, weighted
/// pointer at an existing immutable version, optionally layering its own
/// `model`/`params` over that version's defaults at render — one addressable
/// implementation, not just a pointer at a version.
#[derive(Debug, Clone)]
pub struct Variant {
    /// Required, unique per prompt — the analytics identifier
    /// (`UsageEvent.prompt_variant`) and the FR-2 binding name.
    pub label: String,
    /// The version binding the template (+ the defaults `model`/`params`
    /// layer over).
    pub version: u32,
    /// Relative integer weight. Assignment normalizes, so `50/50` and `25/25`
    /// are the SAME split ([ADR-152] D3).
    pub weight: u32,
    /// FR-2: overrides the version's `default_model` when set.
    pub model: Option<String>,
    /// FR-2: layered over the version's `default_params` at render —
    /// variant-key-wins SHALLOW merge for objects (see [`merge_params`]).
    pub params: Option<serde_json::Value>,
}

/// Max declared variants per prompt (fail-closed bound; a split is a small
/// A/B/n, not an unbounded fan-out). Validated at load. (Renamed from
/// `MAX_EXPERIMENT_VARIANTS` with the [ADR-152] D1 cutover.)
pub const MAX_VARIANTS: usize = 16;

/// Deterministic, replica-stable, sticky, WEIGHT-STABLE variant assignment
/// ([ADR-152] D3).
///
/// `cohort_key`:
///   * `Some(key)` — STICKY: the same `(prompt_id, key)` always maps to the
///     same arm, identically on every replica (the fixed-seed FNV-1a
///     [`stable_bucket`], never the per-process `RandomState`).
///   * `None` — the CONTROL arm (the first declared variant): with no cohort
///     we serve a stable control rather than splitting an unattributable
///     request (no silent per-request flapping).
///
/// Mapping: `u = stable_bucket(prompt_id, key) / 2^64` (the cohort's fixed
/// position in the unit interval), walked against CUMULATIVE NORMALIZED
/// weights. This replaces the old `hash % total_weight` mapping, whose
/// assignments reshuffled every cohort on ANY weight change (even a pure
/// scaling like 50/50 → 25/25 — the FR-24 violation [ADR-152] D3 names):
///
///   * **Scaling-invariant, exactly.** Shares are `wᵢ/Σw`; scaling every
///     weight by `k` yields the same real quotients, and IEEE-754 division is
///     correctly rounded, so the f64 thresholds — and every assignment — are
///     bit-identical.
///   * **Minimal movement under shifts.** A cohort's `u` never moves; only
///     the cumulative thresholds do, so the moved population is bounded by
///     the total threshold shift. For a TWO-arm split (the expected shape)
///     the guarantee is exact: a moved cohort lands only on the arm whose
///     normalized share grew. For 3+ arms a middle arm's interval can shift
///     even when its own share did not, so the destination guarantee is
///     per-boundary rather than global — the property tests pin the exact
///     two-arm law and the aggregate movement bound.
///
/// Empty `variants` ⇒ `None` (impossible post-load — validated non-empty when
/// declared — but total on a hand-built registry).
pub fn assign_variant<'a>(
    prompt_id: &str,
    variants: &'a [Variant],
    cohort_key: Option<&str>,
) -> Option<&'a Variant> {
    let first = variants.first()?;
    let Some(key) = cohort_key else {
        return Some(first);
    };
    let total: u64 = variants.iter().map(|v| v.weight as u64).sum();
    if total == 0 {
        // Defensive: validated > 0 at load. Fall back to control.
        return Some(first);
    }
    // The cohort's position in [0, 1): the full 64-bit hash as a fraction of
    // 2^64. (`u64::MAX as f64 + 1.0` is exactly 2^64.)
    let u = stable_bucket(prompt_id, key) as f64 / (u64::MAX as f64 + 1.0);
    let total_f = total as f64;
    let mut cumulative = 0.0f64;
    for variant in variants {
        cumulative += variant.weight as f64 / total_f;
        if u < cumulative {
            return Some(variant);
        }
    }
    // Floating-point tail: the cumulative sum can land a hair under 1.0; the
    // remaining sliver belongs to the last arm.
    variants.last()
}

/// FR-2 params layering ([ADR-152] D2): the variant's `params` over the
/// version's `default_params`, VARIANT-KEY-WINS SHALLOW merge. Two JSON
/// objects merge key-by-key (the variant's value replaces the version's
/// wholesale — no deep merge: params are one level of OpenAI knobs, and a
/// deep merge would make `{"response_format": ...}` overrides order-dependent
/// surprises). Any non-object on either side ⇒ the variant's params win
/// wholesale when present, else the version's.
fn merge_params(
    version: Option<&serde_json::Value>,
    variant: Option<&serde_json::Value>,
) -> Option<serde_json::Value> {
    match (version, variant) {
        (None, None) => None,
        (Some(v), None) => Some(v.clone()),
        (None, Some(o)) => Some(o.clone()),
        (Some(serde_json::Value::Object(base)), Some(serde_json::Value::Object(over))) => {
            let mut merged = base.clone();
            for (k, v) in over {
                merged.insert(k.clone(), v.clone());
            }
            Some(serde_json::Value::Object(merged))
        }
        // Non-object params on either side: the variant's override wins wholesale.
        (_, Some(o)) => Some(o.clone()),
    }
}

/// Variant-label sanity ([ADR-152] D2): non-empty, ≤ 64 bytes, ASCII
/// `[A-Za-z0-9_-]` (the analytics-safe charset `prompt_variant` rides), and
/// never the reserved `v<digits>` version-pin form — the reference grammar
/// owns that shape, so a variant labelled `v3` could never be told apart from
/// a version pin in tooling.
fn variant_label_is_sane(label: &str) -> bool {
    if label.is_empty() || label.len() > 64 {
        return false;
    }
    if !label
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
    {
        return false;
    }
    if let Some(digits) = label.strip_prefix('v') {
        if !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()) {
            return false;
        }
    }
    true
}

/// [ADR-152] D4: the durable ATTRIBUTION hash of a raw cohort key — FNV-1a
/// (64-bit) over the key alone, as 16 lowercase hex chars. This is the JOIN
/// key the telemetry plane persists, unreversed: the raw cohort value (often
/// the OpenAI `user` field) NEVER lands durable (R16); two rows join on the
/// hash. Deliberately keyless and prompt-independent, so an episode-level
/// feedback fold can recompute it from its own target id and match the
/// inference side. Same constants as [`stable_bucket`], different domain
/// (the key alone — assignment hashes `(prompt_id, key)` and stays separate).
pub fn cohort_hash_hex(cohort_key: &str) -> String {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = FNV_OFFSET;
    for &b in cohort_key.as_bytes() {
        hash ^= b as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    format!("{hash:016x}")
}

/// FNV-1a (64-bit) over `name \0 cohort_key` (the [ADR-152] D3 domain is
/// `(prompt_id, cohort)` — the prompt IS the experiment). A FIXED-SEED, fully
/// specified hash — deliberately NOT `std::hash::RandomState` (which is seeded
/// per process and would assign the SAME cohort to different arms on different
/// replicas). FNV-1a is stable across builds, processes, and architectures, so a
/// cohort's assignment is identical everywhere. Hand-rolled to avoid a new
/// dependency (frugality) and to make the stability guarantee explicit.
fn stable_bucket(name: &str, cohort_key: &str) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = FNV_OFFSET;
    let mut mix = |bytes: &[u8]| {
        for &b in bytes {
            hash ^= b as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    };
    mix(name.as_bytes());
    mix(&[0u8]); // domain separator so ("ab","c") != ("a","bc")
    mix(cohort_key.as_bytes());
    hash
}

/// The render-target missing-variable policy (FR-5c). Default `Error`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MissingPolicy {
    #[default]
    Error,
    Empty,
}

// --- Errors (FR-16) ----------------------------------------------------------

/// A resolution/render error, carrying the FR-16 OpenAI error `code`, HTTP
/// status, and optional `param`. The binary maps these into the OpenAI envelope.
#[derive(Debug, Clone)]
pub enum PromptError {
    /// 404 `prompt_not_found` — unknown prompt (FR-4). No existence leak across
    /// tenants (FR-15): a foreign prompt is indistinguishable from a missing one.
    PromptNotFound { reference: String },
    /// 404 `prompt_version_not_found` — unknown label or out-of-range version.
    VersionNotFound { reference: String },
    /// 400 `prompt_render_failed` — a missing required/undeclared variable
    /// (param = name), an unresolvable partial, or a model/params binding failure.
    RenderFailed {
        message: String,
        param: Option<String>,
    },
    /// 400 `prompt_partial_cycle` — a partial cycle or over-depth include.
    PartialCycle { message: String },
    /// 400 `prompt_too_large` — the rendered output or expansion work exceeds
    /// the render caps (FR-18; security-review expansion-bomb guard).
    TooLarge { message: String },
}

impl PromptError {
    pub fn prompt_not_found(reference: &str) -> Self {
        Self::PromptNotFound {
            reference: reference.to_string(),
        }
    }
    pub fn version_not_found(reference: &str) -> Self {
        Self::VersionNotFound {
            reference: reference.to_string(),
        }
    }
    pub fn render_failed(message: String, param: Option<String>) -> Self {
        Self::RenderFailed { message, param }
    }
    pub fn partial_cycle(message: String) -> Self {
        Self::PartialCycle { message }
    }
    pub fn too_large(message: String) -> Self {
        Self::TooLarge { message }
    }

    pub fn code(&self) -> &'static str {
        match self {
            Self::PromptNotFound { .. } => "prompt_not_found",
            Self::VersionNotFound { .. } => "prompt_version_not_found",
            Self::RenderFailed { .. } => "prompt_render_failed",
            Self::PartialCycle { .. } => "prompt_partial_cycle",
            Self::TooLarge { .. } => "prompt_too_large",
        }
    }

    pub fn status(&self) -> u16 {
        match self {
            Self::PromptNotFound { .. } | Self::VersionNotFound { .. } => 404,
            Self::RenderFailed { .. } | Self::PartialCycle { .. } | Self::TooLarge { .. } => 400,
        }
    }

    pub fn message(&self) -> String {
        match self {
            Self::PromptNotFound { reference } => format!("prompt '{reference}' not found"),
            Self::VersionNotFound { reference } => {
                format!("no such version/label for reference '{reference}'")
            }
            Self::RenderFailed { message, .. } => message.clone(),
            Self::PartialCycle { message } => message.clone(),
            Self::TooLarge { message } => message.clone(),
        }
    }

    pub fn param(&self) -> Option<&str> {
        match self {
            Self::RenderFailed { param: Some(p), .. } => Some(p.as_str()),
            _ => None,
        }
    }
}

impl std::fmt::Display for PromptError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message())
    }
}

impl std::error::Error for PromptError {}

// --- A resolved reference (FR-4) ---------------------------------------------

/// The outcome of resolving a `{ref}` against the snapshot: the concrete version,
/// its integer number, and the label it was reached by (FR-17 needs the label).
#[derive(Debug)]
pub struct Resolved<'a> {
    pub prompt_id: String,
    pub version_number: u32,
    pub label: Option<String>,
    /// [ADR-152] D2: the SERVED variant's label when the reference resolved via
    /// the prompt's weighted split. `None` for explicit `@vN`/`@label`
    /// resolution and for prompts with no declared variants — those paths are
    /// unchanged.
    pub variant: Option<String>,
    pub version: &'a Version,
    pub prompt: &'a StoredPrompt,
}

/// A fully rendered prompt (FR-8): the resolved binding metadata plus the
/// rendered text.
#[derive(Debug, Clone)]
pub struct Rendered {
    pub prompt_id: String,
    pub version: u32,
    pub label: Option<String>,
    /// [ADR-152] D2: the SERVED variant's label when the render resolved via
    /// the weighted split, else `None`. Lets the binary annotate the usage
    /// event (`UsageEvent.prompt_variant`) for variant analytics.
    pub variant: Option<String>,
    /// The effective model: the served variant's FR-2 `model` override when
    /// one applies, else the version's `default_model`.
    pub model: Option<String>,
    /// The effective params: the served variant's FR-2 `params` layered over
    /// the version's `default_params` ([`merge_params`], variant-key-wins
    /// shallow merge).
    pub params: Option<serde_json::Value>,
    pub text: String,
}

// --- The registry ------------------------------------------------------------

/// The lock-free, immutable prompt snapshot, keyed `(tenant_id, prompt_id)`.
#[derive(Debug, Default)]
pub struct PromptRegistry {
    tenants: HashMap<String, HashMap<String, StoredPrompt>>,
}

impl PromptRegistry {
    /// An empty registry (no file present): every reference 404s.
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn tenant_count(&self) -> usize {
        self.tenants.len()
    }

    pub fn prompt_count(&self) -> usize {
        self.tenants.values().map(HashMap::len).sum()
    }

    fn lookup(&self, tenant_id: &str, id: &str) -> Option<&StoredPrompt> {
        self.tenants.get(tenant_id).and_then(|m| m.get(id))
    }

    /// Resolve a `{ref}` against this tenant's snapshot (FR-4). `tenant_id` is
    /// server-controlled (the authenticated `TenantContext`) — the structural
    /// isolation guarantee (FR-15).
    ///
    /// This is the cohort-FREE entry point: a bare reference on a prompt with
    /// declared variants resolves the CONTROL arm (no cohort ⇒ control).
    /// Partial includes and the `GET /v1/prompts/{ref}` surface use this path,
    /// so they are stable + side-effect-free. Use `resolve_with_cohort` to
    /// apply a cohort.
    pub fn resolve<'a>(
        &'a self,
        tenant_id: &str,
        reference: &str,
    ) -> Result<Resolved<'a>, PromptError> {
        self.resolve_with_cohort(tenant_id, reference, None)
    }

    /// Resolve a `{ref}` under the [ADR-152] D2 selection precedence:
    ///
    ///   explicit `@vN`  >  explicit `@label`  >  weighted-when-variants-declared
    ///   (a BARE reference on a prompt with a declared split; `cohort_key`
    ///   drives the sticky [`assign_variant`] mapping, `None` ⇒ the control
    ///   arm)  >  `latest`.
    ///
    /// Explicit selectors pin exactly what they name and never split; the old
    /// label-beats-experiment collision rule died with the `experiments` block
    /// (variant labels are validated at load to never collide with label
    /// names, so `@name` is unambiguous).
    pub fn resolve_with_cohort<'a>(
        &'a self,
        tenant_id: &str,
        reference: &str,
        cohort_key: Option<&str>,
    ) -> Result<Resolved<'a>, PromptError> {
        let (id, selector) = parse_reference(reference);
        let prompt = self
            .lookup(tenant_id, id)
            .ok_or_else(|| PromptError::prompt_not_found(reference))?;

        let (version_number, label, variant) = match selector {
            Selector::Latest => {
                // The weighted leg: a bare reference on a prompt with a
                // declared split serves the assigned variant; no split ⇒
                // `latest`, byte-identical to before.
                match assign_variant(&prompt.id, &prompt.variants, cohort_key) {
                    Some(v) => (v.version, None, Some(v.label.clone())),
                    None => (prompt.latest_version, None, None),
                }
            }
            Selector::Label(l) => {
                if let Some(v) = prompt.labels.get(l).copied() {
                    (v, Some(l.to_string()), None)
                } else {
                    return Err(PromptError::version_not_found(reference));
                }
            }
            Selector::Version(n) => {
                if !prompt.versions.contains_key(&n) {
                    return Err(PromptError::version_not_found(reference));
                }
                (n, None, None)
            }
        };

        let version = prompt
            .versions
            .get(&version_number)
            .map(Arc::as_ref)
            .ok_or_else(|| PromptError::version_not_found(reference))?;

        Ok(Resolved {
            prompt_id: prompt.id.clone(),
            version_number,
            label,
            variant,
            version,
            prompt,
        })
    }

    /// Resolve + render (FR-5/FR-6/FR-8). Side-effect-free and deterministic
    /// (NFR-3): the same snapshot + ref + variables yields byte-identical output.
    /// Cohort-FREE: a bare reference on a split prompt renders its control arm.
    pub fn render(
        &self,
        tenant_id: &str,
        reference: &str,
        vars: &BTreeMap<String, serde_json::Value>,
        missing: MissingPolicy,
    ) -> Result<Rendered, PromptError> {
        self.render_with_cohort(tenant_id, reference, vars, missing, None)
    }

    /// Resolve + render with a cohort key for sticky weighted variant
    /// assignment ([ADR-152] D2/D3). Identical to `render` for explicit
    /// `@vN`/`@label` references and for prompts with no declared split (and
    /// for bare split references when `cohort_key` is `None` ⇒ the control
    /// arm), so an unsplit prompt is byte-identical. Determinism is preserved:
    /// the SAME snapshot + ref + cohort + variables yields byte-identical
    /// output, and the assignment is replica-stable (fixed-seed hash).
    pub fn render_with_cohort(
        &self,
        tenant_id: &str,
        reference: &str,
        vars: &BTreeMap<String, serde_json::Value>,
        missing: MissingPolicy,
        cohort_key: Option<&str>,
    ) -> Result<Rendered, PromptError> {
        let resolved = self.resolve_with_cohort(tenant_id, reference, cohort_key)?;
        let ctx = RenderCtx {
            registry: self,
            tenant_id,
            vars,
            missing,
        };
        let mut out = String::new();
        // Seed the cycle stack with the root prompt so a partial pointing back at
        // the root is detected (FR-6).
        let mut stack = vec![resolved.prompt_id.clone()];
        let mut includes: usize = 0;
        render_nodes(
            &ctx,
            &resolved.version.nodes,
            &resolved.version.declared,
            &mut stack,
            0,
            &mut includes,
            &mut out,
        )?;
        check_output_cap(&out)?;
        // FR-2 ([ADR-152] D2): when a variant served the render, its
        // `model`/`params` layer over the version's defaults. Label-unique at
        // load, so the lookup is unambiguous; an explicit `@vN`/`@label`
        // resolution has no served variant and keeps the version's defaults.
        let served = resolved
            .variant
            .as_deref()
            .and_then(|label| resolved.prompt.variants.iter().find(|v| v.label == label));
        let model = served
            .and_then(|v| v.model.clone())
            .or_else(|| resolved.version.default_model.clone());
        let params = merge_params(
            resolved.version.default_params.as_ref(),
            served.and_then(|v| v.params.as_ref()),
        );
        Ok(Rendered {
            prompt_id: resolved.prompt_id,
            version: resolved.version_number,
            label: resolved.label,
            variant: resolved.variant,
            model,
            params,
            text: out,
        })
    }

    /// Load + validate a registry from a JSON string (`{"prompts":[...]}`),
    /// fail-closed (FR-11). `origin` names the source in errors.
    pub fn load_from_json(
        content: &str,
        origin: &str,
        bounds: &Bounds,
    ) -> Result<Self, PromptLoadError> {
        let file: PromptFile =
            serde_json::from_str(content).map_err(|source| PromptLoadError::Parse {
                origin: origin.to_string(),
                source,
            })?;

        let mut tenants: HashMap<String, HashMap<String, StoredPrompt>> = HashMap::new();

        for rec in file.prompts {
            // [ADR-152] D1: the pre-cutover `experiments` block is a dedicated,
            // fail-closed load error — never silently ignored (serde would
            // otherwise drop the unknown key and quietly serve an unsplit
            // prompt where the operator declared a split).
            if rec.experiments.is_some() {
                return Err(PromptLoadError::LegacyExperimentsBlock { id: rec.id });
            }
            if rec.versions.is_empty() {
                return Err(PromptLoadError::NoVersions { id: rec.id });
            }
            if rec.labels.len() > bounds.max_labels {
                return Err(PromptLoadError::TooManyLabels {
                    id: rec.id,
                    count: rec.labels.len(),
                    max: bounds.max_labels,
                });
            }

            let mut versions: BTreeMap<u32, Arc<Version>> = BTreeMap::new();
            for vrec in rec.versions {
                if vrec.version < 1 {
                    return Err(PromptLoadError::InvalidVersionNumber {
                        id: rec.id,
                        version: vrec.version,
                    });
                }
                if vrec.template.len() > bounds.max_template_bytes {
                    return Err(PromptLoadError::TemplateTooLarge {
                        id: rec.id,
                        version: vrec.version,
                        size: vrec.template.len(),
                        max: bounds.max_template_bytes,
                    });
                }
                if vrec.variables.len() > bounds.max_variables {
                    return Err(PromptLoadError::TooManyVariables {
                        id: rec.id,
                        version: vrec.version,
                        count: vrec.variables.len(),
                        max: bounds.max_variables,
                    });
                }
                if versions.contains_key(&vrec.version) {
                    return Err(PromptLoadError::DuplicateVersion {
                        id: rec.id,
                        version: vrec.version,
                    });
                }
                let nodes = parse_template(&vrec.template).map_err(|reason| {
                    PromptLoadError::TemplateParse {
                        id: rec.id.clone(),
                        version: vrec.version,
                        reason,
                    }
                })?;
                let declared = vrec
                    .variables
                    .iter()
                    .map(|v| (v.name.clone(), v.required))
                    .collect();
                versions.insert(
                    vrec.version,
                    Arc::new(Version {
                        version: vrec.version,
                        template: vrec.template,
                        variables: vrec.variables,
                        default_model: vrec.default_model,
                        default_params: vrec.default_params,
                        nodes,
                        declared,
                    }),
                );
            }

            for (label, v) in &rec.labels {
                if !versions.contains_key(v) {
                    return Err(PromptLoadError::LabelTargetMissing {
                        id: rec.id,
                        label: label.clone(),
                        version: *v,
                    });
                }
            }

            // [ADR-152] D2: validate + build the first-class variant split,
            // fail-closed. Declaration order is preserved (first = control).
            let mut total_weight: u64 = 0;
            let mut variants: Vec<Variant> = Vec::with_capacity(rec.variants.len());
            if rec.variants.len() > MAX_VARIANTS {
                return Err(PromptLoadError::TooManyVariants {
                    id: rec.id,
                    count: rec.variants.len(),
                    max: MAX_VARIANTS,
                });
            }
            for vrec in rec.variants {
                // (a) Label sanity: required, non-empty, bounded, analytics-safe
                //     charset. A `v<digits>` label is rejected — the reference
                //     grammar reserves that form for version pins, so such a
                //     label could never be told apart in tooling.
                if !variant_label_is_sane(&vrec.label) {
                    return Err(PromptLoadError::VariantLabelInvalid {
                        id: rec.id,
                        label: vrec.label,
                    });
                }
                // (b) Unique per prompt, and never colliding with a label-map
                //     name — `@name` must stay unambiguous (the D2 rule that
                //     replaced the old label-beats-experiment precedence).
                if variants.iter().any(|v: &Variant| v.label == vrec.label)
                    || rec.labels.contains_key(&vrec.label)
                {
                    return Err(PromptLoadError::VariantLabelCollision {
                        id: rec.id,
                        label: vrec.label,
                    });
                }
                // (c) The bound version must exist.
                if !versions.contains_key(&vrec.version) {
                    return Err(PromptLoadError::VariantVersionMissing {
                        id: rec.id,
                        label: vrec.label,
                        version: vrec.version,
                    });
                }
                total_weight += vrec.weight as u64;
                variants.push(Variant {
                    label: vrec.label,
                    version: vrec.version,
                    weight: vrec.weight,
                    model: vrec.model,
                    params: vrec.params,
                });
            }
            // (d) A DECLARED split must carry weight (all-zero weights would
            //     silently collapse every cohort onto the control arm).
            if !variants.is_empty() && total_weight == 0 {
                return Err(PromptLoadError::VariantsZeroWeight { id: rec.id });
            }

            let latest = match rec.latest_version {
                Some(v) => {
                    if !versions.contains_key(&v) {
                        return Err(PromptLoadError::LatestVersionMissing {
                            id: rec.id,
                            version: v,
                        });
                    }
                    v
                }
                // versions is non-empty (checked); the max key is the latest.
                None => *versions
                    .keys()
                    .next_back()
                    .expect("versions is non-empty (validated above)"),
            };

            let entry = tenants.entry(rec.tenant_id.clone()).or_default();
            if entry.contains_key(&rec.id) {
                return Err(PromptLoadError::DuplicatePrompt {
                    tenant: rec.tenant_id,
                    id: rec.id,
                });
            }
            if entry.len() >= bounds.max_prompts_per_tenant {
                return Err(PromptLoadError::TooManyPrompts {
                    tenant: rec.tenant_id,
                    max: bounds.max_prompts_per_tenant,
                });
            }
            entry.insert(
                rec.id.clone(),
                StoredPrompt {
                    id: rec.id,
                    name: rec.name,
                    description: rec.description,
                    latest_version: latest,
                    labels: rec.labels,
                    variants,
                    versions,
                },
            );
        }

        Ok(Self { tenants })
    }

    /// Load a registry from a file path, fail-closed (FR-11).
    pub fn load_from_file(path: &str, bounds: &Bounds) -> Result<Self, PromptLoadError> {
        let content = std::fs::read_to_string(path).map_err(|source| PromptLoadError::Read {
            path: path.to_string(),
            source,
        })?;
        Self::load_from_json(&content, path, bounds)
    }
}

/// Hot-swappable handle around the snapshot (mirrors `SharedAuthState` /
/// `SharedPolicyRegistry`): readers are wait-free, a future control-plane push
/// `store()`s a new snapshot atomically.
pub type SharedPromptRegistry = Arc<ArcSwap<PromptRegistry>>;

/// Build a hot-swappable handle from an initial registry.
pub fn new_shared_registry(registry: PromptRegistry) -> SharedPromptRegistry {
    Arc::new(ArcSwap::from_pointee(registry))
}

// --- Reference grammar -------------------------------------------------------

enum Selector<'a> {
    Latest,
    Label(&'a str),
    Version(u32),
}

/// `prompt_x` → Latest; `prompt_x@v3` → Version(3); `prompt_x@prod` → Label.
/// The `v<digits>` form is reserved for version pinning (a label literally named
/// `v3` is shadowed — documented).
fn parse_reference(reference: &str) -> (&str, Selector<'_>) {
    match reference.split_once('@') {
        None => (reference, Selector::Latest),
        Some((id, sel)) => {
            if let Some(num) = sel.strip_prefix('v') {
                if !num.is_empty() && num.bytes().all(|b| b.is_ascii_digit()) {
                    if let Ok(n) = num.parse::<u32>() {
                        return (id, Selector::Version(n));
                    }
                }
            }
            (id, Selector::Label(sel))
        }
    }
}

// --- The renderer (logic-less mustache-lite) ---------------------------------

#[derive(Debug)]
enum Token {
    Text(String),
    Var(String),
    SectionOpen(String),
    InvertedOpen(String),
    Close(String),
    Partial(String),
}

#[derive(Debug)]
enum Node {
    Text(String),
    Var(String),
    Partial(String),
    Section {
        name: String,
        inverted: bool,
        children: Vec<Node>,
    },
}

struct RenderCtx<'a> {
    registry: &'a PromptRegistry,
    tenant_id: &'a str,
    vars: &'a BTreeMap<String, serde_json::Value>,
    missing: MissingPolicy,
}

/// Substitution rendering (FR-5a): no HTML escaping. Strings render verbatim;
/// null renders empty; everything else (bool/number/array/object) renders as its
/// compact JSON.
fn value_to_string(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Null => String::new(),
        other => other.to_string(),
    }
}

/// Section truthiness (FR-5b, documented): boolean `true` OR a non-empty string.
/// Everything else is falsy. Sections are presence gates, not iterators.
fn is_truthy(v: &serde_json::Value) -> bool {
    match v {
        serde_json::Value::Bool(b) => *b,
        serde_json::Value::String(s) => !s.is_empty(),
        _ => false,
    }
}

fn parse_template(tpl: &str) -> Result<Vec<Node>, String> {
    let tokens = tokenize(tpl)?;
    let mut pos = 0;
    let nodes = parse_nodes(&tokens, &mut pos, None)?;
    Ok(nodes)
}

fn tokenize(tpl: &str) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let mut rest = tpl;
    let mut text = String::new();
    while !rest.is_empty() {
        if let Some(after) = rest.strip_prefix("{{") {
            if !text.is_empty() {
                tokens.push(Token::Text(std::mem::take(&mut text)));
            }
            let (inner, remaining, triple) = if let Some(a3) = after.strip_prefix('{') {
                let end = a3
                    .find("}}}")
                    .ok_or_else(|| "unclosed triple-stache '{{{'".to_string())?;
                (a3[..end].trim(), &a3[end + 3..], true)
            } else {
                let end = after
                    .find("}}")
                    .ok_or_else(|| "unclosed tag '{{'".to_string())?;
                (after[..end].trim(), &after[end + 2..], false)
            };
            tokens.push(classify_tag(inner, triple)?);
            rest = remaining;
        } else if let Some(ch) = rest.chars().next() {
            text.push(ch);
            rest = &rest[ch.len_utf8()..];
        } else {
            break;
        }
    }
    if !text.is_empty() {
        tokens.push(Token::Text(text));
    }
    Ok(tokens)
}

fn classify_tag(inner: &str, triple: bool) -> Result<Token, String> {
    if triple {
        return non_empty(inner).map(|n| Token::Var(n.to_string()));
    }
    if let Some(r) = inner.strip_prefix('>') {
        return non_empty(r.trim()).map(|n| Token::Partial(n.to_string()));
    }
    if let Some(r) = inner.strip_prefix('#') {
        return non_empty(r.trim()).map(|n| Token::SectionOpen(n.to_string()));
    }
    if let Some(r) = inner.strip_prefix('^') {
        return non_empty(r.trim()).map(|n| Token::InvertedOpen(n.to_string()));
    }
    if let Some(r) = inner.strip_prefix('/') {
        return non_empty(r.trim()).map(|n| Token::Close(n.to_string()));
    }
    if let Some(r) = inner.strip_prefix('&') {
        return non_empty(r.trim()).map(|n| Token::Var(n.to_string()));
    }
    non_empty(inner).map(|n| Token::Var(n.to_string()))
}

fn non_empty(s: &str) -> Result<&str, String> {
    if s.is_empty() {
        Err("empty mustache tag '{{}}'".to_string())
    } else {
        Ok(s)
    }
}

fn parse_nodes(
    tokens: &[Token],
    pos: &mut usize,
    close: Option<&str>,
) -> Result<Vec<Node>, String> {
    let mut nodes = Vec::new();
    while *pos < tokens.len() {
        match &tokens[*pos] {
            Token::Text(t) => {
                nodes.push(Node::Text(t.clone()));
                *pos += 1;
            }
            Token::Var(n) => {
                nodes.push(Node::Var(n.clone()));
                *pos += 1;
            }
            Token::Partial(r) => {
                nodes.push(Node::Partial(r.clone()));
                *pos += 1;
            }
            Token::SectionOpen(n) => {
                let name = n.clone();
                *pos += 1;
                let children = parse_nodes(tokens, pos, Some(&name))?;
                nodes.push(Node::Section {
                    name,
                    inverted: false,
                    children,
                });
            }
            Token::InvertedOpen(n) => {
                let name = n.clone();
                *pos += 1;
                let children = parse_nodes(tokens, pos, Some(&name))?;
                nodes.push(Node::Section {
                    name,
                    inverted: true,
                    children,
                });
            }
            Token::Close(n) => match close {
                Some(expected) if expected == n => {
                    *pos += 1;
                    return Ok(nodes);
                }
                Some(expected) => {
                    return Err(format!(
                        "mismatched section close: expected {{{{/{expected}}}}} got {{{{/{n}}}}}"
                    ));
                }
                None => return Err(format!("unexpected section close {{{{/{n}}}}}")),
            },
        }
    }
    if let Some(expected) = close {
        return Err(format!("unclosed section {{{{#{expected}}}}}"));
    }
    Ok(nodes)
}

/// Render-expansion caps (security review, PR #58): depth bounds the PATH but
/// not the FAN-OUT — without these, one 64 KiB template can demand petabytes
/// of output synchronously (untimeoutable). Output ≤ 256 KiB (surfaces FR-16's
/// prompt_too_large), total partial expansions ≤ 64 per render.
const MAX_RENDER_OUTPUT_BYTES: usize = 256 * 1024;
const MAX_PARTIAL_EXPANSIONS: usize = 64;

fn check_output_cap(out: &str) -> Result<(), PromptError> {
    if out.len() > MAX_RENDER_OUTPUT_BYTES {
        return Err(PromptError::too_large(format!(
            "rendered output exceeds the {MAX_RENDER_OUTPUT_BYTES}-byte cap"
        )));
    }
    Ok(())
}

fn render_nodes(
    ctx: &RenderCtx<'_>,
    nodes: &[Node],
    declared: &BTreeMap<String, bool>,
    stack: &mut Vec<String>,
    depth: usize,
    includes: &mut usize,
    out: &mut String,
) -> Result<(), PromptError> {
    for node in nodes {
        match node {
            Node::Text(t) => {
                out.push_str(t);
                check_output_cap(out)?;
            }
            Node::Var(name) => match ctx.vars.get(name) {
                Some(v) => {
                    out.push_str(&value_to_string(v));
                    check_output_cap(out)?;
                }
                None => match ctx.missing {
                    MissingPolicy::Empty => {}
                    MissingPolicy::Error => {
                        // A declared-optional variable that is unsupplied renders
                        // empty even under strict mode. A required OR undeclared
                        // unsupplied variable is a loud failure (FR-5c).
                        if declared.get(name).copied() != Some(false) {
                            return Err(PromptError::render_failed(
                                format!("missing value for required variable '{name}'"),
                                Some(name.clone()),
                            ));
                        }
                    }
                },
            },
            Node::Section {
                name,
                inverted,
                children,
            } => {
                let truthy = ctx.vars.get(name).map(is_truthy).unwrap_or(false);
                if truthy != *inverted {
                    render_nodes(ctx, children, declared, stack, depth, includes, out)?;
                    check_output_cap(out)?;
                }
            }
            Node::Partial(reference) => {
                if depth + 1 > MAX_PARTIAL_DEPTH {
                    return Err(PromptError::partial_cycle(format!(
                        "partial include depth exceeds {MAX_PARTIAL_DEPTH} at '{reference}' (chain: {})",
                        stack.join(" -> ")
                    )));
                }
                let resolved = ctx
                    .registry
                    .resolve(ctx.tenant_id, reference)
                    .map_err(|e| {
                        PromptError::render_failed(
                            format!("unresolved partial '{reference}': {}", e.message()),
                            Some(reference.clone()),
                        )
                    })?;
                if stack.iter().any(|p| p == &resolved.prompt_id) {
                    return Err(PromptError::partial_cycle(format!(
                        "partial cycle: {} -> {}",
                        stack.join(" -> "),
                        resolved.prompt_id
                    )));
                }
                stack.push(resolved.prompt_id.clone());
                {
                    *includes += 1;
                    if *includes > MAX_PARTIAL_EXPANSIONS {
                        return Err(PromptError::too_large(format!(
                            "render exceeds the {MAX_PARTIAL_EXPANSIONS}-partial-expansion cap"
                        )));
                    }
                    render_nodes(
                        ctx,
                        &resolved.version.nodes,
                        &resolved.version.declared,
                        stack,
                        depth + 1,
                        includes,
                        out,
                    )?;
                    check_output_cap(out)?;
                }
                stack.pop();
            }
        }
    }
    Ok(())
}

// --- File / load model -------------------------------------------------------

#[derive(Debug, Deserialize)]
struct PromptFile {
    prompts: Vec<PromptRecord>,
}

#[derive(Debug, Deserialize)]
struct PromptRecord {
    /// Server-controlled tenant scope (FR-15). This is trusted config (the file
    /// is git-versioned / CP-pushed), NEVER a client value — the lookup key's
    /// tenant component always comes from the authenticated `TenantContext`.
    tenant_id: String,
    id: String,
    name: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    latest_version: Option<u32>,
    #[serde(default)]
    labels: BTreeMap<String, u32>,
    /// [ADR-152] D1: the RETIRED pre-cutover `experiments` block, kept only so
    /// its presence can be detected. `Some(..)` — any shape, any content —
    /// refuses the load with [`PromptLoadError::LegacyExperimentsBlock`]
    /// naming the migration; without this field serde would silently drop the
    /// unknown key and serve an unsplit prompt where the operator declared a
    /// split.
    #[serde(default)]
    experiments: Option<serde_json::Value>,
    /// [ADR-152] D2: the first-class weighted variant split. Absent/empty ⇒
    /// no split (byte-identical resolution); declaration order is
    /// load-bearing — the FIRST declared variant is the control arm.
    #[serde(default)]
    variants: Vec<VariantRecord>,
    versions: Vec<VersionRecord>,
}

/// Wire form of one [ADR-152] D2 variant: a required unique `label`, the
/// version it binds, its relative `weight`, and the optional FR-2
/// `model`/`params` overrides layered over that version's defaults.
#[derive(Debug, Deserialize)]
struct VariantRecord {
    label: String,
    version: u32,
    weight: u32,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    params: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct VersionRecord {
    version: u32,
    template: String,
    #[serde(default)]
    variables: Vec<Variable>,
    #[serde(default)]
    default_model: Option<String>,
    #[serde(default)]
    default_params: Option<serde_json::Value>,
}

/// Typed, fail-closed load errors (FR-11), hand-rolled (no `thiserror`, matching
/// `AuthLoadError`).
#[derive(Debug)]
pub enum PromptLoadError {
    Read {
        path: String,
        source: std::io::Error,
    },
    Parse {
        origin: String,
        source: serde_json::Error,
    },
    DuplicatePrompt {
        tenant: String,
        id: String,
    },
    NoVersions {
        id: String,
    },
    DuplicateVersion {
        id: String,
        version: u32,
    },
    InvalidVersionNumber {
        id: String,
        version: u32,
    },
    TemplateParse {
        id: String,
        version: u32,
        reason: String,
    },
    TemplateTooLarge {
        id: String,
        version: u32,
        size: usize,
        max: usize,
    },
    TooManyVariables {
        id: String,
        version: u32,
        count: usize,
        max: usize,
    },
    TooManyLabels {
        id: String,
        count: usize,
        max: usize,
    },
    TooManyPrompts {
        tenant: String,
        max: usize,
    },
    LabelTargetMissing {
        id: String,
        label: String,
        version: u32,
    },
    LatestVersionMissing {
        id: String,
        version: u32,
    },
    /// [ADR-152] D1: the file still carries the retired `experiments` block.
    LegacyExperimentsBlock {
        id: String,
    },
    TooManyVariants {
        id: String,
        count: usize,
        max: usize,
    },
    VariantLabelInvalid {
        id: String,
        label: String,
    },
    VariantLabelCollision {
        id: String,
        label: String,
    },
    VariantVersionMissing {
        id: String,
        label: String,
        version: u32,
    },
    VariantsZeroWeight {
        id: String,
    },
}

impl std::fmt::Display for PromptLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Read { path, source } => {
                write!(f, "cannot read prompt registry '{path}': {source}")
            }
            Self::Parse { origin, source } => {
                write!(f, "cannot parse prompt registry '{origin}': {source}")
            }
            Self::DuplicatePrompt { tenant, id } => {
                write!(f, "duplicate prompt '{id}' for tenant '{tenant}'")
            }
            Self::NoVersions { id } => write!(f, "prompt '{id}' has no versions"),
            Self::DuplicateVersion { id, version } => {
                write!(f, "prompt '{id}' has duplicate version {version}")
            }
            Self::InvalidVersionNumber { id, version } => {
                write!(
                    f,
                    "prompt '{id}' has invalid version {version} (must be >= 1)"
                )
            }
            Self::TemplateParse {
                id,
                version,
                reason,
            } => write!(
                f,
                "prompt '{id}' v{version} template is malformed: {reason}"
            ),
            Self::TemplateTooLarge {
                id,
                version,
                size,
                max,
            } => write!(
                f,
                "prompt '{id}' v{version} template is {size} bytes (max {max})"
            ),
            Self::TooManyVariables {
                id,
                version,
                count,
                max,
            } => write!(
                f,
                "prompt '{id}' v{version} declares {count} variables (max {max})"
            ),
            Self::TooManyLabels { id, count, max } => {
                write!(f, "prompt '{id}' has {count} labels (max {max})")
            }
            Self::TooManyPrompts { tenant, max } => {
                write!(f, "tenant '{tenant}' exceeds the prompt cap (max {max})")
            }
            Self::LabelTargetMissing { id, label, version } => write!(
                f,
                "prompt '{id}' label '{label}' points at missing version {version}"
            ),
            Self::LatestVersionMissing { id, version } => {
                write!(f, "prompt '{id}' latest_version {version} does not exist")
            }
            Self::LegacyExperimentsBlock { id } => write!(
                f,
                "prompt '{id}' uses the retired 'experiments' block: weights moved onto \
                 first-class 'variants' (ADR-152 hard cutover) — migrate the config to the \
                 new shape; see configs/prompts.example.json"
            ),
            Self::TooManyVariants { id, count, max } => {
                write!(f, "prompt '{id}' declares {count} variants (max {max})")
            }
            Self::VariantLabelInvalid { id, label } => write!(
                f,
                "prompt '{id}' variant label '{label}' is invalid (non-empty ASCII \
                 [A-Za-z0-9_-] up to 64 bytes, and never the reserved v<digits> form)"
            ),
            Self::VariantLabelCollision { id, label } => write!(
                f,
                "prompt '{id}' variant label '{label}' is duplicated or collides with a \
                 label of the same name"
            ),
            Self::VariantVersionMissing { id, label, version } => write!(
                f,
                "prompt '{id}' variant '{label}' references missing version {version}"
            ),
            Self::VariantsZeroWeight { id } => {
                write!(f, "prompt '{id}' declares variants with zero total weight")
            }
        }
    }
}

impl std::error::Error for PromptLoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Read { source, .. } => Some(source),
            Self::Parse { source, .. } => Some(source),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn reg(content: &str) -> PromptRegistry {
        PromptRegistry::load_from_json(content, "test", &Bounds::default()).expect("valid registry")
    }

    fn vars(v: serde_json::Value) -> BTreeMap<String, serde_json::Value> {
        serde_json::from_value(v).expect("vars object")
    }

    const GREETING: &str = r#"{"prompts":[
        {"tenant_id":"t","id":"prompt_x","name":"X","latest_version":2,
         "labels":{"prod":1,"staging":2},
         "versions":[
           {"version":1,"template":"v1 {{name}}","variables":[{"name":"name"}],"default_model":"gpt-4o"},
           {"version":2,"template":"v2 {{name}}","variables":[{"name":"name"}],"default_model":"gpt-4o"}
         ]}
    ]}"#;

    // --- FR-4: reference resolution (all three forms + the 404s) --------------

    #[test]
    fn resolves_latest_label_and_pinned_version() {
        let r = reg(GREETING);
        assert_eq!(r.resolve("t", "prompt_x").unwrap().version_number, 2); // latest
        let prod = r.resolve("t", "prompt_x@prod").unwrap();
        assert_eq!(prod.version_number, 1);
        assert_eq!(prod.label.as_deref(), Some("prod"));
        assert_eq!(r.resolve("t", "prompt_x@v2").unwrap().version_number, 2);
        assert!(r.resolve("t", "prompt_x@v2").unwrap().label.is_none());
    }

    #[test]
    fn unknown_prompt_and_version_have_distinct_codes() {
        let r = reg(GREETING);
        assert_eq!(
            r.resolve("t", "prompt_missing").unwrap_err().code(),
            "prompt_not_found"
        );
        assert_eq!(
            r.resolve("t", "prompt_x@nope").unwrap_err().code(),
            "prompt_version_not_found"
        );
        assert_eq!(
            r.resolve("t", "prompt_x@v9").unwrap_err().code(),
            "prompt_version_not_found"
        );
    }

    // --- FR-15: structural tenant isolation (no existence leak) ---------------

    #[test]
    fn tenant_cannot_resolve_another_tenants_prompt() {
        let r = reg(GREETING);
        let err = r.resolve("other_tenant", "prompt_x").unwrap_err();
        assert_eq!(err.code(), "prompt_not_found"); // 404, not a leak
    }

    // --- FR-5: render determinism, no-escape, sections, missing policies ------

    #[test]
    fn render_is_deterministic() {
        let r = reg(GREETING);
        let v = vars(json!({ "name": "Ada" }));
        let a = r.render("t", "prompt_x", &v, MissingPolicy::Error).unwrap();
        let b = r.render("t", "prompt_x", &v, MissingPolicy::Error).unwrap();
        assert_eq!(a.text, "v2 Ada");
        assert_eq!(a.text, b.text);
    }

    #[test]
    fn substitution_does_not_html_escape() {
        let r = reg(
            r#"{"prompts":[{"tenant_id":"t","id":"prompt_h","name":"H","versions":[
            {"version":1,"template":"{{x}}|{{{x}}}","variables":[{"name":"x"}]}]}]}"#,
        );
        let out = r
            .render(
                "t",
                "prompt_h",
                &vars(json!({"x":"<b>&\"</b>"})),
                MissingPolicy::Error,
            )
            .unwrap();
        // Triple-stache is a no-op alias; neither form escapes.
        assert_eq!(out.text, "<b>&\"</b>|<b>&\"</b>");
    }

    #[test]
    fn sections_present_and_absent() {
        let r = reg(
            r#"{"prompts":[{"tenant_id":"t","id":"prompt_s","name":"S","versions":[
            {"version":1,"template":"{{#b}}Y{{/b}}{{^b}}N{{/b}}","variables":[{"name":"b","required":false}]}]}]}"#,
        );
        let render = |v: serde_json::Value| {
            r.render("t", "prompt_s", &vars(v), MissingPolicy::Error)
                .unwrap()
                .text
        };
        assert_eq!(render(json!({"b":true})), "Y");
        assert_eq!(render(json!({"b":"x"})), "Y"); // non-empty string is truthy
        assert_eq!(render(json!({"b":false})), "N");
        assert_eq!(render(json!({"b":""})), "N"); // empty string is falsy
        assert_eq!(render(json!({})), "N"); // absent → inverted renders
    }

    #[test]
    fn missing_variable_policy_error_vs_empty() {
        // Undeclared {{z}} and required-missing {{a}} both error under default;
        // both render empty under "empty".
        let r = reg(
            r#"{"prompts":[{"tenant_id":"t","id":"prompt_m","name":"M","versions":[
            {"version":1,"template":"[{{a}}{{z}}]","variables":[{"name":"a"}]}]}]}"#,
        );
        let err = r
            .render("t", "prompt_m", &vars(json!({})), MissingPolicy::Error)
            .unwrap_err();
        assert_eq!(err.code(), "prompt_render_failed");
        assert_eq!(err.param(), Some("a"));
        let empty = r
            .render("t", "prompt_m", &vars(json!({})), MissingPolicy::Empty)
            .unwrap();
        assert_eq!(empty.text, "[]");
    }

    #[test]
    fn declared_optional_missing_renders_empty_under_strict() {
        let r = reg(
            r#"{"prompts":[{"tenant_id":"t","id":"prompt_o","name":"O","versions":[
            {"version":1,"template":"[{{a}}]","variables":[{"name":"a","required":false}]}]}]}"#,
        );
        let out = r
            .render("t", "prompt_o", &vars(json!({})), MissingPolicy::Error)
            .unwrap();
        assert_eq!(out.text, "[]");
    }

    // --- FR-6: partials, depth, cycle -----------------------------------------

    const CHAIN: &str = r#"{"prompts":[
        {"tenant_id":"t","id":"prompt_p0","name":"0","versions":[{"version":1,"template":"0{{> prompt_p1}}","variables":[]}]},
        {"tenant_id":"t","id":"prompt_p1","name":"1","versions":[{"version":1,"template":"1{{> prompt_p2}}","variables":[]}]},
        {"tenant_id":"t","id":"prompt_p2","name":"2","versions":[{"version":1,"template":"2{{> prompt_p3}}","variables":[]}]},
        {"tenant_id":"t","id":"prompt_p3","name":"3","versions":[{"version":1,"template":"3{{> prompt_p4}}","variables":[]}]},
        {"tenant_id":"t","id":"prompt_p4","name":"4","versions":[{"version":1,"template":"4","variables":[]}]}
    ]}"#;

    #[test]
    fn three_deep_partial_chain_renders_but_four_is_over_depth() {
        let r = reg(CHAIN);
        // p1 -> p2 -> p3 -> p4 == 3 includes -> renders.
        let ok = r
            .render("t", "prompt_p1", &vars(json!({})), MissingPolicy::Error)
            .unwrap();
        assert_eq!(ok.text, "1234");
        // p0 -> p1 -> p2 -> p3 -> p4 == 4 includes -> over-depth.
        let err = r
            .render("t", "prompt_p0", &vars(json!({})), MissingPolicy::Error)
            .unwrap_err();
        assert_eq!(err.code(), "prompt_partial_cycle");
    }

    #[test]
    fn partial_cycle_is_detected_with_both_ids() {
        let r = reg(r#"{"prompts":[
            {"tenant_id":"t","id":"prompt_a","name":"A","versions":[{"version":1,"template":"a{{> prompt_b}}","variables":[]}]},
            {"tenant_id":"t","id":"prompt_b","name":"B","versions":[{"version":1,"template":"b{{> prompt_a}}","variables":[]}]}
        ]}"#);
        let err = r
            .render("t", "prompt_a", &vars(json!({})), MissingPolicy::Error)
            .unwrap_err();
        assert_eq!(err.code(), "prompt_partial_cycle");
        let msg = err.message();
        assert!(msg.contains("prompt_a") && msg.contains("prompt_b"));
    }

    // --- FR-18: bounds (fail-closed at load) ----------------------------------

    #[test]
    fn oversize_template_refuses_load() {
        let big = "a".repeat(64 * 1024 + 1);
        let json = format!(
            r#"{{"prompts":[{{"tenant_id":"t","id":"prompt_x","name":"X","versions":[{{"version":1,"template":"{big}","variables":[]}}]}}]}}"#
        );
        let err = PromptRegistry::load_from_json(&json, "test", &Bounds::default()).unwrap_err();
        assert!(matches!(err, PromptLoadError::TemplateTooLarge { .. }));
    }

    #[test]
    fn too_many_variables_refuses_load() {
        let vlist: Vec<String> = (0..129).map(|i| format!(r#"{{"name":"v{i}"}}"#)).collect();
        let json = format!(
            r#"{{"prompts":[{{"tenant_id":"t","id":"prompt_x","name":"X","versions":[{{"version":1,"template":"t","variables":[{}]}}]}}]}}"#,
            vlist.join(",")
        );
        let err = PromptRegistry::load_from_json(&json, "test", &Bounds::default()).unwrap_err();
        assert!(matches!(err, PromptLoadError::TooManyVariables { .. }));
    }

    #[test]
    fn per_tenant_cap_refuses_load() {
        let bounds = Bounds {
            max_prompts_per_tenant: 1,
            ..Bounds::default()
        };
        let json = r#"{"prompts":[
            {"tenant_id":"t","id":"prompt_a","name":"A","versions":[{"version":1,"template":"a","variables":[]}]},
            {"tenant_id":"t","id":"prompt_b","name":"B","versions":[{"version":1,"template":"b","variables":[]}]}
        ]}"#;
        let err = PromptRegistry::load_from_json(json, "test", &bounds).unwrap_err();
        assert!(matches!(err, PromptLoadError::TooManyPrompts { .. }));
    }

    // --- FR-11: fail-closed load ----------------------------------------------

    #[test]
    fn malformed_json_refuses_load() {
        let err =
            PromptRegistry::load_from_json("{ not json", "test", &Bounds::default()).unwrap_err();
        assert!(matches!(err, PromptLoadError::Parse { .. }));
    }

    #[test]
    fn label_pointing_at_missing_version_refuses_load() {
        let json = r#"{"prompts":[{"tenant_id":"t","id":"prompt_x","name":"X","labels":{"prod":5},
            "versions":[{"version":1,"template":"t","variables":[]}]}]}"#;
        let err = PromptRegistry::load_from_json(json, "test", &Bounds::default()).unwrap_err();
        assert!(matches!(err, PromptLoadError::LabelTargetMissing { .. }));
    }

    #[test]
    fn empty_registry_is_valid_and_serves_404() {
        let r = PromptRegistry::empty();
        assert_eq!(r.prompt_count(), 0);
        assert_eq!(
            r.resolve("t", "prompt_x").unwrap_err().code(),
            "prompt_not_found"
        );
    }
    // --- render-expansion caps (security review, PR #58) -----------------------

    #[test]
    fn render_output_cap_fires_prompt_too_large() {
        // One variable repeated heavily, fed a large value: output exceeds the
        // 256 KiB cap and must fail loud with prompt_too_large, never OOM.
        let template = "{{v}}".repeat(40);
        let json = format!(
            r#"{{"prompts":[{{"tenant_id":"t","id":"prompt_big","name":"b","latest_version":1,
                "versions":[{{"version":1,"template":"{template}","variables":[{{"name":"v"}}]}}]}}]}}"#
        );
        let reg = PromptRegistry::load_from_json(&json, "test", &Bounds::default()).unwrap();
        let big = "x".repeat(10_000);
        let mut vars = BTreeMap::new();
        vars.insert("v".to_string(), serde_json::Value::String(big));
        let err = reg
            .render("t", "prompt_big", &vars, MissingPolicy::Error)
            .unwrap_err();
        assert_eq!(err.code(), "prompt_too_large");
    }

    // --- [ADR-152] D1/D2/D3: first-class weighted variants ---------------------

    const SPLIT: &str = r#"{"prompts":[
        {"tenant_id":"t","id":"prompt_x","name":"X","latest_version":3,
         "labels":{"prod":1},
         "variants":[
            {"label":"formal","version":2,"weight":50},
            {"label":"casual","version":3,"weight":50}
         ],
         "versions":[
           {"version":1,"template":"v1 {{name}}","variables":[{"name":"name"}]},
           {"version":2,"template":"formal {{name}}","variables":[{"name":"name"}]},
           {"version":3,"template":"casual {{name}}","variables":[{"name":"name"}]}
         ]}
    ]}"#;

    /// Build a variant list for direct [`assign_variant`] property tests.
    fn arms(weights: &[u32]) -> Vec<Variant> {
        weights
            .iter()
            .enumerate()
            .map(|(i, &w)| Variant {
                label: format!("arm-{i}"),
                version: 1,
                weight: w,
                model: None,
                params: None,
            })
            .collect()
    }

    #[test]
    fn bare_reference_on_a_split_prompt_serves_a_variant() {
        let r = reg(SPLIT);
        let res = r
            .resolve_with_cohort("t", "prompt_x", Some("user-1"))
            .unwrap();
        // One of the two declared arms, label None, the served-variant
        // annotation populated.
        assert!(res.version_number == 2 || res.version_number == 3);
        assert!(res.label.is_none());
        let variant = res.variant.clone().unwrap();
        assert!(variant == "formal" || variant == "casual");
    }

    /// [ADR-152] D2 precedence: explicit selectors PIN and never split.
    #[test]
    fn explicit_selectors_beat_the_weighted_split() {
        let r = reg(SPLIT);
        let pinned = r
            .resolve_with_cohort("t", "prompt_x@v1", Some("user-1"))
            .unwrap();
        assert_eq!(pinned.version_number, 1);
        assert!(pinned.variant.is_none());
        let labelled = r
            .resolve_with_cohort("t", "prompt_x@prod", Some("user-1"))
            .unwrap();
        assert_eq!(labelled.version_number, 1);
        assert_eq!(labelled.label.as_deref(), Some("prod"));
        assert!(labelled.variant.is_none());
    }

    #[test]
    fn assignment_is_sticky_and_replica_stable() {
        let r = reg(SPLIT);
        // Same cohort key ⇒ same variant, every call (sticky).
        let a = r
            .resolve_with_cohort("t", "prompt_x", Some("cohort-abc"))
            .unwrap();
        for _ in 0..50 {
            let b = r
                .resolve_with_cohort("t", "prompt_x", Some("cohort-abc"))
                .unwrap();
            assert_eq!(a.version_number, b.version_number);
            assert_eq!(a.variant, b.variant);
        }
        // Replica-stability: recompute the D3 mapping directly from the
        // fixed-seed hash (the domain is (prompt_id, cohort)) — the resolver
        // must agree with the raw math, not merely with itself.
        let u = super::stable_bucket("prompt_x", "cohort-abc") as f64 / (u64::MAX as f64 + 1.0);
        let expected_version = if u < 0.5 { 2 } else { 3 };
        assert_eq!(a.version_number, expected_version);
    }

    #[test]
    fn no_cohort_serves_the_control_arm() {
        let r = reg(SPLIT);
        // No cohort ⇒ the FIRST declared arm (control) = version 2 / "formal".
        let res = r.resolve_with_cohort("t", "prompt_x", None).unwrap();
        assert_eq!(res.version_number, 2);
        assert_eq!(res.variant.as_deref(), Some("formal"));
        // Plain resolve() is cohort-free and must also return the control.
        let plain = r.resolve("t", "prompt_x").unwrap();
        assert_eq!(plain.version_number, 2);
    }

    #[test]
    fn render_threads_variant_and_uses_assigned_version() {
        let r = reg(SPLIT);
        let v = vars(json!({ "name": "Ada" }));
        let out = r
            .render_with_cohort(
                "t",
                "prompt_x",
                &v,
                MissingPolicy::Error,
                Some("cohort-xyz"),
            )
            .unwrap();
        let variant = out.variant.clone().unwrap();
        if out.version == 2 {
            assert_eq!(out.text, "formal Ada");
            assert_eq!(variant, "formal");
        } else {
            assert_eq!(out.text, "casual Ada");
            assert_eq!(variant, "casual");
        }
    }

    /// FR-2: a variant's `model` replaces the version default; its `params`
    /// shallow-merge OVER the version's `default_params`, variant-key-wins.
    #[test]
    fn variant_model_and_params_layer_over_version_defaults() {
        let json = r#"{"prompts":[
            {"tenant_id":"t","id":"prompt_x","name":"X","latest_version":1,
             "variants":[
                {"label":"tuned","version":1,"weight":1,
                 "model":"gpt-4o-mini",
                 "params":{"temperature":0.1,"top_p":0.5}}
             ],
             "versions":[
               {"version":1,"template":"x","variables":[],
                "default_model":"gpt-4o",
                "default_params":{"temperature":0.9,"max_tokens":64}}
             ]}
        ]}"#;
        let r = reg(json);
        let out = r
            .render_with_cohort(
                "t",
                "prompt_x",
                &BTreeMap::new(),
                MissingPolicy::Error,
                Some("k"),
            )
            .unwrap();
        assert_eq!(out.variant.as_deref(), Some("tuned"));
        assert_eq!(out.model.as_deref(), Some("gpt-4o-mini"));
        let params = out.params.unwrap();
        // Variant keys win; untouched version keys survive (shallow merge).
        assert_eq!(params["temperature"], 0.1);
        assert_eq!(params["top_p"], 0.5);
        assert_eq!(params["max_tokens"], 64);

        // An explicit @vN pin bypasses the variant AND its overrides.
        let pinned = r
            .render_with_cohort(
                "t",
                "prompt_x@v1",
                &BTreeMap::new(),
                MissingPolicy::Error,
                Some("k"),
            )
            .unwrap();
        assert!(pinned.variant.is_none());
        assert_eq!(pinned.model.as_deref(), Some("gpt-4o"));
        assert_eq!(pinned.params.unwrap()["temperature"], 0.9);
    }

    // --- [ADR-152] D3: the weight-stable assignment properties -----------------

    /// Seeded distribution converges to the configured weights.
    #[test]
    fn assignment_distribution_converges_to_weights() {
        let variants = arms(&[90, 10]);
        let n = 10_000;
        let mut first = 0u32;
        for i in 0..n {
            let key = format!("user-{i}");
            if assign_variant("prompt_x", &variants, Some(&key))
                .unwrap()
                .label
                == "arm-0"
            {
                first += 1;
            }
        }
        let frac = f64::from(first) / f64::from(n);
        assert!((0.86..0.94).contains(&frac), "fraction was {frac}");
    }

    /// EXACT scaling invariance: `50/50` and `25/25` (and any k-scaling) are
    /// the same split, assignment-for-assignment — the FR-24 fix the old
    /// `hash % total_weight` mapping violated.
    #[test]
    fn assignment_is_invariant_under_weight_scaling() {
        let cases: &[(&[u32], &[u32])] = &[
            (&[50, 50], &[25, 25]),
            (&[90, 10], &[9, 1]),
            (&[3, 5, 7], &[300, 500, 700]),
        ];
        for (w1, w2) in cases {
            let a1 = arms(w1);
            let a2 = arms(w2);
            for i in 0..2_000 {
                let key = format!("user-{i}");
                assert_eq!(
                    assign_variant("prompt_x", &a1, Some(&key)).unwrap().label,
                    assign_variant("prompt_x", &a2, Some(&key)).unwrap().label,
                    "scaling {w1:?} → {w2:?} must not move cohort {key}"
                );
            }
        }
    }

    /// Minimal movement, the exact TWO-arm law: on a weight shift, a moved
    /// cohort lands only on the arm whose normalized share GREW, and the moved
    /// fraction tracks the share change.
    #[test]
    fn two_arm_shift_moves_cohorts_only_onto_the_grown_arm() {
        let before = arms(&[50, 50]);
        let after = arms(&[70, 30]); // arm-0's share grew 0.5 → 0.7
        let n = 10_000u32;
        let mut moved_to_grown = 0u32;
        for i in 0..n {
            let key = format!("user-{i}");
            let b = assign_variant("prompt_x", &before, Some(&key))
                .unwrap()
                .label
                .clone();
            let a = assign_variant("prompt_x", &after, Some(&key))
                .unwrap()
                .label
                .clone();
            if a != b {
                assert_eq!(a, "arm-0", "cohort {key} moved onto a SHRUNK arm");
                moved_to_grown += 1;
            }
        }
        // The moved fraction ≈ the share growth (0.2), never wholesale reshuffle.
        let frac = f64::from(moved_to_grown) / f64::from(n);
        assert!((0.16..0.24).contains(&frac), "moved fraction was {frac}");
    }

    /// The multi-arm aggregate LAW: for a cumulative-threshold scheme, the
    /// moved measure under a weight shift equals the total displacement of the
    /// INTERIOR BOUNDARIES — Σ|C_i(after) − C_i(before)| — which can EXCEED
    /// the total positive share change (the interior-shift effect documented
    /// on [`assign_variant`]; the global per-cohort destination property
    /// belongs to rendezvous-class schemes, deferred per [ADR-152] D3). For
    /// [40,30,30]→[60,20,20]: boundaries move 0.4→0.6 and 0.7→0.8, so the
    /// moved measure is exactly 0.30 — while the share change is only 0.20.
    /// The pin is two-sided: the sampled moved fraction must MATCH the
    /// boundary-displacement law (within sampling noise), which both bounds
    /// the movement (never `hash % total`'s wholesale reshuffle) and proves
    /// the implementation walks the thresholds it claims to.
    #[test]
    fn multi_arm_shift_movement_matches_boundary_displacement() {
        let before = arms(&[40, 30, 30]);
        let after = arms(&[60, 20, 20]);
        // Interior cumulative boundaries: 0.4→0.6 (|Δ|=0.2), 0.7→0.8 (|Δ|=0.1).
        let expected_moved_measure = 0.30;
        let n = 10_000u32;
        let mut moved = 0u32;
        for i in 0..n {
            let key = format!("user-{i}");
            let b = assign_variant("prompt_x", &before, Some(&key))
                .unwrap()
                .label
                .clone();
            let a = assign_variant("prompt_x", &after, Some(&key))
                .unwrap()
                .label
                .clone();
            if a != b {
                moved += 1;
            }
        }
        let frac = f64::from(moved) / f64::from(n);
        // The keys hash through fixed-seed FNV-1a, so this is ONE deterministic
        // sample, not a resampling experiment: its empirical measure at these
        // boundary intervals deviates a fixed ~0.021 from the ideal 0.30 at
        // n=10_000 (observed 0.321, identical on every run). The ±0.03
        // tolerance covers that fixed-sample bias while still discriminating
        // the boundary-displacement law (0.30) from both the naive
        // share-change bound (0.20) and a wholesale reshuffle (~0.66).
        assert!(
            (frac - expected_moved_measure).abs() < 0.03,
            "moved fraction {frac} does not match the boundary-displacement law (expected ≈{expected_moved_measure})"
        );
    }

    // --- [ADR-152] D1: the cutover load rules ----------------------------------

    #[test]
    fn legacy_experiments_block_refuses_load_with_the_migration_error() {
        let json = r#"{"prompts":[
            {"tenant_id":"t","id":"prompt_x","name":"X","latest_version":1,
             "experiments":{"e":{"variants":[{"version":1,"weight":1}]}},
             "versions":[{"version":1,"template":"x","variables":[]}]}
        ]}"#;
        let err = PromptRegistry::load_from_json(json, "test", &Bounds::default()).unwrap_err();
        assert!(matches!(
            err,
            PromptLoadError::LegacyExperimentsBlock { .. }
        ));
        let msg = err.to_string();
        assert!(msg.contains("ADR-152"), "must name the migration: {msg}");
        assert!(
            msg.contains("prompts.example.json"),
            "must point at the example: {msg}"
        );
    }

    #[test]
    fn variant_zero_weight_refuses_load() {
        let json = r#"{"prompts":[
            {"tenant_id":"t","id":"prompt_x","name":"X","latest_version":2,
             "variants":[
                {"label":"a","version":1,"weight":0},
                {"label":"b","version":2,"weight":0}
             ],
             "versions":[
               {"version":1,"template":"a","variables":[]},
               {"version":2,"template":"b","variables":[]}
             ]}
        ]}"#;
        let err = PromptRegistry::load_from_json(json, "test", &Bounds::default()).unwrap_err();
        assert!(matches!(err, PromptLoadError::VariantsZeroWeight { .. }));
    }

    #[test]
    fn variant_label_rules_refuse_bad_and_colliding_labels() {
        // A label in the reserved v<digits> form.
        let reserved = r#"{"prompts":[
            {"tenant_id":"t","id":"prompt_x","name":"X","latest_version":1,
             "variants":[{"label":"v2","version":1,"weight":1}],
             "versions":[{"version":1,"template":"x","variables":[]}]}
        ]}"#;
        assert!(matches!(
            PromptRegistry::load_from_json(reserved, "test", &Bounds::default()).unwrap_err(),
            PromptLoadError::VariantLabelInvalid { .. }
        ));
        // A duplicate variant label.
        let dup = r#"{"prompts":[
            {"tenant_id":"t","id":"prompt_x","name":"X","latest_version":1,
             "variants":[
                {"label":"a","version":1,"weight":1},
                {"label":"a","version":1,"weight":1}
             ],
             "versions":[{"version":1,"template":"x","variables":[]}]}
        ]}"#;
        assert!(matches!(
            PromptRegistry::load_from_json(dup, "test", &Bounds::default()).unwrap_err(),
            PromptLoadError::VariantLabelCollision { .. }
        ));
        // A variant label colliding with a label-map name.
        let collide = r#"{"prompts":[
            {"tenant_id":"t","id":"prompt_x","name":"X","latest_version":1,
             "labels":{"prod":1},
             "variants":[{"label":"prod","version":1,"weight":1}],
             "versions":[{"version":1,"template":"x","variables":[]}]}
        ]}"#;
        assert!(matches!(
            PromptRegistry::load_from_json(collide, "test", &Bounds::default()).unwrap_err(),
            PromptLoadError::VariantLabelCollision { .. }
        ));
    }

    /// [ADR-152] D4: the cohort attribution hash never leaks the raw key, is
    /// deterministic, and pins the exact FNV-1a fold (recomputed inline so an
    /// algorithm change cannot slip through as "still deterministic").
    #[test]
    fn cohort_hash_is_stable_hex_and_never_the_raw_key() {
        let raw = "user-42@example.com";
        let h = cohort_hash_hex(raw);
        assert_eq!(h, cohort_hash_hex(raw), "deterministic");
        assert_ne!(h, raw, "the raw key must never be the durable value");
        assert!(!h.contains(raw));
        assert_eq!(h.len(), 16);
        assert!(h.bytes().all(|b| b.is_ascii_hexdigit()));
        assert_ne!(h, cohort_hash_hex("user-43@example.com"));

        // Pin the algorithm itself: an independent inline FNV-1a fold.
        let mut expect: u64 = 0xcbf2_9ce4_8422_2325;
        for &b in raw.as_bytes() {
            expect ^= b as u64;
            expect = expect.wrapping_mul(0x0000_0100_0000_01b3);
        }
        assert_eq!(h, format!("{expect:016x}"));
    }

    #[test]
    fn variant_missing_version_refuses_load() {
        let json = r#"{"prompts":[
            {"tenant_id":"t","id":"prompt_x","name":"X","latest_version":1,
             "variants":[{"label":"a","version":9,"weight":1}],
             "versions":[{"version":1,"template":"x","variables":[]}]}
        ]}"#;
        let err = PromptRegistry::load_from_json(json, "test", &Bounds::default()).unwrap_err();
        assert!(matches!(err, PromptLoadError::VariantVersionMissing { .. }));
    }

    #[test]
    fn too_many_variants_refuses_load() {
        let arm_rows: Vec<String> = (1..=(MAX_VARIANTS + 1))
            .map(|i| format!(r#"{{"label":"arm-{i}","version":1,"weight":1}}"#))
            .collect();
        let json = format!(
            r#"{{"prompts":[
                {{"tenant_id":"t","id":"prompt_x","name":"X","latest_version":1,
                 "variants":[{}],
                 "versions":[{{"version":1,"template":"x","variables":[]}}]}}
            ]}}"#,
            arm_rows.join(",")
        );
        let err = PromptRegistry::load_from_json(&json, "test", &Bounds::default()).unwrap_err();
        assert!(matches!(err, PromptLoadError::TooManyVariants { .. }));
    }

    #[test]
    fn prompt_without_variants_is_byte_identical_resolution() {
        // The pre-existing GREETING config (no split) resolves exactly as
        // before: latest/label/@vN, no served-variant annotation anywhere.
        let r = reg(GREETING);
        let latest = r.resolve("t", "prompt_x").unwrap();
        assert_eq!(latest.version_number, 2);
        assert!(latest.variant.is_none());
        let prod = r
            .resolve_with_cohort("t", "prompt_x@prod", Some("k"))
            .unwrap();
        assert_eq!(prod.version_number, 1);
        assert_eq!(prod.label.as_deref(), Some("prod"));
        // A cohort key on an unsplit prompt is ignored (no variants declared).
        assert!(prod.variant.is_none());
    }

    #[test]
    fn partial_expansion_count_cap_fires() {
        // Sibling fan-out: depth stays 1 but the SAME partial included >64
        // times must trip the expansion cap (depth bounds the path, not fan-out).
        let includes = "{{> prompt_leaf}}".repeat(70);
        let json = format!(
            r#"{{"prompts":[
                {{"tenant_id":"t","id":"prompt_fan","name":"f","latest_version":1,
                  "versions":[{{"version":1,"template":"{includes}","variables":[]}}]}},
                {{"tenant_id":"t","id":"prompt_leaf","name":"l","latest_version":1,
                  "versions":[{{"version":1,"template":"leaf","variables":[]}}]}}
            ]}}"#
        );
        let reg = PromptRegistry::load_from_json(&json, "test", &Bounds::default()).unwrap();
        let vars: BTreeMap<String, serde_json::Value> = BTreeMap::new();
        let err = reg
            .render("t", "prompt_fan", &vars, MissingPolicy::Error)
            .unwrap_err();
        assert_eq!(err.code(), "prompt_too_large");
    }
}
