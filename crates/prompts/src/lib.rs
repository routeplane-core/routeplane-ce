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
/// versions keyed by number, and (PRD-010, A/B testing) optional named
/// experiments that resolve a version by WEIGHTED, sticky-by-cohort assignment.
#[derive(Debug)]
pub struct StoredPrompt {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub latest_version: u32,
    pub labels: BTreeMap<String, u32>,
    /// PRD-010 (A/B testing): named experiments. Resolved like a label
    /// (`prompt_x@my_experiment`), but the selector yields a version by weighted
    /// assignment over `variants` rather than a fixed mapping. Empty by default —
    /// a prompt with no experiments behaves EXACTLY as before (byte-identical).
    pub experiments: BTreeMap<String, Experiment>,
    versions: BTreeMap<u32, Arc<Version>>,
}

/// PRD-010 (A/B testing): a named weighted experiment over existing immutable
/// versions. Resolution is deterministic + replica-stable (a fixed-seed hash, not
/// per-process `RandomState`) and sticky by cohort key.
#[derive(Debug, Clone)]
pub struct Experiment {
    pub variants: Vec<Variant>,
}

/// One arm of an experiment: an existing version number and its integer weight.
/// `label` is a human-readable arm name carried into analytics (the served
/// variant), distinct from the resolved integer `version`.
#[derive(Debug, Clone)]
pub struct Variant {
    pub label: String,
    pub version: u32,
    pub weight: u32,
}

/// Max variants per experiment (fail-closed bound; an experiment is a small A/B/n
/// split, not an unbounded fan-out). Validated at load.
pub const MAX_EXPERIMENT_VARIANTS: usize = 16;

impl Experiment {
    /// Total weight across arms (sum is validated > 0 at load, so this is a
    /// non-zero modulus base for assignment).
    fn total_weight(&self) -> u64 {
        self.variants.iter().map(|v| v.weight as u64).sum()
    }

    /// Deterministic, replica-stable, sticky weighted assignment. Returns the
    /// chosen `(version, variant_label)`.
    ///
    /// `cohort_key`:
    ///   * `Some(key)` — STICKY: the same `(experiment_name, key)` always maps to
    ///     the same arm, identically on every replica (a fixed-seed FNV-1a hash,
    ///     never the per-process `RandomState`).
    ///   * `None` — falls back to the CONTROL arm (the first declared variant).
    ///     Sticky-by-cohort is the goal; with no cohort we serve a stable control
    ///     rather than splitting an unattributable request (the safest default —
    ///     no silent per-request flapping).
    ///
    /// Assignment: `bucket = stable_hash(name, key) % total_weight`, then the arm
    /// whose cumulative weight first covers `bucket`. Empty `variants` is
    /// impossible post-load (validated), but is handled fail-safe (`None`).
    pub fn assign(&self, name: &str, cohort_key: Option<&str>) -> Option<(u32, String)> {
        let first = self.variants.first()?;
        let Some(key) = cohort_key else {
            // No cohort ⇒ control (first declared arm). Documented default.
            return Some((first.version, first.label.clone()));
        };
        let total = self.total_weight();
        if total == 0 {
            // Defensive: validated > 0 at load. Fall back to control.
            return Some((first.version, first.label.clone()));
        }
        let bucket = stable_bucket(name, key) % total;
        let mut cumulative: u64 = 0;
        for variant in &self.variants {
            cumulative += variant.weight as u64;
            if bucket < cumulative {
                return Some((variant.version, variant.label.clone()));
            }
        }
        // Unreachable (bucket < total == cumulative sum), but stay fail-safe.
        Some((first.version, first.label.clone()))
    }
}

/// FNV-1a (64-bit) over `experiment_name \0 cohort_key`. A FIXED-SEED, fully
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
    /// PRD-010 (A/B testing): when the reference resolved via an experiment, the
    /// `(experiment_name, served_variant_label)`. `None` for plain
    /// latest/label/version resolution — so non-experiment paths are unchanged.
    pub experiment: Option<(String, String)>,
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
    /// PRD-010 (A/B testing): `(experiment_name, served_variant_label)` when the
    /// render resolved via an experiment, else `None`. Lets the binary annotate
    /// the usage event with the served variant for variant analytics.
    pub experiment: Option<(String, String)>,
    pub model: Option<String>,
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
    /// This is the cohort-FREE entry point: an `@name` selector that matches a
    /// LABEL resolves to that label's version; one that matches an EXPERIMENT
    /// resolves to the experiment's CONTROL arm (no cohort ⇒ control). Partial
    /// includes and the `GET /v1/prompts/{ref}` surface use this path, so they are
    /// stable + side-effect-free. Use `resolve_with_cohort` to apply a cohort.
    pub fn resolve<'a>(
        &'a self,
        tenant_id: &str,
        reference: &str,
    ) -> Result<Resolved<'a>, PromptError> {
        self.resolve_with_cohort(tenant_id, reference, None)
    }

    /// Resolve a `{ref}`, applying `cohort_key` when the selector names an
    /// EXPERIMENT (PRD-010 A/B testing). For latest/label/`@vN` references the
    /// cohort is ignored — those paths are byte-identical to before. For an
    /// experiment reference the cohort drives the sticky weighted assignment;
    /// `None` ⇒ the control arm.
    ///
    /// Precedence on a name collision: a LABEL wins over an experiment of the same
    /// name. This is unreachable in practice — `load_*` rejects any
    /// label/experiment name collision fail-closed — but the precedence is fixed
    /// and documented so the resolver is total even on a hand-built registry.
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

        let (version_number, label, experiment) = match selector {
            Selector::Latest => (prompt.latest_version, None, None),
            Selector::Label(l) => {
                // Label takes precedence over an experiment of the same name.
                if let Some(v) = prompt.labels.get(l).copied() {
                    (v, Some(l.to_string()), None)
                } else if let Some(exp) = prompt.experiments.get(l) {
                    let (v, variant) = exp
                        .assign(l, cohort_key)
                        .ok_or_else(|| PromptError::version_not_found(reference))?;
                    (v, None, Some((l.to_string(), variant)))
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
            experiment,
            version,
            prompt,
        })
    }

    /// Resolve + render (FR-5/FR-6/FR-8). Side-effect-free and deterministic
    /// (NFR-3): the same snapshot + ref + variables yields byte-identical output.
    /// Cohort-FREE: an experiment reference renders its control arm.
    pub fn render(
        &self,
        tenant_id: &str,
        reference: &str,
        vars: &BTreeMap<String, serde_json::Value>,
        missing: MissingPolicy,
    ) -> Result<Rendered, PromptError> {
        self.render_with_cohort(tenant_id, reference, vars, missing, None)
    }

    /// Resolve + render with a cohort key for sticky A/B experiment assignment
    /// (PRD-010). Identical to `render` for non-experiment references (and for
    /// experiment references when `cohort_key` is `None` ⇒ the control arm), so a
    /// prompt with no experiments is byte-identical. Determinism is preserved: the
    /// SAME snapshot + ref + cohort + variables yields byte-identical output, and
    /// the assignment is replica-stable (fixed-seed hash).
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
        Ok(Rendered {
            prompt_id: resolved.prompt_id,
            version: resolved.version_number,
            label: resolved.label,
            experiment: resolved.experiment,
            model: resolved.version.default_model.clone(),
            params: resolved.version.default_params.clone(),
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

            // PRD-010 (A/B testing): validate + build experiments, fail-closed.
            let mut experiments: BTreeMap<String, Experiment> = BTreeMap::new();
            for (exp_name, erec) in rec.experiments {
                // (a) No collision with a label name — the `@name` selector must be
                //     unambiguous (label-vs-experiment precedence is documented, but
                //     a collision is a config bug, rejected fail-closed).
                if rec.labels.contains_key(&exp_name) {
                    return Err(PromptLoadError::ExperimentNameCollision {
                        id: rec.id,
                        name: exp_name,
                    });
                }
                // (b) Non-empty, bounded variant count.
                if erec.variants.is_empty() {
                    return Err(PromptLoadError::ExperimentNoVariants {
                        id: rec.id,
                        name: exp_name,
                    });
                }
                if erec.variants.len() > MAX_EXPERIMENT_VARIANTS {
                    return Err(PromptLoadError::ExperimentTooManyVariants {
                        id: rec.id,
                        name: exp_name,
                        count: erec.variants.len(),
                        max: MAX_EXPERIMENT_VARIANTS,
                    });
                }
                // (c) Each arm points at an existing version; (d) weights sum > 0.
                let mut total_weight: u64 = 0;
                let mut variants: Vec<Variant> = Vec::with_capacity(erec.variants.len());
                for vrec in erec.variants {
                    if !versions.contains_key(&vrec.version) {
                        return Err(PromptLoadError::ExperimentVariantMissing {
                            id: rec.id,
                            name: exp_name,
                            version: vrec.version,
                        });
                    }
                    total_weight += vrec.weight as u64;
                    // A variant label defaults to `v<version>` when omitted, so the
                    // served-variant analytics annotation is always populated.
                    let label = vrec.label.unwrap_or_else(|| format!("v{}", vrec.version));
                    variants.push(Variant {
                        label,
                        version: vrec.version,
                        weight: vrec.weight,
                    });
                }
                if total_weight == 0 {
                    return Err(PromptLoadError::ExperimentZeroWeight {
                        id: rec.id,
                        name: exp_name,
                    });
                }
                experiments.insert(exp_name, Experiment { variants });
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
                    experiments,
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
    /// PRD-010 (A/B testing): optional named experiments. Absent by default, so a
    /// prompt config without experiments parses + behaves identically to before.
    #[serde(default)]
    experiments: BTreeMap<String, ExperimentRecord>,
    versions: Vec<VersionRecord>,
}

#[derive(Debug, Deserialize)]
struct ExperimentRecord {
    variants: Vec<VariantRecord>,
}

#[derive(Debug, Deserialize)]
struct VariantRecord {
    version: u32,
    weight: u32,
    /// Human-readable arm name for analytics; defaults to `v<version>` if omitted.
    #[serde(default)]
    label: Option<String>,
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
    ExperimentNameCollision {
        id: String,
        name: String,
    },
    ExperimentNoVariants {
        id: String,
        name: String,
    },
    ExperimentTooManyVariants {
        id: String,
        name: String,
        count: usize,
        max: usize,
    },
    ExperimentVariantMissing {
        id: String,
        name: String,
        version: u32,
    },
    ExperimentZeroWeight {
        id: String,
        name: String,
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
            Self::ExperimentNameCollision { id, name } => write!(
                f,
                "prompt '{id}' experiment '{name}' collides with a label of the same name"
            ),
            Self::ExperimentNoVariants { id, name } => {
                write!(f, "prompt '{id}' experiment '{name}' has no variants")
            }
            Self::ExperimentTooManyVariants {
                id,
                name,
                count,
                max,
            } => write!(
                f,
                "prompt '{id}' experiment '{name}' has {count} variants (max {max})"
            ),
            Self::ExperimentVariantMissing { id, name, version } => write!(
                f,
                "prompt '{id}' experiment '{name}' references missing version {version}"
            ),
            Self::ExperimentZeroWeight { id, name } => {
                write!(f, "prompt '{id}' experiment '{name}' has zero total weight")
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

    // --- PRD-010 (A/B testing): experiments ------------------------------------

    const EXP: &str = r#"{"prompts":[
        {"tenant_id":"t","id":"prompt_x","name":"X","latest_version":3,
         "labels":{"prod":1},
         "experiments":{"tone":{"variants":[
            {"version":2,"weight":50,"label":"formal"},
            {"version":3,"weight":50,"label":"casual"}
         ]}},
         "versions":[
           {"version":1,"template":"v1 {{name}}","variables":[{"name":"name"}]},
           {"version":2,"template":"formal {{name}}","variables":[{"name":"name"}]},
           {"version":3,"template":"casual {{name}}","variables":[{"name":"name"}]}
         ]}
    ]}"#;

    #[test]
    fn experiment_parses_and_resolves_a_variant() {
        let r = reg(EXP);
        let res = r
            .resolve_with_cohort("t", "prompt_x@tone", Some("user-1"))
            .unwrap();
        // The resolved version is one of the two arms, label is None (experiment),
        // and the experiment annotation carries (name, served_variant).
        assert!(res.version_number == 2 || res.version_number == 3);
        assert!(res.label.is_none());
        let (exp_name, variant) = res.experiment.clone().unwrap();
        assert_eq!(exp_name, "tone");
        assert!(variant == "formal" || variant == "casual");
    }

    #[test]
    fn experiment_assignment_is_sticky_and_replica_stable() {
        let r = reg(EXP);
        // Same cohort key ⇒ same variant, every call (sticky).
        let a = r
            .resolve_with_cohort("t", "prompt_x@tone", Some("cohort-abc"))
            .unwrap();
        for _ in 0..50 {
            let b = r
                .resolve_with_cohort("t", "prompt_x@tone", Some("cohort-abc"))
                .unwrap();
            assert_eq!(a.version_number, b.version_number);
            assert_eq!(a.experiment, b.experiment);
        }
        // Replica-stability: the bucket is a fixed-seed hash, so a known cohort
        // maps to a fixed bucket regardless of process. Assert the raw mapping is
        // stable (not just self-consistent) by recomputing it directly.
        let total: u64 = 100;
        let bucket = super::stable_bucket("tone", "cohort-abc") % total;
        let expected_version = if bucket < 50 { 2 } else { 3 };
        assert_eq!(a.version_number, expected_version);
    }

    #[test]
    fn experiment_distribution_roughly_matches_weights() {
        // 90/10 split over many distinct cohort keys lands near the weights.
        let json = r#"{"prompts":[
            {"tenant_id":"t","id":"prompt_x","name":"X","latest_version":2,
             "experiments":{"split":{"variants":[
                {"version":1,"weight":90,"label":"a"},
                {"version":2,"weight":10,"label":"b"}
             ]}},
             "versions":[
               {"version":1,"template":"a","variables":[]},
               {"version":2,"template":"b","variables":[]}
             ]}
        ]}"#;
        let r = reg(json);
        let n = 10_000;
        let mut a = 0u32;
        for i in 0..n {
            let key = format!("user-{i}");
            let res = r
                .resolve_with_cohort("t", "prompt_x@split", Some(&key))
                .unwrap();
            if res.version_number == 1 {
                a += 1;
            }
        }
        let frac = a as f64 / n as f64;
        // Expect ~0.90; allow generous slack for the finite sample.
        assert!(frac > 0.86 && frac < 0.94, "fraction was {frac}");
    }

    #[test]
    fn experiment_no_cohort_falls_back_to_control() {
        let r = reg(EXP);
        // No cohort ⇒ the FIRST declared arm (control) = version 2 / "formal".
        let res = r.resolve_with_cohort("t", "prompt_x@tone", None).unwrap();
        assert_eq!(res.version_number, 2);
        assert_eq!(res.experiment.unwrap().1, "formal");
        // Plain resolve() is cohort-free and must also return the control.
        let plain = r.resolve("t", "prompt_x@tone").unwrap();
        assert_eq!(plain.version_number, 2);
    }

    #[test]
    fn experiment_render_threads_annotation_and_uses_assigned_version() {
        let r = reg(EXP);
        let v = vars(json!({ "name": "Ada" }));
        let out = r
            .render_with_cohort(
                "t",
                "prompt_x@tone",
                &v,
                MissingPolicy::Error,
                Some("cohort-xyz"),
            )
            .unwrap();
        // The rendered text matches the assigned version's template.
        let (_, variant) = out.experiment.clone().unwrap();
        if out.version == 2 {
            assert_eq!(out.text, "formal Ada");
            assert_eq!(variant, "formal");
        } else {
            assert_eq!(out.text, "casual Ada");
            assert_eq!(variant, "casual");
        }
    }

    #[test]
    fn experiment_label_default_is_v_prefixed_version() {
        let json = r#"{"prompts":[
            {"tenant_id":"t","id":"prompt_x","name":"X","latest_version":1,
             "experiments":{"e":{"variants":[{"version":1,"weight":1}]}},
             "versions":[{"version":1,"template":"x","variables":[]}]}
        ]}"#;
        let r = reg(json);
        let res = r.resolve_with_cohort("t", "prompt_x@e", Some("k")).unwrap();
        assert_eq!(res.experiment.unwrap().1, "v1");
    }

    #[test]
    fn experiment_empty_variants_refuses_load() {
        let json = r#"{"prompts":[
            {"tenant_id":"t","id":"prompt_x","name":"X","latest_version":1,
             "experiments":{"e":{"variants":[]}},
             "versions":[{"version":1,"template":"x","variables":[]}]}
        ]}"#;
        let err = PromptRegistry::load_from_json(json, "test", &Bounds::default()).unwrap_err();
        assert!(matches!(err, PromptLoadError::ExperimentNoVariants { .. }));
    }

    #[test]
    fn experiment_zero_weight_refuses_load() {
        let json = r#"{"prompts":[
            {"tenant_id":"t","id":"prompt_x","name":"X","latest_version":2,
             "experiments":{"e":{"variants":[
                {"version":1,"weight":0},{"version":2,"weight":0}
             ]}},
             "versions":[
               {"version":1,"template":"a","variables":[]},
               {"version":2,"template":"b","variables":[]}
             ]}
        ]}"#;
        let err = PromptRegistry::load_from_json(json, "test", &Bounds::default()).unwrap_err();
        assert!(matches!(err, PromptLoadError::ExperimentZeroWeight { .. }));
    }

    #[test]
    fn experiment_missing_version_refuses_load() {
        let json = r#"{"prompts":[
            {"tenant_id":"t","id":"prompt_x","name":"X","latest_version":1,
             "experiments":{"e":{"variants":[{"version":9,"weight":1}]}},
             "versions":[{"version":1,"template":"x","variables":[]}]}
        ]}"#;
        let err = PromptRegistry::load_from_json(json, "test", &Bounds::default()).unwrap_err();
        assert!(matches!(
            err,
            PromptLoadError::ExperimentVariantMissing { .. }
        ));
    }

    #[test]
    fn experiment_name_colliding_with_label_refuses_load() {
        let json = r#"{"prompts":[
            {"tenant_id":"t","id":"prompt_x","name":"X","latest_version":1,
             "labels":{"prod":1},
             "experiments":{"prod":{"variants":[{"version":1,"weight":1}]}},
             "versions":[{"version":1,"template":"x","variables":[]}]}
        ]}"#;
        let err = PromptRegistry::load_from_json(json, "test", &Bounds::default()).unwrap_err();
        assert!(matches!(
            err,
            PromptLoadError::ExperimentNameCollision { .. }
        ));
    }

    #[test]
    fn experiment_too_many_variants_refuses_load() {
        let arms: Vec<String> = (1..=(MAX_EXPERIMENT_VARIANTS + 1))
            .map(|_| r#"{"version":1,"weight":1}"#.to_string())
            .collect();
        let json = format!(
            r#"{{"prompts":[
                {{"tenant_id":"t","id":"prompt_x","name":"X","latest_version":1,
                 "experiments":{{"e":{{"variants":[{}]}}}},
                 "versions":[{{"version":1,"template":"x","variables":[]}}]}}
            ]}}"#,
            arms.join(",")
        );
        let err = PromptRegistry::load_from_json(&json, "test", &Bounds::default()).unwrap_err();
        assert!(matches!(
            err,
            PromptLoadError::ExperimentTooManyVariants { .. }
        ));
    }

    #[test]
    fn prompt_without_experiments_is_byte_identical_resolution() {
        // The pre-existing GREETING config (no experiments) resolves exactly as
        // before: latest/label/@vN, no experiment annotation anywhere.
        let r = reg(GREETING);
        let latest = r.resolve("t", "prompt_x").unwrap();
        assert_eq!(latest.version_number, 2);
        assert!(latest.experiment.is_none());
        let prod = r
            .resolve_with_cohort("t", "prompt_x@prod", Some("k"))
            .unwrap();
        assert_eq!(prod.version_number, 1);
        assert_eq!(prod.label.as_deref(), Some("prod"));
        // A cohort key on a non-experiment ref is ignored (no experiment set).
        assert!(prod.experiment.is_none());
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
