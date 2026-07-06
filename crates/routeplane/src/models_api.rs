//! `GET /v1/models` + `GET /v1/models/{id}` — the OpenAI-compatible
//! model-discovery surface (PARITY: LiteLLM and Portkey both expose it, and
//! every OpenAI-compatible client — the OpenAI SDK, LangChain, LlamaIndex —
//! calls it to enumerate models).
//!
//! Contract (verified against OpenAI / LiteLLM):
//!   * `GET /v1/models`      → 200 `{"object":"list","data":[<model>, …]}`
//!   * `GET /v1/models/{id}` → 200 the single `<model>` object, or 404 (the
//!     shared OpenAI error envelope) for an unknown id.
//!   * A `<model>` is `{"id","object":"model","created":<unix>,"owned_by"}`.
//!
//! Auth: this rides the AUTHED router (the same `auth_middleware` layer as
//! `/v1/chat/completions`), so an unauthenticated caller is rejected with the
//! standard 401 invalid_api_key envelope before the handler ever runs. There is
//! NO per-tenant model allowlist today — the full catalog is returned to any
//! authenticated caller (matching LiteLLM's default behavior).
//!
//! The catalog is a **static, curated** list (a `LazyLock`): the well-known
//! model ids each registered provider's adapter accepts, tagged with the
//! provider name as `owned_by`. `created` is a **fixed constant** per entry
//! (not `now()`) — OpenAI returns stable per-model timestamps and determinism
//! keeps the golden/parity guards happy. Env-configured deployments
//! (azure_openai, self_hosted) are folded in at request time when discoverable.
//!
//! No DB, no network, no locks — a read of static data. Nothing here can panic
//! on the request thread (no `unwrap`/`expect`).

use crate::proxy::AppState;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::Serialize;
use std::sync::{Arc, LazyLock};

/// A single OpenAI-shaped model object.
///
/// The first four fields (`id`/`object`/`created`/`owned_by`) are the **exact**
/// OpenAI model shape and MUST NOT change — every OpenAI-compatible SDK depends
/// on them. Enrichment ([ADR-035]) is ADDITIVE under a single reserved extension
/// key, `routeplane`, which OpenAI SDKs ignore as an unknown field. The extension
/// is omitted entirely when an entry has no known metadata, so a bare entry is
/// byte-identical to the pre-enrichment shape.
///
/// [ADR-035]: ../../../../docs/adr/035-catalog-enrichment-and-compliance-gating.md
#[derive(Debug, Clone, Serialize)]
pub struct ModelObject {
    /// The model id a caller passes as `model` in a request (e.g. `gpt-4o`).
    pub id: String,
    /// Always the literal string `"model"` (the OpenAI discriminator).
    pub object: &'static str,
    /// Unix epoch seconds. Stable per model (matches OpenAI's fixed timestamps).
    pub created: u64,
    /// The provider that owns the model — the `x-routeplane-provider` name to
    /// route to (e.g. `openai`, `anthropic`, `gemini`).
    pub owned_by: String,
    /// Reserved Routeplane extension object ([ADR-035] catalog enrichment): per-model
    /// cost / modalities / capabilities / context_window. `None` ⇒ omitted from the
    /// wire (additive; OpenAI core fields untouched). NOT part of the OpenAI contract.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub routeplane: Option<ModelMetadata>,
}

/// The Routeplane per-model metadata block ([ADR-035] §1, ATTRIBUTE half).
///
/// Data-only, static/const-derived — no DB, no network, no standing cost (it rides
/// the same static-catalog posture as the rest of this module). This pass implements
/// the catalog-ATTRIBUTE enrichment (`cost`/`modalities`/`capabilities`/
/// `context_window`) AND the ADR-035 §4 org compliance-framework gate substrate
/// (`compliance_restrictions` — the frameworks that do NOT accept a model;
/// consumed by the proxy's `403 model_compliance_excluded` default-deny via
/// [`compliance_restrictions_for`]). The other ADR-035 attributes (`use_cases`,
/// `latency_class`, `resident_regions`, `billing_modes`, `status`) are out of
/// scope for this pass.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ModelMetadata {
    /// Per-1k-token economics, derived from the `limits` crate `PRICE_BOOK` (no
    /// duplicated rates). `None` for models the price book does not cover.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost: Option<CostMeta>,
    /// Input/output modalities the model accepts/produces (`text|audio|vision|embeddings`).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub modalities: Vec<&'static str>,
    /// Feature capabilities (`function-calling|json-mode|long-context|streaming`).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub capabilities: Vec<&'static str>,
    /// Max context window in tokens, where known. `None` when not confidently known
    /// (we do not fabricate a number we cannot justify — ADR-035 source discipline).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u32>,
    /// Compliance frameworks that do **NOT** accept this model as standard ([ADR-035]
    /// §4 — the org-gate exclusion set). A tenant operating under any framework in
    /// this list is blocked (`strict`) or warned (`warn`) from reaching the model.
    /// Conservative/defensible: empty for a model we cannot justify restricting.
    /// Surfaced on the `routeplane` extension so a CISO can audit the posture from
    /// `GET /v1/models`; it is additive (omitted when empty → byte-identical).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub compliance_restrictions: Vec<&'static str>,
}

impl ModelMetadata {
    /// `true` when this block carries no information at all, so the enclosing
    /// `routeplane` extension should be omitted (keeps a bare entry byte-identical).
    fn is_empty(&self) -> bool {
        self.cost.is_none()
            && self.modalities.is_empty()
            && self.capabilities.is_empty()
            && self.context_window.is_none()
            && self.compliance_restrictions.is_empty()
    }
}

/// Per-model cost metadata ([ADR-035] §1 `cost`). Integer **micro-USD per 1k
/// tokens** (no floats in money), derived from the `limits` crate `PRICE_BOOK`
/// (micro-USD per *million* tokens) via `per_1k = per_million / 1000`. The rate
/// table is NOT duplicated here — `routeplane_limits::price_for` is the single
/// source of truth.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct CostMeta {
    /// Prompt/input cost, micro-USD per 1k tokens.
    pub input_per_1k_micro_usd: u64,
    /// Completion/output cost, micro-USD per 1k tokens.
    pub output_per_1k_micro_usd: u64,
    /// A 0..1 band over the known price range (cheap→0, expensive→1), for the
    /// [ADR-033] smart-router cost signal. Deterministic; see `normalized_cost`.
    pub normalized_cost_param: f32,
    /// Provenance of the figures — `"routeplane_price_book"` (ADR-051 / ADR-035 FR-24).
    pub source: &'static str,
}

impl ModelObject {
    fn new(id: impl Into<String>, owned_by: impl Into<String>, created: u64) -> Self {
        let id: String = id.into();
        let routeplane = enrich(&id);
        ModelObject {
            id,
            object: "model",
            created,
            owned_by: owned_by.into(),
            routeplane,
        }
    }
}

// ---------------------------------------------------------------------------
// Enrichment ([ADR-035] §1, ATTRIBUTE half) — static, const-derived, no DB.
// ---------------------------------------------------------------------------

/// Per-model (or per-substring) static metadata: modalities / capabilities /
/// context_window. Mirrors the `PRICE_BOOK` **most-specific-first** convention so
/// the first row whose substring the model id contains wins (`gpt-4o-mini` resolves
/// before `gpt-4o`; `claude-3-5-haiku` before `claude-3`). `cost` is NOT here — it
/// is derived from the `limits` `PRICE_BOOK` at build time, never duplicated.
///
/// Conservative by design: well-known models carry accurate values; an unknown id
/// matching no row gets an empty/None block (the `routeplane` extension is then
/// omitted). `context_window` is set only where confidently known.
struct MetaRow {
    /// Substring matched against the model id (most-specific-first).
    pat: &'static str,
    modalities: &'static [&'static str],
    capabilities: &'static [&'static str],
    context_window: Option<u32>,
    /// Frameworks that do NOT accept this model ([ADR-035] §4). Most rows are
    /// `&[]` (no defensible restriction); a few well-known cases are seeded.
    /// Framework codes are the §5 registry identifiers (`DPDP`/`HIPAA`/…) — config
    /// strings, never user content, so citing them in a 403 is no-reflection-safe.
    compliance_restrictions: &'static [&'static str],
}

// Capability/modality vocab (ADR-035 §1): modalities text|audio|vision|embeddings;
// capabilities function-calling|json-mode|long-context|streaming.
const META_TABLE: &[MetaRow] = &[
    // ---- OpenAI chat (gpt-4o-mini before gpt-4o; gpt-4-turbo before gpt-4) ----
    MetaRow {
        pat: "gpt-4o-mini-tts",
        modalities: &["audio"],
        capabilities: &["streaming"],
        context_window: None,
        compliance_restrictions: &[],
    },
    MetaRow {
        pat: "gpt-4o-transcribe",
        modalities: &["audio"],
        capabilities: &["streaming"],
        context_window: None,
        compliance_restrictions: &[],
    },
    MetaRow {
        pat: "gpt-4o-mini",
        modalities: &["text", "vision"],
        capabilities: &["function-calling", "json-mode", "streaming", "long-context"],
        context_window: Some(128_000),
        compliance_restrictions: &[],
    },
    MetaRow {
        pat: "gpt-4o",
        modalities: &["text", "vision"],
        capabilities: &["function-calling", "json-mode", "streaming", "long-context"],
        context_window: Some(128_000),
        compliance_restrictions: &[],
    },
    MetaRow {
        pat: "gpt-4-turbo",
        modalities: &["text", "vision"],
        capabilities: &["function-calling", "json-mode", "streaming", "long-context"],
        context_window: Some(128_000),
        compliance_restrictions: &[],
    },
    MetaRow {
        pat: "gpt-4",
        modalities: &["text"],
        capabilities: &["function-calling", "json-mode", "streaming"],
        context_window: Some(8_192),
        compliance_restrictions: &[],
    },
    MetaRow {
        pat: "gpt-3.5-turbo",
        modalities: &["text"],
        capabilities: &["function-calling", "json-mode", "streaming"],
        context_window: Some(16_385),
        compliance_restrictions: &[],
    },
    MetaRow {
        pat: "o1-mini",
        modalities: &["text"],
        capabilities: &["streaming", "long-context"],
        context_window: Some(128_000),
        compliance_restrictions: &[],
    },
    MetaRow {
        pat: "o1",
        modalities: &["text", "vision"],
        capabilities: &["function-calling", "json-mode", "streaming", "long-context"],
        context_window: Some(200_000),
        compliance_restrictions: &[],
    },
    // ---- OpenAI non-chat ----
    MetaRow {
        pat: "text-embedding",
        modalities: &["embeddings"],
        capabilities: &[],
        context_window: Some(8_191),
        compliance_restrictions: &[],
    },
    MetaRow {
        pat: "moderation",
        modalities: &["text"],
        capabilities: &[],
        context_window: None,
        compliance_restrictions: &[],
    },
    MetaRow {
        pat: "gpt-image-1",
        modalities: &["vision"],
        capabilities: &[],
        context_window: None,
        compliance_restrictions: &[],
    },
    MetaRow {
        pat: "dall-e",
        modalities: &["vision"],
        capabilities: &[],
        context_window: None,
        compliance_restrictions: &[],
    },
    MetaRow {
        pat: "whisper",
        modalities: &["audio"],
        capabilities: &[],
        context_window: None,
        compliance_restrictions: &[],
    },
    MetaRow {
        pat: "tts",
        modalities: &["audio"],
        capabilities: &["streaming"],
        context_window: None,
        compliance_restrictions: &[],
    },
    // ---- Anthropic (claude-3-5-haiku before claude-3-haiku before claude-3) ----
    MetaRow {
        pat: "claude-3-5-sonnet",
        modalities: &["text", "vision"],
        capabilities: &["function-calling", "json-mode", "streaming", "long-context"],
        context_window: Some(200_000),
        compliance_restrictions: &[],
    },
    MetaRow {
        pat: "claude-3-5-haiku",
        modalities: &["text", "vision"],
        capabilities: &["function-calling", "json-mode", "streaming", "long-context"],
        context_window: Some(200_000),
        compliance_restrictions: &[],
    },
    MetaRow {
        pat: "claude-3-opus",
        modalities: &["text", "vision"],
        capabilities: &["function-calling", "json-mode", "streaming", "long-context"],
        context_window: Some(200_000),
        compliance_restrictions: &[],
    },
    MetaRow {
        pat: "claude-3-haiku",
        modalities: &["text", "vision"],
        capabilities: &["function-calling", "json-mode", "streaming", "long-context"],
        context_window: Some(200_000),
        compliance_restrictions: &[],
    },
    MetaRow {
        pat: "claude",
        modalities: &["text", "vision"],
        capabilities: &["function-calling", "json-mode", "streaming", "long-context"],
        context_window: Some(200_000),
        compliance_restrictions: &[],
    },
    // ---- Gemini ----
    MetaRow {
        pat: "gemini-2.0-flash",
        modalities: &["text", "vision", "audio"],
        capabilities: &["function-calling", "json-mode", "streaming", "long-context"],
        context_window: Some(1_048_576),
        compliance_restrictions: &[],
    },
    MetaRow {
        pat: "gemini-1.5-pro",
        modalities: &["text", "vision", "audio"],
        capabilities: &["function-calling", "json-mode", "streaming", "long-context"],
        context_window: Some(2_097_152),
        compliance_restrictions: &[],
    },
    MetaRow {
        pat: "gemini-1.5-flash",
        modalities: &["text", "vision", "audio"],
        capabilities: &["function-calling", "json-mode", "streaming", "long-context"],
        context_window: Some(1_048_576),
        compliance_restrictions: &[],
    },
    MetaRow {
        pat: "gemini",
        modalities: &["text", "vision"],
        capabilities: &["function-calling", "json-mode", "streaming", "long-context"],
        context_window: Some(1_048_576),
        compliance_restrictions: &[],
    },
    // ---- Mistral ----
    MetaRow {
        pat: "mistral-embed",
        modalities: &["embeddings"],
        capabilities: &[],
        context_window: Some(8_192),
        compliance_restrictions: &[],
    },
    MetaRow {
        pat: "mistral-large",
        modalities: &["text"],
        capabilities: &["function-calling", "json-mode", "streaming", "long-context"],
        context_window: Some(128_000),
        compliance_restrictions: &[],
    },
    MetaRow {
        pat: "mistral-small",
        modalities: &["text"],
        capabilities: &["function-calling", "json-mode", "streaming", "long-context"],
        context_window: Some(128_000),
        compliance_restrictions: &[],
    },
    MetaRow {
        pat: "open-mistral-nemo",
        modalities: &["text"],
        capabilities: &["function-calling", "streaming", "long-context"],
        context_window: Some(128_000),
        compliance_restrictions: &[],
    },
    // ---- Cohere ----
    MetaRow {
        pat: "rerank",
        modalities: &["text"],
        capabilities: &[],
        context_window: None,
        compliance_restrictions: &[],
    },
    MetaRow {
        pat: "embed",
        modalities: &["embeddings"],
        capabilities: &[],
        context_window: None,
        compliance_restrictions: &[],
    },
    MetaRow {
        pat: "command-r-plus",
        modalities: &["text"],
        capabilities: &["function-calling", "streaming", "long-context"],
        context_window: Some(128_000),
        compliance_restrictions: &[],
    },
    MetaRow {
        pat: "command-r",
        modalities: &["text"],
        capabilities: &["function-calling", "streaming", "long-context"],
        context_window: Some(128_000),
        compliance_restrictions: &[],
    },
    // ---- Bedrock (vendor-qualified ids) ----
    MetaRow {
        pat: "amazon.titan-text",
        modalities: &["text"],
        capabilities: &["streaming"],
        context_window: Some(8_192),
        compliance_restrictions: &[],
    },
    MetaRow {
        pat: "meta.llama3",
        modalities: &["text"],
        capabilities: &["streaming", "long-context"],
        context_window: Some(8_192),
        compliance_restrictions: &[],
    },
    MetaRow {
        pat: "anthropic.claude",
        modalities: &["text", "vision"],
        capabilities: &["function-calling", "streaming", "long-context"],
        context_window: Some(200_000),
        compliance_restrictions: &[],
    },
    // ---- Groq ----
    MetaRow {
        pat: "whisper-large",
        modalities: &["audio"],
        capabilities: &[],
        context_window: None,
        compliance_restrictions: &[],
    },
    MetaRow {
        pat: "llama-3.3-70b-versatile",
        modalities: &["text"],
        capabilities: &["function-calling", "streaming", "long-context"],
        context_window: Some(128_000),
        compliance_restrictions: &[],
    },
    MetaRow {
        pat: "llama-3.1-8b-instant",
        modalities: &["text"],
        capabilities: &["function-calling", "streaming", "long-context"],
        context_window: Some(128_000),
        compliance_restrictions: &[],
    },
    MetaRow {
        pat: "gemma2-9b-it",
        modalities: &["text"],
        capabilities: &["streaming"],
        context_window: Some(8_192),
        compliance_restrictions: &[],
    },
    MetaRow {
        pat: "qwen",
        modalities: &["text"],
        capabilities: &["function-calling", "streaming", "long-context"],
        context_window: Some(131_072),
        compliance_restrictions: &[],
    },
    MetaRow {
        pat: "deepseek-r1-distill",
        modalities: &["text"],
        capabilities: &["streaming", "long-context"],
        context_window: Some(128_000),
        compliance_restrictions: &[],
    },
    MetaRow {
        pat: "moonshotai/kimi-k2",
        modalities: &["text"],
        capabilities: &["function-calling", "streaming", "long-context"],
        context_window: Some(128_000),
        compliance_restrictions: &[],
    },
    // ---- DeepSeek (native, China-hosted) ----
    // ADR-035 §4 seed: the DeepSeek FIRST-PARTY API is hosted in China with no
    // data-residency guarantee and no BAA. We exclude it under data-sovereignty
    // frameworks (DPDP/RBI — regulated Indian data must not egress) and HIPAA (no
    // Business Associate Agreement → not acceptable for PHI). This is a HOSTING-
    // JURISDICTION judgement, not a model-weights one: the same open weights served
    // US-side (Together `DeepSeek-V3`, Groq `deepseek-r1-distill`) carry NO
    // restriction below — the distinction is defensible and auditable.
    MetaRow {
        pat: "deepseek-reasoner",
        modalities: &["text"],
        capabilities: &["streaming", "long-context"],
        context_window: Some(64_000),
        compliance_restrictions: &["DPDP", "RBI", "HIPAA"],
    },
    MetaRow {
        pat: "deepseek-v4",
        modalities: &["text"],
        capabilities: &["function-calling", "json-mode", "streaming", "long-context"],
        context_window: Some(128_000),
        compliance_restrictions: &["DPDP", "RBI", "HIPAA"],
    },
    MetaRow {
        pat: "deepseek-chat",
        modalities: &["text"],
        capabilities: &["function-calling", "json-mode", "streaming", "long-context"],
        context_window: Some(64_000),
        compliance_restrictions: &["DPDP", "RBI", "HIPAA"],
    },
    // ---- Together (vendor-namespaced) ----
    MetaRow {
        pat: "bge-large",
        modalities: &["embeddings"],
        capabilities: &[],
        context_window: Some(512),
        compliance_restrictions: &[],
    },
    MetaRow {
        pat: "Llama-3.3-70B",
        modalities: &["text"],
        capabilities: &["function-calling", "streaming", "long-context"],
        context_window: Some(128_000),
        compliance_restrictions: &[],
    },
    MetaRow {
        pat: "Qwen2.5-72B",
        modalities: &["text"],
        capabilities: &["function-calling", "streaming", "long-context"],
        context_window: Some(32_768),
        compliance_restrictions: &[],
    },
    MetaRow {
        pat: "DeepSeek-V3",
        modalities: &["text"],
        capabilities: &["function-calling", "streaming", "long-context"],
        context_window: Some(128_000),
        compliance_restrictions: &[],
    },
    MetaRow {
        pat: "Mixtral-8x7B",
        modalities: &["text"],
        capabilities: &["function-calling", "streaming", "long-context"],
        context_window: Some(32_768),
        compliance_restrictions: &[],
    },
    // ---- xAI (Grok) ----
    // ADR-035 §4 seed: xAI's API offers no Business Associate Agreement, so the
    // Grok models are excluded under HIPAA (not acceptable for PHI). They carry no
    // data-sovereignty restriction (US-hosted), so DPDP/RBI tenants are unaffected.
    MetaRow {
        pat: "grok-4",
        modalities: &["text", "vision"],
        capabilities: &["function-calling", "json-mode", "streaming", "long-context"],
        context_window: Some(256_000),
        compliance_restrictions: &["HIPAA"],
    },
    MetaRow {
        pat: "grok-3",
        modalities: &["text"],
        capabilities: &["function-calling", "json-mode", "streaming", "long-context"],
        context_window: Some(131_072),
        compliance_restrictions: &["HIPAA"],
    },
];

/// Normalize a per-1k input cost (micro-USD) into a deterministic 0..1 band for
/// the smart-router cost signal ([ADR-035] §1 `normalized_cost_param`, ADR-033).
///
/// Log-scale over a fixed, documented anchor range so cheap and premium models
/// spread sensibly (linear would crush everything below ~$1/1k into ~0):
///
/// * floor   = 100 micro-USD / 1k  (≈ gemini-1.5-flash input, $0.075/M)  → 0.0
/// * ceiling = 15_000 micro-USD / 1k (≈ o1 / claude-3-opus input, $15/M) → 1.0
///
/// A value below the floor clamps to 0.0; above the ceiling clamps to 1.0. Pure
/// function of the input rate — stable across calls (no clock/RNG), so the golden
/// guards stay deterministic. Only the INPUT rate drives the band (output tracks
/// input closely enough; a single scalar is what the router consumes).
fn normalized_cost(input_per_1k_micro_usd: u64) -> f32 {
    const FLOOR: f64 = 100.0;
    const CEIL: f64 = 15_000.0;
    let v = (input_per_1k_micro_usd as f64).clamp(FLOOR, CEIL);
    let n = (v.ln() - FLOOR.ln()) / (CEIL.ln() - FLOOR.ln());
    (n as f32).clamp(0.0, 1.0)
}

/// Build the `cost` block for a model from the `limits` `PRICE_BOOK` (single source
/// of truth — no duplicated rates). Per-million rates are converted to per-1k
/// (`per_1k = per_million / 1000`). `None` when the price book does not cover the
/// model (we do not invent economics — ADR-035 source discipline).
fn cost_meta(id: &str) -> Option<CostMeta> {
    let (prompt_per_million, completion_per_million) = routeplane_limits::price_for(id)?;
    let input_per_1k = prompt_per_million / 1_000;
    let output_per_1k = completion_per_million / 1_000;
    Some(CostMeta {
        input_per_1k_micro_usd: input_per_1k,
        output_per_1k_micro_usd: output_per_1k,
        normalized_cost_param: normalized_cost(input_per_1k),
        source: "routeplane_price_book",
    })
}

/// Assemble the `routeplane` extension for a model id: cost from the `PRICE_BOOK`,
/// modalities/capabilities/context_window from `META_TABLE` (most-specific-first).
/// Returns `None` when nothing is known (so the extension is omitted and the entry
/// stays byte-identical to the bare OpenAI shape).
fn enrich(id: &str) -> Option<ModelMetadata> {
    let cost = cost_meta(id);
    let row = META_TABLE.iter().find(|r| id.contains(r.pat));
    let (modalities, capabilities, context_window, compliance_restrictions) = match row {
        Some(r) => (
            r.modalities.to_vec(),
            r.capabilities.to_vec(),
            r.context_window,
            r.compliance_restrictions.to_vec(),
        ),
        None => (Vec::new(), Vec::new(), None, Vec::new()),
    };
    let meta = ModelMetadata {
        cost,
        modalities,
        capabilities,
        context_window,
        compliance_restrictions,
    };
    if meta.is_empty() {
        None
    } else {
        Some(meta)
    }
}

/// The compliance frameworks that do NOT accept `model` ([ADR-035] §4 — the
/// org-gate exclusion set), resolved by the SAME most-specific-first substring
/// match as catalog enrichment. Returns a borrow of static data — no allocation,
/// no DB, no lock (a single `META_TABLE` scan). The empty slice is the common
/// case (no defensible restriction), so a tenant with no frameworks pays only an
/// `is_empty()` check on the caller side and never reaches this scan.
///
/// Framework codes are the §5 registry identifiers (config strings, never user
/// content), so the proxy may cite them verbatim in the `403
/// model_compliance_excluded` message without violating no-reflection.
pub fn compliance_restrictions_for(model: &str) -> &'static [&'static str] {
    match META_TABLE.iter().find(|r| model.contains(r.pat)) {
        Some(r) => r.compliance_restrictions,
        None => &[],
    }
}

/// The `{"object":"list","data":[…]}` envelope for `GET /v1/models`.
#[derive(Debug, Serialize)]
pub struct ModelList {
    pub object: &'static str,
    pub data: Vec<ModelObject>,
}

/// A stable, fixed `created` timestamp for catalog entries.
///
/// OpenAI returns a per-model creation timestamp that does not change between
/// calls; we mirror that stability with a single fixed constant rather than
/// `now()` (which would break determinism and the A/B golden/parity guards).
/// 2024-01-01T00:00:00Z.
const CATALOG_CREATED: u64 = 1_704_067_200;

/// Compact internal catalog row: `(model_id, owned_by_provider)`.
///
/// To extend the catalog: add a row here under the right provider, using the
/// exact model id that provider's adapter accepts in `model`. Keep `owned_by`
/// equal to the provider's registry key (the `x-routeplane-provider` name) so a
/// caller can route directly to it. New providers added to
/// `build_provider_registry()` should get their well-known ids listed here.
const CATALOG: &[(&str, &str)] = &[
    // ---- openai ----
    ("gpt-4o", "openai"),
    ("gpt-4o-mini", "openai"),
    ("gpt-4-turbo", "openai"),
    ("gpt-4", "openai"),
    ("gpt-3.5-turbo", "openai"),
    ("o1", "openai"),
    ("o1-mini", "openai"),
    ("text-embedding-3-small", "openai"),
    ("text-embedding-3-large", "openai"),
    ("text-embedding-ada-002", "openai"),
    // OpenAI moderation (parity with OpenAI's /v1/moderations; routed via
    // /v1/moderations). The built-in `local` moderator has no catalog id (it is
    // a reserved provider sentinel, not a model).
    ("omni-moderation-latest", "openai"),
    ("text-moderation-latest", "openai"),
    // OpenAI image generation (parity with OpenAI's /v1/images/generations;
    // routed via /v1/images/generations).
    ("gpt-image-1", "openai"),
    ("dall-e-3", "openai"),
    // OpenAI audio transcription / speech-to-text (parity with OpenAI's
    // /v1/audio/transcriptions; routed via /v1/audio/transcriptions).
    ("whisper-1", "openai"),
    ("gpt-4o-transcribe", "openai"),
    // OpenAI text-to-speech / TTS (parity with OpenAI's /v1/audio/speech; routed
    // via /v1/audio/speech). Completes the audio pair with transcription.
    ("tts-1", "openai"),
    ("tts-1-hd", "openai"),
    ("gpt-4o-mini-tts", "openai"),
    // ---- anthropic ----
    ("claude-3-5-sonnet-latest", "anthropic"),
    ("claude-3-5-haiku-latest", "anthropic"),
    ("claude-3-opus-latest", "anthropic"),
    ("claude-3-haiku-20240307", "anthropic"),
    // ---- gemini ----
    ("gemini-2.0-flash", "gemini"),
    ("gemini-1.5-pro", "gemini"),
    ("gemini-1.5-flash", "gemini"),
    // ---- mistral ----
    ("mistral-large-latest", "mistral"),
    ("mistral-small-latest", "mistral"),
    ("open-mistral-nemo", "mistral"),
    ("mistral-embed", "mistral"),
    // ---- cohere ----
    ("command-r-plus", "cohere"),
    ("command-r", "cohere"),
    ("embed-v4.0", "cohere"),
    ("embed-english-v3.0", "cohere"),
    ("embed-multilingual-v3.0", "cohere"),
    // Cohere rerank (parity with LiteLLM's /rerank; routed via /v1/rerank).
    ("rerank-v3.5", "cohere"),
    ("rerank-v4.0-pro", "cohere"),
    // ---- bedrock ---- (Bedrock model ids are vendor-qualified)
    ("anthropic.claude-3-5-sonnet-20240620-v1:0", "bedrock"),
    ("anthropic.claude-3-haiku-20240307-v1:0", "bedrock"),
    ("amazon.titan-text-express-v1", "bedrock"),
    ("meta.llama3-70b-instruct-v1:0", "bedrock"),
    // Groq — OpenAI-compatible, ultra-low-latency open-weight inference.
    ("llama-3.3-70b-versatile", "groq"),
    ("llama-3.1-8b-instant", "groq"),
    ("gemma2-9b-it", "groq"),
    ("qwen/qwen3-32b", "groq"),
    ("deepseek-r1-distill-llama-70b", "groq"),
    ("moonshotai/kimi-k2-instruct", "groq"),
    // Groq Whisper — flagship fast/cheap speech-to-text (routed via
    // /v1/audio/transcriptions).
    ("whisper-large-v3", "groq"),
    ("whisper-large-v3-turbo", "groq"),
    // DeepSeek — OpenAI-compatible chat + reasoning models. Current (2026):
    // deepseek-v4-flash / -pro. Legacy still-valid (deprecate 2026-07-24):
    // deepseek-chat / deepseek-reasoner.
    ("deepseek-v4-flash", "deepseek"),
    ("deepseek-v4-pro", "deepseek"),
    ("deepseek-chat", "deepseek"),
    ("deepseek-reasoner", "deepseek"),
    // Together AI — OpenAI-compatible chat + streaming + embeddings over ~100+
    // namespaced open-weight models. Chat ids are vendor-namespaced; the embed id
    // is routed via /v1/embeddings (owned_by: together).
    ("meta-llama/Llama-3.3-70B-Instruct-Turbo", "together"),
    ("Qwen/Qwen2.5-72B-Instruct-Turbo", "together"),
    ("deepseek-ai/DeepSeek-V3", "together"),
    ("mistralai/Mixtral-8x7B-Instruct-v0.1", "together"),
    ("BAAI/bge-large-en-v1.5", "together"),
    // Fireworks AI — OpenAI-compatible chat + streaming + embeddings over
    // namespaced open-weight models. Chat ids are vendor-namespaced
    // (`accounts/fireworks/models/<name>`); the embed id is routed via
    // /v1/embeddings (owned_by: fireworks).
    (
        "accounts/fireworks/models/llama-v3p1-70b-instruct",
        "fireworks",
    ),
    (
        "accounts/fireworks/models/qwen2p5-72b-instruct",
        "fireworks",
    ),
    ("accounts/fireworks/models/deepseek-v3", "fireworks"),
    (
        "accounts/fireworks/models/mixtral-8x22b-instruct",
        "fireworks",
    ),
    ("nomic-ai/nomic-embed-text-v1.5", "fireworks"),
    // xAI (Grok) — OpenAI-compatible chat + reasoning models. Current (2026):
    // grok-4.3 / grok-4-0709 / grok-3 / grok-3-fast.
    ("grok-4.3", "xai"),
    ("grok-4-0709", "xai"),
    ("grok-3", "xai"),
    ("grok-3-fast", "xai"),
    // OpenRouter — OpenAI-compatible meta-aggregator. Model ids are
    // `provider/model` form and pass through verbatim (owned_by: openrouter).
    // Unknown/unlisted ids still work — the adapter forwards whatever the client
    // sends; these are just the well-known catalog entries.
    ("openai/gpt-4o", "openrouter"),
    ("anthropic/claude-sonnet-4", "openrouter"),
    ("meta-llama/llama-3.3-70b-instruct", "openrouter"),
    ("deepseek/deepseek-chat", "openrouter"),
];

/// The static, always-available catalog (the registered providers' well-known
/// model ids). Built once on first access; subsequent reads are a cheap slice
/// borrow. Env-discoverable deployments (azure_openai, self_hosted) are folded
/// in at request time on top of this — they are NOT baked here because they
/// depend on process env that may differ per deployment.
static STATIC_CATALOG: LazyLock<Vec<ModelObject>> = LazyLock::new(|| {
    CATALOG
        .iter()
        .map(|(id, owned_by)| ModelObject::new(*id, *owned_by, CATALOG_CREATED))
        .collect()
});

/// Fold env-discoverable deployments (azure_openai deployment name, self_hosted
/// configured model) onto the static catalog. Returns the full per-request
/// catalog. Pure reads of process env; no panic, no network.
///
/// * `azure_openai`: `AZURE_OPENAI_DEPLOYMENT` is the deployment name a caller
///   addresses (Azure deployments are named, not fixed model ids), so when set
///   we surface it as an `owned_by: azure_openai` entry.
/// * `self_hosted`: there is no single canonical model id (the endpoint serves
///   whatever it was started with); if `SELF_HOSTED_MODEL` is set we surface it,
///   otherwise self_hosted contributes nothing (documented follow-up: a live
///   `/v1/models` passthrough to the upstream once true native discovery lands).
fn discovered_models() -> Vec<ModelObject> {
    let mut extra = Vec::new();
    if let Ok(dep) = std::env::var("AZURE_OPENAI_DEPLOYMENT") {
        let dep = dep.trim();
        if !dep.is_empty() {
            extra.push(ModelObject::new(dep, "azure_openai", CATALOG_CREATED));
        }
    }
    if let Ok(model) = std::env::var("SELF_HOSTED_MODEL") {
        let model = model.trim();
        if !model.is_empty() {
            extra.push(ModelObject::new(model, "self_hosted", CATALOG_CREATED));
        }
    }
    extra
}

/// Build the full, de-duplicated catalog for this request: the static set plus
/// any env-discovered deployments. De-dup is by `id` (an env deployment that
/// shadows a static id keeps the first/static entry — stable, deterministic).
fn full_catalog() -> Vec<ModelObject> {
    let mut out: Vec<ModelObject> = STATIC_CATALOG.clone();
    for m in discovered_models() {
        if !out.iter().any(|e| e.id == m.id) {
            out.push(m);
        }
    }
    out
}

/// ADR-086: the operator-defined **combos** as OpenAI model objects, so a stock
/// client can discover them via `GET /v1/models` and address them via the `model`
/// field. Each combo is a `combo:`-namespaced entry in the routing-policy registry
/// (`owned_by = "routeplane"` — the gateway owns the chain). Additive: with no
/// combos configured this yields an empty vec, so the catalog is byte-identical.
fn combo_models(state: &AppState) -> Vec<ModelObject> {
    let snapshot = state.policies.load();
    routeplane_policy::combo_names(&snapshot)
        .map(|name| ModelObject {
            id: name.to_string(),
            object: "model",
            created: CATALOG_CREATED,
            owned_by: "routeplane".to_string(),
            routeplane: None,
        })
        .collect()
}

/// Is `id` a model in the built-in static catalog? Drives the documented
/// custom-provider precedence rule: a runtime custom provider's model mapping
/// never shadows a built-in catalog id (the custom provider stays reachable via
/// an explicit `x-routeplane-provider`). A short static-slice scan, no lock.
pub fn is_builtin_model(id: &str) -> bool {
    STATIC_CATALOG.iter().any(|m| m.id == id)
}

/// Runtime custom-provider models (`owned_by = <provider name>`), folded onto
/// the catalog at request time — the same posture as the env-discovered
/// deployments. ADDITIVE + deduplicated by id against everything already in
/// `existing` (built-in catalog + combos win a contested id, matching the
/// routing precedence). Empty registry ⇒ empty vec ⇒ byte-identical list.
fn custom_provider_models(state: &AppState, existing: &[ModelObject]) -> Vec<ModelObject> {
    state
        .custom_providers
        .model_entries()
        .into_iter()
        .filter(|(id, _)| !existing.iter().any(|m| &m.id == id))
        .map(|(id, owner)| ModelObject {
            id,
            object: "model",
            created: CATALOG_CREATED,
            owned_by: owner,
            routeplane: None,
        })
        .collect()
}

/// `GET /v1/models` — the full catalog as the OpenAI `{"object":"list", …}`
/// envelope. Authed (the route rides the same auth layer as chat). 200 always.
/// The static provider catalog plus any operator-defined combos (ADR-086) plus
/// any runtime custom-provider models.
pub async fn list_models(State(state): State<Arc<AppState>>) -> Response {
    let mut data = full_catalog();
    data.extend(combo_models(&state));
    let custom = custom_provider_models(&state, &data);
    data.extend(custom);
    let list = ModelList {
        object: "list",
        data,
    };
    (StatusCode::OK, Json(list)).into_response()
}

/// `GET /v1/models/{id}` — the single model object (base model OR combo), or a 404
/// OpenAI error envelope for an unknown id. Authed.
pub async fn retrieve_model(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Response {
    if let Some(model) = full_catalog().into_iter().find(|m| m.id == id) {
        return (StatusCode::OK, Json(model)).into_response();
    }
    if let Some(model) = combo_models(&state).into_iter().find(|m| m.id == id) {
        return (StatusCode::OK, Json(model)).into_response();
    }
    // Runtime custom-provider model (additive; built-ins/combos matched above).
    if let Some((model_id, owner)) = state
        .custom_providers
        .model_entries()
        .into_iter()
        .find(|(m, _)| m == &id)
    {
        return (
            StatusCode::OK,
            Json(ModelObject {
                id: model_id,
                object: "model",
                created: CATALOG_CREATED,
                owned_by: owner,
                routeplane: None,
            }),
        )
            .into_response();
    }
    crate::api_error::error_response(
        StatusCode::NOT_FOUND,
        "model_not_found",
        format!("The model '{id}' does not exist or is not available through this gateway."),
        "invalid_request_error",
        Some("model"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_catalog_covers_every_registered_provider() {
        // Each provider registered in build_provider_registry() that has a
        // well-known fixed model id must contribute at least one catalog entry.
        // (azure_openai/self_hosted are env-discovered, so not asserted here.)
        for provider in [
            "openai",
            "anthropic",
            "gemini",
            "mistral",
            "cohere",
            "bedrock",
            "groq",
            "deepseek",
            "together",
            "fireworks",
            "xai",
        ] {
            assert!(
                STATIC_CATALOG.iter().any(|m| m.owned_by == provider),
                "catalog missing any model for provider {provider}"
            );
        }
    }

    #[test]
    fn every_entry_has_the_openai_model_shape() {
        for m in STATIC_CATALOG.iter() {
            assert_eq!(m.object, "model");
            assert!(!m.id.is_empty());
            assert!(!m.owned_by.is_empty());
            assert_eq!(m.created, CATALOG_CREATED);
        }
    }

    #[test]
    fn cohere_embed_models_are_listed() {
        // Cohere has a first-party embeddings API (parity with LiteLLM); its
        // embed model ids must be discoverable via /v1/models so OpenAI-SDK
        // clients can enumerate them and route to owned_by: cohere.
        for id in [
            "embed-v4.0",
            "embed-english-v3.0",
            "embed-multilingual-v3.0",
        ] {
            assert!(
                STATIC_CATALOG
                    .iter()
                    .any(|m| m.id == id && m.owned_by == "cohere"),
                "catalog missing cohere embed model {id}"
            );
        }
    }

    #[test]
    fn cohere_rerank_models_are_listed() {
        // Cohere has a first-party rerank API (parity with LiteLLM's /rerank);
        // its rerank model ids must be discoverable via /v1/models so clients
        // can enumerate them and route to owned_by: cohere via /v1/rerank.
        for id in ["rerank-v3.5", "rerank-v4.0-pro"] {
            assert!(
                STATIC_CATALOG
                    .iter()
                    .any(|m| m.id == id && m.owned_by == "cohere"),
                "catalog missing cohere rerank model {id}"
            );
        }
    }

    #[test]
    fn openai_moderation_model_is_listed() {
        // OpenAI's moderation models must be discoverable via /v1/models so
        // clients can enumerate them and route to owned_by: openai via
        // /v1/moderations (parity with OpenAI's /v1/moderations).
        assert!(
            STATIC_CATALOG
                .iter()
                .any(|m| m.id == "omni-moderation-latest" && m.owned_by == "openai"),
            "catalog missing openai moderation model omni-moderation-latest"
        );
    }

    #[test]
    fn openai_image_models_are_listed() {
        // OpenAI's image models must be discoverable via /v1/models so clients
        // can enumerate them and route to owned_by: openai via
        // /v1/images/generations (parity with OpenAI's /v1/images/generations).
        for id in ["gpt-image-1", "dall-e-3"] {
            assert!(
                STATIC_CATALOG
                    .iter()
                    .any(|m| m.id == id && m.owned_by == "openai"),
                "catalog missing openai image model {id}"
            );
        }
    }

    #[test]
    fn openai_tts_models_are_listed() {
        // OpenAI's TTS models must be discoverable via /v1/models so clients can
        // enumerate them and route to owned_by: openai via /v1/audio/speech
        // (parity with OpenAI's /v1/audio/speech).
        for id in ["tts-1", "gpt-4o-mini-tts"] {
            assert!(
                STATIC_CATALOG
                    .iter()
                    .any(|m| m.id == id && m.owned_by == "openai"),
                "catalog missing openai tts model {id}"
            );
        }
    }

    #[test]
    fn catalog_ids_are_unique() {
        let mut ids: Vec<&str> = CATALOG.iter().map(|(id, _)| *id).collect();
        ids.sort_unstable();
        let before = ids.len();
        ids.dedup();
        assert_eq!(before, ids.len(), "duplicate model id in CATALOG");
    }

    // ---- ADR-035 catalog enrichment (ATTRIBUTE half) ----------------------

    fn find<'a>(id: &str) -> &'a ModelObject {
        STATIC_CATALOG
            .iter()
            .find(|m| m.id == id)
            .unwrap_or_else(|| panic!("catalog missing {id}"))
    }

    #[test]
    fn openai_core_fields_are_exact_and_unchanged() {
        // SDK-compat: the four OpenAI core fields MUST stay exactly as-is. The
        // enrichment is purely additive under the `routeplane` extension.
        let m = find("gpt-4o");
        assert_eq!(m.id, "gpt-4o");
        assert_eq!(m.object, "model");
        assert_eq!(m.created, CATALOG_CREATED);
        assert_eq!(m.owned_by, "openai");
    }

    #[test]
    fn core_fields_serialize_with_exact_openai_keys() {
        // The serialized object keeps the OpenAI keys verbatim and adds only the
        // reserved `routeplane` extension (which OpenAI SDKs ignore). A bare entry
        // (no metadata) omits the extension entirely → byte-identical legacy shape.
        let m = ModelObject::new("totally-unknown-model-xyz", "openai", CATALOG_CREATED);
        assert!(
            m.routeplane.is_none(),
            "unknown model must carry no metadata"
        );
        let v = serde_json::to_value(&m).expect("serialize");
        let obj = v.as_object().expect("object");
        let mut keys: Vec<&str> = obj.keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(keys, vec!["created", "id", "object", "owned_by"]);
    }

    #[test]
    fn cost_per_1k_math_is_correct_from_price_book() {
        // gpt-4o is pinned in PRICE_BOOK at 3_000_000 / 10_000_000 micro-USD per
        // MILLION tokens. per_1k = per_million / 1000 ⇒ 3_000 / 10_000.
        let cost = find("gpt-4o")
            .routeplane
            .as_ref()
            .and_then(|m| m.cost.as_ref())
            .expect("gpt-4o has cost");
        assert_eq!(cost.input_per_1k_micro_usd, 3_000);
        assert_eq!(cost.output_per_1k_micro_usd, 10_000);
        assert_eq!(cost.source, "routeplane_price_book");
        // gpt-4o-mini: 150_000 / 600_000 per million ⇒ 150 / 600 per 1k.
        let mini = find("gpt-4o-mini")
            .routeplane
            .as_ref()
            .and_then(|m| m.cost.as_ref())
            .expect("gpt-4o-mini has cost");
        assert_eq!(mini.input_per_1k_micro_usd, 150);
        assert_eq!(mini.output_per_1k_micro_usd, 600);
    }

    #[test]
    fn normalized_cost_band_orders_cheap_below_expensive() {
        // Deterministic 0..1 band: a cheap flash model sits well below a premium
        // reasoning model, and both are clamped within [0,1].
        let flash = normalized_cost(75); // gemini-1.5-flash input ($0.075/M → 75/1k)
        let opus = normalized_cost(15_000); // claude-3-opus / o1 input ($15/M → 15_000/1k)
        assert!((0.0..=1.0).contains(&flash));
        assert!((0.0..=1.0).contains(&opus));
        assert!(flash < opus, "cheap model must normalize below premium");
        assert!(opus >= 0.99, "ceiling input clamps to ~1.0");
        assert!(flash <= 0.01, "floor input clamps to ~0.0");
    }

    #[test]
    fn gpt_4o_carries_vision_and_tools_and_context() {
        let meta = find("gpt-4o").routeplane.as_ref().expect("gpt-4o enriched");
        assert!(meta.modalities.contains(&"text"));
        assert!(meta.modalities.contains(&"vision"));
        assert!(meta.capabilities.contains(&"function-calling"));
        assert!(meta.capabilities.contains(&"json-mode"));
        assert!(meta.capabilities.contains(&"streaming"));
        assert_eq!(meta.context_window, Some(128_000));
    }

    #[test]
    fn claude_carries_vision_and_200k_context() {
        let meta = find("claude-3-opus-latest")
            .routeplane
            .as_ref()
            .expect("claude enriched");
        assert!(meta.modalities.contains(&"vision"));
        assert_eq!(meta.context_window, Some(200_000));
    }

    #[test]
    fn embeddings_model_has_embeddings_modality_only() {
        let meta = find("text-embedding-3-small")
            .routeplane
            .as_ref()
            .expect("embedding enriched");
        assert_eq!(meta.modalities, vec!["embeddings"]);
        // Embeddings are not chat: no function-calling/json-mode.
        assert!(meta.capabilities.is_empty());
    }

    #[test]
    fn rerank_model_is_text_modality() {
        let meta = find("rerank-v3.5")
            .routeplane
            .as_ref()
            .expect("rerank enriched");
        assert_eq!(meta.modalities, vec!["text"]);
    }

    #[test]
    fn whisper_and_tts_are_audio_modality() {
        for id in ["whisper-1", "tts-1", "gpt-4o-mini-tts"] {
            let meta = find(id).routeplane.as_ref().expect("audio enriched");
            assert_eq!(meta.modalities, vec!["audio"], "{id} should be audio");
        }
    }

    #[test]
    fn gemini_has_vision_and_long_context() {
        let meta = find("gemini-1.5-pro")
            .routeplane
            .as_ref()
            .expect("gemini enriched");
        assert!(meta.modalities.contains(&"vision"));
        assert!(meta.capabilities.contains(&"long-context"));
        assert!(meta.context_window.is_some_and(|c| c >= 1_000_000));
    }

    #[test]
    fn most_specific_first_resolution_for_meta() {
        // gpt-4o-mini must NOT inherit gpt-4o's row by accident — both happen to
        // share modalities/context here, but the audio sub-ids must win: the
        // gpt-4o-mini-tts row (audio) precedes the gpt-4o-mini chat row.
        let tts = find("gpt-4o-mini-tts")
            .routeplane
            .as_ref()
            .expect("tts enriched");
        assert_eq!(tts.modalities, vec!["audio"]);
    }

    #[test]
    fn unknown_model_omits_the_routeplane_extension() {
        let m = ModelObject::new("some-future-model-v9", "openai", CATALOG_CREATED);
        assert!(m.routeplane.is_none());
    }

    // ---- ADR-035 §4 compliance-restriction substrate ----------------------

    #[test]
    fn china_hosted_deepseek_is_restricted_under_sovereignty_and_hipaa() {
        // The DeepSeek native (China-hosted) API is excluded under DPDP/RBI
        // (data-sovereignty) and HIPAA (no BAA). The accessor and the catalog
        // extension must agree.
        for id in ["deepseek-chat", "deepseek-reasoner", "deepseek-v4-pro"] {
            let r = compliance_restrictions_for(id);
            assert!(r.contains(&"DPDP"), "{id} should be DPDP-restricted");
            assert!(r.contains(&"RBI"), "{id} should be RBI-restricted");
            assert!(r.contains(&"HIPAA"), "{id} should be HIPAA-restricted");
        }
        let meta = find("deepseek-chat")
            .routeplane
            .as_ref()
            .expect("deepseek enriched");
        assert!(meta.compliance_restrictions.contains(&"DPDP"));
    }

    #[test]
    fn grok_is_hipaa_only_restricted_not_sovereignty() {
        // xAI offers no BAA → HIPAA-excluded; but US-hosted → DPDP/RBI tenants
        // are unaffected (the defensible jurisdiction-vs-BAA split).
        let r = compliance_restrictions_for("grok-4.3");
        assert_eq!(r, &["HIPAA"]);
        assert!(!r.contains(&"DPDP"));
    }

    #[test]
    fn us_hosted_open_weight_deepseek_is_unrestricted() {
        // Same model weights, US hosting (Together / Groq) ⇒ NO restriction.
        assert!(compliance_restrictions_for("deepseek-ai/DeepSeek-V3").is_empty());
        assert!(compliance_restrictions_for("deepseek-r1-distill-llama-70b").is_empty());
    }

    #[test]
    fn unrestricted_models_return_empty_slice() {
        // The common case: a mainstream model carries no exclusion, and an
        // unknown id matches no row (also empty).
        assert!(compliance_restrictions_for("gpt-4o").is_empty());
        assert!(compliance_restrictions_for("claude-3-5-sonnet-latest").is_empty());
        assert!(compliance_restrictions_for("totally-unknown-xyz").is_empty());
    }

    #[test]
    fn restriction_uses_most_specific_first_match() {
        // `compliance_restrictions_for` shares the META_TABLE scan, so the same
        // most-specific-first discipline holds (deepseek-r1-distill — US/Groq —
        // precedes nothing China-side here, and stays empty).
        assert!(compliance_restrictions_for("deepseek-r1-distill-llama-70b").is_empty());
        assert!(!compliance_restrictions_for("deepseek-reasoner").is_empty());
    }
}
