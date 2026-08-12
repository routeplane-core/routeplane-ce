//! Rung-0 exact-match response cache (G2.5 — [PRD-007] behavior, [ADR-022]
//! mechanics). In-memory, per-replica, zero standing cost, zero hot-path locks.
//!
//! - **Read path (hot)**: one SHA-256 over the shaping-resolved request (done by
//!   the caller via [`exact_key`]) + one `ArcSwap::load` + one `HashMap` probe +
//!   a TTL check. No lock, no CAS contention.
//! - **Write path (off-path)**: [`ExactCache::insert`] is a bounded tenant-lane
//!   `try_send` to a single dedicated writer thread; ready-tenant fair service
//!   prevents one tenant from starving another. The writer clones the
//!   shard map, inserts, evicts (tenant-local TTL-first, then FIFO) and publishes
//!   the new `Arc`. Readers never wait; a full lane drops the write (counted).
//! - **Isolation is structural** ([PRD-007] FR-7): `tenant_id` and `namespace`
//!   are fields of [`CacheKey`] participating in `Eq`/`Hash` — not string
//!   prefixes — so a cross-tenant hit is impossible by construction.
//!
//! This crate is deliberately runtime-free (no tokio) and network-free; the
//! injectable [`Clock`] mirrors the `router` crate's deterministic-test doctrine.
//!
//! [PRD-007]: ../../../docs/product/prd/007-caching.md
//! [ADR-022]: ../../../docs/adr/022-cache-architecture.md

pub mod idempotency;

use arc_swap::ArcSwap;
use bytes::Bytes;
use routeplane_types::{ChatCompletionRequest, Message, TenantId, Tool};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TryRecvError, TrySendError};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{SystemTime, UNIX_EPOCH};

/// Shard count (ADR-022 §2). Shard chosen by the top 6 bits of the key hash.
pub const SHARD_COUNT: usize = 64;
/// Per-entry hard cap (PRD-007 FR-8: platform cap 256 KiB). The config-level
/// `max_response_bytes` may only lower this; the core enforces it again as
/// defense in depth.
pub const MAX_ENTRY_BYTES: usize = 256 * 1024;
/// Default per-replica byte budget (ADR-022 §3, pool-std default). The actual
/// value is a cell tfvars parameter delivered as an env var (see the binary's
/// `CacheSettings`).
pub const DEFAULT_BUDGET_BYTES: usize = 64 * 1024 * 1024;
/// Approximate fixed per-entry overhead (key strings + hash + map slot),
/// included in byte accounting so the budget reflects real memory, not just
/// body bytes.
const KEY_OVERHEAD_BYTES: usize = 160;
/// Per-tenant operation ceiling in addition to the byte reservations below.
const WRITE_LANE_CAPACITY: usize = 16;
/// The queued-body half of the configured cache envelope never reserves more
/// than 16 MiB. Smaller cells split the envelope evenly; larger cells leave the
/// remainder to stored entries.
const MAX_WRITE_RETENTION_BYTES: usize = 16 * 1024 * 1024;
/// Maximum distinct purge-generation scopes retained per tenant. The wildcard
/// flush-all scope counts as one; bumping an existing scope allocates no state.
pub const MAX_PURGED_SCOPES_PER_TENANT: usize = 64;

// --- Clock (injectable, same pattern as `router`) ------------------------------

/// Milliseconds-since-epoch clock, injectable so TTL/eviction tests are
/// deterministic (no sleeps, no flake).
pub trait Clock: Send + Sync {
    fn now_ms(&self) -> u64;
}

/// Production clock.
#[derive(Debug, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now_ms(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }
}

// --- Key (PRD-007 FR-5 + FR-7) --------------------------------------------------

/// A namespaced exact-match cache key. `tenant_id` and `namespace` are
/// STRUCTURAL components (they participate in `Eq`/`Hash`), so tenant A can
/// never hit tenant B's entry even with an identical request hash and an
/// identical namespace string (FR-7 / NFR-5).
///
/// The `hash` is a one-way SHA-256 over the canonical request — no raw prompt
/// material is recoverable from a key (PRD-007 §6.1), and classification-
/// positive requests never produce a key at all (proxy bypass, FR-10.1).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey {
    tenant_id: TenantId,
    namespace: String,
    hash: [u8; 32],
}

impl CacheKey {
    pub fn tenant_id(&self) -> &TenantId {
        &self.tenant_id
    }

    fn shard(&self) -> usize {
        // Top 6 bits of the digest → 0..64 (ADR-022 §2).
        (self.hash[0] >> 2) as usize
    }
}

/// The canonical-request view that is hashed (FR-5): `model`, the full
/// `messages` array, and every output-affecting generation parameter.
/// `stream` (transport, not content) and `user` (attribution metadata) are
/// EXCLUDED by omission. Field order is fixed by this struct, so two client
/// payloads with different JSON key order produce the same digest.
///
/// The tool-calling / structured-output / determinism parameters
/// (`tools`, `tool_choice`, `parallel_tool_calls`, `response_format`, `seed`,
/// `logprobs`, `top_logprobs`, `logit_bias`, `service_tier`,
/// `reasoning_effort`, `max_completion_tokens`)
/// ARE output-affecting and MUST participate in the digest: two requests with
/// identical messages/model but a different `response_format` or `tools` array
/// produce different responses, so they must key distinctly (else request B is
/// served request A's body). They are appended AFTER the legacy fields and every
/// one carries `skip_serializing_if = "Option::is_none"`, so a request that omits
/// them serializes byte-identically to the pre-tool-calling form — gen-0 keys and
/// the golden/`ab_parity` snapshots stay stable.
#[derive(Serialize)]
struct KeyView<'a> {
    model: &'a str,
    messages: &'a [Message],
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    /// Reasoning-model output cap — output-affecting exactly like `max_tokens`
    /// (a truncated response cached under `max_completion_tokens: 16` must not
    /// be served to a `max_completion_tokens: 4096` request). Skip-if-none so
    /// requests that omit it hash byte-identically to gen-0 keys.
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<&'a [String]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    n: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    // --- output-affecting fields threaded into `crates/types` after the original
    // KeyView was written (tool calling, structured outputs, determinism) ---
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a [Tool]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<&'a serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<&'a serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    logprobs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_logprobs: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    logit_bias: Option<&'a BTreeMap<String, f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    service_tier: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<&'a str>,
}

/// Build the exact-match key for a (tenant, namespace, shaping-resolved request,
/// normalized provider chain) tuple — PRD-007 FR-5/FR-7.
///
/// The caller MUST pass the request in its **shaping-resolved, pre-masking**
/// form (FR-5 is explicit on both, and both are load-bearing: post-shaping so
/// configs that shape differently never collide; pre-masking so two requests
/// differing only in masked values never collide).
///
/// This is the generation-0 (never-purged) form, kept as the canonical key so
/// existing golden/parity snapshots stay byte-identical. To fold a flush
/// generation in (PRD-007 FR-19) use [`exact_key_gen`].
pub fn exact_key(
    tenant_id: &TenantId,
    namespace: &str,
    req: &ChatCompletionRequest,
    provider_chain: &[String],
) -> CacheKey {
    exact_key_gen(tenant_id, namespace, req, provider_chain, 0)
}

/// Build the exact-match key, folding a per-`(tenant, namespace)` flush
/// **generation** into the hash (PRD-007 FR-19, [`FlushRegistry`]).
///
/// CRITICAL byte-identity invariant: when `generation == 0` (the default — no
/// purge has ever been issued for this `(tenant, namespace)`) the generation is
/// NOT mixed into the digest at all, so the produced key is **byte-identical**
/// to the pre-FR-19 key. This keeps `ab_parity`/`golden` snapshots stable and
/// leaves already-cached gen-0 entries reachable. A purge bumps the generation
/// to `1, 2, …`; from then on the digest absorbs the generation, so every
/// prior-generation entry becomes unreachable (a fresh miss) and ages out via
/// the existing TTL/FIFO eviction — O(1), lock-free, no shard iteration.
pub fn exact_key_gen(
    tenant_id: &TenantId,
    namespace: &str,
    req: &ChatCompletionRequest,
    provider_chain: &[String],
    generation: u64,
) -> CacheKey {
    let view = KeyView {
        model: &req.model,
        messages: &req.messages,
        temperature: req.temperature,
        top_p: req.top_p,
        max_tokens: req.max_tokens,
        max_completion_tokens: req.max_completion_tokens,
        stop: req.stop.as_deref(),
        n: req.n,
        presence_penalty: req.presence_penalty,
        frequency_penalty: req.frequency_penalty,
        tools: req.tools.as_deref(),
        tool_choice: req.tool_choice.as_ref(),
        parallel_tool_calls: req.parallel_tool_calls,
        response_format: req.response_format.as_ref(),
        seed: req.seed,
        logprobs: req.logprobs,
        top_logprobs: req.top_logprobs,
        logit_bias: req.logit_bias.as_ref(),
        service_tier: req.service_tier.as_deref(),
        reasoning_effort: req.reasoning_effort.as_deref(),
    };
    let mut hasher = Sha256::new();
    // Serializing a plain-data view cannot fail; the empty-vec fallback keeps
    // this infallible on the request path (a degenerate hash means a
    // conservative shared-miss bucket, never a panic).
    hasher.update(serde_json::to_vec(&view).unwrap_or_default());
    hasher.update([0x1f]);
    for p in provider_chain {
        hasher.update(p.trim().to_ascii_lowercase().as_bytes());
        hasher.update([0x1f]);
    }
    // Gen-0 is byte-identical to the legacy key: only mix the generation in once
    // a purge has bumped it past 0. A distinct domain-separator tag (`0x1e`)
    // precedes the LE bytes so the appended generation can never alias the
    // provider-chain separator stream above.
    if generation > 0 {
        hasher.update([0x1e]);
        hasher.update(generation.to_le_bytes());
    }
    let hash: [u8; 32] = hasher.finalize().into();
    CacheKey {
        tenant_id: tenant_id.clone(),
        namespace: namespace.to_string(),
        hash,
    }
}

// --- Flush-generation registry (PRD-007 FR-19) ----------------------------------

/// The reserved scope under which a tenant-wide (no-namespace) "flush-all" purge
/// is recorded. It is namespace-disjoint from every real cache namespace by
/// construction: the policy layer restricts a real namespace to `[a-z0-9_-]{1,64}`,
/// so `*` can never be a legitimate cache namespace and can never collide with one.
///
/// [`FlushRegistry::generation_effective`] folds this scope's generation into the
/// effective generation of EVERY namespace, so a flush-all actually invalidates
/// all of a tenant's namespaces (not just a `*` scope that no read path consults).
pub const WILDCARD_NAMESPACE: &str = "*";

/// A lock-free, wait-free-read registry of per-`(tenant, namespace)` flush
/// generations (PRD-007 FR-19 "flush generations"; the G3.3 contract reserved in
/// `crates/cache/CLAUDE.md`).
///
/// Purge is implemented WITHOUT iterating or mutating the sharded store: bumping
/// a `(tenant, namespace)` generation changes the derived cache key for every
/// subsequent request in that scope ([`exact_key_gen`]), so all prior-generation
/// entries become unreachable and age out via the existing TTL/FIFO eviction.
/// This is O(1) and touches no shard.
///
/// - **Read path (hot, every cacheable request)**: one `ArcSwap::load`
///   (an `Arc` clone — a refcount bump, no allocation) + one `HashMap` probe.
///   Wait-free; no lock, no CAS loop. A missing entry means generation 0 (the
///   default), which yields the byte-identical legacy key.
/// - **Write path (purge, rare)**: copy-on-write inside the target tenant's
///   startup-allocated cell — clone at most 64 scopes, bump one, and publish.
///   A `compare_and_swap` retry loop makes concurrent purges for that tenant
///   safe without blocking readers or cloning another tenant's state.
///
/// Per-replica, like the cache itself (ADR-022 §3): a purge clears THIS replica's
/// view; multi-replica coordinated purge is a documented follow-on, consistent
/// with the per-replica cache posture (no Redis here — that is a trigger-gated
/// rung, ADR-022 §1).
pub struct FlushRegistry {
    // Immutable outer authority registry; only the bounded target tenant cell
    // changes on purge, so one tenant's purge work is independent of all others.
    generations: HashMap<TenantId, Arc<ArcSwap<HashMap<String, u64>>>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlushError {
    InvalidNamespace,
    ScopeLimit,
    UnknownTenant,
}

fn is_valid_flush_namespace(namespace: &str) -> bool {
    namespace == WILDCARD_NAMESPACE
        || (!namespace.is_empty()
            && namespace.len() <= 64
            && namespace.bytes().all(|byte| {
                byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_' || byte == b'-'
            }))
}

impl FlushRegistry {
    pub fn new(tenant_ids: Vec<TenantId>) -> Self {
        let generations = tenant_ids
            .into_iter()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .map(|tenant_id| (tenant_id, Arc::new(ArcSwap::from_pointee(HashMap::new()))))
            .collect();
        Self { generations }
    }

    /// Wait-free read of the current generation for a `(tenant, namespace)`
    /// scope. Absent ⇒ 0 (the never-purged default → byte-identical legacy key).
    /// One `ArcSwap::load` + one map probe; safe from any thread, never blocks.
    pub fn generation(&self, tenant_id: &TenantId, namespace: &str) -> u64 {
        self.generations
            .get(tenant_id)
            .and_then(|cell| cell.load().get(namespace).copied())
            .unwrap_or(0)
    }

    /// The EFFECTIVE flush generation for a `(tenant, namespace)` on the read
    /// path: the namespace's own generation folded with the tenant-wide
    /// [`WILDCARD_NAMESPACE`] ("flush-all") generation.
    ///
    /// This is what makes a no-namespace purge real. A flush-all bumps only the
    /// `*` scope; if the key-derivation read consulted the request's own
    /// namespace alone (as it once did) the flush-all would be a silent no-op —
    /// every entry kept being served. Folding `*` in here means bumping it
    /// invalidates every namespace at once.
    ///
    /// The fold is a **saturating SUM**, not a `max`: a single purge bumps
    /// exactly one component by 1, so the sum strictly increases on EVERY purge
    /// (namespace-specific OR wildcard) — no purge can ever be a no-op, and no
    /// prior effective generation ever recurs (so no stale entry resurrects). A
    /// `max` would silently swallow a namespace purge whenever it merely caught
    /// up to an equal wildcard generation.
    ///
    /// Both components default to 0 when absent, so an un-purged tenant yields 0
    /// — the byte-identical legacy (gen-0) key. Reading the wildcard scope itself
    /// returns just its own generation (no self-fold).
    pub fn generation_effective(&self, tenant_id: &TenantId, namespace: &str) -> u64 {
        let wildcard = self.generation(tenant_id, WILDCARD_NAMESPACE);
        if namespace == WILDCARD_NAMESPACE {
            return wildcard;
        }
        self.generation(tenant_id, namespace)
            .saturating_add(wildcard)
    }

    /// Bump (purge) the generation for one `(tenant, namespace)` scope and return
    /// the NEW generation. Copy-on-write under a CAS retry loop so concurrent
    /// purges to other scopes never lose an update and never block a reader.
    /// Off the hot path (the `/v1/cache/purge` surface; rare).
    pub fn bump(&self, tenant_id: &TenantId, namespace: &str) -> Result<u64, FlushError> {
        let Some(cell) = self.generations.get(tenant_id) else {
            return Err(FlushError::UnknownTenant);
        };
        if !is_valid_flush_namespace(namespace) {
            return Err(FlushError::InvalidNamespace);
        }
        loop {
            let current = cell.load();
            let existing = current.get(namespace).copied();
            if existing.is_none() {
                // The wildcard is the tenant's emergency flush-all authority.
                // Reserve one of the fixed 64 slots until it is first used so
                // named scopes can never make tenant-wide invalidation return
                // `ScopeLimit` permanently for the life of the replica.
                let reserved_wildcard = usize::from(
                    namespace != WILDCARD_NAMESPACE && !current.contains_key(WILDCARD_NAMESPACE),
                );
                if current.len().saturating_add(reserved_wildcard) >= MAX_PURGED_SCOPES_PER_TENANT {
                    return Err(FlushError::ScopeLimit);
                }
            }
            let next = existing.unwrap_or(0).saturating_add(1);
            let mut map: HashMap<String, u64> = (**current).clone();
            map.insert(namespace.to_string(), next);
            let new = Arc::new(map);
            let prev = cell.compare_and_swap(&*current, Arc::clone(&new));
            // `compare_and_swap` returns the value that was in place; the swap
            // succeeded iff it is pointer-equal to what we loaded.
            if Arc::ptr_eq(&prev, &current) {
                return Ok(next);
            }
            // Lost the race to a concurrent purge of another (or the same) scope;
            // retry with the fresh snapshot. Readers were never blocked.
        }
    }

    /// Number of distinct purged scopes (diagnostics/`/status`; off the hot path).
    pub fn purged_scope_count(&self) -> usize {
        self.generations
            .values()
            .map(|cell| cell.load().len())
            .sum()
    }
}

// --- Entry ----------------------------------------------------------------------

/// A stored response. `body` is the POST-guardrail serialized response, byte
/// identical to what the original client received (FR-8/FR-9). The usage block
/// and model ride along so a hit's usage event records real token counts and
/// an `estimated_saved_cost` without re-parsing the body (FR-16).
#[derive(Debug, Clone)]
pub struct CacheEntry {
    pub body: Bytes,
    pub model: String,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    pub inserted_at_ms: u64,
    pub ttl_ms: u64,
}

impl CacheEntry {
    fn cost(&self) -> usize {
        self.body.len() + self.model.len() + KEY_OVERHEAD_BYTES
    }

    fn expired(&self, now_ms: u64) -> bool {
        now_ms >= self.inserted_at_ms.saturating_add(self.ttl_ms)
    }
}

// --- Status (PRD-007 FR-15/FR-16) ------------------------------------------------

/// The five-value cache verdict. Header form is kebab-case (`semantic-hit`),
/// usage-event form is snake_case (`semantic_hit`) — FR-15/FR-16. Header is
/// ABSENT when no cache config was supplied (FR-2: absence of signal, never a
/// fake `miss`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheStatus {
    Hit,
    Miss,
    Refreshed,
    SemanticHit,
    Bypass,
}

impl CacheStatus {
    pub fn header_value(self) -> &'static str {
        match self {
            CacheStatus::Hit => "hit",
            CacheStatus::Miss => "miss",
            CacheStatus::Refreshed => "refreshed",
            CacheStatus::SemanticHit => "semantic-hit",
            CacheStatus::Bypass => "bypass",
        }
    }

    pub fn event_value(self) -> &'static str {
        match self {
            CacheStatus::Hit => "hit",
            CacheStatus::Miss => "miss",
            CacheStatus::Refreshed => "refreshed",
            CacheStatus::SemanticHit => "semantic_hit",
            CacheStatus::Bypass => "bypass",
        }
    }
}

// --- Core (pure storage; single-writer discipline) -------------------------------

type ShardMap = HashMap<CacheKey, Arc<CacheEntry>>;

/// Immutable startup-time ownership plan for exact-cache bytes and writer
/// lanes. Only explicit canonical tenant ids are admitted. Sorting before
/// allocation makes the split deterministic across registry key ordering, and
/// deduplication means multiple virtual keys for one tenant never buy that
/// tenant extra capacity.
///
/// The configured body envelope is split into queued and stored bytes, then each
/// half is divided across tenants. Storage shares are tenant-global across all
/// 64 shards: shard selection never shrinks the effective entry ceiling.
#[derive(Debug)]
pub struct TenantCapacityPlan {
    tenant_ids: Vec<TenantId>,
    tenant_index: HashMap<TenantId, usize>,
    storage_budgets: Vec<usize>,
    writer_budgets: Vec<usize>,
    budget_bytes: usize,
    storage_budget_bytes: usize,
    writer_budget_bytes: usize,
    // Tenant-share ceiling after fixed key/map overhead, but before model
    // metadata and the independent 256 KiB response-body cap are applied.
    // Keeping this raw value prevents a large-share tenant from being charged
    // `model_len` against the platform body cap itself.
    effective_payload_budget_bytes: Vec<usize>,
    min_positive_effective_payload_budget_bytes: usize,
}

impl TenantCapacityPlan {
    pub fn new(budget_bytes: usize, tenant_ids: Vec<TenantId>) -> Self {
        fn partition(total: usize, count: usize) -> Vec<usize> {
            if count == 0 {
                return Vec::new();
            }
            let base = total / count;
            let remainder = total % count;
            (0..count)
                .map(|index| base + usize::from(index < remainder))
                .collect()
        }

        let tenant_ids: Vec<TenantId> = tenant_ids
            .into_iter()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let tenant_index = tenant_ids
            .iter()
            .enumerate()
            .map(|(index, tenant_id)| (tenant_id.clone(), index))
            .collect();
        let writer_budget_bytes = (budget_bytes / 2).min(MAX_WRITE_RETENTION_BYTES);
        let storage_budget_bytes = budget_bytes.saturating_sub(writer_budget_bytes);
        let storage_budgets = partition(storage_budget_bytes, tenant_ids.len());
        let writer_budgets = partition(writer_budget_bytes, tenant_ids.len());
        let effective_payload_budget_bytes: Vec<usize> = storage_budgets
            .iter()
            .zip(&writer_budgets)
            .map(|(stored, queued)| (*stored).min(*queued).saturating_sub(KEY_OVERHEAD_BYTES))
            .collect();
        let min_positive_effective_payload_budget_bytes = effective_payload_budget_bytes
            .iter()
            .copied()
            .filter(|bytes| *bytes > 0)
            .min()
            .unwrap_or(0);
        Self {
            tenant_ids,
            tenant_index,
            storage_budgets,
            writer_budgets,
            budget_bytes,
            storage_budget_bytes,
            writer_budget_bytes,
            effective_payload_budget_bytes,
            min_positive_effective_payload_budget_bytes,
        }
    }

    fn tenant_ids(&self) -> &[TenantId] {
        &self.tenant_ids
    }

    pub fn tenant_count(&self) -> usize {
        self.tenant_ids.len()
    }

    pub fn budget_bytes(&self) -> usize {
        self.budget_bytes
    }

    pub fn storage_budget_bytes(&self) -> usize {
        self.storage_budget_bytes
    }

    pub fn writer_budget_bytes(&self) -> usize {
        self.writer_budget_bytes
    }

    pub fn min_effective_entry_bytes(&self) -> usize {
        self.min_positive_effective_payload_budget_bytes
            .min(MAX_ENTRY_BYTES)
    }

    fn max_response_bytes_for_tenant(&self, tenant_id: &TenantId, model_len: usize) -> usize {
        let Some(index) = self.tenant_index.get(tenant_id).copied() else {
            return 0;
        };
        self.effective_payload_budget_bytes[index]
            .saturating_sub(model_len)
            .min(MAX_ENTRY_BYTES)
    }

    fn owns_positive_share(&self, tenant_id: &TenantId) -> bool {
        self.tenant_index
            .get(tenant_id)
            .is_some_and(|index| self.effective_payload_budget_bytes[*index] > 0)
    }

    #[cfg(test)]
    fn planned_bytes(&self) -> usize {
        self.storage_budgets.iter().sum::<usize>() + self.writer_budgets.iter().sum::<usize>()
    }
}

/// The sharded store. `lookup` is safe from any thread (lock-free);
/// `apply_insert` must only ever be called from the single writer (the
/// [`ExactCache`] writer thread in production, the test body in unit tests).
pub struct CacheCore {
    shards: Vec<ArcSwap<ShardMap>>,
    capacity: Arc<TenantCapacityPlan>,
    // Single-writer-maintained totals provide the ordinary under-budget insert
    // fast path. Atomics make aggregate diagnostics race-safe without a lock.
    tenant_bytes: Vec<AtomicUsize>,
    clock: Arc<dyn Clock>,
    oversize_drops: AtomicU64,
    write_drops: AtomicU64,
    // Cumulative read-path hit/miss counters (process lifetime; scale-to-zero
    // resets them, like the rest of the cache state). One relaxed `fetch_add`
    // per lookup — an atomic, NOT a lock, so the read path stays lock-free.
    hits: AtomicU64,
    misses: AtomicU64,
}

impl CacheCore {
    pub fn new(budget_bytes: usize, tenant_ids: Vec<TenantId>, clock: Arc<dyn Clock>) -> Self {
        Self::with_capacity(
            Arc::new(TenantCapacityPlan::new(budget_bytes, tenant_ids)),
            clock,
        )
    }

    fn with_capacity(capacity: Arc<TenantCapacityPlan>, clock: Arc<dyn Clock>) -> Self {
        let tenant_bytes = (0..capacity.tenant_count())
            .map(|_| AtomicUsize::new(0))
            .collect();
        Self {
            shards: (0..SHARD_COUNT)
                .map(|_| ArcSwap::from_pointee(ShardMap::new()))
                .collect(),
            capacity,
            tenant_bytes,
            clock,
            oversize_drops: AtomicU64::new(0),
            write_drops: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Lock-free read: one atomic load + one map probe + a TTL check (NFR-1).
    /// Expired entries are treated as misses and lazily collected at the next
    /// shard rebuild (ADR-022 §2). Records a hit/miss counter (one relaxed
    /// atomic add) for the `/status` surface — still lock-free.
    pub fn lookup(&self, key: &CacheKey) -> Option<Arc<CacheEntry>> {
        let guard = self.shards[key.shard()].load();
        let hit = match guard.get(key) {
            Some(entry) if !entry.expired(self.clock.now_ms()) => Some(Arc::clone(entry)),
            _ => None,
        };
        if hit.is_some() {
            self.hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
        }
        hit
    }

    pub fn hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }

    pub fn misses(&self) -> u64 {
        self.misses.load(Ordering::Relaxed)
    }

    /// Lock-free snapshot of `(entries, approx_bytes)` across all shards. Off the
    /// hot path (the `/status` surface, human polling cadence): each shard is one
    /// `ArcSwap::load` — the same lock-free read `lookup` uses — so it never
    /// blocks a request. Approximate because shards are read independently (a
    /// concurrent write may land between loads), which is fine for a gauge.
    pub fn stats_snapshot(&self) -> (usize, usize) {
        let mut entries = 0;
        let mut bytes = 0;
        for shard in &self.shards {
            let guard = shard.load();
            entries += guard.len();
            bytes += guard.values().map(|e| e.cost()).sum::<usize>();
        }
        (entries, bytes)
    }

    /// Writer-only: clone-insert-evict-publish. Eviction is TTL-first, then
    /// FIFO by insert time, until the INSERTING TENANT is back under its
    /// non-borrowable tenant-global share. Other tenants are never candidates.
    /// An entry over the per-entry cap (or over its tenant slice by itself) is
    /// dropped and counted, never stored. An unregistered tenant is a write
    /// drop: no capacity or lane is ever allocated dynamically.
    fn apply_insert(&self, key: CacheKey, entry: Arc<CacheEntry>) {
        let idx = key.shard();
        let Some(tenant_index) = self.capacity.tenant_index.get(&key.tenant_id).copied() else {
            self.write_drops.fetch_add(1, Ordering::Relaxed);
            return;
        };
        let tenant_budget = self.capacity.storage_budgets[tenant_index];
        let cost = entry.cost();
        if entry.body.len() > MAX_ENTRY_BYTES || cost > tenant_budget {
            self.oversize_drops.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let tenant_id = key.tenant_id.clone();
        let now = self.clock.now_ms();
        let current = self.shards[idx].load();
        let mut target_map: ShardMap = (**current).clone();
        let replaced_cost = target_map.get(&key).map(|prior| prior.cost()).unwrap_or(0);
        target_map.insert(key, entry);

        let projected_total = self.tenant_bytes[tenant_index]
            .load(Ordering::Acquire)
            .saturating_sub(replaced_cost)
            .saturating_add(cost);
        if projected_total <= tenant_budget {
            self.shards[idx].store(Arc::new(target_map));
            self.tenant_bytes[tenant_index].store(projected_total, Ordering::Release);
            return;
        }

        let mut candidates: Vec<(CacheKey, u64, usize, bool)> = Vec::new();
        let mut tenant_total = 0usize;
        for shard_index in 0..SHARD_COUNT {
            if shard_index == idx {
                for (candidate_key, candidate_entry) in &target_map {
                    if candidate_key.tenant_id == tenant_id {
                        let candidate_cost = candidate_entry.cost();
                        tenant_total = tenant_total.saturating_add(candidate_cost);
                        candidates.push((
                            candidate_key.clone(),
                            candidate_entry.inserted_at_ms,
                            candidate_cost,
                            candidate_entry.expired(now),
                        ));
                    }
                }
            } else {
                let shard = self.shards[shard_index].load();
                for (candidate_key, candidate_entry) in shard.iter() {
                    if candidate_key.tenant_id == tenant_id {
                        let candidate_cost = candidate_entry.cost();
                        tenant_total = tenant_total.saturating_add(candidate_cost);
                        candidates.push((
                            candidate_key.clone(),
                            candidate_entry.inserted_at_ms,
                            candidate_cost,
                            candidate_entry.expired(now),
                        ));
                    }
                }
            }
        }

        let mut removals: HashMap<usize, Vec<CacheKey>> = HashMap::new();
        if tenant_total > tenant_budget {
            for (candidate_key, _, candidate_cost, _expired) in
                candidates.iter().filter(|candidate| candidate.3)
            {
                removals
                    .entry(candidate_key.shard())
                    .or_default()
                    .push(candidate_key.clone());
                tenant_total = tenant_total.saturating_sub(*candidate_cost);
            }
        }
        if tenant_total > tenant_budget {
            let mut by_age: Vec<(CacheKey, u64, usize)> = candidates
                .into_iter()
                .filter(|candidate| !candidate.3)
                .map(|(candidate_key, inserted_at_ms, cost, _)| {
                    (candidate_key, inserted_at_ms, cost)
                })
                .collect();
            by_age.sort_by_key(|(_, inserted, _)| *inserted);
            for (candidate_key, _, candidate_cost) in by_age {
                if tenant_total <= tenant_budget {
                    break;
                }
                removals
                    .entry(candidate_key.shard())
                    .or_default()
                    .push(candidate_key);
                tenant_total = tenant_total.saturating_sub(candidate_cost);
            }
        }

        for (shard_index, keys) in removals {
            if shard_index == idx {
                for candidate_key in keys {
                    target_map.remove(&candidate_key);
                }
            } else {
                let current = self.shards[shard_index].load();
                let mut map: ShardMap = (**current).clone();
                for candidate_key in keys {
                    map.remove(&candidate_key);
                }
                self.shards[shard_index].store(Arc::new(map));
            }
        }
        self.shards[idx].store(Arc::new(target_map));
        self.tenant_bytes[tenant_index].store(tenant_total, Ordering::Release);
    }

    pub fn now_ms(&self) -> u64 {
        self.clock.now_ms()
    }

    pub fn record_oversize(&self) {
        self.oversize_drops.fetch_add(1, Ordering::Relaxed);
    }

    pub fn oversize_drops(&self) -> u64 {
        self.oversize_drops.load(Ordering::Relaxed)
    }

    pub fn write_drops(&self) -> u64 {
        self.write_drops.load(Ordering::Relaxed)
    }

    pub fn tenant_count(&self) -> usize {
        self.capacity.tenant_count()
    }

    pub fn storage_budget_bytes(&self) -> usize {
        self.capacity.storage_budget_bytes()
    }

    pub fn min_effective_entry_bytes(&self) -> usize {
        self.capacity.min_effective_entry_bytes()
    }

    pub fn max_response_bytes_for_tenant(&self, tenant_id: &TenantId, model_len: usize) -> usize {
        self.capacity
            .max_response_bytes_for_tenant(tenant_id, model_len)
    }
}

// --- Public handle (write-behind via a single writer thread) ---------------------

/// One write-behind insert (kept as a struct so the call site stays readable
/// and under the clippy argument bound).
#[derive(Debug, Clone)]
pub struct CacheWrite {
    pub key: CacheKey,
    pub body: Bytes,
    pub ttl_seconds: u64,
    pub model: String,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

struct ReservedInsert {
    key: CacheKey,
    entry: Arc<CacheEntry>,
    queued_bytes: Arc<AtomicUsize>,
    cost: usize,
}

impl Drop for ReservedInsert {
    fn drop(&mut self) {
        self.queued_bytes.fetch_sub(self.cost, Ordering::Release);
    }
}

enum LaneOp {
    Insert(ReservedInsert),
    /// Per-lane barrier. Sending one into every lane preserves FIFO order inside
    /// each tenant while avoiding a shared control queue.
    Barrier(SyncSender<()>),
}

/// Receivers owned exclusively by the one writer thread. The request path puts
/// at most one ready token per tenant into the fixed ready queue. The writer
/// consumes one operation per turn, then requeues that tenant only when work
/// remains. Inactive tenants are never scanned per operation.
struct ReadyLaneReceivers {
    lanes: Vec<Receiver<LaneOp>>,
    scheduled: Vec<Arc<AtomicBool>>,
    ready: Receiver<usize>,
    local_ready: VecDeque<usize>,
    pending: Vec<Option<LaneOp>>,
}

impl ReadyLaneReceivers {
    fn collect_newly_ready(&mut self) {
        while let Ok(index) = self.ready.try_recv() {
            self.local_ready.push_back(index);
        }
    }

    fn take_ready(&mut self, index: usize) -> Option<(usize, LaneOp)> {
        if let Some(op) = self.pending[index].take() {
            return Some((index, op));
        }
        match self.lanes[index].try_recv() {
            Ok(op) => Some((index, op)),
            Err(TryRecvError::Empty | TryRecvError::Disconnected) => {
                // A stale token is harmless. Clear it so a later producer can
                // schedule fresh work.
                self.scheduled[index].store(false, Ordering::Release);
                None
            }
        }
    }

    fn try_next(&mut self) -> Option<(usize, LaneOp)> {
        loop {
            // Merge producer signals into the one writer-owned FIFO before
            // choosing a turn. Neither newly-ready nor already-backlogged lanes
            // can jump the other queue indefinitely.
            self.collect_newly_ready();
            let index = self.local_ready.pop_front()?;
            if let Some(ready) = self.take_ready(index) {
                return Some(ready);
            }
        }
    }

    fn next(&mut self) -> Option<(usize, LaneOp)> {
        loop {
            if let Some(ready) = self.try_next() {
                return Some(ready);
            }
            let index = match self.ready.recv() {
                Ok(index) => index,
                Err(_) => return self.try_next(),
            };
            self.local_ready.push_back(index);
        }
    }

    fn finish_turn(&mut self, index: usize) {
        match self.lanes[index].try_recv() {
            Ok(op) => {
                self.pending[index] = Some(op);
                self.local_ready.push_back(index);
                return;
            }
            Err(TryRecvError::Disconnected) => {
                self.scheduled[index].store(false, Ordering::Release);
                return;
            }
            Err(TryRecvError::Empty) => {}
        }

        // Close the producer/consumer race without a scan: clear the bit, then
        // double-check the lane. A producer that wins the CAS sends the global
        // ready token; otherwise the writer owns the discovered work locally.
        self.scheduled[index].store(false, Ordering::Release);
        if let Ok(op) = self.lanes[index].try_recv() {
            self.pending[index] = Some(op);
            if self.scheduled[index]
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                self.local_ready.push_back(index);
            }
        }
    }
}

/// Immutable tenant-to-lane admission map shared by request threads. Each send
/// is a tenant-local bounded `try_send` after reserving from that tenant's share
/// of one fixed queued-byte envelope.
struct WriteAdmission {
    tenant_index: HashMap<TenantId, usize>,
    lanes: Vec<SyncSender<LaneOp>>,
    ready: SyncSender<usize>,
    scheduled: Vec<Arc<AtomicBool>>,
    queued_bytes: Vec<Arc<AtomicUsize>>,
    writer_budgets: Vec<usize>,
    effective_entry_budgets: Vec<usize>,
}

impl WriteAdmission {
    fn new(capacity: &TenantCapacityPlan, lane_capacity: usize) -> (Self, ReadyLaneReceivers) {
        let mut lane_senders = Vec::with_capacity(capacity.tenant_count());
        let mut lane_receivers = Vec::with_capacity(capacity.tenant_count());
        let mut scheduled = Vec::with_capacity(capacity.tenant_count());
        let mut queued_bytes = Vec::with_capacity(capacity.tenant_count());
        for _ in capacity.tenant_ids() {
            let (sender, receiver) = sync_channel(lane_capacity);
            lane_senders.push(sender);
            lane_receivers.push(receiver);
            scheduled.push(Arc::new(AtomicBool::new(false)));
            queued_bytes.push(Arc::new(AtomicUsize::new(0)));
        }
        let (ready, ready_receiver) = sync_channel(capacity.tenant_count().saturating_add(1));
        let writer_budgets = capacity.writer_budgets.clone();
        let effective_entry_budgets = capacity
            .storage_budgets
            .iter()
            .zip(&writer_budgets)
            .map(|(stored, queued)| (*stored).min(*queued))
            .collect();
        (
            Self {
                tenant_index: capacity.tenant_index.clone(),
                lanes: lane_senders,
                ready,
                scheduled: scheduled.clone(),
                queued_bytes,
                writer_budgets,
                effective_entry_budgets,
            },
            ReadyLaneReceivers {
                lanes: lane_receivers,
                scheduled,
                ready: ready_receiver,
                local_ready: VecDeque::new(),
                pending: (0..capacity.tenant_count()).map(|_| None).collect(),
            },
        )
    }

    fn schedule(&self, lane_index: usize) {
        if self.scheduled[lane_index]
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return;
        }
        if self.ready.try_send(lane_index).is_err() {
            // Disconnection is the only reachable failure: with at most one
            // token per tenant the queue has one spare slot and cannot fill.
            self.scheduled[lane_index].store(false, Ordering::Release);
        }
    }

    fn reserve_bytes(&self, lane_index: usize, cost: usize) -> bool {
        let counter = &self.queued_bytes[lane_index];
        let limit = self.writer_budgets[lane_index];
        let mut current = counter.load(Ordering::Acquire);
        loop {
            let Some(next) = current.checked_add(cost) else {
                return false;
            };
            if next > limit {
                return false;
            }
            match counter.compare_exchange_weak(current, next, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => return true,
                Err(observed) => current = observed,
            }
        }
    }

    fn try_insert(
        &self,
        write: CacheWrite,
        inserted_at_ms: u64,
        write_drops: &AtomicU64,
        oversize_drops: &AtomicU64,
    ) -> bool {
        let Some(lane_index) = self.tenant_index.get(&write.key.tenant_id).copied() else {
            write_drops.fetch_add(1, Ordering::Relaxed);
            return false;
        };
        let entry = Arc::new(CacheEntry {
            body: write.body,
            model: write.model,
            prompt_tokens: write.prompt_tokens,
            completion_tokens: write.completion_tokens,
            total_tokens: write.total_tokens,
            inserted_at_ms,
            ttl_ms: write.ttl_seconds.saturating_mul(1000),
        });
        let cost = entry.cost();
        if entry.body.len() > MAX_ENTRY_BYTES || cost > self.effective_entry_budgets[lane_index] {
            oversize_drops.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        if !self.reserve_bytes(lane_index, cost) {
            write_drops.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        let insert = ReservedInsert {
            key: write.key,
            entry,
            queued_bytes: Arc::clone(&self.queued_bytes[lane_index]),
            cost,
        };
        match self.lanes[lane_index].try_send(LaneOp::Insert(insert)) {
            Ok(()) => {
                self.schedule(lane_index);
                true
            }
            Err(TrySendError::Full(_) | TrySendError::Disconnected(_)) => {
                // The returned operation drops here and releases its byte
                // reservation through `ReservedInsert::drop`.
                write_drops.fetch_add(1, Ordering::Relaxed);
                false
            }
        }
    }

    fn flush(&self) {
        if self.lanes.is_empty() {
            return;
        }
        let (ack_tx, ack_rx) = sync_channel(self.lanes.len());
        let mut expected = 0;
        for (lane_index, lane) in self.lanes.iter().enumerate() {
            if lane.send(LaneOp::Barrier(ack_tx.clone())).is_ok() {
                expected += 1;
                self.schedule(lane_index);
            }
        }
        drop(ack_tx);
        for _ in 0..expected {
            if ack_rx.recv().is_err() {
                break;
            }
        }
    }

    #[cfg(feature = "bench-internals")]
    fn flush_lane(&self, lane_index: usize) {
        let (ack_tx, ack_rx) = sync_channel(0);
        if self.lanes[lane_index].send(LaneOp::Barrier(ack_tx)).is_ok() {
            self.schedule(lane_index);
            let _ = ack_rx.recv();
        }
    }

    fn queued_bytes(&self) -> usize {
        self.queued_bytes
            .iter()
            .map(|bytes| bytes.load(Ordering::Acquire))
            .sum()
    }
}

/// The process-wide exact-match cache handle held by `AppState`.
///
/// Reads go straight to the core (lock-free). Writes use one bounded lane per
/// configured tenant and ONE dedicated OS writer thread. The thread drains the
/// ready lanes one operation per turn, retaining ADR-022's single-mutator
/// guarantee without scanning inactive tenants. The cache crate remains free
/// of any async-runtime dependency.
pub struct ExactCache {
    core: Arc<CacheCore>,
    admission: WriteAdmission,
    writer: Option<JoinHandle<()>>,
}

struct WriterGate {
    reached: SyncSender<()>,
    permit: Receiver<()>,
}

impl ExactCache {
    /// Production constructor (system clock). Spawns the writer thread; called
    /// once at startup (and per test), never on a request path.
    pub fn new(budget_bytes: usize, tenant_ids: Vec<TenantId>) -> Self {
        Self::with_clock(budget_bytes, tenant_ids, Arc::new(SystemClock))
    }

    /// Test constructor with an injectable clock.
    pub fn with_clock(
        budget_bytes: usize,
        tenant_ids: Vec<TenantId>,
        clock: Arc<dyn Clock>,
    ) -> Self {
        Self::with_clock_and_writer_gate(budget_bytes, tenant_ids, clock, WRITE_LANE_CAPACITY, None)
    }

    fn with_clock_and_writer_gate(
        budget_bytes: usize,
        tenant_ids: Vec<TenantId>,
        clock: Arc<dyn Clock>,
        lane_capacity: usize,
        mut writer_gate: Option<WriterGate>,
    ) -> Self {
        let capacity = Arc::new(TenantCapacityPlan::new(budget_bytes, tenant_ids));
        let core = Arc::new(CacheCore::with_capacity(Arc::clone(&capacity), clock));
        let (admission, mut receivers) = WriteAdmission::new(&capacity, lane_capacity);
        let writer_core = Arc::clone(&core);
        // Startup-time spawn (not the request path): a failure to create the
        // writer thread here is an unrecoverable process-level condition.
        let writer = std::thread::Builder::new()
            .name("rp-cache-writer".to_string())
            .spawn(move || {
                while let Some((lane_index, op)) = receivers.next() {
                    let gate_disconnected = if let Some(gate) = writer_gate.as_ref() {
                        let _ = gate.reached.send(());
                        gate.permit.recv().is_err()
                    } else {
                        false
                    };
                    if gate_disconnected {
                        // The deterministic benchmark gate is one-shot. A
                        // disconnected permit releases this and all future
                        // production-writer operations.
                        writer_gate = None;
                    }
                    match op {
                        LaneOp::Insert(insert) => {
                            writer_core.apply_insert(insert.key.clone(), Arc::clone(&insert.entry));
                        }
                        LaneOp::Barrier(ack) => {
                            let _ = ack.send(());
                        }
                    }
                    receivers.finish_turn(lane_index);
                }
            })
            .expect("failed to spawn the cache writer thread at startup");
        Self {
            core,
            admission,
            writer: Some(writer),
        }
    }

    /// Lock-free lookup (the hot path).
    pub fn lookup(&self, key: &CacheKey) -> Option<Arc<CacheEntry>> {
        self.core.lookup(key)
    }

    /// Write-behind insert (FR-8 / ADR-022 §4): immutable tenant lookup plus one
    /// tenant-local bounded `try_send`; the actual map rebuild happens on the
    /// writer thread. A full or unknown tenant lane drops the write (counted) —
    /// the cache is an optimization, never a dependency.
    pub fn insert(&self, write: CacheWrite) -> bool {
        self.admission.try_insert(
            write,
            self.core.now_ms(),
            &self.core.write_drops,
            &self.core.oversize_drops,
        )
    }

    /// Count an oversize body the proxy declined to store (FR-8).
    pub fn record_oversize(&self) {
        self.core.record_oversize();
    }

    pub fn oversize_drops(&self) -> u64 {
        self.core.oversize_drops()
    }

    pub fn write_drops(&self) -> u64 {
        self.core.write_drops()
    }

    pub fn tenant_count(&self) -> usize {
        self.core.tenant_count()
    }

    /// Whether this explicit typed tenant owns a positive startup-allocated
    /// cache share. Registered zero-share tenants remain storage-inert.
    pub fn owns_tenant(&self, tenant_id: &TenantId) -> bool {
        self.core.capacity.owns_positive_share(tenant_id)
    }

    /// Bytes currently retained by all writer lanes combined.
    pub fn queued_bytes(&self) -> usize {
        self.admission.queued_bytes()
    }

    /// Fixed replica storage share after reserving the writer envelope.
    pub fn storage_budget_bytes(&self) -> usize {
        self.core.storage_budget_bytes()
    }

    /// Smallest positive entry ceiling among tenants that own usable capacity.
    pub fn min_effective_entry_bytes(&self) -> usize {
        self.core.min_effective_entry_bytes()
    }

    /// Largest serialized response body this tenant can admit for a model of
    /// this length. Unknown and zero-share tenants return zero.
    pub fn max_response_bytes_for_tenant(&self, tenant_id: &TenantId, model_len: usize) -> usize {
        self.core
            .max_response_bytes_for_tenant(tenant_id, model_len)
    }

    /// Cumulative read-path hits (lock-free counter). See [`CacheCore::hits`].
    pub fn hits(&self) -> u64 {
        self.core.hits()
    }

    /// Cumulative read-path misses (lock-free counter). See [`CacheCore::misses`].
    pub fn misses(&self) -> u64 {
        self.core.misses()
    }

    /// Lock-free `(entries, approx_bytes)` snapshot for the `/status` surface.
    pub fn stats_snapshot(&self) -> (usize, usize) {
        self.core.stats_snapshot()
    }

    /// BLOCKING writer barrier — waits until every previously-queued write has
    /// been applied. For tests and diagnostics only; never call on a request
    /// path. (The writer is an independent OS thread, so blocking here cannot
    /// deadlock an async runtime.)
    pub fn flush(&self) {
        self.admission.flush();
    }

    /// Consume the cache, close every writer lane, and wait for the dedicated
    /// writer to terminate. Process teardown normally drops the handle and lets
    /// the already-closed worker exit independently; tests and embedders can use
    /// this explicit lifecycle seam when a bounded, joined shutdown is needed.
    pub fn shutdown(self) {
        let Self {
            core,
            admission,
            writer,
        } = self;
        drop(admission);
        drop(core);
        if let Some(handle) = writer {
            handle.join().expect("cache writer must shut down cleanly");
        }
    }
}

/// Deterministic harness for the required Criterion target. It owns the real
/// [`ExactCache`] and production writer loop. The saturated variant pauses that
/// writer after dequeueing one operation, then fills the same tenant lane so a
/// timed `try_send` drop is deterministic under a live writer.
#[cfg(feature = "bench-internals")]
#[doc(hidden)]
pub mod bench_support {
    use super::*;

    pub struct AdmissionHarness {
        tenant_ids: Arc<Vec<TenantId>>,
        cache: Arc<ExactCache>,
        lane_capacity: usize,
        permit: Option<SyncSender<()>>,
        reached: Option<Receiver<()>>,
        sequence: Arc<AtomicU64>,
    }

    #[derive(Clone)]
    pub struct AdmissionHandle {
        tenant_ids: Arc<Vec<TenantId>>,
        cache: Arc<ExactCache>,
        sequence: Arc<AtomicU64>,
    }

    impl AdmissionHarness {
        pub fn new(tenant_count: usize, lane_capacity: usize) -> Self {
            Self::build(tenant_count, lane_capacity, false)
        }

        pub fn new_paused(tenant_count: usize, lane_capacity: usize) -> Self {
            Self::build(tenant_count, lane_capacity, true)
        }

        fn build(tenant_count: usize, lane_capacity: usize, paused: bool) -> Self {
            assert!(tenant_count > 0);
            assert!(lane_capacity > 0);
            let tenant_ids: Vec<TenantId> = (0..tenant_count)
                .map(|index| TenantId::new(format!("tenant_{index:04}")).expect("valid tenant"))
                .collect();
            let (writer_gate, permit, reached) = if paused {
                let (reached_tx, reached_rx) = sync_channel(0);
                let (permit_tx, permit_rx) = sync_channel(0);
                (
                    Some(WriterGate {
                        reached: reached_tx,
                        permit: permit_rx,
                    }),
                    Some(permit_tx),
                    Some(reached_rx),
                )
            } else {
                (None, None, None)
            };
            let cache = ExactCache::with_clock_and_writer_gate(
                DEFAULT_BUDGET_BYTES,
                tenant_ids.clone(),
                Arc::new(SystemClock),
                lane_capacity,
                writer_gate,
            );
            Self {
                tenant_ids: Arc::new(tenant_ids),
                cache: Arc::new(cache),
                lane_capacity,
                permit,
                reached,
                sequence: Arc::new(AtomicU64::new(0)),
            }
        }

        pub fn handle(&self) -> AdmissionHandle {
            AdmissionHandle {
                tenant_ids: Arc::clone(&self.tenant_ids),
                cache: Arc::clone(&self.cache),
                sequence: Arc::clone(&self.sequence),
            }
        }

        pub fn tenant_id(&self, index: usize) -> &str {
            self.tenant_ids[index].as_str()
        }

        pub fn saturate_lane(&mut self, index: usize) {
            assert!(self.handle().record(index));
            self.reached
                .take()
                .expect("paused harness has a writer signal")
                .recv()
                .expect("production writer reached the pause gate");
            for _ in 0..self.lane_capacity {
                assert!(self.handle().record(index));
            }
        }

        pub fn release_all(&self) {
            self.cache.flush();
        }

        pub fn release_tenant(&self, index: usize) {
            self.cache.admission.flush_lane(index);
        }

        pub fn dropped_total(&self) -> u64 {
            self.cache.write_drops() + self.cache.oversize_drops()
        }
    }

    impl Drop for AdmissionHarness {
        fn drop(&mut self) {
            // Disconnecting the one-shot permit releases a paused production
            // writer. Once saturation consumed `reached`, a barrier proves it
            // drained before the harness goes away.
            self.permit.take();
            if self.reached.is_none() {
                self.cache.flush();
            }
        }
    }

    impl AdmissionHandle {
        pub fn record(&self, index: usize) -> bool {
            let mut hash = [0; 32];
            let hash_index = u64::try_from(index).expect("tenant index must fit in u64");
            hash[..8].copy_from_slice(&hash_index.to_le_bytes());
            // Keep a small populated working set so long percentile runs
            // exercise the real writer without turning the admission benchmark
            // into an unbounded eviction-throughput benchmark.
            let sequence = self.sequence.fetch_add(1, Ordering::Relaxed) % 64;
            hash[8..16].copy_from_slice(&sequence.to_le_bytes());
            self.cache.insert(CacheWrite {
                key: CacheKey {
                    tenant_id: self.tenant_ids[index].clone(),
                    namespace: "bench".into(),
                    hash,
                },
                body: Bytes::from_static(b"{\"ok\":true}"),
                ttl_seconds: 300,
                model: "bench-model".into(),
                prompt_tokens: 1,
                completion_tokens: 1,
                total_tokens: 2,
            })
        }
    }
}

// =============================== tests ==========================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64 as TestAtomicU64;

    /// Deterministic test clock.
    struct FakeClock(TestAtomicU64);

    impl FakeClock {
        fn at(ms: u64) -> Arc<Self> {
            Arc::new(FakeClock(TestAtomicU64::new(ms)))
        }
        fn set(&self, ms: u64) {
            self.0.store(ms, Ordering::Relaxed);
        }
    }

    impl Clock for FakeClock {
        fn now_ms(&self) -> u64 {
            self.0.load(Ordering::Relaxed)
        }
    }

    fn req_from_json(json: &str) -> ChatCompletionRequest {
        serde_json::from_str(json).expect("request deserializes")
    }

    fn chain(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    fn tenant(id: &str) -> TenantId {
        TenantId::new(id).expect("test tenant id must be canonical")
    }

    fn tenants(ids: &[&str]) -> Vec<TenantId> {
        ids.iter().map(|id| tenant(id)).collect()
    }

    fn exact_key(
        tenant_id: &str,
        namespace: &str,
        req: &ChatCompletionRequest,
        provider_chain: &[String],
    ) -> CacheKey {
        super::exact_key(&tenant(tenant_id), namespace, req, provider_chain)
    }

    fn exact_key_gen(
        tenant_id: &str,
        namespace: &str,
        req: &ChatCompletionRequest,
        provider_chain: &[String],
        generation: u64,
    ) -> CacheKey {
        super::exact_key_gen(
            &tenant(tenant_id),
            namespace,
            req,
            provider_chain,
            generation,
        )
    }

    fn entry(body_len: usize, inserted_at_ms: u64, ttl_ms: u64) -> Arc<CacheEntry> {
        Arc::new(CacheEntry {
            body: Bytes::from(vec![b'x'; body_len]),
            model: "gpt-4o".into(),
            prompt_tokens: 1,
            completion_tokens: 2,
            total_tokens: 3,
            inserted_at_ms,
            ttl_ms,
        })
    }

    fn raw_key(tenant_id: &str, ns: &str, key_index: usize) -> CacheKey {
        let mut hash = [0u8; 32];
        let first_byte = u8::try_from(key_index & usize::from(u8::MAX))
            .expect("masked test key index fits in u8");
        hash[0] = first_byte;
        hash[1] = first_byte; // differentiate keys with the same shard byte
        hash[8..16].copy_from_slice(
            &u64::try_from(key_index)
                .expect("test key index fits in u64")
                .to_le_bytes(),
        );
        CacheKey {
            tenant_id: tenant(tenant_id),
            namespace: ns.into(),
            hash,
        }
    }

    fn write(key: CacheKey) -> CacheWrite {
        CacheWrite {
            key,
            body: Bytes::from_static(b"{\"ok\":true}"),
            ttl_seconds: 300,
            model: "m".into(),
            prompt_tokens: 1,
            completion_tokens: 1,
            total_tokens: 2,
        }
    }

    // --- keying determinism (FR-5 / AC-2) -------------------------------------

    #[test]
    fn key_is_deterministic_across_client_json_field_order() {
        // Same logical request, different JSON key order → same typed struct →
        // same digest (canonicalization by construction).
        let a = req_from_json(
            r#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}],"temperature":0.5,"max_tokens":64}"#,
        );
        let b = req_from_json(
            r#"{"max_tokens":64,"temperature":0.5,"messages":[{"role":"user","content":"hi"}],"model":"gpt-4o"}"#,
        );
        let c = chain(&["openai"]);
        assert_eq!(
            exact_key("t", "default", &a, &c),
            exact_key("t", "default", &b, &c)
        );
    }

    #[test]
    fn stream_and_user_are_excluded_from_the_key() {
        let base = req_from_json(r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#);
        let mut streamed = base.clone();
        streamed.stream = Some(true);
        let mut with_user = base.clone();
        with_user.user = Some("end-user-7".into());
        let c = chain(&["openai"]);
        assert_eq!(
            exact_key("t", "default", &base, &c),
            exact_key("t", "default", &streamed, &c)
        );
        assert_eq!(
            exact_key("t", "default", &base, &c),
            exact_key("t", "default", &with_user, &c)
        );
    }

    #[test]
    fn output_affecting_changes_change_the_key() {
        let base = req_from_json(r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#);
        let c = chain(&["openai"]);
        let k0 = exact_key("t", "default", &base, &c);

        let mut temp = base.clone();
        temp.temperature = Some(0.9);
        assert_ne!(k0, exact_key("t", "default", &temp, &c));

        let mut model = base.clone();
        model.model = "m2".into();
        assert_ne!(k0, exact_key("t", "default", &model, &c));

        let mut msg = base.clone();
        msg.messages[0].content = "hi!".into();
        assert_ne!(k0, exact_key("t", "default", &msg, &c));
    }

    #[test]
    fn response_shaping_params_change_the_key() {
        // FR-5 regression: the output-affecting generation parameters threaded
        // into ChatCompletionRequest AFTER the original KeyView (tool calling,
        // structured outputs, determinism) MUST participate in the digest.
        // Otherwise two same-tenant/same-namespace requests differing only in
        // (say) `response_format` collide, and request B is served request A's
        // body — e.g. a `{"type":"json_object"}` request receives a plain-text
        // completion, or a tools-carrying agent request receives a tool-free
        // answer.
        let base = req_from_json(r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#);
        let c = chain(&["openai"]);
        let k0 = exact_key("t", "default", &base, &c);

        // Each payload differs from `base` in exactly ONE output-affecting field;
        // each must therefore key distinctly.
        let variants = [
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"get_weather"}}]}"#,
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"tool_choice":"required"}"#,
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"parallel_tool_calls":false}"#,
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"response_format":{"type":"json_object"}}"#,
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"seed":42}"#,
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"logprobs":true}"#,
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"top_logprobs":5}"#,
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"logit_bias":{"123":-100.0}}"#,
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"service_tier":"flex"}"#,
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"reasoning_effort":"high"}"#,
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"max_completion_tokens":16}"#,
        ];
        for v in variants {
            let req = req_from_json(v);
            assert_ne!(
                k0,
                exact_key("t", "default", &req, &c),
                "output-affecting field must change the cache key: {v}"
            );
        }

        // Two DIFFERENT response_format values must not collide either (json_object
        // vs a json_schema shape → different responses).
        let json_object = req_from_json(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"response_format":{"type":"json_object"}}"#,
        );
        let json_schema = req_from_json(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"response_format":{"type":"json_schema","json_schema":{"name":"x"}}}"#,
        );
        assert_ne!(
            exact_key("t", "default", &json_object, &c),
            exact_key("t", "default", &json_schema, &c)
        );

        // A request that OMITS all the new fields is byte-identical to the legacy
        // key (skip_serializing_if = none ⇒ no new bytes) — golden/ab_parity stay
        // stable.
        assert_eq!(k0, exact_key_gen("t", "default", &base, &c, 0));
    }

    #[test]
    fn provider_chain_is_normalized_but_order_sensitive() {
        let req = req_from_json(r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#);
        // Case/whitespace normalization → same key.
        assert_eq!(
            exact_key("t", "default", &req, &chain(&["OpenAI "])),
            exact_key("t", "default", &req, &chain(&["openai"]))
        );
        // Reorder → different key (FR-5: accepted cost of determinism).
        assert_ne!(
            exact_key("t", "default", &req, &chain(&["openai", "anthropic"])),
            exact_key("t", "default", &req, &chain(&["anthropic", "openai"]))
        );
    }

    // --- immutable tenant capacity/admission plan -----------------------------

    #[test]
    fn capacity_plan_is_canonical_deduplicated_deterministic_and_exact() {
        let budget = 64 * 1024 + 7;
        let plan = TenantCapacityPlan::new(budget, tenants(&["tenant_b", "tenant_a", "tenant_b"]));
        assert_eq!(plan.tenant_ids(), &[tenant("tenant_a"), tenant("tenant_b")]);
        assert_eq!(plan.tenant_count(), 2);
        assert_eq!(plan.budget_bytes(), budget);
        assert_eq!(plan.planned_bytes(), budget, "no hidden over-allocation");
        assert_eq!(
            plan.storage_budget_bytes() + plan.writer_budget_bytes(),
            budget
        );
        assert_eq!(plan.tenant_index.get(&tenant("tenant_a")).copied(), Some(0));
        assert_eq!(plan.tenant_index.get(&tenant("tenant_b")).copied(), Some(1));
        assert_eq!(plan.tenant_index.get(&tenant("unknown")).copied(), None);
    }

    #[test]
    fn derived_entry_ceiling_reports_the_platform_cap_without_double_charging_model_bytes() {
        let plan = TenantCapacityPlan::new(DEFAULT_BUDGET_BYTES, tenants(&["tenant_a"]));
        assert_eq!(
            plan.min_effective_entry_bytes(),
            MAX_ENTRY_BYTES,
            "the status-facing ceiling must not exceed the platform cap"
        );
        assert_eq!(
            plan.max_response_bytes_for_tenant(&tenant("tenant_a"), 4 * 1024),
            MAX_ENTRY_BYTES,
            "ample tenant share keeps the full body cap after model metadata"
        );
    }

    #[test]
    fn sixty_four_tenants_retain_representative_response_capacity() {
        let ids: Vec<TenantId> = (0..64)
            .map(|index| tenant(&format!("tenant_{index:02}")))
            .collect();
        for budget in [16 * 1024 * 1024, DEFAULT_BUDGET_BYTES] {
            let plan = TenantCapacityPlan::new(budget, ids.clone());
            assert!(
                plan.min_effective_entry_bytes() >= 8 * 1024,
                "an 8 KiB response must remain cacheable at budget={budget}"
            );
            assert_eq!(plan.planned_bytes(), budget);

            let core = CacheCore::new(budget, ids.clone(), FakeClock::at(1_000));
            let key = raw_key("tenant_63", "ns", 63);
            core.apply_insert(key.clone(), entry(8 * 1024, 1_000, 60_000));
            assert!(
                core.lookup(&key).is_some(),
                "the last tenant must retain an actual 8 KiB cache entry at budget={budget}"
            );
        }
    }

    #[test]
    fn advertised_model_aware_body_ceiling_is_storable_at_the_boundary() {
        let cache = ExactCache::with_clock(1024 * 1024, tenants(&["t"]), FakeClock::at(1_000));
        let model = "model-with-metadata";
        let tenant = tenant("t");
        let ceiling = cache.max_response_bytes_for_tenant(&tenant, model.len());
        let accepted_key = raw_key("t", "ns", 1);
        assert!(cache.insert(CacheWrite {
            key: accepted_key.clone(),
            body: Bytes::from(vec![b'x'; ceiling]),
            ttl_seconds: 300,
            model: model.into(),
            prompt_tokens: 1,
            completion_tokens: 1,
            total_tokens: 2,
        }));
        cache.flush();
        assert!(cache.lookup(&accepted_key).is_some());

        assert!(!cache.insert(CacheWrite {
            key: raw_key("t", "ns", 2),
            body: Bytes::from(vec![b'x'; ceiling + 1]),
            ttl_seconds: 300,
            model: model.into(),
            prompt_tokens: 1,
            completion_tokens: 1,
            total_tokens: 2,
        }));
    }

    #[test]
    fn aggregate_queued_bytes_never_exceed_fixed_writer_envelope() {
        let ids: Vec<TenantId> = (0..64)
            .map(|index| tenant(&format!("tenant_{index:02}")))
            .collect();
        let plan = TenantCapacityPlan::new(16 * 1024 * 1024, ids.clone());
        let writer_budget = plan.writer_budget_bytes();
        let (admission, _receivers) = WriteAdmission::new(&plan, WRITE_LANE_CAPACITY);
        let write_drops = AtomicU64::new(0);
        let oversize_drops = AtomicU64::new(0);
        for (tenant_index, tenant_id) in ids.iter().enumerate() {
            for write_index in 0..WRITE_LANE_CAPACITY {
                let mut hash = [0; 32];
                hash[..8].copy_from_slice(
                    &u64::try_from(tenant_index * WRITE_LANE_CAPACITY + write_index)
                        .expect("test index fits")
                        .to_le_bytes(),
                );
                let _ = admission.try_insert(
                    CacheWrite {
                        key: CacheKey {
                            tenant_id: tenant_id.clone(),
                            namespace: "ns".into(),
                            hash,
                        },
                        body: Bytes::from(vec![b'x'; 8 * 1024]),
                        ttl_seconds: 300,
                        model: "m".into(),
                        prompt_tokens: 1,
                        completion_tokens: 1,
                        total_tokens: 2,
                    },
                    1,
                    &write_drops,
                    &oversize_drops,
                );
            }
        }
        assert!(admission.queued_bytes() <= writer_budget);
        assert_eq!(oversize_drops.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn tenant_lanes_isolate_saturation_and_schedule_round_robin() {
        let plan = TenantCapacityPlan::new(128 * 1024, tenants(&["tenant_b", "tenant_a"]));
        let (admission, mut receivers) = WriteAdmission::new(&plan, 2);
        let write_drops = AtomicU64::new(0);
        let oversize_drops = AtomicU64::new(0);

        assert!(admission.try_insert(
            write(raw_key("tenant_a", "ns", 0)),
            1,
            &write_drops,
            &oversize_drops,
        ));
        assert!(admission.try_insert(
            write(raw_key("tenant_a", "ns", 1)),
            2,
            &write_drops,
            &oversize_drops,
        ));
        assert!(admission.try_insert(
            write(raw_key("tenant_b", "ns", 2)),
            3,
            &write_drops,
            &oversize_drops,
        ));
        assert!(!admission.try_insert(
            write(raw_key("tenant_a", "ns", 3)),
            4,
            &write_drops,
            &oversize_drops,
        ));
        assert!(!admission.try_insert(
            write(raw_key("unknown", "ns", 4)),
            5,
            &write_drops,
            &oversize_drops,
        ));
        assert_eq!(write_drops.load(Ordering::Relaxed), 2);
        assert_eq!(oversize_drops.load(Ordering::Relaxed), 0);

        let mut served = Vec::new();
        for _ in 0..3 {
            let (lane_index, op) = receivers.try_next().expect("queued write");
            match op {
                LaneOp::Insert(insert) => served.push(insert.key.tenant_id.clone()),
                LaneOp::Barrier(_) => panic!("no barrier queued"),
            }
            receivers.finish_turn(lane_index);
        }
        assert_eq!(
            served,
            [tenant("tenant_a"), tenant("tenant_b"), tenant("tenant_a")]
        );
    }

    #[test]
    fn newly_rearmed_lane_cannot_jump_an_existing_local_backlog() {
        let plan = TenantCapacityPlan::new(256 * 1024, tenants(&["tenant_a", "tenant_b"]));
        let (admission, mut receivers) = WriteAdmission::new(&plan, 4);
        let write_drops = AtomicU64::new(0);
        let oversize_drops = AtomicU64::new(0);

        for index in 0..3 {
            assert!(admission.try_insert(
                write(raw_key("tenant_b", "ns", index)),
                u64::try_from(index).expect("test index fits"),
                &write_drops,
                &oversize_drops,
            ));
        }
        assert!(admission.try_insert(
            write(raw_key("tenant_a", "ns", 10)),
            10,
            &write_drops,
            &oversize_drops,
        ));

        let (lane_b, first_b) = receivers.try_next().expect("B starts first");
        drop(first_b);
        receivers.finish_turn(lane_b);
        let (lane_a, first_a) = receivers.try_next().expect("A receives its turn");
        drop(first_a);
        receivers.finish_turn(lane_a);

        // B is already backlogged in the writer-owned FIFO. A producer now
        // rearms A just in time through the global ready queue. A must be
        // appended behind B, never jump it.
        assert!(admission.try_insert(
            write(raw_key("tenant_a", "ns", 11)),
            11,
            &write_drops,
            &oversize_drops,
        ));
        let (next_lane, next) = receivers.try_next().expect("backlogged B remains ready");
        match next {
            LaneOp::Insert(insert) => {
                assert_eq!(insert.key.tenant_id, tenant("tenant_b"));
            }
            LaneOp::Barrier(_) => panic!("expected insert"),
        }
        receivers.finish_turn(next_lane);

        let (_, following) = receivers.try_next().expect("rearmed A follows B");
        match following {
            LaneOp::Insert(insert) => {
                assert_eq!(insert.key.tenant_id, tenant("tenant_a"));
            }
            LaneOp::Barrier(_) => panic!("expected insert"),
        }
    }

    #[test]
    fn coalesced_and_stale_ready_tokens_do_not_lose_future_work() {
        let plan = TenantCapacityPlan::new(128 * 1024, tenants(&["tenant_a", "tenant_b"]));
        let (admission, mut receivers) = WriteAdmission::new(&plan, 4);
        let write_drops = AtomicU64::new(0);
        let oversize_drops = AtomicU64::new(0);

        assert!(admission.try_insert(
            write(raw_key("tenant_a", "ns", 0)),
            1,
            &write_drops,
            &oversize_drops,
        ));
        let (lane_a, first) = receivers.try_next().expect("first A write");
        drop(first);

        // The scheduled bit coalesces this second write while A's first turn is
        // still active. finish_turn must discover and locally requeue it.
        assert!(admission.try_insert(
            write(raw_key("tenant_a", "ns", 1)),
            2,
            &write_drops,
            &oversize_drops,
        ));
        receivers.finish_turn(lane_a);
        let (lane_a_again, second) = receivers.try_next().expect("coalesced A write");
        drop(second);
        receivers.finish_turn(lane_a_again);

        // Inject the only stale-token shape the receiver tolerates: a ready
        // signal whose lane has already drained. It must be discarded without
        // preventing a later producer from scheduling another tenant.
        admission
            .ready
            .try_send(lane_a)
            .expect("ready queue has space");
        assert!(receivers.try_next().is_none());
        assert!(admission.try_insert(
            write(raw_key("tenant_b", "ns", 2)),
            3,
            &write_drops,
            &oversize_drops,
        ));
        let (_, third) = receivers.try_next().expect("B remains schedulable");
        match third {
            LaneOp::Insert(insert) => assert_eq!(insert.key.tenant_id, tenant("tenant_b")),
            LaneOp::Barrier(_) => panic!("expected insert"),
        }
    }

    #[test]
    fn disconnected_receiver_releases_reservations_and_flush_returns() {
        let plan = TenantCapacityPlan::new(128 * 1024, tenants(&["tenant_a"]));
        let (admission, receivers) = WriteAdmission::new(&plan, 1);
        drop(receivers);
        let write_drops = AtomicU64::new(0);
        let oversize_drops = AtomicU64::new(0);

        assert!(!admission.try_insert(
            write(raw_key("tenant_a", "ns", 0)),
            1,
            &write_drops,
            &oversize_drops,
        ));
        assert_eq!(admission.queued_bytes(), 0);
        admission.flush();
        assert_eq!(write_drops.load(Ordering::Relaxed), 1);
    }

    // --- structural isolation (FR-7 / AC-3) ------------------------------------

    #[test]
    fn tenant_isolation_is_structural() {
        let clock = FakeClock::at(1_000);
        let core = CacheCore::new(1024 * 1024, tenants(&["tenant_a", "tenant_b"]), clock);
        let req = req_from_json(r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#);
        let c = chain(&["openai"]);
        let key_a = exact_key("tenant_a", "default", &req, &c);
        let key_b = exact_key("tenant_b", "default", &req, &c);
        assert_ne!(key_a, key_b);
        core.apply_insert(key_a.clone(), entry(10, 1_000, 60_000));
        assert!(core.lookup(&key_a).is_some());
        // Identical request + identical namespace string, different tenant → MISS.
        assert!(core.lookup(&key_b).is_none());
    }

    #[test]
    fn hit_miss_counters_and_snapshot() {
        let clock = FakeClock::at(1_000);
        let core = CacheCore::new(
            1024 * 1024,
            tenants(&["t"]),
            Arc::clone(&clock) as Arc<dyn Clock>,
        );
        let key = raw_key("t", "default", 7);
        // Miss before insert.
        assert!(core.lookup(&key).is_none());
        assert_eq!((core.hits(), core.misses()), (0, 1));
        core.apply_insert(key.clone(), entry(10, 1_000, 60_000));
        // Hit after insert.
        assert!(core.lookup(&key).is_some());
        assert_eq!((core.hits(), core.misses()), (1, 1));
        let (entries, bytes) = core.stats_snapshot();
        assert_eq!(entries, 1);
        assert!(bytes > 0);
    }

    #[test]
    fn namespace_partitions_within_a_tenant() {
        let clock = FakeClock::at(1_000);
        let core = CacheCore::new(1024 * 1024, tenants(&["t"]), clock);
        let req = req_from_json(r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#);
        let c = chain(&["openai"]);
        let prod = exact_key("t", "prod", &req, &c);
        let dev = exact_key("t", "dev", &req, &c);
        core.apply_insert(prod.clone(), entry(10, 1_000, 60_000));
        assert!(core.lookup(&prod).is_some());
        assert!(core.lookup(&dev).is_none());
    }

    // --- TTL (FR-17 / AC-4) ------------------------------------------------------

    #[test]
    fn ttl_expiry_is_a_miss_with_injectable_clock() {
        let clock = FakeClock::at(1_000);
        let core = CacheCore::new(
            1024 * 1024,
            tenants(&["t"]),
            Arc::clone(&clock) as Arc<dyn Clock>,
        );
        let key = raw_key("t", "default", 0);
        core.apply_insert(key.clone(), entry(10, 1_000, 5_000));
        assert!(core.lookup(&key).is_some(), "fresh entry hits");
        clock.set(5_999);
        assert!(core.lookup(&key).is_some(), "still inside TTL");
        clock.set(6_000);
        assert!(core.lookup(&key).is_none(), "expired at inserted_at + ttl");
    }

    // --- eviction: TTL-first, then FIFO; budget enforced (NFR-3) ----------------

    #[test]
    fn eviction_drops_expired_first_then_oldest_inserted() {
        // 2 KiB total => 1 KiB storage for this tenant. Entry cost = 466 bytes.
        let clock = FakeClock::at(10_000);
        let core = CacheCore::new(
            2 * 1024,
            tenants(&["t"]),
            Arc::clone(&clock) as Arc<dyn Clock>,
        );
        // Same shard (hash[0] = 0) for all three keys.
        let expired = raw_key("t", "ns", 0);
        let old = raw_key("t", "ns", 1);
        let new = raw_key("t", "ns", 2);
        // An expired entry (inserted long ago, tiny TTL) + a live old entry.
        core.apply_insert(expired.clone(), entry(300, 1_000, 10));
        core.apply_insert(old.clone(), entry(300, 2_000, 600_000));
        // Inserting a third 300-byte entry pushes the shard over 1024 bytes:
        // the EXPIRED entry must go first, sparing the live old one.
        core.apply_insert(new.clone(), entry(300, 10_000, 600_000));
        assert!(core.lookup(&expired).is_none());
        assert!(
            core.lookup(&old).is_some(),
            "live entry survives TTL-first pass"
        );
        assert!(core.lookup(&new).is_some());

        // Now a fourth live entry exceeds the budget again with nothing expired:
        // FIFO evicts the OLDEST-inserted live entry.
        let newest = raw_key("t", "ns", 3);
        core.apply_insert(newest.clone(), entry(300, 10_001, 600_000));
        assert!(
            core.lookup(&old).is_none(),
            "oldest-inserted evicted (FIFO)"
        );
        assert!(core.lookup(&new).is_some());
        assert!(core.lookup(&newest).is_some());
    }

    #[test]
    fn noisy_tenant_evicts_only_its_own_entries() {
        // 4 KiB total => 2 KiB storage => 1 KiB non-borrowable storage for each
        // of two tenants. Each entry costs 466 bytes.
        let clock = FakeClock::at(10_000);
        let core = CacheCore::new(4 * 1024, tenants(&["tenant_a", "tenant_b"]), clock);
        let quiet = raw_key("tenant_b", "ns", 0);
        core.apply_insert(quiet.clone(), entry(300, 1_000, 600_000));

        let a_old = raw_key("tenant_a", "ns", 1);
        let a_new = raw_key("tenant_a", "ns", 2);
        let a_newest = raw_key("tenant_a", "ns", 3);
        core.apply_insert(a_old.clone(), entry(300, 2_000, 600_000));
        core.apply_insert(a_new.clone(), entry(300, 3_000, 600_000));
        core.apply_insert(a_newest.clone(), entry(300, 4_000, 600_000));

        assert!(
            core.lookup(&quiet).is_some(),
            "tenant B cannot be evicted by A"
        );
        assert!(core.lookup(&a_old).is_none(), "A's oldest entry is evicted");
        assert!(core.lookup(&a_new).is_some());
        assert!(core.lookup(&a_newest).is_some());
        let (_, bytes) = core.stats_snapshot();
        assert!(bytes <= 2 * 1024, "the storage byte cap remains hard");
    }

    #[test]
    fn oversize_entry_is_never_stored_and_is_counted() {
        let clock = FakeClock::at(1_000);
        let core = CacheCore::new(DEFAULT_BUDGET_BYTES, tenants(&["t"]), clock);
        let key = raw_key("t", "ns", 0);
        core.apply_insert(key.clone(), entry(MAX_ENTRY_BYTES + 1, 1_000, 60_000));
        assert!(core.lookup(&key).is_none());
        assert_eq!(core.oversize_drops(), 1);
    }

    // --- write-behind handle ------------------------------------------------------

    #[test]
    fn write_behind_insert_is_visible_after_flush() {
        let clock = FakeClock::at(1_000);
        let cache = ExactCache::with_clock(1024 * 1024, tenants(&["t"]), clock);
        let req = req_from_json(r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#);
        let key = exact_key("t", "default", &req, &chain(&["openai"]));
        cache.insert(CacheWrite {
            key: key.clone(),
            body: Bytes::from_static(b"{\"ok\":true}"),
            ttl_seconds: 300,
            model: "m".into(),
            prompt_tokens: 1,
            completion_tokens: 1,
            total_tokens: 2,
        });
        cache.flush();
        let got = cache.lookup(&key).expect("entry visible after barrier");
        assert_eq!(&got.body[..], b"{\"ok\":true}");
        assert_eq!(got.total_tokens, 2);
    }

    #[test]
    fn multi_tenant_writer_flush_services_quiet_lane_and_releases_reservations() {
        let clock = FakeClock::at(1_000);
        let cache = ExactCache::with_clock(1024 * 1024, tenants(&["tenant_a", "tenant_b"]), clock);

        for index in 0..WRITE_LANE_CAPACITY {
            cache.insert(write(raw_key("tenant_a", "ns", index)));
        }
        let quiet_key = raw_key("tenant_b", "ns", 200);
        cache.insert(write(quiet_key.clone()));

        cache.flush();

        assert!(
            cache.lookup(&quiet_key).is_some(),
            "tenant B's writer lane must make progress despite tenant A's burst"
        );
        assert_eq!(
            cache.queued_bytes(),
            0,
            "the barrier must observe every reservation released"
        );
    }

    #[test]
    fn concurrent_producers_complete_a_bounded_writer_barrier() {
        let cache = Arc::new(ExactCache::with_clock(
            2 * 1024 * 1024,
            tenants(&["tenant_a", "tenant_b"]),
            FakeClock::at(1_000),
        ));
        let mut producers = Vec::new();
        for producer_index in 0..8usize {
            let cache = Arc::clone(&cache);
            producers.push(std::thread::spawn(move || {
                let tenant_id = if producer_index % 2 == 0 {
                    "tenant_a"
                } else {
                    "tenant_b"
                };
                for write_index in 0..64usize {
                    let hash_index = producer_index * 64 + write_index;
                    cache.insert(write(raw_key(tenant_id, "ns", hash_index)));
                }
            }));
        }
        for producer in producers {
            producer.join().expect("producer must not panic");
        }

        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let barrier_cache = Arc::clone(&cache);
        let barrier = std::thread::spawn(move || {
            barrier_cache.flush();
            done_tx.send(()).expect("test receiver remains live");
        });
        done_rx
            .recv_timeout(std::time::Duration::from_secs(2))
            .expect("writer barrier must finish without a lost wakeup or deadlock");
        barrier.join().expect("barrier thread must not panic");
        assert_eq!(cache.queued_bytes(), 0);
    }

    #[test]
    fn configured_envelope_bounds_stored_and_queued_response_bytes_together() {
        let budget = 2 * 1024 * 1024;
        let cache = ExactCache::with_clock(
            budget,
            tenants(&["tenant_a", "tenant_b"]),
            FakeClock::at(1_000),
        );
        for index in 0..256 {
            let tenant_id = if index % 2 == 0 {
                "tenant_a"
            } else {
                "tenant_b"
            };
            let _ = cache.insert(CacheWrite {
                key: raw_key(tenant_id, "ns", index),
                body: Bytes::from(vec![b'x'; 16 * 1024]),
                ttl_seconds: 300,
                model: "m".into(),
                prompt_tokens: 1,
                completion_tokens: 1,
                total_tokens: 2,
            });
            let (_, stored_bytes) = cache.stats_snapshot();
            assert!(
                stored_bytes.saturating_add(cache.queued_bytes()) <= budget,
                "stored and queued response retention must share one fixed envelope"
            );
        }
        cache.flush();
        let (_, stored_bytes) = cache.stats_snapshot();
        assert!(stored_bytes <= cache.storage_budget_bytes());
        assert_eq!(cache.queued_bytes(), 0);
    }

    #[test]
    fn large_registry_writer_flush_and_joined_shutdown_are_bounded() {
        let tenant_ids: Vec<TenantId> = (0..1_000)
            .map(|index| tenant(&format!("tenant_{index:04}")))
            .collect();
        let last = tenant_ids
            .last()
            .expect("large registry is non-empty")
            .clone();
        let cache = ExactCache::with_clock(DEFAULT_BUDGET_BYTES, tenant_ids, FakeClock::at(1_000));
        assert!(cache.insert(write(CacheKey {
            tenant_id: last,
            namespace: "ns".into(),
            hash: [7; 32],
        })));

        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let shutdown = std::thread::spawn(move || {
            cache.flush();
            cache.shutdown();
            done_tx.send(()).expect("test receiver remains live");
        });
        done_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("large-registry writer flush and shutdown must be bounded");
        shutdown.join().expect("shutdown thread must not panic");
    }

    #[test]
    fn empty_tenant_registry_flush_is_a_noop() {
        let cache = ExactCache::with_clock(1024, Vec::new(), FakeClock::at(1_000));
        cache.flush();
        assert_eq!(cache.tenant_count(), 0);
        assert_eq!(cache.queued_bytes(), 0);
    }

    #[test]
    fn unknown_and_zero_share_tenants_are_storage_inert() {
        let cache = ExactCache::with_clock(1024, tenants(&["known"]), FakeClock::at(1_000));
        assert!(
            !cache.insert(write(raw_key("unknown", "ns", 1))),
            "an unregistered tenant must not acquire a lane dynamically"
        );

        // An uneven fixed split can leave one tenant with a positive one-byte
        // payload share and a neighbor with zero. The latter must not poison
        // the positive tenant's ceiling or borrow its capacity.
        let zero_share = ExactCache::with_clock(
            642,
            tenants(&["tenant_a", "tenant_b"]),
            FakeClock::at(1_000),
        );
        let tenant_a = tenant("tenant_a");
        let tenant_b = tenant("tenant_b");
        assert_eq!(zero_share.max_response_bytes_for_tenant(&tenant_a, 0), 1);
        assert_eq!(zero_share.max_response_bytes_for_tenant(&tenant_b, 0), 0);
        assert!(zero_share.owns_tenant(&tenant_a));
        assert!(!zero_share.owns_tenant(&tenant_b));
        assert!(!zero_share.insert(write(raw_key("tenant_b", "ns", 2))));
        assert!(zero_share.insert(CacheWrite {
            key: raw_key("tenant_a", "ns", 3),
            body: Bytes::from_static(b"x"),
            ttl_seconds: 300,
            model: String::new(),
            prompt_tokens: 1,
            completion_tokens: 1,
            total_tokens: 2,
        }));
        zero_share.flush();
        assert_eq!(zero_share.stats_snapshot(), (1, KEY_OVERHEAD_BYTES + 1));
    }

    // --- flush generations (PRD-007 FR-19) ------------------------------------

    #[test]
    fn gen_zero_key_is_byte_identical_to_legacy_key() {
        // The golden/ab_parity byte-identity proof: exact_key (the legacy form)
        // MUST equal exact_key_gen(.., 0). The struct derives Eq over all fields
        // including the 32-byte hash, so equality here is byte-level identity of
        // the digest.
        let req = req_from_json(
            r#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}],"temperature":0.5,"max_tokens":64}"#,
        );
        let c = chain(&["openai", "anthropic"]);
        let legacy = exact_key("t", "default", &req, &c);
        let gen0 = exact_key_gen("t", "default", &req, &c, 0);
        assert_eq!(legacy, gen0);
        assert_eq!(legacy.hash, gen0.hash, "digest is byte-identical at gen 0");
    }

    #[test]
    fn bumping_generation_changes_the_derived_key() {
        let req = req_from_json(r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#);
        let c = chain(&["openai"]);
        let g0 = exact_key_gen("t", "ns", &req, &c, 0);
        let g1 = exact_key_gen("t", "ns", &req, &c, 1);
        let g2 = exact_key_gen("t", "ns", &req, &c, 2);
        assert_ne!(g0, g1, "a purge (gen 1) makes the prior key unreachable");
        assert_ne!(g1, g2);
        assert_ne!(g0, g2);
    }

    #[test]
    fn purge_makes_a_cached_key_miss_then_fresh_entry_stores_under_new_gen() {
        let clock = FakeClock::at(1_000);
        let core = CacheCore::new(1024 * 1024, tenants(&["t"]), clock);
        let reg = FlushRegistry::new(tenants(&["t"]));
        let tenant_t = tenant("t");
        let req = req_from_json(r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#);
        let c = chain(&["openai"]);

        // Cache under the current (gen 0) key, then confirm a hit.
        let g0 = reg.generation(&tenant_t, "ns");
        assert_eq!(g0, 0);
        let key0 = exact_key_gen("t", "ns", &req, &c, g0);
        core.apply_insert(key0.clone(), entry(10, 1_000, 60_000));
        assert!(core.lookup(&key0).is_some());

        // Purge → bump → the new-generation key MISSES (the old entry is orphaned
        // and will age out via TTL/FIFO).
        let g1 = reg.bump(&tenant_t, "ns");
        assert_eq!(g1, Ok(1));
        assert_eq!(reg.generation(&tenant_t, "ns"), 1);
        let key1 = exact_key_gen("t", "ns", &req, &c, reg.generation(&tenant_t, "ns"));
        assert!(core.lookup(&key1).is_none(), "purged key misses");

        // A fresh entry stores under the new generation and is reachable.
        core.apply_insert(key1.clone(), entry(10, 1_000, 60_000));
        assert!(core.lookup(&key1).is_some());
    }

    #[test]
    fn purge_is_tenant_and_namespace_scoped() {
        let reg = FlushRegistry::new(tenants(&["tenant_a", "tenant_b"]));
        let tenant_a = tenant("tenant_a");
        let tenant_b = tenant("tenant_b");
        assert_eq!(reg.bump(&tenant_a, "ns"), Ok(1));
        // Tenant A's purge does not touch tenant B (cross-tenant isolation).
        assert_eq!(reg.generation(&tenant_a, "ns"), 1);
        assert_eq!(reg.generation(&tenant_b, "ns"), 0);
        // And it does not touch a different namespace of the same tenant.
        assert_eq!(reg.generation(&tenant_a, "other"), 0);

        let req = req_from_json(r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#);
        let c = chain(&["openai"]);
        // Tenant B's derived key is unchanged by tenant A's purge (still gen 0 =
        // byte-identical to the legacy key).
        let b_key = exact_key_gen("tenant_b", "ns", &req, &c, reg.generation(&tenant_b, "ns"));
        assert_eq!(b_key, exact_key("tenant_b", "ns", &req, &c));
    }

    #[test]
    fn repeated_purges_monotonically_increase_generation() {
        let reg = FlushRegistry::new(tenants(&["t"]));
        let tenant_t = tenant("t");
        assert_eq!(reg.bump(&tenant_t, "ns"), Ok(1));
        assert_eq!(reg.bump(&tenant_t, "ns"), Ok(2));
        assert_eq!(reg.bump(&tenant_t, "ns"), Ok(3));
        assert_eq!(reg.generation(&tenant_t, "ns"), 3);
        assert_eq!(reg.purged_scope_count(), 1);
    }

    #[test]
    fn purge_rejects_tenants_outside_the_startup_authority_registry() {
        let reg = FlushRegistry::new(tenants(&["tenant_a"]));
        assert_eq!(
            reg.bump(&tenant("tenant_b"), "ns"),
            Err(FlushError::UnknownTenant)
        );
        assert_eq!(reg.purged_scope_count(), 0);
    }

    #[test]
    fn purge_rejects_invalid_namespaces_without_retaining_them() {
        let reg = FlushRegistry::new(tenants(&["tenant_a"]));
        let tenant_a = tenant("tenant_a");
        let too_long = "x".repeat(65);
        for namespace in ["", "Upper", "bad.dot", "bad/slash", too_long.as_str()] {
            assert_eq!(
                reg.bump(&tenant_a, namespace),
                Err(FlushError::InvalidNamespace)
            );
            assert_eq!(reg.generation(&tenant_a, namespace), 0);
        }
        assert_eq!(reg.purged_scope_count(), 0);
        assert_eq!(reg.bump(&tenant_a, WILDCARD_NAMESPACE), Ok(1));
    }

    #[test]
    fn concurrent_purges_preserve_every_generation_increment() {
        let reg = Arc::new(FlushRegistry::new(tenants(&["tenant_a", "tenant_b"])));
        let mut workers = Vec::new();
        for _ in 0..8 {
            let reg = Arc::clone(&reg);
            workers.push(std::thread::spawn(move || {
                for _ in 0..100 {
                    reg.bump(&tenant("tenant_a"), "ns")
                        .expect("registered tenant purge succeeds");
                }
            }));
        }
        for worker in workers {
            worker.join().expect("purge worker must not panic");
        }
        assert_eq!(reg.generation(&tenant("tenant_a"), "ns"), 800);
        assert_eq!(reg.bump(&tenant("tenant_b"), "ns"), Ok(1));
        assert_eq!(reg.generation(&tenant("tenant_b"), "ns"), 1);
    }

    #[test]
    fn purge_scope_registry_is_bounded_per_tenant() {
        let reg = FlushRegistry::new(tenants(&["tenant_a", "tenant_b"]));
        let tenant_a = tenant("tenant_a");
        let tenant_b = tenant("tenant_b");
        for index in 0..(MAX_PURGED_SCOPES_PER_TENANT - 1) {
            assert_eq!(
                reg.bump(&tenant_a, &format!("ns_{index}")),
                Ok(1),
                "tenant A scope {index} fits"
            );
        }
        assert_eq!(
            reg.bump(&tenant_a, "reserved_for_wildcard"),
            Err(FlushError::ScopeLimit)
        );
        assert_eq!(reg.bump(&tenant_a, WILDCARD_NAMESPACE), Ok(1));
        assert_eq!(reg.bump(&tenant_a, WILDCARD_NAMESPACE), Ok(2));
        assert_eq!(reg.bump(&tenant_a, "ns_0"), Ok(2));
        assert_eq!(reg.bump(&tenant_b, "own_scope"), Ok(1));
        assert_eq!(reg.purged_scope_count(), MAX_PURGED_SCOPES_PER_TENANT + 1);
    }

    #[test]
    fn flush_all_folds_into_every_namespace_effective_generation() {
        // Regression for the silent flush-all no-op: a no-namespace ("flush-all")
        // purge bumps ONLY the reserved wildcard scope. `generation_effective`
        // must fold that into EVERY namespace's effective generation, so the
        // bump actually invalidates a namespaced key. (Before the fold the
        // wildcard generation was dead — no read path ever consulted it.)
        let reg = FlushRegistry::new(tenants(&["t", "u"]));
        let tenant_t = tenant("t");
        let tenant_u = tenant("u");
        // Un-purged: effective generation is 0 for every namespace (⇒ the
        // byte-identical legacy gen-0 key).
        assert_eq!(reg.generation_effective(&tenant_t, "default"), 0);
        assert_eq!(reg.generation_effective(&tenant_t, "other"), 0);

        // Flush-all: bump the tenant-wide wildcard scope.
        assert_eq!(reg.bump(&tenant_t, WILDCARD_NAMESPACE), Ok(1));
        // EVERY namespace of this tenant now sees effective generation 1.
        assert_eq!(reg.generation_effective(&tenant_t, "default"), 1);
        assert_eq!(reg.generation_effective(&tenant_t, "other"), 1);
        // Reading the wildcard scope itself does not double-count.
        assert_eq!(reg.generation_effective(&tenant_t, WILDCARD_NAMESPACE), 1);
        // Cross-tenant isolation: a different tenant is untouched.
        assert_eq!(reg.generation_effective(&tenant_u, "default"), 0);

        // A namespace-specific purge STILL strictly increases that namespace's
        // effective generation — SUM, not max, so it is never swallowed by an
        // equal wildcard generation.
        assert_eq!(reg.bump(&tenant_t, "default"), Ok(1)); // "default"'s own gen → 1
        assert_eq!(reg.generation_effective(&tenant_t, "default"), 2); // 1 (ns) + 1 (wildcard)
        assert_eq!(reg.generation_effective(&tenant_t, "other"), 1); // "other"'s own gen still 0
    }

    #[test]
    fn flush_all_changes_a_namespaced_derived_key() {
        // The end the fold serves: a flush-all must change the DERIVED KEY of a
        // concrete namespace (the actual invalidation mechanism), not merely a
        // counter.
        let reg = FlushRegistry::new(tenants(&["z"]));
        let tenant_z = tenant("z");
        let req = req_from_json(r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#);
        let c = chain(&["openai"]);
        let before = exact_key_gen(
            "z",
            "default",
            &req,
            &c,
            reg.generation_effective(&tenant_z, "default"),
        );
        // gen-0 effective key is byte-identical to the legacy key.
        assert_eq!(before, exact_key("z", "default", &req, &c));

        assert_eq!(reg.bump(&tenant_z, WILDCARD_NAMESPACE), Ok(1)); // flush-all for tenant z
        let after = exact_key_gen(
            "z",
            "default",
            &req,
            &c,
            reg.generation_effective(&tenant_z, "default"),
        );
        assert_ne!(
            before, after,
            "flush-all changes a namespaced key (not a no-op)"
        );
    }

    #[test]
    fn cache_status_wire_forms_match_prd() {
        assert_eq!(CacheStatus::Hit.header_value(), "hit");
        assert_eq!(CacheStatus::SemanticHit.header_value(), "semantic-hit");
        assert_eq!(CacheStatus::SemanticHit.event_value(), "semantic_hit");
        assert_eq!(CacheStatus::Bypass.event_value(), "bypass");
        assert_eq!(CacheStatus::Refreshed.header_value(), "refreshed");
        assert_eq!(CacheStatus::Miss.event_value(), "miss");
    }
}
