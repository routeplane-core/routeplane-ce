//! Rung-0 exact-match response cache (G2.5 — [PRD-007] behavior, [ADR-022]
//! mechanics). In-memory, per-replica, zero standing cost, zero hot-path locks.
//!
//! - **Read path (hot)**: one SHA-256 over the shaping-resolved request (done by
//!   the caller via [`exact_key`]) + one `ArcSwap::load` + one `HashMap` probe +
//!   a TTL check. No lock, no CAS contention.
//! - **Write path (off-path)**: [`ExactCache::insert`] is a bounded `try_send`
//!   to a single dedicated writer thread; the writer clones the shard map,
//!   inserts, evicts (TTL-first, then FIFO by insert time) and publishes the new
//!   `Arc`. Readers never wait; a full channel drops the write (counted) rather
//!   than ever blocking a request.
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
use routeplane_types::{ChatCompletionRequest, Message, Tool};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, SyncSender};
use std::sync::Arc;
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
/// Bound on the write channel: a stalled writer or a pathological write burst
/// sheds inserts (counted) instead of growing memory or blocking requests.
const WRITE_CHANNEL_CAPACITY: usize = 1024;

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
    tenant_id: String,
    namespace: String,
    hash: [u8; 32],
}

impl CacheKey {
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
    tenant_id: &str,
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
    tenant_id: &str,
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
        tenant_id: tenant_id.to_string(),
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
/// - **Write path (purge, rare)**: copy-on-write — clone the map, bump (or
///   insert) the one entry, `store` the new `Arc`. A `compare_and_swap` retry
///   loop makes concurrent purges to different scopes safe without ever blocking
///   a reader.
///
/// Per-replica, like the cache itself (ADR-022 §3): a purge clears THIS replica's
/// view; multi-replica coordinated purge is a documented follow-on, consistent
/// with the per-replica cache posture (no Redis here — that is a trigger-gated
/// rung, ADR-022 §1).
#[derive(Default)]
pub struct FlushRegistry {
    // Key is (tenant_id, namespace). The map is small (one entry per purged
    // scope, only growing on the rare purge path), cloned wholesale on each
    // copy-on-write swap — cheap because purges are rare and the map is tiny.
    generations: ArcSwap<HashMap<(String, String), u64>>,
}

impl FlushRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Wait-free read of the current generation for a `(tenant, namespace)`
    /// scope. Absent ⇒ 0 (the never-purged default → byte-identical legacy key).
    /// One `ArcSwap::load` + one map probe; safe from any thread, never blocks.
    pub fn generation(&self, tenant_id: &str, namespace: &str) -> u64 {
        let guard = self.generations.load();
        // Borrow the tuple parts to avoid allocating a `(String, String)` probe
        // key on the hot read path.
        guard
            .get(&(tenant_id.to_string(), namespace.to_string()))
            .copied()
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
    pub fn generation_effective(&self, tenant_id: &str, namespace: &str) -> u64 {
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
    pub fn bump(&self, tenant_id: &str, namespace: &str) -> u64 {
        let scope = (tenant_id.to_string(), namespace.to_string());
        loop {
            let current = self.generations.load();
            let next = current.get(&scope).copied().unwrap_or(0).saturating_add(1);
            let mut map: HashMap<(String, String), u64> = (**current).clone();
            map.insert(scope.clone(), next);
            let new = Arc::new(map);
            let prev = self
                .generations
                .compare_and_swap(&*current, Arc::clone(&new));
            // `compare_and_swap` returns the value that was in place; the swap
            // succeeded iff it is pointer-equal to what we loaded.
            if Arc::ptr_eq(&prev, &current) {
                return next;
            }
            // Lost the race to a concurrent purge of another (or the same) scope;
            // retry with the fresh snapshot. Readers were never blocked.
        }
    }

    /// Number of distinct purged scopes (diagnostics/`/status`; off the hot path).
    pub fn purged_scope_count(&self) -> usize {
        self.generations.load().len()
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

/// The sharded store. `lookup` is safe from any thread (lock-free);
/// `apply_insert` must only ever be called from the single writer (the
/// [`ExactCache`] writer thread in production, the test body in unit tests).
pub struct CacheCore {
    shards: Vec<ArcSwap<ShardMap>>,
    budget_per_shard: usize,
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
    pub fn new(budget_bytes: usize, clock: Arc<dyn Clock>) -> Self {
        Self {
            shards: (0..SHARD_COUNT)
                .map(|_| ArcSwap::from_pointee(ShardMap::new()))
                .collect(),
            budget_per_shard: (budget_bytes / SHARD_COUNT).max(1),
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
    /// FIFO by insert time, until the shard is back under its byte budget.
    /// An entry over the per-entry cap (or over the whole shard budget by
    /// itself) is dropped and counted, never stored.
    pub fn apply_insert(&self, key: CacheKey, entry: Arc<CacheEntry>) {
        let cost = entry.cost();
        if cost > MAX_ENTRY_BYTES || cost > self.budget_per_shard {
            self.oversize_drops.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let idx = key.shard();
        let now = self.clock.now_ms();
        let current = self.shards[idx].load();
        let mut map: ShardMap = (**current).clone();
        map.insert(key, entry);

        let mut total: usize = map.values().map(|e| e.cost()).sum();
        if total > self.budget_per_shard {
            // 1) TTL-first: drop everything already expired.
            map.retain(|_, e| {
                if e.expired(now) {
                    total -= e.cost();
                    false
                } else {
                    true
                }
            });
        }
        if total > self.budget_per_shard {
            // 2) FIFO by insert time (approximate-LRU; ADR-022 §2).
            let mut by_age: Vec<(CacheKey, u64, usize)> = map
                .iter()
                .map(|(k, e)| (k.clone(), e.inserted_at_ms, e.cost()))
                .collect();
            by_age.sort_by_key(|(_, inserted, _)| *inserted);
            for (k, _, c) in by_age {
                if total <= self.budget_per_shard {
                    break;
                }
                if map.remove(&k).is_some() {
                    total -= c;
                }
            }
        }
        self.shards[idx].store(Arc::new(map));
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
}

// --- Public handle (write-behind via a single writer thread) ---------------------

/// One write-behind insert (kept as a struct so the call site stays readable
/// and under the clippy argument bound).
#[derive(Debug)]
pub struct CacheWrite {
    pub key: CacheKey,
    pub body: Bytes,
    pub ttl_seconds: u64,
    pub model: String,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

enum WriteOp {
    Insert(CacheKey, Arc<CacheEntry>),
    /// Test/diagnostic barrier: acked once every previously-queued write has
    /// been applied. Never used on the request path.
    Barrier(SyncSender<()>),
}

/// The process-wide exact-match cache handle held by `AppState`.
///
/// Reads go straight to the core (lock-free). Writes are a bounded `try_send`
/// to ONE dedicated OS writer thread — a deliberate simplification of
/// ADR-022 §2's per-shard writer tasks with identical reader guarantees (see
/// the PR body): writers are serialized globally, readers never wait, and the
/// cache crate stays free of any async-runtime dependency.
pub struct ExactCache {
    core: Arc<CacheCore>,
    tx: SyncSender<WriteOp>,
}

impl ExactCache {
    /// Production constructor (system clock). Spawns the writer thread; called
    /// once at startup (and per test), never on a request path.
    pub fn new(budget_bytes: usize) -> Self {
        Self::with_clock(budget_bytes, Arc::new(SystemClock))
    }

    /// Test constructor with an injectable clock.
    pub fn with_clock(budget_bytes: usize, clock: Arc<dyn Clock>) -> Self {
        let core = Arc::new(CacheCore::new(budget_bytes, clock));
        let (tx, rx) = sync_channel::<WriteOp>(WRITE_CHANNEL_CAPACITY);
        let writer_core = Arc::clone(&core);
        // Startup-time spawn (not the request path): a failure to create the
        // writer thread here is an unrecoverable process-level condition.
        std::thread::Builder::new()
            .name("rp-cache-writer".to_string())
            .spawn(move || {
                while let Ok(op) = rx.recv() {
                    match op {
                        WriteOp::Insert(key, entry) => writer_core.apply_insert(key, entry),
                        WriteOp::Barrier(ack) => {
                            let _ = ack.send(());
                        }
                    }
                }
                // All senders dropped (process shutdown) → thread ends.
            })
            .expect("failed to spawn the cache writer thread at startup");
        Self { core, tx }
    }

    /// Lock-free lookup (the hot path).
    pub fn lookup(&self, key: &CacheKey) -> Option<Arc<CacheEntry>> {
        self.core.lookup(key)
    }

    /// Write-behind insert (FR-8 / ADR-022 §4): O(1) bounded `try_send`; the
    /// actual map rebuild happens on the writer thread. A full channel drops
    /// the write (counted) — the cache is an optimization, never a dependency.
    pub fn insert(&self, write: CacheWrite) {
        let entry = Arc::new(CacheEntry {
            body: write.body,
            model: write.model,
            prompt_tokens: write.prompt_tokens,
            completion_tokens: write.completion_tokens,
            total_tokens: write.total_tokens,
            inserted_at_ms: self.core.now_ms(),
            ttl_ms: write.ttl_seconds.saturating_mul(1000),
        });
        if self.tx.try_send(WriteOp::Insert(write.key, entry)).is_err() {
            self.core.write_drops.fetch_add(1, Ordering::Relaxed);
        }
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
        let (ack_tx, ack_rx) = sync_channel(1);
        if self.tx.send(WriteOp::Barrier(ack_tx)).is_ok() {
            let _ = ack_rx.recv();
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

    fn raw_key(tenant: &str, ns: &str, first_byte: u8) -> CacheKey {
        let mut hash = [0u8; 32];
        hash[0] = first_byte;
        hash[1] = first_byte; // differentiate keys with the same shard byte
        CacheKey {
            tenant_id: tenant.into(),
            namespace: ns.into(),
            hash,
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

    // --- structural isolation (FR-7 / AC-3) ------------------------------------

    #[test]
    fn tenant_isolation_is_structural() {
        let clock = FakeClock::at(1_000);
        let core = CacheCore::new(1024 * 1024, clock);
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
        let core = CacheCore::new(1024 * 1024, Arc::clone(&clock) as Arc<dyn Clock>);
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
        let core = CacheCore::new(1024 * 1024, clock);
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
        let core = CacheCore::new(1024 * 1024, Arc::clone(&clock) as Arc<dyn Clock>);
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
        // budget 64 KiB → 1024 bytes/shard. Entry cost = body + model(6) + 160.
        let clock = FakeClock::at(10_000);
        let core = CacheCore::new(64 * 1024, Arc::clone(&clock) as Arc<dyn Clock>);
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
    fn oversize_entry_is_never_stored_and_is_counted() {
        let clock = FakeClock::at(1_000);
        let core = CacheCore::new(DEFAULT_BUDGET_BYTES, clock);
        let key = raw_key("t", "ns", 0);
        core.apply_insert(key.clone(), entry(MAX_ENTRY_BYTES + 1, 1_000, 60_000));
        assert!(core.lookup(&key).is_none());
        assert_eq!(core.oversize_drops(), 1);
    }

    // --- write-behind handle ------------------------------------------------------

    #[test]
    fn write_behind_insert_is_visible_after_flush() {
        let clock = FakeClock::at(1_000);
        let cache = ExactCache::with_clock(1024 * 1024, clock);
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
        let core = CacheCore::new(1024 * 1024, clock);
        let reg = FlushRegistry::new();
        let req = req_from_json(r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#);
        let c = chain(&["openai"]);

        // Cache under the current (gen 0) key, then confirm a hit.
        let g0 = reg.generation("t", "ns");
        assert_eq!(g0, 0);
        let key0 = exact_key_gen("t", "ns", &req, &c, g0);
        core.apply_insert(key0.clone(), entry(10, 1_000, 60_000));
        assert!(core.lookup(&key0).is_some());

        // Purge → bump → the new-generation key MISSES (the old entry is orphaned
        // and will age out via TTL/FIFO).
        let g1 = reg.bump("t", "ns");
        assert_eq!(g1, 1);
        assert_eq!(reg.generation("t", "ns"), 1);
        let key1 = exact_key_gen("t", "ns", &req, &c, reg.generation("t", "ns"));
        assert!(core.lookup(&key1).is_none(), "purged key misses");

        // A fresh entry stores under the new generation and is reachable.
        core.apply_insert(key1.clone(), entry(10, 1_000, 60_000));
        assert!(core.lookup(&key1).is_some());
    }

    #[test]
    fn purge_is_tenant_and_namespace_scoped() {
        let reg = FlushRegistry::new();
        reg.bump("tenant_a", "ns");
        // Tenant A's purge does not touch tenant B (cross-tenant isolation).
        assert_eq!(reg.generation("tenant_a", "ns"), 1);
        assert_eq!(reg.generation("tenant_b", "ns"), 0);
        // And it does not touch a different namespace of the same tenant.
        assert_eq!(reg.generation("tenant_a", "other"), 0);

        let req = req_from_json(r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#);
        let c = chain(&["openai"]);
        // Tenant B's derived key is unchanged by tenant A's purge (still gen 0 =
        // byte-identical to the legacy key).
        let b_key = exact_key_gen("tenant_b", "ns", &req, &c, reg.generation("tenant_b", "ns"));
        assert_eq!(b_key, exact_key("tenant_b", "ns", &req, &c));
    }

    #[test]
    fn repeated_purges_monotonically_increase_generation() {
        let reg = FlushRegistry::new();
        assert_eq!(reg.bump("t", "ns"), 1);
        assert_eq!(reg.bump("t", "ns"), 2);
        assert_eq!(reg.bump("t", "ns"), 3);
        assert_eq!(reg.generation("t", "ns"), 3);
        assert_eq!(reg.purged_scope_count(), 1);
    }

    #[test]
    fn flush_all_folds_into_every_namespace_effective_generation() {
        // Regression for the silent flush-all no-op: a no-namespace ("flush-all")
        // purge bumps ONLY the reserved wildcard scope. `generation_effective`
        // must fold that into EVERY namespace's effective generation, so the
        // bump actually invalidates a namespaced key. (Before the fold the
        // wildcard generation was dead — no read path ever consulted it.)
        let reg = FlushRegistry::new();
        // Un-purged: effective generation is 0 for every namespace (⇒ the
        // byte-identical legacy gen-0 key).
        assert_eq!(reg.generation_effective("t", "default"), 0);
        assert_eq!(reg.generation_effective("t", "other"), 0);

        // Flush-all: bump the tenant-wide wildcard scope.
        assert_eq!(reg.bump("t", WILDCARD_NAMESPACE), 1);
        // EVERY namespace of this tenant now sees effective generation 1.
        assert_eq!(reg.generation_effective("t", "default"), 1);
        assert_eq!(reg.generation_effective("t", "other"), 1);
        // Reading the wildcard scope itself does not double-count.
        assert_eq!(reg.generation_effective("t", WILDCARD_NAMESPACE), 1);
        // Cross-tenant isolation: a different tenant is untouched.
        assert_eq!(reg.generation_effective("u", "default"), 0);

        // A namespace-specific purge STILL strictly increases that namespace's
        // effective generation — SUM, not max, so it is never swallowed by an
        // equal wildcard generation.
        assert_eq!(reg.bump("t", "default"), 1); // "default"'s own gen → 1
        assert_eq!(reg.generation_effective("t", "default"), 2); // 1 (ns) + 1 (wildcard)
        assert_eq!(reg.generation_effective("t", "other"), 1); // "other"'s own gen still 0
    }

    #[test]
    fn flush_all_changes_a_namespaced_derived_key() {
        // The end the fold serves: a flush-all must change the DERIVED KEY of a
        // concrete namespace (the actual invalidation mechanism), not merely a
        // counter.
        let reg = FlushRegistry::new();
        let req = req_from_json(r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#);
        let c = chain(&["openai"]);
        let before = exact_key_gen(
            "z",
            "default",
            &req,
            &c,
            reg.generation_effective("z", "default"),
        );
        // gen-0 effective key is byte-identical to the legacy key.
        assert_eq!(before, exact_key("z", "default", &req, &c));

        reg.bump("z", WILDCARD_NAMESPACE); // flush-all for tenant z
        let after = exact_key_gen(
            "z",
            "default",
            &req,
            &c,
            reg.generation_effective("z", "default"),
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
