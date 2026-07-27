use arc_swap::ArcSwap;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

mod strategy;

pub use strategy::{CandidateSpec, ProviderRouting, Rng, Router, RouterConfig, RoutingStrategy};
// `InFlightGuard` is defined and `pub` in this module (the RAII gauge guard used
// by the proxy's attempt loop), so it is already exported at the crate root.

const DEFAULT_FAILURE_THRESHOLD: u64 = 5; // consecutive failures to open
const DEFAULT_SUCCESS_THRESHOLD: u64 = 3; // successes in half-open to close
const DEFAULT_COOLDOWN_MS: u64 = 30_000; // open -> half-open after 30s

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    Closed,
    HalfOpen,
    Open,
}

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// A lock-free circuit breaker. All state lives in atomics, so checking and
/// updating a provider's health never takes a mutex on the request hot path.
///
/// Closed -> (failure_threshold consecutive failures) -> Open
/// Open    -> (cooldown elapsed) -> HalfOpen
/// HalfOpen-> (success_threshold successes) -> Closed
/// HalfOpen-> (any failure) -> Open
pub struct CircuitBreaker {
    state: AtomicU8, // 0=closed, 1=half-open, 2=open
    failures: AtomicU64,
    successes: AtomicU64,
    opened_at: AtomicU64, // unix millis when opened
    failure_threshold: u64,
    success_threshold: u64,
    cooldown_ms: u64,
    now: Box<dyn Fn() -> u64 + Send + Sync>, // injectable clock (unix millis)
}

impl CircuitBreaker {
    pub fn new() -> Self {
        Self::with_config(
            DEFAULT_FAILURE_THRESHOLD,
            DEFAULT_SUCCESS_THRESHOLD,
            DEFAULT_COOLDOWN_MS,
        )
    }

    pub fn with_config(failure_threshold: u64, success_threshold: u64, cooldown_ms: u64) -> Self {
        Self::with_clock(
            failure_threshold,
            success_threshold,
            cooldown_ms,
            Box::new(now_millis),
        )
    }

    /// Construct with an injectable clock — used by tests to drive the cooldown
    /// transition deterministically (no sleeps, no flake).
    pub fn with_clock(
        failure_threshold: u64,
        success_threshold: u64,
        cooldown_ms: u64,
        now: Box<dyn Fn() -> u64 + Send + Sync>,
    ) -> Self {
        Self {
            state: AtomicU8::new(0),
            failures: AtomicU64::new(0),
            successes: AtomicU64::new(0),
            opened_at: AtomicU64::new(0),
            failure_threshold,
            success_threshold,
            cooldown_ms,
            now,
        }
    }

    /// Current state, applying the cooldown transition (Open -> HalfOpen) on read.
    /// Reading self-heals an expired breaker so the next request gets a trial.
    pub fn state(&self) -> CircuitState {
        let raw = self.state.load(Ordering::Acquire);
        if raw == 2
            && (self.now)().saturating_sub(self.opened_at.load(Ordering::Acquire))
                >= self.cooldown_ms
        {
            // CAS so exactly one thread flips Open -> HalfOpen. ONLY the winner
            // resets the trial-success counter and reports HalfOpen. A losing
            // thread (another already flipped, or a trial has since closed/
            // reopened the breaker) must NOT blindly wipe successes a completed
            // trial may already have recorded, nor report a stale HalfOpen — it
            // re-reads the live state below.
            if self
                .state
                .compare_exchange(2, 1, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                self.successes.store(0, Ordering::Release);
                return CircuitState::HalfOpen;
            }
            return match self.state.load(Ordering::Acquire) {
                0 => CircuitState::Closed,
                1 => CircuitState::HalfOpen,
                _ => CircuitState::Open,
            };
        }
        match raw {
            0 => CircuitState::Closed,
            1 => CircuitState::HalfOpen,
            _ => CircuitState::Open,
        }
    }

    /// Whether a request may be attempted, IGNORING half-open concurrency
    /// (Closed or HalfOpen). The half-open probe cap lives in [`Self::admits`],
    /// which the [`HealthTracker`] applies with the live in-flight trial count;
    /// this bare check is the concurrency-agnostic view used for diagnostics.
    pub fn is_available(&self) -> bool {
        self.state() != CircuitState::Open
    }

    /// Whether a request may be ADMITTED given `in_flight_trials` currently
    /// outstanding against this provider. Closed always admits; Open never does;
    /// HalfOpen admits only while fewer than `success_threshold` trials are in
    /// flight — a lock-free concurrency cap so a still-down provider is not
    /// funnelled full traffic during its half-open window (the reopen signal only
    /// arrives when a trial fails, which for a hard-down provider takes a whole
    /// attempt-timeout to materialize).
    ///
    /// The cap is SOFT: a caller reads the trial count, then dispatches (and only
    /// then increments the gauge), so a burst can momentarily overshoot the
    /// threshold. It is nonetheless self-correcting and leak-proof — once the
    /// gauge reaches the threshold new arrivals fail over, and the first failed
    /// trial reopens the breaker ([`Self::record_failure`]) — which is what sheds
    /// the load the pre-cap breaker never did. A HARD cap needs a probe permit
    /// held across the dispatch at the call site (the proxy attempt loop).
    pub fn admits(&self, in_flight_trials: u64) -> bool {
        match self.state() {
            CircuitState::Closed => true,
            CircuitState::Open => false,
            CircuitState::HalfOpen => in_flight_trials < self.success_threshold,
        }
    }

    pub fn record_success(&self) {
        match self.state.load(Ordering::Acquire) {
            1 => {
                let s = self.successes.fetch_add(1, Ordering::AcqRel) + 1;
                if s >= self.success_threshold {
                    self.close();
                }
            }
            _ => {
                // Closed: a success resets the consecutive-failure count.
                self.failures.store(0, Ordering::Release);
            }
        }
    }

    pub fn record_failure(&self) {
        match self.state.load(Ordering::Acquire) {
            1 => self.open(), // a half-open trial failed -> reopen
            _ => {
                let f = self.failures.fetch_add(1, Ordering::AcqRel) + 1;
                if f >= self.failure_threshold {
                    self.open();
                }
            }
        }
    }

    fn open(&self) {
        self.state.store(2, Ordering::Release);
        self.opened_at.store((self.now)(), Ordering::Release);
        self.successes.store(0, Ordering::Release);
    }

    fn close(&self) {
        self.state.store(0, Ordering::Release);
        self.failures.store(0, Ordering::Release);
        self.successes.store(0, Ordering::Release);
    }
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self::new()
    }
}

/// EWMA smoothing factor for latency, in 1/1000ths. The new sample is weighted
/// `LATENCY_ALPHA_MILLI / 1000`, the prior EWMA the remainder. 0.2 reacts to
/// shifts within a handful of samples without over-weighting a single spike.
const LATENCY_ALPHA_MILLI: u64 = 200;

/// Sentinel meaning "no latency sample recorded yet" for the atomic EWMA store.
/// `u64::MAX` ms is unreachable as a real observation, so it can't collide with
/// a measured value.
const LATENCY_UNSET: u64 = u64::MAX;

/// A lock-free exponentially-weighted moving average of observed latency, in
/// whole milliseconds, stored in a single atomic — same spirit as
/// `CircuitBreaker`: no mutex on the request hot path.
///
/// `record` folds each new sample in with a CAS loop; `read` returns `None`
/// until the first sample lands, so the latency strategy can treat untried
/// providers optimistically.
struct LatencyEwma {
    ewma_ms: AtomicU64,
}

impl LatencyEwma {
    fn new() -> Self {
        Self {
            ewma_ms: AtomicU64::new(LATENCY_UNSET),
        }
    }

    /// Fold a new latency sample into the EWMA. Lock-free via a CAS retry loop;
    /// contention here is negligible (one update per completed provider call).
    fn record(&self, sample_ms: u64) {
        loop {
            let prior = self.ewma_ms.load(Ordering::Acquire);
            let next = if prior == LATENCY_UNSET {
                sample_ms // first sample seeds the average
            } else {
                // ewma = alpha*sample + (1-alpha)*prior, in integer milli-units.
                (LATENCY_ALPHA_MILLI * sample_ms + (1000 - LATENCY_ALPHA_MILLI) * prior) / 1000
            };
            if self
                .ewma_ms
                .compare_exchange_weak(prior, next, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return;
            }
        }
    }

    /// Current EWMA in ms, or `None` if no sample has been recorded yet.
    fn read(&self) -> Option<u64> {
        match self.ewma_ms.load(Ordering::Acquire) {
            LATENCY_UNSET => None,
            v => Some(v),
        }
    }
}

/// A lock-free in-flight (outstanding-request) gauge: one `AtomicU64` per
/// provider, incremented when an attempt is DISPATCHED and decremented when it
/// COMPLETES. Held behind an `Arc` so an [`InFlightGuard`] can own a cheap clone
/// and decrement on `Drop` no matter how the attempt exits (success, error, `?`,
/// cancellation, or panic-unwind) — the gauge can never leak stuck-high.
///
/// Read by [`RoutingStrategy::LeastBusy`] to order candidates by fewest
/// outstanding requests. Maintained ALWAYS (one relaxed add/sub per attempt),
/// but it only AFFECTS ordering under `LeastBusy`; every other strategy ignores
/// it, so keeping it current never perturbs their ordering.
#[derive(Default)]
struct InFlightGauge {
    count: AtomicU64,
}

/// RAII guard returned by [`HealthTracker::enter_in_flight`]. Holds an `Arc` to
/// the target provider's gauge; the increment happens at construction, the
/// decrement at [`Drop`]. Because `Drop` runs on EVERY exit path of the scope it
/// lives in, the in-flight count is balanced even when the provider call returns
/// an error, hits `?`, is cancelled (its future dropped), or panics.
///
/// `#[must_use]`: a guard that is created and immediately dropped would inc then
/// dec with no work in between, which is almost always a bug at the call site.
#[must_use = "hold the guard for the duration of the provider call; dropping it early decrements the in-flight gauge"]
pub struct InFlightGuard {
    gauge: Arc<InFlightGauge>,
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        // Saturating-style: fetch_sub wraps on underflow, but inc/dec are always
        // balanced (one guard per attempt), so the count cannot go negative.
        self.gauge.count.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Outcome of a HARD-cap probe admission ([`HealthTracker::try_enter_probe`]).
pub enum ProbeAdmission {
    /// Admitted — hold the [`InFlightGuard`] across the provider call. In HalfOpen
    /// the guard IS the reserved probe permit (the gauge slot was taken atomically
    /// with the cap check); in Closed it is the ordinary in-flight meter.
    Admitted(InFlightGuard),
    /// Refused: the breaker is Open, OR HalfOpen with its probe cap already
    /// saturated by outstanding trials. The caller must fail over WITHOUT
    /// dispatching — this is the load the hard cap sheds.
    Rejected,
    /// The provider has no registered breaker/gauge (unknown to health). Proceed
    /// UNTRACKED, exactly as before the gauge existed — fail-open, byte-identical
    /// to the pre-cap `enter_in_flight` returning `None`.
    Untracked,
}

/// The three per-provider health cells, bundled so the whole set for one
/// provider lives behind a single `Arc` in the registry map. The atomics inside
/// each cell do all the mutation, so a live `Arc<ProviderHealth>` is safely
/// shared and mutated without a lock — a registry swap that clones the map keeps
/// these same `Arc`s, so a provider's accumulated state survives it (ADR-113).
struct ProviderHealth {
    breaker: CircuitBreaker,
    latency: LatencyEwma,
    in_flight: Arc<InFlightGauge>,
}

impl ProviderHealth {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            breaker: CircuitBreaker::new(),
            latency: LatencyEwma::new(),
            in_flight: Arc::new(InFlightGauge::default()),
        })
    }
}

/// The owner key for a provider health cell registered without a tenant: the
/// built-ins registered at construction, and operator-global custom providers
/// (the self-host/CE boot file, `tenant_id: None`). The single shared
/// definition — the binary's `CustomProviderStore` imports THIS constant, so
/// the health registry and the adapter registry can never disagree on the
/// sentinel.
pub const GLOBAL_OWNER: &str = "";

/// Per-provider health, scoped by OWNER: one circuit breaker plus one latency
/// EWMA plus one in-flight gauge per `(owner, provider)` cell. Built-ins are
/// registered at startup under [`GLOBAL_OWNER`]; custom providers (ADR-099)
/// are registered at runtime via [`register`](Self::register) under their
/// owning tenant. The provider map is held behind an [`ArcSwap`] so
/// registration is an RCU swap — every read on the request path is a single
/// lock-free atomic load, never a mutex (ADR-113). The atomics inside each
/// `ProviderHealth` do all per-provider mutation, so an in-flight record is
/// never lost to a concurrent registration.
///
/// # Scope invariant: the breaker follows the adapter
///
/// Every accessor resolves `(owner, provider)` **tenant-first, then
/// [`GLOBAL_OWNER`]** — the exact resolution order of the adapter registry
/// (`CustomProviderStore::entry_for` / `AppState::resolve_provider`). If
/// provider resolution yields a distinct adapter for a tenant, the health
/// lookup for that tenant yields a distinct breaker/EWMA/gauge; if it falls
/// through to a shared (built-in or operator-global) adapter, the health state
/// is the shared cell. Callers therefore pass the AUTHENTICATED tenant
/// unconditionally and never branch on "is this a built-in" — the fallback IS
/// the built-in path. A provider name is request-influenced free text, so a
/// process-wide registry keyed by the bare name let one tenant's dead `myvllm`
/// open the breaker for every other tenant's unrelated `myvllm` (the
/// cross-tenant DoS this keying closes); the sibling `key_cooldowns` cells
/// below were always tenant-keyed for the same reason.
pub struct HealthTracker {
    /// `owner → provider name → health cells`. Nested (not a `(String, String)`
    /// tuple key) so the hot path probes with two borrowed `&str`s — a tuple key
    /// cannot be probed by reference and would force two `String` allocations
    /// per read, inside `sort_by_key` comparator closures on the ordering path.
    providers: ArcSwap<HashMap<String, HashMap<String, Arc<ProviderHealth>>>>,
    /// ADR-087 multi-account: per-key rate-limit **cooldown** cells (`cooled_until`
    /// epoch-millis; `0` = not cooled). A fixed-size array indexed by
    /// `hash(tenant, provider, key_index)`, allocated once — lock-free reads/writes
    /// over a single atomic, no dynamic map, no key-registry coupling, nothing to
    /// rebuild on a key reload. Cross-tenant correct (tenant is in the hash); a rare
    /// hash collision shares a cooldown benignly (a healthy key is briefly skipped,
    /// self-heals at expiry — never a wrong-key-used). Distinct from the per-provider
    /// `CircuitBreaker` (fault-detection); a cooldown cell cools a key on the FIRST
    /// 429 and honors Retry-After.
    key_cooldowns: Box<[AtomicU64]>,
}

/// Number of per-key cooldown cells (power of two so the hash maps with a mask).
/// 4096 × 8 B = 32 KiB, allocated once; ample for the handful of `(tenant, provider,
/// key)` tuples a deployment holds, so collisions are negligible.
const KEY_COOLDOWN_CELLS: usize = 4096;

impl HealthTracker {
    /// Construct with the built-in provider set, registered under
    /// [`GLOBAL_OWNER`] — built-ins are process-shared adapters, so their health
    /// is deliberately shared across tenants (real fleet signal about a real
    /// upstream). The signature deliberately takes bare names: a built-in has no
    /// owner by construction.
    pub fn new<I, S>(providers: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut by_name: HashMap<String, Arc<ProviderHealth>> = HashMap::new();
        for p in providers {
            by_name.insert(p.into(), ProviderHealth::new());
        }
        let mut map: HashMap<String, HashMap<String, Arc<ProviderHealth>>> = HashMap::new();
        map.insert(GLOBAL_OWNER.to_string(), by_name);
        let key_cooldowns = (0..KEY_COOLDOWN_CELLS)
            .map(|_| AtomicU64::new(0))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            providers: ArcSwap::from_pointee(map),
            key_cooldowns,
        }
    }

    /// Resolve `(owner, provider)` to its health cells and run `f` on them —
    /// the ONE resolution point every accessor routes through, so the scope rule
    /// cannot drift per-method. Tenant cell first, then the [`GLOBAL_OWNER`]
    /// fallback — mirroring `CustomProviderStore::entry_for` /
    /// `AppState::resolve_provider` exactly (the breaker follows the adapter).
    /// `None` ⇒ unknown to health under BOTH owners (the callers' fail-open
    /// contract). Holds the `ArcSwap` load guard only for the closure's duration:
    /// two borrowed-`&str` probes, no allocation, no `Arc` clone on the read path
    /// (a closure that must outlive the guard clones only what it keeps — see
    /// [`enter_in_flight`](Self::enter_in_flight)).
    fn with_health<R>(
        &self,
        owner: &str,
        provider: &str,
        f: impl FnOnce(&ProviderHealth) -> R,
    ) -> Option<R> {
        let snapshot = self.providers.load();
        let health = snapshot
            .get(owner)
            .and_then(|by_name| by_name.get(provider))
            .or_else(|| {
                snapshot
                    .get(GLOBAL_OWNER)
                    .and_then(|by_name| by_name.get(provider))
            })?;
        Some(f(health))
    }

    /// Register `(owner, provider)` health cells if absent (idempotent). Built-ins
    /// are registered at construction under [`GLOBAL_OWNER`]; this is how a runtime
    /// custom provider (ADR-099) gets a circuit breaker + latency EWMA + in-flight
    /// gauge — under its OWNING tenant, so two tenants' same-named providers (a
    /// supported state — per-owner uniqueness) never share a breaker. Pass
    /// [`GLOBAL_OWNER`] only for an ownerless (operator-global boot-file)
    /// provider.
    ///
    /// Additive-only: an existing cell is left **untouched**, so a breaker an
    /// operator just watched open is never reset by re-registering it (or by
    /// registering a different provider). The RCU closure clones the outer map and
    /// the touched owner's inner map — which clones the `Arc`s, preserving every
    /// provider's accumulated state — and swaps in the extended copy. Off the hot
    /// path (called on registry upsert / boot replay); registration is the ONLY
    /// way a cell is created — no read accessor ever grows the map, so the
    /// request path stays incapable of minting registry entries.
    pub fn register(&self, owner: &str, provider: impl Into<String>) {
        let name = provider.into();
        // Fast path: already present ⇒ no allocation, no swap (and never a reset).
        // Deliberately owner-exact (NOT the global fallback): a tenant registering
        // `myvllm` must get its own cell even when a global `myvllm` exists.
        if self
            .providers
            .load()
            .get(owner)
            .is_some_and(|by_name| by_name.contains_key(&name))
        {
            return;
        }
        self.providers.rcu(|cur| {
            let mut next = HashMap::clone(cur);
            next.entry(owner.to_string())
                .or_default()
                .entry(name.clone())
                .or_insert_with(ProviderHealth::new);
            next
        });
    }

    /// Test-only: replace `(owner, provider)`'s breaker with a clock-injectable
    /// one (fresh latency/gauge), so cooldown→half-open transitions are
    /// deterministic. Uses a single `store` (not `rcu`): `CircuitBreaker` holds a
    /// `Box<dyn Fn>` clock and is not `Clone`, so it cannot be moved into an
    /// `FnMut` retry closure — and tests are single-threaded here, so there is no
    /// swap to contend with.
    #[cfg(test)]
    fn set_breaker_for_test(&self, owner: &str, provider: &str, breaker: CircuitBreaker) {
        let mut next = (**self.providers.load()).clone();
        next.entry(owner.to_string()).or_default().insert(
            provider.to_string(),
            Arc::new(ProviderHealth {
                breaker,
                latency: LatencyEwma::new(),
                in_flight: Arc::new(InFlightGauge::default()),
            }),
        );
        self.providers.store(Arc::new(next));
    }

    /// The per-key cooldown cell for `(tenant, provider, key_index)` (ADR-087).
    fn key_cell(&self, tenant: &str, provider: &str, key_index: usize) -> &AtomicU64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        tenant.hash(&mut h);
        provider.hash(&mut h);
        key_index.hash(&mut h);
        // KEY_COOLDOWN_CELLS is a power of two, so `& (N-1)` == `% N`.
        &self.key_cooldowns[(h.finish() as usize) & (KEY_COOLDOWN_CELLS - 1)]
    }

    /// Is pool key `key_index` of `(tenant, provider)` available (not cooled down)
    /// at `now_ms`? `now_ms` is passed in so the check is pure/testable (no stored
    /// clock). A never-cooled key (cell `0`) is always available.
    pub fn key_available(
        &self,
        tenant: &str,
        provider: &str,
        key_index: usize,
        now_ms: u64,
    ) -> bool {
        self.key_cell(tenant, provider, key_index)
            .load(Ordering::Acquire)
            <= now_ms
    }

    /// The `cooled_until` timestamp (epoch ms; `0` = never cooled) for a pool key.
    pub fn key_cooled_until(&self, tenant: &str, provider: &str, key_index: usize) -> u64 {
        self.key_cell(tenant, provider, key_index)
            .load(Ordering::Acquire)
    }

    /// Cool a pool key until `until_ms` (epoch ms). **Extend-only** (never shortens
    /// an existing cooldown), so a transient `5xx` cannot cut short a `401` dead-key
    /// window; an expired/absent cooldown (`cell <= until_ms`) is set fresh.
    pub fn cool_key(&self, tenant: &str, provider: &str, key_index: usize, until_ms: u64) {
        let cell = self.key_cell(tenant, provider, key_index);
        let mut cur = cell.load(Ordering::Acquire);
        while until_ms > cur {
            match cell.compare_exchange_weak(cur, until_ms, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => break,
                Err(observed) => cur = observed,
            }
        }
    }

    /// Clear a pool key's cooldown (a success — the key demonstrably works).
    pub fn clear_key(&self, tenant: &str, provider: &str, key_index: usize) {
        self.key_cell(tenant, provider, key_index)
            .store(0, Ordering::Release);
    }

    /// Record an observed latency sample (milliseconds) for `(owner, provider)`,
    /// folding it into that cell's EWMA (tenant cell first, then the global
    /// fallback — the adapter's resolution order). Unknown providers are ignored.
    pub fn record_latency(&self, owner: &str, provider: &str, ms: u64) {
        self.with_health(owner, provider, |h| h.latency.record(ms));
    }

    /// Current latency EWMA (ms) for `(owner, provider)`, or `None` if no sample
    /// has been recorded yet (or the provider is unknown). Used by the `Latency`
    /// strategy to order providers and to treat untried ones optimistically.
    pub fn latency_ms(&self, owner: &str, provider: &str) -> Option<u64> {
        self.with_health(owner, provider, |h| h.latency.read())
            .flatten()
    }

    /// Current number of outstanding (in-flight) requests for `(owner, provider)`.
    /// Read by the [`RoutingStrategy::LeastBusy`] ordering. An unknown provider
    /// (no gauge registered) reports the maximum so it sorts LAST — least-busy
    /// never prefers a provider it cannot meter.
    pub fn in_flight(&self, owner: &str, provider: &str) -> u64 {
        self.with_health(owner, provider, |h| {
            h.in_flight.count.load(Ordering::Relaxed)
        })
        .unwrap_or(u64::MAX)
    }

    /// Mark an attempt as DISPATCHED to `(owner, provider)`: increments its
    /// in-flight gauge and returns an [`InFlightGuard`] that decrements it on
    /// `Drop`. Hold the guard across the provider call so the count is balanced
    /// on every exit path (success, error, `?`, cancellation, panic). Returns
    /// `None` for an unknown provider (no gauge to track) — the caller proceeds
    /// untracked, exactly as before the gauge existed.
    pub fn enter_in_flight(&self, owner: &str, provider: &str) -> Option<InFlightGuard> {
        // Clone only the gauge Arc INSIDE the closure (it must outlive the load
        // guard, inside the returned InFlightGuard) — not the whole ProviderHealth.
        let gauge = self.with_health(owner, provider, |h| h.in_flight.clone())?;
        gauge.count.fetch_add(1, Ordering::Relaxed);
        Some(InFlightGuard { gauge })
    }

    /// Available unless the circuit is Open, OR the breaker is HalfOpen and its
    /// probe cap is already saturated by outstanding trials — the half-open
    /// concurrency cap ([`CircuitBreaker::admits`]) reuses the live in-flight
    /// gauge as the trial counter (in HalfOpen every in-flight request IS a
    /// trial, since nothing else dispatches while Open), so it needs no separate
    /// permit bookkeeping and can never leak. Unknown providers (no breaker
    /// registered under either owner) are treated as available.
    pub fn is_available(&self, owner: &str, provider: &str) -> bool {
        self.with_health(owner, provider, |h| {
            h.breaker.admits(h.in_flight.count.load(Ordering::Relaxed))
        })
        .unwrap_or(true)
    }

    /// Atomically ADMIT a probe against `provider` **and** reserve its in-flight
    /// slot in one operation — the HARD half-open concurrency cap. The
    /// [`is_available`](Self::is_available) + [`enter_in_flight`](Self::enter_in_flight)
    /// pair a caller would otherwise use leaves a check→dispatch gap: two tasks can
    /// each read `in_flight < success_threshold` before either increments the gauge,
    /// so a concurrent burst overshoots the cap ([`CircuitBreaker::admits`] documents
    /// this softness). Here the threshold check and the gauge increment are one CAS,
    /// so at most `success_threshold` trials are ever in flight against a HalfOpen
    /// provider. Lock-free (a single atomic gauge, RAII-released on `Drop`); the
    /// breaker-state read is a separate load, but a transition racing it is benign
    /// and self-correcting (a HalfOpen→Closed race admits one extra; a Closed→HalfOpen
    /// race is the same one-instant softness the gauge always had).
    pub fn try_enter_probe(&self, owner: &str, provider: &str) -> ProbeAdmission {
        // The whole admit runs inside the `with_health` closure: `breaker`
        // borrows through the load guard; only the gauge Arc is cloned (it
        // outlives the guard in a returned InFlightGuard).
        self.with_health(owner, provider, |health| {
            let breaker = &health.breaker;
            let gauge = health.in_flight.clone();
            match breaker.state() {
                CircuitState::Open => ProbeAdmission::Rejected,
                // Closed: admit unconditionally — the probe cap is half-open-only.
                // One relaxed add, identical to `enter_in_flight`.
                CircuitState::Closed => {
                    gauge.count.fetch_add(1, Ordering::Relaxed);
                    ProbeAdmission::Admitted(InFlightGuard { gauge })
                }
                // HalfOpen: conditional increment. Admit only while fewer than
                // `success_threshold` trials are outstanding; the CAS folds the cap
                // check and the reservation into one atomic step (no overshoot).
                CircuitState::HalfOpen => {
                    let cap = breaker.success_threshold;
                    loop {
                        let cur = gauge.count.load(Ordering::Acquire);
                        if cur >= cap {
                            return ProbeAdmission::Rejected;
                        }
                        if gauge
                            .count
                            .compare_exchange_weak(
                                cur,
                                cur + 1,
                                Ordering::AcqRel,
                                Ordering::Acquire,
                            )
                            .is_ok()
                        {
                            return ProbeAdmission::Admitted(InFlightGuard { gauge });
                        }
                    }
                }
            }
        })
        // Unknown to health (no cell under either owner): proceed untracked —
        // fail-open, byte-identical to the pre-gauge behaviour.
        .unwrap_or(ProbeAdmission::Untracked)
    }

    pub fn record_success(&self, owner: &str, provider: &str) {
        self.with_health(owner, provider, |h| h.breaker.record_success());
    }

    pub fn record_failure(&self, owner: &str, provider: &str) {
        self.with_health(owner, provider, |h| h.breaker.record_failure());
    }

    pub fn state(&self, owner: &str, provider: &str) -> CircuitState {
        self.with_health(owner, provider, |h| h.breaker.state())
            .unwrap_or(CircuitState::Closed)
    }

    /// The provider names registered under [`GLOBAL_OWNER`] ONLY: the built-ins
    /// plus any operator-global (boot-file) custom providers. This feeds the
    /// UNAUTHENTICATED `/status` surface, so it must never touch the outer
    /// (tenant) key set — a tenant-registered provider name is customer-chosen
    /// free text and nothing tenant-shaped may reach that surface. A lock-free
    /// load of the current registry snapshot, safe off the hot path. Returns
    /// owned `String`s (the snapshot guard does not outlive the call); order is
    /// unspecified, so the caller sorts for a stable view.
    pub fn global_provider_names(&self) -> Vec<String> {
        self.providers
            .load()
            .get(GLOBAL_OWNER)
            .map(|by_name| by_name.keys().cloned().collect())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_closed_and_available() {
        let cb = CircuitBreaker::new();
        assert_eq!(cb.state(), CircuitState::Closed);
        assert!(cb.is_available());
    }

    #[test]
    fn global_provider_names_lists_registered_builtins() {
        let h = HealthTracker::new(["openai", "anthropic", "gemini"]);
        let mut names = h.global_provider_names();
        names.sort_unstable();
        assert_eq!(names, vec!["anthropic", "gemini", "openai"]);
    }

    // --- ADR-113 dynamic registration (custom providers) ----------------------
    // Callers pass the authenticated tenant unconditionally; built-ins resolve
    // through the GLOBAL_OWNER fallback, so these register/lookup tests drive
    // the tenant-scoped path exactly as the proxy does.

    #[test]
    fn unregistered_custom_provider_is_available_but_untracked() {
        let h = HealthTracker::new(["openai"]);
        // Fail-open: an unknown provider is available and its records no-op.
        assert!(h.is_available("t_a", "vllm_local"));
        assert_eq!(h.state("t_a", "vllm_local"), CircuitState::Closed);
        for _ in 0..DEFAULT_FAILURE_THRESHOLD {
            h.record_failure("t_a", "vllm_local"); // no breaker ⇒ silently ignored
        }
        assert!(
            h.is_available("t_a", "vllm_local"),
            "still untracked, never trips"
        );
        assert!(matches!(
            h.try_enter_probe("t_a", "vllm_local"),
            ProbeAdmission::Untracked
        ));
    }

    #[test]
    fn register_gives_a_custom_provider_a_breaker_that_trips() {
        let h = HealthTracker::new(["openai"]);
        h.register("t_a", "vllm_local");
        assert!(h.is_available("t_a", "vllm_local"));
        // Now it is tracked: DEFAULT_FAILURE_THRESHOLD failures open the circuit,
        // and order-time `is_available` will drop it (closing ledger #11).
        for _ in 0..DEFAULT_FAILURE_THRESHOLD {
            h.record_failure("t_a", "vllm_local");
        }
        assert_eq!(h.state("t_a", "vllm_local"), CircuitState::Open);
        assert!(!h.is_available("t_a", "vllm_local"));
    }

    #[test]
    fn register_is_idempotent_and_never_resets_an_open_breaker() {
        let h = HealthTracker::new(["openai"]);
        h.register("t_a", "vllm_local");
        for _ in 0..DEFAULT_FAILURE_THRESHOLD {
            h.record_failure("t_a", "vllm_local");
        }
        assert_eq!(h.state("t_a", "vllm_local"), CircuitState::Open);
        // Re-registering (an update upsert) must NOT reset the breaker an operator
        // just watched open.
        h.register("t_a", "vllm_local");
        assert_eq!(h.state("t_a", "vllm_local"), CircuitState::Open);
        assert!(!h.is_available("t_a", "vllm_local"));
    }

    #[test]
    fn registering_one_provider_preserves_anothers_state() {
        let h = HealthTracker::new(["openai"]);
        h.register("t_a", "vllm_a");
        for _ in 0..DEFAULT_FAILURE_THRESHOLD {
            h.record_failure("t_a", "vllm_a");
        }
        assert_eq!(h.state("t_a", "vllm_a"), CircuitState::Open);
        // Registering a DIFFERENT provider (even under a different owner) swaps
        // the map; the clone preserves vllm_a's live breaker Arc, so its Open
        // state survives.
        h.register("t_b", "vllm_b");
        assert_eq!(h.state("t_a", "vllm_a"), CircuitState::Open);
        assert!(h.is_available("t_b", "vllm_b"));
    }

    #[test]
    fn register_is_a_noop_for_a_builtin() {
        let h = HealthTracker::new(["openai"]);
        for _ in 0..DEFAULT_FAILURE_THRESHOLD {
            h.record_failure("t_a", "openai");
        }
        assert_eq!(h.state("t_a", "openai"), CircuitState::Open);
        // Re-registering a built-in under GLOBAL_OWNER must not reset its breaker.
        h.register(GLOBAL_OWNER, "openai");
        assert_eq!(h.state("t_a", "openai"), CircuitState::Open);
    }

    // --- tenant-scoped health cells --------------------------------------------

    /// The cross-tenant DoS this keying closes.
    ///
    /// Cross-tenant name collisions are a SUPPORTED state (`CustomProviderStore`
    /// is keyed per-owner), but the health registry was keyed by the BARE
    /// provider name, so two tenants' distinct `myvllm` providers — different
    /// base_url, different upstream, different credentials — collapsed onto ONE
    /// breaker/EWMA/gauge. Tenant A pointing its `myvllm` at a dead host and
    /// sending a handful of requests opened the breaker for tenant B's healthy,
    /// unrelated `myvllm`. FAILS ON THE OLD CODE (single flat map): A's
    /// failures tripped B's availability.
    #[test]
    fn same_provider_name_across_tenants_does_not_share_a_breaker() {
        let h = HealthTracker::new(["openai"]);
        h.register("t_alpha", "myvllm");
        h.register("t_beta", "myvllm");

        // Drive t_alpha's myvllm to Open.
        for _ in 0..DEFAULT_FAILURE_THRESHOLD {
            h.record_failure("t_alpha", "myvllm");
        }
        assert_eq!(h.state("t_alpha", "myvllm"), CircuitState::Open);
        assert!(!h.is_available("t_alpha", "myvllm"));

        // The victim tenant's same-named provider is untouched.
        assert!(
            h.is_available("t_beta", "myvllm"),
            "cross-tenant DoS: tenant A driving its own `myvllm` breaker Open \
             must not fast-fail tenant B's distinct provider of the same name"
        );
        assert_eq!(
            h.state("t_beta", "myvllm"),
            CircuitState::Closed,
            "t_beta's breaker must still be Closed after t_alpha's failures"
        );

        // And t_beta owns its own independent lifecycle: it can trip on its own.
        for _ in 0..DEFAULT_FAILURE_THRESHOLD {
            h.record_failure("t_beta", "myvllm");
        }
        assert_eq!(h.state("t_beta", "myvllm"), CircuitState::Open);
        assert!(!h.is_available("t_beta", "myvllm"));
    }

    /// The secondary cross-tenant channels: the latency EWMA (routing bias) and
    /// the in-flight gauge (LeastBusy ordering) must be per-cell too. FAILS ON
    /// THE OLD CODE: A's 5000ms sample skewed B's `latency` ordering, and A's
    /// outstanding requests inflated B's least-busy count.
    #[test]
    fn latency_and_in_flight_do_not_bleed_across_tenants() {
        let h = HealthTracker::new(["openai"]);
        h.register("t_alpha", "myvllm");
        h.register("t_beta", "myvllm");

        h.record_latency("t_alpha", "myvllm", 5_000);
        assert_eq!(h.latency_ms("t_alpha", "myvllm"), Some(5_000));
        assert_eq!(
            h.latency_ms("t_beta", "myvllm"),
            None,
            "tenant A's latency sample must not seed tenant B's EWMA \
             (cross-tenant routing-bias channel)"
        );

        let guard = h.enter_in_flight("t_alpha", "myvllm").unwrap();
        assert_eq!(h.in_flight("t_alpha", "myvllm"), 1);
        assert_eq!(
            h.in_flight("t_beta", "myvllm"),
            0,
            "tenant A's outstanding request must not inflate tenant B's \
             least-busy gauge"
        );
        drop(guard);
        assert_eq!(h.in_flight("t_alpha", "myvllm"), 0);
    }

    /// Pins the INTENDED sharing: a built-in provider is one process-wide
    /// adapter talking to one real upstream, so its breaker is a fleet signal
    /// and IS shared across tenants. If a later "fix" per-tenants the
    /// built-ins, this fails and forces that to be a deliberate decision.
    #[test]
    fn builtin_breaker_is_shared_across_tenants() {
        let h = HealthTracker::new(["openai"]);
        for _ in 0..DEFAULT_FAILURE_THRESHOLD {
            h.record_failure("t_alpha", "openai");
        }
        assert_eq!(h.state("t_alpha", "openai"), CircuitState::Open);
        assert!(
            !h.is_available("t_beta", "openai"),
            "built-in health is deliberately SHARED: openai being down is true \
             for every tenant, so t_alpha's observed failures must fast-fail \
             t_beta too (one global cell, by construction)"
        );
        assert_eq!(h.state(GLOBAL_OWNER, "openai"), CircuitState::Open);
    }

    /// Resolution order mirrors the adapter registry: a tenant's OWN cell
    /// shadows a global one of the same name (a tenant's own provider wins in
    /// `CustomProviderStore::entry_for`, so its breaker must win here too).
    #[test]
    fn tenant_entry_shadows_the_global_one() {
        let h = HealthTracker::new(["openai"]);
        h.register(GLOBAL_OWNER, "shared");
        h.register("t_a", "shared");

        // Trip the GLOBAL cell; the tenant with its OWN cell is unaffected…
        for _ in 0..DEFAULT_FAILURE_THRESHOLD {
            h.record_failure(GLOBAL_OWNER, "shared");
        }
        assert!(
            h.is_available("t_a", "shared"),
            "a tenant with its own provider must read its OWN cell, not the \
             global one (shadowing mirrors adapter resolution)"
        );
        // …while a tenant WITHOUT its own cell falls through to the global one.
        assert!(!h.is_available("t_b", "shared"));

        // And records from the shadowed tenant land on ITS cell, not the global.
        h.record_latency("t_a", "shared", 100);
        assert_eq!(h.latency_ms("t_a", "shared"), Some(100));
        assert_eq!(h.latency_ms("t_b", "shared"), None); // global cell: no sample
    }

    /// The `/status` contract asserted at the source: the global name fold must
    /// never see a tenant-owned registration (a provider name is
    /// customer-chosen free text; `/status` is unauthenticated).
    #[test]
    fn global_provider_names_excludes_tenant_owned() {
        let h = HealthTracker::new(["openai"]);
        h.register(GLOBAL_OWNER, "ollama_box");
        h.register("t_acme", "acme-bank-internal");
        let mut names = h.global_provider_names();
        names.sort_unstable();
        assert_eq!(
            names,
            vec!["ollama_box", "openai"],
            "a tenant-registered provider name must never surface through the \
             global fold that feeds the unauthenticated /status page"
        );
    }

    // --- ADR-087 per-key cooldown cells ---------------------------------------

    #[test]
    fn key_cooldown_default_available_then_cools_then_recovers() {
        let h = HealthTracker::new(["openai"]);
        // Never cooled ⇒ available at any time.
        assert!(h.key_available("t_a", "openai", 0, 1_000));
        assert_eq!(h.key_cooled_until("t_a", "openai", 0), 0);
        // Cool until t=5000 ⇒ unavailable before, available at/after expiry.
        h.cool_key("t_a", "openai", 0, 5_000);
        assert!(!h.key_available("t_a", "openai", 0, 4_999));
        assert!(h.key_available("t_a", "openai", 0, 5_000));
        // Success clears it.
        h.cool_key("t_a", "openai", 0, 9_000);
        h.clear_key("t_a", "openai", 0);
        assert!(h.key_available("t_a", "openai", 0, 1));
    }

    #[test]
    fn cool_key_is_extend_only() {
        let h = HealthTracker::new(["openai"]);
        h.cool_key("t_a", "openai", 0, 600_000); // 401 dead-key window
        h.cool_key("t_a", "openai", 0, 2_000); // a later transient 5xx must NOT shorten it
        assert_eq!(h.key_cooled_until("t_a", "openai", 0), 600_000);
    }

    #[test]
    fn key_cooldown_is_cross_tenant_and_per_index() {
        let h = HealthTracker::new(["openai"]);
        h.cool_key("t_a", "openai", 0, 10_000);
        // Different tenant, same provider+index ⇒ NOT cooled (no cross-tenant bleed).
        assert!(h.key_available("t_b", "openai", 0, 1));
        // Same tenant+provider, different index ⇒ independent.
        assert!(h.key_available("t_a", "openai", 1, 1));
    }

    #[test]
    fn opens_after_threshold_failures() {
        let cb = CircuitBreaker::with_config(3, 2, 30_000);
        for _ in 0..3 {
            cb.record_failure();
        }
        assert_eq!(cb.state(), CircuitState::Open);
        assert!(!cb.is_available());
    }

    #[test]
    fn half_opens_after_cooldown_then_closes_on_success() {
        let cb = CircuitBreaker::with_config(2, 2, 0); // 0ms cooldown
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::HalfOpen); // cooldown elapsed
        assert!(cb.is_available());
        cb.record_success();
        cb.record_success();
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn half_open_failure_reopens() {
        use std::sync::Arc;
        // Deterministic clock the test advances by hand.
        let clock = Arc::new(AtomicU64::new(0));
        let c = clock.clone();
        let cb = CircuitBreaker::with_clock(2, 2, 100, Box::new(move || c.load(Ordering::Relaxed)));

        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open); // t=0, cooldown not elapsed

        clock.store(200, Ordering::Relaxed); // advance past the 100ms cooldown
        assert_eq!(cb.state(), CircuitState::HalfOpen); // self-heals to a trial

        cb.record_failure(); // trial fails -> reopen (opened_at = 200)
        assert_eq!(cb.state(), CircuitState::Open); // t=200, cooldown not elapsed again
    }

    #[test]
    fn half_open_admits_are_capped_at_success_threshold() {
        use std::sync::Arc;
        // Deterministic clock the test advances by hand.
        let clock = Arc::new(AtomicU64::new(0));
        let c = clock.clone();
        // failure_threshold=2, success_threshold=3, cooldown=100ms.
        let cb = CircuitBreaker::with_clock(2, 3, 100, Box::new(move || c.load(Ordering::Relaxed)));

        // Closed: admits unconditionally — the probe cap is half-open-only.
        assert!(cb.admits(0));
        assert!(cb.admits(1_000_000));

        // Trip Open: admits nothing, whatever the in-flight load.
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
        assert!(!cb.admits(0));

        // Cooldown elapses → the breaker self-heals to HalfOpen on read, and then
        // admits AT MOST `success_threshold` (3) concurrent trials; excess arrivals
        // are refused so a still-down provider is not funnelled full traffic during
        // its half-open window.
        clock.store(200, Ordering::Relaxed);
        assert!(cb.admits(0)); // 0 < 3
        assert!(cb.admits(2)); // 2 < 3 → the 3rd concurrent trial is still admitted
        assert!(!cb.admits(3)); // 3 !< 3 → capped
        assert!(!cb.admits(100));
        // Reading the cap must not itself change state — still HalfOpen.
        assert_eq!(cb.state(), CircuitState::HalfOpen);
    }

    #[test]
    fn success_resets_failures_when_closed() {
        let cb = CircuitBreaker::with_config(3, 2, 30_000);
        cb.record_failure();
        cb.record_failure();
        cb.record_success(); // reset consecutive failures
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed); // only 2 since reset, threshold 3
    }

    #[test]
    fn health_tracker_gates_known_and_passes_unknown() {
        let h = HealthTracker::new(["openai"]);
        assert!(h.is_available(GLOBAL_OWNER, "openai"));
        assert!(h.is_available(GLOBAL_OWNER, "unknown")); // no breaker -> available
        for _ in 0..5 {
            h.record_failure(GLOBAL_OWNER, "openai");
        }
        assert!(!h.is_available(GLOBAL_OWNER, "openai"));
    }

    #[test]
    fn is_available_caps_half_open_probes_via_in_flight_gauge() {
        use std::sync::Arc;
        // Build the tracker as usual, then swap openai's breaker for a
        // clock-injectable one so the cooldown → half-open transition is
        // deterministic (the default breaker uses the wall clock). The latency
        // and in-flight gauges from `new` stay intact.
        let h = HealthTracker::new(["openai"]);
        let clock = Arc::new(AtomicU64::new(0));
        let c = clock.clone();
        // failure_threshold=2, success_threshold=3, cooldown=100ms.
        h.set_breaker_for_test(
            GLOBAL_OWNER,
            "openai",
            CircuitBreaker::with_clock(2, 3, 100, Box::new(move || c.load(Ordering::Relaxed))),
        );

        // Trip Open — refused regardless of load.
        h.record_failure(GLOBAL_OWNER, "openai");
        h.record_failure(GLOBAL_OWNER, "openai");
        assert!(!h.is_available(GLOBAL_OWNER, "openai"));

        // Cooldown elapses → HalfOpen. The probe cap now reuses the live in-flight
        // gauge as the trial counter: admit up to success_threshold (3) concurrent
        // trials, then fail excess arrivals over.
        clock.store(200, Ordering::Relaxed);
        let g1 = h.enter_in_flight(GLOBAL_OWNER, "openai").unwrap();
        let g2 = h.enter_in_flight(GLOBAL_OWNER, "openai").unwrap();
        assert_eq!(h.in_flight(GLOBAL_OWNER, "openai"), 2);
        // 2 trials in flight < 3 → still admits
        assert!(h.is_available(GLOBAL_OWNER, "openai"));
        let g3 = h.enter_in_flight(GLOBAL_OWNER, "openai").unwrap();
        assert_eq!(h.in_flight(GLOBAL_OWNER, "openai"), 3);
        // 3 in flight → cap saturated, fail over
        assert!(!h.is_available(GLOBAL_OWNER, "openai"));
        drop(g3); // a trial completes → a probe slot frees up
        assert!(h.is_available(GLOBAL_OWNER, "openai"));
        drop(g1);
        drop(g2);
    }

    #[test]
    fn try_enter_probe_hard_caps_half_open_concurrency() {
        use std::sync::Arc;
        // Deterministic clock for the cooldown → half-open transition.
        let h = HealthTracker::new(["openai"]);
        let clock = Arc::new(AtomicU64::new(0));
        let c = clock.clone();
        // failure_threshold=2, success_threshold=2, cooldown=100ms.
        h.set_breaker_for_test(
            GLOBAL_OWNER,
            "openai",
            CircuitBreaker::with_clock(2, 2, 100, Box::new(move || c.load(Ordering::Relaxed))),
        );

        // Closed: admits unconditionally (probe cap is half-open-only).
        match h.try_enter_probe(GLOBAL_OWNER, "openai") {
            ProbeAdmission::Admitted(g) => {
                assert_eq!(h.in_flight(GLOBAL_OWNER, "openai"), 1);
                drop(g);
                assert_eq!(h.in_flight(GLOBAL_OWNER, "openai"), 0);
            }
            _ => panic!("Closed breaker must admit"),
        }

        // Unknown provider: Untracked (fail-open), no gauge touched.
        assert!(matches!(
            h.try_enter_probe(GLOBAL_OWNER, "unknown"),
            ProbeAdmission::Untracked
        ));

        // Trip Open — every probe refused.
        h.record_failure(GLOBAL_OWNER, "openai");
        h.record_failure(GLOBAL_OWNER, "openai");
        assert!(matches!(
            h.try_enter_probe(GLOBAL_OWNER, "openai"),
            ProbeAdmission::Rejected
        ));

        // Cooldown elapses → HalfOpen. The HARD cap admits at most
        // success_threshold (2) trials — reservation is ATOMIC with the check, so
        // unlike the soft `is_available` gate there is no check→dispatch overshoot.
        clock.store(200, Ordering::Relaxed);
        let p1 = match h.try_enter_probe(GLOBAL_OWNER, "openai") {
            ProbeAdmission::Admitted(g) => g,
            _ => panic!("1st half-open probe must be admitted"),
        };
        let p2 = match h.try_enter_probe(GLOBAL_OWNER, "openai") {
            ProbeAdmission::Admitted(g) => g,
            _ => panic!("2nd half-open probe must be admitted"),
        };
        assert_eq!(h.in_flight(GLOBAL_OWNER, "openai"), 2);
        // 3rd probe: cap saturated → Rejected, and the gauge did NOT increment
        // (the CAS never fired) — proving the reservation is hard, not soft.
        assert!(matches!(
            h.try_enter_probe(GLOBAL_OWNER, "openai"),
            ProbeAdmission::Rejected
        ));
        assert_eq!(h.in_flight(GLOBAL_OWNER, "openai"), 2);
        // A trial completes → a permit frees → the next probe is admitted again.
        drop(p2);
        assert_eq!(h.in_flight(GLOBAL_OWNER, "openai"), 1);
        let p3 = match h.try_enter_probe(GLOBAL_OWNER, "openai") {
            ProbeAdmission::Admitted(g) => g,
            _ => panic!("probe must be admitted after a permit frees"),
        };
        assert_eq!(h.in_flight(GLOBAL_OWNER, "openai"), 2);
        drop(p1);
        drop(p3);
        assert_eq!(h.in_flight(GLOBAL_OWNER, "openai"), 0);
    }

    #[test]
    fn latency_unset_until_first_sample_then_seeds() {
        let h = HealthTracker::new(["openai"]);
        assert_eq!(h.latency_ms(GLOBAL_OWNER, "openai"), None);
        h.record_latency(GLOBAL_OWNER, "openai", 100);
        // first sample seeds EWMA
        assert_eq!(h.latency_ms(GLOBAL_OWNER, "openai"), Some(100));
    }

    #[test]
    fn latency_ewma_smooths_toward_new_samples() {
        let h = HealthTracker::new(["openai"]);
        h.record_latency(GLOBAL_OWNER, "openai", 100); // seed = 100
        h.record_latency(GLOBAL_OWNER, "openai", 200); // 0.2*200 + 0.8*100 = 120
        assert_eq!(h.latency_ms(GLOBAL_OWNER, "openai"), Some(120));
        h.record_latency(GLOBAL_OWNER, "openai", 200); // 0.2*200 + 0.8*120 = 136
        assert_eq!(h.latency_ms(GLOBAL_OWNER, "openai"), Some(136));
    }

    #[test]
    fn latency_unknown_provider_is_ignored_and_none() {
        let h = HealthTracker::new(["openai"]);
        h.record_latency(GLOBAL_OWNER, "unknown", 500); // no-op
        assert_eq!(h.latency_ms(GLOBAL_OWNER, "unknown"), None);
    }
}
