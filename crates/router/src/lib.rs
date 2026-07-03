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

/// Per-provider health: one circuit breaker plus one latency EWMA plus one
/// in-flight gauge per provider, pre-registered at startup so the map is
/// read-only (the atomics do all the mutation, so no lock is needed on the
/// request path).
pub struct HealthTracker {
    breakers: HashMap<String, CircuitBreaker>,
    latency: HashMap<String, LatencyEwma>,
    in_flight: HashMap<String, Arc<InFlightGauge>>,
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
    pub fn new<I, S>(providers: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut breakers = HashMap::new();
        let mut latency = HashMap::new();
        let mut in_flight = HashMap::new();
        for p in providers {
            let name: String = p.into();
            breakers.insert(name.clone(), CircuitBreaker::new());
            latency.insert(name.clone(), LatencyEwma::new());
            in_flight.insert(name, Arc::new(InFlightGauge::default()));
        }
        let key_cooldowns = (0..KEY_COOLDOWN_CELLS)
            .map(|_| AtomicU64::new(0))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            breakers,
            latency,
            in_flight,
            key_cooldowns,
        }
    }

    pub fn breaker(&self, provider: &str) -> Option<&CircuitBreaker> {
        self.breakers.get(provider)
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

    /// Record an observed latency sample (milliseconds) for `provider`, folding
    /// it into that provider's EWMA. Unknown providers are ignored.
    pub fn record_latency(&self, provider: &str, ms: u64) {
        if let Some(l) = self.latency.get(provider) {
            l.record(ms);
        }
    }

    /// Current latency EWMA (ms) for `provider`, or `None` if no sample has been
    /// recorded yet (or the provider is unknown). Used by the `Latency` strategy
    /// to order providers and to treat untried ones optimistically.
    pub fn latency_ms(&self, provider: &str) -> Option<u64> {
        self.latency.get(provider).and_then(|l| l.read())
    }

    /// Current number of outstanding (in-flight) requests for `provider`. Read by
    /// the [`RoutingStrategy::LeastBusy`] ordering. An unknown provider (no gauge
    /// registered) reports the maximum so it sorts LAST — least-busy never
    /// prefers a provider it cannot meter.
    pub fn in_flight(&self, provider: &str) -> u64 {
        self.in_flight
            .get(provider)
            .map(|g| g.count.load(Ordering::Relaxed))
            .unwrap_or(u64::MAX)
    }

    /// Mark an attempt as DISPATCHED to `provider`: increments its in-flight gauge
    /// and returns an [`InFlightGuard`] that decrements it on `Drop`. Hold the
    /// guard across the provider call so the count is balanced on every exit path
    /// (success, error, `?`, cancellation, panic). Returns `None` for an unknown
    /// provider (no gauge to track) — the caller proceeds untracked, exactly as
    /// before the gauge existed.
    pub fn enter_in_flight(&self, provider: &str) -> Option<InFlightGuard> {
        let gauge = self.in_flight.get(provider)?.clone();
        gauge.count.fetch_add(1, Ordering::Relaxed);
        Some(InFlightGuard { gauge })
    }

    /// Available unless the circuit is Open, OR the breaker is HalfOpen and its
    /// probe cap is already saturated by outstanding trials — the half-open
    /// concurrency cap ([`CircuitBreaker::admits`]) reuses the live in-flight
    /// gauge as the trial counter (in HalfOpen every in-flight request IS a
    /// trial, since nothing else dispatches while Open), so it needs no separate
    /// permit bookkeeping and can never leak. Unknown providers (no breaker
    /// registered) are treated as available.
    pub fn is_available(&self, provider: &str) -> bool {
        self.breakers
            .get(provider)
            .map(|b| b.admits(self.in_flight(provider)))
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
    pub fn try_enter_probe(&self, provider: &str) -> ProbeAdmission {
        let Some(breaker) = self.breakers.get(provider) else {
            return ProbeAdmission::Untracked;
        };
        let gauge = match self.in_flight.get(provider) {
            Some(g) => g.clone(),
            None => return ProbeAdmission::Untracked,
        };
        match breaker.state() {
            CircuitState::Open => ProbeAdmission::Rejected,
            // Closed: admit unconditionally — the probe cap is half-open-only. One
            // relaxed add, identical to `enter_in_flight`.
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
                        .compare_exchange_weak(cur, cur + 1, Ordering::AcqRel, Ordering::Acquire)
                        .is_ok()
                    {
                        return ProbeAdmission::Admitted(InFlightGuard { gauge });
                    }
                }
            }
        }
    }

    pub fn record_success(&self, provider: &str) {
        if let Some(b) = self.breakers.get(provider) {
            b.record_success();
        }
    }

    pub fn record_failure(&self, provider: &str) {
        if let Some(b) = self.breakers.get(provider) {
            b.record_failure();
        }
    }

    pub fn state(&self, provider: &str) -> CircuitState {
        self.breakers
            .get(provider)
            .map(|b| b.state())
            .unwrap_or(CircuitState::Closed)
    }

    /// The registered provider names. The breaker map is built once at startup
    /// and never mutated (the atomics do all per-provider mutation), so this is
    /// a lock-free read over an immutable structure — safe to call off the hot
    /// path (e.g. the read-only `/status` surface). Order is unspecified; the
    /// caller sorts for a stable view.
    pub fn provider_names(&self) -> Vec<&str> {
        self.breakers.keys().map(String::as_str).collect()
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
    fn provider_names_lists_registered_providers() {
        let h = HealthTracker::new(["openai", "anthropic", "gemini"]);
        let mut names = h.provider_names();
        names.sort_unstable();
        assert_eq!(names, vec!["anthropic", "gemini", "openai"]);
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
        assert!(h.is_available("openai"));
        assert!(h.is_available("unknown")); // no breaker -> available
        for _ in 0..5 {
            h.record_failure("openai");
        }
        assert!(!h.is_available("openai"));
    }

    #[test]
    fn is_available_caps_half_open_probes_via_in_flight_gauge() {
        use std::sync::Arc;
        // Build the tracker as usual, then swap openai's breaker for a
        // clock-injectable one so the cooldown → half-open transition is
        // deterministic (the default breaker uses the wall clock). The latency
        // and in-flight gauges from `new` stay intact.
        let mut h = HealthTracker::new(["openai"]);
        let clock = Arc::new(AtomicU64::new(0));
        let c = clock.clone();
        // failure_threshold=2, success_threshold=3, cooldown=100ms.
        h.breakers.insert(
            "openai".to_string(),
            CircuitBreaker::with_clock(2, 3, 100, Box::new(move || c.load(Ordering::Relaxed))),
        );

        // Trip Open — refused regardless of load.
        h.record_failure("openai");
        h.record_failure("openai");
        assert!(!h.is_available("openai"));

        // Cooldown elapses → HalfOpen. The probe cap now reuses the live in-flight
        // gauge as the trial counter: admit up to success_threshold (3) concurrent
        // trials, then fail excess arrivals over.
        clock.store(200, Ordering::Relaxed);
        let g1 = h.enter_in_flight("openai").unwrap();
        let g2 = h.enter_in_flight("openai").unwrap();
        assert_eq!(h.in_flight("openai"), 2);
        assert!(h.is_available("openai")); // 2 trials in flight < 3 → still admits
        let g3 = h.enter_in_flight("openai").unwrap();
        assert_eq!(h.in_flight("openai"), 3);
        assert!(!h.is_available("openai")); // 3 in flight → cap saturated, fail over
        drop(g3); // a trial completes → a probe slot frees up
        assert!(h.is_available("openai"));
        drop(g1);
        drop(g2);
    }

    #[test]
    fn try_enter_probe_hard_caps_half_open_concurrency() {
        use std::sync::Arc;
        // Deterministic clock for the cooldown → half-open transition.
        let mut h = HealthTracker::new(["openai"]);
        let clock = Arc::new(AtomicU64::new(0));
        let c = clock.clone();
        // failure_threshold=2, success_threshold=2, cooldown=100ms.
        h.breakers.insert(
            "openai".to_string(),
            CircuitBreaker::with_clock(2, 2, 100, Box::new(move || c.load(Ordering::Relaxed))),
        );

        // Closed: admits unconditionally (probe cap is half-open-only).
        match h.try_enter_probe("openai") {
            ProbeAdmission::Admitted(g) => {
                assert_eq!(h.in_flight("openai"), 1);
                drop(g);
                assert_eq!(h.in_flight("openai"), 0);
            }
            _ => panic!("Closed breaker must admit"),
        }

        // Unknown provider: Untracked (fail-open), no gauge touched.
        assert!(matches!(
            h.try_enter_probe("unknown"),
            ProbeAdmission::Untracked
        ));

        // Trip Open — every probe refused.
        h.record_failure("openai");
        h.record_failure("openai");
        assert!(matches!(
            h.try_enter_probe("openai"),
            ProbeAdmission::Rejected
        ));

        // Cooldown elapses → HalfOpen. The HARD cap admits at most
        // success_threshold (2) trials — reservation is ATOMIC with the check, so
        // unlike the soft `is_available` gate there is no check→dispatch overshoot.
        clock.store(200, Ordering::Relaxed);
        let p1 = match h.try_enter_probe("openai") {
            ProbeAdmission::Admitted(g) => g,
            _ => panic!("1st half-open probe must be admitted"),
        };
        let p2 = match h.try_enter_probe("openai") {
            ProbeAdmission::Admitted(g) => g,
            _ => panic!("2nd half-open probe must be admitted"),
        };
        assert_eq!(h.in_flight("openai"), 2);
        // 3rd probe: cap saturated → Rejected, and the gauge did NOT increment
        // (the CAS never fired) — proving the reservation is hard, not soft.
        assert!(matches!(
            h.try_enter_probe("openai"),
            ProbeAdmission::Rejected
        ));
        assert_eq!(h.in_flight("openai"), 2);
        // A trial completes → a permit frees → the next probe is admitted again.
        drop(p2);
        assert_eq!(h.in_flight("openai"), 1);
        let p3 = match h.try_enter_probe("openai") {
            ProbeAdmission::Admitted(g) => g,
            _ => panic!("probe must be admitted after a permit frees"),
        };
        assert_eq!(h.in_flight("openai"), 2);
        drop(p1);
        drop(p3);
        assert_eq!(h.in_flight("openai"), 0);
    }

    #[test]
    fn latency_unset_until_first_sample_then_seeds() {
        let h = HealthTracker::new(["openai"]);
        assert_eq!(h.latency_ms("openai"), None);
        h.record_latency("openai", 100);
        assert_eq!(h.latency_ms("openai"), Some(100)); // first sample seeds EWMA
    }

    #[test]
    fn latency_ewma_smooths_toward_new_samples() {
        let h = HealthTracker::new(["openai"]);
        h.record_latency("openai", 100); // seed = 100
        h.record_latency("openai", 200); // 0.2*200 + 0.8*100 = 120
        assert_eq!(h.latency_ms("openai"), Some(120));
        h.record_latency("openai", 200); // 0.2*200 + 0.8*120 = 136
        assert_eq!(h.latency_ms("openai"), Some(136));
    }

    #[test]
    fn latency_unknown_provider_is_ignored_and_none() {
        let h = HealthTracker::new(["openai"]);
        h.record_latency("unknown", 500); // no-op
        assert_eq!(h.latency_ms("unknown"), None);
    }
}
