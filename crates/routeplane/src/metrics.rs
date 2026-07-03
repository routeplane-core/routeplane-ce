//! Lock-free Prometheus `/metrics` surface for the data plane.
//!
//! Observability parity gap (LiteLLM/Portkey/Cloudflare AI Gateway all expose a
//! full Prometheus scrape surface): the legacy `/metrics` exposed only the
//! capacity-shed count. This module builds out the SRE-grade golden signals
//! (request counts, latency histogram, tokens, cost, cache, provider errors,
//! hedged wins) that an SRE scrapes.
//!
//! Design constraints (CLAUDE.md):
//! - **Lock-free increments.** Every recording path is a handful of
//!   `AtomicU64::fetch_add(Relaxed)` — wait-free, no allocation, no `Mutex`. This
//!   is the same atomics-only doctrine as the `router` `CircuitBreaker`/EWMA and
//!   the binary-level `SHED_TOTAL`. We do NOT pull in the `prometheus` crate: its
//!   default registry guards the metric map behind a `RwLock`/`Mutex` and its
//!   `HistogramVec` label lookup takes a lock on first touch per label set — that
//!   is exactly the hot-path contention we forbid. A fixed, pre-allocated atomic
//!   table is both simpler and strictly wait-free, and the Prometheus text
//!   exposition format is trivial to render by hand.
//! - **Bounded label cardinality (unauth endpoint).** `/metrics` is
//!   unauthenticated (operational, like `/healthz`), so a label MUST NOT carry
//!   tenant_id, key, user, model, or any content. The only dynamic label is
//!   `provider`, bounded to the known registry set (~10) + the cache sentinels;
//!   anything else collapses to `other`. There is NO per-model label (raw
//!   client-supplied model is unbounded + a fingerprinting risk).
//! - **Observe-only.** Recording never changes a response; the render path
//!   (`GET /metrics`) is off the hot path and may allocate the output `String`.
//! - **Frugal.** In-memory atomics, $0 standing cost, scale-to-zero resets them —
//!   the same posture as the in-memory observability ring (no DB, no ADR needed).

use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};

/// The bounded provider label space. Matches the `HealthTracker::new([...])`
/// registry in `main.rs` plus the two cache sentinels and an `other` catch-all,
/// so the `provider` label is always one of a small fixed set — cardinality can
/// never grow with traffic (defense for an unauth endpoint). A `provider` string
/// not in this list (a future sentinel, a typo) maps to `other` rather than
/// minting a new series.
const PROVIDERS: &[&str] = &[
    "openai",
    "anthropic",
    "gemini",
    "azure_openai",
    "mistral",
    "cohere",
    "bedrock",
    "groq",
    "deepseek",
    "self_hosted",
    "cache",
    "semantic_cache",
    "other",
];

/// Index of the `other` catch-all bucket (last entry).
const OTHER_IDX: usize = PROVIDERS.len() - 1;

/// Map a raw provider string to its bounded label index. Unknown ⇒ `other`.
/// Pure, allocation-free; a short linear scan over a ~13-entry static slice is
/// cheaper than hashing on the hot path.
fn provider_idx(provider: &str) -> usize {
    let mut i = 0;
    while i < OTHER_IDX {
        if PROVIDERS[i] == provider {
            return i;
        }
        i += 1;
    }
    OTHER_IDX
}

/// Request outcome classes — the bounded `outcome` label of `rp_requests_total`.
/// Mirrors the closed set of terminal request dispositions the proxy can record.
#[derive(Clone, Copy)]
pub enum Outcome {
    Success,
    Error,
    ResidencyBlocked,
    GuardrailDenied,
    RateLimited,
    BudgetExceeded,
}

impl Outcome {
    const ALL: [Outcome; 6] = [
        Outcome::Success,
        Outcome::Error,
        Outcome::ResidencyBlocked,
        Outcome::GuardrailDenied,
        Outcome::RateLimited,
        Outcome::BudgetExceeded,
    ];

    fn idx(self) -> usize {
        self as usize
    }

    fn label(self) -> &'static str {
        match self {
            Outcome::Success => "success",
            Outcome::Error => "error",
            Outcome::ResidencyBlocked => "residency_blocked",
            Outcome::GuardrailDenied => "guardrail_denied",
            Outcome::RateLimited => "rate_limited",
            Outcome::BudgetExceeded => "budget_exceeded",
        }
    }
}

const OUTCOME_COUNT: usize = 6;

/// Upper bounds (ms) for the `rp_request_duration_ms` histogram. Tuned for LLM
/// round-trip / time-to-first-chunk latencies (tens of ms to a minute). The
/// implicit `+Inf` bucket is the `_count`. Buckets are cumulative ("le") in the
/// rendered output, the Prometheus convention.
const LATENCY_BUCKETS_MS: &[u64] = &[50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000, 60000];

/// Cache event type label space (`rp_cache_events_total{type,result}`).
const CACHE_TYPES: &[&str] = &["exact", "semantic"];
/// Cache result label space.
const CACHE_RESULTS: &[&str] = &["hit", "miss"];

/// Per-provider latency histogram: one atomic per finite bucket. Buckets are
/// recorded NON-cumulatively here (each observation falls in exactly one bucket
/// or the implicit `+Inf` overflow); the render path accumulates them into the
/// cumulative `le` form Prometheus expects. `sum_ms`/`count` track the standard
/// `_sum`/`_count` series.
struct LatencyHistogram {
    /// One counter per finite bucket in `LATENCY_BUCKETS_MS`.
    buckets: [AtomicU64; 10],
    /// Observations above the largest finite bucket (the `+Inf`-only tail).
    overflow: AtomicU64,
    /// Sum of all observed latencies in ms (the `_sum` series).
    sum_ms: AtomicU64,
    /// Total observation count (the `_count` series, == `+Inf` cumulative bucket).
    count: AtomicU64,
}

impl LatencyHistogram {
    const fn new() -> Self {
        // `AtomicU64` is not `Copy`, so the array can't be `[AtomicU64::new(0); N]`;
        // spell out the ten finite buckets (kept in lockstep with LATENCY_BUCKETS_MS).
        Self {
            buckets: [
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
            ],
            overflow: AtomicU64::new(0),
            sum_ms: AtomicU64::new(0),
            count: AtomicU64::new(0),
        }
    }

    /// Wait-free record of one observation. Single linear scan of the bucket
    /// bounds + three `fetch_add`s; no allocation, no lock.
    fn observe(&self, ms: u64) {
        let mut placed = false;
        for (i, &bound) in LATENCY_BUCKETS_MS.iter().enumerate() {
            if ms <= bound {
                self.buckets[i].fetch_add(1, Ordering::Relaxed);
                placed = true;
                break;
            }
        }
        if !placed {
            self.overflow.fetch_add(1, Ordering::Relaxed);
        }
        self.sum_ms.fetch_add(ms, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
    }
}

/// The process-global metrics table. All fields are atomics or arrays of atomics
/// sized at compile time, so the whole struct is a fixed allocation and every
/// recording method is wait-free. Constructed once and read via the static
/// `metrics()` accessor.
pub struct Metrics {
    /// `rp_requests_total{provider,outcome}` — [provider][outcome].
    requests: [[AtomicU64; OUTCOME_COUNT]; PROVIDERS.len()],
    /// `rp_request_duration_ms` histogram, per provider.
    duration: [LatencyHistogram; PROVIDERS.len()],
    /// `rp_tokens_total{kind=prompt}`.
    prompt_tokens: AtomicU64,
    /// `rp_tokens_total{kind=completion}`.
    completion_tokens: AtomicU64,
    /// `rp_tokens_total{kind=cached}` — prompt-cache READ tokens (the cached
    /// SUBSET of prompt tokens) surfaced by providers (Anthropic
    /// `cache_read_input_tokens` / OpenAI `prompt_tokens_details.cached_tokens`).
    /// Lets an SRE/FinOps dashboard show cache savings. Bounded cardinality: it is
    /// just another `kind` label on the existing `rp_tokens_total` family.
    cached_tokens: AtomicU64,
    /// `rp_cost_micro_usd_total`.
    cost_micro_usd: AtomicU64,
    /// `rp_cache_events_total{type,result}` — [type][result].
    cache_events: [[AtomicU64; 2]; 2],
    /// `rp_provider_errors_total{provider}`.
    provider_errors: [AtomicU64; PROVIDERS.len()],
    /// `rp_hedged_wins_total`.
    hedged_wins: AtomicU64,
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Metrics {
    /// Construct a fresh zeroed table. `const` so the process-global static lives
    /// in the data segment; also used by tests to build a LOCAL table whose
    /// counts are deterministic (independent of the shared static).
    pub const fn new() -> Self {
        // `const fn` array init: `AtomicU64`/`LatencyHistogram` are not `Copy`, so
        // each nested array is spelled out. Verbose but compile-time-sized and
        // wait-free thereafter. Kept in lockstep with PROVIDERS.len() == 13.
        const fn zero_outcomes() -> [AtomicU64; OUTCOME_COUNT] {
            [
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
            ]
        }
        Metrics {
            requests: [
                zero_outcomes(),
                zero_outcomes(),
                zero_outcomes(),
                zero_outcomes(),
                zero_outcomes(),
                zero_outcomes(),
                zero_outcomes(),
                zero_outcomes(),
                zero_outcomes(),
                zero_outcomes(),
                zero_outcomes(),
                zero_outcomes(),
                zero_outcomes(),
            ],
            duration: [
                LatencyHistogram::new(),
                LatencyHistogram::new(),
                LatencyHistogram::new(),
                LatencyHistogram::new(),
                LatencyHistogram::new(),
                LatencyHistogram::new(),
                LatencyHistogram::new(),
                LatencyHistogram::new(),
                LatencyHistogram::new(),
                LatencyHistogram::new(),
                LatencyHistogram::new(),
                LatencyHistogram::new(),
                LatencyHistogram::new(),
            ],
            prompt_tokens: AtomicU64::new(0),
            completion_tokens: AtomicU64::new(0),
            cached_tokens: AtomicU64::new(0),
            cost_micro_usd: AtomicU64::new(0),
            cache_events: [
                [AtomicU64::new(0), AtomicU64::new(0)],
                [AtomicU64::new(0), AtomicU64::new(0)],
            ],
            provider_errors: [
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
            ],
            hedged_wins: AtomicU64::new(0),
        }
    }

    // --- recording (wait-free; called on the request path) -------------------

    /// Bump `rp_requests_total{provider,outcome}`. `provider` is bounded to the
    /// known set (unknown ⇒ `other`).
    pub fn inc_request(&self, provider: &str, outcome: Outcome) {
        self.requests[provider_idx(provider)][outcome.idx()].fetch_add(1, Ordering::Relaxed);
    }

    /// Observe one `rp_request_duration_ms` sample for `provider`.
    pub fn observe_duration(&self, provider: &str, ms: u64) {
        self.duration[provider_idx(provider)].observe(ms);
    }

    /// Add to `rp_tokens_total` (both kinds in one call — the natural shape of a
    /// usage event).
    pub fn add_tokens(&self, prompt: u64, completion: u64) {
        if prompt > 0 {
            self.prompt_tokens.fetch_add(prompt, Ordering::Relaxed);
        }
        if completion > 0 {
            self.completion_tokens
                .fetch_add(completion, Ordering::Relaxed);
        }
    }

    /// Add to `rp_tokens_total{kind=cached}` — prompt-cache READ tokens. Separate
    /// from `add_tokens` so the existing two-kind call sites are untouched; a
    /// no-op when zero (a non-cached request never bumps the series).
    pub fn add_cached_tokens(&self, cached: u64) {
        if cached > 0 {
            self.cached_tokens.fetch_add(cached, Ordering::Relaxed);
        }
    }

    /// Add to `rp_cost_micro_usd_total`.
    pub fn add_cost_micro_usd(&self, micro_usd: u64) {
        if micro_usd > 0 {
            self.cost_micro_usd.fetch_add(micro_usd, Ordering::Relaxed);
        }
    }

    /// Bump `rp_cache_events_total{type,result}`. `is_semantic` selects the type
    /// label; `hit` selects the result label.
    pub fn inc_cache(&self, is_semantic: bool, hit: bool) {
        let t = usize::from(is_semantic);
        let r = usize::from(!hit); // 0 = hit, 1 = miss (CACHE_RESULTS order)
        self.cache_events[t][r].fetch_add(1, Ordering::Relaxed);
    }

    /// Bump `rp_provider_errors_total{provider}`.
    pub fn inc_provider_error(&self, provider: &str) {
        self.provider_errors[provider_idx(provider)].fetch_add(1, Ordering::Relaxed);
    }

    /// Bump `rp_hedged_wins_total`.
    pub fn inc_hedged_win(&self) {
        self.hedged_wins.fetch_add(1, Ordering::Relaxed);
    }

    // --- render (off the hot path; may allocate) -----------------------------

    /// Render the full Prometheus text exposition (format `0.0.4`). Reads every
    /// atomic with `Relaxed` ordering — a scrape is a point-in-time snapshot and
    /// does not need to synchronize-with the writers. Series whose every sample
    /// is zero are still emitted (a stable scrape surface is friendlier to
    /// dashboards than series that blink in and out), except `provider`-labeled
    /// series are emitted for every known provider so the label set is stable.
    pub fn render(&self, shed_total: u64) -> String {
        let mut out = String::with_capacity(4096);

        // rp_requests_total
        let _ = writeln!(
            out,
            "# HELP rp_requests_total Total chat/completions requests by provider and terminal outcome."
        );
        let _ = writeln!(out, "# TYPE rp_requests_total counter");
        for (pi, provider) in PROVIDERS.iter().enumerate() {
            for outcome in Outcome::ALL {
                let v = self.requests[pi][outcome.idx()].load(Ordering::Relaxed);
                let _ = writeln!(
                    out,
                    "rp_requests_total{{provider=\"{}\",outcome=\"{}\"}} {}",
                    provider,
                    outcome.label(),
                    v
                );
            }
        }

        // rp_request_duration_ms (histogram)
        let _ = writeln!(
            out,
            "# HELP rp_request_duration_ms Upstream request latency (ms) by provider (buffered round-trip / stream TTFB)."
        );
        let _ = writeln!(out, "# TYPE rp_request_duration_ms histogram");
        for (pi, provider) in PROVIDERS.iter().enumerate() {
            let h = &self.duration[pi];
            let mut cumulative = 0u64;
            for (i, &bound) in LATENCY_BUCKETS_MS.iter().enumerate() {
                cumulative += h.buckets[i].load(Ordering::Relaxed);
                let _ = writeln!(
                    out,
                    "rp_request_duration_ms_bucket{{provider=\"{provider}\",le=\"{bound}\"}} {cumulative}"
                );
            }
            // The +Inf bucket equals the total count (cumulative + overflow).
            let count = h.count.load(Ordering::Relaxed);
            let _ = writeln!(
                out,
                "rp_request_duration_ms_bucket{{provider=\"{provider}\",le=\"+Inf\"}} {count}"
            );
            let _ = writeln!(
                out,
                "rp_request_duration_ms_sum{{provider=\"{}\"}} {}",
                provider,
                h.sum_ms.load(Ordering::Relaxed)
            );
            let _ = writeln!(
                out,
                "rp_request_duration_ms_count{{provider=\"{provider}\"}} {count}"
            );
        }

        // rp_tokens_total
        let _ = writeln!(
            out,
            "# HELP rp_tokens_total Total tokens processed by kind (prompt|completion|cached). `cached` is the prompt-cache READ subset of `prompt`."
        );
        let _ = writeln!(out, "# TYPE rp_tokens_total counter");
        let _ = writeln!(
            out,
            "rp_tokens_total{{kind=\"prompt\"}} {}",
            self.prompt_tokens.load(Ordering::Relaxed)
        );
        let _ = writeln!(
            out,
            "rp_tokens_total{{kind=\"completion\"}} {}",
            self.completion_tokens.load(Ordering::Relaxed)
        );
        let _ = writeln!(
            out,
            "rp_tokens_total{{kind=\"cached\"}} {}",
            self.cached_tokens.load(Ordering::Relaxed)
        );

        // rp_cost_micro_usd_total
        let _ = writeln!(
            out,
            "# HELP rp_cost_micro_usd_total Total attributed upstream cost in micro-USD (1e-6 USD)."
        );
        let _ = writeln!(out, "# TYPE rp_cost_micro_usd_total counter");
        let _ = writeln!(
            out,
            "rp_cost_micro_usd_total {}",
            self.cost_micro_usd.load(Ordering::Relaxed)
        );

        // rp_cache_events_total
        let _ = writeln!(
            out,
            "# HELP rp_cache_events_total Cache lookups by cache type and result."
        );
        let _ = writeln!(out, "# TYPE rp_cache_events_total counter");
        for (ti, ty) in CACHE_TYPES.iter().enumerate() {
            for (ri, result) in CACHE_RESULTS.iter().enumerate() {
                let _ = writeln!(
                    out,
                    "rp_cache_events_total{{type=\"{}\",result=\"{}\"}} {}",
                    ty,
                    result,
                    self.cache_events[ti][ri].load(Ordering::Relaxed)
                );
            }
        }

        // rp_provider_errors_total
        let _ = writeln!(
            out,
            "# HELP rp_provider_errors_total Failed upstream provider attempts by provider."
        );
        let _ = writeln!(out, "# TYPE rp_provider_errors_total counter");
        for (pi, provider) in PROVIDERS.iter().enumerate() {
            let _ = writeln!(
                out,
                "rp_provider_errors_total{{provider=\"{}\"}} {}",
                provider,
                self.provider_errors[pi].load(Ordering::Relaxed)
            );
        }

        // rp_hedged_wins_total
        let _ = writeln!(
            out,
            "# HELP rp_hedged_wins_total Successful responses served by a speculative hedge attempt (ADR-057)."
        );
        let _ = writeln!(out, "# TYPE rp_hedged_wins_total counter");
        let _ = writeln!(
            out,
            "rp_hedged_wins_total {}",
            self.hedged_wins.load(Ordering::Relaxed)
        );

        // shed_total — preserved from the legacy /metrics body. Kept under its
        // original (unprefixed) name AND mirrored under rp_shed_total so the new
        // surface is internally consistent without breaking any scraper/test that
        // already keys on `shed_total` (ADR-025 §3).
        let _ = writeln!(
            out,
            "# HELP rp_shed_total Cumulative ingress requests shed under capacity saturation (ADR-025)."
        );
        let _ = writeln!(out, "# TYPE rp_shed_total counter");
        let _ = writeln!(out, "rp_shed_total {shed_total}");
        let _ = writeln!(
            out,
            "# HELP shed_total Deprecated alias of rp_shed_total (kept for back-compat)."
        );
        let _ = writeln!(out, "# TYPE shed_total counter");
        let _ = writeln!(out, "shed_total {shed_total}");

        out
    }
}

// --- OTLP metrics export snapshot (PRD-009 FR-16 / routeplane#216) ------------

/// A point-in-time, owning copy of the metric table for the OTLP metrics
/// exporter (`otel.rs`). Reading is a wait-free batch of `Relaxed` loads off the
/// hot path; it carries the label TEXT so the serializer needs no metrics-module
/// internals. Mirrors the Prometheus `render` dimensions exactly — provider-only
/// (NO per-model/key label: the same bounded-cardinality / no-fingerprinting
/// decision the Prometheus surface made; per-model breakdown lives in the
/// per-event store, not the metric push).
pub struct MetricsSnapshot {
    /// `(provider, outcome, cumulative count)` for every nonzero request cell.
    pub requests: Vec<(&'static str, &'static str, u64)>,
    /// Per-provider request-duration histograms (only providers with traffic).
    pub durations: Vec<HistogramSnapshot>,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub cached_tokens: u64,
    pub cost_micro_usd: u64,
    pub shed_total: u64,
}

/// One provider's latency histogram, in OTLP shape.
pub struct HistogramSnapshot {
    pub provider: &'static str,
    /// `N+1` per-bucket counts — one per explicit bound plus the final `+Inf`
    /// (overflow) bucket. NOT cumulative (OTLP histograms are per-bucket, unlike
    /// Prometheus's `le` cumulative buckets).
    pub bucket_counts: Vec<u64>,
    /// The `N` explicit upper bounds, in ms.
    pub bounds: &'static [u64],
    pub sum_ms: u64,
    pub count: u64,
}

impl Metrics {
    /// Snapshot the whole table for the OTLP exporter. Cumulative since process
    /// start (the OTLP `Sum`s/`Histogram`s are `AGGREGATION_TEMPORALITY_CUMULATIVE`),
    /// matching the Prometheus counters' monotonic semantics. `shed_total` is the
    /// binary-level load-shed counter, passed in (it lives outside this module).
    pub fn snapshot(&self, shed_total: u64) -> MetricsSnapshot {
        let mut requests = Vec::new();
        for (pi, &provider) in PROVIDERS.iter().enumerate() {
            for outcome in Outcome::ALL {
                let c = self.requests[pi][outcome.idx()].load(Ordering::Relaxed);
                if c > 0 {
                    requests.push((provider, outcome.label(), c));
                }
            }
        }
        let mut durations = Vec::new();
        for (pi, &provider) in PROVIDERS.iter().enumerate() {
            let h = &self.duration[pi];
            let count = h.count.load(Ordering::Relaxed);
            if count == 0 {
                continue;
            }
            let mut bucket_counts: Vec<u64> = h
                .buckets
                .iter()
                .map(|b| b.load(Ordering::Relaxed))
                .collect();
            bucket_counts.push(h.overflow.load(Ordering::Relaxed)); // the +Inf bucket
            durations.push(HistogramSnapshot {
                provider,
                bucket_counts,
                bounds: LATENCY_BUCKETS_MS,
                sum_ms: h.sum_ms.load(Ordering::Relaxed),
                count,
            });
        }
        MetricsSnapshot {
            requests,
            durations,
            prompt_tokens: self.prompt_tokens.load(Ordering::Relaxed),
            completion_tokens: self.completion_tokens.load(Ordering::Relaxed),
            cached_tokens: self.cached_tokens.load(Ordering::Relaxed),
            cost_micro_usd: self.cost_micro_usd.load(Ordering::Relaxed),
            shed_total,
        }
    }
}

/// Process-global metrics table. A single `static` (no `Arc`, no lazy init) so
/// the recording paths reach it with zero indirection beyond a function call —
/// the same pattern as the binary-level `SHED_TOTAL`. `const`-constructed, so it
/// lives in the binary's data segment with no startup cost.
static METRICS: Metrics = Metrics::new();

/// Accessor for the process-global metrics table.
pub fn metrics() -> &'static Metrics {
    &METRICS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_idx_bounds_unknown_to_other() {
        assert_eq!(provider_idx("openai"), 0);
        assert_eq!(provider_idx("self_hosted"), 9);
        assert_eq!(provider_idx("cache"), 10);
        // Unknown / raw client-supplied / typo'd ⇒ the bounded `other` bucket,
        // never a new series (cardinality is fixed on an unauth endpoint).
        assert_eq!(provider_idx("totally-made-up-model-name"), OTHER_IDX);
        assert_eq!(provider_idx("(sovereign_block)"), OTHER_IDX);
        assert_eq!(provider_idx(""), OTHER_IDX);
    }

    #[test]
    fn histogram_buckets_place_observations_and_render_cumulatively() {
        let h = LatencyHistogram::new();
        h.observe(10); // -> le=50
        h.observe(75); // -> le=100
        h.observe(80); // -> le=100
        h.observe(120_000); // -> overflow (+Inf only)
        assert_eq!(h.count.load(Ordering::Relaxed), 4);
        assert_eq!(h.sum_ms.load(Ordering::Relaxed), 10 + 75 + 80 + 120_000);
        assert_eq!(h.buckets[0].load(Ordering::Relaxed), 1); // le=50
        assert_eq!(h.buckets[1].load(Ordering::Relaxed), 2); // le=100
        assert_eq!(h.overflow.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn render_emits_help_type_and_sample_lines_for_every_metric() {
        // Use a LOCAL table (not the process-global static) so the assertions are
        // deterministic regardless of what other tests recorded.
        let m = Metrics::new();
        m.inc_request("openai", Outcome::Success);
        m.inc_request("openai", Outcome::Success);
        m.inc_request("anthropic", Outcome::Error);
        m.inc_request("(sovereign_block)", Outcome::ResidencyBlocked); // -> other
        m.observe_duration("openai", 120);
        m.add_tokens(30, 12);
        m.add_cached_tokens(24); // prompt-cache READ subset
        m.add_cost_micro_usd(4567);
        m.inc_cache(false, true); // exact hit
        m.inc_cache(true, false); // semantic miss
        m.inc_provider_error("anthropic");
        m.inc_hedged_win();

        let body = m.render(7);

        // Valid exposition: HELP + TYPE for each metric family.
        for family in [
            "rp_requests_total",
            "rp_request_duration_ms",
            "rp_tokens_total",
            "rp_cost_micro_usd_total",
            "rp_cache_events_total",
            "rp_provider_errors_total",
            "rp_hedged_wins_total",
            "rp_shed_total",
            "shed_total",
        ] {
            assert!(
                body.contains(&format!("# HELP {family} ")),
                "missing HELP for {family}"
            );
            assert!(
                body.contains(&format!("# TYPE {family} ")),
                "missing TYPE for {family}"
            );
        }

        // Sample lines with the expected values.
        assert!(body.contains("rp_requests_total{provider=\"openai\",outcome=\"success\"} 2"));
        assert!(body.contains("rp_requests_total{provider=\"anthropic\",outcome=\"error\"} 1"));
        assert!(
            body.contains("rp_requests_total{provider=\"other\",outcome=\"residency_blocked\"} 1")
        );
        // Histogram: 120ms lands in le=250 (cumulative through 50,100 is 0).
        assert!(body.contains("rp_request_duration_ms_bucket{provider=\"openai\",le=\"50\"} 0"));
        assert!(body.contains("rp_request_duration_ms_bucket{provider=\"openai\",le=\"250\"} 1"));
        assert!(body.contains("rp_request_duration_ms_bucket{provider=\"openai\",le=\"+Inf\"} 1"));
        assert!(body.contains("rp_request_duration_ms_sum{provider=\"openai\"} 120"));
        assert!(body.contains("rp_request_duration_ms_count{provider=\"openai\"} 1"));
        assert!(body.contains("rp_tokens_total{kind=\"prompt\"} 30"));
        assert!(body.contains("rp_tokens_total{kind=\"completion\"} 12"));
        assert!(body.contains("rp_tokens_total{kind=\"cached\"} 24"));
        assert!(body.contains("rp_cost_micro_usd_total 4567"));
        assert!(body.contains("rp_cache_events_total{type=\"exact\",result=\"hit\"} 1"));
        assert!(body.contains("rp_cache_events_total{type=\"semantic\",result=\"miss\"} 1"));
        assert!(body.contains("rp_provider_errors_total{provider=\"anthropic\"} 1"));
        assert!(body.contains("rp_hedged_wins_total 1"));
        assert!(body.contains("rp_shed_total 7"));
        assert!(body.contains("shed_total 7"));

        // Cardinality: NO raw-model label anywhere, and the only `provider`
        // values are from the bounded set (no sentinel string leaked through).
        assert!(!body.contains("model="));
        assert!(!body.contains("(sovereign_block)"));
    }
}
