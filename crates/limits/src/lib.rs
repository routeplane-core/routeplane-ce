//! Routeplane budget & rate-limit enforcement — the hot-path admission engine
//! ([PRD-008] §9, [ADR-023] Mode L).
//!
//! # What this crate owns
//! Per-(scope, policy) atomic counters and the **check-before / settle-after**
//! admission decision. The proxy owns eligibility, `router` owns ordering/health,
//! this crate owns **admission** ([ADR-023] Consequences — the crate-boundary
//! doctrine extends cleanly).
//!
//! # Non-negotiable invariants (mirrors `crates/router`)
//! - **Lock-free hot path.** All counter state is `AtomicU64`; admission performs
//!   only atomic loads / CAS on pre-resolved slots — no mutex, no allocation, no
//!   I/O ([ADR-023] §1, NFR-1).
//! - **Injectable clock.** Every method takes `now_ms` (unix millis) as a
//!   parameter; the engine never calls `Instant::now()` itself, so window
//!   rollover and budget reset are deterministic under test (the router
//!   discipline). [`now_unix_ms`] is provided for the caller.
//! - **Saturating arithmetic, never panics.** Counts saturate; date math falls
//!   back rather than `unwrap`-ing on a request thread.
//!
//! # Window mechanics ([ADR-023] §2)
//! Fixed windows with epoch-CAS rollover, packed `(epoch:24 | count:40)` in a
//! single atomic. A request observing a stale epoch CAS-rolls to the new epoch
//! with a reset count; losers retry the load. This is the same atomic
//! state-machine discipline as the `CircuitBreaker`. Fixed windows (not GCRA /
//! token-bucket) are the ratified choice precisely because the
//! `x-ratelimit-*-reset` / `Retry-After` values fall directly out of the data
//! structure ([ADR-023] §2 — deviation note in the PR body: the build brief said
//! "token-bucket"; canon is fixed-window, which we follow). The minute epoch is
//! truncated to 24 bits — adjacency (and therefore rollover/equality) is
//! preserved; the only collision is between windows ~31 years apart.
//!
//! # Mode L only (this build)
//! Local-only enforcement, **zero standing cost** ([ADR-023] §6). Cross-replica
//! reconciliation (Mode R) is a named seam ([`LimitReconciler`]) with an
//! in-memory stub; the durable Redis/Postgres path and CP integration are
//! explicitly out of scope.
//!
//! [PRD-008]: ../../../docs/product/prd/008-model-catalog-keys-budgets-rate-limits.md
//! [ADR-023]: ../../../docs/adr/023-hot-path-budget-rate-limit-enforcement.md

use arc_swap::ArcSwap;
use chrono::{Datelike, NaiveDate, TimeZone, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// Clock
// ---------------------------------------------------------------------------

/// Wall-clock unix millis for the caller to feed into the engine. The engine
/// itself never reads the clock (injectable-clock discipline); this is the one
/// place a real timestamp is produced, on the proxy hot path, once per request.
pub fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Config (rides the VirtualKey via serde; absent = unlimited)
// ---------------------------------------------------------------------------

/// `hard` rejects; `soft` (alias `shadow`) counts and never blocks.
///
/// Default is `Hard`: in collapsed-config mode (no control plane) an operator
/// who authors a limit into `keys.json` means it. This is a deliberate deviation
/// from [PRD-008] FR-13/FR-15's soft/shadow defaults (which target CP-created
/// objects, to avoid a surprise hard-stop on rollout); argued in the PR body.
/// `soft`/`shadow` remain opt-in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LimitMode {
    #[default]
    Hard,
    #[serde(alias = "shadow")]
    Soft,
}

impl LimitMode {
    #[inline]
    fn is_hard(self) -> bool {
        matches!(self, LimitMode::Hard)
    }
}

/// The limits attached to one virtual key: an optional per-key policy, an optional
/// per-tenant policy, and an optional map of **per-model** policies. Per-tenant
/// counters are shared across every key of the same tenant (declare the tenant
/// policy on any one of the tenant's keys).
///
/// `model_limits` maps a **model-id substring** (the same most-specific-first
/// substring convention as the [`PRICE_BOOK`], e.g. `"gpt-4"` matches `gpt-4`,
/// `gpt-4o`, `gpt-4-turbo`) to a [`LimitPolicy`] capping requests/tokens/spend for
/// that model independently of the overall key/tenant budget. The counter for a
/// model policy is per-**(this key, matched model)** — see [`LimitScope::Model`].
/// Default **absent ⇒ no per-model limits ⇒ byte-identical** to the legacy path:
/// the model-scope machinery is gated entirely on this map being non-empty, so an
/// unconfigured key touches zero model-scoped atomics on the hot path.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KeyLimits {
    #[serde(default)]
    pub key: Option<LimitPolicy>,
    #[serde(default)]
    pub tenant: Option<LimitPolicy>,
    /// model-id substring → policy. Absent/empty ⇒ no per-model limits.
    #[serde(default)]
    pub model_limits: HashMap<String, LimitPolicy>,
}

impl KeyLimits {
    fn is_empty(&self) -> bool {
        self.key.as_ref().is_none_or(LimitPolicy::is_empty)
            && self.tenant.as_ref().is_none_or(LimitPolicy::is_empty)
            && self.model_limits_is_empty()
    }

    /// `true` when no model pattern configures an actual limit (so the whole
    /// model-scope path can be skipped). An empty map, or one whose every policy is
    /// itself empty, counts as no per-model limits.
    fn model_limits_is_empty(&self) -> bool {
        self.model_limits.is_empty() || self.model_limits.values().all(LimitPolicy::is_empty)
    }
}

/// A reusable policy: an id (for the `x-routeplane-limit-policy` header), an
/// optional [`RateLimits`] block and an optional [`BudgetLimits`] block.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LimitPolicy {
    #[serde(default)]
    pub policy_id: Option<String>,
    #[serde(default)]
    pub rate: Option<RateLimits>,
    #[serde(default)]
    pub budget: Option<BudgetLimits>,
}

impl LimitPolicy {
    fn is_empty(&self) -> bool {
        self.rate.as_ref().is_none_or(RateLimits::is_empty)
            && self.budget.as_ref().is_none_or(BudgetLimits::is_empty)
    }
}

/// Per-minute fixed-window rate limits (requests and/or tokens). Hour/day windows
/// ([PRD-008] FR-14) are a follow-up; this build ships per-minute.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RateLimits {
    #[serde(default)]
    pub requests_per_min: Option<u64>,
    #[serde(default)]
    pub tokens_per_min: Option<u64>,
    #[serde(default)]
    pub mode: LimitMode,
}

impl RateLimits {
    fn is_empty(&self) -> bool {
        self.requests_per_min.is_none() && self.tokens_per_min.is_none()
    }
}

/// Calendar-anchored spend/token budgets ([PRD-008] §7.3). Cost is integer
/// micro-USD (no floats in money — [ADR-023] §1). `anchor_offset_minutes` is the
/// fixed offset from UTC for the period boundary (IST = 330; [PRD-008] FR-10a) —
/// a fixed offset, not a TZ database, which is exact for India (no DST) and a
/// documented simplification elsewhere.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BudgetLimits {
    #[serde(default)]
    pub cost_micro_usd_daily: Option<u64>,
    #[serde(default)]
    pub cost_micro_usd_monthly: Option<u64>,
    #[serde(default)]
    pub tokens_daily: Option<u64>,
    #[serde(default)]
    pub tokens_monthly: Option<u64>,
    #[serde(default)]
    pub anchor_offset_minutes: i32,
    #[serde(default)]
    pub mode: LimitMode,
    /// Optional **soft spend threshold**, expressed as integer **permille**
    /// (‰, 0–1000) of the *cost* budget — `800` = 80 %. When set, crossing it
    /// fires a one-shot spend alert per budget window (off-path) and surfaces a
    /// warning header on responses while in the warning zone; the hard cap
    /// (402) is unchanged. Permille (not float percent) keeps the
    /// consumed-fraction comparison exact integer math, consistent with the
    /// integer-money discipline ([ADR-023] §1). Applies to the daily and
    /// monthly **cost** budgets only (the LiteLLM `soft_budget` analogue —
    /// spend, not tokens). Default **unset** ⇒ no soft alerting ⇒ byte-identical
    /// to the legacy budget path. Values > 1000 are clamped to 1000 at build
    /// (never disables the eventual hard cap); `0` means "warn immediately on any
    /// spend".
    #[serde(default)]
    pub soft_threshold_permille: Option<u16>,

    // --- Multi-currency / INR budget authoring (PRD-008 FR-22/23, re-added atop
    //     #170). A cost cap is authored EITHER in micro-USD (above) OR a display
    //     currency, never both per window; the display currency is resolved to
    //     integer micro-USD at config load (off the hot path, no float, no
    //     hot-path FX) by `resolved_cost_caps`. Default-absent ⇒ byte-identical. ---
    /// ISO-4217 display currency a cost cap is authored in (e.g. `"INR"`). Absent
    /// ⇒ micro-USD authoring. Display/audit label only — enforcement is currency-free.
    #[serde(default)]
    pub authored_currency: Option<String>,
    /// Daily cost cap in whole units of `authored_currency`, converted to micro-USD
    /// at load via `fx_units_per_usd`.
    #[serde(default)]
    pub authored_cost_daily: Option<u64>,
    /// Monthly cost cap in whole units of `authored_currency`.
    #[serde(default)]
    pub authored_cost_monthly: Option<u64>,
    /// The PINNED FX rate — whole units of `authored_currency` per 1 USD (e.g. `83`
    /// for INR). Required (non-zero) whenever any authored cost cap is present:
    /// `micro_usd = authored × 1_000_000 / fx_units_per_usd` (floored integer).
    #[serde(default)]
    pub fx_units_per_usd: Option<u64>,
    /// As-of date the pinned rate was set (free-form ISO date) — display/audit only.
    #[serde(default)]
    pub fx_as_of: Option<String>,
}

/// A config-time (off-hot-path) budget-authoring error, surfaced to the binary at
/// startup so an unsafe budget config fails closed (refuse to start) — a
/// silently-uncapped or silently-deny-all spend cap is a money bug ([ADR-023] §1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BudgetConfigError {
    /// A window declared both a micro-USD and an authored (display-currency) cap.
    CostCapAuthoredTwice,
    /// An authored cost cap was present with no pinned `fx_units_per_usd`.
    AuthoredCostWithoutFxRate,
    /// `fx_units_per_usd` was zero (would convert to a deny-all $0 cap).
    ZeroFxRate,
}

impl std::fmt::Display for BudgetConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CostCapAuthoredTwice => {
                write!(
                    f,
                    "a window declares both a micro-USD and an authored cost cap"
                )
            }
            Self::AuthoredCostWithoutFxRate => {
                write!(f, "an authored cost cap requires a pinned fx_units_per_usd")
            }
            Self::ZeroFxRate => write!(f, "fx_units_per_usd must be non-zero"),
        }
    }
}

impl BudgetLimits {
    /// Resolve the effective **micro-USD** cost caps for the (daily, monthly)
    /// windows (FR-22). A window authored directly in micro-USD passes through; a
    /// window authored in a display currency is converted with the pinned rate;
    /// declaring both for one window is an error. Run once at config load — never
    /// on the hot path.
    pub fn resolved_cost_caps(&self) -> Result<(Option<u64>, Option<u64>), BudgetConfigError> {
        let convert = |authored: u64| -> Result<u64, BudgetConfigError> {
            match self.fx_units_per_usd {
                None => Err(BudgetConfigError::AuthoredCostWithoutFxRate),
                Some(0) => Err(BudgetConfigError::ZeroFxRate),
                Some(rate) => Ok(authored.saturating_mul(1_000_000) / rate),
            }
        };
        let resolve_window =
            |micro: Option<u64>, authored: Option<u64>| -> Result<Option<u64>, BudgetConfigError> {
                match (micro, authored) {
                    (Some(_), Some(_)) => Err(BudgetConfigError::CostCapAuthoredTwice),
                    (Some(m), None) => Ok(Some(m)),
                    (None, Some(a)) => Ok(Some(convert(a)?)),
                    (None, None) => Ok(None),
                }
            };
        let daily = resolve_window(self.cost_micro_usd_daily, self.authored_cost_daily)?;
        let monthly = resolve_window(self.cost_micro_usd_monthly, self.authored_cost_monthly)?;
        Ok((daily, monthly))
    }
}

impl BudgetLimits {
    fn is_empty(&self) -> bool {
        self.cost_micro_usd_daily.is_none()
            && self.cost_micro_usd_monthly.is_none()
            && self.tokens_daily.is_none()
            && self.tokens_monthly.is_none()
    }
}

// ---------------------------------------------------------------------------
// Pricing (PLACEHOLDER pending FR-24 vendored pricing book)
// ---------------------------------------------------------------------------

pub mod pricing;

/// Configurable FX rate table for multi-currency cost attribution ([PRD-015]
/// FR-2). Hot-swappable behind an `ArcSwap` (mirrors the prompt/policy
/// registries); ship-dark default is byte-identical to the legacy placeholder.
/// See the module docs.
pub mod fx;

/// Auth-failure rate limiting (security gap R0.2) — a lock-free, fixed-memory
/// per-source-IP failed-authentication tracker, reusing this crate's packed
/// atomic / injectable-clock discipline. Consumed by the gateway's auth
/// middleware to throttle brute-force key spraying with escalating backoff.
pub mod auth_failures;

/// Mode D — **opt-in** distributed (Redis-backed) rate limiting ([ADR-056]).
/// Compiled only under the `redis-limits` cargo feature; OFF by default so the
/// standard build is $0-idle and byte-identical to Mode L. See the module docs.
#[cfg(feature = "redis-limits")]
pub mod distributed;

/// Per-model price book (ADR-051, grounds [PRD-015] FR-1). Each row is
/// `(model_substring, prompt_rate, completion_rate)` where the rates are integer
/// **micro-USD per million tokens** — so sub-$1/M SKUs are representable exactly
/// (e.g. `$0.15/M` → `150_000`), unlike a whole-micro-USD-per-token rate.
///
/// Ordered **most-specific first**: the first row whose substring the served model
/// contains wins, so `gpt-4o-mini` resolves before `gpt-4o` and `claude-3-opus`
/// before generic `claude`. Rates are dated public list prices (mid-2026) — the
/// vendored MIT dataset ([PRD-008] FR-24) is the durable replacement.
///
/// `gpt-4o` (3M/10M) and `claude-3` (3M/15M) are **pinned** at the legacy
/// placeholder rates: they are the only model strings the A/B golden corpus
/// exercises, so this keeps that snapshot byte-identical and leaves existing
/// budget burn rates unchanged (ADR-051 §4).
const PRICE_BOOK: &[(&str, u64, u64)] = &[
    // OpenAI
    ("gpt-4o-mini", 150_000, 600_000),
    ("gpt-4o", 3_000_000, 10_000_000), // pinned (legacy parity)
    ("o1-mini", 1_100_000, 4_400_000),
    ("o1", 15_000_000, 60_000_000),
    ("gpt-4-turbo", 10_000_000, 30_000_000),
    ("gpt-3.5", 500_000, 1_500_000),
    // Anthropic
    ("claude-3-5-haiku", 800_000, 4_000_000),
    ("claude-3-haiku", 250_000, 1_250_000),
    ("claude-3-opus", 15_000_000, 75_000_000),
    ("claude-3-5-sonnet", 3_000_000, 15_000_000),
    ("claude-3", 3_000_000, 15_000_000), // pinned (legacy parity)
    ("claude", 3_000_000, 15_000_000),
    // Google
    ("gemini-1.5-flash", 75_000, 300_000),
    ("gemini-2.0-flash", 100_000, 400_000),
    ("gemini-1.5-pro", 1_250_000, 5_000_000),
    ("gemini", 1_250_000, 5_000_000),
];

/// Default rates (micro-USD per million tokens) for a model not in [`PRICE_BOOK`].
const DEFAULT_RATE: (u64, u64) = (2_000_000, 6_000_000);

/// Resolve the per-model `(prompt_rate, completion_rate)` from the [`PRICE_BOOK`]
/// in **micro-USD per million tokens**, using the same most-specific-first
/// substring match as [`estimate_cost_micro_usd`]. Returns `None` when the model
/// matches no row (so the caller can decide whether to fall back to
/// [`DEFAULT_RATE`] or omit cost metadata) — distinct from the cost estimator,
/// which always returns a number. Pure read of static data; never panics.
///
/// Exposed for the catalog-enrichment surface (ADR-035 `cost`): the `/v1/models`
/// metadata derives `input/output_per_1k_micro_usd` from these per-million rates
/// (`per_1k = per_million / 1000`) rather than duplicating the rate table.
pub fn price_for(model: &str) -> Option<(u64, u64)> {
    PRICE_BOOK
        .iter()
        .find(|(pat, _, _)| model.contains(pat))
        .map(|(_, p, c)| (*p, *c))
}

/// Estimate request cost in **integer micro-USD** from token counts, using the
/// per-model [`PRICE_BOOK`] (ADR-051). The canonical unit is unchanged — only the
/// rate precision is finer. Saturating; never panics.
pub fn estimate_cost_micro_usd(model: &str, prompt_tokens: u32, completion_tokens: u32) -> u64 {
    let (prompt_rate, completion_rate) = price_for(model).unwrap_or(DEFAULT_RATE);
    let micro = (prompt_tokens as u64)
        .saturating_mul(prompt_rate)
        .saturating_add((completion_tokens as u64).saturating_mul(completion_rate));
    // Rates are per *million* tokens; bring back to micro-USD.
    micro / 1_000_000
}

// ---------------------------------------------------------------------------
// Packed atomic window counter — (epoch:24 | count:40)
// ---------------------------------------------------------------------------

const EPOCH_BITS: u64 = 24;
const COUNT_BITS: u64 = 40;
const COUNT_MASK: u64 = (1 << COUNT_BITS) - 1;
const EPOCH_MASK: u64 = (1 << EPOCH_BITS) - 1;

#[inline]
fn pack(epoch: u64, count: u64) -> u64 {
    ((epoch & EPOCH_MASK) << COUNT_BITS) | (count & COUNT_MASK)
}

#[inline]
fn unpack(p: u64) -> (u64, u64) {
    (p >> COUNT_BITS, p & COUNT_MASK)
}

/// A single fixed-window counter. Packs the (truncated) window epoch and the
/// in-window count into one `AtomicU64`, rolling over via CAS. Count saturates at
/// `COUNT_MASK` (~1.1e12 — for micro-USD budgets that is a ~$1.1M-per-window cap,
/// documented; for tokens/requests it is unreachable).
#[derive(Debug)]
struct WindowCounter {
    packed: AtomicU64,
}

impl WindowCounter {
    fn new() -> Self {
        Self {
            packed: AtomicU64::new(0),
        }
    }

    /// Current count for `epoch` (0 if the stored window is stale, i.e. rolled).
    fn peek(&self, epoch: u64) -> u64 {
        let (e, c) = unpack(self.packed.load(Ordering::Acquire));
        if e == (epoch & EPOCH_MASK) {
            c
        } else {
            0
        }
    }

    /// Atomically consume `amount` against `limit` for `epoch`. Admits when the
    /// pre-consume base is `< limit` (so a single request may overshoot by its
    /// own `amount` — the FR-16 bound for amount>1; amount=1 is exact). Returns
    /// `Ok(new_count)` or `Err(base)` when already at/over the limit.
    fn try_consume(&self, amount: u64, limit: u64, epoch: u64) -> Result<u64, u64> {
        let et = epoch & EPOCH_MASK;
        loop {
            let cur = self.packed.load(Ordering::Acquire);
            let (e, c) = unpack(cur);
            let base = if e == et { c } else { 0 };
            if base >= limit {
                return Err(base);
            }
            let next = base.saturating_add(amount).min(COUNT_MASK);
            let newp = pack(et, next);
            if self
                .packed
                .compare_exchange_weak(cur, newp, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return Ok(next);
            }
        }
    }

    /// Settle-after debit: unconditionally add `amount` to the current window
    /// (saturating), rolling a stale epoch to a fresh window seeded at `amount`.
    fn debit(&self, amount: u64, epoch: u64) {
        let et = epoch & EPOCH_MASK;
        loop {
            let cur = self.packed.load(Ordering::Acquire);
            let (e, c) = unpack(cur);
            let base = if e == et { c } else { 0 };
            let next = base.saturating_add(amount).min(COUNT_MASK);
            let newp = pack(et, next);
            if self
                .packed
                .compare_exchange_weak(cur, newp, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return;
            }
        }
    }
}

/// One soft-threshold "alerted" flag, packed `(epoch:24 | fired:1)` in a single
/// `AtomicU64` — the same epoch-CAS rollover discipline as [`WindowCounter`], so
/// the flag resets automatically when the budget window rolls (a fresh epoch
/// observes `fired = 0`). Edge-triggered: the FIRST caller to observe the soft
/// threshold crossed in a given window wins the CAS and gets `true` (fire once);
/// every subsequent caller in the same window gets `false` (no alert storm).
/// Lock-free, no allocation.
#[derive(Debug)]
struct AlertFlag {
    packed: AtomicU64,
}

impl AlertFlag {
    fn new() -> Self {
        Self {
            packed: AtomicU64::new(0),
        }
    }

    /// Attempt to claim the one-shot fire for `epoch`. Returns `true` exactly
    /// once per (window) epoch — the edge trigger. A stale epoch is treated as a
    /// fresh window (the flag implicitly reset on rollover). Losers of the CAS
    /// retry their load and observe the now-set flag, returning `false`.
    fn try_fire(&self, epoch: u64) -> bool {
        let et = epoch & EPOCH_MASK;
        loop {
            let cur = self.packed.load(Ordering::Acquire);
            let (e, fired) = unpack(cur);
            // Same epoch AND already fired ⇒ nothing to do (no storm).
            if e == et && fired != 0 {
                return false;
            }
            let newp = pack(et, 1);
            if self
                .packed
                .compare_exchange_weak(cur, newp, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return true;
            }
        }
    }
}

/// Consumed fraction in **permille** (‰), integer-exact: `consumed * 1000 /
/// budget`, saturating. A zero budget yields `1000` (fully consumed) so a
/// `0`-budget never spuriously sits below threshold. Capped at `1000`.
#[inline]
fn consumed_permille(consumed: u64, budget: u64) -> u64 {
    if budget == 0 {
        return 1000;
    }
    (consumed.saturating_mul(1000) / budget).min(1000)
}

// ---------------------------------------------------------------------------
// Period epochs + reset boundaries (off-hot-path calendar math)
// ---------------------------------------------------------------------------

const MINUTE_MS: i64 = 60_000;
const DAY_MS: i64 = 86_400_000;

#[inline]
fn minute_epoch(now_ms: u64) -> u64 {
    now_ms / MINUTE_MS as u64
}

#[inline]
fn minute_reset_ms(now_ms: u64) -> u64 {
    (minute_epoch(now_ms) + 1) * MINUTE_MS as u64
}

#[inline]
fn day_epoch(now_ms: u64, offset_min: i32) -> u64 {
    let shifted = now_ms as i64 + offset_min as i64 * MINUTE_MS;
    shifted.div_euclid(DAY_MS).max(0) as u64
}

fn day_reset_ms(now_ms: u64, offset_min: i32) -> u64 {
    let off = offset_min as i64 * MINUTE_MS;
    let shifted = now_ms as i64 + off;
    let next_local_midnight = (shifted.div_euclid(DAY_MS) + 1) * DAY_MS;
    (next_local_midnight - off).max(0) as u64
}

fn month_parts(now_ms: u64, offset_min: i32) -> (i32, u32) {
    let shifted = now_ms as i64 + offset_min as i64 * MINUTE_MS;
    match Utc.timestamp_millis_opt(shifted).single() {
        Some(dt) => (dt.year(), dt.month()),
        None => (1970, 1),
    }
}

fn month_epoch(now_ms: u64, offset_min: i32) -> u64 {
    let (y, m) = month_parts(now_ms, offset_min);
    ((y as i64) * 12 + (m as i64 - 1)).max(0) as u64
}

fn month_reset_ms(now_ms: u64, offset_min: i32) -> u64 {
    let (y, m) = month_parts(now_ms, offset_min);
    let (ny, nm) = if m >= 12 { (y + 1, 1) } else { (y, m + 1) };
    let local_next = NaiveDate::from_ymd_opt(ny, nm, 1)
        .and_then(|d| d.and_hms_opt(0, 0, 0))
        .map(|ndt| ndt.and_utc().timestamp_millis());
    match local_next {
        Some(ms) => (ms - offset_min as i64 * MINUTE_MS).max(0) as u64,
        // Fallback: never panic on a request thread — approximate +31 days.
        None => now_ms.saturating_add(31 * DAY_MS as u64),
    }
}

#[inline]
fn reset_secs(reset_ms: u64, now_ms: u64) -> u64 {
    let delta = reset_ms.saturating_sub(now_ms);
    delta.div_ceil(1000).max(1)
}

// ---------------------------------------------------------------------------
// Verdict surface (consumed by the proxy to build OpenAI-shaped responses)
// ---------------------------------------------------------------------------

/// Which limit fired — drives the `x-routeplane-limit-type` header value and the
/// 429-vs-402 split.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LimitKind {
    RateRequests,
    RateTokens,
    BudgetCost,
    BudgetTokens,
}

impl LimitKind {
    /// The `x-routeplane-limit-type` header value.
    pub fn header(self) -> &'static str {
        match self {
            LimitKind::RateRequests => "rate_limit_requests",
            LimitKind::RateTokens => "rate_limit_tokens",
            LimitKind::BudgetCost => "budget_cost",
            LimitKind::BudgetTokens => "budget_tokens",
        }
    }

    fn is_budget(self) -> bool {
        matches!(self, LimitKind::BudgetCost | LimitKind::BudgetTokens)
    }
}

/// The attachment scope the breached policy lives at — the
/// `x-routeplane-limit-scope` header value. (Workspace/Platform reserved for the
/// CP hierarchy; this build resolves Key + Tenant + Model.)
///
/// `Model` is a per-**(owning scope, served model)** cap (PRD-008 §9 best-of-breed
/// parity with LiteLLM/Portkey model-level RPM/TPM): an operator caps an expensive
/// model independently of the overall key/tenant budget. The owning scope identity
/// (the key or tenant the model policy is declared on) is folded into the counter
/// key, so tenant A's `gpt-4` budget is separate from tenant B's, and separate from
/// the overall key budget. Incremental on ADR-023 (no new ADR — same atomic window
/// discipline, no new standing cost).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LimitScope {
    Key,
    Tenant,
    Model,
}

impl LimitScope {
    pub fn header(self) -> &'static str {
        match self {
            LimitScope::Key => "key",
            LimitScope::Tenant => "tenant",
            LimitScope::Model => "model",
        }
    }
}

/// The per-(scope) request-rate configuration exported for the distributed
/// engine ([ADR-056], Mode D). The local engine keeps its counters private;
/// Mode D needs only the *global limit* + *structural identity* to enforce the
/// same cap in Redis, so this is a small read-only projection — no atomics, no
/// counters. `identity`/`scope` reproduce Mode L's *structural* isolation exactly
/// (a different key/tenant/model resolves a different counter) so the two modes
/// are interchangeable regardless of `policy_id` reuse.
#[derive(Debug, Clone)]
pub struct RequestRateSpec {
    pub scope: LimitScope,
    /// The operator-facing policy id, surfaced in the 429 breach header
    /// ([PRD-008] FR-18). This is NOT the counter key: an operator may reuse one
    /// id (e.g. `"standard"`) across many tenants, so it cannot isolate counters —
    /// see [`RequestRateSpec::identity`].
    pub policy_id: String,
    /// The STRUCTURAL counter identity (always embeds the owning key/tenant/model,
    /// e.g. `tenant:<tenant_id>`), independent of any reusable `policy_id`. Mode L
    /// isolates counters structurally in the registry maps; Mode D keys its Redis
    /// counter off this same identity so two tenants that share a `policy_id` never
    /// collapse onto one counter (the [ADR-023] structural-isolation / [ADR-064]
    /// cross-tenant rule).
    pub identity: String,
    /// The GLOBAL (fleet-wide) per-window request limit — the configured cap, NOT
    /// the Mode-L per-replica share-clamp. Mode D counts globally in Redis across
    /// every replica, so it must enforce the configured value; the share-clamp
    /// ([ADR-023] §6) is a Mode-L-only correction for per-replica local counting
    /// and would under-enforce the global cap by `max_replicas×` if applied here.
    pub limit: u64,
    /// `false` for soft/shadow (count, never block).
    pub hard: bool,
}

/// How rate/budget admission is enforced — the unified mode selector the caller
/// switches on. `Local` is the lock-free per-replica engine (the default, [ADR-023]
/// Mode L); `Distributed` is the opt-in Redis-backed engine ([ADR-056] Mode D),
/// which carries its own fail-open/closed fallback policy. Kept here (not in the
/// feature-gated module) so the caller can name the default without the feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EnforcementMode {
    /// Per-replica atomic enforcement — zero standing cost (default).
    #[default]
    Local,
    /// Distributed enforcement via Redis (opt-in, standing cost). Only
    /// constructible when the `redis-limits` feature is built; the variant is
    /// always present so config can name it.
    Distributed,
}

/// A hard-limit breach, carrying everything the proxy needs to build the
/// OpenAI-shaped 429/402 envelope + truthful headers ([PRD-008] FR-17/18/19).
#[derive(Debug, Clone)]
pub struct Breach {
    kind: LimitKind,
    scope: LimitScope,
    policy_id: String,
    limit: u64,
    reset_ms: Option<u64>,
    now_ms: u64,
}

impl Breach {
    /// A request-rate breach produced by the distributed engine ([ADR-056]).
    /// Shaped identically to a Mode L `RateRequests` breach (same headers,
    /// `Retry-After` at the minute boundary) so the proxy's 429 path is unchanged.
    pub fn distributed_rate(scope: LimitScope, policy_id: String, limit: u64, now_ms: u64) -> Self {
        Breach {
            kind: LimitKind::RateRequests,
            scope,
            policy_id,
            limit,
            reset_ms: Some(minute_reset_ms(now_ms)),
            now_ms,
        }
    }

    /// A fail-CLOSED breach when Redis is unavailable and the configured fallback
    /// is `DistributedFallback::Closed` ([ADR-056]). Surfaced as a 429 (the
    /// caller must not be told the cap was actually hit), with a short
    /// `Retry-After` so the client retries after the next window.
    pub fn distributed_unavailable(now_ms: u64) -> Self {
        Breach {
            kind: LimitKind::RateRequests,
            scope: LimitScope::Key,
            policy_id: "distributed_unavailable".to_string(),
            limit: 0,
            reset_ms: Some(minute_reset_ms(now_ms)),
            now_ms,
        }
    }

    pub fn kind(&self) -> LimitKind {
        self.kind
    }
    pub fn kind_header(&self) -> &'static str {
        self.kind.header()
    }
    pub fn scope_header(&self) -> &'static str {
        self.scope.header()
    }
    pub fn policy_id(&self) -> &str {
        &self.policy_id
    }
    pub fn limit(&self) -> u64 {
        self.limit
    }
    pub fn is_budget(&self) -> bool {
        self.kind.is_budget()
    }
    /// Seconds until the breached window/period resets (`Retry-After`). `None`
    /// only for lifetime caps (not modelled here, so always `Some`).
    pub fn retry_after_secs(&self) -> Option<u64> {
        self.reset_ms.map(|r| reset_secs(r, self.now_ms))
    }
    /// Alias used for `x-ratelimit-reset-requests`.
    pub fn reset_secs(&self) -> Option<u64> {
        self.retry_after_secs()
    }
    /// Human-readable `error.message` for the OpenAI envelope.
    pub fn message(&self) -> String {
        match self.kind {
            LimitKind::RateRequests => format!(
                "Rate limit exceeded: request limit of {} per minute reached for the {} scope (policy '{}')",
                self.limit,
                self.scope.header(),
                self.policy_id
            ),
            LimitKind::RateTokens => format!(
                "Rate limit exceeded: token limit of {} per minute reached for the {} scope (policy '{}')",
                self.limit,
                self.scope.header(),
                self.policy_id
            ),
            LimitKind::BudgetCost => format!(
                "Budget exceeded: spend cap reached for the {} scope (policy '{}')",
                self.scope.header(),
                self.policy_id
            ),
            LimitKind::BudgetTokens => format!(
                "Budget exceeded: token cap of {} reached for the {} scope (policy '{}')",
                self.limit,
                self.scope.header(),
                self.policy_id
            ),
        }
    }
}

/// Advisory headers for a successful response ([PRD-008] FR-19). Every field is
/// `None` when the corresponding limit is unconfigured, so a legacy key produces
/// an empty header set (byte-identical). Values reflect the local replica's view
/// (documented "approximate" under the ADR-023 consistency model).
#[derive(Debug, Clone, Copy, Default)]
pub struct Advisory {
    pub limit_requests: Option<u64>,
    pub remaining_requests: Option<u64>,
    pub reset_requests_secs: Option<u64>,
    pub limit_tokens: Option<u64>,
    pub remaining_tokens: Option<u64>,
    pub budget_remaining_micro_usd: Option<u64>,
}

impl Advisory {
    pub fn is_empty(&self) -> bool {
        self.limit_requests.is_none()
            && self.remaining_requests.is_none()
            && self.limit_tokens.is_none()
            && self.remaining_tokens.is_none()
            && self.budget_remaining_micro_usd.is_none()
    }
}

/// The admission outcome.
#[derive(Debug, Clone)]
pub enum Admission {
    Allowed(Advisory),
    Denied(Breach),
}

/// Which budget period a soft-spend signal refers to (for the alert detail /
/// header). Daily vs monthly cost budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetPeriod {
    Daily,
    Monthly,
}

impl BudgetPeriod {
    /// Closed-vocab detail code for the off-path alert and the header scope hint.
    pub fn code(self) -> &'static str {
        match self {
            BudgetPeriod::Daily => "daily",
            BudgetPeriod::Monthly => "monthly",
        }
    }
}

/// An **edge-triggered** soft-budget crossing — produced at most once per budget
/// window per scope when consumed spend first reaches the configured soft
/// threshold. Carries only labels/counts (no PII): the breaching scope, the
/// period, the configured threshold (‰) and the observed consumed fraction (‰).
/// Consumed by the proxy to emit an off-path spend-alert via the EXISTING export
/// seam — never blocks, never allocates on the non-crossing path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpendAlert {
    pub scope: LimitScope,
    pub period: BudgetPeriod,
    pub threshold_permille: u16,
    pub consumed_permille: u16,
}

/// The always-cheap **warning-zone** signal for the synchronous response header
/// `x-routeplane-budget-warning`. Present whenever the tightest configured cost
/// budget is at/over its soft threshold; absent below it (so a request outside
/// the zone is byte-identical). Distinct from the edge-triggered [`SpendAlert`]:
/// this is recomputed cheaply from the counter on every in-zone response.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BudgetWarning {
    pub scope: LimitScope,
    pub period: BudgetPeriod,
    pub threshold_permille: u16,
    pub consumed_permille: u16,
}

// ---------------------------------------------------------------------------
// ScopeCounters — the pre-resolved atomic slots for one (scope, policy)
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct ScopeCounters {
    scope: LimitScope,
    policy_id: String,
    /// The structural counter identity (always the fallback `scope:owner`, e.g.
    /// `tenant:<tenant_id>`), independent of any operator-supplied `policy_id`.
    /// Mode D ([ADR-056]) keys its Redis counter off this so a reused `policy_id`
    /// cannot collapse two tenants' global counters. Mode L never reads it (its
    /// isolation is the registry map key); it exists only to project into
    /// [`RequestRateSpec::identity`].
    identity: String,
    // rate (per-minute)
    requests_per_min: Option<u64>,
    /// The UNCLAMPED, fleet-wide `requests_per_min` cap for Mode D. `requests_per_min`
    /// above is share-clamped for the per-replica LOCAL window; Redis counts
    /// globally across replicas, so Mode D must enforce this configured value or it
    /// under-enforces by `max_replicas×` ([ADR-023] §6 vs [ADR-056]).
    req_per_min_global: Option<u64>,
    tokens_per_min: Option<u64>,
    rate_mode: LimitMode,
    c_req: WindowCounter,
    c_tok_rate: WindowCounter,
    // budgets
    cost_daily: Option<u64>,
    cost_monthly: Option<u64>,
    tokens_daily: Option<u64>,
    tokens_monthly: Option<u64>,
    anchor_offset_minutes: i32,
    budget_mode: LimitMode,
    c_cost_daily: WindowCounter,
    c_cost_monthly: WindowCounter,
    c_tok_daily: WindowCounter,
    c_tok_monthly: WindowCounter,
    // Soft spend threshold (‰ of the cost budget). `None` ⇒ no soft alerting
    // (byte-identical legacy path). The per-window edge-trigger flags reset on
    // window roll exactly like the counters they shadow.
    soft_threshold_permille: Option<u16>,
    soft_alerted_cost_daily: AlertFlag,
    soft_alerted_cost_monthly: AlertFlag,
}

/// Per-replica share of a fleet-wide limit (ADR-023 §6): with `max_replicas`
/// stateless replicas each enforcing locally (Mode L), each must allow only
/// `ceil(limit / max_replicas)` so the AGGREGATE stays ~the configured cap
/// instead of `max_replicas ×` it. `max_replicas <= 1` is the identity (no
/// clamp). A configured `0` (deny-all) stays `0`.
fn share_per_replica(v: Option<u64>, max_replicas: u32) -> Option<u64> {
    v.map(|x| {
        if max_replicas <= 1 {
            x
        } else {
            x.div_ceil(max_replicas as u64)
        }
    })
}

impl ScopeCounters {
    /// Build the atomic slots for a policy. Returns `None` when the policy
    /// configures no limit at all (so the slot is simply absent ⇒ unlimited).
    fn from_policy(
        scope: LimitScope,
        policy: &LimitPolicy,
        fallback_id: &str,
        max_replicas: u32,
    ) -> Option<Arc<Self>> {
        if policy.is_empty() {
            return None;
        }
        let policy_id = policy
            .policy_id
            .clone()
            .unwrap_or_else(|| fallback_id.to_string());
        let rate = policy.rate.clone().unwrap_or_default();
        let budget = policy.budget.clone().unwrap_or_default();
        // Security-review hardening: a configured limit ABOVE the 40-bit count
        // mask could never be reached by the saturating counter, silently
        // disabling enforcement (fail-open). Clamp at build so the fail-closed
        // property (saturated counter >= limit keeps denying) holds for every
        // representable configuration.
        // Share-clamp to this replica's fair share (ADR-023 §6), THEN clamp to
        // the 40-bit counter mask (fail-closed representability).
        let clamp = |v: Option<u64>| share_per_replica(v, max_replicas).map(|x| x.min(COUNT_MASK));
        // The GLOBAL (fleet-wide) request cap for Mode D (ADR-056): NOT share-clamped
        // — Redis counts across all replicas, so the per-replica share would 429 the
        // fleet at limit/max_replicas. Still mask-bounded (representability parity).
        let req_per_min_global = rate.requests_per_min.map(|x| x.min(COUNT_MASK));
        // FR-22: resolve display-currency cost caps to micro-USD (off the hot path).
        // The binary validates the same config at startup and refuses to serve on
        // error; should an invalid budget reach this infallible builder, fall CLOSED
        // — deny-all $0 on both windows, never a silently-uncapped spend.
        let (cost_daily_micro, cost_monthly_micro) =
            budget.resolved_cost_caps().unwrap_or((Some(0), Some(0)));
        Some(Arc::new(Self {
            scope,
            policy_id,
            // The fallback id already embeds the owning key/tenant/model, so it
            // doubles as the structural counter identity regardless of whether an
            // explicit policy_id was supplied above.
            identity: fallback_id.to_string(),
            requests_per_min: clamp(rate.requests_per_min),
            req_per_min_global,
            tokens_per_min: clamp(rate.tokens_per_min),
            rate_mode: rate.mode,
            c_req: WindowCounter::new(),
            c_tok_rate: WindowCounter::new(),
            cost_daily: clamp(cost_daily_micro),
            cost_monthly: clamp(cost_monthly_micro),
            tokens_daily: clamp(budget.tokens_daily),
            tokens_monthly: clamp(budget.tokens_monthly),
            anchor_offset_minutes: budget.anchor_offset_minutes,
            budget_mode: budget.mode,
            c_cost_daily: WindowCounter::new(),
            c_cost_monthly: WindowCounter::new(),
            c_tok_daily: WindowCounter::new(),
            c_tok_monthly: WindowCounter::new(),
            // Clamp ‰ to [0, 1000] at build: a value above full budget can never
            // disable the eventual hard 402 (the soft surface is advisory only),
            // and clamping keeps the consumed-vs-threshold comparison well-defined.
            soft_threshold_permille: budget.soft_threshold_permille.map(|p| p.min(1000)),
            soft_alerted_cost_daily: AlertFlag::new(),
            soft_alerted_cost_monthly: AlertFlag::new(),
        }))
    }

    /// True when two counter slots share the SAME policy — every configured limit
    /// field matches (the live window state is deliberately excluded). Used by
    /// [`LimitRegistry::replace_preserving_counters`] to decide whether an atomic
    /// swap can keep an unchanged scope's in-window counts instead of resetting
    /// them. Compares only the post-clamp config the slot was built from.
    fn same_policy(&self, other: &Self) -> bool {
        self.scope == other.scope
            && self.policy_id == other.policy_id
            && self.requests_per_min == other.requests_per_min
            // The GLOBAL cap can change while the share-clamped local value does not
            // (e.g. 99→100 at max_replicas=4 both clamp to 25); include it so a
            // Mode-D cap change is never masked into a preserved (stale) counter.
            && self.req_per_min_global == other.req_per_min_global
            && self.tokens_per_min == other.tokens_per_min
            && self.rate_mode == other.rate_mode
            && self.cost_daily == other.cost_daily
            && self.cost_monthly == other.cost_monthly
            && self.tokens_daily == other.tokens_daily
            && self.tokens_monthly == other.tokens_monthly
            && self.anchor_offset_minutes == other.anchor_offset_minutes
            && self.budget_mode == other.budget_mode
            && self.soft_threshold_permille == other.soft_threshold_permille
    }

    fn rate_breach(&self, kind: LimitKind, limit: u64, now_ms: u64) -> Breach {
        Breach {
            kind,
            scope: self.scope,
            policy_id: self.policy_id.clone(),
            limit,
            reset_ms: Some(minute_reset_ms(now_ms)),
            now_ms,
        }
    }

    fn budget_breach(&self, kind: LimitKind, monthly: bool, limit: u64, now_ms: u64) -> Breach {
        let reset = if monthly {
            month_reset_ms(now_ms, self.anchor_offset_minutes)
        } else {
            day_reset_ms(now_ms, self.anchor_offset_minutes)
        };
        Breach {
            kind,
            scope: self.scope,
            policy_id: self.policy_id.clone(),
            limit,
            reset_ms: Some(reset),
            now_ms,
        }
    }

    /// Phase 1 — read-only check-before: token-rate headroom + every budget.
    /// Returns the first HARD breach (soft limits never block here).
    fn check(&self, now_ms: u64) -> Option<Breach> {
        if self.rate_mode.is_hard() {
            if let Some(limit) = self.tokens_per_min {
                if self.c_tok_rate.peek(minute_epoch(now_ms)) >= limit {
                    return Some(self.rate_breach(LimitKind::RateTokens, limit, now_ms));
                }
            }
        }
        if self.budget_mode.is_hard() {
            let off = self.anchor_offset_minutes;
            if let Some(limit) = self.cost_daily {
                if self.c_cost_daily.peek(day_epoch(now_ms, off)) >= limit {
                    return Some(self.budget_breach(LimitKind::BudgetCost, false, limit, now_ms));
                }
            }
            if let Some(limit) = self.cost_monthly {
                if self.c_cost_monthly.peek(month_epoch(now_ms, off)) >= limit {
                    return Some(self.budget_breach(LimitKind::BudgetCost, true, limit, now_ms));
                }
            }
            if let Some(limit) = self.tokens_daily {
                if self.c_tok_daily.peek(day_epoch(now_ms, off)) >= limit {
                    return Some(self.budget_breach(LimitKind::BudgetTokens, false, limit, now_ms));
                }
            }
            if let Some(limit) = self.tokens_monthly {
                if self.c_tok_monthly.peek(month_epoch(now_ms, off)) >= limit {
                    return Some(self.budget_breach(LimitKind::BudgetTokens, true, limit, now_ms));
                }
            }
        }
        None
    }

    /// Phase 2 — atomically consume one request slot. Hard mode rejects on a full
    /// window; soft/shadow mode counts unconditionally (so the limit can be sized
    /// from real traffic) and never blocks.
    fn consume_request(&self, now_ms: u64) -> Option<Breach> {
        let limit = self.requests_per_min?;
        let ep = minute_epoch(now_ms);
        if self.rate_mode.is_hard() {
            match self.c_req.try_consume(1, limit, ep) {
                Ok(_) => None,
                Err(_) => Some(self.rate_breach(LimitKind::RateRequests, limit, now_ms)),
            }
        } else {
            self.c_req.debit(1, ep);
            None
        }
    }

    /// Settle-after: debit actual usage. Token-rate, token budgets, and cost
    /// budgets are debited unconditionally (counting happens in every mode; only
    /// blocking is gated by mode at admit).
    ///
    /// Returns an edge-triggered [`SpendAlert`] iff this debit is the one that
    /// first carries the consumed *cost* fraction at/over the configured soft
    /// threshold for a budget window (fire-once-per-window via the lock-free
    /// [`AlertFlag`]). `None` when no soft threshold is configured (the common
    /// path: zero extra atomics touched) or the threshold is not (newly) crossed.
    /// The daily budget is checked before the monthly one; at most one alert is
    /// returned per settle (the tighter daily window wins when both newly cross).
    fn settle(&self, now_ms: u64, total_tokens: u64, cost_micro_usd: u64) -> Option<SpendAlert> {
        let ep_min = minute_epoch(now_ms);
        let off = self.anchor_offset_minutes;
        if self.tokens_per_min.is_some() {
            self.c_tok_rate.debit(total_tokens, ep_min);
        }
        let mut alert: Option<SpendAlert> = None;
        if let Some(limit) = self.cost_daily {
            let ep = day_epoch(now_ms, off);
            self.c_cost_daily.debit(cost_micro_usd, ep);
            alert = self.maybe_spend_alert(
                limit,
                self.c_cost_daily.peek(ep),
                BudgetPeriod::Daily,
                &self.soft_alerted_cost_daily,
                ep,
            );
        }
        if self.cost_monthly.is_some() {
            self.c_cost_monthly
                .debit(cost_micro_usd, month_epoch(now_ms, off));
        }
        if alert.is_none() {
            if let Some(limit) = self.cost_monthly {
                let ep = month_epoch(now_ms, off);
                alert = self.maybe_spend_alert(
                    limit,
                    self.c_cost_monthly.peek(ep),
                    BudgetPeriod::Monthly,
                    &self.soft_alerted_cost_monthly,
                    ep,
                );
            }
        }
        if self.tokens_daily.is_some() {
            self.c_tok_daily.debit(total_tokens, day_epoch(now_ms, off));
        }
        if self.tokens_monthly.is_some() {
            self.c_tok_monthly
                .debit(total_tokens, month_epoch(now_ms, off));
        }
        alert
    }

    /// Edge-trigger helper: given a cost budget `limit`, the post-debit
    /// `consumed`, the period, and the per-window alert flag, return a
    /// [`SpendAlert`] iff a soft threshold is configured, the consumed fraction
    /// is at/over it, AND this is the first crossing observed in `epoch`
    /// (one-shot CAS). No soft threshold ⇒ the cheap early `?` short-circuit,
    /// so the non-configured path touches no extra atomic.
    #[inline]
    fn maybe_spend_alert(
        &self,
        limit: u64,
        consumed: u64,
        period: BudgetPeriod,
        flag: &AlertFlag,
        epoch: u64,
    ) -> Option<SpendAlert> {
        let threshold = self.soft_threshold_permille?;
        let permille = consumed_permille(consumed, limit);
        if permille < threshold as u64 {
            return None;
        }
        if !flag.try_fire(epoch) {
            return None;
        }
        Some(SpendAlert {
            scope: self.scope,
            period,
            threshold_permille: threshold,
            consumed_permille: permille as u16,
        })
    }

    /// Read-only warning-zone probe for the synchronous header. Returns the
    /// tightest (highest consumed ‰) cost budget that is at/over the soft
    /// threshold for the current window, or `None` when no soft threshold is
    /// configured or spend is below it. Pure atomic loads — no CAS, no mutation,
    /// no allocation. The daily and monthly cost budgets are both considered.
    fn warning(&self, now_ms: u64) -> Option<BudgetWarning> {
        let threshold = self.soft_threshold_permille?;
        let off = self.anchor_offset_minutes;
        let mut best: Option<BudgetWarning> = None;
        let mut consider = |limit: Option<u64>, consumed: u64, period: BudgetPeriod| {
            if let Some(limit) = limit {
                let permille = consumed_permille(consumed, limit);
                if permille >= threshold as u64 {
                    let w = BudgetWarning {
                        scope: self.scope,
                        period,
                        threshold_permille: threshold,
                        consumed_permille: permille as u16,
                    };
                    if best.is_none_or(|b| w.consumed_permille > b.consumed_permille) {
                        best = Some(w);
                    }
                }
            }
        };
        consider(
            self.cost_daily,
            self.c_cost_daily.peek(day_epoch(now_ms, off)),
            BudgetPeriod::Daily,
        );
        consider(
            self.cost_monthly,
            self.c_cost_monthly.peek(month_epoch(now_ms, off)),
            BudgetPeriod::Monthly,
        );
        best
    }

    /// Fold this scope's headroom into the advisory accumulator (tightest wins).
    fn contribute_advisory(&self, now_ms: u64, a: &mut Advisory) {
        if let Some(limit) = self.requests_per_min {
            let rem = limit.saturating_sub(self.c_req.peek(minute_epoch(now_ms)));
            if a.remaining_requests.is_none_or(|r| rem < r) {
                a.remaining_requests = Some(rem);
                a.limit_requests = Some(limit);
                a.reset_requests_secs = Some(reset_secs(minute_reset_ms(now_ms), now_ms));
            }
        }
        if let Some(limit) = self.tokens_per_min {
            let rem = limit.saturating_sub(self.c_tok_rate.peek(minute_epoch(now_ms)));
            if a.remaining_tokens.is_none_or(|r| rem < r) {
                a.remaining_tokens = Some(rem);
                a.limit_tokens = Some(limit);
            }
        }
        let off = self.anchor_offset_minutes;
        if let Some(limit) = self.cost_daily {
            let rem = limit.saturating_sub(self.c_cost_daily.peek(day_epoch(now_ms, off)));
            if a.budget_remaining_micro_usd.is_none_or(|r| rem < r) {
                a.budget_remaining_micro_usd = Some(rem);
            }
        }
        if let Some(limit) = self.cost_monthly {
            let rem = limit.saturating_sub(self.c_cost_monthly.peek(month_epoch(now_ms, off)));
            if a.budget_remaining_micro_usd.is_none_or(|r| rem < r) {
                a.budget_remaining_micro_usd = Some(rem);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Registry + per-request guards
// ---------------------------------------------------------------------------

/// The pre-built per-model counters for one key: an ordered list of
/// `(model_substring, counters)` rows, most-specific-first (longest pattern first,
/// ties broken lexicographically for determinism) so resolution matches the same
/// way the [`PRICE_BOOK`] does (`gpt-4o-mini` before `gpt-4o` before `gpt-4`).
///
/// Bounded by the number of CONFIGURED patterns on the key — never grows with the
/// set of client-supplied model strings. An arbitrary request model that matches
/// no configured pattern resolves to `None` (no counter, no allocation), so a
/// client cannot inflate memory by spraying distinct model strings.
type ModelCounterList = Vec<(String, Arc<ScopeCounters>)>;

#[derive(Debug, Default)]
struct RegistrySnapshot {
    key_counters: HashMap<String, Arc<ScopeCounters>>,
    tenant_counters: HashMap<String, Arc<ScopeCounters>>,
    /// routeplane_key → ordered per-model counter rows. Absent ⇒ the key has no
    /// per-model limits (the common path; the hot path never touches this map).
    model_counters: HashMap<String, ModelCounterList>,
}

/// Input row for [`LimitRegistry::build`] — owned so it can be produced from the
/// loaded `AuthState` keys without lifetime entanglement.
pub struct KeyLimitsInput {
    pub routeplane_key: String,
    pub tenant_id: String,
    pub limits: KeyLimits,
}

/// The counter registry. Read path is wait-free: an [`ArcSwap`] snapshot rebuilt
/// only on (future) control-plane policy refresh — the same pattern as
/// `AuthState`. The hot path does two `HashMap` reads against an immutable
/// snapshot, no lock.
/// Rewrite `next`'s scope slots to reuse the prior counter-bearing `Arc` wherever
/// the policy is unchanged ([`ScopeCounters::same_policy`]), so a registry swap
/// keeps the in-window counts of every scope that did not change. Helper for
/// [`LimitRegistry::replace_preserving_counters`].
fn preserve_scope_counters(
    next: &mut HashMap<String, Arc<ScopeCounters>>,
    prev: &HashMap<String, Arc<ScopeCounters>>,
) {
    for (key, sc) in next.iter_mut() {
        if let Some(prev_sc) = prev.get(key) {
            if sc.same_policy(prev_sc) {
                *sc = prev_sc.clone();
            }
        }
    }
}

#[derive(Debug)]
pub struct LimitRegistry {
    inner: ArcSwap<RegistrySnapshot>,
}

impl Default for LimitRegistry {
    fn default() -> Self {
        Self::empty()
    }
}

impl LimitRegistry {
    /// An empty registry — every key resolves to unlimited guards.
    pub fn empty() -> Self {
        Self {
            inner: ArcSwap::from_pointee(RegistrySnapshot::default()),
        }
    }

    /// Build the registry from the loaded keys. Per-key counters are keyed by
    /// `routeplane_key`; per-tenant counters by `tenant_id` and **first-wins** if
    /// multiple keys of the same tenant declare a tenant policy (deterministic;
    /// declare the tenant policy once).
    pub fn build(entries: impl IntoIterator<Item = KeyLimitsInput>) -> Self {
        // Default: single replica ⇒ share-clamp is the identity (no behavior
        // change vs pre-ADR-023-§6). Production passes the real replica count.
        Self::build_with_replicas(entries, 1)
    }

    /// Build with the deployment's `max_replicas` so each stateless replica
    /// enforces only `ceil(limit / max_replicas)` of every configured cap
    /// (ADR-023 §6 share-clamp). `max_replicas <= 1` is identical to [`build`].
    pub fn build_with_replicas(
        entries: impl IntoIterator<Item = KeyLimitsInput>,
        max_replicas: u32,
    ) -> Self {
        let mut snap = RegistrySnapshot::default();
        for e in entries {
            if e.limits.is_empty() {
                continue;
            }
            if let Some(p) = &e.limits.key {
                let fallback = format!("key:{}", e.routeplane_key);
                if let Some(sc) =
                    ScopeCounters::from_policy(LimitScope::Key, p, &fallback, max_replicas)
                {
                    snap.key_counters.insert(e.routeplane_key.clone(), sc);
                }
            }
            if let Some(p) = &e.limits.tenant {
                if !snap.tenant_counters.contains_key(&e.tenant_id) {
                    let fallback = format!("tenant:{}", e.tenant_id);
                    if let Some(sc) =
                        ScopeCounters::from_policy(LimitScope::Tenant, p, &fallback, max_replicas)
                    {
                        snap.tenant_counters.insert(e.tenant_id.clone(), sc);
                    }
                }
            }
            // Per-model counters (PRD-008 §9). One pre-built counter slot per
            // CONFIGURED (key, pattern) — memory is bounded by the configured
            // pattern count, NOT by the set of client model strings. Empty patterns
            // and empty policies are skipped (no slot ⇒ unlimited for that model).
            if !e.limits.model_limits_is_empty() {
                let mut rows: ModelCounterList = Vec::new();
                for (pattern, policy) in &e.limits.model_limits {
                    let pat = pattern.trim();
                    if pat.is_empty() || policy.is_empty() {
                        continue;
                    }
                    // The counter key (policy_id fallback) folds BOTH the owning key
                    // identity AND the matched model pattern, so a model cap on key
                    // A is a different counter from the same model on key B, and
                    // from the overall key budget — per-(key, model) isolation.
                    let fallback = format!("model:{}:{}", e.routeplane_key, pat);
                    if let Some(sc) = ScopeCounters::from_policy(
                        LimitScope::Model,
                        policy,
                        &fallback,
                        max_replicas,
                    ) {
                        rows.push((pat.to_string(), sc));
                    }
                }
                if !rows.is_empty() {
                    // Most-specific-first: longest pattern wins, lexicographic tie
                    // break (deterministic) — matches the PRICE_BOOK convention.
                    rows.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then_with(|| a.0.cmp(&b.0)));
                    snap.model_counters.insert(e.routeplane_key.clone(), rows);
                }
            }
        }
        Self {
            inner: ArcSwap::from_pointee(snap),
        }
    }

    /// Atomically swap in a new registry snapshot (future CP push). Readers never
    /// lock; in-flight requests keep their already-resolved [`LimitGuards`].
    ///
    /// DISCLOSED (security review): the new snapshot's counters start at ZERO —
    /// a swap resets all Mode L in-window counts (Mode R rebases from the
    /// durable aggregate; Mode L cannot). For CP config distribution prefer
    /// [`replace_preserving_counters`](Self::replace_preserving_counters), which
    /// keeps the counts of every scope whose policy did not change.
    pub fn replace(&self, entries: impl IntoIterator<Item = KeyLimitsInput>) {
        let built = Self::build(entries);
        self.inner.store(built.inner.load_full());
    }

    /// Like [`replace`](Self::replace), but **preserves the in-window counters of
    /// every scope whose policy is unchanged**. A plain `replace()` zeroes all
    /// Mode-L counters on swap, so a CP config change affecting ONE tenant would
    /// reset EVERY tenant's rate/budget window — editing tenant B's limit would
    /// let tenant A burst (ADR-064 cross-tenant blast radius). This variant builds
    /// the new snapshot, then for each (key/tenant/model) scope reuses the prior
    /// `Arc<ScopeCounters>` — and thus its live counts — when [`ScopeCounters::same_policy`]
    /// holds. Only genuinely changed or newly-added scopes start a fresh window;
    /// removed scopes simply disappear. Same lock-free single-store swap as
    /// `replace` (one `ArcSwap` store); the diff work is off the hot path.
    pub fn replace_preserving_counters(&self, entries: impl IntoIterator<Item = KeyLimitsInput>) {
        let built = Self::build(entries);
        let prev = self.inner.load_full();
        // `built` is freshly constructed and unshared ⇒ we can own its snapshot
        // outright and rewrite the unchanged scopes to point back at the prior
        // (counter-bearing) Arcs. If — impossibly — it were shared, degrade to a
        // plain counter-resetting swap rather than fail.
        let mut next = match Arc::try_unwrap(built.inner.into_inner()) {
            Ok(snap) => snap,
            Err(arc) => {
                self.inner.store(arc);
                return;
            }
        };
        preserve_scope_counters(&mut next.key_counters, &prev.key_counters);
        preserve_scope_counters(&mut next.tenant_counters, &prev.tenant_counters);
        for (key, rows) in next.model_counters.iter_mut() {
            let Some(prev_rows) = prev.model_counters.get(key) else {
                continue;
            };
            for (pat, sc) in rows.iter_mut() {
                if let Some((_, prev_sc)) = prev_rows.iter().find(|(p, _)| p == pat) {
                    if sc.same_policy(prev_sc) {
                        *sc = prev_sc.clone();
                    }
                }
            }
        }
        self.inner.store(Arc::new(next));
    }

    /// Resolve the per-request guards for the authenticated key/tenant. Counters
    /// are keyed off the authenticated context ONLY — never client headers (the
    /// [ADR-023] bypass rule). Cross-tenant isolation is structural: a different
    /// `tenant_id` maps to a different `Arc`, so there is no shared keyed map a
    /// request could mis-index into.
    pub fn resolve(&self, routeplane_key: &str, tenant_id: &str) -> LimitGuards {
        let snap = self.inner.load();
        LimitGuards {
            key: snap.key_counters.get(routeplane_key).cloned(),
            tenant: snap.tenant_counters.get(tenant_id).cloned(),
            model: None,
        }
    }

    /// Resolve the per-request guards INCLUDING the per-model scope for the served
    /// model (PRD-008 §9). The model-scoped counter is the FIRST configured pattern
    /// (most-specific-first) that the `served_model` contains — the same substring
    /// match as the [`PRICE_BOOK`]. A model matching no configured pattern (the
    /// common case, and any arbitrary client model string) resolves `model: None`,
    /// so no counter is created and the path is byte-identical to [`resolve`].
    ///
    /// The matched [`ScopeCounters`] is a pre-built slot keyed per-(this key,
    /// pattern), so the same `LimitGuards` value drives `admit`, `settle`,
    /// `advisory`, and `warning` against the SAME model counter — the served model
    /// is baked into the resolved guard, not re-threaded through each call.
    pub fn resolve_for_model(
        &self,
        routeplane_key: &str,
        tenant_id: &str,
        served_model: &str,
    ) -> LimitGuards {
        let snap = self.inner.load();
        let model = snap.model_counters.get(routeplane_key).and_then(|rows| {
            rows.iter()
                .find(|(pat, _)| served_model.contains(pat.as_str()))
                .map(|(_, sc)| sc.clone())
        });
        LimitGuards {
            key: snap.key_counters.get(routeplane_key).cloned(),
            tenant: snap.tenant_counters.get(tenant_id).cloned(),
            model,
        }
    }
}

/// The pre-resolved counter handles for one request — AND-composed across the
/// key, tenant, and (served-)model scopes ([PRD-008] FR-7 / §9,
/// most-restrictive-wins). Cheap to clone (a few `Arc` bumps) so it rides into the
/// streaming task. Empty ⇒ unlimited ⇒ byte-identical legacy behaviour. The model
/// scope is `Some` only when the served model matched a configured per-model
/// pattern for this key; the SAME guard then drives admit/settle/advisory/warning
/// against that one model counter, so settle debits exactly the counter admit
/// checked.
#[derive(Debug, Clone)]
pub struct LimitGuards {
    key: Option<Arc<ScopeCounters>>,
    tenant: Option<Arc<ScopeCounters>>,
    model: Option<Arc<ScopeCounters>>,
}

impl LimitGuards {
    pub fn is_unlimited(&self) -> bool {
        self.key.is_none() && self.tenant.is_none() && self.model.is_none()
    }

    fn scopes(&self) -> impl Iterator<Item = &Arc<ScopeCounters>> {
        self.key
            .iter()
            .chain(self.tenant.iter())
            .chain(self.model.iter())
    }

    /// Check-before / settle-after admission. Phase 1 checks budgets + token-rate
    /// across every scope read-only; phase 2 consumes one request slot per scope.
    /// The first hard breach wins (fail-stop, before any provider call — FR-20).
    pub fn admit(&self, now_ms: u64) -> Admission {
        for s in self.scopes() {
            if let Some(b) = s.check(now_ms) {
                return Admission::Denied(b);
            }
        }
        for s in self.scopes() {
            if let Some(b) = s.consume_request(now_ms) {
                return Admission::Denied(b);
            }
        }
        Admission::Allowed(self.advisory(now_ms))
    }

    /// Read-only check-before across every scope (budgets + token-rate), WITHOUT
    /// consuming a request slot. The first hard breach wins. Used by the
    /// distributed engine ([ADR-056]) to preserve the local 402/budget path
    /// unchanged while it escalates only the per-minute *request* count to Redis.
    /// On an unlimited guard this is a no-op `Allowed` (byte-identical).
    pub fn check_only(&self, now_ms: u64) -> Admission {
        for s in self.scopes() {
            if let Some(b) = s.check(now_ms) {
                return Admission::Denied(b);
            }
        }
        Admission::Allowed(self.advisory(now_ms))
    }

    /// The per-scope request-rate specs ([`RequestRateSpec`]) for the distributed
    /// engine — the read-only projection of each configured `requests_per_min`
    /// cap (share-clamped, with its hard/soft mode). Empty when no scope
    /// configures a request rate (Mode D then has nothing to enforce in Redis and
    /// defers to the local check). No counter state is exposed.
    pub fn request_rate_specs(&self) -> Vec<RequestRateSpec> {
        self.scopes()
            .filter_map(|s| {
                // Export the GLOBAL cap (not the share-clamped local one) and the
                // structural identity (not the reusable policy_id) — Mode D counts
                // globally in Redis and must isolate per key/tenant/model.
                s.req_per_min_global.map(|limit| RequestRateSpec {
                    scope: s.scope,
                    policy_id: s.policy_id.clone(),
                    identity: s.identity.clone(),
                    limit,
                    hard: s.rate_mode.is_hard(),
                })
            })
            .collect()
    }

    /// Settle actual usage after a completed response (stream end for SSE).
    ///
    /// Returns any **edge-triggered** [`SpendAlert`]s produced by this debit —
    /// at most one per scope (key and/or tenant), and only the first time each
    /// crosses its soft threshold in the current budget window (fire-once). An
    /// empty `Vec` is the overwhelmingly common case (no soft threshold, or not
    /// newly crossed) and allocates nothing. The proxy folds these into the
    /// existing off-path export seam; the hot-path settle itself never blocks.
    pub fn settle(&self, now_ms: u64, total_tokens: u64, cost_micro_usd: u64) -> Vec<SpendAlert> {
        let mut alerts: Vec<SpendAlert> = Vec::new();
        for s in self.scopes() {
            if let Some(a) = s.settle(now_ms, total_tokens, cost_micro_usd) {
                alerts.push(a);
            }
        }
        alerts
    }

    /// The tightest soft-budget **warning** across scopes for the synchronous
    /// `x-routeplane-budget-warning` header, or `None` when no scope is in its
    /// warning zone (so the header is simply absent ⇒ byte-identical below the
    /// threshold). Read-only, lock-free.
    pub fn warning(&self, now_ms: u64) -> Option<BudgetWarning> {
        let mut best: Option<BudgetWarning> = None;
        for s in self.scopes() {
            if let Some(w) = s.warning(now_ms) {
                if best.is_none_or(|b| w.consumed_permille > b.consumed_permille) {
                    best = Some(w);
                }
            }
        }
        best
    }

    /// Advisory headers reflecting post-admission headroom (tightest scope).
    pub fn advisory(&self, now_ms: u64) -> Advisory {
        let mut a = Advisory::default();
        for s in self.scopes() {
            s.contribute_advisory(now_ms, &mut a);
        }
        a
    }
}

// ---------------------------------------------------------------------------
// Reconciliation seam (ADR-023 §3) — trait + in-memory stub; Mode R out of scope
// ---------------------------------------------------------------------------

/// One slot's local delta pushed to the durable aggregate on the T_flush cadence
/// ([ADR-023] §3). Off the hot path.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReconcileDelta {
    pub requests: u64,
    pub tokens: u64,
    pub cost_micro_usd: u64,
}

/// The durable cross-replica base for a slot (Mode R). In Mode L this is always
/// zero — enforcement is local-only.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReconcileBase {
    pub requests: u64,
    pub tokens: u64,
    pub cost_micro_usd: u64,
}

/// The async durable reconciliation seam ([ADR-023] §3). A background Tokio task
/// (off the request path) would `flush_delta` local deltas and `pull_base` the
/// global aggregate through the [ADR-005] `Store`. CP/Redis/Postgres integration
/// is explicitly **out of scope** for this build (Mode L); this trait fixes the
/// shape so Mode R is a wiring change, not a redesign.
pub trait LimitReconciler: Send + Sync {
    fn flush_delta(&self, scope: LimitScope, policy_id: &str, delta: ReconcileDelta);
    fn pull_base(&self, scope: LimitScope, policy_id: &str) -> ReconcileBase;
}

/// Mode L default: no durable backend. Base is always zero; flushes are dropped.
#[derive(Debug, Default)]
pub struct NoopReconciler;

impl LimitReconciler for NoopReconciler {
    fn flush_delta(&self, _scope: LimitScope, _policy_id: &str, _delta: ReconcileDelta) {}
    fn pull_base(&self, _scope: LimitScope, _policy_id: &str) -> ReconcileBase {
        ReconcileBase::default()
    }
}

/// An exercisable in-process stub standing in for the durable aggregate (tests /
/// future Mode R prototyping). Uses a `Mutex` — that is fine because it is
/// **off** the request hot path (the reconciler runs on a background cadence).
#[derive(Debug, Default)]
pub struct InMemoryReconciler {
    agg: Mutex<HashMap<(LimitScope, String), ReconcileBase>>,
}

impl LimitReconciler for InMemoryReconciler {
    fn flush_delta(&self, scope: LimitScope, policy_id: &str, delta: ReconcileDelta) {
        if let Ok(mut g) = self.agg.lock() {
            let e = g.entry((scope, policy_id.to_string())).or_default();
            e.requests = e.requests.saturating_add(delta.requests);
            e.tokens = e.tokens.saturating_add(delta.tokens);
            e.cost_micro_usd = e.cost_micro_usd.saturating_add(delta.cost_micro_usd);
        }
    }
    fn pull_base(&self, scope: LimitScope, policy_id: &str) -> ReconcileBase {
        self.agg
            .lock()
            .ok()
            .and_then(|g| g.get(&(scope, policy_id.to_string())).copied())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kl(json: &str) -> KeyLimits {
        serde_json::from_str(json).expect("KeyLimits deserializes")
    }

    fn entry(key: &str, tenant: &str, json: &str) -> KeyLimitsInput {
        KeyLimitsInput {
            routeplane_key: key.into(),
            tenant_id: tenant.into(),
            limits: kl(json),
        }
    }

    // --- share-clamp (ADR-023 §6) --------------------------------------------

    #[test]
    fn resolved_cost_caps_converts_inr_and_fails_closed() {
        // FR-22: a budget authored in INR resolves to micro-USD at the pinned rate.
        let inr = BudgetLimits {
            authored_currency: Some("INR".into()),
            authored_cost_daily: Some(8300), // ₹8300 @ ₹83/USD = $100 = 100_000_000 µ$
            fx_units_per_usd: Some(83),
            ..Default::default()
        };
        assert_eq!(inr.resolved_cost_caps().unwrap(), (Some(100_000_000), None));
        // Micro-USD authoring passes through untouched.
        let usd = BudgetLimits {
            cost_micro_usd_monthly: Some(5_000_000),
            ..Default::default()
        };
        assert_eq!(usd.resolved_cost_caps().unwrap(), (None, Some(5_000_000)));
        // Fail-closed: both forms for one window, authored without a rate, zero rate.
        assert_eq!(
            BudgetLimits {
                cost_micro_usd_daily: Some(1),
                authored_cost_daily: Some(1),
                fx_units_per_usd: Some(83),
                ..Default::default()
            }
            .resolved_cost_caps(),
            Err(BudgetConfigError::CostCapAuthoredTwice)
        );
        assert_eq!(
            BudgetLimits {
                authored_cost_daily: Some(100),
                ..Default::default()
            }
            .resolved_cost_caps(),
            Err(BudgetConfigError::AuthoredCostWithoutFxRate)
        );
        assert_eq!(
            BudgetLimits {
                authored_cost_daily: Some(100),
                fx_units_per_usd: Some(0),
                ..Default::default()
            }
            .resolved_cost_caps(),
            Err(BudgetConfigError::ZeroFxRate)
        );
    }

    #[test]
    fn share_per_replica_math() {
        assert_eq!(share_per_replica(Some(10), 1), Some(10)); // identity at 1
        assert_eq!(share_per_replica(Some(10), 4), Some(3)); // ceil(10/4)
        assert_eq!(share_per_replica(Some(12), 4), Some(3)); // exact
        assert_eq!(share_per_replica(Some(1), 2), Some(1)); // never below 1 for >0
        assert_eq!(share_per_replica(Some(0), 4), Some(0)); // deny-all preserved
        assert_eq!(share_per_replica(None, 4), None); // unlimited stays unlimited
    }

    #[test]
    fn build_with_replicas_enforces_the_per_replica_share() {
        // 12 req/min across 4 replicas → each replica admits only ceil(12/4)=3.
        let reg = LimitRegistry::build_with_replicas(
            vec![entry(
                "rp_s",
                "t_s",
                r#"{"key":{"rate":{"requests_per_min":12}}}"#,
            )],
            4,
        );
        let now = 0;
        let g = reg.resolve("rp_s", "t_s");
        assert!(matches!(g.admit(now), Admission::Allowed(_)));
        assert!(matches!(g.admit(now), Admission::Allowed(_)));
        assert!(matches!(g.admit(now), Admission::Allowed(_)));
        assert!(matches!(g.admit(now), Admission::Denied(_))); // 4th: clamped to 3
    }

    #[test]
    fn build_default_is_unclamped() {
        // build() == single replica → the full configured limit is enforced.
        let reg = LimitRegistry::build(vec![entry(
            "rp_u",
            "t_u",
            r#"{"key":{"rate":{"requests_per_min":2}}}"#,
        )]);
        let now = 0;
        let g = reg.resolve("rp_u", "t_u");
        assert!(matches!(g.admit(now), Admission::Allowed(_)));
        assert!(matches!(g.admit(now), Admission::Allowed(_)));
        assert!(matches!(g.admit(now), Admission::Denied(_)));
    }

    // --- Mode-D projection (request_rate_specs): structural identity + global cap.
    // request_rate_specs() is NOT feature-gated, so these run in the default gate
    // (the distributed.rs counter_key tests cover the Redis-key layer under the
    // `redis-limits` feature). Regression guards for the two ADR-023/056 bugs. ----

    #[test]
    fn request_rate_specs_key_off_structural_identity_not_reused_policy_id() {
        // Two tenants share an explicit reusable policy_id "standard". Mode L keeps
        // them structurally isolated (distinct Arc slots); the exported spec must
        // carry DISTINCT identities so Mode D's Redis counter does not collapse them
        // (ADR-023 structural isolation / ADR-064 cross-tenant blast radius).
        let reg = LimitRegistry::build(vec![
            entry(
                "rp_a",
                "t_a",
                r#"{"tenant":{"policy_id":"standard","rate":{"requests_per_min":100}}}"#,
            ),
            entry(
                "rp_b",
                "t_b",
                r#"{"tenant":{"policy_id":"standard","rate":{"requests_per_min":100}}}"#,
            ),
        ]);
        let sa = reg.resolve("rp_a", "t_a").request_rate_specs();
        let sb = reg.resolve("rp_b", "t_b").request_rate_specs();
        assert_eq!(sa.len(), 1);
        assert_eq!(sb.len(), 1);
        // Same operator-facing policy id (surfaced in the 429 header)...
        assert_eq!(sa[0].policy_id, "standard");
        assert_eq!(sb[0].policy_id, "standard");
        // ...but DISTINCT structural counter identities.
        assert_ne!(sa[0].identity, sb[0].identity);
        assert_eq!(sa[0].identity, "tenant:t_a");
        assert_eq!(sb[0].identity, "tenant:t_b");
    }

    #[test]
    fn request_rate_specs_export_global_cap_while_local_stays_share_clamped() {
        // max_replicas=4 share-clamps the LOCAL counter to ceil(100/4)=25, but the
        // Mode-D Redis counter is GLOBAL across replicas — the spec must export the
        // configured 100, not 25 (else the fleet 429s at 25/min, 4× too tight).
        let reg = LimitRegistry::build_with_replicas(
            vec![entry(
                "rp_g",
                "t_g",
                r#"{"key":{"rate":{"requests_per_min":100}}}"#,
            )],
            4,
        );
        let g = reg.resolve("rp_g", "t_g");
        let specs = g.request_rate_specs();
        assert_eq!(specs.len(), 1);
        assert_eq!(
            specs[0].limit, 100,
            "Mode D exports the GLOBAL configured cap"
        );
        assert_eq!(specs[0].identity, "key:rp_g");
        // The LOCAL Mode-L window is still share-clamped to 25: 25 admits, then deny.
        let now = 60_000;
        for _ in 0..25 {
            assert!(matches!(g.admit(now), Admission::Allowed(_)));
        }
        assert!(
            matches!(g.admit(now), Admission::Denied(_)),
            "26th LOCAL request in the window is denied at the per-replica share"
        );
    }

    // --- packing -------------------------------------------------------------

    #[test]
    fn pack_unpack_round_trips_and_truncates() {
        let (e, c) = unpack(pack(123, 456));
        assert_eq!((e, c), (123, 456));
        // count saturates within COUNT_MASK; epoch truncates to 24 bits.
        let (e2, c2) = unpack(pack(EPOCH_MASK + 5, COUNT_MASK));
        assert_eq!(e2, (EPOCH_MASK + 5) & EPOCH_MASK);
        assert_eq!(c2, COUNT_MASK);
    }

    // --- window counter: burst / boundary / saturation -----------------------

    #[test]
    fn try_consume_admits_up_to_limit_then_rejects_burst() {
        let w = WindowCounter::new();
        let ep = 100;
        assert_eq!(w.try_consume(1, 3, ep), Ok(1));
        assert_eq!(w.try_consume(1, 3, ep), Ok(2));
        assert_eq!(w.try_consume(1, 3, ep), Ok(3));
        // 4th in the same window is rejected (Err carries the current base).
        assert_eq!(w.try_consume(1, 3, ep), Err(3));
    }

    #[test]
    fn window_rolls_over_at_epoch_boundary() {
        let w = WindowCounter::new();
        // Fill window 100.
        for _ in 0..3 {
            let _ = w.try_consume(1, 3, 100);
        }
        assert_eq!(w.try_consume(1, 3, 100), Err(3));
        // The next window (epoch 101) is a fresh count — boundary reset.
        assert_eq!(w.peek(101), 0);
        assert_eq!(w.try_consume(1, 3, 101), Ok(1));
        // The stale window is no longer visible.
        assert_eq!(w.peek(100), 0);
    }

    #[test]
    fn debit_saturates_and_never_panics() {
        let w = WindowCounter::new();
        w.debit(u64::MAX, 7);
        assert_eq!(w.peek(7), COUNT_MASK);
        // A further debit stays clamped.
        w.debit(1000, 7);
        assert_eq!(w.peek(7), COUNT_MASK);
    }

    #[test]
    fn try_consume_amount_overshoots_by_at_most_one_request() {
        // amount>1 admits while base<limit, so a single request overshoots by its
        // own size (FR-16 bound), but the next is rejected.
        let w = WindowCounter::new();
        assert_eq!(w.try_consume(10, 5, 0), Ok(10)); // base 0 < 5 → admit, overshoot
        assert_eq!(w.try_consume(10, 5, 0), Err(10)); // now over the limit
    }

    // --- period math ---------------------------------------------------------

    #[test]
    fn minute_epoch_and_reset() {
        // 90_000 ms = minute 1; reset at 120_000.
        assert_eq!(minute_epoch(90_000), 1);
        assert_eq!(minute_reset_ms(90_000), 120_000);
        assert_eq!(reset_secs(120_000, 90_000), 30);
    }

    #[test]
    fn ist_anchor_shifts_the_day_boundary() {
        // 2024-01-01T19:00:00Z is 2024-01-02T00:30 IST → already the next IST day.
        let utc_1900 = 1_704_135_600_000u64; // 2024-01-01T19:00:00Z
        let utc_1700 = utc_1900 - 2 * 3_600_000; // 17:00Z = 22:30 IST, same IST day
        let ist = 330;
        assert_ne!(day_epoch(utc_1900, ist), day_epoch(utc_1700, ist));
        // In UTC both are the same calendar day.
        assert_eq!(day_epoch(utc_1900, 0), day_epoch(utc_1700, 0));
    }

    #[test]
    fn month_epoch_and_reset_are_calendar_anchored() {
        // 2024-02-15T00:00:00Z.
        let feb = 1_707_955_200_000u64;
        let e = month_epoch(feb, 0);
        // March 1 of the same year is the next month.
        let mar = 1_709_251_200_000u64; // 2024-03-01T00:00:00Z
        assert_eq!(month_epoch(mar, 0), e + 1);
        // Reset of February points at March 1.
        assert_eq!(month_reset_ms(feb, 0), mar);
    }

    // --- registry resolution + isolation -------------------------------------

    #[test]
    fn resolve_returns_unlimited_for_unknown_key() {
        let reg = LimitRegistry::empty();
        assert!(reg.resolve("rp_nope", "t_nope").is_unlimited());
    }

    #[test]
    fn per_key_and_per_tenant_counters_do_not_bleed_across_tenants() {
        // Two tenants, each with a 1 req/min hard key limit. Exhausting tenant A
        // must not affect tenant B (structural isolation: distinct Arcs).
        let reg = LimitRegistry::build(vec![
            entry("rp_a", "t_a", r#"{"key":{"rate":{"requests_per_min":1}}}"#),
            entry("rp_b", "t_b", r#"{"key":{"rate":{"requests_per_min":1}}}"#),
        ]);
        let now = 60_000;
        let ga = reg.resolve("rp_a", "t_a");
        assert!(matches!(ga.admit(now), Admission::Allowed(_)));
        assert!(matches!(ga.admit(now), Admission::Denied(_))); // A exhausted

        // B is untouched.
        let gb = reg.resolve("rp_b", "t_b");
        assert!(matches!(gb.admit(now), Admission::Allowed(_)));
    }

    #[test]
    fn replace_preserving_counters_keeps_unchanged_scopes_and_resets_changed() {
        // ADR-064: a CP config edit to ONE tenant must not reset another tenant's
        // window. Tenant A (unchanged) keeps its counts; tenant B (edited) resets.
        let reg = LimitRegistry::build(vec![
            entry(
                "rp_a",
                "t_a",
                r#"{"tenant":{"rate":{"requests_per_min":2}}}"#,
            ),
            entry(
                "rp_b",
                "t_b",
                r#"{"tenant":{"rate":{"requests_per_min":2}}}"#,
            ),
        ]);
        let now = 60_000;

        // Both tenants burn their 2-request windows up front, so the swap below is
        // tested in BOTH directions: A (unchanged) must stay exhausted, B (changed)
        // must reset and admit again.
        let ga = reg.resolve("rp_a", "t_a");
        assert!(matches!(ga.admit(now), Admission::Allowed(_)));
        assert!(matches!(ga.admit(now), Admission::Allowed(_)));
        assert!(matches!(ga.admit(now), Admission::Denied(_)), "A exhausted");
        let gb = reg.resolve("rp_b", "t_b");
        assert!(matches!(gb.admit(now), Admission::Allowed(_)));
        assert!(matches!(gb.admit(now), Admission::Allowed(_)));
        assert!(matches!(gb.admit(now), Admission::Denied(_)), "B exhausted");

        // Operator edits ONLY tenant B (2 → 9 req/min); A's policy is byte-identical.
        reg.replace_preserving_counters(vec![
            entry(
                "rp_a",
                "t_a",
                r#"{"tenant":{"rate":{"requests_per_min":2}}}"#,
            ),
            entry(
                "rp_b",
                "t_b",
                r#"{"tenant":{"rate":{"requests_per_min":9}}}"#,
            ),
        ]);

        // A's window is PRESERVED across the swap — still exhausted, not reset.
        let ga2 = reg.resolve("rp_a", "t_a");
        assert!(
            matches!(ga2.admit(now), Admission::Denied(_)),
            "unchanged tenant A keeps its exhausted window — no cross-tenant reset"
        );

        // B's policy CHANGED ⇒ its (previously exhausted) window is RESET, and the
        // new cap of 9 now enforces. If `same_policy` wrongly preserved B, this very
        // first admit would still be Denied — so this exercises the reset path too.
        let gb2 = reg.resolve("rp_b", "t_b");
        for i in 0..9 {
            assert!(
                matches!(gb2.admit(now), Admission::Allowed(_)),
                "B admit {i} after reset to new cap 9"
            );
        }
        assert!(
            matches!(gb2.admit(now), Admission::Denied(_)),
            "B at new cap 9"
        );
    }

    #[test]
    fn plain_replace_resets_counters_unlike_preserving_variant() {
        // Contrast/guard: the disclosed counter-RESETTING replace() zeroes A's
        // window even though A's policy is unchanged — the exact cross-tenant blast
        // radius that replace_preserving_counters exists to avoid.
        let reg = LimitRegistry::build(vec![entry(
            "rp_a",
            "t_a",
            r#"{"tenant":{"rate":{"requests_per_min":2}}}"#,
        )]);
        let now = 60_000;
        let ga = reg.resolve("rp_a", "t_a");
        assert!(matches!(ga.admit(now), Admission::Allowed(_)));
        assert!(matches!(ga.admit(now), Admission::Allowed(_)));
        assert!(matches!(ga.admit(now), Admission::Denied(_)));

        reg.replace(vec![entry(
            "rp_a",
            "t_a",
            r#"{"tenant":{"rate":{"requests_per_min":2}}}"#,
        )]);
        let ga2 = reg.resolve("rp_a", "t_a");
        assert!(
            matches!(ga2.admit(now), Admission::Allowed(_)),
            "plain replace() resets the window (counters start at zero)"
        );
    }

    #[test]
    fn tenant_scope_is_shared_across_two_keys_of_the_same_tenant() {
        // The tenant policy (2 req/min) is declared once; both keys share it.
        let reg = LimitRegistry::build(vec![
            entry(
                "rp_k1",
                "t_shared",
                r#"{"tenant":{"policy_id":"ten","rate":{"requests_per_min":2}}}"#,
            ),
            entry("rp_k2", "t_shared", r#"{}"#),
        ]);
        let now = 60_000;
        // Two requests across the two keys exhaust the shared tenant window.
        assert!(matches!(
            reg.resolve("rp_k1", "t_shared").admit(now),
            Admission::Allowed(_)
        ));
        assert!(matches!(
            reg.resolve("rp_k2", "t_shared").admit(now),
            Admission::Allowed(_)
        ));
        match reg.resolve("rp_k1", "t_shared").admit(now) {
            Admission::Denied(b) => {
                assert_eq!(b.scope_header(), "tenant");
                assert_eq!(b.policy_id(), "ten");
                assert_eq!(b.kind(), LimitKind::RateRequests);
            }
            _ => panic!("expected tenant-scope denial"),
        }
    }

    // --- verdicts: 429 vs 402, retry-after, advisory -------------------------

    #[test]
    fn rate_breach_is_429_shaped_with_retry_after() {
        let reg = LimitRegistry::build(vec![entry(
            "rp_r",
            "t_r",
            r#"{"key":{"policy_id":"pk","rate":{"requests_per_min":1}}}"#,
        )]);
        let now = 90_000; // minute 1, 30s to reset
        let g = reg.resolve("rp_r", "t_r");
        assert!(matches!(g.admit(now), Admission::Allowed(_)));
        match g.admit(now) {
            Admission::Denied(b) => {
                assert!(!b.is_budget());
                assert_eq!(b.kind_header(), "rate_limit_requests");
                assert_eq!(b.scope_header(), "key");
                assert_eq!(b.policy_id(), "pk");
                assert_eq!(b.limit(), 1);
                assert_eq!(b.retry_after_secs(), Some(30));
            }
            _ => panic!("expected rate denial"),
        }
    }

    #[test]
    fn budget_cost_breach_is_402_after_settle_across_two_calls() {
        let reg = LimitRegistry::build(vec![entry(
            "rp_b",
            "t_b",
            r#"{"key":{"budget":{"cost_micro_usd_daily":1}}}"#,
        )]);
        let now = 86_400_000; // a day boundary
        let g = reg.resolve("rp_b", "t_b");
        // Call 1: headroom (0 < 1) → allowed; settle a real cost.
        assert!(matches!(g.admit(now), Admission::Allowed(_)));
        g.settle(now, 12, estimate_cost_micro_usd("gpt-4o", 7, 5));
        // Call 2: the debited spend now exceeds the $0.000001 daily cap → 402.
        match g.admit(now) {
            Admission::Denied(b) => {
                assert!(b.is_budget());
                assert_eq!(b.kind_header(), "budget_cost");
                assert!(b.retry_after_secs().is_some());
            }
            _ => panic!("expected budget denial"),
        }
    }

    #[test]
    fn soft_mode_counts_but_never_blocks() {
        let reg = LimitRegistry::build(vec![entry(
            "rp_s",
            "t_s",
            r#"{"key":{"rate":{"requests_per_min":1,"mode":"shadow"}}}"#,
        )]);
        let now = 60_000;
        let g = reg.resolve("rp_s", "t_s");
        for _ in 0..5 {
            assert!(matches!(g.admit(now), Admission::Allowed(_)));
        }
    }

    #[test]
    fn advisory_reports_remaining_and_decrements_after_admission() {
        let reg = LimitRegistry::build(vec![entry(
            "rp_h",
            "t_h",
            r#"{"key":{"rate":{"requests_per_min":5,"tokens_per_min":1000},"budget":{"cost_micro_usd_daily":1000000}}}"#,
        )]);
        let now = 60_000;
        let g = reg.resolve("rp_h", "t_h");
        let adv = match g.admit(now) {
            Admission::Allowed(a) => a,
            _ => panic!("allowed"),
        };
        assert_eq!(adv.limit_requests, Some(5));
        assert_eq!(adv.remaining_requests, Some(4)); // one consumed
        assert_eq!(adv.limit_tokens, Some(1000));
        assert_eq!(adv.remaining_tokens, Some(1000)); // not settled yet
        assert_eq!(adv.budget_remaining_micro_usd, Some(1_000_000));
        assert!(!adv.is_empty());
    }

    #[test]
    fn unlimited_guards_produce_empty_advisory() {
        let g = LimitRegistry::empty().resolve("rp_x", "t_x");
        match g.admit(60_000) {
            Admission::Allowed(a) => assert!(a.is_empty()),
            _ => panic!("unlimited always allows"),
        }
    }

    #[test]
    fn legacy_keylimits_absent_is_unlimited() {
        // serde-default: a key with no `limits` field deserializes to None
        // upstream; an explicitly empty object is also unlimited.
        let empty = kl("{}");
        assert!(empty.is_empty());
        let reg = LimitRegistry::build(vec![KeyLimitsInput {
            routeplane_key: "rp_legacy".into(),
            tenant_id: "t_legacy".into(),
            limits: empty,
        }]);
        assert!(reg.resolve("rp_legacy", "t_legacy").is_unlimited());
    }

    // --- pricing book (ADR-051) ----------------------------------------------

    #[test]
    fn pinned_models_keep_legacy_golden_rates() {
        // The two strings the A/B golden corpus exercises must price exactly as
        // the old 4-bucket placeholder did, or the snapshot would drift.
        // gpt-4o: 3 µ$/prompt-tok, 10 µ$/completion-tok.
        assert_eq!(estimate_cost_micro_usd("gpt-4o", 1000, 1000), 13_000);
        // claude-3: 3 / 15 — the golden record case (5 prompt, 3 completion) = 60.
        assert_eq!(estimate_cost_micro_usd("claude-3", 5, 3), 60);
    }

    #[test]
    fn longest_prefix_wins_most_specific_first() {
        // gpt-4o-mini must resolve to its own ($0.15/$0.60) row, not gpt-4o's.
        assert_eq!(
            estimate_cost_micro_usd("gpt-4o-mini", 1_000_000, 0),
            150_000
        );
        assert_eq!(estimate_cost_micro_usd("gpt-4o", 1_000_000, 0), 3_000_000);
        // claude-3-opus before generic claude.
        assert_eq!(
            estimate_cost_micro_usd("claude-3-opus", 0, 1_000_000),
            75_000_000
        );
    }

    #[test]
    fn sub_dollar_per_million_skus_are_representable() {
        // gemini-1.5-flash at $0.075/M prompt — impossible under the old
        // whole-micro-USD-per-token floor; now exact.
        assert_eq!(
            estimate_cost_micro_usd("gemini-1.5-flash", 1_000_000, 0),
            75_000
        );
    }

    #[test]
    fn unknown_model_falls_back_to_default_rate() {
        assert_eq!(
            estimate_cost_micro_usd("some-future-model", 1_000_000, 1_000_000),
            8_000_000 // 2M prompt + 6M completion default
        );
    }

    #[test]
    fn pricing_saturates_and_never_panics_on_max_tokens() {
        // u32::MAX tokens against the priciest row must not overflow/panic.
        let _ = estimate_cost_micro_usd("o1", u32::MAX, u32::MAX);
    }

    // --- soft budget thresholds + spend alerts -------------------------------

    #[test]
    fn consumed_permille_is_integer_exact_at_the_boundary() {
        assert_eq!(consumed_permille(0, 1000), 0);
        assert_eq!(consumed_permille(800, 1000), 800); // exact 80%
        assert_eq!(consumed_permille(799, 1000), 799);
        assert_eq!(consumed_permille(1000, 1000), 1000);
        assert_eq!(consumed_permille(2000, 1000), 1000); // capped at 1000
                                                         // zero budget reads as fully consumed (never spuriously below threshold).
        assert_eq!(consumed_permille(0, 0), 1000);
        // large values saturate, never panic/overflow.
        assert_eq!(consumed_permille(u64::MAX, 1), 1000);
    }

    #[test]
    fn alert_flag_fires_once_per_epoch_then_resets_on_roll() {
        let f = AlertFlag::new();
        assert!(f.try_fire(10)); // first crossing this window
        assert!(!f.try_fire(10)); // already fired — no storm
        assert!(!f.try_fire(10));
        // The next window (fresh epoch) sees the flag implicitly reset.
        assert!(f.try_fire(11));
        assert!(!f.try_fire(11));
        // A stale epoch is also a fresh window.
        assert!(f.try_fire(99));
    }

    #[test]
    fn soft_threshold_parses_from_budget_config() {
        let configured =
            kl(r#"{"key":{"budget":{"cost_micro_usd_daily":1000,"soft_threshold_permille":800}}}"#);
        let b = configured.key.unwrap().budget.unwrap();
        assert_eq!(b.soft_threshold_permille, Some(800));
        // Absent ⇒ None (legacy, no soft alerting).
        let legacy = kl(r#"{"key":{"budget":{"cost_micro_usd_daily":1000}}}"#);
        assert_eq!(
            legacy.key.unwrap().budget.unwrap().soft_threshold_permille,
            None
        );
    }

    #[test]
    fn settle_returns_no_alert_when_no_soft_threshold_configured() {
        // Byte-identical legacy path: a budget with no soft threshold never
        // produces a SpendAlert, no matter how much is consumed.
        let reg = LimitRegistry::build(vec![entry(
            "rp_n",
            "t_n",
            r#"{"key":{"budget":{"cost_micro_usd_daily":1000}}}"#,
        )]);
        let now = 86_400_000;
        let g = reg.resolve("rp_n", "t_n");
        assert!(matches!(g.admit(now), Admission::Allowed(_)));
        assert!(g.settle(now, 0, 950).is_empty()); // 95% — still no alert
        assert!(g.warning(now).is_none()); // and no warning surface
    }

    #[test]
    fn settle_fires_spend_alert_once_per_window_on_crossing() {
        // 80% soft threshold on a 1000 µ$ daily cost budget.
        let reg = LimitRegistry::build(vec![entry(
            "rp_s",
            "t_s",
            r#"{"key":{"budget":{"cost_micro_usd_daily":1000,"soft_threshold_permille":800}}}"#,
        )]);
        let now = 86_400_000; // a day boundary
        let g = reg.resolve("rp_s", "t_s");

        // Below threshold: no alert.
        let a1 = g.settle(now, 0, 700); // 70%
        assert!(a1.is_empty());
        // Crossing 80%: exactly one alert.
        let a2 = g.settle(now, 0, 150); // now 850 = 85%
        assert_eq!(a2.len(), 1);
        let alert = a2[0];
        assert_eq!(alert.scope, LimitScope::Key);
        assert_eq!(alert.period, BudgetPeriod::Daily);
        assert_eq!(alert.threshold_permille, 800);
        assert_eq!(alert.consumed_permille, 850);
        // Subsequent settles in the SAME window: no further alert (no storm).
        assert!(g.settle(now, 0, 50).is_empty()); // 90%, still no new alert
        assert!(g.settle(now, 0, 0).is_empty());
    }

    #[test]
    fn spend_alert_resets_and_refires_when_the_budget_window_rolls() {
        let reg = LimitRegistry::build(vec![entry(
            "rp_r",
            "t_r",
            r#"{"key":{"budget":{"cost_micro_usd_daily":1000,"soft_threshold_permille":800}}}"#,
        )]);
        let day0 = 0u64;
        let g = reg.resolve("rp_r", "t_r");
        assert_eq!(g.settle(day0, 0, 900).len(), 1); // cross in day 0
        assert!(g.settle(day0, 0, 0).is_empty()); // no re-fire same window
                                                  // Next calendar day: counter AND alert flag reset → can fire again.
        let day1 = day0 + DAY_MS as u64;
        assert!(g.warning(day1).is_none()); // fresh window, below threshold
        assert_eq!(g.settle(day1, 0, 850).len(), 1); // crosses again
    }

    #[test]
    fn warning_header_present_in_zone_absent_below() {
        let reg = LimitRegistry::build(vec![entry(
            "rp_w",
            "t_w",
            r#"{"key":{"budget":{"cost_micro_usd_daily":1000,"soft_threshold_permille":800}}}"#,
        )]);
        let now = 86_400_000;
        let g = reg.resolve("rp_w", "t_w");
        let _ = g.settle(now, 0, 500); // 50%
        assert!(g.warning(now).is_none()); // below the zone
        let _ = g.settle(now, 0, 350); // 85%
        let w = g.warning(now).expect("in the warning zone");
        assert_eq!(w.consumed_permille, 850);
        assert_eq!(w.threshold_permille, 800);
        assert_eq!(w.period, BudgetPeriod::Daily);
    }

    #[test]
    fn soft_alert_does_not_change_the_hard_402_cap() {
        // Soft threshold at 50% on a 100 µ$ daily cap; the hard cap still denies
        // at 100% exactly as before.
        let reg = LimitRegistry::build(vec![entry(
            "rp_h",
            "t_h",
            r#"{"key":{"budget":{"cost_micro_usd_daily":100,"soft_threshold_permille":500}}}"#,
        )]);
        let now = 86_400_000;
        let g = reg.resolve("rp_h", "t_h");
        assert!(matches!(g.admit(now), Admission::Allowed(_)));
        let alerts = g.settle(now, 0, 100); // hit the cap exactly; also >= 50% soft
        assert_eq!(alerts.len(), 1); // soft alert fired
                                     // Hard cap unchanged: next admit is a 402 budget breach.
        match g.admit(now) {
            Admission::Denied(b) => {
                assert!(b.is_budget());
                assert_eq!(b.kind_header(), "budget_cost");
            }
            _ => panic!("expected hard budget denial"),
        }
    }

    #[test]
    fn zero_permille_threshold_warns_on_any_spend() {
        let reg = LimitRegistry::build(vec![entry(
            "rp_z",
            "t_z",
            r#"{"key":{"budget":{"cost_micro_usd_daily":1000,"soft_threshold_permille":0}}}"#,
        )]);
        let now = 86_400_000;
        let g = reg.resolve("rp_z", "t_z");
        // Even a tiny spend is >= 0‰ → fires once.
        assert_eq!(g.settle(now, 0, 1).len(), 1);
        assert!(g.warning(now).is_some());
    }

    #[test]
    fn over_1000_permille_threshold_is_clamped_at_build() {
        // A misconfigured 1500‰ clamps to 1000‰ — it can only fire at full
        // consumption, and never disables the hard cap.
        let reg = LimitRegistry::build(vec![entry(
            "rp_c",
            "t_c",
            r#"{"key":{"budget":{"cost_micro_usd_daily":1000,"soft_threshold_permille":1500}}}"#,
        )]);
        let now = 86_400_000;
        let g = reg.resolve("rp_c", "t_c");
        assert!(g.settle(now, 0, 999).is_empty()); // 99.9% < 100% clamp
        assert_eq!(g.settle(now, 0, 1).len(), 1); // 100% → fires
    }

    // --- reconciliation stub -------------------------------------------------

    #[test]
    fn in_memory_reconciler_accumulates_deltas() {
        let r = InMemoryReconciler::default();
        r.flush_delta(
            LimitScope::Tenant,
            "tp",
            ReconcileDelta {
                requests: 3,
                tokens: 100,
                cost_micro_usd: 50,
            },
        );
        r.flush_delta(
            LimitScope::Tenant,
            "tp",
            ReconcileDelta {
                requests: 2,
                tokens: 10,
                cost_micro_usd: 5,
            },
        );
        let base = r.pull_base(LimitScope::Tenant, "tp");
        assert_eq!(
            base,
            ReconcileBase {
                requests: 5,
                tokens: 110,
                cost_micro_usd: 55,
            }
        );
        // Noop reconciler is always zero (Mode L).
        assert_eq!(
            NoopReconciler.pull_base(LimitScope::Key, "anything"),
            ReconcileBase::default()
        );
    }

    // --- per-model limits (PRD-008 §9) ---------------------------------------

    #[test]
    fn model_limits_config_parses() {
        let l = kl(r#"{
                "model_limits": {
                    "gpt-4": {"policy_id":"gpt4-cap","rate":{"requests_per_min":5}},
                    "gpt-3.5": {"budget":{"cost_micro_usd_daily":1000}}
                }
            }"#);
        assert_eq!(l.model_limits.len(), 2);
        assert!(!l.model_limits_is_empty());
        assert!(!l.is_empty());
    }

    #[test]
    fn empty_model_limits_map_is_treated_as_unconfigured() {
        // An empty map (and a map of empty policies) ⇒ no per-model limits ⇒ the
        // whole key is empty ⇒ no counters ⇒ byte-identical legacy path.
        assert!(kl(r#"{"model_limits":{}}"#).is_empty());
        assert!(kl(r#"{"model_limits":{"gpt-4":{}}}"#).is_empty());
    }

    #[test]
    fn model_scope_header_is_model() {
        assert_eq!(LimitScope::Model.header(), "model");
    }

    #[test]
    fn per_key_model_request_cap_trips_with_scope_model_and_isolates_other_models() {
        // gpt-4: 2 req/min; no cap on gpt-3.5.
        let reg = LimitRegistry::build(vec![entry(
            "rp_m",
            "t_m",
            r#"{"model_limits":{"gpt-4":{"policy_id":"gpt4-cap","rate":{"requests_per_min":2}}}}"#,
        )]);
        let now = 60_000;
        // Two gpt-4 requests admit; the third trips with scope=model.
        assert!(matches!(
            reg.resolve_for_model("rp_m", "t_m", "gpt-4o").admit(now),
            Admission::Allowed(_)
        ));
        assert!(matches!(
            reg.resolve_for_model("rp_m", "t_m", "gpt-4o").admit(now),
            Admission::Allowed(_)
        ));
        match reg.resolve_for_model("rp_m", "t_m", "gpt-4o").admit(now) {
            Admission::Denied(b) => {
                assert_eq!(b.scope_header(), "model");
                assert_eq!(b.policy_id(), "gpt4-cap");
                assert_eq!(b.kind(), LimitKind::RateRequests);
            }
            _ => panic!("expected model-scope denial"),
        }
        // A DIFFERENT model under the SAME key is unaffected (separate counter).
        assert!(matches!(
            reg.resolve_for_model("rp_m", "t_m", "gpt-3.5-turbo")
                .admit(now),
            Admission::Allowed(_)
        ));
    }

    #[test]
    fn per_key_model_is_isolated_across_keys() {
        // Same model pattern on two keys ⇒ distinct per-(key,model) counters.
        let reg = LimitRegistry::build(vec![
            entry(
                "rp_x",
                "t_x",
                r#"{"model_limits":{"gpt-4":{"rate":{"requests_per_min":1}}}}"#,
            ),
            entry(
                "rp_y",
                "t_y",
                r#"{"model_limits":{"gpt-4":{"rate":{"requests_per_min":1}}}}"#,
            ),
        ]);
        let now = 60_000;
        assert!(matches!(
            reg.resolve_for_model("rp_x", "t_x", "gpt-4o").admit(now),
            Admission::Allowed(_)
        ));
        assert!(matches!(
            reg.resolve_for_model("rp_x", "t_x", "gpt-4o").admit(now),
            Admission::Denied(_)
        )); // x exhausted
            // y untouched
        assert!(matches!(
            reg.resolve_for_model("rp_y", "t_y", "gpt-4o").admit(now),
            Admission::Allowed(_)
        ));
    }

    #[test]
    fn model_token_budget_debits_on_settle() {
        // gpt-4: 10-token daily budget. Two requests' worth of tokens trips it.
        let reg = LimitRegistry::build(vec![entry(
            "rp_t",
            "t_t",
            r#"{"model_limits":{"gpt-4":{"budget":{"tokens_daily":10}}}}"#,
        )]);
        let now = 0;
        let g = reg.resolve_for_model("rp_t", "t_t", "gpt-4o");
        assert!(matches!(g.admit(now), Admission::Allowed(_)));
        g.settle(now, 8, 0); // debit 8 tokens
                             // A different model has no counter ⇒ no debit there.
        let g2 = reg.resolve_for_model("rp_t", "t_t", "gpt-4o");
        // 8 < 10 ⇒ still allowed.
        assert!(matches!(g2.admit(now), Admission::Allowed(_)));
        g2.settle(now, 8, 0); // total 16 >= 10
        let g3 = reg.resolve_for_model("rp_t", "t_t", "gpt-4o");
        match g3.admit(now) {
            Admission::Denied(b) => {
                assert_eq!(b.scope_header(), "model");
                assert_eq!(b.kind(), LimitKind::BudgetTokens);
            }
            _ => panic!("expected model token-budget denial"),
        }
    }

    #[test]
    fn model_cost_budget_debits_on_settle() {
        let reg = LimitRegistry::build(vec![entry(
            "rp_c",
            "t_c",
            r#"{"model_limits":{"gpt-4":{"budget":{"cost_micro_usd_daily":100}}}}"#,
        )]);
        let now = 0;
        let g = reg.resolve_for_model("rp_c", "t_c", "gpt-4o");
        assert!(matches!(g.admit(now), Admission::Allowed(_)));
        g.settle(now, 0, 100); // spend the whole daily cost budget
        match reg.resolve_for_model("rp_c", "t_c", "gpt-4o").admit(now) {
            Admission::Denied(b) => {
                assert_eq!(b.scope_header(), "model");
                assert_eq!(b.kind(), LimitKind::BudgetCost);
                assert!(b.is_budget());
            }
            _ => panic!("expected model cost-budget denial"),
        }
    }

    #[test]
    fn tightest_wins_across_key_tenant_and_model() {
        // key: 10 req/min, tenant: 10 req/min, model gpt-4: 1 req/min.
        // The model cap is tightest, so the SECOND gpt-4 request is denied at
        // scope=model even though key/tenant have headroom.
        let reg = LimitRegistry::build(vec![entry(
            "rp_w",
            "t_w",
            r#"{
                "key":{"rate":{"requests_per_min":10}},
                "tenant":{"rate":{"requests_per_min":10}},
                "model_limits":{"gpt-4":{"policy_id":"gpt4","rate":{"requests_per_min":1}}}
            }"#,
        )]);
        let now = 60_000;
        assert!(matches!(
            reg.resolve_for_model("rp_w", "t_w", "gpt-4o").admit(now),
            Admission::Allowed(_)
        ));
        match reg.resolve_for_model("rp_w", "t_w", "gpt-4o").admit(now) {
            Admission::Denied(b) => {
                assert_eq!(b.scope_header(), "model");
                assert_eq!(b.policy_id(), "gpt4");
            }
            _ => panic!("expected the model cap (tightest) to win"),
        }
        // A model with NO per-model cap still gets key/tenant enforcement only.
        // The two gpt-4o admits above already consumed 2 of the 10 key/tenant
        // slots in this window, so 8 gemini requests remain before the key/tenant
        // cap (10) trips at scope != model.
        let g = reg.resolve_for_model("rp_w", "t_w", "gemini-1.5-pro");
        for _ in 0..8 {
            assert!(matches!(g.admit(now), Admission::Allowed(_)));
        }
        match g.admit(now) {
            Admission::Denied(b) => assert_ne!(b.scope_header(), "model"),
            _ => panic!("expected key/tenant cap to trip"),
        }
    }

    #[test]
    fn unconfigured_model_string_creates_no_counter() {
        // Only "gpt-4" is configured. An arbitrary client model string that matches
        // no pattern resolves to NO model scope (model: None) — bounded memory: a
        // client cannot create a counter by spraying distinct model strings.
        let reg = LimitRegistry::build(vec![entry(
            "rp_g",
            "t_g",
            r#"{"model_limits":{"gpt-4":{"rate":{"requests_per_min":1}}}}"#,
        )]);
        let g = reg.resolve_for_model("rp_g", "t_g", "some-random-model-xyz");
        assert!(g.model.is_none());
        // With no key/tenant policy either, the guard is fully unlimited.
        assert!(g.is_unlimited());
        // Many distinct unmatched model strings ⇒ still no model counter each time.
        for i in 0..1000 {
            let m = format!("junk-model-{i}");
            assert!(reg.resolve_for_model("rp_g", "t_g", &m).model.is_none());
        }
    }

    #[test]
    fn most_specific_model_pattern_wins() {
        // Both "gpt-4" and "gpt-4o-mini" configured; a gpt-4o-mini request must
        // resolve the more-specific pattern (longest-match-first).
        let reg = LimitRegistry::build(vec![entry(
            "rp_s",
            "t_s",
            r#"{"model_limits":{
                "gpt-4":{"policy_id":"broad","rate":{"requests_per_min":99}},
                "gpt-4o-mini":{"policy_id":"specific","rate":{"requests_per_min":1}}
            }}"#,
        )]);
        let now = 60_000;
        assert!(matches!(
            reg.resolve_for_model("rp_s", "t_s", "gpt-4o-mini")
                .admit(now),
            Admission::Allowed(_)
        ));
        match reg
            .resolve_for_model("rp_s", "t_s", "gpt-4o-mini")
            .admit(now)
        {
            Admission::Denied(b) => assert_eq!(b.policy_id(), "specific"),
            _ => panic!("expected the specific gpt-4o-mini cap to win"),
        }
    }

    #[test]
    fn resolve_without_model_never_sees_model_scope() {
        // The legacy 2-arg resolve() never resolves a model scope even when one is
        // configured — byte-identical for callers that don't opt in.
        let reg = LimitRegistry::build(vec![entry(
            "rp_n",
            "t_n",
            r#"{"model_limits":{"gpt-4":{"rate":{"requests_per_min":1}}}}"#,
        )]);
        let g = reg.resolve("rp_n", "t_n");
        assert!(g.model.is_none());
        assert!(g.is_unlimited());
    }
}
