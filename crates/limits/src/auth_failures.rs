//! Auth-failure rate limiting — brute-force resistance for the gateway's auth
//! middleware (security gap R0.2).
//!
//! # What this is
//! A **lock-free, fixed-memory** per-source-IP failed-authentication tracker.
//! `auth.rs` records every failed authentication (missing/invalid key, non-Active
//! tenant) against the request's source IP; once a source crosses a threshold of
//! failures inside a sliding window, the tracker short-circuits subsequent
//! requests from that source with an escalating backoff — surfaced by `auth.rs`
//! as HTTP 429 + `Retry-After`, *before* the constant-time key lookup runs.
//!
//! Constant-time lookup ([`crate`]'s sibling in `auth.rs`) already closes the
//! timing side-channel; this closes the *online brute-force* hole (an attacker
//! spraying candidate `rp_` keys).
//!
//! # Why it lives in `routeplane-limits`
//! Same doctrine as the budget/rate-limit engine in [`crate`]: an atomic,
//! network-free, injectable-clock admission primitive. It reuses the exact packed
//! `(epoch | count)` `AtomicU64` discipline and the `now_ms` clock parameter, so
//! it is deterministic under test and never touches a `Mutex` on the request path.
//!
//! # Non-negotiable invariants (mirror the budget engine)
//! - **Lock-free.** State is a fixed `Box<[AtomicU64]>` of packed slots; record +
//!   check are pure atomic loads / CAS. No `Mutex`, no allocation, no I/O per call.
//! - **Bounded memory.** A FIXED number of slots, sized once at construction — it
//!   cannot grow with the number of distinct source IPs. Source IPs are hashed
//!   into slots; a slot self-evicts when its window rolls (a fresh IP hashing into
//!   a stale slot simply starts a new window). Slot collisions can only ever cause
//!   a colliding IP to be throttled *sooner* (fail-closed), never later.
//!
//!   ⚠️ That "fail-closed" framing is correct about BRUTE FORCE and wrong about
//!   TENANT ISOLATION — which is exactly why it stayed comfortable for so long.
//!   This throttle runs PRE-AUTH, so the slot key structurally cannot carry a
//!   tenant dimension; a shared slot therefore throttles a *different tenant's*
//!   valid-key traffic, not just the offender's. Two defences, both required:
//!   the slot count is sized so accidental aliasing is rare, and the hash seed
//!   is PER-INSTANCE so the mapping cannot be computed offline and aimed at a
//!   chosen victim.
//! - **Injectable clock.** Every method takes `now_ms`; the tracker never reads
//!   the clock itself. The caller uses [`crate::now_unix_ms`].
//! - **Saturating arithmetic, never panics on a request thread.**
//! - **Fail-closed & zero-cost-when-off.** The whole feature is an `Option` at the
//!   call site; when unconfigured, `auth.rs` is byte-identical to today.

use std::collections::hash_map::RandomState;
use std::hash::BuildHasher;
use std::sync::atomic::{AtomicU64, Ordering};

// Packed slot layout: (window_epoch:40 | count:24) in one AtomicU64.
//
// The window epoch is `now_ms / window_ms` truncated to 40 bits — at a 60s
// window that is ~69 million years before the epoch counter wraps, so adjacency
// (and therefore the equality test that drives rollover) is preserved in
// practice. The count is 24 bits (saturates at ~16.7M) — far above any threshold
// an operator would set, and saturating either way (a saturated count stays at or
// above the threshold ⇒ fail-closed).
const EPOCH_BITS: u64 = 40;
const COUNT_BITS: u64 = 24;
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

/// Tunables for [`AuthFailureTracker`]. Defaults are sane brute-force-resistant
/// values; all are env-overridable by the caller (`auth.rs` reads the env).
#[derive(Debug, Clone, Copy)]
pub struct AuthFailureConfig {
    /// Failures within one window before the source is throttled. Reaching this
    /// count trips the 429 path for the rest of the backoff.
    pub threshold: u64,
    /// Sliding-window length in milliseconds. Failures are counted per window;
    /// a quiet window resets the count (eviction-by-rollover).
    pub window_ms: u64,
    /// Base backoff in milliseconds, scaled exponentially by how far over the
    /// threshold the source is: `backoff = base * 2^(overshoot)`, capped at
    /// [`AuthFailureConfig::backoff_cap_ms`]. Drives the `Retry-After` value.
    pub backoff_base_ms: u64,
    /// Upper bound on the computed backoff (so `Retry-After` never explodes).
    pub backoff_cap_ms: u64,
    /// Number of atomic slots. Fixed ⇒ bounded memory; sized to keep collisions
    /// rare for the expected distinct-IP volume of a single replica.
    pub slots: usize,
}

impl Default for AuthFailureConfig {
    fn default() -> Self {
        Self {
            threshold: 5,
            window_ms: 60_000,       // 1 minute
            backoff_base_ms: 1_000,  // 1s, doubling per overshoot
            backoff_cap_ms: 300_000, // 5 minutes
            // 64Ki slots = 512 KiB of atomics, still trivially bounded. 4096
            // was too few for the ACCIDENTAL half of the collision problem: a
            // replica seeing ~100 distinct client IPs had a ~70% chance
            // (birthday) that some pair shared a slot, and a shared slot means
            // one tenant's failed-auth burst 429s a DIFFERENT tenant's
            // valid-key traffic for up to `backoff_cap_ms`, with nothing in
            // either tenant's view explaining why. 16x the slots is 16x less of
            // that.
            //
            // This does NOT fix the TARGETED attack — only the per-instance
            // `hasher` seed does, because an attacker who can compute the
            // mapping offline just searches a bigger space. Both changes are
            // needed; neither substitutes for the other.
            slots: 65_536,
        }
    }
}

impl AuthFailureConfig {
    /// Clamp to safe, representable bounds (never panics; never produces a config
    /// that silently disables enforcement). A zero/oversized threshold or a zero
    /// slot count would break the fail-closed property, so we floor/ceil them.
    fn sanitized(mut self) -> Self {
        self.threshold = self.threshold.clamp(1, COUNT_MASK);
        self.window_ms = self.window_ms.max(1);
        self.backoff_base_ms = self.backoff_base_ms.max(1);
        self.backoff_cap_ms = self.backoff_cap_ms.max(self.backoff_base_ms);
        self.slots = self.slots.clamp(1, 1 << 20);
        self
    }
}

/// The outcome of consulting the tracker on the auth path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthThrottle {
    /// The source is under the threshold; proceed with the (constant-time) lookup.
    Allow,
    /// The source is over the threshold; short-circuit with 429 + this
    /// `Retry-After` (seconds, always >= 1).
    Throttled { retry_after_secs: u64 },
}

impl AuthThrottle {
    pub fn is_throttled(self) -> bool {
        matches!(self, AuthThrottle::Throttled { .. })
    }
}

/// Lock-free, fixed-memory per-source failed-auth tracker.
///
/// Held behind an `Arc` and injected as a request extension. Cloning the `Arc` is
/// the only per-request cost on the disabled→enabled boundary; the hot operations
/// (`check`, `record_failure`) are atomic loads / CAS against a pre-resolved slot.
#[derive(Debug)]
pub struct AuthFailureTracker {
    cfg: AuthFailureConfig,
    slots: Box<[AtomicU64]>,
    /// Per-instance hash seed. `DefaultHasher::new()` — what this used to
    /// build — is FIXED-seed and therefore identical in every process, so an
    /// attacker could compute OFFLINE which of its own egress IPs lands in the
    /// same `hash % slots` slot as a victim's, then park that victim in
    /// `Retry-After` backoff with a handful of deliberately-bad-key requests,
    /// without ever touching the victim's traffic or guessing anything secret.
    ///
    /// This check is pre-auth, so the slot key CANNOT carry a tenant
    /// dimension; the seed is the only thing standing between a shared slot
    /// array and a targeted cross-tenant denial. `RandomState` reseeds per
    /// process, which makes that offline search infeasible. Read-only after
    /// construction ⇒ the hot path stays lock-free.
    hasher: RandomState,
}

impl AuthFailureTracker {
    /// Build a tracker with the given config (sanitized to safe bounds).
    pub fn new(cfg: AuthFailureConfig) -> Self {
        let cfg = cfg.sanitized();
        let mut v = Vec::with_capacity(cfg.slots);
        v.resize_with(cfg.slots, || AtomicU64::new(0));
        Self {
            cfg,
            slots: v.into_boxed_slice(),
            hasher: RandomState::new(),
        }
    }

    /// The active config (after sanitization) — exposed for logging at startup.
    pub fn config(&self) -> &AuthFailureConfig {
        &self.cfg
    }

    #[inline]
    fn epoch(&self, now_ms: u64) -> u64 {
        now_ms / self.cfg.window_ms
    }

    /// Map a source key (the source IP string) to a slot index. A stable,
    /// allocation-free hash; the slot count is fixed so this is the *only* place
    /// the per-IP-to-memory mapping happens.
    ///
    /// Stable *within* one process (a repeat offender must keep accumulating
    /// against one slot) and unpredictable *across* processes (see
    /// [`AuthFailureTracker::hasher`]).
    #[inline]
    fn slot_index(&self, source: &str) -> usize {
        self.hasher.hash_one(source) as usize % self.slots.len()
    }

    #[inline]
    fn slot_for(&self, source: &str) -> &AtomicU64 {
        // Index is always in range (modulo slots.len(), which is >= 1).
        &self.slots[self.slot_index(source)]
    }

    /// Current failure count for `source` in the active window (0 if its slot is
    /// stale / rolled). A pure atomic load — used by [`Self::check`].
    fn current_count(&self, source: &str, now_ms: u64) -> u64 {
        let ep = self.epoch(now_ms) & EPOCH_MASK;
        let (e, c) = unpack(self.slot_for(source).load(Ordering::Acquire));
        if e == ep {
            c
        } else {
            0
        }
    }

    /// Exponential backoff (ms) for a source whose in-window count is `count`,
    /// given the configured threshold: `base * 2^(count - threshold)`, capped.
    /// `count <= threshold` yields the base (the first trip), and the shift is
    /// bounded so it can never overflow.
    fn backoff_ms(&self, count: u64) -> u64 {
        let overshoot = count.saturating_sub(self.cfg.threshold);
        // Cap the shift: 2^40 * base already saturates past any sane cap, and a
        // shift >= 63 is UB for u64 — clamp well below that.
        let shift = overshoot.min(40) as u32;
        let scaled = self.cfg.backoff_base_ms.saturating_mul(1u64 << shift);
        scaled.min(self.cfg.backoff_cap_ms)
    }

    /// Read-only admission check, run *before* the key lookup. Returns
    /// [`AuthThrottle::Throttled`] (with a `Retry-After`) when `source` is at or
    /// over the threshold in the current window, else [`AuthThrottle::Allow`].
    ///
    /// Read-only by design: a *throttled* request must NOT count as a fresh
    /// failure (otherwise a throttled attacker would extend their own backoff for
    /// free, and a legitimate client that trips once could never recover). Only a
    /// genuine auth failure that is *allowed through* records ([`Self::record_failure`]).
    pub fn check(&self, source: &str, now_ms: u64) -> AuthThrottle {
        let count = self.current_count(source, now_ms);
        if count >= self.cfg.threshold {
            let secs = self.backoff_ms(count).div_ceil(1000).max(1);
            AuthThrottle::Throttled {
                retry_after_secs: secs,
            }
        } else {
            AuthThrottle::Allow
        }
    }

    /// Record one failed authentication for `source` in the current window,
    /// rolling a stale slot to a fresh window seeded at 1. Lock-free CAS loop;
    /// the count saturates (fail-closed). Returns the post-increment count (for
    /// the caller's structured logging / future security-event emission).
    pub fn record_failure(&self, source: &str, now_ms: u64) -> u64 {
        let ep = self.epoch(now_ms) & EPOCH_MASK;
        let slot = self.slot_for(source);
        loop {
            let cur = slot.load(Ordering::Acquire);
            let (e, c) = unpack(cur);
            let base = if e == ep { c } else { 0 };
            let next = base.saturating_add(1).min(COUNT_MASK);
            let newp = pack(ep, next);
            if slot
                .compare_exchange_weak(cur, newp, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return next;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(threshold: u64, window_ms: u64) -> AuthFailureConfig {
        AuthFailureConfig {
            threshold,
            window_ms,
            backoff_base_ms: 1_000,
            backoff_cap_ms: 300_000,
            slots: 1_024,
        }
    }

    #[test]
    fn pack_unpack_round_trips() {
        let (e, c) = unpack(pack(42, 7));
        assert_eq!((e, c), (42, 7));
        // pack() MASKS (truncates) both fields; the count is saturated to
        // COUNT_MASK by the caller (record_failure) BEFORE packing, so the slot
        // never wraps — verified by `count_saturates_and_never_panics`.
        let (e2, c2) = unpack(pack(EPOCH_MASK + 3, COUNT_MASK));
        assert_eq!(e2, (EPOCH_MASK + 3) & EPOCH_MASK);
        assert_eq!(c2, COUNT_MASK);
    }

    #[test]
    fn under_threshold_allows() {
        let t = AuthFailureTracker::new(cfg(5, 60_000));
        let now = 0;
        // 4 failures (< 5) — still allowed.
        for _ in 0..4 {
            assert_eq!(t.check("1.2.3.4", now), AuthThrottle::Allow);
            t.record_failure("1.2.3.4", now);
        }
        assert_eq!(t.check("1.2.3.4", now), AuthThrottle::Allow);
    }

    #[test]
    fn threshold_trips_throttle_with_retry_after() {
        let t = AuthFailureTracker::new(cfg(5, 60_000));
        let now = 0;
        // Accumulate exactly the threshold.
        for _ in 0..5 {
            t.record_failure("9.9.9.9", now);
        }
        match t.check("9.9.9.9", now) {
            AuthThrottle::Throttled { retry_after_secs } => {
                // First trip (count == threshold) → base backoff = 1000ms = 1s.
                assert_eq!(retry_after_secs, 1);
            }
            AuthThrottle::Allow => panic!("expected throttle at threshold"),
        }
    }

    #[test]
    fn backoff_escalates_exponentially_and_is_capped() {
        let t = AuthFailureTracker::new(AuthFailureConfig {
            threshold: 3,
            window_ms: 60_000,
            backoff_base_ms: 1_000,
            backoff_cap_ms: 8_000,
            slots: 64,
        });
        // count == threshold (3): 1000 * 2^0 = 1s
        assert_eq!(t.backoff_ms(3), 1_000);
        // count threshold+1 (4): 1000 * 2^1 = 2s
        assert_eq!(t.backoff_ms(4), 2_000);
        // count threshold+2 (5): 1000 * 2^2 = 4s
        assert_eq!(t.backoff_ms(5), 4_000);
        // count threshold+3 (6): 1000 * 2^3 = 8s
        assert_eq!(t.backoff_ms(6), 8_000);
        // count threshold+4 (7): would be 16s but capped at 8s.
        assert_eq!(t.backoff_ms(7), 8_000);
        // Far overshoot never panics / overflows.
        assert_eq!(t.backoff_ms(u64::MAX), 8_000);
    }

    #[test]
    fn window_rollover_resets_the_count() {
        let win = 60_000;
        let t = AuthFailureTracker::new(cfg(2, win));
        let now = 0;
        t.record_failure("5.5.5.5", now);
        t.record_failure("5.5.5.5", now);
        assert!(t.check("5.5.5.5", now).is_throttled());
        // A request in the NEXT window sees a fresh (rolled) count — eviction by
        // rollover, no growth, no manual sweep.
        let next_window = win; // epoch 1
        assert_eq!(t.check("5.5.5.5", next_window), AuthThrottle::Allow);
        assert_eq!(t.current_count("5.5.5.5", next_window), 0);
    }

    #[test]
    fn distinct_sources_are_independent() {
        let t = AuthFailureTracker::new(cfg(2, 60_000));
        let now = 0;
        t.record_failure("10.0.0.1", now);
        t.record_failure("10.0.0.1", now);
        assert!(t.check("10.0.0.1", now).is_throttled());
        // A source in a DIFFERENT slot is untouched. The peer is resolved
        // against this instance rather than hard-coded: the seed is
        // per-instance now, so a fixed peer string would alias the exhausted
        // slot on 1-in-`slots` runs and flake. Resolving it keeps the assertion
        // about *independence*, which is what this test is for.
        let busy = t.slot_index("10.0.0.1");
        let peer = (0..1_000)
            .map(|i| format!("10.0.1.{i}"))
            .find(|s| t.slot_index(s) != busy)
            .expect("a non-colliding peer must exist among 1000 candidates");
        assert_eq!(t.check(&peer, now), AuthThrottle::Allow);
    }

    #[test]
    fn throttled_check_does_not_itself_increment() {
        // check() is read-only: consulting a throttled source must not extend its
        // own backoff. The count only moves on record_failure().
        let t = AuthFailureTracker::new(cfg(2, 60_000));
        let now = 0;
        t.record_failure("2.2.2.2", now);
        t.record_failure("2.2.2.2", now);
        let before = t.current_count("2.2.2.2", now);
        let _ = t.check("2.2.2.2", now);
        let _ = t.check("2.2.2.2", now);
        assert_eq!(t.current_count("2.2.2.2", now), before);
    }

    #[test]
    fn count_saturates_and_never_panics() {
        let t = AuthFailureTracker::new(cfg(1, 60_000));
        let now = 0;
        // Hammering one source must never panic; the count clamps at the mask.
        for _ in 0..10_000 {
            t.record_failure("8.8.8.8", now);
        }
        assert!(t.check("8.8.8.8", now).is_throttled());
    }

    #[test]
    fn sanitize_floors_unsafe_config() {
        let c = AuthFailureConfig {
            threshold: 0,
            window_ms: 0,
            backoff_base_ms: 0,
            backoff_cap_ms: 0,
            slots: 0,
        }
        .sanitized();
        assert_eq!(c.threshold, 1);
        assert!(c.window_ms >= 1);
        assert!(c.backoff_base_ms >= 1);
        assert!(c.backoff_cap_ms >= c.backoff_base_ms);
        assert!(c.slots >= 1);
        // A 1-slot tracker still functions (over-throttles collisions, never
        // under-throttles — fail-closed).
        let t = AuthFailureTracker::new(c);
        t.record_failure("x", 0);
        assert!(t.check("x", 0).is_throttled());
    }

    #[test]
    fn bounded_memory_independent_of_distinct_sources() {
        // 64 slots, but throw 100k distinct sources at it — memory is fixed.
        let t = AuthFailureTracker::new(AuthFailureConfig {
            threshold: 1_000_000, // high so collisions don't trip the assertion
            window_ms: 60_000,
            backoff_base_ms: 1_000,
            backoff_cap_ms: 300_000,
            slots: 64,
        });
        for i in 0..100_000u64 {
            t.record_failure(&format!("src-{i}"), 0);
        }
        assert_eq!(t.slots.len(), 64); // never grew
    }

    /// REGRESSION (cross-tenant control) — the TARGETED half.
    ///
    /// `slot_for` used to build `DefaultHasher::new()`, which is FIXED-seed and
    /// therefore produces the identical `source -> slot` mapping in every
    /// process on every host. This throttle runs PRE-AUTH, so the slot key
    /// cannot carry a tenant dimension; the seed is the only thing preventing
    /// one tenant from aiming the shared slot array at another.
    ///
    /// The scenario, with two named tenants. Tenant `acme` is the victim, whose
    /// fleet egresses from 198.51.100.7. Tenant `globex` is the attacker, and
    /// controls the 203.0.0.0/16 elastic-IP pool.
    ///
    /// `globex` computes OFFLINE (no requests, nothing secret) which of its own
    /// addresses lands in `acme`'s slot, then sends `threshold` bad-key requests
    /// from it. With a fixed seed that answer transfers to EVERY replica, so
    /// `acme`'s next VALID-key request is 429'd for up to `backoff_cap_ms`.
    ///
    /// This asserts the offline answer does NOT transfer to freshly-seeded
    /// processes. On the old code all 16 transfer (deterministic failure); with
    /// a per-instance `RandomState` each has an independent 1/`slots` chance, so
    /// `>= 2` transferring has probability ~C(16,2)/4096^2 ~= 7e-6.
    ///
    /// `slots` is pinned small here on purpose: it makes the attacker's search
    /// cheap enough to run in a unit test, and it proves the SEED alone carries
    /// this property — raising the slot count does not (an attacker just
    /// searches a bigger space). The slot count defends the accidental half,
    /// asserted separately below.
    #[test]
    fn a_tenants_offline_slot_search_does_not_transfer_to_other_instances() {
        const VICTIM: &str = "198.51.100.7"; // tenant `acme`
        let attack_cfg = AuthFailureConfig {
            threshold: 2,
            window_ms: 60_000,
            backoff_base_ms: 1_000,
            backoff_cap_ms: 300_000,
            slots: 4_096,
        };
        let now = 0;

        // Step 1 — the offline search, done once against one instance. Expected
        // depth is ~`slots` (4096) into a 65536-address pool, so "not found" has
        // probability e^-16 ~= 1e-7.
        let recon = AuthFailureTracker::new(attack_cfg);
        let victim_slot = recon.slot_index(VICTIM);
        let weapon = (0..65_536u32) // tenant `globex`'s own 203.0.0.0/16
            .map(|i| format!("203.0.{}.{}", i / 256, i % 256))
            .find(|ip| recon.slot_index(ip) == victim_slot)
            .expect("a colliding source must exist in a /16 search");

        // It works against the instance it was computed on — that is the
        // capability being denied, not a property we want.
        for _ in 0..attack_cfg.threshold {
            recon.record_failure(&weapon, now);
        }
        assert!(
            recon.check(VICTIM, now).is_throttled(),
            "sanity: the search must actually find a colliding source"
        );

        // Step 2 — fire the SAME address at independently-constructed trackers,
        // standing in for other replicas / restarts.
        let transferred = (0..16)
            .filter(|_| {
                let replica = AuthFailureTracker::new(attack_cfg);
                for _ in 0..attack_cfg.threshold {
                    replica.record_failure(&weapon, now);
                }
                replica.check(VICTIM, now).is_throttled()
            })
            .count();

        assert!(
            transferred <= 1,
            "an offline-computed slot collision transferred to {transferred}/16 \
             freshly-seeded instances — the slot hash is process-stable again, \
             which lets one tenant park another tenant's valid traffic in \
             Retry-After backoff"
        );
    }

    /// The direction this fix could plausibly break: the mapping must still be
    /// STABLE within one instance, or a repeat offender would spray across
    /// slots and never accumulate to the threshold.
    #[test]
    fn slot_mapping_is_stable_within_an_instance() {
        let t = AuthFailureTracker::new(AuthFailureConfig::default());
        let first = t.slot_index("198.51.100.4");
        for _ in 0..1_000 {
            assert_eq!(t.slot_index("198.51.100.4"), first);
        }
    }

    /// REGRESSION (cross-tenant control) — the ACCIDENTAL half, which the seed
    /// does NOT address.
    ///
    /// At 4096 slots a replica seeing ~100 distinct client IPs had a ~70%
    /// birthday chance that some pair aliased. Then tenant `acme` rotates a key,
    /// its fleet emits a burst of stale-key 401s, and tenant `globex`'s next
    /// VALID request is 429'd — with nothing in either tenant's view explaining
    /// why.
    #[test]
    fn default_slot_count_keeps_accidental_collisions_rare() {
        let cfg = AuthFailureConfig::default().sanitized();
        assert!(
            cfg.slots >= 65_536,
            "default slots {} is too few — accidental cross-tenant aliasing",
            cfg.slots
        );
        let t = AuthFailureTracker::new(cfg);
        let idx: std::collections::HashSet<usize> = (0..100)
            .map(|i| t.slot_index(&format!("203.0.113.{i}")))
            .collect();
        // 100 distinct sources should almost always occupy ~100 distinct slots.
        assert!(
            idx.len() >= 97,
            "100 sources collapsed into {} slots — aliasing is too likely",
            idx.len()
        );
    }
}
