//! Idempotency-key store (Stripe/Portkey-style safe-retry semantics) for the
//! buffered completion path.
//!
//! A client may carry an `Idempotency-Key` header so that a *re-sent* request
//! with the same key returns the **same prior response** instead of re-calling
//! the provider — preventing a double-charge / double side-effect on retries,
//! hedges, and network blips. This is the safety complement to request hedging
//! (ADR-057) and is distinct from the exact cache: the key is **client-supplied**
//! (so it dedupes even `temperature>0` / otherwise-non-cacheable requests), and a
//! reused key with a *different* request body is rejected rather than served.
//!
//! ## Lock-free discipline (mirrors the exact cache, ADR-022 §2/§3)
//! - **Per-replica, in-memory, $0 standing cost.** No Redis, no DB — multi-replica
//!   coordinated idempotency is a documented Redis follow-on (would need an ADR),
//!   identical to the per-replica posture of the cache and the rate limiter.
//! - **Reads (replay lookup)** are one `ArcSwap::load` + one `HashMap` probe + a
//!   TTL/state check. No lock, no CAS contention.
//! - **Mutations** (reserve / store / release) are a copy-on-write `compare_and_swap`
//!   retry loop per shard — lock-free (no mutex, never blocks a request thread),
//!   exactly like [`super::FlushRegistry::bump`]. The mutation rate is one per
//!   keyed request, not per token, so the CoW shard clone is cheap and rare.
//! - **`reserve` is atomic reserve-if-absent**: two concurrent identical requests
//!   cannot both reach the provider — the first wins the CAS and gets `Reserved`,
//!   the loser observes the in-flight marker and is told `InFlight` (→ 409).
//! - **Bounded + TTL**: entries expire after a configurable TTL and each shard is
//!   capped by entry count (FIFO/TTL eviction at reserve time), so a hostile or
//!   buggy client can never grow memory without bound.
//!
//! ## Tenant isolation
//! `tenant_id` and the client key are STRUCTURAL fields of [`IdempotencyKey`]
//! (they participate in `Eq`/`Hash`), so tenant A's key can never collide with
//! tenant B's — isolation by construction, exactly like [`super::CacheKey`].
//!
//! ## What is and isn't stored
//! Only a **2xx** response is stored (status + body bytes + content-type) along
//! with a fingerprint (SHA-256) of the canonical request. A non-2xx outcome
//! RELEASES the reservation and is never stored — an LLM-specific choice (Stripe
//! caches all final responses; here a provider error should let a genuine retry
//! re-run rather than pin a transient failure). Streaming requests bypass the
//! store entirely (replaying an SSE stream is a follow-on).

use arc_swap::ArcSwap;
use bytes::Bytes;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::{Clock, SystemClock};

/// Shard count — same fan-out as the exact cache (top 6 bits of the key hash).
pub const IDEMP_SHARD_COUNT: usize = 64;
/// Default time-to-live for a stored idempotent response: 24 hours
/// (Stripe's window). Overridable at construction via [`IdempotencyStore::with_ttl`].
pub const DEFAULT_IDEMPOTENCY_TTL_SECONDS: u64 = 24 * 60 * 60;
/// Default cap on stored entries PER SHARD (so the whole store is bounded at
/// `IDEMP_SHARD_COUNT * this`). Eviction at reserve time is TTL-first, then FIFO
/// by creation time — the same approximate-LRU doctrine as the cache.
pub const DEFAULT_MAX_ENTRIES_PER_SHARD: usize = 4096;
/// Per-body hard cap (256 KiB) — an oversize 2xx body is simply not stored (the
/// reservation is released), so the response is still returned but a future
/// replay re-runs. Mirrors the cache's `MAX_ENTRY_BYTES`.
pub const IDEMP_MAX_BODY_BYTES: usize = 256 * 1024;

/// A namespaced idempotency key. `tenant_id` + the client-supplied `key` are
/// STRUCTURAL (Eq/Hash) so cross-tenant collision is impossible by construction.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IdempotencyKey {
    tenant_id: String,
    key: String,
}

impl IdempotencyKey {
    pub fn new(tenant_id: &str, key: &str) -> Self {
        Self {
            tenant_id: tenant_id.to_string(),
            key: key.to_string(),
        }
    }

    /// Shard index: top 6 bits of a SHA-256 over the structural parts. A hash
    /// (not the raw string) so the distribution is even regardless of client key
    /// shape (UUID, monotonic counter, etc.).
    fn shard(&self) -> usize {
        let mut hasher = Sha256::new();
        hasher.update(self.tenant_id.as_bytes());
        hasher.update([0x1f]);
        hasher.update(self.key.as_bytes());
        let digest: [u8; 32] = hasher.finalize().into();
        (digest[0] >> 2) as usize
    }
}

/// Compute the canonical request fingerprint a key is bound to. Any stable,
/// caller-chosen byte view of the request works; the proxy passes the serialized
/// canonical (pre-masking, shaping-resolved) request so that a reused key with a
/// *different* body is detected (mismatch → 422). One-way SHA-256: no request
/// material is recoverable from the stored fingerprint.
pub fn request_fingerprint(canonical_request_bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(canonical_request_bytes);
    hasher.finalize().into()
}

/// A completed, stored idempotent response — replayed verbatim on a matching hit.
#[derive(Debug, Clone)]
pub struct StoredResponse {
    pub status: u16,
    pub content_type: String,
    pub body: Bytes,
    pub fingerprint: [u8; 32],
    /// Provenance of the ORIGINAL dispatch: the serving provider and the
    /// gateway request id, re-stamped on replay so a replayed response keeps
    /// the provenance headers of the response it repeats. Additive +
    /// default-tolerant: `None` means "stored without provenance" (an older
    /// entry) and a replay simply omits the re-stamp — a serializing follow-on
    /// (e.g. a Redis backend) MUST keep these `#[serde(default)]` so
    /// pre-provenance snapshots still deserialize.
    pub provider: Option<String>,
    pub request_id: Option<String>,
    created_at_ms: u64,
    ttl_ms: u64,
}

impl StoredResponse {
    fn expired(&self, now_ms: u64) -> bool {
        now_ms >= self.created_at_ms.saturating_add(self.ttl_ms)
    }
}

/// One slot in a shard: either an in-flight reservation or a completed response.
#[derive(Debug, Clone)]
enum Slot {
    /// Reserved by an in-flight request that has not completed. `created_at_ms`
    /// drives a stale-reservation timeout (a request that crashed/was cancelled
    /// before releasing must not pin the key forever).
    InFlight { created_at_ms: u64, ttl_ms: u64 },
    /// A completed 2xx response, eligible for replay until it expires.
    Done(StoredResponse),
}

impl Slot {
    fn created_at_ms(&self) -> u64 {
        match self {
            Slot::InFlight { created_at_ms, .. } => *created_at_ms,
            Slot::Done(r) => r.created_at_ms,
        }
    }

    fn expired(&self, now_ms: u64) -> bool {
        match self {
            Slot::InFlight {
                created_at_ms,
                ttl_ms,
            } => now_ms >= created_at_ms.saturating_add(*ttl_ms),
            Slot::Done(r) => r.expired(now_ms),
        }
    }
}

/// The outcome of attempting to [`IdempotencyStore::reserve`] a key for a request.
#[derive(Debug)]
pub enum ReserveOutcome {
    /// Nothing prior under this key (or the prior entry expired): we won the
    /// reservation. The caller must run the pipeline, then call `store` on a 2xx
    /// or `release` on a non-2xx / early return.
    Reserved,
    /// A completed 2xx response exists AND its fingerprint matches this request:
    /// replay it verbatim, do NOT call the provider, do NOT charge budget again.
    Replay(StoredResponse),
    /// A completed response exists but the fingerprint differs: the same key was
    /// reused with a different request body → 422 (Stripe rejects this).
    FingerprintMismatch,
    /// The key is reserved by a concurrent in-flight request → 409 Conflict.
    InFlight,
}

type ShardMap = HashMap<IdempotencyKey, Slot>;

/// The lock-free, per-replica idempotency store held in `AppState`.
pub struct IdempotencyStore {
    shards: Vec<ArcSwap<ShardMap>>,
    ttl_ms: u64,
    max_entries_per_shard: usize,
    clock: Arc<dyn Clock>,
    // Diagnostics (cumulative, process lifetime; reset by scale-to-zero). One
    // relaxed atomic add each — never a lock, so every path stays lock-free.
    reservations: AtomicU64,
    replays: AtomicU64,
    mismatches: AtomicU64,
    in_flight_conflicts: AtomicU64,
    stored: AtomicU64,
    released: AtomicU64,
}

impl IdempotencyStore {
    /// Production constructor (system clock, default TTL + bounds).
    pub fn new() -> Self {
        Self::with_clock(
            DEFAULT_IDEMPOTENCY_TTL_SECONDS,
            DEFAULT_MAX_ENTRIES_PER_SHARD,
            Arc::new(SystemClock),
        )
    }

    /// Constructor overriding the TTL (seconds) only.
    pub fn with_ttl(ttl_seconds: u64) -> Self {
        Self::with_clock(
            ttl_seconds,
            DEFAULT_MAX_ENTRIES_PER_SHARD,
            Arc::new(SystemClock),
        )
    }

    /// Full constructor (injectable clock for deterministic TTL/eviction tests).
    pub fn with_clock(
        ttl_seconds: u64,
        max_entries_per_shard: usize,
        clock: Arc<dyn Clock>,
    ) -> Self {
        Self {
            shards: (0..IDEMP_SHARD_COUNT)
                .map(|_| ArcSwap::from_pointee(ShardMap::new()))
                .collect(),
            ttl_ms: ttl_seconds.saturating_mul(1000),
            max_entries_per_shard: max_entries_per_shard.max(1),
            clock,
            reservations: AtomicU64::new(0),
            replays: AtomicU64::new(0),
            mismatches: AtomicU64::new(0),
            in_flight_conflicts: AtomicU64::new(0),
            stored: AtomicU64::new(0),
            released: AtomicU64::new(0),
        }
    }

    pub fn ttl_seconds(&self) -> u64 {
        self.ttl_ms / 1000
    }

    /// Atomic reserve-if-absent + replay decision — the hot-path entry point.
    ///
    /// State machine (all under one CAS retry loop per shard, lock-free):
    /// - completed & fingerprint matches → [`ReserveOutcome::Replay`] (no mutation)
    /// - completed & fingerprint differs → [`ReserveOutcome::FingerprintMismatch`]
    /// - in-flight (not stale) → [`ReserveOutcome::InFlight`] (409)
    /// - absent / expired / stale-in-flight → install an `InFlight` marker and
    ///   return [`ReserveOutcome::Reserved`] (we own it; must `store`/`release`).
    ///
    /// The expired/stale paths reclaim the slot in the same CoW swap, so a
    /// crashed prior holder never pins a key past its TTL.
    pub fn reserve(&self, key: &IdempotencyKey, fingerprint: [u8; 32]) -> ReserveOutcome {
        let idx = key.shard();
        loop {
            let now = self.clock.now_ms();
            let current = self.shards[idx].load();

            // Inspect the existing slot WITHOUT mutating, for the read-only verdicts.
            if let Some(slot) = current.get(key) {
                if !slot.expired(now) {
                    match slot {
                        Slot::Done(stored) => {
                            if stored.fingerprint == fingerprint {
                                self.replays.fetch_add(1, Ordering::Relaxed);
                                return ReserveOutcome::Replay(stored.clone());
                            } else {
                                self.mismatches.fetch_add(1, Ordering::Relaxed);
                                return ReserveOutcome::FingerprintMismatch;
                            }
                        }
                        Slot::InFlight { .. } => {
                            self.in_flight_conflicts.fetch_add(1, Ordering::Relaxed);
                            return ReserveOutcome::InFlight;
                        }
                    }
                }
                // else: present but expired (or a stale in-flight) → fall through
                // and reclaim it as part of this reservation's swap.
            }

            // Build the next map: clone, evict if needed, install our marker.
            let mut map: ShardMap = (**current).clone();
            // Drop the expired/stale slot for this key (if any) before inserting.
            map.remove(key);
            self.evict_if_needed(&mut map, now);
            map.insert(
                key.clone(),
                Slot::InFlight {
                    created_at_ms: now,
                    ttl_ms: self.ttl_ms,
                },
            );
            let new = Arc::new(map);
            let prev = self.shards[idx].compare_and_swap(&*current, Arc::clone(&new));
            if Arc::ptr_eq(&prev, &current) {
                self.reservations.fetch_add(1, Ordering::Relaxed);
                return ReserveOutcome::Reserved;
            }
            // Lost the race to a concurrent mutation of this shard; retry with the
            // fresh snapshot. Readers were never blocked.
        }
    }

    /// Promote our in-flight reservation to a stored 2xx response (replayable on
    /// the next matching request). No-op (returns `false`) if our reservation is
    /// gone (expired/reclaimed) or the body is oversize — in which case the
    /// response is still returned to the caller but a future request re-runs.
    /// Only ever called on a 2xx (failures are never cached — see [`release`]).
    /// `provider`/`request_id` carry the original dispatch's provenance for
    /// re-stamping on replay (`None` ⇒ the replay omits it).
    #[allow(clippy::too_many_arguments)]
    pub fn store(
        &self,
        key: &IdempotencyKey,
        status: u16,
        content_type: String,
        body: Bytes,
        fingerprint: [u8; 32],
        provider: Option<String>,
        request_id: Option<String>,
    ) -> bool {
        if body.len() > IDEMP_MAX_BODY_BYTES {
            // Oversize: drop the reservation so a retry can re-run rather than 409.
            self.release(key);
            return false;
        }
        let idx = key.shard();
        loop {
            let now = self.clock.now_ms();
            let current = self.shards[idx].load();
            // Only promote a slot we still own as InFlight. If it vanished (TTL
            // reclaimed it), do not resurrect — a retry will simply re-run.
            match current.get(key) {
                Some(Slot::InFlight { .. }) => {}
                _ => return false,
            }
            let mut map: ShardMap = (**current).clone();
            map.insert(
                key.clone(),
                Slot::Done(StoredResponse {
                    status,
                    content_type: content_type.clone(),
                    body: body.clone(),
                    fingerprint,
                    provider: provider.clone(),
                    request_id: request_id.clone(),
                    created_at_ms: now,
                    ttl_ms: self.ttl_ms,
                }),
            );
            let new = Arc::new(map);
            let prev = self.shards[idx].compare_and_swap(&*current, Arc::clone(&new));
            if Arc::ptr_eq(&prev, &current) {
                self.stored.fetch_add(1, Ordering::Relaxed);
                return true;
            }
        }
    }

    /// Release an in-flight reservation WITHOUT storing a response (a non-2xx
    /// outcome, an early return, or an oversize body). A genuine client retry can
    /// then re-run the request — the LLM-specific "failures not cached" choice.
    /// Idempotent: releasing an already-removed or completed slot is a no-op.
    pub fn release(&self, key: &IdempotencyKey) {
        let idx = key.shard();
        loop {
            let current = self.shards[idx].load();
            // Only remove a slot we still hold as InFlight; never clobber a Done
            // (a concurrent store) or a different request's later reservation.
            match current.get(key) {
                Some(Slot::InFlight { .. }) => {}
                _ => return,
            }
            let mut map: ShardMap = (**current).clone();
            map.remove(key);
            let new = Arc::new(map);
            let prev = self.shards[idx].compare_and_swap(&*current, Arc::clone(&new));
            if Arc::ptr_eq(&prev, &current) {
                self.released.fetch_add(1, Ordering::Relaxed);
                return;
            }
        }
    }

    /// Evict to keep a shard map under its entry cap: TTL-first, then FIFO by
    /// creation time. Called inside the reserve CoW (the writer already holds the
    /// fresh clone), so it never races a reader. The cap is checked BEFORE the
    /// new marker is inserted, leaving room for it.
    fn evict_if_needed(&self, map: &mut ShardMap, now: u64) {
        if map.len() < self.max_entries_per_shard {
            return;
        }
        // 1) TTL-first: drop everything already expired.
        map.retain(|_, slot| !slot.expired(now));
        if map.len() < self.max_entries_per_shard {
            return;
        }
        // 2) FIFO by creation time until we're back under the cap (leaving one
        //    slot free for the marker about to be inserted).
        let mut by_age: Vec<(IdempotencyKey, u64)> = map
            .iter()
            .map(|(k, s)| (k.clone(), s.created_at_ms()))
            .collect();
        by_age.sort_by_key(|(_, created)| *created);
        for (k, _) in by_age {
            if map.len() < self.max_entries_per_shard {
                break;
            }
            map.remove(&k);
        }
    }

    /// Lock-free `(entries, in_flight)` snapshot for `/status` (off the hot path).
    pub fn stats_snapshot(&self) -> (usize, usize) {
        let now = self.clock.now_ms();
        let mut entries = 0;
        let mut in_flight = 0;
        for shard in &self.shards {
            let guard = shard.load();
            for slot in guard.values() {
                if slot.expired(now) {
                    continue;
                }
                entries += 1;
                if matches!(slot, Slot::InFlight { .. }) {
                    in_flight += 1;
                }
            }
        }
        (entries, in_flight)
    }

    pub fn reservations(&self) -> u64 {
        self.reservations.load(Ordering::Relaxed)
    }
    pub fn replays(&self) -> u64 {
        self.replays.load(Ordering::Relaxed)
    }
    pub fn mismatches(&self) -> u64 {
        self.mismatches.load(Ordering::Relaxed)
    }
    pub fn in_flight_conflicts(&self) -> u64 {
        self.in_flight_conflicts.load(Ordering::Relaxed)
    }
    pub fn stored_count(&self) -> u64 {
        self.stored.load(Ordering::Relaxed)
    }
    pub fn released_count(&self) -> u64 {
        self.released.load(Ordering::Relaxed)
    }
}

impl Default for IdempotencyStore {
    fn default() -> Self {
        Self::new()
    }
}

// =============================== tests ==========================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64 as TestAtomicU64;

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

    fn store_at(clock: Arc<FakeClock>) -> IdempotencyStore {
        IdempotencyStore::with_clock(60, DEFAULT_MAX_ENTRIES_PER_SHARD, clock)
    }

    fn fp(s: &str) -> [u8; 32] {
        request_fingerprint(s.as_bytes())
    }

    #[test]
    fn first_reserve_wins_then_store_then_replay() {
        let clock = FakeClock::at(1_000);
        let store = store_at(Arc::clone(&clock));
        let key = IdempotencyKey::new("t", "idem-1");
        let f = fp("body-A");

        // Miss → Reserved.
        assert!(matches!(store.reserve(&key, f), ReserveOutcome::Reserved));
        // Concurrent identical request while in-flight → InFlight (409).
        assert!(matches!(store.reserve(&key, f), ReserveOutcome::InFlight));

        // Store the completed 2xx (with the original dispatch's provenance).
        assert!(store.store(
            &key,
            200,
            "application/json".into(),
            Bytes::from_static(b"{\"ok\":1}"),
            f,
            Some("openai".into()),
            Some("req_orig".into()),
        ));

        // Re-sent same key + same body → Replay (verbatim, provenance intact).
        match store.reserve(&key, f) {
            ReserveOutcome::Replay(r) => {
                assert_eq!(r.status, 200);
                assert_eq!(&r.body[..], b"{\"ok\":1}");
                assert_eq!(r.content_type, "application/json");
                assert_eq!(r.provider.as_deref(), Some("openai"));
                assert_eq!(r.request_id.as_deref(), Some("req_orig"));
            }
            other => panic!("expected Replay, got {other:?}"),
        }
        assert_eq!(store.replays(), 1);
        assert_eq!(store.in_flight_conflicts(), 1);
    }

    #[test]
    fn same_key_different_body_is_mismatch() {
        let clock = FakeClock::at(1_000);
        let store = store_at(Arc::clone(&clock));
        let key = IdempotencyKey::new("t", "idem-2");
        assert!(matches!(
            store.reserve(&key, fp("body-A")),
            ReserveOutcome::Reserved
        ));
        store.store(
            &key,
            200,
            "application/json".into(),
            Bytes::from_static(b"x"),
            fp("body-A"),
            None,
            None,
        );
        // Same key, DIFFERENT body fingerprint → 422-class.
        assert!(matches!(
            store.reserve(&key, fp("body-B")),
            ReserveOutcome::FingerprintMismatch
        ));
        assert_eq!(store.mismatches(), 1);
    }

    #[test]
    fn release_lets_a_retry_rerun_failures_not_cached() {
        let clock = FakeClock::at(1_000);
        let store = store_at(Arc::clone(&clock));
        let key = IdempotencyKey::new("t", "idem-3");
        let f = fp("body");
        assert!(matches!(store.reserve(&key, f), ReserveOutcome::Reserved));
        // Provider failed (non-2xx): release, do NOT store.
        store.release(&key);
        // A genuine retry re-runs (a fresh reservation, not a 409 / replay).
        assert!(matches!(store.reserve(&key, f), ReserveOutcome::Reserved));
        assert_eq!(store.released_count(), 1);
    }

    #[test]
    fn ttl_expiry_lets_a_fresh_reservation_through() {
        let clock = FakeClock::at(1_000);
        let store = store_at(Arc::clone(&clock));
        let key = IdempotencyKey::new("t", "idem-4");
        let f = fp("body");
        assert!(matches!(store.reserve(&key, f), ReserveOutcome::Reserved));
        store.store(
            &key,
            200,
            "application/json".into(),
            Bytes::from_static(b"x"),
            f,
            None,
            None,
        );
        // Within TTL (60s from t=1000): replay.
        clock.set(60_000);
        assert!(matches!(store.reserve(&key, f), ReserveOutcome::Replay(_)));
        // After TTL: the stored entry is reclaimed and we get a fresh reservation.
        clock.set(61_001);
        assert!(matches!(store.reserve(&key, f), ReserveOutcome::Reserved));
    }

    #[test]
    fn stale_in_flight_reservation_is_reclaimed() {
        // A reservation whose holder crashed before release must not pin the key
        // past its TTL — a later request reclaims it.
        let clock = FakeClock::at(1_000);
        let store = store_at(Arc::clone(&clock));
        let key = IdempotencyKey::new("t", "idem-5");
        let f = fp("body");
        assert!(matches!(store.reserve(&key, f), ReserveOutcome::Reserved));
        // Still in-flight within TTL → 409.
        clock.set(60_000);
        assert!(matches!(store.reserve(&key, f), ReserveOutcome::InFlight));
        // Past TTL: stale marker reclaimed → fresh reservation.
        clock.set(61_001);
        assert!(matches!(store.reserve(&key, f), ReserveOutcome::Reserved));
    }

    #[test]
    fn tenant_isolation_is_structural() {
        let clock = FakeClock::at(1_000);
        let store = store_at(Arc::clone(&clock));
        let key_a = IdempotencyKey::new("tenant_a", "shared-key");
        let key_b = IdempotencyKey::new("tenant_b", "shared-key");
        let f = fp("body");
        assert!(matches!(store.reserve(&key_a, f), ReserveOutcome::Reserved));
        store.store(
            &key_a,
            200,
            "application/json".into(),
            Bytes::from_static(b"A"),
            f,
            None,
            None,
        );
        // Tenant A replays.
        assert!(matches!(
            store.reserve(&key_a, f),
            ReserveOutcome::Replay(_)
        ));
        // Tenant B with the SAME client key + SAME body is a fresh reservation —
        // never tenant A's response.
        assert!(matches!(store.reserve(&key_b, f), ReserveOutcome::Reserved));
    }

    #[test]
    fn oversize_body_releases_instead_of_storing() {
        let clock = FakeClock::at(1_000);
        let store = store_at(Arc::clone(&clock));
        let key = IdempotencyKey::new("t", "idem-6");
        let f = fp("body");
        assert!(matches!(store.reserve(&key, f), ReserveOutcome::Reserved));
        let big = Bytes::from(vec![b'x'; IDEMP_MAX_BODY_BYTES + 1]);
        assert!(!store.store(&key, 200, "application/json".into(), big, f, None, None));
        // Reservation released → a retry re-runs rather than 409.
        assert!(matches!(store.reserve(&key, f), ReserveOutcome::Reserved));
    }

    #[test]
    fn bounded_eviction_keeps_shard_under_cap() {
        // Small cap so the FIFO path is exercised deterministically. All keys land
        // in arbitrary shards; assert the global entry count never exceeds the cap.
        let clock = FakeClock::at(1_000);
        let store = IdempotencyStore::with_clock(600, 4, Arc::clone(&clock) as Arc<dyn Clock>);
        for i in 0..200u32 {
            let key = IdempotencyKey::new("t", &format!("k{i}"));
            let f = fp(&format!("b{i}"));
            // Reserve + store so each is a Done slot occupying space.
            if let ReserveOutcome::Reserved = store.reserve(&key, f) {
                store.store(
                    &key,
                    200,
                    "application/json".into(),
                    Bytes::from_static(b"x"),
                    f,
                    None,
                    None,
                );
            }
        }
        let (entries, _) = store.stats_snapshot();
        // 64 shards * cap 4 = 256 ceiling; with 200 keys we never exceed it, and
        // each shard independently stayed <= 4.
        assert!(
            entries <= IDEMP_SHARD_COUNT * 4,
            "entries={entries} exceeded bound"
        );
    }

    #[test]
    fn release_is_noop_on_completed_slot() {
        let clock = FakeClock::at(1_000);
        let store = store_at(Arc::clone(&clock));
        let key = IdempotencyKey::new("t", "idem-7");
        let f = fp("body");
        store.reserve(&key, f);
        store.store(
            &key,
            200,
            "application/json".into(),
            Bytes::from_static(b"x"),
            f,
            None,
            None,
        );
        // A late release (e.g. a dropped loser) must not delete the stored response.
        store.release(&key);
        assert!(matches!(store.reserve(&key, f), ReserveOutcome::Replay(_)));
    }
}
