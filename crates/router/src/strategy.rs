//! Routing strategies and candidate ordering.
//!
//! The proxy classifies residency and produces the *eligible* candidate set
//! (either the sovereign-resident providers or the client's
//! `x-routeplane-provider` chain, or — for G2.2 — the policy's flattened
//! targets). This module turns that set into an ordered attempt list for a
//! given [`RoutingStrategy`], dropping providers whose circuit is currently OPEN.
//!
//! Ordering reads health/latency from the lock-free [`crate::HealthTracker`];
//! nothing here touches the network or takes a lock on the hot path. `Weighted`
//! uses an injectable RNG so its tests are deterministic.
//!
//! G2.2 / ADR-021 §5 adds [`Router::order_candidates_with_specs`]: ordering over
//! per-target `weight`/`cost` OVERRIDES (from a routing-policy config), falling
//! back to [`RouterConfig`] defaults when an override is absent. The router stays
//! network-free, types-free, and lock-free — it learns nothing about configs,
//! requests, or JSON.

use std::collections::HashMap;
use std::sync::atomic::Ordering;

use crate::HealthTracker;

/// How to order eligible providers into an attempt sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RoutingStrategy {
    /// Try providers in the order they were given (the fallback chain order).
    #[default]
    Priority,
    /// Order by per-provider weight, randomized proportional to weight.
    Weighted,
    /// Cheapest provider first, by relative cost.
    Cost,
    /// Lowest observed latency first; untried providers sort optimistically.
    Latency,
    /// Rotate through the surviving (circuit-closed) candidates with a lock-free
    /// atomic cursor, so successive requests spread evenly across the pool
    /// (LiteLLM-style round-robin). The first candidate is chosen by
    /// `cursor mod len`; the rest follow in wrap-around order.
    RoundRobin,
    /// Fewest outstanding (in-flight) requests first, reading the lock-free
    /// per-provider in-flight gauge; ties break on the existing stable input
    /// order (LiteLLM-style least-busy / least-outstanding-requests).
    LeastBusy,
}

impl RoutingStrategy {
    /// Parse from the `x-routeplane-strategy` header value. Case-insensitive;
    /// unknown/empty inputs map to [`RoutingStrategy::Priority`].
    pub fn parse(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "weighted" => RoutingStrategy::Weighted,
            "cost" => RoutingStrategy::Cost,
            "latency" => RoutingStrategy::Latency,
            "round_robin" | "roundrobin" | "round-robin" => RoutingStrategy::RoundRobin,
            "least_busy" | "leastbusy" | "least-busy" => RoutingStrategy::LeastBusy,
            // "priority" and everything else (incl. empty/unknown) -> default.
            _ => RoutingStrategy::Priority,
        }
    }
}

/// Per-provider routing metadata used by the cost/weighted strategies.
#[derive(Debug, Clone, Copy)]
pub struct ProviderRouting {
    /// Relative selection weight for [`RoutingStrategy::Weighted`]. Higher =
    /// more likely to be tried first. Must be > 0 to participate; 0 is treated
    /// as 1 so a provider is never starved entirely.
    pub weight: u32,
    /// Relative cost (cheaper = lower) for [`RoutingStrategy::Cost`]. Unitless;
    /// only the ordering between providers matters.
    pub cost: u32,
}

impl Default for ProviderRouting {
    fn default() -> Self {
        // Neutral defaults for an unknown provider: equal weight, mid cost.
        Self {
            weight: 1,
            cost: 100,
        }
    }
}

/// A candidate plus optional per-request `weight`/`cost` overrides from a routing
/// policy config (ADR-021 §5). `None` overrides fall back to [`RouterConfig`]
/// defaults — so a spec list with all-`None` orders identically to the legacy
/// `&[String]` path (the back-compat guarantee, AC-1).
#[derive(Debug, Clone)]
pub struct CandidateSpec {
    pub name: String,
    pub weight: Option<u32>,
    pub cost: Option<u32>,
}

impl CandidateSpec {
    /// A spec with no overrides (legacy behavior).
    pub fn plain(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            weight: None,
            cost: None,
        }
    }
}

/// Per-provider routing config (weight + relative cost).
///
/// TODO(ADR-004): these defaults are hard-coded for the four known providers.
/// Request-scoped weights/costs now arrive via policy configs (ADR-021 §5);
/// the *defaults* still move to control-plane config later.
#[derive(Debug, Clone, Default)]
pub struct RouterConfig {
    providers: HashMap<String, ProviderRouting>,
}

impl RouterConfig {
    /// Build with sane defaults for the four known providers. Relative cost is a
    /// coarse ranking of typical blended $/token (cheaper = lower), not a price.
    pub fn with_defaults() -> Self {
        let mut providers = HashMap::new();
        providers.insert(
            "gemini".into(),
            ProviderRouting {
                weight: 1,
                cost: 30,
            },
        );
        providers.insert(
            "openai".into(),
            ProviderRouting {
                weight: 1,
                cost: 50,
            },
        );
        providers.insert(
            "azure_openai".into(),
            ProviderRouting {
                weight: 1,
                cost: 55,
            },
        );
        providers.insert(
            "anthropic".into(),
            ProviderRouting {
                weight: 1,
                cost: 80,
            },
        );
        Self { providers }
    }

    /// Set (or override) the routing metadata for one provider.
    pub fn set(&mut self, provider: impl Into<String>, routing: ProviderRouting) {
        self.providers.insert(provider.into(), routing);
    }

    /// Routing metadata for `provider`, falling back to neutral defaults for
    /// providers not present in the config.
    pub fn get(&self, provider: &str) -> ProviderRouting {
        self.providers.get(provider).copied().unwrap_or_default()
    }
}

/// Deterministic, seedable RNG (SplitMix64). Used only by `Weighted` ordering;
/// injectable so tests pin a seed and assert an exact order — no flake, no
/// `rand` dependency.
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Seed the RNG. Any seed is valid (including 0).
    pub fn seeded(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Next pseudo-random `u64` (SplitMix64).
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform value in `[0, bound)`. `bound` must be > 0.
    fn next_below(&mut self, bound: u64) -> u64 {
        self.next_u64() % bound
    }
}

/// Stateless ordering policy. Holds the [`RouterConfig`]; reads live health and
/// latency from a borrowed [`HealthTracker`] per call. Cheap to construct and
/// `Send + Sync`, so it lives in `AppState` behind the shared `Arc`.
///
/// The only mutable state is `rr_cursor`, a single lock-free `AtomicUsize`
/// rotation cursor used exclusively by [`RoutingStrategy::RoundRobin`]. It is
/// advanced (one `fetch_add`) only when round-robin is selected, so every other
/// strategy is byte-identical to the pre-RoundRobin router.
pub struct Router {
    config: RouterConfig,
    /// Monotonic round-robin rotation cursor. Wraps naturally on overflow; only
    /// `cursor % live.len()` is observed, so wrap-around is harmless.
    rr_cursor: std::sync::atomic::AtomicUsize,
}

impl Router {
    /// Construct with the given routing config.
    pub fn new(config: RouterConfig) -> Self {
        Self {
            config,
            rr_cursor: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Construct with the built-in defaults for the four known providers.
    pub fn with_defaults() -> Self {
        Self::new(RouterConfig::with_defaults())
    }

    /// Read-only view of the routing config (mainly for tests/diagnostics).
    pub fn config(&self) -> &RouterConfig {
        &self.config
    }

    /// Order the eligible candidates into an attempt sequence for `strategy`,
    /// using process-default RNG seeding for `Weighted`.
    pub fn order_candidates(
        &self,
        eligible: &[String],
        strategy: RoutingStrategy,
        health: &HealthTracker,
    ) -> Vec<String> {
        let mut rng = Rng::seeded(default_weighted_seed());
        self.order_candidates_with_rng(eligible, strategy, health, &mut rng)
    }

    /// Like [`Self::order_candidates`] but with an injected RNG, so `Weighted`
    /// ordering is deterministic under test.
    pub fn order_candidates_with_rng(
        &self,
        eligible: &[String],
        strategy: RoutingStrategy,
        health: &HealthTracker,
        rng: &mut Rng,
    ) -> Vec<String> {
        // 1. Drop circuit-OPEN providers up front (preserving input order).
        let mut live: Vec<String> = eligible
            .iter()
            .filter(|name| health.is_available(name))
            .cloned()
            .collect();

        // 2. Order the survivors by strategy.
        match strategy {
            RoutingStrategy::Priority => {}
            RoutingStrategy::Cost => {
                live.sort_by_key(|name| self.config.get(name).cost);
            }
            RoutingStrategy::Latency => {
                live.sort_by_key(|name| health.latency_ms(name).map(|ms| ms + 1).unwrap_or(0));
            }
            RoutingStrategy::Weighted => {
                live = self.weighted_order(live, rng);
            }
            RoutingStrategy::RoundRobin => {
                live = self.round_robin_order(live);
            }
            RoutingStrategy::LeastBusy => {
                // Stable sort by ascending in-flight count; ties keep input order.
                live.sort_by_key(|name| health.in_flight(name));
            }
        }

        live
    }

    /// G2.2 / ADR-021 §5: order [`CandidateSpec`]s, honoring per-target
    /// `weight`/`cost` overrides and falling back to [`RouterConfig`] defaults.
    /// Process-default RNG seeding for `Weighted`.
    pub fn order_candidates_with_specs(
        &self,
        specs: &[CandidateSpec],
        strategy: RoutingStrategy,
        health: &HealthTracker,
    ) -> Vec<String> {
        let mut rng = Rng::seeded(default_weighted_seed());
        self.order_candidates_with_specs_rng(specs, strategy, health, &mut rng)
    }

    /// Like [`Self::order_candidates_with_specs`] but with an injected RNG.
    pub fn order_candidates_with_specs_rng(
        &self,
        specs: &[CandidateSpec],
        strategy: RoutingStrategy,
        health: &HealthTracker,
        rng: &mut Rng,
    ) -> Vec<String> {
        // 1. Drop circuit-OPEN providers, preserving input (policy) order.
        let mut live: Vec<&CandidateSpec> = specs
            .iter()
            .filter(|s| health.is_available(&s.name))
            .collect();

        // 2. Order by strategy, reading overrides first, defaults second.
        match strategy {
            RoutingStrategy::Priority => {}
            RoutingStrategy::Cost => {
                live.sort_by_key(|s| s.cost.unwrap_or_else(|| self.config.get(&s.name).cost));
            }
            RoutingStrategy::Latency => {
                live.sort_by_key(|s| health.latency_ms(&s.name).map(|ms| ms + 1).unwrap_or(0));
            }
            RoutingStrategy::Weighted => {
                live = self.weighted_order_specs(live, rng);
            }
            RoutingStrategy::RoundRobin => {
                live = self.round_robin_order(live);
            }
            RoutingStrategy::LeastBusy => {
                live.sort_by_key(|s| health.in_flight(&s.name));
            }
        }

        live.into_iter().map(|s| s.name.clone()).collect()
    }

    /// Round-robin rotation over the surviving (circuit-closed) candidates.
    /// Reads one `fetch_add` from the lock-free [`Self::rr_cursor`] and rotates
    /// the live list left so the candidate at `cursor % len` leads; the rest
    /// follow in wrap-around order. The cursor advances ONLY here (round-robin),
    /// so the other strategies never touch it. Rotating among the SURVIVORS (the
    /// OPEN-circuit drop already happened) means a tripped provider is skipped
    /// without stalling the rotation. Generic over the element type so the
    /// name-list and spec-list paths share one implementation.
    fn round_robin_order<T>(&self, mut live: Vec<T>) -> Vec<T> {
        let len = live.len();
        if len <= 1 {
            return live;
        }
        // Relaxed is sufficient: this counter only spreads load and observes no
        // other memory; the exact interleaving under contention is irrelevant.
        let start = self.rr_cursor.fetch_add(1, Ordering::Relaxed) % len;
        live.rotate_left(start);
        live
    }

    /// Weighted random ordering over names (legacy path).
    fn weighted_order(&self, mut pool: Vec<String>, rng: &mut Rng) -> Vec<String> {
        let mut ordered = Vec::with_capacity(pool.len());
        while !pool.is_empty() {
            let weights: Vec<u64> = pool
                .iter()
                .map(|name| self.config.get(name).weight.max(1) as u64)
                .collect();
            let total: u64 = weights.iter().sum();
            let mut pick = rng.next_below(total.max(1));
            let mut chosen = 0;
            for (i, w) in weights.iter().enumerate() {
                if pick < *w {
                    chosen = i;
                    break;
                }
                pick -= *w;
            }
            ordered.push(pool.remove(chosen));
        }
        ordered
    }

    /// Weighted random ordering over specs, honoring weight overrides.
    fn weighted_order_specs<'a>(
        &self,
        mut pool: Vec<&'a CandidateSpec>,
        rng: &mut Rng,
    ) -> Vec<&'a CandidateSpec> {
        let mut ordered = Vec::with_capacity(pool.len());
        while !pool.is_empty() {
            let weights: Vec<u64> = pool
                .iter()
                .map(|s| {
                    s.weight
                        .unwrap_or_else(|| self.config.get(&s.name).weight)
                        .max(1) as u64
                })
                .collect();
            let total: u64 = weights.iter().sum();
            let mut pick = rng.next_below(total.max(1));
            let mut chosen = 0;
            for (i, w) in weights.iter().enumerate() {
                if pick < *w {
                    chosen = i;
                    break;
                }
                pick -= *w;
            }
            ordered.push(pool.remove(chosen));
        }
        ordered
    }
}

impl Default for Router {
    fn default() -> Self {
        Self::with_defaults()
    }
}

/// Seed for the process-default weighted RNG: wall-clock nanos, so production
/// traffic spreads across weighted providers instead of always picking the same
/// order. Tests use the `_with_rng` variants with a fixed seed.
fn default_weighted_seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tracker() -> HealthTracker {
        HealthTracker::new(["openai", "anthropic", "gemini", "azure_openai"])
    }

    fn names(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    fn specs(v: &[&str]) -> Vec<CandidateSpec> {
        v.iter().map(|s| CandidateSpec::plain(*s)).collect()
    }

    #[test]
    fn parse_is_case_insensitive_and_defaults_to_priority() {
        assert_eq!(
            RoutingStrategy::parse("Weighted"),
            RoutingStrategy::Weighted
        );
        assert_eq!(RoutingStrategy::parse("  COST "), RoutingStrategy::Cost);
        assert_eq!(RoutingStrategy::parse("latency"), RoutingStrategy::Latency);
        assert_eq!(
            RoutingStrategy::parse("priority"),
            RoutingStrategy::Priority
        );
        assert_eq!(
            RoutingStrategy::parse("nonsense"),
            RoutingStrategy::Priority
        );
        assert_eq!(RoutingStrategy::parse(""), RoutingStrategy::Priority);
    }

    #[test]
    fn parse_round_robin_and_least_busy_names() {
        for s in [
            "round_robin",
            "roundrobin",
            "round-robin",
            "ROUND_ROBIN",
            " Round-Robin ",
        ] {
            assert_eq!(
                RoutingStrategy::parse(s),
                RoutingStrategy::RoundRobin,
                "failed on {s:?}"
            );
        }
        for s in [
            "least_busy",
            "leastbusy",
            "least-busy",
            "LEAST_BUSY",
            " Least-Busy ",
        ] {
            assert_eq!(
                RoutingStrategy::parse(s),
                RoutingStrategy::LeastBusy,
                "failed on {s:?}"
            );
        }
    }

    #[test]
    fn round_robin_rotates_across_eligible_candidates() {
        let r = Router::with_defaults();
        let h = tracker();
        let eligible = names(&["openai", "anthropic", "gemini"]);
        // Cursor starts at 0: first call leads with index 0, then 1, then 2, wrap.
        let o0 = r.order_candidates(&eligible, RoutingStrategy::RoundRobin, &h);
        let o1 = r.order_candidates(&eligible, RoutingStrategy::RoundRobin, &h);
        let o2 = r.order_candidates(&eligible, RoutingStrategy::RoundRobin, &h);
        let o3 = r.order_candidates(&eligible, RoutingStrategy::RoundRobin, &h);
        assert_eq!(o0, names(&["openai", "anthropic", "gemini"]));
        assert_eq!(o1, names(&["anthropic", "gemini", "openai"]));
        assert_eq!(o2, names(&["gemini", "openai", "anthropic"]));
        // Wrap-around: cursor=3, 3 % 3 == 0 → identical to the first ordering.
        assert_eq!(o3, o0);
        // Every ordering is a permutation of the full eligible set.
        for o in [&o0, &o1, &o2, &o3] {
            let mut s = o.clone();
            s.sort();
            assert_eq!(s, names(&["anthropic", "gemini", "openai"]));
        }
    }

    #[test]
    fn round_robin_skips_circuit_open_provider() {
        let r = Router::with_defaults();
        let h = tracker();
        for _ in 0..5 {
            h.record_failure("anthropic"); // trip anthropic OPEN
        }
        let eligible = names(&["openai", "anthropic", "gemini"]);
        // Survivors are [openai, gemini] (len 2): rotation cycles between them and
        // never includes the OPEN provider.
        let o0 = r.order_candidates(&eligible, RoutingStrategy::RoundRobin, &h);
        let o1 = r.order_candidates(&eligible, RoutingStrategy::RoundRobin, &h);
        assert_eq!(o0, names(&["openai", "gemini"]));
        assert_eq!(o1, names(&["gemini", "openai"]));
        assert!(!o0.contains(&"anthropic".to_string()));
        assert!(!o1.contains(&"anthropic".to_string()));
    }

    #[test]
    fn round_robin_single_candidate_is_stable() {
        let r = Router::with_defaults();
        let h = tracker();
        let eligible = names(&["openai"]);
        for _ in 0..3 {
            assert_eq!(
                r.order_candidates(&eligible, RoutingStrategy::RoundRobin, &h),
                names(&["openai"])
            );
        }
    }

    #[test]
    fn least_busy_orders_fewest_in_flight_first() {
        let r = Router::with_defaults();
        let h = tracker();
        // openai: 2 in flight, anthropic: 0, gemini: 1.
        let _g1 = h.enter_in_flight("openai").unwrap();
        let _g2 = h.enter_in_flight("openai").unwrap();
        let _g3 = h.enter_in_flight("gemini").unwrap();
        let eligible = names(&["openai", "anthropic", "gemini"]);
        let out = r.order_candidates(&eligible, RoutingStrategy::LeastBusy, &h);
        assert_eq!(out, names(&["anthropic", "gemini", "openai"]));
    }

    #[test]
    fn least_busy_tie_breaks_on_input_order() {
        let r = Router::with_defaults();
        let h = tracker();
        // All zero in flight → stable sort preserves the input order.
        let eligible = names(&["gemini", "openai", "anthropic"]);
        let out = r.order_candidates(&eligible, RoutingStrategy::LeastBusy, &h);
        assert_eq!(out, eligible);
    }

    #[test]
    fn in_flight_guard_balances_on_drop() {
        let h = tracker();
        assert_eq!(h.in_flight("openai"), 0);
        {
            let _g = h.enter_in_flight("openai").unwrap();
            assert_eq!(h.in_flight("openai"), 1);
            let _g2 = h.enter_in_flight("openai").unwrap();
            assert_eq!(h.in_flight("openai"), 2);
        } // both guards drop here
        assert_eq!(h.in_flight("openai"), 0);
    }

    #[test]
    fn in_flight_unknown_provider_sorts_last() {
        let r = Router::with_defaults();
        let h = HealthTracker::new(["openai"]);
        // "ghost" has no gauge → in_flight == u64::MAX → sorts last under LeastBusy.
        let eligible = names(&["ghost", "openai"]);
        let out = r.order_candidates(&eligible, RoutingStrategy::LeastBusy, &h);
        assert_eq!(out, names(&["openai", "ghost"]));
        assert_eq!(h.in_flight("ghost"), u64::MAX);
    }

    #[test]
    fn priority_preserves_input_order() {
        let r = Router::with_defaults();
        let h = tracker();
        let eligible = names(&["anthropic", "openai", "gemini"]);
        let out = r.order_candidates(&eligible, RoutingStrategy::Priority, &h);
        assert_eq!(out, eligible);
    }

    #[test]
    fn cost_orders_cheapest_first() {
        let r = Router::with_defaults();
        let h = tracker();
        let eligible = names(&["anthropic", "openai", "gemini", "azure_openai"]);
        let out = r.order_candidates(&eligible, RoutingStrategy::Cost, &h);
        assert_eq!(
            out,
            names(&["gemini", "openai", "azure_openai", "anthropic"])
        );
    }

    #[test]
    fn latency_orders_lowest_first_untried_lead() {
        let r = Router::with_defaults();
        let h = tracker();
        h.record_latency("openai", 300);
        h.record_latency("anthropic", 100);
        let eligible = names(&["openai", "anthropic", "gemini"]);
        let out = r.order_candidates(&eligible, RoutingStrategy::Latency, &h);
        assert_eq!(out, names(&["gemini", "anthropic", "openai"]));
    }

    #[test]
    fn latency_all_untried_keeps_input_order() {
        let r = Router::with_defaults();
        let h = tracker();
        let eligible = names(&["openai", "anthropic", "gemini"]);
        let out = r.order_candidates(&eligible, RoutingStrategy::Latency, &h);
        assert_eq!(out, eligible);
    }

    #[test]
    fn drops_circuit_open_providers() {
        let r = Router::with_defaults();
        let h = tracker();
        for _ in 0..5 {
            h.record_failure("openai");
        }
        let eligible = names(&["openai", "anthropic", "gemini"]);
        let out = r.order_candidates(&eligible, RoutingStrategy::Priority, &h);
        assert_eq!(out, names(&["anthropic", "gemini"]));
    }

    #[test]
    fn weighted_is_deterministic_for_a_fixed_seed() {
        let r = Router::with_defaults();
        let h = tracker();
        let eligible = names(&["openai", "anthropic", "gemini"]);

        let mut rng_a = Rng::seeded(42);
        let out_a =
            r.order_candidates_with_rng(&eligible, RoutingStrategy::Weighted, &h, &mut rng_a);
        let mut rng_b = Rng::seeded(42);
        let out_b =
            r.order_candidates_with_rng(&eligible, RoutingStrategy::Weighted, &h, &mut rng_b);

        assert_eq!(out_a, out_b);
        assert_eq!(out_a.len(), 3);
        let mut sorted = out_a.clone();
        sorted.sort();
        assert_eq!(sorted, names(&["anthropic", "gemini", "openai"]));
    }

    #[test]
    fn weighted_favors_higher_weight_first() {
        let mut cfg = RouterConfig::with_defaults();
        cfg.set(
            "gemini",
            ProviderRouting {
                weight: 1000,
                cost: 30,
            },
        );
        cfg.set(
            "openai",
            ProviderRouting {
                weight: 1,
                cost: 50,
            },
        );
        cfg.set(
            "anthropic",
            ProviderRouting {
                weight: 1,
                cost: 80,
            },
        );
        let r = Router::new(cfg);
        let h = tracker();
        let eligible = names(&["openai", "anthropic", "gemini"]);

        let mut first_is_gemini = 0;
        for seed in 0..200u64 {
            let mut rng = Rng::seeded(seed);
            let out =
                r.order_candidates_with_rng(&eligible, RoutingStrategy::Weighted, &h, &mut rng);
            if out[0] == "gemini" {
                first_is_gemini += 1;
            }
        }
        assert!(first_is_gemini > 190, "gemini led {first_is_gemini}/200");
    }

    #[test]
    fn weighted_drops_open_circuits_before_ordering() {
        let r = Router::with_defaults();
        let h = tracker();
        for _ in 0..5 {
            h.record_failure("openai");
        }
        let eligible = names(&["openai", "anthropic", "gemini"]);
        let mut rng = Rng::seeded(7);
        let out = r.order_candidates_with_rng(&eligible, RoutingStrategy::Weighted, &h, &mut rng);
        assert!(!out.contains(&"openai".to_string()));
        assert_eq!(out.len(), 2);
    }

    // --- CandidateSpec API (ADR-021 §5) ---------------------------------------

    #[test]
    fn specs_all_none_match_legacy_priority_and_cost() {
        let r = Router::with_defaults();
        let h = tracker();
        let eligible = names(&["anthropic", "openai", "gemini"]);
        let s = specs(&["anthropic", "openai", "gemini"]);
        // Priority: identical input order.
        assert_eq!(
            r.order_candidates_with_specs(&s, RoutingStrategy::Priority, &h),
            r.order_candidates(&eligible, RoutingStrategy::Priority, &h),
        );
        // Cost: defaults match the legacy cost ordering.
        assert_eq!(
            r.order_candidates_with_specs(&s, RoutingStrategy::Cost, &h),
            r.order_candidates(&eligible, RoutingStrategy::Cost, &h),
        );
    }

    #[test]
    fn cost_override_beats_default() {
        let r = Router::with_defaults();
        let h = tracker();
        // anthropic defaults to the most expensive (80), but a per-request
        // override of 1 makes it the cheapest → first.
        let s = vec![
            CandidateSpec {
                name: "openai".into(),
                weight: None,
                cost: None,
            },
            CandidateSpec {
                name: "anthropic".into(),
                weight: None,
                cost: Some(1),
            },
        ];
        let out = r.order_candidates_with_specs(&s, RoutingStrategy::Cost, &h);
        assert_eq!(out, names(&["anthropic", "openai"]));
    }

    #[test]
    fn weight_override_front_loads_in_weighted() {
        let r = Router::with_defaults();
        let h = tracker();
        let s = vec![
            CandidateSpec {
                name: "openai".into(),
                weight: Some(1),
                cost: None,
            },
            CandidateSpec {
                name: "anthropic".into(),
                weight: Some(1000),
                cost: None,
            },
        ];
        let mut led = 0;
        for seed in 0..200u64 {
            let mut rng = Rng::seeded(seed);
            let out =
                r.order_candidates_with_specs_rng(&s, RoutingStrategy::Weighted, &h, &mut rng);
            if out[0] == "anthropic" {
                led += 1;
            }
        }
        assert!(led > 190, "anthropic led {led}/200");
    }
}
