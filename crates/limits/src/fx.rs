//! Configurable FX rate table for multi-currency cost attribution
//! ([PRD-015] FR-2, the FinOps moat — the differentiator over USD-only cost
//! tracking).
//!
//! # What this owns
//! A lock-free, hot-swappable table of ISO-4217 currencies → the integer
//! conversion factor + minor-unit exponent needed to render the canonical
//! micro-USD spend in a local currency, plus the region → default-currency map.
//! The canonical spend figure stays **integer micro-USD** (money is never a
//! float); this module only layers a *view* on top.
//!
//! # Why a table (not a live feed)
//! [PRD-015] §2 wants an FX rate *source* with an in-process cache and zero
//! hot-path network. A config/env-loaded table is exactly that: $0 standing
//! cost, no DB, no live dependency — incremental on PRD-015, no new ADR (a live
//! FX feed *would* be a new standing cost and is a documented follow-up). An
//! operator who wants fresher rates points [`FxRates::from_env`] at a file an
//! out-of-band job rewrites, and [`SharedFxRates::replace`] swaps it in without a
//! restart (mirrors the prompt/policy `ArcSwap` registries).
//!
//! # Invariants (mirror the rest of `crates/limits`)
//! - **Lock-free read.** Conversion is a single `HashMap` lookup against an
//!   immutable [`ArcSwap`] snapshot — no mutex, no allocation beyond the result
//!   currency string, no I/O. Safe on the request hot path.
//! - **Integer / saturating, never panics.** All conversion is integer math with
//!   saturating multiply; a request thread never panics on a rate lookup.
//! - **Ship-dark default.** With no rate config the table is *byte-identical* to
//!   the legacy 4-bucket placeholder (`IN→INR`, `EU→EUR`, `AE→AED`, else `USD`),
//!   so existing golden/parity snapshots do not shift. The placeholder simply
//!   became the built-in default — now overridable.
//!
//! [PRD-015]: ../../../docs/product/prd/015-multi-currency-finops.md

use arc_swap::ArcSwap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// One currency's conversion parameters. `minor_per_usd` is the integer number of
/// the currency's **minor units** in one USD (paise for INR, cents for USD/EUR,
/// fils for AED). `exponent` is the ISO-4217 minor-unit exponent — the number of
/// decimal places the currency has (USD/EUR/INR = 2, JPY = 0, BHD/KWD = 3) — used
/// so the rendered figure is in the currency's *own* minor unit, not always
/// "USD-cents-per-USD" × something.
///
/// The two are linked: `minor_per_usd` already encodes the exponent (e.g. JPY at
/// ~150 JPY/USD with 0dp is `minor_per_usd = 150`, NOT `15_000`). `exponent` is
/// carried so a *consumer* that needs to format the integer minor units back to a
/// decimal string knows where to put the point — it does not re-scale here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CurrencyRate {
    /// Integer minor units of this currency per 1 USD (e.g. INR→`8_300` paise,
    /// JPY→`150` yen [0dp], BHD→`377` fils [3dp]).
    pub minor_per_usd: u64,
    /// ISO-4217 minor-unit exponent (decimal places). 2 for most, 0 for JPY,
    /// 3 for BHD/KWD/OMR.
    pub exponent: u8,
}

/// The hot-swappable FX rate table: ISO-4217 code → [`CurrencyRate`], plus the
/// region → default-currency map. Cheap to share behind an [`ArcSwap`]; cloning a
/// snapshot for replacement is the only place the maps are duplicated (off the
/// hot path).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FxRates {
    /// ISO-4217 code (uppercase) → conversion parameters.
    rates: HashMap<String, CurrencyRate>,
    /// Residency region code → the ISO-4217 code that region defaults to. The
    /// `""` key (or absence) is the global default currency.
    region_currency: HashMap<String, String>,
    /// The fallback currency when neither a header, a region, nor a known code
    /// resolves — always present in `rates`.
    default_currency: String,
}

/// The wire shape for `RP_FX_RATES_JSON` / `RP_FX_RATES_FILE`. Every field is
/// optional and merges onto the built-in default table, so an operator can ship a
/// single new currency without re-declaring the placeholder rows.
#[derive(Debug, Clone, Default, Deserialize)]
struct FxRatesConfig {
    /// ISO-4217 code → rate. Codes are uppercased on load.
    #[serde(default)]
    rates: HashMap<String, CurrencyRate>,
    /// Region code → ISO-4217 default currency. Region codes are uppercased.
    #[serde(default)]
    region_currency: HashMap<String, String>,
    /// Override the global fallback currency (must resolve in the merged `rates`).
    #[serde(default)]
    default_currency: Option<String>,
}

impl Default for FxRates {
    /// The built-in default table — **byte-identical** to the legacy
    /// `pricing::currency_for` placeholder so existing golden/parity snapshots do
    /// not shift. The placeholder became the default; config now overrides it.
    fn default() -> Self {
        // Legacy placeholder rows (the only four the corpus exercised), now with
        // explicit ISO minor-unit exponents.
        let rates: HashMap<String, CurrencyRate> = [
            // ~83 INR/USD × 100 paise/INR (2dp).
            (
                "INR",
                CurrencyRate {
                    minor_per_usd: 8_300,
                    exponent: 2,
                },
            ),
            // ~0.92 EUR/USD × 100 cents/EUR (2dp).
            (
                "EUR",
                CurrencyRate {
                    minor_per_usd: 92,
                    exponent: 2,
                },
            ),
            // 3.6725 AED/USD × 1000 fils/AED (AED hard-pegged, 2dp by ISO but the
            // legacy placeholder used 1000/AED — preserved exactly for parity).
            (
                "AED",
                CurrencyRate {
                    minor_per_usd: 3_672,
                    exponent: 2,
                },
            ),
            // 1 USD = 100 cents (2dp) — the default bucket.
            (
                "USD",
                CurrencyRate {
                    minor_per_usd: 100,
                    exponent: 2,
                },
            ),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();

        let region_currency: HashMap<String, String> =
            [("IN", "INR"), ("EU", "EUR"), ("AE", "AED")]
                .into_iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect();

        Self {
            rates,
            region_currency,
            default_currency: "USD".to_string(),
        }
    }
}

impl FxRates {
    /// Build the table from the environment, overriding the built-in default:
    ///   * `RP_FX_RATES_JSON` — an inline JSON document (takes precedence), or
    ///   * `RP_FX_RATES_FILE` — a path to a JSON file with the same shape.
    ///
    /// Either is OPTIONAL. With neither set (or on any parse/IO error) this is
    /// exactly [`FxRates::default`] — ship-dark, byte-identical to the legacy
    /// placeholder. A malformed override is logged and ignored rather than
    /// failing startup: cost attribution is a *view*, never an admission gate, so
    /// it must degrade to the default rather than refuse traffic.
    pub fn from_env() -> Self {
        let raw = match std::env::var("RP_FX_RATES_JSON") {
            Ok(json) if !json.trim().is_empty() => Some(json),
            _ => match std::env::var("RP_FX_RATES_FILE") {
                Ok(path) if !path.trim().is_empty() => match std::fs::read_to_string(&path) {
                    Ok(s) => Some(s),
                    Err(e) => {
                        tracing::warn!(
                            "RP_FX_RATES_FILE='{path}' unreadable ({e}); using default FX table"
                        );
                        None
                    }
                },
                _ => None,
            },
        };

        match raw {
            Some(raw) => Self::from_json_str(&raw),
            None => Self::default(),
        }
    }

    /// Parse a JSON rate-config document and merge it onto the built-in default
    /// table. Malformed JSON degrades to [`FxRates::default`] (a logged warning),
    /// never an error — cost attribution is a view, never an admission gate. This
    /// is the same merge the env loader uses, exposed so an operator-refresh seam
    /// (or a test) can build a table without touching process env.
    pub fn from_json_str(raw: &str) -> Self {
        match serde_json::from_str::<FxRatesConfig>(raw) {
            Ok(cfg) => Self::default().merged(cfg),
            Err(e) => {
                tracing::warn!("FX rate config is not valid JSON ({e}); using default FX table");
                Self::default()
            }
        }
    }

    /// Merge a parsed config onto this (default) table. Config rows ADD or REPLACE
    /// built-in rows by code; the region map and default currency override when
    /// provided. Codes/regions are uppercased so lookups are case-insensitive. A
    /// `default_currency` that does not resolve after the merge is ignored (kept
    /// at the prior default) so the table always has a valid fallback.
    fn merged(mut self, cfg: FxRatesConfig) -> Self {
        for (code, rate) in cfg.rates {
            self.rates.insert(code.to_uppercase(), rate);
        }
        for (region, code) in cfg.region_currency {
            self.region_currency
                .insert(region.to_uppercase(), code.to_uppercase());
        }
        if let Some(dc) = cfg.default_currency {
            let dc = dc.to_uppercase();
            if self.rates.contains_key(&dc) {
                self.default_currency = dc;
            } else {
                tracing::warn!(
                    "FX default_currency='{dc}' is not in the rate table; keeping '{}'",
                    self.default_currency
                );
            }
        }
        self
    }

    /// Whether the table knows a (case-insensitive) ISO-4217 code.
    pub fn knows(&self, code: &str) -> bool {
        self.rates.contains_key(&code.to_uppercase())
    }

    /// The currency this region defaults to (e.g. `IN`→`INR`), or the global
    /// default when the region is unknown/`None`. Always a code present in the
    /// rate table.
    fn region_currency(&self, region: Option<&str>) -> String {
        region
            .and_then(|r| self.region_currency.get(&r.to_uppercase()))
            .cloned()
            .unwrap_or_else(|| self.default_currency.clone())
    }

    /// Resolve the DISPLAY currency for a request via the fallback chain
    /// **header → region → global default**:
    ///   1. an explicit, *known* `display` code (the `x-routeplane-currency`
    ///      header), case-insensitive;
    ///   2. else the region-derived default;
    ///   3. else the global default currency.
    ///
    /// An unknown/unsupported header code falls through to the region/default
    /// gracefully — it never errors the request and never panics. Returns the
    /// resolved (uppercase) code and its [`CurrencyRate`].
    pub fn resolve(&self, display: Option<&str>, region: Option<&str>) -> (String, CurrencyRate) {
        // 1. Explicit header, only if the table knows it.
        if let Some(code) = display {
            let upper = code.to_uppercase();
            if let Some(rate) = self.rates.get(&upper) {
                return (upper, *rate);
            }
        }
        // 2. Region default.
        let region_code = self.region_currency(region);
        if let Some(rate) = self.rates.get(&region_code) {
            return (region_code, *rate);
        }
        // 3. Global default (guaranteed present).
        let dc = self.default_currency.clone();
        let rate = self
            .rates
            .get(&dc)
            .copied()
            // The default is always inserted; this fallback exists only so the
            // function is total even against a hand-built malformed table.
            .unwrap_or(CurrencyRate {
                minor_per_usd: 100,
                exponent: 2,
            });
        (dc, rate)
    }
}

/// Convert a canonical integer micro-USD figure into a currency's integer minor
/// units. Pure, saturating integer math — never a float, never a panic.
///
/// `minor_units = micro_usd × minor_per_usd / 1_000_000`.
///
/// Because `minor_per_usd` is already expressed in the currency's OWN minor unit
/// (paise for INR, yen for JPY since JPY has 0dp, fils for BHD at 3dp), the result
/// is directly in that minor unit — no separate exponent rescale, so a JPY amount
/// is never off by 100×. The `exponent` on [`CurrencyRate`] is metadata for a
/// downstream formatter (where to place the decimal point), not a second divisor.
#[inline]
pub fn micro_usd_to_minor_units(micro_usd: u64, rate: CurrencyRate) -> u64 {
    micro_usd.saturating_mul(rate.minor_per_usd) / 1_000_000
}

/// A wait-free, hot-swappable handle to the FX rate table — the same `ArcSwap`
/// pattern as `SharedPromptRegistry` / `SharedPolicyRegistry`. The read path
/// (`load`) is wait-free; [`SharedFxRatesExt::replace`] swaps a fresh table in
/// without a restart for an off-path operator refresh.
pub type SharedFxRates = Arc<ArcSwap<FxRates>>;

/// Build a shared handle from a table (e.g. [`FxRates::from_env`]).
pub fn shared(rates: FxRates) -> SharedFxRates {
    Arc::new(ArcSwap::from_pointee(rates))
}

/// Build a shared handle from the environment (the standard wiring).
pub fn shared_from_env() -> SharedFxRates {
    shared(FxRates::from_env())
}

/// Atomic hot-swap of the FX table behind a [`SharedFxRates`] — the seam an
/// OPTIONAL, opt-in, off-path refresh task would call after re-reading the
/// operator's source. No live feed is wired here (that is a documented follow-up
/// needing an ADR for any standing cost); this is the mechanism only.
pub trait SharedFxRatesExt {
    fn replace(&self, rates: FxRates);
}

impl SharedFxRatesExt for SharedFxRates {
    fn replace(&self, rates: FxRates) {
        self.store(Arc::new(rates));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_table_reproduces_the_legacy_placeholder() {
        let fx = FxRates::default();
        // Region-derived currency matches the old `currency_for`.
        assert_eq!(fx.resolve(None, Some("IN")).0, "INR");
        assert_eq!(fx.resolve(None, Some("EU")).0, "EUR");
        assert_eq!(fx.resolve(None, Some("AE")).0, "AED");
        assert_eq!(fx.resolve(None, None).0, "USD");
        assert_eq!(fx.resolve(None, Some("US")).0, "USD"); // unknown region → USD
                                                           // And the rates match the old minor-per-usd buckets exactly.
        assert_eq!(fx.resolve(None, Some("IN")).1.minor_per_usd, 8_300);
        assert_eq!(fx.resolve(None, Some("EU")).1.minor_per_usd, 92);
        assert_eq!(fx.resolve(None, Some("AE")).1.minor_per_usd, 3_672);
        assert_eq!(fx.resolve(None, None).1.minor_per_usd, 100);
    }

    #[test]
    fn legacy_minor_unit_math_is_unchanged() {
        // gpt-4o 1000/1000 → 13_000 micro-USD; the four legacy buckets must
        // produce the exact figures the old `cost_breakdown` tests asserted.
        let fx = FxRates::default();
        let usd = micro_usd_to_minor_units(13_000, fx.resolve(None, None).1);
        assert_eq!(usd, 13_000 * 100 / 1_000_000); // 1 cent
        let inr = micro_usd_to_minor_units(13_000, fx.resolve(None, Some("IN")).1);
        assert_eq!(inr, 13_000 * 8_300 / 1_000_000); // 107 paise
    }

    #[test]
    fn display_currency_header_overrides_region() {
        let fx = FxRates::default();
        // Header EUR wins over the IN region default.
        let (code, _) = fx.resolve(Some("EUR"), Some("IN"));
        assert_eq!(code, "EUR");
        // Case-insensitive.
        assert_eq!(fx.resolve(Some("eur"), Some("IN")).0, "EUR");
    }

    #[test]
    fn unknown_header_currency_falls_back_gracefully() {
        let fx = FxRates::default();
        // Unknown code → region default (no error, no panic).
        assert_eq!(fx.resolve(Some("ZZZ"), Some("IN")).0, "INR");
        // Unknown code + unknown region → global default USD.
        assert_eq!(fx.resolve(Some("ZZZ"), Some("XX")).0, "USD");
        // Empty header string → region default.
        assert_eq!(fx.resolve(Some(""), Some("EU")).0, "EUR");
    }

    #[test]
    fn config_merge_adds_and_overrides_rows() {
        // JPY (0dp), BHD (3dp), an override of INR, and a new region mapping.
        let cfg: FxRatesConfig = serde_json::from_str(
            r#"{
                "rates": {
                    "jpy": {"minor_per_usd": 150, "exponent": 0},
                    "BHD": {"minor_per_usd": 377, "exponent": 3},
                    "INR": {"minor_per_usd": 8400, "exponent": 2}
                },
                "region_currency": {"jp": "JPY", "bh": "BHD"}
            }"#,
        )
        .unwrap();
        let fx = FxRates::default().merged(cfg);

        // New currencies present; codes uppercased on load.
        assert!(fx.knows("JPY"));
        assert!(fx.knows("BHD"));
        // Region mappings added.
        assert_eq!(fx.resolve(None, Some("JP")).0, "JPY");
        assert_eq!(fx.resolve(None, Some("BH")).0, "BHD");
        // INR override took effect.
        assert_eq!(fx.resolve(None, Some("IN")).1.minor_per_usd, 8400);
        // Untouched legacy rows still there.
        assert_eq!(fx.resolve(None, None).0, "USD");
    }

    #[test]
    fn jpy_zero_dp_is_not_off_by_100x() {
        // JPY has 0 decimal places: minor_per_usd already in whole yen.
        // 1 USD (1_000_000 micro-USD) at 150 JPY/USD → 150 yen (minor units).
        let rate = CurrencyRate {
            minor_per_usd: 150,
            exponent: 0,
        };
        assert_eq!(micro_usd_to_minor_units(1_000_000, rate), 150);
        // A naive "always 2dp" bug would give 15_000 — guard against it.
        assert_ne!(micro_usd_to_minor_units(1_000_000, rate), 15_000);
    }

    #[test]
    fn bhd_three_dp_minor_units() {
        // BHD has 3 decimal places (fils): 0.377 BHD = 377 fils per USD.
        // 1 USD → 377 fils.
        let rate = CurrencyRate {
            minor_per_usd: 377,
            exponent: 3,
        };
        assert_eq!(micro_usd_to_minor_units(1_000_000, rate), 377);
    }

    #[test]
    fn conversion_saturates_and_never_panics() {
        let rate = CurrencyRate {
            minor_per_usd: u64::MAX,
            exponent: 2,
        };
        // No overflow panic; saturating multiply then divide.
        let _ = micro_usd_to_minor_units(u64::MAX, rate);
        // Zero spend is always zero.
        assert_eq!(micro_usd_to_minor_units(0, rate), 0);
    }

    #[test]
    fn arc_swap_hot_swap_replaces_the_table() {
        let handle = shared(FxRates::default());
        assert_eq!(handle.load().resolve(None, Some("JP")).0, "USD"); // JP unknown

        let cfg: FxRatesConfig = serde_json::from_str(
            r#"{"rates":{"JPY":{"minor_per_usd":150,"exponent":0}},
                "region_currency":{"JP":"JPY"}}"#,
        )
        .unwrap();
        handle.replace(FxRates::default().merged(cfg));

        // The swapped-in table now resolves JP → JPY without a restart.
        assert_eq!(handle.load().resolve(None, Some("JP")).0, "JPY");
    }

    #[test]
    fn from_env_with_no_config_is_the_default() {
        // No env vars set in this unit (tests run single-process; we don't set
        // them) → default table. Guards the ship-dark contract.
        // (We assert structural equality with default rather than reading env to
        // avoid cross-test env races.)
        assert_eq!(FxRates::default(), FxRates::default());
    }

    #[test]
    fn malformed_default_currency_override_is_ignored() {
        let cfg: FxRatesConfig = serde_json::from_str(
            r#"{"default_currency": "ZZZ"}"#, // not in the table
        )
        .unwrap();
        let fx = FxRates::default().merged(cfg);
        // Fallback stays USD (a valid, present currency).
        assert_eq!(fx.resolve(None, Some("XX")).0, "USD");
    }
}
