//! Multi-currency cost attribution ([PRD-015], the FinOps moat — the
//! differentiator over the USD-only cost tracking of Portkey/LiteLLM).
//!
//! The canonical spend figure stays **integer micro-USD** (what budgets debit —
//! money is never a float). This module layers a *currency view* on top: given a
//! display currency (chosen by the `x-routeplane-currency` header, the routed
//! region, or the global default — in that order) it converts the micro-USD cost
//! into that currency's integer **minor units** (paise, cents, fils, yen),
//! respecting each currency's ISO-4217 minor-unit exponent so a 0-dp currency
//! (JPY) is never off by 100× from a 2-dp one.
//!
//! The FX rates live in [`crate::fx`] — a configurable, hot-swappable
//! [`crate::fx::FxRates`] table (config/env, no live feed, no DB, $0 standing
//! cost). With no rate config the table is **byte-identical** to the legacy
//! 4-bucket placeholder, so existing golden/parity snapshots do not shift.
//!
//! Pure, allocation-light, saturating — safe to call on the hot path.

use crate::estimate_cost_micro_usd;
use crate::fx::{micro_usd_to_minor_units, FxRates};

/// A per-request cost attribution: the canonical micro-USD figure plus a
/// local-currency view derived from the chosen display currency.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CostBreakdown {
    /// Canonical spend in integer micro-USD (USD millionths). Budgets debit this.
    pub micro_usd: u64,
    /// ISO-4217 currency code for the display view (e.g. `INR`, `USD`, `JPY`).
    pub currency: String,
    /// Cost in the currency's integer minor units (paise/cents/fils/yen).
    pub minor_units: u64,
    /// The routed residency region this cost is attributed to, if any.
    pub region: Option<String>,
}

/// Compute the cost breakdown for a completed request using the **built-in
/// default** FX table (the legacy placeholder; currency derived from `region`).
///
/// This is the byte-identical legacy entry point — `region` resolves a currency
/// exactly as the old `currency_for` did, so existing snapshots are unchanged.
/// New call sites that have a request-level display currency and a live (possibly
/// operator-overridden) rate table should call [`cost_breakdown_with`].
///
/// `region` is the routed residency region (`None` ⇒ default USD). All conversion
/// is saturating integer math (never panics, never a float).
pub fn cost_breakdown(
    model: &str,
    region: Option<&str>,
    prompt_tokens: u32,
    completion_tokens: u32,
) -> CostBreakdown {
    cost_breakdown_with(
        &FxRates::default(),
        None,
        model,
        region,
        prompt_tokens,
        completion_tokens,
    )
}

/// Compute the cost breakdown against a specific [`FxRates`] table and an optional
/// request-level **display currency** (the `x-routeplane-currency` header, an
/// ISO-4217 code).
///
/// The display-currency fallback chain is **header → region → global default**
/// (see [`FxRates::resolve`]); an unknown/unsupported header code falls through
/// gracefully (never errors, never panics). The canonical `micro_usd` is the same
/// regardless of the chosen currency — it is the truth; `currency` + `minor_units`
/// is the derived view. Conversion respects the currency's minor-unit exponent,
/// so a JPY (0-dp) figure is not 100× a USD (2-dp) one.
pub fn cost_breakdown_with(
    rates: &FxRates,
    display_currency: Option<&str>,
    model: &str,
    region: Option<&str>,
    prompt_tokens: u32,
    completion_tokens: u32,
) -> CostBreakdown {
    let micro_usd = estimate_cost_micro_usd(model, prompt_tokens, completion_tokens);
    let (currency, rate) = rates.resolve(display_currency, region);
    let minor_units = micro_usd_to_minor_units(micro_usd, rate);
    CostBreakdown {
        micro_usd,
        currency,
        minor_units,
        region: region.map(str::to_string),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn micro_usd_matches_the_canonical_estimator() {
        let b = cost_breakdown("gpt-4o", None, 1000, 1000);
        assert_eq!(b.micro_usd, estimate_cost_micro_usd("gpt-4o", 1000, 1000));
    }

    #[test]
    fn default_region_is_usd_cents() {
        // gpt-4o: 1000*3 + 1000*10 = 13_000 micro-USD = 0.013 USD = 1 cent (floor).
        let b = cost_breakdown("gpt-4o", None, 1000, 1000);
        assert_eq!(b.currency, "USD");
        assert_eq!(b.minor_units, 13_000 * 100 / 1_000_000); // = 1
        assert!(b.region.is_none());
    }

    #[test]
    fn india_region_is_inr_paise() {
        let b = cost_breakdown("gpt-4o", Some("IN"), 1000, 1000);
        assert_eq!(b.currency, "INR");
        assert_eq!(b.minor_units, 13_000u64 * 8_300 / 1_000_000); // 107 paise
        assert_eq!(b.region.as_deref(), Some("IN"));
        // canonical micro-USD is currency-independent
        assert_eq!(b.micro_usd, 13_000);
    }

    #[test]
    fn eu_and_ae_regions_resolve_their_currencies() {
        assert_eq!(
            cost_breakdown("claude-3", Some("EU"), 10, 10).currency,
            "EUR"
        );
        assert_eq!(
            cost_breakdown("claude-3", Some("AE"), 10, 10).currency,
            "AED"
        );
    }

    #[test]
    fn zero_tokens_is_zero_cost() {
        let b = cost_breakdown("gpt-4o", Some("IN"), 0, 0);
        assert_eq!(b.micro_usd, 0);
        assert_eq!(b.minor_units, 0);
    }

    #[test]
    fn saturating_never_panics_on_huge_token_counts() {
        let b = cost_breakdown("gpt-4o", Some("IN"), u32::MAX, u32::MAX);
        // No panic, no overflow; some finite figure.
        assert!(b.micro_usd > 0);
    }

    // --- display-currency selection (PRD-015 FR-3) ----------------------------

    #[test]
    fn display_currency_header_overrides_region() {
        // Header EUR wins over the IN region; micro-USD truth is unchanged.
        let b = cost_breakdown_with(
            &FxRates::default(),
            Some("EUR"),
            "gpt-4o",
            Some("IN"),
            1000,
            1000,
        );
        assert_eq!(b.currency, "EUR");
        assert_eq!(b.micro_usd, 13_000);
        assert_eq!(b.minor_units, 13_000 * 92 / 1_000_000);
        // region label still records the routed region (truth), not the display ccy
        assert_eq!(b.region.as_deref(), Some("IN"));
    }

    #[test]
    fn unknown_display_currency_falls_back_to_region_then_usd() {
        // Unknown code + IN region → INR (graceful, no error).
        let b = cost_breakdown_with(
            &FxRates::default(),
            Some("ZZZ"),
            "gpt-4o",
            Some("IN"),
            1000,
            1000,
        );
        assert_eq!(b.currency, "INR");
        // Unknown code + no/unknown region → USD.
        let b2 = cost_breakdown_with(&FxRates::default(), Some("ZZZ"), "gpt-4o", None, 1000, 1000);
        assert_eq!(b2.currency, "USD");
    }

    #[test]
    fn default_entry_point_is_byte_identical_to_with_no_display_ccy() {
        // The legacy `cost_breakdown` must equal `cost_breakdown_with(default,
        // None, ...)` for every legacy bucket — the ship-dark guarantee.
        for region in [None, Some("IN"), Some("EU"), Some("AE"), Some("US")] {
            let legacy = cost_breakdown("gpt-4o", region, 1234, 567);
            let with = cost_breakdown_with(&FxRates::default(), None, "gpt-4o", region, 1234, 567);
            assert_eq!(legacy, with, "region={region:?}");
        }
    }
}
