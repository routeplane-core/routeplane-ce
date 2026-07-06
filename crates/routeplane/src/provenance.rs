//! Response provenance headers — the one place the success-path trio
//! (`x-routeplane-provider`, `x-routeplane-trace-id`, `x-routeplane-request-id`)
//! is stamped, so every serving endpoint (chat, cache hits, embeddings, images,
//! rerank, audio) emits the identical contract instead of a hand-rolled subset
//! (upstream 2026-07 provenance fix).

use axum::http::{HeaderMap, HeaderValue};

/// Branding contract: echoes which provider served this response. The
/// acceptance suite asserts it on live completions.
pub const PROVIDER_HEADER: &str = "x-routeplane-provider";

/// Response header carrying the per-request correlation id (`req_<uuid>`) so a
/// client can attach feedback to this exact trace via `POST /v1/feedback`
/// (Portkey/Helicone parity). Additive: the golden/A-B parity corpus has no such
/// header, and the parity check is `is_additive_superset` (baseline ⊆ current),
/// so a NEW header never regresses parity even though its value is per-request.
pub const TRACE_ID_HEADER: &str = "x-routeplane-trace-id";

/// The same per-request correlation id is also emitted as
/// `x-routeplane-request-id` — the header name that predates the `trace-id`
/// standardisation. Both carry the identical `req_<uuid>` value, so a client
/// using either name correlates the same request. Additive (parity-safe).
pub const REQUEST_ID_HEADER: &str = "x-routeplane-request-id";

/// Stamp the provenance trio onto a success response. `trace-id` and
/// `request-id` carry the SAME `req_<uuid>` value (see the const docs above).
/// Both values are gateway-generated (never client-controlled), so the
/// `from_str` guards are for type safety, not sanitization; on the (impossible
/// in practice) invalid-value case the header is simply omitted.
pub fn stamp_provenance(headers: &mut HeaderMap, provider: &str, request_id: &str) {
    if let Ok(v) = HeaderValue::from_str(provider) {
        headers.insert(PROVIDER_HEADER, v);
    }
    if let Ok(v) = HeaderValue::from_str(request_id) {
        headers.insert(TRACE_ID_HEADER, v.clone());
        headers.insert(REQUEST_ID_HEADER, v);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stamps_all_three_headers() {
        let mut headers = HeaderMap::new();
        stamp_provenance(&mut headers, "openai", "req_abc123");
        assert_eq!(headers.get(PROVIDER_HEADER).unwrap(), "openai");
        assert_eq!(headers.get(TRACE_ID_HEADER).unwrap(), "req_abc123");
        assert_eq!(headers.get(REQUEST_ID_HEADER).unwrap(), "req_abc123");
    }

    #[test]
    fn trace_and_request_id_carry_the_same_value() {
        let mut headers = HeaderMap::new();
        stamp_provenance(&mut headers, "(cache)", "req_xyz");
        assert_eq!(
            headers.get(TRACE_ID_HEADER).unwrap(),
            headers.get(REQUEST_ID_HEADER).unwrap()
        );
    }
}
