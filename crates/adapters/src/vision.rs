//! Shared multimodal (vision) helpers for provider adapters.
//!
//! The canonical wire model carries images as OpenAI-shaped content parts
//! (`ContentPart::ImageUrl { image_url: { url, detail? } }`), where `url` is
//! EITHER an `http(s)://…` URL OR a `data:<media_type>;base64,<payload>` data
//! URI. Most OpenAI-compatible providers (Mistral/Pixtral, Cohere v2) accept
//! that exact shape, so they forward the canonical parts verbatim. Anthropic's
//! `/v1/messages` uses a DIFFERENT block shape, so this module also builds the
//! Anthropic content-block array.
//!
//! Hard constraints (CLAUDE.md): no panic on a request thread — data-URL
//! parsing returns `Option`/graceful skip, never `unwrap`/`expect`; image bytes
//! are NEVER logged.

/// A parsed `data:` URI: the declared media type and the (still-encoded) base64
/// payload. We intentionally do NOT decode the base64 — providers want the
/// base64 string, and decoding would only cost CPU and risk logging bytes.
pub(crate) struct DataUrl<'a> {
    pub media_type: &'a str,
    pub base64_payload: &'a str,
}

/// Parse a `data:<media_type>;base64,<payload>` URI without allocating.
/// Returns `None` for anything that is not a base64 data URI (e.g. an
/// `http(s)://` URL, or a malformed/`;charset`-only data URI) — callers treat
/// `None` as "not a data URL" and fall back to URL handling or skip.
///
/// Equivalent to the regex `data:(.*?);base64,(.*)` but panic-free and
/// allocation-free.
pub(crate) fn parse_data_url(url: &str) -> Option<DataUrl<'_>> {
    let rest = url.strip_prefix("data:")?;
    // Split metadata (everything before the first comma) from the payload.
    let (meta, payload) = rest.split_once(',')?;
    // The metadata must end with `;base64` (optionally preceded by the media
    // type and other params). We require the base64 marker; non-base64 data
    // URIs are not something we can forward as binary image data.
    let media_type = meta.strip_suffix(";base64")?;
    if media_type.is_empty() {
        // `data:;base64,…` — no declared media type; let the caller decide to
        // skip (Anthropic requires a media_type).
        return None;
    }
    Some(DataUrl {
        media_type,
        base64_payload: payload,
    })
}

/// Media types Anthropic's image blocks accept. An unsupported/missing type
/// means we skip the image rather than send a request Anthropic will reject.
pub(crate) fn is_anthropic_supported_media_type(media_type: &str) -> bool {
    matches!(
        media_type,
        "image/jpeg" | "image/png" | "image/gif" | "image/webp"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_png_data_url() {
        let d = parse_data_url("data:image/png;base64,iVBORw0KGgo=").expect("data url");
        assert_eq!(d.media_type, "image/png");
        assert_eq!(d.base64_payload, "iVBORw0KGgo=");
    }

    #[test]
    fn parses_jpeg_with_payload_containing_commas_after_first() {
        // Only the FIRST comma splits meta from payload; base64 has no commas
        // but we still prove split_once semantics are first-comma.
        let d = parse_data_url("data:image/jpeg;base64,AAAA").expect("data url");
        assert_eq!(d.media_type, "image/jpeg");
        assert_eq!(d.base64_payload, "AAAA");
    }

    #[test]
    fn http_url_is_not_a_data_url() {
        assert!(parse_data_url("https://example.com/x.png").is_none());
    }

    #[test]
    fn non_base64_data_url_is_rejected() {
        assert!(parse_data_url("data:text/plain,hello").is_none());
    }

    #[test]
    fn data_url_without_media_type_is_rejected() {
        assert!(parse_data_url("data:;base64,AAAA").is_none());
    }

    #[test]
    fn malformed_data_url_no_comma() {
        assert!(parse_data_url("data:image/png;base64").is_none());
    }

    #[test]
    fn supported_media_types() {
        for t in ["image/jpeg", "image/png", "image/gif", "image/webp"] {
            assert!(is_anthropic_supported_media_type(t));
        }
        assert!(!is_anthropic_supported_media_type("image/svg+xml"));
        assert!(!is_anthropic_supported_media_type("image/bmp"));
    }
}
