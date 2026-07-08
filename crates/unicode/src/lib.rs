//! Invisible / zero-width Unicode smuggling-normalization primitives (PRD-036 M31).
//!
//! Shared by the guardrails masker (`routeplane-guardrails`) and the sovereign
//! residency classifier (`routeplane-residency`) so the definition of an
//! "invisible" character lives in exactly ONE place. The two crates previously
//! carried duplicate copies of [`is_invisible`], which drifted — a char stripped
//! by one but not the other silently reopens a PII-smuggling bypass (a regulated
//! identifier interleaved with the un-stripped char classifies/masks as clean).
//! See ADR-118 / PRD-058. Pure: std only, no dependencies, no I/O.

use std::borrow::Cow;

/// True when `c` is an "invisible" / non-rendering control character that has no
/// legitimate place in a prompt or response but is a known prompt-injection and
/// data-exfiltration channel (PRD-036 M31): zero-width spaces/joiners, the BOM,
/// soft hyphen, bidi overrides/isolates, and the Unicode **Tags** block
/// (U+E0000–U+E007F) used to smuggle hidden ASCII instructions. Ordinary
/// whitespace (space/tab/newline) is deliberately NOT included.
///
/// This is the single source of truth for the set — extend it here, never in a
/// consumer.
#[must_use]
pub fn is_invisible(c: char) -> bool {
    matches!(c,
        '\u{00AD}'                  // soft hyphen
        | '\u{180E}'                // Mongolian vowel separator (deprecated)
        | '\u{200B}'..='\u{200F}'   // zero-width space/non-joiner/joiner + LRM/RLM
        | '\u{202A}'..='\u{202E}'   // bidi embeddings / overrides
        | '\u{2060}'..='\u{2064}'   // word joiner + invisible operators
        | '\u{2066}'..='\u{2069}'   // bidi isolates
        | '\u{FEFF}'                // BOM / zero-width no-break space
        | '\u{E0000}'..='\u{E007F}' // Unicode Tags block (ASCII smuggling)
    )
}

/// True when `text` contains any invisible/control smuggling character.
#[must_use]
pub fn contains_invisible_unicode(text: &str) -> bool {
    text.chars().any(is_invisible)
}

/// Strip invisible/control smuggling characters. Zero-copy (`Cow::Borrowed`) when
/// the text is clean — the common case.
#[must_use]
pub fn strip_invisible(text: &str) -> Cow<'_, str> {
    if !contains_invisible_unicode(text) {
        return Cow::Borrowed(text);
    }
    Cow::Owned(text.chars().filter(|c| !is_invisible(*c)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_and_strips_the_full_set() {
        // ZWSP + a bidi override + a Tags-block char interleaved in "ignore this".
        let smuggled = "ig\u{200B}nore\u{202E} this\u{E0041}";
        assert!(contains_invisible_unicode(smuggled));
        assert_eq!(strip_invisible(smuggled).as_ref(), "ignore this");
    }

    #[test]
    fn clean_text_is_zero_copy() {
        // Ordinary whitespace, a Devanagari virama, and an emoji are all visible.
        let clean = "a normal\tline\nwith \u{094D} and 🙂";
        assert!(!contains_invisible_unicode(clean));
        assert!(matches!(strip_invisible(clean), Cow::Borrowed(_)));
    }

    #[test]
    fn covers_every_range_boundary() {
        for c in [
            '\u{00AD}',
            '\u{180E}',
            '\u{200B}',
            '\u{200F}',
            '\u{202A}',
            '\u{202E}',
            '\u{2060}',
            '\u{2064}',
            '\u{2066}',
            '\u{2069}',
            '\u{FEFF}',
            '\u{E0000}',
            '\u{E007F}',
        ] {
            assert!(is_invisible(c), "U+{:04X} must be invisible", c as u32);
        }
        for c in ['a', ' ', '\t', '\n', '9', '\u{094D}', '🙂'] {
            assert!(!is_invisible(c), "U+{:04X} must be visible", c as u32);
        }
    }
}
