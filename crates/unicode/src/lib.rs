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

/// Fold a single confusable/homoglyph or non-ASCII decimal digit to its ASCII
/// skeleton. Returns `c` unchanged when there is nothing to fold.
///
/// **Strictly 1 char → 1 char.** This is load-bearing, not an implementation
/// detail: the masker locates PII on the folded copy and then redacts the
/// ORIGINAL text. A fold that changed the character count would shift every
/// subsequent index and corrupt the output, so a mapping is only admissible here
/// if it is one-to-one. (Byte lengths still differ — Cyrillic `А` is 2 bytes,
/// ASCII `A` is 1 — so consumers must map by CHAR index, never byte offset.)
///
/// Scope is deliberately the recognizer alphabet, not the full ~6 000-entry UTS
/// #39 confusables table: the regulated-identifier patterns are `[A-Z]`, `[a-z]`
/// and `[0-9]` (Aadhaar, PAN, SSN, IBAN, phone), so folding Latin-lookalike
/// letters and decimal digits closes the evasion without importing a large
/// table or a dependency. Extend the tables here, never in a consumer.
#[must_use]
pub fn fold_confusable(c: char) -> char {
    // Fast path: ASCII is already its own skeleton and is the overwhelming
    // majority of every prompt. This runs per char on the masking path, so the
    // common case must be a single comparison rather than the block scan below.
    if c.is_ascii() {
        return c;
    }
    // Non-ASCII decimal digits. Every Unicode decimal-digit block is a
    // contiguous run of ten from its own ZERO, so an offset from the block's
    // zero is the digit's value. `regex`'s `\d` already MATCHES these, but
    // `char::to_digit` is ASCII-only — so a Devanagari Aadhaar matched the shape
    // and then failed the Verhoeff checksum, and went unflagged.
    const DIGIT_ZEROS: [char; 12] = [
        '\u{0660}', // Arabic-Indic
        '\u{06F0}', // Extended Arabic-Indic (Persian/Urdu)
        '\u{0966}', // Devanagari
        '\u{09E6}', // Bengali
        '\u{0A66}', // Gurmukhi
        '\u{0AE6}', // Gujarati
        '\u{0B66}', // Oriya
        '\u{0BE6}', // Tamil
        '\u{0C66}', // Telugu
        '\u{0CE6}', // Kannada
        '\u{0D66}', // Malayalam
        '\u{FF10}', // Fullwidth
    ];
    for zero in DIGIT_ZEROS {
        let (z, cp) = (zero as u32, c as u32);
        if cp >= z && cp < z + 10 {
            // `as u8 as char` is safe: the value is 0..=9 offset from b'0'.
            return (b'0' + (cp - z) as u8) as char;
        }
    }
    // Fullwidth Latin letters (U+FF21..U+FF3A upper, U+FF41..U+FF5A lower).
    if ('\u{FF21}'..='\u{FF3A}').contains(&c) {
        return (b'A' + (c as u32 - 0xFF21) as u8) as char;
    }
    if ('\u{FF41}'..='\u{FF5A}').contains(&c) {
        return (b'a' + (c as u32 - 0xFF41) as u8) as char;
    }
    // Cyrillic / Greek letters that render identically to a Latin letter. Only
    // true visual identities — never a merely similar glyph, which would fold
    // legitimate Cyrillic or Greek prose into false positives.
    match c {
        'А' | 'Α' => 'A',
        'В' | 'Β' => 'B',
        'С' => 'C',
        'Е' | 'Ε' => 'E',
        'Н' | 'Η' => 'H',
        'І' | 'Ι' => 'I',
        'Ј' => 'J',
        'К' | 'Κ' => 'K',
        'М' | 'Μ' => 'M',
        // Cyrillic Н (U+041D) renders as Latin H and is folded above; Greek Ν
        // (U+039D) is the one that renders as Latin N.
        'Ν' => 'N',
        'О' | 'Ο' => 'O',
        'Р' | 'Ρ' => 'P',
        'Ѕ' => 'S',
        'Т' | 'Τ' => 'T',
        'Х' | 'Χ' => 'X',
        'У' | 'Υ' => 'Y',
        'Ζ' => 'Z',
        'а' => 'a',
        'с' => 'c',
        'е' => 'e',
        'о' => 'o',
        'р' => 'p',
        'ѕ' => 's',
        'х' => 'x',
        'у' => 'y',
        'і' => 'i',
        'ј' => 'j',
        _ => c,
    }
}

/// True when `text` contains any character [`fold_confusable`] would rewrite.
#[must_use]
pub fn contains_confusable(text: &str) -> bool {
    text.chars().any(|c| fold_confusable(c) != c)
}

/// Fold confusables/homoglyphs and non-ASCII digits to their ASCII skeleton.
/// Zero-copy (`Cow::Borrowed`) when the text is clean — the common case.
///
/// Char count is preserved exactly (see [`fold_confusable`]), so a match found
/// on the folded copy occupies the same CHAR range in the original.
#[must_use]
pub fn fold_confusables(text: &str) -> Cow<'_, str> {
    if !contains_confusable(text) {
        return Cow::Borrowed(text);
    }
    Cow::Owned(text.chars().map(fold_confusable).collect())
}

/// The full recognizer front-end: strip invisible smuggling characters, then
/// fold confusables/non-ASCII digits. This is what a classifier or masker should
/// call — applying only one of the two leaves the other evasion open.
///
/// Zero-copy when the text needs neither pass.
#[must_use]
pub fn normalize_for_recognition(text: &str) -> Cow<'_, str> {
    match strip_invisible(text) {
        Cow::Borrowed(s) => fold_confusables(s),
        Cow::Owned(s) => Cow::Owned(fold_confusables(&s).into_owned()),
    }
}

#[cfg(test)]
mod confusable_tests {
    use super::*;

    /// The 1:1 invariant the masker depends on. If this ever fails, redaction
    /// indices computed on the folded copy no longer line up with the original.
    #[test]
    fn fold_preserves_char_count_exactly() {
        for s in [
            "АВСЕНІЈКМΝОРЅТХУΖ",
            "аседорѕхуіј",
            "０１２３４５６７８９",
            "०१२३४५६७८९",
            "٠١٢٣٤٥٦٧٨٩",
            "ＡＢＣａｂｃ",
            "ordinary ascii 12345",
            "日本語とemoji🎉",
        ] {
            assert_eq!(
                fold_confusables(s).chars().count(),
                s.chars().count(),
                "fold changed char count for {s:?} — masker indices would shift"
            );
        }
    }

    #[test]
    fn folds_non_ascii_digits_to_ascii() {
        // Devanagari, Arabic-Indic, Bengali, Tamil, fullwidth.
        assert_eq!(fold_confusables("२३४१२३४१२३४६"), "234123412346");
        assert_eq!(fold_confusables("٢٣٤١٢٣٤١٢٣٤٦"), "234123412346");
        assert_eq!(fold_confusables("২৩৪১২৩৪১২৩৪৬"), "234123412346");
        assert_eq!(fold_confusables("１２３４"), "1234");
    }

    #[test]
    fn folds_cyrillic_and_greek_homoglyphs_to_latin() {
        // A PAN shaped `[A-Z]{5}[0-9]{4}[A-Z]` written with Cyrillic А/В/С/Е/Х —
        // visually identical, but `[A-Z]` never matched it.
        assert_eq!(fold_confusables("АВСЕХ1234Х"), "ABCEX1234X");
        // Greek Alpha/Beta/Epsilon likewise.
        assert_eq!(fold_confusables("ΑΒΕ"), "ABE");
    }

    #[test]
    fn leaves_ordinary_text_untouched_and_zero_copy() {
        for clean in ["AAAPA1234A", "234123412346", "hello world", "日本語 🎉"] {
            assert!(
                matches!(fold_confusables(clean), Cow::Borrowed(_)),
                "{clean:?} should not allocate"
            );
            assert_eq!(fold_confusables(clean), clean);
        }
    }

    /// Genuine Cyrillic/Greek prose must not be mangled into Latin — that would
    /// turn a false-negative bypass into a false-positive one.
    #[test]
    fn does_not_fold_letters_that_merely_look_similar() {
        // Ж, Ф, Д, Я, Ю have no Latin lookalike and must survive verbatim.
        assert_eq!(fold_confusables("ЖФДЯЮ"), "ЖФДЯЮ");
        assert_eq!(fold_confusables("Σ Δ Ω Λ Ξ"), "Σ Δ Ω Λ Ξ");
    }

    /// The combined front-end must close BOTH evasions at once — the realistic
    /// attack interleaves a zero-width space AND uses a homoglyph.
    #[test]
    fn normalize_closes_invisible_and_confusable_together() {
        // Cyrillic А + a zero-width space inside a PAN.
        let evasive = "А\u{200B}BCPA1234A";
        assert_eq!(normalize_for_recognition(evasive), "ABCPA1234A");
        // Devanagari digits + soft hyphen inside an Aadhaar.
        let aadhaar = "२३४१\u{00AD}२३४१२३४६";
        assert_eq!(normalize_for_recognition(aadhaar), "234123412346");
        // Clean text stays zero-copy through both passes.
        assert!(matches!(
            normalize_for_recognition("plain ascii"),
            Cow::Borrowed(_)
        ));
    }
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
