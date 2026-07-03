use lazy_static::lazy_static;
use regex::Regex;
use routeplane_types::Region;

mod verhoeff;

/// Version tag of this classifier, recorded in every sovereign-audit-ledger
/// entry (PRD-001 FR-6 / decision 001-A: boolean classifier + version now;
/// nullable confidence reserved in the schema). Bump semantics ride the crate
/// version: changing recognizers without a version bump would make the
/// artifact's limitations disclosure lie about which classifier produced an
/// entry.
pub const CLASSIFIER_VERSION: &str = concat!("routeplane-residency/", env!("CARGO_PKG_VERSION"));

lazy_static! {
    static ref EMAIL: Regex = Regex::new(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}").unwrap();
    // Phone: require an explicit phone shape, not "any 10 digits" (Task #6 fixes
    // the over-broad regex). We accept:
    //   * an international/E.164-style number with a leading '+' and 8–15 digits
    //     (optionally separated), OR
    //   * a clearly-separated grouped number (separators REQUIRED) so a bare
    //     10-digit string (an order id, a PAN-adjacent number, etc.) is NOT a
    //     false positive, OR
    //   * India's canonical domestic mobile display — 10 digits grouped 5-5
    //     (`98765 43210` / `98765-43210`) without the +91 prefix. The separator is
    //     REQUIRED (so a bare 10-digit run is still NOT flagged) and the leading
    //     digit must be 6-9 (India mobiles never start below 6). This is the
    //     India-first launch market's own profile #0; unlike the ID recognizers
    //     PHONE has no checksum gate, so a benign 5-space-5 digit pair with a 6-9
    //     lead (e.g. "range 60000 90000") is an accepted false-positive tradeoff —
    //     the separator + lead-digit structure is the guard.
    static ref PHONE: Regex = Regex::new(
        r"(?x)
        (?:
            \+\d[\d\ \-\.]{6,18}\d        # +CC then 8-15 more digits w/ optional seps
          | \b\d{3}[-.\ ]\d{3}[-.\ ]\d{4}\b  # 3-3-4 with REQUIRED separators
          | \b[6-9]\d{4}[\ \-]\d{5}\b       # India domestic mobile 5-5, sep REQUIRED, lead 6-9
        )"
    ).unwrap();
    // India identifiers — DPDP profile #0. Other jurisdictions plug in their own
    // recognizers (SSN, IBAN, NINO, ...) the same way.
    //
    // Aadhaar candidate shape: 12 digits in 4-4-4 groups, first digit 2-9 (UIDAI
    // never issues an Aadhaar starting 0 or 1). Group separators are optional and
    // may be spaces or hyphens — `2341 2341 2346`, `2341-2341-2346`, and the
    // double-spaced `2341  2341  2346` are all accepted, matching how users
    // actually type the number and staying consistent with the other
    // separator-tolerant recognizers (Emirates ID / TFN / My Number). The match is
    // then validated with the Verhoeff checksum (which strips separators first), so
    // a random 12-digit string is NOT flagged.
    static ref AADHAAR: Regex = Regex::new(r"\b[2-9]\d{3}[\s-]{0,2}\d{4}[\s-]{0,2}\d{4}\b").unwrap();
    static ref PAN: Regex = Regex::new(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b").unwrap();

    // US SSN — HIPAA / US profile. Separators REQUIRED (3-2-4) so a bare
    // 9-digit run is not a false positive (same discipline as PHONE/AADHAAR);
    // the match is then validated against SSA structural rules (`is_valid_ssn`).
    static ref SSN: Regex = Regex::new(r"\b\d{3}-\d{2}-\d{4}\b").unwrap();

    // IBAN — EU / GDPR profile. 2-letter country + 2 check digits + the BBAN,
    // accepting BOTH the compact electronic form (`GB82WEST12345698765432`) and the
    // ISO 13616 print form of space-separated 4-char groups
    // (`GB82 WEST 1234 5698 7654 32` — how IBANs appear on invoices, bank letters,
    // and most human-pasted text). The `{3,8}` group bound spans the same
    // 11–30-char BBAN range as the old compact `{11,30}` (⌈11/4⌉=3 … ⌈30/4⌉=8), so
    // it is a strict superset. The candidate is then mod-97 validated
    // (`is_valid_iban`, which strips the grouping spaces first), so a random
    // alphanumeric run that merely matches the shape is rejected.
    static ref IBAN: Regex =
        Regex::new(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{1,4}){3,8}\b").unwrap();

    // Brazil CPF — LGPD profile. Formatted form `DDD.DDD.DDD-DD` REQUIRED (so a
    // bare 11-digit run is not a false positive); validated by the two mod-11
    // check digits (`is_valid_cpf`).
    static ref CPF: Regex = Regex::new(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b").unwrap();

    // Singapore NRIC/FIN — PDPA profile. Prefix S/T/F/G/M + 7 digits + checksum
    // letter; the letter is then verified (`is_valid_nric`), so a random
    // letter-7digits-letter run that merely matches the shape is rejected.
    static ref NRIC: Regex = Regex::new(r"\b[STFGM]\d{7}[A-Z]\b").unwrap();

    // UAE Emirates ID — PDPL profile. Format `784-YYYY-NNNNNNN-C` (15 digits,
    // optional hyphens). The mandatory `784` prefix + a plausible 4-digit
    // registration year (19xx/20xx) is the structural gate (`is_valid_emirates_id`).
    // NOTE: UAE publishes NO official check-digit algorithm; the commonly-cited
    // Luhn check is unconfirmed and is known to reject genuinely-valid cards, so
    // we deliberately do NOT apply a checksum (would cause false negatives). The
    // 784 + year structure is far more discriminating than a bare 15-digit run.
    static ref EMIRATES_ID: Regex =
        Regex::new(r"\b784[- ]?\d{4}[- ]?\d{7}[- ]?\d\b").unwrap();

    // Saudi National ID / Iqama — KSA PDPL profile. 10 digits, first digit 1
    // (Saudi national) or 2 (Iqama / resident); validated by a Luhn-style mod-10
    // checksum doubling the leftmost-and-alternating digits (`is_valid_saudi_id`).
    // Bare-digit-run safe: a random 10-digit string fails the checksum.
    static ref SAUDI_ID: Regex = Regex::new(r"\b[12]\d{9}\b").unwrap();

    // India IFSC — RBI bank-branch code. 4 bank letters + mandatory `0` (reserved)
    // + 6 alphanumeric branch chars (`is_valid_ifsc`). The fixed `0` at position 5
    // is the structural discriminator (RBI-assigned, not free-form).
    static ref IFSC: Regex = Regex::new(r"\b[A-Z]{4}0[A-Z0-9]{6}\b").unwrap();

    // Australia TFN — Privacy Act profile. 9 digits (commonly 3-3-3 grouped);
    // validated by the ATO weighted mod-11 checksum (`is_valid_tfn`). Separators
    // optional; the checksum is the gate, so a bare 9-digit run only matches if it
    // satisfies the weighted sum.
    static ref TFN: Regex = Regex::new(r"\b\d{3}[- ]?\d{3}[- ]?\d{3}\b").unwrap();

    // Japan My Number (個人番号) — APPI profile. 12 digits (optionally 4-4-4
    // grouped); validated by the weighted mod-11 check digit (`is_valid_my_number`).
    static ref MY_NUMBER: Regex = Regex::new(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}\b").unwrap();
}

/// True if `candidate` (a digits-and-spaces Aadhaar candidate) passes the
/// Verhoeff checksum used by UIDAI. Strips separators first.
fn is_valid_aadhaar(candidate: &str) -> bool {
    let digits: Vec<u8> = candidate
        .chars()
        .filter(|c| c.is_ascii_digit())
        .map(|c| c as u8 - b'0')
        .collect();
    if digits.len() != 12 {
        return false;
    }
    verhoeff::validate(&digits)
}

/// True if `candidate` is a structurally valid PAN: 5 letters, 4 digits, 1
/// letter, where the 4th character (entity type) is one of the allowed codes.
fn is_valid_pan(candidate: &str) -> bool {
    let b = candidate.as_bytes();
    if b.len() != 10 {
        return false;
    }
    // First 5 alpha, next 4 digits, last alpha is enforced by the regex; here we
    // additionally validate the 4th char (holder type) against PAN's allowed
    // entity codes — rejects sequences that match the shape but can't be a PAN.
    // Standard PAN 4th-character entity codes (P,C,H,F,A,T,B,L,J,G,E,D,K).
    const VALID_ENTITY: &[u8] = b"PCHFATBLJGEDK";
    VALID_ENTITY.contains(&b[3])
}

/// True if `candidate` is a structurally valid US SSN (`AAA-GG-SSSS`) per the
/// SSA's published allocation rules — rejects shapes that can never be issued:
/// area `000`, `666`, or `900–999`; group `00`; serial `0000`.
fn is_valid_ssn(candidate: &str) -> bool {
    let digits: Vec<u8> = candidate
        .chars()
        .filter(|c| c.is_ascii_digit())
        .map(|c| c as u8 - b'0')
        .collect();
    if digits.len() != 9 {
        return false;
    }
    let area = digits[0] as u16 * 100 + digits[1] as u16 * 10 + digits[2] as u16;
    let group = digits[3] * 10 + digits[4];
    let serial = digits[5..9].iter().fold(0u16, |a, &d| a * 10 + d as u16);
    area != 0 && area != 666 && area < 900 && group != 0 && serial != 0
}

/// True if `candidate` is a valid IBAN by the ISO 13616 mod-97 check: move the
/// first four chars to the end, map letters to numbers (A=10…Z=35), and verify
/// the resulting integer ≡ 1 (mod 97). Computed digit-by-digit so no big-integer
/// type is needed. Input is the regex match, which may be either the compact form
/// or the ISO 13616 print form (4-char groups separated by single spaces); the
/// grouping spaces are stripped first so both forms validate identically against
/// the canonical compact IBAN (len 15–34, all `[A-Z0-9]`).
fn is_valid_iban(candidate: &str) -> bool {
    // Drop the print-format grouping spaces (the regex only ever inserts single
    // ASCII spaces between groups) so the length check and mod-97 arithmetic run
    // on the canonical compact IBAN.
    let compact: Vec<u8> = candidate.bytes().filter(|b| *b != b' ').collect();
    if compact.len() < 15 || compact.len() > 34 {
        return false;
    }
    // Rearrange: BBAN (from index 4) first, then the country+check (first 4).
    let rearranged = compact[4..].iter().chain(compact[..4].iter());
    let mut remainder: u32 = 0;
    for &c in rearranged {
        match c {
            b'0'..=b'9' => remainder = (remainder * 10 + (c - b'0') as u32) % 97,
            b'A'..=b'Z' => {
                // Two-digit value 10–35 → fold in as two decimal digits.
                let v = (c - b'A') as u32 + 10;
                remainder = (remainder * 100 + v) % 97;
            }
            _ => return false,
        }
    }
    remainder == 1
}

/// True if `candidate` (`DDD.DDD.DDD-DD`) is a valid Brazil CPF by its two mod-11
/// check digits. Also rejects the all-same-digit sequences (`000…`, `111…`) that
/// pass the arithmetic but are never issued.
fn is_valid_cpf(candidate: &str) -> bool {
    let d: Vec<u8> = candidate
        .chars()
        .filter(|c| c.is_ascii_digit())
        .map(|c| c as u8 - b'0')
        .collect();
    if d.len() != 11 || d.iter().all(|&x| x == d[0]) {
        return false;
    }
    // Check digit over the first `n` digits with descending weights (n+1 … 2).
    let check = |n: usize| -> u8 {
        let sum: u32 = (0..n)
            .map(|i| d[i] as u32 * (n as u32 + 1 - i as u32))
            .sum();
        let r = (sum * 10) % 11;
        if r == 10 {
            0
        } else {
            r as u8
        }
    };
    check(9) == d[9] && check(10) == d[10]
}

/// True if `candidate` (`[STFGM]DDDDDDD[A-Z]`) is a valid Singapore NRIC/FIN: the
/// trailing letter must match the weighted-checksum letter for the prefix class
/// (S/T citizen-PR vs F/G foreigner vs M; T/G/M shift the table).
fn is_valid_nric(candidate: &str) -> bool {
    let b = candidate.as_bytes();
    if b.len() != 9 {
        return false;
    }
    let prefix = b[0];
    let digits: Vec<u32> = b[1..8].iter().map(|c| (c - b'0') as u32).collect();
    const WEIGHTS: [u32; 7] = [2, 7, 6, 5, 4, 3, 2];
    let mut sum: u32 = digits.iter().zip(WEIGHTS).map(|(d, w)| d * w).sum();
    // Newer prefixes shift the checksum by a fixed offset.
    sum += match prefix {
        b'T' | b'G' => 4,
        b'M' => 3,
        _ => 0,
    };
    let idx = (sum % 11) as usize;
    let table: &[u8; 11] = match prefix {
        b'S' | b'T' => b"JZIHGFEDCBA",
        b'F' | b'G' => b"XWUTRQPNMLK",
        b'M' => b"XWUTRQPNJLK",
        _ => return false,
    };
    b[8] == table[idx]
}

/// True if `candidate` (`784-YYYY-NNNNNNN-C`, hyphens optional) is a structurally
/// valid UAE Emirates ID. The `784` UAE country prefix is enforced by the regex;
/// here we additionally require the 4-digit registration year to be plausible
/// (1900–2099) — that prefix-plus-year structure is the false-positive guard.
///
/// We intentionally apply NO check-digit validation: UAE publishes no official
/// algorithm and the commonly-cited Luhn check is unconfirmed and rejects real
/// cards (would create false negatives on genuine PII — the worse error for a
/// residency classifier). Structure alone is the gate.
fn is_valid_emirates_id(candidate: &str) -> bool {
    let d: Vec<u8> = candidate
        .chars()
        .filter(|c| c.is_ascii_digit())
        .map(|c| c as u8 - b'0')
        .collect();
    if d.len() != 15 {
        return false;
    }
    // Country prefix 784 (enforced by regex, re-checked defensively).
    if d[0] != 7 || d[1] != 8 || d[2] != 4 {
        return false;
    }
    // Registration year (digits 3..7) must be a plausible 19xx/20xx year.
    let year = d[3] as u16 * 1000 + d[4] as u16 * 100 + d[5] as u16 * 10 + d[6] as u16;
    (1900..=2099).contains(&year)
}

/// True if `candidate` (10 digits) is a valid Saudi National ID (first digit 1)
/// or Iqama / resident ID (first digit 2) per the published Luhn-style mod-10
/// checksum: starting from the leftmost digit, double every other digit (indices
/// 0,2,4,6,8), summing the two decimal digits of any doubled value > 9; add the
/// remaining digits as-is; the total must be ≡ 0 (mod 10). Algorithm per the
/// widely-used `alhazmy13/Saudi-ID-Validator` reference implementation.
fn is_valid_saudi_id(candidate: &str) -> bool {
    let b = candidate.as_bytes();
    if b.len() != 10 || (b[0] != b'1' && b[0] != b'2') {
        return false;
    }
    let mut sum: u32 = 0;
    for (i, &c) in b.iter().enumerate() {
        if !c.is_ascii_digit() {
            return false;
        }
        let digit = (c - b'0') as u32;
        if i % 2 == 0 {
            let doubled = digit * 2;
            sum += doubled / 10 + doubled % 10;
        } else {
            sum += digit;
        }
    }
    sum % 10 == 0
}

/// True if `candidate` is a structurally valid India IFSC code: 4 bank letters,
/// the mandatory reserved `0` at position 5, then 6 alphanumeric branch chars.
/// The fixed `0` is RBI-assigned (not free-form), which is the discriminator that
/// keeps an arbitrary 11-char alnum token from matching.
fn is_valid_ifsc(candidate: &str) -> bool {
    let b = candidate.as_bytes();
    if b.len() != 11 {
        return false;
    }
    b[..4].iter().all(u8::is_ascii_uppercase)
        && b[4] == b'0'
        && b[5..].iter().all(u8::is_ascii_alphanumeric)
}

/// True if `candidate` (9 digits, separators optional) is a valid Australian Tax
/// File Number by the ATO weighted mod-11 checksum: multiply each digit by the
/// fixed weights `[1,4,3,7,5,8,6,9,10]`; the sum must be ≡ 0 (mod 11). The
/// checksum (not the shape) is the gate, so a bare 9-digit run only matches if it
/// satisfies the weighted sum.
fn is_valid_tfn(candidate: &str) -> bool {
    let d: Vec<u32> = candidate
        .chars()
        .filter(|c| c.is_ascii_digit())
        .map(|c| (c as u8 - b'0') as u32)
        .collect();
    if d.len() != 9 {
        return false;
    }
    const WEIGHTS: [u32; 9] = [1, 4, 3, 7, 5, 8, 6, 9, 10];
    let sum: u32 = d.iter().zip(WEIGHTS).map(|(d, w)| d * w).sum();
    sum % 11 == 0
}

/// True if `candidate` (12 digits, separators optional) is a valid Japan
/// My Number (個人番号) by its weighted mod-11 check digit. Over the first 11
/// digits taken right-to-left, the weight at right-position `p` (1-based) is
/// `p+1` for `p ≤ 6` and `p−5` for `p > 6`; let `r = sum mod 11`; the check digit
/// is `0` when `r ≤ 1`, otherwise `11 − r`, and must equal the 12th digit.
fn is_valid_my_number(candidate: &str) -> bool {
    let d: Vec<u32> = candidate
        .chars()
        .filter(|c| c.is_ascii_digit())
        .map(|c| (c as u8 - b'0') as u32)
        .collect();
    if d.len() != 12 {
        return false;
    }
    let mut sum: u32 = 0;
    // Iterate the first 11 digits right-to-left; `p` is the 1-based right position.
    for (idx, p) in (1..=11u32).enumerate() {
        let digit = d[10 - idx];
        let weight = if p <= 6 { p + 1 } else { p - 5 };
        sum += digit * weight;
    }
    let r = sum % 11;
    let check = if r <= 1 { 0 } else { 11 - r };
    check == d[11]
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EntityType {
    Email,
    Phone,
    Aadhaar,
    Pan,
    /// US SSN (HIPAA / US profile).
    Ssn,
    /// IBAN (EU / GDPR profile).
    Iban,
    /// Brazil CPF (LGPD profile).
    Cpf,
    /// Singapore NRIC/FIN (PDPA profile).
    Nric,
    /// UAE Emirates ID (PDPL profile).
    EmiratesId,
    /// Saudi National ID / Iqama (KSA PDPL profile).
    SaudiId,
    /// India IFSC bank-branch code (RBI profile).
    Ifsc,
    /// Australia Tax File Number (Privacy Act profile).
    Tfn,
    /// Japan My Number / Individual Number (APPI profile).
    MyNumber,
}

#[derive(Debug, Clone)]
pub struct Classification {
    pub contains_personal_data: bool,
    pub entities: Vec<EntityType>,
}

/// The residency engine: classifies whether text contains regulated personal
/// data, and decides whether region-locked (sovereign) routing must be enforced.
///
/// This is the globalized DPDP engine — India (Aadhaar/PAN) is profile #0; the
/// same engine serves GDPR/CCPA/etc. by swapping the recognizer set and policy.
pub struct ResidencyEngine;

impl ResidencyEngine {
    pub fn new() -> Self {
        Self
    }

    /// Scan text for regulated personal-data entities.
    pub fn classify(&self, text: &str) -> Classification {
        let mut entities = Vec::new();
        // Aadhaar: shape match THEN Verhoeff checksum — a random 12-digit string
        // is not flagged (Task #6).
        if AADHAAR
            .find_iter(text)
            .any(|m| is_valid_aadhaar(m.as_str()))
        {
            entities.push(EntityType::Aadhaar);
        }
        // PAN: shape match THEN entity-code validation.
        if PAN.find_iter(text).any(|m| is_valid_pan(m.as_str())) {
            entities.push(EntityType::Pan);
        }
        // US SSN: shape (with required separators) THEN SSA structural rules.
        if SSN.find_iter(text).any(|m| is_valid_ssn(m.as_str())) {
            entities.push(EntityType::Ssn);
        }
        // IBAN: shape THEN mod-97 checksum.
        if IBAN.find_iter(text).any(|m| is_valid_iban(m.as_str())) {
            entities.push(EntityType::Iban);
        }
        // Brazil CPF: formatted shape THEN two mod-11 check digits.
        if CPF.find_iter(text).any(|m| is_valid_cpf(m.as_str())) {
            entities.push(EntityType::Cpf);
        }
        // Singapore NRIC/FIN: shape THEN checksum-letter verification.
        if NRIC.find_iter(text).any(|m| is_valid_nric(m.as_str())) {
            entities.push(EntityType::Nric);
        }
        // UAE Emirates ID: 784 + plausible-year structure (no unconfirmed checksum).
        if EMIRATES_ID
            .find_iter(text)
            .any(|m| is_valid_emirates_id(m.as_str()))
        {
            entities.push(EntityType::EmiratesId);
        }
        // Saudi National ID / Iqama: shape THEN Luhn-style mod-10 checksum.
        if SAUDI_ID
            .find_iter(text)
            .any(|m| is_valid_saudi_id(m.as_str()))
        {
            entities.push(EntityType::SaudiId);
        }
        // India IFSC: 4 letters + reserved `0` + 6 alnum (structural).
        if IFSC.find_iter(text).any(|m| is_valid_ifsc(m.as_str())) {
            entities.push(EntityType::Ifsc);
        }
        // Australia TFN: shape THEN ATO weighted mod-11 checksum.
        if TFN.find_iter(text).any(|m| is_valid_tfn(m.as_str())) {
            entities.push(EntityType::Tfn);
        }
        // Japan My Number: shape THEN weighted mod-11 check digit.
        if MY_NUMBER
            .find_iter(text)
            .any(|m| is_valid_my_number(m.as_str()))
        {
            entities.push(EntityType::MyNumber);
        }
        if EMAIL.is_match(text) {
            entities.push(EntityType::Email);
        }
        if PHONE.is_match(text) {
            entities.push(EntityType::Phone);
        }
        Classification {
            contains_personal_data: !entities.is_empty(),
            entities,
        }
    }

    /// Decide the sovereign routing constraint. If the caller requested a
    /// residency region AND the request carries personal data, routing must be
    /// locked to providers resident in that region. Returns `None` when no
    /// enforcement is required (no region requested, or no personal data).
    pub fn required_region(
        &self,
        requested: Option<&Region>,
        classification: &Classification,
    ) -> Option<Region> {
        match requested {
            Some(region) if classification.contains_personal_data => Some(region.clone()),
            _ => None,
        }
    }
}

impl Default for ResidencyEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_aadhaar() {
        let c = ResidencyEngine::new().classify("My Aadhaar is 4321 4321 4321");
        assert!(c.contains_personal_data);
        assert!(c.entities.contains(&EntityType::Aadhaar));
    }

    #[test]
    fn detects_pan() {
        let c = ResidencyEngine::new().classify("PAN: ABCDE1234F");
        assert!(c.contains_personal_data);
        assert!(c.entities.contains(&EntityType::Pan));
    }

    #[test]
    fn detects_email() {
        let c = ResidencyEngine::new().classify("reach me at user@example.com please");
        assert!(c.contains_personal_data);
        assert!(c.entities.contains(&EntityType::Email));
    }

    #[test]
    fn clean_text_is_not_personal_data() {
        let c = ResidencyEngine::new().classify("what is the capital of France?");
        assert!(!c.contains_personal_data);
        assert!(c.entities.is_empty());
    }

    // --- false-positive guards (Task #6) -------------------------------------

    #[test]
    fn random_12_digit_number_is_not_aadhaar() {
        // Verhoeff-invalid 12-digit string (e.g. an order id) must NOT classify
        // as Aadhaar.
        let c = ResidencyEngine::new().classify("order number 234123412345 shipped");
        assert!(!c.entities.contains(&EntityType::Aadhaar));
    }

    #[test]
    fn aadhaar_starting_with_one_is_rejected_by_shape() {
        // UIDAI never issues Aadhaar starting with 0 or 1.
        let c = ResidencyEngine::new().classify("code 1234 5678 9012 here");
        assert!(!c.entities.contains(&EntityType::Aadhaar));
    }

    #[test]
    fn valid_verhoeff_aadhaar_is_detected() {
        let c = ResidencyEngine::new().classify("aadhaar 2341 2341 2346");
        assert!(c.entities.contains(&EntityType::Aadhaar));
    }

    #[test]
    fn dash_and_multi_space_aadhaar_is_detected() {
        // Same Verhoeff-valid Aadhaar (digits 234123412346) written with the
        // separator styles users actually type: hyphen-grouped, double-spaced, and
        // hyphen/space-mixed. All must classify as Aadhaar so the sovereign region
        // lock is not silently bypassed (the previous `\s?` regex accepted only a
        // single space between groups).
        for form in [
            "customer aadhaar: 2341-2341-2346",
            "aadhaar 2341  2341  2346 on file",
            "id 2341-2341 2346 here",
            "234123412346", // bare/compact form still accepted
        ] {
            let c = ResidencyEngine::new().classify(form);
            assert!(
                c.entities.contains(&EntityType::Aadhaar),
                "should detect Aadhaar in {form:?}"
            );
        }
    }

    #[test]
    fn dashed_invalid_verhoeff_aadhaar_is_rejected() {
        // Negative: a dash-grouped 12-digit string whose Verhoeff check digit is
        // wrong (digits 234123412345) must NOT be flagged — the widened separator
        // set does not weaken the checksum gate.
        let c = ResidencyEngine::new().classify("ref 2341-2341-2345 shipped");
        assert!(!c.entities.contains(&EntityType::Aadhaar));
    }

    #[test]
    fn bare_ten_digit_number_is_not_a_phone() {
        // A 10-digit run with NO separators must not be flagged as a phone
        // (the previous regex flagged any 10 digits).
        let c = ResidencyEngine::new().classify("invoice 4155550123 total due");
        assert!(!c.entities.contains(&EntityType::Phone));
    }

    #[test]
    fn separated_phone_is_detected() {
        let c = ResidencyEngine::new().classify("call 415-555-0123 today");
        assert!(c.entities.contains(&EntityType::Phone));
    }

    #[test]
    fn e164_phone_is_detected() {
        let c = ResidencyEngine::new().classify("reach me at +91 98765 43210");
        assert!(c.entities.contains(&EntityType::Phone));
    }

    #[test]
    fn india_domestic_mobile_5_5_is_detected() {
        // India's canonical domestic mobile display (10 digits grouped 5-5, no +91),
        // with the separator either a space or a hyphen. Must be detected so a
        // DPDP-scoped request carrying only this form still triggers the region lock.
        for form in [
            "customer phone 98765 43210, please follow up",
            "call me on 98765-43210 tomorrow",
            "70000 12345", // lead digit 7 is a valid Indian mobile prefix
        ] {
            let c = ResidencyEngine::new().classify(form);
            assert!(
                c.entities.contains(&EntityType::Phone),
                "should detect India mobile in {form:?}"
            );
        }
    }

    #[test]
    fn bare_and_low_lead_india_mobile_are_not_phones() {
        // Negative: the separator is still REQUIRED (a bare 10-digit run is not a
        // phone), and the 5-5 form only counts with a 6-9 lead digit (India mobiles
        // never start below 6), so a 5-space-5 pair starting 1-5 is not flagged.
        let bare = ResidencyEngine::new().classify("order 9876543210 shipped");
        assert!(!bare.entities.contains(&EntityType::Phone));
        let low_lead = ResidencyEngine::new().classify("codes 12345 67890 batch");
        assert!(!low_lead.entities.contains(&EntityType::Phone));
    }

    #[test]
    fn pan_with_invalid_entity_code_is_rejected() {
        // 4th char 'Z' is not a valid PAN entity type code.
        let c = ResidencyEngine::new().classify("ref ABCZE1234F here");
        assert!(!c.entities.contains(&EntityType::Pan));
    }

    #[test]
    fn enforces_region_when_pii_and_region_present() {
        let engine = ResidencyEngine::new();
        let c = engine.classify("PAN ABCDE1234F");
        let region = Region::new("IN");
        assert_eq!(
            engine.required_region(Some(&region), &c),
            Some(Region::new("IN"))
        );
    }

    #[test]
    fn no_enforcement_without_region_header() {
        let engine = ResidencyEngine::new();
        let c = engine.classify("My Aadhaar is 4321 4321 4321");
        assert_eq!(engine.required_region(None, &c), None);
    }

    #[test]
    fn no_enforcement_when_no_personal_data() {
        let engine = ResidencyEngine::new();
        let c = engine.classify("hello world");
        assert_eq!(engine.required_region(Some(&Region::new("EU")), &c), None);
    }

    // --- US SSN (HIPAA profile) ----------------------------------------------

    #[test]
    fn detects_valid_ssn() {
        let c = ResidencyEngine::new().classify("SSN 123-45-6789 on file");
        assert!(c.contains_personal_data);
        assert!(c.entities.contains(&EntityType::Ssn));
    }

    #[test]
    fn structurally_impossible_ssn_is_rejected() {
        // Area 666 and 900+ and 000 are never issued; group 00 / serial 0000 invalid.
        for bad in [
            "666-45-6789",
            "000-45-6789",
            "900-45-6789",
            "123-00-6789",
            "123-45-0000",
        ] {
            let c = ResidencyEngine::new().classify(bad);
            assert!(
                !c.entities.contains(&EntityType::Ssn),
                "should reject {bad}"
            );
        }
    }

    #[test]
    fn bare_nine_digits_is_not_an_ssn() {
        // No separators ⇒ not matched (avoids flagging a 9-digit id).
        let c = ResidencyEngine::new().classify("ref 123456789 shipped");
        assert!(!c.entities.contains(&EntityType::Ssn));
    }

    // --- IBAN (GDPR / EU profile) --------------------------------------------

    #[test]
    fn detects_valid_iban() {
        // Canonical valid examples (ISO 13616 mod-97).
        for ok in ["GB82WEST12345698765432", "DE89370400440532013000"] {
            let c = ResidencyEngine::new().classify(&format!("pay to {ok} please"));
            assert!(c.entities.contains(&EntityType::Iban), "should detect {ok}");
        }
    }

    #[test]
    fn iban_with_bad_checksum_is_rejected() {
        // Last digit altered ⇒ mod-97 fails ⇒ not flagged.
        let c = ResidencyEngine::new().classify("acct GB82WEST12345698765433 here");
        assert!(!c.entities.contains(&EntityType::Iban));
    }

    #[test]
    fn detects_iban_print_format() {
        // ISO 13616 print form (space-separated 4-char groups) — the way IBANs
        // appear on invoices and bank letters — must be detected, not just the
        // compact electronic form. Same underlying valid IBANs as the compact test.
        for ok in ["GB82 WEST 1234 5698 7654 32", "DE89 3704 0044 0532 0130 00"] {
            let c = ResidencyEngine::new().classify(&format!("please pay to {ok} today"));
            assert!(
                c.entities.contains(&EntityType::Iban),
                "should detect print-format IBAN {ok}"
            );
        }
    }

    #[test]
    fn iban_print_format_with_bad_checksum_is_rejected() {
        // Negative: the print form is space-stripped and mod-97 validated, so a
        // spaced candidate with a wrong final digit is still rejected.
        let c = ResidencyEngine::new().classify("acct GB82 WEST 1234 5698 7654 33 here");
        assert!(!c.entities.contains(&EntityType::Iban));
    }

    #[test]
    fn ssn_and_iban_trigger_region_lock_like_any_pii() {
        let engine = ResidencyEngine::new();
        let us = engine.classify("ssn 123-45-6789");
        assert_eq!(
            engine.required_region(Some(&Region::new("US")), &us),
            Some(Region::new("US"))
        );
        let eu = engine.classify("iban GB82WEST12345698765432");
        assert_eq!(
            engine.required_region(Some(&Region::new("EU")), &eu),
            Some(Region::new("EU"))
        );
    }

    // --- Brazil CPF (LGPD profile) -------------------------------------------

    #[test]
    fn detects_valid_cpf() {
        let c = ResidencyEngine::new().classify("CPF 111.444.777-35 cadastrado");
        assert!(c.entities.contains(&EntityType::Cpf));
    }

    #[test]
    fn cpf_with_bad_check_digit_is_rejected() {
        let c = ResidencyEngine::new().classify("CPF 111.444.777-36 here");
        assert!(!c.entities.contains(&EntityType::Cpf));
    }

    #[test]
    fn cpf_all_same_digits_is_rejected() {
        // 111.111.111-11 passes the arithmetic but is never issued.
        let c = ResidencyEngine::new().classify("111.111.111-11");
        assert!(!c.entities.contains(&EntityType::Cpf));
    }

    #[test]
    fn bare_eleven_digits_is_not_a_cpf() {
        let c = ResidencyEngine::new().classify("ref 11144477735 ok");
        assert!(!c.entities.contains(&EntityType::Cpf));
    }

    // --- Singapore NRIC/FIN (PDPA profile) -----------------------------------

    #[test]
    fn detects_valid_nric() {
        let c = ResidencyEngine::new().classify("NRIC S1234567D on record");
        assert!(c.entities.contains(&EntityType::Nric));
    }

    #[test]
    fn nric_with_wrong_checksum_letter_is_rejected() {
        let c = ResidencyEngine::new().classify("id S1234567A here");
        assert!(!c.entities.contains(&EntityType::Nric));
    }

    #[test]
    fn cpf_and_nric_trigger_region_lock() {
        let engine = ResidencyEngine::new();
        let br = engine.classify("cpf 111.444.777-35");
        assert_eq!(
            engine.required_region(Some(&Region::new("BR")), &br),
            Some(Region::new("BR"))
        );
        let sg = engine.classify("nric S1234567D");
        assert_eq!(
            engine.required_region(Some(&Region::new("SG")), &sg),
            Some(Region::new("SG"))
        );
    }

    // --- UAE Emirates ID (PDPL profile) --------------------------------------

    #[test]
    fn detects_valid_emirates_id() {
        // 784 prefix + plausible 1973 registration year + 7-digit serial + check.
        for ok in [
            "784-1973-1234567-8",
            "784197312345678",
            "784 1973 1234567 8",
        ] {
            let c = ResidencyEngine::new().classify(&format!("EID {ok} on file"));
            assert!(
                c.entities.contains(&EntityType::EmiratesId),
                "should detect {ok}"
            );
        }
    }

    #[test]
    fn emirates_id_wrong_prefix_or_year_is_rejected() {
        // Wrong country prefix (not 784) and an implausible registration year are
        // both rejected — a bare 15-digit run is not flagged.
        let bad_prefix = ResidencyEngine::new().classify("id 123197312345678 here");
        assert!(!bad_prefix.entities.contains(&EntityType::EmiratesId));
        // 784 then year 1234 (< 1900) is implausible.
        let bad_year = ResidencyEngine::new().classify("id 784123412345678 here");
        assert!(!bad_year.entities.contains(&EntityType::EmiratesId));
    }

    // --- Saudi National ID / Iqama (KSA PDPL profile) ------------------------

    #[test]
    fn detects_valid_saudi_id_and_iqama() {
        // 1101798278 (national, type 1) and 2101798276 (iqama, type 2) both pass
        // the Luhn-style mod-10 checksum.
        let nat = ResidencyEngine::new().classify("national id 1101798278");
        assert!(nat.entities.contains(&EntityType::SaudiId));
        let iqama = ResidencyEngine::new().classify("iqama 2101798276");
        assert!(iqama.entities.contains(&EntityType::SaudiId));
    }

    #[test]
    fn saudi_id_bad_checksum_or_prefix_is_rejected() {
        // Flip a digit ⇒ checksum fails; a random 10-digit run is not flagged.
        let bad_sum = ResidencyEngine::new().classify("ref 1101798279 x");
        assert!(!bad_sum.entities.contains(&EntityType::SaudiId));
        // First digit 3 is neither national (1) nor resident (2).
        let bad_prefix = ResidencyEngine::new().classify("ref 3101798278 x");
        assert!(!bad_prefix.entities.contains(&EntityType::SaudiId));
    }

    // --- India IFSC (RBI profile) -------------------------------------------

    #[test]
    fn detects_valid_ifsc() {
        let c = ResidencyEngine::new().classify("branch HDFC0000043 NEFT");
        assert!(c.entities.contains(&EntityType::Ifsc));
    }

    #[test]
    fn ifsc_without_reserved_zero_is_rejected() {
        // 5th char must be the reserved `0`; a generic 11-char token is rejected.
        let c = ResidencyEngine::new().classify("token HDFCX000043 here");
        assert!(!c.entities.contains(&EntityType::Ifsc));
    }

    // --- Australia TFN (Privacy Act profile) --------------------------------

    #[test]
    fn detects_valid_tfn() {
        // 876543210 and 123456782 satisfy the ATO weighted mod-11 checksum.
        for ok in ["876543210", "876 543 210", "123-456-782"] {
            let c = ResidencyEngine::new().classify(&format!("TFN {ok}"));
            assert!(c.entities.contains(&EntityType::Tfn), "should detect {ok}");
        }
    }

    #[test]
    fn bare_nine_digit_run_is_not_a_tfn() {
        // 123456789 fails the weighted mod-11 ⇒ a random 9-digit run is not flagged.
        let c = ResidencyEngine::new().classify("order 123456789 shipped");
        assert!(!c.entities.contains(&EntityType::Tfn));
    }

    // --- Japan My Number (APPI profile) -------------------------------------

    #[test]
    fn detects_valid_my_number() {
        // 465281266333 carries a valid weighted mod-11 check digit.
        for ok in ["465281266333", "4652-8126-6333", "4652 8126 6333"] {
            let c = ResidencyEngine::new().classify(&format!("My Number {ok}"));
            assert!(
                c.entities.contains(&EntityType::MyNumber),
                "should detect {ok}"
            );
        }
    }

    #[test]
    fn my_number_bad_check_digit_is_rejected() {
        // Flip the trailing check digit ⇒ rejected; a random 12-digit run is not
        // flagged.
        let c = ResidencyEngine::new().classify("ref 465281266334 x");
        assert!(!c.entities.contains(&EntityType::MyNumber));
    }

    // --- region-lock parity for the new jurisdictions -----------------------

    #[test]
    fn new_jurisdictions_trigger_region_lock() {
        let engine = ResidencyEngine::new();
        let ae = engine.classify("eid 784197312345678");
        assert_eq!(
            engine.required_region(Some(&Region::new("AE")), &ae),
            Some(Region::new("AE"))
        );
        let sa = engine.classify("id 1101798278");
        assert_eq!(
            engine.required_region(Some(&Region::new("SA")), &sa),
            Some(Region::new("SA"))
        );
        let au = engine.classify("tfn 876543210");
        assert_eq!(
            engine.required_region(Some(&Region::new("AU")), &au),
            Some(Region::new("AU"))
        );
        let jp = engine.classify("my number 465281266333");
        assert_eq!(
            engine.required_region(Some(&Region::new("JP")), &jp),
            Some(Region::new("JP"))
        );
    }
}
