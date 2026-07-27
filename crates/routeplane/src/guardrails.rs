use serde::{Deserialize, Serialize};
// Only the enterprise-only `TokenizerKey` custody type holds an `Arc`.
#[cfg(feature = "enterprise")]
use std::sync::Arc;

/// How inbound/outbound PII is handled for a request (ADR-031 mask vs ADR-044
/// reversible tokenize). DEFAULT is [`PiiMode::Mask`] — irreversible, identical
/// to the always-on behavior — so absence of any tokenize opt-in is byte-for-byte
/// the pre-ADR-044 path.
#[derive(Debug, Serialize, Deserialize, Clone, Copy, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum PiiMode {
    /// Irreversible `[<CATEGORY>_MASKED]` redaction (the always-on default).
    #[default]
    Mask,
    /// Reversible, format-preserving tokenization (ADR-044): the provider sees a
    /// same-format surrogate; the gateway restores the original on egress. Falls
    /// back to [`PiiMode::Mask`] when no tokenizer key is configured (ship-dark).
    Tokenize,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct GuardrailConfig {
    pub mask_pii: bool,
    /// PII handling mode for this request. `Mask` (default) ⇒ identical legacy
    /// behavior; `Tokenize` ⇒ reversible round-trip when a key is configured.
    #[serde(default)]
    pub pii_mode: PiiMode,
}

/// Process-wide reversible tokenizer key custody (ADR-044). Resolved ONCE at
/// startup from the environment / Key Vault (OIDC-managed-identity in prod; a raw
/// env key is acceptable for dev). The key bytes never appear in logs. When no
/// key is configured this is `None` and `tokenize` mode gracefully degrades to
/// irreversible masking — the feature is ship-dark safe.
///
/// The [`routeplane_guardrails_advanced::tokenize::Tokenizer`] holds the AES-256 key
/// schedule; it is built once and shared (cheap `Arc` clone), never per request.
///
/// MOAT (ADR-088): reversible tokenization (ADR-044) rides `enterprise` — this
/// custody type and its helpers are absent on the CE build (`pii_mode=tokenize`
/// there simply never resolves to a tokenizer, degrading to masking).
#[cfg(feature = "enterprise")]
#[derive(Clone, Default)]
pub struct TokenizerKey(pub Option<Arc<routeplane_guardrails_advanced::tokenize::Tokenizer>>);

#[cfg(feature = "enterprise")]
impl TokenizerKey {
    /// Build the tokenizer from custody:
    /// - `ROUTEPLANE_TOKENIZE_KEY_HEX` — 64 hex chars (32 bytes), OR
    /// - `ROUTEPLANE_TOKENIZE_KEY` — a raw passphrase, SHA-256'd to 32 bytes
    ///   (dev convenience; prod should inject the Key-Vault-held key as hex).
    ///
    /// On a missing/invalid key this returns `None` and logs ONCE at info — the
    /// key bytes are NEVER logged. `tokenize` mode then falls back to masking.
    #[must_use]
    pub fn from_env() -> Self {
        use routeplane_guardrails_advanced::tokenize::Tokenizer;
        if let Ok(hex) = std::env::var("ROUTEPLANE_TOKENIZE_KEY_HEX") {
            match decode_hex_32(hex.trim()) {
                Some(bytes) => match Tokenizer::new(&bytes) {
                    Ok(t) => {
                        tracing::info!(
                            "reversible PII tokenization enabled (ADR-044): key loaded from ROUTEPLANE_TOKENIZE_KEY_HEX"
                        );
                        return Self(Some(Arc::new(t)));
                    }
                    Err(_) => {
                        tracing::info!(
                            "ROUTEPLANE_TOKENIZE_KEY_HEX present but invalid; tokenize mode will fall back to masking"
                        );
                        return Self(None);
                    }
                },
                None => {
                    tracing::info!(
                        "ROUTEPLANE_TOKENIZE_KEY_HEX present but not 64 hex chars; tokenize mode will fall back to masking"
                    );
                    return Self(None);
                }
            }
        }
        if let Ok(pass) = std::env::var("ROUTEPLANE_TOKENIZE_KEY") {
            let pass = pass.trim();
            if !pass.is_empty() {
                let bytes = sha256_32(pass.as_bytes());
                if let Ok(t) = Tokenizer::new(&bytes) {
                    tracing::info!(
                        "reversible PII tokenization enabled (ADR-044): key derived from ROUTEPLANE_TOKENIZE_KEY"
                    );
                    return Self(Some(Arc::new(t)));
                }
            }
        }
        // No key configured: ship-dark. tokenize mode degrades to masking.
        Self(None)
    }

    /// The tokenizer, if a key is in custody.
    #[must_use]
    pub fn tokenizer(&self) -> Option<&routeplane_guardrails_advanced::tokenize::Tokenizer> {
        self.0.as_deref()
    }
}

/// Decode exactly 32 bytes from a 64-char hex string. `None` on any non-hex char
/// or wrong length. Never logs the input. (Enterprise-only: used solely by
/// [`TokenizerKey::from_env`].)
#[cfg(feature = "enterprise")]
fn decode_hex_32(s: &str) -> Option<[u8; 32]> {
    if s.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    let b = s.as_bytes();
    for (i, slot) in out.iter_mut().enumerate() {
        let hi = (b[i * 2] as char).to_digit(16)?;
        let lo = (b[i * 2 + 1] as char).to_digit(16)?;
        *slot = (hi * 16 + lo) as u8;
    }
    Some(out)
}

/// SHA-256 a passphrase to 32 key bytes (dev-key derivation; reuses the ledger's
/// hashing dependency, no new crate). (Enterprise-only: used solely by
/// [`TokenizerKey::from_env`].)
#[cfg(feature = "enterprise")]
fn sha256_32(input: &[u8]) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(input);
    h.finalize().into()
}

pub struct GuardrailEngine;

impl Default for GuardrailEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl GuardrailEngine {
    pub fn new() -> Self {
        Self
    }

    /// Always-on masking: delegate to the deterministic detector library
    /// (`routeplane_guardrails::detect`, ADR-031 / PRD-036 Ring 1). This is a
    /// strict superset of the previous email+phone-only behavior — the
    /// `[EMAIL_MASKED]` / `[PHONE_MASKED]` placeholders stay byte-compatible,
    /// and we now also mask secrets, India PAN/Aadhaar (Verhoeff-gated), cards
    /// (Luhn-gated), SSN, and IPv4. Zero-copy + zero-alloc on clean input.
    pub fn process_text(&self, text: &str, config: &GuardrailConfig) -> String {
        if !config.mask_pii {
            return text.to_string();
        }
        routeplane_guardrails::detect::redact(text).into_owned()
    }
}

impl GuardrailConfig {
    /// The always-on irreversible-masking config (`mask_pii: true`, `pii_mode:
    /// Mask`) — the byte-identical legacy default used by every non-tokenize call
    /// site.
    #[must_use]
    pub fn masking() -> Self {
        Self {
            mask_pii: true,
            pii_mode: PiiMode::Mask,
        }
    }

    /// The reversible-tokenization config (`pii_mode: Tokenize`). Masking stays on
    /// as the fallback for non-reversible categories and when no key is in custody.
    #[must_use]
    pub fn tokenizing() -> Self {
        Self {
            mask_pii: true,
            pii_mode: PiiMode::Tokenize,
        }
    }
}

/// Hold-back window for [`StreamMasker`], in bytes.
///
/// Must exceed the longest span any recognizer can match, or an identifier could
/// straddle the boundary undetected. 256 comfortably covers every current
/// recognizer (Aadhaar 12, PAN 10, cards ≤19, SSN 11, IPv4 ≤15, emails, and the
/// provider API-key shapes) with room to spare. Raising it costs perceived
/// streaming latency — output is held back by at most this many bytes until
/// [`StreamMasker::flush`] — so it is a floor, not a target.
#[allow(dead_code)]
pub const STREAM_MASK_WINDOW_BYTES: usize = 256;

/// Hard ceiling on retained bytes before [`StreamMasker`] force-flushes.
///
/// Bounds memory per stream: without it, pathological input that keeps looking
/// like a straddling match could buffer a whole response.
#[allow(dead_code)]
const STREAM_MASK_MAX_CARRY_BYTES: usize = STREAM_MASK_WINDOW_BYTES * 4;

/// Stateful output masker that survives chunk boundaries ([PRD-071]).
///
/// # Why this exists
///
/// Per-chunk masking is blind to an identifier split across two SSE frames:
/// neither frame contains a complete match, so neither is redacted and the
/// client reassembles the original. That failure is silent and depends on where
/// the provider happened to split its tokens.
///
/// # How it stays correct
///
/// Masking is a **local substitution** over matches, so when no match straddles
/// the hold-back boundary, `mask(whole) == mask(settled) ++ mask(tail)`. The
/// masker exploits exactly that: it masks the settled prefix in isolation *and*
/// in context, and only emits when the two agree. Disagreement means a match
/// spans the boundary, so it emits nothing and waits for more input.
///
/// Because masking changes length, index arithmetic over the masked text is not
/// sound — hence the agreement check rather than a byte offset. A false negative
/// on the check costs one extra round of buffering; it can never leak.
///
/// # Contract
///
/// The caller MUST call [`flush`](Self::flush) when the upstream stream ends —
/// including on abort — or the retained tail is dropped and the response is
/// silently truncated.
///
/// [PRD-071]: https://github.com/routeplane-core/docs/blob/main/product/prd/071-streaming-guardrails-rolling-buffer.md
#[allow(dead_code)]
#[derive(Debug, Default)]
pub struct StreamMasker {
    /// Raw (unmasked) text retained until it can be proven settled.
    carry: String,
}

#[allow(dead_code)]
impl StreamMasker {
    #[must_use]
    pub fn new() -> Self {
        Self {
            carry: String::new(),
        }
    }

    /// Feed the next raw chunk; returns the masked text that is safe to emit now
    /// (often empty while a boundary is unsettled).
    ///
    /// `mask` is the masking function to apply — supplied by the caller so this
    /// type stays independent of the engine/config/capability plumbing.
    pub fn push<F>(&mut self, chunk: &str, mask: F) -> String
    where
        F: Fn(&str) -> String,
    {
        self.carry.push_str(chunk);

        if self.carry.len() <= STREAM_MASK_WINDOW_BYTES {
            return String::new();
        }

        // Force progress before unbounded growth: mask everything we hold and
        // start clean. Safe — we have the complete carry, so any match wholly
        // inside it is caught; only a match continuing into the *next* chunk can
        // be split, which is the pre-existing per-chunk behaviour and strictly
        // no worse.
        if self.carry.len() > STREAM_MASK_MAX_CARRY_BYTES {
            return self.flush(mask);
        }

        let split = floor_char_boundary(&self.carry, self.carry.len() - STREAM_MASK_WINDOW_BYTES);
        if split == 0 {
            return String::new();
        }

        let masked_settled = mask(&self.carry[..split]);
        let masked_full = mask(&self.carry);

        if masked_full.starts_with(&masked_settled) {
            self.carry.drain(..split);
            masked_settled
        } else {
            // A detection straddles the boundary — hold and wait for the rest.
            String::new()
        }
    }

    /// Mask and emit everything still retained. Call once when the stream ends.
    pub fn flush<F>(&mut self, mask: F) -> String
    where
        F: Fn(&str) -> String,
    {
        if self.carry.is_empty() {
            return String::new();
        }
        let out = mask(&self.carry);
        self.carry.clear();
        out
    }

    /// Bytes currently retained — for tests and bounded-memory assertions.
    #[must_use]
    pub fn buffered_len(&self) -> usize {
        self.carry.len()
    }
}

/// Largest `i <= idx` that is a UTF-8 char boundary of `s`.
///
/// `str::floor_char_boundary` is still unstable, and slicing mid-codepoint
/// panics — which on the streaming path would abort a live response.
#[allow(dead_code)]
fn floor_char_boundary(s: &str, idx: usize) -> usize {
    let mut i = idx.min(s.len());
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

#[cfg(test)]
mod tests {
    use super::*;

    fn engine() -> GuardrailEngine {
        GuardrailEngine::new()
    }

    #[test]
    fn masks_email_and_phone() {
        let cfg = GuardrailConfig::masking();
        let out = engine().process_text("mail a@b.com or call 415-555-0123", &cfg);
        assert!(out.contains("[EMAIL_MASKED]"));
        assert!(out.contains("[PHONE_MASKED]"));
        assert!(!out.contains("a@b.com"));
    }

    #[test]
    fn disabled_masking_is_passthrough() {
        let cfg = GuardrailConfig {
            mask_pii: false,
            ..Default::default()
        };
        let input = "mail a@b.com";
        assert_eq!(engine().process_text(input, &cfg), input);
    }

    #[test]
    fn name_field_is_maskable_via_process_text() {
        // The proxy masks Message.name by passing it through process_text (Task
        // #6); this confirms an email-shaped name would be masked.
        let cfg = GuardrailConfig::masking();
        let out = engine().process_text("user@corp.com", &cfg);
        assert_eq!(out, "[EMAIL_MASKED]");
    }

    #[test]
    fn now_masks_secrets_and_india_pii() {
        // The Ring-1 upgrade: a strict superset of email+phone.
        let cfg = GuardrailConfig::masking();
        let out = engine().process_text("PAN ABCDE1234F key sk-abcdefABCDEF0123456789", &cfg);
        assert!(out.contains("[PAN_MASKED]"));
        assert!(out.contains("[OPENAI_KEY_MASKED]"));
        assert!(!out.contains("ABCDE1234F"));
        assert!(!out.contains("sk-abcdef"));
    }

    #[test]
    fn clean_text_is_unchanged() {
        let cfg = GuardrailConfig::masking();
        let input = "a perfectly ordinary sentence";
        assert_eq!(engine().process_text(input, &cfg), input);
    }

    // ---- StreamMasker (PRD-071) ------------------------------------------
    //
    // Padding exceeds STREAM_MASK_WINDOW_BYTES so the settled/carry split is
    // actually exercised; without it everything buffers to `flush` and the
    // boundary logic is never reached.

    fn masker_fn() -> impl Fn(&str) -> String {
        let eng = GuardrailEngine::new();
        let cfg = GuardrailConfig::masking();
        move |s: &str| eng.process_text(s, &cfg)
    }

    fn drive(chunks: &[&str]) -> String {
        let mask = masker_fn();
        let mut m = StreamMasker::new();
        let mut out = String::new();
        for c in chunks {
            out.push_str(&m.push(c, &mask));
        }
        out.push_str(&m.flush(&mask));
        out
    }

    #[test]
    fn pii_split_across_chunk_boundary_is_masked() {
        // The regression this type exists for: per-chunk masking misses this.
        let pad = "x".repeat(400);
        let text = format!("{pad} PAN ABCDE1234F {pad}");
        // Split squarely inside the PAN.
        let cut = text.find("ABCDE1234F").unwrap() + 4;
        let out = drive(&[&text[..cut], &text[cut..]]);
        assert!(!out.contains("ABCDE1234F"), "raw PAN survived the boundary");
        assert!(out.contains("[PAN_MASKED]"));
    }

    #[test]
    fn masked_at_every_split_point_of_the_identifier() {
        let pad = "x".repeat(400);
        let text = format!("{pad} PAN ABCDE1234F {pad}");
        let start = text.find("ABCDE1234F").unwrap();
        for cut in start..start + "ABCDE1234F".len() {
            let out = drive(&[&text[..cut], &text[cut..]]);
            assert!(!out.contains("ABCDE1234F"), "leaked at split {cut}");
        }
    }

    #[test]
    fn clean_stream_round_trips_byte_identically() {
        // Backward compatibility: no PII ⇒ output is the concatenated input.
        let text = "y".repeat(1000);
        let out = drive(&[&text[..300], &text[300..700], &text[700..]]);
        assert_eq!(out, text);
    }

    #[test]
    fn nothing_is_lost_without_flush_being_called_twice() {
        let mask = masker_fn();
        let mut m = StreamMasker::new();
        let text = "z".repeat(600);
        let mut out = m.push(&text, &mask);
        out.push_str(&m.flush(&mask));
        assert_eq!(out, text);
        // A second flush is a no-op, not a duplicate emission.
        assert!(m.flush(&mask).is_empty());
        assert_eq!(m.buffered_len(), 0);
    }

    #[test]
    fn retained_bytes_stay_bounded() {
        let mask = masker_fn();
        let mut m = StreamMasker::new();
        for _ in 0..50 {
            let _ = m.push(&"q".repeat(200), &mask);
            assert!(
                m.buffered_len() <= STREAM_MASK_MAX_CARRY_BYTES,
                "carry grew past the ceiling: {}",
                m.buffered_len()
            );
        }
    }

    #[test]
    fn multibyte_boundaries_do_not_panic_or_corrupt() {
        // Slicing mid-codepoint panics, which on the streaming path would abort a
        // live response. Mixed 1/2/3-byte units guarantee the masker's internal
        // `len - WINDOW` offset lands mid-codepoint and has to be floored.
        let unit = "aé€"; // 6 bytes
        let text = unit.repeat(200); // 1200 bytes
        let cut = unit.len() * 100; // a real char boundary, so the test itself is safe
        let out = drive(&[&text[..cut], &text[cut..]]);
        assert_eq!(out, text);
    }

    #[test]
    fn floor_char_boundary_never_splits_a_codepoint() {
        let s = "aé€";
        for i in 0..=s.len() {
            let f = floor_char_boundary(s, i);
            assert!(
                s.is_char_boundary(f),
                "offset {f} from {i} is not a boundary"
            );
            assert!(f <= i);
        }
    }
}
