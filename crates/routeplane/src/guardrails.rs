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
}
