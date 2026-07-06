//! Console session bridge (Community Edition Console email+password auth).
//!
//! A valid login mints a signed **HS256 JWT** (via `jsonwebtoken`) that the
//! Console SPA presents as `Authorization: Bearer <jwt>` — header-based,
//! matching the gateway's existing no-cookie/CORS model (no cookies ⇒ no CSRF
//! surface; an httpOnly-cookie hardening pass is a documented follow-up). The
//! browser only ever holds this short-lived session token; the `rp_` gateway
//! key NEVER leaves the server.
//!
//! `auth_middleware` (auth.rs) consults [`ConsoleAuthBridge::verify`] strictly
//! as a FALLBACK after the rp_-key paths decline: signature + `exp` first
//! (constant-time HMAC inside `ring`; algorithm pinned to HS256 so `none`/RSA
//! confusion is structurally rejected), then the account's live
//! `session_version` (logout revocation), and only then does the request
//! resolve to the CONFIGURED console gateway key — the single-tenant CE
//! authorization bridge.
//!
//! Signing secret: `RP_CONSOLE_SESSION_SECRET` (stable across restarts) or a
//! per-boot random 32-byte secret from the OS CSPRNG (sessions reset on
//! restart — `main` warns). The secret is never logged.

use crate::console_accounts::ConsoleAccountStore;
use jsonwebtoken::{Algorithm, DecodingKey, EncodingKey, Validation};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Session lifetime: ~12h (the task contract). A fixed constant — a config
/// knob would be surface without a driver; revisit if a customer asks.
pub const SESSION_TTL_SECS: u64 = 12 * 60 * 60;

/// The signed session claims. `sub` = the account email (lowercased); `sv` =
/// the account's session version at mint time (logout revocation seam).
#[derive(Serialize, Deserialize)]
struct ConsoleClaims {
    sub: String,
    iat: u64,
    exp: u64,
    #[serde(default)]
    sv: u64,
}

/// The per-request console identity, inserted as a request extension by
/// `auth_middleware` on a successful session auth (alongside the resolved
/// `VirtualKey`/`TenantContext`). `/v1/console/me` and `/v1/console/logout`
/// extract it; its ABSENCE on an rp_-key-authed request is what makes those
/// endpoints session-only.
#[derive(Clone, Debug)]
pub struct ConsoleSession {
    pub email: String,
    pub created_at: Option<String>,
}

/// Shared handle injected as a request extension (the `SharedAuthState`
/// pattern): built once at boot, cloned per router.
pub type SharedConsoleAuth = Arc<ConsoleAuthBridge>;

/// Everything the session seam needs: the HS256 key pair, the pinned
/// validation, the account store (revocation lookups), and the gateway key a
/// valid session authorizes as. NO `Debug` derive — holds key material.
pub struct ConsoleAuthBridge {
    encoding: EncodingKey,
    decoding: DecodingKey,
    validation: Validation,
    /// The `routeplane_key` a valid session resolves to (single-tenant CE).
    /// Server-side CONFIG, not a request-supplied secret — the registry probe
    /// on it needs no constant-time treatment.
    pub console_key: String,
    /// The account store (revocation + /me lookups + signup/logout mutations).
    pub accounts: Arc<ConsoleAccountStore>,
}

impl ConsoleAuthBridge {
    pub fn new(secret: &[u8], console_key: String, accounts: Arc<ConsoleAccountStore>) -> Self {
        // Pin the accepted algorithm set to exactly HS256: `none`, RS256-key-
        // confusion, and downgrade tokens are rejected before any claim is read.
        // `exp` is required + validated by default (60s default leeway).
        let validation = Validation::new(Algorithm::HS256);
        Self {
            encoding: EncodingKey::from_secret(secret),
            decoding: DecodingKey::from_secret(secret),
            validation,
            console_key,
            accounts,
        }
    }

    /// Mint a session token for an (already-authenticated) account. The error
    /// string never carries the secret or the token.
    pub fn issue(&self, email: &str, session_version: u64) -> Result<String, String> {
        let now = chrono::Utc::now().timestamp().max(0) as u64;
        let claims = ConsoleClaims {
            sub: email.to_string(),
            iat: now,
            exp: now.saturating_add(SESSION_TTL_SECS),
            sv: session_version,
        };
        jsonwebtoken::encode(
            &jsonwebtoken::Header::new(Algorithm::HS256),
            &claims,
            &self.encoding,
        )
        .map_err(|_| "failed to sign session token".to_string())
    }

    /// Verify a presented session token. `None` on ANY failure (bad signature,
    /// wrong algorithm, expired, unknown account, stale session version) —
    /// deliberately reason-free so the caller's 401 stays generic. Pure CPU +
    /// one lock-free `ArcSwap` probe: hot-path-safe, no locks, no I/O.
    pub fn verify(&self, token: &str) -> Option<ConsoleSession> {
        let data =
            jsonwebtoken::decode::<ConsoleClaims>(token, &self.decoding, &self.validation).ok()?;
        let account = self.accounts.find(&data.claims.sub)?;
        // Logout revocation: a token minted before the last bump is dead.
        if account.session_version != data.claims.sv {
            return None;
        }
        Some(ConsoleSession {
            email: account.email,
            created_at: account.created_at,
        })
    }
}

/// Resolve the session-signing secret: `RP_CONSOLE_SESSION_SECRET` when set
/// (trimmed, non-empty), else 32 random bytes from the OS CSPRNG. Returns
/// `(secret, generated)`; a CSPRNG failure is a boot-fatal `Err` (a guessable
/// session secret is an auth bypass — fail closed, never fall back to a weak
/// source). The secret value is never logged by any caller.
pub fn session_secret_from_env() -> Result<(Vec<u8>, bool), String> {
    match std::env::var("RP_CONSOLE_SESSION_SECRET") {
        Ok(s) if !s.trim().is_empty() => Ok((s.trim().as_bytes().to_vec(), false)),
        _ => {
            let mut buf = [0u8; 32];
            getrandom::getrandom(&mut buf).map_err(|e| format!("csprng failure: {e}"))?;
            Ok((buf.to_vec(), true))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::console_accounts::ConsoleAccountStore;

    async fn bridge_with_account(secret: &[u8]) -> ConsoleAuthBridge {
        let accounts = Arc::new(ConsoleAccountStore::ephemeral());
        accounts
            .create("op@example.com".into(), "$argon2id$fakehash".into())
            .await
            .map_err(|_| "create")
            .expect("create");
        ConsoleAuthBridge::new(secret, "rp_console".into(), accounts)
    }

    #[tokio::test]
    async fn issue_verify_round_trip() {
        let bridge = bridge_with_account(b"test-secret-0123456789").await;
        let token = bridge.issue("op@example.com", 0).expect("issue");
        let session = bridge.verify(&token).expect("valid session verifies");
        assert_eq!(session.email, "op@example.com");
        assert!(session.created_at.is_some());
    }

    #[tokio::test]
    async fn wrong_secret_and_garbage_tokens_rejected() {
        let bridge = bridge_with_account(b"secret-a-0123456789").await;
        let other = bridge_with_account(b"secret-b-0123456789").await;
        let token = other.issue("op@example.com", 0).expect("issue");
        assert!(
            bridge.verify(&token).is_none(),
            "cross-secret token rejected"
        );
        assert!(bridge.verify("not.a.jwt").is_none());
        assert!(bridge.verify("").is_none());
    }

    #[tokio::test]
    async fn expired_token_rejected() {
        let bridge = bridge_with_account(b"test-secret-0123456789").await;
        // Hand-craft a token whose exp is an hour in the past (well beyond the
        // 60s default leeway), signed with the SAME secret.
        let now = chrono::Utc::now().timestamp().max(0) as u64;
        let claims = ConsoleClaims {
            sub: "op@example.com".into(),
            iat: now - 7200,
            exp: now - 3600,
            sv: 0,
        };
        let token = jsonwebtoken::encode(
            &jsonwebtoken::Header::new(Algorithm::HS256),
            &claims,
            &EncodingKey::from_secret(b"test-secret-0123456789"),
        )
        .expect("encode");
        assert!(bridge.verify(&token).is_none(), "expired token rejected");
    }

    #[tokio::test]
    async fn unknown_account_and_stale_session_version_rejected() {
        let bridge = bridge_with_account(b"test-secret-0123456789").await;
        // A perfectly signed token for an email with no account.
        let token = bridge.issue("ghost@example.com", 0).expect("issue");
        assert!(bridge.verify(&token).is_none(), "unknown account rejected");

        // Logout revocation: bump the version, the old token dies.
        let token = bridge.issue("op@example.com", 0).expect("issue");
        assert!(bridge.verify(&token).is_some());
        bridge
            .accounts
            .bump_session_version("op@example.com")
            .await
            .expect("bump");
        assert!(
            bridge.verify(&token).is_none(),
            "token minted before logout is revoked"
        );
        // A re-login (current version) works again.
        let fresh = bridge.issue("op@example.com", 1).expect("issue");
        assert!(bridge.verify(&fresh).is_some());
    }

    #[test]
    fn secret_from_env_generates_32_random_bytes_when_unset() {
        // NOTE: reads the ambient env — RP_CONSOLE_SESSION_SECRET is not set in
        // the test environment (we deliberately avoid set_var: process-global,
        // racy under parallel tests).
        let (secret, generated) = session_secret_from_env().expect("csprng");
        if generated {
            assert_eq!(secret.len(), 32);
            let (second, _) = session_secret_from_env().expect("csprng");
            assert_ne!(secret, second, "two generated secrets must differ");
        } else {
            assert!(!secret.is_empty());
        }
    }
}
