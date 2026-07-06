//! Console account store (Community Edition Console email+password auth).
//!
//! Mirrors `custom_providers.rs` exactly: an [`ArcSwap`]-backed immutable
//! snapshot read lock-free on the request path, persisted to
//! `configs/console-accounts.json` (0600, atomic temp-file + fsync + rename,
//! gitignored + dockerignored — it holds password HASHES), loaded once at boot
//! and hot-swapped whole on every mutation. ABSENT/empty file ⇒ start empty
//! (signup creates the first account); PRESENT-but-invalid ⇒ refuse startup
//! (the `keys.json` fail-closed doctrine).
//!
//! **Secret handling:** passwords are argon2id-hashed ([`hash_password`]) the
//! moment they arrive and NEVER stored, logged, or echoed in plaintext. The
//! stored PHC hash string is itself write-only over the API (no endpoint
//! returns it) and the account struct deliberately has no `Debug` derive so a
//! stray `{:?}` cannot leak it.

use arc_swap::ArcSwap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

/// One console account, as persisted to `configs/console-accounts.json`.
///
/// NO `Debug` derive — `password_hash` must never ride a `{:?}`.
#[derive(Clone, Serialize, Deserialize)]
pub struct ConsoleAccount {
    /// Lowercased, unique (the store's key). Shape-validated at signup/load.
    pub email: String,
    /// argon2id PHC string (`$argon2id$v=19$…`). NEVER logged or returned.
    pub password_hash: String,
    /// RFC-3339 creation stamp; set server-side at signup.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
    /// Monotonic per-account session version. Every session JWT is minted with
    /// the CURRENT value; logout bumps it (persisted), instantly invalidating
    /// every outstanding token for this account — real revocation without a
    /// session table. Serde-default 0 so a legacy record deserialises.
    #[serde(default)]
    pub session_version: u64,
}

/// An immutable snapshot of the account registry. Swapped whole on mutation.
#[derive(Default)]
struct Accounts {
    by_email: HashMap<String, ConsoleAccount>,
}

impl Accounts {
    /// Build a snapshot, enforcing the load-time invariants: emails must be
    /// non-empty, already-lowercase, and unique; hashes must be non-empty.
    /// Error strings never carry a hash.
    fn build(accounts: Vec<ConsoleAccount>) -> Result<Self, String> {
        let mut by_email: HashMap<String, ConsoleAccount> = HashMap::with_capacity(accounts.len());
        for account in accounts {
            if account.email.is_empty() || account.email != account.email.to_ascii_lowercase() {
                return Err(format!(
                    "invalid account email '{}' (must be non-empty and lowercase)",
                    account.email
                ));
            }
            if account.password_hash.is_empty() {
                return Err(format!(
                    "account '{}' has an empty password hash",
                    account.email
                ));
            }
            if by_email.contains_key(&account.email) {
                return Err(format!("duplicate account email '{}'", account.email));
            }
            by_email.insert(account.email.clone(), account);
        }
        Ok(Self { by_email })
    }

    fn sorted(&self) -> Vec<ConsoleAccount> {
        let mut accounts: Vec<ConsoleAccount> = self.by_email.values().cloned().collect();
        accounts.sort_by(|a, b| a.email.cmp(&b.email));
        accounts
    }
}

/// The persisted file shape (`configs/console-accounts.json`), mirroring the
/// `{"providers":[…]}` convention of `configs/providers.json`.
#[derive(Serialize, Deserialize, Default)]
struct PersistFile {
    #[serde(default)]
    accounts: Vec<ConsoleAccount>,
}

/// Typed create failure: the duplicate case gets its own variant so the
/// handler can render a 409 instead of a 500.
pub enum CreateError {
    /// An account with this email already exists.
    Duplicate,
    /// A build/persist failure (message never carries a hash).
    Store(String),
}

/// The console account store: lock-free snapshot reads + serialized,
/// persist-then-swap mutations (the `CustomProviderStore` structure verbatim).
pub struct ConsoleAccountStore {
    shared: Arc<ArcSwap<Accounts>>,
    /// `None` ⇒ in-memory only (tests); `Some` ⇒ atomic 0600 file persistence.
    path: Option<PathBuf>,
    /// Serializes mutations ONLY — never touched on the request path.
    write_lock: tokio::sync::Mutex<()>,
}

impl ConsoleAccountStore {
    /// Boot-time load. Absent or blank file ⇒ empty store; present-but-invalid
    /// ⇒ `Err` (the caller refuses startup — fail-closed, the keys.json
    /// doctrine). Error strings never carry a password hash value beyond the
    /// serde position info.
    pub fn load(path: PathBuf) -> Result<Self, String> {
        let snapshot = if path.exists() {
            let raw = std::fs::read_to_string(&path)
                .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
            if raw.trim().is_empty() {
                Accounts::default()
            } else {
                let file: PersistFile = serde_json::from_str(&raw)
                    .map_err(|e| format!("invalid console-accounts JSON: {e}"))?;
                Accounts::build(file.accounts)?
            }
        } else {
            Accounts::default()
        };
        Ok(Self {
            shared: Arc::new(ArcSwap::from_pointee(snapshot)),
            path: Some(path),
            write_lock: tokio::sync::Mutex::new(()),
        })
    }

    /// In-memory-only store (tests): mutations skip persistence.
    #[cfg(test)]
    pub fn ephemeral() -> Self {
        Self {
            shared: Arc::new(ArcSwap::from_pointee(Accounts::default())),
            path: None,
            write_lock: tokio::sync::Mutex::new(()),
        }
    }

    // ---- request-path reads (one wait-free ArcSwap load + HashMap probe) ----

    /// Look up an account by (already-normalized) email. A plain `HashMap`
    /// probe is fine here: the email is an IDENTIFIER, not the credential —
    /// the secret comparison happens in the constant-time argon2 verify /
    /// HMAC signature check, never on this lookup.
    pub fn find(&self, email: &str) -> Option<ConsoleAccount> {
        self.shared.load().by_email.get(email).cloned()
    }

    pub fn len(&self) -> usize {
        self.shared.load().by_email.len()
    }

    /// Companion to `len` (clippy `len_without_is_empty`); used by tests.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    // ---- mutations (signup/logout path ONLY; persist-then-swap) ----

    /// Create a new account. `email` must already be normalized
    /// ([`normalize_email`]) and `password_hash` a PHC string from
    /// [`hash_password`]. Sets `created_at`/`session_version` server-side.
    /// Duplicate email ⇒ [`CreateError::Duplicate`]. Persist FIRST, swap
    /// second — a reader never sees an account that is not durable.
    pub async fn create(
        &self,
        email: String,
        password_hash: String,
    ) -> Result<ConsoleAccount, CreateError> {
        let _g = self.write_lock.lock().await;
        let snapshot = self.shared.load();
        if snapshot.by_email.contains_key(&email) {
            return Err(CreateError::Duplicate);
        }
        let account = ConsoleAccount {
            email,
            password_hash,
            created_at: Some(chrono::Utc::now().to_rfc3339()),
            session_version: 0,
        };
        let mut accounts = snapshot.sorted();
        drop(snapshot);
        accounts.push(account.clone());
        accounts.sort_by(|a, b| a.email.cmp(&b.email));
        let next = Accounts::build(accounts.clone()).map_err(CreateError::Store)?;
        self.persist(accounts).await.map_err(CreateError::Store)?;
        self.shared.store(Arc::new(next));
        Ok(account)
    }

    /// Bump an account's session version (logout ⇒ revoke every outstanding
    /// session token for the account). Returns the NEW version, or `Ok(None)`
    /// when the email is unknown (idempotent from the caller's view).
    pub async fn bump_session_version(&self, email: &str) -> Result<Option<u64>, String> {
        let _g = self.write_lock.lock().await;
        let mut accounts = self.shared.load().sorted();
        let Some(account) = accounts.iter_mut().find(|a| a.email == email) else {
            return Ok(None);
        };
        account.session_version = account.session_version.wrapping_add(1);
        let new_version = account.session_version;
        let next = Accounts::build(accounts.clone())?;
        self.persist(accounts).await?;
        self.shared.store(Arc::new(next));
        Ok(Some(new_version))
    }

    /// Persist the full registry atomically (temp 0600 + fsync + rename) on a
    /// blocking pool worker — the shared helper from `custom_providers.rs`.
    async fn persist(&self, accounts: Vec<ConsoleAccount>) -> Result<(), String> {
        let Some(path) = self.path.clone() else {
            return Ok(());
        };
        let bytes = serde_json::to_vec_pretty(&PersistFile { accounts })
            .map_err(|e| format!("serialize console accounts: {e}"))?;
        tokio::task::spawn_blocking(move || crate::custom_providers::write_atomic(&path, &bytes))
            .await
            .map_err(|e| format!("persist task join error: {e}"))?
    }
}

// ---------------------------------------------------------------------------
// Password hashing (argon2id) + input shape validation
// ---------------------------------------------------------------------------

/// argon2id-hash a password into a PHC string (random 16-byte salt via the OS
/// CSPRNG). Uses the `Argon2::default()` parameters (argon2id v19, the OWASP-
/// recommended memory-hard defaults). CPU-bound ~tens of ms — callers run it on
/// `spawn_blocking`, never inline on a runtime worker. The error string never
/// contains the password.
pub fn hash_password(password: &str) -> Result<String, String> {
    use argon2::password_hash::SaltString;
    use argon2::{Argon2, PasswordHasher};
    let mut salt_bytes = [0u8; 16];
    getrandom::getrandom(&mut salt_bytes).map_err(|e| format!("csprng failure: {e}"))?;
    let salt =
        SaltString::encode_b64(&salt_bytes).map_err(|_| "salt encoding failed".to_string())?;
    Argon2::default()
        .hash_password(password.as_bytes(), &salt)
        .map(|h| h.to_string())
        .map_err(|_| "password hashing failed".to_string())
}

/// Constant-time argon2id verification (the `argon2` crate's verify is
/// constant-time over the digest comparison). A malformed stored hash verifies
/// `false` — never panics, never leaks why. CPU-bound like [`hash_password`];
/// callers use `spawn_blocking`.
pub fn verify_password(password: &str, phc: &str) -> bool {
    use argon2::password_hash::PasswordHash;
    use argon2::{Argon2, PasswordVerifier};
    match PasswordHash::new(phc) {
        Ok(parsed) => Argon2::default()
            .verify_password(password.as_bytes(), &parsed)
            .is_ok(),
        Err(_) => false,
    }
}

/// Normalize + shape-validate an email: trim, lowercase, ≤254 chars, no
/// whitespace/control chars, exactly one `@` with a non-empty local part and a
/// dotted domain. SHAPE validation only (no deliverability check — this is a
/// self-host operator login, not a mailing list). The error string never
/// echoes the input.
pub fn normalize_email(raw: &str) -> Result<String, &'static str> {
    let email = raw.trim().to_ascii_lowercase();
    if email.is_empty() || email.len() > 254 {
        return Err("email must be 1-254 characters");
    }
    if email.chars().any(|c| c.is_whitespace() || c.is_control()) {
        return Err("email must not contain whitespace or control characters");
    }
    let Some((local, domain)) = email.split_once('@') else {
        return Err("email must contain an '@'");
    };
    if local.is_empty() || local.len() > 64 {
        return Err("email local part must be 1-64 characters");
    }
    if domain.is_empty() || domain.contains('@') {
        return Err("email must contain exactly one '@'");
    }
    if !domain.contains('.') || domain.starts_with('.') || domain.ends_with('.') {
        return Err("email domain must be a dotted hostname");
    }
    Ok(email)
}

/// Password policy: ≥ 10 characters (the signup contract), ≤ 512 bytes (bounds
/// the argon2 work per unauthenticated request — a cheap DoS guard). The error
/// string never echoes the password.
pub fn validate_password(password: &str) -> Result<(), &'static str> {
    if password.chars().count() < 10 {
        return Err("password must be at least 10 characters");
    }
    if password.len() > 512 {
        return Err("password must be at most 512 bytes");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_accounts_path(tag: &str) -> PathBuf {
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "rp_console_accounts_{tag}_{}_{seq}.json",
            std::process::id()
        ))
    }

    #[test]
    fn argon2id_round_trip_and_wrong_password_rejected() {
        let hash = hash_password("correct horse battery").expect("hash");
        assert!(hash.starts_with("$argon2id$"), "must be argon2id PHC");
        assert!(
            !hash.contains("correct horse battery"),
            "no plaintext in PHC"
        );
        assert!(verify_password("correct horse battery", &hash));
        assert!(!verify_password("wrong horse battery", &hash));
        assert!(!verify_password(
            "correct horse battery",
            "not-a-phc-string"
        ));
        // Two hashes of the same password differ (random salt).
        let hash2 = hash_password("correct horse battery").expect("hash");
        assert_ne!(hash, hash2);
    }

    #[test]
    fn email_normalization_and_shape_validation() {
        assert_eq!(
            normalize_email("  Admin@Example.COM ").unwrap(),
            "admin@example.com"
        );
        assert!(normalize_email("").is_err());
        assert!(normalize_email("no-at-sign").is_err());
        assert!(normalize_email("@example.com").is_err());
        assert!(normalize_email("a@").is_err());
        assert!(normalize_email("a@b").is_err()); // no dot in domain
        assert!(normalize_email("a@.com").is_err());
        assert!(normalize_email("a@b.com@c.com").is_err()); // two @
        assert!(normalize_email("a b@example.com").is_err()); // whitespace
    }

    #[test]
    fn password_policy_bounds() {
        assert!(validate_password("short-pw").is_err()); // 8 < 10
        assert!(validate_password("exactly-10").is_ok());
        assert!(validate_password(&"x".repeat(513)).is_err());
    }

    #[tokio::test]
    async fn create_find_duplicate_and_bump_in_memory() {
        let store = ConsoleAccountStore::ephemeral();
        assert!(store.is_empty());
        assert!(store.find("op@example.com").is_none());

        let account = store
            .create("op@example.com".into(), "$argon2id$fakehash".into())
            .await
            .map_err(|_| "create")
            .expect("first create succeeds");
        assert_eq!(account.session_version, 0);
        assert!(account.created_at.is_some());
        assert_eq!(store.len(), 1);

        // Duplicate email is rejected with the typed variant.
        assert!(matches!(
            store
                .create("op@example.com".into(), "$argon2id$other".into())
                .await,
            Err(CreateError::Duplicate)
        ));

        // Logout revocation: the bump is visible to the next find().
        assert_eq!(
            store.bump_session_version("op@example.com").await.unwrap(),
            Some(1)
        );
        assert_eq!(store.find("op@example.com").unwrap().session_version, 1);
        // Unknown email is a no-op, not an error.
        assert_eq!(
            store
                .bump_session_version("ghost@example.com")
                .await
                .unwrap(),
            None
        );
    }

    #[tokio::test]
    async fn persistence_round_trips_atomically_with_0600_perms() {
        let path = temp_accounts_path("persist");
        let _ = std::fs::remove_file(&path);

        let store = ConsoleAccountStore::load(path.clone()).expect("absent file loads empty");
        assert!(store.is_empty());
        store
            .create("op@example.com".into(), "$argon2id$fakehash".into())
            .await
            .map_err(|_| "create")
            .expect("create");

        assert!(path.exists());
        assert!(!path.with_file_name("console-accounts.json.tmp").exists());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600, "accounts file must be 0600 (holds hashes)");
        }

        // A restart sees the same account AND the persisted session version.
        store.bump_session_version("op@example.com").await.unwrap();
        let reloaded = ConsoleAccountStore::load(path.clone()).expect("reload");
        let account = reloaded.find("op@example.com").expect("account survives");
        assert_eq!(account.session_version, 1);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn malformed_or_duplicate_accounts_file_is_fail_closed() {
        let path = temp_accounts_path("malformed");
        std::fs::write(&path, "{ not json").unwrap();
        assert!(ConsoleAccountStore::load(path.clone()).is_err());

        // Duplicate emails in the file refuse to load.
        std::fs::write(
            &path,
            r#"{"accounts":[
                {"email":"a@b.co","password_hash":"$x"},
                {"email":"a@b.co","password_hash":"$y"}
            ]}"#,
        )
        .unwrap();
        assert!(ConsoleAccountStore::load(path.clone()).is_err());

        // A non-lowercase email in the file refuses to load (the store's
        // uniqueness key is the lowercased form).
        std::fs::write(
            &path,
            r#"{"accounts":[{"email":"Admin@b.co","password_hash":"$x"}]}"#,
        )
        .unwrap();
        assert!(ConsoleAccountStore::load(path.clone()).is_err());
        let _ = std::fs::remove_file(&path);
    }
}
