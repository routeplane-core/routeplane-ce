//! Runtime custom-provider registry (Community Edition operator surface).
//!
//! An operator adds a custom **OpenAI-compatible** provider (name + base URL +
//! upstream API key + model ids) via the authed `/v1/providers` API and can use
//! it immediately — no restart. The registry is:
//!
//! * **Lock-free on the hot path** — the live snapshot rides an [`ArcSwap`]
//!   (the same posture as `SharedAuthState` / `SharedFxRates` /
//!   `SharedPolicyRegistry`); every request-path read is one wait-free
//!   `load()` + a `HashMap` probe. An EMPTY registry (the default) makes every
//!   probe a miss, so the ship-dark path is byte-identical to today.
//! * **Persisted to `configs/providers.json`** (a JSON file, NOT a database —
//!   the no-DB constraint holds): loaded once at boot; every mutation writes
//!   the whole file atomically (temp file + rename) with restrictive 0600
//!   permissions — it holds upstream secrets, exactly like `configs/keys.json`
//!   (both are gitignored + dockerignored). ABSENT/empty file ⇒ start empty;
//!   PRESENT-but-invalid ⇒ refuse startup (the keys.json fail-closed doctrine).
//! * **Mutated off the hot path** — upsert/delete serialize behind a
//!   `tokio::sync::Mutex` held ONLY by the admin endpoints (never the chat
//!   path), persist via `spawn_blocking` (no blocking file I/O on a runtime
//!   worker), and only then swap the `ArcSwap` — so a reader never observes an
//!   entry that is not durable.
//!
//! **Secret handling:** the `api_key` is WRITE-ONLY over the API. The only
//! egress points for the raw key are (a) the 0600 persistence file and (b) the
//! `Authorization: Bearer` header to the provider's own upstream at dispatch.
//! Every listable view ([`ProviderView`]) carries the masked form; the config
//! struct deliberately has no `Debug` derive so a stray `{:?}` cannot leak it.

use arc_swap::ArcSwap;
use routeplane_adapters::openai_compatible::SelfHostedProvider;
use routeplane_adapters::Provider;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// One operator-defined OpenAI-compatible provider, as authored over the API
/// and as persisted to `configs/providers.json`.
///
/// NO `Debug` derive — `api_key` is a secret and must never ride a `{:?}`.
#[derive(Clone, Serialize, Deserialize)]
pub struct CustomProviderConfig {
    /// Registry key + the `x-routeplane-provider` routing name
    /// (`^[a-z0-9_-]{1,64}$`; built-in names are rejected at registration).
    pub name: String,
    /// Base URL of the OpenAI-compatible upstream (http/https, host required;
    /// trailing `/` stripped). The adapter appends `/v1/chat/completions` etc.
    pub base_url: String,
    /// Upstream API key, sent as `Authorization: Bearer <key>` to `base_url`.
    /// WRITE-ONLY over the gateway API: never returned raw, never logged.
    pub api_key: String,
    /// Model ids this provider serves. A chat request whose `model` matches one
    /// (and names no provider/config explicitly) routes here.
    pub models: Vec<String>,
    /// RFC-3339 creation stamp; set server-side at create, preserved on update.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
}

/// The masked, API-safe view of a provider — the ONLY shape the read endpoints
/// return. `api_key` here is ALWAYS the masked form (see [`mask_key`]).
#[derive(Clone, Serialize)]
pub struct ProviderView {
    /// Always the literal `"provider"` (OpenAI-style discriminator).
    pub object: &'static str,
    pub name: String,
    pub base_url: String,
    /// MASKED (`…last4`) — never the raw key.
    pub api_key: String,
    pub models: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
}

/// Mask an upstream key for display: keep the last 4 characters when the key is
/// long enough to keep that safe (≥ 8 chars), otherwise hide it entirely.
/// Char-boundary-safe (never byte-slices, so a multi-byte input cannot panic).
pub fn mask_key(key: &str) -> String {
    let n = key.chars().count();
    if n >= 8 {
        let last4: String = key.chars().skip(n - 4).collect();
        format!("…{last4}")
    } else {
        "…".to_string()
    }
}

fn view_of(cfg: &CustomProviderConfig) -> ProviderView {
    ProviderView {
        object: "provider",
        name: cfg.name.clone(),
        base_url: cfg.base_url.clone(),
        api_key: mask_key(&cfg.api_key),
        models: cfg.models.clone(),
        created_at: cfg.created_at.clone(),
    }
}

/// Syntactic validation + normalization for an authored provider config.
/// Returns `(param, message)` for the OpenAI-shaped 400 on failure. Reserved
/// (built-in) name collisions are checked by the handler, which owns the
/// built-in registry. The message never contains the api_key.
pub fn validate_and_normalize(cfg: &mut CustomProviderConfig) -> Result<(), (String, String)> {
    // --- name: ^[a-z0-9_-]{1,64}$ ---
    cfg.name = cfg.name.trim().to_string();
    if cfg.name.is_empty() || cfg.name.len() > 64 {
        return Err((
            "name".into(),
            "provider name must be 1-64 characters of [a-z0-9_-]".into(),
        ));
    }
    if !cfg
        .name
        .bytes()
        .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'-')
    {
        return Err((
            "name".into(),
            "provider name must match ^[a-z0-9_-]+$ (lowercase)".into(),
        ));
    }
    // --- base_url: http/https + host, no credentials/query/fragment ---
    cfg.base_url = cfg.base_url.trim().trim_end_matches('/').to_string();
    let parsed = url::Url::parse(&cfg.base_url).map_err(|_| {
        (
            "base_url".to_string(),
            "base_url must be a valid URL".to_string(),
        )
    })?;
    if parsed.scheme() != "http" && parsed.scheme() != "https" {
        return Err((
            "base_url".into(),
            "base_url scheme must be http or https".into(),
        ));
    }
    if parsed.host_str().map(str::is_empty).unwrap_or(true) {
        return Err(("base_url".into(), "base_url must include a host".into()));
    }
    if !parsed.username().is_empty() || parsed.password().is_some() {
        return Err((
            "base_url".into(),
            "base_url must not embed credentials".into(),
        ));
    }
    if parsed.query().is_some() || parsed.fragment().is_some() {
        return Err((
            "base_url".into(),
            "base_url must not carry a query string or fragment".into(),
        ));
    }
    // --- api_key: required (self-hosted servers that need no key still accept
    //     any placeholder bearer), bounded ---
    if cfg.api_key.is_empty() || cfg.api_key.len() > 512 {
        return Err((
            "api_key".into(),
            "api_key is required (1-512 characters)".into(),
        ));
    }
    // --- models: 1..=128 non-empty, deduplicated ids ---
    let mut models: Vec<String> = Vec::with_capacity(cfg.models.len());
    for m in &cfg.models {
        let m = m.trim();
        if m.is_empty() || m.len() > 256 {
            return Err((
                "models".into(),
                "each model id must be 1-256 non-blank characters".into(),
            ));
        }
        if !models.iter().any(|e| e == m) {
            models.push(m.to_string());
        }
    }
    if models.is_empty() || models.len() > 128 {
        return Err(("models".into(), "models must list 1-128 model ids".into()));
    }
    cfg.models = models;
    Ok(())
}

// --- SSRF guard on the operator-supplied base_url -------------------------------
//
// A custom `base_url` is an outbound request the gateway makes on the operator's
// behalf — an SSRF primitive if it can be pointed at the cloud metadata endpoint
// (169.254.169.254), the loopback interface, or an internal/private host the
// gateway can reach but the operator should not. We refuse those at REGISTRATION
// (fail-closed), resolving hostnames so a name that resolves to a blocked IP is
// caught too.
//
// Self-host nuance: a self-hoster legitimately runs Ollama/vLLM on loopback or a
// private VPC address. That case is an EXPLICIT opt-in
// (`RP_CUSTOM_PROVIDER_ALLOW_PRIVATE=on`) which relaxes loopback + RFC1918/ULA —
// but **link-local / cloud-metadata (169.254.0.0/16, fe80::/10) is ALWAYS
// refused**: there is no legitimate reason to proxy to it, and it is the primary
// credential-theft target.

/// Whether the operator has opted into private/loopback custom-provider
/// endpoints (in-VPC self-host). Link-local/metadata stays blocked regardless.
#[must_use]
pub fn custom_provider_allow_private() -> bool {
    std::env::var("RP_CUSTOM_PROVIDER_ALLOW_PRIVATE")
        .map(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "on" | "true" | "yes"
            )
        })
        .unwrap_or(false)
}

fn ipv6_is_link_local(a: &std::net::Ipv6Addr) -> bool {
    (a.segments()[0] & 0xffc0) == 0xfe80
}
fn ipv6_is_unique_local(a: &std::net::Ipv6Addr) -> bool {
    // fc00::/7 (ULA) — the IPv6 analogue of RFC1918.
    (a.segments()[0] & 0xfe00) == 0xfc00
}

/// Classify a resolved IP. `Some(reason)` ⇒ refuse. Link-local/metadata is
/// refused even when `allow_private` is set; loopback + private are refused only
/// when it is not.
fn ip_is_blocked(ip: std::net::IpAddr, allow_private: bool) -> Option<&'static str> {
    use std::net::IpAddr;
    match ip {
        IpAddr::V4(v4) => {
            if v4.is_unspecified() || v4.is_multicast() || v4.is_broadcast() {
                return Some("non-routable address");
            }
            // 169.254.0.0/16 — includes the cloud metadata endpoint 169.254.169.254.
            if v4.is_link_local() {
                return Some("link-local/metadata address");
            }
            if !allow_private && (v4.is_loopback() || v4.is_private()) {
                return Some("private/loopback address");
            }
            None
        }
        IpAddr::V6(v6) => {
            if v6.is_unspecified() || v6.is_multicast() {
                return Some("non-routable address");
            }
            if ipv6_is_link_local(&v6) {
                return Some("link-local address");
            }
            // An IPv4-mapped v6 address (::ffff:a.b.c.d) must be classified by its
            // embedded v4 — otherwise ::ffff:169.254.169.254 would slip through.
            if let Some(v4) = v6.to_ipv4_mapped() {
                return ip_is_blocked(IpAddr::V4(v4), allow_private);
            }
            if !allow_private && (v6.is_loopback() || ipv6_is_unique_local(&v6)) {
                return Some("private/loopback address");
            }
            None
        }
    }
}

/// Fail-closed SSRF check for a normalized `base_url`. Resolves the host (IP
/// literal ⇒ checked directly; hostname ⇒ every resolved address checked) and
/// refuses link-local/metadata always, loopback/private unless opted in.
/// `Err((param, message))` matches the validator's error shape.
pub fn ssrf_check(base_url: &str, allow_private: bool) -> Result<(), (String, String)> {
    use std::net::{IpAddr, ToSocketAddrs};
    let parsed = url::Url::parse(base_url).map_err(|_| {
        (
            "base_url".to_string(),
            "base_url must be a valid URL".to_string(),
        )
    })?;
    let host = parsed.host_str().ok_or_else(|| {
        (
            "base_url".to_string(),
            "base_url must include a host".to_string(),
        )
    })?;
    let port = parsed.port_or_known_default().unwrap_or(443);

    let refuse = |reason: &str| {
        (
            "base_url".to_string(),
            format!(
                "base_url host is a {reason} — refused (SSRF guard). \
                 Set RP_CUSTOM_PROVIDER_ALLOW_PRIVATE=on to allow loopback/private \
                 endpoints for in-VPC servers (link-local/metadata stays blocked)."
            ),
        )
    };

    if let Ok(ip) = host.parse::<IpAddr>() {
        if let Some(reason) = ip_is_blocked(ip, allow_private) {
            return Err(refuse(reason));
        }
        return Ok(());
    }

    // Resolve and check every address the host maps to (a name resolving to a
    // blocked IP is refused). A host that does NOT resolve is DEFERRED, not
    // refused: it is not a reachable SSRF target now, and the actual dispatch
    // would fail anyway — blocking it here would only break legitimate
    // internal-DNS setups. (Registration-time resolution cannot defeat
    // DNS-rebinding regardless; that is a documented limitation.)
    match (host, port).to_socket_addrs() {
        Ok(addrs) => {
            for sa in addrs {
                if let Some(reason) = ip_is_blocked(sa.ip(), allow_private) {
                    return Err(refuse(reason));
                }
            }
            Ok(())
        }
        Err(_) => Ok(()),
    }
}

/// One live registry entry: the authored config + its adapter, built ONCE per
/// snapshot (never per request) so the hot path only ever clones an `Arc`.
pub struct CustomProviderEntry {
    pub config: CustomProviderConfig,
    pub adapter: Arc<dyn Provider>,
}

/// An immutable snapshot of the runtime registry. Swapped whole on mutation.
#[derive(Default)]
pub struct RuntimeProviders {
    by_name: HashMap<String, Arc<CustomProviderEntry>>,
    /// model id → provider name, for header-less model-based routing. When two
    /// custom providers claim the same model id, the lexicographically-first
    /// provider NAME wins (deterministic across restarts).
    model_index: HashMap<String, String>,
}

impl RuntimeProviders {
    fn build(configs: Vec<CustomProviderConfig>) -> Result<Self, String> {
        let mut by_name: HashMap<String, Arc<CustomProviderEntry>> =
            HashMap::with_capacity(configs.len());
        for cfg in configs {
            if by_name.contains_key(&cfg.name) {
                return Err(format!("duplicate provider name '{}'", cfg.name));
            }
            // The adapter is the generic OpenAI-compatible SelfHostedProvider
            // pointed at this provider's base_url. No residency region: a
            // custom provider is never eligible for sovereign routing unless a
            // future pass adds an explicit region field (conservative default).
            let adapter: Arc<dyn Provider> =
                Arc::new(SelfHostedProvider::new(cfg.base_url.clone(), String::new()));
            by_name.insert(
                cfg.name.clone(),
                Arc::new(CustomProviderEntry {
                    config: cfg,
                    adapter,
                }),
            );
        }
        // Deterministic model index: iterate providers in name order so a
        // contested model id always resolves to the same provider.
        let mut names: Vec<&String> = by_name.keys().collect();
        names.sort_unstable();
        let mut model_index: HashMap<String, String> = HashMap::new();
        for name in names {
            if let Some(entry) = by_name.get(name.as_str()) {
                for model in &entry.config.models {
                    model_index
                        .entry(model.clone())
                        .or_insert_with(|| name.to_string());
                }
            }
        }
        Ok(Self {
            by_name,
            model_index,
        })
    }

    fn configs_sorted(&self) -> Vec<CustomProviderConfig> {
        let mut configs: Vec<CustomProviderConfig> =
            self.by_name.values().map(|e| e.config.clone()).collect();
        configs.sort_by(|a, b| a.name.cmp(&b.name));
        configs
    }
}

/// The persisted file shape (`configs/providers.json`), mirroring the
/// `{"keys":[…]}` convention of `configs/keys.json`.
#[derive(Serialize, Deserialize, Default)]
struct PersistFile {
    #[serde(default)]
    providers: Vec<CustomProviderConfig>,
}

/// Shared handle: the hot path reads `shared` lock-free; the admin endpoints
/// mutate behind `write_lock` + atomic persistence.
pub type SharedRuntimeProviders = Arc<ArcSwap<RuntimeProviders>>;

pub struct CustomProviderStore {
    shared: SharedRuntimeProviders,
    /// `None` ⇒ in-memory only (tests); `Some` ⇒ atomic 0600 file persistence.
    path: Option<PathBuf>,
    /// Serializes admin mutations ONLY — never touched on the request path.
    write_lock: tokio::sync::Mutex<()>,
}

impl CustomProviderStore {
    /// Boot-time load. Absent or blank file ⇒ empty registry (ship-dark).
    /// Present-but-invalid ⇒ `Err` (the caller refuses startup, fail-closed —
    /// the same doctrine as `keys.json`). The error string never carries a key.
    pub fn load(path: PathBuf) -> Result<Self, String> {
        let snapshot = if path.exists() {
            let raw = std::fs::read_to_string(&path)
                .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
            if raw.trim().is_empty() {
                RuntimeProviders::default()
            } else {
                let file: PersistFile = serde_json::from_str(&raw)
                    .map_err(|e| format!("invalid provider registry JSON: {e}"))?;
                let mut configs = file.providers;
                for cfg in configs.iter_mut() {
                    validate_and_normalize(cfg).map_err(|(param, msg)| {
                        format!("invalid provider entry ({param}): {msg}")
                    })?;
                }
                RuntimeProviders::build(configs)?
            }
        } else {
            RuntimeProviders::default()
        };
        Ok(Self {
            shared: Arc::new(ArcSwap::from_pointee(snapshot)),
            path: Some(path),
            write_lock: tokio::sync::Mutex::new(()),
        })
    }

    /// In-memory-only store (tests / `AppState::for_tests`): mutations skip
    /// persistence, everything else is identical.
    pub fn ephemeral() -> Self {
        Self {
            shared: Arc::new(ArcSwap::from_pointee(RuntimeProviders::default())),
            path: None,
            write_lock: tokio::sync::Mutex::new(()),
        }
    }

    // ---- hot-path reads (one wait-free ArcSwap load + HashMap probe each) ----

    /// The adapter for a custom provider, or `None`. The `Arc` clone is one
    /// refcount bump; the adapter itself was built at registration time.
    pub fn adapter(&self, name: &str) -> Option<Arc<dyn Provider>> {
        self.shared
            .load()
            .by_name
            .get(name)
            .map(|e| e.adapter.clone())
    }

    /// The upstream API key for a custom provider (dispatch-only use). This is
    /// the ONLY raw-key read accessor; callers must never log or echo it.
    pub fn api_key(&self, name: &str) -> Option<String> {
        self.shared
            .load()
            .by_name
            .get(name)
            .map(|e| e.config.api_key.clone())
    }

    /// Is `name` a registered custom provider?
    pub fn contains(&self, name: &str) -> bool {
        self.shared.load().by_name.contains_key(name)
    }

    /// The custom provider registered for `model`, if any (header-less
    /// model-based routing). One load + probe; empty registry ⇒ instant miss.
    pub fn provider_for_model(&self, model: &str) -> Option<String> {
        self.shared.load().model_index.get(model).cloned()
    }

    // ---- discovery / observability reads (admin surfaces, not the chat path) ----

    /// Masked views of every provider, sorted by name.
    pub fn list(&self) -> Vec<ProviderView> {
        self.shared
            .load()
            .configs_sorted()
            .iter()
            .map(view_of)
            .collect()
    }

    /// Sorted provider names (the `/status` provider-list fold).
    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.shared.load().by_name.keys().cloned().collect();
        names.sort_unstable();
        names
    }

    /// `(model id, owning provider name)` pairs for the `/v1/models` fold,
    /// sorted by model id (deterministic listing).
    pub fn model_entries(&self) -> Vec<(String, String)> {
        let snapshot = self.shared.load();
        let mut entries: Vec<(String, String)> = snapshot
            .model_index
            .iter()
            .map(|(m, p)| (m.clone(), p.clone()))
            .collect();
        entries.sort();
        entries
    }

    pub fn len(&self) -> usize {
        self.shared.load().by_name.len()
    }

    /// `allow(dead_code)`: the BINARY target has no caller (main logs `len()`),
    /// but the library's test consumers use it (the `get_recent_events`
    /// precedent) and clippy's `len_without_is_empty` wants it beside `len`.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    // ---- mutations (admin path ONLY; persist-then-swap) ----

    /// Upsert a (pre-validated) provider: persist the new registry file
    /// atomically FIRST, then swap the live snapshot — a reader never sees an
    /// entry that is not durable. Returns the masked view + `true` if created
    /// (vs updated). Errors never contain key material.
    pub async fn upsert(
        &self,
        mut cfg: CustomProviderConfig,
    ) -> Result<(ProviderView, bool), String> {
        let _g = self.write_lock.lock().await;
        let mut configs = self.shared.load().configs_sorted();
        let created = match configs.iter_mut().find(|c| c.name == cfg.name) {
            Some(existing) => {
                // Preserve the original creation stamp on update.
                cfg.created_at = existing.created_at.clone();
                *existing = cfg.clone();
                false
            }
            None => {
                cfg.created_at = Some(chrono::Utc::now().to_rfc3339());
                configs.push(cfg.clone());
                configs.sort_by(|a, b| a.name.cmp(&b.name));
                true
            }
        };
        let next = RuntimeProviders::build(configs.clone())?;
        self.persist(configs).await?;
        self.shared.store(Arc::new(next));
        Ok((view_of(&cfg), created))
    }

    /// Remove a provider. `Ok(false)` when the name was not registered.
    pub async fn remove(&self, name: &str) -> Result<bool, String> {
        let _g = self.write_lock.lock().await;
        let mut configs = self.shared.load().configs_sorted();
        let before = configs.len();
        configs.retain(|c| c.name != name);
        if configs.len() == before {
            return Ok(false);
        }
        let next = RuntimeProviders::build(configs.clone())?;
        self.persist(configs).await?;
        self.shared.store(Arc::new(next));
        Ok(true)
    }

    /// Persist the full registry atomically: serialize, then (on a blocking
    /// pool worker — no file I/O on a runtime thread) write `<file>.tmp` with
    /// 0600 perms, fsync, and rename over the target. `None` path ⇒ no-op.
    async fn persist(&self, configs: Vec<CustomProviderConfig>) -> Result<(), String> {
        let Some(path) = self.path.clone() else {
            return Ok(());
        };
        let bytes = serde_json::to_vec_pretty(&PersistFile { providers: configs })
            .map_err(|e| format!("serialize provider registry: {e}"))?;
        tokio::task::spawn_blocking(move || write_atomic(&path, &bytes))
            .await
            .map_err(|e| format!("persist task join error: {e}"))?
    }
}

/// Atomic, restrictive-permission file write: temp file (0600) + fsync + rename.
/// `pub(crate)`: shared with `console_accounts.rs`, which persists the console
/// account registry under the identical secret-file posture.
pub(crate) fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    use std::io::Write as _;
    if let Some(dir) = path.parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
        }
    }
    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| format!("invalid registry path {}", path.display()))?;
    let tmp = path.with_file_name(format!("{file_name}.tmp"));
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        opts.mode(0o600);
    }
    let mut f = opts
        .open(&tmp)
        .map_err(|e| format!("open {}: {e}", tmp.display()))?;
    #[cfg(unix)]
    {
        // `mode(0o600)` applies only when the temp file is CREATED; if a stale
        // tmp existed with wider perms, force them down before writing secrets.
        use std::os::unix::fs::PermissionsExt as _;
        let _ = std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600));
    }
    f.write_all(bytes)
        .map_err(|e| format!("write {}: {e}", tmp.display()))?;
    f.sync_all()
        .map_err(|e| format!("sync {}: {e}", tmp.display()))?;
    drop(f);
    std::fs::rename(&tmp, path)
        .map_err(|e| format!("rename {} -> {}: {e}", tmp.display(), path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(name: &str, models: &[&str]) -> CustomProviderConfig {
        CustomProviderConfig {
            name: name.into(),
            base_url: "http://vllm.internal:8000".into(),
            api_key: "sk-custom-abcdef1234".into(),
            models: models.iter().map(|m| m.to_string()).collect(),
            created_at: None,
        }
    }

    fn temp_registry_path(tag: &str) -> PathBuf {
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "rp_custom_providers_{tag}_{}_{seq}.json",
            std::process::id()
        ))
    }

    // --- SSRF guard: adversarial coverage (the metadata endpoint, loopback,
    //     private ranges, IPv4-mapped v6, and the opt-in) ---------------------
    use std::net::IpAddr;

    fn ip(s: &str) -> IpAddr {
        s.parse().expect("test ip literal")
    }

    #[test]
    fn ssrf_metadata_endpoint_is_always_blocked_even_with_opt_in() {
        // The single most important case: the cloud metadata IP must be refused
        // whether or not private endpoints are opted in.
        assert!(ip_is_blocked(ip("169.254.169.254"), false).is_some());
        assert!(ip_is_blocked(ip("169.254.169.254"), true).is_some());
        // Whole link-local /16.
        assert!(ip_is_blocked(ip("169.254.0.1"), true).is_some());
        // IPv4-mapped IPv6 must classify by the embedded v4 (no bypass).
        assert!(ip_is_blocked(ip("::ffff:169.254.169.254"), true).is_some());
        // IPv6 link-local.
        assert!(ip_is_blocked(ip("fe80::1"), true).is_some());
    }

    #[test]
    fn ssrf_loopback_and_private_blocked_by_default_allowed_on_opt_in() {
        for addr in [
            "127.0.0.1",
            "10.0.0.5",
            "172.16.0.1",
            "192.168.1.10",
            "::1",
            "fc00::1", // IPv6 ULA
        ] {
            assert!(
                ip_is_blocked(ip(addr), false).is_some(),
                "{addr} must be blocked by default"
            );
            assert!(
                ip_is_blocked(ip(addr), true).is_none(),
                "{addr} must be allowed under the in-VPC opt-in"
            );
        }
    }

    #[test]
    fn ssrf_unspecified_and_multicast_always_blocked() {
        assert!(ip_is_blocked(ip("0.0.0.0"), true).is_some());
        assert!(ip_is_blocked(ip("::"), true).is_some());
        assert!(ip_is_blocked(ip("224.0.0.1"), true).is_some());
        assert!(ip_is_blocked(ip("255.255.255.255"), true).is_some());
    }

    #[test]
    fn ssrf_public_addresses_pass() {
        assert!(ip_is_blocked(ip("1.1.1.1"), false).is_none());
        assert!(ip_is_blocked(ip("8.8.8.8"), false).is_none());
        assert!(ip_is_blocked(ip("2606:4700:4700::1111"), false).is_none());
    }

    #[test]
    fn ssrf_check_refuses_ip_literal_metadata_and_localhost_name() {
        // IP-literal base_url straight to metadata.
        assert!(ssrf_check("http://169.254.169.254", false).is_err());
        assert!(ssrf_check("http://169.254.169.254:80/v1", true).is_err());
        // A hostname that resolves to loopback (localhost) is refused by default…
        assert!(ssrf_check("http://localhost:11434", false).is_err());
        // …and permitted under the in-VPC opt-in.
        assert!(ssrf_check("http://localhost:11434", true).is_ok());
        // A public IP literal passes.
        assert!(ssrf_check("https://1.1.1.1", false).is_ok());
        // A host that does not resolve is DEFERRED (allowed at registration —
        // it is not a reachable SSRF target; dispatch fails on its own).
        assert!(ssrf_check("http://nonexistent.invalid:8000", false).is_ok());
    }

    #[test]
    fn mask_key_keeps_only_last_four_and_never_panics_on_multibyte() {
        assert_eq!(mask_key("sk-custom-abcdef1234"), "…1234");
        assert_eq!(mask_key("short"), "…"); // < 8 chars: fully hidden
        assert_eq!(mask_key(""), "…");
        // Multi-byte input must not panic on a char boundary.
        assert_eq!(mask_key("ключключключ"), "…ключ");
    }

    #[test]
    fn validation_rejects_bad_names_urls_and_models() {
        let mut c = cfg("My-VLLM", &["llama3"]); // uppercase
        assert_eq!(validate_and_normalize(&mut c).unwrap_err().0, "name");

        let mut c = cfg("myvllm", &["llama3"]);
        c.base_url = "ftp://host:21".into();
        assert_eq!(validate_and_normalize(&mut c).unwrap_err().0, "base_url");

        let mut c = cfg("myvllm", &["llama3"]);
        c.base_url = "http://user:pass@host:8000".into();
        assert_eq!(validate_and_normalize(&mut c).unwrap_err().0, "base_url");

        let mut c = cfg("myvllm", &["llama3"]);
        c.base_url = "not a url".into();
        assert_eq!(validate_and_normalize(&mut c).unwrap_err().0, "base_url");

        let mut c = cfg("myvllm", &[]);
        assert_eq!(validate_and_normalize(&mut c).unwrap_err().0, "models");

        let mut c = cfg("myvllm", &["llama3"]);
        c.api_key = String::new();
        assert_eq!(validate_and_normalize(&mut c).unwrap_err().0, "api_key");
    }

    #[test]
    fn validation_normalizes_trailing_slash_and_dedupes_models() {
        let mut c = cfg("myvllm", &["llama3", "llama3", " qwen "]);
        c.base_url = "http://vllm.internal:8000/".into();
        validate_and_normalize(&mut c).expect("valid config");
        assert_eq!(c.base_url, "http://vllm.internal:8000");
        assert_eq!(c.models, vec!["llama3".to_string(), "qwen".to_string()]);
    }

    #[tokio::test]
    async fn upsert_lookup_remove_round_trip_in_memory() {
        let store = CustomProviderStore::ephemeral();
        assert!(store.is_empty());
        assert!(store.provider_for_model("llama3").is_none());

        let (view, created) = store.upsert(cfg("myvllm", &["llama3"])).await.unwrap();
        assert!(created);
        assert_eq!(view.api_key, "…1234", "view must be masked");
        assert!(view.created_at.is_some());

        // Hot-path reads see it immediately (the ArcSwap swap happened).
        assert!(store.contains("myvllm"));
        assert_eq!(
            store.provider_for_model("llama3").as_deref(),
            Some("myvllm")
        );
        assert_eq!(
            store.api_key("myvllm").as_deref(),
            Some("sk-custom-abcdef1234")
        );
        assert!(store.adapter("myvllm").is_some());

        // Update preserves created_at and reports created=false.
        let created_at = view.created_at.clone();
        let mut updated = cfg("myvllm", &["llama3", "qwen"]);
        updated.api_key = "sk-custom-zzzz9999".into();
        let (view2, created2) = store.upsert(updated).await.unwrap();
        assert!(!created2);
        assert_eq!(view2.created_at, created_at);
        assert_eq!(store.provider_for_model("qwen").as_deref(), Some("myvllm"));

        assert!(store.remove("myvllm").await.unwrap());
        assert!(!store.remove("myvllm").await.unwrap());
        assert!(store.provider_for_model("llama3").is_none());
    }

    #[tokio::test]
    async fn contested_model_resolves_to_lexicographically_first_provider() {
        let store = CustomProviderStore::ephemeral();
        store.upsert(cfg("zeta", &["shared-model"])).await.unwrap();
        store.upsert(cfg("alpha", &["shared-model"])).await.unwrap();
        assert_eq!(
            store.provider_for_model("shared-model").as_deref(),
            Some("alpha"),
            "deterministic precedence: first provider name in sort order wins"
        );
    }

    #[tokio::test]
    async fn persistence_round_trips_atomically_with_0600_perms() {
        let path = temp_registry_path("persist");
        let _ = std::fs::remove_file(&path);

        let store = CustomProviderStore::load(path.clone()).expect("absent file loads empty");
        assert!(store.is_empty());
        store.upsert(cfg("myvllm", &["llama3"])).await.unwrap();

        // The file exists, the tmp is gone (rename), and perms are 0600.
        assert!(path.exists());
        assert!(!path.with_file_name("providers.json.tmp").exists());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600, "registry file must be 0600 (holds secrets)");
        }

        // A fresh store (a restart) sees the same registry.
        let reloaded = CustomProviderStore::load(path.clone()).expect("reload");
        assert!(reloaded.contains("myvllm"));
        assert_eq!(
            reloaded.api_key("myvllm").as_deref(),
            Some("sk-custom-abcdef1234")
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn malformed_registry_file_is_fail_closed() {
        let path = temp_registry_path("malformed");
        std::fs::write(&path, "{ not json").unwrap();
        assert!(CustomProviderStore::load(path.clone()).is_err());
        let _ = std::fs::remove_file(&path);
    }
}
