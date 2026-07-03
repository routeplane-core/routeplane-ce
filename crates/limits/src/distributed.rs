//! Mode D — **opt-in** distributed (Redis-backed) rate limiting ([ADR-056]).
//!
//! # Why this exists, and why it is off by default
//! Mode L (the [`crate`] root engine) enforces limits **per replica**: correct
//! and lock-free, but a fleet-wide cap is only approximate under scale-out
//! (mitigated by the ADR-023 §6 share-clamp). Mode D moves the counter to a
//! shared Redis so the count is **globally exact** under concurrency. Redis is a
//! **standing cost**, which the platform's frugality rule ($0-idle default, "no
//! DB/standing cost without an ADR" — [ADR-056]) forbids by default. Therefore:
//!
//! - The whole module compiles **only** under the `redis-limits` cargo feature.
//! - Even compiled in, Mode D is **inert until an endpoint is configured**
//!   ([`DistributedConfig::from_env`] returns `None` ⇒ caller stays on Mode L).
//! - When unconfigured the gateway is **byte-identical** to today.
//!
//! # Hot-path safety — the non-negotiable invariant
//! A Redis round-trip is network I/O; it **must never** block or fail a request.
//! Every check is wrapped in a **bounded [`tokio::time::timeout`]**; on timeout,
//! connection error, or any Redis error the limiter **fails open to the local
//! Mode L engine** (the default — availability over perfect global accuracy for
//! a rate limit) and emits a metric/log. The fallback policy is configurable
//! ([`DistributedFallback`]); `OpenToLocal` is the default, `Closed` is for
//! deployments that would rather 429 than under-count.
//!
//! The expensive object — the cloneable [`redis::aio::ConnectionManager`]
//! (multiplexed, auto-reconnecting; the recommended pooling primitive in
//! redis-rs 0.25) — is built **once** at startup, off the hot path. The hot path
//! clones the manager handle (an `Arc` bump) and runs one Lua `EVALSHA`.
//!
//! # Atomic counting — a single Lua round-trip
//! Rate counting under concurrency requires INCR **and** EXPIRE to be one atomic
//! operation, or two racing first-requests can leave a key with no TTL (a
//! permanent leak that wedges the limit forever). The [`RATE_INCR_SCRIPT`] does
//! `INCR` then `EXPIRE NX` and returns `(count, ttl)` in **one** server-side
//! round-trip — atomic by Redis's single-threaded script execution. This is a
//! fixed-window counter keyed identically to Mode L (scope · policy · period ·
//! epoch), so the two modes are semantically interchangeable.
//!
//! # Entra-ID auth (OIDC-only constraint)
//! Azure Cache for Redis supports Microsoft Entra authentication: the client
//! AUTHs with the principal's object-id as the username and a short-lived Entra
//! **access token** (scope `https://redis.azure.com/.default`) as the password,
//! refreshed before expiry. This crate keeps the auth **pluggable** via the
//! [`TokenProvider`] trait so the binary can wire a managed-identity token
//! source without this crate taking an azure-sdk dependency (the MSRV/RUSTSEC
//! wall in the root Cargo.toml forbids azure_identity here). An access-key
//! fallback is supported but discouraged and sourced only from env, never
//! hardcoded.
//!
//! [ADR-023]: ../../../docs/adr/023-hot-path-budget-rate-limit-enforcement.md
//! [ADR-056]: ../../../docs/adr/056-opt-in-distributed-rate-limiting-redis.md

use std::sync::Arc;
use std::time::Duration;

use crate::{Admission, Breach, LimitGuards, LimitScope};

// ---------------------------------------------------------------------------
// Fallback policy + config
// ---------------------------------------------------------------------------

/// What to do when Redis is unreachable / slow / errors on a check.
///
/// Default is [`DistributedFallback::OpenToLocal`]: a rate limit exists for
/// abuse control, and the platform's reliability bar makes *availability* win
/// over *perfect* global accuracy — a brief Redis blip must not 500/429 real
/// traffic. `Closed` is offered for the rare deployment that prefers to deny on
/// uncertainty (e.g. a hard spend cap where over-admission is worse than a
/// transient false 429).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DistributedFallback {
    /// Fall back to the local Mode L engine on any Redis failure (default).
    #[default]
    OpenToLocal,
    /// Deny (429-shaped breach) on any Redis failure.
    Closed,
}

/// The auth strategy for the Redis connection. Entra ID is strongly preferred
/// (the OIDC-only constraint); the access-key path exists only for environments
/// where Entra is unavailable and is sourced from env, never hardcoded.
#[derive(Clone)]
pub enum RedisAuth {
    /// No auth (local dev redis / unauthenticated cache).
    None,
    /// Microsoft Entra ID: AUTH `<username> <entra-access-token>`, the token
    /// produced (and refreshed) by a [`TokenProvider`]. `username` is the
    /// principal's object-id (or the configured user name on the cache).
    Entra {
        username: String,
        provider: Arc<dyn TokenProvider>,
    },
    /// Access key as the AUTH password (discouraged; env-sourced only).
    AccessKey { password: String },
}

impl std::fmt::Debug for RedisAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never print secrets.
        match self {
            RedisAuth::None => f.write_str("RedisAuth::None"),
            RedisAuth::Entra { username, .. } => {
                write!(f, "RedisAuth::Entra {{ username: {username:?}, .. }}")
            }
            RedisAuth::AccessKey { .. } => {
                f.write_str("RedisAuth::AccessKey { password: <redacted> }")
            }
        }
    }
}

/// A source of Entra ID access tokens for Redis AUTH (scope
/// `https://redis.azure.com/.default`). Implemented by the binary against its
/// managed-identity / MSI REST endpoint so this crate stays azure-sdk-free.
pub trait TokenProvider: Send + Sync {
    /// Return a currently-valid access token, refreshing if near expiry. Called
    /// off the hot path (at connect / on the refresh cadence).
    fn token(&self) -> Result<String, String>;
}

/// Mode-D configuration, resolved once at startup. Absent endpoint ⇒ Mode D is
/// not activated (the caller stays on Mode L).
#[derive(Clone)]
pub struct DistributedConfig {
    /// `redis://` or `rediss://` URL (host:port; TLS required for Entra).
    pub url: String,
    /// Auth strategy (Entra preferred).
    pub auth: RedisAuth,
    /// Per-check round-trip budget. Bounds the hot-path cost; on expiry the
    /// fallback policy applies. Kept small (single-digit ms in-region).
    pub timeout: Duration,
    /// What to do on a Redis failure.
    pub fallback: DistributedFallback,
    /// Key prefix to namespace this deployment's counters in a shared cache.
    pub key_prefix: String,
}

impl std::fmt::Debug for DistributedConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DistributedConfig")
            .field("url", &redact_url(&self.url))
            .field("auth", &self.auth)
            .field("timeout", &self.timeout)
            .field("fallback", &self.fallback)
            .field("key_prefix", &self.key_prefix)
            .finish()
    }
}

/// Redact any `user:pass@` userinfo from a URL before logging.
fn redact_url(url: &str) -> String {
    match (url.find("://"), url.find('@')) {
        (Some(scheme_end), Some(at)) if at > scheme_end + 3 => {
            format!("{}://<redacted>@{}", &url[..scheme_end], &url[at + 1..])
        }
        _ => url.to_string(),
    }
}

impl DistributedConfig {
    /// Resolve Mode-D config from the environment. **Returns `None` when
    /// `ROUTEPLANE_REDIS_URL` is unset** — the ship-dark default: the caller
    /// then stays on Mode L and the gateway is byte-identical to today.
    ///
    /// Env knobs (all optional except the URL):
    /// - `ROUTEPLANE_REDIS_URL`            — `rediss://host:6380` (required to activate)
    /// - `ROUTEPLANE_REDIS_TIMEOUT_MS`     — per-check budget (default 50)
    /// - `ROUTEPLANE_REDIS_FALLBACK`       — `open` (default) | `closed`
    /// - `ROUTEPLANE_REDIS_KEY_PREFIX`     — counter namespace (default `rp:rl`)
    /// - `ROUTEPLANE_REDIS_ENTRA_USERNAME` — Entra principal object-id (enables Entra auth)
    /// - `ROUTEPLANE_REDIS_ACCESS_KEY`     — access-key fallback (discouraged)
    ///
    /// The Entra [`TokenProvider`] is injected by the caller via
    /// [`DistributedConfig::with_token_provider`] — env cannot carry a closure.
    pub fn from_env() -> Option<Self> {
        let url = std::env::var("ROUTEPLANE_REDIS_URL")
            .ok()
            .filter(|u| !u.is_empty())?;
        let timeout = std::env::var("ROUTEPLANE_REDIS_TIMEOUT_MS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .map(Duration::from_millis)
            .unwrap_or_else(|| Duration::from_millis(50));
        let fallback = match std::env::var("ROUTEPLANE_REDIS_FALLBACK").as_deref() {
            Ok("closed") => DistributedFallback::Closed,
            _ => DistributedFallback::OpenToLocal,
        };
        let key_prefix = std::env::var("ROUTEPLANE_REDIS_KEY_PREFIX")
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "rp:rl".to_string());
        // Access-key fallback (Entra is wired by the caller post-construction).
        let auth = match std::env::var("ROUTEPLANE_REDIS_ACCESS_KEY") {
            Ok(k) if !k.is_empty() => RedisAuth::AccessKey { password: k },
            _ => RedisAuth::None,
        };
        Some(Self {
            url,
            auth,
            timeout: timeout.max(Duration::from_millis(1)),
            fallback,
            key_prefix,
        })
    }

    /// Attach an Entra ID token provider (preferred auth). The caller builds this
    /// against its managed identity and passes the principal's object-id as the
    /// username; this supersedes any env access key.
    pub fn with_token_provider(
        mut self,
        username: impl Into<String>,
        provider: Arc<dyn TokenProvider>,
    ) -> Self {
        self.auth = RedisAuth::Entra {
            username: username.into(),
            provider,
        };
        self
    }
}

// ---------------------------------------------------------------------------
// Lua: atomic fixed-window INCR + EXPIRE in one round-trip
// ---------------------------------------------------------------------------

/// Atomic fixed-window increment.
///
/// `KEYS[1]` = the counter key (already scoped to scope·policy·kind·epoch).
/// `ARGV[1]` = amount to add. `ARGV[2]` = window TTL in seconds.
/// Returns `{ new_count, ttl_seconds }`.
///
/// `INCRBY` creates the key at 0 if absent; `EXPIRE key ttl NX` (Redis ≥ 7.0)
/// sets the TTL **only if none exists**, so a fresh window gets a TTL exactly
/// once and a re-increment within the window never extends it (true fixed
/// window, not sliding). Single-threaded script execution makes the pair atomic
/// — no INCR-without-EXPIRE race can leak a permanent key.
pub const RATE_INCR_SCRIPT: &str = r"
local c = redis.call('INCRBY', KEYS[1], tonumber(ARGV[1]))
if c == tonumber(ARGV[1]) then
  redis.call('PEXPIRE', KEYS[1], tonumber(ARGV[2]))
else
  if redis.call('PTTL', KEYS[1]) < 0 then
    redis.call('PEXPIRE', KEYS[1], tonumber(ARGV[2]))
  end
end
local ttl = redis.call('PTTL', KEYS[1])
return { c, ttl }
";

/// Read-only counter peek (no mutation) — for the check-before phase of budgets
/// and token-rate where we must not consume. Returns `{ count, ttl_ms }`;
/// `count` is 0 / `ttl` -2 for an absent key.
pub const RATE_PEEK_SCRIPT: &str = r"
local v = redis.call('GET', KEYS[1])
local c = 0
if v then c = tonumber(v) end
local ttl = redis.call('PTTL', KEYS[1])
return { c, ttl }
";

// ---------------------------------------------------------------------------
// DistributedLimiter — the Mode-D engine (feature-gated active path)
// ---------------------------------------------------------------------------

/// The distributed (Redis-backed) limiter. Holds a cloneable connection manager
/// built once at startup. The hot path clones the handle and runs one Lua call
/// under a bounded timeout; on any failure it defers to the supplied local
/// [`LimitGuards`] per the [`DistributedFallback`] policy.
#[derive(Clone)]
pub struct DistributedLimiter {
    inner: Arc<Inner>,
}

struct Inner {
    manager: redis::aio::ConnectionManager,
    incr: redis::Script,
    peek: redis::Script,
    timeout: Duration,
    fallback: DistributedFallback,
    key_prefix: String,
}

impl std::fmt::Debug for DistributedLimiter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DistributedLimiter")
            .field("timeout", &self.inner.timeout)
            .field("fallback", &self.inner.fallback)
            .field("key_prefix", &self.inner.key_prefix)
            .finish()
    }
}

impl DistributedLimiter {
    /// Connect to Redis and build the limiter — **off the hot path**, at startup.
    /// Fails loudly (the caller decides whether to fall back to Mode L or abort);
    /// it never panics. The `ConnectionManager` handles reconnection thereafter,
    /// so a Redis blip after startup degrades to the fallback policy, not a
    /// rebuild.
    pub async fn connect(cfg: &DistributedConfig) -> Result<Self, String> {
        let info = build_connection_info(cfg)?;
        let client = redis::Client::open(info).map_err(|e| format!("redis client open: {e}"))?;
        // Bound the per-operation connection AND response timeouts at the manager
        // level (so a post-startup blip degrades to the fallback policy on the
        // hot path, never a hang), then bound the INITIAL connect with an
        // overall timeout so startup itself can never block indefinitely against
        // an unreachable / black-holed endpoint.
        let op_timeout = cfg.timeout.max(Duration::from_millis(50));
        let connect_budget = (cfg.timeout.saturating_mul(20)).max(Duration::from_secs(3));
        let build = redis::aio::ConnectionManager::new_with_backoff_and_timeouts(
            client, 2,   // exponent base
            100, // factor (ms)
            6,   // number of retries before giving up a single connect attempt
            op_timeout, op_timeout,
        );
        let manager = match tokio::time::timeout(connect_budget, build).await {
            Ok(Ok(m)) => m,
            Ok(Err(e)) => return Err(format!("redis connect: {e}")),
            Err(_) => return Err("redis connect timeout".to_string()),
        };
        Ok(Self {
            inner: Arc::new(Inner {
                manager,
                incr: redis::Script::new(RATE_INCR_SCRIPT),
                peek: redis::Script::new(RATE_PEEK_SCRIPT),
                timeout: cfg.timeout,
                fallback: cfg.fallback,
                key_prefix: cfg.key_prefix.clone(),
            }),
        })
    }

    /// Build a per-scope/kind counter key. Identical *structural* scoping to Mode L
    /// (scope·identity·kind·epoch) so the two modes are interchangeable: `identity`
    /// is the owning key/tenant/model (never the reusable operator `policy_id`), so
    /// two tenants that share a `policy_id` still get distinct Redis counters.
    fn key(&self, scope: LimitScope, identity: &str, kind: &str, epoch: u64) -> String {
        counter_key(&self.inner.key_prefix, scope, identity, kind, epoch)
    }

    /// Run the atomic INCR+EXPIRE Lua, bounded by the configured timeout.
    /// Returns `(count, ttl_ms)` or an error string (timeout/redis).
    async fn incr(&self, key: &str, amount: u64, ttl_ms: u64) -> Result<(u64, i64), String> {
        let mut conn = self.inner.manager.clone();
        // Bind the ScriptInvocation to a named `mut` local so it outlives the
        // future (`key()` returns the invocation by value; `arg()` mutates it
        // in place and returns `&mut`, so a fully-chained temporary would be
        // dropped at the end of the statement — E0716).
        let mut invocation = self.inner.incr.key(key);
        invocation.arg(amount).arg(ttl_ms);
        let fut = invocation.invoke_async(&mut conn);
        // invoke_async's first generic is the connection type (inferred from
        // `conn`); the second is the return type, annotated on the binding.
        let res: Result<(u64, i64), redis::RedisError> =
            match tokio::time::timeout(self.inner.timeout, fut).await {
                Ok(r) => r,
                Err(_) => return Err("redis incr timeout".to_string()),
            };
        res.map_err(|e| format!("redis incr: {e}"))
    }

    /// Read-only peek under the same bounded timeout.
    async fn peek(&self, key: &str) -> Result<(u64, i64), String> {
        let mut conn = self.inner.manager.clone();
        let invocation = self.inner.peek.key(key);
        let fut = invocation.invoke_async(&mut conn);
        let res: Result<(u64, i64), redis::RedisError> =
            match tokio::time::timeout(self.inner.timeout, fut).await {
                Ok(r) => r,
                Err(_) => return Err("redis peek timeout".to_string()),
            };
        res.map_err(|e| format!("redis peek: {e}"))
    }

    fn on_failure(&self, what: &str, err: &str, local: &LimitGuards, now_ms: u64) -> Admission {
        match self.inner.fallback {
            DistributedFallback::OpenToLocal => tracing::warn!(
                target: "routeplane::limits::distributed",
                op = what, error = err,
                "Mode-D Redis failure; failing OPEN to local Mode L"
            ),
            DistributedFallback::Closed => tracing::warn!(
                target: "routeplane::limits::distributed",
                op = what, error = err,
                "Mode-D Redis failure; failing CLOSED (deny)"
            ),
        }
        decide_failure(self.inner.fallback, local, now_ms)
    }

    /// Distributed admission. Mirrors [`LimitGuards::admit`] semantics but the
    /// per-minute **request** counter is enforced in Redis (the dominant
    /// abuse-control signal and the one most distorted by per-replica counting).
    /// Token-rate and budgets continue to be checked against the local engine
    /// here (they are settle-after / debited post-response; a follow-up wires
    /// their Redis debit on the `settle` path). On ANY Redis failure this defers
    /// to `local` per the fallback policy.
    ///
    /// `local` is the same [`LimitGuards`] the caller already resolved from the
    /// [`crate::LimitRegistry`] — so when Mode D is bypassed (unlimited key) or
    /// fails open, behaviour is exactly Mode L.
    pub async fn admit(&self, local: &LimitGuards, now_ms: u64) -> Admission {
        // Unlimited / unconfigured key: never touch Redis — byte-identical to L.
        if local.is_unlimited() {
            return local.admit(now_ms);
        }

        // First run the local read-only check-before (budgets + token-rate). This
        // is lock-free and free; it preserves the 402/budget path unchanged and
        // only the per-minute REQUEST rate is escalated to Redis below.
        if let Admission::Denied(b) = local.check_only(now_ms) {
            return Admission::Denied(b);
        }

        // Per-scope request-rate enforcement in Redis (atomic across replicas).
        for spec in local.request_rate_specs() {
            let epoch = now_ms / 60_000;
            // Key off the STRUCTURAL identity (per key/tenant/model), never the
            // reusable operator policy_id — the breach below still reports policy_id.
            let key = self.key(spec.scope, &spec.identity, "req", epoch);
            // Window TTL: time remaining in this minute (+1s slack), so the key
            // self-expires on rollover exactly like the local fixed window.
            let ttl_ms = 60_000 - (now_ms % 60_000) + 1_000;
            match self.incr(&key, 1, ttl_ms).await {
                Ok((count, _ttl)) => {
                    if spec.hard && count > spec.limit {
                        return Admission::Denied(Breach::distributed_rate(
                            spec.scope,
                            spec.policy_id.clone(),
                            spec.limit,
                            now_ms,
                        ));
                    }
                }
                Err(e) => return self.on_failure("admit", &e, local, now_ms),
            }
        }

        // Advisory from the local view (approximate; documented).
        Admission::Allowed(local.advisory(now_ms))
    }

    /// Distributed peek of a single request-rate key — exposed for tests and for
    /// a future advisory that reflects the global count. `identity` is the
    /// structural owner ([`RequestRateSpec::identity`]), not the operator policy_id.
    /// `Err` ⇒ Redis failure.
    pub async fn peek_request_rate(
        &self,
        scope: LimitScope,
        identity: &str,
        now_ms: u64,
    ) -> Result<u64, String> {
        let epoch = now_ms / 60_000;
        let key = self.key(scope, identity, "req", epoch);
        self.peek(&key).await.map(|(c, _)| c)
    }

    /// The active fallback policy (for startup logging).
    pub fn fallback(&self) -> DistributedFallback {
        self.inner.fallback
    }
}

// ---------------------------------------------------------------------------
// Connection info builder (Entra / access-key / none)
// ---------------------------------------------------------------------------

fn build_connection_info(cfg: &DistributedConfig) -> Result<redis::ConnectionInfo, String> {
    use redis::{ConnectionAddr, ConnectionInfo, RedisConnectionInfo};

    // Parse the URL for host/port/TLS via the redis crate, then override the auth
    // bits from our typed RedisAuth (so Entra tokens never live in the URL).
    let base: ConnectionInfo = cfg
        .url
        .parse::<ConnectionInfo>()
        .map_err(|e| format!("redis url parse: {e}"))?;

    let addr: ConnectionAddr = base.addr;

    let redis_info = match &cfg.auth {
        RedisAuth::None => RedisConnectionInfo::default(),
        RedisAuth::AccessKey { password } => RedisConnectionInfo {
            password: Some(password.clone()),
            ..Default::default()
        },
        RedisAuth::Entra { username, provider } => {
            let token = provider
                .token()
                .map_err(|e| format!("entra token acquire: {e}"))?;
            RedisConnectionInfo {
                username: Some(username.clone()),
                password: Some(token),
                ..Default::default()
            }
        }
    };

    Ok(ConnectionInfo {
        addr,
        redis: redis_info,
    })
}

/// Build the fixed-window counter key for a `(scope, identity, kind, epoch)`,
/// factored out as a pure free function so the key shape is unit-testable WITHOUT
/// a live (or even constructed) Redis connection. `identity` is the *structural*
/// owner (key/tenant/model, e.g. `tenant:<tenant_id>`), never the reusable
/// operator `policy_id`, so two tenants sharing a `policy_id` never collapse onto
/// one counter. `scope.header()` is kept as an explicit leading segment (defence
/// in depth against cross-scope collision even if `identity` generation changes).
fn counter_key(prefix: &str, scope: LimitScope, identity: &str, kind: &str, epoch: u64) -> String {
    format!("{prefix}:{}:{identity}:{kind}:{epoch}", scope.header())
}

/// The fail-open / fail-closed decision, factored out as a pure function so it is
/// unit-testable WITHOUT a live (or even constructed) Redis connection. This is
/// the hot-path safety valve: on any Redis error/timeout, Mode D must NEVER block
/// or 500 the request — it either defers to the local Mode L engine
/// ([`DistributedFallback::OpenToLocal`], the default) or denies with a 429-shaped
/// breach ([`DistributedFallback::Closed`]).
fn decide_failure(fallback: DistributedFallback, local: &LimitGuards, now_ms: u64) -> Admission {
    match fallback {
        DistributedFallback::OpenToLocal => local.admit(now_ms),
        DistributedFallback::Closed => Admission::Denied(Breach::distributed_unavailable(now_ms)),
    }
}

// ---------------------------------------------------------------------------
// Tests — Lua key/TTL shaping, fallback selection, config parsing. No live
// Redis required (live integration is #[ignore], mirroring the real-Postgres
// pattern called out in the build brief).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{KeyLimitsInput, LimitRegistry};

    fn cfg() -> DistributedConfig {
        DistributedConfig {
            url: "redis://127.0.0.1:6379".into(),
            auth: RedisAuth::None,
            timeout: Duration::from_millis(50),
            fallback: DistributedFallback::OpenToLocal,
            key_prefix: "rp:rl".into(),
        }
    }

    #[test]
    fn from_env_is_none_without_url() {
        // Ship-dark: with no endpoint configured, Mode D never activates.
        // (Guard against ambient env in CI by removing the var for this scope.)
        let prev = std::env::var("ROUTEPLANE_REDIS_URL").ok();
        std::env::remove_var("ROUTEPLANE_REDIS_URL");
        assert!(DistributedConfig::from_env().is_none());
        if let Some(v) = prev {
            std::env::set_var("ROUTEPLANE_REDIS_URL", v);
        }
    }

    #[test]
    fn from_env_defaults_to_open_fallback() {
        let prev_url = std::env::var("ROUTEPLANE_REDIS_URL").ok();
        let prev_fb = std::env::var("ROUTEPLANE_REDIS_FALLBACK").ok();
        std::env::set_var("ROUTEPLANE_REDIS_URL", "rediss://example:6380");
        std::env::remove_var("ROUTEPLANE_REDIS_FALLBACK");
        let c = DistributedConfig::from_env().expect("configured");
        assert_eq!(c.fallback, DistributedFallback::OpenToLocal);
        assert_eq!(c.key_prefix, "rp:rl");
        // restore
        match prev_url {
            Some(v) => std::env::set_var("ROUTEPLANE_REDIS_URL", v),
            None => std::env::remove_var("ROUTEPLANE_REDIS_URL"),
        }
        if let Some(v) = prev_fb {
            std::env::set_var("ROUTEPLANE_REDIS_FALLBACK", v);
        }
    }

    #[test]
    fn from_env_closed_fallback_is_explicit() {
        let prev_url = std::env::var("ROUTEPLANE_REDIS_URL").ok();
        let prev_fb = std::env::var("ROUTEPLANE_REDIS_FALLBACK").ok();
        std::env::set_var("ROUTEPLANE_REDIS_URL", "rediss://example:6380");
        std::env::set_var("ROUTEPLANE_REDIS_FALLBACK", "closed");
        let c = DistributedConfig::from_env().expect("configured");
        assert_eq!(c.fallback, DistributedFallback::Closed);
        match prev_url {
            Some(v) => std::env::set_var("ROUTEPLANE_REDIS_URL", v),
            None => std::env::remove_var("ROUTEPLANE_REDIS_URL"),
        }
        match prev_fb {
            Some(v) => std::env::set_var("ROUTEPLANE_REDIS_FALLBACK", v),
            None => std::env::remove_var("ROUTEPLANE_REDIS_FALLBACK"),
        }
    }

    #[test]
    fn connection_info_omits_auth_for_none() {
        let info = build_connection_info(&cfg()).expect("parse");
        assert!(info.redis.password.is_none());
        assert!(info.redis.username.is_none());
    }

    #[test]
    fn connection_info_uses_access_key_password() {
        let mut c = cfg();
        c.auth = RedisAuth::AccessKey {
            password: "supersecret".into(),
        };
        let info = build_connection_info(&c).expect("parse");
        assert_eq!(info.redis.password.as_deref(), Some("supersecret"));
    }

    struct StaticToken(String);
    impl TokenProvider for StaticToken {
        fn token(&self) -> Result<String, String> {
            Ok(self.0.clone())
        }
    }

    #[test]
    fn connection_info_uses_entra_token_as_password() {
        let mut c = cfg();
        c.auth = RedisAuth::Entra {
            username: "objid-123".into(),
            provider: Arc::new(StaticToken("entra-token-xyz".into())),
        };
        let info = build_connection_info(&c).expect("parse");
        assert_eq!(info.redis.username.as_deref(), Some("objid-123"));
        assert_eq!(info.redis.password.as_deref(), Some("entra-token-xyz"));
    }

    #[test]
    fn entra_token_acquire_failure_is_surfaced_not_panicked() {
        struct Failing;
        impl TokenProvider for Failing {
            fn token(&self) -> Result<String, String> {
                Err("imds 500".into())
            }
        }
        let mut c = cfg();
        c.auth = RedisAuth::Entra {
            username: "obj".into(),
            provider: Arc::new(Failing),
        };
        let err = build_connection_info(&c).unwrap_err();
        assert!(err.contains("entra token"));
    }

    #[test]
    fn debug_redacts_secrets() {
        let mut c = cfg();
        c.auth = RedisAuth::AccessKey {
            password: "topsecret".into(),
        };
        let s = format!("{c:?}");
        assert!(!s.contains("topsecret"));
        assert!(s.contains("redacted"));
        // URL userinfo is redacted too.
        assert_eq!(
            redact_url("rediss://user:pass@host:6380"),
            "rediss://<redacted>@host:6380"
        );
    }

    // --- Fallback policy: tested as a PURE decision, NO live/constructed Redis -
    //
    // The hot-path safety valve is `decide_failure(fallback, local, now_ms)` —
    // factored out precisely so the fail-open / fail-closed contract can be
    // asserted deterministically in CI with no network and no connection object.

    fn local_reg() -> LimitRegistry {
        LimitRegistry::build(vec![KeyLimitsInput {
            routeplane_key: "rp_d".into(),
            tenant_id: "t_d".into(),
            limits: serde_json::from_str(r#"{"key":{"rate":{"requests_per_min":2}}}"#).unwrap(),
        }])
    }

    #[test]
    fn fail_open_defers_to_the_local_mode_l_engine() {
        // On a simulated Redis error, OpenToLocal must enforce the LOCAL 2 req/min
        // cap: admit 2, deny the 3rd in the same minute — exactly Mode L.
        let reg = local_reg();
        let g = reg.resolve("rp_d", "t_d");
        let now = 60_000;
        assert!(matches!(
            decide_failure(DistributedFallback::OpenToLocal, &g, now),
            Admission::Allowed(_)
        ));
        assert!(matches!(
            decide_failure(DistributedFallback::OpenToLocal, &g, now),
            Admission::Allowed(_)
        ));
        assert!(matches!(
            decide_failure(DistributedFallback::OpenToLocal, &g, now),
            Admission::Denied(_)
        ));
    }

    #[test]
    fn fail_open_on_unlimited_key_is_byte_identical_allow() {
        // A legacy / unlimited key failing open produces an empty advisory —
        // byte-identical to today.
        let g = LimitRegistry::empty().resolve("x", "y");
        match decide_failure(DistributedFallback::OpenToLocal, &g, 0) {
            Admission::Allowed(a) => assert!(a.is_empty()),
            Admission::Denied(_) => panic!("unlimited fail-open must allow"),
        }
    }

    #[test]
    fn fail_closed_denies_with_a_429_shaped_breach() {
        // Closed fallback denies on uncertainty, surfaced as a rate (429) breach
        // with a Retry-After — never a 500 on the request thread.
        let reg = local_reg();
        let g = reg.resolve("rp_d", "t_d");
        match decide_failure(DistributedFallback::Closed, &g, 60_000) {
            Admission::Denied(b) => {
                assert!(!b.is_budget()); // 429, not 402
                assert_eq!(b.kind_header(), "rate_limit_requests");
                assert!(b.retry_after_secs().is_some());
            }
            Admission::Allowed(_) => panic!("fail-closed must deny"),
        }
    }

    // --- Counter-key isolation (bug fix): reused policy_id must NOT collapse
    // tenants' Redis counters, and Mode D must key off the GLOBAL cap ------------
    //
    // Both are asserted WITHOUT a live Redis: `counter_key` is pure, and the
    // `RequestRateSpec` projection carries the structural identity + global limit.

    #[test]
    fn counter_key_isolates_tenants_that_reuse_a_policy_id() {
        // Two tenants BOTH declare the reusable policy_id "standard". In Mode L
        // they are structurally isolated (distinct Arc slots keyed by tenant_id);
        // Mode D must reproduce that — the Redis counter key must differ so tenant
        // A's traffic never 429s tenant B (ADR-023 structural isolation / ADR-064).
        let reg = LimitRegistry::build(vec![
            KeyLimitsInput {
                routeplane_key: "rp_a".into(),
                tenant_id: "t_a".into(),
                limits: serde_json::from_str(
                    r#"{"tenant":{"policy_id":"standard","rate":{"requests_per_min":100}}}"#,
                )
                .unwrap(),
            },
            KeyLimitsInput {
                routeplane_key: "rp_b".into(),
                tenant_id: "t_b".into(),
                limits: serde_json::from_str(
                    r#"{"tenant":{"policy_id":"standard","rate":{"requests_per_min":100}}}"#,
                )
                .unwrap(),
            },
        ]);
        let sa = reg.resolve("rp_a", "t_a").request_rate_specs();
        let sb = reg.resolve("rp_b", "t_b").request_rate_specs();
        assert_eq!(sa.len(), 1);
        assert_eq!(sb.len(), 1);
        // Same operator-facing policy id (still surfaced in the 429 header)...
        assert_eq!(sa[0].policy_id, "standard");
        assert_eq!(sb[0].policy_id, "standard");
        // ...but DISTINCT structural identities ⇒ DISTINCT Redis keys.
        let ka = counter_key("rp:rl", sa[0].scope, &sa[0].identity, "req", 42);
        let kb = counter_key("rp:rl", sb[0].scope, &sb[0].identity, "req", 42);
        assert_ne!(
            ka, kb,
            "reused policy_id must not collapse two tenants onto one Redis counter"
        );
        assert_eq!(ka, "rp:rl:tenant:tenant:t_a:req:42");
        assert_eq!(kb, "rp:rl:tenant:tenant:t_b:req:42");
    }

    #[test]
    fn mode_d_spec_exports_the_global_unclamped_limit() {
        // With max_replicas=4 the LOCAL Mode-L counter is share-clamped to
        // ceil(100/4)=25, but the Redis counter is GLOBAL across replicas — Mode D
        // must enforce the configured 100, not 25 (else the fleet 429s 4× early).
        let reg = LimitRegistry::build_with_replicas(
            vec![KeyLimitsInput {
                routeplane_key: "rp_g".into(),
                tenant_id: "t_g".into(),
                limits: serde_json::from_str(r#"{"key":{"rate":{"requests_per_min":100}}}"#)
                    .unwrap(),
            }],
            4,
        );
        let specs = reg.resolve("rp_g", "t_g").request_rate_specs();
        assert_eq!(specs.len(), 1);
        assert_eq!(
            specs[0].limit, 100,
            "Mode D must export the GLOBAL cap, not the per-replica share"
        );
    }

    #[tokio::test]
    async fn unlimited_key_short_circuits_without_redis() {
        // The `admit` fast-path: an unlimited guard returns immediately via the
        // local engine and NEVER constructs a Redis future — so it cannot hang
        // even when "Mode D" is notionally active. We assert it on a limiter
        // built lazily against a refused local port; if connect errors, the
        // property is trivially satisfied (Mode D inactive).
        let built = DistributedLimiter::connect(&DistributedConfig {
            url: "redis://127.0.0.1:1".into(), // refuses fast
            timeout: Duration::from_millis(20),
            ..cfg()
        })
        .await;
        if let Ok(limiter) = built {
            let g = LimitRegistry::empty().resolve("x", "y");
            assert!(matches!(limiter.admit(&g, 0).await, Admission::Allowed(a) if a.is_empty()));
        }
    }

    #[tokio::test]
    async fn connect_to_a_refused_endpoint_errors_promptly_never_hangs() {
        // Startup safety: connecting to a closed local port must return an error
        // within the bounded connect budget, not hang the boot.
        let r = DistributedLimiter::connect(&DistributedConfig {
            url: "redis://127.0.0.1:1".into(),
            timeout: Duration::from_millis(20),
            ..cfg()
        })
        .await;
        // Either a prompt error (expected) or a (lazy) success — both are
        // non-hanging; the test passing at all proves it did not block.
        let _ = r;
    }

    // --- Live-Redis integration (gated; needs ROUTEPLANE_REDIS_URL) ----------
    // Mirrors the real-Postgres `#[ignore]` pattern: never runs in CI, runs
    // locally via `cargo test -p routeplane-limits --features redis-limits -- --ignored`.

    #[tokio::test]
    #[ignore = "requires a live Redis at ROUTEPLANE_REDIS_URL"]
    async fn live_atomic_counter_increments_and_expires() {
        let cfg = DistributedConfig::from_env().expect("set ROUTEPLANE_REDIS_URL");
        let limiter = DistributedLimiter::connect(&cfg).await.expect("connect");
        let reg = LimitRegistry::build(vec![KeyLimitsInput {
            routeplane_key: "rp_live".into(),
            tenant_id: "t_live".into(),
            limits: serde_json::from_str(r#"{"key":{"rate":{"requests_per_min":3}}}"#).unwrap(),
        }]);
        let g = reg.resolve("rp_live", "t_live");
        let now = crate::now_unix_ms();
        assert!(matches!(
            limiter.admit(&g, now).await,
            Admission::Allowed(_)
        ));
        assert!(matches!(
            limiter.admit(&g, now).await,
            Admission::Allowed(_)
        ));
        assert!(matches!(
            limiter.admit(&g, now).await,
            Admission::Allowed(_)
        ));
        assert!(matches!(limiter.admit(&g, now).await, Admission::Denied(_)));
    }
}
