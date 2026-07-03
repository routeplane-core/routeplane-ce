//! Runtime tunables for the gateway, read once at startup from the environment
//! with production-sane defaults. Centralised here so the timeout/deadline and
//! the Tower-stack limits (concurrency, load-shed, body size, request timeout)
//! have a single, documented source of truth instead of magic numbers scattered
//! across `proxy.rs` and `main.rs`.
//!
//! Everything is overridable via env (no rebuild to tune a deployment), and
//! every value falls back to a default if the var is missing or unparseable.

use std::time::Duration;

/// Deadline / per-attempt timeout policy for the provider fallback chain.
#[derive(Debug, Clone, Copy)]
pub struct DeadlineConfig {
    /// Total wall-clock budget for the whole request across ALL fallback
    /// attempts. Each attempt runs under `min(per_attempt, remaining)`; once the
    /// deadline is spent, no further candidates are tried.
    pub request_deadline: Duration,
    /// Per-attempt cap on a single provider call (buffered request, or
    /// time-to-first-chunk for streaming). Shrinks as the deadline is consumed.
    pub per_attempt_timeout: Duration,
}

impl Default for DeadlineConfig {
    fn default() -> Self {
        Self {
            request_deadline: Duration::from_millis(120_000),
            per_attempt_timeout: Duration::from_millis(60_000),
        }
    }
}

impl DeadlineConfig {
    pub fn from_env() -> Self {
        let d = Self::default();
        Self {
            request_deadline: env_duration_ms("ROUTEPLANE_REQUEST_DEADLINE_MS", d.request_deadline),
            per_attempt_timeout: env_duration_ms(
                "ROUTEPLANE_PER_ATTEMPT_TIMEOUT_MS",
                d.per_attempt_timeout,
            ),
        }
    }
}

/// Limits applied by the Tower middleware stack in `main.rs`.
#[derive(Debug, Clone, Copy)]
pub struct ServerLimits {
    /// Max concurrent in-flight requests before load-shedding to 503.
    pub max_concurrency: usize,
    /// Hard request-handling timeout enforced by `tower_http::TimeoutLayer`
    /// (a backstop above the provider-chain deadline, covering the whole handler
    /// including body read + middleware).
    pub request_timeout: Duration,
    /// Max accepted request body size in bytes (reject oversized payloads with
    /// 413 before they are buffered into memory).
    pub max_body_bytes: usize,
    /// Max accepted body size for the audio-transcription route
    /// (`/v1/audio/transcriptions`). Audio uploads are large (OpenAI caps STT at
    /// 25 MB), so the small global `max_body_bytes` (2 MiB, sized for chat JSON)
    /// would 413 every real audio file. This LARGER cap is applied ONLY to the
    /// audio route via its own `RequestBodyLimitLayer` — the global limit is
    /// unchanged for every other route.
    pub audio_max_body_bytes: usize,
}

impl Default for ServerLimits {
    fn default() -> Self {
        Self {
            // Scale-to-zero serverless: a single replica should shed rather than
            // OOM/queue unboundedly. 256 in-flight is a safe Alpha default.
            max_concurrency: 256,
            // Above the 120s provider deadline so the deadline (not this) is the
            // primary control, while still bounding a wedged handler.
            request_timeout: Duration::from_millis(150_000),
            // 2 MiB: generous for chat payloads, small enough to stop abuse.
            max_body_bytes: 2 * 1024 * 1024,
            // 26 MiB: matches OpenAI's 25 MB STT cap with multipart overhead
            // headroom. Applied ONLY to /v1/audio/transcriptions.
            audio_max_body_bytes: 26 * 1024 * 1024,
        }
    }
}

impl ServerLimits {
    pub fn from_env() -> Self {
        let d = Self::default();
        Self {
            max_concurrency: env_usize("ROUTEPLANE_MAX_CONCURRENCY", d.max_concurrency),
            request_timeout: env_duration_ms("ROUTEPLANE_REQUEST_TIMEOUT_MS", d.request_timeout),
            max_body_bytes: env_usize("ROUTEPLANE_MAX_BODY_BYTES", d.max_body_bytes),
            audio_max_body_bytes: env_usize("RP_AUDIO_MAX_BODY_BYTES", d.audio_max_body_bytes),
        }
    }
}

/// Limits for Guardrails v2 webhook checks (G2.6). Deliberately strict: a
/// webhook is on the request path only when a tenant configured one, and it
/// must never become an amplification/exfil channel — short timeout, small
/// response cap, no redirects (enforced in `webhook_client.rs`).
#[derive(Debug, Clone, Copy)]
pub struct GuardrailWebhookLimits {
    /// End-to-end cap on one webhook call (connect + TLS + body).
    pub timeout: Duration,
    /// Max accepted webhook response size in bytes (the verdict object is tiny;
    /// anything large is a misbehaving or hostile endpoint).
    pub max_response_bytes: usize,
}

impl Default for GuardrailWebhookLimits {
    fn default() -> Self {
        Self {
            timeout: Duration::from_millis(3_000),
            max_response_bytes: 64 * 1024,
        }
    }
}

impl GuardrailWebhookLimits {
    pub fn from_env() -> Self {
        let d = Self::default();
        Self {
            timeout: env_duration_ms("ROUTEPLANE_GUARDRAIL_WEBHOOK_TIMEOUT_MS", d.timeout),
            max_response_bytes: env_usize(
                "ROUTEPLANE_GUARDRAIL_WEBHOOK_MAX_BYTES",
                d.max_response_bytes,
            ),
        }
    }
}

/// Rung-0 exact-match cache settings (G2.5 / ADR-022 §3). The per-replica byte
/// budget is a CELL parameter (tfvars → container env var, ADR-013: cells
/// differ by tfvars + entitlements only): 64 MiB default for pool-std /
/// dedicated cells, 16 MiB for pool-free (set ROUTEPLANE_CACHE_BUDGET_BYTES in
/// the cell's tfvars). Scale-to-zero resets the cache — accepted and
/// documented; the lost-continuity telemetry is the rung-1 trigger input.
#[derive(Debug, Clone, Copy)]
pub struct CacheSettings {
    /// Total per-replica budget in bytes, enforced at write time across the
    /// cache's 64 shards. Never grows unbounded (PRD-007 NFR-3).
    pub budget_bytes: usize,
}

impl Default for CacheSettings {
    fn default() -> Self {
        Self {
            budget_bytes: routeplane_cache::DEFAULT_BUDGET_BYTES,
        }
    }
}

impl CacheSettings {
    pub fn from_env() -> Self {
        let d = Self::default();
        Self {
            budget_bytes: env_usize("ROUTEPLANE_CACHE_BUDGET_BYTES", d.budget_bytes),
        }
    }
}

fn env_duration_ms(var: &str, default: Duration) -> Duration {
    std::env::var(var)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|v| *v > 0)
        .map(Duration::from_millis)
        .unwrap_or(default)
}

fn env_usize(var: &str, default: usize) -> usize {
    std::env::var(var)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(default)
}
