//! Shared `reqwest::Client` construction for every provider adapter.
//!
//! A single, pooled `reqwest::Client` per provider is built once at startup and
//! reused for the life of the process (it pools connections internally — never
//! construct one per request). This module centralises the *timeout* policy so
//! every adapter inherits the same defence-in-depth bounds:
//!
//!   * `connect_timeout` — cap on establishing the TCP+TLS connection. A
//!     provider whose endpoint is black-holed must fail fast, not hang forever.
//!   * `timeout` — overall request cap. This is the adapter-local backstop; the
//!     proxy *also* wraps every call in a `tokio::time::timeout` under the
//!     request-level deadline (see `crates/routeplane/src/proxy.rs`). Having
//!     both means a hang can never wedge a worker even if the proxy-side guard
//!     is ever bypassed — a hung socket becomes a recorded error, which is what
//!     lets the circuit breaker actually trip.
//!
//! Defaults are deliberately generous (LLM completions are slow) but finite, and
//! are overridable via environment so they can be tuned per deployment without a
//! rebuild:
//!
//!   * `ROUTEPLANE_PROVIDER_CONNECT_TIMEOUT_MS` (default 5_000)
//!   * `ROUTEPLANE_PROVIDER_REQUEST_TIMEOUT_MS` (default 120_000)

use crate::ProviderError;
use reqwest::Client;
use std::time::Duration;

const DEFAULT_CONNECT_TIMEOUT_MS: u64 = 5_000;
const DEFAULT_REQUEST_TIMEOUT_MS: u64 = 120_000;

fn env_ms(var: &str, default: u64) -> u64 {
    std::env::var(var)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(default)
}

/// Build the shared, pooled client every adapter uses, with finite connect and
/// overall-request timeouts applied. Falls back to a default `Client` only if
/// the builder somehow fails (it should not with these options) so adapter
/// construction stays infallible.
pub fn build_provider_client() -> Client {
    let connect_ms = env_ms(
        "ROUTEPLANE_PROVIDER_CONNECT_TIMEOUT_MS",
        DEFAULT_CONNECT_TIMEOUT_MS,
    );
    let request_ms = env_ms(
        "ROUTEPLANE_PROVIDER_REQUEST_TIMEOUT_MS",
        DEFAULT_REQUEST_TIMEOUT_MS,
    );

    Client::builder()
        .connect_timeout(Duration::from_millis(connect_ms))
        .timeout(Duration::from_millis(request_ms))
        .build()
        .unwrap_or_default()
}

/// Shared pooled client for STREAMING provider calls: finite connect timeout,
/// but NO whole-body timeout — a legitimate SSE generation can exceed any total
/// cap (the old shared 120s `timeout` deterministically killed every stream
/// longer than 120s mid-body, which the proxy then masked as a clean end).
/// Liveness is enforced where it belongs: the proxy bounds the gap BETWEEN
/// chunks (`ROUTEPLANE_STREAM_IDLE_TIMEOUT_MS`), so a hung upstream still
/// terminates while a slow-but-alive generation does not.
pub fn streaming_client() -> Client {
    static CLIENT: std::sync::OnceLock<Client> = std::sync::OnceLock::new();
    CLIENT
        .get_or_init(|| {
            let connect_ms = env_ms(
                "ROUTEPLANE_PROVIDER_CONNECT_TIMEOUT_MS",
                DEFAULT_CONNECT_TIMEOUT_MS,
            );
            Client::builder()
                .connect_timeout(Duration::from_millis(connect_ms))
                .build()
                .unwrap_or_default()
        })
        .clone()
}

/// Convert a transport-level `reqwest::Error` into a typed [`ProviderError`]
/// whose message can NEVER echo a secret (Task #3d), classified for retry (G2.3).
///
/// `reqwest::Error`'s `Display` can include the request URL — and providers like
/// Gemini carry the API key in the URL query string (`?key=...`). Propagating
/// such an error with `?` would surface that key. We therefore (a) call
/// `.without_url()` to strip the URL, and (b) classify into a fixed, secret-free
/// variant. Timeouts → `Timeout`; connect/request → `Network` (both Always
/// retryable); decode/body → `Translation` (a malformed upstream body won't fix
/// on retry).
pub fn sanitize_transport_error(provider: &str, err: reqwest::Error) -> ProviderError {
    let err = err.without_url();
    if err.is_timeout() {
        ProviderError::timeout(provider, "request timed out")
    } else if err.is_connect() {
        ProviderError::network(provider, "connection failed")
    } else if err.is_request() {
        ProviderError::network(provider, "request error")
    } else if err.is_decode() {
        ProviderError::translation(format!("{provider}: response decode error"))
    } else if err.is_body() {
        ProviderError::translation(format!("{provider}: request/response body error"))
    } else {
        ProviderError::network(provider, "upstream transport error")
    }
}

/// Build a typed [`ProviderError`] from a non-success upstream response (G2.3).
/// Reads the status, an optional `Retry-After` (seconds form), and the body, then
/// classifies into the variant the retry loop reads. The body is retained in the
/// message for debugging but no key material ever appears (the URL is never
/// rendered, and provider auth is sent as a header/query the body does not echo).
pub async fn error_from_response(provider: &str, resp: reqwest::Response) -> ProviderError {
    let status = resp.status().as_u16();
    let retry_after = resp
        .headers()
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(Duration::from_secs);
    // Bound the captured body (security review: a hostile upstream can return
    // megabyte error pages, and bodies flow into client-visible error text).
    let body = resp.text().await.unwrap_or_default();
    let body = truncate_body(body);
    classify_status(provider, status, retry_after, body)
}

const MAX_ERROR_BODY_BYTES: usize = 2048;

fn truncate_body(mut body: String) -> String {
    if body.len() > MAX_ERROR_BODY_BYTES {
        let mut cut = MAX_ERROR_BODY_BYTES;
        while cut > 0 && !body.is_char_boundary(cut) {
            cut -= 1;
        }
        body.truncate(cut);
        body.push_str("…[truncated]");
    }
    body
}

fn classify_status(
    provider: &str,
    status: u16,
    retry_after: Option<Duration>,
    body: String,
) -> ProviderError {
    if status == 401 || status == 403 {
        ProviderError::Auth {
            provider: provider.to_string(),
            status,
            body,
        }
    } else if status == 408 {
        ProviderError::Timeout {
            provider: provider.to_string(),
            detail: if body.is_empty() {
                "upstream returned 408".to_string()
            } else {
                format!("upstream 408: {body}")
            },
        }
    } else if status == 429 {
        ProviderError::RateLimited {
            provider: provider.to_string(),
            retry_after,
            body,
        }
    } else if (500..=599).contains(&status) {
        ProviderError::Upstream5xx {
            provider: provider.to_string(),
            status,
            body,
        }
    } else {
        ProviderError::BadRequest {
            provider: provider.to_string(),
            status,
            body,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RetryClass;

    #[test]
    fn classifies_statuses_into_retry_classes() {
        assert_eq!(
            classify_status("openai", 429, None, "rl".into()).retry_class(),
            RetryClass::Status(429)
        );
        assert_eq!(
            classify_status("openai", 503, None, "down".into()).retry_class(),
            RetryClass::Status(503)
        );
        assert_eq!(
            classify_status("openai", 401, None, "bad key".into()).retry_class(),
            RetryClass::Never
        );
        assert_eq!(
            classify_status("openai", 400, None, "bad req".into()).retry_class(),
            RetryClass::Never
        );
        assert_eq!(
            classify_status("openai", 408, None, "".into()).retry_class(),
            RetryClass::Always
        );
    }

    #[test]
    fn rate_limited_message_carries_status_not_secrets() {
        let e = classify_status(
            "openai",
            429,
            Some(Duration::from_secs(2)),
            "slow down".into(),
        );
        let msg = e.to_string();
        assert!(msg.contains("429"));
        assert!(msg.contains("slow down"));
        assert_eq!(e.status(), Some(429));
    }
}
