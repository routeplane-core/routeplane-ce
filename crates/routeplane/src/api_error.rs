//! OpenAI-shaped error envelopes for the gateway's error path.
//!
//! The success path is byte-compatible with the OpenAI API; the error path must
//! be too, or an OpenAI SDK pointed at the gateway throws a body-parse error
//! instead of surfacing a clean typed error (`AuthenticationError`,
//! `BadRequestError`, …). This is the one place auth failures, request-body
//! rejections, and upstream-exhaustion failures are rendered as
//! `{"error": {message, type, code, param}}`, matching the envelopes the
//! limits/guardrails/prompts code already emits. (Found by the 2026-06-12 live
//! dogfood: 401 returned an empty body, malformed JSON / all-failed were
//! plaintext, and a missing field returned 422 where OpenAI uses 400.)

use axum::{
    extract::{rejection::JsonRejection, FromRequest, Request},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::de::DeserializeOwned;

/// Build an OpenAI-shaped error response.
pub fn error_response(
    status: StatusCode,
    code: &str,
    message: impl Into<String>,
    error_type: &str,
    param: Option<&str>,
) -> Response {
    let body = serde_json::json!({
        "error": {
            "message": message.into(),
            "type": error_type,
            "param": param,
            "code": code,
        }
    });
    (status, Json(body)).into_response()
}

/// 401 — missing or invalid `x-routeplane-api-key`. OpenAI-shaped so an SDK
/// raises a clean AuthenticationError instead of choking on an empty body.
pub fn unauthorized() -> Response {
    error_response(
        StatusCode::UNAUTHORIZED,
        "invalid_api_key",
        "Missing or invalid x-routeplane-api-key.",
        "invalid_request_error",
        None,
    )
}

/// 422 — sovereign-residency refusal: the request carries regulated personal
/// data but no provider resident in the required `region` is configured, so the
/// gateway refuses rather than route regulated data out of jurisdiction. This is
/// the flagship compliance path (DPDP / India-first), so it gets a FULL OpenAI
/// envelope with the machine-branchable code `routeplane_sovereign_block` —
/// clients can detect a residency refusal programmatically (it was bare
/// `text/plain` before, so SDKs saw a typed 422 with an unparseable body).
pub fn sovereign_block(region: &str) -> Response {
    error_response(
        StatusCode::UNPROCESSABLE_ENTITY,
        "routeplane_sovereign_block",
        format!(
            "Sovereign routing: request contains personal data but no {region}-resident provider is configured"
        ),
        "invalid_request_error",
        None,
    )
}

/// 500 — a programmer/config error (e.g. a missing required extension). Generic
/// body; the cause is logged, never disclosed.
pub fn internal_error() -> Response {
    error_response(
        StatusCode::INTERNAL_SERVER_ERROR,
        "internal_error",
        "The gateway encountered an internal error.",
        "api_error",
        None,
    )
}

/// 500 — every eligible upstream provider failed. The internal cause (provider
/// error, missing key, timeout) is logged server-side and NEVER echoed to the
/// caller: the body must not disclose gateway configuration state (the dogfood
/// saw `"API key for openai not configured"` leak to the client).
pub fn upstream_all_failed() -> Response {
    error_response(
        StatusCode::INTERNAL_SERVER_ERROR,
        "upstream_error",
        "All eligible upstream providers failed to serve the request.",
        "api_error",
        None,
    )
}

/// 501 — a deliberately-unsupported endpoint (FR-13, PRD-011). Rather than the
/// bare unknown-route 404, the gateway returns an OpenAI-shaped decline that
/// NAMES `endpoint_not_supported` and POINTS the caller at the supported
/// endpoint, so an SDK surfaces a clean typed error instead of choking on an
/// unknown route. `type` is `invalid_request_error` — the decline is about the
/// request targeting an unsupported endpoint, so the caller should switch
/// endpoints, not retry.
pub fn endpoint_not_supported(message: impl Into<String>) -> Response {
    error_response(
        StatusCode::NOT_IMPLEMENTED,
        "endpoint_not_supported",
        message,
        "invalid_request_error",
        None,
    )
}

/// 403 — the org compliance-framework gate ([ADR-035] §4) excluded the requested
/// model: the model's `compliance_restrictions` intersect the tenant's
/// `compliance_frameworks` and the tenant is in `strict` mode. The message cites
/// the offending framework NAME(s) — §5 registry identifiers, which are config
/// strings, never user content (no-reflection-safe). OpenAI-shaped so an SDK
/// surfaces a clean `PermissionDeniedError`. The `routeplane_*` code is
/// `model_compliance_excluded`; `param` is `model` (the pinned, refused model).
///
/// This joins the gateway failure surface alongside the 422 sovereign-residency
/// refusal and the 446 guardrail denial.
pub fn model_compliance_excluded(model: &str, frameworks: &[&str]) -> Response {
    let list = frameworks.join(", ");
    error_response(
        StatusCode::FORBIDDEN,
        "model_compliance_excluded",
        format!(
            "Model '{model}' is excluded by your organization's compliance framework(s): {list}. \
             Select a model permitted under {list}, or contact your administrator."
        ),
        "invalid_request_error",
        Some("model"),
    )
}

/// 403 — the CP→DP config overlay ([ADR-063] / [PRD-039]) marks the requested
/// model explicitly DISABLED for the calling tenant. Distinct from the compliance
/// gate (which excludes a model by org framework): this is a per-tenant operator
/// toggle authored in the Console and distributed to the gateway. OpenAI-shaped so
/// an SDK surfaces a clean `PermissionDeniedError`; the `routeplane_*` code is
/// `model_disabled_for_tenant` and `param` is `model`. The message names only the
/// model id (the request's own value — no other tenant/config state is disclosed).
///
/// Default-allow + fail-open posture: this is returned ONLY when the overlay holds
/// an explicit `enabled = false` for `(tenant, model)`. No overlay entry ⇒ allow
/// (so the disabled gateway, with no poller, never returns this).
pub fn model_disabled_for_tenant(model: &str) -> Response {
    error_response(
        StatusCode::FORBIDDEN,
        "model_disabled_for_tenant",
        format!(
            "Model '{model}' is disabled for this tenant. \
             Enable it in the Routeplane Console or select a different model."
        ),
        "invalid_request_error",
        Some("model"),
    )
}

/// 402 — a Routeplane **Enterprise-only** endpoint called on the Community
/// Edition build. ONE uniform decline for the whole Enterprise surface
/// (`/v1/finops/usage`, `/v1/moderations`, `/v1/mcp/*`, `/v1/prompts*`,
/// `/v1/guardrails/outcomes`), replacing the previous mix of 403
/// `feature_not_entitled` / 405 / bare 404s.
///
/// **Why 402 Payment Required:** the request is well-formed and authorization
/// is not the issue — the feature is simply not purchasable in this build. 402
/// is the clearest "upgrade needed" semantic, it is already in the gateway's
/// vocabulary (the budget-exceeded limit rejection uses it), and the error
/// infra (`error_response`) carries any `StatusCode`, so nothing forces a 403
/// fallback. `type` stays `invalid_request_error` so OpenAI SDKs surface a
/// clean typed error; the `x-routeplane-upgrade` header gives tooling a
/// machine-readable pointer (the `x-routeplane-*` convention).
#[cfg(not(feature = "enterprise"))]
pub fn enterprise_only(endpoint: &str) -> Response {
    let mut resp = error_response(
        StatusCode::PAYMENT_REQUIRED,
        "enterprise_only",
        format!(
            "{endpoint} is a Routeplane Enterprise feature and is not available in the \
             Community Edition. Learn more at https://routeplane.ai."
        ),
        "invalid_request_error",
        None,
    );
    resp.headers_mut().insert(
        "x-routeplane-upgrade",
        axum::http::HeaderValue::from_static("https://routeplane.ai"),
    );
    resp
}

/// Mount the Community Edition **Enterprise-only stub routes**: every
/// data-plane endpoint that exists only on the Enterprise build gets a
/// lightweight route answering [`enterprise_only`] (402) instead of the
/// accidental 404/405 an unmounted path produced (which, with the bundled
/// Console enabled, could even be the SPA fallback's `index.html`).
///
/// Mounted OUTSIDE the auth layer, deliberately: these paths required no key
/// to observe their 404/405 before (they were simply absent), so keeping the
/// stubs public preserves that posture and guarantees a caller sees
/// `enterprise_only` — never a 401 first. The body is a static upsell message
/// (no tenant/config state), so there is nothing to protect. Enterprise-only
/// endpoints that DO exist on the CE build (`/v1/finops/*`, `/v1/prompts/
/// {reference}*`) keep their existing authed mount and return the same
/// envelope from their in-handler gate.
///
/// `any(...)` handlers: the surface mixes GET and POST routes upstream, and a
/// wrong-method probe should get the same uniform decline, not a 405.
#[cfg(not(feature = "enterprise"))]
pub fn mount_enterprise_only_stubs<S: Clone + Send + Sync + 'static>(
    router: axum::Router<S>,
) -> axum::Router<S> {
    use axum::routing::any;

    /// Render the uniform decline, naming the endpoint actually called.
    async fn stub(uri: axum::http::Uri) -> Response {
        enterprise_only(uri.path())
    }

    router
        // Advanced-guardrails moat (ADR-088): the callable moderation endpoint
        // + the detection-telemetry read.
        .route("/v1/moderations", any(stub))
        .route("/v1/guardrails/outcomes", any(stub))
        // Agentic-security moat (ADR-016/055): the entire /v1/mcp/* surface
        // (13 routes + registry extensions on the Enterprise build) — one
        // wildcard covers them all, present and future.
        .route("/v1/mcp", any(stub))
        .route("/v1/mcp/{*rest}", any(stub))
        // Prompt-management collection surface (list/create — an Enterprise
        // Console feature; the CE build mounts only /v1/prompts/{reference}*,
        // which gate in-handler with the same envelope).
        .route("/v1/prompts", any(stub))
}

/// `Json<T>` with an OpenAI-shaped **400** on any extraction failure. axum's
/// stock `Json` rejects malformed bodies as plaintext 400 and *schema* failures
/// as 422; OpenAI uses 400 for both. This normalizes the status to 400 and
/// renders the envelope, so chat/embeddings emit identical, SDK-friendly body
/// errors. On success it is a transparent wrapper around the parsed value.
pub struct OpenAiJson<T>(pub T);

impl<T, S> FromRequest<S> for OpenAiJson<T>
where
    T: DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = Response;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        match Json::<T>::from_request(req, state).await {
            Ok(Json(value)) => Ok(OpenAiJson(value)),
            Err(rej) => Err(json_rejection_response(&rej)),
        }
    }
}

fn json_rejection_response(rej: &JsonRejection) -> Response {
    // axum's message names the failing field / parse position and carries no
    // secret, so it is passed through for a helpful error — but the status is
    // pinned to 400 (OpenAI's choice for both syntax and schema failures).
    error_response(
        StatusCode::BAD_REQUEST,
        "invalid_request_error",
        rej.body_text(),
        "invalid_request_error",
        None,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::{to_bytes, Body};
    use axum::http::Request as HttpRequest;

    async fn body_json(resp: Response) -> serde_json::Value {
        let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn unauthorized_is_openai_shaped() {
        let resp = unauthorized();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        let v = body_json(resp).await;
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert_eq!(v["error"]["code"], "invalid_api_key");
        assert!(v["error"]["message"].as_str().unwrap().contains("api-key"));
    }

    #[tokio::test]
    async fn all_failed_is_generic_and_leaks_no_config() {
        let resp = upstream_all_failed();
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let v = body_json(resp).await;
        assert_eq!(v["error"]["code"], "upstream_error");
        let msg = v["error"]["message"].as_str().unwrap().to_lowercase();
        // Must never disclose provider/config detail.
        assert!(!msg.contains("api key"));
        assert!(!msg.contains("openai"));
        assert!(!msg.contains("not configured"));
    }

    #[tokio::test]
    async fn compliance_excluded_is_403_and_cites_frameworks() {
        let resp = model_compliance_excluded("deepseek-chat", &["DPDP", "HIPAA"]);
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        let v = body_json(resp).await;
        assert_eq!(v["error"]["code"], "model_compliance_excluded");
        assert_eq!(v["error"]["param"], "model");
        let msg = v["error"]["message"].as_str().unwrap();
        // Cites framework NAMES (config identifiers — no-reflection-safe) and the model.
        assert!(msg.contains("DPDP"));
        assert!(msg.contains("HIPAA"));
        assert!(msg.contains("deepseek-chat"));
    }

    #[tokio::test]
    async fn model_disabled_is_403_and_names_model_only() {
        let resp = model_disabled_for_tenant("blocked-model");
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        let v = body_json(resp).await;
        assert_eq!(v["error"]["code"], "model_disabled_for_tenant");
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert_eq!(v["error"]["param"], "model");
        let msg = v["error"]["message"].as_str().unwrap();
        assert!(msg.contains("blocked-model"));
        // No tenant id / config state disclosed beyond the model the caller sent.
        assert!(msg.to_lowercase().contains("disabled"));
    }

    #[cfg(not(feature = "enterprise"))]
    #[tokio::test]
    async fn enterprise_only_is_402_with_uniform_envelope_and_upgrade_header() {
        let resp = enterprise_only("/v1/moderations");
        assert_eq!(resp.status(), StatusCode::PAYMENT_REQUIRED);
        assert_eq!(
            resp.headers()
                .get("x-routeplane-upgrade")
                .and_then(|v| v.to_str().ok()),
            Some("https://routeplane.ai")
        );
        let v = body_json(resp).await;
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert_eq!(v["error"]["code"], "enterprise_only");
        assert_eq!(v["error"]["param"], serde_json::Value::Null);
        let msg = v["error"]["message"].as_str().unwrap();
        assert!(msg.starts_with("/v1/moderations is a Routeplane Enterprise feature"));
        assert!(msg.contains("Community Edition"));
        assert!(msg.contains("https://routeplane.ai"));
    }

    #[cfg(not(feature = "enterprise"))]
    #[tokio::test]
    async fn every_enterprise_stub_route_returns_the_uniform_402() {
        use tower::ServiceExt as _;
        let app = mount_enterprise_only_stubs(axum::Router::<()>::new()).with_state(());

        // (method, path) pairs across the whole stubbed surface — including a
        // deep /v1/mcp/* path (the wildcard) and a wrong-method probe (was 405).
        let probes = [
            ("POST", "/v1/moderations"),
            ("GET", "/v1/moderations"),
            ("GET", "/v1/guardrails/outcomes"),
            ("GET", "/v1/mcp"),
            ("POST", "/v1/mcp/tool-call/authorize"),
            ("GET", "/v1/mcp/hitl/status/abc"),
            ("GET", "/v1/prompts"),
            ("POST", "/v1/prompts"),
        ];
        for (m, p) in probes {
            let req = HttpRequest::builder()
                .method(m)
                .uri(p)
                .body(Body::empty())
                .unwrap();
            let resp = app.clone().oneshot(req).await.unwrap();
            assert_eq!(
                resp.status(),
                StatusCode::PAYMENT_REQUIRED,
                "{m} {p} must be the uniform enterprise_only 402"
            );
            assert_eq!(
                resp.headers()
                    .get("x-routeplane-upgrade")
                    .and_then(|v| v.to_str().ok()),
                Some("https://routeplane.ai"),
                "{m} {p} must carry the upgrade header"
            );
            let v = body_json(resp).await;
            assert_eq!(v["error"]["code"], "enterprise_only", "{m} {p}");
            assert!(
                v["error"]["message"].as_str().unwrap().starts_with(p),
                "{m} {p}: message must name the endpoint called"
            );
        }
    }

    #[allow(dead_code)]
    #[derive(serde::Deserialize)]
    struct Demo {
        required: i32,
    }

    #[tokio::test]
    async fn malformed_json_is_400_envelope() {
        let req = HttpRequest::builder()
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from("{ not json"))
            .unwrap();
        let resp = OpenAiJson::<Demo>::from_request(req, &())
            .await
            .err()
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            body_json(resp).await["error"]["type"],
            "invalid_request_error"
        );
    }

    #[tokio::test]
    async fn missing_required_field_is_400_not_422() {
        // The dogfood crack: axum's Json returns 422 here. We pin 400 (OpenAI's
        // status for an invalid request body).
        let req = HttpRequest::builder()
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from("{}"))
            .unwrap();
        let resp = OpenAiJson::<Demo>::from_request(req, &())
            .await
            .err()
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }
}
