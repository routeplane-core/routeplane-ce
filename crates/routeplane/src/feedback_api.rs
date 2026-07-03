//! `POST /v1/feedback` — the Portkey/Helicone-compatible feedback route (PARITY:
//! Portkey ships a Feedback API that attaches a weighted score to a request
//! trace; Helicone has feedback/scoring). It pairs with the prompt A/B testing +
//! analytics already shipped — variant analytics need a quality signal to learn
//! from, and this is that signal.
//!
//! ## Contract (mirrors Portkey)
//! `POST /v1/feedback`, body
//! `{ "trace_id": <str>, "value": <int -10..=10>, "weight": <float 0.0..=1.0,
//! default 1.0>, "metadata": <optional object> }`. On success it returns a small
//! JSON ack `{ "status": "recorded", "trace_id": "..." }`. Validation failures
//! return the OpenAI-style error envelope with a 400 (the same envelope the rest
//! of the gateway emits) so an SDK surfaces a clean typed error.
//!
//! ## Where the trace_id comes from
//! The proxy now emits `x-routeplane-trace-id` (the per-request `req_<uuid>`
//! correlation id it already generated internally) on chat + streaming responses
//! — additive, so the golden/A-B parity corpus stays a subset of the current
//! headers. A client reads that header off a completion and feeds it back here.
//! The gateway does NOT verify the trace exists (it would need a request-id index
//! the in-memory ring does not keep, and a non-blocking analytics signal should
//! not 404 on a slightly-stale id) — it validates shape + records.
//!
//! ## Off the hot path, in-memory, frugal
//! Feedback is recorded OFF any provider path: it goes into the same in-memory
//! observability ring (last 1000 events) as a synthetic `(feedback)` join event
//! via the lock-free `record_usage` (a single bounded `try_send`). No DB, no new
//! standing cost — identical posture to the existing observability surface, so no
//! ADR is required (feedback/eval signal sits in PRD-009/PRD-010 scope). Adding
//! *durable* feedback persistence later WOULD need an ADR (No-DB-without-an-ADR).
//!
//! ## No-raw-PII / bounded metadata
//! `metadata` is caller-supplied and could carry PII or grow unbounded. It is
//! therefore NEVER persisted raw and NEVER routed into the tamper-evident ledger
//! or any no-raw-PII surface. The handler caps the number of keys and the
//! per-string length, and records only a COUNT of accepted keys into the ring
//! (the in-memory analytics surface), not the values. Oversized metadata is
//! rejected with a 400 rather than silently truncated.

use crate::auth::{TenantContext, VirtualKey};
use crate::observability::UsageEvent;
use crate::proxy::AppState;
use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    Extension, Json,
};
use serde_json::json;
use std::sync::Arc;

use routeplane_types::FeedbackRequest;

/// Max accepted `trace_id` length — generous vs the `req_<uuid32>` ids the
/// gateway emits (~36 chars) but a hard bound so a caller cannot push an
/// arbitrarily long string into the in-memory ring.
const MAX_TRACE_ID_LEN: usize = 256;

/// Bound on caller `metadata`: at most this many keys, each key + string value
/// capped at [`MAX_METADATA_STR_LEN`]. Oversized metadata is a 400 (rejected,
/// not silently truncated) so the caller knows their payload was not stored.
const MAX_METADATA_KEYS: usize = 16;
const MAX_METADATA_STR_LEN: usize = 256;

pub async fn feedback(
    State(state): State<Arc<AppState>>,
    Extension(virtual_key): Extension<VirtualKey>,
    // Tenant context is injected by `auth_middleware`; its presence is the
    // auth gate (a missing key never reaches here — middleware returns 401).
    Extension(_tenant_ctx): Extension<TenantContext>,
    crate::api_error::OpenAiJson(payload): crate::api_error::OpenAiJson<FeedbackRequest>,
) -> Response {
    // 1. Validate `trace_id`: non-empty + length-bounded.
    let trace_id = payload.trace_id.trim();
    if trace_id.is_empty() {
        return invalid("`trace_id` must be a non-empty string.", "trace_id");
    }
    if trace_id.len() > MAX_TRACE_ID_LEN {
        return invalid(
            "`trace_id` exceeds the maximum length of 256 characters.",
            "trace_id",
        );
    }

    // 2. Validate `value` in -10..=10. `i8` already bounds the wire to
    //    -128..=127; this enforces the Portkey range.
    if !(-10..=10).contains(&payload.value) {
        return invalid("`value` must be an integer between -10 and 10.", "value");
    }

    // 3. Validate `weight` in 0.0..=1.0 (default 1.0 when omitted). Reject NaN
    //    explicitly (a NaN fails every comparison and must not slip through).
    let weight = match payload.weight {
        None => 1.0_f32,
        Some(w) if w.is_finite() && (0.0..=1.0).contains(&w) => w,
        Some(_) => {
            return invalid("`weight` must be a number between 0.0 and 1.0.", "weight");
        }
    };

    // 4. Bound + label-clean `metadata`. We DO NOT persist the values — only a
    //    count of accepted keys lands in the ring. Oversized/ill-shaped metadata
    //    is rejected (never silently truncated) so retention is bounded AND the
    //    caller is told. Raw caller bytes never reach an audit-grade surface.
    let metadata_keys = match bound_metadata(payload.metadata.as_ref()) {
        Ok(n) => n,
        Err((message, param)) => return invalid(message, param),
    };

    // 5. Record OFF the hot path: a single lock-free `try_send` into the
    //    in-memory observability ring as a synthetic `(feedback)` event. This is
    //    NOT a provider call and touches no ledger / no-raw-PII surface. The
    //    tenant is identified by key ownership (virtual_key.name), the same
    //    tenant-isolation basis the analytics/chargeback surfaces already use.
    state
        .observability_engine
        .record_usage(UsageEvent::feedback(
            virtual_key.name.clone(),
            trace_id.to_string(),
            payload.value,
            weight,
            metadata_keys,
        ));

    (
        StatusCode::OK,
        Json(json!({ "status": "recorded", "trace_id": trace_id })),
    )
        .into_response()
}

/// Validate + bound caller `metadata`, returning the count of accepted keys.
/// Enforces: an object (not a scalar/array), at most [`MAX_METADATA_KEYS`] keys,
/// and every key + string value at most [`MAX_METADATA_STR_LEN`] bytes. The
/// values themselves are NOT retained — this only gates retention to a bounded,
/// label-safe shape and returns the key count for the analytics ring.
fn bound_metadata(
    metadata: Option<&serde_json::Value>,
) -> Result<u32, (&'static str, &'static str)> {
    let Some(value) = metadata else {
        return Ok(0);
    };
    // `null` is treated as "no metadata".
    if value.is_null() {
        return Ok(0);
    }
    let obj = value
        .as_object()
        .ok_or(("`metadata` must be a JSON object.", "metadata"))?;
    if obj.len() > MAX_METADATA_KEYS {
        return Err(("`metadata` must contain at most 16 keys.", "metadata"));
    }
    for (k, v) in obj {
        if k.len() > MAX_METADATA_STR_LEN {
            return Err((
                "`metadata` keys must be at most 256 characters.",
                "metadata",
            ));
        }
        // Cap string values; non-string scalars (number/bool/null) are bounded
        // by construction. Nested objects/arrays are rejected — feedback metadata
        // is a flat label map, not a document store.
        match v {
            serde_json::Value::String(s) if s.len() > MAX_METADATA_STR_LEN => {
                return Err((
                    "`metadata` string values must be at most 256 characters.",
                    "metadata",
                ));
            }
            serde_json::Value::Object(_) | serde_json::Value::Array(_) => {
                return Err((
                    "`metadata` values must be scalars (no nested objects or arrays).",
                    "metadata",
                ));
            }
            _ => {}
        }
    }
    Ok(obj.len() as u32)
}

/// A 400 with the OpenAI-style invalid-request envelope, matching the rest of
/// the gateway's error surface.
fn invalid(message: &str, param: &str) -> Response {
    crate::api_error::error_response(
        StatusCode::BAD_REQUEST,
        "invalid_request_error",
        message,
        "invalid_request_error",
        Some(param),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(pairs: &[(&str, serde_json::Value)]) -> serde_json::Value {
        let mut m = serde_json::Map::new();
        for (k, v) in pairs {
            m.insert((*k).to_string(), v.clone());
        }
        serde_json::Value::Object(m)
    }

    #[test]
    fn bound_metadata_none_and_null_are_zero() {
        assert_eq!(bound_metadata(None), Ok(0));
        assert_eq!(bound_metadata(Some(&serde_json::Value::Null)), Ok(0));
    }

    #[test]
    fn bound_metadata_counts_accepted_keys() {
        let m = meta(&[
            ("source", json!("eval")),
            ("rating", json!(5)),
            ("ok", json!(true)),
        ]);
        assert_eq!(bound_metadata(Some(&m)), Ok(3));
    }

    #[test]
    fn bound_metadata_rejects_non_object() {
        assert!(bound_metadata(Some(&json!("a string"))).is_err());
        assert!(bound_metadata(Some(&json!([1, 2, 3]))).is_err());
    }

    #[test]
    fn bound_metadata_rejects_too_many_keys() {
        let pairs: Vec<(String, serde_json::Value)> = (0..MAX_METADATA_KEYS + 1)
            .map(|i| (format!("k{i}"), json!(i)))
            .collect();
        let mut m = serde_json::Map::new();
        for (k, v) in &pairs {
            m.insert(k.clone(), v.clone());
        }
        assert!(bound_metadata(Some(&serde_json::Value::Object(m))).is_err());
    }

    #[test]
    fn bound_metadata_rejects_oversized_value_and_nested() {
        let big = "x".repeat(MAX_METADATA_STR_LEN + 1);
        assert!(bound_metadata(Some(&meta(&[("k", json!(big))]))).is_err());
        assert!(bound_metadata(Some(&meta(&[("k", json!({"nested": 1}))]))).is_err());
    }
}
