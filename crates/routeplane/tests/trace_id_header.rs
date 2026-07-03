//! Asserts the additive `x-routeplane-trace-id` response header (the per-request
//! correlation id the Feedback API references) actually ships on a normal
//! buffered chat completion. The golden snapshot strips this header (its value is
//! a random `req_<uuid>` per request and its presence is additive), so a
//! dedicated test must prove it is emitted — and that the value is the
//! `req_`-prefixed correlation id a client can feed to `POST /v1/feedback`.

mod common;

use axum::body::to_bytes;
use common::{build_stub_state, drive_buffered_resp};

#[tokio::test]
async fn trace_id_header_present_on_buffered_success() {
    let state = build_stub_state();
    let resp = drive_buffered_resp(&state).await;
    assert_eq!(resp.status(), axum::http::StatusCode::OK);

    let trace = resp
        .headers()
        .get("x-routeplane-trace-id")
        .expect("x-routeplane-trace-id header present on a normal response")
        .to_str()
        .expect("trace id is valid ascii")
        .to_string();
    // It is the gateway's per-request correlation id — `req_<uuid>` — which is
    // exactly the value a client passes back as the feedback `trace_id`.
    assert!(
        trace.starts_with("req_"),
        "trace id should be the req_<uuid> correlation id, got {trace}"
    );
    assert!(trace.len() > 4);

    // PRD-009 (#160), re-added: the SAME correlation id is also emitted as
    // `x-routeplane-request-id` (the pre-#170 header name), so a client using
    // either name correlates the identical request.
    let request_id = resp
        .headers()
        .get("x-routeplane-request-id")
        .expect("x-routeplane-request-id header present on a normal response")
        .to_str()
        .expect("request id is valid ascii");
    assert_eq!(
        request_id, trace,
        "request-id and trace-id carry the same value"
    );

    // Drain the body (hygiene).
    let _ = to_bytes(resp.into_body(), usize::MAX).await;
}
