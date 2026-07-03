//! OpenTelemetry export skeleton (PRD-009).
//!
//! Translates [`UsageEvent`]s into OTLP-compatible JSON spans and optionally
//! pushes them to an OTLP HTTP endpoint. The skeleton emits the standard OTLP
//! JSON wire format (`ResourceSpans` → `ScopeSpans` → `Spans`) so a real
//! collector (OTel Collector, Jaeger, Grafana Tempo, SigNoz) can ingest it.
//!
//! # Scope
//! This build ships a **file/stdout exporter** — every usage event is rendered
//! as an OTLP JSON span and written to a configurable sink (stdout or a
//! rotating log file). The HTTP OTLP exporter is a seam (trait + env gate):
//! set `OTEL_EXPORTER_OTLP_ENDPOINT` to enable it; absent ⇒ stdout only.
//!
//! # Wire format
//! Each LLM event produces one OTLP span with:
//!   - `name`: `"{operation} {model}"` (e.g. `"chat gpt-4o"`) per the GenAI
//!     span-naming rule. Synthetic non-LLM sentinel events
//!     (`(prompt_render)` / `(feedback)` / `(sovereign_block)` / …) get a
//!     `"routeplane.<sentinel>"` name and carry NO `gen_ai.*` attributes, so
//!     downstream tools never mis-parse them as LLM spans.
//!   - `kind`: `SPAN_KIND_CLIENT` (we are the client of the upstream LLM)
//!   - Attributes: `gen_ai.operation.name`, `gen_ai.system`,
//!     `gen_ai.provider.name`, `gen_ai.request.model`,
//!     `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`,
//!     `gen_ai.usage.total_tokens`, plus the `routeplane.*` custom attributes
//!     (`routeplane.tenant`, `routeplane.region`,
//!     `routeplane.sovereign_routed`, `routeplane.cache_status`, …).
//!
//! The attributes follow the [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
//! (the `gen_ai.*` namespace) for LLM observability — the stabilized token
//! attribute names (`input_tokens`/`output_tokens`) and `gen_ai.provider.name`
//! that Datadog LLM Observability, Arize, Langfuse, and Honeycomb auto-parse.
//! `gen_ai.system` is retained alongside `gen_ai.provider.name` for back-compat
//! with older collectors.
//!
//! # Status
//! Translation + rendering are production-ready and tested, wired into the
//! `record_usage` path, and now export over **OTLP/HTTP JSON**: when an endpoint
//! is configured (`OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`, or
//! `OTEL_EXPORTER_OTLP_ENDPOINT` + `/v1/traces`), the span is POSTed to the
//! collector on a detached background task — **never on the request path** (the
//! hot path is lock-free/no-hop; export is best-effort and must not add latency
//! or a reliability coupling). Export is gated behind `OTEL_EXPORT_ENABLED=true`
//! (default: disabled ⇒ `export_event` is a no-op); with no endpoint set it
//! falls back to a debug log.

#![allow(dead_code)]

use crate::observability::UsageEvent;
use serde::Serialize;

/// The OpenTelemetry span kind. We are always the CLIENT of the upstream LLM.
const SPAN_KIND_CLIENT: i32 = 2;

/// Status code: 0 = UNSET, 1 = OK, 2 = ERROR.
const STATUS_OK: i32 = 1;
const STATUS_ERROR: i32 = 2;

/// One OTLP span attribute.
#[derive(Debug, Clone, Serialize)]
pub struct OtelAttribute {
    key: String,
    value: OtelValue,
}

// The `…Value` postfix is intentional: these mirror OTLP `AnyValue`'s field
// names (stringValue/intValue/boolValue/doubleValue), so the shared suffix is
// the spec, not a smell. `camelCase` emits exactly those OTLP/HTTP-JSON keys
// (e.g. `{"stringValue": …}`) — matching the hand-written resource attributes
// and what collectors require; `lowercase` would emit `stringvalue` and be
// rejected.
#[allow(clippy::enum_variant_names)]
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum OtelValue {
    StringValue(String),
    IntValue(i64),
    BoolValue(bool),
    DoubleValue(f64),
}

/// A single OTLP span (the minimal shape a collector can ingest).
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OtelSpan {
    trace_id: String,
    span_id: String,
    name: String,
    kind: i32,
    start_time_unix_nano: u64,
    end_time_unix_nano: u64,
    attributes: Vec<OtelAttribute>,
    status: OtelStatus,
}

#[derive(Debug, Clone, Serialize)]
pub struct OtelStatus {
    code: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    message: Option<String>,
}

/// A synthetic, non-LLM sentinel event: its `provider` is a `(…)` marker
/// (`(prompt_render)`, `(feedback)`, `(sovereign_block)`, `(guardrails_denied)`,
/// `(cache)`, …) rather than a real upstream provider. These are not LLM calls,
/// so they must NOT be emitted as GenAI LLM spans — same convention the
/// observability aggregates use (`provider.starts_with('(')`).
fn is_sentinel(event: &UsageEvent) -> bool {
    event.provider.starts_with('(')
}

/// The GenAI operation name for this event. There is no operation marker on
/// `UsageEvent` today and both the chat and embeddings paths emit the same
/// `UsageEvent::success`/`failure` shape, so a real provider event cannot yet be
/// distinguished as `"embeddings"` — we default to `"chat"`. Threading an
/// operation discriminator onto `UsageEvent` is a tracked follow-on; once it
/// lands this returns `"embeddings"` for the embeddings path.
fn operation_name(_event: &UsageEvent) -> &'static str {
    "chat"
}

/// Translate a [`UsageEvent`] into an OTLP span aligned to the OTel GenAI
/// semantic conventions. Pure function — no I/O.
///
/// Real provider events carry the standard `gen_ai.*` attributes (so Datadog /
/// Arize / Langfuse / Honeycomb recognize the span as an LLM span) plus the
/// `routeplane.*` custom attributes. Synthetic non-LLM sentinel events
/// ([`is_sentinel`]) get a plain `routeplane.<sentinel>` span with the
/// `routeplane.*` attributes only and NO `gen_ai.*` — they are not LLM calls and
/// must not be mis-parsed as such.
pub fn event_to_otel_span(event: &UsageEvent) -> OtelSpan {
    let start_nanos = (event.timestamp.timestamp_nanos_opt().unwrap_or(0)) as u64;
    // Duration is approximated from token counts (no latency field on UsageEvent
    // yet). The latency stats engine tracks real latency; a future enhancement
    // threads it onto UsageEvent. For now, the span is point-in-time.
    let end_nanos = start_nanos;
    let trace_id = hex_hash(&format!(
        "{}:{}:{}:{}",
        event.timestamp, event.virtual_key_name, event.provider, event.model
    ));
    let span_id = hex_hash(&format!("{}:{}", trace_id, event.total_tokens));

    let sentinel = is_sentinel(event);

    // Span name: GenAI rule is "{operation} {model}" for LLM spans; sentinels get
    // a `routeplane.<marker>` name (strip the `(…)` so the span name stays clean).
    let name = if sentinel {
        let marker = event.provider.trim_start_matches('(').trim_end_matches(')');
        format!("routeplane.{marker}")
    } else {
        format!("{} {}", operation_name(event), event.model)
    };

    let mut attributes = Vec::new();

    if !sentinel {
        let op = operation_name(event);
        attributes.push(attr_str("gen_ai.operation.name", op));
        // `gen_ai.system` (older) + `gen_ai.provider.name` (newer semconv) carry
        // the same provider value — emitting both is common for collector
        // back-compat.
        attributes.push(attr_str("gen_ai.system", &event.provider));
        attributes.push(attr_str("gen_ai.provider.name", &event.provider));
        attributes.push(attr_str("gen_ai.request.model", &event.model));
        // Stabilized token names: input/output (NOT the deprecated
        // prompt/completion). total_tokens is a routeplane extra, kept for
        // convenience.
        attributes.push(attr_int(
            "gen_ai.usage.input_tokens",
            event.prompt_tokens as i64,
        ));
        attributes.push(attr_int(
            "gen_ai.usage.output_tokens",
            event.completion_tokens as i64,
        ));
        attributes.push(attr_int(
            "gen_ai.usage.total_tokens",
            event.total_tokens as i64,
        ));
        // Prompt-cache READ tokens, when the provider reported them. Mapped to the
        // GenAI `gen_ai.usage.cached_input_tokens` attribute (the cached SUBSET of
        // input tokens) so cache-aware tools surface it; only present when the
        // event carries it.
        if let Some(cached) = event.cached_tokens {
            attributes.push(attr_int("gen_ai.usage.cached_input_tokens", cached as i64));
        }
    }

    // routeplane.* custom attributes — emitted for every event (LLM + sentinel).
    attributes.push(attr_str("routeplane.tenant", &event.virtual_key_name));
    attributes.push(attr_bool(
        "routeplane.sovereign_routed",
        event.sovereign_routed,
    ));
    if let Some(ref region) = event.region {
        attributes.push(attr_str("routeplane.region", region));
    }
    if let Some(ref status) = event.cache_status {
        attributes.push(attr_str("routeplane.cache_status", status));
    }
    if let Some(ref err) = event.error {
        attributes.push(attr_str("routeplane.error", err));
    }

    let (code, message) = if event.success {
        (STATUS_OK, None)
    } else {
        (
            STATUS_ERROR,
            event
                .error
                .clone()
                .or_else(|| Some("request failed".into())),
        )
    };

    OtelSpan {
        trace_id,
        span_id,
        name,
        kind: SPAN_KIND_CLIENT,
        start_time_unix_nano: start_nanos,
        end_time_unix_nano: end_nanos,
        attributes,
        status: OtelStatus { code, message },
    }
}

/// Render a span as OTLP JSON (the shape an OTLP HTTP/JSON collector expects).
pub fn render_otlp_json(span: &OtelSpan) -> String {
    // The full OTLP envelope: ResourceSpans → ScopeSpans → Spans.
    let envelope = serde_json::json!({
        "resourceSpans": [{
            "resource": {
                "attributes": [
                    {"key": "service.name", "value": {"stringValue": "routeplane"}},
                    {"key": "service.version", "value": {"stringValue": env!("CARGO_PKG_VERSION")}}
                ]
            },
            "scopeSpans": [{
                "scope": {
                    "name": "routeplane",
                    "version": env!("CARGO_PKG_VERSION")
                },
                "spans": [span]
            }]
        }]
    });
    serde_json::to_string(&envelope).unwrap_or_else(|e| format!("{{\"error\":\"{e}\"}}"))
}

/// Lazily-built shared HTTP client for OTLP export. A bounded timeout means a
/// slow/unreachable collector never piles up background tasks or holds sockets —
/// export is best-effort telemetry, never a reliability dependency.
fn otlp_client() -> &'static reqwest::Client {
    static CLIENT: std::sync::OnceLock<reqwest::Client> = std::sync::OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new())
    })
}

/// Resolve the OTLP/HTTP traces endpoint per the OpenTelemetry env-var spec:
/// `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` is used verbatim; otherwise
/// `OTEL_EXPORTER_OTLP_ENDPOINT` is treated as a base and `/v1/traces` is
/// appended (a trailing slash on the base is tolerated). `None` if neither set.
fn traces_endpoint() -> Option<String> {
    if let Ok(full) = std::env::var("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") {
        let full = full.trim();
        if !full.is_empty() {
            return Some(full.to_string());
        }
    }
    let base = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT").ok()?;
    let base = base.trim().trim_end_matches('/');
    if base.is_empty() {
        return None;
    }
    Some(format!("{base}/v1/traces"))
}

/// POST one OTLP/HTTP JSON payload to the collector. Best-effort: surfaces the
/// transport/status error to the caller to log; never retried inline.
async fn post_otlp(endpoint: &str, body: String) -> Result<(), reqwest::Error> {
    otlp_client()
        .post(endpoint)
        .header("content-type", "application/json")
        .body(body)
        .send()
        .await?
        .error_for_status()?;
    Ok(())
}

/// Export a usage event. Gated on `OTEL_EXPORT_ENABLED=true` (default: off).
/// With an OTLP endpoint configured AND a Tokio runtime present, the rendered
/// OTLP/HTTP JSON span is POSTed on a **detached background task** — off the
/// request path. Otherwise (no endpoint / no runtime) it falls back to a debug
/// log. Disabled ⇒ immediate no-op (byte-identical to pre-export behavior, so
/// the A/B parity guard stays green when the env gate is unset).
pub fn export_event(event: &UsageEvent) {
    if !export_enabled() {
        return;
    }
    let span = event_to_otel_span(event);
    let json = render_otlp_json(&span);

    match (traces_endpoint(), tokio::runtime::Handle::try_current()) {
        (Some(endpoint), Ok(handle)) => {
            // Fire-and-forget: a failed/slow collector logs a warning and is
            // dropped — it never blocks, delays, or fails the request.
            handle.spawn(async move {
                if let Err(e) = post_otlp(&endpoint, json).await {
                    tracing::warn!(target: "routeplane::otel", "OTLP export failed: {e}");
                }
            });
        }
        _ => {
            tracing::debug!(target: "routeplane::otel", "{}", json);
        }
    }
}

/// The single OTLP export gate (`OTEL_EXPORT_ENABLED=true|1`, default off). Both
/// the per-event span export and the periodic metrics export honour it, so the
/// disabled default is byte-identical (the A/B parity guard stays green).
fn export_enabled() -> bool {
    std::env::var("OTEL_EXPORT_ENABLED")
        .map(|v| v.eq_ignore_ascii_case("true") || v == "1")
        .unwrap_or(false)
}

/// Resolve the OTLP/HTTP **metrics** endpoint per the OTel env-var spec:
/// `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT` verbatim, else
/// `OTEL_EXPORTER_OTLP_ENDPOINT` as a base with `/v1/metrics` appended.
fn metrics_endpoint() -> Option<String> {
    if let Ok(full) = std::env::var("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT") {
        let full = full.trim();
        if !full.is_empty() {
            return Some(full.to_string());
        }
    }
    let base = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT").ok()?;
    let base = base.trim().trim_end_matches('/');
    if base.is_empty() {
        return None;
    }
    Some(format!("{base}/v1/metrics"))
}

fn unix_nanos() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

fn str_attr(key: &str, value: &str) -> serde_json::Value {
    serde_json::json!({ "key": key, "value": { "stringValue": value } })
}

/// Serialize a metric [`crate::metrics::MetricsSnapshot`] as one OTLP/HTTP JSON
/// payload (`ResourceMetrics` → `ScopeMetrics` → `Metrics`). All series are
/// `AGGREGATION_TEMPORALITY_CUMULATIVE` (=2) — the gateway's atomics are
/// monotonic since process start — under the versioned `routeplane.*` namespace
/// (PRD-009 FR-16 stable contract). Provider-dimensioned only (no per-model/key
/// label — the bounded-cardinality decision the metric table makes).
pub fn render_otlp_metrics_json(
    snap: &crate::metrics::MetricsSnapshot,
    start_nanos: u128,
    now_nanos: u128,
) -> String {
    let start_s = start_nanos.to_string();
    let now_s = now_nanos.to_string();
    let start: &str = &start_s;
    let now: &str = &now_s;
    const CUMULATIVE: i32 = 2;

    let request_points: Vec<serde_json::Value> = snap
        .requests
        .iter()
        .map(|(provider, outcome, count)| {
            serde_json::json!({
                "attributes": [str_attr("provider", provider), str_attr("outcome", outcome)],
                "startTimeUnixNano": start,
                "timeUnixNano": now,
                "asInt": count.to_string(),
            })
        })
        .collect();

    let duration_points: Vec<serde_json::Value> = snap
        .durations
        .iter()
        .map(|h| {
            serde_json::json!({
                "attributes": [str_attr("provider", h.provider)],
                "startTimeUnixNano": start,
                "timeUnixNano": now,
                "count": h.count.to_string(),
                "sum": h.sum_ms as f64,
                "bucketCounts": h.bucket_counts.iter().map(|c| c.to_string()).collect::<Vec<_>>(),
                "explicitBounds": h.bounds.iter().map(|&b| b as f64).collect::<Vec<_>>(),
            })
        })
        .collect();

    let token_point = |kind: &str, v: u64| {
        serde_json::json!({
            "attributes": [str_attr("kind", kind)],
            "startTimeUnixNano": start,
            "timeUnixNano": now,
            "asInt": v.to_string(),
        })
    };
    let sum = |monotonic: bool, points: serde_json::Value| {
        serde_json::json!({
            "aggregationTemporality": CUMULATIVE,
            "isMonotonic": monotonic,
            "dataPoints": points,
        })
    };

    serde_json::json!({
        "resourceMetrics": [{
            "resource": { "attributes": [str_attr("service.name", "routeplane")] },
            "scopeMetrics": [{
                "scope": { "name": "routeplane", "version": env!("CARGO_PKG_VERSION") },
                "metrics": [
                    { "name": "routeplane.requests", "unit": "1",
                      "sum": sum(true, serde_json::Value::Array(request_points)) },
                    { "name": "routeplane.request.duration", "unit": "ms",
                      "histogram": { "aggregationTemporality": CUMULATIVE, "dataPoints": duration_points } },
                    { "name": "routeplane.tokens", "unit": "1",
                      "sum": sum(true, serde_json::json!([
                          token_point("prompt", snap.prompt_tokens),
                          token_point("completion", snap.completion_tokens),
                          token_point("cached", snap.cached_tokens),
                      ])) },
                    { "name": "routeplane.cost.usd", "unit": "USD",
                      "sum": sum(true, serde_json::json!([{
                          "startTimeUnixNano": start, "timeUnixNano": now,
                          "asDouble": snap.cost_micro_usd as f64 / 1_000_000.0,
                      }])) },
                    { "name": "routeplane.requests.shed", "unit": "1",
                      "sum": sum(true, serde_json::json!([{
                          "startTimeUnixNano": start, "timeUnixNano": now,
                          "asInt": snap.shed_total.to_string(),
                      }])) },
                ]
            }]
        }]
    })
    .to_string()
}

/// Spawn the periodic OTLP **metrics** exporter. No-op unless `OTEL_EXPORT_ENABLED`
/// is set AND a metrics endpoint is configured — so the default is byte-identical.
/// Off the hot path entirely: a tokio interval task snapshots the process metric
/// table, serializes OTLP/HTTP JSON, and POSTs it. A failed/slow collector logs
/// and is dropped; the request path never touches this. Call once at startup
/// inside the Tokio runtime; `shed_total` supplies the binary-level shed counter.
pub fn spawn_metrics_exporter(shed_total: impl Fn() -> u64 + Send + 'static) {
    if !export_enabled() {
        return;
    }
    let Some(endpoint) = metrics_endpoint() else {
        return;
    };
    let interval_ms = std::env::var("OTEL_METRIC_EXPORT_INTERVAL_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(30_000);
    let start_nanos = unix_nanos();
    tracing::info!(
        target: "routeplane::otel",
        "OTLP metrics export ENABLED (endpoint={endpoint}, interval={interval_ms}ms)"
    );
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(std::time::Duration::from_millis(interval_ms));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tick.tick().await;
            let snap = crate::metrics::metrics().snapshot(shed_total());
            let json = render_otlp_metrics_json(&snap, start_nanos, unix_nanos());
            if let Err(e) = post_otlp(&endpoint, json).await {
                tracing::warn!(target: "routeplane::otel", "OTLP metrics export failed: {e}");
            }
        }
    });
}

fn attr_str(key: &str, value: &str) -> OtelAttribute {
    OtelAttribute {
        key: key.into(),
        value: OtelValue::StringValue(value.into()),
    }
}

fn attr_int(key: &str, value: i64) -> OtelAttribute {
    OtelAttribute {
        key: key.into(),
        value: OtelValue::IntValue(value),
    }
}

fn attr_bool(key: &str, value: bool) -> OtelAttribute {
    OtelAttribute {
        key: key.into(),
        value: OtelValue::BoolValue(value),
    }
}

/// Simple hex digest for trace/span IDs (SHA-256, truncated to 32 hex chars
/// for a 128-bit trace ID — the OTLP minimum). Not cryptographically load-
/// bearing; the IDs just need to be unique-enough for collector dedup.
fn hex_hash(input: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    let hash = hasher.finalize();
    // First 16 bytes = 128 bits = 32 hex chars (OTLP trace ID length).
    hash[..16].iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn sample_event() -> UsageEvent {
        UsageEvent {
            timestamp: Utc::now(),
            virtual_key_name: "rp_test_key".into(),
            provider: "openai".into(),
            model: "gpt-4o".into(),
            prompt_tokens: 10,
            completion_tokens: 20,
            total_tokens: 30,
            cached_tokens: None,
            use_case: None,
            region: Some("US".into()),
            sovereign_routed: false,
            success: true,
            error: None,
            guardrails: None,
            cache_hit: None,
            cache_status: None,
            cache_namespace: None,
            estimated_saved_cost_micro_usd: None,
            prompt_id: None,
            prompt_version: None,
            prompt_label: None,
            prompt_experiment: None,
            prompt_variant: None,
            config_ref: None,
            config_match: None,
            cost: None,
            latency_ms: None,
            hedged: false,
            output_masked: None,
            feedback_trace_id: None,
            feedback_value: None,
            feedback_weight: None,
            feedback_metadata_keys: None,
        }
    }

    /// Find the string value of an attribute by key, if present.
    fn attr_value_str<'a>(span: &'a OtelSpan, key: &str) -> Option<&'a str> {
        span.attributes.iter().find(|a| a.key == key).and_then(|a| {
            if let OtelValue::StringValue(s) = &a.value {
                Some(s.as_str())
            } else {
                None
            }
        })
    }

    fn attr_value_int(span: &OtelSpan, key: &str) -> Option<i64> {
        span.attributes.iter().find(|a| a.key == key).and_then(|a| {
            if let OtelValue::IntValue(v) = &a.value {
                Some(*v)
            } else {
                None
            }
        })
    }

    #[test]
    fn event_to_span_has_gen_ai_semconv_attributes() {
        let event = sample_event();
        let span = event_to_otel_span(&event);
        // GenAI span-naming rule: "{operation} {model}".
        assert_eq!(span.name, "chat gpt-4o");
        assert_eq!(span.kind, SPAN_KIND_CLIENT);
        let keys: Vec<&str> = span.attributes.iter().map(|a| a.key.as_str()).collect();
        // Standard GenAI attributes (the stabilized names tools auto-parse).
        assert!(keys.contains(&"gen_ai.operation.name"));
        assert!(keys.contains(&"gen_ai.system"));
        assert!(keys.contains(&"gen_ai.provider.name"));
        assert!(keys.contains(&"gen_ai.request.model"));
        assert!(keys.contains(&"gen_ai.usage.input_tokens"));
        assert!(keys.contains(&"gen_ai.usage.output_tokens"));
        assert!(keys.contains(&"gen_ai.usage.total_tokens"));
        assert!(keys.contains(&"routeplane.sovereign_routed"));

        // The DEPRECATED token names must be gone.
        assert!(!keys.contains(&"gen_ai.usage.prompt_tokens"));
        assert!(!keys.contains(&"gen_ai.usage.completion_tokens"));

        // Values: operation defaults to "chat"; provider mirrored across both
        // semconv attrs; token counts map prompt→input, completion→output.
        assert_eq!(attr_value_str(&span, "gen_ai.operation.name"), Some("chat"));
        assert_eq!(attr_value_str(&span, "gen_ai.system"), Some("openai"));
        assert_eq!(
            attr_value_str(&span, "gen_ai.provider.name"),
            Some("openai")
        );
        assert_eq!(attr_value_int(&span, "gen_ai.usage.input_tokens"), Some(10));
        assert_eq!(
            attr_value_int(&span, "gen_ai.usage.output_tokens"),
            Some(20)
        );
        assert_eq!(attr_value_int(&span, "gen_ai.usage.total_tokens"), Some(30));
    }

    #[test]
    fn cached_tokens_map_to_gen_ai_cached_input_tokens_when_present() {
        let mut event = sample_event();
        event.cached_tokens = Some(7);
        let span = event_to_otel_span(&event);
        assert_eq!(
            attr_value_int(&span, "gen_ai.usage.cached_input_tokens"),
            Some(7)
        );

        // Absent ⇒ attribute omitted (don't fabricate a zero).
        let bare = event_to_otel_span(&sample_event());
        assert!(attr_value_int(&bare, "gen_ai.usage.cached_input_tokens").is_none());
    }

    #[test]
    fn sentinel_events_are_not_emitted_as_gen_ai_llm_spans() {
        // A synthetic sentinel (provider starts with `(`) must NOT carry any
        // gen_ai.* attribute — tools must not mis-parse it as an LLM span — and
        // gets a routeplane.<marker> span name rather than "{op} {model}".
        let mut event = sample_event();
        event.provider = "(sovereign_block)".into();
        event.success = false;
        event.error = Some("sovereign_block".into());
        let span = event_to_otel_span(&event);

        assert_eq!(span.name, "routeplane.sovereign_block");
        let has_gen_ai = span.attributes.iter().any(|a| a.key.starts_with("gen_ai."));
        assert!(!has_gen_ai, "sentinel spans must carry no gen_ai.* attrs");
        // The routeplane.* custom attributes are still present.
        assert!(attr_value_str(&span, "routeplane.tenant").is_some());
    }

    #[test]
    fn no_finish_reasons_attribute_when_event_lacks_one() {
        // UsageEvent carries no finish reason today; we must not fabricate
        // gen_ai.response.finish_reasons (tracked follow-on).
        let span = event_to_otel_span(&sample_event());
        let keys: Vec<&str> = span.attributes.iter().map(|a| a.key.as_str()).collect();
        assert!(!keys.contains(&"gen_ai.response.finish_reasons"));
    }

    #[test]
    fn successful_event_has_ok_status() {
        let event = sample_event();
        let span = event_to_otel_span(&event);
        assert_eq!(span.status.code, STATUS_OK);
        assert!(span.status.message.is_none());
    }

    #[test]
    fn failed_event_has_error_status_with_message() {
        let mut event = sample_event();
        event.success = false;
        event.error = Some("upstream_timeout".into());
        let span = event_to_otel_span(&event);
        assert_eq!(span.status.code, STATUS_ERROR);
        assert_eq!(span.status.message.as_deref(), Some("upstream_timeout"));
    }

    #[test]
    fn otlp_json_envelope_has_resource_spans() {
        let event = sample_event();
        let span = event_to_otel_span(&event);
        let json = render_otlp_json(&span);
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed.get("resourceSpans").is_some());
        let rs = &parsed["resourceSpans"][0];
        assert!(rs.get("resource").is_some());
        assert!(rs.get("scopeSpans").is_some());
        let spans = &rs["scopeSpans"][0]["spans"];
        assert!(spans.as_array().unwrap().len() == 1);
    }

    #[test]
    fn trace_id_is_32_hex_chars() {
        let event = sample_event();
        let span = event_to_otel_span(&event);
        assert_eq!(span.trace_id.len(), 32);
        assert!(span.trace_id.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn sovereign_region_is_an_attribute_when_present() {
        let event = sample_event();
        let span = event_to_otel_span(&event);
        let region_attr = span
            .attributes
            .iter()
            .find(|a| a.key == "routeplane.region");
        assert!(region_attr.is_some());
    }

    #[test]
    fn absent_region_is_not_an_attribute() {
        let mut event = sample_event();
        event.region = None;
        let span = event_to_otel_span(&event);
        let region_attr = span
            .attributes
            .iter()
            .find(|a| a.key == "routeplane.region");
        assert!(region_attr.is_none());
    }

    #[test]
    fn hex_hash_deterministic() {
        let a = hex_hash("hello");
        let b = hex_hash("hello");
        assert_eq!(a, b);
        assert_eq!(a.len(), 32);
    }

    #[test]
    fn traces_endpoint_resolves_per_otel_spec() {
        // TRACES_ENDPOINT (verbatim) wins over the base; base gets /v1/traces
        // appended with the trailing slash tolerated; neither set ⇒ None.
        // Env is process-global; this is the only test that mutates these vars.
        std::env::remove_var("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT");
        std::env::remove_var("OTEL_EXPORTER_OTLP_ENDPOINT");
        assert_eq!(traces_endpoint(), None);

        std::env::set_var("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318/");
        assert_eq!(
            traces_endpoint().as_deref(),
            Some("http://collector:4318/v1/traces")
        );

        std::env::set_var(
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
            "http://collector:4318/custom/traces",
        );
        assert_eq!(
            traces_endpoint().as_deref(),
            Some("http://collector:4318/custom/traces")
        );

        std::env::remove_var("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT");
        std::env::remove_var("OTEL_EXPORTER_OTLP_ENDPOINT");
    }

    #[test]
    fn span_attribute_values_use_otlp_camelcase_keys() {
        // OTLP/HTTP JSON requires AnyValue keys in camelCase (stringValue,
        // intValue, …); `lowercase` would be rejected by collectors.
        let json = render_otlp_json(&event_to_otel_span(&sample_event()));
        let attrs = serde_json::from_str::<serde_json::Value>(&json).unwrap()["resourceSpans"][0]
            ["scopeSpans"][0]["spans"][0]["attributes"]
            .clone();
        let s = attrs.to_string();
        assert!(
            s.contains("\"stringValue\""),
            "expected camelCase stringValue"
        );
        assert!(!s.contains("\"stringvalue\""), "lowercase is non-compliant");
        assert!(s.contains("\"intValue\""), "expected camelCase intValue");
    }

    #[tokio::test]
    async fn post_otlp_sends_json_span_to_collector() {
        use wiremock::matchers::{header, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/traces"))
            .and(header("content-type", "application/json"))
            .respond_with(ResponseTemplate::new(200))
            .expect(1)
            .mount(&server)
            .await;

        let endpoint = format!("{}/v1/traces", server.uri());
        let body = render_otlp_json(&event_to_otel_span(&sample_event()));
        post_otlp(&endpoint, body)
            .await
            .expect("OTLP POST succeeds");
        // MockServer drop asserts the `.expect(1)` — the span actually went out.
    }

    #[tokio::test]
    async fn post_otlp_surfaces_collector_error_status() {
        use wiremock::matchers::method;
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(503))
            .mount(&server)
            .await;

        let body = render_otlp_json(&event_to_otel_span(&sample_event()));
        let err = post_otlp(&format!("{}/v1/traces", server.uri()), body).await;
        assert!(err.is_err(), "a 5xx from the collector must surface as Err");
    }

    // --- FR-16 OTLP metrics export (routeplane#216) ---

    #[test]
    fn render_otlp_metrics_json_emits_routeplane_metrics() {
        use crate::metrics::{HistogramSnapshot, MetricsSnapshot};
        let snap = MetricsSnapshot {
            requests: vec![("openai", "success", 5), ("anthropic", "error", 2)],
            durations: vec![HistogramSnapshot {
                provider: "openai",
                bucket_counts: vec![1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                bounds: &[50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000, 60000],
                sum_ms: 800,
                count: 6,
            }],
            prompt_tokens: 100,
            completion_tokens: 50,
            cached_tokens: 10,
            cost_micro_usd: 1_234_000,
            shed_total: 3,
        };
        let v: serde_json::Value =
            serde_json::from_str(&render_otlp_metrics_json(&snap, 1_000, 2_000)).unwrap();
        let metrics = &v["resourceMetrics"][0]["scopeMetrics"][0]["metrics"];
        let names: Vec<&str> = metrics
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["name"].as_str().unwrap())
            .collect();
        for want in [
            "routeplane.requests",
            "routeplane.request.duration",
            "routeplane.tokens",
            "routeplane.cost.usd",
            "routeplane.requests.shed",
        ] {
            assert!(names.contains(&want), "missing metric {want}");
        }
        // requests: cumulative monotonic Sum, one point per (provider, outcome).
        let reqs = &metrics[0]["sum"];
        assert_eq!(reqs["aggregationTemporality"], 2);
        assert_eq!(reqs["isMonotonic"], true);
        assert_eq!(reqs["dataPoints"].as_array().unwrap().len(), 2);
        assert_eq!(reqs["dataPoints"][0]["asInt"], "5");
        assert_eq!(
            reqs["dataPoints"][0]["attributes"][0]["value"]["stringValue"],
            "openai"
        );
        // histogram: N+1 per-bucket counts for N explicit bounds.
        let h = &metrics[1]["histogram"]["dataPoints"][0];
        assert_eq!(h["explicitBounds"].as_array().unwrap().len(), 10);
        assert_eq!(h["bucketCounts"].as_array().unwrap().len(), 11);
        assert_eq!(h["count"], "6");
        // micro-usd → USD double.
        let cost = metrics[3]["sum"]["dataPoints"][0]["asDouble"]
            .as_f64()
            .unwrap();
        assert!((cost - 1.234).abs() < 1e-9, "cost was {cost}");
    }

    #[tokio::test]
    async fn metrics_export_posts_otlp_json_to_v1_metrics() {
        use crate::metrics::MetricsSnapshot;
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/metrics"))
            .respond_with(ResponseTemplate::new(200))
            .expect(1)
            .mount(&server)
            .await;
        let snap = MetricsSnapshot {
            requests: vec![("openai", "success", 1)],
            durations: vec![],
            prompt_tokens: 1,
            completion_tokens: 1,
            cached_tokens: 0,
            cost_micro_usd: 1,
            shed_total: 0,
        };
        let body = render_otlp_metrics_json(&snap, 1, 2);
        post_otlp(&format!("{}/v1/metrics", server.uri()), body)
            .await
            .expect("metrics POST to /v1/metrics should succeed");
    }
}
