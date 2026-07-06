//! `POST /v1/messages` — the NATIVE Anthropic Messages API surface (PARITY:
//! Portkey and LiteLLM both expose Anthropic's `/v1/messages` shape so a client
//! using the official Anthropic SDK can point `base_url` at the gateway
//! unchanged). Routeplane is otherwise OpenAI-shape-only (`/v1/chat/completions`),
//! so Anthropic-SDK users could not use it.
//!
//! This is a TRANSLATION surface in front of the EXISTING completion pipeline —
//! it does NOT bypass any gateway control. The handler:
//!   1. parses the Anthropic-native request body,
//!   2. translates it INBOUND to the canonical `ChatCompletionRequest`,
//!   3. funnels it through the SAME `proxy::chat_completions_core` the OpenAI
//!      handler uses (auth context, residency classify-then-mask, routing,
//!      eligibility, limits admission, guardrails before/after, provider call,
//!      usage/ledger/export, cache — every control applies identically), then
//!   4. translates the canonical OpenAI-shaped `Response` back OUTBOUND into the
//!      Anthropic Messages response shape.
//!
//! ## Default provider
//! `/v1/messages` defaults the provider chain to `anthropic` (the client is
//! speaking the Anthropic shape). It does so by injecting a synthetic
//! `x-routeplane-provider: anthropic` header ONLY when the client did not set one
//! — so the core's existing header-driven provider resolution is reused verbatim.
//! ANY provider still works (translation is to the provider-neutral canonical
//! shape); `x-routeplane-provider` overrides as on every other route.
//!
//! ## Masking / residency
//! Because the translated canonical request runs through `chat_completions_core`,
//! the classify-BEFORE-mask residency step, PII masking, guardrails, and limits
//! all run on the translated content exactly as they do for `/v1/chat/completions`
//! — `/v1/messages` is not a PII-egress bypass.
//!
//! ## Streaming
//! Anthropic streaming uses its OWN SSE event sequence (`message_start`,
//! `content_block_*`, `message_delta`, `message_stop`), which differs from the
//! OpenAI `chat.completion.chunk` sequence the core emits. Re-emitting that
//! native sequence requires threading an Anthropic-mode flag deep into the shared
//! streaming hot path (`proxy::stream_chat_completions`), which risks the
//! byte-identical guarantee on the OpenAI streaming path. To stay safe + correct
//! this iteration, `stream: true` on `/v1/messages` returns a documented 400
//! (Anthropic error shape) directing the caller to `/v1/chat/completions` for
//! streaming. Native Anthropic SSE re-emission is a tracked follow-on.
//!
//! ## Error shape
//! 4xx errors raised BY THIS HANDLER (missing `max_tokens`, `stream:true`) are
//! rendered in Anthropic's `{type:"error", error:{type, message}}` shape for SDK
//! compatibility. Errors raised by the shared CORE (401 auth, 422 residency, 446
//! guardrail, 429/402 limits, 5xx upstream) keep the Routeplane/OpenAI envelope —
//! they are produced before/within the pipeline and are documented as such.
//!
//! ADR note: a translation surface over the EXISTING pipeline, reusing
//! serde + the adapter block shapes, with NO new standing cost, NO DB, and NO new
//! dependency. No architectural shift — like rerank/moderations/embeddings, no
//! ADR is written.

use crate::auth::{TenantContext, TenantGuardrails, VirtualKey};
use crate::proxy::{chat_completions_core, AppState};
use axum::{
    body::to_bytes,
    extract::State,
    http::{HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Extension,
};
use routeplane_types::{
    ChatCompletionRequest, ChatCompletionResponse, ContentPart, ImageUrlContent, Message,
    MessageContent,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::Arc;

/// Max bytes we will buffer when reading the core's response body back for
/// outbound translation. The core's chat JSON is small (a single completion);
/// this bound mirrors the body caps elsewhere and prevents an unbounded read.
const MAX_CORE_BODY_BYTES: usize = 4 * 1024 * 1024;

// --- Inbound: Anthropic Messages request -------------------------------------

/// The native Anthropic `/v1/messages` request. Unknown/extra fields are
/// tolerated (serde ignores them by default) so a forward-compatible Anthropic
/// field never 400s here. `max_tokens` is REQUIRED by Anthropic; it is modelled
/// `Option` so we can return a clean Anthropic-shaped 400 (rather than serde's
/// generic envelope) when it is missing.
#[derive(Debug, Deserialize)]
pub struct AnthropicMessagesRequest {
    model: String,
    /// REQUIRED by Anthropic — validated in the handler so the 400 is
    /// Anthropic-shaped (`invalid_request_error`), not a serde parse error.
    #[serde(default)]
    max_tokens: Option<u32>,
    /// Top-level system prompt: a bare string OR an array of blocks.
    #[serde(default)]
    system: Option<Value>,
    #[serde(default)]
    messages: Vec<AnthropicInboundMessage>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    stop_sequences: Option<Vec<String>>,
    #[serde(default)]
    stream: Option<bool>,
}

/// One inbound Anthropic message: `role` (`user`/`assistant`) plus `content` that
/// is EITHER a bare string OR an array of typed content blocks.
#[derive(Debug, Deserialize)]
struct AnthropicInboundMessage {
    role: String,
    content: AnthropicInboundContent,
}

/// Anthropic message content: a bare string or an ordered array of blocks.
/// `#[serde(untagged)]` — a JSON string deserializes to `Text`, an array to
/// `Blocks` (each block kept as a raw `Value` so we can map text/image and
/// tolerate any other block type without rejecting the request).
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum AnthropicInboundContent {
    Text(String),
    Blocks(Vec<Value>),
}

/// Translate the native Anthropic request into the canonical
/// `ChatCompletionRequest`. The `system` prompt becomes a leading system-role
/// `Message` (the canonical shape the Anthropic adapter then lifts back to the
/// native top-level `system` field on egress); each Anthropic message becomes a
/// canonical `Message` with text → `Text` and image blocks → `ContentPart::ImageUrl`.
fn to_canonical_request(req: AnthropicMessagesRequest) -> ChatCompletionRequest {
    let mut messages: Vec<Message> = Vec::with_capacity(req.messages.len() + 1);

    // A `system` prompt (string or block array) maps to a leading system-role
    // message. The downstream Anthropic adapter re-lifts a system-role message to
    // the native top-level `system` field, so the round-trip is faithful.
    if let Some(system) = &req.system {
        if let Some(text) = flatten_system(system) {
            if !text.is_empty() {
                messages.push(Message {
                    role: "system".to_string(),
                    content: MessageContent::Text(text),
                    name: None,
                    cache_control: None,
                    tool_calls: None,
                    tool_call_id: None,
                    refusal: None,
                    reasoning_content: None,
                });
            }
        }
    }

    for m in req.messages {
        messages.push(Message {
            role: m.role,
            content: to_canonical_content(m.content),
            name: None,
            cache_control: None,
            tool_calls: None,
            tool_call_id: None,
            refusal: None,
            reasoning_content: None,
        });
    }

    ChatCompletionRequest {
        model: req.model,
        messages,
        temperature: req.temperature,
        top_p: req.top_p,
        // Anthropic streaming is not yet translatable on this surface (see the
        // module docs); the handler rejects `stream:true` BEFORE this point, so
        // the canonical request is always non-streaming here.
        stream: None,
        max_tokens: req.max_tokens,
        stop: req.stop_sequences,
        n: None,
        presence_penalty: None,
        frequency_penalty: None,
        user: None,
        tools: None,
        tool_choice: None,
        parallel_tool_calls: None,
        ..Default::default()
    }
}

/// Flatten Anthropic's `system` (a bare string OR an array of text blocks) into a
/// single string. Block arrays are joined with `\n\n` (the same join the
/// Anthropic adapter uses when it re-emits a multi-part system). Non-text blocks
/// are ignored. Returns `None` for an unrepresentable shape.
fn flatten_system(system: &Value) -> Option<String> {
    match system {
        Value::String(s) => Some(s.clone()),
        Value::Array(blocks) => {
            let parts: Vec<String> = blocks
                .iter()
                .filter_map(|b| {
                    // `{ "type": "text", "text": "..." }`
                    b.get("text").and_then(Value::as_str).map(str::to_string)
                })
                .collect();
            Some(parts.join("\n\n"))
        }
        _ => None,
    }
}

/// Translate one Anthropic message content into canonical [`MessageContent`].
/// A bare string → `Text`. A block array → `Parts` (text blocks → text parts,
/// image blocks → `ContentPart::ImageUrl` reconstructing the canonical URL from
/// the Anthropic `source`). A block array that holds ONLY text collapses to the
/// bare-string `Text` form so a text-only message round-trips byte-identically.
fn to_canonical_content(content: AnthropicInboundContent) -> MessageContent {
    match content {
        AnthropicInboundContent::Text(s) => MessageContent::Text(s),
        AnthropicInboundContent::Blocks(blocks) => {
            let mut parts: Vec<ContentPart> = Vec::with_capacity(blocks.len());
            let mut has_image = false;
            for block in &blocks {
                match block.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        if let Some(text) = block.get("text").and_then(Value::as_str) {
                            parts.push(ContentPart::Text {
                                text: text.to_string(),
                                cache_control: None,
                            });
                        }
                    }
                    Some("image") => {
                        if let Some(url) = anthropic_image_to_url(block.get("source")) {
                            has_image = true;
                            parts.push(ContentPart::ImageUrl {
                                image_url: ImageUrlContent { url, detail: None },
                            });
                        }
                        // An unrepresentable image source is skipped gracefully —
                        // never a panic, never a malformed canonical request.
                    }
                    // Any other block type (tool_use, tool_result, …) is ignored
                    // on inbound for this pass; its text is not part of the prompt
                    // we forward. Tolerated, not rejected.
                    _ => {}
                }
            }
            // Text-only block array → collapse to bare string (byte-identical to a
            // plain text message on the canonical wire).
            if !has_image {
                let joined: String = parts
                    .iter()
                    .filter_map(|p| match p {
                        ContentPart::Text { text, .. } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("");
                return MessageContent::Text(joined);
            }
            MessageContent::Parts(parts)
        }
    }
}

/// Reconstruct a canonical image URL from an Anthropic `image.source`:
///   * `{type:"base64", media_type, data}` → `data:<media_type>;base64,<data>`
///   * `{type:"url", url}`                 → the URL verbatim
///
/// Returns `None` for any other / malformed source (skipped by the caller).
fn anthropic_image_to_url(source: Option<&Value>) -> Option<String> {
    let source = source?;
    match source.get("type").and_then(Value::as_str) {
        Some("base64") => {
            let media_type = source.get("media_type").and_then(Value::as_str)?;
            let data = source.get("data").and_then(Value::as_str)?;
            Some(format!("data:{media_type};base64,{data}"))
        }
        Some("url") => source
            .get("url")
            .and_then(Value::as_str)
            .map(str::to_string),
        _ => None,
    }
}

// --- Outbound: canonical response → Anthropic Messages response --------------

/// Map an OpenAI-family `finish_reason` to an Anthropic `stop_reason`.
fn map_finish_reason(finish: &str) -> &'static str {
    match finish {
        "stop" => "end_turn",
        "length" => "max_tokens",
        "tool_calls" => "tool_use",
        "content_filter" => "end_turn",
        // Anthropic-native values may already have flowed through unmapped (e.g.
        // an Anthropic provider whose stop_reason the adapter passed verbatim).
        "end_turn" | "max_tokens" | "stop_sequence" | "tool_use" => match finish {
            "end_turn" => "end_turn",
            "max_tokens" => "max_tokens",
            "stop_sequence" => "stop_sequence",
            _ => "tool_use",
        },
        _ => "end_turn",
    }
}

/// Translate a canonical [`ChatCompletionResponse`] into the Anthropic Messages
/// response JSON. `id` passes through (it is already `msg_...` when an Anthropic
/// provider served the request; for any other provider we keep the provider id).
/// `choices[0].message.content` becomes a single `text` content block;
/// `usage.prompt_tokens`/`completion_tokens` become `input_tokens`/`output_tokens`;
/// cache READ tokens (when present) surface as Anthropic's
/// `cache_read_input_tokens` (cache WRITE → `cache_creation_input_tokens`).
fn to_anthropic_response(resp: ChatCompletionResponse) -> Value {
    let first = resp.choices.into_iter().next();
    let (text, stop_reason) = match first {
        Some(choice) => {
            let text = choice.message.content.as_text();
            let stop = map_finish_reason(&choice.finish_reason);
            (text, stop)
        }
        None => (String::new(), "end_turn"),
    };

    let mut usage = json!({
        "input_tokens": resp.usage.prompt_tokens,
        "output_tokens": resp.usage.completion_tokens,
    });
    if let (Some(obj), Some(read)) = (usage.as_object_mut(), resp.usage.cached_tokens) {
        obj.insert("cache_read_input_tokens".to_string(), json!(read));
    }
    if let (Some(obj), Some(write)) = (usage.as_object_mut(), resp.usage.cache_creation_tokens) {
        obj.insert("cache_creation_input_tokens".to_string(), json!(write));
    }

    json!({
        "id": resp.id,
        "type": "message",
        "role": "assistant",
        "model": resp.model,
        "content": [ { "type": "text", "text": text } ],
        "stop_reason": stop_reason,
        "stop_sequence": Value::Null,
        "usage": usage,
    })
}

// --- Error shaping (Anthropic shape) -----------------------------------------

/// Render an Anthropic-shaped error `{type:"error", error:{type, message}}`.
fn anthropic_error(status: StatusCode, error_type: &str, message: &str) -> Response {
    let body = json!({
        "type": "error",
        "error": { "type": error_type, "message": message }
    });
    (status, axum::Json(body)).into_response()
}

// --- Handler -----------------------------------------------------------------

pub async fn messages(
    State(state): State<Arc<AppState>>,
    Extension(virtual_key): Extension<VirtualKey>,
    Extension(tenant_ctx): Extension<TenantContext>,
    Extension(tenant_guardrails): Extension<TenantGuardrails>,
    mut headers: HeaderMap,
    crate::api_error::OpenAiJson(req): crate::api_error::OpenAiJson<AnthropicMessagesRequest>,
) -> Response {
    // max_tokens is REQUIRED by Anthropic — reject a missing value with a clean
    // Anthropic-shaped 400 (not a generic serde envelope).
    if req.max_tokens.is_none() {
        return anthropic_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "`max_tokens` is required for /v1/messages.",
        );
    }

    // Streaming is not yet translatable on this surface (Anthropic's native SSE
    // sequence differs from the core's OpenAI chunk stream). Reject with a
    // documented Anthropic-shaped 400 directing the caller to the OpenAI route.
    if req.stream.unwrap_or(false) {
        return anthropic_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "Streaming is not yet supported on /v1/messages. Use POST /v1/chat/completions with \"stream\": true (OpenAI-compatible SSE).",
        );
    }

    // Default the provider chain to `anthropic` for this surface (the client is
    // speaking the Anthropic shape) UNLESS the caller explicitly overrode it —
    // OR the model id is registered on a runtime custom provider. In the custom
    // case we leave the header ABSENT so the core's header-less MODEL routing
    // resolves it (the same precedence as /v1/chat/completions: a custom
    // provider never shadows a built-in catalog id). The probe is one lock-free
    // `ArcSwap::load` + `HashMap` miss when the registry is empty ⇒ the
    // synthetic-anthropic default is byte-identical. The custom path then rides
    // the SAME pipeline: this handler's inbound Anthropic→canonical translation,
    // the core's controls, the OpenAI-compatible `SelfHostedProvider` egress to
    // the provider's /v1/chat/completions, and the outbound canonical→Anthropic
    // translation below — usage records under the custom provider's name.
    let model_routes_to_custom = !crate::models_api::is_builtin_model(&req.model)
        && state
            .custom_providers
            .provider_for_model(&req.model)
            .is_some();
    if !headers.contains_key("x-routeplane-provider") && !model_routes_to_custom {
        headers.insert(
            "x-routeplane-provider",
            HeaderValue::from_static("anthropic"),
        );
    }

    let canonical = to_canonical_request(req);

    // Funnel through the SAME completion core as /v1/chat/completions — every
    // control (classify-then-mask, residency, guardrails, limits, routing,
    // usage/ledger/export, cache) applies identically. The returned Response is
    // OpenAI-shaped; we translate its body back to the Anthropic shape on 2xx.
    let core_resp = chat_completions_core(
        state,
        virtual_key,
        tenant_ctx,
        tenant_guardrails,
        headers,
        canonical,
    )
    .await;

    translate_core_response(core_resp).await
}

/// Translate the core's OpenAI-shaped `Response` into the Anthropic Messages
/// response. On a 2xx JSON body we re-shape it; on any non-2xx (auth 401,
/// residency 422, guardrail 446, limit 429/402, upstream 5xx) we pass the core's
/// Routeplane/OpenAI envelope through UNCHANGED (documented — those are pipeline
/// decisions, not this handler's). Response headers (rate-limit advisories,
/// trace id, cache status) are preserved.
async fn translate_core_response(resp: Response) -> Response {
    let status = resp.status();
    // Non-2xx: pass the core's envelope through unchanged (documented).
    if !status.is_success() {
        return resp;
    }

    let (mut parts, body) = resp.into_parts();
    let bytes = match to_bytes(body, MAX_CORE_BODY_BYTES).await {
        Ok(b) => b,
        Err(_) => {
            return anthropic_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "api_error",
                "The gateway could not read the upstream response.",
            );
        }
    };

    let canonical: ChatCompletionResponse = match serde_json::from_slice(&bytes) {
        Ok(c) => c,
        Err(_) => {
            // The 2xx body was not a chat completion (should not happen on this
            // path). Fail closed with an Anthropic-shaped 500 rather than leak a
            // non-translated body.
            return anthropic_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "api_error",
                "The gateway received an untranslatable upstream response.",
            );
        }
    };

    let anthropic_body = to_anthropic_response(canonical);
    let serialized = match serde_json::to_vec(&anthropic_body) {
        Ok(v) => v,
        Err(_) => {
            return anthropic_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "api_error",
                "The gateway could not encode the response.",
            );
        }
    };

    // Reuse the core's response headers (rate-limit advisories, x-routeplane-*,
    // trace id) but force a JSON content-type for the re-serialized body and drop
    // the now-stale content-length (axum recomputes it from the new body).
    parts.headers.insert(
        axum::http::header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    parts.headers.remove(axum::http::header::CONTENT_LENGTH);

    (parts.status, parts.headers, serialized).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req_value(v: Value) -> AnthropicMessagesRequest {
        serde_json::from_value(v).unwrap()
    }

    #[test]
    fn inbound_maps_system_and_user_text() {
        let r = req_value(json!({
            "model": "claude-3-5-sonnet",
            "max_tokens": 100,
            "system": "be terse",
            "messages": [ { "role": "user", "content": "hello" } ]
        }));
        let c = to_canonical_request(r);
        assert_eq!(c.model, "claude-3-5-sonnet");
        assert_eq!(c.max_tokens, Some(100));
        assert_eq!(c.messages.len(), 2);
        assert_eq!(c.messages[0].role, "system");
        assert_eq!(c.messages[0].content.as_text(), "be terse");
        assert_eq!(c.messages[1].role, "user");
        assert_eq!(c.messages[1].content.as_text(), "hello");
    }

    #[test]
    fn inbound_flattens_system_block_array() {
        let r = req_value(json!({
            "model": "claude",
            "max_tokens": 10,
            "system": [
                { "type": "text", "text": "part one" },
                { "type": "text", "text": "part two" }
            ],
            "messages": []
        }));
        let c = to_canonical_request(r);
        assert_eq!(c.messages[0].role, "system");
        assert_eq!(c.messages[0].content.as_text(), "part one\n\npart two");
    }

    #[test]
    fn inbound_text_block_array_collapses_to_bare_string() {
        let r = req_value(json!({
            "model": "claude",
            "max_tokens": 10,
            "messages": [ { "role": "user", "content": [
                { "type": "text", "text": "a" },
                { "type": "text", "text": "b" }
            ] } ]
        }));
        let c = to_canonical_request(r);
        match &c.messages[0].content {
            MessageContent::Text(s) => assert_eq!(s, "ab"),
            other => panic!("expected bare Text, got {other:?}"),
        }
    }

    #[test]
    fn inbound_base64_image_block_becomes_data_url() {
        let r = req_value(json!({
            "model": "claude",
            "max_tokens": 10,
            "messages": [ { "role": "user", "content": [
                { "type": "text", "text": "what is this" },
                { "type": "image", "source": {
                    "type": "base64", "media_type": "image/png", "data": "AAAA"
                } }
            ] } ]
        }));
        let c = to_canonical_request(r);
        match &c.messages[0].content {
            MessageContent::Parts(parts) => {
                assert_eq!(parts.len(), 2);
                match &parts[1] {
                    ContentPart::ImageUrl { image_url } => {
                        assert_eq!(image_url.url, "data:image/png;base64,AAAA");
                    }
                    other => panic!("expected ImageUrl, got {other:?}"),
                }
            }
            other => panic!("expected Parts, got {other:?}"),
        }
    }

    #[test]
    fn inbound_url_image_block_passes_url_through() {
        let r = req_value(json!({
            "model": "claude",
            "max_tokens": 10,
            "messages": [ { "role": "user", "content": [
                { "type": "image", "source": { "type": "url", "url": "https://x/c.jpg" } }
            ] } ]
        }));
        let c = to_canonical_request(r);
        match &c.messages[0].content {
            MessageContent::Parts(parts) => match &parts[0] {
                ContentPart::ImageUrl { image_url } => {
                    assert_eq!(image_url.url, "https://x/c.jpg");
                }
                other => panic!("expected ImageUrl, got {other:?}"),
            },
            other => panic!("expected Parts, got {other:?}"),
        }
    }

    #[test]
    fn inbound_tolerates_unknown_fields_and_unknown_blocks() {
        // Unknown top-level fields and unknown block types must NOT 400.
        let r = req_value(json!({
            "model": "claude",
            "max_tokens": 10,
            "metadata": { "user_id": "u1" },
            "messages": [ { "role": "user", "content": [
                { "type": "tool_use", "id": "t1", "name": "f", "input": {} },
                { "type": "text", "text": "hi" }
            ] } ]
        }));
        let c = to_canonical_request(r);
        // tool_use is ignored; only the text survives.
        assert_eq!(c.messages[0].content.as_text(), "hi");
    }

    #[test]
    fn outbound_maps_response_to_anthropic_shape() {
        use routeplane_types::{Choice, Usage};
        let resp = ChatCompletionResponse {
            id: "msg_123".into(),
            object: "chat.completion".into(),
            created: 1,
            model: "claude-3-5-sonnet".into(),
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: "assistant".into(),
                    content: "the answer".into(),
                    name: None,
                    cache_control: None,
                    tool_calls: None,
                    tool_call_id: None,
                    refusal: None,
                    reasoning_content: None,
                },
                finish_reason: "stop".into(),
                logprobs: None,
            }],
            usage: Usage {
                prompt_tokens: 12,
                completion_tokens: 7,
                total_tokens: 19,
                cached_tokens: None,
                cache_creation_tokens: None,
            },
            system_fingerprint: None,
            service_tier: None,
        };
        let v = to_anthropic_response(resp);
        assert_eq!(v["id"], "msg_123");
        assert_eq!(v["type"], "message");
        assert_eq!(v["role"], "assistant");
        assert_eq!(v["model"], "claude-3-5-sonnet");
        assert_eq!(v["content"][0]["type"], "text");
        assert_eq!(v["content"][0]["text"], "the answer");
        assert_eq!(v["stop_reason"], "end_turn");
        assert_eq!(v["stop_sequence"], Value::Null);
        assert_eq!(v["usage"]["input_tokens"], 12);
        assert_eq!(v["usage"]["output_tokens"], 7);
        // No cache fields present when None.
        assert!(v["usage"].get("cache_read_input_tokens").is_none());
    }

    #[test]
    fn outbound_maps_finish_reasons() {
        assert_eq!(map_finish_reason("stop"), "end_turn");
        assert_eq!(map_finish_reason("length"), "max_tokens");
        assert_eq!(map_finish_reason("tool_calls"), "tool_use");
        // Already-Anthropic values pass through faithfully.
        assert_eq!(map_finish_reason("end_turn"), "end_turn");
        assert_eq!(map_finish_reason("stop_sequence"), "stop_sequence");
        // Unknown → safe default.
        assert_eq!(map_finish_reason("weird"), "end_turn");
    }

    #[test]
    fn outbound_surfaces_cache_tokens_when_present() {
        use routeplane_types::{Choice, Usage};
        let resp = ChatCompletionResponse {
            id: "msg_1".into(),
            object: "chat.completion".into(),
            created: 1,
            model: "claude".into(),
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: "assistant".into(),
                    content: "ok".into(),
                    name: None,
                    cache_control: None,
                    tool_calls: None,
                    tool_call_id: None,
                    refusal: None,
                    reasoning_content: None,
                },
                finish_reason: "length".into(),
                logprobs: None,
            }],
            usage: Usage {
                prompt_tokens: 50,
                completion_tokens: 3,
                total_tokens: 53,
                cached_tokens: Some(40),
                cache_creation_tokens: Some(10),
            },
            system_fingerprint: None,
            service_tier: None,
        };
        let v = to_anthropic_response(resp);
        assert_eq!(v["stop_reason"], "max_tokens");
        assert_eq!(v["usage"]["cache_read_input_tokens"], 40);
        assert_eq!(v["usage"]["cache_creation_input_tokens"], 10);
    }
}
