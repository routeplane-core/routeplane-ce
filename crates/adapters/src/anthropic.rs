use crate::sse::SseLineBuffer;
use crate::vision::{is_anthropic_supported_media_type, parse_data_url};
use crate::{ChunkStream, Provider, ProviderError};
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use routeplane_types::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, Choice, ChunkChoice,
    ContentPart, Delta, FunctionCallChunk, Message, MessageContent, ToolCallChunk, Usage,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

pub struct AnthropicProvider {
    client: Client,
    /// Base URL for the Messages API. Defaults to the public Anthropic API;
    /// overridable so wiremock-backed tests can point the adapter at a mock
    /// server (engineering-design §24) without touching the hot path.
    base_url: String,
}

const ANTHROPIC_DEFAULT_BASE_URL: &str = "https://api.anthropic.com";

impl AnthropicProvider {
    pub fn new() -> Self {
        Self {
            client: crate::client::build_provider_client(),
            base_url: ANTHROPIC_DEFAULT_BASE_URL.to_string(),
        }
    }

    /// Construct pointing at a custom base URL (for tests / self-hosted proxies).
    /// A trailing slash is trimmed so URL joins stay correct (same contract as
    /// the Azure adapter's endpoint handling).
    pub fn with_base_url(base_url: impl Into<String>) -> Self {
        Self {
            client: crate::client::build_provider_client(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
        }
    }
}

impl Default for AnthropicProvider {
    fn default() -> Self {
        Self::new()
    }
}

/// Anthropic's `/v1/messages` requires `max_tokens` and does NOT accept a
/// `system` role inside `messages[]` — the system prompt is a TOP-LEVEL field.
#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    messages: Vec<AnthropicMessage>,
    max_tokens: u32,
    /// Top-level system prompt (Task #4): system messages are concatenated here,
    /// not passed as a `messages[]` entry (Anthropic rejects role="system").
    /// Anthropic accepts the system either as a bare STRING or as an array of
    /// text blocks (the form required to attach `cache_control`). We emit the
    /// bare string unless a system message requested prompt caching, in which
    /// case we emit a text block carrying `cache_control` — so the non-caching
    /// wire stays byte-identical (golden/parity unchanged).
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<AnthropicSystem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<String>>,
    /// Tool definitions translated to Anthropic's shape
    /// (`{name, description?, input_schema}`). `None` ⇒ omitted ⇒ byte-identical
    /// to a non-tool request.
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<Value>>,
    /// Anthropic's `tool_choice` object (translated from the canonical
    /// string/object `tool_choice`). `None` ⇒ omitted.
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<Value>,
    /// Anthropic request metadata — carries the canonical `user` as
    /// `metadata.user_id` instead of dropping it. `None` ⇒ omitted.
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<AnthropicMetadata>,
}

/// Anthropic's `metadata` object. Only `user_id` is mapped (from the canonical
/// `user`); other keys are Anthropic-internal.
#[derive(Debug, Serialize)]
struct AnthropicMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    user_id: Option<String>,
}

/// The top-level `system` of an Anthropic request: a bare string (no caching) or
/// an array of system text blocks (the form that can carry `cache_control`).
/// `#[serde(untagged)]` serializes `Text` as a JSON string and `Blocks` as a JSON
/// array — exactly Anthropic's two accepted shapes. The bare-string form keeps
/// the pre-caching wire byte-identical.
#[derive(Debug, Serialize)]
#[serde(untagged)]
enum AnthropicSystem {
    Text(String),
    Blocks(Vec<Value>),
}

/// Anthropic requires `max_tokens`; when the caller omits it we fall back to this
/// (the previous hardcoded value) so behaviour is unchanged for callers that do
/// not set it, but a caller-supplied `max_tokens` now flows through (Task #4,
/// and the substrate G2.3 param shaping relies on — see PRD-006 §4.1e).
const ANTHROPIC_DEFAULT_MAX_TOKENS: u32 = 1024;

/// Build the native Anthropic request from the canonical one, lifting any
/// `system` messages to the top-level field and mapping the threaded fields.
///
/// Prompt-caching passthrough (best-of-breed cost): a `cache_control` marker on a
/// `system` message, on a non-system message, or on an individual text content
/// part is emitted onto the corresponding native Anthropic block as a cache
/// breakpoint. A request with NO `cache_control` anywhere produces a wire body
/// byte-identical to before (system stays a bare string, text-only messages stay
/// bare strings).
/// Anthropic's `/v1/messages` returns a SINGLE completion — `n>1` cannot be
/// honored. Reject it explicitly with a non-retryable 422 rather than silently
/// returning one choice when the caller asked for several. `n` unset or `1` ⇒ Ok.
fn reject_multi_completion(request: &ChatCompletionRequest) -> Result<(), ProviderError> {
    match request.n {
        Some(n) if n > 1 => Err(ProviderError::BadRequest {
            provider: "anthropic".to_string(),
            status: 422,
            body: format!(
                "n_not_supported: anthropic /v1/messages returns a single completion, so n={n} \
                 cannot be honored — request n=1 or route to a provider that supports n>1"
            ),
        }),
        _ => Ok(()),
    }
}

fn build_anthropic_request(request: &ChatCompletionRequest) -> AnthropicRequest {
    // System messages: collect (text, optional cache_control). We keep the
    // per-message marker so a caller can cache a large system preamble.
    let mut system_parts: Vec<(String, Option<Value>)> = Vec::new();
    let mut messages: Vec<AnthropicMessage> = Vec::new();
    for m in &request.messages {
        if m.role == "system" {
            // Anthropic's `system` is top-level — flatten via as_text (any image
            // parts on a system message are dropped, as before).
            system_parts.push((m.content.as_text(), m.cache_control.clone()));
        } else if m.role == "tool" {
            // The tool-RESULT turn (OpenAI `role:"tool"`). Anthropic carries this
            // as a USER message whose content is a `tool_result` block keyed by
            // the `tool_use_id`. Adjacent tool results could be merged, but one
            // block per message is valid and keeps the mapping simple.
            let tool_use_id = m.tool_call_id.clone().unwrap_or_default();
            let block = json!({
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": m.content.as_text(),
            });
            messages.push(AnthropicMessage {
                role: "user".to_string(),
                content: AnthropicMessageContent::Blocks(vec![block]),
            });
        } else if m.role == "assistant" && m.tool_calls.is_some() {
            // The assistant tool-CALL turn: map each canonical tool_call to an
            // Anthropic `tool_use` block (`input` is the parsed arguments object).
            // Any text content precedes the tool_use blocks.
            let mut blocks: Vec<Value> = Vec::new();
            let text = m.content.as_text();
            if !text.is_empty() {
                blocks.push(json!({ "type": "text", "text": text }));
            }
            if let Some(calls) = &m.tool_calls {
                for call in calls {
                    // OpenAI arguments is a JSON-encoded STRING; Anthropic's
                    // `input` is an object. Parse it; fall back to an empty object
                    // on malformed JSON (never panic on the request thread).
                    let input: Value = serde_json::from_str(&call.function.arguments)
                        .unwrap_or_else(|_| json!({}));
                    blocks.push(json!({
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.function.name,
                        "input": input,
                    }));
                }
            }
            messages.push(AnthropicMessage {
                role: "assistant".to_string(),
                content: AnthropicMessageContent::Blocks(blocks),
            });
        } else {
            messages.push(AnthropicMessage {
                role: m.role.clone(),
                content: anthropic_content(&m.content, m.cache_control.as_ref()),
            });
        }
    }
    let system = build_anthropic_system(&system_parts);

    AnthropicRequest {
        model: request.model.clone(),
        messages,
        // max_tokens is threaded from the canonical request; Anthropic REQUIRES
        // it, so an absent value falls back to the documented default (G2.3).
        // `max_completion_tokens` (the o-series replacement) takes precedence
        // when set — Anthropic has one cap, so both map into it.
        max_tokens: request
            .max_completion_tokens
            .or(request.max_tokens)
            .unwrap_or(ANTHROPIC_DEFAULT_MAX_TOKENS),
        system,
        // Canonical `temperature` is OpenAI's 0..=2 domain; Anthropic accepts
        // only 0..=1 and 400s above 1. CLAMP so an otherwise-valid request still
        // succeeds on Anthropic (critical on a failover from an OpenAI-domain
        // provider, where a hard 400 mid-incident is what the chain must avoid).
        temperature: request.temperature.map(|t| t.clamp(0.0, 1.0)),
        top_p: request.top_p,
        stop_sequences: request.stop.clone(),
        tools: build_anthropic_tools(request.tools.as_deref()),
        tool_choice: request.tool_choice.as_ref().map(map_anthropic_tool_choice),
        // Map the canonical `user` → Anthropic `metadata.user_id` (was dropped).
        metadata: request
            .user
            .clone()
            .map(|u| AnthropicMetadata { user_id: Some(u) }),
    }
}

/// Translate canonical OpenAI tool definitions to Anthropic's `tools` shape.
/// OpenAI: `{type:"function", function:{name, description?, parameters}}`.
/// Anthropic: `{name, description?, input_schema}` (the JSON-Schema field is
/// renamed `parameters` → `input_schema`). Returns `None` for an absent/empty
/// list so the wire stays byte-identical for non-tool requests.
fn build_anthropic_tools(tools: Option<&[routeplane_types::Tool]>) -> Option<Vec<Value>> {
    let tools = tools?;
    if tools.is_empty() {
        return None;
    }
    let mapped = tools
        .iter()
        .map(|t| {
            let mut obj = serde_json::Map::new();
            obj.insert("name".to_string(), json!(t.function.name));
            if let Some(desc) = &t.function.description {
                obj.insert("description".to_string(), json!(desc));
            }
            // Anthropic requires `input_schema` (an object). Default to an empty
            // object schema when the function declared no parameters.
            let schema = t
                .function
                .parameters
                .clone()
                .unwrap_or_else(|| json!({"type": "object", "properties": {}}));
            obj.insert("input_schema".to_string(), schema);
            Value::Object(obj)
        })
        .collect();
    Some(mapped)
}

/// Translate the canonical `tool_choice` (OpenAI string or object) to Anthropic's
/// `tool_choice` object. OpenAI `"auto"`→`{type:"auto"}`, `"required"`→
/// `{type:"any"}`, `"none"`→`{type:"none"}` (Anthropic added `none` support),
/// and the force-a-function object `{type:"function",function:{name}}`→
/// `{type:"tool",name}`. An already-object value of an unrecognised shape is
/// passed through verbatim (forward-compat) rather than dropped.
fn map_anthropic_tool_choice(choice: &Value) -> Value {
    match choice {
        Value::String(s) => match s.as_str() {
            "auto" => json!({"type": "auto"}),
            "required" => json!({"type": "any"}),
            "none" => json!({"type": "none"}),
            other => json!({"type": other}),
        },
        Value::Object(obj) => {
            // OpenAI's force-a-function object → Anthropic `{type:"tool", name}`.
            if obj.get("type").and_then(|t| t.as_str()) == Some("function") {
                if let Some(name) = obj
                    .get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(|n| n.as_str())
                {
                    return json!({"type": "tool", "name": name});
                }
            }
            // Unknown object shape — pass through verbatim.
            choice.clone()
        }
        // Any other JSON kind — pass through (defensive, no panic).
        other => other.clone(),
    }
}

/// Build the top-level `system` field from the collected system messages.
///
/// Byte-identical fast path: when NO system message carries a `cache_control`
/// marker, the parts are joined with `\n\n` and emitted as a bare STRING — exactly
/// the pre-caching wire. When at least one system message requested caching, the
/// system is emitted as an array of `{"type":"text","text":…}` blocks (one per
/// system message), and the requesting message's block carries its
/// `cache_control` so it becomes a cache breakpoint.
fn build_anthropic_system(parts: &[(String, Option<Value>)]) -> Option<AnthropicSystem> {
    if parts.is_empty() {
        return None;
    }
    let any_cached = parts.iter().any(|(_, cc)| cc.is_some());
    if !any_cached {
        // No caching requested → bare string (byte-identical legacy wire).
        let joined = parts
            .iter()
            .map(|(t, _)| t.as_str())
            .collect::<Vec<_>>()
            .join("\n\n");
        return Some(AnthropicSystem::Text(joined));
    }
    // Caching requested on at least one system message → block array form.
    let blocks = parts
        .iter()
        .map(|(text, cc)| {
            let mut block = json!({ "type": "text", "text": text });
            attach_cache_control(&mut block, cc.as_ref());
            block
        })
        .collect();
    Some(AnthropicSystem::Blocks(blocks))
}

/// Attach a `cache_control` marker to an Anthropic content block in place, when
/// present. The marker is an OPAQUE passthrough of whatever the caller supplied
/// (canonically `{"type":"ephemeral"}`); we never synthesize or rewrite it. No-op
/// when `cache_control` is `None`, so the block is byte-identical to before.
fn attach_cache_control(block: &mut Value, cache_control: Option<&Value>) {
    if let (Some(obj), Some(cc)) = (block.as_object_mut(), cache_control) {
        obj.insert("cache_control".to_string(), cc.clone());
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct AnthropicMessage {
    role: String,
    /// Anthropic accepts EITHER a bare string OR an array of content blocks.
    /// We emit a bare string for text-only messages (byte-identical to the
    /// pre-vision behaviour, so golden/parity tests don't shift) and a block
    /// array only when the message carries image parts.
    content: AnthropicMessageContent,
}

/// The `content` of an Anthropic message: a bare string (text-only) or an array
/// of typed blocks (multimodal). `#[serde(untagged)]` makes `Text` serialize as
/// a JSON string and `Blocks` as a JSON array — exactly Anthropic's wire shape.
#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
enum AnthropicMessageContent {
    /// Text-only content — serializes as a bare JSON string (unchanged wire).
    Text(String),
    /// Multimodal content — an ordered array of text/image blocks.
    Blocks(Vec<Value>),
}

/// Translate a canonical [`MessageContent`] into Anthropic's message `content`,
/// applying prompt-caching markers.
///
/// `msg_cache_control` is the marker on the whole `Message` (the
/// `Message.cache_control` field). When present it makes the message a cache
/// breakpoint: it is attached to the LAST emitted block (Anthropic's convention —
/// a breakpoint caches everything up to and including the marked block).
///
/// Text-only content (`Text`, or `Parts` with no image parts AND no per-part /
/// message-level cache marker) collapses to the bare-string form so the wire is
/// byte-identical to the pre-vision/pre-caching adapter. Otherwise we emit
/// Anthropic's block array:
///
///   * text part   → `{"type":"text","text":…[,"cache_control":…]}`
///   * data-URL img → `{"type":"image","source":{"type":"base64","media_type":…,"data":…}}`
///   * http(s) img  → `{"type":"image","source":{"type":"url","url":…}}`
///
/// An image with an unsupported/missing media type is SKIPPED gracefully (text
/// parts are preserved) rather than producing a request Anthropic would reject.
fn anthropic_content(
    content: &MessageContent,
    msg_cache_control: Option<&Value>,
) -> AnthropicMessageContent {
    // Whether any text part carries its own cache_control marker.
    let part_cached = matches!(
        content,
        MessageContent::Parts(parts)
            if parts.iter().any(|p| matches!(
                p,
                ContentPart::Text { cache_control: Some(_), .. }
            ))
    );
    // Bare-string fast path: only when there is NO image, NO per-part marker, and
    // NO message-level marker — i.e. nothing that needs the block-array form.
    let needs_blocks = content.has_images() || part_cached || msg_cache_control.is_some();
    if !needs_blocks {
        return AnthropicMessageContent::Text(content.as_text());
    }

    let mut blocks: Vec<Value> = match content {
        // A bare Text message that nonetheless requested message-level caching:
        // emit a single text block so the marker has a block to attach to.
        MessageContent::Text(s) => vec![json!({ "type": "text", "text": s })],
        MessageContent::Parts(parts) => {
            let mut bs: Vec<Value> = Vec::with_capacity(parts.len());
            for part in parts {
                match part {
                    ContentPart::Text {
                        text,
                        cache_control,
                    } => {
                        let mut block = json!({ "type": "text", "text": text });
                        // Per-part marker rides its own text block.
                        attach_cache_control(&mut block, cache_control.as_ref());
                        bs.push(block);
                    }
                    ContentPart::ImageUrl { image_url } => {
                        if let Some(block) = anthropic_image_block(&image_url.url) {
                            bs.push(block);
                        }
                        // else: unsupported/missing media type — skip the image
                        // (do NOT panic, do NOT log the bytes).
                    }
                }
            }
            bs
        }
    };

    // A message-level marker attaches to the LAST block (the breakpoint).
    if let (Some(cc), Some(last)) = (msg_cache_control, blocks.last_mut()) {
        attach_cache_control(last, Some(cc));
    }

    AnthropicMessageContent::Blocks(blocks)
}

/// Map one image `url` to an Anthropic `image` block, or `None` when it cannot
/// be represented (unsupported/missing data-URL media type). HTTP(S) URLs map
/// to a `url` source; `data:<media_type>;base64,<data>` URIs split into a
/// `base64` source.
fn anthropic_image_block(url: &str) -> Option<Value> {
    if let Some(data) = parse_data_url(url) {
        if !is_anthropic_supported_media_type(data.media_type) {
            return None;
        }
        return Some(json!({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": data.media_type,
                "data": data.base64_payload,
            }
        }));
    }
    if url.starts_with("http://") || url.starts_with("https://") {
        return Some(json!({
            "type": "image",
            "source": { "type": "url", "url": url }
        }));
    }
    // Not a data URL and not an http(s) URL (e.g. an unsupported data: media
    // type already returned None above) — skip gracefully.
    None
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    id: String,
    model: String,
    content: Vec<AnthropicContent>,
    usage: AnthropicUsage,
    stop_reason: String,
}

#[derive(Debug, Deserialize)]
struct AnthropicContent {
    /// Anthropic's `type` discriminator on content blocks ("text", "tool_use", …).
    #[serde(rename = "type")]
    content_type: String,
    /// Present on `text` blocks. Optional so a `tool_use` block (which has no
    /// `text`) deserializes without error.
    #[serde(default)]
    text: Option<String>,
    /// `tool_use` block id (the correlation handle echoed on the tool result).
    #[serde(default)]
    id: Option<String>,
    /// `tool_use` block: the function name the model chose to call.
    #[serde(default)]
    name: Option<String>,
    /// `tool_use` block: the arguments as a JSON OBJECT (Anthropic's `input`).
    /// Re-serialized to a JSON STRING for the canonical OpenAI `arguments`.
    #[serde(default)]
    input: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
    /// Prompt-cache READ tokens — input served from the cache (billed cheaper).
    /// Absent on responses without prompt caching → maps to `cached_tokens: None`.
    #[serde(default)]
    cache_read_input_tokens: Option<u32>,
    /// Prompt-cache WRITE tokens — input written into the cache on this request.
    /// Absent without caching → maps to `cache_creation_tokens: None`.
    #[serde(default)]
    cache_creation_input_tokens: Option<u32>,
}

#[async_trait]
impl Provider for AnthropicProvider {
    fn name(&self) -> &'static str {
        "anthropic"
    }

    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
        api_key: String,
    ) -> Result<ChatCompletionResponse, ProviderError> {
        reject_multi_completion(&request)?;
        let url = format!("{}/v1/messages", self.base_url);

        let anthropic_req = build_anthropic_request(&request);

        let response = self
            .client
            .post(url)
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&anthropic_req)
            .send()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("anthropic", e))?;

        if !response.status().is_success() {
            return Err(crate::client::error_from_response("anthropic", response).await);
        }

        let result = response
            .json::<AnthropicResponse>()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("anthropic", e))?;

        // Concatenate text blocks; collect tool_use blocks → canonical tool_calls.
        let mut response_text = String::new();
        let mut tool_calls: Vec<routeplane_types::ToolCall> = Vec::new();
        for block in &result.content {
            match block.content_type.as_str() {
                "text" => {
                    if let Some(t) = &block.text {
                        response_text.push_str(t);
                    }
                }
                "tool_use" => {
                    // Anthropic's `input` is an OBJECT; OpenAI's `arguments` is a
                    // JSON-encoded STRING — re-serialize it. Never panic on the
                    // request thread: fall back to "{}" if serialization fails.
                    let arguments = block
                        .input
                        .as_ref()
                        .map(|i| serde_json::to_string(i).unwrap_or_else(|_| "{}".to_string()))
                        .unwrap_or_else(|| "{}".to_string());
                    tool_calls.push(routeplane_types::ToolCall {
                        id: block.id.clone().unwrap_or_default(),
                        tool_type: "function".to_string(),
                        function: routeplane_types::FunctionCall {
                            name: block.name.clone().unwrap_or_default(),
                            arguments,
                        },
                    });
                }
                _ => {} // future block types ignored gracefully
            }
        }
        // Map Anthropic's `stop_reason: "tool_use"` to OpenAI's "tool_calls";
        // otherwise normalize via map_stop_reason ("end_turn"/"stop_sequence" →
        // "stop", "max_tokens" → "length") so the buffered path emits the same
        // OpenAI-canonical finish_reason as the streaming path (message_delta),
        // rather than leaking Anthropic's raw enum to OpenAI-SDK clients.
        let finish_reason = if result.stop_reason == "tool_use" {
            "tool_calls".to_string()
        } else {
            map_stop_reason(&result.stop_reason)
        };
        let tool_calls = if tool_calls.is_empty() {
            None
        } else {
            Some(tool_calls)
        };

        Ok(ChatCompletionResponse {
            id: result.id,
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp() as u64,
            model: result.model,
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: "assistant".to_string(),
                    content: response_text.into(),
                    name: None,
                    cache_control: None,
                    tool_calls,
                    tool_call_id: None,
                    refusal: None,
                    reasoning_content: None,
                },
                finish_reason,
                // Anthropic does not return OpenAI-style logprobs.
                logprobs: None,
            }],
            usage: {
                // Anthropic reports THREE disjoint input counts: `input_tokens`
                // (fresh, non-cached) plus `cache_read_input_tokens` and
                // `cache_creation_input_tokens` — none folded into the others. The
                // canonical `prompt_tokens` is OpenAI-shaped: the FULL prompt input
                // (fresh + cache read + cache write), with `cached_tokens` being the
                // cache-read SUBSET counted within it. Summing here keeps cost/spend
                // attribution correct on the prompt-caching path (otherwise
                // prompt_tokens undercounts and cached_tokens can exceed it). Both
                // cache_* are None on non-cached responses ⇒ prompt_tokens ==
                // input_tokens ⇒ byte-identical to the pre-caching shape.
                let cache_read = result.usage.cache_read_input_tokens.unwrap_or(0);
                let cache_creation = result.usage.cache_creation_input_tokens.unwrap_or(0);
                let prompt_tokens = result.usage.input_tokens + cache_read + cache_creation;
                Usage {
                    prompt_tokens,
                    completion_tokens: result.usage.output_tokens,
                    total_tokens: prompt_tokens + result.usage.output_tokens,
                    // Prompt-caching surfacing: cache READ → cached_tokens (the
                    // subset within prompt_tokens), cache WRITE →
                    // cache_creation_tokens. Absent on non-cached responses (None).
                    cached_tokens: result.usage.cache_read_input_tokens,
                    cache_creation_tokens: result.usage.cache_creation_input_tokens,
                }
            },
            // OpenAI-only response metadata Anthropic does not report.
            system_fingerprint: None,
            service_tier: None,
        })
    }

    async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
        api_key: String,
    ) -> Result<ChunkStream, ProviderError> {
        reject_multi_completion(&request)?;
        let url = format!("{}/v1/messages", self.base_url);

        // max_tokens, system-role lifting and stop sequences are threaded through
        // identically to the buffered path (Task #4) via the shared builder.
        let anthropic_req = build_anthropic_request(&request);
        let mut body = serde_json::to_value(&anthropic_req)?;
        body["stream"] = json!(true);

        let resp = crate::client::streaming_client()
            .post(url)
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("anthropic", e))?;

        // Establishment failure -> typed Err so the proxy can retry/fall back.
        if !resp.status().is_success() {
            return Err(crate::client::error_from_response("anthropic", resp).await);
        }

        Ok(Box::pin(anthropic_sse_to_chunks(resp.bytes_stream())))
    }
}

/// Per-stream translation state. Anthropic spreads identity, content and usage
/// across distinct SSE event types, so we accumulate the bits we need to emit
/// OpenAI-shaped chunks (which want id/model/created on *every* chunk).
#[derive(Default)]
struct AnthropicStreamState {
    id: String,
    model: String,
    created: u64,
    input_tokens: u32,
    /// Prompt-cache READ / WRITE tokens from the `message_start` usage block, when
    /// caching applied. Carried to the final `message_delta` usage chunk so a
    /// streamed response surfaces cache savings too. `None` (omitted) without
    /// caching → byte-identical stream wire.
    cache_read_tokens: Option<u32>,
    cache_creation_tokens: Option<u32>,
    /// Next tool-call index to assign (OpenAI's `tool_calls[].index` is the
    /// position within the tool_calls array, NOT Anthropic's content-block index).
    /// Anthropic may interleave text + multiple `tool_use` blocks at arbitrary
    /// block indices, so we map each NEW tool_use block to a sequential 0-based
    /// tool-call index here.
    next_tool_index: u32,
    /// Maps the CURRENT Anthropic content-block index → its assigned tool-call
    /// index, so a following `input_json_delta` (which only carries the block
    /// index) emits its `arguments` fragment under the SAME tool-call index as the
    /// `content_block_start` that opened it. `None` when the open block is a text
    /// block (its deltas stream as content, not tool_calls).
    current_block_tool_index: Option<u32>,
    /// Whether any `tool_use` block was emitted — if so, the final finish_reason
    /// is "tool_calls" (matching OpenAI), regardless of Anthropic's stop_reason
    /// wording.
    saw_tool_use: bool,
}

/// Translate ONE Anthropic SSE `data:` payload into zero-or-more canonical
/// chunks, mutating `state`. Returns `Ok(true)` when the stream should end
/// (`message_stop`). Kept as a free function so it is unit-testable without a
/// live connection.
fn translate_anthropic_event(
    payload: &str,
    state: &mut AnthropicStreamState,
    out: &mut Vec<ChatCompletionChunk>,
) -> Result<bool, ProviderError> {
    let v: serde_json::Value = serde_json::from_str(payload).map_err(|e| -> ProviderError {
        format!("Anthropic stream parse error: {e}: {payload}").into()
    })?;

    match v.get("type").and_then(|t| t.as_str()).unwrap_or("") {
        "message_start" => {
            let msg = &v["message"];
            state.id = msg["id"].as_str().unwrap_or("anthropic-stream").to_string();
            state.model = msg["model"].as_str().unwrap_or_default().to_string();
            state.created = chrono::Utc::now().timestamp() as u64;
            state.input_tokens = msg["usage"]["input_tokens"].as_u64().unwrap_or(0) as u32;
            // Prompt-caching: Anthropic reports cache read/write in the
            // message_start usage block. Capture them (None when absent) to emit
            // on the final usage chunk.
            state.cache_read_tokens = msg["usage"]["cache_read_input_tokens"]
                .as_u64()
                .map(|t| t as u32);
            state.cache_creation_tokens = msg["usage"]["cache_creation_input_tokens"]
                .as_u64()
                .map(|t| t as u32);
            // Emit a leading role delta, matching OpenAI's first chunk.
            out.push(ChatCompletionChunk {
                id: state.id.clone(),
                object: "chat.completion.chunk".to_string(),
                created: state.created,
                model: state.model.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: Delta {
                        role: Some("assistant".to_string()),
                        ..Delta::default()
                    },
                    finish_reason: None,
                    logprobs: None,
                }],
                usage: None,
                system_fingerprint: None,
                service_tier: None,
            });
            Ok(false)
        }
        "content_block_start" => {
            // A new content block opened. A `tool_use` block carries the call's
            // id + name here (the OpenAI first-fragment shape); a `text` block
            // just streams text deltas as before. `input_json_delta` fragments
            // that follow only carry the block index, so remember this block's
            // assigned tool-call index for the deltas to attach to.
            let block = &v["content_block"];
            if block["type"].as_str() == Some("tool_use") {
                let tool_index = state.next_tool_index;
                state.next_tool_index += 1;
                state.current_block_tool_index = Some(tool_index);
                state.saw_tool_use = true;
                let id = block["id"].as_str().unwrap_or_default().to_string();
                let name = block["name"].as_str().unwrap_or_default().to_string();
                // First fragment for this tool-call index: id + type + name, with
                // empty arguments (which then stream via input_json_delta) — the
                // exact shape OpenAI emits.
                out.push(tool_call_chunk(
                    state,
                    ToolCallChunk {
                        index: tool_index,
                        id: Some(id),
                        tool_type: Some("function".to_string()),
                        function: Some(FunctionCallChunk {
                            name: Some(name),
                            arguments: Some(String::new()),
                        }),
                    },
                ));
            } else {
                // Text (or any non-tool) block — deltas stream as content.
                state.current_block_tool_index = None;
            }
            Ok(false)
        }
        "content_block_stop" => {
            // The current block ended; clear the tool-index mapping so a stray
            // delta can't attach to a finished block.
            state.current_block_tool_index = None;
            Ok(false)
        }
        "content_block_delta" => {
            let delta_type = v["delta"]["type"].as_str();
            if delta_type == Some("text_delta") {
                if let Some(text) = v["delta"]["text"].as_str() {
                    out.push(ChatCompletionChunk::content_delta(
                        state.id.clone(),
                        state.model.clone(),
                        state.created,
                        text,
                    ));
                }
            } else if delta_type == Some("input_json_delta") {
                // The tool-call arguments stream as partial JSON. Emit it as an
                // `arguments` fragment under the open block's tool-call index
                // (no id/type/name on continuation fragments — exactly OpenAI).
                if let Some(tool_index) = state.current_block_tool_index {
                    if let Some(partial) = v["delta"]["partial_json"].as_str() {
                        out.push(tool_call_chunk(
                            state,
                            ToolCallChunk {
                                index: tool_index,
                                id: None,
                                tool_type: None,
                                function: Some(FunctionCallChunk {
                                    name: None,
                                    arguments: Some(partial.to_string()),
                                }),
                            },
                        ));
                    }
                }
            }
            Ok(false)
        }
        "message_delta" => {
            // stop_reason + cumulative output_tokens arrive here. Map Anthropic's
            // stop_reason ("end_turn" etc.) onto OpenAI's "stop"/passthrough. When
            // a tool_use block streamed, the finish_reason is "tool_calls" (OpenAI's
            // convention) — Anthropic signals this as stop_reason:"tool_use", which
            // map_stop_reason handles, but we also force it if we saw any tool_use
            // block, to be robust to stop_reason wording.
            let stop = v["delta"]["stop_reason"].as_str().map(|s| {
                if s == "tool_use" || state.saw_tool_use {
                    "tool_calls".to_string()
                } else {
                    map_stop_reason(s)
                }
            });
            let output_tokens = v["usage"]["output_tokens"].as_u64().unwrap_or(0) as u32;
            // prompt_tokens is the FULL prompt input (fresh + cache read + cache
            // write); cached_tokens is the cache-read subset within it. Mirrors the
            // buffered path so streamed + buffered usage agree. Both cache_* are None
            // without caching ⇒ prompt_tokens == input_tokens ⇒ byte-identical.
            let cache_read = state.cache_read_tokens.unwrap_or(0);
            let cache_creation = state.cache_creation_tokens.unwrap_or(0);
            let prompt_tokens = state.input_tokens + cache_read + cache_creation;
            out.push(ChatCompletionChunk {
                id: state.id.clone(),
                object: "chat.completion.chunk".to_string(),
                created: state.created,
                model: state.model.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: Delta::default(),
                    finish_reason: stop,
                    logprobs: None,
                }],
                usage: Some(Usage {
                    prompt_tokens,
                    completion_tokens: output_tokens,
                    total_tokens: prompt_tokens + output_tokens,
                    cached_tokens: state.cache_read_tokens,
                    cache_creation_tokens: state.cache_creation_tokens,
                }),
                system_fingerprint: None,
                service_tier: None,
            });
            Ok(false)
        }
        "message_stop" => Ok(true),
        "error" => {
            // Anthropic emits a mid-stream error frame
            // (`event: error` / `data: {"type":"error","error":{"type":…}}`) and
            // then closes the socket. Without this arm it fell into `_ => Ok(false)`
            // and was swallowed: no chunk, stream not ended → the graceful socket
            // close made the proxy emit `data: [DONE]` and record a clean success,
            // so a provider-failed, TRUNCATED stream was reported as complete.
            // Surface it as an `Err` so the proxy terminates with an error frame
            // (NOT `[DONE]`). Only the stable error TYPE is carried (never the
            // free-form message) to keep provider internals out of the client frame.
            let etype = v["error"]["type"].as_str().unwrap_or("error");
            Err(format!("anthropic stream error: {etype}").into())
        }
        _ => Ok(false), // ping, content_block_start/stop, etc.
    }
}

/// Build a single-choice chunk carrying one incremental tool-call delta, stamped
/// with the stream's id/model/created (OpenAI wants those on every chunk).
fn tool_call_chunk(state: &AnthropicStreamState, tc: ToolCallChunk) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: state.id.clone(),
        object: "chat.completion.chunk".to_string(),
        created: state.created,
        model: state.model.clone(),
        choices: vec![ChunkChoice {
            index: 0,
            delta: Delta {
                tool_calls: Some(vec![tc]),
                ..Delta::default()
            },
            finish_reason: None,
            logprobs: None,
        }],
        usage: None,
        system_fingerprint: None,
        service_tier: None,
    }
}

fn map_stop_reason(anthropic: &str) -> String {
    match anthropic {
        "end_turn" | "stop_sequence" => "stop".to_string(),
        "max_tokens" => "length".to_string(),
        other => other.to_string(),
    }
}

/// Translate an Anthropic `/v1/messages` SSE byte stream into canonical chunks.
fn anthropic_sse_to_chunks(
    mut bytes: impl futures::Stream<Item = reqwest::Result<bytes::Bytes>> + Unpin + Send + 'static,
) -> impl futures::Stream<Item = Result<ChatCompletionChunk, ProviderError>> + Send + 'static {
    async_stream::stream! {
        let mut sse = SseLineBuffer::new();
        let mut state = AnthropicStreamState::default();
        while let Some(item) = bytes.next().await {
            let chunk = match item {
                Ok(b) => b,
                Err(e) => { yield Err(format!("Anthropic stream transport error: {e}").into()); break; }
            };
            sse.push(&chunk);
            while let Some(payload) = sse.next_payload() {
                let mut out = Vec::new();
                match translate_anthropic_event(&payload, &mut state, &mut out) {
                    Ok(done) => {
                        for c in out { yield Ok(c); }
                        if done { return; }
                    }
                    Err(e) => { yield Err(e); return; }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;

    #[test]
    fn message_start_emits_role_and_records_input_tokens() {
        let mut state = AnthropicStreamState::default();
        let mut out = Vec::new();
        let payload = r#"{"type":"message_start","message":{"id":"msg_1","model":"claude-3-5-sonnet","usage":{"input_tokens":12}}}"#;
        let done = translate_anthropic_event(payload, &mut state, &mut out).unwrap();
        assert!(!done);
        assert_eq!(state.input_tokens, 12);
        assert_eq!(state.id, "msg_1");
        assert_eq!(out[0].choices[0].delta.role.as_deref(), Some("assistant"));
    }

    #[test]
    fn text_delta_becomes_content_chunk() {
        let mut state = AnthropicStreamState {
            id: "msg_1".into(),
            model: "claude".into(),
            created: 1,
            input_tokens: 0,
            ..Default::default()
        };
        let mut out = Vec::new();
        let payload = r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#;
        translate_anthropic_event(payload, &mut state, &mut out).unwrap();
        assert_eq!(out[0].choices[0].delta.content.as_deref(), Some("Hello"));
    }

    #[test]
    fn message_delta_carries_finish_and_usage() {
        let mut state = AnthropicStreamState {
            id: "msg_1".into(),
            model: "claude".into(),
            created: 1,
            input_tokens: 12,
            ..Default::default()
        };
        let mut out = Vec::new();
        let payload = r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":7}}"#;
        translate_anthropic_event(payload, &mut state, &mut out).unwrap();
        assert_eq!(out[0].choices[0].finish_reason.as_deref(), Some("stop"));
        let usage = out[0].usage.as_ref().unwrap();
        assert_eq!(usage.prompt_tokens, 12);
        assert_eq!(usage.completion_tokens, 7);
        assert_eq!(usage.total_tokens, 19);
    }

    #[test]
    fn message_stop_ends_stream() {
        let mut state = AnthropicStreamState::default();
        let mut out = Vec::new();
        let done =
            translate_anthropic_event(r#"{"type":"message_stop"}"#, &mut state, &mut out).unwrap();
        assert!(done);
    }

    #[test]
    fn mid_stream_error_frame_is_surfaced_not_swallowed() {
        // Anthropic's documented mid-stream error frame must become a stream Err
        // (so the proxy terminates WITHOUT [DONE]) — not fall into the no-op
        // default arm that silently ended a truncated stream as a clean success.
        let mut state = AnthropicStreamState::default();
        let mut out = Vec::new();
        let res = translate_anthropic_event(
            r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#,
            &mut state,
            &mut out,
        );
        let err = res.expect_err("an error frame must surface as Err");
        assert!(err.to_string().contains("overloaded_error"));
        assert!(!err.to_string().contains("Overloaded"));
        assert!(out.is_empty(), "an error frame emits no content chunk");
    }

    // --- request translation (Task #4) ---------------------------------------

    fn msg(role: &str, content: &str) -> routeplane_types::Message {
        routeplane_types::Message {
            role: role.into(),
            content: content.into(),
            name: None,
            cache_control: None,
            tool_calls: None,
            tool_call_id: None,
            refusal: None,
            reasoning_content: None,
        }
    }

    fn req(messages: Vec<routeplane_types::Message>) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "claude-3-5-sonnet".into(),
            messages,
            temperature: None,
            top_p: None,
            stream: None,
            max_tokens: None,
            stop: None,
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

    #[test]
    fn system_message_is_lifted_to_top_level_not_in_messages() {
        let r = req(vec![msg("system", "be terse"), msg("user", "hello")]);
        let a = build_anthropic_request(&r);
        // No caching requested ⇒ system is a bare STRING (byte-identical legacy wire).
        let v = serde_json::to_value(&a).unwrap();
        assert_eq!(v["system"], serde_json::json!("be terse"));
        assert!(v["system"].is_string());
        assert_eq!(a.messages.len(), 1);
        assert_eq!(a.messages[0].role, "user");
        assert!(a.messages.iter().all(|m| m.role != "system"));
    }

    // --- tool / function calling (native Anthropic translation) ---------------

    use routeplane_types::{FunctionCall, FunctionDef, Tool, ToolCall};

    fn weather_tool() -> Tool {
        Tool {
            tool_type: "function".into(),
            function: FunctionDef {
                name: "get_weather".into(),
                description: Some("Get the weather".into()),
                parameters: Some(json!({
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"]
                })),
            },
        }
    }

    #[test]
    fn tools_map_to_anthropic_input_schema() {
        let mut r = req(vec![msg("user", "weather?")]);
        r.tools = Some(vec![weather_tool()]);
        r.tool_choice = Some(json!("auto"));
        let a = build_anthropic_request(&r);
        let v = serde_json::to_value(&a).unwrap();
        // OpenAI `parameters` is renamed to Anthropic `input_schema`; name/desc kept.
        assert_eq!(v["tools"][0]["name"], "get_weather");
        assert_eq!(v["tools"][0]["description"], "Get the weather");
        assert_eq!(v["tools"][0]["input_schema"]["type"], "object");
        assert!(v["tools"][0].get("parameters").is_none());
        // tool_choice "auto" → {type:"auto"}.
        assert_eq!(v["tool_choice"], json!({"type": "auto"}));
    }

    #[test]
    fn tool_choice_required_maps_to_any_and_force_maps_to_tool() {
        assert_eq!(
            map_anthropic_tool_choice(&json!("required")),
            json!({"type": "any"})
        );
        assert_eq!(
            map_anthropic_tool_choice(&json!("none")),
            json!({"type": "none"})
        );
        assert_eq!(
            map_anthropic_tool_choice(
                &json!({"type": "function", "function": {"name": "get_weather"}})
            ),
            json!({"type": "tool", "name": "get_weather"})
        );
    }

    #[test]
    fn assistant_tool_calls_become_tool_use_blocks() {
        let mut assistant = msg("assistant", "");
        assistant.tool_calls = Some(vec![ToolCall {
            id: "call_abc".into(),
            tool_type: "function".into(),
            function: FunctionCall {
                name: "get_weather".into(),
                arguments: "{\"location\":\"SF\"}".into(),
            },
        }]);
        let a = build_anthropic_request(&req(vec![msg("user", "weather?"), assistant]));
        let v = serde_json::to_value(&a).unwrap();
        let blocks = &v["messages"][1]["content"];
        assert_eq!(blocks[0]["type"], "tool_use");
        assert_eq!(blocks[0]["id"], "call_abc");
        assert_eq!(blocks[0]["name"], "get_weather");
        // OpenAI arguments STRING parsed into Anthropic input OBJECT.
        assert_eq!(blocks[0]["input"]["location"], "SF");
    }

    #[test]
    fn tool_role_message_becomes_tool_result_block() {
        let mut tool_msg = msg("tool", "{\"temp\":21}");
        tool_msg.tool_call_id = Some("call_abc".into());
        let a = build_anthropic_request(&req(vec![tool_msg]));
        let v = serde_json::to_value(&a).unwrap();
        // Anthropic carries the tool result as a USER message with a tool_result block.
        assert_eq!(v["messages"][0]["role"], "user");
        assert_eq!(v["messages"][0]["content"][0]["type"], "tool_result");
        assert_eq!(v["messages"][0]["content"][0]["tool_use_id"], "call_abc");
        assert_eq!(v["messages"][0]["content"][0]["content"], "{\"temp\":21}");
    }

    #[tokio::test]
    async fn response_tool_use_maps_to_canonical_tool_calls() {
        use wiremock::matchers::{body_partial_json, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        // Anthropic returns a tool_use content block + stop_reason "tool_use".
        let resp = serde_json::json!({
            "id": "msg_1", "model": "claude-3-5-sonnet", "stop_reason": "tool_use",
            "content": [
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather",
                 "input": {"location": "SF"}}
            ],
            "usage": {"input_tokens": 9, "output_tokens": 12}
        });
        // Assert the outbound tools carry input_schema.
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(body_partial_json(serde_json::json!({
                "tools": [{"name": "get_weather", "input_schema": {"type": "object"}}]
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(resp))
            .mount(&server)
            .await;

        let provider = AnthropicProvider::with_base_url(server.uri());
        let mut r = req(vec![msg("user", "weather in SF?")]);
        r.tools = Some(vec![weather_tool()]);
        let out = provider
            .chat_completion(r, "ak-test".into())
            .await
            .expect("mock call succeeds");
        // stop_reason "tool_use" → finish_reason "tool_calls".
        assert_eq!(out.choices[0].finish_reason, "tool_calls");
        // text concatenated, tool_use → canonical tool_call (input OBJECT → STRING).
        assert_eq!(out.choices[0].message.content.as_text(), "Let me check.");
        let calls = out.choices[0].message.tool_calls.as_ref().unwrap();
        assert_eq!(calls[0].id, "toolu_1");
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments, "{\"location\":\"SF\"}");
    }

    #[test]
    fn max_tokens_is_threaded_not_hardcoded() {
        let mut r = req(vec![msg("user", "hi")]);
        r.max_tokens = Some(4096);
        r.stop = Some(vec!["END".into()]);
        let a = build_anthropic_request(&r);
        assert_eq!(a.max_tokens, 4096); // not the old hardcoded 1024
        assert_eq!(a.stop_sequences.as_deref(), Some(&["END".to_string()][..]));
    }

    #[test]
    fn max_tokens_defaults_when_absent() {
        let a = build_anthropic_request(&req(vec![msg("user", "hi")]));
        assert_eq!(a.max_tokens, ANTHROPIC_DEFAULT_MAX_TOKENS);
    }

    #[test]
    fn temperature_is_clamped_into_anthropics_domain() {
        let mut r = req(vec![msg("user", "hi")]);
        r.temperature = Some(1.5);
        assert_eq!(build_anthropic_request(&r).temperature, Some(1.0));
        r.temperature = Some(0.5);
        assert_eq!(build_anthropic_request(&r).temperature, Some(0.5));
        r.temperature = None;
        assert_eq!(build_anthropic_request(&r).temperature, None);
    }

    #[test]
    fn user_maps_to_metadata_user_id() {
        let mut r = req(vec![msg("user", "hi")]);
        r.user = Some("end-user-42".into());
        let a = build_anthropic_request(&r);
        assert_eq!(
            a.metadata.as_ref().and_then(|m| m.user_id.as_deref()),
            Some("end-user-42")
        );
        r.user = None;
        assert!(build_anthropic_request(&r).metadata.is_none());
    }

    #[test]
    fn n_greater_than_one_is_rejected_422() {
        let mut r = req(vec![msg("user", "hi")]);
        r.n = Some(2);
        let err = reject_multi_completion(&r).expect_err("n>1 must be rejected");
        assert_eq!(err.status(), Some(422));
        assert!(err.to_string().contains("n_not_supported"));
        r.n = Some(1);
        assert!(reject_multi_completion(&r).is_ok());
        r.n = None;
        assert!(reject_multi_completion(&r).is_ok());
    }

    // --- vision passthrough (native Anthropic image blocks) -------------------

    use routeplane_types::{ContentPart, ImageUrlContent, MessageContent};

    fn img_msg(parts: Vec<ContentPart>) -> routeplane_types::Message {
        routeplane_types::Message {
            role: "user".into(),
            content: MessageContent::Parts(parts),
            name: None,
            cache_control: None,
            tool_calls: None,
            tool_call_id: None,
            refusal: None,
            reasoning_content: None,
        }
    }

    fn text_part(s: &str) -> ContentPart {
        ContentPart::Text {
            text: s.into(),
            cache_control: None,
        }
    }

    fn image_part(url: &str) -> ContentPart {
        ContentPart::ImageUrl {
            image_url: ImageUrlContent {
                url: url.into(),
                detail: None,
            },
        }
    }

    #[test]
    fn openai_only_request_fields_never_leak_into_anthropic_body() {
        // response_format / logit_bias / logprobs / top_logprobs / service_tier /
        // seed / reasoning_effort have NO place in Anthropic's native /v1/messages
        // body. The native builder maps only named fields, so they must be absent
        // (the strip is implicit — never mapped, never emitted).
        let mut r = req(vec![msg("user", "hi")]);
        r.response_format = Some(serde_json::json!({"type": "json_object"}));
        r.seed = Some(7);
        r.logprobs = Some(true);
        r.top_logprobs = Some(5);
        r.service_tier = Some("flex".into());
        r.reasoning_effort = Some("high".into());
        let mut lb = std::collections::BTreeMap::new();
        lb.insert("123".to_string(), 1.0f32);
        r.logit_bias = Some(lb);

        let a = build_anthropic_request(&r);
        let v = serde_json::to_value(&a).unwrap();
        for leaked in [
            "response_format",
            "logit_bias",
            "logprobs",
            "top_logprobs",
            "service_tier",
            "seed",
            "reasoning_effort",
        ] {
            assert!(
                v.get(leaked).is_none(),
                "{leaked} must NOT appear in the Anthropic request body"
            );
        }
    }

    #[test]
    fn text_only_message_serializes_as_bare_string_byte_identical() {
        // The text-only wire MUST stay a bare string (not a block array), so the
        // golden/parity corpus does not shift.
        let a = build_anthropic_request(&req(vec![msg("user", "hello")]));
        let v = serde_json::to_value(&a).unwrap();
        assert_eq!(v["messages"][0]["content"], serde_json::json!("hello"));
        assert!(v["messages"][0]["content"].is_string());
    }

    #[test]
    fn text_only_parts_message_still_serializes_as_bare_string() {
        // A `Parts` message with no images must ALSO collapse to a bare string.
        let r = req(vec![img_msg(vec![text_part("a"), text_part("b")])]);
        let a = build_anthropic_request(&r);
        let v = serde_json::to_value(&a).unwrap();
        assert_eq!(v["messages"][0]["content"], serde_json::json!("ab"));
    }

    #[test]
    fn data_url_image_becomes_base64_source_block() {
        let r = req(vec![img_msg(vec![
            text_part("describe"),
            image_part("data:image/png;base64,iVBORw0KGgo="),
        ])]);
        let a = build_anthropic_request(&r);
        let v = serde_json::to_value(&a).unwrap();
        let content = &v["messages"][0]["content"];
        assert!(content.is_array());
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "describe");
        assert_eq!(content[1]["type"], "image");
        assert_eq!(content[1]["source"]["type"], "base64");
        assert_eq!(content[1]["source"]["media_type"], "image/png");
        assert_eq!(content[1]["source"]["data"], "iVBORw0KGgo=");
    }

    #[test]
    fn http_url_image_becomes_url_source_block() {
        let r = req(vec![img_msg(vec![image_part(
            "https://example.com/cat.jpg",
        )])]);
        let a = build_anthropic_request(&r);
        let v = serde_json::to_value(&a).unwrap();
        let content = &v["messages"][0]["content"];
        assert_eq!(content[0]["type"], "image");
        assert_eq!(content[0]["source"]["type"], "url");
        assert_eq!(content[0]["source"]["url"], "https://example.com/cat.jpg");
    }

    #[test]
    fn unsupported_media_type_image_is_skipped_text_kept() {
        let r = req(vec![img_msg(vec![
            text_part("look"),
            image_part("data:image/svg+xml;base64,PHN2Zz4="),
        ])]);
        let a = build_anthropic_request(&r);
        let v = serde_json::to_value(&a).unwrap();
        let content = &v["messages"][0]["content"];
        // The text block survives; the unsupported image is dropped (no panic).
        assert!(content.is_array());
        assert_eq!(content.as_array().unwrap().len(), 1);
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "look");
    }

    // --- prompt-caching passthrough (Anthropic cache_control) ------------------

    fn ephemeral() -> Value {
        json!({"type": "ephemeral"})
    }

    #[test]
    fn cached_system_message_becomes_text_block_with_cache_control() {
        // A system message carrying cache_control emits the system as a block
        // array with the ephemeral marker (NOT a bare string).
        let mut sys = msg("system", "huge cacheable preamble");
        sys.cache_control = Some(ephemeral());
        let a = build_anthropic_request(&req(vec![sys, msg("user", "hi")]));
        let v = serde_json::to_value(&a).unwrap();
        assert!(v["system"].is_array());
        assert_eq!(v["system"][0]["type"], "text");
        assert_eq!(v["system"][0]["text"], "huge cacheable preamble");
        assert_eq!(v["system"][0]["cache_control"], ephemeral());
    }

    #[test]
    fn uncached_system_stays_bare_string_byte_identical() {
        // No cache_control anywhere ⇒ system is a bare string (legacy wire).
        let a = build_anthropic_request(&req(vec![msg("system", "be terse"), msg("user", "hi")]));
        let v = serde_json::to_value(&a).unwrap();
        assert!(v["system"].is_string());
        assert_eq!(v["system"], "be terse");
    }

    #[test]
    fn message_level_cache_control_marks_last_block() {
        // A message-level marker forces the block-array form and attaches to the
        // LAST block (the cache breakpoint).
        let mut m = img_msg(vec![text_part("a"), text_part("b")]);
        m.cache_control = Some(ephemeral());
        let a = build_anthropic_request(&req(vec![m]));
        let v = serde_json::to_value(&a).unwrap();
        let content = &v["messages"][0]["content"];
        assert!(content.is_array());
        // First block has no marker; the LAST block carries it.
        assert!(content[0].get("cache_control").is_none());
        assert_eq!(content[1]["cache_control"], ephemeral());
    }

    #[test]
    fn per_part_cache_control_rides_its_own_text_block() {
        let cached_part = ContentPart::Text {
            text: "cache me".into(),
            cache_control: Some(ephemeral()),
        };
        let a =
            build_anthropic_request(&req(vec![img_msg(vec![cached_part, text_part("not me")])]));
        let v = serde_json::to_value(&a).unwrap();
        let content = &v["messages"][0]["content"];
        assert_eq!(content[0]["text"], "cache me");
        assert_eq!(content[0]["cache_control"], ephemeral());
        assert!(content[1].get("cache_control").is_none());
    }

    #[tokio::test]
    async fn buffered_emits_cache_control_on_wire_and_maps_cache_usage() {
        use wiremock::matchers::{body_partial_json, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        // Response carries cache read + creation tokens → must map to canonical.
        let resp = serde_json::json!({
            "id":"msg_1","model":"claude-3-5-sonnet","stop_reason":"end_turn",
            "content":[{"type":"text","text":"ok"}],
            "usage":{
                "input_tokens":50,"output_tokens":3,
                "cache_read_input_tokens":40,"cache_creation_input_tokens":10
            }
        });
        // Assert the outbound system block carries the ephemeral cache marker.
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(body_partial_json(serde_json::json!({
                "system": [
                    {"type":"text","text":"cacheable preamble",
                     "cache_control":{"type":"ephemeral"}}
                ]
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(resp))
            .mount(&server)
            .await;

        let provider = AnthropicProvider::with_base_url(server.uri());
        let mut sys = msg("system", "cacheable preamble");
        sys.cache_control = Some(ephemeral());
        let out = provider
            .chat_completion(req(vec![sys, msg("user", "hi")]), "ak-test".into())
            .await
            .expect("mock call succeeds");
        // Cache READ → cached_tokens; cache WRITE → cache_creation_tokens.
        assert_eq!(out.usage.cached_tokens, Some(40));
        assert_eq!(out.usage.cache_creation_tokens, Some(10));
    }

    #[tokio::test]
    async fn buffered_maps_stop_reason_and_sums_cache_tokens() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        // Truncation + prompt caching: Anthropic's input_tokens is the FRESH input
        // only; cache read/creation are disjoint counts. The buffered path must map
        // stop_reason "max_tokens" → OpenAI "length" and sum prompt_tokens to the
        // FULL input (50 + 40 + 10 = 100), with cached_tokens the read subset.
        let resp = serde_json::json!({
            "id":"msg_2","model":"claude-3-5-sonnet","stop_reason":"max_tokens",
            "content":[{"type":"text","text":"trunc"}],
            "usage":{
                "input_tokens":50,"output_tokens":7,
                "cache_read_input_tokens":40,"cache_creation_input_tokens":10
            }
        });
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(200).set_body_json(resp))
            .mount(&server)
            .await;

        let provider = AnthropicProvider::with_base_url(server.uri());
        let out = provider
            .chat_completion(req(vec![msg("user", "hi")]), "ak-test".into())
            .await
            .expect("mock call succeeds");
        // stop_reason "max_tokens" → OpenAI "length" (was leaked verbatim before).
        assert_eq!(out.choices[0].finish_reason, "length");
        // prompt_tokens = fresh + cache read + cache creation.
        assert_eq!(out.usage.prompt_tokens, 100);
        assert_eq!(out.usage.total_tokens, 107);
        // cached_tokens stays the cache-read SUBSET within prompt_tokens.
        assert_eq!(out.usage.cached_tokens, Some(40));
        assert_eq!(out.usage.cache_creation_tokens, Some(10));
    }

    #[test]
    fn response_without_cache_usage_maps_to_none() {
        // A non-cached Anthropic usage (no cache_* fields) → both None, so the
        // canonical Usage is byte-identical to the pre-caching shape.
        let raw = r#"{"input_tokens":5,"output_tokens":2}"#;
        let u: AnthropicUsage = serde_json::from_str(raw).unwrap();
        assert_eq!(u.cache_read_input_tokens, None);
        assert_eq!(u.cache_creation_input_tokens, None);
    }

    #[test]
    fn stream_message_start_captures_cache_tokens_and_message_delta_emits_them() {
        let mut state = AnthropicStreamState::default();
        let mut out = Vec::new();
        let start = r#"{"type":"message_start","message":{"id":"m1","model":"claude","usage":{"input_tokens":50,"cache_read_input_tokens":40,"cache_creation_input_tokens":10}}}"#;
        translate_anthropic_event(start, &mut state, &mut out).unwrap();
        assert_eq!(state.cache_read_tokens, Some(40));
        assert_eq!(state.cache_creation_tokens, Some(10));

        let mut out2 = Vec::new();
        let delta = r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3}}"#;
        translate_anthropic_event(delta, &mut state, &mut out2).unwrap();
        let usage = out2[0].usage.as_ref().unwrap();
        // prompt_tokens is the FULL input (fresh 50 + cache read 40 + cache write
        // 10 = 100), mirroring the buffered path; cached_tokens is the read subset.
        assert_eq!(usage.prompt_tokens, 100);
        assert_eq!(usage.total_tokens, 103);
        assert_eq!(usage.cached_tokens, Some(40));
        assert_eq!(usage.cache_creation_tokens, Some(10));
    }

    #[tokio::test]
    async fn buffered_forwards_image_block_to_anthropic_wire() {
        use wiremock::matchers::{body_partial_json, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        let resp = serde_json::json!({
            "id":"msg_1","model":"claude-3-5-sonnet","stop_reason":"end_turn",
            "content":[{"type":"text","text":"a cat"}],
            "usage":{"input_tokens":9,"output_tokens":2}
        });
        // Assert the outbound body carries the native Anthropic image block.
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(body_partial_json(serde_json::json!({
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this"},
                        {"type": "image", "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": "AAAA"
                        }}
                    ]
                }]
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(resp))
            .mount(&server)
            .await;

        let provider = AnthropicProvider::with_base_url(server.uri());
        let r = req(vec![img_msg(vec![
            text_part("what is this"),
            image_part("data:image/jpeg;base64,AAAA"),
        ])]);
        let out = provider
            .chat_completion(r, "ak-test".into())
            .await
            .expect("mock call succeeds");
        assert_eq!(out.choices[0].message.content.as_text(), "a cat");
    }

    // --- wiremock: max_tokens reaches the wire (proves shaping is not lossy) ---

    #[tokio::test]
    async fn buffered_threads_max_tokens_to_anthropic_wire() {
        use wiremock::matchers::{body_partial_json, header, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        let resp = serde_json::json!({
            "id":"msg_1","model":"claude-3-5-sonnet","stop_reason":"end_turn",
            "content":[{"type":"text","text":"hello"}],
            "usage":{"input_tokens":5,"output_tokens":1}
        });
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(header("x-api-key", "ak-test"))
            .and(body_partial_json(serde_json::json!({"max_tokens": 4096})))
            .respond_with(ResponseTemplate::new(200).set_body_json(resp))
            .mount(&server)
            .await;

        let provider = AnthropicProvider::with_base_url(server.uri());
        let mut r = req(vec![msg("user", "hi")]);
        r.max_tokens = Some(4096);
        let out = provider
            .chat_completion(r, "ak-test".into())
            .await
            .expect("mock call succeeds");
        assert_eq!(out.choices[0].message.content.as_text(), "hello");
    }

    #[tokio::test]
    async fn max_completion_tokens_maps_to_cap_and_never_leaks_to_anthropic() {
        // Native-dialect contract: `max_completion_tokens` wins over `max_tokens`
        // for Anthropic's single cap, and the raw OpenAI key is NEVER forwarded
        // (Anthropic rejects unknown OpenAI keys).
        use wiremock::matchers::{body_partial_json, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        let resp = serde_json::json!({
            "id":"msg_1","model":"claude-3-5-sonnet","stop_reason":"end_turn",
            "content":[{"type":"text","text":"hello"}],
            "usage":{"input_tokens":5,"output_tokens":1}
        });
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(body_partial_json(serde_json::json!({"max_tokens": 2048})))
            .respond_with(ResponseTemplate::new(200).set_body_json(resp))
            .mount(&server)
            .await;

        let provider = AnthropicProvider::with_base_url(server.uri());
        let mut r = req(vec![msg("user", "hi")]);
        r.max_tokens = Some(4096);
        r.max_completion_tokens = Some(2048); // takes precedence over max_tokens
        provider
            .chat_completion(r, "ak-test".into())
            .await
            .expect("mock call succeeds");

        // Replay the recorded body: no `max_completion_tokens` key may leak
        // alongside the mapped cap.
        let received = &server.received_requests().await.unwrap()[0];
        let sent: serde_json::Value = serde_json::from_slice(&received.body).unwrap();
        assert!(
            sent.get("max_completion_tokens").is_none(),
            "the OpenAI field itself must not leak; only the mapped max_tokens"
        );
        assert_eq!(sent["max_tokens"], 2048);
    }

    #[tokio::test]
    async fn translates_full_anthropic_sse_stream() {
        let raw = concat!(
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-3-5-sonnet\",\"usage\":{\"input_tokens\":10}}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hel\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"lo\"}}\n\n",
            "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\n",
            "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
        );
        let byte_stream = stream::iter(vec![Ok(bytes::Bytes::from(raw.to_string()))]);
        let chunks: Vec<_> = anthropic_sse_to_chunks(byte_stream)
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();
        assert_eq!(chunks.len(), 4);
        assert_eq!(
            chunks[0].choices[0].delta.role.as_deref(),
            Some("assistant")
        );
        assert_eq!(chunks[1].choices[0].delta.content.as_deref(), Some("Hel"));
        assert_eq!(chunks[2].choices[0].delta.content.as_deref(), Some("lo"));
        assert_eq!(chunks[3].choices[0].finish_reason.as_deref(), Some("stop"));
        assert_eq!(chunks[3].usage.as_ref().unwrap().total_tokens, 12);
    }

    // --- streaming tool calls (native Anthropic translation) ------------------

    #[test]
    fn content_block_start_tool_use_emits_id_name_first_delta() {
        // content_block_start for a tool_use block → the FIRST tool-call delta:
        // id + type + name + empty arguments (the OpenAI first-fragment shape).
        let mut state = AnthropicStreamState {
            id: "msg_1".into(),
            model: "claude".into(),
            created: 1,
            ..Default::default()
        };
        let mut out = Vec::new();
        let payload = r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather"}}"#;
        translate_anthropic_event(payload, &mut state, &mut out).unwrap();
        assert_eq!(out.len(), 1);
        let tc = out[0].choices[0].delta.tool_calls.as_ref().unwrap();
        assert_eq!(tc[0].index, 0);
        assert_eq!(tc[0].id.as_deref(), Some("toolu_1"));
        assert_eq!(tc[0].tool_type.as_deref(), Some("function"));
        let func = tc[0].function.as_ref().unwrap();
        assert_eq!(func.name.as_deref(), Some("get_weather"));
        assert_eq!(func.arguments.as_deref(), Some(""));
        // Block-index → tool-index mapping is recorded for following deltas.
        assert_eq!(state.current_block_tool_index, Some(0));
        assert!(state.saw_tool_use);
    }

    #[test]
    fn input_json_delta_streams_arguments_without_id_or_name() {
        // After a tool_use content_block_start, an input_json_delta streams the
        // arguments as a partial-JSON fragment — no id/type/name (continuation).
        let mut state = AnthropicStreamState {
            id: "msg_1".into(),
            model: "claude".into(),
            created: 1,
            ..Default::default()
        };
        let mut out = Vec::new();
        translate_anthropic_event(
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather"}}"#,
            &mut state,
            &mut out,
        )
        .unwrap();
        let mut out2 = Vec::new();
        translate_anthropic_event(
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"location\":"}}"#,
            &mut state,
            &mut out2,
        )
        .unwrap();
        let tc = out2[0].choices[0].delta.tool_calls.as_ref().unwrap();
        assert_eq!(tc[0].index, 0);
        assert!(tc[0].id.is_none());
        assert!(tc[0].tool_type.is_none());
        let func = tc[0].function.as_ref().unwrap();
        assert!(func.name.is_none());
        assert_eq!(func.arguments.as_deref(), Some("{\"location\":"));
    }

    #[test]
    fn interleaved_text_and_tool_blocks_get_sequential_tool_indices() {
        // A text block (index 0) then two tool_use blocks (index 1, 2) → the
        // tool-call indices are 0 and 1 (position in the tool_calls array, NOT
        // the Anthropic content-block index).
        let mut state = AnthropicStreamState {
            id: "msg_1".into(),
            model: "claude".into(),
            created: 1,
            ..Default::default()
        };
        // Text block opens at content-block index 0.
        let mut out = Vec::new();
        translate_anthropic_event(
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            &mut state,
            &mut out,
        )
        .unwrap();
        assert!(out.is_empty()); // text block start emits nothing
        assert_eq!(state.current_block_tool_index, None);
        // First tool_use at content-block index 1 → tool index 0.
        let mut o1 = Vec::new();
        translate_anthropic_event(
            r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"t1","name":"a"}}"#,
            &mut state,
            &mut o1,
        )
        .unwrap();
        assert_eq!(
            o1[0].choices[0].delta.tool_calls.as_ref().unwrap()[0].index,
            0
        );
        // content_block_stop clears the mapping.
        let mut ostop = Vec::new();
        translate_anthropic_event(
            r#"{"type":"content_block_stop","index":1}"#,
            &mut state,
            &mut ostop,
        )
        .unwrap();
        assert_eq!(state.current_block_tool_index, None);
        // Second tool_use at content-block index 2 → tool index 1.
        let mut o2 = Vec::new();
        translate_anthropic_event(
            r#"{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"t2","name":"b"}}"#,
            &mut state,
            &mut o2,
        )
        .unwrap();
        assert_eq!(
            o2[0].choices[0].delta.tool_calls.as_ref().unwrap()[0].index,
            1
        );
    }

    #[test]
    fn stop_reason_tool_use_maps_to_tool_calls_finish() {
        let mut state = AnthropicStreamState {
            id: "msg_1".into(),
            model: "claude".into(),
            created: 1,
            input_tokens: 9,
            saw_tool_use: true,
            ..Default::default()
        };
        let mut out = Vec::new();
        let payload = r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":12}}"#;
        translate_anthropic_event(payload, &mut state, &mut out).unwrap();
        assert_eq!(
            out[0].choices[0].finish_reason.as_deref(),
            Some("tool_calls")
        );
    }

    #[tokio::test]
    async fn translates_full_anthropic_tool_use_stream() {
        // A full Anthropic tool_use stream: message_start → content_block_start
        // (tool_use) → input_json_delta x2 → content_block_stop → message_delta
        // (stop_reason tool_use) → message_stop. The canonical chunks: role, the
        // id/name first tool-call delta, two incremental argument deltas, then the
        // tool_calls finish chunk.
        let raw = concat!(
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-3-5-sonnet\",\"usage\":{\"input_tokens\":9}}}\n\n",
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"get_weather\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"location\\\":\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"\\\"SF\\\"}\"}}\n\n",
            "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":12}}\n\n",
            "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
        );
        let byte_stream = stream::iter(vec![Ok(bytes::Bytes::from(raw.to_string()))]);
        let chunks: Vec<_> = anthropic_sse_to_chunks(byte_stream)
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();
        // role, first tool-call delta, arg delta, arg delta, finish.
        assert_eq!(chunks.len(), 5);
        assert_eq!(
            chunks[0].choices[0].delta.role.as_deref(),
            Some("assistant")
        );
        // First tool-call delta carries id + name + empty args.
        let first = chunks[1].choices[0].delta.tool_calls.as_ref().unwrap();
        assert_eq!(first[0].index, 0);
        assert_eq!(first[0].id.as_deref(), Some("toolu_1"));
        assert_eq!(first[0].tool_type.as_deref(), Some("function"));
        assert_eq!(
            first[0].function.as_ref().unwrap().name.as_deref(),
            Some("get_weather")
        );
        assert_eq!(
            first[0].function.as_ref().unwrap().arguments.as_deref(),
            Some("")
        );
        // Incremental argument fragments (no id/name on continuation deltas).
        let a1 = chunks[2].choices[0].delta.tool_calls.as_ref().unwrap();
        assert!(a1[0].id.is_none());
        assert_eq!(
            a1[0].function.as_ref().unwrap().arguments.as_deref(),
            Some("{\"location\":")
        );
        let a2 = chunks[3].choices[0].delta.tool_calls.as_ref().unwrap();
        assert_eq!(
            a2[0].function.as_ref().unwrap().arguments.as_deref(),
            Some("\"SF\"}")
        );
        // Reassembled arguments form the full JSON object.
        let reassembled: String = [a1, a2]
            .iter()
            .map(|d| d[0].function.as_ref().unwrap().arguments.clone().unwrap())
            .collect();
        assert_eq!(reassembled, "{\"location\":\"SF\"}");
        // Finish chunk → tool_calls.
        assert_eq!(
            chunks[4].choices[0].finish_reason.as_deref(),
            Some("tool_calls")
        );
    }
}
