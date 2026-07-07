use crate::sse::SseLineBuffer;
use crate::{ChunkStream, Provider, ProviderError};
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use routeplane_types::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, Choice, ChunkChoice,
    ContentPart, Delta, EmbeddingData, EmbeddingRequest, EmbeddingResponse, EmbeddingUsage,
    FunctionCallChunk, Message, MessageContent, RerankDocument, RerankRequest, RerankResponse,
    RerankResult, RerankUsage, Tool, ToolCall, ToolCallChunk, Usage,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// Cohere adapter — Cohere's v2 API (`/v2/chat` and `/v2/embed`) uses a
/// different wire shape than OpenAI, so this adapter translates in both
/// directions. Models: `command-r-plus`, `command-r`, `command-light`, etc.
///
/// Cohere is notable for its multilingual strength and on-prem deployment
/// options — useful for sovereign routing to jurisdictions where Cohere has
/// data centres (Canada, EU, US).
pub struct CohereProvider {
    client: Client,
    base_url: String,
}

const COHERE_DEFAULT_BASE_URL: &str = "https://api.cohere.com";

impl CohereProvider {
    pub fn new() -> Self {
        Self {
            client: crate::client::build_provider_client(),
            base_url: COHERE_DEFAULT_BASE_URL.to_string(),
        }
    }

    pub fn with_base_url(base_url: impl Into<String>) -> Self {
        Self {
            client: crate::client::build_provider_client(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
        }
    }
}

impl Default for CohereProvider {
    fn default() -> Self {
        Self::new()
    }
}

// --- Cohere v2 chat wire shapes -----------------------------------------------

#[derive(Debug, Serialize)]
struct CohereChatRequest {
    model: String,
    messages: Vec<CohereMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    p: Option<f32>, // top_p (Cohere uses `p`)
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<String>>,
    /// Cohere v2 chat supports these at the top level under the SAME names as the
    /// canonical request, so they thread straight through. Note the value-semantics
    /// gap: Cohere's penalties are 0.0–1.0 where OpenAI's are -2.0–2.0 — this is a
    /// raw passthrough (no remap), matching the other adapters. `None` +
    /// skip_serializing_if ⇒ a request that omits them is byte-identical.
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<i64>,
    /// Tool/function declarations. Cohere v2's `tools` shape is identical to the
    /// canonical OpenAI [`Tool`] (`{type:"function",function:{name,description,
    /// parameters}}`), so we forward the canonical structs verbatim. `None` ⇒
    /// omitted ⇒ byte-identical to a non-tool request.
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<Tool>>,
    /// Cohere v2 `tool_choice` (`"REQUIRED"`/`"NONE"`). Mapped from the canonical
    /// `tool_choice`; unmappable shapes are omitted rather than sent invalid.
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<String>,
    /// Cohere v2 structured-output toggle, mapped from the canonical OpenAI
    /// `response_format`. Both `{"type":"json_object"}` and
    /// `{"type":"json_schema",…}` map to Cohere's JSON-object mode. Previously the
    /// field was silently dropped, so a structured-output request got free-form
    /// prose that then failed the caller's `JSON.parse`. `None` ⇒ omitted ⇒
    /// byte-identical to a plain request.
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<serde_json::Value>,
    stream: bool,
}

#[derive(Debug, Serialize)]
struct CohereMessage {
    role: String,
    /// Cohere v2 allows a message with no content (an assistant tool-call turn
    /// carries `tool_calls` and no text). Omitted when empty so such a turn is
    /// well-formed and a normal text message is byte-identical.
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<CohereMessageContent>,
    /// Assistant tool-CALL turn: the tool calls the assistant decided to make.
    /// Cohere v2's shape is identical to the canonical [`ToolCall`]
    /// (`{id,type:"function",function:{name,arguments}}`), so we forward them
    /// verbatim. `None` ⇒ omitted ⇒ byte-identical to a plain message.
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCall>>,
    /// Tool-RESULT turn (`role:"tool"`): the id of the tool call this message
    /// answers (matches the assistant `ToolCall.id`). `None` ⇒ omitted.
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

/// Cohere v2 `chat` accepts a message `content` that is EITHER a bare string OR
/// an array of OpenAI-shaped content parts (`{"type":"text",…}` /
/// `{"type":"image_url","image_url":{"url",…}}`). We emit the bare string for
/// text-only messages (byte-identical to the pre-vision wire) and forward the
/// canonical parts verbatim when the message carries images — Cohere's vision
/// content shape IS the OpenAI/canonical shape, so no per-part translation is
/// needed.
#[derive(Debug, Serialize)]
#[serde(untagged)]
enum CohereMessageContent {
    /// Text-only content — serializes as a bare JSON string (unchanged wire).
    Text(String),
    /// Multimodal content — the canonical OpenAI-shaped content parts, forwarded
    /// verbatim (Cohere v2 accepts this shape directly).
    Parts(Vec<ContentPart>),
}

/// Translate a canonical [`MessageContent`] into Cohere's message `content`.
/// Text-only collapses to a bare string (wire unchanged); a message with image
/// parts forwards the canonical parts array unchanged (Cohere v2 == OpenAI
/// content-part shape).
fn cohere_content(content: &MessageContent) -> CohereMessageContent {
    match content {
        MessageContent::Parts(parts) if content.has_images() => {
            CohereMessageContent::Parts(parts.clone())
        }
        // Text, or Parts with no images: flatten to the bare-string form.
        _ => CohereMessageContent::Text(content.as_text()),
    }
}

#[derive(Debug, Deserialize)]
struct CohereChatResponse {
    id: String,
    message: CohereChatMessage,
    usage: Option<CohereUsage>,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CohereChatMessage {
    /// The response role — always "assistant" for single-choice responses.
    /// Deserialized but not consumed (the canonical response hardcodes "assistant").
    #[allow(dead_code)]
    role: String,
    /// Cohere v2 returns assistant `content` as an array of typed blocks
    /// (`[{"type":"text","text":…}]`) on a text turn, and OMITS it on a pure
    /// tool-call turn. Carried as a raw [`Value`] so both the legacy bare-string
    /// form and the v2 block-array form parse; flattened to text via
    /// [`cohere_content_text`]. `default` so a tool-only turn (no content) parses.
    #[serde(default)]
    content: Option<Value>,
    /// Tool calls the model decided to make. Cohere v2's shape matches the
    /// canonical [`ToolCall`] (`{id,type:"function",function:{name,arguments}}`).
    /// Absent on a plain text response.
    #[serde(default)]
    tool_calls: Option<Vec<ToolCall>>,
    /// Cohere's reasoning text preceding the tool calls. Deserialized so it does
    /// NOT leak into `content`, but intentionally NOT surfaced in the canonical
    /// response (OpenAI has no equivalent field).
    #[serde(default)]
    #[allow(dead_code)]
    tool_plan: Option<String>,
}

/// Flatten Cohere v2's assistant `content` into plain text. v2 returns
/// `[{"type":"text","text":…}, …]`; older/streaming shapes may hand back a bare
/// string. Both collapse to the concatenated text. A missing/non-text content
/// (a pure tool-call turn) yields the empty string — never a panic.
fn cohere_content_text(content: &Option<Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| {
                if p.get("type").and_then(|t| t.as_str()) == Some("text") {
                    p.get("text").and_then(|t| t.as_str())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

#[derive(Debug, Deserialize)]
struct CohereUsage {
    #[serde(default)]
    tokens: Option<CohereTokenCounts>,
    #[serde(default)]
    prompt_tokens: Option<u32>,
    #[serde(default)]
    completion_tokens: Option<u32>,
    #[serde(default)]
    total_tokens: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct CohereTokenCounts {
    #[serde(default)]
    input_tokens: Option<u32>,
    #[serde(default)]
    output_tokens: Option<u32>,
}

/// Translate canonical messages to Cohere's v2 message array. System messages
/// become `role: "system"`; user/assistant pass through. Tool-calling turns map
/// to Cohere v2's OpenAI-shaped message forms: an assistant turn carrying
/// `tool_calls`, and a `role:"tool"` result turn carrying `tool_call_id` +
/// content.
fn build_cohere_request(request: &ChatCompletionRequest, stream: bool) -> CohereChatRequest {
    let messages = request.messages.iter().map(build_cohere_message).collect();
    CohereChatRequest {
        model: request.model.clone(),
        messages,
        // `max_completion_tokens` (the o-series replacement) takes precedence
        // over `max_tokens` — Cohere has one cap, so both map into it.
        max_tokens: request.max_completion_tokens.or(request.max_tokens),
        temperature: request.temperature,
        p: request.top_p,
        stop_sequences: request.stop.clone(),
        // Cohere v2 supports these natively — thread them through so they are no
        // longer silently dropped (upstream fidelity fix). Raw passthrough (see the struct note).
        presence_penalty: request.presence_penalty,
        frequency_penalty: request.frequency_penalty,
        seed: request.seed,
        // Tools: Cohere v2's shape == the canonical Tool, forward verbatim. An
        // absent/empty list stays omitted so a non-tool request is byte-identical.
        tools: cohere_tools(request.tools.as_deref()),
        tool_choice: request
            .tool_choice
            .as_ref()
            .and_then(map_cohere_tool_choice),
        response_format: map_cohere_response_format(request.response_format.as_ref()),
        stream,
    }
}

/// Map the canonical OpenAI `response_format` to Cohere v2's `response_format`.
/// Both `{"type":"json_object"}` and `{"type":"json_schema",…}` map to Cohere's
/// JSON-object mode (`{"type":"json_object"}`) — this forces valid JSON output
/// (the essential fix: the field was previously dropped, yielding prose). Any
/// other (or absent) shape yields `None` so a non-JSON request stays
/// byte-identical and no invalid field is ever sent.
fn map_cohere_response_format(rf: Option<&serde_json::Value>) -> Option<serde_json::Value> {
    match rf?.get("type").and_then(|t| t.as_str()) {
        Some("json_object") | Some("json_schema") => {
            Some(serde_json::json!({ "type": "json_object" }))
        }
        _ => None,
    }
}

/// Map one canonical [`Message`] to a Cohere v2 message.
///
/// * a `role:"tool"` message → Cohere's tool-RESULT turn (`tool_call_id` +
///   string content).
/// * an `assistant` message with `tool_calls` → Cohere's tool-CALL turn
///   (`tool_calls`, content omitted when empty so the turn is well-formed).
/// * everything else → the unchanged text/vision message (byte-identical wire).
fn build_cohere_message(m: &Message) -> CohereMessage {
    if m.role == "tool" {
        // Tool-RESULT turn. Cohere v2 accepts a bare string for tool content
        // (OpenAI shape); the result text rides `content`, keyed by tool_call_id.
        return CohereMessage {
            role: "tool".to_string(),
            content: Some(CohereMessageContent::Text(m.content.as_text())),
            tool_calls: None,
            tool_call_id: m.tool_call_id.clone(),
        };
    }
    if m.role == "assistant" && m.tool_calls.is_some() {
        // Assistant tool-CALL turn. Forward tool_calls verbatim (Cohere shape ==
        // canonical). Emit content only when the assistant also produced text,
        // so a pure tool-call turn omits it (well-formed v2 wire).
        let text = m.content.as_text();
        return CohereMessage {
            role: "assistant".to_string(),
            content: if text.is_empty() {
                None
            } else {
                Some(CohereMessageContent::Text(text))
            },
            tool_calls: m.tool_calls.clone(),
            tool_call_id: None,
        };
    }
    CohereMessage {
        role: match m.role.as_str() {
            "assistant" => "assistant".to_string(),
            "system" => "system".to_string(),
            _ => "user".to_string(),
        },
        content: Some(cohere_content(&m.content)),
        tool_calls: None,
        tool_call_id: None,
    }
}

/// Forward canonical tool declarations to Cohere v2 `tools`. Cohere's shape is
/// identical to the canonical [`Tool`], so we clone them through. `None`/empty ⇒
/// omitted so a non-tool request's wire is byte-identical.
fn cohere_tools(tools: Option<&[Tool]>) -> Option<Vec<Tool>> {
    let tools = tools?;
    if tools.is_empty() {
        None
    } else {
        Some(tools.to_vec())
    }
}

/// Map the canonical `tool_choice` (OpenAI string or object) to Cohere v2's
/// enum (`"REQUIRED"`/`"NONE"`). OpenAI `"required"`→`REQUIRED`, `"none"`→`NONE`.
/// `"auto"` is Cohere's default (model chooses), so it is OMITTED rather than
/// sent. A force-a-specific-function object has no v2 equivalent → omitted (do
/// not send an invalid value); the model still sees the tool via `tools`.
fn map_cohere_tool_choice(choice: &Value) -> Option<String> {
    match choice {
        Value::String(s) => match s.to_ascii_lowercase().as_str() {
            "required" => Some("REQUIRED".to_string()),
            "none" => Some("NONE".to_string()),
            // "auto" (and anything else) → let Cohere default.
            _ => None,
        },
        // Object form (force a named function) — no v2 equivalent; omit.
        _ => None,
    }
}

fn cohere_usage_to_canonical(usage: &Option<CohereUsage>) -> Usage {
    match usage {
        Some(u) => {
            let (prompt, completion) = if let Some(ref tc) = u.tokens {
                (tc.input_tokens.unwrap_or(0), tc.output_tokens.unwrap_or(0))
            } else {
                (
                    u.prompt_tokens.unwrap_or(0),
                    u.completion_tokens.unwrap_or(0),
                )
            };
            Usage {
                prompt_tokens: prompt,
                completion_tokens: completion,
                total_tokens: u.total_tokens.unwrap_or(prompt + completion),
                cached_tokens: None,
                cache_creation_tokens: None,
            }
        }
        None => Usage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
            cached_tokens: None,
            cache_creation_tokens: None,
        },
    }
}

/// A unique response id per Cohere stream. Cohere's `message-start` DOES carry a
/// top-level `id` (captured into the stream state below), but if it is ever
/// absent we synthesize a unique fallback so a streamed response never collapses
/// to the old shared literal `"cohere-stream"` (which broke client dedupe /
/// correlation keyed on the completion id, upstream fidelity fix). Monotonic within the process: a
/// high-resolution timestamp plus an atomic sequence so two streams that start in
/// the same instant still differ.
fn cohere_response_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let ts = chrono::Utc::now()
        .timestamp_nanos_opt()
        .unwrap_or(0)
        .unsigned_abs();
    format!("cohere-{ts:x}{seq:x}")
}

/// Per-stream translation state for Cohere v2. Cohere spreads identity across a
/// `message-start` event and streams tool-call arguments INCREMENTALLY (like
/// OpenAI) across `tool-call-start`/`tool-call-delta` events. OpenAI-shaped chunks
/// want id/model/created on EVERY chunk, so we hold the id (from message-start, or
/// a synthesized fallback), the model (the request model — Cohere stream events
/// don't echo it, matching the buffered path's `model: request.model`) and a
/// created timestamp stamped ONCE per stream (not per-chunk-regenerated, not 0).
/// `saw_tool_call` drives the synthesized finish_reason: we NEVER emit
/// `finish_reason:"tool_calls"` on a stream that carried zero tool-call deltas
/// (that is the exact broken-agent-loop shape upstream fidelity work fixes).
struct CohereStreamState {
    id: String,
    model: String,
    created: u64,
    saw_tool_call: bool,
}

impl CohereStreamState {
    fn new(model: String) -> Self {
        Self {
            // Fallback id until `message-start` supplies the real one.
            id: cohere_response_id(),
            model,
            // Stamped once, here — reused across every chunk of this stream.
            created: chrono::Utc::now().timestamp() as u64,
            saw_tool_call: false,
        }
    }
}

/// Build a single-choice tool-call-delta chunk from `state` (id/model/created) and
/// one [`ToolCallChunk`] fragment — mirrors anthropic.rs's `tool_call_chunk`.
fn cohere_tool_call_chunk(state: &CohereStreamState, tc: ToolCallChunk) -> ChatCompletionChunk {
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

/// Parse a Cohere v2 streaming event, mutating per-stream `state`. Cohere's SSE
/// format is JSON lines keyed by a `type` field. We translate:
/// * `message-start` — capture the response `id` (emits no chunk).
/// * `content-delta` — a text delta.
/// * `tool-call-start` — the FIRST fragment of a tool call (id + type + name +
///   empty arguments), keyed by the event's top-level `index`. Cohere carries the
///   call under `delta.message.tool_calls` as a SINGLE object (not an array).
/// * `tool-call-delta` — an incremental `arguments` fragment (no id/type/name),
///   under `delta.message.tool_calls.function.arguments`. Cohere streams arguments
///   incrementally (like OpenAI), so we emit fragments per delta.
/// * `message-end` — finish reason + usage.
///
/// Returns `Ok(None)` for events we don't translate (`tool-call-end`,
/// `content-start`/`content-end`, `tool-plan-delta`, keepalives). The `id`/`type`/
/// `name`-first then `arguments`-fragments accumulation is the same per-index shape
/// OpenAI/Anthropic/Gemini use.
fn parse_cohere_stream_event(
    line: &str,
    state: &mut CohereStreamState,
) -> Result<Option<ChatCompletionChunk>, ProviderError> {
    // The SseLineBuffer strips the `data:` prefix, so `line` is the JSON body.
    let value: serde_json::Value = serde_json::from_str(line).map_err(|e| {
        ProviderError::translation(format!("Cohere stream parse error: {e}: {line}"))
    })?;

    let event_type = value.get("type").and_then(|t| t.as_str());

    // message-start: capture the response id (Cohere puts it at the top level).
    // The model is not echoed per-event, so `state.model` (the request model)
    // stands. No leading role delta is emitted — byte-identical to the pre-tool
    // stream, which never emitted one.
    if event_type == Some("message-start") {
        if let Some(id) = value.get("id").and_then(|v| v.as_str()) {
            state.id = id.to_string();
        }
        return Ok(None);
    }

    // tool-call-start: first fragment for a tool call — id + type + name, with the
    // (empty) arguments that then stream via `tool-call-delta`. The tool-call index
    // rides the event's top-level `index`; the call itself is a single object under
    // `delta.message.tool_calls`.
    if event_type == Some("tool-call-start") {
        state.saw_tool_call = true;
        let index = value.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
        let tc = value
            .get("delta")
            .and_then(|d| d.get("message"))
            .and_then(|m| m.get("tool_calls"));
        let id = tc
            .and_then(|t| t.get("id"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let function = tc.and_then(|t| t.get("function"));
        let name = function
            .and_then(|f| f.get("name"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let arguments = function
            .and_then(|f| f.get("arguments"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        return Ok(Some(cohere_tool_call_chunk(
            state,
            ToolCallChunk {
                index,
                id,
                tool_type: Some("function".to_string()),
                function: Some(FunctionCallChunk {
                    name,
                    arguments: Some(arguments),
                }),
            },
        )));
    }

    // tool-call-delta: an incremental `arguments` fragment for the open tool call
    // (no id/type/name on continuation fragments — exactly OpenAI). Keyed by the
    // event's top-level `index`.
    if event_type == Some("tool-call-delta") {
        let index = value.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
        let arguments = value
            .get("delta")
            .and_then(|d| d.get("message"))
            .and_then(|m| m.get("tool_calls"))
            .and_then(|t| t.get("function"))
            .and_then(|f| f.get("arguments"))
            .and_then(|v| v.as_str());
        if let Some(arguments) = arguments {
            return Ok(Some(cohere_tool_call_chunk(
                state,
                ToolCallChunk {
                    index,
                    id: None,
                    tool_type: None,
                    function: Some(FunctionCallChunk {
                        name: None,
                        arguments: Some(arguments.to_string()),
                    }),
                },
            )));
        }
        return Ok(None);
    }

    // content-delta: extract the text delta. (`tool-call-end` and other structural
    // events carry no content/finish and fall through to `Ok(None)`.)
    if let Some(delta) = value
        .get("delta")
        .and_then(|d| d.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|c| c.get("text"))
        .and_then(|t| t.as_str())
    {
        if !delta.is_empty() {
            return Ok(Some(ChatCompletionChunk::content_delta(
                state.id.clone(),
                state.model.clone(),
                state.created,
                delta,
            )));
        }
    }

    // message-end: finish reason + usage. Cohere v2 nests both under `delta`
    // (`delta.finish_reason`, `delta.usage`) — the SAME `delta.*` nesting the
    // content path reads — so read there FIRST. The top-level / `response.*`
    // fallbacks preserve the older synthetic shapes some callers/tests emit.
    if let Some(finish) = value
        .get("delta")
        .and_then(|d| d.get("finish_reason"))
        .or_else(|| value.get("finish_reason"))
        .or_else(|| value.get("response").and_then(|r| r.get("finish_reason")))
        .and_then(|v| v.as_str())
    {
        // OpenAI convention: once tool-call deltas streamed, the finish is
        // "tool_calls" regardless of Cohere's exact reason wording. The SAFETY
        // INTERIM lives in the else arm: with NO tool-call deltas we NEVER surface
        // "tool_calls" — a bare `TOOL_CALL` with zero streamed calls degrades to
        // "stop" (emitting "tool_calls" with no tool_calls is the upstream fidelity fix bug).
        let mapped = if state.saw_tool_call {
            "tool_calls".to_string()
        } else {
            match finish {
                "COMPLETE" | "STOP_SEQUENCE" | "TOOL_CALL" => "stop",
                "MAX_TOKENS" => "length",
                other => other,
            }
            .to_string()
        };
        // Surface token usage on the terminal chunk (matching the buffered path and
        // every OpenAI-wire adapter's `include_usage`). Cohere v2 reports it under
        // `delta.usage` (`{tokens:{input_tokens,output_tokens}}`); accept a
        // top-level `usage` too. Absent ⇒ None ⇒ byte-identical to before.
        let usage = value
            .get("delta")
            .and_then(|d| d.get("usage"))
            .or_else(|| value.get("usage"))
            .and_then(|u| serde_json::from_value::<CohereUsage>(u.clone()).ok())
            .map(|u| cohere_usage_to_canonical(&Some(u)));
        return Ok(Some(ChatCompletionChunk {
            id: state.id.clone(),
            object: "chat.completion.chunk".to_string(),
            created: state.created,
            model: state.model.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta::default(),
                finish_reason: Some(mapped),
                logprobs: None,
            }],
            usage,
            system_fingerprint: None,
            service_tier: None,
        }));
    }

    Ok(None)
}

#[async_trait]
impl Provider for CohereProvider {
    fn name(&self) -> &'static str {
        "cohere"
    }

    fn resident_regions(&self) -> Vec<String> {
        let region = std::env::var("COHERE_REGION").unwrap_or_default();
        if region.is_empty() {
            Vec::new()
        } else {
            vec![region]
        }
    }

    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
        api_key: String,
    ) -> Result<ChatCompletionResponse, ProviderError> {
        let url = format!("{}/v2/chat", self.base_url);
        let cohere_req = build_cohere_request(&request, false);

        let response = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {api_key}"))
            .json(&cohere_req)
            .send()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("cohere", e))?;

        if !response.status().is_success() {
            return Err(crate::client::error_from_response("cohere", response).await);
        }

        let result = response
            .json::<CohereChatResponse>()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("cohere", e))?;

        let usage = cohere_usage_to_canonical(&result.usage);

        // Flatten the assistant text (v2 returns a `[{type:"text",text}]` block
        // array; `tool_plan` is intentionally NOT folded in — OpenAI has no such
        // field, and surfacing it as content would corrupt the message).
        let response_text = cohere_content_text(&result.message.content);
        // Map Cohere's tool calls verbatim (shape == canonical). Empty/absent ⇒
        // None ⇒ byte-identical to a plain text response.
        let tool_calls = result.message.tool_calls.filter(|calls| !calls.is_empty());
        let has_tool_calls = tool_calls.is_some();

        Ok(ChatCompletionResponse {
            id: result.id,
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp() as u64,
            model: request.model,
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: "assistant".to_string(),
                    content: response_text.into(),
                    name: None,
                    cache_control: None,
                    // Cohere v2 tool calls mapped to canonical tool_calls
                    // (id/type/function.name/function.arguments — arguments is
                    // already the JSON-encoded string OpenAI expects).
                    tool_calls,
                    tool_call_id: None,
                    refusal: None,
                    reasoning_content: None,
                },
                // OpenAI compatibility: whenever tool calls are present the
                // finish_reason MUST be "tool_calls" (clients branch on it),
                // regardless of Cohere's reported reason. Otherwise map Cohere's
                // reason, defaulting to "stop".
                finish_reason: if has_tool_calls {
                    "tool_calls".to_string()
                } else {
                    result
                        .finish_reason
                        .map(|f| match f.as_str() {
                            "COMPLETE" | "STOP_SEQUENCE" => "stop".to_string(),
                            "MAX_TOKENS" => "length".to_string(),
                            "TOOL_CALL" => "tool_calls".to_string(),
                            other => other.to_string(),
                        })
                        .unwrap_or_else(|| "stop".to_string())
                },
                // Cohere v2 chat does not return OpenAI-style logprobs.
                logprobs: None,
            }],
            usage,
            // OpenAI-only response metadata Cohere does not report.
            system_fingerprint: None,
            service_tier: None,
        })
    }

    async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
        api_key: String,
    ) -> Result<ChunkStream, ProviderError> {
        let url = format!("{}/v2/chat", self.base_url);
        // Capture the model before the request is borrowed — Cohere stream events
        // don't echo it, so the per-stream state carries it (matches the buffered
        // path's `model: request.model`).
        let model = request.model.clone();
        let cohere_req = build_cohere_request(&request, true);

        let resp = crate::client::streaming_client()
            .post(url)
            .header("Authorization", format!("Bearer {api_key}"))
            .json(&cohere_req)
            .send()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("cohere", e))?;

        if !resp.status().is_success() {
            return Err(crate::client::error_from_response("cohere", resp).await);
        }

        let mut buf = SseLineBuffer::new();
        let stream = resp
            .bytes_stream()
            .map(|result| result.map_err(|e| ProviderError::network("cohere", e.to_string())));

        Ok(Box::pin(async_stream::try_stream! {
            // One state (id/model/created stamped once, saw_tool_call) shared across
            // every event of THIS stream.
            let mut state = CohereStreamState::new(model);
            let mut stream = std::pin::pin!(stream);
            while let Some(chunk) = stream.next().await {
                let bytes = chunk?;
                buf.push(&bytes);
                while let Some(payload) = buf.next_payload() {
                    if let Some(parsed) = parse_cohere_stream_event(&payload, &mut state)? {
                        yield parsed;
                    }
                }
            }
        }))
    }

    async fn embeddings(
        &self,
        request: EmbeddingRequest,
        api_key: String,
    ) -> Result<EmbeddingResponse, ProviderError> {
        let url = format!("{}/v2/embed", self.base_url);

        // Normalize the OpenAI `input` to Cohere's `texts`: a single string →
        // `["string"]`, an array of strings → as-is (order is load-bearing — the
        // response `index` maps back to it). The canonical `EmbeddingInput` only
        // models the string / string-array forms (no token-id arrays), so there
        // is no other shape to reject here — but an empty batch is a malformed
        // request: fail it cleanly with a typed 422 *before* the network call
        // rather than round-tripping to Cohere for a guaranteed error.
        let texts = request.input.to_vec();
        if texts.is_empty() {
            return Err(ProviderError::BadRequest {
                provider: "cohere".to_string(),
                status: 422,
                body: "invalid_request: embeddings `input` must contain at least one string"
                    .to_string(),
            });
        }

        // Cohere v2 embed wire shape. `input_type` is REQUIRED for v3+ models;
        // OpenAI's request has no equivalent, so we default to "search_document"
        // (the correct choice for general document/text embedding). We always
        // request `embedding_types: ["float"]` and read `embeddings.float[i]`
        // back below. The key flows in the Authorization header only — never the
        // body, never a log.
        let input_type = "search_document";
        let body = json!({
            "model": request.model,
            "texts": texts,
            "input_type": input_type,
            "embedding_types": ["float"],
        });

        let response = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {api_key}"))
            .json(&body)
            .send()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("cohere", e))?;

        if !response.status().is_success() {
            return Err(crate::client::error_from_response("cohere", response).await);
        }

        // Cohere embed response: { id, embeddings: { float: [[...], ...] }, ... }
        let result: serde_json::Value = response
            .json()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("cohere", e))?;

        let float_arrays = result
            .get("embeddings")
            .and_then(|e| e.get("float"))
            .and_then(|f| f.as_array())
            .cloned()
            .unwrap_or_default();

        let mut data = Vec::new();
        for (idx, arr) in float_arrays.into_iter().enumerate() {
            let vec: Vec<f32> = arr
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_f64().map(|f| f as f32))
                        .collect()
                })
                .unwrap_or_default();
            data.push(EmbeddingData {
                object: "embedding".to_string(),
                index: idx as u32,
                embedding: vec.into(),
            });
        }

        let total_tokens = result
            .get("meta")
            .and_then(|m| m.get("billed_units"))
            .and_then(|b| b.get("input_tokens"))
            .and_then(|t| t.as_u64())
            .unwrap_or(0) as u32;

        Ok(EmbeddingResponse {
            object: "list".to_string(),
            data,
            model: request.model,
            usage: EmbeddingUsage {
                prompt_tokens: total_tokens,
                total_tokens,
            },
        })
    }

    async fn rerank(
        &self,
        request: RerankRequest,
        api_key: String,
    ) -> Result<RerankResponse, ProviderError> {
        let url = format!("{}/v2/rerank", self.base_url);

        // An empty document set is a malformed request: fail it cleanly with a
        // typed 422 *before* the network call rather than round-tripping to
        // Cohere for a guaranteed error (mirrors the embeddings empty-input
        // guard). No panic on malformed input.
        if request.documents.is_empty() {
            return Err(ProviderError::BadRequest {
                provider: "cohere".to_string(),
                status: 422,
                body: "invalid_request: rerank `documents` must contain at least one string"
                    .to_string(),
            });
        }

        // Cohere v2 rerank wire shape. `top_n` is forwarded only when set
        // (Cohere defaults to ranking all documents). The key flows in the
        // Authorization header only — never the body, never a log.
        let mut body = json!({
            "model": request.model,
            "query": request.query,
            "documents": request.documents,
        });
        if let Some(top_n) = request.top_n {
            body["top_n"] = json!(top_n);
        }

        let response = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {api_key}"))
            .json(&body)
            .send()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("cohere", e))?;

        if !response.status().is_success() {
            return Err(crate::client::error_from_response("cohere", response).await);
        }

        // Cohere v2 rerank response:
        // { id, results: [{ index, relevance_score }, ...],
        //   meta: { billed_units: { search_units } } }
        let result: serde_json::Value = response
            .json()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("cohere", e))?;

        let id = result
            .get("id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let raw_results = result
            .get("results")
            .and_then(|r| r.as_array())
            .cloned()
            .unwrap_or_default();

        let mut results = Vec::with_capacity(raw_results.len());
        for item in raw_results {
            let index = item.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
            let relevance_score = item
                .get("relevance_score")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            // Echo the document text back only when the caller asked for it
            // (return_documents). The result `index` maps back to the input
            // `documents` order, so we read straight from the request.
            let document = if request.return_documents {
                request
                    .documents
                    .get(index as usize)
                    .map(|text| RerankDocument { text: text.clone() })
            } else {
                None
            };
            results.push(RerankResult {
                index,
                relevance_score,
                document,
            });
        }

        let search_units = result
            .get("meta")
            .and_then(|m| m.get("billed_units"))
            .and_then(|b| b.get("search_units"))
            .and_then(|t| t.as_u64())
            .unwrap_or(0) as u32;

        Ok(RerankResponse {
            id,
            model: request.model,
            results,
            usage: RerankUsage {
                search_units,
                total_tokens: search_units,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RetryClass;
    use routeplane_types::Message;
    use wiremock::matchers::{body_partial_json, header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn req(model: &str) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: model.into(),
            messages: vec![Message {
                role: "user".into(),
                content: "hi".into(),
                name: None,
                cache_control: None,
                tool_calls: None,
                tool_call_id: None,
                refusal: None,
                reasoning_content: None,
            }],
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
    fn name_is_cohere() {
        let p = CohereProvider::new();
        assert_eq!(p.name(), "cohere");
    }

    #[tokio::test]
    async fn buffered_call_hits_cohere_v2_chat() {
        let server = MockServer::start().await;
        let resp = serde_json::json!({
            "id": "cohere-1",
            "message": {"role": "assistant", "content": "Hello from Cohere"},
            "finish_reason": "COMPLETE",
            "usage": {"tokens": {"input_tokens": 3, "output_tokens": 4}}
        });
        Mock::given(method("POST"))
            .and(path("/v2/chat"))
            .and(header("authorization", "Bearer sk-cohere"))
            .and(body_partial_json(
                serde_json::json!({ "model": "command-r" }),
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(resp))
            .mount(&server)
            .await;

        let p = CohereProvider::with_base_url(server.uri());
        let out = p
            .chat_completion(req("command-r"), "sk-cohere".into())
            .await
            .expect("mock call succeeds");
        assert_eq!(
            out.choices[0].message.content.as_text(),
            "Hello from Cohere"
        );
        assert_eq!(out.choices[0].finish_reason, "stop"); // COMPLETE → stop
        assert_eq!(out.usage.prompt_tokens, 3);
        assert_eq!(out.usage.completion_tokens, 4);
    }

    #[tokio::test]
    async fn max_completion_tokens_maps_to_cap_and_never_leaks_to_cohere() {
        // Native-dialect contract: `max_completion_tokens` wins over `max_tokens`
        // for Cohere's single `max_tokens` cap, and the raw OpenAI key is NEVER
        // forwarded (Cohere rejects unknown OpenAI keys).
        let server = MockServer::start().await;
        let resp = serde_json::json!({
            "id": "cohere-2",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
            "finish_reason": "COMPLETE",
            "usage": {"tokens": {"input_tokens": 3, "output_tokens": 1}}
        });
        Mock::given(method("POST"))
            .and(path("/v2/chat"))
            .and(body_partial_json(serde_json::json!({ "max_tokens": 2048 })))
            .respond_with(ResponseTemplate::new(200).set_body_json(resp))
            .mount(&server)
            .await;

        let p = CohereProvider::with_base_url(server.uri());
        let mut r = req("command-r");
        r.max_tokens = Some(4096);
        r.max_completion_tokens = Some(2048); // takes precedence over max_tokens
        p.chat_completion(r, "sk-cohere".into())
            .await
            .expect("mock call succeeds");

        // Replay the recorded body: no raw OpenAI cap field — only the mapped
        // native cap.
        let received = &server.received_requests().await.unwrap()[0];
        let sent: serde_json::Value = serde_json::from_slice(&received.body).unwrap();
        assert!(sent.get("max_completion_tokens").is_none());
        assert_eq!(sent["max_tokens"], 2048);
    }

    #[tokio::test]
    async fn upstream_429_is_typed_rate_limited() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v2/chat"))
            .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
            .mount(&server)
            .await;

        let p = CohereProvider::with_base_url(server.uri());
        let err = p
            .chat_completion(req("command-r"), "sk-cohere".into())
            .await
            .expect_err("429 should be an Err");
        assert_eq!(err.retry_class(), RetryClass::Status(429));
    }

    #[tokio::test]
    async fn embeddings_batch_translates_cohere_v2_embed_shape() {
        let server = MockServer::start().await;
        let resp = serde_json::json!({
            "id": "emb-1",
            "embeddings": {"float": [[0.1, 0.2], [0.3, 0.4]]},
            "meta": {"billed_units": {"input_tokens": 10}}
        });
        // Assert the request body hits /v2/embed with input_type +
        // embedding_types set, the OpenAI input normalized to `texts`, and the
        // Bearer key in the header (never the body).
        Mock::given(method("POST"))
            .and(path("/v2/embed"))
            .and(header("authorization", "Bearer sk-cohere"))
            .and(body_partial_json(serde_json::json!({
                "model": "embed-multilingual-v3.0",
                "texts": ["hello", "world"],
                "input_type": "search_document",
                "embedding_types": ["float"]
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(resp))
            .mount(&server)
            .await;

        let p = CohereProvider::with_base_url(server.uri());
        let request = EmbeddingRequest {
            model: "embed-multilingual-v3.0".into(),
            input: routeplane_types::EmbeddingInput::Batch(vec!["hello".into(), "world".into()]),
            user: None,
            encoding_format: None,
            dimensions: None,
        };
        let out = p
            .embeddings(request, "sk-cohere".into())
            .await
            .expect("embeddings");
        assert_eq!(out.object, "list");
        assert_eq!(out.model, "embed-multilingual-v3.0");
        assert_eq!(out.data.len(), 2);
        // index is load-bearing — it maps back to the input order.
        assert_eq!(out.data[0].index, 0);
        assert_eq!(out.data[0].object, "embedding");
        assert_eq!(
            out.data[0].embedding.as_floats().unwrap().to_vec(),
            vec![0.1_f32, 0.2]
        );
        assert_eq!(out.data[1].index, 1);
        assert_eq!(
            out.data[1].embedding.as_floats().unwrap().to_vec(),
            vec![0.3_f32, 0.4]
        );
        // meta.billed_units.input_tokens → prompt_tokens == total_tokens.
        assert_eq!(out.usage.prompt_tokens, 10);
        assert_eq!(out.usage.total_tokens, 10);
    }

    #[tokio::test]
    async fn embeddings_single_string_yields_one_vector() {
        let server = MockServer::start().await;
        let resp = serde_json::json!({
            "id": "emb-2",
            "embeddings": {"float": [[0.5, 0.6, 0.7]]},
            "meta": {"billed_units": {"input_tokens": 3}}
        });
        Mock::given(method("POST"))
            .and(path("/v2/embed"))
            .and(body_partial_json(serde_json::json!({
                "model": "embed-v4.0",
                "texts": ["just one"],
                "input_type": "search_document",
                "embedding_types": ["float"]
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(resp))
            .mount(&server)
            .await;

        let p = CohereProvider::with_base_url(server.uri());
        let request = EmbeddingRequest {
            model: "embed-v4.0".into(),
            input: routeplane_types::EmbeddingInput::Single("just one".into()),
            user: None,
            encoding_format: None,
            dimensions: None,
        };
        let out = p
            .embeddings(request, "sk-cohere".into())
            .await
            .expect("embeddings");
        assert_eq!(out.data.len(), 1);
        assert_eq!(out.data[0].index, 0);
        assert_eq!(
            out.data[0].embedding.as_floats().unwrap().to_vec(),
            vec![0.5_f32, 0.6, 0.7]
        );
        assert_eq!(out.usage.total_tokens, 3);
    }

    #[tokio::test]
    async fn embeddings_missing_usage_maps_to_zero() {
        let server = MockServer::start().await;
        // No `meta` block → usage falls back to 0 (not a panic).
        let resp = serde_json::json!({
            "id": "emb-3",
            "embeddings": {"float": [[0.1]]}
        });
        Mock::given(method("POST"))
            .and(path("/v2/embed"))
            .respond_with(ResponseTemplate::new(200).set_body_json(resp))
            .mount(&server)
            .await;

        let p = CohereProvider::with_base_url(server.uri());
        let request = EmbeddingRequest {
            model: "embed-v4.0".into(),
            input: routeplane_types::EmbeddingInput::Single("x".into()),
            user: None,
            encoding_format: None,
            dimensions: None,
        };
        let out = p
            .embeddings(request, "sk-cohere".into())
            .await
            .expect("embeddings");
        assert_eq!(out.usage.prompt_tokens, 0);
        assert_eq!(out.usage.total_tokens, 0);
    }

    #[tokio::test]
    async fn embeddings_empty_input_is_clean_error_not_panic() {
        // An empty batch must be rejected as a typed 422 BadRequest *before* any
        // network call — never a panic, never a silent empty 200.
        let p = CohereProvider::with_base_url("http://127.0.0.1:1"); // never hit
        let request = EmbeddingRequest {
            model: "embed-v4.0".into(),
            input: routeplane_types::EmbeddingInput::Batch(vec![]),
            user: None,
            encoding_format: None,
            dimensions: None,
        };
        let err = p
            .embeddings(request, "sk-cohere".into())
            .await
            .expect_err("empty input should be an Err");
        assert_eq!(err.status(), Some(422));
        assert_eq!(err.retry_class(), RetryClass::Never);
        // The key must never leak into the error.
        assert!(!err.to_string().contains("sk-cohere"));
    }

    // --- vision passthrough (Cohere v2 OpenAI-shaped image_url parts) ---------

    #[test]
    fn text_only_message_serializes_as_bare_string_byte_identical() {
        // Cohere's v2 message content must stay a bare string for text-only
        // messages (byte-identical to the pre-vision wire).
        let creq = build_cohere_request(&req("command-r"), false);
        let v = serde_json::to_value(&creq).unwrap();
        assert_eq!(v["messages"][0]["content"], serde_json::json!("hi"));
        assert!(v["messages"][0]["content"].is_string());
        // No-tools request: tools / tool_choice keys MUST be absent, and the
        // message MUST NOT carry tool_calls / tool_call_id (byte-identical wire).
        assert!(v.get("tools").is_none());
        assert!(v.get("tool_choice").is_none());
        assert!(v["messages"][0].get("tool_calls").is_none());
        assert!(v["messages"][0].get("tool_call_id").is_none());
        // upstream fidelity fix: the new penalty/seed fields MUST stay omitted when unset
        // (Option + skip_serializing_if ⇒ byte-identical to the pre-fix wire).
        assert!(v.get("presence_penalty").is_none());
        assert!(v.get("frequency_penalty").is_none());
        assert!(v.get("seed").is_none());
    }

    #[test]
    fn request_threads_presence_frequency_penalty_and_seed() {
        // upstream fidelity fix: Cohere v2 chat supports presence_penalty / frequency_penalty / seed
        // at the top level; they must be forwarded (were silently dropped).
        let mut r = req("command-r");
        r.presence_penalty = Some(0.5);
        r.frequency_penalty = Some(0.25);
        r.seed = Some(7);
        let v = serde_json::to_value(build_cohere_request(&r, false)).unwrap();
        assert_eq!(v["presence_penalty"], 0.5);
        assert_eq!(v["frequency_penalty"], 0.25);
        assert_eq!(v["seed"], 7);
    }

    // --- tool / function calling (Cohere v2 /v2/chat) -------------------------

    fn weather_tool() -> Tool {
        Tool {
            tool_type: "function".into(),
            function: routeplane_types::FunctionDef {
                name: "get_weather".into(),
                description: Some("Get the weather for a city".into()),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                })),
            },
        }
    }

    #[test]
    fn request_forwards_tools_in_cohere_v2_shape() {
        // Cohere v2 `tools` == the canonical Tool shape; forward verbatim.
        let mut r = req("command-r-plus");
        r.tools = Some(vec![weather_tool()]);
        let creq = build_cohere_request(&r, false);
        let v = serde_json::to_value(&creq).unwrap();
        assert_eq!(v["tools"][0]["type"], "function");
        assert_eq!(v["tools"][0]["function"]["name"], "get_weather");
        assert_eq!(
            v["tools"][0]["function"]["description"],
            "Get the weather for a city"
        );
        assert_eq!(
            v["tools"][0]["function"]["parameters"]["properties"]["city"]["type"],
            "string"
        );
    }

    #[test]
    fn request_maps_tool_choice_common_cases() {
        // "required" → REQUIRED, "none" → NONE, "auto" → omitted (Cohere default),
        // a force-a-function object → omitted (no v2 equivalent; not sent invalid).
        let mut r = req("command-r");
        r.tool_choice = Some(serde_json::json!("required"));
        let v = serde_json::to_value(build_cohere_request(&r, false)).unwrap();
        assert_eq!(v["tool_choice"], "REQUIRED");

        r.tool_choice = Some(serde_json::json!("none"));
        let v = serde_json::to_value(build_cohere_request(&r, false)).unwrap();
        assert_eq!(v["tool_choice"], "NONE");

        r.tool_choice = Some(serde_json::json!("auto"));
        let v = serde_json::to_value(build_cohere_request(&r, false)).unwrap();
        assert!(v.get("tool_choice").is_none());

        r.tool_choice = Some(serde_json::json!({
            "type": "function", "function": {"name": "get_weather"}
        }));
        let v = serde_json::to_value(build_cohere_request(&r, false)).unwrap();
        assert!(v.get("tool_choice").is_none());
    }

    #[test]
    fn request_maps_response_format_to_cohere_json_object_mode() {
        // Previously dropped → structured-output requests got prose. Both
        // json_object and json_schema map to Cohere's JSON-object mode.
        let mut r = req("command-r");
        assert!(
            serde_json::to_value(build_cohere_request(&r, false))
                .unwrap()
                .get("response_format")
                .is_none(),
            "absent response_format stays omitted (byte-identical)"
        );

        r.response_format = Some(serde_json::json!({"type": "json_object"}));
        let v = serde_json::to_value(build_cohere_request(&r, false)).unwrap();
        assert_eq!(v["response_format"]["type"], "json_object");

        r.response_format = Some(serde_json::json!({
            "type": "json_schema",
            "json_schema": {"schema": {"type": "object"}}
        }));
        let v = serde_json::to_value(build_cohere_request(&r, false)).unwrap();
        assert_eq!(v["response_format"]["type"], "json_object");

        r.response_format = Some(serde_json::json!({"type": "weird"}));
        let v = serde_json::to_value(build_cohere_request(&r, false)).unwrap();
        assert!(v.get("response_format").is_none());
    }

    #[test]
    fn request_builds_assistant_tool_call_and_tool_result_turns() {
        // A multi-turn tool conversation: user → assistant(tool_calls) →
        // tool(result) must map to Cohere v2's message shapes.
        let mut r = req("command-r-plus");
        r.messages = vec![
            Message {
                role: "user".into(),
                content: "weather in paris?".into(),
                name: None,
                cache_control: None,
                tool_calls: None,
                tool_call_id: None,
                refusal: None,
                reasoning_content: None,
            },
            Message {
                role: "assistant".into(),
                content: "".into(), // pure tool-call turn (no text)
                name: None,
                cache_control: None,
                tool_calls: Some(vec![ToolCall {
                    id: "call_abc".into(),
                    tool_type: "function".into(),
                    function: routeplane_types::FunctionCall {
                        name: "get_weather".into(),
                        arguments: "{\"city\":\"paris\"}".into(),
                    },
                }]),
                tool_call_id: None,
                refusal: None,
                reasoning_content: None,
            },
            Message {
                role: "tool".into(),
                content: "18C and sunny".into(),
                name: None,
                cache_control: None,
                tool_calls: None,
                tool_call_id: Some("call_abc".into()),
                refusal: None,
                reasoning_content: None,
            },
        ];
        let v = serde_json::to_value(build_cohere_request(&r, false)).unwrap();

        // user turn unchanged (bare string content).
        assert_eq!(v["messages"][0]["role"], "user");
        assert_eq!(v["messages"][0]["content"], "weather in paris?");

        // assistant tool-CALL turn: tool_calls forwarded verbatim, content omitted.
        assert_eq!(v["messages"][1]["role"], "assistant");
        assert!(v["messages"][1].get("content").is_none());
        assert_eq!(v["messages"][1]["tool_calls"][0]["id"], "call_abc");
        assert_eq!(v["messages"][1]["tool_calls"][0]["type"], "function");
        assert_eq!(
            v["messages"][1]["tool_calls"][0]["function"]["name"],
            "get_weather"
        );
        assert_eq!(
            v["messages"][1]["tool_calls"][0]["function"]["arguments"],
            "{\"city\":\"paris\"}"
        );

        // tool-RESULT turn: role:"tool" + tool_call_id + string content.
        assert_eq!(v["messages"][2]["role"], "tool");
        assert_eq!(v["messages"][2]["tool_call_id"], "call_abc");
        assert_eq!(v["messages"][2]["content"], "18C and sunny");
    }

    #[tokio::test]
    async fn buffered_request_forwards_tools_to_cohere_v2() {
        let server = MockServer::start().await;
        let resp = serde_json::json!({
            "id": "cohere-tool-1",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
            "finish_reason": "COMPLETE",
            "usage": {"tokens": {"input_tokens": 5, "output_tokens": 1}}
        });
        Mock::given(method("POST"))
            .and(path("/v2/chat"))
            .and(body_partial_json(serde_json::json!({
                "tools": [{
                    "type": "function",
                    "function": {"name": "get_weather"}
                }],
                "tool_choice": "REQUIRED"
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(resp))
            .mount(&server)
            .await;

        let p = CohereProvider::with_base_url(server.uri());
        let mut r = req("command-r-plus");
        r.tools = Some(vec![weather_tool()]);
        r.tool_choice = Some(serde_json::json!("required"));
        let out = p
            .chat_completion(r, "sk-cohere".into())
            .await
            .expect("mock call succeeds");
        // v2 returns content as a text block array — flattened to plain text.
        assert_eq!(out.choices[0].message.content.as_text(), "ok");
    }

    #[tokio::test]
    async fn buffered_response_maps_tool_calls_and_finish_reason() {
        let server = MockServer::start().await;
        // Cohere v2 tool-call response: tool_plan + tool_calls, no text content,
        // finish_reason TOOL_CALL.
        let resp = serde_json::json!({
            "id": "cohere-tool-2",
            "message": {
                "role": "assistant",
                "tool_plan": "I will look up the weather.",
                "tool_calls": [{
                    "id": "get_weather_xyz",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": "{\"city\":\"paris\"}"
                    }
                }]
            },
            "finish_reason": "TOOL_CALL",
            "usage": {"tokens": {"input_tokens": 12, "output_tokens": 8}}
        });
        Mock::given(method("POST"))
            .and(path("/v2/chat"))
            .respond_with(ResponseTemplate::new(200).set_body_json(resp))
            .mount(&server)
            .await;

        let p = CohereProvider::with_base_url(server.uri());
        let mut r = req("command-r-plus");
        r.tools = Some(vec![weather_tool()]);
        let out = p
            .chat_completion(r, "sk-cohere".into())
            .await
            .expect("mock call succeeds");

        let msg = &out.choices[0].message;
        // tool_plan must NOT leak into content.
        assert_eq!(msg.content.as_text(), "");
        assert!(!msg.content.as_text().contains("look up"));
        let calls = msg.tool_calls.as_ref().expect("tool_calls present");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "get_weather_xyz");
        assert_eq!(calls[0].tool_type, "function");
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments, "{\"city\":\"paris\"}");
        // finish_reason mapped to OpenAI's "tool_calls".
        assert_eq!(out.choices[0].finish_reason, "tool_calls");
    }

    #[tokio::test]
    async fn buffered_response_without_tools_is_unchanged() {
        // A plain text response (no tool_calls) maps to content + finish "stop"
        // with tool_calls omitted — byte-identical to the pre-tool behaviour.
        let server = MockServer::start().await;
        let resp = serde_json::json!({
            "id": "cohere-plain",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
            "finish_reason": "COMPLETE",
            "usage": {"tokens": {"input_tokens": 3, "output_tokens": 2}}
        });
        Mock::given(method("POST"))
            .and(path("/v2/chat"))
            .respond_with(ResponseTemplate::new(200).set_body_json(resp))
            .mount(&server)
            .await;

        let p = CohereProvider::with_base_url(server.uri());
        let out = p
            .chat_completion(req("command-r"), "sk-cohere".into())
            .await
            .expect("mock call succeeds");
        assert_eq!(out.choices[0].message.content.as_text(), "hello");
        assert!(out.choices[0].message.tool_calls.is_none());
        assert_eq!(out.choices[0].finish_reason, "stop");
        // The serialized canonical message must NOT carry a tool_calls key.
        let v = serde_json::to_value(&out.choices[0].message).unwrap();
        assert!(v.get("tool_calls").is_none());
    }

    #[test]
    fn image_message_serializes_as_openai_content_parts() {
        use routeplane_types::{ContentPart, ImageUrlContent, MessageContent};

        let mut r = req("command-a-vision-07-2025");
        r.messages = vec![Message {
            role: "user".into(),
            content: MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "what is this".into(),
                    cache_control: None,
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrlContent {
                        url: "data:image/png;base64,iVBORw0KGgo=".into(),
                        detail: Some("high".into()),
                    },
                },
            ]),
            name: None,
            cache_control: None,
            tool_calls: None,
            tool_call_id: None,
            refusal: None,
            reasoning_content: None,
        }];
        let creq = build_cohere_request(&r, false);
        let v = serde_json::to_value(&creq).unwrap();
        let content = &v["messages"][0]["content"];
        assert!(content.is_array());
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "what is this");
        assert_eq!(content[1]["type"], "image_url");
        assert_eq!(
            content[1]["image_url"]["url"],
            "data:image/png;base64,iVBORw0KGgo="
        );
        assert_eq!(content[1]["image_url"]["detail"], "high");
    }

    #[tokio::test]
    async fn buffered_forwards_image_part_to_cohere_v2_wire() {
        let server = MockServer::start().await;
        let resp = serde_json::json!({
            "id": "cohere-2",
            "message": {"role": "assistant", "content": "a cat"},
            "finish_reason": "COMPLETE",
            "usage": {"tokens": {"input_tokens": 9, "output_tokens": 2}}
        });
        Mock::given(method("POST"))
            .and(path("/v2/chat"))
            .and(body_partial_json(serde_json::json!({
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}}
                    ]
                }]
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(resp))
            .mount(&server)
            .await;

        let p = CohereProvider::with_base_url(server.uri());
        let mut r = req("command-a-vision-07-2025");
        r.messages = vec![Message {
            role: "user".into(),
            content: routeplane_types::MessageContent::Parts(vec![
                routeplane_types::ContentPart::Text {
                    text: "what is this".into(),
                    cache_control: None,
                },
                routeplane_types::ContentPart::ImageUrl {
                    image_url: routeplane_types::ImageUrlContent {
                        url: "https://example.com/cat.jpg".into(),
                        detail: None,
                    },
                },
            ]),
            name: None,
            cache_control: None,
            tool_calls: None,
            tool_call_id: None,
            refusal: None,
            reasoning_content: None,
        }];
        let out = p
            .chat_completion(r, "sk-cohere".into())
            .await
            .expect("mock call succeeds");
        assert_eq!(out.choices[0].message.content.as_text(), "a cat");
    }

    /// A stream state with known id/model/created so a chunk's identity is
    /// asserted against fixed values (the real stream captures the id from
    /// `message-start` and stamps `created` once).
    fn stream_state() -> CohereStreamState {
        CohereStreamState {
            id: "resp-1".into(),
            model: "command-r".into(),
            created: 42,
            saw_tool_call: false,
        }
    }

    #[test]
    fn stream_event_parses_content_delta() {
        // The chunk's id/model/created now come from the per-stream STATE (not the
        // event) — upstream fidelity fix: no more synthetic `id:"cohere-stream"` / `created:0`.
        let mut state = stream_state();
        let line = r#"{"type":"content-delta","delta":{"message":{"content":{"text":"Hi"}}}}"#;
        let chunk = parse_cohere_stream_event(line, &mut state)
            .unwrap()
            .unwrap();
        assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("Hi"));
        assert_eq!(chunk.id, "resp-1");
        assert_eq!(chunk.model, "command-r");
        assert_eq!(chunk.created, 42);
    }

    #[test]
    fn stream_event_parses_message_end() {
        let mut state = stream_state();
        let line = r#"{"type":"message-end","delta":{"finish_reason":"COMPLETE"}}"#;
        let chunk = parse_cohere_stream_event(line, &mut state)
            .unwrap()
            .unwrap();
        assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("stop"));
        // Terminal chunk reuses the stream id (upstream fidelity fix), never the old "cohere-stream".
        assert_eq!(chunk.id, "resp-1");
        assert_ne!(chunk.id, "cohere-stream");
    }

    #[test]
    fn stream_event_captures_id_from_message_start() {
        // upstream fidelity fix: message-start carries the top-level response id; capture it into
        // state so every following chunk reuses it (emits no chunk itself).
        let mut state = stream_state();
        let line = r#"{"type":"message-start","id":"c2f3-msg-1","delta":{"message":{"role":"assistant"}}}"#;
        assert!(parse_cohere_stream_event(line, &mut state)
            .unwrap()
            .is_none());
        assert_eq!(state.id, "c2f3-msg-1");
        // A following content delta now carries the captured id.
        let line = r#"{"type":"content-delta","delta":{"message":{"content":{"text":"x"}}}}"#;
        let chunk = parse_cohere_stream_event(line, &mut state)
            .unwrap()
            .unwrap();
        assert_eq!(chunk.id, "c2f3-msg-1");
    }

    #[test]
    fn stream_message_end_reads_delta_nested_finish_and_usage() {
        // Real Cohere v2 `message-end` nests both finish_reason and usage under
        // `delta` (the same `delta.*` nesting the content path reads). The
        // terminal chunk must fire with the mapped finish_reason AND surface token
        // usage (previously dropped to None).
        let mut state = stream_state();
        let line = r#"{"type":"message-end",
            "delta":{"finish_reason":"MAX_TOKENS",
                     "usage":{"tokens":{"input_tokens":11,"output_tokens":7}}}}"#;
        let chunk = parse_cohere_stream_event(line, &mut state)
            .unwrap()
            .unwrap();
        assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("length"));
        let usage = chunk.usage.expect("terminal chunk carries usage");
        assert_eq!(usage.prompt_tokens, 11);
        assert_eq!(usage.completion_tokens, 7);
        assert_eq!(usage.total_tokens, 18);
    }

    #[test]
    fn stream_tool_call_start_then_delta_map_to_index_keyed_tool_call_chunks() {
        // upstream fidelity fix: Cohere v2 streams tool calls as `tool-call-start` (id/type/name +
        // empty args) then incremental `tool-call-delta` arguments fragments, keyed
        // by the event's top-level `index`. Both map to canonical ToolCallChunk
        // deltas (first fragment carries id/type/name; continuations carry only an
        // `arguments` fragment — exactly OpenAI).
        let mut state = stream_state();

        let start = r#"{"type":"tool-call-start","index":0,"delta":{"message":{"tool_calls":{
            "id":"get_weather_x","type":"function",
            "function":{"name":"get_weather","arguments":""}}}}}"#;
        let c = parse_cohere_stream_event(start, &mut state)
            .unwrap()
            .unwrap();
        let tc = c.choices[0].delta.tool_calls.as_ref().unwrap();
        assert_eq!(tc[0].index, 0);
        assert_eq!(tc[0].id.as_deref(), Some("get_weather_x"));
        assert_eq!(tc[0].tool_type.as_deref(), Some("function"));
        let f = tc[0].function.as_ref().unwrap();
        assert_eq!(f.name.as_deref(), Some("get_weather"));
        assert_eq!(f.arguments.as_deref(), Some(""));
        // Tool-call deltas carry no content.
        assert!(c.choices[0].delta.content.is_none());
        // saw_tool_call now latched — drives the terminal finish_reason.
        assert!(state.saw_tool_call);

        let delta = r#"{"type":"tool-call-delta","index":0,"delta":{"message":{"tool_calls":{
            "function":{"arguments":"{\"city\":\"paris\"}"}}}}}"#;
        let c = parse_cohere_stream_event(delta, &mut state)
            .unwrap()
            .unwrap();
        let tc = c.choices[0].delta.tool_calls.as_ref().unwrap();
        assert_eq!(tc[0].index, 0);
        // Continuation fragment: no id/type/name, only an arguments fragment.
        assert!(tc[0].id.is_none());
        assert!(tc[0].tool_type.is_none());
        let f = tc[0].function.as_ref().unwrap();
        assert!(f.name.is_none());
        assert_eq!(f.arguments.as_deref(), Some("{\"city\":\"paris\"}"));

        // Terminal chunk: TOOL_CALL after tool-call deltas → "tool_calls".
        let end = r#"{"type":"message-end","delta":{"finish_reason":"TOOL_CALL"}}"#;
        let c = parse_cohere_stream_event(end, &mut state).unwrap().unwrap();
        assert_eq!(c.choices[0].finish_reason.as_deref(), Some("tool_calls"));
    }

    #[test]
    fn stream_tool_call_end_emits_no_chunk() {
        // `tool-call-end` is structural — no canonical chunk (args already streamed).
        let mut state = stream_state();
        let line = r#"{"type":"tool-call-end","index":0}"#;
        assert!(parse_cohere_stream_event(line, &mut state)
            .unwrap()
            .is_none());
    }

    #[test]
    fn stream_message_end_tool_call_without_deltas_degrades_to_stop() {
        // upstream fidelity fix SAFETY INTERIM (adversarial): a `message-end` reporting `TOOL_CALL`
        // with ZERO preceding tool-call events must NEVER surface
        // finish_reason:"tool_calls" (that's the empty-tool_calls shape that breaks
        // agent loops). It degrades to "stop".
        let mut state = stream_state();
        assert!(!state.saw_tool_call);
        let line = r#"{"type":"message-end","delta":{"finish_reason":"TOOL_CALL"}}"#;
        let chunk = parse_cohere_stream_event(line, &mut state)
            .unwrap()
            .unwrap();
        assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("stop"));
    }

    #[tokio::test]
    async fn streaming_tool_call_end_to_end_shares_id_and_maps_finish() {
        // streaming tool-call and gen-param end-to-end: a real SSE tool-call stream through
        // `chat_completion_stream` yields tool-call deltas whose arguments
        // reassemble, a terminal finish_reason:"tool_calls", ONE id captured from
        // message-start shared across every chunk (never the literal
        // "cohere-stream"), and a non-zero `created`.
        use futures::StreamExt;
        let server = MockServer::start().await;
        let events = [
            serde_json::json!({"type":"message-start","id":"c2f3-msg-1",
                "delta":{"message":{"role":"assistant"}}}),
            serde_json::json!({"type":"tool-call-start","index":0,"delta":{"message":{"tool_calls":{
                "id":"get_weather_x","type":"function",
                "function":{"name":"get_weather","arguments":""}}}}}),
            serde_json::json!({"type":"tool-call-delta","index":0,"delta":{"message":{"tool_calls":{
                "function":{"arguments":"{\"city\":"}}}}}),
            serde_json::json!({"type":"tool-call-delta","index":0,"delta":{"message":{"tool_calls":{
                "function":{"arguments":"\"paris\"}"}}}}}),
            serde_json::json!({"type":"tool-call-end","index":0}),
            serde_json::json!({"type":"message-end","delta":{"finish_reason":"TOOL_CALL",
                "usage":{"tokens":{"input_tokens":10,"output_tokens":5}}}}),
        ];
        let sse: String = events.iter().map(|e| format!("data: {e}\n\n")).collect();
        Mock::given(method("POST"))
            .and(path("/v2/chat"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(sse.into_bytes(), "text/event-stream"),
            )
            .mount(&server)
            .await;

        let p = CohereProvider::with_base_url(server.uri());
        let mut r = req("command-r-plus");
        r.tools = Some(vec![weather_tool()]);
        let stream = p
            .chat_completion_stream(r, "sk-cohere".into())
            .await
            .expect("stream establishment succeeds");
        let chunks: Vec<_> = stream.map(|c| c.expect("chunk ok")).collect().await;

        // The tool name arrives on the first fragment.
        let name = chunks
            .iter()
            .find_map(|c| {
                c.choices[0]
                    .delta
                    .tool_calls
                    .as_ref()
                    .and_then(|t| t[0].function.as_ref())
                    .and_then(|f| f.name.clone())
            })
            .expect("a tool_call name delta");
        assert_eq!(name, "get_weather");
        // The incremental arguments fragments reassemble to the whole JSON.
        let args: String = chunks
            .iter()
            .filter_map(|c| c.choices[0].delta.tool_calls.as_ref())
            .flat_map(|t| t.iter())
            .filter_map(|tc| tc.function.as_ref().and_then(|f| f.arguments.clone()))
            .collect();
        assert_eq!(args, "{\"city\":\"paris\"}");
        // Terminal finish_reason is "tool_calls".
        assert_eq!(
            chunks.last().unwrap().choices[0].finish_reason.as_deref(),
            Some("tool_calls")
        );
        // upstream fidelity fix: one id from message-start, shared by every chunk; non-zero created.
        assert_eq!(chunks[0].id, "c2f3-msg-1");
        assert!(chunks.iter().all(|c| c.id == "c2f3-msg-1"));
        assert!(chunks.iter().all(|c| c.id != "cohere-stream"));
        assert!(chunks.iter().all(|c| c.created != 0));
    }

    #[tokio::test]
    async fn streaming_plain_text_never_yields_tool_calls_finish() {
        // upstream fidelity fix (b): a text-only stream (no tool-call events) must never surface
        // finish_reason:"tool_calls".
        use futures::StreamExt;
        let server = MockServer::start().await;
        let events = [
            serde_json::json!({"type":"message-start","id":"c2f3-msg-2",
                "delta":{"message":{"role":"assistant"}}}),
            serde_json::json!({"type":"content-delta",
                "delta":{"message":{"content":{"text":"Hello"}}}}),
            serde_json::json!({"type":"message-end","delta":{"finish_reason":"COMPLETE",
                "usage":{"tokens":{"input_tokens":2,"output_tokens":1}}}}),
        ];
        let sse: String = events.iter().map(|e| format!("data: {e}\n\n")).collect();
        Mock::given(method("POST"))
            .and(path("/v2/chat"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(sse.into_bytes(), "text/event-stream"),
            )
            .mount(&server)
            .await;

        let p = CohereProvider::with_base_url(server.uri());
        let stream = p
            .chat_completion_stream(req("command-r"), "sk-cohere".into())
            .await
            .expect("stream establishment succeeds");
        let chunks: Vec<_> = stream.map(|c| c.expect("chunk ok")).collect().await;
        assert_eq!(
            chunks.last().unwrap().choices[0].finish_reason.as_deref(),
            Some("stop")
        );
        assert!(chunks
            .iter()
            .all(|c| c.choices[0].finish_reason.as_deref() != Some("tool_calls")));
        // No chunk carried a tool_calls delta either.
        assert!(chunks
            .iter()
            .all(|c| c.choices[0].delta.tool_calls.is_none()));
    }

    // --- rerank (Cohere v2 /v2/rerank) ----------------------------------------

    fn rerank_req(model: &str, top_n: Option<u32>, return_documents: bool) -> RerankRequest {
        RerankRequest {
            model: model.into(),
            query: "what is the capital of france?".into(),
            documents: vec![
                "berlin is in germany".into(),
                "paris is the capital of france".into(),
                "tokyo is in japan".into(),
            ],
            top_n,
            return_documents,
        }
    }

    #[tokio::test]
    async fn rerank_hits_v2_rerank_and_maps_results_in_order() {
        let server = MockServer::start().await;
        // Cohere returns results ordered by relevance desc; index maps back to
        // the input documents array.
        let resp = serde_json::json!({
            "id": "rerank-1",
            "results": [
                {"index": 1, "relevance_score": 0.98},
                {"index": 0, "relevance_score": 0.10},
                {"index": 2, "relevance_score": 0.02}
            ],
            "meta": {"billed_units": {"search_units": 1}}
        });
        // Assert the request hits /v2/rerank with model+query+documents+top_n
        // and the Bearer key in the header (never the body).
        Mock::given(method("POST"))
            .and(path("/v2/rerank"))
            .and(header("authorization", "Bearer sk-cohere"))
            .and(body_partial_json(serde_json::json!({
                "model": "rerank-v3.5",
                "query": "what is the capital of france?",
                "documents": [
                    "berlin is in germany",
                    "paris is the capital of france",
                    "tokyo is in japan"
                ],
                "top_n": 2
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(resp))
            .mount(&server)
            .await;

        let p = CohereProvider::with_base_url(server.uri());
        let out = p
            .rerank(
                rerank_req("rerank-v3.5", Some(2), false),
                "sk-cohere".into(),
            )
            .await
            .expect("rerank");
        assert_eq!(out.id.as_deref(), Some("rerank-1"));
        assert_eq!(out.model, "rerank-v3.5");
        assert_eq!(out.results.len(), 3);
        // Order preserved (relevance desc); index maps back to input order.
        assert_eq!(out.results[0].index, 1);
        assert_eq!(out.results[0].relevance_score, 0.98);
        assert_eq!(out.results[1].index, 0);
        assert_eq!(out.results[2].index, 2);
        // return_documents was false → no echoed text.
        assert!(out.results[0].document.is_none());
        // meta.billed_units.search_units → usage.
        assert_eq!(out.usage.search_units, 1);
        assert_eq!(out.usage.total_tokens, 1);
    }

    #[tokio::test]
    async fn rerank_echoes_documents_when_requested() {
        let server = MockServer::start().await;
        let resp = serde_json::json!({
            "id": "rerank-2",
            "results": [
                {"index": 1, "relevance_score": 0.98}
            ],
            "meta": {"billed_units": {"search_units": 1}}
        });
        Mock::given(method("POST"))
            .and(path("/v2/rerank"))
            .respond_with(ResponseTemplate::new(200).set_body_json(resp))
            .mount(&server)
            .await;

        let p = CohereProvider::with_base_url(server.uri());
        let out = p
            .rerank(rerank_req("rerank-v3.5", None, true), "sk-cohere".into())
            .await
            .expect("rerank");
        // index 1 maps back to documents[1].
        assert_eq!(
            out.results[0].document.as_ref().map(|d| d.text.as_str()),
            Some("paris is the capital of france")
        );
    }

    #[tokio::test]
    async fn rerank_missing_meta_maps_usage_to_zero() {
        let server = MockServer::start().await;
        let resp = serde_json::json!({
            "id": "rerank-3",
            "results": [{"index": 0, "relevance_score": 0.5}]
        });
        Mock::given(method("POST"))
            .and(path("/v2/rerank"))
            .respond_with(ResponseTemplate::new(200).set_body_json(resp))
            .mount(&server)
            .await;

        let p = CohereProvider::with_base_url(server.uri());
        let out = p
            .rerank(rerank_req("rerank-v3.5", None, false), "sk-cohere".into())
            .await
            .expect("rerank");
        assert_eq!(out.usage.search_units, 0);
        assert_eq!(out.usage.total_tokens, 0);
    }

    #[tokio::test]
    async fn rerank_empty_documents_is_clean_error_not_panic() {
        // An empty document set must be rejected as a typed 422 BadRequest
        // *before* any network call — never a panic, never a silent empty 200.
        let p = CohereProvider::with_base_url("http://127.0.0.1:1"); // never hit
        let request = RerankRequest {
            model: "rerank-v3.5".into(),
            query: "q".into(),
            documents: vec![],
            top_n: None,
            return_documents: false,
        };
        let err = p
            .rerank(request, "sk-cohere".into())
            .await
            .expect_err("empty documents should be an Err");
        assert_eq!(err.status(), Some(422));
        assert_eq!(err.retry_class(), RetryClass::Never);
        // The key must never leak into the error.
        assert!(!err.to_string().contains("sk-cohere"));
    }

    #[tokio::test]
    async fn rerank_top_n_omitted_when_absent() {
        let server = MockServer::start().await;
        let resp = serde_json::json!({
            "id": "rerank-4",
            "results": [{"index": 0, "relevance_score": 0.5}],
            "meta": {"billed_units": {"search_units": 1}}
        });
        // When top_n is None it must NOT appear on the wire (Cohere then ranks
        // all documents).
        Mock::given(method("POST"))
            .and(path("/v2/rerank"))
            .and(body_partial_json(
                serde_json::json!({ "model": "rerank-v3.5" }),
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(resp))
            .mount(&server)
            .await;

        let p = CohereProvider::with_base_url(server.uri());
        let out = p
            .rerank(rerank_req("rerank-v3.5", None, false), "sk-cohere".into())
            .await
            .expect("rerank");
        assert_eq!(out.results.len(), 1);
    }

    #[tokio::test]
    async fn rerank_upstream_429_is_typed_rate_limited() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v2/rerank"))
            .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
            .mount(&server)
            .await;

        let p = CohereProvider::with_base_url(server.uri());
        let err = p
            .rerank(rerank_req("rerank-v3.5", None, false), "sk-cohere".into())
            .await
            .expect_err("429 should be an Err");
        assert_eq!(err.retry_class(), RetryClass::Status(429));
    }
}
