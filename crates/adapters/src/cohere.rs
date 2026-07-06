use crate::sse::SseLineBuffer;
use crate::{ChunkStream, Provider, ProviderError};
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use routeplane_types::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, Choice, ChunkChoice,
    ContentPart, Delta, EmbeddingData, EmbeddingRequest, EmbeddingResponse, EmbeddingUsage,
    Message, MessageContent, RerankDocument, RerankRequest, RerankResponse, RerankResult,
    RerankUsage, Tool, ToolCall, Usage,
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
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        p: request.top_p,
        stop_sequences: request.stop.clone(),
        // Tools: Cohere v2's shape == the canonical Tool, forward verbatim. An
        // absent/empty list stays omitted so a non-tool request is byte-identical.
        tools: cohere_tools(request.tools.as_deref()),
        tool_choice: request
            .tool_choice
            .as_ref()
            .and_then(map_cohere_tool_choice),
        stream,
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

/// Parse a Cohere streaming event. Cohere's SSE format uses JSON lines with
/// `event` types. We care about `content-delta` (text) and `message-end`
/// (finish + usage). Returns `Ok(None)` for events we don't translate
/// (keepalive, etc.).
///
/// Tool-calling streaming decision: buffered tool calls are complete (above), but
/// STREAMING tool calls are a documented follow-on. Cohere v2 emits a distinct
/// `tool-call-start`/`tool-call-delta`/`tool-call-end` event sequence (separate
/// from `content-delta`); mapping those incrementally to canonical
/// `Delta.tool_calls` (index-keyed id/name first, then `arguments` fragments) is
/// the same per-index accumulation done for Anthropic/Gemini in iter-35 and is
/// tracked as the Cohere streaming-tool-calls follow-on. The buffered path —
/// which the trait's one-shot fallback also covers — handles tool calls today;
/// the `TOOL_CALL` finish reason is already mapped here. Until then a STREAMING
/// tool-call response surfaces text deltas + the mapped finish reason, never a
/// panic or corrupted chunk.
fn parse_cohere_stream_event(line: &str) -> Result<Option<ChatCompletionChunk>, ProviderError> {
    // Cohere emits `event: content-delta\ndata: {"text": "..."}\n\n` etc.
    // The SseLineBuffer strips the `data:` prefix, so `line` is the JSON body.
    let value: serde_json::Value = serde_json::from_str(line).map_err(|e| {
        ProviderError::translation(format!("Cohere stream parse error: {e}: {line}"))
    })?;

    // content-delta: extract the text delta.
    if let Some(delta) = value
        .get("delta")
        .and_then(|d| d.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|c| c.get("text"))
        .and_then(|t| t.as_str())
    {
        if !delta.is_empty() {
            return Ok(Some(ChatCompletionChunk::content_delta(
                value
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("cohere-stream"),
                value
                    .get("model")
                    .and_then(|v| v.as_str())
                    .unwrap_or("cohere"),
                value.get("timestamp").and_then(|v| v.as_u64()).unwrap_or(0),
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
        let mapped = match finish {
            "COMPLETE" | "STOP_SEQUENCE" => "stop",
            "MAX_TOKENS" => "length",
            "TOOL_CALL" => "tool_calls",
            other => other,
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
            id: value
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or("cohere-stream")
                .to_string(),
            object: "chat.completion.chunk".to_string(),
            created: value.get("timestamp").and_then(|v| v.as_u64()).unwrap_or(0),
            model: value
                .get("model")
                .and_then(|v| v.as_str())
                .unwrap_or("cohere")
                .to_string(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta::default(),
                finish_reason: Some(mapped.to_string()),
            }],
            usage,
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
            let mut stream = std::pin::pin!(stream);
            while let Some(chunk) = stream.next().await {
                let bytes = chunk?;
                buf.push(&bytes);
                while let Some(payload) = buf.next_payload() {
                    if let Some(parsed) = parse_cohere_stream_event(&payload)? {
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
            },
            Message {
                role: "tool".into(),
                content: "18C and sunny".into(),
                name: None,
                cache_control: None,
                tool_calls: None,
                tool_call_id: Some("call_abc".into()),
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
        }];
        let out = p
            .chat_completion(r, "sk-cohere".into())
            .await
            .expect("mock call succeeds");
        assert_eq!(out.choices[0].message.content.as_text(), "a cat");
    }

    #[test]
    fn stream_event_parses_content_delta() {
        let line =
            r#"{"id":"s1","delta":{"message":{"content":{"text":"Hi"}}},"model":"command-r"}"#;
        let chunk = parse_cohere_stream_event(line).unwrap().unwrap();
        assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("Hi"));
    }

    #[test]
    fn stream_event_parses_message_end() {
        let line = r#"{"id":"s1","finish_reason":"COMPLETE","model":"command-r"}"#;
        let chunk = parse_cohere_stream_event(line).unwrap().unwrap();
        assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn stream_message_end_reads_delta_nested_finish_and_usage() {
        // Real Cohere v2 `message-end` nests both finish_reason and usage under
        // `delta` (the same `delta.*` nesting the content path reads). The
        // terminal chunk must fire with the mapped finish_reason AND surface token
        // usage (previously dropped to None).
        let line = r#"{"type":"message-end","id":"s2","model":"command-r",
            "delta":{"finish_reason":"MAX_TOKENS",
                     "usage":{"tokens":{"input_tokens":11,"output_tokens":7}}}}"#;
        let chunk = parse_cohere_stream_event(line).unwrap().unwrap();
        assert_eq!(chunk.choices[0].finish_reason.as_deref(), Some("length"));
        let usage = chunk.usage.expect("terminal chunk carries usage");
        assert_eq!(usage.prompt_tokens, 11);
        assert_eq!(usage.completion_tokens, 7);
        assert_eq!(usage.total_tokens, 18);
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
