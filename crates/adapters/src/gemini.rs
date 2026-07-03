use crate::sse::SseLineBuffer;
use crate::{ChunkStream, Provider, ProviderError};
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use routeplane_types::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, Choice, ChunkChoice, Delta,
    EmbeddingData, EmbeddingRequest, EmbeddingResponse, EmbeddingUsage, FunctionCallChunk, Message,
    ToolCallChunk, Usage,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

pub struct GeminiProvider {
    client: Client,
    /// Base URL for the Generative Language API. Defaults to the public Google
    /// endpoint; overridable so wiremock-backed tests can point the adapter at
    /// a mock server (engineering-design §24) without touching the hot path.
    base_url: String,
}

const GEMINI_DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com";

impl GeminiProvider {
    pub fn new() -> Self {
        Self {
            client: crate::client::build_provider_client(),
            base_url: GEMINI_DEFAULT_BASE_URL.to_string(),
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

impl Default for GeminiProvider {
    fn default() -> Self {
        Self::new()
    }
}

// Field names below mirror Google's Gemini JSON wire format verbatim
// (camelCase). `systemInstruction` and `generationConfig` are top-level.
#[derive(Debug, Serialize)]
#[allow(non_snake_case)]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
    /// System prompt lifted out of `contents` (Task #4) — Gemini takes the
    /// system message as a separate top-level instruction, NOT as a "model"
    /// turn (the previous bug turned a system message into a model turn).
    #[serde(skip_serializing_if = "Option::is_none")]
    systemInstruction: Option<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generationConfig: Option<GeminiGenerationConfig>,
    /// Tool calling: Gemini takes a list of `{functionDeclarations:[...]}` tool
    /// objects. We emit a single tool object carrying all declarations. `None`
    /// ⇒ omitted ⇒ byte-identical to a non-tool request.
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<Value>>,
}

/// A Gemini content turn. `parts` are raw JSON values because a part can be
/// `{text}`, `{functionCall:{name,args}}`, or `{functionResponse:{name,response}}`
/// — a flat struct can't model all three. Outbound we build the right shape;
/// inbound we read parts as Values. Kept untagged-friendly (raw Value) so the
/// text-only wire is byte-identical to the pre-tool shape.
#[derive(Debug, Serialize, Deserialize)]
struct GeminiContent {
    role: String,
    parts: Vec<Value>,
}

/// Build a `{text: ...}` Gemini part (the common text case).
fn gemini_text_part(text: impl Into<String>) -> Value {
    serde_json::json!({ "text": text.into() })
}

/// Gemini's `generationConfig` — where max_tokens/temperature/top_p/stop go.
#[derive(Debug, Serialize, Default)]
#[allow(non_snake_case)]
struct GeminiGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    maxOutputTokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    topP: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stopSequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    candidateCount: Option<u32>,
    /// Gemini's JSON-mode toggle, mapped from the canonical OpenAI
    /// `response_format`. Either JSON shape (`{"type":"json_object"}` or
    /// `{"type":"json_schema",...}`) sets `"application/json"`. `None` ⇒ omitted.
    #[serde(skip_serializing_if = "Option::is_none")]
    responseMimeType: Option<String>,
    /// Gemini's response schema, mapped from a structured-outputs
    /// `response_format.json_schema.schema` when present. `None` ⇒ omitted.
    #[serde(skip_serializing_if = "Option::is_none")]
    responseSchema: Option<Value>,
}

impl GeminiGenerationConfig {
    /// Build from the canonical request; `None` when every field is empty so we
    /// don't emit an empty `generationConfig` object.
    fn from_request(request: &ChatCompletionRequest) -> Option<Self> {
        let (response_mime_type, response_schema) =
            map_response_format(request.response_format.as_ref());
        let cfg = GeminiGenerationConfig {
            maxOutputTokens: request.max_tokens,
            temperature: request.temperature,
            topP: request.top_p,
            stopSequences: request.stop.clone(),
            candidateCount: request.n,
            responseMimeType: response_mime_type,
            responseSchema: response_schema,
        };
        if cfg.maxOutputTokens.is_none()
            && cfg.temperature.is_none()
            && cfg.topP.is_none()
            && cfg.stopSequences.is_none()
            && cfg.candidateCount.is_none()
            && cfg.responseMimeType.is_none()
            && cfg.responseSchema.is_none()
        {
            None
        } else {
            Some(cfg)
        }
    }
}

/// Map the canonical OpenAI `response_format` to Gemini's
/// `(responseMimeType, responseSchema)`. Both `{"type":"json_object"}` and
/// `{"type":"json_schema","json_schema":{"schema":{...}}}` set the MIME type to
/// `application/json`; a `json_schema` additionally carries the schema. Any other
/// (or absent) value yields `(None, None)` so a non-JSON request stays
/// byte-identical (no `responseMimeType` key). OpenAI-only request fields like
/// `logit_bias`/`logprobs`/`seed`/`service_tier`/`reasoning_effort` are NOT
/// mapped (Gemini's native body has no equivalent) and so never leak.
fn map_response_format(rf: Option<&Value>) -> (Option<String>, Option<Value>) {
    let Some(rf) = rf else {
        return (None, None);
    };
    match rf.get("type").and_then(|t| t.as_str()) {
        Some("json_object") => (Some("application/json".to_string()), None),
        Some("json_schema") => {
            // OpenAI nests the schema under `json_schema.schema`.
            let schema = rf
                .get("json_schema")
                .and_then(|js| js.get("schema"))
                .cloned();
            (Some("application/json".to_string()), schema)
        }
        // Unknown / "text" / malformed ⇒ no JSON mode (byte-identical).
        _ => (None, None),
    }
}

// Field names below mirror Google's Gemini JSON wire format verbatim
// (camelCase). Kept as-is so serde deserializes without per-field renames;
// non_snake_case is silenced because these are external API field names.
#[derive(Debug, Deserialize)]
#[allow(non_snake_case)]
struct GeminiResponse {
    candidates: Vec<GeminiCandidate>,
    usageMetadata: Option<GeminiUsage>,
}

#[derive(Debug, Deserialize)]
#[allow(non_snake_case)]
struct GeminiCandidate {
    content: GeminiContent,
    finishReason: String,
}

#[derive(Debug, Deserialize)]
#[allow(non_snake_case)]
struct GeminiUsage {
    promptTokenCount: u32,
    candidatesTokenCount: u32,
    totalTokenCount: u32,
}

// --- Embeddings (batchEmbedContents) ----------------------------------------
// Gemini's embeddings dialect: POST .../{model}:batchEmbedContents?key=, with
// body `{ requests: [{ model, content: { parts: [{ text }] }, outputDimensionality? }] }`
// and response `{ embeddings: [{ values: [f32] }] }`. We ALWAYS use the batch
// verb (a single input is a one-element batch) so the response array order maps
// cleanly back to the OpenAI `data[].index` order. Gemini's embed API returns no
// token usage, so usage is reported as zero (documented fidelity gap, PRD-011).

#[derive(Debug, Serialize)]
struct GeminiEmbedBatchRequest {
    requests: Vec<GeminiEmbedRequest>,
}

#[derive(Debug, Serialize)]
#[allow(non_snake_case)]
struct GeminiEmbedRequest {
    model: String,
    content: GeminiEmbedContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    outputDimensionality: Option<u32>,
}

#[derive(Debug, Serialize)]
struct GeminiEmbedContent {
    parts: Vec<GeminiTextPart>,
}

/// A `{text}` part used by the embeddings dialect (which is always text-only),
/// kept as a typed struct since chat parts are now raw Values to support
/// functionCall/functionResponse shapes.
#[derive(Debug, Serialize)]
struct GeminiTextPart {
    text: String,
}

#[derive(Debug, Deserialize)]
struct GeminiEmbedBatchResponse {
    #[serde(default)]
    embeddings: Vec<GeminiEmbedding>,
}

#[derive(Debug, Deserialize)]
struct GeminiEmbedding {
    #[serde(default)]
    values: Vec<f32>,
}

#[async_trait]
impl Provider for GeminiProvider {
    fn name(&self) -> &'static str {
        "gemini"
    }

    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
        api_key: String,
    ) -> Result<ChatCompletionResponse, ProviderError> {
        let url = format!(
            "{}/v1beta/models/{}:generateContent?key={}",
            self.base_url, request.model, api_key
        );

        let gemini_req = build_gemini_request(&request);

        // The API key is in the URL query string for Gemini, so any error that
        // could carry the URL MUST be sanitized (Task #3d) — never propagate the
        // raw reqwest error with `?`.
        let response = self
            .client
            .post(url)
            .json(&gemini_req)
            .send()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("gemini", e))?;

        if !response.status().is_success() {
            return Err(crate::client::error_from_response("gemini", response).await);
        }

        let result = response
            .json::<GeminiResponse>()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("gemini", e))?;

        let candidate = result
            .candidates
            .first()
            .ok_or("No candidates in Gemini response")?;

        // Walk the candidate's parts: concatenate `text` parts; map each
        // `functionCall` part → a canonical tool_call. Gemini gives no per-call
        // id, so we synthesize a stable one from the index (callers correlate by
        // name + order, and we echo this id back when threading the tool result).
        let mut response_text = String::new();
        let mut tool_calls: Vec<routeplane_types::ToolCall> = Vec::new();
        for (i, part) in candidate.content.parts.iter().enumerate() {
            if let Some(t) = part.get("text").and_then(|t| t.as_str()) {
                response_text.push_str(t);
            } else if let Some(fc) = part.get("functionCall") {
                let name = fc
                    .get("name")
                    .and_then(|n| n.as_str())
                    .unwrap_or_default()
                    .to_string();
                // Gemini's `args` is a JSON OBJECT; OpenAI's `arguments` is a
                // JSON-encoded STRING — re-serialize. Never panic on the request
                // thread.
                let arguments = fc
                    .get("args")
                    .map(|a| serde_json::to_string(a).unwrap_or_else(|_| "{}".to_string()))
                    .unwrap_or_else(|| "{}".to_string());
                tool_calls.push(routeplane_types::ToolCall {
                    id: format!("call_{i}"),
                    tool_type: "function".to_string(),
                    function: routeplane_types::FunctionCall { name, arguments },
                });
            }
        }
        let has_tool_calls = !tool_calls.is_empty();
        let tool_calls = if has_tool_calls {
            Some(tool_calls)
        } else {
            None
        };
        // Gemini's finishReason for a tool call is still "STOP"; OpenAI expects
        // "tool_calls" when the turn produced tool calls. Otherwise normalize
        // Gemini's uppercase enum via map_gemini_finish ("STOP" → "stop",
        // "MAX_TOKENS" → "length", "SAFETY"/"RECITATION" → "content_filter") so the
        // buffered path emits the same OpenAI-canonical finish_reason as the
        // streaming path, instead of leaking the raw Gemini enum to clients.
        let finish_reason = if has_tool_calls {
            "tool_calls".to_string()
        } else {
            map_gemini_finish(&candidate.finishReason)
        };

        let usage = result.usageMetadata.unwrap_or(GeminiUsage {
            promptTokenCount: 0,
            candidatesTokenCount: 0,
            totalTokenCount: 0,
        });

        Ok(ChatCompletionResponse {
            id: "gemini-resp".to_string(), // Gemini doesn't provide a top-level ID in this format
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
                    tool_calls,
                    tool_call_id: None,
                },
                finish_reason,
                // Gemini's generateContent has no per-choice logprobs in this
                // dialect; absent ⇒ omitted (byte-identical to before).
                logprobs: None,
            }],
            usage: Usage {
                prompt_tokens: usage.promptTokenCount,
                completion_tokens: usage.candidatesTokenCount,
                total_tokens: usage.totalTokenCount,
                cached_tokens: None,
                cache_creation_tokens: None,
            },
            // OpenAI-only response metadata Gemini does not report.
            system_fingerprint: None,
            service_tier: None,
        })
    }

    async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
        api_key: String,
    ) -> Result<ChunkStream, ProviderError> {
        // `streamGenerateContent` with `alt=sse` returns a `text/event-stream`
        // of `data:` lines, each a partial `GenerateContentResponse`.
        let url = format!(
            "{}/v1beta/models/{}:streamGenerateContent?alt=sse&key={}",
            self.base_url, request.model, api_key
        );

        let model = request.model.clone();
        let gemini_req = build_gemini_request(&request);

        // Key is in the URL — sanitize any transport error (Task #3d).
        let resp = self
            .client
            .post(url)
            .json(&gemini_req)
            .send()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("gemini", e))?;

        // Establishment failure -> typed Err so the proxy can retry/fall back.
        if !resp.status().is_success() {
            return Err(crate::client::error_from_response("gemini", resp).await);
        }

        Ok(Box::pin(gemini_sse_to_chunks(resp.bytes_stream(), model)))
    }

    async fn embeddings(
        &self,
        request: EmbeddingRequest,
        api_key: String,
    ) -> Result<EmbeddingResponse, ProviderError> {
        let inputs = request.input.to_vec();
        let model_path = format!("models/{}", request.model);
        let requests: Vec<GeminiEmbedRequest> = inputs
            .iter()
            .map(|text| GeminiEmbedRequest {
                model: model_path.clone(),
                content: GeminiEmbedContent {
                    parts: vec![GeminiTextPart { text: text.clone() }],
                },
                outputDimensionality: request.dimensions,
            })
            .collect();
        let body = GeminiEmbedBatchRequest { requests };

        // Key rides the URL query string (as with generateContent) — any
        // transport error MUST be sanitized so it can never echo the key (#3d).
        let url = format!(
            "{}/v1beta/models/{}:batchEmbedContents?key={}",
            self.base_url, request.model, api_key
        );
        let resp = self
            .client
            .post(url)
            .json(&body)
            .send()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("gemini", e))?;

        if !resp.status().is_success() {
            return Err(crate::client::error_from_response("gemini", resp).await);
        }

        let parsed = resp
            .json::<GeminiEmbedBatchResponse>()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("gemini", e))?;

        // Map Gemini's `embeddings[]` back to OpenAI `data[]`, preserving order
        // as the canonical `index` (Gemini returns vectors in request order).
        let data: Vec<EmbeddingData> = parsed
            .embeddings
            .into_iter()
            .enumerate()
            .map(|(i, e)| EmbeddingData {
                object: "embedding".to_string(),
                index: i as u32,
                embedding: e.values.into(),
            })
            .collect();

        Ok(EmbeddingResponse {
            object: "list".to_string(),
            data,
            model: request.model,
            // Gemini's embed API returns no usageMetadata → zeroed (documented).
            usage: EmbeddingUsage {
                prompt_tokens: 0,
                total_tokens: 0,
            },
        })
    }
}

fn build_gemini_request(request: &ChatCompletionRequest) -> GeminiRequest {
    // Lift system messages to `systemInstruction`; map remaining roles to
    // Gemini's two-role model: user -> "user", assistant (and anything else)
    // -> "model". A system message is NO LONGER mis-encoded as a model turn
    // (Task #4 bug fix).
    let mut system_parts: Vec<String> = Vec::new();
    let mut contents: Vec<GeminiContent> = Vec::new();
    for m in &request.messages {
        if m.role == "system" {
            system_parts.push(m.content.as_text());
            continue;
        }
        if m.role == "tool" {
            // The tool-RESULT turn. Gemini carries it as a "user"-role content
            // with a `functionResponse` part. Gemini keys it by function NAME (no
            // call id), so we use the tool_call_id as a best-effort name fallback;
            // a richer multi-call mapping is a follow-on.
            let name = m.tool_call_id.clone().unwrap_or_default();
            // The result content is text; wrap it as `{result: <text>}` so the
            // model receives a structured response object.
            let part = json!({
                "functionResponse": {
                    "name": name,
                    "response": { "result": m.content.as_text() }
                }
            });
            contents.push(GeminiContent {
                role: "user".to_string(),
                parts: vec![part],
            });
            continue;
        }
        let role = if m.role == "user" { "user" } else { "model" };
        // The assistant tool-CALL turn → `functionCall` parts (plus any text).
        if m.role == "assistant" && m.tool_calls.is_some() {
            let mut parts: Vec<Value> = Vec::new();
            let text = m.content.as_text();
            if !text.is_empty() {
                parts.push(gemini_text_part(text));
            }
            if let Some(calls) = &m.tool_calls {
                for call in calls {
                    // OpenAI arguments is a JSON STRING; Gemini's `args` is an
                    // object — parse it (empty object on malformed JSON).
                    let args: Value = serde_json::from_str(&call.function.arguments)
                        .unwrap_or_else(|_| json!({}));
                    parts.push(json!({
                        "functionCall": { "name": call.function.name, "args": args }
                    }));
                }
            }
            contents.push(GeminiContent {
                role: role.to_string(),
                parts,
            });
            continue;
        }
        contents.push(GeminiContent {
            role: role.to_string(),
            parts: vec![gemini_text_part(m.content.as_text())],
        });
    }

    let system_instruction = if system_parts.is_empty() {
        None
    } else {
        Some(GeminiContent {
            role: "user".to_string(),
            parts: vec![gemini_text_part(system_parts.join("\n\n"))],
        })
    };

    GeminiRequest {
        contents,
        systemInstruction: system_instruction,
        generationConfig: GeminiGenerationConfig::from_request(request),
        tools: build_gemini_tools(request.tools.as_deref()),
    }
}

/// Translate canonical OpenAI tool definitions to Gemini's `tools` shape: a
/// single tool object `{functionDeclarations:[{name, description?, parameters?}]}`.
/// Gemini's `parameters` IS a JSON Schema (same as OpenAI's), so the schema maps
/// straight across. Returns `None` for an absent/empty list (byte-identical wire).
fn build_gemini_tools(tools: Option<&[routeplane_types::Tool]>) -> Option<Vec<Value>> {
    let tools = tools?;
    if tools.is_empty() {
        return None;
    }
    let declarations: Vec<Value> = tools
        .iter()
        .map(|t| {
            let mut obj = serde_json::Map::new();
            obj.insert("name".to_string(), json!(t.function.name));
            if let Some(desc) = &t.function.description {
                obj.insert("description".to_string(), json!(desc));
            }
            if let Some(params) = &t.function.parameters {
                obj.insert("parameters".to_string(), params.clone());
            }
            Value::Object(obj)
        })
        .collect();
    Some(vec![json!({ "functionDeclarations": declarations })])
}

/// Per-stream translation state for Gemini. Gemini gives no per-call id and its
/// `finishReason` stays `"STOP"` even for a tool call, so we track the running
/// tool-call index (for id synthesis + `tool_calls[].index`) and whether any
/// `functionCall` part streamed (to synthesize the `"tool_calls"` finish_reason).
#[derive(Default)]
struct GeminiStreamState {
    /// Next tool-call index / id-synthesis counter. Gemini supplies no id, so we
    /// mint `call_<n>` mirroring the buffered path's `format!("call_{i}")`.
    next_tool_index: u32,
    /// Whether any tool call streamed — drives the synthesized finish_reason.
    saw_tool_call: bool,
}

/// Translate ONE Gemini SSE `data:` payload (a partial `GenerateContentResponse`)
/// into zero-or-more canonical chunks, mutating `state`. Pure function for
/// unit-testability.
fn translate_gemini_chunk(
    payload: &str,
    created: u64,
    model: &str,
    state: &mut GeminiStreamState,
    out: &mut Vec<ChatCompletionChunk>,
) -> Result<(), ProviderError> {
    let v: serde_json::Value = serde_json::from_str(payload).map_err(|e| -> ProviderError {
        format!("Gemini stream parse error: {e}: {payload}").into()
    })?;

    let candidate = match v["candidates"].get(0) {
        Some(c) => c,
        None => return Ok(()), // usage-only or empty payload
    };

    // Walk all parts of this chunk: `text` parts stream as content (a normal text
    // stream has a single text part, so the emitted chunk is byte-identical to the
    // previous `parts[0]["text"]` behaviour); a `functionCall` part — which Gemini
    // delivers WHOLE (name + full args object) — becomes ONE complete tool_call
    // delta (id + type + name + arguments together), since there is nothing to
    // stream incrementally.
    if let Some(parts) = candidate["content"]["parts"].as_array() {
        for part in parts {
            if let Some(text) = part["text"].as_str() {
                if !text.is_empty() {
                    out.push(ChatCompletionChunk::content_delta(
                        "gemini-stream",
                        model,
                        created,
                        text,
                    ));
                }
            } else if let Some(fc) = part.get("functionCall") {
                let tool_index = state.next_tool_index;
                state.next_tool_index += 1;
                state.saw_tool_call = true;
                let name = fc
                    .get("name")
                    .and_then(|n| n.as_str())
                    .unwrap_or_default()
                    .to_string();
                // Gemini's `args` is a JSON OBJECT; OpenAI's `arguments` is a
                // JSON-encoded STRING — re-serialize. Never panic on the stream
                // task: fall back to "{}" on a serialization failure.
                let arguments = fc
                    .get("args")
                    .map(|a| serde_json::to_string(a).unwrap_or_else(|_| "{}".to_string()))
                    .unwrap_or_else(|| "{}".to_string());
                out.push(ChatCompletionChunk {
                    id: "gemini-stream".to_string(),
                    object: "chat.completion.chunk".to_string(),
                    created,
                    model: model.to_string(),
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: Delta {
                            role: None,
                            content: None,
                            tool_calls: Some(vec![ToolCallChunk {
                                index: tool_index,
                                // Synthesize a stable id (mirrors the buffered
                                // path's `call_<i>`), since Gemini provides none.
                                id: Some(format!("call_{tool_index}")),
                                tool_type: Some("function".to_string()),
                                function: Some(FunctionCallChunk {
                                    name: Some(name),
                                    arguments: Some(arguments),
                                }),
                            }]),
                        },
                        finish_reason: None,
                    }],
                    usage: None,
                });
            }
        }
    }

    if let Some(finish) = candidate["finishReason"].as_str() {
        let usage = v.get("usageMetadata").map(|u| Usage {
            prompt_tokens: u["promptTokenCount"].as_u64().unwrap_or(0) as u32,
            completion_tokens: u["candidatesTokenCount"].as_u64().unwrap_or(0) as u32,
            total_tokens: u["totalTokenCount"].as_u64().unwrap_or(0) as u32,
            cached_tokens: None,
            cache_creation_tokens: None,
        });
        // Gemini's finishReason stays "STOP" for a tool call; OpenAI expects
        // "tool_calls" when the turn produced tool calls.
        let finish_reason = if state.saw_tool_call {
            "tool_calls".to_string()
        } else {
            map_gemini_finish(finish)
        };
        out.push(ChatCompletionChunk {
            id: "gemini-stream".to_string(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model.to_string(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta::default(),
                finish_reason: Some(finish_reason),
            }],
            usage,
        });
    }

    Ok(())
}

fn map_gemini_finish(reason: &str) -> String {
    match reason {
        "STOP" => "stop".to_string(),
        "MAX_TOKENS" => "length".to_string(),
        "SAFETY" | "RECITATION" => "content_filter".to_string(),
        other => other.to_string(),
    }
}

/// Translate a Gemini `streamGenerateContent` SSE byte stream into canonical chunks.
fn gemini_sse_to_chunks(
    mut bytes: impl futures::Stream<Item = reqwest::Result<bytes::Bytes>> + Unpin + Send + 'static,
    model: String,
) -> impl futures::Stream<Item = Result<ChatCompletionChunk, ProviderError>> + Send + 'static {
    async_stream::stream! {
        let created = chrono::Utc::now().timestamp() as u64;
        let mut sse = SseLineBuffer::new();
        let mut state = GeminiStreamState::default();
        while let Some(item) = bytes.next().await {
            let chunk = match item {
                Ok(b) => b,
                Err(e) => { yield Err(format!("Gemini stream transport error: {e}").into()); break; }
            };
            sse.push(&chunk);
            while let Some(payload) = sse.next_payload() {
                let mut out = Vec::new();
                match translate_gemini_chunk(&payload, created, &model, &mut state, &mut out) {
                    Ok(()) => { for c in out { yield Ok(c); } }
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

    fn msg(role: &str, content: &str) -> routeplane_types::Message {
        routeplane_types::Message {
            role: role.into(),
            content: content.into(),
            name: None,
            cache_control: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    fn req(messages: Vec<routeplane_types::Message>) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "gemini-1.5-flash".into(),
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
    fn system_message_becomes_system_instruction_not_a_model_turn() {
        let g = build_gemini_request(&req(vec![msg("system", "be terse"), msg("user", "hello")]));
        let si = g.systemInstruction.expect("systemInstruction set");
        assert_eq!(si.parts[0]["text"], "be terse");
        assert_eq!(g.contents.len(), 1);
        assert_eq!(g.contents[0].role, "user");
        assert!(g.contents.iter().all(|c| c.parts[0]["text"] != "be terse"));
    }

    // --- tool / function calling (native Gemini translation) ------------------

    use routeplane_types::{FunctionDef, Tool};

    fn weather_tool() -> Tool {
        Tool {
            tool_type: "function".into(),
            function: FunctionDef {
                name: "get_weather".into(),
                description: Some("Get the weather".into()),
                parameters: Some(json!({
                    "type": "object",
                    "properties": {"location": {"type": "string"}}
                })),
            },
        }
    }

    #[test]
    fn tools_map_to_function_declarations() {
        let mut r = req(vec![msg("user", "weather?")]);
        r.tools = Some(vec![weather_tool()]);
        let g = build_gemini_request(&r);
        let v = serde_json::to_value(&g).unwrap();
        // Gemini: tools[0].functionDeclarations[0] carries name/description/parameters.
        let decl = &v["tools"][0]["functionDeclarations"][0];
        assert_eq!(decl["name"], "get_weather");
        assert_eq!(decl["description"], "Get the weather");
        assert_eq!(decl["parameters"]["type"], "object");
    }

    #[tokio::test]
    async fn response_function_call_maps_to_tool_calls() {
        use wiremock::matchers::{body_partial_json, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        // Gemini returns a functionCall part; finishReason stays STOP.
        let resp = serde_json::json!({
            "candidates": [{
                "content": {"role": "model", "parts": [
                    {"functionCall": {"name": "get_weather", "args": {"location": "SF"}}}
                ]},
                "finishReason": "STOP"
            }],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 4, "totalTokenCount": 9}
        });
        // Assert the outbound body carries functionDeclarations.
        Mock::given(method("POST"))
            .and(path("/v1beta/models/gemini-1.5-flash:generateContent"))
            .and(body_partial_json(serde_json::json!({
                "tools": [{"functionDeclarations": [{"name": "get_weather"}]}]
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(resp))
            .mount(&server)
            .await;

        let provider = GeminiProvider::with_base_url(server.uri());
        let mut r = req(vec![msg("user", "weather in SF?")]);
        r.tools = Some(vec![weather_tool()]);
        let out = provider
            .chat_completion(r, "gk-test".into())
            .await
            .expect("mock call succeeds");
        // functionCall → canonical tool_call (args OBJECT → arguments STRING);
        // finish_reason synthesized to "tool_calls".
        assert_eq!(out.choices[0].finish_reason, "tool_calls");
        let calls = out.choices[0].message.tool_calls.as_ref().unwrap();
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments, "{\"location\":\"SF\"}");
    }

    #[test]
    fn assistant_role_maps_to_model() {
        let g = build_gemini_request(&req(vec![
            msg("user", "hi"),
            msg("assistant", "hello there"),
        ]));
        assert_eq!(g.contents[0].role, "user");
        assert_eq!(g.contents[1].role, "model");
    }

    #[test]
    fn generation_config_threads_fields() {
        let mut r = req(vec![msg("user", "hi")]);
        r.max_tokens = Some(512);
        r.stop = Some(vec!["END".into()]);
        let g = build_gemini_request(&r);
        let cfg = g.generationConfig.expect("generationConfig set");
        assert_eq!(cfg.maxOutputTokens, Some(512));
        assert_eq!(cfg.stopSequences.as_deref(), Some(&["END".to_string()][..]));
    }

    #[test]
    fn no_generation_config_when_all_absent() {
        let g = build_gemini_request(&req(vec![msg("user", "hi")]));
        assert!(g.generationConfig.is_none());
    }

    #[test]
    fn response_format_json_object_maps_to_response_mime_type() {
        let mut r = req(vec![msg("user", "hi")]);
        r.response_format = Some(serde_json::json!({"type": "json_object"}));
        let g = build_gemini_request(&r);
        let cfg = g.generationConfig.expect("generationConfig set");
        assert_eq!(cfg.responseMimeType.as_deref(), Some("application/json"));
        assert!(cfg.responseSchema.is_none());
    }

    #[test]
    fn response_format_json_schema_maps_mime_and_schema() {
        let mut r = req(vec![msg("user", "hi")]);
        r.response_format = Some(serde_json::json!({
            "type": "json_schema",
            "json_schema": {"name": "p", "schema": {"type": "object"}}
        }));
        let g = build_gemini_request(&r);
        let cfg = g.generationConfig.expect("generationConfig set");
        assert_eq!(cfg.responseMimeType.as_deref(), Some("application/json"));
        assert_eq!(
            cfg.responseSchema,
            Some(serde_json::json!({"type": "object"}))
        );
    }

    #[test]
    fn openai_only_request_fields_never_leak_into_gemini_body() {
        // seed / logit_bias / logprobs / service_tier / reasoning_effort have no
        // Gemini equivalent and must not appear anywhere in the native body.
        let mut r = req(vec![msg("user", "hi")]);
        r.seed = Some(9);
        r.logprobs = Some(true);
        r.service_tier = Some("flex".into());
        r.reasoning_effort = Some("high".into());
        let g = build_gemini_request(&r);
        let v = serde_json::to_value(&g).unwrap();
        let flat = v.to_string();
        for leaked in [
            "seed",
            "logit_bias",
            "logprobs",
            "service_tier",
            "reasoning_effort",
        ] {
            assert!(
                !flat.contains(leaked),
                "{leaked} must not leak to Gemini body"
            );
        }
    }

    #[test]
    fn text_part_becomes_content_chunk() {
        let mut out = Vec::new();
        let mut state = GeminiStreamState::default();
        let payload = r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"Hi"}]}}]}"#;
        translate_gemini_chunk(payload, 1, "gemini-1.5-flash", &mut state, &mut out).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].choices[0].delta.content.as_deref(), Some("Hi"));
    }

    #[test]
    fn final_chunk_carries_finish_and_usage() {
        let mut out = Vec::new();
        let mut state = GeminiStreamState::default();
        let payload = r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"!"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":4,"candidatesTokenCount":3,"totalTokenCount":7}}"#;
        translate_gemini_chunk(payload, 1, "gemini-1.5-flash", &mut state, &mut out).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].choices[0].delta.content.as_deref(), Some("!"));
        assert_eq!(out[1].choices[0].finish_reason.as_deref(), Some("stop"));
        assert_eq!(out[1].usage.as_ref().unwrap().total_tokens, 7);
    }

    #[test]
    fn stream_function_call_part_becomes_tool_call_delta() {
        // A streamed functionCall part → one complete tool_call delta (id + type +
        // name + arguments together, since Gemini delivers the call whole). The
        // synthesized id mirrors the buffered path (`call_<n>`).
        let mut out = Vec::new();
        let mut state = GeminiStreamState::default();
        let payload = r#"{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"get_weather","args":{"location":"SF"}}}]}}]}"#;
        translate_gemini_chunk(payload, 1, "gemini-1.5-flash", &mut state, &mut out).unwrap();
        assert_eq!(out.len(), 1);
        let tc = out[0].choices[0].delta.tool_calls.as_ref().unwrap();
        assert_eq!(tc[0].index, 0);
        assert_eq!(tc[0].id.as_deref(), Some("call_0"));
        assert_eq!(tc[0].tool_type.as_deref(), Some("function"));
        let func = tc[0].function.as_ref().unwrap();
        assert_eq!(func.name.as_deref(), Some("get_weather"));
        assert_eq!(func.arguments.as_deref(), Some("{\"location\":\"SF\"}"));
        // No content on a tool-call delta.
        assert!(out[0].choices[0].delta.content.is_none());
    }

    #[test]
    fn stream_function_call_synthesizes_tool_calls_finish_reason() {
        // Gemini's finishReason stays STOP for a tool call; once a functionCall
        // streamed, the synthesized finish_reason must be "tool_calls".
        let mut out = Vec::new();
        let mut state = GeminiStreamState::default();
        let call = r#"{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"f","args":{}}}]}}]}"#;
        translate_gemini_chunk(call, 1, "gemini-1.5-flash", &mut state, &mut out).unwrap();
        let mut out2 = Vec::new();
        let fin = r#"{"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":4,"candidatesTokenCount":3,"totalTokenCount":7}}"#;
        translate_gemini_chunk(fin, 1, "gemini-1.5-flash", &mut state, &mut out2).unwrap();
        assert_eq!(
            out2[0].choices[0].finish_reason.as_deref(),
            Some("tool_calls")
        );
        assert_eq!(out2[0].usage.as_ref().unwrap().total_tokens, 7);
    }

    #[tokio::test]
    async fn translates_full_gemini_tool_call_stream() {
        let raw = concat!(
            "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"functionCall\":{\"name\":\"get_weather\",\"args\":{\"location\":\"SF\"}}}]}}]}\n\n",
            "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":5,\"candidatesTokenCount\":4,\"totalTokenCount\":9}}\n\n",
        );
        let byte_stream = stream::iter(vec![Ok(bytes::Bytes::from(raw.to_string()))]);
        let chunks: Vec<_> = gemini_sse_to_chunks(byte_stream, "gemini-1.5-flash".to_string())
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();
        assert_eq!(chunks.len(), 2);
        let tc = chunks[0].choices[0].delta.tool_calls.as_ref().unwrap();
        assert_eq!(tc[0].id.as_deref(), Some("call_0"));
        assert_eq!(
            tc[0].function.as_ref().unwrap().name.as_deref(),
            Some("get_weather")
        );
        assert_eq!(
            tc[0].function.as_ref().unwrap().arguments.as_deref(),
            Some("{\"location\":\"SF\"}")
        );
        assert_eq!(
            chunks[1].choices[0].finish_reason.as_deref(),
            Some("tool_calls")
        );
    }

    #[tokio::test]
    async fn translates_full_gemini_sse_stream() {
        let raw = concat!(
            "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"Hel\"}]}}]}\n\n",
            "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"lo\"}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":4,\"candidatesTokenCount\":2,\"totalTokenCount\":6}}\n\n",
        );
        let byte_stream = stream::iter(vec![Ok(bytes::Bytes::from(raw.to_string()))]);
        let chunks: Vec<_> = gemini_sse_to_chunks(byte_stream, "gemini-1.5-flash".to_string())
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].choices[0].delta.content.as_deref(), Some("Hel"));
        assert_eq!(chunks[1].choices[0].delta.content.as_deref(), Some("lo"));
        assert_eq!(chunks[2].choices[0].finish_reason.as_deref(), Some("stop"));
        assert_eq!(chunks[2].usage.as_ref().unwrap().total_tokens, 6);
    }
}
