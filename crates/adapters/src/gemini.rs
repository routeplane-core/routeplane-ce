use crate::sse::SseLineBuffer;
use crate::vision::parse_data_url;
use crate::{ChunkStream, Provider, ProviderError};
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use routeplane_types::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, Choice, ChunkChoice,
    ContentPart, Delta, EmbeddingData, EmbeddingRequest, EmbeddingResponse, EmbeddingUsage,
    FunctionCallChunk, Message, MessageContent, ToolCallChunk, Usage,
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
    // Gemini v1beta generationConfig DOES support these (a prior comment wrongly
    // claimed otherwise) — mapped from the canonical request so they are no longer
    // silently dropped. All Option + skip_serializing_if ⇒ a request that omits
    // them serializes byte-identically.
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presencePenalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequencyPenalty: Option<f32>,
    /// Gemini's boolean log-probs toggle, from the canonical `logprobs`.
    #[serde(skip_serializing_if = "Option::is_none")]
    responseLogprobs: Option<bool>,
    /// Number of top log-probs to return, from the canonical `top_logprobs`
    /// (Gemini honours it only when `responseLogprobs` is true).
    #[serde(skip_serializing_if = "Option::is_none")]
    logprobs: Option<u32>,
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
            // `max_completion_tokens` (the o-series replacement) takes precedence
            // over `max_tokens` — Gemini has one cap, so both map into it.
            maxOutputTokens: request.max_completion_tokens.or(request.max_tokens),
            temperature: request.temperature,
            topP: request.top_p,
            stopSequences: request.stop.clone(),
            candidateCount: request.n,
            seed: request.seed,
            presencePenalty: request.presence_penalty,
            frequencyPenalty: request.frequency_penalty,
            responseLogprobs: request.logprobs,
            logprobs: request.top_logprobs,
            responseMimeType: response_mime_type,
            responseSchema: response_schema,
        };
        if cfg.maxOutputTokens.is_none()
            && cfg.temperature.is_none()
            && cfg.topP.is_none()
            && cfg.stopSequences.is_none()
            && cfg.candidateCount.is_none()
            && cfg.seed.is_none()
            && cfg.presencePenalty.is_none()
            && cfg.frequencyPenalty.is_none()
            && cfg.responseLogprobs.is_none()
            && cfg.logprobs.is_none()
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
    /// Context-cache read tokens (Gemini context caching). Absent on responses
    /// that didn't hit a cached prefix ⇒ `None`. Lifted to `Usage.cached_tokens`.
    #[serde(default)]
    cachedContentTokenCount: Option<u32>,
    /// Gemini 2.5 thinking-model reasoning tokens. Gemini reports these SEPARATELY
    /// from `candidatesTokenCount` (which EXCLUDES them) but INCLUDES them in
    /// `totalTokenCount`. Absent on non-thinking responses ⇒ `None`. Folded into
    /// `completion_tokens` (OpenAI counts reasoning as completion) so the
    /// `total == prompt + completion` invariant holds and output-token cost is
    /// attributed correctly.
    #[serde(default)]
    thoughtsTokenCount: Option<u32>,
}

/// A unique response id per Gemini call (Gemini's generateContent gives no
/// top-level id, and the old hardcoded "gemini-resp"/"gemini-stream" broke
/// client dedupe/correlation keyed on the completion id). Monotonic within the
/// process: a high-resolution timestamp plus an atomic sequence so two calls in
/// the same instant still differ. A stream reuses ONE id across all its chunks.
fn gemini_response_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let ts = chrono::Utc::now()
        .timestamp_nanos_opt()
        .unwrap_or(0)
        .unsigned_abs();
    format!("gemini-{ts:x}{seq:x}")
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

        let gemini_req = build_gemini_request(&request)?;

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

        if result.candidates.is_empty() {
            return Err("No candidates in Gemini response".into());
        }

        // Fan ALL candidates into choices[] (index = candidate order). Previously
        // only candidates[0] was surfaced while `candidateCount = n` was billed and
        // `candidatesTokenCount` sums output across EVERY candidate — the client
        // paid for N and received 1. Each candidate's parts are walked the same
        // way: concatenate `text`; map each `functionCall` → a canonical tool_call
        // (Gemini gives no per-call id, so synthesize a stable `call_<i>`).
        let choices: Vec<Choice> = result
            .candidates
            .iter()
            .enumerate()
            .map(|(ci, candidate)| {
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
                        // Gemini's `args` is a JSON OBJECT; OpenAI's `arguments` is
                        // a JSON-encoded STRING — re-serialize. Never panic.
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
                // Gemini's finishReason stays "STOP" for a tool call; OpenAI
                // expects "tool_calls". Otherwise normalize the uppercase enum via
                // map_gemini_finish so buffered == streaming canonical form.
                let finish_reason = if has_tool_calls {
                    "tool_calls".to_string()
                } else {
                    map_gemini_finish(&candidate.finishReason)
                };
                Choice {
                    index: ci as u32,
                    message: Message {
                        role: "assistant".to_string(),
                        content: response_text.into(),
                        name: None,
                        cache_control: None,
                        tool_calls: if has_tool_calls {
                            Some(tool_calls)
                        } else {
                            None
                        },
                        tool_call_id: None,
                        refusal: None,
                        reasoning_content: None,
                    },
                    finish_reason,
                    // Gemini's generateContent has no per-choice logprobs here.
                    logprobs: None,
                }
            })
            .collect();

        let usage = result.usageMetadata.unwrap_or(GeminiUsage {
            promptTokenCount: 0,
            candidatesTokenCount: 0,
            totalTokenCount: 0,
            cachedContentTokenCount: None,
            thoughtsTokenCount: None,
        });

        Ok(ChatCompletionResponse {
            id: gemini_response_id(),
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp() as u64,
            model: request.model,
            choices,
            usage: Usage {
                prompt_tokens: usage.promptTokenCount,
                // Fold thinking tokens into completion (OpenAI counts reasoning as
                // completion; Gemini excludes them from candidatesTokenCount). None
                // on non-thinking responses ⇒ byte-identical to the old mapping.
                completion_tokens: usage.candidatesTokenCount
                    + usage.thoughtsTokenCount.unwrap_or(0),
                total_tokens: usage.totalTokenCount,
                // Gemini context-cache read tokens (upstream fidelity fix) — was silently dropped.
                cached_tokens: usage.cachedContentTokenCount,
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

        // Streaming surfaces only the first candidate, so don't request (and get
        // billed for) N: clamp candidateCount to 1 here. The buffered path fans
        // ALL candidates out instead (upstream fidelity fix).
        let mut request = request;
        if request.n.is_some_and(|n| n > 1) {
            request.n = Some(1);
        }

        let model = request.model.clone();
        let gemini_req = build_gemini_request(&request)?;
        // One unique id shared across every chunk of THIS stream (upstream fidelity fix).
        let stream_id = gemini_response_id();

        // Key is in the URL — sanitize any transport error (Task #3d).
        let resp = crate::client::streaming_client()
            .post(url)
            .json(&gemini_req)
            .send()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("gemini", e))?;

        // Establishment failure -> typed Err so the proxy can retry/fall back.
        if !resp.status().is_success() {
            return Err(crate::client::error_from_response("gemini", resp).await);
        }

        Ok(Box::pin(gemini_sse_to_chunks(
            resp.bytes_stream(),
            model,
            stream_id,
        )))
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

/// Build the Gemini `contents` parts for a message, preserving IMAGE parts as
/// Gemini `inlineData` (was: every part stringified via `as_text()`, silently
/// dropping images so a vision request got a confidently-wrong text-only answer
/// — upstream fidelity fix). Data-URL images decode to `{inlineData:{mimeType,data}}`; an http(s)
/// image URL cannot be forwarded as inline bytes and Gemini's `fileData` needs a
/// GCS/Files-API URI, so we FAIL LOUD with a typed 422 rather than drop it
/// (PRD-011 FR-10). A plain-text message stays byte-identical (one text part).
fn gemini_content_parts(content: &MessageContent) -> Result<Vec<Value>, ProviderError> {
    match content {
        MessageContent::Text(t) => Ok(vec![gemini_text_part(t.clone())]),
        MessageContent::Parts(parts) => {
            let mut out: Vec<Value> = Vec::with_capacity(parts.len());
            for p in parts {
                match p {
                    ContentPart::Text { text, .. } => out.push(gemini_text_part(text.clone())),
                    ContentPart::ImageUrl { image_url } => {
                        if let Some(data) = parse_data_url(&image_url.url) {
                            out.push(json!({
                                "inlineData": {
                                    "mimeType": data.media_type,
                                    "data": data.base64_payload,
                                }
                            }));
                        } else {
                            return Err(ProviderError::BadRequest {
                                provider: "gemini".to_string(),
                                status: 422,
                                body: "gemini requires inline (data:) image URLs; a plain http(s) \
                                       image URL cannot be forwarded as inline bytes"
                                    .to_string(),
                            });
                        }
                    }
                }
            }
            Ok(out)
        }
    }
}

fn build_gemini_request(request: &ChatCompletionRequest) -> Result<GeminiRequest, ProviderError> {
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
            parts: gemini_content_parts(&m.content)?,
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

    Ok(GeminiRequest {
        contents,
        systemInstruction: system_instruction,
        generationConfig: GeminiGenerationConfig::from_request(request),
        tools: build_gemini_tools(request.tools.as_deref()),
    })
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
    id: &str,
    state: &mut GeminiStreamState,
    out: &mut Vec<ChatCompletionChunk>,
) -> Result<(), ProviderError> {
    let v: serde_json::Value = serde_json::from_str(payload).map_err(|e| -> ProviderError {
        format!("Gemini stream parse error: {e}: {payload}").into()
    })?;

    // A mid-stream error frame (`data: {"error":{"code":…,"status":…,"message":…}}`)
    // carries no `candidates`, so without this guard it fell into the
    // `None => Ok(())` arm below and was SWALLOWED — the graceful socket close then
    // made the proxy emit `data: [DONE]` and report a clean success for a
    // provider-failed, truncated stream. Surface it as an `Err` so the proxy
    // terminates WITHOUT `[DONE]`. Only the stable status/code is carried (never
    // the free-form message).
    if let Some(err) = v.get("error").filter(|e| e.is_object()) {
        let status = err
            .get("status")
            .and_then(|s| s.as_str())
            .map(str::to_string)
            .or_else(|| {
                err.get("code")
                    .and_then(|c| c.as_u64())
                    .map(|c| c.to_string())
            })
            .unwrap_or_else(|| "error".to_string());
        return Err(format!("gemini stream error: {status}").into());
    }

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
                    out.push(ChatCompletionChunk::content_delta(id, model, created, text));
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
                    id: id.to_string(),
                    object: "chat.completion.chunk".to_string(),
                    created,
                    model: model.to_string(),
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: Delta {
                            role: None,
                            content: None,
                            refusal: None,
                            reasoning_content: None,
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
                        logprobs: None,
                    }],
                    usage: None,
                    system_fingerprint: None,
                    service_tier: None,
                });
            }
        }
    }

    if let Some(finish) = candidate["finishReason"].as_str() {
        let usage = v.get("usageMetadata").map(|u| Usage {
            prompt_tokens: u["promptTokenCount"].as_u64().unwrap_or(0) as u32,
            // Fold thinking tokens into completion (see the buffered path). Absent
            // on non-thinking responses ⇒ byte-identical to the old mapping.
            completion_tokens: (u["candidatesTokenCount"].as_u64().unwrap_or(0)
                + u["thoughtsTokenCount"].as_u64().unwrap_or(0))
                as u32,
            total_tokens: u["totalTokenCount"].as_u64().unwrap_or(0) as u32,
            // Gemini context-cache read tokens (upstream fidelity fix) — was silently dropped.
            cached_tokens: u["cachedContentTokenCount"].as_u64().map(|c| c as u32),
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
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model.to_string(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta::default(),
                finish_reason: Some(finish_reason),
                logprobs: None,
            }],
            usage,
            system_fingerprint: None,
            service_tier: None,
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
    id: String,
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
                match translate_gemini_chunk(&payload, created, &model, &id, &mut state, &mut out) {
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
            refusal: None,
            reasoning_content: None,
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
        let g = build_gemini_request(&req(vec![msg("system", "be terse"), msg("user", "hello")]))
            .unwrap();
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
        let g = build_gemini_request(&r).unwrap();
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

    #[tokio::test]
    async fn max_completion_tokens_maps_to_cap_and_never_leaks_to_gemini() {
        // Native-dialect contract: `max_completion_tokens` wins over `max_tokens`
        // for Gemini's `generationConfig.maxOutputTokens`, and the raw OpenAI
        // key is NEVER forwarded (Gemini rejects unknown keys).
        use wiremock::matchers::{body_partial_json, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        let resp = serde_json::json!({
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": "hello"}]},
                "finishReason": "STOP"
            }],
            "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 1, "totalTokenCount": 4}
        });
        Mock::given(method("POST"))
            .and(path("/v1beta/models/gemini-1.5-flash:generateContent"))
            .and(body_partial_json(serde_json::json!({
                "generationConfig": {"maxOutputTokens": 2048}
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(resp))
            .mount(&server)
            .await;

        let provider = GeminiProvider::with_base_url(server.uri());
        let mut r = req(vec![msg("user", "hi")]);
        r.max_tokens = Some(4096);
        r.max_completion_tokens = Some(2048); // takes precedence over max_tokens
        provider
            .chat_completion(r, "gk-test".into())
            .await
            .expect("mock call succeeds");

        // Replay the recorded body: no raw OpenAI cap field — only the mapped
        // native cap.
        let received = &server.received_requests().await.unwrap()[0];
        let sent: serde_json::Value = serde_json::from_slice(&received.body).unwrap();
        assert!(sent.get("max_completion_tokens").is_none());
        assert!(sent.get("max_tokens").is_none());
    }

    #[test]
    fn assistant_role_maps_to_model() {
        let g = build_gemini_request(&req(vec![
            msg("user", "hi"),
            msg("assistant", "hello there"),
        ]))
        .unwrap();
        assert_eq!(g.contents[0].role, "user");
        assert_eq!(g.contents[1].role, "model");
    }

    #[test]
    fn generation_config_threads_fields() {
        let mut r = req(vec![msg("user", "hi")]);
        r.max_tokens = Some(512);
        r.stop = Some(vec!["END".into()]);
        let g = build_gemini_request(&r).unwrap();
        let cfg = g.generationConfig.expect("generationConfig set");
        assert_eq!(cfg.maxOutputTokens, Some(512));
        assert_eq!(cfg.stopSequences.as_deref(), Some(&["END".to_string()][..]));
    }

    #[test]
    fn no_generation_config_when_all_absent() {
        let g = build_gemini_request(&req(vec![msg("user", "hi")])).unwrap();
        assert!(g.generationConfig.is_none());
    }

    #[test]
    fn response_format_json_object_maps_to_response_mime_type() {
        let mut r = req(vec![msg("user", "hi")]);
        r.response_format = Some(serde_json::json!({"type": "json_object"}));
        let g = build_gemini_request(&r).unwrap();
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
        let g = build_gemini_request(&r).unwrap();
        let cfg = g.generationConfig.expect("generationConfig set");
        assert_eq!(cfg.responseMimeType.as_deref(), Some("application/json"));
        assert_eq!(
            cfg.responseSchema,
            Some(serde_json::json!({"type": "object"}))
        );
    }

    #[test]
    fn gemini_supported_gen_params_map_and_unsupported_never_leak() {
        // upstream fidelity fix: seed / presence_penalty / frequency_penalty / logprobs / top_logprobs
        // DO have v1beta generationConfig equivalents and must map (a prior comment
        // wrongly claimed otherwise). logit_bias / service_tier / reasoning_effort
        // have no Gemini equivalent and must NOT appear in the native body.
        let mut r = req(vec![msg("user", "hi")]);
        r.seed = Some(9);
        r.presence_penalty = Some(0.5);
        r.frequency_penalty = Some(0.25);
        r.logprobs = Some(true);
        r.top_logprobs = Some(3);
        r.service_tier = Some("flex".into());
        r.reasoning_effort = Some("high".into());
        let g = build_gemini_request(&r).unwrap();
        let flat = serde_json::to_value(&g).unwrap().to_string();
        let cfg = g.generationConfig.as_ref().expect("generationConfig set");
        assert_eq!(cfg.seed, Some(9));
        assert_eq!(cfg.presencePenalty, Some(0.5));
        assert_eq!(cfg.frequencyPenalty, Some(0.25));
        assert_eq!(cfg.responseLogprobs, Some(true));
        assert_eq!(cfg.logprobs, Some(3));

        for leaked in ["logit_bias", "service_tier", "reasoning_effort"] {
            assert!(
                !flat.contains(leaked),
                "{leaked} has no Gemini equivalent and must not leak"
            );
        }
    }

    #[test]
    fn text_part_becomes_content_chunk() {
        let mut out = Vec::new();
        let mut state = GeminiStreamState::default();
        let payload = r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"Hi"}]}}]}"#;
        translate_gemini_chunk(
            payload,
            1,
            "gemini-1.5-flash",
            "test-id",
            &mut state,
            &mut out,
        )
        .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].choices[0].delta.content.as_deref(), Some("Hi"));
    }

    #[test]
    fn final_chunk_carries_finish_and_usage() {
        let mut out = Vec::new();
        let mut state = GeminiStreamState::default();
        let payload = r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"!"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":4,"candidatesTokenCount":3,"totalTokenCount":7}}"#;
        translate_gemini_chunk(
            payload,
            1,
            "gemini-1.5-flash",
            "test-id",
            &mut state,
            &mut out,
        )
        .unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].choices[0].delta.content.as_deref(), Some("!"));
        assert_eq!(out[1].choices[0].finish_reason.as_deref(), Some("stop"));
        assert_eq!(out[1].usage.as_ref().unwrap().total_tokens, 7);
    }

    #[test]
    fn stream_error_frame_is_surfaced_not_swallowed() {
        // A mid-stream error frame (no `candidates`) must become a stream Err — so
        // the proxy terminates WITHOUT [DONE] — not fall into the `None => Ok(())`
        // arm that silently ended a truncated stream as a clean success.
        let mut out = Vec::new();
        let mut state = GeminiStreamState::default();
        let payload =
            r#"{"error":{"code":503,"status":"UNAVAILABLE","message":"The model is overloaded."}}"#;
        let err = translate_gemini_chunk(
            payload,
            1,
            "gemini-1.5-flash",
            "test-id",
            &mut state,
            &mut out,
        )
        .expect_err("an error frame must surface as Err");
        assert!(err.to_string().contains("UNAVAILABLE"));
        assert!(!err.to_string().contains("overloaded"));
        assert!(out.is_empty(), "an error frame emits no content chunk");
    }

    #[test]
    fn thinking_tokens_fold_into_completion_and_preserve_the_total_invariant() {
        // gemini-2.5 thinking: candidatesTokenCount EXCLUDES thoughtsTokenCount,
        // which is in totalTokenCount. completion must be candidates + thoughts so
        // total == prompt + completion and reasoning is attributed to output.
        let mut out = Vec::new();
        let mut state = GeminiStreamState::default();
        let payload = r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":264,"candidatesTokenCount":104,"thoughtsTokenCount":989,"totalTokenCount":1357}}"#;
        translate_gemini_chunk(payload, 1, "gemini-2.5-pro", "id", &mut state, &mut out).unwrap();
        let usage = out.last().unwrap().usage.as_ref().unwrap();
        assert_eq!(usage.prompt_tokens, 264);
        assert_eq!(usage.completion_tokens, 104 + 989);
        assert_eq!(usage.total_tokens, 1357);
        assert_eq!(
            usage.total_tokens,
            usage.prompt_tokens + usage.completion_tokens
        );
    }

    #[test]
    fn stream_function_call_part_becomes_tool_call_delta() {
        // A streamed functionCall part → one complete tool_call delta (id + type +
        // name + arguments together, since Gemini delivers the call whole). The
        // synthesized id mirrors the buffered path (`call_<n>`).
        let mut out = Vec::new();
        let mut state = GeminiStreamState::default();
        let payload = r#"{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"get_weather","args":{"location":"SF"}}}]}}]}"#;
        translate_gemini_chunk(
            payload,
            1,
            "gemini-1.5-flash",
            "test-id",
            &mut state,
            &mut out,
        )
        .unwrap();
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
        translate_gemini_chunk(call, 1, "gemini-1.5-flash", "test-id", &mut state, &mut out)
            .unwrap();
        let mut out2 = Vec::new();
        let fin = r#"{"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":4,"candidatesTokenCount":3,"totalTokenCount":7}}"#;
        translate_gemini_chunk(fin, 1, "gemini-1.5-flash", "test-id", &mut state, &mut out2)
            .unwrap();
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
        let chunks: Vec<_> = gemini_sse_to_chunks(
            byte_stream,
            "gemini-1.5-flash".to_string(),
            "test-id".to_string(),
        )
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
        let chunks: Vec<_> = gemini_sse_to_chunks(
            byte_stream,
            "gemini-1.5-flash".to_string(),
            "test-id".to_string(),
        )
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
        // upstream fidelity fix: every chunk of one stream shares the id passed in.
        assert!(chunks.iter().all(|c| c.id == "test-id"));
    }

    /// upstream fidelity fix — a message with an image part builds a Gemini `inlineData` part
    /// (mimeType + decoded base64), instead of silently stringifying it away.
    #[test]
    fn vision_data_url_becomes_inline_data() {
        use routeplane_types::{ContentPart, ImageUrlContent, MessageContent};
        let m = routeplane_types::Message {
            role: "user".into(),
            content: MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "what is this?".into(),
                    cache_control: None,
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrlContent {
                        url: "data:image/png;base64,AAECAwQ=".into(),
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
        };
        let g = build_gemini_request(&req(vec![m])).unwrap();
        let parts = &g.contents[0].parts;
        assert_eq!(parts[0]["text"], "what is this?");
        assert_eq!(parts[1]["inlineData"]["mimeType"], "image/png");
        assert_eq!(parts[1]["inlineData"]["data"], "AAECAwQ=");
    }

    /// upstream fidelity fix — a non-inlinable http(s) image URL FAILS LOUD with a typed 422 rather
    /// than being silently dropped (PRD-011 FR-10).
    #[test]
    fn vision_http_url_is_rejected_422() {
        use routeplane_types::{ContentPart, ImageUrlContent, MessageContent};
        let m = routeplane_types::Message {
            role: "user".into(),
            content: MessageContent::Parts(vec![ContentPart::ImageUrl {
                image_url: ImageUrlContent {
                    url: "https://example.com/cat.png".into(),
                    detail: None,
                },
            }]),
            name: None,
            cache_control: None,
            tool_calls: None,
            tool_call_id: None,
            refusal: None,
            reasoning_content: None,
        };
        match build_gemini_request(&req(vec![m])) {
            Err(ProviderError::BadRequest { status, .. }) => assert_eq!(status, 422),
            other => panic!("expected 422 BadRequest, got {other:?}"),
        }
    }

    /// upstream fidelity fix — an n>1 response fans ALL candidates into choices[] (was: only [0]
    /// surfaced while candidatesTokenCount billed for all N). upstream fidelity fix — cached tokens
    /// lift. upstream fidelity fix — the response id is unique, not the old "gemini-resp".
    #[tokio::test]
    async fn multi_candidate_fans_out_and_lifts_cached_tokens() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        let resp = serde_json::json!({
            "candidates": [
                {"content": {"role": "model", "parts": [{"text": "one"}]}, "finishReason": "STOP"},
                {"content": {"role": "model", "parts": [{"text": "two"}]}, "finishReason": "STOP"}
            ],
            "usageMetadata": {
                "promptTokenCount": 5, "candidatesTokenCount": 8, "totalTokenCount": 13,
                "cachedContentTokenCount": 3
            }
        });
        Mock::given(method("POST"))
            .and(path("/v1beta/models/gemini-1.5-flash:generateContent"))
            .respond_with(ResponseTemplate::new(200).set_body_json(resp))
            .mount(&server)
            .await;

        let provider = GeminiProvider::with_base_url(server.uri());
        let mut r = req(vec![msg("user", "hi")]);
        r.n = Some(2);
        let out = provider
            .chat_completion(r, "gk-test".into())
            .await
            .expect("mock call succeeds");
        // Both candidates surfaced, indexed in order.
        assert_eq!(out.choices.len(), 2);
        assert_eq!(out.choices[0].index, 0);
        assert_eq!(out.choices[0].message.content.as_text(), "one");
        assert_eq!(out.choices[1].index, 1);
        assert_eq!(out.choices[1].message.content.as_text(), "two");
        // upstream fidelity fix: cached read tokens are lifted.
        assert_eq!(out.usage.cached_tokens, Some(3));
        // upstream fidelity fix: unique id, not the old hardcoded constant.
        assert!(out.id.starts_with("gemini-"));
        assert_ne!(out.id, "gemini-resp");
    }

    /// upstream fidelity fix — two calls yield DIFFERENT response ids.
    #[test]
    fn response_ids_are_unique() {
        assert_ne!(gemini_response_id(), gemini_response_id());
    }
}
