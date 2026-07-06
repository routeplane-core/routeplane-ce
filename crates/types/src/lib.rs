use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A data-residency jurisdiction (e.g. "IN", "EU", "US"). Kept as a free-form
/// code so adding a jurisdiction is configuration, not a code change — this is
/// what lets the sovereign-routing engine generalize past India (DPDP) to
/// GDPR/CCPA/etc.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Region(pub String);

impl Region {
    pub fn new(code: impl Into<String>) -> Self {
        Region(code.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Default, Serialize, Deserialize, Clone)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Max tokens to generate. Threaded through every adapter (Task #4); was
    /// previously dropped, which is why Anthropic hardcoded 1024.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// OpenAI's replacement for `max_tokens` on o-series / reasoning models
    /// (those models 400 on `max_tokens`). The reasoning-model output cap:
    /// forwarded verbatim to OpenAI-wire providers; native dialects
    /// (Anthropic/Gemini/Cohere) map it into their single cap, preferring it
    /// over `max_tokens` when both are set. `None` ⇒ omitted ⇒ byte-identical
    /// to a pre-field request.
    // NOTE: forward-compat unknown-field passthrough (a flattened `extra` map)
    // is deferred to a dedicated ADR — unknown request fields are dropped at
    // deserialization, the pre-existing safe posture.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    /// Stop sequence(s). OpenAI accepts EITHER a bare string (`"\n"`) OR an array
    /// (`["\n", "END"]`); we accept both on the wire (via [`deserialize_stop`]) and
    /// normalize to a list, then serialize it through to each provider that
    /// supports it — all of which take an array (OpenAI `stop`, Anthropic
    /// `stop_sequences`, Gemini `stopSequences`, Cohere `stop_sequences`). Without
    /// the custom deserializer a bare-string `stop` fails to parse into `Vec<String>`
    /// and the gateway 400s a request that api.openai.com accepts. `null`/absent ⇒
    /// `None` ⇒ omitted; the array form is unchanged, so a request that used the
    /// array form (or omitted `stop`) stays byte-identical.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_stop"
    )]
    pub stop: Option<Vec<String>>,
    /// Number of choices to generate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    /// Presence penalty (OpenAI-family).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    /// Frequency penalty (OpenAI-family).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    /// End-user identifier for abuse monitoring / per-user analytics.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    /// Function/tool definitions the model may call (OpenAI tool calling). Each
    /// entry is a [`Tool`] (today always `type: "function"`). Threaded through
    /// every adapter — OpenAI-family verbatim, Anthropic → `tools`
    /// (name/description/input_schema), Gemini → `functionDeclarations`. `None`
    /// ⇒ omitted ⇒ byte-identical to a pre-tool-calling request. A field added
    /// here but not mapped in an adapter is silently dropped (the golden rule),
    /// which is exactly the parity bug this closes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    /// How the model should choose a tool. OpenAI accepts EITHER a string
    /// (`"auto"`, `"none"`, `"required"`) OR an object
    /// (`{"type":"function","function":{"name":"..."}}`) — modelled as a raw
    /// [`serde_json::Value`] so the full fidelity of both shapes round-trips
    /// without a lossy fixed schema. `None` ⇒ omitted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
    /// Whether the model may emit multiple tool calls in one turn (OpenAI
    /// `parallel_tool_calls`). `None` ⇒ omitted ⇒ the provider's default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    /// Output-format control (OpenAI `response_format`). Carries BOTH the legacy
    /// JSON mode (`{"type":"json_object"}`) and structured outputs
    /// (`{"type":"json_schema","json_schema":{...}}`) shapes verbatim, so the full
    /// fidelity of either round-trips without a lossy fixed schema. Forwarded
    /// verbatim to OpenAI-family providers; STRIPPED for Anthropic (which has no
    /// `response_format`); mapped to Gemini's `generationConfig.responseMimeType`
    /// (+ `responseSchema`) when a JSON shape is requested. `None` ⇒ omitted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<serde_json::Value>,
    /// Deterministic-sampling seed (OpenAI `seed`). Best-effort determinism for a
    /// fixed seed + params. Forwarded verbatim to OpenAI-family; not supported by
    /// Anthropic/Gemini native bodies (stripped/ignored). `None` ⇒ omitted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    /// Whether to return log probabilities of the output tokens (OpenAI
    /// `logprobs`). OpenAI-family only. `None` ⇒ omitted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    /// Number of most-likely tokens to return at each position (OpenAI
    /// `top_logprobs`, 0–20; requires `logprobs:true`). OpenAI-family only.
    /// `None` ⇒ omitted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,
    /// Token-id → bias map adjusting sampling likelihood (OpenAI `logit_bias`,
    /// values −100..100). Typed map matches OpenAI's shape; deterministic key
    /// order on serialize (BTreeMap). OpenAI-family only. `None` ⇒ omitted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<BTreeMap<String, f32>>,
    /// Latency/throughput tier hint (OpenAI `service_tier`, e.g. `"auto"`,
    /// `"default"`, `"flex"`). OpenAI-family only. `None` ⇒ omitted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    /// Reasoning-effort hint for o-series / reasoning models (OpenAI
    /// `reasoning_effort`: `"low"`|`"medium"`|`"high"`). OpenAI-family only.
    /// `None` ⇒ omitted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
}

/// Deserialize the `stop` field, accepting OpenAI's TWO wire forms — a bare string
/// (`"\n"`) OR an array of strings (`["\n","END"]`) — and normalizing both to
/// `Option<Vec<String>>`. Without this, a bare-string `stop` fails to deserialize
/// into `Vec<String>` and the gateway 400s a request that api.openai.com accepts
/// (an OpenAI-compat violation). A JSON `null` (and, with `#[serde(default)]`, an
/// absent field) maps to `None`; the array form is unchanged, so a request that
/// used the array form (or omitted `stop`) stays byte-identical. Used via
/// `#[serde(default, deserialize_with = ...)]`.
fn deserialize_stop<'de, D>(deserializer: D) -> Result<Option<Vec<String>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum StopField {
        /// A single stop string (`"\n"`).
        One(String),
        /// An explicit list of stop strings.
        Many(Vec<String>),
    }
    let opt = Option::<StopField>::deserialize(deserializer)?;
    Ok(opt.map(|s| match s {
        StopField::One(one) => vec![one],
        StopField::Many(many) => many,
    }))
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Message {
    pub role: String,
    /// Message content — either a plain string OR an array of content parts
    /// (multimodal: text + image_url). `#[serde(untagged)]` on [`MessageContent`]
    /// preserves byte-level backward compatibility: existing `{"content": "hi"}`
    /// payloads deserialize identically, while new multimodal requests using the
    /// OpenAI array form `{"content": [{"type":"text","text":"hi"},{"type":"image_url",...}]}`
    /// are also accepted. Providers that don't support multimodal receive the
    /// string form (text parts concatenated) via [`MessageContent::as_text`].
    ///
    /// A tool-call assistant message legitimately carries `content: null` (OpenAI
    /// emits null content alongside `tool_calls`). `deserialize_content` maps a
    /// JSON `null` (and an absent field) to the default empty content so such a
    /// message deserializes instead of erroring; a present string/array is
    /// unchanged, so normal messages stay byte-identical.
    #[serde(default, deserialize_with = "deserialize_content")]
    pub content: MessageContent,
    /// Optional author name (OpenAI's per-message `name`). May itself carry PII
    /// (a person's name), so the guardrail engine masks it too (Task #6).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Provider-native prompt-caching marker (Anthropic prompt caching). When set
    /// (canonically `{"type":"ephemeral"}`) the Anthropic adapter attaches it to
    /// the corresponding native block so the message — or the system block, when
    /// this is a `system`-role message — becomes a cache breakpoint, cutting
    /// repeated-context cost ~90%. It is an OPAQUE passthrough: providers that do
    /// NOT accept it (OpenAI, which caches automatically and rejects unknown
    /// fields) must STRIP it from the outbound body — see each adapter. Default
    /// `None` ⇒ omitted ⇒ byte-identical to a request without caching.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<serde_json::Value>,
    /// Tool calls the assistant decided to make (OpenAI tool calling). Present on
    /// an ASSISTANT-role response message when the model invoked one or more
    /// tools, and echoed back by the caller on the assistant turn of a multi-turn
    /// tool conversation. `None` ⇒ omitted ⇒ byte-identical to a plain message.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// The id of the tool call this message answers — set on a `role:"tool"`
    /// message (the tool-result turn), matching the `id` of the `ToolCall` the
    /// assistant emitted. `None` ⇒ omitted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// OpenAI's `refusal` — set on an assistant message when the model declined
    /// a structured-outputs request. Passthrough: present only when the upstream
    /// supplied it, so a response without it stays byte-identical.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
    /// Reasoning text emitted alongside `content` by reasoning models
    /// (DeepSeek/xAI `reasoning_content`). Passthrough: forwarded to the client
    /// verbatim when present, omitted otherwise (byte-identical).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

// --- Tool / function-calling types (OpenAI tool calling) -----------------------
//
// The canonical OpenAI tool-calling surface. A request carries `tools` (the
// declarations) + optional `tool_choice`/`parallel_tool_calls`; the assistant's
// response message carries `tool_calls`; the caller then sends a `role:"tool"`
// message with `tool_call_id` carrying the result. Adapters translate this to/from
// each provider's native shape (Anthropic tool_use/tool_result blocks, Gemini
// functionDeclarations/functionCall). Every field uses `skip_serializing_if`/
// optional so a pre-tool-calling request stays byte-identical.

/// A tool the model may call. OpenAI's only `type` today is `"function"`.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct Tool {
    /// Always `"function"` today (kept as a free-form string for forward-compat
    /// with future tool types, e.g. OpenAI's built-in tools).
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionDef,
}

/// A function declaration: name, optional human description, and an optional
/// JSON-Schema `parameters` object. The schema is carried as a raw
/// [`serde_json::Value`] (no new dependency) so an arbitrary JSON Schema
/// round-trips verbatim.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct FunctionDef {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// The function's parameter schema (JSON Schema). `None` ⇒ omitted (a
    /// parameterless function).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

/// A tool call the assistant emitted in its response. `id` is the correlation
/// handle the caller echoes on the matching `role:"tool"` result message.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionCall,
}

/// The function invocation inside a [`ToolCall`]. Per OpenAI, `arguments` is a
/// JSON-ENCODED STRING (not a nested object), so callers `json.loads()` it; we
/// preserve that exact shape rather than parsing it into a Value.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct FunctionCall {
    pub name: String,
    /// The arguments as a JSON-encoded string (OpenAI's wire contract).
    pub arguments: String,
}

// --- Multimodal content types (PRD-011 multimodal passthrough) -----------------
//
// OpenAI's multimodal wire form: `content` can be EITHER a plain string (text-
// only messages) OR an array of typed content parts (text + image_url). The
// `#[serde(untagged)]` enum makes both forms round-trip byte-identical.
// Providers that don't support images extract text via `as_text()`.

/// A message's content: plain text OR an array of multimodal parts.
///
/// `#[serde(untagged)]` — a JSON string deserializes to `Text`, a JSON array to
/// `Parts`. Variant order matters: `Text` must be first so a bare string matches
/// before the array variant is tried.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Plain text content (the original, pre-multimodal form).
    Text(String),
    /// Multimodal content: an ordered array of text/image parts.
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    /// Extract the text-only view: for `Text`, the string itself; for `Parts`,
    /// concatenate every text part in order (image parts are skipped). This is
    /// what text-only providers (Anthropic, legacy adapters) receive.
    pub fn as_text(&self) -> String {
        match self {
            MessageContent::Text(s) => s.clone(),
            MessageContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::Text { text, .. } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(""),
        }
    }

    /// Whether this content contains any image parts (multimodal).
    pub fn has_images(&self) -> bool {
        matches!(self, MessageContent::Parts(parts) if parts.iter().any(|p| matches!(p, ContentPart::ImageUrl { .. })))
    }

    /// Apply a text-transform function to every text part, preserving the
    /// content structure (Text stays Text; Parts keeps image parts untouched
    /// and transforms each text part in place). Used by the guardrail engine
    /// to mask PII in multimodal messages without dropping image content.
    pub fn map_text<F: Fn(&str) -> String>(&self, f: F) -> Self {
        match self {
            MessageContent::Text(s) => MessageContent::Text(f(s)),
            MessageContent::Parts(parts) => MessageContent::Parts(
                parts
                    .iter()
                    .map(|p| match p {
                        // Preserve the cache_control marker while transforming the
                        // text (PII masking must not strip a cache breakpoint).
                        ContentPart::Text {
                            text,
                            cache_control,
                        } => ContentPart::Text {
                            text: f(text),
                            cache_control: cache_control.clone(),
                        },
                        other => other.clone(),
                    })
                    .collect(),
            ),
        }
    }
}

/// Default impl so existing code that constructs `Message { content: "hi".into(), .. }`
/// continues to compile (backward compat for in-code construction).
impl Default for MessageContent {
    fn default() -> Self {
        MessageContent::Text(String::new())
    }
}

impl From<String> for MessageContent {
    fn from(s: String) -> Self {
        MessageContent::Text(s)
    }
}

impl From<&str> for MessageContent {
    fn from(s: &str) -> Self {
        MessageContent::Text(s.to_string())
    }
}

/// Deserialize the `Message.content` field, tolerating a JSON `null` (mapped to
/// the default empty content). OpenAI emits `content: null` on a tool-call
/// assistant message; without this, the untagged [`MessageContent`] enum would
/// fail to match `null` and the whole response would error. A present
/// string/array deserializes exactly as before (byte-identical for normal
/// messages). Used via `#[serde(default, deserialize_with = ...)]`.
fn deserialize_content<'de, D>(deserializer: D) -> Result<MessageContent, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let opt = Option::<MessageContent>::deserialize(deserializer)?;
    Ok(opt.unwrap_or_default())
}

/// One part of a multimodal message content array. OpenAI-shaped: each part
/// carries a `type` discriminator.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    /// A text block: `{"type": "text", "text": "..."}`.
    ///
    /// `cache_control` is the provider-native prompt-caching marker (Anthropic):
    /// when present (`{"type":"ephemeral"}`) the Anthropic adapter emits it on the
    /// corresponding native text block so that block becomes a cache breakpoint.
    /// Opaque passthrough; default `None` ⇒ omitted ⇒ byte-identical. Providers
    /// that reject unknown fields (OpenAI) strip it.
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<serde_json::Value>,
    },
    /// An image reference: `{"type": "image_url", "image_url": {"url": "...", "detail": "..."}}`.
    /// The `url` can be an HTTP(S) URL or a base64 data URI (`data:image/png;base64,...`).
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrlContent },
}

/// The payload of an `image_url` content part.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageUrlContent {
    /// The image URL (HTTP(S) or base64 data URI).
    pub url: String,
    /// Optional detail level hint: `"low"`, `"high"`, or `"auto"`. Controls the
    /// token budget the provider spends on image understanding.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
    /// Backend configuration fingerprint OpenAI returns alongside `seed`
    /// determinism (`system_fingerprint`). Passthrough: present only when the
    /// upstream supplied it (OpenAI-family), absent otherwise so a response from a
    /// provider that does not report it stays byte-identical. `None` ⇒ omitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
    /// The service tier that actually processed the request (OpenAI
    /// `service_tier`). Echoed back to the caller when the upstream reports it.
    /// `None` ⇒ omitted ⇒ byte-identical for providers that don't report it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Choice {
    pub index: u32,
    pub message: Message,
    pub finish_reason: String,
    /// Per-choice log-probability payload (OpenAI `logprobs`), present only when
    /// the caller requested it AND the upstream returned it. Carried as a raw
    /// [`serde_json::Value`] so OpenAI's nested
    /// `{content:[{token,logprob,bytes,top_logprobs:[...]}]}` shape round-trips
    /// verbatim to the client. `None` ⇒ omitted ⇒ byte-identical otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<serde_json::Value>,
}

#[derive(Debug, Default, Serialize, Deserialize, Clone)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    /// Prompt-cache READ tokens: prompt tokens served from a provider's prompt
    /// cache (billed far cheaper than fresh input). Maps from Anthropic's
    /// `usage.cache_read_input_tokens` and OpenAI's
    /// `usage.prompt_tokens_details.cached_tokens`. `None` (and omitted from the
    /// wire) for providers/responses with no cache info, so a non-cached response
    /// is byte-identical. OpenAI counts these WITHIN `prompt_tokens`; this field is
    /// the cached SUBSET, surfaced so callers can see (and price) cache savings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u32>,
    /// Prompt-cache WRITE tokens: prompt tokens written into the cache on this
    /// request (Anthropic's `usage.cache_creation_input_tokens`; billed at a small
    /// premium over fresh input). `None`/omitted when the provider does not report
    /// cache creation (OpenAI's automatic cache reports no creation count), keeping
    /// non-cached responses byte-identical.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_creation_tokens: Option<u32>,
}

// --- Streaming (SSE) types ---------------------------------------------------
//
// When a client sends `"stream": true`, Routeplane responds with
// `Content-Type: text/event-stream` and emits a sequence of OpenAI-compatible
// `chat.completion.chunk` objects, one per `data:` SSE line, terminated by a
// literal `data: [DONE]`. These mirror OpenAI's streaming wire format so an
// OpenAI SDK using `stream=True` works against Routeplane unchanged. Each
// provider adapter translates its own native stream into this shape.

/// A single streamed chunk — the OpenAI `chat.completion.chunk` object.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatCompletionChunk {
    pub id: String,
    /// Always `"chat.completion.chunk"`.
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
    /// Usage is normally absent on chunks; OpenAI only sends it (on a final,
    /// otherwise-empty chunk) when `stream_options.include_usage` is set, and
    /// Anthropic surfaces it in its `message_delta`. We carry it when the
    /// provider supplies it so observability can record real token counts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// Backend configuration fingerprint on streamed chunks (OpenAI
    /// `system_fingerprint`). Passthrough: present only when the upstream
    /// supplied it, absent otherwise (byte-identical). `None` ⇒ omitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
    /// The service tier that processed the request (OpenAI `service_tier`).
    /// Passthrough on chunks, mirroring [`ChatCompletionResponse`]. `None` ⇒
    /// omitted ⇒ byte-identical for providers that don't report it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
}

/// One choice within a streamed chunk. The incremental content lives in `delta`.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: Delta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    /// Per-choice log-probability payload on a streamed chunk (OpenAI
    /// `logprobs`). Carried as a raw [`serde_json::Value`] (same rationale as
    /// [`Choice::logprobs`]) so the nested shape round-trips verbatim. `None` ⇒
    /// omitted ⇒ byte-identical.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<serde_json::Value>,
}

/// The incremental delta for a streamed choice. All fields are optional because
/// a chunk may carry only a role (the first chunk), only content, or only a
/// finish_reason (the last chunk).
#[derive(Debug, Default, Serialize, Deserialize, Clone)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Incremental tool-call deltas (OpenAI streams tool calls piecewise, keyed
    /// by `index`: the first delta for an index carries `id`/`type`/the function
    /// `name`, subsequent deltas append `arguments` fragments). `None` ⇒ omitted
    /// ⇒ byte-identical to a content-only delta.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallChunk>>,
    /// Streamed `refusal` delta (OpenAI structured outputs). Passthrough; `None`
    /// ⇒ omitted ⇒ byte-identical.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
    /// Streamed reasoning-text delta (DeepSeek/xAI `reasoning_content`).
    /// Passthrough; `None` ⇒ omitted ⇒ byte-identical.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

/// One incremental tool-call delta within a streamed [`Delta`]. `index`
/// identifies which tool call this fragment belongs to (a turn may stream
/// several in parallel). `id`/`type`/`function.name` arrive on the first
/// fragment for an index; `function.arguments` accumulates across fragments.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct ToolCallChunk {
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub tool_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<FunctionCallChunk>,
}

/// The function fragment inside a [`ToolCallChunk`]. Both fields are optional:
/// `name` arrives on the first fragment, `arguments` streams in pieces.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct FunctionCallChunk {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

impl ChatCompletionChunk {
    /// Convenience constructor for a single-choice content delta chunk.
    pub fn content_delta(
        id: impl Into<String>,
        model: impl Into<String>,
        created: u64,
        content: impl Into<String>,
    ) -> Self {
        ChatCompletionChunk {
            id: id.into(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model.into(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta {
                    role: None,
                    content: Some(content.into()),
                    tool_calls: None,
                    refusal: None,
                    reasoning_content: None,
                },
                finish_reason: None,
                logprobs: None,
            }],
            usage: None,
            system_fingerprint: None,
            service_tier: None,
        }
    }
}

// --- Embeddings (/v1/embeddings) types --------------------------------------
//
// OpenAI-shaped embeddings surface (PRD-011 §5, FR-1). `input` accepts either a
// bare string OR an array of strings — serialized untagged so the wire matches
// OpenAI exactly (an OpenAI SDK's `embeddings.create` works unchanged). The
// response mirrors OpenAI's `{ object:"list", data:[...], model, usage }`.
// Extend HERE first, then thread through every embeddings-capable adapter
// (openai / azure_openai / gemini); Anthropic uses the trait default (422).

/// The `input` of an embeddings request: a single string OR an array of strings.
/// `#[serde(untagged)]` makes the wire accept both forms (and emit the same form
/// back), matching OpenAI byte-for-byte. Variant order matters for untagged
/// deserialization: a JSON string deserializes to `Single`, a JSON array to
/// `Batch`.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(untagged)]
pub enum EmbeddingInput {
    /// A single string to embed.
    Single(String),
    /// A batch of strings to embed, in order.
    Batch(Vec<String>),
}

impl EmbeddingInput {
    /// Flatten to an ordered `Vec` of inputs (length 1 for `Single`, N for
    /// `Batch`). The order is load-bearing: the response's `index` field maps
    /// back to this order.
    pub fn to_vec(&self) -> Vec<String> {
        match self {
            EmbeddingInput::Single(s) => vec![s.clone()],
            EmbeddingInput::Batch(v) => v.clone(),
        }
    }

    /// Number of inputs.
    pub fn len(&self) -> usize {
        match self {
            EmbeddingInput::Single(_) => 1,
            EmbeddingInput::Batch(v) => v.len(),
        }
    }

    /// Whether there are no inputs (only possible for an empty `Batch`).
    pub fn is_empty(&self) -> bool {
        matches!(self, EmbeddingInput::Batch(v) if v.is_empty())
    }
}

/// An OpenAI-shaped embeddings request (PRD-011 FR-1).
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: EmbeddingInput,
    /// `"float"` (default) or `"base64"`. Threaded to providers that honor it;
    /// providers that only emit float vectors ignore it (documented degrade).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<String>,
    /// Requested output dimensionality (OpenAI text-embedding-3-*; Gemini
    /// `outputDimensionality`). Absent ⇒ the provider's native default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
    /// End-user identifier for abuse monitoring (OpenAI-family).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

/// An OpenAI-shaped embeddings response (PRD-011 FR-1).
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EmbeddingResponse {
    /// Always `"list"`.
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: EmbeddingUsage,
}

/// The payload of one embedding: either a JSON array of floats
/// (`encoding_format:"float"`, the default) OR a base64-encoded string of packed
/// little-endian `f32`s (`encoding_format:"base64"`). The openai-python SDK
/// requests base64 BY DEFAULT when numpy is installed (it decodes client-side), so
/// OpenAI returns each `embedding` as a base64 STRING — modelling only `Vec<f32>`
/// made that response fail the typed decode in `openai.rs`'s
/// `.json::<EmbeddingResponse>()`, surfacing as a 500 that ALSO recorded a false
/// failure against the shared provider circuit breaker. `#[serde(untagged)]` accepts
/// and re-emits whichever form the upstream returned, byte-faithfully: a float array
/// stays a float array (byte-identical to before), a base64 string is forwarded
/// verbatim as a string instead of exploding. Variant order matters — a JSON array
/// must match `Floats` before the `Base64` string variant is tried.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum EmbeddingVector {
    /// The float vector form (`encoding_format:"float"`, the default).
    Floats(Vec<f32>),
    /// The base64-packed-`f32` form (`encoding_format:"base64"`), forwarded verbatim.
    Base64(String),
}

impl EmbeddingVector {
    /// The float slice when the upstream returned the float form; `None` for the
    /// base64 string (which this gateway forwards verbatim without decoding).
    pub fn as_floats(&self) -> Option<&[f32]> {
        match self {
            EmbeddingVector::Floats(v) => Some(v),
            EmbeddingVector::Base64(_) => None,
        }
    }

    /// Consume into the owned float vector (float form only; `None` for base64).
    pub fn into_floats(self) -> Option<Vec<f32>> {
        match self {
            EmbeddingVector::Floats(v) => Some(v),
            EmbeddingVector::Base64(_) => None,
        }
    }

    /// Length of the underlying payload (float count, or base64 string length).
    pub fn len(&self) -> usize {
        match self {
            EmbeddingVector::Floats(v) => v.len(),
            EmbeddingVector::Base64(s) => s.len(),
        }
    }

    /// Whether the payload is empty.
    pub fn is_empty(&self) -> bool {
        match self {
            EmbeddingVector::Floats(v) => v.is_empty(),
            EmbeddingVector::Base64(s) => s.is_empty(),
        }
    }
}

impl From<Vec<f32>> for EmbeddingVector {
    fn from(v: Vec<f32>) -> Self {
        EmbeddingVector::Floats(v)
    }
}

/// One embedding vector in the response `data[]`, keyed by `index` to the
/// corresponding input.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EmbeddingData {
    /// Always `"embedding"`.
    pub object: String,
    pub index: u32,
    /// The vector payload — a float array or a base64 string (see [`EmbeddingVector`]).
    pub embedding: EmbeddingVector,
}

/// Embeddings token accounting. There are no completion tokens for an
/// embeddings call, so OpenAI omits that field: `prompt_tokens == total_tokens`.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EmbeddingUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

// --- Rerank (/v1/rerank) types ----------------------------------------------
//
// Reranking surface (parity with LiteLLM's `/rerank` and Cohere/Jina's native
// rerank API — core to RAG pipelines: given a query and a candidate document
// set, return the documents ordered by relevance). The public wire shape mirrors
// Cohere's / LiteLLM's `/rerank` so OpenAI-ecosystem + LiteLLM clients interop
// unchanged. Extend HERE first, then thread through every rerank-capable
// adapter (Cohere today); providers without a first-party rerank endpoint use
// the trait default (422 `rerank_not_supported`).

/// An OpenAI/Cohere-shaped rerank request.
///
/// `documents` is an ordered list of candidate strings; the response's `index`
/// field maps each result back to this order (load-bearing). `top_n` caps the
/// number of returned results. `return_documents` (default `false`, matching
/// Cohere/LiteLLM) asks the gateway to echo the document text back in each
/// result so a client need not keep the original list around.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RerankRequest {
    pub model: String,
    pub query: String,
    pub documents: Vec<String>,
    /// Cap on the number of ranked results returned. Absent ⇒ the provider's
    /// default (Cohere returns all documents ranked).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_n: Option<u32>,
    /// Echo the document text back in each result. Default `false` (Cohere /
    /// LiteLLM default), so the wire stays minimal unless explicitly requested.
    #[serde(default)]
    pub return_documents: bool,
}

/// One ranked result: the input document's `index` and its `relevance_score`
/// (in `[0, 1]`, higher = more relevant). `document` is present only when the
/// request set `return_documents: true`.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RerankResult {
    /// Index into the request `documents` array (maps the result back to its
    /// input — load-bearing).
    pub index: u32,
    pub relevance_score: f64,
    /// The echoed document text, present only when `return_documents` was set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document: Option<RerankDocument>,
}

/// The echoed document in a [`RerankResult`]. Cohere/LiteLLM model this as an
/// object `{"text": "..."}`; we mirror that shape so an existing client parses
/// it unchanged.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RerankDocument {
    pub text: String,
}

/// An OpenAI/Cohere-shaped rerank response. Results are ordered by
/// `relevance_score` descending (the upstream ordering is preserved).
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RerankResponse {
    /// Provider-supplied request id, when present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub model: String,
    pub results: Vec<RerankResult>,
    pub usage: RerankUsage,
}

/// Rerank token / search-unit accounting. Cohere bills rerank in
/// `search_units` (not tokens); we surface that count so observability and
/// cost attribution have a real number rather than a fabricated token total.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RerankUsage {
    /// Cohere `meta.billed_units.search_units`. 0 when the provider omits it.
    pub search_units: u32,
    /// Total billable units for this rerank call (== `search_units` today —
    /// kept distinct so a future per-token reranker can populate it).
    pub total_tokens: u32,
}

// --- Moderation (/v1/moderations) types -------------------------------------
//
// Content-moderation surface (PARITY: OpenAI exposes `/v1/moderations` and
// LiteLLM proxies `/moderations`). The public wire shape mirrors OpenAI's
// Moderation object 1:1 so an existing OpenAI-SDK client (or LangChain's
// `OpenAIModerationChain`) interops unchanged. Extend HERE first.
//
// `categories` and `category_scores` are kept as flat `BTreeMap<String, …>`
// (deterministic key order on serialize) rather than a fixed struct: OpenAI's
// category set evolves (it added `illicit`, `hate/threatening`, … over time),
// and Routeplane's local moderator emits its own canonical taxonomy
// ([`routeplane_guardrails::moderation::ModerationCategory`]). A map keeps the
// wire faithful to whatever the source produced without a lossy fixed schema.

/// An OpenAI-shaped moderation request. `input` is a single string or a batch
/// (one result per input, in order) — the same `string | array` shape as
/// embeddings. `model` is optional; the provider supplies its default
/// (`omni-moderation-latest` for OpenAI) when absent.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModerationRequest {
    /// One string, or an ordered batch (one [`ModerationResult`] per input).
    pub input: ModerationInput,
    /// The moderation model. Absent ⇒ the provider's default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

/// Moderation input: a single string or an ordered batch. Mirrors
/// [`EmbeddingInput`] (untagged `string | array`) so the wire is identical to
/// OpenAI's `input` field.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(untagged)]
pub enum ModerationInput {
    Single(String),
    Batch(Vec<String>),
}

impl ModerationInput {
    /// Flatten to an ordered `Vec` (length 1 for `Single`, N for `Batch`). The
    /// order is load-bearing: `results[i]` maps to input `i`.
    pub fn to_vec(&self) -> Vec<String> {
        match self {
            ModerationInput::Single(s) => vec![s.clone()],
            ModerationInput::Batch(v) => v.clone(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            ModerationInput::Single(_) => 1,
            ModerationInput::Batch(v) => v.len(),
        }
    }

    /// Whether there are no inputs (only possible for an empty `Batch`).
    pub fn is_empty(&self) -> bool {
        matches!(self, ModerationInput::Batch(v) if v.is_empty())
    }
}

/// An OpenAI-shaped moderation response. `results[i]` corresponds to input `i`.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModerationResponse {
    /// `modr-...` (OpenAI prefix).
    pub id: String,
    pub model: String,
    pub results: Vec<ModerationResult>,
}

/// One moderation result. `flagged` is true when any category tripped its
/// threshold. `categories` is the per-category boolean flag map and
/// `category_scores` the per-category probability map — the two maps share the
/// same key set (the category labels the source emitted).
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModerationResult {
    pub flagged: bool,
    pub categories: BTreeMap<String, bool>,
    pub category_scores: BTreeMap<String, f64>,
}

// --- Feedback (/v1/feedback) types ------------------------------------------
//
// Feedback API (PARITY: Portkey ships `POST /v1/feedback`; Helicone has
// feedback/scoring). A client attaches a weighted quality score to a prior
// request trace so eval/quality loops (and the prompt A/B variant analytics
// already shipped) have a signal to learn from. This is a GATEWAY-control
// surface — feedback never flows to a provider; it is recorded OFF the hot path
// into the in-memory observability ring. The wire shape mirrors Portkey's
// verified contract 1:1.

/// An OpenAI/Portkey-shaped feedback submission. `trace_id` references a prior
/// request (the gateway surfaces it as `x-routeplane-trace-id` on responses);
/// `value` is a weighted score in `-10..=10`; `weight` (default `1.0`) is its
/// relative weight in `0.0..=1.0`; `metadata` is an OPTIONAL caller object that
/// is bounded + label-cleaned at the route edge (never persisted raw, never
/// routed into any tamper-evident surface). Range/shape validation lives in the
/// handler (`feedback_api.rs`) so a clean OpenAI-style 400 envelope can be
/// returned per field.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FeedbackRequest {
    /// The request trace this feedback refers to (the response's
    /// `x-routeplane-trace-id`). Non-empty + length-bounded at the route edge.
    pub trace_id: String,
    /// Weighted quality score in `-10..=10` (validated in the handler).
    pub value: i8,
    /// Relative weight in `0.0..=1.0`; absent ⇒ `1.0` (validated in the handler).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weight: Option<f32>,
    /// Optional caller metadata object. Bounded (key/value count + length capped)
    /// and label-cleaned before any retention — arbitrary unbounded user metadata
    /// is NOT stored into audit-grade surfaces.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

// --- Image generation (/v1/images/generations) types ------------------------
//
// Image-generation surface (PARITY: OpenAI exposes `/v1/images/generations` and
// LiteLLM/Portkey proxy image generation). The public wire shape mirrors
// OpenAI's Images object 1:1 so an existing OpenAI-SDK client interops
// unchanged. Extend HERE first, then thread through every adapter.
//
// The request stays close to OpenAI: only `prompt` is strictly required (the
// adapter supplies a default `model` when absent). The well-known fields (`n`,
// `size`, `quality`, `response_format`) are mapped explicitly; anything else a
// caller sends (e.g. `style`, `background`, `user`) is preserved verbatim via a
// flattened `extra` map so a forward-compatible OpenAI field is NOT silently
// dropped on its way upstream.

/// An OpenAI-shaped image-generation request. Only `prompt` is required; `model`
/// defaults at the adapter (OpenAI ⇒ `gpt-image-1`). The well-known optional
/// fields are explicit; any other caller-supplied field is carried through
/// untouched in [`extra`](ImageGenerationRequest::extra) so the translation is
/// not lossy for forward-compatible OpenAI fields.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ImageGenerationRequest {
    /// The image model. Absent ⇒ the provider's default (`gpt-image-1`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// The text prompt describing the image(s) to generate. Required.
    pub prompt: String,
    /// How many images to generate. Absent ⇒ the provider's default (1).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    /// Image dimensions, e.g. `"1024x1024"`. Absent ⇒ the provider's default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,
    /// Rendering quality (e.g. `"standard"`/`"hd"` for dall-e-3, `"high"`/
    /// `"medium"`/`"low"` for gpt-image-1). Absent ⇒ the provider's default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality: Option<String>,
    /// `"url"` or `"b64_json"`. Absent ⇒ the provider's default (gpt-image-1 is
    /// always b64; dall-e-3 defaults to url).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,
    /// Any other caller-supplied OpenAI field (e.g. `style`, `background`,
    /// `user`) carried through verbatim so the mapping is not lossy.
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

/// An OpenAI-shaped image-generation response. `data[i]` is one generated image,
/// carrying either a hosted `url` or inline `b64_json` (mutually exclusive per
/// the request's `response_format`).
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ImageGenerationResponse {
    /// Unix epoch seconds the batch was created (OpenAI returns this).
    pub created: i64,
    pub data: Vec<ImageData>,
    /// gpt-image-1 returns a top-level `usage` block; tolerated (and echoed) when
    /// present, omitted otherwise. Not required by the contract.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<serde_json::Value>,
}

/// One generated image: a hosted `url` OR inline `b64_json` (never both for a
/// given `response_format`). `revised_prompt` is present for dall-e-3, which
/// rewrites the prompt before generating.
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct ImageData {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub b64_json: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revised_prompt: Option<String>,
}

/// Parameters for an audio-transcription (speech-to-text) request, threaded from
/// the multipart handler to the adapter. The audio bytes themselves are NOT held
/// here (they ride in [`TranscriptionInput`]) — this is the small, text-only
/// field set parsed out of the `multipart/form-data` body.
///
/// `multipart/form-data` is the inbound contract for `/v1/audio/transcriptions`
/// (OpenAI/Groq), so there is no JSON request type — the handler parses the
/// multipart and fills this struct. It exists only to thread the fields cleanly
/// to the adapter so the translation is not lossy.
#[derive(Debug, Clone, Default)]
pub struct TranscriptionParams {
    /// The STT model (required by OpenAI/Groq), e.g. `whisper-1`,
    /// `gpt-4o-transcribe`, `whisper-large-v3`.
    pub model: String,
    /// Optional ISO-639-1 language hint (e.g. `en`) — improves accuracy/latency.
    pub language: Option<String>,
    /// Optional prompt to guide the model's style or continue a prior segment.
    pub prompt: Option<String>,
    /// Output format: `json` (default), `text`, `srt`, `verbose_json`, `vtt`.
    pub response_format: Option<String>,
    /// Sampling temperature in `[0, 1]`. Absent ⇒ the provider's default.
    pub temperature: Option<f32>,
}

/// The full transcription request handed to an adapter: the binary audio file
/// plus the threaded [`TranscriptionParams`]. The bytes are carried by value
/// (already buffered by the handler under a route-specific body cap) and are
/// NEVER logged.
#[derive(Clone)]
pub struct TranscriptionInput {
    /// Raw audio bytes (flac/mp3/mp4/mpeg/mpga/m4a/ogg/wav/webm).
    pub file_bytes: Vec<u8>,
    /// The original filename — the upstream uses its extension to sniff the codec,
    /// so it must be forwarded (default `audio.wav` when the client omitted it).
    pub filename: String,
    /// The form fields (model/language/prompt/response_format/temperature).
    pub params: TranscriptionParams,
}

impl std::fmt::Debug for TranscriptionInput {
    /// Custom Debug that NEVER prints the audio bytes (only their length) — the
    /// bytes are user content and must not leak into logs.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TranscriptionInput")
            .field("file_bytes_len", &self.file_bytes.len())
            .field("filename", &self.filename)
            .field("params", &self.params)
            .finish()
    }
}

/// An OpenAI-shaped transcription response. The primary (`json`) contract is
/// `{"text": "<transcript>"}`; richer formats (`verbose_json`) add fields like
/// `language`, `duration`, `segments` — those are tolerated and echoed verbatim
/// via [`extra`](TranscriptionResponse::extra) so the mapping is not lossy.
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct TranscriptionResponse {
    /// The transcribed text.
    pub text: String,
    /// Any other field the upstream returned (verbose_json: `language`,
    /// `duration`, `segments`, `words`, …) carried through untouched.
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

/// An OpenAI-shaped text-to-speech (TTS) request for `/v1/audio/speech` (parity
/// with OpenAI; LiteLLM/Portkey proxy it). Unlike transcription, the INBOUND
/// contract is JSON (the `input` is text), but the RESPONSE is raw binary audio
/// — so there is no response type here (the handler/adapter return the bytes
/// plus a `Content-Type`).
///
/// The well-known optional fields are explicit; any other caller-supplied field
/// is carried through untouched in [`extra`](SpeechRequest::extra) so the
/// translation is not lossy for forward-compatible OpenAI fields.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SpeechRequest {
    /// The TTS model. Absent ⇒ the provider's default (`gpt-4o-mini-tts`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// The text to synthesize. Required. This is user TEXT bound for an external
    /// API → PII-masked at the route edge before egress (same posture as the
    /// image-generation prompt).
    pub input: String,
    /// The voice (e.g. `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`).
    /// Required by OpenAI.
    pub voice: String,
    /// Output container: `mp3` (default), `opus`, `aac`, `flac`, `wav`, `pcm`.
    /// Absent ⇒ the provider's default (mp3). Drives the response `Content-Type`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,
    /// Playback speed in `[0.25, 4.0]`. Absent ⇒ the provider's default (1.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed: Option<f32>,
    /// Any other caller-supplied OpenAI field carried through verbatim so the
    /// mapping is not lossy.
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- new OpenAI-compat request/response fields (PRD-011 unified shapes) -----

    #[test]
    fn request_round_trips_response_format_json_schema() {
        // Structured outputs: the full `{"type":"json_schema","json_schema":{...}}`
        // shape must round-trip verbatim (carried as a serde_json::Value).
        let original = serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "person",
                    "strict": true,
                    "schema": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                        "additionalProperties": false
                    }
                }
            }
        });
        let parsed: ChatCompletionRequest = serde_json::from_value(original.clone()).unwrap();
        let back = serde_json::to_value(&parsed).unwrap();
        assert_eq!(back["response_format"], original["response_format"]);
    }

    #[test]
    fn request_round_trips_response_format_json_object() {
        let original = serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "json_object"}
        });
        let parsed: ChatCompletionRequest = serde_json::from_value(original.clone()).unwrap();
        let back = serde_json::to_value(&parsed).unwrap();
        assert_eq!(
            back["response_format"],
            serde_json::json!({"type": "json_object"})
        );
    }

    #[test]
    fn request_round_trips_seed_logprobs_logit_bias_and_tiers() {
        let original = serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "seed": 42,
            "logprobs": true,
            "top_logprobs": 5,
            "logit_bias": {"50256": -100.0, "1234": 12.5},
            "service_tier": "flex",
            "reasoning_effort": "high"
        });
        let parsed: ChatCompletionRequest = serde_json::from_value(original.clone()).unwrap();
        assert_eq!(parsed.seed, Some(42));
        assert_eq!(parsed.logprobs, Some(true));
        assert_eq!(parsed.top_logprobs, Some(5));
        assert_eq!(parsed.service_tier.as_deref(), Some("flex"));
        assert_eq!(parsed.reasoning_effort.as_deref(), Some("high"));
        let lb = parsed.logit_bias.as_ref().unwrap();
        assert_eq!(lb.get("50256"), Some(&-100.0));
        let back = serde_json::to_value(&parsed).unwrap();
        assert_eq!(back["seed"], 42);
        assert_eq!(back["logprobs"], true);
        assert_eq!(back["top_logprobs"], 5);
        assert_eq!(back["service_tier"], "flex");
        assert_eq!(back["reasoning_effort"], "high");
        // logit_bias is a typed map → object with deterministic key order.
        assert_eq!(back["logit_bias"]["50256"], -100.0);
        assert_eq!(back["logit_bias"]["1234"], 12.5);
    }

    #[test]
    fn request_without_new_fields_is_byte_identical() {
        // The additive Option + skip_serializing_if guarantee: a request that sets
        // none of the new fields serializes WITHOUT any of their keys (golden /
        // ab_parity stay byte-identical).
        let req = ChatCompletionRequest {
            model: "gpt-4o".into(),
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
            ..Default::default()
        };
        let v = serde_json::to_value(&req).unwrap();
        for absent in [
            "response_format",
            "seed",
            "logprobs",
            "top_logprobs",
            "logit_bias",
            "service_tier",
            "reasoning_effort",
            "max_completion_tokens",
        ] {
            assert!(
                v.get(absent).is_none(),
                "{absent} must be omitted when unset"
            );
        }
        // Exactly the two required keys present (no new field leaked in).
        assert_eq!(v.as_object().unwrap().len(), 2);
    }

    #[test]
    fn request_round_trips_max_completion_tokens() {
        // The typed max_completion_tokens field must parse and re-serialize
        // verbatim. Unknown fields (e.g. `prediction`, `store`) are DROPPED at
        // deserialization — the pre-existing safe posture; forward-compat
        // passthrough is deferred to a dedicated ADR.
        let original = serde_json::json!({
            "model": "o4-mini",
            "messages": [{"role": "user", "content": "hi"}],
            "max_completion_tokens": 4096,
            "prediction": {"type": "content", "content": "guess"},
            "store": true
        });
        let parsed: ChatCompletionRequest = serde_json::from_value(original.clone()).unwrap();
        assert_eq!(parsed.max_completion_tokens, Some(4096));
        let back = serde_json::to_value(&parsed).unwrap();
        assert_eq!(back["max_completion_tokens"], 4096);
        // Unknown inbound keys do not survive (dropped at ingress).
        assert!(back.get("prediction").is_none());
        assert!(back.get("store").is_none());
    }

    #[test]
    fn chunk_round_trips_reasoning_refusal_logprobs_and_fingerprint() {
        // The streaming passthrough fields must parse out of an OpenAI-wire
        // chunk and re-serialize verbatim; absent ⇒ omitted (byte-identical).
        let original = serde_json::json!({
            "id": "c1",
            "object": "chat.completion.chunk",
            "created": 1u64,
            "model": "deepseek-reasoner",
            "system_fingerprint": "fp_xyz",
            "service_tier": "default",
            "choices": [{
                "index": 0,
                "delta": {
                    "content": "42",
                    "reasoning_content": "thinking ...",
                    "refusal": null
                },
                "finish_reason": null,
                "logprobs": {"content": []}
            }]
        });
        let parsed: ChatCompletionChunk = serde_json::from_value(original).unwrap();
        assert_eq!(parsed.system_fingerprint.as_deref(), Some("fp_xyz"));
        assert_eq!(parsed.service_tier.as_deref(), Some("default"));
        assert_eq!(
            parsed.choices[0].delta.reasoning_content.as_deref(),
            Some("thinking ...")
        );
        assert!(parsed.choices[0].delta.refusal.is_none());
        assert_eq!(
            parsed.choices[0].logprobs,
            Some(serde_json::json!({"content": []}))
        );
        let back = serde_json::to_value(&parsed).unwrap();
        assert_eq!(back["system_fingerprint"], "fp_xyz");
        assert_eq!(
            back["choices"][0]["delta"]["reasoning_content"],
            "thinking ..."
        );
        assert_eq!(
            back["choices"][0]["logprobs"],
            serde_json::json!({"content": []})
        );
        // A null/absent refusal stays omitted on the way back out.
        assert!(back["choices"][0]["delta"].get("refusal").is_none());
    }

    #[test]
    fn message_round_trips_refusal_and_reasoning_content() {
        let original = serde_json::json!({
            "role": "assistant",
            "content": "the answer is 42",
            "refusal": null,
            "reasoning_content": "step by step ..."
        });
        let msg: Message = serde_json::from_value(original).unwrap();
        assert!(msg.refusal.is_none());
        assert_eq!(msg.reasoning_content.as_deref(), Some("step by step ..."));
        let back = serde_json::to_value(&msg).unwrap();
        assert_eq!(back["reasoning_content"], "step by step ...");
        assert!(back.get("refusal").is_none());
        // Absent on input ⇒ absent on output (byte-identical guarantee).
        let plain: Message = serde_json::from_value(serde_json::json!({
            "role": "assistant",
            "content": "hi"
        }))
        .unwrap();
        let back = serde_json::to_value(&plain).unwrap();
        assert!(back.get("refusal").is_none());
        assert!(back.get("reasoning_content").is_none());
    }

    #[test]
    fn request_accepts_bare_string_and_array_stop() {
        // Regression: OpenAI accepts `stop` as EITHER a bare string OR an array. The
        // bare-string form previously failed to deserialize into `Vec<String>` and
        // 400'd a request api.openai.com accepts — both forms must now parse and
        // normalize to a list.
        let bare: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "stop": "\n"
        }))
        .unwrap();
        assert_eq!(bare.stop.as_deref(), Some(&["\n".to_string()][..]));

        let array: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "stop": ["\n", "END"]
        }))
        .unwrap();
        assert_eq!(
            array.stop.as_deref(),
            Some(&["\n".to_string(), "END".to_string()][..])
        );

        // `null` normalizes to None (as does an absent field via `#[serde(default)]`).
        let nulled: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "stop": null
        }))
        .unwrap();
        assert!(nulled.stop.is_none());

        // Both non-null forms serialize to the array shape every provider takes; the
        // array form is byte-identical to its input, and an unset `stop` is omitted.
        assert_eq!(
            serde_json::to_value(&bare).unwrap()["stop"],
            serde_json::json!(["\n"])
        );
        assert_eq!(
            serde_json::to_value(&array).unwrap()["stop"],
            serde_json::json!(["\n", "END"])
        );
        let none = ChatCompletionRequest {
            model: "gpt-4o".into(),
            ..Default::default()
        };
        assert!(serde_json::to_value(&none).unwrap().get("stop").is_none());
    }

    #[test]
    fn response_round_trips_logprobs_fingerprint_and_service_tier() {
        // OpenAI returns per-choice logprobs + system_fingerprint + service_tier;
        // they must parse back and re-serialize to the client (passthrough).
        let original = serde_json::json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1700000000u64,
            "model": "gpt-4o",
            "system_fingerprint": "fp_abc123",
            "service_tier": "default",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
                "logprobs": {
                    "content": [
                        {"token": "hi", "logprob": -0.31, "bytes": [104, 105], "top_logprobs": []}
                    ]
                }
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        });
        let parsed: ChatCompletionResponse = serde_json::from_value(original.clone()).unwrap();
        assert_eq!(parsed.system_fingerprint.as_deref(), Some("fp_abc123"));
        assert_eq!(parsed.service_tier.as_deref(), Some("default"));
        assert!(parsed.choices[0].logprobs.is_some());
        let back = serde_json::to_value(&parsed).unwrap();
        assert_eq!(back["system_fingerprint"], "fp_abc123");
        assert_eq!(back["service_tier"], "default");
        assert_eq!(
            back["choices"][0]["logprobs"],
            original["choices"][0]["logprobs"]
        );
    }

    #[test]
    fn response_without_new_fields_is_byte_identical() {
        let original = serde_json::json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1700000000u64,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        });
        let parsed: ChatCompletionResponse = serde_json::from_value(original.clone()).unwrap();
        let back = serde_json::to_value(&parsed).unwrap();
        assert!(back.get("system_fingerprint").is_none());
        assert!(back.get("service_tier").is_none());
        assert!(back["choices"][0].get("logprobs").is_none());
    }

    #[test]
    fn transcription_response_round_trips_json_shape() {
        // The primary `json` contract: a bare `{"text": "..."}`.
        let raw = r#"{"text":"hello world"}"#;
        let resp: TranscriptionResponse = serde_json::from_str(raw).unwrap();
        assert_eq!(resp.text, "hello world");
        assert!(resp.extra.is_empty());
        let back = serde_json::to_value(&resp).unwrap();
        assert_eq!(back["text"], "hello world");
        // No spurious keys emitted when extra is empty.
        assert_eq!(back.as_object().unwrap().len(), 1);
    }

    #[test]
    fn transcription_response_tolerates_verbose_json_extra() {
        // verbose_json carries language/duration/segments — they must survive
        // round-trip via the flattened `extra` map (not silently dropped).
        let raw = r#"{"text":"hi","language":"english","duration":1.23,"segments":[]}"#;
        let resp: TranscriptionResponse = serde_json::from_str(raw).unwrap();
        assert_eq!(resp.text, "hi");
        assert_eq!(resp.extra.get("language").unwrap(), "english");
        assert!(resp.extra.contains_key("duration"));
        assert!(resp.extra.contains_key("segments"));
        let back = serde_json::to_value(&resp).unwrap();
        assert_eq!(back["language"], "english");
        assert_eq!(back["duration"], 1.23);
    }

    #[test]
    fn transcription_input_debug_never_prints_audio_bytes() {
        // Load-bearing privacy guard: Debug must show only the byte LENGTH, never
        // the bytes (audio is user content — never logged).
        let input = TranscriptionInput {
            file_bytes: vec![1, 2, 3, 4, 5, 6, 7, 8],
            filename: "speech.wav".into(),
            params: TranscriptionParams {
                model: "whisper-1".into(),
                ..Default::default()
            },
        };
        let dbg = format!("{input:?}");
        assert!(dbg.contains("file_bytes_len"));
        assert!(dbg.contains('8'));
        assert!(dbg.contains("speech.wav"));
        // The raw byte sequence must NOT appear.
        assert!(!dbg.contains("[1, 2, 3"));
    }

    #[test]
    fn speech_request_round_trips_openai_shape() {
        // A full OpenAI TTS request must deserialize into the typed fields and
        // serialize back 1:1 (no spurious keys, optional fields omitted when
        // absent).
        let raw = r#"{"model":"tts-1","input":"Hello world","voice":"alloy","response_format":"mp3","speed":1.5}"#;
        let req: SpeechRequest = serde_json::from_str(raw).unwrap();
        assert_eq!(req.model.as_deref(), Some("tts-1"));
        assert_eq!(req.input, "Hello world");
        assert_eq!(req.voice, "alloy");
        assert_eq!(req.response_format.as_deref(), Some("mp3"));
        assert_eq!(req.speed, Some(1.5));
        assert!(req.extra.is_empty());

        let back = serde_json::to_value(&req).unwrap();
        assert_eq!(back["model"], "tts-1");
        assert_eq!(back["input"], "Hello world");
        assert_eq!(back["voice"], "alloy");
        assert_eq!(back["response_format"], "mp3");
        assert_eq!(back["speed"], 1.5);
    }

    #[test]
    fn speech_request_minimal_omits_optionals_and_keeps_extra() {
        // A bare `{input, voice}` (no model/response_format/speed) is valid —
        // the adapter fills the provider default model. Unknown forward-compatible
        // fields ride `extra` so the mapping is not lossy.
        let raw = r#"{"input":"hi","voice":"nova","instructions":"speak slowly"}"#;
        let req: SpeechRequest = serde_json::from_str(raw).unwrap();
        assert!(req.model.is_none());
        assert!(req.response_format.is_none());
        assert!(req.speed.is_none());
        assert_eq!(req.extra.get("instructions").unwrap(), "speak slowly");

        let back = serde_json::to_value(&req).unwrap();
        let obj = back.as_object().unwrap();
        // Only input + voice + the flattened extra survive; no null optionals.
        assert!(!obj.contains_key("model"));
        assert!(!obj.contains_key("response_format"));
        assert!(!obj.contains_key("speed"));
        assert_eq!(obj["instructions"], "speak slowly");
    }

    #[test]
    fn chunk_serializes_to_openai_shape() {
        let chunk = ChatCompletionChunk {
            id: "rp_chunk_1".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 1_700_000_000,
            model: "gpt-4o".to_string(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta {
                    role: Some("assistant".to_string()),
                    content: Some("Hello".to_string()),
                    tool_calls: None,
                    refusal: None,
                    reasoning_content: None,
                },
                finish_reason: None,
                logprobs: None,
            }],
            usage: None,
            system_fingerprint: None,
            service_tier: None,
        };
        let json = serde_json::to_value(&chunk).unwrap();
        assert_eq!(json["object"], "chat.completion.chunk");
        assert_eq!(json["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(json["choices"][0]["delta"]["content"], "Hello");
        // Absent fields must be omitted, matching OpenAI's wire format.
        assert!(json["choices"][0].get("finish_reason").is_none());
        assert!(json["choices"][0].get("logprobs").is_none());
        assert!(json.get("usage").is_none());
        assert!(json.get("system_fingerprint").is_none());
        assert!(json.get("service_tier").is_none());
    }

    #[test]
    fn empty_delta_omits_role_and_content() {
        let chunk = ChatCompletionChunk {
            id: "rp_chunk_2".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 1,
            model: "m".to_string(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta::default(),
                finish_reason: Some("stop".to_string()),
                logprobs: None,
            }],
            usage: None,
            system_fingerprint: None,
            service_tier: None,
        };
        let json = serde_json::to_value(&chunk).unwrap();
        assert!(json["choices"][0]["delta"].get("role").is_none());
        assert!(json["choices"][0]["delta"].get("content").is_none());
        assert_eq!(json["choices"][0]["finish_reason"], "stop");
    }

    #[test]
    fn content_delta_constructor() {
        let chunk = ChatCompletionChunk::content_delta("id1", "gpt", 5, "world");
        assert_eq!(chunk.object, "chat.completion.chunk");
        assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("world"));
        assert_eq!(chunk.choices[0].delta.role, None);
    }

    #[test]
    fn request_omits_absent_optional_fields() {
        // The threaded optional fields (Task #4) must be omitted from the wire
        // when None, so an OpenAI SDK sees exactly what it sent.
        let req = ChatCompletionRequest {
            model: "gpt-4o".into(),
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
        };
        let v = serde_json::to_value(&req).unwrap();
        for absent in [
            "temperature",
            "top_p",
            "stream",
            "max_tokens",
            "stop",
            "n",
            "presence_penalty",
            "frequency_penalty",
            "user",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
        ] {
            assert!(v.get(absent).is_none(), "{absent} should be omitted");
        }
        assert!(v["messages"][0].get("name").is_none());
        // cache_control must be omitted when None (prompt-caching ship-dark).
        assert!(v["messages"][0].get("cache_control").is_none());
    }

    // --- Prompt-caching passthrough + usage surfacing --------------------------

    #[test]
    fn cache_control_omitted_when_absent_byte_identical() {
        // A message/content-part with no cache_control serializes WITHOUT the key —
        // byte-identical to a pre-caching request (the ship-dark guarantee).
        let msg = Message {
            role: "system".into(),
            content: MessageContent::Parts(vec![ContentPart::Text {
                text: "big preamble".into(),
                cache_control: None,
            }]),
            name: None,
            cache_control: None,
            tool_calls: None,
            tool_call_id: None,
            refusal: None,
            reasoning_content: None,
        };
        let v = serde_json::to_value(&msg).unwrap();
        assert!(v.get("cache_control").is_none());
        assert!(v["content"][0].get("cache_control").is_none());
    }

    #[test]
    fn cache_control_round_trips_on_message_and_part() {
        // The opaque marker survives round-trip on both the message and a text part.
        let raw = serde_json::json!({
            "role": "system",
            "content": [
                {"type": "text", "text": "cached preamble",
                 "cache_control": {"type": "ephemeral"}}
            ],
            "cache_control": {"type": "ephemeral"}
        });
        let msg: Message = serde_json::from_value(raw.clone()).unwrap();
        assert_eq!(
            msg.cache_control,
            Some(serde_json::json!({"type": "ephemeral"}))
        );
        match &msg.content {
            MessageContent::Parts(parts) => match &parts[0] {
                ContentPart::Text {
                    cache_control: Some(cc),
                    ..
                } => assert_eq!(cc, &serde_json::json!({"type": "ephemeral"})),
                _ => panic!("expected a cached text part"),
            },
            _ => panic!("expected Parts"),
        }
        let back = serde_json::to_value(&msg).unwrap();
        assert_eq!(back, raw);
    }

    #[test]
    fn usage_cache_fields_omitted_when_none_byte_identical() {
        // A Usage with no cache info serializes to exactly the legacy 3-field shape.
        let u = Usage {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
            cached_tokens: None,
            cache_creation_tokens: None,
        };
        let v = serde_json::to_value(&u).unwrap();
        assert_eq!(v.as_object().unwrap().len(), 3);
        assert!(v.get("cached_tokens").is_none());
        assert!(v.get("cache_creation_tokens").is_none());
    }

    #[test]
    fn usage_cache_fields_present_when_some() {
        let u = Usage {
            prompt_tokens: 100,
            completion_tokens: 20,
            total_tokens: 120,
            cached_tokens: Some(80),
            cache_creation_tokens: Some(15),
        };
        let v = serde_json::to_value(&u).unwrap();
        assert_eq!(v["cached_tokens"], 80);
        assert_eq!(v["cache_creation_tokens"], 15);
        // Deserializes back (and tolerates absent cache fields too).
        let back: Usage = serde_json::from_value(
            serde_json::json!({"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}),
        )
        .unwrap();
        assert!(back.cached_tokens.is_none());
    }

    // --- Embeddings (PRD-011 §5 FR-1): both input forms + omitted optionals ---

    #[test]
    fn embedding_request_accepts_string_and_array_input() {
        // Bare string -> Single.
        let s: EmbeddingRequest = serde_json::from_value(serde_json::json!({
            "model": "text-embedding-3-small",
            "input": "hello"
        }))
        .unwrap();
        assert!(matches!(s.input, EmbeddingInput::Single(_)));
        assert_eq!(s.input.len(), 1);
        assert_eq!(s.input.to_vec(), vec!["hello".to_string()]);

        // Array -> Batch, order preserved.
        let a: EmbeddingRequest = serde_json::from_value(serde_json::json!({
            "model": "text-embedding-3-small",
            "input": ["a", "b", "c"]
        }))
        .unwrap();
        assert!(matches!(a.input, EmbeddingInput::Batch(_)));
        assert_eq!(
            a.input.to_vec(),
            vec!["a".to_string(), "b".to_string(), "c".to_string()]
        );
    }

    #[test]
    fn embedding_request_omits_absent_optionals_and_single_is_bare_string() {
        let req = EmbeddingRequest {
            model: "text-embedding-3-small".into(),
            input: EmbeddingInput::Single("hi".into()),
            encoding_format: None,
            dimensions: None,
            user: None,
        };
        let v = serde_json::to_value(&req).unwrap();
        // Single serializes as a bare string (OpenAI-compatible), not an array.
        assert_eq!(v["input"], "hi");
        for absent in ["encoding_format", "dimensions", "user"] {
            assert!(v.get(absent).is_none(), "{absent} should be omitted");
        }
    }

    #[test]
    fn embedding_request_round_trips_both_input_forms() {
        for input in [serde_json::json!("solo"), serde_json::json!(["one", "two"])] {
            let original = serde_json::json!({"model": "m", "input": input, "dimensions": 256});
            let parsed: EmbeddingRequest = serde_json::from_value(original.clone()).unwrap();
            let back = serde_json::to_value(&parsed).unwrap();
            assert_eq!(back["input"], original["input"]);
            assert_eq!(back["dimensions"], 256);
        }
    }

    #[test]
    fn embedding_response_serializes_to_openai_shape() {
        let resp = EmbeddingResponse {
            object: "list".into(),
            data: vec![EmbeddingData {
                object: "embedding".into(),
                index: 0,
                embedding: EmbeddingVector::Floats(vec![0.1, 0.2]),
            }],
            model: "text-embedding-3-small".into(),
            usage: EmbeddingUsage {
                prompt_tokens: 4,
                total_tokens: 4,
            },
        };
        let v = serde_json::to_value(&resp).unwrap();
        assert_eq!(v["object"], "list");
        assert_eq!(v["data"][0]["object"], "embedding");
        assert_eq!(v["data"][0]["index"], 0);
        assert_eq!(v["usage"]["prompt_tokens"], 4);
        assert_eq!(v["usage"]["total_tokens"], 4);
        // Round-trip back to the struct.
        let back: EmbeddingResponse = serde_json::from_value(v).unwrap();
        assert_eq!(back.data.len(), 1);
        assert_eq!(back.data[0].index, 0);
        assert_eq!(back.data[0].embedding.len(), 2);
    }

    #[test]
    fn embedding_response_accepts_base64_and_float_forms() {
        // Regression: the openai-python SDK requests `encoding_format:"base64"` BY
        // DEFAULT when numpy is installed, so OpenAI returns each `embedding` as a
        // base64 STRING. Modelling only `Vec<f32>` made this fail the typed decode
        // in `openai.rs`'s `.json::<EmbeddingResponse>()` — a 500 that also tripped
        // the shared circuit breaker. The base64 form must now parse AND round-trip
        // verbatim (byte-faithful passthrough to the client).
        let base64_body = serde_json::json!({
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": "AACAPwAAAEA="}],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 4, "total_tokens": 4}
        });
        let parsed: EmbeddingResponse = serde_json::from_value(base64_body).unwrap();
        assert_eq!(
            parsed.data[0].embedding,
            EmbeddingVector::Base64("AACAPwAAAEA=".to_string())
        );
        assert!(parsed.data[0].embedding.as_floats().is_none());
        let back = serde_json::to_value(&parsed).unwrap();
        assert_eq!(back["data"][0]["embedding"], "AACAPwAAAEA=");

        // The float form still parses to `Floats` and re-serializes to a bare array
        // (byte-identical to the pre-change behaviour — the golden/parity baseline).
        let float_body = serde_json::json!({
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 4, "total_tokens": 4}
        });
        let parsed: EmbeddingResponse = serde_json::from_value(float_body).unwrap();
        assert_eq!(
            parsed.data[0].embedding.as_floats(),
            Some(&[0.1_f32, 0.2][..])
        );
        // Re-serializes to a bare JSON array — byte-identical to what a plain
        // `Vec<f32>` would have emitted before this change (the `f32`→JSON widening
        // is inherent and pre-existing, so compare against that exact form).
        let back = serde_json::to_value(&parsed).unwrap();
        assert_eq!(
            back["data"][0]["embedding"],
            serde_json::to_value(vec![0.1_f32, 0.2]).unwrap()
        );
        assert!(back["data"][0]["embedding"].is_array());
    }

    // --- Rerank (/v1/rerank): request/response (de)serialization --------------

    #[test]
    fn rerank_request_deserializes_minimal_and_defaults_return_documents_false() {
        let r: RerankRequest = serde_json::from_value(serde_json::json!({
            "model": "rerank-v3.5",
            "query": "what is the capital of france?",
            "documents": ["paris is the capital", "berlin is in germany"]
        }))
        .unwrap();
        assert_eq!(r.model, "rerank-v3.5");
        assert_eq!(r.query, "what is the capital of france?");
        assert_eq!(r.documents.len(), 2);
        assert_eq!(r.top_n, None);
        // `return_documents` defaults to false when absent (Cohere/LiteLLM default).
        assert!(!r.return_documents);
    }

    #[test]
    fn rerank_request_round_trips_with_top_n_and_return_documents() {
        let original = serde_json::json!({
            "model": "rerank-v4.0-pro",
            "query": "q",
            "documents": ["a", "b", "c"],
            "top_n": 2,
            "return_documents": true
        });
        let parsed: RerankRequest = serde_json::from_value(original.clone()).unwrap();
        assert_eq!(parsed.top_n, Some(2));
        assert!(parsed.return_documents);
        let back = serde_json::to_value(&parsed).unwrap();
        assert_eq!(back["top_n"], 2);
        assert_eq!(back["return_documents"], true);
        assert_eq!(back["documents"], serde_json::json!(["a", "b", "c"]));
    }

    #[test]
    fn rerank_request_omits_top_n_when_absent() {
        let req = RerankRequest {
            model: "rerank-v3.5".into(),
            query: "q".into(),
            documents: vec!["d".into()],
            top_n: None,
            return_documents: false,
        };
        let v = serde_json::to_value(&req).unwrap();
        assert!(
            v.get("top_n").is_none(),
            "top_n should be omitted when None"
        );
        // return_documents is always serialized (no skip) — minimal, explicit.
        assert_eq!(v["return_documents"], false);
    }

    #[test]
    fn rerank_response_serializes_to_cohere_shape_and_round_trips() {
        let resp = RerankResponse {
            id: Some("rerank-1".into()),
            model: "rerank-v3.5".into(),
            results: vec![
                RerankResult {
                    index: 0,
                    relevance_score: 0.99,
                    document: None,
                },
                RerankResult {
                    index: 1,
                    relevance_score: 0.12,
                    document: Some(RerankDocument {
                        text: "berlin is in germany".into(),
                    }),
                },
            ],
            usage: RerankUsage {
                search_units: 1,
                total_tokens: 1,
            },
        };
        let v = serde_json::to_value(&resp).unwrap();
        assert_eq!(v["id"], "rerank-1");
        assert_eq!(v["results"][0]["index"], 0);
        assert_eq!(v["results"][0]["relevance_score"], 0.99);
        // document omitted on result[0] (return_documents not requested for it).
        assert!(v["results"][0].get("document").is_none());
        assert_eq!(v["results"][1]["document"]["text"], "berlin is in germany");
        assert_eq!(v["usage"]["search_units"], 1);
        // Round-trip back to the struct.
        let back: RerankResponse = serde_json::from_value(v).unwrap();
        assert_eq!(back.results.len(), 2);
        assert_eq!(back.results[0].index, 0);
        assert_eq!(
            back.results[1].document.as_ref().unwrap().text,
            "berlin is in germany"
        );
    }

    #[test]
    fn rerank_response_omits_id_when_absent() {
        let resp = RerankResponse {
            id: None,
            model: "rerank-v3.5".into(),
            results: vec![],
            usage: RerankUsage {
                search_units: 0,
                total_tokens: 0,
            },
        };
        let v = serde_json::to_value(&resp).unwrap();
        assert!(v.get("id").is_none(), "id should be omitted when None");
    }

    // --- Moderation (/v1/moderations) ------------------------------------------

    #[test]
    fn moderation_request_accepts_single_and_batch_input() {
        let single: ModerationRequest = serde_json::from_value(serde_json::json!({
            "input": "is this ok?",
            "model": "omni-moderation-latest"
        }))
        .unwrap();
        assert!(matches!(single.input, ModerationInput::Single(_)));
        assert_eq!(single.input.len(), 1);
        assert_eq!(single.input.to_vec(), vec!["is this ok?".to_string()]);
        assert_eq!(single.model.as_deref(), Some("omni-moderation-latest"));

        let batch: ModerationRequest = serde_json::from_value(serde_json::json!({
            "input": ["a", "b", "c"]
        }))
        .unwrap();
        assert!(matches!(batch.input, ModerationInput::Batch(_)));
        assert_eq!(batch.input.len(), 3);
        assert!(batch.model.is_none(), "model is optional");
    }

    #[test]
    fn moderation_request_omits_model_when_absent() {
        let req = ModerationRequest {
            input: ModerationInput::Single("hi".into()),
            model: None,
        };
        let v = serde_json::to_value(&req).unwrap();
        assert!(
            v.get("model").is_none(),
            "model should be omitted when None"
        );
        assert_eq!(v["input"], "hi");
    }

    #[test]
    fn moderation_response_serializes_to_openai_shape() {
        let mut categories = BTreeMap::new();
        categories.insert("hate".to_string(), true);
        categories.insert("violence".to_string(), false);
        let mut category_scores = BTreeMap::new();
        category_scores.insert("hate".to_string(), 0.91_f64);
        category_scores.insert("violence".to_string(), 0.02_f64);

        let resp = ModerationResponse {
            id: "modr-abc".into(),
            model: "omni-moderation-latest".into(),
            results: vec![ModerationResult {
                flagged: true,
                categories,
                category_scores,
            }],
        };
        let v = serde_json::to_value(&resp).unwrap();
        assert_eq!(v["id"], "modr-abc");
        assert_eq!(v["model"], "omni-moderation-latest");
        assert_eq!(v["results"][0]["flagged"], true);
        assert_eq!(v["results"][0]["categories"]["hate"], true);
        assert_eq!(v["results"][0]["categories"]["violence"], false);
        assert_eq!(v["results"][0]["category_scores"]["hate"], 0.91);

        // Round-trips losslessly.
        let back: ModerationResponse = serde_json::from_value(v).unwrap();
        assert!(back.results[0].flagged);
        assert_eq!(back.results[0].categories.get("hate"), Some(&true));
    }

    // --- Image generation (/v1/images/generations) -----------------------------

    #[test]
    fn image_request_only_prompt_required_and_omits_absent_fields() {
        let req: ImageGenerationRequest = serde_json::from_value(serde_json::json!({
            "prompt": "a red panda"
        }))
        .unwrap();
        assert_eq!(req.prompt, "a red panda");
        assert!(req.model.is_none());
        assert!(req.n.is_none());
        assert!(req.size.is_none());
        assert!(req.quality.is_none());
        assert!(req.response_format.is_none());
        assert!(req.extra.is_empty());

        // Absent optionals must not appear on the wire (so a default-fill at the
        // adapter is honored upstream).
        let v = serde_json::to_value(&req).unwrap();
        assert_eq!(v["prompt"], "a red panda");
        assert!(v.get("model").is_none());
        assert!(v.get("n").is_none());
        assert!(v.get("size").is_none());
        assert!(v.get("quality").is_none());
        assert!(v.get("response_format").is_none());
    }

    #[test]
    fn image_request_threads_all_known_fields_and_extras() {
        let req: ImageGenerationRequest = serde_json::from_value(serde_json::json!({
            "model": "dall-e-3",
            "prompt": "a city skyline",
            "n": 2,
            "size": "1024x1024",
            "quality": "hd",
            "response_format": "b64_json",
            "style": "vivid",
            "user": "u_42"
        }))
        .unwrap();
        assert_eq!(req.model.as_deref(), Some("dall-e-3"));
        assert_eq!(req.n, Some(2));
        assert_eq!(req.size.as_deref(), Some("1024x1024"));
        assert_eq!(req.quality.as_deref(), Some("hd"));
        assert_eq!(req.response_format.as_deref(), Some("b64_json"));
        // Forward-compatible OpenAI fields are preserved, not dropped.
        assert_eq!(
            req.extra.get("style").and_then(|v| v.as_str()),
            Some("vivid")
        );
        assert_eq!(req.extra.get("user").and_then(|v| v.as_str()), Some("u_42"));

        // They survive the round-trip back onto the wire (flattened, not nested).
        let v = serde_json::to_value(&req).unwrap();
        assert_eq!(v["model"], "dall-e-3");
        assert_eq!(v["n"], 2);
        assert_eq!(v["style"], "vivid");
        assert_eq!(v["user"], "u_42");
        assert!(
            v.get("extra").is_none(),
            "extra must be flattened, not nested"
        );
    }

    #[test]
    fn image_response_url_variant_serializes_to_openai_shape() {
        let resp: ImageGenerationResponse = serde_json::from_value(serde_json::json!({
            "created": 1_700_000_000i64,
            "data": [
                { "url": "https://example.com/img.png", "revised_prompt": "a vivid city skyline" }
            ]
        }))
        .unwrap();
        assert_eq!(resp.created, 1_700_000_000);
        assert_eq!(resp.data.len(), 1);
        assert_eq!(
            resp.data[0].url.as_deref(),
            Some("https://example.com/img.png")
        );
        assert!(resp.data[0].b64_json.is_none());
        assert_eq!(
            resp.data[0].revised_prompt.as_deref(),
            Some("a vivid city skyline")
        );
        assert!(resp.usage.is_none());

        let v = serde_json::to_value(&resp).unwrap();
        assert_eq!(v["created"], 1_700_000_000i64);
        assert_eq!(v["data"][0]["url"], "https://example.com/img.png");
        // The absent b64 field must not be emitted (so the wire matches OpenAI).
        assert!(v["data"][0].get("b64_json").is_none());
        assert!(v.get("usage").is_none());
    }

    #[test]
    fn image_response_b64_variant_with_usage_round_trips() {
        // gpt-image-1 returns b64_json and may include a top-level usage block —
        // tolerate it without requiring it.
        let resp: ImageGenerationResponse = serde_json::from_value(serde_json::json!({
            "created": 1_700_000_001i64,
            "data": [ { "b64_json": "aGVsbG8=" } ],
            "usage": { "total_tokens": 1234, "input_tokens": 10 }
        }))
        .unwrap();
        assert_eq!(resp.data[0].b64_json.as_deref(), Some("aGVsbG8="));
        assert!(resp.data[0].url.is_none());
        assert!(resp.usage.is_some());

        let v = serde_json::to_value(&resp).unwrap();
        assert_eq!(v["data"][0]["b64_json"], "aGVsbG8=");
        assert!(v["data"][0].get("url").is_none());
        assert_eq!(v["usage"]["total_tokens"], 1234);

        // Round-trips losslessly.
        let back: ImageGenerationResponse = serde_json::from_value(v).unwrap();
        assert_eq!(back.data[0].b64_json.as_deref(), Some("aGVsbG8="));
    }

    // --- Multimodal content (PRD-011 multimodal passthrough) -------------------

    #[test]
    fn plain_string_content_deserializes_to_text_variant() {
        // Backward compat: existing `{"content": "hi"}` payloads still work.
        let msg: Message = serde_json::from_value(serde_json::json!({
            "role": "user",
            "content": "hello world"
        }))
        .unwrap();
        assert_eq!(msg.content, MessageContent::Text("hello world".into()));
        assert_eq!(msg.content.as_text(), "hello world");
        assert!(!msg.content.has_images());
    }

    #[test]
    fn plain_string_content_serializes_as_bare_string() {
        // Byte-level backward compat: Text variant serializes as a bare string.
        let msg = Message {
            role: "user".into(),
            content: MessageContent::Text("hi".into()),
            name: None,
            cache_control: None,
            tool_calls: None,
            tool_call_id: None,
            refusal: None,
            reasoning_content: None,
        };
        let v = serde_json::to_value(&msg).unwrap();
        assert_eq!(v["content"], "hi"); // bare string, not an array
    }

    #[test]
    fn multimodal_array_content_deserializes_to_parts() {
        let msg: Message = serde_json::from_value(serde_json::json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "describe this image"},
                {"type": "image_url", "image_url": {"url": "https://example.com/img.png", "detail": "high"}}
            ]
        }))
        .unwrap();
        match &msg.content {
            MessageContent::Parts(parts) => {
                assert_eq!(parts.len(), 2);
                assert!(
                    matches!(&parts[0], ContentPart::Text { text, .. } if text == "describe this image")
                );
                match &parts[1] {
                    ContentPart::ImageUrl { image_url } => {
                        assert_eq!(image_url.url, "https://example.com/img.png");
                        assert_eq!(image_url.detail.as_deref(), Some("high"));
                    }
                    _ => panic!("expected ImageUrl part"),
                }
            }
            _ => panic!("expected Parts variant"),
        }
        assert_eq!(msg.content.as_text(), "describe this image");
        assert!(msg.content.has_images());
    }

    #[test]
    fn multimodal_content_round_trips() {
        let original = serde_json::json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
            ]
        });
        let msg: Message = serde_json::from_value(original.clone()).unwrap();
        let back = serde_json::to_value(&msg).unwrap();
        assert_eq!(back["role"], "user");
        assert_eq!(back["content"][0]["type"], "text");
        assert_eq!(back["content"][0]["text"], "look");
        assert_eq!(back["content"][1]["type"], "image_url");
        assert_eq!(
            back["content"][1]["image_url"]["url"],
            "data:image/png;base64,abc"
        );
    }

    #[test]
    fn from_str_impl_enables_existing_construction_patterns() {
        // `content: "hi".into()` must still compile and produce Text variant.
        let msg = Message {
            role: "user".into(),
            content: "hello".into(),
            name: None,
            cache_control: None,
            tool_calls: None,
            tool_call_id: None,
            refusal: None,
            reasoning_content: None,
        };
        assert_eq!(msg.content.as_text(), "hello");
    }

    #[test]
    fn as_text_concatenates_multiple_text_parts() {
        let content = MessageContent::Parts(vec![
            ContentPart::Text {
                text: "first ".into(),
                cache_control: None,
            },
            ContentPart::ImageUrl {
                image_url: ImageUrlContent {
                    url: "https://example.com/img.png".into(),
                    detail: None,
                },
            },
            ContentPart::Text {
                text: "second".into(),
                cache_control: None,
            },
        ]);
        // Text parts concatenated in order; image parts skipped.
        assert_eq!(content.as_text(), "first second");
        assert!(content.has_images());
    }

    // --- Tool / function calling (OpenAI tool calling) -------------------------

    #[test]
    fn request_with_tools_round_trips_openai_shape() {
        // A request carrying `tools` + `tool_choice` + `parallel_tool_calls`
        // must round-trip byte-for-byte (no field dropped at the type layer).
        let raw = serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "weather in SF?"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"]
                    }
                }
            }],
            "tool_choice": "auto",
            "parallel_tool_calls": true
        });
        let req: ChatCompletionRequest = serde_json::from_value(raw.clone()).unwrap();
        let tools = req.tools.as_ref().expect("tools present");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].tool_type, "function");
        assert_eq!(tools[0].function.name, "get_weather");
        assert!(tools[0].function.parameters.is_some());
        assert_eq!(req.tool_choice, Some(serde_json::json!("auto")));
        assert_eq!(req.parallel_tool_calls, Some(true));

        let back = serde_json::to_value(&req).unwrap();
        assert_eq!(back["tools"], raw["tools"]);
        assert_eq!(back["tool_choice"], "auto");
        assert_eq!(back["parallel_tool_calls"], true);
    }

    #[test]
    fn tool_choice_accepts_object_form() {
        // tool_choice can be an OBJECT (force a specific function) — preserved
        // verbatim as a Value, not coerced to a string.
        let raw = serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}}
        });
        let req: ChatCompletionRequest = serde_json::from_value(raw.clone()).unwrap();
        assert_eq!(req.tool_choice, Some(raw["tool_choice"].clone()));
        let back = serde_json::to_value(&req).unwrap();
        assert_eq!(back["tool_choice"], raw["tool_choice"]);
    }

    #[test]
    fn response_message_with_tool_calls_round_trips() {
        // An assistant response message carrying tool_calls (content null) must
        // round-trip; arguments stays a JSON-ENCODED STRING per OpenAI.
        let raw = serde_json::json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_abc",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"location\":\"SF\"}"}
            }]
        });
        let msg: Message = serde_json::from_value(raw.clone()).unwrap();
        let calls = msg.tool_calls.as_ref().expect("tool_calls present");
        assert_eq!(calls[0].id, "call_abc");
        assert_eq!(calls[0].tool_type, "function");
        assert_eq!(calls[0].function.name, "get_weather");
        // arguments is the raw JSON string (callers json.loads() it).
        assert_eq!(calls[0].function.arguments, "{\"location\":\"SF\"}");
        let back = serde_json::to_value(&msg).unwrap();
        assert_eq!(back["tool_calls"], raw["tool_calls"]);
    }

    #[test]
    fn tool_role_result_message_round_trips() {
        // The tool-result turn: role="tool" + tool_call_id matching the call.
        let raw = serde_json::json!({
            "role": "tool",
            "content": "{\"temp\":21}",
            "tool_call_id": "call_abc"
        });
        let msg: Message = serde_json::from_value(raw.clone()).unwrap();
        assert_eq!(msg.role, "tool");
        assert_eq!(msg.tool_call_id.as_deref(), Some("call_abc"));
        let back = serde_json::to_value(&msg).unwrap();
        assert_eq!(back["tool_call_id"], "call_abc");
        // No spurious tool_calls key on a tool-result message.
        assert!(back.get("tool_calls").is_none());
    }

    #[test]
    fn message_without_tool_fields_is_byte_identical() {
        // A plain message emits NO tool_calls / tool_call_id keys — the parity
        // guarantee (golden/ab_parity corpus has no tools).
        let msg = Message {
            role: "user".into(),
            content: "hi".into(),
            name: None,
            cache_control: None,
            tool_calls: None,
            tool_call_id: None,
            refusal: None,
            reasoning_content: None,
        };
        let v = serde_json::to_value(&msg).unwrap();
        assert!(v.get("tool_calls").is_none());
        assert!(v.get("tool_call_id").is_none());
    }

    #[test]
    fn streaming_delta_with_partial_tool_call_round_trips() {
        // OpenAI streams tool calls incrementally by index: the first delta
        // carries id/type/name, later deltas append arguments fragments.
        let first = serde_json::json!({
            "tool_calls": [{
                "index": 0,
                "id": "call_abc",
                "type": "function",
                "function": {"name": "get_weather", "arguments": ""}
            }]
        });
        let d: Delta = serde_json::from_value(first.clone()).unwrap();
        let tc = d.tool_calls.as_ref().expect("tool_calls present");
        assert_eq!(tc[0].index, 0);
        assert_eq!(tc[0].id.as_deref(), Some("call_abc"));
        assert_eq!(tc[0].tool_type.as_deref(), Some("function"));
        assert_eq!(
            tc[0].function.as_ref().unwrap().name.as_deref(),
            Some("get_weather")
        );
        let back = serde_json::to_value(&d).unwrap();
        assert_eq!(back["tool_calls"], first["tool_calls"]);

        // A later fragment: only index + an arguments piece.
        let frag = serde_json::json!({
            "tool_calls": [{"index": 0, "function": {"arguments": "{\"loc"}}]
        });
        let d2: Delta = serde_json::from_value(frag.clone()).unwrap();
        let tc2 = d2.tool_calls.as_ref().unwrap();
        assert_eq!(tc2[0].index, 0);
        assert!(tc2[0].id.is_none());
        assert_eq!(
            tc2[0].function.as_ref().unwrap().arguments.as_deref(),
            Some("{\"loc")
        );
        let back2 = serde_json::to_value(&d2).unwrap();
        // Absent id/type/name omitted on the wire (byte-minimal fragments).
        assert!(back2["tool_calls"][0].get("id").is_none());
        assert!(back2["tool_calls"][0].get("type").is_none());
        assert!(back2["tool_calls"][0]["function"].get("name").is_none());
    }

    #[test]
    fn content_only_delta_omits_tool_calls() {
        // A content-only delta must NOT emit a tool_calls key (parity guarantee).
        let d = Delta {
            role: None,
            content: Some("hi".into()),
            tool_calls: None,
            refusal: None,
            reasoning_content: None,
        };
        let v = serde_json::to_value(&d).unwrap();
        assert!(v.get("tool_calls").is_none());
        // The new passthrough fields must be omitted too (parity guarantee).
        assert!(v.get("refusal").is_none());
        assert!(v.get("reasoning_content").is_none());
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    // Strategy producing an arbitrary canonical request with the new fields set
    // to arbitrary Some/None values.
    prop_compose! {
        fn arb_request()(
            model in "[a-zA-Z0-9._-]{1,20}",
            role in prop::sample::select(vec!["system", "user", "assistant"]),
            content in ".{0,64}",
            name in proptest::option::of("[a-zA-Z ]{1,12}"),
            temperature in proptest::option::of(0.0f32..2.0),
            top_p in proptest::option::of(0.0f32..1.0),
            stream in proptest::option::of(any::<bool>()),
            max_tokens in proptest::option::of(1u32..8192),
            stop in proptest::option::of(prop::collection::vec("[a-z]{1,5}", 1..3)),
            n in proptest::option::of(1u32..4),
            presence_penalty in proptest::option::of(-2.0f32..2.0),
            frequency_penalty in proptest::option::of(-2.0f32..2.0),
            user in proptest::option::of("[a-z0-9]{1,10}"),
        ) -> ChatCompletionRequest {
            ChatCompletionRequest {
                model,
                messages: vec![Message { role: role.to_string(), content: content.into(), name, cache_control: None, tool_calls: None, tool_call_id: None, refusal: None, reasoning_content: None }],
                temperature, top_p, stream, max_tokens, stop, n,
                presence_penalty, frequency_penalty, user,
                tools: None, tool_choice: None, parallel_tool_calls: None,
                ..Default::default()
            }
        }
    }

    proptest! {
        // Round-trip: any canonical request serializes and deserializes back to
        // an equivalent value — no field is silently dropped at the type layer
        // (Task #4). This is the type-level guard; per-adapter mapping is covered
        // by the wiremock adapter tests.
        #[test]
        fn request_json_round_trips(req in arb_request()) {
            let json = serde_json::to_string(&req).unwrap();
            let back: ChatCompletionRequest = serde_json::from_str(&json).unwrap();
            let back_json = serde_json::to_string(&back).unwrap();
            prop_assert_eq!(json, back_json);
        }
    }
}
