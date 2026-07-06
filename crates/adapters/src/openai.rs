use crate::sse::SseLineBuffer;
use crate::SpeechAudio;
use crate::{ChunkStream, Provider, ProviderError};
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use routeplane_types::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, EmbeddingRequest,
    EmbeddingResponse, ImageGenerationRequest, ImageGenerationResponse, ModerationRequest,
    ModerationResponse, SpeechRequest, TranscriptionInput, TranscriptionResponse,
};
use serde_json::json;

pub struct OpenAIProvider {
    client: Client,
    /// Base URL for the chat-completions endpoint. Defaults to the public OpenAI
    /// API; overridable so wiremock-backed tests can point the adapter at a mock
    /// server (engineering-design §24 / Task #4) without touching the hot path.
    base_url: String,
}

const OPENAI_DEFAULT_BASE_URL: &str = "https://api.openai.com";

impl OpenAIProvider {
    pub fn new() -> Self {
        Self {
            client: crate::client::build_provider_client(),
            base_url: OPENAI_DEFAULT_BASE_URL.to_string(),
        }
    }

    /// Construct pointing at a custom base URL (for tests / self-hosted proxies).
    pub fn with_base_url(base_url: impl Into<String>) -> Self {
        Self {
            client: crate::client::build_provider_client(),
            base_url: base_url.into(),
        }
    }
}

impl Default for OpenAIProvider {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse one OpenAI SSE `data:` payload into a canonical chunk.
///
/// OpenAI streaming chunks are already `chat.completion.chunk` objects, so this
/// is largely a passthrough deserialize. Returns:
///   * `Ok(None)`  — the `[DONE]` sentinel (caller should end the stream),
///   * `Ok(Some)`  — a parsed chunk,
///   * `Err(..)`   — malformed JSON.
///
/// Kept as a free function (not a method) so it is trivially unit-testable
/// without a live HTTP connection. Azure OpenAI reuses it verbatim.
pub(crate) fn parse_openai_sse_payload(
    payload: &str,
) -> Result<Option<ChatCompletionChunk>, ProviderError> {
    if payload == "[DONE]" {
        return Ok(None);
    }
    // Deserialize via a Value first so we can lift OpenAI's nested
    // `usage.prompt_tokens_details.cached_tokens` into the canonical flat
    // `usage.cached_tokens` (prompt-caching surfacing). A non-cached chunk is
    // untouched, so the wire stays byte-identical. Cheap: usage rides only the
    // final chunk; content chunks have no `usage` key and skip the lift.
    let mut v: serde_json::Value = serde_json::from_str(payload).map_err(|e| -> ProviderError {
        format!("OpenAI stream parse error: {e}: {payload}").into()
    })?;
    lift_openai_cached_tokens(&mut v);
    let chunk: ChatCompletionChunk = serde_json::from_value(v).map_err(|e| -> ProviderError {
        format!("OpenAI stream parse error: {e}: {payload}").into()
    })?;
    Ok(Some(chunk))
}

/// Lift OpenAI's nested `usage.prompt_tokens_details.cached_tokens` into the
/// canonical flat `usage.cached_tokens` on a response/chunk JSON value, in place.
/// OpenAI counts cached tokens as a SUBSET of `prompt_tokens` and reports the
/// breakdown under `prompt_tokens_details`; the canonical `Usage` surfaces that
/// subset directly. No-op when there is no `usage` block or no cached-token
/// detail (a non-cached response), so the mapping is byte-identical when caching
/// did not occur. OpenAI reports no cache-CREATION count (its cache is automatic),
/// so `cache_creation_tokens` stays absent.
pub(crate) fn lift_openai_cached_tokens(v: &mut serde_json::Value) {
    let Some(cached) = v
        .get("usage")
        .and_then(|u| u.get("prompt_tokens_details"))
        .and_then(|d| d.get("cached_tokens"))
        .and_then(|c| c.as_u64())
    else {
        return;
    };
    if let Some(usage) = v.get_mut("usage").and_then(|u| u.as_object_mut()) {
        usage.insert("cached_tokens".to_string(), json!(cached));
    }
}

/// Strip gateway-internal / response-only message fields from an outbound
/// OpenAI-dialect request body, in place:
///
/// - `cache_control` — an Anthropic-only passthrough; OpenAI caches
///   automatically and rejects unknown fields on some endpoints, so it MUST
///   NOT leak into the OpenAI request (stripped from messages AND array-form
///   content parts).
/// - `reasoning_content` / `refusal` — RESPONSE-only fields on the canonical
///   `Message`. The standard client pattern appends `choices[0].message` to
///   the next request's history, which would round-trip them upstream;
///   DeepSeek documents a **400** when `reasoning_content` appears in input
///   messages, and other providers likewise reject them as input. Stripped
///   from every outbound request message.
///
/// A request carrying none of these is untouched (byte-identical). Shared by
/// every OpenAI-wire adapter (Azure, self-hosted, Groq, DeepSeek,
/// Mistral-as-OpenAI, Bedrock, ...) via [`strip_cache_control_for_openai`].
pub(crate) fn strip_cache_control_for_openai(body: &mut serde_json::Value) {
    let Some(messages) = body.get_mut("messages").and_then(|m| m.as_array_mut()) else {
        return;
    };
    for msg in messages.iter_mut() {
        if let Some(obj) = msg.as_object_mut() {
            obj.remove("cache_control");
            // Response-only fields must never be echoed upstream as input.
            obj.remove("reasoning_content");
            obj.remove("refusal");
        }
        // Strip from array-form content parts too (a per-part marker).
        if let Some(parts) = msg.get_mut("content").and_then(|c| c.as_array_mut()) {
            for part in parts.iter_mut() {
                if let Some(p) = part.as_object_mut() {
                    p.remove("cache_control");
                }
            }
        }
    }
}

#[async_trait]
impl Provider for OpenAIProvider {
    fn name(&self) -> &'static str {
        "openai"
    }

    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
        api_key: String,
    ) -> Result<ChatCompletionResponse, ProviderError> {
        let url = format!("{}/v1/chat/completions", self.base_url);

        // Serialize then strip the Anthropic-only `cache_control` marker — OpenAI
        // caches automatically and rejects unknown fields, so it must NOT leak
        // upstream. A request without `cache_control` serializes byte-identically
        // (the strip is a no-op), so parity/golden are unchanged.
        let mut body = serde_json::to_value(&request)?;
        strip_cache_control_for_openai(&mut body);

        let response = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {api_key}"))
            .json(&body)
            .send()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("openai", e))?;

        // Non-success → typed, status-classified error (G2.3) so the retry loop
        // can decide on `on_status` rather than string-sniffing.
        if !response.status().is_success() {
            return Err(crate::client::error_from_response("openai", response).await);
        }

        // Deserialize via a Value so we can lift OpenAI's nested
        // `usage.prompt_tokens_details.cached_tokens` into the canonical flat
        // `usage.cached_tokens` (prompt-caching surfacing). A non-cached response
        // is untouched.
        let mut raw: serde_json::Value = response
            .json()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("openai", e))?;
        lift_openai_cached_tokens(&mut raw);
        serde_json::from_value::<ChatCompletionResponse>(raw)
            .map_err(|e| ProviderError::translation(format!("openai response parse error: {e}")))
    }

    async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
        api_key: String,
    ) -> Result<ChunkStream, ProviderError> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        // We serialize the canonical request and force `stream: true`, plus ask
        // for usage on the final chunk (OpenAI omits usage from streams unless
        // `stream_options.include_usage` is set) so observability gets real
        // token counts.
        let mut body = serde_json::to_value(&request)?;
        body["stream"] = json!(true);
        body["stream_options"] = json!({ "include_usage": true });
        // Strip the Anthropic-only cache marker before egress (OpenAI rejects it).
        strip_cache_control_for_openai(&mut body);

        let resp = crate::client::streaming_client()
            .post(url)
            .header("Authorization", format!("Bearer {api_key}"))
            .json(&body)
            .send()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("openai", e))?;

        // Establishment failure -> typed Err so the proxy can retry/fall back.
        if !resp.status().is_success() {
            return Err(crate::client::error_from_response("openai", resp).await);
        }

        Ok(Box::pin(openai_sse_to_chunks(resp.bytes_stream())))
    }

    async fn embeddings(
        &self,
        request: EmbeddingRequest,
        api_key: String,
    ) -> Result<EmbeddingResponse, ProviderError> {
        // OpenAI's /v1/embeddings IS the canonical dialect: the request and the
        // response serialize 1:1, so this is a typed passthrough (no translation).
        let url = format!("{}/v1/embeddings", self.base_url);

        let response = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {api_key}"))
            .json(&request)
            .send()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("openai", e))?;

        if !response.status().is_success() {
            return Err(crate::client::error_from_response("openai", response).await);
        }

        response
            .json::<EmbeddingResponse>()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("openai", e))
    }

    async fn moderations(
        &self,
        mut request: ModerationRequest,
        api_key: String,
    ) -> Result<ModerationResponse, ProviderError> {
        // OpenAI's /v1/moderations IS the canonical dialect: `{input, model}` in,
        // `{id, model, results:[{flagged, categories, category_scores}]}` out —
        // both serialize 1:1 to the canonical types, so this is a typed
        // passthrough (no translation). OpenAI requires a `model`; default to
        // `omni-moderation-latest` when the caller omitted it.
        if request.model.is_none() {
            request.model = Some(OPENAI_DEFAULT_MODERATION_MODEL.to_string());
        }
        let url = format!("{}/v1/moderations", self.base_url);

        let response = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {api_key}"))
            .json(&request)
            .send()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("openai", e))?;

        if !response.status().is_success() {
            return Err(crate::client::error_from_response("openai", response).await);
        }

        response
            .json::<ModerationResponse>()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("openai", e))
    }

    async fn transcribe(
        &self,
        audio: TranscriptionInput,
        api_key: String,
    ) -> Result<TranscriptionResponse, ProviderError> {
        let url = format!("{}/v1/audio/transcriptions", self.base_url);
        transcribe_multipart(&self.client, "openai", &url, audio, api_key).await
    }

    async fn translate(
        &self,
        audio: TranscriptionInput,
        api_key: String,
    ) -> Result<TranscriptionResponse, ProviderError> {
        // OpenAI's /v1/audio/translations is the near-twin of transcriptions:
        // same multipart contract, but the output is always English and there is
        // NO `language` field. `whisper-1` supports translations (the gpt-4o-
        // transcribe models do not). The shared helper omits `language`.
        let url = format!("{}/v1/audio/translations", self.base_url);
        translate_multipart(&self.client, "openai", &url, audio, api_key).await
    }

    async fn image_generation(
        &self,
        mut request: ImageGenerationRequest,
        api_key: String,
    ) -> Result<ImageGenerationResponse, ProviderError> {
        // OpenAI's /v1/images/generations IS the canonical dialect: the request
        // and response serialize 1:1 to the canonical types, so this is a typed
        // passthrough (no translation). OpenAI requires a `model`; default to
        // `gpt-image-1` when the caller omitted it so a bare
        // `{"prompt": "..."}` works like OpenAI's. The response can be a large
        // b64 body (gpt-image-1) — reqwest buffers it; we never log it.
        if request.model.is_none() {
            request.model = Some(OPENAI_DEFAULT_IMAGE_MODEL.to_string());
        }
        let url = format!("{}/v1/images/generations", self.base_url);

        let response = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {api_key}"))
            .json(&request)
            .send()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("openai", e))?;

        if !response.status().is_success() {
            return Err(crate::client::error_from_response("openai", response).await);
        }

        response
            .json::<ImageGenerationResponse>()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("openai", e))
    }

    async fn speech(
        &self,
        mut request: SpeechRequest,
        api_key: String,
    ) -> Result<SpeechAudio, ProviderError> {
        // OpenAI's /v1/audio/speech IS the canonical dialect: `{model, input,
        // voice, response_format?, speed?}` in, BINARY audio bytes out (NOT JSON).
        // The request serializes 1:1 to the canonical type, so this is a typed
        // passthrough. OpenAI requires a `model`; default to `gpt-4o-mini-tts`
        // when the caller omitted it so a bare `{"input": "...", "voice": "..."}`
        // works. The response body is raw audio — reqwest buffers it; we NEVER log
        // it or the api_key.
        if request.model.is_none() {
            request.model = Some(OPENAI_DEFAULT_TTS_MODEL.to_string());
        }
        // Capture the requested format so we can derive the Content-Type as a
        // fallback if the upstream omits the header (it normally sends one).
        let requested_format = request.response_format.clone();
        let url = format!("{}/v1/audio/speech", self.base_url);

        let response = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {api_key}"))
            .json(&request)
            .send()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("openai", e))?;

        if !response.status().is_success() {
            return Err(crate::client::error_from_response("openai", response).await);
        }

        // Prefer the upstream Content-Type; fall back to deriving it from the
        // requested response_format (default mp3 ⇒ audio/mpeg).
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string())
            .unwrap_or_else(|| content_type_for_format(requested_format.as_deref()).to_string());

        let bytes = response
            .bytes()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("openai", e))?;

        Ok(SpeechAudio {
            bytes: bytes.to_vec(),
            content_type,
        })
    }
}

/// OpenAI's current default TTS model (verified contract). Applied when the
/// caller omits `model`, so a bare `{"input": "...", "voice": "..."}` works.
const OPENAI_DEFAULT_TTS_MODEL: &str = "gpt-4o-mini-tts";

/// Map a TTS `response_format` to the HTTP `Content-Type` to return when the
/// upstream omits the header. The default format is mp3 (⇒ `audio/mpeg`), per
/// OpenAI's verified contract:
///   mp3→audio/mpeg, opus→audio/ogg, aac→audio/aac, flac→audio/flac,
///   wav→audio/wav, pcm→audio/pcm.
pub(crate) fn content_type_for_format(response_format: Option<&str>) -> &'static str {
    match response_format.unwrap_or("mp3") {
        "opus" => "audio/ogg",
        "aac" => "audio/aac",
        "flac" => "audio/flac",
        "wav" => "audio/wav",
        "pcm" => "audio/pcm",
        // mp3 (default) and any unrecognised format fall back to mp3's type.
        _ => "audio/mpeg",
    }
}

/// OpenAI's current default moderation model (verified contract). Applied when
/// the caller omits `model`, so a bare `{"input": "..."}` works like OpenAI's.
const OPENAI_DEFAULT_MODERATION_MODEL: &str = "omni-moderation-latest";

/// OpenAI's current default image model (verified contract). Applied when the
/// caller omits `model`, so a bare `{"prompt": "..."}` works like OpenAI's.
const OPENAI_DEFAULT_IMAGE_MODEL: &str = "gpt-image-1";

/// Translate an OpenAI/Azure `text/event-stream` byte stream into canonical
/// chunks. Shared by both adapters (Azure OpenAI is wire-identical).
pub(crate) fn openai_sse_to_chunks(
    mut bytes: impl futures::Stream<Item = reqwest::Result<bytes::Bytes>> + Unpin + Send + 'static,
) -> impl futures::Stream<Item = Result<ChatCompletionChunk, ProviderError>> + Send + 'static {
    async_stream::stream! {
        let mut sse = SseLineBuffer::new();
        while let Some(item) = bytes.next().await {
            let chunk = match item {
                Ok(b) => b,
                // Mid-stream transport error: surface as an in-stream Err and stop.
                Err(e) => { yield Err(format!("OpenAI stream transport error: {e}").into()); break; }
            };
            sse.push(&chunk);
            while let Some(payload) = sse.next_payload() {
                match parse_openai_sse_payload(&payload) {
                    Ok(Some(c)) => yield Ok(c),
                    Ok(None) => return, // [DONE]
                    Err(e) => { yield Err(e); return; }
                }
            }
        }
    }
}

/// Best-effort audio MIME type from a filename extension. OpenAI/Groq sniff the
/// codec from the multipart filename extension primarily, but sending a correct
/// `Content-Type` part is good hygiene. Unknown extensions fall back to the
/// generic `application/octet-stream` (the upstream still uses the filename).
fn audio_mime_from_filename(filename: &str) -> &'static str {
    let ext = filename
        .rsplit('.')
        .next()
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "flac" => "audio/flac",
        "mp3" | "mpga" | "mpeg" => "audio/mpeg",
        "mp4" | "m4a" => "audio/mp4",
        "ogg" => "audio/ogg",
        "wav" => "audio/wav",
        "webm" => "audio/webm",
        _ => "application/octet-stream",
    }
}

/// POST a `multipart/form-data` transcription request to an OpenAI-dialect
/// `/v1/audio/transcriptions` endpoint and map the response to the canonical
/// `{text}` shape. Shared by the OpenAI and Groq adapters (Groq's STT surface is
/// wire-identical — only the base path differs, which the caller bakes into
/// `url`). Thin wrapper over [`audio_multipart`] with `include_language = true`.
pub(crate) async fn transcribe_multipart(
    client: &Client,
    provider: &'static str,
    url: &str,
    audio: TranscriptionInput,
    api_key: String,
) -> Result<TranscriptionResponse, ProviderError> {
    audio_multipart(client, provider, url, audio, api_key, true).await
}

/// POST a `multipart/form-data` translation request to an OpenAI-dialect
/// `/v1/audio/translations` endpoint (speech-in-any-language → English text) and
/// map the response to the canonical `{text}` shape. Shared by the OpenAI and
/// Groq adapters (the surface is wire-identical to transcriptions — only the base
/// path differs, baked into `url`). Thin wrapper over [`audio_multipart`] with
/// `include_language = false`: the translations contract has NO `language` field
/// (the output is always English), so even if `TranscriptionParams::language` is
/// populated it is deliberately NOT sent.
pub(crate) async fn translate_multipart(
    client: &Client,
    provider: &'static str,
    url: &str,
    audio: TranscriptionInput,
    api_key: String,
) -> Result<TranscriptionResponse, ProviderError> {
    audio_multipart(client, provider, url, audio, api_key, false).await
}

/// POST a `multipart/form-data` audio request (transcription OR translation) to an
/// OpenAI-dialect endpoint and map the response to the canonical `{text}` shape.
/// The transcriptions and translations endpoints are the near-twins OpenAI/Groq
/// expose: identical multipart contract EXCEPT translations has no `language`
/// field (output is always English). `include_language` parameterizes that one
/// difference; the `url` parameterizes the endpoint path (so the same code serves
/// `/v1/audio/transcriptions`, `/v1/audio/translations`, and Groq's
/// `/openai/v1/...` variants). Shared by the OpenAI and Groq adapters.
///
/// Privacy: the audio bytes and `api_key` are NEVER logged. The file part is the
/// raw bytes with the client's filename + a sniffed MIME; the text fields
/// (model/[language]/prompt/response_format/temperature) are added only when
/// present, so a bare `{file, model}` upload matches OpenAI's behaviour.
///
/// Response handling: for the primary `json`/`verbose_json` formats the upstream
/// returns JSON, which deserializes into `TranscriptionResponse` (the flattened
/// `extra` map tolerates verbose fields). For the non-JSON formats (`text`,
/// `srt`, `vtt`) the upstream returns a bare body — we wrap it in `{text: <body>}`
/// so the route always returns a consistent canonical shape.
async fn audio_multipart(
    client: &Client,
    provider: &'static str,
    url: &str,
    audio: TranscriptionInput,
    api_key: String,
    include_language: bool,
) -> Result<TranscriptionResponse, ProviderError> {
    let TranscriptionInput {
        file_bytes,
        filename,
        params,
    } = audio;

    let mime = audio_mime_from_filename(&filename);
    let file_part = reqwest::multipart::Part::bytes(file_bytes)
        .file_name(filename)
        .mime_str(mime)
        .map_err(|e| ProviderError::translation(format!("invalid audio mime: {e}")))?;

    let mut form = reqwest::multipart::Form::new()
        .part("file", file_part)
        .text("model", params.model);
    // `language` is a transcriptions-only field. The translations contract has no
    // `language` (output is always English), so it is sent ONLY when the caller
    // asked for transcription (`include_language`) — never for translations.
    if include_language {
        if let Some(language) = params.language {
            form = form.text("language", language);
        }
    }
    if let Some(prompt) = params.prompt {
        form = form.text("prompt", prompt);
    }
    let response_format = params.response_format.clone();
    if let Some(fmt) = params.response_format {
        form = form.text("response_format", fmt);
    }
    if let Some(temperature) = params.temperature {
        form = form.text("temperature", temperature.to_string());
    }

    let response = client
        .post(url)
        .header("Authorization", format!("Bearer {api_key}"))
        .multipart(form)
        .send()
        .await
        .map_err(|e| crate::client::sanitize_transport_error(provider, e))?;

    if !response.status().is_success() {
        return Err(crate::client::error_from_response(provider, response).await);
    }

    // Non-JSON response formats (text/srt/vtt) return a bare body, not JSON.
    let is_json_format = response_format
        .as_deref()
        .map(|f| f == "json" || f == "verbose_json")
        .unwrap_or(true); // default response_format is `json`
    let body = response
        .text()
        .await
        .map_err(|e| crate::client::sanitize_transport_error(provider, e))?;

    if is_json_format {
        serde_json::from_str::<TranscriptionResponse>(&body).map_err(|e| {
            // Never echo the body (it is the transcript = user content).
            ProviderError::translation(format!("{provider} transcription parse error: {e}"))
        })
    } else {
        Ok(TranscriptionResponse {
            text: body,
            extra: Default::default(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RetryClass;
    use futures::stream;

    #[test]
    fn parses_openai_content_delta() {
        let payload = r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}"#;
        let chunk = parse_openai_sse_payload(payload).unwrap().unwrap();
        assert_eq!(chunk.object, "chat.completion.chunk");
        assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("Hi"));
    }

    #[test]
    fn done_sentinel_returns_none() {
        assert!(parse_openai_sse_payload("[DONE]").unwrap().is_none());
    }

    // --- wiremock end-to-end (Task #4 / engineering-design §24) ---------------

    #[tokio::test]
    async fn buffered_call_threads_new_request_fields_to_wire() {
        use routeplane_types::Message;
        use wiremock::matchers::{body_partial_json, header, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        let resp = serde_json::json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1700000000u64,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}
        });

        // Assert the threaded fields (max_tokens, stop, presence_penalty, user)
        // actually appear in the outbound body — i.e. translation is not lossy.
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(header("authorization", "Bearer sk-test"))
            .and(body_partial_json(serde_json::json!({
                "model": "gpt-4o",
                "max_tokens": 256,
                "stop": ["STOP"],
                "presence_penalty": 0.5,
                "user": "u_123"
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(resp))
            .mount(&server)
            .await;

        let provider = OpenAIProvider::with_base_url(server.uri());
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
            max_tokens: Some(256),
            stop: Some(vec!["STOP".into()]),
            n: None,
            presence_penalty: Some(0.5),
            frequency_penalty: None,
            user: Some("u_123".into()),
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            ..Default::default()
        };
        let out = provider
            .chat_completion(req, "sk-test".into())
            .await
            .expect("mock call succeeds");
        assert_eq!(out.choices[0].message.content.as_text(), "hello");
        assert_eq!(out.usage.total_tokens, 6);
    }

    // --- new OpenAI-compat passthrough fields (response_format/seed/logprobs) ---

    #[tokio::test]
    async fn response_format_and_seed_reach_wire_and_logprobs_parse_back() {
        use routeplane_types::Message;
        use wiremock::matchers::{body_partial_json, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        // Upstream echoes structured-output content + per-choice logprobs +
        // system_fingerprint + service_tier (all OpenAI returns).
        let resp = serde_json::json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1700000000u64,
            "model": "gpt-4o",
            "system_fingerprint": "fp_zzz",
            "service_tier": "default",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "{\"name\":\"Ada\"}"},
                "finish_reason": "stop",
                "logprobs": {"content": [{"token": "{", "logprob": -0.1, "bytes": [123], "top_logprobs": []}]}
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
        });

        // Assert the new fields actually reach the upstream body (not dropped).
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(body_partial_json(serde_json::json!({
                "seed": 42,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "person", "schema": {"type": "object"}}
                },
                "logprobs": true,
                "top_logprobs": 3,
                "service_tier": "default",
                "reasoning_effort": "high",
                "logit_bias": {"50256": -100.0}
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(resp))
            .mount(&server)
            .await;

        let provider = OpenAIProvider::with_base_url(server.uri());
        let mut logit_bias = std::collections::BTreeMap::new();
        logit_bias.insert("50256".to_string(), -100.0f32);
        let req = ChatCompletionRequest {
            model: "gpt-4o".into(),
            messages: vec![Message {
                role: "user".into(),
                content: "name a person".into(),
                name: None,
                cache_control: None,
                tool_calls: None,
                tool_call_id: None,
                refusal: None,
                reasoning_content: None,
            }],
            seed: Some(42),
            response_format: Some(serde_json::json!({
                "type": "json_schema",
                "json_schema": {"name": "person", "schema": {"type": "object"}}
            })),
            logprobs: Some(true),
            top_logprobs: Some(3),
            service_tier: Some("default".into()),
            reasoning_effort: Some("high".into()),
            logit_bias: Some(logit_bias),
            ..Default::default()
        };
        let out = provider
            .chat_completion(req, "sk-test".into())
            .await
            .expect("mock call succeeds");
        // Response passthrough fields parse back via from_value for free.
        assert_eq!(out.system_fingerprint.as_deref(), Some("fp_zzz"));
        assert_eq!(out.service_tier.as_deref(), Some("default"));
        assert!(out.choices[0].logprobs.is_some());
    }

    // --- request passthrough: max_completion_tokens -----------------------------

    #[tokio::test]
    async fn max_completion_tokens_reaches_wire_verbatim() {
        // OpenAI-wire contract: the typed `max_completion_tokens` must reach the
        // upstream body verbatim (the whole typed request is serialized — this
        // test pins it). Unknown caller-supplied fields are dropped at
        // deserialization (forward-compat passthrough deferred to a dedicated ADR).
        use routeplane_types::Message;
        use wiremock::matchers::{body_partial_json, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        let resp = serde_json::json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1700000000u64,
            "model": "o4-mini",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        });
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(body_partial_json(serde_json::json!({
                "max_completion_tokens": 4096
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(resp))
            .mount(&server)
            .await;

        let provider = OpenAIProvider::with_base_url(server.uri());
        let req = ChatCompletionRequest {
            model: "o4-mini".into(),
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
            max_completion_tokens: Some(4096),
            ..Default::default()
        };
        provider
            .chat_completion(req, "sk-test".into())
            .await
            .expect("mock call succeeds");
    }

    // --- tool / function calling (verbatim OpenAI passthrough) -----------------

    #[tokio::test]
    async fn tools_are_forwarded_verbatim_and_tool_calls_parse_out() {
        use routeplane_types::{FunctionDef, Message, Tool};
        use wiremock::matchers::{body_partial_json, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        // The model responds with a tool_call (content null, finish tool_calls).
        let resp = serde_json::json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1700000000u64,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{\"location\":\"SF\"}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21}
        });

        // Assert the outbound body carries tools verbatim (OpenAI shape) +
        // tool_choice + parallel_tool_calls.
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(body_partial_json(serde_json::json!({
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the weather",
                        "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}
                    }
                }],
                "tool_choice": "auto",
                "parallel_tool_calls": true
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(resp))
            .mount(&server)
            .await;

        let provider = OpenAIProvider::with_base_url(server.uri());
        let req = ChatCompletionRequest {
            model: "gpt-4o".into(),
            messages: vec![Message {
                role: "user".into(),
                content: "weather in SF?".into(),
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
            tools: Some(vec![Tool {
                tool_type: "function".into(),
                function: FunctionDef {
                    name: "get_weather".into(),
                    description: Some("Get the weather".into()),
                    parameters: Some(serde_json::json!({
                        "type": "object",
                        "properties": {"location": {"type": "string"}}
                    })),
                },
            }]),
            tool_choice: Some(serde_json::json!("auto")),
            parallel_tool_calls: Some(true),
            ..Default::default()
        };
        let out = provider
            .chat_completion(req, "sk-test".into())
            .await
            .expect("mock call succeeds");
        // The response tool_calls map back to canonical (verbatim passthrough).
        assert_eq!(out.choices[0].finish_reason, "tool_calls");
        let calls = out.choices[0]
            .message
            .tool_calls
            .as_ref()
            .expect("tool_calls present");
        assert_eq!(calls[0].id, "call_abc");
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments, "{\"location\":\"SF\"}");
    }

    #[tokio::test]
    async fn streaming_tool_call_deltas_are_captured() {
        // OpenAI streams tool calls piecewise by index. The shared SSE helper must
        // carry the tool_call deltas through to the canonical Delta.
        let raw = concat!(
            "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"tool_calls\":[{\"index\":0,\"id\":\"call_abc\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"location\\\":\\\"SF\\\"}\"}}]},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        let byte_stream = stream::iter(vec![Ok(bytes::Bytes::from(raw.to_string()))]);
        let chunks: Vec<_> = openai_sse_to_chunks(byte_stream)
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();
        assert_eq!(chunks.len(), 3);
        let first = chunks[0].choices[0].delta.tool_calls.as_ref().unwrap();
        assert_eq!(first[0].index, 0);
        assert_eq!(first[0].id.as_deref(), Some("call_abc"));
        assert_eq!(
            first[0].function.as_ref().unwrap().name.as_deref(),
            Some("get_weather")
        );
        let second = chunks[1].choices[0].delta.tool_calls.as_ref().unwrap();
        assert_eq!(
            second[0].function.as_ref().unwrap().arguments.as_deref(),
            Some("{\"location\":\"SF\"}")
        );
        assert_eq!(
            chunks[2].choices[0].finish_reason.as_deref(),
            Some("tool_calls")
        );
    }

    // --- streaming passthrough: reasoning_content / refusal / logprobs ---------

    #[tokio::test]
    async fn streaming_passes_reasoning_refusal_and_logprobs_through() {
        // Reasoning-model / structured-outputs streaming fields must ride the
        // shared OpenAI SSE translation to the canonical chunks untouched — not
        // be dropped by the typed parse.
        let raw = concat!(
            "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"deepseek-reasoner\",\"system_fingerprint\":\"fp_x\",\"service_tier\":\"default\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"reasoning_content\":\"thinking \"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"deepseek-reasoner\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"42\"},\"finish_reason\":null,\"logprobs\":{\"content\":[]}}]}\n\n",
            "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"deepseek-reasoner\",\"choices\":[{\"index\":0,\"delta\":{\"refusal\":\"I cannot\"},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        let byte_stream = stream::iter(vec![Ok(bytes::Bytes::from(raw.to_string()))]);
        let chunks: Vec<_> = openai_sse_to_chunks(byte_stream)
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();
        assert_eq!(chunks.len(), 3);
        assert_eq!(
            chunks[0].choices[0].delta.reasoning_content.as_deref(),
            Some("thinking ")
        );
        assert_eq!(chunks[0].system_fingerprint.as_deref(), Some("fp_x"));
        assert_eq!(chunks[0].service_tier.as_deref(), Some("default"));
        assert_eq!(
            chunks[1].choices[0].logprobs,
            Some(serde_json::json!({"content": []}))
        );
        assert_eq!(
            chunks[2].choices[0].delta.refusal.as_deref(),
            Some("I cannot")
        );
        // Re-serialization keeps the fields (client sees them verbatim) and a
        // chunk WITHOUT them stays clean (byte-identical parity).
        let v = serde_json::to_value(&chunks[0]).unwrap();
        assert_eq!(v["choices"][0]["delta"]["reasoning_content"], "thinking ");
        let v1 = serde_json::to_value(&chunks[1]).unwrap();
        assert!(v1.get("system_fingerprint").is_none());
        assert!(v1["choices"][0]["delta"].get("reasoning_content").is_none());
    }

    // --- prompt-caching: strip cache_control on egress + surface cached_tokens --

    #[test]
    fn strip_cache_control_removes_marker_from_messages_and_parts() {
        // The Anthropic-only marker is removed from both the message and its
        // content parts; a body with no marker is left untouched.
        let mut body = serde_json::json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "x", "cache_control": {"type":"ephemeral"}},
                {"role": "user", "content": [
                    {"type":"text","text":"y","cache_control":{"type":"ephemeral"}}
                ]}
            ]
        });
        strip_cache_control_for_openai(&mut body);
        assert!(body["messages"][0].get("cache_control").is_none());
        assert!(body["messages"][1]["content"][0]
            .get("cache_control")
            .is_none());

        // No-op on a clean body (byte-identical guarantee).
        let mut clean = serde_json::json!({"messages":[{"role":"user","content":"hi"}]});
        let before = clean.clone();
        strip_cache_control_for_openai(&mut clean);
        assert_eq!(clean, before);
    }

    #[tokio::test]
    async fn reasoning_content_and_refusal_are_not_echoed_upstream_in_request_messages() {
        // The standard client pattern appends `choices[0].message` (which may
        // carry the response-only `reasoning_content`/`refusal`) to the next
        // request's history. Those fields must be stripped from outbound request
        // messages — DeepSeek 400s when `reasoning_content` appears in input.
        use routeplane_types::Message;
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        let resp = serde_json::json!({
            "id": "chatcmpl-2",
            "object": "chat.completion",
            "created": 1700000000u64,
            "model": "deepseek-reasoner",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        });
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(resp))
            .mount(&server)
            .await;

        let provider = OpenAIProvider::with_base_url(server.uri());
        // Follow-up turn: history includes a prior assistant message that came
        // back from a reasoning model with reasoning_content + refusal set.
        let req = ChatCompletionRequest {
            model: "deepseek-reasoner".into(),
            messages: vec![
                Message {
                    role: "user".into(),
                    content: "solve it".into(),
                    name: None,
                    cache_control: None,
                    tool_calls: None,
                    tool_call_id: None,
                    refusal: None,
                    reasoning_content: None,
                },
                Message {
                    role: "assistant".into(),
                    content: "42".into(),
                    name: None,
                    cache_control: None,
                    tool_calls: None,
                    tool_call_id: None,
                    refusal: Some("nope".into()),
                    reasoning_content: Some("chain of thought ...".into()),
                },
                Message {
                    role: "user".into(),
                    content: "why?".into(),
                    name: None,
                    cache_control: None,
                    tool_calls: None,
                    tool_call_id: None,
                    refusal: None,
                    reasoning_content: None,
                },
            ],
            ..Default::default()
        };
        provider
            .chat_completion(req, "sk-test".into())
            .await
            .expect("mock call succeeds");

        // Replay the recorded body: no request message may carry the
        // response-only fields.
        let received = &server.received_requests().await.unwrap()[0];
        let sent: serde_json::Value = serde_json::from_slice(&received.body).unwrap();
        for m in sent["messages"].as_array().unwrap() {
            assert!(
                m.get("reasoning_content").is_none(),
                "reasoning_content must be stripped from outbound request messages"
            );
            assert!(
                m.get("refusal").is_none(),
                "refusal must be stripped from outbound request messages"
            );
        }
        assert_eq!(sent["messages"][1]["content"], "42");
    }

    #[tokio::test]
    async fn cache_control_is_not_forwarded_to_openai_and_cached_tokens_map_through() {
        use routeplane_types::Message;
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        // OpenAI reports cached prompt tokens nested under prompt_tokens_details.
        let resp = serde_json::json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1700000000u64,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105,
                "prompt_tokens_details": {"cached_tokens": 80}
            }
        });
        // The mock asserts the OUTBOUND body has NO cache_control by capturing it
        // and inspecting it after the call (wiremock body matchers can't assert
        // ABSENCE cleanly, so we verify via a custom matcher on the recorded req).
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(resp))
            .mount(&server)
            .await;

        let provider = OpenAIProvider::with_base_url(server.uri());
        let req = ChatCompletionRequest {
            model: "gpt-4o".into(),
            messages: vec![Message {
                role: "system".into(),
                content: "cached preamble".into(),
                name: None,
                // A caller mistakenly (or portably) set the Anthropic marker — it
                // MUST be stripped before reaching OpenAI.
                cache_control: Some(serde_json::json!({"type": "ephemeral"})),
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
        let out = provider
            .chat_completion(req, "sk-test".into())
            .await
            .expect("mock call succeeds");
        // OpenAI's nested cached_tokens lifted into the canonical flat field.
        assert_eq!(out.usage.cached_tokens, Some(80));
        // OpenAI reports no cache CREATION (automatic cache) → stays None.
        assert_eq!(out.usage.cache_creation_tokens, None);

        // Verify the outbound body never carried cache_control by replaying the
        // recorded request body.
        let received = &server.received_requests().await.unwrap()[0];
        let sent: serde_json::Value = serde_json::from_slice(&received.body).unwrap();
        assert!(
            sent["messages"][0].get("cache_control").is_none(),
            "cache_control must NOT be forwarded to OpenAI"
        );
    }

    #[tokio::test]
    async fn upstream_429_is_typed_rate_limited_without_leaking_key() {
        use routeplane_types::Message;
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
            .mount(&server)
            .await;

        let provider = OpenAIProvider::with_base_url(server.uri());
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
        let err = provider
            .chat_completion(req, "sk-test".into())
            .await
            .expect_err("429 should be an Err");
        // Typed: drives retry classification, not just a string.
        assert_eq!(err.retry_class(), RetryClass::Status(429));
        let msg = err.to_string();
        assert!(msg.contains("429"));
        assert!(!msg.contains("sk-test"));
    }

    #[tokio::test]
    async fn translates_full_openai_sse_stream() {
        // Two content chunks, a final finish chunk with usage, then [DONE].
        let raw = concat!(
            "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":1,\"total_tokens\":6}}\n\n",
            "data: [DONE]\n\n",
        );
        // Split mid-line to exercise the line buffer across reads.
        let (a, b) = raw.split_at(40);
        let byte_stream = stream::iter(vec![
            Ok(bytes::Bytes::from(a.to_string())),
            Ok(bytes::Bytes::from(b.to_string())),
        ]);
        let chunks: Vec<_> = openai_sse_to_chunks(byte_stream)
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();
        assert_eq!(chunks.len(), 3);
        assert_eq!(
            chunks[0].choices[0].delta.role.as_deref(),
            Some("assistant")
        );
        assert_eq!(chunks[1].choices[0].delta.content.as_deref(), Some("Hello"));
        assert_eq!(chunks[2].choices[0].finish_reason.as_deref(), Some("stop"));
        assert_eq!(chunks[2].usage.as_ref().unwrap().total_tokens, 6);
    }

    // --- moderations (/v1/moderations) wiremock --------------------------------

    #[tokio::test]
    async fn moderations_hits_endpoint_and_maps_response() {
        use routeplane_types::{ModerationInput, ModerationRequest};
        use wiremock::matchers::{body_partial_json, header, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        let resp = serde_json::json!({
            "id": "modr-123",
            "model": "omni-moderation-latest",
            "results": [{
                "flagged": true,
                "categories": { "hate": true, "violence": false },
                "category_scores": { "hate": 0.92, "violence": 0.01 }
            }]
        });

        // Assert the request hits /v1/moderations with the input + the defaulted
        // model (caller omitted it ⇒ omni-moderation-latest).
        Mock::given(method("POST"))
            .and(path("/v1/moderations"))
            .and(header("authorization", "Bearer sk-test"))
            .and(body_partial_json(serde_json::json!({
                "input": "I will hurt them",
                "model": "omni-moderation-latest"
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(resp))
            .mount(&server)
            .await;

        let provider = OpenAIProvider::with_base_url(server.uri());
        let req = ModerationRequest {
            input: ModerationInput::Single("I will hurt them".into()),
            model: None, // exercises the default-model fill
        };
        let out = provider
            .moderations(req, "sk-test".into())
            .await
            .expect("mock call succeeds");
        assert_eq!(out.id, "modr-123");
        assert_eq!(out.model, "omni-moderation-latest");
        assert_eq!(out.results.len(), 1);
        assert!(out.results[0].flagged);
        assert_eq!(out.results[0].categories.get("hate"), Some(&true));
        assert_eq!(out.results[0].category_scores.get("hate"), Some(&0.92));
    }

    #[tokio::test]
    async fn moderations_default_via_trait_is_not_supported() {
        use crate::Provider;
        use routeplane_types::{ModerationInput, ModerationRequest};

        // A provider that does NOT override `moderations` returns the typed 422
        // capability-gap error (never a panic).
        struct NoModerationProvider;
        #[async_trait]
        impl Provider for NoModerationProvider {
            fn name(&self) -> &'static str {
                "noop"
            }
            async fn chat_completion(
                &self,
                _r: ChatCompletionRequest,
                _k: String,
            ) -> Result<ChatCompletionResponse, ProviderError> {
                unreachable!()
            }
        }
        let err = NoModerationProvider
            .moderations(
                ModerationRequest {
                    input: ModerationInput::Single("x".into()),
                    model: None,
                },
                "k".into(),
            )
            .await
            .expect_err("default should be unsupported");
        match err {
            ProviderError::BadRequest { status, body, .. } => {
                assert_eq!(status, 422);
                assert!(body.starts_with("moderation_not_supported"));
            }
            other => panic!("expected BadRequest 422, got {other:?}"),
        }
    }

    // --- image generation (/v1/images/generations) wiremock -------------------

    #[tokio::test]
    async fn image_generation_hits_endpoint_and_maps_b64_response() {
        use wiremock::matchers::{body_partial_json, header, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        // gpt-image-1 returns b64_json and may include a top-level usage block.
        let resp = serde_json::json!({
            "created": 1_700_000_000i64,
            "data": [ { "b64_json": "aGVsbG8=" } ],
            "usage": { "total_tokens": 42 }
        });

        // Assert the request hits /v1/images/generations with the prompt + the
        // defaulted model (caller omitted it ⇒ gpt-image-1).
        Mock::given(method("POST"))
            .and(path("/v1/images/generations"))
            .and(header("authorization", "Bearer sk-test"))
            .and(body_partial_json(serde_json::json!({
                "prompt": "a red panda",
                "model": "gpt-image-1"
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(resp))
            .mount(&server)
            .await;

        let provider = OpenAIProvider::with_base_url(server.uri());
        let req = ImageGenerationRequest {
            model: None, // exercises the default-model fill
            prompt: "a red panda".into(),
            n: None,
            size: None,
            quality: None,
            response_format: None,
            extra: Default::default(),
        };
        let out = provider
            .image_generation(req, "sk-test".into())
            .await
            .expect("mock call succeeds");
        assert_eq!(out.created, 1_700_000_000);
        assert_eq!(out.data.len(), 1);
        assert_eq!(out.data[0].b64_json.as_deref(), Some("aGVsbG8="));
        assert!(out.data[0].url.is_none());
        assert!(out.usage.is_some());
    }

    #[tokio::test]
    async fn image_generation_maps_url_response_and_threads_fields() {
        use wiremock::matchers::{body_partial_json, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        // dall-e-3 can return a hosted url + revised_prompt.
        let resp = serde_json::json!({
            "created": 1_700_000_111i64,
            "data": [ {
                "url": "https://img.example/abc.png",
                "revised_prompt": "a vivid city skyline at dusk"
            } ]
        });

        // The caller-supplied model + size + response_format must reach the wire.
        Mock::given(method("POST"))
            .and(path("/v1/images/generations"))
            .and(body_partial_json(serde_json::json!({
                "model": "dall-e-3",
                "prompt": "a city skyline",
                "size": "1024x1024",
                "response_format": "url"
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(resp))
            .mount(&server)
            .await;

        let provider = OpenAIProvider::with_base_url(server.uri());
        let req = ImageGenerationRequest {
            model: Some("dall-e-3".into()),
            prompt: "a city skyline".into(),
            n: Some(1),
            size: Some("1024x1024".into()),
            quality: None,
            response_format: Some("url".into()),
            extra: Default::default(),
        };
        let out = provider
            .image_generation(req, "sk-test".into())
            .await
            .expect("mock call succeeds");
        assert_eq!(
            out.data[0].url.as_deref(),
            Some("https://img.example/abc.png")
        );
        assert!(out.data[0].b64_json.is_none());
        assert_eq!(
            out.data[0].revised_prompt.as_deref(),
            Some("a vivid city skyline at dusk")
        );
    }

    #[tokio::test]
    async fn image_generation_default_via_trait_is_not_supported() {
        use crate::Provider;

        // A provider that does NOT override `image_generation` returns the typed
        // 422 capability-gap error (never a panic).
        struct NoImageProvider;
        #[async_trait]
        impl Provider for NoImageProvider {
            fn name(&self) -> &'static str {
                "noop"
            }
            async fn chat_completion(
                &self,
                _r: ChatCompletionRequest,
                _k: String,
            ) -> Result<ChatCompletionResponse, ProviderError> {
                unreachable!()
            }
        }
        let err = NoImageProvider
            .image_generation(
                ImageGenerationRequest {
                    model: None,
                    prompt: "x".into(),
                    n: None,
                    size: None,
                    quality: None,
                    response_format: None,
                    extra: Default::default(),
                },
                "k".into(),
            )
            .await
            .expect_err("default should be unsupported");
        match err {
            ProviderError::BadRequest { status, body, .. } => {
                assert_eq!(status, 422);
                assert!(body.starts_with("image_generation_not_supported"));
            }
            other => panic!("expected BadRequest 422, got {other:?}"),
        }
    }

    // --- audio transcription (/v1/audio/transcriptions) wiremock --------------

    #[tokio::test]
    async fn transcribe_hits_endpoint_with_multipart_and_maps_text() {
        use routeplane_types::{TranscriptionInput, TranscriptionParams};
        use wiremock::matchers::{header, header_exists, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        // The primary `json` contract returns `{"text": "..."}`.
        let resp = serde_json::json!({ "text": "hello from whisper" });

        // Assert: hits /v1/audio/transcriptions, Bearer auth, and the body is a
        // multipart/form-data upload (Content-Type starts with multipart/...).
        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
            .and(header("authorization", "Bearer sk-test"))
            .and(header_exists("content-type"))
            .respond_with(ResponseTemplate::new(200).set_body_json(resp))
            .mount(&server)
            .await;

        let provider = OpenAIProvider::with_base_url(server.uri());
        let audio = TranscriptionInput {
            file_bytes: b"RIFF....fake-wav-bytes".to_vec(),
            filename: "speech.wav".into(),
            params: TranscriptionParams {
                model: "whisper-1".into(),
                language: Some("en".into()),
                ..Default::default()
            },
        };
        let out = provider
            .transcribe(audio, "sk-test".into())
            .await
            .expect("mock call succeeds");
        assert_eq!(out.text, "hello from whisper");
    }

    #[tokio::test]
    async fn transcribe_non_json_format_returns_body_as_text() {
        use routeplane_types::{TranscriptionInput, TranscriptionParams};
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        // response_format=text ⇒ upstream returns a BARE body (not JSON); the
        // adapter must wrap it in `{text: <body>}`.
        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
            .respond_with(ResponseTemplate::new(200).set_body_string("plain transcript line"))
            .mount(&server)
            .await;

        let provider = OpenAIProvider::with_base_url(server.uri());
        let audio = TranscriptionInput {
            file_bytes: b"bytes".to_vec(),
            filename: "a.mp3".into(),
            params: TranscriptionParams {
                model: "whisper-1".into(),
                response_format: Some("text".into()),
                ..Default::default()
            },
        };
        let out = provider
            .transcribe(audio, "sk-test".into())
            .await
            .expect("mock call succeeds");
        assert_eq!(out.text, "plain transcript line");
    }

    #[tokio::test]
    async fn transcribe_default_via_trait_is_not_supported() {
        use crate::Provider;
        use routeplane_types::{TranscriptionInput, TranscriptionParams};

        // A provider that does NOT override `transcribe` returns the typed 422
        // capability-gap error (never a panic).
        struct NoAudioProvider;
        #[async_trait]
        impl Provider for NoAudioProvider {
            fn name(&self) -> &'static str {
                "noop"
            }
            async fn chat_completion(
                &self,
                _r: ChatCompletionRequest,
                _k: String,
            ) -> Result<ChatCompletionResponse, ProviderError> {
                unreachable!()
            }
        }
        let err = NoAudioProvider
            .transcribe(
                TranscriptionInput {
                    file_bytes: vec![1, 2, 3],
                    filename: "x.wav".into(),
                    params: TranscriptionParams {
                        model: "whisper-1".into(),
                        ..Default::default()
                    },
                },
                "k".into(),
            )
            .await
            .expect_err("default should be unsupported");
        match err {
            ProviderError::BadRequest { status, body, .. } => {
                assert_eq!(status, 422);
                assert!(body.starts_with("transcription_not_supported"));
            }
            other => panic!("expected BadRequest 422, got {other:?}"),
        }
    }

    // --- audio translation (/v1/audio/translations) wiremock ------------------

    #[tokio::test]
    async fn translate_hits_translations_endpoint_and_maps_text() {
        use routeplane_types::{TranscriptionInput, TranscriptionParams};
        use wiremock::matchers::{header, header_exists, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        // The `json` contract returns `{"text": "<english translation>"}`.
        let resp = serde_json::json!({ "text": "hello in english" });

        // Assert: hits /v1/audio/TRANSLATIONS (not transcriptions), Bearer auth,
        // and the body is a multipart/form-data upload.
        Mock::given(method("POST"))
            .and(path("/v1/audio/translations"))
            .and(header("authorization", "Bearer sk-test"))
            .and(header_exists("content-type"))
            .respond_with(ResponseTemplate::new(200).set_body_json(resp))
            .mount(&server)
            .await;

        let provider = OpenAIProvider::with_base_url(server.uri());
        let audio = TranscriptionInput {
            file_bytes: b"RIFF....fake-wav-bytes".to_vec(),
            filename: "speech.wav".into(),
            params: TranscriptionParams {
                model: "whisper-1".into(),
                // A `language` is present in the input but MUST NOT be sent for
                // translations (output is always English). Asserted below.
                language: Some("fr".into()),
                ..Default::default()
            },
        };
        let out = provider
            .translate(audio, "sk-test".into())
            .await
            .expect("mock call succeeds");
        assert_eq!(out.text, "hello in english");

        // Inspect the recorded multipart body: it must carry the model + file, but
        // NO `language` field (the translations contract has none). The body is a
        // multipart/form-data blob; a substring check is sufficient and robust.
        let received = &server.received_requests().await.unwrap()[0];
        let body = String::from_utf8_lossy(&received.body);
        assert!(
            body.contains("name=\"model\""),
            "translations body must carry the model field"
        );
        assert!(
            body.contains("whisper-1"),
            "translations body must carry the model value"
        );
        assert!(
            !body.contains("name=\"language\""),
            "translations must NOT send a `language` field (output is always English)"
        );
    }

    #[tokio::test]
    async fn translate_non_json_format_returns_body_as_text() {
        use routeplane_types::{TranscriptionInput, TranscriptionParams};
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        // response_format=text ⇒ upstream returns a BARE body; the adapter must
        // wrap it in `{text: <body>}` (same as transcriptions).
        Mock::given(method("POST"))
            .and(path("/v1/audio/translations"))
            .respond_with(ResponseTemplate::new(200).set_body_string("plain english line"))
            .mount(&server)
            .await;

        let provider = OpenAIProvider::with_base_url(server.uri());
        let audio = TranscriptionInput {
            file_bytes: b"bytes".to_vec(),
            filename: "a.mp3".into(),
            params: TranscriptionParams {
                model: "whisper-1".into(),
                response_format: Some("text".into()),
                ..Default::default()
            },
        };
        let out = provider
            .translate(audio, "sk-test".into())
            .await
            .expect("mock call succeeds");
        assert_eq!(out.text, "plain english line");
    }

    #[tokio::test]
    async fn translate_default_via_trait_is_not_supported() {
        use crate::Provider;
        use routeplane_types::{TranscriptionInput, TranscriptionParams};

        // A provider that does NOT override `translate` returns the typed 422
        // capability-gap error (never a panic).
        struct NoAudioProvider;
        #[async_trait]
        impl Provider for NoAudioProvider {
            fn name(&self) -> &'static str {
                "noop"
            }
            async fn chat_completion(
                &self,
                _r: ChatCompletionRequest,
                _k: String,
            ) -> Result<ChatCompletionResponse, ProviderError> {
                unreachable!()
            }
        }
        let err = NoAudioProvider
            .translate(
                TranscriptionInput {
                    file_bytes: vec![1, 2, 3],
                    filename: "x.wav".into(),
                    params: TranscriptionParams {
                        model: "whisper-1".into(),
                        ..Default::default()
                    },
                },
                "k".into(),
            )
            .await
            .expect_err("default should be unsupported");
        match err {
            ProviderError::BadRequest { status, body, .. } => {
                assert_eq!(status, 422);
                assert!(body.starts_with("translation_not_supported"));
            }
            other => panic!("expected BadRequest 422, got {other:?}"),
        }
    }

    // --- text-to-speech (/v1/audio/speech) wiremock ---------------------------

    #[tokio::test]
    async fn speech_hits_endpoint_and_returns_binary_with_content_type() {
        use wiremock::matchers::{body_partial_json, header, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        // The upstream returns RAW audio bytes (not JSON) with an audio/* type.
        let audio_bytes: Vec<u8> = vec![0xFF, 0xFB, 0x90, 0x00, 0x01, 0x02, 0x03];

        // Assert: hits /v1/audio/speech, Bearer auth, JSON body carrying the
        // input + voice + the defaulted model (caller omitted it ⇒ gpt-4o-mini-tts).
        Mock::given(method("POST"))
            .and(path("/v1/audio/speech"))
            .and(header("authorization", "Bearer sk-test"))
            .and(body_partial_json(serde_json::json!({
                "input": "Hello world",
                "voice": "alloy",
                "model": "gpt-4o-mini-tts"
            })))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "audio/mpeg")
                    .set_body_bytes(audio_bytes.clone()),
            )
            .mount(&server)
            .await;

        let provider = OpenAIProvider::with_base_url(server.uri());
        let req = SpeechRequest {
            model: None, // exercises the default-model fill
            input: "Hello world".into(),
            voice: "alloy".into(),
            response_format: None,
            speed: None,
            extra: Default::default(),
        };
        let out = provider
            .speech(req, "sk-test".into())
            .await
            .expect("mock call succeeds");
        assert_eq!(out.bytes, audio_bytes);
        assert_eq!(out.content_type, "audio/mpeg");
    }

    #[tokio::test]
    async fn speech_derives_content_type_from_format_when_upstream_omits_header() {
        use wiremock::matchers::{body_partial_json, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        // No content-type header on the response → adapter must derive it from the
        // requested response_format (wav ⇒ audio/wav). The caller-supplied model +
        // response_format + speed must reach the wire.
        Mock::given(method("POST"))
            .and(path("/v1/audio/speech"))
            .and(body_partial_json(serde_json::json!({
                "model": "tts-1",
                "input": "hi",
                "voice": "nova",
                "response_format": "wav",
                "speed": 1.25
            })))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(vec![1u8, 2, 3]))
            .mount(&server)
            .await;

        let provider = OpenAIProvider::with_base_url(server.uri());
        let req = SpeechRequest {
            model: Some("tts-1".into()),
            input: "hi".into(),
            voice: "nova".into(),
            response_format: Some("wav".into()),
            speed: Some(1.25),
            extra: Default::default(),
        };
        let out = provider
            .speech(req, "sk-test".into())
            .await
            .expect("mock call succeeds");
        assert_eq!(out.bytes, vec![1u8, 2, 3]);
        assert_eq!(out.content_type, "audio/wav");
    }

    #[tokio::test]
    async fn speech_default_via_trait_is_not_supported() {
        use crate::Provider;

        // A provider that does NOT override `speech` returns the typed 422
        // capability-gap error (never a panic).
        struct NoSpeechProvider;
        #[async_trait]
        impl Provider for NoSpeechProvider {
            fn name(&self) -> &'static str {
                "noop"
            }
            async fn chat_completion(
                &self,
                _r: ChatCompletionRequest,
                _k: String,
            ) -> Result<ChatCompletionResponse, ProviderError> {
                unreachable!()
            }
        }
        let err = NoSpeechProvider
            .speech(
                SpeechRequest {
                    model: None,
                    input: "x".into(),
                    voice: "alloy".into(),
                    response_format: None,
                    speed: None,
                    extra: Default::default(),
                },
                "k".into(),
            )
            .await
            .expect_err("default should be unsupported");
        match err {
            ProviderError::BadRequest { status, body, .. } => {
                assert_eq!(status, 422);
                assert!(body.starts_with("speech_not_supported"));
            }
            other => panic!("expected BadRequest 422, got {other:?}"),
        }
    }

    #[test]
    fn content_type_mapping_covers_all_formats() {
        assert_eq!(content_type_for_format(None), "audio/mpeg");
        assert_eq!(content_type_for_format(Some("mp3")), "audio/mpeg");
        assert_eq!(content_type_for_format(Some("opus")), "audio/ogg");
        assert_eq!(content_type_for_format(Some("aac")), "audio/aac");
        assert_eq!(content_type_for_format(Some("flac")), "audio/flac");
        assert_eq!(content_type_for_format(Some("wav")), "audio/wav");
        assert_eq!(content_type_for_format(Some("pcm")), "audio/pcm");
    }

    // --- ADR-041 C6.7: inbound Authorization is NEVER forwarded upstream -------
    //
    // Load-bearing negative test for the C3 invariant. The scenario: a caller
    // authenticates to the gateway with `Authorization: Bearer rp_abc` (the new
    // SDK-compat inbound fallback). The gateway resolves the tenant's provider
    // key (`sk-test`) server-side and hands the adapter ONLY `(request, api_key)`
    // — the `Provider` trait never sees the inbound `HeaderMap` (lib.rs `Provider`).
    // This test proves the upstream request carries `Bearer sk-test` and NEVER
    // the caller's `rp_abc`: it mounts a mock that matches ONLY on
    // `authorization == "Bearer sk-test"`, so a leaked `rp_abc` would 404 the
    // call and fail the assertion. We additionally assert the body was never
    // tagged with the inbound key.
    #[tokio::test]
    async fn inbound_bearer_rp_key_is_never_forwarded_upstream() {
        use routeplane_types::Message;
        use wiremock::matchers::{header, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        let resp = serde_json::json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1700000000u64,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        });

        // The mock ONLY answers when the upstream Authorization is the RESOLVED
        // provider key. If the caller's inbound `rp_abc` leaked through, this
        // matcher would not match → 404 → `chat_completion` returns Err → the
        // `.expect(...)` below fails. This is the structural proof of C3.
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(header("authorization", "Bearer sk-test"))
            .respond_with(ResponseTemplate::new(200).set_body_json(resp))
            .mount(&server)
            .await;

        let provider = OpenAIProvider::with_base_url(server.uri());
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

        // The adapter is handed the RESOLVED provider key `sk-test`, exactly as
        // the proxy does after `resolve_api_key` — the inbound `rp_abc` the
        // caller sent is not in scope here by construction (the trait signature
        // has no inbound-header parameter). The call succeeds ONLY because the
        // upstream Authorization is `Bearer sk-test`.
        let out = provider
            .chat_completion(req, "sk-test".into())
            .await
            .expect("mock matched on the resolved provider key, not the caller's rp_ key");
        assert_eq!(out.choices[0].message.content.as_text(), "ok");
    }
}
