use async_trait::async_trait;
use futures::stream::BoxStream;
use routeplane_types::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, EmbeddingRequest,
    EmbeddingResponse, ImageGenerationRequest, ImageGenerationResponse, ModerationRequest,
    ModerationResponse, RerankRequest, RerankResponse, SpeechRequest, TranscriptionInput,
    TranscriptionResponse,
};
use std::time::Duration;

/// The binary result of a text-to-speech (`/v1/audio/speech`) call: the raw
/// audio bytes plus the `Content-Type` to return to the client (derived from the
/// upstream response header or the requested `response_format`). There is no
/// JSON response type — the route returns these bytes verbatim. The bytes are
/// user content (synthesized speech) and MUST NEVER be logged.
pub struct SpeechAudio {
    /// The raw audio bytes (mp3/opus/aac/flac/wav/pcm). Unbounded — TTS output is
    /// not capped by the inbound `RequestBodyLimit` (that bounds the request).
    pub bytes: Vec<u8>,
    /// The HTTP `Content-Type` for the response (e.g. `audio/mpeg` for mp3).
    pub content_type: String,
}

impl std::fmt::Debug for SpeechAudio {
    /// Custom Debug that NEVER prints the audio bytes (only their length) — the
    /// bytes are user content and must not leak into logs.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpeechAudio")
            .field("bytes_len", &self.bytes.len())
            .field("content_type", &self.content_type)
            .finish()
    }
}

pub mod anthropic;
pub mod azure_openai;
pub mod bedrock;
pub mod client;
pub mod cohere;
pub mod deepseek;
pub mod fireworks;
pub mod gemini;
pub mod groq;
pub mod mistral;
pub mod openai;
pub mod openai_compat;
pub mod openai_compatible;
pub mod openrouter;
pub mod sse;
pub mod together;
pub mod vision;
pub mod xai;

/// The typed error every provider call (buffered and streamed) returns (G2.3 /
/// ADR-021 §4). Replaces the previous `Box<dyn Error + Send + Sync>` so retry
/// policy can classify on the upstream HTTP status WITHOUT string-sniffing.
///
/// Backward compatibility: `From<String>`, `From<&str>` and `From<serde_json::Error>`
/// are provided so existing `"...".into()` / `?` call sites in the adapters keep
/// compiling — those map to `Translation` (a parsing/translation failure, never
/// retryable). HTTP-status branches now go through [`client::error_from_response`],
/// which classifies into the variant the retry loop reads.
#[derive(Debug)]
pub enum ProviderError {
    /// Per-attempt timeout or upstream 408. Always retryable (deadline permitting).
    Timeout { provider: String, detail: String },
    /// Upstream 429. Retryable when `429 ∈ retry.on_status`.
    RateLimited {
        provider: String,
        retry_after: Option<Duration>,
        body: String,
    },
    /// Upstream 401/403. NEVER retryable (a key-burning loop).
    Auth {
        provider: String,
        status: u16,
        body: String,
    },
    /// Other non-retryable 4xx (400/404/422/…).
    BadRequest {
        provider: String,
        status: u16,
        body: String,
    },
    /// Upstream 5xx. Retryable when `status ∈ retry.on_status`.
    Upstream5xx {
        provider: String,
        status: u16,
        body: String,
    },
    /// Transport failure (connect/request). Always retryable.
    Network { provider: String, detail: String },
    /// Request/response translation or body-decode failure. NEVER retryable
    /// (retrying won't fix malformed bytes).
    Translation { detail: String },
}

/// How the proxy retry loop should treat an error (it owns the `on_status` set).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryClass {
    /// Retry while attempts + deadline remain, regardless of `on_status`.
    Always,
    /// Retry only when this status is in the target's `on_status` set.
    Status(u16),
    /// Never retry.
    Never,
}

impl ProviderError {
    pub fn timeout(provider: impl Into<String>, detail: impl Into<String>) -> Self {
        ProviderError::Timeout {
            provider: provider.into(),
            detail: detail.into(),
        }
    }
    pub fn network(provider: impl Into<String>, detail: impl Into<String>) -> Self {
        ProviderError::Network {
            provider: provider.into(),
            detail: detail.into(),
        }
    }
    pub fn translation(detail: impl Into<String>) -> Self {
        ProviderError::Translation {
            detail: detail.into(),
        }
    }

    /// The provider has no first-party embeddings endpoint (Anthropic today).
    /// Maps to a 422 `embeddings_not_supported` envelope at the `/v1/embeddings`
    /// route edge (PRD-011 FR-2/FR-3). Modelled as a non-retryable `BadRequest`
    /// (422) — a capability gap, not a transient fault. The body is prefixed
    /// `embeddings_not_supported` so the route can recognise it without parsing.
    pub fn embeddings_not_supported(provider: impl Into<String>) -> Self {
        let provider = provider.into();
        let body = format!(
            "embeddings_not_supported: provider '{provider}' has no first-party embeddings endpoint"
        );
        ProviderError::BadRequest {
            provider,
            status: 422,
            body,
        }
    }

    /// The provider has no first-party rerank endpoint. Maps to a 422
    /// `rerank_not_supported` envelope at the `/v1/rerank` route edge (mirrors
    /// [`embeddings_not_supported`]). Modelled as a non-retryable `BadRequest`
    /// (422) — a capability gap, not a transient fault. The body is prefixed
    /// `rerank_not_supported` so the route can recognise it without parsing.
    pub fn rerank_not_supported(provider: impl Into<String>) -> Self {
        let provider = provider.into();
        let body = format!(
            "rerank_not_supported: provider '{provider}' has no first-party rerank endpoint"
        );
        ProviderError::BadRequest {
            provider,
            status: 422,
            body,
        }
    }

    /// The provider has no first-party moderation endpoint. Maps to a 422
    /// `moderation_not_supported` envelope at the `/v1/moderations` route edge
    /// (mirrors [`embeddings_not_supported`] / [`rerank_not_supported`]).
    /// Modelled as a non-retryable `BadRequest` (422) — a capability gap, not a
    /// transient fault. The body is prefixed `moderation_not_supported` so the
    /// route can recognise it without parsing.
    pub fn moderation_not_supported(provider: impl Into<String>) -> Self {
        let provider = provider.into();
        let body = format!(
            "moderation_not_supported: provider '{provider}' has no first-party moderation endpoint"
        );
        ProviderError::BadRequest {
            provider,
            status: 422,
            body,
        }
    }

    /// The provider has no first-party image-generation endpoint. Maps to a 422
    /// `image_generation_not_supported` envelope at the `/v1/images/generations`
    /// route edge (mirrors `moderation_not_supported`). Modelled as a
    /// non-retryable `BadRequest` (422) — a capability gap, not a transient
    /// fault. The body is prefixed `image_generation_not_supported` so the route
    /// can recognise it without parsing.
    pub fn image_generation_not_supported(provider: impl Into<String>) -> Self {
        let provider = provider.into();
        let body = format!(
            "image_generation_not_supported: provider '{provider}' has no first-party image-generation endpoint"
        );
        ProviderError::BadRequest {
            provider,
            status: 422,
            body,
        }
    }

    /// The provider has no first-party audio-transcription endpoint. Maps to a
    /// 422 `transcription_not_supported` envelope at the
    /// `/v1/audio/transcriptions` route edge (mirrors the other
    /// `*_not_supported` helpers). Modelled as a non-retryable `BadRequest`
    /// (422) — a capability gap, not a transient fault. The body is prefixed
    /// `transcription_not_supported` so the route can recognise it without
    /// parsing.
    pub fn transcription_not_supported(provider: impl Into<String>) -> Self {
        let provider = provider.into();
        let body = format!(
            "transcription_not_supported: provider '{provider}' has no first-party audio-transcription endpoint"
        );
        ProviderError::BadRequest {
            provider,
            status: 422,
            body,
        }
    }

    /// The provider has no first-party audio-translation endpoint. Maps to a 422
    /// `translation_not_supported` envelope at the `/v1/audio/translations` route
    /// edge (mirrors `transcription_not_supported`). Modelled as a non-retryable
    /// `BadRequest` (422) — a capability gap, not a transient fault. The body is
    /// prefixed `translation_not_supported` so the route can recognise it without
    /// parsing.
    pub fn translation_not_supported(provider: impl Into<String>) -> Self {
        let provider = provider.into();
        let body = format!(
            "translation_not_supported: provider '{provider}' has no first-party audio-translation endpoint"
        );
        ProviderError::BadRequest {
            provider,
            status: 422,
            body,
        }
    }

    /// The provider has no first-party text-to-speech endpoint. Maps to a 422
    /// `speech_not_supported` envelope at the `/v1/audio/speech` route edge
    /// (mirrors the other `*_not_supported` helpers). Modelled as a non-retryable
    /// `BadRequest` (422) — a capability gap, not a transient fault. The body is
    /// prefixed `speech_not_supported` so the route can recognise it without
    /// parsing.
    pub fn speech_not_supported(provider: impl Into<String>) -> Self {
        let provider = provider.into();
        let body = format!(
            "speech_not_supported: provider '{provider}' has no first-party text-to-speech endpoint"
        );
        ProviderError::BadRequest {
            provider,
            status: 422,
            body,
        }
    }

    /// The requested `model` maps to no Azure deployment when an authoritative
    /// deployment map (`AZURE_OPENAI_DEPLOYMENTS`) is configured. Modelled as a
    /// non-retryable `BadRequest` (422) — refusing SILENT model substitution is
    /// the whole point for model-pinned compliance workloads, so we never fall
    /// back to a different deployment. The body is prefixed
    /// `azure_deployment_unmapped` so the route can recognise it without parsing.
    pub fn azure_deployment_unmapped(
        provider: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        let provider = provider.into();
        let model = model.into();
        let body = format!(
            "azure_deployment_unmapped: model '{model}' maps to no configured Azure \
             deployment (set it in AZURE_OPENAI_DEPLOYMENTS)"
        );
        ProviderError::BadRequest {
            provider,
            status: 422,
            body,
        }
    }

    /// The upstream HTTP status, when the error carries one.
    pub fn status(&self) -> Option<u16> {
        match self {
            ProviderError::RateLimited { .. } => Some(429),
            ProviderError::Auth { status, .. }
            | ProviderError::BadRequest { status, .. }
            | ProviderError::Upstream5xx { status, .. } => Some(*status),
            ProviderError::Timeout { .. }
            | ProviderError::Network { .. }
            | ProviderError::Translation { .. } => None,
        }
    }

    pub fn retry_class(&self) -> RetryClass {
        match self {
            ProviderError::Timeout { .. } | ProviderError::Network { .. } => RetryClass::Always,
            ProviderError::RateLimited { .. } => RetryClass::Status(429),
            ProviderError::Upstream5xx { status, .. } => RetryClass::Status(*status),
            ProviderError::Auth { .. }
            | ProviderError::BadRequest { .. }
            | ProviderError::Translation { .. } => RetryClass::Never,
        }
    }
}

impl std::fmt::Display for ProviderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProviderError::Timeout { provider, detail } => {
                write!(f, "{provider}: {detail}")
            }
            ProviderError::RateLimited { provider, body, .. } => {
                write!(f, "{provider} API error (429): {body}")
            }
            // Auth bodies can echo provider-key fingerprints (e.g. OpenAI's
            // "Incorrect API key provided: sk-…") — never shown to clients.
            ProviderError::Auth {
                provider, status, ..
            } => write!(f, "{provider} authentication failed ({status})"),
            ProviderError::BadRequest {
                provider,
                status,
                body,
            }
            | ProviderError::Upstream5xx {
                provider,
                status,
                body,
            } => write!(f, "{provider} API error ({status}): {body}"),
            ProviderError::Network { provider, detail } => write!(f, "{provider}: {detail}"),
            ProviderError::Translation { detail } => write!(f, "{detail}"),
        }
    }
}

impl std::error::Error for ProviderError {}

impl From<String> for ProviderError {
    fn from(detail: String) -> Self {
        ProviderError::Translation { detail }
    }
}

impl From<&str> for ProviderError {
    fn from(detail: &str) -> Self {
        ProviderError::Translation {
            detail: detail.to_string(),
        }
    }
}

impl From<serde_json::Error> for ProviderError {
    fn from(e: serde_json::Error) -> Self {
        ProviderError::Translation {
            detail: format!("translation error: {e}"),
        }
    }
}

/// A canonical, OpenAI-shaped chunk stream. Each provider's `chat_completion_stream`
/// returns this after the upstream connection is established; the proxy forwards
/// each chunk to the client as one SSE `data:` line.
pub type ChunkStream = BoxStream<'static, Result<ChatCompletionChunk, ProviderError>>;

#[async_trait]
pub trait Provider: Send + Sync {
    fn name(&self) -> &'static str;

    /// Data-residency jurisdictions this provider serves data from (e.g. "IN").
    /// Empty (the default) means no residency guarantee — such a provider is
    /// never eligible when sovereign routing to a specific region is enforced.
    fn resident_regions(&self) -> Vec<String> {
        Vec::new()
    }

    /// Whether this provider is resident in the given jurisdiction.
    fn is_resident_in(&self, region: &str) -> bool {
        self.resident_regions().iter().any(|r| r == region)
    }

    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
        api_key: String,
    ) -> Result<ChatCompletionResponse, ProviderError>;

    /// Streaming counterpart of `chat_completion`. Establishes the upstream
    /// connection (sending the native "stream" flag) and returns a canonical,
    /// OpenAI-shaped chunk stream.
    ///
    /// Establishment semantics matter for fallback: returning `Ok(stream)` means
    /// the upstream request succeeded and the proxy is now committed to this
    /// provider; returning `Err` means establishment failed and the proxy may
    /// fall back to the next candidate. A failure that occurs *after* the stream
    /// has begun is surfaced as an `Err` item *inside* the stream (it ends the
    /// SSE response, it does not trigger fallback).
    ///
    /// The default implementation adapts the buffered `chat_completion` into a
    /// one-shot stream, so providers that have no native streaming still satisfy
    /// the OpenAI streaming contract (just without incremental delivery).
    async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
        api_key: String,
    ) -> Result<ChunkStream, ProviderError> {
        let resp = self.chat_completion(request, api_key).await?;
        Ok(Box::pin(sse::buffered_response_as_stream(resp)))
    }

    /// Compute embeddings for one or more inputs (PRD-011 §5, FR-2). The default
    /// returns a typed `embeddings_not_supported` error — a 422 at the
    /// `/v1/embeddings` route edge, never a panic — so a provider without a
    /// first-party embeddings endpoint (Anthropic today) degrades cleanly.
    /// Providers that DO offer embeddings (OpenAI, Azure OpenAI, Gemini) override
    /// this and return real vectors mapped back to the OpenAI shape.
    async fn embeddings(
        &self,
        _request: EmbeddingRequest,
        _api_key: String,
    ) -> Result<EmbeddingResponse, ProviderError> {
        Err(ProviderError::embeddings_not_supported(self.name()))
    }

    /// Rerank a candidate document set against a query (parity with LiteLLM's
    /// `/rerank`; core to RAG pipelines). The default returns a typed
    /// `rerank_not_supported` error — a 422 at the `/v1/rerank` route edge,
    /// never a panic — so a provider without a first-party rerank endpoint
    /// degrades cleanly. Providers that DO offer reranking (Cohere today)
    /// override this and return real relevance scores mapped back to the
    /// canonical shape.
    async fn rerank(
        &self,
        _request: RerankRequest,
        _api_key: String,
    ) -> Result<RerankResponse, ProviderError> {
        Err(ProviderError::rerank_not_supported(self.name()))
    }

    /// Classify one or more inputs for policy-violating content (parity with
    /// OpenAI's `/v1/moderations`; LiteLLM proxies `/moderations`). The default
    /// returns a typed `moderation_not_supported` error — a 422 at the
    /// `/v1/moderations` route edge, never a panic — so a provider without a
    /// first-party moderation endpoint degrades cleanly. Providers that DO offer
    /// moderation (OpenAI today) override this and return real category scores
    /// mapped back to the OpenAI shape. The `local` source (in the binary) runs
    /// Routeplane's built-in moderator instead of calling out.
    async fn moderations(
        &self,
        _request: ModerationRequest,
        _api_key: String,
    ) -> Result<ModerationResponse, ProviderError> {
        Err(ProviderError::moderation_not_supported(self.name()))
    }

    /// Generate one or more images from a text prompt (parity with OpenAI's
    /// `/v1/images/generations`; LiteLLM/Portkey proxy image generation). The
    /// default returns a typed `image_generation_not_supported` error — a 422 at
    /// the `/v1/images/generations` route edge, never a panic — so a provider
    /// without a first-party image endpoint degrades cleanly. Providers that DO
    /// offer image generation (OpenAI today) override this and return the
    /// generated images mapped back to the OpenAI shape.
    async fn image_generation(
        &self,
        _request: ImageGenerationRequest,
        _api_key: String,
    ) -> Result<ImageGenerationResponse, ProviderError> {
        Err(ProviderError::image_generation_not_supported(self.name()))
    }

    /// Transcribe an audio file to text (parity with OpenAI's
    /// `/v1/audio/transcriptions`; Groq's Whisper is the flagship fast/cheap STT
    /// backend). The inbound contract is `multipart/form-data` (binary audio),
    /// so the handler buffers the file and threads it here via
    /// [`TranscriptionInput`]. The default returns a typed
    /// `transcription_not_supported` error — a 422 at the route edge, never a
    /// panic — so a provider without a first-party STT endpoint degrades
    /// cleanly. Providers that DO offer transcription (OpenAI, Groq) override
    /// this and POST a multipart body upstream, mapping the response back to the
    /// canonical `{text}` shape. Implementations MUST NEVER log the audio bytes
    /// or the api_key.
    async fn transcribe(
        &self,
        _audio: TranscriptionInput,
        _api_key: String,
    ) -> Result<TranscriptionResponse, ProviderError> {
        Err(ProviderError::transcription_not_supported(self.name()))
    }

    /// Translate an audio file (speech in any language) into ENGLISH text (parity
    /// with OpenAI's `/v1/audio/translations`; Groq's `whisper-large-v3` is the
    /// flagship fast/cheap backend). The near-twin of [`transcribe`]: the same
    /// `multipart/form-data` inbound contract and the same [`TranscriptionInput`]
    /// type, EXCEPT there is no `language` field (the output is always English) —
    /// adapters simply do not send `language` for translations. Translations is a
    /// Whisper feature (`whisper-1`, `whisper-large-v3`); the gpt-4o-transcribe
    /// models do not support it. The default returns a typed
    /// `translation_not_supported` error — a 422 at the route edge, never a panic —
    /// so a provider without a first-party translation endpoint degrades cleanly.
    /// Implementations MUST NEVER log the audio bytes or the api_key.
    async fn translate(
        &self,
        _audio: TranscriptionInput,
        _api_key: String,
    ) -> Result<TranscriptionResponse, ProviderError> {
        Err(ProviderError::translation_not_supported(self.name()))
    }

    /// Synthesize speech from text (parity with OpenAI's `/v1/audio/speech`;
    /// LiteLLM/Portkey proxy it). Completes the audio pair with [`transcribe`].
    /// The inbound contract is JSON ([`SpeechRequest`]); the RESPONSE is raw
    /// binary audio, so this returns [`SpeechAudio`] (bytes + `Content-Type`),
    /// NOT a JSON type. The default returns a typed `speech_not_supported` error
    /// — a 422 at the `/v1/audio/speech` route edge, never a panic — so a provider
    /// without a first-party TTS endpoint degrades cleanly. Providers that DO
    /// offer TTS (OpenAI today) override this and POST `{model, input, voice, …}`
    /// upstream, capturing the response bytes + Content-Type. Implementations MUST
    /// NEVER log the audio bytes or the api_key.
    async fn speech(
        &self,
        _request: SpeechRequest,
        _api_key: String,
    ) -> Result<SpeechAudio, ProviderError> {
        Err(ProviderError::speech_not_supported(self.name()))
    }
}
