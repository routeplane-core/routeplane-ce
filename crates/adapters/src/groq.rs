use crate::openai::{openai_sse_to_chunks, transcribe_multipart, translate_multipart};
use crate::{ChunkStream, Provider, ProviderError};
use async_trait::async_trait;
use reqwest::Client;
use routeplane_types::{
    ChatCompletionRequest, ChatCompletionResponse, EmbeddingRequest, EmbeddingResponse,
    TranscriptionInput, TranscriptionResponse,
};
use serde_json::json;

/// Groq adapter — OpenAI-compatible wire format. Groq runs open-weight models
/// (Llama, Gemma, Qwen, DeepSeek-R1-distill, Kimi-K2) on its LPU inference
/// hardware and is the leading **ultra-low-latency** inference provider, so it
/// is a natural fallback target for latency-strategy routing.
///
/// Wire dialect: chat completions, SSE streaming and tool/function-calling all
/// follow OpenAI conventions, so the canonical `routeplane_types` models
/// serialize/deserialize directly and the SSE translation is shared with
/// [`crate::openai`] (`openai_sse_to_chunks`). Only the base URL, the provider
/// name and the (absent) embeddings capability differ from OpenAI.
///
/// **Base-URL footgun:** Groq serves its OpenAI-compatible surface under
/// `https://api.groq.com/openai/v1` — note the `/openai/v1` prefix. The
/// chat-completions path is therefore `{base}/openai/v1/chat/completions`;
/// dropping the `/v1` (or the `/openai`) yields a 404. `GROQ_DEFAULT_BASE_URL`
/// holds the host root and the adapter appends the full `/openai/v1/...` path.
///
/// Embeddings: Groq has **no** first-party embeddings endpoint, so `embeddings`
/// degrades to a typed `embeddings_not_supported` 422 (same as Anthropic) —
/// never a panic.
pub struct GroqProvider {
    client: Client,
    base_url: String,
}

/// Host root for Groq's OpenAI-compatible API. The `/openai/v1/...` path is
/// appended by each call — **including the `/v1`** (the known footgun).
const GROQ_DEFAULT_BASE_URL: &str = "https://api.groq.com";

impl GroqProvider {
    pub fn new() -> Self {
        Self {
            client: crate::client::build_provider_client(),
            base_url: GROQ_DEFAULT_BASE_URL.to_string(),
        }
    }

    /// Test/override constructor pointing at a custom base URL (e.g. a wiremock
    /// server). The `/openai/v1/...` path is still appended, so tests assert the
    /// full Groq path is hit.
    pub fn with_base_url(base_url: impl Into<String>) -> Self {
        Self {
            client: crate::client::build_provider_client(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
        }
    }
}

impl Default for GroqProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Provider for GroqProvider {
    fn name(&self) -> &'static str {
        "groq"
    }

    /// Groq inference is US-based today. Residency is opt-in via `GROQ_REGION`
    /// (e.g. "US"); empty (the default) means no residency guarantee, so Groq is
    /// never eligible when sovereign routing to a specific region is enforced —
    /// the conservative default that keeps residency filtering correct.
    fn resident_regions(&self) -> Vec<String> {
        let region = std::env::var("GROQ_REGION").unwrap_or_default();
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
        let url = format!("{}/openai/v1/chat/completions", self.base_url);

        // Strip the Anthropic-only `cache_control` marker before egress.
        let mut body = serde_json::to_value(&request)?;
        crate::openai::strip_cache_control_for_openai(&mut body);

        let response = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {api_key}"))
            .json(&body)
            .send()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("groq", e))?;

        if !response.status().is_success() {
            return Err(crate::client::error_from_response("groq", response).await);
        }

        response
            .json::<ChatCompletionResponse>()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("groq", e))
    }

    async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
        api_key: String,
    ) -> Result<ChunkStream, ProviderError> {
        let url = format!("{}/openai/v1/chat/completions", self.base_url);
        // Groq uses the same SSE wire format as OpenAI: force `stream:true` and
        // request usage on the final chunk so observability records real tokens.
        let mut body = serde_json::to_value(&request)?;
        body["stream"] = json!(true);
        body["stream_options"] = json!({ "include_usage": true });
        // Strip the Anthropic-only cache marker before egress.
        crate::openai::strip_cache_control_for_openai(&mut body);

        let resp = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {api_key}"))
            .json(&body)
            .send()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("groq", e))?;

        if !resp.status().is_success() {
            return Err(crate::client::error_from_response("groq", resp).await);
        }

        Ok(Box::pin(openai_sse_to_chunks(resp.bytes_stream())))
    }

    /// Groq has no first-party embeddings endpoint — degrade explicitly to a
    /// typed 422 (`embeddings_not_supported`), never a silent drop or a panic.
    async fn embeddings(
        &self,
        _request: EmbeddingRequest,
        _api_key: String,
    ) -> Result<EmbeddingResponse, ProviderError> {
        Err(ProviderError::embeddings_not_supported(self.name()))
    }

    /// Groq's Whisper (`whisper-large-v3` / `whisper-large-v3-turbo`) is the
    /// flagship fast/cheap STT use case. The surface is OpenAI-wire-identical, so
    /// the shared `transcribe_multipart` helper does the work — only the
    /// `/openai/v1/...` path prefix (the base-URL footgun) differs.
    async fn transcribe(
        &self,
        audio: TranscriptionInput,
        api_key: String,
    ) -> Result<TranscriptionResponse, ProviderError> {
        let url = format!("{}/openai/v1/audio/transcriptions", self.base_url);
        transcribe_multipart(&self.client, "groq", &url, audio, api_key).await
    }

    /// Groq's `whisper-large-v3` supports translations (speech-in-any-language →
    /// English text). The surface is OpenAI-wire-identical to translations — only
    /// the `/openai/v1/...` path prefix (the base-URL footgun) differs. The shared
    /// helper omits the `language` field (translations output is always English).
    async fn translate(
        &self,
        audio: TranscriptionInput,
        api_key: String,
    ) -> Result<TranscriptionResponse, ProviderError> {
        let url = format!("{}/openai/v1/audio/translations", self.base_url);
        translate_multipart(&self.client, "groq", &url, audio, api_key).await
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
    fn name_is_groq() {
        let p = GroqProvider::new();
        assert_eq!(p.name(), "groq");
    }

    #[test]
    fn region_is_opt_in_and_empty_by_default() {
        // No GROQ_REGION set in this test process ⇒ no residency guarantee.
        let p = GroqProvider::new();
        assert!(p.resident_regions().is_empty());
        assert!(!p.is_resident_in("US"));
    }

    #[tokio::test]
    async fn buffered_call_hits_openai_v1_path_with_bearer_auth() {
        let server = MockServer::start().await;
        let resp = serde_json::json!({
            "id": "chatcmpl-groq-1",
            "object": "chat.completion",
            "created": 1700000000u64,
            "model": "llama-3.3-70b-versatile",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "fast hello from Groq"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
        });
        // The /openai/v1 prefix is the footgun — assert the full path is hit.
        Mock::given(method("POST"))
            .and(path("/openai/v1/chat/completions"))
            .and(header("authorization", "Bearer gsk-test"))
            .and(body_partial_json(
                serde_json::json!({ "model": "llama-3.3-70b-versatile" }),
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(resp))
            .mount(&server)
            .await;

        let p = GroqProvider::with_base_url(server.uri());
        let out = p
            .chat_completion(req("llama-3.3-70b-versatile"), "gsk-test".into())
            .await
            .expect("mock call succeeds");
        assert_eq!(
            out.choices[0].message.content.as_text(),
            "fast hello from Groq"
        );
        assert_eq!(out.usage.total_tokens, 7);
    }

    #[tokio::test]
    async fn upstream_429_is_typed_rate_limited_without_leaking_key() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/openai/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
            .mount(&server)
            .await;

        let p = GroqProvider::with_base_url(server.uri());
        let err = p
            .chat_completion(req("llama-3.1-8b-instant"), "gsk-test".into())
            .await
            .expect_err("429 should be an Err");
        assert_eq!(err.retry_class(), RetryClass::Status(429));
        assert!(!err.to_string().contains("gsk-test"));
    }

    #[tokio::test]
    async fn transcribe_hits_openai_v1_audio_path_with_multipart() {
        use routeplane_types::{TranscriptionInput, TranscriptionParams};
        use wiremock::matchers::{header, header_exists, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        let resp = serde_json::json!({ "text": "fast whisper from Groq" });

        // The /openai/v1 prefix is the footgun — assert the FULL Groq audio path
        // is hit with a multipart body + Bearer auth.
        Mock::given(method("POST"))
            .and(path("/openai/v1/audio/transcriptions"))
            .and(header("authorization", "Bearer gsk-test"))
            .and(header_exists("content-type"))
            .respond_with(ResponseTemplate::new(200).set_body_json(resp))
            .mount(&server)
            .await;

        let p = GroqProvider::with_base_url(server.uri());
        let audio = TranscriptionInput {
            file_bytes: b"fake-audio".to_vec(),
            filename: "speech.m4a".into(),
            params: TranscriptionParams {
                model: "whisper-large-v3".into(),
                ..Default::default()
            },
        };
        let out = p
            .transcribe(audio, "gsk-test".into())
            .await
            .expect("mock call succeeds");
        assert_eq!(out.text, "fast whisper from Groq");
    }

    #[tokio::test]
    async fn translate_hits_openai_v1_audio_translations_path() {
        use routeplane_types::{TranscriptionInput, TranscriptionParams};
        use wiremock::matchers::{header, header_exists, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        let resp = serde_json::json!({ "text": "fast english from Groq" });

        // The /openai/v1 prefix is the footgun — assert the FULL Groq audio
        // TRANSLATIONS path is hit with a multipart body + Bearer auth.
        Mock::given(method("POST"))
            .and(path("/openai/v1/audio/translations"))
            .and(header("authorization", "Bearer gsk-test"))
            .and(header_exists("content-type"))
            .respond_with(ResponseTemplate::new(200).set_body_json(resp))
            .mount(&server)
            .await;

        let p = GroqProvider::with_base_url(server.uri());
        let audio = TranscriptionInput {
            file_bytes: b"fake-audio".to_vec(),
            filename: "speech.m4a".into(),
            params: TranscriptionParams {
                model: "whisper-large-v3".into(),
                // Present but must not be sent for translations.
                language: Some("de".into()),
                ..Default::default()
            },
        };
        let out = p
            .translate(audio, "gsk-test".into())
            .await
            .expect("mock call succeeds");
        assert_eq!(out.text, "fast english from Groq");

        // No `language` field in the outbound multipart body.
        let received = &server.received_requests().await.unwrap()[0];
        let body = String::from_utf8_lossy(&received.body);
        assert!(body.contains("whisper-large-v3"));
        assert!(
            !body.contains("name=\"language\""),
            "translations must NOT send a `language` field"
        );
    }

    #[tokio::test]
    async fn embeddings_are_unsupported_422_not_a_panic() {
        let p = GroqProvider::new();
        let request = EmbeddingRequest {
            model: "whatever".into(),
            input: routeplane_types::EmbeddingInput::Single("hello".into()),
            user: None,
            encoding_format: None,
            dimensions: None,
        };
        let err = p
            .embeddings(request, "gsk-test".into())
            .await
            .expect_err("groq has no embeddings endpoint");
        assert_eq!(err.status(), Some(422));
        assert!(err.to_string().contains("embeddings_not_supported"));
    }
}
