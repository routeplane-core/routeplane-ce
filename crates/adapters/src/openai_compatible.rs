use crate::openai::openai_sse_to_chunks;
use crate::{ChunkStream, Provider, ProviderError, SpeechAudio};
use async_trait::async_trait;
use reqwest::Client;
use routeplane_types::{
    ChatCompletionRequest, ChatCompletionResponse, EmbeddingRequest, EmbeddingResponse,
    RerankRequest, RerankResponse, SpeechRequest,
};
use serde_json::json;

/// Generic OpenAI-compatible adapter for **self-hosted / open-weights** inference
/// servers — vLLM, Ollama, LocalAI, TGI, LM Studio, or any gateway that speaks the
/// OpenAI `/v1/chat/completions` + `/v1/embeddings` dialect.
///
/// These servers are wire-identical to OpenAI, so the canonical `routeplane_types`
/// models serialize/deserialize directly and the SSE translation is shared with
/// [`crate::openai`] (`openai_sse_to_chunks`) — only the base URL, the provider
/// name, and the (optional) residency jurisdiction differ.
///
/// Self-hosting is a first-class sovereignty story: when the operator runs the
/// model inside their own jurisdiction/VPC, set `SELF_HOSTED_REGION` so the
/// residency engine can route regulated traffic here and keep it in-region.
/// Default region is empty (no residency guarantee), the conservative default.
pub struct SelfHostedProvider {
    client: Client,
    /// Base URL of the OpenAI-compatible server, e.g. `http://vllm.internal:8000`
    /// or Ollama's `http://localhost:11434`. The `/v1/...` path is appended here.
    base_url: String,
    /// Residency jurisdiction this deployment serves (e.g. "IN"); empty ⇒ none.
    region: String,
    /// Whether a streaming request asks for `stream_options.include_usage`.
    /// Default true (OpenAI/vLLM honour it and it gives real token counts); a
    /// strict older OpenAI-ish runtime that 400s on the unknown field can opt
    /// out via [`without_stream_usage`], so it doesn't lose ALL streaming (#35).
    stream_include_usage: bool,
}

/// Whether `SELF_HOSTED_STREAM_INCLUDE_USAGE` explicitly opts OUT of
/// `stream_options.include_usage`. Default (unset/anything else) keeps it ON.
/// Pure so it is unit-testable without mutating the process environment.
fn stream_usage_disabled(val: Option<String>) -> bool {
    val.map(|v| {
        matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "off" | "false" | "no"
        )
    })
    .unwrap_or(false)
}

impl SelfHostedProvider {
    pub fn new(base_url: String, region: String) -> Self {
        Self {
            client: crate::client::build_provider_client(),
            // Normalize a trailing slash so `{base}/v1/...` never doubles it — the
            // env/from_env path previously skipped this (only test ctors trimmed).
            base_url: base_url.trim_end_matches('/').to_string(),
            region,
            stream_include_usage: true,
        }
    }

    /// Opt out of `stream_options.include_usage` for a strict upstream that
    /// rejects the field (loses per-request token accounting on that provider,
    /// but keeps streaming working).
    #[must_use]
    pub fn without_stream_usage(mut self) -> Self {
        self.stream_include_usage = false;
        self
    }

    /// Build from environment:
    ///   SELF_HOSTED_BASE_URL (e.g. http://vllm.internal:8000),
    ///   SELF_HOSTED_REGION   (residency jurisdiction; default empty = none),
    ///   SELF_HOSTED_STREAM_INCLUDE_USAGE=off  (opt out of
    ///     `stream_options.include_usage` for a strict runtime that 400s on it,
    ///     mirroring the custom-provider config field — #35).
    pub fn from_env() -> Self {
        let p = Self::new(
            std::env::var("SELF_HOSTED_BASE_URL").unwrap_or_default(),
            std::env::var("SELF_HOSTED_REGION").unwrap_or_default(),
        );
        if stream_usage_disabled(std::env::var("SELF_HOSTED_STREAM_INCLUDE_USAGE").ok()) {
            p.without_stream_usage()
        } else {
            p
        }
    }

    /// Test constructor pointing at a custom base URL (e.g. a wiremock server),
    /// no residency claim.
    pub fn with_base_url(base_url: impl Into<String>) -> Self {
        Self::new(base_url.into(), String::new())
    }
}

#[async_trait]
impl Provider for SelfHostedProvider {
    fn name(&self) -> &'static str {
        "self_hosted"
    }

    fn resident_regions(&self) -> Vec<String> {
        if self.region.is_empty() {
            Vec::new()
        } else {
            vec![self.region.clone()]
        }
    }

    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
        api_key: String,
    ) -> Result<ChatCompletionResponse, ProviderError> {
        if self.base_url.is_empty() {
            return Err("self-hosted endpoint not configured (set SELF_HOSTED_BASE_URL)".into());
        }
        let url = format!("{}/v1/chat/completions", self.base_url);

        // Strip the Anthropic-only `cache_control` marker before egress (a generic
        // OpenAI-compatible endpoint does not accept it).
        let mut body = serde_json::to_value(&request)?;
        crate::openai::strip_cache_control_for_openai(&mut body);

        let response = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {api_key}"))
            .json(&body)
            .send()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("self_hosted", e))?;

        if !response.status().is_success() {
            return Err(crate::client::error_from_response("self_hosted", response).await);
        }

        // Lift OpenAI's nested `usage.prompt_tokens_details.cached_tokens` into the
        // canonical flat `usage.cached_tokens`, like the OpenAI-wire adapters — the
        // direct typed deserialize dropped it (FinOps undercount vs the streaming
        // path; vLLM prefix caching reports it). No-op when the block is absent.
        let mut v: serde_json::Value = response
            .json()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("self_hosted", e))?;
        crate::openai::lift_openai_cached_tokens(&mut v);
        serde_json::from_value::<ChatCompletionResponse>(v).map_err(|e| -> ProviderError {
            format!("self_hosted response parse error: {e}").into()
        })
    }

    async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
        api_key: String,
    ) -> Result<ChunkStream, ProviderError> {
        if self.base_url.is_empty() {
            return Err("self-hosted endpoint not configured (set SELF_HOSTED_BASE_URL)".into());
        }
        let url = format!("{}/v1/chat/completions", self.base_url);
        // Wire-identical to OpenAI: force `stream:true` and request usage on the
        // final chunk so observability records real token counts.
        let mut body = serde_json::to_value(&request)?;
        body["stream"] = json!(true);
        if self.stream_include_usage {
            body["stream_options"] = json!({ "include_usage": true });
        }
        // Strip the Anthropic-only cache marker before egress.
        crate::openai::strip_cache_control_for_openai(&mut body);

        let resp = crate::client::streaming_client()
            .post(url)
            .header("Authorization", format!("Bearer {api_key}"))
            .json(&body)
            .send()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("self_hosted", e))?;

        if !resp.status().is_success() {
            return Err(crate::client::error_from_response("self_hosted", resp).await);
        }

        Ok(Box::pin(openai_sse_to_chunks(resp.bytes_stream())))
    }

    async fn embeddings(
        &self,
        request: EmbeddingRequest,
        api_key: String,
    ) -> Result<EmbeddingResponse, ProviderError> {
        if self.base_url.is_empty() {
            return Err("self-hosted endpoint not configured (set SELF_HOSTED_BASE_URL)".into());
        }
        // OpenAI-compatible servers expose `/v1/embeddings` with the canonical
        // request/response shape (vLLM, LocalAI); a typed passthrough, no translation.
        let url = format!("{}/v1/embeddings", self.base_url);

        let response = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {api_key}"))
            .json(&request)
            .send()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("self_hosted", e))?;

        if !response.status().is_success() {
            return Err(crate::client::error_from_response("self_hosted", response).await);
        }

        response
            .json::<EmbeddingResponse>()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("self_hosted", e))
    }

    async fn rerank(
        &self,
        request: RerankRequest,
        api_key: String,
    ) -> Result<RerankResponse, ProviderError> {
        if self.base_url.is_empty() {
            return Err("self-hosted endpoint not configured (set SELF_HOSTED_BASE_URL)".into());
        }
        // A self-hosted reranker (e.g. a TEI/Infinity cross-encoder, or a local
        // Cohere-compatible server) exposes `/v1/rerank` with the canonical
        // Cohere/LiteLLM shape, so this is a typed passthrough — no translation.
        let url = format!("{}/v1/rerank", self.base_url);

        let response = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {api_key}"))
            .json(&request)
            .send()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("self_hosted", e))?;

        if !response.status().is_success() {
            return Err(crate::client::error_from_response("self_hosted", response).await);
        }

        response
            .json::<RerankResponse>()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("self_hosted", e))
    }

    async fn speech(
        &self,
        request: SpeechRequest,
        api_key: String,
    ) -> Result<SpeechAudio, ProviderError> {
        if self.base_url.is_empty() {
            return Err("self-hosted endpoint not configured (set SELF_HOSTED_BASE_URL)".into());
        }
        // OpenAI-compatible servers expose `/v1/audio/speech` with the canonical
        // `{model?, input, voice, ...}` request and BINARY audio out. A FAITHFUL
        // passthrough: no default model is injected (unlike the first-party
        // OpenAI adapter) — the upstream validates its own model/endpoint
        // support and its error surfaces cleanly as a typed ProviderError.
        // The audio bytes and the api_key are NEVER logged.
        let requested_format = request.response_format.clone();
        let url = format!("{}/v1/audio/speech", self.base_url);

        let response = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {api_key}"))
            .json(&request)
            .send()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("self_hosted", e))?;

        if !response.status().is_success() {
            return Err(crate::client::error_from_response("self_hosted", response).await);
        }

        // Prefer the upstream Content-Type; fall back to deriving it from the
        // requested response_format (default mp3 ⇒ audio/mpeg) — the same
        // convention as the first-party OpenAI adapter.
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string())
            .unwrap_or_else(|| {
                crate::openai::content_type_for_format(requested_format.as_deref()).to_string()
            });

        let bytes = response
            .bytes()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("self_hosted", e))?;

        Ok(SpeechAudio {
            bytes: bytes.to_vec(),
            content_type,
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
    fn stream_usage_disabled_only_on_explicit_off_values() {
        for v in ["off", "OFF", "0", "false", "No", " off "] {
            assert!(
                stream_usage_disabled(Some(v.to_string())),
                "{v:?} should disable"
            );
        }
        for v in ["on", "1", "true", "yes", "", "banana"] {
            assert!(
                !stream_usage_disabled(Some(v.to_string())),
                "{v:?} should keep default-on"
            );
        }
        assert!(!stream_usage_disabled(None), "unset keeps default-on");
    }

    #[test]
    fn name_is_self_hosted_and_region_drives_residency() {
        let p = SelfHostedProvider::new("http://vllm.internal:8000".into(), "IN".into());
        assert_eq!(p.name(), "self_hosted");
        assert!(p.is_resident_in("IN"));
        // No region configured ⇒ never eligible for sovereign routing.
        let none = SelfHostedProvider::with_base_url("http://localhost:11434");
        assert!(none.resident_regions().is_empty());
    }

    #[tokio::test]
    async fn unconfigured_base_url_is_a_clean_error_not_a_panic() {
        let p = SelfHostedProvider::new(String::new(), String::new());
        let err = p
            .chat_completion(req("llama3"), "k".into())
            .await
            .expect_err("empty base URL must error");
        assert!(err.to_string().contains("SELF_HOSTED_BASE_URL"));
    }

    #[tokio::test]
    async fn buffered_call_hits_openai_path_with_bearer_auth() {
        let server = MockServer::start().await;
        let resp = serde_json::json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1700000000u64,
            "model": "llama3",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "hello from vllm"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
        });
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(header("authorization", "Bearer sk-local"))
            .and(body_partial_json(serde_json::json!({ "model": "llama3" })))
            .respond_with(ResponseTemplate::new(200).set_body_json(resp))
            .mount(&server)
            .await;

        let p = SelfHostedProvider::with_base_url(server.uri());
        let out = p
            .chat_completion(req("llama3"), "sk-local".into())
            .await
            .expect("mock call succeeds");
        assert_eq!(out.choices[0].message.content.as_text(), "hello from vllm");
        assert_eq!(out.usage.total_tokens, 7);
    }

    #[tokio::test]
    async fn upstream_429_is_typed_rate_limited_without_leaking_key() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
            .mount(&server)
            .await;

        let p = SelfHostedProvider::with_base_url(server.uri());
        let err = p
            .chat_completion(req("llama3"), "sk-local".into())
            .await
            .expect_err("429 should be an Err");
        assert_eq!(err.retry_class(), RetryClass::Status(429));
        assert!(!err.to_string().contains("sk-local"));
    }

    #[tokio::test]
    async fn embeddings_passthrough_against_openai_compatible_server() {
        let server = MockServer::start().await;
        let resp = serde_json::json!({
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
            "model": "nomic-embed-text",
            "usage": {"prompt_tokens": 3, "total_tokens": 3}
        });
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .and(header("authorization", "Bearer sk-local"))
            .respond_with(ResponseTemplate::new(200).set_body_json(resp))
            .mount(&server)
            .await;

        let p = SelfHostedProvider::with_base_url(server.uri());
        let request = EmbeddingRequest {
            model: "nomic-embed-text".into(),
            input: routeplane_types::EmbeddingInput::Single("hello".into()),
            user: None,
            encoding_format: None,
            dimensions: None,
        };
        let out = p
            .embeddings(request, "sk-local".into())
            .await
            .expect("embeddings passthrough succeeds");
        assert_eq!(out.data.len(), 1);
        assert_eq!(out.model, "nomic-embed-text");
    }

    #[tokio::test]
    async fn streaming_default_requests_usage_in_final_chunk() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string("data: [DONE]\n\n"),
            )
            .mount(&server)
            .await;

        let p = SelfHostedProvider::with_base_url(server.uri());
        let _stream = p
            .chat_completion_stream(req("llama3"), "sk-local".into())
            .await
            .expect("stream call succeeds");

        let reqs = server.received_requests().await.expect("recorded requests");
        let body: serde_json::Value =
            serde_json::from_slice(&reqs[0].body).expect("outbound body is json");
        assert_eq!(body["stream"], serde_json::json!(true));
        assert_eq!(
            body["stream_options"],
            serde_json::json!({ "include_usage": true }),
            "the default must ask for usage on the final chunk"
        );
    }

    #[tokio::test]
    async fn without_stream_usage_omits_stream_options() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string("data: [DONE]\n\n"),
            )
            .mount(&server)
            .await;

        let p = SelfHostedProvider::with_base_url(server.uri()).without_stream_usage();
        let _stream = p
            .chat_completion_stream(req("llama3"), "sk-local".into())
            .await
            .expect("stream call succeeds");

        let reqs = server.received_requests().await.expect("recorded requests");
        let body: serde_json::Value =
            serde_json::from_slice(&reqs[0].body).expect("outbound body is json");
        assert_eq!(body["stream"], serde_json::json!(true));
        assert!(
            body.get("stream_options").is_none(),
            "a strict runtime 400s on the unknown field; the opt-out must omit it entirely"
        );
    }

    #[tokio::test]
    async fn speech_passthrough_sends_bearer_and_keeps_upstream_content_type() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/audio/speech"))
            .and(header("authorization", "Bearer sk-local"))
            .and(body_partial_json(
                serde_json::json!({ "model": "kokoro", "input": "hello", "voice": "alloy" }),
            ))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_bytes(b"FAKE-MP3".to_vec())
                    .insert_header("content-type", "audio/mpeg"),
            )
            .mount(&server)
            .await;

        let p = SelfHostedProvider::with_base_url(server.uri());
        let request: routeplane_types::SpeechRequest = serde_json::from_value(serde_json::json!({
            "model": "kokoro",
            "input": "hello",
            "voice": "alloy"
        }))
        .unwrap();
        let out = p
            .speech(request, "sk-local".into())
            .await
            .expect("speech passthrough succeeds");
        assert_eq!(out.bytes, b"FAKE-MP3");
        assert_eq!(out.content_type, "audio/mpeg");
    }

    #[tokio::test]
    async fn speech_upstream_error_is_typed_not_a_panic() {
        // Faithful passthrough: the upstream decides whether the model supports
        // TTS — its 400 surfaces as a typed ProviderError, never a panic.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/audio/speech"))
            .respond_with(ResponseTemplate::new(400).set_body_string("model has no audio output"))
            .mount(&server)
            .await;

        let p = SelfHostedProvider::with_base_url(server.uri());
        let request: routeplane_types::SpeechRequest =
            serde_json::from_value(serde_json::json!({ "input": "hello", "voice": "alloy" }))
                .unwrap();
        let err = p
            .speech(request, "sk-local".into())
            .await
            .expect_err("upstream 400 should be an Err");
        assert_eq!(err.status(), Some(400));
    }
}
