use crate::openai::openai_sse_to_chunks;
use crate::{ChunkStream, Provider, ProviderError};
use async_trait::async_trait;
use reqwest::Client;
use routeplane_types::{
    ChatCompletionRequest, ChatCompletionResponse, EmbeddingRequest, EmbeddingResponse,
};
use serde_json::json;

/// xAI (Grok) adapter — OpenAI-compatible wire format. xAI serves the frontier
/// Grok chat + reasoning models (`grok-4.3`, `grok-4-0709`, `grok-3`,
/// `grok-3-fast`) and is a natural quality/breadth fallback target for routing.
///
/// Wire dialect: chat completions, SSE streaming and tool/function-calling all
/// follow OpenAI conventions, so the canonical `routeplane_types` models
/// serialize/deserialize directly and the SSE translation is shared with
/// [`crate::openai`] (`openai_sse_to_chunks`). Only the base URL, the provider
/// name and the (absent) embeddings capability differ from OpenAI.
///
/// **Base URL:** xAI's host root is `https://api.x.ai/v1` — the `/v1` is part of
/// the host root, and the chat-completions path is `{base}/chat/completions`
/// (i.e. `https://api.x.ai/v1/chat/completions`). The adapter appends
/// `/chat/completions` to that base, exactly like the DeepSeek dialect.
///
/// **Reasoning models:** the Grok reasoning tier can return extra fields (e.g.
/// `reasoning_content`) on the message/delta. The canonical `routeplane_types`
/// models tolerate unknown response fields (they do not use
/// `deny_unknown_fields`), so such a field is ignored on deserialize rather than
/// breaking the parse — no special-casing, no panic.
///
/// Embeddings: xAI has **no** first-party embeddings endpoint, so `embeddings`
/// degrades to a typed `embeddings_not_supported` 422 (same as Groq, DeepSeek
/// and Anthropic) — never a panic.
pub struct XaiProvider {
    client: Client,
    base_url: String,
}

/// Host root for xAI's OpenAI-compatible API. The `/v1` is part of the host root
/// and the `/chat/completions` path is appended by each call.
const XAI_DEFAULT_BASE_URL: &str = "https://api.x.ai/v1";

impl XaiProvider {
    pub fn new() -> Self {
        Self {
            client: crate::client::build_provider_client(),
            base_url: XAI_DEFAULT_BASE_URL.to_string(),
        }
    }

    /// Test/override constructor pointing at a custom base URL (e.g. a wiremock
    /// server). The `/chat/completions` path is still appended, so tests assert
    /// the full xAI path is hit.
    pub fn with_base_url(base_url: impl Into<String>) -> Self {
        Self {
            client: crate::client::build_provider_client(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
        }
    }
}

impl Default for XaiProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Provider for XaiProvider {
    fn name(&self) -> &'static str {
        "xai"
    }

    /// xAI inference is US-based today. Residency is opt-in via `XAI_REGION`
    /// (e.g. "US"); empty (the default) means no residency guarantee, so xAI is
    /// never eligible when sovereign routing to a specific region is enforced —
    /// the conservative default that keeps residency filtering correct.
    fn resident_regions(&self) -> Vec<String> {
        let region = std::env::var("XAI_REGION").unwrap_or_default();
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
        let url = format!("{}/chat/completions", self.base_url);

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
            .map_err(|e| crate::client::sanitize_transport_error("xai", e))?;

        if !response.status().is_success() {
            return Err(crate::client::error_from_response("xai", response).await);
        }

        response
            .json::<ChatCompletionResponse>()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("xai", e))
    }

    async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
        api_key: String,
    ) -> Result<ChunkStream, ProviderError> {
        let url = format!("{}/chat/completions", self.base_url);
        // xAI uses the same SSE wire format as OpenAI: force `stream:true` and
        // request usage on the final chunk so observability records real tokens.
        let mut body = serde_json::to_value(&request)?;
        body["stream"] = json!(true);
        body["stream_options"] = json!({ "include_usage": true });
        // Strip the Anthropic-only cache marker before egress.
        crate::openai::strip_cache_control_for_openai(&mut body);

        let resp = crate::client::streaming_client()
            .post(url)
            .header("Authorization", format!("Bearer {api_key}"))
            .json(&body)
            .send()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("xai", e))?;

        if !resp.status().is_success() {
            return Err(crate::client::error_from_response("xai", resp).await);
        }

        Ok(Box::pin(openai_sse_to_chunks(resp.bytes_stream())))
    }

    /// xAI has no first-party embeddings endpoint — degrade explicitly to a typed
    /// 422 (`embeddings_not_supported`), never a silent drop or a panic.
    async fn embeddings(
        &self,
        _request: EmbeddingRequest,
        _api_key: String,
    ) -> Result<EmbeddingResponse, ProviderError> {
        Err(ProviderError::embeddings_not_supported(self.name()))
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
    fn name_is_xai() {
        let p = XaiProvider::new();
        assert_eq!(p.name(), "xai");
    }

    #[test]
    fn region_is_opt_in_and_empty_by_default() {
        // No XAI_REGION set in this test process ⇒ no residency guarantee, so xAI
        // (US-based) is never eligible under sovereign routing unless opted in.
        let p = XaiProvider::new();
        assert!(p.resident_regions().is_empty());
        assert!(!p.is_resident_in("US"));
    }

    #[tokio::test]
    async fn buffered_call_hits_chat_completions_path_with_bearer_auth() {
        let server = MockServer::start().await;
        let resp = serde_json::json!({
            "id": "chatcmpl-xai-1",
            "object": "chat.completion",
            "created": 1700000000u64,
            "model": "grok-4.3",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "hello from Grok"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
        });
        // xAI serves chat at {base}/chat/completions (the /v1 is part of the host
        // root, supplied by the test base URL).
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(header("authorization", "Bearer xai-test"))
            .and(body_partial_json(
                serde_json::json!({ "model": "grok-4.3" }),
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(resp))
            .mount(&server)
            .await;

        let p = XaiProvider::with_base_url(server.uri());
        let out = p
            .chat_completion(req("grok-4.3"), "xai-test".into())
            .await
            .expect("mock call succeeds");
        assert_eq!(out.choices[0].message.content.as_text(), "hello from Grok");
        assert_eq!(out.usage.total_tokens, 7);
    }

    #[tokio::test]
    async fn upstream_429_is_typed_rate_limited_without_leaking_key() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
            .mount(&server)
            .await;

        let p = XaiProvider::with_base_url(server.uri());
        let err = p
            .chat_completion(req("grok-3-fast"), "xai-test".into())
            .await
            .expect_err("429 should be an Err");
        assert_eq!(err.retry_class(), RetryClass::Status(429));
        assert!(!err.to_string().contains("xai-test"));
    }

    #[tokio::test]
    async fn embeddings_are_unsupported_422_not_a_panic() {
        let p = XaiProvider::new();
        let request = EmbeddingRequest {
            model: "whatever".into(),
            input: routeplane_types::EmbeddingInput::Single("hello".into()),
            user: None,
            encoding_format: None,
            dimensions: None,
        };
        let err = p
            .embeddings(request, "xai-test".into())
            .await
            .expect_err("xai has no embeddings endpoint");
        assert_eq!(err.status(), Some(422));
        assert!(err.to_string().contains("embeddings_not_supported"));
    }
}
