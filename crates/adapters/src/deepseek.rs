use crate::openai::openai_sse_to_chunks;
use crate::{ChunkStream, Provider, ProviderError};
use async_trait::async_trait;
use reqwest::Client;
use routeplane_types::{
    ChatCompletionRequest, ChatCompletionResponse, EmbeddingRequest, EmbeddingResponse,
};
use serde_json::json;

/// DeepSeek adapter — OpenAI-compatible wire format. DeepSeek serves frontier
/// open-weight chat + reasoning models (`deepseek-v4-flash`/`-pro`, plus the
/// legacy `deepseek-chat`/`deepseek-reasoner`) and is a natural cost/quality
/// fallback target for routing.
///
/// Wire dialect: chat completions, SSE streaming and tool/function-calling all
/// follow OpenAI conventions, so the canonical `routeplane_types` models
/// serialize/deserialize directly and the SSE translation is shared with
/// [`crate::openai`] (`openai_sse_to_chunks`). Only the base URL, the provider
/// name and the (absent) embeddings capability differ from OpenAI.
///
/// **Base URL:** DeepSeek's host root is `https://api.deepseek.com` and the
/// chat-completions path is `{base}/chat/completions` (the `/v1` suffix is only
/// an optional OpenAI-SDK compat alias — the base + `/chat/completions` is the
/// canonical form, so the adapter appends that path to the host root, exactly
/// like the OpenAI dialect).
///
/// **Reasoning models:** `deepseek-reasoner` (and the v4 reasoning tier) can
/// return an extra `reasoning_content` field on the message/delta. The canonical
/// `routeplane_types` models tolerate unknown response fields (they do not use
/// `deny_unknown_fields`), so this field is ignored on deserialize rather than
/// breaking the parse — no special-casing, no panic.
///
/// Embeddings: DeepSeek has **no** first-party embeddings endpoint, so
/// `embeddings` degrades to a typed `embeddings_not_supported` 422 (same as Groq
/// and Anthropic) — never a panic.
pub struct DeepSeekProvider {
    client: Client,
    base_url: String,
}

/// Host root for DeepSeek's OpenAI-compatible API. The `/chat/completions` path
/// is appended by each call (no `/v1` prefix — that is only an optional alias).
const DEEPSEEK_DEFAULT_BASE_URL: &str = "https://api.deepseek.com";

impl DeepSeekProvider {
    pub fn new() -> Self {
        Self {
            client: crate::client::build_provider_client(),
            base_url: DEEPSEEK_DEFAULT_BASE_URL.to_string(),
        }
    }

    /// Test/override constructor pointing at a custom base URL (e.g. a wiremock
    /// server). The `/chat/completions` path is still appended, so tests assert
    /// the full DeepSeek path is hit.
    pub fn with_base_url(base_url: impl Into<String>) -> Self {
        Self {
            client: crate::client::build_provider_client(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
        }
    }
}

impl Default for DeepSeekProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Provider for DeepSeekProvider {
    fn name(&self) -> &'static str {
        "deepseek"
    }

    /// DeepSeek is China-based and offers **no** data-residency guarantee for
    /// any specific jurisdiction. Residency is therefore opt-in via
    /// `DEEPSEEK_REGION`; empty (the default) means no residency guarantee, so
    /// DeepSeek is **never** eligible when sovereign routing to a specific
    /// region is enforced unless an operator explicitly opts in. For a
    /// regulated-data gateway this conservative default is load-bearing — we do
    /// not declare DeepSeek resident in the US or any region by default.
    fn resident_regions(&self) -> Vec<String> {
        let region = std::env::var("DEEPSEEK_REGION").unwrap_or_default();
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
            .map_err(|e| crate::client::sanitize_transport_error("deepseek", e))?;

        if !response.status().is_success() {
            return Err(crate::client::error_from_response("deepseek", response).await);
        }

        response
            .json::<ChatCompletionResponse>()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("deepseek", e))
    }

    async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
        api_key: String,
    ) -> Result<ChunkStream, ProviderError> {
        let url = format!("{}/chat/completions", self.base_url);
        // DeepSeek uses the same SSE wire format as OpenAI: force `stream:true`
        // and request usage on the final chunk so observability records real
        // tokens.
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
            .map_err(|e| crate::client::sanitize_transport_error("deepseek", e))?;

        if !resp.status().is_success() {
            return Err(crate::client::error_from_response("deepseek", resp).await);
        }

        Ok(Box::pin(openai_sse_to_chunks(resp.bytes_stream())))
    }

    /// DeepSeek has no first-party embeddings endpoint — degrade explicitly to a
    /// typed 422 (`embeddings_not_supported`), never a silent drop or a panic.
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
    fn name_is_deepseek() {
        let p = DeepSeekProvider::new();
        assert_eq!(p.name(), "deepseek");
    }

    #[test]
    fn region_is_opt_in_and_empty_by_default() {
        // No DEEPSEEK_REGION set in this test process ⇒ no residency guarantee,
        // so DeepSeek (China-based) is never eligible under sovereign routing.
        let p = DeepSeekProvider::new();
        assert!(p.resident_regions().is_empty());
        assert!(!p.is_resident_in("US"));
        assert!(!p.is_resident_in("CN"));
        assert!(!p.is_resident_in("IN"));
    }

    #[tokio::test]
    async fn buffered_call_hits_chat_completions_path_with_bearer_auth() {
        let server = MockServer::start().await;
        let resp = serde_json::json!({
            "id": "chatcmpl-ds-1",
            "object": "chat.completion",
            "created": 1700000000u64,
            "model": "deepseek-v4-pro",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "hello from DeepSeek"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
        });
        // DeepSeek serves chat at {base}/chat/completions (no /v1 prefix).
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(header("authorization", "Bearer sk-ds-test"))
            .and(body_partial_json(
                serde_json::json!({ "model": "deepseek-v4-pro" }),
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(resp))
            .mount(&server)
            .await;

        let p = DeepSeekProvider::with_base_url(server.uri());
        let out = p
            .chat_completion(req("deepseek-v4-pro"), "sk-ds-test".into())
            .await
            .expect("mock call succeeds");
        assert_eq!(
            out.choices[0].message.content.as_text(),
            "hello from DeepSeek"
        );
        assert_eq!(out.usage.total_tokens, 7);
    }

    #[tokio::test]
    async fn reasoning_content_field_passes_through_to_the_client() {
        // deepseek-reasoner returns an extra `reasoning_content` field on the
        // message; the canonical types carry it as a typed passthrough field, so
        // it must survive the parse AND reach the client — not be dropped.
        let server = MockServer::start().await;
        let resp = serde_json::json!({
            "id": "chatcmpl-ds-r1",
            "object": "chat.completion",
            "created": 1700000000u64,
            "model": "deepseek-reasoner",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "the answer is 42",
                    "reasoning_content": "let me think step by step ..."
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 9, "total_tokens": 14}
        });
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(resp))
            .mount(&server)
            .await;

        let p = DeepSeekProvider::with_base_url(server.uri());
        let out = p
            .chat_completion(req("deepseek-reasoner"), "sk-ds-test".into())
            .await
            .expect("reasoning_content field must not break deserialization");
        assert_eq!(out.choices[0].message.content.as_text(), "the answer is 42");
        // PASSTHROUGH: the reasoning text is surfaced on the canonical message,
        // not silently dropped (response/chunk passthrough).
        assert_eq!(
            out.choices[0].message.reasoning_content.as_deref(),
            Some("let me think step by step ...")
        );
        assert_eq!(out.usage.total_tokens, 14);
    }

    #[tokio::test]
    async fn upstream_429_is_typed_rate_limited_without_leaking_key() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
            .mount(&server)
            .await;

        let p = DeepSeekProvider::with_base_url(server.uri());
        let err = p
            .chat_completion(req("deepseek-v4-flash"), "sk-ds-test".into())
            .await
            .expect_err("429 should be an Err");
        assert_eq!(err.retry_class(), RetryClass::Status(429));
        assert!(!err.to_string().contains("sk-ds-test"));
    }

    #[tokio::test]
    async fn embeddings_are_unsupported_422_not_a_panic() {
        let p = DeepSeekProvider::new();
        let request = EmbeddingRequest {
            model: "whatever".into(),
            input: routeplane_types::EmbeddingInput::Single("hello".into()),
            user: None,
            encoding_format: None,
            dimensions: None,
        };
        let err = p
            .embeddings(request, "sk-ds-test".into())
            .await
            .expect_err("deepseek has no embeddings endpoint");
        assert_eq!(err.status(), Some(422));
        assert!(err.to_string().contains("embeddings_not_supported"));
    }
}
