use crate::openai::openai_sse_to_chunks;
use crate::{ChunkStream, Provider, ProviderError};
use async_trait::async_trait;
use reqwest::Client;
use routeplane_types::{
    ChatCompletionRequest, ChatCompletionResponse, EmbeddingRequest, EmbeddingResponse,
};
use serde_json::json;

/// OpenRouter adapter — OpenAI-compatible **meta-aggregator** (one key →
/// hundreds of models behind a single OpenAI-shaped surface). It is the same
/// integration LiteLLM/Portkey ship, and Routeplane advertises it on the
/// marketing site, so this adapter closes the site-vs-reality gap.
///
/// Wire dialect: chat completions, SSE streaming and tool/function-calling all
/// follow OpenAI conventions, so the canonical `routeplane_types` models
/// serialize/deserialize directly and the SSE translation is shared with
/// [`crate::openai`] (`openai_sse_to_chunks`). Only the base URL, the key env,
/// two optional attribution headers and the model-id form differ from OpenAI.
///
/// **Base URL:** OpenRouter serves its OpenAI-compatible surface under
/// `https://openrouter.ai/api/v1` — the `/api/v1` is part of the host root. The
/// chat-completions path is therefore `{base}/api/v1/chat/completions`;
/// `OPENROUTER_DEFAULT_BASE_URL` holds the host root and each call appends the
/// full `/api/v1/...` path (the same host+path handling as the Groq footgun).
///
/// **Model ids** are `provider/model` form (e.g. `openai/gpt-4o`,
/// `anthropic/claude-sonnet-4`, `meta-llama/llama-3.3-70b-instruct`,
/// `deepseek/deepseek-chat`). The adapter is a pass-through: it forwards whatever
/// `model` the client sends and never rewrites it.
///
/// **Attribution headers:** OpenRouter recommends `HTTP-Referer` and `X-Title`
/// so requests are attributed to the calling app on its leaderboards. These are
/// OpenRouter's required attribution mechanism (not Routeplane's public
/// `x-routeplane-*` headers) and are harmless to send, so the adapter sets them
/// as static constants on every chat + stream request.
///
/// Embeddings: OpenRouter is chat-focused and has **no** first-party embeddings
/// endpoint, so `embeddings` degrades to a typed `embeddings_not_supported` 422
/// (same as Groq, DeepSeek and Anthropic) — never a panic.
pub struct OpenRouterProvider {
    client: Client,
    base_url: String,
}

/// Host root for OpenRouter's OpenAI-compatible API. The `/api/v1/...` path is
/// appended by each call — including the `/api/v1` (part of the host root).
const OPENROUTER_DEFAULT_BASE_URL: &str = "https://openrouter.ai";

/// Optional OpenRouter attribution header identifying the calling app on
/// OpenRouter's public leaderboards. Recommended, harmless, and distinct from
/// Routeplane's public `x-routeplane-*` headers.
const OPENROUTER_HTTP_REFERER: &str = "https://routeplane.ai";

/// Optional OpenRouter attribution header (the app title shown on OpenRouter's
/// leaderboards). Pairs with `HTTP-Referer`.
const OPENROUTER_X_TITLE: &str = "Routeplane";

impl OpenRouterProvider {
    pub fn new() -> Self {
        Self {
            client: crate::client::build_provider_client(),
            base_url: OPENROUTER_DEFAULT_BASE_URL.to_string(),
        }
    }

    /// Test/override constructor pointing at a custom base URL (e.g. a wiremock
    /// server). The `/api/v1/...` path is still appended, so tests assert the
    /// full OpenRouter path is hit.
    pub fn with_base_url(base_url: impl Into<String>) -> Self {
        Self {
            client: crate::client::build_provider_client(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
        }
    }
}

impl Default for OpenRouterProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Provider for OpenRouterProvider {
    fn name(&self) -> &'static str {
        "openrouter"
    }

    /// OpenRouter is a meta-aggregator that fans requests out to many upstream
    /// providers across jurisdictions and offers **no** data-residency guarantee.
    /// Residency is therefore opt-in via `OPENROUTER_REGION`; empty (the default)
    /// means no residency guarantee, so OpenRouter is **never** eligible when
    /// sovereign routing to a specific region is enforced unless an operator
    /// explicitly opts in. For a regulated-data gateway this conservative default
    /// is load-bearing — we do not declare an aggregator resident in any region
    /// by default.
    fn resident_regions(&self) -> Vec<String> {
        let region = std::env::var("OPENROUTER_REGION").unwrap_or_default();
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
        let url = format!("{}/api/v1/chat/completions", self.base_url);

        // Strip the Anthropic-only `cache_control` marker before egress.
        let mut body = serde_json::to_value(&request)?;
        crate::openai::strip_cache_control_for_openai(&mut body);

        let response = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {api_key}"))
            .header("HTTP-Referer", OPENROUTER_HTTP_REFERER)
            .header("X-Title", OPENROUTER_X_TITLE)
            .json(&body)
            .send()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("openrouter", e))?;

        if !response.status().is_success() {
            return Err(crate::client::error_from_response("openrouter", response).await);
        }

        response
            .json::<ChatCompletionResponse>()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("openrouter", e))
    }

    async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
        api_key: String,
    ) -> Result<ChunkStream, ProviderError> {
        let url = format!("{}/api/v1/chat/completions", self.base_url);
        // OpenRouter uses the same SSE wire format as OpenAI: force `stream:true`
        // and request usage on the final chunk so observability records real
        // tokens.
        let mut body = serde_json::to_value(&request)?;
        body["stream"] = json!(true);
        body["stream_options"] = json!({ "include_usage": true });
        // Strip the Anthropic-only cache marker before egress.
        crate::openai::strip_cache_control_for_openai(&mut body);

        let resp = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {api_key}"))
            .header("HTTP-Referer", OPENROUTER_HTTP_REFERER)
            .header("X-Title", OPENROUTER_X_TITLE)
            .json(&body)
            .send()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("openrouter", e))?;

        if !resp.status().is_success() {
            return Err(crate::client::error_from_response("openrouter", resp).await);
        }

        Ok(Box::pin(openai_sse_to_chunks(resp.bytes_stream())))
    }

    /// OpenRouter has no first-party embeddings endpoint (it is chat-focused) —
    /// degrade explicitly to a typed 422 (`embeddings_not_supported`), never a
    /// silent drop or a panic.
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
    fn name_is_openrouter() {
        let p = OpenRouterProvider::new();
        assert_eq!(p.name(), "openrouter");
    }

    #[test]
    fn region_is_opt_in_and_empty_by_default() {
        // No OPENROUTER_REGION set in this test process ⇒ no residency guarantee,
        // so the aggregator is never eligible under sovereign routing.
        let p = OpenRouterProvider::new();
        assert!(p.resident_regions().is_empty());
        assert!(!p.is_resident_in("US"));
        assert!(!p.is_resident_in("IN"));
    }

    #[tokio::test]
    async fn buffered_call_hits_api_v1_path_with_bearer_and_attribution_headers() {
        let server = MockServer::start().await;
        let resp = serde_json::json!({
            "id": "chatcmpl-or-1",
            "object": "chat.completion",
            "created": 1700000000u64,
            "model": "openai/gpt-4o",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "hello from OpenRouter"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
        });
        // OpenRouter serves chat at {base}/api/v1/chat/completions — assert the
        // full path, Bearer auth, AND the two attribution headers are present.
        Mock::given(method("POST"))
            .and(path("/api/v1/chat/completions"))
            .and(header("authorization", "Bearer sk-or-test"))
            .and(header("http-referer", "https://routeplane.ai"))
            .and(header("x-title", "Routeplane"))
            .and(body_partial_json(
                serde_json::json!({ "model": "openai/gpt-4o" }),
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(resp))
            .mount(&server)
            .await;

        let p = OpenRouterProvider::with_base_url(server.uri());
        let out = p
            .chat_completion(req("openai/gpt-4o"), "sk-or-test".into())
            .await
            .expect("mock call succeeds");
        assert_eq!(
            out.choices[0].message.content.as_text(),
            "hello from OpenRouter"
        );
        assert_eq!(out.usage.total_tokens, 7);
    }

    #[tokio::test]
    async fn streaming_call_hits_api_v1_path_with_attribution_headers() {
        let server = MockServer::start().await;
        // Two SSE data lines + the terminal [DONE], OpenAI wire format.
        let sse_body = concat!(
            "data: {\"id\":\"or-s1\",\"object\":\"chat.completion.chunk\",",
            "\"created\":1700000000,\"model\":\"anthropic/claude-sonnet-4\",",
            "\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},",
            "\"finish_reason\":null}]}\n\n",
            "data: [DONE]\n\n"
        );
        Mock::given(method("POST"))
            .and(path("/api/v1/chat/completions"))
            .and(header("authorization", "Bearer sk-or-test"))
            .and(header("http-referer", "https://routeplane.ai"))
            .and(header("x-title", "Routeplane"))
            .and(body_partial_json(serde_json::json!({ "stream": true })))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse_body),
            )
            .mount(&server)
            .await;

        let p = OpenRouterProvider::with_base_url(server.uri());
        let mut stream = p
            .chat_completion_stream(req("anthropic/claude-sonnet-4"), "sk-or-test".into())
            .await
            .expect("stream establishes");

        use futures::StreamExt;
        let mut saw_content = false;
        while let Some(item) = stream.next().await {
            let chunk = item.expect("chunk parses");
            if let Some(choice) = chunk.choices.first() {
                if choice.delta.content.as_deref() == Some("hi") {
                    saw_content = true;
                }
            }
        }
        assert!(saw_content, "expected a content delta from the SSE stream");
    }

    #[tokio::test]
    async fn upstream_429_is_typed_rate_limited_without_leaking_key() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
            .mount(&server)
            .await;

        let p = OpenRouterProvider::with_base_url(server.uri());
        let err = p
            .chat_completion(
                req("meta-llama/llama-3.3-70b-instruct"),
                "sk-or-test".into(),
            )
            .await
            .expect_err("429 should be an Err");
        assert_eq!(err.retry_class(), RetryClass::Status(429));
        assert!(!err.to_string().contains("sk-or-test"));
    }

    #[tokio::test]
    async fn embeddings_are_unsupported_422_not_a_panic() {
        let p = OpenRouterProvider::new();
        let request = EmbeddingRequest {
            model: "whatever".into(),
            input: routeplane_types::EmbeddingInput::Single("hello".into()),
            user: None,
            encoding_format: None,
            dimensions: None,
        };
        let err = p
            .embeddings(request, "sk-or-test".into())
            .await
            .expect_err("openrouter has no embeddings endpoint");
        assert_eq!(err.status(), Some(422));
        assert!(err.to_string().contains("embeddings_not_supported"));
    }
}
