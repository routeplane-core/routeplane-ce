use crate::openai::openai_sse_to_chunks;
use crate::{ChunkStream, Provider, ProviderError};
use async_trait::async_trait;
use reqwest::Client;
use routeplane_types::{
    ChatCompletionRequest, ChatCompletionResponse, EmbeddingRequest, EmbeddingResponse,
};
use serde_json::json;

/// Fireworks AI adapter — OpenAI-compatible wire format. Fireworks serves
/// open-weight models (Llama, Qwen, DeepSeek, Mixtral, …) behind one
/// OpenAI-compatible surface and — like Together (and unlike Groq/DeepSeek) —
/// ALSO exposes a first-party embeddings endpoint. It is a natural cost/breadth
/// fallback target for routing and closes the open-model breadth gap (a
/// confirmed provider gap vs LiteLLM/Portkey).
///
/// Wire dialect: chat completions, SSE streaming, tool/function-calling AND
/// embeddings all follow OpenAI conventions, so the canonical `routeplane_types`
/// models serialize/deserialize directly: the SSE translation is shared with
/// [`crate::openai`] (`openai_sse_to_chunks`) and the embeddings request/response
/// serialize 1:1 (a typed passthrough, exactly like [`crate::openai`]). Only the
/// base URL, the provider name and the key env differ.
///
/// **Base URL:** Fireworks' inference host root already includes the `/v1`
/// segment — `https://api.fireworks.ai/inference/v1` — so the chat path is
/// `{base}/chat/completions` and the embeddings path is `{base}/embeddings` (the
/// adapter appends only the trailing segment to the configured base, mirroring
/// Together/DeepSeek host-root handling). Model ids are NAMESPACED (e.g.
/// `accounts/fireworks/models/llama-v3p1-70b-instruct`), which the canonical
/// `model` string carries verbatim.
///
/// Embeddings: Fireworks' `/embeddings` IS the OpenAI dialect (`{model, input}`
/// in, `{object:"list", data:[{embedding,index}], usage}` out), so the call is a
/// typed passthrough — never a panic, never a translation.
pub struct FireworksProvider {
    client: Client,
    base_url: String,
}

/// Host root for Fireworks' OpenAI-compatible API. The `/inference/v1` segment is
/// part of the base; each call appends only the trailing path
/// (`/chat/completions`, `/embeddings`).
const FIREWORKS_DEFAULT_BASE_URL: &str = "https://api.fireworks.ai/inference/v1";

impl FireworksProvider {
    pub fn new() -> Self {
        Self {
            client: crate::client::build_provider_client(),
            base_url: FIREWORKS_DEFAULT_BASE_URL.to_string(),
        }
    }

    /// Test/override constructor pointing at a custom base URL (e.g. a wiremock
    /// server). The trailing path is still appended, so tests assert the full
    /// Fireworks path is hit.
    pub fn with_base_url(base_url: impl Into<String>) -> Self {
        Self {
            client: crate::client::build_provider_client(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
        }
    }
}

impl Default for FireworksProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Provider for FireworksProvider {
    fn name(&self) -> &'static str {
        "fireworks"
    }

    /// Fireworks AI inference is US-based today and offers no per-jurisdiction
    /// residency guarantee. Residency is therefore opt-in via `FIREWORKS_REGION`;
    /// empty (the default) means no residency guarantee, so Fireworks is **never**
    /// eligible when sovereign routing to a specific region is enforced unless an
    /// operator explicitly opts in. For a regulated-data gateway this conservative
    /// default is load-bearing — we do not declare Fireworks resident in the US or
    /// any region by default.
    fn resident_regions(&self) -> Vec<String> {
        let region = std::env::var("FIREWORKS_REGION").unwrap_or_default();
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
            .map_err(|e| crate::client::sanitize_transport_error("fireworks", e))?;

        if !response.status().is_success() {
            return Err(crate::client::error_from_response("fireworks", response).await);
        }

        response
            .json::<ChatCompletionResponse>()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("fireworks", e))
    }

    async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
        api_key: String,
    ) -> Result<ChunkStream, ProviderError> {
        let url = format!("{}/chat/completions", self.base_url);
        // Fireworks uses the same SSE wire format as OpenAI: force `stream:true`
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
            .map_err(|e| crate::client::sanitize_transport_error("fireworks", e))?;

        if !resp.status().is_success() {
            return Err(crate::client::error_from_response("fireworks", resp).await);
        }

        Ok(Box::pin(openai_sse_to_chunks(resp.bytes_stream())))
    }

    /// Fireworks' `/embeddings` IS the OpenAI dialect: the request and response
    /// serialize 1:1 to the canonical types, so this is a typed passthrough (no
    /// translation), mirroring [`crate::openai::OpenAIProvider::embeddings`] and
    /// [`crate::together::TogetherProvider::embeddings`]. The only difference is
    /// the base URL.
    async fn embeddings(
        &self,
        request: EmbeddingRequest,
        api_key: String,
    ) -> Result<EmbeddingResponse, ProviderError> {
        let url = format!("{}/embeddings", self.base_url);

        let response = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {api_key}"))
            .json(&request)
            .send()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("fireworks", e))?;

        if !response.status().is_success() {
            return Err(crate::client::error_from_response("fireworks", response).await);
        }

        response
            .json::<EmbeddingResponse>()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("fireworks", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RetryClass;
    use routeplane_types::{EmbeddingInput, Message};
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
    fn name_is_fireworks() {
        let p = FireworksProvider::new();
        assert_eq!(p.name(), "fireworks");
    }

    #[test]
    fn region_is_opt_in_and_empty_by_default() {
        // No FIREWORKS_REGION set in this test process ⇒ no residency guarantee,
        // so Fireworks (US-based) is never eligible under sovereign routing.
        let p = FireworksProvider::new();
        assert!(p.resident_regions().is_empty());
        assert!(!p.is_resident_in("US"));
    }

    #[tokio::test]
    async fn buffered_call_hits_chat_completions_path_with_bearer_auth() {
        let server = MockServer::start().await;
        let resp = serde_json::json!({
            "id": "chatcmpl-fw-1",
            "object": "chat.completion",
            "created": 1700000000u64,
            "model": "accounts/fireworks/models/llama-v3p1-70b-instruct",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "hello from Fireworks"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
        });
        // Fireworks serves chat at {base}/chat/completions (base already has
        // /inference/v1).
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(header("authorization", "Bearer fw-test"))
            .and(body_partial_json(serde_json::json!({
                "model": "accounts/fireworks/models/llama-v3p1-70b-instruct"
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(resp))
            .mount(&server)
            .await;

        let p = FireworksProvider::with_base_url(server.uri());
        let out = p
            .chat_completion(
                req("accounts/fireworks/models/llama-v3p1-70b-instruct"),
                "fw-test".into(),
            )
            .await
            .expect("mock call succeeds");
        assert_eq!(
            out.choices[0].message.content.as_text(),
            "hello from Fireworks"
        );
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

        let p = FireworksProvider::with_base_url(server.uri());
        let err = p
            .chat_completion(
                req("accounts/fireworks/models/qwen2p5-72b-instruct"),
                "fw-test".into(),
            )
            .await
            .expect_err("429 should be an Err");
        assert_eq!(err.retry_class(), RetryClass::Status(429));
        assert!(!err.to_string().contains("fw-test"));
    }

    #[tokio::test]
    async fn embeddings_hit_embeddings_path_and_map_vectors() {
        let server = MockServer::start().await;
        let resp = serde_json::json!({
            "object": "list",
            "data": [
                {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}
            ],
            "model": "nomic-ai/nomic-embed-text-v1.5",
            "usage": {"prompt_tokens": 2, "total_tokens": 2}
        });
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .and(header("authorization", "Bearer fw-test"))
            .and(body_partial_json(serde_json::json!({
                "model": "nomic-ai/nomic-embed-text-v1.5"
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(resp))
            .mount(&server)
            .await;

        let p = FireworksProvider::with_base_url(server.uri());
        let request = EmbeddingRequest {
            model: "nomic-ai/nomic-embed-text-v1.5".into(),
            input: EmbeddingInput::Single("hello".into()),
            user: None,
            encoding_format: None,
            dimensions: None,
        };
        let out = p
            .embeddings(request, "fw-test".into())
            .await
            .expect("embeddings call succeeds");
        assert_eq!(out.data.len(), 1);
        assert_eq!(out.data[0].embedding.len(), 3);
        assert_eq!(out.usage.total_tokens, 2);
    }
}
