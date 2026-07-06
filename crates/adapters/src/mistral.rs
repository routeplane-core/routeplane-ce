use crate::openai::openai_sse_to_chunks;
use crate::{ChunkStream, Provider, ProviderError};
use async_trait::async_trait;
use reqwest::Client;
use routeplane_types::{
    ChatCompletionRequest, ChatCompletionResponse, EmbeddingRequest, EmbeddingResponse,
};
use serde_json::json;

/// Mistral AI adapter — OpenAI-compatible wire format (Mistral's La Plateforme
/// exposes `/v1/chat/completions` and `/v1/embeddings` with the same shape as
/// OpenAI). Models: `mistral-large-latest`, `mistral-small-latest`,
/// `codestral-latest`, `open-mistral-nemo`, etc.
///
/// Mistral is a major European LLM provider — adding it broadens the sovereign
/// routing story (EU-hosted inference for GDPR-resident traffic).
pub struct MistralProvider {
    client: Client,
    base_url: String,
}

const MISTRAL_DEFAULT_BASE_URL: &str = "https://api.mistral.ai";

impl MistralProvider {
    pub fn new() -> Self {
        Self {
            client: crate::client::build_provider_client(),
            base_url: MISTRAL_DEFAULT_BASE_URL.to_string(),
        }
    }

    pub fn with_base_url(base_url: impl Into<String>) -> Self {
        Self {
            client: crate::client::build_provider_client(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
        }
    }
}

impl Default for MistralProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Provider for MistralProvider {
    fn name(&self) -> &'static str {
        "mistral"
    }

    /// Mistral's EU endpoints serve EU residents — set `MISTRAL_REGION` to
    /// declare residency (e.g. "EU" for GDPR routing).
    fn resident_regions(&self) -> Vec<String> {
        let region = std::env::var("MISTRAL_REGION").unwrap_or_default();
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
        let url = format!("{}/v1/chat/completions", self.base_url);

        // Strip the Anthropic-only `cache_control` marker before egress (Mistral
        // does not accept it). No-op when absent ⇒ byte-identical.
        let mut body = serde_json::to_value(&request)?;
        crate::openai::strip_cache_control_for_openai(&mut body);

        let response = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {api_key}"))
            .json(&body)
            .send()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("mistral", e))?;

        if !response.status().is_success() {
            return Err(crate::client::error_from_response("mistral", response).await);
        }

        response
            .json::<ChatCompletionResponse>()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("mistral", e))
    }

    async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
        api_key: String,
    ) -> Result<ChunkStream, ProviderError> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        // Mistral uses the same SSE wire format as OpenAI.
        let mut body = serde_json::to_value(&request)?;
        body["stream"] = json!(true);
        body["stream_options"] = json!({ "include_usage": true });
        // Strip the Anthropic-only cache marker before egress (Mistral rejects it).
        crate::openai::strip_cache_control_for_openai(&mut body);

        let resp = crate::client::streaming_client()
            .post(url)
            .header("Authorization", format!("Bearer {api_key}"))
            .json(&body)
            .send()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("mistral", e))?;

        if !resp.status().is_success() {
            return Err(crate::client::error_from_response("mistral", resp).await);
        }

        Ok(Box::pin(openai_sse_to_chunks(resp.bytes_stream())))
    }

    async fn embeddings(
        &self,
        request: EmbeddingRequest,
        api_key: String,
    ) -> Result<EmbeddingResponse, ProviderError> {
        let url = format!("{}/v1/embeddings", self.base_url);

        let response = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {api_key}"))
            .json(&request)
            .send()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("mistral", e))?;

        if !response.status().is_success() {
            return Err(crate::client::error_from_response("mistral", response).await);
        }

        response
            .json::<EmbeddingResponse>()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("mistral", e))
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
    fn name_is_mistral() {
        let p = MistralProvider::new();
        assert_eq!(p.name(), "mistral");
    }

    #[tokio::test]
    async fn buffered_call_hits_mistral_path_with_bearer_auth() {
        let server = MockServer::start().await;
        let resp = serde_json::json!({
            "id": "cmpl-m-1",
            "object": "chat.completion",
            "created": 1700000000u64,
            "model": "mistral-small-latest",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Bonjour from Mistral"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
        });
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(header("authorization", "Bearer sk-mistral"))
            .and(body_partial_json(
                serde_json::json!({ "model": "mistral-small-latest" }),
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(resp))
            .mount(&server)
            .await;

        let p = MistralProvider::with_base_url(server.uri());
        let out = p
            .chat_completion(req("mistral-small-latest"), "sk-mistral".into())
            .await
            .expect("mock call succeeds");
        assert_eq!(
            out.choices[0].message.content.as_text(),
            "Bonjour from Mistral"
        );
        assert_eq!(out.usage.total_tokens, 7);
    }

    #[tokio::test]
    async fn upstream_429_is_typed_rate_limited() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
            .mount(&server)
            .await;

        let p = MistralProvider::with_base_url(server.uri());
        let err = p
            .chat_completion(req("mistral-small-latest"), "sk-mistral".into())
            .await
            .expect_err("429 should be an Err");
        assert_eq!(err.retry_class(), RetryClass::Status(429));
        assert!(!err.to_string().contains("sk-mistral"));
    }

    // --- vision passthrough (OpenAI-shaped image_url parts) -------------------

    #[tokio::test]
    async fn forwards_image_url_part_to_mistral_wire() {
        use routeplane_types::{ContentPart, ImageUrlContent, MessageContent};

        let server = MockServer::start().await;
        let resp = serde_json::json!({
            "id": "cmpl-m-2",
            "object": "chat.completion",
            "created": 1700000000u64,
            "model": "pixtral-12b-2409",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "a tower"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10}
        });
        // The OpenAI-shaped content part must reach Mistral verbatim.
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(body_partial_json(serde_json::json!({
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/eiffel.jpg"}}
                    ]
                }]
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(resp))
            .mount(&server)
            .await;

        let p = MistralProvider::with_base_url(server.uri());
        let mut r = req("pixtral-12b-2409");
        r.messages = vec![Message {
            role: "user".into(),
            content: MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "what is this".into(),
                    cache_control: None,
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrlContent {
                        url: "https://example.com/eiffel.jpg".into(),
                        detail: None,
                    },
                },
            ]),
            name: None,
            cache_control: None,
            tool_calls: None,
            tool_call_id: None,
        }];
        let out = p
            .chat_completion(r, "sk-mistral".into())
            .await
            .expect("mock call succeeds");
        assert_eq!(out.choices[0].message.content.as_text(), "a tower");
    }

    #[test]
    fn text_only_message_serializes_as_bare_string_byte_identical() {
        // Mistral serializes the canonical request directly; a text-only message
        // must remain a bare string on the wire (no array), so existing behaviour
        // is byte-identical.
        let v = serde_json::to_value(req("mistral-small-latest")).unwrap();
        assert_eq!(v["messages"][0]["content"], serde_json::json!("hi"));
        assert!(v["messages"][0]["content"].is_string());
    }

    #[tokio::test]
    async fn embeddings_passthrough() {
        let server = MockServer::start().await;
        let resp = serde_json::json!({
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
            "model": "mistral-embed",
            "usage": {"prompt_tokens": 2, "total_tokens": 2}
        });
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .and(header("authorization", "Bearer sk-mistral"))
            .respond_with(ResponseTemplate::new(200).set_body_json(resp))
            .mount(&server)
            .await;

        let p = MistralProvider::with_base_url(server.uri());
        let request = EmbeddingRequest {
            model: "mistral-embed".into(),
            input: routeplane_types::EmbeddingInput::Single("hello".into()),
            user: None,
            encoding_format: None,
            dimensions: None,
        };
        let out = p
            .embeddings(request, "sk-mistral".into())
            .await
            .expect("embeddings passthrough");
        assert_eq!(out.data.len(), 1);
        assert_eq!(out.model, "mistral-embed");
    }
}
