use crate::{ChunkStream, Provider, ProviderError};
use async_trait::async_trait;
use reqwest::Client;
use routeplane_types::{ChatCompletionRequest, ChatCompletionResponse};

/// AWS Bedrock adapter — targets Bedrock's OpenAI-compatible API endpoint
/// (available since late 2024 for supported models via the Bedrock Converse
/// API's OpenAI compatibility layer). This adapter uses the `/model/{model_id}
/// /v1/chat/completions` shape exposed by Bedrock's cross-region inference
/// profiles.
///
/// Authentication: Bedrock uses AWS SigV4, but the OpenAI-compatible layer
/// accepts a Bearer token (an IAM role session token or a proxy-forwarded
/// credential). Operators configure `BEDROCK_BASE_URL` (the region-specific
/// OpenAI-compatible endpoint) and `BEDROCK_REGION` for residency claims.
///
/// For environments that need full SigV4 signing, operators should deploy an
/// AWS API Gateway or LiteLLM proxy in front of Bedrock — the adapter then
/// talks to the proxy with a static key. This keeps the Rust workspace free
/// of AWS SDK deps (which would bump the MSRV past 1.86).
pub struct BedrockProvider {
    client: Client,
    base_url: String,
    region: String,
}

impl BedrockProvider {
    pub fn new() -> Self {
        Self {
            client: crate::client::build_provider_client(),
            base_url: std::env::var("BEDROCK_BASE_URL").unwrap_or_default(),
            region: std::env::var("BEDROCK_REGION").unwrap_or_default(),
        }
    }

    pub fn with_base_url(base_url: impl Into<String>, region: impl Into<String>) -> Self {
        Self {
            client: crate::client::build_provider_client(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            region: region.into(),
        }
    }
}

impl Default for BedrockProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Provider for BedrockProvider {
    fn name(&self) -> &'static str {
        "bedrock"
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
            return Err("bedrock endpoint not configured (set BEDROCK_BASE_URL)".into());
        }
        // Bedrock's OpenAI-compatible layer uses the same wire shape.
        let url = format!("{}/v1/chat/completions", self.base_url);

        // Strip the Anthropic-only `cache_control` marker before egress — Bedrock's
        // OpenAI-compat layer rejects unknown fields, so a request carrying it
        // (routed here after Anthropic) would 400. No-op when absent ⇒
        // byte-identical, matching every sibling OpenAI-wire adapter.
        let mut body = serde_json::to_value(&request)?;
        crate::openai::strip_cache_control_for_openai(&mut body);

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {api_key}"))
            .json(&body)
            .send()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("bedrock", e))?;

        if !response.status().is_success() {
            return Err(crate::client::error_from_response("bedrock", response).await);
        }

        response
            .json::<ChatCompletionResponse>()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("bedrock", e))
    }

    async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
        api_key: String,
    ) -> Result<ChunkStream, ProviderError> {
        if self.base_url.is_empty() {
            return Err("bedrock endpoint not configured (set BEDROCK_BASE_URL)".into());
        }
        // Bedrock's OpenAI-compatible layer uses OpenAI-shaped SSE.
        let url = format!("{}/v1/chat/completions", self.base_url);
        let mut body = serde_json::to_value(&request)?;
        body["stream"] = serde_json::json!(true);
        // Ask for usage on the terminal SSE chunk so streaming records real token
        // counts (like every other OpenAI-wire adapter), not zero.
        body["stream_options"] = serde_json::json!({ "include_usage": true });
        // Strip the Anthropic-only cache marker before egress (Bedrock rejects it).
        crate::openai::strip_cache_control_for_openai(&mut body);

        let resp = crate::client::streaming_client()
            .post(&url)
            .header("Authorization", format!("Bearer {api_key}"))
            .json(&body)
            .send()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("bedrock", e))?;

        if !resp.status().is_success() {
            return Err(crate::client::error_from_response("bedrock", resp).await);
        }

        Ok(Box::pin(crate::openai::openai_sse_to_chunks(
            resp.bytes_stream(),
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RetryClass;
    use routeplane_types::Message;
    use wiremock::matchers::{header, method, path};
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
    fn name_is_bedrock_and_region_drives_residency() {
        let p = BedrockProvider::with_base_url("https://bedrock.us-east-1.amazonaws.com", "US");
        assert_eq!(p.name(), "bedrock");
        assert!(p.is_resident_in("US"));
        assert!(!p.is_resident_in("EU"));
    }

    #[test]
    fn no_region_means_no_residency_claim() {
        let p = BedrockProvider::with_base_url("https://bedrock.example.com", "");
        assert!(p.resident_regions().is_empty());
    }

    #[tokio::test]
    async fn unconfigured_base_url_is_a_clean_error() {
        let p = BedrockProvider::with_base_url("", "");
        let err = p
            .chat_completion(req("anthropic.claude-3-sonnet"), "k".into())
            .await
            .expect_err("empty base URL must error");
        assert!(err.to_string().contains("BEDROCK_BASE_URL"));
    }

    #[tokio::test]
    async fn buffered_call_against_openai_compatible_endpoint() {
        let server = MockServer::start().await;
        let resp = serde_json::json!({
            "id": "br-1",
            "object": "chat.completion",
            "created": 1700000000u64,
            "model": "anthropic.claude-3-sonnet-20240229-v1:0",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello from Bedrock"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
        });
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(header("authorization", "Bearer br-token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(resp))
            .mount(&server)
            .await;

        let p = BedrockProvider::with_base_url(server.uri(), "US");
        let out = p
            .chat_completion(
                req("anthropic.claude-3-sonnet-20240229-v1:0"),
                "br-token".into(),
            )
            .await
            .expect("mock call succeeds");
        assert_eq!(
            out.choices[0].message.content.as_text(),
            "Hello from Bedrock"
        );
        assert_eq!(out.usage.total_tokens, 7);
    }

    #[tokio::test]
    async fn buffered_strips_anthropic_cache_control_before_egress() {
        let server = MockServer::start().await;
        let resp = serde_json::json!({
            "id": "br-2",
            "object": "chat.completion",
            "created": 1700000000u64,
            "model": "anthropic.claude-3-sonnet-20240229-v1:0",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
        });
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(resp))
            .mount(&server)
            .await;

        let p = BedrockProvider::with_base_url(server.uri(), "US");
        // A caller carrying the Anthropic-only marker (e.g. after routing off
        // Anthropic) — Bedrock's OpenAI-compat layer rejects unknown fields, so it
        // MUST be stripped before egress (as every sibling OpenAI-wire adapter does).
        let mut request = req("anthropic.claude-3-sonnet-20240229-v1:0");
        request.messages[0].cache_control = Some(serde_json::json!({"type": "ephemeral"}));
        p.chat_completion(request, "br-token".into())
            .await
            .expect("mock call succeeds");

        let received = &server.received_requests().await.unwrap()[0];
        let sent: serde_json::Value = serde_json::from_slice(&received.body).unwrap();
        assert!(
            sent["messages"][0].get("cache_control").is_none(),
            "cache_control must NOT be forwarded to Bedrock"
        );
    }

    #[tokio::test]
    async fn upstream_5xx_is_retryable() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(500).set_body_string("internal error"))
            .mount(&server)
            .await;

        let p = BedrockProvider::with_base_url(server.uri(), "US");
        let err = p
            .chat_completion(req("anthropic.claude-3-sonnet"), "k".into())
            .await
            .expect_err("500 should be an Err");
        assert_eq!(err.retry_class(), RetryClass::Status(500));
    }
}
