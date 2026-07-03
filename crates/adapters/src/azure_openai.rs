use crate::openai::openai_sse_to_chunks;
use crate::{ChunkStream, Provider, ProviderError};
use async_trait::async_trait;
use reqwest::Client;
use routeplane_types::{
    ChatCompletionRequest, ChatCompletionResponse, EmbeddingRequest, EmbeddingResponse,
};
use serde_json::json;

/// Azure OpenAI adapter — the region-resident "sovereign" provider.
///
/// Azure OpenAI is OpenAI-compatible, so the canonical request/response models
/// serialize directly. It declares a residency jurisdiction (e.g. "IN" for an
/// India Central deployment) so the residency engine can route personal-data
/// traffic to it and keep that data in-region.
pub struct AzureOpenAiProvider {
    client: Client,
    endpoint: String,    // e.g. https://aigw-india.openai.azure.com
    deployment: String,  // deployment name, e.g. gpt-4o
    api_version: String, // e.g. 2024-10-21
    region: String,      // residency jurisdiction, e.g. "IN"
}

impl AzureOpenAiProvider {
    pub fn new(endpoint: String, deployment: String, api_version: String, region: String) -> Self {
        Self {
            client: crate::client::build_provider_client(),
            endpoint,
            deployment,
            api_version,
            region,
        }
    }

    /// Build from environment:
    ///   AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT,
    ///   AZURE_OPENAI_API_VERSION (default 2024-10-21),
    ///   AZURE_OPENAI_REGION (default "IN").
    pub fn from_env() -> Self {
        Self::new(
            std::env::var("AZURE_OPENAI_ENDPOINT").unwrap_or_default(),
            std::env::var("AZURE_OPENAI_DEPLOYMENT").unwrap_or_default(),
            std::env::var("AZURE_OPENAI_API_VERSION").unwrap_or_else(|_| "2024-10-21".to_string()),
            std::env::var("AZURE_OPENAI_REGION").unwrap_or_else(|_| "IN".to_string()),
        )
    }
}

#[async_trait]
impl Provider for AzureOpenAiProvider {
    fn name(&self) -> &'static str {
        "azure_openai"
    }

    fn resident_regions(&self) -> Vec<String> {
        vec![self.region.clone()]
    }

    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
        api_key: String,
    ) -> Result<ChatCompletionResponse, ProviderError> {
        if self.endpoint.is_empty() {
            return Err("Azure OpenAI endpoint not configured (set AZURE_OPENAI_ENDPOINT)".into());
        }

        let url = self.chat_url();

        // Azure OpenAI is OpenAI-wire-compatible: strip the Anthropic-only
        // `cache_control` marker before egress (Azure caches automatically and
        // rejects unknown fields). No-op when absent ⇒ byte-identical.
        let mut body = serde_json::to_value(&request)?;
        crate::openai::strip_cache_control_for_openai(&mut body);

        let response = self
            .client
            .post(&url)
            .header("api-key", api_key)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("azure_openai", e))?;

        if !response.status().is_success() {
            return Err(crate::client::error_from_response("azure_openai", response).await);
        }

        // Lift Azure/OpenAI's nested `prompt_tokens_details.cached_tokens` into the
        // canonical flat `usage.cached_tokens` (prompt-caching surfacing).
        let mut raw: serde_json::Value = response
            .json()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("azure_openai", e))?;
        crate::openai::lift_openai_cached_tokens(&mut raw);
        serde_json::from_value::<ChatCompletionResponse>(raw).map_err(|e| {
            ProviderError::translation(format!("azure_openai response parse error: {e}"))
        })
    }

    async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
        api_key: String,
    ) -> Result<ChunkStream, ProviderError> {
        if self.endpoint.is_empty() {
            return Err("Azure OpenAI endpoint not configured (set AZURE_OPENAI_ENDPOINT)".into());
        }

        let url = self.chat_url();
        // Azure OpenAI is wire-compatible with OpenAI, so the streaming body and
        // SSE event format are identical — only the URL/auth header differ.
        let mut body = serde_json::to_value(&request)?;
        body["stream"] = json!(true);
        body["stream_options"] = json!({ "include_usage": true });
        // Strip the Anthropic-only cache marker before egress (Azure rejects it).
        crate::openai::strip_cache_control_for_openai(&mut body);

        let resp = self
            .client
            .post(&url)
            .header("api-key", api_key)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("azure_openai", e))?;

        if !resp.status().is_success() {
            return Err(crate::client::error_from_response("azure_openai", resp).await);
        }

        Ok(Box::pin(openai_sse_to_chunks(resp.bytes_stream())))
    }

    async fn embeddings(
        &self,
        request: EmbeddingRequest,
        api_key: String,
    ) -> Result<EmbeddingResponse, ProviderError> {
        if self.endpoint.is_empty() {
            return Err("Azure OpenAI endpoint not configured (set AZURE_OPENAI_ENDPOINT)".into());
        }

        // Azure embeddings live on the deployment's `/embeddings` path with the
        // api-version query and the `api-key` header. The body/response are
        // OpenAI-wire-compatible (the deployment selects the model; the body's
        // `model` is accepted and ignored by Azure).
        let url = self.embeddings_url();

        let response = self
            .client
            .post(&url)
            .header("api-key", api_key)
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("azure_openai", e))?;

        if !response.status().is_success() {
            return Err(crate::client::error_from_response("azure_openai", response).await);
        }

        response
            .json::<EmbeddingResponse>()
            .await
            .map_err(|e| crate::client::sanitize_transport_error("azure_openai", e))
    }
}

impl AzureOpenAiProvider {
    fn chat_url(&self) -> String {
        format!(
            "{}/openai/deployments/{}/chat/completions?api-version={}",
            self.endpoint.trim_end_matches('/'),
            self.deployment,
            self.api_version
        )
    }

    fn embeddings_url(&self) -> String {
        format!(
            "{}/openai/deployments/{}/embeddings?api-version={}",
            self.endpoint.trim_end_matches('/'),
            self.deployment,
            self.api_version
        )
    }
}
