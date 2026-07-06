//! A single, parameterized adapter for the large family of providers that speak
//! the **OpenAI wire protocol verbatim** — same `/chat/completions` request and
//! response JSON, same `Authorization: Bearer <key>` auth, same SSE streaming
//! shape (`chat.completion.chunk` + `[DONE]`). The only things that vary between
//! them are the *name*, the *base URL*, and whether they offer an embeddings
//! endpoint.
//!
//! This mirrors how LiteLLM / Portkey model their "openai-like" providers: one
//! translation path, configured per provider. It lets Routeplane front Groq,
//! Together, Fireworks, DeepSeek, Mistral, OpenRouter, xAI, Perplexity, Cerebras,
//! and friends without N near-identical adapter modules — each is one row in
//! `build_provider_registry()` (`crates/routeplane/src/proxy.rs`).
//!
//! Providers whose request/response shape genuinely differs (Anthropic's
//! `/v1/messages`, Gemini's `generateContent`) keep their own dedicated adapter —
//! this type is ONLY for byte-compatible OpenAI dialects.

use crate::openai::openai_sse_to_chunks;
use crate::{ChunkStream, Provider, ProviderError};
use async_trait::async_trait;
use reqwest::Client;
use routeplane_types::{
    ChatCompletionRequest, ChatCompletionResponse, EmbeddingRequest, EmbeddingResponse,
};
use serde_json::json;

/// An OpenAI-wire-compatible provider, configured by name + base URL.
///
/// `base_url` must include the provider's version path segment (e.g.
/// `https://api.groq.com/openai/v1`, `https://api.mistral.ai/v1`,
/// `https://api.perplexity.ai`) — the adapter appends `/chat/completions` and
/// `/embeddings` to it. This matches each provider's published OpenAI-compatible
/// base, so a single rule covers bases that do and don't carry a `/v1`.
pub struct OpenAICompatProvider {
    client: Client,
    /// Static so `name()` can return `&'static str`; always a compile-time
    /// literal supplied at registry-construction time.
    name: &'static str,
    base_url: String,
    /// Whether this provider exposes an OpenAI-shaped `/embeddings` endpoint.
    /// When false, `embeddings()` degrades to a typed 422 (the trait default
    /// behavior) rather than calling an endpoint that does not exist.
    supports_embeddings: bool,
    /// Whether to request `stream_options.include_usage` on streaming calls.
    /// True for providers that honor it (so observability gets real token
    /// counts on the final chunk); set false for the rare dialect that rejects
    /// the field with a 400.
    stream_include_usage: bool,
}

impl OpenAICompatProvider {
    /// Construct an OpenAI-compatible provider. Defaults: no embeddings,
    /// `stream_options.include_usage` enabled.
    pub fn new(name: &'static str, base_url: impl Into<String>) -> Self {
        Self {
            client: crate::client::build_provider_client(),
            name,
            base_url: base_url.into(),
            supports_embeddings: false,
            stream_include_usage: true,
        }
    }

    /// Declare that this provider exposes an OpenAI-shaped `/embeddings` endpoint.
    pub fn with_embeddings(mut self) -> Self {
        self.supports_embeddings = true;
        self
    }

    /// Disable `stream_options.include_usage` for a dialect that rejects it.
    pub fn without_stream_usage(mut self) -> Self {
        self.stream_include_usage = false;
        self
    }
}

#[async_trait]
impl Provider for OpenAICompatProvider {
    fn name(&self) -> &'static str {
        self.name
    }

    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
        api_key: String,
    ) -> Result<ChatCompletionResponse, ProviderError> {
        let url = format!("{}/chat/completions", self.base_url);

        let response = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {api_key}"))
            .json(&request)
            .send()
            .await
            .map_err(|e| crate::client::sanitize_transport_error(self.name, e))?;

        if !response.status().is_success() {
            return Err(crate::client::error_from_response(self.name, response).await);
        }

        response
            .json::<ChatCompletionResponse>()
            .await
            .map_err(|e| crate::client::sanitize_transport_error(self.name, e))
    }

    async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
        api_key: String,
    ) -> Result<ChunkStream, ProviderError> {
        let url = format!("{}/chat/completions", self.base_url);

        // Force `stream: true`; ask for usage on the final chunk when the dialect
        // honors it (OpenAI omits usage from streams unless asked).
        let mut body = serde_json::to_value(&request)?;
        body["stream"] = json!(true);
        if self.stream_include_usage {
            body["stream_options"] = json!({ "include_usage": true });
        }

        let resp = crate::client::streaming_client()
            .post(url)
            .header("Authorization", format!("Bearer {api_key}"))
            .json(&body)
            .send()
            .await
            .map_err(|e| crate::client::sanitize_transport_error(self.name, e))?;

        // Establishment failure -> typed Err so the proxy can retry / fall back.
        if !resp.status().is_success() {
            return Err(crate::client::error_from_response(self.name, resp).await);
        }

        Ok(Box::pin(openai_sse_to_chunks(resp.bytes_stream())))
    }

    async fn embeddings(
        &self,
        request: EmbeddingRequest,
        api_key: String,
    ) -> Result<EmbeddingResponse, ProviderError> {
        if !self.supports_embeddings {
            return Err(ProviderError::embeddings_not_supported(self.name));
        }

        let url = format!("{}/embeddings", self.base_url);

        let response = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {api_key}"))
            .json(&request)
            .send()
            .await
            .map_err(|e| crate::client::sanitize_transport_error(self.name, e))?;

        if !response.status().is_success() {
            return Err(crate::client::error_from_response(self.name, response).await);
        }

        response
            .json::<EmbeddingResponse>()
            .await
            .map_err(|e| crate::client::sanitize_transport_error(self.name, e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_is_the_configured_static_str() {
        let p = OpenAICompatProvider::new("groq", "https://api.groq.com/openai/v1");
        assert_eq!(p.name(), "groq");
    }

    #[test]
    fn embeddings_default_off_then_opt_in() {
        let p = OpenAICompatProvider::new("groq", "http://x");
        assert!(!p.supports_embeddings);
        let p = OpenAICompatProvider::new("mistral", "http://x").with_embeddings();
        assert!(p.supports_embeddings);
    }

    #[test]
    fn stream_usage_default_on_then_opt_out() {
        let p = OpenAICompatProvider::new("groq", "http://x");
        assert!(p.stream_include_usage);
        let p = OpenAICompatProvider::new("groq", "http://x").without_stream_usage();
        assert!(!p.stream_include_usage);
    }
}
