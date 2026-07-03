//! Wiremock adapter tests for /v1/embeddings translation (G2.7, PRD-011 §5).
//! Covers OpenAI (canonical passthrough), Azure OpenAI (deployment URL +
//! api-key header), Gemini (batchEmbedContents translation incl. multi-input
//! index order), and Anthropic's default → typed `embeddings_not_supported`
//! (422) with NO real network call (engineering-design §24).

use routeplane_adapters::anthropic::AnthropicProvider;
use routeplane_adapters::azure_openai::AzureOpenAiProvider;
use routeplane_adapters::gemini::GeminiProvider;
use routeplane_adapters::openai::OpenAIProvider;
use routeplane_adapters::Provider;
use routeplane_types::{EmbeddingInput, EmbeddingRequest};
use serde_json::json;
use wiremock::matchers::{body_json, body_partial_json, header, method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn req(model: &str, input: EmbeddingInput) -> EmbeddingRequest {
    EmbeddingRequest {
        model: model.to_string(),
        input,
        encoding_format: None,
        dimensions: None,
        user: None,
    }
}

#[tokio::test]
async fn openai_embeddings_passes_through_request_and_response() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .and(header("authorization", "Bearer sk-test"))
        .and(body_json(json!({
            "model": "text-embedding-3-small",
            "input": ["alpha", "beta"]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "object": "list",
            "data": [
                {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
                {"object": "embedding", "index": 1, "embedding": [0.3, 0.4]}
            ],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 4, "total_tokens": 4}
        })))
        .expect(1)
        .mount(&server)
        .await;

    let provider = OpenAIProvider::with_base_url(server.uri());
    let out = provider
        .embeddings(
            req(
                "text-embedding-3-small",
                EmbeddingInput::Batch(vec!["alpha".into(), "beta".into()]),
            ),
            "sk-test".to_string(),
        )
        .await
        .expect("buffered embeddings call succeeds");

    assert_eq!(out.object, "list");
    assert_eq!(out.data.len(), 2);
    assert_eq!(out.data[0].index, 0);
    assert_eq!(out.data[1].index, 1);
    assert_eq!(out.data[1].embedding.len(), 2);
    assert!((out.data[1].embedding.as_floats().unwrap()[0] as f64 - 0.3).abs() < 1e-6);
    assert_eq!(out.usage.total_tokens, 4);
}

#[tokio::test]
async fn azure_embeddings_hits_deployment_url_with_api_key_header() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/openai/deployments/text-embed-india/embeddings"))
        .and(query_param("api-version", "2024-10-21"))
        .and(header("api-key", "azure-test-key"))
        .and(body_partial_json(json!({ "input": "hello" })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.5]}],
            "model": "text-embedding-3-large",
            "usage": {"prompt_tokens": 2, "total_tokens": 2}
        })))
        .expect(1)
        .mount(&server)
        .await;

    let provider = AzureOpenAiProvider::new(
        server.uri(),
        "text-embed-india".to_string(),
        "2024-10-21".to_string(),
        "IN".to_string(),
    );
    let out = provider
        .embeddings(
            req(
                "text-embedding-3-large",
                EmbeddingInput::Single("hello".into()),
            ),
            "azure-test-key".to_string(),
        )
        .await
        .expect("azure embeddings call succeeds");

    assert_eq!(out.data.len(), 1);
    assert_eq!(out.model, "text-embedding-3-large");
    assert_eq!(out.usage.total_tokens, 2);
}

#[tokio::test]
async fn gemini_embeddings_translate_batch_and_preserve_index_order() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1beta/models/text-embedding-004:batchEmbedContents"))
        .and(query_param("key", "test-key"))
        .and(body_json(json!({
            "requests": [
                {"model": "models/text-embedding-004", "content": {"parts": [{"text": "first"}]}},
                {"model": "models/text-embedding-004", "content": {"parts": [{"text": "second"}]}}
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "embeddings": [
                {"values": [0.11, 0.12]},
                {"values": [0.21, 0.22]}
            ]
        })))
        .expect(1)
        .mount(&server)
        .await;

    let provider = GeminiProvider::with_base_url(server.uri());
    let out = provider
        .embeddings(
            req(
                "text-embedding-004",
                EmbeddingInput::Batch(vec!["first".into(), "second".into()]),
            ),
            "test-key".to_string(),
        )
        .await
        .expect("gemini embeddings call succeeds");

    assert_eq!(out.object, "list");
    assert_eq!(out.data.len(), 2);
    // The load-bearing assertion: request order maps to response index order.
    assert_eq!(out.data[0].index, 0);
    assert!((out.data[0].embedding.as_floats().unwrap()[0] as f64 - 0.11).abs() < 1e-6);
    assert_eq!(out.data[1].index, 1);
    assert!((out.data[1].embedding.as_floats().unwrap()[0] as f64 - 0.21).abs() < 1e-6);
    // Gemini's embed API returns no usage → zeroed (documented fidelity gap).
    assert_eq!(out.usage.total_tokens, 0);
}

#[tokio::test]
async fn gemini_embeddings_error_never_echoes_the_query_key() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(400).set_body_string("bad request"))
        .mount(&server)
        .await;

    let provider = GeminiProvider::with_base_url(server.uri());
    let err = provider
        .embeddings(
            req("text-embedding-004", EmbeddingInput::Single("hi".into())),
            "secret-key".to_string(),
        )
        .await
        .expect_err("400 surfaces as Err");
    let text = err.to_string();
    assert!(text.contains("gemini API error (400"), "got: {text}");
    assert!(
        !text.contains("secret-key"),
        "API key must never appear in errors"
    );
}

#[tokio::test]
async fn anthropic_embeddings_is_typed_unsupported_without_network() {
    // Anthropic uses the trait DEFAULT — a typed 422 embeddings_not_supported,
    // NOT a panic and NOT a network call (FR-2/FR-3).
    let provider = AnthropicProvider::new();
    let err = provider
        .embeddings(
            req("claude-3-5-sonnet", EmbeddingInput::Single("hello".into())),
            "unused".to_string(),
        )
        .await
        .expect_err("anthropic has no first-party embeddings");
    assert_eq!(err.status(), Some(422));
    let text = err.to_string();
    assert!(text.contains("embeddings_not_supported"), "got: {text}");
    assert!(text.contains("anthropic"), "got: {text}");
}
