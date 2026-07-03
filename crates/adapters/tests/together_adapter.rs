//! Wiremock adapter tests for the Together AI provider (engineering-design §24).
//! Together is OpenAI-wire-compatible, so these mirror the Groq/DeepSeek/OpenAI
//! adapter tests. The load-bearing differences asserted here are: (a) chat at
//! `{base}/chat/completions` with namespaced model ids + Bearer auth, (b) SSE
//! streaming reusing the shared OpenAI translation, and (c) — unlike Groq/DeepSeek
//! — a FIRST-PARTY embeddings endpoint at `{base}/embeddings` (OpenAI dialect) that
//! maps vectors through. The resolved key never leaks into errors.

mod common;

use common::{collect_ok, msg, request};
use routeplane_adapters::together::TogetherProvider;
use routeplane_adapters::Provider;
use routeplane_types::{EmbeddingInput, EmbeddingRequest};
use serde_json::json;
use wiremock::matchers::{body_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn buffered_request_hits_chat_completions_path_and_passes_response_through() {
    let server = MockServer::start().await;

    // Together serves its OpenAI-compatible chat surface at {base}/chat/completions
    // (the base already carries /v1). Model ids are namespaced — assert verbatim.
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer tg-test"))
        .and(body_json(json!({
            "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 256,
            "user": "u_123"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-tg",
            "object": "chat.completion",
            "created": 1_700_000_000u64,
            "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello from Together!"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}
        })))
        .expect(1)
        .mount(&server)
        .await;

    let provider = TogetherProvider::with_base_url(server.uri());
    assert_eq!(provider.name(), "together");

    let mut req = request(
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        vec![msg("user", "hello")],
    );
    req.max_tokens = Some(256);
    req.user = Some("u_123".to_string());

    let out = provider
        .chat_completion(req, "tg-test".to_string())
        .await
        .expect("buffered call against the mock succeeds");

    assert_eq!(out.id, "chatcmpl-tg");
    assert_eq!(out.model, "meta-llama/Llama-3.3-70B-Instruct-Turbo");
    assert_eq!(
        out.choices[0].message.content.as_text(),
        "Hello from Together!"
    );
    assert_eq!(out.choices[0].finish_reason, "stop");
    assert_eq!(out.usage.total_tokens, 12);
}

#[tokio::test]
async fn streaming_forces_stream_flags_and_translates_sse_to_chunks() {
    let server = MockServer::start().await;

    // OpenAI-identical SSE, reused verbatim via the shared translation: role
    // chunk, two content chunks, a finish chunk, a usage-only chunk, then [DONE].
    let sse_body = concat!(
        "data: {\"id\":\"t1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"Qwen/Qwen2.5-72B-Instruct-Turbo\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"t1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"Qwen/Qwen2.5-72B-Instruct-Turbo\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hel\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"t1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"Qwen/Qwen2.5-72B-Instruct-Turbo\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"t1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"Qwen/Qwen2.5-72B-Instruct-Turbo\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        "data: {\"id\":\"t1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"Qwen/Qwen2.5-72B-Instruct-Turbo\",\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2,\"total_tokens\":7}}\n\n",
        "data: [DONE]\n\n",
        "data: {\"after\":\"done; must never surface\"}\n\n",
    );

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer tg-test"))
        .and(body_json(json!({
            "model": "Qwen/Qwen2.5-72B-Instruct-Turbo",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true,
            "stream_options": {"include_usage": true}
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(sse_body.as_bytes().to_vec(), "text/event-stream"),
        )
        .expect(1)
        .mount(&server)
        .await;

    let provider = TogetherProvider::with_base_url(server.uri());
    let stream = provider
        .chat_completion_stream(
            request(
                "Qwen/Qwen2.5-72B-Instruct-Turbo",
                vec![msg("user", "hello")],
            ),
            "tg-test".to_string(),
        )
        .await
        .expect("stream establishment succeeds");
    let chunks = collect_ok(stream).await;

    assert_eq!(chunks.len(), 5, "[DONE] must end the stream");
    assert_eq!(
        chunks[0].choices[0].delta.role.as_deref(),
        Some("assistant")
    );
    assert_eq!(chunks[1].choices[0].delta.content.as_deref(), Some("Hel"));
    assert_eq!(chunks[2].choices[0].delta.content.as_deref(), Some("lo"));
    assert_eq!(chunks[3].choices[0].finish_reason.as_deref(), Some("stop"));
    let usage = chunks[4].usage.as_ref().expect("final chunk carries usage");
    assert_eq!(usage.total_tokens, 7);
}

#[tokio::test]
async fn embeddings_hit_embeddings_path_and_map_vectors() {
    let server = MockServer::start().await;

    // Together's first-party /embeddings is the OpenAI dialect (1:1 passthrough).
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .and(header("authorization", "Bearer tg-test"))
        .and(body_json(json!({
            "model": "BAAI/bge-large-en-v1.5",
            "input": ["alpha", "beta"]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "object": "list",
            "data": [
                {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
                {"object": "embedding", "index": 1, "embedding": [0.3, 0.4]}
            ],
            "model": "BAAI/bge-large-en-v1.5",
            "usage": {"prompt_tokens": 4, "total_tokens": 4}
        })))
        .expect(1)
        .mount(&server)
        .await;

    let provider = TogetherProvider::with_base_url(server.uri());
    let out = provider
        .embeddings(
            EmbeddingRequest {
                model: "BAAI/bge-large-en-v1.5".into(),
                input: EmbeddingInput::Batch(vec!["alpha".into(), "beta".into()]),
                user: None,
                encoding_format: None,
                dimensions: None,
            },
            "tg-test".to_string(),
        )
        .await
        .expect("buffered embeddings call succeeds");

    assert_eq!(out.object, "list");
    assert_eq!(out.data.len(), 2);
    assert_eq!(out.data[0].index, 0);
    assert_eq!(out.data[1].index, 1);
    assert!((out.data[1].embedding.as_floats().unwrap()[0] as f64 - 0.3).abs() < 1e-6);
    assert_eq!(out.usage.total_tokens, 4);
}

#[tokio::test]
async fn upstream_error_is_typed_and_never_echoes_the_key() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(401).set_body_string("invalid credentials"))
        .expect(1)
        .mount(&server)
        .await;

    let provider = TogetherProvider::with_base_url(server.uri());
    let err = provider
        .chat_completion(
            request("deepseek-ai/DeepSeek-V3", vec![msg("user", "hi")]),
            "tg-secret".to_string(),
        )
        .await
        .expect_err("401 should be an Err");

    let text = err.to_string();
    assert!(
        text.contains("together authentication failed (401"),
        "got: {text}"
    );
    assert!(!text.contains("invalid credentials"), "got: {text}");
    assert!(
        !text.contains("tg-secret"),
        "API key must never appear in errors"
    );
}
