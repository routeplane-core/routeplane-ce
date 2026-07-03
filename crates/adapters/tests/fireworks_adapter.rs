//! Wiremock adapter tests for the Fireworks AI provider (engineering-design §24).
//! Fireworks is OpenAI-wire-compatible, so these mirror the Together/Groq/OpenAI
//! adapter tests. The load-bearing differences asserted here are: (a) chat at
//! `{base}/chat/completions` with namespaced (`accounts/fireworks/models/<name>`)
//! model ids + Bearer auth, (b) SSE streaming reusing the shared OpenAI
//! translation, and (c) — like Together — a FIRST-PARTY embeddings endpoint at
//! `{base}/embeddings` (OpenAI dialect) that maps vectors through. The base URL
//! root carries `/inference/v1`, so the asserted paths are the trailing segments.
//! The resolved key never leaks into errors.

mod common;

use common::{collect_ok, msg, request};
use routeplane_adapters::fireworks::FireworksProvider;
use routeplane_adapters::Provider;
use routeplane_types::{EmbeddingInput, EmbeddingRequest};
use serde_json::json;
use wiremock::matchers::{body_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn buffered_request_hits_chat_completions_path_and_passes_response_through() {
    let server = MockServer::start().await;

    // Fireworks serves its OpenAI-compatible chat surface at
    // {base}/chat/completions (the base already carries /inference/v1). Model ids
    // are namespaced — assert verbatim.
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer fw-test"))
        .and(body_json(json!({
            "model": "accounts/fireworks/models/llama-v3p1-70b-instruct",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 256,
            "user": "u_123"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-fw",
            "object": "chat.completion",
            "created": 1_700_000_000u64,
            "model": "accounts/fireworks/models/llama-v3p1-70b-instruct",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello from Fireworks!"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}
        })))
        .expect(1)
        .mount(&server)
        .await;

    let provider = FireworksProvider::with_base_url(server.uri());
    assert_eq!(provider.name(), "fireworks");

    let mut req = request(
        "accounts/fireworks/models/llama-v3p1-70b-instruct",
        vec![msg("user", "hello")],
    );
    req.max_tokens = Some(256);
    req.user = Some("u_123".to_string());

    let out = provider
        .chat_completion(req, "fw-test".to_string())
        .await
        .expect("buffered call against the mock succeeds");

    assert_eq!(out.id, "chatcmpl-fw");
    assert_eq!(
        out.model,
        "accounts/fireworks/models/llama-v3p1-70b-instruct"
    );
    assert_eq!(
        out.choices[0].message.content.as_text(),
        "Hello from Fireworks!"
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
        "data: {\"id\":\"f1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"accounts/fireworks/models/qwen2p5-72b-instruct\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"f1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"accounts/fireworks/models/qwen2p5-72b-instruct\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hel\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"f1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"accounts/fireworks/models/qwen2p5-72b-instruct\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"f1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"accounts/fireworks/models/qwen2p5-72b-instruct\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        "data: {\"id\":\"f1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"accounts/fireworks/models/qwen2p5-72b-instruct\",\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2,\"total_tokens\":7}}\n\n",
        "data: [DONE]\n\n",
        "data: {\"after\":\"done; must never surface\"}\n\n",
    );

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer fw-test"))
        .and(body_json(json!({
            "model": "accounts/fireworks/models/qwen2p5-72b-instruct",
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

    let provider = FireworksProvider::with_base_url(server.uri());
    let stream = provider
        .chat_completion_stream(
            request(
                "accounts/fireworks/models/qwen2p5-72b-instruct",
                vec![msg("user", "hello")],
            ),
            "fw-test".to_string(),
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

    // Fireworks' first-party /embeddings is the OpenAI dialect (1:1 passthrough).
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .and(header("authorization", "Bearer fw-test"))
        .and(body_json(json!({
            "model": "nomic-ai/nomic-embed-text-v1.5",
            "input": ["alpha", "beta"]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "object": "list",
            "data": [
                {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
                {"object": "embedding", "index": 1, "embedding": [0.3, 0.4]}
            ],
            "model": "nomic-ai/nomic-embed-text-v1.5",
            "usage": {"prompt_tokens": 4, "total_tokens": 4}
        })))
        .expect(1)
        .mount(&server)
        .await;

    let provider = FireworksProvider::with_base_url(server.uri());
    let out = provider
        .embeddings(
            EmbeddingRequest {
                model: "nomic-ai/nomic-embed-text-v1.5".into(),
                input: EmbeddingInput::Batch(vec!["alpha".into(), "beta".into()]),
                user: None,
                encoding_format: None,
                dimensions: None,
            },
            "fw-test".to_string(),
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

    let provider = FireworksProvider::with_base_url(server.uri());
    let err = provider
        .chat_completion(
            request(
                "accounts/fireworks/models/deepseek-v3",
                vec![msg("user", "hi")],
            ),
            "fw-secret".to_string(),
        )
        .await
        .expect_err("401 should be an Err");

    let text = err.to_string();
    assert!(
        text.contains("fireworks authentication failed (401"),
        "got: {text}"
    );
    assert!(!text.contains("invalid credentials"), "got: {text}");
    assert!(
        !text.contains("fw-secret"),
        "API key must never appear in errors"
    );
}
