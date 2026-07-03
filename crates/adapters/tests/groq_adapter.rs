//! Wiremock adapter tests for the Groq provider (engineering-design §24).
//! Groq is OpenAI-wire-compatible, so these mirror the OpenAI adapter tests —
//! the load-bearing differences asserted here are (a) the `/openai/v1/...` base
//! path (the known footgun), and (b) embeddings degrading to a typed 422 rather
//! than hitting the network, since Groq has no embeddings endpoint.

mod common;

use common::{collect_ok, msg, request};
use routeplane_adapters::groq::GroqProvider;
use routeplane_adapters::Provider;
use routeplane_types::{EmbeddingInput, EmbeddingRequest};
use serde_json::json;
use wiremock::matchers::{body_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn buffered_request_hits_openai_v1_path_and_passes_response_through() {
    let server = MockServer::start().await;

    // The path footgun: Groq's OpenAI-compatible surface lives under
    // `/openai/v1/chat/completions`. Assert the full prefix is on the wire.
    Mock::given(method("POST"))
        .and(path("/openai/v1/chat/completions"))
        .and(header("authorization", "Bearer gsk-test"))
        .and(body_json(json!({
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 256,
            "user": "u_123"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-groq",
            "object": "chat.completion",
            "created": 1_700_000_000u64,
            "model": "llama-3.3-70b-versatile",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello fast!"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}
        })))
        .expect(1)
        .mount(&server)
        .await;

    let provider = GroqProvider::with_base_url(server.uri());
    assert_eq!(provider.name(), "groq");

    let mut req = request("llama-3.3-70b-versatile", vec![msg("user", "hello")]);
    req.max_tokens = Some(256);
    req.user = Some("u_123".to_string());

    let out = provider
        .chat_completion(req, "gsk-test".to_string())
        .await
        .expect("buffered call against the mock succeeds");

    assert_eq!(out.id, "chatcmpl-groq");
    assert_eq!(out.model, "llama-3.3-70b-versatile");
    assert_eq!(out.choices[0].message.content.as_text(), "Hello fast!");
    assert_eq!(out.choices[0].finish_reason, "stop");
    assert_eq!(out.usage.prompt_tokens, 5);
    assert_eq!(out.usage.completion_tokens, 7);
    assert_eq!(out.usage.total_tokens, 12);
}

#[tokio::test]
async fn streaming_forces_stream_flags_and_translates_sse_to_chunks() {
    let server = MockServer::start().await;

    // OpenAI-identical SSE: role chunk, two content chunks, a finish chunk, a
    // usage-only chunk (empty choices), then [DONE]. Data after [DONE] never
    // surfaces. Reused verbatim from the shared OpenAI SSE translation.
    let sse_body = concat!(
        "data: {\"id\":\"g1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"llama-3.3-70b-versatile\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"g1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"llama-3.3-70b-versatile\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hel\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"g1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"llama-3.3-70b-versatile\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"g1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"llama-3.3-70b-versatile\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        "data: {\"id\":\"g1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"llama-3.3-70b-versatile\",\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2,\"total_tokens\":7}}\n\n",
        "data: [DONE]\n\n",
        "data: {\"after\":\"done; must never surface\"}\n\n",
    );

    Mock::given(method("POST"))
        .and(path("/openai/v1/chat/completions"))
        .and(header("authorization", "Bearer gsk-test"))
        .and(body_json(json!({
            "model": "llama-3.3-70b-versatile",
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

    let provider = GroqProvider::with_base_url(server.uri());
    let stream = provider
        .chat_completion_stream(
            request("llama-3.3-70b-versatile", vec![msg("user", "hello")]),
            "gsk-test".to_string(),
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
    assert!(
        chunks[4].choices.is_empty(),
        "usage-only chunk has no choices"
    );
    let usage = chunks[4].usage.as_ref().expect("final chunk carries usage");
    assert_eq!(usage.total_tokens, 7);
}

#[tokio::test]
async fn streaming_establishment_failure_is_err_and_never_echoes_the_key() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/openai/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(401).set_body_string("invalid credentials"))
        .expect(1)
        .mount(&server)
        .await;

    let provider = GroqProvider::with_base_url(server.uri());
    let err = match provider
        .chat_completion_stream(
            request("llama-3.1-8b-instant", vec![msg("user", "hi")]),
            "gsk-test".to_string(),
        )
        .await
    {
        Ok(_) => panic!("non-2xx establishment must be Err so the proxy can fall back"),
        Err(e) => e,
    };

    let text = err.to_string();
    assert!(
        text.contains("groq authentication failed (401"),
        "got: {text}"
    );
    assert!(!text.contains("invalid credentials"), "got: {text}");
    assert!(
        !text.contains("gsk-test"),
        "API key must never appear in errors"
    );
}

#[tokio::test]
async fn embeddings_are_unsupported_422_without_a_network_call() {
    // Groq has no embeddings endpoint — the call must degrade to a typed 422
    // (`embeddings_not_supported`) before any HTTP request, never a panic. We
    // intentionally point at a dead address to prove no network call happens.
    let provider = GroqProvider::with_base_url("http://127.0.0.1:1");
    let request = EmbeddingRequest {
        model: "anything".into(),
        input: EmbeddingInput::Single("hello".into()),
        user: None,
        encoding_format: None,
        dimensions: None,
    };
    let err = provider
        .embeddings(request, "gsk-test".to_string())
        .await
        .expect_err("groq has no embeddings endpoint");
    assert_eq!(err.status(), Some(422));
    assert!(err.to_string().contains("embeddings_not_supported"));
}
