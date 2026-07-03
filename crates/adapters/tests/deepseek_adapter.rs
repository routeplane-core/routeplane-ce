//! Wiremock adapter tests for the DeepSeek provider (engineering-design §24).
//! DeepSeek is OpenAI-wire-compatible, so these mirror the Groq/OpenAI adapter
//! tests — the load-bearing differences asserted here are (a) the base
//! `/chat/completions` path (no `/v1` prefix — that is only an optional alias),
//! (b) the extra `reasoning_content` field from reasoning models being ignored
//! rather than breaking deserialization, and (c) embeddings degrading to a typed
//! 422 rather than hitting the network, since DeepSeek has no embeddings endpoint.

mod common;

use common::{collect_ok, msg, request};
use routeplane_adapters::deepseek::DeepSeekProvider;
use routeplane_adapters::Provider;
use routeplane_types::{EmbeddingInput, EmbeddingRequest};
use serde_json::json;
use wiremock::matchers::{body_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn buffered_request_hits_chat_completions_path_and_passes_response_through() {
    let server = MockServer::start().await;

    // DeepSeek serves its OpenAI-compatible chat surface at the host root
    // `{base}/chat/completions` — assert the full base path is on the wire.
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer sk-ds-test"))
        .and(body_json(json!({
            "model": "deepseek-v4-pro",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 256,
            "user": "u_123"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-ds",
            "object": "chat.completion",
            "created": 1_700_000_000u64,
            "model": "deepseek-v4-pro",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello from DeepSeek!"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}
        })))
        .expect(1)
        .mount(&server)
        .await;

    let provider = DeepSeekProvider::with_base_url(server.uri());
    assert_eq!(provider.name(), "deepseek");

    let mut req = request("deepseek-v4-pro", vec![msg("user", "hello")]);
    req.max_tokens = Some(256);
    req.user = Some("u_123".to_string());

    let out = provider
        .chat_completion(req, "sk-ds-test".to_string())
        .await
        .expect("buffered call against the mock succeeds");

    assert_eq!(out.id, "chatcmpl-ds");
    assert_eq!(out.model, "deepseek-v4-pro");
    assert_eq!(
        out.choices[0].message.content.as_text(),
        "Hello from DeepSeek!"
    );
    assert_eq!(out.choices[0].finish_reason, "stop");
    assert_eq!(out.usage.prompt_tokens, 5);
    assert_eq!(out.usage.completion_tokens, 7);
    assert_eq!(out.usage.total_tokens, 12);
}

#[tokio::test]
async fn reasoning_content_field_is_ignored_not_a_parse_error() {
    // deepseek-reasoner returns an extra `reasoning_content` field; the canonical
    // types tolerate unknown fields, so deserialization must succeed and ignore
    // it — never a panic / Translation error.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-ds-r1",
            "object": "chat.completion",
            "created": 1_700_000_000u64,
            "model": "deepseek-reasoner",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "the answer is 42",
                    "reasoning_content": "let me think step by step ..."
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 9, "total_tokens": 14}
        })))
        .expect(1)
        .mount(&server)
        .await;

    let provider = DeepSeekProvider::with_base_url(server.uri());
    let out = provider
        .chat_completion(
            request("deepseek-reasoner", vec![msg("user", "what is 6*7?")]),
            "sk-ds-test".to_string(),
        )
        .await
        .expect("unknown reasoning_content field must not break deserialization");

    assert_eq!(out.choices[0].message.content.as_text(), "the answer is 42");
    assert_eq!(out.usage.total_tokens, 14);
}

#[tokio::test]
async fn streaming_forces_stream_flags_and_translates_sse_to_chunks() {
    let server = MockServer::start().await;

    // OpenAI-identical SSE: role chunk, two content chunks, a finish chunk, a
    // usage-only chunk (empty choices), then [DONE]. Data after [DONE] never
    // surfaces. Reused verbatim from the shared OpenAI SSE translation.
    let sse_body = concat!(
        "data: {\"id\":\"d1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"deepseek-v4-flash\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"d1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"deepseek-v4-flash\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hel\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"d1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"deepseek-v4-flash\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"d1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"deepseek-v4-flash\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        "data: {\"id\":\"d1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"deepseek-v4-flash\",\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2,\"total_tokens\":7}}\n\n",
        "data: [DONE]\n\n",
        "data: {\"after\":\"done; must never surface\"}\n\n",
    );

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer sk-ds-test"))
        .and(body_json(json!({
            "model": "deepseek-v4-flash",
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

    let provider = DeepSeekProvider::with_base_url(server.uri());
    let stream = provider
        .chat_completion_stream(
            request("deepseek-v4-flash", vec![msg("user", "hello")]),
            "sk-ds-test".to_string(),
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
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(401).set_body_string("invalid credentials"))
        .expect(1)
        .mount(&server)
        .await;

    let provider = DeepSeekProvider::with_base_url(server.uri());
    let err = match provider
        .chat_completion_stream(
            request("deepseek-v4-flash", vec![msg("user", "hi")]),
            "sk-ds-test".to_string(),
        )
        .await
    {
        Ok(_) => panic!("non-2xx establishment must be Err so the proxy can fall back"),
        Err(e) => e,
    };

    let text = err.to_string();
    assert!(
        text.contains("deepseek authentication failed (401"),
        "got: {text}"
    );
    assert!(!text.contains("invalid credentials"), "got: {text}");
    assert!(
        !text.contains("sk-ds-test"),
        "API key must never appear in errors"
    );
}

#[tokio::test]
async fn embeddings_are_unsupported_422_without_a_network_call() {
    // DeepSeek has no embeddings endpoint — the call must degrade to a typed 422
    // (`embeddings_not_supported`) before any HTTP request, never a panic. We
    // intentionally point at a dead address to prove no network call happens.
    let provider = DeepSeekProvider::with_base_url("http://127.0.0.1:1");
    let request = EmbeddingRequest {
        model: "anything".into(),
        input: EmbeddingInput::Single("hello".into()),
        user: None,
        encoding_format: None,
        dimensions: None,
    };
    let err = provider
        .embeddings(request, "sk-ds-test".to_string())
        .await
        .expect_err("deepseek has no embeddings endpoint");
    assert_eq!(err.status(), Some(422));
    assert!(err.to_string().contains("embeddings_not_supported"));
}
