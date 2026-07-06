//! Wiremock adapter tests for the xAI (Grok) provider (engineering-design §24).
//! xAI is OpenAI-wire-compatible, so these mirror the Groq/DeepSeek/OpenAI
//! adapter tests — the load-bearing assertions here are (a) the
//! `/chat/completions` path is hit under the `https://api.x.ai/v1` host root
//! (supplied here by the wiremock base URL), (b) SSE streaming translates to
//! canonical chunks via the shared OpenAI translation with the forced
//! `stream`/`stream_options` flags, (c) Bearer auth carries the key while the
//! key never leaks into errors, and (d) embeddings degrade to a typed 422
//! without a network call, since xAI has no first-party embeddings endpoint.

mod common;

use common::{collect_ok, msg, request};
use routeplane_adapters::xai::XaiProvider;
use routeplane_adapters::Provider;
use routeplane_types::{EmbeddingInput, EmbeddingRequest};
use serde_json::json;
use wiremock::matchers::{body_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn buffered_request_hits_chat_completions_path_and_passes_response_through() {
    let server = MockServer::start().await;

    // xAI serves its OpenAI-compatible chat surface at `{base}/chat/completions`
    // (the `/v1` is part of the host root) — assert the full path is on the wire
    // with Bearer auth.
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer xai-test"))
        .and(body_json(json!({
            "model": "grok-4.3",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 256,
            "user": "u_123"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-xai",
            "object": "chat.completion",
            "created": 1_700_000_000u64,
            "model": "grok-4.3",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello from Grok!"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}
        })))
        .expect(1)
        .mount(&server)
        .await;

    let provider = XaiProvider::with_base_url(server.uri());
    assert_eq!(provider.name(), "xai");

    let mut req = request("grok-4.3", vec![msg("user", "hello")]);
    req.max_tokens = Some(256);
    req.user = Some("u_123".to_string());

    let out = provider
        .chat_completion(req, "xai-test".to_string())
        .await
        .expect("buffered call against the mock succeeds");

    assert_eq!(out.id, "chatcmpl-xai");
    assert_eq!(out.model, "grok-4.3");
    assert_eq!(out.choices[0].message.content.as_text(), "Hello from Grok!");
    assert_eq!(out.choices[0].finish_reason, "stop");
    assert_eq!(out.usage.prompt_tokens, 5);
    assert_eq!(out.usage.completion_tokens, 7);
    assert_eq!(out.usage.total_tokens, 12);
}

#[tokio::test]
async fn reasoning_content_field_passes_through_to_the_client() {
    // The Grok reasoning tier can return an extra `reasoning_content` field; the
    // canonical types carry it as a typed passthrough field, so it must survive
    // deserialization AND reach the client — not be dropped.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-xai-r1",
            "object": "chat.completion",
            "created": 1_700_000_000u64,
            "model": "grok-4-0709",
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

    let provider = XaiProvider::with_base_url(server.uri());
    let out = provider
        .chat_completion(
            request("grok-4-0709", vec![msg("user", "what is 6*7?")]),
            "xai-test".to_string(),
        )
        .await
        .expect("reasoning_content field must not break deserialization");

    assert_eq!(out.choices[0].message.content.as_text(), "the answer is 42");
    // PASSTHROUGH: the reasoning text is surfaced on the canonical message, not
    // silently dropped (response/chunk passthrough).
    assert_eq!(
        out.choices[0].message.reasoning_content.as_deref(),
        Some("let me think step by step ...")
    );
    assert_eq!(out.usage.total_tokens, 14);
}

#[tokio::test]
async fn streaming_forces_stream_flags_and_translates_sse_to_chunks() {
    let server = MockServer::start().await;

    // OpenAI-identical SSE: role chunk, two content chunks, a finish chunk, a
    // usage-only chunk (empty choices), then [DONE]. Data after [DONE] never
    // surfaces. Reused verbatim from the shared OpenAI SSE translation.
    let sse_body = concat!(
        "data: {\"id\":\"x1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"grok-3-fast\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"x1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"grok-3-fast\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hel\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"x1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"grok-3-fast\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"x1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"grok-3-fast\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        "data: {\"id\":\"x1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"grok-3-fast\",\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2,\"total_tokens\":7}}\n\n",
        "data: [DONE]\n\n",
        "data: {\"after\":\"done; must never surface\"}\n\n",
    );

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer xai-test"))
        .and(body_json(json!({
            "model": "grok-3-fast",
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

    let provider = XaiProvider::with_base_url(server.uri());
    let stream = provider
        .chat_completion_stream(
            request("grok-3-fast", vec![msg("user", "hello")]),
            "xai-test".to_string(),
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

    let provider = XaiProvider::with_base_url(server.uri());
    let err = match provider
        .chat_completion_stream(
            request("grok-4.3", vec![msg("user", "hi")]),
            "xai-test".to_string(),
        )
        .await
    {
        Ok(_) => panic!("non-2xx establishment must be Err so the proxy can fall back"),
        Err(e) => e,
    };

    let text = err.to_string();
    assert!(
        text.contains("xai authentication failed (401"),
        "got: {text}"
    );
    assert!(!text.contains("invalid credentials"), "got: {text}");
    assert!(
        !text.contains("xai-test"),
        "API key must never appear in errors"
    );
}

#[tokio::test]
async fn embeddings_are_unsupported_422_without_a_network_call() {
    // xAI has no embeddings endpoint — the call must degrade to a typed 422
    // (`embeddings_not_supported`) before any HTTP request, never a panic. We
    // intentionally point at a dead address to prove no network call happens.
    let provider = XaiProvider::with_base_url("http://127.0.0.1:1");
    let request = EmbeddingRequest {
        model: "anything".into(),
        input: EmbeddingInput::Single("hello".into()),
        user: None,
        encoding_format: None,
        dimensions: None,
    };
    let err = provider
        .embeddings(request, "xai-test".to_string())
        .await
        .expect_err("xai has no embeddings endpoint");
    assert_eq!(err.status(), Some(422));
    assert!(err.to_string().contains("embeddings_not_supported"));
}
