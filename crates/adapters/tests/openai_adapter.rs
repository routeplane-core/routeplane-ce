//! Wiremock adapter tests for the OpenAI provider (engineering-design §24).
//! Covers: (a) canonical -> native request translation asserted on the wire,
//! (b) native -> canonical buffered response translation incl. usage, and
//! (c) SSE streaming translation incl. `[DONE]` end-of-stream handling.

mod common;

use common::{collect_ok, msg, request};
use routeplane_adapters::openai::OpenAIProvider;
use routeplane_adapters::Provider;
use serde_json::json;
use wiremock::matchers::{body_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn buffered_request_translates_exactly_and_response_passes_through() {
    let server = MockServer::start().await;

    // Exact-body match: OpenAI is the canonical dialect, so the gateway must
    // forward precisely what the client sent — threaded optionals present,
    // unset optionals absent, nothing injected.
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("authorization", "Bearer test-key"))
        .and(body_json(json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 256,
            "stop": ["STOP"],
            "presence_penalty": 0.5,
            "user": "u_123"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-abc",
            "object": "chat.completion",
            "created": 1_700_000_000u64,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}
        })))
        .expect(1)
        .mount(&server)
        .await;

    let provider = OpenAIProvider::with_base_url(server.uri());
    assert_eq!(provider.name(), "openai");

    let mut req = request("gpt-4o", vec![msg("user", "hello")]);
    req.max_tokens = Some(256);
    req.stop = Some(vec!["STOP".to_string()]);
    req.presence_penalty = Some(0.5);
    req.user = Some("u_123".to_string());

    let out = provider
        .chat_completion(req, "test-key".to_string())
        .await
        .expect("buffered call against the mock succeeds");

    assert_eq!(out.id, "chatcmpl-abc");
    assert_eq!(out.object, "chat.completion");
    assert_eq!(out.created, 1_700_000_000);
    assert_eq!(out.model, "gpt-4o");
    assert_eq!(out.choices[0].message.role, "assistant");
    assert_eq!(out.choices[0].message.content.as_text(), "Hello!");
    assert_eq!(out.choices[0].finish_reason, "stop");
    assert_eq!(out.usage.prompt_tokens, 5);
    assert_eq!(out.usage.completion_tokens, 7);
    assert_eq!(out.usage.total_tokens, 12);
}

#[tokio::test]
async fn streaming_forces_stream_flags_and_translates_sse_to_chunks() {
    let server = MockServer::start().await;

    // Realistic OpenAI stream: role chunk, two content chunks, a finish chunk,
    // then a usage-only chunk with empty `choices` (sent because the adapter
    // asks for stream_options.include_usage), then [DONE]. Data after [DONE]
    // must never surface.
    let sse_body = concat!(
        "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hel\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2,\"total_tokens\":7}}\n\n",
        "data: [DONE]\n\n",
        "data: {\"after\":\"done; must never surface\"}\n\n",
    );

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("authorization", "Bearer test-key"))
        .and(body_json(json!({
            "model": "gpt-4o",
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

    let provider = OpenAIProvider::with_base_url(server.uri());
    let stream = provider
        .chat_completion_stream(
            request("gpt-4o", vec![msg("user", "hello")]),
            "test-key".to_string(),
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
    assert_eq!(usage.prompt_tokens, 5);
    assert_eq!(usage.completion_tokens, 2);
    assert_eq!(usage.total_tokens, 7);
}

#[tokio::test]
async fn streaming_establishment_failure_is_err_and_never_echoes_the_key() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(401).set_body_string("invalid credentials"))
        .expect(1)
        .mount(&server)
        .await;

    let provider = OpenAIProvider::with_base_url(server.uri());
    let err = match provider
        .chat_completion_stream(
            request("gpt-4o", vec![msg("user", "hi")]),
            "test-key".to_string(),
        )
        .await
    {
        Ok(_) => panic!("non-2xx establishment must be Err so the proxy can fall back"),
        Err(e) => e,
    };

    let text = err.to_string();
    assert!(
        text.contains("openai authentication failed (401"),
        "got: {text}"
    );
    // Hardening (G2.3 security review): auth bodies are SUPPRESSED — upstream
    // 401 bodies can echo provider-key fingerprints, so the body must NOT
    // appear in the client-visible error.
    assert!(!text.contains("invalid credentials"), "got: {text}");
    assert!(
        !text.contains("test-key"),
        "API key must never appear in errors"
    );
}
