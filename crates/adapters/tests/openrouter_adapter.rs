//! Wiremock adapter tests for the OpenRouter provider (engineering-design §24).
//! OpenRouter is an OpenAI-wire-compatible meta-aggregator, so these mirror the
//! Groq/DeepSeek adapter tests — the load-bearing differences asserted here are
//! (a) the `/api/v1/chat/completions` path (the `/api/v1` is part of the host
//! root), (b) the two OpenRouter attribution headers (`HTTP-Referer` + `X-Title`)
//! being present on chat AND stream requests, (c) `provider/model`-form model ids
//! passing through verbatim, (d) the API key never leaking into errors, and
//! (e) embeddings degrading to a typed 422 (OpenRouter is chat-focused) rather
//! than hitting the network.

mod common;

use common::{collect_ok, msg, request};
use routeplane_adapters::openrouter::OpenRouterProvider;
use routeplane_adapters::Provider;
use routeplane_types::{EmbeddingInput, EmbeddingRequest};
use serde_json::json;
use wiremock::matchers::{body_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn buffered_request_hits_api_v1_path_with_bearer_and_attribution_headers() {
    let server = MockServer::start().await;

    // OpenRouter serves its OpenAI-compatible chat surface at
    // `{base}/api/v1/chat/completions` — assert the full path, the Bearer auth,
    // AND both recommended attribution headers are on the wire. The model id is
    // the `provider/model` form and must pass through verbatim.
    Mock::given(method("POST"))
        .and(path("/api/v1/chat/completions"))
        .and(header("authorization", "Bearer sk-or-test"))
        .and(header("http-referer", "https://routeplane.ai"))
        .and(header("x-title", "Routeplane"))
        .and(body_json(json!({
            "model": "openai/gpt-4o",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 256,
            "user": "u_123"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-or",
            "object": "chat.completion",
            "created": 1_700_000_000u64,
            "model": "openai/gpt-4o",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello from OpenRouter!"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}
        })))
        .expect(1)
        .mount(&server)
        .await;

    let provider = OpenRouterProvider::with_base_url(server.uri());
    assert_eq!(provider.name(), "openrouter");

    let mut req = request("openai/gpt-4o", vec![msg("user", "hello")]);
    req.max_tokens = Some(256);
    req.user = Some("u_123".to_string());

    let out = provider
        .chat_completion(req, "sk-or-test".to_string())
        .await
        .expect("buffered call against the mock succeeds");

    assert_eq!(out.id, "chatcmpl-or");
    assert_eq!(out.model, "openai/gpt-4o");
    assert_eq!(
        out.choices[0].message.content.as_text(),
        "Hello from OpenRouter!"
    );
    assert_eq!(out.choices[0].finish_reason, "stop");
    assert_eq!(out.usage.prompt_tokens, 5);
    assert_eq!(out.usage.completion_tokens, 7);
    assert_eq!(out.usage.total_tokens, 12);
}

#[tokio::test]
async fn streaming_forces_stream_flags_with_attribution_and_translates_sse() {
    let server = MockServer::start().await;

    // OpenAI-identical SSE: role chunk, two content chunks, a finish chunk, a
    // usage-only chunk (empty choices), then [DONE]. Data after [DONE] never
    // surfaces. Reused verbatim from the shared OpenAI SSE translation.
    let sse_body = concat!(
        "data: {\"id\":\"o1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"anthropic/claude-sonnet-4\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"o1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"anthropic/claude-sonnet-4\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hel\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"o1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"anthropic/claude-sonnet-4\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"o1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"anthropic/claude-sonnet-4\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        "data: {\"id\":\"o1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"anthropic/claude-sonnet-4\",\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2,\"total_tokens\":7}}\n\n",
        "data: [DONE]\n\n",
        "data: {\"after\":\"done; must never surface\"}\n\n",
    );

    Mock::given(method("POST"))
        .and(path("/api/v1/chat/completions"))
        .and(header("authorization", "Bearer sk-or-test"))
        .and(header("http-referer", "https://routeplane.ai"))
        .and(header("x-title", "Routeplane"))
        .and(body_json(json!({
            "model": "anthropic/claude-sonnet-4",
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

    let provider = OpenRouterProvider::with_base_url(server.uri());
    let stream = provider
        .chat_completion_stream(
            request("anthropic/claude-sonnet-4", vec![msg("user", "hello")]),
            "sk-or-test".to_string(),
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
        .and(path("/api/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(401).set_body_string("invalid credentials"))
        .expect(1)
        .mount(&server)
        .await;

    let provider = OpenRouterProvider::with_base_url(server.uri());
    let err = match provider
        .chat_completion_stream(
            request("openai/gpt-4o", vec![msg("user", "hi")]),
            "sk-or-test".to_string(),
        )
        .await
    {
        Ok(_) => panic!("non-2xx establishment must be Err so the proxy can fall back"),
        Err(e) => e,
    };

    let text = err.to_string();
    assert!(
        text.contains("openrouter authentication failed (401"),
        "got: {text}"
    );
    assert!(!text.contains("invalid credentials"), "got: {text}");
    assert!(
        !text.contains("sk-or-test"),
        "API key must never appear in errors"
    );
}

#[tokio::test]
async fn embeddings_are_unsupported_422_without_a_network_call() {
    // OpenRouter is chat-focused with no embeddings endpoint — the call must
    // degrade to a typed 422 (`embeddings_not_supported`) before any HTTP
    // request, never a panic. We point at a dead address to prove no network
    // call happens.
    let provider = OpenRouterProvider::with_base_url("http://127.0.0.1:1");
    let request = EmbeddingRequest {
        model: "anything".into(),
        input: EmbeddingInput::Single("hello".into()),
        user: None,
        encoding_format: None,
        dimensions: None,
    };
    let err = provider
        .embeddings(request, "sk-or-test".to_string())
        .await
        .expect_err("openrouter has no embeddings endpoint");
    assert_eq!(err.status(), Some(422));
    assert!(err.to_string().contains("embeddings_not_supported"));
}
