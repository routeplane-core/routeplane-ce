//! Wiremock adapter tests for the parameterized `OpenAICompatProvider` (the
//! family of OpenAI-wire-compatible providers: groq, mistral, deepseek, …).
//! Covers: (a) canonical -> wire request translation incl. threaded optionals,
//! (b) the configured `name()` flowing into errors, (c) SSE streaming with the
//! forced stream flags, (d) the embeddings opt-in vs the typed 422 degrade, and
//! (e) typed error classification that never leaks the key.

mod common;

use common::{collect_ok, msg, request};
use routeplane_adapters::openai_compat::OpenAICompatProvider;
use routeplane_adapters::{Provider, RetryClass};
use routeplane_types::{EmbeddingInput, EmbeddingRequest};
use serde_json::json;
use wiremock::matchers::{body_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn buffered_request_translates_and_uses_configured_name() {
    let server = MockServer::start().await;

    // The adapter appends `/chat/completions` to the configured base URL and
    // forwards the canonical body verbatim (OpenAI wire dialect). `body_json` is
    // an EXACT match — it passes only because `ChatCompletionRequest` skips
    // `None` optionals, so the unset fields never serialize.
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer gsk-test"))
        .and(body_json(json!({
            "model": "llama-3.3-70b",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 128,
            "stop": ["STOP"],
            "presence_penalty": 0.25
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-groq-1",
            "object": "chat.completion",
            "created": 1_700_000_000u64,
            "model": "llama-3.3-70b",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hi from Groq"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7}
        })))
        .expect(1)
        .mount(&server)
        .await;

    let provider = OpenAICompatProvider::new("groq", server.uri());
    assert_eq!(provider.name(), "groq");

    let mut req = request("llama-3.3-70b", vec![msg("user", "hello")]);
    req.max_tokens = Some(128);
    req.stop = Some(vec!["STOP".to_string()]);
    req.presence_penalty = Some(0.25);

    let out = provider
        .chat_completion(req, "gsk-test".to_string())
        .await
        .expect("buffered call succeeds");

    assert_eq!(out.id, "chatcmpl-groq-1");
    assert_eq!(out.choices[0].message.content, "Hi from Groq".into());
    assert_eq!(out.usage.total_tokens, 7);
}

#[tokio::test]
async fn streaming_forces_stream_flags_and_translates_sse() {
    let server = MockServer::start().await;

    let sse_body = concat!(
        "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Yo\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":1,\"total_tokens\":4}}\n\n",
        "data: [DONE]\n\n",
    );

    // The adapter forces `stream: true` and adds `stream_options.include_usage`
    // (default-on) — assert both land on the wire.
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_json(json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
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

    let provider = OpenAICompatProvider::new("together", server.uri());
    let stream = provider
        .chat_completion_stream(request("m", vec![msg("user", "hi")]), "tk".to_string())
        .await
        .expect("stream establishment succeeds");
    let chunks = collect_ok(stream).await;

    assert_eq!(chunks.len(), 3, "[DONE] ends the stream");
    assert_eq!(chunks[1].choices[0].delta.content.as_deref(), Some("Yo"));
    assert_eq!(chunks[2].usage.as_ref().unwrap().total_tokens, 4);
}

#[tokio::test]
async fn embeddings_opt_in_hits_endpoint() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .and(header("authorization", "Bearer mk"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
            "model": "mistral-embed",
            "usage": {"prompt_tokens": 2, "total_tokens": 2}
        })))
        .expect(1)
        .mount(&server)
        .await;

    let provider = OpenAICompatProvider::new("mistral", server.uri()).with_embeddings();
    let resp = provider
        .embeddings(
            EmbeddingRequest {
                model: "mistral-embed".into(),
                input: EmbeddingInput::Single("hi".into()),
                encoding_format: None,
                dimensions: None,
                user: None,
            },
            "mk".to_string(),
        )
        .await
        .expect("embeddings call succeeds");
    assert_eq!(resp.data[0].embedding.len(), 3);
    assert_eq!(resp.usage.total_tokens, 2);
}

#[tokio::test]
async fn embeddings_without_support_degrades_to_typed_422_without_network() {
    // No mock mounted: a chat-only provider must NOT call any endpoint — it
    // returns the typed 422 capability-gap error directly.
    let server = MockServer::start().await;
    let provider = OpenAICompatProvider::new("groq", server.uri());

    let err = provider
        .embeddings(
            EmbeddingRequest {
                model: "whatever".into(),
                input: EmbeddingInput::Single("hi".into()),
                encoding_format: None,
                dimensions: None,
                user: None,
            },
            "gsk".to_string(),
        )
        .await
        .expect_err("chat-only provider has no embeddings endpoint");

    assert_eq!(err.status(), Some(422));
    let msg = err.to_string();
    assert!(msg.contains("embeddings_not_supported"), "got: {msg}");
    assert!(msg.contains("groq"), "got: {msg}");
}

#[tokio::test]
async fn upstream_429_is_typed_rate_limited_and_never_echoes_key() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(429).set_body_string("slow down"))
        .expect(1)
        .mount(&server)
        .await;

    let provider = OpenAICompatProvider::new("deepseek", server.uri());
    let err = provider
        .chat_completion(
            request("deepseek-chat", vec![msg("user", "hi")]),
            "sk-secret".to_string(),
        )
        .await
        .expect_err("429 is an Err");

    assert_eq!(err.retry_class(), RetryClass::Status(429));
    let text = err.to_string();
    assert!(text.contains("429"), "got: {text}");
    assert!(text.contains("deepseek"), "got: {text}");
    assert!(
        !text.contains("sk-secret"),
        "key must never appear in errors"
    );
}

#[tokio::test]
async fn stream_usage_opt_out_omits_stream_options() {
    let server = MockServer::start().await;

    let sse_body =
        "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"x\"},\"finish_reason\":null}]}\n\ndata: [DONE]\n\n";

    // With `.without_stream_usage()`, `stream_options` must NOT appear — assert
    // the exact body so a leaked `stream_options` field fails the match.
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_json(json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(sse_body.as_bytes().to_vec(), "text/event-stream"),
        )
        .expect(1)
        .mount(&server)
        .await;

    let provider = OpenAICompatProvider::new("quirky", server.uri()).without_stream_usage();
    let stream = provider
        .chat_completion_stream(request("m", vec![msg("user", "hi")]), "k".to_string())
        .await
        .expect("stream establishment succeeds");
    let chunks = collect_ok(stream).await;
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].choices[0].delta.content.as_deref(), Some("x"));
}
