//! Wiremock adapter tests for the Anthropic provider (engineering-design §24).
//! Anthropic is the most-translated dialect: `/v1/messages`, `x-api-key` +
//! `anthropic-version` auth, top-level `system`, required `max_tokens`,
//! `input_tokens`/`output_tokens` usage names, and a multi-event SSE stream.

mod common;

use common::{collect_ok, msg, request};
use routeplane_adapters::anthropic::AnthropicProvider;
use routeplane_adapters::Provider;
use serde_json::json;
use wiremock::matchers::{body_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn buffered_request_maps_to_native_messages_api_shape() {
    let server = MockServer::start().await;

    // Exact-body match proves: system lifted to the TOP-LEVEL field (never
    // inside messages[]), max_tokens present (Anthropic requires it) at the
    // compatibility default 1024 when the caller omits it, and no unset
    // optional fields emitted.
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("x-api-key", "test-key"))
        .and(header("anthropic-version", "2023-06-01"))
        .and(body_json(json!({
            "model": "claude-3-5-sonnet",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 1024,
            "system": "be terse"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_01",
            "model": "claude-3-5-sonnet",
            "content": [{"type": "text", "text": "Hello from Claude"}],
            "usage": {"input_tokens": 17, "output_tokens": 5},
            "stop_reason": "end_turn"
        })))
        .expect(1)
        .mount(&server)
        .await;

    let provider = AnthropicProvider::with_base_url(server.uri());
    assert_eq!(provider.name(), "anthropic");

    let req = request(
        "claude-3-5-sonnet",
        vec![msg("system", "be terse"), msg("user", "hello")],
    );
    let out = provider
        .chat_completion(req, "test-key".to_string())
        .await
        .expect("buffered call against the mock succeeds");

    assert_eq!(out.id, "msg_01");
    assert_eq!(out.object, "chat.completion");
    assert!(out.created > 0, "created is synthesized by the adapter");
    assert_eq!(out.model, "claude-3-5-sonnet");
    assert_eq!(out.choices[0].message.role, "assistant");
    assert_eq!(
        out.choices[0].message.content.as_text(),
        "Hello from Claude"
    );
    // The buffered path normalizes Anthropic's stop_reason to the OpenAI-canonical
    // finish_reason ("end_turn" -> "stop"), matching the streaming path.
    assert_eq!(out.choices[0].finish_reason, "stop");
    // Anthropic's input_tokens/output_tokens map to prompt/completion; the
    // total is synthesized by the adapter.
    assert_eq!(out.usage.prompt_tokens, 17);
    assert_eq!(out.usage.completion_tokens, 5);
    assert_eq!(out.usage.total_tokens, 22);
}

#[tokio::test]
async fn caller_supplied_options_thread_to_the_wire() {
    let server = MockServer::start().await;

    // max_tokens flows through when set (1024 is only the absent-field
    // default); stop -> stop_sequences; temperature passes through; no
    // system message means no top-level "system" field.
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("x-api-key", "test-key"))
        .and(body_json(json!({
            "model": "claude-3-5-sonnet",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4096,
            "temperature": 0.5,
            "stop_sequences": ["END"]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_02",
            "model": "claude-3-5-sonnet",
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 2, "output_tokens": 1},
            "stop_reason": "end_turn"
        })))
        .expect(1)
        .mount(&server)
        .await;

    let provider = AnthropicProvider::with_base_url(server.uri());
    let mut req = request("claude-3-5-sonnet", vec![msg("user", "hi")]);
    req.max_tokens = Some(4096);
    req.temperature = Some(0.5);
    req.stop = Some(vec!["END".to_string()]);

    let out = provider
        .chat_completion(req, "test-key".to_string())
        .await
        .expect("buffered call against the mock succeeds");
    assert_eq!(out.choices[0].message.content.as_text(), "ok");
    assert_eq!(out.usage.total_tokens, 3);
}

#[tokio::test]
async fn streaming_translates_native_event_stream_to_openai_chunks() {
    let server = MockServer::start().await;

    // Native event stream: message_start (identity + input_tokens), ping
    // (ignored), two text deltas, message_delta (stop_reason + cumulative
    // output_tokens), message_stop. Data after message_stop must never surface.
    let sse_body = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_01\",\"model\":\"claude-3-5-sonnet\",\"usage\":{\"input_tokens\":10}}}\n\n",
        "event: ping\n",
        "data: {\"type\":\"ping\"}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hel\"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"lo\"}}\n\n",
        "event: message_delta\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\n",
        "event: message_stop\n",
        "data: {\"type\":\"message_stop\"}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"must not surface\"}}\n\n",
    );

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("x-api-key", "test-key"))
        .and(header("anthropic-version", "2023-06-01"))
        .and(body_json(json!({
            "model": "claude-3-5-sonnet",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 1024,
            "stream": true
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(sse_body.as_bytes().to_vec(), "text/event-stream"),
        )
        .expect(1)
        .mount(&server)
        .await;

    let provider = AnthropicProvider::with_base_url(server.uri());
    let stream = provider
        .chat_completion_stream(
            request("claude-3-5-sonnet", vec![msg("user", "hello")]),
            "test-key".to_string(),
        )
        .await
        .expect("stream establishment succeeds");
    let chunks = collect_ok(stream).await;

    assert_eq!(
        chunks.len(),
        4,
        "message_stop must end the stream; trailing data must not surface"
    );
    // Every chunk is OpenAI-shaped and carries the upstream identity.
    for c in &chunks {
        assert_eq!(c.object, "chat.completion.chunk");
        assert_eq!(c.id, "msg_01");
        assert_eq!(c.model, "claude-3-5-sonnet");
    }
    assert_eq!(
        chunks[0].choices[0].delta.role.as_deref(),
        Some("assistant")
    );
    assert_eq!(chunks[1].choices[0].delta.content.as_deref(), Some("Hel"));
    assert_eq!(chunks[2].choices[0].delta.content.as_deref(), Some("lo"));
    // end_turn maps to OpenAI's "stop" on the streaming path; usage combines
    // input_tokens (message_start) with output_tokens (message_delta).
    assert_eq!(chunks[3].choices[0].finish_reason.as_deref(), Some("stop"));
    let usage = chunks[3]
        .usage
        .as_ref()
        .expect("finish chunk carries usage");
    assert_eq!(usage.prompt_tokens, 10);
    assert_eq!(usage.completion_tokens, 2);
    assert_eq!(usage.total_tokens, 12);
}

#[tokio::test]
async fn streaming_establishment_failure_is_err_and_never_echoes_the_key() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
        .expect(1)
        .mount(&server)
        .await;

    let provider = AnthropicProvider::with_base_url(server.uri());
    let err = match provider
        .chat_completion_stream(
            request("claude-3-5-sonnet", vec![msg("user", "hi")]),
            "test-key".to_string(),
        )
        .await
    {
        Ok(_) => panic!("non-2xx establishment must be Err so the proxy can fall back"),
        Err(e) => e,
    };

    let text = err.to_string();
    assert!(text.contains("anthropic API error (429"), "got: {text}");
    assert!(
        !text.contains("test-key"),
        "API key must never appear in errors"
    );
}
