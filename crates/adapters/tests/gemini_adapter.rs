//! Wiremock adapter tests for the Gemini provider (engineering-design §24).
//! Gemini's dialect: `:generateContent` / `:streamGenerateContent?alt=sse`
//! verbs, API key in the URL query string (no auth header), `contents[]` with
//! a two-role model (user/model), top-level `systemInstruction`, and
//! `generationConfig` for sampling options.

mod common;

use common::{collect_ok, msg, request};
use routeplane_adapters::gemini::GeminiProvider;
use routeplane_adapters::Provider;
use serde_json::json;
use wiremock::matchers::{body_json, method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn buffered_request_maps_to_native_generate_content_shape() {
    let server = MockServer::start().await;

    // Exact-body match proves: system lifted to systemInstruction (never a
    // "model" turn), options mapped into generationConfig (max_tokens ->
    // maxOutputTokens, stop -> stopSequences), and the key rides the query
    // string -- there is no Authorization header in this dialect.
    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-1.5-flash:generateContent"))
        .and(query_param("key", "test-key"))
        .and(body_json(json!({
            "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
            "systemInstruction": {"role": "user", "parts": [{"text": "be terse"}]},
            "generationConfig": {"maxOutputTokens": 256, "stopSequences": ["END"]}
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": "Namaste"}]},
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 8,
                "candidatesTokenCount": 3,
                "totalTokenCount": 11
            }
        })))
        .expect(1)
        .mount(&server)
        .await;

    let provider = GeminiProvider::with_base_url(server.uri());
    assert_eq!(provider.name(), "gemini");

    let mut req = request(
        "gemini-1.5-flash",
        vec![msg("system", "be terse"), msg("user", "hello")],
    );
    req.max_tokens = Some(256);
    req.stop = Some(vec!["END".to_string()]);

    let out = provider
        .chat_completion(req, "test-key".to_string())
        .await
        .expect("buffered call against the mock succeeds");

    assert_eq!(
        out.id, "gemini-resp",
        "synthesized; Gemini has no top-level id"
    );
    assert_eq!(out.object, "chat.completion");
    assert_eq!(out.model, "gemini-1.5-flash", "echoes the requested model");
    assert_eq!(out.choices[0].message.role, "assistant");
    assert_eq!(out.choices[0].message.content.as_text(), "Namaste");
    // The buffered path normalizes Gemini's uppercase finishReason to the
    // OpenAI-canonical value ("STOP" -> "stop"), matching the streaming path.
    assert_eq!(out.choices[0].finish_reason, "stop");
    assert_eq!(out.usage.prompt_tokens, 8);
    assert_eq!(out.usage.completion_tokens, 3);
    assert_eq!(out.usage.total_tokens, 11);
}

#[tokio::test]
async fn buffered_maps_non_stop_finish_reasons_to_openai_canonical() {
    // A truncated response: Gemini reports "MAX_TOKENS", which the buffered path
    // must normalize to OpenAI's "length" (previously leaked verbatim, so clients
    // could not detect truncation).
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-1.5-flash:generateContent"))
        .and(query_param("key", "test-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": "trunc"}]},
                "finishReason": "MAX_TOKENS"
            }],
            "usageMetadata": {
                "promptTokenCount": 4,
                "candidatesTokenCount": 5,
                "totalTokenCount": 9
            }
        })))
        .mount(&server)
        .await;

    let provider = GeminiProvider::with_base_url(server.uri());
    let out = provider
        .chat_completion(
            request("gemini-1.5-flash", vec![msg("user", "hello")]),
            "test-key".to_string(),
        )
        .await
        .expect("buffered call against the mock succeeds");
    assert_eq!(out.choices[0].finish_reason, "length");
}

#[tokio::test]
async fn role_mapping_and_no_empty_config_objects_on_the_wire() {
    let server = MockServer::start().await;

    // assistant -> "model"; with no system message and no options set, the
    // exact-body match proves systemInstruction and generationConfig are
    // entirely absent (no empty objects emitted).
    Mock::given(method("POST"))
        .and(path("/v1beta/models/gemini-1.5-flash:generateContent"))
        .and(query_param("key", "test-key"))
        .and(body_json(json!({
            "contents": [
                {"role": "user", "parts": [{"text": "hi"}]},
                {"role": "model", "parts": [{"text": "hello there"}]},
                {"role": "user", "parts": [{"text": "again"}]}
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": "ok"}]},
                "finishReason": "STOP"
            }]
        })))
        .expect(1)
        .mount(&server)
        .await;

    let provider = GeminiProvider::with_base_url(server.uri());
    let req = request(
        "gemini-1.5-flash",
        vec![
            msg("user", "hi"),
            msg("assistant", "hello there"),
            msg("user", "again"),
        ],
    );
    let out = provider
        .chat_completion(req, "test-key".to_string())
        .await
        .expect("buffered call against the mock succeeds");

    assert_eq!(out.choices[0].message.content.as_text(), "ok");
    // Pin current behavior: a response without usageMetadata yields zeroed usage.
    assert_eq!(out.usage.prompt_tokens, 0);
    assert_eq!(out.usage.completion_tokens, 0);
    assert_eq!(out.usage.total_tokens, 0);
}

#[tokio::test]
async fn streaming_uses_alt_sse_and_translates_to_openai_chunks() {
    let server = MockServer::start().await;

    // Gemini SSE: each data payload is a partial GenerateContentResponse; the
    // final one carries finishReason + usageMetadata. There is no [DONE]
    // sentinel -- the stream ends when the body ends.
    let sse_body = concat!(
        "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"Nama\"}]}}]}\n\n",
        "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"ste\"}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":4,\"candidatesTokenCount\":2,\"totalTokenCount\":6}}\n\n",
    );

    Mock::given(method("POST"))
        .and(path(
            "/v1beta/models/gemini-1.5-flash:streamGenerateContent",
        ))
        .and(query_param("alt", "sse"))
        .and(query_param("key", "test-key"))
        .and(body_json(json!({
            "contents": [{"role": "user", "parts": [{"text": "hello"}]}]
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(sse_body.as_bytes().to_vec(), "text/event-stream"),
        )
        .expect(1)
        .mount(&server)
        .await;

    let provider = GeminiProvider::with_base_url(server.uri());
    let stream = provider
        .chat_completion_stream(
            request("gemini-1.5-flash", vec![msg("user", "hello")]),
            "test-key".to_string(),
        )
        .await
        .expect("stream establishment succeeds");
    let chunks = collect_ok(stream).await;

    // "Nama" + "ste" + finish chunk; end-of-stream is end-of-body.
    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0].id, "gemini-stream");
    assert_eq!(chunks[0].model, "gemini-1.5-flash");
    assert_eq!(chunks[0].choices[0].delta.content.as_deref(), Some("Nama"));
    assert_eq!(chunks[1].choices[0].delta.content.as_deref(), Some("ste"));
    // STOP maps to "stop" on the streaming path.
    assert_eq!(chunks[2].choices[0].finish_reason.as_deref(), Some("stop"));
    let usage = chunks[2].usage.as_ref().expect("final chunk carries usage");
    assert_eq!(usage.prompt_tokens, 4);
    assert_eq!(usage.completion_tokens, 2);
    assert_eq!(usage.total_tokens, 6);
}

#[tokio::test]
async fn upstream_error_never_echoes_the_query_string_key() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(400).set_body_string("bad request"))
        .expect(1)
        .mount(&server)
        .await;

    let provider = GeminiProvider::with_base_url(server.uri());
    let err = provider
        .chat_completion(
            request("gemini-1.5-flash", vec![msg("user", "hi")]),
            "test-key".to_string(),
        )
        .await
        .expect_err("400 must surface as Err");

    let text = err.to_string();
    assert!(text.contains("gemini API error (400"), "got: {text}");
    // The Gemini key rides the URL query string -- it must never leak into
    // the error message (Task #3d sanitization).
    assert!(
        !text.contains("test-key"),
        "API key must never appear in errors"
    );
}
