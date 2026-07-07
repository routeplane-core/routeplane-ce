//! Wiremock adapter tests for the Azure OpenAI provider (engineering-design
//! §24 -- this is the spec's canonical mock: POST
//! /openai/deployments/gpt-4o-india/chat/completions). Azure OpenAI is
//! wire-compatible with OpenAI; only the URL shape (deployment path +
//! api-version query) and auth header (`api-key`, not Bearer) differ. It is
//! also the residency-bearing provider for sovereign routing.

mod common;

use common::{collect_ok, msg, request};
use routeplane_adapters::azure_openai::AzureOpenAiProvider;
use routeplane_adapters::Provider;
use serde_json::json;
use wiremock::matchers::{body_json, header, method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn provider_for(server: &MockServer) -> AzureOpenAiProvider {
    AzureOpenAiProvider::new(
        server.uri(),
        "gpt-4o-india".to_string(),
        "2024-10-21".to_string(),
        "IN".to_string(),
    )
}

#[tokio::test]
async fn buffered_request_hits_deployment_path_with_api_key_header() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/openai/deployments/gpt-4o-india/chat/completions"))
        .and(query_param("api-version", "2024-10-21"))
        .and(header("api-key", "azure-test-key"))
        .and(body_json(json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hello"}]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-az",
            "object": "chat.completion",
            "created": 1_700_000_000u64,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello from in-region"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9}
        })))
        .expect(1)
        .mount(&server)
        .await;

    let provider = provider_for(&server);
    assert_eq!(provider.name(), "azure_openai");

    let out = provider
        .chat_completion(
            request("gpt-4o", vec![msg("user", "hello")]),
            "azure-test-key".to_string(),
        )
        .await
        .expect("buffered call against the mock succeeds");

    assert_eq!(out.id, "chatcmpl-az");
    assert_eq!(
        out.choices[0].message.content.as_text(),
        "Hello from in-region"
    );
    assert_eq!(out.choices[0].finish_reason, "stop");
    assert_eq!(out.usage.prompt_tokens, 5);
    assert_eq!(out.usage.completion_tokens, 4);
    assert_eq!(out.usage.total_tokens, 9);
}

#[tokio::test]
async fn streaming_is_openai_wire_compatible() {
    let server = MockServer::start().await;

    let sse_body = concat!(
        "data: {\"id\":\"az1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hi\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"az1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":1,\"total_tokens\":4}}\n\n",
        "data: [DONE]\n\n",
    );

    Mock::given(method("POST"))
        .and(path("/openai/deployments/gpt-4o-india/chat/completions"))
        .and(query_param("api-version", "2024-10-21"))
        .and(header("api-key", "azure-test-key"))
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

    let provider = provider_for(&server);
    let stream = provider
        .chat_completion_stream(
            request("gpt-4o", vec![msg("user", "hello")]),
            "azure-test-key".to_string(),
        )
        .await
        .expect("stream establishment succeeds");
    let chunks = collect_ok(stream).await;

    assert_eq!(chunks.len(), 2, "[DONE] must end the stream");
    assert_eq!(
        chunks[0].choices[0].delta.role.as_deref(),
        Some("assistant")
    );
    assert_eq!(chunks[0].choices[0].delta.content.as_deref(), Some("Hi"));
    assert_eq!(chunks[1].choices[0].finish_reason.as_deref(), Some("stop"));
    assert_eq!(
        chunks[1]
            .usage
            .as_ref()
            .expect("usage on final chunk")
            .total_tokens,
        4
    );
}

#[test]
fn declares_its_residency_region_for_sovereign_eligibility() {
    let provider = AzureOpenAiProvider::new(
        "https://example.invalid".to_string(),
        "gpt-4o-india".to_string(),
        "2024-10-21".to_string(),
        "IN".to_string(),
    );
    assert_eq!(provider.resident_regions(), vec!["IN".to_string()]);
    assert!(provider.is_resident_in("IN"));
    assert!(!provider.is_resident_in("EU"));
}

#[tokio::test]
async fn unconfigured_endpoint_fails_fast_without_network() {
    let provider = AzureOpenAiProvider::new(
        String::new(),
        "gpt-4o-india".to_string(),
        "2024-10-21".to_string(),
        "IN".to_string(),
    );
    let err = provider
        .chat_completion(request("gpt-4o", vec![msg("user", "hi")]), "k".to_string())
        .await
        .expect_err("missing endpoint must be a configuration error");
    assert!(err.to_string().contains("AZURE_OPENAI_ENDPOINT"));
}

/// A configured model→deployment map is authoritative: the request's `model`
/// selects the deployment (the URL path segment), NOT the single default — this
/// is what stops silent model substitution for multi-model Azure tenants.
#[tokio::test]
async fn mapped_model_routes_to_its_deployment() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        // routed to the *mapped* deployment, not the "default-dep" fallback
        .and(path("/openai/deployments/gpt-4o-india/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-az", "object": "chat.completion", "created": 1_700_000_000u64,
            "model": "gpt-4o",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        })))
        .mount(&server)
        .await;

    let map = std::collections::HashMap::from([("gpt-4o".to_string(), "gpt-4o-india".to_string())]);
    let provider = AzureOpenAiProvider::new(
        server.uri(),
        "default-dep".to_string(), // must be OVERRIDDEN by the map for gpt-4o
        "2024-10-21".to_string(),
        "IN".to_string(),
    )
    .with_deployments(map);

    let out = provider
        .chat_completion(
            request("gpt-4o", vec![msg("user", "hi")]),
            "azure-test-key".to_string(),
        )
        .await
        .expect("mapped model routes and succeeds");
    assert_eq!(out.choices[0].message.content.as_text(), "hi");
}

/// With a map configured, a model that is NOT in it is REFUSED (422) — never
/// silently substituted onto another deployment — and the refusal happens
/// locally, before any network call.
#[tokio::test]
async fn unmapped_model_is_refused_422_without_network() {
    let map = std::collections::HashMap::from([("gpt-4o".to_string(), "gpt-4o-india".to_string())]);
    let provider = AzureOpenAiProvider::new(
        "https://example.invalid".to_string(), // non-empty so we pass the endpoint check
        "default-dep".to_string(),
        "2024-10-21".to_string(),
        "IN".to_string(),
    )
    .with_deployments(map);

    let err = provider
        .chat_completion(
            request("gpt-4o-mini", vec![msg("user", "hi")]),
            "k".to_string(),
        )
        .await
        .expect_err("an unmapped model must be refused, not substituted");
    assert_eq!(
        err.status(),
        Some(422),
        "unmapped model is a 422 capability gap"
    );
    assert!(
        err.to_string().contains("azure_deployment_unmapped"),
        "body carries the recognizable code: {err}"
    );
}

/// No map configured ⇒ legacy single-deployment mode: every model routes to the
/// one default deployment (byte-identical to before the map existed).
#[tokio::test]
async fn no_map_routes_every_model_to_the_default_deployment() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/openai/deployments/solo-dep/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-az", "object": "chat.completion", "created": 1_700_000_000u64,
            "model": "anything",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        })))
        .mount(&server)
        .await;

    let provider = AzureOpenAiProvider::new(
        server.uri(),
        "solo-dep".to_string(),
        "2024-10-21".to_string(),
        "IN".to_string(),
    );
    // A model unknown to any map still routes to the single deployment.
    let out = provider
        .chat_completion(
            request("some-unmapped-model", vec![msg("user", "hi")]),
            "azure-test-key".to_string(),
        )
        .await
        .expect("single-deployment mode serves any model");
    assert_eq!(out.choices[0].message.content.as_text(), "ok");
}
