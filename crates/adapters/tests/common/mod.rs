//! Shared helpers for the wiremock adapter-test layer (engineering-design §24:
//! mock-LLM `MockServer`s — these tests never burn real provider tokens and
//! never leave loopback).
//!
//! Each `tests/*.rs` file is compiled as its own crate and includes this module
//! via `mod common;`, so a helper used by one binary may be unused in another —
//! `dead_code` is allowed here on purpose so the workspace-wide
//! `warnings = "deny"` gate stays green.
#![allow(dead_code)]

use futures::StreamExt;
use routeplane_adapters::ChunkStream;
use routeplane_types::{ChatCompletionChunk, ChatCompletionRequest, Message};

/// A canonical message with no author `name`.
pub fn msg(role: &str, content: &str) -> Message {
    Message {
        role: role.to_string(),
        content: content.into(),
        name: None,
        cache_control: None,
        tool_calls: None,
        tool_call_id: None,
    }
}

/// A canonical request with every optional field unset. Tests opt fields in
/// explicitly, so each wire-shape assertion documents exactly what was sent.
pub fn request(model: &str, messages: Vec<Message>) -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: model.to_string(),
        messages,
        temperature: None,
        top_p: None,
        stream: None,
        max_tokens: None,
        stop: None,
        n: None,
        presence_penalty: None,
        frequency_penalty: None,
        user: None,
        tools: None,
        tool_choice: None,
        parallel_tool_calls: None,
        ..Default::default()
    }
}

/// Drain a canonical chunk stream to completion, panicking on any in-stream
/// `Err` item — streaming tests then assert on the resulting chunk sequence.
pub async fn collect_ok(stream: ChunkStream) -> Vec<ChatCompletionChunk> {
    stream
        .map(|item| item.expect("stream yielded an in-stream error"))
        .collect::<Vec<_>>()
        .await
}
