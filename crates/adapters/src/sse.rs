//! Shared Server-Sent-Events (SSE) plumbing for streaming provider adapters.
//!
//! Every streaming-capable upstream (OpenAI, Azure OpenAI, Anthropic, Gemini)
//! speaks SSE, but the *event payloads* differ. This module owns the parts that
//! are identical across providers:
//!
//!   * `SseLineBuffer` — a small, allocation-light state machine that turns the
//!     arbitrarily-chunked bytes from `reqwest::Response::bytes_stream()` into
//!     whole `data:` payloads. A single TCP read can contain a partial line, a
//!     line and a half, or several lines; this buffers across reads so the
//!     per-provider translator only ever sees complete `data:` payloads.
//!   * `buffered_response_as_stream` — wraps a fully-buffered
//!     `ChatCompletionResponse` as a one-shot chunk stream, used as the trait's
//!     default streaming impl (a provider with no native streaming still honors
//!     the OpenAI streaming contract).
//!
//! The convention `[DONE]` is OpenAI/Azure-specific; this module just extracts
//! the payload string after `data:` and lets the caller decide what `[DONE]`
//! means for its dialect.

use futures::Stream;
use routeplane_types::{
    ChatCompletionChunk, ChatCompletionResponse, Choice, ChunkChoice, Delta, FunctionCallChunk,
    ToolCall, ToolCallChunk,
};

/// Convert a buffered response's complete `tool_calls` into a single set of
/// streaming `ToolCallChunk` deltas (each carries the whole call — id + type +
/// full function name/arguments — since a buffered provider has no token-level
/// granularity). `None` when there are none, so a content-only delta stays
/// byte-identical.
fn tool_calls_to_chunks(calls: &[ToolCall]) -> Option<Vec<ToolCallChunk>> {
    if calls.is_empty() {
        return None;
    }
    Some(
        calls
            .iter()
            .enumerate()
            .map(|(i, tc)| ToolCallChunk {
                index: i as u32,
                id: Some(tc.id.clone()),
                tool_type: Some(tc.tool_type.clone()),
                function: Some(FunctionCallChunk {
                    name: Some(tc.function.name.clone()),
                    arguments: Some(tc.function.arguments.clone()),
                }),
            })
            .collect(),
    )
}

/// The opening delta for one response choice: role + content + any tool_calls /
/// refusal / reasoning_content, with the choice's index and logprobs preserved.
/// This is what stops the trait default from silently dropping tool calls (which
/// left clients a `finish_reason:"tool_calls"` with NO tool_calls), extra
/// choices, and logprobs.
fn opening_chunk_choice(c: &Choice) -> ChunkChoice {
    let content = c.message.content.as_text();
    let has_tool_calls = c.message.tool_calls.as_ref().is_some_and(|t| !t.is_empty());
    ChunkChoice {
        index: c.index,
        delta: Delta {
            role: Some("assistant".to_string()),
            // Omit an empty content string when the turn is a tool call / refusal
            // (that is how a native stream renders it), else pass the text.
            content: if content.is_empty() && has_tool_calls {
                None
            } else {
                Some(content)
            },
            tool_calls: c
                .message
                .tool_calls
                .as_deref()
                .and_then(tool_calls_to_chunks),
            refusal: c.message.refusal.clone(),
            reasoning_content: c.message.reasoning_content.clone(),
        },
        finish_reason: None,
        logprobs: c.logprobs.clone(),
    }
}

/// Incrementally re-assembles SSE `data:` payloads from a byte stream.
///
/// Feed it raw bytes as they arrive (`push`), then drain whole payloads with
/// `next_payload()`. SSE frames are newline-delimited; a payload is the text
/// following a `data:` prefix on a single line. We treat `\n` as the line
/// terminator and tolerate a trailing `\r` (CRLF). Blank lines (event
/// separators) and non-`data:` fields (`event:`, `id:`, `:` comments / keep-
/// alives) are ignored — providers here put their whole JSON on `data:` lines.
#[derive(Default)]
pub struct SseLineBuffer {
    buf: Vec<u8>,
}

impl SseLineBuffer {
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    /// Append a freshly-read byte chunk. Bytes are buffered **raw** and decoding
    /// is deferred to [`next_payload`](Self::next_payload). This matters for
    /// non-ASCII: a `bytes_stream()` read can split a multibyte UTF-8 codepoint
    /// (Devanagari/Tamil/CJK/emoji) across two chunks, and decoding each chunk in
    /// isolation with `from_utf8_lossy` would turn the split codepoint into
    /// permanent `U+FFFD` mojibake mid-stream even though the upstream sent valid
    /// UTF-8.
    pub fn push(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }

    /// Pop the next complete `data:` payload, or `None` if no full line is
    /// buffered yet (caller should read more bytes). Returns the payload with
    /// the `data:` prefix and surrounding whitespace stripped. Non-`data` lines
    /// are consumed and skipped internally.
    pub fn next_payload(&mut self) -> Option<String> {
        loop {
            let newline = self.buf.iter().position(|&b| b == b'\n')?;
            // Drain the COMPLETE line (including the trailing '\n'). Decoding is
            // safe here: `\n` (0x0A) is never a UTF-8 continuation byte, so a line
            // boundary never falls inside a codepoint — the only place a split can
            // occur is a chunk boundary, and those are already reassembled here.
            let line_bytes: Vec<u8> = self.buf.drain(..=newline).collect();
            let line = String::from_utf8_lossy(&line_bytes);
            let line = line.trim_end_matches('\n').trim_end_matches('\r');

            if let Some(rest) = line.strip_prefix("data:") {
                return Some(rest.trim().to_string());
            }
            // Blank line, comment (`:`), or other SSE field — skip and continue.
        }
    }
}

/// Adapt a fully-buffered response into a single-chunk stream followed by a
/// terminating finish-reason chunk that carries usage. Used as the default
/// `Provider::chat_completion_stream` implementation.
pub fn buffered_response_as_stream(
    resp: ChatCompletionResponse,
) -> impl Stream<Item = Result<ChatCompletionChunk, crate::ProviderError>> + Send + 'static {
    async_stream::stream! {
        // First chunk: ONE ChunkChoice per response choice (not just the first),
        // each carrying its full delta — content, tool_calls, refusal,
        // reasoning_content, logprobs. A buffered provider has no token-level
        // granularity, so each choice's whole delta arrives in this one chunk.
        let opening: Vec<ChunkChoice> = if resp.choices.is_empty() {
            vec![ChunkChoice {
                index: 0,
                delta: Delta { role: Some("assistant".to_string()), content: Some(String::new()), ..Delta::default() },
                finish_reason: None,
                logprobs: None,
            }]
        } else {
            resp.choices.iter().map(opening_chunk_choice).collect()
        };
        yield Ok(ChatCompletionChunk {
            id: resp.id.clone(),
            object: "chat.completion.chunk".to_string(),
            created: resp.created,
            model: resp.model.clone(),
            choices: opening,
            usage: None,
            system_fingerprint: resp.system_fingerprint.clone(),
            service_tier: resp.service_tier.clone(),
        });

        // Final chunk: per-choice finish_reason + usage (mirrors include_usage).
        let closing: Vec<ChunkChoice> = if resp.choices.is_empty() {
            vec![ChunkChoice { index: 0, delta: Delta::default(), finish_reason: Some("stop".to_string()), logprobs: None }]
        } else {
            resp.choices.iter().map(|c| ChunkChoice {
                index: c.index,
                delta: Delta::default(),
                finish_reason: Some(if c.finish_reason.is_empty() { "stop".to_string() } else { c.finish_reason.clone() }),
                logprobs: None,
            }).collect()
        };
        yield Ok(ChatCompletionChunk {
            id: resp.id,
            object: "chat.completion.chunk".to_string(),
            created: resp.created,
            model: resp.model,
            choices: closing,
            usage: Some(resp.usage),
            system_fingerprint: resp.system_fingerprint,
            service_tier: resp.service_tier,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffers_partial_lines_across_pushes() {
        let mut b = SseLineBuffer::new();
        b.push(b"data: hel");
        assert_eq!(b.next_payload(), None); // no newline yet
        b.push(b"lo\n");
        assert_eq!(b.next_payload().as_deref(), Some("hello"));
        assert_eq!(b.next_payload(), None);
    }

    #[test]
    fn handles_multiple_lines_in_one_push_and_crlf() {
        let mut b = SseLineBuffer::new();
        b.push(b"data: one\r\ndata: two\r\n");
        assert_eq!(b.next_payload().as_deref(), Some("one"));
        assert_eq!(b.next_payload().as_deref(), Some("two"));
        assert_eq!(b.next_payload(), None);
    }

    #[test]
    fn skips_blank_and_non_data_lines() {
        let mut b = SseLineBuffer::new();
        b.push(b": keep-alive\n\nevent: message\ndata: payload\n");
        assert_eq!(b.next_payload().as_deref(), Some("payload"));
    }

    #[test]
    fn multibyte_utf8_split_across_pushes_is_not_corrupted() {
        // Devanagari "नमस्ते": each codepoint is 3 bytes. A read boundary lands
        // INSIDE the first codepoint; buffering raw bytes avoids U+FFFD mojibake.
        let full = "data: नमस्ते\n".as_bytes();
        let split = 7;
        assert!(
            full[split] & 0b1100_0000 == 0b1000_0000,
            "split must be mid-codepoint"
        );
        let mut b = SseLineBuffer::new();
        b.push(&full[..split]);
        assert_eq!(b.next_payload(), None);
        b.push(&full[split..]);
        assert_eq!(b.next_payload().as_deref(), Some("नमस्ते"));
    }

    #[test]
    fn multibyte_emoji_split_survives() {
        let full = "data: 😀\n".as_bytes();
        let mut b = SseLineBuffer::new();
        b.push(&full[..8]);
        b.push(&full[8..]);
        assert_eq!(b.next_payload().as_deref(), Some("😀"));
    }

    // --- buffered_response_as_stream: faithful translation (ledger #38) --------

    use futures::StreamExt;
    use routeplane_types::{Choice, FunctionCall, Message, ToolCall, Usage};

    fn msg(content: &str, tool_calls: Option<Vec<ToolCall>>, refusal: Option<String>) -> Message {
        Message {
            role: "assistant".into(),
            content: content.into(),
            name: None,
            cache_control: None,
            tool_calls,
            tool_call_id: None,
            refusal,
            reasoning_content: None,
        }
    }

    fn resp(choices: Vec<Choice>) -> ChatCompletionResponse {
        ChatCompletionResponse {
            id: "resp-1".into(),
            object: "chat.completion".into(),
            created: 0,
            model: "m".into(),
            choices,
            usage: Usage::default(),
            system_fingerprint: None,
            service_tier: None,
        }
    }

    async fn chunks(r: ChatCompletionResponse) -> Vec<ChatCompletionChunk> {
        buffered_response_as_stream(r)
            .map(|c| c.expect("chunk ok"))
            .collect()
            .await
    }

    #[tokio::test]
    async fn tool_calls_survive_the_buffered_stream() {
        let tc = ToolCall {
            id: "call_1".into(),
            tool_type: "function".into(),
            function: FunctionCall {
                name: "get_weather".into(),
                arguments: "{\"city\":\"NYC\"}".into(),
            },
        };
        let out = chunks(resp(vec![Choice {
            index: 0,
            message: msg("", Some(vec![tc]), None),
            finish_reason: "tool_calls".into(),
            logprobs: None,
        }]))
        .await;

        // The opening delta carries the tool call (was silently dropped before).
        let opening = &out[0].choices[0];
        let tcs = opening
            .delta
            .tool_calls
            .as_ref()
            .expect("tool_calls must survive the default stream");
        assert_eq!(tcs[0].id.as_deref(), Some("call_1"));
        assert_eq!(
            tcs[0].function.as_ref().and_then(|f| f.name.as_deref()),
            Some("get_weather")
        );
        assert!(
            opening.delta.content.is_none(),
            "pure tool-call turn omits content"
        );
        // finish_reason:tool_calls is no longer orphaned (a delta with no calls).
        assert_eq!(
            out.last().unwrap().choices[0].finish_reason.as_deref(),
            Some("tool_calls")
        );
    }

    #[tokio::test]
    async fn all_choices_and_logprobs_survive() {
        let out = chunks(resp(vec![
            Choice {
                index: 0,
                message: msg("hello", None, None),
                finish_reason: "stop".into(),
                logprobs: Some(serde_json::json!({"content": []})),
            },
            Choice {
                index: 1,
                message: msg("world", None, None),
                finish_reason: "length".into(),
                logprobs: None,
            },
        ]))
        .await;

        // Both choices stream (index 0 AND 1) — not just the first.
        assert_eq!(out[0].choices.len(), 2);
        assert_eq!(out[0].choices[0].index, 0);
        assert_eq!(out[0].choices[1].index, 1);
        assert_eq!(out[0].choices[0].delta.content.as_deref(), Some("hello"));
        assert_eq!(out[0].choices[1].delta.content.as_deref(), Some("world"));
        assert!(out[0].choices[0].logprobs.is_some(), "logprobs preserved");
        // Per-choice finish_reason on the closing chunk.
        let closing = out.last().unwrap();
        assert_eq!(closing.choices.len(), 2);
        assert_eq!(closing.choices[0].finish_reason.as_deref(), Some("stop"));
        assert_eq!(closing.choices[1].finish_reason.as_deref(), Some("length"));
    }

    #[tokio::test]
    async fn refusal_delta_is_carried() {
        let out = chunks(resp(vec![Choice {
            index: 0,
            message: msg("", None, Some("I can't help with that".into())),
            finish_reason: "stop".into(),
            logprobs: None,
        }]))
        .await;
        assert_eq!(
            out[0].choices[0].delta.refusal.as_deref(),
            Some("I can't help with that")
        );
    }
}
