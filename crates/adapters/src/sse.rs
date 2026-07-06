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
use routeplane_types::{ChatCompletionChunk, ChatCompletionResponse, ChunkChoice, Delta};

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
    buf: String,
}

impl SseLineBuffer {
    pub fn new() -> Self {
        Self { buf: String::new() }
    }

    /// Append a freshly-read byte chunk. Invalid UTF-8 is replaced rather than
    /// erroring — upstreams send UTF-8 JSON, and a lossy boundary is preferable
    /// to dropping the whole stream.
    pub fn push(&mut self, bytes: &[u8]) {
        self.buf.push_str(&String::from_utf8_lossy(bytes));
    }

    /// Pop the next complete `data:` payload, or `None` if no full line is
    /// buffered yet (caller should read more bytes). Returns the payload with
    /// the `data:` prefix and surrounding whitespace stripped. Non-`data` lines
    /// are consumed and skipped internally.
    pub fn next_payload(&mut self) -> Option<String> {
        loop {
            let newline = self.buf.find('\n')?;
            // Split off the line (without the trailing '\n').
            let line: String = self.buf.drain(..=newline).collect();
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
        let (content, finish) = resp
            .choices
            .first()
            .map(|c| (c.message.content.as_text(), c.finish_reason.clone()))
            .unwrap_or_default();

        // First chunk: role + full content (we have no token-level granularity
        // for a buffered provider, so the "stream" is one content chunk).
        yield Ok(ChatCompletionChunk {
            id: resp.id.clone(),
            object: "chat.completion.chunk".to_string(),
            created: resp.created,
            model: resp.model.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta { role: Some("assistant".to_string()), content: Some(content), ..Delta::default() },
                finish_reason: None,
                logprobs: None,
            }],
            usage: None,
            system_fingerprint: None,
            service_tier: None,
        });

        // Final chunk: finish_reason + usage (mirrors include_usage behavior).
        yield Ok(ChatCompletionChunk {
            id: resp.id,
            object: "chat.completion.chunk".to_string(),
            created: resp.created,
            model: resp.model,
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta::default(),
                finish_reason: Some(if finish.is_empty() { "stop".to_string() } else { finish }),
                logprobs: None,
            }],
            usage: Some(resp.usage),
            system_fingerprint: None,
            service_tier: None,
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
}
