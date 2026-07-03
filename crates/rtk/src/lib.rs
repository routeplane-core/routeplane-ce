//! RTK Token Compression — deterministic tool_result block filters.
//!
//! Port of 9router's RTK Token Saver: auto-compresses `tool_result` content
//! before forwarding to the LLM, saving 20–40% input tokens per request.
//!
//! ## Design
//! - **Auto-detect**: peek at the first 1KB of content, pick the best filter.
//! - **11 filters**: git-diff, git-status, grep, find, ls, tree, dedup-log,
//!   smart-truncate, read-numbered, search-list, build-output.
//! - **Fail-safe**: never returns empty, never grows input.
//! - **PII-safe**: deterministic string transforms, no ML, no external calls.
//! - **<1ms**: pure string processing, no allocations beyond the output buffer.
//!
//! ## Integration
//! The proxy calls [`compress_tool_results`] on the request body before
//! forwarding to the provider. Gated per-tenant via entitlement.

/// The detection window for auto-picking a filter (first 1KB).
const PEEK_BYTES: usize = 1024;

/// Maximum output size — if compression doesn't shrink, return original.
const MAX_COMPRESSION_RATIO: f64 = 0.95;

/// Available compression filters, auto-detected from content.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Filter {
    /// `git diff` output — collapse unchanged context lines, keep hunks.
    GitDiff,
    /// `git status` output — collapse verbose untracked listings.
    GitStatus,
    /// `grep` output — deduplicate file-level matches, keep first N per file.
    Grep,
    /// `find` output — collapse deep directory listings.
    Find,
    /// `ls` output — collapse long directory listings.
    Ls,
    /// `tree` output — prune deep subtrees, keep top N levels.
    Tree,
    /// Repetitive log output — deduplicate consecutive identical blocks.
    DedupLog,
    /// Generic smart truncation — keep first + last N lines of long output.
    SmartTruncate,
    /// Numbered file content (`cat -n` / Read tool) — collapse unchanged regions.
    ReadNumbered,
    /// Search result lists — collapse to summary + top results.
    SearchList,
    /// Build/compile output — collapse per-file compilation lines, keep errors.
    BuildOutput,
}

/// Detect the best filter for the given content by peeking at the first
/// [`PEEK_BYTES`] bytes. Returns `None` if no filter matches (content is
/// not a recognized tool_result pattern).
pub fn detect_filter(content: &str) -> Option<Filter> {
    let peek = if content.len() > PEEK_BYTES {
        // Floor to the largest char boundary <= PEEK_BYTES. Slicing at a raw
        // byte offset panics if it lands inside a multi-byte UTF-8 character
        // (e.g. Hindi/CJK/emoji/em-dash straddling byte 1024) — a reachable
        // panic on the chat hot path. `str::floor_char_boundary` is unstable on
        // the pinned Rust 1.86, so walk back manually (byte 0 is always a
        // boundary, so this always terminates).
        let mut end = PEEK_BYTES;
        while !content.is_char_boundary(end) {
            end -= 1;
        }
        &content[..end]
    } else {
        content
    };

    // Order matters — more specific patterns first.
    if peek.starts_with("diff --git") || peek.contains("@@ ") && peek.contains("+++") {
        Some(Filter::GitDiff)
    } else if peek.starts_with("On branch ") || peek.contains("Changes not staged") {
        Some(Filter::GitStatus)
    } else if peek.contains("warning[") || peek.contains("error[E") || peek.contains("Compiling ") {
        Some(Filter::BuildOutput)
    } else if peek
        .lines()
        .take(5)
        .any(|l| l.starts_with("   1\t") || l.starts_with("     1\t"))
    {
        Some(Filter::ReadNumbered)
    } else if peek
        .lines()
        .take(5)
        .any(|l| l.contains(':') && l.contains(" match"))
    {
        Some(Filter::Grep)
    } else if peek.starts_with(".")
        || peek
            .lines()
            .take(10)
            .all(|l| l.starts_with("./") || l.starts_with('/'))
    {
        Some(Filter::Find)
    } else if peek.contains("├──") || peek.contains("└──") || peek.contains("│") {
        Some(Filter::Tree)
    } else if peek
        .lines()
        .take(5)
        .any(|l| l.starts_with("total ") || l.starts_with("-rw") || l.starts_with("drwx"))
    {
        Some(Filter::Ls)
    } else if peek.lines().count() > 50 && has_repetitive_blocks(peek) {
        Some(Filter::DedupLog)
    } else if content.lines().count() >= 60 {
        Some(Filter::SmartTruncate)
    } else {
        None
    }
}

/// Compress tool_result content using auto-detected filter. Returns the
/// compressed string, or the original if compression doesn't help.
///
/// **Fail-safe guarantees:**
/// - Never returns empty string (returns original if compression yields empty).
/// - Never returns a string longer than the input.
/// - If detected filter produces output > [`MAX_COMPRESSION_RATIO`] of input,
///   returns original.
pub fn compress(content: &str) -> String {
    if content.len() < 256 {
        // Too short to benefit from compression.
        return content.to_string();
    }

    match detect_filter(content) {
        Some(filter) => {
            let compressed = apply_filter(filter, content);
            // Fail-safe: never empty, never longer.
            if compressed.is_empty()
                || compressed.len() >= content.len()
                || (compressed.len() as f64 / content.len() as f64) > MAX_COMPRESSION_RATIO
            {
                content.to_string()
            } else {
                compressed
            }
        }
        None => content.to_string(),
    }
}

/// Compress all tool_result text blocks in a chat message's content array.
/// This is the main entry point for the proxy — it processes the request
/// body's messages, finding tool_result content blocks and compressing them.
///
/// Returns the number of bytes saved across all compressed blocks.
pub fn compress_tool_results(messages_json: &mut serde_json::Value) -> usize {
    let mut saved = 0;

    if let Some(messages) = messages_json
        .get_mut("messages")
        .and_then(|m| m.as_array_mut())
    {
        for msg in messages.iter_mut() {
            if msg.get("role").and_then(|r| r.as_str()) == Some("tool") {
                // Compress the content field (string or array of text blocks).
                if let Some(content) = msg.get_mut("content") {
                    if let Some(text) = content.as_str() {
                        let compressed = compress(text);
                        if compressed.len() < text.len() {
                            saved += text.len() - compressed.len();
                            *content = serde_json::Value::String(compressed);
                        }
                    } else if let Some(blocks) = content.as_array_mut() {
                        for block in blocks.iter_mut() {
                            if block.get("type").and_then(|t| t.as_str()) == Some("text") {
                                if let Some(text) = block.get_mut("text").and_then(|t| t.as_str()) {
                                    let text_owned = text.to_string();
                                    let compressed = compress(&text_owned);
                                    if compressed.len() < text_owned.len() {
                                        saved += text_owned.len() - compressed.len();
                                        block["text"] = serde_json::Value::String(compressed);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    saved
}

fn apply_filter(filter: Filter, content: &str) -> String {
    match filter {
        Filter::GitDiff => compress_git_diff(content),
        Filter::GitStatus => compress_git_status(content),
        Filter::Grep => compress_grep(content),
        Filter::Find => compress_find(content),
        Filter::Ls => compress_ls(content),
        Filter::Tree => compress_tree(content),
        Filter::DedupLog => compress_dedup_log(content),
        Filter::SmartTruncate => compress_smart_truncate(content),
        Filter::ReadNumbered => compress_read_numbered(content),
        Filter::SearchList => compress_search_list(content),
        Filter::BuildOutput => compress_build_output(content),
    }
}

// ---------------------------------------------------------------------------
// Filter implementations
// ---------------------------------------------------------------------------

/// Collapse unchanged context lines in git diff, keep only hunk headers and
/// changed lines (+/-). Limit context to 1 line around changes.
fn compress_git_diff(content: &str) -> String {
    let mut out = String::with_capacity(content.len() / 2);
    let mut last_was_context = false;
    let mut skipped_context = 0usize;

    for line in content.lines() {
        if line.starts_with("diff --git")
            || line.starts_with("+++")
            || line.starts_with("---")
            || line.starts_with("@@")
            || line.starts_with('+')
            || line.starts_with('-')
        {
            // Header or changed line — always keep, flush any skipped context.
            if skipped_context > 0 {
                out.push_str(&format!(
                    "  ... ({skipped_context} context lines omitted)\n"
                ));
                skipped_context = 0;
            }
            out.push_str(line);
            out.push('\n');
            last_was_context = false;
        } else if line.starts_with(' ') {
            // Context line — keep first one after a change, skip the rest.
            if !last_was_context {
                out.push_str(line);
                out.push('\n');
                last_was_context = true;
            } else {
                skipped_context += 1;
            }
        } else {
            out.push_str(line);
            out.push('\n');
            last_was_context = false;
        }
    }
    if skipped_context > 0 {
        out.push_str(&format!(
            "  ... ({skipped_context} context lines omitted)\n"
        ));
    }
    out
}

/// Collapse verbose untracked file listings in git status.
fn compress_git_status(content: &str) -> String {
    let mut out = String::with_capacity(content.len() / 2);
    // Whether we are inside the `Untracked files:` section, and how many
    // tab-indented untracked files we have seen so far within it. The old
    // logic only incremented the counter once it was already > 5, so it never
    // rose past the header's initial value and the collapse never fired.
    let mut in_untracked = false;
    let mut untracked_count = 0usize;

    for line in content.lines() {
        // A tab-indented line inside the section is an untracked file: keep the
        // first 5, count the rest for the summary.
        if in_untracked && line.starts_with('\t') {
            untracked_count += 1;
            if untracked_count <= 5 {
                out.push_str(line);
                out.push('\n');
            }
            continue;
        }

        // Any other line: the space-indented hint ("  (use \"git add ...\")")
        // stays part of the section; a blank or non-indented line ends it, so
        // flush the summary for the collapsed remainder first.
        if in_untracked && !line.starts_with(' ') {
            if untracked_count > 5 {
                out.push_str(&format!(
                    "\t... ({} more untracked files)\n",
                    untracked_count - 5
                ));
            }
            in_untracked = false;
            untracked_count = 0;
        }

        out.push_str(line);
        out.push('\n');
        if line == "Untracked files:" {
            in_untracked = true; // Start counting after the header.
            untracked_count = 0;
        }
    }

    // Flush if the input ended while still inside the untracked section.
    if in_untracked && untracked_count > 5 {
        out.push_str(&format!(
            "\t... ({} more untracked files)\n",
            untracked_count - 5
        ));
    }
    out
}

/// Deduplicate grep matches — keep first 5 matches per file, summarize rest.
fn compress_grep(content: &str) -> String {
    let mut out = String::with_capacity(content.len() / 2);
    let mut current_file = String::new();
    let mut matches_in_file = 0usize;
    let mut skipped = 0usize;

    for line in content.lines() {
        if let Some(file) = line.split(':').next() {
            if file != current_file {
                if skipped > 0 {
                    out.push_str(&format!(
                        "  ... ({skipped} more matches in {current_file})\n"
                    ));
                }
                current_file = file.to_string();
                matches_in_file = 0;
                skipped = 0;
            }
        }
        if matches_in_file < 5 {
            out.push_str(line);
            out.push('\n');
            matches_in_file += 1;
        } else {
            skipped += 1;
        }
    }
    if skipped > 0 {
        out.push_str(&format!(
            "  ... ({skipped} more matches in {current_file})\n"
        ));
    }
    out
}

/// Collapse deep directory listings — keep top 3 levels, summarize deeper.
fn compress_find(content: &str) -> String {
    let mut out = String::with_capacity(content.len() / 2);
    let mut deep_count = 0usize;

    for line in content.lines() {
        let depth = line.matches('/').count();
        if depth <= 3 {
            if deep_count > 0 {
                out.push_str(&format!("... ({deep_count} deeper paths omitted)\n"));
                deep_count = 0;
            }
            out.push_str(line);
            out.push('\n');
        } else {
            deep_count += 1;
        }
    }
    if deep_count > 0 {
        out.push_str(&format!("... ({deep_count} deeper paths omitted)\n"));
    }
    out
}

/// Collapse long directory listings — keep first 20 entries, summarize rest.
fn compress_ls(content: &str) -> String {
    let lines: Vec<&str> = content.lines().collect();
    if lines.len() <= 25 {
        return content.to_string();
    }
    let mut out = String::with_capacity(content.len() / 2);
    for line in &lines[..20] {
        out.push_str(line);
        out.push('\n');
    }
    out.push_str(&format!(
        "... ({} more entries omitted)\n",
        lines.len() - 20
    ));
    out
}

/// Prune deep subtrees — keep top 4 levels, collapse deeper.
fn compress_tree(content: &str) -> String {
    let mut out = String::with_capacity(content.len() / 2);
    let mut skipped = 0usize;

    for line in content.lines() {
        // Count indentation depth (tree uses 4-space or box-drawing indentation).
        let depth = line
            .chars()
            .take_while(|c| *c == ' ' || *c == '│' || *c == '├' || *c == '└' || *c == '─')
            .count()
            / 4;
        if depth <= 4 || line.contains("directories") || line.contains("files") {
            if skipped > 0 {
                out.push_str(&format!("    ... ({skipped} items pruned)\n"));
                skipped = 0;
            }
            out.push_str(line);
            out.push('\n');
        } else {
            skipped += 1;
        }
    }
    if skipped > 0 {
        out.push_str(&format!("    ... ({skipped} items pruned)\n"));
    }
    out
}

/// Deduplicate consecutive identical or near-identical log blocks.
fn compress_dedup_log(content: &str) -> String {
    let lines: Vec<&str> = content.lines().collect();
    let mut out = String::with_capacity(content.len() / 2);
    let mut last_line = "";
    let mut repeat_count = 0usize;

    for line in &lines {
        if *line == last_line {
            repeat_count += 1;
        } else {
            if repeat_count > 2 {
                out.push_str(&format!("  ... (repeated {repeat_count} times)\n"));
            } else {
                for _ in 0..repeat_count {
                    out.push_str(last_line);
                    out.push('\n');
                }
            }
            out.push_str(line);
            out.push('\n');
            last_line = line;
            repeat_count = 0;
        }
    }
    // Final flush: mirror the in-loop branch so a run of 1–2 trailing repeats
    // is re-emitted instead of being silently dropped (only the > 2 case was
    // handled before, so a log ending in "OK\nOK" lost the duplicate).
    if repeat_count > 2 {
        out.push_str(&format!("  ... (repeated {repeat_count} times)\n"));
    } else {
        for _ in 0..repeat_count {
            out.push_str(last_line);
            out.push('\n');
        }
    }
    out
}

/// Keep first 30 + last 20 lines of long output, summarize middle.
fn compress_smart_truncate(content: &str) -> String {
    let lines: Vec<&str> = content.lines().collect();
    if lines.len() <= 60 {
        return content.to_string();
    }
    let mut out = String::with_capacity(content.len() / 3);
    for line in &lines[..30] {
        out.push_str(line);
        out.push('\n');
    }
    out.push_str(&format!(
        "\n... ({} lines omitted — {} total lines)\n\n",
        lines.len() - 50,
        lines.len()
    ));
    for line in &lines[lines.len() - 20..] {
        out.push_str(line);
        out.push('\n');
    }
    out
}

/// Collapse unchanged regions in numbered file content — keep first 10 +
/// last 10 lines, summarize middle.
fn compress_read_numbered(content: &str) -> String {
    let lines: Vec<&str> = content.lines().collect();
    if lines.len() <= 30 {
        return content.to_string();
    }
    let mut out = String::with_capacity(content.len() / 3);
    for line in &lines[..10] {
        out.push_str(line);
        out.push('\n');
    }
    out.push_str(&format!("  ... ({} lines omitted)\n", lines.len() - 20));
    for line in &lines[lines.len() - 10..] {
        out.push_str(line);
        out.push('\n');
    }
    out
}

/// Collapse search result lists — keep summary + top 10 results.
fn compress_search_list(content: &str) -> String {
    let lines: Vec<&str> = content.lines().collect();
    if lines.len() <= 15 {
        return content.to_string();
    }
    let mut out = String::with_capacity(content.len() / 3);
    for line in &lines[..12] {
        out.push_str(line);
        out.push('\n');
    }
    out.push_str(&format!(
        "... ({} more results omitted)\n",
        lines.len() - 12
    ));
    out
}

/// Collapse per-file compilation lines, keep errors and warnings.
fn compress_build_output(content: &str) -> String {
    let mut out = String::with_capacity(content.len() / 2);
    let mut compiling_count = 0usize;

    for line in content.lines() {
        if line.starts_with("   Compiling ") || line.starts_with("   Downloading ") {
            compiling_count += 1;
        } else {
            if compiling_count > 3 {
                out.push_str(&format!("   ... ({compiling_count} crates compiled)\n"));
            } else {
                for _ in 0..compiling_count {
                    out.push_str("   Compiling ...\n");
                }
            }
            compiling_count = 0;
            out.push_str(line);
            out.push('\n');
        }
    }
    if compiling_count > 3 {
        out.push_str(&format!("   ... ({compiling_count} crates compiled)\n"));
    }
    out
}

/// Check if content has repetitive blocks (consecutive similar lines).
fn has_repetitive_blocks(peek: &str) -> bool {
    let lines: Vec<&str> = peek.lines().collect();
    if lines.len() < 10 {
        return false;
    }
    let mut repeats = 0;
    for i in 1..lines.len() {
        if lines[i] == lines[i - 1] {
            repeats += 1;
        }
    }
    repeats > lines.len() / 4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_content_is_not_compressed() {
        let input = "Hello world";
        assert_eq!(compress(input), input);
    }

    #[test]
    fn detects_git_diff() {
        let content = "diff --git a/foo.rs b/foo.rs\nindex abc..def 100644\n--- a/foo.rs\n+++ b/foo.rs\n@@ -1,3 +1,4 @@\n fn main() {\n+    println!(\"hello\");\n }";
        assert_eq!(detect_filter(content), Some(Filter::GitDiff));
    }

    #[test]
    fn detects_git_status() {
        let content = "On branch main\nYour branch is up to date.\n\nChanges not staged for commit:\n  modified: foo.rs";
        assert_eq!(detect_filter(content), Some(Filter::GitStatus));
    }

    #[test]
    fn detects_build_output() {
        let content = "   Compiling serde v1.0.228\n   Compiling tokio v1.52.3\n   Compiling axum v0.8.0\nerror[E0308]: mismatched types";
        assert_eq!(detect_filter(content), Some(Filter::BuildOutput));
    }

    #[test]
    fn detects_tree_output() {
        let content = "src/\n├── main.rs\n├── lib.rs\n├── router/\n│   ├── mod.rs\n│   └── strategy.rs\n└── types/\n    └── mod.rs\n\n5 directories, 7 files";
        assert_eq!(detect_filter(content), Some(Filter::Tree));
    }

    #[test]
    fn detects_ls_output() {
        let content = "total 48\ndrwxr-xr-x  12 user staff 384 Jun 29 10:00 .\ndrwxr-xr-x   5 user staff 160 Jun 28 09:00 ..\n-rw-r--r--   1 user staff 220 Jun 29 10:00 Cargo.toml\n-rw-r--r--   1 user staff 8192 Jun 29 10:00 Cargo.lock";
        assert_eq!(detect_filter(content), Some(Filter::Ls));
    }

    #[test]
    fn git_diff_compression_removes_context() {
        let mut diff = String::from(
            "diff --git a/foo.rs b/foo.rs\n--- a/foo.rs\n+++ b/foo.rs\n@@ -1,20 +1,5 @@\n",
        );
        for i in 1..=15 {
            diff.push_str(&format!(" context line {i}\n"));
        }
        diff.push_str("+new line\n");
        let compressed = compress(&diff);
        assert!(
            compressed.len() < diff.len(),
            "compression should shrink diff"
        );
        assert!(compressed.contains("+new line"));
        assert!(compressed.contains("diff --git"));
    }

    #[test]
    fn smart_truncate_keeps_first_and_last() {
        let mut content = String::new();
        for i in 0..100 {
            content.push_str(&format!("line {i}\n"));
        }
        let compressed = compress(&content);
        assert!(compressed.contains("line 0"));
        assert!(compressed.contains("line 99"));
        assert!(compressed.contains("omitted"));
        assert!(compressed.len() < content.len());
    }

    #[test]
    fn build_output_collapses_compiling() {
        let mut content = String::new();
        for i in 0..20 {
            content.push_str(&format!("   Compiling crate_{i} v1.0.0\n"));
        }
        content.push_str("error[E0308]: mismatched types\n  --> src/main.rs:5:10\n");
        let compressed = compress(&content);
        assert!(compressed.contains("crates compiled"));
        assert!(compressed.contains("error[E0308]"));
        assert!(compressed.len() < content.len());
    }

    #[test]
    fn never_returns_empty() {
        let content = "x".repeat(300);
        let compressed = compress(&content);
        assert!(!compressed.is_empty());
    }

    #[test]
    fn never_grows_input() {
        let content = "This is a short tool result that should not be compressed at all because it is under 256 bytes.";
        let compressed = compress(content);
        assert!(compressed.len() <= content.len());
    }

    #[test]
    fn compress_tool_results_saves_bytes() {
        let mut body = serde_json::json!({
            "messages": [
                {"role": "user", "content": "run git diff"},
                {"role": "tool", "content": generate_large_diff()}
            ]
        });
        let saved = compress_tool_results(&mut body);
        assert!(saved > 0, "should save bytes on a large diff");
    }

    #[test]
    fn detect_filter_survives_multibyte_char_at_peek_boundary() {
        // 1023 ASCII bytes then a 3-byte UTF-8 char ('।', U+0964) whose bytes
        // occupy 1023..1026 — so byte offset PEEK_BYTES (1024) lands mid-char.
        // A raw `&content[..PEEK_BYTES]` byte slice previously panicked here
        // ("byte index 1024 is not a char boundary") on the chat hot path.
        let mut content = "a".repeat(PEEK_BYTES - 1);
        content.push('।');
        content.push_str(&"b".repeat(300));
        assert!(content.len() > PEEK_BYTES);
        assert!(
            !content.is_char_boundary(PEEK_BYTES),
            "test precondition: byte 1024 must be inside a multi-byte char"
        );

        // Must not panic — either via detect_filter directly or the compress()
        // entry point the proxy calls on every tool_result.
        let _ = detect_filter(&content);
        let compressed = compress(&content);
        assert!(!compressed.is_empty());
    }

    #[test]
    fn git_status_collapses_many_untracked_files() {
        let mut content = String::from(
            "On branch main\nUntracked files:\n  (use \"git add <file>...\" to include in what will be committed)\n",
        );
        for i in 0..30 {
            content.push_str(&format!("\tsrc/generated/module_{i:03}.rs\n"));
        }
        assert_eq!(detect_filter(&content), Some(Filter::GitStatus));

        let compressed = compress_git_status(&content);
        assert!(
            compressed.len() < content.len(),
            "collapsing 30 untracked files should shrink the output"
        );
        // First 5 kept verbatim, the remaining 25 summarized.
        assert!(compressed.contains("src/generated/module_000.rs"));
        assert!(compressed.contains("src/generated/module_004.rs"));
        assert!(!compressed.contains("src/generated/module_005.rs"));
        assert!(compressed.contains("25 more untracked files"));
        // The public entry point applies it (ratio guard passes).
        assert!(compress(&content).len() < content.len());
    }

    #[test]
    fn dedup_log_preserves_trailing_repeats() {
        // A repetitive log ending in a run of 3 identical lines: the loop emits
        // the first, leaving repeat_count == 2 at exit. The old final flush only
        // handled repeat_count > 2, silently dropping these two trailing lines.
        let mut content = String::new();
        for i in 0..50 {
            content.push_str(&format!("event {i}\n"));
        }
        content.push_str("connection closed\n");
        content.push_str("connection closed\n");
        content.push_str("connection closed\n");

        let compressed = compress_dedup_log(&content);
        assert_eq!(
            compressed.matches("connection closed").count(),
            3,
            "all three trailing repeats must survive"
        );
    }

    fn generate_large_diff() -> String {
        let mut diff = String::from(
            "diff --git a/big.rs b/big.rs\n--- a/big.rs\n+++ b/big.rs\n@@ -1,50 +1,5 @@\n",
        );
        for i in 1..=40 {
            diff.push_str(&format!(" unchanged line {i}\n"));
        }
        diff.push_str("-removed line\n+added line\n");
        diff
    }
}
