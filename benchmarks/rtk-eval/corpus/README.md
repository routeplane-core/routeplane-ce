# RTK eval corpus — provenance and hygiene

`traces.jsonl` is the frozen, committed corpus the RTK trace-replay eval
(`cargo run --release -p rtk-eval`) scores against. One JSON object per line:

```json
{
  "id": "read-rtk-lib",
  "category": "read",
  "source": "real-command" | "synthetic-generator",
  "description": "…",
  "messages": [ { "role": "system|user|assistant|tool", "content": "…", … } ]
}
```

Messages are OpenAI chat-completion shaped: `tool`-role messages carry
`tool_call_id` and either string content (`MessageContent::Text`) or an
array-of-parts content (`MessageContent::Parts` — one trace,
`read-types-parts-form`, deliberately uses the parts form to exercise that
wire shape).

## Provenance

- **`source: "real-command"`** (22 of 24 traces): tool payloads are the real
  stdout/stderr of the named command, run against the Routeplane repository
  itself at corpus-capture time (see `generate.py` — the exact command is
  recorded in each trace's `tool_calls[0].function.arguments`). Categories:
  `git diff` ranges over recent first-party commits, `git status`, `cat -n`
  file reads (the Read-tool shape), `grep -rn`, `find`, `ls -la`, `tree -L 3`,
  from-scratch `cargo build` / `cargo build --release` / `cargo test` logs of
  the benchmarks workspace, and a `head -200 Cargo.lock` generic-text read.
  Surrounding conversation text (system prompt, user asks, assistant replies)
  is hand-written by the Routeplane team.
- **`source: "synthetic-generator"`** (2 of 24 traces): repetitive service-log
  payloads produced by the labeled generator functions in `generate.py`
  (`synth_health_log`, `synth_retry_log`). They exist because a realistic
  repetitive-log capture would require harvesting production logs; they are
  clearly labeled so anyone can discount them — the aggregate barely moves
  without them.

No filter-flattering curation was applied: the corpus deliberately includes
shapes RTK does **not** compress (sub-256-byte outputs, an insertion-only
diff, already-shallow `tree` output, plain `grep -rn` content matches), and
they score 0% in RESULTS.md.

## Licensing

All `real-command` payloads are outputs of commands run against the
Routeplane repository's own first-party code (GitHub org `routeplane-core`).
No third-party repository content is embedded. The corpus is licensed under
the same terms as this repository.

## Identity hygiene (launch requirement: identity-clean)

`generate.py` rewrites machine-identifying strings before anything is
written: absolute repo paths → `/workspace/routeplane`, `$HOME` →
`/home/dev`, the local username → `dev` (this also covers the owner column of
`ls -la` output). Generation **fails** if any forbidden token or email-like
string survives. The same gate is enforced forever after by
`cargo test -p rtk-eval` (`corpus_is_identity_clean`), so CI re-checks the
committed file on every run. Authorship-carrying git commands (`git log`,
`git blame`, `git shortlog`) are never used.

## Freezing and regeneration

The committed `traces.jsonl` is a **frozen snapshot** — several inputs are
repo-state-dependent (git status/diff ranges, build timing lines), so
re-running `python3 generate.py` produces an *equivalent* corpus, not a
byte-identical one. RESULTS.md pins the sha256 of the exact corpus it was
produced from. If you regenerate, re-run the eval and commit both files
together; published numbers always refer to the committed pair.

`raw/` holds cached cargo logs used during generation; it is gitignored
(intermediate, reproducible).
