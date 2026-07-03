# Contributing to Routeplane

Thanks for your interest in contributing. Routeplane CE is maintained by the Routeplane team
(**maintainers@routeplane.ai**) and we welcome issues, discussions, and pull requests.

## The golden path: clone, build, test

Prerequisites:

- **Rust 1.88** — the repo pins the toolchain in `rust-toolchain.toml`, so a stock
  [rustup](https://rustup.rs) install resolves it automatically.
- On Linux you also need `pkg-config` and OpenSSL headers:

```bash
sudo apt-get install -y pkg-config libssl-dev   # Debian/Ubuntu
```

macOS needs no extra packages (TLS comes from Security.framework).

Then:

```bash
git clone https://github.com/routeplane-core/routeplane-ce.git
cd routeplane-ce
cargo build                    # workspace build
cargo test                     # full test suite — unit + wiremock adapter integration tests
cargo clippy --all-targets -- -D warnings   # lint; warnings are denied in CI
cargo fmt --all -- --check     # formatting gate
```

All tests run offline — provider adapter tests use wiremock, not real provider APIs. You do not
need any API keys to build or test.

To run the gateway locally, copy `.env.example` to `.env`, copy `configs/keys.example.json` to
`configs/keys.json`, and:

```bash
cargo run -p routeplane
```

## What CI runs on your PR

Pull requests from forks run the same required checks as maintainer PRs, with no secrets
exposed to fork workflows:

- `cargo fmt --all -- --check`
- `cargo clippy --all-targets -- -D warnings`
- `cargo test` (the full workspace suite)
- `cargo deny check` (licenses + advisories)
- A secret scan (gitleaks) and an identity/content guard over the diff

If those pass and a maintainer approves, the PR is merged by squash. Image builds, signing,
releases, and deployment promotion are **maintainer-run** and never triggered by fork PRs —
you do not need to worry about them.

## Commit and PR conventions

- **Conventional Commits** for PR titles (they become the squash commit):
  `feat: ...`, `fix: ...`, `docs: ...`, `chore: ...`, with an optional single lowercase scope,
  e.g. `fix(adapters): handle empty SSE keepalive lines`.
- Keep PRs focused. Small, reviewable changes merge fast; grab-bag PRs stall.
- **Tests are required** for behavior changes. Bug fixes should include a test that fails
  without the fix. Adapter changes should extend the wiremock integration tests.
- **No AI-attribution trailers.** Do not add `Co-Authored-By` trailers naming AI assistants, or
  "generated with ..." footers, to commits or PR descriptions. Use whatever tools you like to
  write the code — you are the author, and you are responsible for what you submit.

## Licensing: no CLA, no DCO sign-off

Routeplane CE is licensed under **Apache-2.0**. We deliberately require **no CLA and no DCO
`Signed-off-by` line**: per the standard inbound = outbound norm (and Apache-2.0 §5), by
submitting a contribution you agree it is licensed under Apache-2.0, the same license you
received the project under. We chose this over a DCO because it is the lowest-friction honest
option: a sign-off line adds ceremony without adding legal substance for a project where the
license already covers contributions. If we ever needed to change this, it would only apply to
future contributions and we would announce it prominently.

## What maintainers handle

- Triage, labels, and milestones.
- Reviews and merges (squash; PR title becomes the commit).
- Releases: version tags, changelog, the signed `ghcr.io/routeplane-core/routeplane-ce` image,
  and SBOM and signature attachment.
- Security reports (see [SECURITY.md](SECURITY.md)) and Code of Conduct enforcement
  (see [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)).

## Where to start

- Issues labeled [`good first issue`](https://github.com/routeplane-core/routeplane-ce/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
  are scoped to be completable without deep codebase knowledge.
- Ask questions in
  [GitHub Discussions](https://github.com/routeplane-core/routeplane-ce/discussions) — Q&A is
  the right category; we answer there so future readers can find it.
