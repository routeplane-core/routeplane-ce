# Routeplane task runner — the golden-path command surface
# (branching-and-devex.md §10, devsecops-pipeline.md §10.4).
#
# Mirrors the .github/workflows/ci.yml gates — common-actions/rust-quality
# (fmt-check -> clippy -D warnings -> cargo test --all) + cargo-audit-deny
# (cargo-deny licenses/bans/sources gate + cargo-audit). `just ci` = everything
# blocking CI runs, locally, before you push. Coverage (rust-coverage) is
# report-only in CI and deliberately excluded here.
#
# NOTE: just uses the comment line directly above a recipe as its doc string.

# List available recipes.
default:
    @just --list

# gitleaks pin — MUST match .devcontainer/Dockerfile (and the sha256 there).
gitleaks_version := "8.30.1"
gitleaks_sha256 := "551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb"

# cargo-binstall pin — MUST match .devcontainer/Dockerfile (CARGO_BINSTALL_*).
binstall_version := "1.20.0"
binstall_sha256 := "4de7b98d09026101d7b1788c6d92f0e28544741a3f4e393d46d2b00677cbbaa2"

# One-time setup: toolchain components + the CI tool set (incl. taplo-cli for
# the taplo pre-commit hook, which is language: system) + gitleaks (the §8.1
# local secret gate, also language: system) + BOTH hook stages (cargo-clippy
# runs at pre-push, so plain `pre-commit install` would silently skip it).
bootstrap: _install-gitleaks _install-binstall
    rustup component add rustfmt clippy
    cargo binstall -y cargo-audit@0.22.2 cargo-deny@0.19.8 cargo-nextest@0.9.137 taplo-cli@0.10.0
    pre-commit install --install-hooks -t pre-commit -t pre-push

# Pinned gitleaks for non-devcontainer users (devcontainer preinstalls it).
# linux x86_64 only — other platforms: install v{{gitleaks_version}} manually
# from https://github.com/gitleaks/gitleaks/releases and put it on PATH.
_install-gitleaks:
    #!/usr/bin/env bash
    set -euo pipefail
    if command -v gitleaks >/dev/null 2>&1; then
        echo "gitleaks already installed: $(gitleaks version 2>/dev/null || true)"
        exit 0
    fi
    if [ "$(uname -s)/$(uname -m)" != "Linux/x86_64" ]; then
        echo "gitleaks: unsupported platform for auto-install — install v{{gitleaks_version}} manually (https://github.com/gitleaks/gitleaks/releases)" >&2
        exit 1
    fi
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT
    curl -fsSL -o "$tmp/gitleaks.tar.gz" \
        "https://github.com/gitleaks/gitleaks/releases/download/v{{gitleaks_version}}/gitleaks_{{gitleaks_version}}_linux_x64.tar.gz"
    echo "{{gitleaks_sha256}}  $tmp/gitleaks.tar.gz" | sha256sum -c -
    mkdir -p "$HOME/.local/bin"
    tar -xz -C "$HOME/.local/bin" -f "$tmp/gitleaks.tar.gz" gitleaks
    echo "gitleaks v{{gitleaks_version}} installed to $HOME/.local/bin (ensure it is on PATH)"

# Pinned cargo-binstall for non-devcontainer users — prebuilt tool installs (no
# compile-from-source wait), matching the devcontainer's approach. The pinned
# tool versions in `bootstrap` then resolve to prebuilt artifacts. linux x86_64
# only — other platforms: use the devcontainer, or install cargo-binstall manually
# from https://github.com/cargo-bins/cargo-binstall/releases.
_install-binstall:
    #!/usr/bin/env bash
    set -euo pipefail
    if command -v cargo-binstall >/dev/null 2>&1; then
        echo "cargo-binstall already installed: $(cargo-binstall -V 2>/dev/null || true)"
        exit 0
    fi
    if [ "$(uname -s)/$(uname -m)" != "Linux/x86_64" ]; then
        echo "cargo-binstall: unsupported platform for auto-install — install v{{binstall_version}} manually (https://github.com/cargo-bins/cargo-binstall/releases) or use the devcontainer" >&2
        exit 1
    fi
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT
    curl -fsSL -o "$tmp/cargo-binstall.tgz" \
        "https://github.com/cargo-bins/cargo-binstall/releases/download/v{{binstall_version}}/cargo-binstall-x86_64-unknown-linux-gnu.tgz"
    echo "{{binstall_sha256}}  $tmp/cargo-binstall.tgz" | sha256sum -c -
    mkdir -p "$HOME/.cargo/bin"
    tar -xz -C "$HOME/.cargo/bin" -f "$tmp/cargo-binstall.tgz" cargo-binstall
    echo "cargo-binstall v{{binstall_version}} installed to $HOME/.cargo/bin (ensure it is on PATH)"

# Production build (matches the Dockerfile's `cargo build --release`).
build:
    cargo build --release

# Run the data-plane gateway locally on PORT (default 8080). Needs a .env with
# provider keys (OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY) loaded via
# dotenvy — see CONTRIBUTING.md. Prefix `RUST_LOG=routeplane=debug` for verbose logs.
run:
    cargo run -p routeplane

# Deliberate deviation from devsecops-pipeline.md §10.4's nextest suggestion:
# CI's rust-quality composite runs `cargo test --all`, and local/CI parity
# wins. Use `just nextest` for the faster local UX.
#
# Exactly what CI's rust-quality composite runs.
test:
    cargo test --all

# Optional faster local test UX (not the CI command — see `just test`).
nextest:
    cargo nextest run --all

# Warnings are also denied via [workspace.lints] in Cargo.toml, so the -D
# flag below is belt-and-braces (same as CI).
#
# fmt-check + clippy, identical to the rust-quality composite.
lint:
    cargo fmt --all -- --check
    cargo clippy --all-targets -- -D warnings

# Apply formatting (the fixing counterpart of `just lint`).
fmt:
    cargo fmt --all

# cargo-deny license/ban/source policy is the blocking gate; cargo-audit
# blocks on advisories with an available fix (no-fix advisories are
# report-only in CI, per devsecops-pipeline.md §6.1).
#
# Mirrors the cargo-audit-deny composite (the deps-audit check).
audit:
    cargo deny check licenses bans sources
    cargo audit

# Local secret scan — mirrors the CI `secret-scan` required check. Scans COMMITTED
# history (like CI's gitleaks-scan), honoring .gitleaks.toml's content regexes;
# never touches the gitignored .env (real keys), unlike a filesystem scan. The
# pre-commit hook only scans STAGED changes; this is the pre-push gate.
# Needs gitleaks on PATH (`just bootstrap` installs it).
secrets:
    gitleaks git . --redact --no-banner

# The full blocking CI gate set (quality + deps-audit + secret-scan), locally,
# before you push — mirrors every required CI check.
ci: lint test audit secrets

# Build the production image locally (same Dockerfile CI builds from).
docker:
    docker build -t routeplane:latest .

# Guardrail eval gate (devsecops-pipeline.md §7) — scores the detector library
# against the CODEOWNER-gated corpus and asserts the manifest recall/precision
# floors. This IS live + enforced: guardrail_eval.rs is auto-discovered by
# `cargo test --all`, so it already runs in `just test` / `just ci` and the CI
# `quality` required check (and the perf-quality-gate hard gate). This recipe
# surfaces the per-category recall/precision scores locally.
eval:
    cargo test -p routeplane --test guardrail_eval -- --nocapture

# Black-box acceptance suite (ADR-049 / PRD-037) against a DEPLOYED gateway.
# No-ops (loud SKIP) unless ACCEPTANCE_BASE_URL is set, so it's safe in `just test`.
#   ACCEPTANCE_BASE_URL=https://<dev-fqdn> ACCEPTANCE_API_KEY=rp_… just acceptance
# Add ACCEPTANCE_COMPLETIONS=1 (needs a working provider on the target) for the
# real-200 + SSE tiers; ACCEPTANCE_EDGE=1 on a Cloudflare-fronted env for the edge tier.
acceptance:
    cargo test -p routeplane --test acceptance -- --nocapture

# ADR-084 build-once + decoupled promotion (build+sign ONCE on dev; FF-promote the
# SAME signed digest — no rebuild). A DELIBERATE action: feature work on dev never
# advances staging/prod on its own. Triggers promote.yml (workflow_dispatch), which
# fast-forwards the lower env-branch onto the higher one AS the routeplane-cd-dispatch
# App; ci.yml's `promote` job then re-resolves the dev-built digest by the (preserved)
# commit SHA and dispatches a deploy of THAT digest (promote-image cosign-copies it
# into the target ACR). The GitHub Environment reviewer (your approval click) gates
# staging/prod in cd.yml — NO wait-timer soak (ADR-084).
#   just promote staging   # dev     -> staging
#   just promote prod      # staging -> main (prod)
promote env:
    #!/usr/bin/env bash
    set -euo pipefail
    case "{{env}}" in
      staging|prod) ;;
      *) echo "usage: just promote <staging|prod>" >&2; exit 2 ;;
    esac
    echo "Dispatching fast-forward promotion → {{env}} (runs as the CD App) …"
    gh workflow run promote.yml --ref dev -f environment="{{env}}"
    echo "Triggered. Watch it:"
    echo "  gh run watch \$(gh run list --workflow=promote.yml -L1 --json databaseId --jq '.[0].databaseId')"

# ADR-084 "what's deployed where" — the env-branch refs ARE the answer: each
# branch points at the dev-built+signed commit whose digest is live on that env
# (build-once: the SAME digest, imported into that env's ACR). Reads dev/staging/main.
deployed:
    #!/usr/bin/env bash
    set -euo pipefail
    repo="$(gh repo view --json nameWithOwner -q .nameWithOwner)"
    printf '%-9s %-14s %s\n' ENV SHA SUBJECT
    for b in dev staging main; do
      if val="$(gh api "repos/${repo}/branches/${b}" \
            --jq '.commit.sha[0:12] + "\t" + (.commit.commit.message | split("\n")[0])' 2>/dev/null)"; then
        printf '%-9s %-14s %s\n' "$b" "${val%%$'\t'*}" "${val#*$'\t'}"
      else
        printf '%-9s %-14s %s\n' "$b" "-" "(branch not created)"
      fi
    done

# ADR-084 fast rollback (devex-gap-audit no-fast-rollback) — shift 100% traffic to
# a prior good revision, NO rebuild/re-verify. List revisions first, then roll back:
#   just revisions ca-routeplane-pool-std rg-routeplane-prod
#   just rollback prod ca-routeplane-pool-std rg-routeplane-prod <revision>
# (assumes multi-revision mode = prod; single-revision envs use the cd.yml hotfix
# redeploy of the prior digest instead). Needs az + gh logged in.
revisions app rg:
    az containerapp revision list -n {{app}} -g {{rg}} -o table

# Triggers infrastructure-live/.github/workflows/rollback.yml. Approve the env's
# GitHub Environment gate (prod = your reviewer click, no soak) to apply.
rollback env app rg revision:
    #!/usr/bin/env bash
    set -euo pipefail
    case "{{env}}" in
      dev|staging|prod|dedicated) ;;
      *) echo "usage: just rollback <dev|staging|prod|dedicated> <app> <rg> <revision>" >&2; exit 2 ;;
    esac
    echo "Rolling back {{app}} ({{env}}) → {{revision}} (100% traffic, no rebuild) …"
    gh workflow run rollback.yml --repo routeplane-core/infrastructure-live \
      -f environment="{{env}}" -f app="{{app}}" -f resource_group="{{rg}}" -f revision="{{revision}}"
    echo "Triggered. Approve the {{env}} Environment gate, then watch:"
    echo "  gh run watch \$(gh run list --repo routeplane-core/infrastructure-live --workflow=rollback.yml -L1 --json databaseId --jq '.[0].databaseId')"

# DORA metrics from existing free signals — gh cd.yml runs, ZERO standing cost
# (devex-gap-audit dora-metrics-unmeasured). Sources infra-live (where env-bound
# deploys are recorded); `repository_dispatch` isolates the app CD spine from
# Terraform applies; a prod go-live = a run whose `shift-prod-traffic` job
# succeeded. Change-fail rate is the rollback-based lower bound (manual dispatches
# are reported separately — not every manual deploy is a failure). MTTR is n/a
# until the first rollback; lead-time needs dev→prod digest correlation (a
# refinement). Needs gh logged in. Usage:
#   just dora        # last 30 days
#   just dora 90     # last 90 days
dora window='30':
    #!/usr/bin/env bash
    set -euo pipefail
    repo="routeplane-core/infrastructure-live"
    since="$(date -u -d "-{{window}} days" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null \
             || date -u -v-{{window}}d +%Y-%m-%dT%H:%M:%SZ)"
    echo "DORA — last {{window}}d (since $since) · source: $repo cd.yml app-deploy runs"
    echo
    tmp="$(mktemp)"; trap 'rm -f "$tmp"' EXIT
    gh run list --repo "$repo" --workflow cd.yml --event repository_dispatch \
        --status success --created ">$since" -L 200 \
        --json databaseId,headSha,updatedAt > "$tmp"
    total=$(jq 'length' "$tmp")
    if [ "$total" -eq 0 ]; then echo "no successful app deploys in window"; exit 0; fi
    prod=0
    while read -r id; do
      [ -z "$id" ] && continue
      is_prod=$(gh run view "$id" --repo "$repo" --json jobs \
                  --jq 'any(.jobs[]; (.name|test("shift-prod-traffic")) and .conclusion=="success")' \
                  2>/dev/null || echo false)
      [ "$is_prod" = "true" ] && prod=$((prod+1))
    done < <(jq -r '.[].databaseId' "$tmp")
    rb=$(gh run list --repo "$repo" --workflow rollback.yml --created ">$since" \
           -L 100 --json databaseId --jq 'length' 2>/dev/null || echo 0)
    hf=$(gh run list --repo "$repo" --workflow cd.yml --event workflow_dispatch \
           --created ">$since" -L 100 --json databaseId --jq 'length' 2>/dev/null || echo 0)
    echo "Deployment frequency : $total app deploys — $prod prod go-lives / {{window}}d"
    if [ "$prod" -gt 0 ]; then
      cfr=$(awk "BEGIN{printf \"%.0f\", $rb*100/$prod}")
      echo "Change-fail rate     : ${cfr}%  ($rb rollback / $prod prod go-lives) — rollback-based lower bound"
    else
      echo "Change-fail rate     : n/a (no prod go-lives in window)"
    fi
    if [ "$rb" -eq 0 ]; then
      echo "MTTR                 : n/a — no restores recorded (rollback.yml never fired)"
    else
      echo "MTTR                 : $rb rollback(s) in window — median restore time (refine with >1 sample)"
    fi
    echo "Manual cd dispatches : $hf in window (informational — manual deploys/hotfixes, not all failures)"
    echo "Lead time            : approx — needs dev→prod digest correlation (tracked refinement)"
