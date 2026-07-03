# Security Policy

Routeplane CE is an AI gateway: it sits in front of your provider API keys and your traffic.
We treat security reports as the highest-priority class of issue.

## Reporting a vulnerability

Email **security@routeplane.ai**.

Please include:

- A description of the issue and the component affected (file/endpoint if known).
- Reproduction steps or a proof of concept.
- The version or image digest you tested (`routeplane --version`, or the `org.opencontainers`
  labels on the image).
- Your assessment of impact, if you have one.

You can also use GitHub's private vulnerability reporting on this repository if you prefer not
to use email. Please do **not** open a public issue for anything you believe is a vulnerability.

## What to expect (coordinated disclosure)

- **Acknowledgement within 72 hours** of your report reaching us.
- An initial assessment (accepted / needs-more-info / not-a-vulnerability) within 7 days.
- We ask for a standard **90-day coordinated disclosure window** while we develop, test, and
  release a fix. We will agree on a disclosure date with you and credit you in the release notes
  (or keep you anonymous — your choice).
- If we ship a fix sooner, we will coordinate earlier disclosure with you rather than sitting on
  the window.

## Scope

**In scope:** this repository — the Routeplane CE gateway binary, its provider adapters, auth
(virtual keys), routing, caching, rate/spend limits, PII masking, RTK compression, the Docker
image `ghcr.io/routeplane-core/routeplane-ce`, the docker-compose deployment path, and
the CI workflows in this repository.

**Out of scope:** the commercial/enterprise platform (report those to security@routeplane.ai as
well, but they are handled outside this repo's process), third-party LLM providers, and issues
requiring a compromised host or a maliciously modified config file.

## Bug bounty

We do **not** run a paid bug bounty program yet. We are a small team and we would rather be
honest about that than promise rewards we cannot pay consistently. We do credit reporters in
release notes and in a HALL_OF_FAME section of this file, and we fix confirmed reports fast.

## Supported versions

Security fixes land on the latest minor release line. We do not backport to older tags during
the 0.x series — upgrade to the newest release to receive fixes.

## Verifying what you run

Do not take our word for what is in the image — verify the artifact:

- Images are **cosign-signed** (keyless, GitHub OIDC). Verification instructions, including the
  exact identity to check, are in the README's "Verify the artifact" section.
- Every release attaches an **SPDX SBOM** generated in public CI, plus the image digest and the
  cosign bundle.
- CE has **no telemetry and no phone-home** — the only outbound connections the gateway makes
  are to the LLM providers you configure. This is verifiable in the source in this repository.
