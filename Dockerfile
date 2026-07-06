# Routeplane Community Edition image.
#
# cargo-chef caches the dependency compile as its own image layer so code-only
# rebuilds are fast. Built on the official rust:1.88 base (no third-party base
# image). A plain `docker build .` produces the CE gateway image.
FROM rust:1.88-slim-bookworm AS chef
RUN apt-get update \
    && apt-get install -y pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/* \
    && cargo install cargo-chef --locked --version 0.1.71
WORKDIR /usr/src/app

# Planner: derive the dependency recipe (no compile) from the manifests.
FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

# Builder: cook the deps (cached unless Cargo.toml/Cargo.lock change), then build
# the gateway binary.
FROM chef AS builder
COPY --from=planner /usr/src/app/recipe.json recipe.json
RUN cargo chef cook --release --recipe-path recipe.json -p routeplane
COPY . .
RUN cargo build --release -p routeplane

# Console builder: compile the Community Edition Console (a static React/Vite
# SPA) to plain assets. Node exists ONLY in this stage — the final image ships
# the built files and never contains Node or the SPA's node_modules. package.json
# is copied first so the dependency install layer caches across source-only edits.
FROM node:22-bookworm-slim AS console-builder
WORKDIR /console
COPY dashboard/package.json ./
RUN npm install --no-audit --no-fund
COPY dashboard/ ./
RUN npm run build

# Runtime base.
#
# Install ca-certificates, then upgrade all OS packages to the latest point
# releases so fixable HIGH/CRITICAL base-layer CVEs are patched.
FROM debian:bookworm-slim AS runtime-base
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && apt-get -y upgrade \
    && rm -rf /var/lib/apt/lists/*

# Run as a non-root user.
RUN groupadd --system --gid 1000 routeplane \
    && useradd --system --uid 1000 --gid routeplane --no-create-home routeplane

WORKDIR /usr/local/bin
ENV PORT=8080
EXPOSE 8080

# Final image: the gateway binary + example configs + license/notices.
FROM runtime-base AS ce
COPY --from=builder --chown=routeplane:routeplane /usr/src/app/target/release/routeplane .
COPY --from=builder --chown=routeplane:routeplane /usr/src/app/configs ./configs
COPY LICENSE THIRD_PARTY_NOTICES.md /usr/local/share/doc/routeplane/
# Ship Cargo.lock so syft's cargo-lock cataloger can enumerate the Rust crates
# in the image SBOM (without it the SBOM lists only OS packages).
COPY --from=builder /usr/src/app/Cargo.lock /usr/local/share/routeplane/Cargo.lock
# The bundled Community Edition Console (static SPA). The gateway serves it
# same-origin when RP_CONSOLE_DIR is set — so this single image is both the
# gateway and its dashboard. No Node in this image; just the built assets.
COPY --from=console-builder --chown=routeplane:routeplane /console/dist /usr/local/share/routeplane/console
ENV RP_CONSOLE_DIR=/usr/local/share/routeplane/console
USER 1000:1000
CMD ["./routeplane"]
