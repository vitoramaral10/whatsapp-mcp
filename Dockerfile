# syntax=docker/dockerfile:1
# Static musl binary on an empty base: no shell, no libc, no package manager.
# libopus and SQLite are compiled in from source; TLS is rustls.
FROM rust:1.99.1-alpine3.24@sha256:27cd9befd4a05d6f6eb407c3e28365f5092f008bfc5946207cca76ae5179ef23 AS build
RUN apk add --no-cache musl-dev cmake make g++ perl
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY .cargo .cargo
COPY src src
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release --locked \
 && cp target/release/whatsapp-mcp /whatsapp-mcp \
 && mkdir -m 1777 /scratch-tmp

FROM scratch
LABEL org.opencontainers.image.source="https://github.com/vitoramaral10/whatsapp-mcp" \
      org.opencontainers.image.description="WhatsApp MCP server in a single static binary" \
      org.opencontainers.image.licenses="MIT"
COPY --from=build /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt
COPY --from=build /whatsapp-mcp /whatsapp-mcp
# SQLite needs a temp dir for large sorts; scratch has none.
COPY --from=build /scratch-tmp /tmp
# The store is a bind mount owned by 1000:1000; media paths keep the old layout.
ENV WHATSAPP_STORE_DIR=/app/whatsapp-bridge/store \
    TMPDIR=/tmp \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
USER 1000:1000
EXPOSE 8000 8080 8090
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s CMD ["/whatsapp-mcp", "healthcheck"]
ENTRYPOINT ["/whatsapp-mcp"]
