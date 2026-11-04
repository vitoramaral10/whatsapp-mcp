# syntax=docker/dockerfile:1
# Static musl binary on an empty base: no shell, no libc, no package manager.
# libopus and SQLite are compiled in from source; TLS is rustls.
FROM rust:1.99.1-alpine3.24 AS build
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
COPY --from=build /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt
COPY --from=build /whatsapp-mcp /whatsapp-mcp
# SQLite needs a temp dir for large sorts; scratch has none.
COPY --from=build /scratch-tmp /tmp
# The store is a bind mount owned by 1000:1000; media paths keep the old layout.
ENV WHATSAPP_STORE_DIR=/app/whatsapp-bridge/store \
    TMPDIR=/tmp \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
USER 1000:1000
EXPOSE 8000 8080
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s CMD ["/whatsapp-mcp", "healthcheck"]
ENTRYPOINT ["/whatsapp-mcp"]
