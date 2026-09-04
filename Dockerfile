# Stage 1: build the Go bridge (whatsmeow). CGO is required by mattn/go-sqlite3.
FROM golang:1.26-bookworm AS bridge-build

WORKDIR /src
COPY whatsapp-bridge/go.mod whatsapp-bridge/go.sum ./
RUN go mod download

COPY whatsapp-bridge/ ./
RUN CGO_ENABLED=1 GOOS=linux go build -o /out/whatsapp-bridge .

# Stage 2: runtime with Python for the MCP server and ffmpeg for voice messages.
FROM python:3.11-slim-bookworm

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# Upper bound matters: the server code is MCP SDK v1 (FastMCP), which 2.x renamed to
# MCPServer. Lower bound is where streamable-http landed, past the 1.6.0 the repo pins.
RUN pip install --no-cache-dir "mcp[cli]>=1.9.0,<2" "httpx>=0.28.1" "requests>=2.32.3"

WORKDIR /app
COPY --from=bridge-build /out/whatsapp-bridge /app/whatsapp-bridge/whatsapp-bridge
COPY whatsapp-mcp-server/ /app/whatsapp-mcp-server/
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# The bridge writes its SQLite stores relative to its working dir; whatsapp.py reads
# ../whatsapp-bridge/store/messages.db, so this layout must be preserved.
RUN mkdir -p /app/whatsapp-bridge/store && chown -R 1000:1000 /app

ENV MCP_TRANSPORT=streamable-http \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=8000 \
    WHATSAPP_API_BASE_URL=http://127.0.0.1:8080/api \
    PYTHONUNBUFFERED=1

EXPOSE 8000
ENTRYPOINT ["/app/entrypoint.sh"]
