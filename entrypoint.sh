#!/bin/sh
# Runs the Go bridge and the Python MCP server side by side. The bridge owns the
# WhatsApp session (QR login on first run, visible via `docker compose logs`) and
# serves the REST API the MCP server calls on 127.0.0.1:8080.
set -e

term() {
    kill -TERM "$bridge_pid" "$mcp_pid" 2>/dev/null || true
    wait "$bridge_pid" "$mcp_pid" 2>/dev/null || true
    exit 0
}
trap term TERM INT

cd /app/whatsapp-bridge
./whatsapp-bridge &
bridge_pid=$!

# Wait for the bridge REST API before starting the MCP server, so the first tool
# call does not hit a connection refused.
i=0
while [ "$i" -lt 120 ]; do
    if python3 -c "import socket,sys; sys.exit(0 if socket.create_connection(('127.0.0.1',8080),1) else 1)" 2>/dev/null; then
        break
    fi
    if ! kill -0 "$bridge_pid" 2>/dev/null; then
        echo "whatsapp-bridge exited before the REST API came up" >&2
        exit 1
    fi
    i=$((i + 1))
    sleep 1
done

cd /app/whatsapp-mcp-server
python3 main.py &
mcp_pid=$!

# `wait -n` is not available in dash, so poll both PIDs and exit as soon as either
# dies, letting Docker's restart policy take over.
while kill -0 "$bridge_pid" 2>/dev/null && kill -0 "$mcp_pid" 2>/dev/null; do
    sleep 5
done

echo "one of the processes exited, shutting down" >&2
term
