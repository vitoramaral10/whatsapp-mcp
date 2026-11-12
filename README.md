# WhatsApp MCP Server

[![ci](https://github.com/vitoramaral10/whatsapp-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/vitoramaral10/whatsapp-mcp/actions/workflows/ci.yml)

A [Model Context Protocol](https://modelcontextprotocol.io) server for your personal WhatsApp
account: search and read messages (images, videos, documents, voice notes), search contacts,
and send text, files and voice notes to people or groups.

> An independent Rust rewrite that started from Luke Harries'
> [whatsapp-mcp](https://github.com/lharries/whatsapp-mcp) (Go bridge + Python server). It shares
> no code with it today, but keeps its tool names, parameters and history database, so existing
> setups can switch over; see [Migrating from lharries/whatsapp-mcp](#migrating-from-lharrieswhatsapp-mcp).

It connects through the WhatsApp Web multi-device protocol using
[whatsapp-rust](https://github.com/oxidezap/whatsapp-rust). Messages are stored in a local
SQLite database and only reach an LLM when the agent calls a tool.

> **Caution:** like many MCP servers, this one is subject to
> [the lethal trifecta](https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/): prompt
> injection could lead to private data exfiltration. Read [Security](#security) before exposing it.

## What runs

One static binary, `whatsapp-mcp`, in a `FROM scratch` image (~20 MB):

- the WhatsApp session (pairing, history sync, live messages, contacts, LID ↔ phone mapping);
- the MCP server, streamable HTTP on `:8000/mcp`;
- a small REST API on `:8080/api` (`POST /api/send`, `POST /api/download`) for other services;
- audio handling without ffmpeg: Ogg/Opus via a statically linked libopus, MP3/AAC/M4A/WAV/FLAC/Vorbis
  via symphonia;
- optional: semantic search through an embedding server, and voice-note transcription through a
  whisper.cpp server.

## Running

```sh
docker build -t whatsapp-mcp .
mkdir -p store && sudo chown 1000:1000 store   # the image runs as 1000:1000
docker run -it -v "$PWD/store:/app/whatsapp-bridge/store" -p 127.0.0.1:8000:8000 whatsapp-mcp
```

`-it` is needed on the first start only: it prints a QR code to scan in WhatsApp → *Linked
devices*. To pair from the phone alone, set `WHATSAPP_PAIR_PHONE` to your number (country code,
digits only) and type the printed 8-character code under *Link with phone number* instead.

A [`docker-compose.example.yml`](docker-compose.example.yml) and an [`.env.example`](.env.example)
show a hardened setup (read-only root filesystem, no capabilities, ports on localhost only).

### Connecting a client

```sh
# Claude Code
claude mcp add --transport http whatsapp http://127.0.0.1:8000/mcp
```

Other clients that speak streamable HTTP:

```json
{ "mcpServers": { "whatsapp": { "type": "http", "url": "http://127.0.0.1:8000/mcp" } } }
```

Clients that only speak stdio can go through [`mcp-remote`](https://www.npmjs.com/package/mcp-remote):
`npx mcp-remote http://127.0.0.1:8000/mcp`.

### Configuration

| Variable | Default | Meaning |
|---|---|---|
| `WHATSAPP_STORE_DIR` | `/app/whatsapp-bridge/store` | History (`messages.db`), session (`whatsapp-rs.db`), downloaded media |
| `WHATSAPP_SESSION_DB` | `whatsapp-rs.db` | Session file name inside the store |
| `WHATSAPP_PAIR_PHONE` | — | Pair with a code instead of a QR |
| `MCP_HOST` / `MCP_PORT` | `0.0.0.0` / `8000` | MCP listener |
| `WHATSAPP_API_HOST` / `WHATSAPP_API_PORT` | `MCP_HOST` / `8080` | REST API listener |
| `WHATSAPP_SEND_DIRS` | store dir | `:`-separated directories `send_file` and `/api/send` may read from; databases and files in the store root are always refused |
| `WHATSAPP_UPLOAD_URL` | — | Public base URL of the upload listener (e.g. `http://homelab:8090`); turns on `return_upload_url` in `send_file` and `send_audio_message` |
| `WHATSAPP_UPLOAD_HOST` / `WHATSAPP_UPLOAD_PORT` | `MCP_HOST` / `8090` | Upload listener; it serves only `PUT /upload/<token>` and starts only with `WHATSAPP_UPLOAD_URL` set |
| `WHATSAPP_MEDIA_HOST_PATH` | — | Host side of the store bind mount, so returned paths open on the host |
| `WHATSAPP_MEDIA_MAX_INLINE_BYTES` | `8388608` | Larger media come back as a path only |
| `WHATSAPP_READ_ONLY` | — | `true` hides every tool that writes (send, edit, delete, react, mark read) and makes `/api/send` refuse |
| `WHATSAPP_ALLOWED_CHATS` | — | `,`-separated JIDs or phone numbers: tools and the REST API read, download and send only in these chats; the history still records every chat |
| `EMBED_URL` / `EMBED_MODEL` | — / `bge-m3` | OpenAI-compatible `/v1/embeddings` server (e.g. `llama-server --embedding`); enables the semantic half of `search_messages` |
| `WHISPER_URL` | — | whisper.cpp server; voice notes of the last 3 days are transcribed as they arrive, and older ones when downloaded; the transcript is saved as the text of the message (`[transcription] ` prefix) so `list_messages` and `search_messages` find it |
| `WHISPER_WAIT_S` / `WHISPER_TIMEOUT_S` | `10` / `180` | How long a call waits for the transcript / per-run limit |
| `WHATSAPP_LOG_LEVEL` | `info` | Log level of the WhatsApp library |
| `WHATSAPP_LOG` | — | Full [`env_logger`](https://docs.rs/env_logger) filter; overrides the level above |

The default store path keeps the layout of the lharries/whatsapp-mcp Go bridge, so existing
deployments keep working.

## MCP tools

- **search_contacts**: contacts by name or phone number
- **search_messages**: best matches first, by meaning (embeddings) and by words (SQLite FTS5),
  merged by reciprocal rank; full-text only when no embedding server is set; each message is
  embedded with up to 4 earlier messages of its conversation (within 2 hours), so a short reply is
  found by the topic it answers, and one stretch of chat takes one result slot
- **list_messages**: messages with filters (date, sender, chat, words via FTS5) and surrounding context
- **list_chats** (`unread_only`), **get_chat**, **get_direct_chat_by_contact**, **get_contact_chats**
- **get_last_interaction**, **get_message_context**
- **send_message**: text to a phone number or JID; `reply_to_message_id` quotes a message, and
  every `@<phone number>` in the text becomes a mention
- **send_file**: image, video or document, from a path in the send dirs, from the client's disk
  with `return_upload_url` + `filename`, or a small `content_base64` + `filename`
- **send_audio_message**: MP3, AAC/M4A, WAV, FLAC, Ogg Vorbis or Ogg Opus, sent as an Opus voice note
- **get_digest**: a period chat by chat, busiest first, with message and unread counts and the
  latest messages in order, for "what happened today" summaries
- **send_reaction**: an emoji on a message (empty removes it)
- **mark_chat_read**: read receipts for the unread messages and the chat read on the phone
- **edit_message**, **delete_message**: rewrite (text only, within WhatsApp's ~15 min window)
  or delete for everyone a message you sent
- **download_media**: the file itself (image block or embedded resource) plus metadata; voice
  notes also carry `transcription`, or `transcription_pending` while whisper is still running

A client on another machine has no path in the send dirs, and a model cannot type a PDF out as
base64. With `return_upload_url: true` (and `WHATSAPP_UPLOAD_URL` set) the send tools send
nothing yet and return a one-shot `upload_url`; the client runs `curl -sS -T <file> '<upload_url>'`
and the message goes out when the upload completes, with the send result as the response. The
URL is a 128-bit random token, works once and expires in 15 minutes; publish the upload port
only where the clients are (a VPN, the LAN), since the token is its only credential.

Every listed message carries its `Message ID` and `Chat JID`, and the send tools return the new
message's ID: those are what replies, edits, deletions and reactions take. Unread counts start
after the later of your own last message and the last read your phone reported; at the first
start with this version everything before counts as read.

Edits rewrite the stored message (`[edited] ` prefix), deletions keep the text under a
`[deleted] ` prefix, and reactions are stored as their own line quoting the message.

Media whose CDN copy expired is requested again from the phone, which must be online. This works
for recent media; for old media the phone uploads a copy under a key it does not share, so those
files cannot be recovered by any linked device.

## REST API

For services on the same network that need to send without speaking MCP. It honours
`WHATSAPP_READ_ONLY` and `WHATSAPP_ALLOWED_CHATS` like the tools do.

```sh
curl -X POST localhost:8080/api/send -H 'Content-Type: application/json' \
  -d '{"recipient": "5511900000000", "message": "hello", "media_path": ""}'
curl -X POST localhost:8080/api/download -H 'Content-Type: application/json' \
  -d '{"message_id": "ABC123", "chat_jid": "5511900000000@s.whatsapp.net"}'
```

Both answer `{"success": bool, "message": string}`; `/api/download` adds `filename` and `path`.

## Security

- **Neither the MCP endpoint nor the REST API has authentication.** Anyone who can reach port
  8000 can read your whole history and send as you; anyone who can reach 8080 can send. Publish
  the ports on `127.0.0.1` only, keep them on a private container network, or put an
  authenticating reverse proxy in front.
- The store holds the session keys: whoever has a copy is logged in as you. The server creates
  it with mode `700`/`600`; keep backups encrypted.
- To limit what a prompt-injected agent can do, combine `WHATSAPP_READ_ONLY=true` with
  `WHATSAPP_ALLOWED_CHATS`.
- Report vulnerabilities as described in [SECURITY.md](SECURITY.md).

## Migrating from lharries/whatsapp-mcp

A store from its Go bridge works as is: `messages.db` keeps its schema, and contacts
and LID mappings are imported once from its `whatsapp.db`. The session itself cannot be carried
over, so the device has to be paired again. The MCP tools keep their names and parameters.

## Development

Building needs Rust 1.99+, cmake, a C/C++ compiler and perl: libopus and SQLite are compiled
from source, and [`.cargo/config.toml`](.cargo/config.toml) sets the flags. The `simd` feature of
whatsapp-rust needs nightly and stays disabled.

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

Tests cover the history store against the Go bridge's format, full-text and hybrid search, the
chat allowlist, unread counts, message text extraction (menus, polls, wrappers, edits,
reactions), mentions, read-only mode, audio decode/encode, media paths and transcripts. CI runs
the same three commands and a Docker build on every push and pull request.

Changes are listed in [CHANGELOG.md](CHANGELOG.md).

## License

MIT, see [LICENSE](LICENSE). Note that `wacore-libsignal`, a dependency, carries
`AGPL-3.0-only` headers on some files
([upstream issue #1828](https://github.com/oxidezap/whatsapp-rust/issues/1828)).
