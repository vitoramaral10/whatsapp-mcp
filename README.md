# WhatsApp MCP Server

A Model Context Protocol (MCP) server for your personal WhatsApp account: search and read
messages (images, videos, documents, voice notes), search contacts, and send text, files and
voice notes to people or groups.

It connects through the WhatsApp Web multi-device protocol using
[whatsapp-rust](https://github.com/oxidezap/whatsapp-rust). Messages are stored in a local
SQLite database and only reach an LLM when the agent calls a tool.

![WhatsApp MCP](./example-use.png)

> *Caution:* like many MCP servers, this one is subject to
> [the lethal trifecta](https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/): prompt
> injection could lead to private data exfiltration.

## What runs

One static Rust binary, `whatsapp-mcp`, in a `FROM scratch` image (~20 MB):

- the WhatsApp session (pairing, history sync, live messages, contacts, LID ↔ phone mapping);
- the MCP server, streamable HTTP on `:8000/mcp`;
- a small REST API on `:8080/api` (`POST /api/send`, `POST /api/download`) for other services;
- audio handling without ffmpeg: Ogg/Opus via a statically linked libopus, MP3/AAC/M4A/WAV/FLAC
  via symphonia.

## Running

```sh
docker build -t whatsapp-mcp .
docker run -it -v "$PWD/store:/app/whatsapp-bridge/store" -p 8000:8000 whatsapp-mcp
```

On first start it prints a QR code to scan in WhatsApp → *Linked devices*. To pair from the
phone alone, set `WHATSAPP_PAIR_PHONE` to your number (country code, digits only) and type the
printed 8-character code under *Link with phone number* instead.

### Configuration

| Variable | Default | Meaning |
|---|---|---|
| `WHATSAPP_STORE_DIR` | `/app/whatsapp-bridge/store` | History (`messages.db`), session (`whatsapp-rs.db`), downloaded media |
| `WHATSAPP_SESSION_DB` | `whatsapp-rs.db` | Session file name inside the store |
| `WHATSAPP_PAIR_PHONE` | — | Pair with a code instead of a QR |
| `MCP_HOST` / `MCP_PORT` | `0.0.0.0` / `8000` | MCP listener |
| `WHATSAPP_API_PORT` | `8080` | REST API listener |
| `WHATSAPP_SEND_DIRS` | store dir | `:`-separated directories `send_file` and `/api/send` may read from; databases and files in the store root are always refused |
| `WHATSAPP_MEDIA_HOST_PATH` | — | Host side of the store bind mount, so returned paths open on the host |
| `WHATSAPP_MEDIA_MAX_INLINE_BYTES` | `8388608` | Larger media come back as a path only |
| `WHATSAPP_READ_ONLY` | — | `true` hides every tool that writes (send, edit, delete, react, mark read) from MCP clients (the REST API still sends) |
| `WHATSAPP_ALLOWED_CHATS` | — | `,`-separated JIDs or phone numbers: the MCP tools read, download and send only in these chats; the history still records every chat, and the REST API, meant for services on a private network, is not limited |
| `EMBED_URL` / `EMBED_MODEL` | — / `bge-m3` | OpenAI-compatible `/v1/embeddings` server (e.g. `llama-server --embedding`); enables the semantic half of `search_messages` |
| `WHISPER_URL` | — | whisper.cpp server; voice notes of the last 3 days are transcribed as they arrive, and older ones when downloaded; the transcript is saved as the text of the message (`[transcription] ` prefix) so `list_messages` and `search_messages` find it |
| `WHISPER_WAIT_S` / `WHISPER_TIMEOUT_S` | `10` / `180` | How long a call waits for the transcript / per-run limit |
| `WHATSAPP_LOG_LEVEL` | `info` | Log level of the WhatsApp library |

A store from the earlier Go bridge works as is: `messages.db` keeps its schema, and contacts
and LID mappings are imported once from its `whatsapp.db`. The session itself cannot be carried
over, so the device has to be paired again.

## MCP tools

- **search_contacts**: contacts by name or phone number
- **search_messages**: best matches first, by meaning (embeddings) and by words (SQLite FTS5),
  merged by reciprocal rank; full-text only when no embedding server is set
- **list_messages**: messages with filters (date, sender, chat, words via FTS5) and surrounding context
- **list_chats** (`unread_only`), **get_chat**, **get_direct_chat_by_contact**, **get_contact_chats**
- **get_last_interaction**, **get_message_context**
- **send_message**: text to a phone number or JID; `reply_to_message_id` quotes a message, and
  every `@<phone number>` in the text becomes a mention
- **send_file**: image, video or document, from a path in the send dirs or `content_base64` + `filename`
- **send_audio_message**: any decodable audio sent as an Opus voice note
- **get_digest**: a period chat by chat, busiest first, with message and unread counts and the
  latest messages in order, for "what happened today" summaries
- **send_reaction**: an emoji on a message (empty removes it)
- **mark_chat_read**: read receipts for the unread messages and the chat read on the phone
- **edit_message**, **delete_message**: rewrite (text only, within WhatsApp's ~15 min window)
  or delete for everyone a message you sent
- **download_media**: the file itself (image block or embedded resource) plus metadata; voice
  notes also carry `transcription`, or `transcription_pending` while whisper is still running

Every listed message carries its `Message ID` and `Chat JID`, and the send tools return the new
message's ID: those are what replies, edits, deletions and reactions take. Unread counts start
after the later of your own last message and the last read your phone reported; at the first
start with this version everything before counts as read.

Edits rewrite the stored message (`[edited] ` prefix), deletions keep the text under a
`[deleted] ` prefix, and reactions are stored as their own line quoting the message. Media whose
CDN copy expired is requested again from the phone, which must be online for that.

## Development

```sh
cargo test
```

Tests cover the history store against the Go bridge's format, full-text and hybrid search, the
chat allowlist, message text extraction (menus, polls, wrappers, edits, reactions), mentions,
read-only mode, audio decode/encode, media paths and transcripts.

## License

MIT. Note that `wacore-libsignal`, a dependency, carries `AGPL-3.0-only` headers on some files
([upstream issue #1828](https://github.com/oxidezap/whatsapp-rust/issues/1828)).
