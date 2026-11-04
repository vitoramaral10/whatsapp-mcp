# Changelog

Notable changes, newest first. The format follows [Keep a Changelog](https://keepachangelog.com),
and versions follow [Semantic Versioning](https://semver.org).

## [Unreleased]

### Added
- Unread counts in `list_chats` (`unread_only`), `mark_chat_read`, `send_reaction` and `get_digest`.
- `edit_message` and `delete_message` for messages you sent.
- Voice notes are transcribed as they arrive, and transcripts are stored as the message text.
- Message IDs in every listing; send tools return the new message's ID.
- CI (fmt, clippy, tests, Docker build), security policy, compose and `.env` examples.

### Changed
- The REST API honours `WHATSAPP_READ_ONLY` and `WHATSAPP_ALLOWED_CHATS`, and listens on
  `WHATSAPP_API_HOST` (defaults to `MCP_HOST`).
- Every tool failure is a tool error (`isError: true`) with `{"success": false, "message": ...}`;
  malformed arguments (bad dates, invalid recipients, unknown `sort_by`) are protocol errors.
- Recipients are validated: 8–15 digits, or a JID on `s.whatsapp.net`, `g.us` or `lid`. A bad
  `WHATSAPP_ALLOWED_CHATS` entry stops startup instead of being ignored.
- `limit` is capped at 200, context at 50 and digest sizes at 100.
- Polls, locations, contacts, invites, events and view-once media are stored with a `[poll]`,
  `[location]`, `[contact]`, `[group invite]`, `[event]` or `[view-once]` prefix.
- `send_file` takes an optional `caption`; base64 uploads accept data URLs and must stay under
  about 3 MB (larger files go through `media_path`).
- Audio over 30 minutes or 100 MiB, and files to send over 100 MiB, are refused.
- The server announces itself as `whatsapp-mcp` with its version and usage instructions.
- Release builds unwind on panic, so a decoder panic on a malformed file fails that call
  instead of the whole server.

### Fixed
- Expired media is requested from the phone with the right address, and only when the CDN
  refuses the file.
- A redelivered message no longer undoes an edit, a deletion or a transcript, and no longer
  drops its stored media fields or its embedding; media fields move as one set.
- Downloads are streamed to a temporary file and renamed when complete, so a partial file is
  never served.
- The client shuts down cleanly on SIGTERM, flushing its session state. A logged-out session
  is logged as such and the history stays readable, instead of a restart loop.
- Message context keeps neighbours sent in the same second, in order, and each message is
  listed once.
- Sender names, contact chats and last interactions match senders stored by number.
- Hybrid search ranks deterministically; vectors of another model are ignored.
- Merging a LID chat keeps the real name, read marks and edited or transcribed text.
- Auto-transcription no longer starves on failed notes or stops for a whisper outage.
- Faster unread counts and contact lookups through new indexes.

## [0.2.0] - 2026-11-03

### Changed
- **Breaking:** the Go bridge and the Python MCP server are replaced by one static Rust binary
  in a `FROM scratch` image. The history store keeps its schema; the device must be paired again.

### Added
- Hybrid search (embeddings + FTS5), mentions, replies, recovery of expired media, chat allowlist
  and read-only mode.

## 2026-09-04

- Docker deployment of the original Go bridge and Python server.

[Unreleased]: https://github.com/vitoramaral10/whatsapp-mcp/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/vitoramaral10/whatsapp-mcp/releases/tag/v0.2.0
