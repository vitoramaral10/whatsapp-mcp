import base64
import json
import mimetypes
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeout
from contextlib import contextmanager

import audio
from typing import List, Dict, Any, Optional
from dataclasses import asdict, is_dataclass
from datetime import datetime
from mcp.server.fastmcp import FastMCP
from mcp.types import BlobResourceContents, EmbeddedResource, ImageContent, TextContent
from whatsapp import (
    search_contacts as whatsapp_search_contacts,
    list_messages as whatsapp_list_messages,
    list_chats as whatsapp_list_chats,
    get_chat as whatsapp_get_chat,
    get_direct_chat_by_contact as whatsapp_get_direct_chat_by_contact,
    get_contact_chats as whatsapp_get_contact_chats,
    get_last_interaction as whatsapp_get_last_interaction,
    get_message_context as whatsapp_get_message_context,
    send_message as whatsapp_send_message,
    send_file as whatsapp_send_file,
    send_audio_message as whatsapp_audio_voice_message,
    download_media as whatsapp_download_media
)

# Initialize FastMCP server
import os

# Transport/bind are configurable so the same server can run as stdio (upstream
# default) or as a streamable-http service inside Docker.
MCP_TRANSPORT = os.environ.get("MCP_TRANSPORT", "stdio")
MCP_HOST = os.environ.get("MCP_HOST", "127.0.0.1")
MCP_PORT = int(os.environ.get("MCP_PORT", "8000"))

# The bridge writes media inside the container, so the path it returns is
# meaningless to a client running anywhere else. When the store is a bind mount,
# these two say how to rewrite the container path into the caller's path.
MEDIA_CONTAINER_PATH = os.environ.get("WHATSAPP_MEDIA_CONTAINER_PATH", "/app/whatsapp-bridge/store")
MEDIA_HOST_PATH = os.environ.get("WHATSAPP_MEDIA_HOST_PATH", "")


def _host_visible_path(file_path: str) -> str:
    """Rewrite a container media path into the path the caller can open."""
    if not MEDIA_HOST_PATH or not file_path.startswith(MEDIA_CONTAINER_PATH):
        return file_path
    return MEDIA_HOST_PATH.rstrip("/") + file_path[len(MEDIA_CONTAINER_PATH):]


def _container_path(file_path: str) -> str:
    """Undo _host_visible_path, so send_* accepts the path download_media returned."""
    host = MEDIA_HOST_PATH.rstrip("/")
    if not host or not file_path.startswith(host + "/"):
        return file_path
    return MEDIA_CONTAINER_PATH + file_path[len(host):]


# Clients rarely share this container's disk, so media also travels inline as
# base64. Above this size download_media falls back to returning only the path.
MEDIA_MAX_INLINE_BYTES = int(os.environ.get("WHATSAPP_MEDIA_MAX_INLINE_BYTES", str(8 * 1024 * 1024)))

# Clients cannot listen to audio, so download_media adds a transcript when a
# whisper.cpp server is configured. On a small GPU one run can outlast the
# gateway's deadline, so it runs in the background and is cached next to the
# file: the call waits WHISPER_WAIT_S, and a later call picks up the text.
WHISPER_URL = os.environ.get("WHISPER_URL", "")
# The slim image has no /etc/mime.types, so voice notes would read as octet-stream.
mimetypes.add_type("audio/ogg", ".ogg")
WHISPER_WAIT_S = float(os.environ.get("WHISPER_WAIT_S", "10"))
WHISPER_TIMEOUT_S = float(os.environ.get("WHISPER_TIMEOUT_S", "180"))
# whisper-server handles one request at a time anyway.
_transcriber = ThreadPoolExecutor(max_workers=1)
_transcribing: Dict[str, Future] = {}


def _transcribe_to_cache(file_path: str) -> str:
    try:
        text = audio.transcribe(file_path, WHISPER_URL, WHISPER_TIMEOUT_S)
        with open(file_path + ".txt.tmp", "w") as f:
            f.write(text)
        os.replace(file_path + ".txt.tmp", file_path + ".txt")
        return text
    finally:
        _transcribing.pop(file_path, None)


def _transcription(file_path: str) -> Dict[str, str]:
    """transcription, transcription_pending or transcription_error for download_media."""
    try:
        with open(file_path + ".txt") as f:
            return {"transcription": f.read()}
    except FileNotFoundError:
        pass
    future = _transcribing.get(file_path)
    if future is None:
        future = _transcribing[file_path] = _transcriber.submit(_transcribe_to_cache, file_path)
    try:
        return {"transcription": future.result(timeout=WHISPER_WAIT_S)}
    except FutureTimeout:
        return {"transcription_pending": "Still transcribing; call download_media again in about 30 seconds"}
    except RuntimeError as e:
        return {"transcription_error": str(e)}


@contextmanager
def _media_file(media_path: str, content_base64: str, filename: str):
    """Yield a path the bridge can read: media_path, or the decoded upload in a temp dir.

    The bridge picks the media type from the extension and uses the basename as the
    document title, so the upload is written under its own filename.
    """
    if not content_base64:
        yield _container_path(media_path)
        return
    name = os.path.basename(filename or "")
    if "." not in name.strip("."):
        raise ValueError("filename with an extension is required with content_base64")
    data = base64.b64decode(content_base64, validate=True)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, name)
        with open(path, "wb") as f:
            f.write(data)
        yield path

mcp =FastMCP("whatsapp", host=MCP_HOST, port=MCP_PORT)


def _jsonable(value):
    """Convert the dataclasses whatsapp.py returns into plain JSON-safe structures.

    The MCP SDK validates tool results against the declared return annotation, so
    Chat/Contact/Message instances have to be flattened before they are returned.
    """
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, datetime):
        return value.isoformat()
    return value


@mcp.tool()
def search_contacts(query: str) -> List[Dict[str, Any]]:
    """Search WhatsApp contacts by name or phone number.
    
    Args:
        query: Search term to match against contact names or phone numbers
    """
    contacts = whatsapp_search_contacts(query)
    return _jsonable(contacts)

@mcp.tool()
def list_messages(
    after: Optional[str] = None,
    before: Optional[str] = None,
    sender_phone_number: Optional[str] = None,
    chat_jid: Optional[str] = None,
    query: Optional[str] = None,
    limit: int = 20,
    page: int = 0,
    include_context: bool = True,
    context_before: int = 1,
    context_after: int = 1
) -> str:
    """Get WhatsApp messages matching specified criteria with optional context.
    
    Args:
        after: Optional ISO-8601 formatted string to only return messages after this date
        before: Optional ISO-8601 formatted string to only return messages before this date
        sender_phone_number: Optional phone number to filter messages by sender
        chat_jid: Optional chat JID to filter messages by chat
        query: Optional search term to filter messages by content
        limit: Maximum number of messages to return (default 20)
        page: Page number for pagination (default 0)
        include_context: Whether to include messages before and after matches (default True)
        context_before: Number of messages to include before each match (default 1)
        context_after: Number of messages to include after each match (default 1)
    """
    messages = whatsapp_list_messages(
        after=after,
        before=before,
        sender_phone_number=sender_phone_number,
        chat_jid=chat_jid,
        query=query,
        limit=limit,
        page=page,
        include_context=include_context,
        context_before=context_before,
        context_after=context_after
    )
    # whatsapp.list_messages returns a preformatted transcript (and [] when the
    # query fails), so the annotation above is str, not a list of dicts.
    return messages if isinstance(messages, str) else ""

@mcp.tool()
def list_chats(
    query: Optional[str] = None,
    limit: int = 20,
    page: int = 0,
    include_last_message: bool = True,
    sort_by: str = "last_active"
) -> List[Dict[str, Any]]:
    """Get WhatsApp chats matching specified criteria.
    
    Args:
        query: Optional search term to filter chats by name or JID
        limit: Maximum number of chats to return (default 20)
        page: Page number for pagination (default 0)
        include_last_message: Whether to include the last message in each chat (default True)
        sort_by: Field to sort results by, either "last_active" or "name" (default "last_active")
    """
    chats = whatsapp_list_chats(
        query=query,
        limit=limit,
        page=page,
        include_last_message=include_last_message,
        sort_by=sort_by
    )
    return _jsonable(chats)

@mcp.tool()
def get_chat(chat_jid: str, include_last_message: bool = True) -> Dict[str, Any]:
    """Get WhatsApp chat metadata by JID.
    
    Args:
        chat_jid: The JID of the chat to retrieve
        include_last_message: Whether to include the last message (default True)
    """
    chat = whatsapp_get_chat(chat_jid, include_last_message)
    return _jsonable(chat)

@mcp.tool()
def get_direct_chat_by_contact(sender_phone_number: str) -> Dict[str, Any]:
    """Get WhatsApp chat metadata by sender phone number.
    
    Args:
        sender_phone_number: The phone number to search for
    """
    chat = whatsapp_get_direct_chat_by_contact(sender_phone_number)
    return _jsonable(chat)

@mcp.tool()
def get_contact_chats(jid: str, limit: int = 20, page: int = 0) -> List[Dict[str, Any]]:
    """Get all WhatsApp chats involving the contact.
    
    Args:
        jid: The contact's JID to search for
        limit: Maximum number of chats to return (default 20)
        page: Page number for pagination (default 0)
    """
    chats = whatsapp_get_contact_chats(jid, limit, page)
    return _jsonable(chats)

@mcp.tool()
def get_last_interaction(jid: str) -> str:
    """Get most recent WhatsApp message involving the contact.
    
    Args:
        jid: The JID of the contact to search for
    """
    message = whatsapp_get_last_interaction(jid)
    return message

@mcp.tool()
def get_message_context(
    message_id: str,
    before: int = 5,
    after: int = 5
) -> Dict[str, Any]:
    """Get context around a specific WhatsApp message.
    
    Args:
        message_id: The ID of the message to get context for
        before: Number of messages to include before the target message (default 5)
        after: Number of messages to include after the target message (default 5)
    """
    context = whatsapp_get_message_context(message_id, before, after)
    return _jsonable(context)

@mcp.tool()
def send_message(
    recipient: str,
    message: str
) -> Dict[str, Any]:
    """Send a WhatsApp message to a person or group. For group chats use the JID.

    Args:
        recipient: The recipient - either a phone number with country code but no + or other symbols,
                 or a JID (e.g., "123456789@s.whatsapp.net" or a group JID like "123456789@g.us")
        message: The message text to send
    
    Returns:
        A dictionary containing success status and a status message
    """
    # Validate input
    if not recipient:
        return {
            "success": False,
            "message": "Recipient must be provided"
        }
    
    # Call the whatsapp_send_message function with the unified recipient parameter
    success, status_message = whatsapp_send_message(recipient, message)
    return {
        "success": success,
        "message": status_message
    }

@mcp.tool()
def send_file(recipient: str, media_path: str = "", content_base64: str = "", filename: str = "") -> Dict[str, Any]:
    """Send a file such as a picture, raw audio, video or document via WhatsApp to the specified recipient. For group messages use the JID.

    Args:
        recipient: The recipient - either a phone number with country code but no + or other symbols,
                 or a JID (e.g., "123456789@s.whatsapp.net" or a group JID like "123456789@g.us")
        media_path: Path to a file already on the WhatsApp server (e.g. one returned by download_media).
                 Leave empty when sending content_base64.
        content_base64: The file content, base64-encoded. Use this to send a file from the client side.
        filename: Required with content_base64; its extension picks the media type and it is shown
                 as the document name (e.g. "relatorio.pdf")

    Returns:
        A dictionary containing success status and a status message
    """
    try:
        with _media_file(media_path, content_base64, filename) as path:
            success, status_message = whatsapp_send_file(recipient, path)
    except ValueError as e:
        success, status_message = False, f"Invalid file content: {e}"
    return {
        "success": success,
        "message": status_message
    }

@mcp.tool()
def send_audio_message(recipient: str, media_path: str = "", content_base64: str = "", filename: str = "") -> Dict[str, Any]:
    """Send any audio file as a WhatsApp audio message to the specified recipient. For group messages use the JID. If it errors due to ffmpeg not being installed, use send_file instead.

    Args:
        recipient: The recipient - either a phone number with country code but no + or other symbols,
                 or a JID (e.g., "123456789@s.whatsapp.net" or a group JID like "123456789@g.us")
        media_path: Path to an audio file already on the WhatsApp server (will be converted to Opus .ogg
                 if it's not a .ogg file). Leave empty when sending content_base64.
        content_base64: The audio content, base64-encoded. Use this to send audio from the client side.
        filename: Required with content_base64; its extension tells the format (e.g. "nota.mp3")

    Returns:
        A dictionary containing success status and a status message
    """
    try:
        with _media_file(media_path, content_base64, filename) as path:
            success, status_message = whatsapp_audio_voice_message(recipient, path)
    except ValueError as e:
        success, status_message = False, f"Invalid file content: {e}"
    return {
        "success": success,
        "message": status_message
    }

@mcp.tool(structured_output=False)
def download_media(message_id: str, chat_jid: str) -> list:
    """Download media from a WhatsApp message and return its content.

    Args:
        message_id: The ID of the message containing the media
        chat_jid: The JID of the chat containing the message

    Returns:
        A JSON text block with success status, message, file_path, mime_type and size_bytes
        (plus transcription, transcription_pending or transcription_error for audio), followed by the file itself: an image block for pictures, an embedded resource
        (base64 blob) for everything else. Files over the inline limit come back as path only.
    """
    file_path = whatsapp_download_media(message_id, chat_jid)
    if not file_path:
        return [TextContent(type="text", text=json.dumps({"success": False, "message": "Failed to download media"}))]

    host_path = _host_visible_path(file_path)
    mime = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
    size = os.path.getsize(file_path)
    info = {"success": True, "message": "Media downloaded successfully",
            "file_path": host_path, "mime_type": mime, "size_bytes": size}
    if mime.startswith("audio/") and WHISPER_URL:
        info.update(_transcription(file_path))
    if size > MEDIA_MAX_INLINE_BYTES:
        info["message"] += f"; over the {MEDIA_MAX_INLINE_BYTES}-byte inline limit, so only the path is returned"
        return [TextContent(type="text", text=json.dumps(info))]

    with open(file_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    if mime.startswith("image/"):
        content = ImageContent(type="image", data=data, mimeType=mime)
    else:
        content = EmbeddedResource(type="resource", resource=BlobResourceContents(
            uri=f"file://{host_path}", mimeType=mime, blob=data))
    return [TextContent(type="text", text=json.dumps(info)), content]

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport=MCP_TRANSPORT)
