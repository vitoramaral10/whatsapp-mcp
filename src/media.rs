//! Media files on disk: where downloads go, how paths look to the caller, what
//! MIME type a file has, and turning a base64 upload into a file to send.

use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use base64::Engine;

/// Host/container path rewriting for a bind-mounted store, so a path returned
/// by download_media is one the caller can open, and send_* accepts it back.
#[derive(Clone, Debug, Default)]
pub struct PathMap {
    pub container: String,
    pub host: String,
    /// The only directories send_* reads from.
    pub send_dirs: Vec<PathBuf>,
}

impl PathMap {
    pub fn to_host(&self, path: &str) -> String {
        match path.strip_prefix(self.container.trim_end_matches('/')) {
            Some(rest) if !self.host.is_empty() && (rest.is_empty() || rest.starts_with('/')) => {
                format!("{}{}", self.host.trim_end_matches('/'), rest)
            }
            _ => path.to_string(),
        }
    }

    pub fn to_container(&self, path: &str) -> String {
        let host = self.host.trim_end_matches('/');
        match path.strip_prefix(host) {
            Some(rest) if !host.is_empty() && rest.starts_with('/') => format!("{}{}", self.container.trim_end_matches('/'), rest),
            _ => path.to_string(),
        }
    }
}

/// MIME type from the extension, from a table compiled into the binary.
pub fn mime_of(path: &Path) -> String {
    // mime_guess says audio/m4a, which is not a registered type.
    if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("m4a")) {
        return "audio/mp4".into();
    }
    mime_guess::from_path(path).first_raw().unwrap_or("application/octet-stream").to_string()
}

/// Width, height and whether it is animated, read from a WebP header; None if not WebP.
pub fn webp_info(d: &[u8]) -> Option<(u32, u32, bool)> {
    if d.len() < 30 || &d[0..4] != b"RIFF" || &d[8..12] != b"WEBP" {
        return None;
    }
    let u24 = |i: usize| u32::from(d[i]) | u32::from(d[i + 1]) << 8 | u32::from(d[i + 2]) << 16;
    match &d[12..16] {
        b"VP8X" => Some((u24(24) + 1, u24(27) + 1, d[20] & 0x02 != 0)),
        b"VP8 " => {
            Some((u32::from(u16::from_le_bytes([d[26], d[27]]) & 0x3fff), u32::from(u16::from_le_bytes([d[28], d[29]]) & 0x3fff), false))
        }
        b"VP8L" => {
            let b = u32::from_le_bytes([d[21], d[22], d[23], d[24]]);
            Some(((b & 0x3fff) + 1, (b >> 14 & 0x3fff) + 1, false))
        }
        _ => None,
    }
}

/// Keeps a message ID usable as one path segment.
pub fn sanitize(value: &str) -> String {
    value.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect()
}

/// Where a message's media is saved. The stored filename comes from the clock,
/// so many messages share it; the message ID prefix keeps each file distinct.
pub fn download_path(store_dir: &Path, chat_jid: &str, message_id: &str, filename: &str) -> PathBuf {
    let name = Path::new(filename).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    store_dir.join(chat_jid.replace(':', "_")).join(format!("{}-{}", sanitize(message_id), name))
}

/// Largest file send_* reads (WhatsApp's own limit for documents).
pub const MAX_SEND_BYTES: u64 = 100 << 20;

/// A file to send: either a path already on this side, or uploaded bytes.
pub struct Outgoing {
    pub data: Vec<u8>,
    /// Basename with extension: picks the media type and is the document title.
    pub name: String,
}

/// The file to send, read off the async workers. `allowed` is the chat
/// allowlist: files under the store must come from an allowed chat's folder.
pub async fn outgoing(
    paths: &PathMap,
    allowed: impl Fn(&str) -> bool + Send + 'static,
    media_path: &str,
    content_base64: &str,
    filename: &str,
) -> Result<Outgoing> {
    let (paths, media_path, content_base64, filename) =
        (paths.clone(), media_path.to_owned(), content_base64.to_owned(), filename.to_owned());
    tokio::task::spawn_blocking(move || read_outgoing(&paths, &allowed, &media_path, &content_base64, &filename)).await?
}

fn read_outgoing(
    paths: &PathMap,
    allowed: &dyn Fn(&str) -> bool,
    media_path: &str,
    content_base64: &str,
    filename: &str,
) -> Result<Outgoing> {
    if !content_base64.is_empty() && !media_path.is_empty() {
        bail!("pass either media_path or content_base64, not both");
    }
    if content_base64.is_empty() {
        if media_path.is_empty() {
            bail!("media_path or content_base64 is required");
        }
        let path = paths.to_container(media_path);
        let mut file = std::fs::File::open(&path).map_err(|e| anyhow::anyhow!("Media file not found: {path}: {e}"))?;
        // Checked on the file already open, so a path swapped after the check
        // cannot redirect the read.
        let real = opened_path(&file).or_else(|_| std::fs::canonicalize(&path))?;
        if !sendable(paths, allowed, &real) {
            bail!("{path} is outside the directories files can be sent from");
        }
        let meta = file.metadata()?;
        if !meta.is_file() {
            bail!("{path} is not a regular file");
        }
        if meta.len() > MAX_SEND_BYTES {
            bail!("{path} is {} bytes, over the {MAX_SEND_BYTES}-byte limit", meta.len());
        }
        let mut data = Vec::new();
        std::io::Read::read_to_end(&mut file, &mut data)?;
        let name = real.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        return Ok(Outgoing { data, name });
    }
    let name = client_name(filename)?;
    // Clients often send a data: URL, or base64 wrapped at 76 columns.
    let b64 = content_base64.trim();
    let b64 = b64.strip_prefix("data:").and_then(|r| r.split_once(";base64,")).map_or(b64, |(_, rest)| rest);
    let b64: Vec<u8> = b64.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    let data = base64::engine::general_purpose::STANDARD.decode(b64)?;
    Ok(Outgoing { data, name })
}

/// The name of a file the client supplies the bytes of: basename only, and an
/// extension, which picks the media type.
pub fn client_name(filename: &str) -> Result<String> {
    let name = Path::new(filename).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    if !name.trim_matches('.').contains('.') {
        bail!("filename with an extension is required with content_base64 or return_upload_url");
    }
    Ok(name)
}

/// Where an open file really is, symlinks resolved.
fn opened_path(file: &std::fs::File) -> std::io::Result<PathBuf> {
    use std::os::fd::AsRawFd;
    std::fs::read_link(format!("/proc/self/fd/{}", file.as_raw_fd()))
}

/// SQLite files: the session keys and the history.
pub fn is_database(name: &str) -> bool {
    [".db", ".db-wal", ".db-shm", ".db-journal"].iter().any(|ext| name.ends_with(ext))
}

/// A path the model was told to send must not reach the session keys or the
/// history: only files under the send dirs, never a database, never a file in
/// the store root, where the databases live, and under the store only from the
/// folder (`<store>/<chat_jid>/`) of a chat the allowlist lets through.
/// `real` is already canonical.
fn sendable(paths: &PathMap, allowed: &dyn Fn(&str) -> bool, real: &Path) -> bool {
    if is_database(&real.file_name().map(|n| n.to_string_lossy()).unwrap_or_default()) {
        return false;
    }
    let canon = |p: &Path| std::fs::canonicalize(p).ok();
    if let Some(store) = canon(Path::new(&paths.container))
        && let Ok(rest) = real.strip_prefix(&store)
    {
        let mut parts = rest.components();
        let chat = parts.next().map(|c| c.as_os_str().to_string_lossy().into_owned()).unwrap_or_default();
        // One component only: a file in the store root.
        if parts.next().is_none() || !allowed(&chat) {
            return false;
        }
    }
    paths.send_dirs.iter().filter_map(|d| canon(d)).any(|d| real.starts_with(d))
}

/// A `file://` URI for a path, percent-encoding what would break it.
pub fn file_uri(path: &str) -> String {
    let mut uri = String::from("file://");
    for b in path.bytes() {
        match b {
            b' ' | b'#' | b'?' | b'%' | 0..=0x1f | 0x7f.. => uri.push_str(&format!("%{b:02X}")),
            _ => uri.push(b as char),
        }
    }
    uri
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map() -> PathMap {
        PathMap { container: "/app/store".into(), host: "/host/store".into(), send_dirs: vec![] }
    }

    #[test]
    fn paths_round_trip_and_leave_others_alone() {
        let m = map();
        assert_eq!(m.to_host("/app/store/c/x.jpg"), "/host/store/c/x.jpg");
        assert_eq!(m.to_container("/host/store/c/x.jpg"), "/app/store/c/x.jpg");
        assert_eq!(m.to_container("/host/storeX/a.jpg"), "/host/storeX/a.jpg");
        assert_eq!(m.to_host("/app/storeX/a.jpg"), "/app/storeX/a.jpg");
        assert_eq!(m.to_container("/app/envios/a.jpg"), "/app/envios/a.jpg");
        let slash = PathMap { container: "/app/store/".into(), host: "/host/store/".into(), send_dirs: vec![] };
        assert_eq!(slash.to_host("/app/store/c/x.jpg"), "/host/store/c/x.jpg");
        assert_eq!(slash.to_container("/host/store/c/x.jpg"), "/app/store/c/x.jpg");
    }

    #[test]
    fn file_uris_are_percent_encoded() {
        assert_eq!(file_uri("/s/55@s.whatsapp.net/a b#1?%.jpg"), "file:///s/55@s.whatsapp.net/a%20b%231%3F%25.jpg");
        assert_eq!(file_uri("/s/relatório.pdf"), "file:///s/relat%C3%B3rio.pdf");
    }

    #[test]
    fn mime_table_covers_whatsapp_types() {
        for (ext, mime) in [
            ("x.webp", "image/webp"),
            ("x.ogg", "audio/ogg"),
            ("x.jpg", "image/jpeg"),
            ("x.mp4", "video/mp4"),
            ("x.pdf", "application/pdf"),
            ("x.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
            ("x.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
            ("x.m4a", "audio/mp4"),
            ("semextensao", "application/octet-stream"),
        ] {
            assert_eq!(mime_of(Path::new(ext)), mime, "{ext}");
        }
    }

    #[test]
    fn webp_headers_give_size_and_animation() {
        let riff = |chunk: &[u8], body: &[u8]| {
            let mut d = b"RIFF\0\0\0\0WEBP".to_vec();
            d.extend_from_slice(chunk);
            d.extend_from_slice(&[0; 4]);
            d.extend_from_slice(body);
            d.resize(30, 0);
            d
        };
        // VP8X: flags with the animation bit, canvas 512x512 stored minus one.
        assert_eq!(webp_info(&riff(b"VP8X", &[0x02, 0, 0, 0, 0xff, 1, 0, 0xff, 1, 0])), Some((512, 512, true)));
        // VP8 lossy: frame tag, start code, then 14-bit width and height.
        assert_eq!(webp_info(&riff(b"VP8 ", &[0, 0, 0, 0x9d, 0x01, 0x2a, 0, 2, 0, 2])), Some((512, 512, false)));
        // VP8L: signature, then (w-1) and (h-1) packed in 14 bits each.
        let b = (511u32 | 511 << 14).to_le_bytes();
        assert_eq!(webp_info(&riff(b"VP8L", &[0x2f, b[0], b[1], b[2], b[3]])), Some((512, 512, false)));
        assert_eq!(webp_info(b"\x89PNG\r\n\x1a\n not a webp at all, padding"), None);
    }

    #[test]
    fn download_path_is_unique_per_message_and_cannot_escape() {
        let p = download_path(Path::new("/s"), "55@s.whatsapp.net", "AB/../C", "../../etc/passwd");
        assert_eq!(p, PathBuf::from("/s/55@s.whatsapp.net/AB____C-passwd"));
    }

    #[test]
    fn sends_only_from_send_dirs_and_never_databases() {
        let root = tempfile::tempdir().unwrap();
        let store = root.path().join("store");
        let chat = store.join("55@s.whatsapp.net");
        let envios = root.path().join("envios");
        std::fs::create_dir_all(&chat).unwrap();
        std::fs::create_dir_all(&envios).unwrap();
        for f in [store.join("whatsapp-rs.db"), store.join("notes.txt"), chat.join("x.jpg"), chat.join("y.db-wal"), envios.join("a.png")] {
            std::fs::write(f, b"x").unwrap();
        }
        std::fs::write(root.path().join("secret.txt"), b"x").unwrap();
        let m = PathMap { container: store.to_string_lossy().into(), host: String::new(), send_dirs: vec![store.clone(), envios.clone()] };
        let path = |p: PathBuf| p.to_string_lossy().into_owned();
        let any = |_: &str| true;
        assert!(read_outgoing(&m, &any, &path(chat.join("x.jpg")), "", "").is_ok());
        assert!(read_outgoing(&m, &any, &path(envios.join("a.png")), "", "").is_ok());
        // With an allowlist, only allowed chats' folders; other send dirs as before.
        let other = |c: &str| c == "99@s.whatsapp.net";
        assert!(read_outgoing(&m, &other, &path(chat.join("x.jpg")), "", "").is_err());
        assert!(read_outgoing(&m, &other, &path(envios.join("a.png")), "", "").is_ok());
        assert!(read_outgoing(&m, &any, &path(chat.join("x.jpg")), "aGk=", "a.txt").is_err(), "both sources");
        for bad in [
            store.join("whatsapp-rs.db"),
            store.join("notes.txt"),
            chat.join("y.db-wal"),
            root.path().join("secret.txt"),
            chat.join("../whatsapp-rs.db"),
            envios.join("../secret.txt"),
        ] {
            assert!(read_outgoing(&m, &any, &path(bad.clone()), "", "").is_err(), "{}", bad.display());
        }
    }

    #[test]
    fn base64_upload_keeps_only_the_basename_and_needs_an_extension() {
        let b64 = base64::engine::general_purpose::STANDARD.encode(b"%PDF-1.4");
        let any = |_: &str| true;
        let o = read_outgoing(&map(), &any, "", &b64, "../relatorio.pdf").unwrap();
        assert_eq!((o.name.as_str(), o.data.as_slice()), ("relatorio.pdf", &b"%PDF-1.4"[..]));
        let wrapped = format!("data:application/pdf;base64,{}\n{}\r\n", &b64[..4], &b64[4..]);
        assert_eq!(read_outgoing(&map(), &any, "", &wrapped, "a.pdf").unwrap().data, b"%PDF-1.4");
        for (c, f) in [("aGk=", "semextensao"), ("aGk=", ""), ("não é base64", "a.txt")] {
            assert!(read_outgoing(&map(), &any, "", c, f).is_err(), "{f}");
        }
    }
}
