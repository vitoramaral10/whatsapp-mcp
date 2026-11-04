//! Media files on disk: where downloads go, how paths look to the caller, what
//! MIME type a file has, and turning a base64 upload into a file to send.

use std::path::{Path, PathBuf};

use anyhow::{bail, Result};
use base64::Engine;

/// Host/container path rewriting for a bind-mounted store, so a path returned
/// by download_media is one the caller can open, and send_* accepts it back.
#[derive(Clone, Debug, Default)]
pub struct PathMap {
    pub container: String,
    pub host: String,
}

impl PathMap {
    pub fn to_host(&self, path: &str) -> String {
        match path.strip_prefix(&self.container) {
            Some(rest) if !self.host.is_empty() && (rest.is_empty() || rest.starts_with('/')) => {
                format!("{}{}", self.host.trim_end_matches('/'), rest)
            }
            _ => path.to_string(),
        }
    }

    pub fn to_container(&self, path: &str) -> String {
        let host = self.host.trim_end_matches('/');
        match path.strip_prefix(host) {
            Some(rest) if !host.is_empty() && rest.starts_with('/') => format!("{}{}", self.container, rest),
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

/// Keeps a message ID usable as one path segment.
pub fn sanitize(value: &str) -> String {
    value
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect()
}

/// Where a message's media is saved. The stored filename comes from the clock,
/// so many messages share it; the message ID prefix keeps each file distinct.
pub fn download_path(store_dir: &Path, chat_jid: &str, message_id: &str, filename: &str) -> PathBuf {
    let name = Path::new(filename).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    store_dir.join(chat_jid.replace(':', "_")).join(format!("{}-{}", sanitize(message_id), name))
}

/// A file to send: either a path already on this side, or uploaded bytes.
pub struct Outgoing {
    pub data: Vec<u8>,
    /// Basename with extension: picks the media type and is the document title.
    pub name: String,
}

pub fn outgoing(paths: &PathMap, media_path: &str, content_base64: &str, filename: &str) -> Result<Outgoing> {
    if content_base64.is_empty() {
        if media_path.is_empty() {
            bail!("media_path or content_base64 is required");
        }
        let path = paths.to_container(media_path);
        let data = std::fs::read(&path).map_err(|e| anyhow::anyhow!("Media file not found: {path}: {e}"))?;
        let name = Path::new(&path).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        return Ok(Outgoing { data, name });
    }
    let name = Path::new(filename).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    if !name.trim_matches('.').contains('.') {
        bail!("filename with an extension is required with content_base64");
    }
    let data = base64::engine::general_purpose::STANDARD.decode(content_base64.trim())?;
    Ok(Outgoing { data, name })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map() -> PathMap {
        PathMap { container: "/app/store".into(), host: "/host/store".into() }
    }

    #[test]
    fn paths_round_trip_and_leave_others_alone() {
        let m = map();
        assert_eq!(m.to_host("/app/store/c/x.jpg"), "/host/store/c/x.jpg");
        assert_eq!(m.to_container("/host/store/c/x.jpg"), "/app/store/c/x.jpg");
        assert_eq!(m.to_container("/host/storeX/a.jpg"), "/host/storeX/a.jpg");
        assert_eq!(m.to_host("/app/storeX/a.jpg"), "/app/storeX/a.jpg");
        assert_eq!(m.to_container("/app/envios/a.jpg"), "/app/envios/a.jpg");
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
    fn download_path_is_unique_per_message_and_cannot_escape() {
        let p = download_path(Path::new("/s"), "55@s.whatsapp.net", "AB/../C", "../../etc/passwd");
        assert_eq!(p, PathBuf::from("/s/55@s.whatsapp.net/AB____C-passwd"));
    }

    #[test]
    fn base64_upload_keeps_only_the_basename_and_needs_an_extension() {
        let b64 = base64::engine::general_purpose::STANDARD.encode(b"%PDF-1.4");
        let o = outgoing(&map(), "", &b64, "../relatorio.pdf").unwrap();
        assert_eq!((o.name.as_str(), o.data.as_slice()), ("relatorio.pdf", &b"%PDF-1.4"[..]));
        for (c, f) in [("aGk=", "semextensao"), ("aGk=", ""), ("não é base64", "a.txt")] {
            assert!(outgoing(&map(), "", c, f).is_err(), "{f}");
        }
    }
}
