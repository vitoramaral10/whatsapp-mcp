//! Text and attachment out of a WhatsApp protobuf message.
//!
//! Beyond plain chats this unwraps container messages (ephemeral, view-once,
//! edited, device-sent) and flattens the interactive types business bots use
//! for menus: without that a menu has no `conversation` field and is dropped,
//! so the chat reads as if the bot answered nothing.

use chrono::{DateTime, Utc};
use whatsapp_rust::prelude::wa;

use crate::store::Media;

/// Non-empty pieces of a rendered message, one per line.
#[derive(Default)]
struct Lines(Vec<String>);

impl Lines {
    fn add(&mut self, text: impl AsRef<str>) {
        let t = text.as_ref().trim();
        if !t.is_empty() {
            self.0.push(t.to_string());
        }
    }
    fn done(self) -> Option<String> {
        (!self.0.is_empty()).then(|| self.0.join("\n"))
    }
}

/// "- label: detail [id]": bots that use native menus expect the id back.
fn bullet(label: &str, detail: &str, id: &str) -> String {
    let (label, id) = match (label.is_empty(), id.is_empty()) {
        (true, true) => return String::new(),
        (true, false) => (id, ""),
        _ => (label, id),
    };
    let mut line = format!("- {label}");
    if !detail.is_empty() {
        line += &format!(": {detail}");
    }
    if !id.is_empty() {
        line += &format!(" [{id}]");
    }
    line
}

fn s(o: &Option<String>) -> &str {
    o.as_deref().unwrap_or("")
}

/// The messages a container wraps, innermost payload one level down.
fn inner(m: &wa::Message) -> impl Iterator<Item = &wa::Message> {
    [
        m.ephemeral_message.as_option().and_then(|f| f.message.as_option()),
        m.view_once_message.as_option().and_then(|f| f.message.as_option()),
        m.view_once_message_v2.as_option().and_then(|f| f.message.as_option()),
        m.view_once_message_v2_extension.as_option().and_then(|f| f.message.as_option()),
        m.document_with_caption_message.as_option().and_then(|f| f.message.as_option()),
        m.edited_message.as_option().and_then(|f| f.message.as_option()),
        m.device_sent_message.as_option().and_then(|f| f.message.as_option()),
    ]
    .into_iter()
    .flatten()
}

pub fn text(m: &wa::Message) -> String {
    text_depth(m, 0).unwrap_or_default()
}

fn text_depth(m: &wa::Message, depth: u8) -> Option<String> {
    // Guards against a malformed message nesting containers forever.
    if depth > 4 {
        return None;
    }
    if let Some(t) = inner(m).find_map(|i| text_depth(i, depth + 1)) {
        return Some(t);
    }
    let nonempty = |t: &str| (!t.is_empty()).then(|| t.to_string());
    if let Some(t) = m.conversation.as_deref().and_then(nonempty) {
        return Some(t);
    }
    if let Some(t) = m.extended_text_message.as_option().and_then(|x| x.text.as_deref()).and_then(nonempty) {
        return Some(t);
    }
    // Media captions: the attachment is recorded separately.
    for caption in [
        m.image_message.as_option().and_then(|x| x.caption.as_deref()),
        m.video_message.as_option().and_then(|x| x.caption.as_deref()),
        m.ptv_message.as_option().and_then(|x| x.caption.as_deref()),
        m.document_message.as_option().and_then(|x| x.caption.as_deref()),
    ]
    .into_iter()
    .flatten()
    {
        if !caption.is_empty() {
            return Some(caption.to_string());
        }
    }
    if let Some(list) = m.list_message.as_option() {
        let mut b = Lines::default();
        b.add(s(&list.title));
        b.add(s(&list.description));
        for section in &list.sections {
            b.add(s(&section.title));
            for row in &section.rows {
                b.add(bullet(s(&row.title), s(&row.description), s(&row.row_id)));
            }
        }
        b.add(s(&list.button_text));
        b.add(s(&list.footer_text));
        if let Some(t) = b.done() {
            return Some(t);
        }
    }
    if let Some(buttons) = m.buttons_message.as_option() {
        let mut b = Lines::default();
        b.add(s(&buttons.content_text));
        for button in &buttons.buttons {
            let label = button.button_text.as_option().map(|t| s(&t.display_text)).unwrap_or("");
            b.add(bullet(label, "", s(&button.button_id)));
        }
        b.add(s(&buttons.footer_text));
        if let Some(t) = b.done() {
            return Some(t);
        }
    }
    if let Some(t) = m.template_message.as_option().and_then(render_template) {
        return Some(t);
    }
    if let Some(t) = m.interactive_message.as_option().and_then(render_interactive) {
        return Some(t);
    }
    // Replies to those menus: outgoing messages are stored too.
    if let Some(r) = m.list_response_message.as_option() {
        let mut b = Lines::default();
        b.add(s(&r.title));
        b.add(s(&r.description));
        b.add(r.single_select_reply.as_option().map(|x| s(&x.selected_row_id)).unwrap_or(""));
        if let Some(t) = b.done() {
            return Some(t);
        }
    }
    if let Some(r) = m.buttons_response_message.as_option() {
        use wa::message::buttons_response_message::Response;
        let shown = match &r.response {
            Some(Response::SelectedDisplayText(t)) => t.as_str(),
            _ => "",
        };
        if let Some(t) = [shown, s(&r.selected_button_id)].into_iter().find(|t| !t.is_empty()) {
            return Some(t.to_string());
        }
    }
    if let Some(r) = m.template_button_reply_message.as_option() {
        if let Some(t) = [s(&r.selected_display_text), s(&r.selected_id)].into_iter().find(|t| !t.is_empty()) {
            return Some(t.to_string());
        }
    }
    if let Some(r) = m.interactive_response_message.as_option() {
        use wa::message::interactive_response_message::InteractiveResponseMessage as Kind;
        let body = r.body.as_option().map(|b| s(&b.text)).unwrap_or("");
        let params = match &r.interactive_response_message {
            Some(Kind::NativeFlowResponseMessage(n)) => s(&n.params_json),
            _ => "",
        };
        if let Some(t) = [body, params].into_iter().find(|t| !t.is_empty()) {
            return Some(t.to_string());
        }
    }
    // Polls read like a menu and were dropped for the same reason.
    for poll in [
        m.poll_creation_message.as_option(),
        m.poll_creation_message_v2.as_option(),
        m.poll_creation_message_v3.as_option(),
        m.poll_creation_message_v5.as_option(),
        m.poll_creation_message_v6.as_option(),
    ]
    .into_iter()
    .flatten()
    {
        let mut b = Lines::default();
        b.add(s(&poll.name));
        for o in &poll.options {
            b.add(bullet(s(&o.option_name), "", ""));
        }
        if let Some(t) = b.done() {
            return Some(t);
        }
    }
    // A clinic asked for its address answers with a pin or a vCard.
    if let Some(l) = m.location_message.as_option() {
        let mut b = Lines::default();
        b.add(s(&l.name));
        b.add(s(&l.address));
        b.add(format!("{:.6}, {:.6}", l.degrees_latitude.unwrap_or(0.0), l.degrees_longitude.unwrap_or(0.0)));
        return b.done();
    }
    if let Some(c) = m.contact_message.as_option() {
        let mut b = Lines::default();
        b.add(s(&c.display_name));
        b.add(s(&c.vcard));
        if let Some(t) = b.done() {
            return Some(t);
        }
    }
    None
}

fn render_template(t: &wa::message::TemplateMessage) -> Option<String> {
    use wa::message::template_message::Format;
    let hydrated = t.hydrated_template.as_option().or(match &t.format {
        Some(Format::HydratedFourRowTemplate(h)) => Some(h.as_ref()),
        _ => None,
    });
    let Some(h) = hydrated else {
        // Newer templates carry an interactive message instead.
        return match &t.format {
            Some(Format::InteractiveMessageTemplate(i)) => render_interactive(i),
            _ => None,
        };
    };
    use wa::message::template_message::hydrated_four_row_template::Title;
    let mut b = Lines::default();
    if let Some(Title::HydratedTitleText(title)) = &h.title {
        b.add(title);
    }
    b.add(s(&h.hydrated_content_text));
    for button in &h.hydrated_buttons {
        use wa::hydrated_template_button::HydratedButton;
        match &button.hydrated_button {
            Some(HydratedButton::QuickReplyButton(q)) => b.add(bullet(s(&q.display_text), "", s(&q.id))),
            Some(HydratedButton::UrlButton(u)) => b.add(bullet(s(&u.display_text), s(&u.url), "")),
            Some(HydratedButton::CallButton(c)) => b.add(bullet(s(&c.display_text), s(&c.phone_number), "")),
            _ => {}
        }
    }
    b.add(s(&h.hydrated_footer_text));
    b.done()
}

fn render_interactive(i: &wa::message::InteractiveMessage) -> Option<String> {
    use wa::message::interactive_message::InteractiveMessage as Kind;
    let mut b = Lines::default();
    if let Some(h) = i.header.as_option() {
        b.add(s(&h.title));
        b.add(s(&h.subtitle));
    }
    b.add(i.body.as_option().map(|x| s(&x.text)).unwrap_or(""));
    if let Some(Kind::NativeFlowMessage(flow)) = &i.interactive_message {
        for button in &flow.buttons {
            b.add(native_flow_button(s(&button.name), s(&button.button_params_json)));
        }
    }
    b.add(i.footer.as_option().map(|x| s(&x.text)).unwrap_or(""));
    b.done()
}

/// A native flow button carries loose JSON whose shape depends on its name
/// (quick_reply, cta_url, single_select); every field is optional.
fn native_flow_button(name: &str, raw: &str) -> String {
    if raw.is_empty() {
        return bullet(name, "", "");
    }
    let Ok(p) = serde_json::from_str::<serde_json::Value>(raw) else {
        // Better a raw payload in the history than an option that vanishes.
        return bullet(name, raw, "");
    };
    let f = |v: &serde_json::Value, k: &str| v.get(k).and_then(|x| x.as_str()).unwrap_or("").to_string();
    let mut b = Lines::default();
    b.add(bullet(&f(&p, "display_text"), &f(&p, "url"), &f(&p, "id")));
    b.add(f(&p, "title"));
    for section in p.get("sections").and_then(|x| x.as_array()).into_iter().flatten() {
        b.add(f(section, "title"));
        for row in section.get("rows").and_then(|x| x.as_array()).into_iter().flatten() {
            b.add(bullet(&f(row, "title"), &f(row, "description"), &f(row, "id")));
        }
    }
    b.done().unwrap_or_else(|| bullet(name, "", ""))
}

/// The attachment, if any. The filename is derived from the message time so a
/// replayed message gets the same name.
pub fn media(m: &wa::Message, ts: DateTime<Utc>) -> Media {
    media_depth(m, ts, 0)
}

fn media_depth(m: &wa::Message, ts: DateTime<Utc>, depth: u8) -> Media {
    if depth > 4 {
        return Media::default();
    }
    if let Some(found) = inner(m).map(|i| media_depth(i, ts, depth + 1)).find(|x| !x.kind.is_empty()) {
        return found;
    }
    let stamp = ts.format("%Y%m%d_%H%M%S");
    macro_rules! media {
        ($x:expr, $kind:expr, $name:expr) => {
            Media {
                kind: $kind.into(),
                filename: $name,
                url: $x.url.clone().unwrap_or_default(),
                direct_path: $x.direct_path.clone().unwrap_or_default(),
                media_key: $x.media_key.clone().unwrap_or_default(),
                file_sha256: $x.file_sha256.clone().unwrap_or_default(),
                file_enc_sha256: $x.file_enc_sha256.clone().unwrap_or_default(),
                file_length: $x.file_length.unwrap_or(0),
            }
        };
    }
    if let Some(x) = m.image_message.as_option() {
        let ext = ext_for(s(&x.mimetype), "jpg");
        return media!(x, "image", format!("image_{stamp}.{ext}"));
    }
    if let Some(x) = m.video_message.as_option().or(m.ptv_message.as_option()) {
        return media!(x, "video", format!("video_{stamp}.mp4"));
    }
    if let Some(x) = m.audio_message.as_option() {
        let ext = ext_for(s(&x.mimetype), "ogg");
        return media!(x, "audio", format!("audio_{stamp}.{ext}"));
    }
    if let Some(x) = m.document_message.as_option() {
        let name = x.file_name.clone().filter(|n| !n.is_empty()).unwrap_or_else(|| {
            format!("document_{stamp}.{}", ext_for(s(&x.mimetype), "bin"))
        });
        return media!(x, "document", name);
    }
    if let Some(x) = m.sticker_message.as_option() {
        return media!(x, "sticker", format!("sticker_{stamp}.webp"));
    }
    Media::default()
}

/// Extension from the message's own MIME type, so a PNG or MP3 is not
/// labelled .jpg/.ogg.
fn ext_for(mime: &str, default: &str) -> String {
    let base = mime.split(';').next().unwrap_or("").trim();
    match base {
        "image/jpeg" | "" => default.into(),
        "audio/ogg" | "audio/opus" => "ogg".into(),
        "audio/mpeg" => "mp3".into(),
        "audio/mp4" | "audio/aac" | "audio/x-m4a" => "m4a".into(),
        other => mime_guess::get_mime_extensions_str(other)
            .and_then(|e| e.first())
            .map(|e| e.to_string())
            .unwrap_or_else(|| default.into()),
    }
}

/// Protobuf fields actually set, so a dropped message names what is missing.
pub fn describe(m: &wa::Message) -> String {
    let dbg = format!("{m:?}");
    // Debug output lists `field: Some(..)`/`field: MessageField(Some(..))`; cheap
    // and good enough for a debug log line.
    let names: Vec<&str> = dbg
        .split(", ")
        .filter_map(|p| {
            let (k, v) = p.split_once(": ")?;
            (v.starts_with("Some") || v.contains("(Some")).then(|| k.trim_start_matches("Message { ").trim())
        })
        .collect();
    if names.is_empty() { "<empty>".into() } else { names.join(",") }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;
    use whatsapp_rust::prelude::MessageField;

    fn ts() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 11, 3, 14, 2, 32).unwrap()
    }

    #[test]
    fn plain_and_wrapped_text() {
        let plain = wa::Message { conversation: Some("oi".into()), ..Default::default() };
        assert_eq!(text(&plain), "oi");
        let wrapped = wa::Message {
            ephemeral_message: MessageField::some(wa::message::FutureProofMessage {
                message: MessageField::some(plain.clone()),
                ..Default::default()
            }),
            ..Default::default()
        };
        assert_eq!(text(&wrapped), "oi");
    }

    #[test]
    fn image_with_caption_and_png_mime() {
        let m = wa::Message {
            image_message: MessageField::some(wa::message::ImageMessage {
                caption: Some("olha".into()),
                mimetype: Some("image/png".into()),
                direct_path: Some("/v/x".into()),
                file_length: Some(3),
                ..Default::default()
            }),
            ..Default::default()
        };
        assert_eq!(text(&m), "olha");
        let media = media(&m, ts());
        assert_eq!((media.kind.as_str(), media.filename.as_str(), media.direct_path.as_str()), ("image", "image_20261103_140232.png", "/v/x"));
    }

    #[test]
    fn voice_note_document_and_sticker_names() {
        let audio = wa::Message {
            audio_message: MessageField::some(wa::message::AudioMessage { mimetype: Some("audio/ogg; codecs=opus".into()), ..Default::default() }),
            ..Default::default()
        };
        assert_eq!(media(&audio, ts()).filename, "audio_20261103_140232.ogg");
        let doc = wa::Message {
            document_message: MessageField::some(wa::message::DocumentMessage { file_name: Some("nota.pdf".into()), ..Default::default() }),
            ..Default::default()
        };
        assert_eq!(media(&doc, ts()).filename, "nota.pdf");
        let sticker = wa::Message { sticker_message: MessageField::some(Default::default()), ..Default::default() };
        assert_eq!(media(&sticker, ts()).kind, "sticker");
    }

    #[test]
    fn list_menu_and_native_flow() {
        let list = wa::Message {
            list_message: MessageField::some(wa::message::ListMessage {
                title: Some("Menu".into()),
                sections: vec![wa::message::list_message::Section {
                    title: Some("Opções".into()),
                    rows: vec![wa::message::list_message::Row { title: Some("Endereço".into()), row_id: Some("1".into()), ..Default::default() }],
                    ..Default::default()
                }],
                ..Default::default()
            }),
            ..Default::default()
        };
        assert_eq!(text(&list), "Menu\nOpções\n- Endereço [1]");
        assert_eq!(native_flow_button("quick_reply", r#"{"display_text":"Sim","id":"y"}"#), "- Sim [y]");
        assert_eq!(native_flow_button("x", "{bad"), "- x: {bad");
    }

    #[test]
    fn nothing_readable_is_empty() {
        assert_eq!(text(&wa::Message::default()), "");
        assert!(media(&wa::Message::default(), ts()).kind.is_empty());
    }
}
