//! Audio without ffmpeg: Ogg/Opus through libopus (statically built), every
//! other format through symphonia. Covers the two jobs the old Python code
//! shelled out for: a 16 kHz WAV for whisper, and an Ogg/Opus voice note for
//! sending, with its real duration and waveform.

use std::io::Cursor;

use anyhow::{Context, Result, anyhow, bail};

const OPUS_RATE: u32 = 48_000;
/// Longest audio decoded. Voice notes from anyone are decoded on arrival, and
/// a few KB of crafted Opus or FLAC can otherwise expand into gigabytes of PCM.
pub const MAX_SECONDS: u64 = 30 * 60;
/// Largest input decoded at all.
pub const MAX_BYTES: usize = 100 << 20;
/// Above this a declared sample rate is bogus, and would defeat MAX_SECONDS.
const MAX_RATE: u32 = 192_000;

/// Mono PCM at some sample rate.
pub struct Pcm {
    pub rate: u32,
    pub samples: Vec<f32>,
}

impl Pcm {
    pub fn seconds(&self) -> f64 {
        self.samples.len() as f64 / self.rate as f64
    }
}

pub fn is_ogg(data: &[u8]) -> bool {
    data.starts_with(b"OggS")
}

/// An Ogg stream whose first packet is an Opus header.
pub fn is_ogg_opus(data: &[u8]) -> bool {
    is_ogg(data) && ogg::PacketReader::new(Cursor::new(data)).read_packet().ok().flatten().is_some_and(|p| p.data.starts_with(b"OpusHead"))
}

/// Decodes any supported container to mono PCM.
pub fn decode(data: &[u8], ext_hint: Option<&str>) -> Result<Pcm> {
    decode_up_to(data, ext_hint, MAX_SECONDS)
}

fn decode_up_to(data: &[u8], ext_hint: Option<&str>, max_seconds: u64) -> Result<Pcm> {
    if data.len() > MAX_BYTES {
        bail!("audio is {} bytes, over the {MAX_BYTES}-byte limit", data.len());
    }
    if is_ogg_opus(data) {
        return decode_ogg_opus(data, max_seconds);
    }
    decode_symphonia(data, ext_hint, max_seconds)
}

/// Errors as soon as more than `max_seconds` of audio came out. Counts frames
/// (mono samples after the downmix), the same as samples / channels.
fn check_length(frames: usize, rate: u32, max_seconds: u64) -> Result<()> {
    if frames as u64 > rate as u64 * max_seconds {
        bail!("audio is longer than {max_seconds} seconds");
    }
    Ok(())
}

fn decode_ogg_opus(data: &[u8], max_seconds: u64) -> Result<Pcm> {
    let mut reader = ogg::PacketReader::new(Cursor::new(data));
    let head = reader.read_packet()?.ok_or_else(|| anyhow!("empty ogg stream"))?;
    if !head.data.starts_with(b"OpusHead") || head.data.len() < 19 {
        bail!("not an Opus stream");
    }
    let channels = head.data[9].clamp(1, 2);
    let pre_skip = u16::from_le_bytes([head.data[10], head.data[11]]) as usize;
    let _tags = reader.read_packet()?;

    let ch = if channels == 2 { opus::Channels::Stereo } else { opus::Channels::Mono };
    let mut decoder = opus::Decoder::new(OPUS_RATE, ch)?;
    let mut frame = vec![0f32; 5760 * channels as usize];
    let mut samples = Vec::new();
    while let Some(packet) = reader.read_packet()? {
        let n = decoder.decode_float(&packet.data, &mut frame, false)?;
        if channels == 2 {
            samples.extend(frame[..n * 2].as_chunks::<2>().0.iter().map(|[l, r]| (l + r) * 0.5));
        } else {
            samples.extend_from_slice(&frame[..n]);
        }
        check_length(samples.len(), OPUS_RATE, max_seconds)?;
    }
    samples.drain(..pre_skip.min(samples.len()));
    Ok(Pcm { rate: OPUS_RATE, samples })
}

fn decode_symphonia(data: &[u8], ext_hint: Option<&str>, max_seconds: u64) -> Result<Pcm> {
    use symphonia::core::{
        audio::{SampleBuffer, SignalSpec},
        codecs::{CODEC_TYPE_NULL, DecoderOptions},
        formats::FormatOptions,
        io::MediaSourceStream,
        meta::MetadataOptions,
        probe::Hint,
    };
    let mss = MediaSourceStream::new(Box::new(Cursor::new(data.to_vec())), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = ext_hint {
        hint.with_extension(ext);
    }
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .context("unsupported audio format")?;
    let mut format = probed.format;
    // The first real audio track: default_track() may be a cover image or a null codec.
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL && t.codec_params.sample_rate.is_some())
        .ok_or_else(|| anyhow!("no audio track"))?
        .clone();
    let rate = track.codec_params.sample_rate.unwrap_or_default();
    if !(1..=MAX_RATE).contains(&rate) {
        bail!("unsupported sample rate {rate}");
    }
    let mut decoder = symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default())?;
    let mut samples = Vec::new();
    let mut buf: Option<(SampleBuffer<f32>, SignalSpec)> = None;
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        };
        if packet.track_id() != track.id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(e) => return Err(e.into()),
        };
        let spec = *decoded.spec();
        let ch = spec.channels.count().max(1);
        // capacity() is frames for the decoded buffer, samples for SampleBuffer.
        if buf.as_ref().is_none_or(|(b, s)| *s != spec || b.capacity() < decoded.capacity() * ch) {
            buf = Some((SampleBuffer::new(decoded.capacity() as u64, spec), spec));
        }
        let (b, _) = buf.as_mut().expect("set above");
        b.copy_interleaved_ref(decoded);
        samples.extend(b.samples().chunks_exact(ch).map(|f| f.iter().sum::<f32>() / ch as f32));
        check_length(samples.len(), rate, max_seconds)?;
    }
    Ok(Pcm { rate, samples })
}

/// Linear resampling. Good enough for speech recognition and for Opus input,
/// both of which band-limit well below the new Nyquist rate anyway.
pub fn resample(pcm: &Pcm, rate: u32) -> Pcm {
    if pcm.rate == rate || pcm.samples.is_empty() {
        return Pcm { rate, samples: pcm.samples.clone() };
    }
    let ratio = pcm.rate as f64 / rate as f64;
    let n = (pcm.samples.len() as f64 / ratio) as usize;
    let last = pcm.samples.len() - 1;
    let samples = (0..n)
        .map(|i| {
            let pos = i as f64 * ratio;
            let j = pos as usize;
            let frac = (pos - j as f64) as f32;
            let a = pcm.samples[j.min(last)];
            let b = pcm.samples[(j + 1).min(last)];
            a + (b - a) * frac
        })
        .collect();
    Pcm { rate, samples }
}

/// 16-bit PCM WAV, the only input whisper-server accepts without --convert.
pub fn to_wav(pcm: &Pcm) -> Vec<u8> {
    let data_len = (pcm.samples.len() * 2) as u32;
    let mut out = Vec::with_capacity(44 + data_len as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&pcm.rate.to_le_bytes());
    out.extend_from_slice(&(pcm.rate * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for s in &pcm.samples {
        out.extend_from_slice(&((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16).to_le_bytes());
    }
    out
}

/// A voice note ready to send: Ogg/Opus bytes plus what AudioMessage needs.
pub struct VoiceNote {
    pub ogg: Vec<u8>,
    pub seconds: u32,
    pub waveform: Vec<u8>,
}

/// Re-encodes anything decodable as a 48 kHz mono Opus voice note. Ogg/Opus
/// input is kept as is; only its duration and waveform are computed.
pub fn voice_note(data: &[u8], ext_hint: Option<&str>) -> Result<VoiceNote> {
    let pcm = resample(&decode(data, ext_hint)?, OPUS_RATE);
    if pcm.samples.is_empty() {
        bail!("audio has no samples");
    }
    let ogg = if is_ogg_opus(data) { data.to_vec() } else { encode_ogg_opus(&pcm)? };
    Ok(VoiceNote { seconds: pcm.seconds().ceil().max(1.0) as u32, waveform: waveform(&pcm), ogg })
}

fn encode_ogg_opus(pcm: &Pcm) -> Result<Vec<u8>> {
    const FRAME: usize = 960; // 20 ms at 48 kHz
    let mut enc = opus::Encoder::new(OPUS_RATE, opus::Channels::Mono, opus::Application::Voip)?;
    enc.set_bitrate(opus::Bitrate::Bits(32_000))?;
    let pre_skip = enc.get_lookahead()? as u16;
    let serial = 0x5741_4d43; // any constant: one logical stream per file
    let mut out = Vec::new();
    let mut w = ogg::PacketWriter::new(&mut out);

    let mut head = b"OpusHead".to_vec();
    head.push(1);
    head.push(1);
    head.extend_from_slice(&pre_skip.to_le_bytes());
    head.extend_from_slice(&OPUS_RATE.to_le_bytes());
    head.extend_from_slice(&0u16.to_le_bytes());
    head.push(0);
    w.write_packet(head, serial, ogg::PacketWriteEndInfo::EndPage, 0)?;
    let mut tags = b"OpusTags".to_vec();
    let vendor = b"whatsapp-mcp";
    tags.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
    tags.extend_from_slice(vendor);
    tags.extend_from_slice(&0u32.to_le_bytes());
    w.write_packet(tags, serial, ogg::PacketWriteEndInfo::EndPage, 0)?;

    let mut buf = vec![0u8; 4000];
    let chunks: Vec<&[f32]> = pcm.samples.chunks(FRAME).collect();
    let mut granule = pre_skip as u64;
    for (i, chunk) in chunks.iter().enumerate() {
        let mut frame = [0f32; FRAME];
        frame[..chunk.len()].copy_from_slice(chunk);
        let n = enc.encode_float(&frame, &mut buf)?;
        granule += chunk.len() as u64;
        let end = if i + 1 == chunks.len() { ogg::PacketWriteEndInfo::EndStream } else { ogg::PacketWriteEndInfo::NormalPacket };
        w.write_packet(buf[..n].to_vec(), serial, end, granule)?;
    }
    drop(w);
    Ok(out)
}

/// 64 loudness buckets scaled to 0..=100, the shape WhatsApp draws.
pub fn waveform(pcm: &Pcm) -> Vec<u8> {
    const BARS: usize = 64;
    let per = (pcm.samples.len() / BARS).max(1);
    let rms: Vec<f32> = (0..BARS)
        .map(|i| {
            let s = &pcm.samples[(i * per).min(pcm.samples.len())..((i + 1) * per).min(pcm.samples.len())];
            if s.is_empty() { 0.0 } else { (s.iter().map(|x| x * x).sum::<f32>() / s.len() as f32).sqrt() }
        })
        .collect();
    let peak = rms.iter().cloned().fold(0.0f32, f32::max);
    rms.iter().map(|r| if peak > 0.0 { (r / peak * 100.0).round() as u8 } else { 0 }).collect()
}

/// WAV at 16 kHz for whisper.
pub fn whisper_wav(data: &[u8], ext_hint: Option<&str>) -> Result<Vec<u8>> {
    let pcm = decode(data, ext_hint)?;
    if pcm.samples.is_empty() {
        bail!("audio has no samples");
    }
    Ok(to_wav(&resample(&pcm, 16_000)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(seconds: f32, rate: u32) -> Pcm {
        let n = (seconds * rate as f32) as usize;
        Pcm { rate, samples: (0..n).map(|i| (i as f32 * 440.0 * std::f32::consts::TAU / rate as f32).sin() * 0.5).collect() }
    }

    #[test]
    fn opus_round_trip_keeps_duration() {
        let ogg = encode_ogg_opus(&tone(2.5, OPUS_RATE)).unwrap();
        assert!(is_ogg(&ogg));
        let note = voice_note(&ogg, Some("ogg")).unwrap();
        assert_eq!(note.seconds, 3);
        assert_eq!(note.waveform.len(), 64);
        assert_eq!(note.ogg, ogg, "ogg/opus input is sent untouched");
        let back = decode(&ogg, None).unwrap();
        assert!((back.seconds() - 2.5).abs() < 0.05, "{}", back.seconds());
    }

    #[test]
    fn wav_input_becomes_voice_note_and_whisper_wav() {
        let wav = to_wav(&tone(1.0, 22_050));
        let note = voice_note(&wav, Some("wav")).unwrap();
        assert!(is_ogg(&note.ogg));
        assert_eq!(note.seconds, 1);
        let w = whisper_wav(&wav, Some("wav")).unwrap();
        assert_eq!(u32::from_le_bytes(w[24..28].try_into().unwrap()), 16_000);
        assert!((w.len() - 44) / 2 >= 15_900);
    }

    #[test]
    fn long_or_huge_input_stops_decoding() {
        let opus = encode_ogg_opus(&tone(2.5, OPUS_RATE)).unwrap();
        let wav = to_wav(&tone(2.5, 22_050));
        for (data, ext) in [(&opus, "ogg"), (&wav, "wav")] {
            let e = decode_up_to(data, Some(ext), 1).err().expect(ext).to_string();
            assert!(e.contains("longer than 1 seconds"), "{ext}: {e}");
            assert!(decode_up_to(data, Some(ext), 3).is_ok(), "{ext}");
        }
        let e = decode(&vec![0u8; MAX_BYTES + 1], None).err().unwrap().to_string();
        assert!(e.contains("limit"), "{e}");
    }

    #[test]
    fn opus_is_detected_by_the_first_packet_only() {
        assert!(is_ogg_opus(&encode_ogg_opus(&tone(0.1, OPUS_RATE)).unwrap()));
        // "OpusHead" past the first packet (in a tag, say) is not an Opus stream.
        let mut fake = Vec::new();
        let mut w = ogg::PacketWriter::new(&mut fake);
        w.write_packet(b"\x01vorbis".to_vec(), 1, ogg::PacketWriteEndInfo::EndPage, 0).unwrap();
        w.write_packet(b"OpusHead".to_vec(), 1, ogg::PacketWriteEndInfo::EndStream, 0).unwrap();
        drop(w);
        assert!(is_ogg(&fake) && !is_ogg_opus(&fake));
    }

    #[test]
    fn garbage_is_an_error_not_a_panic() {
        assert!(whisper_wav(b"not audio at all", None).is_err());
        assert!(voice_note(b"OggS garbage", Some("ogg")).is_err());
    }
}

#[cfg(test)]
mod real_media {
    /// Set WA_SAMPLE_OGG to a real voice note to run this.
    #[test]
    fn decodes_a_real_whatsapp_voice_note() {
        let Ok(path) = std::env::var("WA_SAMPLE_OGG") else { return };
        let data = std::fs::read(path).unwrap();
        let pcm = super::decode(&data, Some("ogg")).unwrap();
        let note = super::voice_note(&data, Some("ogg")).unwrap();
        eprintln!("seconds={:.2} note={}s wav={}B", pcm.seconds(), note.seconds, super::whisper_wav(&data, Some("ogg")).unwrap().len());
        assert!((pcm.seconds() - 16.7).abs() < 0.2);
    }
}
