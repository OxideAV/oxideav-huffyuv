//! Minimal RIFF / AVI 1.0 walker for HuffYUV fixture decode.
//!
//! This is a **pure-Rust, dep-free** AVI reader covering only the
//! subset the round-3 lockstep test harness needs: locate the
//! `BITMAPINFOHEADER` (parsed by [`crate::header::StreamConfig`]),
//! enumerate `00dc` frame chunks in the `movi` list, and hand each
//! frame's bytes to [`crate::decoder::decode_frame`].
//!
//! References:
//!
//! - **AVI RIFF Form** (Microsoft "AVI File Format" / `vfw.h`):
//!   `RIFF { AVI ( hdrl ( avih ( strl ( strh strf ) ) ) movi ( ##dc ) idx1 ) }`
//! - **RIFF chunk format**: `<4-byte FourCC><4-byte little-endian
//!   size><payload>[<1-byte pad if size is odd>]`. List chunks
//!   (`LIST` / `RIFF`) have an extra 4-byte sub-fourcc at the start
//!   of their payload.
//!
//! The walker is intentionally tiny (≈ 200 LOC). It accepts any
//! `LIST hdrl` ordering, ignores chunks it doesn't need, and rejects
//! anything that isn't well-formed RIFF (so a corrupt fixture
//! produces an [`Error::InvalidData`] rather than a panic).
//!
//! # Limitations
//!
//! - Only the **first video stream** (`vids` stream type) is
//!   surfaced. Multi-stream files (audio + video) work but only the
//!   video is decoded.
//! - The `idx1` index chunk is not consulted; we walk the `movi`
//!   list linearly. AVI 2.0 (`OpenDML`) extension lists (`AVIX`) are
//!   recognised at the top level but their entries are not yet
//!   surfaced (single-segment HuffYUV files don't need them).
//! - Audio chunks (`##wb`, `##pc`, etc.) are skipped silently.
//!
//! # Forbidden references
//!
//! The implementation is a clean-room translation of Microsoft's
//! public `vfw.h` / `aviriff.h` enumerations and the public RIFF
//! specification only. **No** `libavformat/avi*.c`,
//! `libavformat/riff*.c`, `MPlayer/libmpdemux/demux_avi.c`,
//! `vlc/modules/demux/avi/*`, or any other third-party AVI demux
//! source was consulted.

use crate::error::{Error, Result};

/// Top-level RIFF FourCC `'RIFF'`.
const FOURCC_RIFF: [u8; 4] = *b"RIFF";
/// Top-level RIFF form id `'AVI '`.
const FOURCC_AVI: [u8; 4] = *b"AVI ";
/// AVI extension RIFF (`'RIFF' size 'AVIX'` — OpenDML).
const FOURCC_AVIX: [u8; 4] = *b"AVIX";
/// LIST chunk FourCC.
const FOURCC_LIST: [u8; 4] = *b"LIST";
/// LIST sub-fourcc for the header list.
const FOURCC_HDRL: [u8; 4] = *b"hdrl";
/// LIST sub-fourcc for a per-stream list.
const FOURCC_STRL: [u8; 4] = *b"strl";
/// LIST sub-fourcc for the movi (video data) list.
const FOURCC_MOVI: [u8; 4] = *b"movi";
/// Stream-header chunk inside `strl`.
const FOURCC_STRH: [u8; 4] = *b"strh";
/// Stream-format chunk inside `strl` — the BITMAPINFOHEADER lives
/// here for video streams.
const FOURCC_STRF: [u8; 4] = *b"strf";
/// `vids` stream type (in the `strh` payload's first 4 bytes).
const STREAM_TYPE_VIDS: [u8; 4] = *b"vids";

/// One video frame extracted from an AVI `movi` list entry.
#[derive(Debug, Clone)]
pub struct AviFrame<'a> {
    /// AVI chunk id — `b"##dc"` for compressed video frames where
    /// the leading `##` is the 2-digit stream index. We accept
    /// `00dc`, `00db`, and any chunk whose third byte is `b'd'` and
    /// fourth byte is either `b'c'` or `b'b'` (compressed / DIB).
    pub chunk_id: [u8; 4],
    /// Raw payload bytes (does NOT include the trailing pad byte
    /// that RIFF inserts after odd-length chunks).
    pub payload: &'a [u8],
}

/// One parsed AVI file's video stream metadata + frame iteration.
#[derive(Debug)]
pub struct AviVideoStream<'a> {
    /// The `strf` payload — i.e. the bytes that
    /// [`crate::header::StreamConfig::parse_bitmapinfoheader`]
    /// expects.
    pub bitmap_info_header: &'a [u8],
    /// The complete `movi` list payload — the caller iterates it
    /// via [`AviVideoStream::frames`].
    pub movi_payload: &'a [u8],
}

impl<'a> AviVideoStream<'a> {
    /// Walk the file's top-level RIFF and locate the first `vids`
    /// stream. Returns `Ok(None)` if the file parses but contains no
    /// video stream; `Err` for malformed RIFF.
    pub fn parse(file: &'a [u8]) -> Result<Option<Self>> {
        let mut cursor = Cursor::new(file);
        let (top_id, top_size) = cursor.read_chunk_header()?;
        if top_id != FOURCC_RIFF {
            return Err(Error::invalid(format!(
                "AVI: expected top RIFF, got {}",
                fourcc_str(&top_id)
            )));
        }
        let riff_payload = cursor.take_payload(top_size)?;
        let mut riff = Cursor::new(riff_payload);
        let form = riff.take(4)?;
        if form != FOURCC_AVI {
            return Err(Error::invalid(format!(
                "AVI: expected AVI form, got {}",
                fourcc_str(form)
            )));
        }

        let mut bih: Option<&'a [u8]> = None;
        let mut movi: Option<&'a [u8]> = None;

        while !riff.is_done() {
            let (id, size) = riff.read_chunk_header()?;
            let payload = riff.take_payload(size)?;
            if id == FOURCC_LIST {
                let sub = if payload.len() >= 4 {
                    let mut s = [0u8; 4];
                    s.copy_from_slice(&payload[..4]);
                    s
                } else {
                    return Err(Error::invalid("AVI: LIST chunk too small"));
                };
                let inner = &payload[4..];
                match sub {
                    FOURCC_HDRL => {
                        if let Some(bytes) = scan_hdrl_for_bitmap_info_header(inner)? {
                            bih = Some(bytes);
                        }
                    }
                    FOURCC_MOVI => {
                        movi = Some(inner);
                    }
                    _ => { /* ignore other LIST kinds (INFO etc.) */ }
                }
            } else if id == FOURCC_AVIX {
                // OpenDML extension; unsupported for now.
            } else {
                // idx1 / JUNK / other; ignore.
            }
        }

        let (Some(bih), Some(movi)) = (bih, movi) else {
            return Ok(None);
        };
        Ok(Some(Self {
            bitmap_info_header: bih,
            movi_payload: movi,
        }))
    }

    /// Iterate `movi` chunk entries; yields each `##d{c,b}` chunk's
    /// payload. Audio / index chunks are skipped silently.
    pub fn frames(&self) -> AviFrameIter<'a> {
        AviFrameIter {
            cursor: Cursor::new(self.movi_payload),
        }
    }
}

/// Iterator over video-frame chunks inside a `movi` list. Returns
/// `Some(Err)` only on malformed RIFF; well-formed non-video chunks
/// are skipped silently.
pub struct AviFrameIter<'a> {
    cursor: Cursor<'a>,
}

impl<'a> Iterator for AviFrameIter<'a> {
    type Item = Result<AviFrame<'a>>;
    fn next(&mut self) -> Option<Self::Item> {
        while !self.cursor.is_done() {
            let header = match self.cursor.read_chunk_header() {
                Ok(h) => h,
                Err(e) => return Some(Err(e)),
            };
            let (id, size) = header;
            let payload = match self.cursor.take_payload(size) {
                Ok(p) => p,
                Err(e) => return Some(Err(e)),
            };
            // Heuristic: video frames are `##dc` (compressed) or
            // `##db` (uncompressed DIB). The two leading bytes are
            // ASCII digits identifying the stream index. A robust
            // demuxer would cross-check against the stream-header
            // type from `hdrl`; for our HuffYUV-only consumer the
            // first match is sufficient.
            if id[2] == b'd' && (id[3] == b'c' || id[3] == b'b') {
                return Some(Ok(AviFrame {
                    chunk_id: id,
                    payload,
                }));
            }
            // Otherwise skip.
        }
        None
    }
}

fn scan_hdrl_for_bitmap_info_header(hdrl: &[u8]) -> Result<Option<&[u8]>> {
    let mut cursor = Cursor::new(hdrl);
    while !cursor.is_done() {
        let (id, size) = cursor.read_chunk_header()?;
        let payload = cursor.take_payload(size)?;
        if id == FOURCC_LIST && payload.len() >= 4 {
            let mut sub = [0u8; 4];
            sub.copy_from_slice(&payload[..4]);
            if sub == FOURCC_STRL {
                if let Some(bih) = scan_strl_for_video_strf(&payload[4..])? {
                    return Ok(Some(bih));
                }
            }
        }
    }
    Ok(None)
}

fn scan_strl_for_video_strf(strl: &[u8]) -> Result<Option<&[u8]>> {
    let mut cursor = Cursor::new(strl);
    let mut is_video = false;
    let mut strf_payload: Option<&[u8]> = None;
    while !cursor.is_done() {
        let (id, size) = cursor.read_chunk_header()?;
        let payload = cursor.take_payload(size)?;
        if id == FOURCC_STRH {
            if payload.len() >= 4 && payload[..4] == STREAM_TYPE_VIDS {
                is_video = true;
            }
        } else if id == FOURCC_STRF {
            strf_payload = Some(payload);
        }
    }
    if is_video {
        Ok(strf_payload)
    } else {
        Ok(None)
    }
}

/// Tiny bytes cursor — RIFF chunks are 4-byte aligned, so each chunk
/// payload is followed by a 1-byte pad when its size is odd.
struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn is_done(&self) -> bool {
        self.pos >= self.data.len()
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.pos + n > self.data.len() {
            return Err(Error::invalid(format!(
                "RIFF: take({}) at pos {} exceeds buffer {}",
                n,
                self.pos,
                self.data.len()
            )));
        }
        let out = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }

    fn read_chunk_header(&mut self) -> Result<([u8; 4], u32)> {
        let id_bytes = self.take(4)?;
        let mut id = [0u8; 4];
        id.copy_from_slice(id_bytes);
        let size_bytes = self.take(4)?;
        let mut sb = [0u8; 4];
        sb.copy_from_slice(size_bytes);
        let size = u32::from_le_bytes(sb);
        Ok((id, size))
    }

    fn take_payload(&mut self, size: u32) -> Result<&'a [u8]> {
        let payload = self.take(size as usize)?;
        // Pad byte for odd-length chunks.
        if size % 2 == 1 && self.pos < self.data.len() {
            self.pos += 1;
        }
        Ok(payload)
    }
}

fn fourcc_str(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

// ───────────────────────── encoder side ─────────────────────────
//
// A minimal AVI writer that wraps a single HuffYUV `strf` and a
// caller-supplied sequence of pre-encoded frame chunks into a
// well-formed AVI 1.0 container. Used by tests to round-trip
// HuffYUV streams through the AVI walker.

/// Build a minimal AVI 1.0 container with one `vids` stream carrying
/// HuffYUV-encoded frames.
///
/// `strf` is the BITMAPINFOHEADER produced by
/// [`crate::encoder::build_bitmapinfoheader`]; `frames` is the
/// sequence of `00dc` chunk payloads (each as produced by
/// [`crate::encoder::encode_frame`]).
///
/// The output is guaranteed to round-trip through
/// [`AviVideoStream::parse`] so the lockstep tests can decode their
/// own AVI without external file dependencies. The `idx1` chunk is
/// omitted (it is optional in AVI 1.0; demuxers use linear `movi`
/// scan as a fallback, which is what our walker does).
pub fn build_minimal_avi(strf: &[u8], frames: &[Vec<u8>], width: u32, height: u32) -> Vec<u8> {
    // ───────── strh (vids stream header, 56 bytes) ─────────
    let mut strh = Vec::with_capacity(56);
    strh.extend_from_slice(b"vids");
    strh.extend_from_slice(b"HFYU"); // fccHandler
    strh.extend_from_slice(&0u32.to_le_bytes()); // dwFlags
    strh.extend_from_slice(&0u16.to_le_bytes()); // wPriority
    strh.extend_from_slice(&0u16.to_le_bytes()); // wLanguage
    strh.extend_from_slice(&0u32.to_le_bytes()); // dwInitialFrames
    strh.extend_from_slice(&1u32.to_le_bytes()); // dwScale
    strh.extend_from_slice(&30u32.to_le_bytes()); // dwRate (30 fps)
    strh.extend_from_slice(&0u32.to_le_bytes()); // dwStart
    strh.extend_from_slice(&(frames.len() as u32).to_le_bytes()); // dwLength
    strh.extend_from_slice(&0u32.to_le_bytes()); // dwSuggestedBufferSize
    strh.extend_from_slice(&u32::MAX.to_le_bytes()); // dwQuality
    strh.extend_from_slice(&0u32.to_le_bytes()); // dwSampleSize
    strh.extend_from_slice(&0i16.to_le_bytes()); // rcFrame.left
    strh.extend_from_slice(&0i16.to_le_bytes()); // rcFrame.top
    strh.extend_from_slice(&(width as i16).to_le_bytes()); // rcFrame.right
    strh.extend_from_slice(&(height as i16).to_le_bytes()); // rcFrame.bottom

    // ───────── strl LIST (strh + strf) ─────────
    let mut strl = Vec::new();
    strl.extend_from_slice(b"strl");
    push_chunk(&mut strl, b"strh", &strh);
    push_chunk(&mut strl, b"strf", strf);

    // ───────── avih (12-int header) ─────────
    let micro_per_frame: u32 = 33333;
    let max_bytes_per_sec: u32 = 0;
    let total_frames = frames.len() as u32;
    let streams: u32 = 1;
    let mut avih = Vec::with_capacity(56);
    avih.extend_from_slice(&micro_per_frame.to_le_bytes());
    avih.extend_from_slice(&max_bytes_per_sec.to_le_bytes());
    avih.extend_from_slice(&0u32.to_le_bytes()); // dwPaddingGranularity
    avih.extend_from_slice(&0x10u32.to_le_bytes()); // dwFlags = AVIF_HASINDEX cleared, AVIF_ISINTERLEAVED off
    avih.extend_from_slice(&total_frames.to_le_bytes());
    avih.extend_from_slice(&0u32.to_le_bytes()); // dwInitialFrames
    avih.extend_from_slice(&streams.to_le_bytes());
    avih.extend_from_slice(&0u32.to_le_bytes()); // dwSuggestedBufferSize
    avih.extend_from_slice(&width.to_le_bytes());
    avih.extend_from_slice(&height.to_le_bytes());
    avih.extend_from_slice(&[0u8; 16]); // reserved

    // ───────── hdrl LIST (avih + strl) ─────────
    let mut hdrl = Vec::new();
    hdrl.extend_from_slice(b"hdrl");
    push_chunk(&mut hdrl, b"avih", &avih);
    push_list(&mut hdrl, &strl);

    // ───────── movi LIST (00dc chunks) ─────────
    let mut movi = Vec::new();
    movi.extend_from_slice(b"movi");
    for f in frames {
        push_chunk(&mut movi, b"00dc", f);
    }

    // ───────── RIFF AVI ─────────
    let mut riff_payload = Vec::new();
    riff_payload.extend_from_slice(b"AVI ");
    push_list(&mut riff_payload, &hdrl);
    push_list(&mut riff_payload, &movi);

    let mut out = Vec::new();
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(riff_payload.len() as u32).to_le_bytes());
    out.extend_from_slice(&riff_payload);
    out
}

fn push_chunk(out: &mut Vec<u8>, fourcc: &[u8; 4], payload: &[u8]) {
    out.extend_from_slice(fourcc);
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(payload);
    if payload.len() % 2 == 1 {
        out.push(0);
    }
}

fn push_list(out: &mut Vec<u8>, payload: &[u8]) {
    out.extend_from_slice(b"LIST");
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(payload);
    if payload.len() % 2 == 1 {
        out.push(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::decode_frame;
    use crate::encoder::encode_frame;
    use crate::header::{Method, PixelFamily, StreamConfig};

    fn synth_yuy2(w: usize, h: usize) -> Vec<u8> {
        let mut v = vec![0u8; w * 2 * h];
        for row in 0..h {
            for col in 0..(w * 2) {
                v[row * w * 2 + col] = ((row * 11 + col * 17) & 0xFF) as u8;
            }
        }
        v
    }

    #[test]
    fn roundtrip_self_authored_avi_yuy2_left_2_frames() {
        let pixels = synth_yuy2(8, 4);
        let (strf, frame) = encode_frame(PixelFamily::Yuy2, Method::Left, 8, 4, &pixels).unwrap();
        let avi = build_minimal_avi(&strf, &[frame.clone(), frame.clone()], 8, 4);
        let stream = AviVideoStream::parse(&avi).unwrap().expect("video stream");
        // Re-parse the BIH and decode each frame.
        let cfg = StreamConfig::parse_bitmapinfoheader(stream.bitmap_info_header).unwrap();
        let mut count = 0;
        for f in stream.frames() {
            let f = f.unwrap();
            assert_eq!(&f.chunk_id, b"00dc");
            let out = decode_frame(&cfg, f.payload).unwrap();
            assert_eq!(out.pixels, pixels);
            count += 1;
        }
        assert_eq!(count, 2);
    }

    #[test]
    fn riff_walker_rejects_non_riff() {
        let err = AviVideoStream::parse(b"RIFXfoo").unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)));
    }

    #[test]
    fn riff_walker_returns_none_when_no_video_stream() {
        // A barely-legal RIFF with an empty AVI form (no LISTs).
        let mut riff = Vec::new();
        riff.extend_from_slice(b"RIFF");
        riff.extend_from_slice(&4u32.to_le_bytes());
        riff.extend_from_slice(b"AVI ");
        let opt = AviVideoStream::parse(&riff).unwrap();
        assert!(opt.is_none());
    }

    #[test]
    fn riff_walker_skips_non_video_frame_chunks_in_movi() {
        // Build an AVI whose movi list contains an `01wb` audio
        // chunk between the two `00dc` video chunks. The walker
        // should skip the audio chunk silently.
        let pixels = synth_yuy2(4, 4);
        let (strf, frame) = encode_frame(PixelFamily::Yuy2, Method::Left, 4, 4, &pixels).unwrap();
        // Manually splice an audio chunk into the movi by building
        // the frames list with a synthetic non-`##d?` payload via
        // direct movi assembly — we use the high-level builder but
        // append an extra audio chunk after.
        let mut avi = build_minimal_avi(&strf, std::slice::from_ref(&frame), 4, 4);
        // Find the `LIST size 'movi'` and inject `01wb size garbage`
        // into its payload — easier path: rebuild by hand here.
        // Instead, re-use builder + craft a synthetic audio chunk.
        let _ = &mut avi; // (kept for clarity; we build a fresh one below)
        let mut movi = Vec::new();
        movi.extend_from_slice(b"movi");
        push_chunk(&mut movi, b"00dc", &frame);
        push_chunk(&mut movi, b"01wb", &[0xAAu8; 7]); // audio, odd-size to exercise pad
        push_chunk(&mut movi, b"00dc", &frame);
        let mut hdrl = Vec::new();
        hdrl.extend_from_slice(b"hdrl");
        // Minimal avih + strl.
        let mut avih = vec![0u8; 56];
        avih[16..20].copy_from_slice(&2u32.to_le_bytes()); // 2 frames
        avih[24..28].copy_from_slice(&1u32.to_le_bytes()); // 1 stream
        avih[32..36].copy_from_slice(&4u32.to_le_bytes()); // width
        avih[36..40].copy_from_slice(&4u32.to_le_bytes()); // height
        push_chunk(&mut hdrl, b"avih", &avih);
        let mut strh = vec![0u8; 56];
        strh[..4].copy_from_slice(b"vids");
        strh[4..8].copy_from_slice(b"HFYU");
        let mut strl = Vec::new();
        strl.extend_from_slice(b"strl");
        push_chunk(&mut strl, b"strh", &strh);
        push_chunk(&mut strl, b"strf", &strf);
        push_list(&mut hdrl, &strl);
        let mut riff_payload = Vec::new();
        riff_payload.extend_from_slice(b"AVI ");
        push_list(&mut riff_payload, &hdrl);
        push_list(&mut riff_payload, &movi);
        let mut out = Vec::new();
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(riff_payload.len() as u32).to_le_bytes());
        out.extend_from_slice(&riff_payload);
        let stream = AviVideoStream::parse(&out).unwrap().expect("video stream");
        let cfg = StreamConfig::parse_bitmapinfoheader(stream.bitmap_info_header).unwrap();
        let frames: Vec<_> = stream.frames().collect::<Result<_>>().unwrap();
        assert_eq!(frames.len(), 2, "audio chunk must be skipped");
        for f in &frames {
            let out = decode_frame(&cfg, f.payload).unwrap();
            assert_eq!(out.pixels, pixels);
        }
    }
}
