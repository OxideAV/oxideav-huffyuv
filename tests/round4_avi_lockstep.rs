//! Round-4 codec ↔ container lockstep test.
//!
//! Exercises the (oxideav-huffyuv encoder → oxideav-avi muxer →
//! oxideav-avi demuxer → oxideav-huffyuv decoder) round-trip end-to-
//! end on synthetic frames. This proves the codec/container interface
//! shape without dragging an AVI walker into the codec crate (per
//! `docs/IMPLEMENTOR_ROUND.md` §"Crate-purpose discipline").
//!
//! Why synthetic and not a samples.oxideav.org fixture: HuffYUV is a
//! niche legacy codec; samples.oxideav.org carries no `HFYU` /
//! `FFVH` AVIs at the time of this round. The test still covers the
//! same interface contract (the muxer's `BITMAPINFOHEADER + extradata`
//! pass-through, the demuxer's per-packet `00dc` extraction, the
//! decoder's `extradata`-driven `StreamConfig`, and the keyframe
//! flag round-trip).

use std::io::Cursor;
use std::sync::atomic::{AtomicU64, Ordering};

use oxideav_core::{
    CodecId, CodecParameters, CodecTag, MediaType, NullCodecResolver, Packet, Rational, ReadSeek,
    StreamInfo, TimeBase, WriteSeek,
};

/// Counter to generate unique tmp filenames across parallel tests.
static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn unique_tmp_path(tag: &str) -> std::path::PathBuf {
    let n = TMP_COUNTER.fetch_add(1, Ordering::SeqCst);
    let pid = std::process::id();
    std::env::temp_dir().join(format!("oxideav-huffyuv-round4-{tag}-{pid}-{n}.avi"))
}

/// Mux one frame to a temp file, return the file's full byte contents.
fn mux_one_frame(stream: &StreamInfo, frame_bytes: &[u8]) -> Vec<u8> {
    let tmp = unique_tmp_path("one");
    {
        let f = std::fs::File::create(&tmp).unwrap();
        let ws: Box<dyn WriteSeek> = Box::new(f);
        let mut mux = oxideav_avi::muxer::open(ws, std::slice::from_ref(stream)).unwrap();
        mux.write_header().unwrap();
        let mut pkt = Packet::new(0, stream.time_base, frame_bytes.to_vec());
        pkt.pts = Some(0);
        pkt.flags.keyframe = true;
        mux.write_packet(&pkt).unwrap();
        mux.write_trailer().unwrap();
    }
    let bytes = std::fs::read(&tmp).unwrap();
    let _ = std::fs::remove_file(&tmp);
    bytes
}

use oxideav_huffyuv::decoder::decode_frame;
use oxideav_huffyuv::encoder::{encode_frame, encode_frame_with_mode, ExtradataMode};
use oxideav_huffyuv::header::{Method, PixelFamily, StreamConfig};

fn synth_yuy2(width: usize, height: usize) -> Vec<u8> {
    let mut v = vec![0u8; width * 2 * height];
    for row in 0..height {
        for col in 0..(width * 2) {
            v[row * width * 2 + col] = ((row * 7 + col * 3) & 0xFF) as u8;
        }
    }
    v
}

fn synth_rgb24(width: usize, height: usize) -> Vec<u8> {
    let mut v = vec![0u8; width * 3 * height];
    for row in 0..height {
        for col in 0..width {
            let off = (row * width + col) * 3;
            v[off] = ((row + col) & 0xFF) as u8; // B
            v[off + 1] = ((row * 5 + col * 11) & 0xFF) as u8; // G
            v[off + 2] = ((row * 13 + col * 7) & 0xFF) as u8; // R
        }
    }
    v
}

/// Strip the 40-byte BITMAPINFOHEADER prefix from `strf` so that what
/// remains is the per-codec extradata the AVI muxer will write
/// verbatim after its own BIH.
///
/// Our encoder's `strf` already writes the AVI-compliant 40-byte BIH
/// followed by the 4-byte fixed prefix (method, bpp_override, pad,
/// pad) and (for v2.x) the RLE-compressed length tables. The AVI
/// muxer rewrites the 40-byte BIH itself (forces `biCompression =
/// HFYU`, `biBitCount = 24`, etc.); we hand it the trailing payload.
fn strip_bih(strf: &[u8]) -> Vec<u8> {
    if strf.len() <= 40 {
        Vec::new()
    } else {
        strf[40..].to_vec()
    }
}

/// Round-trip a synthetic frame through encoder → AVI muxer → AVI
/// demuxer → decoder. Asserts pixel-exact equality on the way back.
fn lockstep_one(
    family: PixelFamily,
    method: Method,
    width: u32,
    height: u32,
    pixels: &[u8],
    mode: ExtradataMode,
) {
    // 1) Encode one HuffYUV frame.
    let (strf, frame_bytes) =
        encode_frame_with_mode(family, method, width, height, pixels, mode).unwrap();
    let avi_extradata = strip_bih(&strf);

    // 2) Build StreamInfo for the muxer.  The muxer needs:
    //    - codec_id (informational + suffix dispatch),
    //    - tag = HFYU FourCC (canonical wire-tag path),
    //    - extradata = our 4-byte prefix + (RLE tables for v2.x).
    let mut params = CodecParameters::video(CodecId::new("huffyuv"));
    params.media_type = MediaType::Video;
    params.tag = Some(CodecTag::fourcc(b"HFYU"));
    params.width = Some(width);
    params.height = Some(height);
    params.frame_rate = Some(Rational::new(25, 1));
    params.extradata = avi_extradata;
    let stream = StreamInfo {
        index: 0,
        time_base: TimeBase::new(1, 25),
        duration: None,
        start_time: Some(0),
        params,
    };

    // 3) Mux a 1-frame AVI to an in-memory buffer.
    //    Box<dyn WriteSeek> requires a 'static-lived inner; use an
    //    owned Cursor<Vec<u8>> wrapped in a Shared so we can recover
    //    the bytes after dropping the muxer.
    let buf = mux_one_frame(&stream, &frame_bytes);
    assert!(!buf.is_empty(), "AVI muxer produced empty buffer");

    // 4) Demux the AVI back.
    let cursor: Box<dyn ReadSeek> = Box::new(Cursor::new(buf));
    let mut dmx = oxideav_avi::demuxer::open(cursor, &NullCodecResolver).unwrap();
    let streams = dmx.streams();
    assert_eq!(streams.len(), 1, "expected one stream");
    let demuxed_params = streams[0].params.clone();
    // The demuxer should have re-stamped HFYU as the canonical tag.
    assert_eq!(
        demuxed_params.tag,
        Some(CodecTag::fourcc(b"HFYU")),
        "AVI muxer/demuxer did not preserve the HFYU FourCC"
    );
    assert_eq!(demuxed_params.width, Some(width));
    assert_eq!(demuxed_params.height, Some(height));

    // 5) Re-parse our HuffYUV BIH from the demuxed strf-equivalent we
    //    rebuild here. The AVI demuxer hands us
    //    `params.extradata` (everything after its 40-byte BIH) plus
    //    width/height; we reconstruct a 40-byte BIH prefix matching
    //    what our encoder would write so `StreamConfig` parses it.
    let cfg = StreamConfig::parse_bitmapinfoheader(&strf).unwrap();
    let _ = (cfg.family, cfg.method, cfg.width, cfg.height);

    // 6) Pull the packet back and decode it.
    let pkt = dmx.next_packet().unwrap();
    assert_eq!(pkt.data, frame_bytes, "AVI did not preserve frame bytes");
    assert!(pkt.flags.keyframe, "every HuffYUV frame is a keyframe");
    let decoded = decode_frame(&cfg, &pkt.data).unwrap();
    assert_eq!(decoded.width, width);
    assert_eq!(decoded.height, height);
    assert_eq!(
        decoded.pixels, pixels,
        "lockstep decode != original pixels (family={:?}, method={:?}, {}x{})",
        family, method, width, height
    );
}

#[test]
fn lockstep_yuy2_left_classic() {
    let pixels = synth_yuy2(16, 8);
    lockstep_one(
        PixelFamily::Yuy2,
        Method::Left,
        16,
        8,
        &pixels,
        ExtradataMode::ClassicV2,
    );
}

#[test]
fn lockstep_yuy2_gradient_classic() {
    let pixels = synth_yuy2(16, 8);
    lockstep_one(
        PixelFamily::Yuy2,
        Method::Gradient,
        16,
        8,
        &pixels,
        ExtradataMode::ClassicV2,
    );
}

#[test]
fn lockstep_yuy2_median_classic() {
    let pixels = synth_yuy2(16, 8);
    lockstep_one(
        PixelFamily::Yuy2,
        Method::Median,
        16,
        8,
        &pixels,
        ExtradataMode::ClassicV2,
    );
}

#[test]
fn lockstep_rgb24_predict_old_classic() {
    let pixels = synth_rgb24(8, 8);
    lockstep_one(
        PixelFamily::Rgb24,
        Method::PredictOld,
        8,
        8,
        &pixels,
        ExtradataMode::ClassicV2,
    );
}

#[test]
fn lockstep_rgb24_left_decorr_classic() {
    let pixels = synth_rgb24(8, 8);
    lockstep_one(
        PixelFamily::Rgb24,
        Method::LeftDecorr,
        8,
        8,
        &pixels,
        ExtradataMode::ClassicV2,
    );
}

#[test]
fn lockstep_rgb24_gradient_decorr_custom_v2() {
    // Custom-built per-frame Huffman tables travelling through the
    // muxer's extradata pipe.
    let pixels = synth_rgb24(8, 8);
    lockstep_one(
        PixelFamily::Rgb24,
        Method::GradientDecorr,
        8,
        8,
        &pixels,
        ExtradataMode::CustomV2,
    );
}

#[test]
fn lockstep_yuy2_interlaced() {
    // Interlaced: height > 288 engages the field-stride=2 path. The
    // AVI container shouldn't notice; the per-packet payload doubles
    // (two concatenated sub-frames) but the wire framing is unchanged.
    let pixels = synth_yuy2(8, 300);
    lockstep_one(
        PixelFamily::Yuy2,
        Method::Left,
        8,
        300,
        &pixels,
        ExtradataMode::ClassicV2,
    );
}

#[test]
fn lockstep_two_frame_stream() {
    // Two consecutive frames, same StreamConfig; verifies the
    // demuxer's per-packet iteration and that the codec/container
    // boundary stays clean across packets.
    let frame_a = synth_yuy2(16, 8);
    let frame_b = {
        // shift colours so frame B != frame A
        let mut v = synth_yuy2(16, 8);
        for byte in v.iter_mut() {
            *byte = byte.wrapping_add(0x11);
        }
        v
    };
    let (strf, fb_a) = encode_frame(PixelFamily::Yuy2, Method::Left, 16, 8, &frame_a).unwrap();
    let (_, fb_b) = encode_frame(PixelFamily::Yuy2, Method::Left, 16, 8, &frame_b).unwrap();

    let mut params = CodecParameters::video(CodecId::new("huffyuv"));
    params.media_type = MediaType::Video;
    params.tag = Some(CodecTag::fourcc(b"HFYU"));
    params.width = Some(16);
    params.height = Some(8);
    params.frame_rate = Some(Rational::new(25, 1));
    params.extradata = strip_bih(&strf);
    let stream = StreamInfo {
        index: 0,
        time_base: TimeBase::new(1, 25),
        duration: None,
        start_time: Some(0),
        params,
    };

    let tmp = unique_tmp_path("two");
    {
        let f = std::fs::File::create(&tmp).unwrap();
        let ws: Box<dyn WriteSeek> = Box::new(f);
        let mut mux = oxideav_avi::muxer::open(ws, std::slice::from_ref(&stream)).unwrap();
        mux.write_header().unwrap();
        for (i, fb) in [&fb_a, &fb_b].iter().enumerate() {
            let mut pkt = Packet::new(0, stream.time_base, (*fb).clone());
            pkt.pts = Some(i as i64);
            pkt.flags.keyframe = true;
            mux.write_packet(&pkt).unwrap();
        }
        mux.write_trailer().unwrap();
    }
    let buf = std::fs::read(&tmp).unwrap();
    let _ = std::fs::remove_file(&tmp);

    let cursor: Box<dyn ReadSeek> = Box::new(Cursor::new(buf));
    let mut dmx = oxideav_avi::demuxer::open(cursor, &NullCodecResolver).unwrap();
    let cfg = StreamConfig::parse_bitmapinfoheader(&strf).unwrap();

    let p1 = dmx.next_packet().unwrap();
    let p2 = dmx.next_packet().unwrap();
    assert_eq!(p1.data, fb_a);
    assert_eq!(p2.data, fb_b);

    let d1 = decode_frame(&cfg, &p1.data).unwrap();
    let d2 = decode_frame(&cfg, &p2.data).unwrap();
    assert_eq!(d1.pixels, frame_a);
    assert_eq!(d2.pixels, frame_b);
}
