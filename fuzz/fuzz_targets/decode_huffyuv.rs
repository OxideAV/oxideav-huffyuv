#![no_main]

//! Decode arbitrary fuzz-supplied bytes through the full HuffYUV /
//! FFVHuff decode chain: `StreamConfig::parse_bitmapinfoheader`
//! (`BITMAPINFOHEADER` + the 4-byte extradata prefix + RLE-compressed
//! Huffman length tables) followed by `decode_frame` (canonical-Huffman
//! build, per-pixel codeword read, LEFT / GRADIENT / MEDIAN predictor
//! inverse, and RGB decorrelation inverse).
//!
//! The contract under test is purely that every call *returns*: a
//! malformed stream yields `Err(huffyuv::Error::…)`, a well-formed one
//! yields `Ok(DecodedFrame)`, and neither path may panic, abort,
//! integer-overflow (in a debug/ASAN build), or index out of bounds —
//! regardless of how hostile the bytes are.
//!
//! # Input framing
//!
//! The fuzz buffer is split into two parts so a single libFuzzer
//! mutation can perturb either the configuration (header + Huffman
//! tables) or the frame body independently:
//!
//! ```text
//! [0..2)  u16-LE  N = extradata length
//! [2..2+N)        strf bytes (BITMAPINFOHEADER + extradata)
//! [2+N..)         per-frame compressed body
//! ```
//!
//! # Why the dimension cap
//!
//! `decode_frame` allocates `width * height * bytes_per_pixel` for the
//! output raster. Those dimensions come straight off the wire, so a
//! valid-but-enormous header (e.g. 65535 × 65535 RGB32 ≈ 17 GiB) is a
//! legitimate *resource* request, not a decoder bug — letting the
//! allocator OOM on it would be a false positive that masks the real
//! logic bugs this harness is built to find. We therefore reject
//! over-large declared frames *in the harness* (mirroring what a real
//! demuxer's sanity limits would do) before driving the decoder, while
//! still exercising every parse/build/predictor path on inputs up to
//! the cap. The library itself is deliberately left free of an
//! arbitrary built-in size policy.

use libfuzzer_sys::fuzz_target;
use oxideav_huffyuv::header::{PixelFamily, StreamConfig};
use oxideav_huffyuv::{decode_frame, DecodedFrame};

/// Upper bound on the declared output raster (16 MiB). Anything larger
/// is a resource request, not a logic path, so the harness skips it.
const MAX_OUTPUT_BYTES: u64 = 16 * 1024 * 1024;

fn bytes_per_pixel(family: PixelFamily) -> u64 {
    match family {
        PixelFamily::Yuy2 => 2,
        PixelFamily::Rgb24 => 3,
        PixelFamily::Rgb32 => 4,
    }
}

fn drive(strf: &[u8], frame: &[u8]) {
    let cfg = match StreamConfig::parse_bitmapinfoheader(strf) {
        Ok(c) => c,
        Err(_) => return,
    };

    // Reject declared frames whose raster would exceed the harness cap.
    // `decode_frame` would otherwise honestly try to allocate it, which
    // is expected behaviour rather than a bug worth flagging.
    let declared = (cfg.width as u64)
        .checked_mul(cfg.height as u64)
        .and_then(|wh| wh.checked_mul(bytes_per_pixel(cfg.family)));
    match declared {
        Some(n) if n <= MAX_OUTPUT_BYTES => {}
        _ => return,
    }

    // The whole point: decode must never panic / overflow / OOB on a
    // body of arbitrary bytes. Return value intentionally discarded
    // (a debug-build round-trip oracle would need a trusted encoder of
    // the *same* arbitrary stream, which doesn't exist).
    let _: Result<DecodedFrame, _> = decode_frame(&cfg, frame);
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }
    let n = u16::from_le_bytes([data[0], data[1]]) as usize;
    let rest = &data[2..];
    if n > rest.len() {
        // Length prefix outruns the buffer: treat the whole remainder
        // as the strf and hand the decoder an empty body. Still a valid
        // exercise of the parse + zero-length-frame paths.
        drive(rest, &[]);
        return;
    }
    let (strf, frame) = rest.split_at(n);
    drive(strf, frame);
});
