#![no_main]

//! Drive arbitrary fuzz-supplied bytes through the full HuffYUV /
//! FFVHuff *encode* chain — `encode_frame_with_mode` — across every
//! legal `(family, method, extradata-mode)` triple, then re-parse the
//! synthesised `BITMAPINFOHEADER` via `StreamConfig::parse_bitmapinfoheader`
//! and re-decode via `decode_frame`. The contract under test is two-fold:
//!
//! 1. **Liveness**: neither the encoder nor the encode→decode round-trip
//!    may panic, abort, integer-overflow (in a debug/ASAN build), or
//!    index out of bounds — regardless of how hostile the input pixel
//!    bytes are.
//! 2. **Bit-exactness**: when `encode_frame_with_mode` returns `Ok`, the
//!    decoder fed the produced `(strf, frame)` pair MUST round-trip the
//!    original pixel buffer byte-for-byte (HuffYUV is lossless per
//!    spec/00 §1.1).
//!
//! Why an encoder-only target alongside the existing
//! `decode_huffyuv.rs`: the encoder is the larger of the two halves
//! (`src/encoder.rs` carries the package-merge length-limited Huffman
//! builder, three extradata modes, and the SWAR forward predictor
//! fusion paths added across rounds 95 / 100 / 103 / 115 / 181 / 186),
//! and the decoder fuzz target tells us nothing about an encoder bug
//! that produces a malformed `strf` or a body the decoder rejects. The
//! roundtrip assertion turns the encoder into a fuzz-discoverable
//! check on its own correctness without needing an external oracle
//! (which the clean-room wall bars anyway).
//!
//! # Input framing
//!
//! The fuzz buffer drives both the configuration selector and the raw
//! pixel input:
//!
//! ```text
//! [0]      u8   family selector mod 3        (Yuy2 / Rgb24 / Rgb32)
//! [1]      u8   method selector mod 6        (PredictOld / Left / Gradient /
//!                                              Median / LeftDecorr / GradientDecorr)
//! [2]      u8   extradata-mode selector mod 3 (ClassicV2 / CustomV2 / V1xCompat)
//! [3]      u8   width-byte                    (mapped to width in [MIN_W, MAX_W])
//! [4]      u8   height-byte                   (mapped to height in [MIN_H, MAX_H],
//!                                              with a low-probability lift to the
//!                                              interlaced regime height > 288)
//! [5..]         raw pixel bytes               (truncated or zero-padded to
//!                                              width * height * bytes_per_pixel)
//! ```
//!
//! Illegal `(family, method)` combinations (e.g. YUY2 + GradientDecorr,
//! RGB + Median) return `Err` from the encoder by design; the harness
//! treats those as "skip" rather than as a bug — they exercise the
//! validation path in `encode_frame_with_mode` itself.
//!
//! # Raster cap rationale
//!
//! The encoder allocates `width * height * bytes_per_pixel` for the
//! residual stream + `row_bytes * height` for the gradient intermediate.
//! Those dimensions come from the fuzz buffer, so a buffer requesting
//! e.g. 1920×1080 RGB32 on every iter would spend the entire fuzz
//! budget memory-shuffling instead of exercising the encoder logic.
//! The width/height mapping below caps the raster at a few KiB per
//! iteration, with a low-probability lift into the interlaced regime
//! (height > 288) so the field-stride=2 split path also gets coverage.

use libfuzzer_sys::fuzz_target;
use oxideav_huffyuv::header::{Method, PixelFamily, StreamConfig};
use oxideav_huffyuv::{
    build_bitmapinfoheader, decode_frame, decode_frame_with_workers, encode_frame_with_mode,
    encode_frame_with_mode_workers, ExtradataMode,
};

/// Minimum width per family. YUY2 macropixels are 2 pixels wide, so
/// any YUY2 width must be a multiple of 2 — width 2 is the smallest
/// legal value. RGB widths can be 1.
const MIN_W_YUY2: u32 = 2;
const MIN_W_RGB: u32 = 1;
/// Cap widths so a worst-case `(width, height) * bpp` raster stays
/// inside a few KiB. Cap chosen so RGB32 320×16 ≈ 20 KiB is the
/// upper bound on the non-interlaced raster.
const MAX_W: u32 = 64;
const MIN_H: u32 = 1;
const MAX_H_PROGRESSIVE: u32 = 32;
/// Interlaced lift: `predict::is_interlaced_height` triggers at
/// `height > 288` (spec/02 §2). Sample a few values just above the
/// boundary so the encoder's `walking-stride` field-split path gets
/// hit too; the cap stays small so a single iter is still cheap.
const INTERLACED_HEIGHTS: [u32; 3] = [290, 300, 320];

fn pick_family(b: u8) -> PixelFamily {
    match b % 3 {
        0 => PixelFamily::Yuy2,
        1 => PixelFamily::Rgb24,
        _ => PixelFamily::Rgb32,
    }
}

fn pick_method(b: u8) -> Method {
    match b % 6 {
        0 => Method::PredictOld,
        1 => Method::Left,
        2 => Method::Gradient,
        3 => Method::Median,
        4 => Method::LeftDecorr,
        _ => Method::GradientDecorr,
    }
}

fn pick_mode(b: u8) -> ExtradataMode {
    match b % 3 {
        0 => ExtradataMode::ClassicV2,
        1 => ExtradataMode::CustomV2,
        _ => ExtradataMode::V1xCompat,
    }
}

fn pick_width(family: PixelFamily, b: u8) -> u32 {
    let min = if family == PixelFamily::Yuy2 {
        MIN_W_YUY2
    } else {
        MIN_W_RGB
    };
    let span = MAX_W - min + 1;
    let mut w = min + (b as u32 % span);
    if family == PixelFamily::Yuy2 && w % 2 != 0 {
        // Round up to the nearest even value so YUY2 macropixels stay
        // intact. `w + 1` is at most MAX_W + 1; cap back to MAX_W.
        w = (w + 1).min(MAX_W);
        if w % 2 != 0 {
            // MAX_W happens to be even (64); this branch is defensive.
            w -= 1;
        }
    }
    w
}

fn pick_height(b: u8) -> u32 {
    // Sample 7/8 of the input space from the progressive range and
    // 1/8 from a small set of interlaced heights, so the field-stride
    // path gets coverage without dominating iteration cost.
    if b & 0xE0 == 0xE0 {
        let idx = (b as usize) % INTERLACED_HEIGHTS.len();
        INTERLACED_HEIGHTS[idx]
    } else {
        MIN_H + (b as u32 % (MAX_H_PROGRESSIVE - MIN_H + 1))
    }
}

fn bytes_per_pixel(family: PixelFamily) -> usize {
    match family {
        PixelFamily::Yuy2 => 2,
        PixelFamily::Rgb24 => 3,
        PixelFamily::Rgb32 => 4,
    }
}

/// Build a pixel buffer of exactly `width * height * bpp` bytes,
/// drawing from `src` and zero-padding any tail. The encoder requires
/// the buffer length to match the declared raster, so the harness must
/// normalise it before calling `encode_frame_with_mode`.
fn build_pixels(width: u32, height: u32, bpp: usize, src: &[u8]) -> Vec<u8> {
    let need = (width as usize) * (height as usize) * bpp;
    let mut out = vec![0u8; need];
    let take = src.len().min(need);
    out[..take].copy_from_slice(&src[..take]);
    out
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 5 {
        return;
    }
    let family = pick_family(data[0]);
    let method = pick_method(data[1]);
    let mode = pick_mode(data[2]);
    let width = pick_width(family, data[3]);
    let height = pick_height(data[4]);
    let bpp = bytes_per_pixel(family);

    let pixels = build_pixels(width, height, bpp, &data[5..]);

    // Try to encode. Illegal (family, method) pairs return Err here
    // — that's the encoder's documented validation behaviour
    // (`spec/01 §3.1`), not a bug. Same for V1xCompat + a residual
    // stream whose symbols collide with a length-0 entry in the v1.x
    // codebooks (handled by `verify_body_in_table` inside the
    // encoder). All `Err` returns are fine; the contract under test
    // is "no panic on any input".
    let serial = encode_frame_with_mode(family, method, width, height, &pixels, mode);
    // Round-420 differential oracle: the two-worker encode path must
    // agree with the serial encoder on every input — same Ok/Err
    // disposition, byte-identical (strf, frame) on Ok. Interlaced
    // heights exercise the parallel residual + emit phases;
    // progressive heights pin the budget as a no-op.
    let parallel = encode_frame_with_mode_workers(family, method, width, height, &pixels, mode, 2);
    let (strf, frame) = match (serial, parallel) {
        (Ok(s), Ok(p)) => {
            assert_eq!(
                s, p,
                "budget-2 encode diverged from serial (family={:?} method={:?} mode={:?} {}×{})",
                family, method, mode, width, height,
            );
            s
        }
        (Err(_), Err(_)) => return,
        (s, p) => panic!(
            "budget-2 encode disposition diverged from serial (serial ok={}, parallel ok={})",
            s.is_ok(),
            p.is_ok(),
        ),
    };

    // The strf the encoder emits must parse back into a usable
    // StreamConfig. A mismatch here is an encoder bug.
    let cfg = StreamConfig::parse_bitmapinfoheader(&strf)
        .expect("encoder-produced strf must parse via StreamConfig::parse_bitmapinfoheader");

    // The encoder's `build_bitmapinfoheader` helper is publicly
    // exposed (muxers call it to synthesise the AVI strf). Exercise
    // it on the same configuration so it is fuzz-covered too. The
    // `extradata_tables` argument is the inner table region; for the
    // ClassicV2 / CustomV2 paths it is `strf[0x2C..]` (the body
    // after the 0x2C-byte fixed prefix), for V1xCompat the strf is
    // exactly 0x28 bytes long with no extradata at all.
    let has_extradata = strf.len() > 0x28;
    let extradata = if has_extradata {
        &strf[0x2C..]
    } else {
        &[][..]
    };
    let strf2 = build_bitmapinfoheader(family, method, width, height, extradata, has_extradata);
    let cfg2 = StreamConfig::parse_bitmapinfoheader(&strf2)
        .expect("build_bitmapinfoheader output must parse back via parse_bitmapinfoheader");
    let decoded2 = decode_frame(&cfg2, &frame)
        .expect("decode against build_bitmapinfoheader cfg must succeed");
    assert_eq!(
        decoded2.pixels, pixels,
        "encode→build_bitmapinfoheader→parse→decode must round-trip bit-exact (family={:?} method={:?} mode={:?} {}×{})",
        family, method, mode, width, height,
    );

    // Primary round-trip: the encoder-produced strf+frame must decode
    // back to the original pixel buffer byte-for-byte. HuffYUV is
    // lossless per spec/00 §1.1 — any mismatch is a real encoder /
    // decoder symmetry bug.
    let decoded =
        decode_frame(&cfg, &frame).expect("encode→decode must succeed on encoder-produced output");
    assert_eq!(
        decoded.pixels, pixels,
        "encode→decode must round-trip bit-exact (family={:?} method={:?} mode={:?} {}×{})",
        family, method, mode, width, height,
    );

    // Round-420: the budget-2 decoder must round-trip the same wire
    // bytes identically (fuzz-side companion to the golden-pin
    // budget-invariance suite).
    let decoded_p = decode_frame_with_workers(&cfg, &frame, 2)
        .expect("budget-2 decode must succeed on encoder-produced output");
    assert_eq!(
        decoded_p.pixels, pixels,
        "budget-2 encode→decode must round-trip bit-exact (family={:?} method={:?} mode={:?} {}×{})",
        family, method, mode, width, height,
    );
});
