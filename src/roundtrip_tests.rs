//! Self-roundtrip tests: synthesise minimal HuffYUV frames via
//! [`crate::encoder::encode_for_test`] and decode them back via
//! [`crate::decoder::decode_frame`].
//!
//! Round-2 covers all six predictor methods × three pixel families
//! (with the spec/01 §3.1 method/family allow-list applied), plus
//! the v1.x-compat (`biSize == 0x28`) and v2.x-custom (computed
//! length tables) extradata paths.

use crate::decoder::decode_frame;
use crate::encoder::{
    bit_cost_for_method, encode_for_test, encode_for_test_with_mode, encode_frame_auto,
    encode_frame_with_mode, ExtradataMode, MethodSelection,
};
use crate::header::{Method, PixelFamily, StreamConfig};
use crate::tables::{
    compute_canonical_lengths, rle_decode_one_channel, rle_decode_three_channels,
    rle_encode_one_channel, rle_encode_three_channels,
};

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

fn synth_rgb32(width: usize, height: usize) -> Vec<u8> {
    let mut v = vec![0u8; width * 4 * height];
    for row in 0..height {
        for col in 0..width {
            let off = (row * width + col) * 4;
            v[off] = ((row + col) & 0xFF) as u8; // B
            v[off + 1] = ((row * 5 + col * 11) & 0xFF) as u8; // G
            v[off + 2] = ((row * 13 + col * 7) & 0xFF) as u8; // R
            v[off + 3] = ((row * 3 + col) & 0xFF) as u8; // A
        }
    }
    v
}

// ───────── default-mode (ClassicV2) — every legal (family, method) pair ─────────
//
// Legal pairs per spec/01 §3.1:
//
// - YUY2: PredictOld, Left, Gradient, Median (4)
// - RGB24: PredictOld, Left, LeftDecorr, GradientDecorr (4)
// - RGB32: PredictOld, Left, LeftDecorr, GradientDecorr (4)

#[test]
fn roundtrip_yuy2_predict_old_8x4() {
    let pixels = synth_yuy2(8, 4);
    let (cfg, frame) =
        encode_for_test(PixelFamily::Yuy2, Method::PredictOld, 8, 4, &pixels).unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

#[test]
fn roundtrip_yuy2_left_4x4() {
    let pixels = synth_yuy2(4, 4);
    let (cfg, frame) = encode_for_test(PixelFamily::Yuy2, Method::Left, 4, 4, &pixels).unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

#[test]
fn roundtrip_yuy2_left_8x4() {
    let pixels = synth_yuy2(8, 4);
    let (cfg, frame) = encode_for_test(PixelFamily::Yuy2, Method::Left, 8, 4, &pixels).unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

#[test]
fn roundtrip_yuy2_gradient_8x4() {
    // YUY2 + gradient predictor: LEFT pass over row 0, gradient post-pass
    // for rows ≥ 1. spec/03 §2.2.
    let pixels = synth_yuy2(8, 4);
    let (cfg, frame) = encode_for_test(PixelFamily::Yuy2, Method::Gradient, 8, 4, &pixels).unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

#[test]
fn roundtrip_yuy2_median_8x6() {
    let pixels = synth_yuy2(8, 6);
    let (cfg, frame) = encode_for_test(PixelFamily::Yuy2, Method::Median, 8, 6, &pixels).unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

/// Round-196 regression: narrow YUY2 Median (width = 2 → row_bytes = 4,
/// which is less than the 8-wire-byte row-1 LEFT exemption defined in
/// spec/03 §2.3.2 + audit/01 §7.2). Earlier `forward_median_subtract`
/// under-sized the LEFT region by clamping the exemption with
/// `8.min(row_bytes)`, while the decoder's `inverse_yuy2_median` used
/// the unclamped `row_bytes + 8` per the spec — the asymmetry produced
/// non-roundtripping wires on every `width < 4` YUY2 Median input.
/// Found by the `encode_huffyuv.rs` cargo-fuzz target. The 18 rows are
/// large enough to push the median region well past the LEFT
/// exemption.
#[test]
fn roundtrip_yuy2_median_2x18_round196() {
    let pixels: Vec<u8> = (0..2 * 18 * 2)
        .map(|x| ((x * 11 + 5) ^ 0x3A) as u8)
        .collect();
    let (cfg, frame) = encode_for_test(PixelFamily::Yuy2, Method::Median, 2, 18, &pixels).unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

/// Round-196 regression — minimal reproducer from the
/// `encode_huffyuv.rs` libfuzzer crash: prefix
/// `[80, 255, 17, 80, 175]` then all-zero pixel tail. With
/// pre-fix `forward_median_subtract` the round-2 byte at
/// position 8 reconstructed to `98` instead of `0`, producing the
/// repeating `98, 0, 98, 0, …` corruption from row 2 onward.
#[test]
fn roundtrip_yuy2_median_2x18_round196_fuzz_minimal() {
    let mut pixels = vec![0u8; 2 * 18 * 2];
    pixels[..5].copy_from_slice(&[80, 255, 17, 80, 175]);
    let (cfg, frame) = encode_for_test(PixelFamily::Yuy2, Method::Median, 2, 18, &pixels).unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

/// Round-202 boundary coverage: exercise the YUY2 Median tail-loop
/// at every per-frame size that brackets the LEFT exemption +
/// AL-index invariants the dead-branch strip relies on.
///
/// - `width = 2 → row_bytes = 4` (narrow): the LEFT exemption end is
///   `row_bytes + 8 = 12`, so the median region is non-empty only
///   when `len > 12`, i.e. `height >= 4` (`len = 4 × 4 = 16 > 12`).
///   This is the boundary the round-196 wire-asymmetry fix targeted;
///   round 202 confirms the dead-branch strip in the post-exemption
///   loop is still wire-stable across the size sweep that brackets
///   `row1_left_end == len` (height 3 → no median region) and
///   `row1_left_end < len` (height 4+ → first median byte at pos 12).
/// - `width = 4 → row_bytes = 8`: the LEFT exemption boundary sits
///   exactly on the row-2 start (`row_bytes + 8 == 2 × row_bytes`).
///   Height 2 → no median region; height 3 → 8 median bytes; height
///   4 → 16 median bytes. The AL lookback at the first median pos
///   reads from row 0 col 6, A from row 1 col 0, L from row 1 col 6
///   — all three reads landing in already-LEFT-decoded territory.
#[test]
fn roundtrip_yuy2_median_round202_boundary_widths() {
    // Skewed pseudo-random pattern that won't accidentally degenerate
    // to all-zero residuals (which would mask many predictor bugs).
    fn synth(n: usize) -> Vec<u8> {
        (0..n).map(|x| ((x * 31 + 7) ^ 0xA5) as u8).collect()
    }
    for &(w, h) in &[
        (2u32, 4u32),
        (2, 5),
        (2, 6),
        (2, 8),
        (4, 3),
        (4, 4),
        (4, 8),
        (6, 4),
        (8, 4),
    ] {
        let pixels = synth((w * h * 2) as usize);
        let (cfg, frame) = encode_for_test(PixelFamily::Yuy2, Method::Median, w, h, &pixels)
            .unwrap_or_else(|e| panic!("encode failed for {}x{}: {:?}", w, h, e));
        let out = decode_frame(&cfg, &frame)
            .unwrap_or_else(|e| panic!("decode failed for {}x{}: {:?}", w, h, e));
        assert_eq!(
            out.pixels, pixels,
            "YUY2 Median roundtrip failed at {}x{}",
            w, h
        );
    }
}

#[test]
fn roundtrip_rgb24_predict_old_4x4() {
    let pixels = synth_rgb24(4, 4);
    let (cfg, frame) =
        encode_for_test(PixelFamily::Rgb24, Method::PredictOld, 4, 4, &pixels).unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

#[test]
fn roundtrip_rgb24_left_4x4() {
    let pixels = synth_rgb24(4, 4);
    let (cfg, frame) = encode_for_test(PixelFamily::Rgb24, Method::Left, 4, 4, &pixels).unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

#[test]
fn roundtrip_rgb24_left_decorr_4x4() {
    let pixels = synth_rgb24(4, 4);
    let (cfg, frame) =
        encode_for_test(PixelFamily::Rgb24, Method::LeftDecorr, 4, 4, &pixels).unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

#[test]
fn roundtrip_rgb24_gradient_decorr_6x4() {
    let pixels = synth_rgb24(6, 4);
    let (cfg, frame) =
        encode_for_test(PixelFamily::Rgb24, Method::GradientDecorr, 6, 4, &pixels).unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

#[test]
fn roundtrip_rgb32_predict_old_4x4() {
    let pixels = synth_rgb32(4, 4);
    let (cfg, frame) =
        encode_for_test(PixelFamily::Rgb32, Method::PredictOld, 4, 4, &pixels).unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

#[test]
fn roundtrip_rgb32_left_4x4() {
    let pixels = synth_rgb32(4, 4);
    let (cfg, frame) = encode_for_test(PixelFamily::Rgb32, Method::Left, 4, 4, &pixels).unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

#[test]
fn roundtrip_rgb32_left_decorr_4x4() {
    let pixels = synth_rgb32(4, 4);
    let (cfg, frame) =
        encode_for_test(PixelFamily::Rgb32, Method::LeftDecorr, 4, 4, &pixels).unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

#[test]
fn roundtrip_rgb32_gradient_decorr_4x4() {
    let pixels = synth_rgb32(4, 4);
    let (cfg, frame) =
        encode_for_test(PixelFamily::Rgb32, Method::GradientDecorr, 4, 4, &pixels).unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

// ───────── round 103: fused decorrelation+gradient encoder path ─────────
//
// Round 103 folds the RGB decorrelation transform into the
// forward-gradient subtract for method 0x41 (GradientDecorr), reading
// un-transformed `pixels` directly so the round-95 `working_owned`
// decorrelated buffer is no longer allocated (spec/03 §2.4.2 — the
// gradient pre-pass reads only decorrelated channel values). The byte
// identity of the fused vs two-pass gradient pre-pass is unit-tested in
// `predict.rs`; these are end-to-end wire round-trips on non-aligned
// widths and a per-row-varying-alpha RGB32 frame, complementing the
// small-size GradientDecorr round-trips above.

#[test]
fn round103_roundtrip_rgb24_gradient_decorr_7x5_custom_v2() {
    // Width 7 → row_bytes 21 (not u64-aligned); exercises the
    // forward_decorr_gradient_subtract tail path on every row.
    let pixels = synth_rgb24(7, 5);
    let (cfg, frame) = encode_for_test_with_mode(
        PixelFamily::Rgb24,
        Method::GradientDecorr,
        7,
        5,
        &pixels,
        ExtradataMode::CustomV2,
    )
    .unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

#[test]
fn round103_roundtrip_rgb32_gradient_decorr_6x4_v1x_compat() {
    let pixels = synth_rgb32(6, 4);
    let (cfg, frame) = encode_for_test_with_mode(
        PixelFamily::Rgb32,
        Method::GradientDecorr,
        6,
        4,
        &pixels,
        ExtradataMode::V1xCompat,
    )
    .unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

#[test]
fn round103_roundtrip_rgb32_gradient_decorr_per_row_alpha() {
    // Alpha varies per row (constant within a row) — confirms the fused
    // GradientDecorr path keeps alpha identity in BOTH the decorrelation
    // step and the gradient subtract (spec/03 §2.4 Validator note).
    let (w, h) = (5usize, 6usize);
    let mut pixels = vec![0u8; w * 4 * h];
    for row in 0..h {
        for col in 0..w {
            let off = (row * w + col) * 4;
            pixels[off] = ((row * 3 + col * 9) & 0xFF) as u8; // B
            pixels[off + 1] = ((row * 7 + col * 5) & 0xFF) as u8; // G
            pixels[off + 2] = ((row * 11 + col) & 0xFF) as u8; // R
            pixels[off + 3] = ((row * 16) & 0xFF) as u8; // A: per-row
        }
    }
    let (cfg, frame) = encode_for_test_with_mode(
        PixelFamily::Rgb32,
        Method::GradientDecorr,
        w as u32,
        h as u32,
        &pixels,
        ExtradataMode::CustomV2,
    )
    .unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

#[test]
fn round103_roundtrip_rgb24_gradient_decorr_interlaced_300() {
    // Height 300 > 288 trips the interlaced field-stride=2 path; each
    // field is GradientDecorr-encoded through the fused helper.
    let pixels = synth_rgb24(8, 300);
    let (cfg, frame) = encode_for_test_with_mode(
        PixelFamily::Rgb24,
        Method::GradientDecorr,
        8,
        300,
        &pixels,
        ExtradataMode::ClassicV2,
    )
    .unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

// ───────── v2.x custom (computed length tables) ─────────
//
// The CustomV2 path skips the classic blobs entirely and synthesises
// per-stream length tables via the package-merge length-limited
// Huffman builder. We exercise it on a few combinations to prove the
// histogram/RLE/length-decode loop is round-trip-correct.

#[test]
fn roundtrip_yuy2_left_custom_v2_8x4() {
    let pixels = synth_yuy2(8, 4);
    let (cfg, frame) = encode_for_test_with_mode(
        PixelFamily::Yuy2,
        Method::Left,
        8,
        4,
        &pixels,
        ExtradataMode::CustomV2,
    )
    .unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

#[test]
fn roundtrip_yuy2_gradient_custom_v2_8x4() {
    let pixels = synth_yuy2(8, 4);
    let (cfg, frame) = encode_for_test_with_mode(
        PixelFamily::Yuy2,
        Method::Gradient,
        8,
        4,
        &pixels,
        ExtradataMode::CustomV2,
    )
    .unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

#[test]
fn roundtrip_yuy2_median_custom_v2_8x6() {
    let pixels = synth_yuy2(8, 6);
    let (cfg, frame) = encode_for_test_with_mode(
        PixelFamily::Yuy2,
        Method::Median,
        8,
        6,
        &pixels,
        ExtradataMode::CustomV2,
    )
    .unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

#[test]
fn roundtrip_rgb24_left_decorr_custom_v2_4x4() {
    let pixels = synth_rgb24(4, 4);
    let (cfg, frame) = encode_for_test_with_mode(
        PixelFamily::Rgb24,
        Method::LeftDecorr,
        4,
        4,
        &pixels,
        ExtradataMode::CustomV2,
    )
    .unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

#[test]
fn roundtrip_rgb24_gradient_decorr_custom_v2_6x4() {
    let pixels = synth_rgb24(6, 4);
    let (cfg, frame) = encode_for_test_with_mode(
        PixelFamily::Rgb24,
        Method::GradientDecorr,
        6,
        4,
        &pixels,
        ExtradataMode::CustomV2,
    )
    .unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

#[test]
fn roundtrip_rgb32_left_custom_v2_4x4() {
    let pixels = synth_rgb32(4, 4);
    let (cfg, frame) = encode_for_test_with_mode(
        PixelFamily::Rgb32,
        Method::Left,
        4,
        4,
        &pixels,
        ExtradataMode::CustomV2,
    )
    .unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

// ───────── v1.x compat (`biSize == 0x28`) ─────────
//
// Verifies the encoder can emit a no-extradata stream that the decoder
// reads via the v1.x precomputed-codes path. Every symbol in the
// residual stream must have a non-zero length in the v1.x codebook;
// the encoder's verify_residuals_in_table check ensures this.

#[test]
fn roundtrip_yuy2_predict_old_v1x_8x4() {
    let pixels = synth_yuy2(8, 4);
    let (cfg, frame) = encode_for_test_with_mode(
        PixelFamily::Yuy2,
        Method::PredictOld,
        8,
        4,
        &pixels,
        ExtradataMode::V1xCompat,
    )
    .unwrap();
    assert!(!cfg.has_extradata);
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

#[test]
fn roundtrip_rgb24_predict_old_v1x_4x4() {
    let pixels = synth_rgb24(4, 4);
    let (cfg, frame) = encode_for_test_with_mode(
        PixelFamily::Rgb24,
        Method::PredictOld,
        4,
        4,
        &pixels,
        ExtradataMode::V1xCompat,
    )
    .unwrap();
    assert!(!cfg.has_extradata);
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

#[test]
fn roundtrip_rgb32_predict_old_v1x_4x4() {
    let pixels = synth_rgb32(4, 4);
    let (cfg, frame) = encode_for_test_with_mode(
        PixelFamily::Rgb32,
        Method::PredictOld,
        4,
        4,
        &pixels,
        ExtradataMode::V1xCompat,
    )
    .unwrap();
    assert!(!cfg.has_extradata);
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

// ───────── v1.x compat — Left / Gradient / Median (extending round 2) ─────────
//
// Round 3 wires the verify pass through ALL legal (family, method)
// pairs; the v1.x codebook covers all 256 symbols (set A max length
// 17, set B max length 26 — confirmed at table-build time), so any
// residual stream produced by the encoder is decodable. The decoder's
// `low3 → method` resolution from `biBitCount` (spec/01 §1.4) routes
// the parse correctly for each method; we assert here that the
// reconstructed pixels match the source.

#[test]
fn roundtrip_yuy2_left_v1x_8x4() {
    let pixels = synth_yuy2(8, 4);
    let (cfg, frame) = encode_for_test_with_mode(
        PixelFamily::Yuy2,
        Method::Left,
        8,
        4,
        &pixels,
        ExtradataMode::V1xCompat,
    )
    .unwrap();
    assert!(!cfg.has_extradata);
    assert_eq!(cfg.method, Method::Left);
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

#[test]
fn roundtrip_yuy2_gradient_v1x_8x4() {
    let pixels = synth_yuy2(8, 4);
    let (cfg, frame) = encode_for_test_with_mode(
        PixelFamily::Yuy2,
        Method::Gradient,
        8,
        4,
        &pixels,
        ExtradataMode::V1xCompat,
    )
    .unwrap();
    assert!(!cfg.has_extradata);
    assert_eq!(cfg.method, Method::Gradient);
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

#[test]
fn roundtrip_yuy2_median_v1x_8x6() {
    let pixels = synth_yuy2(8, 6);
    let (cfg, frame) = encode_for_test_with_mode(
        PixelFamily::Yuy2,
        Method::Median,
        8,
        6,
        &pixels,
        ExtradataMode::V1xCompat,
    )
    .unwrap();
    assert!(!cfg.has_extradata);
    assert_eq!(cfg.method, Method::Median);
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

#[test]
fn roundtrip_rgb24_left_v1x_4x4() {
    let pixels = synth_rgb24(4, 4);
    let (cfg, frame) = encode_for_test_with_mode(
        PixelFamily::Rgb24,
        Method::Left,
        4,
        4,
        &pixels,
        ExtradataMode::V1xCompat,
    )
    .unwrap();
    assert!(!cfg.has_extradata);
    assert_eq!(cfg.method, Method::Left);
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

#[test]
fn roundtrip_rgb24_left_decorr_v1x_4x4() {
    let pixels = synth_rgb24(4, 4);
    let (cfg, frame) = encode_for_test_with_mode(
        PixelFamily::Rgb24,
        Method::LeftDecorr,
        4,
        4,
        &pixels,
        ExtradataMode::V1xCompat,
    )
    .unwrap();
    assert!(!cfg.has_extradata);
    assert_eq!(cfg.method, Method::LeftDecorr);
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

#[test]
fn roundtrip_rgb24_gradient_decorr_v1x_6x4() {
    let pixels = synth_rgb24(6, 4);
    let (cfg, frame) = encode_for_test_with_mode(
        PixelFamily::Rgb24,
        Method::GradientDecorr,
        6,
        4,
        &pixels,
        ExtradataMode::V1xCompat,
    )
    .unwrap();
    assert!(!cfg.has_extradata);
    assert_eq!(cfg.method, Method::GradientDecorr);
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

#[test]
fn roundtrip_rgb32_left_v1x_4x4() {
    let pixels = synth_rgb32(4, 4);
    let (cfg, frame) = encode_for_test_with_mode(
        PixelFamily::Rgb32,
        Method::Left,
        4,
        4,
        &pixels,
        ExtradataMode::V1xCompat,
    )
    .unwrap();
    assert!(!cfg.has_extradata);
    assert_eq!(cfg.method, Method::Left);
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

#[test]
fn roundtrip_rgb32_left_decorr_v1x_4x4() {
    let pixels = synth_rgb32(4, 4);
    let (cfg, frame) = encode_for_test_with_mode(
        PixelFamily::Rgb32,
        Method::LeftDecorr,
        4,
        4,
        &pixels,
        ExtradataMode::V1xCompat,
    )
    .unwrap();
    assert!(!cfg.has_extradata);
    assert_eq!(cfg.method, Method::LeftDecorr);
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

#[test]
fn roundtrip_rgb32_gradient_decorr_v1x_4x4() {
    let pixels = synth_rgb32(4, 4);
    let (cfg, frame) = encode_for_test_with_mode(
        PixelFamily::Rgb32,
        Method::GradientDecorr,
        4,
        4,
        &pixels,
        ExtradataMode::V1xCompat,
    )
    .unwrap();
    assert!(!cfg.has_extradata);
    assert_eq!(cfg.method, Method::GradientDecorr);
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

// ───────── primary-LUT decoder fast-path coverage ─────────
//
// `tables::decode_one` now hits a 65 536-entry primary LUT for every
// code length ≤ 16. Long codes (length > 16) trickle through
// `decode_one_slow`. v1.x set B has max length 26, so the slow path
// is exercised by every v1.x roundtrip above. The check below
// instruments the LUT directly to confirm it returns the same answer
// as the per-symbol entries on every well-formed window.

#[test]
fn primary_lut_matches_per_symbol_entries_classic_yuv_left() {
    use crate::tables::{decode_one, rle_decode_three_channels, HuffTable};
    // Use the YUV-LEFT classic blob's slot-1 (Y) length table.
    let blob = crate::tables::classic_blob_bytes(PixelFamily::Yuy2, Method::Left);
    let lengths = rle_decode_three_channels(blob).unwrap();
    let table = HuffTable::build_from_lengths(&lengths[0]).unwrap();
    // For every symbol whose length ≤ 16, the LUT slot under the
    // canonical code's prefix must yield (sym, length) directly.
    for (sym, e) in table.entries.iter().enumerate() {
        if e.length == 0 || e.length > 16 {
            continue;
        }
        // Construct a window whose top `length` bits exactly match
        // the canonical code for `sym`; trailing bits arbitrary.
        let window = e.code | 0xDEAD; // arbitrary low garbage
        let (decoded_sym, decoded_len) = decode_one(&table, window).unwrap();
        assert_eq!(decoded_sym as usize, sym, "sym {sym} mismatched");
        assert_eq!(decoded_len, e.length, "len for sym {sym} mismatched");
    }
}

#[test]
fn primary_lut_overflow_falls_back_to_slow_path_v1x_set_b() {
    use crate::tables::{
        decode_one, rle_decode_one_channel, v1x_codes_set_b, v1x_lengths_set_b, v1x_table_from_pair,
    };
    let mut cur: &[u8] = v1x_lengths_set_b();
    let lens_b = rle_decode_one_channel(&mut cur).unwrap();
    let codes_b = v1x_codes_set_b();
    let mut codes_arr = [0u8; 256];
    codes_arr.copy_from_slice(codes_b);
    let table = v1x_table_from_pair(&lens_b, &codes_arr).unwrap();
    // v1.x set B max length is 26 — the LUT will mark some prefixes
    // as overflow; pick a symbol with length > 16 and confirm the
    // slow path returns it.
    let (sym, e) = table
        .entries
        .iter()
        .enumerate()
        .find(|(_, e)| e.length > 16)
        .expect("set B has codes longer than 16 bits");
    let window = e.code; // exact MSB-aligned code; trailing bits = 0
    let (decoded_sym, decoded_len) = decode_one(&table, window).unwrap();
    assert_eq!(decoded_sym as usize, sym);
    assert_eq!(decoded_len, e.length);
}

// ───────── tables — RLE encoder + canonical-length builder unit tests ─────────

#[test]
fn rle_encode_round_trips_uniform_run() {
    let mut lens = [0u8; 256];
    for slot in lens.iter_mut() {
        *slot = 5;
    }
    let mut buf = Vec::new();
    rle_encode_one_channel(&lens, &mut buf).unwrap();
    let mut cur: &[u8] = &buf;
    let decoded = rle_decode_one_channel(&mut cur).unwrap();
    assert_eq!(decoded, lens);
}

#[test]
fn rle_encode_round_trips_mixed_lengths() {
    let mut lens = [0u8; 256];
    for (i, slot) in lens.iter_mut().enumerate() {
        *slot = ((i * 31 + 7) % 32) as u8;
    }
    let mut buf = Vec::new();
    rle_encode_one_channel(&lens, &mut buf).unwrap();
    let mut cur: &[u8] = &buf;
    let decoded = rle_decode_one_channel(&mut cur).unwrap();
    assert_eq!(decoded, lens);
}

#[test]
fn rle_encode_three_channels_round_trips() {
    let mut a = [0u8; 256];
    let mut b = [0u8; 256];
    let mut c = [0u8; 256];
    for i in 0..256 {
        a[i] = ((i * 3) % 32) as u8;
        b[i] = ((i * 5 + 1) % 32) as u8;
        c[i] = ((i * 7 + 2) % 32) as u8;
    }
    let blob = rle_encode_three_channels(&[a, b, c]).unwrap();
    let decoded = rle_decode_three_channels(&blob).unwrap();
    assert_eq!(decoded[0], a);
    assert_eq!(decoded[1], b);
    assert_eq!(decoded[2], c);
}

#[test]
fn canonical_lengths_uniform_histogram_assigns_8_bit_codes() {
    let mut h = [0u32; 256];
    for slot in h.iter_mut() {
        *slot = 1;
    }
    let lens = compute_canonical_lengths(&h).unwrap();
    // 256 equiprobable symbols → 8-bit codes for all.
    for &l in lens.iter() {
        assert_eq!(l, 8);
    }
}

#[test]
fn canonical_lengths_skewed_histogram_under_31_bits() {
    let mut h = [0u32; 256];
    for (i, slot) in h.iter_mut().enumerate() {
        *slot = (256 - i) as u32 * (256 - i) as u32;
    }
    let lens = compute_canonical_lengths(&h).unwrap();
    let max = *lens.iter().max().unwrap();
    assert!(max <= 31);
    // Kraft equality.
    let mut kraft: u64 = 0;
    for &l in lens.iter() {
        if l > 0 {
            kraft += 1u64 << (31 - l);
        }
    }
    assert_eq!(kraft, 1u64 << 31);
}

#[test]
fn canonical_lengths_single_symbol_assigns_one_bit() {
    let mut h = [0u32; 256];
    h[7] = 100;
    let lens = compute_canonical_lengths(&h).unwrap();
    // Round-6 fix: round-out single-symbol histograms with a length-1
    // dummy entry on symbol 0 so the canonical builder's Kraft
    // accumulator still wraps to 0 in `HuffTable::build_from_lengths`.
    // The dummy is never emitted (its histogram count is 0).
    assert_eq!(lens[7], 1);
    assert_eq!(lens[0], 1);
    for (i, &l) in lens.iter().enumerate() {
        if i != 7 && i != 0 {
            assert_eq!(l, 0, "stray length at symbol {i}");
        }
    }
}

// ───────── Round-4: interlaced field-stride=2 (height > 288) ─────────
//
// spec/02 §2 / spec/05 (planned): when biHeight > 288 the codec
// splits the frame into two fields (even rows = top; odd rows =
// bottom), predicts each independently, and concatenates the two
// bit-streams in the per-frame chunk. These tests verify the
// self-roundtrip for that path across the three pixel families and
// the four predictor methods.

#[test]
fn roundtrip_yuy2_left_interlaced_8x300() {
    // 8×300 YUY2 = 4 800 bytes. height > 288 → interlaced path.
    let pixels = synth_yuy2(8, 300);
    let (cfg, frame) = encode_for_test(PixelFamily::Yuy2, Method::Left, 8, 300, &pixels).unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.height, 300);
    assert_eq!(out.pixels, pixels);
}

#[test]
fn roundtrip_yuy2_gradient_interlaced_8x320() {
    let pixels = synth_yuy2(8, 320);
    let (cfg, frame) =
        encode_for_test(PixelFamily::Yuy2, Method::Gradient, 8, 320, &pixels).unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

#[test]
fn roundtrip_yuy2_median_interlaced_8x300() {
    let pixels = synth_yuy2(8, 300);
    let (cfg, frame) = encode_for_test(PixelFamily::Yuy2, Method::Median, 8, 300, &pixels).unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

#[test]
fn roundtrip_rgb24_left_decorr_interlaced_8x300() {
    let pixels = synth_rgb24(8, 300);
    let (cfg, frame) =
        encode_for_test(PixelFamily::Rgb24, Method::LeftDecorr, 8, 300, &pixels).unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

#[test]
fn roundtrip_rgb24_gradient_decorr_interlaced_8x320() {
    let pixels = synth_rgb24(8, 320);
    let (cfg, frame) =
        encode_for_test(PixelFamily::Rgb24, Method::GradientDecorr, 8, 320, &pixels).unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

#[test]
fn roundtrip_rgb32_left_decorr_interlaced_8x300() {
    let pixels = synth_rgb32(8, 300);
    let (cfg, frame) =
        encode_for_test(PixelFamily::Rgb32, Method::LeftDecorr, 8, 300, &pixels).unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

#[test]
fn roundtrip_rgb32_predict_old_interlaced_8x300() {
    let pixels = synth_rgb32(8, 300);
    let (cfg, frame) =
        encode_for_test(PixelFamily::Rgb32, Method::PredictOld, 8, 300, &pixels).unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

#[test]
fn roundtrip_yuy2_left_just_below_threshold_8x288() {
    // Boundary: 288 = NOT interlaced; 289 = interlaced.
    let pixels = synth_yuy2(8, 288);
    let (cfg, frame) = encode_for_test(PixelFamily::Yuy2, Method::Left, 8, 288, &pixels).unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

#[test]
fn roundtrip_yuy2_left_just_above_threshold_8x290() {
    let pixels = synth_yuy2(8, 290);
    let (cfg, frame) = encode_for_test(PixelFamily::Yuy2, Method::Left, 8, 290, &pixels).unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

#[test]
fn roundtrip_yuy2_custom_v2_interlaced_8x300() {
    // Interlaced + CustomV2 (per-frame computed length tables): the
    // histogram must be the union of both fields' residuals, which we
    // verify by encode/decode round-tripping.
    let pixels = synth_yuy2(8, 300);
    let (cfg, frame) = encode_for_test_with_mode(
        PixelFamily::Yuy2,
        Method::Left,
        8,
        300,
        &pixels,
        ExtradataMode::CustomV2,
    )
    .unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

// ───────── Round-5: walking-stride interlaced encoder ─────────
//
// The round-5 encoder eliminates the round-4 `predict::split_fields`
// allocations: instead of building two contiguous half-height buffers
// (`top` + `bot`), the encoder walks the source raster with row-stride
// 2 into a single field-sized scratch buffer, reused across both
// fields, with one combined residual-body Vec for histogram + verify.
//
// These tests verify:
// 1. The walking-stride path remains bit-identical to the round-4
//    `split_fields` flow on every (family, method) interlaced pair
//    (round-trip self-check; if either path drifted, decode would
//    fail).
// 2. Odd-height interlaced (top has one more row than bot) round-trips.
// 3. A larger interlaced frame (480p-class) round-trips without a
//    panic — exercises the per-field row-bytes × field-h math the
//    walking-stride path uses to size scratch + combined-body.

#[test]
fn roundtrip_yuy2_left_interlaced_odd_height_8x301() {
    // 301 = top has 151 rows, bot has 150 rows (top ≠ bot is the
    // interesting case: walking-stride must resize scratch down for
    // bot, and combined-body must size to the asymmetric total).
    let pixels = synth_yuy2(8, 301);
    let (cfg, frame) = encode_for_test(PixelFamily::Yuy2, Method::Left, 8, 301, &pixels).unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.height, 301);
    assert_eq!(out.pixels, pixels);
}

#[test]
fn roundtrip_rgb24_predict_old_interlaced_odd_height_8x301() {
    let pixels = synth_rgb24(8, 301);
    let (cfg, frame) =
        encode_for_test(PixelFamily::Rgb24, Method::PredictOld, 8, 301, &pixels).unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

#[test]
fn roundtrip_rgb32_gradient_decorr_interlaced_odd_height_8x301() {
    let pixels = synth_rgb32(8, 301);
    let (cfg, frame) =
        encode_for_test(PixelFamily::Rgb32, Method::GradientDecorr, 8, 301, &pixels).unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

#[test]
fn roundtrip_yuy2_left_interlaced_480p_class() {
    // 96 × 480 = 480p-class interlaced frame, 480 > 288 so the
    // walking-stride field path engages. ~92 KB working set; sanity
    // check that the scratch + combined-body sizing is correct at
    // a frame size larger than the small interlaced cases above.
    let pixels = synth_yuy2(96, 480);
    let (cfg, frame) = encode_for_test(PixelFamily::Yuy2, Method::Left, 96, 480, &pixels).unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.width, 96);
    assert_eq!(out.height, 480);
    assert_eq!(out.pixels, pixels);
}

#[test]
fn roundtrip_rgb24_left_decorr_interlaced_480p_class() {
    let pixels = synth_rgb24(64, 480);
    let (cfg, frame) =
        encode_for_test(PixelFamily::Rgb24, Method::LeftDecorr, 64, 480, &pixels).unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

#[test]
fn roundtrip_yuy2_median_interlaced_v1x_compat() {
    // Round-5 walking-stride path × V1xCompat extradata: ensures the
    // combined-body verify pass still walks slot phases correctly when
    // the bot half follows the top half in one Vec.
    let pixels = synth_yuy2(8, 300);
    let (cfg, frame) = encode_for_test_with_mode(
        PixelFamily::Yuy2,
        Method::Median,
        8,
        300,
        &pixels,
        ExtradataMode::V1xCompat,
    )
    .unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

#[test]
fn roundtrip_rgb24_left_decorr_interlaced_custom_v2_walking_stride() {
    // Walking-stride × CustomV2 × RGB24-decorr: the combined-body's
    // bot half MUST start at `i % 3 == 0` for the histogram + emit
    // slot mapping to remain correct. If `top.body_len % 3` ever
    // drifted from 0 (e.g. someone reintroduced the 4-byte-seed at
    // the wrong end), this test would catch it.
    let pixels = synth_rgb24(8, 320);
    let (cfg, frame) = encode_for_test_with_mode(
        PixelFamily::Rgb24,
        Method::LeftDecorr,
        8,
        320,
        &pixels,
        ExtradataMode::CustomV2,
    )
    .unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

#[test]
fn roundtrip_rgb32_predict_old_interlaced_custom_v2_walking_stride() {
    // Walking-stride × CustomV2 × RGB32: bot section starts at
    // `i % 4 == 0` since each top-pixel residual contributes 4 body
    // bytes.
    let pixels = synth_rgb32(8, 320);
    let (cfg, frame) = encode_for_test_with_mode(
        PixelFamily::Rgb32,
        Method::PredictOld,
        8,
        320,
        &pixels,
        ExtradataMode::CustomV2,
    )
    .unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

// ───────── round-6: bit-cost-driven encoder method auto-selection ─────────
//
// The new `encode_frame_auto` API runs every legal predictor for the
// family, scores each one with `bit_cost_for_method` (Σ length×count
// using package-merge optimal lengths), and emits the winner. The
// tests below assert four things:
//
//   (a) the auto-selected output round-trips identically to the input,
//   (b) the chosen method is at least as good (cost-wise) as any
//       individually scored candidate,
//   (c) on inputs with a structural bias (smooth gradient → Gradient
//       wins; piecewise-constant → Left wins; color-correlated RGB
//       → Decorr wins), Auto picks the expected predictor,
//   (d) Auto + interlaced + CustomV2 still round-trips at heights
//       > 288.

/// Constant-luma frame — Left should win (every residual is zero
/// after subtracting the left neighbour).
fn synth_yuy2_constant_luma(width: usize, height: usize, y: u8, u: u8, v: u8) -> Vec<u8> {
    let mut buf = vec![0u8; width * 2 * height];
    for row in 0..height {
        for pair in 0..(width / 2) {
            let off = row * width * 2 + pair * 4;
            buf[off] = y;
            buf[off + 1] = u;
            buf[off + 2] = y;
            buf[off + 3] = v;
        }
    }
    buf
}

/// 2D-textured frame whose luma forms a diagonal ramp with a small
/// per-pixel jitter. Gradient (which subtracts row-above first)
/// reduces the per-byte variance more than LEFT does on this input,
/// so its package-merge length tables shrink the body.
fn synth_yuy2_diagonal_textured(width: usize, height: usize) -> Vec<u8> {
    let mut buf = vec![0u8; width * 2 * height];
    for row in 0..height {
        for col2 in 0..(width * 2) {
            // Diagonal slope of ~1 per pixel and per row, plus a
            // slow-varying jitter that's not aligned with either
            // axis (so neither LEFT-along-rows nor pure-row-diff
            // perfectly cancels it).
            let v =
                (row.wrapping_mul(5).wrapping_add(col2.wrapping_mul(7)) ^ (row & 0x3) << 4) & 0xFF;
            buf[row * width * 2 + col2] = v as u8;
        }
    }
    buf
}

/// RGB24 with strongly-correlated chroma: G varies pseudo-randomly,
/// B == G + small fixed offset, R == G + small fixed offset.  After
/// LEFT prediction the per-channel residual distributions are
/// approximately a wide gaussian around 0.  After LEFT-decorr the
/// slot1 (B - G) and slot3 (R - G) channels are nearly constant
/// (only the per-pixel difference of the fixed offset survives,
/// which is zero everywhere), so they compress to a small number of
/// bits while slot2 (G) keeps the same cost as plain Left.
/// Net effect: decorr should win on this input.
fn synth_rgb24_grey_with_chroma(width: usize, height: usize) -> Vec<u8> {
    let mut buf = vec![0u8; width * 3 * height];
    // A pseudo-random luma generator: linear-congruential, gives a
    // fairly wide G distribution so the per-channel histograms are
    // wide enough for package-merge to assign codes longer than 1
    // bit on average.
    let mut state: u32 = 0x1234_5678;
    let mut next = || {
        state = state.wrapping_mul(1_103_515_245).wrapping_add(12345);
        (state >> 16) as u8
    };
    for row in 0..height {
        for col in 0..width {
            let off = (row * width + col) * 3;
            let g = next();
            // Fixed per-image offsets so B-G and R-G end up constant
            // (slot1, slot3 single-symbol histograms after decorr).
            buf[off] = g.wrapping_add(3); // B
            buf[off + 1] = g; // G
            buf[off + 2] = g.wrapping_add(7); // R
        }
    }
    buf
}

#[test]
fn auto_picks_left_for_constant_luma_yuy2() {
    let pixels = synth_yuy2_constant_luma(16, 8, 0x80, 0x80, 0x80);
    let (_strf, _frame, chosen) = encode_frame_auto(
        PixelFamily::Yuy2,
        MethodSelection::Auto,
        16,
        8,
        &pixels,
        ExtradataMode::CustomV2,
    )
    .unwrap();
    // Left or Gradient both make all-zero residuals on a constant
    // frame, so the cost will tie at zero body bits. The tie-break
    // rule (first in `legal_methods` order) makes Left the winner.
    assert_eq!(chosen, Method::Left);
}

#[test]
fn auto_picks_lower_or_equal_cost_yuy2_diagonal() {
    // Diagonal-textured YUY2 with a slow per-pixel jitter: at this
    // size, Gradient and Left produce different package-merge
    // length tables. The auto-picker should land on whichever is
    // smaller — we just confirm the relationship rather than pin a
    // specific predictor, since the metric depends on the
    // package-merge tie-break.
    let pixels = synth_yuy2_diagonal_textured(32, 32);
    let costs: Vec<(Method, u64)> = [Method::Left, Method::Gradient, Method::Median]
        .iter()
        .map(|&m| {
            (
                m,
                bit_cost_for_method(PixelFamily::Yuy2, m, 32, 32, &pixels).unwrap(),
            )
        })
        .collect();
    let min_cost = costs.iter().map(|&(_, c)| c).min().unwrap();
    let (_strf, _frame, chosen) = encode_frame_auto(
        PixelFamily::Yuy2,
        MethodSelection::Auto,
        32,
        32,
        &pixels,
        ExtradataMode::CustomV2,
    )
    .unwrap();
    let chosen_cost = costs.iter().find(|&&(m, _)| m == chosen).unwrap().1;
    assert_eq!(
        chosen_cost, min_cost,
        "auto winner {chosen:?} cost {chosen_cost} ≠ best {min_cost}; all={costs:?}"
    );
}

#[test]
fn auto_picks_decorr_or_better_for_chroma_correlated_rgb24() {
    let pixels = synth_rgb24_grey_with_chroma(32, 32);
    let cost_left = bit_cost_for_method(PixelFamily::Rgb24, Method::Left, 32, 32, &pixels).unwrap();
    let cost_decorr =
        bit_cost_for_method(PixelFamily::Rgb24, Method::LeftDecorr, 32, 32, &pixels).unwrap();
    // With B ≈ G ≈ R + small jitter, the decorrelated channels carry
    // less entropy than the plain channels — LeftDecorr should beat
    // Left on this input.
    assert!(
        cost_decorr < cost_left,
        "expected LeftDecorr ({cost_decorr}) < Left ({cost_left})"
    );
    let (_strf, _frame, chosen) = encode_frame_auto(
        PixelFamily::Rgb24,
        MethodSelection::Auto,
        32,
        32,
        &pixels,
        ExtradataMode::CustomV2,
    )
    .unwrap();
    // Whatever the auto-picker chose, its cost must equal the
    // minimum over all legal RGB methods.
    let costs: Vec<(Method, u64)> = [Method::Left, Method::LeftDecorr, Method::GradientDecorr]
        .iter()
        .map(|&m| {
            (
                m,
                bit_cost_for_method(PixelFamily::Rgb24, m, 32, 32, &pixels).unwrap(),
            )
        })
        .collect();
    let min_cost = costs.iter().map(|&(_, c)| c).min().unwrap();
    let chosen_cost = costs.iter().find(|&&(m, _)| m == chosen).unwrap().1;
    assert_eq!(
        chosen_cost, min_cost,
        "auto winner {chosen:?} cost {chosen_cost} ≠ best {min_cost}"
    );
    // And the chosen method must NOT be plain Left here.
    assert_ne!(chosen, Method::Left);
}

#[test]
fn auto_roundtrips_yuy2_custom_v2() {
    let pixels = synth_yuy2(16, 16);
    let (strf, frame, _chosen) = encode_frame_auto(
        PixelFamily::Yuy2,
        MethodSelection::Auto,
        16,
        16,
        &pixels,
        ExtradataMode::CustomV2,
    )
    .unwrap();
    let cfg = StreamConfig::parse_bitmapinfoheader(&strf).unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

#[test]
fn auto_roundtrips_rgb24_classic_v2() {
    let pixels = synth_rgb24(16, 16);
    let (strf, frame, _chosen) = encode_frame_auto(
        PixelFamily::Rgb24,
        MethodSelection::Auto,
        16,
        16,
        &pixels,
        ExtradataMode::ClassicV2,
    )
    .unwrap();
    let cfg = StreamConfig::parse_bitmapinfoheader(&strf).unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

#[test]
fn auto_roundtrips_rgb32_custom_v2() {
    let pixels = synth_rgb32(16, 16);
    let (strf, frame, _chosen) = encode_frame_auto(
        PixelFamily::Rgb32,
        MethodSelection::Auto,
        16,
        16,
        &pixels,
        ExtradataMode::CustomV2,
    )
    .unwrap();
    let cfg = StreamConfig::parse_bitmapinfoheader(&strf).unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

#[test]
fn auto_roundtrips_interlaced_yuy2_320p() {
    let pixels = synth_yuy2(16, 300);
    let (strf, frame, _chosen) = encode_frame_auto(
        PixelFamily::Yuy2,
        MethodSelection::Auto,
        16,
        300,
        &pixels,
        ExtradataMode::CustomV2,
    )
    .unwrap();
    let cfg = StreamConfig::parse_bitmapinfoheader(&strf).unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

#[test]
fn auto_roundtrips_interlaced_rgb24_300p() {
    let pixels = synth_rgb24(8, 290);
    let (strf, frame, _chosen) = encode_frame_auto(
        PixelFamily::Rgb24,
        MethodSelection::Auto,
        8,
        290,
        &pixels,
        ExtradataMode::CustomV2,
    )
    .unwrap();
    let cfg = StreamConfig::parse_bitmapinfoheader(&strf).unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

#[test]
fn auto_winner_is_at_most_any_fixed_candidate_yuy2() {
    let pixels = synth_yuy2(32, 16);
    let (_strf, _frame, chosen) = encode_frame_auto(
        PixelFamily::Yuy2,
        MethodSelection::Auto,
        32,
        16,
        &pixels,
        ExtradataMode::CustomV2,
    )
    .unwrap();
    let chosen_cost = bit_cost_for_method(PixelFamily::Yuy2, chosen, 32, 16, &pixels).unwrap();
    for &m in &[Method::Left, Method::Gradient, Method::Median] {
        let c = bit_cost_for_method(PixelFamily::Yuy2, m, 32, 16, &pixels).unwrap();
        assert!(
            chosen_cost <= c,
            "auto winner {:?} cost {} > {:?} cost {}",
            chosen,
            chosen_cost,
            m,
            c
        );
    }
}

#[test]
fn auto_winner_is_at_most_any_fixed_candidate_rgb24() {
    let pixels = synth_rgb24(16, 16);
    let (_strf, _frame, chosen) = encode_frame_auto(
        PixelFamily::Rgb24,
        MethodSelection::Auto,
        16,
        16,
        &pixels,
        ExtradataMode::CustomV2,
    )
    .unwrap();
    let chosen_cost = bit_cost_for_method(PixelFamily::Rgb24, chosen, 16, 16, &pixels).unwrap();
    for &m in &[Method::Left, Method::LeftDecorr, Method::GradientDecorr] {
        let c = bit_cost_for_method(PixelFamily::Rgb24, m, 16, 16, &pixels).unwrap();
        assert!(
            chosen_cost <= c,
            "auto winner {:?} cost {} > {:?} cost {}",
            chosen,
            chosen_cost,
            m,
            c
        );
    }
}

#[test]
fn fixed_selection_returns_back_unchanged() {
    let pixels = synth_yuy2(8, 8);
    let (_strf, _frame, chosen) = encode_frame_auto(
        PixelFamily::Yuy2,
        MethodSelection::Fixed(Method::Median),
        8,
        8,
        &pixels,
        ExtradataMode::CustomV2,
    )
    .unwrap();
    assert_eq!(chosen, Method::Median);
}

#[test]
fn bit_cost_rejects_illegal_method_for_family() {
    // Median is YUV-only; must error on RGB24.
    let pixels = synth_rgb24(4, 4);
    let err = bit_cost_for_method(PixelFamily::Rgb24, Method::Median, 4, 4, &pixels);
    assert!(err.is_err());
}

// ───────── round-7: encoder auto + residual reuse, V1xCompat caching ─────────
//
// Round 7 collapses the auto-selector's residual computation from
// N + 1 (one per candidate inside `bit_cost_for_method` + one more
// inside `encode_frame_with_mode` for the winner) down to N. The
// pre-computed residuals carry forward into a private
// `encode_with_precomputed` so the winner's body bytes aren't
// re-derived from `pixels`. V1xCompat tables are now cached behind a
// per-family `OnceLock` instead of being rebuilt (LUT-bake included)
// on every encode call. The tests in this section confirm:
//
//   (a) wire identity between `encode_frame_auto(Fixed(m))` and
//       `encode_frame_with_mode(m)` for every legal (family, method)
//       pair under each extradata mode (no drift from sharing residuals);
//   (b) the chosen method's auto-emitted bytes match the bytes you'd
//       get by calling `encode_frame_with_mode(chosen)` directly;
//   (c) V1xCompat repeated-call wire stability (cache returns clones
//       that produce identical bit-streams);
//   (d) interlaced auto-selection wire-matches the explicit-method
//       interlaced path.

#[test]
fn round7_auto_fixed_wire_matches_encode_frame_with_mode_yuy2() {
    let pixels = synth_yuy2(16, 12);
    for &m in &[Method::Left, Method::Gradient, Method::Median] {
        for &mode in &[
            ExtradataMode::ClassicV2,
            ExtradataMode::CustomV2,
            ExtradataMode::V1xCompat,
        ] {
            let (strf_a, frame_a) =
                encode_frame_with_mode(PixelFamily::Yuy2, m, 16, 12, &pixels, mode).unwrap();
            let (strf_b, frame_b, chosen) = encode_frame_auto(
                PixelFamily::Yuy2,
                MethodSelection::Fixed(m),
                16,
                12,
                &pixels,
                mode,
            )
            .unwrap();
            assert_eq!(chosen, m, "Fixed(m) must return m back");
            assert_eq!(strf_a, strf_b, "{m:?} {mode:?} strf drift");
            assert_eq!(frame_a, frame_b, "{m:?} {mode:?} frame drift");
        }
    }
}

#[test]
fn round7_auto_fixed_wire_matches_encode_frame_with_mode_rgb24() {
    let pixels = synth_rgb24(12, 12);
    for &m in &[Method::Left, Method::LeftDecorr, Method::GradientDecorr] {
        for &mode in &[ExtradataMode::ClassicV2, ExtradataMode::CustomV2] {
            let (strf_a, frame_a) =
                encode_frame_with_mode(PixelFamily::Rgb24, m, 12, 12, &pixels, mode).unwrap();
            let (strf_b, frame_b, _) = encode_frame_auto(
                PixelFamily::Rgb24,
                MethodSelection::Fixed(m),
                12,
                12,
                &pixels,
                mode,
            )
            .unwrap();
            assert_eq!(strf_a, strf_b, "{m:?} {mode:?} strf drift");
            assert_eq!(frame_a, frame_b, "{m:?} {mode:?} frame drift");
        }
    }
}

#[test]
fn round7_auto_fixed_wire_matches_encode_frame_with_mode_rgb32() {
    let pixels = synth_rgb32(12, 12);
    for &m in &[Method::Left, Method::LeftDecorr, Method::GradientDecorr] {
        for &mode in &[ExtradataMode::ClassicV2, ExtradataMode::CustomV2] {
            let (strf_a, frame_a) =
                encode_frame_with_mode(PixelFamily::Rgb32, m, 12, 12, &pixels, mode).unwrap();
            let (strf_b, frame_b, _) = encode_frame_auto(
                PixelFamily::Rgb32,
                MethodSelection::Fixed(m),
                12,
                12,
                &pixels,
                mode,
            )
            .unwrap();
            assert_eq!(strf_a, strf_b, "{m:?} {mode:?} strf drift");
            assert_eq!(frame_a, frame_b, "{m:?} {mode:?} frame drift");
        }
    }
}

#[test]
fn round7_auto_winner_matches_explicit_winner_encode_yuy2() {
    // Auto picks a winner; encoding that same winner via
    // `encode_frame_with_mode` must produce identical bytes (auto's
    // shared-residual path must not drift from the explicit path).
    let pixels = synth_yuy2(24, 16);
    for &mode in &[ExtradataMode::CustomV2, ExtradataMode::ClassicV2] {
        let (strf_auto, frame_auto, chosen) = encode_frame_auto(
            PixelFamily::Yuy2,
            MethodSelection::Auto,
            24,
            16,
            &pixels,
            mode,
        )
        .unwrap();
        let (strf_exp, frame_exp) =
            encode_frame_with_mode(PixelFamily::Yuy2, chosen, 24, 16, &pixels, mode).unwrap();
        assert_eq!(
            strf_auto, strf_exp,
            "auto-vs-explicit strf drift, {chosen:?} {mode:?}"
        );
        assert_eq!(
            frame_auto, frame_exp,
            "auto-vs-explicit frame drift, {chosen:?} {mode:?}"
        );
    }
}

#[test]
fn round7_auto_winner_matches_explicit_winner_encode_interlaced() {
    let pixels = synth_yuy2(16, 300);
    let (strf_auto, frame_auto, chosen) = encode_frame_auto(
        PixelFamily::Yuy2,
        MethodSelection::Auto,
        16,
        300,
        &pixels,
        ExtradataMode::CustomV2,
    )
    .unwrap();
    let (strf_exp, frame_exp) = encode_frame_with_mode(
        PixelFamily::Yuy2,
        chosen,
        16,
        300,
        &pixels,
        ExtradataMode::CustomV2,
    )
    .unwrap();
    assert_eq!(
        strf_auto, strf_exp,
        "interlaced auto-vs-explicit strf drift"
    );
    assert_eq!(
        frame_auto, frame_exp,
        "interlaced auto-vs-explicit frame drift"
    );
}

#[test]
fn round7_v1x_cache_returns_stable_wire_bytes() {
    // V1xCompat tables are now cached behind a per-family OnceLock.
    // Two back-to-back encodes on the same input must produce
    // bit-identical bytes (regression guard against accidental
    // cache-mutation: the cached tables are cloned out per call, and
    // the helper must NOT hand out aliased mutable state).
    let pixels = synth_yuy2(8, 8);
    let (strf1, frame1) = encode_frame_with_mode(
        PixelFamily::Yuy2,
        Method::PredictOld,
        8,
        8,
        &pixels,
        ExtradataMode::V1xCompat,
    )
    .unwrap();
    let (strf2, frame2) = encode_frame_with_mode(
        PixelFamily::Yuy2,
        Method::PredictOld,
        8,
        8,
        &pixels,
        ExtradataMode::V1xCompat,
    )
    .unwrap();
    assert_eq!(strf1, strf2);
    assert_eq!(frame1, frame2);
    // RGB family hits a different cache slot — confirm RGB-then-YUV
    // doesn't poison YUV's cached slot1/slot2/slot3 triple (each
    // family must end up with its correct (A, B, B) vs (A, A, A)
    // tuple).
    let rgb_pixels = synth_rgb24(4, 4);
    let (_strf_rgb, frame_rgb1) = encode_frame_with_mode(
        PixelFamily::Rgb24,
        Method::PredictOld,
        4,
        4,
        &rgb_pixels,
        ExtradataMode::V1xCompat,
    )
    .unwrap();
    let (_strf_rgb, frame_rgb2) = encode_frame_with_mode(
        PixelFamily::Rgb24,
        Method::PredictOld,
        4,
        4,
        &rgb_pixels,
        ExtradataMode::V1xCompat,
    )
    .unwrap();
    assert_eq!(frame_rgb1, frame_rgb2);
    // And re-encode YUY2 a third time after touching RGB — still
    // stable.
    let (_strf3, frame3) = encode_frame_with_mode(
        PixelFamily::Yuy2,
        Method::PredictOld,
        8,
        8,
        &pixels,
        ExtradataMode::V1xCompat,
    )
    .unwrap();
    assert_eq!(frame1, frame3);
}

// ───────── round-91: flat overflow-entries slow path + SWAR gradient ─────────
//
// Round 91 lands two pure-performance improvements that must be
// bit-identical to round 7:
//
//   (a) `tables::HuffTable::overflow_entries` — a flat slice of
//       only the codes with length > 16, with `mask` precomputed
//       at build time. `decode_one_slow` walks this dense list
//       (typically ≤ 6 entries for the six classic v2.x blobs;
//       up to ~210 for v1.x set B) instead of the round-7
//       256-entry scan over `table.entries` that paid a
//       `length == 0 || length <= 16 { continue }` cost on
//       every iteration.
//   (b) `predict::inverse_gradient_post` — SWAR per-byte modular
//       add walking 8 bytes per u64 step. Mirrors spec/03 §2.2.2's
//       documented MMX 8-byte-wide post-pass.
//
// Earlier round-91 work also explored a span-replicated
// secondary index keyed on the 16-bit window prefix; benching
// showed that approach was a net regression vs round-7 on v1.x
// set B (non-canonical codes cluster heavily on prefix 0,
// turning the per-bucket scan into a Vec-indirection-laden
// near-256-entry walk). The flat overflow_entries Vec is the
// minimal change that's strictly better on every input.
//
// The tests below verify both paths against the round-7 baseline
// behaviour: every long code in v1.x set B must decode through
// the new slow path identically, and every (family, method)
// gradient encode must round-trip pixel-exact through the SWAR
// post-pass.

#[test]
fn round91_overflow_entries_decode_matches_round7_slow_path_v1x_set_b() {
    use crate::tables::{
        rle_decode_one_channel, v1x_codes_set_b, v1x_lengths_set_b, v1x_table_from_pair,
    };
    let mut cur: &[u8] = v1x_lengths_set_b();
    let lens_b = rle_decode_one_channel(&mut cur).unwrap();
    let codes_b = v1x_codes_set_b();
    let mut codes_arr = [0u8; 256];
    codes_arr.copy_from_slice(codes_b);
    let table = v1x_table_from_pair(&lens_b, &codes_arr).unwrap();
    // For every long code, the round-91 flat-overflow slow path
    // (`decode_one_slow`) and the high-level `decode_one` (which
    // dispatches via the primary LUT to the same slow path) must
    // agree on (symbol, length).
    let mut covered = 0;
    for (sym, e) in table.entries.iter().enumerate() {
        if e.length <= 16 || e.length == 0 {
            continue;
        }
        let window = e.code;
        let via_decode_one = crate::tables::decode_one(&table, window).unwrap();
        let via_slow = crate::tables::decode_one_slow(&table, window).unwrap();
        assert_eq!(
            via_decode_one, via_slow,
            "sym {sym} decode_one {via_decode_one:?} ≠ slow {via_slow:?}"
        );
        assert_eq!(via_decode_one.0 as usize, sym);
        assert_eq!(via_decode_one.1, e.length);
        covered += 1;
    }
    // v1.x set B is documented as having codes up to length 26
    // (spec/03 §4.1 evidence row); at least a handful of overflow
    // codes must have been covered for this test to be meaningful.
    assert!(
        covered >= 4,
        "expected ≥ 4 overflow codes in v1.x set B; got {covered}"
    );
}

#[test]
fn round91_overflow_entries_match_count_of_long_codes() {
    // Sanity: `build_overflow_entries` emits exactly one entry per
    // code with length > 16. Round-91 dropped the span-replicated
    // secondary-index design (a net regression vs round-7 at v1.x
    // set B's ~210 long codes — the indirection cost beat the
    // narrowed scan count). The flat overflow_entries Vec is now
    // a pure short-circuit-elimination win: the slow loop walks
    // ≤ 210 entries instead of 256, with `mask` precomputed once
    // at build time.
    use crate::tables::{
        rle_decode_one_channel, v1x_codes_set_b, v1x_lengths_set_b, v1x_table_from_pair,
    };
    let mut cur: &[u8] = v1x_lengths_set_b();
    let lens_b = rle_decode_one_channel(&mut cur).unwrap();
    let codes_b = v1x_codes_set_b();
    let mut codes_arr = [0u8; 256];
    codes_arr.copy_from_slice(codes_b);
    let table = v1x_table_from_pair(&lens_b, &codes_arr).unwrap();
    let table_len = table.overflow_entries.len();
    let expected: usize = table.entries.iter().filter(|e| e.length > 16).count();
    assert_eq!(
        table_len, expected,
        "overflow_entries.len() {table_len} ≠ long-code count {expected}"
    );
}

#[test]
fn round91_overflow_entries_bounded_below_256() {
    // The slow path now walks at most `overflow_entries.len()`
    // entries — strictly less than the round-7 256-entry scan
    // over `table.entries`. This invariant is what makes the
    // round-91 slow path a pure speedup.
    use crate::tables::{
        rle_decode_one_channel, v1x_codes_set_b, v1x_lengths_set_b, v1x_table_from_pair,
    };
    let mut cur: &[u8] = v1x_lengths_set_b();
    let lens_b = rle_decode_one_channel(&mut cur).unwrap();
    let codes_b = v1x_codes_set_b();
    let mut codes_arr = [0u8; 256];
    codes_arr.copy_from_slice(codes_b);
    let table = v1x_table_from_pair(&lens_b, &codes_arr).unwrap();
    assert!(
        table.overflow_entries.len() < 256,
        "overflow_entries.len() {} ≥ 256 (would not beat round-7 baseline)",
        table.overflow_entries.len()
    );
    assert!(
        !table.overflow_entries.is_empty(),
        "v1.x set B should have at least one long-code overflow entry"
    );
}

#[test]
fn round91_classic_v2_overflow_entries_only_long_codes() {
    // Round-91 invariant: every entry in `overflow_entries` has
    // `length > 16`. (No leakage from the ≤ 16-bit codes that the
    // primary LUT serves directly.) The classic-v2 YUV-LEFT
    // blob's slot tables top out at length 17 (spec/04 §3
    // Extractor evidence), so this should hold for all three
    // slots without any short-code contamination.
    use crate::tables::{rle_decode_three_channels, HuffTable};
    let blob = crate::tables::classic_blob_bytes(PixelFamily::Yuy2, Method::Left);
    let lengths = rle_decode_three_channels(blob).unwrap();
    for slot in lengths.iter() {
        let table = HuffTable::build_from_lengths(slot).unwrap();
        for e in table.overflow_entries.iter() {
            assert!(
                e.length > 16,
                "overflow_entries contains length-{} entry (≤ 16 = primary LUT)",
                e.length
            );
        }
        let expected: usize = table.entries.iter().filter(|e| e.length > 16).count();
        assert_eq!(table.overflow_entries.len(), expected);
    }
}

#[test]
fn round91_v1x_yuy2_median_8x4_uses_overflow_path() {
    // Force overflow traffic: YUY2 + Median + V1xCompat uses set B
    // for U/V slots, max length 26 — the per-byte decode will hit
    // overflow rows at every U/V byte whose code ≥ 17 bits. End-to-end
    // round-trip must still match the source pixels exactly.
    let pixels = synth_yuy2(8, 4);
    let (cfg, frame) = encode_for_test_with_mode(
        PixelFamily::Yuy2,
        Method::Median,
        8,
        4,
        &pixels,
        ExtradataMode::V1xCompat,
    )
    .unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

/// Reference (round-7) gradient post-pass: byte-by-byte add. Used by
/// the SWAR equivalence test below to verify the chunked u64 path
/// produces identical bytes.
fn reference_inverse_gradient_post(out: &mut [u8], row_bytes: usize, height: usize) {
    if height < 2 || row_bytes == 0 {
        return;
    }
    for row in 1..height {
        let above = (row - 1) * row_bytes;
        let curr = row * row_bytes;
        for col in 0..row_bytes {
            out[curr + col] = out[curr + col].wrapping_add(out[above + col]);
        }
    }
}

#[test]
fn round91_swar_gradient_matches_byte_loop_aligned_8() {
    // row_bytes = 16 (multiple of 8), height = 4, deterministic data.
    let mut a: Vec<u8> = (0u32..64).map(|i| ((i * 37) & 0xFF) as u8).collect();
    let mut b = a.clone();
    crate::predict::inverse_gradient_post(&mut a, 16, 4);
    reference_inverse_gradient_post(&mut b, 16, 4);
    assert_eq!(a, b);
}

#[test]
fn round91_swar_gradient_matches_byte_loop_unaligned_tail() {
    // row_bytes = 11 (NOT a multiple of 8) — the SWAR path takes one
    // u64-step (cols 0..8) and three byte-steps (8..11) per row.
    let mut a: Vec<u8> = (0u32..55).map(|i| ((i * 53) & 0xFF) as u8).collect();
    let mut b = a.clone();
    crate::predict::inverse_gradient_post(&mut a, 11, 5);
    reference_inverse_gradient_post(&mut b, 11, 5);
    assert_eq!(a, b);
}

#[test]
fn round91_swar_gradient_matches_byte_loop_height_1_noop() {
    // height < 2: no add pass happens.
    let mut a: Vec<u8> = vec![42; 16];
    let mut b = a.clone();
    crate::predict::inverse_gradient_post(&mut a, 16, 1);
    reference_inverse_gradient_post(&mut b, 16, 1);
    assert_eq!(a, b);
    // Also row_bytes = 0 trips the early return.
    let mut a2: Vec<u8> = vec![];
    let mut b2 = a2.clone();
    crate::predict::inverse_gradient_post(&mut a2, 0, 4);
    reference_inverse_gradient_post(&mut b2, 0, 4);
    assert_eq!(a2, b2);
}

#[test]
fn round91_swar_gradient_handles_modular_wrap() {
    // 8 bytes whose pairwise additions wrap past 255 — exercises the
    // mod-256 SWAR identity end-to-end. Each pair (above, curr)
    // chosen so curr + above ≥ 0x100 to force a per-byte wrap.
    let above = [0xC0u8, 0xC1, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7];
    let curr = [0x50u8, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57];
    let mut a: Vec<u8> = Vec::new();
    a.extend_from_slice(&above);
    a.extend_from_slice(&curr);
    let mut b = a.clone();
    crate::predict::inverse_gradient_post(&mut a, 8, 2);
    reference_inverse_gradient_post(&mut b, 8, 2);
    assert_eq!(a, b, "SWAR add must mod-256 wrap per-byte");
}

#[test]
fn round91_swar_gradient_end_to_end_yuy2_gradient_320x16() {
    // End-to-end: encode a 320×16 YUY2 frame with Gradient predictor,
    // decode, verify pixel-exact equality. The decode path's gradient
    // post-pass now uses the SWAR path; any byte-wise drift from the
    // round-7 byte loop would corrupt the output.
    let pixels = synth_yuy2(320, 16);
    let (cfg, frame) =
        encode_for_test(PixelFamily::Yuy2, Method::Gradient, 320, 16, &pixels).unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

#[test]
fn round91_swar_gradient_end_to_end_rgb24_gradient_decorr_320x16() {
    // End-to-end with RGB24 GradientDecorr — the gradient post-pass
    // runs on 24-bit-per-pixel rows (row_bytes = 960) so the SWAR
    // chunking covers cleanly. Decorrelation inverse runs after.
    let pixels = synth_rgb24(320, 16);
    let (cfg, frame) =
        encode_for_test(PixelFamily::Rgb24, Method::GradientDecorr, 320, 16, &pixels).unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

#[test]
fn round91_swar_gradient_end_to_end_rgb32_gradient_decorr_320x16() {
    let pixels = synth_rgb32(320, 16);
    let (cfg, frame) =
        encode_for_test(PixelFamily::Rgb32, Method::GradientDecorr, 320, 16, &pixels).unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

// ───────── Round-100: fused LEFT + decorrelation residual path ─────────
//
// spec/03 §2.4.1: RGB-with-decorrelation fuses the decorrelation
// transform into the LEFT residual at the per-byte computation, with
// no intermediate decorrelated buffer. The encoder's LeftDecorr path
// now reads `pixels` directly via `forward_left_decorr_residuals`,
// skipping the round-95 `working_owned` full-frame allocation. These
// tests guard the fused path's wire correctness across families,
// extradata modes, larger rasters, the interlaced trigger, and a
// strongly chroma-correlated synthetic where decorrelation is
// non-trivial.

/// Strongly chroma-correlated RGB raster: B and R track G closely, so
/// the decorrelated channels (B−G, R−G) collapse toward small values.
/// Any drift in the fused decorrelation arithmetic would corrupt the
/// reconstructed pixels.
fn synth_rgb24_chroma_correlated(width: usize, height: usize) -> Vec<u8> {
    let mut v = vec![0u8; width * 3 * height];
    for row in 0..height {
        for col in 0..width {
            let off = (row * width + col) * 3;
            let g = ((row * 9 + col * 17) & 0xFF) as u8;
            v[off] = g.wrapping_add(((col * 2) & 0x0F) as u8); // B ≈ G
            v[off + 1] = g; // G
            v[off + 2] = g.wrapping_sub(((row + 3) & 0x0F) as u8); // R ≈ G
        }
    }
    v
}

#[test]
fn round100_fused_left_decorr_rgb24_classic_v2_64x16() {
    let pixels = synth_rgb24(64, 16);
    let (cfg, frame) =
        encode_for_test(PixelFamily::Rgb24, Method::LeftDecorr, 64, 16, &pixels).unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

#[test]
fn round100_fused_left_decorr_rgb32_classic_v2_64x16() {
    let pixels = synth_rgb32(64, 16);
    let (cfg, frame) =
        encode_for_test(PixelFamily::Rgb32, Method::LeftDecorr, 64, 16, &pixels).unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

#[test]
fn round100_fused_left_decorr_rgb24_custom_v2_chroma_correlated() {
    let pixels = synth_rgb24_chroma_correlated(32, 24);
    let (cfg, frame) = encode_for_test_with_mode(
        PixelFamily::Rgb24,
        Method::LeftDecorr,
        32,
        24,
        &pixels,
        ExtradataMode::CustomV2,
    )
    .unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

#[test]
fn round100_fused_left_decorr_rgb24_v1x_compat() {
    let pixels = synth_rgb24(16, 8);
    let (cfg, frame) = encode_for_test_with_mode(
        PixelFamily::Rgb24,
        Method::LeftDecorr,
        16,
        8,
        &pixels,
        ExtradataMode::V1xCompat,
    )
    .unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
    assert_eq!(cfg.method, Method::LeftDecorr);
}

#[test]
fn round100_fused_left_decorr_rgb32_interlaced_8x300() {
    // Height 300 > 288 → interlaced field-stride=2 path. The fused
    // decorrelation must produce wire-correct per-field residuals.
    let pixels = synth_rgb32(8, 300);
    let (cfg, frame) =
        encode_for_test(PixelFamily::Rgb32, Method::LeftDecorr, 8, 300, &pixels).unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

#[test]
fn round100_fused_left_decorr_alpha_varies_per_row_rgb32() {
    // Alpha varies row-by-row (spec/03 §2.4 Validator fixture shape):
    // alpha is LEFT-predicted but NOT decorrelated. A drift that
    // decorrelated alpha (subtracting G) would corrupt reconstruction.
    let w = 16usize;
    let h = 16usize;
    let mut pixels = vec![0u8; w * 4 * h];
    for row in 0..h {
        for col in 0..w {
            let off = (row * w + col) * 4;
            pixels[off] = ((col * 3) & 0xFF) as u8; // B
            pixels[off + 1] = ((row * 5 + col) & 0xFF) as u8; // G
            pixels[off + 2] = ((row * 7) & 0xFF) as u8; // R
            pixels[off + 3] = ((row * 16) & 0xFF) as u8; // A = row*16
        }
    }
    let (cfg, frame) = encode_for_test(
        PixelFamily::Rgb32,
        Method::LeftDecorr,
        w as u32,
        h as u32,
        &pixels,
    )
    .unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}
