//! Self-roundtrip tests: synthesise minimal HuffYUV frames via
//! [`crate::encoder::encode_for_test`] and decode them back via
//! [`crate::decoder::decode_frame`].
//!
//! Round-2 covers all six predictor methods × three pixel families
//! (with the spec/01 §3.1 method/family allow-list applied), plus
//! the v1.x-compat (`biSize == 0x28`) and v2.x-custom (computed
//! length tables) extradata paths.

use crate::decoder::decode_frame;
use crate::encoder::{encode_for_test, encode_for_test_with_mode, ExtradataMode};
use crate::header::{Method, PixelFamily};
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
    assert_eq!(lens[7], 1);
    for (i, &l) in lens.iter().enumerate() {
        if i != 7 {
            assert_eq!(l, 0);
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
