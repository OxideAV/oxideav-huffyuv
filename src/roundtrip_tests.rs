//! Self-roundtrip tests: synthesise minimal HuffYUV frames via
//! [`crate::encoder::encode_for_test`] and decode them back via
//! [`crate::decoder::decode_frame`].
//!
//! Per the round-1 deliverable, no checked-in fixtures yet; we
//! exercise the four method/family combinations the spec gates on.

use crate::decoder::decode_frame;
use crate::encoder::encode_for_test;
use crate::header::{Method, PixelFamily};

fn synth_yuy2(width: usize, height: usize) -> Vec<u8> {
    let mut v = vec![0u8; width * 2 * height];
    for row in 0..height {
        for col in 0..(width * 2) {
            // Mildly varying so the residuals exercise a range of
            // symbols. Wrap-around is fine.
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
fn roundtrip_yuy2_predict_old_8x4() {
    // predict_old is the same wire algorithm as left; verify the
    // method-byte path through.
    let pixels = synth_yuy2(8, 4);
    let (cfg, frame) =
        encode_for_test(PixelFamily::Yuy2, Method::PredictOld, 8, 4, &pixels).unwrap();
    let out = decode_frame(&cfg, &frame).unwrap();
    assert_eq!(out.pixels, pixels);
}

#[test]
fn roundtrip_yuy2_median_8x6() {
    // Median requires height ≥ 2 (the post-pass kicks in at row 1
    // byte 8). Use 8×6 so the median exemption + post-pass both run.
    let pixels = synth_yuy2(8, 6);
    let (cfg, frame) = encode_for_test(PixelFamily::Yuy2, Method::Median, 8, 6, &pixels).unwrap();
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
fn roundtrip_rgb32_left_4x4() {
    let pixels = synth_rgb32(4, 4);
    let (cfg, frame) = encode_for_test(PixelFamily::Rgb32, Method::Left, 4, 4, &pixels).unwrap();
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
