//! Criterion benchmarks for the HuffYUV decoder hot paths.
//!
//! All inputs are synthesised on the fly (no committed binary fixtures)
//! by [`encode_frame_with_mode`] against a deterministic xorshift32
//! pattern, so the benches stay self-contained and run without `docs/`
//! or third-party samples. Each scenario covers a representative
//! `(family, method, extradata-mode, size)` quadruple so individual
//! optimisations can be diffed against the matching baseline.
//!
//! Coverage mirrors the README's "measured" headline scenarios so a
//! future round can read a regression directly off the criterion delta:
//!
//!   - **yuy2_320x240_left_classic**: the ClassicV2 LEFT-only YUY2
//!     baseline (round 3's headline 5.3 ms/frame ClassicV2 LEFT
//!     fast-LUT).
//!   - **yuy2_320x240_gradient_classic**: ClassicV2 Gradient YUY2 —
//!     exercises the round-91 flat-`overflow_entries` slow path + the
//!     round-91 SWAR `inverse_gradient_post`.
//!   - **yuy2_320x240_median_classic**: ClassicV2 Median YUY2 —
//!     exercises `inverse_median_post` (the round-91 flat-overflow
//!     slow path applies here too: v1.x median was the headline
//!     2.2× round-91 speedup).
//!   - **yuy2_1280x720_left_classic**: 720p ClassicV2 LEFT YUY2 — a
//!     larger raster than the README headline so future SWAR /
//!     fast-LUT work can show a wall-time delta visible at HD.
//!   - **yuy2_320x240_left_v1x**: V1xCompat LEFT YUY2 — exercises the
//!     v1.x precomputed-codes path with the round-7 OnceLock
//!     per-family table cache (the README's 0.40 ms/frame V1xCompat
//!     headline).
//!   - **rgb24_320x240_left_decorr_classic**: RGB24 LeftDecorr —
//!     exercises `inverse_rgb_decorr_bgr` (round-100's fused-decorr
//!     residual path is encoder-side; the decoder still runs the
//!     inverse pass per pixel).
//!   - **rgb32_320x240_gradient_decorr_classic**: RGB32 GradientDecorr
//!     — exercises `inverse_rgb_decorr_bgra` + the gradient post-pass
//!     end-to-end (alpha-NOT-decorrelated per spec/03 §2.4).
//!
//! Run with:
//!     cargo bench -p oxideav-huffyuv --bench decode

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use oxideav_huffyuv::{
    decode_frame, encode_frame_with_mode, ExtradataMode, Method, PixelFamily, StreamConfig,
};

/// Cheap deterministic xorshift32 — synthesises "natural-ish" per-pixel
/// values for the bench inputs so residuals are not all zero (which
/// would let the Huffman path short-circuit on a tiny histogram).
fn xorshift_byte(state: &mut u32) -> u8 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    (*state & 0xff) as u8
}

/// Build a `width × height` buffer of `bytes_per_pixel`-wide pixels
/// holding a smooth diagonal gradient + small per-channel noise. The
/// gradient keeps residuals small (= high Huffman compression) and the
/// noise prevents the single-symbol degenerate histogram path from
/// being hit by accident.
fn build_pixels(width: usize, height: usize, bytes_per_pixel: usize) -> Vec<u8> {
    let row_bytes = width * bytes_per_pixel;
    let mut out = vec![0u8; row_bytes * height];
    let mut state: u32 = 0xdead_beef;
    for r in 0..height {
        for c in 0..width {
            let base = ((r as u32).wrapping_add(c as u32) >> 1) & 0xff;
            let off = r * row_bytes + c * bytes_per_pixel;
            for ch in 0..bytes_per_pixel {
                let noise = xorshift_byte(&mut state) as u32 & 0x07;
                let chan_bias = (ch as u32).wrapping_mul(7);
                out[off + ch] = (base.wrapping_add(noise).wrapping_add(chan_bias) & 0xff) as u8;
            }
        }
    }
    out
}

fn bytes_per_pixel(family: PixelFamily) -> usize {
    match family {
        PixelFamily::Yuy2 => 2,
        PixelFamily::Rgb24 => 3,
        PixelFamily::Rgb32 => 4,
    }
}

/// Encode one synthetic frame, parse the BIH back, and return a
/// `(StreamConfig, frame_bytes)` pair the decode loop can iterate on.
fn make_decode_input(
    family: PixelFamily,
    method: Method,
    width: u32,
    height: u32,
    mode: ExtradataMode,
) -> (StreamConfig, Vec<u8>) {
    let bpp = bytes_per_pixel(family);
    let pixels = build_pixels(width as usize, height as usize, bpp);
    let (strf, frame) = encode_frame_with_mode(family, method, width, height, &pixels, mode)
        .expect("encode for decode bench");
    let cfg = StreamConfig::parse_bitmapinfoheader(&strf).expect("parse BIH for decode bench");
    (cfg, frame)
}

fn bench_scenario(
    c: &mut Criterion,
    name: &str,
    family: PixelFamily,
    method: Method,
    width: u32,
    height: u32,
    mode: ExtradataMode,
) {
    let (cfg, frame) = make_decode_input(family, method, width, height, mode);
    let mut g = c.benchmark_group(name);
    g.throughput(Throughput::Bytes(
        (width as u64) * (height as u64) * bytes_per_pixel(family) as u64,
    ));
    g.bench_function(
        BenchmarkId::from_parameter(format!("{width}x{height}")),
        |b| {
            b.iter(|| {
                decode_frame(criterion::black_box(&cfg), criterion::black_box(&frame))
                    .expect("decode")
            });
        },
    );
    g.finish();
}

fn bench_yuy2_320x240_left_classic(c: &mut Criterion) {
    bench_scenario(
        c,
        "decode_yuy2_320x240_left_classic",
        PixelFamily::Yuy2,
        Method::Left,
        320,
        240,
        ExtradataMode::ClassicV2,
    );
}

fn bench_yuy2_320x240_gradient_classic(c: &mut Criterion) {
    bench_scenario(
        c,
        "decode_yuy2_320x240_gradient_classic",
        PixelFamily::Yuy2,
        Method::Gradient,
        320,
        240,
        ExtradataMode::ClassicV2,
    );
}

fn bench_yuy2_320x240_median_classic(c: &mut Criterion) {
    bench_scenario(
        c,
        "decode_yuy2_320x240_median_classic",
        PixelFamily::Yuy2,
        Method::Median,
        320,
        240,
        ExtradataMode::ClassicV2,
    );
}

fn bench_yuy2_1280x720_left_classic(c: &mut Criterion) {
    bench_scenario(
        c,
        "decode_yuy2_1280x720_left_classic",
        PixelFamily::Yuy2,
        Method::Left,
        1280,
        720,
        ExtradataMode::ClassicV2,
    );
}

fn bench_yuy2_320x240_left_v1x(c: &mut Criterion) {
    bench_scenario(
        c,
        "decode_yuy2_320x240_left_v1x",
        PixelFamily::Yuy2,
        Method::Left,
        320,
        240,
        ExtradataMode::V1xCompat,
    );
}

fn bench_rgb24_320x240_left_decorr_classic(c: &mut Criterion) {
    bench_scenario(
        c,
        "decode_rgb24_320x240_left_decorr_classic",
        PixelFamily::Rgb24,
        Method::LeftDecorr,
        320,
        240,
        ExtradataMode::ClassicV2,
    );
}

fn bench_rgb32_320x240_gradient_decorr_classic(c: &mut Criterion) {
    bench_scenario(
        c,
        "decode_rgb32_320x240_gradient_decorr_classic",
        PixelFamily::Rgb32,
        Method::GradientDecorr,
        320,
        240,
        ExtradataMode::ClassicV2,
    );
}

criterion_group!(
    benches,
    bench_yuy2_320x240_left_classic,
    bench_yuy2_320x240_gradient_classic,
    bench_yuy2_320x240_median_classic,
    bench_yuy2_1280x720_left_classic,
    bench_yuy2_320x240_left_v1x,
    bench_rgb24_320x240_left_decorr_classic,
    bench_rgb32_320x240_gradient_decorr_classic,
);
criterion_main!(benches);
