//! Criterion benchmarks for the HuffYUV encoder hot paths.
//!
//! Mirrors the `decode` bench coverage so an encoder-side optimisation
//! (e.g. the round-95 SWAR forward gradient, the round-100/103 fused
//! decorrelation residual paths, the round-115 single-pass forward
//! median) can be diffed against the matching baseline. Each iteration
//! re-runs `encode_frame_with_mode` from the same pre-built
//! `&pixels` slice (the encoder doesn't mutate its input), so the
//! bench is steady-state allocation-bound rather than memcpy-bound.
//!
//! The symmetric-encode matrix spans all three pixel families
//! (YUY2 / RGB24 / RGB32) across both extradata generations:
//!
//!   - **v2.x** (`ClassicV2`): per-stream classic Huffman length
//!     tables; the default on-wire shape.
//!   - **v1.x** (`V1xCompat`): the precomputed-codebook fallback
//!     (`biSize == 0x28`), which writes the residual stream against
//!     the v1.x code templates instead of building a per-frame table.
//!
//! The RGB32 rows close the gap the decode/roundtrip benches already
//! covered — before this round the encode bench had no RGB32 scenario
//! at all, so an RGB32-specific encoder optimisation had no baseline.
//!
//! Run with:
//!     cargo bench -p oxideav-huffyuv --bench encode

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use oxideav_huffyuv::{
    encode_frame_auto, encode_frame_with_mode, encode_frame_with_mode_workers, ExtradataMode,
    Method, MethodSelection, PixelFamily,
};

fn xorshift_byte(state: &mut u32) -> u8 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    (*state & 0xff) as u8
}

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

fn bench_fixed(
    c: &mut Criterion,
    name: &str,
    family: PixelFamily,
    method: Method,
    width: u32,
    height: u32,
    mode: ExtradataMode,
) {
    let pixels = build_pixels(width as usize, height as usize, bytes_per_pixel(family));
    let mut g = c.benchmark_group(name);
    g.throughput(Throughput::Bytes(pixels.len() as u64));
    g.bench_function(
        BenchmarkId::from_parameter(format!("{width}x{height}")),
        |b| {
            b.iter(|| {
                let (_strf, _frame) = encode_frame_with_mode(
                    family,
                    method,
                    width,
                    height,
                    criterion::black_box(&pixels),
                    mode,
                )
                .expect("encode");
            });
        },
    );
    g.finish();
}

fn bench_auto(
    c: &mut Criterion,
    name: &str,
    family: PixelFamily,
    width: u32,
    height: u32,
    mode: ExtradataMode,
) {
    let pixels = build_pixels(width as usize, height as usize, bytes_per_pixel(family));
    let mut g = c.benchmark_group(name);
    g.throughput(Throughput::Bytes(pixels.len() as u64));
    g.bench_function(
        BenchmarkId::from_parameter(format!("{width}x{height}/auto")),
        |b| {
            b.iter(|| {
                let (_strf, _frame, _picked) = encode_frame_auto(
                    family,
                    MethodSelection::Auto,
                    width,
                    height,
                    criterion::black_box(&pixels),
                    mode,
                )
                .expect("encode auto");
            });
        },
    );
    g.finish();
}

fn bench_yuy2_320x240_left_classic(c: &mut Criterion) {
    bench_fixed(
        c,
        "encode_yuy2_320x240_left_classic",
        PixelFamily::Yuy2,
        Method::Left,
        320,
        240,
        ExtradataMode::ClassicV2,
    );
}

fn bench_yuy2_320x240_median_classic(c: &mut Criterion) {
    bench_fixed(
        c,
        "encode_yuy2_320x240_median_classic",
        PixelFamily::Yuy2,
        Method::Median,
        320,
        240,
        ExtradataMode::ClassicV2,
    );
}

fn bench_yuy2_320x240_gradient_classic(c: &mut Criterion) {
    bench_fixed(
        c,
        "encode_yuy2_320x240_gradient_classic",
        PixelFamily::Yuy2,
        Method::Gradient,
        320,
        240,
        ExtradataMode::ClassicV2,
    );
}

fn bench_yuy2_1280x720_left_classic(c: &mut Criterion) {
    bench_fixed(
        c,
        "encode_yuy2_1280x720_left_classic",
        PixelFamily::Yuy2,
        Method::Left,
        1280,
        720,
        ExtradataMode::ClassicV2,
    );
}

fn bench_yuy2_320x240_left_v1x(c: &mut Criterion) {
    bench_fixed(
        c,
        "encode_yuy2_320x240_left_v1x",
        PixelFamily::Yuy2,
        Method::Left,
        320,
        240,
        ExtradataMode::V1xCompat,
    );
}

fn bench_rgb24_320x240_left_classic(c: &mut Criterion) {
    bench_fixed(
        c,
        "encode_rgb24_320x240_left_classic",
        PixelFamily::Rgb24,
        Method::Left,
        320,
        240,
        ExtradataMode::ClassicV2,
    );
}

fn bench_rgb24_320x240_left_decorr_classic(c: &mut Criterion) {
    bench_fixed(
        c,
        "encode_rgb24_320x240_left_decorr_classic",
        PixelFamily::Rgb24,
        Method::LeftDecorr,
        320,
        240,
        ExtradataMode::ClassicV2,
    );
}

fn bench_rgb24_320x240_gradient_decorr_classic(c: &mut Criterion) {
    bench_fixed(
        c,
        "encode_rgb24_320x240_gradient_decorr_classic",
        PixelFamily::Rgb24,
        Method::GradientDecorr,
        320,
        240,
        ExtradataMode::ClassicV2,
    );
}

fn bench_rgb24_320x240_left_v1x(c: &mut Criterion) {
    bench_fixed(
        c,
        "encode_rgb24_320x240_left_v1x",
        PixelFamily::Rgb24,
        Method::Left,
        320,
        240,
        ExtradataMode::V1xCompat,
    );
}

fn bench_rgb32_320x240_left_classic(c: &mut Criterion) {
    bench_fixed(
        c,
        "encode_rgb32_320x240_left_classic",
        PixelFamily::Rgb32,
        Method::Left,
        320,
        240,
        ExtradataMode::ClassicV2,
    );
}

fn bench_rgb32_320x240_left_decorr_classic(c: &mut Criterion) {
    bench_fixed(
        c,
        "encode_rgb32_320x240_left_decorr_classic",
        PixelFamily::Rgb32,
        Method::LeftDecorr,
        320,
        240,
        ExtradataMode::ClassicV2,
    );
}

fn bench_rgb32_320x240_gradient_decorr_classic(c: &mut Criterion) {
    bench_fixed(
        c,
        "encode_rgb32_320x240_gradient_decorr_classic",
        PixelFamily::Rgb32,
        Method::GradientDecorr,
        320,
        240,
        ExtradataMode::ClassicV2,
    );
}

fn bench_rgb32_320x240_left_v1x(c: &mut Criterion) {
    bench_fixed(
        c,
        "encode_rgb32_320x240_left_v1x",
        PixelFamily::Rgb32,
        Method::Left,
        320,
        240,
        ExtradataMode::V1xCompat,
    );
}

fn bench_yuy2_320x320_left_interlaced(c: &mut Criterion) {
    // height 320 > 288 engages the encoder field-split path
    // (compact_field_rows for both fields + concatenated bodies).
    bench_fixed(
        c,
        "encode_yuy2_320x320_left_interlaced",
        PixelFamily::Yuy2,
        Method::Left,
        320,
        320,
        ExtradataMode::ClassicV2,
    );
}

fn bench_yuy2_320x240_auto_custom(c: &mut Criterion) {
    // Round-7 auto-selector residual-reuse headline scenario (CustomV2
    // exercises the package-merge length builder per candidate).
    bench_auto(
        c,
        "encode_yuy2_320x240_auto_custom",
        PixelFamily::Yuy2,
        320,
        240,
        ExtradataMode::CustomV2,
    );
}

fn bench_rgb24_320x240_auto_custom(c: &mut Criterion) {
    bench_auto(
        c,
        "encode_rgb24_320x240_auto_custom",
        PixelFamily::Rgb24,
        320,
        240,
        ExtradataMode::CustomV2,
    );
}

// ───────── Round-419 large-frame additions ─────────
//
// Before round 419 the only ≥ HD encode scenario was YUY2 Left — the
// RGB families and the auto-selector had no large-frame gate, so a
// regression that scaled with raster size (allocation churn, cache
// behaviour of the 64-Ki LUTs, package-merge per candidate) could
// hide behind the 320-class scenarios. 1280×720 crosses the
// `is_interlaced_height` threshold, so these also exercise the
// field-split encode path on the wider-stride families.

fn bench_rgb24_1280x720_left_decorr_classic(c: &mut Criterion) {
    // Large-raster RGB24 decorrelated encode (interlaced by height).
    bench_fixed(
        c,
        "encode_rgb24_1280x720_left_decorr_classic",
        PixelFamily::Rgb24,
        Method::LeftDecorr,
        1280,
        720,
        ExtradataMode::ClassicV2,
    );
}

fn bench_yuy2_1280x720_auto_custom(c: &mut Criterion) {
    // Large-raster auto-selector gate: 3 candidate methods × 3-channel
    // histograms + package-merge on a 1.8-MiB raster, then the winner's
    // CustomV2 emit. Ties the round-419 package-merge rewrite to an
    // HD-sized whole-frame number.
    bench_auto(
        c,
        "encode_yuy2_1280x720_auto_custom",
        PixelFamily::Yuy2,
        1280,
        720,
        ExtradataMode::CustomV2,
    );
}

/// Round-420 worker-scaling gates: 1 vs 2 workers on interlaced HD
/// encodes (`encode_frame_with_mode_workers`). Unlike decode, encode
/// has no split-finding prefix (each field packs into its own buffer
/// and the buffers concatenate), so both the residual phase and the
/// emit phase parallelise fully — these gates read as the ceiling of
/// the crate's ExecutionContext scaling.
fn bench_hd_interlaced_encode_worker_scaling(c: &mut Criterion) {
    for (name, family, method) in [
        (
            "encode_workers_yuy2_1280x720_left_classic",
            PixelFamily::Yuy2,
            Method::Left,
        ),
        (
            "encode_workers_yuy2_1280x720_median_classic",
            PixelFamily::Yuy2,
            Method::Median,
        ),
        (
            "encode_workers_rgb32_1280x720_gradient_decorr_classic",
            PixelFamily::Rgb32,
            Method::GradientDecorr,
        ),
    ] {
        let (w, h) = (1280u32, 720u32);
        let pixels = build_pixels(w as usize, h as usize, bytes_per_pixel(family));
        let mut g = c.benchmark_group(name);
        g.throughput(Throughput::Bytes(pixels.len() as u64));
        for workers in [1usize, 2] {
            g.bench_function(BenchmarkId::from_parameter(format!("{workers}w")), |b| {
                b.iter(|| {
                    encode_frame_with_mode_workers(
                        family,
                        method,
                        w,
                        h,
                        criterion::black_box(&pixels),
                        ExtradataMode::ClassicV2,
                        workers,
                    )
                    .expect("encode")
                });
            });
        }
        g.finish();
    }
}

criterion_group!(
    benches,
    bench_yuy2_320x240_left_classic,
    bench_yuy2_320x240_median_classic,
    bench_yuy2_320x240_gradient_classic,
    bench_yuy2_1280x720_left_classic,
    bench_yuy2_320x240_left_v1x,
    bench_rgb24_320x240_left_classic,
    bench_rgb24_320x240_left_decorr_classic,
    bench_rgb24_320x240_gradient_decorr_classic,
    bench_rgb24_320x240_left_v1x,
    bench_rgb32_320x240_left_classic,
    bench_rgb32_320x240_left_decorr_classic,
    bench_rgb32_320x240_gradient_decorr_classic,
    bench_rgb32_320x240_left_v1x,
    bench_yuy2_320x320_left_interlaced,
    bench_yuy2_320x240_auto_custom,
    bench_rgb24_320x240_auto_custom,
    bench_rgb24_1280x720_left_decorr_classic,
    bench_yuy2_1280x720_auto_custom,
    bench_hd_interlaced_encode_worker_scaling,
);
criterion_main!(benches);
