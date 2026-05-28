//! Encode-then-decode roundtrip benchmark. A single-figure end-to-end
//! pipeline health check that sums the encode + decode wall times so a
//! regression in either side is visible.
//!
//! Each iteration re-encodes from the same pre-built `&pixels` slice,
//! parses the BIH back, and decodes — the same shape as the
//! `tests/round4_avi_lockstep` end-to-end test but without the
//! container layer (so the bench measures codec-internal cost, not
//! container marshalling).
//!
//! Run with:
//!     cargo bench -p oxideav-huffyuv --bench roundtrip

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use oxideav_huffyuv::{
    decode_frame, encode_frame_with_mode, ExtradataMode, Method, PixelFamily, StreamConfig,
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

fn bench_roundtrip(
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
                let (strf, frame) = encode_frame_with_mode(
                    family,
                    method,
                    width,
                    height,
                    criterion::black_box(&pixels),
                    mode,
                )
                .expect("encode");
                let cfg = StreamConfig::parse_bitmapinfoheader(&strf).expect("parse BIH");
                let _decoded = decode_frame(&cfg, &frame).expect("decode");
            });
        },
    );
    g.finish();
}

fn bench_yuy2_320x240_left_classic(c: &mut Criterion) {
    bench_roundtrip(
        c,
        "roundtrip_yuy2_320x240_left_classic",
        PixelFamily::Yuy2,
        Method::Left,
        320,
        240,
        ExtradataMode::ClassicV2,
    );
}

fn bench_yuy2_320x240_gradient_classic(c: &mut Criterion) {
    bench_roundtrip(
        c,
        "roundtrip_yuy2_320x240_gradient_classic",
        PixelFamily::Yuy2,
        Method::Gradient,
        320,
        240,
        ExtradataMode::ClassicV2,
    );
}

fn bench_yuy2_320x240_median_classic(c: &mut Criterion) {
    bench_roundtrip(
        c,
        "roundtrip_yuy2_320x240_median_classic",
        PixelFamily::Yuy2,
        Method::Median,
        320,
        240,
        ExtradataMode::ClassicV2,
    );
}

fn bench_rgb24_320x240_left_decorr_classic(c: &mut Criterion) {
    bench_roundtrip(
        c,
        "roundtrip_rgb24_320x240_left_decorr_classic",
        PixelFamily::Rgb24,
        Method::LeftDecorr,
        320,
        240,
        ExtradataMode::ClassicV2,
    );
}

fn bench_rgb32_320x240_gradient_decorr_classic(c: &mut Criterion) {
    bench_roundtrip(
        c,
        "roundtrip_rgb32_320x240_gradient_decorr_classic",
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
    bench_rgb24_320x240_left_decorr_classic,
    bench_rgb32_320x240_gradient_decorr_classic,
);
criterion_main!(benches);
