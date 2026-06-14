//! Criterion benchmarks for the HuffYUV / FFVHuff **table-build**
//! primitives — the per-frame work the CustomV2 / auto-select encode
//! path pays that the whole-frame `decode` / `encode` / `roundtrip`
//! benches cannot isolate.
//!
//! `BENCHMARKS.md` names this path as the encode-side secondary
//! hotspot: `encode_frame_auto` "builds a package-merge length table
//! per candidate method" and runs at 3–3.5× the cost of a fixed-method
//! encode. The full-frame `encode_*_auto_custom` scenarios in
//! `benches/encode.rs` conflate residual computation, histogramming,
//! length-table building, RLE packing, and the Huffman emit, so a
//! future package-merge or table-build optimisation cannot be read off
//! a clean Criterion delta. This bench isolates the four table-build
//! primitives directly so each one carries its own baseline:
//!
//!   - **`compute_canonical_lengths`** (`src/tables.rs`) — the
//!     length-limited (max-31) package-merge Huffman length builder the
//!     CustomV2 path runs once per channel per frame. This is the
//!     secondary hotspot `BENCHMARKS.md` calls out.
//!   - **`build_from_lengths`** (`src/tables.rs`) — the
//!     canonical-Huffman code builder (spec/04 §2.6, longest-length
//!     tier first) plus the 64 Ki-entry primary LUT + overflow-entry
//!     bake that the **decoder** runs once per channel per frame to
//!     turn a length table into a fast-decode `HuffTable`.
//!   - **`compute_canonical_lengths` → `build_from_lengths`** fused —
//!     the end-to-end "histogram → decode-ready table" pipeline, so the
//!     two primitives' combined per-frame cost has a single baseline.
//!   - **`rle_encode_one_channel` → `rle_decode_one_channel`** — the
//!     RLE length-table codec (spec/04 §2.2 decode / spec/01 §5) the
//!     CustomV2 path uses to pack a length table into the AVI extradata
//!     and the decoder uses to unpack it.
//!
//! Inputs are synthesised on the fly (no committed fixtures, no `docs/`
//! access at run time) from deterministic histograms shaped like real
//! predictor-residual distributions — heavily peaked near residual 0
//! with a long low-probability tail — so the package-merge tree depth
//! and the per-tier length spread match what `histogramise` produces
//! on a natural frame. A flat (uniform) histogram and a degenerate
//! few-symbol histogram bracket the easy/hard ends of the builder.
//!
//! Run with:
//!     cargo bench -p oxideav-huffyuv --bench tables

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use oxideav_huffyuv::tables::{
    compute_canonical_lengths, rle_decode_one_channel, rle_encode_one_channel, HuffTable,
};

/// A peaked, long-tailed 256-entry frequency histogram approximating a
/// real predictor-residual distribution: most residuals cluster around
/// 0 / 255 (the modular-wrap neighbours of 0), with an exponentially
/// decaying tail across the rest of the symbol space. `total` scales
/// the counts so the package-merge sees frame-realistic magnitudes.
fn peaked_histogram(total: u32) -> [u32; 256] {
    let mut h = [0u32; 256];
    // Geometric-ish decay outward from residual 0 (and its wrap
    // neighbour 255). Distance from the nearest of {0, 255} drives the
    // weight; every symbol keeps a floor count of 1 so the full
    // 256-symbol alphabet is live (max package-merge tree depth).
    let mut sum = 0u64;
    let mut raw = [0u64; 256];
    for (s, slot) in raw.iter_mut().enumerate() {
        let d = (s.min(256 - s)) as u32; // 0 at s==0, small near 0 and 255
                                         // Weight halves roughly every 4 symbols of distance; +1 floor.
        let w = (1u64 << 20) >> (d / 4).min(20);
        *slot = w + 1;
        sum += w + 1;
    }
    for (s, &w) in raw.iter().enumerate() {
        // Scale to `total`, keeping a floor of 1 so no symbol drops out.
        let scaled = ((w as u128 * total as u128) / sum as u128) as u32;
        h[s] = scaled.max(1);
    }
    h
}

/// A flat (uniform) 256-entry histogram — every symbol equally likely.
/// This is the package-merge builder's balanced-tree worst case for
/// per-tier package count and forces an (almost) full-depth 8-bit code
/// assignment.
fn flat_histogram(per_symbol: u32) -> [u32; 256] {
    [per_symbol; 256]
}

/// A degenerate few-symbol histogram: a handful of live symbols with
/// the rest zero. Exercises the small-alphabet fast paths
/// (`active.len() <= 2`) and the short-tier branch of the builder.
fn sparse_histogram() -> [u32; 256] {
    let mut h = [0u32; 256];
    h[0] = 10_000;
    h[1] = 3_000;
    h[2] = 800;
    h[255] = 120;
    h[128] = 5;
    h
}

fn bench_compute_canonical_lengths(c: &mut Criterion) {
    let peaked = peaked_histogram(320 * 240); // ~one YUY2 channel of a 320x240 frame
    let flat = flat_histogram(300);
    let sparse = sparse_histogram();

    let mut g = c.benchmark_group("tables_compute_canonical_lengths");
    // One call processes a 256-entry histogram; report symbols/s.
    g.throughput(Throughput::Elements(256));
    for (name, hist) in [("peaked", &peaked), ("flat", &flat), ("sparse", &sparse)] {
        g.bench_function(BenchmarkId::from_parameter(name), |b| {
            b.iter(|| {
                compute_canonical_lengths(criterion::black_box(hist)).expect("canonical lengths")
            });
        });
    }
    g.finish();
}

fn bench_build_from_lengths(c: &mut Criterion) {
    // Derive realistic length tables from the histograms above, then
    // benchmark the decode-table build (canonical code assignment + LUT
    // + overflow bake) on them — this is the decoder's per-channel cost.
    let peaked = compute_canonical_lengths(&peaked_histogram(320 * 240)).unwrap();
    let flat = compute_canonical_lengths(&flat_histogram(300)).unwrap();
    let sparse = compute_canonical_lengths(&sparse_histogram()).unwrap();

    let mut g = c.benchmark_group("tables_build_from_lengths");
    g.throughput(Throughput::Elements(256));
    for (name, lengths) in [("peaked", &peaked), ("flat", &flat), ("sparse", &sparse)] {
        g.bench_function(BenchmarkId::from_parameter(name), |b| {
            b.iter(|| {
                HuffTable::build_from_lengths(criterion::black_box(lengths)).expect("build table")
            });
        });
    }
    g.finish();
}

fn bench_histogram_to_table_pipeline(c: &mut Criterion) {
    // End-to-end "histogram → decode-ready HuffTable" — the combined
    // per-channel cost the CustomV2 encode (length build) + decode
    // (table build) pay together.
    let peaked = peaked_histogram(320 * 240);
    let flat = flat_histogram(300);

    let mut g = c.benchmark_group("tables_histogram_to_table");
    g.throughput(Throughput::Elements(256));
    for (name, hist) in [("peaked", &peaked), ("flat", &flat)] {
        g.bench_function(BenchmarkId::from_parameter(name), |b| {
            b.iter(|| {
                let lengths =
                    compute_canonical_lengths(criterion::black_box(hist)).expect("lengths");
                HuffTable::build_from_lengths(&lengths).expect("table")
            });
        });
    }
    g.finish();
}

fn bench_rle_roundtrip(c: &mut Criterion) {
    // The CustomV2 wire path: pack a 256-entry length table to RLE
    // bytes, then unpack it again (the decoder side). One channel.
    let peaked = compute_canonical_lengths(&peaked_histogram(320 * 240)).unwrap();
    let flat = compute_canonical_lengths(&flat_histogram(300)).unwrap();
    let sparse = compute_canonical_lengths(&sparse_histogram()).unwrap();

    let mut g = c.benchmark_group("tables_rle_roundtrip");
    g.throughput(Throughput::Elements(256));
    for (name, lengths) in [("peaked", &peaked), ("flat", &flat), ("sparse", &sparse)] {
        g.bench_function(BenchmarkId::from_parameter(name), |b| {
            b.iter(|| {
                let mut out = Vec::with_capacity(64);
                rle_encode_one_channel(criterion::black_box(lengths), &mut out).expect("rle enc");
                let mut cursor: &[u8] = &out;
                rle_decode_one_channel(&mut cursor).expect("rle dec")
            });
        });
    }
    g.finish();
}

criterion_group!(
    benches,
    bench_compute_canonical_lengths,
    bench_build_from_lengths,
    bench_histogram_to_table_pipeline,
    bench_rle_roundtrip,
);
criterion_main!(benches);
