# oxideav-huffyuv benchmarks

Criterion micro-benchmarks for the HuffYUV / FFVHuff decode fast-LUT
path and the v1.x / v2.x symmetric encode path. All inputs are
synthesised on the fly from a deterministic xorshift32 gradient
(`build_pixels`) — no committed fixtures, no `docs/` access at run
time, no third-party samples. Each scenario isolates one
`(family, method, extradata-mode, size)` quadruple so an optimisation
lands as a readable Criterion delta against the matching baseline.

```
cargo bench -p oxideav-huffyuv --bench decode
cargo bench -p oxideav-huffyuv --bench encode
cargo bench -p oxideav-huffyuv --bench roundtrip
cargo bench -p oxideav-huffyuv --bench tables
```

## Coverage

**Decode** (`benches/decode.rs`) — all six on-wire methods plus the
progressive/interlaced split:

| method               | family | mode      | size      | interlaced |
| -------------------- | ------ | --------- | --------- | ---------- |
| `predict_old` (`-2`) | YUY2   | ClassicV2 | 320×240   | no         |
| Left (`0x00`)        | YUY2   | ClassicV2 | 320×240   | no         |
| Gradient (`0x01`)    | YUY2   | ClassicV2 | 320×240   | no         |
| Median (`0x02`)      | YUY2   | ClassicV2 | 320×240   | no         |
| Left                 | YUY2   | V1xCompat | 320×240   | no         |
| Left                 | YUY2   | ClassicV2 | 320×288   | no (ctrl)  |
| Left                 | YUY2   | ClassicV2 | 320×320   | **yes**    |
| Left                 | YUY2   | ClassicV2 | 1280×720  | **yes**    |
| LeftDecorr (`0x40`)  | RGB24  | ClassicV2 | 320×240   | no         |
| LeftDecorr           | RGB24  | ClassicV2 | 320×320   | **yes**    |
| GradientDecorr (`0x41`) | RGB32 | ClassicV2 | 320×240 | no         |

The 320×288 (largest progressive raster, `height == 288` is NOT
`> 288`) / 320×320 (first interlaced raster) Left pair isolates the
`decode_frame_interlaced` field-split overhead from the raster-size
delta.

**Encode** (`benches/encode.rs`) — symmetric v1.x / v2.x across all
three families:

| family | methods                                   | modes               |
| ------ | ----------------------------------------- | ------------------- |
| YUY2   | Left, Gradient, Median, Left(interlaced)  | ClassicV2, V1xCompat, CustomV2(auto) |
| RGB24  | Left, LeftDecorr, GradientDecorr          | ClassicV2, V1xCompat, CustomV2(auto) |
| RGB32  | Left, LeftDecorr, GradientDecorr          | ClassicV2, V1xCompat |

**Table-build primitives** (`benches/tables.rs`) — the per-channel,
per-frame work the CustomV2 / auto-select path pays, isolated from the
whole-frame encode/decode so a future package-merge or table-build
optimisation lands as a clean Criterion delta (the `encode_*_auto_custom`
scenarios conflate it with residuals + histogramming + emit):

| primitive                       | what it costs                             |
| ------------------------------- | ----------------------------------------- |
| `compute_canonical_lengths`     | length-limited (max-31) package-merge — **encoder** length builder, once per channel per CustomV2 frame |
| `HuffTable::build_from_lengths` | canonical code assign (spec/04 §2.6) + 64 Ki LUT + overflow bake — **decoder** table build, once per channel per frame |
| histogram → table (fused)       | the two combined: "residual histogram → decode-ready `HuffTable`" |
| RLE round-trip                  | `rle_encode_one_channel` → `rle_decode_one_channel` length-table codec (spec/04 §2.2, spec/01 §5) |

Each primitive runs against three histogram shapes: `peaked` (geometric
decay outward from residual 0 / its wrap-neighbour 255, full 256-symbol
alphabet live — frame-realistic), `flat` (uniform — package-merge
balanced-tree worst case), and `sparse` (5 live symbols — small-alphabet
fast path).

## Ranked hotspot table

Measured on aarch64 (Apple, macOS), Criterion 0.5, dev machine,
`--measurement-time 3 --sample-size 20`. Absolute wall times are
hardware-relative; the **throughput column and the cross-scenario
ranking are the stable, machine-independent signal** for picking the
next PROFILE-OPT target.

### Decode (cost ranked slowest → fastest per byte)

| scenario                                | wall (median) | throughput   |
| --------------------------------------- | ------------- | ------------ |
| rgb24 320×240 LeftDecorr                 | 2.17 ms       | 101 MiB/s    |
| rgb24 320×320 LeftDecorr (interlaced)    | 2.91 ms       | 101 MiB/s    |
| yuy2  320×240 Median                     | 1.43 ms       | 102 MiB/s    |
| rgb32 320×240 GradientDecorr             | 2.60 ms       | 113 MiB/s    |
| yuy2  320×320 Left (interlaced)          | 1.69 ms       | 116 MiB/s    |
| yuy2  320×288 Left (progressive ctrl)    | 1.53 ms       | 115 MiB/s    |
| yuy2  320×240 Left                       | 1.28 ms       | 115 MiB/s    |
| yuy2  320×240 predict_old                | 1.28 ms       | 114 MiB/s    |
| yuy2  320×240 Gradient                   | 1.27 ms       | 116 MiB/s    |
| yuy2  320×240 Left (v1.x)                | 1.28 ms       | 115 MiB/s    |
| yuy2  1280×720 Left (interlaced)         | 14.3 ms       | 123 MiB/s    |

### Encode (cost ranked slowest → fastest per byte)

| scenario                                | wall (median) | throughput   |
| --------------------------------------- | ------------- | ------------ |
| **auto-select RGB24 CustomV2**          | 2.11 ms       | **108 MiB/s** |
| **auto-select YUY2 CustomV2**           | 1.43 ms       | **104 MiB/s** |
| rgb32 320×240 GradientDecorr            | 1.19 ms       | 244 MiB/s    |
| yuy2  320×240 Median                    | 0.58 ms       | 243 MiB/s    |
| rgb24 320×240 GradientDecorr           | 0.86 ms       | 253 MiB/s    |
| rgb24 320×240 LeftDecorr               | 0.76 ms       | 270 MiB/s    |
| rgb32 320×240 LeftDecorr               | 0.96 ms       | 276 MiB/s    |
| yuy2  320×240 Gradient                  | 0.47 ms       | 295 MiB/s    |
| rgb24 320×240 Left v1.x                 | 0.73 ms       | 309 MiB/s    |
| rgb32 320×240 Left v1.x                 | 0.92 ms       | 317 MiB/s    |
| rgb32 320×240 Left                      | 0.93 ms       | 320 MiB/s    |
| rgb24 320×240 Left                      | 0.66 ms       | 336 MiB/s    |
| yuy2  320×320 Left (interlaced)         | 0.55 ms       | 351 MiB/s    |
| yuy2  320×240 Left                      | 0.42 ms       | 355 MiB/s    |
| yuy2  1280×720 Left (interlaced)        | 4.50 ms       | 376 MiB/s    |
| yuy2  320×240 Left v1.x                 | 0.37 ms       | 384 MiB/s    |

### Table-build primitives (per channel, per frame)

| scenario                                       | wall (median) |
| ---------------------------------------------- | ------------- |
| `compute_canonical_lengths` peaked             | 589 µs        |
| `compute_canonical_lengths` flat               | 415 µs        |
| `compute_canonical_lengths` sparse             | 7.6 µs        |
| `build_from_lengths` peaked                     | 30 µs         |
| `build_from_lengths` flat                       | 27 µs         |
| `build_from_lengths` sparse                     | 28 µs         |
| histogram → table (fused) peaked                | 633 µs        |
| histogram → table (fused) flat                  | 472 µs        |
| RLE round-trip peaked                           | 209 ns        |
| RLE round-trip flat                             | 164 ns        |
| RLE round-trip sparse                           | 177 ns        |

**Headline finding:** the package-merge length builder
(`compute_canonical_lengths`, ~589 µs peaked) dominates the table-build
pipeline by ~20× over the decoder-side `build_from_lengths` (~30 µs) and
by ~2800× over the RLE codec (~209 ns). For a 3-channel CustomV2 frame
the encoder pays ~1.8 ms in package-merge alone — which is why
`encode_frame_auto` (one length-table build per candidate method) is the
3–3.5× encode-side outlier. Any auto-select speedup should target this
primitive; the decode-side table build and the RLE codec are negligible.

## Interpretation

1. **Decode is the whole-pipeline floor.** Decode tops out at
   ~100–123 MiB/s regardless of predictor; encode of the same frame
   is 2–3.5× faster per byte. End-to-end (`roundtrip`) cost is
   decode-dominated.

2. **Decode throughput is flat across predictors** (101–116 MiB/s for
   YUY2 Left / Gradient / Median / predict_old / v1.x). The predictor
   post-pass (`inverse_*_post`) is NOT the decode bottleneck — if it
   were, Median (its own scan) would diverge sharply from Left. The
   uniform ceiling points at the **per-symbol Huffman read** in the
   inner decode loop (`decoder::decode_*_field` bit-reader + LUT
   lookup), which every method runs identically. The RGB-decorr and
   1280×720 rows confirm the rate is byte-volume-bound, not
   method-bound.

3. **Interlaced overhead is ~10%** (320×320 interlaced 1.69 ms vs
   320×288 progressive control 1.53 ms — +10% wall on +11% rows, so
   the field-split + `interleave_fields` row-gather is close to free
   per extra row; the cost is the raster, not the split).

4. **The encode auto-selector is the encode-side hotspot** — CustomV2
   `encode_frame_auto` runs at ~104–108 MiB/s, a clean 3–3.5× slower
   than any single-method fixed encode, because it builds a
   package-merge length table per candidate method.

## Next PROFILE-OPT target

**Primary (decode): the inner per-symbol Huffman read in
`decoder::decode_{yuy2,rgb24,rgb32}_field`.** The flat ~100–120 MiB/s
ceiling across all six predictors is the strongest evidence that the
bit-reader + canonical-LUT lookup, not the predictor inverse,
governs decode throughput. A profile run (e.g. `samply` / `perf`) on
`decode_yuy2_320x240_left_classic` should attribute the bulk of
samples to the symbol-read path; widening the LUT, refilling the bit
buffer in larger chunks, or batching the macropixel read are the
candidate wins. Use the YUY2 Left 320×240 scenario as the headline
regression gate and the 1280×720 interlaced scenario as the HD
confirmation.

**Secondary (encode): the `compute_canonical_lengths` package-merge
length builder.** The isolated table-build bench
(`benches/tables.rs`) confirms this primitive (~589 µs peaked, ~1.8 ms
for a 3-channel frame) is the entire table-build cost — `build_from_lengths`
(~30 µs) and the RLE codec (~209 ns) are noise beside it. At 3–3.5× the
cost of fixed-method encode it dominates any auto-selected encode;
caching / reusing per-candidate length-table work across the method
sweep, or a cheaper bounded-depth length algorithm, is the lever. Gate
the whole-frame effect on `encode_yuy2_320x240_auto_custom` and the
isolated primitive on `tables_compute_canonical_lengths/peaked`.
