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

Post-round-310 (refilling 64-bit `bitio::BitReader`). The pre-r310
column is retained alongside so the bit-reader win is readable inline;
the round-310 rewrite eliminated the per-symbol from-scratch window
reconstruction that set the flat pre-r310 ~100–123 MiB/s ceiling.

| scenario                                | wall (r310)   | throughput (r310) | wall (pre-r310) |
| --------------------------------------- | ------------- | ----------------- | --------------- |
| rgb24 320×240 LeftDecorr                 | 1.29 ms       | 170 MiB/s         | 2.17 ms         |
| rgb32 320×240 GradientDecorr             | 1.50 ms       | 196 MiB/s         | 2.60 ms         |
| yuy2  320×240 Median                     | 0.87 ms       | 168 MiB/s         | 1.43 ms         |
| yuy2  320×320 Left (interlaced)          | 0.91 ms       | 215 MiB/s         | 1.69 ms         |
| yuy2  320×288 Left (progressive ctrl)    | 0.82 ms       | 214 MiB/s         | 1.53 ms         |
| yuy2  320×240 predict_old                | 0.72 ms       | 203 MiB/s         | 1.28 ms         |
| yuy2  320×240 Gradient                   | 0.71 ms       | 207 MiB/s         | 1.27 ms         |
| yuy2  320×240 Left                       | 0.68 ms       | 217 MiB/s         | 1.18 ms         |
| yuy2  320×240 Left (v1.x)                | 0.65 ms       | 226 MiB/s         | 1.28 ms         |
| yuy2  1280×720 Left (interlaced)         | 7.44 ms       | 236 MiB/s         | 13.81 ms        |

The decode rate roughly doubled across the board (~35–46% wall
reduction) and is now no longer the pre-r310 flat ceiling; the
predictor-post-pass differences (Median / decorr still the per-byte
laggards) are starting to surface above the bit-read floor.

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

1. **Decode is still the whole-pipeline floor, but the floor moved.**
   Round 310's refilling bit reader lifted decode from the flat pre-r310
   ~100–123 MiB/s ceiling to ~170–236 MiB/s. Encode of the same frame
   is now ~1.3–2× faster per byte (was 2–3.5×); end-to-end
   (`roundtrip`) cost remains decode-dominated but by a smaller margin.

2. **Decode throughput is no longer flat across predictors.** Pre-r310
   the rate was uniform (101–116 MiB/s for YUY2 Left / Gradient /
   Median / predict_old / v1.x) — the strongest evidence that the
   per-symbol Huffman read, not the `inverse_*_post` predictor pass,
   governed throughput. Round 310 removed that bottleneck (per-symbol
   from-scratch window reconstruction → a 64-bit accumulator topped up
   one word per ~32 consumed bits), and the predictor post-passes now
   surface above the bit-read floor: Median (0.87 ms, its own
   gradient/median scan) and the RGB-decorr passes (LeftDecorr 1.29 ms,
   GradientDecorr 1.50 ms — extra per-byte reconstruction work) are
   measurably the per-byte laggards, while the cheap LEFT/v1.x paths
   run fastest.

3. **Interlaced overhead is ~11%** (320×320 interlaced 0.91 ms vs
   320×288 progressive control 0.82 ms — close to the +11% row delta,
   so the field-split + `interleave_fields` row-gather is near-free per
   extra row; the cost is the raster, not the split).

4. **The encode auto-selector is the encode-side hotspot** — CustomV2
   `encode_frame_auto` runs at ~104–108 MiB/s, a clean 3–3.5× slower
   than any single-method fixed encode, because it builds a
   package-merge length table per candidate method.

## Next PROFILE-OPT target

**Primary (encode): the `compute_canonical_lengths` package-merge
length builder.** With the round-310 decode bit-read bottleneck cleared,
the encode auto-selector is the workspace's largest remaining
per-byte cost. The isolated table-build bench (`benches/tables.rs`)
confirms this primitive (~589 µs peaked, ~1.8 ms for a 3-channel frame)
is the entire table-build cost — `build_from_lengths` (~30 µs) and the
RLE codec (~209 ns) are noise beside it. At 3–3.5× the cost of
fixed-method encode it dominates any auto-selected encode; caching /
reusing per-candidate length-table work across the method sweep, or a
cheaper bounded-depth length algorithm, is the lever. Gate the
whole-frame effect on `encode_yuy2_320x240_auto_custom` and the
isolated primitive on `tables_compute_canonical_lengths/peaked`.

**Secondary (decode): the predictor inverse post-passes now visible
above the round-310 bit-read floor.** Median (`inverse_yuy2_median`)
and the RGB-decorr inverses (`inverse_rgb_decorr_bgr/bgra`,
`inverse_gradient_post`) are the per-byte laggards in the post-r310
decode table; a profile run on `decode_rgb24_320x240_left_decorr` or
`decode_yuy2_320x240_median_classic` should now attribute the spread
above the LEFT path to the post-pass rather than the symbol read. The
LEFT / gradient / median / decorr inverses are already SWAR /
macropixel-step / `chunks_exact_mut` rewritten (r91 / r181 / r255 /
r304), so the next decode win is likely batching the macropixel
symbol read (decode N codewords before applying the predictor) rather
than another post-pass rewrite.

*(Superseded) Primary pre-r310 target — the inner per-symbol Huffman
read in `decoder::decode_{yuy2,rgb24,rgb32}_field` — was landed in
round 310 (refilling 64-bit `bitio::BitReader`): ≈ 35–46% decode
wall reduction, ~115 → ~217 MiB/s on the YUY2 Left 320×240 headline
gate.)*
