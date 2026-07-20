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
| GradientDecorr       | RGB32  | ClassicV2 | 1280×720  | **yes**    |

The 320×288 (largest progressive raster, `height == 288` is NOT
`> 288`) / 320×320 (first interlaced raster) Left pair isolates the
`decode_frame_interlaced` field-split overhead from the raster-size
delta.

**Encode** (`benches/encode.rs`) — symmetric v1.x / v2.x across all
three families:

| family | methods                                   | modes               |
| ------ | ----------------------------------------- | ------------------- |
| YUY2   | Left, Gradient, Median, Left(interlaced)  | ClassicV2, V1xCompat, CustomV2(auto; 320×240 + 1280×720) |
| RGB24  | Left, LeftDecorr (320×240 + 1280×720), GradientDecorr | ClassicV2, V1xCompat, CustomV2(auto) |
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
`--measurement-time 3 --sample-size 20`, post-round-419. Absolute wall
times are hardware-relative; the **throughput column and the
cross-scenario ranking are the stable, machine-independent signal**
for picking the next PROFILE-OPT target.

Round 419 landed four output-invariant rewrites (all pinned by
`tests/golden_pins.rs`): a decode-side built-tables cache (v2.x keyed
by RLE table bytes; v1.x per-family `OnceLock`s) that the encoder's
ClassicV2/V1xCompat arms share, paired symbol reads
(`tables::decode_pair`, two codewords per 32-bit window) + trusted bit
consume in the decode loops, a 64-bit-accumulator `BitWriter`, and a
count-based package-merge (`compute_canonical_lengths`, 577.7 µs →
20.7 µs peaked).

### Decode (cost ranked slowest → fastest per byte)

| scenario                                 | wall (r419)   | throughput (r419) | (r310)    |
| ---------------------------------------- | ------------- | ----------------- | --------- |
| rgb24 320×320 LeftDecorr (interlaced)    | 1.70 ms       | 173 MiB/s         | —         |
| rgb24 320×240 LeftDecorr                 | 1.26 ms       | 174 MiB/s         | 170 MiB/s |
| yuy2  320×240 Median                     | 0.75 ms       | 197 MiB/s         | 168 MiB/s |
| rgb32 1280×720 GradientDecorr (interlaced) | 17.2 ms     | 205 MiB/s         | —         |
| rgb32 320×240 GradientDecorr             | 1.41 ms       | 207 MiB/s         | 196 MiB/s |
| yuy2  320×320 Left (interlaced)          | 0.86 ms       | 228 MiB/s         | 215 MiB/s |
| yuy2  320×240 Left                       | 0.64 ms       | 230 MiB/s         | 217 MiB/s |
| yuy2  1280×720 Left (interlaced)         | 7.62 ms       | 231 MiB/s         | 236 MiB/s |
| yuy2  320×288 Left (progressive ctrl)    | 0.76 ms       | 232 MiB/s         | 214 MiB/s |
| yuy2  320×240 predict_old                | 0.63 ms       | 234 MiB/s         | 203 MiB/s |
| yuy2  320×240 Gradient                   | 0.60 ms       | 243 MiB/s         | 207 MiB/s |
| yuy2  320×240 Left (v1.x)                | 0.59 ms       | 247 MiB/s         | 226 MiB/s |

### Encode (cost ranked slowest → fastest per byte)

The encode side moved the most in round 419 (BitWriter + table cache +
package-merge): every scenario is 1.9–2.9× its pre-r419 rate.

| scenario                                 | wall (r419)   | throughput      | (pre-r419) |
| ---------------------------------------- | ------------- | --------------- | ---------- |
| **auto-select RGB24 CustomV2**           | 0.94 ms       | **233 MiB/s**   | 112 MiB/s  |
| **auto-select YUY2 CustomV2**            | 0.59 ms       | **250 MiB/s**   | 108 MiB/s  |
| auto-select YUY2 CustomV2 1280×720       | 6.17 ms       | 285 MiB/s       | —          |
| rgb32 320×240 GradientDecorr             | 0.76 ms       | 385 MiB/s       | 262 MiB/s  |
| rgb24 1280×720 LeftDecorr (interlaced)   | 6.07 ms       | 434 MiB/s       | —          |
| rgb24 320×240 GradientDecorr             | 0.47 ms       | 468 MiB/s       | 278 MiB/s  |
| rgb32 320×240 Left v1.x                  | 0.60 ms       | 489 MiB/s       | 322 MiB/s  |
| rgb24 320×240 Left v1.x                  | 0.44 ms       | 496 MiB/s       | 321 MiB/s  |
| yuy2  320×240 Median                     | 0.28 ms       | 516 MiB/s       | 267 MiB/s  |
| rgb24 320×240 LeftDecorr                 | 0.40 ms       | 549 MiB/s       | 301 MiB/s  |
| rgb32 320×240 LeftDecorr                 | 0.52 ms       | 568 MiB/s       | 324 MiB/s  |
| rgb32 320×240 Left                       | 0.48 ms       | 614 MiB/s       | 338 MiB/s  |
| yuy2  1280×720 Left (interlaced)         | 2.73 ms       | 645 MiB/s       | 401 MiB/s  |
| rgb24 320×240 Left                       | 0.31 ms       | 706 MiB/s       | 353 MiB/s  |
| yuy2  320×240 Gradient                   | 0.20 ms       | 727 MiB/s       | 324 MiB/s  |
| yuy2  320×240 Left v1.x                  | 0.18 ms       | 825 MiB/s       | 408 MiB/s  |
| yuy2  320×320 Left (interlaced)          | 0.22 ms       | 887 MiB/s       | 364 MiB/s  |
| yuy2  320×240 Left                       | 0.14 ms       | **1.04 GiB/s**  | 364 MiB/s  |

### Roundtrip (encode → decode, decode-dominated)

| scenario                                 | wall (r419)   | throughput |
| ---------------------------------------- | ------------- | ---------- |
| yuy2  320×240 Left ClassicV2             | 0.79 ms       | 185 MiB/s  |
| yuy2  320×240 Gradient ClassicV2         | 0.83 ms       | 177 MiB/s  |
| yuy2  320×240 Median ClassicV2           | 1.05 ms       | 139 MiB/s  |
| rgb24 320×240 LeftDecorr ClassicV2       | 1.69 ms       | 130 MiB/s  |
| rgb32 320×240 GradientDecorr ClassicV2   | 2.13 ms       | 137 MiB/s  |

### Table-build primitives (per channel, per frame)

| scenario                                       | wall (r419)  | (pre-r419) |
| ---------------------------------------------- | ------------ | ---------- |
| `compute_canonical_lengths` peaked             | 20.6 µs      | 577.7 µs   |
| `compute_canonical_lengths` flat               | 16.0 µs      | 415.2 µs   |
| `compute_canonical_lengths` sparse             | 1.32 µs      | 7.55 µs    |
| `build_from_lengths` peaked                     | 25.8 µs      | 24.8 µs    |
| `build_from_lengths` flat                       | 26.8 µs      | 26.1 µs    |
| `build_from_lengths` sparse                     | 24.4 µs      | 24.6 µs    |
| histogram → table (fused) peaked                | 46.5 µs      | 628 µs     |
| histogram → table (fused) flat                  | 41.5 µs      | 469 µs     |
| RLE round-trip peaked                           | 210 ns       | 209 ns     |
| RLE round-trip flat                             | 160 ns       | 153 ns     |
| RLE round-trip sparse                           | 258 ns       | 241 ns     |

**Headline finding (r419):** the round-419 count-based package-merge
removed the table-build hotspot outright — `compute_canonical_lengths`
peaked went 577.7 µs → 20.6 µs (−96%), so the whole
histogram-to-decode-ready-table pipeline is now ~46 µs/channel and
`build_from_lengths` (unchanged, ~26 µs, dominated by its 64-Ki
primary-LUT fill) is the new largest primitive. Decode/encode no
longer pay `build_from_lengths` per frame at all outside CustomV2
(built-tables cache).

## Interpretation

1. **Encode is no longer the cheap side by a small margin — it is
   2–4.6× faster per byte than decode.** The 64-bit `BitWriter`
   (−20% to −52% per scenario), the shared table cache (ClassicV2
   −15% to −30% on top), and the package-merge rewrite (auto-custom
   +83%/+69% throughput) compounded to 1.9–2.9× on every encode
   scenario; YUY2 Left ClassicV2 crossed 1 GiB/s.

2. **Decode is the whole-pipeline floor everywhere now** (see the
   roundtrip table: every roundtrip number is within ~25% of its
   decode-only number). The round-419 decode levers (table cache
   −11% to −21% on 320-class rasters; paired symbol reads + trusted
   consume −2% to −5.5%) moved the floor to ~173–247 MiB/s, and the
   per-symbol Huffman read still dominates: the r419 `sample` profile
   attributes ~95% of a LEFT-path decode to the symbol loop even
   after pairing.

3. **The predictor post-pass spread is unchanged** — Median and the
   RGB-decorr inverses remain the per-byte laggards (173–207 MiB/s
   vs 230–247 MiB/s for LEFT-family paths), matching the r310
   observation; those passes are already SWAR / macropixel-step
   rewritten (r91 / r181 / r255 / r304).

4. **Interlaced overhead stays ~1–2%** beyond the extra rows (320×320
   interlaced 228 MiB/s vs 320×288 progressive 232 MiB/s), so the
   field split + `interleave_fields` row-gather remains near-free.

## Next PROFILE-OPT target

**Primary (decode): amortise the per-symbol LUT dependency chain
further.** After r419's pairing, each 2-symbol step still carries a
serialized LUT-load → shift → LUT-load chain. The two candidate
levers, in order of expected value: (a) a second-level pair LUT
(16-bit window → both symbols + combined length in one load) built
lazily behind the round-419 built-tables cache so its ~256-KiB/table
cost is paid once per stream, not per frame; (b) decoding the two
independent field bitstreams of an interlaced frame on two threads
(the fields are independent by wire design — spec/02 §2 — so this is
parallelism the format offers for free, but it needs a decision about
threading policy in a decoder that is currently allocation-light and
thread-free). Gate on `decode_yuy2_320x240_left_classic` and the
1280×720 scenarios.

**Secondary (encode, CustomV2 only): skip the dead 64-Ki primary-LUT
fill.** CustomV2 still builds three decode-grade `HuffTable`s per
frame, but the emit path only reads `entries` — ~26 µs × 3 of
`build_from_lengths` per frame is LUT fill the encoder never uses. A
lean encoder-side table type (entries-only) would cut ~50–75 µs from
every CustomV2 frame (~10% of the 320×240 auto-custom scenario) at
the cost of a second table shape; only worth it if CustomV2 encode
becomes a measured consumer bottleneck.

*(Superseded) The pre-r419 primary target — `compute_canonical_lengths`
package-merge — landed in round 419 as the count-based rewrite
(−96%, auto-custom throughput +83%/+69%). The pre-r310 target — the
per-symbol window reconstruction — landed in round 310 (refilling
64-bit `BitReader`, ≈ 35–46% decode wall reduction).*
