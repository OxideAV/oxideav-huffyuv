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

**Decode** (`benches/decode.rs`) — all six on-wire methods on both
rasters (plain, and field-coded by height, spec/06 §1.1):

| method               | family | mode      | size      | field-coded |
| -------------------- | ------ | --------- | --------- | ---------- |
| `predict_old` (`-2`) | YUY2   | ClassicV2 | 320×240   | no         |
| Left (`0x00`)        | YUY2   | ClassicV2 | 320×240   | no         |
| Gradient (`0x01`)    | YUY2   | ClassicV2 | 320×240   | no         |
| Median (`0x02`)      | YUY2   | ClassicV2 | 320×240   | no         |
| Left                 | YUY2   | V1xCompat | 320×240   | no         |
| Left                 | YUY2   | ClassicV2 | 320×288   | no (ctrl)  |
| Left                 | YUY2   | ClassicV2 | 320×320   | **yes**    |
| Left                 | YUY2   | ClassicV2 | 1280×720  | **yes**    |
| Gradient             | YUY2   | ClassicV2 | 1280×720  | **yes** (r493) |
| Median               | YUY2   | ClassicV2 | 1280×720  | **yes** (r493) |
| LeftDecorr           | RGB24  | ClassicV2 | 1280×720  | **yes** (r493) |
| LeftDecorr (`0x40`)  | RGB24  | ClassicV2 | 320×240   | no         |
| LeftDecorr           | RGB24  | ClassicV2 | 320×320   | **yes**    |
| GradientDecorr (`0x41`) | RGB32 | ClassicV2 | 320×240 | no         |
| GradientDecorr       | RGB32  | ClassicV2 | 1280×720  | **yes**    |

The 320×288 (tallest plain raster, `height == 288` is NOT `> 288`) /
320×320 (first field-coded raster) Left pair shows that field coding
costs nothing in the entropy loop: it only moves the predictors'
"above" reference.

**Encode** (`benches/encode.rs`) — symmetric v1.x / v2.x across all
three families:

| family | methods                                   | modes               |
| ------ | ----------------------------------------- | ------------------- |
| YUY2   | Left, Gradient, Median (320×240 + 1280×720), Left 320×320 | ClassicV2, V1xCompat, CustomV2(auto; 320×240 + 1280×720) |
| RGB24  | Left, LeftDecorr (320×240 + 1280×720), GradientDecorr | ClassicV2, V1xCompat, CustomV2(auto) |
| RGB32  | Left, LeftDecorr, GradientDecorr (320×240 + 1280×720) | ClassicV2, V1xCompat |

Worker-scaling gates: `encode_workers_*` (fixed method, 1 / 2 / 4
workers, 1280×720) and `encode_auto_workers_*` (auto-selector, 1 / 2 /
3 workers).

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

> **Read with round 493 in mind.** The absolute tables in this
> section were measured in rounds 419–440. The plain-raster LEFT /
> gradient rows still describe today's code. The Median rows and
> every decorrelated-RGB row were measured on streams with the
> pre-r493 wire defects (see CHANGELOG), and the rows marked
> "interlaced" measured the retired two-field layout; treat those as
> history. Current relative numbers are in "Round 493" below.

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

### Round 493

All numbers below are same-session A/B runs of two builds of this
crate — the commit before the performance phase (`cc4601f`) and the
current tree — alternated on one machine, **minimum of 4 runs per
build** (the host was shared with nine other build jobs; a minimum
approximates the uncontended wall, and rows within about ±10 % of
1.00 are noise — the unchanged YUY2 Left decode path reads 0.89–1.00
across runs). Apple Silicon, Criterion 0.5, synthetic gradient +
noise rasters, `ClassicV2` tables. Output is byte-identical between
the two builds (golden pins) and across worker budgets.

**Encode, 1 worker** — fused position-based residual kernels:

| scenario | before | after | speed-up |
| --- | --- | --- | --- |
| yuy2 320×240 Median | 0.367 ms | 0.196 ms | **1.87×** |
| yuy2 1280×720 Median | 5.22 ms | 3.51 ms | **1.49×** |
| rgb32 1280×720 GradientDecorr | 9.85 ms | 7.47 ms | **1.32×** |
| rgb24 1280×720 LeftDecorr | 6.48 ms | 5.17 ms | **1.25×** |
| rgb32 320×240 GradientDecorr | 0.724 ms | 0.595 ms | 1.22× |
| rgb24 320×240 LeftDecorr | 0.474 ms | 0.396 ms | 1.20× |
| yuy2 1280×720 auto CustomV2 | 6.33 ms | 5.24 ms | 1.21× |
| rgb24 320×240 Left | 0.394 ms | 0.348 ms | 1.13× |
| rgb24 320×240 GradientDecorr | 0.532 ms | 0.494 ms | 1.08× |
| yuy2 320×240 Left | 0.204 ms | 0.187 ms | noise |
| yuy2 1280×720 Left | 2.84 ms | 3.04 ms | noise |
| yuy2 320×240 Gradient | 0.225 / 0.517 ms | 0.271 / 0.441 ms | noise (0.83× and 1.17× in two runs) |

**Encode, fixed method, worker scaling** (1280×720;
`encode_frame_with_mode_workers`; "before" is serial at every budget
because the previous build had no split for a single-stream frame):

| scenario | before (any budget) | 1 worker | 2 workers | 4 workers |
| --- | --- | --- | --- | --- |
| yuy2 Left | 2.91 ms | 3.04 ms | 1.79 ms (**1.6×**) | 1.05 ms (**2.8×**) |
| yuy2 Median | 5.15 ms | 3.20 ms | 1.83 ms (**2.7×**) | 1.06 ms (**4.8×**) |
| rgb32 GradientDecorr | 9.70 ms | 7.01 ms | 4.03 ms (**2.4×**) | 2.14 ms (**4.7×**) |

(Speed-ups are against the "before" column, so they include the
serial kernel gain; against the new 1-worker wall, 4 workers give
2.9× / 3.0× / 3.3×.) At 4 workers a 1280×720 YUY2 frame encodes in
about 1 ms — roughly 1.7 GiB/s.

**Decode, 1 worker** (decode has no parallel split: one codeword
stream, no offsets on the wire):

| scenario | before | after | speed-up |
| --- | --- | --- | --- |
| yuy2 320×240 Median | 0.555 ms | 0.494 ms | **1.12×** |
| yuy2 1280×720 Median | 7.26 ms | 6.58 ms | **1.10×** |
| yuy2 1280×720 Gradient | 6.59 ms | 5.89 ms | 1.12× (noise band) |
| yuy2 Left, rgb24 LeftDecorr, rgb32 GradientDecorr | — | — | unchanged code, 0.88–1.00× = noise |

Formats newly gated this round (after, 1 worker, same run): yuy2
1280×720 Median decode 6.58 ms (267 MiB/s), yuy2 1280×720 Gradient
decode 5.89 ms (298 MiB/s), rgb24 1280×720 LeftDecorr decode 10.5 ms
(250 MiB/s), rgb32 1280×720 GradientDecorr decode 12.5 ms
(281 MiB/s).

**Encoded size against the black-box encoder** — one 320×240 frame,
same picture and same predictor on both sides, frame bytes
(`tests/blackbox_matrix.rs::encoded_size_against_black_box_encoder`
prints this table and asserts the `custom` column never exceeds the
black box):

| source | format | predictor | black box | ours classic | ours custom | ours auto+custom |
|---|---|---|---|---|---|---|
| pattern + noise | yuv422p | left | 131736 | 156964 (+19.2 %) | 107776 (-18.2 %) | 105768 (-19.7 %, Median) |
| pattern + noise | yuv422p | plane | 143908 | 196968 (+36.9 %) | 114168 (-20.7 %) | 105768 (-26.5 %, Median) |
| pattern + noise | yuv422p | median | 129356 | 176336 (+36.3 %) | 105768 (-18.2 %) | 105768 (-18.2 %, Median) |
| pattern + noise | rgb24 | left | 165296 | 184664 (+11.7 %) | 147024 (-11.1 %) | 138468 (-16.2 %, Left) |
| pattern + noise | rgb24 | plane | 181488 | 211072 (+16.3 %) | 157992 (-12.9 %) | 138468 (-23.7 %, Left) |
| pattern + noise | bgra | left | 184496 | 203864 (+10.5 %) | 159092 (-13.8 %) | 149084 (-19.2 %, Left) |
| pattern + noise | bgra | plane | 200688 | 230272 (+14.7 %) | 171840 (-14.4 %) | 149084 (-25.7 %, Left) |
| pattern | yuv422p | left | 51888 | 58652 (+13.0 %) | 30352 (-41.5 %) | 24704 (-52.4 %, Median) |
| pattern | yuv422p | plane | 45412 | 49524 (+9.1 %) | 25248 (-44.4 %) | 24704 (-45.6 %, Median) |
| pattern | yuv422p | median | 44576 | 44184 (-0.9 %) | 24704 (-44.6 %) | 24704 (-44.6 %, Median) |
| pattern | rgb24 | left | 70548 | 74320 (+5.3 %) | 40676 (-42.3 %) | 36728 (-47.9 %, GradientDecorr) |
| pattern | rgb24 | plane | 67984 | 72988 (+7.4 %) | 36728 (-46.0 %) | 36728 (-46.0 %, GradientDecorr) |
| pattern | bgra | left | 89748 | 93520 (+4.2 %) | 50276 (-44.0 %) | 46328 (-48.4 %, GradientDecorr) |
| pattern | bgra | plane | 87184 | 92188 (+5.7 %) | 46328 (-46.9 %) | 46328 (-46.9 %, GradientDecorr) |
| mandelbrot | yuv422p | left | 69392 | 77732 (+12.0 %) | 66604 (-4.0 %) | 50688 (-27.0 %, Gradient) |
| mandelbrot | yuv422p | plane | 58280 | 65132 (+11.8 %) | 50688 (-13.0 %) | 50688 (-13.0 %, Gradient) |
| mandelbrot | yuv422p | median | 58876 | 64644 (+9.8 %) | 52544 (-10.8 %) | 50688 (-13.9 %, Gradient) |
| mandelbrot | rgb24 | left | 108116 | 124596 (+15.2 %) | 99568 (-7.9 %) | 72500 (-32.9 %, GradientDecorr) |
| mandelbrot | rgb24 | plane | 87504 | 101188 (+15.6 %) | 72500 (-17.1 %) | 72500 (-17.1 %, GradientDecorr) |
| mandelbrot | bgra | left | 127316 | 143796 (+12.9 %) | 111740 (-12.2 %) | 82100 (-35.5 %, GradientDecorr) |
| mandelbrot | bgra | plane | 106704 | 120388 (+12.8 %) | 82100 (-23.1 %) | 82100 (-23.1 %, GradientDecorr) |

With the predictor fixed, both encoders code the same residual
symbols, so the difference is the Huffman tables alone: per-frame
optimal tables (`custom-v2`) are 4–46 % smaller, the classic tables
are 1 % smaller to 37 % larger, and letting the selector pick the
predictor (`auto`) is 13–52 % smaller.

**Stream-level `custom-v2` tables** (registry encoder; 96×64
synthetic ramp + noise, frame bytes):

| family / method | per-frame optimum | stream tables | classic tables | picture after a flat opening frame |
| --- | --- | --- | --- | --- |
| YUY2 Median | 5 792 | 5 820 (+0.5 %) | 7 768 | 8 196 (+5.5 % over classic) |
| RGB24 GradientDecorr | 10 616 | 10 648 (+0.3 %) | 13 864 | 15 104 (+8.9 %) |
| RGB32 GradientDecorr | 13 964 | 14 012 (+0.3 %) | 18 096 | 19 656 (+8.6 %) |

### Auto-selector worker scaling (round 429)

Round 429 threads the budget through `encode_frame_auto_workers`: the
per-candidate scoring (3 legal methods per family, each a full
residual + histogram + package-merge pass) fans out across the
granted workers. (The table below predates round 493; since then the
winner's emit also splits across workers on any frame large enough —
see "Round 493".) Criterion gates `encode_auto_workers_*` (sample-size 20,
Apple Silicon round machine):

| scenario | 1 worker | 2 workers | 3 workers |
| -------- | -------- | --------- | --------- |
| yuy2 1280×720 auto CustomV2 (progressive) | 6.42 ms (274 MiB/s) | 3.95 ms (445 MiB/s), **1.62×** | 3.96 ms (flat) |
| yuy2 1280×720 auto ClassicV2 (progressive) | 6.75 ms (260 MiB/s) | 3.91 ms (450 MiB/s), **1.73×** | 3.91 ms (flat) |
| rgb32 960×400 auto ClassicV2 (interlaced) | 6.29 ms (233 MiB/s) | 3.50 ms (419 MiB/s), **1.80×** | 2.93 ms (499 MiB/s), **2.15×** |

Reading the shape: at budget 2 the three candidates split 2 + 1, so
the wall is the caller's two-candidate chunk — the ~1.6–1.8× is the
scoring hiding behind the heavier chunk plus the (interlaced-only)
two-worker emit. Budget 3 is flat on progressive scenarios because
the wall becomes the single heaviest candidate (Median for YUY2)
plus the serial winner emit; on the interlaced scenario budget 3
keeps paying because the emit fans out too. Output is byte-identical
across budgets on every scenario (r429 golden-pin auto budget
sweep), and the budget-1 path is the exact pre-r429 serial selector.

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

4. **Field coding is free in the entropy loop** — it only doubles the
   prediction row length (spec/06 §2.1); nothing is split, gathered
   or reordered.

## Next PROFILE-OPT target

*(Landed r429)* The r419 primary target — a second-level full pair
LUT (16-bit window → both symbols + combined length in one 32-bit
load), built lazily behind the round-419 built-tables cache — shipped
in round 429 (`tables::build_pair_sym_lut`, wired into all three
family loops). Measured on the r419 gates: yuy2 320×240 Left
0.64 → 0.43 ms (−33%, 230 → 345 MiB/s), yuy2 320×240 Median −23%,
yuy2 1280×720 Left interlaced 7.73 → 5.31 ms (−31%), rgb24 320×240
LeftDecorr 1.26 → 0.95 ms (−25%, 174 → 232 MiB/s), rgb32 1280×720
GradientDecorr interlaced −25%. Byte-identical output (golden matrix
+ randomized LUT-vs-walk sweep).

*(Landed r440)* The r429 primary — RGB24's lone third `decode_one` —
shipped as the round-440 phase-flip pairing: the decode body (and the
split scanner) walk two pixels per step so the six codewords group as
`(w0, w1)`, `(w2, w0′)`, `(w1′, w2′)`, with three new pair-LUT combos
(`s3s1`, `s2s3`, `s3s2`) serving the flipped phases. Measured on the
gates (shared round host, same-session A/B):
`decode_rgb24_320x240_left_decorr_classic` 839 → 790 µs (−8.4%,
262 → 278 MiB/s), 320×320 interlaced −5.4%, and the new RGB24
worker gate reaches 1.35× at 2 workers (11.39 → 8.44 ms on 720p —
the family previously had no worker gate at all).

*(Landed r440)* The r419 secondary — CustomV2's dead 64-Ki
primary-LUT fill — shipped as the lean encoder tables:
`tables::canonical_entries_from_lengths` runs the canonical-assign
phase alone and the emit-side helpers take bare `[HuffEntry; 256]`
slot views. `encode_yuy2_320x240_auto_custom` 579 → 521 µs (−10.0%,
253 → 281 MiB/s), `encode_rgb24_320x240_auto_custom` −7.1%; the 720p
auto-custom scenarios are neutral (the fixed ~78 µs/frame saving
amortises over the larger raster).

*(Landed r493)* The residual fan-out listed here as unproven shipped,
together with the part it assumed impossible: the bit-emit is no
longer serial. Codewords are a pure concatenation, so each worker
packs its piece of the body into its own bit string and the strings
are spliced at their bit offsets — see "Round 493".

**Remaining candidate:** serial decode is entropy-bound; every family
consumes its whole wire cycle as LUT-fronted pair reads. Decode cannot
be split (one codeword stream, no offsets on the wire), so further
decode work should start from a fresh profile of the symbol loop.

*(Withdrawn r493)* The round-420 two-field parallel decode / encode
and its split scanner applied to a field layout no other
implementation shares; they were removed with it.

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
