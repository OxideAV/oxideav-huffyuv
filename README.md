# oxideav-huffyuv

Pure-Rust HuffYUV / FFVHuff lossless video codec for the
[oxideav](https://github.com/OxideAV/oxideav-workspace) framework.

## Status

**Round 214 — macropixel-step YUY2 Huffman-decode body.** The
pre-r214 `decode_yuy2_field` loop ran `match byte_idx % 4` on every
output byte to pick the per-channel slot (Y₁/Y₂ → slot1, U → slot2,
V → slot3), so the slot-pointer reload sat on the critical path of
every Huffman lookup. Round 214 pins spec/03 §1.2's three-slot
architecture at the source by stepping four output bytes per outer
iteration with the slot resolved at compile time — the inner body
becomes a fixed straight-line `decode_one(slot1) → decode_one(slot2)
→ decode_one(slot1) → decode_one(slot3)` sequence plus four indexed
stores per 4-byte macropixel. The decode-side analogue of round 181's
LEFT macropixel-step rewrite (also branch-elimination on the same
4-byte cycle, mirrored to the Huffman-decode loop now that the LEFT
inverse already shed its `i & 3` switch). `(total_bytes - 4) % 4 ==
0` is invariant in the in-spec input space (YUY2 width is even per
the macropixel-pair invariant the decoder already checks), so the
macropixel body covers every remaining byte; a 1..=3-byte scalar
fall-through is kept for defence-in-depth against future
pixel-family extensions. Wire-identical to round 208 — six new
`round214_yuy2_decode_macropixel_tests` lock the rewrite at
byte-equality, including a `*_matches_per_byte_reference` witness
that diffs the production decode against an inlined copy of the
pre-r214 per-byte slot-dispatch body across Left / Gradient / Median
predictors. Lib test count: 164 → 170. **Round 208 — decoder
LEFT-helper consolidation.** Drops three single-use decoder-local
wrappers (`decoder::inverse_left_per_channel`,
`decoder::inverse_yuy2_left`, `decoder::inverse_yuy2_left_range`) and
re-points the YUY2 + RGB24 + RGB32 decode paths at the public
predict-side helpers (`predict::inverse_left_row` for the per-channel
stride-`n` LEFT walk; `predict::inverse_yuy2_left_macropixel` for the
YUY2 byte-position-stride LEFT walk) directly. The first was a
byte-for-byte duplicate of `inverse_left_row` (same `if out.len() <=
n { return; }` guard, same `out[i] = out[i].wrapping_add(out[i - n])`
body); the two YUY2-LEFT shims were thin pass-throughs into
`predict::inverse_yuy2_left_macropixel` already, left over from the
round-181 macropixel-step rewrite. The decoder now uses one source
of truth for both LEFT predictors — the predict-side helpers in
`predict.rs` per spec/03 §2.1 (per-channel) + §2.1.1 (YUY2). Three
new regression-guard tests (`round208_reuse_tests`) lock the
predict-side helpers against a naive scalar reference of the spec
form so a future refactor cannot silently invert LEFT differently.
Lib test count: 161 → 164. **Round 202 — YUY2 Median tail-loop
dead-branch strip on both sides of the codec.** spec/03 §2.3.2 +
audit/01 §7.2's wire-byte LEFT
exemption (`row_bytes + 8` bytes of LEFT before MEDIAN engages) means
every iteration of the post-exemption median loop satisfies
`pos >= row_bytes + 8`, so the pre-round-202 `if pos < 2 || pos <
row_bytes { continue; }` and `if pos >= row_bytes + 2 { … } else { 0 }`
arms — present in both `predict::inverse_median_post` and
`decoder::inverse_yuy2_median` — were provably dead, and
`predict::forward_median_subtract`'s `al = 0` encoder mirror was the
matching dead else-arm. The bodies become straight-line
`for pos in row1_median_start..n {}` walks with three branch-free
lookback indices (`pos - 2`, `pos - row_bytes`, `pos - row_bytes -
2`) — also dropping the `pos.wrapping_sub(2)` substitute that was
only there to silence a no-longer-needed signed-wrap concern. Lib
test count: 160 → 161 (new
`roundtrip_yuy2_median_round202_boundary_widths` sweeps nine
`(width, height)` pairs bracketing the LEFT-exemption + AL-index
boundaries — widths 2 / 4 / 6 / 8 × heights 3..8). Decode bench
(`decode_yuy2_320x240_median_classic`) lands at ≈ 1.45 ms/frame
(within noise of the pre-202 baseline — the median tail-loop was
already a small fraction of total decode time, where Huffman
bit-decode dominates; the value is correctness clarity, not
throughput delta). Round 186 — `predict::forward_rgb_left_subtract_linear(src, dst,
n_channels)` collapses the encoder's per-channel triple/quad-pass
stride-`n` LEFT-residual loop (RGB24: 3 passes with stride 3; RGB32:
4 passes with stride 4) into a single linear stride-1 walk producing
every per-channel residual in one traversal per spec/03 §2.1
(encoder evidence `@0x10001850` for RGB24-LEFT and
`@0x10001b21..@0x10001b3c` for the RGB32 byte-3 / A emit reusing the
same offset-`n`-back rule); cuts traversal count by 3× (RGB24) /
4× (RGB32) and exposes a contiguous SIMD-friendly inner subtract;
isolated LEFT-subtract microbench shows ≈ 20× speedup across
RGB24/RGB32 @ 320×240 / 1280×720 on M1 (LLVM constant-folds
`n_channels` from the `#[inline]` helper into NEON `vsubq_u8` /
SSE2 `psubb` on the interior). Round 181 (branch-free YUY2 LEFT
macropixel-step rewrite — `predict::inverse_yuy2_left_macropixel` +
`forward_yuy2_left_subtract` — replaces the per-byte `i & 3` switch
on both encoder and decoder sides with a single straight-line Y₁ /
U / Y₂ / V body per spec/03 §2.1.1; isolated-LEFT-pass microbench
shows ≈ 4.7× speedup on the inverse and ≈ 16× speedup on the
forward; round 174 added the Criterion bench harness
(`benches/{decode,encode,roundtrip}.rs`, 22 scenarios); round 134
added the cargo-fuzz harness + fixed 2 input-driven panics; round
115 factored the YUY2 forward-median pre-pass into a tested
`predict.rs` helper; round 103 fused the decorrelation+gradient
encoder residual path; round 100 added the fused
LEFT+decorrelation path.** Rounds
1 (decoder), 2 (encoder), 3 (decoder fast-LUT), 4 (interlace +
lockstep), 5 (walking-stride encoder memory optimisation), 6
(predictor RDO + single-symbol fix), 7 (auto-selector residual
sharing + V1xCompat OnceLock cache), 91 (flat overflow_entries
Vec + SWAR gradient inverse), 95 (SWAR gradient forward + drop
redundant `pixels.to_vec()` / `working.clone()` allocations), 100
(fused LEFT+decorrelation residual — no intermediate decorrelated
buffer), and 103 (fused decorrelation+gradient residual — drops
the last decorrelated-buffer allocation, on the GradientDecorr
path) all ship from the strict-isolation
clean-room workspace at
[`docs/video/huffyuv/`](https://github.com/OxideAV/docs/tree/master/video/huffyuv).
The previous (pre-orphan) implementation was retired alongside the
docs audit dated 2026-05-06; the prior history is preserved on the
`old` branch.

This crate decodes/encodes HuffYUV / FFVHuff frames given raw
codec bytes; AVI / OpenDML container handling lives one layer up
in `oxideav-avi`. End-to-end (encode → AVI mux → AVI demux →
decode) is exercised in `tests/round4_avi_lockstep.rs` via a
test-only `[dev-dependencies] oxideav-avi`.

## What works (Round 1)

- **`BITMAPINFOHEADER` + extradata parse** per `spec/01`, including
  the `biSize ∈ {0x28, > 0x28}` / low-3-bits-of-`biBitCount`
  shortcuts and the `bpp_override` byte.
- **Per-frame decode** for the 8-bit pixel families:
  YUY2 (16-bit, top-down), RGB24 (24-bit, bottom-up), RGB32 (32-bit,
  bottom-up).
- **All six legal predictor methods**: `predict_old` (`-2`), `Left`
  (`0`), `Gradient` (`1`, YUV-only), `Median` (`2`, YUV-only),
  `LeftDecorr` (`0x40`, RGB-only), `GradientDecorr` (`0x41`,
  RGB-only).
- **Both Huffman-table paths**: v2.x extradata (RLE-compressed length
  tables, canonical Huffman code build with the binary's
  longest-length-first variant) and v1.x precomputed-codes
  (`tables/06..09`).
- **Native FourCCs `HFYU` + `FFVH`** registered via `oxideav-core`'s
  codec registry so `oxideav-avi` resolves a HuffYUV stream's
  `biCompression` straight through `CodecResolver`.

## What works (Round 2)

- **Frame encoder** for the same six methods × three pixel families
  the decoder accepts (where the spec/01 §3.1 method/family allow-list
  permits — RGB doesn't carry `Median` and YUV doesn't carry the
  decorr methods, so the cross-product is 12 legal pairs).
- Three extradata paths via `ExtradataMode`:
  - **`ClassicV2`** (default) — embeds the matching pre-baked classic
    blob (the proprietary's default, spec/04 §3).
  - **`CustomV2`** — builds per-channel histograms from the residual
    stream, runs a length-limited (max-31) package-merge Huffman
    builder, and RLE-encodes the resulting length tables in line.
  - **`V1xCompat`** — emits `biSize == 0x28` with no extradata; the
    decoder reads its codebook from the v1.x precomputed-codes set.
- Public **`build_bitmapinfoheader`** BIH writer for muxers that need
  to synthesise an AVI `strf` payload.

## What works (Round 3)

- **Decoder fast-LUT** (`tables::HuffTable::primary_lut`): per-table
  65 536-entry lookup keyed on the top 16 bits of the bit window;
  codes ≤ 16 bits hit a single indexed `u16` load
  (`(length << 8) | symbol`), codes longer than 16 bits route to a
  per-symbol slow path. Measured ≈ 3.2× speed-up on a 320×240 YUY2
  LEFT/ClassicV2 frame (16.9 ms/frame → 5.3 ms/frame).
- **v1.x compat** (`ExtradataMode::V1xCompat`) now exercised against
  every legal (family, method) pair — the v1.x precomputed-codes
  set covers all 256 symbols (set A max length 17, set B max
  length 26), so `Left`, `Gradient`, `Median`, `LeftDecorr`, and
  `GradientDecorr` all round-trip via the no-extradata `biSize ==
  0x28` BIH layout that v1.x decoders expect.

## What works (Round 4)

- **Interlaced field-stride=2 prediction** (`biHeight > 288`,
  spec/02 §2 + spec/05 planned): when a frame trips the
  `is_interlaced_height` threshold the encoder splits it into two
  fields (even rows = top; odd rows = bottom), predicts each
  independently with the shared per-stream Huffman tables, and
  concatenates `top_seed | top_bits | bot_seed | bot_bits` on the
  wire. The decoder reverses the split via
  `BitReader::bytes_consumed` + `interleave_fields`. Self-roundtrip
  tests at heights 288 (boundary), 290, 300, and 320 across all
  three pixel families and four predictor methods.
- **AVI lockstep** (`tests/round4_avi_lockstep.rs`, 8 tests): the
  test-only `oxideav-avi` dev-dep wraps the codec/container
  interface end-to-end. Each test encodes a synthetic frame,
  muxes it via `oxideav_avi::muxer`, demuxes via
  `oxideav_avi::demuxer`, and decodes the resulting packet,
  asserting pixel-exact equality. Covers YUY2 / RGB24, every
  predictor, both `ClassicV2` and `CustomV2` extradata paths, the
  interlaced trigger, and a two-frame stream.

## What works (Round 5)

- **Walking-stride encoder for interlaced**: the round-4 interlaced
  path materialised two contiguous half-height field buffers
  (`predict::split_fields`) plus a `combined_residuals_for_stats`
  body for the histogram + verify passes (clone of both fields'
  bodies concatenated). Round 5 walks the source raster with
  row-stride 2 directly (`encoder::compact_field_rows`) into a
  single field-sized scratch reused across both fields, and fills
  one combined-body `Vec<u8>` in two halves
  (`combined_body[..top_body_len]` for top, `[top_body_len..]` for
  bot). The histogram + verify + emit passes consume slices of the
  combined body directly via slice-based helpers
  (`compute_lengths_from_body`, `verify_body_in_table`,
  `emit_bitstream_parts`) — no per-field `Residuals` clone, no
  per-field body Vec at emit time. Per-family slot-phase alignment
  is preserved (top body length is always a multiple of 4 / 3 / 4
  for YUY2 / RGB24 / RGB32, so the bot half resumes the histogram
  + emit slot mapping at the same `i % cycle == 0` phase).
  Encoder peak working set drops from ~3.5× source-frame size
  (round 4: split_top + split_bot + per-field
  intermediate/residuals/body + combined-stats clone) to ~2.5×
  (round 5: scratch + per-field intermediate/residuals/body +
  combined). Wire bytes are bit-identical to round 4; the
  regression is guarded by 8 new round-trip tests covering odd
  height, 480p-class, V1xCompat walking-stride, and CustomV2
  walking-stride across all three pixel families. Lib-test count:
  72 → 80.

## What works (Round 6)

- **Encoder predictor auto-selection.** New
  `encode_frame_auto(family, MethodSelection, w, h, pixels, mode)`
  runs every legal predictor for the family, scores each one with
  `bit_cost_for_method` (= `Σ length[s] × count[s]` against the
  package-merge optimal Huffman length tables — the same metric the
  `CustomV2` extradata path would minimise on the chosen method),
  and emits the winner. Returns the chosen `Method` back to the
  caller. Pin a method via `MethodSelection::Fixed(...)` to
  short-circuit. Body-identical predictors (`PredictOld` vs `Left`)
  are deduplicated in `Auto`'s candidate list; callers needing
  `PredictOld` on the wire should use `Fixed`.
- **Public `bit_cost_for_method` helper** for callers who want to
  inspect the trade-off without re-running the full encode (e.g., a
  muxer caching the cheapest predictor between similar frames).
- **`compute_canonical_lengths` single-symbol fix.** A single-symbol
  histogram now rounds out the length table with a length-1 dummy
  entry so `HuffTable::build_from_lengths`'s Kraft accumulator wraps
  to zero. Unblocks `CustomV2` + `Auto` on degenerate inputs (e.g.,
  constant-luma frames where one channel's residuals collapse to a
  single value). Dummy is never emitted (its histogram count is 0)
  and the wire bytes for any multi-symbol input are unchanged.
  Lib test count: 80 → 92.

## What works (Round 7)

- **Auto-selector residual reuse.** Round 6's
  `encode_frame_auto` did `N + 1` residual passes: one per
  candidate inside `bit_cost_for_method` + a final pass inside
  `encode_frame_with_mode` for the winner. Round 7 introduces a
  private `PrecomputedFrame` carrier + `encode_with_precomputed`
  helper so the winner's residual body bytes flow straight from
  the scoring loop into the emit pass — `N` traversals instead of
  `N + 1`. Wire-identical to round 6; regression-guarded by 5 new
  drift tests that assert `encode_frame_auto(Fixed(m))` bytes ==
  `encode_frame_with_mode(m)` bytes across every legal
  `(family, method, extradata-mode)` triple, plus an interlaced
  auto-vs-explicit drift test. **Measured**: YUY2 320×240 auto
  CustomV2 from 1.46 ms/frame → 1.37 ms/frame (≈5%); RGB24
  320×240 auto CustomV2 from 1.65 ms/frame → 1.52 ms/frame
  (≈9%).
- **V1xCompat table cache.** The v1.x precomputed-code tables
  are deterministic per pixel family (spec/04 §4.1: YUY2 = `(A,
  B, B)`; RGB24/RGB32 = `(A, A, A)`), and each [`HuffTable`]
  carries a 128 KiB primary LUT — re-baking the LUT on every
  encode call wasted ~80 µs per frame. Round 7 caches the
  three-tuple behind a per-family `OnceLock` and hands out clones
  per call. **Measured**: YUY2 320×240 V1xCompat from 0.47
  ms/frame → 0.40 ms/frame (≈16%). Per-family isolation
  verified by a regression test that interleaves YUY2/RGB24
  V1xCompat encodes and asserts the wire bytes for each family
  stay stable across the interleaving.
- Lib test count: 92 → 98.

## What works (Round 91)

- **Flat `overflow_entries` slow path** (`tables::HuffTable`):
  round 7's `decode_one_slow` walked all 256 entries of
  `table.entries` with a `length == 0 || length <= 16 { continue }`
  short-circuit on every iteration. Round 91 precomputes a flat
  `Vec<OverflowEntry>` at table-build time, holding only the
  long codes (length > 16) with their `mask` (`!0u32 << (32 - L)`)
  pre-baked. The slow path now iterates exactly that slice (≤ 2
  entries for most v2.x classic blobs, ~210 for v1.x set B —
  always strictly less than 256). **Measured (release, M1 host,
  320×240)**: v1.x median decode 2.81 ms/frame → 1.29 ms/frame
  (**≈ 2.2× speedup**); ClassicV2 gradient YUY2 1.72 ms/frame →
  0.67 ms/frame (**≈ 2.6× speedup**). Wire-identical to round 7;
  the slow path's output bytes are unchanged.
- **SWAR `inverse_gradient_post`** (`predict::inverse_gradient_post`):
  the round-7 byte-by-byte modular-add loop is replaced by an
  8-bytes-per-u64 SWAR add (`(a & 0x7F7F…).wrapping_add(b &
  0x7F7F…) ^ ((a ^ b) & 0x8080…)` — the textbook byte-wise
  wrapping add via 64-bit math, no `unsafe`, no
  vendor-intrinsics dependency). Mirrors spec/03 §2.2.2's
  documented MMX 8-byte-wide post-pass
  (`@0x10001dfb`–`@0x10001e8c`: 8-byte SIMD load + lane-wise
  wrap-add + store + 32-byte unrolled stride). LLVM
  autovectorises the inner u64 loop into SSE2 `paddb` on x86_64
  and NEON `vaddq_u8` on aarch64. Bit-identical to the round-7
  byte loop (regression-guarded by `round91_swar_gradient_*`
  tests covering aligned-8, unaligned-tail, modular-wrap, and
  height-1 no-op cases).
- Lib test count: 98 → 110 (+12 round-91 tests covering the
  overflow-entries shape, the SWAR equivalence, and end-to-end
  YUY2/RGB24/RGB32 gradient round-trips at 320×16 to exercise
  the new u64 path on realistic row widths).

## What works (Round 95)

- **`predict::forward_gradient_subtract`** — encoder analogue of
  round 91's `inverse_gradient_post`. Produces the
  LEFT-predict-row-0 + per-row pixel-minus-pixel-above layout
  documented at spec/03 §2.2.2 `@0x10001eab..@0x10001f9e` via a
  chunked u64 SWAR `psubb`-style subtract
  (`(a | 0x80…).wrapping_sub(b & 0x7F…) ^ ((a ^ !b) & 0x80…)`)
  — byte-wise wrapping subtract with no inter-byte borrow, no
  `unsafe`, no vendor intrinsics. LLVM autovectorises the inner
  u64 loop into SSE2 `psubb` on x86_64 and NEON `vsubq_u8` on
  aarch64.
- **Drop the `intermediate` and `working` clones** on
  non-gradient / non-decorrelating encoder paths. Round 6's
  `yuy2_residuals` / `rgb24_residuals` / `rgb32_residuals` each
  allocated a `row_bytes × height` `intermediate: Vec<u8>` per
  frame — even when the predictor was Left / Median / PredictOld
  (= a pure `pixels.to_vec()` clone of the input) — plus an
  extra `working.clone()` on RGB methods that didn't need
  decorrelation. Round 95 borrows `pixels` / `working` directly
  through `Option<Vec<u8>>` + `as_deref().unwrap_or(...)` and
  only allocates the intermediate when the gradient pre-pass
  needs one. Saves up to 2 × row_bytes × h bytes per frame on
  YUY2 Left / Median / PredictOld and 1 × row_bytes × h bytes on
  RGB Left / LeftDecorr / GradientDecorr.
- **Measured (release, M1 host, 500 iters)**:
  YUY2 320×240 Median 1.10 ms/frame → 0.64 ms/frame
  (≈ **1.7× speedup**); YUY2 1280×720 Left 7.65 ms/frame →
  5.09 ms/frame (≈ **1.5× speedup**); RGB24 320×240 Left
  1.00 ms/frame → 0.85 ms/frame (≈ 15%); RGB24 320×240
  LeftDecorr 1.00 ms/frame → 0.86 ms/frame (≈ 13%); YUY2
  320×240 gradient 0.52 ms/frame → 0.46 ms/frame (≈ 10%).
- Wire-identical to round 91; lib test count: 110 → 115 (+5
  `round95_swar_subtract_*` equivalence tests covering
  aligned-8, unaligned-tail, modular-wrap, height-1 no-op, and
  forward-then-inverse round-trip).

## What works (Round 100)

- **`predict::forward_left_decorr_residuals`** — fused
  LEFT+decorrelation encoder residual path per spec/03 §2.4.1
  ("the decorrelation transform is fused with the predictor at
  the residual computation, not applied as a separate pre-pass …
  there is no intermediate decorrelated buffer"; encoder evidence
  `@0x1000198e..@0x10001996`, the four-instruction fused chain
  `(cur.B − cur.G) − (prev.B − prev.G)`). Computes the
  decorrelated-LEFT residuals (`G` identity, `B−G`, `R−G`, plus
  alpha LEFT-but-NOT-decorrelated per the §2.4 Validator note)
  directly from the caller's un-transformed `pixels` in a single
  pass, then the body builder reads them in wire order.
- **Allocation elimination on the LeftDecorr path.** Round 95's
  `rgb24_residuals` / `rgb32_residuals` still materialised a
  full-frame `working_owned: Vec<u8>` (the decorrelated buffer,
  `row_bytes × h` bytes per frame) for *every* decorrelating
  method, then ran a second per-channel LEFT-subtract pass over
  it. Round 100 routes the **LeftDecorr** method (`0x40`,
  decorrelate + no gradient) through the fused helper, skipping
  the `working_owned` allocation entirely. The **GradientDecorr**
  method (`0x41`) still materialises `working` because its
  gradient pre-pass reads the decorrelated buffer as input — that
  path is unchanged.
- Wire-identical to round 95 — the fused residuals are
  byte-for-byte equal to the round-95 two-pass
  "materialise-then-subtract" output (regression-guarded by
  `round100_fused_decorr_matches_two_pass_rgb24/rgb32`), and
  every pre-existing LeftDecorr round-trip + the
  `lockstep_rgb24_left_decorr_classic` AVI-lockstep test stays
  green. Lib test count: 115 → 126 (+5 fused-equivalence /
  modular-wrap / alpha / inverse-round-trip unit tests in
  `predict.rs`, +6 end-to-end LeftDecorr round-trips in
  `roundtrip_tests.rs` across RGB24/RGB32 × ClassicV2 / CustomV2
  / V1xCompat, an interlaced-height-300 LeftDecorr frame, and a
  per-row-varying-alpha RGB32 frame).

## What works (Round 103)

- **`predict::forward_decorr_gradient_subtract`** — fused
  decorrelation+gradient encoder pre-pass for method `0x41`
  (GradientDecorr), the §2.4.2 counterpart to round 100's
  LeftDecorr fusion. Per spec/03 §2.4.2 the gradient+decorrelation
  residual is `c_dec[i] − gradient(c_dec[i−1], c_dec_above[i],
  c_dec_above_left[i])` per decorrelated channel, which §2.2.2
  decomposes into a LEFT-pass over the row-above differences plus
  the per-channel LEFT-subtract series. The §2.2.2 gradient
  pre-pass reads only **decorrelated** channel values, so this
  helper folds the per-pixel decorrelation (`B−G`, `G` identity,
  `R−G`, and — RGB32 — alpha identity / NOT decorrelated per the
  §2.4 Validator note) straight into the same-column gradient
  subtract, reading un-transformed `pixels` directly. Row 0 is the
  decorrelated values verbatim (the §2.2.1 first-row LEFT
  exemption); rows ≥ 1 are the same-column decorrelated subtract.
- **Last decorrelated-buffer allocation eliminated.** Round 100
  fused LeftDecorr (`0x40`) but GradientDecorr (`0x41`) still
  materialised the full-frame `working_owned: Vec<u8>`
  (`row_bytes × h` bytes/frame) because its gradient pre-pass read
  the decorrelated buffer as input. Round 103 routes GradientDecorr
  through the fused helper, skipping that allocation. With both
  decorrelation paths now fused, `rgb24_residuals` /
  `rgb32_residuals` never allocate a decorrelated buffer anymore —
  only the gradient `intermediate` (which Gradient / GradientDecorr
  already required).
- Wire-identical to round 100 — the fused gradient pre-pass output
  is byte-for-byte equal to the round-95/100 two-pass
  "materialise the decorrelated buffer, then forward-gradient
  subtract" output (regression-guarded by
  `round103_fused_decorr_gradient_matches_two_pass_rgb24/rgb32`),
  and every pre-existing GradientDecorr round-trip + the
  AVI-lockstep tests stay green. Lib test count: 126 → 135 (+5
  fused-equivalence / modular-wrap / alpha-identity / height-1
  unit tests in `predict.rs`, +4 end-to-end GradientDecorr
  round-trips in `roundtrip_tests.rs`: RGB24 7×5 CustomV2
  non-aligned width, RGB32 6×4 V1xCompat, a per-row-varying-alpha
  RGB32 frame, and an RGB24 interlaced-height-300 frame).

## What works (Round 115)

- **`predict::forward_median_subtract`** — the YUY2 forward MEDIAN
  pre-pass factored out of the encoder into a dedicated, tested
  `predict.rs` helper (the encoder analogue of `inverse_median_post`,
  matching how round 95 factored `forward_gradient_subtract` and
  rounds 100/103 factored the decorrelation forwards). Produces the
  complete YUY2 median residual stream in a **single pass** per
  spec/03 §2.3 / §2.3.2: LEFT residuals for row 0 and the first 8 wire
  bytes of row 1 (the §2.3.2 MMX-8-byte first-second-row exemption),
  and MEDIAN residuals (`pixel − median3(L, A, G)` with the §2.3
  output offsets `−2 / −row_stride / −row_stride − 2`) for the rest.
- **Wasted-LEFT-recompute eliminated.** The previous `yuy2_residuals`
  median path ran a full-frame LEFT subtract over the entire frame and
  then **overwrote** the median region with median residuals —
  computing the median region's LEFT residuals only to throw them
  away. Round 115 computes LEFT only for the exempt region and median
  directly for the rest, in one traversal.
- Wire-identical to round 103 — the single-pass output is
  byte-for-byte equal to the prior two-phase
  "full-LEFT-then-overwrite" output (regression-guarded by
  `round115_forward_median_matches_two_phase` /
  `round115_forward_median_modular_wrap`), the height-1 and
  short-second-row edge cases stay all-LEFT, a decoder-model
  round-trip reconstructs the source exactly, and every pre-existing
  YUY2 median round-trip + the `lockstep_yuy2_median_classic`
  AVI-lockstep test stays green. Lib test count: 135 → 140 (+5
  round-115 tests in `predict.rs`).

## What works (Round 134)

- **Fuzz harness** (`fuzz/fuzz_targets/decode_huffyuv.rs`) — a
  cargo-fuzz target driving the full decode chain
  (`parse_bitmapinfoheader` → `decode_frame`) on arbitrary bytes,
  framed `[u16-LE strf-len][strf][frame body]` so one mutation can hit
  either the header/Huffman-table config or the per-frame body.
  Declared rasters above a 16 MiB cap are skipped in the harness (an
  expected resource request, not a logic bug). Seeded with six
  encoder-emitted valid streams plus the fuzz-found crash regressions;
  daily CI via `.github/workflows/fuzz.yml`. 60s baseline post-fix:
  ~21.7k execs, zero crashes / OOMs / timeouts.
- **Two input-driven panics fixed**, both now clean `Err`:
  - Zero-width/height frames whose output raster was smaller than the
    uncompressed seed pixel panicked on the seed write in
    `decode_{yuy2,rgb24,rgb32}_field`; each now rejects a too-small
    raster, and the interlaced bottom-field split point is clamped
    against the buffer length.
  - A header declaring `biSize > 0x29` on the `low3 != 0` method path
    (which skips the `biSize <= len` check) read the bpp-override byte
    at `+0x29` out of bounds; the read now gates on the real buffer
    length too.
  - 6 regression tests added (5 in `decoder.rs`, 1 in `header.rs`). Lib
    test count: 140 → 146.

## What works (Round 174)

- **Criterion bench harness** (`benches/decode.rs`, `benches/encode.rs`,
  `benches/roundtrip.rs`) covering 22 representative
  `(family, method, extradata-mode, raster)` scenarios — chosen to
  mirror the README's "measured" headline numbers from rounds 3, 7,
  91, 95, 100, 103, and 115 so future optimisation rounds can read a
  regression directly off the criterion delta. Inputs synthesised on
  the fly from a deterministic xorshift32 + diagonal-gradient pattern
  (no committed binary fixtures, no `docs/` dependency, no
  third-party samples). Wired via `[dev-dependencies] criterion =
  "0.5"` + three `[[bench]] harness = false` entries; the lib test
  surface (146 tests) is unchanged.
- **Coverage**:
  - **decode** (7): YUY2 320×240 LEFT/Gradient/Median ClassicV2,
    YUY2 1280×720 LEFT ClassicV2, YUY2 320×240 LEFT V1xCompat, RGB24
    320×240 LeftDecorr ClassicV2, RGB32 320×240 GradientDecorr
    ClassicV2.
  - **encode** (10): eight fixed-method ClassicV2 / V1xCompat
    scenarios that the README headline numbers covered, plus two
    `MethodSelection::Auto` + CustomV2 scenarios (YUY2 / RGB24).
  - **roundtrip** (5): end-to-end encode → parse-BIH → decode
    pipeline health checks.
- **Local `--quick` smoke (M1, warm-up 1 s, measurement 2 s)**: all
  22 benches execute cleanly; representative throughputs ~100-110
  MiB/s on decode and ~240-325 MiB/s on encode at 320×240 / 1280×720.
  The numbers aren't a regression baseline yet (README headlines
  came from 500-iter `cargo bench` runs, not Criterion `--quick`),
  but `cargo bench -p oxideav-huffyuv` now produces directly
  comparable per-scenario figures for the next optimisation round.

## What works (Round 181)

- **`predict::inverse_yuy2_left_macropixel(out, begin, end)`** —
  decoder YUY2 LEFT inverse rewritten as a branch-free
  macropixel-step body per spec/03 §2.1.1
  (`@0x100020f4..@0x1000210e`: `Y₁ ← prev_Y₂`, `prev_Y₂ ← Y₁`,
  `U ← prev_U`, `prev_U ← U`, `Y₂ ← Y₁` (intra-pair), `prev_Y₂
  ← Y₂`, `V ← prev_V`, `prev_V ← V`). Three rolling channel
  accumulators (`prev_y` / `prev_u` / `prev_v`) replace the prior
  per-iteration `match i & 3` switch on the lookback stride; the
  inner loop is now a straight-line 8-add / 4-store sequence
  advancing one 4-byte macropixel per step. A 1-3 byte scalar tail
  preserves the prior per-byte semantics for ranges whose end is
  not macropixel-aligned (in practice never triggered — YUY2 row
  widths are always multiples of 4).
- **`predict::forward_yuy2_left_subtract(src, dst)`** — encoder
  analogue: same macropixel-step body, but with `src[i+2] − src[i]`
  expressing the §2.1.1 Y₂-from-same-pair-Y₁ intra-pair LEFT rule
  directly off the read-only `src` slice (no in-place dependency
  chain). LLVM autovectorises the four-channel body into NEON
  `vsubq_u8` on aarch64 and SSE2 `psubb` on x86_64.
- **Wire-identical** to round 174 — every pre-existing YUY2
  round-trip test, the `lockstep_yuy2_*` AVI-lockstep tests, and
  6 new equivalence + 1 forward-then-inverse round-trip tests in
  `predict.rs` all stay green. The equivalence tests diff the new
  helpers against in-test copies of the prior per-byte-branch
  loops across raster widths 2 / 4 / 320 / 640, the modular-wrap
  edge case, the median row-1-first-8 range, and small-buffer
  no-ops.
- **Measured (release, M1, single-threaded isolated LEFT pass —
  no Huffman)**:
  - Inverse YUY2 LEFT @ 320×240: ~148 µs/frame → ~31 µs/frame
    (**≈ 4.7× speedup**).
  - Inverse YUY2 LEFT @ 1280×720: ~1.79 ms/frame → ~0.37 ms/frame
    (**≈ 4.8× speedup**).
  - Forward YUY2 LEFT @ 320×240: ~40 µs/frame → ~2.4 µs/frame
    (**≈ 16× speedup**).
  - Forward YUY2 LEFT @ 1280×720: ~490 µs/frame → ~30 µs/frame
    (**≈ 16× speedup**).
  - End-to-end criterion delta on the YUY2-LEFT 320×240 decode
    bench is a more modest 1-2% (Huffman bit-decode still
    dominates total frame time; the LEFT pass was already a small
    fraction of total decode).
- Lib test count: 146 → 153 (+7 round-181 tests in `predict.rs`).

## What works (Round 186)

- **`predict::forward_rgb_left_subtract_linear(src, dst, n_channels)`**
  — single linear stride-1 walk producing the per-channel LEFT
  residuals for RGB24 (`n = 3`) and RGB32 (`n = 4`) buffers.
  Replaces the encoder's pre-round-186 per-channel
  triple/quad-pass stride-`n` loop
  (`for ch in 0..N { idx = ch + N; while idx < len { residuals[idx]
  = pred_input[idx] − pred_input[idx − N]; idx += N; } }`) with a
  single contiguous `for i in N..len { dst[i] = src[i] −
  src[i − N]; }` pass. The per-channel residual identity holds for
  every output position because `i` taken mod `n_channels` selects
  the channel, so the same `src[i] − src[i − n_channels]`
  expression computes the correct per-channel residual at every
  byte without an explicit channel split. Per spec/03 §2.1 (encoder
  evidence `@0x10001850` for the RGB24-LEFT byte-B emit + spec/03
  §1.2 / §2.1 evidence `@0x10001b21..@0x10001b3c` for the RGB32
  byte-3 / A emit reusing the same offset-`n`-back rule).
- **Allocation + pass-count reduction.** The pre-round-186 encoder
  re-traversed the entire residual buffer `n_channels` times (once
  per channel) and produced strided reads/writes that LLVM cannot
  fuse into vector instructions. The linear walk traverses once
  (`n` × fewer cache-line loads) and the inner subtract is a
  `src[i] − src[i − n]` over consecutive bytes, which LLVM
  autovectorises into NEON `vsubq_u8` on aarch64 and SSE2 `psubb`
  on x86_64. The helper is `#[inline]` so the `n_channels`
  argument constant-folds at the encoder call sites (both pass the
  literal `3` or `4`), letting the compiler specialise the inner
  body for each width.
- **Measured (release, M1, single-threaded isolated LEFT-subtract
  microbench — `src.len()` bytes / `iters = 2000` for 320×240 /
  200 for 1280×720, no Huffman)**:
  - RGB24 320×240 LEFT: ~86.7 µs/frame → ~4.3 µs/frame
    (**≈ 20× speedup**).
  - RGB24 1280×720 LEFT: ~1036 µs/frame → ~51.7 µs/frame
    (**≈ 20× speedup**).
  - RGB32 320×240 LEFT: ~114.9 µs/frame → ~5.8 µs/frame
    (**≈ 20× speedup**).
  - RGB32 1280×720 LEFT: ~1380 µs/frame → ~69.6 µs/frame
    (**≈ 20× speedup**).
  - End-to-end criterion delta on the encode benches will be more
    modest (Huffman bit-encode dominates total frame time; the
    LEFT-subtract pass was already a small fraction of total
    encode), but the LEFT-pass itself was the dominant non-Huffman
    cost on the non-gradient / non-decorrelated RGB encoder paths.
- **Wire-identical to round 181** — every per-channel residual byte
  the linear walk produces equals the byte the prior triple/quad-pass
  loop produced (regression-guarded by
  `round186_rgb_left_linear_matches_per_channel_rgb24` /
  `_rgb32` covering widths 1 / 4 / 320 and heights 1 / 4,
  `round186_rgb_left_linear_modular_wrap` for the 0xFF/0x01
  alternator that forces every byte to mod-256 wrap,
  `round186_rgb_left_linear_short_buffer_seed_only` for the
  smaller-than-one-pixel degenerate case, and
  `round186_rgb_left_linear_then_inverse_roundtrips` end-to-end via
  the decoder's per-channel `wrapping_add` inverse). Every
  pre-existing RGB24 / RGB32 LEFT / Gradient / LeftDecorr /
  GradientDecorr round-trip + the `lockstep_rgb24_*` AVI-lockstep
  tests stay green. Lib test count: 153 → 158.

## What works (Round 202)

- **YUY2 Median tail-loop dead-branch strip.** Both inverses
  (`predict::inverse_median_post`, `decoder::inverse_yuy2_median`)
  and the encoder analogue (`predict::forward_median_subtract`)
  shed the three intra-loop dead branches they had carried since
  round 1:
  - `if pos < 2 || pos < row_bytes { pos += 1; continue; }` — never
    triggered because the loop starts at `row1_median_start =
    (row_bytes + 8).min(n)`, so `pos >= row_bytes + 8 >= row_bytes
    >= 2` on every iteration.
  - `let al = if pos >= row_bytes + 2 { out[pos - row_bytes - 2] }
    else { 0 };` — the `else { 0 }` arm is unreachable for the same
    reason (`pos - row_bytes >= 8 >= 2`).
  - `let l = out[pos.wrapping_sub(2)];` — the `wrapping_sub`
    substitute was only there to keep the index expression
    well-formed under the now-dropped `pos < 2` reachability.
    Replaced with the plain non-wrapping `out[pos - 2]`.
- **Wire-identical to round 196.** spec/03 §2.3.2 + audit/01 §7.2's
  wire-byte LEFT-exemption invariants (`row1_median_start >= row_bytes
  + 8`, hence `>= row_bytes + 2`) anchor every dropped branch as
  provably dead; the three lookback indices in the new straight-line
  body (`out[pos - 2]`, `out[pos - row_bytes]`, `out[pos - row_bytes
  - 2]`) are all in-bounds for every iteration. `debug_assert!`s at
  function entry pin the row-stride invariant so future refactors
  can't silently violate it. The branch-free body is a small win on
  the median tail-loop itself, but the value is correctness
  clarity — the prior form's wrap-arithmetic + multi-arm `if` was
  carrying state the spec rules out.
- **Boundary regression coverage.** New test
  `roundtrip_yuy2_median_round202_boundary_widths` sweeps nine
  `(width, height)` pairs that bracket the LEFT-exemption +
  AL-index boundaries:
  - Width 2 (`row_bytes = 4 < 8` — the narrow case the round-196
    wire-asymmetry fix targeted): heights 4, 5, 6, 8 — confirming
    the post-202 loop still wire-roundtrips when the LEFT
    exemption extends past the row-1 boundary.
  - Width 4 (`row_bytes = 8`, exemption ends exactly at row-2
    start): heights 3, 4, 8 — exercising the AL-lookback into row
    0 from the first median pos in row 2.
  - Widths 6, 8 × height 4 — exemption sits mid-row 1, median
    region starts later within row 1.
- **Lib test count: 160 → 161.** Bench delta (320×240 YUY2 Median
  ClassicV2 decode) is within Criterion noise of the pre-202
  baseline (≈ 1.45 ms/frame; the median tail-loop accounted for a
  small fraction of total decode time because the Huffman bit-decode
  dominates the cost on the ClassicV2 path).

## What works (Round 214)

- **`decode_yuy2_field` macropixel-step Huffman-decode body.** The
  pre-r214 loop ran `match byte_idx % 4` on every output byte to pick
  the per-channel slot — for a 1920×1080 YUY2 frame that's ~4M
  per-byte branches with a `&tables.slotN` reload on the critical
  path of every `decode_one` call. Round 214 hoists the spec/03 §1.2
  three-slot wire-byte pattern out of the loop entirely by stepping
  four output bytes per outer iteration with the slot resolved at
  compile time. The inner body is now a fixed straight-line
  4-decode / 4-store sequence (`decode_one(slot1) → decode_one(slot2)
  → decode_one(slot1) → decode_one(slot3)`) the optimiser can
  schedule freely. spec/03 §1.2 invariants used: byte +0 (Y₁) → slot1,
  byte +1 (U) → slot2, byte +2 (Y₂) → slot1, byte +3 (V) → slot3.
  Decode-side mirror of round 181's LEFT macropixel-step rewrite.
- **Wire-identical to round 208** — the new step body produces the
  identical pre-predictor sample stream for every YUY2 input that
  `(total_bytes - 4) % 4 == 0` covers (the in-spec input space; YUY2
  width is even per the §2.1.1 macropixel-pair invariant the decoder
  already checks). A 1..=3-byte scalar fall-through preserves the
  prior per-byte semantics for any future layout that lands the body
  on a non-macropixel boundary. Six new
  `round214_yuy2_decode_macropixel_tests` cover (a) round-trip at
  widths 2 / 4 / 8 / 16 bracketing the macropixel-step boundary, (b)
  the v1.x compat path where slot1/slot2/slot3 hold distinct tables
  (slot mix-up would surface as a Huffman-table mismatch before the
  predictor pass), and (c) a `*_matches_per_byte_reference` witness
  that diffs the production decode against an inlined copy of the
  pre-r214 per-byte slot-dispatch loop across Left / Gradient /
  Median predictors. Lib test count: 164 → 170.

## Out of scope (deferred)

- 10/12-bit FFVHuff family — explicitly excluded from the
  clean-room workspace (`docs/video/huffyuv/README.md` "Scope":
  "FFVHuff is **out of scope** for this workspace; only the
  original `HFYU` FOURCC HuffYUV from Rudiak-Gould's binary
  distribution is targeted here").
- Third-party-fixture lockstep — the host
  `samples.oxideav.org` carried no `HFYU` / `FFVH` fixtures at
  round-4 time, so the lockstep test runs against synthetic
  encoder output. When fixtures land we expect to add a
  `samples_lockstep_*` test that consumes them via the
  `oxideav-avi` dev-dep without any change to this crate's
  source layout.
- The interlaced wire format follows the spec/02 §2 directive
  ("two fields each of height H/2, predicted independently") but
  spec/05 — the canonical interlace chapter — is still flagged
  "planned" in the docs workspace. The implementation is
  self-roundtrip-correct; lockstep against a third-party
  interlaced HuffYUV encoder will need spec/05 + a fixture before
  it can be claimed bit-exact.

## Cargo features

- **`registry`** (default): wire the crate into `oxideav-core`'s
  codec registry. `default-features = false` builds the decoder
  standalone with no `oxideav-core` dependency.
