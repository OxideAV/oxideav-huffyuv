# oxideav-huffyuv

Pure-Rust HuffYUV / FFVHuff lossless video codec for the
[oxideav](https://github.com/OxideAV/oxideav-workspace) framework.

## Status

**Round 7 — auto-selector residual reuse + V1xCompat table cache.**
Rounds 1 (decoder), 2 (encoder), 3 (decoder fast-LUT), 4
(interlace + lockstep), 5 (walking-stride encoder memory
optimisation), 6 (predictor RDO + single-symbol fix),
and 7 (auto-selector residual sharing + V1xCompat OnceLock cache)
all ship from the strict-isolation
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
