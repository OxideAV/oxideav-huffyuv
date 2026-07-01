# Changelog

All notable changes to this crate are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Round 382: decoder-contract conformance suite for spec/04 §3.4 —
  the decoder MUST accept a v2.x extradata stream carrying **any**
  three RLE-compressed 256-entry length tables, "including but not
  limited to the six classic blobs", deriving codes by canonical
  Huffman construction (spec/03 §3) rather than classic-blob lookup.
  Eight new tests hand-build a deliberately non-classic length table
  (a flat all-length-8 code — Kraft sum `256 × 2⁻⁸ = 1`, byte-distinct
  from every classic blob), RLE-compress it into extradata, encode a
  genuine predicted-residual frame body against that exact table via
  `emit_bitstream_parts`, and prove the decoder reconstructs the
  source pixels bit-exactly. Coverage spans all seven legal
  `(family, method)` residual paths (YUY2 Left/Gradient/Median,
  RGB24 Left/LeftDecorr, RGB32 LeftDecorr/GradientDecorr) plus a tall
  (`biHeight = 300 > 288`) YUY2 interlaced frame that exercises the
  arbitrary tables through both field halves. A `assert_not_classic`
  guard confirms the emitted extradata differs from all six classic
  blobs, so a classic-blob-locked decoder would fail the suite. No
  wire-format change; the prior 273 lib + 8 lockstep tests stay green
  (281 lib total).

- Round 335: direct unit coverage for the interlace field-stitching
  primitives `predict::split_fields` / `predict::interleave_fields`
  (spec/02 §2). Three new tests harden invariants previously exercised
  only indirectly through the end-to-end interlaced decode: (1) a
  single-row frame puts its lone row in the even-parity top field with
  an empty bottom field and round-trips; (2) the documented degenerate
  guards (`row_bytes == 0` or `height == 0`, reachable from the decoder
  when `StreamConfig::row_bytes()` saturates to 0 on a zero-width
  stream) return empty field buffers / empty raster rather than
  indexing out of bounds; (3) a `(row_bytes, height)` sweep over both
  parities and a wide stride asserts the spec row distribution
  (`top = ceil(h/2)` even rows, `bot = floor(h/2)` odd rows, each row
  placed byte-for-byte) and the `interleave∘split == identity`
  contract. No wire-format change; all prior decode / roundtrip /
  AVI-lockstep tests stay green (273 lib + 8 lockstep).

- Round 322: the extradata `interlace_flag` byte at BIH `+0x2A`
  (spec/01 §3 + audit/01-validation-report.md §7.5) is now parsed,
  exposed on `StreamConfig::interlace_flag`, and honoured as the
  **primary** interlaced indicator. The i386 build's encoder writes
  this byte (`0x10` interlaced / `0x20` non-interlaced via a height-≤-288
  test) and its decoder reads the **high nibble** back, dispatching on
  `1` (interlaced) / `2` (non-interlaced) and falling back to the
  `biHeight > 288` heuristic only when the byte is `0x00` (the x86-64
  build / clean-room default). `StreamConfig::is_interlaced` now defers
  to the new `predict::interlaced_from_flag_and_height`, so a stream
  carrying an explicit flag is no longer forced through the height
  heuristic — an `0x10` flag on a short frame decodes interlaced and an
  `0x20` flag on a tall (> 288) frame decodes progressive. The v2.x
  `build_bitmapinfoheader` writer now emits the i386-style flag (via the
  new `predict::interlace_flag_for_height`) instead of the prior
  hard-coded `0x00`, so our extradata round-trips bug-for-bug with the
  i386 decoder's primary indicator while still being decoded correctly
  by any decoder that only reads the byte's high nibble. Both new
  helpers are unit-tested for the full decode table (high-nibble
  dispatch, low-nibble-ignored, unrecognised-nibble fallback) and
  encode/decode round-trip; the BIH writer + parser path is covered
  end-to-end (encode → parse → decode) for both a short and a tall
  frame. Wire-identical for the residual bitstream (the flag lives only
  in the extradata prefix); all prior decode/roundtrip + AVI-lockstep
  tests stay green.

### Changed

- Round 310: refilling 64-bit bit reader in `bitio::BitReader`. The
  pre-r310 reader recomputed the 32-bit decode window from scratch on
  every `peek_window` call — it indexed back into the source slice,
  reconstructed up to two 32-bit LE words byte-by-byte through a
  bounds-checked `[0u8; 4]` loop, and stitched them with a shift. On
  the per-symbol decode critical path (one `peek_window` +
  `consume_bits` per Huffman codeword, four per YUY2 macropixel) that
  was two word reconstructions × four bounds-checked byte loads for
  every symbol — the flat ~100–120 MiB/s decode ceiling the
  `BENCHMARKS.md` hotspot table flagged as bit-read-bound (uniform
  across all six predictors ⇒ the bit read, not the `inverse_*_post`
  predictor pass, governs decode throughput). Round 310 keeps a 64-bit
  MSB-aligned accumulator holding the already-loaded leading stream
  bits: `peek_window` serves `(acc >> 32) as u32` with no byte access,
  and `consume_bits` only advances a bit counter and tops the
  accumulator back up to ≥ 32 valid bits by pulling whole 32-bit LE
  words from the source (each word read exactly once over the field
  decode, amortised one bounds-checked word fetch per ~32 consumed
  bits vs. the prior two-words-per-symbol). The `read_word_le` fast
  path takes a single `data.get(base..base + 4)` →
  `u32::from_le_bytes` load when the window is in range, zero-padding
  only the truncated final word per spec/02 §4 ("final partial word …
  trailing bits unspecified"). The accumulator is MSB-aligned so the
  served window is byte-for-byte the value the pre-r310 reconstruction
  produced (top bit = next-unread bit, zero pad past end-of-stream),
  and `cursor_bits` / `bytes_consumed` are unchanged — the interlaced
  field-split that reads `bytes_consumed` to find the next field's seed
  is unaffected. Wire-identical (every one of the 259 pre-r310
  decode/roundtrip tests stays green). Measured (release, aarch64,
  Criterion 0.5, controlled back-to-back, `--measurement-time 3
  --sample-size 20`): decode_yuy2_320x240_left_classic 1.18 ms → 0.68
  ms (**≈ 43% faster**, ~115 → ~217 MiB/s); decode_yuy2_1280x720_left
  13.81 ms → 7.44 ms (**≈ 46% faster**); decode_rgb24_320x240_left_decorr
  1.95 ms → 1.26 ms (**≈ 35% faster**). Five new
  `round310_*` tests in `bitio.rs` lock the rewrite: two window-
  equivalence witnesses (uniform step across 13 stream lengths × 10
  consume strides, and a 4000-iteration variable-step walk with
  Huffman-realistic 1..=32-bit advances) diff the refilling
  `peek_window` against an inlined copy of the pre-r310 from-scratch
  reconstruction at every cursor — including a 96-bit over-read past
  end-of-data so the zero-padded truncated-final-word window is
  covered; plus an out-of-range `consume_bits` rejection (cursor +
  window unperturbed), a `bytes_consumed` word-rounding fixture, and
  an empty-stream zero-window fixture. Lib test count: 259 → 264.

- Round 304: `predict::inverse_rgb_decorr_bgr` /
  `inverse_rgb_decorr_bgra` (the RGB-decorr decode post-pass that
  reconstructs `B = (B−G) + G` / `R = (R−G) + G` from the
  per-pixel G) rewritten from an index-arithmetic
  `while i + n <= out.len()` loop into `chunks_exact_mut(3)` /
  `chunks_exact_mut(4)`. Per-pixel reconstruction is independent
  (no cross-pixel carry — the LEFT / gradient inverse already
  ran), so a fixed-size pixel window per iteration lets the
  compiler collapse the per-access bounds checks to the iterator's
  single length-aligned stride and autovectorise the strided
  wrapping-add. This was the last inverse predictor body still
  running a naive scalar per-pixel loop (LEFT macropixel r181/r208,
  gradient SWAR r91, median r255 already optimised). Wire-identical
  to round 277 — five new `round304_inverse_decorr_*` tests lock
  the rewrite: two `*_matches_pre_r304_reference` witnesses diff
  the production output against an inlined copy of the pre-r304
  index loop across byte counts 0..=64 (covering partial trailing
  pixels where `len % stride != 0`), a modular-wrap fixture, an
  alpha-pass-through fixture, and a truncated-buffer remainder
  fixture. Lib test count: 254 → 259.

### Fixed

- Fuzz-found (scheduled `tables_huffyuv` run 2026-06-09,
  crash-8fd2645b…): `HuffTable::build_from_lengths` validated
  Kraft equality by requiring the 32-bit MSB-aligned code
  accumulator to wrap to exactly 0 — but a length distribution
  whose Kraft sum is an exact integer multiple of 1.0 (e.g. four
  2-bit lengths + two 1-bit lengths = 2.0) wraps the accumulator
  to 0 more than once and aliased the check, so the build
  accepted the table and assigned two symbols the same all-zero
  code (symbol 0 at length 2 and symbol 18 at length 1), breaking
  the prefix-free decode contract (`decode_one` on symbol 0's own
  code window returned symbol 18). The build now tallies the
  Kraft sum exactly in 64-bit fixed point (unit 2⁻³²; a length-L
  code contributes 2³²⁻ᴸ) and rejects any non-empty table whose
  tally ≠ 2³², per the spec/03 §3 `sum 2^-L_i == 1` equality
  invariant. Strictly tightening: every previously-accepted valid
  table (sum exactly 1.0) still builds; under-subscribed and
  fractionally over-subscribed tables were already rejected by
  the old check. New
  `build_from_lengths_rejects_integer_kraft_oversubscription`
  regression test pins the minimised fuzz shape (sum 2.0), a sum
  3.0 variant, and a Kraft-equal control whose every entry still
  self-consistently decodes. Lib test count: 244 → 245. No
  reachable wire-path change: v2.x extradata streams with a
  Kraft-violating length table previously built a corrupt table
  and mis-decoded; they now error out cleanly at table-build
  time.

### Added

- Round-286 (bench-only; `src/` byte-identical): extended the
  Criterion suite for full predictor + interlaced + family
  coverage and added `BENCHMARKS.md` with a ranked hotspot table
  naming the next PROFILE-OPT target. `benches/decode.rs` now
  covers all six on-wire methods (adds `predict_old`) plus a
  320×288-progressive / 320×320-interlaced Left pair (and an
  RGB24-LeftDecorr interlaced row) that isolates the
  `decode_frame_interlaced` field-split overhead from the
  raster-size delta. `benches/encode.rs` closes the RGB32 gap the
  decode/roundtrip benches already had (Left / LeftDecorr /
  GradientDecorr v2.x + Left v1.x) and adds an RGB24 v1.x and a
  YUY2 interlaced encode row, giving a symmetric v1.x / v2.x ×
  YUY2 / RGB24 / RGB32 matrix. Headline finding: decode is the
  whole-pipeline floor at a flat ~100–123 MiB/s across all six
  predictors (predictor-independent ⇒ the per-symbol Huffman read
  in `decoder::decode_*_field`, not the `inverse_*_post` pass,
  governs decode throughput), making it the primary profile-opt
  target; the encode-side `encode_frame_auto` package-merge
  selector is the secondary target at ~3–3.5× the cost of
  fixed-method encode. No `src/` change.

- Round-277: wire-position (slot, store-offset) binding hoist on
  the RGB24 / RGB32 Huffman-decode loops
  (`decoder::decode_rgb24_field` / `decode_rgb32_field`). The
  pre-r277 loops re-evaluated the loop-invariant `decorrelate`
  branch on every pixel and re-resolved the slot pointers + BGR
  store offsets inside each arm; the binding is now resolved once
  at function entry per the spec/03 §1.4 wire codeword order
  (`B, G, R` / `G, B−G, R−G`, RGB32 appending the
  mode-independent slot-3 alpha code) + §1.2 slot mapping, and
  the loop bodies are fixed straight-line three- / four-decode
  sequences with no per-pixel branch — the decode-side analogue
  of the r239 / r245 encoder emit-loop bindings, completing the
  r214 / r239 / r245 hoist series on the two remaining RGB decode
  loops. Wire-identical; nine new
  `round277_rgb_decode_binding_tests` lock the rewrite (two
  per-pixel-branch reference witnesses under CustomV2
  content-distinct tables, decorr/no-decorr/PredictOld
  round-trips at widths 1 / 2 / 4 / 8, and V1xCompat
  content-identical-table round-trips pinning the store-offset
  half of the binding). Lib test count: 245 → 254.
- Round-262: the six remaining per-family open-coded `width ×
  {2,3,4}` wire-stride sites now route through the round-261
  `PixelFamily::row_bytes` accessor (spec/02 §3 wire-byte layout
  table) — `decoder::decode_yuy2_field` /
  `decoder::decode_rgb24_field` / `decoder::decode_rgb32_field`
  on the decode side and `encoder::yuy2_residuals` /
  `encoder::rgb24_residuals` / `encoder::rgb32_residuals` on the
  encode side. Each function is family-specific, so the call
  names the family variant explicitly
  (`PixelFamily::Yuy2.row_bytes(width)` etc.) and the inlined
  `{2,3,4}` literal disappears from the call site; the family →
  stride mapping now has exactly one origin (the r261 accessor,
  pinned against the spec table by the r261
  `*_bytes_per_pixel_step_matches_spec_table` witness). The
  accessor's `saturating_mul` is strictly safer than the prior
  plain multiply on 32-bit targets (a hostile `u32::MAX` width
  saturates instead of overflowing); on 64-bit the product
  `u32::MAX × 4` always fits, so behaviour is identical for every
  reachable input. Wire-identical to round 261 — six new
  `round262_*` tests lock the migration: three
  `round262_{yuy2,rgb24,rgb32}_decoded_raster_len_matches_family_row_bytes`
  round-trips pin the decoded raster length at
  `family.row_bytes(width) × height` AND bit-exact content per
  family (Left + a second method each: Median / LeftDecorr /
  GradientDecorr); a
  `round262_degenerate_dims_still_rejected_post_accessor` sweep
  re-pins the fuzz-found `total_bytes < {3,4}` degenerate-raster
  rejection across all three families × (0×4 / 4×0 / 0×0) now
  that the stride comes from the saturating accessor; a
  `round262_residuals_size_contract_matches_family_row_bytes`
  fixture drives each residual builder with exact /
  one-short / one-long pixel buffers and asserts the
  accept/reject boundary sits exactly at
  `family.row_bytes(width) × height`; and a
  `round262_residual_body_len_follows_bytes_per_pixel_step`
  fixture pins the residual body lengths to the spec/03 §1.1 /
  §1.3 codes-per-pixel counts expressed through
  `bytes_per_pixel_step` (YUY2: `row_bytes × h − 4` after the
  seed macropixel; RGB24 / RGB32: `(n_pixels − 1) × {3,4}` across
  no-decorr + decorr methods). Lib test count: 238 → 244.

- Round-261: typed accessors `PixelFamily::bytes_per_pixel_step`,
  `PixelFamily::row_bytes(width)`, `StreamConfig::row_bytes()`, and
  `StreamConfig::is_interlaced()`. The decode + encode paths
  previously open-coded the `match family { Yuy2 => width × 2, Rgb24
  => width × 3, Rgb32 => width × 4 }` family → wire-stride dispatch
  at two call sites (`decoder::decode_frame` interlaced field-merge
  setup, `encoder::encode_field` field-stride setup); both now defer
  to the single `PixelFamily::row_bytes` / `StreamConfig::row_bytes`
  accessor sourced from spec/02 §3 wire-byte layout table
  (`Y₁ U Y₂ V` = 4 bytes/macropixel = 2 bytes/pixel; RGB24 = 3
  bytes/pixel `+0:B +1:G +2:R`; RGB32 = 4 bytes/pixel `+0:B +1:G
  +2:R +3:A`). `PixelFamily::row_bytes` uses `saturating_mul` so a
  hostile `u32::MAX` width can't overflow `usize` on a 32-bit target.
  `StreamConfig::is_interlaced` is a thin wrapper over
  `predict::is_interlaced_height(self.height)` (spec/02 §2
  `biHeight > 288` threshold) so callers that hold a `StreamConfig`
  don't need to import the predict module to ask the same question.
  Wire-identical to round 255 — five new
  `round261_*` accessor tests lock the rewrite: a
  `*_bytes_per_pixel_step_matches_spec_table` witness pins the
  `{2,3,4}` literal against the spec/02 §3 lines 790–793 table; a
  `*_pixel_family_row_bytes_matches_inline_match` witness diffs the
  new `row_bytes(width)` accessor against the open-coded
  `width as usize * {2,3,4}` pattern across the bench-reference
  width set (0 / 1 / 2 / 4 / 8 / 16 / 160 / 320 / 720 / 1024 /
  1920); a `*_row_bytes_saturates_on_overflow` fixture exercises
  the `u32::MAX` saturating-multiply contract; a
  `*_stream_config_row_bytes_delegates_to_family` round-trip
  exercises the StreamConfig accessor through a real
  `parse_bitmapinfoheader` path on YUY2 320×16 (the bench-reference
  raster), RGB24 1024×720 (the spec/01 §7 worked example), and
  RGB32 720×480 (a DV-class active raster); and a
  `*_stream_config_is_interlaced_matches_height_threshold` fixture
  walks the strict `> 288` threshold (0 / 1 / 288 / 289 / 480 /
  720 / 1080) cross-checking the StreamConfig wrapper against the
  predict-module source of truth. Lib test count: 233 → 238.

- Round-255: half-macropixel (2-byte) step body for the YUY2 inverse
  MEDIAN region in `decoder::inverse_yuy2_median`. The pre-r255
  median region was a per-byte loop that re-issued three
  uniform-offset lookbacks (`L = out[pos − 2]`, `A = out[pos −
  row_bytes]`, `AL = out[pos − row_bytes − 2]`) and a
  `gradient_predictor → median3 → wrapping_add` chain on every byte
  position one at a time. The inverse direction reads `L` from the
  same `out` buffer it writes to, so a full 4-byte unroll (as r253
  applied to the encoder's forward MEDIAN body) would introduce
  read-after-write aliasing within the unrolled step: intra-step
  offset +2 / +3 reads `out[pos] / out[pos + 1]`, which are the
  writes from intra-step offset +0 / +1. At a 2-byte step, both
  `L` sources within the step (`out[pos − 2]`, `out[pos − 1]`) are
  read from positions finalised before the step began (the previous
  step's writes, or the LEFT-region bytes for the very first step
  at `pos = row_bytes + 8`), so the two `gradient_predictor →
  median3 → wrapping_add` chains within the step are independent and
  the compiler is free to schedule them across functional units.
  `row1_left_end = (row_bytes + 8).min(len)` is always a multiple
  of 2 for in-spec YUY2 input (row_bytes = 2 × width ⇒ row_bytes ≡
  0 mod 2; the +8 keeps the alignment), so the 2-byte step body
  covers every wire byte in the median region; a 1-byte scalar
  fall-through is kept for defence-in-depth (mirroring the r221 /
  r227 / r239 / r242 / r245 / r250 / r253 fall-throughs).
  Decoder-side companion to r253's encoder forward MEDIAN
  macropixel-step rewrite, applied to the same §2.3 four-byte YUY2
  median rhythm but at half the unroll factor to respect the
  natural sequential `L` dependency the inverse predictor carries
  per spec/03 §2.3. Wire-identical to round 253 — five new
  `round255_inverse_yuy2_median_step_tests` lock the rewrite: a
  `*_matches_per_byte_reference` witness diffs the production
  output against an inlined copy of the pre-r255 per-byte body
  across YUY2 widths 2 / 4 / 8 / 16 / 160 / 320 (covering the
  narrow `row_bytes < 8` regime where the LEFT exemption extends
  past row 1, the bench-reference 320×16 raster, and several
  intermediate sizes); a `*_modular_wrap` fixture forces mod-256
  wraps inside the gradient and the final add on every iteration;
  a `*_boundary_row1_step_alignment` fixture pins the
  `row1_left_end % 2 == 0` and `(len − row1_left_end) % 2 == 0`
  invariants the step body relies on; a `*_scalar_tail_safety`
  fixture exercises the 1-byte scalar fall-through against an
  out-of-spec odd-length buffer (the in-spec body is multiple-of-2
  so the tail is defence-in-depth only); and an
  `*_encoder_roundtrip` fixture chains
  `encode_frame_with_mode → decode_frame` and asserts bit-exact
  pixel reconstruction across the same widths the r253 forward
  tests cover (so any forward / inverse drift would surface as a
  roundtrip mismatch). Lib test count: 228 → 233.

- Round-253: macropixel-step body for the YUY2 forward MEDIAN
  predict region in `predict::forward_median_subtract`. The
  pre-r253 median region was a per-byte loop that re-issued three
  uniform-offset lookbacks (`L = pixels[pos − 2]`, `A = pixels[pos
  − row_bytes]`, `AL = pixels[pos − row_bytes − 2]`) and a
  `gradient_predictor → median3 → wrapping_sub` chain on every
  byte position one at a time. Spec/03 §2.3 evidence at
  `huffyuv.dll@0x10002130..@0x10002138` (and the parallel
  `@0x10001fab..@0x10002095` encoder trace) pins those three
  lookback offsets at the **same** displacement for every
  median-region byte, independent of intra-macropixel position —
  unlike spec/03 §2.1.1 YUY2 LEFT (which alternates stride 2 / 4
  by `pos & 3`). The four `median3` computations at positions
  `pos+0`, `pos+1`, `pos+2`, `pos+3` are therefore independent
  (each reads from disjoint input offsets and writes to a
  disjoint output offset), and a 4-byte unroll exposes
  instruction-level parallelism so the compiler can schedule the
  four chains freely across functional units. `median_start =
  (row_bytes + 8).min(n)` is always a multiple of 4 for in-spec
  YUY2 input (row_bytes = 2 × width with width even ⇒ row_bytes
  ≡ 0 mod 4; the +8 keeps the alignment), so the macropixel-step
  body covers every wire byte in the median region; a 1..=3-byte
  scalar fall-through is kept for defence-in-depth (mirroring
  the r221 / r227 / r239 / r242 / r245 / r250 fall-throughs).
  Predict-side companion to r214 / r221's YUY2 macropixel-step
  decode + emit (which targeted the §2.1.1 four-byte LEFT
  rhythm) — r253 applies the same macropixel-step rewrite shape
  to the §2.3 four-byte median rhythm on the encoder predict
  body. Wire-identical to round 250 — five new
  `round253_forward_median_macropixel_*` tests lock the rewrite:
  one `*_matches_per_byte_reference` witness diffs the
  production output against an inlined copy of the pre-r253
  per-byte body across YUY2 widths 2 / 4 / 8 / 16 / 160 / 320
  (covering the narrow `row_bytes < 8` regime where the LEFT
  exemption extends past row 1, the bench-reference 320×16
  raster, and several intermediate sizes); a modular-wrap
  fixture forces mod-256 wraps inside the gradient + the final
  subtract on every iteration; a `boundary_row1_step_pattern`
  fixture pins the `median_start % 4 == 0` and `(n − median_start)
  % 4 == 0` invariants the macropixel-step body relies on; a
  `*_then_inverse_roundtrips` fixture chains
  `forward_median_subtract → YUY2 LEFT add over the exempt
  region → median ADD for the rest` and asserts bit-exact pixel
  reconstruction (the wire-format claim); and a
  `*_scalar_tail_safety` fixture exercises the 1..=3-byte
  scalar fall-through under release builds (the in-spec body is
  multiple-of-4 so the tail is defence-in-depth only). Lib test
  count: 223 → 228.


- Round-250: pixel-step RGB32 histogram body in `histogramise`. The
  pre-r250 loop ran `match i % 4` on every body byte to pick the
  histogram AND a `method.decorrelate()` branch on every iteration —
  two per-byte branches the optimiser could not eliminate because
  `i` was the iterator state and the answer to
  `method.decorrelate()` never changes mid-frame. Round 250 hoists
  both decisions out of the loop: the per-position histogram
  references are resolved once at function entry by the
  `(h_pos0, h_pos1, h_pos2)` binding (paired by
  `method.decorrelate()`), then the body steps four bytes per outer
  iteration with the slot resolved at compile time — `h_pos0[+0]
  += 1 → h_pos1[+1] += 1 → h_pos2[+2] += 1 → h_pos2[+3] += 1` per
  wire pixel. spec/03 §1.3 pins RGB32 at exactly four Huffman
  codewords per pixel; §1.2 fixes the position → slot mapping at
  `(slot1, slot2, slot3, slot3)` for no-decorrelate methods
  (B / G / R / A) and `(slot2, slot1, slot3, slot3)` for decorrelate
  methods (G / B−G / R−G / A). Alpha shares the slot-3 codebook per
  the §1.2 evidence at `@0x10001b21` (no-decorr A emit) and
  `@0x10001c6d` (decorr A emit) — so the +3 (alpha) position
  re-uses `h_pos2`, accumulating into slot 3 alongside the +2
  (R / R−G) column. `body.len()` is always a multiple of 4 in the
  in-spec input space (the body is `(n_pixels − 1) × 4` bytes per
  `rgb32_residuals`), so the pixel-step body covers every count
  byte; a 1..=3-byte scalar fall-through is kept for
  defence-in-depth, mirroring the r221 / r227 / r239 / r242 / r245
  fall-throughs. Histogram-side companion to r245's RGB32 emit
  rewrite, applied to the same §1.3 four-byte RGB32 wire cycle (and
  a direct mirror of r242's RGB24 histogram pixel-step body,
  extended to the alpha position). Wire-identical to round 245 —
  six new `round250_rgb32_histogram_pixel_step_tests` lock the
  rewrite: a per-byte witness diffs the production histograms
  against an inlined copy of the pre-r250 per-byte body over real
  residual bodies + a `0..128` synthetic body that densely covers
  every residue position across 32 wire pixels; a no-decorr-vs-
  decorr quadruple pair pins the slot1/slot2 column swap; an alpha-
  shares-slot-3 aggregation fixture lays the same value at +2 and
  +3 within each pixel and verifies slot 3 accumulates two
  increments per pixel; and two end-to-end CustomV2 round-trip
  sweeps at widths 1 / 2 / 4 / 8 across
  Left / LeftDecorr / GradientDecorr exercise the histogram →
  canonical-length → emit chain. Lib test count: 217 → 223.


- Round-245: pixel-step RGB32 Huffman-encode body in
  `emit_bitstream_parts`. The pre-r245 loop ran `match i % 4` on every
  body byte to pick the slot AND a `method.decorrelate()` branch on
  every iteration — two per-byte branches the optimiser could not
  eliminate because the iterator state changed every step. Round 245
  hoists both decisions out of the loop: the slot quadruple is
  resolved once at function entry by the `(s_pos0, s_pos1, s_pos2,
  s_pos3)` binding (paired by `method.decorrelate()`), then the body
  steps four bytes per outer iteration with the slot resolved at
  compile time — `lookup_code(s_pos0) → write_msb →
  lookup_code(s_pos1) → write_msb → lookup_code(s_pos2) → write_msb →
  lookup_code(s_pos3) → write_msb` straight-line per wire pixel.
  spec/03 §1.3 pins RGB32 at exactly four Huffman codewords per pixel;
  §1.2 fixes the position → slot mapping at `(slot1, slot2, slot3,
  slot3)` for no-decorrelate methods (B / G / R / A) and `(slot2,
  slot1, slot3, slot3)` for decorrelate methods (G / B−G / R−G / A).
  Alpha shares the slot-3 codebook per the §1.2 evidence at
  `@0x10001b21` (no-decorr A emit) and `@0x10001c6d` (decorr A emit).
  `body.len()` is always a multiple of 4 in the in-spec input space
  (the body is `(n_pixels − 1) × 4` bytes per `rgb32_residuals`), so
  the pixel-step body covers every emit byte; a 1..=3-byte scalar
  fall-through is kept for defence-in-depth, mirroring the
  r221 / r227 / r239 fall-throughs. Encoder analogue of r239's RGB24
  emit rewrite, applied to the §1.3 four-byte RGB32 wire cycle (and a
  direct mirror of the r221 YUY2 macropixel-step body on the §1.2
  four-byte YUY2 cycle). Wire-identical to round 242 — nine new
  `round245_rgb32_emit_pixel_step_tests` lock the rewrite: eight
  pixel-step-boundary round-trips covering Left at widths 1 / 2 / 4 /
  8 under `ClassicV2`, `PredictOld` (alternate no-decorr entry),
  `LeftDecorr` + `GradientDecorr` (the decorr-pair slot quadruple),
  and `V1xCompat` (the content-identical `(A, A, A)` triple, plus the
  alpha-shares-slot-3 reuse degenerating to a third reference to the
  single table); and a `*_matches_per_byte_reference` witness diffs
  the production emit bit stream against an inlined copy of the
  pre-r245 per-byte slot dispatch across `Left` / `LeftDecorr` /
  `GradientDecorr` under `CustomV2` (content-distinct tables, so a
  slot mix-up surfaces as a Huffman-code mismatch on the wire even
  before the round-trip predictor pass — and the alpha-shares-slot3
  reuse means the witness also covers the "two positions, one table"
  case spec/03 §1.2 calls out). Lib test count: 208 → 217.


- Round-242: pixel-step RGB24 histogram body in `histogramise`. The
  pre-r242 loop ran `match i % 3` on every body byte to pick the slot
  AND a `method.decorrelate()` branch on every iteration — two
  per-byte branches the optimiser could not eliminate because `i` was
  the iterator state and the answer to `method.decorrelate()` never
  changes mid-frame. Round 242 hoists both decisions out of the loop:
  the per-position histogram triple is resolved once at function entry
  by the `(h_pos0, h_pos1, h_pos2)` binding (paired by
  `method.decorrelate()`), then the body steps three bytes per outer
  iteration with the slot resolved at compile time — three indexed
  counter increments per wire pixel. spec/03 §1.1 pins RGB24 at
  exactly three Huffman codewords per pixel (the §3.2/§3.3 spec/02
  correction); §1.2 fixes the position → slot mapping at
  `(slot1, slot2, slot3)` for no-decorrelate methods (B / G / R) and
  `(slot2, slot1, slot3)` for decorrelate methods (G / B−G / R−G).
  `body.len()` is always a multiple of 3 in the in-spec input space
  (the body is `(n_pixels − 1) × 3` bytes per `rgb24_residuals`), so
  the pixel-step body covers every count byte; a 1..=2-byte scalar
  fall-through is kept for defence-in-depth, mirroring the
  r221 / r227 / r239 fall-throughs. Histogram-side companion to
  r239's RGB24 emit rewrite (and mirror of r227's YUY2 histogram
  macropixel-step body, applied to the §1.2 three-byte RGB24 wire
  cycle). Wire-identical to round 239 — seven new
  `round242_rgb24_histogram_pixel_step_tests` lock the rewrite:
  one `*_matches_per_byte_reference` witness drives Left /
  LeftDecorr / GradientDecorr at widths 1 / 2 / 4 / 8 and diffs the
  production histograms against an inlined copy of the pre-r242
  per-byte body element-by-element (and asserts the per-slot count
  total equals `body.len()`); two synthetic-body witnesses
  (no-decorr triple and decorr triple) drive a `0..96` body that
  densely covers every residue position with distinct values so a
  slot mix-up surfaces as counts attributed to the wrong histogram;
  four end-to-end CustomV2 round-trips at widths 1 / 2 / 4 / 8
  across `Left` / `LeftDecorr` / `GradientDecorr` exercise the
  histogram → canonical-length → emit chain (CustomV2 builds the
  per-slot length tables straight from `histogramise`, so any
  histogram drift would change the emitted lengths and break the
  round-trip). Lib test count: 201 → 208.


- Round-239: pixel-step RGB24 Huffman-encode body in
  `emit_bitstream_parts`. The pre-r239 loop ran `match i % 3` on every
  body byte to pick the slot AND a `method.decorrelate()` branch on
  every iteration — two per-byte branches the optimiser could not
  eliminate because the iterator state changed every step. Round 239
  hoists both decisions out of the loop: the slot triple is resolved
  once at function entry by the `(s_pos0, s_pos1, s_pos2)` binding
  (paired by `method.decorrelate()`), then the body steps three bytes
  per outer iteration with the slot resolved at compile time —
  `lookup_code(s_pos0) → write_msb → lookup_code(s_pos1) → write_msb →
  lookup_code(s_pos2) → write_msb` straight-line per wire pixel. spec/03
  §1.1 pins RGB24 at exactly three Huffman codewords per pixel (the
  §3.2/§3.3 spec/02 correction); §1.2 fixes the position → slot mapping
  at `(slot1, slot2, slot3)` for no-decorrelate methods (B / G / R)
  and `(slot2, slot1, slot3)` for decorrelate methods (G / B−G / R−G).
  `body.len()` is always a multiple of 3 in the in-spec input space
  (the body is `(n_pixels − 1) × 3` bytes per `rgb24_residuals`), so
  the pixel-step body covers every emit byte; a 1..=2-byte scalar
  fall-through is kept for defence-in-depth against future pixel-family
  extensions, mirroring the r221 / r227 YUY2 fall-throughs. Encoder
  analogue of r221's YUY2 pixel-step body, applied to the §1.2 three-
  byte RGB24 wire cycle. Wire-identical to round 234 — locked by nine
  new `round239_rgb24_emit_pixel_step_tests` covering: four
  pixel-step-boundary round-trips at widths 1 / 2 / 4 / 8 (`Left` +
  `ClassicV2`); one `PredictOld` variant locking the alternate
  no-decorr method entry; one `LeftDecorr` + one `GradientDecorr`
  round-trip locking the decorr-pair slot triple; one V1xCompat
  round-trip locking the content-identical `(A, A, A)` triple; and a
  `*_matches_per_byte_reference` witness diffing the production emit
  bit stream against an inlined copy of the pre-r239 per-byte slot
  dispatch body across `Left` / `LeftDecorr` / `GradientDecorr` under
  `CustomV2` (content-distinct slot tables, so a slot mix-up surfaces
  as a Huffman-code mismatch on the wire even before the round-trip
  predictor pass). Lib test count: 192 → 201.

- Round-234: `fuzz/fuzz_targets/tables_huffyuv.rs` — third cargo-fuzz
  target driving the table-build primitives directly. The previous
  decode-only (r134) and encode-only (r196) targets reach the
  table-build surface only via a valid BIH parse — the decode-only
  target rejects malformed extradata before it lands on
  `HuffTable::build_from_lengths`, and the encode-only target only
  ever hands `compute_canonical_lengths` a histogram built from real
  residual bytes the encoder just produced. Round 234 slices its fuzz
  buffer five ways with a selector byte and drives each table-build
  primitive directly:
  - `drive_rle_decode` — `rle_decode_one_channel` and
    `rle_decode_three_channels` on arbitrary bytes; on success the
    output length table must round-trip byte-exact through
    `rle_encode_one_channel` → `rle_decode_one_channel`.
  - `drive_rle_encode_roundtrip` — `rle_encode_one_channel` →
    `rle_decode_one_channel` on a fuzz-derived 256-byte length table
    coerced into `0..=31` via `byte & 0x1F`. The pair must round-trip
    byte-exact.
  - `drive_build_from_lengths` — `HuffTable::build_from_lengths` on a
    fuzz-derived length table; the call either errors (Kraft
    inequality) or succeeds with the self-consistent-decode contract:
    every nonzero-length entry's MSB-aligned code must `decode_one`
    back to itself.
  - `drive_compute_canonical_lengths` — the package-merge
    length-limited length computation on a fuzz-derived 256-entry
    histogram (built by packing pairs of fuzz bytes into LE `u16`
    counts); on success the returned length table must build through
    `HuffTable::build_from_lengths` and decode self-consistently
    (that's the encoder's documented downstream chain).
  - `drive_v1x_table_from_pair` — `v1x_table_from_pair` on a
    fuzz-derived `(lengths, codes)` pair; spec/04 §4.2 permits the
    v1.x set to be non-canonical, so only the `decode_one` liveness
    property is asserted (Ok return for every nonzero-length entry's
    own code window, even if the returned `(symbol, length)` differs
    from the looked-up entry under a v1.x prefix collision).
  - Six handcrafted corpus seeds under
    `fuzz/corpus/tables_huffyuv/` cover all five drivers plus the
    empty-input early return.
- Round-234: four `round234_…` invariant unit tests in
  `src/tables.rs` reproduce the fuzz target's contracts on fixed
  inputs (the all-zero / single-tier / sparse RLE shapes; three
  Kraft-equal length distributions plus an eight-symbol uniform
  alphabet for `build_from_lengths`; four histogram shapes —
  uniform, single-symbol, two-symbol, skewed — for
  `compute_canonical_lengths`; the proprietary's v1.x set-A pair for
  `v1x_table_from_pair`). Runs on every `cargo test`; the fuzz
  harness only runs on the daily CI schedule. Lib test count: 188 →
  192.
- Round-234: `tables_huffyuv` registered in `fuzz/Cargo.toml` and
  auto-discovered by the daily reusable fuzz workflow. Each of the
  three targets (`decode_huffyuv` / `encode_huffyuv` /
  `tables_huffyuv`) now gets ~600 s of the 30-minute total budget per
  daily run.

### Changed

- Round-227: macropixel-step YUY2 histogram + verify bodies. Two
  YUY2-body iterators still ran per-byte `match byte_idx & 3`
  slot dispatch after r214 (decode loop) and r221 (emit loop)
  closed the equivalent dispatch on the Huffman-decode and
  Huffman-emit sides:
  - `encoder::histogramise` — called once per frame on the
    CustomV2 path and once per candidate inside
    `bit_cost_for_method` on the `MethodSelection::Auto` path,
    so the per-byte dispatch was multiplied by the number of
    candidates per frame on the auto-selector path.
  - `encoder::verify_body_in_table` — V1xCompat sanity walk
    that checks every body symbol against its slot's v1.x
    precomputed-code length to reject inputs whose residual
    symbol set isn't covered by the v1.x tables.
  Round 227 hoists the spec/03 §1.2 three-slot architecture out
  of both loops at the source by stepping four body bytes per
  outer iteration with the slot resolved at compile time. The
  inner histogram body becomes a fixed straight-line `h1[+0]
  += 1 → h2[+1] += 1 → h1[+2] += 1 → h3[+3] += 1` sequence per
  4-byte macropixel; the inner verify body becomes a matching
  `slot1[+0] / slot2[+1] / slot1[+2] / slot3[+3]` `entries[sym]
  .length == 0` gate sequence. `body.len()` is always a multiple
  of 4 in the in-spec input space (YUY2 width is even per
  spec/02 §3.1's macropixel-pair invariant and `body.len() =
  total_bytes − 4` per field, so the body length sits at
  `(width × 2 × height) − 4`, divisible by 4), so the macropixel
  body covers every input byte. A 1..=3-byte scalar fall-through
  is kept for defence-in-depth, mirroring the same shape landed
  in r214 / r221. Wire-identical to round 221 — twelve new
  `round227_yuy2_histogram_verify_macropixel_tests` pin the
  rewrite at byte-equality, including two
  `*_matches_per_byte_reference` witnesses that diff the
  production histograms + verify result against an inlined copy
  of the pre-r227 per-byte slot-dispatch body across Left /
  Gradient / Median predictors (success path) and against a
  slot2-targeted out-of-codebook symbol (verify failure path),
  plus eight end-to-end round-trips (four CustomV2 widths
  exercising the histograms → length-table → emit chain at
  widths 2 / 4 / 8 / 16, four V1xCompat widths exercising the
  verify walk at the same widths). Lib test count: 176 → 188.

- Round-221: macropixel-step YUY2 Huffman-encode body. The
  encoder mirror of round-214's decode-side rewrite. The pre-r221
  `emit_bitstream_parts` YUY2 branch ran `match byte_idx & 3` on
  every body byte to pick the per-channel slot (Y₁/Y₂ → slot1,
  U → slot2, V → slot3), so the slot-pointer reload + branch sat on
  the critical path of every Huffman emit. Round 221 pins
  spec/03 §1.2's three-slot architecture at the source by stepping
  four body bytes per outer iteration with the slot resolved at
  compile time — the inner body becomes a fixed straight-line
  `lookup_code(slot1) → write_msb → lookup_code(slot2) → write_msb
  → lookup_code(slot1) → write_msb → lookup_code(slot3) →
  write_msb` sequence per 4-byte macropixel. `body.len() & !3` is
  the aligned upper bound; `body.len() % 4 == 0` holds in the
  in-spec input space (YUY2 width is even per the spec/02 §3.1
  macropixel-pair invariant and `body.len() = total_bytes − 4` per
  field), so the macropixel body covers every emit byte. A
  1..=3-byte scalar fall-through is kept for defence-in-depth,
  mirroring the same shape landed in r214 on the decoder side.
  Wire-identical to round 214 — six new
  `round221_yuy2_emit_macropixel_tests` lock the rewrite at byte
  equality, including a `*_matches_per_byte_reference` witness that
  diffs the production emit against an inlined copy of the pre-r221
  per-byte slot-dispatch body across Left / Gradient / Median
  predictors. Lib test count: 170 → 176.

- Round-214: macropixel-step YUY2 Huffman-decode body. The
  pre-r214 `decode_yuy2_field` loop ran `match byte_idx % 4` on every
  output byte to pick the per-channel slot (Y₁/Y₂ → slot1, U → slot2,
  V → slot3), so the slot-pointer reload sat on the critical path of
  every Huffman lookup. Round 214 pins spec/03 §1.2's three-slot
  architecture at the source by stepping four output bytes per outer
  iteration with the slot resolved at compile time — the inner body
  becomes a fixed straight-line `decode_one(slot1) →
  decode_one(slot2) → decode_one(slot1) → decode_one(slot3)` sequence
  plus four indexed stores per 4-byte macropixel. The
  decode-side analogue of round 181's LEFT macropixel-step rewrite
  (also branch-elimination on the same 4-byte cycle, mirrored to the
  Huffman-decode loop now that the LEFT inverse has already shed its
  `i & 3` switch). `(total_bytes - 4) % 4 == 0` is invariant in the
  in-spec input space (YUY2 width is even per the macropixel-pair
  invariant the decoder already checks), so the macropixel body
  covers every remaining byte; a 1..=3-byte scalar fall-through is
  kept for defence-in-depth against future pixel-family extensions.
  Wire-identical to round 208 — every pre-existing YUY2 round-trip
  test stays green and 6 new `round214_yuy2_decode_macropixel_tests`
  pin the rewrite at byte-equality. Lib test count: 164 → 170.

- Round-208: drop three single-use decoder-local LEFT wrappers
  (`decoder::inverse_left_per_channel`, `decoder::inverse_yuy2_left`,
  `decoder::inverse_yuy2_left_range`) and re-point the YUY2 + RGB24 +
  RGB32 decode paths at the public predict-side helpers
  (`predict::inverse_left_row` for the per-channel stride-`n` LEFT
  walk; `predict::inverse_yuy2_left_macropixel` for the YUY2
  byte-position-stride LEFT walk) directly. `inverse_left_per_channel`
  was a byte-for-byte duplicate of `inverse_left_row` — same
  `if out.len() <= n { return; }` guard, same `out[i] =
  out[i].wrapping_add(out[i - n])` body, just expressed as a `while
  idx < ...` loop in one and a `for` loop in the other — and the two
  YUY2-LEFT shims (`inverse_yuy2_left` /
  `inverse_yuy2_left_range`) were already thin pass-throughs into
  `predict::inverse_yuy2_left_macropixel` (left over from the round-181
  macropixel-step rewrite that retired the original per-byte `i & 3`
  switch). The cleanup gives the decoder one source of truth for both
  LEFT predictors (`predict.rs`) instead of two — matches spec/03 §2.1
  (per-channel) and §2.1.1 (YUY2) as the single canonical
  implementation. The three deleted helpers were 28 lines of
  duplication. Lib test count: 161 → 164 (new `round208_reuse_tests`
  module: `predict_inverse_left_row_matches_decoder_naive_rgb24` /
  `_rgb32` /
  `predict_inverse_yuy2_left_macropixel_matches_decoder_naive` — each
  checks the predict-side helper produces byte-identical output to a
  naive scalar reference of the spec/03 §2.1 / §2.1.1 textual form, so
  a future refactor cannot silently invert LEFT differently on either
  side).

- Round-202: strip three intra-loop dead branches from the YUY2 Median
  tail-loop on both sides of the codec. After spec/03 §2.3.2 +
  audit/01 §7.2's wire-byte LEFT exemption (the loop start sits at
  `row_bytes + 8` capped at the buffer length), every iteration
  satisfies `pos >= row_bytes + 8`, which implies `pos >= 2`,
  `pos >= row_bytes`, and `pos - row_bytes >= 8 >= 2`. The pre-round-202
  body of `predict::inverse_median_post` /
  `decoder::inverse_yuy2_median` carried per-iteration `if pos < 2 ||
  pos < row_bytes { continue; }` and `if pos >= row_bytes + 2 { … }
  else { 0 }` arms that the loop precondition makes provably dead;
  `predict::forward_median_subtract`'s encoder analogue carried the
  matching `al = 0` else-arm. The cleanup replaces the `while`/`if`
  shape with a straight-line `for pos in row1_median_start..n {}`
  body that reads `out[pos - 2]`, `out[pos - row_bytes]`, and
  `out[pos - row_bytes - 2]` directly — three branch-free
  `wrapping_*` median-builder steps per iteration. The wrap-arithmetic
  `pos.wrapping_sub(2)` substitute in the pre-202 form (placed there to
  silence the formerly-not-provably-non-wrapping `pos - 2`) is also
  dropped: with the dead branches gone, `pos - 2` is a plain
  non-wrapping subtraction. `debug_assert!`s anchor the row-stride
  invariants at function entry. Lib test count: 160 → 161 (new
  `roundtrip_yuy2_median_round202_boundary_widths` sweeps nine
  `(width, height)` pairs bracketing the LEFT-exemption + AL-index
  boundaries — widths 2 / 4 / 6 / 8 × heights 3..8 — to keep the
  invariants the dead-branch strip relies on regression-guarded).

### Added

- Round-196: `fuzz/fuzz_targets/encode_huffyuv.rs` — second cargo-fuzz
  target driving `encode_frame_with_mode` across every legal
  `(family, method, extradata-mode)` triple on arbitrary input pixels
  (5-byte selector prefix + raw pixel tail; widths 1–64, heights 1–32
  plus a low-probability lift into the interlaced regime >288). Each
  iteration encodes, parses the encoder-produced strf back via
  `StreamConfig::parse_bitmapinfoheader`, re-decodes via
  `decode_frame`, and asserts bit-exact round-trip equality of the
  raster (HuffYUV is lossless per spec/00 §1.1). Also exercises the
  publicly-exposed `build_bitmapinfoheader` strf-write helper on the
  same configuration so muxer-side callers are fuzz-covered too. Seven
  curated corpus seeds (six valid `(family, method, mode)` triples +
  the round-196 regression below). 200k-run smoke against the fixed
  encoder: 103.5k execs in 181 s with `oom/timeout/crash: 0/0/0`.
- Round-196: two new regression tests in `roundtrip_tests.rs`
  (`roundtrip_yuy2_median_2x18_round196` + the matching minimal
  fuzz-discovered repro `_fuzz_minimal` with prefix
  `[80, 255, 17, 80, 175]` and zero-padded tail) covering the
  `width = 2` YUY2 Median path (row_bytes = 4 < the 8-wire-byte LEFT
  exemption). Lib test count: 158 → 160.

### Fixed

- Round-196 (fuzz-found): `predict::forward_median_subtract` under-sized
  the YUY2 Median LEFT-exemption region for narrow streams. spec/03
  §2.3.2 + audit/01 §7.2 define the exemption as "first 4 pixel (2
  pairs, 8 **wire** bytes) of the second row are compressed with the
  predict left algorithm," independent of row stride (i386 decoder
  loop bound at `huffyuv.dll@0x100020e8` sets the LEFT-region end at
  `row_start_of_row_1 + row_stride + 8`). Prior code clamped the
  exemption with `8.min(row_bytes)`, so for `row_bytes < 8` (e.g.
  width = 2 → row_bytes = 4) the LEFT region ended at the row 1
  boundary instead of `row_bytes + 8`; the decoder
  (`inverse_yuy2_median`) had this right per the spec, so the encoder
  was emitting MEDIAN residuals for wire bytes the decoder would
  treat as LEFT — producing wrong reconstruction from row 2 onward on
  every `width < 4` YUY2 Median input. Fix: drop the clamp; use
  `(row_bytes + 8).min(n)` symmetrically. Same correction applied to
  the dead-but-documented `inverse_median_post` and the
  `naive_two_phase_forward_median` reference helper so the spec
  reference cannot drift. Discovered by the new `encode_huffyuv.rs`
  cargo-fuzz harness; pinned by the two new round-196 round-trip
  tests.

- Round-186: `predict::forward_rgb_left_subtract_linear(src, dst,
  n_channels)` — single linear stride-1 walk producing the per-channel
  LEFT residuals for RGB24 (`n = 3`) and RGB32 (`n = 4`) buffers,
  replacing the encoder's prior per-channel triple/quad-pass
  stride-`n` loop. The per-channel identity `dst[i] =
  src[i].wrapping_sub(src[i − n])` is the same for every channel
  (spec/03 §2.1, encoder evidence `@0x10001850` for RGB24 and
  `@0x10001b21..@0x10001b3c` for the RGB32 byte-3/A emit reusing the
  same offset-N-back rule), so the three (RGB24) or four (RGB32)
  strided passes collapse into one linear pass. Cuts traversal count
  by `n` (3× / 4× fewer cache-line loads) and exposes a contiguous
  inner subtract that LLVM autovectorises into NEON `vsubq_u8` on
  aarch64 / SSE2 `psubb` on x86_64 (the helper is `#[inline]` so
  `n_channels` constant-folds at every encoder call site, where it is
  the literal `3` or `4`).
- Round-186: 5 new equivalence + roundtrip tests in `predict.rs`
  (`round186_rgb_left_linear_matches_per_channel_rgb24`,
  `..._rgb32`, `..._modular_wrap`,
  `round186_rgb_left_linear_short_buffer_seed_only`,
  `round186_rgb_left_linear_then_inverse_roundtrips`) that diff the
  linear walk against an in-test copy of the prior per-channel
  stride-`n` loop across widths 1 / 4 / 16 / 320, heights 1 / 4 / 8,
  the 0xFF/0x01 modular-wrap alternator, and a buffer-smaller-than-
  one-pixel degenerate case, plus an end-to-end forward-then-inverse
  round-trip via the decoder's per-channel `wrapping_add` walk.

### Changed

- Round-186: `encoder::rgb24_residuals` / `rgb32_residuals` route the
  non-fused LEFT-residual emit through
  `predict::forward_rgb_left_subtract_linear` instead of the prior
  per-channel `for ch in 0..N { idx = ch + N; while idx < len { …
  idx += N; } }` loop. Wire-identical (every output byte is a pure
  function of `i mod n_channels`, so the linear walk produces the
  same residuals at every position); regression-guarded by the round
  186 equivalence tests + every pre-existing RGB24 / RGB32 LEFT /
  Gradient / LeftDecorr / GradientDecorr round-trip and the
  `lockstep_rgb24_*` AVI-lockstep tests.
- Round-186 measured (release, M1, isolated LEFT-subtract pass —
  no Huffman): RGB24 320×240 LEFT 86.7 µs/frame → 4.3 µs/frame
  (**≈ 20× speedup**); RGB24 1280×720 LEFT 1036 µs/frame →
  51.7 µs/frame (**≈ 20× speedup**); RGB32 320×240 LEFT
  114.9 µs/frame → 5.8 µs/frame (**≈ 20× speedup**); RGB32
  1280×720 LEFT 1380 µs/frame → 69.6 µs/frame (**≈ 20× speedup**).
  Lib test count: 153 → 158.

- Round-181: branch-free YUY2 LEFT macropixel-step helpers in
  `predict.rs`:
  - `inverse_yuy2_left_macropixel(out, begin, end)` — decoder LEFT
    inverse over byte range `[begin, end)` as a single straight-line
    Y₁ / U / Y₂ / V body per spec/03 §2.1.1 (the
    `@0x100020f4..@0x1000210e` macropixel-step trace), with three
    rolling channel accumulators (`prev_y` / `prev_u` / `prev_v`)
    replacing the per-iteration `i & 3` switch.
  - `forward_yuy2_left_subtract(src, dst)` — encoder analogue,
    reading the un-modified pre-pass input (raw pixels or the
    gradient pre-pass output) and writing the LEFT residuals into
    `dst` in a 4-byte-per-macropixel straight-line body, with the
    Y₂-from-same-pair-Y₁ intra-pair LEFT rule of §2.1.1 expressed
    directly as `src[i+2] − src[i]`.
- Round-181: 7 new equivalence + roundtrip tests in `predict.rs`
  (`round181_yuy2_left_macropixel_matches_branchy_full_frame`,
  `..._row1_first8`, `..._modular_wrap`,
  `..._short_buffer_noop`, `round181_yuy2_forward_left_matches_branchy`,
  `..._short_buffers`, `..._forward_then_inverse_roundtrips`) that
  diff the new helpers against in-test copies of the prior
  per-byte-branch loops across YUY2 widths 2 / 4 / 320 / 640 and
  cover the modular-wrap edge case + the median row-1-first-8
  range. Lib-test count: 146 → 153.

- Round-174: `benches/` Criterion harness (`decode`, `encode`,
  `roundtrip`) covering 22 representative
  `(pixel-family, method, extradata-mode, raster)` scenarios. Inputs
  are synthesised on the fly with a deterministic xorshift32 +
  diagonal-gradient pattern, so the benches stay self-contained (no
  committed binary fixtures, no `docs/` dependency). Coverage mirrors
  the README's "measured" headline scenarios so a future round can
  read a regression directly off the criterion delta:
  - **decode** (7 scenarios): YUY2 320×240 LEFT/Gradient/Median
    ClassicV2, YUY2 1280×720 LEFT ClassicV2, YUY2 320×240 LEFT
    V1xCompat (the round-3 fast-LUT, round-91 SWAR gradient post-pass,
    round-91 flat overflow-entries slow path, and round-7 OnceLock
    table cache all surface here).
  - **encode** (10 scenarios): the eight ClassicV2 / V1xCompat fixed
    methods that the README quoted ms/frame numbers for plus two
    `MethodSelection::Auto` + CustomV2 scenarios (YUY2 + RGB24).
  - **roundtrip** (5 scenarios): end-to-end encode → parse-BIH →
    decode pipeline health check.
  - Wired via `[dev-dependencies] criterion = "0.5"` + three
    `[[bench]] harness = false` entries. Local `--quick` run on M1
    (warm-up 1 s, measurement 2 s) confirms all 22 benches execute
    cleanly with representative throughputs of ~100-110 MiB/s on
    decode and ~240-325 MiB/s on encode at 320×240 / 1280×720 — the
    figures aren't a regression baseline yet (the README's reference
    numbers came from 500-iter `cargo bench` runs, not Criterion's
    `--quick` mode), but the same `cargo bench -p oxideav-huffyuv`
    invocation now produces directly comparable per-scenario
    numbers for future optimisation rounds.

- Round-134: `fuzz/` cargo-fuzz harness (`decode_huffyuv`) driving the
  full decode chain — `StreamConfig::parse_bitmapinfoheader`
  (`BITMAPINFOHEADER` + 4-byte extradata prefix + RLE-compressed
  Huffman length tables) followed by `decode_frame` (canonical-Huffman
  build, per-pixel codeword read, LEFT / GRADIENT / MEDIAN predictor
  inverse, RGB decorrelation inverse). The fuzz buffer is split
  `[u16-LE strf-len][strf][frame body]` so a single mutation can perturb
  either the configuration or the frame body; declared rasters above a
  16 MiB cap are skipped in the harness (a resource request, not a
  logic bug). Seeded from six encoder-emitted valid streams (YUY2 /
  RGB24 / RGB32 across LEFT / MEDIAN / decorr methods, v2.x-classic /
  v2.x-custom / v1.x-compat) plus three fuzz-found crash regressions.
  Daily CI via `.github/workflows/fuzz.yml` (org `crate-fuzz.yml`,
  30-minute budget). 60s local baseline post-fix: ~21.7k execs,
  `oom/timeout/crash: 0/0/0`.

### Changed

- Round-181: decoder `inverse_yuy2_left_range` (called by
  `decode_yuy2_field` for the LEFT-only and Median predictor paths
  and by `decode_frame_interlaced` per field) routes through
  `predict::inverse_yuy2_left_macropixel`. Encoder
  `yuy2_residuals` (Left + Gradient methods, all three extradata
  modes) routes through `predict::forward_yuy2_left_subtract`.
  Wire bytes are byte-for-byte identical to round 174 (covered by
  the new equivalence tests + every pre-existing YUY2 round-trip
  + the `lockstep_yuy2_*` AVI-lockstep tests).
- Round-181 microbench (release, M1, single-threaded, isolated
  LEFT pass — no Huffman): inverse YUY2 LEFT at 320×240 dropped
  from ~148 µs → ~31 µs per frame (**≈ 4.7× speedup**); 1280×720
  from ~1.79 ms → ~0.37 ms (**≈ 4.8× speedup**). Forward YUY2
  LEFT at 320×240 dropped from ~40 µs → ~2.4 µs per frame
  (**≈ 16× speedup**, LLVM autovectorises the read-only `src`-only
  body into NEON `vsubq_u8`); 1280×720 from ~490 µs → ~30 µs
  (**≈ 16× speedup**). End-to-end criterion delta on the YUY2-LEFT
  decode bench (320×240) is a more modest 1-2% since the Huffman
  bit-decode dominates total decode time; the LEFT pass was
  already a small fraction of the per-frame budget.

### Fixed

- Round-134 (fuzz): two input-driven panics found by the new
  `decode_huffyuv` harness, both turned into clean `Err` returns:
  - **Degenerate output dimensions.** `decode_{yuy2,rgb24,rgb32}_field`
    guarded the *input* frame length (`< 4`) but allocated the *output*
    raster as `width * height * bpp` and then wrote the uncompressed
    seed pixel into it. A zero width or height produced a raster smaller
    than the seed, panicking on the slice/index write
    (`decoder.rs:219` `range end index 4 out of range for slice of
    length 0`). Now each field decoder rejects a raster too small to
    hold its seed pixel. Also clamped the interlaced bottom-field split
    point (`frame_bytes[top_consumed..]`) so an over-consuming top field
    cannot index past the buffer.
  - **Oversized declared `biSize` with a short buffer.** On the
    `low3 != 0` method path the `biSize <= strf.len()` check was
    skipped, so a header declaring `biSize > 0x29` while supplying only
    40 bytes indexed the missing bpp-override byte at `+0x29`
    (`header.rs:236` `index out of bounds: the len is 40 but the index
    is 41`). The override read now gates on the real buffer length too.
  - Locked in by 5 `decoder.rs` regression tests
    (`degenerate_dims_tests`) and 1 `header.rs` regression test
    (`oversized_bisize_with_short_buffer_no_panic`).

## [0.0.2](https://github.com/OxideAV/oxideav-huffyuv/releases/tag/v0.0.2) - 2026-05-24

### Other

- factor YUY2 forward-median into single-pass predict helper (round 115)
- fuse decorrelation into the GradientDecorr gradient pre-pass (round 103)
- Round 100: fused LEFT+decorrelation encoder residual path
- Round 95: encoder SWAR forward gradient + drop redundant intermediate clones
- Round 91: flat overflow_entries slow path + SWAR gradient post-pass
- Round 7: auto-selector residual reuse + V1xCompat OnceLock cache
- Round 6: encoder predictor auto-selection (bit-cost RDO)
- Round 5 docs: fix memory-reduction headline (~1.5x, not ~2x)
- Round 5: walking-stride interlaced encoder (~2x memory reduction)
- Round 4 fixup: lockstep tests skip on published oxideav-avi 0.0.5
- Round 4: interlaced field-stride=2 + oxideav-avi container lockstep
- drop in-tree AVI walker — container handling lives in oxideav-avi
- Round-3 push: decoder fast-LUT + AVI walker + v1.x compat coverage
- Round-2 encoder push: full HuffYUV/FFVHuff frame encoder
- Vendor table data into src/tables_data/ for crate-self-contained build
- Round-1 8-bit decoder for HuffYUV/FFVHuff streams
- Round 0 — clean-room rebuild scaffold (orphan master)

### Added

- Round-115: YUY2 forward MEDIAN pre-pass factored into a tested
  `predict.rs` helper (single-pass; eliminates the wasted full-frame
  LEFT recompute on the median path).
  - `predict::forward_median_subtract`: the encoder analogue of
    `inverse_median_post` (spec/03 §2.3 / §2.3.2), matching the
    round-95 `forward_gradient_subtract` and round-100/103
    decorrelation-forward factoring. Produces the complete YUY2 median
    residual stream in a single pass: LEFT residuals for row 0 and the
    first 8 wire bytes of row 1 (the §2.3.2 MMX-8-byte
    first-second-row exemption), and MEDIAN residuals
    (`pixel − median3(L, A, G)` with the §2.3 output offsets
    `−2 / −row_stride / −row_stride − 2`) for the remaining bytes.
  - Encoder simplification: the previous `yuy2_residuals` median path
    computed a full-frame LEFT residual stream and then **overwrote**
    the median region with median residuals — recomputing the median
    region's LEFT residuals only to discard them. Round 115 routes the
    Median predictor through `forward_median_subtract`, computing LEFT
    only for the exempt region and median directly for the rest, in one
    traversal.
  - Wire-identical to round 103: the single-pass output is
    byte-for-byte equal to the prior two-phase
    "full-LEFT-then-overwrite" output. Regression-guarded by 5 new
    unit tests in `predict.rs`
    (`round115_forward_median_matches_two_phase`,
    `round115_forward_median_modular_wrap`,
    `round115_forward_median_height_1_is_all_left`,
    `round115_forward_median_short_second_row_is_all_left`,
    `round115_forward_median_roundtrips_via_decoder_model`) plus every
    pre-existing YUY2 median round-trip and the
    `lockstep_yuy2_median_classic` AVI-lockstep test. Lib test count:
    135 → 140.

- Round-103: fused decorrelation+gradient encoder residual path
  (completes the round-100 decorrelation fusion for method `0x41`).
  - `predict::forward_decorr_gradient_subtract`: folds the RGB
    decorrelation transform (`B−G`, `G` identity, `R−G`, and — for
    RGB32 — alpha identity / NOT decorrelated per the spec/03 §2.4
    Validator note) into the spec/03 §2.2.2 forward-gradient
    same-column subtract, computing the gradient pre-pass output
    directly from the caller's un-transformed `pixels` in a single
    pass. Per spec/03 §2.4.2 the gradient+decorrelation residual is
    `c_dec[i] − gradient(c_dec[i−1], c_dec_above[i], c_dec_above_left[i])`
    per decorrelated channel, decomposed (§2.2.2) into a LEFT-pass
    over the row-above differences plus the per-channel LEFT-subtract
    series; the §2.2.2 gradient pre-pass therefore reads only
    decorrelated channel values, which this helper produces without
    materialising them. Row 0 is the decorrelated values verbatim
    (the §2.2.1 first-row LEFT exemption); rows ≥ 1 are the
    same-column decorrelated subtract.
  - Encoder allocation elimination: round-100 fused the LeftDecorr
    path (`0x40`) but the GradientDecorr path (`0x41`) still
    materialised a full-frame `working_owned: Vec<u8>` (the
    decorrelated buffer, `row_bytes × h` bytes/frame) because its
    gradient pre-pass read the decorrelated buffer as input. Round
    103 routes GradientDecorr through `forward_decorr_gradient_subtract`,
    skipping the `working_owned` allocation entirely. With both
    decorrelation paths now fused (round-100 LeftDecorr + round-103
    GradientDecorr), `rgb24_residuals` / `rgb32_residuals` never
    allocate the decorrelated buffer anymore — only the gradient
    `intermediate` (which Gradient / GradientDecorr already required).
    Net saving on the GradientDecorr path: `row_bytes × h` bytes
    per frame.
  - Wire-identical to round 100: the fused gradient pre-pass output
    is byte-for-byte equal to the round-95/100 two-pass
    "materialise the decorrelated `working` buffer, then
    `forward_gradient_subtract` over it" output. Regression-guarded
    by 5 new unit tests in `predict.rs`
    (`round103_fused_decorr_gradient_matches_two_pass_rgb24` /
    `_rgb32`, `round103_fused_decorr_gradient_modular_wrap`,
    `round103_fused_decorr_gradient_alpha_identity_not_decorrelated`,
    `round103_fused_decorr_gradient_height_1_no_op`) plus 4
    end-to-end GradientDecorr round-trips in `roundtrip_tests.rs`
    (RGB24 7×5 CustomV2 non-aligned width, RGB32 6×4 V1xCompat, an
    RGB32 per-row-varying-alpha frame, and an RGB24 interlaced
    height-300 frame), and by every pre-existing GradientDecorr
    round-trip + the AVI-lockstep tests. Lib test count: 126 → 135.

- Round-100: fused LEFT+decorrelation encoder residual path.
  - `predict::forward_left_decorr_residuals`: computes the
    decorrelated-LEFT residuals (`G` identity, `B−G`, `R−G`, and —
    for RGB32 — alpha LEFT-predicted but NOT decorrelated per the
    spec/03 §2.4 Validator note) directly from the caller's
    un-transformed `pixels` in a single fused pass, per spec/03
    §2.4.1 ("the decorrelation transform is fused with the
    predictor at the residual computation, not applied as a
    separate pre-pass … there is no intermediate decorrelated
    buffer"). Mirrors the encoder's documented four-instruction
    fused chain `(cur.B − cur.G) − (prev.B − prev.G)` at
    `system32/huffyuv.dll@0x1000198e..@0x10001996`.
  - Encoder allocation elimination: round-95's `rgb24_residuals` /
    `rgb32_residuals` materialised a full-frame
    `working_owned: Vec<u8>` (the decorrelated buffer, `row_bytes ×
    h` bytes/frame) for *every* decorrelating method and then ran a
    second per-channel LEFT-subtract pass over it. Round 100 routes
    the **LeftDecorr** method (`0x40` = decorrelate + no gradient)
    through the fused helper, skipping the `working_owned`
    allocation entirely. The **GradientDecorr** method (`0x41`)
    still materialises `working` because its gradient pre-pass
    reads the decorrelated buffer as its input — that path is
    unchanged.
  - Wire-identical to round 95: the fused residuals are
    byte-for-byte equal to the round-95 two-pass
    "materialise-then-subtract" output. Regression-guarded by 5 new
    unit tests in `predict.rs`
    (`round100_fused_decorr_matches_two_pass_rgb24` / `_rgb32`,
    `round100_fused_decorr_modular_wrap`,
    `round100_fused_decorr_alpha_left_predicted_not_decorrelated`,
    `round100_fused_decorr_roundtrips_via_inverse_rgb24`) plus 6
    end-to-end LeftDecorr round-trips in `roundtrip_tests.rs`
    (RGB24/RGB32 × ClassicV2 / CustomV2 / V1xCompat, an
    interlaced-height-300 frame, and a per-row-varying-alpha RGB32
    frame), and by the pre-existing
    `lockstep_rgb24_left_decorr_classic` AVI-lockstep test. Lib
    test count: 115 → 126.

- Round-95: encoder forward gradient pre-pass + intermediate
  allocation elimination.
  - `predict::forward_gradient_subtract`: encoder analogue of
    round-91's `inverse_gradient_post`, implementing the
    LEFT-predict-row-0 + per-row gradient-residual subtract layout
    documented at spec/03 §2.2.2 `@0x10001eab..@0x10001f9e` (the
    encoder evidence for the two-phase forward gradient). Same
    SWAR identity as round 91's add, generalised to subtract:
    `(a | MASK_HI).wrapping_sub(b & MASK_LO) ^ ((a ^ !b) & MASK_HI)`
    — byte-wise wrapping subtract with no inter-byte borrow, no
    `unsafe`, no vendor intrinsics. LLVM autovectorises into SSE2
    `psubb` / NEON `vsubq_u8`. Bit-identical to a per-byte
    `wrapping_sub` loop — regression-guarded by 5 new tests
    (aligned-8 stride, unaligned-tail row width, modular-wrap pump,
    height-1 no-op, and forward-then-inverse round-trip).
  - Encoder allocation elimination: the previous
    `yuy2_residuals` / `rgb24_residuals` / `rgb32_residuals`
    paths allocated an `intermediate: Vec<u8>` of size
    `row_bytes × h` per frame even when the predictor was Left /
    Median / PredictOld (a `pixels.to_vec()` clone with no
    transform applied), and an additional `working.clone()` for
    RGB methods when neither decorrelation nor gradient was active.
    Round 95 borrows `pixels` / `working` directly via
    `Option<Vec<u8>>` + `as_deref().unwrap_or(...)` and only
    allocates the intermediate when the gradient pre-pass needs
    one. Saves up to **2 × row_bytes × h** bytes per frame on
    Left / Median / PredictOld YUY2 paths and **1 × row_bytes × h**
    bytes on Left / LeftDecorr / GradientDecorr RGB paths.
  - **Measured (release, M1 host, 500 iters)**:
    - YUY2 320×240 Median: 1.10 ms/frame → 0.64 ms/frame
      (**≈ 1.7× speedup**) — biggest win because Median had two
      back-to-back full-frame Vec clones (`pixels.to_vec()` for
      the non-gradient path + post-LEFT median-overlay pass).
    - YUY2 1280×720 Left: 7.65 ms/frame → 5.09 ms/frame
      (**≈ 1.5× speedup**).
    - RGB24 320×240 Left: 1.00 ms/frame → 0.85 ms/frame (≈ 15%).
    - RGB24 320×240 LeftDecorr: 1.00 ms/frame → 0.86 ms/frame
      (≈ 13%).
    - YUY2 320×240 gradient: 0.52 ms/frame → 0.46 ms/frame
      (≈ 10%) — gradient gets less benefit because the
      gradient intermediate is still required; only the
      pre-existing copy-to-intermediate is now consolidated
      into the forward-subtract helper's single pass.
  - Wire-identical to round 91 — regression-guarded by every
    pre-existing round-trip + AVI-lockstep test (115 lib tests +
    8 lockstep tests all green) plus the 5 new
    `round95_swar_subtract_*` tests in `predict.rs`. Lib test
    count: 110 → 115.

- Round-91: decoder slow-path flat overflow-entries table + SWAR
  gradient post-pass.
  - `tables::HuffTable::overflow_entries`: a flat `Vec<OverflowEntry>`
    precomputed at table-build time. Round 7's `decode_one_slow`
    walked all 256 entries of `table.entries` per overflow byte with
    a `length == 0 || length <= 16 { continue }` short-circuit on
    every iteration — paying full per-iteration cost even on the
    proprietary's classic v2.x blobs that have only 0..2 length-17
    codes. Round 91 emits an `OverflowEntry { code, mask, length,
    symbol }` per long code at build time (with `mask = !0 << (32 -
    L)` pre-baked), and the slow path now iterates exactly that
    slice. For the six classic v2.x blobs (max length ≤ 17,
    typically 0..2 long codes) overflow decode visits ≤ 2 entries
    instead of 256; for v1.x set B (~210 long codes per spec/04
    §4.2 non-canonical layout) the scan drops from 256 → ≤ 210
    with the mask pre-baked. **Measured (release, M1 host, 320×240
    YUY2)**: v1.x median decode 2.81 ms/frame → 1.29 ms/frame
    (**≈ 2.2× speedup**); ClassicV2 gradient YUY2 1.72 ms/frame →
    0.67 ms/frame (**≈ 2.6× speedup**); ClassicV2 gradient-decorr
    RGB24 320×240 ≈ 0.98 ms/frame. Wire-identical to round 7 —
    regression-guarded by tests that compare every long code's
    `decode_one` output against `decode_one_slow` directly.
  - `predict::inverse_gradient_post`: byte-by-byte modular-add
    inner loop replaced by an 8-bytes-per-u64 SWAR add
    (`(a & 0x7F7F…).wrapping_add(b & 0x7F7F…) ^ ((a ^ b) &
    0x8080…)` — textbook byte-wise wrapping add via 64-bit math,
    no `unsafe`, no vendor-intrinsics dependency). Mirrors
    spec/03 §2.2.2's documented MMX 8-byte-wide gradient
    post-pass at `system32/huffyuv.dll@0x10001dfb..@0x10001e8c`.
    LLVM autovectorises the inner u64 loop into SSE2 `paddb` on
    x86_64 and NEON `vaddq_u8` on aarch64. Bit-identical to the
    byte loop on every input — regression-guarded by 4 SWAR
    equivalence tests (aligned-8 stride, unaligned-tail,
    modular-wrap, height-1 no-op) + 3 end-to-end gradient
    round-trip tests at 320×16 across YUY2 / RGB24 / RGB32.
  - Earlier round-91 work also explored a span-replicated
    secondary index keyed on the 16-bit window prefix; benching
    showed it was a net regression vs round 7 on v1.x set B
    (non-canonical codes cluster heavily on prefix 0, turning
    the per-bucket scan into a Vec-indirection-laden near-256-entry
    walk). The flat overflow_entries Vec is the minimal change
    that's strictly better on every input — kept for that reason.
  - Lib test count: 98 → 110 (+12 round-91 tests covering the
    overflow-entries shape, the SWAR equivalence, and end-to-end
    YUY2/RGB24/RGB32 gradient round-trips at 320×16 to exercise
    the new u64 path on realistic row widths).

- Round-7: encoder auto-selector residual reuse + V1xCompat table
  cache.
  - Round 6's `encode_frame_auto` did `N + 1` residual passes (one
    per candidate inside `bit_cost_for_method` PLUS one more inside
    `encode_frame_with_mode` for the winner). Round 7 collapses
    that to `N`: a private `PrecomputedFrame` carrier holds the
    seed(s) + combined body once, scoring runs over the cached
    body, and the winner's body bytes flow straight into a new
    private `encode_with_precomputed` helper. Wire bytes
    bit-identical to round 6 — regression-guarded by 5 new
    auto-vs-explicit drift tests across every legal
    `(family, method, mode)` triple.
  - Measured (release-mode, YUY2 320×240, 300 iters):
    auto CustomV2 1.46 ms → 1.37 ms (≈5%); RGB24 auto CustomV2
    1.65 ms → 1.52 ms (≈9%).
  - V1xCompat per-family `OnceLock` cache for the v1.x
    precomputed-code triple `(slot1, slot2, slot3)`. The codebook
    is deterministic per family (spec/04 §4.1) and each
    `HuffTable` carries a 128 KiB primary LUT — re-baking on
    every encode wasted ~80 µs/frame. Round 7 caches the triple
    behind two `OnceLock`s (YUY2 vs RGB) and hands out clones per
    call, with a regression test that interleaves YUY2/RGB24
    encodes to confirm the per-family slots stay isolated.
  - Measured: YUY2 320×240 V1xCompat 0.47 ms → 0.40 ms (≈16%).
  - Lib test count: 92 → 98.

- Round-6: encoder predictor auto-selection (bit-cost-driven). New
  public API:
  - [`encoder::MethodSelection`] (`Fixed(Method)` / `Auto`) and
    [`encoder::encode_frame_auto`] — runs every legal predictor for
    the family, scores each one with the package-merge optimal
    Huffman length tables, and emits the winner. Returns
    `(strf, frame, chosen_method)` so muxers can record the picked
    method.
  - [`encoder::bit_cost_for_method`] — per-method bit-count estimate
    in `Σ length[s] × count[s]` units against the package-merge
    length tables. Stand-alone helper for callers that want to
    inspect the trade-off without driving a full encode.
  - `MethodSelection::legal_methods(family)` — enumerates the
    deterministic candidate-method order for a family (YUY2: Left /
    Gradient / Median; RGB24/RGB32: Left / LeftDecorr /
    GradientDecorr). `PredictOld` is intentionally omitted from
    `Auto` (wire-distinct but body-identical to `Left`; callers
    needing `PredictOld` on the wire should use
    `MethodSelection::Fixed`).
  - 12 new self-roundtrip tests: auto round-trips on YUY2 /
    RGB24 / RGB32 in both extradata modes (CustomV2 + ClassicV2);
    auto on interlaced heights 290 / 300; auto on a chroma-
    correlated RGB24 synthetic where LeftDecorr provably beats
    Left; auto on a diagonally-textured YUY2; auto-winner ≤
    every-fixed-candidate inequality tests; `MethodSelection::Fixed`
    returns the pinned method back unchanged; `bit_cost_for_method`
    rejects illegal `(family, method)` pairs. Lib test count:
    80 → 92.
- `tables::compute_canonical_lengths` now handles the single-symbol
  histogram case by emitting a length-1 dummy entry on the next
  available symbol, so the canonical builder's Kraft accumulator
  still wraps to zero. The dummy is never emitted (its histogram
  count is 0) and the wire body is unchanged on any frame with
  a multi-symbol residual stream; the change unblocks CustomV2 +
  Auto on degenerate inputs (e.g., constant-luma frames where one
  channel's residuals collapse to a single value).

### Added (round 5)

- Round-5: walking-stride interlaced encoder. The round-4 interlaced
  path materialised two contiguous half-height field buffers via
  `predict::split_fields` plus a third `combined_residuals_for_stats`
  body (clone of both fields' bodies concatenated) for histogram +
  verify. Round 5 replaces both with:
  - One field-sized scratch (`compact_field_rows`) reused for top
    then bot field by walking the source raster with row-stride 2.
  - One combined-body `Vec<u8>` filled in two halves
    (`combined_body[..top_body_len]` = top, rest = bot); the
    histogram + emit passes consume slices of it via new
    slice-based helpers (`compute_lengths_from_body`,
    `verify_body_in_table`, `emit_bitstream_parts`) — no per-field
    `Residuals` clone, no per-field body Vec at emit time.
    Per-family slot-phase alignment is preserved (top body length
    is a multiple of 4 / 3 / 4 for YUY2 / RGB24 / RGB32
    respectively, so the bot half resumes at the same
    `i % cycle == 0` phase).
  - Result: encoder peak working set drops from ~3.5× source-frame
    size (split_top + split_bot + per-field
    intermediate/residuals/body + combined-stats clone) to ~2.5×
    (scratch + per-field intermediate/residuals/body + combined).
    Wire bytes identical to round 4 — regression guarded by 8 new
    round-trip tests covering odd height, 480p-class, `V1xCompat`
    walking-stride, and `CustomV2` walking-stride across all three
    pixel families.
- 8 new round-5 self-roundtrip tests in `roundtrip_tests.rs` exercising
  the walking-stride interlaced encoder (odd height 8×301 across all
  three families; 480p-class 96×480 and 64×480; V1xCompat × Median;
  CustomV2 × LeftDecorr / PredictOld). Bumps the lib test count from
  72 to 80.

### Round-5 follow-ups

- `tests/round4_avi_lockstep.rs` still gates lockstep tests on
  `try_open_mux` because published `oxideav-avi 0.0.5` predates the
  `params.tag = CodecTag::fourcc("HFYU")` generic-FourCC pipeline
  (landed at `oxideav-avi@1c28877` on master). When `oxideav-avi
  0.0.6+` publishes, the skip-gate can drop. Coordination only;
  no encoder change needed.
- HFYU / FFVH fixtures still 404 on samples.oxideav.org; once a
  third-party-encoded fixture lands, a `tests/round5_real_fixture_lockstep.rs`
  can wire the sample-corpus path. Documented gap, deferred.

### Added (round 4)

- Round-4: codec ↔ container lockstep + interlaced field-stride=2:
  - `tests/round4_avi_lockstep.rs` (8 tests) — end-to-end
    encoder → `oxideav-avi` muxer → demuxer → decoder roundtrip on
    synthetic frames, exercising the codec/container interface
    without dragging an AVI walker into this crate. Test-only
    `[dev-dependencies] oxideav-avi` per `docs/IMPLEMENTOR_ROUND.md`
    §"Crate-purpose discipline". Covers YUY2/RGB24 + every predictor,
    `ClassicV2` and `CustomV2` extradata paths, interlaced (height
    > 288), and a two-frame stream.
  - Interlaced field-stride=2 prediction path (spec/02 §2 +
    spec/05 planned) on both encode and decode sides: when
    `biHeight > 288` the encoder splits the frame into two fields
    (even rows = top; odd rows = bottom), predicts each
    independently with the shared per-stream Huffman tables, and
    concatenates `top_seed | top_bits | bot_seed | bot_bits` on
    the wire. Decoder reverses the split via `BitReader::bytes_consumed`
    + `interleave_fields`. 10 new self-roundtrip tests at heights
    288 (boundary), 290, 300, and 320 across all three pixel
    families and four predictor methods, plus the threshold
    constants test in `predict.rs`.
  - `predict::is_interlaced_height`, `predict::split_fields`,
    `predict::interleave_fields` — public helpers usable by
    integration tests verifying the spec/02 §2 trigger.

- Round-3: decoder fast-LUT + v1.x compat for all predictors:
  - `tables::HuffTable::primary_lut` — 65 536-entry primary LUT
    keyed on the top 16 bits of the bit window. Codes ≤ 16 bits hit
    a single indexed load (`(length << 8) | symbol`); codes longer
    than 16 bits route to `decode_one_slow` (per-symbol scan).
    spec/03 §3.2.2. Measured ≈ 3.2× decode speed-up on a 320×240
    YUY2 LEFT/ClassicV2 frame (16.9 ms/frame → 5.3 ms/frame).
  - 9 new v1.x-compat self-roundtrip tests covering all eight
    legal (family, method) pairs that aren't `predict_old`
    (round 2 only tested PredictOld); confirms the v1.x
    precomputed-codes set covers every residual symbol.
  - 2 LUT-coverage tests (`primary_lut_matches_per_symbol_entries`,
    `primary_lut_overflow_falls_back_to_slow_path_v1x_set_b`).

### Removed

- The round-3 in-tree `avi` module (516 LOC) was reverted: AVI
  demuxing is the responsibility of `oxideav-avi`, not the codec
  crate. `oxideav-huffyuv` should only encode/decode HuffYUV /
  FFVHuff frames given raw codec bytes; container handling lives
  one layer up. Future fixture-lockstep tests will consume frames
  from `oxideav-avi` (when wired) rather than re-implement AVI
  walking inside this crate. Same architectural principle that
  moved AVI-codec-tag resolution out of `oxideav-magicyuv` /
  `oxideav-utvideo`.

- Round-2 encoder push: full HuffYUV / FFVHuff frame encoder for the
  three 8-bit pixel families (YUY2, RGB24, RGB32) and all six legal
  predictor methods (`predict_old`, `Left`, `Gradient`, `Median`,
  `LeftDecorr`, `GradientDecorr`):
  - Three extradata paths via `ExtradataMode`:
    - `ClassicV2` (default) — embeds the matching pre-baked classic
      blob in the BIH (the proprietary's default, spec/04 §3).
    - `CustomV2` — derives per-channel histograms from the residual
      stream and runs a length-limited (max-31) package-merge
      Huffman builder, RLE-encoding the resulting length tables in
      line.
    - `V1xCompat` — emits `biSize == 0x28` with no extradata; the
      decoder reads its codebook from the v1.x precomputed-codes set.
  - `build_bitmapinfoheader` public BIH write-side helper, the
    complement to `StreamConfig::parse_bitmapinfoheader` for muxers
    that need to synthesise an AVI `strf`.
  - 22 self-roundtrip tests covering every legal (family, method)
    pair under `ClassicV2`, six (family, method) pairs under
    `CustomV2`, and three (family) v1.x-compat pairs.
  - `tables::compute_canonical_lengths` (package-merge), and
    `tables::rle_encode_one_channel` / `rle_encode_three_channels`
    public encoder utilities.

- Round-1 8-bit decoder implemented from the clean-room workspace
  `docs/video/huffyuv/spec/00..04` + `tables/00..09`:
  - `BITMAPINFOHEADER` + extradata parse for HFYU/FFVH streams,
    including the `bpp_override` byte and v1.x low-3-bits-of-
    `biBitCount` predictor selector.
  - Per-frame decode for YUY2 / RGB24 / RGB32 with all six legal
    predictor methods (`predict_old`, `Left`, `Gradient`, `Median`,
    `LeftDecorr`, `GradientDecorr`).
  - v2.x extradata path (RLE-compressed Huffman length tables +
    longest-length-first canonical build) and v1.x precomputed-codes
    fallback (`tables/06..09`).
  - Self-roundtrip tests for `yuv-left`, `yuv-median`,
    `rgb-left`, `rgb-left-decorr`, `rgb-gradient-decorr`,
    `rgb32-left`, `rgb32-gradient-decorr` at small sizes.
  - `register_codecs(reg)` claims the FourCCs `HFYU` and `FFVH` so
    that `oxideav-avi` resolves a HuffYUV stream's `biCompression`
    straight through `CodecResolver`.

### Changed

- Clean-room rebuild from a fresh orphan `master`. The previous
  implementation was retired by the OxideAV docs audit dated
  2026-05-06; the prior history is preserved on the `old` branch.
