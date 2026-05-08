# Changelog

All notable changes to this crate are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
