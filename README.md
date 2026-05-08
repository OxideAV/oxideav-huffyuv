# oxideav-huffyuv

Pure-Rust HuffYUV / FFVHuff lossless video codec for the
[oxideav](https://github.com/OxideAV/oxideav-workspace) framework.

## Status

**Round 4 — interlaced field-stride=2 + container lockstep.**
Rounds 1 (decoder), 2 (encoder), 3 (decoder fast-LUT), and 4
(interlace + lockstep) all ship from the strict-isolation
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
