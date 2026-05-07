# oxideav-huffyuv

Pure-Rust HuffYUV / FFVHuff lossless video codec for the
[oxideav](https://github.com/OxideAV/oxideav-workspace) framework.

## Status

**Round 2 — encoder push.** Round 1 (decoder) and round 2 (encoder)
both ship from the strict-isolation clean-room workspace at
[`docs/video/huffyuv/`](https://github.com/OxideAV/docs/tree/master/video/huffyuv).
The previous (pre-orphan) implementation was retired alongside the
docs audit dated 2026-05-06; the prior history is preserved on the
`old` branch.

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

## Out of scope (deferred)

- 10/12-bit FFVHuff family.
- Interlaced field-stride=2 prediction (`biHeight > 288`).

## Cargo features

- **`registry`** (default): wire the crate into `oxideav-core`'s
  codec registry. `default-features = false` builds the decoder
  standalone with no `oxideav-core` dependency.
