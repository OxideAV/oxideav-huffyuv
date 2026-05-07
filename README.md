# oxideav-huffyuv

Pure-Rust HuffYUV / FFVHuff lossless video codec for the
[oxideav](https://github.com/OxideAV/oxideav-workspace) framework.

## Status

**Round 1 — clean-room rebuild.** The crate's `master` branch is a
fresh orphan; the previous implementation was retired alongside the
docs audit dated 2026-05-06 (the source-of-record trace document for
this codec did not satisfy clean-room separation). This round-1 build
implements the wire format from the strict-isolation clean-room
workspace at [`docs/video/huffyuv/`](https://github.com/OxideAV/docs/tree/master/video/huffyuv).

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

## Out of scope (deferred)

- 10/12-bit FFVHuff family.
- Interlaced field-stride=2 prediction (`biHeight > 288`).
- Encoder is round-1 self-test only; a public encoder API will land
  with the Auditor's lockstep trace harness in round 2+.

## Cargo features

- **`registry`** (default): wire the crate into `oxideav-core`'s
  codec registry. `default-features = false` builds the decoder
  standalone with no `oxideav-core` dependency.
