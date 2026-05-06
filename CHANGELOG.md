# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.3](https://github.com/OxideAV/oxideav-huffyuv/compare/v0.0.2...v0.0.3) - 2026-05-06

### Other

- prepend retirement notice (docs audit 2026-05-06)
- registry calls: rename make_decoder/make_encoder → first_decoder/first_encoder
- fix rustfmt formatting in decoder.rs and ffmpeg_interop.rs
- update lib.rs doc to reflect completed features
- add gbrp/gbrap, yuv411p, interlaced decode+encode

## [0.0.2](https://github.com/OxideAV/oxideav-huffyuv/compare/v0.0.1...v0.0.2) - 2026-05-04

### Other

- move 0.0.1-followup CHANGELOG block under [Unreleased]
- add encoder + 9..16-bit decode + ffmpeg cross-decode suite
- release v0.0.1

### Added

- **GBRP / GBRAP planar decode + encode** — v3 planar G→B→R[→A] layouts at
  8 / 10 / 12-bit depths (stored in `Gbrp10Le` / `Gbrap10Le` / `Gbrp12Le` /
  `Gbrap12Le` PixelFormat containers). All three predictors supported.
  Cross-decoded against ffmpeg's `gbrp` / `gbrap` / `gbrp10le` formats;
  self-roundtrip verified.
- **YUV 4:1:1 planar decode + encode** — v3 `chroma_h_shift=2` layout
  (`Yuv411P`). Plane-sequential bitstream; all three predictors.
  Cross-decoded against `ffmpeg -c:v ffvhuff -pix_fmt yuv411p`.
- **Interlaced bootstrap for decode** — when `InterlaceMode::Interlaced`
  (or `AutoByHeight` with height > 288), row 1 is fully LEFT-predicted
  as a fresh bottom-field start; for MEDIAN the row-1 partial-left
  bootstrap shifts to row 2. Applies to both v2 (YUV 4:2:2) and v3
  planar paths. Cross-decoded against
  `ffmpeg -flags +ilme+ildct -top 1 -c:v huffyuv`.
- **Encoder** — bit-exact `huffyuv` / `ffvhuff` frame encoder.
  Supports v2 yuv422p / yuv420p / rgb24 / bgra and v3 yuv444p / yuv422p /
  yuv420p / yuv411p / yuv444p10le / yuv422p10le / yuv420p10le /
  yuv444p12le / yuv422p12le / yuv420p12le / gray8 / gray10le / gray12le /
  gray16le / gbrp10le / gbrap10le / gbrp12le / gbrap12le.
  Per-frame histogram-driven canonical Huffman tables (`-context 1`
  style); LEFT / GRADIENT / MEDIAN predictors; auto v2 vs v3 extradata
  selection via the `huffyuv` vs `ffvhuff` codec id. Cross-decode
  round-trips through ffmpeg are verified for yuv422p (LEFT, PLANE),
  yuv420p (LEFT), gray8 (PLANE), yuv411p (LEFT), gbrp10le (LEFT).
- **High-bit-depth decode** — v3 planar 10 / 12 / 16-bit support via
  `yuv{420,422,444}p{10,12}le` / `gray{10,12,16}le` PixelFormats.
  Includes the trace-doc §5.5 / §9.7 "(sample >> 2) Huffman + 2 raw
  bits" splice for 15 / 16-bit alphabets.
- **Predictor cross-row threading** — both `top_left` and `left`
  registers are now threaded across rows for GRADIENT and MEDIAN
  (matches the upstream encoder's per-frame-per-plane register
  carry; verified against `ffmpeg -pred plane` and `-pred median`
  cross-decode tests).
- **Modular Paeth median** — fix `L+T-TL` to wrap mod 2^bps before
  the order-statistic median picks the middle (the previous
  i32-clamp implementation produced wrong predictions whenever the
  pseudo-`c` argument overflowed u8 / u16).
- **Bit writer + canonical-Huffman length builder** — new
  `bitwriter`, `length_builder`, and `rle_encode` modules supporting
  the encoder.

### Fixed

- **GRADIENT predictor** — the previous decoder applied the wrong
  per-sample formula (`+= top[x] - top[x-1]` inside a phase-2 walk
  instead of the canonical `pred = left + top[x] - top_left` per
  sample); rewritten to the canonical PLANE form. The pre-fix
  decoder happened to round-trip with our previous broken encoder
  (which used the same wrong formula) but never decoded a real
  ffmpeg-encoded `-pred plane` bitstream correctly.
- **v3 extradata `chroma` flag** — `chroma` is now derived as
  `yuv || (extra[2] & 0x02)` (matching the empirically-observed
  ffmpeg encoder, which only sets bit 1 for the non-yuv multi-plane
  layouts gbrp / gbrap; yuv layouts leave bit 1 clear and rely on
  bit 0 to imply 3 planes).

### Pixel formats

| Feature                                | Status        |
| -------------------------------------- | ------------- |
| AVI extradata parser (v2 + v3)         | implemented   |
| RLE-coded canonical-Huffman lengths    | implemented   |
| Predictor: LEFT                        | implemented   |
| Predictor: GRADIENT (PLANE)            | implemented   |
| Predictor: MEDIAN                      | implemented   |
| YUV 4:2:2 planar (v2 `bsbpp=16`)       | implemented   |
| YUV 4:2:0 planar (v2 `bsbpp=12`)       | implemented   |
| YUV 4:4:4 planar (v3, 8-bit)           | implemented   |
| YUV 4:1:1 planar (v3, 8-bit)           | implemented (decode + encode) |
| Gray 8 (v3)                            | implemented   |
| RGB 24 packed + decorrelate (v2)       | implemented   |
| BGRA 32 packed + decorrelate (v2)      | implemented   |
| Per-frame Huffman tables (FFVHuff)     | implemented   |
| GBRP 8/10/12-bit planar (v3)           | implemented (decode + encode) |
| GBRAP 8/10/12-bit planar+alpha (v3)    | implemented (decode + encode) |
| 10/12-bit YUV 4:2:0/4:2:2/4:4:4        | implemented   |
| 10/12/16-bit Gray (incl. 2-raw splice) | implemented   |
| **Encoder** (LEFT / GRADIENT / MEDIAN) | implemented   |
| Interlaced bootstrap                   | implemented (decode)          |
| `hymt` slice variant                   | not in scope  |

## [0.0.1](https://github.com/OxideAV/oxideav-huffyuv/releases/tag/v0.0.1) - 2026-05-03

### Other

- replace never-match regex with semver_check = false
- bootstrap clean-room HuffYUV / FFVHuff decoder
- Initial commit
