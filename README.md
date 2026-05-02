# oxideav-huffyuv

Pure-Rust **HuffYUV** / **FFVHuff** lossless intra-only video decoder, part of
the [oxideav](https://github.com/OxideAV) framework.

HuffYUV (FourCC `HFYU`, Ben Rudiak-Gould, 2000) and its FFmpeg-introduced
sibling FFVHuff (FourCC `FFVH`) are lossless intra-only video codecs typically
wrapped in AVI. Each frame is encoded by spatially predicting samples from
already-decoded neighbours and entropy-coding the residual with a per-channel
canonical Huffman code. The crate decodes both bitstream variants from the
same reverse-engineered behaviour-trace spec described in
`docs/video/huffyuv/huffyuv-trace-reverse-engineering.md` (the "trace doc").

## Status

This is the initial bootstrap drop. Decode coverage:

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
| YUV 4:1:1 planar (v3, 8-bit)           | implemented   |
| Gray 8 (v3)                            | implemented   |
| RGB 24 packed + decorrelate (v2)       | implemented   |
| BGRA 32 packed + decorrelate (v2)      | implemented   |
| Per-frame Huffman tables (FFVHuff `-context 1`) | implemented |
| GBRP / GBRAP planar (v3)               | implemented   |
| ≥9-bit depths (v3)                     | not yet       |
| 15/16-bit Huffman + 2 raw bits split   | not yet       |
| Encoder                                | not yet       |
| Interlaced bootstrap                   | not yet       |
| `hymt` slice variant                   | not in scope  |

The crate is **decode-only** today and exposes itself through
`oxideav-core`'s `CodecRegistry`. Encoding is a future work item.

## Bitstream notes

Reproduced from the trace doc; see that document for full justification:

- **Container**: Microsoft AVI. `strf` payload = `BITMAPINFOHEADER` (40 B) +
  Huffyuv-specific extradata. `biCompression` = `'HFYU'` (0x55594648) for
  HuffYUV, `'FFVH'` (0x48564646) for FFVHuff.
- **Extradata header** (4 bytes after the BIH):
  - `extra[0]` — predictor (low 6 bits) | decorrelate (bit 6).
  - `extra[1]` — v2: bitstream-bpp identifier (12/16/24/32).
              v3: `(bps-1)<<4 | chroma_h_shift | chroma_v_shift<<2`.
  - `extra[2]` — v3 channel flags + interlace + `bit 6` per-frame tables.
  - `extra[3]` — version tag: 0 = v2, 1 = v3.
- **Code-length table RLE**: `b & 0x1F` = code length, `b >> 5` = run; if run
  is zero, the next byte gives a 1..255 long run.
- **Canonical Huffman** assignment (shortest codes first, lexicographic
  within length). The wire never carries codes, only lengths.
- **Predictors**: LEFT (1-D prefix sum across rows), GRADIENT (left + above −
  above-left), MEDIAN (Paeth: median of L, T, L+T−TL).
- **Decorrelate** (RGB only, when `extra[0] & 0x40`): Huffman-coded residuals
  are (B−G), G, (R−G); the decoder inverts this after spatial prediction.
- **Frame layout**: per-format prelude (4 raw bytes for v2 YUV/RGB, none for
  v3 planar), Huffman residuals, 31-bit zero tail. Whole packet is
  byte-swapped within every 32-bit word; we undo that before bit-reading.
- **Per-frame tables** (`extra[2] & 0x40`, FFVHuff only): a fresh RLE-coded
  length-table set is prepended to every `00dc` payload, replacing the
  extradata tables for that frame only.

## Edge cases

- **Width constraints** — yuv420/422 require even width; yuv422 + MEDIAN
  requires width ≡ 0 mod 4 (the median row-1 bootstrap consumes the first
  4 luma + 2 chroma pairs as left-predicted samples).
- **RGB scan order** — RGB streams are stored bottom-up in the AVI; row 0
  in the bitstream is the bottom-left pixel of the displayed frame.
- **Median row-1 bootstrap** — for v2 YUV 4:2:2 the first 4 luma + 2
  chroma pairs of row 1 are LEFT-predicted; the rest of row 1 then runs
  median against row 0. For v3 planar formats, row 0 is fully LEFT and
  rows 1..H-1 are MEDIAN.
- **Joint VLC** — the upstream codec builds a 12-bit joint lookup for
  speed; we use plain single-symbol lookups (the trace doc confirms they
  are bit-identical).

## Round-trip tests

- `tests/synth_left.rs` — hand-built 4×4 yuv422p stream (LEFT predictor)
  reproduces a known sample matrix bit-exactly.
- `tests/ffmpeg_interop.rs` — invokes `ffmpeg -c:v huffyuv -pix_fmt
  yuv422p -f avi …` on a tiny generated PNG, then decodes the AVI through
  this crate and asserts every plane sample matches what `ffmpeg
  -i … -c:v rawvideo` produces. The test is `#[ignore]`d when no
  `ffmpeg` binary is on `$PATH`, so it is opt-in for CI.

## License

MIT (see `LICENSE`).
