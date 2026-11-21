# oxideav-huffyuv

[![CI](https://github.com/OxideAV/oxideav-huffyuv/actions/workflows/ci.yml/badge.svg)](https://github.com/OxideAV/oxideav-huffyuv/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/oxideav-huffyuv.svg)](https://crates.io/crates/oxideav-huffyuv) [![docs.rs](https://docs.rs/oxideav-huffyuv/badge.svg)](https://docs.rs/oxideav-huffyuv) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Pure-Rust HuffYUV lossless video codec for the
[oxideav](https://github.com/OxideAV/oxideav-workspace) framework:
the `HFYU` format (packed 4:2:2, RGB24, RGB32 with alpha; 8-bit) in
full, plus `FFVH`-tagged streams that stay inside that layout. The
`FFVH` extensions (planar and high-bit-depth formats, per-frame
tables) are refused — see *FourCCs*.

This crate decodes and encodes HuffYUV frames from raw codec bytes; AVI / OpenDML container handling lives one layer up in
`oxideav-avi`. End-to-end (encode → AVI mux → AVI demux → decode) is
exercised in `tests/round4_avi_lockstep.rs` via a test-only
`[dev-dependencies] oxideav-avi`.

Implemented clean-room from the strict-isolation reference workspace
at
[`docs/video/huffyuv/`](https://github.com/OxideAV/docs/tree/master/video/huffyuv).
No external library source consulted.

## Capabilities

### Decode

- **`BITMAPINFOHEADER` + extradata parse**, including the
  `biSize ∈ {0x28, > 0x28}` / low-3-bits-of-`biBitCount` shortcuts
  and the `bpp_override` byte.
- **Per-frame decode** for the 8-bit pixel families: YUY2 (16-bit,
  top-down), RGB24 (24-bit, bottom-up), RGB32 (32-bit with alpha,
  bottom-up).
- **All six predictor methods**: `predict_old` (`-2`), `Left` (`0`),
  `Gradient` (`1`, YUV-only), `Median` (`2`, YUV-only), `LeftDecorr`
  (`0x40`, RGB-only), `GradientDecorr` (`0x41`, RGB-only).
- **Both Huffman-table paths**: v2.x extradata (RLE-compressed length
  tables, canonical Huffman code build) and v1.x precomputed-codes.
- **Field-coded ("interlaced") frames** per spec/06: the extradata
  interlace byte (BIH `+0x2A`) decides when its high nibble is `1`
  (fields) or `2` (plain raster), otherwise `biHeight > 288` does. A
  field-coded frame has the same single first pixel and single
  codeword stream as a plain one; only the predictors' "above"
  reference moves two frame rows back. Odd heights are handled.
- A 65 536-entry primary lookup table per Huffman table services codes
  ≤ 16 bits in a single indexed load; longer codes fall through to a
  flat overflow table. Full pair LUTs resolve two codewords per 32-bit
  load on the decode critical path, for every family (RGB24 walks two
  pixels per step so its third codeword pairs across the pixel
  boundary).
- **Arbitrary v2.x extradata** (spec/04 §3.4): the decoder accepts
  *any* set of three RLE-compressed 256-entry length tables in the
  extradata — not only the six classic blobs — deriving per-symbol
  codes by canonical-Huffman construction, and ignores the RGB24 "X"
  pad byte's value (spec/02 §8 #2).

### Encode

- **Frame encoder** for the same six methods × three pixel families
  (12 legal `(family, method)` pairs — RGB carries no `Median`, YUV
  carries no decorr methods).
- Three extradata paths via `ExtradataMode`:
  - **`ClassicV2`** (default) — embeds the matching pre-baked classic
    blob (spec/04 §3).
  - **`CustomV2`** — builds per-channel histograms from the residual
    stream, runs a length-limited (max-31) package-merge Huffman
    builder, and RLE-encodes the resulting length tables inline. The
    RLE stream is guaranteed free of in-band `0x00` bytes (spec/04
    §3.3 C-string-terminator convention).
  - **`V1xCompat`** — emits `biSize == 0x28` with no extradata; the
    decoder reads its codebook from the v1.x precomputed-codes set.
- **Predictor auto-selection** (`encode_frame_auto` with
  `MethodSelection::Auto`) runs every legal predictor for the family,
  scores each with `bit_cost_for_method` against the package-merge
  optimal length tables, and emits the winner (returning the chosen
  `Method`). Pin a method via `MethodSelection::Fixed(...)`.
  `encode_frame_auto_workers` fans the candidate scoring out across a
  caller-granted thread budget with identical output.
- **Multi-worker encode** under a caller-granted budget
  (`encode_frame_with_mode_workers`, `EncodeOptions::worker_budget`,
  or `ExecutionContext` through the registry): the residual body is
  cut into one piece per worker, packed in parallel and spliced back
  bit-exactly. Serial by default; small frames stay serial.
- Public **`build_bitmapinfoheader`** BIH writer for muxers
  synthesising an AVI `strf` payload; writes the interlace byte
  `0x10` / `0x20` at BIH `+0x2A` (spec/06 §1.2).
- **Field coding** for `biHeight > 288` by default (spec/06 §1.1),
  with the matching interlace byte, so the stream decodes the same
  way under the height rule and under the byte rule. `FieldCoding`
  (`Auto` / `Fields` / `Plain`, via `EncodeOptions` +
  `encode_frame_with_options`, or the registry option
  `field-coding`) overrides it for interlaced material of 288 lines
  or fewer and progressive material above; such streams rely on the
  interlace byte (spec/06 §1.3) and need a v2.x header.

### Registry

- Registry `Decoder` **and** `Encoder` implementations wired by
  `register()`, with `make_decoder` / `make_encoder` also exposed
  directly. Both honour `set_execution_context` (serial default; on
  the encoder a budget ≥ 2 engages the piecewise frame encode and the
  parallel auto-selector scoring) with byte-identical output across
  budgets.
- Encoder options bag (`HuffYuvEncoderOptions`, schema-registered):
  `format` = `auto`|`yuy2`|`rgb24`|`rgb32` (`auto` derives from
  `CodecParameters::pixel_format` — `Yuyv422`/`Bgr24`/`Bgra`),
  `method` = `auto`|`predict-old`|`left`|`gradient`|`median`|
  `left-decorr`|`gradient-decorr` (`auto` bit-cost-selects on the
  first frame and pins the winner — the method byte is stream-level),
  `mode` = `classic-v2`|`custom-v2`|`v1x`,
  `field-coding` = `auto`|`fields`|`plain`.
- `output_params()` carries the AVI `strf` as extradata (final at
  construction for a fixed method with `classic-v2`/`v1x`; finalized
  by the first `send_frame` otherwise). `custom-v2` streams pin
  tables fitted to the first frame and blended with the classic
  distribution so every symbol has a code: later frames always
  encode, a frame like the first costs 0.3–0.5 % over its own
  optimum, and a frame unlike it stays near the classic size.
- Packets are intra-only keyframes, pts passed through verbatim
  (`dts == pts`), time base from `CodecParameters::frame_rate`.
- Frames are top row first for all three families: the registry path
  converts to and from the bottom-up RGB wire raster (the direct API
  works on the wire raster as-is). The decoder announces size and
  pixel format through `output_params()`; the encoder accepts planes
  with row padding.

### FourCCs

Native FourCCs `HFYU` + `FFVH` are registered via `oxideav-core`'s
codec registry (codec id `"huffyuv"`). The encoder writes `HFYU`.

`FFVH` is an extended variant of the format with no specification in
the clean-room reference workspace. A stream tagged `FFVH` is decoded
only when its header stays byte-for-byte inside the `HFYU` layout
(packed 4:2:2 / RGB24 / RGB32, 8-bit, stream-level tables); every
other `FFVH` stream — planar 4:2:0 / 4:1:0 / 4:1:1 / 4:4:4 / 4:4:0,
gray, alpha planes, 9–16-bit, planar RGB, per-frame tables — is
refused with `Error::Unsupported` rather than mis-decoded.

## Interoperability (live black-box matrix)

`tests/blackbox_matrix.rs` generates and checks two sweeps against
the `ffmpeg` command-line tool used as an opaque binary (the test
skips when the tool is absent; nothing large is committed):

| Sweep | Cells | Result |
|-------|------:|--------|
| Foreign → ours: both black-box encoders × 3 packed formats × 3 predictors × field flag on/off × 13–18 rasters (2×1 … 720×576, odd widths and heights, both sides of 288 lines) × 1–3 frames | 380 writable `HFYU`-layout streams | **380 / 380 bit-exact** against the source raster (and against the black-box decoder wherever it accepts its own stream) |
| Foreign → ours: `FFVH`-only variants (23 extra pixel formats × 3 predictors × per-frame tables on/off, plus per-frame tables on the packed formats) | 180 writable streams | **180 / 180 refused** cleanly (unspecified extensions) |
| Ours → foreign: 3 families × 4 methods × 3 extradata modes × 12–13 rasters, each also with the field-coding choice forced against the height rule where the header can carry it | 536 comparable streams | **536 / 536 accepted without a diagnostic and bit-exact** |
| Registry path both ways: 3 families, 36×290 field-coded, 3 frames; foreign file → registry frames, registry packets (padded rows, 3 extradata modes) → black-box decoder | 3 + 9 streams | **12 / 12 bit-exact, frames top row first** |

Not comparable (128 further cells): method byte `-2` (`predict_old`)
next to extradata. spec/01 §3.1 makes it legal and equivalent to
`left`; the black-box decoder reads that byte as predictor flags and
rejects or mis-reads it. Use `left` when the stream must play in other
decoders; `predict_old` without extradata (`v1x`) is accepted.

`tests/foreign_pins.rs` pins six tiny third-party streams as bytes so
the fixed bugs stay fixed without the tool installed. The nine staged
third-party fixture files of the reference workspace
(`docs/video/huffyuv-fixtures/`, two frames each) also decode
bit-exactly: 18 / 18 frames, up from 2 / 18 before round 493.

Encoded size against the black-box encoder on the same rasters is in
[`BENCHMARKS.md`](BENCHMARKS.md).

## Public API

- `decode_frame(&StreamConfig, frame_bytes) -> Result<DecodedFrame>`
- `decode_frame_with_workers(&StreamConfig, frame_bytes, worker_budget)`
  — same output at every budget (a frame is one codeword stream, so
  decode has a single unit of work)
- `encode_frame` / `encode_frame_with_mode` / `encode_frame_auto`
- `encode_frame_with_options(family, width, height, pixels,
  &EncodeOptions) -> Result<EncodedFrame>` — method selection,
  extradata mode, field coding and thread budget in one record
- `encode_frame_with_mode_workers(..., worker_budget)` — splits the
  frame across the granted workers, same wire bytes at every budget
- `encode_frame_auto_workers(..., worker_budget)` — budgeted
  candidate scoring, identical output across budgets
  (`encode_frame_auto` ≡ budget 1)
- `build_bitmapinfoheader`, `build_bitmapinfoheader_field_coded`,
  `bit_cost_for_method`
- `Method`, `PixelFamily`, `Predictor`, `StreamConfig`,
  `ExtradataMode`, `MethodSelection`, `FieldCoding`, `EncodeOptions`,
  `EncodedFrame`
- `register` / `register_codecs` / `make_decoder` / `make_encoder` /
  `HuffYuvEncoderOptions` / `CODEC_ID_STR`

## Cargo features

- **`registry`** (default): wires the crate into `oxideav-core`'s
  codec registry. `default-features = false` builds the codec
  standalone with no `oxideav-core` dependency.

## Performance

The full Criterion suite and the ranked hotspot table live in
[`BENCHMARKS.md`](BENCHMARKS.md). Four bench targets
(`benches/{decode,encode,roundtrip,tables}.rs`) cover the
`(family, method, extradata-mode, raster)` matrix up to 1280×720;
inputs are synthesised on the fly from a deterministic xorshift32 +
diagonal-gradient pattern (no committed fixtures). On Apple Silicon
(post-round-419: built-tables cache shared by decode + encode, paired
symbol reads, 64-bit-accumulator `BitWriter`, count-based
package-merge) YUY2-LEFT 320×240 decodes at ~0.64 ms/frame
(~230 MiB/s) and encodes at ~0.14 ms/frame (~1.0 GiB/s); decode runs
~173–247 MiB/s and encode ~233 MiB/s–1.0 GiB/s across the matrix.
Performance work is output-invariant against the 78-entry golden-hash
suite in `tests/golden_pins.rs` (re-pinned in round 493 after the
wire-format corrections; the new pins are the bytes the black-box
decoder accepts).

The two-worker field-parallel decode / encode figures published for
rounds 420–440 are withdrawn: they measured a two-independent-fields
layout that this crate alone read and wrote (see *Interoperability*).
A real field-coded frame is one codeword stream with no split point.

Round 429 extends the budget to the auto-selector
(`encode_frame_auto_workers`): 720p YUY2 auto encodes drop 6.42 →
3.95 ms (custom, 1.62×) and 6.75 → 3.91 ms (classic, 1.73×) at 2
workers (measured when only the candidate scoring fanned out).

Round 429 also lands the r419 serial-decode lever: full pair LUTs
(16-bit window → both symbols + combined length in one 32-bit load,
lazily built per stream behind the table cache) cut serial decode
23–33% across the gates — YUY2 Left 320×240 0.64 → 0.43 ms
(345 MiB/s), YUY2 Left 720p interlaced 7.73 → 5.31 ms, RGB24
LeftDecorr 1.26 → 0.95 ms.

Round 440 adds **RGB24 phase-flip pairing** (two pixels per decode
step so the third codeword pairs across the pixel boundary — 320×240
LeftDecorr −8.4% serial) and **lean CustomV2 encoder tables** (the
emit path reads only `(code, length)` entries, so the per-frame 64-Ki
decode-LUT bake is skipped — 320×240 auto-custom encodes −10.0% YUY2
/ −7.1% RGB24).

Round 493 (same-session A/B against the pre-phase commit, minimum of
4 runs each on a shared host, byte-identical output): the residual
passes became single fused kernels — serial encode **1.5–1.9×** for
YUY2 Median, **1.2–1.3×** for decorrelated RGB at 1280×720 — and the
encoder now splits **any** large frame across the granted workers
(piecewise residuals, per-piece bit packing, bit strings spliced at
their bit offsets): a 1280×720 frame at 4 workers encodes in 1.05 ms
(YUY2 Left, 2.8×), 1.06 ms (YUY2 Median, 4.8×) and 2.14 ms (RGB32
GradientDecorr, 4.7×) against 2.9 / 5.2 / 9.7 ms before. Decode:
YUY2 Median 1.1×, the rest unchanged (decode cannot be split — one
codeword stream). Against the black-box encoder on the same picture
and predictor, `custom-v2` frames are 4–46 % smaller and `classic-v2`
frames 1 % smaller to 37 % larger. Tables in
[`BENCHMARKS.md`](BENCHMARKS.md).

## Fuzzing

`fuzz/` ships three cargo-fuzz targets driven daily by
`.github/workflows/fuzz.yml`:

- `decode_huffyuv` — the full decode chain
  (`parse_bitmapinfoheader` → `decode_frame`) on arbitrary bytes,
  framed `[u16-LE strf-len][strf][frame body]`.
- `encode_huffyuv` — the encode path on fuzz-derived rasters +
  options.
- `tables_huffyuv` — the table-build primitives directly (RLE
  encode/decode, canonical length computation, table build, and the
  v1.x table builder) with self-consistency contracts.

The contract is panic-/OOM-freedom on any input; declared rasters
above a size cap are skipped as an expected resource request, not a
logic bug.

## Not supported

- The `FFVH` extensions listed under *FourCCs* (planar and
  high-bit-depth formats, alpha planes, per-frame tables): the
  clean-room reference workspace specifies only `HFYU`.
- Top-down (negative `biHeight`) streams are read with the absolute
  height; spec/06 §1.5 leaves their handling unestablished.

## License

MIT — see [`LICENSE`](LICENSE).
