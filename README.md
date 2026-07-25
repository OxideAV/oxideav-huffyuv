# oxideav-huffyuv

[![CI](https://github.com/OxideAV/oxideav-huffyuv/actions/workflows/ci.yml/badge.svg)](https://github.com/OxideAV/oxideav-huffyuv/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/oxideav-huffyuv.svg)](https://crates.io/crates/oxideav-huffyuv) [![docs.rs](https://docs.rs/oxideav-huffyuv/badge.svg)](https://docs.rs/oxideav-huffyuv) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Pure-Rust HuffYUV / FFVHuff lossless video codec for the
[oxideav](https://github.com/OxideAV/oxideav-workspace) framework.

This crate decodes and encodes HuffYUV / FFVHuff frames from raw
codec bytes; AVI / OpenDML container handling lives one layer up in
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
  top-down), RGB24 (24-bit, bottom-up), RGB32 (32-bit, bottom-up).
- **All six predictor methods**: `predict_old` (`-2`), `Left` (`0`),
  `Gradient` (`1`, YUV-only), `Median` (`2`, YUV-only), `LeftDecorr`
  (`0x40`, RGB-only), `GradientDecorr` (`0x41`, RGB-only).
- **Both Huffman-table paths**: v2.x extradata (RLE-compressed length
  tables, canonical Huffman code build) and v1.x precomputed-codes.
- **Interlaced field-stride prediction**: the frame splits into two
  independently-predicted fields and is reassembled on decode. The
  trigger honours the extradata `interlace_flag` byte (BIH `+0x2A`,
  high nibble `1`=interlaced / `2`=non-interlaced) as the **primary**
  indicator when set, falling back to the `biHeight > 288` heuristic
  when the byte is `0x00` (the x86-64 build / clean-room default).
- A 65 536-entry primary lookup table per Huffman table services codes
  ≤ 16 bits in a single indexed load; longer codes fall through to a
  flat overflow table.
- **Arbitrary v2.x extradata** (spec/04 §3.4): the decoder accepts
  *any* set of three RLE-compressed 256-entry length tables in the
  extradata — not only the six classic blobs — deriving per-symbol
  codes by canonical-Huffman construction, and ignores the RGB24 "X"
  pad byte's value (spec/02 §8 #2). A dedicated conformance suite pins
  both contracts.
- **Two-worker interlaced decode** (round 420): the two
  independently-coded fields decode on parallel workers when the
  caller grants a thread budget ≥ 2 (`decode_frame_with_workers`, or
  `ExecutionContext` through the registry `Decoder`). The wire
  carries no bottom-field offset, so the second worker is fed by a
  bit-exact length-only scan of the top field's codeword stream.
  Serial by default; output byte-identical across budgets (golden-pin
  invariance suite + fuzz differential oracle).

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
    §3.3 C-string-terminator convention: length-0 runs are emitted as
    short-form chunks, never long-form), so `lstrcpyA`/`lstrlenA`-based
    third-party tooling can copy/measure the extradata without
    truncating it.
  - **`V1xCompat`** — emits `biSize == 0x28` with no extradata; the
    decoder reads its codebook from the v1.x precomputed-codes set.
- **Predictor auto-selection** (`encode_frame_auto` with
  `MethodSelection::Auto`) runs every legal predictor for the family,
  scores each with `bit_cost_for_method` against the package-merge
  optimal length tables, and emits the winner (returning the chosen
  `Method`). Pin a method via `MethodSelection::Fixed(...)`.
- Public **`build_bitmapinfoheader`** BIH writer for muxers
  synthesising an AVI `strf` payload; emits the i386-style
  `interlace_flag` byte (`0x10`/`0x20`) at BIH `+0x2A`.
- **Interlaced encode** for `biHeight > 288`, walking the source
  raster with row-stride 2 into a single reused scratch field buffer.
- **Two-worker interlaced encode** (round 420,
  `encode_frame_with_mode_workers`): with a thread budget ≥ 2 the two
  fields' residual and bit-pack phases run on parallel workers; wire
  bytes stay byte-identical to the serial encoder.
- **Budgeted auto-selection** (round 429,
  `encode_frame_auto_workers`): the per-candidate scoring fans out
  across the granted workers (serial in-order reduction, so winner /
  tie-break / error order match the serial selector exactly) and the
  winner's emit gets the interlaced two-worker clamp — identical
  `(strf, bytes, method)` across budgets.

### Registry (round 429: full dual-API on both sides)

- Registry `Decoder` **and** `Encoder` implementations wired by
  `register()`, with `make_decoder` / `make_encoder` also exposed
  directly. Both honour `set_execution_context` (serial default;
  budget ≥ 2 engages the field-parallel interlaced paths and the
  parallel auto scoring) with byte-identical output across budgets.
- Encoder options bag (`HuffYuvEncoderOptions`, schema-registered):
  `format` = `auto`|`yuy2`|`rgb24`|`rgb32` (`auto` derives from
  `CodecParameters::pixel_format` — `Yuyv422`/`Bgr24`/`Bgra`),
  `method` = `auto`|`predict-old`|`left`|`gradient`|`median`|
  `left-decorr`|`gradient-decorr` (`auto` bit-cost-selects on the
  first frame and pins the winner — the method byte is stream-level),
  `mode` = `classic-v2`|`custom-v2`|`v1x`.
- `output_params()` carries the AVI `strf` as extradata (final at
  construction for a fixed method with `classic-v2`/`v1x`; finalized
  by the first `send_frame` otherwise). `custom-v2` streams pin the
  first frame's Huffman tables — repeat content re-encodes
  byte-identically, and frames needing symbols the pinned tables
  never coded are rejected instead of emitting an undecodable stream.
- Packets are intra-only keyframes, pts passed through verbatim
  (`dts == pts`), time base from `CodecParameters::frame_rate`.

### FourCCs

Native FourCCs `HFYU` + `FFVH` are registered via `oxideav-core`'s
codec registry (codec id `"huffyuv"`), so `oxideav-avi` resolves a
HuffYUV stream's `biCompression` straight through `CodecResolver`.
The encoder advertises `HFYU` on `output_params().tag` (matching the
`biCompression` it writes).

## Public API

- `decode_frame(&StreamConfig, frame_bytes) -> Result<DecodedFrame>`
- `decode_frame_with_workers(&StreamConfig, frame_bytes, worker_budget)`
  — same output, field-parallel interlaced decode under an
  `ExecutionContext`-style thread budget (`decode_frame` ≡ budget 1)
- `encode_frame` / `encode_frame_with_mode` / `encode_frame_auto`
- `encode_frame_with_mode_workers(..., worker_budget)` — budgeted
  field-parallel interlaced encode, wire-identical to the serial path
- `encode_frame_auto_workers(..., worker_budget)` — budgeted
  candidate scoring + winner emit, identical output across budgets
  (`encode_frame_auto` ≡ budget 1)
- `build_bitmapinfoheader`, `bit_cost_for_method`
- `Method`, `PixelFamily`, `Predictor`, `StreamConfig`,
  `ExtradataMode`, `MethodSelection`
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
~173–247 MiB/s and encode ~233 MiB/s–1.0 GiB/s across the matrix. All
round-419 performance work is output-invariant, pinned by the 78-entry
golden-hash suite in `tests/golden_pins.rs`.

Round 420 adds `ExecutionContext`-budgeted field-parallel interlaced
decode/encode: with 2 granted workers, interlaced 720p decode runs
1.33–1.47× (e.g. YUY2 Left 7.91 → 5.47 ms via a pair-length-LUT
split scan) and encode 1.72–1.99× (RGB32 GradientDecorr
10.78 → 5.43 ms) over the unchanged serial walls — byte-identical
output across budgets. See the worker-scaling table in
[`BENCHMARKS.md`](BENCHMARKS.md).

Round 429 extends the budget to the auto-selector
(`encode_frame_auto_workers`): 720p YUY2 auto encodes drop 6.42 →
3.95 ms (custom, 1.62×) and 6.75 → 3.91 ms (classic, 1.73×) at 2
workers; an interlaced RGB32 960×400 auto encode reaches 2.15×
(6.29 → 2.93 ms) at 3 workers with scoring and emit both fanned out.

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

## Out of scope (deferred)

- 10/12-bit FFVHuff family — excluded from the clean-room reference
  workspace; only the original `HFYU` FourCC HuffYUV is targeted.
- Third-party-fixture lockstep — no `HFYU` / `FFVH` sample fixtures
  are available, so the lockstep test runs against synthetic encoder
  output. The interlaced wire format is self-roundtrip-correct;
  bit-exact lockstep against a third-party interlaced encoder awaits
  the canonical interlace spec chapter + a fixture.

## License

MIT — see [`LICENSE`](LICENSE).
