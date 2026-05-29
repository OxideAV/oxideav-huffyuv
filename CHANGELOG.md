# Changelog

All notable changes to this crate are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
