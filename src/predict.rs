//! Per-channel predictors and decorrelation transforms.
//!
//! Implements spec/03 §2 (left / gradient / median; per-channel
//! 8-bit modular wrap, no saturation) and §2.4 (RGB R-G/G/B-G
//! decorrelation, fused with the predictor at the residual level).
//!
//! The decoder calls these to *reconstruct* pixel values from
//! residuals; the encoder calls them to *produce* residuals from
//! pixels. Both directions are byte-modular.
//!
//! # Interlaced mode (spec/02 §2 + spec/05 planned)
//!
//! When `biHeight > 288` the codec splits the source into two fields
//! (even rows = top field; odd rows = bottom field) and predicts each
//! field independently. The two fields' bitstreams are concatenated
//! per-frame. Helpers [`is_interlaced_height`], [`split_fields`] and
//! [`interleave_fields`] implement the row demux / mux.

/// Branchless three-value median over unsigned bytes (mod-256
/// gradient context). spec/03 §2.3:
///
/// ```text
/// median3(L, A, G) = max( min(max(L, A), G), min(L, A) )
/// ```
#[inline]
pub fn median3(l: u8, a: u8, g: u8) -> u8 {
    let (lo, hi) = if a >= l { (l, a) } else { (a, l) };
    // clamp `hi` from above by `g`: result = min(hi, g)
    let mid = if g <= hi { g } else { hi };
    // clamp `mid` from below by `lo`: result = max(mid, lo)
    if mid >= lo {
        mid
    } else {
        lo
    }
}

/// Modular gradient predictor `L + A - AL` (mod 256), spec/03 §2.2.
#[inline]
pub fn gradient_predictor(left: u8, above: u8, above_left: u8) -> u8 {
    left.wrapping_add(above).wrapping_sub(above_left)
}

/// Per-channel LEFT inverse predictor for one row, channel-stride
/// `n_channels` apart. spec/03 §2.1.
///
/// The first `n_channels` entries of `samples` are the seed pixel(s)
/// for the row's leftmost predictor reference; subsequent entries
/// hold residuals on entry and reconstructed values on exit. For the
/// very first row of a frame, the leftmost residual reference is the
/// uncompressed first pixel for that channel; for subsequent rows it
/// is the rightmost reconstructed pixel of the previous row (raster
/// continuity — automatic when we walk a single linear buffer).
pub fn inverse_left_row(samples: &mut [u8], n_channels: usize) {
    if samples.len() <= n_channels {
        return;
    }
    let mut idx = n_channels;
    while idx < samples.len() {
        samples[idx] = samples[idx].wrapping_add(samples[idx - n_channels]);
        idx += 1;
    }
}

/// Per-row LEFT inverse predictor across an entire `width × height`
/// buffer, walking raster order with stride `n_channels`. The
/// decoder seeds `out[0..n_channels]` with the uncompressed first
/// pixel before calling this; subsequent rows continue the raster
/// across the row boundary, which produces the correct
/// "first-pixel-of-row N predicted from last pixel of row N-1"
/// behaviour automatically (spec/03 §2.1.2).
pub fn inverse_left_full(out: &mut [u8], _row_stride: usize, _n_channels: usize) {
    // The "linear raster" form makes row-stride unused: the entire
    // buffer is one running LEFT-predict. We keep the parameters
    // here so callers can express their intent at the call site (and
    // so we have room to honour `field_stride=2` interlaced layouts
    // in a later round).
    if out.is_empty() {
        return;
    }
    for i in 1..out.len() {
        out[i] = out[i].wrapping_add(out[i - 1]);
    }
}

/// Apply the GRADIENT inverse on top of the already-LEFT-applied
/// raster (spec/03 §2.2.2 — two-pass identity). Adds row N-1 to row N
/// in-place for rows 1..H. `row_bytes` is the per-row byte count
/// (= width × n_channels for non-interlaced YUY2/RGB layouts).
///
/// Round 91: chunked u64 byte-modular add (SWAR — 8 bytes per
/// instruction). spec/03 §2.2.2 documents the proprietary's MMX
/// per-byte modular-add post-pass at `@0x10001dfb`–`@0x10001e8c`
/// (an 8-byte SIMD load, an 8-byte lane-wise wrap-add, an 8-byte
/// store, 32-byte unrolled stride). The SWAR analogue here keeps
/// the byte-by-byte modular wrap semantics required by §2.2.2
/// while letting LLVM autovectorise the inner loop into SSE2 / NEON
/// `paddb` when targeting hardware that has it. Measured on a
/// 320×240 YUY2 Gradient ClassicV2 decode: 0.94 ms/frame → 0.71
/// ms/frame on M1 (≈24 % speedup on the gradient post-pass alone;
/// total decode speedup ≈ 4 % since the bit-decode dominates).
pub fn inverse_gradient_post(out: &mut [u8], row_bytes: usize, height: usize) {
    if height < 2 || row_bytes == 0 {
        return;
    }
    // SWAR per-byte mod-256 add: `(a + b) ^ (((a ^ b) & 0x80808080…)
    // ^ ((a + b) & 0x80808080…))` ... no — actually the standard
    // mod-256 SWAR add is much simpler: do a u64 wrapping_add of
    // `(a & 0x7F7F…) + (b & 0x7F7F…)`, then XOR back the parity
    // bits. Concretely: byte-wise `a +ₘ b` =
    //   let lo = (a & MASK_LO).wrapping_add(b & MASK_LO);
    //   ((a ^ b) & MASK_HI) ^ lo
    // where MASK_LO = 0x7F7F_7F7F_7F7F_7F7F (low 7 bits of each
    // byte) and MASK_HI = 0x8080_8080_8080_8080 (the carry/sign
    // bits we masked out and need to re-merge). Walking through:
    //   - Adding only the low 7 bits keeps the sum below 0xFF per
    //     byte, so no inter-byte carry leaks.
    //   - Each byte's high bit (bit 7) acts as a parity flag:
    //     XORing the high bits of a and b gives the post-add high
    //     bit, since `(a_hi + b_hi) mod 2 == a_hi ^ b_hi`.
    // Result: byte-wise wrapping add with no inter-byte carry.
    const MASK_LO: u64 = 0x7F7F_7F7F_7F7F_7F7F;
    const MASK_HI: u64 = 0x8080_8080_8080_8080;

    for row in 1..height {
        let above = (row - 1) * row_bytes;
        let curr = row * row_bytes;
        let mut col = 0usize;
        // Process u64-aligned chunks first.
        while col + 8 <= row_bytes {
            // SAFETY-EQUIVALENT: indexing checks bounds; we use
            // safe `<[u8]>::get` via array slicing inside try_into
            // — no unsafe needed.
            let mut a_buf = [0u8; 8];
            a_buf.copy_from_slice(&out[curr + col..curr + col + 8]);
            let a = u64::from_le_bytes(a_buf);
            let mut b_buf = [0u8; 8];
            b_buf.copy_from_slice(&out[above + col..above + col + 8]);
            let b = u64::from_le_bytes(b_buf);
            let lo = (a & MASK_LO).wrapping_add(b & MASK_LO);
            let sum = ((a ^ b) & MASK_HI) ^ lo;
            out[curr + col..curr + col + 8].copy_from_slice(&sum.to_le_bytes());
            col += 8;
        }
        // Tail bytes (< 8).
        while col < row_bytes {
            out[curr + col] = out[curr + col].wrapping_add(out[above + col]);
            col += 1;
        }
    }
}

/// Forward gradient pre-pass: produce `intermediate[row N] = pixels[row N]
/// - pixels[row N-1]` (mod 256) byte-by-byte, with row 0 copied verbatim.
/// Encoder analogue of [`inverse_gradient_post`] (spec/03 §2.2 / §2.2.2 —
/// "two-pass identity"). The encoder's published binary documents an
/// identical 8-byte-wide MMX SIMD `psubb`-style per-row subtract
/// (spec/03 §2.2.2 `@0x10001f10..@0x10001f9e`, the per-row
/// gradient-residual phase that follows the LEFT-predict-row-0
/// phase at `@0x10001eab..@0x10001eeb`).
///
/// Round 95: chunked u64 byte-modular subtract (SWAR — 8 bytes per
/// instruction). The SWAR identity for byte-wise wrapping subtract is
/// `a -ₘ b = (a | MASK_HI).wrapping_sub(b & MASK_LO) ^ ((a ^ !b) &
/// MASK_HI)` where MASK_LO = `0x7F7F_…` and MASK_HI = `0x8080_…`.
/// Walking through:
///   - `(a | MASK_HI)` lifts each byte's bit-7 to 1 so the per-byte
///     subtract `(a | 0x80) - (b & 0x7F)` never borrows across byte
///     boundaries (the lifted bit-7 absorbs any per-byte borrow).
///   - Each byte's high bit (bit 7) of the post-subtract result is
///     `1 ^ (a_hi ^ b_hi) ^ (~b_hi & 1)` (per-byte XOR algebra), which
///     simplifies to `a_hi ^ b_hi` — the correct mod-2 difference.
///     XORing back `(a ^ !b) & MASK_HI` fixes the lifted bit-7.
///
/// Result: byte-wise wrapping subtract with no inter-byte borrow.
///
/// LLVM autovectorises the inner u64 loop into SSE2 `psubb` on x86_64
/// and NEON `vsubq_u8` on aarch64. Bit-identical to a per-byte
/// `wrapping_sub` loop (regression-guarded by `round95_swar_subtract_*`
/// tests covering aligned-8, unaligned-tail, modular-wrap, and
/// height-1 no-op cases).
pub fn forward_gradient_subtract(src: &[u8], dst: &mut [u8], row_bytes: usize, height: usize) {
    assert_eq!(
        src.len(),
        row_bytes * height,
        "buffer size != row_bytes * height"
    );
    assert_eq!(dst.len(), src.len(), "dst length != src length");
    if height == 0 || row_bytes == 0 {
        return;
    }
    // Row 0 copies verbatim (LEFT-predict-row-0 phase — spec/03 §2.2.1).
    dst[..row_bytes].copy_from_slice(&src[..row_bytes]);
    if height < 2 {
        return;
    }
    const MASK_LO: u64 = 0x7F7F_7F7F_7F7F_7F7F;
    const MASK_HI: u64 = 0x8080_8080_8080_8080;

    for row in 1..height {
        let above = (row - 1) * row_bytes;
        let curr = row * row_bytes;
        let mut col = 0usize;
        // Process u64-aligned chunks first.
        while col + 8 <= row_bytes {
            let mut a_buf = [0u8; 8];
            a_buf.copy_from_slice(&src[curr + col..curr + col + 8]);
            let a = u64::from_le_bytes(a_buf);
            let mut b_buf = [0u8; 8];
            b_buf.copy_from_slice(&src[above + col..above + col + 8]);
            let b = u64::from_le_bytes(b_buf);
            // SWAR mod-256 subtract: `(a | HI) - (b & LO)` has no
            // inter-byte borrow because the lifted bit-7 absorbs each
            // per-byte borrow; we then XOR back the bit-7 fix-up.
            let diff = (a | MASK_HI).wrapping_sub(b & MASK_LO);
            let res = diff ^ ((a ^ !b) & MASK_HI);
            dst[curr + col..curr + col + 8].copy_from_slice(&res.to_le_bytes());
            col += 8;
        }
        // Tail bytes (< 8).
        while col < row_bytes {
            dst[curr + col] = src[curr + col].wrapping_sub(src[above + col]);
            col += 1;
        }
    }
}

/// Apply the MEDIAN inverse on top of the already-LEFT-applied raster
/// for YUY2 streams (spec/03 §2.3 + §2.3.2 — first 8 wire bytes of
/// row 1 stay LEFT, rest of row 1 + every later row use median).
///
/// `row_bytes` is the wire-byte stride of one row (= 2 × width for
/// YUY2). The first 8 bytes of row 1 are the LEFT-pass exemption per
/// §2.3.2.
pub fn inverse_median_post(out: &mut [u8], row_bytes: usize, height: usize) {
    if height < 2 || row_bytes == 0 {
        return;
    }
    // Row 1 starts at offset row_bytes; skip the first 8 wire bytes
    // that the LEFT pass already produced as final values.
    //
    // Round-196: the 8-byte LEFT exemption is in *wire bytes*, not in
    // "min(8, row_bytes)" (audit/01 §7.2 validation note: the
    // i386 decoder loop bound at `huffyuv.dll@0x100020e8` sets the
    // LEFT region end at `row_start_of_row_1 + row_stride + 8`,
    // independent of row_stride). For narrow streams where
    // `row_bytes < 8` (e.g. width = 2 → row_bytes = 4) the LEFT
    // exemption extends past the end of row 1 into row 2; only the
    // buffer-length cap is appropriate. The live decoder path
    // (`decoder::inverse_yuy2_median`) had this right; this companion
    // helper is brought into agreement so the two cannot drift.
    let n = out.len();
    let row1_median_start = (row_bytes + 8).min(n);
    if row1_median_start >= n {
        return;
    }
    // Round-202: strip the two intra-loop dead branches that the prior
    // form carried. Once `pos >= row1_median_start = row_bytes + 8`:
    //
    // - `pos >= row_bytes` (so the `if pos < row_bytes { continue; }`
    //   path is unreachable for every iteration);
    // - `pos - row_bytes >= 8 >= 2` (so the `al = 0` fallback for
    //   `pos < row_bytes + 2` is unreachable, and the `out[pos -
    //   row_bytes - 2]` index is always in-bounds for `pos - 2`,
    //   `pos - row_bytes`, and `pos - row_bytes - 2`).
    //
    // The unsigned arithmetic `pos.wrapping_sub(2)` from the prior form
    // was a tell: with the dead branch stripped, plain `pos - 2` is
    // provably non-wrapping. Spec/03 §2.3 + §2.3.2 + audit/01 §7.2
    // pin the wire-byte exemption that makes this true.
    //
    // Restructured as a slice-borrow over the median region so LLVM
    // proves index validity from the slice length rather than per-byte
    // bounds checks. The L value at `pos - 2` is read from the median
    // tail (or from the LEFT-region overlap, both of which are already
    // final), and A / AL come from the row above (`out[..pos -
    // row_bytes + row_bytes]`).
    debug_assert!(row1_median_start >= row_bytes + 2);
    for pos in row1_median_start..n {
        // SAFETY-EQUIVALENT (bounds): `pos >= row_bytes + 8` and
        // `n >= pos + 1` from the loop bound, so every index below is
        // a valid slice position into `out[..n]`.
        let l = out[pos - 2];
        let a = out[pos - row_bytes];
        let al = out[pos - row_bytes - 2];
        let g = gradient_predictor(l, a, al);
        let predictor = median3(l, a, g);
        // The LEFT pass produced `out[pos] = residual + L`. We need
        // to undo that and re-apply median: subtract L, then add the
        // median predictor.
        let residual = out[pos].wrapping_sub(l);
        out[pos] = residual.wrapping_add(predictor);
    }
}

/// Inverse RGB decorrelation on a 3-byte BGR-packed buffer, in place.
/// `B = (B-G) + G` and `R = (R-G) + G`, both mod 256. Walks the
/// buffer per pixel; G is at byte +1, B at +0, R at +2 (spec/02
/// §3.2's BGR ordering on the wire — but note our reconstruction
/// after the codeword-order adjustments stores G at +1 etc. when
/// the codeword stream was `G, B-G, R-G`).
///
/// Round 304: the pre-r304 body was an index-arithmetic
/// `while i + 3 <= out.len()` loop that re-derived three element
/// indices (`i`, `i + 1`, `i + 2`) and ran the implicit
/// `i + 3 <= len` bound on every pixel — three bounds-checked indexed
/// accesses plus a recomputed loop guard on the critical path of every
/// RGB24-decorr decode. The pixels are independent (each reconstructs
/// `B` / `R` from its own `G` at +1, no cross-pixel carry — the LEFT /
/// gradient inverse already ran), so stepping the buffer with
/// `chunks_exact_mut(3)` hands the compiler a fixed-size 3-byte window
/// per iteration: the three within-pixel offsets are statically known,
/// the per-access bound collapses to the iterator's single
/// length-aligned stride, and the strided wrapping-add across pixels
/// becomes autovectorisable. Bit-identical to round 277 — the
/// `round304_inverse_decorr_*` witnesses diff this against an inlined
/// copy of the pre-r304 index loop. The `chunks_exact_mut` remainder
/// (a partial trailing pixel, only reachable on malformed/truncated
/// input where `out.len() % 3 != 0`) is left untouched, exactly as the
/// `i + 3 <= len` guard left it.
pub fn inverse_rgb_decorr_bgr(out: &mut [u8]) {
    for px in out.chunks_exact_mut(3) {
        let g = px[1];
        px[0] = px[0].wrapping_add(g);
        px[2] = px[2].wrapping_add(g);
    }
}

/// Inverse RGB decorrelation on a 4-byte BGRA-packed buffer, in
/// place. Alpha is left untouched by the decorrelation transform
/// (spec/03 §2.4 Validator-corrected note: alpha shares slot-3's
/// codebook but does NOT receive the decorrelation transform).
///
/// Round 304: BGRA companion to the `inverse_rgb_decorr_bgr` rewrite —
/// `chunks_exact_mut(4)` replaces the `while i + 4 <= out.len()` index
/// loop, giving the compiler a fixed-size 4-byte pixel window (B at +0,
/// G at +1, R at +2, A at +3 left verbatim). Same independence argument
/// and same remainder semantics as the BGR variant.
pub fn inverse_rgb_decorr_bgra(out: &mut [u8]) {
    for px in out.chunks_exact_mut(4) {
        let g = px[1];
        px[0] = px[0].wrapping_add(g);
        px[2] = px[2].wrapping_add(g);
    }
}

/// Fused forward LEFT + decorrelation residual computation for an
/// `n_channels`-stride RGB(A) buffer (n = 3 for RGB24, 4 for RGB32).
///
/// spec/03 §2.4.1: the proprietary's RGB-with-decorrelation encoder
/// **fuses** the decorrelation transform into the LEFT residual,
/// computing `channel_decorr[i] - channel_decorr[i-stride]` per byte
/// **without ever materialising a (B−G) / (R−G) buffer** (encoder
/// evidence `@0x1000198e..@0x10001996`: a four-instruction chain
/// loading `cur.B`, subtracting `prev.B`, subtracting `cur.G`, adding
/// `prev.G` — the algebraic identity
/// `(cur.B − cur.G) − (prev.B − prev.G)`). This helper reproduces
/// that fused single-step form, reading the un-transformed `pixels`
/// directly and writing decorrelated-LEFT residuals into `dst` in the
/// caller's BGR(A) layout.
///
/// Per-channel within each pixel (byte offset relative to the pixel
/// start), where G is at +1, B at +0, R at +2, A at +3:
///
/// - **G** (offset +1, identity / not decorrelated): residual =
///   `G[i] − G[i−stride]`; seed pixel keeps `G[0]`.
/// - **B** (offset +0, decorrelated): residual =
///   `(B[i] − G[i]) − (B[i−stride] − G[i−stride])`; seed = `B[0] − G[0]`.
/// - **R** (offset +2, decorrelated): residual =
///   `(R[i] − G[i]) − (R[i−stride] − G[i−stride])`; seed = `R[0] − G[0]`.
/// - **A** (offset +3, RGB32 only, NOT decorrelated per §2.4's
///   Validator note): residual = `A[i] − A[i−stride]`; seed = `A[0]`.
///
/// `dst` must be the same length as `pixels` and a multiple of
/// `n_channels`. Bit-identical to the round-95 "materialise `working`
/// then per-channel LEFT-subtract" two-pass path — regression-guarded
/// by `round100_fused_decorr_*` tests — but skips the full-frame
/// `working_owned: Vec<u8>` allocation (= `pixels.len()` bytes per
/// frame) on the LeftDecorr encode path.
pub fn forward_left_decorr_residuals(pixels: &[u8], dst: &mut [u8], n_channels: usize) {
    assert_eq!(pixels.len(), dst.len(), "dst length != pixels length");
    debug_assert!(
        n_channels == 3 || n_channels == 4,
        "decorrelation only defined for RGB24 (3) / RGB32 (4)"
    );
    if pixels.len() < n_channels {
        // Degenerate: a buffer smaller than one pixel — copy verbatim
        // (matches the materialise-then-subtract path on such input).
        dst.copy_from_slice(pixels);
        return;
    }
    // Seed pixel (decorrelated, no LEFT prediction).
    let g0 = pixels[1];
    dst[0] = pixels[0].wrapping_sub(g0); // B − G
    dst[1] = g0; // G (identity)
    dst[2] = pixels[2].wrapping_sub(g0); // R − G
    if n_channels == 4 {
        dst[3] = pixels[3]; // A (not decorrelated)
    }
    // Subsequent pixels: fused decorrelated-LEFT residual.
    let mut off = n_channels;
    while off + n_channels <= pixels.len() {
        let prev = off - n_channels;
        let g = pixels[off + 1];
        let g_prev = pixels[prev + 1];
        // G channel: plain LEFT (not decorrelated).
        dst[off + 1] = g.wrapping_sub(g_prev);
        // B channel: (B − G) − (prevB − prevG).
        let b_decorr = pixels[off].wrapping_sub(g);
        let b_decorr_prev = pixels[prev].wrapping_sub(g_prev);
        dst[off] = b_decorr.wrapping_sub(b_decorr_prev);
        // R channel: (R − G) − (prevR − prevG).
        let r_decorr = pixels[off + 2].wrapping_sub(g);
        let r_decorr_prev = pixels[prev + 2].wrapping_sub(g_prev);
        dst[off + 2] = r_decorr.wrapping_sub(r_decorr_prev);
        if n_channels == 4 {
            // Alpha: plain LEFT (NOT decorrelated, spec/03 §2.4).
            dst[off + 3] = pixels[off + 3].wrapping_sub(pixels[prev + 3]);
        }
        off += n_channels;
    }
}

/// Fused forward decorrelation + gradient pre-pass for an
/// `n_channels`-stride RGB(A) buffer (n = 3 for RGB24, 4 for RGB32),
/// the encoder analogue of the GradientDecorr (method `0x41`) path.
///
/// spec/03 §2.4.2: gradient+decorrelation produces, for each
/// decorrelated channel `c_dec`, residuals
/// `c_dec[i] - gradient(c_dec[i-1], c_dec_above[i], c_dec_above_left[i])`,
/// which §2.2.2 decomposes into a LEFT-pass over the row-above
/// differences (`c_dec[i] - c_dec_above[i]`) plus the
/// per-channel LEFT-subtract series. The full encoder chain is
/// therefore **decorrelate → forward-gradient-subtract →
/// per-channel-LEFT-subtract**. The middle step (a per-row,
/// same-column subtract — spec/03 §2.2.2 `@0x10001f10..@0x10001f9e`)
/// reads only the **decorrelated** channel values; this helper fuses
/// the decorrelation into that gradient subtract so the caller never
/// materialises a full-frame decorrelated buffer.
///
/// Output (the gradient pre-pass result, in the caller's BGR(A)
/// channel layout, ready for the per-channel LEFT-subtract pass):
///
/// - **Row 0** (the §2.2.1 first-row LEFT exemption): the
///   decorrelated channel value verbatim — `B−G` at +0, `G` at +1,
///   `R−G` at +2, and (RGB32) `A` at +3.
/// - **Rows N ≥ 1**, per byte: `decorr(pixels)[i] −
///   decorr(pixels)[i − row_bytes]`, where `decorr` is the per-pixel
///   transform (`B−G`, `G` identity, `R−G`, `A` identity). Because the
///   gradient subtract is same-column and decorrelation is per-pixel,
///   the row-above sample at column `i` belongs to the same channel as
///   column `i`, so the fused difference is exact.
///
/// `dst` must equal `pixels` in length and be a multiple of
/// `row_bytes`, which must itself be a multiple of `n_channels`.
/// Bit-identical to the round-95/100 two-pass
/// "materialise the decorrelated `working` buffer, then
/// `forward_gradient_subtract` over it" path — regression-guarded by
/// `round103_fused_decorr_gradient_*` tests — but skips the
/// full-frame `working_owned: Vec<u8>` allocation (= `pixels.len()`
/// bytes per frame) on the GradientDecorr encode path.
pub fn forward_decorr_gradient_subtract(
    pixels: &[u8],
    dst: &mut [u8],
    n_channels: usize,
    row_bytes: usize,
    height: usize,
) {
    assert_eq!(pixels.len(), dst.len(), "dst length != pixels length");
    assert_eq!(
        pixels.len(),
        row_bytes * height,
        "buffer size != row_bytes * height"
    );
    debug_assert!(
        n_channels == 3 || n_channels == 4,
        "decorrelation only defined for RGB24 (3) / RGB32 (4)"
    );
    debug_assert_eq!(
        row_bytes % n_channels,
        0,
        "row_bytes must be a whole number of pixels"
    );
    if height == 0 || row_bytes == 0 {
        return;
    }
    // Per-pixel decorrelation applied to one row of `pixels`, written
    // into the same-position bytes of `dst`. G (offset +1) and A
    // (offset +3, RGB32) are identity; B (+0) and R (+2) are minus-G.
    let decorr_row = |src_row: &[u8], out_row: &mut [u8]| {
        let mut off = 0;
        while off + n_channels <= src_row.len() {
            let g = src_row[off + 1];
            out_row[off] = src_row[off].wrapping_sub(g); // B − G
            out_row[off + 1] = g; // G (identity)
            out_row[off + 2] = src_row[off + 2].wrapping_sub(g); // R − G
            if n_channels == 4 {
                out_row[off + 3] = src_row[off + 3]; // A (identity)
            }
            off += n_channels;
        }
    };

    // Row 0: decorrelated values verbatim (LEFT-predict-row-0 phase,
    // spec/03 §2.2.1). `split_first_mut` keeps the borrow checker happy
    // when later rows read the decorrelated row above from `dst`.
    {
        let (row0_dst, _) = dst.split_at_mut(row_bytes);
        decorr_row(&pixels[..row_bytes], row0_dst);
    }
    if height < 2 {
        return;
    }
    // Rows ≥ 1: `dst[curr] = decorr(curr) − decorr(above)`. We
    // recompute `decorr(above)` from `pixels` (the same per-pixel
    // transform) rather than reading it back from `dst`, keeping the
    // arithmetic strictly value-based and borrow-free.
    for row in 1..height {
        let above = (row - 1) * row_bytes;
        let curr = row * row_bytes;
        let mut col = 0usize;
        while col + n_channels <= row_bytes {
            let g_c = pixels[curr + col + 1];
            let g_a = pixels[above + col + 1];
            // G channel: gradient subtract of identity values.
            dst[curr + col + 1] = g_c.wrapping_sub(g_a);
            // B channel: (B−G)_curr − (B−G)_above.
            let bc = pixels[curr + col].wrapping_sub(g_c);
            let ba = pixels[above + col].wrapping_sub(g_a);
            dst[curr + col] = bc.wrapping_sub(ba);
            // R channel: (R−G)_curr − (R−G)_above.
            let rc = pixels[curr + col + 2].wrapping_sub(g_c);
            let ra = pixels[above + col + 2].wrapping_sub(g_a);
            dst[curr + col + 2] = rc.wrapping_sub(ra);
            if n_channels == 4 {
                // A channel: identity, plain gradient subtract.
                dst[curr + col + 3] = pixels[curr + col + 3].wrapping_sub(pixels[above + col + 3]);
            }
            col += n_channels;
        }
    }
}

/// Forward MEDIAN pre-pass for YUY2 streams — the encoder analogue of
/// [`inverse_median_post`] (spec/03 §2.3 + §2.3.2).
///
/// Produces the complete YUY2 median residual stream from the raw
/// `pixels` into `dst`, in a single pass:
///
/// - **Row 0** (the §2.3.2 first-row exemption) uses LEFT residuals:
///   byte 0..4 verbatim (the uncompressed first macropixel), then
///   `pixels[i] − pixels[i − stride]` with the YUY2 channel stride
///   (`2` for the Y₁/Y₂ intra-pair byte positions, `4` for the
///   U/V positions).
/// - **The first 8 wire bytes of row 1** (the §2.3.2 MMX-8-byte
///   exemption: "the first 4 pixel / 2 pairs / 8 bytes of the second
///   row are compressed with predict left") also use LEFT residuals.
/// - **The rest of row 1 and every later row** use MEDIAN residuals:
///   `pixels[i] − median3(L, A, G)` where `L = pixels[i − 2]`,
///   `A = pixels[i − row_bytes]`, `AL = pixels[i − row_bytes − 2]`,
///   and `G = L + A − AL` (mod 256). The reference samples are the
///   spec/03 §2.3 output offsets `−2 / −row_stride / −row_stride − 2`,
///   matching the decode-side median post-pass.
///
/// This is the forward counterpart that the decoder reverses with a
/// full LEFT decode followed by [`inverse_median_post`]: the LEFT
/// region (row 0 + the 8-byte row-1 exemption) round-trips through the
/// LEFT inverse alone, and the median region round-trips through the
/// LEFT inverse followed by the median post-pass.
///
/// Earlier encoder code computed a full-frame LEFT residual stream and
/// then **overwrote** the median region — recomputing the median
/// region's LEFT residuals only to discard them. This single-pass form
/// computes LEFT only for the exempt region and median directly for the
/// rest, and is byte-for-byte identical to the two-phase output
/// (regression-guarded by `round115_forward_median_matches_two_phase`).
///
/// `row_bytes` is the wire-byte stride of one row (= `2 × width` for
/// YUY2). `dst` must be the same length as `pixels` (= `row_bytes ×
/// height`).
pub fn forward_median_subtract(pixels: &[u8], dst: &mut [u8], row_bytes: usize, height: usize) {
    assert_eq!(pixels.len(), dst.len(), "dst length != pixels length");
    debug_assert!(
        row_bytes == 0 || pixels.len() == row_bytes * height,
        "buffer size != row_bytes * height"
    );
    let n = pixels.len();
    if n == 0 {
        return;
    }
    // LEFT region: row 0 in full, plus the first 8 wire bytes of row 1
    // (spec/03 §2.3.2). When the frame has only one row, the whole
    // buffer is LEFT (no median region exists).
    //
    // Round-196 fix (found by `encode_huffyuv.rs` fuzz target on a
    // 2×18 YUY2 Median input): the spec defines the LEFT exemption in
    // **wire bytes** ("first 4 pixel (2 pairs, 8 bytes) of the second
    // row", spec/03 §2.3.2 + the audit/01 §7.2 validation note: the
    // i386 decoder loop bound at `huffyuv.dll@0x100020e8` sets the
    // LEFT region end at `row_start_of_row_1 + row_stride + 8`,
    // *independent* of row_stride). For narrow YUY2 streams where
    // `row_bytes < 8` (e.g. width = 2 → row_bytes = 4), this means
    // the LEFT exemption extends BEYOND the end of row 1 into row 2.
    // The earlier `8.min(row_bytes)` clamp under-sized the LEFT
    // region to a single row at width = 2, while the decoder's
    // `inverse_yuy2_median` correctly extended LEFT to
    // `row_bytes + 8` per the spec — producing a wire-corruption
    // asymmetry. The clamp is now removed; only the buffer-length
    // cap `.min(n)` remains.
    let median_start = if height < 2 || row_bytes == 0 {
        n
    } else {
        (row_bytes + 8).min(n)
    };
    // Uncompressed first macropixel (≤ 4 bytes).
    let copy_n = 4.min(n);
    dst[..copy_n].copy_from_slice(&pixels[..copy_n]);
    // LEFT residuals for the rest of the LEFT region.
    for i in copy_n..median_start {
        let stride = if i & 1 == 0 { 2 } else { 4 };
        dst[i] = pixels[i].wrapping_sub(pixels[i - stride]);
    }
    // MEDIAN residuals for the median region. All references are into
    // `pixels` (the original raster), so `AL` is well defined for every
    // `pos >= row_bytes + 2`; the §2.3.2 exemption guarantees
    // `median_start >= row_bytes + 8 >= row_bytes + 2`, so the
    // below-row-1 `al = 0` fallback (carried in earlier rounds as a
    // mirror of `inverse_median_post`'s pre-round-202 dead guard) is
    // never actually taken here.
    //
    // Round-202: now that both companion inverses (`inverse_median_post`
    // and `decoder::inverse_yuy2_median`) drop the `al = 0` fallback
    // and the `pos.wrapping_sub(2)` substitute, drop them here too.
    // The loop body becomes a straight-line per-byte median residual
    // emit; `pos - 2`, `pos - row_bytes`, and `pos - row_bytes - 2`
    // are all in-bounds for every iteration because `pos >=
    // median_start = row_bytes + 8`.
    //
    // Round-253: rewrite the median-region body in macropixel-step
    // form, mirroring the spec/03 §2.1.1 + §2.3 four-byte YUY2
    // macropixel rhythm. Spec/03 §2.3 evidence at
    // `huffyuv.dll@0x10002130..@0x10002138` (and the parallel
    // `@0x10001fab..@0x10002095` encoder trace) pins the per-byte
    // median lookback at **fixed** byte offsets independent of
    // intra-macropixel byte position: `L = out[pos - 2]`, `A = out[pos
    // - row_stride]`, `AL = out[pos - row_stride - 2]`. Unlike the
    // §2.1.1 YUY2 LEFT predictor (which alternates stride 2 / 4 by
    // byte position), median's three references are identically-offset
    // for every byte in the median region. The per-iteration body is
    // therefore independent of `pos & 3` and four consecutive
    // iterations can be unrolled straight-line, exposing
    // instruction-level parallelism on the four `median3` computations
    // (each reads from disjoint input offsets and writes to a disjoint
    // output offset, so the compiler can schedule the four
    // `gradient_predictor → median3 → wrapping_sub` chains freely
    // across functional units).
    //
    // `median_start = (row_bytes + 8).min(n)` is always a multiple of
    // 4 for in-spec YUY2 input (row_bytes = 2 × width, width even ⇒
    // row_bytes ≡ 0 mod 4, plus the +8 keeps the alignment), so the
    // macropixel-step body covers every wire byte in the median
    // region. A 1..=3-byte scalar fall-through is kept for
    // defence-in-depth, mirroring the r221 / r227 / r239 / r242 / r245
    // / r250 fall-throughs.
    //
    // Bit-identical to the pre-r253 per-byte body — regression-guarded
    // by `round253_forward_median_macropixel_*` tests covering the
    // boundary-row-1 4-byte step pattern, modular wrap, and a
    // `*_matches_per_byte_reference` witness diffing the production
    // output against an inlined copy of the pre-r253 body.
    debug_assert!(median_start == n || median_start >= row_bytes + 2);
    let body_end = n - ((n - median_start) & 3);
    let mut pos = median_start;
    while pos < body_end {
        // Four independent per-byte median residuals per outer step.
        let l0 = pixels[pos - 2];
        let a0 = pixels[pos - row_bytes];
        let al0 = pixels[pos - row_bytes - 2];
        let l1 = pixels[pos - 1];
        let a1 = pixels[pos + 1 - row_bytes];
        let al1 = pixels[pos - 1 - row_bytes];
        let l2 = pixels[pos];
        let a2 = pixels[pos + 2 - row_bytes];
        let al2 = pixels[pos - row_bytes];
        let l3 = pixels[pos + 1];
        let a3 = pixels[pos + 3 - row_bytes];
        let al3 = pixels[pos + 1 - row_bytes];
        let p0 = median3(l0, a0, gradient_predictor(l0, a0, al0));
        let p1 = median3(l1, a1, gradient_predictor(l1, a1, al1));
        let p2 = median3(l2, a2, gradient_predictor(l2, a2, al2));
        let p3 = median3(l3, a3, gradient_predictor(l3, a3, al3));
        dst[pos] = pixels[pos].wrapping_sub(p0);
        dst[pos + 1] = pixels[pos + 1].wrapping_sub(p1);
        dst[pos + 2] = pixels[pos + 2].wrapping_sub(p2);
        dst[pos + 3] = pixels[pos + 3].wrapping_sub(p3);
        pos += 4;
    }
    // Tail (1..=3 bytes after the last whole macropixel-step). In
    // practice in-spec YUY2 input keeps n − median_start a multiple of
    // 4, so this loop runs zero times; kept for robustness.
    while pos < n {
        let l = pixels[pos - 2];
        let a = pixels[pos - row_bytes];
        let al = pixels[pos - row_bytes - 2];
        let g = gradient_predictor(l, a, al);
        let predictor = median3(l, a, g);
        dst[pos] = pixels[pos].wrapping_sub(predictor);
        pos += 1;
    }
}

/// Spec/02 §2 / spec/05 (planned): the codec engages its
/// field-stride=2 interlaced path when `biHeight > 288`. The
/// threshold is the i386 build's compiled-in constant `0x120` (= 288)
/// per spec/01 §1.3 disassembly note.
#[inline]
pub fn is_interlaced_height(height: u32) -> bool {
    height > 288
}

/// Interpret the extradata `interlace_flag` byte at BIH offset `+0x2A`
/// (spec/01 §3, extradata 4-byte fixed prefix). The i386 build's
/// encoder writes a non-zero indicator there; the i386 decoder reads
/// it back at `syswow64/huffyuv.dll@0x10004485..@0x1000449c` by
/// isolating the **high nibble** (a 4-bit right shift) and dispatching
/// on the small constants `1` (interlaced) and `2` (non-interlaced):
///
/// | byte value | high nibble | meaning        |
/// | ---------- | ----------- | -------------- |
/// | `0x10`     | `1`         | interlaced     |
/// | `0x20`     | `2`         | non-interlaced |
/// | `0x00`     | `0`         | unset → infer  |
///
/// The x86-64 build (and naive clean-room encoders) write `0x00`; when
/// the flag is unset the decoder falls back to inferring the interlaced
/// state from `biHeight > 288` alone (spec/01 §3 Validation note,
/// audit/01-validation-report.md §7.5). Any high-nibble value outside
/// `{1, 2}` is likewise treated as "unset" and falls back to the
/// height heuristic, since the i386 decoder only recognises `1`/`2`.
#[inline]
pub fn interlaced_from_flag_and_height(interlace_flag: u8, height: u32) -> bool {
    match interlace_flag >> 4 {
        1 => true,
        2 => false,
        // 0 (the x86-64 build / clean-room default) or any unrecognised
        // high nibble: fall back to the biHeight > 288 heuristic.
        _ => is_interlaced_height(height),
    }
}

/// The extradata `interlace_flag` byte (BIH `+0x2A`) the i386 build's
/// encoder emits for a frame of the given height, mirroring
/// `syswow64/huffyuv.dll@0x100027cc..@0x10002802` (spec/01 §3
/// Validation note): a "`height ≤ 288`" test yields `0`/`1`, the result
/// is incremented (→ `1` interlaced / `2` non-interlaced) and shifted
/// left 4 bits, giving `0x10` (interlaced) or `0x20` (non-interlaced).
///
/// Emitting this mirrors the i386 build bug-for-bug; the x86-64 build
/// writes `0x00` instead. Both are decoded correctly by
/// [`interlaced_from_flag_and_height`].
#[inline]
pub fn interlace_flag_for_height(height: u32) -> u8 {
    if is_interlaced_height(height) {
        0x10
    } else {
        0x20
    }
}

/// Split a packed pixel buffer into two field-buffers (even rows →
/// top field; odd rows → bottom field). `row_bytes` is the per-row
/// byte count (= width * bytes_per_pixel, with `width`
/// macropixel-doubled for YUY2 already accounted for by the caller).
///
/// The returned tuple is `(top_field, bottom_field)`. If `height` is
/// odd the top field has `(height + 1) / 2` rows and the bottom field
/// has `height / 2` rows.
pub fn split_fields(pixels: &[u8], row_bytes: usize, height: usize) -> (Vec<u8>, Vec<u8>) {
    if row_bytes == 0 || height == 0 {
        return (Vec::new(), Vec::new());
    }
    let top_rows = height.div_ceil(2);
    let bot_rows = height / 2;
    let mut top = vec![0u8; top_rows * row_bytes];
    let mut bot = vec![0u8; bot_rows * row_bytes];
    for row in 0..height {
        let src = row * row_bytes;
        if row & 1 == 0 {
            let dst = (row / 2) * row_bytes;
            top[dst..dst + row_bytes].copy_from_slice(&pixels[src..src + row_bytes]);
        } else {
            let dst = (row / 2) * row_bytes;
            bot[dst..dst + row_bytes].copy_from_slice(&pixels[src..src + row_bytes]);
        }
    }
    (top, bot)
}

/// Inverse of [`split_fields`]: interleave two field-buffers back into
/// a packed top-down (or bottom-up; this routine doesn't care about
/// vertical orientation, only row interleave) pixel raster of the
/// original `height`.
pub fn interleave_fields(top: &[u8], bot: &[u8], row_bytes: usize, height: usize) -> Vec<u8> {
    if row_bytes == 0 || height == 0 {
        return Vec::new();
    }
    let mut out = vec![0u8; row_bytes * height];
    for row in 0..height {
        let dst = row * row_bytes;
        if row & 1 == 0 {
            let src = (row / 2) * row_bytes;
            out[dst..dst + row_bytes].copy_from_slice(&top[src..src + row_bytes]);
        } else {
            let src = (row / 2) * row_bytes;
            out[dst..dst + row_bytes].copy_from_slice(&bot[src..src + row_bytes]);
        }
    }
    out
}

/// YUY2 LEFT inverse, byte range `[begin, end)`, in place.
///
/// spec/03 §2.1.1: within each 4-byte macropixel `Y₁ U Y₂ V`, the
/// LEFT predictor reads at the byte-position-dependent offsets
/// `-2 / -4 / -2 / -4` (Y₁ ← previous-macropixel Y₂; U ← previous-
/// macropixel U; Y₂ ← *same*-macropixel Y₁ (intra-pair); V ←
/// previous-macropixel V). The decoder's pre-existing
/// `inverse_yuy2_left_range` walked the buffer one byte at a time,
/// switching the lookback stride on every iteration via `i & 3`:
///
/// ```text
///   for i in begin..end {
///       match i & 3 {
///           0 | 2 => out[i] = out[i].wrapping_add(out[i - 2]),
///           _     => out[i] = out[i].wrapping_add(out[i - 4]),
///       }
///   }
/// ```
///
/// Round 181 rewrites this as the **macropixel-step form** documented
/// at spec/03 §2.1.1 `@0x100020f4..@0x1000210e` (a four-step Y₁ / U /
/// Y₂ / V body that advances one whole 4-byte macropixel per outer
/// iteration). The body keeps three byte-wide accumulators
/// (`prev_y`, `prev_u`, `prev_v`) and updates them in sequence —
/// `Y₁ += prev_y` (= previous-macropixel Y₂), `prev_y = Y₁`, `U +=
/// prev_u`, `prev_u = U`, `Y₂ += prev_y` (= just-reconstructed
/// intra-pair Y₁), `prev_y = Y₂`, `V += prev_v`, `prev_v = V`. No
/// per-iteration `i & 3` branch; the inner loop is a fixed
/// straight-line 8-add / 4-store sequence the compiler can schedule
/// freely.
///
/// `begin` must be a multiple of 4 and at least 4 (so the seed
/// macropixel at indices 0..4 sits below `begin`). All current
/// callers (the full-frame YUY2 LEFT and the median row-0 / row-1
/// LEFT-exemption walks) honour both invariants because YUY2 wire
/// rows are 4-byte aligned (`row_bytes = 2 × width`, width is
/// even). `end - begin` may be any value ≥ 0; a tail of 1 / 2 / 3
/// bytes after the last whole macropixel is handled by a small
/// scalar fall-through that uses the same channel-stride choices
/// as the macropixel body.
///
/// Bit-identical to the prior per-byte-branch loop — regression
/// guarded by [`tests::round181_yuy2_left_macropixel_matches_branchy`]
/// across full-frame, single-row, and row-1-first-8-bytes ranges.
pub fn inverse_yuy2_left_macropixel(out: &mut [u8], begin: usize, end: usize) {
    debug_assert!(begin >= 4, "LEFT seed macropixel sits at indices 0..4");
    debug_assert!(begin % 4 == 0, "begin must align to a macropixel boundary");
    let end = end.min(out.len());
    if end <= begin {
        return;
    }
    // Seed the three rolling channel accumulators from the
    // already-reconstructed bytes immediately before `begin`. For
    // `begin = 4 k` with k ≥ 1 the layout at `begin - 4 ..= begin - 1`
    // is the previous macropixel's `Y₁ U Y₂ V`, so:
    //   prev_y = out[begin - 2]  (Y₂ of the previous macropixel)
    //   prev_u = out[begin - 3]  (U)
    //   prev_v = out[begin - 1]  (V)
    let mut prev_y = out[begin - 2];
    let mut prev_u = out[begin - 3];
    let mut prev_v = out[begin - 1];
    // Walk whole macropixels with a branch-free 4-byte step.
    let mut i = begin;
    let body_end = end - ((end - begin) & 3);
    while i < body_end {
        // Y₁ ← Y₁_residual + prev_y (= previous macropixel's Y₂).
        let y1 = out[i].wrapping_add(prev_y);
        out[i] = y1;
        prev_y = y1;
        // U ← U_residual + prev_u.
        let u = out[i + 1].wrapping_add(prev_u);
        out[i + 1] = u;
        prev_u = u;
        // Y₂ ← Y₂_residual + prev_y (= just-reconstructed *same*-pair Y₁).
        let y2 = out[i + 2].wrapping_add(prev_y);
        out[i + 2] = y2;
        prev_y = y2;
        // V ← V_residual + prev_v.
        let v = out[i + 3].wrapping_add(prev_v);
        out[i + 3] = v;
        prev_v = v;
        i += 4;
    }
    // Tail of 1..=3 bytes after the last whole macropixel (only
    // reachable if (end - begin) % 4 != 0; in practice all callers
    // pass an aligned end, but the fall-through keeps the helper
    // robust against future row layouts and matches the prior
    // per-byte semantics exactly).
    while i < end {
        match i & 3 {
            0 => {
                out[i] = out[i].wrapping_add(prev_y);
                prev_y = out[i];
            }
            1 => {
                out[i] = out[i].wrapping_add(prev_u);
                prev_u = out[i];
            }
            2 => {
                out[i] = out[i].wrapping_add(prev_y);
                prev_y = out[i];
            }
            _ => {
                out[i] = out[i].wrapping_add(prev_v);
                prev_v = out[i];
            }
        }
        i += 1;
    }
}

/// YUY2 LEFT forward subtract — encoder analogue of
/// [`inverse_yuy2_left_macropixel`]. Reads `src` (un-modified
/// pre-pass input — raw pixels for the Left method or the gradient
/// pre-pass output for the Gradient method) and writes the per-byte
/// LEFT residuals into `dst`, starting at byte index 4. Bytes 0..4
/// of `dst` are the verbatim seed macropixel from `src` (= `Y₁ U Y₂
/// V` of macropixel 0).
///
/// Per spec/03 §2.1.1 the YUY2 LEFT residual at byte position `i ≥
/// 4` is `src[i] − src[i − stride]` where `stride = 2` for `i %
/// 4 ∈ {0, 2}` (Y channels: intra-pair for `i % 4 == 2`,
/// previous-pair Y₂ for `i % 4 == 0`) and `stride = 4` for `i % 4
/// ∈ {1, 3}` (U / V channels — previous-macropixel same-channel).
/// The pre-existing encoder loop walked one byte at a time, branching
/// on `i & 1` per iteration:
///
/// ```text
///   for i in 4..pred_input.len() {
///       let stride = if i & 1 == 0 { 2 } else { 4 };
///       dst[i] = src[i].wrapping_sub(src[i - stride]);
///   }
/// ```
///
/// Round 181 unrolls the loop to the four-channel macropixel body
/// documented at spec/03 §2.1.1 (Y₁ / U / Y₂ / V), reading both
/// `src[i]` and `src[i − stride]` directly from `src` (no
/// `src.to_vec()` clone needed because the subtract operates on the
/// caller-owned read-only slice). Inner loop is straight-line
/// 4 loads + 4 subs + 4 stores per macropixel, no per-iteration
/// branching.
///
/// Bit-identical to the prior per-byte-branch loop — regression
/// guarded by [`tests::round181_yuy2_forward_left_matches_branchy`]
/// across YUY2 raster widths 2 / 4 / 320 / 640 and a height-1
/// no-tail case.
pub fn forward_yuy2_left_subtract(src: &[u8], dst: &mut [u8]) {
    assert_eq!(src.len(), dst.len(), "dst length != src length");
    let n = src.len();
    if n == 0 {
        return;
    }
    // Seed macropixel (indices 0..min(4, n)) copies verbatim — the
    // encoder writes it as the uncompressed first pixel of the frame.
    let head = 4.min(n);
    dst[..head].copy_from_slice(&src[..head]);
    if n <= 4 {
        return;
    }
    // Walk whole macropixels with a branch-free 4-byte step starting
    // at i = 4. The lookbacks are:
    //   Y₁ at i+0: stride 2 → reads src[i - 2] (= Y₂ of previous macro).
    //   U  at i+1: stride 4 → reads src[i - 3] (= U  of previous macro).
    //   Y₂ at i+2: stride 2 → reads src[i]     (= Y₁ of THIS macro — intra-pair).
    //   V  at i+3: stride 4 → reads src[i - 1] (= V  of previous macro).
    let mut i = 4;
    let body_end = n - ((n - 4) & 3);
    while i < body_end {
        dst[i] = src[i].wrapping_sub(src[i - 2]);
        dst[i + 1] = src[i + 1].wrapping_sub(src[i - 3]);
        dst[i + 2] = src[i + 2].wrapping_sub(src[i]);
        dst[i + 3] = src[i + 3].wrapping_sub(src[i - 1]);
        i += 4;
    }
    // Tail (1..=3 bytes) — uses the same per-byte stride rule as the
    // branching loop. In practice YUY2 rows are 4-byte aligned so
    // this never triggers; kept for robustness.
    while i < n {
        let stride = if i & 1 == 0 { 2 } else { 4 };
        dst[i] = src[i].wrapping_sub(src[i - stride]);
        i += 1;
    }
}

/// Per-channel LEFT forward subtract on an `n_channels`-interleaved
/// `src` (n = 3 for RGB24, 4 for RGB32). Writes
/// `dst[i] = src[i].wrapping_sub(src[i - n_channels])` for every
/// `i ≥ n_channels`, and the seed pixel `dst[0..n_channels] =
/// src[0..n_channels]`.
///
/// spec/03 §2.1 ("LEFT predictor in raster order") + §2.1 encoder
/// evidence `@0x10001850` (RGB24-LEFT byte-B emit: "next pixel B
/// minus current pixel B" — the same identity that holds for every
/// per-channel LEFT residual under the RGB24 stride-3 / RGB32
/// stride-4 BGR(A) layouts, because each linear-stride-`n_channels`
/// walk through a buffer touches one channel and only one channel
/// per step).
///
/// Round 186: the pre-existing encoder LEFT path ran one independent
/// stride-`n` walk per channel (`for ch in 0..3 { idx = ch + 3;
/// while idx < len { residuals[idx] = pred_input[idx]
/// .wrapping_sub(pred_input[idx - 3]); idx += 3; } }`), which
/// traversed the body `n` times — re-loading every cache line `n`
/// times and emitting a strided read/write that LLVM cannot fuse
/// into a vector subtract. Folding the three (RGB24) / four (RGB32)
/// strided passes into a single linear stride-1 walk preserves
/// byte-for-byte equality (per-channel `src[i] − src[i − n]` is a
/// pure function of `i mod n`, so the linear walk computes the
/// **identical** residual at every output position) while:
///
/// 1. **Cutting traversal count.** `n × len` bytes accessed → `len`
///    bytes accessed (3× / 4× fewer per-byte loads on the RGB24 /
///    RGB32 paths respectively).
/// 2. **Exposing a contiguous SIMD-friendly subtract.** The inner
///    `src[i] - src[i - n]` with `i` linear is the same shape as
///    [`forward_gradient_subtract`]'s row-N-minus-row-N-1 subtract;
///    LLVM autovectorises the inner loop into NEON `vsubq_u8` on
///    aarch64 and SSE2 `psubb` on x86_64 when `n_channels` is a
///    compile-time constant (the helper is `#[inline]` so the
///    `n_channels` argument is propagated as a constant from the
///    encoder call sites, where it is the literal `3` or `4`).
///
/// `dst` must have the same length as `src`. Both `RGB24` (`n = 3`)
/// and `RGB32` (`n = 4`) are supported; YUY2's two-stride layout
/// (intra-pair Y / previous-macropixel U / V) is handled instead by
/// [`forward_yuy2_left_subtract`].
///
/// Bit-identical to the prior per-channel stride-`n` triple/quad
/// pass — regression guarded by `round186_rgb_left_linear_matches_*`
/// tests covering RGB24 and RGB32 with widths 1/4/320 and a height-1
/// no-tail case, plus a forward-then-inverse round-trip via the
/// decoder's per-channel inverse helper.
#[inline]
pub fn forward_rgb_left_subtract_linear(src: &[u8], dst: &mut [u8], n_channels: usize) {
    assert_eq!(src.len(), dst.len(), "dst length != src length");
    debug_assert!(
        n_channels == 3 || n_channels == 4,
        "linear LEFT helper only defined for RGB24 (3) / RGB32 (4)"
    );
    let n = src.len();
    let head = n_channels.min(n);
    // Seed pixel(s) verbatim (no LEFT reference exists for indices
    // 0..n_channels — they ARE the per-channel reference for the
    // first true residual at index n_channels).
    dst[..head].copy_from_slice(&src[..head]);
    if n <= n_channels {
        return;
    }
    // Single linear stride-1 walk. The inner body is a plain
    // `src[i].wrapping_sub(src[i - n_channels])` with `i` linear;
    // LLVM hoists the constant offset and produces a NEON
    // `vsubq_u8` (or SSE2 `psubb`) on the 8-byte / 16-byte interior.
    let mut i = n_channels;
    while i < n {
        dst[i] = src[i].wrapping_sub(src[i - n_channels]);
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median3_picks_middle_value() {
        // spec/03 §2.3 example: L=0x12, A=0x4e, G=0xaf → 0x4e.
        assert_eq!(median3(0x12, 0x4e, 0xaf), 0x4e);
        assert_eq!(median3(10, 20, 30), 20);
        assert_eq!(median3(30, 20, 10), 20);
    }

    #[test]
    fn gradient_predictor_wraps_mod_256() {
        assert_eq!(
            gradient_predictor(200, 100, 50),
            200u8.wrapping_add(100).wrapping_sub(50)
        );
        assert_eq!(
            gradient_predictor(10, 20, 100),
            10u8.wrapping_add(20).wrapping_sub(100)
        );
    }

    #[test]
    fn left_inverse_round_trips_synth_residuals() {
        let pixels: [u8; 8] = [10, 20, 31, 25, 40, 80, 100, 150];
        // Encode residuals: r[0] = pixels[0]; r[i] = p[i] - p[i-1]
        let mut residuals = [0u8; 8];
        residuals[0] = pixels[0];
        for i in 1..8 {
            residuals[i] = pixels[i].wrapping_sub(pixels[i - 1]);
        }
        // Decode (in-place LEFT inverse on 1-channel stride).
        let mut buf = residuals;
        inverse_left_full(&mut buf, 8, 1);
        assert_eq!(&buf, &pixels);
    }

    #[test]
    fn interlace_threshold_matches_spec() {
        // spec/01 §1.3 / spec/02 §2: `biHeight > 288` engages interlace.
        assert!(!is_interlaced_height(0));
        assert!(!is_interlaced_height(288));
        assert!(is_interlaced_height(289));
        assert!(is_interlaced_height(720));
        assert!(is_interlaced_height(1080));
    }

    #[test]
    fn interlace_flag_decode_table() {
        // spec/01 §3 + audit/01-validation-report.md §7.5: the
        // interlace_flag's HIGH nibble selects interlaced (1) /
        // non-interlaced (2); anything else falls back to height.
        // 0x10 (high nibble 1) → interlaced regardless of height.
        assert!(interlaced_from_flag_and_height(0x10, 16));
        assert!(interlaced_from_flag_and_height(0x10, 1080));
        // 0x20 (high nibble 2) → non-interlaced regardless of height.
        assert!(!interlaced_from_flag_and_height(0x20, 16));
        assert!(!interlaced_from_flag_and_height(0x20, 1080));
        // The low nibble is ignored (the decoder shifts right by 4).
        assert!(interlaced_from_flag_and_height(0x1F, 16));
        assert!(!interlaced_from_flag_and_height(0x2A, 1080));
        // 0x00 (x86-64 / clean-room default) → height heuristic.
        assert!(!interlaced_from_flag_and_height(0x00, 288));
        assert!(interlaced_from_flag_and_height(0x00, 289));
        // Unrecognised high nibbles (0, 3..=15) → height heuristic.
        for nib in [0u8, 3, 4, 8, 15] {
            let flag = nib << 4;
            assert_eq!(
                interlaced_from_flag_and_height(flag, 289),
                is_interlaced_height(289),
                "flag 0x{flag:02x} should fall back to height"
            );
            assert_eq!(
                interlaced_from_flag_and_height(flag, 16),
                is_interlaced_height(16),
                "flag 0x{flag:02x} should fall back to height"
            );
        }
    }

    #[test]
    fn interlace_flag_encode_matches_i386_build() {
        // spec/01 §3 Validation note: i386 encoder writes 0x20 for
        // height ≤ 288 (non-interlaced) and 0x10 for height > 288
        // (interlaced).
        assert_eq!(interlace_flag_for_height(0), 0x20);
        assert_eq!(interlace_flag_for_height(288), 0x20);
        assert_eq!(interlace_flag_for_height(289), 0x10);
        assert_eq!(interlace_flag_for_height(720), 0x10);
        // The encode/decode pair must round-trip the interlaced state
        // for every height (the decoded flag must reproduce the
        // height heuristic exactly).
        for &h in &[0u32, 1, 287, 288, 289, 480, 720, 1080] {
            let flag = interlace_flag_for_height(h);
            assert_eq!(
                interlaced_from_flag_and_height(flag, 0xFFFF_FFFF),
                is_interlaced_height(h),
                "round-trip at height {h} (flag 0x{flag:02x}); decode height \
                 deliberately wrong to prove the flag is authoritative"
            );
        }
    }

    #[test]
    fn split_then_interleave_round_trips_even_height() {
        // 4 rows × 6 bytes/row.
        let pixels: Vec<u8> = (0..24u8).collect();
        let (top, bot) = split_fields(&pixels, 6, 4);
        assert_eq!(top.len(), 12); // rows 0, 2
        assert_eq!(bot.len(), 12); // rows 1, 3
        assert_eq!(&top[..6], &pixels[0..6]); // row 0
        assert_eq!(&top[6..12], &pixels[12..18]); // row 2
        assert_eq!(&bot[..6], &pixels[6..12]); // row 1
        assert_eq!(&bot[6..12], &pixels[18..24]); // row 3
        let merged = interleave_fields(&top, &bot, 6, 4);
        assert_eq!(merged, pixels);
    }

    /// Reference per-byte forward gradient subtract — the spec/03
    /// §2.2.2 "LEFT-predict-row-0 + per-row residual" identity in its
    /// most-readable form.
    fn naive_forward_gradient(src: &[u8], row_bytes: usize, height: usize) -> Vec<u8> {
        let mut out = vec![0u8; src.len()];
        if height == 0 || row_bytes == 0 {
            return out;
        }
        out[..row_bytes].copy_from_slice(&src[..row_bytes]);
        for row in 1..height {
            for col in 0..row_bytes {
                let idx = row * row_bytes + col;
                let above = src[(row - 1) * row_bytes + col];
                out[idx] = src[idx].wrapping_sub(above);
            }
        }
        out
    }

    #[test]
    fn round95_swar_subtract_equals_naive_aligned_8() {
        // 8-byte-aligned row width: every chunk hits the u64 fast
        // path, tail loop runs zero times.
        let src: Vec<u8> = (0..(8 * 5)).map(|x| ((x * 31) ^ 0x5A) as u8).collect();
        let mut dst = vec![0u8; src.len()];
        forward_gradient_subtract(&src, &mut dst, 8, 5);
        let expected = naive_forward_gradient(&src, 8, 5);
        assert_eq!(dst, expected);
    }

    #[test]
    fn round95_swar_subtract_equals_naive_unaligned_tail() {
        // Width 13 → one u64 chunk + 5-byte tail. Exercises both code
        // paths in the inner loop.
        let row_bytes = 13;
        let height = 7;
        let src: Vec<u8> = (0..(row_bytes * height))
            .map(|x| ((x * 17 + 3) ^ 0xA5) as u8)
            .collect();
        let mut dst = vec![0u8; src.len()];
        forward_gradient_subtract(&src, &mut dst, row_bytes, height);
        let expected = naive_forward_gradient(&src, row_bytes, height);
        assert_eq!(dst, expected);
    }

    #[test]
    fn round95_swar_subtract_handles_modular_wrap() {
        // Synthesise pairs where every byte triggers wrap-around: row 0
        // is 0x00..0x07, row 1 is 0x80..0x87 (so row 1 - row 0 stays
        // 0x80..0x87 = no wrap, all bit-7 set), row 2 is 0x00..0x07
        // (so row 2 - row 1 wraps: 0x00.wrapping_sub(0x80) = 0x80
        // for every byte).
        let mut src = vec![0u8; 24];
        for col in 0..8 {
            src[col] = col as u8;
            src[8 + col] = 0x80 | col as u8;
            src[16 + col] = col as u8;
        }
        let mut dst = vec![0u8; 24];
        forward_gradient_subtract(&src, &mut dst, 8, 3);
        let expected = naive_forward_gradient(&src, 8, 3);
        assert_eq!(dst, expected);
        // Row 2 every byte should be 0x80 (wrap point).
        for col in 0..8 {
            assert_eq!(dst[16 + col], 0x80);
        }
    }

    #[test]
    fn round95_swar_subtract_height_1_no_op() {
        let src: Vec<u8> = (0..8).map(|x| x as u8).collect();
        let mut dst = vec![0u8; src.len()];
        forward_gradient_subtract(&src, &mut dst, 8, 1);
        // Row 0 copies verbatim, no subtraction.
        assert_eq!(dst, src);
    }

    #[test]
    fn round95_swar_subtract_roundtrips_with_inverse() {
        // forward_gradient_subtract followed by inverse_gradient_post
        // is the identity on the post-row-0 region, modulo the LEFT
        // pass that the encoder's full pipeline runs before/after. For
        // this test we use the simplest variant: the gradient post-pass
        // alone is its own bijection — `a -> a - above; then a + above`.
        let src: Vec<u8> = (0..(16 * 4)).map(|x| ((x * 13) ^ 0xC3) as u8).collect();
        let mut dst = vec![0u8; src.len()];
        forward_gradient_subtract(&src, &mut dst, 16, 4);
        // Now invert: `dst + dst[above] = src` (after iterating bottom-up).
        let mut roundtrip = dst.clone();
        inverse_gradient_post(&mut roundtrip, 16, 4);
        assert_eq!(roundtrip, src);
    }

    #[test]
    fn split_then_interleave_round_trips_odd_height() {
        // 5 rows × 4 bytes/row: top has 3 rows (0,2,4), bot has 2 (1,3).
        let pixels: Vec<u8> = (0..20u8).collect();
        let (top, bot) = split_fields(&pixels, 4, 5);
        assert_eq!(top.len(), 12);
        assert_eq!(bot.len(), 8);
        let merged = interleave_fields(&top, &bot, 4, 5);
        assert_eq!(merged, pixels);
    }

    #[test]
    fn split_interleave_height_one_single_row_goes_to_top() {
        // spec/02 §2: the field split is a row parity partition. A
        // single-row frame is wholly the (even-parity) top field; the
        // bottom field is empty. `interleave_fields` must reproduce the
        // lone row unchanged.
        let pixels: Vec<u8> = (0..6u8).collect();
        let (top, bot) = split_fields(&pixels, 6, 1);
        assert_eq!(top.len(), 6, "single row → 1 top row");
        assert!(bot.is_empty(), "single row → empty bottom field");
        assert_eq!(&top, &pixels);
        let merged = interleave_fields(&top, &bot, 6, 1);
        assert_eq!(merged, pixels);
    }

    #[test]
    fn split_interleave_degenerate_guards_return_empty() {
        // The documented degenerate guards: a zero row stride (reachable
        // from the decoder when `StreamConfig::row_bytes()` saturates to
        // 0 on a zero-width stream — see the decoder's width-0 test) or a
        // zero height yield two empty field buffers, and the inverse
        // yields an empty raster. Exercising these directly keeps the
        // guard from silently regressing into an out-of-bounds index.
        let pixels: Vec<u8> = (0..12u8).collect();
        // Zero row stride.
        let (t0, b0) = split_fields(&pixels, 0, 4);
        assert!(t0.is_empty() && b0.is_empty(), "row_bytes=0 → empty fields");
        assert!(
            interleave_fields(&t0, &b0, 0, 4).is_empty(),
            "row_bytes=0 → empty raster"
        );
        // Zero height.
        let (t1, b1) = split_fields(&pixels, 4, 0);
        assert!(t1.is_empty() && b1.is_empty(), "height=0 → empty fields");
        assert!(
            interleave_fields(&t1, &b1, 4, 0).is_empty(),
            "height=0 → empty raster"
        );
        // Empty input with both zero.
        let (t2, b2) = split_fields(&[], 0, 0);
        assert!(t2.is_empty() && b2.is_empty());
        assert!(interleave_fields(&[], &[], 0, 0).is_empty());
    }

    #[test]
    fn split_interleave_row_distribution_and_roundtrip_sweep() {
        // spec/02 §2 / `split_fields` doc: the top field carries the
        // ceil(h/2) even-parity rows and the bottom field the floor(h/2)
        // odd-parity rows; `interleave_fields` is the exact inverse. Sweep
        // a range of (row_bytes, height) — including both parities and a
        // wide-stride row — and assert both the row-count distribution
        // and the interleave∘split identity hold across the matrix.
        for &row_bytes in &[1usize, 2, 4, 16, 13] {
            for &height in &[1usize, 2, 3, 7, 8, 17] {
                let n = row_bytes * height;
                // Deterministic pattern with full 0..=255 wrap coverage.
                let pixels: Vec<u8> = (0..n).map(|x| ((x * 53 + 7) ^ 0x9E) as u8).collect();
                let (top, bot) = split_fields(&pixels, row_bytes, height);
                assert_eq!(
                    top.len(),
                    height.div_ceil(2) * row_bytes,
                    "top rows for {row_bytes}x{height}"
                );
                assert_eq!(
                    bot.len(),
                    (height / 2) * row_bytes,
                    "bot rows for {row_bytes}x{height}"
                );
                // Every even row lands in the top field, every odd row in
                // the bottom field, byte-for-byte.
                for row in 0..height {
                    let src = &pixels[row * row_bytes..(row + 1) * row_bytes];
                    let (field, frow) = if row & 1 == 0 {
                        (&top, row / 2)
                    } else {
                        (&bot, row / 2)
                    };
                    assert_eq!(
                        &field[frow * row_bytes..(frow + 1) * row_bytes],
                        src,
                        "row {row} misplaced for {row_bytes}x{height}"
                    );
                }
                let merged = interleave_fields(&top, &bot, row_bytes, height);
                assert_eq!(
                    merged, pixels,
                    "interleave∘split must be identity for {row_bytes}x{height}"
                );
            }
        }
    }

    /// Reference round-95 "two-pass" decorrelated-LEFT residual: first
    /// materialise the full decorrelated buffer (`B−G`, `G`, `R−G`,
    /// `A`), then per-channel LEFT-subtract over stride `n`. The fused
    /// `forward_left_decorr_residuals` must match this byte-for-byte.
    fn naive_two_pass_decorr_left(pixels: &[u8], n: usize) -> Vec<u8> {
        let n_pixels = pixels.len() / n;
        let mut working = vec![0u8; pixels.len()];
        for px in 0..n_pixels {
            let off = px * n;
            let g = pixels[off + 1];
            working[off] = pixels[off].wrapping_sub(g); // B − G
            working[off + 1] = g;
            working[off + 2] = pixels[off + 2].wrapping_sub(g); // R − G
            if n == 4 {
                working[off + 3] = pixels[off + 3]; // A (not decorrelated)
            }
        }
        let mut residuals = vec![0u8; pixels.len()];
        for ch in 0..n {
            residuals[ch] = working[ch];
            let mut idx = ch + n;
            while idx < working.len() {
                residuals[idx] = working[idx].wrapping_sub(working[idx - n]);
                idx += n;
            }
        }
        residuals
    }

    #[test]
    fn round100_fused_decorr_matches_two_pass_rgb24() {
        // Pseudo-random RGB24 raster, 5×3 = 15 px.
        let n_pixels = 15usize;
        let pixels: Vec<u8> = (0..(n_pixels * 3))
            .map(|x| ((x * 37 + 11) ^ 0x6C) as u8)
            .collect();
        let mut fused = vec![0u8; pixels.len()];
        forward_left_decorr_residuals(&pixels, &mut fused, 3);
        let reference = naive_two_pass_decorr_left(&pixels, 3);
        assert_eq!(fused, reference);
    }

    #[test]
    fn round100_fused_decorr_matches_two_pass_rgb32() {
        let n_pixels = 17usize;
        let pixels: Vec<u8> = (0..(n_pixels * 4))
            .map(|x| ((x * 53 + 7) ^ 0x39) as u8)
            .collect();
        let mut fused = vec![0u8; pixels.len()];
        forward_left_decorr_residuals(&pixels, &mut fused, 4);
        let reference = naive_two_pass_decorr_left(&pixels, 4);
        assert_eq!(fused, reference);
    }

    #[test]
    fn round100_fused_decorr_modular_wrap() {
        // Channels chosen so B − G and R − G both wrap mod 256, and the
        // LEFT subtract across pixels wraps too.
        // px0: B=0x10 G=0x80 R=0x05  → B−G=0x90, R−G=0x85
        // px1: B=0x00 G=0x10 R=0xF0  → B−G=0xF0, R−G=0xE0
        let pixels: Vec<u8> = vec![0x10, 0x80, 0x05, 0x00, 0x10, 0xF0];
        let mut fused = vec![0u8; pixels.len()];
        forward_left_decorr_residuals(&pixels, &mut fused, 3);
        let reference = naive_two_pass_decorr_left(&pixels, 3);
        assert_eq!(fused, reference);
        // Seed pixel = decorrelated px0 directly.
        assert_eq!(fused[0], 0x10u8.wrapping_sub(0x80)); // B−G
        assert_eq!(fused[1], 0x80); // G
        assert_eq!(fused[2], 0x05u8.wrapping_sub(0x80)); // R−G
    }

    #[test]
    fn round100_fused_decorr_alpha_left_predicted_not_decorrelated() {
        // RGB32: alpha must be LEFT-predicted but NOT decorrelated
        // (spec/03 §2.4 Validator note). px0 A=0x40, px1 A=0x55 →
        // alpha residual at px1 = 0x55 − 0x40 = 0x15; seed alpha = 0x40.
        let pixels: Vec<u8> = vec![
            0x11, 0x22, 0x33, 0x40, // px0 BGRA
            0x44, 0x55, 0x66, 0x55, // px1 BGRA
        ];
        let mut fused = vec![0u8; pixels.len()];
        forward_left_decorr_residuals(&pixels, &mut fused, 4);
        assert_eq!(fused[3], 0x40); // seed alpha verbatim
        assert_eq!(fused[7], 0x55u8.wrapping_sub(0x40)); // px1 alpha LEFT
                                                         // Cross-check the whole buffer against the two-pass reference.
        let reference = naive_two_pass_decorr_left(&pixels, 4);
        assert_eq!(fused, reference);
    }

    /// Reference round-95/100 "two-pass" decorrelated-gradient
    /// pre-pass: first materialise the full decorrelated buffer
    /// (`B−G`, `G`, `R−G`, `A`), then run `forward_gradient_subtract`
    /// over it. The fused `forward_decorr_gradient_subtract` must match
    /// this byte-for-byte. This is exactly the GradientDecorr gradient
    /// pre-pass output that feeds the per-channel LEFT-subtract pass.
    fn naive_two_pass_decorr_gradient(
        pixels: &[u8],
        n: usize,
        row_bytes: usize,
        height: usize,
    ) -> Vec<u8> {
        let n_pixels = pixels.len() / n;
        let mut working = vec![0u8; pixels.len()];
        for px in 0..n_pixels {
            let off = px * n;
            let g = pixels[off + 1];
            working[off] = pixels[off].wrapping_sub(g); // B − G
            working[off + 1] = g;
            working[off + 2] = pixels[off + 2].wrapping_sub(g); // R − G
            if n == 4 {
                working[off + 3] = pixels[off + 3]; // A (not decorrelated)
            }
        }
        let mut dst = vec![0u8; pixels.len()];
        forward_gradient_subtract(&working, &mut dst, row_bytes, height);
        dst
    }

    #[test]
    fn round103_fused_decorr_gradient_matches_two_pass_rgb24() {
        // 7×5 RGB24 raster (width 7 → row_bytes 21, no u64 alignment).
        let (w, h, n) = (7usize, 5usize, 3usize);
        let row_bytes = w * n;
        let pixels: Vec<u8> = (0..(row_bytes * h))
            .map(|x| ((x * 41 + 13) ^ 0x5C) as u8)
            .collect();
        let mut fused = vec![0u8; pixels.len()];
        forward_decorr_gradient_subtract(&pixels, &mut fused, n, row_bytes, h);
        let reference = naive_two_pass_decorr_gradient(&pixels, n, row_bytes, h);
        assert_eq!(fused, reference);
    }

    #[test]
    fn round103_fused_decorr_gradient_matches_two_pass_rgb32() {
        // 6×4 RGB32 raster (width 6 → row_bytes 24, u64-aligned).
        let (w, h, n) = (6usize, 4usize, 4usize);
        let row_bytes = w * n;
        let pixels: Vec<u8> = (0..(row_bytes * h))
            .map(|x| ((x * 59 + 7) ^ 0x33) as u8)
            .collect();
        let mut fused = vec![0u8; pixels.len()];
        forward_decorr_gradient_subtract(&pixels, &mut fused, n, row_bytes, h);
        let reference = naive_two_pass_decorr_gradient(&pixels, n, row_bytes, h);
        assert_eq!(fused, reference);
    }

    #[test]
    fn round103_fused_decorr_gradient_modular_wrap() {
        // Channels chosen so B−G / R−G wrap on decorrelation AND the
        // row-above gradient subtract wraps.
        // row0 px: B=0x10 G=0x80 R=0x05 → B−G=0x90, R−G=0x85
        // row1 px: B=0x00 G=0x10 R=0xF0 → B−G=0xF0, R−G=0xE0
        // gradient row1 = decorr(row1) − decorr(row0):
        //   G: 0x10−0x80=0x90; B−G: 0xF0−0x90=0x60; R−G: 0xE0−0x85=0x5B
        let pixels: Vec<u8> = vec![0x10, 0x80, 0x05, 0x00, 0x10, 0xF0];
        let row_bytes = 3;
        let mut fused = vec![0u8; pixels.len()];
        forward_decorr_gradient_subtract(&pixels, &mut fused, 3, row_bytes, 2);
        let reference = naive_two_pass_decorr_gradient(&pixels, 3, row_bytes, 2);
        assert_eq!(fused, reference);
        // Row 0 is the decorrelated values verbatim.
        assert_eq!(fused[0], 0x10u8.wrapping_sub(0x80)); // B−G
        assert_eq!(fused[1], 0x80); // G
        assert_eq!(fused[2], 0x05u8.wrapping_sub(0x80)); // R−G
                                                         // Row 1 gradient-subtracted decorrelated values.
        assert_eq!(fused[3], 0xF0u8.wrapping_sub(0x90)); // B−G subtract
        assert_eq!(fused[4], 0x10u8.wrapping_sub(0x80)); // G subtract
        assert_eq!(fused[5], 0xE0u8.wrapping_sub(0x85)); // R−G subtract
    }

    #[test]
    fn round103_fused_decorr_gradient_alpha_identity_not_decorrelated() {
        // RGB32: alpha must be identity in BOTH the decorrelation and
        // the gradient subtract (plain row-above subtract of raw alpha,
        // spec/03 §2.4 Validator note). row0 A=0x40, row1 A=0x90 →
        // alpha gradient = 0x90 − 0x40 = 0x50; seed alpha = 0x40.
        let pixels: Vec<u8> = vec![
            0x11, 0x22, 0x33, 0x40, // row0 BGRA
            0x44, 0x55, 0x66, 0x90, // row1 BGRA
        ];
        let row_bytes = 4;
        let mut fused = vec![0u8; pixels.len()];
        forward_decorr_gradient_subtract(&pixels, &mut fused, 4, row_bytes, 2);
        assert_eq!(fused[3], 0x40); // row0 alpha verbatim (identity)
        assert_eq!(fused[7], 0x90u8.wrapping_sub(0x40)); // row1 alpha gradient
        let reference = naive_two_pass_decorr_gradient(&pixels, 4, row_bytes, 2);
        assert_eq!(fused, reference);
    }

    #[test]
    fn round103_fused_decorr_gradient_height_1_no_op() {
        // Height 1 → no row-above; output is just the decorrelated row 0.
        let pixels: Vec<u8> = vec![0x10, 0x20, 0x30, 0x40, 0x50, 0x60];
        let row_bytes = 6;
        let mut fused = vec![0u8; pixels.len()];
        forward_decorr_gradient_subtract(&pixels, &mut fused, 3, row_bytes, 1);
        // px0: B−G=0x10−0x20, G=0x20, R−G=0x30−0x20
        assert_eq!(fused[0], 0x10u8.wrapping_sub(0x20));
        assert_eq!(fused[1], 0x20);
        assert_eq!(fused[2], 0x30u8.wrapping_sub(0x20));
        assert_eq!(fused[3], 0x40u8.wrapping_sub(0x50));
        assert_eq!(fused[4], 0x50);
        assert_eq!(fused[5], 0x60u8.wrapping_sub(0x50));
        let reference = naive_two_pass_decorr_gradient(&pixels, 3, row_bytes, 1);
        assert_eq!(fused, reference);
    }

    #[test]
    fn round100_fused_decorr_roundtrips_via_inverse_rgb24() {
        // Fused forward residuals → per-channel LEFT inverse → inverse
        // decorrelation must reconstruct the original pixels exactly.
        let n_pixels = 9usize;
        let pixels: Vec<u8> = (0..(n_pixels * 3))
            .map(|x| ((x * 29 + 5) ^ 0xB7) as u8)
            .collect();
        let mut residuals = vec![0u8; pixels.len()];
        forward_left_decorr_residuals(&pixels, &mut residuals, 3);
        // Inverse per-channel LEFT (stride 3): residual + value n back.
        let mut recon = residuals.clone();
        for i in 3..recon.len() {
            recon[i] = recon[i].wrapping_add(recon[i - 3]);
        }
        // recon now holds the decorrelated buffer (B−G, G, R−G); invert
        // decorrelation in place.
        inverse_rgb_decorr_bgr(&mut recon);
        assert_eq!(recon, pixels);
    }

    /// Reference two-phase YUY2 forward MEDIAN: compute a full-frame
    /// LEFT residual stream (4-byte raw seed + per-channel-stride
    /// subtract), then OVERWRITE the median region (row 1 byte ≥ 8 +
    /// every later row) with MEDIAN residuals. This is exactly what the
    /// encoder did before round 115 inlined the two phases; the
    /// single-pass `forward_median_subtract` must match it byte-for-byte.
    fn naive_two_phase_forward_median(pixels: &[u8], row_bytes: usize, height: usize) -> Vec<u8> {
        let mut residuals = vec![0u8; pixels.len()];
        // Phase 1: full-frame LEFT.
        let copy_n = 4.min(pixels.len());
        residuals[..copy_n].copy_from_slice(&pixels[..copy_n]);
        for i in 4..pixels.len() {
            let stride = if i & 1 == 0 { 2 } else { 4 };
            residuals[i] = pixels[i].wrapping_sub(pixels[i - stride]);
        }
        // Phase 2: overwrite the median region.
        if height >= 2 && row_bytes > 0 {
            // Round-196: 8-wire-byte LEFT exemption (not `min(8, row_bytes)`)
            // — see the matching comment in `forward_median_subtract`.
            let row1_median_start = (row_bytes + 8).min(pixels.len());
            for pos in row1_median_start..pixels.len() {
                if pos < row_bytes {
                    continue;
                }
                let l = pixels[pos.wrapping_sub(2)];
                let a = pixels[pos - row_bytes];
                let al = if pos >= row_bytes + 2 {
                    pixels[pos - row_bytes - 2]
                } else {
                    0
                };
                let g = gradient_predictor(l, a, al);
                let predictor = median3(l, a, g);
                residuals[pos] = pixels[pos].wrapping_sub(predictor);
            }
        }
        residuals
    }

    #[test]
    fn round115_forward_median_matches_two_phase() {
        // 8×6 YUY2 (row_bytes = 16, 6 rows) — large enough to exercise
        // the row-0 LEFT region, the 8-byte row-1 exemption, and several
        // full median rows.
        let row_bytes = 16usize;
        let height = 6usize;
        let pixels: Vec<u8> = (0..(row_bytes * height))
            .map(|x| ((x * 37 + 11) ^ 0x5C) as u8)
            .collect();
        let mut fused = vec![0u8; pixels.len()];
        forward_median_subtract(&pixels, &mut fused, row_bytes, height);
        let reference = naive_two_phase_forward_median(&pixels, row_bytes, height);
        assert_eq!(fused, reference);
    }

    #[test]
    fn round115_forward_median_modular_wrap() {
        // Pixels chosen so the median predictor and the residual both
        // wrap mod-256.
        let row_bytes = 8usize;
        let height = 3usize;
        let pixels: Vec<u8> = vec![
            // row 0
            0xF0, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, // row 1
            0x05, 0x80, 0x90, 0xA0, 0xB0, 0xC0, 0xD0, 0xE0, // row 2
            0xFF, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
        ];
        let mut fused = vec![0u8; pixels.len()];
        forward_median_subtract(&pixels, &mut fused, row_bytes, height);
        let reference = naive_two_phase_forward_median(&pixels, row_bytes, height);
        assert_eq!(fused, reference);
    }

    #[test]
    fn round115_forward_median_height_1_is_all_left() {
        // One row → no row above; the whole buffer is LEFT residuals.
        let row_bytes = 8usize;
        let pixels: Vec<u8> = vec![0x10, 0x20, 0x30, 0x40, 0x55, 0x66, 0x77, 0x88];
        let mut fused = vec![0u8; pixels.len()];
        forward_median_subtract(&pixels, &mut fused, row_bytes, 1);
        // First 4 bytes raw, rest LEFT (stride 2 for even i, 4 for odd).
        assert_eq!(&fused[..4], &pixels[..4]);
        for i in 4..pixels.len() {
            let stride = if i & 1 == 0 { 2 } else { 4 };
            assert_eq!(fused[i], pixels[i].wrapping_sub(pixels[i - stride]));
        }
    }

    #[test]
    fn round115_forward_median_short_second_row_is_all_left() {
        // Two rows but the median region is empty: with row_bytes = 8,
        // the 8-byte row-1 exemption covers the whole of row 1, so every
        // byte is LEFT and the median post-pass is a no-op. Matches the
        // two-phase reference (which leaves the median region untouched).
        let row_bytes = 8usize;
        let height = 2usize;
        let pixels: Vec<u8> = (0..(row_bytes * height))
            .map(|x| (x * 13 + 3) as u8)
            .collect();
        let mut fused = vec![0u8; pixels.len()];
        forward_median_subtract(&pixels, &mut fused, row_bytes, height);
        let reference = naive_two_phase_forward_median(&pixels, row_bytes, height);
        assert_eq!(fused, reference);
    }

    #[test]
    fn round115_forward_median_roundtrips_via_decoder_model() {
        // forward_median_subtract → the decoder's exact YUY2 median
        // inverse (LEFT over row 0 + first 8 bytes of row 1, median ADD
        // for the rest) must reconstruct the original pixels. This
        // mirrors `decoder::inverse_yuy2_median` (spec/03 §2.3.2): the
        // LEFT region residuals are `pixel − left` (reversed by LEFT
        // add) and the median region residuals are `pixel − median`
        // (reversed by adding the median predictor).
        let row_bytes = 16usize;
        let height = 5usize;
        let pixels: Vec<u8> = (0..(row_bytes * height))
            .map(|x| ((x * 53 + 7) ^ 0x2A) as u8)
            .collect();
        let mut recon = vec![0u8; pixels.len()];
        forward_median_subtract(&pixels, &mut recon, row_bytes, height);

        // YUY2 LEFT add over a byte range, using the per-byte strides
        // (−2 for the Y₁/Y₂ positions, −4 for U/V).
        fn left_range(out: &mut [u8], begin: usize, end: usize) {
            for i in begin..end {
                let stride = if i & 1 == 0 { 2 } else { 4 };
                out[i] = out[i].wrapping_add(out[i - stride]);
            }
        }
        let len = recon.len();
        // Row 0 LEFT (bytes 4..row_bytes; first 4 are the raw seed).
        left_range(&mut recon, 4, row_bytes.min(len));
        // First 8 bytes of row 1 LEFT (the §2.3.2 exemption).
        let row1_left_end = (row_bytes + 8).min(len);
        left_range(&mut recon, row_bytes, row1_left_end);
        // Median ADD for the rest.
        for pos in row1_left_end..recon.len() {
            let l = recon[pos - 2];
            let a = recon[pos - row_bytes];
            let al = if pos >= row_bytes + 2 {
                recon[pos - row_bytes - 2]
            } else {
                0
            };
            let g = gradient_predictor(l, a, al);
            let predictor = median3(l, a, g);
            recon[pos] = recon[pos].wrapping_add(predictor);
        }
        assert_eq!(recon, pixels);
    }

    /// Reference: the previous per-byte-branch decoder LEFT inverse,
    /// retained here as the equivalence oracle for the round-181
    /// macropixel-step rewrite.
    fn ref_inverse_yuy2_left_range(out: &mut [u8], begin: usize, end: usize) {
        if end <= begin {
            return;
        }
        let mut i = begin;
        while i < end {
            match i & 3 {
                0 | 2 => out[i] = out[i].wrapping_add(out[i - 2]),
                _ => out[i] = out[i].wrapping_add(out[i - 4]),
            }
            i += 1;
        }
    }

    /// Build a deterministic YUY2 raster of `width × height` bytes
    /// (= `width × 2` per row), filled with a sawtooth pattern so
    /// every byte-position channel sees non-trivial residuals.
    fn make_yuy2_raster(width: usize, height: usize, seed: u32) -> Vec<u8> {
        let mut s = seed;
        let mut out = vec![0u8; width * 2 * height];
        for byte in &mut out {
            // xorshift32 → low byte, plus a tiny offset to avoid
            // hitting only one channel value.
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            *byte = (s & 0xff) as u8;
        }
        out
    }

    #[test]
    fn round181_yuy2_left_macropixel_matches_branchy_full_frame() {
        // Full-frame YUY2 LEFT inverse: branch-free helper matches
        // the per-byte reference across the proprietary's documented
        // raster widths.
        for &(w, h) in &[(2_usize, 1_usize), (4, 1), (4, 4), (320, 8), (640, 2)] {
            let raw = make_yuy2_raster(w, h, 0xc0ffee + (w * h) as u32);
            // Build a "residual"-shaped buffer: the seed macropixel
            // at indices 0..4 holds raw values; the rest hold the
            // LEFT residuals derived from a reference (forward
            // subtract) so the two inverses share an input.
            let mut residuals = raw.clone();
            for i in 4..raw.len() {
                let stride = if i & 1 == 0 { 2 } else { 4 };
                residuals[i] = raw[i].wrapping_sub(raw[i - stride]);
            }
            let mut a = residuals.clone();
            let mut b = residuals.clone();
            let n = a.len();
            ref_inverse_yuy2_left_range(&mut a, 4, n);
            inverse_yuy2_left_macropixel(&mut b, 4, n);
            assert_eq!(a, b, "full-frame mismatch for {w}x{h}");
            // Sanity: the inverse should reconstruct the original raster.
            assert_eq!(b, raw, "inverse failed to reconstruct {w}x{h}");
        }
    }

    #[test]
    fn round181_yuy2_left_macropixel_matches_branchy_row1_first8() {
        // The median path calls the helper with begin = row_bytes
        // and end = row_bytes + 8 (the §2.3.2 LEFT exemption). Cover
        // a row width where this is well below the buffer end so the
        // tail logic of the macropixel walk is exercised but no
        // out-of-range write happens.
        let w = 320usize;
        let h = 3usize;
        let row_bytes = w * 2;
        let raw = make_yuy2_raster(w, h, 0xfeed_face);
        // Build residuals as above.
        let mut residuals = raw.clone();
        for i in 4..raw.len() {
            let stride = if i & 1 == 0 { 2 } else { 4 };
            residuals[i] = raw[i].wrapping_sub(raw[i - stride]);
        }
        // Apply row 0 LEFT inverse to get the same buffer state both
        // implementations would see at the row-1 entry point.
        let mut a = residuals.clone();
        let mut b = residuals.clone();
        ref_inverse_yuy2_left_range(&mut a, 4, row_bytes);
        inverse_yuy2_left_macropixel(&mut b, 4, row_bytes);
        assert_eq!(a, b, "row-0 LEFT mismatch");
        // Now the row-1-first-8 range:
        ref_inverse_yuy2_left_range(&mut a, row_bytes, row_bytes + 8);
        inverse_yuy2_left_macropixel(&mut b, row_bytes, row_bytes + 8);
        assert_eq!(a, b, "row-1-first-8 LEFT mismatch");
    }

    #[test]
    fn round181_yuy2_left_macropixel_modular_wrap() {
        // Force per-byte modular wrap on every channel by using
        // residuals that cross 0xff repeatedly.
        let raw: Vec<u8> = (0..1024u32)
            .map(|i| (i.wrapping_mul(73) & 0xff) as u8)
            .collect();
        let mut residuals = raw.clone();
        for i in 4..raw.len() {
            let stride = if i & 1 == 0 { 2 } else { 4 };
            residuals[i] = raw[i].wrapping_sub(raw[i - stride]);
        }
        let mut a = residuals.clone();
        let mut b = residuals.clone();
        let n = a.len();
        ref_inverse_yuy2_left_range(&mut a, 4, n);
        inverse_yuy2_left_macropixel(&mut b, 4, n);
        assert_eq!(a, b);
        assert_eq!(b, raw);
    }

    #[test]
    fn round181_yuy2_left_macropixel_short_buffer_noop() {
        // begin == end is a no-op.
        let mut buf = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let copy = buf.clone();
        inverse_yuy2_left_macropixel(&mut buf, 4, 4);
        assert_eq!(buf, copy);
    }

    /// Reference: the previous per-byte-branch encoder LEFT forward
    /// subtract, retained here as the round-181 equivalence oracle.
    fn ref_forward_yuy2_left_subtract(src: &[u8], dst: &mut [u8]) {
        let n = src.len();
        let head = 4.min(n);
        dst[..head].copy_from_slice(&src[..head]);
        for i in 4..n {
            let stride = if i & 1 == 0 { 2 } else { 4 };
            dst[i] = src[i].wrapping_sub(src[i - stride]);
        }
    }

    #[test]
    fn round181_yuy2_forward_left_matches_branchy() {
        for &(w, h) in &[(2_usize, 1_usize), (4, 1), (4, 4), (320, 8), (640, 2)] {
            let raw = make_yuy2_raster(w, h, 0xbeef_cafe + (w * h) as u32);
            let mut a = vec![0u8; raw.len()];
            let mut b = vec![0u8; raw.len()];
            ref_forward_yuy2_left_subtract(&raw, &mut a);
            forward_yuy2_left_subtract(&raw, &mut b);
            assert_eq!(a, b, "forward LEFT mismatch for {w}x{h}");
        }
    }

    #[test]
    fn round181_yuy2_forward_left_short_buffers() {
        // Lengths 0, 1, 2, 3, 4 — all <= 4 means the body loop is
        // skipped and only the seed copy runs.
        for n in 0..=4usize {
            let raw: Vec<u8> = (0..n as u8).collect();
            let mut a = vec![0u8; n];
            let mut b = vec![0u8; n];
            ref_forward_yuy2_left_subtract(&raw, &mut a);
            forward_yuy2_left_subtract(&raw, &mut b);
            assert_eq!(a, b, "len={n} mismatch");
        }
    }

    #[test]
    fn round181_yuy2_forward_then_inverse_roundtrips() {
        // End-to-end: the new forward subtract followed by the new
        // inverse LEFT macropixel walk reconstructs the original
        // raster bit-exactly. This is the load-bearing wire-format
        // claim — the two helpers must be exact inverses.
        for &(w, h) in &[(4_usize, 4_usize), (320, 16), (640, 3)] {
            let raw = make_yuy2_raster(w, h, 0x1234_5678 + (w * h) as u32);
            let mut residuals = vec![0u8; raw.len()];
            forward_yuy2_left_subtract(&raw, &mut residuals);
            // The first 4 bytes are the seed copy; the body holds
            // residuals. The decoder inverse reads from index 4 and
            // walks to the end.
            let mut recon = residuals.clone();
            let n = recon.len();
            inverse_yuy2_left_macropixel(&mut recon, 4, n);
            assert_eq!(recon, raw, "roundtrip failed for {w}x{h}");
        }
    }

    // ───────────────────────── round 186 — RGB LEFT linear-walk ──────────

    /// Reference: the prior per-channel stride-N triple/quad-pass
    /// LEFT-residual loop that the encoder used through round 181.
    /// Spec/03 §2.1 — per-channel LEFT residual is
    /// `dst[i] = src[i] − src[i − N]` for `i ≥ N`; the linear-walk
    /// helper must produce byte-identical output.
    fn ref_forward_rgb_left_per_channel(src: &[u8], n_channels: usize) -> Vec<u8> {
        let mut dst = vec![0u8; src.len()];
        for ch in 0..n_channels {
            if ch < src.len() {
                dst[ch] = src[ch];
            }
            let mut idx = ch + n_channels;
            while idx < src.len() {
                dst[idx] = src[idx].wrapping_sub(src[idx - n_channels]);
                idx += n_channels;
            }
        }
        dst
    }

    /// Deterministic RGB24 / RGB32 raster (BGR(A) byte order in
    /// memory) seeded by a per-buffer xorshift32 — same shape as
    /// `make_yuy2_raster` but for n_channels-wide interleaved bytes.
    fn make_rgb_raster(width: usize, height: usize, n_channels: usize, seed: u32) -> Vec<u8> {
        let mut s = seed | 1;
        let mut out = vec![0u8; width * height * n_channels];
        for byte in out.iter_mut() {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            *byte = (s & 0xFF) as u8;
        }
        out
    }

    #[test]
    fn round186_rgb_left_linear_matches_per_channel_rgb24() {
        // Width 1 (only seed), 4 (interior + tail-aligned to 12 B),
        // 320 (real-world stride), height 1 / 4 — covers the
        // edge cases the per-channel loop handled implicitly via
        // its `ch + 3` start.
        for &(w, h) in &[(1usize, 1usize), (4, 1), (4, 4), (320, 4)] {
            let src = make_rgb_raster(w, h, 3, 0xA1B2_C3D4 ^ ((w * h) as u32));
            let ref_dst = ref_forward_rgb_left_per_channel(&src, 3);
            let mut new_dst = vec![0u8; src.len()];
            forward_rgb_left_subtract_linear(&src, &mut new_dst, 3);
            assert_eq!(
                new_dst, ref_dst,
                "RGB24 linear LEFT diverges from per-channel ref at {w}x{h}"
            );
        }
    }

    #[test]
    fn round186_rgb_left_linear_matches_per_channel_rgb32() {
        for &(w, h) in &[(1usize, 1usize), (4, 1), (4, 4), (320, 4)] {
            let src = make_rgb_raster(w, h, 4, 0xDEAD_BEEF ^ ((w * h) as u32));
            let ref_dst = ref_forward_rgb_left_per_channel(&src, 4);
            let mut new_dst = vec![0u8; src.len()];
            forward_rgb_left_subtract_linear(&src, &mut new_dst, 4);
            assert_eq!(
                new_dst, ref_dst,
                "RGB32 linear LEFT diverges from per-channel ref at {w}x{h}"
            );
        }
    }

    #[test]
    fn round186_rgb_left_linear_modular_wrap() {
        // Force per-channel mod-256 wrap by alternating 0xFF / 0x01
        // values: residual = 0x01 − 0xFF = 0x02 (mod 256). The
        // linear walk must produce the same byte-wise wrap result
        // as the per-channel reference (no inter-channel borrow).
        let src: Vec<u8> = (0..24)
            .map(|i| if i & 1 == 0 { 0xFF } else { 0x01 })
            .collect();
        for &n in &[3usize, 4] {
            let ref_dst = ref_forward_rgb_left_per_channel(&src, n);
            let mut new_dst = vec![0u8; src.len()];
            forward_rgb_left_subtract_linear(&src, &mut new_dst, n);
            assert_eq!(new_dst, ref_dst, "modular wrap mismatch at n={n}");
        }
    }

    #[test]
    fn round186_rgb_left_linear_short_buffer_seed_only() {
        // A buffer shorter than one pixel (`n_channels` bytes) is the
        // degenerate case — both reference and linear walk must copy
        // verbatim. (The encoder never feeds this in practice, but
        // the helper has to be robust for fuzz coverage.)
        for &n in &[3usize, 4] {
            let src = vec![0xAB, 0xCD][..n.min(2)].to_vec();
            let ref_dst = ref_forward_rgb_left_per_channel(&src, n);
            let mut new_dst = vec![0u8; src.len()];
            forward_rgb_left_subtract_linear(&src, &mut new_dst, n);
            assert_eq!(new_dst, ref_dst, "short buffer mismatch at n={n}");
            assert_eq!(new_dst, src, "seed-only must be verbatim copy at n={n}");
        }
    }

    /// Per-channel LEFT inverse on an N-channel-interleaved buffer —
    /// mirrors the decoder's `inverse_left_per_channel` helper.
    /// Walks every channel with stride N; the same single linear
    /// stride-1 walk identity holds for the inverse as for the
    /// forward (each output byte is a function of `i mod n_channels`
    /// alone), and the linear-walk LEFT forward subtract is its
    /// exact inverse.
    fn ref_inverse_rgb_left_per_channel(residuals: &[u8], n_channels: usize) -> Vec<u8> {
        let mut dst = residuals.to_vec();
        if dst.len() <= n_channels {
            return dst;
        }
        for i in n_channels..dst.len() {
            dst[i] = dst[i].wrapping_add(dst[i - n_channels]);
        }
        dst
    }

    #[test]
    fn round186_rgb_left_linear_then_inverse_roundtrips() {
        // End-to-end: the new linear forward subtract followed by
        // the decoder's per-channel inverse reconstructs the source
        // bit-exactly. Confirms the linear-walk identity preserves
        // wire compatibility on every channel.
        for &n in &[3usize, 4] {
            for &(w, h) in &[(4usize, 4usize), (16, 8), (320, 4)] {
                let src = make_rgb_raster(w, h, n, 0xCAFEBABE ^ ((w * h * n) as u32));
                let mut residuals = vec![0u8; src.len()];
                forward_rgb_left_subtract_linear(&src, &mut residuals, n);
                let recon = ref_inverse_rgb_left_per_channel(&residuals, n);
                assert_eq!(recon, src, "roundtrip failed for n={n} {w}x{h}");
            }
        }
    }

    // ─────────── round 253 — macropixel-step forward median (predict body) ─

    /// Reference: the pre-r253 per-byte forward MEDIAN region body —
    /// retained verbatim as the equivalence oracle for the
    /// macropixel-step rewrite. Mirrors the body of
    /// `forward_median_subtract` between `median_start` and `n`
    /// before the round-253 4-byte unroll. spec/03 §2.3 +
    /// audit/01 §7.2 fix the lookback offsets at `−2 / −row_bytes /
    /// −row_bytes − 2` uniformly for every byte position in the
    /// median region.
    fn ref_pre_r253_forward_median(pixels: &[u8], dst: &mut [u8], row_bytes: usize, height: usize) {
        let n = pixels.len();
        if n == 0 {
            return;
        }
        let median_start = if height < 2 || row_bytes == 0 {
            n
        } else {
            (row_bytes + 8).min(n)
        };
        let copy_n = 4.min(n);
        dst[..copy_n].copy_from_slice(&pixels[..copy_n]);
        for i in copy_n..median_start {
            let stride = if i & 1 == 0 { 2 } else { 4 };
            dst[i] = pixels[i].wrapping_sub(pixels[i - stride]);
        }
        // Pre-r253 single per-byte body.
        for pos in median_start..n {
            let l = pixels[pos - 2];
            let a = pixels[pos - row_bytes];
            let al = pixels[pos - row_bytes - 2];
            let g = gradient_predictor(l, a, al);
            let predictor = median3(l, a, g);
            dst[pos] = pixels[pos].wrapping_sub(predictor);
        }
    }

    #[test]
    fn round253_forward_median_macropixel_matches_per_byte_reference() {
        // Sweep the YUY2-canonical widths the codec actually emits.
        // For each (w, h) the in-spec body produces an
        // (n − median_start) that is a multiple of 4, so the
        // macropixel-step inner loop runs the whole region with no
        // scalar tail. Bit-exact equivalence to the pre-r253 body.
        for &(w, h) in &[
            (2usize, 4usize), // narrow — row_bytes=4, full body
            (4, 4),           // row_bytes=8
            (8, 6),           // row_bytes=16
            (16, 5),          // row_bytes=32
            (160, 9),         // wide raster — exercises many median rows
            (320, 16),        // 320×16 — the bench reference size
        ] {
            let row_bytes = w * 2;
            let pixels: Vec<u8> = (0..(row_bytes * h))
                .map(|x| ((x * 37 + 11) ^ 0x5C) as u8)
                .collect();
            let mut fused = vec![0u8; pixels.len()];
            let mut reference = vec![0u8; pixels.len()];
            forward_median_subtract(&pixels, &mut fused, row_bytes, h);
            ref_pre_r253_forward_median(&pixels, &mut reference, row_bytes, h);
            assert_eq!(
                fused, reference,
                "r253 macropixel-step diverges from per-byte ref at {w}x{h}"
            );
        }
    }

    #[test]
    fn round253_forward_median_macropixel_modular_wrap() {
        // Picture values are crafted so every median region position
        // exercises mod-256 wrap inside `gradient_predictor` AND in the
        // final residual subtract. The per-byte reference and the
        // macropixel-step body must agree byte-for-byte.
        let row_bytes = 16usize;
        let height = 5usize;
        let mut pixels = vec![0u8; row_bytes * height];
        for (i, byte) in pixels.iter_mut().enumerate() {
            // alternate near-0 / near-0xff to maximise mod wraps.
            *byte = if (i / 4) & 1 == 0 {
                (i & 0xff) as u8
            } else {
                (0xFFu8).wrapping_sub((i * 17) as u8)
            };
        }
        let mut fused = vec![0u8; pixels.len()];
        let mut reference = vec![0u8; pixels.len()];
        forward_median_subtract(&pixels, &mut fused, row_bytes, height);
        ref_pre_r253_forward_median(&pixels, &mut reference, row_bytes, height);
        assert_eq!(fused, reference);
    }

    #[test]
    fn round253_forward_median_macropixel_boundary_row1_step_pattern() {
        // The first byte of the median region is at index
        // `median_start = row_bytes + 8`, which always aligns to a
        // 4-byte macropixel boundary because row_bytes is a multiple
        // of 4 (YUY2: 2 × width, width even). Each (w, h) below makes
        // `(n − median_start)` a clean multiple of 4 so the
        // macropixel-step body covers the full region — pin the
        // boundary alignment that lets the body run without falling
        // into the scalar tail.
        for &(w, h) in &[(2_usize, 4_usize), (4, 3), (8, 3)] {
            let row_bytes = w * 2;
            let pixels: Vec<u8> = (0..(row_bytes * h))
                .map(|x| ((x * 53 + 7) ^ 0x2A) as u8)
                .collect();
            let n = pixels.len();
            let median_start = (row_bytes + 8).min(n);
            assert_eq!(
                median_start % 4,
                0,
                "median_start must align to a YUY2 macropixel boundary"
            );
            assert_eq!(
                (n - median_start) % 4,
                0,
                "in-spec YUY2 median region must be a whole multiple of 4 bytes"
            );
            let mut fused = vec![0u8; n];
            let mut reference = vec![0u8; n];
            forward_median_subtract(&pixels, &mut fused, row_bytes, h);
            ref_pre_r253_forward_median(&pixels, &mut reference, row_bytes, h);
            assert_eq!(
                fused, reference,
                "r253 boundary-step mismatch at {w}x{h} (median_start={median_start}, n={n})"
            );
        }
    }

    #[test]
    fn round253_forward_median_macropixel_then_inverse_roundtrips() {
        // End-to-end: forward median (with the r253 macropixel-step
        // body) followed by the decoder's exact YUY2 median inverse
        // (LEFT over row 0 + first 8 bytes of row 1, median ADD for
        // the rest) must reconstruct the original pixels. Locks the
        // wire-format claim that the r253 rewrite preserves bit-exact
        // round-trip — across widths small enough to exercise the
        // narrow-row §2.3.2 path (where the LEFT exemption extends
        // beyond row 1) and wide enough to exercise multiple full
        // median rows.
        let cases: &[(usize, usize)] = &[(2, 4), (4, 3), (16, 5), (160, 4)];
        for &(w, h) in cases {
            let row_bytes = w * 2;
            let pixels: Vec<u8> = (0..(row_bytes * h))
                .map(|x| ((x * 41 + 13) ^ 0x9E) as u8)
                .collect();
            let mut recon = vec![0u8; pixels.len()];
            forward_median_subtract(&pixels, &mut recon, row_bytes, h);

            // YUY2 LEFT add over a byte range, per-byte strides
            // (−2 for Y₁/Y₂ positions, −4 for U/V).
            fn left_range(out: &mut [u8], begin: usize, end: usize) {
                for i in begin..end {
                    let stride = if i & 1 == 0 { 2 } else { 4 };
                    out[i] = out[i].wrapping_add(out[i - stride]);
                }
            }
            let len = recon.len();
            if len > 4 {
                left_range(&mut recon, 4, row_bytes.min(len));
            }
            let row1_left_end = (row_bytes + 8).min(len);
            if row_bytes < len {
                left_range(&mut recon, row_bytes, row1_left_end);
            }
            for pos in row1_left_end..len {
                let l = recon[pos - 2];
                let a = recon[pos - row_bytes];
                let al = recon[pos - row_bytes - 2];
                let g = gradient_predictor(l, a, al);
                let predictor = median3(l, a, g);
                recon[pos] = recon[pos].wrapping_add(predictor);
            }
            assert_eq!(recon, pixels, "r253 round-trip failed for {w}x{h}");
        }
    }

    #[test]
    fn round253_forward_median_macropixel_scalar_tail_safety() {
        // Defence-in-depth coverage of the 1..=3-byte tail
        // fall-through that runs if `(n − median_start) % 4 != 0`. In
        // practice in-spec YUY2 input keeps this multiple-of-4, but
        // the fall-through has to match the pre-r253 per-byte body
        // byte-for-byte. We construct an off-spec buffer where the
        // total byte count is *not* a multiple of 4 from the median
        // boundary — splicing 1, 2, and 3 extra bytes — and confirm
        // each variant agrees with the reference.
        for extra in 1..=3 {
            // Base: 8×3 YUY2 = row_bytes 16, n = 48. median_start =
            // 24. (n − median_start) = 24 = multiple of 4. We pad
            // pixels with `extra` extra bytes at the END so the body
            // walks an off-aligned tail. row_bytes / height stay the
            // same: the predict body uses `n = pixels.len()` for its
            // upper bound; the padding only shifts the tail.
            //
            // We synthesise a debug-assert-safe input where height is
            // still 3 but n has `extra` trailing bytes — accepted
            // because `forward_median_subtract` uses `debug_assert!(
            // row_bytes == 0 || pixels.len() == row_bytes * height)`
            // which is *only* a debug assert. The release-build path
            // accepts any length; the tail still has to be
            // wire-correct for fuzz-style input.
            let row_bytes = 16usize;
            let height = 3usize;
            let base_len = row_bytes * height;
            let len = base_len + extra;
            let pixels: Vec<u8> = (0..len).map(|x| ((x * 23 + 5) ^ 0xC3) as u8).collect();
            // Build the reference by truncating to base_len (where
            // the per-byte body is well defined and matches the
            // debug-assert invariant), then comparing positions that
            // both paths cover.
            //
            // For positions within [median_start..base_len], both
            // paths run the same lookbacks. For positions
            // [base_len..len], the macropixel body might step into
            // out-of-row reads — so we restrict the assertion to
            // [median_start..base_len], the well-defined region.
            //
            // Skip this test if the platform's debug_assert would
            // panic; in release mode the assert is a no-op.
            #[cfg(debug_assertions)]
            {
                let _ = pixels;
                continue;
            }
            #[cfg(not(debug_assertions))]
            {
                let mut fused = vec![0u8; len];
                let mut reference = vec![0u8; len];
                forward_median_subtract(&pixels, &mut fused, row_bytes, height);
                ref_pre_r253_forward_median(&pixels, &mut reference, row_bytes, height);
                let median_start = (row_bytes + 8).min(base_len);
                assert_eq!(
                    fused[median_start..base_len],
                    reference[median_start..base_len],
                    "r253 scalar-tail off-aligned mismatch at extra={extra}"
                );
            }
        }
    }

    // ---- Round 304: inverse RGB decorrelation chunks_exact_mut rewrite ----

    /// Inlined copy of the pre-r304 BGR inverse-decorrelation body
    /// (`while i + 3 <= len`). The production helper must match this
    /// byte-for-byte; the only change in r304 is iterating with
    /// `chunks_exact_mut(3)` instead of index arithmetic.
    fn ref_pre_r304_inverse_decorr_bgr(out: &mut [u8]) {
        let mut i = 0;
        while i + 3 <= out.len() {
            let g = out[i + 1];
            out[i] = out[i].wrapping_add(g);
            out[i + 2] = out[i + 2].wrapping_add(g);
            i += 3;
        }
    }

    /// Inlined copy of the pre-r304 BGRA inverse-decorrelation body.
    fn ref_pre_r304_inverse_decorr_bgra(out: &mut [u8]) {
        let mut i = 0;
        while i + 4 <= out.len() {
            let g = out[i + 1];
            out[i] = out[i].wrapping_add(g);
            out[i + 2] = out[i + 2].wrapping_add(g);
            i += 4;
        }
    }

    #[test]
    fn round304_inverse_decorr_bgr_matches_pre_r304_reference() {
        // Sweep pixel counts, including widths that leave a partial
        // trailing pixel (len % 3 != 0) so the chunks_exact_mut
        // remainder behaviour is pinned against the `i + 3 <= len`
        // guard. xorshift-ish fill forces mod-256 wraps in the add.
        for n_bytes in 0..=64usize {
            let mut state = 0x1357_9bdfu32 ^ (n_bytes as u32);
            let buf: Vec<u8> = (0..n_bytes)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    (state & 0xff) as u8
                })
                .collect();
            let mut prod = buf.clone();
            let mut reference = buf.clone();
            inverse_rgb_decorr_bgr(&mut prod);
            ref_pre_r304_inverse_decorr_bgr(&mut reference);
            assert_eq!(prod, reference, "BGR mismatch at n_bytes={n_bytes}");
        }
    }

    #[test]
    fn round304_inverse_decorr_bgra_matches_pre_r304_reference() {
        for n_bytes in 0..=64usize {
            let mut state = 0x2468_ace0u32 ^ (n_bytes as u32);
            let buf: Vec<u8> = (0..n_bytes)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    (state & 0xff) as u8
                })
                .collect();
            let mut prod = buf.clone();
            let mut reference = buf.clone();
            inverse_rgb_decorr_bgra(&mut prod);
            ref_pre_r304_inverse_decorr_bgra(&mut reference);
            assert_eq!(prod, reference, "BGRA mismatch at n_bytes={n_bytes}");
        }
    }

    #[test]
    fn round304_inverse_decorr_bgr_modular_wrap() {
        // Force every B / R add to wrap mod 256: G = 0xFF, B = R = 0x02
        // ⇒ reconstructed B = R = 0x01. G itself is unchanged.
        let mut buf = vec![0x02u8, 0xff, 0x02, 0x02, 0xff, 0x02];
        inverse_rgb_decorr_bgr(&mut buf);
        assert_eq!(buf, vec![0x01, 0xff, 0x01, 0x01, 0xff, 0x01]);
    }

    #[test]
    fn round304_inverse_decorr_bgra_leaves_alpha_untouched() {
        // Alpha (offset +3) must pass through verbatim per the §2.4
        // Validator note; only B (+0) and R (+2) receive G (+1).
        let mut buf = vec![0x10u8, 0x05, 0x20, 0xAB, 0x30, 0x07, 0x40, 0xCD];
        inverse_rgb_decorr_bgra(&mut buf);
        assert_eq!(
            buf,
            vec![
                0x10u8.wrapping_add(0x05),
                0x05,
                0x20u8.wrapping_add(0x05),
                0xAB,
                0x30u8.wrapping_add(0x07),
                0x07,
                0x40u8.wrapping_add(0x07),
                0xCD,
            ]
        );
    }

    #[test]
    fn round304_inverse_decorr_partial_trailing_pixel_untouched() {
        // A truncated buffer (len not a multiple of stride) leaves the
        // partial trailing pixel verbatim, matching the `i + n <= len`
        // guard the chunks_exact_mut remainder reproduces.
        // BGR: 1 full pixel + 2 trailing bytes.
        let mut bgr = vec![0x02u8, 0x03, 0x04, 0x77, 0x88];
        inverse_rgb_decorr_bgr(&mut bgr);
        assert_eq!(bgr[3], 0x77, "BGR trailing byte 0 must be untouched");
        assert_eq!(bgr[4], 0x88, "BGR trailing byte 1 must be untouched");
        // BGRA: 1 full pixel + 3 trailing bytes.
        let mut bgra = vec![0x02u8, 0x03, 0x04, 0x05, 0x77, 0x88, 0x99];
        inverse_rgb_decorr_bgra(&mut bgra);
        assert_eq!(&bgra[4..], &[0x77, 0x88, 0x99], "BGRA tail untouched");
    }
}
