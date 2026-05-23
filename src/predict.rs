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
    // Row 1 starts at offset row_bytes; skip the first 8 bytes that
    // the LEFT pass already produced as final values.
    let row1_start = row_bytes;
    let row1_median_start = row1_start + 8.min(row_bytes);
    // For each byte from there to end-of-buffer, replace
    // residual-with-LEFT with proper median reconstruction.
    let mut pos = row1_median_start;
    while pos < out.len() {
        // Determine row position; we work raster and step pos one byte
        // at a time, with `row_bytes` lookbacks.
        if pos < row_bytes {
            pos += 1;
            continue;
        }
        let l = out[pos.wrapping_sub(2)]; // 2-byte channel stride within YUY2.
                                          // Note: spec/03 §2.3 documents output offsets -2 / -row_stride
                                          // / -row_stride - 2 as the LEFT / ABOVE / ABOVE-LEFT in the
                                          // 4-byte-stride YUY2 layout. The "2-byte left" matches
                                          // intra-pair Y₁→Y₂ / Y₂→Y₁ alternation; for U / V it is
                                          // previous-pair's same-channel byte.
        let a = out[pos - row_bytes];
        let al = if pos >= row_bytes + 2 {
            out[pos - row_bytes - 2]
        } else {
            0
        };
        let g = gradient_predictor(l, a, al);
        let predictor = median3(l, a, g);
        // The LEFT pass produced `out[pos] = residual + L`. We need
        // to undo that and re-apply median: subtract L, then add the
        // median predictor.
        let residual = out[pos].wrapping_sub(l);
        out[pos] = residual.wrapping_add(predictor);
        pos += 1;
    }
}

/// Inverse RGB decorrelation on a 3-byte BGR-packed buffer, in place.
/// `B = (B-G) + G` and `R = (R-G) + G`, both mod 256. Walks the
/// buffer per pixel; G is at byte +1, B at +0, R at +2 (spec/02
/// §3.2's BGR ordering on the wire — but note our reconstruction
/// after the codeword-order adjustments stores G at +1 etc. when
/// the codeword stream was `G, B-G, R-G`).
pub fn inverse_rgb_decorr_bgr(out: &mut [u8]) {
    let mut i = 0;
    while i + 3 <= out.len() {
        let g = out[i + 1];
        out[i] = out[i].wrapping_add(g);
        out[i + 2] = out[i + 2].wrapping_add(g);
        i += 3;
    }
}

/// Inverse RGB decorrelation on a 4-byte BGRA-packed buffer, in
/// place. Alpha is left untouched by the decorrelation transform
/// (spec/03 §2.4 Validator-corrected note: alpha shares slot-3's
/// codebook but does NOT receive the decorrelation transform).
pub fn inverse_rgb_decorr_bgra(out: &mut [u8]) {
    let mut i = 0;
    while i + 4 <= out.len() {
        let g = out[i + 1];
        out[i] = out[i].wrapping_add(g);
        out[i + 2] = out[i + 2].wrapping_add(g);
        i += 4;
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

/// Spec/02 §2 / spec/05 (planned): the codec engages its
/// field-stride=2 interlaced path when `biHeight > 288`. The
/// threshold is the i386 build's compiled-in constant `0x120` (= 288)
/// per spec/01 §1.3 disassembly note.
#[inline]
pub fn is_interlaced_height(height: u32) -> bool {
    height > 288
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
}
