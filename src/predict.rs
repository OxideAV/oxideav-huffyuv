//! Per-channel predictors and decorrelation transforms.
//!
//! Implements spec/03 §2 (left / gradient / median; per-channel
//! 8-bit modular wrap, no saturation) and §2.4 (RGB R-G/G/B-G
//! decorrelation, fused with the predictor at the residual level).
//!
//! The decoder calls these to *reconstruct* pixel values from
//! residuals; the encoder calls them to *produce* residuals from
//! pixels. Both directions are byte-modular.

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
pub fn inverse_gradient_post(out: &mut [u8], row_bytes: usize, height: usize) {
    if height < 2 || row_bytes == 0 {
        return;
    }
    // Walk row N from row 1 onward, adding the byte at the same column
    // of row N-1 (mod 256).
    for row in 1..height {
        let above = (row - 1) * row_bytes;
        let curr = row * row_bytes;
        for col in 0..row_bytes {
            out[curr + col] = out[curr + col].wrapping_add(out[above + col]);
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
}
