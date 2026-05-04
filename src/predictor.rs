//! Spatial-predictor primitives (trace doc §4).
//!
//! The decoder walks an entire row of residuals first (Huffman-decoded
//! into raw `u8` values), then folds the predictor over the row to
//! recover sample values. All arithmetic is modular `u8`/`u16` (samples
//! wrap naturally inside their bit-depth).

/// LEFT (Predict Left): `s[x] = s[x-1] + r[x]` across the row, with the
/// initial `left` taken from the trailing register of the previous row
/// (or a per-format prelude for the very first row). Returns the `left`
/// register value at the end of the row.
pub fn pred_left_inplace(row: &mut [u8], mut left: u8) -> u8 {
    for v in row.iter_mut() {
        let next = left.wrapping_add(*v);
        *v = next;
        left = next;
    }
    left
}

/// GRADIENT / PLANE (Predict Gradient).
///
/// Canonical 2-D first-difference: `pred[x] = left + top[x] − top_left`
/// where `left` is the running register threaded across rows
/// (`samples[x-1, y]`) and `top_left` is `top[x-1]` (with `top[-1]`
/// taken as the **last** `top` value of the previous row's processing
/// — equivalently the previous row's last `top` sample). For the very
/// first GRADIENT row the caller passes the appropriate `top_left`
/// (typically `0`).
///
/// The decoded sample is `pred[x] + residual[x]`. After each step
/// `left` is updated to the just-decoded sample and `top_left` to
/// `top[x]`. Returns `(left, top_left_at_row_end)` so the caller can
/// thread both across rows.
pub fn pred_gradient_inplace_full(
    row: &mut [u8],
    top: &[u8],
    mut left: u8,
    mut top_left: u8,
) -> (u8, u8) {
    debug_assert_eq!(row.len(), top.len());
    for x in 0..row.len() {
        let pred = left.wrapping_add(top[x]).wrapping_sub(top_left);
        let s = pred.wrapping_add(row[x]);
        row[x] = s;
        top_left = top[x];
        left = s;
    }
    (left, top_left)
}

/// Convenience wrapper for callers that don't need to thread
/// `top_left` (e.g. the first GRADIENT row, or the predictor unit
/// tests). Equivalent to `pred_gradient_inplace_full(row, top, left, 0)`
/// returning only the new `left` register.
pub fn pred_gradient_inplace(row: &mut [u8], top: &[u8], left: u8) -> u8 {
    pred_gradient_inplace_full(row, top, left, 0).0
}

/// MEDIAN (Paeth predictor, trace doc §4.3): emit
/// `s[x] = median(L, T, L+T-TL) + r[x]` per sample.
///
/// All arithmetic is **modular** in the sample width — the
/// `L+T-TL` term wraps around mod 256 (or mod 2^bps for HBD)
/// before the sort-by-value happens. This matches the
/// upstream encoder, which computes the predictor in u8 space.
///
/// Returns the trailing `left` register. The `top_left` register is
/// threaded across calls via [`pred_median_inplace_full`] when the
/// caller needs to chain rows; this convenience overload accepts the
/// initial `top_left` and discards the trailing one.
pub fn pred_median_inplace(row: &mut [u8], top: &[u8], left: u8, top_left: u8) -> u8 {
    pred_median_inplace_full(row, top, left, top_left).0
}

/// Same as [`pred_median_inplace`] but returns both the trailing
/// `left` and the trailing `top_left` register so the caller can
/// thread the `top_left` across rows (the upstream encoder maintains
/// a single `top_left` per plane that walks through every MEDIAN call
/// for the whole frame, not just within one row).
pub fn pred_median_inplace_full(
    row: &mut [u8],
    top: &[u8],
    mut left: u8,
    mut top_left: u8,
) -> (u8, u8) {
    debug_assert_eq!(row.len(), top.len());
    for x in 0..row.len() {
        let pred = paeth_median_u8(left, top[x], top_left);
        let s = pred.wrapping_add(row[x]);
        row[x] = s;
        top_left = top[x];
        left = s;
    }
    (left, top_left)
}

/// Order-statistic median of three samples, computed in **modular
/// u8 space**: the `L+T-TL` value wraps around 256 before the sort
/// happens. Used by both the decoder (predict) and the encoder
/// (residual derivation).
pub fn paeth_median_u8(l: u8, t: u8, tl: u8) -> u8 {
    let c = l.wrapping_add(t).wrapping_sub(tl);
    // Sort (l, t, c) by their unsigned u8 value and return the middle.
    // 3-element sorted median:
    //   median = max(min(a,b), min(max(a,b), c))   (classic formula)
    let lo = l.min(t);
    let hi = l.max(t);
    hi.min(c.max(lo))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn left_prefix_sum_round_trip() {
        let mut row = [10u8, 0, 5, 0, 250]; // residuals
        let left = pred_left_inplace(&mut row, 0);
        assert_eq!(row, [10, 10, 15, 15, 9]); // 250 wraps: 15 + 250 = 265 mod 256 = 9
        assert_eq!(left, 9);
    }

    /// Zero-residuals + all-zero top → decoder produces a constant row
    /// equal to the prev_left register (each pred[x] = left + 0 - 0 =
    /// left, then sample = left + 0 = left, etc.).
    #[test]
    fn gradient_zero_input_returns_register() {
        let top = [0u8; 4];
        let mut row = [0u8; 4];
        let left = pred_gradient_inplace(&mut row, &top, 7);
        assert_eq!(row, [7, 7, 7, 7]);
        assert_eq!(left, 7);
    }

    /// Encoder ↔ decoder symmetry: residuals derived from
    /// `r[x] = sample[x] - pred[x]` (where pred[x] = left + top[x] -
    /// top_left, with the same register threading the decoder uses)
    /// should round-trip through `pred_gradient_inplace`.
    #[test]
    fn gradient_symmetric_round_trip() {
        let top: [u8; 6] = [10, 20, 30, 40, 50, 60];
        let original: [u8; 6] = [12, 25, 33, 41, 49, 58];
        let prev_left: u8 = 5;

        let mut residuals = [0u8; 6];
        let mut left = prev_left;
        let mut top_left = 0u8;
        for x in 0..6 {
            let pred = left.wrapping_add(top[x]).wrapping_sub(top_left);
            residuals[x] = original[x].wrapping_sub(pred);
            top_left = top[x];
            left = original[x];
        }

        let mut row = residuals;
        let _ = pred_gradient_inplace(&mut row, &top, prev_left);
        assert_eq!(row, original);
    }

    #[test]
    fn median_matches_paeth() {
        // top: [10, 20, 30]
        // start left=0, top_left=0 → samples 0+r[0], pick median(0,10,10)=10
        // residuals = [5, 0, 0] → s[0] = 10 + 5 = 15
        //   next: median(15, 20, 15+20-10=25)=20 → s[1] = 20
        //   next: median(20, 30, 20+30-20=30)=30 → s[2] = 30
        let top = [10u8, 20, 30];
        let mut row = [5u8, 0, 0];
        let left = pred_median_inplace(&mut row, &top, 0, 0);
        assert_eq!(row, [15, 20, 30]);
        assert_eq!(left, 30);
    }

    /// Modular wrap test: `L + T − TL` overflows u8 → must be
    /// reduced mod 256 BEFORE the median sort happens (otherwise the
    /// predictor returns the wrong value at boundaries).
    #[test]
    fn median_wraps_in_u8_space() {
        // L=210, T=106, TL=41 → c (mod 256) = 19. Sort {210, 106, 19} → median = 106.
        // The naive i32 implementation would clamp(106, 210)=210, which
        // doesn't match the upstream encoder.
        assert_eq!(paeth_median_u8(210, 106, 41), 106);
    }
}
