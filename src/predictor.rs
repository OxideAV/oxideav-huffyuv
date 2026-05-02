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

/// GRADIENT / PLANE (Predict Gradient):
/// `s[x] = s[x-1] + s_top[x] - s_top[x-1] + r[x]` for the second and
/// later rows. The decoder achieves this by first running LEFT over
/// the row to recover a 1-D prefix sum, then adding `top[x] -
/// top[x-1]` for each x (trace doc §4.2).
///
/// `top` is the previous decoded row (already in sample space).
pub fn pred_gradient_inplace(row: &mut [u8], top: &[u8], mut left: u8) -> u8 {
    debug_assert_eq!(row.len(), top.len());
    // First run LEFT to convert per-sample residuals into prefix sums.
    pred_left_inplace(row, left);
    // Then add top[x] - top[x-1].
    let mut top_left = 0u8; // top[-1] is taken as 0 (the encoder does the same).
    for x in 0..row.len() {
        let t = top[x];
        row[x] = row[x].wrapping_add(t).wrapping_sub(top_left);
        top_left = t;
    }
    // The `left` register threads through the row's predictions; for
    // Predict Gradient the upstream codec treats it as the value of
    // the last reconstructed sample (so the next row's PLANE pass
    // continues coherently). Mirror that.
    left = *row.last().unwrap_or(&left);
    left
}

/// MEDIAN (Paeth predictor, trace doc §4.3): emit
/// `s[x] = median(L, T, L+T-TL) + r[x]` per sample.
///
/// Returns the trailing `left` register.
pub fn pred_median_inplace(row: &mut [u8], top: &[u8], mut left: u8, mut top_left: u8) -> u8 {
    debug_assert_eq!(row.len(), top.len());
    for x in 0..row.len() {
        let l = left as i32;
        let t = top[x] as i32;
        let tl = top_left as i32;
        let pred = paeth_median(l, t, l + t - tl) as u8;
        let s = pred.wrapping_add(row[x]);
        row[x] = s;
        top_left = top[x];
        left = s;
    }
    left
}

fn paeth_median(a: i32, b: i32, c: i32) -> i32 {
    // Order-statistic median of three, identical to the classic
    // `median(L, T, L+T-TL)` formulation. The `c` value is allowed to
    // be negative or > 255 here; that is fine since we only need its
    // sort order vs. `a` and `b`.
    let mn = a.min(b);
    let mx = a.max(b);
    c.clamp(mn, mx)
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

    #[test]
    fn gradient_matches_hand_calc() {
        // top: 1, 2, 3, 4
        // residuals: 0, 0, 0, 0 → after LEFT: 0,0,0,0; then +top[x]-top[x-1]
        // → +1, +(2-1)=+1, +(3-2)=+1, +(4-3)=+1 → 1,1,1,1
        let top = [1u8, 2, 3, 4];
        let mut row = [0u8; 4];
        let left = pred_gradient_inplace(&mut row, &top, 0);
        assert_eq!(row, [1, 1, 1, 1]);
        assert_eq!(left, 1);
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
}
