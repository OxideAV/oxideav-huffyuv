//! Canonical-Huffman code-length builder driven by a symbol-frequency
//! histogram.
//!
//! The HuffYUV / FFVHuff encoder (trace doc §3.3 + §9.8) does **not**
//! ship pre-baked codes on the wire — it runs a regular Huffman build
//! from a fixed parabolic prior (when `-context 0`) or from the
//! per-frame symbol histogram (when `-context 1`). Both sides then
//! reconstruct the codes from the lengths via the canonical
//! convention in [`crate::huffman::HuffTable::build_codes`].
//!
//! This module provides the missing piece for the encoder: turn a
//! frequency array into a length array that satisfies
//! `Σ 2^(-len_i) = 1` and respects an optional `max_length` cap.
//!
//! Algorithm: standard min-heap Huffman (priority queue of
//! `(count, node_id)` pairs, merge two cheapest nodes), followed by
//! a length-limit balancer that promotes/demotes single bits when any
//! length would exceed `max_length`. The balancer is the simple
//! "BRCI" / "balanced canonical Huffman" trick: walk lengths from
//! deepest down, lift each over-deep symbol up to `max_length`, then
//! restore the Kraft-McMillan equality by deepening shallower symbols
//! one bit at a time. This is **not** length-optimal under the cap (a
//! true package-merge would be) but matches what the upstream codec
//! does — its `ff_huff_gen_len_table` uses the same algorithm.

use oxideav_core::{Error, Result};

/// Maximum Huffman length the codec is willing to emit. Trace doc §9.8
/// reports an absolute cap of 16 bits for ≤14-bit alphabets and 28
/// bits for the 16384-symbol case. We use 32 as the framework upper
/// bound (matches `crate::huffman::MAX_CODE_LEN`).
pub const ABSOLUTE_MAX_LEN: u8 = 31;

/// Build a canonical-Huffman length table from a per-symbol frequency
/// histogram. `freqs.len()` must be the alphabet size.
///
/// `max_length` caps the deepest emitted code length; passing
/// `ABSOLUTE_MAX_LEN` disables the cap effectively.
///
/// Symbols with `freqs[i] == 0` get length 0 by default — except when
/// the alphabet would otherwise contain only one or zero present
/// symbols, in which case the builder force-includes the lowest-index
/// missing symbol (and assigns it length 1) so the decoder always sees
/// a complete prefix code.
pub fn build_lengths(freqs: &[u64], max_length: u8) -> Result<Vec<u8>> {
    let n = freqs.len();
    if n == 0 {
        return Err(Error::invalid("huffyuv length-builder: empty alphabet"));
    }
    if max_length == 0 {
        return Err(Error::invalid(
            "huffyuv length-builder: max_length must be ≥ 1",
        ));
    }
    let cap = max_length.min(ABSOLUTE_MAX_LEN);

    let present: Vec<u32> = (0..n).filter(|&i| freqs[i] > 0).map(|i| i as u32).collect();

    // Force at least two symbols so the canonical-Huffman code is
    // complete (canonical builder requires Σ 2^(-len) = 1, which
    // collapses if there's only one symbol). We pick the two
    // lowest-index symbols regardless of their frequency.
    let mut working: Vec<u32> = present.clone();
    if working.len() < 2 {
        let mut i = 0u32;
        while working.len() < 2 {
            let v = i;
            if !working.contains(&v) {
                working.push(v);
            }
            i += 1;
            if (i as usize) >= n.max(2) {
                break;
            }
        }
        if working.len() < 2 {
            return Err(Error::invalid(
                "huffyuv length-builder: alphabet has fewer than 2 symbols (cannot build code)",
            ));
        }
    }

    // Run the standard Huffman tree build. Use a binary min-heap on
    // (count, tie_breaker, node_id). Internal nodes get ids ≥ n.
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    // Per-node parent + which side (so we can compute path-lengths).
    // Index 0..n: leaf nodes (one per alphabet slot, regardless of
    // whether they're "present" — non-present nodes simply never get
    // pushed onto the heap, so their parent stays sentinel).
    let mut parent: Vec<i32> = vec![-1; n];
    let mut tie: u64 = 0;
    let mut heap: BinaryHeap<Reverse<(u64, u64, i32)>> = BinaryHeap::new();
    for &sym in working.iter() {
        let c = freqs[sym as usize].max(1);
        heap.push(Reverse((c, tie, sym as i32)));
        tie += 1;
    }
    let mut next_internal: i32 = n as i32;
    while heap.len() >= 2 {
        let Reverse((c1, _, a)) = heap.pop().unwrap();
        let Reverse((c2, _, b)) = heap.pop().unwrap();
        let parent_id = next_internal;
        next_internal += 1;
        // Grow parent to cover the new internal node.
        parent.push(-1);
        // a, b are existing nodes (leaf or internal); record their parent.
        parent[a as usize] = parent_id;
        parent[b as usize] = parent_id;
        heap.push(Reverse((c1 + c2, tie, parent_id)));
        tie += 1;
    }

    // Compute leaf depths by walking up.
    let mut lengths: Vec<u8> = vec![0; n];
    for &sym in working.iter() {
        let mut d: u32 = 0;
        let mut cur = sym as i32;
        while parent[cur as usize] >= 0 {
            cur = parent[cur as usize];
            d += 1;
        }
        lengths[sym as usize] = d.min(255) as u8;
    }

    // If no symbol got length 0..=cap (all uncapped depths fit), we're
    // already done.
    if lengths.iter().all(|&l| l <= cap) {
        return validated(lengths, cap);
    }

    // Length-limit balancing: bring every length down to `cap` by
    // promoting deepest leaves first, then restore Kraft equality by
    // deepening currently-shallow leaves one bit at a time. Operate on
    // the union of present + force-added symbols.
    enforce_max_length(&mut lengths, cap);
    validated(lengths, cap)
}

fn enforce_max_length(lengths: &mut [u8], cap: u8) {
    let active_idxs: Vec<usize> = (0..lengths.len()).filter(|&i| lengths[i] > 0).collect();
    if active_idxs.is_empty() {
        return;
    }
    // Truncate everything above `cap`.
    for &i in active_idxs.iter() {
        if lengths[i] > cap {
            lengths[i] = cap;
        }
    }
    // Compute Kraft sum measured in 2^-cap units.
    let kraft = |l: &[u8], cap: u8| -> u128 {
        let mut s: u128 = 0;
        for &i in active_idxs.iter() {
            let li = l[i];
            if li > 0 {
                s += 1u128 << (cap - li);
            }
        }
        s
    };
    let target: u128 = 1u128 << cap;
    while kraft(lengths, cap) > target {
        // Over-budget: deepen the currently-shallowest leaf by one bit
        // (which subtracts 2^(cap-old) - 2^(cap-old-1) = 2^(cap-old-1)).
        // Pick the shallowest leaf at length `< cap` so we can deepen it.
        let mut best: Option<usize> = None;
        let mut best_len: u8 = 0;
        for &i in active_idxs.iter() {
            let l = lengths[i];
            if l > 0 && l < cap && (best.is_none() || l < best_len) {
                best = Some(i);
                best_len = l;
            }
        }
        match best {
            Some(i) => lengths[i] += 1,
            None => break, // can't make progress; leave as-is
        }
    }
    if kraft(lengths, cap) < target {
        // Under-budget: this branch is rare (truncation usually
        // over-shoots). To gain budget we'd shorten the deepest leaf,
        // but doing so would re-introduce length collisions; instead
        // we leave the slightly-incomplete code in place — the
        // downstream `validated()` will return an error so the caller
        // can request a less aggressive cap.
    }
}

fn validated(lengths: Vec<u8>, cap: u8) -> Result<Vec<u8>> {
    // Sanity check: must satisfy Kraft-McMillan equality and stay within cap.
    let mut total: u128 = 0;
    for &l in lengths.iter() {
        if l == 0 {
            continue;
        }
        if l > cap {
            return Err(Error::invalid(format!(
                "huffyuv length-builder: produced length {l} exceeds cap {cap}"
            )));
        }
        total += 1u128 << (32 - l as u32);
    }
    let need: u128 = 1u128 << 32;
    if total != need {
        return Err(Error::invalid(format!(
            "huffyuv length-builder: lengths fail Kraft-McMillan ({total} != 2^32)"
        )));
    }
    Ok(lengths)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::huffman::HuffTable;

    #[test]
    fn flat_distribution_yields_balanced_lengths() {
        let n = 4usize;
        let freqs = vec![10u64; n];
        let lens = build_lengths(&freqs, 16).unwrap();
        assert_eq!(lens, vec![2u8; 4]);
    }

    #[test]
    fn skewed_distribution() {
        // 5 symbols, one dominates.
        let freqs = vec![100u64, 10, 1, 1, 1];
        let lens = build_lengths(&freqs, 16).unwrap();
        // Symbol 0 should be the shortest. Assert we satisfy Kraft.
        let codes = HuffTable::build_codes(&lens).unwrap();
        assert!(lens[0] <= lens[1]);
        assert!(lens[1] <= lens[2]);
        // Round-trip through the decoder builder.
        let _t = HuffTable::from_lengths(&lens).unwrap();
        assert!(codes[0].1 >= 1);
    }

    #[test]
    fn single_symbol_alphabet_is_promoted_to_two() {
        let mut freqs = vec![0u64; 256];
        freqs[42] = 9999;
        let lens = build_lengths(&freqs, 16).unwrap();
        // At least two symbols must have non-zero length to satisfy
        // Kraft equality (a single 1-symbol "code" cannot be encoded
        // by a prefix-free code of length 0).
        let nonzero = lens.iter().filter(|&&l| l > 0).count();
        assert!(nonzero >= 2, "got nonzero={nonzero} lens={:?}", &lens[..5]);
        let _t = HuffTable::from_lengths(&lens).unwrap();
    }

    #[test]
    fn length_cap_respected_for_skewed_alphabet() {
        // Heavily skewed so the unconstrained Huffman would generate a
        // very deep code.
        let mut freqs = vec![1u64; 64];
        freqs[0] = 1u64 << 60;
        let lens = build_lengths(&freqs, 12).unwrap();
        for (i, &l) in lens.iter().enumerate() {
            assert!(l <= 12, "symbol {i}: length {l} exceeds cap 12");
        }
    }
}
