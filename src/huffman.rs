//! Canonical-Huffman code-table builder + decoder.
//!
//! HuffYUV ships only the per-symbol code lengths on the wire; codes are
//! reconstructed by both sides via the canonical convention — sort
//! symbols by `(length, symbol)`, then assign codes in that order
//! starting from 0 and shifting one bit left every time the length
//! grows. A length of 0 means "symbol not present in this table".
//!
//! The code-bit ordering matches the bit-reader's MSB-first read order:
//! the first decoded bit is the high bit of the canonical code.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;

/// Maximum supported code length. The trace doc reports the deepest
/// captured Huffman code is 28 bits (16384-alphabet gray16 case); we
/// support up to 32 bits to cover any reasonable encoder output without
/// risking a 64-bit overflow during code construction.
pub const MAX_CODE_LEN: u8 = 32;

/// A canonical-Huffman decoder.
///
/// Implemented as a sparse linear-search table keyed by length, then a
/// sorted code value. This is fast enough for our decode rates (the per-
/// frame inner loop dwarfs the table overhead) and sidesteps the
/// memory cost of a 16384-entry direct-lookup table for the deepest
/// alphabets. A future optimisation is the upstream "joint VLC" trick
/// (trace doc §4.5) which is purely a speed win and bit-equivalent.
#[derive(Clone, Debug)]
pub struct HuffTable {
    /// `entries[len-1]` is the sorted-by-code list of `(code, symbol)` for
    /// symbols whose code length is `len`. Empty for unused lengths.
    entries: Vec<Vec<(u32, u32)>>,
    /// `min_code[len-1]` is the smallest canonical code at length `len`,
    /// or `u32::MAX` if no symbol has that length.
    min_code: Vec<u32>,
}

impl HuffTable {
    /// Build from per-symbol code lengths.
    ///
    /// Returns an error if `lengths` is empty, contains a length larger
    /// than [`MAX_CODE_LEN`], or fails the Kraft–McMillan equality
    /// `Σ 2^(-len_i) = 1` (lengths must form a complete prefix code).
    /// HuffYUV's canonical-Huffman builder always emits complete codes,
    /// so an incomplete length set indicates a corrupt or non-conformant
    /// stream.
    pub fn from_lengths(lengths: &[u8]) -> Result<Self> {
        if lengths.is_empty() {
            return Err(Error::invalid("huffyuv huffman: empty length table"));
        }
        // Bound check.
        for (sym, &len) in lengths.iter().enumerate() {
            if len > MAX_CODE_LEN {
                return Err(Error::invalid(format!(
                    "huffyuv huffman: code length {len} for symbol {sym} exceeds {MAX_CODE_LEN}"
                )));
            }
        }
        // Bucket symbols by length.
        let mut by_len: Vec<Vec<u32>> = vec![Vec::new(); MAX_CODE_LEN as usize + 1];
        for (sym, &len) in lengths.iter().enumerate() {
            if len > 0 {
                by_len[len as usize].push(sym as u32);
            }
        }
        // Kraft-McMillan check (skip when there is exactly one symbol at
        // length 0 — i.e. a 1-symbol alphabet — which conventionally
        // assigns code 0 with length 1; HuffYUV never produces that
        // shape, so we just enforce the standard equality).
        let mut total: u64 = 0;
        for len in 1..=MAX_CODE_LEN as usize {
            total += (by_len[len].len() as u64) << (MAX_CODE_LEN as usize - len);
        }
        if total != 1u64 << MAX_CODE_LEN as usize {
            return Err(Error::invalid(format!(
                "huffyuv huffman: lengths fail Kraft-McMillan ({} != 2^{MAX_CODE_LEN})",
                total
            )));
        }
        // Assign canonical codes using the "longest-first" convention
        // observed in the FFmpeg HuffYUV / FFVHuff encoder. Lay the
        // length-MAX codes out contiguously starting at 0, then derive
        // shorter lengths by truncating to their high bits. A simple
        // way to compute it: keep a "next-free" counter measured in
        // length-MAX_used units (where MAX_used is the deepest length
        // actually present), and walk lengths from longest to shortest.
        // For each symbol at length L the code is the top L bits of the
        // next-free counter; we then advance by `1 << (MAX_used - L)`.
        //
        // (Equivalent to: code at length L starts at
        // `cumulative_count[L+..MAX] >> (MAX - L)`, then symbols within
        // a length advance by 1 in length-L units.)
        let mut max_used: usize = 0;
        for len in 1..=MAX_CODE_LEN as usize {
            if !by_len[len].is_empty() {
                max_used = len;
            }
        }
        let mut entries: Vec<Vec<(u32, u32)>> = vec![Vec::new(); MAX_CODE_LEN as usize];
        let mut min_code = vec![u32::MAX; MAX_CODE_LEN as usize];
        if max_used == 0 {
            return Ok(Self { entries, min_code });
        }
        let mut next: u64 = 0;
        for len in (1..=max_used).rev() {
            let step: u64 = 1 << (max_used - len);
            for (i, &sym) in by_len[len].iter().enumerate() {
                let code_at_max = next;
                let code_l = (code_at_max >> (max_used - len)) as u32;
                entries[len - 1].push((code_l, sym));
                if i == 0 {
                    min_code[len - 1] = code_l;
                }
                next += step;
            }
        }
        Ok(Self { entries, min_code })
    }

    /// Decode the next symbol from `r`. Returns `Error::Eof` if the
    /// reader runs out of bits mid-code.
    pub fn read_symbol(&self, r: &mut BitReader<'_>) -> Result<u32> {
        let mut code: u32 = 0;
        for len in 1..=MAX_CODE_LEN as usize {
            let bit = r.read_bit()?;
            code = (code << 1) | bit;
            let bucket = &self.entries[len - 1];
            if bucket.is_empty() {
                continue;
            }
            // Symbols at this length live in `code` ∈
            // [min_code[len-1], min_code[len-1] + bucket.len()).
            let base = self.min_code[len - 1];
            if code >= base && (code - base) < bucket.len() as u32 {
                return Ok(bucket[(code - base) as usize].1);
            }
        }
        Err(Error::invalid(
            "huffyuv huffman: code longer than MAX_CODE_LEN",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Standard "go go gophers" canonical Huffman example: 4 symbols at
    /// lengths 2,2,2,2 → codes 00,01,10,11.
    #[test]
    fn four_equal_lengths() {
        let mut lens = vec![0u8; 4];
        lens.iter_mut().for_each(|l| *l = 2);
        let t = HuffTable::from_lengths(&lens).unwrap();
        // Build a buffer carrying codes 00,01,10,11 → bytes 0b00_01_10_11 = 0x1B.
        // BitReader is MSB-first.
        let buf = [0x1Bu8];
        let mut r = BitReader::new(&buf);
        assert_eq!(t.read_symbol(&mut r).unwrap(), 0);
        assert_eq!(t.read_symbol(&mut r).unwrap(), 1);
        assert_eq!(t.read_symbol(&mut r).unwrap(), 2);
        assert_eq!(t.read_symbol(&mut r).unwrap(), 3);
    }

    /// Kraft-McMillan rejection when the lengths don't sum to 1.
    #[test]
    fn rejects_incomplete() {
        // 3 symbols at length 1 each — 3 * 2^-1 = 1.5 > 1.
        let lens = [1u8, 1, 1];
        let err = HuffTable::from_lengths(&lens).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("Kraft"), "got: {msg}");
    }

    /// Mixed lengths with the classic "1, 2, 3, 3" shape. Under the
    /// FFmpeg HuffYUV "longest-first" canonical convention, the codes
    /// derived from these lengths are: symbol 0 → `1`, symbol 1 → `01`,
    /// symbol 2 → `000`, symbol 3 → `001`. We verify by writing a
    /// stream of codes and reading them back.
    #[test]
    fn mixed_lengths() {
        let lens = [1u8, 2, 3, 3];
        let t = HuffTable::from_lengths(&lens).unwrap();
        // Stream (sym 0, 1, 2, 3) = 1 01 000 001 = 9 bits.
        // bits: 1,0,1,0,0,0,0,0,1
        // first byte (MSB-first) = 1010_0000 = 0xA0
        // second byte top bit = 1 → 1000_0000 = 0x80
        let buf = [0xA0u8, 0x80];
        let mut r = BitReader::new(&buf);
        assert_eq!(t.read_symbol(&mut r).unwrap(), 0);
        assert_eq!(t.read_symbol(&mut r).unwrap(), 1);
        assert_eq!(t.read_symbol(&mut r).unwrap(), 2);
        assert_eq!(t.read_symbol(&mut r).unwrap(), 3);
    }
}
