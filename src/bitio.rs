//! Bit-stream framing for HuffYUV's per-frame compressed body.
//!
//! spec/02 §4 fixes the convention:
//!
//! - 32-bit little-endian words.
//! - Each codeword's MSB sits earliest in stream order; the first bit
//!   of the first codeword sits at bit 31 of the first 32-bit word.
//! - Codewords abut without alignment.
//! - Final partial word is left-aligned at the top; trailing bits are
//!   unspecified.
//!
//! Both the reader (decode side) and the writer (encode side) live
//! here. The writer is used by the round-1 self-test encoder under
//! `#[cfg(test)] mod encoder` (see `encoder.rs` in this crate).

use crate::error::{Error, Result};

/// MSB-first bit reader over a `&[u8]` source consumed as 32-bit LE
/// words. Each call to [`Self::peek_window`] returns a 32-bit window
/// whose top bit is the next unread bit of the stream.
///
/// # Refill model (round 310)
///
/// The pre-r310 reader recomputed the 32-bit window from scratch on
/// every [`Self::peek_window`] call: it indexed back into `data`,
/// reconstructed up to two 32-bit LE words byte-by-byte through a
/// bounds-checked `[0u8; 4]` loop, and stitched them with a shift.
/// On the per-symbol decode critical path (one `peek_window` +
/// `consume_bits` per Huffman codeword — four per YUY2 macropixel)
/// that's two word reconstructions × four bounds-checked byte loads
/// for every symbol, which the BENCHMARKS.md hotspot table identified
/// as the flat ~100–120 MiB/s decode ceiling across all six
/// predictors (the rate is bit-read-bound, not predictor-bound).
///
/// Round 310 keeps a 64-bit MSB-aligned accumulator `acc` holding the
/// already-loaded leading bits of the stream. `peek_window` serves the
/// top 32 bits straight from `acc` with no byte access; `consume_bits`
/// only advances a bit counter and tops the accumulator back up to at
/// least 32 valid bits by pulling whole 32-bit LE words from `data`.
/// Each source word is read exactly once over the life of the field
/// decode (amortised one bounds-checked word fetch per ~32 consumed
/// bits, vs. the prior two-words-per-symbol), and the
/// `from_le_bytes(<[u8; 4]>::try_from(...))` fast path lets the
/// compiler emit a single aligned/unaligned 32-bit load when the slice
/// is in range.
///
/// The accumulator is MSB-aligned: the next-unread bit sits at bit 63,
/// so `(acc >> 32) as u32` is exactly the window the pre-r310
/// reconstruction produced (top bit = next-unread bit, zero-padded
/// past end-of-stream). `valid_bits` tracks how many of `acc`'s top
/// bits came from `data` (the rest are zero pad), so the public
/// surface — [`Self::cursor_bits`], [`Self::bytes_consumed`], the
/// zero-padded over-read window — is byte-for-byte identical to the
/// pre-r310 behaviour.
pub struct BitReader<'a> {
    data: &'a [u8],
    /// Total number of bits ALREADY consumed.
    cursor_bits: u64,
    /// MSB-aligned bit accumulator. The next-unread bit sits at bit
    /// 63; bits below `64 - valid_bits` are zero pad past end-of-data.
    acc: u64,
    /// Number of accumulator bits (from bit 63 downward) backed by
    /// `data` (not zero pad). Always `>= 32` after [`Self::refill`]
    /// unless the stream is fully exhausted.
    valid_bits: u32,
    /// Byte index of the next 32-bit LE word to pull from `data`,
    /// always a multiple of 4.
    byte_pos: usize,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        let mut r = Self {
            data,
            cursor_bits: 0,
            acc: 0,
            valid_bits: 0,
            byte_pos: 0,
        };
        r.refill();
        r
    }

    /// Bit-cursor position (bits consumed since construction).
    pub fn cursor_bits(&self) -> u64 {
        self.cursor_bits
    }

    /// Number of source bytes the bit cursor has crossed, rounded up
    /// to the next 4-byte word boundary. Used by the interlaced
    /// decoder to advance from the top field's bit-stream tail to the
    /// bottom field's uncompressed seed (spec/02 §4: each field's bit
    /// stream is flushed in 32-bit LE words; the next field starts on
    /// the byte after the final flushed word).
    pub fn bytes_consumed(&self) -> usize {
        let words = self.cursor_bits.div_ceil(32) as usize;
        words * 4
    }

    /// Peek the next 32 bits as a u32 with the next-unread bit at bit
    /// position 31 (MSB). Returns a zero-padded window when the
    /// underlying stream is exhausted.
    #[inline]
    pub fn peek_window(&self) -> u32 {
        (self.acc >> 32) as u32
    }

    /// Advance the cursor by `n_bits` (n in 1..=32).
    #[inline]
    pub fn consume_bits(&mut self, n_bits: u32) -> Result<()> {
        if n_bits == 0 || n_bits > 32 {
            return Err(Error::invalid(format!(
                "BitReader: consume_bits({n_bits}) out of 1..=32"
            )));
        }
        self.consume_bits_trusted(n_bits);
        Ok(())
    }

    /// Round-419 hot-path variant of [`Self::consume_bits`] for callers
    /// whose `n_bits` provably lies in `1..=32` already (Huffman code
    /// lengths come out of a built [`crate::tables::HuffTable`], whose
    /// build rejects lengths outside `1..=31`, and the primary-LUT fast
    /// path caps them at 16). Skips the per-symbol range check +
    /// `Result` plumbing on the decode critical path; the range
    /// contract is kept as a `debug_assert`.
    #[doc(hidden)]
    #[inline]
    pub fn consume_bits_trusted(&mut self, n_bits: u32) {
        debug_assert!((1..=32).contains(&n_bits));
        self.cursor_bits = self.cursor_bits.wrapping_add(n_bits as u64);
        // Drop `n_bits` from the top of the accumulator (which sits at
        // bit 63) and shrink the valid-bit count, then top it back up.
        self.acc <<= n_bits;
        self.valid_bits = self.valid_bits.saturating_sub(n_bits);
        if self.valid_bits < 32 {
            self.refill();
        }
    }

    /// Pull whole 32-bit LE words from `data` into the low end of the
    /// accumulator until `valid_bits >= 32` (the window peek always has
    /// 32 backed-or-zero-pad bits available) or `data` is exhausted.
    /// Words past end-of-data contribute nothing (the accumulator's
    /// low bits stay zero), reproducing the pre-r310 zero-padded
    /// over-read window exactly.
    #[inline]
    fn refill(&mut self) {
        while self.valid_bits <= 32 {
            if self.byte_pos >= self.data.len() {
                break;
            }
            let word = self.read_word_le(self.byte_pos);
            self.byte_pos += 4;
            // Place the freshly-read word's MSB at the first free
            // accumulator slot below the currently-valid region, i.e.
            // at bit position `63 - valid_bits`. Shifting a u64 word
            // up by `32 - valid_bits` lands its top bit there.
            self.acc |= (word as u64) << (32 - self.valid_bits);
            self.valid_bits += 32;
        }
    }

    #[inline]
    fn read_word_le(&self, base: usize) -> u32 {
        // Fast path: a full 4-byte window in range becomes one 32-bit
        // load. The tail path zero-pads the missing high-address bytes,
        // matching spec/02 §4's "final partial word, trailing bits
        // unspecified (we treat as zero)".
        if let Some(slice) = self.data.get(base..base + 4) {
            u32::from_le_bytes(<[u8; 4]>::try_from(slice).unwrap())
        } else {
            let mut buf = [0u8; 4];
            for (i, slot) in buf.iter_mut().enumerate() {
                if let Some(&b) = self.data.get(base + i) {
                    *slot = b;
                }
            }
            u32::from_le_bytes(buf)
        }
    }
}

/// MSB-first bit writer that flushes to a `Vec<u8>` as 32-bit LE
/// words. spec/02 §4: each codeword's bits flow into the stream in
/// reading order; flushed words are LE-stored.
///
/// # Accumulator model (round 419)
///
/// The pre-r419 writer kept a 32-bit accumulator and split each code
/// across a data-dependent 1–2-iteration loop (chunk carve, two
/// conditional shifts, an in-loop flush test) — per-codeword work on
/// the encoder's emit hot path (one `write_msb` per residual byte).
///
/// Round 419 widens the accumulator to 64 bits with the buffered bits
/// occupying the TOP `bits_buffered` positions. A write becomes
/// branch-free bit placement — the MSB-aligned 32-bit code is placed
/// at bit `63 - bits_buffered` with one shift+or (`bits_buffered < 32`
/// always holds on entry, and `n_bits ≤ 32`, so nothing ever falls off
/// the low end) — followed by a single "≥ 32 buffered → flush top LE
/// word" test. The emitted byte stream is identical: whole 32-bit LE
/// words in the same order, partial-word tail zero-padded, exactly as
/// the 32-bit writer produced (regression-guarded by the existing
/// writer/reader round-trip tests and the round-419 golden pins).
pub struct BitWriter {
    out: Vec<u8>,
    /// 64-bit accumulator; buffered bits sit in the TOP
    /// `bits_buffered` positions (newest at bit `63 - bits_buffered`),
    /// all lower bits zero.
    acc: u64,
    /// Number of bits currently held in `acc` (0..32 outside
    /// `write_msb`; a flush inside `write_msb` restores the invariant
    /// before returning).
    bits_buffered: u32,
}

impl BitWriter {
    pub fn new() -> Self {
        Self {
            out: Vec::new(),
            acc: 0,
            bits_buffered: 0,
        }
    }

    /// Write the top `n_bits` of `code` (an MSB-aligned 32-bit value).
    /// `n_bits` is in 1..=32.
    #[inline]
    pub fn write_msb(&mut self, code: u32, n_bits: u32) {
        debug_assert!((1..=32).contains(&n_bits));
        debug_assert!(self.bits_buffered < 32);
        // MSB-alignment contract: bits below `32 - n_bits` must be
        // zero (they are ORed into the accumulator's zero region).
        // Every in-crate producer satisfies this (`HuffEntry::code`
        // and the v1.x builder both left-align with a zero tail).
        debug_assert!(n_bits == 32 || code & ((1u32 << (32 - n_bits)) - 1) == 0);
        // `code` is MSB-aligned with zeros below bit `32 - n_bits`, so
        // the whole 32-bit value can be placed as-is: its significant
        // bits land at `63 - bits_buffered` downward and the zero tail
        // merges into the accumulator's zero region.
        self.acc |= ((code as u64) << 32) >> self.bits_buffered;
        self.bits_buffered += n_bits;
        if self.bits_buffered >= 32 {
            let word = (self.acc >> 32) as u32;
            self.out.extend_from_slice(&word.to_le_bytes());
            self.acc <<= 32;
            self.bits_buffered -= 32;
        }
    }

    /// Finalise the stream — flush any partial word as a 4-byte LE
    /// store, with the buffered bits left-aligned at the high end and
    /// trailing bits unspecified (we emit zero, which is what spec/02
    /// §4 permits).
    pub fn finish(mut self) -> Vec<u8> {
        if self.bits_buffered > 0 {
            let word = (self.acc >> 32) as u32;
            self.out.extend_from_slice(&word.to_le_bytes());
            self.acc = 0;
            self.bits_buffered = 0;
        }
        self.out
    }
}

impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writer_then_reader_round_trip_short_codes() {
        let mut w = BitWriter::new();
        // Write three 4-bit MSB-aligned codes: 0b1010, 0b0011, 0b1111.
        w.write_msb(0b1010 << 28, 4);
        w.write_msb(0b0011 << 28, 4);
        w.write_msb(0b1111 << 28, 4);
        let buf = w.finish();
        // Read back with peek_window + consume_bits.
        let mut r = BitReader::new(&buf);
        assert_eq!(r.peek_window() >> 28, 0b1010);
        r.consume_bits(4).unwrap();
        assert_eq!(r.peek_window() >> 28, 0b0011);
        r.consume_bits(4).unwrap();
        assert_eq!(r.peek_window() >> 28, 0b1111);
        r.consume_bits(4).unwrap();
    }

    /// Inlined copy of the pre-r310 from-scratch window reconstruction
    /// (recompute the 32-bit window directly from `data` at an absolute
    /// bit cursor). The round-310 refilling reader must produce a
    /// byte-identical window at every reachable cursor.
    fn pre_r310_peek(data: &[u8], cursor_bits: u64) -> u32 {
        let read_word_le = |word_idx: usize| -> u32 {
            let base = word_idx * 4;
            let mut buf = [0u8; 4];
            for (i, slot) in buf.iter_mut().enumerate() {
                if let Some(&b) = data.get(base + i) {
                    *slot = b;
                }
            }
            u32::from_le_bytes(buf)
        };
        let word_idx = (cursor_bits / 32) as usize;
        let bit_off = (cursor_bits % 32) as u32;
        let cur = read_word_le(word_idx);
        if bit_off == 0 {
            cur
        } else {
            let nxt = read_word_le(word_idx + 1);
            (cur << bit_off) | (nxt >> (32 - bit_off))
        }
    }

    #[test]
    fn round310_refill_window_matches_pre_r310_uniform_step() {
        // Pseudo-random-but-deterministic byte stream; walk it with a
        // fixed-stride consume and assert the window matches the
        // from-scratch reconstruction at every step, across stream
        // lengths that bracket the word boundary (including non-multiple-
        // of-4 lengths so the EOF zero-pad tail is exercised).
        for &len in &[0usize, 1, 3, 4, 5, 7, 8, 9, 16, 17, 31, 64, 257] {
            let mut data = vec![0u8; len];
            let mut s: u32 = 0x1357_9bdf ^ (len as u32).wrapping_mul(2654435761);
            for b in data.iter_mut() {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                *b = (s >> 24) as u8;
            }
            for &step in &[1u32, 3, 7, 11, 13, 16, 17, 23, 31, 32] {
                let mut r = BitReader::new(&data);
                let mut cursor: u64 = 0;
                // Walk past the end of the backing data to exercise the
                // zero-padded over-read window the decoder relies on for
                // a truncated/partial final word.
                let total_bits = (len as u64) * 8 + 96;
                while cursor < total_bits {
                    assert_eq!(
                        r.peek_window(),
                        pre_r310_peek(&data, cursor),
                        "len={len} step={step} cursor={cursor}"
                    );
                    assert_eq!(r.cursor_bits(), cursor);
                    r.consume_bits(step).unwrap();
                    cursor += step as u64;
                }
            }
        }
    }

    #[test]
    fn round310_refill_window_matches_pre_r310_variable_step() {
        // Variable consume lengths (1..=32) driven by the data itself —
        // mimics the real decode loop where each consume is a Huffman
        // code length in 1..=31. Asserts the window equivalence holds
        // for non-uniform advances that land the cursor at every
        // residue mod 32.
        let mut data = vec![0u8; 512];
        let mut s: u32 = 0x2468_ace0;
        for b in data.iter_mut() {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            *b = (s >> 24) as u8;
        }
        let mut r = BitReader::new(&data);
        let mut cursor: u64 = 0;
        let mut t: u32 = 0x9e37_79b9;
        for _ in 0..4000 {
            assert_eq!(r.peek_window(), pre_r310_peek(&data, cursor));
            assert_eq!(r.cursor_bits(), cursor);
            t = t.wrapping_mul(1664525).wrapping_add(1013904223);
            let step = 1 + (t >> 27) % 32; // 1..=32
            r.consume_bits(step).unwrap();
            cursor += step as u64;
        }
    }

    #[test]
    fn round310_consume_bits_rejects_out_of_range() {
        let data = [0xAAu8; 8];
        let mut r = BitReader::new(&data);
        assert!(r.consume_bits(0).is_err());
        assert!(r.consume_bits(33).is_err());
        // A rejected consume must not perturb the cursor or window.
        assert_eq!(r.cursor_bits(), 0);
        assert_eq!(r.peek_window(), pre_r310_peek(&data, 0));
    }

    #[test]
    fn round310_bytes_consumed_rounds_up_to_word() {
        // bytes_consumed() must stay the word-rounded cursor regardless
        // of the new refill bookkeeping (the interlaced decoder relies
        // on it to find the next field's seed).
        let data = [0u8; 32];
        let mut r = BitReader::new(&data);
        assert_eq!(r.bytes_consumed(), 0);
        r.consume_bits(1).unwrap();
        assert_eq!(r.bytes_consumed(), 4);
        r.consume_bits(31).unwrap(); // cursor now 32 bits
        assert_eq!(r.bytes_consumed(), 4);
        r.consume_bits(1).unwrap(); // cursor now 33 bits
        assert_eq!(r.bytes_consumed(), 8);
    }

    #[test]
    fn round310_empty_stream_serves_zero_window() {
        let r = BitReader::new(&[]);
        assert_eq!(r.peek_window(), 0);
        assert_eq!(r.cursor_bits(), 0);
        assert_eq!(r.bytes_consumed(), 0);
    }

    #[test]
    fn writer_long_codes_cross_word_boundary() {
        let mut w = BitWriter::new();
        // A 17-bit code followed by another 17-bit code — guaranteed
        // to cross the 32-bit word boundary.
        let code1 = 0b1_0101_0101_0101_0101_u32 << (32 - 17);
        let code2 = 0b1_1001_1001_1001_1001_u32 << (32 - 17);
        w.write_msb(code1, 17);
        w.write_msb(code2, 17);
        let buf = w.finish();
        assert_eq!(buf.len() % 4, 0);
        let mut r = BitReader::new(&buf);
        assert_eq!(
            (r.peek_window() >> 15) & ((1u32 << 17) - 1),
            0b1_0101_0101_0101_0101
        );
        r.consume_bits(17).unwrap();
        assert_eq!(
            (r.peek_window() >> 15) & ((1u32 << 17) - 1),
            0b1_1001_1001_1001_1001
        );
    }
}
