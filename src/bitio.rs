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
/// words. Each call to [`Self::read_window`] returns a 32-bit window
/// whose top bit is the next unread bit of the stream.
pub struct BitReader<'a> {
    data: &'a [u8],
    /// Total number of bits ALREADY consumed.
    cursor_bits: u64,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            cursor_bits: 0,
        }
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
    pub fn peek_window(&self) -> u32 {
        let word_idx = (self.cursor_bits / 32) as usize;
        let bit_off = (self.cursor_bits % 32) as u32;
        let cur = self.read_word_le(word_idx);
        if bit_off == 0 {
            cur
        } else {
            // Need the high (32 - bit_off) bits of `cur` as the top of
            // the window, plus the high `bit_off` bits of the next
            // word as the low bits of the window.
            let nxt = self.read_word_le(word_idx + 1);
            // The next-unread bit sits at bit position
            // `(31 - bit_off)` of `cur`. Shift left by `bit_off` to
            // put it at position 31, then OR in the top `bit_off`
            // bits of `nxt`.
            (cur << bit_off) | (nxt >> (32 - bit_off))
        }
    }

    /// Advance the cursor by `n_bits` (n in 1..=32).
    pub fn consume_bits(&mut self, n_bits: u32) -> Result<()> {
        if n_bits == 0 || n_bits > 32 {
            return Err(Error::invalid(format!(
                "BitReader: consume_bits({n_bits}) out of 1..=32"
            )));
        }
        self.cursor_bits = self.cursor_bits.wrapping_add(n_bits as u64);
        Ok(())
    }

    fn read_word_le(&self, word_idx: usize) -> u32 {
        let base = word_idx * 4;
        let mut buf = [0u8; 4];
        for (i, slot) in buf.iter_mut().enumerate() {
            if let Some(&b) = self.data.get(base + i) {
                *slot = b;
            }
        }
        u32::from_le_bytes(buf)
    }
}

/// MSB-first bit writer that flushes to a `Vec<u8>` as 32-bit LE
/// words. spec/02 §4: each codeword's bits flow into the stream in
/// reading order; flushed words are LE-stored.
pub struct BitWriter {
    out: Vec<u8>,
    /// 32-bit accumulator; bits placed top-down (newest at bit
    /// position `(31 - bits_buffered)`).
    acc: u32,
    /// Number of bits currently held in `acc` (0..32).
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
    pub fn write_msb(&mut self, code: u32, n_bits: u32) {
        debug_assert!((1..=32).contains(&n_bits));
        let mut bits_remaining = n_bits;
        let mut payload = code;
        while bits_remaining > 0 {
            let space = 32 - self.bits_buffered;
            let take = bits_remaining.min(space);
            // The next `take` bits to write are the top `take` of
            // `payload`. Shift them into the accumulator at position
            // `(31 - bits_buffered)..(31 - bits_buffered - take + 1)`.
            let chunk = if take == 32 {
                payload
            } else {
                payload >> (32 - take)
            };
            // Mask out any low bits below `take` then shift into
            // position.
            let shift = 32 - self.bits_buffered - take;
            self.acc |= chunk << shift;
            self.bits_buffered += take;
            // Consume `take` bits from `payload`.
            if take == 32 {
                payload = 0;
            } else {
                payload <<= take;
            }
            bits_remaining -= take;
            if self.bits_buffered == 32 {
                self.flush_word();
            }
        }
    }

    fn flush_word(&mut self) {
        let bytes = self.acc.to_le_bytes();
        self.out.extend_from_slice(&bytes);
        self.acc = 0;
        self.bits_buffered = 0;
    }

    /// Finalise the stream — flush any partial word as a 4-byte LE
    /// store, with the buffered bits left-aligned at the high end and
    /// trailing bits unspecified (we emit zero, which is what spec/02
    /// §4 permits).
    pub fn finish(mut self) -> Vec<u8> {
        if self.bits_buffered > 0 {
            self.flush_word();
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
