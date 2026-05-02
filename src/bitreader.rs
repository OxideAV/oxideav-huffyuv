//! Tiny MSB-first bit reader used by the HuffYUV decoder.
//!
//! HuffYUV's bit reader walks bytes most-significant-bit first. The
//! on-disk packet is byte-swapped within every 32-bit word; the
//! [`unswap_payload`] helper undoes the swap before bit reading begins.

use oxideav_core::{Error, Result};

/// MSB-first bit reader.
pub struct BitReader<'a> {
    src: &'a [u8],
    /// Byte cursor into `src` for the next byte to be loaded.
    byte_pos: usize,
    /// Bits currently buffered in the high bits of `bit_buf`.
    bit_count: u8,
    /// Bit buffer; new bits enter at the bottom and are read out from
    /// position `bit_count - 1` downwards.
    bit_buf: u64,
}

impl<'a> BitReader<'a> {
    pub fn new(src: &'a [u8]) -> Self {
        Self {
            src,
            byte_pos: 0,
            bit_count: 0,
            bit_buf: 0,
        }
    }

    /// Number of bytes consumed so far (rounded down — does not include
    /// any partially-consumed byte still in the buffer).
    pub fn byte_pos(&self) -> usize {
        // The byte at byte_pos has not yet been pulled into the buffer.
        // Bits already pulled but not yet read live in `bit_buf`.
        self.byte_pos
            .saturating_sub((self.bit_count as usize).div_ceil(8))
    }

    fn refill(&mut self) -> Result<()> {
        while self.bit_count <= 56 && self.byte_pos < self.src.len() {
            self.bit_buf = (self.bit_buf << 8) | (self.src[self.byte_pos] as u64);
            self.byte_pos += 1;
            self.bit_count += 8;
        }
        Ok(())
    }

    /// Read one bit; 0 or 1.
    pub fn read_bit(&mut self) -> Result<u32> {
        if self.bit_count == 0 {
            self.refill()?;
            if self.bit_count == 0 {
                return Err(Error::Eof);
            }
        }
        self.bit_count -= 1;
        Ok(((self.bit_buf >> self.bit_count) & 1) as u32)
    }

    /// Read `n` bits (n ≤ 32). Returns `Error::Eof` if fewer than `n`
    /// bits remain.
    pub fn read_bits(&mut self, n: u8) -> Result<u32> {
        debug_assert!(n <= 32);
        if n == 0 {
            return Ok(0);
        }
        if self.bit_count < n {
            self.refill()?;
            if self.bit_count < n {
                return Err(Error::Eof);
            }
        }
        self.bit_count -= n;
        let mask = if n == 32 { u32::MAX } else { (1u32 << n) - 1 };
        Ok(((self.bit_buf >> self.bit_count) as u32) & mask)
    }
}

/// Undo the encoder's per-32-bit-word byte-swap so a bit reader walking
/// the result sees the logical bit order (trace doc §1.2).
///
/// The output length equals the input rounded **down** to a multiple of
/// 4 bytes; trailing partial words are discarded — they only ever hold
/// padding zeros (trace doc §5.6).
pub fn unswap_payload(src: &[u8]) -> Vec<u8> {
    let words = src.len() / 4;
    let mut out = Vec::with_capacity(words * 4);
    for w in 0..words {
        let off = w * 4;
        // 32-bit word byte-swap: AB CD EF GH → DC BA HG FE? Actually a
        // swap of an LE-stored u32 viewed as BE: bytes [b0,b1,b2,b3]
        // become [b3,b2,b1,b0]. The trace doc describes
        // "32-bit byte-swapped within every 32-bit word" — a u32 read as
        // little-endian on disk corresponds to a u32 written as
        // big-endian by the bit writer, so unswapping is a simple
        // 4-byte reverse per word.
        out.push(src[off + 3]);
        out.push(src[off + 2]);
        out.push(src[off + 1]);
        out.push(src[off]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_bits_msb_first() {
        let buf = [0b1010_1100u8, 0b1111_0000];
        let mut r = BitReader::new(&buf);
        assert_eq!(r.read_bit().unwrap(), 1);
        assert_eq!(r.read_bit().unwrap(), 0);
        assert_eq!(r.read_bit().unwrap(), 1);
        assert_eq!(r.read_bit().unwrap(), 0);
        assert_eq!(r.read_bits(4).unwrap(), 0b1100);
        assert_eq!(r.read_bits(4).unwrap(), 0b1111);
    }

    #[test]
    fn unswap_round_trip() {
        let src = [0x01u8, 0x02, 0x03, 0x04, 0xAA, 0xBB, 0xCC, 0xDD];
        let unswapped = unswap_payload(&src);
        assert_eq!(
            unswapped,
            vec![0x04, 0x03, 0x02, 0x01, 0xDD, 0xCC, 0xBB, 0xAA]
        );
    }
}
