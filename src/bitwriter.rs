//! MSB-first bit writer used by the HuffYUV / FFVHuff encoder.
//!
//! Bits are appended starting from the highest unused bit in the
//! current pending byte; once 8 bits accumulate, the byte flushes into
//! the destination buffer. A trailing flush pads with zeros (the codec
//! finishes every packet with a 31-bit zero tail anyway, trace doc
//! §5.6, so the boundary padding is harmless).

/// MSB-first bit writer.
pub struct BitWriter {
    out: Vec<u8>,
    /// Pending byte (high bits already filled, low bits zero).
    pending: u8,
    /// Number of bits already filled into `pending` (0..7 inclusive).
    bits_in: u8,
}

impl BitWriter {
    pub fn new() -> Self {
        Self {
            out: Vec::new(),
            pending: 0,
            bits_in: 0,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            out: Vec::with_capacity(capacity),
            pending: 0,
            bits_in: 0,
        }
    }

    /// Write `n` low bits of `value` (n ≤ 32). Bits emitted MSB-first
    /// (i.e. bit `n-1` of `value` lands first on the wire).
    pub fn write_bits(&mut self, n: u8, value: u32) {
        debug_assert!(n <= 32);
        if n == 0 {
            return;
        }
        // Emit one bit at a time. The fast path could shift whole bytes
        // out of `value`, but the encoder runs at frame rates well
        // below the bit reader's bottleneck so we keep this simple and
        // obviously-correct.
        let mut i = n;
        while i > 0 {
            i -= 1;
            let bit = ((value >> i) & 1) as u8;
            self.pending = (self.pending << 1) | bit;
            self.bits_in += 1;
            if self.bits_in == 8 {
                self.out.push(self.pending);
                self.pending = 0;
                self.bits_in = 0;
            }
        }
    }

    /// Pad with zeros to the next byte boundary, then return the buffer.
    pub fn finish(mut self) -> Vec<u8> {
        if self.bits_in != 0 {
            self.pending <<= 8 - self.bits_in;
            self.out.push(self.pending);
        }
        self.out
    }

    /// Number of bits written so far (including any pending fractional
    /// byte).
    pub fn bit_count(&self) -> usize {
        self.out.len() * 8 + self.bits_in as usize
    }
}

impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// Encoder side of [`crate::bitreader::unswap_payload`]: rewrite each
/// 32-bit word in `src` with its bytes reversed. The output length
/// equals the input rounded down to a multiple of 4 bytes; trailing
/// partial words are dropped (the encoder always emits 32-bit-aligned
/// payloads — see §5.6 / §9.5 of the trace doc).
pub fn swap_payload(src: &[u8]) -> Vec<u8> {
    let words = src.len() / 4;
    let mut out = Vec::with_capacity(words * 4);
    for w in 0..words {
        let off = w * 4;
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
    use crate::bitreader::{unswap_payload, BitReader};

    #[test]
    fn writer_round_trips_through_reader() {
        let mut w = BitWriter::new();
        w.write_bits(1, 1);
        w.write_bits(3, 0b101);
        w.write_bits(8, 0xAA);
        w.write_bits(4, 0b1100);
        let buf = w.finish();
        let mut r = BitReader::new(&buf);
        assert_eq!(r.read_bits(1).unwrap(), 1);
        assert_eq!(r.read_bits(3).unwrap(), 0b101);
        assert_eq!(r.read_bits(8).unwrap(), 0xAA);
        assert_eq!(r.read_bits(4).unwrap(), 0b1100);
    }

    #[test]
    fn swap_unswap_round_trip() {
        let src = [0x01u8, 0x02, 0x03, 0x04, 0xAA, 0xBB, 0xCC, 0xDD];
        let swapped = swap_payload(&src);
        let back = unswap_payload(&swapped);
        assert_eq!(back, src);
    }
}
