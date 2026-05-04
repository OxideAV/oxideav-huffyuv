//! RLE encoder for Huffman code-length tables (mirror of
//! [`crate::rle::decode_lengths`]).
//!
//! Format (trace doc §3):
//!
//! ```text
//! emit lengths run-length-encoded:
//!   byte = (val & 0x1F) | (run << 5)         when run ≤ 7
//!   byte = (val & 0x1F), then ext = run      when run > 7  (1..255)
//! ```

/// Encode a length array as the byte-stream the decoder expects.
pub fn encode_lengths(lengths: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(lengths.len());
    let mut i = 0usize;
    while i < lengths.len() {
        let v = lengths[i] & 0x1F;
        let mut run: usize = 1;
        while i + run < lengths.len() && (lengths[i + run] & 0x1F) == v && run < 255 {
            run += 1;
        }
        if run < 8 {
            out.push(v | ((run as u8) << 5));
        } else {
            // Long-run extension: high-3-bits zero indicates the next
            // byte carries the full 1..255 repeat count.
            out.push(v);
            out.push(run as u8);
        }
        i += run;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rle;

    #[test]
    fn round_trip_through_decoder() {
        let lens: Vec<u8> = (0..256).map(|i| (i % 16) as u8).collect();
        let blob = encode_lengths(&lens);
        let (decoded, used) = rle::decode_lengths(&blob, 256).unwrap();
        assert_eq!(used, blob.len());
        assert_eq!(decoded, lens);
    }

    #[test]
    fn long_runs() {
        let lens = vec![8u8; 256];
        let blob = encode_lengths(&lens);
        let (decoded, used) = rle::decode_lengths(&blob, 256).unwrap();
        assert_eq!(decoded, lens);
        assert_eq!(used, blob.len());
        // 256 = 255 + 1: one long-run (val=8, run=0, ext=255) + a short
        // run-1 (val=8, run=1) → 3 bytes total.
        assert_eq!(blob.len(), 3);
    }
}
