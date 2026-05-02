//! RLE-coded code-length decoder.
//!
//! HuffYUV stores each Huffman table's symbol → code-length mapping as
//! a tight run-length encoding (trace doc §3):
//!
//! ```text
//! read byte b
//!   val    = b & 0x1F      ; low 5 bits = code length (0..31)
//!   repeat = b >> 5        ; high 3 bits = run length (0..7)
//!   if repeat == 0:
//!       repeat = next_byte ; long-run extension: 1..255
//!   emit val, repeat times
//! ```
//!
//! 5 bits suffice because the maximum supported code length is bounded
//! by the wider Huffman builder's `MAX_BITS = 32` ceiling; the deepest
//! captured length in the corpus was 28 bits (gray16 16384-alphabet).

use oxideav_core::{Error, Result};

/// Decode `n_symbols` code lengths from `src`, returning the per-symbol
/// length array and the number of bytes consumed.
///
/// The decoder reads at most one byte past the last needed symbol when
/// the final long-run consumes fewer symbols than its repeat count would
/// suggest; this is a fatal corruption error per the trace doc — every
/// well-formed table emits exactly `n_symbols` lengths.
pub fn decode_lengths(src: &[u8], n_symbols: usize) -> Result<(Vec<u8>, usize)> {
    let mut out = Vec::with_capacity(n_symbols);
    let mut pos = 0usize;
    while out.len() < n_symbols {
        if pos >= src.len() {
            return Err(Error::invalid(format!(
                "huffyuv RLE: truncated table after {} of {} symbols",
                out.len(),
                n_symbols
            )));
        }
        let b = src[pos];
        pos += 1;
        let val = b & 0x1F;
        let mut repeat = (b >> 5) as usize;
        if repeat == 0 {
            if pos >= src.len() {
                return Err(Error::invalid(
                    "huffyuv RLE: truncated long-run extension byte",
                ));
            }
            repeat = src[pos] as usize;
            pos += 1;
        }
        let remaining = n_symbols - out.len();
        if repeat > remaining {
            return Err(Error::invalid(format!(
                "huffyuv RLE: run {} overshoots remaining {} symbols",
                repeat, remaining
            )));
        }
        for _ in 0..repeat {
            out.push(val);
        }
    }
    Ok((out, pos))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_runs_and_zero_extension() {
        // 0x43 = (val=3, run=2) → 3,3
        // 0x25 = (val=5, run=1) → 5
        // 0x80 = (val=0, run=4) → 0,0,0,0
        // 0x07 = (val=7, run=0) → next byte is the long-run length: 0x03 → 7,7,7
        let src = [0x43u8, 0x25, 0x80, 0x07, 0x03];
        let (lens, used) = decode_lengths(&src, 10).unwrap();
        assert_eq!(used, src.len());
        assert_eq!(lens, vec![3, 3, 5, 0, 0, 0, 0, 7, 7, 7]);
    }

    #[test]
    fn rejects_overshoot() {
        // run=4 of length=2 with only 3 remaining symbols.
        let src = [0x82u8];
        let err = decode_lengths(&src, 3).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("overshoot"), "got: {msg}");
    }
}
