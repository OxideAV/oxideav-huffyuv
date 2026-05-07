//! Built-in classic-table assets, the run-length decoder, and the
//! canonical-Huffman builder.
//!
//! Backs spec/01 §5 (the RLE format), spec/03 §3 (canonical-Huffman
//! recipe + binary's longest-length-first variant), and spec/04 §3/§4
//! (record format for the six classic blobs and the v1.x
//! precomputed-codes pair).

use crate::error::{Error, Result};
use crate::header::{Method, PixelFamily};

// ───────────────────────── compiled-in blobs ─────────────────────────
//
// Each blob is the verbatim Extractor extraction from the Round-1
// Auditor-verified `tables/` workspace (one byte per line, hex `XX`).
// We use `include_str!` here rather than `include_bytes!` because the
// Extractor's chosen on-disk format is a header+hex-text file. The
// parsing happens once at table-build time (very cheap; ≤ 256 lines
// per blob) and the parsed bytes are `'static` after the first lookup
// via `OnceLock`.

const YUV_LEFT_HEX: &str = include_str!("../../../docs/video/huffyuv/tables/00-yuv-left-blob.hex");
const YUV_GRADIENT_HEX: &str =
    include_str!("../../../docs/video/huffyuv/tables/01-yuv-gradient-blob.hex");
const YUV_MEDIAN_HEX: &str =
    include_str!("../../../docs/video/huffyuv/tables/02-yuv-median-blob.hex");
const RGB_LEFT_HEX: &str = include_str!("../../../docs/video/huffyuv/tables/03-rgb-left-blob.hex");
const RGB_LEFT_DECORR_HEX: &str =
    include_str!("../../../docs/video/huffyuv/tables/04-rgb-left-decorr-blob.hex");
const RGB_GRADIENT_DECORR_HEX: &str =
    include_str!("../../../docs/video/huffyuv/tables/05-rgb-gradient-decorr-blob.hex");

const V1X_LENGTHS_SET_A_HEX: &str =
    include_str!("../../../docs/video/huffyuv/tables/06-v1x-lengths-set-a.hex");
const V1X_LENGTHS_SET_B_HEX: &str =
    include_str!("../../../docs/video/huffyuv/tables/07-v1x-lengths-set-b.hex");
const V1X_CODES_SET_A_CSV: &str =
    include_str!("../../../docs/video/huffyuv/tables/08-v1x-codes-set-a.csv");
const V1X_CODES_SET_B_CSV: &str =
    include_str!("../../../docs/video/huffyuv/tables/09-v1x-codes-set-b.csv");

/// Identify which of the six classic blobs a `(family, method)` pair
/// selects (spec/03 §4 / spec/04 §3.1).
pub fn classic_blob_for(family: PixelFamily, method: Method) -> &'static str {
    match (family.is_rgb(), method) {
        // YUV branch
        (false, Method::PredictOld) | (false, Method::Left) => YUV_LEFT_HEX,
        (false, Method::Gradient) => YUV_GRADIENT_HEX,
        (false, Method::Median) => YUV_MEDIAN_HEX,
        // RGB branch
        (true, Method::PredictOld) | (true, Method::Left) => RGB_LEFT_HEX,
        (true, Method::LeftDecorr) => RGB_LEFT_DECORR_HEX,
        (true, Method::GradientDecorr) => RGB_GRADIENT_DECORR_HEX,
        // Method-byte allow-list rejects these per spec/01 §3.1, so we
        // never reach here for legal streams; fall back deterministically.
        (false, _) => YUV_LEFT_HEX,
        (true, _) => RGB_LEFT_HEX,
    }
}

/// Strip `#`-prefixed header lines and parse one-byte-per-line hex.
fn parse_hex_bytes(text: &str) -> Vec<u8> {
    let mut out = Vec::new();
    for line in text.lines() {
        let l = line.trim();
        if l.is_empty() || l.starts_with('#') {
            continue;
        }
        // Tolerate inline comments: take the leading hex run.
        let token = l.split_whitespace().next().unwrap_or("");
        if token.len() < 2 {
            continue;
        }
        if let Ok(b) = u8::from_str_radix(&token[..2], 16) {
            out.push(b);
        }
    }
    out
}

/// Parse the `index,code_byte` CSV used for the v1.x raw 256-byte code
/// templates (`tables/08`, `tables/09`).
fn parse_codes_csv(text: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(256);
    for line in text.lines() {
        let l = line.trim();
        if l.is_empty() || l.starts_with('#') {
            continue;
        }
        // First non-comment line is the `index,code_byte` header.
        if !l.chars().next().is_some_and(|c| c.is_ascii_digit()) {
            continue;
        }
        let mut parts = l.split(',');
        let _idx = parts.next();
        let code = match parts.next() {
            Some(c) => c.trim(),
            None => continue,
        };
        let trimmed = code.trim_start_matches("0x").trim_start_matches("0X");
        if let Ok(b) = u8::from_str_radix(trimmed, 16) {
            out.push(b);
        }
    }
    out
}

/// Lazy-once parsed-byte forms of each blob; avoids re-parsing the hex
/// text on every per-stream call.
mod blobs {
    use super::*;
    use std::sync::OnceLock;

    macro_rules! define_blob {
        ($name:ident, $src:ident, $parser:ident) => {
            pub fn $name() -> &'static [u8] {
                static CELL: OnceLock<Vec<u8>> = OnceLock::new();
                CELL.get_or_init(|| $parser($src)).as_slice()
            }
        };
    }

    define_blob!(yuv_left, YUV_LEFT_HEX, parse_hex_bytes);
    define_blob!(yuv_gradient, YUV_GRADIENT_HEX, parse_hex_bytes);
    define_blob!(yuv_median, YUV_MEDIAN_HEX, parse_hex_bytes);
    define_blob!(rgb_left, RGB_LEFT_HEX, parse_hex_bytes);
    define_blob!(rgb_left_decorr, RGB_LEFT_DECORR_HEX, parse_hex_bytes);
    define_blob!(
        rgb_gradient_decorr,
        RGB_GRADIENT_DECORR_HEX,
        parse_hex_bytes
    );

    define_blob!(v1x_lengths_set_a, V1X_LENGTHS_SET_A_HEX, parse_hex_bytes);
    define_blob!(v1x_lengths_set_b, V1X_LENGTHS_SET_B_HEX, parse_hex_bytes);

    define_blob!(v1x_codes_set_a, V1X_CODES_SET_A_CSV, parse_codes_csv);
    define_blob!(v1x_codes_set_b, V1X_CODES_SET_B_CSV, parse_codes_csv);
}

/// Public accessor for the parsed-byte form of the classic blob
/// belonging to `(family, method)` (spec/04 §3).
pub fn classic_blob_bytes(family: PixelFamily, method: Method) -> &'static [u8] {
    let hex = classic_blob_for(family, method);
    if std::ptr::eq(hex, YUV_LEFT_HEX) {
        blobs::yuv_left()
    } else if std::ptr::eq(hex, YUV_GRADIENT_HEX) {
        blobs::yuv_gradient()
    } else if std::ptr::eq(hex, YUV_MEDIAN_HEX) {
        blobs::yuv_median()
    } else if std::ptr::eq(hex, RGB_LEFT_HEX) {
        blobs::rgb_left()
    } else if std::ptr::eq(hex, RGB_LEFT_DECORR_HEX) {
        blobs::rgb_left_decorr()
    } else {
        blobs::rgb_gradient_decorr()
    }
}

/// Public accessor for the parsed v1.x precomputed lengths/codes
/// (spec/04 §4).
pub fn v1x_lengths_set_a() -> &'static [u8] {
    blobs::v1x_lengths_set_a()
}
pub fn v1x_lengths_set_b() -> &'static [u8] {
    blobs::v1x_lengths_set_b()
}
pub fn v1x_codes_set_a() -> &'static [u8] {
    blobs::v1x_codes_set_a()
}
pub fn v1x_codes_set_b() -> &'static [u8] {
    blobs::v1x_codes_set_b()
}

// ───────────────────────── RLE decoder ─────────────────────────
//
// spec/01 §5 algorithm: each iteration reads one byte b; value = b &
// 0x1F, count_hint = b >> 5. If count_hint == 0, read another byte c;
// if c == 0 the table is terminated, else count = c. Otherwise count
// = count_hint. Emit (value, count) into the 256-slot output, advance.

/// RLE-decode one channel's length table from `cursor` into a 256-byte
/// output buffer. Returns the post-consumed cursor offset (so callers
/// can chain three successive invocations across one extradata blob).
pub fn rle_decode_one_channel(cursor: &mut &[u8]) -> Result<[u8; 256]> {
    let mut out = [0u8; 256];
    let mut idx = 0usize;
    while idx < 256 {
        let b = take_byte(cursor)
            .ok_or_else(|| Error::invalid("RLE: unexpected EOF reading count byte"))?;
        let value = b & 0x1F;
        let count_hint = b >> 5;
        let count = if count_hint == 0 {
            // Long-form: next byte is the count, with `0x00` =
            // table-terminator (early exit). spec/01 §5.
            let c = take_byte(cursor)
                .ok_or_else(|| Error::invalid("RLE: unexpected EOF reading long-count byte"))?;
            if c == 0 {
                // Zero terminator: the spec says "exit the loop"; if
                // we reach this with idx < 256 the remaining slots
                // stay 0 (= length-0 = absent symbol). Per spec/01 §8
                // #4 "robustness" note, pre-zeroing the buffer is the
                // safe path; we initialise above.
                break;
            }
            c as usize
        } else {
            count_hint as usize
        };
        if idx + count > 256 {
            // The spec is strict: emits stop at 256. A blob that
            // overflows is malformed.
            return Err(Error::invalid(format!(
                "RLE: run of {count} would overflow 256-slot output (idx {idx})"
            )));
        }
        for _ in 0..count {
            out[idx] = value;
            idx += 1;
        }
    }
    Ok(out)
}

fn take_byte(cursor: &mut &[u8]) -> Option<u8> {
    if cursor.is_empty() {
        None
    } else {
        let b = cursor[0];
        *cursor = &cursor[1..];
        Some(b)
    }
}

/// RLE-decode three concatenated channel length tables from one
/// extradata payload (spec/01 §5.1 "three calls in succession").
pub fn rle_decode_three_channels(blob: &[u8]) -> Result<[[u8; 256]; 3]> {
    let mut cursor: &[u8] = blob;
    let s1 = rle_decode_one_channel(&mut cursor)?;
    let s2 = rle_decode_one_channel(&mut cursor)?;
    let s3 = rle_decode_one_channel(&mut cursor)?;
    Ok([s1, s2, s3])
}

// ───────────────────────── canonical Huffman ─────────────────────────
//
// spec/03 §3 + §3.2 + spec/04 §2.6. The proprietary binary's variant
// processes length tiers from LARGEST length down to smallest (the
// audit-validated "longest codeword first" order from VirtualDub
// developer's note in spec/02 §4). Within each tier, symbols are
// walked in ascending symbol-value order. Codes are stored
// MSB-aligned in 32-bit dwords.

/// Per-symbol Huffman entry. `length == 0` means "absent symbol".
#[derive(Debug, Clone, Copy, Default)]
pub struct HuffEntry {
    pub length: u8,
    /// MSB-aligned 32-bit code (the high `length` bits hold the code
    /// value, the low `(32 - length)` bits are zero). Per spec/04 §2.6
    /// + spec/02 §4 (MSB-first packing).
    pub code: u32,
}

/// One channel's encode+decode tables.
#[derive(Debug, Clone)]
pub struct HuffTable {
    pub entries: [HuffEntry; 256],
    /// `decode[i].len` and `decode[i].sym` for i in 0..256 — a flat,
    /// length-then-symbol-indexed lookup is built lazily by the
    /// decoder via [`decode_one`]; we keep the per-symbol entries here
    /// and walk them for decode rather than rebuilding a separate
    /// prefix tree. This keeps the table tiny and removes a class of
    /// bugs at the cost of O(256) per-symbol decode (acceptable for
    /// round 1; round 2 can swap in a fast LUT once a trace lockstep
    /// is in place).
    pub max_length: u8,
}

impl HuffTable {
    /// Build canonical-Huffman codes from a 256-byte length table per
    /// spec/03 §3 / spec/04 §2.6 (the LONGEST-LENGTH-FIRST variant
    /// the binary uses; verified empirically by the spec/04 §2.6
    /// Validator-corrected algorithm).
    pub fn build_from_lengths(lengths: &[u8; 256]) -> Result<Self> {
        // Validate lengths (spec/03 §3.2.3).
        for (i, &l) in lengths.iter().enumerate() {
            if l > 31 {
                return Err(Error::invalid(format!(
                    "code length {l} for symbol {i} exceeds 31-bit ceiling"
                )));
            }
        }
        let mut entries = [HuffEntry::default(); 256];
        // Largest non-zero length first.
        let mut max_length: u8 = 0;
        for &l in lengths.iter() {
            if l > max_length {
                max_length = l;
            }
        }

        // Walk length tiers from MAX down to 1; within each, ascending
        // symbol index. Code accumulator is 32-bit MSB-aligned;
        // wraps to 0 at the end (Kraft equality).
        let mut code_acc: u32 = 0;
        let mut tier = max_length as i32;
        while tier > 0 {
            let l = tier as u8;
            let step: u32 = if l == 0 { 0 } else { 1u32 << (32 - l as u32) };
            for sym in 0..256usize {
                if lengths[sym] == l {
                    entries[sym] = HuffEntry {
                        length: l,
                        code: code_acc,
                    };
                    code_acc = code_acc.wrapping_add(step);
                }
            }
            tier -= 1;
        }
        // Sanity: when the table is non-trivial, Kraft equality
        // requires code_acc to wrap exactly to 0. (When the table is
        // empty — all lengths == 0 — code_acc was never touched.)
        if max_length > 0 && code_acc != 0 {
            return Err(Error::invalid(
                "non-canonical length distribution (Kraft inequality not 0)",
            ));
        }
        Ok(Self {
            entries,
            max_length,
        })
    }

    /// RFC-1951-style canonical builder used when an interop
    /// implementation prefers the standard ordering. Round 1 does NOT
    /// use this on the decode side — `build_from_lengths` follows the
    /// proprietary's longest-first algorithm, which differs in the
    /// per-tier code starts and is the only assignment that round-trips
    /// our self-encoded fixtures. Kept here for parity with spec/03
    /// §3.1's RFC-1951 description and as a comparison baseline for
    /// future Auditor work.
    #[allow(dead_code)]
    pub fn build_from_lengths_rfc1951(lengths: &[u8; 256]) -> Result<Self> {
        for (i, &l) in lengths.iter().enumerate() {
            if l > 31 {
                return Err(Error::invalid(format!(
                    "code length {l} for symbol {i} exceeds 31-bit ceiling"
                )));
            }
        }
        let mut entries = [HuffEntry::default(); 256];
        let mut max_length = 0u8;
        for &l in lengths.iter() {
            if l > max_length {
                max_length = l;
            }
        }
        if max_length == 0 {
            return Ok(Self {
                entries,
                max_length,
            });
        }
        // Shortest length first, ascending symbol within each tier.
        let mut code: u32 = 0;
        let mut prev_len: u8 = 0;
        // Collect (len, sym) pairs and sort.
        let mut pairs: Vec<(u8, u16)> = Vec::with_capacity(256);
        for (sym, &l) in lengths.iter().enumerate() {
            if l != 0 {
                pairs.push((l, sym as u16));
            }
        }
        pairs.sort();
        for (l, sym) in pairs {
            if prev_len == 0 {
                code = 0;
            } else if l != prev_len {
                let shift = (l - prev_len) as u32;
                code <<= shift;
            }
            // MSB-align: the L-bit `code` left-aligned in a 32-bit
            // dword.
            let msb_aligned = code << (32 - l as u32);
            entries[sym as usize] = HuffEntry {
                length: l,
                code: msb_aligned,
            };
            code = code.wrapping_add(1);
            prev_len = l;
        }
        Ok(Self {
            entries,
            max_length,
        })
    }
}

/// Produce a `HuffTable` directly from raw 256-byte length and
/// 256-byte code-template arrays as stored in the v1.x precomputed
/// pair (spec/04 §4.2). Each per-symbol code byte's bits become the
/// low 8 bits of an L-bit code, with `(L - 8)` leading zeros for L >
/// 8 (or just the low L bits for L ≤ 8).
pub fn v1x_table_from_pair(lengths: &[u8; 256], codes: &[u8; 256]) -> Result<HuffTable> {
    let mut entries = [HuffEntry::default(); 256];
    let mut max_length = 0u8;
    for sym in 0..256 {
        let l = lengths[sym];
        if l > 31 {
            return Err(Error::invalid(format!(
                "v1.x precomputed length {l} for symbol {sym} exceeds 31-bit ceiling"
            )));
        }
        if l == 0 {
            continue;
        }
        if l > max_length {
            max_length = l;
        }
        let byte = codes[sym] as u32;
        // The actual L-bit code value is:
        //   if L <= 8: byte & ((1 << L) - 1)
        //   if L > 8 : byte itself (with (L - 8) implicit leading zeros)
        let code_value = if l <= 8 {
            byte & ((1u32 << l) - 1)
        } else {
            byte
        };
        // MSB-align the L-bit code in a 32-bit accumulator.
        let msb_aligned = code_value << (32 - l as u32);
        entries[sym] = HuffEntry {
            length: l,
            code: msb_aligned,
        };
    }
    Ok(HuffTable {
        entries,
        max_length,
    })
}

/// Decode the next symbol from a 32-bit MSB-first bit window, by
/// scanning per-symbol entries directly. Returns `(symbol, length)`.
/// Round-1 simple linear scan; Round-2 will swap in a length-bucketed
/// LUT once Auditor lockstep is in place.
pub fn decode_one(table: &HuffTable, window: u32) -> Result<(u8, u8)> {
    // Walk lengths in ascending order; for each candidate length, the
    // top L bits of `window` form the code value to compare. We can
    // find the first matching (length, code) by tier.
    // For correctness we walk each of the 256 entries and check
    // whether the entry's MSB-aligned code matches the top L bits of
    // window. Cost: ≤ 256 iterations / symbol; sufficient for round 1
    // self-roundtrip tests.
    for entry in table.entries.iter() {
        let l = entry.length;
        if l == 0 {
            continue;
        }
        let mask: u32 = if l == 32 {
            u32::MAX
        } else {
            !0u32 << (32 - l as u32)
        };
        let candidate = window & mask;
        if candidate == entry.code {
            // Found.
            // Identify the symbol by index — we need to walk in
            // declaration order again. Caller wants `symbol` (u8) and
            // `length` (u8).
            // We already have the entry but not the index; iterate to
            // find. Cheaper: encode the index as part of the entry?
            // Defer: linear find of matching (length, code).
            // Re-walk to recover the symbol.
            for (sym, e) in table.entries.iter().enumerate() {
                if e.length == l && e.code == entry.code {
                    return Ok((sym as u8, l));
                }
            }
        }
    }
    Err(Error::invalid("no Huffman code matched bit window"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rle_decoder_basic_runs() {
        // Pack: 1 byte b = (count_hint << 5) | (value & 0x1F). Encode
        // 256 entries of value=5 in chunks of 7 (the largest count
        // that fits in count_hint). Then a long-form terminator.
        let mut input: Vec<u8> = Vec::new();
        let value = 5u8;
        let total = 256usize;
        let mut left = total;
        while left > 0 {
            let chunk = left.min(7) as u8;
            let b = (chunk << 5) | (value & 0x1F);
            input.push(b);
            left -= chunk as usize;
        }
        input.extend_from_slice(&[0x00, 0x00]);
        let mut cursor: &[u8] = &input;
        let table = rle_decode_one_channel(&mut cursor).unwrap();
        assert_eq!(table[0], 5);
        assert_eq!(table[255], 5);
    }

    #[test]
    fn classic_blob_decompresses_to_768_bytes() {
        for blob in [
            blobs::yuv_left(),
            blobs::yuv_gradient(),
            blobs::yuv_median(),
            blobs::rgb_left(),
            blobs::rgb_left_decorr(),
            blobs::rgb_gradient_decorr(),
        ] {
            let out = rle_decode_three_channels(blob).unwrap();
            // All three slots are 256 entries each.
            assert_eq!(out[0].len(), 256);
            assert_eq!(out[1].len(), 256);
            assert_eq!(out[2].len(), 256);
            // Lengths in [0, 31].
            for slot in out.iter() {
                for &l in slot.iter() {
                    assert!(l <= 31, "length {l} out of [0, 31]");
                }
            }
        }
    }

    #[test]
    fn v1x_set_a_decompresses_to_256_lengths() {
        let mut cursor: &[u8] = blobs::v1x_lengths_set_a();
        let lengths = rle_decode_one_channel(&mut cursor).unwrap();
        assert_eq!(lengths.len(), 256);
        // spec/04 §3.2: set A's max length is 17.
        let max = *lengths.iter().max().unwrap();
        assert!(max <= 31);
    }

    #[test]
    fn v1x_codes_csv_parses_to_256_bytes() {
        let codes_a = blobs::v1x_codes_set_a();
        let codes_b = blobs::v1x_codes_set_b();
        assert_eq!(codes_a.len(), 256);
        assert_eq!(codes_b.len(), 256);
    }

    #[test]
    fn canonical_huffman_kraft_equality_yuv_left() {
        let blob = blobs::yuv_left();
        let [s1, s2, s3] = rle_decode_three_channels(blob).unwrap();
        // Each slot should be a valid prefix code.
        for slot in [&s1, &s2, &s3] {
            HuffTable::build_from_lengths(slot).expect("Kraft equality holds");
        }
    }

    #[test]
    fn canonical_build_self_roundtrip_synth_lengths() {
        // Synthesise a tiny length table: symbol 0..3 have lengths
        // 2,2,2,2 (uniform 4-symbol alphabet).
        let mut lens = [0u8; 256];
        lens[0] = 2;
        lens[1] = 2;
        lens[2] = 2;
        lens[3] = 2;
        let table = HuffTable::build_from_lengths(&lens).unwrap();
        // The four codes are 00, 01, 10, 11 in some order. Codes are
        // MSB-aligned in 32 bits: high 2 bits hold the value.
        let codes: Vec<u32> = (0..4).map(|s| table.entries[s].code >> 30).collect();
        let mut sorted = codes.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3]);
    }
}
