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

const YUV_LEFT_HEX: &str = include_str!("tables_data/00-yuv-left-blob.hex");
const YUV_GRADIENT_HEX: &str = include_str!("tables_data/01-yuv-gradient-blob.hex");
const YUV_MEDIAN_HEX: &str = include_str!("tables_data/02-yuv-median-blob.hex");
const RGB_LEFT_HEX: &str = include_str!("tables_data/03-rgb-left-blob.hex");
const RGB_LEFT_DECORR_HEX: &str = include_str!("tables_data/04-rgb-left-decorr-blob.hex");
const RGB_GRADIENT_DECORR_HEX: &str = include_str!("tables_data/05-rgb-gradient-decorr-blob.hex");

const V1X_LENGTHS_SET_A_HEX: &str = include_str!("tables_data/06-v1x-lengths-set-a.hex");
const V1X_LENGTHS_SET_B_HEX: &str = include_str!("tables_data/07-v1x-lengths-set-b.hex");
const V1X_CODES_SET_A_CSV: &str = include_str!("tables_data/08-v1x-codes-set-a.csv");
const V1X_CODES_SET_B_CSV: &str = include_str!("tables_data/09-v1x-codes-set-b.csv");

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

// ───────────────────────── RLE encoder ─────────────────────────
//
// spec/01 §5 (encoder side). The decoder grammar is:
//   leading byte b: value = b & 0x1F, count_hint = b >> 5
//   if count_hint == 0: read second byte c; c == 0 ends the channel,
//                       otherwise c is the run count
//   else                count = count_hint (1..7)
//
// The decoder's outer loop terminates at `idx == 256` regardless of
// whether a terminator follows. The six classic blobs (151..217 wire
// bytes for 768 expanded length-table bytes) contain NO `0x00` bytes:
// each per-channel block fills exactly 256 slots and the next block
// starts immediately. The lstrcpyA terminator at the end of the whole
// 3-channel concat is just the C-string null and is not a per-channel
// in-band signal — see [`spec/01`](spec/01) §5 for the audit notes.
//
// So our encoder does NOT emit a `0x00 0x00` between channels. Each
// `rle_encode_one_channel` call emits exactly the bytes the decoder
// will consume to fill 256 slots, and nothing more.

/// RLE-encode one 256-byte length table to `out`. spec/01 §5: each
/// run uses the count-hint when count ≤ 7, otherwise the long-form
/// `value 0x_NN` (NN > 0) for runs of 8..255. No trailing `0x00 0x00`
/// is appended — see module-level note.
pub fn rle_encode_one_channel(lengths: &[u8; 256], out: &mut Vec<u8>) -> Result<()> {
    for &l in lengths.iter() {
        if l > 31 {
            return Err(Error::invalid(format!(
                "RLE encode: length {l} out of [0, 31]"
            )));
        }
    }
    // Run-length pass.
    let mut i = 0usize;
    while i < 256 {
        let v = lengths[i];
        let mut run = 1usize;
        while i + run < 256 && lengths[i + run] == v && run < 255 {
            run += 1;
        }
        // Emit (v, run).
        let mut remaining = run;
        while remaining > 0 {
            if remaining <= 7 {
                let chunk = remaining as u8;
                let b = (chunk << 5) | (v & 0x1F);
                // chunk ≥ 1 → count_hint ≥ 1 → `b` ≥ 0x20 regardless
                // of `v`. Safe (no in-band `0x00` produced).
                out.push(b);
                remaining = 0;
            } else {
                // Long-form: count_hint = 0 (leading byte = v),
                // followed by an explicit count in 1..=255.
                let chunk = remaining.min(255) as u8;
                let leading = v & 0x1F;
                out.push(leading);
                out.push(chunk);
                remaining -= chunk as usize;
            }
        }
        i += run;
    }
    Ok(())
}

/// RLE-encode three per-channel length tables into one extradata
/// `tables` blob (the bytes that live at +0x2C in the BIH).
pub fn rle_encode_three_channels(lengths: &[[u8; 256]; 3]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    for slot in lengths.iter() {
        rle_encode_one_channel(slot, &mut out)?;
    }
    Ok(out)
}

// ───────── Length-limited canonical-Huffman length computation ─────────
//
// Round-2 v2.x custom-tables encoder path: given a per-channel byte
// histogram, compute the length table that minimises the expected
// codeword length subject to `max_length ≤ 31` (spec/03 §3.2.3 #2).
//
// We use the "package-merge" algorithm (Larmore-Hirschberg) in the
// plain length-limited Huffman form. It's the textbook recipe; the
// algorithm itself is mathematics, not codec-specific. For 256-symbol
// alphabets it runs in O(256 * 31) = O(8192) work, which is trivial.
//
// Spec invariants we honour:
// - lengths in 0..=31 (spec/03 §3.2.3 #2)
// - length 0 ⇔ symbol absent (zero count)
// - Kraft equality (sum 2^-L_i == 1) when at least one symbol is
//   present.
// - For a single-symbol histogram we still emit length 1 (single-bit
//   code) so that decode windows have at least one bit to consume.

/// Build a per-symbol length table from a 256-entry histogram. Symbols
/// with a zero count get length 0; everything else gets a length in
/// 1..=31.
pub fn compute_canonical_lengths(histogram: &[u32; 256]) -> Result<[u8; 256]> {
    // Collect (sym, freq) for present symbols.
    let mut active: Vec<(u16, u32)> = Vec::new();
    for (sym, &f) in histogram.iter().enumerate() {
        if f > 0 {
            active.push((sym as u16, f));
        }
    }
    let mut lengths = [0u8; 256];
    if active.is_empty() {
        return Ok(lengths);
    }
    if active.len() == 1 {
        // A single-symbol alphabet still has to satisfy Kraft equality
        // (`build_from_lengths` accumulates codes per length tier and
        // requires the accumulator to wrap to 0). One length-1 entry
        // alone has Kraft sum 1/2; we round it out by giving an
        // unused dummy symbol the other length-1 code. The dummy is
        // never emitted (its histogram count is 0), so the wire
        // bytes are unchanged — this is purely a table-shape
        // adjustment for the decoder-side build invariant.
        let only = active[0].0 as usize;
        lengths[only] = 1;
        let dummy = if only == 0 { 1 } else { 0 };
        lengths[dummy] = 1;
        return Ok(lengths);
    }
    if active.len() == 2 {
        lengths[active[0].0 as usize] = 1;
        lengths[active[1].0 as usize] = 1;
        return Ok(lengths);
    }

    const MAX_L: usize = 31;
    let n = active.len();
    // Sort active symbols by frequency ascending (then by symbol).
    active.sort_by_key(|&(s, f)| (f, s));

    // package-merge state: per length-tier `tier`, a vector of
    // "packages" each carrying weight + a bitset (or list) of which
    // original symbols it contributed to. Since n ≤ 256, we can store
    // package memberships compactly as `Vec<u16>` (sorted symbol ids).
    #[derive(Clone)]
    struct Pkg {
        weight: u64,
        // List of original symbol indices (sorted) contributing to this package.
        members: Vec<u16>,
    }

    // Tier 0 (deepest): one package per active symbol with weight = freq.
    let mut tier: Vec<Pkg> = active
        .iter()
        .map(|&(s, f)| Pkg {
            weight: f as u64,
            members: vec![s],
        })
        .collect();

    let leaves: Vec<Pkg> = tier.clone();

    // Iterate up MAX_L - 1 levels; each level packages the previous
    // tier (consecutive pairs) and merges with the leaf list.
    for _ in 0..(MAX_L - 1) {
        // Package: pair up tier[0..2*k] into k packages.
        let mut packaged: Vec<Pkg> = Vec::with_capacity(tier.len() / 2);
        let mut i = 0;
        while i + 1 < tier.len() {
            let mut members = tier[i].members.clone();
            members.extend_from_slice(&tier[i + 1].members);
            members.sort_unstable();
            packaged.push(Pkg {
                weight: tier[i].weight + tier[i + 1].weight,
                members,
            });
            i += 2;
        }
        // Merge with leaves (both already sorted by weight ascending).
        let mut merged: Vec<Pkg> = Vec::with_capacity(packaged.len() + leaves.len());
        let mut a = 0usize;
        let mut b = 0usize;
        while a < packaged.len() && b < leaves.len() {
            if packaged[a].weight <= leaves[b].weight {
                merged.push(packaged[a].clone());
                a += 1;
            } else {
                merged.push(leaves[b].clone());
                b += 1;
            }
        }
        while a < packaged.len() {
            merged.push(packaged[a].clone());
            a += 1;
        }
        while b < leaves.len() {
            merged.push(leaves[b].clone());
            b += 1;
        }
        tier = merged;
    }

    // Take the cheapest 2*n - 2 packages from the final tier; for each,
    // each contributing leaf gets +1 length contribution.
    // Actually the standard formulation: take 2*n - 2 cheapest from the
    // top tier; symbol's length = number of times it appears among the
    // 2n-2 selected packages.
    let take = 2 * n - 2;
    if tier.len() < take {
        return Err(Error::invalid(
            "package-merge: insufficient packages — alphabet too small for max length",
        ));
    }
    // tier is already sorted by weight ascending.
    let mut counts = [0u8; 256];
    for pkg in tier.iter().take(take) {
        for &s in pkg.members.iter() {
            counts[s as usize] = counts[s as usize].saturating_add(1);
        }
    }
    for &(s, _) in active.iter() {
        let l = counts[s as usize];
        if l == 0 || l > MAX_L as u8 {
            return Err(Error::invalid(format!(
                "package-merge: symbol {s} got out-of-range length {l}"
            )));
        }
        lengths[s as usize] = l;
    }
    // Sanity: Kraft equality.
    let mut kraft: u64 = 0;
    for &l in lengths.iter() {
        if l > 0 {
            kraft = kraft.saturating_add(1u64 << (MAX_L as u8 - l));
        }
    }
    let target: u64 = 1u64 << (MAX_L as u8);
    if kraft != target {
        return Err(Error::invalid(format!(
            "package-merge: Kraft sum {kraft} != target {target}"
        )));
    }
    Ok(lengths)
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
    /// Maximum non-zero length present in `entries` (1..=31, or 0 if
    /// the table is empty).
    pub max_length: u8,
    /// Fast decode LUT keyed on the top 16 bits of the bit window.
    /// Each `u16` slot encodes either:
    /// - **fast hit** (top bit clear): low 8 bits = symbol; bits 8..15
    ///   = length (1..=16). The decoder consumes `length` bits.
    /// - **overflow** (top bit set, value `0x8000`): the code matching
    ///   this 16-bit window is longer than 16 bits; caller falls back
    ///   to [`decode_one_slow`] which walks the per-symbol entries on
    ///   the full 32-bit window.
    ///
    /// Built once at table-construction time by
    /// [`HuffTable::build_from_lengths`] /
    /// [`v1x_table_from_pair`]; subsequent decode is O(1) for the
    /// common path. spec/03 §3.2.2 (the proprietary's primary-LUT +
    /// secondary-walk pattern). 65 536 × `u16` = 128 KiB per table,
    /// allocated on the heap to avoid blowing the stack.
    pub primary_lut: Box<[u16; 65536]>,
}

/// Sentinel value in [`HuffTable::primary_lut`] that means "code is
/// longer than 16 bits; walk the slow path".
pub const PRIMARY_LUT_OVERFLOW: u16 = 0x8000;

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
        let primary_lut = build_primary_lut(&entries);
        Ok(Self {
            entries,
            max_length,
            primary_lut,
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
            let primary_lut = build_primary_lut(&entries);
            return Ok(Self {
                entries,
                max_length,
                primary_lut,
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
        let primary_lut = build_primary_lut(&entries);
        Ok(Self {
            entries,
            max_length,
            primary_lut,
        })
    }
}

/// Build the 65 536-entry primary LUT for [`decode_one`]'s fast
/// path. For each (length, code) ≤ 16 bits, every 16-bit window whose
/// top `length` bits match the code is filled with `(length << 8) |
/// symbol`. Slots whose top bits select a code longer than 16 bits
/// are filled with [`PRIMARY_LUT_OVERFLOW`]. spec/03 §3.2.2 (the
/// proprietary's primary-LUT-then-walk pattern: ~99 % of codes hit
/// the primary table for the binary's ship-shipped 8-bit-mode
/// classic blobs).
fn build_primary_lut(entries: &[HuffEntry; 256]) -> Box<[u16; 65536]> {
    let mut lut: Box<[u16; 65536]> = vec![PRIMARY_LUT_OVERFLOW; 65536]
        .into_boxed_slice()
        .try_into()
        .expect("65536 element vec");
    for (sym, e) in entries.iter().enumerate() {
        let l = e.length;
        if l == 0 || l > 16 {
            // Length 0 = absent; length > 16 = overflow handled by
            // slow path. The slow path does the full per-symbol scan
            // when the LUT slot is `PRIMARY_LUT_OVERFLOW`.
            continue;
        }
        // Top L bits of the 32-bit MSB-aligned code, right-aligned.
        let prefix = (e.code >> (32 - l as u32)) as u16;
        // The 16-bit LUT slot has L "fixed" high bits (= prefix shifted
        // up by 16-L) and (16-L) "free" low bits. We fill all 2^(16-L)
        // such slots with the same hit-record.
        let shift = 16 - l as u32;
        let base = (prefix as u32) << shift;
        let count = 1u32 << shift;
        let payload = ((l as u16) << 8) | sym as u16;
        for i in 0..count {
            let idx = (base | i) as usize;
            lut[idx] = payload;
        }
    }
    lut
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
    let primary_lut = build_primary_lut(&entries);
    Ok(HuffTable {
        entries,
        max_length,
        primary_lut,
    })
}

/// Decode the next symbol from a 32-bit MSB-first bit window. Returns
/// `(symbol, length)`. Hot path: a single 16-bit indexed load into
/// the primary LUT (≤ 16-bit codes). For codes longer than 16 bits
/// the LUT slot is [`PRIMARY_LUT_OVERFLOW`] and we fall through to
/// [`decode_one_slow`]. spec/03 §3.2.2.
#[inline]
pub fn decode_one(table: &HuffTable, window: u32) -> Result<(u8, u8)> {
    let prefix = (window >> 16) as usize;
    let slot = table.primary_lut[prefix];
    if slot != PRIMARY_LUT_OVERFLOW {
        // Fast hit: low 8 bits = symbol, bits 8..15 = length.
        let length = (slot >> 8) as u8;
        let symbol = (slot & 0xFF) as u8;
        return Ok((symbol, length));
    }
    decode_one_slow(table, window)
}

/// Slow-path decoder for codes longer than 16 bits — walks the
/// per-symbol entries on the full 32-bit window. Only invoked from
/// [`decode_one`] when the primary LUT signals overflow.
pub fn decode_one_slow(table: &HuffTable, window: u32) -> Result<(u8, u8)> {
    for (sym, entry) in table.entries.iter().enumerate() {
        let l = entry.length;
        if l == 0 || l <= 16 {
            // Length 0 = absent; length ≤ 16 would have been served by
            // the LUT — if we reached the slow path, the matching code
            // must be > 16 bits (overflow slot).
            continue;
        }
        let mask: u32 = if l == 32 {
            u32::MAX
        } else {
            !0u32 << (32 - l as u32)
        };
        if (window & mask) == entry.code {
            return Ok((sym as u8, l));
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
