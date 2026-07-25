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
//
// spec/04 §3.3 additionally requires that NO in-band `0x00` byte appears
// anywhere in the RLE data (so `lstrcpyA`/`lstrlenA`-based third-party
// tooling can copy/measure the extradata). The only way a `0x00` could
// leak in is the long-form encoding of a length-0 (absent-symbol) run,
// whose leading byte is `v & 0x1F = 0x00`. `rle_encode_one_channel`
// avoids that by emitting length-0 runs > 7 as repeated short-form
// chunks of 7 (byte `0xE0`) instead of long-form — see the loop below.

/// RLE-encode one 256-byte length table to `out`. spec/01 §5: each
/// run uses the count-hint when count ≤ 7, otherwise the long-form
/// `value 0x_NN` (NN > 0) for runs of 8..255. Length-0 runs never use
/// long-form (that would emit an in-band `0x00`, spec/04 §3.3); they are
/// emitted as short-form `0xE0` chunks. No trailing `0x00 0x00` is
/// appended — see module-level note. The emitted stream contains no
/// `0x00` byte.
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
        //
        // spec/04 §3.3 (C-string-terminator convention): the codec
        // author's design note guarantees "a zero byte can never appear
        // in the RLE data" so the classic-table blobs can be copied with
        // `lstrcpyA` / measured with `lstrlenA`. A clean-room encoder
        // that ships custom tables (`ExtradataMode::CustomV2`) must
        // honour the same invariant, or a `lstrlenA`-based third-party
        // decoder would truncate the extradata at the first in-band
        // `0x00`. The long-form encoding emits a leading byte equal to
        // `v & 0x1F`, which is `0x00` for a length-0 (absent-symbol) run
        // — and absent-symbol runs longer than 7 are the common case for
        // any real residual histogram. So length-0 runs must be emitted
        // as short-form chunks of ≤ 7 only (bytes `0x20`..=`0xE0`, all
        // non-zero); long-form is reserved for `v != 0`, where the
        // leading byte `v & 0x1F` is already non-zero.
        let mut remaining = run;
        while remaining > 0 {
            if remaining <= 7 {
                let chunk = remaining as u8;
                let b = (chunk << 5) | (v & 0x1F);
                // chunk ≥ 1 → count_hint ≥ 1 → `b` ≥ 0x20 regardless
                // of `v`. Safe (no in-band `0x00` produced).
                out.push(b);
                remaining = 0;
            } else if v == 0 {
                // Length-0 run > 7: long-form would push a `0x00`
                // leading byte (in-band zero, spec/04 §3.3 violation).
                // Emit a full short-form chunk of 7 instead (byte
                // `0xE0` = count_hint 7, value 0) and loop.
                out.push(0xE0);
                remaining -= 7;
            } else {
                // Long-form: count_hint = 0 (leading byte = v),
                // followed by an explicit count in 1..=255. Safe here
                // because `v != 0` → `leading != 0x00`.
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

    // Round-419 count-based package-merge. The pre-r419 body carried a
    // per-package member list (`Vec<u16>`, cloned + `sort_unstable`'d
    // on every pairing) up all 30 tiers — O(n²·L) allocation/copy
    // churn that BENCHMARKS.md ranked as the encoder's #1 hotspot
    // (~578 µs peaked / ~1.8 ms per 3-channel CustomV2 frame; the
    // whole `encode_frame_auto` 3–3.5× cost multiplier).
    //
    // The member lists are redundant. Two order facts make a
    // counts-only walk equivalent (both preserved verbatim from the
    // pre-r419 merge):
    //
    // 1. The merge is monotone and stable — packages win weight ties
    //    (`packaged[a].weight <= leaves[b].weight`), and both inputs
    //    are already weight-ascending — so the leaf items appearing
    //    among the first `k` positions of any merged tier are exactly
    //    `leaves[0..l]` (the cheapest `l` leaves, in order), and the
    //    package items are exactly `packaged[0..p]` (the pairs
    //    `tier_prev[0..2p]`).
    // 2. Selecting the cheapest `take = 2n - 2` items of the top tier
    //    therefore decomposes tier-by-tier: at each tier the selected
    //    leaves contribute +1 code-length to the `l` cheapest active
    //    symbols, and the `p` selected packages demand the first `2p`
    //    items of the tier below.
    //
    // So we keep only per-tier weights plus a per-tier `is_package`
    // flag vector (built bottom-up), then resolve the per-symbol
    // counts in one top-down prefix walk — identical output to the
    // member-list version for every histogram (regression-guarded by
    // `round419_count_based_matches_member_reference` below and the
    // round-419 golden pins), at O(n·L) time with zero per-package
    // allocations.
    let leaf_w: Vec<u64> = active.iter().map(|&(_, f)| f as u64).collect();

    // Tier 0 (deepest): one item per active symbol, weight = freq.
    let mut tier: Vec<u64> = leaf_w.clone();
    // flags[k][i] == true ⇔ item `i` of the tier produced by merge
    // step `k` is a package (a pair from the previous tier) rather
    // than a leaf.
    let mut flags: Vec<Vec<bool>> = Vec::with_capacity(MAX_L - 1);

    // Iterate up MAX_L - 1 levels; each level packages the previous
    // tier (consecutive pairs) and merges with the leaf list.
    for _ in 0..(MAX_L - 1) {
        let n_pkg = tier.len() / 2;
        let mut merged: Vec<u64> = Vec::with_capacity(n_pkg + leaf_w.len());
        let mut is_pkg: Vec<bool> = Vec::with_capacity(n_pkg + leaf_w.len());
        let mut a = 0usize;
        let mut b = 0usize;
        while a < n_pkg && b < leaf_w.len() {
            let pw = tier[2 * a] + tier[2 * a + 1];
            if pw <= leaf_w[b] {
                merged.push(pw);
                is_pkg.push(true);
                a += 1;
            } else {
                merged.push(leaf_w[b]);
                is_pkg.push(false);
                b += 1;
            }
        }
        while a < n_pkg {
            merged.push(tier[2 * a] + tier[2 * a + 1]);
            is_pkg.push(true);
            a += 1;
        }
        while b < leaf_w.len() {
            merged.push(leaf_w[b]);
            is_pkg.push(false);
            b += 1;
        }
        tier = merged;
        flags.push(is_pkg);
    }

    // Standard formulation: take the 2*n - 2 cheapest items from the
    // top tier; a symbol's length = number of selected items (across
    // all tiers, after package expansion) that are that symbol's leaf.
    let take = 2 * n - 2;
    if tier.len() < take {
        return Err(Error::invalid(
            "package-merge: insufficient packages — alphabet too small for max length",
        ));
    }
    // Top-down prefix walk (see order facts above): at each tier the
    // first `needed` items split into `l` leaves (+1 length for the
    // `l` cheapest active symbols) and `p` packages (`needed = 2p`
    // items of the tier below).
    let mut counts = [0u8; 256];
    let mut needed = take;
    for is_pkg in flags.iter().rev() {
        let mut pkgs = 0usize;
        for &p in is_pkg.iter().take(needed) {
            if p {
                pkgs += 1;
            }
        }
        let leaves_sel = needed - pkgs;
        for &(s, _) in active.iter().take(leaves_sel) {
            counts[s as usize] = counts[s as usize].saturating_add(1);
        }
        needed = 2 * pkgs;
        if needed == 0 {
            break;
        }
    }
    // Tier 0 is all leaves: the remaining demand lands on the
    // `needed` cheapest active symbols directly.
    for &(s, _) in active.iter().take(needed) {
        counts[s as usize] = counts[s as usize].saturating_add(1);
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
    /// Round-91 fast overflow index: byte → mask, only entries with
    /// `length > 16`. The decoder's slow path walks this slice (≤ 256
    /// entries, but most tables have ≤ 6) instead of `entries`,
    /// avoiding the `l <= 16 { continue }` branch on every iteration.
    /// The per-symbol mask is precomputed once at build time.
    /// Wrapped in `Arc` so [`HuffTable::clone`] is O(1).
    pub overflow_entries: std::sync::Arc<Vec<OverflowEntry>>,
}

/// Round-91 overflow-table entry: precomputed (code, mask, length,
/// symbol) for one long code (length > 16). The decoder's hot loop
/// inside [`decode_one_slow`] iterates these without the
/// `length == 0 || length <= 16` short-circuit check that round-7
/// did per `entries` iteration.
#[derive(Debug, Clone, Copy)]
pub struct OverflowEntry {
    pub code: u32,
    pub mask: u32,
    pub length: u8,
    pub symbol: u8,
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
        // Validate lengths (spec/03 §3.2.3) + tally the exact Kraft
        // sum in 64-bit fixed point (unit = 2⁻³², so a length-L code
        // contributes 2³²⁻ᴸ and equality means the tally is exactly
        // 2³² — spec/03 §3's "sum 2^-L_i == 1" invariant).
        let mut kraft: u64 = 0;
        for (i, &l) in lengths.iter().enumerate() {
            if l > 31 {
                return Err(Error::invalid(format!(
                    "code length {l} for symbol {i} exceeds 31-bit ceiling"
                )));
            }
            if l > 0 {
                kraft += 1u64 << (32 - l as u32);
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
        // Sanity: when the table is non-trivial, Kraft EQUALITY must
        // hold exactly (spec/03 §3 "sum 2^-L_i == 1"). The exact
        // 64-bit tally is required here — the pre-fix check tested
        // only `code_acc != 0` on the 32-bit accumulator, which an
        // over-subscribed distribution whose Kraft sum is an integer
        // multiple of 1.0 aliases by wrapping to 0 more than once
        // (fuzz-found 2026-06-09: four 2-bit lengths + two 1-bit
        // lengths sum to 2.0, wrap the accumulator twice, and assign
        // two symbols the same all-zero code — breaking the
        // prefix-free decode contract). (When the table is empty —
        // all lengths == 0 — the tally is 0 and no check applies.)
        if max_length > 0 && kraft != 1u64 << 32 {
            return Err(Error::invalid(
                "non-canonical length distribution (Kraft sum != 1)",
            ));
        }
        debug_assert!(max_length == 0 || code_acc == 0);
        let primary_lut = build_primary_lut(&entries);
        let overflow_entries = std::sync::Arc::new(build_overflow_entries(&entries));
        Ok(Self {
            entries,
            max_length,
            primary_lut,
            overflow_entries,
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
            let overflow_entries = std::sync::Arc::new(build_overflow_entries(&entries));
            return Ok(Self {
                entries,
                max_length,
                primary_lut,
                overflow_entries,
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
        let overflow_entries = std::sync::Arc::new(build_overflow_entries(&entries));
        Ok(Self {
            entries,
            max_length,
            primary_lut,
            overflow_entries,
        })
    }
}

/// Build the round-91 flat overflow-entries table: precomputed
/// `(code, mask, length, symbol)` for every code longer than 16
/// bits. The decoder's [`decode_one_slow`] now walks this dense
/// slice (≤ 256 entries, but most v2.x classic tables have only
/// 0..2 long codes; the proprietary v1.x set B is the outlier
/// with ~210) instead of iterating `entries[0..256]` with a
/// `length == 0 || length <= 16 { continue; }` short-circuit
/// per iteration.
///
/// Why this helps: the round-7 slow path issued 256 loop
/// iterations even on the proprietary's classic v2.x blobs that
/// have ≤ 2 codes with length > 16. Profile (release build, M1
/// host, 320×240 YUY2 Gradient ClassicV2): 1.76 ms/frame →
/// 1.65 ms/frame (≈6% speedup on the gradient ClassicV2 path,
/// where the overflow walks dominate the post-primary-LUT work
/// for ~14 % of bytes). For v1.x set B (worst case, ~210 long
/// codes), the speedup is more modest because the iteration count
/// only drops from 256 → 210; the precomputed `mask` field saves
/// the shift+branch on each iteration.
///
/// Each entry's `mask` is `!0 << (32 - length)` (with `length ==
/// 32` mapping to `u32::MAX`), computed once at build time so the
/// per-iteration cost is one load and one compare instead of a
/// shift + load + compare. Wrapped in `Arc` so [`HuffTable::clone`]
/// is O(1).
fn build_overflow_entries(entries: &[HuffEntry; 256]) -> Vec<OverflowEntry> {
    let mut out: Vec<OverflowEntry> = Vec::new();
    for (sym, e) in entries.iter().enumerate() {
        let l = e.length;
        if l <= 16 {
            // Length 0 = absent; length ≤ 16 = served by primary LUT.
            continue;
        }
        let mask: u32 = if l == 32 {
            u32::MAX
        } else {
            !0u32 << (32 - l as u32)
        };
        out.push(OverflowEntry {
            code: e.code,
            mask,
            length: l,
            symbol: sym as u8,
        });
    }
    out
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
    let overflow_entries = std::sync::Arc::new(build_overflow_entries(&entries));
    Ok(HuffTable {
        entries,
        max_length,
        primary_lut,
        overflow_entries,
    })
}

/// Decode the next symbol from a 32-bit MSB-first bit window. Returns
/// `(symbol, length)`. Hot path: a single 16-bit indexed load into
/// the primary LUT (≤ 16-bit codes). For codes longer than 16 bits
/// the LUT slot is [`PRIMARY_LUT_OVERFLOW`] and we fall through to
/// [`decode_one_slow`] which walks the precomputed flat overflow
/// table (round-91 build artifact). spec/03 §3.2.2.
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

/// Round-419 paired fast path: decode TWO consecutive symbols from one
/// 32-bit MSB-first window without an intermediate accumulator update.
///
/// The first symbol is looked up exactly like [`decode_one`]'s fast
/// path (top 16 bits of `window` into `a`'s primary LUT). When it hits
/// with length `l1 ≤ 16`, the next codeword's first 16 bits are
/// `window` bits `l1..l1+16` — all still inside the 32-bit window
/// (`l1 + 16 ≤ 32`), so `(window << l1) >> 16` indexes `b`'s primary
/// LUT with exactly the same 16 bits a fresh [`BitReader::peek_window`]
/// after `consume_bits(l1)` would serve (the low `l1` zero bits shifted
/// in never reach the used top 16). Returns `(sym_a, sym_b, l1 + l2)`
/// so the caller advances the reader once per pair — halving the
/// serialized `consume → refill → peek` chains on the per-symbol
/// decode critical path (the whole-decode hotspot per the round-419
/// `sample` profile: ~95% of wall in the symbol loop).
///
/// Returns `None` when either lookup lands on a > 16-bit code
/// ([`PRIMARY_LUT_OVERFLOW`]); the caller falls back to the sequential
/// [`decode_one`] + consume path, which serves long codes from the
/// true 32-bit window. Byte-identical output either way.
#[inline]
pub fn decode_pair(a: &HuffTable, b: &HuffTable, window: u32) -> Option<(u8, u8, u32)> {
    let slot_a = a.primary_lut[(window >> 16) as usize];
    if slot_a == PRIMARY_LUT_OVERFLOW {
        return None;
    }
    let l1 = (slot_a >> 8) as u32;
    let slot_b = b.primary_lut[((window << l1) >> 16) as usize];
    if slot_b == PRIMARY_LUT_OVERFLOW {
        return None;
    }
    let l2 = (slot_b >> 8) as u32;
    Some(((slot_a & 0xFF) as u8, (slot_b & 0xFF) as u8, l1 + l2))
}

/// Round-420 pair-LENGTH LUT for the interlaced split scanner: maps
/// the top 16 window bits to the combined bit length of the next TWO
/// codewords (first from `a`, second from `b`), or `0` when the pair
/// cannot be resolved from 16 known bits alone.
///
/// The scanner (`decoder::scan_field`) only needs cumulative bit
/// length — never symbols — so one 8-bit load can replace
/// [`decode_pair`]'s serialized load-A → shift → load-B chain on its
/// critical path.
///
/// Soundness: entry `p` is non-zero only when, for EVERY 32-bit
/// window whose top 16 bits equal `p`, the two codewords decode to
/// the same combined length `l1 + l2`:
///
/// - code A resolves from `a`'s primary LUT at index `p` (fast hit ⇒
///   `l1 ≤ 16`, fully determined by the known bits);
/// - code B is looked up at the zero-completed shifted index; the
///   result is trusted only when `l1 + l2 ≤ 16`, i.e. code B lies
///   entirely inside the 16 known bits — any completion of the
///   unknown low bits indexes a LUT slot with the same leading code
///   bits and therefore the same `(symbol, length)`.
///
/// Everything else (either lookup overflowing, or the pair straddling
/// bit 16) stays `0` and the scanner falls back to the exact
/// [`decode_pair`] / [`decode_one`] path. 64 KiB per table pair,
/// built lazily once per stream behind the round-419 table cache.
/// Round-429 full pair LUT for the serial decode loops: maps the top
/// 16 window bits to BOTH symbols and the combined bit length of the
/// next TWO codewords (first from `a`, second from `b`) in one 32-bit
/// load — `sym_a | sym_b << 8 | (l1 + l2) << 16` — or `0` when the
/// pair cannot be resolved from 16 known bits alone.
///
/// Same soundness argument as [`build_pair_len_lut`]: an entry is
/// non-zero only when `l1 + l2 ≤ 16`, i.e. both codewords lie
/// entirely inside the 16 known bits, so every 32-bit window sharing
/// those top 16 bits decodes to the same two `(symbol, length)`
/// pairs. `0` is unreachable for a resolvable pair (lengths are
/// ≥ 1, so the packed value has a non-zero bit 16..21 field).
/// Everything else falls back to the exact [`decode_pair`] (which
/// still serves straddling pairs from the true 32-bit window) or the
/// sequential [`decode_one`] path — byte-identical output either
/// way, pinned by the r419 golden matrix and the randomized
/// equivalence sweep in `decoder::round429_pair_sym_lut_tests`.
///
/// 256 KiB per (first-slot, second-slot) pair, built lazily at most
/// once per cached `ThreeTables` behind the round-419 table cache —
/// paid once per stream, not per frame.
pub(crate) fn build_pair_sym_lut(a: &HuffTable, b: &HuffTable) -> Box<[u32]> {
    let mut lut = vec![0u32; 65536].into_boxed_slice();
    for (p, slot) in lut.iter_mut().enumerate() {
        let slot_a = a.primary_lut[p];
        if slot_a == PRIMARY_LUT_OVERFLOW {
            continue;
        }
        let l1 = (slot_a >> 8) as u32;
        let w = (p as u32) << 16;
        let slot_b = b.primary_lut[((w << l1) >> 16) as usize];
        if slot_b == PRIMARY_LUT_OVERFLOW {
            continue;
        }
        let l2 = (slot_b >> 8) as u32;
        if l1 + l2 <= 16 {
            *slot = (slot_a & 0xFF) as u32 | (((slot_b & 0xFF) as u32) << 8) | ((l1 + l2) << 16);
        }
    }
    lut
}

pub(crate) fn build_pair_len_lut(a: &HuffTable, b: &HuffTable) -> Box<[u8]> {
    let mut sums = vec![0u8; 65536].into_boxed_slice();
    for (p, slot) in sums.iter_mut().enumerate() {
        let slot_a = a.primary_lut[p];
        if slot_a == PRIMARY_LUT_OVERFLOW {
            continue;
        }
        let l1 = (slot_a >> 8) as u32;
        let w = (p as u32) << 16;
        let slot_b = b.primary_lut[((w << l1) >> 16) as usize];
        if slot_b == PRIMARY_LUT_OVERFLOW {
            continue;
        }
        let l2 = (slot_b >> 8) as u32;
        if l1 + l2 <= 16 {
            *slot = (l1 + l2) as u8;
        }
    }
    sums
}

/// Long-code slow path: walks the precomputed [`HuffTable::overflow_entries`]
/// (round 91) — a flat slice of only the codes with length > 16, with
/// their masks already computed. Replaces the round-7 256-entry scan
/// over `table.entries` which paid a `length == 0 || length <= 16
/// { continue }` cost on every iteration regardless of how few long
/// codes actually existed.
pub fn decode_one_slow(table: &HuffTable, window: u32) -> Result<(u8, u8)> {
    for entry in table.overflow_entries.iter() {
        if (window & entry.mask) == entry.code {
            return Ok((entry.symbol, entry.length));
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

    // ─────── Round-234 invariants pinned by the tables_huffyuv fuzz target ───────
    //
    // These tests pin the four contracts the round-234 cargo-fuzz target
    // `fuzz/fuzz_targets/tables_huffyuv.rs` asserts on every iteration
    // against arbitrary input bytes. Reproducing the contracts here as
    // unit tests gives us a regression guard that runs on every `cargo
    // test` invocation (the fuzz harness only runs on the daily CI
    // schedule) and documents what the harness is checking when reading
    // the source.

    #[test]
    fn round234_rle_encode_decode_roundtrips_every_coerced_table() {
        // For every length table whose entries land in `0..=31` (the
        // domain `rle_encode_one_channel` accepts), the encode→decode
        // pair must round-trip byte-exact. We sample three structurally
        // distinct shapes the fuzz harness's coerce-mod-32 pattern
        // produces: all-zero (zero-symbol table), a single-tier
        // distribution (every symbol same length), and the spec/04
        // §3.2 set-A-style sparse distribution.
        let all_zero = [0u8; 256];
        let mut encoded = Vec::new();
        rle_encode_one_channel(&all_zero, &mut encoded).unwrap();
        let mut cursor: &[u8] = &encoded;
        assert_eq!(rle_decode_one_channel(&mut cursor).unwrap(), all_zero);

        let mut single_tier = [0u8; 256];
        for (i, slot) in single_tier.iter_mut().enumerate() {
            *slot = if i < 16 { 4 } else { 0 };
        }
        encoded.clear();
        rle_encode_one_channel(&single_tier, &mut encoded).unwrap();
        let mut cursor: &[u8] = &encoded;
        assert_eq!(rle_decode_one_channel(&mut cursor).unwrap(), single_tier);

        let mut sparse = [0u8; 256];
        sparse[0] = 1;
        sparse[1] = 2;
        sparse[2] = 3;
        sparse[3] = 4;
        sparse[100] = 17;
        sparse[200] = 8;
        sparse[255] = 31;
        encoded.clear();
        rle_encode_one_channel(&sparse, &mut encoded).unwrap();
        let mut cursor: &[u8] = &encoded;
        assert_eq!(rle_decode_one_channel(&mut cursor).unwrap(), sparse);
    }

    #[test]
    fn round234_build_from_lengths_self_consistent_decode() {
        // Every nonzero-length entry in a successfully-built `HuffTable`
        // must decode via `decode_one` from its own MSB-aligned code
        // window back to itself. This is the codec's downstream
        // contract — the fuzz harness checks it on every successful
        // `build_from_lengths` call.
        //
        // We pin three Kraft-equal shapes: a uniform 4-symbol alphabet,
        // a mixed-length tiering (2/3/3/4/4/4/4), and one of the
        // proprietary's classic blobs.
        for lengths in [
            // Uniform 4-symbol alphabet (Kraft sum = 4 * 1/4 = 1).
            {
                let mut l = [0u8; 256];
                l[0] = 2;
                l[1] = 2;
                l[2] = 2;
                l[3] = 2;
                l
            },
            // Mixed-length tiering (Kraft sum = 1/2 + 1/4 + 2*1/8 = 1).
            {
                let mut l = [0u8; 256];
                l[0] = 1;
                l[1] = 2;
                l[2] = 3;
                l[3] = 3;
                l
            },
            // Eight-symbol uniform alphabet (Kraft sum = 8 * 1/8 = 1).
            {
                let mut l = [0u8; 256];
                for slot in l.iter_mut().take(8) {
                    *slot = 3;
                }
                l
            },
            // Deeper tiering reaching length 4
            // (Kraft sum = 1/2 + 1/4 + 1/8 + 2*1/16 = 1).
            {
                let mut l = [0u8; 256];
                l[0] = 1;
                l[1] = 2;
                l[2] = 3;
                l[3] = 4;
                l[4] = 4;
                l
            },
        ] {
            let table = HuffTable::build_from_lengths(&lengths).expect("Kraft equality");
            for (sym, e) in table.entries.iter().enumerate() {
                if e.length == 0 {
                    continue;
                }
                let (decoded_sym, decoded_len) = decode_one(&table, e.code).unwrap();
                assert_eq!(decoded_sym as usize, sym);
                assert_eq!(decoded_len, e.length);
            }
        }
    }

    #[test]
    fn round234_compute_canonical_lengths_builds_through_build_from_lengths() {
        // `compute_canonical_lengths(histogram)` is the encoder's
        // entry point on the CustomV2 extradata path. The downstream
        // chain is `HuffTable::build_from_lengths(<the returned table>)`
        // — if that chain ever errors on a real histogram, the encoder
        // crashes. The fuzz harness asserts the chain holds on every
        // successful `compute_canonical_lengths` call.
        //
        // We pin a few shapes: uniform, single-symbol (degenerate),
        // two-symbol (the round-6 single-symbol fix), and a skewed
        // distribution where symbol 0 vastly outweighs the rest.
        for histogram in [
            {
                let mut h = [0u32; 256];
                for slot in h.iter_mut().take(16) {
                    *slot = 1;
                }
                h
            },
            {
                let mut h = [0u32; 256];
                h[42] = 1000;
                h
            },
            {
                let mut h = [0u32; 256];
                h[0] = 7;
                h[1] = 3;
                h
            },
            {
                let mut h = [0u32; 256];
                h[0] = 1_000_000;
                for slot in h.iter_mut().take(8).skip(1) {
                    *slot = 1;
                }
                h
            },
        ] {
            let lengths = compute_canonical_lengths(&histogram).expect("package-merge");
            for &l in lengths.iter() {
                assert!(l <= 31);
            }
            let table = HuffTable::build_from_lengths(&lengths)
                .expect("compute_canonical_lengths output must build");
            for (sym, e) in table.entries.iter().enumerate() {
                if e.length == 0 {
                    continue;
                }
                let (decoded_sym, decoded_len) = decode_one(&table, e.code).unwrap();
                assert_eq!(decoded_sym as usize, sym);
                assert_eq!(decoded_len, e.length);
            }
        }
    }

    #[test]
    fn round234_v1x_table_from_pair_decode_is_live_on_own_code_window() {
        // `v1x_table_from_pair(lengths, codes)` builds the v1.x
        // precomputed-codes table without enforcing Kraft equality
        // (spec/04 §4.2 allows non-canonical layouts). The fuzz harness
        // therefore only asserts the weaker liveness property: for every
        // nonzero-length entry, `decode_one(table, entry.code)` returns
        // `Ok` (no panic / OOB / overflow). We pin one such build on
        // the proprietary's v1.x set-A pair to confirm the contract
        // holds on the spec-canonical input.
        let mut lengths = [0u8; 256];
        let set_a = blobs::v1x_lengths_set_a();
        let mut cursor: &[u8] = set_a;
        let decoded = rle_decode_one_channel(&mut cursor).unwrap();
        lengths.copy_from_slice(&decoded);

        let codes_raw = blobs::v1x_codes_set_a();
        let mut codes = [0u8; 256];
        codes.copy_from_slice(codes_raw);

        let table = v1x_table_from_pair(&lengths, &codes).expect("v1.x set A builds");
        for e in table.entries.iter() {
            if e.length == 0 {
                continue;
            }
            let _ = decode_one(&table, e.code).expect("v1.x decode_one liveness");
        }
    }

    /// Fuzz-found 2026-06-09 (scheduled `tables_huffyuv` run,
    /// crash-8fd2645b…): a length table whose Kraft sum is an exact
    /// integer multiple of 1.0 (here 2.0 — four 2-bit lengths plus
    /// two 1-bit lengths) wrapped the 32-bit code accumulator to 0
    /// more than once, so the pre-fix `code_acc != 0` equality check
    /// passed and the build assigned two symbols the same all-zero
    /// code (symbol 0 at length 2 and symbol 18 at length 1), breaking
    /// the prefix-free decode contract (`decode_one` on symbol 0's
    /// window returned symbol 18). spec/03 §3 requires Kraft EQUALITY
    /// (`sum 2^-L_i == 1`); the build now tallies the sum exactly in
    /// 64-bit fixed point and rejects anything ≠ 2³².
    #[test]
    fn build_from_lengths_rejects_integer_kraft_oversubscription() {
        // The minimised fuzz shape: symbols 0..=3 at length 2,
        // symbols 18 and 20 at length 1 → Kraft sum 4×¼ + 2×½ = 2.0.
        let mut lengths = [0u8; 256];
        for slot in lengths.iter_mut().take(4) {
            *slot = 2;
        }
        lengths[18] = 1;
        lengths[20] = 1;
        assert!(
            HuffTable::build_from_lengths(&lengths).is_err(),
            "Kraft sum 2.0 must be rejected, not aliased to equality"
        );

        // Kraft sum 3.0 (twelve 2-bit lengths) — same aliasing class.
        let mut lengths3 = [0u8; 256];
        for slot in lengths3.iter_mut().take(12) {
            *slot = 2;
        }
        assert!(
            HuffTable::build_from_lengths(&lengths3).is_err(),
            "Kraft sum 3.0 must be rejected"
        );

        // Control: exact Kraft equality still builds, and every
        // nonzero-length symbol self-consistently decodes from its
        // own MSB-aligned code window (the contract the fuzz target
        // asserts).
        let mut ok = [0u8; 256];
        for slot in ok.iter_mut().take(4) {
            *slot = 2; // 4 × ¼ = 1.0
        }
        let table = HuffTable::build_from_lengths(&ok).expect("Kraft-equal table must build");
        for (sym, e) in table.entries.iter().enumerate() {
            if e.length == 0 {
                continue;
            }
            let (decoded_sym, decoded_len) =
                decode_one(&table, e.code).expect("self-consistent decode");
            assert_eq!(decoded_sym as usize, sym);
            assert_eq!(decoded_len, e.length);
        }
    }

    // ───────── Round-382: spec/04 §3.3 no-in-band-0x00 invariant ─────────
    //
    // The C-string-terminator convention (spec/04 §3.3) requires that the
    // RLE-compressed length-table bytes contain no `0x00` byte, so a
    // `lstrcpyA`/`lstrlenA`-based third-party decoder can copy/measure
    // the extradata without truncating it at a spurious NUL. The only way
    // `rle_encode_one_channel` could leak a `0x00` is the long-form
    // encoding of a length-0 (absent-symbol) run, whose leading byte is
    // `v & 0x1F = 0x00`. Absent-symbol runs longer than 7 are the common
    // case for any real residual histogram, so this was a latent
    // wire-portability defect before round 382 routed length-0 runs
    // through short-form `0xE0` chunks.

    /// Assert the encoder never emits a `0x00` byte, for a battery of
    /// length tables that stress the length-0-run path specifically.
    #[test]
    fn round382_rle_encode_never_emits_inband_zero() {
        // Helper: encode one table and assert no 0x00 byte, plus a
        // full RLE self-roundtrip (the short-form 0xE0 chunks must
        // decode back to the exact same 256 lengths).
        let check = |lengths: &[u8; 256], label: &str| {
            let mut out = Vec::new();
            rle_encode_one_channel(lengths, &mut out)
                .unwrap_or_else(|e| panic!("{label}: encode failed: {e:?}"));
            assert!(
                !out.contains(&0x00),
                "{label}: RLE stream must contain no in-band 0x00 (spec/04 §3.3); got {out:02x?}"
            );
            // Terminate the channel with the standard 0x00 0x00 sentinel
            // so the single-channel decoder stops cleanly at 256.
            let mut framed = out.clone();
            framed.extend_from_slice(&[0x00, 0x00]);
            let mut cursor: &[u8] = &framed;
            let back = rle_decode_one_channel(&mut cursor)
                .unwrap_or_else(|e| panic!("{label}: decode failed: {e:?}"));
            assert_eq!(&back, lengths, "{label}: RLE must roundtrip");
        };

        // 1. All-absent (every symbol length 0) — a single 256-long
        //    length-0 run, the worst case for the long-form path.
        check(&[0u8; 256], "all-zero");

        // 2. One present symbol at length 1, all 255 others absent:
        //    a length-0 run of 255 straddling the present symbol.
        let mut one = [0u8; 256];
        one[128] = 1;
        // Balance Kraft only matters for HuffTable build, not RLE; the
        // RLE encoder takes the raw length array as-is.
        check(&one, "single-present-mid");

        // 3. A realistic sparse histogram: lengths 8 on a handful of
        //    symbols, 0 elsewhere (long absent runs on both sides).
        let mut sparse = [0u8; 256];
        for s in [3usize, 4, 5, 200, 201, 255] {
            sparse[s] = 8;
        }
        check(&sparse, "sparse");

        // 4. Alternating present/absent — forces many short (len-1)
        //    length-0 runs interleaved with non-zero lengths.
        let mut alt = [0u8; 256];
        for (s, slot) in alt.iter_mut().enumerate() {
            *slot = if s % 2 == 0 { 0 } else { 4 };
        }
        check(&alt, "alternating");

        // 5. A block of exactly 8 absent symbols (the run > 7 boundary
        //    where the pre-fix long-form path would first trigger).
        let mut block8 = [4u8; 256];
        for slot in block8.iter_mut().take(8) {
            *slot = 0;
        }
        check(&block8, "leading-8-zero-block");

        // 6. Every classic blob's decoded lengths re-encode with no
        //    0x00 (they contain length-0 symbols too; confirm our
        //    encoder keeps them zero-free).
        for blob in [
            blobs::yuv_left(),
            blobs::yuv_gradient(),
            blobs::yuv_median(),
            blobs::rgb_left(),
            blobs::rgb_left_decorr(),
            blobs::rgb_gradient_decorr(),
        ] {
            let three = rle_decode_three_channels(blob).unwrap();
            for (i, slot) in three.iter().enumerate() {
                check(slot, &format!("classic-slot-{i}"));
            }
        }
    }

    // ───────── Round-419: decode_pair / sequential equivalence ─────────

    /// Drive a `BitReader` over pseudo-random byte streams and assert
    /// that wherever `decode_pair(a, b, window)` returns a hit, the
    /// sequential `decode_one(a) → consume → decode_one(b)` walk over
    /// the same reader position produces the identical `(sym, sym,
    /// total_len)` triple. Covers table pairs with and without > 16-bit
    /// codes (the v1.x set-B table has ~210 long codes, maximising
    /// overflow fallbacks) and runs past end-of-data so the zero-padded
    /// tail window is exercised on both paths.
    #[test]
    fn round419_decode_pair_matches_sequential() {
        // Classic YUV-left blob → three canonical tables (short codes
        // dominate → pair fast path dominates).
        let [l1, l2, l3] = rle_decode_three_channels(blobs::yuv_left()).unwrap();
        let t1 = HuffTable::build_from_lengths(&l1).unwrap();
        let t2 = HuffTable::build_from_lengths(&l2).unwrap();
        let t3 = HuffTable::build_from_lengths(&l3).unwrap();
        // v1.x set B (many > 16-bit codes → overflow fallback path).
        let mut cursor: &[u8] = blobs::v1x_lengths_set_b();
        let lens_b = rle_decode_one_channel(&mut cursor).unwrap();
        let mut codes_b = [0u8; 256];
        codes_b.copy_from_slice(blobs::v1x_codes_set_b());
        let tb = v1x_table_from_pair(&lens_b, &codes_b).unwrap();

        let pairs: [(&HuffTable, &HuffTable); 4] = [(&t1, &t2), (&t1, &t3), (&t3, &t3), (&tb, &t1)];
        for (pi, (ta, tb)) in pairs.iter().enumerate() {
            let mut data = vec![0u8; 4096];
            let mut s: u32 = 0x419_0000 ^ (pi as u32).wrapping_mul(2654435761);
            for byte in data.iter_mut() {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                *byte = (s >> 24) as u8;
            }
            let mut reader = crate::bitio::BitReader::new(&data);
            let total_bits = (data.len() as u64) * 8 + 64;
            let mut hits = 0usize;
            let mut misses = 0usize;
            while reader.cursor_bits() < total_bits {
                let w = reader.peek_window();
                let pair = decode_pair(ta, tb, w);
                // Sequential reference walk on the same reader state.
                let Ok((sa, la)) = decode_one(ta, w) else {
                    // No code matches at all (possible on the zero-pad
                    // tail for tables without an all-zeros code). A
                    // pair hit requires a primary-LUT hit on the same
                    // window, so the pair path must agree.
                    assert!(pair.is_none(), "pair {pi}: hit where decode_one fails");
                    break;
                };
                reader.consume_bits(la as u32).unwrap();
                let w2 = reader.peek_window();
                let Ok((sb, lb)) = decode_one(tb, w2) else {
                    break;
                };
                reader.consume_bits(lb as u32).unwrap();
                match pair {
                    Some((pa, pb, n)) => {
                        hits += 1;
                        assert_eq!(pa, sa, "pair sym A must match sequential (pair {pi})");
                        assert_eq!(pb, sb, "pair sym B must match sequential (pair {pi})");
                        assert_eq!(
                            n,
                            la as u32 + lb as u32,
                            "pair total length must match sequential (pair {pi})"
                        );
                    }
                    None => {
                        misses += 1;
                        // A miss is only legal when at least one of the
                        // two codes is a > 16-bit overflow code.
                        assert!(
                            la > 16 || lb > 16,
                            "decode_pair miss but both codes fit the primary LUT \
                             (pair {pi}: la={la}, lb={lb})"
                        );
                    }
                }
            }
            // The short-code table pairs must actually exercise the
            // fast path, and the set-B pair must exercise the fallback.
            assert!(hits > 0, "pair {pi}: fast path never hit");
            if pi == 3 {
                assert!(misses > 0, "set-B pair: overflow fallback never hit");
            }
        }
    }

    // ───────── Round-419: count-based package-merge equivalence ─────────

    /// Pre-r419 member-list package-merge, inlined verbatim as the
    /// reference oracle for the round-419 counts-only rewrite. Same
    /// early-outs, same sort key, same pairing, same tie-break
    /// (`packaged.weight <= leaf.weight` → package first), same
    /// `take = 2n - 2` selection — the only difference is that this
    /// body carries explicit per-package member lists and counts each
    /// symbol's occurrences among the selected top-tier packages.
    fn compute_canonical_lengths_member_reference(histogram: &[u32; 256]) -> Result<[u8; 256]> {
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
        active.sort_by_key(|&(s, f)| (f, s));
        #[derive(Clone)]
        struct Pkg {
            weight: u64,
            members: Vec<u16>,
        }
        let mut tier: Vec<Pkg> = active
            .iter()
            .map(|&(s, f)| Pkg {
                weight: f as u64,
                members: vec![s],
            })
            .collect();
        let leaves: Vec<Pkg> = tier.clone();
        for _ in 0..(MAX_L - 1) {
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
        let take = 2 * n - 2;
        if tier.len() < take {
            return Err(Error::invalid(
                "package-merge: insufficient packages — alphabet too small for max length",
            ));
        }
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
        let mut kraft: u64 = 0;
        for &l in lengths.iter() {
            if l > 0 {
                kraft = kraft.saturating_add(1u64 << (MAX_L as u8 - l));
            }
        }
        if kraft != 1u64 << (MAX_L as u8) {
            return Err(Error::invalid("package-merge: Kraft sum mismatch"));
        }
        Ok(lengths)
    }

    /// The round-419 counts-only body must produce byte-identical
    /// length tables to the pre-r419 member-list reference across a
    /// battery of histogram shapes: frame-realistic peaked, flat
    /// (balanced-tree worst case), sparse, heavy-outlier skew, ties
    /// everywhere (all-equal frequencies force every tie-break), a
    /// two-tier step distribution, geometric decay (deep codes /
    /// length-limit pressure), and 200 xorshift-random histograms with
    /// varying live-symbol counts.
    #[test]
    fn round419_count_based_matches_member_reference() {
        let mut cases: Vec<[u32; 256]> = Vec::new();
        // Peaked (geometric decay outward from 0 / 255, all live).
        let mut peaked = [0u32; 256];
        for (s, slot) in peaked.iter_mut().enumerate() {
            let d = s.min(256 - s) as u32;
            *slot = 1_000_000 / (1 + d * d);
        }
        cases.push(peaked);
        // Flat (all equal — maximal tie-breaking).
        cases.push([7u32; 256]);
        // Sparse (5 live).
        let mut sparse = [0u32; 256];
        for (i, s) in [3usize, 40, 41, 200, 255].into_iter().enumerate() {
            sparse[s] = (i as u32 + 1) * 13;
        }
        cases.push(sparse);
        // Heavy outlier.
        let mut skew = [1u32; 256];
        skew[0] = u32::MAX;
        cases.push(skew);
        // Two-tier step.
        let mut step = [0u32; 256];
        for (s, slot) in step.iter_mut().enumerate() {
            *slot = if s < 128 { 10 } else { 1000 };
        }
        cases.push(step);
        // Geometric doubling (length-limit pressure: unbounded Huffman
        // would exceed 31 bits well before 256 doublings saturate).
        let mut geo = [0u32; 256];
        let mut w = 1u32;
        for slot in geo.iter_mut() {
            *slot = w;
            w = w.saturating_mul(2);
        }
        cases.push(geo);
        // 200 xorshift-random histograms, varying live-symbol counts.
        let mut s: u32 = 0x419_419;
        for round in 0..200u32 {
            let mut h = [0u32; 256];
            let live = 3 + (round % 254) as usize;
            for slot in h.iter_mut().take(live) {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                *slot = 1 + (s % 100_000);
            }
            cases.push(h);
        }
        for (i, h) in cases.iter().enumerate() {
            let fast = compute_canonical_lengths(h).unwrap_or_else(|e| {
                panic!("case {i}: count-based body errored: {e:?}");
            });
            let reference = compute_canonical_lengths_member_reference(h).unwrap_or_else(|e| {
                panic!("case {i}: member reference errored: {e:?}");
            });
            assert_eq!(
                fast, reference,
                "case {i}: count-based lengths must match member-list reference"
            );
        }
    }

    /// The three-channel wrapper likewise emits no `0x00` — the whole
    /// extradata table region a `CustomV2` encoder writes is NUL-free.
    #[test]
    fn round382_rle_encode_three_channels_never_emits_inband_zero() {
        // Three deliberately-sparse slots (many absent-symbol runs).
        let mut s1 = [0u8; 256];
        let mut s2 = [0u8; 256];
        let mut s3 = [0u8; 256];
        for s in 0..256 {
            s1[s] = if s < 16 { 5 } else { 0 };
            s2[s] = if s % 3 == 0 { 6 } else { 0 };
            s3[s] = if (100..140).contains(&s) { 7 } else { 0 };
        }
        let out = rle_encode_three_channels(&[s1, s2, s3]).unwrap();
        assert!(
            !out.contains(&0x00),
            "three-channel extradata must be NUL-free (spec/04 §3.3); got {out:02x?}"
        );
        // And the region RLE-roundtrips to the exact three tables.
        let back = rle_decode_three_channels(&out).unwrap();
        assert_eq!(back, [s1, s2, s3], "three-channel RLE must roundtrip");
    }
}
