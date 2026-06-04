#![no_main]

//! Drive arbitrary fuzz-supplied bytes through the HuffYUV / FFVHuff
//! table-build primitives — the RLE length-table decoder, the
//! canonical-Huffman code builder, the package-merge length-limited
//! length computation, and the v1.x `(length, code-template)` table
//! constructor — directly, without going through the
//! `parse_bitmapinfoheader` / `decode_frame` / `encode_frame_with_mode`
//! whole-chain entry points the existing `decode_huffyuv.rs` and
//! `encode_huffyuv.rs` targets exercise.
//!
//! # What this adds over the existing fuzz targets
//!
//! The whole-chain targets only ever reach the table-build code through
//! a valid BIH whose extradata RLE-decompresses into per-channel length
//! tables that satisfy Kraft equality. Hostile bytes that decompress
//! into a structurally-wrong length table never reach
//! `HuffTable::build_from_lengths` on the decode-only target (the BIH
//! parse rejects malformed extradata first) and never reach
//! `compute_canonical_lengths` on the encode-only target (the histogram
//! it sees is always built from real residual bytes the encoder just
//! produced, so it lives in a narrow input space).
//!
//! This target slices the same fuzz buffer four different ways and
//! drives each table-build primitive directly:
//!
//! 1. **RLE decode** of arbitrary bytes (`rle_decode_one_channel` +
//!    `rle_decode_three_channels`). Whatever cursor / loop / count
//!    arithmetic the spec/01 §5 grammar permits must not panic on any
//!    input.
//! 2. **RLE encode → decode round-trip** on length tables sampled out
//!    of the fuzz buffer (`rle_encode_one_channel` → `rle_decode_one_channel`).
//!    A valid length table (every entry in `0..=31`) must round-trip
//!    byte-exact through the encoder/decoder pair.
//! 3. **Canonical Huffman build** (`HuffTable::build_from_lengths`).
//!    The build either errors (non-Kraft length distribution) or
//!    succeeds; if it succeeds, every per-symbol entry must
//!    self-consistent-decode via `decode_one` from its own
//!    MSB-aligned code window — that's the codec's downstream
//!    contract.
//! 4. **Package-merge canonical-length computation**
//!    (`compute_canonical_lengths`). The output length table — when
//!    the call returns `Ok` — must Kraft-equal and must build through
//!    `HuffTable::build_from_lengths` successfully, since that's the
//!    chain the encoder relies on.
//! 5. **v1.x table from (lengths, codes)** (`v1x_table_from_pair`).
//!    Same self-consistent-decode contract on every nonzero-length
//!    entry.
//!
//! # Input framing
//!
//! The fuzz buffer is split into four independent regions so a single
//! mutation can perturb any one driver:
//!
//! ```text
//! [0..=2)        u8 selector for "what to drive"
//! [2 ..= N)      driver-specific bytes (see per-driver code below)
//! ```
//!
//! The selector keeps the iteration cost per fuzz input low: one driver
//! per iter rather than running all five primitives back-to-back. The
//! libFuzzer mutator quickly learns which selector value reaches the
//! interesting code paths from corpus feedback alone.

use libfuzzer_sys::fuzz_target;
use oxideav_huffyuv::tables::{
    compute_canonical_lengths, decode_one, rle_decode_one_channel, rle_decode_three_channels,
    rle_encode_one_channel, v1x_table_from_pair, HuffTable,
};

/// Map an arbitrary `u8` into the legal length range `0..=31` so the
/// caller of `rle_encode_one_channel` / `compute_canonical_lengths` /
/// `HuffTable::build_from_lengths` is fed a value the spec/01 §5
/// grammar can represent. Modulo `32` is the simplest stable map; the
/// fuzzer's mutator works on the underlying byte so coverage spans the
/// full range.
fn coerce_length(b: u8) -> u8 {
    b & 0x1F
}

/// Build a 256-byte length table from `src`, coerced into `0..=31`,
/// zero-padded if `src` is shorter than 256 bytes.
fn build_lengths(src: &[u8]) -> [u8; 256] {
    let mut out = [0u8; 256];
    let take = src.len().min(256);
    for (i, &b) in src[..take].iter().enumerate() {
        out[i] = coerce_length(b);
    }
    out
}

/// Build a 256-entry histogram from `src` by mapping each pair of bytes
/// into a little-endian `u16` count for one symbol, zero-padded.
fn build_histogram(src: &[u8]) -> [u32; 256] {
    let mut out = [0u32; 256];
    for (i, chunk) in src.chunks(2).enumerate().take(256) {
        let lo = chunk.first().copied().unwrap_or(0) as u32;
        let hi = chunk.get(1).copied().unwrap_or(0) as u32;
        out[i] = (hi << 8) | lo;
    }
    out
}

/// Confirm every nonzero-length symbol in a freshly-built `HuffTable`
/// decodes via `decode_one` from its own MSB-aligned code window back
/// to itself. This is the codec's downstream contract: every code the
/// build produces must round-trip through the decoder. If a build
/// succeeds but some entry fails this check, that's a real
/// build-side bug.
fn assert_self_consistent_decode(table: &HuffTable) {
    for (sym, e) in table.entries.iter().enumerate() {
        if e.length == 0 {
            continue;
        }
        // `decode_one` reads the top `length` bits of the 32-bit
        // window. The entry's `code` is already MSB-aligned, so we can
        // pass it directly as the bit window.
        let (decoded_sym, decoded_len) =
            decode_one(table, e.code).expect("self-consistent decode must succeed");
        assert_eq!(
            decoded_sym as usize, sym,
            "decode_one returned symbol {} for code that maps to symbol {}",
            decoded_sym, sym
        );
        assert_eq!(
            decoded_len, e.length,
            "decode_one returned length {} for symbol {} (expected {})",
            decoded_len, sym, e.length
        );
    }
}

/// Driver 1: RLE-decode of arbitrary bytes. Must never panic / OOB /
/// overflow regardless of input. Length-table outputs that round-trip
/// through `rle_encode_one_channel` are an additional invariant.
fn drive_rle_decode(data: &[u8]) {
    let mut cursor: &[u8] = data;
    let result = rle_decode_one_channel(&mut cursor);
    if let Ok(table) = result {
        // Re-encode and re-decode for round-trip parity: a valid length
        // table (every entry in 0..=31, which the decoder guarantees by
        // construction since `value = byte & 0x1F`) must come back
        // byte-exact through the encoder/decoder pair.
        let mut re_encoded = Vec::new();
        rle_encode_one_channel(&table, &mut re_encoded)
            .expect("rle_encode of a decoder-produced table must succeed");
        let mut re_cursor: &[u8] = &re_encoded;
        let re_decoded = rle_decode_one_channel(&mut re_cursor)
            .expect("rle_decode of rle_encode output must succeed");
        assert_eq!(
            re_decoded, table,
            "RLE encode→decode must round-trip a valid length table byte-exact"
        );
    }

    // Three-channel variant: same liveness contract, no round-trip
    // (the encoder side appends bytes for each channel and the
    // decoder's three-channel wrapper consumes them in succession; the
    // round-trip is the same property covered above per-channel).
    let _ = rle_decode_three_channels(data);
}

/// Driver 2: RLE encode → decode round-trip on a fuzz-derived length
/// table. The encoder rejects entries > 31, so we coerce into
/// `0..=31` before driving.
fn drive_rle_encode_roundtrip(data: &[u8]) {
    let lengths = build_lengths(data);
    let mut out = Vec::new();
    rle_encode_one_channel(&lengths, &mut out)
        .expect("rle_encode of a coerced 0..=31 table must succeed");
    let mut cursor: &[u8] = &out;
    let decoded = rle_decode_one_channel(&mut cursor)
        .expect("rle_decode of rle_encode output must succeed");
    assert_eq!(
        decoded, lengths,
        "RLE encode→decode must round-trip every coerced 0..=31 length table"
    );
}

/// Driver 3: canonical-Huffman build on a fuzz-derived length table.
/// The call either errors (Kraft inequality) or succeeds with every
/// nonzero-length symbol self-consistently decoding back via
/// `decode_one`.
fn drive_build_from_lengths(data: &[u8]) {
    let lengths = build_lengths(data);
    if let Ok(table) = HuffTable::build_from_lengths(&lengths) {
        assert_self_consistent_decode(&table);
    }
}

/// Driver 4: package-merge canonical-length computation on a
/// fuzz-derived histogram. When the call succeeds, the returned length
/// table must build through `HuffTable::build_from_lengths` without
/// error (that's the documented chain the encoder relies on per
/// `encoder.rs`).
fn drive_compute_canonical_lengths(data: &[u8]) {
    let histogram = build_histogram(data);
    if let Ok(lengths) = compute_canonical_lengths(&histogram) {
        // All entries must be in 0..=31 by the function's contract.
        for (sym, &l) in lengths.iter().enumerate() {
            assert!(
                l <= 31,
                "compute_canonical_lengths returned length {} for symbol {} (must be 0..=31)",
                l,
                sym
            );
        }
        // The returned table must build successfully — that's the
        // encoder's documented downstream chain.
        let table = HuffTable::build_from_lengths(&lengths)
            .expect("compute_canonical_lengths output must build through HuffTable::build_from_lengths");
        assert_self_consistent_decode(&table);
    }
}

/// Driver 5: v1.x table from `(lengths, codes)` on a fuzz-derived
/// pair. When the build succeeds, every nonzero-length entry must
/// self-consistently decode. Note: `v1x_table_from_pair` does NOT
/// enforce Kraft equality (the v1.x precomputed-codes set is allowed
/// to be non-canonical per spec/04 §4.2), so the
/// self-consistent-decode property is the only contract worth
/// asserting.
fn drive_v1x_table_from_pair(data: &[u8]) {
    // First 256 bytes (or less, zero-padded) are lengths; next 256 are
    // codes; tail ignored.
    let mut lengths = [0u8; 256];
    let mut codes = [0u8; 256];
    let lens_take = data.len().min(256);
    for (i, &b) in data[..lens_take].iter().enumerate() {
        lengths[i] = coerce_length(b);
    }
    let codes_start = lens_take;
    let codes_take = data.len().saturating_sub(codes_start).min(256);
    if codes_take > 0 {
        codes[..codes_take]
            .copy_from_slice(&data[codes_start..codes_start + codes_take]);
    }
    if let Ok(table) = v1x_table_from_pair(&lengths, &codes) {
        // The v1.x build is allowed to produce a non-canonical table
        // (different per-symbol codes can share the same bit prefix in
        // a way Kraft equality would reject); the spec/04 §4.2 set B
        // is one such example. The self-consistent-decode property
        // therefore only holds for entries whose 32-bit MSB-aligned
        // code uniquely identifies them under the primary LUT +
        // overflow walk — i.e. the entries whose code window
        // round-trips via `decode_one` back to themselves.
        //
        // We assert the weaker liveness property here: `decode_one`
        // must return Ok for every nonzero-length entry's own code
        // window, even if the returned `(symbol, length)` pair is
        // some other symbol (because two v1.x entries collided on
        // their MSB-aligned prefix).
        for (sym, e) in table.entries.iter().enumerate() {
            if e.length == 0 {
                continue;
            }
            let _ = decode_one(&table, e.code).expect(
                "decode_one must return Ok for a v1.x entry's own code window",
            );
            let _ = sym;
        }
    }
}

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }
    let selector = data[0];
    let rest = &data[1..];
    match selector % 5 {
        0 => drive_rle_decode(rest),
        1 => drive_rle_encode_roundtrip(rest),
        2 => drive_build_from_lengths(rest),
        3 => drive_compute_canonical_lengths(rest),
        _ => drive_v1x_table_from_pair(rest),
    }
});
