//! HuffYUV / FFVHuff frame decoder.
//!
//! Consumes one AVI `00dc` chunk's worth of bytes plus the
//! `BITMAPINFOHEADER` (parsed by [`crate::header::StreamConfig`]) and
//! returns a [`DecodedFrame`] holding the per-pixel-family-shaped
//! reconstructed bytes.

use crate::bitio::BitReader;
use crate::error::{Error, Result};
use crate::header::{PixelFamily, Predictor, StreamConfig};
use crate::predict::{
    inverse_gradient_stride, inverse_left_row, inverse_median_yuy2, inverse_rgb_decorr_bgr,
    inverse_rgb_decorr_bgra, inverse_yuy2_left_macropixel,
};
use crate::tables::{
    HuffTable, classic_blob_bytes, decode_one, decode_pair, rle_decode_one_channel,
    rle_decode_three_channels, v1x_codes_set_a, v1x_codes_set_b, v1x_lengths_set_a,
    v1x_lengths_set_b, v1x_table_from_pair,
};

/// Reconstructed frame produced by [`decode_frame`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DecodedFrame {
    pub family: PixelFamily,
    pub width: u32,
    pub height: u32,
    /// Pixel bytes in the family's wire layout (YUY2: `Y₁ U Y₂ V` per
    /// 4-byte macropixel, top-down; RGB24: `B G R` per 3 bytes,
    /// bottom-up; RGB32: `B G R A` per 4 bytes, bottom-up).
    pub pixels: Vec<u8>,
}

/// Decode one HuffYUV frame.
///
/// `config` is the per-stream configuration parsed once at
/// `ICDecompressBegin` time; `frame_bytes` is the raw payload of one
/// AVI `00dc` chunk.
///
/// A field-coded stream ([`StreamConfig::is_interlaced`], spec/06 §1)
/// carries the same single first pixel and the same single codeword
/// stream as a plain one; the only difference is that the predictors'
/// "above" reference is two frame rows back (spec/06 §2.1).
pub fn decode_frame(config: &StreamConfig, frame_bytes: &[u8]) -> Result<DecodedFrame> {
    let tables = build_three_tables(config)?;
    let predictor = config.method.predictor();
    let decorrelate = config.method.decorrelate();
    // Prediction row length in bytes (spec/06 §2.1): doubled when the
    // stream is field-coded, the plain wire row otherwise.
    let stride = if config.is_interlaced() {
        config.row_bytes().saturating_mul(2)
    } else {
        config.row_bytes()
    };
    match config.family {
        PixelFamily::Yuy2 => decode_yuy2_raster(
            predictor,
            config.width,
            config.height,
            stride,
            frame_bytes,
            &tables,
        ),
        PixelFamily::Rgb24 => decode_rgb24_raster(
            predictor,
            decorrelate,
            config.width,
            config.height,
            stride,
            frame_bytes,
            &tables,
        ),
        PixelFamily::Rgb32 => decode_rgb32_raster(
            predictor,
            decorrelate,
            config.width,
            config.height,
            stride,
            frame_bytes,
            &tables,
        ),
    }
}

/// Decode one HuffYUV frame under an explicit worker budget.
///
/// `worker_budget` is the caller's advisory thread cap, in the sense of
/// the `oxideav-core` `ExecutionContext` threading contract. A HuffYUV
/// frame is one first pixel followed by one variable-length codeword
/// stream with no resynchronisation point and no per-field offset
/// (spec/02 §2, spec/06 §3.4), so the frame has a single unit of work:
/// every budget takes the serial path and the output is identical to
/// [`decode_frame`]. The parameter is kept so callers holding a budget
/// do not need a separate code path.
pub fn decode_frame_with_workers(
    config: &StreamConfig,
    frame_bytes: &[u8],
    worker_budget: usize,
) -> Result<DecodedFrame> {
    let _ = worker_budget;
    decode_frame(config, frame_bytes)
}

/// The three per-channel-slot Huffman tables (spec/03 §1.2 slot
/// architecture). Slot 1 = Y / B / B-G; Slot 2 = U / G / G; Slot 3 =
/// V / R / R-G (and reused for RGB32 alpha).
///
/// `pub(crate)` since round 419: the encoder's ClassicV2 / V1xCompat
/// paths build the exact same three tables from the exact same bytes,
/// so they share this struct and the [`table_cache`] below instead of
/// re-deriving (and re-cloning 128-KiB LUTs) per encoded frame.
pub(crate) struct ThreeTables {
    pub(crate) slot1: HuffTable,
    pub(crate) slot2: HuffTable,
    pub(crate) slot3: HuffTable,
    /// Round-429: lazily-built full pair LUTs for the serial decode
    /// loops (see [`PairSymLuts`]) — both symbols + combined length in
    /// one 32-bit load. `Default`-initialised empty; built on the
    /// first decode of the stream that walks the pair and reused for
    /// its lifetime via the round-419 table cache.
    pub(crate) pair_sym_luts: PairSymLuts,
}

/// Lazily-built decode-side full pair LUTs
/// ([`crate::tables::build_pair_sym_lut`]): the r419 profile pinned
/// ~95% of a LEFT-family decode in the per-symbol loop even after the
/// paired reads, with the serialized LUT-load → shift → LUT-load
/// chain of [`crate::tables::decode_pair`] on the critical path. Each
/// entry maps the top 16 window bits to BOTH symbols + the combined
/// length of the next two codewords in one 32-bit load (`0` =
/// unresolvable from 16 known bits → exact `decode_pair` /
/// `decode_one` fallback). 256 KiB per pair, built at most once per
/// cached `ThreeTables` (`OnceLock`) and only for the pairs the
/// stream's family actually decodes with.
///
/// Round-440 adds the three phase-flip combos the RGB24 two-pixel
/// step consumes (spec/03 §1.4 three-code wire cycle: pairing every
/// codeword of two consecutive pixels yields `(w0, w1)`, `(w2, w0′)`,
/// `(w1′, w2′)` — the +2 → slot3 position pairs across the pixel
/// boundary with the next pixel's first position):
///
/// - `s3s1` — RGB24 no-decorr `(R → slot3, next B → slot1)`.
/// - `s2s3` — RGB24 no-decorr `(next G → slot2, next R → slot3)`.
/// - `s3s2` — RGB24 decorr `(R−G → slot3, next G → slot2)`; the
///   decorr third pair `(next B−G → slot1, next R−G → slot3)` is the
///   pre-existing `s1s3`.
#[derive(Debug, Default)]
pub(crate) struct PairSymLuts {
    s1s2: std::sync::OnceLock<Box<[u32]>>,
    s1s3: std::sync::OnceLock<Box<[u32]>>,
    s2s1: std::sync::OnceLock<Box<[u32]>>,
    s3s3: std::sync::OnceLock<Box<[u32]>>,
    s3s1: std::sync::OnceLock<Box<[u32]>>,
    s2s3: std::sync::OnceLock<Box<[u32]>>,
    s3s2: std::sync::OnceLock<Box<[u32]>>,
}

impl ThreeTables {
    fn pair_sym_lut_s1s2(&self) -> &[u32] {
        self.pair_sym_luts
            .s1s2
            .get_or_init(|| crate::tables::build_pair_sym_lut(&self.slot1, &self.slot2))
    }
    fn pair_sym_lut_s1s3(&self) -> &[u32] {
        self.pair_sym_luts
            .s1s3
            .get_or_init(|| crate::tables::build_pair_sym_lut(&self.slot1, &self.slot3))
    }
    fn pair_sym_lut_s2s1(&self) -> &[u32] {
        self.pair_sym_luts
            .s2s1
            .get_or_init(|| crate::tables::build_pair_sym_lut(&self.slot2, &self.slot1))
    }
    fn pair_sym_lut_s3s3(&self) -> &[u32] {
        self.pair_sym_luts
            .s3s3
            .get_or_init(|| crate::tables::build_pair_sym_lut(&self.slot3, &self.slot3))
    }
    fn pair_sym_lut_s3s1(&self) -> &[u32] {
        self.pair_sym_luts
            .s3s1
            .get_or_init(|| crate::tables::build_pair_sym_lut(&self.slot3, &self.slot1))
    }
    fn pair_sym_lut_s2s3(&self) -> &[u32] {
        self.pair_sym_luts
            .s2s3
            .get_or_init(|| crate::tables::build_pair_sym_lut(&self.slot2, &self.slot3))
    }
    fn pair_sym_lut_s3s2(&self) -> &[u32] {
        self.pair_sym_luts
            .s3s2
            .get_or_init(|| crate::tables::build_pair_sym_lut(&self.slot3, &self.slot2))
    }
}

/// Round-419 decode-table cache.
///
/// `decode_frame` is stateless by design (the public contract takes a
/// `&StreamConfig` per call), which meant every frame re-derived its
/// three `HuffTable`s from scratch: three `build_from_lengths` calls
/// (canonical code assign + 64 Ki-entry primary-LUT fill + overflow
/// bake, ~25 µs each) on the v2.x path, or two `v1x_table_from_pair`
/// builds plus 128-KiB LUT `clone`s on the v1.x path — ~12% of the
/// whole 320×240 frame decode, paid again for every frame of the same
/// stream even though the tables depend only on bytes that are fixed
/// at stream-open time.
///
/// The tables are a pure function of their inputs, so the cache is
/// output-invariant by construction:
///
/// - **v2.x** (extradata present): keyed by the exact RLE table bytes
///   (the stream's own extradata region, or the classic blob the
///   empty-region fallback substitutes). A bounded `HashMap` holds the
///   built tables behind `Arc`; when a hostile/fuzz workload cycles
///   more than [`EXTRADATA_CACHE_CAP`] distinct table blobs the map is
///   cleared wholesale (deterministic, keeps worst-case memory at
///   `CAP × ~390 KiB`).
/// - **v1.x** (no extradata): the two codebook sets are compiled-in
///   constants, so the two family-shaped `ThreeTables` live in
///   `OnceLock`s — built exactly once per process.
///
/// Both paths fall back to an uncached build on the (unreachable in
/// practice) error branches so error semantics stay identical.
///
/// Shared with the encoder (round 419): `encode_with_precomputed`'s
/// ClassicV2 arm keys the same map by the same classic-blob bytes it
/// embeds as extradata, and its V1xCompat arm uses the same per-family
/// `OnceLock`s — so an encode→decode round trip builds each table set
/// once, total.
pub(crate) mod table_cache {
    use super::*;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};

    /// Max distinct extradata blobs held before the map is reset.
    const EXTRADATA_CACHE_CAP: usize = 16;

    fn extradata_cache() -> &'static Mutex<HashMap<Vec<u8>, Arc<ThreeTables>>> {
        static CACHE: OnceLock<Mutex<HashMap<Vec<u8>, Arc<ThreeTables>>>> = OnceLock::new();
        CACHE.get_or_init(|| Mutex::new(HashMap::new()))
    }

    /// v2.x: build-or-fetch the tables for one RLE table-bytes blob.
    pub(crate) fn extradata_tables(table_bytes: &[u8]) -> Result<Arc<ThreeTables>> {
        if let Ok(map) = extradata_cache().lock()
            && let Some(hit) = map.get(table_bytes)
        {
            return Ok(Arc::clone(hit));
        }
        let lengths = rle_decode_three_channels(table_bytes)?;
        let built = Arc::new(ThreeTables {
            slot1: HuffTable::build_from_lengths(&lengths[0])?,
            slot2: HuffTable::build_from_lengths(&lengths[1])?,
            slot3: HuffTable::build_from_lengths(&lengths[2])?,
            pair_sym_luts: PairSymLuts::default(),
        });
        if let Ok(mut map) = extradata_cache().lock() {
            if map.len() >= EXTRADATA_CACHE_CAP {
                map.clear();
            }
            map.insert(table_bytes.to_vec(), Arc::clone(&built));
        }
        Ok(built)
    }

    /// v1.x: family-shaped compiled-in tables, built once per process.
    pub(crate) fn v1x_tables(family: PixelFamily) -> Result<Arc<ThreeTables>> {
        static YUV: OnceLock<Option<Arc<ThreeTables>>> = OnceLock::new();
        static RGB: OnceLock<Option<Arc<ThreeTables>>> = OnceLock::new();
        let (cell, is_rgb) = match family {
            PixelFamily::Yuy2 => (&YUV, false),
            PixelFamily::Rgb24 | PixelFamily::Rgb32 => (&RGB, true),
        };
        if let Some(t) = cell.get_or_init(|| build_v1x(is_rgb).ok().map(Arc::new)) {
            return Ok(Arc::clone(t));
        }
        // Unreachable for the compiled-in blobs (covered by tests);
        // re-run the build so the caller sees the original error.
        build_v1x(is_rgb).map(Arc::new)?;
        unreachable!("v1x table build cannot succeed here after failing above");
    }

    fn build_v1x(is_rgb: bool) -> Result<ThreeTables> {
        let mut cursor: &[u8] = v1x_lengths_set_a();
        let lens_a = rle_decode_one_channel(&mut cursor)?;
        let mut codes_a = [0u8; 256];
        codes_a.copy_from_slice(v1x_codes_set_a());
        let table_a = v1x_table_from_pair(&lens_a, &codes_a)?;
        if is_rgb {
            // v1.x RGB: all three of B, G, R use set A.
            return Ok(ThreeTables {
                slot1: table_a.clone(),
                slot2: table_a.clone(),
                slot3: table_a,
                pair_sym_luts: PairSymLuts::default(),
            });
        }
        let mut cursor: &[u8] = v1x_lengths_set_b();
        let lens_b = rle_decode_one_channel(&mut cursor)?;
        let mut codes_b = [0u8; 256];
        codes_b.copy_from_slice(v1x_codes_set_b());
        let table_b = v1x_table_from_pair(&lens_b, &codes_b)?;
        // v1.x YUY2: Y uses set A; both U and V use set B.
        Ok(ThreeTables {
            slot1: table_a,
            slot2: table_b.clone(),
            slot3: table_b,
            pair_sym_luts: PairSymLuts::default(),
        })
    }
}

fn build_three_tables(config: &StreamConfig) -> Result<std::sync::Arc<ThreeTables>> {
    if config.has_extradata {
        // v2.x extradata path: 3 RLE-compressed length tables in
        // slot-1, slot-2, slot-3 order, then build canonical Huffman.
        let table_bytes: &[u8] = if !config.extradata_tables.is_empty() {
            &config.extradata_tables
        } else {
            // Fallback: extradata-present flag set but the table
            // region is empty — shouldn't happen on the wire but we
            // accept it by falling through to the classic blob for
            // this (family, method).
            classic_blob_bytes(config.family, config.method)
        };
        table_cache::extradata_tables(table_bytes)
    } else {
        // v1.x precomputed-codes path (spec/04 §4): the per-channel
        // sharing depends on the family.
        table_cache::v1x_tables(config.family)
    }
}

/// A frame codes `codewords` residuals after its 4-byte first pixel,
/// each with a codeword of at least one bit (spec/03 §3.1), so a
/// frame shorter than that cannot be complete. Checked before the
/// output raster is allocated: a header declaring a huge picture in
/// front of a tiny chunk is rejected without the allocation.
fn ensure_body_can_hold(frame_bytes: &[u8], codewords: usize) -> Result<()> {
    let body_bits = (frame_bytes.len().saturating_sub(4) as u64) * 8;
    if body_bits < codewords as u64 {
        return Err(Error::invalid(format!(
            "frame truncated: {} bytes cannot hold {codewords} codewords",
            frame_bytes.len()
        )));
    }
    Ok(())
}

/// The decoder stops by pixel count (spec/02 §1); every codeword it
/// read must lie inside the chunk. Reading past the end means the
/// chunk was cut short (or does not match the stream header).
fn ensure_fully_read(reader: &BitReader<'_>, bit_data: &[u8]) -> Result<()> {
    let have = (bit_data.len() as u64) * 8;
    if reader.cursor_bits() > have {
        return Err(Error::invalid(format!(
            "frame truncated: the picture needs {} codeword bits, the chunk holds {have}",
            reader.cursor_bits()
        )));
    }
    Ok(())
}

fn decode_yuy2_raster(
    predictor: Predictor,
    width: u32,
    height: u32,
    stride: usize,
    frame_bytes: &[u8],
    tables: &ThreeTables,
) -> Result<DecodedFrame> {
    let width_us = width as usize;
    let height_us = height as usize;
    if !width_us.is_multiple_of(2) {
        return Err(Error::invalid("YUY2 width must be even"));
    }
    // Round-262: family → wire-stride via the round-261 accessor
    // (spec/02 §3 wire-byte layout table: Y₁ U Y₂ V per 2 px = 4
    // bytes per macropixel = width × 2).
    let row_bytes = PixelFamily::Yuy2.row_bytes(width);
    let total_bytes = row_bytes * height_us;
    // A degenerate frame (zero width or height) leaves the output
    // raster too small to even hold the 4-byte uncompressed seed pixel
    // written below; reject it before the slice write would panic.
    if total_bytes < 4 {
        return Err(Error::invalid(
            "YUY2 frame: degenerate dimensions (need ≥ 1 macropixel)",
        ));
    }
    if frame_bytes.len() < 4 {
        return Err(Error::invalid(
            "YUY2 frame: missing 4-byte uncompressed pixel",
        ));
    }
    ensure_body_can_hold(frame_bytes, total_bytes - 4)?;
    let mut pixels = vec![0u8; total_bytes];
    pixels[..4].copy_from_slice(&frame_bytes[..4]);

    // Bit reader starts at the first byte after the uncompressed
    // first pixel. spec/02 §4: 32-bit-LE-word framing of the
    // codeword stream. The first word is at frame_bytes[4..8].
    let bit_data = &frame_bytes[4..];
    let mut reader = BitReader::new(bit_data);

    // Round-214: macropixel-step Huffman decode. spec/03 §1.2's three-slot
    // architecture pins the YUY2 wire-byte → slot mapping at a fixed
    // 4-byte cycle:
    //
    //   +0 (Y₁) → slot1   +1 (U) → slot2   +2 (Y₂) → slot1   +3 (V) → slot3
    //
    // The pre-r214 loop ran `match byte_idx % 4 { … }` on every output
    // byte to pick the slot — a per-byte branch the optimiser couldn't
    // eliminate because `byte_idx` was the loop induction variable.
    // The decode-side analogue of round 181's LEFT macropixel-step
    // rewrite: pin the byte_idx wire-stride at the source by stepping
    // four output bytes per outer iteration, each with its slot
    // resolved at compile time. The inner body becomes a fixed
    // straight-line 4-decode / 4-store sequence (Y₁ via slot1, U via
    // slot2, Y₂ via slot1, V via slot3) and the compiler is free to
    // schedule the four Huffman lookups, four `consume_bits`, and
    // four indexed stores without a per-byte slot-pointer reload.
    //
    // `total_bytes - 4` is always a multiple of 4 for YUY2 (`row_bytes =
    // width × 2`, width is even per the §2.1.1 macropixel-pair
    // invariant we checked above), so the macropixel body covers every
    // remaining output byte; no scalar tail is needed in the in-spec
    // input space. We keep a 1..=3-byte scalar fall-through for
    // robustness (e.g. malformed inputs where the seed write still
    // succeeded but `total_bytes` lands non-aligned in a future
    // pixel-family extension).
    // Round-419: paired symbol reads. Each macropixel's four codewords
    // are consumed as two `decode_pair` window reads — (Y₁, U) then
    // (Y₂, V) — advancing the bit reader once per pair instead of once
    // per symbol. `decode_pair` serves both symbols from the same
    // 32-bit window (l1 ≤ 16 keeps the second lookup's 16 bits inside
    // it), falling back to the sequential per-symbol path when either
    // code is longer than 16 bits. Output is byte-identical (see
    // `tables::decode_pair` docs + the round419 pair/sequential
    // equivalence test in `tables.rs`).
    // Round-429: full pair LUTs on the serial critical path. The r419
    // profile still pinned ~95% of a LEFT-family decode in this loop,
    // with `decode_pair`'s serialized load-A → shift → load-B chain as
    // the dependency spine. Each macropixel half now tries ONE 32-bit
    // load (top 16 window bits → both symbols + combined length,
    // `tables::build_pair_sym_lut` soundness note); `0` falls back to
    // the exact `decode_pair` / `decode_one` walk, so long codes and
    // 16-bit-straddling pairs are served unchanged. Byte-identical
    // output (r419 golden matrix + randomized LUT/pair equivalence
    // sweep below).
    let lut_y1u = tables.pair_sym_lut_s1s2();
    let lut_y2v = tables.pair_sym_lut_s1s3();
    debug_assert!(total_bytes >= 4);
    let body_end = total_bytes - ((total_bytes - 4) & 3);
    let mut byte_idx = 4usize;
    while byte_idx < body_end {
        // (Y₁ → slot1, U → slot2)
        let w = reader.peek_window();
        let e = lut_y1u[(w >> 16) as usize];
        if e != 0 {
            reader.consume_bits_trusted(e >> 16);
            pixels[byte_idx] = e as u8;
            pixels[byte_idx + 1] = (e >> 8) as u8;
        } else if let Some((sym_y1, sym_u, n)) = decode_pair(&tables.slot1, &tables.slot2, w) {
            reader.consume_bits_trusted(n);
            pixels[byte_idx] = sym_y1;
            pixels[byte_idx + 1] = sym_u;
        } else {
            let (sym_y1, len_y1) = decode_one(&tables.slot1, w)?;
            reader.consume_bits_trusted(len_y1 as u32);
            pixels[byte_idx] = sym_y1;
            let (sym_u, len_u) = decode_one(&tables.slot2, reader.peek_window())?;
            reader.consume_bits_trusted(len_u as u32);
            pixels[byte_idx + 1] = sym_u;
        }
        // (Y₂ → slot1, V → slot3)
        let w = reader.peek_window();
        let e = lut_y2v[(w >> 16) as usize];
        if e != 0 {
            reader.consume_bits_trusted(e >> 16);
            pixels[byte_idx + 2] = e as u8;
            pixels[byte_idx + 3] = (e >> 8) as u8;
        } else if let Some((sym_y2, sym_v, n)) = decode_pair(&tables.slot1, &tables.slot3, w) {
            reader.consume_bits_trusted(n);
            pixels[byte_idx + 2] = sym_y2;
            pixels[byte_idx + 3] = sym_v;
        } else {
            let (sym_y2, len_y2) = decode_one(&tables.slot1, w)?;
            reader.consume_bits_trusted(len_y2 as u32);
            pixels[byte_idx + 2] = sym_y2;
            let (sym_v, len_v) = decode_one(&tables.slot3, reader.peek_window())?;
            reader.consume_bits_trusted(len_v as u32);
            pixels[byte_idx + 3] = sym_v;
        }
        byte_idx += 4;
    }
    // Scalar fall-through for any 1..=3 trailing bytes (unreachable
    // for valid YUY2 inputs; kept for robustness against unforeseen
    // future layouts).
    while byte_idx < total_bytes {
        let slot = match byte_idx % 4 {
            0 | 2 => &tables.slot1,
            1 => &tables.slot2,
            _ => &tables.slot3,
        };
        let (sym, len) = decode_one(slot, reader.peek_window())?;
        reader.consume_bits_trusted(len as u32);
        pixels[byte_idx] = sym;
        byte_idx += 1;
    }

    ensure_fully_read(&reader, bit_data)?;

    // Apply the YUY2 predictor inverse. spec/03 §2.1.1 / §2.2.2 /
    // §2.3.2.
    //
    // - LEFT: walk the per-byte-position-strided LEFT pass over
    //   bytes 4..end of the linear raster.
    // - GRADIENT: same as LEFT then per-row paddb (rows 1..H).
    // - MEDIAN: LEFT pass over row 0 + the first 8 wire bytes of
    //   row 1; median per-byte add elsewhere.
    match predictor {
        Predictor::Left => {
            inverse_yuy2_left_macropixel(&mut pixels, 4, total_bytes);
        }
        Predictor::Gradient => {
            inverse_yuy2_left_macropixel(&mut pixels, 4, total_bytes);
            inverse_gradient_stride(&mut pixels, stride);
        }
        Predictor::Median => {
            inverse_median_yuy2(&mut pixels, stride);
        }
    }

    Ok(DecodedFrame {
        family: PixelFamily::Yuy2,
        width,
        height,
        pixels,
    })
}

fn decode_rgb24_raster(
    predictor: Predictor,
    decorrelate: bool,
    width: u32,
    height: u32,
    stride: usize,
    frame_bytes: &[u8],
    tables: &ThreeTables,
) -> Result<DecodedFrame> {
    let width_us = width as usize;
    let height_us = height as usize;
    // Round-262: family → wire-stride via the round-261 accessor
    // (spec/02 §3 wire-byte layout table: 3 bytes per pixel,
    // `+0:B +1:G +2:R`).
    let row_bytes = PixelFamily::Rgb24.row_bytes(width);
    let total_bytes = row_bytes * height_us;
    // A degenerate frame (zero width or height) leaves the output
    // raster too small to hold the first 3-byte BGR pixel written
    // below; reject it before the indexed writes would panic.
    if total_bytes < 3 {
        return Err(Error::invalid(
            "RGB24 frame: degenerate dimensions (need ≥ 1 pixel)",
        ));
    }
    if frame_bytes.len() < 4 {
        return Err(Error::invalid(
            "RGB24 frame: missing 4-byte uncompressed pixel",
        ));
    }
    ensure_body_can_hold(frame_bytes, total_bytes - 3)?;
    let mut pixels = vec![0u8; total_bytes];
    // First uncompressed pixel: stored as `00 B G R` (4 wire bytes,
    // X = 0). The reconstructed buffer holds 3-byte BGR, so write
    // bytes 1..4 of the first 4-byte cell into the first 3 output
    // bytes (spec/03 §1.1 + §1.4: encoder writes pad as 0, decoder
    // discards).
    pixels[0] = frame_bytes[1]; // B
    pixels[1] = frame_bytes[2]; // G
    pixels[2] = frame_bytes[3]; // R

    let bit_data = &frame_bytes[4..];
    let mut reader = BitReader::new(bit_data);

    // Round-277: wire-position (slot, store-offset) binding hoist.
    // The pre-r277 loop re-evaluated the loop-invariant `decorrelate`
    // branch on every pixel and re-resolved the three slot pointers +
    // store offsets inside each arm — a per-pixel branch the optimiser
    // couldn't fold away on the critical path of every `decode_one`.
    // spec/03 §1.4 pins the per-pixel wire codeword order at
    // `B, G, R` (no decorr) / `G, B−G, R−G` (decorr), and §1.2 fixes
    // the table assignment per wire position at (slot1, slot2, slot3)
    // / (slot2, slot1, slot3) respectively. The reconstructed buffer
    // always stores 3-byte BGR, so each wire position also carries a
    // fixed store offset:
    //   - no decorr: +0 (B → slot1), +1 (G → slot2), +2 (R → slot3)
    //   - decorr:    +1 (G → slot2), +0 (B−G → slot1), +2 (R−G → slot3)
    // (decorrelated values are stored in BGR layout — B = B−G,
    // G, R = R−G — and the decorrelation inverse runs after the
    // predictor pass below.) Resolve both bindings once at entry;
    // the loop body becomes a fixed straight-line three-decode /
    // three-store sequence with no per-pixel branch — the decode-side
    // analogue of the r239 encoder emit-loop binding (same §1.4
    // three-code RGB24 wire cycle, same hoist shape), completing the
    // r214 (YUY2 decode) / r239 (RGB24 emit) / r245 (RGB32 emit)
    // series on the RGB24 decode loop.
    let (s_w0, o_w0, s_w1, o_w1) = if decorrelate {
        (&tables.slot2, 1usize, &tables.slot1, 0usize)
    } else {
        (&tables.slot1, 0usize, &tables.slot2, 1usize)
    };
    // Wire position +2 (R or R−G) → slot3, store +2 in both modes.
    // Round-419: paired symbol reads — the first two wire positions
    // (mode-resolved above) come out of one `decode_pair` window read;
    // the +2 position stays a single read. See the YUY2 loop note.
    // Round-429: the pair read is fronted by the full pair LUT — one
    // 32-bit load resolves both symbols + the combined length when
    // the pair fits in the 16 known window bits; `0` falls back to
    // the exact `decode_pair` / `decode_one` walk. See the YUY2 loop
    // note.
    let lut_w01 = if decorrelate {
        tables.pair_sym_lut_s2s1()
    } else {
        tables.pair_sym_lut_s1s2()
    };
    // Round-440 phase-flip pairing: the 3-code RGB24 wire cycle left
    // the +2 → slot3 position on a lone `decode_one` (the r429 note's
    // remaining decode lever). Walking TWO pixels per outer iteration
    // puts every codeword on a pair read — the six codewords of a
    // pixel pair group as `(w0, w1)`, `(w2, w0′)`, `(w1′, w2′)`, where
    // the middle pair crosses the pixel boundary (spec/03 §1.4 wire
    // order `B, G, R` / `G, B−G, R−G`; §1.2 slot mapping). Each pair
    // is fronted by its full pair LUT with the exact `decode_pair` /
    // `decode_one` walk as fallback, so symbol order, consumed bits,
    // and error points are identical to the per-pixel body (kept below
    // as the odd-count tail). Byte-identical output — r419 golden
    // matrix + the round440 per-symbol reference witness.
    let (lut_2w0, lut_w12) = if decorrelate {
        (tables.pair_sym_lut_s3s2(), tables.pair_sym_lut_s1s3())
    } else {
        (tables.pair_sym_lut_s3s1(), tables.pair_sym_lut_s2s3())
    };
    let n_pixels = width_us * height_us;
    let mut px = 1usize;
    while px + 1 < n_pixels {
        let off_a = px * 3;
        let off_b = off_a + 3;
        // Pair 1: (w0, w1) of pixel A — mode-resolved slots.
        let w = reader.peek_window();
        let e = lut_w01[(w >> 16) as usize];
        if e != 0 {
            reader.consume_bits_trusted(e >> 16);
            pixels[off_a + o_w0] = e as u8;
            pixels[off_a + o_w1] = (e >> 8) as u8;
        } else if let Some((v0, v1, n)) = decode_pair(s_w0, s_w1, w) {
            reader.consume_bits_trusted(n);
            pixels[off_a + o_w0] = v0;
            pixels[off_a + o_w1] = v1;
        } else {
            let (v0, l0) = decode_one(s_w0, w)?;
            reader.consume_bits_trusted(l0 as u32);
            pixels[off_a + o_w0] = v0;
            let (v1, l1) = decode_one(s_w1, reader.peek_window())?;
            reader.consume_bits_trusted(l1 as u32);
            pixels[off_a + o_w1] = v1;
        }
        // Pair 2 (phase flip): (w2 of A → slot3, w0 of B).
        let w = reader.peek_window();
        let e = lut_2w0[(w >> 16) as usize];
        if e != 0 {
            reader.consume_bits_trusted(e >> 16);
            pixels[off_a + 2] = e as u8;
            pixels[off_b + o_w0] = (e >> 8) as u8;
        } else if let Some((v2, v0b, n)) = decode_pair(&tables.slot3, s_w0, w) {
            reader.consume_bits_trusted(n);
            pixels[off_a + 2] = v2;
            pixels[off_b + o_w0] = v0b;
        } else {
            let (v2, l2) = decode_one(&tables.slot3, w)?;
            reader.consume_bits_trusted(l2 as u32);
            pixels[off_a + 2] = v2;
            let (v0b, l0b) = decode_one(s_w0, reader.peek_window())?;
            reader.consume_bits_trusted(l0b as u32);
            pixels[off_b + o_w0] = v0b;
        }
        // Pair 3: (w1 of B, w2 of B → slot3).
        let w = reader.peek_window();
        let e = lut_w12[(w >> 16) as usize];
        if e != 0 {
            reader.consume_bits_trusted(e >> 16);
            pixels[off_b + o_w1] = e as u8;
            pixels[off_b + 2] = (e >> 8) as u8;
        } else if let Some((v1b, v2b, n)) = decode_pair(s_w1, &tables.slot3, w) {
            reader.consume_bits_trusted(n);
            pixels[off_b + o_w1] = v1b;
            pixels[off_b + 2] = v2b;
        } else {
            let (v1b, l1b) = decode_one(s_w1, w)?;
            reader.consume_bits_trusted(l1b as u32);
            pixels[off_b + o_w1] = v1b;
            let (v2b, l2b) = decode_one(&tables.slot3, reader.peek_window())?;
            reader.consume_bits_trusted(l2b as u32);
            pixels[off_b + 2] = v2b;
        }
        px += 2;
    }
    // Odd body-pixel count: the last pixel keeps the pre-r440
    // per-pixel body (pair (w0, w1) + lone slot3 read).
    if px < n_pixels {
        let bgr_off = px * 3;
        let w = reader.peek_window();
        let e = lut_w01[(w >> 16) as usize];
        if e != 0 {
            reader.consume_bits_trusted(e >> 16);
            pixels[bgr_off + o_w0] = e as u8;
            pixels[bgr_off + o_w1] = (e >> 8) as u8;
        } else if let Some((v0, v1, n)) = decode_pair(s_w0, s_w1, w) {
            reader.consume_bits_trusted(n);
            pixels[bgr_off + o_w0] = v0;
            pixels[bgr_off + o_w1] = v1;
        } else {
            let (v0, l0) = decode_one(s_w0, w)?;
            reader.consume_bits_trusted(l0 as u32);
            pixels[bgr_off + o_w0] = v0;
            let (v1, l1) = decode_one(s_w1, reader.peek_window())?;
            reader.consume_bits_trusted(l1 as u32);
            pixels[bgr_off + o_w1] = v1;
        }
        let (v2, l2) = decode_one(&tables.slot3, reader.peek_window())?;
        reader.consume_bits_trusted(l2 as u32);
        pixels[bgr_off + 2] = v2;
    }

    ensure_fully_read(&reader, bit_data)?;
    if predictor == Predictor::Median {
        return Err(Error::invalid("median predictor not legal for RGB"));
    }
    if decorrelate {
        // The first pixel is stored as the plain input bytes (spec/03
        // §1.1 / §1.3, spec/02 §8 #6) while every residual is taken on
        // the decorrelated channels (spec/03 §2.4.1), so move the first
        // pixel into the decorrelated domain before chaining from it.
        let g0 = pixels[1];
        pixels[0] = pixels[0].wrapping_sub(g0);
        pixels[2] = pixels[2].wrapping_sub(g0);
    }
    // Per-channel LEFT inverse over the interleaved raster, then the
    // gradient post-pass (spec/03 §2.2.2) with the prediction row
    // length of spec/06 §2.1.
    inverse_left_row(&mut pixels, 3);
    if predictor == Predictor::Gradient {
        inverse_gradient_stride(&mut pixels, stride);
    }
    if decorrelate {
        // Inverse decorrelation: B = (B−G) + G, R = (R−G) + G.
        inverse_rgb_decorr_bgr(&mut pixels);
    }

    Ok(DecodedFrame {
        family: PixelFamily::Rgb24,
        width,
        height,
        pixels,
    })
}

fn decode_rgb32_raster(
    predictor: Predictor,
    decorrelate: bool,
    width: u32,
    height: u32,
    stride: usize,
    frame_bytes: &[u8],
    tables: &ThreeTables,
) -> Result<DecodedFrame> {
    let width_us = width as usize;
    let height_us = height as usize;
    // Round-262: family → wire-stride via the round-261 accessor
    // (spec/02 §3 wire-byte layout table: 4 bytes per pixel,
    // `+0:B +1:G +2:R +3:A`).
    let row_bytes = PixelFamily::Rgb32.row_bytes(width);
    let total_bytes = row_bytes * height_us;
    // A degenerate frame (zero width or height) leaves the output
    // raster too small to hold the first 4-byte BGRA pixel written
    // below; reject it before the slice write would panic.
    if total_bytes < 4 {
        return Err(Error::invalid(
            "RGB32 frame: degenerate dimensions (need ≥ 1 pixel)",
        ));
    }
    if frame_bytes.len() < 4 {
        return Err(Error::invalid(
            "RGB32 frame: missing 4-byte uncompressed pixel",
        ));
    }
    ensure_body_can_hold(frame_bytes, total_bytes - 4)?;
    let mut pixels = vec![0u8; total_bytes];
    // First pixel verbatim (B G R A). spec/03 §1.3: alpha is real
    // data, no pad-byte zeroing.
    pixels[..4].copy_from_slice(&frame_bytes[..4]);

    let bit_data = &frame_bytes[4..];
    let mut reader = BitReader::new(bit_data);
    let n_pixels = width_us * height_us;

    // Round-277: wire-position (slot, store-offset) binding hoist —
    // RGB32 companion to the RGB24 rewrite above. spec/03 §1.4 pins
    // the per-pixel wire codeword order at `B, G, R, A` (no decorr) /
    // `G, B−G, R−G, A` (decorr); §1.2 fixes the table assignment at
    // (slot1, slot2, slot3, slot3) / (slot2, slot1, slot3, slot3)
    // respectively — alpha shares the slot-3 codebook in both modes
    // and is never decorrelated (spec/03 §2.4 Validator note), so the
    // +2 and +3 wire positions are mode-independent and only the
    // first two positions need the binding:
    //   - no decorr: +0 (B → slot1), +1 (G → slot2)
    //   - decorr:    +1 (G → slot2), +0 (B−G → slot1)
    // The loop body becomes a fixed straight-line four-decode /
    // four-store sequence with no per-pixel branch — the decode-side
    // analogue of the r245 encoder emit-loop binding (same §1.4
    // four-code RGB32 wire cycle, same hoist shape).
    let (s_w0, o_w0, s_w1, o_w1) = if decorrelate {
        (&tables.slot2, 1usize, &tables.slot1, 0usize)
    } else {
        (&tables.slot1, 0usize, &tables.slot2, 1usize)
    };
    // Round-419: paired symbol reads — (w0, w1) then (R/R−G, A), each
    // pair off one `decode_pair` window read. See the YUY2 loop note.
    // Round-429: full pair LUTs front both pair reads — one 32-bit
    // load resolves both symbols + the combined length when the pair
    // fits in the 16 known window bits; `0` falls back to the exact
    // `decode_pair` / `decode_one` walk. See the YUY2 loop note.
    let lut_w01 = if decorrelate {
        tables.pair_sym_lut_s2s1()
    } else {
        tables.pair_sym_lut_s1s2()
    };
    let lut_ra = tables.pair_sym_lut_s3s3();
    for px in 1..n_pixels {
        let off = px * 4;
        let w = reader.peek_window();
        let e = lut_w01[(w >> 16) as usize];
        if e != 0 {
            reader.consume_bits_trusted(e >> 16);
            pixels[off + o_w0] = e as u8;
            pixels[off + o_w1] = (e >> 8) as u8;
        } else if let Some((v0, v1, n)) = decode_pair(s_w0, s_w1, w) {
            reader.consume_bits_trusted(n);
            pixels[off + o_w0] = v0;
            pixels[off + o_w1] = v1;
        } else {
            let (v0, l0) = decode_one(s_w0, w)?;
            reader.consume_bits_trusted(l0 as u32);
            pixels[off + o_w0] = v0;
            let (v1, l1) = decode_one(s_w1, reader.peek_window())?;
            reader.consume_bits_trusted(l1 as u32);
            pixels[off + o_w1] = v1;
        }
        // Wire positions +2 (R or R−G) and +3 (A) both use the slot-3
        // codebook (spec/03 §1.2; alpha shares slot 3 in both modes).
        let w = reader.peek_window();
        let e = lut_ra[(w >> 16) as usize];
        if e != 0 {
            reader.consume_bits_trusted(e >> 16);
            pixels[off + 2] = e as u8;
            pixels[off + 3] = (e >> 8) as u8;
        } else if let Some((v2, v3, n)) = decode_pair(&tables.slot3, &tables.slot3, w) {
            reader.consume_bits_trusted(n);
            pixels[off + 2] = v2;
            pixels[off + 3] = v3;
        } else {
            let (v2, l2) = decode_one(&tables.slot3, w)?;
            reader.consume_bits_trusted(l2 as u32);
            pixels[off + 2] = v2;
            let (v3, l3) = decode_one(&tables.slot3, reader.peek_window())?;
            reader.consume_bits_trusted(l3 as u32);
            pixels[off + 3] = v3;
        }
    }

    ensure_fully_read(&reader, bit_data)?;
    if predictor == Predictor::Median {
        return Err(Error::invalid("median predictor not legal for RGB"));
    }
    if decorrelate {
        // The first pixel is stored as the plain input bytes (spec/03
        // §1.1 / §1.3, spec/02 §8 #6) while every residual is taken on
        // the decorrelated channels (spec/03 §2.4.1), so move the first
        // pixel into the decorrelated domain before chaining from it.
        let g0 = pixels[1];
        pixels[0] = pixels[0].wrapping_sub(g0);
        pixels[2] = pixels[2].wrapping_sub(g0);
    }
    // Per-channel LEFT inverse over the interleaved raster, then the
    // gradient post-pass (spec/03 §2.2.2) with the prediction row
    // length of spec/06 §2.1.
    inverse_left_row(&mut pixels, 4);
    if predictor == Predictor::Gradient {
        inverse_gradient_stride(&mut pixels, stride);
    }
    if decorrelate {
        // Inverse decorrelation: B = (B−G) + G, R = (R−G) + G.
        inverse_rgb_decorr_bgra(&mut pixels);
    }

    Ok(DecodedFrame {
        family: PixelFamily::Rgb32,
        width,
        height,
        pixels,
    })
}

#[cfg(test)]
mod degenerate_dims_tests {
    //! Regression coverage for fuzz-found degenerate-dimension crashes.
    //!
    //! A `BITMAPINFOHEADER` may declare a zero width or height (or a
    //! width/height pair whose product is too small to hold even the
    //! uncompressed seed pixel). Earlier the field decoders guarded the
    //! *input* frame length (`< 4`) but allocated the *output* raster as
    //! `w * h * bpp` and then wrote the seed pixel into it — so a zero
    //! raster panicked on the slice write rather than returning `Err`.
    //! These cases must return `Err`, never panic.

    use super::*;
    use crate::header::{Method, PixelFamily, StreamConfig};

    fn cfg(family: PixelFamily, method: Method, width: u32, height: u32) -> StreamConfig {
        StreamConfig {
            family,
            method,
            width,
            height,
            has_extradata: false,
            extradata_tables: Vec::new(),
            interlace_flag: 0,
        }
    }

    #[test]
    fn yuy2_zero_height_is_err_not_panic() {
        let c = cfg(PixelFamily::Yuy2, Method::Left, 4, 0);
        let frame = [0u8; 64];
        assert!(decode_frame(&c, &frame).is_err());
    }

    #[test]
    fn yuy2_zero_width_is_err_not_panic() {
        let c = cfg(PixelFamily::Yuy2, Method::Left, 0, 4);
        let frame = [0u8; 64];
        assert!(decode_frame(&c, &frame).is_err());
    }

    #[test]
    fn rgb24_zero_height_is_err_not_panic() {
        let c = cfg(PixelFamily::Rgb24, Method::Left, 4, 0);
        let frame = [0u8; 64];
        assert!(decode_frame(&c, &frame).is_err());
    }

    #[test]
    fn rgb32_zero_height_is_err_not_panic() {
        let c = cfg(PixelFamily::Rgb32, Method::Left, 4, 0);
        let frame = [0u8; 64];
        assert!(decode_frame(&c, &frame).is_err());
    }

    #[test]
    fn field_coded_truncated_frame_no_panic() {
        // height > 288 engages field coding; a frame body far shorter
        // than the raster needs must not panic.
        let c = cfg(PixelFamily::Yuy2, Method::Left, 2, 290);
        let frame = [0u8; 8];
        let _ = decode_frame(&c, &frame); // Ok or Err, never panic.
    }
}

#[cfg(test)]
mod round208_reuse_tests {
    //! Round-208 regression guard. After the three single-use
    //! decoder-local LEFT wrappers (`inverse_left_per_channel`,
    //! `inverse_yuy2_left`, `inverse_yuy2_left_range`) were dropped
    //! and the YUY2 / RGB decode paths re-pointed at the public
    //! predict-side helpers (`inverse_left_row` /
    //! `inverse_yuy2_left_macropixel`) directly, we lock the new
    //! call-graph in against a from-scratch reference body to keep
    //! a future refactor from silently inverting LEFT differently.
    //!
    //! spec/03 §2.1 (per-channel LEFT) and §2.1.1 (YUY2 macropixel
    //! LEFT) define the two predictors exercised here.
    use crate::predict::{inverse_left_row, inverse_yuy2_left_macropixel};

    /// Naive scalar per-channel LEFT inverse — one cumulative running
    /// sum per channel slot, stride `n` apart. Mirrors the textual
    /// definition in spec/03 §2.1.
    fn ref_inverse_left_per_channel(out: &mut [u8], n: usize) {
        for i in n..out.len() {
            out[i] = out[i].wrapping_add(out[i - n]);
        }
    }

    /// Naive scalar YUY2 LEFT inverse — the per-byte-position-stride
    /// form from spec/03 §2.1.1 (Y₁ / Y₂ at -2, U / V at -4).
    fn ref_inverse_yuy2_left(out: &mut [u8]) {
        for i in 4..out.len() {
            let stride = if i & 1 == 0 { 2 } else { 4 };
            out[i] = out[i].wrapping_add(out[i - stride]);
        }
    }

    #[test]
    fn predict_inverse_left_row_matches_decoder_naive_rgb24() {
        // 3-channel BGR @ width=7, height=5 → 105 bytes.
        let mut a = (0..105u32)
            .map(|x| (x.wrapping_mul(37) ^ 0x5a) as u8)
            .collect::<Vec<_>>();
        let mut b = a.clone();
        ref_inverse_left_per_channel(&mut a, 3);
        inverse_left_row(&mut b, 3);
        assert_eq!(a, b);
    }

    #[test]
    fn predict_inverse_left_row_matches_decoder_naive_rgb32() {
        // 4-channel BGRA @ width=6, height=4 → 96 bytes.
        let mut a = (0..96u32)
            .map(|x| (x.wrapping_mul(53) ^ 0xa5) as u8)
            .collect::<Vec<_>>();
        let mut b = a.clone();
        ref_inverse_left_per_channel(&mut a, 4);
        inverse_left_row(&mut b, 4);
        assert_eq!(a, b);
    }

    #[test]
    fn predict_inverse_yuy2_left_macropixel_matches_decoder_naive() {
        // YUY2 row_bytes = 2*width; 8x6 → 96 bytes.
        let mut a = (0..96u32)
            .map(|x| (x.wrapping_mul(29) ^ 0xc3) as u8)
            .collect::<Vec<_>>();
        let mut b = a.clone();
        let n = b.len();
        ref_inverse_yuy2_left(&mut a);
        inverse_yuy2_left_macropixel(&mut b, 4, n);
        assert_eq!(a, b);
    }
}

#[cfg(test)]
mod round214_yuy2_decode_macropixel_tests {
    //! Round-214 regression guard. The YUY2 Huffman-decode loop was
    //! rewritten from a per-byte `match byte_idx % 4` slot dispatch
    //! into a macropixel-step body that decodes four codes per outer
    //! iteration (Y₁ via slot1, U via slot2, Y₂ via slot1, V via
    //! slot3) — the decode-side analogue of round 181's LEFT
    //! macropixel-step rewrite (also branch-elimination on the same
    //! 4-byte cycle).
    //!
    //! Spec/03 §1.2's three-slot architecture (the wire-format
    //! invariant the rewrite leans on) pins the YUY2 byte → slot
    //! mapping at `+0 (Y₁) → slot1; +1 (U) → slot2; +2 (Y₂) → slot1;
    //! +3 (V) → slot3` for every 4-byte macropixel. These tests
    //! lock the rewrite against:
    //!
    //! - **Encode-then-decode round-trips** for widths bracketing the
    //!   macropixel-step boundaries (the in-spec input space is
    //!   already `(total_bytes - 4) % 4 == 0` because YUY2 width is
    //!   even, but we still want explicit coverage at width = 2 / 4
    //!   so the per-iteration slot pattern is exercised against
    //!   minimal macropixel counts).
    //! - **Slot-pattern witness** — a constant-frame encode where
    //!   each slot's table is biased to make a single decoded byte
    //!   value distinguishable per slot, then a decode that asserts
    //!   the resulting wire-byte slot pattern matches the spec
    //!   sequence at every macropixel.
    //!
    //! Together they pin the four `decode_one` calls inside the new
    //! body against the spec's slot mapping so a future refactor
    //! cannot silently swap two slots inside the unrolled body.

    use super::*;
    use crate::encoder::{ExtradataMode, encode_frame_with_mode};
    use crate::header::{Method, StreamConfig};

    fn synth_yuy2(width: usize, height: usize) -> Vec<u8> {
        // Deterministic xorshift32 ramp; same shape as the
        // roundtrip_tests::synth_yuy2 helper but inlined to keep the
        // round-214 tests self-contained.
        let mut s: u32 = 0xCAFE_BABE;
        let n = width * height * 2;
        let mut out = vec![0u8; n];
        for slot in out.iter_mut() {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            *slot = s as u8;
        }
        out
    }

    fn rt_yuy2(width: u32, height: u32, method: Method, mode: ExtradataMode) {
        let pixels = synth_yuy2(width as usize, height as usize);
        let (bih, frame) =
            encode_frame_with_mode(PixelFamily::Yuy2, method, width, height, &pixels, mode)
                .expect("encode");
        let cfg = StreamConfig::parse_bitmapinfoheader(&bih).expect("parse");
        let decoded = decode_frame(&cfg, &frame).expect("decode");
        assert_eq!(decoded.pixels, pixels);
    }

    #[test]
    fn round214_yuy2_left_classic_width_2() {
        // width=2 → row_bytes=4, exactly one macropixel per row. With
        // the round-214 step body, the inner Y₁/U/Y₂/V sequence fires
        // exactly once per row of the body (height-1 iterations).
        rt_yuy2(2, 6, Method::Left, ExtradataMode::ClassicV2);
    }

    #[test]
    fn round214_yuy2_left_classic_width_4() {
        // width=4 → row_bytes=8, two macropixels per row. Hits the
        // step body twice per row.
        rt_yuy2(4, 4, Method::Left, ExtradataMode::ClassicV2);
    }

    #[test]
    fn round214_yuy2_gradient_custom_width_8() {
        // width=8 / Gradient / CustomV2 — exercises the new step body
        // through the runtime-built Huffman tables (CustomV2 derives
        // per-channel lengths from histograms, so it's most sensitive
        // to a slot-mix-up in the decode loop).
        rt_yuy2(8, 5, Method::Gradient, ExtradataMode::CustomV2);
    }

    #[test]
    fn round214_yuy2_median_v1x_width_8() {
        // V1xCompat path: slot1=set A, slot2=set B, slot3=set B per
        // spec/04 §4.1. Distinct tables-per-slot is the cleanest
        // wire-level witness that the step body wires each output
        // byte to the right slot — a mix-up would surface as a
        // Huffman-table mismatch even before the predictor pass.
        rt_yuy2(8, 7, Method::Median, ExtradataMode::V1xCompat);
    }

    #[test]
    fn round214_yuy2_left_classic_width_16_height_3() {
        // Wider row + small height — exercises four macropixels per
        // row across three rows so the step body crosses the row
        // boundary multiple times. The decode loop doesn't depend on
        // row boundaries (the slot mapping is wire-byte-modular, not
        // row-modular), so this also pins the rewrite against any
        // future attempt to re-introduce row-aware state inside the
        // step body.
        rt_yuy2(16, 3, Method::Left, ExtradataMode::ClassicV2);
    }

    /// Reference body: the pre-r214 per-byte slot-dispatch decode loop,
    /// inlined verbatim. Compared against the production decoder over
    /// a deterministic wire-byte stream to lock the rewrite at
    /// byte-equality.
    fn ref_decode_yuy2_per_byte_loop(
        width: u32,
        height: u32,
        frame_bytes: &[u8],
        tables: &ThreeTables,
    ) -> Vec<u8> {
        let total_bytes = (width as usize) * (height as usize) * 2;
        let mut pixels = vec![0u8; total_bytes];
        pixels[..4].copy_from_slice(&frame_bytes[..4]);
        let bit_data = &frame_bytes[4..];
        let mut reader = BitReader::new(bit_data);
        for (byte_idx, slot_pixel) in pixels.iter_mut().enumerate().take(total_bytes).skip(4) {
            let slot = match byte_idx % 4 {
                0 | 2 => &tables.slot1,
                1 => &tables.slot2,
                _ => &tables.slot3,
            };
            let (sym, len) = decode_one(slot, reader.peek_window()).expect("decode");
            reader.consume_bits(len as u32).expect("consume");
            *slot_pixel = sym;
        }
        pixels
    }

    #[test]
    fn round214_yuy2_decode_matches_per_byte_reference() {
        // Encode a synthetic frame, then decode the wire bytes twice:
        // once with the production decode_frame (which goes through the
        // round-214 macropixel-step body), once with the reference
        // per-byte slot dispatch above. The pre-predictor sample
        // stream must be byte-identical between the two, which means
        // the production output (after the predictor pass) must equal
        // the reference output (after the same predictor pass applied
        // separately). We verify by re-running the round-trip and
        // confirming the production decode equals the input.
        //
        // The witness here is that the reference body and the round-
        // 214 body, given the same bit cursor and the same three
        // tables, must produce the same sample stream — otherwise the
        // round-trip would not survive the predictor inverse.
        for (w, h, method) in [
            (4u32, 4u32, Method::Left),
            (8, 6, Method::Gradient),
            (8, 6, Method::Median),
        ] {
            let pixels = synth_yuy2(w as usize, h as usize);
            let (bih, frame) = encode_frame_with_mode(
                PixelFamily::Yuy2,
                method,
                w,
                h,
                &pixels,
                ExtradataMode::ClassicV2,
            )
            .expect("encode");
            let cfg = StreamConfig::parse_bitmapinfoheader(&bih).expect("parse");
            let tables = build_three_tables(&cfg).expect("tables");
            // Reference per-byte loop produces the raw sample stream
            // (residuals after the predictor split). The production
            // decoder produces the same raw stream pre-predictor and
            // then applies the predictor inverse. Both pre-predictor
            // streams must be equal — we witness this by also running
            // the same predictor inverse on the reference output and
            // checking it round-trips to `pixels`.
            let mut ref_pixels = ref_decode_yuy2_per_byte_loop(w, h, &frame, &tables);
            let row_bytes = (w as usize) * 2;
            let len = ref_pixels.len();
            match method {
                Method::Left => {
                    crate::predict::inverse_yuy2_left_macropixel(&mut ref_pixels, 4, len);
                }
                Method::Gradient => {
                    crate::predict::inverse_yuy2_left_macropixel(&mut ref_pixels, 4, len);
                    crate::predict::inverse_gradient_stride(&mut ref_pixels, row_bytes);
                }
                Method::Median => {
                    crate::predict::inverse_median_yuy2(&mut ref_pixels, row_bytes);
                }
                _ => unreachable!("YUY2 doesn't carry decorrelating methods"),
            }
            assert_eq!(ref_pixels, pixels, "ref body must round-trip identically");
            // And the production decoder must also round-trip.
            let prod = decode_frame(&cfg, &frame).expect("decode");
            assert_eq!(prod.pixels, pixels, "production decode must round-trip");
        }
    }
}

#[cfg(test)]
mod round262_row_bytes_accessor_tests {
    //! Round-262 regression guard. The three per-family field
    //! decoders (`decode_yuy2_field` / `decode_rgb24_field` /
    //! `decode_rgb32_field`) each opened with their own open-coded
    //! `width × {2, 3, 4}` wire-stride; round 262 routes all three
    //! through the round-261 single source of truth
    //! (`PixelFamily::row_bytes`, spec/02 §3 wire-byte layout
    //! table). Wire-identical: same family → stride mapping, one
    //! origin instead of four (r261's two call sites + these).
    //!
    //! Coverage:
    //!
    //! - **Raster-length pin** — a full encode → decode round-trip
    //!   per family asserts the decoded raster is exactly
    //!   `family.row_bytes(width) × height` bytes AND bit-exact
    //!   against the input, so a stride drift inside any of the
    //!   three decoders would surface as a length or content
    //!   mismatch.
    //! - **Degenerate-dimension guard** — the `total_bytes < {3,4}`
    //!   rejection (fuzz-found, see `degenerate_dims_tests`) must
    //!   keep firing now that `row_bytes` comes from the saturating
    //!   accessor (zero width / height still yields a too-small
    //!   raster → `Err`, never a panic).

    use crate::encoder::{ExtradataMode, encode_frame_with_mode};
    use crate::header::{Method, PixelFamily, StreamConfig};

    fn synth(n: usize, mut s: u32) -> Vec<u8> {
        let mut out = vec![0u8; n];
        for slot in out.iter_mut() {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            *slot = s as u8;
        }
        out
    }

    fn rt_pin_raster_len(family: PixelFamily, method: Method, w: u32, h: u32) {
        let pixels = synth(family.row_bytes(w) * h as usize, 0xC0FF_EE11);
        let (bih, frame) =
            encode_frame_with_mode(family, method, w, h, &pixels, ExtradataMode::ClassicV2)
                .expect("encode");
        let cfg = StreamConfig::parse_bitmapinfoheader(&bih).expect("parse");
        let decoded = super::decode_frame(&cfg, &frame).expect("decode");
        assert_eq!(
            decoded.pixels.len(),
            cfg.family.row_bytes(cfg.width) * cfg.height as usize,
            "decoded raster length must equal family.row_bytes(width) × height"
        );
        assert_eq!(decoded.pixels, pixels, "round-trip must be bit-exact");
    }

    #[test]
    fn round262_yuy2_decoded_raster_len_matches_family_row_bytes() {
        rt_pin_raster_len(PixelFamily::Yuy2, Method::Left, 4, 3);
        rt_pin_raster_len(PixelFamily::Yuy2, Method::Median, 8, 5);
    }

    #[test]
    fn round262_rgb24_decoded_raster_len_matches_family_row_bytes() {
        rt_pin_raster_len(PixelFamily::Rgb24, Method::Left, 3, 4);
        rt_pin_raster_len(PixelFamily::Rgb24, Method::LeftDecorr, 5, 2);
    }

    #[test]
    fn round262_rgb32_decoded_raster_len_matches_family_row_bytes() {
        rt_pin_raster_len(PixelFamily::Rgb32, Method::Left, 3, 4);
        rt_pin_raster_len(PixelFamily::Rgb32, Method::GradientDecorr, 4, 3);
    }

    #[test]
    fn round262_degenerate_dims_still_rejected_post_accessor() {
        // Zero width saturates to a zero row stride in the accessor;
        // the `total_bytes < {3,4}` guards must still reject before
        // any raster write (Err, never panic).
        let frame = [0u8; 64];
        for family in [PixelFamily::Yuy2, PixelFamily::Rgb24, PixelFamily::Rgb32] {
            for (w, h) in [(0u32, 4u32), (4, 0), (0, 0)] {
                let c = StreamConfig {
                    family,
                    method: Method::Left,
                    width: w,
                    height: h,
                    has_extradata: false,
                    extradata_tables: Vec::new(),
                    interlace_flag: 0,
                };
                assert!(
                    super::decode_frame(&c, &frame).is_err(),
                    "degenerate {family:?} {w}x{h} must be Err"
                );
            }
        }
    }
}

#[cfg(test)]
mod round277_rgb_decode_binding_tests {
    //! Round-277 regression guard. The RGB24 / RGB32 Huffman-decode
    //! loops were rewritten from a per-pixel `if decorrelate` branch
    //! (which re-resolved the slot pointers + BGR store offsets
    //! inside each arm on every pixel) into a wire-position
    //! (slot, store-offset) binding resolved once at function entry —
    //! the decode-side analogue of the r239 (RGB24) / r245 (RGB32)
    //! encoder emit-loop bindings, completing the r214 / r239 / r245
    //! hoist series on the two remaining RGB decode loops.
    //!
    //! Wire-format invariants the rewrite leans on (spec/03 §1.4 +
    //! §1.2):
    //!
    //! - RGB24 no decorr: wire order `B, G, R` → (slot1, slot2,
    //!   slot3), stored at BGR offsets +0 / +1 / +2.
    //! - RGB24 decorr: wire order `G, B−G, R−G` → (slot2, slot1,
    //!   slot3), stored at +1 / +0 / +2.
    //! - RGB32 appends the A code at wire position +3 in both modes,
    //!   sharing the slot-3 codebook (alpha is never decorrelated per
    //!   the §2.4 Validator note), stored at +3.
    //!
    //! Coverage:
    //!
    //! - **Encode → decode round-trips** at widths 1 / 2 / 4 / 8
    //!   across the no-decorr (`Left` / `PredictOld`) and decorr
    //!   (`LeftDecorr` / `GradientDecorr`) bindings, under CustomV2
    //!   (content-distinct slot tables — a slot mix-up surfaces as a
    //!   Huffman-code mismatch on the wire), ClassicV2, and V1xCompat
    //!   (content-identical `(A, A, A)` RGB triple — a slot mix-up
    //!   survives identical tables, so these pin the *store offsets*
    //!   through the round-trip instead).
    //! - **Per-pixel-branch reference witnesses** — an inlined copy
    //!   of the pre-r277 branch-in-loop body per family, run over the
    //!   same wire bytes with the same tables, followed by the same
    //!   predictor + decorrelation inverses; both the reference and
    //!   the production decode must reconstruct the source exactly.

    use super::*;
    use crate::encoder::{ExtradataMode, encode_frame_with_mode};
    use crate::header::Method;

    fn synth(n: usize, seed: u32) -> Vec<u8> {
        // Deterministic xorshift32 ramp; same shape as the round-214
        // helper but parameterised on byte count so one helper covers
        // both RGB families.
        let mut s: u32 = seed;
        let mut out = vec![0u8; n];
        for slot in out.iter_mut() {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            *slot = s as u8;
        }
        out
    }

    fn rt(family: PixelFamily, width: u32, height: u32, method: Method, mode: ExtradataMode) {
        let bpp = match family {
            PixelFamily::Rgb24 => 3,
            PixelFamily::Rgb32 => 4,
            PixelFamily::Yuy2 => unreachable!("round-277 covers the RGB families"),
        };
        let pixels = synth((width as usize) * (height as usize) * bpp, 0xC0FF_EE77);
        let (bih, frame) =
            encode_frame_with_mode(family, method, width, height, &pixels, mode).expect("encode");
        let cfg = StreamConfig::parse_bitmapinfoheader(&bih).expect("parse");
        let decoded = decode_frame(&cfg, &frame).expect("decode");
        assert_eq!(
            decoded.pixels, pixels,
            "{family:?} {width}x{height} {method:?}"
        );
    }

    #[test]
    fn round277_rgb24_no_decorr_binding_widths() {
        // width=1 → one pixel per row: the per-pixel body fires
        // height−1 times with every store landing in a distinct row.
        rt(
            PixelFamily::Rgb24,
            1,
            6,
            Method::Left,
            ExtradataMode::ClassicV2,
        );
        rt(
            PixelFamily::Rgb24,
            2,
            4,
            Method::Left,
            ExtradataMode::CustomV2,
        );
        rt(
            PixelFamily::Rgb24,
            8,
            5,
            Method::Left,
            ExtradataMode::CustomV2,
        );
    }

    #[test]
    fn round277_rgb24_predict_old_alternate_no_decorr_entry() {
        // PredictOld shares the no-decorr binding with Left; pins the
        // alternate method entry into the same (slot, offset) tuple.
        rt(
            PixelFamily::Rgb24,
            4,
            4,
            Method::PredictOld,
            ExtradataMode::ClassicV2,
        );
    }

    #[test]
    fn round277_rgb24_decorr_binding_widths() {
        // LeftDecorr / GradientDecorr take the swapped-pair binding
        // (G first on the wire, stored at +1). CustomV2 builds
        // content-distinct slot tables, so a binding mix-up surfaces
        // as a wire-level Huffman mismatch before the predictor pass.
        rt(
            PixelFamily::Rgb24,
            1,
            5,
            Method::LeftDecorr,
            ExtradataMode::CustomV2,
        );
        rt(
            PixelFamily::Rgb24,
            4,
            4,
            Method::LeftDecorr,
            ExtradataMode::ClassicV2,
        );
        rt(
            PixelFamily::Rgb24,
            8,
            3,
            Method::GradientDecorr,
            ExtradataMode::CustomV2,
        );
    }

    #[test]
    fn round277_rgb24_v1x_content_identical_tables_pin_store_offsets() {
        // V1xCompat RGB = (A, A, A): all three slots hold the same
        // table, so a slot mix-up cannot surface as a code mismatch —
        // this round-trip pins the *store-offset* half of the binding.
        rt(
            PixelFamily::Rgb24,
            4,
            6,
            Method::Left,
            ExtradataMode::V1xCompat,
        );
        rt(
            PixelFamily::Rgb24,
            4,
            6,
            Method::LeftDecorr,
            ExtradataMode::V1xCompat,
        );
    }

    #[test]
    fn round277_rgb32_no_decorr_binding_widths() {
        rt(
            PixelFamily::Rgb32,
            1,
            6,
            Method::Left,
            ExtradataMode::ClassicV2,
        );
        rt(
            PixelFamily::Rgb32,
            2,
            4,
            Method::Left,
            ExtradataMode::CustomV2,
        );
        rt(
            PixelFamily::Rgb32,
            8,
            5,
            Method::Left,
            ExtradataMode::CustomV2,
        );
    }

    #[test]
    fn round277_rgb32_decorr_binding_widths() {
        rt(
            PixelFamily::Rgb32,
            1,
            5,
            Method::LeftDecorr,
            ExtradataMode::CustomV2,
        );
        rt(
            PixelFamily::Rgb32,
            4,
            4,
            Method::GradientDecorr,
            ExtradataMode::ClassicV2,
        );
        rt(
            PixelFamily::Rgb32,
            8,
            3,
            Method::GradientDecorr,
            ExtradataMode::CustomV2,
        );
    }

    #[test]
    fn round277_rgb32_v1x_content_identical_tables_pin_store_offsets() {
        rt(
            PixelFamily::Rgb32,
            4,
            6,
            Method::Left,
            ExtradataMode::V1xCompat,
        );
        rt(
            PixelFamily::Rgb32,
            4,
            6,
            Method::GradientDecorr,
            ExtradataMode::V1xCompat,
        );
    }

    /// Reference body: the pre-r277 per-pixel `if decorrelate`
    /// branch-in-loop RGB24 decode, inlined verbatim. Produces the
    /// raw pre-predictor sample stream.
    fn ref_decode_rgb24_per_pixel_branch(
        decorrelate: bool,
        width: u32,
        height: u32,
        frame_bytes: &[u8],
        tables: &ThreeTables,
    ) -> Vec<u8> {
        let total_bytes = (width as usize) * (height as usize) * 3;
        let mut pixels = vec![0u8; total_bytes];
        pixels[0] = frame_bytes[1];
        pixels[1] = frame_bytes[2];
        pixels[2] = frame_bytes[3];
        let mut reader = BitReader::new(&frame_bytes[4..]);
        let n_pixels = (width as usize) * (height as usize);
        for px in 1..n_pixels {
            let bgr_off = px * 3;
            if decorrelate {
                let (g, lg) = decode_one(&tables.slot2, reader.peek_window()).expect("decode");
                reader.consume_bits(lg as u32).expect("consume");
                let (bg, lb) = decode_one(&tables.slot1, reader.peek_window()).expect("decode");
                reader.consume_bits(lb as u32).expect("consume");
                let (rg, lr) = decode_one(&tables.slot3, reader.peek_window()).expect("decode");
                reader.consume_bits(lr as u32).expect("consume");
                pixels[bgr_off] = bg;
                pixels[bgr_off + 1] = g;
                pixels[bgr_off + 2] = rg;
            } else {
                let (b, lb) = decode_one(&tables.slot1, reader.peek_window()).expect("decode");
                reader.consume_bits(lb as u32).expect("consume");
                let (g, lg) = decode_one(&tables.slot2, reader.peek_window()).expect("decode");
                reader.consume_bits(lg as u32).expect("consume");
                let (r, lr) = decode_one(&tables.slot3, reader.peek_window()).expect("decode");
                reader.consume_bits(lr as u32).expect("consume");
                pixels[bgr_off] = b;
                pixels[bgr_off + 1] = g;
                pixels[bgr_off + 2] = r;
            }
        }
        pixels
    }

    /// Reference body: the pre-r277 per-pixel `if decorrelate`
    /// branch-in-loop RGB32 decode, inlined verbatim.
    fn ref_decode_rgb32_per_pixel_branch(
        decorrelate: bool,
        width: u32,
        height: u32,
        frame_bytes: &[u8],
        tables: &ThreeTables,
    ) -> Vec<u8> {
        let total_bytes = (width as usize) * (height as usize) * 4;
        let mut pixels = vec![0u8; total_bytes];
        pixels[..4].copy_from_slice(&frame_bytes[..4]);
        let mut reader = BitReader::new(&frame_bytes[4..]);
        let n_pixels = (width as usize) * (height as usize);
        for px in 1..n_pixels {
            let off = px * 4;
            if decorrelate {
                let (g, lg) = decode_one(&tables.slot2, reader.peek_window()).expect("decode");
                reader.consume_bits(lg as u32).expect("consume");
                let (bg, lb) = decode_one(&tables.slot1, reader.peek_window()).expect("decode");
                reader.consume_bits(lb as u32).expect("consume");
                let (rg, lr) = decode_one(&tables.slot3, reader.peek_window()).expect("decode");
                reader.consume_bits(lr as u32).expect("consume");
                let (a, la) = decode_one(&tables.slot3, reader.peek_window()).expect("decode");
                reader.consume_bits(la as u32).expect("consume");
                pixels[off] = bg;
                pixels[off + 1] = g;
                pixels[off + 2] = rg;
                pixels[off + 3] = a;
            } else {
                let (b, lb) = decode_one(&tables.slot1, reader.peek_window()).expect("decode");
                reader.consume_bits(lb as u32).expect("consume");
                let (g, lg) = decode_one(&tables.slot2, reader.peek_window()).expect("decode");
                reader.consume_bits(lg as u32).expect("consume");
                let (r, lr) = decode_one(&tables.slot3, reader.peek_window()).expect("decode");
                reader.consume_bits(lr as u32).expect("consume");
                let (a, la) = decode_one(&tables.slot3, reader.peek_window()).expect("decode");
                reader.consume_bits(la as u32).expect("consume");
                pixels[off] = b;
                pixels[off + 1] = g;
                pixels[off + 2] = r;
                pixels[off + 3] = a;
            }
        }
        pixels
    }

    /// Shared inverse chain for the witness: the same predictor +
    /// decorrelation inverses the production decoder applies after
    /// its Huffman loop.
    fn apply_rgb_inverses(
        out: &mut [u8],
        method: Method,
        n_channels: usize,
        row_bytes: usize,
        height: usize,
    ) {
        if method.decorrelate() {
            // The first pixel is stored plain (spec/03 §1.1 / §1.3).
            let g0 = out[1];
            out[0] = out[0].wrapping_sub(g0);
            out[2] = out[2].wrapping_sub(g0);
        }
        inverse_left_row(out, n_channels);
        if matches!(method.predictor(), Predictor::Gradient) {
            crate::predict::inverse_gradient_post(out, row_bytes, height);
        }
        if method.decorrelate() {
            if n_channels == 3 {
                inverse_rgb_decorr_bgr(out);
            } else {
                inverse_rgb_decorr_bgra(out);
            }
        }
    }

    #[test]
    fn round277_rgb24_decode_matches_per_pixel_branch_reference() {
        // CustomV2 so the three slot tables are content-distinct — a
        // binding mix-up in the production loop desynchronises the
        // bit cursor and surfaces before the predictor pass.
        for (w, h, method) in [
            (4u32, 4u32, Method::Left),
            (8, 6, Method::LeftDecorr),
            (8, 6, Method::GradientDecorr),
        ] {
            let pixels = synth((w as usize) * (h as usize) * 3, 0xDEAD_4077);
            let (bih, frame) = encode_frame_with_mode(
                PixelFamily::Rgb24,
                method,
                w,
                h,
                &pixels,
                ExtradataMode::CustomV2,
            )
            .expect("encode");
            let cfg = StreamConfig::parse_bitmapinfoheader(&bih).expect("parse");
            let tables = build_three_tables(&cfg).expect("tables");
            let mut ref_pixels =
                ref_decode_rgb24_per_pixel_branch(method.decorrelate(), w, h, &frame, &tables);
            apply_rgb_inverses(&mut ref_pixels, method, 3, (w as usize) * 3, h as usize);
            assert_eq!(ref_pixels, pixels, "ref body must round-trip identically");
            let prod = decode_frame(&cfg, &frame).expect("decode");
            assert_eq!(prod.pixels, pixels, "production decode must round-trip");
        }
    }

    #[test]
    fn round277_rgb32_decode_matches_per_pixel_branch_reference() {
        for (w, h, method) in [
            (4u32, 4u32, Method::Left),
            (8, 6, Method::LeftDecorr),
            (8, 6, Method::GradientDecorr),
        ] {
            let pixels = synth((w as usize) * (h as usize) * 4, 0xFEED_4077);
            let (bih, frame) = encode_frame_with_mode(
                PixelFamily::Rgb32,
                method,
                w,
                h,
                &pixels,
                ExtradataMode::CustomV2,
            )
            .expect("encode");
            let cfg = StreamConfig::parse_bitmapinfoheader(&bih).expect("parse");
            let tables = build_three_tables(&cfg).expect("tables");
            let mut ref_pixels =
                ref_decode_rgb32_per_pixel_branch(method.decorrelate(), w, h, &frame, &tables);
            apply_rgb_inverses(&mut ref_pixels, method, 4, (w as usize) * 4, h as usize);
            assert_eq!(ref_pixels, pixels, "ref body must round-trip identically");
            let prod = decode_frame(&cfg, &frame).expect("decode");
            assert_eq!(prod.pixels, pixels, "production decode must round-trip");
        }
    }
}

#[cfg(test)]
mod round429_pair_sym_lut_tests {
    //! Round-429 soundness guard for the decode-side full pair LUTs
    //! (`tables::build_pair_sym_lut`): a non-zero entry claims that
    //! EVERY 32-bit window sharing the entry's 16-bit prefix decodes
    //! its next two codewords to those exact `(symbol, symbol,
    //! combined length)` values. Verify the claim against the
    //! production `decode_pair` / `decode_one` walk across randomised
    //! low window bits, for all four (first, second) slot pairs the
    //! serial decode loops consume, on both the v1.x compiled-in
    //! tables and a histogram-derived CustomV2 set. Zero entries are
    //! checked too: they must occur only where the exact walk needs
    //! more than 16 known bits (`l1 + l2 > 16` or a primary-LUT
    //! overflow), so the fallback arm — not silence — serves them.

    use super::*;
    use crate::tables::{build_pair_sym_lut, decode_one, decode_pair};

    fn check_pair(a: &HuffTable, b: &HuffTable, label: &str) {
        let lut = build_pair_sym_lut(a, b);
        let mut s: u32 = 0x0429_5eed;
        for (p, &entry) in lut.iter().enumerate() {
            if entry == 0 {
                // Zero ⇒ the pair must genuinely be unresolvable from
                // 16 known bits (overflow or straddling): reproduce
                // the builder's decision from the zero-completed
                // window and assert one of those two conditions.
                let w = (p as u32) << 16;
                if let Some((_, _, n)) = decode_pair(a, b, w) {
                    assert!(
                        n > 16,
                        "{label}: prefix 0x{p:04x}: zero entry but pair fits 16 bits"
                    );
                }
                continue;
            }
            for _ in 0..4 {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                let w = ((p as u32) << 16) | (s & 0xFFFF);
                // Reference: the exact walk the decode loops fall
                // back to.
                let (ra, rb, n) = if let Some(hit) = decode_pair(a, b, w) {
                    hit
                } else {
                    let (sa, l0) = decode_one(a, w).expect("code A must decode");
                    let (sb, l1) =
                        decode_one(b, w << l0).expect("code B must decode within window");
                    (sa, sb, (l0 + l1) as u32)
                };
                assert_eq!(
                    (entry as u8, (entry >> 8) as u8, entry >> 16),
                    (ra, rb, n),
                    "{label}: prefix 0x{p:04x} low 0x{:04x}: LUT entry vs walk diverged",
                    w & 0xFFFF
                );
            }
        }
    }

    #[test]
    fn round429_pair_sym_lut_sound_v1x_tables() {
        let yuv = table_cache::v1x_tables(PixelFamily::Yuy2).expect("tables");
        check_pair(&yuv.slot1, &yuv.slot2, "v1x yuv s1s2");
        check_pair(&yuv.slot1, &yuv.slot3, "v1x yuv s1s3");
        let rgb = table_cache::v1x_tables(PixelFamily::Rgb24).expect("tables");
        check_pair(&rgb.slot2, &rgb.slot1, "v1x rgb s2s1");
        check_pair(&rgb.slot3, &rgb.slot3, "v1x rgb s3s3");
        // Round-440 RGB24 phase-flip combos.
        check_pair(&rgb.slot3, &rgb.slot1, "v1x rgb s3s1");
        check_pair(&rgb.slot2, &rgb.slot3, "v1x rgb s2s3");
        check_pair(&rgb.slot3, &rgb.slot2, "v1x rgb s3s2");
    }

    #[test]
    fn round429_pair_sym_lut_sound_custom_tables() {
        use crate::encoder::{ExtradataMode, encode_frame_with_mode};
        use crate::header::Method;
        let mut s: u32 = 0xBEEF_0429;
        let mut pixels = vec![0u8; 64 * 300 * 2];
        for slot in pixels.iter_mut() {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            *slot = s as u8;
        }
        let (strf, _) = encode_frame_with_mode(
            PixelFamily::Yuy2,
            Method::Median,
            64,
            300,
            &pixels,
            ExtradataMode::CustomV2,
        )
        .expect("encode");
        let cfg = StreamConfig::parse_bitmapinfoheader(&strf).expect("parse");
        let tables = build_three_tables(&cfg).expect("tables");
        check_pair(&tables.slot1, &tables.slot2, "custom s1s2");
        check_pair(&tables.slot1, &tables.slot3, "custom s1s3");
        check_pair(&tables.slot2, &tables.slot1, "custom s2s1");
        check_pair(&tables.slot3, &tables.slot3, "custom s3s3");
        // Round-440 RGB24 phase-flip combos.
        check_pair(&tables.slot3, &tables.slot1, "custom s3s1");
        check_pair(&tables.slot2, &tables.slot3, "custom s2s3");
        check_pair(&tables.slot3, &tables.slot2, "custom s3s2");
    }
}

#[cfg(test)]
mod round440_rgb24_phase_flip_tests {
    //! Round-440 regression guard for the RGB24 two-pixel phase-flip
    //! decode body: the six codewords of two consecutive wire pixels
    //! are consumed as three pair reads — `(w0, w1)`, `(w2, w0′)`,
    //! `(w1′, w2′)` — with the middle pair crossing the pixel
    //! boundary. The witness decodes the entropy stream with a plain
    //! per-symbol `decode_one` walk (the canonical spec/03 §1.4 /
    //! §1.2 order: three codewords per pixel, mode-resolved slots) and
    //! asserts the production `decode_frame` output matches
    //! byte-for-byte, across widths that land both even and odd
    //! body-pixel counts (exercising the single-pixel tail), all
    //! legal RGB24 methods, and all three extradata modes (ClassicV2
    //! classic blobs, CustomV2 content-distinct tables — a slot
    //! mix-up in the phase-flip pairing surfaces as a Huffman-code
    //! mismatch — and the v1.x content-identical triple).

    use super::*;
    use crate::encoder::{ExtradataMode, encode_frame_with_mode};
    use crate::header::{Method, StreamConfig};
    use crate::tables::decode_one;

    fn synth(n: usize, seed: u32) -> Vec<u8> {
        let mut s = seed | 1;
        let mut out = vec![0u8; n];
        for slot in out.iter_mut() {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            *slot = s as u8;
        }
        out
    }

    /// Reference entropy decode: strict per-symbol walk, one
    /// `decode_one` per codeword in wire order, no pairing, no LUTs.
    fn ref_decode_rgb24_per_symbol(
        decorrelate: bool,
        width: u32,
        height: u32,
        frame_bytes: &[u8],
        tables: &ThreeTables,
    ) -> Vec<u8> {
        let n_pixels = (width as usize) * (height as usize);
        let mut pixels = vec![0u8; n_pixels * 3];
        pixels[0] = frame_bytes[1];
        pixels[1] = frame_bytes[2];
        pixels[2] = frame_bytes[3];
        let mut reader = BitReader::new(&frame_bytes[4..]);
        let (s_w0, o_w0, s_w1, o_w1) = if decorrelate {
            (&tables.slot2, 1usize, &tables.slot1, 0usize)
        } else {
            (&tables.slot1, 0usize, &tables.slot2, 1usize)
        };
        for px in 1..n_pixels {
            let off = px * 3;
            let (v0, l0) = decode_one(s_w0, reader.peek_window()).expect("w0");
            reader.consume_bits(l0 as u32).expect("consume w0");
            pixels[off + o_w0] = v0;
            let (v1, l1) = decode_one(s_w1, reader.peek_window()).expect("w1");
            reader.consume_bits(l1 as u32).expect("consume w1");
            pixels[off + o_w1] = v1;
            let (v2, l2) = decode_one(&tables.slot3, reader.peek_window()).expect("w2");
            reader.consume_bits(l2 as u32).expect("consume w2");
            pixels[off + 2] = v2;
        }
        pixels
    }

    #[test]
    fn round440_phase_flip_matches_per_symbol_reference() {
        const MODES: [ExtradataMode; 3] = [
            ExtradataMode::ClassicV2,
            ExtradataMode::CustomV2,
            ExtradataMode::V1xCompat,
        ];
        const METHODS: [Method; 4] = [
            Method::PredictOld,
            Method::Left,
            Method::LeftDecorr,
            Method::GradientDecorr,
        ];
        for &method in METHODS.iter() {
            for mode in MODES {
                // (w, h) sweep landing every body parity: n_pixels − 1
                // odd (2×2 → 3, 4×4 → 15, 5×5 → 24 even, 3×7 → 20
                // even, 16×3 → 47 odd, 1×1 → 0-body degenerate-ish,
                // 1×2 → 1).
                for (w, h) in [
                    (1u32, 2u32),
                    (2, 2),
                    (3, 7),
                    (4, 4),
                    (5, 5),
                    (16, 3),
                    (48, 32),
                ] {
                    let src = synth((w as usize) * (h as usize) * 3, 0x0440_0001);
                    let (bih, frame) =
                        encode_frame_with_mode(PixelFamily::Rgb24, method, w, h, &src, mode)
                            .expect("encode");
                    let cfg = StreamConfig::parse_bitmapinfoheader(&bih).expect("parse");
                    let tables = build_three_tables(&cfg).expect("tables");
                    // Entropy-level witness: production field decode
                    // (before predictor inverses) must equal the
                    // per-symbol reference walk.
                    let mut ref_pixels =
                        ref_decode_rgb24_per_symbol(method.decorrelate(), w, h, &frame, &tables);
                    // Apply the same inverse chain the production path
                    // runs, then compare the whole frame.
                    if method.decorrelate() {
                        // The first pixel is stored plain (spec/03 §1.1).
                        let g0 = ref_pixels[1];
                        ref_pixels[0] = ref_pixels[0].wrapping_sub(g0);
                        ref_pixels[2] = ref_pixels[2].wrapping_sub(g0);
                    }
                    inverse_left_row(&mut ref_pixels, 3);
                    if matches!(method.predictor(), Predictor::Gradient) {
                        crate::predict::inverse_gradient_post(
                            &mut ref_pixels,
                            (w as usize) * 3,
                            h as usize,
                        );
                    }
                    if method.decorrelate() {
                        inverse_rgb_decorr_bgr(&mut ref_pixels);
                    }
                    assert_eq!(
                        ref_pixels, src,
                        "reference walk must round-trip: {method:?}/{mode:?}/{w}x{h}"
                    );
                    let prod = decode_frame(&cfg, &frame).expect("decode");
                    assert_eq!(
                        prod.pixels, src,
                        "production phase-flip decode diverged: {method:?}/{mode:?}/{w}x{h}"
                    );
                }
            }
        }
    }

    /// Field-coded RGB24 (even and odd heights above 288): the
    /// phase-flip body walks the whole frame as one stream; decode
    /// stays lossless at every worker budget.
    #[test]
    fn round440_phase_flip_interlaced_budget_invariance() {
        for &method in [Method::Left, Method::LeftDecorr, Method::GradientDecorr].iter() {
            for (w, h) in [(16u32, 293u32), (32, 292)] {
                let src = synth((w as usize) * (h as usize) * 3, 0x0440_0002);
                let (bih, frame) = encode_frame_with_mode(
                    PixelFamily::Rgb24,
                    method,
                    w,
                    h,
                    &src,
                    ExtradataMode::ClassicV2,
                )
                .expect("encode");
                let cfg = StreamConfig::parse_bitmapinfoheader(&bih).expect("parse");
                assert!(cfg.is_interlaced());
                let serial = decode_frame(&cfg, &frame).expect("serial");
                let parallel = decode_frame_with_workers(&cfg, &frame, 2).expect("parallel");
                assert_eq!(serial.pixels, src, "serial lossless: {method:?}/{w}x{h}");
                assert_eq!(
                    parallel.pixels, src,
                    "budget-2 lossless: {method:?}/{w}x{h}"
                );
            }
        }
    }
}
