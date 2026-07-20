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
    interleave_fields, inverse_gradient_post, inverse_left_row, inverse_rgb_decorr_bgr,
    inverse_rgb_decorr_bgra, inverse_yuy2_left_macropixel,
};
use crate::tables::{
    classic_blob_bytes, decode_one, decode_pair, rle_decode_one_channel, rle_decode_three_channels,
    v1x_codes_set_a, v1x_codes_set_b, v1x_lengths_set_a, v1x_lengths_set_b, v1x_table_from_pair,
    HuffTable,
};

/// Reconstructed frame produced by [`decode_frame`].
#[derive(Debug, Clone)]
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
/// When `config.height > 288` the spec engages the field-stride=2
/// interlaced path (spec/02 §2): the chunk holds two concatenated
/// sub-frames (top field then bottom field), each with its own
/// uncompressed seed + bit-packed body. We decode each field as if
/// its own H/2-tall frame, then interleave rows back into a single
/// `height`-row raster.
pub fn decode_frame(config: &StreamConfig, frame_bytes: &[u8]) -> Result<DecodedFrame> {
    decode_frame_with_workers(config, frame_bytes, 1)
}

/// Decode one HuffYUV frame under an explicit worker budget.
///
/// `worker_budget` is the caller's advisory thread cap, in the sense of
/// the `oxideav-core` `ExecutionContext` threading contract: the codec
/// runs serial until told otherwise, and every internal fan-out is
/// bounded by `budget.min(work_units).max(1)`. This crate's only
/// independent work units are the two fields of an interlaced stream
/// (spec/02 §2: each field is coded independently — own uncompressed
/// seed, own bit-packed body — with the two per-field streams
/// concatenated on the wire), so:
///
/// - `worker_budget <= 1`, or a progressive stream: exactly the
///   [`decode_frame`] serial code path. Byte-identical output.
/// - `worker_budget >= 2` AND the stream is interlaced (with a
///   non-empty bottom field): the two fields are decoded on two
///   parallel workers via [`decode_frame_interlaced_parallel`]. The
///   output is byte-identical to the serial path (regression-guarded
///   by the round-420 invariance tests + the r419 golden pins).
///
/// The registry `Decoder` routes its stored `ExecutionContext` budget
/// here; direct-API callers can pass a budget explicitly. Budgets
/// above 2 are clamped — there is no third independent unit to feed.
pub fn decode_frame_with_workers(
    config: &StreamConfig,
    frame_bytes: &[u8],
    worker_budget: usize,
) -> Result<DecodedFrame> {
    let three_tables = build_three_tables(config)?;
    // Honour the extradata interlace_flag (BIH +0x2A) when set, falling
    // back to the biHeight > 288 heuristic otherwise — see
    // `StreamConfig::is_interlaced`.
    if config.is_interlaced() {
        // Two independently-coded fields = two parallelisable work
        // units. Inline worker clamp per the ExecutionContext
        // threading contract — `threads.min(units).max(1)`, written
        // as `clamp(1, units)` with `units = 2`.
        let workers = worker_budget.clamp(1, 2);
        if workers >= 2 {
            return decode_frame_interlaced_parallel(config, frame_bytes, &three_tables);
        }
        return decode_frame_interlaced(config, frame_bytes, &three_tables);
    }
    decode_field(
        config.family,
        config.method.predictor(),
        config.method.decorrelate(),
        config.width,
        config.height,
        frame_bytes,
        &three_tables,
    )
    .map(|(frame, _consumed)| frame)
}

fn decode_frame_interlaced(
    config: &StreamConfig,
    frame_bytes: &[u8],
    tables: &ThreeTables,
) -> Result<DecodedFrame> {
    let h = config.height as usize;
    let top_h = h.div_ceil(2) as u32;
    let bot_h = (h / 2) as u32;
    let (top_frame, top_consumed) = decode_field(
        config.family,
        config.method.predictor(),
        config.method.decorrelate(),
        config.width,
        top_h,
        frame_bytes,
        tables,
    )?;
    let bot_frame = if bot_h > 0 {
        // A malformed/truncated top field can leave the bit cursor past
        // the end of the chunk, so `top_consumed` may exceed the buffer
        // length; clamp the split point to avoid a slice-bounds panic.
        // The bottom field then sees the (possibly empty) remainder and
        // its own length guard rejects it cleanly.
        let split = top_consumed.min(frame_bytes.len());
        let rest = &frame_bytes[split..];
        let (bf, _) = decode_field(
            config.family,
            config.method.predictor(),
            config.method.decorrelate(),
            config.width,
            bot_h,
            rest,
            tables,
        )?;
        Some(bf)
    } else {
        None
    };
    // Round-261: single source of truth (`StreamConfig::row_bytes`)
    // for the family → wire-stride dispatch; spec/02 §3 wire-byte
    // layout table.
    let row_bytes = config.row_bytes();
    let merged = if let Some(bot) = bot_frame {
        interleave_fields(&top_frame.pixels, &bot.pixels, row_bytes, h)
    } else {
        top_frame.pixels
    };
    Ok(DecodedFrame {
        family: config.family,
        width: config.width,
        height: config.height,
        pixels: merged,
    })
}

/// Round-420 two-worker interlaced decode.
///
/// The two fields of an interlaced stream are coded independently
/// (own seed, own bit stream), but the wire carries **no bottom-field
/// offset**: the bottom field starts at the top field's word-rounded
/// consumed-byte count, which is only known after walking the top
/// field's variable-length codeword stream. Field-parallel decode
/// therefore needs a split-finding prefix:
///
/// - **Worker (spawned):** full top-field decode, started immediately
///   at t=0 — it never waits on the split.
/// - **Caller thread:** runs [`scan_field`] over the top field — a
///   bit-exact *length-only* walk of the same codeword stream (same
///   tables, same slot order, same `BitReader`, no pixel stores, no
///   predictor passes, no output allocation) — to recover the split
///   point, then decodes the bottom field from there.
///
/// Wall clock goes from `top + bottom` to `max(top, scan + bottom)`;
/// the scan is cheaper than a full field decode (no raster writes, no
/// predictor inverse), so the parallel path wins whenever the fields
/// carry real work. Output and error behaviour are identical to the
/// serial path: the scan consumes bit-for-bit what the top-field
/// decode consumes (including `decode_one` failures on invalid
/// windows, which surface identically), the split is clamped to the
/// chunk length exactly like the serial path, and the top field's
/// result is inspected first so a malformed top field reports the
/// same error the serial path reports.
fn decode_frame_interlaced_parallel(
    config: &StreamConfig,
    frame_bytes: &[u8],
    tables: &ThreeTables,
) -> Result<DecodedFrame> {
    let h = config.height as usize;
    let top_h = h.div_ceil(2) as u32;
    let bot_h = (h / 2) as u32;
    if bot_h == 0 {
        // Single-field degenerate (forced-interlaced height 1): no
        // second unit to parallelise; keep the serial path.
        return decode_frame_interlaced(config, frame_bytes, tables);
    }
    let family = config.family;
    let predictor = config.method.predictor();
    let decorrelate = config.method.decorrelate();
    let width = config.width;
    let (top_joined, bot_result) = std::thread::scope(|s| {
        let top_handle = s.spawn(|| {
            decode_field(
                family,
                predictor,
                decorrelate,
                width,
                top_h,
                frame_bytes,
                tables,
            )
        });
        // Same clamp as the serial path: a malformed/truncated top
        // field can push the scan cursor past the end of the chunk;
        // the bottom field then sees the (possibly empty) remainder
        // and its own length guard rejects it cleanly.
        let bot_result = scan_field(family, decorrelate, width, top_h, frame_bytes, tables)
            .and_then(|top_consumed| {
                let split = top_consumed.min(frame_bytes.len());
                decode_field(
                    family,
                    predictor,
                    decorrelate,
                    width,
                    bot_h,
                    &frame_bytes[split..],
                    tables,
                )
            });
        (top_handle.join(), bot_result)
    });
    let (top_frame, _top_consumed) = match top_joined {
        Ok(res) => res?,
        Err(panic) => std::panic::resume_unwind(panic),
    };
    let (bot_frame, _bot_consumed) = bot_result?;
    // Round-261: single source of truth (`StreamConfig::row_bytes`)
    // for the family → wire-stride dispatch; spec/02 §3 wire-byte
    // layout table.
    let row_bytes = config.row_bytes();
    let merged = interleave_fields(&top_frame.pixels, &bot_frame.pixels, row_bytes, h);
    Ok(DecodedFrame {
        family: config.family,
        width: config.width,
        height: config.height,
        pixels: merged,
    })
}

/// Decode one field (or one full progressive frame).  Returns the
/// reconstructed pixels plus the byte count consumed from
/// `frame_bytes` (used by the interlaced path to find the start of
/// the bottom field).
fn decode_field(
    family: PixelFamily,
    predictor: Predictor,
    decorrelate: bool,
    width: u32,
    height: u32,
    frame_bytes: &[u8],
    tables: &ThreeTables,
) -> Result<(DecodedFrame, usize)> {
    match family {
        PixelFamily::Yuy2 => decode_yuy2_field(predictor, width, height, frame_bytes, tables),
        PixelFamily::Rgb24 => {
            decode_rgb24_field(predictor, decorrelate, width, height, frame_bytes, tables)
        }
        PixelFamily::Rgb32 => {
            decode_rgb32_field(predictor, decorrelate, width, height, frame_bytes, tables)
        }
    }
}

/// Round-420 split scanner: walk one field's entropy stream and
/// return the byte count `decode_field` would report as consumed —
/// WITHOUT reconstructing pixels, running predictor inverses, or
/// allocating an output raster.
///
/// This is the split-finding prefix of the two-worker interlaced
/// path: the wire carries no bottom-field offset, so the caller
/// thread scans the top field's codeword stream to learn where the
/// bottom field's seed starts while a second worker performs the full
/// top-field decode in parallel.
///
/// Bit-exactness contract (guarded by the round-420 scan-equivalence
/// tests): for every input on which `decode_field` succeeds, the scan
/// consumes the identical bit count — it replicates the per-family
/// dimension guards, the slot order of every codeword (spec/02 §3 /
/// spec/03 §1.4 wire cycles), and the shared `BitReader` word-rounded
/// [`BitReader::bytes_consumed`] tail — and for every input on which
/// `decode_field` fails inside the entropy loop, the scan fails with
/// the same error. (`decode_field` errors that fire *after* the
/// entropy loop — the RGB median-illegal check — are not replicated
/// here; the parallel caller surfaces them from the top-field decode
/// result, which it inspects first.)
///
/// The predictor never affects the bit stream (spec/02 §5: byte
/// ordering, channel-table assignment and bit packing are
/// predictor-independent), so the scanner takes no `Predictor`.
fn scan_field(
    family: PixelFamily,
    decorrelate: bool,
    width: u32,
    height: u32,
    frame_bytes: &[u8],
    tables: &ThreeTables,
) -> Result<usize> {
    match family {
        PixelFamily::Yuy2 => scan_yuy2_field(width, height, frame_bytes, tables),
        PixelFamily::Rgb24 | PixelFamily::Rgb32 => {
            scan_rgb_field(family, decorrelate, width, height, frame_bytes, tables)
        }
    }
}

/// YUY2 arm of [`scan_field`]. Mirrors `decode_yuy2_field`'s guards
/// and its macropixel-step entropy loop (slot cycle `+0 → slot1,
/// +1 → slot2, +2 → slot1, +3 → slot3`; paired window reads with the
/// sequential long-code fallback) with the stores elided.
fn scan_yuy2_field(
    width: u32,
    height: u32,
    frame_bytes: &[u8],
    tables: &ThreeTables,
) -> Result<usize> {
    let width_us = width as usize;
    let height_us = height as usize;
    if width_us % 2 != 0 {
        return Err(Error::invalid("YUY2 width must be even"));
    }
    let row_bytes = PixelFamily::Yuy2.row_bytes(width);
    let total_bytes = row_bytes * height_us;
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
    let mut reader = BitReader::new(&frame_bytes[4..]);
    let body_end = total_bytes - ((total_bytes - 4) & 3);
    let mut byte_idx = 4usize;
    // Round-420 pair-length LUTs: one 8-bit load resolves the summed
    // bit length of two codewords when the pair fits inside the 16
    // known window bits (`tables::build_pair_len_lut` soundness note);
    // `0` falls back to the exact decode_pair / decode_one walk the
    // decoder itself uses, so the consumed bit count stays identical.
    let lut12 = tables.scan_lut_s1s2();
    let lut13 = tables.scan_lut_s1s3();
    while byte_idx < body_end {
        // (Y₁ → slot1, U → slot2)
        let w = reader.peek_window();
        let s = lut12[(w >> 16) as usize];
        if s != 0 {
            reader.consume_bits_trusted(s as u32);
        } else if let Some((_, _, n)) = decode_pair(&tables.slot1, &tables.slot2, w) {
            reader.consume_bits_trusted(n);
        } else {
            let (_, len_y1) = decode_one(&tables.slot1, w)?;
            reader.consume_bits_trusted(len_y1 as u32);
            let (_, len_u) = decode_one(&tables.slot2, reader.peek_window())?;
            reader.consume_bits_trusted(len_u as u32);
        }
        // (Y₂ → slot1, V → slot3)
        let w = reader.peek_window();
        let s = lut13[(w >> 16) as usize];
        if s != 0 {
            reader.consume_bits_trusted(s as u32);
        } else if let Some((_, _, n)) = decode_pair(&tables.slot1, &tables.slot3, w) {
            reader.consume_bits_trusted(n);
        } else {
            let (_, len_y2) = decode_one(&tables.slot1, w)?;
            reader.consume_bits_trusted(len_y2 as u32);
            let (_, len_v) = decode_one(&tables.slot3, reader.peek_window())?;
            reader.consume_bits_trusted(len_v as u32);
        }
        byte_idx += 4;
    }
    // Scalar fall-through mirror of the decode loop's 1..=3-byte tail
    // (unreachable for in-spec YUY2 input; kept so the scan stays
    // bit-exact even on unforeseen future layouts).
    while byte_idx < total_bytes {
        let slot = match byte_idx % 4 {
            0 | 2 => &tables.slot1,
            1 => &tables.slot2,
            _ => &tables.slot3,
        };
        let (_, len) = decode_one(slot, reader.peek_window())?;
        reader.consume_bits_trusted(len as u32);
        byte_idx += 1;
    }
    Ok(4 + reader.bytes_consumed())
}

/// RGB arm of [`scan_field`], shared between RGB24 and RGB32.
/// Mirrors `decode_rgb24_field` / `decode_rgb32_field`: the same
/// dimension guards (via the same saturating
/// [`PixelFamily::row_bytes`] accessor), the same mode-resolved
/// first-two-position slot binding (spec/03 §1.4: `B, G` no-decorr /
/// `G, B−G` decorr), position +2 (R / R−G) via slot3, and — RGB32
/// only — position +3 (A) via slot3.
fn scan_rgb_field(
    family: PixelFamily,
    decorrelate: bool,
    width: u32,
    height: u32,
    frame_bytes: &[u8],
    tables: &ThreeTables,
) -> Result<usize> {
    debug_assert!(matches!(family, PixelFamily::Rgb24 | PixelFamily::Rgb32));
    let is_rgb32 = matches!(family, PixelFamily::Rgb32);
    let width_us = width as usize;
    let height_us = height as usize;
    let row_bytes = family.row_bytes(width);
    let total_bytes = row_bytes * height_us;
    if is_rgb32 {
        if total_bytes < 4 {
            return Err(Error::invalid(
                "RGB32 frame: degenerate dimensions (need ≥ 1 pixel)",
            ));
        }
    } else if total_bytes < 3 {
        return Err(Error::invalid(
            "RGB24 frame: degenerate dimensions (need ≥ 1 pixel)",
        ));
    }
    if frame_bytes.len() < 4 {
        return Err(Error::invalid(if is_rgb32 {
            "RGB32 frame: missing 4-byte uncompressed pixel"
        } else {
            "RGB24 frame: missing 4-byte uncompressed pixel"
        }));
    }
    let mut reader = BitReader::new(&frame_bytes[4..]);
    let (s_w0, s_w1) = if decorrelate {
        (&tables.slot2, &tables.slot1)
    } else {
        (&tables.slot1, &tables.slot2)
    };
    // Round-420 pair-length LUTs (see the YUY2 scanner note): the
    // first two wire positions resolve through the mode-matched pair
    // LUT, RGB32's (+2, +3) pair through the slot3/slot3 LUT.
    let lut01 = if decorrelate {
        tables.scan_lut_s2s1()
    } else {
        tables.scan_lut_s1s2()
    };
    let lut33 = if is_rgb32 {
        tables.scan_lut_s3s3()
    } else {
        &[][..]
    };
    let n_pixels = width_us * height_us;
    for _px in 1..n_pixels {
        let w = reader.peek_window();
        let s = lut01[(w >> 16) as usize];
        if s != 0 {
            reader.consume_bits_trusted(s as u32);
        } else if let Some((_, _, n)) = decode_pair(s_w0, s_w1, w) {
            reader.consume_bits_trusted(n);
        } else {
            let (_, l0) = decode_one(s_w0, w)?;
            reader.consume_bits_trusted(l0 as u32);
            let (_, l1) = decode_one(s_w1, reader.peek_window())?;
            reader.consume_bits_trusted(l1 as u32);
        }
        if !is_rgb32 {
            let (_, l2) = decode_one(&tables.slot3, reader.peek_window())?;
            reader.consume_bits_trusted(l2 as u32);
        } else {
            // Wire positions +2 (R or R−G) and +3 (A) both use the
            // slot-3 codebook (spec/03 §1.2; alpha shares slot 3).
            let w = reader.peek_window();
            let s = lut33[(w >> 16) as usize];
            if s != 0 {
                reader.consume_bits_trusted(s as u32);
            } else if let Some((_, _, n)) = decode_pair(&tables.slot3, &tables.slot3, w) {
                reader.consume_bits_trusted(n);
            } else {
                let (_, l2) = decode_one(&tables.slot3, w)?;
                reader.consume_bits_trusted(l2 as u32);
                let (_, l3) = decode_one(&tables.slot3, reader.peek_window())?;
                reader.consume_bits_trusted(l3 as u32);
            }
        }
    }
    Ok(4 + reader.bytes_consumed())
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
    /// Round-420: lazily-built pair-length LUTs for the interlaced
    /// split scanner (see [`ScanLuts`]). `Default`-initialised empty;
    /// built on first budget-≥ 2 interlaced decode of the stream and
    /// reused for its lifetime via the round-419 table cache.
    pub(crate) scan_luts: ScanLuts,
}

/// Lazily-built scan-side pair-length LUTs
/// ([`crate::tables::build_pair_len_lut`]), one per (first-slot,
/// second-slot) pair the field scanners consume:
///
/// - `s1s2` — YUY2 `(Y₁ → slot1, U → slot2)`; also RGB no-decorr
///   `(B → slot1, G → slot2)`.
/// - `s1s3` — YUY2 `(Y₂ → slot1, V → slot3)`.
/// - `s2s1` — RGB decorr `(G → slot2, B−G → slot1)`.
/// - `s3s3` — RGB32 `(R/R−G → slot3, A → slot3)`.
///
/// Each is 64 KiB and built at most once per cached `ThreeTables`
/// (`OnceLock`), only when the two-worker interlaced path actually
/// scans with that pair — progressive and serial decodes never pay
/// for them.
#[derive(Debug, Default)]
pub(crate) struct ScanLuts {
    s1s2: std::sync::OnceLock<Box<[u8]>>,
    s1s3: std::sync::OnceLock<Box<[u8]>>,
    s2s1: std::sync::OnceLock<Box<[u8]>>,
    s3s3: std::sync::OnceLock<Box<[u8]>>,
}

impl ThreeTables {
    fn scan_lut_s1s2(&self) -> &[u8] {
        self.scan_luts
            .s1s2
            .get_or_init(|| crate::tables::build_pair_len_lut(&self.slot1, &self.slot2))
    }
    fn scan_lut_s1s3(&self) -> &[u8] {
        self.scan_luts
            .s1s3
            .get_or_init(|| crate::tables::build_pair_len_lut(&self.slot1, &self.slot3))
    }
    fn scan_lut_s2s1(&self) -> &[u8] {
        self.scan_luts
            .s2s1
            .get_or_init(|| crate::tables::build_pair_len_lut(&self.slot2, &self.slot1))
    }
    fn scan_lut_s3s3(&self) -> &[u8] {
        self.scan_luts
            .s3s3
            .get_or_init(|| crate::tables::build_pair_len_lut(&self.slot3, &self.slot3))
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
        if let Ok(map) = extradata_cache().lock() {
            if let Some(hit) = map.get(table_bytes) {
                return Ok(Arc::clone(hit));
            }
        }
        let lengths = rle_decode_three_channels(table_bytes)?;
        let built = Arc::new(ThreeTables {
            slot1: HuffTable::build_from_lengths(&lengths[0])?,
            slot2: HuffTable::build_from_lengths(&lengths[1])?,
            slot3: HuffTable::build_from_lengths(&lengths[2])?,
            scan_luts: ScanLuts::default(),
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
                scan_luts: ScanLuts::default(),
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
            scan_luts: ScanLuts::default(),
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

fn decode_yuy2_field(
    predictor: Predictor,
    width: u32,
    height: u32,
    frame_bytes: &[u8],
    tables: &ThreeTables,
) -> Result<(DecodedFrame, usize)> {
    let width_us = width as usize;
    let height_us = height as usize;
    if width_us % 2 != 0 {
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
    debug_assert!(total_bytes >= 4);
    let body_end = total_bytes - ((total_bytes - 4) & 3);
    let mut byte_idx = 4usize;
    while byte_idx < body_end {
        // (Y₁ → slot1, U → slot2)
        let w = reader.peek_window();
        if let Some((sym_y1, sym_u, n)) = decode_pair(&tables.slot1, &tables.slot2, w) {
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
        if let Some((sym_y2, sym_v, n)) = decode_pair(&tables.slot1, &tables.slot3, w) {
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
            inverse_gradient_post(&mut pixels, row_bytes, height_us);
        }
        Predictor::Median => {
            inverse_yuy2_median(&mut pixels, row_bytes);
        }
    }

    let consumed = 4 + reader.bytes_consumed();
    Ok((
        DecodedFrame {
            family: PixelFamily::Yuy2,
            width,
            height,
            pixels,
        },
        consumed,
    ))
}

fn decode_rgb24_field(
    predictor: Predictor,
    decorrelate: bool,
    width: u32,
    height: u32,
    frame_bytes: &[u8],
    tables: &ThreeTables,
) -> Result<(DecodedFrame, usize)> {
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
    let n_pixels = width_us * height_us;
    for px in 1..n_pixels {
        let bgr_off = px * 3;
        let w = reader.peek_window();
        if let Some((v0, v1, n)) = decode_pair(s_w0, s_w1, w) {
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

    let consumed = 4 + reader.bytes_consumed();

    // Predictor inverse on the per-channel residual stream. Channels
    // are interleaved 3-byte BGR; LEFT walks each channel via stride
    // 3.
    inverse_left_row(&mut pixels, 3);
    match predictor {
        Predictor::Left => {}
        Predictor::Gradient => inverse_gradient_post(&mut pixels, row_bytes, height_us),
        Predictor::Median => {
            return Err(Error::invalid("median predictor not legal for RGB"));
        }
    }
    if decorrelate {
        // Inverse decorrelation: B = (B-G) + G, R = (R-G) + G.
        inverse_rgb_decorr_bgr(&mut pixels);
    }

    Ok((
        DecodedFrame {
            family: PixelFamily::Rgb24,
            width,
            height,
            pixels,
        },
        consumed,
    ))
}

fn decode_rgb32_field(
    predictor: Predictor,
    decorrelate: bool,
    width: u32,
    height: u32,
    frame_bytes: &[u8],
    tables: &ThreeTables,
) -> Result<(DecodedFrame, usize)> {
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
    for px in 1..n_pixels {
        let off = px * 4;
        let w = reader.peek_window();
        if let Some((v0, v1, n)) = decode_pair(s_w0, s_w1, w) {
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
        if let Some((v2, v3, n)) = decode_pair(&tables.slot3, &tables.slot3, w) {
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

    let consumed = 4 + reader.bytes_consumed();

    // Per-channel LEFT inverse, stride 4.
    inverse_left_row(&mut pixels, 4);
    match predictor {
        Predictor::Left => {}
        Predictor::Gradient => inverse_gradient_post(&mut pixels, row_bytes, height_us),
        Predictor::Median => {
            return Err(Error::invalid("median predictor not legal for RGB"));
        }
    }
    if decorrelate {
        inverse_rgb_decorr_bgra(&mut pixels);
    }

    Ok((
        DecodedFrame {
            family: PixelFamily::Rgb32,
            width,
            height,
            pixels,
        },
        consumed,
    ))
}

/// YUY2 MEDIAN inverse: spec/03 §2.3.2. LEFT-predicts row 0 + the
/// first 8 wire bytes of row 1, then per-byte median add for rows 1
/// pos ≥ 8 + every later row.
///
/// Round-202: strip the two dead branches the median tail-loop had
/// been carrying since round 1. After the LEFT-exemption pass the
/// loop start is at `row_bytes + 8` (capped at `len`), so for every
/// iteration `pos >= row_bytes + 8 >= 8 >= 2` and `pos - row_bytes >=
/// 8 >= 2` — i.e. neither `pos < 2 || pos < row_bytes` nor `pos <
/// row_bytes + 2` is ever true here, so both `if`-arms were dead.
/// The cleanup leaves a single straight-line per-byte median body
/// whose three lookback indices (`pos - 2`, `pos - row_bytes`, `pos -
/// row_bytes - 2`) are all provably in-bounds. Spec/03 §2.3.2 +
/// audit/01 §7.2 anchor the wire-byte LEFT exemption that makes
/// the AL-lookback safe without a per-iteration guard.
///
/// Round-208: drops three single-use decoder-local wrappers
/// (`inverse_left_per_channel`, `inverse_yuy2_left`,
/// `inverse_yuy2_left_range`) and consumes the public predict-side
/// helpers directly. `inverse_left_per_channel` was a byte-for-byte
/// duplicate of `predict::inverse_left_row`, and the two YUY2-LEFT
/// shims were thin pass-throughs into
/// `predict::inverse_yuy2_left_macropixel` already (left over from
/// the round-181 macropixel-step rewrite). Single source of truth
/// for both predictors now lives in `predict.rs`.
fn inverse_yuy2_median(out: &mut [u8], row_bytes: usize) {
    let len = out.len();
    if len <= 4 {
        return;
    }
    // Row 0 LEFT pass.
    let row0_end = row_bytes.min(len);
    inverse_yuy2_left_macropixel(out, 4, row0_end);
    // First 8 bytes of row 1 (LEFT exemption).
    if len <= row_bytes {
        return;
    }
    let row1_left_end = (row_bytes + 8).min(len);
    inverse_yuy2_left_macropixel(out, row_bytes, row1_left_end);
    if row1_left_end >= len {
        return;
    }
    // Median per-byte add for the remaining bytes of row 1 + all
    // rows ≥ 2. spec/03 §2.3 trace: L at -2, A at -row_stride,
    // AL at -row_stride - 2. With `row1_left_end >= row_bytes + 8`,
    // every lookback below is in-bounds.
    //
    // Round-255: rewrite as a half-macropixel (2-byte) step body to
    // expose instruction-level parallelism between the two independent
    // median computations within each step, mirroring the predict-side
    // r253 macropixel-step rewrite shape but at half the unroll factor
    // (the inverse predictor reads L from the same `out` buffer it
    // writes to, so the L source at intra-step offset +2 / +3 would
    // alias the writes at intra-step offset +0 / +1 — a 4-byte unroll
    // would introduce a read-after-write within the unrolled body).
    //
    // At a 2-byte step, both L sources within the step
    // (`out[pos - 2]`, `out[pos - 1]`) are read from positions
    // finalised before the step began — the previous step's writes
    // (or the LEFT-region bytes, for the very first step at
    // `pos = row_bytes + 8`). The two `gradient_predictor → median3
    // → wrapping_add` chains for the step's two output bytes are
    // therefore independent and the compiler is free to schedule
    // them across functional units.
    //
    // `row1_left_end = (row_bytes + 8).min(len)` is always a multiple
    // of 2 for in-spec YUY2 input (row_bytes = 2 × width ⇒ row_bytes
    // ≡ 0 mod 2, plus the +8 keeps the alignment), so the 2-byte
    // step body covers every wire byte in the median region. A 1-byte
    // scalar fall-through is kept for defence-in-depth, mirroring the
    // r221 / r227 / r239 / r242 / r245 / r250 / r253 fall-throughs.
    //
    // Bit-identical to the pre-r255 per-byte body — regression-guarded
    // by `round255_inverse_yuy2_median_step_*` tests below.
    debug_assert!(row1_left_end >= row_bytes + 2);
    let body_end = len - ((len - row1_left_end) & 1);
    let mut pos = row1_left_end;
    while pos < body_end {
        // Two independent per-byte median adds per outer step. Each
        // pair reads from disjoint reference offsets and writes to
        // disjoint output offsets; the writes do not feed back into
        // the reads within the same step.
        let l0 = out[pos - 2];
        let a0 = out[pos - row_bytes];
        let al0 = out[pos - row_bytes - 2];
        let l1 = out[pos - 1];
        let a1 = out[pos + 1 - row_bytes];
        let al1 = out[pos - row_bytes - 1];
        let g0 = l0.wrapping_add(a0).wrapping_sub(al0);
        let g1 = l1.wrapping_add(a1).wrapping_sub(al1);
        let p0 = crate::predict::median3(l0, a0, g0);
        let p1 = crate::predict::median3(l1, a1, g1);
        out[pos] = out[pos].wrapping_add(p0);
        out[pos + 1] = out[pos + 1].wrapping_add(p1);
        pos += 2;
    }
    // Tail (0..=1 bytes after the last whole 2-byte step). In practice
    // in-spec YUY2 input keeps `len − row1_left_end` even, so this
    // loop runs zero times; kept for robustness.
    while pos < len {
        let l = out[pos - 2];
        let a = out[pos - row_bytes];
        let al = out[pos - row_bytes - 2];
        let g = l.wrapping_add(a).wrapping_sub(al);
        let predictor = crate::predict::median3(l, a, g);
        out[pos] = out[pos].wrapping_add(predictor);
        pos += 1;
    }
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
    fn interlaced_truncated_top_field_no_panic() {
        // height > 288 engages the interlaced path; a tiny frame body
        // makes the top field over-consume so the bottom-field split
        // point would exceed the buffer. Must not panic.
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
    use crate::encoder::{encode_frame_with_mode, ExtradataMode};
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
                    crate::predict::inverse_gradient_post(&mut ref_pixels, row_bytes, h as usize);
                }
                Method::Median => {
                    super::inverse_yuy2_median(&mut ref_pixels, row_bytes);
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
mod round255_inverse_yuy2_median_step_tests {
    //! Round-255 regression guard. The decoder's
    //! [`super::inverse_yuy2_median`] median-region body was rewritten
    //! from a per-byte add loop into a half-macropixel (2-byte) step
    //! body, mirroring r253's predict-side macropixel-step rewrite shape
    //! at half the unroll factor.
    //!
    //! The inverse predictor reads `L` from the same `out` buffer it
    //! writes to. A 4-byte unroll like r253's predict-side body would
    //! introduce read-after-write aliasing (intra-step offset +2 / +3
    //! reads `out[pos] / out[pos + 1]`, which are the writes from
    //! intra-step offset +0 / +1). The 2-byte step body avoids this:
    //! both L sources within a step (`out[pos - 2]`, `out[pos - 1]`)
    //! are finalised before the step begins, so the two median chains
    //! within a step are independent and the compiler can schedule
    //! them across functional units. Successive 2-byte steps still
    //! carry the natural sequential L dependency the median predictor
    //! requires per spec/03 §2.3.
    //!
    //! These tests pin the rewrite at byte-equality against an inlined
    //! copy of the pre-r255 per-byte body, cover the modular-wrap edge,
    //! pin the `(len − row1_left_end) % 2 == 0` alignment invariant the
    //! step body relies on, exercise the 1-byte scalar tail
    //! fall-through, and run the encoder→decoder roundtrip across the
    //! same YUY2 widths the r253 forward tests cover (so any drift
    //! between forward and inverse predictor would surface as a
    //! roundtrip mismatch even before the per-byte witness fires).
    use crate::encoder::{encode_frame_with_mode, ExtradataMode};
    use crate::header::{Method, PixelFamily, StreamConfig};
    use crate::predict::median3;

    /// Inlined copy of the pre-r255 decoder median body, used as the
    /// per-byte reference oracle. Identical semantics to the
    /// production code path before the 2-byte step rewrite.
    fn ref_inverse_yuy2_median_per_byte(out: &mut [u8], row_bytes: usize) {
        let len = out.len();
        if len <= 4 {
            return;
        }
        let row0_end = row_bytes.min(len);
        crate::predict::inverse_yuy2_left_macropixel(out, 4, row0_end);
        if len <= row_bytes {
            return;
        }
        let row1_left_end = (row_bytes + 8).min(len);
        crate::predict::inverse_yuy2_left_macropixel(out, row_bytes, row1_left_end);
        if row1_left_end >= len {
            return;
        }
        for pos in row1_left_end..len {
            let l = out[pos - 2];
            let a = out[pos - row_bytes];
            let al = out[pos - row_bytes - 2];
            let g = l.wrapping_add(a).wrapping_sub(al);
            let predictor = median3(l, a, g);
            out[pos] = out[pos].wrapping_add(predictor);
        }
    }

    /// Deterministic byte stream — the median-region body is exercised
    /// regardless of input statistics because every input byte feeds
    /// into the lookback chain on subsequent rows.
    fn synth_residuals(n: usize, seed: u32) -> Vec<u8> {
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

    #[test]
    fn round255_inverse_yuy2_median_matches_per_byte_reference() {
        // The 2-byte step body must produce byte-identical output to
        // the pre-r255 per-byte loop across the YUY2 widths the r253
        // forward body covers: narrow widths where row_bytes < 8 (the
        // LEFT exemption extends past row 1 into row 2), the
        // bench-reference 320×16 raster, and intermediate sizes.
        for (w, h) in [
            (2usize, 18usize),
            (4, 4),
            (8, 6),
            (16, 3),
            (160, 8),
            (320, 16),
        ] {
            let row_bytes = w * 2;
            let n = row_bytes * h;
            let buf = synth_residuals(n, 0xCAFE_BABE);
            let mut a = buf.clone();
            let mut b = buf;
            ref_inverse_yuy2_median_per_byte(&mut a, row_bytes);
            super::inverse_yuy2_median(&mut b, row_bytes);
            assert_eq!(
                a, b,
                "2-byte step body must match per-byte reference (w={w}, h={h})"
            );
        }
    }

    #[test]
    fn round255_inverse_yuy2_median_modular_wrap() {
        // Force mod-256 wraps inside the gradient and the final add by
        // priming the entire buffer with 0xFE / 0xFF / 0x01 / 0x02
        // values that overflow when added in the predictor chain.
        let row_bytes = 16;
        let h = 6;
        let n = row_bytes * h;
        let mut buf = Vec::with_capacity(n);
        for i in 0..n {
            buf.push(match i & 3 {
                0 => 0xFEu8,
                1 => 0xFFu8,
                2 => 0x01u8,
                _ => 0x02u8,
            });
        }
        let mut a = buf.clone();
        let mut b = buf;
        ref_inverse_yuy2_median_per_byte(&mut a, row_bytes);
        super::inverse_yuy2_median(&mut b, row_bytes);
        assert_eq!(a, b, "modular wrap path must match per-byte reference");
    }

    #[test]
    fn round255_inverse_yuy2_median_boundary_row1_step_alignment() {
        // Pin the alignment invariants the 2-byte step body relies on:
        // - `row1_left_end = (row_bytes + 8).min(len)` must be a
        //   multiple of 2 for in-spec YUY2 input (row_bytes = 2 * width
        //   with width even ⇒ row_bytes ≡ 0 mod 2; the +8 keeps the
        //   alignment).
        // - `(len - row1_left_end) % 2 == 0` ⇒ the scalar 1-byte tail
        //   fall-through runs zero times for every in-spec width.
        for w in [2usize, 4, 8, 16, 160, 320] {
            let row_bytes = w * 2;
            let h = 4;
            let len = row_bytes * h;
            let row1_left_end = (row_bytes + 8).min(len);
            assert_eq!(
                row1_left_end & 1,
                0,
                "row1_left_end must be even for in-spec YUY2 (w={w})"
            );
            assert_eq!(
                (len - row1_left_end) & 1,
                0,
                "(len - row1_left_end) must be even for in-spec YUY2 (w={w})"
            );
        }
    }

    #[test]
    fn round255_inverse_yuy2_median_scalar_tail_safety() {
        // Force the 1-byte scalar fall-through by constructing an
        // out-of-spec buffer where (len - row1_left_end) is odd. Even
        // though in-spec YUY2 widths keep this region even, the tail
        // loop must remain memory-safe and produce the same output as
        // the per-byte reference body — this is the defence-in-depth
        // claim the rewrite makes.
        let row_bytes = 16;
        let h = 5;
        let len = row_bytes * h - 1; // one byte short of full row
        let buf = synth_residuals(len, 0xDEAD_BEEF);
        let mut a = buf.clone();
        let mut b = buf;
        ref_inverse_yuy2_median_per_byte(&mut a, row_bytes);
        super::inverse_yuy2_median(&mut b, row_bytes);
        assert_eq!(a, b, "scalar tail fall-through must match reference");
    }

    #[test]
    fn round255_inverse_yuy2_median_encoder_roundtrip() {
        // End-to-end witness: encode → decode must round-trip
        // bit-exactly across the same widths the r253 forward tests
        // cover. The production decoder now routes the median region
        // through the 2-byte step body — any drift between the forward
        // predict and the inverse would surface here.
        for (w, h) in [(2u32, 18u32), (4, 6), (8, 5), (16, 3)] {
            // Synthesise pixels and roundtrip through ClassicV2 Median.
            let pixels = synth_residuals((w as usize) * (h as usize) * 2, 0x1234_5678);
            let (bih, frame) = encode_frame_with_mode(
                PixelFamily::Yuy2,
                Method::Median,
                w,
                h,
                &pixels,
                ExtradataMode::ClassicV2,
            )
            .expect("encode");
            let cfg = StreamConfig::parse_bitmapinfoheader(&bih).expect("parse");
            let decoded = super::decode_frame(&cfg, &frame).expect("decode");
            assert_eq!(
                decoded.pixels, pixels,
                "round-trip must be bit-exact (w={w}, h={h})"
            );
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

    use crate::encoder::{encode_frame_with_mode, ExtradataMode};
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
    use crate::encoder::{encode_frame_with_mode, ExtradataMode};
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
        inverse_left_row(out, n_channels);
        if matches!(method.predictor(), Predictor::Gradient) {
            inverse_gradient_post(out, row_bytes, height);
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
mod round420_field_parallel_tests {
    //! Round-420 regression guard for the two-worker interlaced decode
    //! path (`decode_frame_with_workers` / `scan_field`).
    //!
    //! Two invariants are pinned:
    //!
    //! 1. **Scan bit-exactness.** [`super::scan_field`] must report the
    //!    identical consumed-byte count [`super::decode_field`] reports
    //!    for the same field bytes — the split point the parallel path
    //!    hands to the bottom-field worker IS this value, so any drift
    //!    would decode the bottom field from a garbage offset. Covered
    //!    across the full (family, method, extradata-mode) matrix on
    //!    progressive frames (a progressive frame is one field), on
    //!    both fields of interlaced frames, and on the v1.x table path.
    //! 2. **Budget invariance.** `decode_frame_with_workers(cfg, f, 2)`
    //!    must produce byte-identical pixels to the serial
    //!    `decode_frame` on every interlaced stream, including odd
    //!    heights (unequal field heights) and forced-interlace tiny
    //!    heights, and must fail (not panic, not diverge) exactly when
    //!    the serial path fails on malformed input.

    use super::*;
    use crate::encoder::{encode_frame_with_mode, ExtradataMode};
    use crate::header::{Method, StreamConfig};

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

    fn bpp(family: PixelFamily) -> usize {
        match family {
            PixelFamily::Yuy2 => 2,
            PixelFamily::Rgb24 => 3,
            PixelFamily::Rgb32 => 4,
        }
    }

    fn legal_methods(family: PixelFamily) -> &'static [Method] {
        match family {
            PixelFamily::Yuy2 => &[
                Method::PredictOld,
                Method::Left,
                Method::Gradient,
                Method::Median,
            ],
            PixelFamily::Rgb24 | PixelFamily::Rgb32 => &[
                Method::PredictOld,
                Method::Left,
                Method::LeftDecorr,
                Method::GradientDecorr,
            ],
        }
    }

    const FAMILIES: [PixelFamily; 3] = [PixelFamily::Yuy2, PixelFamily::Rgb24, PixelFamily::Rgb32];
    const MODES: [ExtradataMode; 3] = [
        ExtradataMode::ClassicV2,
        ExtradataMode::CustomV2,
        ExtradataMode::V1xCompat,
    ];

    /// Scan == decode consumed-bytes, progressive frames (one field),
    /// full method/mode matrix.
    #[test]
    fn round420_scan_matches_decode_consumed_progressive_matrix() {
        for family in FAMILIES {
            for &method in legal_methods(family) {
                for mode in MODES {
                    // All widths even so the YUY2 macropixel guard is
                    // satisfied for every family.
                    for (w, h) in [(2u32, 2u32), (4, 4), (16, 3), (48, 32)] {
                        let pixels = synth((w as usize) * (h as usize) * bpp(family), 0x0420_0001);
                        let (bih, frame) =
                            encode_frame_with_mode(family, method, w, h, &pixels, mode)
                                .expect("encode");
                        let cfg = StreamConfig::parse_bitmapinfoheader(&bih).expect("parse");
                        let tables = build_three_tables(&cfg).expect("tables");
                        let (_, consumed) = decode_field(
                            family,
                            method.predictor(),
                            method.decorrelate(),
                            w,
                            h,
                            &frame,
                            &tables,
                        )
                        .expect("decode_field");
                        let scanned =
                            scan_field(family, method.decorrelate(), w, h, &frame, &tables)
                                .expect("scan_field");
                        assert_eq!(
                            scanned, consumed,
                            "scan/decode consumed drift: {family:?}/{method:?}/{mode:?}/{w}x{h}"
                        );
                    }
                }
            }
        }
    }

    /// Scan == decode consumed-bytes on BOTH fields of interlaced
    /// frames (the split point itself, then the bottom field's tail).
    #[test]
    fn round420_scan_matches_decode_consumed_interlaced_fields() {
        for family in FAMILIES {
            for &method in legal_methods(family) {
                // Odd height: top field is one row taller than bottom.
                for (w, h) in [(32u32, 292u32), (16, 293)] {
                    let pixels = synth((w as usize) * (h as usize) * bpp(family), 0x0420_0002);
                    let (bih, frame) = encode_frame_with_mode(
                        family,
                        method,
                        w,
                        h,
                        &pixels,
                        ExtradataMode::ClassicV2,
                    )
                    .expect("encode");
                    let cfg = StreamConfig::parse_bitmapinfoheader(&bih).expect("parse");
                    assert!(cfg.is_interlaced(), "test premise: h > 288 is interlaced");
                    let tables = build_three_tables(&cfg).expect("tables");
                    let top_h = (h as usize).div_ceil(2) as u32;
                    let bot_h = h / 2;
                    let (_, top_consumed) = decode_field(
                        family,
                        method.predictor(),
                        method.decorrelate(),
                        w,
                        top_h,
                        &frame,
                        &tables,
                    )
                    .expect("decode top");
                    let top_scanned =
                        scan_field(family, method.decorrelate(), w, top_h, &frame, &tables)
                            .expect("scan top");
                    assert_eq!(
                        top_scanned, top_consumed,
                        "top-field split drift: {family:?}/{method:?}/{w}x{h}"
                    );
                    let rest = &frame[top_consumed.min(frame.len())..];
                    let (_, bot_consumed) = decode_field(
                        family,
                        method.predictor(),
                        method.decorrelate(),
                        w,
                        bot_h,
                        rest,
                        &tables,
                    )
                    .expect("decode bot");
                    let bot_scanned =
                        scan_field(family, method.decorrelate(), w, bot_h, rest, &tables)
                            .expect("scan bot");
                    assert_eq!(
                        bot_scanned, bot_consumed,
                        "bottom-field consumed drift: {family:?}/{method:?}/{w}x{h}"
                    );
                }
            }
        }
    }

    /// Scan replicates decode_field's guard errors on degenerate
    /// dimensions and truncated input.
    #[test]
    fn round420_scan_replicates_guard_errors() {
        let tables = table_cache::v1x_tables(PixelFamily::Yuy2).expect("tables");
        // Odd YUY2 width.
        assert!(scan_field(PixelFamily::Yuy2, false, 3, 4, &[0u8; 64], &tables).is_err());
        // Zero-height rasters.
        assert!(scan_field(PixelFamily::Yuy2, false, 4, 0, &[0u8; 64], &tables).is_err());
        // Input shorter than the 4-byte seed.
        assert!(scan_field(PixelFamily::Yuy2, false, 4, 4, &[0u8; 3], &tables).is_err());
        let rgb_tables = table_cache::v1x_tables(PixelFamily::Rgb24).expect("tables");
        assert!(scan_field(PixelFamily::Rgb24, false, 4, 0, &[0u8; 64], &rgb_tables).is_err());
        assert!(scan_field(PixelFamily::Rgb32, true, 0, 4, &[0u8; 64], &rgb_tables).is_err());
        assert!(scan_field(PixelFamily::Rgb32, false, 4, 4, &[0u8; 2], &rgb_tables).is_err());
    }

    /// Budget-2 decode must be byte-identical to serial decode on
    /// interlaced streams across the method/mode matrix, including an
    /// odd interlaced height (top field one row taller).
    #[test]
    fn round420_workers2_matches_serial_interlaced_matrix() {
        for family in FAMILIES {
            for &method in legal_methods(family) {
                for mode in MODES {
                    for (w, h) in [(32u32, 292u32), (16, 293), (48, 289)] {
                        let pixels = synth((w as usize) * (h as usize) * bpp(family), 0x0420_0003);
                        let (bih, frame) =
                            encode_frame_with_mode(family, method, w, h, &pixels, mode)
                                .expect("encode");
                        let cfg = StreamConfig::parse_bitmapinfoheader(&bih).expect("parse");
                        assert!(cfg.is_interlaced());
                        let serial = decode_frame(&cfg, &frame).expect("serial decode");
                        let parallel =
                            decode_frame_with_workers(&cfg, &frame, 2).expect("parallel decode");
                        assert_eq!(
                            serial.pixels, parallel.pixels,
                            "budget-2 divergence: {family:?}/{method:?}/{mode:?}/{w}x{h}"
                        );
                        assert_eq!(
                            parallel.pixels, pixels,
                            "budget-2 lossless drift: {family:?}/{method:?}/{mode:?}/{w}x{h}"
                        );
                    }
                }
            }
        }
    }

    /// Budgets 0 / 1 route through the serial path; budgets > 2 clamp
    /// to the two field units. All must equal serial output.
    #[test]
    fn round420_budget_clamp_all_equal_serial() {
        let (w, h) = (32u32, 292u32);
        let pixels = synth((w as usize) * (h as usize) * 2, 0x0420_0004);
        let (bih, frame) = encode_frame_with_mode(
            PixelFamily::Yuy2,
            Method::Median,
            w,
            h,
            &pixels,
            ExtradataMode::ClassicV2,
        )
        .expect("encode");
        let cfg = StreamConfig::parse_bitmapinfoheader(&bih).expect("parse");
        let serial = decode_frame(&cfg, &frame).expect("decode");
        for budget in [0usize, 1, 2, 3, 8, 64] {
            let out = decode_frame_with_workers(&cfg, &frame, budget).expect("decode");
            assert_eq!(out.pixels, serial.pixels, "budget {budget} diverged");
        }
    }

    /// Progressive streams ignore the budget entirely (single unit).
    #[test]
    fn round420_progressive_budget_is_noop() {
        let (w, h) = (48u32, 32u32);
        let pixels = synth((w as usize) * (h as usize) * 3, 0x0420_0005);
        let (bih, frame) = encode_frame_with_mode(
            PixelFamily::Rgb24,
            Method::LeftDecorr,
            w,
            h,
            &pixels,
            ExtradataMode::CustomV2,
        )
        .expect("encode");
        let cfg = StreamConfig::parse_bitmapinfoheader(&bih).expect("parse");
        let serial = decode_frame(&cfg, &frame).expect("decode");
        let budgeted = decode_frame_with_workers(&cfg, &frame, 8).expect("decode");
        assert_eq!(serial.pixels, budgeted.pixels);
    }

    /// Malformed interlaced input: the parallel path must fail exactly
    /// when the serial path fails (same Ok/Err disposition, no panic),
    /// mirroring `degenerate_dims_tests::interlaced_truncated_top_field_no_panic`.
    #[test]
    fn round420_workers2_truncated_matches_serial_disposition() {
        let cfg = StreamConfig {
            family: PixelFamily::Yuy2,
            method: Method::Left,
            width: 2,
            height: 290,
            has_extradata: false,
            extradata_tables: Vec::new(),
            interlace_flag: 0,
        };
        for len in [0usize, 3, 4, 5, 8, 16, 64] {
            let frame = vec![0u8; len];
            let serial = decode_frame(&cfg, &frame);
            let parallel = decode_frame_with_workers(&cfg, &frame, 2);
            match (&serial, &parallel) {
                (Ok(a), Ok(b)) => assert_eq!(a.pixels, b.pixels, "len {len}: Ok pixels diverged"),
                (Err(_), Err(_)) => {}
                _ => panic!(
                    "len {len}: serial/parallel disposition diverged: serial {}, parallel {}",
                    if serial.is_ok() { "Ok" } else { "Err" },
                    if parallel.is_ok() { "Ok" } else { "Err" },
                ),
            }
        }
    }

    /// Truncated-mid-stream interlaced frames (cut at every word
    /// boundary of a real stream) keep serial/parallel dispositions in
    /// lockstep, and Ok outputs byte-identical.
    #[test]
    fn round420_workers2_word_truncation_sweep() {
        let (w, h) = (16u32, 292u32);
        let pixels = synth((w as usize) * (h as usize) * 2, 0x0420_0006);
        let (bih, frame) = encode_frame_with_mode(
            PixelFamily::Yuy2,
            Method::Left,
            w,
            h,
            &pixels,
            ExtradataMode::ClassicV2,
        )
        .expect("encode");
        let cfg = StreamConfig::parse_bitmapinfoheader(&bih).expect("parse");
        let mut cut = 0usize;
        while cut < frame.len() {
            let sub = &frame[..cut];
            let serial = decode_frame(&cfg, sub);
            let parallel = decode_frame_with_workers(&cfg, sub, 2);
            match (&serial, &parallel) {
                (Ok(a), Ok(b)) => assert_eq!(a.pixels, b.pixels, "cut {cut}: pixels diverged"),
                (Err(_), Err(_)) => {}
                _ => panic!("cut {cut}: serial/parallel disposition diverged"),
            }
            cut += 32; // word-multiple stride keeps the sweep bounded
        }
    }
}

#[cfg(test)]
mod round420_pair_len_lut_tests {
    //! Round-420 soundness guard for the scan-side pair-length LUTs
    //! (`tables::build_pair_len_lut`): a non-zero entry claims that
    //! EVERY 32-bit window sharing the entry's 16-bit prefix decodes
    //! its next two codewords to that exact combined length. Verify
    //! the claim against the production `decode_pair` /
    //! `decode_one` walk across randomised low window bits, for all
    //! four (first, second) slot pairs the scanners consume, on both
    //! the v1.x compiled-in tables and a histogram-derived CustomV2
    //! set (whose code lengths differ per slot).

    use super::*;
    use crate::tables::{build_pair_len_lut, decode_one, decode_pair};

    fn check_pair(a: &HuffTable, b: &HuffTable, label: &str) {
        let lut = build_pair_len_lut(a, b);
        let mut s: u32 = 0x0420_5eed;
        for (p, &sum) in lut.iter().enumerate() {
            if sum == 0 {
                continue;
            }
            for _ in 0..4 {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                let w = ((p as u32) << 16) | (s & 0xFFFF);
                // Reference: the exact two-symbol walk the decoder
                // (and the scanner's fallback) performs.
                let n = if let Some((_, _, n)) = decode_pair(a, b, w) {
                    n
                } else {
                    let (_, l0) = decode_one(a, w).expect("code A must decode");
                    // Second window after consuming l0 bits: low bits
                    // beyond the 32-bit window are unknown to this
                    // test, but sum ≤ 16 guarantees code B lies inside
                    // the first 16 bits, so shifting w suffices.
                    let (_, l1) = decode_one(b, w << l0).expect("code B must decode within window");
                    (l0 + l1) as u32
                };
                assert_eq!(
                    n,
                    sum as u32,
                    "{label}: prefix 0x{p:04x} low 0x{:04x}: LUT sum {sum} vs walk {n}",
                    w & 0xFFFF
                );
            }
        }
    }

    #[test]
    fn round420_pair_len_lut_sound_v1x_tables() {
        let yuv = table_cache::v1x_tables(PixelFamily::Yuy2).expect("tables");
        check_pair(&yuv.slot1, &yuv.slot2, "v1x yuv s1s2");
        check_pair(&yuv.slot1, &yuv.slot3, "v1x yuv s1s3");
        let rgb = table_cache::v1x_tables(PixelFamily::Rgb24).expect("tables");
        check_pair(&rgb.slot2, &rgb.slot1, "v1x rgb s2s1");
        check_pair(&rgb.slot3, &rgb.slot3, "v1x rgb s3s3");
    }

    #[test]
    fn round420_pair_len_lut_sound_custom_tables() {
        use crate::encoder::{encode_frame_with_mode, ExtradataMode};
        use crate::header::Method;
        // A CustomV2 encode derives per-channel tables from the frame
        // histograms — three genuinely distinct codebooks.
        let mut s: u32 = 0xBEEF_0042;
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
    }
}
