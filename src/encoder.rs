//! HuffYUV / FFVHuff frame encoder.
//!
//! Round-2 deliverable: produces wire-conformant frames that
//! [`crate::decoder::decode_frame`] (and any third-party HuffYUV
//! decoder) can reconstruct losslessly. Three modes are supported:
//!
//! - [`ExtradataMode::ClassicV2`] — embed one of the six pre-baked
//!   classic blobs (the binary's default). spec/04 §3.
//! - [`ExtradataMode::CustomV2`] — derive per-channel histograms from
//!   `pixels`, run the package-merge length-limited Huffman builder,
//!   and RLE-encode the resulting length tables in-line. spec/03 §3 +
//!   spec/01 §5.
//! - [`ExtradataMode::V1xCompat`] — emit `biSize == 0x28` with no
//!   extradata; the decoder reads its codebook from the v1.x
//!   precomputed-codes set. spec/01 §6.
//!
//! All three encode predict_old / Left / Gradient / Median / LeftDecorr
//! / GradientDecorr per `spec/03 §2` for the legal (family, method)
//! pairs.

use crate::bitio::BitWriter;
use crate::error::{Error, Result};
use crate::header::{FOURCC_HFYU, Method, PixelFamily, Predictor, StreamConfig};
use crate::predict::{
    forward_decorr_gradient_stride, forward_gradient_stride, forward_left_decorr_residuals,
    forward_median_yuy2, forward_rgb_left_subtract_linear, forward_yuy2_left_subtract,
    is_interlaced_height,
};
#[cfg(test)]
use crate::tables::HuffTable;
use crate::tables::{
    HuffEntry, classic_blob_bytes, compute_canonical_lengths, rle_encode_three_channels,
};

/// Round-440: the emit pipeline's per-slot table view. The encoder
/// only ever reads per-symbol `(code, length)` pairs — never the
/// 64-Ki primary decode LUT or the overflow slice a full
/// [`HuffTable`] carries — so every emit-side helper takes the bare
/// entries array. ClassicV2 / V1xCompat borrow `.entries` out of the
/// shared decode-table cache; CustomV2 builds the entries alone
/// (`tables::canonical_entries_from_lengths`), skipping the dead LUT
/// bake it used to pay per frame.
type SlotEntries = [HuffEntry; 256];

/// Selects which BIH/extradata path the encoder writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum ExtradataMode {
    /// v2.x with the classic-table seed for this `(family, method)`
    /// pair (matches the proprietary's default; see spec/04 §3.4).
    #[default]
    ClassicV2,
    /// v2.x with per-frame custom-built length tables (package-merge,
    /// max-length 31). Pure-Rust path that doesn't need the classic
    /// blobs in the BIH.
    CustomV2,
    /// v1.x compatibility: `biSize == 0x28`, no extradata; the decoder
    /// uses the precomputed v1.x codebooks from `tables/06..09`. Only
    /// `predict_old` / `Left` are loss-of-fidelity-free here, and the
    /// length tables MUST NOT assign length-0 to any symbol the
    /// residual stream emits.
    V1xCompat,
}

/// How the encoder decides whether a frame is field-coded (spec/06).
///
/// A field-coded frame is predicted with "above" two frame rows back
/// (spec/06 §2.1) and announced by the extradata interlace byte
/// `0x10`; a plain frame carries `0x20` (spec/06 §1.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum FieldCoding {
    /// The original codec's rule: field-code exactly when the height
    /// exceeds 288 lines (spec/06 §1.1). Every decoder agrees on such
    /// a stream, whether it follows the interlace byte or the height.
    #[default]
    Auto,
    /// Field-code whatever the height (interlaced material of 288
    /// lines or fewer). Decoders that go by the height alone — the
    /// x86-64 build of the original codec, spec/06 §1.3 — mis-read
    /// such a stream at 288 lines or fewer.
    Fields,
    /// Plain raster whatever the height (progressive material taller
    /// than 288 lines). Decoders that go by the height alone mis-read
    /// such a stream above 288 lines.
    Plain,
}

impl FieldCoding {
    /// Resolve the choice for a frame of `height` lines written with
    /// `mode`. A `biSize == 0x28` header has no interlace byte, so a
    /// decoder can only apply the height rule to it: a choice that
    /// disagrees with that rule is rejected for
    /// [`ExtradataMode::V1xCompat`].
    pub fn resolve(self, height: u32, mode: ExtradataMode) -> Result<bool> {
        let by_height = is_interlaced_height(height);
        let wanted = match self {
            FieldCoding::Auto => by_height,
            FieldCoding::Fields => true,
            FieldCoding::Plain => false,
        };
        if wanted != by_height && mode == ExtradataMode::V1xCompat {
            return Err(Error::unsupported(
                "field coding against the 288-line rule needs the extradata interlace \
                 byte, which a v1.x header does not carry",
            ));
        }
        Ok(wanted)
    }
}

/// Options for [`encode_frame_with_options`].
#[derive(Debug, Clone, Copy, Default)]
#[non_exhaustive]
pub struct EncodeOptions {
    /// Predictor choice. Defaults to [`MethodSelection::Auto`].
    pub method: MethodSelection,
    /// Header / table layout. Defaults to [`ExtradataMode::ClassicV2`].
    pub mode: ExtradataMode,
    /// Field-coding decision. Defaults to [`FieldCoding::Auto`].
    pub field_coding: FieldCoding,
    /// Advisory thread budget; `0` and `1` both mean serial. Only the
    /// auto-selector's candidate scoring fans out.
    pub worker_budget: usize,
}

impl EncodeOptions {
    /// Auto-selected predictor, classic tables, height-rule field
    /// coding, serial.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set `method`.
    pub fn with_method(mut self, v: MethodSelection) -> Self {
        self.method = v;
        self
    }

    /// Set `mode`.
    pub fn with_mode(mut self, v: ExtradataMode) -> Self {
        self.mode = v;
        self
    }

    /// Set `field_coding`.
    pub fn with_field_coding(mut self, v: FieldCoding) -> Self {
        self.field_coding = v;
        self
    }

    /// Set `worker_budget`.
    pub fn with_worker_budget(mut self, v: usize) -> Self {
        self.worker_budget = v;
        self
    }
}

/// Result of [`encode_frame_with_options`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct EncodedFrame {
    /// AVI `strf` payload (`BITMAPINFOHEADER` + extradata).
    pub strf: Vec<u8>,
    /// One `00dc` chunk body.
    pub frame: Vec<u8>,
    /// The method written into the header.
    pub method: Method,
}

/// Encode one frame with every encoder choice in one options record.
///
/// With default options this is [`encode_frame_auto`] in
/// [`ExtradataMode::ClassicV2`]; `options.field_coding` additionally
/// lets a caller field-code material of 288 lines or fewer, or keep
/// taller progressive material on the plain raster (spec/06 §1).
pub fn encode_frame_with_options(
    family: PixelFamily,
    width: u32,
    height: u32,
    pixels: &[u8],
    options: &EncodeOptions,
) -> Result<EncodedFrame> {
    let field_coded = options.field_coding.resolve(height, options.mode)?;
    let (strf, frame, method) = encode_auto_inner(
        family,
        options.method,
        width,
        height,
        field_coded,
        pixels,
        options.mode,
        options.worker_budget,
    )?;
    Ok(EncodedFrame {
        strf,
        frame,
        method,
    })
}

/// Synthesise a complete `BITMAPINFOHEADER` payload + per-frame
/// compressed bytes. Returns `(strf, frame)` where `strf` is the AVI
/// `strf` payload (parseable by [`StreamConfig::parse_bitmapinfoheader`])
/// and `frame` is the raw `00dc` chunk body.
///
/// This is the round-1-compatible default: uses
/// [`ExtradataMode::ClassicV2`].
pub fn encode_frame(
    family: PixelFamily,
    method: Method,
    width: u32,
    height: u32,
    pixels: &[u8],
) -> Result<(Vec<u8>, Vec<u8>)> {
    encode_frame_with_mode(
        family,
        method,
        width,
        height,
        pixels,
        ExtradataMode::ClassicV2,
    )
}

/// Encode a frame using the caller-selected extradata path.
pub fn encode_frame_with_mode(
    family: PixelFamily,
    method: Method,
    width: u32,
    height: u32,
    pixels: &[u8],
    mode: ExtradataMode,
) -> Result<(Vec<u8>, Vec<u8>)> {
    if family.is_rgb() && !method.is_rgb_legal() {
        return Err(Error::invalid("encoder: method not legal for RGB"));
    }
    if !family.is_rgb() && !method.is_yuv_legal() {
        return Err(Error::invalid("encoder: method not legal for YUV"));
    }
    // Round-7: factor the residual computation out so it can be
    // shared across the auto-selector's candidate scoring + the final
    // emit pass (one residual computation per candidate instead of
    // candidate + winner-restart = N+1).
    let field_coded = is_interlaced_height(height);
    let frame = residuals_for(family, method, width, height, field_coded, pixels)?;
    encode_with_precomputed(family, method, width, height, field_coded, &frame, mode)
}

/// Encode a frame under an explicit worker budget.
///
/// Same wire output as [`encode_frame_with_mode`] for every input and
/// every budget. A fixed-method frame is one residual pass feeding one
/// continuous codeword stream (spec/02 §2; a field-coded frame has no
/// separately coded halves either, spec/06 §3.4), so there is a single
/// unit of work and this currently runs serially whatever the budget;
/// the budget matters for [`encode_frame_auto_workers`], whose
/// candidate scoring does fan out.
pub fn encode_frame_with_mode_workers(
    family: PixelFamily,
    method: Method,
    width: u32,
    height: u32,
    pixels: &[u8],
    mode: ExtradataMode,
    worker_budget: usize,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let _ = worker_budget;
    encode_frame_with_mode(family, method, width, height, pixels, mode)
}

/// Encode one frame's wire bytes against caller-pinned per-slot
/// Huffman tables (round 429).
///
/// Stream-encode support for the registry `Encoder`: HuffYUV Huffman
/// tables are STREAM-level (they travel in the `BITMAPINFOHEADER`
/// extradata, not per frame), so a multi-frame `CustomV2` stream must
/// emit every frame with the tables pinned from its first frame —
/// re-deriving per-frame-optimal tables (what
/// [`encode_frame_with_mode`] does with [`ExtradataMode::CustomV2`])
/// would change the extradata mid-stream and make frames 2..N
/// undecodable. The residual + emit pipeline is exactly the
/// [`encode_frame_with_mode`] one; only the table sourcing differs.
///
/// Because pinned first-frame tables assign length 0 to symbols the
/// first frame never emitted, the body is verified against the tables
/// before emit — a later frame whose residuals need an uncovered
/// symbol returns `Error::Unsupported` instead of writing an
/// undecodable zero-length codeword.
///
/// Only the `registry` stream encoder needs pinned-table emission, so
/// the function is compiled out of standalone builds.
#[cfg(feature = "registry")]
pub(crate) fn encode_body_with_pinned_tables(
    family: PixelFamily,
    method: Method,
    width: u32,
    height: u32,
    pixels: &[u8],
    tabs: &crate::decoder::ThreeTables,
    field_coded: bool,
) -> Result<Vec<u8>> {
    if family.is_rgb() && !method.is_rgb_legal() {
        return Err(Error::invalid("encoder: method not legal for RGB"));
    }
    if !family.is_rgb() && !method.is_yuv_legal() {
        return Err(Error::invalid("encoder: method not legal for YUV"));
    }
    let frame = residuals_for(family, method, width, height, field_coded, pixels)?;
    verify_body_in_table(
        family,
        method,
        &frame.body,
        &tabs.slot1.entries,
        &tabs.slot2.entries,
        &tabs.slot3.entries,
    )
    .map_err(|_| {
        Error::unsupported(
            "stream tables: frame emits residual symbols absent from the pinned \
             first-frame Huffman tables (custom-v2 tables are stream-level and \
             derive from the first frame; use classic-v2 for drifting content)",
        )
    })?;
    emit_frame_bytes(
        family,
        method,
        &frame,
        &tabs.slot1.entries,
        &tabs.slot2.entries,
        &tabs.slot3.entries,
    )
}

/// Emit one wire frame from a pre-computed residual stream.
///
/// Internal round-7 helper that lets the auto-selector reuse the
/// residual body it already computed for scoring instead of throwing
/// it away and rebuilding from `pixels` inside
/// [`encode_frame_with_mode`]. Direct callers should keep using
/// [`encode_frame_with_mode`] / [`encode_frame_auto`].
fn encode_with_precomputed(
    family: PixelFamily,
    method: Method,
    width: u32,
    height: u32,
    field_coded: bool,
    frame: &Residuals,
    mode: ExtradataMode,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let stats_body: &[u8] = &frame.body;

    // Round-419: the ClassicV2 / V1xCompat table sets are pure
    // functions of compiled-in bytes, so they come out of the decoder's
    // shared `table_cache` (keyed by the same classic-blob bytes the
    // extradata embeds / the same per-family v1.x `OnceLock`s) instead
    // of being rebuilt — three `build_from_lengths` runs including the
    // 64-Ki primary-LUT fill, which the round-419 `sample` profile
    // attributed ~33% of a ClassicV2 320×240 encode to — or re-cloned
    // (3 × 128-KiB LUT memcpy per V1xCompat frame) on every call.
    // Round-440: CustomV2 (the one per-frame build left) no longer
    // builds decode-grade tables at all — the emit path only reads
    // per-symbol `(code, length)` entries, so it takes the
    // canonical-assign phase alone (`canonical_entries_from_lengths`)
    // and skips the 64-Ki primary-LUT fill + overflow bake
    // (~26 µs × 3 of dead work per frame, the r419 BENCHMARKS
    // secondary target). Same assignment, same validation, same
    // errors; byte-identical wire (golden pins + the round440
    // entries-equivalence test in `tables`).
    enum EncTables {
        Shared(std::sync::Arc<crate::decoder::ThreeTables>),
        Lean(Box<[SlotEntries; 3]>),
    }
    impl EncTables {
        fn slots(&self) -> (&SlotEntries, &SlotEntries, &SlotEntries) {
            match self {
                EncTables::Shared(t) => (&t.slot1.entries, &t.slot2.entries, &t.slot3.entries),
                EncTables::Lean(b) => (&b[0], &b[1], &b[2]),
            }
        }
    }
    let (extradata_tables, tabs, has_extradata): (Vec<u8>, EncTables, bool) = match mode {
        ExtradataMode::ClassicV2 => {
            let extra = classic_blob_bytes(family, method).to_vec();
            let tabs = crate::decoder::table_cache::extradata_tables(&extra)?;
            (extra, EncTables::Shared(tabs), true)
        }
        ExtradataMode::CustomV2 => {
            let lengths = compute_lengths_from_body(family, method, stats_body);
            let tabs = Box::new([
                crate::tables::canonical_entries_from_lengths(&lengths[0])?.0,
                crate::tables::canonical_entries_from_lengths(&lengths[1])?.0,
                crate::tables::canonical_entries_from_lengths(&lengths[2])?.0,
            ]);
            let extra = rle_encode_three_channels(&lengths)?;
            (extra, EncTables::Lean(tabs), true)
        }
        ExtradataMode::V1xCompat => {
            let tabs = crate::decoder::table_cache::v1x_tables(family)?;
            // Verify residuals only use symbols that have non-zero
            // length in the v1.x codebooks — otherwise the wire would
            // be undecodable. (The classic v1.x sets cover all 256
            // symbols, so this is a belt-and-braces check.)
            verify_body_in_table(
                family,
                method,
                stats_body,
                &tabs.slot1.entries,
                &tabs.slot2.entries,
                &tabs.slot3.entries,
            )?;
            (Vec::new(), EncTables::Shared(tabs), false)
        }
    };
    let (slot1, slot2, slot3) = tabs.slots();

    let strf = build_bitmapinfoheader_field_coded(
        family,
        method,
        width,
        height,
        &extradata_tables,
        has_extradata,
        field_coded,
    );

    let frame_bytes = emit_frame_bytes(family, method, frame, slot1, slot2, slot3)?;

    let _ = (width, height);
    Ok((strf, frame_bytes))
}

/// Emit the wire bytes for one pre-computed residual frame using the
/// caller-supplied per-slot Huffman tables: the uncompressed first
/// pixel followed by one continuous codeword stream (spec/02 §2 — a
/// field-coded frame has the same shape, spec/06 §3.4).
fn emit_frame_bytes(
    family: PixelFamily,
    method: Method,
    frame: &Residuals,
    slot1: &SlotEntries,
    slot2: &SlotEntries,
    slot3: &SlotEntries,
) -> Result<Vec<u8>> {
    emit_bitstream_parts(
        family,
        method,
        &frame.seed,
        &frame.body,
        slot1,
        slot2,
        slot3,
    )
}

/// Compute the per-frame residual stream once. Shared by
/// [`encode_frame_with_mode`], [`encode_frame_auto`] and
/// [`bit_cost_for_method`].
///
/// The encoder field-codes exactly when `height > 288` (spec/06 §1.1)
/// and [`build_bitmapinfoheader`] writes the matching interlace byte
/// (spec/06 §1.2), so the two builds of the original codec and any
/// decoder following spec/06 §1.3 agree on the raster. Field coding
/// only doubles the prediction row length (spec/06 §2.1).
fn compute_frame_residuals(
    family: PixelFamily,
    method: Method,
    width: u32,
    height: u32,
    pixels: &[u8],
) -> Result<Residuals> {
    residuals_for(
        family,
        method,
        width,
        height,
        is_interlaced_height(height),
        pixels,
    )
}

/// Residual stream for an explicit field-coding decision: field coding
/// only doubles the prediction row length (spec/06 §2.1).
fn residuals_for(
    family: PixelFamily,
    method: Method,
    width: u32,
    height: u32,
    field_coded: bool,
    pixels: &[u8],
) -> Result<Residuals> {
    let row_bytes = family.row_bytes(width);
    let stride = if field_coded {
        row_bytes.saturating_mul(2)
    } else {
        row_bytes
    };
    match family {
        PixelFamily::Yuy2 => yuy2_residuals(method, width, height, stride, pixels),
        PixelFamily::Rgb24 | PixelFamily::Rgb32 => {
            rgb_residuals(family, method, width, height, stride, pixels)
        }
    }
}

/// Build a `BITMAPINFOHEADER` for the encoded stream. `with_extradata`
/// selects the v2.x layout (`biSize > 0x28`, 4-byte fixed prefix +
/// `extradata_tables`). `extradata_tables` MUST be empty when
/// `with_extradata == false`.
///
/// This is the public write-side complement to
/// [`StreamConfig::parse_bitmapinfoheader`]: muxers can call this to
/// produce the `strf` payload they hand to the AVI writer.
pub fn build_bitmapinfoheader(
    family: PixelFamily,
    method: Method,
    width: u32,
    height: u32,
    extradata_tables: &[u8],
    with_extradata: bool,
) -> Vec<u8> {
    build_bitmapinfoheader_field_coded(
        family,
        method,
        width,
        height,
        extradata_tables,
        with_extradata,
        is_interlaced_height(height),
    )
}

/// [`build_bitmapinfoheader`] with the field-coding decision stated
/// by the caller instead of derived from the height: the interlace
/// byte at BIH `+0x2A` is `0x10` when `field_coded`, `0x20` otherwise
/// (spec/06 §1.2). A header without extradata has no such byte and
/// ignores the argument.
pub fn build_bitmapinfoheader_field_coded(
    family: PixelFamily,
    method: Method,
    width: u32,
    height: u32,
    extradata_tables: &[u8],
    with_extradata: bool,
    field_coded: bool,
) -> Vec<u8> {
    if !with_extradata {
        // v1.x layout (spec/01 §6): biSize = 0x28, no extradata.
        // For v1.x to round-trip, biBitCount carries the method via
        // the low-3-bits selector (spec/01 §1.4). For PredictOld we
        // emit low3 == 0 (= predict_old default).
        let mut v = vec![0u8; 0x28];
        let bi_size: u32 = 0x28;
        let bit_count_base = match family {
            PixelFamily::Yuy2 => 16u16,
            PixelFamily::Rgb24 => 24u16,
            PixelFamily::Rgb32 => 32u16,
        };
        let low3: u16 = match method {
            Method::PredictOld => 0,
            Method::Left => 1,
            Method::LeftDecorr => 2,
            Method::Gradient => 3,
            Method::GradientDecorr => 3, // distinguished by bit_count_base ≥ 0x18.
            Method::Median => 4,
        };
        let bit_count = bit_count_base | low3;
        v[0..4].copy_from_slice(&bi_size.to_le_bytes());
        v[4..8].copy_from_slice(&(width as i32).to_le_bytes());
        v[8..12].copy_from_slice(&(height as i32).to_le_bytes());
        v[12..14].copy_from_slice(&1u16.to_le_bytes());
        v[0x0E..0x10].copy_from_slice(&bit_count.to_le_bytes());
        v[0x10..0x14].copy_from_slice(&FOURCC_HFYU.to_le_bytes());
        // biSizeImage / biXPelsPerMeter / biYPelsPerMeter / biClrUsed /
        // biClrImportant left zero.
        return v;
    }

    let bi_size = 0x2C + extradata_tables.len() as u32;
    let bit_count: u16 = match family {
        PixelFamily::Yuy2 => 24, // encoder forces ≥ 24 even for YUY2 (spec/01 §1.5).
        PixelFamily::Rgb24 => 24,
        PixelFamily::Rgb32 => 32,
    };
    let mut v = vec![0u8; bi_size as usize];
    v[0..4].copy_from_slice(&bi_size.to_le_bytes());
    v[4..8].copy_from_slice(&(width as i32).to_le_bytes());
    v[8..12].copy_from_slice(&(height as i32).to_le_bytes());
    v[12..14].copy_from_slice(&1u16.to_le_bytes());
    v[0x0E..0x10].copy_from_slice(&bit_count.to_le_bytes());
    v[0x10..0x14].copy_from_slice(&FOURCC_HFYU.to_le_bytes());
    v[0x28] = method.to_byte() as u8;
    v[0x29] = match family {
        PixelFamily::Yuy2 => 16,
        PixelFamily::Rgb24 => 24,
        PixelFamily::Rgb32 => 32,
    };
    // Interlace byte at +0x2A (spec/01 §3, spec/06 §1.2): `0x10`
    // field-coded, `0x20` plain raster. A decoder that reads the byte
    // follows it; one that does not applies the height rule.
    v[0x2A] = if field_coded { 0x10 } else { 0x20 };
    v[0x2B] = 0;
    v[0x2C..0x2C + extradata_tables.len()].copy_from_slice(extradata_tables);
    v
}

/// Per-method bit-cost estimate for the package-merge optimal
/// length tables built from this `(method, pixels)`'s residual
/// stream. Used by [`encode_frame_auto`] to pick the predictor that
/// produces the smallest encoded body; exposed publicly so
/// muxer-side encoders can inspect the trade-off without doing two
/// emit passes.
///
/// Returns the total number of code-bits (excluding the 4-byte
/// uncompressed first pixel) the body would occupy using the
/// per-channel package-merge length tables. This is the same metric
/// the auto-selector minimises.
///
/// Returns `Err` if the `(family, method)` pair isn't legal
/// per `spec/01 §3.1`, or if the input buffer size is wrong for
/// `width × height × bytes-per-pixel`.
pub fn bit_cost_for_method(
    family: PixelFamily,
    method: Method,
    width: u32,
    height: u32,
    pixels: &[u8],
) -> Result<u64> {
    if family.is_rgb() && !method.is_rgb_legal() {
        return Err(Error::invalid("bit_cost: method not legal for RGB"));
    }
    if !family.is_rgb() && !method.is_yuv_legal() {
        return Err(Error::invalid("bit_cost: method not legal for YUV"));
    }
    // Round-7: share `compute_frame_residuals` with the encoder path
    // so a subsequent `encode_frame_auto` can reuse this residual
    // computation rather than redoing it.
    let frame = compute_frame_residuals(family, method, width, height, pixels)?;
    let (h1, h2, h3) = histogramise(family, method, &frame.body);
    Ok(bit_cost_from_histograms(&h1, &h2, &h3))
}

/// Sum `Σ length[symbol] × count[symbol]` over each slot's
/// package-merge optimal length table. Each call runs the
/// length-limited package-merge builder three times (once per slot),
/// then walks the histogram. Total work is `O(256 × 31 × 3) +
/// O(256 × 3)` per call — trivially small even at 4K.
fn bit_cost_from_histograms(h1: &[u32; 256], h2: &[u32; 256], h3: &[u32; 256]) -> u64 {
    let l1 = compute_canonical_lengths(h1).unwrap_or([1u8; 256]);
    let l2 = compute_canonical_lengths(h2).unwrap_or([1u8; 256]);
    let l3 = compute_canonical_lengths(h3).unwrap_or([1u8; 256]);
    let mut bits: u64 = 0;
    for (h, l) in [(h1, &l1), (h2, &l2), (h3, &l3)] {
        for s in 0..256 {
            if h[s] != 0 {
                bits += (l[s] as u64) * (h[s] as u64);
            }
        }
    }
    bits
}

/// Selects which `(family, method)` combination [`encode_frame_auto`]
/// considers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum MethodSelection {
    /// Pin the predictor — same behaviour as the fixed-method
    /// `encode_frame_with_mode`.
    Fixed(Method),
    /// Try every legal predictor for this family, pick the one with
    /// the smallest [`bit_cost_for_method`].
    #[default]
    Auto,
}

impl MethodSelection {
    fn candidates(&self, family: PixelFamily) -> Vec<Method> {
        match self {
            MethodSelection::Fixed(m) => vec![*m],
            MethodSelection::Auto => Self::legal_methods(family),
        }
    }

    /// All `(family, method)` pairs the encoder may legally produce.
    /// Order is deterministic so test output is stable on ties.
    pub fn legal_methods(family: PixelFamily) -> Vec<Method> {
        if family.is_rgb() {
            vec![
                Method::Left,
                Method::LeftDecorr,
                Method::GradientDecorr,
                // PredictOld is wire-distinct but produces an
                // identical body to Left for the auto-selector's
                // purposes (same predictor + no decorrelation); we
                // omit it from Auto since Left dominates and Auto
                // picks one winner deterministically. Callers needing
                // `PredictOld` on the wire should use `Fixed`.
            ]
        } else {
            vec![Method::Left, Method::Gradient, Method::Median]
        }
    }
}

/// Encode a frame, picking the predictor with the smallest bit cost
/// over the legal methods for this `family`. Returns the encoded
/// `(strf, frame_bytes)` plus the [`Method`] the auto-selector chose.
///
/// `selection` controls which methods are considered:
/// [`MethodSelection::Auto`] tries every legal predictor for the
/// family; [`MethodSelection::Fixed`] runs only the named method and
/// returns it back unchanged (equivalent to calling
/// [`encode_frame_with_mode`] directly but with the matching
/// `chosen_method` return).
///
/// `mode` selects the extradata path the chosen frame uses
/// ([`ExtradataMode::CustomV2`] yields the smallest wire frame for
/// nontrivial content because it ships per-frame-optimal Huffman
/// length tables; [`ExtradataMode::ClassicV2`] is bigger on the body
/// but skips the per-frame extradata RLE and matches the
/// proprietary's default; [`ExtradataMode::V1xCompat`] forces the
/// v1.x precomputed codebook).
pub fn encode_frame_auto(
    family: PixelFamily,
    selection: MethodSelection,
    width: u32,
    height: u32,
    pixels: &[u8],
    mode: ExtradataMode,
) -> Result<(Vec<u8>, Vec<u8>, Method)> {
    encode_frame_auto_workers(family, selection, width, height, pixels, mode, 1)
}

/// Score one auto-selector candidate: residual stream + package-merge
/// bit cost. Round-7 semantics unchanged — the returned
/// [`Residuals`] carries forward into the winner's final emit
/// so the body bytes are never re-derived from `pixels`.
fn score_candidate(
    family: PixelFamily,
    method: Method,
    width: u32,
    height: u32,
    field_coded: bool,
    pixels: &[u8],
) -> Result<(u64, Residuals)> {
    let frame = residuals_for(family, method, width, height, field_coded, pixels)?;
    let (h1, h2, h3) = histogramise(family, method, &frame.body);
    Ok((bit_cost_from_histograms(&h1, &h2, &h3), frame))
}

/// [`encode_frame_auto`] under an explicit worker budget (round 429).
///
/// Same `(strf, wire bytes, chosen method)` as [`encode_frame_auto`]
/// for every input — the budget only changes how the work is
/// scheduled, following the `oxideav-core` `ExecutionContext`
/// threading contract (serial until told otherwise; fan-out bounded
/// inline by `budget.min(units).max(1)`). One fan-out is gated on
/// the budget:
///
/// - **Candidate scoring** (`units = candidates.len()`): each legal
///   method's residual + package-merge bit-cost evaluation is
///   independent of the others, so the candidate list is split into
///   contiguous chunks across the granted workers (caller thread
///   included). Every candidate's result lands in its own slot, then
///   a serial in-order reduction applies the exact round-7 rules —
///   first error in candidate order surfaces, smallest cost wins,
///   first-in-order wins ties — so the winner (and therefore the
///   wire) cannot depend on scheduling.
///
/// The winner's emit is a single codeword stream and runs serially.
///
/// Byte-identity across budgets is guarded by the r429 auto-selector
/// budget sweep in the golden-pin suite.
pub fn encode_frame_auto_workers(
    family: PixelFamily,
    selection: MethodSelection,
    width: u32,
    height: u32,
    pixels: &[u8],
    mode: ExtradataMode,
    worker_budget: usize,
) -> Result<(Vec<u8>, Vec<u8>, Method)> {
    encode_auto_inner(
        family,
        selection,
        width,
        height,
        is_interlaced_height(height),
        pixels,
        mode,
        worker_budget,
    )
}

/// Auto-selector body shared by [`encode_frame_auto_workers`] (height
/// rule) and [`encode_frame_with_options`] (caller's field-coding
/// decision).
#[allow(clippy::too_many_arguments)] // one frame description, passed flat like its public callers
pub(crate) fn encode_auto_inner(
    family: PixelFamily,
    selection: MethodSelection,
    width: u32,
    height: u32,
    field_coded: bool,
    pixels: &[u8],
    mode: ExtradataMode,
    worker_budget: usize,
) -> Result<(Vec<u8>, Vec<u8>, Method)> {
    let candidates = selection.candidates(family);
    for &m in &candidates {
        if family.is_rgb() && !m.is_rgb_legal() {
            return Err(Error::invalid("encoder: method not legal for RGB"));
        }
        if !family.is_rgb() && !m.is_yuv_legal() {
            return Err(Error::invalid("encoder: method not legal for YUV"));
        }
    }
    if candidates.is_empty() {
        return Err(Error::invalid("encode_frame_auto: no legal methods"));
    }
    if let [only] = candidates[..] {
        // A pinned method needs no scoring pass.
        let frame = residuals_for(family, only, width, height, field_coded, pixels)?;
        let (strf, bytes) =
            encode_with_precomputed(family, only, width, height, field_coded, &frame, mode)?;
        return Ok((strf, bytes, only));
    }
    // Inline scoring fan-out clamp per the ExecutionContext threading
    // contract — `threads.min(units).max(1)` with `units` = number of
    // candidate methods.
    let scoring_workers = worker_budget.min(candidates.len()).max(1);
    // Round-7: compute residuals ONCE per candidate (was: once per
    // candidate inside `bit_cost_for_method` PLUS once again inside
    // `encode_frame_with_mode` for the winner = N+1 traversals).
    // The pre-computed residuals carry forward into the final
    // `encode_with_precomputed` so the winner's body bytes don't get
    // re-derived from `pixels`. Wire-identical to round 6.
    //
    // Tie-break: first in `candidates` order (so output is
    // deterministic).
    let mut best: Option<(Method, u64, Residuals)> = None;
    if scoring_workers >= 2 {
        // Parallel scoring: per-candidate result slots filled by
        // contiguous chunks (first chunk on the caller thread, so at
        // most `scoring_workers - 1` threads are spawned and the
        // caller counts against the budget), then reduced serially in
        // candidate order below — winner, tie-break, and error order
        // all match the serial arm exactly.
        let mut results: Vec<Option<Result<(u64, Residuals)>>> = std::iter::repeat_with(|| None)
            .take(candidates.len())
            .collect();
        let chunk_len = candidates.len().div_ceil(scoring_workers);
        std::thread::scope(|s| {
            let (own_slots, mut slots_rest) = results.split_at_mut(chunk_len);
            let (own_cands, mut cands_rest) = candidates.split_at(chunk_len);
            while !cands_rest.is_empty() {
                let n = chunk_len.min(cands_rest.len());
                let (chunk_slots, tail) = slots_rest.split_at_mut(n);
                slots_rest = tail;
                let (chunk_cands, ctail) = cands_rest.split_at(n);
                cands_rest = ctail;
                s.spawn(move || {
                    for (slot, &m) in chunk_slots.iter_mut().zip(chunk_cands) {
                        *slot = Some(score_candidate(
                            family,
                            m,
                            width,
                            height,
                            field_coded,
                            pixels,
                        ));
                    }
                });
            }
            for (slot, &m) in own_slots.iter_mut().zip(own_cands) {
                *slot = Some(score_candidate(
                    family,
                    m,
                    width,
                    height,
                    field_coded,
                    pixels,
                ));
            }
        });
        for (&m, slot) in candidates.iter().zip(results) {
            let (cost, frame) = slot.expect("every scoring slot is filled by its chunk")?;
            match best {
                None => best = Some((m, cost, frame)),
                Some((_, prev_cost, _)) if cost < prev_cost => best = Some((m, cost, frame)),
                _ => {}
            }
        }
    } else {
        for &m in &candidates {
            let (cost, frame) = score_candidate(family, m, width, height, field_coded, pixels)?;
            match best {
                None => best = Some((m, cost, frame)),
                Some((_, prev_cost, _)) if cost < prev_cost => best = Some((m, cost, frame)),
                _ => {}
            }
        }
    }
    let (chosen, _, frame) = best.expect("non-empty candidates → some winner");
    let (strf, bytes) =
        encode_with_precomputed(family, chosen, width, height, field_coded, &frame, mode)?;
    Ok((strf, bytes, chosen))
}

/// Convenience wrapper for tests: encode + parse the strf into a
/// resolved [`StreamConfig`] in one call.
pub fn encode_for_test(
    family: PixelFamily,
    method: Method,
    width: u32,
    height: u32,
    pixels: &[u8],
) -> Result<(StreamConfig, Vec<u8>)> {
    let (strf, frame_bytes) = encode_frame(family, method, width, height, pixels)?;
    let cfg = StreamConfig::parse_bitmapinfoheader(&strf)?;
    Ok((cfg, frame_bytes))
}

/// Like [`encode_for_test`], but selects the extradata path explicitly.
pub fn encode_for_test_with_mode(
    family: PixelFamily,
    method: Method,
    width: u32,
    height: u32,
    pixels: &[u8],
    mode: ExtradataMode,
) -> Result<(StreamConfig, Vec<u8>)> {
    let (strf, frame_bytes) = encode_frame_with_mode(family, method, width, height, pixels, mode)?;
    let cfg = StreamConfig::parse_bitmapinfoheader(&strf)?;
    Ok((cfg, frame_bytes))
}

// ───────────────────────── residual computation ─────────────────────────
//
// `compute_residuals` returns the per-byte residual stream that the
// emit pass walks (one byte = one Huffman codeword for non-uncompressed
// positions). Layout per family:
//
// - YUY2: row-major, byte offset i ∈ 0..(w*h*2). Bytes 0..3 are the
//   uncompressed first macropixel; subsequent bytes are residuals.
// - RGB24: row-major, byte offset i ∈ 0..(w*h*3). Bytes 0..2 are the
//   first pixel's working B/G/R (decorrelated when applicable); the
//   wire stream prefixes a 0 pad byte. Subsequent bytes are residuals.
// - RGB32: row-major, byte offset i ∈ 0..(w*h*4). Bytes 0..3 are the
//   first pixel's working B/G/R/A (decorrelated when applicable);
//   subsequent bytes are residuals.

#[derive(Debug, Clone)]
struct Residuals {
    /// The "uncompressed first pixel" payload as wire bytes (4 bytes
    /// for all three families: YUY2 = first macropixel, RGB24 = `00 B
    /// G R`, RGB32 = `B G R A`).
    seed: [u8; 4],
    /// Residuals for byte positions in the linear raster, starting at
    /// the byte AFTER the uncompressed seed. For RGB24 the seed
    /// covers 3 source bytes; we still index per-source-byte after
    /// that. The encoder maps `(byte_idx → slot)` to look up the right
    /// Huffman table per spec/03 §1.2.
    body: Vec<u8>,
}

fn yuy2_residuals(
    method: Method,
    width: u32,
    height: u32,
    stride: usize,
    pixels: &[u8],
) -> Result<Residuals> {
    let w = width as usize;
    let h = height as usize;
    // spec/02 §3 wire-byte layout table: Y₁ U Y₂ V per 2 px.
    let row_bytes = PixelFamily::Yuy2.row_bytes(width);
    if pixels.len() != row_bytes * h {
        return Err(Error::invalid(format!(
            "encoder: YUY2 pixel-buffer size {} ≠ expected {}",
            pixels.len(),
            row_bytes * h
        )));
    }
    if !w.is_multiple_of(2) {
        return Err(Error::invalid("encoder: YUY2 width must be even"));
    }
    let mut residuals = vec![0u8; pixels.len()];
    match method.predictor() {
        // spec/03 §2.3 / §2.3.2, spec/06 §3.3: LEFT residuals up to
        // one prediction row + 8 wire bytes, median after.
        Predictor::Median => forward_median_yuy2(pixels, &mut residuals, stride),
        // spec/03 §2.2.2, spec/06 §3.2: subtract the byte one
        // prediction row back, then LEFT-code the result.
        Predictor::Gradient => {
            let mut above_removed = vec![0u8; pixels.len()];
            forward_gradient_stride(pixels, &mut above_removed, stride);
            forward_yuy2_left_subtract(&above_removed, &mut residuals);
        }
        // spec/03 §2.1.1.
        Predictor::Left => forward_yuy2_left_subtract(pixels, &mut residuals),
    }
    let mut seed = [0u8; 4];
    let copy_n = 4.min(pixels.len());
    seed[..copy_n].copy_from_slice(&pixels[..copy_n]);
    let body = residuals[copy_n..].to_vec();
    Ok(Residuals { seed, body })
}

/// RGB24 (`n_channels == 3`) / RGB32 (`n_channels == 4`) residuals.
///
/// The first pixel goes out as the plain input bytes — `00 B G R` or
/// `B G R A` (spec/03 §1.1, §1.3; spec/02 §8 #6) — also when the
/// method decorrelates: the residual chain then starts from the
/// decorrelated first pixel (spec/03 §2.4.1 fused formula), which the
/// decoder derives from those same bytes.
fn rgb_residuals(
    family: PixelFamily,
    method: Method,
    width: u32,
    height: u32,
    stride: usize,
    pixels: &[u8],
) -> Result<Residuals> {
    let n_channels = family.bytes_per_pixel_step();
    let row_bytes = family.row_bytes(width);
    let h = height as usize;
    if pixels.len() != row_bytes * h {
        return Err(Error::invalid(format!(
            "encoder: {} pixel-buffer size {} ≠ expected {}",
            if n_channels == 3 { "RGB24" } else { "RGB32" },
            pixels.len(),
            row_bytes * h
        )));
    }
    if pixels.len() < n_channels {
        return Err(Error::invalid(
            "encoder: RGB frame needs at least one pixel",
        ));
    }
    let decorrelate = method.decorrelate();
    let mut residuals = vec![0u8; pixels.len()];
    match (method.predictor(), decorrelate) {
        (Predictor::Median, _) => {
            return Err(Error::invalid("encoder: method not legal for RGB"));
        }
        (Predictor::Left, false) => {
            forward_rgb_left_subtract_linear(pixels, &mut residuals, n_channels);
        }
        (Predictor::Left, true) => {
            forward_left_decorr_residuals(pixels, &mut residuals, n_channels);
        }
        (Predictor::Gradient, false) => {
            let mut above_removed = vec![0u8; pixels.len()];
            forward_gradient_stride(pixels, &mut above_removed, stride);
            forward_rgb_left_subtract_linear(&above_removed, &mut residuals, n_channels);
        }
        (Predictor::Gradient, true) => {
            let mut above_removed = vec![0u8; pixels.len()];
            forward_decorr_gradient_stride(pixels, &mut above_removed, n_channels, stride);
            forward_rgb_left_subtract_linear(&above_removed, &mut residuals, n_channels);
        }
    }
    let seed = if n_channels == 3 {
        [0, pixels[0], pixels[1], pixels[2]]
    } else {
        [pixels[0], pixels[1], pixels[2], pixels[3]]
    };
    // Body in wire order (spec/03 §1.4): `B, G, R[, A]` plain,
    // `G, B−G, R−G[, A]` decorrelated.
    let mut body = Vec::with_capacity(pixels.len() - n_channels);
    for px in residuals[n_channels..].chunks_exact(n_channels) {
        if decorrelate {
            body.push(px[1]);
            body.push(px[0]);
        } else {
            body.push(px[0]);
            body.push(px[1]);
        }
        body.extend_from_slice(&px[2..]);
    }
    Ok(Residuals { seed, body })
}

// ───────────────────────── histogram + length building ─────────────────────────

/// Walks `body` directly to build the per-channel histograms, then
/// runs the package-merge length-limited Huffman builder.
fn compute_lengths_from_body(family: PixelFamily, method: Method, body: &[u8]) -> [[u8; 256]; 3] {
    let (h1, h2, h3) = histogramise(family, method, body);
    [
        compute_canonical_lengths(&h1).unwrap_or([1u8; 256]),
        compute_canonical_lengths(&h2).unwrap_or([1u8; 256]),
        compute_canonical_lengths(&h3).unwrap_or([1u8; 256]),
    ]
}

fn histogramise(
    family: PixelFamily,
    method: Method,
    body: &[u8],
) -> ([u32; 256], [u32; 256], [u32; 256]) {
    let mut h1 = [0u32; 256];
    let mut h2 = [0u32; 256];
    let mut h3 = [0u32; 256];
    match family {
        PixelFamily::Yuy2 => {
            // Round-227: macropixel-step YUY2 histogram body. The
            // spec/03 §1.2 three-slot architecture pins the YUY2
            // wire-byte → slot mapping at a fixed 4-byte cycle:
            //
            //   +0 (Y₁) → slot1   +1 (U)  → slot2
            //   +2 (Y₂) → slot1   +3 (V)  → slot3
            //
            // The pre-r227 loop ran `match byte_idx & 3 { … }` on
            // every body byte to pick the histogram. body[i]
            // corresponds to source byte index (i + 4), and `(i + 4)
            // & 3 == i & 3`, so the cycle starts in phase at i = 0
            // with slot1 / slot2 / slot1 / slot3. Round 227 steps
            // four body bytes per outer iteration with the slot
            // resolved at compile time — same shape as the r214
            // decode-side and r221 emit-side rewrites — so the
            // optimiser can schedule the four indexed counter
            // increments freely. `body.len() % 4 == 0` in the
            // in-spec input space (YUY2 width is even per the
            // spec/02 §3.1 macropixel-pair invariant and `body.len()
            // = total_bytes − 4` per field), so the macropixel body
            // covers every count byte. A 1..=3-byte scalar
            // fall-through is kept for defence-in-depth against
            // future pixel-family extensions, mirroring the r214 /
            // r221 fall-throughs.
            let body_aligned = body.len() & !3;
            let mut i = 0usize;
            while i < body_aligned {
                h1[body[i] as usize] += 1;
                h2[body[i + 1] as usize] += 1;
                h1[body[i + 2] as usize] += 1;
                h3[body[i + 3] as usize] += 1;
                i += 4;
            }
            while i < body.len() {
                let byte_idx = i + 4;
                match byte_idx & 3 {
                    0 | 2 => h1[body[i] as usize] += 1,
                    1 => h2[body[i] as usize] += 1,
                    _ => h3[body[i] as usize] += 1,
                }
                i += 1;
            }
        }
        PixelFamily::Rgb24 => {
            // Round-242: pixel-step RGB24 histogram body. spec/03 §1.1
            // pins RGB24 at exactly three Huffman codewords per pixel
            // (the §3.2/§3.3 spec/02 correction); §1.2 fixes the
            // position → slot mapping at:
            //
            //   no decorr: pos +0 (B) → slot1   +1 (G) → slot2   +2 (R) → slot3
            //   decorr   : pos +0 (G) → slot2   +1 (B−G) → slot1  +2 (R−G) → slot3
            //
            // The pre-r242 loop ran `match i % 3` on every body byte
            // AND a `method.decorrelate()` branch every iteration — two
            // per-byte branches the optimiser could not eliminate
            // because `i` was the iterator state and the answer to
            // `method.decorrelate()` never changes mid-frame. Round 242
            // hoists both decisions out of the loop: the per-position
            // histogram triple is resolved once at function entry by
            // the `(h_pos0, h_pos1, h_pos2)` binding (paired by
            // `method.decorrelate()`), then the body steps three bytes
            // per outer iteration with the slot resolved at compile
            // time — three indexed counter increments per wire pixel.
            // Histogram-side companion to r239's RGB24 emit rewrite
            // (and mirror of r227's YUY2 histogram macropixel-step
            // body, applied to the §1.2 three-byte RGB24 wire cycle).
            //
            // `body.len()` is always a multiple of 3 in the in-spec
            // input space (the body is `(n_pixels − 1) × 3` bytes per
            // `rgb24_residuals`), so the pixel-step body covers every
            // count byte. A 1..=2-byte scalar fall-through is kept for
            // defence-in-depth against future pixel-family extensions,
            // mirroring the r221 / r227 / r239 fall-throughs.
            let (h_pos0, h_pos1, h_pos2): (&mut [u32; 256], &mut [u32; 256], &mut [u32; 256]) =
                if method.decorrelate() {
                    (&mut h2, &mut h1, &mut h3)
                } else {
                    (&mut h1, &mut h2, &mut h3)
                };
            let body_aligned = (body.len() / 3) * 3;
            let mut i = 0usize;
            while i < body_aligned {
                h_pos0[body[i] as usize] += 1;
                h_pos1[body[i + 1] as usize] += 1;
                h_pos2[body[i + 2] as usize] += 1;
                i += 3;
            }
            // Scalar fall-through for any 1..=2 trailing bytes
            // (unreachable for valid RGB24 inputs; kept for
            // robustness).
            while i < body.len() {
                let in_pixel = i % 3;
                match in_pixel {
                    0 => h_pos0[body[i] as usize] += 1,
                    1 => h_pos1[body[i] as usize] += 1,
                    _ => h_pos2[body[i] as usize] += 1,
                }
                i += 1;
            }
        }
        PixelFamily::Rgb32 => {
            // Round-250: pixel-step RGB32 histogram body. spec/03 §1.3
            // pins RGB32 at exactly four Huffman codewords per pixel —
            // the wire body walks a fixed 4-byte cycle whose slot
            // mapping comes from spec/03 §1.2 (table at the end of §1.2
            // + alpha-shares-slot-3 evidence at `@0x10001b21` and
            // `@0x10001c6d` in §1.2):
            //
            //   no decorr: pos +0 (B)   → slot1   +1 (G) → slot2
            //              pos +2 (R)   → slot3   +3 (A) → slot3
            //   decorr   : pos +0 (G)   → slot2   +1 (B−G) → slot1
            //              pos +2 (R−G) → slot3   +3 (A) → slot3
            //
            // The pre-r250 loop ran `match i % 4` on every body byte to
            // pick the histogram AND a `method.decorrelate()` branch
            // every iteration — two per-byte branches the optimiser
            // could not eliminate because `i` was the iterator state
            // and the answer to `method.decorrelate()` never changes
            // mid-frame. Round 250 hoists both decisions out of the
            // loop: the per-position histogram quadruple is resolved
            // once at function entry by the `(h_pos0, h_pos1, h_pos2,
            // h_pos3)` binding (paired by `method.decorrelate()`),
            // then the body steps four bytes per outer iteration with
            // the slot resolved at compile time — four indexed counter
            // increments per wire pixel. Histogram-side companion to
            // r245's RGB32 emit rewrite (and direct mirror of r242's
            // RGB24 histogram pixel-step body, applied to the §1.3
            // four-byte RGB32 wire cycle).
            //
            // `body.len()` is always a multiple of 4 in the in-spec
            // input space (the body is `(n_pixels − 1) × 4` bytes per
            // `rgb32_residuals`), so the pixel-step body covers every
            // count byte. A 1..=3-byte scalar fall-through is kept for
            // defence-in-depth against future pixel-family extensions,
            // mirroring the r221 / r227 / r239 / r242 / r245 fall-
            // throughs.
            //
            // Slot 3 receives two positions per pixel (pos +2 and pos
            // +3) — both the R / R−G residual and the A residual
            // share the slot-3 codebook per §1.2. The binding is
            // written `(_, _, &mut h3, &mut h3)` for the pos +2 / pos
            // +3 pair, but Rust's borrow checker treats `&mut h3,
            // &mut h3` as two simultaneous mutable borrows of the same
            // array — which is rejected even though every body
            // iteration only writes one of the two references. We
            // sidestep that by binding the +3 (alpha) position to a
            // single `h3` reference via a raw `h3[…]` write in the
            // body block, and reserving the position-quadruple for the
            // three histograms that don't alias.
            let (h_pos0, h_pos1, h_pos2): (&mut [u32; 256], &mut [u32; 256], &mut [u32; 256]) =
                if method.decorrelate() {
                    // pos +0 (G) → slot2, pos +1 (B−G) → slot1,
                    // pos +2 (R−G) → slot3.
                    (&mut h2, &mut h1, &mut h3)
                } else {
                    // pos +0 (B) → slot1, pos +1 (G) → slot2,
                    // pos +2 (R) → slot3.
                    (&mut h1, &mut h2, &mut h3)
                };
            let body_aligned = body.len() & !3;
            let mut i = 0usize;
            while i < body_aligned {
                h_pos0[body[i] as usize] += 1;
                h_pos1[body[i + 1] as usize] += 1;
                // pos +2 and pos +3 both feed slot 3 — `h_pos2` is the
                // slot-3 binding for pos +2, and pos +3 (alpha) also
                // increments slot 3 via the same `h_pos2` reference
                // (alpha shares the slot-3 codebook per §1.2).
                h_pos2[body[i + 2] as usize] += 1;
                h_pos2[body[i + 3] as usize] += 1;
                i += 4;
            }
            // Scalar fall-through for any 1..=3 trailing bytes
            // (unreachable for valid RGB32 inputs; kept for
            // robustness).
            while i < body.len() {
                let in_pixel = i % 4;
                match in_pixel {
                    0 => h_pos0[body[i] as usize] += 1,
                    1 => h_pos1[body[i] as usize] += 1,
                    // pos +2 (R / R−G) and pos +3 (A) both → slot 3.
                    _ => h_pos2[body[i] as usize] += 1,
                }
                i += 1;
            }
        }
    }
    (h1, h2, h3)
}

// ───────────────────────── v1.x compat helpers ─────────────────────────

/// Cache for the per-family V1xCompat tables. The codebook is
/// deterministic per family (spec/04 §4.1: YUY2 = (A, B, B); RGB =
/// (A, A, A)), and each [`HuffTable`] carries a 128 KiB primary LUT
/// — re-deriving + LUT-baking on every frame is pure waste. Round 7
/// caches the three-tuple behind a `OnceLock` per family so the
/// proprietary's V1xCompat path (commonly used by AviSynth + VirtualDub
/// pipelines that need to interop with the original binary) costs the
/// LUT-build once at process start rather than per encode call.
/// V1xCompat verification: walk the body in slot order and ensure
/// each emitted symbol has a non-zero length in its slot's table.
fn verify_body_in_table(
    family: PixelFamily,
    method: Method,
    body: &[u8],
    s1: &SlotEntries,
    s2: &SlotEntries,
    s3: &SlotEntries,
) -> Result<()> {
    // Round-227: hoist YUY2 macropixel-step verification out of the
    // generic per-byte byte_idx-driven match. The other two pixel
    // families still use the per-byte `i % 3 / i % 4` dispatch
    // below — `verify_body_in_table` is RGB-agnostic until those
    // paths grow a step body of their own, which is a separate
    // wire-cycle (3 vs 4) and warrants its own round.
    if let PixelFamily::Yuy2 = family {
        // spec/03 §1.2: YUY2 wire-byte → slot at a fixed 4-byte
        // cycle (+0/+2 Y → slot1; +1 U → slot2; +3 V → slot3).
        // `byte_idx = i + 4` so `(i + 4) & 3 == i & 3`, i.e. the
        // cycle is phase-aligned at i = 0. `body.len() % 4 == 0`
        // holds in the in-spec input space (see the
        // `emit_bitstream_parts` r221 comment for the derivation).
        // A 1..=3-byte scalar fall-through stays for defence-in-
        // depth, mirroring r214 / r221.
        let body_aligned = body.len() & !3;
        let mut i = 0usize;
        while i < body_aligned {
            let sym0 = body[i];
            if s1[sym0 as usize].length == 0 {
                return Err(Error::unsupported(format!(
                    "v1.x compat: residual symbol 0x{sym0:02x} not in v1.x codebook"
                )));
            }
            let sym1 = body[i + 1];
            if s2[sym1 as usize].length == 0 {
                return Err(Error::unsupported(format!(
                    "v1.x compat: residual symbol 0x{sym1:02x} not in v1.x codebook"
                )));
            }
            let sym2 = body[i + 2];
            if s1[sym2 as usize].length == 0 {
                return Err(Error::unsupported(format!(
                    "v1.x compat: residual symbol 0x{sym2:02x} not in v1.x codebook"
                )));
            }
            let sym3 = body[i + 3];
            if s3[sym3 as usize].length == 0 {
                return Err(Error::unsupported(format!(
                    "v1.x compat: residual symbol 0x{sym3:02x} not in v1.x codebook"
                )));
            }
            i += 4;
        }
        while i < body.len() {
            let byte_idx = i + 4;
            let slot = match byte_idx & 3 {
                0 | 2 => s1,
                1 => s2,
                _ => s3,
            };
            let sym = body[i];
            if slot[sym as usize].length == 0 {
                return Err(Error::unsupported(format!(
                    "v1.x compat: residual symbol 0x{sym:02x} not in v1.x codebook"
                )));
            }
            i += 1;
        }
        return Ok(());
    }
    for (byte_idx, (i, &sym)) in (4usize..).zip(body.iter().enumerate()) {
        let slot = match family {
            PixelFamily::Yuy2 => match byte_idx & 3 {
                0 | 2 => s1,
                1 => s2,
                _ => s3,
            },
            PixelFamily::Rgb24 => {
                let in_pixel = i % 3;
                if method.decorrelate() {
                    match in_pixel {
                        0 => s2,
                        1 => s1,
                        _ => s3,
                    }
                } else {
                    match in_pixel {
                        0 => s1,
                        1 => s2,
                        _ => s3,
                    }
                }
            }
            PixelFamily::Rgb32 => {
                let in_pixel = i % 4;
                if method.decorrelate() {
                    match in_pixel {
                        0 => s2,
                        1 => s1,
                        _ => s3,
                    }
                } else {
                    match in_pixel {
                        0 => s1,
                        1 => s2,
                        _ => s3,
                    }
                }
            }
        };
        if slot[sym as usize].length == 0 {
            return Err(Error::unsupported(format!(
                "v1.x compat: residual symbol 0x{sym:02x} not in v1.x codebook"
            )));
        }
    }
    Ok(())
}

// ───────────────────────── bit-emit pass ─────────────────────────

/// Emit one frame: the 4-byte uncompressed first pixel followed by
/// the bit-packed residual body (spec/02 §2, §4).
///
/// Round-7 removed the `emit_bitstream` / `EmitParams` wrapper that
/// previously fronted this routine — all callers now compute the
/// seed/body parts via [`Residuals`] and hand them through
/// directly.
#[allow(clippy::too_many_arguments)]
fn emit_bitstream_parts(
    family: PixelFamily,
    method: Method,
    seed: &[u8; 4],
    body: &[u8],
    slot1: &SlotEntries,
    slot2: &SlotEntries,
    slot3: &SlotEntries,
) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    out.extend_from_slice(seed);
    let mut writer = BitWriter::new();
    match family {
        PixelFamily::Yuy2 => {
            // Round-221: macropixel-step YUY2 Huffman-encode body.
            // spec/03 §1.2's three-slot architecture pins the YUY2
            // wire-byte → slot mapping at a fixed 4-byte cycle:
            //
            //   +0 (Y₁) → slot1   +1 (U) → slot2
            //   +2 (Y₂) → slot1   +3 (V) → slot3
            //
            // The pre-r221 loop ran `match byte_idx & 3 { … }` on
            // every body byte to pick the slot — a per-byte branch
            // the optimiser couldn't eliminate because `byte_idx` was
            // tied to the iterator state. r214 rewrote the decoder's
            // mirror loop (`decode_yuy2_field`) to step four output
            // bytes per outer iteration with the slot resolved at
            // compile time; this round mirrors the same shape on the
            // encoder side. `lookup_code` + `write_msb` per slot,
            // four pairs straight-line per macropixel.
            //
            // `body.len()` is always a multiple of 4 in the in-spec
            // input space: spec/02 §3.1 fixes YUY2 width even (the
            // §2.1.1 macropixel-pair invariant the predict pipeline
            // already enforces), so `total_bytes = width × 2 ×
            // height` is divisible by 4 and `body.len() = total_bytes
            // − 4` (per-field for interlaced) is a multiple of 4.
            // We keep a 1..=3-byte scalar fall-through for
            // defence-in-depth against future pixel-family
            // extensions — same shape as the decoder-side fall-
            // through landed in r214.
            let body_aligned = body.len() & !3;
            let mut i = 0usize;
            while i < body_aligned {
                // byte_idx = i + 4 → byte_idx & 3 == i & 3, and
                // i is 4-aligned here so the four slots line up at
                // i+0 / i+1 / i+2 / i+3 = Y₁ / U / Y₂ / V.
                let (c0, l0) = lookup_code(slot1, body[i])?;
                writer.write_msb(c0, l0);
                let (c1, l1) = lookup_code(slot2, body[i + 1])?;
                writer.write_msb(c1, l1);
                let (c2, l2) = lookup_code(slot1, body[i + 2])?;
                writer.write_msb(c2, l2);
                let (c3, l3) = lookup_code(slot3, body[i + 3])?;
                writer.write_msb(c3, l3);
                i += 4;
            }
            // Scalar fall-through for any 1..=3 trailing bytes
            // (unreachable for valid YUY2 inputs; kept for
            // robustness).
            while i < body.len() {
                let byte_idx = i + 4;
                let slot = match byte_idx & 3 {
                    0 | 2 => slot1,
                    1 => slot2,
                    _ => slot3,
                };
                let (code, length) = lookup_code(slot, body[i])?;
                writer.write_msb(code, length);
                i += 1;
            }
        }
        PixelFamily::Rgb24 => {
            // Round-239: pixel-step RGB24 Huffman-encode body. spec/03
            // §1.1 (the §3.2/§3.3 spec/02 correction) pins RGB24 at
            // exactly three Huffman codewords per pixel — the wire body
            // walks a fixed 3-byte cycle whose slot mapping comes from
            // spec/03 §1.2:
            //
            //   no decorr: pos +0 (B) → slot1   +1 (G) → slot2   +2 (R) → slot3
            //   decorr   : pos +0 (G) → slot2   +1 (B−G) → slot1  +2 (R−G) → slot3
            //
            // The pre-r239 loop ran `match i % 3 { … }` on every body
            // byte to pick the slot — a per-byte branch the optimiser
            // could not eliminate because `i` was tied to the iterator
            // state, plus a second branch on `method.decorrelate()`
            // every iteration even though the answer never changes
            // mid-frame. Round 239 hoists both decisions out of the
            // loop: the slot triple is resolved once at function entry
            // by the `(s_pos0, s_pos1, s_pos2)` binding (paired by
            // `method.decorrelate()`), then the body steps three bytes
            // per outer iteration with the slot resolved at compile
            // time — same shape the r221 YUY2 emit rewrite landed on the
            // §1.2 four-byte cycle.
            //
            // `body.len()` is always a multiple of 3 in the in-spec
            // input space: `rgb24_residuals` builds the body as
            // `(n_pixels − 1) × 3` bytes (one wire pixel = 3 codes per
            // §1.1), so `body.len() % 3 == 0`. We keep a 1..=2-byte
            // scalar fall-through for defence-in-depth against future
            // pixel-family extensions, same shape as the r221 / r227
            // YUY2 fall-throughs.
            let (s_pos0, s_pos1, s_pos2) = if method.decorrelate() {
                (slot2, slot1, slot3)
            } else {
                (slot1, slot2, slot3)
            };
            let body_aligned = (body.len() / 3) * 3;
            let mut i = 0usize;
            while i < body_aligned {
                let (c0, l0) = lookup_code(s_pos0, body[i])?;
                writer.write_msb(c0, l0);
                let (c1, l1) = lookup_code(s_pos1, body[i + 1])?;
                writer.write_msb(c1, l1);
                let (c2, l2) = lookup_code(s_pos2, body[i + 2])?;
                writer.write_msb(c2, l2);
                i += 3;
            }
            // Scalar fall-through for any 1..=2 trailing bytes
            // (unreachable for valid RGB24 inputs; kept for
            // robustness).
            while i < body.len() {
                let in_pixel = i % 3;
                let slot = match in_pixel {
                    0 => s_pos0,
                    1 => s_pos1,
                    _ => s_pos2,
                };
                let (code, length) = lookup_code(slot, body[i])?;
                writer.write_msb(code, length);
                i += 1;
            }
        }
        PixelFamily::Rgb32 => {
            // Round-245: pixel-step RGB32 Huffman-encode body. spec/03
            // §1.3 pins RGB32 at exactly four Huffman codewords per
            // pixel — the wire body walks a fixed 4-byte cycle whose
            // slot mapping comes from spec/03 §1.2 (table at the end of
            // §1.2 + alpha-shares-slot-3 evidence at `@0x10001b21` and
            // `@0x10001c6d` in §1.2):
            //
            //   no decorr: pos +0 (B)   → slot1   +1 (G) → slot2
            //              pos +2 (R)   → slot3   +3 (A) → slot3
            //   decorr   : pos +0 (G)   → slot2   +1 (B−G) → slot1
            //              pos +2 (R−G) → slot3   +3 (A) → slot3
            //
            // The pre-r245 loop ran `match i % 4 { … }` on every body
            // byte to pick the slot — a per-byte branch the optimiser
            // could not eliminate because `i` was tied to the iterator
            // state, plus a second branch on `method.decorrelate()`
            // every iteration even though the answer never changes
            // mid-frame. Round 245 hoists both decisions out of the
            // loop: the slot quadruple is resolved once at function
            // entry by the `(s_pos0, s_pos1, s_pos2, s_pos3)` binding
            // (paired by `method.decorrelate()`), then the body steps
            // four bytes per outer iteration with the slot resolved at
            // compile time — the RGB32 analogue of the r239 RGB24 emit
            // rewrite, applied to the §1.3 four-byte wire cycle (and a
            // direct mirror of the r221 YUY2 macropixel-step body on
            // the §1.2 four-byte YUY2 cycle).
            //
            // `body.len()` is always a multiple of 4 in the in-spec
            // input space: `rgb32_residuals` builds the body as
            // `(n_pixels − 1) × 4` bytes (one wire pixel = 4 codes per
            // §1.3), so `body.len() % 4 == 0`. We keep a 1..=3-byte
            // scalar fall-through for defence-in-depth against future
            // pixel-family extensions, mirroring the r221 / r227 / r239
            // fall-throughs.
            let (s_pos0, s_pos1, s_pos2, s_pos3) = if method.decorrelate() {
                (slot2, slot1, slot3, slot3)
            } else {
                (slot1, slot2, slot3, slot3)
            };
            let body_aligned = body.len() & !3;
            let mut i = 0usize;
            while i < body_aligned {
                let (c0, l0) = lookup_code(s_pos0, body[i])?;
                writer.write_msb(c0, l0);
                let (c1, l1) = lookup_code(s_pos1, body[i + 1])?;
                writer.write_msb(c1, l1);
                let (c2, l2) = lookup_code(s_pos2, body[i + 2])?;
                writer.write_msb(c2, l2);
                let (c3, l3) = lookup_code(s_pos3, body[i + 3])?;
                writer.write_msb(c3, l3);
                i += 4;
            }
            // Scalar fall-through for any 1..=3 trailing bytes
            // (unreachable for valid RGB32 inputs; kept for
            // robustness).
            while i < body.len() {
                let in_pixel = i % 4;
                let slot = match in_pixel {
                    0 => s_pos0,
                    1 => s_pos1,
                    2 => s_pos2,
                    _ => s_pos3,
                };
                let (code, length) = lookup_code(slot, body[i])?;
                writer.write_msb(code, length);
                i += 1;
            }
        }
    }
    out.extend_from_slice(&writer.finish());
    Ok(out)
}

fn lookup_code(table: &SlotEntries, sym: u8) -> Result<(u32, u32)> {
    let entry: HuffEntry = table[sym as usize];
    if entry.length == 0 {
        return Err(Error::invalid(format!(
            "encoder: symbol 0x{:02x} not in Huffman table",
            sym
        )));
    }
    Ok((entry.code, entry.length as u32))
}

#[cfg(test)]
mod round221_yuy2_emit_macropixel_tests {
    //! Round-221 regression guard. The encoder's YUY2 Huffman-emit
    //! loop in [`emit_bitstream_parts`] was rewritten from a per-byte
    //! `match byte_idx & 3` slot dispatch into a macropixel-step body
    //! that emits four codes per outer iteration (Y₁ via slot1, U via
    //! slot2, Y₂ via slot1, V via slot3) — the encoder-side analogue
    //! of round 214's decode-side macropixel rewrite (also branch
    //! elimination on the same 4-byte cycle).
    //!
    //! spec/03 §1.2's three-slot architecture (the wire-format
    //! invariant the rewrite leans on) pins the YUY2 byte → slot
    //! mapping at `+0 (Y₁) → slot1; +1 (U) → slot2; +2 (Y₂) → slot1;
    //! +3 (V) → slot3` for every 4-byte macropixel.
    //!
    //! Coverage:
    //!
    //! - **Encode-then-decode round-trips** for widths bracketing the
    //!   macropixel-step boundary (2 / 4 / 8 / 16). The in-spec input
    //!   space is already `body.len() % 4 == 0` because YUY2 width is
    //!   even per the §2.1.1 macropixel-pair invariant the predict
    //!   pipeline already enforces, but explicit small-width coverage
    //!   exercises the new step body against minimal macropixel
    //!   counts.
    //! - **Wire-byte witness** — the production emit is diffed
    //!   against an inlined copy of the pre-r221 per-byte slot
    //!   dispatch body across Left / Gradient / Median predictors.
    //!   The wire bytes must be byte-identical between the two.
    //! - **V1xCompat path** — slot1 / slot2 / slot3 hold distinct
    //!   tables (slot1 = set A, slot2 = set B, slot3 = set B per
    //!   spec/04 §4.1), so a slot mix-up inside the unrolled body
    //!   would surface as a Huffman-code mismatch on the wire even
    //!   before the round-trip predictor pass.

    use super::*;

    fn synth_yuy2(width: usize, height: usize) -> Vec<u8> {
        // Deterministic xorshift32 ramp; same shape as the
        // round214 decoder-side helper but inlined to keep this
        // module self-contained.
        let mut s: u32 = 0xDEAD_BEEF;
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
        use crate::decoder::decode_frame;
        let pixels = synth_yuy2(width as usize, height as usize);
        let (bih, frame) =
            encode_frame_with_mode(PixelFamily::Yuy2, method, width, height, &pixels, mode)
                .expect("encode");
        let cfg = StreamConfig::parse_bitmapinfoheader(&bih).expect("parse");
        let decoded = decode_frame(&cfg, &frame).expect("decode");
        assert_eq!(decoded.pixels, pixels);
    }

    #[test]
    fn round221_yuy2_left_classic_width_2() {
        // width=2 → row_bytes=4, exactly one macropixel per row.
        // Step body fires exactly (height − 1) × 1 + (W/2 − 1) times
        // depending on field split; the small case exercises the
        // body against minimal macropixel counts.
        rt_yuy2(2, 6, Method::Left, ExtradataMode::ClassicV2);
    }

    #[test]
    fn round221_yuy2_left_classic_width_4() {
        // width=4 → row_bytes=8, two macropixels per row.
        rt_yuy2(4, 4, Method::Left, ExtradataMode::ClassicV2);
    }

    #[test]
    fn round221_yuy2_gradient_custom_width_8() {
        // width=8 / Gradient / CustomV2 — runtime-built Huffman
        // tables (CustomV2 derives per-channel lengths from
        // histograms, so a slot mix-up inside the unrolled body
        // surfaces as a Huffman-code mismatch).
        rt_yuy2(8, 5, Method::Gradient, ExtradataMode::CustomV2);
    }

    #[test]
    fn round221_yuy2_median_v1x_width_8() {
        // V1xCompat path: slot1 = set A, slot2 = set B, slot3 = set
        // B per spec/04 §4.1. Distinct tables-per-slot is the
        // cleanest wire-level witness that the step body wires each
        // residual byte to the right slot.
        rt_yuy2(8, 7, Method::Median, ExtradataMode::V1xCompat);
    }

    #[test]
    fn round221_yuy2_left_classic_width_16_height_3() {
        // Wider row + small height — four macropixels per row across
        // three rows so the step body crosses the row boundary
        // multiple times. The emit loop doesn't depend on row
        // boundaries (the slot mapping is wire-byte-modular, not
        // row-modular), so this also pins the rewrite against a
        // future attempt to re-introduce row-aware state inside the
        // step body.
        rt_yuy2(16, 3, Method::Left, ExtradataMode::ClassicV2);
    }

    /// Reference body: the pre-r221 per-byte slot-dispatch emit loop
    /// inlined verbatim. Drives a wire-byte witness alongside the
    /// production [`emit_bitstream_parts`] over the same `body`,
    /// `slot1`, `slot2`, `slot3` triple. The two streams must be
    /// byte-identical.
    fn ref_emit_yuy2_per_byte(
        seed: &[u8; 4],
        body: &[u8],
        slot1: &SlotEntries,
        slot2: &SlotEntries,
        slot3: &SlotEntries,
    ) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(seed);
        let mut writer = BitWriter::new();
        for (byte_idx, &sym) in (4usize..).zip(body.iter()) {
            let slot = match byte_idx & 3 {
                0 | 2 => slot1,
                1 => slot2,
                _ => slot3,
            };
            let (code, length) = lookup_code(slot, sym).expect("ref lookup");
            writer.write_msb(code, length);
        }
        out.extend_from_slice(&writer.finish());
        out
    }

    #[test]
    fn round221_yuy2_emit_matches_per_byte_reference() {
        // For each (width, height, predictor) triple, drive the
        // residual pipeline twice with identical inputs: once via
        // the production emit_bitstream_parts (round-221 step body),
        // once via the pre-r221 per-byte reference above. The
        // resulting wire frames must be byte-identical — any
        // divergence implies the step body's slot mapping has
        // drifted from the spec/03 §1.2 invariant.
        //
        // We exercise CustomV2 mode specifically because
        // (a) it produces three distinct, content-dependent
        // length tables (slot1 / slot2 / slot3), so a slot mix-up
        // changes code lengths immediately, and (b) it runs the
        // full encode pipeline from `pixels`. The ClassicV2 path
        // would also work but its three tables for YUY2 happen to
        // coincide on enough symbols that mismatches could mask.
        for (w, h, method) in [
            (4u32, 4u32, Method::Left),
            (8, 6, Method::Gradient),
            (8, 6, Method::Median),
            (16, 3, Method::Left),
        ] {
            let pixels = synth_yuy2(w as usize, h as usize);
            // Production wire bytes via the full encode_frame path.
            let (_bih, prod_frame) = encode_frame_with_mode(
                PixelFamily::Yuy2,
                method,
                w,
                h,
                &pixels,
                ExtradataMode::CustomV2,
            )
            .expect("encode");

            // Re-derive the seed + body the encoder fed to
            // emit_bitstream_parts. compute_frame_residuals +
            // compute_lengths_from_body gives us a body identical
            // to the production path; build_three_tables (via
            // CustomV2's emitted extradata) gives us slot1/2/3.
            let frame = compute_frame_residuals(PixelFamily::Yuy2, method, w, h, &pixels)
                .expect("residuals");
            let lengths = compute_lengths_from_body(PixelFamily::Yuy2, method, &frame.body);
            // Build the slot HuffTables from the computed lengths.
            // Matches the v2.x Custom path in encode_with_precomputed.
            let slot1 = HuffTable::build_from_lengths(&lengths[0]).expect("slot1");
            let slot2 = HuffTable::build_from_lengths(&lengths[1]).expect("slot2");
            let slot3 = HuffTable::build_from_lengths(&lengths[2]).expect("slot3");

            // Reference emit using the pre-r221 per-byte body.
            let ref_frame = ref_emit_yuy2_per_byte(
                &frame.seed,
                &frame.body,
                &slot1.entries,
                &slot2.entries,
                &slot3.entries,
            );

            // The production wire bytes start with the bih extradata
            // we discarded; the round we care about is the frame
            // chunk itself, which is what encode_frame_with_mode
            // already returns as the second tuple element. Cross-
            // check production wire equals our local reference
            // assembly using identical tables.
            let prod_emit = emit_bitstream_parts(
                PixelFamily::Yuy2,
                method,
                &frame.seed,
                &frame.body,
                &slot1.entries,
                &slot2.entries,
                &slot3.entries,
            )
            .expect("prod emit");
            assert_eq!(
                prod_emit, ref_frame,
                "round221 wire-byte divergence @ {}x{} {:?}",
                w, h, method
            );
            // And the production end-to-end frame must contain the
            // same emit-region bytes (its tail = prod_emit).
            assert!(
                prod_frame.windows(prod_emit.len()).any(|w| w == prod_emit),
                "round221 production frame does not contain reference emit @ {}x{} {:?}",
                w,
                h,
                method
            );
        }
    }
}

#[cfg(test)]
mod round227_yuy2_histogram_verify_macropixel_tests {
    //! Round-227 regression guard. The encoder's YUY2 histogram body
    //! (`histogramise`) and v1.x verification body (`verify_body_in_table`)
    //! were both rewritten from a per-byte `match byte_idx & 3` slot
    //! dispatch into a 4-byte macropixel-step body, mirroring the
    //! same branch elimination round 214 applied to the decoder's
    //! YUY2 Huffman-decode loop and round 221 applied to the
    //! encoder's YUY2 Huffman-emit loop.
    //!
    //! spec/03 §1.2's three-slot architecture pins the YUY2 wire-byte
    //! → slot mapping at a fixed 4-byte cycle: `+0 (Y₁) → slot1;
    //! +1 (U) → slot2; +2 (Y₂) → slot1; +3 (V) → slot3`. The
    //! histogram body now counts four bytes per outer iteration into
    //! the fixed slots; the verify body now checks four bytes per
    //! outer iteration against the same fixed slot tables.
    //!
    //! Coverage:
    //!
    //! - **Per-byte witness** — both rewrites are diffed against an
    //!   inlined copy of the pre-r227 per-byte slot dispatch over a
    //!   deterministic body. Histograms must be element-wise equal;
    //!   verify must return the same `Result` (and the same first
    //!   offending symbol on the failure path).
    //! - **End-to-end CustomV2 round-trip** at widths bracketing the
    //!   macropixel-step boundary (2 / 4 / 8 / 16) ensures the
    //!   histograms drive the same canonical length tables and the
    //!   wire frames stay bit-identical after the rewrite.
    //! - **End-to-end V1xCompat round-trip** at the same widths
    //!   exercises the verify body: V1xCompat fails closed if any
    //!   residual symbol lands on a slot whose v1.x precomputed code
    //!   table has length 0, so a slot mix-up inside the unrolled
    //!   verify would surface as a false rejection (or, worse, a
    //!   false acceptance that the v1.x decoder would then mis-
    //!   route).

    use super::*;

    fn synth_yuy2(width: usize, height: usize) -> Vec<u8> {
        let mut s: u32 = 0x1234_5678;
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

    /// Reference body: the pre-r227 per-byte slot-dispatch histogram
    /// inlined verbatim. Output must match `histogramise(Yuy2, ...)`
    /// element-by-element across all three slot tables.
    fn ref_histogramise_yuy2_per_byte(body: &[u8]) -> ([u32; 256], [u32; 256], [u32; 256]) {
        let mut h1 = [0u32; 256];
        let mut h2 = [0u32; 256];
        let mut h3 = [0u32; 256];
        for (i, &b) in body.iter().enumerate() {
            let byte_idx = i + 4;
            match byte_idx & 3 {
                0 | 2 => h1[b as usize] += 1,
                1 => h2[b as usize] += 1,
                _ => h3[b as usize] += 1,
            }
        }
        (h1, h2, h3)
    }

    /// Reference body: the pre-r227 per-byte slot-dispatch verify
    /// inlined verbatim. Must return the same `Result` shape as the
    /// production verify, including the first symbol carried in the
    /// error path so the diagnostic text stays stable.
    fn ref_verify_body_in_table_yuy2_per_byte(
        body: &[u8],
        s1: &SlotEntries,
        s2: &SlotEntries,
        s3: &SlotEntries,
    ) -> Result<()> {
        for (byte_idx, (_i, &sym)) in (4usize..).zip(body.iter().enumerate()) {
            let slot = match byte_idx & 3 {
                0 | 2 => s1,
                1 => s2,
                _ => s3,
            };
            if slot[sym as usize].length == 0 {
                return Err(Error::unsupported(format!(
                    "v1.x compat: residual symbol 0x{sym:02x} not in v1.x codebook"
                )));
            }
        }
        Ok(())
    }

    #[test]
    fn round227_yuy2_histogram_matches_per_byte_reference() {
        // Drive the residual pipeline across Left / Gradient / Median
        // at widths 2 / 4 / 8 / 16. For each frame, take the
        // combined_body the encoder feeds to histogramise and compare
        // the production histograms against the per-byte reference.
        for &(w, h) in &[(2u32, 4u32), (4, 4), (8, 6), (16, 3)] {
            for &method in &[Method::Left, Method::Gradient, Method::Median] {
                let pixels = synth_yuy2(w as usize, h as usize);
                let frame = compute_frame_residuals(PixelFamily::Yuy2, method, w, h, &pixels)
                    .expect("residuals");
                let (h1_prod, h2_prod, h3_prod) =
                    histogramise(PixelFamily::Yuy2, method, &frame.body);
                let (h1_ref, h2_ref, h3_ref) = ref_histogramise_yuy2_per_byte(&frame.body);
                assert_eq!(
                    h1_prod, h1_ref,
                    "round227 slot1 histogram drift @ {}x{} {:?}",
                    w, h, method
                );
                assert_eq!(
                    h2_prod, h2_ref,
                    "round227 slot2 histogram drift @ {}x{} {:?}",
                    w, h, method
                );
                assert_eq!(
                    h3_prod, h3_ref,
                    "round227 slot3 histogram drift @ {}x{} {:?}",
                    w, h, method
                );
                // Sanity: histograms must sum to body.len() (every
                // body byte is counted exactly once).
                let total: u64 = h1_prod
                    .iter()
                    .chain(h2_prod.iter())
                    .chain(h3_prod.iter())
                    .map(|&c| c as u64)
                    .sum();
                assert_eq!(
                    total,
                    frame.body.len() as u64,
                    "round227 histogram total != body.len() @ {}x{} {:?}",
                    w,
                    h,
                    method
                );
            }
        }
    }

    #[test]
    fn round227_yuy2_histogram_synth_body_matches_per_byte_reference() {
        // Direct check against a synthetic body that exercises every
        // residue position. Uses a 32-byte body (= 8 macropixels) so
        // the slot positions are densely covered with distinct values
        // — a slot mix-up inside the step body would surface as
        // counts attributed to the wrong slot table.
        let body: Vec<u8> = (0u8..32).collect();
        let (h1_prod, h2_prod, h3_prod) = histogramise(PixelFamily::Yuy2, Method::Left, &body);
        let (h1_ref, h2_ref, h3_ref) = ref_histogramise_yuy2_per_byte(&body);
        assert_eq!(h1_prod, h1_ref);
        assert_eq!(h2_prod, h2_ref);
        assert_eq!(h3_prod, h3_ref);

        // And the fixed expectations:
        // slot1 counts bytes at i = 0, 2, 4, 6, 8, …, 30 (16 values:
        // even indices), each exactly once.
        // slot2 counts bytes at i = 1, 5, 9, 13, 17, 21, 25, 29 (8
        // values), each exactly once.
        // slot3 counts bytes at i = 3, 7, 11, 15, 19, 23, 27, 31 (8
        // values), each exactly once.
        for v in (0u8..32).step_by(2) {
            assert_eq!(
                h1_prod[v as usize], 1,
                "slot1 should hold byte {v} exactly once",
            );
        }
        for v in [1u8, 5, 9, 13, 17, 21, 25, 29] {
            assert_eq!(
                h2_prod[v as usize], 1,
                "slot2 should hold byte {v} exactly once",
            );
        }
        for v in [3u8, 7, 11, 15, 19, 23, 27, 31] {
            assert_eq!(
                h3_prod[v as usize], 1,
                "slot3 should hold byte {v} exactly once",
            );
        }
    }

    #[test]
    fn round227_yuy2_verify_matches_per_byte_reference_success_path() {
        // Build the v1.x YUY2 codebook triple and ask both the
        // production verify and the per-byte reference to walk a
        // synthetic body. Both must accept.
        let tabs = crate::decoder::table_cache::v1x_tables(PixelFamily::Yuy2).expect("v1x build");
        let (s1, s2, s3) = (tabs.slot1.clone(), tabs.slot2.clone(), tabs.slot3.clone());
        let body: Vec<u8> = (0u8..64).collect();
        let prod = verify_body_in_table(
            PixelFamily::Yuy2,
            Method::Left,
            &body,
            &s1.entries,
            &s2.entries,
            &s3.entries,
        );
        let refr =
            ref_verify_body_in_table_yuy2_per_byte(&body, &s1.entries, &s2.entries, &s3.entries);
        match (prod, refr) {
            (Ok(()), Ok(())) => {}
            (Err(e1), Err(e2)) => assert_eq!(format!("{e1}"), format!("{e2}")),
            (a, b) => panic!("round227 verify result drift: prod={a:?} ref={b:?}"),
        }
    }

    #[test]
    fn round227_yuy2_verify_matches_per_byte_reference_failure_path() {
        // Build the v1.x YUY2 codebook triple, then synthesise a
        // body containing a symbol the v1.x set rejects, placed at a
        // known slot position (i % 4 == 1, so the production verify
        // tests it against slot2). The production verify and the
        // per-byte reference must both reject, with the same error
        // text.
        let tabs = crate::decoder::table_cache::v1x_tables(PixelFamily::Yuy2).expect("v1x build");
        let (s1, s2, s3) = (tabs.slot1.clone(), tabs.slot2.clone(), tabs.slot3.clone());
        // Pick a symbol that has length 0 in slot2's v1.x table. The
        // v1.x precomputed-code set B doesn't cover the full 256-
        // symbol space densely on every position, so we scan for an
        // uncovered symbol in slot2.
        let mut bad_sym: Option<u8> = None;
        for s in 0u16..256 {
            let s = s as u8;
            if s2.entries[s as usize].length == 0 {
                bad_sym = Some(s);
                break;
            }
        }
        if let Some(sym) = bad_sym {
            // Body layout: index 1 falls on slot2 (the +1 macropixel
            // position). Pad with a valid slot1 symbol at index 0
            // (length-1 entry by construction — slot1 = set A covers
            // every symbol).
            let mut body = vec![0u8; 8];
            body[1] = sym;
            let prod = verify_body_in_table(
                PixelFamily::Yuy2,
                Method::Left,
                &body,
                &s1.entries,
                &s2.entries,
                &s3.entries,
            );
            let refr = ref_verify_body_in_table_yuy2_per_byte(
                &body,
                &s1.entries,
                &s2.entries,
                &s3.entries,
            );
            match (prod, refr) {
                (Err(e1), Err(e2)) => assert_eq!(format!("{e1}"), format!("{e2}")),
                (a, b) => panic!("round227 verify failure-path drift: prod={a:?} ref={b:?}",),
            }
        }
        // If every symbol is covered by slot2 (set B), the failure
        // path can't be triggered through this fixture — that's
        // fine; the success-path test above is the primary guard.
    }

    #[test]
    fn round227_yuy2_custom_v2_roundtrip_width_2() {
        // Exercises histogramise via the CustomV2 path: the encoder
        // builds per-slot length tables from the slot-attributed
        // histograms, so any drift in the histograms would change
        // the emitted lengths and break the round-trip.
        rt_yuy2_custom(2, 6, Method::Left);
    }

    #[test]
    fn round227_yuy2_custom_v2_roundtrip_width_4() {
        rt_yuy2_custom(4, 4, Method::Gradient);
    }

    #[test]
    fn round227_yuy2_custom_v2_roundtrip_width_8() {
        rt_yuy2_custom(8, 6, Method::Median);
    }

    #[test]
    fn round227_yuy2_custom_v2_roundtrip_width_16() {
        rt_yuy2_custom(16, 3, Method::Left);
    }

    #[test]
    fn round227_yuy2_v1x_compat_roundtrip_width_2() {
        // Exercises verify_body_in_table via the V1xCompat path.
        rt_yuy2_v1x(2, 6, Method::Left);
    }

    #[test]
    fn round227_yuy2_v1x_compat_roundtrip_width_4() {
        rt_yuy2_v1x(4, 4, Method::Gradient);
    }

    #[test]
    fn round227_yuy2_v1x_compat_roundtrip_width_8() {
        rt_yuy2_v1x(8, 6, Method::Median);
    }

    #[test]
    fn round227_yuy2_v1x_compat_roundtrip_width_16() {
        rt_yuy2_v1x(16, 3, Method::Left);
    }

    fn rt_yuy2_custom(width: u32, height: u32, method: Method) {
        use crate::decoder::decode_frame;
        let pixels = synth_yuy2(width as usize, height as usize);
        let (bih, frame) = encode_frame_with_mode(
            PixelFamily::Yuy2,
            method,
            width,
            height,
            &pixels,
            ExtradataMode::CustomV2,
        )
        .expect("encode");
        let cfg = StreamConfig::parse_bitmapinfoheader(&bih).expect("parse");
        let decoded = decode_frame(&cfg, &frame).expect("decode");
        assert_eq!(decoded.pixels, pixels);
    }

    fn rt_yuy2_v1x(width: u32, height: u32, method: Method) {
        use crate::decoder::decode_frame;
        let pixels = synth_yuy2(width as usize, height as usize);
        let (bih, frame) = encode_frame_with_mode(
            PixelFamily::Yuy2,
            method,
            width,
            height,
            &pixels,
            ExtradataMode::V1xCompat,
        )
        .expect("encode");
        let cfg = StreamConfig::parse_bitmapinfoheader(&bih).expect("parse");
        let decoded = decode_frame(&cfg, &frame).expect("decode");
        assert_eq!(decoded.pixels, pixels);
    }
}

#[cfg(test)]
mod round239_rgb24_emit_pixel_step_tests {
    //! Round-239 regression guard. The encoder's RGB24 Huffman-emit
    //! loop in [`emit_bitstream_parts`] was rewritten from a per-byte
    //! `match i % 3` slot dispatch (plus a per-iteration
    //! `method.decorrelate()` branch) into a pixel-step body that
    //! resolves the three slot pointers once at function entry and
    //! emits three codes per outer iteration.
    //!
    //! spec/03 §1.1 pins RGB24 at exactly three Huffman codewords per
    //! pixel (the §3.2/§3.3 spec/02 correction); §1.2 + the table at
    //! the end of §1.2 fixes the position → slot mapping at:
    //!
    //! - no-decorr methods (`Left`, `PredictOld`): pos +0 (B) → slot1;
    //!   pos +1 (G) → slot2; pos +2 (R) → slot3.
    //! - decorr methods (`LeftDecorr`, `GradientDecorr`): pos +0 (G) →
    //!   slot2; pos +1 (B−G) → slot1; pos +2 (R−G) → slot3.
    //!
    //! Coverage:
    //!
    //! - **Encode-then-decode round-trips** across the four legal RGB24
    //!   methods (`Left`, `PredictOld`, `LeftDecorr`, `GradientDecorr`)
    //!   at widths bracketing the pixel-step boundary (1 / 2 / 4 / 8).
    //!   The in-spec input space is already `body.len() % 3 == 0`
    //!   because the rgb24 body is `(n_pixels − 1) × 3` bytes, but
    //!   explicit small-width coverage exercises the new step body
    //!   against minimal pixel counts.
    //! - **Wire-byte witness** — the production emit is diffed against
    //!   an inlined copy of the pre-r239 per-byte slot dispatch body
    //!   across `Left` / `LeftDecorr` / `GradientDecorr` predictors,
    //!   using `CustomV2` so the three slot tables are content-distinct
    //!   (a slot mix-up would surface as a Huffman-code mismatch on the
    //!   wire even before the round-trip predictor pass).
    //! - **V1xCompat path** — exercises the same step body against the
    //!   `(A, A, A)` v1.x precomputed-code triple (spec/04 §4.1) so
    //!   the rewrite is locked against both content-distinct and
    //!   content-identical slot triples.
    //!
    //! Wire-identical to round 234 — every pre-existing RGB24
    //! round-trip + the AVI-lockstep RGB24 tests stay green.
    use super::*;
    use crate::decoder::decode_frame;

    fn synth_rgb24(width: usize, height: usize) -> Vec<u8> {
        // Deterministic xorshift32 ramp; same pattern shape as the
        // r214 / r221 YUY2 helpers but inlined here so this module
        // stays self-contained.
        let mut s: u32 = 0xCAFE_F00D;
        let n = width * height * 3;
        let mut out = vec![0u8; n];
        for px in out.iter_mut() {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            *px = s as u8;
        }
        out
    }

    fn rt_rgb24(width: u32, height: u32, method: Method, mode: ExtradataMode) {
        let pixels = synth_rgb24(width as usize, height as usize);
        let (bih, frame) =
            encode_frame_with_mode(PixelFamily::Rgb24, method, width, height, &pixels, mode)
                .expect("encode");
        let cfg = StreamConfig::parse_bitmapinfoheader(&bih).expect("parse");
        let decoded = decode_frame(&cfg, &frame).expect("decode");
        assert_eq!(decoded.pixels, pixels);
    }

    #[test]
    fn round239_rgb24_left_classic_width_1() {
        // width=1, height=3 → n_pixels=3, body=(3-1)×3=6 bytes. Two
        // pixel-step iterations; minimum non-trivial body.
        rt_rgb24(1, 3, Method::Left, ExtradataMode::ClassicV2);
    }

    #[test]
    fn round239_rgb24_left_classic_width_2() {
        rt_rgb24(2, 3, Method::Left, ExtradataMode::ClassicV2);
    }

    #[test]
    fn round239_rgb24_left_classic_width_4() {
        rt_rgb24(4, 4, Method::Left, ExtradataMode::ClassicV2);
    }

    #[test]
    fn round239_rgb24_left_classic_width_8() {
        // Wider raster — exercises the step body across multiple rows
        // (the slot mapping is wire-pixel-modular, not row-modular, so
        // this also pins the rewrite against any future attempt to
        // re-introduce row-aware state inside the step body).
        rt_rgb24(8, 4, Method::Left, ExtradataMode::ClassicV2);
    }

    #[test]
    fn round239_rgb24_predict_old_classic() {
        // `PredictOld` is the spec/01 §3.1 signed `−2` method byte —
        // shares the no-decorrelate slot triple with `Left`; this test
        // pins the rewrite against the alternate no-decorr method
        // entry.
        rt_rgb24(4, 4, Method::PredictOld, ExtradataMode::ClassicV2);
    }

    #[test]
    fn round239_rgb24_left_decorr_classic() {
        // Switches to the (slot2, slot1, slot3) decorr triple; a
        // mis-resolved triple would corrupt the round-trip.
        rt_rgb24(4, 4, Method::LeftDecorr, ExtradataMode::ClassicV2);
    }

    #[test]
    fn round239_rgb24_gradient_decorr_classic() {
        rt_rgb24(4, 4, Method::GradientDecorr, ExtradataMode::ClassicV2);
    }

    #[test]
    fn round239_rgb24_left_v1x_compat() {
        // V1xCompat: slot1 = slot2 = slot3 = set A (spec/04 §4.1).
        // Content-identical triple — the step body must still walk the
        // three positions correctly even when all three tables are
        // the same instance.
        rt_rgb24(4, 4, Method::Left, ExtradataMode::V1xCompat);
    }

    /// Reference body: the pre-r239 per-byte slot-dispatch emit loop,
    /// inlined verbatim. Compared against the production emit over a
    /// deterministic body stream to lock the rewrite at byte equality.
    fn ref_emit_rgb24_per_byte_loop(
        method: Method,
        body: &[u8],
        slot1: &SlotEntries,
        slot2: &SlotEntries,
        slot3: &SlotEntries,
    ) -> Vec<u8> {
        let mut writer = BitWriter::new();
        for (i, &sym) in body.iter().enumerate() {
            let in_pixel = i % 3;
            let slot = if method.decorrelate() {
                match in_pixel {
                    0 => slot2,
                    1 => slot1,
                    _ => slot3,
                }
            } else {
                match in_pixel {
                    0 => slot1,
                    1 => slot2,
                    _ => slot3,
                }
            };
            let (code, length) = lookup_code(slot, sym).expect("ref lookup");
            writer.write_msb(code, length);
        }
        writer.finish()
    }

    #[test]
    fn round239_rgb24_emit_matches_per_byte_reference() {
        // Encode under CustomV2 so the three slot tables are
        // content-distinct (per-channel Huffman-built from the actual
        // residual histograms — a slot mix-up between the production
        // step body and the reference per-byte body would surface as a
        // Huffman-code mismatch in the emitted bit stream before any
        // round-trip predictor pass.
        //
        // We can't call `emit_bitstream_parts` directly on a forged
        // body without rebuilding the tables, so the witness compares
        // the production frame's emitted bits (everything after the
        // 4-byte seed) to the reference per-byte emit driven by the
        // same residuals body + the same tables. Both must be
        // byte-identical.
        for (w, h, method) in [
            (4u32, 4u32, Method::Left),
            (4, 4, Method::LeftDecorr),
            (4, 4, Method::GradientDecorr),
        ] {
            let pixels = synth_rgb24(w as usize, h as usize);
            // Production frame: contains the seed + the production
            // emit-pass output.
            let (_, frame) = encode_frame_with_mode(
                PixelFamily::Rgb24,
                method,
                w,
                h,
                &pixels,
                ExtradataMode::CustomV2,
            )
            .expect("encode");
            // Re-derive the residuals body + the per-channel tables the
            // encoder built for this frame so we can run the reference
            // per-byte emit against the same inputs.
            let residuals = compute_frame_residuals(PixelFamily::Rgb24, method, w, h, &pixels)
                .expect("residuals");
            let (h1, h2, h3) = histogramise(PixelFamily::Rgb24, method, &residuals.body);
            let len1 = compute_canonical_lengths(&h1).expect("lens1");
            let len2 = compute_canonical_lengths(&h2).expect("lens2");
            let len3 = compute_canonical_lengths(&h3).expect("lens3");
            let s1 = HuffTable::build_from_lengths(&len1).expect("t1");
            let s2 = HuffTable::build_from_lengths(&len2).expect("t2");
            let s3 = HuffTable::build_from_lengths(&len3).expect("t3");
            let ref_bits = ref_emit_rgb24_per_byte_loop(
                method,
                &residuals.body,
                &s1.entries,
                &s2.entries,
                &s3.entries,
            );
            // The production frame begins with the 4-byte uncompressed
            // seed; the emitted bit stream follows.
            let prod_bits = &frame[4..];
            assert_eq!(
                prod_bits,
                ref_bits.as_slice(),
                "round-239 step body must emit bit-identical wire bytes to the pre-r239 per-byte slot dispatch (method = {:?})",
                method
            );
        }
    }
}

#[cfg(test)]
mod round242_rgb24_histogram_pixel_step_tests {
    //! Round-242 regression guard. The encoder's RGB24 histogram body
    //! (`histogramise`) was rewritten from a per-byte `match i % 3`
    //! slot dispatch (plus a per-iteration `method.decorrelate()`
    //! branch) into a pixel-step body that resolves the three
    //! per-position histogram references once at function entry and
    //! counts three bytes per outer iteration.
    //!
    //! Histogram-side companion to round 239's RGB24 emit rewrite (and
    //! mirror of round 227's YUY2 histogram macropixel-step body
    //! applied to the §1.2 three-byte RGB24 wire cycle).
    //!
    //! spec/03 §1.1 pins RGB24 at exactly three Huffman codewords per
    //! pixel (the §3.2/§3.3 spec/02 correction); §1.2 fixes the
    //! position → slot mapping at:
    //!
    //! - no-decorr methods (`Left`, `PredictOld`): pos +0 (B) → slot1;
    //!   pos +1 (G) → slot2; pos +2 (R) → slot3.
    //! - decorr methods (`LeftDecorr`, `GradientDecorr`): pos +0 (G) →
    //!   slot2; pos +1 (B−G) → slot1; pos +2 (R−G) → slot3.
    //!
    //! Coverage:
    //!
    //! - **Per-byte witness** — the production histogram triple is
    //!   diffed element-by-element against an inlined copy of the
    //!   pre-r242 per-byte slot-dispatch body over both real residual
    //!   bodies (taken from `compute_frame_residuals`) and a synthetic
    //!   `0..96` body that densely covers every residue position. Any
    //!   slot mix-up inside the step body would surface as counts
    //!   attributed to the wrong histogram.
    //! - **Histogram total sanity** — `h1 + h2 + h3` summed across all
    //!   256 buckets must equal `body.len()` (every body byte counted
    //!   exactly once). This catches a step body that drops or
    //!   double-counts bytes even if the slot attribution happens to
    //!   match the reference on a particular fixture.
    //! - **End-to-end CustomV2 round-trips** at widths 1 / 2 / 4 / 8
    //!   across `Left` / `LeftDecorr` / `GradientDecorr`: the CustomV2
    //!   path builds the per-slot length tables straight from
    //!   `histogramise`, so any histogram drift would change the
    //!   emitted lengths and break the round-trip.
    //!
    //! Wire-identical to round 239 — every pre-existing RGB24 round-
    //! trip + the AVI-lockstep RGB24 tests stay green.

    use super::*;

    fn synth_rgb24(width: usize, height: usize) -> Vec<u8> {
        // Deterministic xorshift32 ramp — same shape as the r239 helper
        // but inlined here so this module stays self-contained.
        let mut s: u32 = 0xDEAD_BEEF;
        let n = width * height * 3;
        let mut out = vec![0u8; n];
        for px in out.iter_mut() {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            *px = s as u8;
        }
        out
    }

    /// Reference body: the pre-r242 per-byte slot-dispatch histogram
    /// inlined verbatim. Output must match `histogramise(Rgb24, method,
    /// ...)` element-by-element across all three histograms.
    fn ref_histogramise_rgb24_per_byte(
        method: Method,
        body: &[u8],
    ) -> ([u32; 256], [u32; 256], [u32; 256]) {
        let mut h1 = [0u32; 256];
        let mut h2 = [0u32; 256];
        let mut h3 = [0u32; 256];
        for (i, &b) in body.iter().enumerate() {
            let in_pixel = i % 3;
            if method.decorrelate() {
                match in_pixel {
                    0 => h2[b as usize] += 1, // G
                    1 => h1[b as usize] += 1, // B-G
                    _ => h3[b as usize] += 1, // R-G
                }
            } else {
                match in_pixel {
                    0 => h1[b as usize] += 1,
                    1 => h2[b as usize] += 1,
                    _ => h3[b as usize] += 1,
                }
            }
        }
        (h1, h2, h3)
    }

    #[test]
    fn round242_rgb24_histogram_matches_per_byte_reference() {
        // Drive the residual pipeline across Left / LeftDecorr /
        // GradientDecorr at widths 1 / 2 / 4 / 8. For each frame, take
        // the combined_body the encoder feeds to histogramise and
        // compare the production histograms against the per-byte
        // reference.
        for &(w, h) in &[(1u32, 3u32), (2, 3), (4, 4), (8, 4)] {
            for &method in &[Method::Left, Method::LeftDecorr, Method::GradientDecorr] {
                let pixels = synth_rgb24(w as usize, h as usize);
                let frame = compute_frame_residuals(PixelFamily::Rgb24, method, w, h, &pixels)
                    .expect("residuals");
                let (h1_prod, h2_prod, h3_prod) =
                    histogramise(PixelFamily::Rgb24, method, &frame.body);
                let (h1_ref, h2_ref, h3_ref) = ref_histogramise_rgb24_per_byte(method, &frame.body);
                assert_eq!(
                    h1_prod, h1_ref,
                    "round242 slot1 histogram drift @ {}x{} {:?}",
                    w, h, method
                );
                assert_eq!(
                    h2_prod, h2_ref,
                    "round242 slot2 histogram drift @ {}x{} {:?}",
                    w, h, method
                );
                assert_eq!(
                    h3_prod, h3_ref,
                    "round242 slot3 histogram drift @ {}x{} {:?}",
                    w, h, method
                );
                // Sanity: histograms must sum to body.len() (every body
                // byte is counted exactly once).
                let total: u64 = h1_prod
                    .iter()
                    .chain(h2_prod.iter())
                    .chain(h3_prod.iter())
                    .map(|&c| c as u64)
                    .sum();
                assert_eq!(
                    total,
                    frame.body.len() as u64,
                    "round242 histogram total != body.len() @ {}x{} {:?}",
                    w,
                    h,
                    method
                );
            }
        }
    }

    #[test]
    fn round242_rgb24_histogram_synth_body_matches_per_byte_reference_no_decorr() {
        // Direct check against a synthetic body that exercises every
        // residue position. Uses a 96-byte body (= 32 wire pixels) so
        // the slot positions are densely covered with distinct values
        // — a slot mix-up inside the step body would surface as counts
        // attributed to the wrong histogram. No-decorr triple:
        // (slot1, slot2, slot3) at (+0, +1, +2).
        let body: Vec<u8> = (0u8..96).collect();
        let (h1_prod, h2_prod, h3_prod) = histogramise(PixelFamily::Rgb24, Method::Left, &body);
        let (h1_ref, h2_ref, h3_ref) = ref_histogramise_rgb24_per_byte(Method::Left, &body);
        assert_eq!(h1_prod, h1_ref);
        assert_eq!(h2_prod, h2_ref);
        assert_eq!(h3_prod, h3_ref);

        // Fixed expectations: bytes at i = 0, 3, 6, ..., 93 fall on +0
        // (slot1 under no-decorr); i = 1, 4, 7, ..., 94 fall on +1
        // (slot2); i = 2, 5, 8, ..., 95 fall on +2 (slot3). Each value
        // appears exactly once at its assigned position.
        for v in (0u8..96).step_by(3) {
            assert_eq!(
                h1_prod[v as usize], 1,
                "no-decorr slot1 should hold byte {v} exactly once",
            );
        }
        for v in (1u8..96).step_by(3) {
            assert_eq!(
                h2_prod[v as usize], 1,
                "no-decorr slot2 should hold byte {v} exactly once",
            );
        }
        for v in (2u8..96).step_by(3) {
            assert_eq!(
                h3_prod[v as usize], 1,
                "no-decorr slot3 should hold byte {v} exactly once",
            );
        }
    }

    #[test]
    fn round242_rgb24_histogram_synth_body_matches_per_byte_reference_decorr() {
        // Same synthetic body, decorr triple: (slot2, slot1, slot3) at
        // (+0, +1, +2). A slot-swap regression between no-decorr and
        // decorr would surface as the slot1/slot2 columns landing in
        // the wrong histogram.
        let body: Vec<u8> = (0u8..96).collect();
        let (h1_prod, h2_prod, h3_prod) =
            histogramise(PixelFamily::Rgb24, Method::LeftDecorr, &body);
        let (h1_ref, h2_ref, h3_ref) = ref_histogramise_rgb24_per_byte(Method::LeftDecorr, &body);
        assert_eq!(h1_prod, h1_ref);
        assert_eq!(h2_prod, h2_ref);
        assert_eq!(h3_prod, h3_ref);

        // Fixed expectations for the decorr triple: pos +0 → slot2,
        // pos +1 → slot1, pos +2 → slot3.
        for v in (0u8..96).step_by(3) {
            assert_eq!(
                h2_prod[v as usize], 1,
                "decorr slot2 should hold pos-+0 byte {v} exactly once",
            );
        }
        for v in (1u8..96).step_by(3) {
            assert_eq!(
                h1_prod[v as usize], 1,
                "decorr slot1 should hold pos-+1 byte {v} exactly once",
            );
        }
        for v in (2u8..96).step_by(3) {
            assert_eq!(
                h3_prod[v as usize], 1,
                "decorr slot3 should hold pos-+2 byte {v} exactly once",
            );
        }
    }

    fn rt_rgb24_custom(width: u32, height: u32, method: Method) {
        use crate::decoder::decode_frame;
        let pixels = synth_rgb24(width as usize, height as usize);
        let (bih, frame) = encode_frame_with_mode(
            PixelFamily::Rgb24,
            method,
            width,
            height,
            &pixels,
            ExtradataMode::CustomV2,
        )
        .expect("encode");
        let cfg = StreamConfig::parse_bitmapinfoheader(&bih).expect("parse");
        let decoded = decode_frame(&cfg, &frame).expect("decode");
        assert_eq!(decoded.pixels, pixels);
    }

    #[test]
    fn round242_rgb24_custom_v2_roundtrip_width_1() {
        // width=1, height=3 → n_pixels=3, body=(3-1)×3=6 bytes. Two
        // pixel-step iterations; minimum non-trivial body.
        rt_rgb24_custom(1, 3, Method::Left);
    }

    #[test]
    fn round242_rgb24_custom_v2_roundtrip_width_2() {
        rt_rgb24_custom(2, 3, Method::LeftDecorr);
    }

    #[test]
    fn round242_rgb24_custom_v2_roundtrip_width_4() {
        rt_rgb24_custom(4, 4, Method::GradientDecorr);
    }

    #[test]
    fn round242_rgb24_custom_v2_roundtrip_width_8() {
        // Wider raster — exercises the step body across multiple rows.
        rt_rgb24_custom(8, 4, Method::Left);
    }
}

#[cfg(test)]
mod round245_rgb32_emit_pixel_step_tests {
    //! Round-245 regression guard. The encoder's RGB32 Huffman-emit
    //! loop in [`emit_bitstream_parts`] was rewritten from a per-byte
    //! `match i % 4` slot dispatch (plus a per-iteration
    //! `method.decorrelate()` branch — two per-byte branches the
    //! optimiser could not eliminate because the iterator state
    //! changed every step) into a pixel-step body that resolves the
    //! four slot pointers once at function entry and emits four codes
    //! per outer iteration straight-line.
    //!
    //! spec/03 §1.3 pins RGB32 at exactly four Huffman codewords per
    //! pixel; §1.2 (table at end of §1.2 + the alpha-shares-slot-3
    //! evidence at `@0x10001b21` and `@0x10001c6d`) fixes the position
    //! → slot mapping at:
    //!
    //! - no-decorr methods (`Left`, `PredictOld`): pos +0 (B) → slot1;
    //!   pos +1 (G) → slot2; pos +2 (R) → slot3; pos +3 (A) → slot3.
    //! - decorr methods (`LeftDecorr`, `GradientDecorr`): pos +0 (G) →
    //!   slot2; pos +1 (B−G) → slot1; pos +2 (R−G) → slot3; pos +3 (A)
    //!   → slot3.
    //!
    //! Coverage:
    //!
    //! - **Encode-then-decode round-trips** across the four legal RGB32
    //!   methods (`Left`, `PredictOld`, `LeftDecorr`, `GradientDecorr`)
    //!   at widths bracketing the pixel-step boundary (1 / 2 / 4 / 8).
    //!   The in-spec input space is already `body.len() % 4 == 0`
    //!   because the rgb32 body is `(n_pixels − 1) × 4` bytes, but
    //!   explicit small-width coverage exercises the new step body
    //!   against minimal pixel counts.
    //! - **Wire-byte witness** — the production emit is diffed against
    //!   an inlined copy of the pre-r245 per-byte slot dispatch body
    //!   across `Left` / `LeftDecorr` / `GradientDecorr` predictors,
    //!   using `CustomV2` so the three slot tables are content-distinct
    //!   (a slot mix-up would surface as a Huffman-code mismatch on the
    //!   wire even before the round-trip predictor pass). Because the
    //!   alpha and R/(R−G) positions both share slot3, the witness also
    //!   covers the "two positions, one table" case spec/03 §1.2 calls
    //!   out for the A residual.
    //! - **V1xCompat path** — exercises the same step body against the
    //!   `(A, A, A)` v1.x precomputed-code triple (spec/04 §4.1) so
    //!   the rewrite is locked against both content-distinct and
    //!   content-identical slot quadruples.
    //!
    //! Wire-identical to round 242 — every pre-existing RGB32
    //! round-trip stays green.
    use super::*;
    use crate::decoder::decode_frame;

    fn synth_rgb32(width: usize, height: usize) -> Vec<u8> {
        // Deterministic xorshift32 ramp; same shape as the r239 RGB24
        // helper but 4 bytes per pixel so alpha varies too.
        let mut s: u32 = 0xC0DE_BABE;
        let n = width * height * 4;
        let mut out = vec![0u8; n];
        for px in out.iter_mut() {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            *px = s as u8;
        }
        out
    }

    fn rt_rgb32(width: u32, height: u32, method: Method, mode: ExtradataMode) {
        let pixels = synth_rgb32(width as usize, height as usize);
        let (bih, frame) =
            encode_frame_with_mode(PixelFamily::Rgb32, method, width, height, &pixels, mode)
                .expect("encode");
        let cfg = StreamConfig::parse_bitmapinfoheader(&bih).expect("parse");
        let decoded = decode_frame(&cfg, &frame).expect("decode");
        assert_eq!(decoded.pixels, pixels);
    }

    #[test]
    fn round245_rgb32_left_classic_width_1() {
        // width=1, height=3 → n_pixels=3, body=(3-1)×4=8 bytes. Two
        // pixel-step iterations; minimum non-trivial body.
        rt_rgb32(1, 3, Method::Left, ExtradataMode::ClassicV2);
    }

    #[test]
    fn round245_rgb32_left_classic_width_2() {
        rt_rgb32(2, 3, Method::Left, ExtradataMode::ClassicV2);
    }

    #[test]
    fn round245_rgb32_left_classic_width_4() {
        rt_rgb32(4, 4, Method::Left, ExtradataMode::ClassicV2);
    }

    #[test]
    fn round245_rgb32_left_classic_width_8() {
        // Wider raster — exercises the step body across multiple rows
        // (the slot mapping is wire-pixel-modular, not row-modular, so
        // this also pins the rewrite against any future attempt to
        // re-introduce row-aware state inside the step body).
        rt_rgb32(8, 4, Method::Left, ExtradataMode::ClassicV2);
    }

    #[test]
    fn round245_rgb32_predict_old_classic() {
        // `PredictOld` is the spec/01 §3.1 signed `−2` method byte —
        // shares the no-decorrelate slot quadruple with `Left`; pins
        // the rewrite against the alternate no-decorr method entry.
        rt_rgb32(4, 4, Method::PredictOld, ExtradataMode::ClassicV2);
    }

    #[test]
    fn round245_rgb32_left_decorr_classic() {
        // Switches to the (slot2, slot1, slot3, slot3) decorr
        // quadruple; a mis-resolved quadruple would corrupt the
        // round-trip.
        rt_rgb32(4, 4, Method::LeftDecorr, ExtradataMode::ClassicV2);
    }

    #[test]
    fn round245_rgb32_gradient_decorr_classic() {
        rt_rgb32(4, 4, Method::GradientDecorr, ExtradataMode::ClassicV2);
    }

    #[test]
    fn round245_rgb32_left_v1x_compat() {
        // V1xCompat: slot1 = slot2 = slot3 = set A (spec/04 §4.1).
        // Content-identical triple — the step body must still walk all
        // four positions correctly even when every slot pointer is the
        // same instance (and the alpha-shares-slot3 reuse degenerates
        // to a third reference to that single table).
        rt_rgb32(4, 4, Method::Left, ExtradataMode::V1xCompat);
    }

    /// Reference body: the pre-r245 per-byte slot-dispatch emit loop,
    /// inlined verbatim. Compared against the production emit over a
    /// deterministic body stream to lock the rewrite at byte equality.
    fn ref_emit_rgb32_per_byte_loop(
        method: Method,
        body: &[u8],
        slot1: &SlotEntries,
        slot2: &SlotEntries,
        slot3: &SlotEntries,
    ) -> Vec<u8> {
        let mut writer = BitWriter::new();
        for (i, &sym) in body.iter().enumerate() {
            let in_pixel = i % 4;
            let slot = if method.decorrelate() {
                match in_pixel {
                    0 => slot2,
                    1 => slot1,
                    _ => slot3,
                }
            } else {
                match in_pixel {
                    0 => slot1,
                    1 => slot2,
                    _ => slot3,
                }
            };
            let (code, length) = lookup_code(slot, sym).expect("ref lookup");
            writer.write_msb(code, length);
        }
        writer.finish()
    }

    #[test]
    fn round245_rgb32_emit_matches_per_byte_reference() {
        // Encode under CustomV2 so the three slot tables are
        // content-distinct (per-channel Huffman-built from the actual
        // residual histograms — a slot mix-up between the production
        // step body and the reference per-byte body would surface as a
        // Huffman-code mismatch in the emitted bit stream before any
        // round-trip predictor pass.
        //
        // We can't call `emit_bitstream_parts` directly on a forged
        // body without rebuilding the tables, so the witness compares
        // the production frame's emitted bits (everything after the
        // 4-byte seed) to the reference per-byte emit driven by the
        // same residuals body + the same tables. Both must be
        // byte-identical.
        for (w, h, method) in [
            (4u32, 4u32, Method::Left),
            (4, 4, Method::LeftDecorr),
            (4, 4, Method::GradientDecorr),
        ] {
            let pixels = synth_rgb32(w as usize, h as usize);
            // Production frame: contains the seed + the production
            // emit-pass output.
            let (_, frame) = encode_frame_with_mode(
                PixelFamily::Rgb32,
                method,
                w,
                h,
                &pixels,
                ExtradataMode::CustomV2,
            )
            .expect("encode");
            // Re-derive the residuals body + the per-channel tables the
            // encoder built for this frame so we can run the reference
            // per-byte emit against the same inputs.
            let residuals = compute_frame_residuals(PixelFamily::Rgb32, method, w, h, &pixels)
                .expect("residuals");
            let (h1, h2, h3) = histogramise(PixelFamily::Rgb32, method, &residuals.body);
            let len1 = compute_canonical_lengths(&h1).expect("lens1");
            let len2 = compute_canonical_lengths(&h2).expect("lens2");
            let len3 = compute_canonical_lengths(&h3).expect("lens3");
            let s1 = HuffTable::build_from_lengths(&len1).expect("t1");
            let s2 = HuffTable::build_from_lengths(&len2).expect("t2");
            let s3 = HuffTable::build_from_lengths(&len3).expect("t3");
            let ref_bits = ref_emit_rgb32_per_byte_loop(
                method,
                &residuals.body,
                &s1.entries,
                &s2.entries,
                &s3.entries,
            );
            // The production frame begins with the 4-byte uncompressed
            // seed; the emitted bit stream follows.
            let prod_bits = &frame[4..];
            assert_eq!(
                prod_bits,
                ref_bits.as_slice(),
                "round-245 step body must emit bit-identical wire bytes to the pre-r245 per-byte slot dispatch (method = {:?})",
                method
            );
        }
    }
}

#[cfg(test)]
mod round250_rgb32_histogram_pixel_step_tests {
    //! Round-250 regression guard. The encoder's RGB32 histogram body
    //! (`histogramise`) was rewritten from a per-byte `match i % 4`
    //! slot dispatch (plus a per-iteration `method.decorrelate()`
    //! branch) into a pixel-step body that resolves the four
    //! per-position histogram references once at function entry and
    //! counts four bytes per outer iteration.
    //!
    //! Histogram-side companion to round 245's RGB32 emit rewrite (and
    //! direct mirror of round 242's RGB24 histogram pixel-step body
    //! applied to the §1.3 four-byte RGB32 wire cycle).
    //!
    //! spec/03 §1.3 pins RGB32 at exactly four Huffman codewords per
    //! pixel; §1.2 (table at end of §1.2 + the alpha-shares-slot-3
    //! evidence at `@0x10001b21` and `@0x10001c6d`) fixes the position
    //! → slot mapping at:
    //!
    //! - no-decorr methods (`Left`, `PredictOld`): pos +0 (B) → slot1;
    //!   pos +1 (G) → slot2; pos +2 (R) → slot3; pos +3 (A) → slot3.
    //! - decorr methods (`LeftDecorr`, `GradientDecorr`): pos +0 (G) →
    //!   slot2; pos +1 (B−G) → slot1; pos +2 (R−G) → slot3; pos +3 (A)
    //!   → slot3.
    //!
    //! Coverage:
    //!
    //! - **Per-byte witness** — the production histogram triple is
    //!   diffed element-by-element against an inlined copy of the
    //!   pre-r250 per-byte slot-dispatch body over both real residual
    //!   bodies (taken from `compute_frame_residuals`) and a synthetic
    //!   `0..128` body that densely covers every residue position
    //!   across 32 wire pixels. Any slot mix-up inside the step body —
    //!   including a swap between the pos +2 (R / R−G) and pos +3 (A)
    //!   slot-3 increments — would surface as counts attributed to the
    //!   wrong histogram.
    //! - **Histogram total sanity** — `h1 + h2 + h3` summed across all
    //!   256 buckets must equal `body.len()` (every body byte counted
    //!   exactly once). This catches a step body that drops or
    //!   double-counts bytes even if the slot attribution happens to
    //!   match the reference on a particular fixture.
    //! - **Alpha-shares-slot-3 aggregation** — pos +2 and pos +3 both
    //!   feed slot 3 per §1.2; a dedicated test fixture lays the same
    //!   value at both positions and verifies slot 3's bucket
    //!   accumulates two increments per pixel.
    //! - **End-to-end CustomV2 round-trips** at widths 1 / 2 / 4 / 8
    //!   across `Left` / `LeftDecorr` / `GradientDecorr`: the CustomV2
    //!   path builds the per-slot length tables straight from
    //!   `histogramise`, so any histogram drift would change the
    //!   emitted lengths and break the round-trip.
    //!
    //! Wire-identical to round 245 — every pre-existing RGB32 round-
    //! trip + the AVI-lockstep RGB32 tests stay green.

    use super::*;

    fn synth_rgb32(width: usize, height: usize) -> Vec<u8> {
        // Deterministic xorshift32 ramp — same shape as the r239 / r242
        // helpers but inlined here so this module stays self-contained.
        let mut s: u32 = 0xDEAD_BEEF;
        let n = width * height * 4;
        let mut out = vec![0u8; n];
        for px in out.iter_mut() {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            *px = s as u8;
        }
        out
    }

    /// Reference body: the pre-r250 per-byte slot-dispatch histogram
    /// inlined verbatim. Output must match `histogramise(Rgb32, method,
    /// ...)` element-by-element across all three histograms.
    fn ref_histogramise_rgb32_per_byte(
        method: Method,
        body: &[u8],
    ) -> ([u32; 256], [u32; 256], [u32; 256]) {
        let mut h1 = [0u32; 256];
        let mut h2 = [0u32; 256];
        let mut h3 = [0u32; 256];
        for (i, &b) in body.iter().enumerate() {
            let in_pixel = i % 4;
            if method.decorrelate() {
                match in_pixel {
                    0 => h2[b as usize] += 1, // G
                    1 => h1[b as usize] += 1, // B-G
                    2 => h3[b as usize] += 1, // R-G
                    _ => h3[b as usize] += 1, // A → slot3
                }
            } else {
                match in_pixel {
                    0 => h1[b as usize] += 1,
                    1 => h2[b as usize] += 1,
                    2 => h3[b as usize] += 1,
                    _ => h3[b as usize] += 1, // A → slot3
                }
            }
        }
        (h1, h2, h3)
    }

    #[test]
    fn round250_rgb32_histogram_matches_per_byte_reference() {
        // Drive the residual pipeline across Left / LeftDecorr /
        // GradientDecorr at widths 1 / 2 / 4 / 8. For each frame, take
        // the combined_body the encoder feeds to histogramise and
        // compare the production histograms against the per-byte
        // reference.
        for &(w, h) in &[(1u32, 3u32), (2, 3), (4, 4), (8, 4)] {
            for &method in &[Method::Left, Method::LeftDecorr, Method::GradientDecorr] {
                let pixels = synth_rgb32(w as usize, h as usize);
                let frame = compute_frame_residuals(PixelFamily::Rgb32, method, w, h, &pixels)
                    .expect("residuals");
                let (h1_prod, h2_prod, h3_prod) =
                    histogramise(PixelFamily::Rgb32, method, &frame.body);
                let (h1_ref, h2_ref, h3_ref) = ref_histogramise_rgb32_per_byte(method, &frame.body);
                assert_eq!(
                    h1_prod, h1_ref,
                    "round250 slot1 histogram drift @ {}x{} {:?}",
                    w, h, method
                );
                assert_eq!(
                    h2_prod, h2_ref,
                    "round250 slot2 histogram drift @ {}x{} {:?}",
                    w, h, method
                );
                assert_eq!(
                    h3_prod, h3_ref,
                    "round250 slot3 histogram drift @ {}x{} {:?}",
                    w, h, method
                );
                // Sanity: histograms must sum to body.len() (every body
                // byte is counted exactly once — the alpha-shares-
                // slot-3 mapping means slot 3 receives two positions
                // per pixel, but each body byte still contributes to
                // exactly one histogram).
                let total: u64 = h1_prod
                    .iter()
                    .chain(h2_prod.iter())
                    .chain(h3_prod.iter())
                    .map(|&c| c as u64)
                    .sum();
                assert_eq!(
                    total,
                    frame.body.len() as u64,
                    "round250 histogram total != body.len() @ {}x{} {:?}",
                    w,
                    h,
                    method
                );
            }
        }
    }

    #[test]
    fn round250_rgb32_histogram_synth_body_matches_per_byte_reference_no_decorr() {
        // Direct check against a synthetic body that exercises every
        // residue position. Uses a 128-byte body (= 32 wire pixels) so
        // the slot positions are densely covered with distinct values
        // — a slot mix-up inside the step body would surface as counts
        // attributed to the wrong histogram. No-decorr quadruple:
        // (slot1, slot2, slot3, slot3) at (+0, +1, +2, +3).
        let body: Vec<u8> = (0u8..128).collect();
        let (h1_prod, h2_prod, h3_prod) = histogramise(PixelFamily::Rgb32, Method::Left, &body);
        let (h1_ref, h2_ref, h3_ref) = ref_histogramise_rgb32_per_byte(Method::Left, &body);
        assert_eq!(h1_prod, h1_ref);
        assert_eq!(h2_prod, h2_ref);
        assert_eq!(h3_prod, h3_ref);

        // Fixed expectations: bytes at i = 0, 4, 8, …, 124 fall on +0
        // (slot1 under no-decorr); i = 1, 5, 9, …, 125 fall on +1
        // (slot2); i = 2, 6, 10, …, 126 fall on +2 (slot3); i = 3, 7,
        // 11, …, 127 fall on +3 (slot3 — alpha shares slot 3 per
        // §1.2). Each value appears exactly once at its assigned
        // position.
        for v in (0u8..128).step_by(4) {
            assert_eq!(
                h1_prod[v as usize], 1,
                "no-decorr slot1 should hold byte {v} exactly once",
            );
        }
        for v in (1u8..128).step_by(4) {
            assert_eq!(
                h2_prod[v as usize], 1,
                "no-decorr slot2 should hold byte {v} exactly once",
            );
        }
        // Slot 3 receives bytes from BOTH +2 (R) and +3 (A) — values
        // 2, 3, 6, 7, 10, 11, … each occur exactly once in `body` and
        // each one lands in slot 3.
        for v in (2u8..128).step_by(4) {
            assert_eq!(
                h3_prod[v as usize], 1,
                "no-decorr slot3 should hold pos-+2 byte {v} exactly once",
            );
        }
        for v in (3u8..128).step_by(4) {
            assert_eq!(
                h3_prod[v as usize], 1,
                "no-decorr slot3 should hold pos-+3 (alpha) byte {v} exactly once",
            );
        }
    }

    #[test]
    fn round250_rgb32_histogram_synth_body_matches_per_byte_reference_decorr() {
        // Same synthetic body, decorr quadruple: (slot2, slot1, slot3,
        // slot3) at (+0, +1, +2, +3). A slot-swap regression between
        // no-decorr and decorr would surface as the slot1/slot2
        // columns landing in the wrong histogram.
        let body: Vec<u8> = (0u8..128).collect();
        let (h1_prod, h2_prod, h3_prod) =
            histogramise(PixelFamily::Rgb32, Method::LeftDecorr, &body);
        let (h1_ref, h2_ref, h3_ref) = ref_histogramise_rgb32_per_byte(Method::LeftDecorr, &body);
        assert_eq!(h1_prod, h1_ref);
        assert_eq!(h2_prod, h2_ref);
        assert_eq!(h3_prod, h3_ref);

        // Fixed expectations for the decorr quadruple: pos +0 → slot2
        // (G), pos +1 → slot1 (B−G), pos +2 → slot3 (R−G), pos +3 →
        // slot3 (A).
        for v in (0u8..128).step_by(4) {
            assert_eq!(
                h2_prod[v as usize], 1,
                "decorr slot2 should hold pos-+0 byte {v} exactly once",
            );
        }
        for v in (1u8..128).step_by(4) {
            assert_eq!(
                h1_prod[v as usize], 1,
                "decorr slot1 should hold pos-+1 byte {v} exactly once",
            );
        }
        for v in (2u8..128).step_by(4) {
            assert_eq!(
                h3_prod[v as usize], 1,
                "decorr slot3 should hold pos-+2 byte {v} exactly once",
            );
        }
        for v in (3u8..128).step_by(4) {
            assert_eq!(
                h3_prod[v as usize], 1,
                "decorr slot3 should hold pos-+3 (alpha) byte {v} exactly once",
            );
        }
    }

    #[test]
    fn round250_rgb32_histogram_alpha_shares_slot3_counts_aggregate() {
        // Spec/03 §1.2's alpha-shares-slot-3 mapping means slot 3
        // accumulates counts from BOTH +2 (R / R−G) and +3 (A) within
        // a single pixel — not separate columns. Construct a body
        // where pos +2 and pos +3 hold the *same* value within each
        // pixel and verify slot 3's bucket for that value reflects two
        // increments per pixel.
        //
        // Body layout (8 wire pixels = 32 bytes):
        //   pixel n: [n*4 + 0] = n+1   (pos +0 → slot1)
        //            [n*4 + 1] = n+1   (pos +1 → slot2)
        //            [n*4 + 2] = 100   (pos +2 → slot3)
        //            [n*4 + 3] = 100   (pos +3 → slot3)
        //
        // Expectation: h3[100] == 16 (two per pixel × 8 pixels).
        // h1[n+1] == 1 for n ∈ 0..8, h2[n+1] == 1 for n ∈ 0..8.
        let mut body = Vec::with_capacity(32);
        for n in 0u8..8 {
            body.push(n + 1);
            body.push(n + 1);
            body.push(100);
            body.push(100);
        }
        let (h1_prod, h2_prod, h3_prod) = histogramise(PixelFamily::Rgb32, Method::Left, &body);
        let (h1_ref, h2_ref, h3_ref) = ref_histogramise_rgb32_per_byte(Method::Left, &body);
        assert_eq!(h1_prod, h1_ref);
        assert_eq!(h2_prod, h2_ref);
        assert_eq!(h3_prod, h3_ref);

        assert_eq!(
            h3_prod[100], 16,
            "slot 3 must accumulate from both +2 and +3 within a pixel"
        );
        for n in 0u8..8 {
            assert_eq!(h1_prod[(n + 1) as usize], 1);
            assert_eq!(h2_prod[(n + 1) as usize], 1);
        }
        let total: u64 = h1_prod
            .iter()
            .chain(h2_prod.iter())
            .chain(h3_prod.iter())
            .map(|&c| c as u64)
            .sum();
        assert_eq!(total, body.len() as u64);
    }

    /// Shared CustomV2 round-trip helper — mirrors the `rt_rgb32`
    /// pattern in `round245_rgb32_emit_pixel_step_tests` but pinned to
    /// `ExtradataMode::CustomV2` so the histogram → length-table →
    /// emit pipeline is the one being exercised.
    fn rt_rgb32_customv2(width: u32, height: u32, method: Method) {
        let pixels = synth_rgb32(width as usize, height as usize);
        let (bih, frame) = encode_frame_with_mode(
            PixelFamily::Rgb32,
            method,
            width,
            height,
            &pixels,
            ExtradataMode::CustomV2,
        )
        .expect("encode");
        let cfg = StreamConfig::parse_bitmapinfoheader(&bih).expect("parse");
        let decoded = crate::decoder::decode_frame(&cfg, &frame).expect("decode");
        assert_eq!(
            decoded.pixels, pixels,
            "round250 CustomV2 round-trip drift @ {}x{} {:?}",
            width, height, method
        );
    }

    #[test]
    fn round250_rgb32_histogram_customv2_roundtrips_left() {
        // The CustomV2 extradata path derives per-channel length tables
        // straight from `histogramise`, so any histogram drift would
        // change the emitted lengths and break the round-trip. Width
        // sweep 1 / 2 / 4 / 8 covers the pixel-step body's boundary
        // cases (= 0 and = 1 inner iteration per row plus multi-pixel
        // rows).
        for &(w, h) in &[(1u32, 3u32), (2, 3), (4, 4), (8, 4)] {
            rt_rgb32_customv2(w, h, Method::Left);
        }
    }

    #[test]
    fn round250_rgb32_histogram_customv2_roundtrips_decorr_methods() {
        for &(w, h) in &[(1u32, 3u32), (2, 3), (4, 4), (8, 4)] {
            for &method in &[Method::LeftDecorr, Method::GradientDecorr] {
                rt_rgb32_customv2(w, h, method);
            }
        }
    }
}

#[cfg(test)]
mod round262_residuals_row_bytes_accessor_tests {
    //! Round-262 regression guard. The three per-family residual
    //! builders (`yuy2_residuals` / `rgb24_residuals` /
    //! `rgb32_residuals`) each opened with their own open-coded
    //! `width × {2, 3, 4}` wire-stride; round 262 routes all three
    //! through the round-261 single source of truth
    //! (`PixelFamily::row_bytes`, spec/02 §3 wire-byte layout
    //! table) — the encoder-side companion to the decoder field
    //! migration in the same round. Wire-identical: same family →
    //! stride mapping, one origin.
    //!
    //! Coverage:
    //!
    //! - **Size-contract pin** — each builder accepts exactly a
    //!   `family.row_bytes(width) × height`-byte pixel buffer and
    //!   rejects ±1-byte buffers, so a stride drift would flip the
    //!   accept/reject boundary.
    //! - **Body-length pin** — the produced residual body length
    //!   follows the spec/03 §1.1 / §1.3 codes-per-pixel counts
    //!   expressed through `bytes_per_pixel_step`: YUY2 carries
    //!   `row_bytes × h − 4` body bytes after the 4-byte seed
    //!   macropixel; RGB24 / RGB32 carry
    //!   `(n_pixels − 1) × bytes_per_pixel_step` (3 / 4 codewords
    //!   per pixel after the uncompressed first pixel).

    use super::*;

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

    #[test]
    fn round262_residuals_size_contract_matches_family_row_bytes() {
        let cases: [(PixelFamily, Method, u32, u32); 3] = [
            (PixelFamily::Yuy2, Method::Left, 4, 3),
            (PixelFamily::Rgb24, Method::Left, 5, 2),
            (PixelFamily::Rgb32, Method::Left, 3, 4),
        ];
        for (family, method, w, h) in cases {
            let exact = family.row_bytes(w) * h as usize;
            let good = synth(exact, 0xBEEF_CAFE);
            let call = |buf: &[u8]| compute_frame_residuals(family, method, w, h, buf);
            assert!(
                call(&good).is_ok(),
                "{family:?}: row_bytes-sized buffer must be accepted"
            );
            assert!(
                call(&good[..exact - 1]).is_err(),
                "{family:?}: one-short buffer must be rejected"
            );
            let long = synth(exact + 1, 0xBEEF_CAFE);
            assert!(
                call(&long).is_err(),
                "{family:?}: one-long buffer must be rejected"
            );
        }
    }

    #[test]
    fn round262_residual_body_len_follows_bytes_per_pixel_step() {
        // YUY2 4×3: body = row_bytes × h − 4 seed bytes.
        let yw = 4u32;
        let yh = 3u32;
        let ypix = synth(PixelFamily::Yuy2.row_bytes(yw) * yh as usize, 0x1357_9BDF);
        let yres =
            compute_frame_residuals(PixelFamily::Yuy2, Method::Left, yw, yh, &ypix).expect("yuy2");
        assert_eq!(
            yres.body.len(),
            PixelFamily::Yuy2.row_bytes(yw) * yh as usize - 4
        );
        // RGB24 / RGB32: body = (n_pixels − 1) × step (spec/03 §1.1 /
        // §1.3 codes-per-pixel after the uncompressed first pixel).
        for (family, method) in [
            (PixelFamily::Rgb24, Method::Left),
            (PixelFamily::Rgb24, Method::LeftDecorr),
            (PixelFamily::Rgb32, Method::Left),
            (PixelFamily::Rgb32, Method::GradientDecorr),
        ] {
            let (w, h) = (5u32, 3u32);
            let pix = synth(family.row_bytes(w) * h as usize, 0x2468_ACE0);
            let res = compute_frame_residuals(family, method, w, h, &pix).expect("residuals");
            let n_pixels = (w * h) as usize;
            assert_eq!(
                res.body.len(),
                (n_pixels - 1) * family.bytes_per_pixel_step(),
                "{family:?}/{method:?}: body length must be (n_pixels − 1) × step"
            );
        }
    }
}

#[cfg(test)]
mod round382_arbitrary_extradata_conformance_tests {
    //! Round-382 decoder-contract conformance suite for spec/04 §3.4.
    //!
    //! §3.4 states the decoder MUST accept a v2.x extradata stream
    //! carrying **any** set of three RLE-compressed 256-entry length
    //! tables — "including but not limited to the six classic blobs" —
    //! and derive the per-symbol codes from those lengths by canonical
    //! Huffman construction (spec/03 §3), not by classic-blob lookup.
    //!
    //! The existing `ClassicV2` / `CustomV2` roundtrip tests always
    //! decode against tables the *encoder* chose; neither pins the
    //! decoder's obligation to accept a length table it would never
    //! itself emit. Here we hand-build a **deliberately non-classic**
    //! length table (a flat all-length-8 code over all 256 symbols —
    //! Kraft sum `256 × 2⁻⁸ = 1`, valid but pessimal, and byte-distinct
    //! from every one of the six classic blobs), RLE-compress it into
    //! the extradata, encode a real frame body against that exact table
    //! through `emit_bitstream_parts`, and prove the decoder
    //! reconstructs the source pixels bit-exactly. A companion assertion
    //! confirms the extradata bytes differ from all six classic blobs,
    //! so a classic-blob-locked decoder would fail this test.
    //!
    //! We drive both the progressive path and — for YUY2 — a tall
    //! (`> 288`) interlaced frame, exercising the arbitrary-table build
    //! through the field-split path as well.

    use super::*;
    use crate::decoder::decode_frame;
    use crate::tables::rle_decode_three_channels;

    /// A flat, valid, non-classic length table: every symbol gets an
    /// 8-bit code. Kraft sum = 256 × 2⁻⁸ = 1, so canonical build
    /// succeeds; no classic blob assigns a uniform length across all
    /// 256 slots, so this is guaranteed distinct.
    fn flat_len8() -> [u8; 256] {
        [8u8; 256]
    }

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

    /// Assert the hand-built extradata is byte-distinct from every one
    /// of the six classic blobs for the given family (a classic-blob
    /// oracle decoder could not have produced these tables).
    fn assert_not_classic(family: PixelFamily, extradata: &[u8]) {
        let methods: &[Method] = if family.is_rgb() {
            &[Method::Left, Method::LeftDecorr, Method::GradientDecorr]
        } else {
            &[Method::Left, Method::Gradient, Method::Median]
        };
        for &m in methods {
            let blob = classic_blob_bytes(family, m);
            assert_ne!(
                extradata, blob,
                "{family:?}/{m:?}: hand-built extradata must not equal a classic blob"
            );
        }
    }

    /// Encode `pixels` for `(family, method)` with the three
    /// hand-supplied length tables, then round-trip through the real
    /// BIH writer + parser + decoder. Proves spec/04 §3.4 for one case.
    fn roundtrip_with_tables(
        family: PixelFamily,
        method: Method,
        width: u32,
        height: u32,
        pixels: &[u8],
        lengths: &[[u8; 256]; 3],
    ) {
        // Build the three per-slot canonical Huffman tables from the
        // hand-supplied lengths (the same construction the decoder runs
        // on the extradata it receives).
        let s1 = HuffTable::build_from_lengths(&lengths[0]).expect("slot1");
        let s2 = HuffTable::build_from_lengths(&lengths[1]).expect("slot2");
        let s3 = HuffTable::build_from_lengths(&lengths[2]).expect("slot3");

        // RLE-compress the lengths into an extradata table region and
        // wrap it in a real BITMAPINFOHEADER via the public writer.
        let extradata = rle_encode_three_channels(lengths).expect("rle encode");
        assert_not_classic(family, &extradata);
        // Sanity: the extradata RLE decodes back to the exact lengths.
        let decoded_lens = rle_decode_three_channels(&extradata).expect("rle roundtrip");
        assert_eq!(&decoded_lens, lengths, "extradata must RLE-roundtrip");

        let strf = build_bitmapinfoheader(family, method, width, height, &extradata, true);

        // Encode the frame body against the hand-supplied tables. We
        // reuse the encoder's own residual computation so the body is a
        // genuine predicted-residual stream, then Huffman-pack it with
        // the flat tables via `emit_bitstream_parts`.
        let frame =
            compute_frame_residuals(family, method, width, height, pixels).expect("residuals");
        let frame_bytes = emit_bitstream_parts(
            family,
            method,
            &frame.seed,
            &frame.body,
            &s1.entries,
            &s2.entries,
            &s3.entries,
        )
        .expect("emit");

        // The decoder rebuilds its tables purely from the extradata
        // lengths — if it were classic-blob-locked, it would decode with
        // the wrong codes and produce garbage.
        let cfg = StreamConfig::parse_bitmapinfoheader(&strf).expect("parse strf");
        assert!(cfg.has_extradata, "cfg must carry the v2.x extradata");
        let out = decode_frame(&cfg, &frame_bytes).expect("decode");
        assert_eq!(
            out.pixels, pixels,
            "{family:?}/{method:?} {width}×{height}: arbitrary-table decode must be pixel-exact"
        );
    }

    #[test]
    fn round382_yuy2_left_flat_tables_progressive() {
        let (w, h) = (8u32, 4u32);
        let pixels = synth(PixelFamily::Yuy2.row_bytes(w) * h as usize, 0x1111_2222);
        let lens = [flat_len8(), flat_len8(), flat_len8()];
        roundtrip_with_tables(PixelFamily::Yuy2, Method::Left, w, h, &pixels, &lens);
    }

    #[test]
    fn round382_yuy2_gradient_flat_tables_progressive() {
        let (w, h) = (8u32, 5u32);
        let pixels = synth(PixelFamily::Yuy2.row_bytes(w) * h as usize, 0x3333_4444);
        let lens = [flat_len8(), flat_len8(), flat_len8()];
        roundtrip_with_tables(PixelFamily::Yuy2, Method::Gradient, w, h, &pixels, &lens);
    }

    #[test]
    fn round382_yuy2_median_flat_tables_progressive() {
        let (w, h) = (8u32, 6u32);
        let pixels = synth(PixelFamily::Yuy2.row_bytes(w) * h as usize, 0x5555_6666);
        let lens = [flat_len8(), flat_len8(), flat_len8()];
        roundtrip_with_tables(PixelFamily::Yuy2, Method::Median, w, h, &pixels, &lens);
    }

    #[test]
    fn round382_rgb24_left_flat_tables_progressive() {
        let (w, h) = (5u32, 3u32);
        let pixels = synth(PixelFamily::Rgb24.row_bytes(w) * h as usize, 0x7777_8888);
        let lens = [flat_len8(), flat_len8(), flat_len8()];
        roundtrip_with_tables(PixelFamily::Rgb24, Method::Left, w, h, &pixels, &lens);
    }

    #[test]
    fn round382_rgb24_left_decorr_flat_tables_progressive() {
        let (w, h) = (6u32, 4u32);
        let pixels = synth(PixelFamily::Rgb24.row_bytes(w) * h as usize, 0x9999_AAAA);
        let lens = [flat_len8(), flat_len8(), flat_len8()];
        roundtrip_with_tables(PixelFamily::Rgb24, Method::LeftDecorr, w, h, &pixels, &lens);
    }

    #[test]
    fn round382_rgb32_left_decorr_flat_tables_progressive() {
        let (w, h) = (4u32, 4u32);
        let pixels = synth(PixelFamily::Rgb32.row_bytes(w) * h as usize, 0xBBBB_CCCC);
        let lens = [flat_len8(), flat_len8(), flat_len8()];
        roundtrip_with_tables(PixelFamily::Rgb32, Method::LeftDecorr, w, h, &pixels, &lens);
    }

    #[test]
    fn round382_rgb32_gradient_decorr_flat_tables_progressive() {
        let (w, h) = (5u32, 5u32);
        let pixels = synth(PixelFamily::Rgb32.row_bytes(w) * h as usize, 0xDDDD_EEEE);
        let lens = [flat_len8(), flat_len8(), flat_len8()];
        roundtrip_with_tables(
            PixelFamily::Rgb32,
            Method::GradientDecorr,
            w,
            h,
            &pixels,
            &lens,
        );
    }

    #[test]
    fn round382_yuy2_left_flat_tables_interlaced_tall() {
        // biHeight > 288 engages the field-split path; the arbitrary
        // tables must serve both fields (spec/02 §2).
        let (w, h) = (8u32, 300u32);
        let pixels = synth(PixelFamily::Yuy2.row_bytes(w) * h as usize, 0x0F0F_1E1E);
        let lens = [flat_len8(), flat_len8(), flat_len8()];
        roundtrip_with_tables(PixelFamily::Yuy2, Method::Left, w, h, &pixels, &lens);
    }
}
