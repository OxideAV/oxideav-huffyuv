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
use crate::header::{Method, PixelFamily, Predictor, StreamConfig, FOURCC_HFYU};
use crate::predict::{
    forward_decorr_gradient_subtract, forward_gradient_subtract, forward_left_decorr_residuals,
    forward_median_subtract, forward_rgb_left_subtract_linear, forward_yuy2_left_subtract,
    is_interlaced_height,
};
use crate::tables::{
    classic_blob_bytes, compute_canonical_lengths, rle_decode_one_channel,
    rle_decode_three_channels, rle_encode_three_channels, v1x_codes_set_a, v1x_codes_set_b,
    v1x_lengths_set_a, v1x_lengths_set_b, v1x_table_from_pair, HuffEntry, HuffTable,
};

/// Selects which BIH/extradata path the encoder writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
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
    let frame = compute_frame_residuals(family, method, width, height, pixels)?;
    encode_with_precomputed(family, method, width, height, &frame, mode)
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
    frame: &PrecomputedFrame,
    mode: ExtradataMode,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let stats_body: &[u8] = &frame.combined_body;

    let (extradata_tables, slot1, slot2, slot3, has_extradata): (
        Vec<u8>,
        HuffTable,
        HuffTable,
        HuffTable,
        bool,
    ) = match mode {
        ExtradataMode::ClassicV2 => {
            let extra = classic_blob_bytes(family, method).to_vec();
            let lengths = rle_decode_three_channels(&extra)?;
            let s1 = HuffTable::build_from_lengths(&lengths[0])?;
            let s2 = HuffTable::build_from_lengths(&lengths[1])?;
            let s3 = HuffTable::build_from_lengths(&lengths[2])?;
            (extra, s1, s2, s3, true)
        }
        ExtradataMode::CustomV2 => {
            let lengths = compute_lengths_from_body(family, method, stats_body);
            let s1 = HuffTable::build_from_lengths(&lengths[0])?;
            let s2 = HuffTable::build_from_lengths(&lengths[1])?;
            let s3 = HuffTable::build_from_lengths(&lengths[2])?;
            let extra = rle_encode_three_channels(&lengths)?;
            (extra, s1, s2, s3, true)
        }
        ExtradataMode::V1xCompat => {
            let (s1, s2, s3) = build_v1x_tables(family)?;
            // Verify residuals only use symbols that have non-zero
            // length in the v1.x codebooks — otherwise the wire would
            // be undecodable. (The classic v1.x sets cover all 256
            // symbols, so this is a belt-and-braces check.)
            verify_body_in_table(family, method, stats_body, &s1, &s2, &s3)?;
            (Vec::new(), s1, s2, s3, false)
        }
    };

    let strf = build_bitmapinfoheader(
        family,
        method,
        width,
        height,
        &extradata_tables,
        has_extradata,
    );

    let frame_bytes = if frame.interlaced {
        let mut out = Vec::new();
        // Top field: pass the seed + the [..top_body_len] slice of
        // the combined body directly to `emit_bitstream_parts`. No
        // per-field body Vec allocation needed.
        let mut top_bytes = emit_bitstream_parts(
            family,
            method,
            &frame.top_seed,
            &frame.combined_body[..frame.top_body_len],
            &slot1,
            &slot2,
            &slot3,
        )?;
        out.append(&mut top_bytes);
        if let Some(bot_seed) = frame.bot_seed_opt {
            let mut bot_bytes = emit_bitstream_parts(
                family,
                method,
                &bot_seed,
                &frame.combined_body[frame.top_body_len..],
                &slot1,
                &slot2,
                &slot3,
            )?;
            out.append(&mut bot_bytes);
        }
        out
    } else {
        emit_bitstream_parts(
            family,
            method,
            &frame.top_seed,
            &frame.combined_body,
            &slot1,
            &slot2,
            &slot3,
        )?
    };

    let _ = (width, height);
    Ok((strf, frame_bytes))
}

/// Round-7 residual carrier: holds the per-field seed(s) + a single
/// combined body Vec covering one progressive frame or both interlaced
/// fields concatenated. Replaces the previous round's mix of
/// `Option<Residuals>` (progressive) + `(top_seed, bot_seed_opt,
/// top_body_len, combined_body)` (interlaced) so the same struct can
/// be reused across the auto-selector's candidate-scoring + final
/// emit passes.
#[derive(Debug, Clone)]
struct PrecomputedFrame {
    interlaced: bool,
    top_seed: [u8; 4],
    /// `None` for progressive frames OR for interlaced frames where the
    /// bottom field is empty (height == 1, etc.).
    bot_seed_opt: Option<[u8; 4]>,
    /// Length of the top field's body within `combined_body`. Equals
    /// `combined_body.len()` for progressive frames.
    top_body_len: usize,
    /// Combined body bytes. Progressive: a single field's body.
    /// Interlaced: top.body || bot.body (the bot half resumes the
    /// per-family slot phase because top.body_len is always a multiple
    /// of the slot cycle — see round-5 notes in this module).
    combined_body: Vec<u8>,
}

/// Compute the per-frame residual stream once. Shared by
/// [`encode_frame_with_mode`], [`encode_frame_auto`] (round-7), and
/// [`bit_cost_for_method`].
///
/// Spec/02 §2 + spec/05 (planned): when biHeight > 288 the codec
/// splits the source into two fields (even rows = top; odd rows =
/// bottom) and predicts each independently. The walking-stride path
/// from round 5 is preserved: a single field-sized scratch buffer is
/// reused across top + bot field, and both fields' residual bodies
/// land in ONE combined `Vec<u8>` (the bot half starts at
/// `top_body_len`).
fn compute_frame_residuals(
    family: PixelFamily,
    method: Method,
    width: u32,
    height: u32,
    pixels: &[u8],
) -> Result<PrecomputedFrame> {
    let interlaced = is_interlaced_height(height);
    if !interlaced {
        let r = compute_residuals(family, method, width, height, pixels)?;
        let top_body_len = r.body.len();
        return Ok(PrecomputedFrame {
            interlaced: false,
            top_seed: r.seed,
            bot_seed_opt: None,
            top_body_len,
            combined_body: r.body,
        });
    }
    let row_bytes = match family {
        PixelFamily::Yuy2 => width as usize * 2,
        PixelFamily::Rgb24 => width as usize * 3,
        PixelFamily::Rgb32 => width as usize * 4,
    };
    let h = height as usize;
    let top_h = h.div_ceil(2) as u32;
    let bot_h = (h / 2) as u32;
    // One field-sized scratch reused across top + bot (round-5
    // walking-stride invariant — preserves the ~1.5× memory
    // reduction over round 4's split_fields approach).
    let mut scratch = vec![0u8; (top_h as usize) * row_bytes];
    let body_capacity = match family {
        PixelFamily::Yuy2 => row_bytes * h - if h > 0 { 4 } else { 0 },
        PixelFamily::Rgb24 => {
            let n_top = (width as usize) * (top_h as usize);
            let n_bot = (width as usize) * (bot_h as usize);
            let mut c = 0usize;
            if n_top > 0 {
                c += (n_top - 1) * 3;
            }
            if n_bot > 0 {
                c += (n_bot - 1) * 3;
            }
            c
        }
        PixelFamily::Rgb32 => {
            let n_top = (width as usize) * (top_h as usize);
            let n_bot = (width as usize) * (bot_h as usize);
            let mut c = 0usize;
            if n_top > 0 {
                c += (n_top - 1) * 4;
            }
            if n_bot > 0 {
                c += (n_bot - 1) * 4;
            }
            c
        }
    };
    let mut combined: Vec<u8> = Vec::with_capacity(body_capacity);
    compact_field_rows(pixels, &mut scratch, row_bytes, h, 0);
    let top_res = compute_residuals(
        family,
        method,
        width,
        top_h,
        &scratch[..(top_h as usize) * row_bytes],
    )?;
    let top_seed = top_res.seed;
    let top_len = top_res.body.len();
    combined.extend_from_slice(&top_res.body);
    drop(top_res);
    let bot_seed_opt = if bot_h > 0 {
        scratch.resize((bot_h as usize) * row_bytes, 0);
        compact_field_rows(pixels, &mut scratch, row_bytes, h, 1);
        let bot_res = compute_residuals(
            family,
            method,
            width,
            bot_h,
            &scratch[..(bot_h as usize) * row_bytes],
        )?;
        let bs = bot_res.seed;
        combined.extend_from_slice(&bot_res.body);
        Some(bs)
    } else {
        None
    };
    Ok(PrecomputedFrame {
        interlaced: true,
        top_seed,
        bot_seed_opt,
        top_body_len: top_len,
        combined_body: combined,
    })
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
    v[0x2A] = 0;
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
/// Returns the total number of code-bits (excluding the per-field
/// 4-byte uncompressed seed) the body would occupy using the
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
    let (h1, h2, h3) = histogramise(family, method, &frame.combined_body);
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
    let candidates = selection.candidates(family);
    if candidates.is_empty() {
        return Err(Error::invalid("encode_frame_auto: no legal methods"));
    }
    // Round-7: compute residuals ONCE per candidate (was: once per
    // candidate inside `bit_cost_for_method` PLUS once again inside
    // `encode_frame_with_mode` for the winner = N+1 traversals).
    // The pre-computed residuals carry forward into the final
    // `encode_with_precomputed` so the winner's body bytes don't get
    // re-derived from `pixels`. Wire-identical to round 6.
    //
    // Tie-break: first in `candidates` order (so output is
    // deterministic).
    let mut best: Option<(Method, u64, PrecomputedFrame)> = None;
    for &m in &candidates {
        let frame = compute_frame_residuals(family, m, width, height, pixels)?;
        let (h1, h2, h3) = histogramise(family, m, &frame.combined_body);
        let cost = bit_cost_from_histograms(&h1, &h2, &h3);
        match best {
            None => best = Some((m, cost, frame)),
            Some((_, prev_cost, _)) if cost < prev_cost => best = Some((m, cost, frame)),
            _ => {}
        }
    }
    let (chosen, _, frame) = best.expect("non-empty candidates → some winner");
    let (strf, bytes) = encode_with_precomputed(family, chosen, width, height, &frame, mode)?;
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

fn compute_residuals(
    family: PixelFamily,
    method: Method,
    width: u32,
    height: u32,
    pixels: &[u8],
) -> Result<Residuals> {
    match family {
        PixelFamily::Yuy2 => yuy2_residuals(method, width, height, pixels),
        PixelFamily::Rgb24 => rgb24_residuals(method, width, height, pixels),
        PixelFamily::Rgb32 => rgb32_residuals(method, width, height, pixels),
    }
}

/// Walking-stride field row gather. Round-5 walks the source raster
/// with row-stride 2 (writing the picked rows into `dst`'s start)
/// instead of allocating a new top-field / bottom-field buffer via
/// `predict::split_fields`. `field_idx == 0` selects even rows
/// (top field); `field_idx == 1` selects odd rows (bottom field).
///
/// `dst` must already be sized to at least `field_h * row_bytes`
/// where `field_h = (full_h + 1 - field_idx) / 2`. The function does
/// not resize.
///
/// Spec/02 §2 says nothing about HOW the encoder happens to extract
/// fields — only that the on-wire chunk concatenates the two
/// per-field bit-streams. The walking-stride read pattern matches
/// the round-4 `split_fields(...)` output byte-for-byte; a
/// regression test in `roundtrip_tests::walking_stride_matches_split_fields_path`
/// guards against drift.
fn compact_field_rows(
    pixels: &[u8],
    dst: &mut [u8],
    row_bytes: usize,
    full_h: usize,
    field_idx: usize,
) {
    debug_assert!(field_idx < 2);
    let mut logical = 0usize;
    let mut row = field_idx;
    while row < full_h {
        let src = row * row_bytes;
        let d = logical * row_bytes;
        dst[d..d + row_bytes].copy_from_slice(&pixels[src..src + row_bytes]);
        row += 2;
        logical += 1;
    }
}

fn yuy2_residuals(method: Method, width: u32, height: u32, pixels: &[u8]) -> Result<Residuals> {
    let w = width as usize;
    let h = height as usize;
    let row_bytes = w * 2;
    if pixels.len() != row_bytes * h {
        return Err(Error::invalid(format!(
            "encoder: YUY2 pixel-buffer size {} ≠ expected {}",
            pixels.len(),
            row_bytes * h
        )));
    }
    if w % 2 != 0 {
        return Err(Error::invalid("encoder: YUY2 width must be even"));
    }
    let mut residuals = vec![0u8; row_bytes * h];
    if method.predictor() == Predictor::Median {
        // Round 115: the YUY2 forward MEDIAN pre-pass is produced in a
        // single pass by `forward_median_subtract` (spec/03 §2.3 /
        // §2.3.2) — LEFT residuals for row 0 + the first 8 wire bytes
        // of row 1, MEDIAN residuals for the rest. Previously this
        // computed a full-frame LEFT residual stream and then
        // overwrote the median region (recomputing the median region's
        // LEFT residuals only to discard them).
        forward_median_subtract(pixels, &mut residuals, row_bytes, h);
    } else {
        // Round 95: avoid the `pixels.to_vec()` clone when no gradient
        // pre-pass is needed by reading directly from `pixels` for the
        // per-channel-stride subtract. For Gradient we still need the
        // intermediate Vec (the per-channel subtract chain reads earlier
        // intermediate bytes; an in-place rewrite would corrupt later
        // reads). The intermediate is now built by `forward_gradient_subtract`
        // (spec/03 §2.2.2 `@0x10001eab..@0x10001f9e`).
        let intermediate: Option<Vec<u8>> = if method.predictor() == Predictor::Gradient {
            let mut iv = vec![0u8; row_bytes * h];
            forward_gradient_subtract(pixels, &mut iv, row_bytes, h);
            Some(iv)
        } else {
            None
        };
        let pred_input: &[u8] = intermediate.as_deref().unwrap_or(pixels);
        // Round 181: route the YUY2 forward LEFT residual through
        // the branch-free macropixel-step helper (spec/03 §2.1.1
        // four-channel Y₁ / U / Y₂ / V body) instead of the per-byte
        // `i & 1` stride-select. Bit-identical output; see
        // `predict::forward_yuy2_left_subtract` for the spec
        // citation + equivalence regression test.
        forward_yuy2_left_subtract(pred_input, &mut residuals);
    }
    let mut seed = [0u8; 4];
    let copy_n = 4.min(pixels.len());
    seed[..copy_n].copy_from_slice(&pixels[..copy_n]);
    let body = residuals[copy_n..].to_vec();
    Ok(Residuals { seed, body })
}

fn rgb24_residuals(method: Method, width: u32, height: u32, pixels: &[u8]) -> Result<Residuals> {
    let w = width as usize;
    let h = height as usize;
    let row_bytes = w * 3;
    if pixels.len() != row_bytes * h {
        return Err(Error::invalid(format!(
            "encoder: RGB24 pixel-buffer size {} ≠ expected {}",
            pixels.len(),
            row_bytes * h
        )));
    }
    let n_pixels = w * h;
    // Round 100: for the LeftDecorr path (decorrelate, no gradient),
    // fuse the decorrelation transform into the LEFT residual per
    // spec/03 §2.4.1 ("the decorrelation transform is fused with the
    // predictor at the residual computation, not applied as a separate
    // pre-pass ... there is no intermediate decorrelated buffer"). This
    // skips the round-95 `working_owned: Vec<u8>` full-frame allocation
    // (= row_bytes × h bytes per frame) entirely.
    let fused_left_decorr = method.decorrelate() && method.predictor() == Predictor::Left;
    // Round 103: for the GradientDecorr path (decorrelate + gradient),
    // fuse the decorrelation into the forward-gradient subtract per
    // spec/03 §2.4.2 (gradient applied to each decorrelated channel;
    // the §2.2.2 gradient pre-pass reads only decorrelated values).
    // `forward_decorr_gradient_subtract` produces the gradient
    // pre-pass output straight from un-transformed `pixels`, so the
    // round-95 `working_owned` decorrelated buffer is no longer needed
    // on this path either. The per-channel LEFT-subtract pass that
    // follows is identical; only the seed sourcing differs (the
    // decorrelated first pixel must be read from `pixels`, not the
    // now-absent `working`).
    let fused_decorr_gradient = method.decorrelate() && method.predictor() == Predictor::Gradient;
    // Round 95: skip the `pixels.to_vec()` clone when no decorrelation
    // is in play — `working` borrows pixels directly. With both
    // decorrelation paths fused (round 100 LeftDecorr + round 103
    // GradientDecorr) `working_owned` is never allocated anymore.
    let working_owned: Option<Vec<u8>> =
        if method.decorrelate() && !fused_left_decorr && !fused_decorr_gradient {
            let mut v = vec![0u8; row_bytes * h];
            for px in 0..n_pixels {
                let off = px * 3;
                let b = pixels[off];
                let g = pixels[off + 1];
                let r = pixels[off + 2];
                v[off] = b.wrapping_sub(g);
                v[off + 1] = g;
                v[off + 2] = r.wrapping_sub(g);
            }
            Some(v)
        } else {
            None
        };
    let working: &[u8] = working_owned.as_deref().unwrap_or(pixels);
    let mut residuals = vec![0u8; row_bytes * h];
    if fused_left_decorr {
        // Fused decorrelated-LEFT residuals straight from `pixels`.
        forward_left_decorr_residuals(pixels, &mut residuals, 3);
    } else {
        // The gradient pre-pass output. For GradientDecorr the fused
        // helper folds decorrelation in (round 103); for plain
        // Gradient it's the same-column row subtract over `working`;
        // for non-gradient methods `pred_input` borrows `working`
        // directly (round 95 — no `working.clone()`).
        let intermediate: Option<Vec<u8>> = if fused_decorr_gradient {
            let mut iv = vec![0u8; row_bytes * h];
            forward_decorr_gradient_subtract(pixels, &mut iv, 3, row_bytes, h);
            Some(iv)
        } else if method.predictor() == Predictor::Gradient {
            let mut iv = vec![0u8; row_bytes * h];
            forward_gradient_subtract(working, &mut iv, row_bytes, h);
            Some(iv)
        } else {
            None
        };
        let pred_input: &[u8] = intermediate.as_deref().unwrap_or(working);
        // Round 186: single linear stride-1 walk replaces the prior
        // three-pass stride-3 channel loop (3× fewer cache traversals
        // + a contiguous SIMD-friendly inner subtract). spec/03 §2.1
        // encoder evidence `@0x10001850`. Bit-identical to the prior
        // per-channel loop — regression guarded by
        // `predict::tests::round186_rgb_left_linear_matches_per_channel_rgb24`.
        forward_rgb_left_subtract_linear(pred_input, &mut residuals, 3);
    }
    // Wire seed: `00 B G R` (the decoder writes pad as 0; first
    // pixel goes into bytes 1..4). For the fused LeftDecorr path the
    // decorrelated seed is already in `residuals[0..3]`; for the fused
    // GradientDecorr path it's the decorrelated first pixel computed
    // from `pixels` (B−G, G, R−G); otherwise it's `working`'s first
    // (decorrelated-or-plain) pixel.
    let mut seed = [0u8; 4];
    if fused_left_decorr {
        seed[1] = residuals[0];
        seed[2] = residuals[1];
        seed[3] = residuals[2];
    } else if fused_decorr_gradient {
        let g0 = pixels[1];
        seed[1] = pixels[0].wrapping_sub(g0); // B − G
        seed[2] = g0; // G
        seed[3] = pixels[2].wrapping_sub(g0); // R − G
    } else {
        seed[1] = working[0];
        seed[2] = working[1];
        seed[3] = working[2];
    }
    // Body covers per-pixel residuals from pixel 1 onward, with order
    // determined by `decorrelate`. We build the body in wire-order
    // here so the emit pass can be a flat slot-iteration.
    let mut body = Vec::with_capacity((n_pixels - 1) * 3);
    for px in 1..n_pixels {
        let off = px * 3;
        if method.decorrelate() {
            // wire order: G (slot2), B-G (slot1), R-G (slot3).
            body.push(residuals[off + 1]);
            body.push(residuals[off]);
            body.push(residuals[off + 2]);
        } else {
            // wire order: B (slot1), G (slot2), R (slot3).
            body.push(residuals[off]);
            body.push(residuals[off + 1]);
            body.push(residuals[off + 2]);
        }
    }
    Ok(Residuals { seed, body })
}

fn rgb32_residuals(method: Method, width: u32, height: u32, pixels: &[u8]) -> Result<Residuals> {
    let w = width as usize;
    let h = height as usize;
    let row_bytes = w * 4;
    if pixels.len() != row_bytes * h {
        return Err(Error::invalid(format!(
            "encoder: RGB32 pixel-buffer size {} ≠ expected {}",
            pixels.len(),
            row_bytes * h
        )));
    }
    let n_pixels = w * h;
    // Round 100: fuse decorrelation into the LEFT residual for the
    // LeftDecorr path (spec/03 §2.4.1 — no intermediate decorrelated
    // buffer), skipping the round-95 `working_owned` full-frame
    // allocation. Alpha is NOT decorrelated (spec/03 §2.4 Validator
    // note); the fused helper LEFT-predicts it like the colour
    // channels.
    let fused_left_decorr = method.decorrelate() && method.predictor() == Predictor::Left;
    // Round 103: fuse decorrelation into the forward-gradient subtract
    // for the GradientDecorr path (spec/03 §2.4.2 — gradient applied to
    // each decorrelated channel). `forward_decorr_gradient_subtract`
    // folds the per-pixel decorrelation (B−G / G / R−G; alpha identity)
    // into the §2.2.2 same-column gradient subtract, reading
    // un-transformed `pixels` directly, so `working_owned` is no longer
    // allocated on this path. Alpha is identity in both the decorr and
    // gradient steps (NOT decorrelated, spec/03 §2.4 Validator note).
    let fused_decorr_gradient = method.decorrelate() && method.predictor() == Predictor::Gradient;
    // Round 95: skip the `pixels.to_vec()` clone when no decorrelation
    // is in play. With both decorr paths fused (round 100 + round 103)
    // `working_owned` is never allocated anymore.
    let working_owned: Option<Vec<u8>> =
        if method.decorrelate() && !fused_left_decorr && !fused_decorr_gradient {
            let mut v = vec![0u8; row_bytes * h];
            for px in 0..n_pixels {
                let off = px * 4;
                let b = pixels[off];
                let g = pixels[off + 1];
                let r = pixels[off + 2];
                let a = pixels[off + 3];
                v[off] = b.wrapping_sub(g);
                v[off + 1] = g;
                v[off + 2] = r.wrapping_sub(g);
                v[off + 3] = a; // alpha NOT decorrelated.
            }
            Some(v)
        } else {
            None
        };
    let working: &[u8] = working_owned.as_deref().unwrap_or(pixels);
    let mut residuals = vec![0u8; row_bytes * h];
    let seed = if fused_left_decorr {
        forward_left_decorr_residuals(pixels, &mut residuals, 4);
        // Fused path: the decorrelated seed pixel is in residuals[0..4]
        // (B−G, G, R−G, A).
        let mut s = [0u8; 4];
        s.copy_from_slice(&residuals[..4]);
        s
    } else {
        // The gradient pre-pass output. For GradientDecorr the fused
        // helper folds decorrelation in (round 103); for plain Gradient
        // it's the same-column row subtract over `working`; for
        // non-gradient methods `pred_input` borrows `working` directly
        // (round 95 — no `working.clone()`).
        let intermediate: Option<Vec<u8>> = if fused_decorr_gradient {
            let mut iv = vec![0u8; row_bytes * h];
            forward_decorr_gradient_subtract(pixels, &mut iv, 4, row_bytes, h);
            Some(iv)
        } else if method.predictor() == Predictor::Gradient {
            let mut iv = vec![0u8; row_bytes * h];
            forward_gradient_subtract(working, &mut iv, row_bytes, h);
            Some(iv)
        } else {
            None
        };
        let pred_input: &[u8] = intermediate.as_deref().unwrap_or(working);
        // Round 186: single linear stride-1 walk replaces the prior
        // four-pass stride-4 channel loop (4× fewer cache traversals
        // + a contiguous SIMD-friendly inner subtract). spec/03 §2.1
        // encoder evidence `@0x10001b21..@0x10001b3c` (the RGB32-LEFT
        // byte-3 / A emit reusing the same offset-N-back identity as
        // the byte-0 / B emit). Bit-identical to the prior
        // per-channel loop — regression guarded by
        // `predict::tests::round186_rgb_left_linear_matches_per_channel_rgb32`.
        forward_rgb_left_subtract_linear(pred_input, &mut residuals, 4);
        // Seed = decorrelated-or-plain first pixel. For the fused
        // GradientDecorr path compute it from `pixels` (B−G, G, R−G, A
        // — alpha identity); otherwise read `working`'s first pixel.
        if fused_decorr_gradient {
            let g0 = pixels[1];
            [
                pixels[0].wrapping_sub(g0), // B − G
                g0,                         // G
                pixels[2].wrapping_sub(g0), // R − G
                pixels[3],                  // A (identity)
            ]
        } else {
            let mut s = [0u8; 4];
            s.copy_from_slice(&working[..4]);
            s
        }
    };
    let mut body = Vec::with_capacity((n_pixels - 1) * 4);
    for px in 1..n_pixels {
        let off = px * 4;
        if method.decorrelate() {
            // wire order: G (slot2), B-G (slot1), R-G (slot3), A (slot3).
            body.push(residuals[off + 1]);
            body.push(residuals[off]);
            body.push(residuals[off + 2]);
            body.push(residuals[off + 3]);
        } else {
            // wire order: B (slot1), G (slot2), R (slot3), A (slot3).
            body.push(residuals[off]);
            body.push(residuals[off + 1]);
            body.push(residuals[off + 2]);
            body.push(residuals[off + 3]);
        }
    }
    Ok(Residuals { seed, body })
}

// ───────────────────────── histogram + length building ─────────────────────────

/// Walks `body` directly to build the per-channel histograms, then
/// runs the package-merge length-limited Huffman builder. Round 5
/// drops the round-4 `&Residuals` wrapper here so the combined-body
/// Vec from the interlaced walking-stride path can drive histogram
/// building without an intermediate per-frame `Residuals` clone.
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
            // body is laid out 4 bytes per pixel in wire order.
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
fn build_v1x_tables(family: PixelFamily) -> Result<(HuffTable, HuffTable, HuffTable)> {
    use std::sync::OnceLock;
    static YUY2_TABLES: OnceLock<Result<(HuffTable, HuffTable, HuffTable)>> = OnceLock::new();
    static RGB_TABLES: OnceLock<Result<(HuffTable, HuffTable, HuffTable)>> = OnceLock::new();
    let cell = match family {
        PixelFamily::Yuy2 => &YUY2_TABLES,
        PixelFamily::Rgb24 | PixelFamily::Rgb32 => &RGB_TABLES,
    };
    let cached = cell.get_or_init(|| build_v1x_tables_uncached(family));
    match cached {
        Ok((s1, s2, s3)) => Ok((s1.clone(), s2.clone(), s3.clone())),
        Err(e) => Err(Error::invalid(format!("v1.x cache build failed: {e}"))),
    }
}

fn build_v1x_tables_uncached(family: PixelFamily) -> Result<(HuffTable, HuffTable, HuffTable)> {
    let mut cur: &[u8] = v1x_lengths_set_a();
    let lens_a = rle_decode_one_channel(&mut cur)?;
    let mut cur: &[u8] = v1x_lengths_set_b();
    let lens_b = rle_decode_one_channel(&mut cur)?;
    let codes_a_buf = v1x_codes_set_a();
    let codes_b_buf = v1x_codes_set_b();
    let mut codes_a = [0u8; 256];
    codes_a.copy_from_slice(codes_a_buf);
    let mut codes_b = [0u8; 256];
    codes_b.copy_from_slice(codes_b_buf);
    let table_a = v1x_table_from_pair(&lens_a, &codes_a)?;
    let table_b = v1x_table_from_pair(&lens_b, &codes_b)?;
    match family {
        PixelFamily::Yuy2 => Ok((table_a, table_b.clone(), table_b)),
        PixelFamily::Rgb24 | PixelFamily::Rgb32 => Ok((table_a.clone(), table_a.clone(), table_a)),
    }
}

/// V1xCompat verification: walk the body in slot order and ensure
/// each emitted symbol has a non-zero length in its slot's table.
/// Round 5 takes `body: &[u8]` directly so the interlaced
/// walking-stride path's combined body can be verified without an
/// intermediate `Residuals` clone.
fn verify_body_in_table(
    family: PixelFamily,
    method: Method,
    body: &[u8],
    s1: &HuffTable,
    s2: &HuffTable,
    s3: &HuffTable,
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
            if s1.entries[sym0 as usize].length == 0 {
                return Err(Error::unsupported(format!(
                    "v1.x compat: residual symbol 0x{sym0:02x} not in v1.x codebook"
                )));
            }
            let sym1 = body[i + 1];
            if s2.entries[sym1 as usize].length == 0 {
                return Err(Error::unsupported(format!(
                    "v1.x compat: residual symbol 0x{sym1:02x} not in v1.x codebook"
                )));
            }
            let sym2 = body[i + 2];
            if s1.entries[sym2 as usize].length == 0 {
                return Err(Error::unsupported(format!(
                    "v1.x compat: residual symbol 0x{sym2:02x} not in v1.x codebook"
                )));
            }
            let sym3 = body[i + 3];
            if s3.entries[sym3 as usize].length == 0 {
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
            if slot.entries[sym as usize].length == 0 {
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
        if slot.entries[sym as usize].length == 0 {
            return Err(Error::unsupported(format!(
                "v1.x compat: residual symbol 0x{sym:02x} not in v1.x codebook"
            )));
        }
    }
    Ok(())
}

// ───────────────────────── bit-emit pass ─────────────────────────

/// Round-5 walking-stride entry: emit one field's seed + body
/// without copying body bytes into a per-field `Residuals`. The
/// interlaced encoder calls this twice on the combined-body Vec
/// (once with body=[..top_body_len], once with body=[top_body_len..])
/// to avoid the round-4 per-field body allocations at emit time.
///
/// Round-7 removed the `emit_bitstream` / `EmitParams` wrapper that
/// previously fronted this routine — all callers now compute the
/// seed/body parts via [`PrecomputedFrame`] and hand them through
/// directly.
#[allow(clippy::too_many_arguments)]
fn emit_bitstream_parts(
    family: PixelFamily,
    method: Method,
    seed: &[u8; 4],
    body: &[u8],
    slot1: &HuffTable,
    slot2: &HuffTable,
    slot3: &HuffTable,
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
                let (code, length) = lookup_code(slot, sym)?;
                writer.write_msb(code, length);
            }
        }
    }
    out.extend_from_slice(&writer.finish());
    Ok(out)
}

fn lookup_code(table: &HuffTable, sym: u8) -> Result<(u32, u32)> {
    let entry: HuffEntry = table.entries[sym as usize];
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
        slot1: &HuffTable,
        slot2: &HuffTable,
        slot3: &HuffTable,
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
            let lengths =
                compute_lengths_from_body(PixelFamily::Yuy2, method, &frame.combined_body);
            // Build the slot HuffTables from the computed lengths.
            // Matches the v2.x Custom path in encode_with_precomputed.
            let slot1 = HuffTable::build_from_lengths(&lengths[0]).expect("slot1");
            let slot2 = HuffTable::build_from_lengths(&lengths[1]).expect("slot2");
            let slot3 = HuffTable::build_from_lengths(&lengths[2]).expect("slot3");

            // Reference emit using the pre-r221 per-byte body.
            let ref_frame = ref_emit_yuy2_per_byte(
                &frame.top_seed,
                &frame.combined_body,
                &slot1,
                &slot2,
                &slot3,
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
                &frame.top_seed,
                &frame.combined_body,
                &slot1,
                &slot2,
                &slot3,
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
        s1: &HuffTable,
        s2: &HuffTable,
        s3: &HuffTable,
    ) -> Result<()> {
        for (byte_idx, (_i, &sym)) in (4usize..).zip(body.iter().enumerate()) {
            let slot = match byte_idx & 3 {
                0 | 2 => s1,
                1 => s2,
                _ => s3,
            };
            if slot.entries[sym as usize].length == 0 {
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
                    histogramise(PixelFamily::Yuy2, method, &frame.combined_body);
                let (h1_ref, h2_ref, h3_ref) = ref_histogramise_yuy2_per_byte(&frame.combined_body);
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
                    frame.combined_body.len() as u64,
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
        let (s1, s2, s3) = build_v1x_tables(PixelFamily::Yuy2).expect("v1x build");
        let body: Vec<u8> = (0u8..64).collect();
        let prod = verify_body_in_table(PixelFamily::Yuy2, Method::Left, &body, &s1, &s2, &s3);
        let refr = ref_verify_body_in_table_yuy2_per_byte(&body, &s1, &s2, &s3);
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
        let (s1, s2, s3) = build_v1x_tables(PixelFamily::Yuy2).expect("v1x build");
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
            let prod = verify_body_in_table(PixelFamily::Yuy2, Method::Left, &body, &s1, &s2, &s3);
            let refr = ref_verify_body_in_table_yuy2_per_byte(&body, &s1, &s2, &s3);
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
        slot1: &HuffTable,
        slot2: &HuffTable,
        slot3: &HuffTable,
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
            let residuals =
                compute_residuals(PixelFamily::Rgb24, method, w, h, &pixels).expect("residuals");
            let (h1, h2, h3) = histogramise(PixelFamily::Rgb24, method, &residuals.body);
            let len1 = compute_canonical_lengths(&h1).expect("lens1");
            let len2 = compute_canonical_lengths(&h2).expect("lens2");
            let len3 = compute_canonical_lengths(&h3).expect("lens3");
            let s1 = HuffTable::build_from_lengths(&len1).expect("t1");
            let s2 = HuffTable::build_from_lengths(&len2).expect("t2");
            let s3 = HuffTable::build_from_lengths(&len3).expect("t3");
            let ref_bits = ref_emit_rgb24_per_byte_loop(method, &residuals.body, &s1, &s2, &s3);
            // The production frame begins with the 4-byte uncompressed
            // seed; the emitted bit stream follows.
            let prod_bits = &frame[4..];
            assert_eq!(
                prod_bits, ref_bits.as_slice(),
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
                    histogramise(PixelFamily::Rgb24, method, &frame.combined_body);
                let (h1_ref, h2_ref, h3_ref) =
                    ref_histogramise_rgb24_per_byte(method, &frame.combined_body);
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
                    frame.combined_body.len() as u64,
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
