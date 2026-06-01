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
    inverse_rgb_decorr_bgra, inverse_yuy2_left_macropixel, is_interlaced_height,
};
use crate::tables::{
    classic_blob_bytes, decode_one, rle_decode_one_channel, rle_decode_three_channels,
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
    let three_tables = build_three_tables(config)?;
    if is_interlaced_height(config.height) {
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
    let row_bytes = match config.family {
        PixelFamily::Yuy2 => config.width as usize * 2,
        PixelFamily::Rgb24 => config.width as usize * 3,
        PixelFamily::Rgb32 => config.width as usize * 4,
    };
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

/// The three per-channel-slot Huffman tables (spec/03 §1.2 slot
/// architecture). Slot 1 = Y / B / B-G; Slot 2 = U / G / G; Slot 3 =
/// V / R / R-G (and reused for RGB32 alpha).
struct ThreeTables {
    slot1: HuffTable,
    slot2: HuffTable,
    slot3: HuffTable,
}

fn build_three_tables(config: &StreamConfig) -> Result<ThreeTables> {
    if config.has_extradata {
        // v2.x extradata path: 3 RLE-compressed length tables in
        // slot-1, slot-2, slot-3 order, then build canonical Huffman.
        let lengths = if !config.extradata_tables.is_empty() {
            rle_decode_three_channels(&config.extradata_tables)?
        } else {
            // Fallback: extradata-present flag set but the table
            // region is empty — shouldn't happen on the wire but we
            // accept it by falling through to the classic blob for
            // this (family, method).
            let blob = classic_blob_bytes(config.family, config.method);
            rle_decode_three_channels(blob)?
        };
        Ok(ThreeTables {
            slot1: HuffTable::build_from_lengths(&lengths[0])?,
            slot2: HuffTable::build_from_lengths(&lengths[1])?,
            slot3: HuffTable::build_from_lengths(&lengths[2])?,
        })
    } else {
        // v1.x precomputed-codes path (spec/04 §4): the per-channel
        // sharing depends on the family.
        let lens_a_blob = v1x_lengths_set_a();
        let lens_b_blob = v1x_lengths_set_b();
        let mut cursor: &[u8] = lens_a_blob;
        let lens_a = rle_decode_one_channel(&mut cursor)?;
        let mut cursor: &[u8] = lens_b_blob;
        let lens_b = rle_decode_one_channel(&mut cursor)?;
        let codes_a_buf = v1x_codes_set_a();
        let codes_b_buf = v1x_codes_set_b();
        let mut codes_a = [0u8; 256];
        codes_a.copy_from_slice(codes_a_buf);
        let mut codes_b = [0u8; 256];
        codes_b.copy_from_slice(codes_b_buf);
        let table_a = v1x_table_from_pair(&lens_a, &codes_a)?;
        let table_b = v1x_table_from_pair(&lens_b, &codes_b)?;

        match config.family {
            PixelFamily::Yuy2 => Ok(ThreeTables {
                // v1.x YUY2: Y uses set A; both U and V use set B.
                slot1: table_a,
                slot2: table_b.clone(),
                slot3: table_b,
            }),
            PixelFamily::Rgb24 | PixelFamily::Rgb32 => Ok(ThreeTables {
                // v1.x RGB: all three of B, G, R use set A.
                slot1: table_a.clone(),
                slot2: table_a.clone(),
                slot3: table_a,
            }),
        }
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
    let row_bytes = width_us * 2; // Y₁ U Y₂ V per 2 px = 4 bytes per pair × (width/2) pairs = width × 2.
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

    // Wire byte → channel slot, repeating every 4 bytes:
    //   +0 (Y₁) → slot1; +1 (U) → slot2; +2 (Y₂) → slot1; +3 (V) → slot3
    // Total per-frame codes: (width × height) channel samples - 4
    // (uncompressed first pixel). For YUY2, channel-sample count =
    // total_bytes - 4.
    for (byte_idx, slot_pixel) in pixels.iter_mut().enumerate().take(total_bytes).skip(4) {
        let slot = match byte_idx % 4 {
            0 | 2 => &tables.slot1,
            1 => &tables.slot2,
            _ => &tables.slot3,
        };
        let window = reader.peek_window();
        let (sym, len) = decode_one(slot, window)?;
        reader.consume_bits(len as u32)?;
        *slot_pixel = sym;
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
    let row_bytes = width_us * 3;
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

    // For each subsequent pixel, the codeword count + slot order
    // depends on whether decorrelation is enabled (spec/03 §1.4):
    //   - non-decorr:  3 codes per pixel: B (slot1), G (slot2), R (slot3)
    //   - decorr:      3 codes per pixel: G (slot2), B-G (slot1), R-G (slot3)
    // The reconstructed buffer always stores 3-byte BGR.
    let n_pixels = width_us * height_us;
    for px in 1..n_pixels {
        let bgr_off = px * 3;
        if decorrelate {
            // G first.
            let (g, lg) = decode_one(&tables.slot2, reader.peek_window())?;
            reader.consume_bits(lg as u32)?;
            // B-G next.
            let (bg, lb) = decode_one(&tables.slot1, reader.peek_window())?;
            reader.consume_bits(lb as u32)?;
            // R-G last.
            let (rg, lr) = decode_one(&tables.slot3, reader.peek_window())?;
            reader.consume_bits(lr as u32)?;
            // Store decorrelated values in BGR layout: B = (B-G), G,
            // R = (R-G). The decorrelation transform inverse runs
            // after the predictor pass below.
            pixels[bgr_off] = bg;
            pixels[bgr_off + 1] = g;
            pixels[bgr_off + 2] = rg;
        } else {
            let (b, lb) = decode_one(&tables.slot1, reader.peek_window())?;
            reader.consume_bits(lb as u32)?;
            let (g, lg) = decode_one(&tables.slot2, reader.peek_window())?;
            reader.consume_bits(lg as u32)?;
            let (r, lr) = decode_one(&tables.slot3, reader.peek_window())?;
            reader.consume_bits(lr as u32)?;
            pixels[bgr_off] = b;
            pixels[bgr_off + 1] = g;
            pixels[bgr_off + 2] = r;
        }
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
    let row_bytes = width_us * 4;
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

    for px in 1..n_pixels {
        let off = px * 4;
        if decorrelate {
            // wire order: G, B-G, R-G, A. Slot mapping: G→slot2,
            // B-G→slot1, R-G→slot3, A→slot3 (alpha shares slot-3
            // codebook per spec/03 §1.2).
            let (g, lg) = decode_one(&tables.slot2, reader.peek_window())?;
            reader.consume_bits(lg as u32)?;
            let (bg, lb) = decode_one(&tables.slot1, reader.peek_window())?;
            reader.consume_bits(lb as u32)?;
            let (rg, lr) = decode_one(&tables.slot3, reader.peek_window())?;
            reader.consume_bits(lr as u32)?;
            let (a, la) = decode_one(&tables.slot3, reader.peek_window())?;
            reader.consume_bits(la as u32)?;
            pixels[off] = bg;
            pixels[off + 1] = g;
            pixels[off + 2] = rg;
            pixels[off + 3] = a;
        } else {
            let (b, lb) = decode_one(&tables.slot1, reader.peek_window())?;
            reader.consume_bits(lb as u32)?;
            let (g, lg) = decode_one(&tables.slot2, reader.peek_window())?;
            reader.consume_bits(lg as u32)?;
            let (r, lr) = decode_one(&tables.slot3, reader.peek_window())?;
            reader.consume_bits(lr as u32)?;
            // A residual reuses slot-3 codebook (spec/03 §1.2).
            let (a, la) = decode_one(&tables.slot3, reader.peek_window())?;
            reader.consume_bits(la as u32)?;
            pixels[off] = b;
            pixels[off + 1] = g;
            pixels[off + 2] = r;
            pixels[off + 3] = a;
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
    debug_assert!(row1_left_end >= row_bytes + 2);
    for pos in row1_left_end..len {
        let l = out[pos - 2];
        let a = out[pos - row_bytes];
        let al = out[pos - row_bytes - 2];
        let g = l.wrapping_add(a).wrapping_sub(al);
        let predictor = crate::predict::median3(l, a, g);
        out[pos] = out[pos].wrapping_add(predictor);
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
