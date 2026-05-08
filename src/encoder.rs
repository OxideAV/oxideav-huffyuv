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
use crate::predict::{gradient_predictor, is_interlaced_height, median3, split_fields};
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

    // Spec/02 §2 + spec/05 (planned): when biHeight > 288 the codec
    // splits the source into two fields (even rows = top; odd rows =
    // bottom) and predicts each independently. The two fields share
    // the same per-stream Huffman tables (the BIH carries one
    // extradata blob), but each field gets its own uncompressed seed +
    // bit-packed body. The on-wire frame is `top_seed | top_bits |
    // bot_seed | bot_bits`.
    let interlaced = is_interlaced_height(height);

    // Compute the per-channel residual stream first (independent of
    // table choice) so we can drive both the histogram computation
    // (CustomV2) and the bit-emit pass. For interlaced streams we
    // compute residuals once per field; for progressive a single
    // whole-frame pass.
    let progressive_residuals = if interlaced {
        None
    } else {
        Some(compute_residuals(family, method, width, height, pixels)?)
    };
    let (top_residuals, bot_residuals) = if interlaced {
        let row_bytes = match family {
            PixelFamily::Yuy2 => width as usize * 2,
            PixelFamily::Rgb24 => width as usize * 3,
            PixelFamily::Rgb32 => width as usize * 4,
        };
        let h = height as usize;
        let (top, bot) = split_fields(pixels, row_bytes, h);
        let top_h = h.div_ceil(2) as u32;
        let bot_h = (h / 2) as u32;
        let top_res = compute_residuals(family, method, width, top_h, &top)?;
        let bot_res = if bot_h > 0 {
            Some(compute_residuals(family, method, width, bot_h, &bot)?)
        } else {
            None
        };
        (Some(top_res), bot_res)
    } else {
        (None, None)
    };

    // For histogram (CustomV2) and table verification (V1xCompat)
    // we treat all residual bodies as one combined symbol stream.
    let combined_residuals_for_stats: Residuals = if interlaced {
        let mut body = Vec::new();
        if let Some(r) = &top_residuals {
            body.extend_from_slice(&r.body);
        }
        if let Some(r) = &bot_residuals {
            body.extend_from_slice(&r.body);
        }
        Residuals {
            seed: top_residuals.as_ref().map(|r| r.seed).unwrap_or([0; 4]),
            body,
        }
    } else {
        // Cheap clone via take + reinsert; we'll keep the original
        // for the emit pass below.
        progressive_residuals.as_ref().unwrap().clone()
    };

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
            let lengths =
                compute_lengths_from_residuals(family, method, &combined_residuals_for_stats);
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
            verify_residuals_in_table(
                family,
                method,
                &combined_residuals_for_stats,
                &s1,
                &s2,
                &s3,
            )?;
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

    let frame_bytes = if interlaced {
        let mut out = Vec::new();
        if let Some(r) = &top_residuals {
            let mut top_bytes = emit_bitstream(&EmitParams {
                family,
                method,
                res: r,
                slot1: &slot1,
                slot2: &slot2,
                slot3: &slot3,
            })?;
            out.append(&mut top_bytes);
        }
        if let Some(r) = &bot_residuals {
            let mut bot_bytes = emit_bitstream(&EmitParams {
                family,
                method,
                res: r,
                slot1: &slot1,
                slot2: &slot2,
                slot3: &slot3,
            })?;
            out.append(&mut bot_bytes);
        }
        out
    } else {
        emit_bitstream(&EmitParams {
            family,
            method,
            res: progressive_residuals.as_ref().unwrap(),
            slot1: &slot1,
            slot2: &slot2,
            slot3: &slot3,
        })?
    };

    // width/height already encoded into the BIH; no further use here.
    let _ = (width, height);
    Ok((strf, frame_bytes))
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
    // For Gradient: build intermediate = pixels - pixels_above (row ≥ 1).
    let intermediate: Vec<u8> = if method.predictor() == Predictor::Gradient {
        let mut iv = vec![0u8; row_bytes * h];
        iv[..row_bytes].copy_from_slice(&pixels[..row_bytes]);
        for row in 1..h {
            for col in 0..row_bytes {
                let idx = row * row_bytes + col;
                let above = pixels[(row - 1) * row_bytes + col];
                iv[idx] = pixels[idx].wrapping_sub(above);
            }
        }
        iv
    } else {
        pixels.to_vec()
    };
    let mut residuals = vec![0u8; row_bytes * h];
    if intermediate.len() >= 4 {
        residuals[..4].copy_from_slice(&intermediate[..4]);
    } else {
        residuals.copy_from_slice(&intermediate);
    }
    for i in 4..intermediate.len() {
        let stride = if i & 1 == 0 { 2 } else { 4 };
        residuals[i] = intermediate[i].wrapping_sub(intermediate[i - stride]);
    }
    if method.predictor() == Predictor::Median {
        // Replace LEFT residuals with MEDIAN residuals for row 1 byte
        // ≥ 8 + every later row. spec/03 §2.3 / §2.3.2.
        let row1_start = row_bytes;
        let row1_median_start = row1_start + 8.min(row_bytes);
        for pos in row1_median_start..pixels.len() {
            if pos < row_bytes {
                continue;
            }
            let l = pixels[pos.wrapping_sub(2)];
            let a = pixels[pos - row_bytes];
            let al = if pos >= row_bytes + 2 {
                pixels[pos - row_bytes - 2]
            } else {
                0
            };
            let g = gradient_predictor(l, a, al);
            let predictor = median3(l, a, g);
            residuals[pos] = pixels[pos].wrapping_sub(predictor);
        }
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
    let working: Vec<u8> = if method.decorrelate() {
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
        v
    } else {
        pixels.to_vec()
    };
    let intermediate: Vec<u8> = if method.predictor() == Predictor::Gradient {
        let mut iv = vec![0u8; row_bytes * h];
        iv[..row_bytes].copy_from_slice(&working[..row_bytes]);
        for row in 1..h {
            for col in 0..row_bytes {
                let idx = row * row_bytes + col;
                let above = working[(row - 1) * row_bytes + col];
                iv[idx] = working[idx].wrapping_sub(above);
            }
        }
        iv
    } else {
        working.clone()
    };
    let mut residuals = vec![0u8; row_bytes * h];
    for ch in 0..3usize {
        residuals[ch] = intermediate[ch];
        let mut idx = ch + 3;
        while idx < intermediate.len() {
            residuals[idx] = intermediate[idx].wrapping_sub(intermediate[idx - 3]);
            idx += 3;
        }
    }
    // Wire seed: `00 B G R` (the decoder writes pad as 0; first
    // working pixel goes into bytes 1..4).
    let mut seed = [0u8; 4];
    seed[1] = working[0];
    seed[2] = working[1];
    seed[3] = working[2];
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
    let working: Vec<u8> = if method.decorrelate() {
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
        v
    } else {
        pixels.to_vec()
    };
    let intermediate: Vec<u8> = if method.predictor() == Predictor::Gradient {
        let mut iv = vec![0u8; row_bytes * h];
        iv[..row_bytes].copy_from_slice(&working[..row_bytes]);
        for row in 1..h {
            for col in 0..row_bytes {
                let idx = row * row_bytes + col;
                let above = working[(row - 1) * row_bytes + col];
                iv[idx] = working[idx].wrapping_sub(above);
            }
        }
        iv
    } else {
        working.clone()
    };
    let mut residuals = vec![0u8; row_bytes * h];
    for ch in 0..4usize {
        residuals[ch] = intermediate[ch];
        let mut idx = ch + 4;
        while idx < intermediate.len() {
            residuals[idx] = intermediate[idx].wrapping_sub(intermediate[idx - 4]);
            idx += 4;
        }
    }
    let mut seed = [0u8; 4];
    seed.copy_from_slice(&working[..4]);
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

fn compute_lengths_from_residuals(
    family: PixelFamily,
    method: Method,
    res: &Residuals,
) -> [[u8; 256]; 3] {
    // Build per-slot histograms by walking `res.body` with the
    // family-specific slot-mapping rule.
    let (h1, h2, h3) = histogramise(family, method, &res.body);
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
            // body[i] corresponds to source byte index (i + 4).
            for (i, &b) in body.iter().enumerate() {
                let byte_idx = i + 4;
                match byte_idx & 3 {
                    0 | 2 => h1[b as usize] += 1,
                    1 => h2[b as usize] += 1,
                    _ => h3[b as usize] += 1,
                }
            }
        }
        PixelFamily::Rgb24 => {
            // body is laid out 3 bytes per pixel in wire order.
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

fn build_v1x_tables(family: PixelFamily) -> Result<(HuffTable, HuffTable, HuffTable)> {
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

fn verify_residuals_in_table(
    family: PixelFamily,
    method: Method,
    res: &Residuals,
    s1: &HuffTable,
    s2: &HuffTable,
    s3: &HuffTable,
) -> Result<()> {
    // Walk body in slot order; ensure each symbol has a non-zero
    // length in its slot's table. Any miss returns an Unsupported
    // error so callers know to fall back to a custom or classic
    // table.
    for (byte_idx, (i, &sym)) in (4usize..).zip(res.body.iter().enumerate()) {
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

struct EmitParams<'a> {
    family: PixelFamily,
    method: Method,
    res: &'a Residuals,
    slot1: &'a HuffTable,
    slot2: &'a HuffTable,
    slot3: &'a HuffTable,
}

fn emit_bitstream(p: &EmitParams) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    out.extend_from_slice(&p.res.seed);
    let mut writer = BitWriter::new();
    match p.family {
        PixelFamily::Yuy2 => {
            for (byte_idx, &sym) in (4usize..).zip(p.res.body.iter()) {
                let slot = match byte_idx & 3 {
                    0 | 2 => p.slot1,
                    1 => p.slot2,
                    _ => p.slot3,
                };
                let (code, length) = lookup_code(slot, sym)?;
                writer.write_msb(code, length);
            }
        }
        PixelFamily::Rgb24 => {
            for (i, &sym) in p.res.body.iter().enumerate() {
                let in_pixel = i % 3;
                let slot = if p.method.decorrelate() {
                    match in_pixel {
                        0 => p.slot2,
                        1 => p.slot1,
                        _ => p.slot3,
                    }
                } else {
                    match in_pixel {
                        0 => p.slot1,
                        1 => p.slot2,
                        _ => p.slot3,
                    }
                };
                let (code, length) = lookup_code(slot, sym)?;
                writer.write_msb(code, length);
            }
        }
        PixelFamily::Rgb32 => {
            for (i, &sym) in p.res.body.iter().enumerate() {
                let in_pixel = i % 4;
                let slot = if p.method.decorrelate() {
                    match in_pixel {
                        0 => p.slot2,
                        1 => p.slot1,
                        _ => p.slot3,
                    }
                } else {
                    match in_pixel {
                        0 => p.slot1,
                        1 => p.slot2,
                        _ => p.slot3,
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
