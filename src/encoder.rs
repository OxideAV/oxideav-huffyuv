//! Round-1 self-test encoder.
//!
//! This is **only** used to synthesise minimal HuffYUV frames so that
//! `decode_frame` has something to round-trip in unit tests. It does
//! the smallest possible job: predict each pixel against its
//! immediate context per `spec/03 §2`, look up the per-symbol
//! Huffman code from the same per-channel tables the decoder uses,
//! and emit a 32-bit-MSB-first packed bitstream per `spec/02 §4`.
//!
//! Round 1 supports the 4 method bytes that the user's deliverable
//! list calls out: `Method::PredictOld` / `Method::Left` (= yuv-left
//! / rgb-left), `Method::Median` (= yuv-median), `Method::LeftDecorr`
//! (= rgb-left-decorr), and `Method::GradientDecorr` (=
//! rgb-gradient-decorr). It uses the v2.x extradata path with one of
//! the six classic blobs as the embedded length tables; the Auditor
//! round can swap in v1.x synthesis when the lockstep harness gates
//! on it.

use crate::bitio::BitWriter;
use crate::error::{Error, Result};
use crate::header::{Method, PixelFamily, Predictor, StreamConfig};
use crate::predict::{gradient_predictor, median3};
use crate::tables::{classic_blob_bytes, rle_decode_three_channels, HuffEntry, HuffTable};

/// Synthesise both the BIH-extradata `extradata_tables` blob (which
/// the encoder embeds verbatim from one of the six classic blobs)
/// and the per-frame compressed bytes for `pixels`. Returns
/// `(strf_bytes, frame_bytes)` — the strf is a complete
/// `BITMAPINFOHEADER` ready to feed back through
/// [`StreamConfig::parse_bitmapinfoheader`].
pub fn encode_frame(
    family: PixelFamily,
    method: Method,
    width: u32,
    height: u32,
    pixels: &[u8],
) -> Result<(Vec<u8>, Vec<u8>)> {
    if family.is_rgb() && !method.is_rgb_legal() {
        return Err(Error::invalid("encoder: method not legal for RGB"));
    }
    if !family.is_rgb() && !method.is_yuv_legal() {
        return Err(Error::invalid("encoder: method not legal for YUV"));
    }

    let extradata_tables = classic_blob_bytes(family, method).to_vec();
    let strf = build_bih(family, method, width, height, &extradata_tables);

    let lengths = rle_decode_three_channels(&extradata_tables)?;
    let slot1 = HuffTable::build_from_lengths(&lengths[0])?;
    let slot2 = HuffTable::build_from_lengths(&lengths[1])?;
    let slot3 = HuffTable::build_from_lengths(&lengths[2])?;

    let frame_bytes = match family {
        PixelFamily::Yuy2 => encode_yuy2(method, width, height, pixels, &slot1, &slot2, &slot3)?,
        PixelFamily::Rgb24 => encode_rgb24(method, width, height, pixels, &slot1, &slot2, &slot3)?,
        PixelFamily::Rgb32 => encode_rgb32(method, width, height, pixels, &slot1, &slot2, &slot3)?,
    };
    Ok((strf, frame_bytes))
}

fn build_bih(
    family: PixelFamily,
    method: Method,
    width: u32,
    height: u32,
    extradata_tables: &[u8],
) -> Vec<u8> {
    // bi_size = 0x28 (BIH) + 0x04 (fixed prefix) + extradata_tables.len()
    let bi_size = 0x2C + extradata_tables.len() as u32;
    let bit_count = match family {
        PixelFamily::Yuy2 => 24, // encoder forces ≥ 24 even for YUY2.
        PixelFamily::Rgb24 => 24,
        PixelFamily::Rgb32 => 32,
    };
    let mut v = vec![0u8; (bi_size as usize).max(0x2C)];
    v[0..4].copy_from_slice(&bi_size.to_le_bytes());
    v[4..8].copy_from_slice(&(width as i32).to_le_bytes());
    v[8..12].copy_from_slice(&(height as i32).to_le_bytes());
    v[12..14].copy_from_slice(&1u16.to_le_bytes());
    v[0x0E..0x10].copy_from_slice(&(bit_count as u16).to_le_bytes());
    v[0x10..0x14].copy_from_slice(&crate::header::FOURCC_HFYU.to_le_bytes());
    // Method byte at +0x28; bpp_override at +0x29.
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

/// Look up the MSB-aligned (code, length) pair for symbol `sym` in
/// `table`. Returns an error when the table assigns no code to that
/// symbol — a clean-room encoder must pick a length table whose
/// non-zero entries cover every byte its residual stream emits.
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

// ───────────────────────── per-family encoders ─────────────────────────

fn encode_yuy2(
    method: Method,
    width: u32,
    height: u32,
    pixels: &[u8],
    slot1: &HuffTable,
    slot2: &HuffTable,
    slot3: &HuffTable,
) -> Result<Vec<u8>> {
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
    // Compute the residual stream that decoders consume. Residuals
    // are produced by the encoder side of the predictor inverse
    // documented in `spec/03 §2`: for LEFT, residual = pixel - left;
    // for GRADIENT, the encoder emits LEFT-residuals first then runs
    // a "subtract row above" post-pass on rows ≥ 1; for MEDIAN, after
    // the same LEFT pass, the per-byte median post-pass replaces the
    // already-LEFT'd residuals with `(pixel - median)` for row ≥ 1
    // starting at byte 8 of row 1.
    // Build intermediate stream. For LEFT, intermediate = pixels.
    // For GRADIENT, intermediate[row 0] = pixels[row 0]; intermediate
    // [row N ≥ 1, col] = pixels[row N, col] - pixels[row N-1, col].
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
    residuals[..4].copy_from_slice(&intermediate[..4]);
    for i in 4..intermediate.len() {
        let stride = if i & 1 == 0 { 2 } else { 4 };
        residuals[i] = intermediate[i].wrapping_sub(intermediate[i - stride]);
    }
    match method.predictor() {
        Predictor::Left | Predictor::Gradient => {}
        Predictor::Median => {
            // For row 1 starting at byte 8 + every later row, replace
            // residual = pixel - median(L, A, G). spec/03 §2.3 traces
            // the offsets in YUY2 4-byte stride: L at -2, A at
            // -row_stride, AL at -row_stride - 2.
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
    }

    // Emit: first 4 bytes uncompressed, then per-byte Huffman codes.
    let mut out = Vec::with_capacity(4 + (row_bytes * h));
    out.extend_from_slice(&pixels[..4]);
    let mut writer = BitWriter::new();
    for (byte_idx, &sym) in residuals.iter().enumerate().skip(4) {
        let slot = match byte_idx % 4 {
            0 | 2 => slot1,
            1 => slot2,
            _ => slot3,
        };
        let (code, length) = lookup_code(slot, sym)?;
        writer.write_msb(code, length);
    }
    out.extend_from_slice(&writer.finish());
    Ok(out)
}

fn encode_rgb24(
    method: Method,
    width: u32,
    height: u32,
    pixels: &[u8], // BGR-packed
    slot1: &HuffTable,
    slot2: &HuffTable,
    slot3: &HuffTable,
) -> Result<Vec<u8>> {
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

    // Apply RGB decorrelation FIRST (if active). The wire-format
    // residuals are LEFT-residuals (or gradient-residuals) of the
    // *decorrelated* per-channel stream.
    let working: Vec<u8> = if method.decorrelate() {
        let mut v = vec![0u8; row_bytes * h];
        for px in 0..n_pixels {
            let off = px * 3;
            let b = pixels[off];
            let g = pixels[off + 1];
            let r = pixels[off + 2];
            v[off] = b.wrapping_sub(g); // (B - G)
            v[off + 1] = g; // G
            v[off + 2] = r.wrapping_sub(g); // (R - G)
        }
        v
    } else {
        pixels.to_vec()
    };

    // Build the per-position "intermediate" stream. For LEFT
    // (non-gradient), intermediate = working. For GRADIENT,
    // intermediate[i] = working[i] - working[above_i] for row ≥ 1,
    // working[i] otherwise. The wire residuals are then per-channel
    // LEFT residuals of the intermediate stream.
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

    // Emit: 4 uncompressed bytes (00 B' G R'), where B'/R' are the
    // decorrelated values when applicable. The first uncompressed
    // pixel becomes the LEFT-pass seed on decode, so it must be the
    // decorrelated value (so that row 0's gradient-post does nothing
    // and recovers the decorrelated stream).
    let mut out = Vec::with_capacity(4 + row_bytes * h);
    out.push(0); // pad
    out.push(working[0]); // B (decorrelated when applicable)
    out.push(working[1]); // G
    out.push(working[2]); // R (decorrelated when applicable)

    let mut writer = BitWriter::new();
    for px in 1..n_pixels {
        let off = px * 3;
        // The wire codeword order for RGB depends on decorrelation:
        //   non-decorr: B (slot1), G (slot2), R (slot3)
        //   decorr:     G (slot2), B-G (slot1), R-G (slot3)
        if method.decorrelate() {
            // residuals layout above stored: off+0 = (B-G) residual,
            // off+1 = G residual, off+2 = (R-G) residual.
            let g_sym = residuals[off + 1];
            let bg_sym = residuals[off];
            let rg_sym = residuals[off + 2];
            let (c, l) = lookup_code(slot2, g_sym)?;
            writer.write_msb(c, l);
            let (c, l) = lookup_code(slot1, bg_sym)?;
            writer.write_msb(c, l);
            let (c, l) = lookup_code(slot3, rg_sym)?;
            writer.write_msb(c, l);
        } else {
            let (c, l) = lookup_code(slot1, residuals[off])?; // B
            writer.write_msb(c, l);
            let (c, l) = lookup_code(slot2, residuals[off + 1])?; // G
            writer.write_msb(c, l);
            let (c, l) = lookup_code(slot3, residuals[off + 2])?; // R
            writer.write_msb(c, l);
        }
    }
    out.extend_from_slice(&writer.finish());
    Ok(out)
}

fn encode_rgb32(
    method: Method,
    width: u32,
    height: u32,
    pixels: &[u8], // BGRA-packed
    slot1: &HuffTable,
    slot2: &HuffTable,
    slot3: &HuffTable,
) -> Result<Vec<u8>> {
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

    let mut out = Vec::with_capacity(4 + row_bytes * h);
    out.extend_from_slice(&working[..4]);
    let mut writer = BitWriter::new();
    for px in 1..n_pixels {
        let off = px * 4;
        if method.decorrelate() {
            let (c, l) = lookup_code(slot2, residuals[off + 1])?; // G
            writer.write_msb(c, l);
            let (c, l) = lookup_code(slot1, residuals[off])?; // B-G
            writer.write_msb(c, l);
            let (c, l) = lookup_code(slot3, residuals[off + 2])?; // R-G
            writer.write_msb(c, l);
            // Alpha residual via slot 3.
            let (c, l) = lookup_code(slot3, residuals[off + 3])?;
            writer.write_msb(c, l);
        } else {
            let (c, l) = lookup_code(slot1, residuals[off])?; // B
            writer.write_msb(c, l);
            let (c, l) = lookup_code(slot2, residuals[off + 1])?; // G
            writer.write_msb(c, l);
            let (c, l) = lookup_code(slot3, residuals[off + 2])?; // R
            writer.write_msb(c, l);
            let (c, l) = lookup_code(slot3, residuals[off + 3])?; // A via slot 3
            writer.write_msb(c, l);
        }
    }
    out.extend_from_slice(&writer.finish());
    Ok(out)
}

/// Synthesise a complete strf payload + frame chunk and return both
/// for tests. Convenience wrapper around [`encode_frame`].
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
