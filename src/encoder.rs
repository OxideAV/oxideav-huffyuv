//! HuffYUV / FFVHuff frame encoder.
//!
//! Mirror of [`crate::decoder`]. Each pushed [`VideoFrame`] is encoded
//! as one self-contained intra packet:
//!
//! 1. (Optional) per-frame Huffman length tables, RLE-encoded, prepended
//!    when `extra[2] & 0x40` is set on the chosen extradata. **Today
//!    every encoder packet carries its own histogram-derived per-frame
//!    tables** (FFVHuff `-context 1` style) — this gives us the
//!    bit-exact-roundtrip property without having to ship a parabolic-
//!    prior table builder.
//! 2. The 4-byte format prelude (raw bytes for v2 YUV/RGB; nothing for
//!    v3 planar / gray).
//! 3. The per-channel residuals computed from the configured spatial
//!    predictor (LEFT / GRADIENT / MEDIAN), Huffman-coded with the
//!    canonical codes derived from the (just-emitted) length tables.
//! 4. A 31-bit zero tail (`put_bits(16,0); put_bits(15,0)`).
//! 5. Pad to 4-byte alignment, then byte-swap each 32-bit word.
//!
//! The encoder accepts a [`CodecParameters`] whose `extradata` field is
//! interpreted as an upstream-emitted extradata blob and re-used
//! verbatim (so muxer-side calls can stuff the same blob into a `strf`
//! chunk). When `extradata` is empty the encoder synthesises its own
//! v2 / v3 extradata blob from `pixel_format` + `predictor` and
//! exposes it via [`Encoder::output_params`].

use oxideav_core::frame::VideoPlane;
use oxideav_core::{
    CodecId, CodecParameters, Encoder, Error, Frame, Packet, PixelFormat, Result, VideoFrame,
};

use crate::bitwriter::{swap_payload, BitWriter};
use crate::extradata::{Extradata, FormatVersion, Predictor, V2Colorspace};
use crate::huffman::HuffTable;
use crate::length_builder::build_lengths;
use crate::rle_encode::encode_lengths;

const HUFF_MAX_LEN: u8 = 16; // matches the upstream encoder cap (`MAX_BITS = 16`).

/// Build a [`HuffyuvEncoder`] for the requested [`CodecParameters`].
///
/// The encoder honours `params.codec_id` (`huffyuv` → emits v2 extradata
/// where representable; `ffvhuff` → emits v3 extradata for everything
/// the v2 layout cannot describe), `params.pixel_format`,
/// `params.width`, `params.height`, plus any extradata overrides.
pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    let width = params
        .width
        .ok_or_else(|| Error::invalid("huffyuv encoder: missing width"))? as usize;
    let height = params
        .height
        .ok_or_else(|| Error::invalid("huffyuv encoder: missing height"))?
        as usize;
    let pixel_format = params
        .pixel_format
        .ok_or_else(|| Error::invalid("huffyuv encoder: missing pixel_format"))?;

    let predictor = predictor_from_options(params)?;
    let prefer_v2 = params.codec_id.as_str() == crate::CODEC_ID_HUFFYUV;
    let extradata_blob = if !params.extradata.is_empty() {
        params.extradata.clone()
    } else {
        synthesise_extradata(pixel_format, predictor, prefer_v2)?
    };
    let extradata = Extradata::parse(&extradata_blob)?;

    crate::decoder::validate_shape_pub(&extradata, width)?;

    let mut output_params = params.clone();
    output_params.extradata = extradata_blob.clone();
    output_params.pixel_format = Some(pixel_format);
    output_params.width = Some(width as u32);
    output_params.height = Some(height as u32);

    Ok(Box::new(HuffyuvEncoder {
        codec_id: params.codec_id.clone(),
        extradata,
        width,
        height,
        pixel_format,
        output_params,
        pending: None,
    }))
}

/// Read the `predictor` option string (default: `left`).
fn predictor_from_options(params: &CodecParameters) -> Result<Predictor> {
    if let Some(v) = params.options.get("predictor") {
        match v.to_ascii_lowercase().as_str() {
            "left" => Ok(Predictor::Left),
            "plane" | "gradient" => Ok(Predictor::Gradient),
            "median" => Ok(Predictor::Median),
            other => Err(Error::invalid(format!(
                "huffyuv encoder: unknown predictor option `{other}`"
            ))),
        }
    } else {
        Ok(Predictor::Left)
    }
}

/// Build a synthetic extradata blob from `(pixel_format, predictor)`.
/// The Huffman tables embedded in this default extradata are
/// **fallback** uniform-length tables; every emitted packet will carry
/// its own per-frame tables on top.
fn synthesise_extradata(
    pixel_format: PixelFormat,
    predictor: Predictor,
    prefer_v2: bool,
) -> Result<Vec<u8>> {
    let mut extra: Vec<u8> = Vec::new();
    let pred_byte = predictor as u8;
    let want_decorrelate = matches!(pixel_format, PixelFormat::Rgb24 | PixelFormat::Bgra);
    let e0 = pred_byte | if want_decorrelate { 0x40 } else { 0x00 };

    // Per-frame-tables flag set so the encoder can ship histogram-built
    // tables on every packet. The decoder honours the flag whether or
    // not the embedded extradata tables are uniform.
    let per_frame_flag: u8 = 0x40;
    // Auto interlace mode (bits 4..5 == 00).

    let (e1, e2, e3, n_tables, symbols) = match (pixel_format, prefer_v2) {
        (PixelFormat::Yuv422P, true) => (16u8, per_frame_flag, 0u8, 3usize, 256usize),
        (PixelFormat::Yuv420P, true) => (12u8, per_frame_flag, 0u8, 3, 256),
        (PixelFormat::Rgb24, true) => (24u8, per_frame_flag, 0u8, 3, 256),
        (PixelFormat::Bgra, true) => (32u8, per_frame_flag, 0u8, 3, 256),
        (pf, _) => v3_extradata_bytes(pf, per_frame_flag, predictor)?,
    };

    extra.extend_from_slice(&[e0, e1, e2, e3]);
    // Fallback Huffman lengths: uniform for the requested alphabet.
    // Compute the smallest power-of-two that ≥ symbols, then assign
    // length = log2 of that across the live entries (and put the leftover
    // mass into the lowest-index symbols if symbols isn't a power of 2).
    let lens = uniform_lengths(symbols);
    let blob = encode_lengths(&lens);
    for _ in 0..n_tables {
        extra.extend_from_slice(&blob);
    }
    Ok(extra)
}

fn uniform_lengths(symbols: usize) -> Vec<u8> {
    if symbols == 0 {
        return Vec::new();
    }
    if symbols == 1 {
        return vec![1];
    }
    let pow = symbols.next_power_of_two();
    let l = pow.trailing_zeros() as u8;
    if symbols == pow {
        vec![l; symbols]
    } else {
        // Extra bit on the prefix-doubled half (canonical-Huffman prefix code).
        let mut out = vec![l; symbols];
        let extras = pow - symbols;
        // Upgrade the last 2*extras symbols to length l+1 — equivalent
        // to a binary tree where the deepest 2*extras leaves are pushed
        // down by one and the (extras) shallow nodes above them
        // disappear. Simple Kraft check: we add `extras * 2 * 2^-(l+1)
        // = extras * 2^-l` and subtract `extras * 2^-l`, net zero.
        let promote = (2 * extras).min(symbols);
        for i in (symbols - promote)..symbols {
            out[i] = l + 1;
        }
        out
    }
}

fn v3_extradata_bytes(
    pf: PixelFormat,
    per_frame_flag: u8,
    _predictor: Predictor,
) -> Result<(u8, u8, u8, usize, usize)> {
    // Returns (e1, e2, e3=1, n_tables, symbols_per_table).
    let (bps, h_shift, v_shift, yuv, chroma, alpha) = match pf {
        PixelFormat::Gray8 => (8u8, 0u8, 0u8, false, false, false),
        PixelFormat::Yuv444P => (8, 0, 0, true, true, false),
        PixelFormat::Yuv422P => (8, 1, 0, true, true, false),
        PixelFormat::Yuv420P => (8, 1, 1, true, true, false),
        PixelFormat::Yuv411P => (8, 2, 0, true, true, false),
        PixelFormat::Yuv420P10Le => (10, 1, 1, true, true, false),
        PixelFormat::Yuv422P10Le => (10, 1, 0, true, true, false),
        PixelFormat::Yuv444P10Le => (10, 0, 0, true, true, false),
        PixelFormat::Yuv420P12Le => (12, 1, 1, true, true, false),
        PixelFormat::Yuv422P12Le => (12, 1, 0, true, true, false),
        PixelFormat::Yuv444P12Le => (12, 0, 0, true, true, false),
        PixelFormat::Gray10Le => (10, 0, 0, false, false, false),
        PixelFormat::Gray12Le => (12, 0, 0, false, false, false),
        PixelFormat::Gray16Le => (16, 0, 0, false, false, false),
        // GBRP / GBRAP planar: yuv=0, chroma=1 (explicit bit-1 set).
        // 8-bit values ride in Gbrp10Le / Gbrap10Le containers but
        // the on-wire bps is 8 (low 8 bits of the 16-bit LE words used).
        PixelFormat::Gbrp10Le => (10, 0, 0, false, true, false),
        PixelFormat::Gbrap10Le => (10, 0, 0, false, true, true),
        PixelFormat::Gbrp12Le => (12, 0, 0, false, true, false),
        PixelFormat::Gbrap12Le => (12, 0, 0, false, true, true),
        other => {
            return Err(Error::unsupported(format!(
                "huffyuv encoder: pixel format {other:?} not supported in v3 extradata"
            )));
        }
    };
    let e1 = ((bps - 1) << 4) | (h_shift & 0x03) | ((v_shift & 0x03) << 2);
    let mut e2 = per_frame_flag;
    if yuv {
        e2 |= 0x01;
    }
    // Bit 1 (`chroma`) only fires for non-yuv multi-plane layouts
    // (gbrp / gbrap). For yuv layouts the chroma is implied by
    // bit 0 — see the parser comment in extradata.rs.
    if chroma && !yuv {
        e2 |= 0x02;
    }
    if alpha {
        e2 |= 0x04;
    }
    let n_tables = 1 + (if chroma { 2 } else { 0 }) + (if alpha { 1 } else { 0 });
    let raw = 1usize << bps;
    let symbols = raw.min(16384);
    Ok((e1, e2, 1u8, n_tables, symbols))
}

struct HuffyuvEncoder {
    codec_id: CodecId,
    extradata: Extradata,
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    output_params: CodecParameters,
    pending: Option<Packet>,
}

impl Encoder for HuffyuvEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.output_params
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        let v = match frame {
            Frame::Video(v) => v,
            _ => {
                return Err(Error::invalid(
                    "huffyuv encoder: only video frames are supported",
                ));
            }
        };
        if self.pending.is_some() {
            return Err(Error::other(
                "huffyuv encoder: receive_packet must be called between frames",
            ));
        }
        let payload = encode_frame(
            &self.extradata,
            v,
            self.width,
            self.height,
            self.pixel_format,
        )?;
        self.pending = Some(
            Packet::new(0, oxideav_core::time::TimeBase::new(1, 25), payload).with_keyframe(true),
        );
        Ok(())
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        self.pending.take().ok_or(Error::NeedMore)
    }

    fn flush(&mut self) -> Result<()> {
        // No deferred state. Subsequent receive_packet returns NeedMore
        // until another send_frame happens.
        Ok(())
    }
}

// ─────────────────────── frame encode dispatch ──────────────────────

/// One residual sample tagged with the Huffman table it belongs to.
type TaggedResidual = (u8, u32); // (table_idx, residual)

fn encode_frame(
    extra: &Extradata,
    v: &VideoFrame,
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
) -> Result<Vec<u8>> {
    // Build a residual stream in **bitstream order** (the order the
    // decoder walks: interleaved per-pixel-group for v2 layouts,
    // plane-sequential for v3). Each residual is tagged with the
    // Huffman table to use, so downstream Huffman code building still
    // works per-table while the bitstream emit can walk the stream
    // as-is.
    let bps = extra.bits_per_sample();
    let mask: u32 = if bps >= 32 {
        u32::MAX
    } else {
        (1u32 << bps) - 1
    };
    let alphabet = bps_alphabet(bps);
    let n_tables = extra.table_count();

    let (stream, prelude) = match (&extra.format, pixel_format) {
        (FormatVersion::V2(V2Colorspace::Yuv422), PixelFormat::Yuv422P) => {
            build_v2_yuv422(extra, v, width, height)?
        }
        (FormatVersion::V2(V2Colorspace::Yuv420), PixelFormat::Yuv420P) => {
            build_v2_yuv420(extra, v, width, height)?
        }
        (FormatVersion::V2(V2Colorspace::Rgb24), PixelFormat::Rgb24) => {
            build_v2_rgb(extra, v, width, height, false)?
        }
        (FormatVersion::V2(V2Colorspace::Bgra32), PixelFormat::Bgra) => {
            build_v2_rgb(extra, v, width, height, true)?
        }
        (FormatVersion::V3(_), PixelFormat::Gray8) => build_v3_gray(extra, v, width, height, 8)?,
        (FormatVersion::V3(_), PixelFormat::Gray10Le) => {
            build_v3_gray(extra, v, width, height, 10)?
        }
        (FormatVersion::V3(_), PixelFormat::Gray12Le) => {
            build_v3_gray(extra, v, width, height, 12)?
        }
        (FormatVersion::V3(_), PixelFormat::Gray16Le) => {
            build_v3_gray(extra, v, width, height, 16)?
        }
        (FormatVersion::V3(_), PixelFormat::Yuv411P) => {
            build_v3_yuv(extra, v, width, height, 2, 0, 8)?
        }
        (FormatVersion::V3(_), PixelFormat::Yuv420P) => {
            build_v3_yuv(extra, v, width, height, 1, 1, 8)?
        }
        (FormatVersion::V3(_), PixelFormat::Yuv422P) => {
            build_v3_yuv(extra, v, width, height, 1, 0, 8)?
        }
        (FormatVersion::V3(_), PixelFormat::Yuv444P) => {
            build_v3_yuv(extra, v, width, height, 0, 0, 8)?
        }
        (FormatVersion::V3(_), PixelFormat::Yuv420P10Le) => {
            build_v3_yuv(extra, v, width, height, 1, 1, 10)?
        }
        (FormatVersion::V3(_), PixelFormat::Yuv422P10Le) => {
            build_v3_yuv(extra, v, width, height, 1, 0, 10)?
        }
        (FormatVersion::V3(_), PixelFormat::Yuv444P10Le) => {
            build_v3_yuv(extra, v, width, height, 0, 0, 10)?
        }
        (FormatVersion::V3(_), PixelFormat::Yuv420P12Le) => {
            build_v3_yuv(extra, v, width, height, 1, 1, 12)?
        }
        (FormatVersion::V3(_), PixelFormat::Yuv422P12Le) => {
            build_v3_yuv(extra, v, width, height, 1, 0, 12)?
        }
        (FormatVersion::V3(_), PixelFormat::Yuv444P12Le) => {
            build_v3_yuv(extra, v, width, height, 0, 0, 12)?
        }
        // GBRP / GBRAP planar — same plane-sequential path as YUV v3.
        (FormatVersion::V3(_), PixelFormat::Gbrp10Le) => {
            build_v3_gbrp(extra, v, width, height, 10, false)?
        }
        (FormatVersion::V3(_), PixelFormat::Gbrap10Le) => {
            build_v3_gbrp(extra, v, width, height, 10, true)?
        }
        (FormatVersion::V3(_), PixelFormat::Gbrp12Le) => {
            build_v3_gbrp(extra, v, width, height, 12, false)?
        }
        (FormatVersion::V3(_), PixelFormat::Gbrap12Le) => {
            build_v3_gbrp(extra, v, width, height, 12, true)?
        }
        _ => {
            return Err(Error::unsupported(format!(
                "huffyuv encode: unsupported (format, pixel_format) combo: {:?} / {:?}",
                extra.format, pixel_format
            )));
        }
    };

    // Per-table histograms derived from the (now order-correct)
    // residual stream.
    let mut hists: Vec<Vec<u64>> = vec![vec![0u64; alphabet]; n_tables];
    let huff_alpha = if bps >= 15 { 16384 } else { alphabet };
    for &(t_idx, r) in stream.iter() {
        let masked = r & mask;
        let sym = if bps >= 15 {
            (masked >> 2) as usize
        } else {
            masked as usize
        };
        hists[t_idx as usize][sym] += 1;
    }
    // Trim every histogram to the Huffman alphabet size before building
    // (for ≤14-bit alphabets `huff_alpha == alphabet`; for 15/16-bit
    // we cap at 16384).
    for h in hists.iter_mut() {
        h.truncate(huff_alpha);
    }
    type HuffTablePair = (Vec<u8>, Vec<(u32, u8)>);
    let mut tables: Vec<HuffTablePair> = Vec::with_capacity(n_tables);
    for hist in hists.iter() {
        let lens = build_lengths(hist, HUFF_MAX_LEN)?;
        let codes = HuffTable::build_codes(&lens)?;
        tables.push((lens, codes));
    }

    // ── byte-stream assembly ────────────────────────────────────────
    let mut bytes = Vec::with_capacity(estimate_packet_bytes(width, height, bps));

    if extra.per_frame_tables {
        for (lens, _) in tables.iter() {
            bytes.extend_from_slice(&encode_lengths(lens));
        }
    }

    // From here on the bitstream lives in a BitWriter that we'll merge
    // with `bytes` via append. The trace doc puts the prelude + Huffman
    // residuals inside the same MSB-first bit stream that follows the
    // RLE table block. To keep byte-level alignment correct, the per-
    // frame tables consume an integer number of bytes (RLE is byte-
    // aligned by construction), so the bit writer can start at the
    // current byte boundary.
    let mut bw = BitWriter::with_capacity(estimate_packet_bytes(width, height, bps));

    // Prelude (raw 8 bits for v2 layouts; no prelude for v3).
    for byte in prelude.iter() {
        bw.write_bits(8, *byte as u32);
    }

    // Emit residuals in stream order using the just-built per-table codes.
    for &(t_idx, r) in stream.iter() {
        let masked = r & mask;
        let (_lens, codes) = &tables[t_idx as usize];
        if bps >= 15 {
            let hi = masked >> 2;
            let lo = masked & 0x3;
            let (c, l) = codes[hi as usize];
            if l == 0 {
                return Err(Error::invalid(format!(
                    "huffyuv encode: histogram missed symbol {hi}"
                )));
            }
            bw.write_bits(l, c);
            bw.write_bits(2, lo);
        } else {
            let (c, l) = codes[masked as usize];
            if l == 0 {
                return Err(Error::invalid(format!(
                    "huffyuv encode: histogram missed symbol {masked}"
                )));
            }
            bw.write_bits(l, c);
        }
    }

    // 31-bit zero tail (16 + 15) — the decoder uses unchecked bit-reader
    // behaviour and walks past the last residual code to the next byte
    // boundary, which the zero tail safely satisfies.
    bw.write_bits(16, 0);
    bw.write_bits(15, 0);

    let mut bit_bytes = bw.finish();
    bytes.append(&mut bit_bytes);

    // Pad to a multiple of 4 bytes (32-bit alignment).
    while bytes.len() % 4 != 0 {
        bytes.push(0);
    }

    // Apply the 32-bit-word byteswap on the whole packet. The per-frame
    // RLE table block is part of the swapped payload — the decoder
    // un-swaps the entire packet first, then walks the tables.
    Ok(swap_payload(&bytes))
}

fn bps_alphabet(bps: u8) -> usize {
    let raw = 1usize << bps;
    raw.min(16384)
}

fn estimate_packet_bytes(width: usize, height: usize, bps: u8) -> usize {
    // Rough upper bound: 2x per-sample bytes is pessimistic but keeps
    // the BitWriter from reallocating mid-frame.
    let bytes_per_sample = if bps <= 8 { 1 } else { 2 };
    width * height * 3 * bytes_per_sample + 4096
}

// ───────────────────── per-format residual builders ─────────────────────

/// Compute LEFT predictor residuals for a single planar row.
/// Returns the new "left" register at row end. Residuals are pushed
/// into `out` (bitstream-order).
fn left_row_residuals_u32(samples: &[u32], mut left: u32, mask: u32, out: &mut Vec<u32>) -> u32 {
    for &s in samples.iter() {
        let r = s.wrapping_sub(left) & mask;
        out.push(r);
        left = s & mask;
    }
    left
}

/// GRADIENT residuals — canonical PLANE encoder mirror of
/// [`crate::predictor::pred_gradient_inplace_full`]. For each sample:
///
/// ```text
///     pred[x] = left + top[x] - top_left
///     r[x]    = sample[x] - pred[x]
/// ```
///
/// Both `left` and `top_left` thread across rows. Returns the
/// updated registers for the next row.
fn gradient_row_residuals_u32(
    samples: &[u32],
    top: &[u32],
    mut left: u32,
    mut top_left: u32,
    mask: u32,
    out: &mut Vec<u32>,
) -> (u32, u32) {
    debug_assert_eq!(samples.len(), top.len());
    for x in 0..samples.len() {
        let pred = (left.wrapping_add(top[x]).wrapping_sub(top_left)) & mask;
        let r = samples[x].wrapping_sub(pred) & mask;
        out.push(r);
        top_left = top[x] & mask;
        left = samples[x] & mask;
    }
    (left, top_left)
}

/// MEDIAN encoder mirror of [`crate::predictor::pred_median_inplace_full`].
/// Returns the trailing `(left, top_left)` registers so the caller
/// can thread `top_left` across rows (matches the upstream encoder's
/// per-frame-per-plane register-threading observed in cross-decode
/// tests).
fn median_row_residuals_u32(
    samples: &[u32],
    top: &[u32],
    mut left: u32,
    mut top_left: u32,
    mask: u32,
    out: &mut Vec<u32>,
) -> (u32, u32) {
    debug_assert_eq!(samples.len(), top.len());
    for x in 0..samples.len() {
        let l = left & mask;
        let t = top[x] & mask;
        let tl = top_left & mask;
        let c = l.wrapping_add(t).wrapping_sub(tl) & mask;
        let lo = l.min(t);
        let hi = l.max(t);
        let pred = hi.min(c.max(lo));
        let r = samples[x].wrapping_sub(pred) & mask;
        out.push(r);
        top_left = top[x] & mask;
        left = samples[x] & mask;
    }
    (left, top_left)
}

// ───────────── v2 yuv422 ─────────────

fn build_v2_yuv422(
    extra: &Extradata,
    v: &VideoFrame,
    width: usize,
    height: usize,
) -> Result<(Vec<TaggedResidual>, Vec<u8>)> {
    if v.planes.len() < 3 {
        return Err(Error::invalid("huffyuv encode v2 yuv422p: need 3 planes"));
    }
    let chroma_w = width / 2;
    let y = plane_to_u32_8(&v.planes[0], width, height)?;
    let u = plane_to_u32_8(&v.planes[1], chroma_w, height)?;
    let v_ = plane_to_u32_8(&v.planes[2], chroma_w, height)?;

    let mask: u32 = 0xFF;

    // Prelude: V0 Y1 U0 Y0 in on-disk order.
    let prelude = vec![v_[0] as u8, y[1] as u8, u[0] as u8, y[0] as u8];

    let mut stream: Vec<TaggedResidual> = Vec::with_capacity((width + 2 * chroma_w) * height);

    // Row 0 — residuals from chroma column 1 onward, in (Y,U,Y,V) order.
    let mut left_y = y[1] & mask;
    let mut left_u = u[0] & mask;
    let mut left_v = v_[0] & mask;
    // GRADIENT cross-row top_left registers (see decoder for matching impl).
    let mut top_left_y: u32 = 0;
    let mut top_left_u: u32 = 0;
    let mut top_left_v: u32 = 0;
    for cx in 1..chroma_w {
        let s_y0 = y[2 * cx];
        stream.push((0, s_y0.wrapping_sub(left_y) & mask));
        left_y = s_y0 & mask;
        let s_u = u[cx];
        stream.push((1, s_u.wrapping_sub(left_u) & mask));
        left_u = s_u & mask;
        let s_y1 = y[2 * cx + 1];
        stream.push((0, s_y1.wrapping_sub(left_y) & mask));
        left_y = s_y1 & mask;
        let s_v = v_[cx];
        stream.push((2, s_v.wrapping_sub(left_v) & mask));
        left_v = s_v & mask;
    }

    // Rows 1..H-1.
    for row in 1..height {
        let row_y_off = row * width;
        let row_c_off = row * chroma_w;
        let prev_y = &y[(row - 1) * width..row * width];
        let prev_u = &u[(row - 1) * chroma_w..row * chroma_w];
        let prev_v = &v_[(row - 1) * chroma_w..row * chroma_w];
        let cur_y = &y[row_y_off..row_y_off + width];
        let cur_u = &u[row_c_off..row_c_off + chroma_w];
        let cur_v = &v_[row_c_off..row_c_off + chroma_w];

        // Pre-build the row of residuals in the (Y,U,Y,V) interleaved
        // order required by the decoder, but to keep the predictor code
        // simple we compute residuals per-channel-plane, then interleave
        // them while pushing.
        let mut tmp_y = Vec::with_capacity(width);
        let mut tmp_u = Vec::with_capacity(chroma_w);
        let mut tmp_v = Vec::with_capacity(chroma_w);

        match extra.predictor {
            Predictor::Left => {
                left_y = left_row_residuals_u32(cur_y, left_y, mask, &mut tmp_y);
                left_u = left_row_residuals_u32(cur_u, left_u, mask, &mut tmp_u);
                left_v = left_row_residuals_u32(cur_v, left_v, mask, &mut tmp_v);
            }
            Predictor::Gradient => {
                let (ny, ntl) =
                    gradient_row_residuals_u32(cur_y, prev_y, left_y, top_left_y, mask, &mut tmp_y);
                left_y = ny;
                top_left_y = ntl;
                let (nu, ntu) =
                    gradient_row_residuals_u32(cur_u, prev_u, left_u, top_left_u, mask, &mut tmp_u);
                left_u = nu;
                top_left_u = ntu;
                let (nv, ntv) =
                    gradient_row_residuals_u32(cur_v, prev_v, left_v, top_left_v, mask, &mut tmp_v);
                left_v = nv;
                top_left_v = ntv;
            }
            Predictor::Median => {
                if row == 1 {
                    let bs_y = 4.min(width);
                    let bs_c = 2.min(chroma_w);
                    left_y = left_row_residuals_u32(&cur_y[..bs_y], left_y, mask, &mut tmp_y);
                    if bs_y < width {
                        let init_tl = prev_y[bs_y - 1];
                        let (nl, ntl) = median_row_residuals_u32(
                            &cur_y[bs_y..],
                            &prev_y[bs_y..],
                            left_y,
                            init_tl,
                            mask,
                            &mut tmp_y,
                        );
                        left_y = nl;
                        top_left_y = ntl;
                    }
                    left_u = left_row_residuals_u32(&cur_u[..bs_c], left_u, mask, &mut tmp_u);
                    if bs_c < chroma_w {
                        let init_tl = prev_u[bs_c - 1];
                        let (nl, ntl) = median_row_residuals_u32(
                            &cur_u[bs_c..],
                            &prev_u[bs_c..],
                            left_u,
                            init_tl,
                            mask,
                            &mut tmp_u,
                        );
                        left_u = nl;
                        top_left_u = ntl;
                    }
                    left_v = left_row_residuals_u32(&cur_v[..bs_c], left_v, mask, &mut tmp_v);
                    if bs_c < chroma_w {
                        let init_tl = prev_v[bs_c - 1];
                        let (nl, ntl) = median_row_residuals_u32(
                            &cur_v[bs_c..],
                            &prev_v[bs_c..],
                            left_v,
                            init_tl,
                            mask,
                            &mut tmp_v,
                        );
                        left_v = nl;
                        top_left_v = ntl;
                    }
                } else {
                    // top_left threads across rows (per the decoder
                    // convention); initial value carries from row 1's
                    // bootstrap result.
                    let (nl, ntl) = median_row_residuals_u32(
                        cur_y, prev_y, left_y, top_left_y, mask, &mut tmp_y,
                    );
                    left_y = nl;
                    top_left_y = ntl;
                    let (nl, ntl) = median_row_residuals_u32(
                        cur_u, prev_u, left_u, top_left_u, mask, &mut tmp_u,
                    );
                    left_u = nl;
                    top_left_u = ntl;
                    let (nl, ntl) = median_row_residuals_u32(
                        cur_v, prev_v, left_v, top_left_v, mask, &mut tmp_v,
                    );
                    left_v = nl;
                    top_left_v = ntl;
                }
            }
        }
        // Interleave (Y,U,Y,V) per chroma column into the output stream.
        for cx in 0..chroma_w {
            stream.push((0, tmp_y[2 * cx]));
            stream.push((1, tmp_u[cx]));
            stream.push((0, tmp_y[2 * cx + 1]));
            stream.push((2, tmp_v[cx]));
        }
    }

    Ok((stream, prelude))
}

// ───────────── v2 yuv420 ─────────────

fn build_v2_yuv420(
    extra: &Extradata,
    v: &VideoFrame,
    width: usize,
    height: usize,
) -> Result<(Vec<TaggedResidual>, Vec<u8>)> {
    if v.planes.len() < 3 {
        return Err(Error::invalid("huffyuv encode v2 yuv420p: need 3 planes"));
    }
    let chroma_w = width / 2;
    let chroma_h = height / 2;
    let y = plane_to_u32_8(&v.planes[0], width, height)?;
    let u = plane_to_u32_8(&v.planes[1], chroma_w, chroma_h)?;
    let v_ = plane_to_u32_8(&v.planes[2], chroma_w, chroma_h)?;
    let mask: u32 = 0xFF;

    let prelude = vec![v_[0] as u8, y[1] as u8, u[0] as u8, y[0] as u8];

    let mut stream: Vec<TaggedResidual> =
        Vec::with_capacity(width * height + 2 * chroma_w * chroma_h);

    // Row 0 (chroma row 0): finish the row from chroma column 1.
    let mut left_y = y[1] & mask;
    let mut left_u = u[0] & mask;
    let mut left_v = v_[0] & mask;
    for cx in 1..chroma_w {
        let s_y0 = y[2 * cx];
        stream.push((0, s_y0.wrapping_sub(left_y) & mask));
        left_y = s_y0 & mask;
        let s_u = u[cx];
        stream.push((1, s_u.wrapping_sub(left_u) & mask));
        left_u = s_u & mask;
        let s_y1 = y[2 * cx + 1];
        stream.push((0, s_y1.wrapping_sub(left_y) & mask));
        left_y = s_y1 & mask;
        let s_v = v_[cx];
        stream.push((2, s_v.wrapping_sub(left_v) & mask));
        left_v = s_v & mask;
    }

    let mut cy: usize = 1;
    let mut top_left_y: u32 = 0;
    let mut top_left_u: u32 = 0;
    let mut top_left_v: u32 = 0;
    for row in 1..height {
        let do_chroma = (row % 2) == 0;
        let prev_y = &y[(row - 1) * width..row * width];
        let cur_y = &y[row * width..(row + 1) * width];
        let mut tmp_y = Vec::with_capacity(width);
        let mut tmp_u = Vec::with_capacity(chroma_w);
        let mut tmp_v = Vec::with_capacity(chroma_w);

        let (cur_u, cur_v, prev_u, prev_v) = if do_chroma {
            let prev_c_off = (cy - 1) * chroma_w;
            (
                &u[cy * chroma_w..(cy + 1) * chroma_w],
                &v_[cy * chroma_w..(cy + 1) * chroma_w],
                &u[prev_c_off..prev_c_off + chroma_w],
                &v_[prev_c_off..prev_c_off + chroma_w],
            )
        } else {
            (&[][..], &[][..], &[][..], &[][..])
        };

        match extra.predictor {
            Predictor::Left => {
                left_y = left_row_residuals_u32(cur_y, left_y, mask, &mut tmp_y);
                if do_chroma {
                    left_u = left_row_residuals_u32(cur_u, left_u, mask, &mut tmp_u);
                    left_v = left_row_residuals_u32(cur_v, left_v, mask, &mut tmp_v);
                }
            }
            Predictor::Gradient => {
                let (ny, ntl) =
                    gradient_row_residuals_u32(cur_y, prev_y, left_y, top_left_y, mask, &mut tmp_y);
                left_y = ny;
                top_left_y = ntl;
                if do_chroma {
                    let (nu, ntu) = gradient_row_residuals_u32(
                        cur_u, prev_u, left_u, top_left_u, mask, &mut tmp_u,
                    );
                    left_u = nu;
                    top_left_u = ntu;
                    let (nv, ntv) = gradient_row_residuals_u32(
                        cur_v, prev_v, left_v, top_left_v, mask, &mut tmp_v,
                    );
                    left_v = nv;
                    top_left_v = ntv;
                }
            }
            Predictor::Median => {
                if row == 1 {
                    let bs_y = 4.min(width);
                    left_y = left_row_residuals_u32(&cur_y[..bs_y], left_y, mask, &mut tmp_y);
                    if bs_y < width {
                        let init_tl = prev_y[bs_y - 1];
                        let (nl, ntl) = median_row_residuals_u32(
                            &cur_y[bs_y..],
                            &prev_y[bs_y..],
                            left_y,
                            init_tl,
                            mask,
                            &mut tmp_y,
                        );
                        left_y = nl;
                        top_left_y = ntl;
                    }
                } else {
                    let (nl, ntl) = median_row_residuals_u32(
                        cur_y, prev_y, left_y, top_left_y, mask, &mut tmp_y,
                    );
                    left_y = nl;
                    top_left_y = ntl;
                }
                if do_chroma {
                    if cy == 1 {
                        let bs_c = 2.min(chroma_w);
                        left_u = left_row_residuals_u32(&cur_u[..bs_c], left_u, mask, &mut tmp_u);
                        if bs_c < chroma_w {
                            let init_tl = prev_u[bs_c - 1];
                            let (nl, ntl) = median_row_residuals_u32(
                                &cur_u[bs_c..],
                                &prev_u[bs_c..],
                                left_u,
                                init_tl,
                                mask,
                                &mut tmp_u,
                            );
                            left_u = nl;
                            top_left_u = ntl;
                        }
                        left_v = left_row_residuals_u32(&cur_v[..bs_c], left_v, mask, &mut tmp_v);
                        if bs_c < chroma_w {
                            let init_tl = prev_v[bs_c - 1];
                            let (nl, ntl) = median_row_residuals_u32(
                                &cur_v[bs_c..],
                                &prev_v[bs_c..],
                                left_v,
                                init_tl,
                                mask,
                                &mut tmp_v,
                            );
                            left_v = nl;
                            top_left_v = ntl;
                        }
                    } else {
                        let (nl, ntl) = median_row_residuals_u32(
                            cur_u, prev_u, left_u, top_left_u, mask, &mut tmp_u,
                        );
                        left_u = nl;
                        top_left_u = ntl;
                        let (nl, ntl) = median_row_residuals_u32(
                            cur_v, prev_v, left_v, top_left_v, mask, &mut tmp_v,
                        );
                        left_v = nl;
                        top_left_v = ntl;
                    }
                }
            }
        }
        if do_chroma {
            for cx in 0..chroma_w {
                stream.push((0, tmp_y[2 * cx]));
                stream.push((1, tmp_u[cx]));
                stream.push((0, tmp_y[2 * cx + 1]));
                stream.push((2, tmp_v[cx]));
            }
            cy += 1;
        } else {
            for r in tmp_y.iter() {
                stream.push((0, *r));
            }
        }
    }
    Ok((stream, prelude))
}

// ───────────── v2 RGB ─────────────

fn build_v2_rgb(
    extra: &Extradata,
    v: &VideoFrame,
    width: usize,
    height: usize,
    has_alpha: bool,
) -> Result<(Vec<TaggedResidual>, Vec<u8>)> {
    let channels = if has_alpha { 4 } else { 3 };
    if v.planes.is_empty() {
        return Err(Error::invalid("huffyuv encode v2 rgb: need 1 packed plane"));
    }
    let plane = &v.planes[0];
    let stride = plane.stride;
    if plane.data.len() < stride * height {
        return Err(Error::invalid(
            "huffyuv encode v2 rgb: packed plane too small for declared height",
        ));
    }
    if stride < width * channels {
        return Err(Error::invalid(
            "huffyuv encode v2 rgb: packed plane stride < width*channels",
        ));
    }
    let mask: u32 = 0xFF;

    // Reorder packed input into "row-major BGRA-on-disk, bottom-up" the
    // way the bitstream wants. Source packed format is RGB24 (R,G,B) for
    // !has_alpha, BGRA for has_alpha. We store as BGR/BGRA per pixel,
    // bottom-up (row 0 = bottom of displayed frame).
    let mut bottom_up: Vec<u8> = vec![0; width * height * channels];
    for y in 0..height {
        let src_off = y * stride;
        let dst_off = (height - 1 - y) * width * channels;
        for x in 0..width {
            let so = src_off + x * channels;
            let dop = dst_off + x * channels;
            if has_alpha {
                // Source already BGRA (matches PixelFormat::Bgra layout).
                bottom_up[dop] = plane.data[so];
                bottom_up[dop + 1] = plane.data[so + 1];
                bottom_up[dop + 2] = plane.data[so + 2];
                bottom_up[dop + 3] = plane.data[so + 3];
            } else {
                // Source RGB, on-disk BGR.
                bottom_up[dop] = plane.data[so + 2];
                bottom_up[dop + 1] = plane.data[so + 1];
                bottom_up[dop + 2] = plane.data[so];
            }
        }
    }

    // Prelude — read row 0 of the bottom-up buffer.
    let p_off = 0;
    let prelude = if has_alpha {
        vec![
            bottom_up[p_off + 3], // A
            bottom_up[p_off + 2], // R
            bottom_up[p_off + 1], // G
            bottom_up[p_off],     // B
        ]
    } else {
        vec![
            bottom_up[p_off + 2], // R
            bottom_up[p_off + 1], // G
            bottom_up[p_off],     // B
            0,                    // pad
        ]
    };

    // Residual stream — interleaved per pixel in (B,G,R[,A]) order.
    // Tables:  0 = B, 1 = G, 2 = R (= A, v2 RGBA re-uses table 2 for
    // alpha per the trace doc).
    let mut stream: Vec<TaggedResidual> = Vec::with_capacity(width * height * channels);

    // On-disk per-pixel layout is (B, G, R[, A]) regardless of has_alpha.
    let mut left_b = bottom_up[0] as u32;
    let mut left_g = bottom_up[1] as u32;
    let mut left_r = bottom_up[2] as u32;
    let mut left_a = if has_alpha { bottom_up[3] as u32 } else { 0 };
    // GRADIENT cross-row top_left registers.
    let mut top_left_b: u32 = 0;
    let mut top_left_g: u32 = 0;
    let mut top_left_r: u32 = 0;
    let mut top_left_a: u32 = 0;

    // Helper to fetch the on-disk per-pixel byte.
    let fetch = |buf: &[u8], y: usize, x: usize, c: usize| -> u32 {
        buf[y * width * channels + x * channels + c] as u32
    };

    // Row 0 (already used col 0 for prelude) — write residuals from x=1.
    for x in 1..width {
        let b = fetch(&bottom_up, 0, x, 0);
        let g = fetch(&bottom_up, 0, x, 1);
        let r = fetch(&bottom_up, 0, x, 2);
        let a = if has_alpha {
            fetch(&bottom_up, 0, x, 3)
        } else {
            0
        };

        let g_res = g.wrapping_sub(left_g) & mask;
        if extra.decorrelate {
            let b_res = b.wrapping_sub(g) & mask;
            let r_res = r.wrapping_sub(g) & mask;
            stream.push((0, b_res));
            stream.push((1, g_res));
            stream.push((2, r_res));
        } else {
            let b_res = b.wrapping_sub(left_b) & mask;
            let r_res = r.wrapping_sub(left_r) & mask;
            stream.push((0, b_res));
            stream.push((1, g_res));
            stream.push((2, r_res));
        }
        if has_alpha {
            let a_res = a.wrapping_sub(left_a) & mask;
            stream.push((2, a_res));
            left_a = a;
        }
        left_g = g;
        left_b = b;
        left_r = r;
    }

    for y in 1..height {
        // Residuals depend on the predictor choice and decorrelate flag.
        // Build per-channel arrays first, then interleave.
        let cur_b: Vec<u32> = (0..width).map(|x| fetch(&bottom_up, y, x, 0)).collect();
        let cur_g: Vec<u32> = (0..width).map(|x| fetch(&bottom_up, y, x, 1)).collect();
        let cur_r: Vec<u32> = (0..width).map(|x| fetch(&bottom_up, y, x, 2)).collect();
        let cur_a: Vec<u32> = if has_alpha {
            (0..width).map(|x| fetch(&bottom_up, y, x, 3)).collect()
        } else {
            Vec::new()
        };
        let prev_b: Vec<u32> = (0..width).map(|x| fetch(&bottom_up, y - 1, x, 0)).collect();
        let prev_g: Vec<u32> = (0..width).map(|x| fetch(&bottom_up, y - 1, x, 1)).collect();
        let prev_r: Vec<u32> = (0..width).map(|x| fetch(&bottom_up, y - 1, x, 2)).collect();
        let prev_a: Vec<u32> = if has_alpha {
            (0..width).map(|x| fetch(&bottom_up, y - 1, x, 3)).collect()
        } else {
            Vec::new()
        };

        let mut tmp_b = Vec::with_capacity(width);
        let mut tmp_g = Vec::with_capacity(width);
        let mut tmp_r = Vec::with_capacity(width);
        let mut tmp_a = Vec::with_capacity(width);

        match extra.predictor {
            Predictor::Left => {
                if extra.decorrelate {
                    left_g = left_row_residuals_u32(&cur_g, left_g, mask, &mut tmp_g);
                    for x in 0..width {
                        tmp_b.push(cur_b[x].wrapping_sub(cur_g[x]) & mask);
                        tmp_r.push(cur_r[x].wrapping_sub(cur_g[x]) & mask);
                    }
                    left_b = cur_b[width - 1];
                    left_r = cur_r[width - 1];
                    if has_alpha {
                        left_a = left_row_residuals_u32(&cur_a, left_a, mask, &mut tmp_a);
                    }
                } else {
                    left_b = left_row_residuals_u32(&cur_b, left_b, mask, &mut tmp_b);
                    left_g = left_row_residuals_u32(&cur_g, left_g, mask, &mut tmp_g);
                    left_r = left_row_residuals_u32(&cur_r, left_r, mask, &mut tmp_r);
                    if has_alpha {
                        left_a = left_row_residuals_u32(&cur_a, left_a, mask, &mut tmp_a);
                    }
                }
            }
            Predictor::Gradient => {
                if extra.decorrelate {
                    let (ng, ntg) = gradient_row_residuals_u32(
                        &cur_g, &prev_g, left_g, top_left_g, mask, &mut tmp_g,
                    );
                    left_g = ng;
                    top_left_g = ntg;
                    for x in 0..width {
                        tmp_b.push(cur_b[x].wrapping_sub(cur_g[x]) & mask);
                        tmp_r.push(cur_r[x].wrapping_sub(cur_g[x]) & mask);
                    }
                    left_b = cur_b[width - 1];
                    left_r = cur_r[width - 1];
                    if has_alpha {
                        let (na, nta) = gradient_row_residuals_u32(
                            &cur_a, &prev_a, left_a, top_left_a, mask, &mut tmp_a,
                        );
                        left_a = na;
                        top_left_a = nta;
                    }
                } else {
                    let (nb, ntb) = gradient_row_residuals_u32(
                        &cur_b, &prev_b, left_b, top_left_b, mask, &mut tmp_b,
                    );
                    left_b = nb;
                    top_left_b = ntb;
                    let (ng, ntg) = gradient_row_residuals_u32(
                        &cur_g, &prev_g, left_g, top_left_g, mask, &mut tmp_g,
                    );
                    left_g = ng;
                    top_left_g = ntg;
                    let (nr, ntr) = gradient_row_residuals_u32(
                        &cur_r, &prev_r, left_r, top_left_r, mask, &mut tmp_r,
                    );
                    left_r = nr;
                    top_left_r = ntr;
                    if has_alpha {
                        let (na, nta) = gradient_row_residuals_u32(
                            &cur_a, &prev_a, left_a, top_left_a, mask, &mut tmp_a,
                        );
                        left_a = na;
                        top_left_a = nta;
                    }
                }
            }
            Predictor::Median => {
                return Err(Error::invalid(
                    "huffyuv encode v2 RGB: MEDIAN predictor is rejected (matches upstream)",
                ));
            }
        }
        for x in 0..width {
            stream.push((0, tmp_b[x]));
            stream.push((1, tmp_g[x]));
            stream.push((2, tmp_r[x]));
            if has_alpha {
                stream.push((2, tmp_a[x]));
            }
        }
    }
    Ok((stream, prelude))
}

// ───────────── v3 single-plane / yuv ─────────────

fn build_v3_gray(
    extra: &Extradata,
    v: &VideoFrame,
    width: usize,
    height: usize,
    bps: u8,
) -> Result<(Vec<TaggedResidual>, Vec<u8>)> {
    if v.planes.is_empty() {
        return Err(Error::invalid("huffyuv encode v3 gray: need 1 plane"));
    }
    let samples = if bps <= 8 {
        plane_to_u32_8(&v.planes[0], width, height)?
    } else {
        plane_to_u32_16(&v.planes[0], width, height)?
    };
    let mut residuals: Vec<u32> = Vec::with_capacity(width * height);
    encode_plane_residuals(extra, &samples, width, height, bps, &mut residuals);
    let stream = residuals.into_iter().map(|r| (0u8, r)).collect();
    Ok((stream, Vec::new()))
}

fn build_v3_yuv(
    extra: &Extradata,
    v: &VideoFrame,
    width: usize,
    height: usize,
    h_shift: u8,
    v_shift: u8,
    bps: u8,
) -> Result<(Vec<TaggedResidual>, Vec<u8>)> {
    if v.planes.len() < 3 {
        return Err(Error::invalid("huffyuv encode v3 yuv: need 3 planes"));
    }
    let chroma_w = width >> h_shift;
    let chroma_h = height >> v_shift;
    let (yp, up, vp) = if bps <= 8 {
        (
            plane_to_u32_8(&v.planes[0], width, height)?,
            plane_to_u32_8(&v.planes[1], chroma_w, chroma_h)?,
            plane_to_u32_8(&v.planes[2], chroma_w, chroma_h)?,
        )
    } else {
        (
            plane_to_u32_16(&v.planes[0], width, height)?,
            plane_to_u32_16(&v.planes[1], chroma_w, chroma_h)?,
            plane_to_u32_16(&v.planes[2], chroma_w, chroma_h)?,
        )
    };

    let mut res_y: Vec<u32> = Vec::with_capacity(width * height);
    let mut res_u: Vec<u32> = Vec::with_capacity(chroma_w * chroma_h);
    let mut res_v: Vec<u32> = Vec::with_capacity(chroma_w * chroma_h);
    encode_plane_residuals(extra, &yp, width, height, bps, &mut res_y);
    encode_plane_residuals(extra, &up, chroma_w, chroma_h, bps, &mut res_u);
    encode_plane_residuals(extra, &vp, chroma_w, chroma_h, bps, &mut res_v);
    let mut stream: Vec<TaggedResidual> =
        Vec::with_capacity(res_y.len() + res_u.len() + res_v.len());
    for r in res_y {
        stream.push((0, r));
    }
    for r in res_u {
        stream.push((1, r));
    }
    for r in res_v {
        stream.push((2, r));
    }
    Ok((stream, Vec::new()))
}

// ───────────── v3 GBRP / GBRAP ─────────────

fn build_v3_gbrp(
    extra: &Extradata,
    v: &VideoFrame,
    width: usize,
    height: usize,
    bps: u8,
    has_alpha: bool,
) -> Result<(Vec<TaggedResidual>, Vec<u8>)> {
    let n_planes = if has_alpha { 4 } else { 3 };
    if v.planes.len() < n_planes {
        return Err(Error::invalid(format!(
            "huffyuv encode v3 gbrp: need {n_planes} planes, got {}",
            v.planes.len()
        )));
    }
    // Planes are stored in G→B→R[→A] order matching the PixelFormat.
    let mut stream: Vec<TaggedResidual> = Vec::with_capacity(width * height * n_planes);
    for p in 0..n_planes {
        let samples = plane_to_u32_16(&v.planes[p], width, height)?;
        let mut residuals: Vec<u32> = Vec::with_capacity(width * height);
        encode_plane_residuals(extra, &samples, width, height, bps, &mut residuals);
        for r in residuals {
            stream.push((p as u8, r));
        }
    }
    Ok((stream, Vec::new()))
}

fn encode_plane_residuals(
    extra: &Extradata,
    samples: &[u32],
    width: usize,
    height: usize,
    bps: u8,
    out: &mut Vec<u32>,
) {
    let mask: u32 = if bps >= 32 {
        u32::MAX
    } else {
        (1u32 << bps) - 1
    };
    if width == 0 || height == 0 {
        return;
    }
    // Row 0: fully LEFT-predicted with initial left=0.
    let row0 = &samples[..width];
    let mut left = left_row_residuals_u32(row0, 0, mask, out);
    let mut top_left: u32 = 0;
    for row in 1..height {
        let prev = &samples[(row - 1) * width..row * width];
        let cur = &samples[row * width..(row + 1) * width];
        match extra.predictor {
            Predictor::Left => {
                left = left_row_residuals_u32(cur, left, mask, out);
            }
            Predictor::Gradient => {
                let (nl, ntl) = gradient_row_residuals_u32(cur, prev, left, top_left, mask, out);
                left = nl;
                top_left = ntl;
            }
            Predictor::Median => {
                let (nl, ntl) = median_row_residuals_u32(cur, prev, left, top_left, mask, out);
                left = nl;
                top_left = ntl;
            }
        }
    }
}

// ──────────────────── plane → u32 sample helpers ────────────────────

fn plane_to_u32_8(plane: &VideoPlane, width: usize, height: usize) -> Result<Vec<u32>> {
    if plane.data.len() < plane.stride * height {
        return Err(Error::invalid(format!(
            "huffyuv encode: plane data len {} < stride {} * height {}",
            plane.data.len(),
            plane.stride,
            height
        )));
    }
    if plane.stride < width {
        return Err(Error::invalid(format!(
            "huffyuv encode: plane stride {} < width {width}",
            plane.stride
        )));
    }
    let mut out = Vec::with_capacity(width * height);
    for y in 0..height {
        let off = y * plane.stride;
        for x in 0..width {
            out.push(plane.data[off + x] as u32);
        }
    }
    Ok(out)
}

fn plane_to_u32_16(plane: &VideoPlane, width: usize, height: usize) -> Result<Vec<u32>> {
    // 16-bit-packed plane stored as u8 little-endian pairs. Stride is in
    // bytes (so plane.stride = width * 2 for tightly packed planes).
    let bytes_per_row = width * 2;
    if plane.stride < bytes_per_row {
        return Err(Error::invalid(format!(
            "huffyuv encode: 16-bit plane stride {} < {bytes_per_row}",
            plane.stride
        )));
    }
    if plane.data.len() < plane.stride * height {
        return Err(Error::invalid(format!(
            "huffyuv encode: 16-bit plane data len {} < stride * height",
            plane.data.len()
        )));
    }
    let mut out = Vec::with_capacity(width * height);
    for y in 0..height {
        let off = y * plane.stride;
        for x in 0..width {
            let lo = plane.data[off + x * 2] as u32;
            let hi = plane.data[off + x * 2 + 1] as u32;
            out.push(lo | (hi << 8));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::frame::VideoPlane;
    use oxideav_core::time::TimeBase;
    use oxideav_core::{CodecId, CodecParameters, CodecRegistry, Frame, PixelFormat, VideoFrame};

    fn frame_yuv422(w: usize, h: usize, seed: u32) -> VideoFrame {
        let cw = w / 2;
        let mut y = vec![0u8; w * h];
        let mut u = vec![0u8; cw * h];
        let mut v = vec![0u8; cw * h];
        for i in 0..(w * h) {
            y[i] = ((seed.wrapping_mul(31).wrapping_add(i as u32)) & 0xFF) as u8;
        }
        for i in 0..(cw * h) {
            u[i] = ((seed.wrapping_mul(17).wrapping_add(i as u32 * 3)) & 0xFF) as u8;
            v[i] = ((seed.wrapping_mul(19).wrapping_add(i as u32 * 5)) & 0xFF) as u8;
        }
        VideoFrame {
            pts: Some(0),
            planes: vec![
                VideoPlane { stride: w, data: y },
                VideoPlane {
                    stride: cw,
                    data: u,
                },
                VideoPlane {
                    stride: cw,
                    data: v,
                },
            ],
        }
    }

    #[test]
    fn round_trip_v2_yuv422_left_8x4() {
        let mut reg = CodecRegistry::new();
        crate::register(&mut reg);
        let mut params = CodecParameters::video(CodecId::new(crate::CODEC_ID_HUFFYUV));
        params.width = Some(8);
        params.height = Some(4);
        params.pixel_format = Some(PixelFormat::Yuv422P);
        let mut enc = reg.make_encoder(&params).expect("encoder");
        let original = frame_yuv422(8, 4, 0xDEAD);
        enc.send_frame(&Frame::Video(original.clone())).unwrap();
        let pkt = enc.receive_packet().unwrap();
        let dec_extra = enc.output_params().extradata.clone();

        let mut dec_params = CodecParameters::video(CodecId::new(crate::CODEC_ID_HUFFYUV));
        dec_params.width = Some(8);
        dec_params.height = Some(4);
        dec_params.pixel_format = Some(PixelFormat::Yuv422P);
        dec_params.extradata = dec_extra;
        let mut dec = reg.make_decoder(&dec_params).expect("decoder");
        let pkt2 = Packet::new(0, TimeBase::new(1, 25), pkt.data.clone()).with_keyframe(true);
        dec.send_packet(&pkt2).unwrap();
        let frame = dec.receive_frame().unwrap();
        let v = match frame {
            Frame::Video(v) => v,
            _ => panic!(),
        };
        for y in 0..4 {
            for x in 0..8 {
                assert_eq!(
                    v.planes[0].data[y * v.planes[0].stride + x],
                    original.planes[0].data[y * original.planes[0].stride + x],
                    "Y[{y}][{x}]"
                );
            }
            for cx in 0..4 {
                assert_eq!(
                    v.planes[1].data[y * v.planes[1].stride + cx],
                    original.planes[1].data[y * original.planes[1].stride + cx],
                    "U[{y}][{cx}]"
                );
                assert_eq!(
                    v.planes[2].data[y * v.planes[2].stride + cx],
                    original.planes[2].data[y * original.planes[2].stride + cx],
                    "V[{y}][{cx}]"
                );
            }
        }
    }

    #[test]
    fn round_trip_v2_yuv422_gradient_8x4() {
        let mut reg = CodecRegistry::new();
        crate::register(&mut reg);
        let mut params = CodecParameters::video(CodecId::new(crate::CODEC_ID_HUFFYUV));
        params.width = Some(8);
        params.height = Some(4);
        params.pixel_format = Some(PixelFormat::Yuv422P);
        params.options.insert("predictor", "gradient");
        let mut enc = reg.make_encoder(&params).expect("encoder");
        let original = frame_yuv422(8, 4, 0xBEEF);
        enc.send_frame(&Frame::Video(original.clone())).unwrap();
        let pkt = enc.receive_packet().unwrap();
        let dec_extra = enc.output_params().extradata.clone();

        let mut dec_params = CodecParameters::video(CodecId::new(crate::CODEC_ID_HUFFYUV));
        dec_params.width = Some(8);
        dec_params.height = Some(4);
        dec_params.pixel_format = Some(PixelFormat::Yuv422P);
        dec_params.extradata = dec_extra;
        let mut dec = reg.make_decoder(&dec_params).expect("decoder");
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 25), pkt.data).with_keyframe(true))
            .unwrap();
        let frame = dec.receive_frame().unwrap();
        let v = match frame {
            Frame::Video(v) => v,
            _ => panic!(),
        };
        for y in 0..4 {
            assert_eq!(
                &v.planes[0].data[y * 8..y * 8 + 8],
                &original.planes[0].data[y * 8..y * 8 + 8]
            );
        }
    }

    #[test]
    fn round_trip_v3_gray8() {
        let mut reg = CodecRegistry::new();
        crate::register(&mut reg);
        let mut params = CodecParameters::video(CodecId::new(crate::CODEC_ID_FFVHUFF));
        params.width = Some(16);
        params.height = Some(8);
        params.pixel_format = Some(PixelFormat::Gray8);
        let mut enc = reg.make_encoder(&params).expect("encoder");
        let mut data = vec![0u8; 16 * 8];
        for (i, b) in data.iter_mut().enumerate() {
            *b = ((i * 7) & 0xFF) as u8;
        }
        let f = VideoFrame {
            pts: Some(0),
            planes: vec![VideoPlane {
                stride: 16,
                data: data.clone(),
            }],
        };
        enc.send_frame(&Frame::Video(f)).unwrap();
        let pkt = enc.receive_packet().unwrap();
        let mut dec_params = CodecParameters::video(CodecId::new(crate::CODEC_ID_FFVHUFF));
        dec_params.width = Some(16);
        dec_params.height = Some(8);
        dec_params.pixel_format = Some(PixelFormat::Gray8);
        dec_params.extradata = enc.output_params().extradata.clone();
        let mut dec = reg.make_decoder(&dec_params).unwrap();
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 25), pkt.data).with_keyframe(true))
            .unwrap();
        let v = match dec.receive_frame().unwrap() {
            Frame::Video(v) => v,
            _ => panic!(),
        };
        assert_eq!(v.planes[0].data, data);
    }
}
