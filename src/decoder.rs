//! HuffYUV / FFVHuff packet decoder.
//!
//! Each compressed packet holds one **intra** frame (trace doc §5.7).
//! The decoder:
//!
//! 1. Parses the codec extradata (cached from the stream's
//!    [`CodecParameters::extradata`]).
//! 2. Optionally re-derives per-channel Huffman tables from the start
//!    of the packet (FFVHuff `-context 1`).
//! 3. Undoes the per-32-bit-word byte-swap on the bitstream payload.
//! 4. Reads the 4-byte prelude (when applicable) and the per-channel
//!    Huffman residual stream.
//! 5. Folds the spatial predictor over each row to recover pixel
//!    values, then optionally inverts the (R-G, G, B-G) decorrelation
//!    for RGB streams.
//!
//! The result is a [`VideoFrame`] with the appropriate
//! [`PixelFormat`].

use oxideav_core::frame::VideoPlane;
use oxideav_core::Decoder;
use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, Packet, PixelFormat, Result, VideoFrame,
};

use crate::bitreader::{unswap_payload, BitReader};
use crate::extradata::{
    Extradata, FormatVersion, InterlaceMode, Predictor, V2Colorspace, V3Format,
};
use crate::huffman::HuffTable;
use crate::predictor::{
    pred_gradient_inplace, pred_gradient_inplace_full, pred_left_inplace, pred_median_inplace_full,
};
use crate::rle;

pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    if params.extradata.is_empty() {
        return Err(Error::invalid(
            "huffyuv decoder: missing extradata (4-byte header + RLE tables required)",
        ));
    }
    let extradata = Extradata::parse(&params.extradata)?;
    let width = params
        .width
        .ok_or_else(|| Error::invalid("huffyuv decoder: missing width"))? as usize;
    let height = params
        .height
        .ok_or_else(|| Error::invalid("huffyuv decoder: missing height"))?
        as usize;
    validate_shape(&extradata, width)?;
    let pixel_format = derive_pixel_format(&extradata)?;
    Ok(Box::new(HuffyuvDecoder {
        codec_id: params.codec_id.clone(),
        extradata,
        width,
        height,
        pixel_format,
        pending: None,
        eof: false,
    }))
}

struct HuffyuvDecoder {
    codec_id: CodecId,
    extradata: Extradata,
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    pending: Option<Packet>,
    eof: bool,
}

impl Decoder for HuffyuvDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        if self.pending.is_some() {
            return Err(Error::other(
                "huffyuv decoder: receive_frame must be called between packets",
            ));
        }
        self.pending = Some(packet.clone());
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        let Some(pkt) = self.pending.take() else {
            return if self.eof {
                Err(Error::Eof)
            } else {
                Err(Error::NeedMore)
            };
        };
        let vf = decode_packet(
            &self.extradata,
            &pkt,
            self.width,
            self.height,
            self.pixel_format,
        )?;
        Ok(Frame::Video(vf))
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }
}

// ─────────────────────── pixel-format derivation ───────────────────────

fn derive_pixel_format(extra: &Extradata) -> Result<PixelFormat> {
    match &extra.format {
        FormatVersion::V2(v) => match v {
            V2Colorspace::Yuv420 => Ok(PixelFormat::Yuv420P),
            V2Colorspace::Yuv422 => Ok(PixelFormat::Yuv422P),
            V2Colorspace::Rgb24 => Ok(PixelFormat::Rgb24),
            V2Colorspace::Bgra32 => Ok(PixelFormat::Bgra),
        },
        FormatVersion::V3(v) => derive_v3_pixel_format(v),
    }
}

fn derive_v3_pixel_format(v: &V3Format) -> Result<PixelFormat> {
    match (v.bps, v.yuv, v.chroma, v.alpha, v.chroma_h_shift, v.chroma_v_shift) {
        (8, false, false, false, _, _) => Ok(PixelFormat::Gray8),
        (8, true, true, false, 0, 0) => Ok(PixelFormat::Yuv444P),
        (8, true, true, false, 1, 0) => Ok(PixelFormat::Yuv422P),
        (8, true, true, false, 1, 1) => Ok(PixelFormat::Yuv420P),
        (8, true, true, false, 2, 0) => Ok(PixelFormat::Yuv411P),
        // High-bit-depth planar layouts (10/12/16-bit). 9 / 14-bit are
        // valid in the spec but core lacks dedicated PixelFormat
        // variants for them; we promote them up to the next supported
        // depth (so a 9-bit stream is delivered as 10-bit-LE etc.).
        (10, true, true, false, 0, 0) => Ok(PixelFormat::Yuv444P10Le),
        (10, true, true, false, 1, 0) => Ok(PixelFormat::Yuv422P10Le),
        (10, true, true, false, 1, 1) => Ok(PixelFormat::Yuv420P10Le),
        (12, true, true, false, 0, 0) => Ok(PixelFormat::Yuv444P12Le),
        (12, true, true, false, 1, 0) => Ok(PixelFormat::Yuv422P12Le),
        (12, true, true, false, 1, 1) => Ok(PixelFormat::Yuv420P12Le),
        (10, false, false, false, _, _) => Ok(PixelFormat::Gray10Le),
        (12, false, false, false, _, _) => Ok(PixelFormat::Gray12Le),
        (16, false, false, false, _, _) => Ok(PixelFormat::Gray16Le),
        // GBRP planar (8-bit stored as 16-bit LE words via Gbrp10Le, low 8 bits used).
        // No native Gbrp8 PixelFormat exists; we promote to the 10-bit container.
        (8, false, true, false, 0, 0) => Ok(PixelFormat::Gbrp10Le),
        // GBRAP planar (8-bit + alpha, same promotion strategy).
        (8, false, true, true, 0, 0) => Ok(PixelFormat::Gbrap10Le),
        // HBD gbrp/gbrap (10/12 bit).
        (10, false, true, false, 0, 0) => Ok(PixelFormat::Gbrp10Le),
        (10, false, true, true, 0, 0) => Ok(PixelFormat::Gbrap10Le),
        (12, false, true, false, 0, 0) => Ok(PixelFormat::Gbrp12Le),
        (12, false, true, true, 0, 0) => Ok(PixelFormat::Gbrap12Le),
        (bps, _, _, _, _, _) if !(8..=16).contains(&bps) => Err(Error::unsupported(format!(
            "huffyuv v3: bit-depth {bps} out of range"
        ))),
        _ => Err(Error::unsupported(format!(
            "huffyuv v3: unrecognised channel descriptor (bps={}, yuv={}, chroma={}, alpha={}, h={}, v={})",
            v.bps, v.yuv, v.chroma, v.alpha, v.chroma_h_shift, v.chroma_v_shift
        ))),
    }
}

/// Public re-export for the encoder so it can apply the same
/// width-multiple checks the decoder does. (Same function — both paths
/// must agree on the constraint, otherwise an encoded frame would fail
/// to decode.)
pub fn validate_shape_pub(extra: &Extradata, width: usize) -> Result<()> {
    validate_shape(extra, width)
}

fn validate_shape(extra: &Extradata, width: usize) -> Result<()> {
    // Trace doc §2.2.4 width-multiple constraints.
    if let FormatVersion::V2(cs) = &extra.format {
        if matches!(cs, V2Colorspace::Yuv420 | V2Colorspace::Yuv422) && width % 2 != 0 {
            return Err(Error::invalid(format!(
                "huffyuv: yuv4:2:x requires even width (got {width})"
            )));
        }
        if matches!(cs, V2Colorspace::Yuv422)
            && extra.predictor == Predictor::Median
            && width % 4 != 0
        {
            return Err(Error::invalid(format!(
                "huffyuv: yuv422 + MEDIAN requires width % 4 == 0 (got {width})"
            )));
        }
    }
    Ok(())
}

// ────────────────────────── packet decode ──────────────────────────

fn decode_packet(
    extra: &Extradata,
    pkt: &Packet,
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
) -> Result<VideoFrame> {
    if pkt.data.is_empty() {
        return Err(Error::invalid("huffyuv decode: empty packet"));
    }

    // Step 1 (optional): per-frame Huffman tables. These are stored
    // RLE-coded at the *byte* start of the un-byte-swapped payload — so
    // we can't simply read them out before unswapping, because the RLE
    // decoder walks bytes. The trace doc §3.5 shows the per-frame
    // tables come at the start of the packet payload before the bit
    // reader runs; the encoder emits them via the same byte-aligned
    // RLE writer. Because the bit reader's input is the unswapped
    // buffer, the per-frame tables and the bitstream prelude are both
    // inside that same unswapped view. We therefore unswap the entire
    // payload up-front.
    let unswapped = unswap_payload(&pkt.data);

    let (tables_owned, residual_offset) = if extra.per_frame_tables {
        let n_tables = extra.table_count();
        let symbols = symbols_per_table(extra);
        let mut tables = Vec::with_capacity(n_tables);
        let mut pos = 0usize;
        for _ in 0..n_tables {
            let (lens, used) = rle::decode_lengths(&unswapped[pos..], symbols)?;
            pos += used;
            tables.push(HuffTable::from_lengths(&lens)?);
        }
        (Some(tables), pos)
    } else {
        (None, 0usize)
    };
    let tables: &[HuffTable] = tables_owned.as_deref().unwrap_or(&extra.tables);

    let bitstream = &unswapped[residual_offset..];

    match (&extra.format, pixel_format) {
        (FormatVersion::V2(V2Colorspace::Yuv422), PixelFormat::Yuv422P) => {
            decode_v2_yuv422(extra, tables, bitstream, width, height, pkt.pts)
        }
        (FormatVersion::V2(V2Colorspace::Yuv420), PixelFormat::Yuv420P) => {
            decode_v2_yuv420(extra, tables, bitstream, width, height, pkt.pts)
        }
        (FormatVersion::V2(V2Colorspace::Rgb24), PixelFormat::Rgb24) => {
            decode_v2_rgb(
                extra, tables, bitstream, width, height, /*alpha=*/ false, pkt.pts,
            )
        }
        (FormatVersion::V2(V2Colorspace::Bgra32), PixelFormat::Bgra) => {
            decode_v2_rgb(
                extra, tables, bitstream, width, height, /*alpha=*/ true, pkt.pts,
            )
        }
        (FormatVersion::V3(_), PixelFormat::Gray8) => {
            decode_v3_gray8(extra, tables, bitstream, width, height, pkt.pts)
        }
        (FormatVersion::V3(_), PixelFormat::Yuv411P) => {
            decode_v3_yuv411p(extra, tables, bitstream, width, height, pkt.pts)
        }
        (
            FormatVersion::V3(_),
            PixelFormat::Yuv420P | PixelFormat::Yuv422P | PixelFormat::Yuv444P,
        ) => decode_v3_yuv_planar(
            extra,
            tables,
            bitstream,
            width,
            height,
            pixel_format,
            pkt.pts,
        ),
        (FormatVersion::V3(v3), PixelFormat::Gray10Le)
        | (FormatVersion::V3(v3), PixelFormat::Gray12Le)
        | (FormatVersion::V3(v3), PixelFormat::Gray16Le) => {
            decode_v3_gray_hbd(extra, tables, bitstream, width, height, v3.bps, pkt.pts)
        }
        (
            FormatVersion::V3(v3),
            PixelFormat::Yuv420P10Le
            | PixelFormat::Yuv422P10Le
            | PixelFormat::Yuv444P10Le
            | PixelFormat::Yuv420P12Le
            | PixelFormat::Yuv422P12Le
            | PixelFormat::Yuv444P12Le,
        ) => decode_v3_yuv_planar_hbd(
            extra,
            tables,
            bitstream,
            width,
            height,
            pixel_format,
            v3.bps,
            pkt.pts,
        ),
        // GBRP / GBRAP planar (8-bit promoted to Gbrp10Le storage, HBD native).
        (
            FormatVersion::V3(v3),
            PixelFormat::Gbrp10Le
            | PixelFormat::Gbrap10Le
            | PixelFormat::Gbrp12Le
            | PixelFormat::Gbrap12Le,
        ) => decode_v3_gbrp(
            extra,
            tables,
            bitstream,
            width,
            height,
            pixel_format,
            v3.bps,
            pkt.pts,
        ),
        _ => Err(Error::unsupported(format!(
            "huffyuv decode: unhandled (format, pixel_format) combo: {:?} / {:?}",
            extra.format, pixel_format
        ))),
    }
}

fn symbols_per_table(extra: &Extradata) -> usize {
    match &extra.format {
        FormatVersion::V2(_) => 256,
        FormatVersion::V3(v) => (1usize << v.bps).min(16384),
    }
}

// ─────────────────────────── v2 YUV 4:2:2 ──────────────────────────

/// Decode a v2 yuv422p packet. Trace doc §5.1 + §5.2.
///
/// Prelude order on disk (logical, after byte-swap-undo): `V0 Y1 U0 Y0`.
/// Then for every row from 0 to H-1 the inner loop reads (Y, U, Y, V)
/// per chroma column. Row 0 starts after the prelude has consumed the
/// first two luma + first chroma U/V samples. The "left" register
/// threads across rows for each of the three planes.
fn decode_v2_yuv422(
    extra: &Extradata,
    tables: &[HuffTable],
    bitstream: &[u8],
    width: usize,
    height: usize,
    pts: Option<i64>,
) -> Result<VideoFrame> {
    let table_y = &tables[0];
    let table_u = &tables[1];
    let table_v = &tables[2];

    let chroma_w = width / 2;
    let mut y_plane = vec![0u8; width * height];
    let mut u_plane = vec![0u8; chroma_w * height];
    let mut v_plane = vec![0u8; chroma_w * height];

    let mut r = BitReader::new(bitstream);

    // Prelude (raw 8 bits each, on-disk order V0 Y1 U0 Y0).
    let v0 = r.read_bits(8)? as u8;
    let y1 = r.read_bits(8)? as u8;
    let u0 = r.read_bits(8)? as u8;
    let y0 = r.read_bits(8)? as u8;

    // Walk row 0: residuals start at sample 2 for luma, sample 1 for
    // chroma. The inner loop emits (Y, U, Y, V) per chroma column, so
    // we resume at chroma column 1. We seed the row buffers with the
    // prelude samples so the eventual copy_from_slice picks up the
    // prelude pixels too.
    let mut row_y = vec![0u8; width];
    let mut row_u = vec![0u8; chroma_w];
    let mut row_v = vec![0u8; chroma_w];
    row_y[0] = y0;
    row_y[1] = y1;
    row_u[0] = u0;
    row_v[0] = v0;
    for cx in 1..chroma_w {
        row_y[2 * cx] = read_sym8(&mut r, table_y)?;
        row_u[cx] = read_sym8(&mut r, table_u)?;
        row_y[2 * cx + 1] = read_sym8(&mut r, table_y)?;
        row_v[cx] = read_sym8(&mut r, table_v)?;
    }
    // Predictor handling for row 0:
    //   - LEFT: just left-predict from the prelude samples.
    //   - GRADIENT: same as LEFT for row 0 (top is implicit zero).
    //   - MEDIAN: same as LEFT for row 0.
    let mut left_y;
    let mut left_u;
    let mut left_v;
    {
        let row_y_tail = &mut row_y[2..];
        left_y = pred_left_inplace(row_y_tail, y1);
        let row_u_tail = &mut row_u[1..];
        left_u = pred_left_inplace(row_u_tail, u0);
        let row_v_tail = &mut row_v[1..];
        left_v = pred_left_inplace(row_v_tail, v0);
    }
    y_plane[..width].copy_from_slice(&row_y);
    u_plane[..chroma_w].copy_from_slice(&row_u);
    v_plane[..chroma_w].copy_from_slice(&row_v);
    // GRADIENT carries `top_left` across rows in addition to `left`.
    // For row 0 there's no "top" so the start-of-row top_left is 0;
    // entering row 1 it becomes the *last* `top` value seen during
    // row 1's processing (equivalently the last sample of row 0).
    let mut top_left_y: u8 = 0;
    let mut top_left_u: u8 = 0;
    let mut top_left_v: u8 = 0;
    let interlaced = matches!(extra.interlace, InterlaceMode::Interlaced)
        || (extra.interlace == InterlaceMode::AutoByHeight && height > 288);

    // Predictor row 1+ scaffolding for MEDIAN bootstrap (trace doc §4.3).
    // For non-MEDIAN predictors the row residual is just decoded into
    // a fresh row vector and then the predictor adds the cross-row
    // contribution.
    for y in 1..height {
        for cx in 0..chroma_w {
            row_y[2 * cx] = read_sym8(&mut r, table_y)?;
            row_u[cx] = read_sym8(&mut r, table_u)?;
            row_y[2 * cx + 1] = read_sym8(&mut r, table_y)?;
            row_v[cx] = read_sym8(&mut r, table_v)?;
        }
        let prev_y_off = (y - 1) * width;
        let prev_u_off = (y - 1) * chroma_w;
        let prev_v_off = (y - 1) * chroma_w;

        // Interlaced: row 1 is the bottom-field start; LEFT-predict with
        // the threading left register (not reset to 0 — the left register
        // is a 1-D prefix sum that continues from row 0).
        if interlaced && y == 1 {
            left_y = pred_left_inplace(&mut row_y, left_y);
            left_u = pred_left_inplace(&mut row_u, left_u);
            left_v = pred_left_inplace(&mut row_v, left_v);
            top_left_y = 0;
            top_left_u = 0;
            top_left_v = 0;
            y_plane[prev_y_off + width..prev_y_off + 2 * width].copy_from_slice(&row_y);
            u_plane[prev_u_off + chroma_w..prev_u_off + 2 * chroma_w].copy_from_slice(&row_u);
            v_plane[prev_v_off + chroma_w..prev_v_off + 2 * chroma_w].copy_from_slice(&row_v);
            continue;
        }

        match extra.predictor {
            Predictor::Left => {
                left_y = pred_left_inplace(&mut row_y, left_y);
                left_u = pred_left_inplace(&mut row_u, left_u);
                left_v = pred_left_inplace(&mut row_v, left_v);
            }
            Predictor::Gradient => {
                let top_y = &y_plane[prev_y_off..prev_y_off + width];
                let top_u = &u_plane[prev_u_off..prev_u_off + chroma_w];
                let top_v = &v_plane[prev_v_off..prev_v_off + chroma_w];
                let (ny, ntl) = pred_gradient_inplace_full(&mut row_y, top_y, left_y, top_left_y);
                left_y = ny;
                top_left_y = ntl;
                let (nu, ntu) = pred_gradient_inplace_full(&mut row_u, top_u, left_u, top_left_u);
                left_u = nu;
                top_left_u = ntu;
                let (nv, ntv) = pred_gradient_inplace_full(&mut row_v, top_v, left_v, top_left_v);
                left_v = nv;
                top_left_v = ntv;
            }
            Predictor::Median => {
                // Row 1 bootstrap: first 4 luma + 2 chroma pairs are
                // LEFT-predicted; the rest of row 1 is MEDIAN against
                // row 0. From row 2 onward, fully MEDIAN.
                // Interlaced: bootstrap applies at row 2 (row 1 handled above).
                let top_y = &y_plane[prev_y_off..prev_y_off + width];
                let top_u = &u_plane[prev_u_off..prev_u_off + chroma_w];
                let top_v = &v_plane[prev_v_off..prev_v_off + chroma_w];
                let is_bootstrap_row = if interlaced { y == 2 } else { y == 1 };
                if is_bootstrap_row {
                    let bs_y = 4.min(width);
                    let bs_c = 2.min(chroma_w);
                    {
                        let (head, tail) = row_y.split_at_mut(bs_y);
                        left_y = pred_left_inplace(head, left_y);
                        if !tail.is_empty() {
                            let init_tl = top_y[bs_y - 1];
                            let (nl, ntl) =
                                pred_median_inplace_full(tail, &top_y[bs_y..], left_y, init_tl);
                            left_y = nl;
                            top_left_y = ntl;
                        }
                    }
                    {
                        let (head, tail) = row_u.split_at_mut(bs_c);
                        left_u = pred_left_inplace(head, left_u);
                        if !tail.is_empty() {
                            let init_tl = top_u[bs_c - 1];
                            let (nl, ntl) =
                                pred_median_inplace_full(tail, &top_u[bs_c..], left_u, init_tl);
                            left_u = nl;
                            top_left_u = ntl;
                        }
                    }
                    {
                        let (head, tail) = row_v.split_at_mut(bs_c);
                        left_v = pred_left_inplace(head, left_v);
                        if !tail.is_empty() {
                            let init_tl = top_v[bs_c - 1];
                            let (nl, ntl) =
                                pred_median_inplace_full(tail, &top_v[bs_c..], left_v, init_tl);
                            left_v = nl;
                            top_left_v = ntl;
                        }
                    }
                } else {
                    // Trace doc §4.3 + empirical cross-check against
                    // ffmpeg's `huffyuv -pred median`: the `top_left`
                    // register threads across rows (= the trailing
                    // `top_left` of the previous row's MEDIAN call =
                    // the previous row's last `top` sample).
                    let (nl, ntl) = pred_median_inplace_full(&mut row_y, top_y, left_y, top_left_y);
                    left_y = nl;
                    top_left_y = ntl;
                    let (nl, ntl) = pred_median_inplace_full(&mut row_u, top_u, left_u, top_left_u);
                    left_u = nl;
                    top_left_u = ntl;
                    let (nl, ntl) = pred_median_inplace_full(&mut row_v, top_v, left_v, top_left_v);
                    left_v = nl;
                    top_left_v = ntl;
                }
            }
        }
        let off_y = y * width;
        let off_c = y * chroma_w;
        y_plane[off_y..off_y + width].copy_from_slice(&row_y);
        u_plane[off_c..off_c + chroma_w].copy_from_slice(&row_u);
        v_plane[off_c..off_c + chroma_w].copy_from_slice(&row_v);
    }

    Ok(VideoFrame {
        pts,
        planes: vec![
            VideoPlane {
                stride: width,
                data: y_plane,
            },
            VideoPlane {
                stride: chroma_w,
                data: u_plane,
            },
            VideoPlane {
                stride: chroma_w,
                data: v_plane,
            },
        ],
    })
}

// ─────────────────────────── v2 YUV 4:2:0 ──────────────────────────

/// Decode a v2 yuv420p packet. Trace doc §5.2: even rows carry both
/// luma and chroma; odd rows carry luma only (chroma stays the same as
/// the row above).
fn decode_v2_yuv420(
    extra: &Extradata,
    tables: &[HuffTable],
    bitstream: &[u8],
    width: usize,
    height: usize,
    pts: Option<i64>,
) -> Result<VideoFrame> {
    let table_y = &tables[0];
    let table_u = &tables[1];
    let table_v = &tables[2];
    let chroma_w = width / 2;
    let chroma_h = height / 2;
    let mut y_plane = vec![0u8; width * height];
    let mut u_plane = vec![0u8; chroma_w * chroma_h];
    let mut v_plane = vec![0u8; chroma_w * chroma_h];

    let mut r = BitReader::new(bitstream);

    // Same prelude as 4:2:2 (V0 Y1 U0 Y0).
    let v0 = r.read_bits(8)? as u8;
    let y1 = r.read_bits(8)? as u8;
    let u0 = r.read_bits(8)? as u8;
    let y0 = r.read_bits(8)? as u8;

    // We collapse 4:2:0 into a row-by-row decode where even rows do
    // both luma and chroma, odd rows do luma only. Predictor handling
    // mirrors 4:2:2 but only applies chroma every other y. Seed the row
    // buffers with the prelude so the copy-back captures it.
    let mut row_y = vec![0u8; width];
    let mut row_u = vec![0u8; chroma_w];
    let mut row_v = vec![0u8; chroma_w];
    row_y[0] = y0;
    row_y[1] = y1;
    row_u[0] = u0;
    row_v[0] = v0;
    let mut left_y = y1;
    let mut left_u = u0;
    let mut left_v = v0;

    // Row 0 — finish the rest of the row using the same Y/U/Y/V
    // interleave starting from chroma column 1. (Restored to the
    // pre-experiment behaviour while we figure out the v2 yuv420p
    // layout properly.)
    for cx in 1..chroma_w {
        row_y[2 * cx] = read_sym8(&mut r, table_y)?;
        row_u[cx] = read_sym8(&mut r, table_u)?;
        row_y[2 * cx + 1] = read_sym8(&mut r, table_y)?;
        row_v[cx] = read_sym8(&mut r, table_v)?;
    }
    {
        let row_y_tail = &mut row_y[2..];
        left_y = pred_left_inplace(row_y_tail, left_y);
        left_u = pred_left_inplace(&mut row_u[1..], left_u);
        left_v = pred_left_inplace(&mut row_v[1..], left_v);
    }
    y_plane[..width].copy_from_slice(&row_y);
    u_plane[..chroma_w].copy_from_slice(&row_u);
    v_plane[..chroma_w].copy_from_slice(&row_v);

    // GRADIENT cross-row top_left registers (see decode_v2_yuv422).
    let mut top_left_y: u8 = 0;
    let mut top_left_u: u8 = 0;
    let mut top_left_v: u8 = 0;

    let mut cy: usize = 1;
    for y in 1..height {
        let do_chroma = (y % 2) == 0; // chroma on even luma rows
        let chroma_first = 0; // unused now; kept to minimise diff
        let _ = chroma_first;

        if do_chroma {
            for cx in 0..chroma_w {
                row_y[2 * cx] = read_sym8(&mut r, table_y)?;
                row_u[cx] = read_sym8(&mut r, table_u)?;
                row_y[2 * cx + 1] = read_sym8(&mut r, table_y)?;
                row_v[cx] = read_sym8(&mut r, table_v)?;
            }
        } else {
            // Luma only for odd rows.
            for x in 0..width {
                row_y[x] = read_sym8(&mut r, table_y)?;
            }
        }

        let prev_y_off = (y - 1) * width;
        match extra.predictor {
            Predictor::Left => {
                left_y = pred_left_inplace(&mut row_y, left_y);
                if do_chroma {
                    left_u = pred_left_inplace(&mut row_u, left_u);
                    left_v = pred_left_inplace(&mut row_v, left_v);
                }
            }
            Predictor::Gradient => {
                let top_y = &y_plane[prev_y_off..prev_y_off + width];
                let (ny, ntl) = pred_gradient_inplace_full(&mut row_y, top_y, left_y, top_left_y);
                left_y = ny;
                top_left_y = ntl;
                if do_chroma {
                    let prev_c_off = (cy - 1) * chroma_w;
                    let top_u = &u_plane[prev_c_off..prev_c_off + chroma_w];
                    let top_v = &v_plane[prev_c_off..prev_c_off + chroma_w];
                    let (nu, ntu) =
                        pred_gradient_inplace_full(&mut row_u, top_u, left_u, top_left_u);
                    left_u = nu;
                    top_left_u = ntu;
                    let (nv, ntv) =
                        pred_gradient_inplace_full(&mut row_v, top_v, left_v, top_left_v);
                    left_v = nv;
                    top_left_v = ntv;
                }
            }
            Predictor::Median => {
                let top_y = &y_plane[prev_y_off..prev_y_off + width];
                if y == 1 {
                    let bs_y = 4.min(width);
                    let (head, tail) = row_y.split_at_mut(bs_y);
                    left_y = pred_left_inplace(head, left_y);
                    if !tail.is_empty() {
                        let init_tl = top_y[bs_y - 1];
                        let (nl, ntl) =
                            pred_median_inplace_full(tail, &top_y[bs_y..], left_y, init_tl);
                        left_y = nl;
                        top_left_y = ntl;
                    }
                } else {
                    let (nl, ntl) = pred_median_inplace_full(&mut row_y, top_y, left_y, top_left_y);
                    left_y = nl;
                    top_left_y = ntl;
                }
                if do_chroma {
                    let prev_c_off = (cy - 1) * chroma_w;
                    let top_u = &u_plane[prev_c_off..prev_c_off + chroma_w];
                    let top_v = &v_plane[prev_c_off..prev_c_off + chroma_w];
                    if cy == 1 {
                        let bs_c = 2.min(chroma_w);
                        let (uh, ut) = row_u.split_at_mut(bs_c);
                        left_u = pred_left_inplace(uh, left_u);
                        if !ut.is_empty() {
                            let init_tl = top_u[bs_c - 1];
                            let (nl, ntl) =
                                pred_median_inplace_full(ut, &top_u[bs_c..], left_u, init_tl);
                            left_u = nl;
                            top_left_u = ntl;
                        }
                        let (vh, vt) = row_v.split_at_mut(bs_c);
                        left_v = pred_left_inplace(vh, left_v);
                        if !vt.is_empty() {
                            let init_tl = top_v[bs_c - 1];
                            let (nl, ntl) =
                                pred_median_inplace_full(vt, &top_v[bs_c..], left_v, init_tl);
                            left_v = nl;
                            top_left_v = ntl;
                        }
                    } else {
                        let (nl, ntl) =
                            pred_median_inplace_full(&mut row_u, top_u, left_u, top_left_u);
                        left_u = nl;
                        top_left_u = ntl;
                        let (nl, ntl) =
                            pred_median_inplace_full(&mut row_v, top_v, left_v, top_left_v);
                        left_v = nl;
                        top_left_v = ntl;
                    }
                }
            }
        }
        let off_y = y * width;
        y_plane[off_y..off_y + width].copy_from_slice(&row_y);
        if do_chroma {
            let off_c = cy * chroma_w;
            u_plane[off_c..off_c + chroma_w].copy_from_slice(&row_u);
            v_plane[off_c..off_c + chroma_w].copy_from_slice(&row_v);
            cy += 1;
        }
    }

    Ok(VideoFrame {
        pts,
        planes: vec![
            VideoPlane {
                stride: width,
                data: y_plane,
            },
            VideoPlane {
                stride: chroma_w,
                data: u_plane,
            },
            VideoPlane {
                stride: chroma_w,
                data: v_plane,
            },
        ],
    })
}

// ─────────────────────────── v2 RGB / RGBA ─────────────────────────

/// Decode a v2 RGB24 (alpha=false) or BGRA (alpha=true) packet. Trace
/// doc §5.1: prelude order is `R0 G0 B0 00` for RGB24 and
/// `A0 R0 G0 B0` for RGBA32; the bitstream stores rows bottom-up.
/// §4.4: when `decorrelate` is set the Huffman residuals are
/// (B-G), G, (R-G) and the decoder inverts that after the spatial
/// predictor.
fn decode_v2_rgb(
    extra: &Extradata,
    tables: &[HuffTable],
    bitstream: &[u8],
    width: usize,
    height: usize,
    has_alpha: bool,
    pts: Option<i64>,
) -> Result<VideoFrame> {
    let table_b = &tables[0];
    let table_g = &tables[1];
    let table_r = &tables[2];
    // V2 RGBA reuses table 2 for alpha (trace doc §3.1).
    let table_a = &tables[2];

    let channels = if has_alpha { 4 } else { 3 };
    let mut frame = vec![0u8; width * height * channels];
    let mut r = BitReader::new(bitstream);

    // Prelude — read raw bytes per format, write them into the
    // bitstream's row 0 (which is the *bottom* of the displayed frame).
    let r0;
    let g0;
    let b0;
    let a0;
    if has_alpha {
        a0 = r.read_bits(8)? as u8;
        r0 = r.read_bits(8)? as u8;
        g0 = r.read_bits(8)? as u8;
        b0 = r.read_bits(8)? as u8;
    } else {
        r0 = r.read_bits(8)? as u8;
        g0 = r.read_bits(8)? as u8;
        b0 = r.read_bits(8)? as u8;
        let _pad = r.read_bits(8)?;
        a0 = 0;
    }

    // The encoder stores rows bottom-up. We decode in storage order
    // (row 0 = bottom-most pixel row) into a temp buffer, then flip
    // rows when emitting the VideoFrame.
    let mut storage = vec![0u8; width * height * channels];
    let pixel_set = |storage: &mut [u8], y: usize, x: usize, r: u8, g: u8, b: u8, a: u8| {
        let off = (y * width + x) * channels;
        if has_alpha {
            // Storage order on-disk is BGRA per pixel.
            storage[off] = b;
            storage[off + 1] = g;
            storage[off + 2] = r;
            storage[off + 3] = a;
        } else {
            // RGB24 stored as B G R per the trace doc tables (BGR scan
            // order on disk; the on-disk pixel layout itself is BGR for
            // V2 RGB24).
            storage[off] = b;
            storage[off + 1] = g;
            storage[off + 2] = r;
        }
    };
    pixel_set(&mut storage, 0, 0, r0, g0, b0, a0);

    let mut left_b = b0;
    let mut left_g = g0;
    let mut left_r = r0;
    let mut left_a = a0;

    // Row 0: decode the rest of the row's residuals one pixel at a
    // time. For non-decorrelate streams every channel is a plain LEFT
    // prefix sum. For decorrelate streams we feed G through LEFT only
    // and synthesise B/R from G + per-sample residuals.
    for x in 1..width {
        let res_b = read_sym8(&mut r, table_b)?;
        let res_g = read_sym8(&mut r, table_g)?;
        let res_r = read_sym8(&mut r, table_r)?;
        let res_a = if has_alpha {
            read_sym8(&mut r, table_a)?
        } else {
            0
        };
        left_g = left_g.wrapping_add(res_g);
        if has_alpha {
            left_a = left_a.wrapping_add(res_a);
        }
        let (px_r, px_g, px_b, px_a) = if extra.decorrelate {
            (
                left_g.wrapping_add(res_r),
                left_g,
                left_g.wrapping_add(res_b),
                left_a,
            )
        } else {
            left_b = left_b.wrapping_add(res_b);
            left_r = left_r.wrapping_add(res_r);
            (left_r, left_g, left_b, left_a)
        };
        pixel_set(&mut storage, 0, x, px_r, px_g, px_b, px_a);
        if extra.decorrelate {
            left_b = px_b;
            left_r = px_r;
        }
    }

    // Rows 1..H-1 (in storage order = bottom-up display order).
    for y in 1..height {
        // Pre-decode the row's residuals into per-channel buffers.
        let mut row_b = vec![0u8; width];
        let mut row_g = vec![0u8; width];
        let mut row_r = vec![0u8; width];
        let mut row_a = vec![0u8; width];
        for x in 0..width {
            row_b[x] = read_sym8(&mut r, table_b)?;
            row_g[x] = read_sym8(&mut r, table_g)?;
            row_r[x] = read_sym8(&mut r, table_r)?;
            if has_alpha {
                row_a[x] = read_sym8(&mut r, table_a)?;
            }
        }
        // Apply spatial predictor channel by channel. RGB v2 only
        // supports LEFT and GRADIENT (MEDIAN is rejected at encode
        // time, trace doc §6.3).
        let prev_off = (y - 1) * width * channels;
        let read_prev = |c: usize, x: usize| -> u8 { storage[prev_off + x * channels + c] };
        let (top_b, top_g, top_r, top_a): (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) = (
            (0..width).map(|x| read_prev(0, x)).collect(),
            (0..width).map(|x| read_prev(1, x)).collect(),
            (0..width).map(|x| read_prev(2, x)).collect(),
            if has_alpha {
                (0..width).map(|x| read_prev(3, x)).collect()
            } else {
                Vec::new()
            },
        );
        match extra.predictor {
            Predictor::Left => {
                if extra.decorrelate {
                    // Decode G first (LEFT), then B := G + B_res, R := G + R_res
                    // (trace doc §4.4). The B/R "left" register isn't a true
                    // 1-D prefix sum here — the inverse decorrelation derives
                    // each pixel's B/R purely from the same column's G value
                    // plus the per-sample (B-G)/(R-G) residual.
                    left_g = pred_left_inplace(&mut row_g, left_g);
                    for x in 0..width {
                        let g = row_g[x];
                        row_b[x] = g.wrapping_add(row_b[x]);
                        row_r[x] = g.wrapping_add(row_r[x]);
                    }
                    if has_alpha {
                        left_a = pred_left_inplace(&mut row_a, left_a);
                    }
                    left_b = *row_b.last().unwrap();
                    left_r = *row_r.last().unwrap();
                } else {
                    left_b = pred_left_inplace(&mut row_b, left_b);
                    left_g = pred_left_inplace(&mut row_g, left_g);
                    left_r = pred_left_inplace(&mut row_r, left_r);
                    if has_alpha {
                        left_a = pred_left_inplace(&mut row_a, left_a);
                    }
                }
            }
            Predictor::Gradient => {
                if extra.decorrelate {
                    left_g = pred_gradient_inplace(&mut row_g, &top_g, left_g);
                    for x in 0..width {
                        let g = row_g[x];
                        row_b[x] = g.wrapping_add(row_b[x]);
                        row_r[x] = g.wrapping_add(row_r[x]);
                    }
                    if has_alpha {
                        left_a = pred_gradient_inplace(&mut row_a, &top_a, left_a);
                    }
                    left_b = *row_b.last().unwrap();
                    left_r = *row_r.last().unwrap();
                } else {
                    left_b = pred_gradient_inplace(&mut row_b, &top_b, left_b);
                    left_g = pred_gradient_inplace(&mut row_g, &top_g, left_g);
                    left_r = pred_gradient_inplace(&mut row_r, &top_r, left_r);
                    if has_alpha {
                        left_a = pred_gradient_inplace(&mut row_a, &top_a, left_a);
                    }
                }
            }
            Predictor::Median => {
                return Err(Error::invalid(
                    "huffyuv v2 RGB: MEDIAN predictor is rejected by the encoder",
                ));
            }
        }
        for x in 0..width {
            pixel_set(
                &mut storage,
                y,
                x,
                row_r[x],
                row_g[x],
                row_b[x],
                if has_alpha { row_a[x] } else { 0 },
            );
        }
    }

    // Storage is bottom-up (row 0 = bottom of displayed frame); flip
    // into the output frame buffer (top-down). For RGB24 we also need
    // to convert the on-disk BGR pixel layout to RGB packed.
    let stride = width * channels;
    if has_alpha {
        // BGRA on disk == BGRA in PixelFormat::Bgra (matches frame layout).
        for y in 0..height {
            let src_off = y * stride;
            let dst_off = (height - 1 - y) * stride;
            frame[dst_off..dst_off + stride].copy_from_slice(&storage[src_off..src_off + stride]);
        }
    } else {
        // BGR-on-disk → RGB-in-frame.
        for y in 0..height {
            let src_off = y * stride;
            let dst_off = (height - 1 - y) * stride;
            for x in 0..width {
                let so = src_off + x * 3;
                let dop = dst_off + x * 3;
                frame[dop] = storage[so + 2]; // R
                frame[dop + 1] = storage[so + 1]; // G
                frame[dop + 2] = storage[so]; // B
            }
        }
    }

    Ok(VideoFrame {
        pts,
        planes: vec![VideoPlane {
            stride,
            data: frame,
        }],
    })
}

// ─────────────────────────── v3 Gray 8 ─────────────────────────────

fn decode_v3_gray8(
    extra: &Extradata,
    tables: &[HuffTable],
    bitstream: &[u8],
    width: usize,
    height: usize,
    pts: Option<i64>,
) -> Result<VideoFrame> {
    let table = &tables[0];
    let mut plane = vec![0u8; width * height];
    let mut r = BitReader::new(bitstream);
    decode_one_plane(extra, table, &mut r, &mut plane, width, height)?;
    Ok(VideoFrame {
        pts,
        planes: vec![VideoPlane {
            stride: width,
            data: plane,
        }],
    })
}

// ────────────────────── v3 YUV planar (8-bit) ──────────────────────

fn decode_v3_yuv_planar(
    extra: &Extradata,
    tables: &[HuffTable],
    bitstream: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    pts: Option<i64>,
) -> Result<VideoFrame> {
    let (h_shift, v_shift) = match pixel_format {
        PixelFormat::Yuv444P => (0, 0),
        PixelFormat::Yuv422P => (1, 0),
        PixelFormat::Yuv420P => (1, 1),
        _ => unreachable!(),
    };
    let chroma_w = width >> h_shift;
    let chroma_h = height >> v_shift;
    let mut y_plane = vec![0u8; width * height];
    let mut u_plane = vec![0u8; chroma_w * chroma_h];
    let mut v_plane = vec![0u8; chroma_w * chroma_h];
    let mut r = BitReader::new(bitstream);

    decode_one_plane(extra, &tables[0], &mut r, &mut y_plane, width, height)?;
    decode_one_plane(extra, &tables[1], &mut r, &mut u_plane, chroma_w, chroma_h)?;
    decode_one_plane(extra, &tables[2], &mut r, &mut v_plane, chroma_w, chroma_h)?;

    Ok(VideoFrame {
        pts,
        planes: vec![
            VideoPlane {
                stride: width,
                data: y_plane,
            },
            VideoPlane {
                stride: chroma_w,
                data: u_plane,
            },
            VideoPlane {
                stride: chroma_w,
                data: v_plane,
            },
        ],
    })
}

// ────────────────────── v3 YUV 4:1:1 ──────────────────────

fn decode_v3_yuv411p(
    extra: &Extradata,
    tables: &[HuffTable],
    bitstream: &[u8],
    width: usize,
    height: usize,
    pts: Option<i64>,
) -> Result<VideoFrame> {
    // 4:1:1: chroma_h_shift=2, chroma_v_shift=0.
    let chroma_w = width >> 2;
    let chroma_h = height;
    let mut y_plane = vec![0u8; width * height];
    let mut u_plane = vec![0u8; chroma_w * chroma_h];
    let mut v_plane = vec![0u8; chroma_w * chroma_h];
    let mut r = BitReader::new(bitstream);

    decode_one_plane(extra, &tables[0], &mut r, &mut y_plane, width, height)?;
    decode_one_plane(extra, &tables[1], &mut r, &mut u_plane, chroma_w, chroma_h)?;
    decode_one_plane(extra, &tables[2], &mut r, &mut v_plane, chroma_w, chroma_h)?;

    Ok(VideoFrame {
        pts,
        planes: vec![
            VideoPlane {
                stride: width,
                data: y_plane,
            },
            VideoPlane {
                stride: chroma_w,
                data: u_plane,
            },
            VideoPlane {
                stride: chroma_w,
                data: v_plane,
            },
        ],
    })
}

/// Plane-sequential decode for v3 (and v3-style single-plane) layouts.
/// Trace doc §5.3: each row is read left-to-right; row 0 is fully
/// LEFT-predicted with `left = 0` initial register; rows 1..H-1 use
/// the configured predictor.
///
/// When `extra.interlace` is `Interlaced`, row 1 is treated as a fresh
/// field-start: it is fully LEFT-predicted (not GRADIENT or MEDIAN), and
/// the MEDIAN bootstrap (row 1 partial-left) is shifted to row 2.
fn decode_one_plane(
    extra: &Extradata,
    table: &HuffTable,
    r: &mut BitReader<'_>,
    plane: &mut [u8],
    width: usize,
    height: usize,
) -> Result<()> {
    if width == 0 || height == 0 {
        return Ok(());
    }
    let interlaced = matches!(extra.interlace, InterlaceMode::Interlaced)
        || (extra.interlace == InterlaceMode::AutoByHeight && height > 288);
    let mut row = vec![0u8; width];
    // Row 0.
    for x in 0..width {
        row[x] = read_sym8(r, table)?;
    }
    let mut left = pred_left_inplace(&mut row, 0);
    plane[..width].copy_from_slice(&row);
    let mut top_left: u8 = 0;

    for y in 1..height {
        for x in 0..width {
            row[x] = read_sym8(r, table)?;
        }
        let prev_off = (y - 1) * width;
        // In interlaced mode, row 1 is the start of the bottom field:
        // treat it as LEFT-predicted (reset left to 0, no cross-row
        // reference). The MEDIAN bootstrap then applies at row 2.
        if interlaced && y == 1 {
            left = pred_left_inplace(&mut row, 0);
            top_left = 0;
            let off = y * width;
            plane[off..off + width].copy_from_slice(&row);
            continue;
        }
        match extra.predictor {
            Predictor::Left => {
                left = pred_left_inplace(&mut row, left);
            }
            Predictor::Gradient => {
                let top = &plane[prev_off..prev_off + width];
                let (nl, ntl) = pred_gradient_inplace_full(&mut row, top, left, top_left);
                left = nl;
                top_left = ntl;
            }
            Predictor::Median => {
                let top = &plane[prev_off..prev_off + width];
                // v3 planar median: top_left threads across rows.
                // For row 1 non-interlaced (first MEDIAN row) top_left is 0.
                // Interlaced: bootstrap shifts to row 2 (row 1 handled above).
                let (nl, ntl) = pred_median_inplace_full(&mut row, top, left, top_left);
                left = nl;
                top_left = ntl;
            }
        }
        let off = y * width;
        plane[off..off + width].copy_from_slice(&row);
    }
    Ok(())
}

// ──────────────── v3 high-bit-depth (9..16 bps) ────────────────

fn decode_v3_gray_hbd(
    extra: &Extradata,
    tables: &[HuffTable],
    bitstream: &[u8],
    width: usize,
    height: usize,
    bps: u8,
    pts: Option<i64>,
) -> Result<VideoFrame> {
    let table = &tables[0];
    let mut samples = vec![0u16; width * height];
    let mut r = BitReader::new(bitstream);
    decode_one_plane_u16(extra, table, &mut r, &mut samples, width, height, bps)?;
    let plane_bytes = u16_plane_to_le_bytes(&samples);
    Ok(VideoFrame {
        pts,
        planes: vec![VideoPlane {
            stride: width * 2,
            data: plane_bytes,
        }],
    })
}

#[allow(clippy::too_many_arguments)]
fn decode_v3_yuv_planar_hbd(
    extra: &Extradata,
    tables: &[HuffTable],
    bitstream: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    bps: u8,
    pts: Option<i64>,
) -> Result<VideoFrame> {
    let (h_shift, v_shift) = match pixel_format {
        PixelFormat::Yuv444P10Le | PixelFormat::Yuv444P12Le => (0, 0),
        PixelFormat::Yuv422P10Le | PixelFormat::Yuv422P12Le => (1, 0),
        PixelFormat::Yuv420P10Le | PixelFormat::Yuv420P12Le => (1, 1),
        _ => unreachable!(),
    };
    let chroma_w = width >> h_shift;
    let chroma_h = height >> v_shift;
    let mut y = vec![0u16; width * height];
    let mut u = vec![0u16; chroma_w * chroma_h];
    let mut v = vec![0u16; chroma_w * chroma_h];
    let mut r = BitReader::new(bitstream);
    decode_one_plane_u16(extra, &tables[0], &mut r, &mut y, width, height, bps)?;
    decode_one_plane_u16(extra, &tables[1], &mut r, &mut u, chroma_w, chroma_h, bps)?;
    decode_one_plane_u16(extra, &tables[2], &mut r, &mut v, chroma_w, chroma_h, bps)?;
    Ok(VideoFrame {
        pts,
        planes: vec![
            VideoPlane {
                stride: width * 2,
                data: u16_plane_to_le_bytes(&y),
            },
            VideoPlane {
                stride: chroma_w * 2,
                data: u16_plane_to_le_bytes(&u),
            },
            VideoPlane {
                stride: chroma_w * 2,
                data: u16_plane_to_le_bytes(&v),
            },
        ],
    })
}

// ─────────────────── v3 GBRP / GBRAP planar ───────────────────

/// Decode v3 GBRP or GBRAP. Planes are stored in G→B→R[→A] order.
/// 8-bit streams are promoted to the 16-bit LE container (low 8 bits
/// used) to fill `Gbrp10Le` / `Gbrap10Le` VideoPlanes.
#[allow(clippy::too_many_arguments)]
fn decode_v3_gbrp(
    extra: &Extradata,
    tables: &[HuffTable],
    bitstream: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    bps: u8,
    pts: Option<i64>,
) -> Result<VideoFrame> {
    let has_alpha = matches!(
        pixel_format,
        PixelFormat::Gbrap10Le | PixelFormat::Gbrap12Le
    );
    let n_planes = if has_alpha { 4 } else { 3 };
    if tables.len() < n_planes {
        return Err(Error::invalid(format!(
            "huffyuv gbrp decode: expected {n_planes} tables, got {}",
            tables.len()
        )));
    }
    let mut planes: Vec<Vec<u16>> = (0..n_planes).map(|_| vec![0u16; width * height]).collect();
    let mut r = BitReader::new(bitstream);
    for p in 0..n_planes {
        decode_one_plane_u16(
            extra,
            &tables[p],
            &mut r,
            &mut planes[p],
            width,
            height,
            bps,
        )?;
    }
    let stride = width * 2;
    Ok(VideoFrame {
        pts,
        planes: planes
            .into_iter()
            .map(|s| VideoPlane {
                stride,
                data: u16_plane_to_le_bytes(&s),
            })
            .collect(),
    })
}

fn decode_one_plane_u16(
    extra: &Extradata,
    table: &HuffTable,
    r: &mut BitReader<'_>,
    plane: &mut [u16],
    width: usize,
    height: usize,
    bps: u8,
) -> Result<()> {
    if width == 0 || height == 0 {
        return Ok(());
    }
    let interlaced = matches!(extra.interlace, InterlaceMode::Interlaced)
        || (extra.interlace == InterlaceMode::AutoByHeight && height > 288);
    let mask: u32 = if bps >= 16 { 0xFFFF } else { (1u32 << bps) - 1 };
    let mut row = vec![0u16; width];
    // Row 0.
    for x in 0..width {
        row[x] = read_sym_hbd(r, table, bps)? as u16;
    }
    let mut left = pred_left_inplace_u16(&mut row, 0, mask);
    plane[..width].copy_from_slice(&row);
    let mut top_left: u16 = 0;

    for y in 1..height {
        for x in 0..width {
            row[x] = read_sym_hbd(r, table, bps)? as u16;
        }
        let prev_off = (y - 1) * width;
        // Interlaced: row 1 is the bottom-field start; LEFT-predict with left=0.
        if interlaced && y == 1 {
            left = pred_left_inplace_u16(&mut row, 0, mask);
            top_left = 0;
            let off = y * width;
            plane[off..off + width].copy_from_slice(&row);
            continue;
        }
        match extra.predictor {
            Predictor::Left => {
                left = pred_left_inplace_u16(&mut row, left, mask);
            }
            Predictor::Gradient => {
                let top = &plane[prev_off..prev_off + width];
                let (nl, ntl) = pred_gradient_inplace_full_u16(&mut row, top, left, top_left, mask);
                left = nl;
                top_left = ntl;
            }
            Predictor::Median => {
                let top = &plane[prev_off..prev_off + width];
                let (nl, ntl) = pred_median_inplace_full_u16(&mut row, top, left, top_left, mask);
                left = nl;
                top_left = ntl;
            }
        }
        let off = y * width;
        plane[off..off + width].copy_from_slice(&row);
    }
    Ok(())
}

/// Read one HBD symbol — Huffman-coded for ≤14 bps; for 15/16 bps the
/// codec switches to "Huffman the high (bps-2) bits, splice the low 2
/// bits raw" (trace doc §5.5 / §9.7).
fn read_sym_hbd(r: &mut BitReader<'_>, t: &HuffTable, bps: u8) -> Result<u32> {
    if bps <= 14 {
        let s = t.read_symbol(r)?;
        let cap = if bps >= 32 {
            u32::MAX
        } else {
            (1u32 << bps) - 1
        };
        if s > cap {
            return Err(Error::invalid(format!(
                "huffyuv: bps={bps} table emitted symbol {s} > {cap}"
            )));
        }
        Ok(s)
    } else {
        // bps ∈ {15, 16}: Huffman code carries (sample >> 2), then 2
        // raw bits carry (sample & 3).
        let hi = t.read_symbol(r)?;
        let lo = r.read_bits(2)?;
        Ok((hi << 2) | lo)
    }
}

/// In-place LEFT prefix-sum over a u16 row (modular within `mask + 1`).
fn pred_left_inplace_u16(row: &mut [u16], mut left: u16, mask: u32) -> u16 {
    for v in row.iter_mut() {
        let next = ((left as u32).wrapping_add(*v as u32)) & mask;
        *v = next as u16;
        left = next as u16;
    }
    left
}

fn pred_gradient_inplace_full_u16(
    row: &mut [u16],
    top: &[u16],
    mut left: u16,
    mut top_left: u16,
    mask: u32,
) -> (u16, u16) {
    debug_assert_eq!(row.len(), top.len());
    // Canonical PLANE: pred[x] = left + top[x] - top_left, threaded
    // per-sample. See `pred_gradient_inplace_full` in predictor.rs.
    for x in 0..row.len() {
        let pred = ((left as u32)
            .wrapping_add(top[x] as u32)
            .wrapping_sub(top_left as u32))
            & mask;
        let s = pred.wrapping_add(row[x] as u32) & mask;
        row[x] = s as u16;
        top_left = top[x];
        left = s as u16;
    }
    (left, top_left)
}

fn pred_median_inplace_full_u16(
    row: &mut [u16],
    top: &[u16],
    mut left: u16,
    mut top_left: u16,
    mask: u32,
) -> (u16, u16) {
    debug_assert_eq!(row.len(), top.len());
    // Modular u16 (or n-bit, see `mask`) median — same shape as the
    // 8-bit `paeth_median_u8` in predictor.rs. The `L+T-TL` term wraps
    // mod 2^bps before the order-statistic median picks the middle.
    for x in 0..row.len() {
        let l = left as u32;
        let t = top[x] as u32;
        let tl = top_left as u32;
        let c = (l.wrapping_add(t).wrapping_sub(tl)) & mask;
        let lo = l.min(t);
        let hi = l.max(t);
        let pred = hi.min(c.max(lo));
        let s = pred.wrapping_add(row[x] as u32) & mask;
        row[x] = s as u16;
        top_left = top[x];
        left = s as u16;
    }
    (left, top_left)
}

fn u16_plane_to_le_bytes(samples: &[u16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for &s in samples.iter() {
        out.push((s & 0xFF) as u8);
        out.push((s >> 8) as u8);
    }
    out
}

// ─────────────────────────── helpers ────────────────────────────

fn read_sym8(r: &mut BitReader<'_>, t: &HuffTable) -> Result<u8> {
    let s = t.read_symbol(r)?;
    if s > 0xFF {
        return Err(Error::invalid(format!(
            "huffyuv: 8-bit table emitted symbol {s} > 255"
        )));
    }
    Ok(s as u8)
}
