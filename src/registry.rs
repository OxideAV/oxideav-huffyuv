//! `oxideav-core` framework integration.
//!
//! Compiled only when the default-on `registry` feature is enabled.
//! Standalone consumers (`default-features = false`) skip this
//! module.

#![cfg(feature = "registry")]

use std::sync::Arc;

use oxideav_core::{
    CodecCapabilities, CodecId, CodecInfo, CodecOptionsStruct, CodecParameters, CodecRegistry,
    CodecTag, Decoder, Encoder, Error as CoreError, ExecutionContext, Frame, OptionField,
    OptionKind, OptionValue, Packet, PixelFormat, Rational, Result as CoreResult, RuntimeContext,
    TimeBase, VideoFrame, VideoPlane, parse_options,
};

use crate::decoder::{DecodedFrame, ThreeTables, decode_frame_with_workers, table_cache};
use crate::encoder::{
    EncodeOptions, ExtradataMode, FieldCoding, FrameShape, MethodSelection,
    build_bitmapinfoheader_field_coded, encode_body_with_pinned_tables,
    encode_stream_opening_frame,
};
use crate::header::{Method, PixelFamily, StreamConfig};
use crate::tables::classic_blob_bytes;

/// Canonical codec id.
pub const CODEC_ID_STR: &str = "huffyuv";

/// Register the HuffYUV / FFVHuff codec with `reg`.
///
/// Claims the two FourCCs the AVI demuxer needs to resolve a HuffYUV
/// stream — `HFYU` (the original VfW FourCC; spec/01 §1.5)
/// and `FFVH` (the FFVHuff extended-variant FourCC). The decoder in this crate
/// only handles the 8-bit family the round-1 deliverable certifies;
/// 10/12-bit FFVHuff variants are deferred to a future round.
///
/// Round 429 closes the dual-API convention on the encode side: the
/// registration carries BOTH factories ([`make_decoder`] +
/// [`make_encoder`]), and both remain directly callable alongside the
/// registry path.
pub fn register_codecs(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::video("huffyuv_sw")
        .with_decode()
        .with_encode()
        .with_lossless(true)
        .with_intra_only(true);
    reg.register(
        CodecInfo::new(CodecId::new(CODEC_ID_STR))
            .capabilities(caps)
            .decoder(make_decoder)
            .encoder(make_encoder)
            .encoder_options::<HuffYuvEncoderOptions>()
            .tags([CodecTag::fourcc(b"HFYU"), CodecTag::fourcc(b"FFVH")]),
    );
}

/// Unified entry point invoked by the macro-generated wrapper.
pub fn register(ctx: &mut RuntimeContext) {
    register_codecs(&mut ctx.codecs);
}

// ──────────────────────── Decoder impl ────────────────────────

/// Registry decoder factory — the exact function
/// [`register_codecs`] wires as the decode-side factory, exposed
/// directly per the workspace dual-API convention.
///
/// `params.extradata` carries the AVI `strf` payload
/// (`BITMAPINFOHEADER` + optional v2.x Huffman tables); it is parsed
/// once here so per-packet decode is header-free.
pub fn make_decoder(params: &CodecParameters) -> CoreResult<Box<dyn Decoder>> {
    let stream_config = if !params.extradata.is_empty() {
        Some(
            StreamConfig::parse_bitmapinfoheader(&params.extradata)
                .map_err(|e| CoreError::invalid(format!("oxideav-huffyuv: {e}")))?,
        )
    } else {
        None
    };
    Ok(Box::new(HuffYuvDecoder {
        codec_id: params.codec_id.clone(),
        stream_config,
        pending: None,
        eof: false,
        worker_budget: 1,
    }))
}

struct HuffYuvDecoder {
    codec_id: CodecId,
    /// Parsed once at decoder-create time when `CodecParameters`
    /// carried a BIH/extradata blob.
    stream_config: Option<StreamConfig>,
    pending: Option<Packet>,
    eof: bool,
    /// Advisory worker budget from [`Decoder::set_execution_context`].
    /// Defaults to 1 — per the `ExecutionContext` threading contract
    /// the codec runs serial until told otherwise. A frame is a single
    /// codeword stream with no resynchronisation point (spec/02 §2,
    /// spec/06 §3.4), so decode has one unit of work and the budget is
    /// recorded but changes nothing; see [`decode_frame_with_workers`].
    worker_budget: usize,
}

impl Decoder for HuffYuvDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> CoreResult<()> {
        if self.pending.is_some() {
            return Err(CoreError::other(
                "oxideav-huffyuv: receive_frame must be called before sending another packet",
            ));
        }
        self.pending = Some(packet.clone());
        Ok(())
    }

    fn receive_frame(&mut self) -> CoreResult<Frame> {
        let Some(pkt) = self.pending.take() else {
            return if self.eof {
                Err(CoreError::Eof)
            } else {
                Err(CoreError::NeedMore)
            };
        };
        let cfg = self.stream_config.as_ref().ok_or_else(|| {
            CoreError::invalid("oxideav-huffyuv: missing BITMAPINFOHEADER in CodecParameters")
        })?;
        let decoded = decode_frame_with_workers(cfg, &pkt.data, self.worker_budget)
            .map_err(|e| CoreError::invalid(format!("oxideav-huffyuv: {e}")))?;
        Ok(Frame::Video(map_to_video_frame(decoded, pkt.pts)))
    }

    fn flush(&mut self) -> CoreResult<()> {
        self.eof = true;
        Ok(())
    }

    fn set_execution_context(&mut self, ctx: &ExecutionContext) {
        // Store the advisory budget; `ctx.threads` is documented ≥ 1
        // but clamp defensively so a zero can never disable decode.
        self.worker_budget = ctx.threads.max(1);
    }

    fn output_pixel_format(&self) -> Option<PixelFormat> {
        self.stream_config
            .as_ref()
            .map(|cfg| pixel_format_for_family(cfg.family))
    }

    /// Size and packed pixel format of the emitted frames, known from
    /// the stream header at construction (spec/01 §1–§4). The header
    /// is stream-level and fixed (spec/02 §1.1), so the answer never
    /// changes.
    fn output_params(&self) -> Option<CodecParameters> {
        let cfg = self.stream_config.as_ref()?;
        let mut params = CodecParameters::video(self.codec_id.clone());
        params.width = Some(cfg.width);
        params.height = Some(cfg.height);
        params.pixel_format = Some(pixel_format_for_family(cfg.family));
        Some(params)
    }
}

/// Reverse the row order of a packed raster of `row_bytes`-byte rows.
/// The two RGB families are stored bottom row first on the wire
/// (spec/02 §2); framework frames carry the top row first.
fn reverse_rows(pixels: &[u8], row_bytes: usize) -> Vec<u8> {
    if row_bytes == 0 {
        return pixels.to_vec();
    }
    let mut out = Vec::with_capacity(pixels.len());
    for row in pixels.chunks_exact(row_bytes).rev() {
        out.extend_from_slice(row);
    }
    out
}

fn map_to_video_frame(frame: DecodedFrame, pts: Option<i64>) -> VideoFrame {
    let stride = frame.family.row_bytes(frame.width);
    let data = if frame.family.is_rgb() {
        reverse_rows(&frame.pixels, stride)
    } else {
        frame.pixels
    };
    let planes = vec![VideoPlane { stride, data }];
    VideoFrame { pts, planes }
}

// ──────────────────────── Encoder impl ────────────────────────

/// Packed input pixel family for a core [`PixelFormat`], when the
/// format matches one of the three 8-bit rasters HuffYUV encodes
/// (spec/02 §3 wire-byte layout): `Yuyv422` ↔ YUY2 (`Y₁ U Y₂ V`),
/// `Bgr24` ↔ RGB24 (`+0:B +1:G +2:R`), `Bgra` ↔ RGB32
/// (`+0:B +1:G +2:R +3:A`).
fn family_from_pixel_format(pf: PixelFormat) -> Option<PixelFamily> {
    match pf {
        PixelFormat::Yuyv422 => Some(PixelFamily::Yuy2),
        PixelFormat::Bgr24 => Some(PixelFamily::Rgb24),
        PixelFormat::Bgra => Some(PixelFamily::Rgb32),
        _ => None,
    }
}

/// Inverse of [`family_from_pixel_format`] — the label advertised on
/// the encoder's output parameters (and matching what the registry
/// decoder reconstructs).
fn pixel_format_for_family(family: PixelFamily) -> PixelFormat {
    match family {
        PixelFamily::Yuy2 => PixelFormat::Yuyv422,
        PixelFamily::Rgb24 => PixelFormat::Bgr24,
        PixelFamily::Rgb32 => PixelFormat::Bgra,
    }
}

/// Time base stamped on emitted packets: one tick per frame when the
/// caller declared a frame rate, else the microsecond fallback base.
/// `pts` is passed through from the input frame verbatim either way.
fn packet_time_base(frame_rate: Option<Rational>) -> TimeBase {
    match frame_rate {
        Some(fr) if fr.num > 0 && fr.den > 0 => TimeBase::new(fr.den, fr.num),
        _ => TimeBase::MICROS,
    }
}

/// Typed options for the registry encoder — the axes the direct API
/// exposes through [`EncodeOptions`]: input pixel family, predictor
/// method, extradata mode, field coding.
#[derive(Debug, Clone, Copy, Default)]
#[non_exhaustive]
pub struct HuffYuvEncoderOptions {
    /// Input pixel family. `None` (bag value `"auto"`, the default)
    /// derives it from `CodecParameters::pixel_format`.
    pub format: Option<PixelFamily>,
    /// Predictor / decorrelation method. `None` (bag value `"auto"`,
    /// the default) runs the bit-cost auto-selector on the first frame
    /// and pins the winner for the stream (the method byte lives in
    /// the stream-level `BITMAPINFOHEADER`, spec/01 §3 — it cannot
    /// change per frame).
    pub method: Option<Method>,
    /// Extradata mode: `"classic-v2"` (default), `"custom-v2"`, or
    /// `"v1x"`. Mirrors [`ExtradataMode`].
    pub mode: ExtradataMode,
    /// Field coding: `"auto"` (default — field-code above 288 lines,
    /// spec/06 §1.1), `"fields"`, or `"plain"`. Mirrors
    /// [`FieldCoding`].
    pub field_coding: FieldCoding,
}

impl HuffYuvEncoderOptions {
    /// Set `format`.
    pub fn with_format(mut self, v: Option<PixelFamily>) -> Self {
        self.format = v;
        self
    }

    /// Set `method`.
    pub fn with_method(mut self, v: Option<Method>) -> Self {
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
}

impl CodecOptionsStruct for HuffYuvEncoderOptions {
    const SCHEMA: &'static [OptionField] = &[
        OptionField {
            name: "format",
            kind: OptionKind::Enum(&["auto", "yuy2", "rgb24", "rgb32"]),
            default: OptionValue::String(String::new()),
            help: "input pixel family (auto = derive from CodecParameters::pixel_format)",
        },
        OptionField {
            name: "method",
            kind: OptionKind::Enum(&[
                "auto",
                "predict-old",
                "left",
                "gradient",
                "median",
                "left-decorr",
                "gradient-decorr",
            ]),
            default: OptionValue::String(String::new()),
            help: "predictor method (auto = bit-cost selection on the first frame, \
                   pinned for the stream)",
        },
        OptionField {
            name: "mode",
            kind: OptionKind::Enum(&["classic-v2", "custom-v2", "v1x"]),
            default: OptionValue::String(String::new()),
            help: "extradata mode (default classic-v2; custom-v2 pins per-stream \
                   optimal tables from the first frame; v1x emits a biSize=0x28 \
                   header with no extradata)",
        },
        OptionField {
            name: "field-coding",
            kind: OptionKind::Enum(&["auto", "fields", "plain"]),
            default: OptionValue::String(String::new()),
            help: "field (interlaced) coding: auto = above 288 lines, fields = always, \
                   plain = never; fields/plain against the 288-line rule need extradata",
        },
    ];

    fn apply(&mut self, key: &str, value: &OptionValue) -> CoreResult<()> {
        match key {
            "format" => {
                self.format = match value.as_str()? {
                    "auto" => None,
                    "yuy2" => Some(PixelFamily::Yuy2),
                    "rgb24" => Some(PixelFamily::Rgb24),
                    "rgb32" => Some(PixelFamily::Rgb32),
                    _ => unreachable!("guarded by SCHEMA"),
                }
            }
            "method" => {
                self.method = match value.as_str()? {
                    "auto" => None,
                    "predict-old" => Some(Method::PredictOld),
                    "left" => Some(Method::Left),
                    "gradient" => Some(Method::Gradient),
                    "median" => Some(Method::Median),
                    "left-decorr" => Some(Method::LeftDecorr),
                    "gradient-decorr" => Some(Method::GradientDecorr),
                    _ => unreachable!("guarded by SCHEMA"),
                }
            }
            "mode" => {
                self.mode = match value.as_str()? {
                    "classic-v2" => ExtradataMode::ClassicV2,
                    "custom-v2" => ExtradataMode::CustomV2,
                    "v1x" => ExtradataMode::V1xCompat,
                    _ => unreachable!("guarded by SCHEMA"),
                }
            }
            "field-coding" => {
                self.field_coding = match value.as_str()? {
                    "auto" => FieldCoding::Auto,
                    "fields" => FieldCoding::Fields,
                    "plain" => FieldCoding::Plain,
                    _ => unreachable!("guarded by SCHEMA"),
                }
            }
            _ => unreachable!("guarded by SCHEMA"),
        }
        Ok(())
    }
}

/// Registry encoder factory (round 429 — dual-API closure on the
/// encode side; also wired by [`register_codecs`]).
///
/// Required parameters: `width` + `height`, and an input family via
/// `pixel_format` (`Yuyv422` / `Bgr24` / `Bgra`) or the `format`
/// option. `options` follow the [`HuffYuvEncoderOptions`] schema.
/// `output_params()` on the returned encoder carries the AVI `strf`
/// payload as `extradata`; it is final from construction for a fixed
/// method with `classic-v2` / `v1x` modes, and finalized by the first
/// `send_frame` when the method is auto-selected or the mode is
/// `custom-v2` (whose Huffman tables derive from the first frame).
pub fn make_encoder(params: &CodecParameters) -> CoreResult<Box<dyn Encoder>> {
    let opts: HuffYuvEncoderOptions = parse_options(&params.options)?;
    let width = params.width.filter(|w| *w > 0).ok_or_else(|| {
        CoreError::invalid("oxideav-huffyuv: encoder needs a non-zero CodecParameters::width")
    })?;
    let height = params.height.filter(|h| *h > 0).ok_or_else(|| {
        CoreError::invalid("oxideav-huffyuv: encoder needs a non-zero CodecParameters::height")
    })?;
    let family = match (opts.format, params.pixel_format) {
        (Some(f), Some(pf)) => {
            if family_from_pixel_format(pf) != Some(f) {
                return Err(CoreError::invalid(format!(
                    "oxideav-huffyuv: `format` option ({f:?}) conflicts with \
                     CodecParameters::pixel_format ({pf:?})"
                )));
            }
            f
        }
        (Some(f), None) => f,
        (None, Some(pf)) => family_from_pixel_format(pf).ok_or_else(|| {
            CoreError::unsupported(format!(
                "oxideav-huffyuv: unsupported input pixel format {pf:?} \
                 (supported: Yuyv422, Bgr24, Bgra)"
            ))
        })?,
        (None, None) => {
            return Err(CoreError::invalid(
                "oxideav-huffyuv: encoder needs an input family — set \
                 CodecParameters::pixel_format or the `format` option",
            ));
        }
    };
    if family == PixelFamily::Yuy2 && width % 2 != 0 {
        return Err(CoreError::invalid(
            "oxideav-huffyuv: YUY2 width must be even (spec/02 §3.1 macropixel pairs)",
        ));
    }
    if let Some(m) = opts.method {
        let legal = if family.is_rgb() {
            m.is_rgb_legal()
        } else {
            m.is_yuv_legal()
        };
        if !legal {
            return Err(CoreError::invalid(format!(
                "oxideav-huffyuv: method {m:?} is not legal for {family:?} (spec/01 §3.1)"
            )));
        }
    }

    let field_coded = opts
        .field_coding
        .resolve(height, opts.mode)
        .map_err(|e| CoreError::invalid(format!("oxideav-huffyuv: {e}")))?;

    let mut output = CodecParameters::video(params.codec_id.clone());
    output.width = Some(width);
    output.height = Some(height);
    output.pixel_format = Some(pixel_format_for_family(family));
    output.frame_rate = params.frame_rate;
    // The encoder always writes the classic VfW FourCC into
    // biCompression (spec/01 §1.5) — advertise the same tag so muxers
    // don't have to grope the strf.
    output.tag = Some(CodecTag::fourcc(b"HFYU"));
    // Stream extradata is knowable up front exactly when the method is
    // fixed and the tables aren't derived from frame content.
    if let Some(m) = opts.method {
        match opts.mode {
            ExtradataMode::ClassicV2 => {
                output.extradata = build_bitmapinfoheader_field_coded(
                    family,
                    m,
                    width,
                    height,
                    classic_blob_bytes(family, m),
                    true,
                    field_coded,
                );
            }
            ExtradataMode::V1xCompat => {
                output.extradata = build_bitmapinfoheader_field_coded(
                    family,
                    m,
                    width,
                    height,
                    &[],
                    false,
                    field_coded,
                );
            }
            // Finalized by the first send_frame.
            ExtradataMode::CustomV2 => {}
        }
    }

    Ok(Box::new(HuffYuvEncoder {
        family,
        width,
        height,
        mode: opts.mode,
        field_coding: opts.field_coding,
        field_coded,
        chosen_method: opts.method,
        pinned_custom: None,
        time_base: packet_time_base(params.frame_rate),
        output,
        pending: None,
        eof: false,
        worker_budget: 1,
    }))
}

struct HuffYuvEncoder {
    family: PixelFamily,
    width: u32,
    height: u32,
    mode: ExtradataMode,
    /// The caller's field-coding choice and its resolution for this
    /// stream's height and mode (spec/06 §1).
    field_coding: FieldCoding,
    field_coded: bool,
    /// The stream's method. `Some` from construction for a fixed
    /// `method` option; pinned by the first frame's auto-selection
    /// otherwise (the method byte is stream-level, spec/01 §3).
    chosen_method: Option<Method>,
    /// `CustomV2` stream state: the first frame's `(strf, tables)`,
    /// reused for every subsequent frame so the wire stays decodable
    /// against the single stream-level extradata blob.
    pinned_custom: Option<(Vec<u8>, Arc<ThreeTables>)>,
    time_base: TimeBase,
    /// Muxer-facing stream description ([`Encoder::output_params`]).
    /// `extradata` finalizes at construction or on the first
    /// `send_frame` — see [`make_encoder`].
    output: CodecParameters,
    pending: Option<Packet>,
    eof: bool,
    /// Advisory worker budget from [`Encoder::set_execution_context`].
    /// Defaults to 1 — per the `ExecutionContext` threading contract
    /// the codec runs serial until told otherwise. Budgets ≥ 2 engage
    /// the piecewise frame encode (residuals and bit packing split
    /// across workers, spliced bit-exactly) and the parallel
    /// auto-selector scoring; every fan-out clamps inline to
    /// `threads.min(units).max(1)`.
    worker_budget: usize,
}

impl HuffYuvEncoder {
    /// Encode one raster through the same engine the direct API uses,
    /// pinning stream-level state (method, custom tables) on the first
    /// frame. Returns `(strf, wire bytes)` — the strf is byte-identical
    /// for every frame of the stream.
    fn encode_pixels(&mut self, pixels: &[u8]) -> CoreResult<(Vec<u8>, Vec<u8>)> {
        let map = |e: crate::error::Error| CoreError::invalid(format!("oxideav-huffyuv: {e}"));
        let options = EncodeOptions::new()
            .with_mode(self.mode)
            .with_field_coding(self.field_coding)
            .with_worker_budget(self.worker_budget);
        // CustomV2 after the first frame: emit against the pinned
        // stream tables.
        if let (Some(m), Some((strf, tabs))) = (self.chosen_method, &self.pinned_custom) {
            let shape = FrameShape {
                family: self.family,
                method: m,
                width: self.width,
                height: self.height,
                field_coded: self.field_coded,
                cover_all: false,
            };
            let bytes = encode_body_with_pinned_tables(&shape, pixels, tabs, self.worker_budget)
                .map_err(map)?;
            return Ok((strf.clone(), bytes));
        }
        let selection = match self.chosen_method {
            Some(m) => MethodSelection::Fixed(m),
            None => MethodSelection::Auto,
        };
        let encoded = encode_stream_opening_frame(
            self.family,
            self.width,
            self.height,
            pixels,
            &options.with_method(selection),
        )
        .map_err(map)?;
        self.chosen_method = Some(encoded.method);
        if self.mode == ExtradataMode::CustomV2 {
            self.pin_custom_tables(&encoded.strf)?;
        }
        Ok((encoded.strf, encoded.frame))
    }

    /// Pin the `CustomV2` stream tables from the first frame's strf.
    /// The v2.x layout puts the RLE-compressed length tables at
    /// `+0x2C` (spec/01 §5) — the same region
    /// [`StreamConfig::parse_bitmapinfoheader`] hands the decoder.
    fn pin_custom_tables(&mut self, strf: &[u8]) -> CoreResult<()> {
        let table_bytes = strf.get(0x2C..).filter(|t| !t.is_empty()).ok_or_else(|| {
            CoreError::other("oxideav-huffyuv: internal error — CustomV2 strf without extradata")
        })?;
        let tabs = table_cache::extradata_tables(table_bytes)
            .map_err(|e| CoreError::invalid(format!("oxideav-huffyuv: {e}")))?;
        self.pinned_custom = Some((strf.to_vec(), tabs));
        Ok(())
    }
}

impl Encoder for HuffYuvEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.output.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.output
    }

    fn send_frame(&mut self, frame: &Frame) -> CoreResult<()> {
        if self.pending.is_some() {
            return Err(CoreError::other(
                "oxideav-huffyuv: receive_packet must be called before sending another frame",
            ));
        }
        if self.eof {
            return Err(CoreError::other("oxideav-huffyuv: send_frame after flush"));
        }
        let Frame::Video(vf) = frame else {
            return Err(CoreError::invalid(
                "oxideav-huffyuv: encoder accepts video frames only",
            ));
        };
        let planes = vf.image_planes();
        if planes.len() != 1 {
            return Err(CoreError::invalid(format!(
                "oxideav-huffyuv: expected 1 packed image plane, got {}",
                planes.len()
            )));
        }
        let plane = &planes[0];
        let row_bytes = self.family.row_bytes(self.width);
        let rows = self.height as usize;
        if plane.stride < row_bytes {
            return Err(CoreError::invalid(format!(
                "oxideav-huffyuv: plane stride {} < packed row bytes {row_bytes} for \
                 {:?} at width {}",
                plane.stride, self.family, self.width
            )));
        }
        // Rows may carry padding beyond the visible width; the last
        // row only needs its visible bytes.
        let needed = plane.stride * (rows - 1) + row_bytes;
        if plane.data.len() < needed {
            return Err(CoreError::invalid(format!(
                "oxideav-huffyuv: plane size {} < expected {needed} ({}×{} {:?}, stride {})",
                plane.data.len(),
                self.width,
                self.height,
                self.family,
                plane.stride
            )));
        }
        // Wire raster: tightly packed rows, top row first for YUY2 and
        // bottom row first for the RGB families (spec/02 §2).
        let gathered: Vec<u8>;
        let pixels: &[u8] = if !self.family.is_rgb() && plane.stride == row_bytes {
            &plane.data[..needed]
        } else {
            let mut buf = Vec::with_capacity(row_bytes * rows);
            let mut push = |r: usize| {
                let start = r * plane.stride;
                buf.extend_from_slice(&plane.data[start..start + row_bytes]);
            };
            if self.family.is_rgb() {
                (0..rows).rev().for_each(&mut push);
            } else {
                (0..rows).for_each(&mut push);
            }
            gathered = buf;
            &gathered
        };
        let (strf, bytes) = self.encode_pixels(pixels)?;
        if self.output.extradata.is_empty() {
            self.output.extradata = strf;
        } else if self.output.extradata != strf {
            return Err(CoreError::other(
                "oxideav-huffyuv: internal error — per-frame strf diverged from the \
                 stream extradata",
            ));
        }
        let mut pkt = Packet::new(0, self.time_base, bytes).with_keyframe(true);
        // Intra-only codec: decode order == display order.
        pkt.pts = vf.pts;
        pkt.dts = vf.pts;
        self.pending = Some(pkt);
        Ok(())
    }

    fn receive_packet(&mut self) -> CoreResult<Packet> {
        match self.pending.take() {
            Some(pkt) => Ok(pkt),
            None if self.eof => Err(CoreError::Eof),
            None => Err(CoreError::NeedMore),
        }
    }

    fn flush(&mut self) -> CoreResult<()> {
        self.eof = true;
        Ok(())
    }

    fn set_execution_context(&mut self, ctx: &ExecutionContext) {
        // Store the advisory budget; `ctx.threads` is documented ≥ 1
        // but clamp defensively so a zero can never disable encode.
        self.worker_budget = ctx.threads.max(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::{ProbeContext, RuntimeContext};

    #[test]
    fn register_via_runtime_context_installs_codec() {
        let mut ctx = RuntimeContext::new();
        register(&mut ctx);
        let codec_id = CodecId::new(CODEC_ID_STR);
        assert!(ctx.codecs.has_decoder(&codec_id));
        // Round 429: dual-API closure — the registration carries the
        // encoder factory and its options schema too.
        assert!(ctx.codecs.has_encoder(&codec_id));
        assert!(ctx.codecs.encoder_options_schema(&codec_id).is_some());
    }

    /// `set_execution_context` end-to-end through the `Decoder` trait
    /// object: a budget-2 registry decoder must emit byte-identical
    /// frames to a default (serial) one on field-coded and plain
    /// streams alike.
    #[test]
    fn round420_execution_context_budget_output_invariant() {
        use crate::encoder::{ExtradataMode, encode_frame_with_mode};
        use crate::header::Method;
        use oxideav_core::TimeBase;

        for (w, h, label) in [(32u32, 292u32, "interlaced"), (48, 32, "progressive")] {
            // Deterministic synthetic YUY2 raster.
            let mut s: u32 = 0x0420_c0de;
            let mut pixels = vec![0u8; (w as usize) * (h as usize) * 2];
            for slot in pixels.iter_mut() {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                *slot = s as u8;
            }
            let (strf, frame_bytes) = encode_frame_with_mode(
                crate::header::PixelFamily::Yuy2,
                Method::Median,
                w,
                h,
                &pixels,
                ExtradataMode::ClassicV2,
            )
            .expect("encode");

            let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
            params.extradata = strf.clone();

            let decode_with = |budget: Option<usize>| -> Vec<Vec<u8>> {
                let mut dec = make_decoder(&params).expect("make_decoder");
                if let Some(threads) = budget {
                    dec.set_execution_context(&ExecutionContext { threads });
                }
                let pkt = Packet::new(0, TimeBase::new(1, 25), frame_bytes.clone());
                dec.send_packet(&pkt).expect("send_packet");
                let Frame::Video(vf) = dec.receive_frame().expect("receive_frame") else {
                    panic!("expected a video frame");
                };
                vf.planes.into_iter().map(|p| p.data).collect()
            };

            let serial = decode_with(None);
            let budget1 = decode_with(Some(1));
            let budget2 = decode_with(Some(2));
            let budget8 = decode_with(Some(8));
            assert_eq!(serial, budget1, "{label}: explicit budget-1 diverged");
            assert_eq!(serial, budget2, "{label}: budget-2 diverged");
            assert_eq!(serial, budget8, "{label}: clamped budget-8 diverged");
            assert_eq!(
                serial[0], pixels,
                "{label}: registry decode must be lossless"
            );
        }
    }

    // ─────────────── round-429 encoder-side trait tests ───────────────

    fn synth_pixels(width: u32, height: u32, family: PixelFamily, salt: u32) -> Vec<u8> {
        let mut s: u32 = 0x0429_c0de ^ salt;
        let mut pixels = vec![0u8; family.row_bytes(width) * height as usize];
        for slot in pixels.iter_mut() {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            *slot = s as u8;
        }
        pixels
    }

    /// The raster the direct API takes for a framework frame: rows as
    /// they are for YUY2, bottom row first for RGB (spec/02 §2).
    fn wire_raster(family: PixelFamily, width: u32, top_down: &[u8]) -> Vec<u8> {
        if family.is_rgb() {
            reverse_rows(top_down, family.row_bytes(width))
        } else {
            top_down.to_vec()
        }
    }

    fn family_opt(f: PixelFamily) -> &'static str {
        match f {
            PixelFamily::Yuy2 => "yuy2",
            PixelFamily::Rgb24 => "rgb24",
            PixelFamily::Rgb32 => "rgb32",
        }
    }

    fn method_opt(m: Method) -> &'static str {
        match m {
            Method::PredictOld => "predict-old",
            Method::Left => "left",
            Method::Gradient => "gradient",
            Method::Median => "median",
            Method::LeftDecorr => "left-decorr",
            Method::GradientDecorr => "gradient-decorr",
        }
    }

    fn mode_opt(m: ExtradataMode) -> &'static str {
        match m {
            ExtradataMode::ClassicV2 => "classic-v2",
            ExtradataMode::CustomV2 => "custom-v2",
            ExtradataMode::V1xCompat => "v1x",
        }
    }

    fn encoder_params(
        family: PixelFamily,
        width: u32,
        height: u32,
        method: Option<Method>,
        mode: ExtradataMode,
    ) -> CodecParameters {
        let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        params.width = Some(width);
        params.height = Some(height);
        params.options.insert("format", family_opt(family));
        params.options.insert("mode", mode_opt(mode));
        if let Some(m) = method {
            params.options.insert("method", method_opt(m));
        }
        params
    }

    fn video_frame(family: PixelFamily, width: u32, pixels: Vec<u8>, pts: i64) -> Frame {
        Frame::Video(VideoFrame {
            pts: Some(pts),
            planes: vec![VideoPlane {
                stride: family.row_bytes(width),
                data: pixels,
            }],
        })
    }

    /// Push one frame through a boxed trait encoder, returning the
    /// emitted packet.
    fn encode_one(enc: &mut Box<dyn Encoder>, frame: &Frame) -> Packet {
        enc.send_frame(frame).expect("send_frame");
        enc.receive_packet().expect("receive_packet")
    }

    fn legal_methods(family: PixelFamily) -> Vec<Method> {
        [
            Method::PredictOld,
            Method::Left,
            Method::Gradient,
            Method::Median,
            Method::LeftDecorr,
            Method::GradientDecorr,
        ]
        .into_iter()
        .filter(|m| {
            if family.is_rgb() {
                m.is_rgb_legal()
            } else {
                m.is_yuv_legal()
            }
        })
        .collect()
    }

    /// Round-429 dual-API invariance: for the full legal
    /// (family, method, mode) matrix at a progressive and an
    /// interlaced size, the trait path emits byte-identical strf +
    /// wire bytes to the direct API, and the packet decodes
    /// losslessly through the registry decoder built from
    /// `output_params()`.
    #[test]
    fn round429_trait_encode_matches_direct_api() {
        use crate::encoder::encode_frame_with_mode;
        for (w, h) in [(16u32, 8u32), (16, 290)] {
            for family in [PixelFamily::Yuy2, PixelFamily::Rgb24, PixelFamily::Rgb32] {
                let pixels = synth_pixels(w, h, family, w ^ h);
                for method in legal_methods(family) {
                    for mode in [
                        ExtradataMode::ClassicV2,
                        ExtradataMode::CustomV2,
                        ExtradataMode::V1xCompat,
                    ] {
                        let scenario = format!("{family:?}/{method:?}/{mode:?}/{w}x{h}");
                        let (strf, bytes) = encode_frame_with_mode(
                            family,
                            method,
                            w,
                            h,
                            &wire_raster(family, w, &pixels),
                            mode,
                        )
                        .expect("direct encode");
                        let params = encoder_params(family, w, h, Some(method), mode);
                        let mut enc = make_encoder(&params).expect("make_encoder");
                        let pkt = encode_one(&mut enc, &video_frame(family, w, pixels.clone(), 0));
                        if mode == ExtradataMode::CustomV2 {
                            // The stream encoder's tables cover every
                            // symbol (they must serve later frames), so
                            // its bytes differ from the single-frame
                            // optimum — by well under 1 % plus the
                            // word rounding on these small rasters.
                            let slack = bytes.len() / 20 + 64;
                            assert!(
                                pkt.data.len() <= bytes.len() + slack,
                                "{scenario}: covering tables cost {} vs {}",
                                pkt.data.len(),
                                bytes.len()
                            );
                        } else {
                            assert_eq!(pkt.data, bytes, "{scenario}: wire bytes diverged");
                            assert_eq!(
                                enc.output_params().extradata,
                                strf,
                                "{scenario}: strf diverged"
                            );
                        }
                        assert!(pkt.flags.keyframe, "{scenario}: intra frame not keyframed");

                        // Full registry loop: decoder built from the
                        // encoder's own output params.
                        let mut dec = make_decoder(enc.output_params()).expect("make_decoder");
                        dec.send_packet(&pkt).expect("send_packet");
                        let Frame::Video(vf) = dec.receive_frame().expect("receive_frame") else {
                            panic!("{scenario}: expected a video frame");
                        };
                        assert_eq!(
                            vf.planes[0].data, pixels,
                            "{scenario}: registry loop not lossless"
                        );
                    }
                }
            }
        }
    }

    /// Round-429: the trait encoder honours `set_execution_context`
    /// with byte-identical output across budgets — fixed-method
    /// frames treat the budget as a no-op, and the auto-selector's
    /// parallel candidate scoring changes nothing on the wire.
    #[test]
    fn round429_trait_budget_output_invariant() {
        for (w, h, label) in [(16u32, 290u32, "interlaced"), (16, 8, "progressive")] {
            for method in [Some(Method::Median), None] {
                let family = PixelFamily::Yuy2;
                let pixels = synth_pixels(w, h, family, 0xb0d6e7);
                let encode_with = |budget: Option<usize>| -> (Vec<u8>, Vec<u8>) {
                    let params = encoder_params(family, w, h, method, ExtradataMode::ClassicV2);
                    let mut enc = make_encoder(&params).expect("make_encoder");
                    if let Some(threads) = budget {
                        enc.set_execution_context(&ExecutionContext { threads });
                    }
                    let pkt = encode_one(&mut enc, &video_frame(family, w, pixels.clone(), 0));
                    (enc.output_params().extradata.clone(), pkt.data)
                };
                let serial = encode_with(None);
                for budget in [1usize, 2, 8] {
                    assert_eq!(
                        encode_with(Some(budget)),
                        serial,
                        "{label}/method={method:?}: budget-{budget} diverged"
                    );
                }
            }
        }
    }

    /// The `field-coding` option reaches the wire: the interlace byte
    /// follows the choice (spec/06 §1.2), every frame of the stream
    /// decodes through the registry decoder, and a choice a v1.x
    /// header cannot express is rejected at construction.
    #[test]
    fn field_coding_option_sets_the_interlace_byte() {
        for (choice, byte) in [("fields", 0x10u8), ("plain", 0x20), ("auto", 0x20)] {
            for mode in [ExtradataMode::ClassicV2, ExtradataMode::CustomV2] {
                for family in [PixelFamily::Yuy2, PixelFamily::Rgb24] {
                    let (w, h) = (16u32, 12u32);
                    let mut params = encoder_params(family, w, h, None, mode);
                    params.options.insert("field-coding", choice);
                    let mut enc = make_encoder(&params).expect("make_encoder");
                    let picture = synth_pixels(w, h, family, 7);
                    let mut packets = Vec::new();
                    for pts in 0..2 {
                        packets.push(encode_one(
                            &mut enc,
                            &video_frame(family, w, picture.clone(), pts),
                        ));
                    }
                    let strf = enc.output_params().extradata.clone();
                    assert_eq!(strf[0x2A], byte, "{choice}/{mode:?}/{family:?}");
                    let mut dec = make_decoder(enc.output_params()).expect("make_decoder");
                    for pkt in &packets {
                        dec.send_packet(pkt).expect("send_packet");
                        let Frame::Video(vf) = dec.receive_frame().expect("receive_frame") else {
                            panic!("expected a video frame");
                        };
                        assert_eq!(vf.planes[0].data, picture, "{choice}/{mode:?}/{family:?}");
                    }
                }
            }
        }
        let mut params = encoder_params(PixelFamily::Yuy2, 16, 12, None, ExtradataMode::V1xCompat);
        params.options.insert("field-coding", "fields");
        assert!(make_encoder(&params).is_err());
    }

    /// Round-429: `method=auto` (the default) picks the bit-cost
    /// winner on the first frame and pins it for the stream — the
    /// first packet is byte-identical to the direct auto-selector,
    /// subsequent frames use the pinned method, and every packet
    /// decodes against the single stream-level extradata.
    #[test]
    fn round429_auto_method_pins_stream() {
        use crate::encoder::{encode_frame_auto, encode_frame_with_mode};
        for family in [PixelFamily::Yuy2, PixelFamily::Rgb32] {
            let (w, h) = (16u32, 12u32);
            let frame1 = synth_pixels(w, h, family, 1);
            let frame2 = synth_pixels(w, h, family, 2);
            let (strf, bytes1, chosen) = encode_frame_auto(
                family,
                MethodSelection::Auto,
                w,
                h,
                &wire_raster(family, w, &frame1),
                ExtradataMode::ClassicV2,
            )
            .expect("direct auto");
            let (_, bytes2) = encode_frame_with_mode(
                family,
                chosen,
                w,
                h,
                &wire_raster(family, w, &frame2),
                ExtradataMode::ClassicV2,
            )
            .expect("direct pinned second frame");

            let params = encoder_params(family, w, h, None, ExtradataMode::ClassicV2);
            let mut enc = make_encoder(&params).expect("make_encoder");
            assert!(
                enc.output_params().extradata.is_empty(),
                "auto extradata must finalize on the first frame, not before"
            );
            let pkt1 = encode_one(&mut enc, &video_frame(family, w, frame1.clone(), 0));
            assert_eq!(pkt1.data, bytes1, "{family:?}: first frame ≠ direct auto");
            assert_eq!(enc.output_params().extradata, strf);
            let pkt2 = encode_one(&mut enc, &video_frame(family, w, frame2.clone(), 1));
            assert_eq!(
                pkt2.data, bytes2,
                "{family:?}: second frame not pinned to the auto winner"
            );

            // Both packets decode against the one stream extradata.
            for (pkt, pixels) in [(&pkt1, &frame1), (&pkt2, &frame2)] {
                let mut dec = make_decoder(enc.output_params()).expect("make_decoder");
                dec.send_packet(pkt).expect("send_packet");
                let Frame::Video(vf) = dec.receive_frame().expect("receive_frame") else {
                    panic!("expected a video frame");
                };
                assert_eq!(
                    &vf.planes[0].data, pixels,
                    "{family:?}: stream not lossless"
                );
            }
        }
    }

    /// `custom-v2` streams pin the first frame's Huffman tables
    /// (extradata is stream-level, spec/02 §1.2): repeat frames
    /// re-encode byte-identically, and because the pinned tables give
    /// every symbol a code, a later frame with completely different
    /// content — here noise after a flat first frame, the worst case —
    /// still encodes and decodes against the same header.
    #[test]
    fn custom_v2_stream_tables_cover_later_frames() {
        for family in [PixelFamily::Yuy2, PixelFamily::Rgb24, PixelFamily::Rgb32] {
            let (w, h) = (16u32, 8u32);
            let flat = vec![0x80u8; family.row_bytes(w) * h as usize];
            let noisy = synth_pixels(w, h, family, 3);

            let params = encoder_params(family, w, h, None, ExtradataMode::CustomV2);
            let mut enc = make_encoder(&params).expect("make_encoder");
            assert!(
                enc.output_params().extradata.is_empty(),
                "custom-v2 extradata must finalize on the first frame"
            );
            let pkt1 = encode_one(&mut enc, &video_frame(family, w, flat.clone(), 0));
            let strf = enc.output_params().extradata.clone();
            assert!(!strf.is_empty());
            // Every symbol of every table has a code.
            let lengths =
                crate::tables::rle_decode_three_channels(&strf[0x2C..]).expect("stream tables");
            assert!(lengths.iter().all(|t| t.iter().all(|&l| l != 0)));

            // Identical content → identical wire bytes, same header.
            let pkt2 = encode_one(&mut enc, &video_frame(family, w, flat.clone(), 1));
            assert_eq!(pkt1.data, pkt2.data, "pinned tables must be deterministic");
            // Unrelated content → still the same header.
            let pkt3 = encode_one(&mut enc, &video_frame(family, w, noisy.clone(), 2));
            assert_eq!(enc.output_params().extradata, strf);

            let mut dec = make_decoder(enc.output_params()).expect("make_decoder");
            for (pkt, want) in [(&pkt1, &flat), (&pkt2, &flat), (&pkt3, &noisy)] {
                dec.send_packet(pkt).expect("send_packet");
                let Frame::Video(vf) = dec.receive_frame().expect("receive_frame") else {
                    panic!("expected a video frame");
                };
                assert_eq!(
                    &vf.planes[0].data, want,
                    "{family:?}: custom-v2 stream not lossless"
                );
            }
        }
    }

    /// What the stream-level `custom-v2` tables cost, on picture-like
    /// content (smooth ramp + a few bits of noise): against the exact
    /// per-frame optimum for the frame they were fitted to, and against
    /// the classic tables for a later frame that looks nothing like it
    /// (the stream opens on a flat frame).
    #[test]
    fn custom_v2_stream_tables_cost_is_bounded() {
        use crate::encoder::encode_frame_with_mode;
        let picture = |w: u32, h: u32, family: PixelFamily, seed: u32| -> Vec<u8> {
            let bpp = family.bytes_per_pixel_step();
            let row = family.row_bytes(w);
            let mut state = seed | 1;
            (0..row * h as usize)
                .map(|i| {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    let base = (i / row + (i % row) / bpp) / 2 + (i % bpp) * 9;
                    (base as u32).wrapping_add(state & 7) as u8
                })
                .collect()
        };
        for family in [PixelFamily::Yuy2, PixelFamily::Rgb24, PixelFamily::Rgb32] {
            let (w, h) = (96u32, 64u32);
            let method = if family.is_rgb() {
                Method::GradientDecorr
            } else {
                Method::Median
            };
            let first = picture(w, h, family, 11);
            let size = |pixels: &[u8], mode| {
                encode_frame_with_mode(family, method, w, h, &wire_raster(family, w, pixels), mode)
                    .expect("direct encode")
                    .1
                    .len()
            };
            let optimum = size(&first, ExtradataMode::CustomV2);
            let classic = size(&first, ExtradataMode::ClassicV2);

            // Fitted frame: within 4 % of the per-frame optimum, and
            // smaller than the classic tables.
            let params = encoder_params(family, w, h, Some(method), ExtradataMode::CustomV2);
            let mut enc = make_encoder(&params).expect("make_encoder");
            let fitted = encode_one(&mut enc, &video_frame(family, w, first.clone(), 0))
                .data
                .len();
            assert!(
                fitted * 100 <= optimum * 104,
                "{family:?}: fitted frame {fitted} vs optimum {optimum}"
            );
            assert!(
                fitted < classic,
                "{family:?}: {fitted} vs classic {classic}"
            );

            // Mismatched stream: tables fitted to a flat frame, then
            // the picture. Never worse than 1.5× the classic tables.
            let mut enc = make_encoder(&params).expect("make_encoder");
            let flat = vec![0x80u8; first.len()];
            encode_one(&mut enc, &video_frame(family, w, flat, 0));
            let mismatched = encode_one(&mut enc, &video_frame(family, w, first.clone(), 1))
                .data
                .len();
            assert!(
                mismatched * 2 <= classic * 3,
                "{family:?}: mismatched frame {mismatched} vs classic {classic}"
            );
            eprintln!(
                "{family:?}: optimum {optimum}, stream tables {fitted}, classic {classic}, \
                 after a flat opening frame {mismatched}"
            );
        }
    }

    /// Round-429: factory-time validation surface.
    #[test]
    fn round429_make_encoder_validation() {
        let base = |mutate: &dyn Fn(&mut CodecParameters)| {
            let mut params = encoder_params(
                PixelFamily::Yuy2,
                16,
                8,
                Some(Method::Left),
                ExtradataMode::ClassicV2,
            );
            mutate(&mut params);
            make_encoder(&params)
        };
        // Baseline constructs.
        assert!(base(&|_| {}).is_ok());
        // Unknown option key.
        assert!(base(&|p| p.options.insert("nope", "1")).is_err());
        // Bad enum value.
        assert!(base(&|p| p.options.insert("mode", "v2")).is_err());
        // Missing dimensions.
        assert!(base(&|p| p.width = None).is_err());
        assert!(base(&|p| p.height = Some(0)).is_err());
        // Odd YUY2 width.
        assert!(base(&|p| p.width = Some(15)).is_err());
        // `format` option conflicting with pixel_format.
        assert!(base(&|p| p.pixel_format = Some(PixelFormat::Bgr24)).is_err());
        // Matching pixel_format is fine.
        assert!(base(&|p| p.pixel_format = Some(PixelFormat::Yuyv422)).is_ok());
        // Method illegal for the family.
        assert!(base(&|p| p.options.insert("method", "median")).is_ok());
        assert!(
            base(&|p| {
                p.options.insert("format", "rgb24");
                p.options.insert("method", "median");
            })
            .is_err()
        );

        // No family from anywhere.
        let mut bare = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        bare.width = Some(16);
        bare.height = Some(8);
        assert!(make_encoder(&bare).is_err());
        // Unmappable pixel_format.
        bare.pixel_format = Some(PixelFormat::Yuv420P);
        assert!(make_encoder(&bare).is_err());
        // Mappable pixel_format alone resolves the family.
        bare.pixel_format = Some(PixelFormat::Bgra);
        let enc = make_encoder(&bare).expect("family from pixel_format");
        assert_eq!(
            enc.output_params().pixel_format,
            Some(PixelFormat::Bgra),
            "output params must advertise the input family"
        );
    }

    /// Round-429: send/receive lifecycle + packet metadata.
    #[test]
    fn round429_encoder_lifecycle_and_metadata() {
        let family = PixelFamily::Yuy2;
        let (w, h) = (16u32, 8u32);
        let pixels = synth_pixels(w, h, family, 4);
        let mut params = encoder_params(family, w, h, Some(Method::Left), ExtradataMode::ClassicV2);
        params.frame_rate = Some(Rational::new(30, 1));
        let mut enc = make_encoder(&params).expect("make_encoder");

        // Fixed method + classic-v2: extradata is final from construction.
        assert!(!enc.output_params().extradata.is_empty());
        assert_eq!(enc.output_params().width, Some(w));
        assert_eq!(enc.output_params().height, Some(h));
        assert_eq!(enc.codec_id().as_str(), CODEC_ID_STR);
        assert_eq!(
            enc.output_params().tag,
            Some(CodecTag::fourcc(b"HFYU")),
            "muxer-facing tag must be the classic VfW FourCC"
        );

        // Empty before any frame.
        assert!(matches!(enc.receive_packet(), Err(CoreError::NeedMore)));

        let frame = video_frame(family, w, pixels.clone(), 42);
        enc.send_frame(&frame).expect("send_frame");
        // Double-send without draining is an error.
        assert!(enc.send_frame(&frame).is_err());
        let pkt = enc.receive_packet().expect("receive_packet");
        assert_eq!(pkt.pts, Some(42), "pts must pass through verbatim");
        assert_eq!(pkt.dts, Some(42), "intra-only: dts == pts");
        assert!(pkt.flags.keyframe);
        assert_eq!(
            (pkt.time_base.0.num, pkt.time_base.0.den),
            (1, 30),
            "declared frame rate must become the packet time base"
        );
        assert!(matches!(enc.receive_packet(), Err(CoreError::NeedMore)));

        // Wrong-shaped inputs are rejected with the state unchanged.
        let bad_stride = Frame::Video(VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: family.row_bytes(w) + 2,
                data: pixels.clone(),
            }],
        });
        assert!(enc.send_frame(&bad_stride).is_err());
        let bad_len = Frame::Video(VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: family.row_bytes(w),
                data: pixels[..pixels.len() - 2].to_vec(),
            }],
        });
        assert!(enc.send_frame(&bad_len).is_err());

        // Flush → Eof; sends after flush are rejected.
        enc.flush().expect("flush");
        assert!(matches!(enc.receive_packet(), Err(CoreError::Eof)));
        assert!(enc.send_frame(&frame).is_err());
    }

    #[test]
    fn register_claims_native_fourccs() {
        let mut reg = CodecRegistry::new();
        register_codecs(&mut reg);
        for tag in [CodecTag::fourcc(b"HFYU"), CodecTag::fourcc(b"FFVH")] {
            let resolved = reg
                .resolve_tag_ref(&ProbeContext::new(&tag))
                .map(|c| c.as_str());
            assert_eq!(resolved, Some(CODEC_ID_STR));
        }
    }
}
