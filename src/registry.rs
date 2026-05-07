//! `oxideav-core` framework integration.
//!
//! Compiled only when the default-on `registry` feature is enabled.
//! Standalone consumers (`default-features = false`) skip this
//! module.

#![cfg(feature = "registry")]

use oxideav_core::{
    CodecCapabilities, CodecId, CodecInfo, CodecParameters, CodecRegistry, CodecTag, Decoder,
    Error as CoreError, Frame, MediaType, Packet, PixelFormat, Result as CoreResult,
    RuntimeContext, VideoFrame, VideoPlane,
};

use crate::decoder::{decode_frame, DecodedFrame};
use crate::header::{PixelFamily, StreamConfig};

/// Canonical codec id.
pub const CODEC_ID_STR: &str = "huffyuv";

/// Register the HuffYUV / FFVHuff codec with `reg`.
///
/// Claims the two FourCCs the AVI demuxer needs to resolve a HuffYUV
/// stream — `HFYU` (the proprietary Windows VfW codec; spec/01 §1.5)
/// and `FFVH` (the FFmpeg-internal alias). The decoder in this crate
/// only handles the 8-bit family the round-1 deliverable certifies;
/// 10/12-bit FFVHuff variants are deferred to a future round.
pub fn register_codecs(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::video("huffyuv_sw")
        .with_decode()
        .with_lossless(true)
        .with_intra_only(true);
    reg.register(
        CodecInfo::new(CodecId::new(CODEC_ID_STR))
            .capabilities(caps)
            .decoder(make_decoder)
            .tags([CodecTag::fourcc(b"HFYU"), CodecTag::fourcc(b"FFVH")]),
    );
}

/// Unified entry point invoked by the macro-generated wrapper.
pub fn register(ctx: &mut RuntimeContext) {
    register_codecs(&mut ctx.codecs);
}

// ──────────────────────── Decoder impl ────────────────────────

fn make_decoder(params: &CodecParameters) -> CoreResult<Box<dyn Decoder>> {
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
    }))
}

struct HuffYuvDecoder {
    codec_id: CodecId,
    /// Parsed once at decoder-create time when `CodecParameters`
    /// carried a BIH/extradata blob.
    stream_config: Option<StreamConfig>,
    pending: Option<Packet>,
    eof: bool,
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
        let decoded = decode_frame(cfg, &pkt.data)
            .map_err(|e| CoreError::invalid(format!("oxideav-huffyuv: {e}")))?;
        Ok(Frame::Video(map_to_video_frame(decoded, pkt.pts)))
    }

    fn flush(&mut self) -> CoreResult<()> {
        self.eof = true;
        Ok(())
    }
}

fn map_to_video_frame(frame: DecodedFrame, pts: Option<i64>) -> VideoFrame {
    let stride = match frame.family {
        PixelFamily::Yuy2 => frame.width as usize * 2,
        PixelFamily::Rgb24 => frame.width as usize * 3,
        PixelFamily::Rgb32 => frame.width as usize * 4,
    };
    let planes = vec![VideoPlane {
        stride,
        data: frame.pixels,
    }];
    let _ = MediaType::Video;
    let _ = PixelFormat::Yuv420P;
    VideoFrame { pts, planes }
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
