//! `oxideav-core` framework integration.
//!
//! Compiled only when the default-on `registry` feature is enabled.
//! Standalone consumers (`default-features = false`) skip this
//! module.

#![cfg(feature = "registry")]

use oxideav_core::{
    CodecCapabilities, CodecId, CodecInfo, CodecParameters, CodecRegistry, CodecTag, Decoder,
    Error as CoreError, ExecutionContext, Frame, MediaType, Packet, PixelFormat,
    Result as CoreResult, RuntimeContext, VideoFrame, VideoPlane,
};

use crate::decoder::{decode_frame_with_workers, DecodedFrame};
use crate::header::{PixelFamily, StreamConfig};

/// Canonical codec id.
pub const CODEC_ID_STR: &str = "huffyuv";

/// Register the HuffYUV / FFVHuff codec with `reg`.
///
/// Claims the two FourCCs the AVI demuxer needs to resolve a HuffYUV
/// stream — `HFYU` (the original VfW FourCC; spec/01 §1.5)
/// and `FFVH` (the FFVHuff extended-variant FourCC). The decoder in this crate
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
    /// the codec runs serial until told otherwise. Budgets ≥ 2 let
    /// interlaced streams decode their two independently-coded fields
    /// on parallel workers (round 420); the fan-out clamp
    /// (`threads.min(units).max(1)`, `units = 2` fields) lives at the
    /// decode site in [`decode_frame_with_workers`].
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

    /// Round-420: `set_execution_context` end-to-end through the
    /// `Decoder` trait object. A budget-2 registry decoder must emit
    /// byte-identical frames to a default (serial) one on an
    /// interlaced stream, and the budget must be a no-op on a
    /// progressive stream.
    #[test]
    fn round420_execution_context_budget_output_invariant() {
        use crate::encoder::{encode_frame_with_mode, ExtradataMode};
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
