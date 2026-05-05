//! Pure-Rust HuffYUV / FFVHuff lossless intra-only video decoder.
//!
//! This crate decodes both `huffyuv` (FourCC `HFYU`) and `ffvhuff`
//! (FourCC `FFVH`) bitstreams from the AVI container, following the
//! reverse-engineered spec at
//! `docs/video/huffyuv/huffyuv-trace-reverse-engineering.md`. No upstream
//! source code or built-in Huffman tables were consulted; the decoder
//! derives every Huffman code from the canonical convention applied to
//! the lengths shipped on the wire (RLE-coded, see `extradata`).
//!
//! **Supported pixel formats**:
//!
//! - YUV 4:2:0 / 4:2:2 (v2 packed-identifier extradata)
//! - YUV 4:4:4 / 4:2:2 / 4:2:0 / 4:1:1 (v3 planar extradata, 8-bit)
//! - YUV 4:4:4 / 4:2:2 / 4:2:0 planar at 10 / 12-bit (v3)
//! - Gray 8 / 10 / 12 / 16-bit (v3, incl. §5.5 2-raw-bit splice)
//! - GBRP / GBRAP planar at 8 / 10 / 12-bit (v3)
//! - RGB24 / BGRA packed (v2, with `decorrelate` flag)
//!
//! **Supported predictors**: LEFT, GRADIENT (called PLANE in the
//! original codec), MEDIAN. RGB v2 streams accept LEFT and PLANE only.
//!
//! **Supported variants**: v2 + v3 extradata, FFVHuff per-frame tables
//! (`-context 1`, signaled by `extra[2] & 0x40`), interlaced bootstrap
//! (row-1 fresh-field-start when `InterlaceMode::Interlaced` or
//! `AutoByHeight` with height > 288).
//!
//! **Not yet supported**:
//!
//! - Interlaced encoder path (decoder only).
//! - The `hymt` slice variant.

#![allow(clippy::needless_range_loop)]

pub mod bitreader;
pub mod bitwriter;
pub mod decoder;
pub mod encoder;
pub mod extradata;
pub mod huffman;
pub mod length_builder;
pub mod predictor;
pub mod rle;
pub mod rle_encode;

use oxideav_core::{CodecCapabilities, CodecId, CodecTag};
use oxideav_core::{CodecInfo, CodecRegistry};

pub const CODEC_ID_HUFFYUV: &str = "huffyuv";
pub const CODEC_ID_FFVHUFF: &str = "ffvhuff";

/// Register both `huffyuv` and `ffvhuff` decoder factories with the
/// supplied [`CodecRegistry`]. Both id-strings resolve to the same
/// decoder factory (the bitstreams differ only in extradata version /
/// FourCC, both of which are inspected at decode time).
pub fn register(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::video("huffyuv_sw")
        .with_lossless(true)
        .with_intra_only(true)
        .with_max_size(65535, 65535);
    reg.register(
        CodecInfo::new(CodecId::new(CODEC_ID_HUFFYUV))
            .capabilities(caps.clone())
            .decoder(decoder::make_decoder)
            .encoder(encoder::make_encoder)
            .tag(CodecTag::fourcc(b"HFYU")),
    );
    reg.register(
        CodecInfo::new(CodecId::new(CODEC_ID_FFVHUFF))
            .capabilities(caps)
            .decoder(decoder::make_decoder)
            .encoder(encoder::make_encoder)
            .tag(CodecTag::fourcc(b"FFVH")),
    );
}
