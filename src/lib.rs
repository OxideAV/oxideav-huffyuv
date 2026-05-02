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
//! **Supported pixel formats** (v2 → 8-bit, v3 → 8-bit only for now):
//!
//! - YUV 4:2:0 / 4:2:2 (v2 packed-identifier extradata)
//! - YUV 4:4:4 / 4:1:1 (v3 planar extradata)
//! - Gray 8 (v3)
//! - GBRP / GBRAP planar (v3, no decorrelate)
//! - RGB24 / BGRA packed (v2, with `decorrelate` flag)
//!
//! **Supported predictors**: LEFT, GRADIENT (called PLANE in the
//! original codec), MEDIAN. RGB v2 streams accept LEFT and PLANE only.
//!
//! **Supported variants**: v2 + v3 extradata, FFVHuff per-frame tables
//! (`-context 1`, signaled by `extra[2] & 0x40`).
//!
//! **Not yet supported**:
//!
//! - Bit depths above 8 (the v3 high-bit-depth tables and the
//!   15/16-bit Huffman + 2 raw bits split).
//! - Interlaced streams (the row-1 bootstrap differs).
//! - Encoder.
//! - The `hymt` slice variant.

#![allow(clippy::needless_range_loop)]

pub mod bitreader;
pub mod decoder;
pub mod extradata;
pub mod huffman;
pub mod predictor;
pub mod rle;

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
            .tag(CodecTag::fourcc(b"HFYU")),
    );
    reg.register(
        CodecInfo::new(CodecId::new(CODEC_ID_FFVHUFF))
            .capabilities(caps)
            .decoder(decoder::make_decoder)
            .tag(CodecTag::fourcc(b"FFVH")),
    );
}
