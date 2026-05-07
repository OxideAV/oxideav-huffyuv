//! Pure-Rust HuffYUV / FFVHuff lossless video codec.
//!
//! Round-1 clean-room rebuild against the strict-isolation workspace
//! at [`docs/video/huffyuv/`]. Decodes 8-bit YUY2 / RGB24 / RGB32
//! HuffYUV streams (FOURCCs `HFYU` and `FFVH`) carrying any of the
//! six legal predictor + decorrelation methods:
//!
//! - `predict_old` (signed `-2`, equivalent to `Left`).
//! - `Left` (`0x00`).
//! - `Gradient` (`0x01`, YUV only).
//! - `Median` (`0x02`, YUV only).
//! - `LeftDecorr` (`0x40`, RGB only).
//! - `GradientDecorr` (`0x41`, RGB only).
//!
//! Both the v2.x extradata path (per-stream Huffman length tables
//! RLE-compressed inside the BIH) and the v1.x precomputed-codes
//! fallback (`biSize == 0x28`, length-table set A and code-template
//! set A/B from the proprietary's `.rdata`) are supported.
//!
//! # Pipeline
//!
//! 1. **BIH parse** ([`header::StreamConfig::parse_bitmapinfoheader`])
//!    — `spec/01 §1..§4`. Resolves family + method + extradata
//!    region.
//! 2. **Per-stream Huffman tables** ([`tables`]). v2.x: RLE-decode
//!    the three concatenated 256-byte length tables, then build
//!    canonical Huffman codes via the longest-length-first algorithm
//!    (`spec/03 §3` + `spec/04 §2.6`). v1.x: derive each per-symbol
//!    code from `(length, code-byte-template)` pairs (`spec/04 §4`).
//! 3. **Per-frame decode** ([`decoder::decode_frame`]). Reads the
//!    uncompressed first pixel verbatim, decodes per-pixel residuals
//!    in the correct slot order (`spec/02 §3` + corrections from
//!    `spec/03 §1.4`), then runs the predictor inverse: LEFT pass
//!    on the linear raster, optionally followed by gradient or
//!    median post-passes (`spec/03 §2`).
//! 4. **Decorrelation inverse** for RGB methods `0x40`/`0x41`:
//!    `B = (B-G) + G`, `R = (R-G) + G`.
//!
//! # Cargo features
//!
//! - **`registry`** (default): wire the crate into `oxideav-core`'s
//!   codec registry. Disabling via `default-features = false` skips
//!   the `oxideav-core` dependency entirely.
//!
//! [`docs/video/huffyuv/`]: https://github.com/OxideAV/docs/tree/master/video/huffyuv

#![forbid(unsafe_code)]

pub mod bitio;
pub mod decoder;
pub mod encoder;
pub mod error;
pub mod header;
pub mod predict;
#[cfg(feature = "registry")]
pub mod registry;
pub mod tables;

#[cfg(test)]
mod roundtrip_tests;

pub use crate::decoder::{decode_frame, DecodedFrame};
pub use crate::encoder::{
    build_bitmapinfoheader, encode_frame, encode_frame_with_mode, ExtradataMode,
};
pub use crate::error::{Error, Result};
pub use crate::header::{Method, PixelFamily, Predictor, StreamConfig};

#[cfg(feature = "registry")]
pub use crate::registry::{register, register_codecs, CODEC_ID_STR};

#[cfg(feature = "registry")]
oxideav_core::register!("oxideav-huffyuv", register);
