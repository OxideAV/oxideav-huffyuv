//! HuffYUV / FFVHuff codec-private extradata parser.
//!
//! Layout (trace doc §2.2):
//!
//! - 4 fixed bytes — predictor & decorrelate, format byte, flags, version.
//! - N RLE-coded code-length tables (N depends on the channel layout).
//!
//! `version_tag = 0` selects the legacy v2 layout (3 tables, packed-
//! identifier `bsbpp` byte). `version_tag = 1` selects the v3 layout
//! introduced for FFVHuff (1 / 3 / 4 tables, explicit chroma-shift bits
//! and `bps`).

use oxideav_core::{Error, Result};

use crate::huffman::HuffTable;
use crate::rle;

/// Spatial predictor (low 6 bits of `extra[0]`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Predictor {
    Left = 0,
    /// Trace doc calls this "PLANE" / "Predict Gradient":
    /// `left + above − above_left`.
    Gradient = 1,
    /// Paeth-style median of `(L, T, L+T-TL)`.
    Median = 2,
}

impl Predictor {
    fn from_bits(b: u8) -> Result<Self> {
        match b {
            0 => Ok(Self::Left),
            1 => Ok(Self::Gradient),
            2 => Ok(Self::Median),
            other => Err(Error::invalid(format!(
                "huffyuv: unknown predictor {other}"
            ))),
        }
    }
}

/// V2 colorspace identifier (`extra[1]` when `extra[3] == 0`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum V2Colorspace {
    /// `bsbpp = 12` — yuv420p (8-bit).
    Yuv420,
    /// `bsbpp = 16` — yuv422p (8-bit).
    Yuv422,
    /// `bsbpp = 24` — packed RGB24.
    Rgb24,
    /// `bsbpp = 32` — packed BGRA.
    Bgra32,
}

impl V2Colorspace {
    fn from_bsbpp(b: u8) -> Result<Self> {
        match b {
            12 => Ok(Self::Yuv420),
            16 => Ok(Self::Yuv422),
            24 => Ok(Self::Rgb24),
            32 => Ok(Self::Bgra32),
            other => Err(Error::invalid(format!("huffyuv v2: unknown bsbpp {other}"))),
        }
    }

    pub fn channel_count(&self) -> usize {
        // v2 always emits 3 tables; for RGBA the alpha plane re-uses
        // table index 2 (trace doc §3.1).
        3
    }

    pub fn is_rgb(&self) -> bool {
        matches!(self, Self::Rgb24 | Self::Bgra32)
    }
}

/// V3 channel/bps descriptor (`extra[1..3]` when `extra[3] == 1`).
#[derive(Copy, Clone, Debug)]
pub struct V3Format {
    /// Bits-per-sample (8..=16).
    pub bps: u8,
    /// Horizontal chroma subsampling shift (`width >> shift`). 0 for
    /// 4:4:4, 1 for 4:2:x, 2 for 4:1:1.
    pub chroma_h_shift: u8,
    /// Vertical chroma subsampling shift. 0 for 4:2:2 / 4:4:4 / 4:1:1,
    /// 1 for 4:2:0.
    pub chroma_v_shift: u8,
    /// True when the planes carry Y/U/V (vs. G/B/R for `gbrp`).
    pub yuv: bool,
    /// True when chroma planes are present (i.e. not gray).
    pub chroma: bool,
    /// True when an alpha plane is present.
    pub alpha: bool,
}

impl V3Format {
    /// Number of stored Huffman tables = 1 + (2 * chroma) + alpha.
    pub fn table_count(&self) -> usize {
        1 + (if self.chroma { 2 } else { 0 }) + (if self.alpha { 1 } else { 0 })
    }

    pub fn channel_count(&self) -> usize {
        self.table_count()
    }
}

/// Interlace-flag selector packed into bits 4..5 of `extra[2]`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum InterlaceMode {
    AutoByHeight,
    Interlaced,
    Progressive,
}

#[derive(Clone, Debug)]
pub enum FormatVersion {
    /// Legacy v2 (`version_tag = 0`).
    V2(V2Colorspace),
    /// v3 (`version_tag = 1`).
    V3(V3Format),
}

/// Parsed extradata: header bits + the per-channel canonical Huffman
/// tables baked from the RLE-coded length blob.
#[derive(Clone, Debug)]
pub struct Extradata {
    pub predictor: Predictor,
    pub decorrelate: bool,
    pub interlace: InterlaceMode,
    /// True when each `00dc` packet prepends a fresh set of RLE-coded
    /// length tables (FFVHuff `-context 1`, `extra[2] & 0x40`).
    pub per_frame_tables: bool,
    pub format: FormatVersion,
    /// Default tables shipped in extradata (fallback when not running
    /// in per-frame-tables mode).
    pub tables: Vec<HuffTable>,
}

impl Extradata {
    /// Parse the codec-private extradata (the bytes after the 40-byte
    /// `BITMAPINFOHEADER`).
    pub fn parse(extradata: &[u8]) -> Result<Self> {
        if extradata.len() < 4 {
            return Err(Error::invalid("huffyuv extradata: shorter than 4 bytes"));
        }
        let e0 = extradata[0];
        let e1 = extradata[1];
        let e2 = extradata[2];
        let e3 = extradata[3];

        let predictor = Predictor::from_bits(e0 & 0x3F)?;
        let decorrelate = (e0 & 0x40) != 0;
        let per_frame_tables = (e2 & 0x40) != 0;
        let interlace = match (e2 >> 4) & 0x03 {
            0 => InterlaceMode::AutoByHeight,
            1 => InterlaceMode::Interlaced,
            2 => InterlaceMode::Progressive,
            _ => InterlaceMode::AutoByHeight,
        };

        let format = match e3 {
            0 => FormatVersion::V2(V2Colorspace::from_bsbpp(e1)?),
            1 => {
                let bps = ((e1 >> 4) + 1) & 0x1F;
                if !(8..=16).contains(&bps) {
                    return Err(Error::invalid(format!(
                        "huffyuv v3: bps {bps} out of range"
                    )));
                }
                let chroma_h_shift = e1 & 0x03;
                let chroma_v_shift = (e1 >> 2) & 0x03;
                let yuv = (e2 & 0x01) != 0;
                // Trace doc §2.2.2: the "chroma" property reported in
                // the per-pixel-format table is `yuv || (e2 & 0x02)` —
                // the explicit chroma bit only fires for non-yuv
                // multi-plane layouts (gbrp/gbrap). yuv layouts (444p,
                // 422p, 420p, 411p, …) flag `yuv=1` and leave bit 1
                // clear; the decoder still walks 3 planes for them.
                let chroma = yuv || (e2 & 0x02) != 0;
                let alpha = (e2 & 0x04) != 0;
                FormatVersion::V3(V3Format {
                    bps,
                    chroma_h_shift,
                    chroma_v_shift,
                    yuv,
                    chroma,
                    alpha,
                })
            }
            other => {
                return Err(Error::invalid(format!(
                    "huffyuv extradata: unknown version_tag {other}"
                )));
            }
        };

        let n_tables = match &format {
            FormatVersion::V2(v) => v.channel_count(),
            FormatVersion::V3(v) => v.table_count(),
        };
        let symbols_per_table = match &format {
            FormatVersion::V2(_) => 256usize,
            FormatVersion::V3(v) => {
                let raw = 1usize << v.bps;
                raw.min(16384)
            }
        };

        let mut pos = 4usize;
        let mut tables = Vec::with_capacity(n_tables);
        for _ in 0..n_tables {
            let (lens, used) = rle::decode_lengths(&extradata[pos..], symbols_per_table)?;
            pos += used;
            tables.push(HuffTable::from_lengths(&lens)?);
        }
        // Trailing bytes after the last table are tolerated (some
        // captures pad to even RLE_size); we don't validate them.

        Ok(Self {
            predictor,
            decorrelate,
            interlace,
            per_frame_tables,
            format,
            tables,
        })
    }

    /// Number of tables stored (= channel count for v3, always 3 for v2).
    pub fn table_count(&self) -> usize {
        match &self.format {
            FormatVersion::V2(v) => v.channel_count(),
            FormatVersion::V3(v) => v.table_count(),
        }
    }

    pub fn bits_per_sample(&self) -> u8 {
        match &self.format {
            FormatVersion::V2(_) => 8,
            FormatVersion::V3(v) => v.bps,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthesise a length array with N symbols sharing the smallest
    /// possible balanced canonical-Huffman lengths and serialise it
    /// through the RLE encoder (mirror image of [`rle::decode_lengths`]).
    fn rle_encode(lengths: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut i = 0;
        while i < lengths.len() {
            let v = lengths[i];
            let mut run = 1usize;
            while i + run < lengths.len() && lengths[i + run] == v && run < 255 {
                run += 1;
            }
            if run < 8 {
                out.push((v & 0x1F) | ((run as u8) << 5));
            } else {
                out.push(v & 0x1F);
                out.push(run as u8);
            }
            i += run;
        }
        out
    }

    /// Build a minimal yuv422p v2 extradata blob with all-equal lengths
    /// (8-bit codes for each of the 256 symbols).
    fn synth_v2_yuv422_extradata() -> Vec<u8> {
        let lens = vec![8u8; 256];
        let rle_blob = rle_encode(&lens);
        let mut out = vec![0x00u8, 0x10, 0x20, 0x00];
        for _ in 0..3 {
            out.extend_from_slice(&rle_blob);
        }
        out
    }

    #[test]
    fn parses_v2_yuv422() {
        let blob = synth_v2_yuv422_extradata();
        let parsed = Extradata::parse(&blob).unwrap();
        assert_eq!(parsed.predictor, Predictor::Left);
        assert!(!parsed.decorrelate);
        assert_eq!(parsed.tables.len(), 3);
        assert_eq!(parsed.bits_per_sample(), 8);
        match parsed.format {
            FormatVersion::V2(V2Colorspace::Yuv422) => {}
            other => panic!("expected V2 Yuv422, got {other:?}"),
        }
    }

    #[test]
    fn parses_v3_gray8_header_bytes() {
        // 0x00 (LEFT, no decorr), 0x70 ((bps-1)<<4 = 7 → bps=8, ch=cv=0),
        // 0x20 (yuv=0, chroma=0, alpha=0, ilace bits = 10 progressive),
        // 0x01 (v3).
        let lens = vec![8u8; 256];
        let rle_blob = rle_encode(&lens);
        let mut blob = vec![0x00u8, 0x70, 0x20, 0x01];
        blob.extend_from_slice(&rle_blob); // 1 table for gray
        let parsed = Extradata::parse(&blob).unwrap();
        match parsed.format {
            FormatVersion::V3(v) => {
                assert_eq!(v.bps, 8);
                assert!(!v.chroma);
                assert!(!v.alpha);
                assert!(!v.yuv);
            }
            other => panic!("expected V3, got {other:?}"),
        }
        assert_eq!(parsed.interlace, InterlaceMode::Progressive);
        assert_eq!(parsed.tables.len(), 1);
    }
}
