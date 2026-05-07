//! HuffYUV / FFVHuff `BITMAPINFOHEADER` + extradata parsing.
//!
//! Implements `spec/01-file-header-and-method.md`:
//!
//! - §1 — fixed BIH layout (`biSize`/`biWidth`/`biHeight`/`biBitCount`/
//!   `biCompression`).
//! - §1.4 — low-3-bits-of-`biBitCount` `→ method` shortcut for v1.x and
//!   v2.x-with-low3-method streams.
//! - §2 — effective-bit-count recovery (`bpp_override` at `+0x29` when
//!   `biSize > 0x29`, otherwise `biBitCount`).
//! - §3 — the 4-byte fixed extradata prefix (method, bpp_override,
//!   pad×2).
//! - §4 — the 16/24/32 → colorspace classifier.
//! - §6 — the `biSize == 0x28` (no-extradata) v1.x compatibility
//!   layout.

use crate::error::{Error, Result};

/// FOURCC of the HuffYUV native container codec, as written into
/// `BITMAPINFOHEADER.biCompression` little-endian. Equivalent to ASCII
/// `H F Y U`. spec/01 §1.5.
pub const FOURCC_HFYU: u32 = 0x55594648;

/// FFmpeg-internal FOURCC for FFVHuff streams. spec/00 explicitly
/// excludes this from the proprietary binary's wire-format study, but
/// the registry has long advertised it; we accept it on the demux
/// path as an alias of HFYU. Bytes `[F,F,V,H]` little-endian =
/// `0x48565646`.
pub const FOURCC_FFVH: u32 = u32::from_le_bytes(*b"FFVH");

/// Predictor + decorrelation method byte (spec/01 §3.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    /// `predict_old` (signed `-2`). Equivalent to plain `Left`. Legal
    /// for both YUV and RGB.
    PredictOld,
    /// `0x00`. Plain LEFT predictor, no decorrelation.
    Left,
    /// `0x01`. Plain GRADIENT predictor, no decorrelation. YUV only.
    Gradient,
    /// `0x02`. MEDIAN predictor, no decorrelation. YUV only.
    Median,
    /// `0x40`. LEFT predictor with R-G/G/B-G decorrelation. RGB only.
    LeftDecorr,
    /// `0x41`. GRADIENT predictor with R-G/G/B-G decorrelation. RGB
    /// only.
    GradientDecorr,
}

impl Method {
    /// Wire-format byte (signed; the legacy `predict_old` stores as
    /// `0xFE`).
    pub fn to_byte(self) -> i8 {
        match self {
            Method::PredictOld => -2,
            Method::Left => 0x00,
            Method::Gradient => 0x01,
            Method::Median => 0x02,
            Method::LeftDecorr => 0x40,
            Method::GradientDecorr => 0x41,
        }
    }

    /// Decode a signed wire byte; respects the `{-2,0,1,2,0x40,0x41}`
    /// allow-list and per-colorspace validity (spec/01 §3.1). Returns
    /// `None` if the byte is out of range.
    pub fn from_byte(byte: i8) -> Option<Self> {
        Some(match byte {
            -2 => Method::PredictOld,
            0x00 => Method::Left,
            0x01 => Method::Gradient,
            0x02 => Method::Median,
            0x40 => Method::LeftDecorr,
            0x41 => Method::GradientDecorr,
            _ => return None,
        })
    }

    /// `true` when this method is allowed for an RGB stream.
    pub fn is_rgb_legal(self) -> bool {
        matches!(
            self,
            Method::PredictOld | Method::Left | Method::LeftDecorr | Method::GradientDecorr
        )
    }

    /// `true` when this method is allowed for a YUV stream.
    pub fn is_yuv_legal(self) -> bool {
        matches!(
            self,
            Method::PredictOld | Method::Left | Method::Gradient | Method::Median
        )
    }

    /// Predictor algorithm with decorrelation (if any) stripped off.
    pub fn predictor(self) -> Predictor {
        match self {
            Method::PredictOld | Method::Left | Method::LeftDecorr => Predictor::Left,
            Method::Gradient | Method::GradientDecorr => Predictor::Gradient,
            Method::Median => Predictor::Median,
        }
    }

    /// True when bit 6 of the method byte is set (= RGB
    /// decorrelation enabled).
    pub fn decorrelate(self) -> bool {
        matches!(self, Method::LeftDecorr | Method::GradientDecorr)
    }
}

/// Predictor algorithm independent of decorrelation flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Predictor {
    Left,
    Gradient,
    Median,
}

/// One of the three legal pixel families for a HuffYUV stream
/// (spec/01 §4 / spec/02 §3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFamily {
    /// 16-bit packed YUY2 (`Y₁ U Y₂ V` per macropixel).
    Yuy2,
    /// 24-bit packed BGR with X pad in first uncompressed pixel,
    /// 3-codes-per-pixel residual stream (B,G,R).
    Rgb24,
    /// 32-bit packed BGRA, 4 codes per pixel after the first.
    Rgb32,
}

impl PixelFamily {
    pub fn is_rgb(self) -> bool {
        matches!(self, PixelFamily::Rgb24 | PixelFamily::Rgb32)
    }

    /// Bit count this family carries on the wire (16/24/32). Note: the
    /// encoder may force `biBitCount` to ≥ 24 even for YUY2 streams;
    /// this accessor returns the *effective* bit count we resolved.
    pub fn effective_bits(self) -> u16 {
        match self {
            PixelFamily::Yuy2 => 16,
            PixelFamily::Rgb24 => 24,
            PixelFamily::Rgb32 => 32,
        }
    }
}

/// Resolved per-stream configuration (spec/01 §1..§4 + extradata
/// table region). `extradata_tables` is `None` for v1.x streams
/// (`biSize == 0x28`).
#[derive(Debug, Clone)]
pub struct StreamConfig {
    pub family: PixelFamily,
    pub method: Method,
    pub width: u32,
    pub height: u32,
    /// `true` when the BIH carried a v2.x extradata payload — i.e. the
    /// per-stream Huffman length tables are RLE-compressed inside the
    /// BIH. `false` selects the v1.x precomputed-codes path
    /// (`tables/06..09`).
    pub has_extradata: bool,
    /// RLE-compressed length-table region — i.e. the `(biSize - 0x2c)`
    /// bytes immediately after the 4-byte fixed prefix. Empty when
    /// `has_extradata == false`.
    pub extradata_tables: Vec<u8>,
}

impl StreamConfig {
    /// Parse a `BITMAPINFOHEADER` + optional extradata buffer. The
    /// input is the AVI `strf` payload exactly as the muxer emits it
    /// — at least 40 bytes (= bare BIH) — and includes any extradata
    /// in-line.
    ///
    /// spec/01 §7 (worked example) walks one of these end-to-end.
    pub fn parse_bitmapinfoheader(strf: &[u8]) -> Result<Self> {
        if strf.len() < 40 {
            return Err(Error::invalid(format!(
                "BITMAPINFOHEADER: need ≥ 40 bytes, got {}",
                strf.len()
            )));
        }
        let bi_size = u32::from_le_bytes([strf[0], strf[1], strf[2], strf[3]]);
        let bi_width = i32::from_le_bytes([strf[4], strf[5], strf[6], strf[7]]).unsigned_abs();
        let bi_height_signed = i32::from_le_bytes([strf[8], strf[9], strf[10], strf[11]]);
        let bi_height = bi_height_signed.unsigned_abs();
        let bi_bit_count = u16::from_le_bytes([strf[0x0E], strf[0x0F]]);
        let bi_compression = u32::from_le_bytes([strf[0x10], strf[0x11], strf[0x12], strf[0x13]]);

        if bi_compression != FOURCC_HFYU && bi_compression != FOURCC_FFVH {
            return Err(Error::invalid(format!(
                "biCompression = 0x{:08x} is not HFYU or FFVH",
                bi_compression
            )));
        }

        // Method resolution (spec/01 §1.4 + §3.1):
        //
        // - Low 3 bits of biBitCount carry a v1.x predictor selector
        //   when non-zero.
        // - Otherwise the method byte at extradata `+0x00` (= BIH
        //   `+0x28`) carries it, when extradata is present
        //   (`biSize > 0x28`).
        // - Otherwise it's predict_old.
        let low3 = (bi_bit_count & 0x07) as u8;
        let extradata_present = (bi_size as usize) > 0x28;
        let method = if low3 != 0 {
            method_from_low3(low3, bi_bit_count)
        } else if extradata_present {
            // Need the method byte; require biSize > 0x28 + at least
            // the 1-byte method.
            if (bi_size as usize) > strf.len() {
                return Err(Error::invalid(format!(
                    "biSize = 0x{:x} exceeds strf payload {} bytes",
                    bi_size,
                    strf.len()
                )));
            }
            if (bi_size as usize) < 0x29 {
                return Err(Error::invalid(format!(
                    "biSize = 0x{:x} too small for extradata method byte",
                    bi_size
                )));
            }
            let method_byte = strf[0x28] as i8;
            Method::from_byte(method_byte).ok_or_else(|| {
                Error::invalid(format!("illegal method byte 0x{:02x}", method_byte as u8))
            })?
        } else {
            Method::PredictOld
        };

        // Effective-bit-count recovery (spec/01 §2).
        let effective_bits: u16 = if (bi_size as usize) > 0x29 {
            // bpp_override may override, when non-zero. Signed byte.
            let bpp = strf[0x29] as i8;
            if bpp != 0 {
                bpp.unsigned_abs() as u16
            } else {
                // Fall back to biBitCount, with the low 3 bits cleared
                // (those bits carried the v1.x method selector).
                bi_bit_count & 0xFFF8
            }
        } else {
            bi_bit_count & 0xFFF8
        };

        let family = match effective_bits {
            16 => PixelFamily::Yuy2,
            24 => PixelFamily::Rgb24,
            32 => PixelFamily::Rgb32,
            n => {
                return Err(Error::invalid(format!(
                    "unsupported effective bit count {n}"
                )));
            }
        };

        // Per-colorspace allow-list (spec/01 §3.1).
        if family.is_rgb() && !method.is_rgb_legal() {
            return Err(Error::invalid(format!(
                "method {method:?} not legal for RGB"
            )));
        }
        if !family.is_rgb() && !method.is_yuv_legal() {
            return Err(Error::invalid(format!(
                "method {method:?} not legal for YUV"
            )));
        }

        // RGB family must use one of the RGB-decorr-or-not methods;
        // YUV only YUV ones; we already rejected non-matches above.
        let mut extradata_tables = Vec::new();
        if extradata_present {
            // 4-byte fixed prefix lives at +0x28..+0x2C; the
            // RLE-compressed tables begin at +0x2C.
            if (bi_size as usize) >= 0x2C {
                let end = (bi_size as usize).min(strf.len());
                if end > 0x2C {
                    extradata_tables.extend_from_slice(&strf[0x2C..end]);
                }
            }
        }

        Ok(Self {
            family,
            method,
            width: bi_width,
            height: bi_height,
            has_extradata: extradata_present,
            extradata_tables,
        })
    }
}

/// Resolve the v1.x `low3 → method` mapping per spec/01 §1.4. The
/// `bi_bit_count` is the full 16-bit field; only the threshold
/// against `0x18` (= 24) for `low3 == 3` is relevant.
fn method_from_low3(low3: u8, bi_bit_count: u16) -> Method {
    match low3 {
        1 => Method::Left,
        2 => Method::LeftDecorr,
        3 => {
            if bi_bit_count >= 0x18 {
                Method::GradientDecorr
            } else {
                Method::Gradient
            }
        }
        4 => Method::Median,
        // 0 is "use extradata, or predict_old"; 5..=7 fall back to
        // predict_old (legacy). spec/01 §1.4 table.
        _ => Method::PredictOld,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fourcc_constants_match_spec() {
        // spec/01 §1.5: FOURCC `'HFYU'` = `0x55594648`.
        assert_eq!(FOURCC_HFYU, 0x55594648);
        let bytes = FOURCC_HFYU.to_le_bytes();
        assert_eq!(&bytes, b"HFYU");
        // FFVH alias.
        assert_eq!(&FOURCC_FFVH.to_le_bytes(), b"FFVH");
    }

    #[test]
    fn method_byte_round_trip() {
        for m in [
            Method::PredictOld,
            Method::Left,
            Method::Gradient,
            Method::Median,
            Method::LeftDecorr,
            Method::GradientDecorr,
        ] {
            assert_eq!(Method::from_byte(m.to_byte()), Some(m));
        }
        assert_eq!(Method::from_byte(0x03), None);
        assert_eq!(Method::from_byte(0x42), None);
    }

    fn make_bih(
        bi_size: u32,
        width: i32,
        height: i32,
        bit_count: u16,
        fourcc: u32,
        extradata: &[u8],
    ) -> Vec<u8> {
        let mut v = vec![0u8; 40 + extradata.len()];
        v[0..4].copy_from_slice(&bi_size.to_le_bytes());
        v[4..8].copy_from_slice(&width.to_le_bytes());
        v[8..12].copy_from_slice(&height.to_le_bytes());
        v[12..14].copy_from_slice(&1u16.to_le_bytes()); // biPlanes
        v[0x0E..0x10].copy_from_slice(&bit_count.to_le_bytes());
        v[0x10..0x14].copy_from_slice(&fourcc.to_le_bytes());
        if !extradata.is_empty() {
            v[40..40 + extradata.len()].copy_from_slice(extradata);
        }
        v
    }

    #[test]
    fn v2_rgb24_gradient_decorr_parses() {
        // spec/01 §7 worked example.
        let extradata = vec![0x41u8, 0x18, 0x00, 0x00, 0xFF];
        let bih = make_bih(
            0x28 + extradata.len() as u32,
            1024,
            720,
            24,
            FOURCC_HFYU,
            &extradata,
        );
        let cfg = StreamConfig::parse_bitmapinfoheader(&bih).unwrap();
        assert_eq!(cfg.family, PixelFamily::Rgb24);
        assert_eq!(cfg.method, Method::GradientDecorr);
        assert_eq!(cfg.width, 1024);
        assert_eq!(cfg.height, 720);
        assert!(cfg.has_extradata);
        assert_eq!(cfg.extradata_tables, vec![0xFFu8]);
    }

    #[test]
    fn v1x_yuy2_no_extradata() {
        // biSize = 0x28, biBitCount=16: v1.x YUY2 with predict_old.
        let bih = make_bih(0x28, 16, 8, 16, FOURCC_HFYU, &[]);
        let cfg = StreamConfig::parse_bitmapinfoheader(&bih).unwrap();
        assert_eq!(cfg.family, PixelFamily::Yuy2);
        assert_eq!(cfg.method, Method::PredictOld);
        assert!(!cfg.has_extradata);
    }

    #[test]
    fn ffvh_fourcc_accepted() {
        // Demuxer alias.
        let bih = make_bih(0x28, 16, 16, 16, FOURCC_FFVH, &[]);
        let cfg = StreamConfig::parse_bitmapinfoheader(&bih).unwrap();
        assert_eq!(cfg.family, PixelFamily::Yuy2);
    }

    #[test]
    fn unsupported_compression_rejected() {
        let bih = make_bih(0x28, 16, 16, 24, 0xDEAD_BEEF, &[]);
        let err = StreamConfig::parse_bitmapinfoheader(&bih).unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)));
    }

    #[test]
    fn rgb24_left_decorr_via_low3() {
        // spec/01 §1.4: low3 == 2 → 0x40 (left + decorr) for RGB.
        let bih = make_bih(0x28, 16, 16, 0x18 | 2, FOURCC_HFYU, &[]);
        let cfg = StreamConfig::parse_bitmapinfoheader(&bih).unwrap();
        // Effective bit count is biBitCount with low 3 bits cleared
        // (= 24); RGB24 + LeftDecorr.
        assert_eq!(cfg.family, PixelFamily::Rgb24);
        assert_eq!(cfg.method, Method::LeftDecorr);
    }
}
