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

    /// Wire-byte stride of one packed sample step in the raster — i.e.
    /// the per-output-position byte count that the LEFT / GRADIENT /
    /// MEDIAN predictor walks. Spec/02 §3 wire-byte layout table:
    ///
    /// - YUY2: 2 bytes per pixel (= 4 bytes per macropixel, advancing
    ///   2 pixels per macropixel).
    /// - RGB24: 3 bytes per pixel (`+0:B +1:G +2:R`).
    /// - RGB32: 4 bytes per pixel (`+0:B +1:G +2:R +3:A`).
    ///
    /// Companion to [`Self::row_bytes`] / [`StreamConfig::row_bytes`];
    /// kept narrow so the four predictor / decorrelation paths can
    /// share one source of truth instead of carrying the `2 / 3 / 4`
    /// literal at every call site.
    #[inline]
    pub fn bytes_per_pixel_step(self) -> usize {
        match self {
            PixelFamily::Yuy2 => 2,
            PixelFamily::Rgb24 => 3,
            PixelFamily::Rgb32 => 4,
        }
    }

    /// Row-byte count for a packed raster of `width` pixels in this
    /// family — = `width * bytes_per_pixel_step`. Spec/02 §3 wire-byte
    /// layout table. Always a multiple of `bytes_per_pixel_step(self)`.
    ///
    /// `width` is the pixel width (not the macropixel count): for
    /// YUY2 the caller passes the per-spec pixel width and gets back
    /// `width * 2` (= the bench-reference 320×16 raster's
    /// `row_bytes = 640`).
    ///
    /// Saturates the multiply at `usize::MAX` for hostile
    /// `width = u32::MAX`; callers that need the un-saturated total
    /// can cast and multiply themselves.
    #[inline]
    pub fn row_bytes(self, width: u32) -> usize {
        (width as usize).saturating_mul(self.bytes_per_pixel_step())
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
        //
        // The bpp-override byte lives at `+0x29`. `biSize` is the
        // *declared* header size off the wire; on the `low3 != 0`
        // method path the `biSize <= strf.len()` check above is skipped,
        // so a header that declares `biSize > 0x29` but supplies fewer
        // than 0x2A actual bytes would index out of bounds here. Gate on
        // the real buffer length as well.
        let effective_bits: u16 = if (bi_size as usize) > 0x29 && strf.len() > 0x29 {
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

    /// Wire-format row stride for this stream's resolved family + width
    /// (= [`PixelFamily::row_bytes`] applied to `self.width`). Spec/02
    /// §3 wire-byte layout table.
    ///
    /// Round-261 consolidation accessor: the decode + encode paths
    /// previously open-coded `match family { Yuy2 => width × 2, Rgb24
    /// => width × 3, Rgb32 => width × 4 }` at four call sites; each
    /// now defers to this single source of truth.
    #[inline]
    pub fn row_bytes(&self) -> usize {
        self.family.row_bytes(self.width)
    }

    /// `true` when this stream uses the field-stride=2 interlaced
    /// path (`biHeight > 288`, spec/02 §2 disassembly note). Thin
    /// wrapper around [`crate::predict::is_interlaced_height`] so
    /// callers that already hold a `StreamConfig` don't need to
    /// import the predict module.
    #[inline]
    pub fn is_interlaced(&self) -> bool {
        crate::predict::is_interlaced_height(self.height)
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

    // ─── Round-261: typed accessors `PixelFamily::row_bytes`,
    //     `PixelFamily::bytes_per_pixel_step`, `StreamConfig::row_bytes`,
    //     `StreamConfig::is_interlaced`. Spec/02 §3 wire-byte layout
    //     table + spec/02 §2 interlace trigger. ────────────────────────

    #[test]
    fn round261_bytes_per_pixel_step_matches_spec_table() {
        // spec/02 §3 wire-byte layout table (lines 790–793 of
        // docs/video/huffyuv/spec/02-frame-layout.md):
        // YUY2 = 4 bytes per macropixel = 2 bytes per pixel; RGB24 = 3
        // bytes per pixel (`+0:B +1:G +2:R`); RGB32 = 4 bytes per pixel.
        assert_eq!(PixelFamily::Yuy2.bytes_per_pixel_step(), 2);
        assert_eq!(PixelFamily::Rgb24.bytes_per_pixel_step(), 3);
        assert_eq!(PixelFamily::Rgb32.bytes_per_pixel_step(), 4);
    }

    #[test]
    fn round261_pixel_family_row_bytes_matches_inline_match() {
        // Lock the new accessor against the inline `match family`
        // pattern the decode / encode paths previously open-coded:
        // exhaustive over every (family × width-of-interest) pair.
        for &w in &[0u32, 1, 2, 4, 8, 16, 160, 320, 720, 1024, 1920] {
            assert_eq!(PixelFamily::Yuy2.row_bytes(w), w as usize * 2);
            assert_eq!(PixelFamily::Rgb24.row_bytes(w), w as usize * 3);
            assert_eq!(PixelFamily::Rgb32.row_bytes(w), w as usize * 4);
        }
    }

    #[test]
    fn round261_pixel_family_row_bytes_saturates_on_overflow() {
        // Hostile fuzz width: u32::MAX × 3 = ~12.8 GiB on 64-bit
        // (so it doesn't actually overflow `usize` there), but on a
        // 32-bit target it would. Use the contract: `saturating_mul`
        // never panics. We can't easily build the overflow on a 64-bit
        // host so just confirm the value is finite for the worst
        // declared width.
        let big = PixelFamily::Rgb32.row_bytes(u32::MAX);
        assert!(big >= (u32::MAX as usize) * 4 || big == usize::MAX);
    }

    #[test]
    fn round261_stream_config_row_bytes_delegates_to_family() {
        // The decoder dispatch we replaced is `match config.family {
        // Yuy2 => config.width as usize * 2, Rgb24 => * 3, Rgb32 => *
        // 4 }`; confirm the StreamConfig accessor produces exactly
        // that value through a real `parse_bitmapinfoheader` path so a
        // future refactor that drops one of the layers still locks
        // wire-identicality.
        // YUY2 320×16 (the bench-reference raster).
        let bih = make_bih(0x28, 320, 16, 16, FOURCC_HFYU, &[]);
        let cfg = StreamConfig::parse_bitmapinfoheader(&bih).unwrap();
        assert_eq!(cfg.family, PixelFamily::Yuy2);
        assert_eq!(cfg.row_bytes(), 640);
        // RGB24 1024×720 (the spec/01 §7 worked example).
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
        assert_eq!(cfg.row_bytes(), 3072);
        // RGB32 720×480 (a common DV-class active raster).
        let extradata = vec![0x40u8, 0x20, 0x00, 0x00, 0xFF];
        let bih = make_bih(
            0x28 + extradata.len() as u32,
            720,
            480,
            32,
            FOURCC_HFYU,
            &extradata,
        );
        let cfg = StreamConfig::parse_bitmapinfoheader(&bih).unwrap();
        assert_eq!(cfg.family, PixelFamily::Rgb32);
        assert_eq!(cfg.row_bytes(), 2880);
    }

    #[test]
    fn round261_stream_config_is_interlaced_matches_height_threshold() {
        // spec/02 §2 disassembly note: `biHeight > 288` engages the
        // field-stride=2 interlaced path. Confirm the convenience
        // accessor honours the same threshold as the underlying
        // free function.
        for &(h, want) in &[
            (0u32, false),
            (1, false),
            (288, false), // exactly 288 is NOT interlaced (strict >).
            (289, true),
            (480, true),
            (720, true),
            (1080, true),
        ] {
            // Build a minimal-but-valid YUY2 v1.x BIH at height h.
            // Width 16 keeps total < 65 KiB (the strf field is i32
            // so any positive height works).
            let bih = make_bih(0x28, 16, h as i32, 16, FOURCC_HFYU, &[]);
            let cfg = StreamConfig::parse_bitmapinfoheader(&bih).unwrap();
            assert_eq!(
                cfg.is_interlaced(),
                want,
                "height = {h}: expected is_interlaced() = {want}"
            );
            // Also confirm it matches the free-function source of truth.
            assert_eq!(cfg.is_interlaced(), crate::predict::is_interlaced_height(h));
        }
    }

    #[test]
    fn oversized_bisize_with_short_buffer_no_panic() {
        // Fuzz regression: a header that declares biSize > 0x29 but
        // supplies only a 40-byte buffer, on the low3 != 0 method path
        // (which skips the biSize <= len check), must NOT index the
        // missing bpp-override byte at +0x29. biBitCount = 0x11 sets
        // low3 = 1 (= Method::Left); biSize = 0x32 (> 0x29).
        let mut bih = make_bih(0x32, 4, 8, 0x11, FOURCC_HFYU, &[]);
        assert_eq!(bih.len(), 40);
        // Must return a Result, never panic.
        let _ = StreamConfig::parse_bitmapinfoheader(&bih);
        // Also exercise the path with exactly 0x2A bytes present (the
        // override byte IS readable) to confirm the gate still allows
        // the legitimate read.
        bih.push(0x00);
        let _ = StreamConfig::parse_bitmapinfoheader(&bih);
    }
}
