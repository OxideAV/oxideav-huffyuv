//! Live black-box interoperability matrix.
//!
//! Two sweeps, both generated on the fly in a temp directory and both
//! skipped at runtime when the `ffmpeg` executable is absent:
//!
//! 1. **Foreign → ours.** A black-box encoder writes `HFYU` AVI files
//!    across every pixel format, predictor, field-coding flag and
//!    raster shape it exposes; each stream is decoded by this crate
//!    and by a black-box decoder and the two outputs must be
//!    byte-identical (the format is lossless, spec/02 §1).
//! 2. **Ours → foreign.** This crate's encoder is swept over every
//!    `(family, method, extradata mode, raster)` cell; each output is
//!    wrapped in a minimal AVI and a black-box decoder must accept it
//!    without a diagnostic and reproduce the source raster exactly.
//!
//! Nothing large is committed: sources are synthesised, streams live
//! in a per-process temp directory that is removed afterwards.

use std::path::{Path, PathBuf};
use std::process::Command;

use oxideav_huffyuv::decoder::decode_frame;
use oxideav_huffyuv::encoder::{
    EncodeOptions, ExtradataMode, FieldCoding, MethodSelection, encode_frame_with_mode,
    encode_frame_with_options,
};
use oxideav_huffyuv::header::{Method, PixelFamily, StreamConfig};

fn ffmpeg_available() -> bool {
    // An interpreter-hosted test run cannot spawn processes.
    if cfg!(miri) {
        return false;
    }
    Command::new("ffmpeg")
        .arg("-version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let p = std::env::temp_dir().join(format!(
            "oxideav-huffyuv-blackbox-{tag}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }
    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The two things this crate needs out of an AVI file: the video
/// stream's `strf` payload and the `NNdc` / `NNdb` chunk payloads.
struct AviStream {
    strf: Vec<u8>,
    frames: Vec<Vec<u8>>,
}

fn walk_riff(data: &[u8], out: &mut AviStream) {
    let mut pos = 0usize;
    while pos + 8 <= data.len() {
        let id = &data[pos..pos + 4];
        let size = u32::from_le_bytes(data[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let body_start = pos + 8;
        let body_end = body_start.saturating_add(size).min(data.len());
        let body = &data[body_start..body_end];
        if id == b"RIFF" || id == b"LIST" {
            if body.len() >= 4 {
                walk_riff(&body[4..], out);
            }
        } else if id == b"strf" {
            if out.strf.is_empty() {
                out.strf = body.to_vec();
            }
        } else if id[0].is_ascii_digit()
            && id[1].is_ascii_digit()
            && (&id[2..] == b"dc" || &id[2..] == b"db")
        {
            out.frames.push(body.to_vec());
        }
        pos = body_end + (size & 1);
    }
}

fn read_avi(path: &Path) -> AviStream {
    let data = std::fs::read(path).unwrap();
    let mut out = AviStream {
        strf: Vec::new(),
        frames: Vec::new(),
    };
    walk_riff(&data, &mut out);
    out
}

/// Flip a bottom-up raster to top-down (or back).
fn flip_rows(pixels: &[u8], row_bytes: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(pixels.len());
    for row in pixels.chunks_exact(row_bytes).rev() {
        out.extend_from_slice(row);
    }
    out
}

struct ForeignCase {
    codec: &'static str,
    pix_fmt: &'static str,
    pred: &'static str,
    ilme: bool,
    context: bool,
    width: u32,
    height: u32,
    frames: u32,
}

impl ForeignCase {
    fn name(&self) -> String {
        format!(
            "{}-{}-{}-{}x{}-n{}{}{}",
            self.codec,
            self.pix_fmt,
            self.pred,
            self.width,
            self.height,
            self.frames,
            if self.ilme { "-ilme" } else { "" },
            if self.context { "-ctx" } else { "" },
        )
    }

    /// `true` when the stream is inside the specified `HFYU` wire
    /// format: one of the three packed 8-bit families, stream-level
    /// tables. Everything else the black-box encoder can write under
    /// the `FFVH` tag is an unspecified extension and must be refused,
    /// never mis-decoded.
    fn specified(&self) -> bool {
        native_output(self.pix_fmt).is_some() && !self.context
    }
}

enum Outcome {
    /// Decoded, and byte-identical to the source raster (and to the
    /// black-box decoder's output whenever that decoder accepts the
    /// stream).
    Exact,
    /// Cleanly rejected by this crate.
    Refused(String),
    /// Wrong pixels, or accepted when it must not be.
    Mismatch(String),
    /// The black-box encoder would not write this combination.
    EncoderRejected,
}

/// Raw output format the black-box decoder is asked for, plus the
/// family this crate should report for it.
fn native_output(pix_fmt: &str) -> Option<(&'static str, PixelFamily)> {
    match pix_fmt {
        "yuv422p" => Some(("yuyv422", PixelFamily::Yuy2)),
        "rgb24" => Some(("bgr24", PixelFamily::Rgb24)),
        "bgra" => Some(("bgra", PixelFamily::Rgb32)),
        _ => None,
    }
}

fn run_foreign_case(dir: &TempDir, case: &ForeignCase) -> Outcome {
    let avi = dir.path(&format!("{}.avi", case.name()));
    // The test pattern has a minimum raster: render at least 32×32 and
    // crop. `format=` pins the conversion inside the graph so the
    // encoder input and the reference dump below are the same bytes.
    let graph = format!(
        "testsrc2=size={}x{}:rate=5,noise=alls=12:allf=t:all_seed=7,crop={}:{}:0:0:exact=1,format={}",
        case.width.max(32),
        case.height.max(32),
        case.width,
        case.height,
        case.pix_fmt
    );
    let frames = case.frames.to_string();
    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-v", "error", "-y", "-f", "lavfi", "-i", &graph]);
    cmd.args(["-frames:v", &frames]);
    cmd.args(["-pix_fmt", case.pix_fmt, "-c:v", case.codec]);
    cmd.args(["-pred", case.pred]);
    if case.ilme {
        cmd.args(["-flags", "+ilme"]);
    }
    if case.context {
        cmd.args(["-context", "1"]);
    }
    cmd.arg(&avi);
    let out = cmd.output().unwrap();
    if !out.status.success() {
        return Outcome::EncoderRejected;
    }
    let stream = read_avi(&avi);
    let config = match StreamConfig::parse_bitmapinfoheader(&stream.strf) {
        Ok(c) => c,
        Err(e) => return Outcome::Refused(format!("{e}")),
    };
    let Some((raw_fmt, family)) = native_output(case.pix_fmt) else {
        return Outcome::Mismatch(format!(
            "header accepted as {:?} for a {} stream",
            config.family, case.pix_fmt
        ));
    };
    if config.family != family {
        return Outcome::Mismatch(format!("family {:?} != {:?}", config.family, family));
    }
    // Reference 1: the source raster itself, repacked losslessly.
    let src_raw = dir.path(&format!("{}.src.raw", case.name()));
    let out = Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-f", "lavfi", "-i", &graph])
        .args(["-frames:v", &frames])
        .args(["-pix_fmt", raw_fmt, "-f", "rawvideo"])
        .arg(&src_raw)
        .output()
        .unwrap();
    assert!(out.status.success(), "source dump failed");
    let expected = std::fs::read(&src_raw).unwrap();
    // Reference 2: the black-box decoder, when it accepts the stream
    // (it declines some raster shapes its own encoder writes).
    let dec_raw = dir.path(&format!("{}.dec.raw", case.name()));
    let out = Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-i"])
        .arg(&avi)
        .args(["-pix_fmt", raw_fmt, "-f", "rawvideo"])
        .arg(&dec_raw)
        .output()
        .unwrap();
    if out.status.success() {
        let decoded = std::fs::read(&dec_raw).unwrap();
        if decoded != expected {
            return Outcome::Mismatch("black-box decoder disagrees with the source".into());
        }
    }
    let row_bytes = config.row_bytes();
    let frame_bytes = row_bytes * case.height as usize;
    if stream.frames.len() != case.frames as usize
        || expected.len() != frame_bytes * stream.frames.len()
    {
        return Outcome::Mismatch(format!(
            "frame count / size: {} chunks, {} expected bytes",
            stream.frames.len(),
            expected.len()
        ));
    }
    for (i, chunk) in stream.frames.iter().enumerate() {
        let decoded = match decode_frame(&config, chunk) {
            Ok(d) => d,
            Err(e) => return Outcome::Refused(format!("frame {i}: {e}")),
        };
        let want = &expected[i * frame_bytes..(i + 1) * frame_bytes];
        let got = if family.is_rgb() {
            flip_rows(&decoded.pixels, row_bytes)
        } else {
            decoded.pixels
        };
        if got != want {
            let diff = got.iter().zip(want).filter(|(a, b)| a != b).count();
            let first = got.iter().zip(want).position(|(a, b)| a != b).unwrap();
            return Outcome::Mismatch(format!(
                "frame {i}: {diff}/{frame_bytes} bytes differ, first at {first}"
            ));
        }
    }
    Outcome::Exact
}

fn foreign_cases() -> Vec<ForeignCase> {
    let mut v = Vec::new();
    // Tiny, odd, non-multiple-of-4, both sides of the 288-line field
    // threshold (spec/06 §1.1), odd heights above it (spec/06 §2.4),
    // and two real-world rasters.
    let shapes: &[(u32, u32)] = &[
        (2, 1),
        (2, 2),
        (4, 3),
        (6, 5),
        (16, 16),
        (18, 7),
        (64, 48),
        (62, 289),
        (64, 288),
        (64, 290),
        (34, 321),
        (320, 240),
        (720, 576),
    ];
    let rgb_extra: &[(u32, u32)] = &[(1, 1), (1, 9), (3, 3), (5, 291), (33, 17)];
    for codec in ["huffyuv", "ffvhuff"] {
        for pix_fmt in ["yuv422p", "rgb24", "bgra"] {
            for pred in ["left", "plane", "median"] {
                for ilme in [false, true] {
                    let extra: &[(u32, u32)] = if pix_fmt == "yuv422p" { &[] } else { rgb_extra };
                    for &(width, height) in shapes.iter().chain(extra) {
                        for context in [false, true] {
                            if context && (codec == "huffyuv" || width != 64) {
                                continue;
                            }
                            v.push(ForeignCase {
                                codec,
                                pix_fmt,
                                pred,
                                ilme,
                                context,
                                width,
                                height,
                                frames: if width == 720 { 1 } else { 3 },
                            });
                        }
                    }
                }
            }
        }
    }
    // Formats only the `FFVH` tag carries: every one must be refused.
    for pix_fmt in [
        "yuv420p",
        "yuv444p",
        "yuv411p",
        "yuv410p",
        "yuv440p",
        "gray",
        "gray16le",
        "gbrp",
        "gbrap",
        "yuva420p",
        "yuva422p",
        "yuva444p",
        "yuv420p9le",
        "yuv420p10le",
        "yuv422p12le",
        "yuv444p14le",
        "yuv422p16le",
        "yuva444p10le",
        "gbrp9le",
        "gbrp10le",
        "gbrp12le",
        "gbrp14le",
        "gbrp16le",
    ] {
        for pred in ["left", "plane", "median"] {
            for context in [false, true] {
                v.push(ForeignCase {
                    codec: "ffvhuff",
                    pix_fmt,
                    pred,
                    ilme: false,
                    context,
                    width: 64,
                    height: 48,
                    frames: 2,
                });
            }
        }
    }
    v
}

#[test]
fn foreign_streams_decode_bit_exact() {
    if !ffmpeg_available() {
        eprintln!("skip: ffmpeg not on PATH");
        return;
    }
    let dir = TempDir::new("foreign");
    let cases = foreign_cases();
    let (mut exact, mut rejected, mut refused) = (0usize, 0usize, 0usize);
    let mut failures = Vec::new();
    for case in &cases {
        match (case.specified(), run_foreign_case(&dir, case)) {
            (_, Outcome::EncoderRejected) => rejected += 1,
            (true, Outcome::Exact) => exact += 1,
            (false, Outcome::Refused(_)) => refused += 1,
            (true, Outcome::Refused(why)) => {
                failures.push(format!("{}: refused: {why}", case.name()));
            }
            (false, Outcome::Exact) => {
                failures.push(format!("{}: unspecified stream accepted", case.name()));
            }
            (_, Outcome::Mismatch(why)) => failures.push(format!("{}: {why}", case.name())),
        }
    }
    eprintln!(
        "foreign → ours: {} cells; {} not writable by the black-box encoder; \
         {} specified streams bit-exact; {} unspecified FFVH streams refused; {} failed",
        cases.len(),
        rejected,
        exact,
        refused,
        failures.len()
    );
    for f in &failures {
        eprintln!("  {f}");
    }
    assert!(
        failures.is_empty(),
        "{} foreign cells failed",
        failures.len()
    );
}

// ───────────────────────────── ours → foreign ─────────────────────────────

fn chunk(id: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(8 + body.len() + 1);
    v.extend_from_slice(id);
    v.extend_from_slice(&(body.len() as u32).to_le_bytes());
    v.extend_from_slice(body);
    if body.len() & 1 == 1 {
        v.push(0);
    }
    v
}

fn list(kind: &[u8; 4], form: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut inner = Vec::with_capacity(4 + body.len());
    inner.extend_from_slice(form);
    inner.extend_from_slice(body);
    chunk(kind, &inner)
}

/// Minimal single-video-stream AVI: `hdrl` (`avih` + one `strl`),
/// `movi` with one `00dc` chunk per frame, and an `idx1` index.
fn write_avi(path: &Path, strf: &[u8], frames: &[Vec<u8>], width: u32, height: u32) {
    let largest = frames.iter().map(Vec::len).max().unwrap_or(0) as u32;
    let n = frames.len() as u32;
    let mut avih = Vec::new();
    for v in [
        200_000u32, 0, 0, 0x10, n, 0, 1, largest, width, height, 0, 0, 0, 0,
    ] {
        avih.extend_from_slice(&v.to_le_bytes());
    }
    let mut strh = Vec::new();
    strh.extend_from_slice(b"vids");
    strh.extend_from_slice(b"HFYU");
    for v in [0u32, 0, 0, 1, 5, 0, n, largest, 0xFFFF_FFFF, 0] {
        strh.extend_from_slice(&v.to_le_bytes());
    }
    for v in [0u16, 0, width as u16, height as u16] {
        strh.extend_from_slice(&v.to_le_bytes());
    }
    let mut strl = chunk(b"strh", &strh);
    strl.extend(chunk(b"strf", strf));
    let mut hdrl = chunk(b"avih", &avih);
    hdrl.extend(list(b"LIST", b"strl", &strl));

    let mut movi = Vec::new();
    let mut idx = Vec::new();
    for f in frames {
        // Index offsets are relative to the `movi` form type.
        let offset = 4 + movi.len() as u32;
        idx.extend_from_slice(b"00dc");
        idx.extend_from_slice(&0x10u32.to_le_bytes());
        idx.extend_from_slice(&offset.to_le_bytes());
        idx.extend_from_slice(&(f.len() as u32).to_le_bytes());
        movi.extend(chunk(b"00dc", f));
    }
    let mut body = list(b"LIST", b"hdrl", &hdrl);
    body.extend(list(b"LIST", b"movi", &movi));
    body.extend(chunk(b"idx1", &idx));
    std::fs::write(path, list(b"RIFF", b"AVI ", &body)).unwrap();
}

/// Smooth gradient + a few bits of noise, varied per channel, so the
/// residual histograms look like real content and the first pixel is
/// not zero in any channel.
fn synth_raster(len: usize, bpp: usize, row_bytes: usize, seed: u32) -> Vec<u8> {
    let mut state = seed | 1;
    (0..len)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let (row, col) = (i / row_bytes, (i % row_bytes) / bpp);
            let base = 40 + (row * 3 + col * 2) / 2 + (i % bpp) * 23;
            (base as u32).wrapping_add(state & 0x0F) as u8
        })
        .collect()
}

fn methods_for(family: PixelFamily) -> &'static [Method] {
    if family.is_rgb() {
        &[
            Method::PredictOld,
            Method::Left,
            Method::LeftDecorr,
            Method::GradientDecorr,
        ]
    } else {
        &[
            Method::PredictOld,
            Method::Left,
            Method::Gradient,
            Method::Median,
        ]
    }
}

#[test]
fn our_streams_decode_bit_exact_in_black_box() {
    if !ffmpeg_available() {
        eprintln!("skip: ffmpeg not on PATH");
        return;
    }
    let dir = TempDir::new("ours");
    // Widths are multiples of 4 for YUY2 (the black-box decoder
    // declines other widths for median); RGB adds odd widths.
    let yuv_shapes: &[(u32, u32)] = &[
        (4, 1),
        (4, 2),
        (4, 3),
        (8, 5),
        (16, 16),
        (64, 48),
        (64, 288),
        (64, 289),
        (64, 290),
        (36, 321),
        (320, 240),
        (720, 576),
    ];
    let rgb_shapes: &[(u32, u32)] = &[
        (1, 1),
        (1, 9),
        (3, 3),
        (4, 2),
        (16, 16),
        (33, 17),
        (64, 48),
        (64, 288),
        (5, 291),
        (64, 290),
        (34, 321),
        (320, 240),
        (720, 576),
    ];
    let mut cells = 0usize;
    let mut not_comparable = 0usize;
    let mut failures = Vec::new();
    for family in [PixelFamily::Yuy2, PixelFamily::Rgb24, PixelFamily::Rgb32] {
        let (raw_fmt, shapes) = match family {
            PixelFamily::Yuy2 => ("yuyv422", yuv_shapes),
            PixelFamily::Rgb24 => ("bgr24", rgb_shapes),
            PixelFamily::Rgb32 => ("bgra", rgb_shapes),
        };
        for &method in methods_for(family) {
            for mode in [
                ExtradataMode::ClassicV2,
                ExtradataMode::CustomV2,
                ExtradataMode::V1xCompat,
            ] {
                // Each raster once under the height rule, plus — where
                // the header can say so — once with the opposite
                // field-coding choice (spec/06 §1.2 interlace byte).
                let runs = shapes.iter().flat_map(|&(w, h)| {
                    let against = if h > 288 {
                        FieldCoding::Plain
                    } else {
                        FieldCoding::Fields
                    };
                    let forced = (mode != ExtradataMode::V1xCompat && w <= 64 && h >= 3)
                        .then_some((w, h, against));
                    std::iter::once((w, h, FieldCoding::Auto)).chain(forced)
                });
                for (width, height, field_coding) in runs {
                    cells += 1;
                    let name =
                        format!("{family:?}-{method:?}-{mode:?}-{width}x{height}-{field_coding:?}");
                    let row_bytes = family.row_bytes(width);
                    let pixels = synth_raster(
                        row_bytes * height as usize,
                        family.bytes_per_pixel_step().min(4),
                        row_bytes,
                        0x9E37_79B9 ^ (width << 12) ^ height,
                    );
                    let options = EncodeOptions::new()
                        .with_method(MethodSelection::Fixed(method))
                        .with_mode(mode)
                        .with_field_coding(field_coding);
                    let (strf, frame) =
                        match encode_frame_with_options(family, width, height, &pixels, &options) {
                            Ok(v) => (v.strf, v.frame),
                            Err(e) => {
                                failures.push(format!("{name}: encode failed: {e}"));
                                continue;
                            }
                        };
                    if field_coding == FieldCoding::Auto {
                        // The options entry point and the classic one
                        // are the same encoder.
                        let classic =
                            encode_frame_with_mode(family, method, width, height, &pixels, mode)
                                .unwrap();
                        assert_eq!((&strf, &frame), (&classic.0, &classic.1), "{name}");
                    }
                    // Own decoder first: the two decoders are compared
                    // against the same source raster.
                    let config = StreamConfig::parse_bitmapinfoheader(&strf).unwrap();
                    let own = decode_frame(&config, &frame).unwrap();
                    assert_eq!(own.pixels, pixels, "{name}: self round trip");

                    // spec/01 §3.1 makes method byte `-2` legal next to
                    // extradata (the original decoder special-cases it,
                    // spec/02 §1.1) and spec/03 §4 makes it the same
                    // bitstream as `left`. The black-box decoder reads
                    // that byte as predictor bit flags instead, so the
                    // cell is not comparable; only the self round trip
                    // above stands for it.
                    if method == Method::PredictOld && mode != ExtradataMode::V1xCompat {
                        not_comparable += 1;
                        continue;
                    }
                    let avi = dir.path(&format!("{name}.avi"));
                    write_avi(&avi, &strf, std::slice::from_ref(&frame), width, height);
                    let raw = dir.path(&format!("{name}.raw"));
                    let out = Command::new("ffmpeg")
                        .args(["-v", "warning", "-y", "-i"])
                        .arg(&avi)
                        .args(["-pix_fmt", raw_fmt, "-f", "rawvideo"])
                        .arg(&raw)
                        .output()
                        .unwrap();
                    // Lines from the raw-output side of the tool (it
                    // comments on the bitrate of tiny rasters) say
                    // nothing about the stream under test.
                    let stderr = String::from_utf8_lossy(&out.stderr);
                    let diag = stderr
                        .lines()
                        .filter(|l| !l.contains("enc:rawvideo"))
                        .collect::<Vec<_>>()
                        .join(" | ");
                    if !out.status.success() || !diag.trim().is_empty() {
                        failures.push(format!("{name}: black-box diagnostics: {}", diag.trim()));
                        continue;
                    }
                    let decoded = std::fs::read(&raw).unwrap();
                    let want = if family.is_rgb() {
                        flip_rows(&pixels, row_bytes)
                    } else {
                        pixels
                    };
                    if decoded != want {
                        let diff = decoded.iter().zip(&want).filter(|(a, b)| a != b).count();
                        failures.push(format!(
                            "{name}: {diff}/{} bytes differ ({} decoded)",
                            want.len(),
                            decoded.len()
                        ));
                    }
                }
            }
        }
    }
    eprintln!(
        "ours → foreign: {cells} cells; {} accepted without diagnostics and bit-exact; \
         {not_comparable} not comparable (method byte -2 next to extradata); {} failed",
        cells - not_comparable - failures.len(),
        failures.len()
    );
    for f in &failures {
        eprintln!("  {f}");
    }
    assert!(
        failures.is_empty(),
        "{} of our cells failed",
        failures.len()
    );
}

// ───────────────────────── registry path, both ways ─────────────────────────

/// Framework frames carry the top row first. The registry decoder must
/// hand out exactly the raster a black-box decoder dumps (no row flip
/// left to the caller) and announce its size and pixel format; the
/// registry encoder must turn such frames — including ones whose rows
/// carry padding — into streams the black-box decoder reads back.
#[cfg(feature = "registry")]
#[test]
fn registry_frames_are_top_down_both_ways() {
    use oxideav_core::{
        CodecId, CodecParameters, Frame, Packet, PixelFormat, TimeBase, VideoFrame, VideoPlane,
    };
    use oxideav_huffyuv::{CODEC_ID_STR, make_decoder, make_encoder};

    if !ffmpeg_available() {
        eprintln!("skip: ffmpeg not on PATH");
        return;
    }
    let dir = TempDir::new("registry");
    let (width, height, frames) = (36u32, 290u32, 3usize);
    for (pix_fmt, raw_fmt, family, label, pred) in [
        (
            "yuv422p",
            "yuyv422",
            PixelFamily::Yuy2,
            PixelFormat::Yuyv422,
            "median",
        ),
        (
            "rgb24",
            "bgr24",
            PixelFamily::Rgb24,
            PixelFormat::Bgr24,
            "plane",
        ),
        (
            "bgra",
            "bgra",
            PixelFamily::Rgb32,
            PixelFormat::Bgra,
            "left",
        ),
    ] {
        // Foreign file → registry decoder.
        let graph = format!(
            "testsrc2=size={width}x{height}:rate=5,noise=alls=12:allf=t:all_seed=3,format={pix_fmt}"
        );
        let avi = dir.path(&format!("{pix_fmt}.avi"));
        let raw = dir.path(&format!("{pix_fmt}.raw"));
        let n = frames.to_string();
        let ok = Command::new("ffmpeg")
            .args([
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                &graph,
                "-frames:v",
                &n,
            ])
            .args(["-c:v", "huffyuv", "-pred", pred, "-flags", "+ilme"])
            .arg(&avi)
            .status()
            .unwrap()
            .success();
        assert!(ok, "{pix_fmt}: black-box encode");
        let ok = Command::new("ffmpeg")
            .args([
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                &graph,
                "-frames:v",
                &n,
            ])
            .args(["-pix_fmt", raw_fmt, "-f", "rawvideo"])
            .arg(&raw)
            .status()
            .unwrap()
            .success();
        assert!(ok, "{pix_fmt}: source dump");
        let source = std::fs::read(&raw).unwrap();
        let row_bytes = family.row_bytes(width);
        let frame_len = row_bytes * height as usize;
        assert_eq!(source.len(), frame_len * frames);

        let stream = read_avi(&avi);
        let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        params.extradata = stream.strf.clone();
        let mut dec = make_decoder(&params).expect("make_decoder");
        let announced = dec.output_params().expect("output_params");
        assert_eq!(
            (announced.width, announced.height, announced.pixel_format),
            (Some(width), Some(height), Some(label)),
            "{pix_fmt}: announced output shape"
        );
        let mut decoded_frames = Vec::new();
        for (i, chunk) in stream.frames.iter().enumerate() {
            dec.send_packet(&Packet::new(0, TimeBase::MICROS, chunk.clone()))
                .unwrap();
            let Frame::Video(vf) = dec.receive_frame().unwrap() else {
                panic!("{pix_fmt}: expected a video frame");
            };
            assert_eq!(vf.planes.len(), 1);
            assert_eq!(vf.planes[0].stride, row_bytes);
            assert_eq!(
                vf.planes[0].data,
                &source[i * frame_len..(i + 1) * frame_len],
                "{pix_fmt}: registry frame {i} must equal the top-down source"
            );
            decoded_frames.push(vf.planes[0].data.clone());
        }

        // Registry encoder (rows padded by 5 bytes) → black-box decoder,
        // as one multi-frame stream per extradata mode.
        for mode in ["classic-v2", "custom-v2", "v1x"] {
            let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
            params.width = Some(width);
            params.height = Some(height);
            params.pixel_format = Some(label);
            params.options.insert("mode", mode);
            let mut enc = make_encoder(&params).expect("make_encoder");
            let stride = row_bytes + 5;
            let mut packets = Vec::new();
            // custom-v2 pins the first frame's tables for the stream;
            // they cover every symbol, so later pictures still encode.
            for (i, picture) in decoded_frames.iter().enumerate() {
                let mut data = Vec::with_capacity(stride * height as usize);
                for row in picture.chunks_exact(row_bytes) {
                    data.extend_from_slice(row);
                    data.extend_from_slice(&[0xA5; 5]);
                }
                let frame = Frame::Video(VideoFrame {
                    pts: Some(i as i64),
                    planes: vec![VideoPlane { stride, data }],
                });
                enc.send_frame(&frame).expect("send_frame");
                packets.push(enc.receive_packet().expect("receive_packet").data.to_vec());
            }
            let ours = dir.path(&format!("{pix_fmt}-{mode}.avi"));
            write_avi(
                &ours,
                &enc.output_params().extradata,
                &packets,
                width,
                height,
            );
            let back = dir.path(&format!("{pix_fmt}-{mode}.raw"));
            let out = Command::new("ffmpeg")
                .args(["-v", "warning", "-y", "-i"])
                .arg(&ours)
                .args(["-pix_fmt", raw_fmt, "-f", "rawvideo"])
                .arg(&back)
                .output()
                .unwrap();
            assert!(
                out.status.success() && out.stderr.is_empty(),
                "{pix_fmt}/{mode}: black-box diagnostics: {}",
                String::from_utf8_lossy(&out.stderr)
            );
            assert_eq!(
                std::fs::read(&back).unwrap(),
                source,
                "{pix_fmt}/{mode}: black-box decode of the registry stream"
            );
        }
    }
}

// ───────────────────────── size against the black box ─────────────────────────

/// Same picture, same predictor: the residual symbols are fixed by the
/// format, so the only freedom is the Huffman tables. Per-frame
/// optimal tables (`CustomV2`) can therefore never lose to another
/// encoder's tables on a single frame; the classic tables may. Prints
/// the comparison (quoted in BENCHMARKS.md) and asserts the bound.
#[test]
fn encoded_size_against_black_box_encoder() {
    use oxideav_huffyuv::encoder::encode_frame_auto;

    if !ffmpeg_available() {
        eprintln!("skip: ffmpeg not on PATH");
        return;
    }
    let dir = TempDir::new("size");
    eprintln!(
        "| source | format | predictor | black box | ours classic | ours custom | ours auto+custom |"
    );
    for (source, graph_head) in [
        (
            "pattern + noise",
            "testsrc2=size=320x240:rate=5,noise=alls=12:allf=t:all_seed=7",
        ),
        ("pattern", "testsrc2=size=320x240:rate=5"),
        ("mandelbrot", "mandelbrot=size=320x240:rate=5"),
    ] {
        for (pix_fmt, raw_fmt, family) in [
            ("yuv422p", "yuyv422", PixelFamily::Yuy2),
            ("rgb24", "bgr24", PixelFamily::Rgb24),
            ("bgra", "bgra", PixelFamily::Rgb32),
        ] {
            let graph = format!("{graph_head},format={pix_fmt}");
            let raw = dir.path("src.raw");
            let ok = Command::new("ffmpeg")
                .args([
                    "-v",
                    "error",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    &graph,
                    "-frames:v",
                    "1",
                ])
                .args(["-pix_fmt", raw_fmt, "-f", "rawvideo"])
                .arg(&raw)
                .status()
                .unwrap()
                .success();
            assert!(ok, "source dump");
            let top_down = std::fs::read(&raw).unwrap();
            let row_bytes = family.row_bytes(320);
            let pixels = if family.is_rgb() {
                flip_rows(&top_down, row_bytes)
            } else {
                top_down
            };
            let preds: &[(&str, Method)] = if family.is_rgb() {
                &[
                    ("left", Method::LeftDecorr),
                    ("plane", Method::GradientDecorr),
                ]
            } else {
                &[
                    ("left", Method::Left),
                    ("plane", Method::Gradient),
                    ("median", Method::Median),
                ]
            };
            for &(pred, method) in preds {
                let avi = dir.path("bb.avi");
                let ok = Command::new("ffmpeg")
                    .args([
                        "-v",
                        "error",
                        "-y",
                        "-f",
                        "lavfi",
                        "-i",
                        &graph,
                        "-frames:v",
                        "1",
                    ])
                    .args(["-c:v", "huffyuv", "-pred", pred])
                    .arg(&avi)
                    .status()
                    .unwrap()
                    .success();
                assert!(ok, "black-box encode");
                let foreign = read_avi(&avi);
                let config = StreamConfig::parse_bitmapinfoheader(&foreign.strf).unwrap();
                assert_eq!(config.method, method, "{pix_fmt}/{pred}: method mapping");
                let theirs = foreign.frames[0].len();
                let size = |mode| {
                    encode_frame_with_mode(family, method, 320, 240, &pixels, mode)
                        .unwrap()
                        .1
                        .len()
                };
                let classic = size(ExtradataMode::ClassicV2);
                let custom = size(ExtradataMode::CustomV2);
                let (_, auto, picked) = encode_frame_auto(
                    family,
                    MethodSelection::Auto,
                    320,
                    240,
                    &pixels,
                    ExtradataMode::CustomV2,
                )
                .unwrap();
                eprintln!(
                    "| {source} | {pix_fmt} | {pred} | {theirs} | {classic} ({:+.1} %) | {custom} ({:+.1} %) | {} ({:+.1} %, {picked:?}) |",
                    100.0 * (classic as f64 / theirs as f64 - 1.0),
                    100.0 * (custom as f64 / theirs as f64 - 1.0),
                    auto.len(),
                    100.0 * (auto.len() as f64 / theirs as f64 - 1.0),
                );
                assert!(
                    custom <= theirs,
                    "{source}/{pix_fmt}/{pred}: per-frame optimal tables ({custom}) \
                     must not exceed the black-box frame ({theirs})"
                );
                assert!(
                    auto.len() <= custom,
                    "auto-selection must not lose to a fixed method"
                );
            }
        }
    }
}
