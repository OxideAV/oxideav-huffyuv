//! Black-box ffmpeg-against-our-decoder round-trip.
//!
//! Skipped when no `ffmpeg` binary is on `$PATH`. We:
//!
//! 1. Generate a tiny `lavfi`-synthesised yuv422p source via ffmpeg
//!    (deterministic, no input file needed).
//! 2. Pipe it through `ffmpeg -c:v huffyuv -f avi` to produce an AVI
//!    encoded with the upstream HuffYUV encoder.
//! 3. Pipe the same source through `ffmpeg -c:v rawvideo -f rawvideo`
//!    to capture the reference pixel matrix.
//! 4. Pull the `00dc` packet payload + `strf` extradata out of the AVI
//!    container ourselves (we do the bare minimum AVI walking we need)
//!    and feed both through `oxideav_huffyuv::decoder` via the codec
//!    registry.
//! 5. Assert every decoded sample matches the reference rawvideo byte
//!    for byte (HuffYUV is lossless).
//!
//! These tests skip cleanly (no failure) when `ffmpeg` isn't on
//! `$PATH`, so they're safe to keep enabled in CI without
//! `#[ignore]`. The pure-Rust `synth_left` test exercises the LEFT
//! decoder without external tooling.

#![allow(clippy::needless_range_loop)]

use std::process::{Command, Stdio};

use oxideav_core::time::TimeBase;
use oxideav_core::{CodecId, CodecParameters, CodecRegistry, Packet, PixelFormat};
use oxideav_huffyuv::register;

const W: u32 = 16;
const H: u32 = 16;

fn ffmpeg_available() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn run_ffmpeg(args: &[&str]) -> Result<Vec<u8>, String> {
    let out = Command::new("ffmpeg")
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("spawning ffmpeg: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "ffmpeg failed: {}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(out.stdout)
}

/// Walk a RIFF AVI to find the `strf` extradata (after the
/// 40-byte BITMAPINFOHEADER) and the first video `00dc` chunk's payload.
/// Returns `(extradata, packet_payload)` on success.
fn extract_huffyuv_from_avi(avi: &[u8]) -> Result<(Vec<u8>, Vec<u8>), String> {
    if &avi[0..4] != b"RIFF" || &avi[8..12] != b"AVI " {
        return Err("not a RIFF/AVI".into());
    }
    let mut extradata: Option<Vec<u8>> = None;
    let mut packet: Option<Vec<u8>> = None;
    // Skip the leading RIFF<size> (8 bytes) + AVI form-type (4 bytes).
    walk_list(&avi[12..], false, &mut extradata, &mut packet)?;
    Ok((
        extradata.ok_or("no strf extradata")?,
        packet.ok_or("no 00dc packet")?,
    ))
}

fn walk_list(
    body: &[u8],
    in_movi: bool,
    extradata: &mut Option<Vec<u8>>,
    packet: &mut Option<Vec<u8>>,
) -> Result<(), String> {
    let mut p = 0;
    while p + 8 <= body.len() {
        let id = &body[p..p + 4];
        let raw_size =
            u32::from_le_bytes([body[p + 4], body[p + 5], body[p + 6], body[p + 7]]) as usize;
        let data_start = p + 8;
        // Some muxers leave the top-level LIST size as 0xFFFFFFFF when
        // they can't predict it (the OpenDML "unknown size" sentinel).
        // Treat that as "extends to end of parent".
        let size = if data_start + raw_size > body.len() {
            body.len() - data_start
        } else {
            raw_size
        };
        let data_end = data_start + size;
        if id == b"LIST" {
            // 4-byte list type, then nested chunks.
            let list_type = &body[data_start..data_start + 4];
            let nested_in_movi = in_movi || list_type == b"movi";
            walk_list(
                &body[data_start + 4..data_end],
                nested_in_movi,
                extradata,
                packet,
            )?;
        } else if id == b"strf" {
            // Skip the 40-byte BITMAPINFOHEADER, the rest is extradata.
            if size > 40 {
                *extradata = Some(body[data_start + 40..data_end].to_vec());
            }
        } else if (id == b"00dc" || id == b"00db") && in_movi && size > 0 && packet.is_none() {
            *packet = Some(body[data_start..data_end].to_vec());
        }
        // Pad to even.
        let advance = 8 + size + (size & 1);
        p += advance;
    }
    Ok(())
}

/// Encode a 1-frame testsrc through ffmpeg's huffyuv encoder, run our
/// decoder against it, and assert byte-equality against the rawvideo
/// reference for the same source.
fn cross_decode_huffyuv(
    pred: &str,
    pix_fmt: &str,
    pf: PixelFormat,
    width: u32,
    height: u32,
    codec: &str,
) {
    let avi = run_ffmpeg(&[
        "-y",
        "-f",
        "lavfi",
        "-i",
        &format!("testsrc=size={width}x{height}:rate=1:duration=0.04"),
        "-c:v",
        codec,
        "-pred",
        pred,
        "-pix_fmt",
        pix_fmt,
        "-f",
        "avi",
        "-",
    ])
    .expect("ffmpeg encode");
    let raw = run_ffmpeg(&[
        "-y",
        "-f",
        "lavfi",
        "-i",
        &format!("testsrc=size={width}x{height}:rate=1:duration=0.04"),
        "-pix_fmt",
        pix_fmt,
        "-vframes",
        "1",
        "-f",
        "rawvideo",
        "-",
    ])
    .expect("ffmpeg rawvideo");

    let (extradata, packet) = extract_huffyuv_from_avi(&avi).expect("avi walk");

    let mut reg = CodecRegistry::new();
    register(&mut reg);
    let mut params = CodecParameters::video(CodecId::new(codec));
    params.width = Some(width);
    params.height = Some(height);
    params.pixel_format = Some(pf);
    params.extradata = extradata;
    let mut decoder = reg.make_decoder(&params).expect("make_decoder");
    let pkt = Packet::new(0, TimeBase::new(1, 25), packet).with_keyframe(true);
    decoder.send_packet(&pkt).expect("send_packet");
    let frame = decoder.receive_frame().expect("receive_frame");
    let video = match frame {
        oxideav_core::Frame::Video(v) => v,
        _ => panic!("expected video"),
    };
    assert_planes_match(&video, &raw, width as usize, height as usize, pf);
}

fn assert_planes_match(
    video: &oxideav_core::VideoFrame,
    raw: &[u8],
    width: usize,
    height: usize,
    pf: PixelFormat,
) {
    let mut ref_off = 0usize;
    let plane_dims: Vec<(usize, usize, usize)> = match pf {
        PixelFormat::Yuv422P => vec![
            (width, height, 1),
            (width / 2, height, 1),
            (width / 2, height, 1),
        ],
        PixelFormat::Yuv420P => vec![
            (width, height, 1),
            (width / 2, height / 2, 1),
            (width / 2, height / 2, 1),
        ],
        PixelFormat::Yuv411P => vec![
            (width, height, 1),
            (width / 4, height, 1),
            (width / 4, height, 1),
        ],
        PixelFormat::Yuv444P => vec![(width, height, 1); 3],
        PixelFormat::Gray8 => vec![(width, height, 1)],
        // GBRP/GBRAP 8-bit stored in Gbrp10Le/Gbrap10Le containers (u16 LE, low 8 bits used).
        // ffmpeg rawvideo `gbrp` is 1 byte/sample; we compare low byte only.
        PixelFormat::Gbrp10Le => {
            assert_eq!(video.planes.len(), 3, "gbrp plane count");
            let n = width * height;
            for p in 0..3 {
                let ref_plane = &raw[p * n..(p + 1) * n];
                for y in 0..height {
                    for x in 0..width {
                        let decoded_lo = video.planes[p].data[y * video.planes[p].stride + x * 2];
                        let decoded_hi =
                            video.planes[p].data[y * video.planes[p].stride + x * 2 + 1];
                        let ref_val = ref_plane[y * width + x];
                        assert_eq!(
                            decoded_lo, ref_val,
                            "gbrp plane {p} pixel [{y}][{x}]: decoded lo={decoded_lo} ref={ref_val}"
                        );
                        assert_eq!(
                            decoded_hi, 0,
                            "gbrp plane {p} pixel [{y}][{x}]: hi byte should be 0"
                        );
                    }
                }
            }
            return;
        }
        PixelFormat::Gbrap10Le => {
            assert_eq!(video.planes.len(), 4, "gbrap plane count");
            let n = width * height;
            for p in 0..4 {
                let ref_plane = &raw[p * n..(p + 1) * n];
                for y in 0..height {
                    for x in 0..width {
                        let decoded_lo = video.planes[p].data[y * video.planes[p].stride + x * 2];
                        let decoded_hi =
                            video.planes[p].data[y * video.planes[p].stride + x * 2 + 1];
                        let ref_val = ref_plane[y * width + x];
                        assert_eq!(
                            decoded_lo, ref_val,
                            "gbrap plane {p} pixel [{y}][{x}]: decoded lo={decoded_lo} ref={ref_val}"
                        );
                        assert_eq!(
                            decoded_hi, 0,
                            "gbrap plane {p} pixel [{y}][{x}]: hi byte should be 0"
                        );
                    }
                }
            }
            return;
        }
        _ => panic!("assert_planes_match: unsupported format {pf:?} (extend the helper)"),
    };
    assert_eq!(video.planes.len(), plane_dims.len(), "plane count");
    for (idx, &(w, h, bytes)) in plane_dims.iter().enumerate() {
        let row_bytes = w * bytes;
        for y in 0..h {
            let row = &video.planes[idx].data[y * video.planes[idx].stride..][..row_bytes];
            let r = &raw[ref_off..ref_off + row_bytes];
            assert_eq!(row, r, "plane {idx} row {y} mismatch");
            ref_off += row_bytes;
        }
    }
}

#[test]
fn ffmpeg_huffyuv_round_trip_yuv422p_left() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not on PATH; skipping");
        return;
    }
    cross_decode_huffyuv("left", "yuv422p", PixelFormat::Yuv422P, W, H, "huffyuv");
}

#[test]
fn ffmpeg_huffyuv_round_trip_yuv422p_plane() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not on PATH; skipping");
        return;
    }
    cross_decode_huffyuv("plane", "yuv422p", PixelFormat::Yuv422P, W, H, "huffyuv");
}

#[test]
fn ffmpeg_huffyuv_round_trip_yuv422p_median() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not on PATH; skipping");
        return;
    }
    cross_decode_huffyuv("median", "yuv422p", PixelFormat::Yuv422P, W, H, "huffyuv");
}

#[test]
fn ffmpeg_ffvhuff_round_trip_yuv420p_left() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not on PATH; skipping");
        return;
    }
    // ffmpeg's `huffyuv` (legacy v2-only) doesn't accept yuv420p
    // input — it silently transcodes to yuv422p. ffvhuff emits v2
    // bsbpp=12 for yuv420p, which our decoder handles via the v2
    // yuv420 codepath.
    cross_decode_huffyuv("left", "yuv420p", PixelFormat::Yuv420P, W, H, "ffvhuff");
}

#[test]
fn ffmpeg_ffvhuff_round_trip_yuv444p_plane() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not on PATH; skipping");
        return;
    }
    cross_decode_huffyuv("plane", "yuv444p", PixelFormat::Yuv444P, W, H, "ffvhuff");
}

#[test]
fn ffmpeg_ffvhuff_round_trip_gray8_left() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not on PATH; skipping");
        return;
    }
    cross_decode_huffyuv("left", "gray", PixelFormat::Gray8, W, H, "ffvhuff");
}

#[test]
fn ffmpeg_ffvhuff_round_trip_gray8_median() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not on PATH; skipping");
        return;
    }
    cross_decode_huffyuv("median", "gray", PixelFormat::Gray8, W, H, "ffvhuff");
}

#[test]
fn ffmpeg_ffvhuff_round_trip_yuv444p_median() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not on PATH; skipping");
        return;
    }
    cross_decode_huffyuv("median", "yuv444p", PixelFormat::Yuv444P, W, H, "ffvhuff");
}

/// 12-bit HBD decode: yuv444p12le via ffvhuff.
#[test]
fn ffmpeg_ffvhuff_round_trip_yuv444p12le_left() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not on PATH; skipping");
        return;
    }
    let avi = run_ffmpeg(&[
        "-y",
        "-f",
        "lavfi",
        "-i",
        &format!("testsrc=size={W}x{H}:rate=1:duration=0.04"),
        "-c:v",
        "ffvhuff",
        "-pred",
        "left",
        "-pix_fmt",
        "yuv444p12le",
        "-f",
        "avi",
        "-",
    ])
    .expect("ffmpeg encode");
    let raw = run_ffmpeg(&[
        "-y",
        "-f",
        "lavfi",
        "-i",
        &format!("testsrc=size={W}x{H}:rate=1:duration=0.04"),
        "-pix_fmt",
        "yuv444p12le",
        "-vframes",
        "1",
        "-f",
        "rawvideo",
        "-",
    ])
    .expect("ffmpeg rawvideo");

    let (extradata, packet) = extract_huffyuv_from_avi(&avi).expect("avi walk");
    let mut reg = CodecRegistry::new();
    register(&mut reg);
    let mut params = CodecParameters::video(CodecId::new("ffvhuff"));
    params.width = Some(W);
    params.height = Some(H);
    params.pixel_format = Some(PixelFormat::Yuv444P12Le);
    params.extradata = extradata;
    let mut decoder = reg.make_decoder(&params).expect("make_decoder");
    let pkt = Packet::new(0, TimeBase::new(1, 25), packet).with_keyframe(true);
    decoder.send_packet(&pkt).expect("send_packet");
    let frame = decoder.receive_frame().expect("receive_frame");
    let video = match frame {
        oxideav_core::Frame::Video(v) => v,
        _ => panic!("expected video"),
    };
    let mut ref_off = 0usize;
    for i in 0..3 {
        let row_bytes = (W as usize) * 2;
        for y in 0..(H as usize) {
            let row = &video.planes[i].data[y * video.planes[i].stride..][..row_bytes];
            let r = &raw[ref_off..ref_off + row_bytes];
            assert_eq!(row, r, "plane {i} row {y} mismatch");
            ref_off += row_bytes;
        }
    }
}

/// 16-bit gray HBD decode + 2-raw-bit splice (trace doc §5.5).
#[test]
fn ffmpeg_ffvhuff_round_trip_gray16le_left() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not on PATH; skipping");
        return;
    }
    let avi = run_ffmpeg(&[
        "-y",
        "-f",
        "lavfi",
        "-i",
        &format!("testsrc=size={W}x{H}:rate=1:duration=0.04"),
        "-c:v",
        "ffvhuff",
        "-pred",
        "left",
        "-pix_fmt",
        "gray16le",
        "-f",
        "avi",
        "-",
    ])
    .expect("ffmpeg encode");
    let raw = run_ffmpeg(&[
        "-y",
        "-f",
        "lavfi",
        "-i",
        &format!("testsrc=size={W}x{H}:rate=1:duration=0.04"),
        "-pix_fmt",
        "gray16le",
        "-vframes",
        "1",
        "-f",
        "rawvideo",
        "-",
    ])
    .expect("ffmpeg rawvideo");
    let (extradata, packet) = extract_huffyuv_from_avi(&avi).expect("avi walk");
    let mut reg = CodecRegistry::new();
    register(&mut reg);
    let mut params = CodecParameters::video(CodecId::new("ffvhuff"));
    params.width = Some(W);
    params.height = Some(H);
    params.pixel_format = Some(PixelFormat::Gray16Le);
    params.extradata = extradata;
    let mut decoder = reg.make_decoder(&params).expect("make_decoder");
    let pkt = Packet::new(0, TimeBase::new(1, 25), packet).with_keyframe(true);
    decoder.send_packet(&pkt).expect("send_packet");
    let video = match decoder.receive_frame().expect("receive_frame") {
        oxideav_core::Frame::Video(v) => v,
        _ => panic!("expected video"),
    };
    let row_bytes = (W as usize) * 2;
    for y in 0..(H as usize) {
        let row = &video.planes[0].data[y * video.planes[0].stride..][..row_bytes];
        let r = &raw[y * row_bytes..(y + 1) * row_bytes];
        assert_eq!(row, r, "row {y} mismatch");
    }
}

/// HBD decode: 10-bit yuv422p via ffvhuff. Compares against rawvideo
/// 10-bit-LE reference.
#[test]
fn ffmpeg_ffvhuff_round_trip_yuv422p10le_left() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not on PATH; skipping");
        return;
    }
    let avi = run_ffmpeg(&[
        "-y",
        "-f",
        "lavfi",
        "-i",
        &format!("testsrc=size={W}x{H}:rate=1:duration=0.04"),
        "-c:v",
        "ffvhuff",
        "-pred",
        "left",
        "-pix_fmt",
        "yuv422p10le",
        "-f",
        "avi",
        "-",
    ])
    .expect("ffmpeg encode");
    let raw = run_ffmpeg(&[
        "-y",
        "-f",
        "lavfi",
        "-i",
        &format!("testsrc=size={W}x{H}:rate=1:duration=0.04"),
        "-pix_fmt",
        "yuv422p10le",
        "-vframes",
        "1",
        "-f",
        "rawvideo",
        "-",
    ])
    .expect("ffmpeg rawvideo");

    let (extradata, packet) = extract_huffyuv_from_avi(&avi).expect("avi walk");
    let mut reg = CodecRegistry::new();
    register(&mut reg);
    let mut params = CodecParameters::video(CodecId::new("ffvhuff"));
    params.width = Some(W);
    params.height = Some(H);
    params.pixel_format = Some(PixelFormat::Yuv422P10Le);
    params.extradata = extradata;
    let mut decoder = reg.make_decoder(&params).expect("make_decoder");
    let pkt = Packet::new(0, TimeBase::new(1, 25), packet).with_keyframe(true);
    decoder.send_packet(&pkt).expect("send_packet");
    let frame = decoder.receive_frame().expect("receive_frame");
    let video = match frame {
        oxideav_core::Frame::Video(v) => v,
        _ => panic!("expected video"),
    };
    // Each plane stride is width*2 bytes; raw rawvideo packs row by row.
    let mut ref_off = 0usize;
    let widths = [W as usize, (W as usize) / 2, (W as usize) / 2];
    let heights = [H as usize, H as usize, H as usize];
    for (i, (&w, &h)) in widths.iter().zip(heights.iter()).enumerate() {
        let row_bytes = w * 2;
        for y in 0..h {
            let row = &video.planes[i].data[y * video.planes[i].stride..][..row_bytes];
            let r = &raw[ref_off..ref_off + row_bytes];
            assert_eq!(row, r, "plane {i} row {y} mismatch");
            ref_off += row_bytes;
        }
    }
}

/// YUV 4:1:1 planar via ffvhuff — decoded by us, compared to rawvideo.
#[test]
fn ffmpeg_ffvhuff_round_trip_yuv411p_left() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not on PATH; skipping");
        return;
    }
    // yuv411p: width must be a multiple of 4.
    cross_decode_huffyuv("left", "yuv411p", PixelFormat::Yuv411P, 16, 16, "ffvhuff");
}

#[test]
fn ffmpeg_ffvhuff_round_trip_yuv411p_plane() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not on PATH; skipping");
        return;
    }
    cross_decode_huffyuv("plane", "yuv411p", PixelFormat::Yuv411P, 16, 16, "ffvhuff");
}

/// GBRP 8-bit via ffvhuff — decoded by us into Gbrp10Le container,
/// compared sample-by-sample to the 8-bit rawvideo reference.
#[test]
fn ffmpeg_ffvhuff_round_trip_gbrp8_left() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not on PATH; skipping");
        return;
    }
    cross_decode_huffyuv("left", "gbrp", PixelFormat::Gbrp10Le, W, H, "ffvhuff");
}

#[test]
fn ffmpeg_ffvhuff_round_trip_gbrp8_median() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not on PATH; skipping");
        return;
    }
    cross_decode_huffyuv("median", "gbrp", PixelFormat::Gbrp10Le, W, H, "ffvhuff");
}

/// GBRAP 8-bit (with alpha) via ffvhuff.
#[test]
fn ffmpeg_ffvhuff_round_trip_gbrap8_left() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not on PATH; skipping");
        return;
    }
    cross_decode_huffyuv("left", "gbrap", PixelFormat::Gbrap10Le, W, H, "ffvhuff");
}

/// Interlaced YUV 4:2:2 (v2 huffyuv) — decoded by us, compared to rawvideo.
/// ffmpeg -flags +ilme+ildct -top 1 forces top-field-first interlaced.
#[test]
fn ffmpeg_huffyuv_round_trip_yuv422p_interlaced_left() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not on PATH; skipping");
        return;
    }
    let avi = run_ffmpeg(&[
        "-y",
        "-f",
        "lavfi",
        "-i",
        &format!("testsrc=size={W}x{H}:rate=1:duration=0.04"),
        "-c:v",
        "huffyuv",
        "-pred",
        "left",
        "-pix_fmt",
        "yuv422p",
        "-flags",
        "+ilme+ildct",
        "-top",
        "1",
        "-f",
        "avi",
        "-",
    ])
    .expect("ffmpeg encode");
    let raw = run_ffmpeg(&[
        "-y",
        "-f",
        "lavfi",
        "-i",
        &format!("testsrc=size={W}x{H}:rate=1:duration=0.04"),
        "-pix_fmt",
        "yuv422p",
        "-vframes",
        "1",
        "-f",
        "rawvideo",
        "-",
    ])
    .expect("ffmpeg rawvideo");
    let (extradata, packet) = extract_huffyuv_from_avi(&avi).expect("avi walk");
    let mut reg = CodecRegistry::new();
    register(&mut reg);
    let mut params = CodecParameters::video(CodecId::new("huffyuv"));
    params.width = Some(W);
    params.height = Some(H);
    params.pixel_format = Some(PixelFormat::Yuv422P);
    params.extradata = extradata;
    let mut decoder = reg.make_decoder(&params).expect("make_decoder");
    let pkt = Packet::new(0, TimeBase::new(1, 25), packet).with_keyframe(true);
    decoder.send_packet(&pkt).expect("send_packet");
    let frame = decoder.receive_frame().expect("receive_frame");
    let video = match frame {
        oxideav_core::Frame::Video(v) => v,
        _ => panic!("expected video"),
    };
    assert_planes_match(&video, &raw, W as usize, H as usize, PixelFormat::Yuv422P);
}

/// Self-roundtrip: encode gbrp8 with our encoder, decode with our decoder,
/// compare to the original planes.
#[test]
fn self_roundtrip_gbrp8_left() {
    let width = 16usize;
    let height = 8usize;
    let n = width * height;
    // Synthesise 3 planes (G, B, R) with distinct patterns.
    let g: Vec<u8> = (0..n).map(|i| ((i * 7 + 11) & 0xFF) as u8).collect();
    let b: Vec<u8> = (0..n).map(|i| ((i * 13 + 23) & 0xFF) as u8).collect();
    let r: Vec<u8> = (0..n).map(|i| ((i * 17 + 37) & 0xFF) as u8).collect();
    // Convert 8-bit planes to 16-bit LE words (low byte = value, hi = 0).
    let to_u16_plane = |src: &[u8]| -> Vec<u8> {
        let mut out = Vec::with_capacity(src.len() * 2);
        for &v in src {
            out.push(v);
            out.push(0);
        }
        out
    };
    let gp = to_u16_plane(&g);
    let bp = to_u16_plane(&b);
    let rp = to_u16_plane(&r);

    use oxideav_core::frame::VideoPlane;
    use oxideav_core::VideoFrame;
    let frame = VideoFrame {
        pts: Some(0),
        planes: vec![
            VideoPlane {
                stride: width * 2,
                data: gp.clone(),
            },
            VideoPlane {
                stride: width * 2,
                data: bp.clone(),
            },
            VideoPlane {
                stride: width * 2,
                data: rp.clone(),
            },
        ],
    };

    use oxideav_core::{CodecId, CodecParameters, CodecRegistry, Frame};
    let mut reg = CodecRegistry::new();
    register(&mut reg);
    let mut enc_params = CodecParameters::video(CodecId::new("ffvhuff"));
    enc_params.width = Some(width as u32);
    enc_params.height = Some(height as u32);
    enc_params.pixel_format = Some(PixelFormat::Gbrp10Le);
    let mut enc = reg.make_encoder(&enc_params).expect("encoder");
    enc.send_frame(&Frame::Video(frame)).unwrap();
    let pkt = enc.receive_packet().unwrap();
    let dec_extra = enc.output_params().extradata.clone();

    let mut dec_params = CodecParameters::video(CodecId::new("ffvhuff"));
    dec_params.width = Some(width as u32);
    dec_params.height = Some(height as u32);
    dec_params.pixel_format = Some(PixelFormat::Gbrp10Le);
    dec_params.extradata = dec_extra;
    let mut dec = reg.make_decoder(&dec_params).expect("decoder");
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 25), pkt.data).with_keyframe(true))
        .unwrap();
    let decoded = match dec.receive_frame().unwrap() {
        Frame::Video(v) => v,
        _ => panic!(),
    };
    assert_eq!(decoded.planes[0].data, gp, "G plane mismatch");
    assert_eq!(decoded.planes[1].data, bp, "B plane mismatch");
    assert_eq!(decoded.planes[2].data, rp, "R plane mismatch");
}

/// Self-roundtrip: encode yuv411p with our encoder, decode with our decoder.
#[test]
fn self_roundtrip_yuv411p_left() {
    let width = 16usize;
    let height = 8usize;
    let cw = width / 4;
    let ch = height;
    let yn = width * height;
    let cn = cw * ch;
    let y: Vec<u8> = (0..yn).map(|i| ((i * 7 + 11) & 0xFF) as u8).collect();
    let u: Vec<u8> = (0..cn).map(|i| ((i * 13 + 23) & 0xFF) as u8).collect();
    let v: Vec<u8> = (0..cn).map(|i| ((i * 17 + 37) & 0xFF) as u8).collect();

    use oxideav_core::frame::VideoPlane;
    use oxideav_core::VideoFrame;
    let frame = VideoFrame {
        pts: Some(0),
        planes: vec![
            VideoPlane {
                stride: width,
                data: y.clone(),
            },
            VideoPlane {
                stride: cw,
                data: u.clone(),
            },
            VideoPlane {
                stride: cw,
                data: v.clone(),
            },
        ],
    };

    use oxideav_core::{CodecId, CodecParameters, CodecRegistry, Frame};
    let mut reg = CodecRegistry::new();
    register(&mut reg);
    let mut enc_params = CodecParameters::video(CodecId::new("ffvhuff"));
    enc_params.width = Some(width as u32);
    enc_params.height = Some(height as u32);
    enc_params.pixel_format = Some(PixelFormat::Yuv411P);
    let mut enc = reg.make_encoder(&enc_params).expect("encoder");
    enc.send_frame(&Frame::Video(frame)).unwrap();
    let pkt = enc.receive_packet().unwrap();
    let dec_extra = enc.output_params().extradata.clone();

    let mut dec_params = CodecParameters::video(CodecId::new("ffvhuff"));
    dec_params.width = Some(width as u32);
    dec_params.height = Some(height as u32);
    dec_params.pixel_format = Some(PixelFormat::Yuv411P);
    dec_params.extradata = dec_extra;
    let mut dec = reg.make_decoder(&dec_params).expect("decoder");
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 25), pkt.data).with_keyframe(true))
        .unwrap();
    let decoded = match dec.receive_frame().unwrap() {
        Frame::Video(v) => v,
        _ => panic!(),
    };
    assert_eq!(decoded.planes[0].data, y, "Y plane mismatch");
    assert_eq!(decoded.planes[1].data, u, "U plane mismatch");
    assert_eq!(decoded.planes[2].data, v, "V plane mismatch");
}

/// Our ffvhuff encoder producing yuv411p decoded by ffmpeg.
#[test]
fn our_encoder_decoded_by_ffmpeg_yuv411p_left() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not on PATH; skipping");
        return;
    }
    cross_encode_then_ffmpeg_decode_v3("left", "yuv411p", PixelFormat::Yuv411P, 16, 16);
}

/// Our ffvhuff encoder producing gbrp10le decoded by ffmpeg.
#[test]
fn our_encoder_decoded_by_ffmpeg_gbrp10le_left() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not on PATH; skipping");
        return;
    }
    // The encoder synthesises bps=10 extradata for Gbrp10Le.
    // ffmpeg can decode the resulting stream as gbrp10le (2 bytes/sample).
    cross_encode_then_ffmpeg_decode_v3("left", "gbrp10le", PixelFormat::Gbrp10Le, W, H);
}

/// Encode a frame through OUR encoder and decode it back through
/// ffmpeg's `huffyuv` decoder. Asserts every plane sample matches a
/// rawvideo reference produced from the same lavfi input.
#[test]
fn our_encoder_decoded_by_ffmpeg_yuv422p_left() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not on PATH; skipping");
        return;
    }
    cross_encode_then_ffmpeg_decode("left", "yuv422p", PixelFormat::Yuv422P, W, H);
}

#[test]
fn our_encoder_decoded_by_ffmpeg_yuv422p_plane() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not on PATH; skipping");
        return;
    }
    cross_encode_then_ffmpeg_decode("plane", "yuv422p", PixelFormat::Yuv422P, W, H);
}

#[test]
fn our_encoder_decoded_by_ffmpeg_yuv420p_left() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not on PATH; skipping");
        return;
    }
    cross_encode_then_ffmpeg_decode("left", "yuv420p", PixelFormat::Yuv420P, W, H);
}

#[test]
fn our_encoder_decoded_by_ffmpeg_gray8_plane() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not on PATH; skipping");
        return;
    }
    cross_encode_then_ffmpeg_decode_v3("plane", "gray", PixelFormat::Gray8, W, H);
}

fn cross_encode_then_ffmpeg_decode(
    pred: &str,
    pix_fmt: &str,
    pf: PixelFormat,
    width: u32,
    height: u32,
) {
    cross_encode_then_ffmpeg_decode_inner(pred, pix_fmt, pf, width, height, "huffyuv");
}

fn cross_encode_then_ffmpeg_decode_v3(
    pred: &str,
    pix_fmt: &str,
    pf: PixelFormat,
    width: u32,
    height: u32,
) {
    cross_encode_then_ffmpeg_decode_inner(pred, pix_fmt, pf, width, height, "ffvhuff");
}

fn cross_encode_then_ffmpeg_decode_inner(
    pred: &str,
    pix_fmt: &str,
    pf: PixelFormat,
    width: u32,
    height: u32,
    codec: &str,
) {
    let raw = run_ffmpeg(&[
        "-y",
        "-f",
        "lavfi",
        "-i",
        &format!("testsrc=size={width}x{height}:rate=1:duration=0.04"),
        "-pix_fmt",
        pix_fmt,
        "-vframes",
        "1",
        "-f",
        "rawvideo",
        "-",
    ])
    .expect("ffmpeg rawvideo");

    // Build a VideoFrame for `raw` according to pf.
    let frame = build_frame_from_raw(&raw, width as usize, height as usize, pf);

    let mut reg = CodecRegistry::new();
    register(&mut reg);
    let mut enc_params = CodecParameters::video(CodecId::new(codec));
    enc_params.width = Some(width);
    enc_params.height = Some(height);
    enc_params.pixel_format = Some(pf);
    enc_params.options.insert("predictor", pred);
    let mut enc = reg.make_encoder(&enc_params).expect("our encoder");
    enc.send_frame(&oxideav_core::Frame::Video(frame.clone()))
        .expect("send_frame");
    let pkt = enc.receive_packet().expect("receive_packet");
    let extradata = enc.output_params().extradata.clone();

    // Wrap into a minimal AVI so ffmpeg can decode it. We emit a
    // RIFF/AVI file whose strf carries our extradata and whose movi
    // contains a single 00dc payload.
    let avi = build_minimal_avi(width, height, codec, &extradata, &pkt.data);

    // Pipe to ffmpeg → rawvideo and compare to `raw`.
    let mut child = std::process::Command::new("ffmpeg")
        .args([
            "-y", "-i", "pipe:0", "-f", "rawvideo", "-pix_fmt", pix_fmt, "pipe:1",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn ffmpeg decoder");
    use std::io::Write;
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(&avi)
        .expect("write avi to ffmpeg stdin");
    drop(child.stdin.take());
    let out = child.wait_with_output().expect("ffmpeg decode");
    if !out.status.success() {
        panic!(
            "ffmpeg decode failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    assert_eq!(
        out.stdout, raw,
        "ffmpeg-decoded our-encoded payload does not match the original raw input"
    );
}

fn build_frame_from_raw(
    raw: &[u8],
    width: usize,
    height: usize,
    pf: PixelFormat,
) -> oxideav_core::VideoFrame {
    use oxideav_core::frame::VideoPlane;
    use oxideav_core::VideoFrame;
    match pf {
        PixelFormat::Yuv422P => {
            let cw = width / 2;
            let y = raw[0..width * height].to_vec();
            let u = raw[width * height..width * height + cw * height].to_vec();
            let v = raw[width * height + cw * height..].to_vec();
            VideoFrame {
                pts: Some(0),
                planes: vec![
                    VideoPlane {
                        stride: width,
                        data: y,
                    },
                    VideoPlane {
                        stride: cw,
                        data: u,
                    },
                    VideoPlane {
                        stride: cw,
                        data: v,
                    },
                ],
            }
        }
        PixelFormat::Yuv420P => {
            let cw = width / 2;
            let ch = height / 2;
            let y = raw[0..width * height].to_vec();
            let u = raw[width * height..width * height + cw * ch].to_vec();
            let v = raw[width * height + cw * ch..].to_vec();
            VideoFrame {
                pts: Some(0),
                planes: vec![
                    VideoPlane {
                        stride: width,
                        data: y,
                    },
                    VideoPlane {
                        stride: cw,
                        data: u,
                    },
                    VideoPlane {
                        stride: cw,
                        data: v,
                    },
                ],
            }
        }
        PixelFormat::Yuv444P => {
            let n = width * height;
            VideoFrame {
                pts: Some(0),
                planes: vec![
                    VideoPlane {
                        stride: width,
                        data: raw[..n].to_vec(),
                    },
                    VideoPlane {
                        stride: width,
                        data: raw[n..2 * n].to_vec(),
                    },
                    VideoPlane {
                        stride: width,
                        data: raw[2 * n..3 * n].to_vec(),
                    },
                ],
            }
        }
        PixelFormat::Yuv411P => {
            let cw = width / 4;
            let yn = width * height;
            let cn = cw * height;
            let y = raw[0..yn].to_vec();
            let u = raw[yn..yn + cn].to_vec();
            let v = raw[yn + cn..yn + 2 * cn].to_vec();
            VideoFrame {
                pts: Some(0),
                planes: vec![
                    VideoPlane {
                        stride: width,
                        data: y,
                    },
                    VideoPlane {
                        stride: cw,
                        data: u,
                    },
                    VideoPlane {
                        stride: cw,
                        data: v,
                    },
                ],
            }
        }
        // gbrp10le from ffmpeg rawvideo is 2 bytes/sample (16-bit LE); pass through directly.
        PixelFormat::Gbrp10Le => {
            let n = width * height * 2; // 2 bytes per sample
            VideoFrame {
                pts: Some(0),
                planes: vec![
                    VideoPlane {
                        stride: width * 2,
                        data: raw[..n].to_vec(),
                    },
                    VideoPlane {
                        stride: width * 2,
                        data: raw[n..2 * n].to_vec(),
                    },
                    VideoPlane {
                        stride: width * 2,
                        data: raw[2 * n..3 * n].to_vec(),
                    },
                ],
            }
        }
        PixelFormat::Gray8 => VideoFrame {
            pts: Some(0),
            planes: vec![VideoPlane {
                stride: width,
                data: raw.to_vec(),
            }],
        },
        other => panic!("build_frame_from_raw: unsupported {other:?}"),
    }
}

fn build_minimal_avi(
    width: u32,
    height: u32,
    codec: &str,
    extradata: &[u8],
    pkt: &[u8],
) -> Vec<u8> {
    // Build the RIFF/AVI structure: RIFF AVI{ LIST hdrl{ avih, LIST strl{
    // strh, strf } }, LIST movi{ 00dc<pkt> } }. The ffmpeg AVI demuxer
    // accepts a fairly loose `avih` so we only fill the fields it
    // actually reads.
    let fcc = match codec {
        "huffyuv" => b"HFYU",
        "ffvhuff" => b"FFVH",
        _ => panic!("build_minimal_avi: unknown codec {codec}"),
    };
    fn pad_even(buf: &mut Vec<u8>) {
        if buf.len() & 1 == 1 {
            buf.push(0);
        }
    }
    fn chunk(out: &mut Vec<u8>, id: &[u8; 4], body: &[u8]) {
        out.extend_from_slice(id);
        out.extend_from_slice(&(body.len() as u32).to_le_bytes());
        out.extend_from_slice(body);
        pad_even(out);
    }
    fn list(id: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"LIST");
        let len = 4 + body.len();
        out.extend_from_slice(&(len as u32).to_le_bytes());
        out.extend_from_slice(id);
        out.extend_from_slice(body);
        out
    }
    let mut avih = Vec::new();
    avih.extend_from_slice(&1_000_000u32.to_le_bytes()); // dwMicroSecPerFrame
    avih.extend_from_slice(&0u32.to_le_bytes()); // dwMaxBytesPerSec
    avih.extend_from_slice(&0u32.to_le_bytes()); // dwPaddingGranularity
    avih.extend_from_slice(&0x10u32.to_le_bytes()); // dwFlags = AVIF_HASINDEX (lie, ffmpeg copes)
    avih.extend_from_slice(&1u32.to_le_bytes()); // dwTotalFrames
    avih.extend_from_slice(&0u32.to_le_bytes()); // dwInitialFrames
    avih.extend_from_slice(&1u32.to_le_bytes()); // dwStreams
    avih.extend_from_slice(&0u32.to_le_bytes()); // dwSuggestedBufferSize
    avih.extend_from_slice(&width.to_le_bytes());
    avih.extend_from_slice(&height.to_le_bytes());
    avih.extend_from_slice(&[0u8; 16]); // dwReserved[4]

    let mut strh = Vec::new();
    strh.extend_from_slice(b"vids");
    strh.extend_from_slice(fcc);
    strh.extend_from_slice(&0u32.to_le_bytes()); // dwFlags
    strh.extend_from_slice(&0u16.to_le_bytes()); // wPriority
    strh.extend_from_slice(&0u16.to_le_bytes()); // wLanguage
    strh.extend_from_slice(&0u32.to_le_bytes()); // dwInitialFrames
    strh.extend_from_slice(&1u32.to_le_bytes()); // dwScale
    strh.extend_from_slice(&1u32.to_le_bytes()); // dwRate
    strh.extend_from_slice(&0u32.to_le_bytes()); // dwStart
    strh.extend_from_slice(&1u32.to_le_bytes()); // dwLength
    strh.extend_from_slice(&0u32.to_le_bytes()); // dwSuggestedBufferSize
    strh.extend_from_slice(&0u32.to_le_bytes()); // dwQuality
    strh.extend_from_slice(&0u32.to_le_bytes()); // dwSampleSize
    strh.extend_from_slice(&[0u8; 16]); // rcFrame

    let bps_packed = 24u16; // not load-bearing for huffyuv
    let mut bih = Vec::new();
    bih.extend_from_slice(&40u32.to_le_bytes()); // biSize
    bih.extend_from_slice(&width.to_le_bytes()); // biWidth
    bih.extend_from_slice(&height.to_le_bytes()); // biHeight
    bih.extend_from_slice(&1u16.to_le_bytes()); // biPlanes
    bih.extend_from_slice(&bps_packed.to_le_bytes()); // biBitCount
    bih.extend_from_slice(fcc); // biCompression
    bih.extend_from_slice(&(width * height).to_le_bytes()); // biSizeImage
    bih.extend_from_slice(&0u32.to_le_bytes()); // biXPelsPerMeter
    bih.extend_from_slice(&0u32.to_le_bytes()); // biYPelsPerMeter
    bih.extend_from_slice(&0u32.to_le_bytes()); // biClrUsed
    bih.extend_from_slice(&0u32.to_le_bytes()); // biClrImportant
    let mut strf = bih;
    strf.extend_from_slice(extradata);

    let mut strl_body = Vec::new();
    chunk(&mut strl_body, b"strh", &strh);
    chunk(&mut strl_body, b"strf", &strf);
    let strl = list(b"strl", &strl_body);

    let mut hdrl_body = Vec::new();
    chunk(&mut hdrl_body, b"avih", &avih);
    hdrl_body.extend_from_slice(&strl);
    let hdrl = list(b"hdrl", &hdrl_body);

    let mut movi_body = Vec::new();
    chunk(&mut movi_body, b"00dc", pkt);
    let movi = list(b"movi", &movi_body);

    let mut riff_body = Vec::new();
    riff_body.extend_from_slice(b"AVI ");
    riff_body.extend_from_slice(&hdrl);
    riff_body.extend_from_slice(&movi);
    let mut out = Vec::new();
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(riff_body.len() as u32).to_le_bytes());
    out.extend_from_slice(&riff_body);
    out
}
