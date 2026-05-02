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
//! This test is `#[ignore]`d so it stays opt-in; it requires `ffmpeg`
//! and currently the upstream encoder default predictor (LEFT). The
//! pure-Rust `synth_left` test exercises the same code path without
//! external tooling.

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

#[test]
#[ignore]
fn ffmpeg_huffyuv_round_trip_yuv422p_left() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not on PATH; skipping");
        return;
    }
    // 1. Encode a 1-frame testsrc into HuffYUV/AVI.
    let avi = run_ffmpeg(&[
        "-y",
        "-f",
        "lavfi",
        "-i",
        &format!("testsrc=size={}x{}:rate=1:duration=0.04", W, H),
        "-c:v",
        "huffyuv",
        "-pred",
        "left",
        "-pix_fmt",
        "yuv422p",
        "-f",
        "avi",
        "-",
    ])
    .expect("ffmpeg encode");
    // 2. Capture the reference rawvideo (yuv422p) for the same source.
    let raw = run_ffmpeg(&[
        "-y",
        "-f",
        "lavfi",
        "-i",
        &format!("testsrc=size={}x{}:rate=1:duration=0.04", W, H),
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

    // 3. Decode through our crate.
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

    // 4. Compare planes byte-for-byte against the reference.
    let chroma_w = (W as usize) / 2;
    let mut ref_off = 0usize;
    for y in 0..(H as usize) {
        let row = &video.planes[0].data[y * video.planes[0].stride..][..(W as usize)];
        let r = &raw[ref_off..ref_off + row.len()];
        assert_eq!(row, r, "Y row {y} mismatch");
        ref_off += row.len();
    }
    for y in 0..(H as usize) {
        let row = &video.planes[1].data[y * video.planes[1].stride..][..chroma_w];
        let r = &raw[ref_off..ref_off + row.len()];
        assert_eq!(row, r, "U row {y} mismatch");
        ref_off += row.len();
    }
    for y in 0..(H as usize) {
        let row = &video.planes[2].data[y * video.planes[2].stride..][..chroma_w];
        let r = &raw[ref_off..ref_off + row.len()];
        assert_eq!(row, r, "V row {y} mismatch");
        ref_off += row.len();
    }
}
