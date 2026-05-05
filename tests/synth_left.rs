//! Synthetic round-trip: hand-build a v2 yuv422p HuffYUV packet (LEFT
//! predictor) for a known 4x2 image and assert the decoder reproduces
//! the original samples bit-exactly.

#![allow(clippy::needless_range_loop)]
#![allow(clippy::vec_init_then_push)]
#![allow(clippy::identity_op)]
//!
//! The test exercises the extradata + RLE + canonical-Huffman + LEFT
//! predictor + 32-bit byte-swap path end-to-end without any external
//! data; everything is built in pure Rust.

use oxideav_core::time::TimeBase;
use oxideav_core::{CodecId, CodecParameters, CodecRegistry, Packet, PixelFormat};
use oxideav_huffyuv::register;

const W: usize = 4;
const H: usize = 2;

/// Image samples (Y plane is 4x2, U/V are 2x2):
///   Y[0]: 16, 32, 48, 64
///   Y[1]: 80, 96, 112, 128
///   U[0]: 0x80, 0x90      U[1]: 0xA0, 0xB0
///   V[0]: 0x10, 0x20      V[1]: 0x30, 0x40
const Y: [[u8; W]; H] = [[16, 32, 48, 64], [80, 96, 112, 128]];
const U: [[u8; W / 2]; H] = [[0x80, 0x90], [0xA0, 0xB0]];
const V: [[u8; W / 2]; H] = [[0x10, 0x20], [0x30, 0x40]];

/// Build a 256-entry equal-length Huffman table (every byte gets a
/// fixed 8-bit code) and the corresponding RLE blob. Codes are exactly
/// the symbol value itself in MSB-first order — so writing a sample
/// straight as 8 bits is its own canonical Huffman code.
fn equal_8bit_lengths_blob() -> (Vec<u8>, Vec<u8>) {
    let lens = vec![8u8; 256];
    // RLE: emit (val=8, run=0)=0x08 then ext=0xFF for the first 255
    // symbols (run=255), then a short (val=8, run=1)=0x28 for symbol 255.
    // Easier: emit two long runs of 128 each.
    let mut blob = Vec::new();
    blob.push(8u8); // val=8, run=0
    blob.push(128u8); // ext=128
    blob.push(8u8); // val=8, run=0
    blob.push(128u8); // ext=128
    (lens, blob)
}

/// Build the v2 yuv422p extradata: 4-byte header + three identical
/// 256-symbol equal-length tables.
fn build_extradata() -> Vec<u8> {
    let (_lens, blob) = equal_8bit_lengths_blob();
    let mut out = vec![0x00u8, 0x10, 0x20, 0x00];
    for _ in 0..3 {
        out.extend_from_slice(&blob);
    }
    out
}

/// Encode the synthetic frame into a packet payload by computing the
/// LEFT-predictor residuals and writing them as 8-bit raw bytes
/// (because canonical-Huffman with all 8-bit lengths makes each code
/// equal to its symbol). Then byte-swap each 32-bit word the same way
/// the encoder would so the decoder can undo it.
fn build_frame_payload() -> Vec<u8> {
    let mut residuals: Vec<u8> = Vec::new();

    // Prelude V0 Y1 U0 Y0
    residuals.push(V[0][0]);
    residuals.push(Y[0][1]);
    residuals.push(U[0][0]);
    residuals.push(Y[0][0]);

    // Row 0 — start at chroma column 1 (the prelude consumed col 0).
    // Inner loop: Y, U, Y, V per chroma column. LEFT residual = sample
    // - prev_sample (per channel).
    let mut left_y = Y[0][1];
    let mut left_u = U[0][0];
    let mut left_v = V[0][0];
    let chroma_w = W / 2;
    for cx in 1..chroma_w {
        let res_y0 = Y[0][2 * cx].wrapping_sub(left_y);
        residuals.push(res_y0);
        left_y = Y[0][2 * cx];
        let res_u = U[0][cx].wrapping_sub(left_u);
        residuals.push(res_u);
        left_u = U[0][cx];
        let res_y1 = Y[0][2 * cx + 1].wrapping_sub(left_y);
        residuals.push(res_y1);
        left_y = Y[0][2 * cx + 1];
        let res_v = V[0][cx].wrapping_sub(left_v);
        residuals.push(res_v);
        left_v = V[0][cx];
    }
    // Row 1+ — full row interleave.
    for y in 1..H {
        for cx in 0..chroma_w {
            let res_y0 = Y[y][2 * cx].wrapping_sub(left_y);
            residuals.push(res_y0);
            left_y = Y[y][2 * cx];
            let res_u = U[y][cx].wrapping_sub(left_u);
            residuals.push(res_u);
            left_u = U[y][cx];
            let res_y1 = Y[y][2 * cx + 1].wrapping_sub(left_y);
            residuals.push(res_y1);
            left_y = Y[y][2 * cx + 1];
            let res_v = V[y][cx].wrapping_sub(left_v);
            residuals.push(res_v);
            left_v = V[y][cx];
        }
    }
    // 31-bit zero tail + align to 4 bytes (trace doc §5.6). Each
    // residual byte is exactly 8 bits (= one 8-bit Huffman code), so
    // the residual stream is already byte-aligned. We just need to
    // pad the whole packet to a multiple of 4.
    while residuals.len() % 4 != 0 {
        residuals.push(0);
    }
    // Apply the encoder's per-32-bit-word byte-swap (the decoder
    // unswaps with `unswap_payload`).
    let mut swapped = Vec::with_capacity(residuals.len());
    for w in 0..(residuals.len() / 4) {
        let off = w * 4;
        swapped.push(residuals[off + 3]);
        swapped.push(residuals[off + 2]);
        swapped.push(residuals[off + 1]);
        swapped.push(residuals[off]);
    }
    swapped
}

#[test]
fn decode_v2_yuv422_left_round_trip() {
    let extradata = build_extradata();
    let payload = build_frame_payload();

    let mut reg = CodecRegistry::new();
    register(&mut reg);

    let mut params = CodecParameters::video(CodecId::new("huffyuv"));
    params.width = Some(W as u32);
    params.height = Some(H as u32);
    params.pixel_format = Some(PixelFormat::Yuv422P);
    params.extradata = extradata;

    let mut decoder = reg.first_decoder(&params).expect("decoder factory");
    let pkt = Packet::new(0, TimeBase::new(1, 25), payload).with_keyframe(true);
    decoder.send_packet(&pkt).expect("send_packet");
    let frame = decoder.receive_frame().expect("receive_frame");
    let video = match frame {
        oxideav_core::Frame::Video(v) => v,
        _ => panic!("expected video frame"),
    };
    assert_eq!(video.planes.len(), 3);
    let (y_plane, u_plane, v_plane) = (&video.planes[0], &video.planes[1], &video.planes[2]);
    let chroma_w = W / 2;

    for y in 0..H {
        for x in 0..W {
            assert_eq!(
                y_plane.data[y * y_plane.stride + x],
                Y[y][x],
                "Y[{y}][{x}] mismatch"
            );
        }
        for cx in 0..chroma_w {
            assert_eq!(
                u_plane.data[y * u_plane.stride + cx],
                U[y][cx],
                "U[{y}][{cx}] mismatch"
            );
            assert_eq!(
                v_plane.data[y * v_plane.stride + cx],
                V[y][cx],
                "V[{y}][{cx}] mismatch"
            );
        }
    }
}
