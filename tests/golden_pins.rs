//! Round-419 golden-output pins.
//!
//! Profile-round guard: every encode/decode optimisation this round
//! (and any future round) must keep the wire bytes and the decoded
//! rasters BYTE-IDENTICAL. This test pins an FNV-1a-64 hash of the
//! `strf` (BITMAPINFOHEADER + extradata) and the frame body for the
//! full legal matrix:
//!
//! - all three pixel families (YUY2 / RGB24 / RGB32),
//! - every method legal for the family (`predict_old`, Left,
//!   Gradient, Median, LeftDecorr, GradientDecorr),
//! - all three extradata modes (ClassicV2 / CustomV2 / V1xCompat),
//! - one progressive raster (48×32) and one interlaced raster
//!   (32×292; height > 288 engages the field-stride=2 path,
//!   spec/02 §2),
//!
//! plus the auto-selector's `(chosen method, wire bytes)` pins per
//! family/mode. Each pinned entry additionally round-trips through
//! [`decode_frame`] and asserts the decoded raster equals the source
//! pixels bit-exactly, so BOTH directions are locked: the encoder by
//! the wire hash, the decoder by lossless reconstruction of the
//! pinned wire bytes.
//!
//! The pinned values were generated from the pre-optimisation
//! (round-418) tree. To regenerate after an INTENTIONAL wire change
//! (there must never be one in a profile round), run:
//!
//! ```text
//! OXIDEAV_HUFFYUV_GOLDEN_DUMP=1 cargo test --test golden_pins -- --nocapture
//! ```
//!
//! and paste the emitted table over `EXPECTED` / `EXPECTED_AUTO`.

use oxideav_huffyuv::{
    decode_frame, encode_frame_auto, encode_frame_with_mode, ExtradataMode, Method,
    MethodSelection, PixelFamily, StreamConfig,
};

/// FNV-1a 64-bit — dependency-free, deterministic across platforms.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Deterministic xorshift32 pixel synth (same recipe as the bench
/// suite: smooth diagonal gradient + 3-bit noise so residual
/// histograms are realistic and every Huffman path is exercised).
fn build_pixels(width: usize, height: usize, bpp: usize) -> Vec<u8> {
    let row_bytes = width * bpp;
    let mut out = vec![0u8; row_bytes * height];
    let mut state: u32 = 0xdead_beef;
    for r in 0..height {
        for c in 0..width {
            let base = ((r as u32).wrapping_add(c as u32) >> 1) & 0xff;
            let off = r * row_bytes + c * bpp;
            for ch in 0..bpp {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                let noise = state & 0x07;
                let chan_bias = (ch as u32).wrapping_mul(7);
                out[off + ch] = (base.wrapping_add(noise).wrapping_add(chan_bias) & 0xff) as u8;
            }
        }
    }
    out
}

fn bpp(family: PixelFamily) -> usize {
    match family {
        PixelFamily::Yuy2 => 2,
        PixelFamily::Rgb24 => 3,
        PixelFamily::Rgb32 => 4,
    }
}

fn family_name(f: PixelFamily) -> &'static str {
    match f {
        PixelFamily::Yuy2 => "yuy2",
        PixelFamily::Rgb24 => "rgb24",
        PixelFamily::Rgb32 => "rgb32",
    }
}

fn method_name(m: Method) -> &'static str {
    match m {
        Method::PredictOld => "predict_old",
        Method::Left => "left",
        Method::Gradient => "gradient",
        Method::Median => "median",
        Method::LeftDecorr => "left_decorr",
        Method::GradientDecorr => "gradient_decorr",
    }
}

fn mode_name(m: ExtradataMode) -> &'static str {
    match m {
        ExtradataMode::ClassicV2 => "classic",
        ExtradataMode::CustomV2 => "custom",
        ExtradataMode::V1xCompat => "v1x",
    }
}

fn legal_methods(f: PixelFamily) -> &'static [Method] {
    match f {
        PixelFamily::Yuy2 => &[
            Method::PredictOld,
            Method::Left,
            Method::Gradient,
            Method::Median,
        ],
        PixelFamily::Rgb24 | PixelFamily::Rgb32 => &[
            Method::PredictOld,
            Method::Left,
            Method::LeftDecorr,
            Method::GradientDecorr,
        ],
    }
}

const FAMILIES: [PixelFamily; 3] = [PixelFamily::Yuy2, PixelFamily::Rgb24, PixelFamily::Rgb32];
const MODES: [ExtradataMode; 3] = [
    ExtradataMode::ClassicV2,
    ExtradataMode::CustomV2,
    ExtradataMode::V1xCompat,
];
/// (width, height): one progressive, one interlaced (h > 288).
const SIZES: [(u32, u32); 2] = [(48, 32), (32, 292)];

/// `(scenario, strf_fnv1a64, frame_fnv1a64)` — regenerate via the
/// dump env var documented in the module header.
const EXPECTED: &[(&str, u64, u64)] = &[
    (
        "yuy2/predict_old/classic/48x32",
        0xc0e9413d971fb2b9,
        0x76939f3aaa7680a6,
    ),
    (
        "yuy2/predict_old/classic/32x292",
        0x7e599ef643dbb4c4,
        0x0262fa1232a31ec5,
    ),
    (
        "yuy2/predict_old/custom/48x32",
        0x2ab00d6609c71b0b,
        0x30dce1437c8de7c9,
    ),
    (
        "yuy2/predict_old/custom/32x292",
        0xd7b621f7e80a1f8c,
        0xa417e79e55f1e281,
    ),
    (
        "yuy2/predict_old/v1x/48x32",
        0xd7e13ba71e7771e0,
        0xc9001dfc5f925f4d,
    ),
    (
        "yuy2/predict_old/v1x/32x292",
        0x783f1ba46e8635bf,
        0x6d57d60de175d513,
    ),
    (
        "yuy2/left/classic/48x32",
        0x16ca082d4dc4faf7,
        0x76939f3aaa7680a6,
    ),
    (
        "yuy2/left/classic/32x292",
        0x1a56cf7852dd4a86,
        0x0262fa1232a31ec5,
    ),
    (
        "yuy2/left/custom/48x32",
        0x3ecafec0ccfda1e9,
        0x30dce1437c8de7c9,
    ),
    (
        "yuy2/left/custom/32x292",
        0x8d666415a813ebce,
        0xa417e79e55f1e281,
    ),
    (
        "yuy2/left/v1x/48x32",
        0xeb4baf1470975ad1,
        0xc9001dfc5f925f4d,
    ),
    (
        "yuy2/left/v1x/32x292",
        0x60c70b5ee8628676,
        0x6d57d60de175d513,
    ),
    (
        "yuy2/gradient/classic/48x32",
        0xf61e075255bb195d,
        0x69fb34c026c2c028,
    ),
    (
        "yuy2/gradient/classic/32x292",
        0x2adac40da45cd86a,
        0x646f0edae87ab62e,
    ),
    (
        "yuy2/gradient/custom/48x32",
        0x7b0532e6e3abf89b,
        0x6c54ba8d4ab0639b,
    ),
    (
        "yuy2/gradient/custom/32x292",
        0x5f2ae94e889268bf,
        0xddfe402551def818,
    ),
    (
        "yuy2/gradient/v1x/48x32",
        0x8cb3988af69f8ce7,
        0x2bce1cf4a384bc3d,
    ),
    (
        "yuy2/gradient/v1x/32x292",
        0x36046a61ef4dd078,
        0x0bfb3bf6a50a9625,
    ),
    (
        "yuy2/median/classic/48x32",
        0x0e633134041d025a,
        0x4e2c16b09a62f83a,
    ),
    (
        "yuy2/median/classic/32x292",
        0x46682019f9d157b7,
        0x218e2a64fd217154,
    ),
    (
        "yuy2/median/custom/48x32",
        0x27c9b3a6ba46656a,
        0xf3074c0758460ac9,
    ),
    (
        "yuy2/median/custom/32x292",
        0x7998eafd87d839f9,
        0x91ae37da102044d6,
    ),
    (
        "yuy2/median/v1x/48x32",
        0x1772fd972713841c,
        0xcc89fe59cee83c34,
    ),
    (
        "yuy2/median/v1x/32x292",
        0x1b3a2270e5cc601b,
        0xcd2cac7f0dc92e62,
    ),
    (
        "rgb24/predict_old/classic/48x32",
        0xa3e75188ae6e6aeb,
        0x696187d49ee24431,
    ),
    (
        "rgb24/predict_old/classic/32x292",
        0xc6c4b590cfda518e,
        0xb529cce731cb6a55,
    ),
    (
        "rgb24/predict_old/custom/48x32",
        0xd77c109df76106f6,
        0xc5ea39cf18fd7802,
    ),
    (
        "rgb24/predict_old/custom/32x292",
        0xfa1930eec8a53624,
        0xc1f0c375be1fe701,
    ),
    (
        "rgb24/predict_old/v1x/48x32",
        0x632ea512a758ee18,
        0x49c1a59c1520781c,
    ),
    (
        "rgb24/predict_old/v1x/32x292",
        0x1f44c4175b661047,
        0x5c89a2b326ea17ea,
    ),
    (
        "rgb24/left/classic/48x32",
        0xbc1ce031140377f1,
        0x696187d49ee24431,
    ),
    (
        "rgb24/left/classic/32x292",
        0x57e0d3efc2844780,
        0xb529cce731cb6a55,
    ),
    (
        "rgb24/left/custom/48x32",
        0x3bb475d0858e4eec,
        0xc5ea39cf18fd7802,
    ),
    (
        "rgb24/left/custom/32x292",
        0x5f0960bd5637d742,
        0xc1f0c375be1fe701,
    ),
    (
        "rgb24/left/v1x/48x32",
        0xab22005c02bb9189,
        0x49c1a59c1520781c,
    ),
    (
        "rgb24/left/v1x/32x292",
        0xc3e5163af7ddf49e,
        0x5c89a2b326ea17ea,
    ),
    (
        "rgb24/left_decorr/classic/48x32",
        0x36f1c0a3a943fe1c,
        0x4ac3f8a578305731,
    ),
    (
        "rgb24/left_decorr/classic/32x292",
        0xdf4dbf65524efc47,
        0x82e9b408dd730b52,
    ),
    (
        "rgb24/left_decorr/custom/48x32",
        0x810c70635c043d5d,
        0x64a64d3e78fa75f4,
    ),
    (
        "rgb24/left_decorr/custom/32x292",
        0xb4cb6b2693c48776,
        0x0e0050896df91d3f,
    ),
    (
        "rgb24/left_decorr/v1x/48x32",
        0x7a6141f3cea6d296,
        0xc84b3e353164bd96,
    ),
    (
        "rgb24/left_decorr/v1x/32x292",
        0x0f33f781ea3ce831,
        0x229b697b27d6109d,
    ),
    (
        "rgb24/gradient_decorr/classic/48x32",
        0xfcd546f4a619af21,
        0x346e254eaf0772ae,
    ),
    (
        "rgb24/gradient_decorr/classic/32x292",
        0x6ced2bf9e3adf608,
        0xd70c3202bc7e8e07,
    ),
    (
        "rgb24/gradient_decorr/custom/48x32",
        0x0dce5b44b76d215c,
        0xc0a6aaf2c17253da,
    ),
    (
        "rgb24/gradient_decorr/custom/32x292",
        0xb412297cb3f680fa,
        0x1e5984dcadde21ed,
    ),
    (
        "rgb24/gradient_decorr/v1x/48x32",
        0x36c073363b705d5f,
        0xead6036c6213062c,
    ),
    (
        "rgb24/gradient_decorr/v1x/32x292",
        0xe684b189c5891840,
        0x16081ab4351fb3d3,
    ),
    (
        "rgb32/predict_old/classic/48x32",
        0xc8020193f954b18b,
        0xec7a418df20ed579,
    ),
    (
        "rgb32/predict_old/classic/32x292",
        0xccba0ec5f6d5ebee,
        0x685d73984be975a8,
    ),
    (
        "rgb32/predict_old/custom/48x32",
        0x8104969485488b6a,
        0xd99bc1047fc50518,
    ),
    (
        "rgb32/predict_old/custom/32x292",
        0x41dc954e6a2901b3,
        0x9749a952c0156064,
    ),
    (
        "rgb32/predict_old/v1x/48x32",
        0x3bffce0a73d611b0,
        0xf63e752b3ea10824,
    ),
    (
        "rgb32/predict_old/v1x/32x292",
        0x3307ae91c6b2c8cf,
        0xe01ef6486d2d225c,
    ),
    (
        "rgb32/left/classic/48x32",
        0x37a94afa16304951,
        0xec7a418df20ed579,
    ),
    (
        "rgb32/left/classic/32x292",
        0xc055fa47b8ac9360,
        0x685d73984be975a8,
    ),
    (
        "rgb32/left/custom/48x32",
        0x0866516b559ae6f0,
        0xd99bc1047fc50518,
    ),
    (
        "rgb32/left/custom/32x292",
        0x6fbbf3dfe34a2b41,
        0x9749a952c0156064,
    ),
    (
        "rgb32/left/v1x/48x32",
        0x8aa810c54d446be1,
        0xf63e752b3ea10824,
    ),
    (
        "rgb32/left/v1x/32x292",
        0xbd6b11885eea4546,
        0xe01ef6486d2d225c,
    ),
    (
        "rgb32/left_decorr/classic/48x32",
        0x714fe4370f5416dc,
        0x44be5197d4b917ef,
    ),
    (
        "rgb32/left_decorr/classic/32x292",
        0x17c2f84a34592657,
        0x65ee3717bd8a60c2,
    ),
    (
        "rgb32/left_decorr/custom/48x32",
        0x79038e6a476a9bb9,
        0x94f682306b7d2438,
    ),
    (
        "rgb32/left_decorr/custom/32x292",
        0x5f324e79a5f98f81,
        0x33d2b8aef9406af9,
    ),
    (
        "rgb32/left_decorr/v1x/48x32",
        0xf9c0341749d88fce,
        0xfb45c197a9bb8f1a,
    ),
    (
        "rgb32/left_decorr/v1x/32x292",
        0xdb247414c8f02379,
        0x73985537761b4318,
    ),
    (
        "rgb32/gradient_decorr/classic/48x32",
        0x4c7602c1f0cf31c1,
        0x9fea1fc733591fa0,
    ),
    (
        "rgb32/gradient_decorr/classic/32x292",
        0xd7c7049230358008,
        0x9f9899424969febd,
    ),
    (
        "rgb32/gradient_decorr/custom/48x32",
        0x9953c138da89aad6,
        0x140b5e46debdf53f,
    ),
    (
        "rgb32/gradient_decorr/custom/32x292",
        0x01c36755a65234ca,
        0x3538774bca904cc1,
    ),
    (
        "rgb32/gradient_decorr/v1x/48x32",
        0xad76a63643764a77,
        0x729810c518c2218b,
    ),
    (
        "rgb32/gradient_decorr/v1x/32x292",
        0xa0e4130309bdd748,
        0x1cff301fdb8abc8c,
    ),
];

/// `(scenario, chosen_method, strf_fnv1a64, frame_fnv1a64)`.
const EXPECTED_AUTO: &[(&str, &str, u64, u64)] = &[
    (
        "yuy2/auto/classic/48x32",
        "median",
        0x0e633134041d025a,
        0x4e2c16b09a62f83a,
    ),
    (
        "yuy2/auto/custom/48x32",
        "median",
        0x27c9b3a6ba46656a,
        0xf3074c0758460ac9,
    ),
    (
        "rgb24/auto/classic/48x32",
        "left",
        0xbc1ce031140377f1,
        0x696187d49ee24431,
    ),
    (
        "rgb24/auto/custom/48x32",
        "left",
        0x3bb475d0858e4eec,
        0xc5ea39cf18fd7802,
    ),
    (
        "rgb32/auto/classic/48x32",
        "left",
        0x37a94afa16304951,
        0xec7a418df20ed579,
    ),
    (
        "rgb32/auto/custom/48x32",
        "left",
        0x0866516b559ae6f0,
        0xd99bc1047fc50518,
    ),
];

fn dump_mode() -> bool {
    std::env::var_os("OXIDEAV_HUFFYUV_GOLDEN_DUMP").is_some()
}

#[test]
fn golden_pins_full_matrix() {
    let dump = dump_mode();
    let mut seen = Vec::new();
    for family in FAMILIES {
        for &method in legal_methods(family) {
            for mode in MODES {
                for (w, h) in SIZES {
                    let scenario = format!(
                        "{}/{}/{}/{}x{}",
                        family_name(family),
                        method_name(method),
                        mode_name(mode),
                        w,
                        h
                    );
                    let pixels = build_pixels(w as usize, h as usize, bpp(family));
                    let (strf, frame) = encode_frame_with_mode(family, method, w, h, &pixels, mode)
                        .unwrap_or_else(|e| panic!("{scenario}: encode failed: {e:?}"));
                    let strf_hash = fnv1a64(&strf);
                    let frame_hash = fnv1a64(&frame);
                    // Decode-side lock: the pinned wire bytes must
                    // reconstruct the source raster bit-exactly.
                    let cfg = StreamConfig::parse_bitmapinfoheader(&strf)
                        .unwrap_or_else(|e| panic!("{scenario}: parse failed: {e:?}"));
                    let decoded = decode_frame(&cfg, &frame)
                        .unwrap_or_else(|e| panic!("{scenario}: decode failed: {e:?}"));
                    assert_eq!(
                        decoded.pixels, pixels,
                        "{scenario}: decode must losslessly invert the pinned wire bytes"
                    );
                    if dump {
                        println!("    (\"{scenario}\", 0x{strf_hash:016x}, 0x{frame_hash:016x}),");
                    }
                    seen.push((scenario, strf_hash, frame_hash));
                }
            }
        }
    }
    if dump {
        return;
    }
    assert_eq!(
        seen.len(),
        EXPECTED.len(),
        "pin matrix drifted: {} scenarios produced, {} pinned",
        seen.len(),
        EXPECTED.len()
    );
    for ((scenario, strf_hash, frame_hash), (exp_scenario, exp_strf, exp_frame)) in
        seen.iter().zip(EXPECTED.iter())
    {
        assert_eq!(scenario, exp_scenario, "scenario order drifted");
        assert_eq!(
            *strf_hash, *exp_strf,
            "{scenario}: strf bytes changed (golden pin violation)"
        );
        assert_eq!(
            *frame_hash, *exp_frame,
            "{scenario}: frame wire bytes changed (golden pin violation)"
        );
    }
}

#[test]
fn golden_pins_auto_selector() {
    let dump = dump_mode();
    let mut seen = Vec::new();
    for family in FAMILIES {
        for mode in [ExtradataMode::ClassicV2, ExtradataMode::CustomV2] {
            let (w, h) = (48u32, 32u32);
            let scenario = format!(
                "{}/auto/{}/{}x{}",
                family_name(family),
                mode_name(mode),
                w,
                h
            );
            let pixels = build_pixels(w as usize, h as usize, bpp(family));
            let (strf, frame, chosen) =
                encode_frame_auto(family, MethodSelection::Auto, w, h, &pixels, mode)
                    .unwrap_or_else(|e| panic!("{scenario}: encode failed: {e:?}"));
            let strf_hash = fnv1a64(&strf);
            let frame_hash = fnv1a64(&frame);
            let cfg = StreamConfig::parse_bitmapinfoheader(&strf)
                .unwrap_or_else(|e| panic!("{scenario}: parse failed: {e:?}"));
            let decoded = decode_frame(&cfg, &frame)
                .unwrap_or_else(|e| panic!("{scenario}: decode failed: {e:?}"));
            assert_eq!(
                decoded.pixels, pixels,
                "{scenario}: decode must losslessly invert the pinned wire bytes"
            );
            if dump {
                println!(
                    "    (\"{scenario}\", \"{}\", 0x{strf_hash:016x}, 0x{frame_hash:016x}),",
                    method_name(chosen)
                );
            }
            seen.push((scenario, method_name(chosen), strf_hash, frame_hash));
        }
    }
    if dump {
        return;
    }
    assert_eq!(seen.len(), EXPECTED_AUTO.len(), "auto pin matrix drifted");
    for (
        (scenario, chosen, strf_hash, frame_hash),
        (exp_scenario, exp_chosen, exp_strf, exp_frame),
    ) in seen.iter().zip(EXPECTED_AUTO.iter())
    {
        assert_eq!(scenario, exp_scenario, "scenario order drifted");
        assert_eq!(
            chosen, exp_chosen,
            "{scenario}: auto-selector chose a different method (golden pin violation)"
        );
        assert_eq!(*strf_hash, *exp_strf, "{scenario}: strf bytes changed");
        assert_eq!(
            *frame_hash, *exp_frame,
            "{scenario}: frame wire bytes changed"
        );
    }
}
