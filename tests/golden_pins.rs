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
#![cfg(feature = "registry")]

use oxideav_huffyuv::{
    ExtradataMode, Method, MethodSelection, PixelFamily, StreamConfig, decode_frame,
    decode_frame_with_workers, encode_frame_auto, encode_frame_auto_workers,
    encode_frame_with_mode, encode_frame_with_mode_workers, make_encoder,
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
        _ => unreachable!("new ExtradataMode variant"),
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
        0x409b5dff73d3ed83,
    ),
    (
        "yuy2/predict_old/custom/48x32",
        0x2ab00d6609c71b0b,
        0x30dce1437c8de7c9,
    ),
    (
        "yuy2/predict_old/custom/32x292",
        0x138b878e598e61ab,
        0x0b16f49ba13e192f,
    ),
    (
        "yuy2/predict_old/v1x/48x32",
        0xd7e13ba71e7771e0,
        0xc9001dfc5f925f4d,
    ),
    (
        "yuy2/predict_old/v1x/32x292",
        0x783f1ba46e8635bf,
        0x8dd7b260f3fe9406,
    ),
    (
        "yuy2/left/classic/48x32",
        0x16ca082d4dc4faf7,
        0x76939f3aaa7680a6,
    ),
    (
        "yuy2/left/classic/32x292",
        0x1a56cf7852dd4a86,
        0x409b5dff73d3ed83,
    ),
    (
        "yuy2/left/custom/48x32",
        0x3ecafec0ccfda1e9,
        0x30dce1437c8de7c9,
    ),
    (
        "yuy2/left/custom/32x292",
        0xaba72375cf0383f9,
        0x0b16f49ba13e192f,
    ),
    (
        "yuy2/left/v1x/48x32",
        0xeb4baf1470975ad1,
        0xc9001dfc5f925f4d,
    ),
    (
        "yuy2/left/v1x/32x292",
        0x60c70b5ee8628676,
        0x8dd7b260f3fe9406,
    ),
    (
        "yuy2/gradient/classic/48x32",
        0xf61e075255bb195d,
        0x69fb34c026c2c028,
    ),
    (
        "yuy2/gradient/classic/32x292",
        0x2adac40da45cd86a,
        0x70d87416799b1c18,
    ),
    (
        "yuy2/gradient/custom/48x32",
        0x7b0532e6e3abf89b,
        0x6c54ba8d4ab0639b,
    ),
    (
        "yuy2/gradient/custom/32x292",
        0x46418357617958ee,
        0x004f85fabd1571a2,
    ),
    (
        "yuy2/gradient/v1x/48x32",
        0x8cb3988af69f8ce7,
        0x2bce1cf4a384bc3d,
    ),
    (
        "yuy2/gradient/v1x/32x292",
        0x36046a61ef4dd078,
        0xbce72ac0452324fa,
    ),
    (
        "yuy2/median/classic/48x32",
        0x0e633134041d025a,
        0x98d319e2ceff9abd,
    ),
    (
        "yuy2/median/classic/32x292",
        0x46682019f9d157b7,
        0xdf20d976e4b8743e,
    ),
    (
        "yuy2/median/custom/48x32",
        0xa85009cccd28dc27,
        0x405991329d7400bc,
    ),
    (
        "yuy2/median/custom/32x292",
        0x91963ae7f33c43c0,
        0xcb37f29e9492147f,
    ),
    (
        "yuy2/median/v1x/48x32",
        0x1772fd972713841c,
        0xb2dfdeafa8b7d919,
    ),
    (
        "yuy2/median/v1x/32x292",
        0x1b3a2270e5cc601b,
        0xa4a7724730c0596d,
    ),
    (
        "rgb24/predict_old/classic/48x32",
        0xa3e75188ae6e6aeb,
        0x696187d49ee24431,
    ),
    (
        "rgb24/predict_old/classic/32x292",
        0xc6c4b590cfda518e,
        0x16d4dd3e1074513f,
    ),
    (
        "rgb24/predict_old/custom/48x32",
        0xd77c109df76106f6,
        0xc5ea39cf18fd7802,
    ),
    (
        "rgb24/predict_old/custom/32x292",
        0x76ecee20d6eb8122,
        0x8248870b6714b408,
    ),
    (
        "rgb24/predict_old/v1x/48x32",
        0x632ea512a758ee18,
        0x49c1a59c1520781c,
    ),
    (
        "rgb24/predict_old/v1x/32x292",
        0x1f44c4175b661047,
        0x364df1ee2e909843,
    ),
    (
        "rgb24/left/classic/48x32",
        0xbc1ce031140377f1,
        0x696187d49ee24431,
    ),
    (
        "rgb24/left/classic/32x292",
        0x57e0d3efc2844780,
        0x16d4dd3e1074513f,
    ),
    (
        "rgb24/left/custom/48x32",
        0x3bb475d0858e4eec,
        0xc5ea39cf18fd7802,
    ),
    (
        "rgb24/left/custom/32x292",
        0x1b978bae31fcabc4,
        0x8248870b6714b408,
    ),
    (
        "rgb24/left/v1x/48x32",
        0xab22005c02bb9189,
        0x49c1a59c1520781c,
    ),
    (
        "rgb24/left/v1x/32x292",
        0xc3e5163af7ddf49e,
        0x364df1ee2e909843,
    ),
    (
        "rgb24/left_decorr/classic/48x32",
        0x36f1c0a3a943fe1c,
        0xcf34f698aa41a96f,
    ),
    (
        "rgb24/left_decorr/classic/32x292",
        0xdf4dbf65524efc47,
        0xa62e63bdfa36c34f,
    ),
    (
        "rgb24/left_decorr/custom/48x32",
        0x810c70635c043d5d,
        0x0cc59da657fd9062,
    ),
    (
        "rgb24/left_decorr/custom/32x292",
        0x798bffbd57c9e106,
        0x1a1c18972f170ed4,
    ),
    (
        "rgb24/left_decorr/v1x/48x32",
        0x7a6141f3cea6d296,
        0x1a329ce10a3e7e68,
    ),
    (
        "rgb24/left_decorr/v1x/32x292",
        0x0f33f781ea3ce831,
        0x46f6890a961a62a1,
    ),
    (
        "rgb24/gradient_decorr/classic/48x32",
        0xfcd546f4a619af21,
        0x85599c0d1461044c,
    ),
    (
        "rgb24/gradient_decorr/classic/32x292",
        0x6ced2bf9e3adf608,
        0xfe7f48c725707b9d,
    ),
    (
        "rgb24/gradient_decorr/custom/48x32",
        0x0dce5b44b76d215c,
        0x21271348042d9bdc,
    ),
    (
        "rgb24/gradient_decorr/custom/32x292",
        0x73c2c47be1e99c2a,
        0x4fa9bd75f2556d03,
    ),
    (
        "rgb24/gradient_decorr/v1x/48x32",
        0x36c073363b705d5f,
        0x68660bac6adc145a,
    ),
    (
        "rgb24/gradient_decorr/v1x/32x292",
        0xe684b189c5891840,
        0xc18e5a3501dcb6b6,
    ),
    (
        "rgb32/predict_old/classic/48x32",
        0xc8020193f954b18b,
        0xec7a418df20ed579,
    ),
    (
        "rgb32/predict_old/classic/32x292",
        0xccba0ec5f6d5ebee,
        0x2ad755703c5fa72f,
    ),
    (
        "rgb32/predict_old/custom/48x32",
        0x8104969485488b6a,
        0xd99bc1047fc50518,
    ),
    (
        "rgb32/predict_old/custom/32x292",
        0xc9bc4e5592d390ca,
        0xe35e7fb3195fbec6,
    ),
    (
        "rgb32/predict_old/v1x/48x32",
        0x3bffce0a73d611b0,
        0xf63e752b3ea10824,
    ),
    (
        "rgb32/predict_old/v1x/32x292",
        0x3307ae91c6b2c8cf,
        0x545a95c9fdfbee67,
    ),
    (
        "rgb32/left/classic/48x32",
        0x37a94afa16304951,
        0xec7a418df20ed579,
    ),
    (
        "rgb32/left/classic/32x292",
        0xc055fa47b8ac9360,
        0x2ad755703c5fa72f,
    ),
    (
        "rgb32/left/custom/48x32",
        0x0866516b559ae6f0,
        0xd99bc1047fc50518,
    ),
    (
        "rgb32/left/custom/32x292",
        0x3029fc3d4bdebcb8,
        0xe35e7fb3195fbec6,
    ),
    (
        "rgb32/left/v1x/48x32",
        0x8aa810c54d446be1,
        0xf63e752b3ea10824,
    ),
    (
        "rgb32/left/v1x/32x292",
        0xbd6b11885eea4546,
        0x545a95c9fdfbee67,
    ),
    (
        "rgb32/left_decorr/classic/48x32",
        0x714fe4370f5416dc,
        0x5887d4e0fce5ee39,
    ),
    (
        "rgb32/left_decorr/classic/32x292",
        0x17c2f84a34592657,
        0x0a7b1409f88f7eb6,
    ),
    (
        "rgb32/left_decorr/custom/48x32",
        0x79038e6a476a9bb9,
        0x9e45ba37e16609ee,
    ),
    (
        "rgb32/left_decorr/custom/32x292",
        0xbbe3a95a78dea788,
        0xe0a5872c15404846,
    ),
    (
        "rgb32/left_decorr/v1x/48x32",
        0xf9c0341749d88fce,
        0x8a5f4679e30261ec,
    ),
    (
        "rgb32/left_decorr/v1x/32x292",
        0xdb247414c8f02379,
        0xc20474a3c6b7ebe8,
    ),
    (
        "rgb32/gradient_decorr/classic/48x32",
        0x4c7602c1f0cf31c1,
        0x5d541b476385bbc6,
    ),
    (
        "rgb32/gradient_decorr/classic/32x292",
        0xd7c7049230358008,
        0x838e678242a8a2aa,
    ),
    (
        "rgb32/gradient_decorr/custom/48x32",
        0x9953c138da89aad6,
        0x56d911c7d5cbdd4d,
    ),
    (
        "rgb32/gradient_decorr/custom/32x292",
        0x0a48bb07423a4e72,
        0x2fde746ba277b1a5,
    ),
    (
        "rgb32/gradient_decorr/v1x/48x32",
        0xad76a63643764a77,
        0x3383a4fa16cf8171,
    ),
    (
        "rgb32/gradient_decorr/v1x/32x292",
        0xa0e4130309bdd748,
        0x103415d77b14a29f,
    ),
];

/// `(scenario, chosen_method, strf_fnv1a64, frame_fnv1a64)`.
const EXPECTED_AUTO: &[(&str, &str, u64, u64)] = &[
    (
        "yuy2/auto/classic/48x32",
        "median",
        0x0e633134041d025a,
        0x98d319e2ceff9abd,
    ),
    (
        "yuy2/auto/custom/48x32",
        "median",
        0xa85009cccd28dc27,
        0x405991329d7400bc,
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

/// Round-420 worker-budget invariance over the golden-pin matrix.
///
/// The two-worker interlaced decode path (`decode_frame_with_workers`
/// with budget ≥ 2) must reproduce the serial decode byte-for-byte on
/// EVERY interlaced golden pin — same wire bytes in (the pinned
/// encoder output; wire hashes are asserted by
/// `golden_pins_full_matrix`), same raster out. Progressive pins are
/// exercised too: for them any budget must be a strict no-op (single
/// work unit). Together with the r419 pins this locks the round-420
/// threading work at "zero wire change, zero output change".
#[test]
fn golden_pins_budget2_decode_invariance() {
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
                    let cfg = StreamConfig::parse_bitmapinfoheader(&strf)
                        .unwrap_or_else(|e| panic!("{scenario}: parse failed: {e:?}"));
                    let serial = decode_frame(&cfg, &frame)
                        .unwrap_or_else(|e| panic!("{scenario}: serial decode failed: {e:?}"));
                    let budget2 = decode_frame_with_workers(&cfg, &frame, 2)
                        .unwrap_or_else(|e| panic!("{scenario}: budget-2 decode failed: {e:?}"));
                    assert_eq!(
                        serial.pixels, budget2.pixels,
                        "{scenario}: budget-2 decode diverged from serial"
                    );
                    assert_eq!(
                        budget2.pixels, pixels,
                        "{scenario}: budget-2 decode is not lossless"
                    );
                }
            }
        }
    }
}

/// Round-420 worker-budget invariance for the ENCODE side over the
/// golden-pin matrix: `encode_frame_with_mode_workers(..., 2)` must
/// emit the exact pinned wire bytes (strf + frame hashes) on every
/// scenario — interlaced pins exercise the two-worker residual + emit
/// phases, progressive pins pin the budget as a strict no-op.
#[test]
fn golden_pins_budget2_encode_invariance() {
    for family in FAMILIES {
        for &method in legal_methods(family) {
            for mode in MODES {
                for (w, h) in SIZES {
                    let scenario = format!(
                        "{}/{}/{}/{}x{} (budget 2)",
                        family_name(family),
                        method_name(method),
                        mode_name(mode),
                        w,
                        h
                    );
                    let pixels = build_pixels(w as usize, h as usize, bpp(family));
                    let (strf, frame) =
                        encode_frame_with_mode_workers(family, method, w, h, &pixels, mode, 2)
                            .unwrap_or_else(|e| panic!("{scenario}: encode failed: {e:?}"));
                    let (strf_serial, frame_serial) =
                        encode_frame_with_mode(family, method, w, h, &pixels, mode)
                            .unwrap_or_else(|e| panic!("{scenario}: serial encode failed: {e:?}"));
                    assert_eq!(strf, strf_serial, "{scenario}: strf diverged from serial");
                    assert_eq!(
                        frame, frame_serial,
                        "{scenario}: wire bytes diverged from serial"
                    );
                }
            }
        }
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

// ───────────────── round-429 trait-path + auto-budget pins ─────────────────

/// Option-bag value for a [`Method`] (the registry encoder's `method`
/// option uses kebab-case where the pin scenarios use snake_case).
fn method_option(m: Method) -> &'static str {
    match m {
        Method::PredictOld => "predict-old",
        Method::Left => "left",
        Method::Gradient => "gradient",
        Method::Median => "median",
        Method::LeftDecorr => "left-decorr",
        Method::GradientDecorr => "gradient-decorr",
    }
}

fn mode_option(m: ExtradataMode) -> &'static str {
    match m {
        ExtradataMode::ClassicV2 => "classic-v2",
        ExtradataMode::CustomV2 => "custom-v2",
        ExtradataMode::V1xCompat => "v1x",
        _ => unreachable!("new ExtradataMode variant"),
    }
}

fn family_option(f: PixelFamily) -> &'static str {
    match f {
        PixelFamily::Yuy2 => "yuy2",
        PixelFamily::Rgb24 => "rgb24",
        PixelFamily::Rgb32 => "rgb32",
    }
}

/// Build a registry encoder for one pin scenario and push one frame
/// through it, returning `(strf, wire bytes)` from the trait surface
/// (`output_params().extradata` + the emitted packet).
///
/// `pixels` is the wire raster the direct API takes; a framework frame
/// carries the top row first, so the RGB families (stored bottom row
/// first, spec/02 §2) are handed over with their rows reversed.
fn trait_encode(
    family: PixelFamily,
    method: Option<Method>,
    mode: ExtradataMode,
    w: u32,
    h: u32,
    pixels: &[u8],
    budget: Option<usize>,
) -> (Vec<u8>, Vec<u8>) {
    use oxideav_core::{CodecId, CodecParameters, ExecutionContext, Frame, VideoFrame, VideoPlane};
    let mut params = CodecParameters::video(CodecId::new(oxideav_huffyuv::CODEC_ID_STR));
    params.width = Some(w);
    params.height = Some(h);
    params.options.insert("format", family_option(family));
    params.options.insert("mode", mode_option(mode));
    if let Some(m) = method {
        params.options.insert("method", method_option(m));
    }
    let mut enc = make_encoder(&params).expect("make_encoder");
    if let Some(threads) = budget {
        enc.set_execution_context(&ExecutionContext { threads });
    }
    let frame = Frame::Video(VideoFrame {
        pts: Some(0),
        planes: vec![VideoPlane {
            stride: family.row_bytes(w),
            data: if family.is_rgb() {
                pixels
                    .chunks_exact(family.row_bytes(w))
                    .rev()
                    .flatten()
                    .copied()
                    .collect()
            } else {
                pixels.to_vec()
            },
        }],
    });
    enc.send_frame(&frame).expect("send_frame");
    let pkt = enc.receive_packet().expect("receive_packet");
    (enc.output_params().extradata.clone(), pkt.data)
}

/// Round-429 trait-path pins: the registry `Encoder` must emit the
/// EXACT r419 pinned bytes on every full-matrix scenario — strf via
/// `output_params()`, wire bytes via the packet — at the default
/// budget AND under budget 2 (interlaced pins exercise the r420
/// field-parallel arms through the trait).
#[test]
fn golden_pins_trait_path_full_matrix() {
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
                    let (strf, frame) =
                        trait_encode(family, Some(method), mode, w, h, &pixels, None);
                    let (strf_b2, frame_b2) =
                        trait_encode(family, Some(method), mode, w, h, &pixels, Some(2));
                    assert_eq!(strf, strf_b2, "{scenario}: trait budget-2 strf diverged");
                    assert_eq!(
                        frame, frame_b2,
                        "{scenario}: trait budget-2 wire bytes diverged"
                    );
                    seen.push((scenario, fnv1a64(&strf), fnv1a64(&frame)));
                }
            }
        }
    }
    assert_eq!(seen.len(), EXPECTED.len(), "trait pin matrix drifted");
    for ((scenario, strf_hash, frame_hash), (exp_scenario, exp_strf, exp_frame)) in
        seen.iter().zip(EXPECTED.iter())
    {
        assert_eq!(scenario, exp_scenario, "scenario order drifted");
        assert_eq!(
            *strf_hash, *exp_strf,
            "{scenario}: trait-path strf diverged from the pinned direct-API bytes"
        );
        assert_eq!(
            *frame_hash, *exp_frame,
            "{scenario}: trait-path wire bytes diverged from the pinned direct-API bytes"
        );
    }
}

/// Round-429 trait-path auto pins: a registry encoder with the default
/// `method=auto` must reproduce the pinned auto-selector output
/// (winner's strf + wire bytes) on its first frame.
#[test]
fn golden_pins_trait_path_auto_selector() {
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
            let (strf, frame) = trait_encode(family, None, mode, w, h, &pixels, None);
            seen.push((scenario, fnv1a64(&strf), fnv1a64(&frame)));
        }
    }
    assert_eq!(
        seen.len(),
        EXPECTED_AUTO.len(),
        "trait auto pin matrix drifted"
    );
    for ((scenario, strf_hash, frame_hash), (exp_scenario, _, exp_strf, exp_frame)) in
        seen.iter().zip(EXPECTED_AUTO.iter())
    {
        assert_eq!(scenario, exp_scenario, "scenario order drifted");
        assert_eq!(
            *strf_hash, *exp_strf,
            "{scenario}: trait auto strf diverged"
        );
        assert_eq!(
            *frame_hash, *exp_frame,
            "{scenario}: trait auto wire bytes diverged"
        );
    }
}

/// Round-429 auto-selector budget sweep: `encode_frame_auto_workers`
/// must return the identical `(strf, wire bytes, chosen method)`
/// triple under budgets 1 / 2 / 8 on every auto pin scenario at BOTH
/// pin sizes — the parallel candidate scoring (progressive + interlaced)
/// and the two-worker winner emit (interlaced) are wire-invariant.
#[test]
fn golden_pins_auto_budget_sweep() {
    for family in FAMILIES {
        for mode in [ExtradataMode::ClassicV2, ExtradataMode::CustomV2] {
            for (w, h) in SIZES {
                let scenario = format!(
                    "{}/auto/{}/{}x{}",
                    family_name(family),
                    mode_name(mode),
                    w,
                    h
                );
                let pixels = build_pixels(w as usize, h as usize, bpp(family));
                let serial = encode_frame_auto(family, MethodSelection::Auto, w, h, &pixels, mode)
                    .unwrap_or_else(|e| panic!("{scenario}: serial auto failed: {e:?}"));
                for budget in [1usize, 2, 8] {
                    let budgeted = encode_frame_auto_workers(
                        family,
                        MethodSelection::Auto,
                        w,
                        h,
                        &pixels,
                        mode,
                        budget,
                    )
                    .unwrap_or_else(|e| panic!("{scenario}: budget-{budget} auto failed: {e:?}"));
                    assert_eq!(
                        budgeted, serial,
                        "{scenario}: budget-{budget} auto diverged from serial"
                    );
                }
            }
        }
    }
}
