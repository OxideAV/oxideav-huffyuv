# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.1](https://github.com/OxideAV/oxideav-huffyuv/releases/tag/v0.0.1) - 2026-05-03

### Other

- replace never-match regex with semver_check = false
- bootstrap clean-room HuffYUV / FFVHuff decoder
- Initial commit

### Added

- Initial scaffold and `huffyuv` v2 / v3 decoder.
- Extradata parser (RLE-coded canonical-Huffman length tables).
- Predictors: LEFT, GRADIENT (PLANE), MEDIAN.
- Pixel formats: yuv422p (v2), yuv420p (v2), yuv444p (v3), gray8 (v3),
  rgb24 (v2 + decorrelate), bgra (v2 + decorrelate).
- Per-frame Huffman tables (`-context 1`, ffvhuff) decode path.
- 32-bit byte-swap unwrap before bit-reading per the on-disk word order.
