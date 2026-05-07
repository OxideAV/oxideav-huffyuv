# oxideav-huffyuv

Pure-Rust HuffYUV / FFVHuff lossless video codec for the
[oxideav](https://github.com/OxideAV/oxideav-workspace) framework.

## Status

**Round 0 — clean-room rebuild scaffold.** This `master` branch is a
fresh orphan. The previous implementation was retired alongside the
docs audit dated 2026-05-06 (see
[`AUDIT-2026-05-06.md`](https://github.com/OxideAV/docs/blob/master/AUDIT-2026-05-06.md)),
which found that the source-of-record trace document for this codec
was authored with a methodology that did not satisfy clean-room
separation. The prior history is preserved on the `old` branch for
archival but is forbidden input for the rebuild.

The new implementation will be built against the strict-isolation clean-room workspace at [`docs/video/huffyuv/`](https://github.com/OxideAV/docs/tree/master/video/huffyuv). The Implementer in this repo reads only `spec/` and `tables/` (plus `reference/docs/` if the workspace exposes a public format spec).

The `oxideav_core::CodecResolver` registration this crate's
`register(ctx)` function provides will be wired up by the
Implementer round; until then the public API surfaces only the
crate-local `Error::NotImplemented` placeholder.
