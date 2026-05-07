//! Pure-Rust HuffYUV / FFVHuff lossless video codec.
//!
//! **Round 0 — clean-room rebuild scaffold.** This is a fresh orphan
//! `master`; the previous implementation was retired alongside the
//! OxideAV docs audit dated 2026-05-06. See `README.md` for the
//! rebuild scope and the strict-isolation clean-room workspace the
//! Implementer rounds will draw from.

#![forbid(unsafe_code)]

/// Crate-local error type. Concrete variants land as the Implementer
/// rounds populate the codec pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Reserved placeholder. Replaced by real variants in round 1.
    NotImplemented,
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::NotImplemented => f.write_str(
                "oxideav-huffyuv: clean-room rebuild in progress — see crates/oxideav-huffyuv/README.md",
            ),
        }
    }
}

impl std::error::Error for Error {}

/// Crate-local Result alias.
pub type Result<T> = core::result::Result<T, Error>;
