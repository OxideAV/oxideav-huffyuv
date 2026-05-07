//! Crate-local error type.
//!
//! Mirrors the constructor shape of `oxideav_core::Error` so callers
//! can build messages identically whether or not the `registry`
//! feature is enabled.

/// HuffYUV decode/encode error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The input bytes did not parse against the wire format.
    InvalidData(String),
    /// The codec configuration is recognised but not supported by
    /// this round of the implementation.
    Unsupported(String),
}

impl Error {
    pub fn invalid<S: Into<String>>(msg: S) -> Self {
        Error::InvalidData(msg.into())
    }
    pub fn unsupported<S: Into<String>>(msg: S) -> Self {
        Error::Unsupported(msg.into())
    }
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::InvalidData(m) => write!(f, "huffyuv: invalid data: {m}"),
            Error::Unsupported(m) => write!(f, "huffyuv: unsupported: {m}"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = core::result::Result<T, Error>;
