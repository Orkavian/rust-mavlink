use core::fmt::{Display, Formatter};
#[cfg(feature = "std")]
use std::error::Error;

/// A byte slice did not contain enough data for a wire-format value.
///
/// This type is intentionally independent of any cursor or I/O abstraction so
/// it remains usable in allocation-free and `no_std` builds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferError {
    /// The input ended before the requested value could be read.
    UnexpectedEnd {
        /// Number of bytes required for the value being read.
        needed: usize,
        /// Number of bytes available at the read position.
        remaining: usize,
    },
}

impl BufferError {
    pub(crate) const fn new(needed: usize, remaining: usize) -> Self {
        Self::UnexpectedEnd { needed, remaining }
    }
}

impl Display for BufferError {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnexpectedEnd { needed, remaining } => {
                write!(f, "needed {needed} bytes but only {remaining} remain")
            }
        }
    }
}

/// Error while parsing a MAVLink message
#[derive(Debug)]
pub enum ParserError {
    /// Enum value for this enum type does not exist
    InvalidEnum { enum_type: &'static str, value: u64 },
    /// Message ID does not exist in this message set
    UnknownMessage { id: u32 },
    /// The input buffer ended before a complete value could be read.
    BufferError(BufferError),
}

impl From<BufferError> for ParserError {
    fn from(error: BufferError) -> Self {
        Self::BufferError(error)
    }
}

impl Display for ParserError {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidEnum { enum_type, value } => write!(
                f,
                "Invalid enum value for enum type {enum_type:?}, got {value:?}"
            ),
            Self::UnknownMessage { id } => write!(f, "Unknown message with ID {id:?}"),
            Self::BufferError(error) => write!(f, "{error}"),
        }
    }
}

#[cfg(feature = "std")]
impl Error for ParserError {}

/// Error while reading and parsing a MAVLink message
#[derive(Debug)]
pub enum MessageReadError {
    /// IO Error while reading
    #[cfg(feature = "std")]
    Io(std::io::Error),
    /// IO Error while reading
    #[cfg(all(feature = "embedded", not(feature = "std")))]
    Io,
    /// Error while parsing
    Parse(ParserError),
}

impl MessageReadError {
    pub fn eof() -> Self {
        #[cfg(feature = "std")]
        return Self::Io(std::io::ErrorKind::UnexpectedEof.into());
        #[cfg(all(feature = "embedded", not(feature = "std")))]
        return Self::Io;
    }
}

impl Display for MessageReadError {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            #[cfg(feature = "std")]
            Self::Io(e) => write!(f, "Failed to read message: {e:#?}"),
            #[cfg(all(feature = "embedded", not(feature = "std")))]
            Self::Io => write!(f, "Failed to read message"),
            Self::Parse(e) => write!(f, "Failed to read message: {e:#?}"),
        }
    }
}

#[cfg(feature = "std")]
impl Error for MessageReadError {}

#[cfg(feature = "std")]
impl From<std::io::Error> for MessageReadError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<ParserError> for MessageReadError {
    fn from(e: ParserError) -> Self {
        Self::Parse(e)
    }
}

/// Error while writing a MAVLink message
#[derive(Debug)]
pub enum MessageWriteError {
    /// IO Error while writing
    #[cfg(feature = "std")]
    Io(std::io::Error),
    /// IO Error while writing
    #[cfg(all(feature = "embedded", not(feature = "std")))]
    Io,
    /// Message does not support MAVLink 1
    MAVLink2Only,
}

impl Display for MessageWriteError {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            #[cfg(feature = "std")]
            Self::Io(e) => write!(f, "Failed to write message: {e:#?}"),
            #[cfg(all(feature = "embedded", not(feature = "std")))]
            Self::Io => write!(f, "Failed to write message"),
            Self::MAVLink2Only => write!(f, "Message is not supported in MAVLink 1"),
        }
    }
}

#[cfg(feature = "std")]
impl Error for MessageWriteError {}

#[cfg(feature = "std")]
impl From<std::io::Error> for MessageWriteError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}
