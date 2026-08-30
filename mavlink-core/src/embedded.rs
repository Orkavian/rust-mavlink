use crate::error::*;

/// `no_std` counterpart to [`std::io::Read`].
pub trait Read {
    /// Read some bytes without waiting to fill the entire destination.
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, MessageReadError>;
}

impl<R: embedded_io::Read> Read for R {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, MessageReadError> {
        embedded_io::Read::read(self, buf).map_err(|_| MessageReadError::Io)
    }
}

/// `no_std` counterpart to [`std::io::Write`].
pub trait Write {
    fn write_all(&mut self, buf: &[u8]) -> Result<(), MessageWriteError>;
}

impl<W: embedded_io::Write> Write for W {
    fn write_all(&mut self, buf: &[u8]) -> Result<(), MessageWriteError> {
        embedded_io::Write::write_all(self, buf).map_err(|_| MessageWriteError::Io)
    }
}
