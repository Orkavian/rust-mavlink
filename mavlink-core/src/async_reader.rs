//! Asynchronous MAVLink stream reader.

#[cfg(feature = "tokio")]
use crate::SigningData;
use crate::frame_decoder::FrameDecoder;
use crate::reader::{try_decode_message, try_decode_raw};
use crate::{MAVLinkMessageRaw, MavHeader, Message, ReadVersion, error::MessageReadError};

/// Allocation-free asynchronous MAVLink reader using the protocol's maximum
/// frame size as its fixed read-ahead capacity.
///
/// The same portable decoder is shared with [`crate::MavlinkReader`]; only the
/// operation used to fetch more bytes is asynchronous.
pub struct AsyncMavlinkReader<R> {
    decoder: FrameDecoder,
    reader: R,
}

impl<R> AsyncMavlinkReader<R> {
    /// Wrap an asynchronous input source in a MAVLink-aware reader.
    #[must_use]
    pub const fn new(reader: R) -> Self {
        Self {
            decoder: FrameDecoder::new(),
            reader,
        }
    }

    /// Borrow the underlying input source.
    #[must_use]
    pub const fn get_ref(&self) -> &R {
        &self.reader
    }

    /// Mutably borrow the underlying input source.
    ///
    /// Reading from it directly bypasses bytes already buffered by this
    /// reader and can therefore cause data loss.
    #[must_use]
    pub fn get_mut(&mut self) -> &mut R {
        &mut self.reader
    }

    /// Return the input source, discarding any prefetched bytes.
    #[must_use]
    pub fn into_inner(self) -> R {
        self.reader
    }
}

#[cfg(feature = "tokio")]
impl<R: tokio::io::AsyncRead + Unpin> AsyncMavlinkReader<R> {
    /// Read and parse the next checksum-valid message matching `version`.
    pub async fn read_message<M: Message>(
        &mut self,
        version: ReadVersion,
    ) -> Result<(MavHeader, M), MessageReadError> {
        self.read_message_inner::<M>(version, None).await
    }

    /// Read the next checksum-valid raw message matching `version`.
    pub async fn read_raw_message<M: Message>(
        &mut self,
        version: ReadVersion,
    ) -> Result<MAVLinkMessageRaw, MessageReadError> {
        self.read_raw_message_inner::<M>(version, None).await
    }

    /// Read, verify, and parse the next message matching `version`.
    #[cfg(feature = "mav2-message-signing")]
    pub async fn read_message_signed<M: Message>(
        &mut self,
        version: ReadVersion,
        signing_data: &SigningData,
    ) -> Result<(MavHeader, M), MessageReadError> {
        self.read_message_inner::<M>(version, Some(signing_data))
            .await
    }

    /// Read and verify the next raw message matching `version`.
    #[cfg(feature = "mav2-message-signing")]
    pub async fn read_raw_message_signed<M: Message>(
        &mut self,
        version: ReadVersion,
        signing_data: &SigningData,
    ) -> Result<MAVLinkMessageRaw, MessageReadError> {
        self.read_raw_message_inner::<M>(version, Some(signing_data))
            .await
    }

    pub(crate) async fn read_message_inner<M: Message>(
        &mut self,
        version: ReadVersion,
        signing_data: Option<&SigningData>,
    ) -> Result<(MavHeader, M), MessageReadError> {
        loop {
            if let Some(message) = try_decode_message::<M>(&mut self.decoder, version, signing_data)
            {
                return message;
            }
            self.read_more::<M>().await?;
        }
    }

    pub(crate) async fn read_raw_message_inner<M: Message>(
        &mut self,
        version: ReadVersion,
        signing_data: Option<&SigningData>,
    ) -> Result<MAVLinkMessageRaw, MessageReadError> {
        loop {
            if let Some(message) = try_decode_raw::<M>(&mut self.decoder, version, signing_data) {
                return message;
            }
            self.read_more::<M>().await?;
        }
    }

    async fn read_more<M: Message>(&mut self) -> Result<(), MessageReadError> {
        use tokio::io::AsyncReadExt;

        let destination = self.decoder.spare_capacity_mut();
        assert!(
            !destination.is_empty(),
            "MAVLink reader buffer is too small for the pending frame"
        );
        let count = self.reader.read(destination).await?;
        if count == 0 {
            return Err(MessageReadError::eof());
        }
        self.decoder
            .commit(count, |_, message_id| M::extra_crc(message_id))
            .expect("read returned more bytes than the provided destination");
        Ok(())
    }
}

#[cfg(all(feature = "embedded", not(feature = "std"), not(feature = "tokio")))]
impl<R: embedded_io_async::Read> AsyncMavlinkReader<R> {
    /// Read and parse the next checksum-valid message matching `version`.
    pub async fn read_message<M: Message>(
        &mut self,
        version: ReadVersion,
    ) -> Result<(MavHeader, M), MessageReadError> {
        loop {
            if let Some(message) = try_decode_message::<M>(&mut self.decoder, version, None) {
                return message;
            }
            self.read_more::<M>().await?;
        }
    }

    /// Read the next checksum-valid raw message matching `version`.
    pub async fn read_raw_message<M: Message>(
        &mut self,
        version: ReadVersion,
    ) -> Result<MAVLinkMessageRaw, MessageReadError> {
        loop {
            if let Some(message) = try_decode_raw::<M>(&mut self.decoder, version, None) {
                return message;
            }
            self.read_more::<M>().await?;
        }
    }

    async fn read_more<M: Message>(&mut self) -> Result<(), MessageReadError> {
        let destination = self.decoder.spare_capacity_mut();
        assert!(
            !destination.is_empty(),
            "MAVLink reader buffer is too small for the pending frame"
        );
        let count = self
            .reader
            .read(destination)
            .await
            .map_err(|_| MessageReadError::Io)?;
        if count == 0 {
            return Err(MessageReadError::eof());
        }
        self.decoder
            .commit(count, |_, message_id| M::extra_crc(message_id))
            .expect("read returned more bytes than the provided destination");
        Ok(())
    }
}
