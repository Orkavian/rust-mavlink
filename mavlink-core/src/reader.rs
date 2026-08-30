//! Blocking MAVLink stream reader.
//!
//! [`MavlinkReader`] combines an ordinary [`Read`] implementation with the
//! protocol decoder. Buffering is an implementation detail: callers ask for
//! complete MAVLink messages rather than manually peeking and consuming bytes.

#[cfg(all(feature = "embedded", not(feature = "std")))]
use crate::embedded::Read;
#[cfg(feature = "std")]
use std::io::Read;

use crate::frame_decoder::{DecodeStatus, FrameDecoder, FrameRef, FrameVersion};
use crate::{
    MAVLinkMessageRaw, MAVLinkV1MessageRaw, MAVLinkV2MessageRaw, MavHeader, MavlinkVersion,
    Message, ReadVersion, SigningData, error::MessageReadError,
};

/// Allocation-free blocking MAVLink reader with fixed maximum-frame read-ahead.
pub struct MavlinkReader<R> {
    decoder: FrameDecoder,
    reader: R,
}

impl<R> MavlinkReader<R> {
    /// Wrap an input source in a MAVLink-aware reader.
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

impl<R: Read> MavlinkReader<R> {
    /// Read and parse the next checksum-valid message matching `version`.
    pub fn read_message<M: Message>(
        &mut self,
        version: ReadVersion,
    ) -> Result<(MavHeader, M), MessageReadError> {
        self.read_message_inner::<M>(version, None)
    }

    /// Read the next checksum-valid raw message matching `version`.
    pub fn read_raw_message<M: Message>(
        &mut self,
        version: ReadVersion,
    ) -> Result<MAVLinkMessageRaw, MessageReadError> {
        self.read_raw_message_inner::<M>(version, None)
    }

    /// Read, verify, and parse the next message matching `version`.
    #[cfg(feature = "mav2-message-signing")]
    pub fn read_message_signed<M: Message>(
        &mut self,
        version: ReadVersion,
        signing_data: &SigningData,
    ) -> Result<(MavHeader, M), MessageReadError> {
        self.read_message_inner::<M>(version, Some(signing_data))
    }

    /// Read and verify the next raw message matching `version`.
    #[cfg(feature = "mav2-message-signing")]
    pub fn read_raw_message_signed<M: Message>(
        &mut self,
        version: ReadVersion,
        signing_data: &SigningData,
    ) -> Result<MAVLinkMessageRaw, MessageReadError> {
        self.read_raw_message_inner::<M>(version, Some(signing_data))
    }

    pub(crate) fn read_message_inner<M: Message>(
        &mut self,
        version: ReadVersion,
        signing_data: Option<&SigningData>,
    ) -> Result<(MavHeader, M), MessageReadError> {
        loop {
            if let Some(message) = try_decode_message::<M>(&mut self.decoder, version, signing_data)
            {
                return message;
            }
            self.read_more::<M>()?;
        }
    }

    pub(crate) fn read_raw_message_inner<M: Message>(
        &mut self,
        version: ReadVersion,
        signing_data: Option<&SigningData>,
    ) -> Result<MAVLinkMessageRaw, MessageReadError> {
        loop {
            if let Some(message) = try_decode_raw::<M>(&mut self.decoder, version, signing_data) {
                return message;
            }
            self.read_more::<M>()?;
        }
    }

    fn read_more<M: Message>(&mut self) -> Result<(), MessageReadError> {
        let destination = self.decoder.spare_capacity_mut();
        assert!(
            !destination.is_empty(),
            "MAVLink reader buffer is too small for the pending frame"
        );
        let count = self.reader.read(destination)?;
        if count == 0 {
            return Err(MessageReadError::eof());
        }
        self.decoder
            .commit(count, |_, message_id| M::extra_crc(message_id))
            .expect("read returned more bytes than the provided destination");
        Ok(())
    }
}

pub(crate) fn try_decode_message<M: Message>(
    decoder: &mut FrameDecoder,
    version: ReadVersion,
    signing_data: Option<&SigningData>,
) -> Option<Result<(MavHeader, M), MessageReadError>> {
    loop {
        let progress = decoder
            .commit(0, |_, message_id| M::extra_crc(message_id))
            .expect("committing zero bytes cannot exceed decoder capacity");
        if progress.status == DecodeStatus::NeedMore {
            return None;
        }

        let frame = decoder.frame().expect("frame-ready status has a frame");
        if !version_accepts(version, frame.version()) {
            decoder.advance();
            continue;
        }

        #[cfg(feature = "mav2-message-signing")]
        if let Some(signing_data) = signing_data {
            let raw = raw_from_frame(frame);
            if !signature_is_valid(&raw, version, signing_data) {
                decoder.advance();
                continue;
            }
            let header = raw_header(&raw);
            let parsed = M::parse(raw.version(), raw.message_id(), raw.payload());
            decoder.advance();
            return Some(parsed.map(|message| (header, message)).map_err(Into::into));
        }

        let _ = signing_data;
        let header = frame_header(frame);
        let parsed = M::parse(
            frame_version(frame.version()),
            frame.message_id(),
            frame.payload(),
        );
        decoder.advance();
        return Some(parsed.map(|message| (header, message)).map_err(Into::into));
    }
}

pub(crate) fn try_decode_raw<M: Message>(
    decoder: &mut FrameDecoder,
    version: ReadVersion,
    signing_data: Option<&SigningData>,
) -> Option<Result<MAVLinkMessageRaw, MessageReadError>> {
    loop {
        let progress = decoder
            .commit(0, |_, message_id| M::extra_crc(message_id))
            .expect("committing zero bytes cannot exceed decoder capacity");
        if progress.status == DecodeStatus::NeedMore {
            return None;
        }

        let frame = decoder.frame().expect("frame-ready status has a frame");
        if !version_accepts(version, frame.version()) {
            decoder.advance();
            continue;
        }

        let raw = raw_from_frame(frame);
        #[cfg(feature = "mav2-message-signing")]
        if let Some(signing_data) = signing_data {
            if !signature_is_valid(&raw, version, signing_data) {
                decoder.advance();
                continue;
            }
        }
        let _ = signing_data;
        decoder.advance();
        return Some(Ok(raw));
    }
}

fn version_accepts(expected: ReadVersion, actual: FrameVersion) -> bool {
    match expected {
        ReadVersion::Any => true,
        ReadVersion::Single(expected) => expected == frame_version(actual),
    }
}

const fn frame_version(version: FrameVersion) -> MavlinkVersion {
    match version {
        FrameVersion::V1 => MavlinkVersion::V1,
        FrameVersion::V2 => MavlinkVersion::V2,
    }
}

fn frame_header(frame: FrameRef<'_>) -> MavHeader {
    MavHeader {
        sequence: frame.sequence(),
        system_id: frame.system_id(),
        component_id: frame.component_id(),
    }
}

#[cfg(feature = "mav2-message-signing")]
fn raw_header(raw: &MAVLinkMessageRaw) -> MavHeader {
    MavHeader {
        sequence: raw.sequence(),
        system_id: raw.system_id(),
        component_id: raw.component_id(),
    }
}

fn raw_from_frame(frame: FrameRef<'_>) -> MAVLinkMessageRaw {
    match frame.version() {
        FrameVersion::V1 => {
            let mut raw = MAVLinkV1MessageRaw::new();
            raw.0[..frame.bytes().len()].copy_from_slice(frame.bytes());
            MAVLinkMessageRaw::V1(raw)
        }
        FrameVersion::V2 => {
            let mut raw = MAVLinkV2MessageRaw::new();
            raw.0[..frame.bytes().len()].copy_from_slice(frame.bytes());
            MAVLinkMessageRaw::V2(raw)
        }
    }
}

#[cfg(feature = "mav2-message-signing")]
fn signature_is_valid(
    raw: &MAVLinkMessageRaw,
    version: ReadVersion,
    signing_data: &SigningData,
) -> bool {
    match raw {
        // Signing is explicitly ignored when the caller requested MAVLink 1.
        MAVLinkMessageRaw::V1(_) if version == ReadVersion::Single(MavlinkVersion::V1) => true,
        MAVLinkMessageRaw::V1(_) => signing_data.config.allow_unsigned,
        MAVLinkMessageRaw::V2(message) => signing_data.verify_signature(message),
    }
}
