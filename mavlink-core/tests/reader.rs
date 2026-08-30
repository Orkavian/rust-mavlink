//! Public reader and streaming-behavior regressions.

#[cfg(feature = "std")]
use mavlink_core::MAVLinkV1MessageRaw;
use mavlink_core::{MAVLinkV2MessageRaw, MavHeader, MavlinkVersion, Message, error::ParserError};

const MESSAGE_ID: u32 = 42;
const EXTRA_CRC: u8 = 91;
const INVALID_VALUE: u8 = u8::MAX;
const HEADER: MavHeader = MavHeader {
    system_id: 11,
    component_id: 22,
    sequence: 33,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TestMessage(u8);

impl Message for TestMessage {
    fn message_id(&self) -> u32 {
        MESSAGE_ID
    }

    fn message_name(&self) -> &'static str {
        "TEST_MESSAGE"
    }

    fn target_system_id(&self) -> Option<u8> {
        None
    }

    fn target_component_id(&self) -> Option<u8> {
        None
    }

    fn ser(&self, _version: MavlinkVersion, bytes: &mut [u8]) -> usize {
        bytes[0] = self.0;
        1
    }

    fn ser_with_metadata(&self, version: MavlinkVersion, bytes: &mut [u8]) -> (usize, u32, u8) {
        (self.ser(version, bytes), MESSAGE_ID, EXTRA_CRC)
    }

    fn parse(_version: MavlinkVersion, msgid: u32, payload: &[u8]) -> Result<Self, ParserError> {
        if msgid != MESSAGE_ID {
            return Err(ParserError::UnknownMessage { id: msgid });
        }

        let value = payload.first().copied().unwrap_or(INVALID_VALUE);
        if value == INVALID_VALUE {
            return Err(ParserError::InvalidEnum {
                enum_type: "TestEnum",
                value: value.into(),
            });
        }
        Ok(Self(value))
    }

    fn message_id_from_name(name: &str) -> Option<u32> {
        (name == "TEST_MESSAGE").then_some(MESSAGE_ID)
    }

    fn default_message_from_id(id: u32) -> Option<Self> {
        (id == MESSAGE_ID).then_some(Self(0))
    }

    #[cfg(feature = "arbitrary")]
    fn random_message_from_id<R: rand::TryRng<Error = core::convert::Infallible>>(
        id: u32,
        _rng: &mut R,
    ) -> Option<Self> {
        Self::default_message_from_id(id)
    }

    fn extra_crc(id: u32) -> u8 {
        if id == MESSAGE_ID { EXTRA_CRC } else { 0 }
    }
}

#[cfg(feature = "std")]
fn v1_frame(value: u8) -> MAVLinkV1MessageRaw {
    let mut frame = MAVLinkV1MessageRaw::new();
    frame.serialize_message(HEADER, &TestMessage(value));
    frame
}

fn v2_frame(value: u8) -> MAVLinkV2MessageRaw {
    let mut frame = MAVLinkV2MessageRaw::new();
    frame.serialize_message(HEADER, &TestMessage(value));
    frame
}

#[cfg(feature = "std")]
mod standard {
    use mavlink_core::{
        MAV_STX_V2, MavlinkReader, Message, ReadVersion, calculate_crc, error::MessageReadError,
    };

    use super::{HEADER, INVALID_VALUE, TestMessage, v1_frame, v2_frame};

    #[test]
    fn adjacent_v1_and_v2_frames_are_independently_readable() {
        let v1 = v1_frame(7);
        let v2 = v2_frame(8);
        let mut stream = Vec::with_capacity(v1.raw_bytes().len() + v2.raw_bytes().len());
        stream.extend_from_slice(v1.raw_bytes());
        stream.extend_from_slice(v2.raw_bytes());
        let mut reader = MavlinkReader::new(stream.as_slice());

        assert_eq!(
            reader
                .read_message::<TestMessage>(ReadVersion::Any)
                .unwrap(),
            (HEADER, TestMessage(7))
        );
        assert_eq!(
            reader
                .read_message::<TestMessage>(ReadVersion::Any)
                .unwrap(),
            (HEADER, TestMessage(8))
        );
    }

    #[test]
    fn invalid_crc_candidate_uses_sliding_recovery() {
        let valid = v2_frame(8);

        // The real frame begins inside a complete false candidate. A decoder
        // that skips the whole rejected candidate instead of sliding one byte
        // loses the valid frame's marker.
        let mut stream = vec![MAV_STX_V2, 0, 0];
        stream.extend_from_slice(valid.raw_bytes());

        let false_checksum = u16::from_le_bytes([stream[10], stream[11]]);
        let false_message_id = u32::from_le_bytes([stream[7], stream[8], stream[9], 0]);
        assert_ne!(
            calculate_crc(&stream[1..10], TestMessage::extra_crc(false_message_id)),
            false_checksum,
            "test setup must begin with a checksum-invalid candidate"
        );
        let mut reader = MavlinkReader::new(stream.as_slice());

        assert_eq!(
            reader
                .read_message::<TestMessage>(mavlink_core::MavlinkVersion::V2.into())
                .unwrap(),
            (HEADER, TestMessage(8))
        );
    }

    #[test]
    fn valid_crc_parse_error_consumes_exactly_one_frame() {
        let invalid = v2_frame(INVALID_VALUE);
        let valid = v2_frame(8);
        let mut stream = invalid.raw_bytes().to_vec();
        stream.extend_from_slice(valid.raw_bytes());
        let mut reader = MavlinkReader::new(stream.as_slice());

        assert!(matches!(
            reader.read_message::<TestMessage>(mavlink_core::MavlinkVersion::V2.into()),
            Err(MessageReadError::Parse(
                mavlink_core::error::ParserError::InvalidEnum { .. }
            ))
        ));
        assert_eq!(
            reader
                .read_message::<TestMessage>(mavlink_core::MavlinkVersion::V2.into())
                .unwrap(),
            (HEADER, TestMessage(8))
        );
    }

    #[test]
    fn version_selection_remains_public() {
        let version = ReadVersion::from(mavlink_core::MavlinkVersion::V2);
        assert_eq!(
            version,
            ReadVersion::Single(mavlink_core::MavlinkVersion::V2)
        );
    }
}

#[cfg(all(feature = "embedded", not(feature = "std")))]
mod embedded {
    use core::convert::Infallible;

    use mavlink_core::MavlinkReader;

    use super::{HEADER, TestMessage, v2_frame};

    struct OneFrame {
        bytes: [u8; 280],
        len: usize,
        delivered: bool,
    }

    impl embedded_io::ErrorType for OneFrame {
        type Error = Infallible;
    }

    impl embedded_io::Read for OneFrame {
        fn read(&mut self, destination: &mut [u8]) -> Result<usize, Self::Error> {
            assert!(
                !self.delivered,
                "read-ahead must use one read-some operation, not fill the tail exactly"
            );
            destination[..self.len].copy_from_slice(&self.bytes[..self.len]);
            self.delivered = true;
            Ok(self.len)
        }
    }

    #[test]
    fn embedded_prefetch_uses_read_some_semantics() {
        let frame = v2_frame(9);
        let mut bytes = [0; 280];
        let len = frame.raw_bytes().len();
        bytes[..len].copy_from_slice(frame.raw_bytes());
        let source = OneFrame {
            bytes,
            len,
            delivered: false,
        };
        let mut reader = MavlinkReader::new(source);

        assert_eq!(
            reader
                .read_message::<TestMessage>(mavlink_core::MavlinkVersion::V2.into())
                .unwrap(),
            (HEADER, TestMessage(9))
        );
    }
}
