//! Allocation-free, incremental MAVLink frame boundary detection.
//!
//! [`FrameDecoder`] owns a fixed read-ahead buffer, sized for one maximum
//! MAVLink frame by default. Callers may feed it arbitrarily-sized chunks,
//! inspect a validated frame through
//! [`FrameDecoder::frame`], then release that frame with
//! [`FrameDecoder::advance`]. The decoder validates the MAVLink checksum but
//! deliberately leaves message parsing and MAVLink 2 signature policy to the
//! caller.

/// Largest possible MAVLink frame: a MAVLink 2 frame with a 255-byte payload
/// and a 13-byte signature trailer.
pub const MAX_FRAME_LEN: usize = crate::consts::MAX_FRAME_SIZE;

const V1_STX: u8 = 0xfe;
const V2_STX: u8 = 0xfd;
const V1_HEADER_LEN: usize = crate::consts::v1::HEADER_SIZE;
const V2_HEADER_LEN: usize = crate::consts::v2::HEADER_SIZE;
const CHECKSUM_LEN: usize = crate::consts::CHECKSUM_SIZE;
const SIGNATURE_LEN: usize = crate::consts::v2::SIGNATURE_SIZE;
const V2_SIGNED_FLAG: u8 = crate::consts::v2::IFLAG_SIGNED;
const V2_SUPPORTED_FLAGS: u8 = crate::consts::v2::SUPPORTED_IFLAGS;
const VECTORIZED_MARKER_SEARCH_MINIMUM: usize = 64;
// Nearby markers are cheaper to find without SIMD setup, even in a long buffer.
const SCALAR_MARKER_SEARCH_PREFIX: usize = 24;

/// MAVLink wire version identified by the frame marker.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrameVersion {
    /// MAVLink 1 (`0xfe`).
    V1,
    /// MAVLink 2 (`0xfd`).
    V2,
}

/// State reached after feeding an input chunk.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecodeStatus {
    /// More bytes are needed to produce a validated frame.
    NeedMore,
    /// A frame is available through [`FrameDecoder::frame`].
    FrameReady,
}

/// Rejected candidate counts observed during one [`FrameDecoder::push`] call.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Rejections {
    /// Complete candidates whose checksum did not match.
    pub invalid_checksum: usize,
    /// MAVLink 2 candidates carrying an unsupported incompatibility flag.
    pub unsupported_flags: usize,
    /// Candidates that cannot fit in this decoder's configured buffer.
    pub buffer_too_small: usize,
}

impl Rejections {
    /// Total rejected candidates.
    #[must_use]
    pub const fn total(self) -> usize {
        self.invalid_checksum + self.unsupported_flags + self.buffer_too_small
    }
}

/// Error returned when more bytes are committed than were exposed by
/// [`FrameDecoder::spare_capacity_mut`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CommitError {
    /// Number of bytes the caller attempted to commit.
    pub requested: usize,
    /// Unused decoder capacity at the time of the call.
    pub available: usize,
}

impl core::fmt::Display for CommitError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            formatter,
            "attempted to commit {} bytes with only {} bytes available",
            self.requested, self.available
        )
    }
}

/// Result of feeding one input chunk into a [`FrameDecoder`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodeProgress {
    /// Number of bytes consumed from the supplied input chunk.
    ///
    /// If a frame becomes ready before the end of a chunk, pass the unconsumed
    /// suffix again after calling [`FrameDecoder::advance`].
    pub consumed: usize,
    /// Candidate rejections encountered while processing the chunk.
    pub rejections: Rejections,
    /// Decoder state after consuming `consumed` bytes.
    pub status: DecodeStatus,
}

/// Borrowed view of one checksum-validated MAVLink frame.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrameRef<'a> {
    bytes: &'a [u8],
    version: FrameVersion,
}

impl<'a> FrameRef<'a> {
    /// Entire frame, including marker, header, payload, checksum, and optional
    /// MAVLink 2 signature.
    #[must_use]
    pub const fn bytes(self) -> &'a [u8] {
        self.bytes
    }

    /// Wire version of this frame.
    #[must_use]
    pub const fn version(self) -> FrameVersion {
        self.version
    }

    /// Message identifier.
    #[must_use]
    pub fn message_id(self) -> u32 {
        match self.version {
            FrameVersion::V1 => u32::from(self.bytes[5]),
            FrameVersion::V2 => {
                u32::from_le_bytes([self.bytes[7], self.bytes[8], self.bytes[9], 0])
            }
        }
    }

    /// Packet sequence number.
    #[must_use]
    pub fn sequence(self) -> u8 {
        match self.version {
            FrameVersion::V1 => self.bytes[2],
            FrameVersion::V2 => self.bytes[4],
        }
    }

    /// Sender system identifier.
    #[must_use]
    pub fn system_id(self) -> u8 {
        match self.version {
            FrameVersion::V1 => self.bytes[3],
            FrameVersion::V2 => self.bytes[5],
        }
    }

    /// Sender component identifier.
    #[must_use]
    pub fn component_id(self) -> u8 {
        match self.version {
            FrameVersion::V1 => self.bytes[4],
            FrameVersion::V2 => self.bytes[6],
        }
    }

    /// Payload bytes.
    #[must_use]
    pub fn payload(self) -> &'a [u8] {
        let start = match self.version {
            FrameVersion::V1 => 1 + V1_HEADER_LEN,
            FrameVersion::V2 => 1 + V2_HEADER_LEN,
        };
        let end = start + usize::from(self.bytes[1]);
        &self.bytes[start..end]
    }

    /// Checksum carried by the frame.
    #[must_use]
    pub fn checksum(self) -> u16 {
        let checksum_start = match self.version {
            FrameVersion::V1 => 1 + V1_HEADER_LEN + self.payload().len(),
            FrameVersion::V2 => 1 + V2_HEADER_LEN + self.payload().len(),
        };
        u16::from_le_bytes([self.bytes[checksum_start], self.bytes[checksum_start + 1]])
    }

    /// MAVLink 2 incompatibility flags, or `None` for MAVLink 1.
    #[must_use]
    pub fn incompatibility_flags(self) -> Option<u8> {
        match self.version {
            FrameVersion::V1 => None,
            FrameVersion::V2 => Some(self.bytes[2]),
        }
    }

    /// Whether this is a signed MAVLink 2 frame.
    #[must_use]
    pub fn is_signed(self) -> bool {
        self.incompatibility_flags()
            .is_some_and(|flags| flags & V2_SIGNED_FLAG != 0)
    }

    /// MAVLink 2 signature trailer, including link ID, timestamp, and the
    /// six-byte signature value.
    #[must_use]
    pub fn signature(self) -> Option<&'a [u8]> {
        if !self.is_signed() {
            return None;
        }
        Some(&self.bytes[self.bytes.len() - SIGNATURE_LEN..])
    }
}

/// Fixed-capacity, allocation-free incremental MAVLink framing decoder.
///
/// A rejected candidate is rescanned beginning with its second byte. This
/// retains possible frame markers embedded in corrupt or falsely detected
/// candidates and matches byte-wise stream resynchronization.
///
/// `BUFFER_SIZE` defaults to [`MAX_FRAME_LEN`]. Smaller buffers are supported:
/// candidates that do not fit are rejected and scanning resumes from their
/// second byte instead of indexing out of bounds or panicking.
pub struct FrameDecoder<const BUFFER_SIZE: usize = MAX_FRAME_LEN> {
    buffer: [u8; BUFFER_SIZE],
    start: usize,
    end: usize,
    frame_len: usize,
    ready: bool,
}

impl<const BUFFER_SIZE: usize> Default for FrameDecoder<BUFFER_SIZE> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const BUFFER_SIZE: usize> FrameDecoder<BUFFER_SIZE> {
    /// Create an empty decoder.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            buffer: [0; BUFFER_SIZE],
            start: 0,
            end: 0,
            frame_len: 0,
            ready: false,
        }
    }

    /// Feed bytes into the decoder.
    ///
    /// `extra_crc` supplies the dialect-specific CRC extra byte for a message
    /// identifier. It is called only after a complete candidate is buffered.
    ///
    /// If this method returns [`DecodeStatus::FrameReady`], inspect the frame
    /// with [`Self::frame`] and call [`Self::advance`] before feeding more data.
    /// Calling `push` while a frame is still ready consumes no input and returns
    /// `FrameReady` again.
    pub fn push<F>(&mut self, input: &[u8], mut extra_crc: F) -> DecodeProgress
    where
        F: FnMut(FrameVersion, u32) -> u8,
    {
        let mut consumed = 0;
        let mut rejections = Rejections::default();

        loop {
            self.inspect(&mut extra_crc, &mut rejections);
            if self.ready {
                return DecodeProgress {
                    consumed,
                    rejections,
                    status: DecodeStatus::FrameReady,
                };
            }

            if consumed == input.len() {
                return DecodeProgress {
                    consumed,
                    rejections,
                    status: DecodeStatus::NeedMore,
                };
            }

            if self.buffered_len() == 0 {
                let remaining = &input[consumed..];
                let Some(marker_offset) = find_marker(remaining) else {
                    consumed = input.len();
                    continue;
                };
                consumed += marker_offset;
                if BUFFER_SIZE == 0 {
                    // There is nowhere to retain even the marker. Treat each
                    // marker as a capacity rejection and continue scanning.
                    rejections.buffer_too_small += 1;
                    consumed += 1;
                    continue;
                }
                self.buffer[0] = input[consumed];
                self.start = 0;
                self.end = 1;
                consumed += 1;
                continue;
            }

            let buffered = self.buffered_len();
            if buffered == BUFFER_SIZE {
                rejections.buffer_too_small += 1;
                self.reject_candidate();
                continue;
            }

            let target = self.next_target_len();
            let amount = (target - buffered)
                .min(input.len() - consumed)
                .min(BUFFER_SIZE - buffered);
            if self.end + amount > BUFFER_SIZE {
                self.compact();
            }
            self.buffer[self.end..self.end + amount]
                .copy_from_slice(&input[consumed..consumed + amount]);
            self.end += amount;
            consumed += amount;
        }
    }

    /// Mutable unused tail of the decoder's read-ahead buffer.
    ///
    /// Sync, Tokio, and embedded adapters can read directly into this slice,
    /// avoiding a second staging buffer. After a read completes, pass its byte
    /// count to [`Self::commit`] to append and immediately inspect for MAVLink
    /// frames.
    /// Calling this method does not initialize or logically append any bytes
    /// by itself.
    #[must_use]
    pub fn spare_capacity_mut(&mut self) -> &mut [u8] {
        if self.end == BUFFER_SIZE && self.start != 0 {
            self.compact();
        }
        &mut self.buffer[self.end..]
    }

    /// Commit bytes written into [`Self::spare_capacity_mut`] and inspect the
    /// enlarged read-ahead buffer for a frame.
    ///
    /// A count larger than the previously available tail is rejected without
    /// changing decoder state. `commit(0, ..)` is useful after [`Self::advance`]
    /// to decode a complete frame already present in the retained suffix.
    pub fn commit<F>(
        &mut self,
        count: usize,
        mut extra_crc: F,
    ) -> Result<DecodeProgress, CommitError>
    where
        F: FnMut(FrameVersion, u32) -> u8,
    {
        let available = BUFFER_SIZE - self.end;
        if count > available {
            return Err(CommitError {
                requested: count,
                available,
            });
        }
        self.end += count;

        let mut rejections = Rejections::default();
        self.inspect(&mut extra_crc, &mut rejections);
        if !self.ready && self.buffered_len() == BUFFER_SIZE && BUFFER_SIZE != 0 {
            rejections.buffer_too_small += 1;
            self.reject_candidate();
            self.inspect(&mut extra_crc, &mut rejections);
        }

        Ok(DecodeProgress {
            consumed: count,
            rejections,
            status: if self.ready {
                DecodeStatus::FrameReady
            } else {
                DecodeStatus::NeedMore
            },
        })
    }

    /// Return the currently ready frame.
    #[must_use]
    pub fn frame(&self) -> Option<FrameRef<'_>> {
        if !self.ready {
            return None;
        }
        Some(FrameRef {
            bytes: &self.buffer[self.start..self.start + self.frame_len],
            version: marker_version(self.buffer[self.start])
                .expect("ready frames have a valid marker"),
        })
    }

    /// Release the ready frame while retaining any bytes buffered after it.
    ///
    /// Returns `false` when no frame was ready. After advancing, call
    /// [`Self::push`] again; an empty input slice is sufficient when a complete
    /// nested candidate was already buffered during resynchronization.
    pub fn advance(&mut self) -> bool {
        if !self.ready {
            return false;
        }

        self.start += self.frame_len;
        self.frame_len = 0;
        self.ready = false;
        self.seek_marker();
        true
    }

    /// Discard the ready frame or any incomplete candidate.
    pub fn reset(&mut self) {
        self.start = 0;
        self.end = 0;
        self.frame_len = 0;
        self.ready = false;
    }

    /// Number of bytes currently retained by the decoder.
    #[must_use]
    pub const fn buffered_len(&self) -> usize {
        self.end - self.start
    }

    /// Bytes retained for the current frame candidate.
    #[must_use]
    fn candidate(&self) -> &[u8] {
        &self.buffer[self.start..self.end]
    }

    /// Configured read-ahead capacity.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        BUFFER_SIZE
    }

    fn inspect<F>(&mut self, extra_crc: &mut F, rejections: &mut Rejections)
    where
        F: FnMut(FrameVersion, u32) -> u8,
    {
        if self.ready {
            return;
        }

        loop {
            let Some(version) = self.candidate_version() else {
                return;
            };

            let header_needed = match version {
                FrameVersion::V1 => 2,
                FrameVersion::V2 => 3,
            };
            if self.buffered_len() < header_needed {
                return;
            }

            if version == FrameVersion::V2 && self.candidate()[2] & !V2_SUPPORTED_FLAGS != 0 {
                rejections.unsupported_flags += 1;
                self.reject_candidate();
                continue;
            }

            let candidate_len = self.candidate_frame_len(version);
            if candidate_len > BUFFER_SIZE {
                rejections.buffer_too_small += 1;
                self.reject_candidate();
                continue;
            }
            if self.buffered_len() < candidate_len {
                return;
            }

            let message_id = self.candidate_message_id(version);
            let checksum_start = self.checksum_start(version);
            let candidate = self.candidate();
            let expected =
                u16::from_le_bytes([candidate[checksum_start], candidate[checksum_start + 1]]);
            let actual = mavlink_crc::calculate(
                &candidate[1..checksum_start],
                extra_crc(version, message_id),
            );

            if actual == expected {
                self.frame_len = candidate_len;
                self.ready = true;
                return;
            }

            rejections.invalid_checksum += 1;
            self.reject_candidate();
        }
    }

    fn candidate_version(&mut self) -> Option<FrameVersion> {
        if self.buffered_len() == 0 {
            return None;
        }
        if let Some(version) = marker_version(self.buffer[self.start]) {
            return Some(version);
        }
        self.seek_marker();
        if self.buffered_len() == 0 {
            None
        } else {
            marker_version(self.buffer[self.start])
        }
    }

    fn next_target_len(&self) -> usize {
        match marker_version(self.buffer[self.start])
            .expect("buffered candidates start with a marker")
        {
            FrameVersion::V1 if self.buffered_len() < 2 => 2,
            FrameVersion::V2 if self.buffered_len() < 3 => 3,
            version => self.candidate_frame_len(version),
        }
    }

    fn candidate_frame_len(&self, version: FrameVersion) -> usize {
        let candidate = self.candidate();
        let payload_len = usize::from(candidate[1]);
        match version {
            FrameVersion::V1 => 1 + V1_HEADER_LEN + payload_len + CHECKSUM_LEN,
            FrameVersion::V2 => {
                let signature_len = if candidate[2] & V2_SIGNED_FLAG == 0 {
                    0
                } else {
                    SIGNATURE_LEN
                };
                1 + V2_HEADER_LEN + payload_len + CHECKSUM_LEN + signature_len
            }
        }
    }

    fn checksum_start(&self, version: FrameVersion) -> usize {
        let payload_len = usize::from(self.candidate()[1]);
        match version {
            FrameVersion::V1 => 1 + V1_HEADER_LEN + payload_len,
            FrameVersion::V2 => 1 + V2_HEADER_LEN + payload_len,
        }
    }

    fn candidate_message_id(&self, version: FrameVersion) -> u32 {
        let candidate = self.candidate();
        match version {
            FrameVersion::V1 => u32::from(candidate[5]),
            FrameVersion::V2 => u32::from_le_bytes([candidate[7], candidate[8], candidate[9], 0]),
        }
    }

    fn reject_candidate(&mut self) {
        let next_marker =
            find_marker(&self.buffer[self.start + 1..self.end]).map(|offset| offset + 1);
        if let Some(next_marker) = next_marker {
            self.start += next_marker;
        } else {
            self.start = 0;
            self.end = 0;
        }
        self.frame_len = 0;
        self.ready = false;
    }

    fn seek_marker(&mut self) {
        let marker = find_marker(&self.buffer[self.start..self.end]);
        if let Some(marker) = marker {
            self.start += marker;
        } else {
            self.start = 0;
            self.end = 0;
        }
    }

    fn compact(&mut self) {
        if self.start == 0 {
            return;
        }

        self.buffer.copy_within(self.start..self.end, 0);
        self.end -= self.start;
        self.start = 0;
    }
}

#[inline(always)]
fn find_marker(bytes: &[u8]) -> Option<usize> {
    let (&first, remaining) = bytes.split_first()?;
    if first == V1_STX || first == V2_STX {
        return Some(0);
    }

    if bytes.len() < VECTORIZED_MARKER_SEARCH_MINIMUM {
        return remaining
            .iter()
            .position(|&byte| byte == V1_STX || byte == V2_STX)
            .map(|offset| offset + 1);
    }

    if let Some(offset) = remaining[..SCALAR_MARKER_SEARCH_PREFIX - 1]
        .iter()
        .position(|&byte| byte == V1_STX || byte == V2_STX)
    {
        return Some(offset + 1);
    }

    memchr::memchr2(
        V1_STX,
        V2_STX,
        &remaining[SCALAR_MARKER_SEARCH_PREFIX - 1..],
    )
    .map(|offset| offset + SCALAR_MARKER_SEARCH_PREFIX)
}

const fn marker_version(marker: u8) -> Option<FrameVersion> {
    match marker {
        V1_STX => Some(FrameVersion::V1),
        V2_STX => Some(FrameVersion::V2),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::vec;
    use std::vec::Vec;

    const EXTRA_CRC: u8 = 77;

    fn v1_frame(message_id: u8, payload: &[u8]) -> Vec<u8> {
        let mut frame = vec![V1_STX, payload.len() as u8, 7, 42, 9, message_id];
        frame.extend_from_slice(payload);
        let crc = mavlink_crc::calculate(&frame[1..], EXTRA_CRC);
        frame.extend_from_slice(&crc.to_le_bytes());
        frame
    }

    fn v2_frame(message_id: u32, payload: &[u8], signed: bool) -> Vec<u8> {
        let id = message_id.to_le_bytes();
        let mut frame = vec![
            V2_STX,
            payload.len() as u8,
            u8::from(signed),
            0,
            8,
            43,
            10,
            id[0],
            id[1],
            id[2],
        ];
        frame.extend_from_slice(payload);
        let crc = mavlink_crc::calculate(&frame[1..], EXTRA_CRC);
        frame.extend_from_slice(&crc.to_le_bytes());
        if signed {
            frame.extend(0_u8..SIGNATURE_LEN as u8);
        }
        frame
    }

    fn extra(_: FrameVersion, _: u32) -> u8 {
        EXTRA_CRC
    }

    #[test]
    fn marker_search_matches_scalar_reference() {
        let mut bytes = [0x55; 512];
        for length in 0..=bytes.len() {
            assert_eq!(find_marker(&bytes[..length]), None);
            for position in 0..length {
                for marker in [V1_STX, V2_STX] {
                    bytes[position] = marker;
                    assert_eq!(find_marker(&bytes[..length]), Some(position));
                    bytes[position] = 0x55;
                }
            }
        }
    }

    fn feed_chunks<const BUFFER_SIZE: usize>(
        decoder: &mut FrameDecoder<BUFFER_SIZE>,
        bytes: &[u8],
        chunk_len: usize,
    ) -> Vec<Vec<u8>> {
        let mut frames = Vec::new();
        let mut offset = 0;
        while offset < bytes.len() {
            let chunk_end = (offset + chunk_len).min(bytes.len());
            let mut chunk_offset = offset;
            while chunk_offset < chunk_end {
                let progress = decoder.push(&bytes[chunk_offset..chunk_end], extra);
                chunk_offset += progress.consumed;
                if progress.status == DecodeStatus::FrameReady {
                    frames.push(decoder.frame().unwrap().bytes().to_vec());
                    assert!(decoder.advance());
                } else {
                    assert_eq!(chunk_offset, chunk_end);
                }
            }
            offset = chunk_end;
        }

        loop {
            let progress = decoder.push(&[], extra);
            if progress.status != DecodeStatus::FrameReady {
                break;
            }
            frames.push(decoder.frame().unwrap().bytes().to_vec());
            assert!(decoder.advance());
        }
        frames
    }

    #[test]
    fn accepts_v1_one_byte_at_a_time() {
        let expected = v1_frame(33, &[1, 2, 3, 4, 5]);
        let mut decoder = FrameDecoder::<MAX_FRAME_LEN>::new();
        assert_eq!(feed_chunks(&mut decoder, &expected, 1), [expected]);
    }

    #[test]
    fn accepts_v2_in_arbitrary_chunks() {
        let expected = v2_frame(0x00ab_cdef, &[1, 2, 3, 4, 5, 6, 7], false);
        for chunk_len in 1..=expected.len() + 2 {
            let mut decoder = FrameDecoder::<MAX_FRAME_LEN>::new();
            let frames = feed_chunks(&mut decoder, &expected, chunk_len);
            assert_eq!(frames.len(), 1);
            assert_eq!(frames[0], expected);
        }
    }

    #[test]
    fn accepts_maximum_v1_and_signed_v2_frames() {
        let v1 = v1_frame(1, &[0x55; 255]);
        let v2 = v2_frame(0x00ff_ffff, &[0xaa; 255], true);
        assert_eq!(v1.len(), 263);
        assert_eq!(v2.len(), MAX_FRAME_LEN);

        let mut decoder = FrameDecoder::<MAX_FRAME_LEN>::new();
        let mut stream = v1.clone();
        stream.extend_from_slice(&v2);
        assert_eq!(feed_chunks(&mut decoder, &stream, 17), [v1, v2]);
    }

    #[test]
    fn exposes_protocol_neutral_metadata_and_signature() {
        let expected = v2_frame(0x0003_0201, &[9, 8, 7], true);
        let mut decoder = FrameDecoder::<MAX_FRAME_LEN>::new();
        let progress = decoder.push(&expected, extra);
        assert_eq!(progress.status, DecodeStatus::FrameReady);

        let frame = decoder.frame().unwrap();
        assert_eq!(frame.version(), FrameVersion::V2);
        assert_eq!(frame.message_id(), 0x0003_0201);
        assert_eq!(frame.sequence(), 8);
        assert_eq!(frame.system_id(), 43);
        assert_eq!(frame.component_id(), 10);
        assert_eq!(frame.payload(), [9, 8, 7]);
        assert_eq!(frame.incompatibility_flags(), Some(V2_SIGNED_FLAG));
        assert!(frame.is_signed());
        assert_eq!(frame.signature(), Some(&(0_u8..13).collect::<Vec<_>>()[..]));
        assert_eq!(
            frame.checksum(),
            u16::from_le_bytes([expected[13], expected[14]])
        );
    }

    #[test]
    fn skips_garbage_and_decodes_adjacent_versions() {
        let v1 = v1_frame(2, &[1, 2]);
        let v2 = v2_frame(300, &[3, 4, 5], false);
        let mut stream = vec![0, 1, 2, 3, 4];
        stream.extend_from_slice(&v1);
        stream.extend_from_slice(&v2);

        let mut decoder = FrameDecoder::<MAX_FRAME_LEN>::new();
        assert_eq!(feed_chunks(&mut decoder, &stream, stream.len()), [v1, v2]);
    }

    #[test]
    fn invalid_checksum_rescans_from_the_next_byte() {
        let nested = v2_frame(44, &[1, 2, 3], false);

        // A false outer candidate contains an entire valid frame. Once the
        // outer checksum fails, byte-wise resynchronization must retain and
        // recognize the nested marker rather than discard the outer body.
        let mut corrupt = vec![V2_STX, nested.len() as u8, 0, 0, 1, 2, 3, 4, 0, 0];
        corrupt.extend_from_slice(&nested);
        corrupt.extend_from_slice(&[0, 0]);

        let mut decoder = FrameDecoder::<MAX_FRAME_LEN>::new();
        let progress = decoder.push(&corrupt, extra);
        assert_eq!(progress.rejections.invalid_checksum, 1);
        assert_eq!(progress.status, DecodeStatus::FrameReady);
        assert_eq!(decoder.frame().unwrap().bytes(), nested);
        assert_eq!(progress.consumed, corrupt.len());
    }

    #[test]
    fn advancing_a_recovered_frame_retains_buffered_suffix() {
        let nested = v1_frame(11, &[4]);
        let following = v1_frame(12, &[5]);

        // Both valid frames reside inside a larger corrupt candidate. The
        // decoder first recovers `nested`, then `advance` must retain
        // `following` without asking the caller to replay any bytes.
        let outer_payload_len = nested.len() + following.len();
        let mut corrupt = vec![V2_STX, outer_payload_len as u8, 0, 0, 1, 2, 3, 4, 0, 0];
        corrupt.extend_from_slice(&nested);
        corrupt.extend_from_slice(&following);
        corrupt.extend_from_slice(&[0, 0]);

        let mut decoder = FrameDecoder::<MAX_FRAME_LEN>::new();
        let progress = decoder.push(&corrupt, extra);
        assert_eq!(progress.status, DecodeStatus::FrameReady);
        assert_eq!(decoder.frame().unwrap().bytes(), nested);
        assert!(decoder.advance());

        let progress = decoder.push(&[], extra);
        assert_eq!(progress.consumed, 0);
        assert_eq!(progress.status, DecodeStatus::FrameReady);
        assert_eq!(decoder.frame().unwrap().bytes(), following);
    }

    #[test]
    fn unsupported_flags_rescan_from_the_next_byte() {
        let valid = v1_frame(17, &[5, 6]);
        let mut stream = vec![V2_STX, 200, 0x80, 12, 13];
        stream.extend_from_slice(&valid);

        let mut decoder = FrameDecoder::<MAX_FRAME_LEN>::new();
        let progress = decoder.push(&stream, extra);
        assert_eq!(progress.rejections.unsupported_flags, 1);
        assert_eq!(progress.status, DecodeStatus::FrameReady);
        assert_eq!(decoder.frame().unwrap().bytes(), valid);
    }

    #[test]
    fn wrong_extra_crc_rejects_an_otherwise_complete_frame() {
        let expected = v1_frame(1, &[9, 9, 9]);
        let mut decoder = FrameDecoder::<MAX_FRAME_LEN>::new();
        let progress = decoder.push(&expected, |_, _| EXTRA_CRC.wrapping_add(1));
        assert_eq!(progress.status, DecodeStatus::NeedMore);
        assert_eq!(progress.rejections.invalid_checksum, 1);
        assert_eq!(progress.rejections.total(), 1);
        assert!(decoder.frame().is_none());
    }

    #[test]
    fn ready_frame_must_be_advanced_before_more_input_is_consumed() {
        let first = v1_frame(1, &[1]);
        let second = v1_frame(2, &[2]);
        let mut decoder = FrameDecoder::<MAX_FRAME_LEN>::new();
        let mut checksum_callbacks = 0;
        let first_progress = decoder.push(&first, |_, _| {
            checksum_callbacks += 1;
            EXTRA_CRC
        });
        assert_eq!(first_progress.status, DecodeStatus::FrameReady);
        assert_eq!(checksum_callbacks, 1);

        let blocked = decoder.push(&second, |_, _| {
            checksum_callbacks += 1;
            EXTRA_CRC
        });
        assert_eq!(blocked.consumed, 0);
        assert_eq!(blocked.status, DecodeStatus::FrameReady);
        assert_eq!(checksum_callbacks, 1);

        let recommitted = decoder
            .commit(0, |_, _| {
                checksum_callbacks += 1;
                EXTRA_CRC
            })
            .unwrap();
        assert_eq!(recommitted.status, DecodeStatus::FrameReady);
        assert_eq!(checksum_callbacks, 1);
        assert!(decoder.advance());

        let second_progress = decoder.push(&second, extra);
        assert_eq!(second_progress.status, DecodeStatus::FrameReady);
        assert_eq!(decoder.frame().unwrap().message_id(), 2);
    }

    #[test]
    fn direct_buffer_fill_retains_multiple_frames() {
        let first = v1_frame(1, &[1, 2]);
        let second = v2_frame(2, &[3, 4], false);
        let mut stream = first.clone();
        stream.extend_from_slice(&second);

        let mut decoder = FrameDecoder::<64>::new();
        assert_eq!(decoder.capacity(), 64);
        let count = {
            let spare = decoder.spare_capacity_mut();
            spare[..stream.len()].copy_from_slice(&stream);
            stream.len()
        };
        let progress = decoder.commit(count, extra).unwrap();
        assert_eq!(progress.consumed, stream.len());
        assert_eq!(progress.status, DecodeStatus::FrameReady);
        assert_eq!(decoder.frame().unwrap().bytes(), first);

        assert!(decoder.advance());
        let progress = decoder.commit(0, extra).unwrap();
        assert_eq!(progress.status, DecodeStatus::FrameReady);
        assert_eq!(decoder.frame().unwrap().bytes(), second);
    }

    #[test]
    fn advancing_many_adjacent_frames_defers_compaction() {
        let frames: Vec<_> = (0..35)
            .map(|message_id| v1_frame(message_id, &[]))
            .collect();
        let stream: Vec<_> = frames.iter().flatten().copied().collect();
        assert_eq!(stream.len(), MAX_FRAME_LEN);

        let mut decoder = FrameDecoder::<MAX_FRAME_LEN>::new();
        decoder.spare_capacity_mut().copy_from_slice(&stream);
        assert_eq!(
            decoder.commit(stream.len(), extra).unwrap().status,
            DecodeStatus::FrameReady
        );

        for (index, expected) in frames.iter().enumerate() {
            assert_eq!(decoder.frame().unwrap().bytes(), expected);
            assert_eq!(decoder.end, stream.len());
            assert!(decoder.advance());

            if index + 1 == frames.len() {
                assert_eq!((decoder.start, decoder.end), (0, 0));
            } else {
                assert_eq!(decoder.start, expected.len() * (index + 1));
                assert_eq!(
                    decoder.commit(0, extra).unwrap().status,
                    DecodeStatus::FrameReady
                );
            }
        }
    }

    #[test]
    fn exhausted_tail_compacts_once_and_preserves_ready_frame_and_suffix() {
        let first = v1_frame(1, &[]);
        let second = v1_frame(2, &[]);
        let third = v1_frame(3, &[]);
        let fourth = v1_frame(4, &[]);
        let mut initial = first.clone();
        initial.extend_from_slice(&second);
        initial.extend_from_slice(&third);

        let mut decoder = FrameDecoder::<24>::new();
        decoder.spare_capacity_mut().copy_from_slice(&initial);
        decoder.commit(initial.len(), extra).unwrap();
        assert_eq!(decoder.frame().unwrap().bytes(), first);
        assert!(decoder.advance());
        decoder.commit(0, extra).unwrap();
        assert_eq!(decoder.frame().unwrap().bytes(), second);
        assert_eq!((decoder.start, decoder.end), (8, 24));

        let count = {
            let spare = decoder.spare_capacity_mut();
            assert_eq!(spare.len(), fourth.len());
            spare.copy_from_slice(&fourth);
            fourth.len()
        };
        assert_eq!((decoder.start, decoder.end), (0, 16));
        assert_eq!(
            decoder.commit(count, extra).unwrap().status,
            DecodeStatus::FrameReady
        );
        assert_eq!(decoder.frame().unwrap().bytes(), second);

        for expected in [&third, &fourth] {
            assert!(decoder.advance());
            assert_eq!(
                decoder.commit(0, extra).unwrap().status,
                DecodeStatus::FrameReady
            );
            assert_eq!(decoder.frame().unwrap().bytes(), expected);
        }
    }

    #[test]
    fn too_small_capacity_rejects_without_panicking_and_recovers() {
        let oversized = v2_frame(1, &[0x55; 20], false);
        let following = v1_frame(2, &[7]);
        let mut stream = oversized;
        stream.extend_from_slice(&following);

        let mut decoder = FrameDecoder::<12>::new();
        let progress = decoder.push(&stream, extra);
        assert!(progress.rejections.buffer_too_small >= 1);
        assert_eq!(progress.status, DecodeStatus::FrameReady);
        assert_eq!(decoder.frame().unwrap().bytes(), following);
    }

    #[test]
    fn zero_capacity_and_overcommit_are_reported_safely() {
        let frame = v1_frame(1, &[1]);
        let mut zero = FrameDecoder::<0>::new();
        let progress = zero.push(&frame, extra);
        assert_eq!(progress.consumed, frame.len());
        assert_eq!(progress.status, DecodeStatus::NeedMore);
        assert_eq!(progress.rejections.buffer_too_small, 1);
        assert!(zero.spare_capacity_mut().is_empty());
        assert_eq!(zero.commit(1, extra).unwrap_err().available, 0);

        let mut decoder = FrameDecoder::<8>::new();
        assert_eq!(
            decoder.commit(9, extra),
            Err(CommitError {
                requested: 9,
                available: 8
            })
        );
        assert_eq!(decoder.buffered_len(), 0);
    }

    #[test]
    fn tiny_capacities_reject_headers_without_indexing_past_the_buffer() {
        let frame = v2_frame(1, &[], false);

        let mut one = FrameDecoder::<1>::new();
        let progress = one.push(&frame, extra);
        assert_eq!(progress.consumed, frame.len());
        assert_eq!(progress.status, DecodeStatus::NeedMore);
        assert!(progress.rejections.buffer_too_small >= 1);

        let mut two = FrameDecoder::<2>::new();
        let progress = two.push(&frame, extra);
        assert_eq!(progress.consumed, frame.len());
        assert_eq!(progress.status, DecodeStatus::NeedMore);
        assert!(progress.rejections.buffer_too_small >= 1);
    }

    #[test]
    fn reset_discards_partial_candidate() {
        let expected = v2_frame(8, &[1, 2, 3], false);
        let mut decoder = FrameDecoder::<MAX_FRAME_LEN>::new();
        assert_eq!(
            decoder.push(&expected[..5], extra).status,
            DecodeStatus::NeedMore
        );
        assert_eq!(decoder.buffered_len(), 5);
        decoder.reset();
        assert_eq!(decoder.buffered_len(), 0);
        assert!(!decoder.advance());
    }
}
