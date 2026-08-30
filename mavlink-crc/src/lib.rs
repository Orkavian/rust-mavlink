//! Fast CRC-16/MCRF4XX for MAVLink.
//!
//! This crate intentionally implements one algorithm. It is `no_std`, performs
//! no allocation, installs no panic handler, and has a safe portable fallback.

#![no_std]
#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(feature = "std")]
extern crate std;

mod backend;
mod scalar;

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), not(miri)))]
mod x86;

/// CRC-16/MCRF4XX initial value used by MAVLink.
pub const INITIAL: u16 = 0xffff;

const LONG_MESSAGE_MINIMUM: usize = 64;

/// Calculates the MAVLink CRC over `bytes`, followed by `extra_crc`.
#[must_use]
#[inline]
pub fn calculate(bytes: &[u8], extra_crc: u8) -> u16 {
    if bytes.len() == 5 {
        return scalar::update_5_and_extra(INITIAL, bytes, extra_crc);
    }
    if bytes.len() == 9 {
        return scalar::update_9_and_extra(INITIAL, bytes, extra_crc);
    }
    if bytes.len() == 14 {
        return scalar::update_14_and_extra(INITIAL, bytes, extra_crc);
    }
    if bytes.len() < 8 {
        return scalar::update_small_and_extra(INITIAL, bytes, extra_crc);
    }
    if bytes.len() < 15 {
        return scalar::update_tail_and_extra(INITIAL, bytes, extra_crc);
    }

    let remainder = bytes.len() % 16;
    if remainder == 15 {
        let final_block = bytes.len() - remainder;
        let crc = update(INITIAL, &bytes[..final_block]);
        return scalar::update_15_and_extra(crc, &bytes[final_block..], extra_crc);
    }
    if bytes.len() >= LONG_MESSAGE_MINIMUM && remainder >= 8 {
        let final_block = bytes.len() - remainder;
        let crc = update(INITIAL, &bytes[..final_block]);
        return scalar::update_tail_and_extra(crc, &bytes[final_block..], extra_crc);
    }

    scalar::update_byte(update(INITIAL, bytes), extra_crc)
}

/// Incremental CRC-16/MCRF4XX state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Digest {
    value: u16,
}

impl Digest {
    /// Creates a new MAVLink CRC digest.
    #[must_use]
    pub const fn new() -> Self {
        Self { value: INITIAL }
    }

    /// Adds bytes to this digest.
    #[inline]
    pub fn update(&mut self, bytes: &[u8]) {
        self.value = update(self.value, bytes);
    }

    /// Adds one byte to this digest.
    #[inline]
    pub fn update_byte(&mut self, byte: u8) {
        self.value = scalar::update_byte(self.value, byte);
    }

    /// Returns the current CRC value.
    #[must_use]
    pub const fn finalize(self) -> u16 {
        self.value
    }
}

impl Default for Digest {
    fn default() -> Self {
        Self::new()
    }
}

#[inline]
fn update(initial: u16, bytes: &[u8]) -> u16 {
    if bytes.len() < LONG_MESSAGE_MINIMUM {
        scalar::update(initial, bytes)
    } else {
        backend::update(initial, bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const POLYNOMIAL: u16 = 0x8408;

    fn reference_byte(mut crc: u16, byte: u8) -> u16 {
        crc ^= u16::from(byte);
        for _ in 0..8 {
            crc = (crc >> 1) ^ ((crc & 1) * POLYNOMIAL);
        }
        crc
    }

    fn reference(initial: u16, bytes: &[u8]) -> u16 {
        bytes
            .iter()
            .fold(initial, |crc, &byte| reference_byte(crc, byte))
    }

    fn data() -> [u8; 266] {
        let mut data = [0; 266];
        for (index, byte) in data.iter_mut().enumerate() {
            *byte = index.wrapping_mul(73).wrapping_add(41) as u8;
        }
        data
    }

    #[test]
    fn standard_check_value() {
        assert_eq!(reference(INITIAL, b"123456789"), 0x6f91);
        let mut digest = Digest::new();
        digest.update(b"123456789");
        assert_eq!(digest.finalize(), 0x6f91);
    }

    #[test]
    fn every_byte_matches_independent_bitwise_reference() {
        for initial in [0, 1, 0x00ff, 0x0100, 0x5555, 0xaaaa, INITIAL] {
            for byte in u8::MIN..=u8::MAX {
                assert_eq!(
                    scalar::update_byte(initial, byte),
                    reference_byte(initial, byte)
                );
            }
        }
    }

    #[test]
    fn every_generated_table_entry_matches_bitwise_reference() {
        // `black_box` makes the const-capable generator execute at runtime too,
        // so coverage includes the code that creates the production tables.
        let tables = scalar::make_tables(std::hint::black_box(POLYNOMIAL));
        assert_eq!(tables, scalar::TABLES);

        for (zero_count, table) in tables.iter().enumerate() {
            for (byte, &actual) in table.iter().enumerate() {
                let mut expected = reference_byte(0, byte as u8);
                for _ in 0..zero_count {
                    expected = reference_byte(expected, 0);
                }
                assert_eq!(actual, expected, "table={zero_count}, byte={byte}");
            }
        }
    }

    #[test]
    fn every_mavlink_length_and_extra_byte_matches_reference() {
        let data = data();
        for length in 0..=265 {
            for extra in u8::MIN..=u8::MAX {
                let expected = reference_byte(reference(INITIAL, &data[..length]), extra);
                assert_eq!(calculate(&data[..length], extra), expected);
            }
        }
    }

    #[test]
    fn streaming_matches_every_split() {
        let data = data();
        for length in 0..=265 {
            let expected = reference(INITIAL, &data[..length]);
            for split in 0..=length {
                let mut digest = Digest::new();
                digest.update(&data[..split]);
                digest.update(&data[split..length]);
                assert_eq!(
                    digest.finalize(),
                    expected,
                    "length={length}, split={split}"
                );
            }
        }
    }

    #[test]
    fn scalar_chunk_boundaries_match_reference() {
        let data = data();
        for length in 0..=data.len() {
            for initial in [0, 1, 0x5555, INITIAL] {
                assert_eq!(
                    scalar::update(initial, &data[..length]),
                    reference(initial, &data[..length])
                );
            }
        }
    }

    #[test]
    fn digest_traits_and_single_byte_api() {
        let mut digest = Digest::default();
        assert_eq!(digest, Digest::new());
        digest.update_byte(42);
        let copy = digest;
        assert_eq!(copy.finalize(), reference_byte(INITIAL, 42));
        assert!(std::format!("{digest:?}").contains("Digest"));
    }
}
