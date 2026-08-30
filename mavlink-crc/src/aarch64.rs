//! PMULL backend for AArch64.
//!
//! This is a fixed-polynomial folding implementation for CRC-16/MCRF4XX. The
//! vector helpers expose only the carry-less multiply operations required by
//! this algorithm.

use core::arch::aarch64::*;

use crate::scalar;

const FOLD_16_LOW: u64 = 0x0000_0000_0001_89ae;
const FOLD_16_HIGH: u64 = 0x0000_0000_0000_8e10;
const FOLD_32_LOW: u64 = 0x0000_0000_0000_5b44;
const FOLD_32_HIGH: u64 = 0x0000_0000_0000_7762;
const FOLD_48_LOW: u64 = 0x0000_0000_0000_0e3a;
const FOLD_48_HIGH: u64 = 0x0000_0000_0000_4d7a;
const FOLD_64_LOW: u64 = 0x0000_0000_0001_4ff2;
const FOLD_64_HIGH: u64 = 0x0000_0000_0001_9a3c;
const FOLD_8: u64 = 0x0000_0000_0001_89ae;
const FOLD_4: u64 = 0x0000_0000_0001_14aa;
const BARRETT_MU: u64 = 0x0000_0001_1c58_1911;
const BARRETT_POLYNOMIAL: u64 = 0x0000_0000_0001_0811;

/// Uses the accelerated backend when the required CPU feature is available.
///
/// Runtime detection is available with `std`. In `no_std` builds, AES must be
/// enabled as a target feature when compiling the application. On AArch64 that
/// feature includes the PMULL instruction used by this backend.
#[inline]
pub(crate) fn update_if_supported(initial: u16, bytes: &[u8]) -> Option<u16> {
    if !pmull_available() {
        return None;
    }

    // SAFETY: `pmull_available` establishes the target feature required by
    // `update_pmull` before control reaches this call.
    Some(unsafe { update_pmull(initial, bytes) })
}

#[cfg(feature = "std")]
#[inline]
fn pmull_available() -> bool {
    std::arch::is_aarch64_feature_detected!("aes")
}

#[cfg(not(feature = "std"))]
#[inline]
const fn pmull_available() -> bool {
    cfg!(target_feature = "aes")
}

/// # Safety
///
/// The caller must guarantee AES/PMULL support. AArch64 always supplies NEON.
/// Every vector load is from a complete 16-byte chunk.
#[target_feature(enable = "aes")]
unsafe fn update_pmull(initial: u16, bytes: &[u8]) -> u16 {
    debug_assert!(bytes.len() >= 16);

    if bytes.len() >= 128 {
        // SAFETY: the caller guarantees AES/PMULL, and this branch checks that
        // two complete four-lane groups are available.
        return unsafe { update_four_lanes(initial, bytes) };
    }

    // SAFETY: NEON is part of the AArch64 baseline.
    let coefficient = unsafe { vector(FOLD_16_LOW, FOLD_16_HIGH) };
    let mut chunks = bytes.chunks_exact(16);
    let first = chunks.next().expect("length was checked by the caller");

    // SAFETY: `first` is a complete 16-byte chunk.
    let first_vector = unsafe { vld1q_u8(first.as_ptr()) };
    let initial_vector = vreinterpretq_u8_u16(vsetq_lane_u16(initial, vdupq_n_u16(0), 0));
    let mut state = veorq_u8(first_vector, initial_vector);

    for chunk in &mut chunks {
        // SAFETY: this function has AES/PMULL enabled.
        let high = unsafe { multiply_10(state, coefficient) };
        // SAFETY: this function has AES/PMULL enabled.
        let low = unsafe { multiply_01(state, coefficient) };
        // SAFETY: `chunks_exact(16)` only yields complete 16-byte chunks.
        let next = unsafe { vld1q_u8(chunk.as_ptr()) };
        state = veorq_u8(veorq_u8(high, low), next);
    }

    // SAFETY: this function has AES/PMULL enabled.
    let mut crc = unsafe { reduce(state) };
    for &byte in chunks.remainder() {
        crc = scalar::update_byte(crc, byte);
    }
    crc
}

#[inline]
#[target_feature(enable = "aes")]
unsafe fn fold(state: uint8x16_t, coefficient: uint8x16_t, next: uint8x16_t) -> uint8x16_t {
    // SAFETY: this function has AES/PMULL enabled.
    let high = unsafe { multiply_10(state, coefficient) };
    // SAFETY: this function has AES/PMULL enabled.
    let low = unsafe { multiply_01(state, coefficient) };
    veorq_u8(veorq_u8(high, low), next)
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn load(chunk: &[u8]) -> uint8x16_t {
    debug_assert_eq!(chunk.len(), 16);
    // SAFETY: the caller supplies one complete chunk.
    unsafe { vld1q_u8(chunk.as_ptr()) }
}

#[target_feature(enable = "aes")]
unsafe fn update_four_lanes(initial: u16, bytes: &[u8]) -> u16 {
    let mut chunks = bytes.chunks_exact(16);
    let first = chunks.next().expect("length was checked by the caller");
    let second = chunks.next().expect("length was checked by the caller");
    let third = chunks.next().expect("length was checked by the caller");
    let fourth = chunks.next().expect("length was checked by the caller");

    let initial_vector = vreinterpretq_u8_u16(vsetq_lane_u16(initial, vdupq_n_u16(0), 0));
    // SAFETY: all four inputs are complete 16-byte chunks. NEON is part of the
    // AArch64 baseline.
    let (mut lane0, mut lane1, mut lane2, mut lane3) = unsafe {
        (
            veorq_u8(load(first), initial_vector),
            load(second),
            load(third),
            load(fourth),
        )
    };
    // SAFETY: NEON is part of the AArch64 baseline.
    let fold64 = unsafe { vector(FOLD_64_LOW, FOLD_64_HIGH) };

    while chunks.len() >= 4 {
        let next0 = chunks.next().expect("four chunks remain");
        let next1 = chunks.next().expect("three chunks remain");
        let next2 = chunks.next().expect("two chunks remain");
        let next3 = chunks.next().expect("one chunk remains");
        // SAFETY: every input is a complete 16-byte chunk and this function
        // enables AES/PMULL.
        (lane0, lane1, lane2, lane3) = unsafe {
            (
                fold(lane0, fold64, load(next0)),
                fold(lane1, fold64, load(next1)),
                fold(lane2, fold64, load(next2)),
                fold(lane3, fold64, load(next3)),
            )
        };
    }

    // Collapse the independent streams in their original byte order.
    // SAFETY: NEON is part of the AArch64 baseline.
    let fold48 = unsafe { vector(FOLD_48_LOW, FOLD_48_HIGH) };
    // SAFETY: NEON is part of the AArch64 baseline.
    let fold32 = unsafe { vector(FOLD_32_LOW, FOLD_32_HIGH) };
    // SAFETY: NEON is part of the AArch64 baseline.
    let fold16 = unsafe { vector(FOLD_16_LOW, FOLD_16_HIGH) };
    // SAFETY: this function enables AES/PMULL.
    let mut state = unsafe {
        let state = fold(lane0, fold48, lane3);
        let state = fold(lane1, fold32, state);
        fold(lane2, fold16, state)
    };

    for chunk in &mut chunks {
        // SAFETY: `chunk` is complete and this function enables AES/PMULL.
        state = unsafe { fold(state, fold16, load(chunk)) };
    }

    // SAFETY: this function has AES/PMULL enabled.
    let mut crc = unsafe { reduce(state) };
    for &byte in chunks.remainder() {
        crc = scalar::update_byte(crc, byte);
    }
    crc
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn vector(low: u64, high: u64) -> uint8x16_t {
    let vector = vsetq_lane_u64(low, vdupq_n_u64(0), 0);
    vreinterpretq_u8_u64(vsetq_lane_u64(high, vector, 1))
}

#[inline]
#[target_feature(enable = "aes")]
unsafe fn multiply_00(left: uint8x16_t, right: uint8x16_t) -> uint8x16_t {
    vreinterpretq_u8_p128(vmull_p64(
        vgetq_lane_p64(vreinterpretq_p64_u8(left), 0),
        vgetq_lane_p64(vreinterpretq_p64_u8(right), 0),
    ))
}

#[inline]
#[target_feature(enable = "aes")]
unsafe fn multiply_01(left: uint8x16_t, right: uint8x16_t) -> uint8x16_t {
    vreinterpretq_u8_p128(vmull_p64(
        vgetq_lane_p64(vreinterpretq_p64_u8(left), 1),
        vgetq_lane_p64(vreinterpretq_p64_u8(right), 0),
    ))
}

#[inline]
#[target_feature(enable = "aes")]
unsafe fn multiply_10(left: uint8x16_t, right: uint8x16_t) -> uint8x16_t {
    vreinterpretq_u8_p128(vmull_p64(
        vgetq_lane_p64(vreinterpretq_p64_u8(left), 0),
        vgetq_lane_p64(vreinterpretq_p64_u8(right), 1),
    ))
}

#[inline]
#[target_feature(enable = "aes")]
unsafe fn multiply_11(left: uint8x16_t, right: uint8x16_t) -> uint8x16_t {
    vreinterpretq_u8_p128(vmull_p64(
        vgetq_lane_p64(vreinterpretq_p64_u8(left), 1),
        vgetq_lane_p64(vreinterpretq_p64_u8(right), 1),
    ))
}

#[inline]
#[target_feature(enable = "aes")]
unsafe fn reduce(mut state: uint8x16_t) -> u16 {
    // SAFETY: this function has AES/PMULL and NEON enabled.
    state = veorq_u8(
        unsafe { multiply_00(state, vector(FOLD_8, 0)) },
        vextq_u8(state, vdupq_n_u8(0), 8),
    );

    // SAFETY: this function has NEON enabled.
    let mask = unsafe { vector(0xffff_ffff_0000_0000, u64::MAX) };
    let low_word = vgetq_lane_u32(vreinterpretq_u32_u8(state), 0);
    let shifted = vreinterpretq_u8_u32(vsetq_lane_u32(low_word, vdupq_n_u32(0), 3));
    state = veorq_u8(
        // SAFETY: this function has AES/PMULL and NEON enabled.
        unsafe { multiply_11(shifted, vector(0, FOLD_4)) },
        vandq_u8(state, mask),
    );

    // SAFETY: this function has NEON enabled.
    let mu_polynomial = unsafe { vector(BARRETT_MU, BARRETT_POLYNOMIAL) };
    // SAFETY: this function has AES/PMULL enabled.
    let quotient = unsafe { multiply_00(state, mu_polynomial) };
    // SAFETY: this function has AES/PMULL enabled.
    let product = unsafe { multiply_10(quotient, mu_polynomial) };
    let result = vreinterpretq_u64_u8(veorq_u8(state, product));
    vgetq_lane_u64(result, 1) as u16
}
