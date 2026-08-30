//! PCLMULQDQ backend for x86 and x86-64.
//!
//! This is a fixed-polynomial folding implementation for CRC-16/MCRF4XX. All
//! coefficients are compile-time constants, so the hot path has no generic
//! algorithm configuration or key setup.

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::scalar;

const FOLD_16_LOW: i64 = 0x0000_0000_0001_89ae;
const FOLD_16_HIGH: i64 = 0x0000_0000_0000_8e10;
const FOLD_32_LOW: i64 = 0x0000_0000_0000_5b44;
const FOLD_32_HIGH: i64 = 0x0000_0000_0000_7762;
const FOLD_48_LOW: i64 = 0x0000_0000_0000_0e3a;
const FOLD_48_HIGH: i64 = 0x0000_0000_0000_4d7a;
const FOLD_64_LOW: i64 = 0x0000_0000_0001_4ff2;
const FOLD_64_HIGH: i64 = 0x0000_0000_0001_9a3c;
const FOLD_8: i64 = 0x0000_0000_0001_89ae;
const FOLD_4: i64 = 0x0000_0000_0001_14aa;
const BARRETT_MU: i64 = 0x0000_0001_1c58_1911;
const BARRETT_POLYNOMIAL: i64 = 0x0000_0000_0001_0811;

/// Uses the accelerated backend when the required CPU features are available.
///
/// Runtime detection is available with `std`. In `no_std` builds, the target
/// features must be enabled when compiling the application.
#[inline]
pub(crate) fn update_if_supported(initial: u16, bytes: &[u8]) -> Option<u16> {
    if !pclmulqdq_available() {
        return None;
    }

    // SAFETY: `pclmulqdq_available` establishes both target features required
    // by `update_pclmulqdq` before control reaches this call.
    Some(unsafe { update_pclmulqdq(initial, bytes) })
}

#[cfg(feature = "std")]
#[inline]
fn pclmulqdq_available() -> bool {
    std::arch::is_x86_feature_detected!("pclmulqdq") && std::arch::is_x86_feature_detected!("sse2")
}

#[cfg(not(feature = "std"))]
#[inline]
const fn pclmulqdq_available() -> bool {
    cfg!(all(target_feature = "pclmulqdq", target_feature = "sse2"))
}

/// # Safety
///
/// The caller must guarantee PCLMULQDQ and SSE2 support. All loads are made
/// from complete 16-byte chunks, so unaligned reads remain within `bytes`.
#[target_feature(enable = "pclmulqdq,sse2")]
unsafe fn update_pclmulqdq(initial: u16, bytes: &[u8]) -> u16 {
    debug_assert!(bytes.len() >= 16);

    if bytes.len() >= 128 {
        // SAFETY: the caller guarantees PCLMULQDQ and SSE2, and this branch
        // checks that two complete four-lane groups are available.
        return unsafe { update_four_lanes(initial, bytes) };
    }

    // SAFETY: this function has SSE2 enabled.
    let coefficient = unsafe { set_u64x(FOLD_16_HIGH as u64, FOLD_16_LOW as u64) };
    let mut chunks = bytes.chunks_exact(16);
    let first = chunks.next().expect("length was checked by the caller");

    // SAFETY: `first` is a complete 16-byte chunk.
    let first_vector = unsafe { _mm_loadu_si128(first.as_ptr().cast()) };
    let initial_vector = _mm_cvtsi32_si128(i32::from(initial));
    let mut state = _mm_xor_si128(first_vector, initial_vector);

    for chunk in &mut chunks {
        let high = _mm_clmulepi64_si128(state, coefficient, 0x10);
        let low = _mm_clmulepi64_si128(state, coefficient, 0x01);
        // SAFETY: `chunks_exact(16)` only yields complete 16-byte chunks.
        let next = unsafe { _mm_loadu_si128(chunk.as_ptr().cast()) };
        state = _mm_xor_si128(_mm_xor_si128(high, low), next);
    }

    // SAFETY: this function has PCLMULQDQ and SSE2 enabled.
    let mut crc = unsafe { reduce(state) };
    for &byte in chunks.remainder() {
        crc = scalar::update_byte(crc, byte);
    }
    crc
}

#[inline]
#[target_feature(enable = "pclmulqdq,sse2")]
unsafe fn fold(state: __m128i, coefficient: __m128i, next: __m128i) -> __m128i {
    let high = _mm_clmulepi64_si128(state, coefficient, 0x10);
    let low = _mm_clmulepi64_si128(state, coefficient, 0x01);
    _mm_xor_si128(_mm_xor_si128(high, low), next)
}

#[inline]
#[target_feature(enable = "sse2")]
unsafe fn load(chunk: &[u8]) -> __m128i {
    debug_assert_eq!(chunk.len(), 16);
    // SAFETY: the caller supplies one complete chunk.
    unsafe { _mm_loadu_si128(chunk.as_ptr().cast()) }
}

#[target_feature(enable = "pclmulqdq,sse2")]
unsafe fn update_four_lanes(initial: u16, bytes: &[u8]) -> u16 {
    let mut chunks = bytes.chunks_exact(16);
    let first = chunks.next().expect("length was checked by the caller");
    let second = chunks.next().expect("length was checked by the caller");
    let third = chunks.next().expect("length was checked by the caller");
    let fourth = chunks.next().expect("length was checked by the caller");

    let initial_vector = _mm_cvtsi32_si128(i32::from(initial));
    // SAFETY: all four inputs are complete 16-byte chunks and this function
    // enables SSE2.
    let (mut lane0, mut lane1, mut lane2, mut lane3) = unsafe {
        (
            _mm_xor_si128(load(first), initial_vector),
            load(second),
            load(third),
            load(fourth),
        )
    };
    // SAFETY: this function has SSE2 enabled.
    let fold64 = unsafe { set_u64x(FOLD_64_HIGH as u64, FOLD_64_LOW as u64) };

    while chunks.len() >= 4 {
        let next0 = chunks.next().expect("four chunks remain");
        let next1 = chunks.next().expect("three chunks remain");
        let next2 = chunks.next().expect("two chunks remain");
        let next3 = chunks.next().expect("one chunk remains");
        // SAFETY: every input is a complete 16-byte chunk and this function
        // enables PCLMULQDQ and SSE2.
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
    // SAFETY: this function has SSE2 enabled.
    let fold48 = unsafe { set_u64x(FOLD_48_HIGH as u64, FOLD_48_LOW as u64) };
    // SAFETY: this function has SSE2 enabled.
    let fold32 = unsafe { set_u64x(FOLD_32_HIGH as u64, FOLD_32_LOW as u64) };
    // SAFETY: this function has SSE2 enabled.
    let fold16 = unsafe { set_u64x(FOLD_16_HIGH as u64, FOLD_16_LOW as u64) };
    // SAFETY: this function enables PCLMULQDQ and SSE2.
    let mut state = unsafe {
        let state = fold(lane0, fold48, lane3);
        let state = fold(lane1, fold32, state);
        fold(lane2, fold16, state)
    };

    for chunk in &mut chunks {
        // SAFETY: `chunk` is complete and this function enables PCLMULQDQ and
        // SSE2.
        state = unsafe { fold(state, fold16, load(chunk)) };
    }

    // SAFETY: this function has PCLMULQDQ and SSE2 enabled.
    let mut crc = unsafe { reduce(state) };
    for &byte in chunks.remainder() {
        crc = scalar::update_byte(crc, byte);
    }
    crc
}

#[inline]
#[target_feature(enable = "sse2")]
unsafe fn set_u64x(high: u64, low: u64) -> __m128i {
    #[cfg(target_arch = "x86_64")]
    {
        _mm_set_epi64x(high as i64, low as i64)
    }

    #[cfg(target_arch = "x86")]
    {
        let low = _mm_set_epi32(0, 0, (low >> 32) as i32, low as i32);
        let high = _mm_set_epi32(0, 0, (high >> 32) as i32, high as i32);
        _mm_unpacklo_epi64(low, high)
    }
}

#[inline]
#[target_feature(enable = "pclmulqdq,sse2")]
unsafe fn reduce(mut state: __m128i) -> u16 {
    // SAFETY: this function has SSE2 enabled.
    let low_coefficient = unsafe { set_u64x(0, FOLD_8 as u64) };
    state = _mm_xor_si128(
        _mm_clmulepi64_si128(state, low_coefficient, 0x00),
        _mm_srli_si128(state, 8),
    );

    // SAFETY: this function has SSE2 enabled.
    let mask = unsafe { set_u64x(u64::MAX, 0xffff_ffff_0000_0000) };
    // SAFETY: this function has SSE2 enabled.
    let high_coefficient = unsafe { set_u64x(FOLD_4 as u64, 0) };
    state = _mm_xor_si128(
        _mm_clmulepi64_si128(_mm_slli_si128(state, 12), high_coefficient, 0x11),
        _mm_and_si128(state, mask),
    );

    // SAFETY: this function has SSE2 enabled.
    let mu_polynomial = unsafe { set_u64x(BARRETT_POLYNOMIAL as u64, BARRETT_MU as u64) };
    let quotient = _mm_clmulepi64_si128(state, mu_polynomial, 0x00);
    let product = _mm_clmulepi64_si128(quotient, mu_polynomial, 0x10);
    let result = _mm_xor_si128(state, product);

    // Shifting first avoids the x86-only `_mm_extract_epi64` intrinsic and keeps
    // this exact implementation available on both x86 and x86-64.
    _mm_cvtsi128_si32(_mm_srli_si128(result, 8)) as u16
}
