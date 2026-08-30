//! Backend selection for long inputs.
//!
//! Architecture modules expose safe, capability-checked entry points. Keeping
//! selection here leaves the public API independent of target-specific code and
//! gives future backends one small integration point.

use crate::scalar;

#[inline]
pub(crate) fn update(initial: u16, bytes: &[u8]) -> u16 {
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), not(miri)))]
    if let Some(crc) = crate::x86::update_if_supported(initial, bytes) {
        return crc;
    }

    scalar::update(initial, bytes)
}
