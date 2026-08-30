//! Safe fallback for every target Rust supports.

const REVERSED_POLYNOMIAL: u16 = 0x8408;

pub(crate) const fn table_entry(mut value: u16, polynomial: u16) -> u16 {
    let mut bit = 0;
    while bit < 8 {
        value = (value >> 1) ^ ((value & 1) * polynomial);
        bit += 1;
    }
    value
}

pub(crate) const fn make_tables(polynomial: u16) -> [[u16; 256]; 16] {
    let mut tables = [[0; 256]; 16];

    let mut byte = 0;
    while byte < 256 {
        tables[0][byte] = table_entry(byte as u16, polynomial);
        byte += 1;
    }

    byte = 0;
    while byte < 256 {
        let mut lane = 1;
        while lane < 16 {
            let previous = tables[lane - 1][byte];
            tables[lane][byte] = (previous >> 8) ^ tables[0][(previous & 0xff) as usize];
            lane += 1;
        }
        byte += 1;
    }

    tables
}

pub(crate) static TABLES: [[u16; 256]; 16] = make_tables(REVERSED_POLYNOMIAL);

#[inline(always)]
pub(crate) fn update_byte(crc: u16, byte: u8) -> u16 {
    (crc >> 8) ^ TABLES[0][(crc as u8 ^ byte) as usize]
}

#[inline(always)]
pub(crate) fn update_small_and_extra(mut crc: u16, bytes: &[u8], extra: u8) -> u16 {
    debug_assert!(bytes.len() < 8);
    for &byte in bytes {
        crc = update_byte(crc, byte);
    }
    update_byte(crc, extra)
}

/// Folds the five-byte MAVLink 1 header and `CRC_EXTRA` in parallel.
#[inline(always)]
pub(crate) fn update_5_and_extra(crc: u16, bytes: &[u8], extra: u8) -> u16 {
    debug_assert_eq!(bytes.len(), 5);
    TABLES[0][extra as usize]
        ^ TABLES[1][bytes[4] as usize]
        ^ TABLES[2][bytes[3] as usize]
        ^ TABLES[3][bytes[2] as usize]
        ^ TABLES[4][(bytes[1] ^ (crc >> 8) as u8) as usize]
        ^ TABLES[5][(bytes[0] ^ crc as u8) as usize]
}

/// Folds the nine-byte MAVLink 2 header and `CRC_EXTRA` in parallel.
#[inline(always)]
pub(crate) fn update_9_and_extra(crc: u16, bytes: &[u8], extra: u8) -> u16 {
    debug_assert_eq!(bytes.len(), 9);
    TABLES[0][extra as usize]
        ^ TABLES[1][bytes[8] as usize]
        ^ TABLES[2][bytes[7] as usize]
        ^ TABLES[3][bytes[6] as usize]
        ^ TABLES[4][bytes[5] as usize]
        ^ TABLES[5][bytes[4] as usize]
        ^ TABLES[6][bytes[3] as usize]
        ^ TABLES[7][bytes[2] as usize]
        ^ TABLES[8][(bytes[1] ^ (crc >> 8) as u8) as usize]
        ^ TABLES[9][(bytes[0] ^ crc as u8) as usize]
}

/// Folds a full nine-byte MAVLink 1 heartbeat payload, its five-byte header,
/// and `CRC_EXTRA` without a serial lookup chain.
#[inline(always)]
pub(crate) fn update_14_and_extra(crc: u16, bytes: &[u8], extra: u8) -> u16 {
    debug_assert_eq!(bytes.len(), 14);
    TABLES[0][extra as usize]
        ^ TABLES[1][bytes[13] as usize]
        ^ TABLES[2][bytes[12] as usize]
        ^ TABLES[3][bytes[11] as usize]
        ^ TABLES[4][bytes[10] as usize]
        ^ TABLES[5][bytes[9] as usize]
        ^ TABLES[6][bytes[8] as usize]
        ^ TABLES[7][bytes[7] as usize]
        ^ TABLES[8][bytes[6] as usize]
        ^ TABLES[9][bytes[5] as usize]
        ^ TABLES[10][bytes[4] as usize]
        ^ TABLES[11][bytes[3] as usize]
        ^ TABLES[12][bytes[2] as usize]
        ^ TABLES[13][(bytes[1] ^ (crc >> 8) as u8) as usize]
        ^ TABLES[14][(bytes[0] ^ crc as u8) as usize]
}

/// Slicing-by-16 keeps the portable path fast without unsafe code or heap use.
#[inline]
pub(crate) fn update(mut crc: u16, bytes: &[u8]) -> u16 {
    let mut chunks = bytes.chunks_exact(16);

    for chunk in &mut chunks {
        let first = chunk[0] ^ crc as u8;
        let second = chunk[1] ^ (crc >> 8) as u8;

        crc = TABLES[0][chunk[15] as usize]
            ^ TABLES[1][chunk[14] as usize]
            ^ TABLES[2][chunk[13] as usize]
            ^ TABLES[3][chunk[12] as usize]
            ^ TABLES[4][chunk[11] as usize]
            ^ TABLES[5][chunk[10] as usize]
            ^ TABLES[6][chunk[9] as usize]
            ^ TABLES[7][chunk[8] as usize]
            ^ TABLES[8][chunk[7] as usize]
            ^ TABLES[9][chunk[6] as usize]
            ^ TABLES[10][chunk[5] as usize]
            ^ TABLES[11][chunk[4] as usize]
            ^ TABLES[12][chunk[3] as usize]
            ^ TABLES[13][chunk[2] as usize]
            ^ TABLES[14][second as usize]
            ^ TABLES[15][first as usize];
    }

    for &byte in chunks.remainder() {
        crc = update_byte(crc, byte);
    }

    crc
}

/// Updates with exactly 15 bytes followed by one separately supplied byte.
///
/// This avoids turning a final 16-byte block into sixteen serial table lookups
/// just because `CRC_EXTRA` is passed separately.
#[inline(always)]
pub(crate) fn update_15_and_extra(crc: u16, bytes: &[u8], extra: u8) -> u16 {
    debug_assert_eq!(bytes.len(), 15);

    let first = bytes[0] ^ crc as u8;
    let second = bytes[1] ^ (crc >> 8) as u8;
    TABLES[0][extra as usize]
        ^ TABLES[1][bytes[14] as usize]
        ^ TABLES[2][bytes[13] as usize]
        ^ TABLES[3][bytes[12] as usize]
        ^ TABLES[4][bytes[11] as usize]
        ^ TABLES[5][bytes[10] as usize]
        ^ TABLES[6][bytes[9] as usize]
        ^ TABLES[7][bytes[8] as usize]
        ^ TABLES[8][bytes[7] as usize]
        ^ TABLES[9][bytes[6] as usize]
        ^ TABLES[10][bytes[5] as usize]
        ^ TABLES[11][bytes[4] as usize]
        ^ TABLES[12][bytes[3] as usize]
        ^ TABLES[13][bytes[2] as usize]
        ^ TABLES[14][second as usize]
        ^ TABLES[15][first as usize]
}

/// Folds an 8-to-14-byte remainder and `CRC_EXTRA` in parallel.
#[inline]
pub(crate) fn update_tail_and_extra(crc: u16, bytes: &[u8], extra: u8) -> u16 {
    debug_assert!((8..15).contains(&bytes.len()));

    let total = bytes.len() + 1;
    let mut result = 0;
    let mut position = 0;
    while position < total {
        let mut byte = if position == bytes.len() {
            extra
        } else {
            bytes[position]
        };
        if position == 0 {
            byte ^= crc as u8;
        } else if position == 1 {
            byte ^= (crc >> 8) as u8;
        }
        result ^= TABLES[total - position - 1][byte as usize];
        position += 1;
    }
    result
}
