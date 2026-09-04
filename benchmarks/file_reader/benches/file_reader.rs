use mavlink_after::{
    AsyncMavlinkReader, MavlinkReader, MavlinkVersion, dialects::ardupilotmega::MavMessage,
};
use mavlink_before::{
    async_peek_reader::AsyncPeekReader, peek_reader::PeekReader, read_versioned_msg,
    read_versioned_msg_async,
};
use std::{
    io::Cursor,
    sync::LazyLock,
    time::{Duration, Instant},
};

const MESSAGE_COUNT: usize = 1_426;
const REPEATS: usize = 5;
const SAMPLES: usize = 15;
const SAMPLE_TIME: Duration = Duration::from_millis(250);

static CLEAN_TLOG: LazyLock<Vec<u8>> =
    LazyLock::new(|| include_bytes!("../../../mavlink/tests/log.tlog").repeat(REPEATS));

/// The clean tlog with deterministic noise inserted between intact frames.
///
/// Noise is added after runs of 1 through 5 frames, while burst sizes cycle from
/// 5 through 30 bytes. With five repetitions, this makes ~10% of the stream noisy.
static NOISY_TLOG: LazyLock<Vec<u8>> = LazyLock::new(|| {
    let mut noisy_tlog = Vec::with_capacity(CLEAN_TLOG.len());
    let mut record_start = 0;
    let mut frames_until_noise = 1;
    let mut next_frame_gap = 2;
    let mut noise_length = 5;
    let mut noise_byte = 0_u8;

    while record_start < CLEAN_TLOG.len() {
        // Each tlog record is an 8-byte timestamp followed by a MAVLink frame.
        let frame_start = record_start + 8;
        assert_eq!(CLEAN_TLOG[frame_start], mavlink_after::MAV_STX_V2);
        assert_eq!(CLEAN_TLOG[frame_start + 2] & 1, 0);

        let frame_end = frame_start
            + mavlink_after::consts::STX_SIZE
            + mavlink_after::consts::v2::HEADER_SIZE
            + usize::from(CLEAN_TLOG[frame_start + 1])
            + mavlink_after::consts::CHECKSUM_SIZE;
        noisy_tlog.extend_from_slice(&CLEAN_TLOG[record_start..frame_end]);
        record_start = frame_end;

        frames_until_noise -= 1;
        if frames_until_noise == 0 {
            for _ in 0..noise_length {
                noisy_tlog.push(noise_byte);
                noise_byte = noise_byte.wrapping_add(1);
            }

            noise_length = if noise_length == 30 {
                5
            } else {
                noise_length + 1
            };
            frames_until_noise = next_frame_gap;
            next_frame_gap = if next_frame_gap == 5 {
                1
            } else {
                next_frame_gap + 1
            };
        }
    }

    noisy_tlog
});

fn main() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();

    println!("stream    api       reader      messages        time     throughput  vs master");
    println!("--------  --------  --------  ----------  ----------  -------------  ---------");

    for (stream_name, stream) in [
        ("clean", CLEAN_TLOG.as_slice()),
        ("noisy", NOISY_TLOG.as_slice()),
    ] {
        let blocking_master = || {
            let mut reader = PeekReader::new(Cursor::new(stream));
            for _ in 0..MESSAGE_COUNT * REPEATS {
                std::hint::black_box(
                    read_versioned_msg::<mavlink_before::dialects::ardupilotmega::MavMessage, _>(
                        &mut reader,
                        mavlink_before::MavlinkVersion::V2.into(),
                    )
                    .unwrap(),
                );
            }
        };
        let blocking_current = || {
            let mut reader = MavlinkReader::with_capacity(
                mavlink_after::consts::DEFAULT_READ_BUFFER_CAPACITY,
                Cursor::new(stream),
            );
            for _ in 0..MESSAGE_COUNT * REPEATS {
                std::hint::black_box(
                    reader
                        .read_message::<MavMessage>(MavlinkVersion::V2)
                        .unwrap(),
                );
            }
        };
        let tokio_master = || {
            runtime.block_on(async {
                let mut reader = AsyncPeekReader::new(stream);
                for _ in 0..MESSAGE_COUNT * REPEATS {
                    std::hint::black_box(
                        read_versioned_msg_async::<
                            mavlink_before::dialects::ardupilotmega::MavMessage,
                            _,
                        >(
                            &mut reader, mavlink_before::MavlinkVersion::V2.into()
                        )
                        .await
                        .unwrap(),
                    );
                }
            });
        };
        let tokio_current = || {
            runtime.block_on(async {
                let mut reader = AsyncMavlinkReader::with_capacity(
                    mavlink_after::consts::DEFAULT_READ_BUFFER_CAPACITY,
                    stream,
                );
                for _ in 0..MESSAGE_COUNT * REPEATS {
                    std::hint::black_box(
                        reader
                            .read_message::<MavMessage>(MavlinkVersion::V2)
                            .await
                            .unwrap(),
                    );
                }
            });
        };
        let readers: [(&str, &str, &dyn Fn()); 4] = [
            ("blocking", "master", &blocking_master),
            ("blocking", "current", &blocking_current),
            ("tokio", "master", &tokio_master),
            ("tokio", "current", &tokio_current),
        ];

        for (_, _, replay) in readers {
            replay();
        }

        let mut samples: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::with_capacity(SAMPLES));
        for round in 0..SAMPLES {
            for offset in 0..readers.len() {
                let index = (round + offset) % readers.len();
                let start = Instant::now();
                let mut iterations = 0;
                while start.elapsed() < SAMPLE_TIME {
                    readers[index].2();
                    iterations += 1;
                }
                samples[index].push(start.elapsed().as_secs_f64() / f64::from(iterations));
            }
        }
        let times = samples.map(|mut values| {
            values.sort_by(f64::total_cmp);
            values[values.len() / 2]
        });

        for (index, ((api, reader_name, _), time)) in readers.into_iter().zip(times).enumerate() {
            let vs_master = if reader_name == "master" {
                "-".to_owned()
            } else {
                format!("{:.2}x", times[index - 1] / time)
            };
            println!(
                "{stream_name:<8}  {api:<8}  {reader_name:<8}  {:>10}  {:>7.3} ms  {:>9.1} MiB/s  {vs_master:>9}",
                MESSAGE_COUNT * REPEATS,
                time * 1_000.0,
                stream.len() as f64 / 1_048_576.0 / time,
            );
        }
    }
}
