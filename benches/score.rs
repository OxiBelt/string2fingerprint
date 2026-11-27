//! Measures scoring throughput for every backend that the running CPU
//! supports.
//!
//! Run with `cargo bench`. The benchmark has no dependencies: it times
//! batches of calls with [`Instant`] and reports the fastest batch, which is
//! the least disturbed by other load. Results are indicative. Compare
//! backends on the same idle machine, and do not compare numbers taken on
//! different machines.
//!
//! Cargo passes `--bench` when it runs a benchmark for measurement. Without
//! that argument, for example under `cargo test --benches`, every case runs
//! once as a smoke test.

use std::env;
use std::hint::black_box;
use std::time::{Duration, Instant};

use string2fingerprint::{Backend, Scorer};

/// Input lengths in bytes: a short field, one block, one window, and a body
/// of several windows.
const LENGTHS: [usize; 5] = [32, 256, 512, 4096, 65536];

/// How long one batch runs.
const BATCH: Duration = Duration::from_millis(60);

/// How many batches are timed for each case.
const BATCHES: usize = 5;

/// Returns `len` bytes of request-like text.
fn text(len: usize) -> Vec<u8> {
    const LINE: &[u8] =
        b"GET /catalog/items?page=12&sort=price_asc&q=blue+ceramic+mug HTTP/1.1\r\n\
        Accept-Language: en-GB,en;q=0.9,ko;q=0.8\r\n\
        Cookie: session=7f3a9c2e41d8; theme=dark\r\n";
    LINE.iter().copied().cycle().take(len).collect()
}

/// Returns the fastest observed time of one call in nanoseconds.
fn measure(scorer: &Scorer, input: &[u8], smoke: bool) -> f64 {
    if smoke {
        black_box(scorer.score(black_box(input)));
        return 0.0;
    }
    // Size a batch so that it runs for about `BATCH`.
    let mut calls = 1u32;
    loop {
        let start = Instant::now();
        for _ in 0..calls {
            black_box(scorer.score(black_box(input)));
        }
        if start.elapsed() >= BATCH / 4 || calls >= 1 << 24 {
            break;
        }
        calls *= 2;
    }
    let mut best = f64::INFINITY;
    for _ in 0..BATCHES {
        let start = Instant::now();
        for _ in 0..calls {
            black_box(scorer.score(black_box(input)));
        }
        best = best.min(start.elapsed().as_secs_f64() * 1e9 / f64::from(calls));
    }
    best
}

fn main() {
    let smoke = !env::args().any(|argument| argument == "--bench");
    println!(
        "{:<14} {:>8} {:>14} {:>10}",
        "backend", "bytes", "ns/call", "MiB/s"
    );
    for &backend in Backend::PREFERENCE {
        let Ok(scorer) = Scorer::new().with_backend(backend) else {
            println!("{:<14} not supported on this CPU", backend.name());
            continue;
        };
        for len in LENGTHS {
            let input = text(len);
            let nanos = measure(&scorer, &input, smoke);
            if smoke {
                println!("{:<14} {len:>8} {:>14} {:>10}", backend.name(), "ok", "-");
            } else {
                // The cast is exact for these lengths.
                let rate = len as f64 / nanos * 1e9 / (1024.0 * 1024.0);
                println!("{:<14} {len:>8} {nanos:>14.0} {rate:>10.1}", backend.name());
            }
        }
    }
}
