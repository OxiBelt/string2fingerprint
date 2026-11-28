//! Decides each line of standard input and prints one row per line.
//!
//! Usage: `detect [PRESET]`. `PRESET` is `fast`, `balanced`, `thorough`, or
//! `strict`, and `balanced` without it. Each output row holds the flagged
//! categories, separated by commas, or `benign`, then a tab and the highest
//! score. Lines are scored as raw bytes without their terminator, which is
//! a line feed or a carriage return and a line feed. Input bytes are never
//! echoed.

use std::env;
use std::io::{self, BufRead, Write};
use std::process::ExitCode;

use string2fingerprint::{Detector, Preset};

fn run(detector: &Detector) -> io::Result<()> {
    let mut out = io::BufWriter::new(io::stdout().lock());
    let mut input = io::stdin().lock();
    let mut line = Vec::new();
    loop {
        line.clear();
        if input.read_until(b'\n', &mut line)? == 0 {
            break;
        }
        if line.last() == Some(&b'\n') {
            line.pop();
            if line.last() == Some(&b'\r') {
                line.pop();
            }
        }
        let detection = detector.detect(&line);
        writeln!(out, "{detection}\t{}", detection.scores().max().1)?;
    }
    out.flush()
}

fn main() -> ExitCode {
    let arguments: Vec<String> = env::args().skip(1).collect();
    let preset = match arguments.as_slice() {
        [] => Preset::Balanced,
        [name] => match Preset::from_name(name) {
            Some(preset) => preset,
            None => {
                eprintln!("detect: unknown preset `{name}`");
                return ExitCode::from(2);
            }
        },
        _ => {
            eprintln!("usage: detect [PRESET]");
            return ExitCode::from(2);
        }
    };
    let detector = Detector::with_preset(preset);
    eprintln!(
        "# preset: {preset}, backend: {}",
        detector.scorer().backend()
    );
    match run(&detector) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("detect: {error}");
            ExitCode::FAILURE
        }
    }
}
