//! Prints the categories that the model of the build script flags in each
//! line of standard input.
//!
//! Each output row holds the names of the flagged categories, separated by
//! commas, or `benign` when the line has none. Lines are scored as raw bytes
//! without their terminator, which is a line feed or a carriage return and a
//! line feed. Input bytes are never echoed. The Docker end-to-end check in
//! `tests/docker/` runs this program.

use std::io::{self, BufRead, Write};
use std::process::ExitCode;

fn run() -> io::Result<()> {
    let detector = s2fp_downstream::detector();
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
        writeln!(out, "{}", detector.detect(&line))?;
    }
    out.flush()
}

fn main() -> ExitCode {
    if std::env::args_os().len() > 1 {
        eprintln!("usage: s2fp-downstream < LINES");
        return ExitCode::from(2);
    }
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("s2fp-downstream: {error}");
            ExitCode::FAILURE
        }
    }
}
