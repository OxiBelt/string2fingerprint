//! Decides each line of standard input on a Tokio runtime and prints one
//! row per line, as the `detect` example does.
//!
//! Usage: `detect_async [PRESET]`, with the `tokio` feature. The arguments
//! and the output are those of `detect`, so both print the same rows for
//! the same input. Lines up to the inline limit are decided in the task
//! that reads them, longer ones on Tokio's blocking pool, while the
//! runtime's workers stay free. Input bytes are never echoed.

use std::env;
use std::io;
use std::process::ExitCode;

use string2fingerprint::Preset;
use string2fingerprint::tokio::AsyncDetector;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter};

async fn run(detector: &AsyncDetector) -> io::Result<()> {
    let mut out = BufWriter::new(tokio::io::stdout());
    let mut input = BufReader::new(tokio::io::stdin());
    loop {
        let mut line = Vec::new();
        if input.read_until(b'\n', &mut line).await? == 0 {
            break;
        }
        if line.last() == Some(&b'\n') {
            line.pop();
            if line.last() == Some(&b'\r') {
                line.pop();
            }
        }
        // The future owns the line, so a long one can go to the pool.
        let detection = detector.detect(line).await;
        let row = format!("{detection}\t{}\n", detection.scores().max().1);
        out.write_all(row.as_bytes()).await?;
    }
    out.flush().await
}

#[tokio::main]
async fn main() -> ExitCode {
    let arguments: Vec<String> = env::args().skip(1).collect();
    let preset = match arguments.as_slice() {
        [] => Preset::Balanced,
        [name] => match Preset::from_name(name) {
            Some(preset) => preset,
            None => {
                eprintln!("detect_async: unknown preset `{name}`");
                return ExitCode::from(2);
            }
        },
        _ => {
            eprintln!("usage: detect_async [PRESET]");
            return ExitCode::from(2);
        }
    };
    let detector = AsyncDetector::with_preset(preset);
    eprintln!(
        "# preset: {preset}, backend: {}, inline limit: {} bytes",
        detector.detector().scorer().backend(),
        detector.inline_limit()
    );
    match run(&detector).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("detect_async: {error}");
            ExitCode::FAILURE
        }
    }
}
