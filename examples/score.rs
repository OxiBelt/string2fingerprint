//! Scores each line of standard input and prints one row per line.
//!
//! Usage: `score [MODEL]`. `MODEL` is the name of an embedded model, such as
//! `compact` or `large`, or the path of a model file. Without `MODEL`, the
//! embedded default model is used. A name takes precedence over a file of
//! that name in the current directory, so write `./compact` for such a file.
//! Each output row holds the eight scores in category order, a tab,
//! and the number of input bytes. Lines are scored as raw bytes without
//! their terminator, which is a line feed or a carriage return and a line
//! feed. Input bytes are never echoed.

use std::env;
use std::fs;
use std::io::{self, BufRead, Write};
use std::process::ExitCode;

use string2fingerprint::model::{EmbeddedModel, Model};
use string2fingerprint::{Category, Scorer};

fn run(model: Option<&str>) -> Result<(), Box<dyn std::error::Error>> {
    let scorer = match model {
        None => Scorer::new(),
        Some(model) => match EmbeddedModel::from_name(model) {
            Some(embedded) => Scorer::with_embedded(embedded),
            None => Scorer::with_model(Model::from_bytes(&fs::read(model)?)?),
        },
    };
    let mut out = io::stdout().lock();
    let names: Vec<&str> = Category::ALL
        .iter()
        .map(|category| category.name())
        .collect();
    writeln!(out, "# backend: {}", scorer.backend())?;
    writeln!(out, "# {}\tbytes", names.join("\t"))?;
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
        let scores = scorer.score(&line);
        for (_, score) in scores.iter() {
            write!(out, "{score}\t")?;
        }
        writeln!(out, "{}", line.len())?;
    }
    Ok(())
}

fn main() -> ExitCode {
    let arguments: Vec<String> = env::args().skip(1).collect();
    if arguments.len() > 1 {
        eprintln!("usage: score [MODEL]");
        return ExitCode::from(2);
    }
    match run(arguments.first().map(String::as_str)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("score: {error}");
            ExitCode::FAILURE
        }
    }
}
