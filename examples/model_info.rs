//! Prints the versions of model files and whether this library loads them.
//!
//! Usage: `model_info MODEL...`, where `MODEL` is the path of a model file or
//! the name of an embedded model, such as `default`.
//!
//! For each model it prints one line: the model, the fingerprint layout
//! version that it was trained on, the versions of the library that may
//! load it, and either the layer sizes and the checksum of a model that
//! loads, or the reason it does not. Model bytes are never printed.
//!
//! The exit status is 0 when every model loads, 1 when one does not, and 2
//! for an invalid command line.

use std::env;
use std::fs;
use std::process::ExitCode;

use string2fingerprint::VERSION;
use string2fingerprint::fingerprint::LAYOUT_VERSION;
use string2fingerprint::model::{EmbeddedModel, Metadata, Model};

/// Describes one model, or returns why it does not load.
fn describe(bytes: &[u8]) -> Result<String, String> {
    let metadata = Metadata::read(bytes).map_err(|error| error.to_string())?;
    let versions = format!(
        "layout {}, library {}",
        metadata.layout_version(),
        metadata.library_requirement()
    );
    match Model::from_bytes(bytes) {
        Ok(model) => {
            let sizes: Vec<String> = model.layer_sizes().map(|size| size.to_string()).collect();
            Ok(format!(
                "{versions}, layers {}, checksum {:#010x}",
                sizes.join("-"),
                model.checksum()
            ))
        }
        Err(error) => Err(format!("{versions}: {error}")),
    }
}

fn main() -> ExitCode {
    let arguments: Vec<String> = env::args().skip(1).collect();
    if arguments.is_empty() || arguments.iter().any(|argument| argument == "--help") {
        eprintln!("usage: model_info MODEL...");
        eprintln!("this library is version {VERSION} with fingerprint layout {LAYOUT_VERSION}");
        return ExitCode::from(2);
    }
    let mut status = ExitCode::SUCCESS;
    for argument in &arguments {
        let bytes = match EmbeddedModel::from_name(argument) {
            Some(model) => Ok(model.bytes().to_vec()),
            None => fs::read(argument),
        };
        let line = match bytes {
            Ok(bytes) => describe(&bytes),
            Err(error) => Err(error.to_string()),
        };
        match line {
            Ok(line) => println!("{argument}: {line}"),
            Err(reason) => {
                println!("{argument}: does not load: {reason}");
                status = ExitCode::FAILURE;
            }
        }
    }
    status
}
