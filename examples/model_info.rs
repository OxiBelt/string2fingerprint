//! Prints the versions of model files and whether this library loads them.
//!
//! Usage: `model_info MODEL...`, where `MODEL` is the path of a model file or
//! the name of an embedded model, such as `default`, or
//! `model_info --library-requirement REQ --out FILE MODEL`.
//!
//! For each model it prints one line: the model, the fingerprint layout
//! version that it was trained on, the versions of the library that may
//! load it, and either the layer sizes and the checksum of a model that
//! loads, or the reason it does not. Model bytes are never printed.
//!
//! With `--library-requirement REQ --out FILE`, it loads the one model with
//! `REQ` in place of its library requirement, as
//! `Model::from_bytes_with_library_requirement` does, writes it to `FILE`,
//! and describes the new file. Use it when a new version of the library
//! still runs a model whose requirement excludes that version, for example
//! for the embedded models after the crate version changes. `REQ` must
//! match this library.
//!
//! The exit status is 0 when every model loads, 1 when one does not or the
//! file cannot be written, and 2 for an invalid command line.

use std::env;
use std::fs;
use std::process::ExitCode;

use string2fingerprint::VERSION;
use string2fingerprint::fingerprint::LAYOUT_VERSION;
use string2fingerprint::model::{EmbeddedModel, Metadata, Model};
use string2fingerprint::version::VersionReq;

const USAGE: &str =
    "usage: model_info MODEL...\n       model_info --library-requirement REQ --out FILE MODEL";

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

/// Reads a model file, or the bytes of an embedded model by its name.
fn read(argument: &str) -> Result<Vec<u8>, String> {
    match EmbeddedModel::from_name(argument) {
        Some(model) => Ok(model.bytes().to_vec()),
        None => fs::read(argument).map_err(|error| error.to_string()),
    }
}

/// Writes `model` with `requirement` to `out`, and describes the new file.
fn restamp(model: &str, requirement: &str, out: &str) -> ExitCode {
    let requirement: VersionReq = match requirement.parse() {
        Ok(requirement) => requirement,
        Err(error) => {
            eprintln!("model_info: `--library-requirement {requirement}`: {error}");
            return ExitCode::from(2);
        }
    };
    if !requirement.matches(&VERSION) {
        eprintln!(
            "model_info: `--library-requirement {requirement}` does not match this library, version {VERSION}"
        );
        return ExitCode::from(2);
    }
    let written = read(model)
        .and_then(|bytes| {
            Model::from_bytes_with_library_requirement(&bytes, requirement)
                .map_err(|error| error.to_string())
        })
        .and_then(|model| {
            let bytes = model.to_bytes();
            fs::write(out, &bytes).map_err(|error| format!("{out}: {error}"))?;
            describe(&bytes)
        });
    match written {
        Ok(line) => {
            println!("{out}: {line}");
            ExitCode::SUCCESS
        }
        Err(reason) => {
            println!("{model}: does not load: {reason}");
            ExitCode::FAILURE
        }
    }
}

fn main() -> ExitCode {
    let arguments: Vec<String> = env::args().skip(1).collect();
    if arguments.is_empty() || arguments.iter().any(|argument| argument == "--help") {
        eprintln!("{USAGE}");
        eprintln!("this library is version {VERSION} with fingerprint layout {LAYOUT_VERSION}");
        return ExitCode::from(2);
    }
    if arguments[0] == "--library-requirement" {
        return match arguments.as_slice() {
            [_, requirement, flag, out, model] if flag == "--out" => {
                restamp(model, requirement, out)
            }
            _ => {
                eprintln!("{USAGE}");
                ExitCode::from(2)
            }
        };
    }
    if let Some(option) = arguments.iter().find(|argument| argument.starts_with("--")) {
        eprintln!("model_info: unknown option `{option}`\n{USAGE}");
        return ExitCode::from(2);
    }
    let mut status = ExitCode::SUCCESS;
    for argument in &arguments {
        match read(argument).and_then(|bytes| describe(&bytes)) {
            Ok(line) => println!("{argument}: {line}"),
            Err(reason) => {
                println!("{argument}: does not load: {reason}");
                status = ExitCode::FAILURE;
            }
        }
    }
    status
}
