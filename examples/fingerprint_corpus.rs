//! Converts a labeled corpus into fingerprints for the training tools.
//!
//! Usage: `fingerprint_corpus <corpus> <features> <labels>`.
//!
//! The corpus is a sequence of records: a little-endian `u32` payload length,
//! one label byte, and the payload. For every record whose payload has a
//! window, the `FINGERPRINT_LEN` features of the first window are appended to
//! `<features>` and the label byte to `<labels>`. Using the library's own
//! extractor keeps the training features identical to the features seen at
//! inference.
//!
//! The metadata of the features goes to `<features>.meta`, one `key value`
//! pair on each line: `layout_version`, the fingerprint layout version of the
//! library, and `library_requirement`, `^` the version of the library. The
//! training tools write both into the model they train on the features.

use std::env;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::process::ExitCode;

use string2fingerprint::VERSION;
use string2fingerprint::fingerprint::{LAYOUT_VERSION, fingerprints};
use string2fingerprint::version::VersionReq;

fn convert(corpus: &[u8], features: &mut impl Write, labels: &mut impl Write) -> io::Result<u64> {
    let invalid = |message| io::Error::new(io::ErrorKind::InvalidData, message);
    let mut rest = corpus;
    let mut written = 0;
    while !rest.is_empty() {
        let (header, tail) = rest
            .split_at_checked(5)
            .ok_or_else(|| invalid("truncated record header"))?;
        let len = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
        let len = usize::try_from(len).map_err(|_| invalid("record is too long"))?;
        let (payload, tail) = tail
            .split_at_checked(len)
            .ok_or_else(|| invalid("truncated record payload"))?;
        rest = tail;
        if let Some(fingerprint) = fingerprints(payload).next_window() {
            features.write_all(fingerprint.as_array())?;
            labels.write_all(&header[4..])?;
            written += 1;
        }
    }
    features.flush()?;
    labels.flush()?;
    Ok(written)
}

fn main() -> ExitCode {
    let arguments: Vec<String> = env::args().skip(1).collect();
    let [corpus, features, labels] = arguments.as_slice() else {
        eprintln!("usage: fingerprint_corpus <corpus> <features> <labels>");
        return ExitCode::from(2);
    };
    let result = fs::read(corpus).and_then(|corpus| {
        let metadata = format!(
            "layout_version {LAYOUT_VERSION}\nlibrary_requirement {}\n",
            VersionReq::caret(&VERSION)
        );
        fs::write(format!("{features}.meta"), metadata)?;
        let mut features = BufWriter::new(File::create(features)?);
        let mut labels = BufWriter::new(File::create(labels)?);
        convert(&corpus, &mut features, &mut labels)
    });
    match result {
        Ok(written) => {
            println!("{written}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("fingerprint_corpus: {error}");
            ExitCode::FAILURE
        }
    }
}
