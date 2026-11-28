//! Loading [`Model`]s without holding a runtime worker: the asynchronous
//! counterparts of [`Model::from_bytes`](crate::model::Model::from_bytes)
//! and of reading a model file.
//!
//! Parsing validates every weight and checks the checksum, which takes
//! time linear in the size of the model (about 200 KiB for the default
//! model). Both functions do that on Tokio's blocking pool, and [`read`]
//! reads the file there too, as [`tokio::fs`] does.
//!
//! ```no_run
//! use string2fingerprint::tokio::AsyncScorer;
//!
//! # async fn run() -> std::io::Result<()> {
//! let model = string2fingerprint::tokio::model::read("models/waf.s2fp").await?;
//! let scorer = AsyncScorer::from(model);
//! let scores = scorer.score(b"q=1' OR '1'='1".to_vec()).await;
//! # let _ = scores;
//! # Ok(())
//! # }
//! ```
//!
//! [`tokio::fs`]: https://docs.rs/tokio/1/tokio/fs/index.html

use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

use super::Offload;
use super::offload::Job;
use crate::fingerprint::FINGERPRINT_LEN;
use crate::model::{MAX_HIDDEN, MAX_LAYERS, Model, ModelError};
use crate::version::{MAX_REQUIREMENT_LEN, MAX_VERSION_LEN};

/// More bytes than any model that [`Model::from_bytes`] accepts: the fixed
/// fields, the longest metadata, and [`MAX_LAYERS`] layers of the widest
/// shape. [`read`] refuses a longer file before it reads all of it.
const MAX_FILE_LEN: usize = 4
    + 4
    + 1
    + MAX_VERSION_LEN
    + 1
    + MAX_REQUIREMENT_LEN
    + 2 * (MAX_LAYERS + 1)
    + MAX_LAYERS * (4 + MAX_HIDDEN * (8 + FINGERPRINT_LEN))
    + 4;

/// Parses and validates a model on Tokio's blocking pool, and returns the
/// result of [`Model::from_bytes`].
///
/// The future owns `bytes` and borrows nothing. Outside a Tokio runtime
/// the model is parsed when the future is first polled.
///
/// ```
/// use string2fingerprint::model::DEFAULT_MODEL;
///
/// # #[tokio::main(flavor = "current_thread")]
/// # async fn main() -> Result<(), string2fingerprint::model::ModelError> {
/// let model = string2fingerprint::tokio::model::from_bytes(DEFAULT_MODEL).await?;
/// assert_eq!(model, string2fingerprint::model::Model::embedded());
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns the [`ModelError`] of [`Model::from_bytes`].
pub fn from_bytes<B>(bytes: B) -> Offload<Result<Model, ModelError>>
where
    B: AsRef<[u8]> + Send + 'static,
{
    Offload::new(Job::blocking(move || Model::from_bytes(bytes.as_ref())))
}

/// Reads the model file at `path` and parses it, both on Tokio's blocking
/// pool.
///
/// A file longer than any valid model is refused after reading a little
/// more than the longest one, so a wrong path cannot make the call read a
/// large file whole. Outside a Tokio runtime the file is read when the
/// future is first polled.
///
/// # Errors
///
/// Returns the error of opening or reading the file. A file that is not a
/// valid model fails with [`io::ErrorKind::InvalidData`], and its
/// [`get_ref`](io::Error::get_ref) is the [`ModelError`] of
/// [`Model::from_bytes`], or a message for a file that is too long:
///
/// ```
/// use std::io;
///
/// use string2fingerprint::model::ModelError;
///
/// fn model_error(error: &io::Error) -> Option<&ModelError> {
///     error.get_ref()?.downcast_ref()
/// }
/// ```
pub fn read(path: impl AsRef<Path>) -> Offload<io::Result<Model>> {
    let path = path.as_ref().to_owned();
    Offload::new(Job::blocking(move || {
        let mut bytes = Vec::new();
        // At most `MAX_FILE_LEN + 1` bytes, which fits in a `u64`.
        File::open(path)?
            .take(MAX_FILE_LEN as u64 + 1)
            .read_to_end(&mut bytes)?;
        if bytes.len() > MAX_FILE_LEN {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "the file is longer than any model",
            ));
        }
        Model::from_bytes(&bytes).map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
    }))
}

#[cfg(test)]
mod tests {
    use core::future::Future;

    use super::*;
    use crate::model::{DEFAULT_MODEL, EmbeddedModel};

    fn block_on<F: Future>(future: F) -> F::Output {
        ::tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("a runtime starts")
            .block_on(future)
    }

    #[test]
    fn the_longest_model_fits_the_file_limit() {
        for model in EmbeddedModel::ALL {
            assert!(model.bytes().len() <= MAX_FILE_LEN);
        }
        // Four layers of the widest shape, one of them the input layer.
        const { assert!(MAX_FILE_LEN > 4 * 256 * 3072) };
    }

    #[test]
    fn parses_like_the_synchronous_function() {
        assert_eq!(
            block_on(from_bytes(DEFAULT_MODEL)),
            Model::from_bytes(DEFAULT_MODEL)
        );
        assert_eq!(
            block_on(from_bytes(vec![0u8; 3])),
            Model::from_bytes(&[0u8; 3])
        );
    }
}
