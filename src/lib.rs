//! Scores untrusted text for how malformed it looks and whether it attempts
//! to escape the context it will be placed in.
//!
//! `string2fingerprint` is meant for places that receive arbitrary bytes from
//! an untrusted source, such as a web reverse proxy. It reduces the input to
//! fixed-size fingerprints, runs them through a small 8-bit quantized neural
//! network, and returns one score for each [`Category`].
//!
//! The library supports Linux on x86-64, AArch64, and RISC-V 64.
//!
//! # Contract
//!
//! - Any byte slice is accepted. Scoring never fails and never panics.
//! - Work is linear in the input length and scoring allocates no heap memory.
//! - Scores depend only on the input bytes and the model, never on the CPU.
//!
//! The design document in `docs/design.md` specifies the full contract.
//!
//! # Interfaces
//!
//! The high-level interface decides inputs in one call. A [`Detector`]
//! pairs a model with a threshold for each category, from a
//! [`Preset`] or a [`DetectorBuilder`],
//! and returns a [`Detection`]:
//!
//! ```
//! use string2fingerprint::prelude::*;
//!
//! let detector = Detector::new();
//! let detection = detector.detect(b"/search?q=1%27%20OR%20%271%27=%271");
//! assert!(detection.contains(Category::SqlInjection));
//! println!("flagged: {detection}");
//! ```
//!
//! A [`Scorer`] returns the scores without deciding them, and both score
//! the fields of a message, batches, readers, and input that arrives in
//! parts through a [`ScoreStream`]. [`Thresholds`] and [`eval`] choose
//! thresholds on labeled samples, and the `train` and `build` features
//! train models.
//!
//! The low-level interface exposes each stage of the pipeline:
//!
//! | Stage | Items |
//! | --- | --- |
//! | Symbols and n-grams | [`fingerprint::lexical_symbol`], [`fingerprint::shape_class`], [`fingerprint::ngrams`] |
//! | Fingerprints | [`fingerprint::Fingerprint`], [`fingerprint::Section`], [`fingerprint::fingerprints`], [`fingerprint::Fingerprinter`], [`fingerprint::window_range`] |
//! | Kernel and backends | [`Backend`], [`kernel::Kernel`], [`kernel::Weights`] |
//! | Network | [`model::Model`], [`model::LayerView`], [`model::Logits`], [`model::Trace`] |
//! | Windows | [`Scorer::windows`], [`Scorer::score_fingerprint`], [`Scorer::logits`] |
//!
//! ```
//! use string2fingerprint::fingerprint::Fingerprinter;
//! use string2fingerprint::kernel::Kernel;
//! use string2fingerprint::model::{Logits, Model};
//! use string2fingerprint::Scorer;
//!
//! // The stages by hand give what the scorer gives.
//! let input = b"<img src=x onerror=alert(document.domain)>";
//! let model = Model::embedded();
//! let kernel = Kernel::detect();
//! let mut fingerprinter = Fingerprinter::new();
//! let mut logits = Logits::from_array([i8::MIN; 8]);
//! fingerprinter.update(input, |window| logits = logits.max_with(model.logits(kernel, window)));
//! if let Some(window) = fingerprinter.finish() {
//!     logits = logits.max_with(model.logits(kernel, window));
//! }
//! assert_eq!(logits.scores(), Scorer::new().score(input));
//! ```

#[cfg(not(target_os = "linux"))]
compile_error!("`string2fingerprint` supports Linux targets only");

mod backend;
#[cfg(feature = "build")]
pub mod build;
mod category;
mod category_set;
pub mod detect;
pub mod eval;
pub mod fingerprint;
pub mod kernel;
pub mod model;
pub mod prelude;
mod scorer;
mod scores;
mod stream;
#[cfg(test)]
mod testutil;
mod thresholds;
#[cfg(feature = "train")]
pub mod train;

pub use backend::{Backend, UnsupportedBackend};
pub use category::Category;
pub use category_set::{CategorySet, ParseCategorySetError};
pub use detect::{Detection, DetectionStream, Detector, DetectorBuilder, Preset};
pub use scorer::{Scorer, ScorerBuilder, ScorerError, WindowScores, Windows};
pub use scores::Scores;
pub use stream::ScoreStream;
pub use thresholds::{Threshold, Thresholds};

/// Loads the model `name` that the build script of the calling crate wrote
/// to `OUT_DIR`, and returns the result of
/// [`Model::from_bytes`](model::Model::from_bytes).
///
/// The bytes of the model become part of the compiled crate. The name is a
/// string literal, the one given to `build::ModelBuild::new` of the `build`
/// feature. This macro needs no feature. The calling crate must have a
/// build script, because Cargo sets `OUT_DIR` only then.
///
/// ```ignore
/// use string2fingerprint::Scorer;
///
/// let model = string2fingerprint::include_model!("waf").expect("the build script validated the model");
/// let scorer = Scorer::with_model(model);
/// ```
#[macro_export]
macro_rules! include_model {
    ($name:literal) => {
        $crate::model::Model::from_bytes(::core::include_bytes!(::core::concat!(
            ::core::env!("OUT_DIR"),
            "/",
            $name,
            ".s2fp"
        )))
    };
}

/// Loads the thresholds of the model `name` that the build script of the
/// calling crate wrote to `OUT_DIR`, and returns the result of
/// [`Thresholds::from_bytes`].
///
/// The build script writes thresholds when it calls
/// `build::ModelBuild::calibrate` of the `build` feature. This macro needs
/// no feature, and its value can initialize a constant:
///
/// ```ignore
/// use string2fingerprint::Thresholds;
///
/// const THRESHOLDS: Thresholds = match string2fingerprint::include_thresholds!("waf") {
///     Some(thresholds) => thresholds,
///     None => panic!("the build script wrote valid thresholds"),
/// };
/// ```
#[macro_export]
macro_rules! include_thresholds {
    ($name:literal) => {
        $crate::Thresholds::from_bytes(::core::include_bytes!(::core::concat!(
            ::core::env!("OUT_DIR"),
            "/",
            $name,
            ".thresholds"
        )))
    };
}
