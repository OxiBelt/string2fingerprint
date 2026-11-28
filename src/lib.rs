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

#[cfg(not(target_os = "linux"))]
compile_error!("`string2fingerprint` supports Linux targets only");

mod backend;
mod category;
mod category_set;
pub mod eval;
pub mod fingerprint;
mod kernel;
pub mod model;
mod scorer;
mod scores;
#[cfg(test)]
mod testutil;
mod thresholds;

pub use backend::{Backend, UnsupportedBackend};
pub use category::Category;
pub use category_set::{CategorySet, ParseCategorySetError};
pub use scorer::Scorer;
pub use scores::Scores;
pub use thresholds::{Threshold, Thresholds};
