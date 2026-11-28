//! Asynchronous scoring and detection for [Tokio](https://tokio.rs), with
//! the `tokio` feature.
//!
//! Scoring is CPU-bound: with an AVX-512 VNNI backend, about 10
//! microseconds for an input of one window (512 bytes) and 1.6 ms for
//! 64 KiB. [`AsyncScorer`] is the asynchronous counterpart of
//! [`Scorer`](crate::Scorer). Their methods run an input of at most the
//! inline limit in the polling task, taking one unit of its cooperative
//! budget, and a longer one on Tokio's blocking pool, and return futures
//! ([`Offload`], [`Parallel`]) that own their input. Outside a Tokio
//! runtime every future runs in the polling task. Results equal those of
//! the synchronous items.
//!
//! The documentation of this module is completed with the rest of the
//! interface.

mod offload;
mod scorer;

pub use offload::{Offload, Parallel};
pub use scorer::AsyncScorer;

/// The default inline limit of [`AsyncScorer`]: the
/// longest input, in bytes, that is scored in the polling task.
///
/// An input of 512 bytes is one window, which takes about 10 microseconds
/// with an AVX-512 VNNI backend and about 50 with the scalar one, within
/// the time that Tokio recommends between two points at which a task
/// yields.
pub const DEFAULT_INLINE_LIMIT: usize = 512;
