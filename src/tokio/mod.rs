//! Asynchronous scoring and detection for [Tokio](https://tokio.rs), with
//! the `tokio` feature.
//!
//! Scoring is CPU-bound: with an AVX-512 VNNI backend, about 10
//! microseconds for an input of one window (512 bytes) and 1.6 ms for
//! 64 KiB, and several times longer with the scalar backend. Run from an
//! asynchronous task, a long input would hold a runtime worker that long and
//! stall every other task on it. The items here run short work in the polling task and the rest
//! on Tokio's [blocking pool], and read and write through Tokio's I/O
//! traits:
//!
//! | Synchronous | Asynchronous | What it adds |
//! | --- | --- | --- |
//! | [`Detector`](crate::Detector) | [`AsyncDetector`] | a shared detector; [`Offload`] and [`Parallel`] futures that own their input |
//! | [`Scorer`](crate::Scorer) | [`AsyncScorer`] | a shared scorer; the same futures |
//! | [`DetectionStream`](crate::DetectionStream), [`ScoreStream`](crate::ScoreStream) | [`AsyncDetectionStream`], [`AsyncScoreStream`] | [`AsyncWrite`]; long parts scored on the blocking pool while the next arrives |
//! | `detect_reader`, `score_reader` on [`io::Read`](std::io::Read) | [`AsyncDetector::detect_reader`], [`AsyncScorer::score_reader`] on [`AsyncRead`] | the same pipeline |
//! | `detect_parallel`, `score_parallel` on scoped threads | the same names on the blocking pool | |
//! | [`fingerprint::Fingerprinter`](crate::fingerprint::Fingerprinter) | [`fingerprint::FingerprintReader`] | windows of an [`AsyncRead`], one at a time |
//! | [`Model::from_bytes`](crate::model::Model::from_bytes) | [`model::from_bytes`], [`model::read`] | parsing and file reading on the blocking pool |
//!
//! Everything else, such as thresholds, presets, builders, kernels, and the
//! results ([`Scores`](crate::Scores), [`Detection`](crate::Detection)), is
//! shared with the synchronous interface. A synchronous [`Detector`] built
//! with [`Detector::builder`] becomes an async one with
//! [`AsyncDetector::from`].
//!
//! ```toml
//! [dependencies]
//! string2fingerprint = { git = "https://github.com/OxiBelt/string2fingerprint", features = ["tokio"] }
//! # The runtime stays the application's own dependency.
//! tokio = { version = "1.54.1", features = ["rt-multi-thread", "macros", "net", "io-util"] }
//! ```
//!
//! ```
//! use string2fingerprint::tokio::AsyncDetector;
//! use string2fingerprint::{Category, Preset};
//!
//! # #[tokio::main(flavor = "current_thread")]
//! # async fn main() {
//! let detector = AsyncDetector::with_preset(Preset::Balanced);
//! let worker = detector.clone();
//! // The future owns its input and the detector, so it can be spawned.
//! let detection = tokio::spawn(async move {
//!     worker.detect(b"/search?q=1%27%20OR%20%271%27=%271".to_vec()).await
//! })
//! .await
//! .expect("the task completes");
//! assert!(detection.contains(Category::SqlInjection));
//! # }
//! ```
//!
//! # Where work runs
//!
//! [`AsyncScorer`] and [`AsyncDetector`] have an [inline
//! limit](AsyncScorer::with_inline_limit), [`DEFAULT_INLINE_LIMIT`] bytes
//! unless set. An input of at most the limit is scored in the task that
//! polls the future, at its first poll. Fields and batches count their
//! total cost, in which every item with a window counts as at least one
//! window (512 bytes), so many short fields do not run in the task. A
//! longer input is scored on the blocking pool of the runtime that is
//! current when the method is called, starting at once, as
//! [`tokio::task::spawn_blocking`] does; the future waits for the result.
//!
//! The methods return the named futures [`Offload`] and [`Parallel`],
//! which own their input, borrow nothing, and are `Send`, `Sync`, and
//! `Unpin`, so they can be spawned or kept in a struct. Dropping one before
//! a thread of the pool took its work drops the work unrun, so the pool
//! does not run work for callers that went away. Work that started
//! finishes, and its result is dropped.
//!
//! The blocking pool starts threads as work arrives, up to the runtime's
//! [`max_blocking_threads`] (512 unless set), and queues work beyond that.
//! A server that scores large untrusted bodies should bound how many it
//! scores at once, as it bounds connections, or lower that limit.
//!
//! Outside a Tokio runtime nothing fails: the futures run every input in
//! the polling task, so another executor can drive them, holding its
//! thread meanwhile. If the runtime shuts down before the pool started a
//! job, the future runs the job itself when it is polled again.
//!
//! # Streams and readers
//!
//! [`AsyncScoreStream`] and [`AsyncDetectionStream`] buffer what is written
//! to them, through their `update` methods or [`AsyncWrite`]. While no part
//! is being scored, they score buffered bytes of at most the inline limit
//! at once and send more to the blocking pool; while a part is scored
//! there, up to 64 KiB of the next part is buffered. So their progress
//! follows the writes, for an early stop, and a stream holds at most two
//! parts. Their `finish` methods wait for the rest and return the result of
//! the whole input, which equals that of the synchronous streams however
//! the input was split. [`AsyncScorer::score_reader`] and
//! [`AsyncDetector::detect_reader`] read an [`AsyncRead`] into such a
//! stream, so reading and scoring overlap.
//!
//! [`fingerprint::FingerprintReader`] is the low-level counterpart: it
//! returns the fingerprint of each window of an [`AsyncRead`] as soon as
//! the window is complete, for callers that run the model by hand.
//!
//! # Cooperative scheduling
//!
//! Work in the polling task takes part in Tokio's [cooperative
//! scheduling], as Tokio's own sockets and channels do: each piece of work
//! that completes, and each write to a stream, takes one unit of the task's
//! budget, and once the budget
//! is spent the future returns `Pending` and runs after the other tasks had
//! their turn. So a loop over short inputs, or a reader that is always
//! ready, cannot keep a worker to itself. Outside a Tokio runtime the
//! budget is unlimited.
//!
//! # Errors
//!
//! Scoring and deciding never fail, as in the synchronous interface. Only
//! I/O fails: the readers return the errors of the inner reader as they
//! are, and [`model::read`] returns `io::Error`s, where a file that is not
//! a valid model has kind [`InvalidData`](std::io::ErrorKind::InvalidData)
//! and wraps the [`ModelError`](crate::model::ModelError).
//!
//! A panic in work on the blocking pool, which the input contract rules
//! out, resumes in the task that awaits the future.
//!
//! # Features
//!
//! The `tokio` feature depends on Tokio with its `rt` feature, for the
//! blocking pool and the cooperative budget. That adds no dependency, and
//! an application that runs a Tokio runtime enables it already. Tokio's
//! I/O traits need no feature. The runtime, its flavor and its other
//! features are the application's choice.
//!
//! [blocking pool]: https://docs.rs/tokio/1/tokio/task/fn.spawn_blocking.html
//! [`tokio::task::spawn_blocking`]: https://docs.rs/tokio/1/tokio/task/fn.spawn_blocking.html
//! [`max_blocking_threads`]: https://docs.rs/tokio/1/tokio/runtime/struct.Builder.html#method.max_blocking_threads
//! [cooperative scheduling]: https://docs.rs/tokio/1/tokio/task/coop/index.html
//! [`AsyncRead`]: https://docs.rs/tokio/1/tokio/io/trait.AsyncRead.html
//! [`AsyncWrite`]: https://docs.rs/tokio/1/tokio/io/trait.AsyncWrite.html
//! [`Detector`]: crate::Detector
//! [`Detector::builder`]: crate::Detector::builder

mod detector;
pub mod fingerprint;
pub mod model;
mod offload;
mod pipeline;
mod scorer;
mod stream;

pub use detector::AsyncDetector;
pub use offload::{Offload, Parallel};
pub use scorer::AsyncScorer;
pub use stream::{AsyncDetectionStream, AsyncScoreStream};

/// The default inline limit of [`AsyncScorer`] and [`AsyncDetector`]: the
/// longest input, in bytes, that is scored in the polling task.
///
/// An input of 512 bytes is one window, which takes about 10 microseconds
/// with an AVX-512 VNNI backend and about 50 with the scalar one, within
/// the time that Tokio recommends between two points at which a task
/// yields.
pub const DEFAULT_INLINE_LIMIT: usize = 512;
