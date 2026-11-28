//! The asynchronous counterpart of [`Detector`].

use core::future::Future;
use std::io;
use std::sync::Arc;

use ::tokio::io::AsyncRead;

use super::offload::Job;
use super::scorer::{collect, shares, total_cost};
use super::{AsyncDetectionStream, AsyncScorer, DEFAULT_INLINE_LIMIT, Offload, Parallel};
use crate::detect::{Detection, Detector, Preset};

/// Decides whether untrusted bytes are flagged on a Tokio runtime without
/// holding a runtime worker: the asynchronous counterpart of [`Detector`].
///
/// An async detector is a shared [`Detector`] and an [inline
/// limit](Self::with_inline_limit), and runs its work as an
/// [`AsyncScorer`] does: an input of at most the inline limit in the
/// polling task, a longer one on Tokio's blocking pool. Cloning it clones
/// an [`Arc`]. The results equal those of the [`Detector`] methods of the
/// same name.
///
/// ```
/// use string2fingerprint::tokio::AsyncDetector;
///
/// # #[tokio::main(flavor = "current_thread")]
/// # async fn main() {
/// let detector = AsyncDetector::new();
/// let detection = detector.detect(b"/files?name=..%2f..%2f..%2fetc%2fpasswd").await;
/// if detection.is_flagged() {
///     // Log or block the request, naming what was flagged.
///     println!("flagged: {detection}");
/// }
/// assert!(!detector.detect(b"/files?name=report-2026.pdf").await.is_flagged());
/// # }
/// ```
///
/// A [`DetectorBuilder`](crate::DetectorBuilder) chooses the model,
/// backend, and thresholds as for the synchronous detector:
///
/// ```
/// use string2fingerprint::tokio::AsyncDetector;
/// use string2fingerprint::{Category, Detector, Preset};
///
/// let detector = AsyncDetector::from(
///     Detector::builder()
///         .preset(Preset::Strict)
///         .disable(Category::Malformed)
///         .build()?,
/// );
/// # Ok::<(), string2fingerprint::ScorerError>(())
/// ```
#[derive(Clone, Debug)]
pub struct AsyncDetector {
    detector: Arc<Detector>,
    inline_limit: usize,
}

impl Default for AsyncDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl From<Detector> for AsyncDetector {
    fn from(detector: Detector) -> Self {
        Self::from(Arc::new(detector))
    }
}

impl From<Arc<Detector>> for AsyncDetector {
    fn from(detector: Arc<Detector>) -> Self {
        Self {
            detector,
            inline_limit: DEFAULT_INLINE_LIMIT,
        }
    }
}

impl From<Preset> for AsyncDetector {
    fn from(preset: Preset) -> Self {
        Self::with_preset(preset)
    }
}

impl AsyncDetector {
    /// Creates an async detector of [`Detector::new`]: [`Preset::Balanced`]
    /// on the preferred backend.
    ///
    /// Each call parses the model and allocates about 200 KiB, in the
    /// calling task. Create one and clone it.
    #[must_use]
    pub fn new() -> Self {
        Self::from(Detector::new())
    }

    /// Creates an async detector of `preset` on the preferred backend.
    #[must_use]
    pub fn with_preset(preset: Preset) -> Self {
        Self::from(Detector::with_preset(preset))
    }

    /// Sets the inline limit: the longest input, in bytes, that is decided
    /// in the polling task. See [`AsyncScorer::with_inline_limit`].
    #[must_use]
    pub const fn with_inline_limit(mut self, bytes: usize) -> Self {
        self.inline_limit = bytes;
        self
    }

    /// Returns the inline limit.
    #[must_use]
    pub const fn inline_limit(&self) -> usize {
        self.inline_limit
    }

    /// Returns the shared detector, whose methods decide in the calling
    /// task.
    #[must_use]
    pub const fn detector(&self) -> &Arc<Detector> {
        &self.detector
    }

    /// Returns the shared detector.
    #[must_use]
    pub fn into_detector(self) -> Arc<Detector> {
        self.detector
    }

    /// Returns an async scorer of a copy of the scorer of this detector,
    /// with the same inline limit.
    ///
    /// The copy allocates about the size of the model. Keep it rather than
    /// calling this for each input.
    #[must_use]
    pub fn to_scorer(&self) -> AsyncScorer {
        AsyncScorer::from(self.detector.scorer().clone()).with_inline_limit(self.inline_limit)
    }

    /// Runs `decide` with the detector, in the polling task when `cost` is
    /// at most the inline limit and on the blocking pool otherwise.
    fn run<O: Send + 'static>(
        &self,
        cost: usize,
        decide: impl FnOnce(&Detector) -> O + Send + 'static,
    ) -> Offload<O> {
        let detector = Arc::clone(&self.detector);
        Offload::new(Job::sized(cost, self.inline_limit, move || {
            decide(&detector)
        }))
    }

    /// Scores `input` and decides the scores. See [`Detector::detect`].
    pub fn detect<T>(&self, input: T) -> Offload<Detection>
    where
        T: AsRef<[u8]> + Send + 'static,
    {
        let len = input.as_ref().len();
        self.run(len, move |detector| detector.detect(input.as_ref()))
    }

    /// Returns whether any category of `input` is flagged.
    pub fn is_flagged<T>(&self, input: T) -> Offload<bool>
    where
        T: AsRef<[u8]> + Send + 'static,
    {
        let len = input.as_ref().len();
        self.run(len, move |detector| detector.is_flagged(input.as_ref()))
    }

    /// Scores the fields of one message apart and decides their combined
    /// scores. See [`Detector::detect_fields`] and
    /// [`AsyncScorer::score_fields`].
    pub fn detect_fields<I>(&self, fields: I) -> Offload<Detection>
    where
        I: IntoIterator,
        I::Item: AsRef<[u8]> + Send + 'static,
    {
        let (fields, cost) = collect(fields);
        self.run(cost, move |detector| detector.detect_fields(fields))
    }

    /// Decides each input and returns the detections in input order. See
    /// [`Detector::detect_batch`] and [`AsyncScorer::score_batch`].
    pub fn detect_batch<I>(&self, inputs: I) -> Offload<Vec<Detection>>
    where
        I: IntoIterator,
        I::Item: AsRef<[u8]> + Send + 'static,
    {
        let (inputs, cost) = collect(inputs);
        self.run(cost, move |detector| detector.detect_batch(inputs))
    }

    /// Decides each input in up to `threads` jobs on Tokio's blocking pool
    /// at once. See [`AsyncScorer::score_parallel`].
    pub fn detect_parallel<T>(&self, inputs: Vec<T>, threads: usize) -> Parallel<Detection>
    where
        T: AsRef<[u8]> + Send + 'static,
    {
        let cost = total_cost(&inputs);
        let jobs: Vec<Job<Vec<Detection>>> = if cost <= self.inline_limit {
            let detector = Arc::clone(&self.detector);
            vec![Job::inline(move || detector.detect_batch(inputs))]
        } else {
            shares(inputs, threads)
                .into_iter()
                .map(|share| {
                    let detector = Arc::clone(&self.detector);
                    Job::blocking(move || detector.detect_batch(share))
                })
                .collect()
        };
        Parallel::new(jobs)
    }

    /// Reads `reader` to its end and decides its scores. See
    /// [`AsyncScorer::score_reader`], and bound an untrusted source.
    ///
    /// # Errors
    ///
    /// Returns the first error of `reader` other than
    /// [`io::ErrorKind::Interrupted`].
    pub fn detect_reader<R>(
        &self,
        reader: R,
    ) -> impl Future<Output = io::Result<Detection>> + use<R>
    where
        R: AsyncRead + Unpin,
    {
        let mut stream = self.stream();
        async move { stream.read_to_end(reader).await }
    }

    /// Returns a stream that decides an input that arrives in parts with
    /// this detector. See [`AsyncDetectionStream`].
    #[must_use]
    pub fn stream(&self) -> AsyncDetectionStream {
        AsyncDetectionStream::new(Arc::clone(&self.detector), self.inline_limit)
    }
}
