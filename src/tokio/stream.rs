//! Scoring and deciding an input that arrives in parts through
//! asynchronous writes.

use core::fmt;
use core::future::poll_fn;
use core::pin::Pin;
use core::task::{Context, Poll};
use std::io;
use std::sync::Arc;

use ::tokio::io::{AsyncRead, AsyncWrite, ReadBuf};

use super::pipeline::{Pipeline, Sink};
use crate::detect::{Detection, DetectionStream, Detector};
use crate::{ScoreStream, Scorer, Scores};

/// The size of the buffer that the `*_reader` methods read into.
const READ_LEN: usize = 8192;

/// Writes `part` whole into `pipeline`.
async fn write_all<S: Sink>(pipeline: &mut Pipeline<S>, mut part: &[u8]) {
    while !part.is_empty() {
        let written = poll_fn(|cx| pipeline.poll_write(cx, part)).await;
        part = &part[written..];
    }
}

/// Waits until every byte written to `pipeline` went through its sink.
async fn flush<S: Sink>(pipeline: &mut Pipeline<S>) {
    poll_fn(|cx| pipeline.poll_flush(cx)).await;
}

/// Writes everything that `reader` holds into `pipeline` and flushes it.
pub(crate) async fn copy<R, S>(mut reader: R, pipeline: &mut Pipeline<S>) -> io::Result<()>
where
    R: AsyncRead + Unpin,
    S: Sink,
{
    let mut buffer = vec![0u8; READ_LEN];
    loop {
        let mut read = ReadBuf::new(&mut buffer);
        match poll_fn(|cx| Pin::new(&mut reader).poll_read(cx, &mut read)).await {
            Ok(()) if read.filled().is_empty() => break,
            // Each write takes a unit of the cooperative budget, so a reader
            // that is always ready, such as a byte slice, cannot keep the
            // worker to itself.
            Ok(()) => write_all(pipeline, read.filled()).await,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    flush(pipeline).await;
    Ok(())
}

/// Scores an input that arrives in parts, such as a request body, on a
/// Tokio runtime: the asynchronous counterpart of [`ScoreStream`].
///
/// [`update`](Self::update) and the [`AsyncWrite`] implementation copy the
/// bytes into a buffer and return. While no part is being scored, buffered
/// bytes of at most the [inline
/// limit](super::AsyncScorer::with_inline_limit) are scored at once in the
/// writing task, and more are scored on Tokio's blocking pool. While a part
/// is scored there, writes gather up to 64 KiB for the next part, and a
/// write waits only when that is full. Parts are scored one at a time and in
/// order, so the stream holds at most two parts. [`finish`](Self::finish)
/// waits for the rest and returns [`Scorer::score`] of the whole input,
/// however it was split. Every write takes one unit of the task's
/// cooperative budget.
///
/// [`scores`](Self::scores) returns the scores of the bytes scored so far,
/// which never decrease while the input grows and are at most those of
/// `finish`. Bytes that are still buffered, or on the blocking pool, are not
/// in them yet; a write takes back a part that the pool finished. Flushing,
/// with
/// [`AsyncWriteExt::flush`](https://docs.rs/tokio/1/tokio/io/trait.AsyncWriteExt.html#method.flush),
/// waits until every byte written is in them.
///
/// The stream never fails: its writes accept at least one byte, flushing
/// and shutting it down succeed, and shutting it down only flushes it. It is
/// `Send` and `Sync`.
///
/// ```
/// use string2fingerprint::tokio::AsyncScorer;
/// use string2fingerprint::Scorer;
///
/// # #[tokio::main(flavor = "current_thread")]
/// # async fn main() {
/// let scorer = AsyncScorer::new();
/// let body = b"name=alice&comment=%3Cscript%3Ealert(document.cookie)%3C%2Fscript%3E";
/// let mut stream = scorer.stream();
/// for part in body.chunks(16) {
///     stream.update(part).await;
/// }
/// assert_eq!(stream.finish().await, Scorer::new().score(body));
/// # }
/// ```
///
/// [`AsyncWrite`]: https://docs.rs/tokio/1/tokio/io/trait.AsyncWrite.html
pub struct AsyncScoreStream {
    scorer: Arc<Scorer>,
    pipeline: Pipeline<ScoreStream<Arc<Scorer>>>,
}

/// Shows the progress and nothing derived from the input.
impl fmt::Debug for AsyncScoreStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AsyncScoreStream")
            .field("len", &self.pipeline.len())
            .finish_non_exhaustive()
    }
}

impl AsyncScoreStream {
    pub(crate) fn new(scorer: Arc<Scorer>, inline_limit: usize) -> Self {
        Self {
            pipeline: Pipeline::new(ScoreStream::new(Arc::clone(&scorer)), inline_limit),
            scorer,
        }
    }

    /// Consumes the next part of the input. The call returns once every
    /// byte of `part` is buffered or scored, and never fails.
    ///
    /// Cancelling it keeps the bytes that it buffered so far, as
    /// [`AsyncWriteExt::write_all`] does; [`len`](Self::len) tells how
    /// many bytes the stream consumed in all.
    ///
    /// [`AsyncWriteExt::write_all`]: https://docs.rs/tokio/1/tokio/io/trait.AsyncWriteExt.html#method.write_all
    pub async fn update(&mut self, part: &[u8]) {
        write_all(&mut self.pipeline, part).await;
    }

    /// Returns, in each category, the highest score of the windows scored
    /// so far, or [`Scores::ZERO`] before the first one.
    #[must_use]
    pub fn scores(&self) -> Scores {
        self.pipeline.progress()
    }

    /// Ends the input, waits until all of it is scored, returns its scores,
    /// and starts a new input.
    pub async fn finish(&mut self) -> Scores {
        flush(&mut self.pipeline).await;
        self.pipeline.finish(ScoreStream::finish)
    }

    /// Discards the input so far and starts a new input. Work for the old
    /// input that no thread of the blocking pool took yet is dropped; work
    /// that started finishes, and its result is dropped.
    pub fn reset(&mut self) {
        let scorer = &self.scorer;
        self.pipeline
            .reset(ScoreStream::reset, || ScoreStream::new(Arc::clone(scorer)));
    }

    /// Returns the number of bytes consumed since the input started.
    #[must_use]
    pub fn len(&self) -> u64 {
        self.pipeline.len()
    }

    /// Returns whether no byte was consumed since the input started.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pipeline.len() == 0
    }

    /// Returns the scorer.
    #[must_use]
    pub fn scorer(&self) -> &Arc<Scorer> {
        &self.scorer
    }

    /// Returns the scorer and drops the input so far, as
    /// [`reset`](Self::reset) does.
    #[must_use]
    pub fn into_scorer(self) -> Arc<Scorer> {
        self.scorer
    }
}

/// Every write accepts at least one byte unless the buffer is empty, and
/// nothing fails.
impl AsyncWrite for AsyncScoreStream {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buffer: &[u8],
    ) -> Poll<io::Result<usize>> {
        self.get_mut().pipeline.poll_write(cx, buffer).map(Ok)
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        self.get_mut().pipeline.poll_flush(cx).map(Ok)
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        self.poll_flush(cx)
    }
}

/// Decides an input that arrives in parts on a Tokio runtime: the
/// asynchronous counterpart of [`DetectionStream`], on top of the buffering
/// of [`AsyncScoreStream`].
///
/// [`detection`](Self::detection) decides the bytes scored so far, which
/// follow the writes as described for [`AsyncScoreStream::scores`]. A
/// category that it flags stays flagged, so a caller can stop reading as
/// soon as it is flagged. [`finish`](Self::finish) decides the whole input
/// as [`Detector::detect`] would.
///
/// ```
/// use string2fingerprint::tokio::AsyncDetector;
///
/// # #[tokio::main(flavor = "current_thread")]
/// # async fn main() {
/// let detector = AsyncDetector::new();
/// let mut stream = detector.stream();
/// for part in [&b"cmd=ping%20-c%201%20"[..], b"127.0.0.1%3Bcat%20/etc/passwd"] {
///     stream.update(part).await;
///     if stream.detection().is_flagged() {
///         break;
///     }
/// }
/// assert!(stream.finish().await.is_flagged());
/// # }
/// ```
pub struct AsyncDetectionStream {
    detector: Arc<Detector>,
    pipeline: Pipeline<DetectionStream<Arc<Detector>>>,
}

/// Shows the progress and nothing derived from the input.
impl fmt::Debug for AsyncDetectionStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AsyncDetectionStream")
            .field("len", &self.pipeline.len())
            .finish_non_exhaustive()
    }
}

impl AsyncDetectionStream {
    pub(crate) fn new(detector: Arc<Detector>, inline_limit: usize) -> Self {
        Self {
            pipeline: Pipeline::new(DetectionStream::new(Arc::clone(&detector)), inline_limit),
            detector,
        }
    }

    /// Consumes the next part of the input. See
    /// [`AsyncScoreStream::update`].
    pub async fn update(&mut self, part: &[u8]) {
        write_all(&mut self.pipeline, part).await;
    }

    /// Decides the bytes scored so far.
    #[must_use]
    pub fn detection(&self) -> Detection {
        self.pipeline.progress()
    }

    /// Ends the input, waits until all of it is scored, decides it, and
    /// starts a new input.
    pub async fn finish(&mut self) -> Detection {
        flush(&mut self.pipeline).await;
        self.pipeline.finish(DetectionStream::finish)
    }

    /// Discards the input so far and starts a new input. See
    /// [`AsyncScoreStream::reset`].
    pub fn reset(&mut self) {
        let detector = &self.detector;
        self.pipeline.reset(DetectionStream::reset, || {
            DetectionStream::new(Arc::clone(detector))
        });
    }

    /// Returns the number of bytes consumed since the input started.
    #[must_use]
    pub fn len(&self) -> u64 {
        self.pipeline.len()
    }

    /// Returns whether no byte was consumed since the input started.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pipeline.len() == 0
    }

    /// Returns the detector.
    #[must_use]
    pub fn detector(&self) -> &Arc<Detector> {
        &self.detector
    }

    /// Returns the detector and drops the input so far, as
    /// [`reset`](Self::reset) does.
    #[must_use]
    pub fn into_detector(self) -> Arc<Detector> {
        self.detector
    }
}

/// Every write accepts at least one byte unless the buffer is empty, and
/// nothing fails.
impl AsyncWrite for AsyncDetectionStream {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buffer: &[u8],
    ) -> Poll<io::Result<usize>> {
        self.get_mut().pipeline.poll_write(cx, buffer).map(Ok)
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        self.get_mut().pipeline.poll_flush(cx).map(Ok)
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        self.poll_flush(cx)
    }
}

impl AsyncScoreStream {
    /// Reads `reader` to its end into this stream, and returns the scores
    /// of the whole input as [`finish`](Self::finish) does.
    pub(crate) async fn read_to_end<R: AsyncRead + Unpin>(
        &mut self,
        reader: R,
    ) -> io::Result<Scores> {
        copy(reader, &mut self.pipeline).await?;
        Ok(self.pipeline.finish(ScoreStream::finish))
    }
}

impl AsyncDetectionStream {
    /// Reads `reader` to its end into this stream, and decides the whole
    /// input as [`finish`](Self::finish) does.
    pub(crate) async fn read_to_end<R: AsyncRead + Unpin>(
        &mut self,
        reader: R,
    ) -> io::Result<Detection> {
        copy(reader, &mut self.pipeline).await?;
        Ok(self.pipeline.finish(DetectionStream::finish))
    }
}
