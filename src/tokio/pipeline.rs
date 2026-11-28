//! The buffering that the asynchronous streams share: bytes are gathered
//! into chunks, and each chunk is scored on the blocking pool while the
//! next one is gathered.

use core::mem;
use core::pin::Pin;
use core::task::{Context, Poll, ready};
use std::sync::Arc;

use ::tokio::task::coop;

use super::offload::Job;
use crate::detect::{Detection, DetectionStream, Detector};
use crate::{ScoreStream, Scorer, Scores};

/// The most bytes that a stream gathers while a job runs, and so the most
/// that one job scores: about 1.6 ms of work with a SIMD backend.
pub(crate) const CHUNK_LEN: usize = 64 * 1024;

/// A synchronous stream that a [`Pipeline`] feeds.
pub(crate) trait Sink: Send + 'static {
    /// What the stream knows of the windows completed so far.
    type Progress: Copy + Send + 'static;

    fn update(&mut self, part: &[u8]);

    fn progress(&self) -> Self::Progress;
}

impl Sink for ScoreStream<Arc<Scorer>> {
    type Progress = Scores;

    fn update(&mut self, part: &[u8]) {
        Self::update(self, part);
    }

    fn progress(&self) -> Scores {
        self.scores()
    }
}

impl Sink for DetectionStream<Arc<Detector>> {
    type Progress = Detection;

    fn update(&mut self, part: &[u8]) {
        Self::update(self, part);
    }

    fn progress(&self) -> Detection {
        self.detection()
    }
}

/// Feeds a [`Sink`] from asynchronous writes.
///
/// A write copies bytes into `pending`. While no job runs, pending bytes of
/// at most the inline limit go to the sink at once, and more move, with
/// the sink, into a job on the blocking pool, which hands both back when it
/// is done. While a job runs, writes gather up to one chunk. Only one job
/// runs at a time, so the sink sees the bytes in order.
pub(crate) struct Pipeline<S: Sink> {
    /// `None` while a job holds it.
    sink: Option<S>,
    job: Option<Job<(S, Vec<u8>)>>,
    /// Bytes written and not yet given to a job.
    pending: Vec<u8>,
    /// The buffer of the last job, kept for the next chunk.
    spare: Vec<u8>,
    /// The progress of the sink when it was last here.
    progress: S::Progress,
    inline_limit: usize,
    /// Bytes written since the input started.
    len: u64,
}

impl<S: Sink> Pipeline<S> {
    pub(crate) fn new(sink: S, inline_limit: usize) -> Self {
        Self {
            progress: sink.progress(),
            sink: Some(sink),
            job: None,
            pending: Vec::new(),
            spare: Vec::new(),
            inline_limit,
            len: 0,
        }
    }

    /// Waits for the running job, if any, and takes the sink back.
    fn poll_job(&mut self, cx: &mut Context<'_>) -> Poll<()> {
        if let Some(job) = &mut self.job {
            let (sink, mut buffer) = ready!(Pin::new(job).poll(cx));
            self.job = None;
            self.progress = sink.progress();
            self.sink = Some(sink);
            buffer.clear();
            self.spare = buffer;
        }
        Poll::Ready(())
    }

    /// Gives the pending bytes and the sink to a new job on the blocking
    /// pool. The sink must be here and no job running.
    fn start(&mut self) {
        let Some(mut sink) = self.sink.take() else {
            unreachable!("a job started while another runs");
        };
        let chunk = mem::replace(&mut self.pending, mem::take(&mut self.spare));
        self.job = Some(Job::blocking(move || {
            sink.update(&chunk);
            (sink, chunk)
        }));
    }

    /// Passes the pending bytes on while no job runs: to the sink here when
    /// they are at most the inline limit, and to a new job otherwise.
    fn advance(&mut self) {
        if self.job.is_some() || self.pending.is_empty() {
            return;
        }
        if self.pending.len() > self.inline_limit {
            self.start();
            return;
        }
        let Some(sink) = &mut self.sink else {
            unreachable!("no job runs, so the sink is here");
        };
        sink.update(&self.pending);
        self.progress = sink.progress();
        self.pending.clear();
    }

    /// Accepts up to the rest of the current chunk from `part`, and
    /// returns the number of bytes accepted, at least one unless `part` is
    /// empty.
    ///
    /// A write takes one unit of the task's cooperative budget. It takes
    /// back a job that finished, and passes the pending bytes on when no
    /// job runs, so the progress follows the writes as closely as the
    /// inline limit allows.
    pub(crate) fn poll_write(&mut self, cx: &mut Context<'_>, part: &[u8]) -> Poll<usize> {
        if part.is_empty() {
            return Poll::Ready(0);
        }
        let coop = ready!(coop::poll_proceed(cx));
        if self.job.is_some() {
            // Without waiting: a job that runs on stays where it is.
            let _ = self.poll_job(cx);
        }
        while self.pending.len() >= CHUNK_LEN {
            if self.job.is_some() {
                ready!(self.poll_job(cx));
            } else {
                self.start();
            }
        }
        if self.pending.capacity() == 0 {
            self.pending.reserve_exact(CHUNK_LEN);
        }
        let take = part.len().min(CHUNK_LEN - self.pending.len());
        self.pending.extend_from_slice(&part[..take]);
        // A slice is shorter than `u64::MAX` bytes on every target.
        self.len = self.len.saturating_add(take as u64);
        self.advance();
        coop.made_progress();
        Poll::Ready(take)
    }

    /// Waits until every byte written so far went through the sink.
    pub(crate) fn poll_flush(&mut self, cx: &mut Context<'_>) -> Poll<()> {
        loop {
            if self.job.is_some() {
                ready!(self.poll_job(cx));
            } else if self.pending.is_empty() {
                return Poll::Ready(());
            } else {
                let coop = ready!(coop::poll_proceed(cx));
                self.advance();
                coop.made_progress();
            }
        }
    }

    /// Ends the input with `finish`, which ends the input of the sink, and
    /// starts a new one. Call it after [`Pipeline::poll_flush`] returned.
    pub(crate) fn finish<T>(&mut self, finish: impl FnOnce(&mut S) -> T) -> T {
        let Some(sink) = &mut self.sink else {
            unreachable!("the input ended before the flush returned");
        };
        let result = finish(sink);
        self.progress = sink.progress();
        self.len = 0;
        result
    }

    /// Returns the progress of the sink when it was last here.
    pub(crate) fn progress(&self) -> S::Progress {
        self.progress
    }

    /// Returns the number of bytes written since the input started.
    pub(crate) fn len(&self) -> u64 {
        self.len
    }

    /// Discards the input so far and starts a new input with `reset` on the
    /// sink, or on the sink that `fresh` returns when a job held it. A job
    /// that no thread of the pool took yet is dropped; one that started
    /// finishes, and its result is dropped.
    pub(crate) fn reset(&mut self, reset: impl FnOnce(&mut S), fresh: impl FnOnce() -> S) {
        self.job = None;
        self.pending.clear();
        let sink = self.sink.get_or_insert_with(fresh);
        reset(sink);
        self.progress = sink.progress();
        self.len = 0;
    }
}
