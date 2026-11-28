//! The asynchronous counterpart of [`Scorer`].

use core::future::Future;
use std::io;
use std::sync::Arc;
use std::thread;

use ::tokio::io::AsyncRead;

use super::offload::Job;
use super::{AsyncScoreStream, DEFAULT_INLINE_LIMIT, Offload, Parallel};
use crate::fingerprint::{BLOCK_LEN, MIN_INPUT_LEN};
use crate::model::{EmbeddedModel, Logits, Model};
use crate::{Scorer, Scores, WindowScores};

/// Scores untrusted bytes on a Tokio runtime without holding a runtime
/// worker: the asynchronous counterpart of [`Scorer`].
///
/// An async scorer is a shared [`Scorer`] and an [inline
/// limit](Self::with_inline_limit). Cloning it clones an [`Arc`], so create
/// one and clone it into each task. Every method that takes input returns a
/// future that owns the input and borrows nothing, so it can be spawned.
/// An input of at most the inline limit is scored in the polling task when
/// the future is first polled; a longer one is scored on Tokio's blocking
/// pool, starting when the method is called. The results equal those of
/// the [`Scorer`] methods of the same name.
///
/// ```
/// use string2fingerprint::tokio::AsyncScorer;
/// use string2fingerprint::Category;
///
/// # #[tokio::main(flavor = "current_thread")]
/// # async fn main() {
/// let scorer = AsyncScorer::new();
/// let scores = scorer.score(b"/search?q=1%27%20OR%20%271%27=%271").await;
/// if scores.get(Category::SqlInjection) >= 192 {
///     // Act on the request, for example log or block it.
/// }
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct AsyncScorer {
    scorer: Arc<Scorer>,
    inline_limit: usize,
}

impl Default for AsyncScorer {
    fn default() -> Self {
        Self::new()
    }
}

impl From<Scorer> for AsyncScorer {
    fn from(scorer: Scorer) -> Self {
        Self::from(Arc::new(scorer))
    }
}

impl From<Arc<Scorer>> for AsyncScorer {
    fn from(scorer: Arc<Scorer>) -> Self {
        Self {
            scorer,
            inline_limit: DEFAULT_INLINE_LIMIT,
        }
    }
}

impl From<Model> for AsyncScorer {
    fn from(model: Model) -> Self {
        Self::from(Scorer::with_model(model))
    }
}

impl From<EmbeddedModel> for AsyncScorer {
    fn from(model: EmbeddedModel) -> Self {
        Self::from(Scorer::with_embedded(model))
    }
}

impl AsyncScorer {
    /// Creates an async scorer of [`Scorer::new`]: the embedded default
    /// model on the preferred backend.
    ///
    /// Each call parses the model and allocates about 200 KiB, in the
    /// calling task. Create one and clone it.
    #[must_use]
    pub fn new() -> Self {
        Self::from(Scorer::new())
    }

    /// Sets the inline limit: the longest input, in bytes, that is scored
    /// in the polling task. Longer inputs are scored on Tokio's blocking
    /// pool. The default is [`DEFAULT_INLINE_LIMIT`].
    ///
    /// With an AVX-512 VNNI backend, an input of one window (512 bytes)
    /// takes about 10 microseconds and 64 KiB about 1.6 ms; the scalar
    /// backend takes several times longer. 0 sends every input with a
    /// window to the blocking pool, and `usize::MAX` scores every input in
    /// the polling task.
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

    /// Returns the shared scorer, whose methods score in the calling task.
    #[must_use]
    pub const fn scorer(&self) -> &Arc<Scorer> {
        &self.scorer
    }

    /// Returns the shared scorer.
    #[must_use]
    pub fn into_scorer(self) -> Arc<Scorer> {
        self.scorer
    }

    /// Runs `score` with the scorer, in the polling task when `cost` is at
    /// most the inline limit and on the blocking pool otherwise.
    fn run<O: Send + 'static>(
        &self,
        cost: usize,
        score: impl FnOnce(&Scorer) -> O + Send + 'static,
    ) -> Offload<O> {
        let scorer = Arc::clone(&self.scorer);
        Offload::new(Job::sized(cost, self.inline_limit, move || score(&scorer)))
    }

    /// Scores `input`. See [`Scorer::score`].
    pub fn score<T>(&self, input: T) -> Offload<Scores>
    where
        T: AsRef<[u8]> + Send + 'static,
    {
        let len = input.as_ref().len();
        self.run(len, move |scorer| scorer.score(input.as_ref()))
    }

    /// Returns the logits of `input`. See [`Scorer::logits`].
    pub fn logits<T>(&self, input: T) -> Offload<Logits>
    where
        T: AsRef<[u8]> + Send + 'static,
    {
        let len = input.as_ref().len();
        self.run(len, move |scorer| scorer.logits(input.as_ref()))
    }

    /// Returns the scores of every window of `input`, in input order. See
    /// [`Scorer::windows`].
    pub fn windows<T>(&self, input: T) -> Offload<Vec<WindowScores>>
    where
        T: AsRef<[u8]> + Send + 'static,
    {
        let len = input.as_ref().len();
        self.run(len, move |scorer| scorer.windows(input.as_ref()).collect())
    }

    /// Scores the fields of one message apart and returns the highest
    /// score of any field in each category. See [`Scorer::score_fields`].
    ///
    /// The fields are collected when the method is called, and their
    /// total cost decides where they are scored: each field of at least
    /// [`MIN_INPUT_LEN`] bytes counts as at least 512 bytes, one window,
    /// because it costs at least one window to score however short it is.
    pub fn score_fields<I>(&self, fields: I) -> Offload<Scores>
    where
        I: IntoIterator,
        I::Item: AsRef<[u8]> + Send + 'static,
    {
        let (fields, cost) = collect(fields);
        self.run(cost, move |scorer| scorer.score_fields(fields))
    }

    /// Scores each input and returns the scores in input order. See
    /// [`Scorer::score_batch`].
    ///
    /// The inputs are collected when the method is called and scored one
    /// after another, in the polling task when their total cost, counted
    /// as for [`score_fields`](Self::score_fields), is at most the inline
    /// limit, and in one job on the blocking pool otherwise. [`score_parallel`](Self::score_parallel) spreads them
    /// over several threads of the pool.
    pub fn score_batch<I>(&self, inputs: I) -> Offload<Vec<Scores>>
    where
        I: IntoIterator,
        I::Item: AsRef<[u8]> + Send + 'static,
    {
        let (inputs, cost) = collect(inputs);
        self.run(cost, move |scorer| scorer.score_batch(inputs))
    }

    /// Scores each input in up to `threads` jobs on Tokio's blocking pool
    /// at once, and returns the scores in input order, which equal those
    /// of [`Scorer::score_batch`].
    ///
    /// `threads` of 0 uses [`std::thread::available_parallelism`]. Each
    /// job scores one contiguous share of the inputs. Inputs whose total
    /// cost is at most the inline limit are scored in the polling task,
    /// like [`score_batch`](Self::score_batch). The pool runs as many jobs
    /// at once as it has threads to spare; Tokio's
    /// [`max_blocking_threads`](https://docs.rs/tokio/1/tokio/runtime/struct.Builder.html#method.max_blocking_threads)
    /// bounds them.
    pub fn score_parallel<T>(&self, inputs: Vec<T>, threads: usize) -> Parallel<Scores>
    where
        T: AsRef<[u8]> + Send + 'static,
    {
        let cost = total_cost(&inputs);
        let jobs: Vec<Job<Vec<Scores>>> = if cost <= self.inline_limit {
            let scorer = Arc::clone(&self.scorer);
            vec![Job::inline(move || scorer.score_batch(inputs))]
        } else {
            shares(inputs, threads)
                .into_iter()
                .map(|share| {
                    let scorer = Arc::clone(&self.scorer);
                    Job::blocking(move || scorer.score_batch(share))
                })
                .collect()
        };
        Parallel::new(jobs)
    }

    /// Reads `reader` to its end without holding the whole input, and
    /// returns [`Scorer::score`] of the bytes read.
    ///
    /// The bytes go through an [`AsyncScoreStream`]: each 64 KiB is scored
    /// on Tokio's blocking pool while the next is read. Work is linear in
    /// the number of bytes read, and a reader that never ends is read
    /// forever, so bound an untrusted source, for example with
    /// [`AsyncReadExt::take`](https://docs.rs/tokio/1/tokio/io/trait.AsyncReadExt.html#method.take):
    ///
    /// ```
    /// use string2fingerprint::tokio::AsyncScorer;
    /// use tokio::io::AsyncReadExt;
    ///
    /// # #[tokio::main(flavor = "current_thread")]
    /// # async fn main() -> std::io::Result<()> {
    /// let body: &[u8] = b"comment=<script>alert(1)</script>";
    /// let scores = AsyncScorer::new().score_reader(body.take(1 << 20)).await?;
    /// # let _ = scores;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns the first error of `reader` other than
    /// [`io::ErrorKind::Interrupted`], which is retried, and the bytes read
    /// so far are not scored.
    pub fn score_reader<R>(&self, reader: R) -> impl Future<Output = io::Result<Scores>> + use<R>
    where
        R: AsyncRead + Unpin,
    {
        let mut stream = self.stream();
        async move { stream.read_to_end(reader).await }
    }

    /// Returns a stream that scores an input that arrives in parts with
    /// this scorer. See [`AsyncScoreStream`].
    #[must_use]
    pub fn stream(&self) -> AsyncScoreStream {
        AsyncScoreStream::new(Arc::clone(&self.scorer), self.inline_limit)
    }

    /// Runs [`Scorer::kernel_self_test`] on Tokio's blocking pool.
    pub fn kernel_self_test(&self) -> Offload<bool> {
        let scorer = Arc::clone(&self.scorer);
        Offload::new(Job::blocking(move || scorer.kernel_self_test()))
    }
}

/// The length of one window, which an input of at least [`MIN_INPUT_LEN`]
/// bytes costs at least.
const WINDOW_LEN: usize = 2 * BLOCK_LEN;

/// Collects `items` and returns them with their total cost.
pub(crate) fn collect<I>(items: I) -> (Vec<I::Item>, usize)
where
    I: IntoIterator,
    I::Item: AsRef<[u8]>,
{
    let items: Vec<I::Item> = items.into_iter().collect();
    let cost = total_cost(&items);
    (items, cost)
}

/// Returns the cost of scoring each of `items`, in bytes of input: the
/// length of each, but at least one window for one that has a window and
/// at least one byte for any other, so that a batch of many empty inputs
/// counts too, saturating at `usize::MAX`.
pub(crate) fn total_cost<T: AsRef<[u8]>>(items: &[T]) -> usize {
    items.iter().fold(0usize, |cost, item| {
        let len = item.as_ref().len();
        let item_cost = if len >= MIN_INPUT_LEN {
            len.max(WINDOW_LEN)
        } else {
            1
        };
        cost.saturating_add(item_cost)
    })
}

/// Splits `items` into up to `threads` contiguous shares of nearly equal
/// count, with 0 threads meaning the available parallelism.
pub(crate) fn shares<T>(items: Vec<T>, threads: usize) -> Vec<Vec<T>> {
    let threads = if threads == 0 {
        thread::available_parallelism().map_or(1, usize::from)
    } else {
        threads
    };
    let threads = threads.min(items.len()).max(1);
    let share = items.len().div_ceil(threads).max(1);
    let mut items = items.into_iter();
    let mut shares = Vec::with_capacity(threads);
    loop {
        let next: Vec<T> = items.by_ref().take(share).collect();
        if next.is_empty() {
            break;
        }
        shares.push(next);
    }
    shares
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shares_keep_the_order_and_cover_every_item() {
        for len in [0usize, 1, 2, 7, 8, 9, 100] {
            for threads in [0usize, 1, 2, 3, 8, 1000] {
                let shares = shares((0..len).collect(), threads);
                assert!(shares.iter().all(|share| !share.is_empty()));
                if threads != 0 {
                    assert!(shares.len() <= threads);
                }
                assert_eq!(shares.concat(), (0..len).collect::<Vec<_>>());
            }
        }
    }

    #[test]
    fn each_input_with_a_window_costs_at_least_one_window() {
        assert_eq!(total_cost::<&[u8]>(&[]), 0);
        assert_eq!(total_cost(&[&b""[..], b"a"]), 2);
        assert_eq!(total_cost(&[&b"ab"[..], b"cde"]), 2 * WINDOW_LEN);
        assert_eq!(total_cost(&[vec![0u8; 1000]]), 1000);
        // 256 fields of two bytes are 256 windows, not 512 bytes.
        assert_eq!(total_cost(&vec![&b"ab"[..]; 256]), 256 * WINDOW_LEN);
    }
}
