//! Scoring of an input that arrives in parts.

use core::fmt;
use std::io;

use crate::fingerprint::Fingerprinter;
use crate::{Scorer, Scores};

/// Scores an input that arrives in parts, such as a request body read from
/// a socket, without holding the input.
///
/// The result of [`finish`](Self::finish) equals [`Scorer::score`] of the
/// whole input, however it was split. [`scores`](Self::scores) gives the
/// scores of the windows completed so far, which never decrease while the
/// input grows, so a caller can stop reading once they reach a threshold.
///
/// A stream holds a [`Fingerprinter`], about 9 KiB, and no heap memory. It
/// borrows its scorer through `S`, which is `&Scorer` for
/// [`Scorer::stream`] and can be an owned or shared scorer such as
/// `Arc<Scorer>`. It also implements [`io::Write`], so [`io::copy`] can
/// feed it.
///
/// ```
/// use string2fingerprint::{Category, Scorer};
///
/// let scorer = Scorer::new();
/// let body = b"name=alice&comment=%3Cscript%3Ealert(document.cookie)%3C%2Fscript%3E";
/// let mut stream = scorer.stream();
/// for part in body.chunks(16) {
///     stream.update(part);
/// }
/// assert_eq!(stream.finish(), scorer.score(body));
/// ```
#[derive(Clone)]
pub struct ScoreStream<S> {
    scorer: S,
    fingerprinter: Fingerprinter,
    scores: Scores,
}

/// Shows the progress and nothing derived from the input.
impl<S: AsRef<Scorer>> fmt::Debug for ScoreStream<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScoreStream")
            .field("len", &self.fingerprinter.len())
            .field("windows", &self.fingerprinter.windows())
            .finish_non_exhaustive()
    }
}

impl<S: AsRef<Scorer>> ScoreStream<S> {
    /// Creates a stream at the start of an input.
    pub fn new(scorer: S) -> Self {
        Self {
            scorer,
            fingerprinter: Fingerprinter::new(),
            scores: Scores::ZERO,
        }
    }

    /// Consumes the next part of the input and scores every window that it
    /// completes.
    ///
    /// The call never fails and never panics, allocates nothing, and takes
    /// time linear in the length of the part.
    pub fn update(&mut self, part: &[u8]) {
        let scorer = self.scorer.as_ref();
        let scores = &mut self.scores;
        self.fingerprinter.update(part, |window| {
            *scores = scores.max_with(scorer.score_fingerprint(window));
        });
    }

    /// Returns, in each category, the highest score of the windows
    /// completed so far, or [`Scores::ZERO`] before the first one.
    ///
    /// The scores of [`finish`](Self::finish) are at least these.
    #[must_use]
    pub fn scores(&self) -> Scores {
        self.scores
    }

    /// Ends the input, returns its scores, and starts a new input.
    pub fn finish(&mut self) -> Scores {
        let mut scores = self.scores;
        if let Some(window) = self.fingerprinter.finish() {
            scores = scores.max_with(self.scorer.as_ref().score_fingerprint(window));
        }
        self.scores = Scores::ZERO;
        scores
    }

    /// Discards the input so far and starts a new input.
    pub fn reset(&mut self) {
        self.fingerprinter.reset();
        self.scores = Scores::ZERO;
    }

    /// Returns the number of bytes consumed since the input started.
    #[must_use]
    pub fn len(&self) -> u64 {
        self.fingerprinter.len()
    }

    /// Returns whether no byte was consumed since the input started.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.fingerprinter.is_empty()
    }

    /// Returns the scorer.
    #[must_use]
    pub fn scorer(&self) -> &Scorer {
        self.scorer.as_ref()
    }

    /// Returns the scorer and drops the input so far.
    pub fn into_scorer(self) -> S {
        self.scorer
    }

    /// Returns the value that lends the scorer.
    pub(crate) fn source(&self) -> &S {
        &self.scorer
    }
}

/// Every write consumes the whole buffer, and flushing does nothing.
impl<S: AsRef<Scorer>> io::Write for ScoreStream<S> {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        self.update(buffer);
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write as _;
    use std::sync::Arc;

    use super::*;
    use crate::fingerprint::BLOCK_LEN;
    use crate::model::tests::random_model;
    use crate::testutil::Rng;

    #[test]
    fn streams_score_like_the_whole_input() {
        let scorer = Scorer::with_model(random_model(&mut Rng::new(51), &[24]));
        let mut rng = Rng::new(52);
        let mut stream = scorer.stream();
        for len in [
            0,
            1,
            2,
            3,
            255,
            256,
            257,
            512,
            513,
            1000,
            2 * BLOCK_LEN * 3 + 1,
        ] {
            let mut input = vec![0u8; len];
            rng.fill(&mut input);
            let expected = scorer.score(&input);
            for part in [1, 7, 256, 300, 4096] {
                let mut previous = Scores::ZERO;
                for chunk in input.chunks(part) {
                    stream.update(chunk);
                    // Partial scores never fall and never exceed the result.
                    assert_eq!(stream.scores().max_with(previous), stream.scores());
                    assert_eq!(stream.scores().max_with(expected), expected);
                    previous = stream.scores();
                }
                assert_eq!(stream.len(), len as u64);
                assert_eq!(stream.finish(), expected, "length {len}, parts of {part}");
                assert!(stream.is_empty());
                assert_eq!(stream.scores(), Scores::ZERO);
            }
        }
    }

    #[test]
    fn streams_score_like_the_whole_input_at_random_cuts() {
        let scorer = Scorer::with_model(random_model(&mut Rng::new(54), &[16]));
        let mut rng = Rng::new(55);
        let mut stream = scorer.stream();
        for len in [0, 2, 300, 511, 512, 513, 1500, 5000] {
            let mut input = vec![0u8; len];
            rng.fill(&mut input);
            let expected = scorer.score(&input);
            for _ in 0..20 {
                let mut cuts: Vec<usize> = (0..rng.next_u32() % 10)
                    .map(|_| rng.next_u32() as usize % (len + 1))
                    .collect();
                cuts.sort_unstable();
                let mut start = 0;
                for &cut in cuts.iter().chain([&len]) {
                    stream.update(&input[start..cut]);
                    start = cut;
                }
                assert_eq!(stream.finish(), expected, "length {len}, cuts {cuts:?}");
            }
        }
    }

    #[test]
    fn streams_accept_shared_scorers_and_writers() {
        let scorer = Arc::new(Scorer::with_model(random_model(&mut Rng::new(53), &[8])));
        let input = vec![b'<'; 900];
        let expected = scorer.score(&input);
        let mut stream = ScoreStream::new(Arc::clone(&scorer));
        std::io::copy(&mut &input[..], &mut stream).unwrap();
        stream.flush().unwrap();
        assert_eq!(
            format!("{stream:?}"),
            "ScoreStream { len: 900, windows: 2, .. }"
        );
        assert_eq!(stream.finish(), expected);
        stream.write_all(&input[..10]).unwrap();
        stream.reset();
        assert!(stream.is_empty());
        assert_eq!(stream.finish(), Scores::ZERO);
        assert_eq!(stream.scorer().model(), scorer.model());
        assert!(Arc::ptr_eq(&stream.into_scorer(), &scorer));
        let mut owned = ScoreStream::new(Scorer::clone(&scorer));
        owned.update(&input);
        assert_eq!(owned.finish(), expected);
    }
}
