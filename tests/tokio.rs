//! Checks that the asynchronous interface of the `tokio` feature gives what
//! the synchronous one gives, through the public items only.

use std::future::Future;
use std::io;
use std::pin::{Pin, pin};
use std::task::{Context, Poll};

use string2fingerprint::prelude::*;
use string2fingerprint::tokio::{
    AsyncDetectionStream, AsyncDetector, AsyncScoreStream, AsyncScorer, DEFAULT_INLINE_LIMIT,
};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt, ReadBuf};

const ATTACKS: &str = include_str!("data/attacks.tsv");
const BENIGN: &str = include_str!("data/benign.txt");

/// Every development string, attacks first, and long inputs made of them.
fn inputs() -> Vec<Vec<u8>> {
    let mut inputs: Vec<Vec<u8>> = ATTACKS
        .lines()
        .filter_map(|line| line.split_once('\t'))
        .map(|(_, payload)| payload.as_bytes().to_vec())
        .chain(BENIGN.lines().map(|line| line.as_bytes().to_vec()))
        .collect();
    let all = inputs.concat();
    // Longer than two chunks of the streams, so that they score on the pool
    // while more is written.
    let long: Vec<u8> = all.iter().copied().cycle().take(200_000).collect();
    inputs.push(all);
    inputs.push(long);
    inputs
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .build()
        .expect("a runtime starts")
}

/// Returns its input in parts of at most `part` bytes.
struct Parts {
    input: Vec<u8>,
    at: usize,
    part: usize,
}

impl AsyncRead for Parts {
    fn poll_read(
        mut self: Pin<&mut Self>,
        _: &mut Context<'_>,
        buffer: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        let take = self
            .part
            .min(self.input.len() - self.at)
            .min(buffer.remaining());
        let at = self.at;
        buffer.put_slice(&self.input[at..at + take]);
        self.at += take;
        Poll::Ready(Ok(()))
    }
}

fn assert_send_static<T: Send + 'static>(_: &T) {}

#[test]
fn the_scorer_scores_as_the_synchronous_one() {
    let sync = Scorer::new();
    runtime().block_on(async {
        for scorer in [
            AsyncScorer::new(),
            AsyncScorer::new().with_inline_limit(0),
            AsyncScorer::new().with_inline_limit(usize::MAX),
        ] {
            for input in inputs() {
                let expected = sync.score(&input);
                let future = scorer.score(input.clone());
                assert_send_static(&future);
                assert_eq!(future.await, expected);
                assert_eq!(scorer.logits(input.clone()).await, sync.logits(&input));
                assert_eq!(
                    scorer.windows(input.clone()).await,
                    sync.windows(&input).collect::<Vec<_>>()
                );
            }
            let inputs = inputs();
            assert_eq!(
                scorer.score_fields(inputs.clone()).await,
                sync.score_fields(&inputs)
            );
            assert_eq!(
                scorer.score_batch(inputs.clone()).await,
                sync.score_batch(&inputs)
            );
            for threads in [0, 1, 3, 1000] {
                assert_eq!(
                    scorer.score_parallel(inputs.clone(), threads).await,
                    sync.score_batch(&inputs)
                );
            }
            assert_eq!(scorer.score_parallel(Vec::<Vec<u8>>::new(), 0).await, []);
            assert!(scorer.kernel_self_test().await);
        }
    });
}

#[test]
fn the_detector_decides_as_the_synchronous_one() {
    for preset in Preset::ALL {
        let sync = Detector::with_preset(*preset);
        let detector = AsyncDetector::from(*preset);
        runtime().block_on(async {
            for input in inputs() {
                let expected = sync.detect(&input);
                let future = detector.detect(input.clone());
                assert_send_static(&future);
                assert_eq!(future.await, expected);
                assert_eq!(
                    detector.is_flagged(input.clone()).await,
                    expected.is_flagged()
                );
            }
            let inputs = inputs();
            assert_eq!(
                detector.detect_fields(inputs.clone()).await,
                sync.detect_fields(&inputs)
            );
            assert_eq!(
                detector.detect_batch(inputs.clone()).await,
                sync.detect_batch(&inputs)
            );
            assert_eq!(
                detector.detect_parallel(inputs.clone(), 4).await,
                sync.detect_batch(&inputs)
            );
            let scorer = detector.to_scorer();
            assert_eq!(scorer.inline_limit(), DEFAULT_INLINE_LIMIT);
            assert_eq!(
                scorer.score(inputs[0].clone()).await,
                sync.scorer().score(&inputs[0])
            );
        });
    }
}

#[test]
fn streams_give_the_result_of_the_whole_input_however_it_is_split() {
    let scorer = AsyncScorer::new();
    let detector = AsyncDetector::new();
    runtime().block_on(async {
        for input in inputs() {
            let expected = scorer.scorer().score(&input);
            for part in [1, 100, 4096, 65_535, 65_536, 100_000, input.len().max(1)] {
                let mut stream: AsyncScoreStream = scorer.stream();
                let mut detection: AsyncDetectionStream = detector.stream();
                for piece in input.chunks(part) {
                    stream.update(piece).await;
                    detection.update(piece).await;
                }
                assert_eq!(stream.len(), input.len() as u64);
                assert_eq!(stream.finish().await, expected);
                assert!(stream.is_empty());
                assert_eq!(detection.finish().await, detector.detector().detect(&input));
            }
        }
    });
}

#[test]
fn stream_progress_never_exceeds_the_result() {
    let scorer = AsyncScorer::new();
    runtime().block_on(async {
        let input = inputs().pop().expect("there are inputs");
        let expected = scorer.scorer().score(&input);
        let mut stream = scorer.stream();
        let mut last = Scores::ZERO;
        for piece in input.chunks(10_000) {
            stream.update(piece).await;
            let scores = stream.scores();
            for category in Category::ALL {
                assert!(scores.get(category) >= last.get(category));
                assert!(scores.get(category) <= expected.get(category));
            }
            last = scores;
        }
        stream.flush().await.expect("flushing never fails");
        // Every complete window is scored after a flush.
        let mut sync = scorer.scorer().stream();
        sync.update(&input);
        assert_eq!(stream.scores(), sync.scores());
        assert_eq!(stream.finish().await, expected);
    });
}

#[test]
fn streams_are_async_writers() {
    let detector = AsyncDetector::new();
    runtime().block_on(async {
        let input = inputs().pop().expect("there are inputs");
        let mut stream = detector.stream();
        let mut reader = Parts {
            input: input.clone(),
            at: 0,
            part: 3000,
        };
        let copied = tokio::io::copy(&mut reader, &mut stream)
            .await
            .expect("the stream never fails");
        assert_eq!(copied, input.len() as u64);
        stream.shutdown().await.expect("the stream never fails");
        assert_eq!(stream.finish().await, detector.detector().detect(&input));
        assert_eq!(stream.write(&[]).await.expect("never fails"), 0);
    });
}

#[test]
fn a_reset_discards_the_input_and_running_work() {
    let scorer = AsyncScorer::new();
    runtime().block_on(async {
        let input = inputs().pop().expect("there are inputs");
        let mut stream = scorer.stream();
        // Two chunks: one is on the pool when the reset comes.
        stream.update(&input[..140_000]).await;
        stream.reset();
        assert!(stream.is_empty());
        assert_eq!(stream.scores(), Scores::ZERO);
        stream.update(b"name=alice").await;
        assert_eq!(stream.finish().await, scorer.scorer().score(b"name=alice"));
    });
}

#[test]
fn cancelled_updates_keep_what_they_buffered() {
    let scorer = AsyncScorer::new();
    runtime().block_on(async {
        let input = inputs().pop().expect("there are inputs");
        let mut stream = scorer.stream();
        // Cancel each update after its first poll.
        let mut at = 0;
        while at < input.len() {
            let before = stream.len();
            {
                let mut update = pin!(stream.update(&input[at..]));
                std::future::poll_fn(|cx| {
                    let _ = update.as_mut().poll(cx);
                    Poll::Ready(())
                })
                .await;
            }
            at += usize::try_from(stream.len() - before).expect("fits");
        }
        assert_eq!(stream.finish().await, scorer.scorer().score(&input));
    });
}

#[test]
fn readers_give_the_result_of_the_whole_input() {
    let scorer = AsyncScorer::new();
    let detector = AsyncDetector::new();
    runtime().block_on(async {
        for input in inputs() {
            for part in [1, 777, 8192, 70_000] {
                let reader = Parts {
                    input: input.clone(),
                    at: 0,
                    part,
                };
                let scores = scorer.score_reader(reader).await.expect("no read fails");
                assert_eq!(scores, scorer.scorer().score(&input));
            }
            let bounded = (&input[..]).take(1000);
            let detection = detector
                .detect_reader(bounded)
                .await
                .expect("no read fails");
            assert_eq!(
                detection,
                detector.detector().detect(&input[..input.len().min(1000)])
            );
        }
    });
}

#[test]
fn a_reader_error_is_returned() {
    struct Failing;
    impl AsyncRead for Failing {
        fn poll_read(
            self: Pin<&mut Self>,
            _: &mut Context<'_>,
            _: &mut ReadBuf<'_>,
        ) -> Poll<io::Result<()>> {
            Poll::Ready(Err(io::ErrorKind::ConnectionReset.into()))
        }
    }
    runtime().block_on(async {
        let error = AsyncDetector::new()
            .detect_reader(Failing)
            .await
            .expect_err("the read fails");
        assert_eq!(error.kind(), io::ErrorKind::ConnectionReset);
    });
}

#[test]
fn work_runs_on_a_current_thread_runtime() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a runtime starts");
    let detector = AsyncDetector::new();
    let input = inputs().pop().expect("there are inputs");
    let expected = detector.detector().detect(&input);
    let detection = runtime.block_on(async {
        let worker = detector.clone();
        let task = tokio::spawn(async move { worker.detect(input).await });
        task.await.expect("the task completes")
    });
    assert_eq!(detection, expected);
}

#[test]
fn stream_progress_follows_small_writes() {
    // An attack that the synchronous stream flags early, in small parts.
    let mut body = b"name=alice&comment=".to_vec();
    body.extend(
        std::iter::repeat_n(
            b"%3Cscript%3Ealert(document.cookie)%3C%2Fscript%3E ".as_slice(),
            40,
        )
        .flatten(),
    );
    let sync = Detector::new();
    let first_sync = {
        let mut stream = sync.stream();
        body.chunks(128).position(|part| {
            stream.update(part);
            stream.detection().is_flagged()
        })
    };
    let first_sync = first_sync.expect("the synchronous stream flags the body");
    let detector = AsyncDetector::new();
    runtime().block_on(async {
        let mut stream = detector.stream();
        let mut first = None;
        for (index, part) in body.chunks(128).enumerate() {
            stream.update(part).await;
            if stream.detection().is_flagged() {
                first = Some(index);
                break;
            }
        }
        // Each write of 128 bytes is at most the inline limit, so it is
        // scored at once: the stream flags at the same part.
        assert_eq!(first, Some(first_sync));
    });
}

#[test]
fn the_prelude_names_the_asynchronous_items() {
    use string2fingerprint::prelude::{AsyncDetector as Detector2, AsyncScorer as Scorer2};
    let detector = Detector2::default();
    assert_eq!(detector.inline_limit(), DEFAULT_INLINE_LIMIT);
    assert_eq!(Scorer2::default().inline_limit(), DEFAULT_INLINE_LIMIT);
}
