//! Checks that the asynchronous interface of the `tokio` feature gives what
//! the synchronous one gives, through the public items only.

use string2fingerprint::prelude::*;
use string2fingerprint::tokio::{AsyncDetector, AsyncScorer, DEFAULT_INLINE_LIMIT};

const ATTACKS: &str = include_str!("data/attacks.tsv");
const BENIGN: &str = include_str!("data/benign.txt");

/// Every development string, attacks first, and all of them in one input.
fn inputs() -> Vec<Vec<u8>> {
    let mut inputs: Vec<Vec<u8>> = ATTACKS
        .lines()
        .filter_map(|line| line.split_once('\t'))
        .map(|(_, payload)| payload.as_bytes().to_vec())
        .chain(BENIGN.lines().map(|line| line.as_bytes().to_vec()))
        .collect();
    inputs.push(inputs.concat());
    inputs
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .build()
        .expect("a runtime starts")
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
fn the_prelude_names_the_asynchronous_items() {
    use string2fingerprint::prelude::{AsyncDetector as Detector2, AsyncScorer as Scorer2};
    let detector = Detector2::default();
    assert_eq!(detector.inline_limit(), DEFAULT_INLINE_LIMIT);
    assert_eq!(Scorer2::default().inline_limit(), DEFAULT_INLINE_LIMIT);
}
