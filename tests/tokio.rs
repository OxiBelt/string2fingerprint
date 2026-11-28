//! Checks that the asynchronous interface of the `tokio` feature gives what
//! the synchronous one gives, through the public items only.

use string2fingerprint::prelude::*;
use string2fingerprint::tokio::AsyncScorer;

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
