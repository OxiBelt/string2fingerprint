//! Evaluates the default model on the hand-written development set.
//!
//! `tests/data/attacks.tsv` and `tests/data/benign.txt` are described in
//! `tests/model_quality.rs`. These tests check that an evaluation of their
//! scores agrees with counts taken directly from the scores, and that
//! calibrated thresholds do what their objective promises on the samples
//! they were chosen on. They assume nothing about the scores themselves.

use string2fingerprint::eval::{Evaluation, Objective, Tally};
use string2fingerprint::{Category, CategorySet, Scorer, Scores, Threshold, Thresholds};

const ATTACKS: &str = include_str!("data/attacks.tsv");
const BENIGN: &str = include_str!("data/benign.txt");

const ATTACK_LINES: u64 = 63;
const BENIGN_LINES: u64 = 72;

const THRESHOLD: u8 = 128;

/// A scored line: the source, its line number, its scores, and its labels.
struct Sample {
    source: &'static str,
    line: usize,
    scores: Scores,
    labels: CategorySet,
}

fn samples() -> Vec<Sample> {
    let scorer = Scorer::new();
    let attacks = ATTACKS.lines().enumerate().map(|(index, line)| {
        let (name, payload) = line.split_once('\t').expect("category, tab, payload");
        let category = Category::from_name(name)
            .unwrap_or_else(|| panic!("unknown category in attack line {}", index + 1));
        Sample {
            source: "attack",
            line: index + 1,
            scores: scorer.score(payload.as_bytes()),
            labels: category.into(),
        }
    });
    let benign = BENIGN.lines().enumerate().map(|(index, line)| Sample {
        source: "benign",
        line: index + 1,
        scores: scorer.score(line.as_bytes()),
        labels: CategorySet::EMPTY,
    });
    attacks.chain(benign).collect()
}

fn evaluation(samples: &[Sample]) -> Evaluation {
    let mut evaluation = Evaluation::new();
    for sample in samples {
        evaluation.record(sample.scores, sample.labels);
    }
    evaluation
}

#[test]
fn evaluation_counts_the_development_set() {
    let samples = samples();
    let evaluation = evaluation(&samples);
    assert_eq!(evaluation.samples(), ATTACK_LINES + BENIGN_LINES);
    assert_eq!(evaluation.benign_samples(), BENIGN_LINES);
    let any = evaluation.any_confusion(Threshold::ALWAYS);
    assert_eq!(any.positives(), ATTACK_LINES);
    assert_eq!(any.negatives(), BENIGN_LINES);
}

#[test]
fn confusion_agrees_with_direct_counts() {
    let samples = samples();
    let evaluation = evaluation(&samples);
    let benign_negatives = evaluation.benign_negatives();
    let threshold = Threshold::at(THRESHOLD);

    for category in Category::ALL {
        let mut counts = [0_u64; 4];
        let mut benign_false_positives = 0;
        for sample in &samples {
            let flagged = sample.scores.get(category) >= THRESHOLD;
            let labeled = sample.labels.contains(category);
            let outcome = match (flagged, labeled) {
                (true, true) => 0,
                (true, false) => 1,
                (false, true) => 2,
                (false, false) => 3,
            };
            counts[outcome] += 1;
            if flagged && sample.labels.is_empty() {
                benign_false_positives += 1;
            }
        }
        let confusion = evaluation.confusion(category, threshold);
        assert_eq!(
            [
                confusion.true_positives,
                confusion.false_positives,
                confusion.false_negatives,
                confusion.true_negatives
            ],
            counts,
            "{category}"
        );
        let benign = benign_negatives.confusion(category, threshold);
        assert_eq!(benign.true_positives, confusion.true_positives);
        assert_eq!(benign.false_negatives, confusion.false_negatives);
        assert_eq!(benign.false_positives, benign_false_positives, "{category}");
        assert_eq!(benign.negatives(), BENIGN_LINES);

        let auc = evaluation.auc(category).expect("positives and negatives");
        assert!((0.0..=1.0).contains(&auc), "{category}: {auc}");
    }

    let mut any = [0_u64; 4];
    for sample in &samples {
        let flagged = sample.scores.any_at_least(THRESHOLD);
        let labeled = !sample.labels.is_empty();
        any[usize::from(!labeled) + 2 * usize::from(!flagged)] += 1;
    }
    let confusion = evaluation.any_confusion(threshold);
    assert_eq!(
        [
            confusion.true_positives,
            confusion.false_positives,
            confusion.false_negatives,
            confusion.true_negatives
        ],
        any
    );

    let thresholds = Thresholds::uniform(threshold);
    let mut tally = Tally::new();
    for sample in &samples {
        tally.record(sample.scores.flagged(&thresholds), sample.labels);
    }
    assert_eq!(tally.samples(), evaluation.samples());
    assert_eq!(tally.any(), confusion);
    for category in Category::ALL {
        assert_eq!(
            tally.category(category),
            evaluation.confusion(category, threshold)
        );
    }
}

#[test]
fn calibrated_thresholds_meet_their_objective() {
    let samples = samples();
    let evaluation = evaluation(&samples);
    let fallback = Thresholds::uniform(Threshold::NEVER);

    // No sample is flagged in a category that it is a negative of: in the
    // default view any category outside its labels, and with benign
    // negatives any category of a benign sample.
    let strict = Objective::MaxFalsePositiveRate(0.0);
    let views = [
        (evaluation.thresholds(strict, fallback), false),
        (
            evaluation.benign_negatives().thresholds(strict, fallback),
            true,
        ),
    ];
    for (thresholds, benign_negatives) in views {
        for sample in &samples {
            if benign_negatives && !sample.labels.is_empty() {
                continue;
            }
            let wrong = sample.scores.flagged(&thresholds) - sample.labels;
            assert!(
                wrong.is_empty(),
                "{} line {} flagged as {wrong}",
                sample.source,
                sample.line
            );
        }
        // Each threshold is the lowest that does so.
        for category in Category::ALL {
            let threshold = thresholds.get(category);
            let Some(lower) = threshold.index().checked_sub(1) else {
                continue;
            };
            let lower = Threshold::from_index(lower).unwrap();
            let view = if benign_negatives {
                evaluation.benign_negatives()
            } else {
                evaluation.clone()
            };
            assert!(
                view.confusion(category, lower).false_positives > 0,
                "{category}"
            );
        }
    }

    let thresholds = evaluation.thresholds(Objective::MinRecall(1.0), fallback);
    for sample in samples.iter().filter(|sample| !sample.labels.is_empty()) {
        let flagged = sample.scores.flagged(&thresholds);
        assert_eq!(
            flagged & sample.labels,
            sample.labels,
            "{} line {} missed",
            sample.source,
            sample.line
        );
    }
    for category in Category::ALL {
        let recall = evaluation
            .confusion(category, thresholds.get(category))
            .recall();
        assert_eq!(recall, Some(1.0), "{category}");
    }
}
