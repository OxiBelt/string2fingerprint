//! End-to-end checks of the public training API.
//!
//! The development set in `tests/data/` is far too small to train a model
//! of any use. These tests check that the pieces fit together: samples go
//! in, a valid model comes out, and it has learned the samples it saw.

#![cfg(feature = "train")]

use string2fingerprint::eval::Objective;
use string2fingerprint::model::Model;
use string2fingerprint::train::{Dataset, DatasetError, Sample, TrainConfig, Trainer};
use string2fingerprint::{Category, CategorySet, Scorer, Threshold, Thresholds};

const ATTACKS: &[u8] = include_bytes!("data/attacks.tsv");
const BENIGN: &[u8] = include_bytes!("data/benign.txt");

fn development_set() -> Dataset {
    let mut data = Dataset::new();
    assert_eq!(data.extend_from_labeled_lines(ATTACKS), Ok(63));
    assert_eq!(data.extend_from_lines(BENIGN, CategorySet::EMPTY), Ok(72));
    data
}

fn config() -> TrainConfig {
    let mut config = TrainConfig::default();
    config.hidden = vec![16];
    config.epochs = 40;
    config.quantized_epochs = 5;
    config.batch_size = 16;
    config.learning_rate = 1e-2;
    config.input_dropout = 0.0;
    config
}

#[test]
fn a_model_trained_on_the_development_set_recalls_it() {
    let data = development_set();
    assert_eq!(data.samples(), 135);
    assert_eq!(data.benign(), 72);
    for category in Category::ALL {
        assert!(data.count(category) > 0, "{category}");
    }

    let trained = Trainer::new(config()).unwrap().train(&data).unwrap();
    let report = trained.report();
    assert!(report.final_loss() < 0.25 * report.epochs()[0].loss);

    // The model survives its byte format.
    let bytes = trained.model().to_bytes();
    let model = Model::from_bytes(&bytes).unwrap();
    assert_eq!(&model, trained.model());

    // Scores of the stored fingerprints equal scores of the bytes.
    let scorer = Scorer::with_model(model);
    let mut scores = data.scores(&scorer);
    for line in ATTACKS
        .split(|&byte| byte == b'\n')
        .filter(|line| !line.is_empty())
    {
        let tab = line.iter().position(|&byte| byte == b'\t').unwrap();
        let (labels, stored) = scores.next().unwrap();
        assert!(!labels.is_empty());
        assert_eq!(stored, scorer.score(&line[tab + 1..]));
    }

    let evaluation = data.evaluate(&scorer);
    assert_eq!(evaluation.samples(), 135);
    assert_eq!(evaluation.benign_samples(), 72);
    let any = evaluation.any_confusion(Threshold::at(128));
    assert!(any.true_positives >= 48, "{any:?}");
    assert!(any.false_positives <= 3, "{any:?}");

    // Thresholds that flag no benign sample of the set still flag most of
    // its attacks, because the model has memorized them.
    let thresholds = evaluation.benign_negatives().thresholds(
        Objective::MaxFalsePositiveRate(0.0),
        Thresholds::uniform(Threshold::NEVER),
    );
    let (mut flagged_attacks, mut flagged_benign) = (0, 0);
    for (labels, scores) in data.scores(&scorer) {
        let flagged = scores.flagged(&thresholds);
        if labels.is_empty() {
            flagged_benign += usize::from(!flagged.is_empty());
        } else {
            flagged_attacks += usize::from(!(flagged & labels).is_empty());
        }
    }
    assert_eq!(flagged_benign, 0);
    assert!(flagged_attacks >= 45, "{flagged_attacks} of 63");
}

#[test]
fn training_twice_gives_the_same_bytes() {
    let data = development_set();
    let mut config = config();
    config.epochs = 6;
    config.quantized_epochs = 2;
    config.input_dropout = 0.2;
    let train = |threads: usize, seed: u64| {
        let mut config = config.clone();
        config.threads = threads;
        config.seed = seed;
        Trainer::new(config)
            .unwrap()
            .train(&data)
            .unwrap()
            .into_model()
            .to_bytes()
    };
    let reference = train(1, 1);
    assert_eq!(train(1, 1), reference);
    assert_eq!(train(4, 1), reference);
    assert_ne!(train(1, 2), reference);
}

#[test]
fn a_held_out_part_is_disjoint_from_the_rest() {
    let data = development_set();
    let (rest, held_out) = data.split(0.2, 5);
    assert_eq!(held_out.samples(), 27);
    assert_eq!(rest.samples(), 108);
    let mut joined = rest.clone();
    joined.extend_from_dataset(&held_out);
    assert_eq!(joined.samples(), data.samples());
    for category in Category::ALL {
        assert_eq!(joined.count(category), data.count(category), "{category}");
    }
}

#[test]
fn a_long_labeled_input_is_refused_without_an_explicit_choice() {
    let mut data = Dataset::new();
    let long = vec![b'A'; 4096];
    let sample = Sample::new(&long, Category::Malformed);
    assert_eq!(data.push(sample), Err(DatasetError::LabeledInputTooLong));
    assert_eq!(data.push(Sample::benign(&long)), Ok(15));
    assert_eq!(data.push(sample.label_every_window()), Ok(15));
    assert_eq!(data.samples(), 2);
    assert_eq!(data.windows(), 30);
}
