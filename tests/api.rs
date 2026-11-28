//! Checks that the low-level and high-level interfaces agree, through the
//! public items only.

use std::io::Write as _;
use std::sync::Arc;

use string2fingerprint::DetectorBuilder;
use string2fingerprint::fingerprint::{
    self, FINGERPRINT_LEN, Fingerprint, Fingerprinter, Section, feature_value,
};
use string2fingerprint::kernel::{Kernel, Weights};
use string2fingerprint::model::{EmbeddedModel, Logits, Model};
use string2fingerprint::prelude::*;

const ATTACKS: &str = include_str!("data/attacks.tsv");
const BENIGN: &str = include_str!("data/benign.txt");

/// Every development string, attacks first, and a long input made of all
/// of them.
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

/// Scores `input` with the low-level stages: the n-grams, the kernel, and
/// the model by hand.
fn by_hand(model: &Model, kernel: Kernel, input: &[u8]) -> Scores {
    let blocks = input.len().div_ceil(fingerprint::BLOCK_LEN);
    let mut counts = vec![[0usize; FINGERPRINT_LEN]; blocks];
    for ngram in fingerprint::ngrams(input) {
        counts[ngram.block()][ngram.feature()] += 1;
    }
    let mut logits = Logits::from_array([i8::MIN; Category::COUNT]);
    for window in 0..fingerprint::window_count(input.len()) {
        let last = (window + 1).min(blocks - 1);
        let mut features = [0u8; FINGERPRINT_LEN];
        for (feature, value) in features.iter_mut().enumerate() {
            *value = feature_value((window..=last).map(|block| counts[block][feature]).sum());
        }
        logits = logits.max_with(model.logits(kernel, &Fingerprint::from_array(features)));
    }
    logits.scores()
}

#[test]
fn every_level_gives_the_same_scores() {
    let scorer = Scorer::new();
    let model = scorer.model();
    for input in inputs() {
        let expected = scorer.score(&input);
        assert_eq!(by_hand(model, Kernel::SCALAR, &input), expected);
        assert_eq!(scorer.logits(&input).scores(), expected);
        let windows = scorer
            .windows(&input)
            .fold(Scores::ZERO, |all, window| all.max_with(window.scores()));
        assert_eq!(windows, expected);
        let mut stream = scorer.stream();
        for part in input.chunks(37) {
            stream.update(part);
        }
        assert_eq!(stream.finish(), expected);
        assert_eq!(scorer.score_reader(&input[..]).unwrap(), expected);
    }
}

#[test]
fn every_supported_kernel_gives_the_same_layers() {
    let model = Model::embedded();
    let input = b"{{7*7}}${jndi:ldap://x/a}";
    let window = fingerprint::fingerprints(input).next().unwrap();
    let reference = model.trace(Kernel::SCALAR, &window);
    for backend in Backend::supported() {
        let kernel = Kernel::new(backend).unwrap();
        assert_eq!(model.trace(kernel, &window), reference, "backend {backend}");
        let scorer = Scorer::builder().kernel(kernel).build().unwrap();
        assert_eq!(scorer.score(input), reference.scores());
    }
    // The first layer is a kernel call plus the biases.
    let layer = model.layer(0).unwrap();
    let weights = Weights::new(layer.weights().to_vec(), layer.inputs()).unwrap();
    let mut sums = vec![0i32; layer.outputs()];
    Kernel::detect()
        .matvec(&weights, window.as_array(), &mut sums)
        .unwrap();
    let with_bias: Vec<i32> = sums
        .iter()
        .zip(layer.biases())
        .map(|(sum, bias)| sum + bias)
        .collect();
    assert_eq!(reference.accumulators(0).unwrap(), &with_bias[..]);
}

#[test]
fn detectors_decide_what_scorers_score() {
    let detector = Detector::new();
    let scorer = detector.scorer();
    let thresholds = detector.thresholds();
    let inputs = inputs();
    for (input, detection) in inputs.iter().zip(detector.detect_batch(&inputs)) {
        assert_eq!(detection.scores(), scorer.score(input));
        assert_eq!(
            detection.flagged(),
            scorer.score(input).flagged(&thresholds)
        );
    }
    assert_eq!(
        detector.detect_parallel(&inputs, 4),
        detector.detect_batch(&inputs)
    );
    let fields: Vec<&[u8]> = inputs.iter().map(Vec::as_slice).collect();
    assert_eq!(
        detector.detect_fields(&fields).scores(),
        scorer.score_fields(&fields)
    );
}

#[test]
fn builders_and_streams_compose() {
    let model = EmbeddedModel::Compact.load();
    let detector = Arc::new(
        DetectorBuilder::default()
            .model_bytes(&model.to_bytes())
            .backend(Backend::Scalar)
            .threshold(Threshold::at(160))
            .disable(Category::Malformed)
            .build()
            .unwrap(),
    );
    assert_eq!(detector.scorer().model(), &model);
    assert_eq!(detector.scorer().backend(), Backend::Scalar);
    let input = b"q=%3Cscript%3Ealert(1)%3C%2Fscript%3E";
    let mut stream = DetectionStream::new(Arc::clone(&detector));
    stream.write_all(input).unwrap();
    let detection = stream.finish();
    assert_eq!(detection, detector.detect(input));
    assert!(!detection.contains(Category::Malformed));

    let mut fingerprinter = Fingerprinter::new();
    let mut windows = Vec::new();
    fingerprinter.update(input, |window| windows.push(window.clone()));
    windows.extend(fingerprinter.finish().cloned());
    assert_eq!(
        windows,
        fingerprint::fingerprints(input).collect::<Vec<_>>()
    );
    let lexical: usize = Section::ALL
        .iter()
        .filter(|section| !section.is_shape())
        .map(|&section| windows[0].section(section).len())
        .sum();
    assert_eq!(lexical, 1280);
}
