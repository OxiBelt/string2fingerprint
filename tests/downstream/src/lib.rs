//! Scores text with a model that the build script trained.

use string2fingerprint::model::Model;
use string2fingerprint::{CategorySet, Detector, Scorer, Thresholds};

/// The thresholds that the build script chose on held-out samples.
pub const THRESHOLDS: Thresholds = match string2fingerprint::include_thresholds!("fixture") {
    Some(thresholds) => thresholds,
    None => panic!("the build script wrote valid thresholds"),
};

/// Returns the model that the build script trained.
#[must_use]
pub fn model() -> Model {
    string2fingerprint::include_model!("fixture").expect("the build script validated the model")
}

/// Returns a detector that decides with the model and the thresholds of the
/// build script.
#[must_use]
pub fn detector() -> Detector {
    Detector::from_parts(Scorer::with_model(model()), THRESHOLDS)
}

/// Returns the categories that the model flags in `input`.
#[must_use]
pub fn flagged(scorer: &Scorer, input: &[u8]) -> CategorySet {
    scorer.score(input).flagged(&THRESHOLDS)
}

#[cfg(test)]
mod tests {
    use string2fingerprint::{Category, Threshold};

    use super::*;

    #[test]
    fn the_embedded_model_has_the_shape_that_the_build_script_set() {
        let sizes: Vec<usize> = model().layer_sizes().collect();
        assert_eq!(sizes, [16, Category::COUNT]);
    }

    #[test]
    fn the_embedded_model_learned_the_training_data() {
        let scorer = Scorer::with_model(model());
        let lines = |text: &'static str| text.lines().filter(|line| !line.is_empty());
        let benign: Vec<&str> = lines(include_str!("../../data/benign.txt")).collect();
        let attacks: Vec<&str> = lines(include_str!("../../data/attacks.tsv"))
            .filter(|line| !line.starts_with('#'))
            .filter_map(|line| line.split_once('\t'))
            .map(|(_, payload)| payload)
            .collect();
        let count = |inputs: &[&str], flags: &dyn Fn(&[u8]) -> CategorySet| {
            inputs
                .iter()
                .filter(|input| !flags(input.as_bytes()).is_empty())
                .count()
        };

        // The development set is small and a fifth of it was held out, so
        // the model is weak. The limits check that the model was trained on
        // this data, not how good it is.
        let at_128 = |input: &[u8]| scorer.score(input).at_least(128);
        assert!(count(&benign, &at_128) * 10 <= benign.len());
        assert!(count(&attacks, &at_128) * 10 >= attacks.len() * 4);

        // The thresholds were chosen on the few held-out samples, so they
        // are only required to flag more attacks than benign text, and
        // little benign text.
        let calibrated = |input: &[u8]| flagged(&scorer, input);
        assert!(count(&attacks, &calibrated) > count(&benign, &calibrated));
        assert!(count(&benign, &calibrated) * 10 <= benign.len());
        assert_ne!(THRESHOLDS, Thresholds::uniform(Threshold::at(128)));
        let detector = detector();
        for input in benign.iter().chain(&attacks) {
            assert_eq!(
                detector.detect(input.as_bytes()).flagged(),
                calibrated(input.as_bytes())
            );
        }
    }
}
