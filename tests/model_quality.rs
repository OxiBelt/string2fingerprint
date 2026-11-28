//! Guards the embedded models against a quality regression.
//!
//! `tests/data/attacks.tsv` and `tests/data/benign.txt` are a small
//! hand-written development set. Its strings do not come from the generator
//! that produces the training data, so it shows whether the model carries
//! over to text of a different origin. It was consulted while the
//! fingerprint layout and the training data were designed, so it is not a
//! held-out test set, and passing it says nothing about real traffic.
//!
//! The limits below are looser than the results of each embedded model,
//! which `tools/train/README.md` records. A retrained model that fails them
//! is clearly worse.

use string2fingerprint::model::EmbeddedModel;
use string2fingerprint::{Category, Scorer};

const ATTACKS: &str = include_str!("data/attacks.tsv");
const BENIGN: &str = include_str!("data/benign.txt");

/// A score at or above this value counts as flagged.
const THRESHOLD: u8 = 128;

const ATTACK_LINES: usize = 63;
const BENIGN_LINES: usize = 72;

/// For each embedded model: the least number of attack lines that it must
/// flag in their own category, and the most benign lines that it may flag in
/// any category. The measured results are 57 and 0 for `compact`, 62 and 0
/// for `default`, 61 and 0 for `large`, and 61 and 0 for `strict`.
fn limits(model: EmbeddedModel) -> (usize, usize) {
    match model {
        EmbeddedModel::Compact => (54, 2),
        EmbeddedModel::Default => (59, 2),
        EmbeddedModel::Large => (58, 2),
        EmbeddedModel::Strict => (58, 1),
        _ => (ATTACK_LINES, 0),
    }
}

fn category(name: &str) -> Category {
    Category::ALL
        .into_iter()
        .find(|category| category.name() == name)
        .unwrap_or_else(|| panic!("unknown category `{name}` in the attack set"))
}

#[test]
fn development_attacks_are_flagged_in_their_category() {
    for model in EmbeddedModel::ALL {
        let scorer = Scorer::with_embedded(model);
        let mut total = 0;
        let mut flagged = 0;
        for (index, line) in ATTACKS.lines().enumerate() {
            let (name, payload) = line.split_once('\t').expect("category, tab, payload");
            let score = scorer.score(payload.as_bytes()).get(category(name));
            total += 1;
            if score >= THRESHOLD {
                flagged += 1;
            } else {
                // Line numbers identify a miss without echoing the payload.
                eprintln!("{model}: attack line {}: {name} scores {score}", index + 1);
            }
        }
        eprintln!("{model}: attacks flagged in their category: {flagged} of {total}");
        assert_eq!(total, ATTACK_LINES);
        assert!(flagged >= limits(model).0, "{model}: {flagged} of {total}");
    }
}

#[test]
fn development_benign_text_is_rarely_flagged() {
    for model in EmbeddedModel::ALL {
        let scorer = Scorer::with_embedded(model);
        let mut total = 0;
        let mut flagged = 0;
        for (index, line) in BENIGN.lines().enumerate() {
            let scores = scorer.score(line.as_bytes());
            total += 1;
            if scores.any_at_least(THRESHOLD) {
                flagged += 1;
                eprintln!("{model}: benign line {}: scores {scores:?}", index + 1);
            }
        }
        eprintln!("{model}: benign lines flagged: {flagged} of {total}");
        assert_eq!(total, BENIGN_LINES);
        assert!(flagged <= limits(model).1, "{model}: {flagged} of {total}");
    }
}

/// For each preset of `Detector`: the least number of attack lines that it
/// must flag in their own category, and the most benign lines that it may
/// flag in any category. The measured results are 57 and 0 for `fast`, 62
/// and 0 for `balanced`, 61 and 0 for `thorough`, and 58 and 0
/// for `strict`.
fn preset_limits(preset: string2fingerprint::Preset) -> (usize, usize) {
    use string2fingerprint::Preset;

    match preset {
        Preset::Fast => (54, 2),
        Preset::Balanced => (59, 2),
        Preset::Thorough => (58, 2),
        Preset::Strict => (55, 1),
        _ => (ATTACK_LINES, 0),
    }
}

#[test]
fn presets_decide_the_development_set() {
    use string2fingerprint::{Detector, Preset};

    for &preset in Preset::ALL {
        let detector = Detector::with_preset(preset);
        let attacks = ATTACKS
            .lines()
            .filter_map(|line| line.split_once('\t'))
            .filter(|(name, payload)| detector.detect(payload.as_bytes()).contains(category(name)))
            .count();
        let benign = BENIGN
            .lines()
            .filter(|line| detector.is_flagged(line.as_bytes()))
            .count();
        eprintln!(
            "{preset}: {attacks} attacks in their category and {benign} benign lines flagged"
        );
        let (least, most) = preset_limits(preset);
        assert!(
            attacks >= least,
            "{preset}: {attacks} of {ATTACK_LINES} attacks"
        );
        assert!(
            benign <= most,
            "{preset}: {benign} of {BENIGN_LINES} benign lines"
        );
    }
}
