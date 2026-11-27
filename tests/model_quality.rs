//! Guards the embedded model against a quality regression.
//!
//! `tests/data/attacks.tsv` and `tests/data/benign.txt` are a small
//! hand-written development set. Its strings do not come from the generator
//! that produces the training data, so it shows whether the model carries
//! over to text of a different origin. It was consulted while the
//! fingerprint layout and the training data were designed, so it is not a
//! held-out test set, and passing it says nothing about real traffic.
//!
//! The limits below are looser than the results of the embedded model, which
//! `tools/train/README.md` records. A retrained model that fails them is
//! clearly worse.

use string2fingerprint::{Category, Scorer};

const ATTACKS: &str = include_str!("data/attacks.tsv");
const BENIGN: &str = include_str!("data/benign.txt");

/// A score at or above this value counts as flagged.
const THRESHOLD: u8 = 128;

/// At least this many attack lines must be flagged in their own category.
const MIN_ATTACKS_FLAGGED: usize = 57;

/// At most this many benign lines may be flagged in any category.
const MAX_BENIGN_FLAGGED: usize = 3;

fn category(name: &str) -> Category {
    Category::ALL
        .into_iter()
        .find(|category| category.name() == name)
        .unwrap_or_else(|| panic!("unknown category `{name}` in the attack set"))
}

#[test]
fn development_attacks_are_flagged_in_their_category() {
    let scorer = Scorer::new();
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
            eprintln!("attack line {}: {name} scores {score}", index + 1);
        }
    }
    eprintln!("attacks flagged in their category: {flagged} of {total}");
    assert_eq!(total, 63);
    assert!(flagged >= MIN_ATTACKS_FLAGGED, "{flagged} of {total}");
}

#[test]
fn development_benign_text_is_rarely_flagged() {
    let scorer = Scorer::new();
    let mut total = 0;
    let mut flagged = 0;
    for (index, line) in BENIGN.lines().enumerate() {
        let scores = scorer.score(line.as_bytes());
        total += 1;
        if scores.any_at_least(THRESHOLD) {
            flagged += 1;
            eprintln!("benign line {}: scores {scores:?}", index + 1);
        }
    }
    eprintln!("benign lines flagged: {flagged} of {total}");
    assert_eq!(total, 72);
    assert!(flagged <= MAX_BENIGN_FLAGGED, "{flagged} of {total}");
}
