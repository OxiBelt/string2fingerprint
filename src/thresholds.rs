//! Thresholds that turn scores into decisions.

use core::fmt;

use crate::{Category, CategorySet};

/// The score from which a category counts as flagged.
///
/// A threshold of `t` flags every score that is at least `t`, so a threshold
/// of 0 flags every input. [`Threshold::NEVER`] flags no score, which turns a
/// category off.
///
/// ```
/// use string2fingerprint::Threshold;
///
/// assert!(Threshold::at(192).flags(192));
/// assert!(!Threshold::at(192).flags(191));
/// assert!(!Threshold::NEVER.flags(255));
/// assert!(Threshold::at(0) < Threshold::at(255) && Threshold::at(255) < Threshold::NEVER);
/// ```
///
/// Thresholds are ordered from the one that flags the most to the one that
/// flags the least.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Threshold(u16);

impl Threshold {
    /// The threshold that every score reaches.
    pub const ALWAYS: Self = Self(0);

    /// The threshold that no score reaches.
    pub const NEVER: Self = Self(256);

    /// The number of distinct thresholds: one for each score, and
    /// [`Threshold::NEVER`].
    pub const COUNT: usize = 257;

    /// Creates the threshold that flags scores of at least `score`.
    #[must_use]
    pub const fn at(score: u8) -> Self {
        Self(score as u16)
    }

    /// Returns the lowest score that is flagged, or `None` for
    /// [`Threshold::NEVER`].
    #[must_use]
    pub const fn score(self) -> Option<u8> {
        if self.0 <= u8::MAX as u16 {
            Some(self.0 as u8)
        } else {
            None
        }
    }

    /// Returns whether `score` is flagged.
    #[must_use]
    pub const fn flags(self, score: u8) -> bool {
        score as u16 >= self.0
    }

    /// Returns the position of this threshold among all thresholds in
    /// ascending order, below [`Threshold::COUNT`].
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    /// Returns the threshold at `index` in ascending order, if any.
    #[must_use]
    pub const fn from_index(index: usize) -> Option<Self> {
        if index < Self::COUNT {
            Some(Self(index as u16))
        } else {
            None
        }
    }

    /// Iterates over every threshold in ascending order.
    pub fn all() -> impl DoubleEndedIterator<Item = Self> + ExactSizeIterator {
        (0..=Self::NEVER.0).map(Self)
    }
}

impl From<u8> for Threshold {
    fn from(score: u8) -> Self {
        Self::at(score)
    }
}

/// Writes the lowest flagged score, or `never`.
impl fmt::Display for Threshold {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.score() {
            Some(score) => write!(f, "{score}"),
            None => f.write_str("never"),
        }
    }
}

/// One [`Threshold`] for each [`Category`].
///
/// Categories differ in how well a model separates them, so one number for
/// all of them either misses attacks in a hard category or flags benign text
/// in an easy one. [`Scores::flagged`](crate::Scores::flagged) applies a
/// threshold to each category.
///
/// ```
/// use string2fingerprint::{Category, Scorer, Threshold, Thresholds};
///
/// let thresholds = Thresholds::uniform(Threshold::at(192))
///     .with(Category::CommandInjection, Threshold::at(230))
///     .with(Category::Malformed, Threshold::NEVER);
///
/// let scores = Scorer::new().score(b"/search?q=1%27%20OR%20%271%27=%271");
/// let flagged = scores.flagged(&thresholds);
/// assert!(!flagged.contains(Category::Malformed));
/// ```
///
/// Thresholds are the caller's choice. The evaluation tools of this crate
/// can derive them from labeled samples of the caller's own traffic.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Thresholds([Threshold; Category::COUNT]);

impl Thresholds {
    /// The number of bytes of [`Thresholds::to_bytes`].
    pub const BYTES: usize = 2 * Category::COUNT;

    /// Creates the same threshold for every category.
    #[must_use]
    pub const fn uniform(threshold: Threshold) -> Self {
        Self([threshold; Category::COUNT])
    }

    /// Creates thresholds from an array indexed by [`Category::index`].
    #[must_use]
    pub const fn from_array(thresholds: [Threshold; Category::COUNT]) -> Self {
        Self(thresholds)
    }

    /// Returns the thresholds as an array indexed by [`Category::index`].
    #[must_use]
    pub const fn as_array(&self) -> &[Threshold; Category::COUNT] {
        &self.0
    }

    /// Returns the threshold of `category`.
    #[must_use]
    pub const fn get(&self, category: Category) -> Threshold {
        self.0[category.index()]
    }

    /// Returns the thresholds with the one of `category` replaced.
    #[must_use]
    pub const fn with(mut self, category: Category, threshold: Threshold) -> Self {
        self.0[category.index()] = threshold;
        self
    }

    /// Iterates over every category and its threshold, ordered by
    /// discriminant.
    pub fn iter(&self) -> impl Iterator<Item = (Category, Threshold)> + '_ {
        Category::ALL.into_iter().zip(self.0)
    }

    /// Returns the categories whose score in `scores`, an array indexed by
    /// [`Category::index`], is flagged.
    pub(crate) fn flagged(&self, scores: &[u8; Category::COUNT]) -> CategorySet {
        let mut bits = 0;
        for (index, (threshold, &score)) in self.0.iter().zip(scores).enumerate() {
            bits |= u8::from(threshold.flags(score)) << index;
        }
        CategorySet::from_bits(bits)
    }

    /// Returns the thresholds as bytes: for each category by index, the
    /// little-endian 16-bit position of its threshold in ascending order,
    /// which is the lowest flagged score, or 256 for [`Threshold::NEVER`].
    ///
    /// The form is stable, so that thresholds can be stored next to the
    /// model they were chosen for.
    #[must_use]
    pub const fn to_bytes(&self) -> [u8; Self::BYTES] {
        let mut bytes = [0; Self::BYTES];
        let mut index = 0;
        while index < Category::COUNT {
            let value = self.0[index].0.to_le_bytes();
            bytes[2 * index] = value[0];
            bytes[2 * index + 1] = value[1];
            index += 1;
        }
        bytes
    }

    /// Reads the form that [`Thresholds::to_bytes`] writes.
    ///
    /// Returns `None` when a value is above 256.
    #[must_use]
    pub const fn from_bytes(bytes: &[u8; Self::BYTES]) -> Option<Self> {
        let mut thresholds = [Threshold::ALWAYS; Category::COUNT];
        let mut index = 0;
        while index < Category::COUNT {
            let value = u16::from_le_bytes([bytes[2 * index], bytes[2 * index + 1]]);
            if value > Threshold::NEVER.0 {
                return None;
            }
            thresholds[index] = Threshold(value);
            index += 1;
        }
        Some(Self(thresholds))
    }
}

impl From<Threshold> for Thresholds {
    fn from(threshold: Threshold) -> Self {
        Self::uniform(threshold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_threshold_flags_scores_at_or_above_it() {
        for threshold in 0..=u8::MAX {
            for score in 0..=u8::MAX {
                assert_eq!(Threshold::at(threshold).flags(score), score >= threshold);
            }
        }
        for score in 0..=u8::MAX {
            assert!(Threshold::ALWAYS.flags(score));
            assert!(!Threshold::NEVER.flags(score));
        }
        assert_eq!(Threshold::ALWAYS, Threshold::at(0));
        assert_eq!(Threshold::from(7), Threshold::at(7));
    }

    #[test]
    fn thresholds_are_ordered_and_indexed() {
        let all: Vec<Threshold> = Threshold::all().collect();
        assert_eq!(all.len(), Threshold::COUNT);
        assert_eq!(all.first(), Some(&Threshold::ALWAYS));
        assert_eq!(all.last(), Some(&Threshold::NEVER));
        assert!(all.is_sorted());
        for (index, &threshold) in all.iter().enumerate() {
            assert_eq!(threshold.index(), index);
            assert_eq!(Threshold::from_index(index), Some(threshold));
            assert_eq!(threshold.score(), u8::try_from(index).ok());
        }
        assert_eq!(Threshold::from_index(Threshold::COUNT), None);
        assert_eq!(Threshold::from_index(usize::MAX), None);
    }

    #[test]
    fn a_threshold_displays_its_score_or_never() {
        assert_eq!(Threshold::at(0).to_string(), "0");
        assert_eq!(Threshold::at(255).to_string(), "255");
        assert_eq!(Threshold::NEVER.to_string(), "never");
    }

    #[test]
    fn thresholds_apply_to_each_category() {
        let thresholds = Thresholds::uniform(Threshold::at(100))
            .with(Category::Obfuscation, Threshold::at(101))
            .with(Category::HeaderInjection, Threshold::NEVER)
            .with(Category::SqlInjection, Threshold::ALWAYS);
        assert_eq!(thresholds.get(Category::Malformed), Threshold::at(100));
        assert_eq!(thresholds.get(Category::Obfuscation), Threshold::at(101));
        assert_eq!(thresholds.iter().count(), Category::COUNT);
        assert_eq!(thresholds.as_array()[7], Threshold::NEVER);
        assert_eq!(Thresholds::from_array(*thresholds.as_array()), thresholds);
        assert_eq!(
            Thresholds::from(Threshold::at(9)),
            Thresholds::uniform(Threshold::at(9))
        );

        let flagged = thresholds.flagged(&[100, 100, 0, 99, 255, 0, 100, 255]);
        let expected = CategorySet::from(Category::Malformed)
            | Category::SqlInjection
            | Category::PathTraversal
            | Category::TemplateInjection;
        assert_eq!(flagged, expected);
    }

    #[test]
    fn bytes_round_trip_and_reject_values_above_never() {
        let thresholds = Thresholds::uniform(Threshold::at(128))
            .with(Category::Malformed, Threshold::ALWAYS)
            .with(Category::Obfuscation, Threshold::at(255))
            .with(Category::HeaderInjection, Threshold::NEVER);
        let bytes = thresholds.to_bytes();
        assert_eq!(bytes[..4], [0, 0, 255, 0]);
        assert_eq!(bytes[4..6], [128, 0]);
        assert_eq!(bytes[14..], [0, 1]);
        assert_eq!(Thresholds::from_bytes(&bytes), Some(thresholds));

        for index in 0..Category::COUNT {
            let mut invalid = bytes;
            invalid[2 * index] = 1;
            invalid[2 * index + 1] = 1;
            assert_eq!(Thresholds::from_bytes(&invalid), None);
            invalid[2 * index + 1] = 0xFF;
            assert_eq!(Thresholds::from_bytes(&invalid), None);
        }
    }

    #[test]
    fn bytes_are_usable_in_constants() {
        const STORED: [u8; Thresholds::BYTES] = Thresholds::uniform(Threshold::at(200)).to_bytes();
        const LOADED: Option<Thresholds> = Thresholds::from_bytes(&STORED);
        assert_eq!(LOADED, Some(Thresholds::uniform(Threshold::at(200))));
    }
}
