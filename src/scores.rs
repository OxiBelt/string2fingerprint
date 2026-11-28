//! Per-category scores.

use core::ops::Index;

use crate::{Category, CategorySet, Threshold, Thresholds};

/// One score for each [`Category`].
///
/// A score is an integer from 0 to 255. Higher means the input looks more
/// like the category. Scores are model confidences, they are not calibrated
/// probabilities, and the caller chooses the threshold that fits its
/// tolerance for false positives.
///
/// Scores are computed with integer arithmetic only, so one model gives the
/// same scores for the same input on every supported CPU and backend.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Scores([u8; Category::COUNT]);

impl Scores {
    /// The scores of an input that shows no anomaly, such as the empty input.
    pub const ZERO: Self = Self([0; Category::COUNT]);

    /// Creates scores from an array indexed by [`Category::index`].
    #[must_use]
    pub const fn from_array(scores: [u8; Category::COUNT]) -> Self {
        Self(scores)
    }

    /// Returns the scores as an array indexed by [`Category::index`].
    #[must_use]
    pub const fn as_array(&self) -> &[u8; Category::COUNT] {
        &self.0
    }

    /// Returns the score of `category`.
    #[must_use]
    pub const fn get(&self, category: Category) -> u8 {
        self.0[category.index()]
    }

    /// Returns the score of `category` scaled to the range from 0.0 to 1.0.
    #[must_use]
    pub fn fraction(&self, category: Category) -> f32 {
        f32::from(self.get(category)) / 255.0
    }

    /// Returns the highest score and its category.
    ///
    /// When several categories share the highest score, the one with the
    /// lowest discriminant is returned.
    #[must_use]
    pub fn max(&self) -> (Category, u8) {
        let mut best = (Category::ALL[0], self.0[0]);
        for (category, score) in self.iter() {
            if score > best.1 {
                best = (category, score);
            }
        }
        best
    }

    /// Returns whether any score is at least `threshold`.
    #[must_use]
    pub fn any_at_least(&self, threshold: u8) -> bool {
        self.0.iter().any(|&score| score >= threshold)
    }

    /// Returns the categories whose score is at least `threshold`.
    ///
    /// [`Scores::flagged`] takes a threshold for each category.
    #[must_use]
    pub fn at_least(&self, threshold: u8) -> CategorySet {
        self.flagged(&Thresholds::uniform(Threshold::at(threshold)))
    }

    /// Returns the categories whose score reaches its threshold.
    ///
    /// With the labels of an input as a [`CategorySet`], `flagged - labels`
    /// are its false positives and `labels - flagged` its false negatives.
    ///
    /// ```
    /// use string2fingerprint::{Category, Scores, Threshold, Thresholds};
    ///
    /// let scores = Scores::from_array([0, 200, 200, 0, 0, 0, 0, 0]);
    /// let thresholds =
    ///     Thresholds::uniform(Threshold::at(128)).with(Category::Obfuscation, Threshold::at(224));
    /// assert_eq!(scores.flagged(&thresholds), Category::SqlInjection.into());
    /// ```
    #[must_use]
    pub fn flagged(&self, thresholds: &Thresholds) -> CategorySet {
        thresholds.flagged(&self.0)
    }

    /// Iterates over every category and its score, ordered by discriminant.
    pub fn iter(&self) -> impl Iterator<Item = (Category, u8)> + '_ {
        Category::ALL.into_iter().zip(self.0)
    }

    /// Keeps, for each category, the higher of the two scores.
    #[must_use]
    pub fn max_with(mut self, other: Self) -> Self {
        for (score, other_score) in self.0.iter_mut().zip(other.0) {
            *score = (*score).max(other_score);
        }
        self
    }
}

impl Index<Category> for Scores {
    type Output = u8;

    fn index(&self, category: Category) -> &u8 {
        &self.0[category.index()]
    }
}

#[cfg(test)]
mod tests {
    use super::Scores;
    use crate::{Category, CategorySet, Threshold, Thresholds};

    #[test]
    fn zero_is_default() {
        assert_eq!(Scores::ZERO, Scores::default());
        assert!(!Scores::ZERO.any_at_least(1));
        assert!(Scores::ZERO.any_at_least(0));
        assert_eq!(Scores::ZERO.max(), (Category::Malformed, 0));
    }

    #[test]
    fn get_and_index_follow_category_order() {
        let scores = Scores::from_array([1, 2, 3, 4, 5, 6, 7, 8]);
        for (index, category) in Category::ALL.into_iter().enumerate() {
            let expected = u8::try_from(index + 1).unwrap();
            assert_eq!(scores.get(category), expected);
            assert_eq!(scores[category], expected);
        }
        assert_eq!(scores.as_array(), &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(scores.iter().count(), Category::COUNT);
    }

    #[test]
    fn max_prefers_lowest_discriminant_on_ties() {
        let scores = Scores::from_array([0, 9, 200, 9, 200, 0, 0, 1]);
        assert_eq!(scores.max(), (Category::SqlInjection, 200));
        assert!(scores.any_at_least(200));
        assert!(!scores.any_at_least(201));
    }

    #[test]
    fn flagged_categories_follow_the_thresholds() {
        let scores = Scores::from_array([0, 9, 200, 9, 200, 0, 255, 1]);
        assert_eq!(scores.at_least(0), CategorySet::ALL);
        assert_eq!(
            scores.at_least(200),
            CategorySet::from(Category::SqlInjection)
                | Category::PathTraversal
                | Category::TemplateInjection
        );
        assert_eq!(scores.at_least(255), Category::TemplateInjection.into());
        for threshold in 0..=u8::MAX {
            assert_eq!(
                !scores.at_least(threshold).is_empty(),
                scores.any_at_least(threshold)
            );
        }
        assert_eq!(
            scores.flagged(&Thresholds::uniform(Threshold::NEVER)),
            CategorySet::EMPTY
        );
        let thresholds = Thresholds::uniform(Threshold::NEVER)
            .with(Category::Obfuscation, Threshold::at(9))
            .with(Category::MarkupInjection, Threshold::at(10));
        assert_eq!(scores.flagged(&thresholds), Category::Obfuscation.into());
        assert_eq!(Scores::ZERO.at_least(1), CategorySet::EMPTY);
    }

    #[test]
    fn fraction_spans_unit_interval() {
        let scores = Scores::from_array([0, 255, 51, 0, 0, 0, 0, 0]);
        assert_eq!(scores.fraction(Category::Malformed), 0.0);
        assert_eq!(scores.fraction(Category::Obfuscation), 1.0);
        assert!((scores.fraction(Category::SqlInjection) - 0.2).abs() < 1e-6);
    }

    #[test]
    fn max_with_is_elementwise() {
        let left = Scores::from_array([1, 9, 3, 0, 255, 0, 7, 0]);
        let right = Scores::from_array([2, 8, 3, 0, 0, 1, 6, 0]);
        let merged = Scores::from_array([2, 9, 3, 0, 255, 1, 7, 0]);
        assert_eq!(left.max_with(right), merged);
        assert_eq!(right.max_with(left), merged);
        assert_eq!(left.max_with(Scores::ZERO), left);
    }
}
