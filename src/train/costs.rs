//! Weights of the two kinds of error in the training loss.

use super::dataset::Dataset;
use crate::Category;

/// How much training cares about each kind of error.
///
/// The loss has one term for each category of each window. A term whose
/// category labels the window pushes the score up, and its weight is the
/// cost of a *false negative* in that category: a missed attack. A term
/// whose category does not label the window pushes the score down, and its
/// weight is the cost of a *false positive*: a wrong flag. A window without
/// any label is benign, and the weight of each of its terms is also
/// multiplied by the benign weight.
///
/// A cost above 1 makes the model avoid that error at the expense of the
/// opposite one. All costs are 1 by default.
///
/// ```
/// use string2fingerprint::Category;
/// use string2fingerprint::train::Costs;
///
/// // A host that blocks on a score: wrong flags on benign text cost four
/// // times as much, and a missed command injection twice as much.
/// let costs = Costs::UNIFORM
///     .benign(4.0)
///     .false_negative(Category::CommandInjection, 2.0);
/// assert_eq!(costs.false_negative_cost(Category::CommandInjection), 2.0);
/// assert_eq!(costs.false_positive_cost(Category::CommandInjection), 1.0);
/// ```
///
/// Costs shift the balance that the model learns. They do not replace a
/// [threshold](crate::Thresholds) chosen on held-out samples, and the two
/// work together: costs move the scores apart where it matters, and the
/// threshold picks the operating point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Costs {
    false_positive: [f32; Category::COUNT],
    false_negative: [f32; Category::COUNT],
    benign: f32,
}

impl Default for Costs {
    fn default() -> Self {
        Self::UNIFORM
    }
}

impl Costs {
    /// Every error costs the same.
    pub const UNIFORM: Self = Self {
        false_positive: [1.0; Category::COUNT],
        false_negative: [1.0; Category::COUNT],
        benign: 1.0,
    };

    /// Sets the cost of flagging `category` on a window that it does not
    /// label.
    #[must_use]
    pub const fn false_positive(mut self, category: Category, cost: f32) -> Self {
        self.false_positive[category.index()] = cost;
        self
    }

    /// Sets the cost of not flagging `category` on a window that it labels.
    #[must_use]
    pub const fn false_negative(mut self, category: Category, cost: f32) -> Self {
        self.false_negative[category.index()] = cost;
        self
    }

    /// Sets the cost of a wrong flag in every category.
    #[must_use]
    pub const fn false_positives(mut self, cost: f32) -> Self {
        self.false_positive = [cost; Category::COUNT];
        self
    }

    /// Sets the cost of a missed label in every category.
    #[must_use]
    pub const fn false_negatives(mut self, cost: f32) -> Self {
        self.false_negative = [cost; Category::COUNT];
        self
    }

    /// Sets the factor on every term of a window without any label.
    ///
    /// The embedded `strict` model was trained with a benign weight of 4.
    #[must_use]
    pub const fn benign(mut self, weight: f32) -> Self {
        self.benign = weight;
        self
    }

    /// Returns the cost of a wrong flag in `category`.
    #[must_use]
    pub const fn false_positive_cost(&self, category: Category) -> f32 {
        self.false_positive[category.index()]
    }

    /// Returns the cost of a missed label in `category`.
    #[must_use]
    pub const fn false_negative_cost(&self, category: Category) -> f32 {
        self.false_negative[category.index()]
    }

    /// Returns the factor on the terms of a benign window.
    #[must_use]
    pub const fn benign_weight(&self) -> f32 {
        self.benign
    }

    /// Returns costs that give the labeled and the unlabeled samples of
    /// each category the same total weight, within a factor of `limit`.
    ///
    /// The cost of a false negative in a category is the number of samples
    /// without its label divided by the number with it, clamped to the
    /// range from `1 / limit` to `limit`. A category without a labeled
    /// sample keeps the cost 1. A rare category then counts as much as a
    /// common one, which raises its recall and its false positives. A
    /// `limit` that is not above 1 gives [`Costs::UNIFORM`].
    #[must_use]
    pub fn balanced(data: &Dataset, limit: f32) -> Self {
        let mut costs = Self::UNIFORM;
        if limit.is_nan() || limit <= 1.0 {
            return costs;
        }
        for category in Category::ALL {
            let positives = data.count(category);
            if positives > 0 {
                let ratio = (data.samples() - positives) as f32 / positives as f32;
                costs.false_negative[category.index()] = ratio.clamp(1.0 / limit, limit);
            }
        }
        costs
    }

    /// Returns whether every cost is finite and not negative.
    pub(super) fn is_valid(&self) -> bool {
        self.false_positive
            .iter()
            .chain(&self.false_negative)
            .chain([&self.benign])
            .all(|cost| cost.is_finite() && *cost >= 0.0)
    }

    /// Returns the weight of the term of `category` for a window.
    #[inline]
    pub(super) fn weight(&self, category: Category, labeled: bool, benign: bool) -> f32 {
        let cost = if labeled {
            self.false_negative[category.index()]
        } else {
            self.false_positive[category.index()]
        };
        if benign { cost * self.benign } else { cost }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::dataset::Sample;

    #[test]
    fn uniform_costs_weigh_every_term_once() {
        assert_eq!(Costs::default(), Costs::UNIFORM);
        assert!(Costs::UNIFORM.is_valid());
        for category in Category::ALL {
            for (labeled, benign) in [(true, false), (false, false), (false, true)] {
                assert_eq!(Costs::UNIFORM.weight(category, labeled, benign), 1.0);
            }
            assert_eq!(Costs::UNIFORM.false_positive_cost(category), 1.0);
            assert_eq!(Costs::UNIFORM.false_negative_cost(category), 1.0);
        }
        assert_eq!(Costs::UNIFORM.benign_weight(), 1.0);
    }

    #[test]
    fn each_cost_applies_to_its_terms() {
        let costs = Costs::UNIFORM
            .false_positive(Category::SqlInjection, 3.0)
            .false_negative(Category::SqlInjection, 5.0)
            .benign(2.0);
        assert_eq!(costs.weight(Category::SqlInjection, true, false), 5.0);
        assert_eq!(costs.weight(Category::SqlInjection, false, false), 3.0);
        assert_eq!(costs.weight(Category::SqlInjection, false, true), 6.0);
        assert_eq!(costs.weight(Category::Malformed, false, true), 2.0);
        assert_eq!(costs.weight(Category::Malformed, false, false), 1.0);
        assert_eq!(costs.weight(Category::Malformed, true, false), 1.0);
        assert_eq!(costs.benign_weight(), 2.0);

        let all = Costs::UNIFORM.false_positives(0.5).false_negatives(7.0);
        for category in Category::ALL {
            assert_eq!(all.false_positive_cost(category), 0.5);
            assert_eq!(all.false_negative_cost(category), 7.0);
        }
    }

    #[test]
    fn invalid_costs_are_detected() {
        for cost in [-1.0, f32::NAN, f32::INFINITY, -0.001] {
            assert!(!Costs::UNIFORM.benign(cost).is_valid());
            assert!(
                !Costs::UNIFORM
                    .false_positive(Category::Malformed, cost)
                    .is_valid()
            );
            assert!(
                !Costs::UNIFORM
                    .false_negative(Category::HeaderInjection, cost)
                    .is_valid()
            );
        }
        assert!(Costs::UNIFORM.benign(0.0).false_negatives(0.0).is_valid());
    }

    #[test]
    fn balanced_costs_follow_the_label_counts() {
        let mut data = Dataset::new();
        for _ in 0..12 {
            data.push(Sample::benign(b"plain text")).unwrap();
        }
        for _ in 0..4 {
            data.push(Sample::new(b"' OR 1=1--", Category::SqlInjection))
                .unwrap();
        }
        data.push(Sample::new(b"../../etc/passwd", Category::PathTraversal))
            .unwrap();
        for _ in 0..15 {
            data.push(Sample::new(b"%zz%", Category::Malformed))
                .unwrap();
        }
        // 32 samples: 4, 1, and 15 labeled.
        let costs = Costs::balanced(&data, 10.0);
        assert_eq!(costs.false_negative_cost(Category::SqlInjection), 7.0);
        assert_eq!(costs.false_negative_cost(Category::PathTraversal), 10.0);
        assert_eq!(costs.false_negative_cost(Category::Malformed), 17.0 / 15.0);
        assert_eq!(costs.false_negative_cost(Category::HeaderInjection), 1.0);
        assert_eq!(costs.benign_weight(), 1.0);
        for category in Category::ALL {
            assert_eq!(costs.false_positive_cost(category), 1.0);
        }
        assert!(costs.is_valid());

        for _ in 0..40 {
            data.push(Sample::new(b"%zz%", Category::Malformed))
                .unwrap();
        }
        let costs = Costs::balanced(&data, 2.0);
        assert_eq!(costs.false_negative_cost(Category::Malformed), 0.5);
        assert_eq!(costs.false_negative_cost(Category::SqlInjection), 2.0);

        for limit in [1.0, 0.5, 0.0, -3.0, f32::NAN] {
            assert_eq!(Costs::balanced(&data, limit), Costs::UNIFORM);
        }
        assert_eq!(Costs::balanced(&Dataset::new(), 10.0), Costs::UNIFORM);
        assert!(Costs::balanced(&data, f32::INFINITY).is_valid());
    }
}
