//! Measures false positives and false negatives on labeled samples and
//! derives thresholds from them.
//!
//! A sample is the [`Scores`] of one input and the [`CategorySet`] of its
//! labels, and the empty set labels benign text. A sample is a *positive* of
//! every category in its labels and a *negative* of every other category.
//!
//! [`Tally`] counts the outcomes of decisions already made at fixed
//! thresholds. [`Evaluation`] keeps the score distributions of the samples,
//! so that every threshold can be evaluated afterwards and a threshold can be
//! chosen for each category with an [`Objective`].
//!
//! ```
//! use string2fingerprint::eval::{Evaluation, Objective};
//! use string2fingerprint::{Category, CategorySet, Scores, Threshold, Thresholds};
//!
//! // Scores would come from `Scorer::score` on the caller's own samples.
//! let sql = |score| Scores::from_array([0, 0, score, 0, 0, 0, 0, 0]);
//! let mut evaluation = Evaluation::new();
//! evaluation.record(sql(210), Category::SqlInjection.into());
//! evaluation.record(sql(140), Category::SqlInjection.into());
//! evaluation.record(sql(150), CategorySet::EMPTY);
//! evaluation.record(Scores::ZERO, CategorySet::EMPTY);
//!
//! let confusion = evaluation.confusion(Category::SqlInjection, Threshold::at(128));
//! assert_eq!(confusion.true_positives, 2);
//! assert_eq!(confusion.false_positives, 1);
//! assert_eq!(confusion.false_positive_rate(), Some(0.5));
//!
//! // The lowest thresholds that flag no negative sample.
//! let fallback = Thresholds::uniform(Threshold::NEVER);
//! let thresholds = evaluation.thresholds(Objective::MaxFalsePositiveRate(0.0), fallback);
//! assert_eq!(thresholds.get(Category::SqlInjection), Threshold::at(151));
//! assert!(sql(150).flagged(&thresholds).is_empty());
//! assert_eq!(sql(210).flagged(&thresholds), Category::SqlInjection.into());
//! ```
//!
//! Results describe only the samples that were recorded. Choose thresholds on
//! samples that were not used to train the model, and check them on samples
//! that were not used to choose them.
//!
//! # Rates
//!
//! A rate of `k` of `n` is the `f64` quotient `k as f64 / n as f64`, which is
//! the `f64` nearest to `k / n` whenever `n` is below 2^53. A limit on a rate
//! is met when that quotient is within the limit, so a limit written as the
//! exact decimal value of `k / n`, such as 0.03 for 3 of 100, is met by `k`
//! of `n`, and the rates of a [`Confusion`] meet the same limits as the
//! [`Objective`] that chose its threshold.

use core::fmt;
use core::ops::{Add, AddAssign};

use crate::{Category, CategorySet, Scores, Threshold, Thresholds};

/// Counts of the four outcomes of a yes-or-no decision.
///
/// A sample is *labeled* when the decision should say yes and *flagged* when
/// it does. All counts saturate at `u64::MAX` instead of overflowing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Confusion {
    /// Samples that are labeled and flagged.
    pub true_positives: u64,
    /// Samples that are flagged but not labeled.
    pub false_positives: u64,
    /// Samples that are labeled but not flagged.
    pub false_negatives: u64,
    /// Samples that are neither labeled nor flagged.
    pub true_negatives: u64,
}

impl Confusion {
    /// Returns the number of labeled samples.
    #[must_use]
    pub const fn positives(&self) -> u64 {
        self.true_positives.saturating_add(self.false_negatives)
    }

    /// Returns the number of samples that are not labeled.
    #[must_use]
    pub const fn negatives(&self) -> u64 {
        self.false_positives.saturating_add(self.true_negatives)
    }

    /// Returns the number of flagged samples.
    #[must_use]
    pub const fn flagged(&self) -> u64 {
        self.true_positives.saturating_add(self.false_positives)
    }

    /// Returns the number of samples.
    #[must_use]
    pub const fn total(&self) -> u64 {
        self.positives().saturating_add(self.negatives())
    }

    /// Counts one sample.
    pub fn record(&mut self, flagged: bool, labeled: bool) {
        let count = match (flagged, labeled) {
            (true, true) => &mut self.true_positives,
            (true, false) => &mut self.false_positives,
            (false, true) => &mut self.false_negatives,
            (false, false) => &mut self.true_negatives,
        };
        *count = count.saturating_add(1);
    }

    /// Returns the share of labeled samples that are flagged, or `None`
    /// without a labeled sample.
    #[must_use]
    pub fn recall(&self) -> Option<f64> {
        rate(self.true_positives, self.positives())
    }

    /// Returns the share of labeled samples that are not flagged, or `None`
    /// without a labeled sample.
    #[must_use]
    pub fn false_negative_rate(&self) -> Option<f64> {
        rate(self.false_negatives, self.positives())
    }

    /// Returns the share of samples without the label that are flagged, or
    /// `None` without such a sample.
    #[must_use]
    pub fn false_positive_rate(&self) -> Option<f64> {
        rate(self.false_positives, self.negatives())
    }

    /// Returns the share of flagged samples that are labeled, or `None` when
    /// no sample is flagged.
    #[must_use]
    pub fn precision(&self) -> Option<f64> {
        rate(self.true_positives, self.flagged())
    }

    /// Returns the F-score, which weighs recall `beta` times as much as
    /// precision: `(1 + beta^2) * TP / ((1 + beta^2) * TP + beta^2 * FN + FP)`.
    ///
    /// Returns `None` when no sample is labeled or flagged, or when `beta` is
    /// not finite and positive.
    #[must_use]
    pub fn f_score(&self, beta: f64) -> Option<f64> {
        if !(beta.is_finite() && beta > 0.0) {
            return None;
        }
        let errors = self.false_negatives.saturating_add(self.false_positives);
        if self.true_positives.saturating_add(errors) == 0 {
            return None;
        }
        if self.true_positives == 0 {
            return Some(0.0);
        }
        // Dividing by `1 + beta^2` keeps every term finite for any `beta`.
        let squared = beta * beta;
        let (recall_weight, precision_weight) = if squared.is_finite() {
            (squared / (1.0 + squared), 1.0 / (1.0 + squared))
        } else {
            (1.0, 0.0)
        };
        let true_positives = self.true_positives as f64;
        Some(
            true_positives
                / (true_positives
                    + recall_weight * self.false_negatives as f64
                    + precision_weight * self.false_positives as f64),
        )
    }

    /// Returns `false_positive * FP + false_negative * FN`, the cost of the
    /// errors when each false positive costs `false_positive` and each false
    /// negative costs `false_negative`.
    #[must_use]
    pub fn cost(&self, false_positive: f64, false_negative: f64) -> f64 {
        false_positive * self.false_positives as f64 + false_negative * self.false_negatives as f64
    }
}

/// Adds the counts, saturating at `u64::MAX`.
impl Add for Confusion {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            true_positives: self.true_positives.saturating_add(other.true_positives),
            false_positives: self.false_positives.saturating_add(other.false_positives),
            false_negatives: self.false_negatives.saturating_add(other.false_negatives),
            true_negatives: self.true_negatives.saturating_add(other.true_negatives),
        }
    }
}

impl AddAssign for Confusion {
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

/// Returns `count / total` as described in the [module documentation](self),
/// or `None` when `total` is zero.
fn rate(count: u64, total: u64) -> Option<f64> {
    if total == 0 {
        None
    } else {
        Some(count as f64 / total as f64)
    }
}

/// Outcomes of decisions already made at fixed thresholds.
///
/// ```
/// use string2fingerprint::eval::Tally;
/// use string2fingerprint::{Category, CategorySet, Scores, Threshold, Thresholds};
///
/// let thresholds = Thresholds::uniform(Threshold::at(128));
/// let mut tally = Tally::new();
/// let scores = Scores::from_array([0, 200, 200, 0, 0, 0, 0, 0]);
/// tally.record(scores.flagged(&thresholds), Category::SqlInjection.into());
/// assert_eq!(tally.category(Category::SqlInjection).true_positives, 1);
/// assert_eq!(tally.category(Category::Obfuscation).false_positives, 1);
/// assert_eq!(tally.any().true_positives, 1);
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Tally {
    categories: [Confusion; Category::COUNT],
    any: Confusion,
}

impl Tally {
    /// Creates a tally without samples.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Counts one sample: `flagged` are the categories that reached their
    /// threshold, such as the result of [`Scores::flagged`], and `labels`
    /// are the categories that apply to the sample.
    pub fn record(&mut self, flagged: CategorySet, labels: CategorySet) {
        for (category, confusion) in Category::ALL.into_iter().zip(&mut self.categories) {
            confusion.record(flagged.contains(category), labels.contains(category));
        }
        self.any.record(!flagged.is_empty(), !labels.is_empty());
    }

    /// Returns the outcomes for `category`: a sample is labeled when
    /// `labels` holds the category and flagged when `flagged` holds it.
    #[must_use]
    pub fn category(&self, category: Category) -> Confusion {
        self.categories[category.index()]
    }

    /// Returns the outcomes for whole samples: a sample is labeled when
    /// `labels` is not empty and flagged when `flagged` is not empty.
    #[must_use]
    pub fn any(&self) -> Confusion {
        self.any
    }

    /// Returns the number of samples counted.
    #[must_use]
    pub fn samples(&self) -> u64 {
        self.any.total()
    }

    /// Adds the counts of `other`, saturating at `u64::MAX`.
    pub fn merge(&mut self, other: &Self) {
        for (confusion, &other) in self.categories.iter_mut().zip(&other.categories) {
            *confusion += other;
        }
        self.any += other.any;
    }
}

/// The number of distinct scores.
const SCORES: usize = 256;

/// Sample counts by score.
type Histogram = [u64; SCORES];

const EMPTY_HISTOGRAM: Histogram = [0; SCORES];

#[derive(Clone, PartialEq, Eq)]
struct Histograms {
    /// For each category, the samples labeled with it.
    positives: [Histogram; Category::COUNT],
    /// For each category, the samples without labels.
    benign: [Histogram; Category::COUNT],
    /// For each category, the samples with labels other than it.
    others: [Histogram; Category::COUNT],
    /// The highest score of each sample without labels.
    benign_max: Histogram,
    /// The highest score of each sample with labels.
    labeled_max: Histogram,
}

/// Score distributions of labeled samples.
///
/// An evaluation keeps, for each category, how many positive and negative
/// samples have each score. It does not depend on a threshold, so every
/// threshold can be evaluated after the samples are recorded. Memory is fixed
/// at about 53 KB on the heap, whatever the number of samples, and counts
/// saturate at `u64::MAX` instead of overflowing.
#[derive(Clone, PartialEq, Eq)]
pub struct Evaluation {
    histograms: Box<Histograms>,
    samples: u64,
    benign_samples: u64,
}

impl Default for Evaluation {
    fn default() -> Self {
        Self::new()
    }
}

/// Writes the sample counts, not the distributions.
impl fmt::Debug for Evaluation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Evaluation")
            .field("samples", &self.samples)
            .field("benign_samples", &self.benign_samples)
            .finish_non_exhaustive()
    }
}

impl Evaluation {
    /// Creates an evaluation without samples.
    #[must_use]
    pub fn new() -> Self {
        Self {
            histograms: Box::new(Histograms {
                positives: [EMPTY_HISTOGRAM; Category::COUNT],
                benign: [EMPTY_HISTOGRAM; Category::COUNT],
                others: [EMPTY_HISTOGRAM; Category::COUNT],
                benign_max: EMPTY_HISTOGRAM,
                labeled_max: EMPTY_HISTOGRAM,
            }),
            samples: 0,
            benign_samples: 0,
        }
    }

    /// Records one sample: the `scores` of an input and the `labels` that
    /// apply to it.
    pub fn record(&mut self, scores: Scores, labels: CategorySet) {
        let histograms = &mut *self.histograms;
        for (index, (category, score)) in scores.iter().enumerate() {
            let histogram = if labels.contains(category) {
                &mut histograms.positives[index]
            } else if labels.is_empty() {
                &mut histograms.benign[index]
            } else {
                &mut histograms.others[index]
            };
            increment(&mut histogram[usize::from(score)]);
        }
        let max = usize::from(scores.max().1);
        if labels.is_empty() {
            increment(&mut histograms.benign_max[max]);
            increment(&mut self.benign_samples);
        } else {
            increment(&mut histograms.labeled_max[max]);
        }
        increment(&mut self.samples);
    }

    /// Adds the samples of `other`, saturating at `u64::MAX`.
    pub fn merge(&mut self, other: &Self) {
        let (histograms, other_histograms) = (&mut *self.histograms, &*other.histograms);
        let pairs = histograms
            .positives
            .iter_mut()
            .zip(&other_histograms.positives)
            .chain(histograms.benign.iter_mut().zip(&other_histograms.benign))
            .chain(histograms.others.iter_mut().zip(&other_histograms.others))
            .chain([
                (&mut histograms.benign_max, &other_histograms.benign_max),
                (&mut histograms.labeled_max, &other_histograms.labeled_max),
            ]);
        for (histogram, other_histogram) in pairs {
            for (count, &other_count) in histogram.iter_mut().zip(other_histogram) {
                *count = count.saturating_add(other_count);
            }
        }
        self.samples = self.samples.saturating_add(other.samples);
        self.benign_samples = self.benign_samples.saturating_add(other.benign_samples);
    }

    /// Returns the number of samples recorded.
    #[must_use]
    pub fn samples(&self) -> u64 {
        self.samples
    }

    /// Returns the number of samples recorded without labels.
    #[must_use]
    pub fn benign_samples(&self) -> u64 {
        self.benign_samples
    }

    /// Returns a copy in which the negatives of each category are only the
    /// samples without labels.
    ///
    /// A sample labeled with other categories then no longer counts as a
    /// negative of a category. It still counts as a positive of its own
    /// categories, in [`Evaluation::samples`], and in
    /// [`Evaluation::any_confusion`]. Use it when a flag in the wrong attack
    /// category is not a false positive.
    #[must_use]
    pub fn benign_negatives(&self) -> Self {
        let mut evaluation = self.clone();
        evaluation.histograms.others = [EMPTY_HISTOGRAM; Category::COUNT];
        evaluation
    }

    /// Returns the outcomes for `category` at `threshold`: a sample is
    /// labeled when its labels hold the category and flagged when its score
    /// in the category reaches the threshold.
    #[must_use]
    pub fn confusion(&self, category: Category, threshold: Threshold) -> Confusion {
        let histograms = &*self.histograms;
        let index = category.index();
        let (false_negatives, true_positives) = split(&histograms.positives[index], threshold);
        let (benign_below, benign_flagged) = split(&histograms.benign[index], threshold);
        let (others_below, others_flagged) = split(&histograms.others[index], threshold);
        confusion(
            true_positives,
            benign_flagged + others_flagged,
            false_negatives,
            benign_below + others_below,
        )
    }

    /// Returns the outcomes for whole samples at one `threshold` for every
    /// category: a sample is labeled when its labels are not empty and
    /// flagged when its highest score reaches the threshold.
    #[must_use]
    pub fn any_confusion(&self, threshold: Threshold) -> Confusion {
        let histograms = &*self.histograms;
        let (false_negatives, true_positives) = split(&histograms.labeled_max, threshold);
        let (true_negatives, false_positives) = split(&histograms.benign_max, threshold);
        confusion(
            true_positives,
            false_positives,
            false_negatives,
            true_negatives,
        )
    }

    /// Returns the area under the ROC curve of `category`: the share of
    /// pairs of a positive and a negative sample in which the positive scores
    /// higher, where a tie counts half.
    ///
    /// Returns `None` without a positive or without a negative sample. The
    /// area is computed exactly in integers and divided once, unless the
    /// counts are too large for 128-bit products.
    #[must_use]
    pub fn auc(&self, category: Category) -> Option<f64> {
        let histograms = &*self.histograms;
        let index = category.index();
        let positives = &histograms.positives[index];
        let negatives = || {
            histograms.benign[index]
                .iter()
                .zip(&histograms.others[index])
                .map(|(&benign, &others)| u128::from(benign) + u128::from(others))
        };
        let positive_total = sum(positives);
        let negative_total: u128 = negatives().sum();
        if positive_total == 0 || negative_total == 0 {
            return None;
        }
        // Twice the area times the number of pairs: each pair scores 2 when
        // the positive is higher and 1 on a tie.
        let exact = || {
            let mut doubled: u128 = 0;
            let mut below: u128 = 0;
            for (&positive, negative) in positives.iter().zip(negatives()) {
                let weight = below.checked_mul(2)?.checked_add(negative)?;
                doubled = doubled.checked_add(u128::from(positive).checked_mul(weight)?)?;
                below += negative;
            }
            let pairs = positive_total.checked_mul(negative_total)?.checked_mul(2)?;
            Some(doubled as f64 / pairs as f64)
        };
        exact().or_else(|| {
            let (positive_total, negative_total) = (positive_total as f64, negative_total as f64);
            let mut area = 0.0;
            let mut below = 0.0;
            for (&positive, negative) in positives.iter().zip(negatives()) {
                let negative = negative as f64 / negative_total;
                area += positive as f64 / positive_total * (below + negative / 2.0);
                below += negative;
            }
            Some(area.clamp(0.0, 1.0))
        })
    }

    /// Iterates over every threshold in ascending order with the outcomes for
    /// `category` at that threshold, as [`Evaluation::confusion`] returns
    /// them.
    pub fn roc(&self, category: Category) -> impl Iterator<Item = (Threshold, Confusion)> + '_ {
        Threshold::all().zip(self.curve(category))
    }

    /// Returns the threshold of `category` that meets `objective`, or `None`
    /// when the objective is undefined for the category, as [`Objective`]
    /// describes.
    #[must_use]
    pub fn threshold(&self, category: Category, objective: Objective) -> Option<Threshold> {
        let curve = self.curve(category);
        let mut thresholds = Threshold::all().zip(curve);
        match objective {
            Objective::MaxFalsePositiveRate(limit) => {
                if !(0.0..=1.0).contains(&limit) || curve[0].negatives() == 0 {
                    return None;
                }
                thresholds
                    .find(|(_, confusion)| {
                        confusion
                            .false_positive_rate()
                            .is_some_and(|rate| rate <= limit)
                    })
                    .map(|(threshold, _)| threshold)
            }
            Objective::MinRecall(limit) => {
                if !(0.0..=1.0).contains(&limit) || curve[0].positives() == 0 {
                    return None;
                }
                thresholds
                    .rev()
                    .find(|(_, confusion)| confusion.recall().is_some_and(|rate| rate >= limit))
                    .map(|(threshold, _)| threshold)
            }
            Objective::MinCost {
                false_positive,
                false_negative,
            } => {
                let valid = |cost: f64| cost.is_finite() && cost >= 0.0;
                if !valid(false_positive) || !valid(false_negative) || curve[0].total() == 0 {
                    return None;
                }
                let costs = curve.map(|confusion| confusion.cost(false_positive, false_negative));
                let least = costs.iter().copied().fold(f64::INFINITY, f64::min);
                let tied = costs.iter().filter(|&&cost| cost == least).count();
                Threshold::all()
                    .zip(costs)
                    .filter(|&(_, cost)| cost == least)
                    .nth(tied.saturating_sub(1) / 2)
                    .map(|(threshold, _)| threshold)
            }
        }
    }

    /// Returns [`Evaluation::threshold`] for every category. A category for
    /// which it is `None` keeps its threshold from `fallback`.
    #[must_use]
    pub fn thresholds(&self, objective: Objective, fallback: Thresholds) -> Thresholds {
        Category::ALL
            .into_iter()
            .fold(fallback, |thresholds, category| {
                match self.threshold(category, objective) {
                    Some(threshold) => thresholds.with(category, threshold),
                    None => thresholds,
                }
            })
    }

    /// Returns the outcomes for `category` at every threshold, by
    /// [`Threshold::index`].
    fn curve(&self, category: Category) -> [Confusion; Threshold::COUNT] {
        let histograms = &*self.histograms;
        let index = category.index();
        let positives = &histograms.positives[index];
        let negatives = histograms.benign[index]
            .iter()
            .zip(&histograms.others[index])
            .map(|(&benign, &others)| u128::from(benign) + u128::from(others));
        let positive_total = sum(positives);
        let negative_total = sum(&histograms.benign[index]) + sum(&histograms.others[index]);
        let mut curve = [Confusion::default(); Threshold::COUNT];
        let (mut true_positives, mut false_positives) = (0, 0);
        curve[SCORES] = confusion(0, 0, positive_total, negative_total);
        for (score, (&positive, negative)) in positives.iter().zip(negatives).enumerate().rev() {
            true_positives += u128::from(positive);
            false_positives += negative;
            curve[score] = confusion(
                true_positives,
                false_positives,
                positive_total - true_positives,
                negative_total - false_positives,
            );
        }
        curve
    }
}

/// What a threshold chosen by [`Evaluation::threshold`] must achieve.
///
/// Rates are compared as the [module documentation](self#rates) describes.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum Objective {
    /// The lowest threshold, the one that flags the most, whose false
    /// positive rate is at most this rate.
    ///
    /// Undefined when the category has no negative sample or the rate is
    /// not within 0.0 to 1.0. [`Threshold::NEVER`] has the rate 0, so a
    /// valid rate always has a threshold.
    MaxFalsePositiveRate(f64),
    /// The highest threshold, the one that flags the least, whose recall is
    /// at least this value.
    ///
    /// Undefined when the category has no positive sample or the value is
    /// not within 0.0 to 1.0. [`Threshold::ALWAYS`] has the recall 1, so a
    /// valid value always has a threshold.
    MinRecall(f64),
    /// The threshold with the least [cost](Confusion::cost) when each false
    /// positive costs `false_positive` and each false negative costs
    /// `false_negative`.
    ///
    /// When several thresholds have the least cost, the lower median of them
    /// in ascending order, so that a gap between the scores of the positives
    /// and the negatives is split in the middle. Undefined when the category
    /// has no sample or a cost is negative or not finite.
    MinCost {
        /// The cost of one false positive.
        false_positive: f64,
        /// The cost of one false negative.
        false_negative: f64,
    },
}

fn increment(count: &mut u64) {
    *count = count.saturating_add(1);
}

fn sum(counts: &[u64]) -> u128 {
    counts.iter().map(|&count| u128::from(count)).sum()
}

/// Returns the counts of `histogram` below `threshold` and at or above it.
fn split(histogram: &Histogram, threshold: Threshold) -> (u128, u128) {
    let (below, flagged) = histogram.split_at(threshold.index().min(SCORES));
    (sum(below), sum(flagged))
}

/// Creates a confusion from wide counts, saturating each at `u64::MAX`.
fn confusion(
    true_positives: u128,
    false_positives: u128,
    false_negatives: u128,
    true_negatives: u128,
) -> Confusion {
    let narrow = |count: u128| u64::try_from(count).unwrap_or(u64::MAX);
    Confusion {
        true_positives: narrow(true_positives),
        false_positives: narrow(false_positives),
        false_negatives: narrow(false_negatives),
        true_negatives: narrow(true_negatives),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::Rng;

    fn confusion_of(tp: u64, fp: u64, fn_: u64, tn: u64) -> Confusion {
        Confusion {
            true_positives: tp,
            false_positives: fp,
            false_negatives: fn_,
            true_negatives: tn,
        }
    }

    fn only(category: Category, score: u8) -> Scores {
        let mut scores = [0; Category::COUNT];
        scores[category.index()] = score;
        Scores::from_array(scores)
    }

    #[test]
    fn confusion_formulas_match_hand_computed_values() {
        let confusion = confusion_of(6, 2, 3, 9);
        assert_eq!(confusion.positives(), 9);
        assert_eq!(confusion.negatives(), 11);
        assert_eq!(confusion.flagged(), 8);
        assert_eq!(confusion.total(), 20);
        assert_eq!(confusion.recall(), Some(6.0 / 9.0));
        assert_eq!(confusion.false_negative_rate(), Some(3.0 / 9.0));
        assert_eq!(confusion.false_positive_rate(), Some(2.0 / 11.0));
        assert_eq!(confusion.precision(), Some(0.75));
        let f1 = confusion.f_score(1.0).unwrap();
        assert!((f1 - 12.0 / 17.0).abs() < 1e-15);
        let f2 = confusion.f_score(2.0).unwrap();
        assert!((f2 - 30.0 / 44.0).abs() < 1e-15);
        let f_half = confusion.f_score(0.5).unwrap();
        assert!((f_half - 7.5 / 10.25).abs() < 1e-15);
        assert_eq!(confusion.cost(1.0, 1.0), 5.0);
        assert_eq!(confusion.cost(10.0, 0.5), 21.5);
        assert_eq!(confusion.cost(0.0, 0.0), 0.0);
    }

    #[test]
    fn f_score_tends_to_precision_and_recall() {
        let confusion = confusion_of(6, 2, 3, 9);
        let tiny = confusion.f_score(1e-200).unwrap();
        assert!((tiny - 0.75).abs() < 1e-12);
        let huge = confusion.f_score(1e200).unwrap();
        assert!((huge - 6.0 / 9.0).abs() < 1e-12);
        assert!((confusion.f_score(f64::MAX).unwrap() - 6.0 / 9.0).abs() < 1e-12);
        assert!((confusion.f_score(f64::MIN_POSITIVE).unwrap() - 0.75).abs() < 1e-12);
        assert_eq!(confusion_of(0, 1, 0, 5).f_score(1.0), Some(0.0));
        assert_eq!(confusion_of(0, 0, 1, 5).f_score(1e-300), Some(0.0));
        assert_eq!(confusion_of(4, 0, 0, 0).f_score(3.0), Some(1.0));
    }

    #[test]
    fn undefined_rates_are_none() {
        let empty = Confusion::default();
        assert_eq!(empty.total(), 0);
        assert_eq!(empty.recall(), None);
        assert_eq!(empty.false_negative_rate(), None);
        assert_eq!(empty.false_positive_rate(), None);
        assert_eq!(empty.precision(), None);
        assert_eq!(empty.f_score(1.0), None);
        assert_eq!(empty.cost(1.0, 1.0), 0.0);

        let negatives_only = confusion_of(0, 0, 0, 4);
        assert_eq!(negatives_only.recall(), None);
        assert_eq!(negatives_only.precision(), None);
        assert_eq!(negatives_only.false_positive_rate(), Some(0.0));
        assert_eq!(negatives_only.f_score(1.0), None);

        let positives_only = confusion_of(3, 0, 1, 0);
        assert_eq!(positives_only.false_positive_rate(), None);
        assert_eq!(positives_only.recall(), Some(0.75));
        assert_eq!(positives_only.precision(), Some(1.0));

        let confusion = confusion_of(6, 2, 3, 9);
        for beta in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.0] {
            assert_eq!(confusion.f_score(beta), None, "{beta}");
        }
    }

    #[test]
    fn record_and_addition_count_and_saturate() {
        let mut confusion = Confusion::default();
        for (flagged, labeled) in [(true, true), (true, false), (false, true), (false, false)] {
            confusion.record(flagged, labeled);
        }
        confusion.record(true, true);
        assert_eq!(confusion, confusion_of(2, 1, 1, 1));
        assert_eq!(
            confusion + confusion_of(1, 2, 3, 4),
            confusion_of(3, 3, 4, 5)
        );

        let full = confusion_of(u64::MAX, u64::MAX - 1, u64::MAX, 1);
        assert_eq!(full.positives(), u64::MAX);
        assert_eq!(full.negatives(), u64::MAX);
        assert_eq!(full.total(), u64::MAX);
        assert_eq!(full.flagged(), u64::MAX);
        let mut sum = full;
        sum += confusion_of(1, 1, 1, u64::MAX);
        assert_eq!(sum, confusion_of(u64::MAX, u64::MAX, u64::MAX, u64::MAX));
        sum.record(true, true);
        sum.record(false, false);
        assert_eq!(sum, confusion_of(u64::MAX, u64::MAX, u64::MAX, u64::MAX));
        assert_eq!(sum.recall(), Some(1.0));
        assert!(sum.cost(f64::MAX, f64::MAX).is_infinite());
    }

    #[test]
    fn tally_counts_categories_and_samples() {
        let mut tally = Tally::new();
        let sql = CategorySet::from(Category::SqlInjection);
        tally.record(sql | Category::Obfuscation, sql);
        tally.record(CategorySet::EMPTY, sql);
        tally.record(Category::Malformed.into(), CategorySet::EMPTY);
        tally.record(CategorySet::EMPTY, CategorySet::EMPTY);
        assert_eq!(tally.samples(), 4);
        assert_eq!(
            tally.category(Category::SqlInjection),
            confusion_of(1, 0, 1, 2)
        );
        assert_eq!(
            tally.category(Category::Obfuscation),
            confusion_of(0, 1, 0, 3)
        );
        assert_eq!(
            tally.category(Category::Malformed),
            confusion_of(0, 1, 0, 3)
        );
        assert_eq!(
            tally.category(Category::HeaderInjection),
            confusion_of(0, 0, 0, 4)
        );
        assert_eq!(tally.any(), confusion_of(1, 1, 1, 1));

        let mut merged = tally.clone();
        merged.merge(&tally);
        assert_eq!(merged.samples(), 8);
        assert_eq!(merged.any(), confusion_of(2, 2, 2, 2));
        assert_eq!(
            merged.category(Category::SqlInjection),
            confusion_of(2, 0, 2, 4)
        );
        let mut empty = Tally::new();
        empty.merge(&Tally::default());
        assert_eq!(empty, Tally::new());
    }

    #[test]
    fn evaluation_confusion_matches_hand_computed_values() {
        let mut evaluation = Evaluation::new();
        let sql = Category::SqlInjection;
        evaluation.record(only(sql, 200), sql.into());
        evaluation.record(only(sql, 100), sql.into());
        evaluation.record(only(sql, 150), CategorySet::EMPTY);
        evaluation.record(Scores::ZERO, CategorySet::EMPTY);
        evaluation.record(only(sql, 120), Category::Obfuscation.into());
        assert_eq!(evaluation.samples(), 5);
        assert_eq!(evaluation.benign_samples(), 2);

        assert_eq!(
            evaluation.confusion(sql, Threshold::ALWAYS),
            confusion_of(2, 3, 0, 0)
        );
        assert_eq!(
            evaluation.confusion(sql, Threshold::at(100)),
            confusion_of(2, 2, 0, 1)
        );
        assert_eq!(
            evaluation.confusion(sql, Threshold::at(101)),
            confusion_of(1, 2, 1, 1)
        );
        assert_eq!(
            evaluation.confusion(sql, Threshold::at(121)),
            confusion_of(1, 1, 1, 2)
        );
        assert_eq!(
            evaluation.confusion(sql, Threshold::at(151)),
            confusion_of(1, 0, 1, 3)
        );
        assert_eq!(
            evaluation.confusion(sql, Threshold::NEVER),
            confusion_of(0, 0, 2, 3)
        );
        assert_eq!(
            evaluation.confusion(Category::Obfuscation, Threshold::at(1)),
            confusion_of(0, 0, 1, 4)
        );

        assert_eq!(
            evaluation.any_confusion(Threshold::ALWAYS),
            confusion_of(3, 2, 0, 0)
        );
        assert_eq!(
            evaluation.any_confusion(Threshold::at(101)),
            confusion_of(2, 1, 1, 1)
        );
        assert_eq!(
            evaluation.any_confusion(Threshold::at(151)),
            confusion_of(1, 0, 2, 2)
        );
        assert_eq!(
            evaluation.any_confusion(Threshold::NEVER),
            confusion_of(0, 0, 3, 2)
        );

        // Positives 200 and 100 against negatives 150, 120, and 0: four of
        // six pairs are ordered.
        assert_eq!(evaluation.auc(sql), Some(4.0 / 6.0));

        let benign = evaluation.benign_negatives();
        assert_eq!(benign.samples(), 5);
        assert_eq!(benign.benign_samples(), 2);
        assert_eq!(
            benign.confusion(sql, Threshold::at(101)),
            confusion_of(1, 1, 1, 1)
        );
        assert_eq!(
            benign.confusion(Category::Obfuscation, Threshold::ALWAYS),
            confusion_of(1, 2, 0, 0)
        );
        assert_eq!(benign.auc(sql), Some(3.0 / 4.0));
        for threshold in Threshold::all() {
            assert_eq!(
                benign.any_confusion(threshold),
                evaluation.any_confusion(threshold)
            );
        }
        assert_eq!(benign.benign_negatives(), benign);
    }

    #[test]
    fn auc_counts_ties_half_and_needs_both_classes() {
        let category = Category::PathTraversal;
        let mut evaluation = Evaluation::new();
        assert_eq!(evaluation.auc(category), None);
        evaluation.record(only(category, 9), category.into());
        assert_eq!(evaluation.auc(category), None);
        evaluation.record(only(category, 9), CategorySet::EMPTY);
        assert_eq!(evaluation.auc(category), Some(0.5));
        evaluation.record(only(category, 8), CategorySet::EMPTY);
        assert_eq!(evaluation.auc(category), Some(0.75));
        evaluation.record(only(category, 255), CategorySet::EMPTY);
        assert_eq!(evaluation.auc(category), Some(0.5));

        let mut negatives_only = Evaluation::new();
        negatives_only.record(Scores::ZERO, CategorySet::EMPTY);
        assert_eq!(negatives_only.auc(category), None);
    }

    #[test]
    fn rate_limits_accept_exact_decimal_boundaries() {
        let category = Category::Malformed;
        for (flagged, limit) in [(1, 0.01), (3, 0.03), (7, 0.07), (10, 0.1), (29, 0.29)] {
            let mut evaluation = Evaluation::new();
            for sample in 0..100 {
                let score = if sample < flagged { 200 } else { 10 };
                evaluation.record(only(category, score), CategorySet::EMPTY);
            }
            let objective = Objective::MaxFalsePositiveRate(limit);
            let threshold = evaluation.threshold(category, objective);
            assert_eq!(threshold, Some(Threshold::at(11)), "{flagged} of 100");
            let rate = evaluation
                .confusion(category, Threshold::at(11))
                .false_positive_rate()
                .unwrap();
            assert!(rate <= limit);
            let below = Objective::MaxFalsePositiveRate(limit - 1e-9);
            assert_eq!(
                evaluation.threshold(category, below),
                Some(Threshold::at(201))
            );
        }

        let mut evaluation = Evaluation::new();
        for sample in 0..3 {
            let score = if sample == 0 { 200 } else { 10 };
            evaluation.record(only(category, score), CategorySet::EMPTY);
        }
        let third = Objective::MaxFalsePositiveRate(1.0 / 3.0);
        assert_eq!(
            evaluation.threshold(category, third),
            Some(Threshold::at(11))
        );
        let rounded = Objective::MaxFalsePositiveRate(0.333);
        assert_eq!(
            evaluation.threshold(category, rounded),
            Some(Threshold::at(201))
        );

        for (detected, limit) in [(7, 0.7), (3, 0.3), (1, 0.1), (9, 0.9)] {
            let mut evaluation = Evaluation::new();
            for sample in 0..10 {
                let score = if sample < detected { 200 } else { 50 };
                evaluation.record(only(category, score), category.into());
            }
            let objective = Objective::MinRecall(limit);
            assert_eq!(
                evaluation.threshold(category, objective),
                Some(Threshold::at(200)),
                "{detected} of 10"
            );
            let above = Objective::MinRecall(limit + 1e-9);
            assert_eq!(
                evaluation.threshold(category, above),
                Some(Threshold::at(50))
            );
        }
    }

    #[test]
    fn objectives_follow_their_tie_rules() {
        let category = Category::CommandInjection;
        let mut evaluation = Evaluation::new();
        for _ in 0..4 {
            evaluation.record(only(category, 200), category.into());
            evaluation.record(only(category, 50), CategorySet::EMPTY);
        }
        let threshold = |objective| evaluation.threshold(category, objective);
        assert_eq!(
            threshold(Objective::MaxFalsePositiveRate(0.0)),
            Some(Threshold::at(51))
        );
        assert_eq!(
            threshold(Objective::MaxFalsePositiveRate(1.0)),
            Some(Threshold::ALWAYS)
        );
        assert_eq!(
            threshold(Objective::MaxFalsePositiveRate(0.5)),
            Some(Threshold::at(51))
        );
        assert_eq!(
            threshold(Objective::MinRecall(1.0)),
            Some(Threshold::at(200))
        );
        assert_eq!(threshold(Objective::MinRecall(0.0)), Some(Threshold::NEVER));
        // Thresholds 51 to 200 cost nothing, and 125 is their lower median.
        let balanced = Objective::MinCost {
            false_positive: 1.0,
            false_negative: 1.0,
        };
        assert_eq!(threshold(balanced), Some(Threshold::at(125)));
        // Free errors tie every threshold.
        let free = Objective::MinCost {
            false_positive: 0.0,
            false_negative: 0.0,
        };
        assert_eq!(threshold(free), Some(Threshold::at(128)));
        // Free false negatives tie thresholds 51 to 256.
        let fp_only = Objective::MinCost {
            false_positive: 1.0,
            false_negative: 0.0,
        };
        assert_eq!(threshold(fp_only), Some(Threshold::at(153)));
        let fn_only = Objective::MinCost {
            false_positive: 0.0,
            false_negative: 1.0,
        };
        assert_eq!(threshold(fn_only), Some(Threshold::at(100)));

        // Overlapping scores: a negative at 200 and positives at 100 and 200.
        let mut overlap = Evaluation::new();
        overlap.record(only(category, 200), CategorySet::EMPTY);
        overlap.record(only(category, 200), category.into());
        overlap.record(only(category, 100), category.into());
        let expensive_misses = Objective::MinCost {
            false_positive: 1.0,
            false_negative: 3.0,
        };
        assert_eq!(
            overlap.threshold(category, expensive_misses),
            Some(Threshold::at(50))
        );
        let expensive_flags = Objective::MinCost {
            false_positive: 3.0,
            false_negative: 1.0,
        };
        // Thresholds 201 to 256 cost 2, and 228 is their lower median.
        assert_eq!(
            overlap.threshold(category, expensive_flags),
            Some(Threshold::at(228))
        );
    }

    #[test]
    fn undefined_objectives_are_none() {
        let category = Category::TemplateInjection;
        let empty = Evaluation::new();
        let cost = Objective::MinCost {
            false_positive: 1.0,
            false_negative: 1.0,
        };
        for objective in [
            Objective::MaxFalsePositiveRate(0.1),
            Objective::MinRecall(0.9),
            cost,
        ] {
            assert_eq!(empty.threshold(category, objective), None);
        }
        let fallback = Thresholds::uniform(Threshold::at(77));
        assert_eq!(empty.thresholds(cost, fallback), fallback);

        let mut positives = Evaluation::new();
        positives.record(only(category, 9), category.into());
        assert_eq!(
            positives.threshold(category, Objective::MaxFalsePositiveRate(0.1)),
            None
        );
        assert_eq!(
            positives.threshold(category, Objective::MinRecall(1.0)),
            Some(Threshold::at(9))
        );
        assert_eq!(positives.threshold(category, cost), Some(Threshold::at(4)));

        let mut negatives = Evaluation::new();
        negatives.record(only(category, 9), CategorySet::EMPTY);
        assert_eq!(
            negatives.threshold(category, Objective::MinRecall(0.5)),
            None
        );
        assert_eq!(
            negatives.threshold(category, Objective::MaxFalsePositiveRate(0.0)),
            Some(Threshold::at(10))
        );

        let mut both = negatives.clone();
        both.merge(&positives);
        for limit in [
            f64::NAN,
            -0.1,
            1.000_000_1,
            2.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ] {
            assert_eq!(
                both.threshold(category, Objective::MaxFalsePositiveRate(limit)),
                None
            );
            assert_eq!(both.threshold(category, Objective::MinRecall(limit)), None);
        }
        for invalid in [
            f64::NAN,
            -1.0,
            -f64::MIN_POSITIVE,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ] {
            for objective in [
                Objective::MinCost {
                    false_positive: invalid,
                    false_negative: 1.0,
                },
                Objective::MinCost {
                    false_positive: 1.0,
                    false_negative: invalid,
                },
            ] {
                assert_eq!(both.threshold(category, objective), None);
            }
        }
        assert!(
            both.threshold(category, Objective::MinRecall(-0.0))
                .is_some()
        );
        assert!(
            both.threshold(category, Objective::MaxFalsePositiveRate(-0.0))
                .is_some()
        );
        let huge = Objective::MinCost {
            false_positive: f64::MAX,
            false_negative: f64::MAX,
        };
        assert!(both.threshold(category, huge).is_some());
    }

    #[test]
    fn thresholds_keep_the_fallback_where_undefined() {
        let mut evaluation = Evaluation::new();
        evaluation.record(
            only(Category::SqlInjection, 180),
            Category::SqlInjection.into(),
        );
        evaluation.record(
            only(Category::SqlInjection, 90),
            Category::SqlInjection.into(),
        );
        let fallback = Thresholds::uniform(Threshold::at(33));
        let thresholds = evaluation.thresholds(Objective::MinRecall(0.5), fallback);
        assert_eq!(
            thresholds,
            fallback.with(Category::SqlInjection, Threshold::at(180))
        );
        let negatives = evaluation.thresholds(Objective::MaxFalsePositiveRate(0.0), fallback);
        assert_eq!(negatives.get(Category::SqlInjection), Threshold::at(33));
        assert_eq!(negatives.get(Category::Malformed), Threshold::at(1));
    }

    /// One labeled sample of the reference implementation.
    struct Sample {
        scores: Scores,
        labels: CategorySet,
    }

    fn random_samples(rng: &mut Rng, count: usize) -> Vec<Sample> {
        (0..count)
            .map(|_| {
                let mut scores = [0; Category::COUNT];
                for score in &mut scores {
                    // A few distinct values make ties common.
                    let value = rng.next_u32();
                    *score = if value.is_multiple_of(3) {
                        ((value >> 8) % 8 * 36) as u8
                    } else {
                        (value >> 8) as u8
                    };
                }
                let bits = rng.next_u32();
                let labels = if bits.is_multiple_of(2) {
                    CategorySet::EMPTY
                } else {
                    CategorySet::from_bits((bits >> 8) as u8 & (bits >> 16) as u8)
                };
                Sample {
                    scores: Scores::from_array(scores),
                    labels,
                }
            })
            .collect()
    }

    fn naive_confusion(
        samples: &[Sample],
        category: Category,
        threshold: Threshold,
        benign_only: bool,
    ) -> Confusion {
        let mut confusion = Confusion::default();
        for sample in samples {
            let labeled = sample.labels.contains(category);
            if !labeled && benign_only && !sample.labels.is_empty() {
                continue;
            }
            confusion.record(threshold.flags(sample.scores.get(category)), labeled);
        }
        confusion
    }

    fn naive_auc(samples: &[Sample], category: Category) -> Option<f64> {
        let (mut doubled, mut pairs) = (0_u64, 0_u64);
        for positive in samples.iter().filter(|s| s.labels.contains(category)) {
            for negative in samples.iter().filter(|s| !s.labels.contains(category)) {
                let (p, n) = (positive.scores.get(category), negative.scores.get(category));
                doubled += match p.cmp(&n) {
                    core::cmp::Ordering::Greater => 2,
                    core::cmp::Ordering::Equal => 1,
                    core::cmp::Ordering::Less => 0,
                };
                pairs += 1;
            }
        }
        (pairs > 0).then(|| doubled as f64 / (2 * pairs) as f64)
    }

    fn naive_threshold(
        samples: &[Sample],
        category: Category,
        objective: Objective,
    ) -> Option<Threshold> {
        let curve: Vec<(Threshold, Confusion)> = Threshold::all()
            .map(|threshold| {
                (
                    threshold,
                    naive_confusion(samples, category, threshold, false),
                )
            })
            .collect();
        let total = curve[0].1;
        match objective {
            Objective::MaxFalsePositiveRate(limit) => {
                if total.negatives() == 0 || !(0.0..=1.0).contains(&limit) {
                    return None;
                }
                curve
                    .iter()
                    .find(|(_, c)| c.false_positives as f64 / c.negatives() as f64 <= limit)
                    .map(|&(t, _)| t)
            }
            Objective::MinRecall(limit) => {
                if total.positives() == 0 || !(0.0..=1.0).contains(&limit) {
                    return None;
                }
                curve
                    .iter()
                    .rev()
                    .find(|(_, c)| c.true_positives as f64 / c.positives() as f64 >= limit)
                    .map(|&(t, _)| t)
            }
            Objective::MinCost {
                false_positive,
                false_negative,
            } => {
                if total.total() == 0 {
                    return None;
                }
                let cost = |c: &Confusion| {
                    false_positive * c.false_positives as f64
                        + false_negative * c.false_negatives as f64
                };
                let least = curve
                    .iter()
                    .map(|(_, c)| cost(c))
                    .fold(f64::INFINITY, f64::min);
                let tied: Vec<Threshold> = curve
                    .iter()
                    .filter(|(_, c)| cost(c) == least)
                    .map(|&(t, _)| t)
                    .collect();
                Some(tied[(tied.len() - 1) / 2])
            }
        }
    }

    #[test]
    fn evaluation_matches_a_naive_reference() {
        let mut rng = Rng::new(0x5EED_E7A1);
        for round in 0..6 {
            let samples = random_samples(&mut rng, [0, 1, 2, 7, 60, 250][round]);
            let mut evaluation = Evaluation::new();
            for sample in &samples {
                evaluation.record(sample.scores, sample.labels);
            }
            let benign = evaluation.benign_negatives();
            assert_eq!(evaluation.samples(), samples.len() as u64);
            let benign_count = samples.iter().filter(|s| s.labels.is_empty()).count();
            assert_eq!(evaluation.benign_samples(), benign_count as u64);

            for category in Category::ALL {
                let roc: Vec<_> = evaluation.roc(category).collect();
                assert_eq!(roc.len(), Threshold::COUNT);
                for (threshold, (roc_threshold, roc_confusion)) in Threshold::all().zip(roc) {
                    let expected = naive_confusion(&samples, category, threshold, false);
                    assert_eq!(roc_threshold, threshold);
                    assert_eq!(roc_confusion, expected);
                    assert_eq!(evaluation.confusion(category, threshold), expected);
                    assert_eq!(
                        benign.confusion(category, threshold),
                        naive_confusion(&samples, category, threshold, true)
                    );
                }
                let (auc, expected) = (evaluation.auc(category), naive_auc(&samples, category));
                assert_eq!(auc.is_some(), expected.is_some());
                if let (Some(auc), Some(expected)) = (auc, expected) {
                    assert!((auc - expected).abs() < 1e-12, "{auc} {expected}");
                }

                let mut objectives = Vec::new();
                for step in 0..=20 {
                    let limit = f64::from(step) / 20.0;
                    objectives.push(Objective::MaxFalsePositiveRate(limit));
                    objectives.push(Objective::MinRecall(limit));
                }
                for (false_positive, false_negative) in
                    [(1.0, 1.0), (1.0, 5.0), (7.0, 1.0), (0.0, 1.0), (0.3, 0.1)]
                {
                    objectives.push(Objective::MinCost {
                        false_positive,
                        false_negative,
                    });
                }
                for objective in objectives {
                    assert_eq!(
                        evaluation.threshold(category, objective),
                        naive_threshold(&samples, category, objective),
                        "round {round}, {category}, {objective:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn tally_agrees_with_evaluation_at_a_uniform_threshold() {
        let mut rng = Rng::new(42);
        let samples = random_samples(&mut rng, 300);
        let mut evaluation = Evaluation::new();
        for sample in &samples {
            evaluation.record(sample.scores, sample.labels);
        }
        for threshold in Threshold::all().step_by(17).chain([Threshold::NEVER]) {
            let thresholds = Thresholds::uniform(threshold);
            let mut tally = Tally::new();
            for sample in &samples {
                tally.record(sample.scores.flagged(&thresholds), sample.labels);
            }
            assert_eq!(tally.samples(), evaluation.samples());
            assert_eq!(
                tally.any(),
                evaluation.any_confusion(threshold),
                "{threshold}"
            );
            for category in Category::ALL {
                assert_eq!(
                    tally.category(category),
                    evaluation.confusion(category, threshold)
                );
            }
        }
    }

    #[test]
    fn merge_equals_recording_everything() {
        let mut rng = Rng::new(7);
        let samples = random_samples(&mut rng, 120);
        let (first, second) = samples.split_at(50);
        let mut all = Evaluation::new();
        let mut left = Evaluation::new();
        let mut right = Evaluation::new();
        for sample in &samples {
            all.record(sample.scores, sample.labels);
        }
        for sample in first {
            left.record(sample.scores, sample.labels);
        }
        for sample in second {
            right.record(sample.scores, sample.labels);
        }
        let mut merged = left.clone();
        merged.merge(&right);
        assert_eq!(merged, all);
        right.merge(&left);
        assert_eq!(right, all);
        let mut unchanged = all.clone();
        unchanged.merge(&Evaluation::default());
        assert_eq!(unchanged, all);
        assert_ne!(left, all);
        assert_eq!(
            format!("{:?}", Evaluation::new()),
            "Evaluation { samples: 0, benign_samples: 0, .. }"
        );
    }

    #[test]
    fn evaluation_counts_saturate() {
        let category = Category::HeaderInjection;
        let mut evaluation = Evaluation::new();
        evaluation.samples = u64::MAX - 1;
        evaluation.benign_samples = u64::MAX - 1;
        evaluation.histograms.positives[category.index()][255] = u64::MAX - 1;
        evaluation.histograms.benign[category.index()][0] = u64::MAX;
        for _ in 0..3 {
            evaluation.record(only(category, 255), category.into());
            evaluation.record(Scores::ZERO, CategorySet::EMPTY);
        }
        assert_eq!(evaluation.samples(), u64::MAX);
        assert_eq!(evaluation.benign_samples(), u64::MAX);
        assert_eq!(
            evaluation.histograms.positives[category.index()][255],
            u64::MAX
        );
        let full = confusion_of(u64::MAX, 0, 0, u64::MAX);
        assert_eq!(evaluation.confusion(category, Threshold::at(255)), full);
        assert_eq!(evaluation.auc(category), Some(1.0));

        // Every bin full: totals exceed `u64::MAX` and products exceed 128
        // bits, so counts clamp and the area falls back to floating point.
        let histograms = &mut *evaluation.histograms;
        for histogram in histograms
            .positives
            .iter_mut()
            .chain(&mut histograms.benign)
            .chain(&mut histograms.others)
            .chain([&mut histograms.benign_max, &mut histograms.labeled_max])
        {
            *histogram = [u64::MAX; SCORES];
        }
        let mut doubled = evaluation.clone();
        doubled.merge(&evaluation);
        assert_eq!(doubled, evaluation);
        let all_max = confusion_of(u64::MAX, u64::MAX, u64::MAX, u64::MAX);
        assert_eq!(evaluation.confusion(category, Threshold::at(128)), all_max);
        assert_eq!(
            evaluation.confusion(category, Threshold::NEVER),
            confusion_of(0, 0, u64::MAX, u64::MAX)
        );
        let auc = evaluation.auc(category).unwrap();
        assert!((auc - 0.5).abs() < 1e-9, "{auc}");
        for objective in [
            Objective::MaxFalsePositiveRate(0.5),
            Objective::MinRecall(0.5),
            Objective::MinCost {
                false_positive: 1.0,
                false_negative: 1.0,
            },
        ] {
            assert!(evaluation.threshold(category, objective).is_some());
        }
        assert_eq!(evaluation.roc(category).count(), Threshold::COUNT);
    }
}
