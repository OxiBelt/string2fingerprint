//! Labeled samples for training and evaluation.

use core::fmt;

use super::math::Rng;
use crate::eval::Evaluation;
use crate::fingerprint::{BLOCK_LEN, FINGERPRINT_LEN, Fingerprint, fingerprints, window_count};
use crate::{CategorySet, Scorer, Scores};

// A feature index is stored in 16 bits.
const _: () = assert!(FINGERPRINT_LEN <= u16::MAX as usize + 1);

/// The longest input that has one window.
pub const WINDOW_LEN: usize = 2 * BLOCK_LEN;

/// One labeled input.
///
/// The labels are the categories that apply to the input. Benign text has
/// no label.
///
/// ```
/// use string2fingerprint::Category;
/// use string2fingerprint::train::Sample;
///
/// let benign = Sample::benign(b"name=O'Brien&city=Cork");
/// let attack = Sample::new(b"name=' OR '1'='1", Category::SqlInjection);
/// # let _ = (benign, attack);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Sample<'a> {
    pub(super) input: &'a [u8],
    pub(super) labels: CategorySet,
    pub(super) weight: f32,
    pub(super) every_window: bool,
}

impl<'a> Sample<'a> {
    /// Creates a sample with the given labels.
    #[must_use]
    pub fn new(input: &'a [u8], labels: impl Into<CategorySet>) -> Self {
        Self {
            input,
            labels: labels.into(),
            weight: 1.0,
            every_window: false,
        }
    }

    /// Creates a sample without any label.
    #[must_use]
    pub fn benign(input: &'a [u8]) -> Self {
        Self::new(input, CategorySet::EMPTY)
    }

    /// Sets how much the sample counts in the training loss, in place of 1.
    ///
    /// A weight of 3 counts like three copies of the sample. Give a higher
    /// weight to the samples that matter most, for example a benign request
    /// that a model wrongly flagged in production, or an attack that it
    /// missed. The weight multiplies the [costs](super::Costs) of the
    /// trainer. It must be finite and not negative.
    #[must_use]
    pub fn weighted(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    /// Applies the labels to every window of an input that is longer than
    /// [`WINDOW_LEN`].
    ///
    /// A model scores one window at a time, and training teaches it the
    /// labels of a sample for each of its windows. A payload usually lies in
    /// one window of a long input, so the other windows would teach the
    /// model to flag the harmless text around it. [`Dataset::push`]
    /// therefore refuses a labeled input with more than one window. Use this
    /// only when every window on its own shows the labels, for example in
    /// text that is malformed throughout. Otherwise cut the sample down to
    /// the payload and its surroundings.
    #[must_use]
    pub fn label_every_window(mut self) -> Self {
        self.every_window = true;
        self
    }

    /// Returns the input bytes.
    #[must_use]
    pub fn input(&self) -> &'a [u8] {
        self.input
    }

    /// Returns the labels.
    #[must_use]
    pub fn labels(&self) -> CategorySet {
        self.labels
    }

    /// Returns the weight.
    #[must_use]
    pub fn weight(&self) -> f32 {
        self.weight
    }
}

/// The reason samples were not added to a [`Dataset`].
///
/// The error never contains sample bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum DatasetError {
    /// A labeled input has more than one window. See
    /// [`Sample::label_every_window`].
    LabeledInputTooLong,
    /// The weight of a sample is negative or not finite.
    InvalidWeight,
    /// A corpus ends inside a record.
    TruncatedCorpus,
    /// A line of labeled text has no tab or names an unknown category. The
    /// value is the line number, from 1.
    InvalidLine(usize),
}

impl fmt::Display for DatasetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LabeledInputTooLong => f.write_str("labeled input is longer than one window"),
            Self::InvalidWeight => f.write_str("sample weight is negative or not finite"),
            Self::TruncatedCorpus => f.write_str("corpus ends inside a record"),
            Self::InvalidLine(line) => {
                write!(f, "line {line} is not labels, a tab, and a payload")
            }
        }
    }
}

impl core::error::Error for DatasetError {}

/// Labeled fingerprints.
///
/// A dataset stores the fingerprint of every window of the samples added to
/// it, not their bytes. An input shorter than
/// [`MIN_INPUT_LEN`](crate::fingerprint::MIN_INPUT_LEN) has no window and is
/// skipped. A stored window takes three bytes for each feature that is not
/// zero, which is at most about 9 KiB and usually far less.
///
/// ```
/// use string2fingerprint::Category;
/// use string2fingerprint::train::{Dataset, Sample};
///
/// let mut data = Dataset::new();
/// data.push(Sample::benign(b"/search?q=rust+programming&page=2"))?;
/// data.push(Sample::new(b"/search?q=1' OR '1'='1", Category::SqlInjection))?;
/// data.extend_from_labeled_lines(b"path_traversal\t../../etc/passwd\nbenign\t/index.html\n")?;
/// assert_eq!(data.samples(), 4);
/// assert_eq!(data.count(Category::PathTraversal), 1);
/// assert_eq!(data.benign(), 2);
/// # Ok::<(), string2fingerprint::train::DatasetError>(())
/// ```
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Dataset {
    /// The features of window `w` are at `starts[w]..starts[w + 1]` of
    /// `indices` and `values`. Holds one more element than there are
    /// windows, unless the dataset was never written to.
    starts: Vec<usize>,
    /// Feature indexes in ascending order within a window.
    indices: Vec<u16>,
    /// Feature values, none of them zero.
    values: Vec<u8>,
    /// The labels of each window.
    labels: Vec<CategorySet>,
    /// The weight of each window, which is finite and not negative.
    weights: Vec<f32>,
    /// The windows of sample `s` are `sample_ends[s - 1]..sample_ends[s]`.
    sample_ends: Vec<usize>,
}

/// The sizes of a dataset, to undo additions.
struct Checkpoint {
    windows: usize,
    features: usize,
    samples: usize,
}

impl Dataset {
    /// Creates an empty dataset.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of samples that have a window.
    #[must_use]
    pub fn samples(&self) -> usize {
        self.sample_ends.len()
    }

    /// Returns the number of windows of all samples. Training sees each
    /// window once in an epoch.
    #[must_use]
    pub fn windows(&self) -> usize {
        self.labels.len()
    }

    /// Returns whether the dataset has no sample.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sample_ends.is_empty()
    }

    /// Returns the number of samples that have every label in `labels`.
    #[must_use]
    pub fn count(&self, labels: impl Into<CategorySet>) -> usize {
        let labels = labels.into();
        self.sample_labels()
            .filter(|&sample| sample & labels == labels)
            .count()
    }

    /// Returns the number of samples without any label.
    #[must_use]
    pub fn benign(&self) -> usize {
        self.sample_labels()
            .filter(|labels| labels.is_empty())
            .count()
    }

    /// Adds a sample and returns the number of windows it has.
    ///
    /// # Errors
    ///
    /// Returns [`DatasetError::LabeledInputTooLong`] for a labeled input
    /// with more than one window, which is one longer than [`WINDOW_LEN`]
    /// bytes, unless the sample chose [`Sample::label_every_window`], and
    /// [`DatasetError::InvalidWeight`] for a weight that is negative or not
    /// finite. The dataset is unchanged then.
    pub fn push(&mut self, sample: Sample<'_>) -> Result<usize, DatasetError> {
        if !(sample.weight.is_finite() && sample.weight >= 0.0) {
            return Err(DatasetError::InvalidWeight);
        }
        let windows = window_count(sample.input.len());
        if windows > 1 && !sample.labels.is_empty() && !sample.every_window {
            return Err(DatasetError::LabeledInputTooLong);
        }
        let mut remaining = fingerprints(sample.input);
        while let Some(fingerprint) = remaining.next_window() {
            self.push_window(fingerprint, sample.labels, sample.weight);
        }
        if windows > 0 {
            self.sample_ends.push(self.labels.len());
        }
        Ok(windows)
    }

    /// Adds a sample of one window that is given as its fingerprint.
    ///
    /// This is for callers that locate the window of a payload in a long
    /// input themselves, with [`fingerprints`].
    pub fn push_fingerprint(&mut self, fingerprint: &Fingerprint, labels: impl Into<CategorySet>) {
        self.push_window(fingerprint, labels.into(), 1.0);
        self.sample_ends.push(self.labels.len());
    }

    /// Adds every sample of `other`.
    pub fn extend_from_dataset(&mut self, other: &Self) {
        for sample in 0..other.samples() {
            for window in other.sample_windows(sample) {
                let (indices, values) = other.window(window);
                self.push_features(indices, values, other.labels[window], other.weights[window]);
            }
            self.sample_ends.push(self.labels.len());
        }
    }

    /// Multiplies the weight of every sample by `factor`.
    ///
    /// To weigh the samples of one source, read them into a dataset of
    /// their own, scale it, and add it with
    /// [`Dataset::extend_from_dataset`].
    ///
    /// # Errors
    ///
    /// Returns [`DatasetError::InvalidWeight`] when `factor` is negative or
    /// not finite, or when a weight would no longer be finite. The dataset
    /// is unchanged then.
    pub fn scale_weights(&mut self, factor: f32) -> Result<(), DatasetError> {
        let valid = factor.is_finite()
            && factor >= 0.0
            && self
                .weights
                .iter()
                .all(|weight| (weight * factor).is_finite());
        if !valid {
            return Err(DatasetError::InvalidWeight);
        }
        for weight in &mut self.weights {
            *weight *= factor;
        }
        Ok(())
    }

    /// Adds each line of `text` as one sample with the same labels, and
    /// returns the number of samples added.
    ///
    /// A line ends with a line feed, or with a carriage return and a line
    /// feed. Lines are raw bytes. A line shorter than two bytes has no
    /// window and adds nothing.
    ///
    /// # Errors
    ///
    /// Returns the error of [`Dataset::push`] for the first line that it
    /// refuses. The dataset is unchanged then.
    pub fn extend_from_lines(
        &mut self,
        text: &[u8],
        labels: impl Into<CategorySet>,
    ) -> Result<usize, DatasetError> {
        let labels = labels.into();
        self.transaction(|data| {
            for line in lines(text) {
                data.push(Sample::new(line, labels))?;
            }
            Ok(())
        })
    }

    /// Adds each line of `text`, which holds labels, a tab, and a payload,
    /// and returns the number of samples added.
    ///
    /// The labels are [category names](crate::Category::name) separated by
    /// commas, or `benign`, as [`CategorySet`] parses them. The payload is
    /// the raw bytes after the first tab. An empty line and a line that
    /// starts with `#` are skipped.
    ///
    /// # Errors
    ///
    /// Returns [`DatasetError::InvalidLine`] with the number of the first
    /// line that has no tab or whose labels do not parse, or the error of
    /// [`Dataset::push`]. The dataset is unchanged then.
    pub fn extend_from_labeled_lines(&mut self, text: &[u8]) -> Result<usize, DatasetError> {
        self.transaction(|data| {
            for (index, line) in lines(text).enumerate() {
                if line.is_empty() || line[0] == b'#' {
                    continue;
                }
                let invalid = DatasetError::InvalidLine(index + 1);
                let tab = line.iter().position(|&byte| byte == b'\t').ok_or(invalid)?;
                let labels: CategorySet = core::str::from_utf8(&line[..tab])
                    .ok()
                    .and_then(|names| names.parse().ok())
                    .ok_or(invalid)?;
                data.push(Sample::new(&line[tab + 1..], labels))?;
            }
            Ok(())
        })
    }

    /// Adds the records of a corpus in the format of the training tools in
    /// `tools/train/`, and returns the number of samples added.
    ///
    /// A record is the payload length as a little-endian 32-bit integer, one
    /// label byte as [`CategorySet::from_bits`] reads it, and the payload.
    ///
    /// # Errors
    ///
    /// Returns [`DatasetError::TruncatedCorpus`] when the corpus ends inside
    /// a record, or the error of [`Dataset::push`]. The dataset is unchanged
    /// then.
    pub fn extend_from_corpus(&mut self, corpus: &[u8]) -> Result<usize, DatasetError> {
        self.transaction(|data| {
            let mut rest = corpus;
            while !rest.is_empty() {
                let (header, tail) = rest
                    .split_first_chunk::<5>()
                    .ok_or(DatasetError::TruncatedCorpus)?;
                let len = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
                let (payload, tail) = usize::try_from(len)
                    .ok()
                    .and_then(|len| tail.split_at_checked(len))
                    .ok_or(DatasetError::TruncatedCorpus)?;
                data.push(Sample::new(payload, CategorySet::from_bits(header[4])))?;
                rest = tail;
            }
            Ok(())
        })
    }

    /// Divides the samples at random into two datasets and returns them:
    /// the first holds the rest, and the second holds about `fraction` of
    /// the samples, from 0.0 to 1.0.
    ///
    /// The division depends only on the number of samples and on `seed`.
    /// All windows of a sample stay together. Train on one part and choose
    /// thresholds on the other, so that the thresholds are not fitted to
    /// text that the model has memorized.
    #[must_use]
    pub fn split(&self, fraction: f64, seed: u64) -> (Self, Self) {
        let mut order: Vec<usize> = (0..self.samples()).collect();
        Rng::new(seed).shuffle(&mut order);
        let fraction = if fraction.is_nan() {
            0.0
        } else {
            fraction.clamp(0.0, 1.0)
        };
        // The product is at most the sample count, so the conversion back
        // is exact after rounding.
        let held_out = (self.samples() as f64 * fraction).round() as usize;
        let (second, first) = order.split_at(held_out.min(order.len()));
        let part = |samples: &[usize]| {
            // Keep the original order within each part.
            let mut samples = samples.to_vec();
            samples.sort_unstable();
            let mut part = Self::new();
            for sample in samples {
                for window in self.sample_windows(sample) {
                    let (indices, values) = self.window(window);
                    part.push_features(indices, values, self.labels[window], self.weights[window]);
                }
                part.sample_ends.push(part.labels.len());
            }
            part
        };
        (part(first), part(second))
    }

    /// Scores every sample with `scorer` and yields its labels and scores,
    /// in the order in which the samples were added.
    ///
    /// The scores of a sample are those that [`Scorer::score`] returns for
    /// its input: the highest score of any of its windows in each category.
    pub fn scores<'a>(
        &'a self,
        scorer: &'a Scorer,
    ) -> impl ExactSizeIterator<Item = (CategorySet, Scores)> + 'a {
        (0..self.samples()).map(move |sample| {
            let mut labels = CategorySet::EMPTY;
            let mut scores = Scores::ZERO;
            for window in self.sample_windows(sample) {
                labels |= self.labels[window];
                scores = scores.max_with(scorer.score_fingerprint(&self.fingerprint(window)));
            }
            (labels, scores)
        })
    }

    /// Scores every sample with `scorer` and returns the distributions of
    /// the scores, from which false positives and false negatives follow
    /// for every threshold.
    #[must_use]
    pub fn evaluate(&self, scorer: &Scorer) -> Evaluation {
        let mut evaluation = Evaluation::new();
        for (labels, scores) in self.scores(scorer) {
            evaluation.record(scores, labels);
        }
        evaluation
    }

    /// Returns the range of windows of `sample`.
    fn sample_windows(&self, sample: usize) -> core::ops::Range<usize> {
        let start = match sample {
            0 => 0,
            _ => self.sample_ends[sample - 1],
        };
        start..self.sample_ends[sample]
    }

    /// Yields the labels of each sample.
    fn sample_labels(&self) -> impl Iterator<Item = CategorySet> + '_ {
        (0..self.samples()).map(|sample| {
            self.sample_windows(sample)
                .fold(CategorySet::EMPTY, |labels, window| {
                    labels | self.labels[window]
                })
        })
    }

    /// Returns the indexes and values of the features of `window` that are
    /// not zero.
    pub(super) fn window(&self, window: usize) -> (&[u16], &[u8]) {
        let range = self.starts[window]..self.starts[window + 1];
        (&self.indices[range.clone()], &self.values[range])
    }

    /// Returns the labels of `window`.
    pub(super) fn window_labels(&self, window: usize) -> CategorySet {
        self.labels[window]
    }

    /// Returns the weight of `window`.
    pub(super) fn window_weight(&self, window: usize) -> f32 {
        self.weights[window]
    }

    /// Returns `window` as a fingerprint.
    pub(super) fn fingerprint(&self, window: usize) -> Fingerprint {
        let mut features = [0u8; FINGERPRINT_LEN];
        let (indices, values) = self.window(window);
        for (&index, &value) in indices.iter().zip(values) {
            features[usize::from(index)] = value;
        }
        Fingerprint::from_array(features)
    }

    fn push_window(&mut self, fingerprint: &Fingerprint, labels: CategorySet, weight: f32) {
        self.open();
        for (index, &value) in fingerprint.as_array().iter().enumerate() {
            if value != 0 {
                // The assertion above bounds the index.
                self.indices.push(index as u16);
                self.values.push(value);
            }
        }
        self.close(labels, weight);
    }

    fn push_features(&mut self, indices: &[u16], values: &[u8], labels: CategorySet, weight: f32) {
        self.open();
        self.indices.extend_from_slice(indices);
        self.values.extend_from_slice(values);
        self.close(labels, weight);
    }

    fn open(&mut self) {
        if self.starts.is_empty() {
            self.starts.push(0);
        }
    }

    fn close(&mut self, labels: CategorySet, weight: f32) {
        self.starts.push(self.indices.len());
        self.labels.push(labels);
        self.weights.push(weight);
    }

    /// Runs `add` and returns the number of samples it added, or undoes its
    /// additions when it fails.
    fn transaction(
        &mut self,
        add: impl FnOnce(&mut Self) -> Result<(), DatasetError>,
    ) -> Result<usize, DatasetError> {
        let checkpoint = Checkpoint {
            windows: self.labels.len(),
            features: self.indices.len(),
            samples: self.sample_ends.len(),
        };
        match add(self) {
            Ok(()) => Ok(self.sample_ends.len() - checkpoint.samples),
            Err(error) => {
                self.labels.truncate(checkpoint.windows);
                self.weights.truncate(checkpoint.windows);
                self.starts.truncate(checkpoint.windows + 1);
                self.indices.truncate(checkpoint.features);
                self.values.truncate(checkpoint.features);
                self.sample_ends.truncate(checkpoint.samples);
                Err(error)
            }
        }
    }
}

/// Yields the lines of `text` without their terminators. Text that ends
/// with a line feed has no empty last line.
fn lines(text: &[u8]) -> impl Iterator<Item = &[u8]> {
    let count = if text.is_empty() { 0 } else { usize::MAX };
    let text = text.strip_suffix(b"\n").unwrap_or(text);
    text.split(|&byte| byte == b'\n')
        .take(count)
        .map(|line| line.strip_suffix(b"\r").unwrap_or(line))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Category;
    use crate::model::tests::random_model;
    use crate::testutil;

    fn corpus(records: &[(&[u8], u8)]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for (payload, label) in records {
            bytes.extend(u32::try_from(payload.len()).unwrap().to_le_bytes());
            bytes.push(*label);
            bytes.extend(*payload);
        }
        bytes
    }

    #[test]
    fn windows_round_trip_as_fingerprints() {
        let mut rng = testutil::Rng::new(1);
        let mut data = Dataset::new();
        let mut expected = Vec::new();
        for len in [2, 3, 40, WINDOW_LEN, WINDOW_LEN + 1, 5 * BLOCK_LEN + 9] {
            let mut input = vec![0u8; len];
            rng.fill(&mut input);
            assert_eq!(data.push(Sample::benign(&input)), Ok(window_count(len)));
            expected.extend(fingerprints(&input));
        }
        assert_eq!(data.samples(), 6);
        assert_eq!(data.windows(), expected.len());
        for (window, fingerprint) in expected.iter().enumerate() {
            assert_eq!(&data.fingerprint(window), fingerprint, "window {window}");
            let (indices, values) = data.window(window);
            assert!(indices.is_sorted());
            assert!(values.iter().all(|&value| value != 0));
            assert_eq!(data.window_labels(window), CategorySet::EMPTY);
        }
    }

    #[test]
    fn inputs_without_a_window_are_skipped() {
        let mut data = Dataset::new();
        assert_eq!(data.push(Sample::benign(b"")), Ok(0));
        assert_eq!(data.push(Sample::new(b"a", Category::Malformed)), Ok(0));
        assert!(data.is_empty());
        assert_eq!((data.samples(), data.windows()), (0, 0));
        assert_eq!(data, Dataset::new());
        assert_eq!(data.push(Sample::benign(b"ab")), Ok(1));
        assert!(!data.is_empty());
    }

    #[test]
    fn long_labeled_inputs_need_an_explicit_choice() {
        let mut data = Dataset::new();
        let long = vec![b'%'; WINDOW_LEN + 1];
        let sample = Sample::new(&long, Category::Malformed);
        assert_eq!(sample.input().len(), WINDOW_LEN + 1);
        assert_eq!(sample.labels(), Category::Malformed.into());
        assert_eq!(data.push(sample), Err(DatasetError::LabeledInputTooLong));
        assert!(data.is_empty());
        assert_eq!(
            data.push(Sample::new(&long[..WINDOW_LEN], Category::Malformed)),
            Ok(1)
        );
        assert_eq!(data.push(sample.label_every_window()), Ok(2));
        assert_eq!(data.push(Sample::benign(&long)), Ok(2));
        assert_eq!(data.samples(), 3);
        assert_eq!(data.windows(), 5);
        assert_eq!(data.count(Category::Malformed), 2);
        assert_eq!(data.benign(), 1);
        for window in 0..3 {
            assert_eq!(data.window_labels(window), Category::Malformed.into());
        }
    }

    #[test]
    fn weights_are_stored_for_each_window_and_validated() {
        let mut data = Dataset::new();
        let long = vec![b'a'; WINDOW_LEN + 1];
        assert_eq!(Sample::benign(b"ab").weight(), 1.0);
        data.push(Sample::benign(b"plain")).unwrap();
        data.push(Sample::benign(&long).weighted(2.5)).unwrap();
        data.push(Sample::new(b"' OR 1=1", Category::SqlInjection).weighted(0.0))
            .unwrap();
        data.push_fingerprint(&Fingerprint::ZERO, CategorySet::EMPTY);
        let weights: Vec<f32> = (0..data.windows())
            .map(|window| data.window_weight(window))
            .collect();
        assert_eq!(weights, [1.0, 2.5, 2.5, 0.0, 1.0]);

        for weight in [-1.0, -0.0001, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let before = data.clone();
            let sample = Sample::benign(b"plain").weighted(weight);
            assert_eq!(
                data.push(sample),
                Err(DatasetError::InvalidWeight),
                "{weight}"
            );
            assert_eq!(data, before);
        }

        let mut scaled = data.clone();
        assert_eq!(scaled.scale_weights(2.0), Ok(()));
        let weights: Vec<f32> = (0..scaled.windows())
            .map(|window| scaled.window_weight(window))
            .collect();
        assert_eq!(weights, [2.0, 5.0, 5.0, 0.0, 2.0]);
        for factor in [-1.0, f32::NAN, f32::INFINITY, f32::MAX] {
            let before = scaled.clone();
            assert_eq!(
                scaled.scale_weights(factor),
                Err(DatasetError::InvalidWeight)
            );
            assert_eq!(scaled, before);
        }
        assert_eq!(Dataset::new().scale_weights(3.0), Ok(()));

        let (rest, held_out) = data.split(0.5, 3);
        let mut joined = rest.clone();
        joined.extend_from_dataset(&held_out);
        let mut joined: Vec<f32> = (0..joined.windows())
            .map(|window| joined.window_weight(window))
            .collect();
        joined.sort_by(f32::total_cmp);
        assert_eq!(joined, [0.0, 1.0, 1.0, 2.5, 2.5]);
    }

    #[test]
    fn counts_follow_the_labels() {
        let mut data = Dataset::new();
        let both = CategorySet::from(Category::SqlInjection) | Category::Obfuscation;
        data.push(Sample::new(b"1%2527 OR 1=1", both)).unwrap();
        data.push(Sample::new(b"' OR 1=1--", Category::SqlInjection))
            .unwrap();
        data.push(Sample::benign(b"hello world")).unwrap();
        data.push_fingerprint(&Fingerprint::ZERO, Category::Malformed);
        assert_eq!(data.samples(), 4);
        assert_eq!(data.count(Category::SqlInjection), 2);
        assert_eq!(data.count(Category::Obfuscation), 1);
        assert_eq!(data.count(both), 1);
        assert_eq!(data.count(Category::Malformed), 1);
        assert_eq!(data.count(Category::HeaderInjection), 0);
        assert_eq!(data.count(CategorySet::EMPTY), 4);
        assert_eq!(data.benign(), 1);
        assert_eq!(data.window(3), (&[][..], &[][..]));
    }

    #[test]
    fn lines_are_split_without_terminators() {
        let collect = |text: &'static [u8]| lines(text).collect::<Vec<_>>();
        assert!(collect(b"").is_empty());
        assert_eq!(collect(b"\n"), [b""]);
        assert_eq!(collect(b"ab"), [b"ab"]);
        assert_eq!(collect(b"ab\n"), [b"ab"]);
        assert_eq!(collect(b"ab\r\ncd\n\nef"), [&b"ab"[..], b"cd", b"", b"ef"]);
        assert_eq!(collect(b"a\rb\r\n"), [b"a\rb"]);

        let mut data = Dataset::new();
        assert_eq!(
            data.extend_from_lines(b"first line\r\n\nx\nsecond\n", CategorySet::EMPTY),
            Ok(2)
        );
        let mut expected = Dataset::new();
        expected.push(Sample::benign(b"first line")).unwrap();
        expected.push(Sample::benign(b"second")).unwrap();
        assert_eq!(data, expected);
    }

    #[test]
    fn labeled_lines_are_parsed_and_errors_name_the_line() {
        let mut data = Dataset::new();
        let text =
            b"# comment\nsql_injection,obfuscation\t1%2527 OR 1=1\n\nbenign\ta\tb c\n\tplain\n";
        assert_eq!(data.extend_from_labeled_lines(text), Ok(3));
        let mut expected = Dataset::new();
        let both = CategorySet::from(Category::SqlInjection) | Category::Obfuscation;
        expected.push(Sample::new(b"1%2527 OR 1=1", both)).unwrap();
        expected.push(Sample::benign(b"a\tb c")).unwrap();
        expected.push(Sample::benign(b"plain")).unwrap();
        assert_eq!(data, expected);

        for (text, line) in [
            (&b"benign\tok\nno tab here\n"[..], 2),
            (b"sql\tpayload\n", 1),
            (b"benign\tok\n\nmalformed,\tpayload", 3),
            (b"\xFF\tpayload", 1),
        ] {
            let before = data.clone();
            assert_eq!(
                data.extend_from_labeled_lines(text),
                Err(DatasetError::InvalidLine(line))
            );
            assert_eq!(data, before);
        }
        let mut long = b"benign\tok\nmalformed\t".to_vec();
        long.extend(vec![b'%'; WINDOW_LEN + 1]);
        let before = data.clone();
        assert_eq!(
            data.extend_from_labeled_lines(&long),
            Err(DatasetError::LabeledInputTooLong)
        );
        assert_eq!(data, before);
    }

    #[test]
    fn corpus_records_are_read_and_truncation_is_rejected() {
        let records: [(&[u8], u8); 4] = [
            (b"' OR 1=1--", 0b0000_0100),
            (b"", 0),
            (b"plain text", 0),
            (b"..%252f..%252fetc/passwd", 0b0001_0010),
        ];
        let bytes = corpus(&records);
        let mut data = Dataset::new();
        assert_eq!(data.extend_from_corpus(&bytes), Ok(3));
        assert_eq!(data.count(Category::SqlInjection), 1);
        assert_eq!(
            data.count(CategorySet::from(Category::PathTraversal) | Category::Obfuscation),
            1
        );
        assert_eq!(data.benign(), 1);
        assert_eq!(data.extend_from_corpus(b""), Ok(0));

        for len in 1..bytes.len() {
            let before = data.clone();
            let result = data.extend_from_corpus(&bytes[..len]);
            let boundaries = [15, 20, 35];
            if boundaries.contains(&len) {
                assert!(result.is_ok(), "length {len}");
            } else {
                assert_eq!(result, Err(DatasetError::TruncatedCorpus), "length {len}");
                assert_eq!(data, before, "length {len}");
            }
        }
        // A length that the corpus cannot hold.
        assert_eq!(
            data.extend_from_corpus(&[0xFF, 0xFF, 0xFF, 0xFF, 0, b'a']),
            Err(DatasetError::TruncatedCorpus)
        );
        let long = vec![b'a'; WINDOW_LEN + 1];
        let before = data.clone();
        assert_eq!(
            data.extend_from_corpus(&corpus(&[(b"ok", 0), (&long, 1)])),
            Err(DatasetError::LabeledInputTooLong)
        );
        assert_eq!(data, before);
    }

    #[test]
    fn split_keeps_samples_whole_and_is_reproducible() {
        let mut rng = testutil::Rng::new(2);
        let mut data = Dataset::new();
        for index in 0..200 {
            let mut input = vec![0u8; 2 + (index % 7) * 150];
            rng.fill(&mut input);
            data.push(Sample::benign(&input)).unwrap();
        }
        let (rest, held_out) = data.split(0.25, 9);
        assert_eq!(held_out.samples(), 50);
        assert_eq!(rest.samples(), 150);
        assert_eq!(rest.windows() + held_out.windows(), data.windows());
        assert_eq!(data.split(0.25, 9), (rest.clone(), held_out.clone()));
        assert_ne!(data.split(0.25, 10).1, held_out);

        // Each window of the original is in exactly one part.
        let windows = |part: &Dataset| -> Vec<Fingerprint> {
            (0..part.windows())
                .map(|window| part.fingerprint(window))
                .collect()
        };
        let mut parts = windows(&rest);
        parts.extend(windows(&held_out));
        let mut all = windows(&data);
        let key = |fingerprint: &Fingerprint| fingerprint.as_array().to_vec();
        parts.sort_by_key(key);
        all.sort_by_key(key);
        assert_eq!(parts, all);

        assert_eq!(data.split(0.0, 1), (data.clone(), Dataset::new()));
        assert_eq!(data.split(1.0, 1), (Dataset::new(), data.clone()));
        assert_eq!(data.split(f64::NAN, 1).1, Dataset::new());
        assert_eq!(data.split(-3.0, 1).1, Dataset::new());
        assert_eq!(data.split(f64::INFINITY, 1).1, data);
        assert_eq!(
            Dataset::new().split(0.5, 1),
            (Dataset::new(), Dataset::new())
        );

        let mut joined = rest.clone();
        joined.extend_from_dataset(&held_out);
        assert_eq!(joined.samples(), data.samples());
        assert_eq!(joined.windows(), data.windows());
    }

    #[test]
    fn scores_equal_those_of_the_inputs() {
        let scorer = Scorer::with_model(random_model(&mut testutil::Rng::new(3), &[16]));
        let mut rng = testutil::Rng::new(4);
        let mut data = Dataset::new();
        let mut expected = Vec::new();
        for (index, len) in [2, 90, WINDOW_LEN, 3 * BLOCK_LEN + 1, 6 * BLOCK_LEN]
            .into_iter()
            .enumerate()
        {
            let mut input = vec![0u8; len];
            rng.fill(&mut input);
            let labels = CategorySet::from_bits(u8::try_from(index).unwrap());
            data.push(Sample::new(&input, labels).label_every_window())
                .unwrap();
            expected.push((labels, scorer.score(&input)));
        }
        assert_eq!(data.scores(&scorer).len(), expected.len());
        assert_eq!(data.scores(&scorer).collect::<Vec<_>>(), expected);

        let mut evaluation = Evaluation::new();
        for &(labels, scores) in &expected {
            evaluation.record(scores, labels);
        }
        assert_eq!(data.evaluate(&scorer), evaluation);
        assert_eq!(evaluation.samples(), 5);
        assert_eq!(Dataset::new().evaluate(&scorer), Evaluation::new());
    }

    #[test]
    fn errors_have_fixed_messages() {
        assert_eq!(
            DatasetError::LabeledInputTooLong.to_string(),
            "labeled input is longer than one window"
        );
        assert_eq!(
            DatasetError::InvalidWeight.to_string(),
            "sample weight is negative or not finite"
        );
        assert_eq!(
            DatasetError::TruncatedCorpus.to_string(),
            "corpus ends inside a record"
        );
        assert_eq!(
            DatasetError::InvalidLine(7).to_string(),
            "line 7 is not labels, a tab, and a payload"
        );
    }
}
