//! One-call decisions: a scorer with a threshold for each category.

use core::fmt;
use std::io;

use crate::kernel::Kernel;
use crate::model::{EmbeddedModel, Model};
use crate::{
    Backend, Category, CategorySet, ScoreStream, Scorer, ScorerBuilder, ScorerError, Scores,
    Threshold, Thresholds,
};

/// A ready-made pair of an embedded model and thresholds.
///
/// The rates below were measured on the synthetic test split of the
/// training tools, 59,991 samples of which 21,446 are benign: the share of
/// attack samples and of benign samples flagged in any category. The data
/// is synthetic, so measure on your own traffic before acting on a
/// decision.
///
/// | Preset | Model | Threshold | Attacks flagged | Benign flagged |
/// | --- | --- | --- | --- | --- |
/// | [`Fast`](Self::Fast) | `compact` | 128 | 97.2% | 1.24% |
/// | [`Balanced`](Self::Balanced) | `default` | 128 | 98.1% | 1.19% |
/// | [`Thorough`](Self::Thorough) | `large` | 128 | 98.5% | 1.04% |
/// | [`Blocking`](Self::Blocking) | `strict` | 192 | 96.7% | 0.29% |
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Preset {
    /// The compact model at the score 128, which is a logit of 0. The
    /// fastest, at a few more missed attacks.
    Fast,
    /// The default model at the score 128. [`Detector::new`] uses it.
    Balanced,
    /// The large model at the score 128. The best separation, at about
    /// three times the cost of [`Preset::Balanced`].
    Thorough,
    /// The strict model at the score 192. A quarter of the benign flags of
    /// [`Preset::Balanced`], for hosts that block on a decision.
    Blocking,
}

impl Preset {
    /// Every preset, from the fastest to the strictest.
    pub const ALL: [Self; 4] = [Self::Fast, Self::Balanced, Self::Thorough, Self::Blocking];

    /// Returns the embedded model of the preset.
    #[must_use]
    pub const fn model(self) -> EmbeddedModel {
        match self {
            Self::Fast => EmbeddedModel::Compact,
            Self::Balanced => EmbeddedModel::Default,
            Self::Thorough => EmbeddedModel::Large,
            Self::Blocking => EmbeddedModel::Strict,
        }
    }

    /// Returns the thresholds of the preset, which are the same for every
    /// category.
    #[must_use]
    pub const fn thresholds(self) -> Thresholds {
        Thresholds::uniform(Threshold::at(match self {
            Self::Fast | Self::Balanced | Self::Thorough => 128,
            Self::Blocking => 192,
        }))
    }

    /// Returns a stable lowercase name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Fast => "fast",
            Self::Balanced => "balanced",
            Self::Thorough => "thorough",
            Self::Blocking => "blocking",
        }
    }

    /// Returns the preset whose [`name`](Self::name) is `name`.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|preset| preset.name() == name)
    }
}

impl fmt::Display for Preset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// The scores of an input and the categories they flag.
///
/// `Display` writes the flagged categories, separated by commas, or
/// `benign` when none is flagged.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Detection {
    scores: Scores,
    flagged: CategorySet,
}

impl Detection {
    /// The detection of an input without evidence, such as the empty input,
    /// under thresholds that flag no score of 0.
    pub const NONE: Self = Self {
        scores: Scores::ZERO,
        flagged: CategorySet::EMPTY,
    };

    /// Decides `scores` with `thresholds`.
    #[must_use]
    pub fn new(scores: Scores, thresholds: &Thresholds) -> Self {
        Self {
            scores,
            flagged: scores.flagged(thresholds),
        }
    }

    /// Returns the scores.
    #[must_use]
    pub const fn scores(&self) -> Scores {
        self.scores
    }

    /// Returns the flagged categories.
    #[must_use]
    pub const fn flagged(&self) -> CategorySet {
        self.flagged
    }

    /// Returns whether any category is flagged.
    #[must_use]
    pub const fn is_flagged(&self) -> bool {
        !self.flagged.is_empty()
    }

    /// Returns whether `category` is flagged.
    #[must_use]
    pub const fn contains(&self, category: Category) -> bool {
        self.flagged.contains(category)
    }

    /// Returns the flagged category with the highest score and its score,
    /// or `None` when no category is flagged.
    ///
    /// When several flagged categories share the highest score, the one
    /// with the lowest discriminant is returned.
    #[must_use]
    pub fn top(&self) -> Option<(Category, u8)> {
        self.iter()
            .fold(None, |best, (category, score)| match best {
                Some((_, best_score)) if best_score >= score => best,
                _ => Some((category, score)),
            })
    }

    /// Iterates over every flagged category and its score, ordered by
    /// discriminant.
    pub fn iter(&self) -> impl Iterator<Item = (Category, u8)> + '_ {
        self.flagged
            .iter()
            .map(|category| (category, self.scores.get(category)))
    }

    /// Combines two detections under the same thresholds: the higher score
    /// in each category, and every category that either flags.
    ///
    /// A threshold flags every score above one it flags, so the result is
    /// the detection of the combined scores.
    #[must_use]
    pub fn max_with(self, other: Self) -> Self {
        Self {
            scores: self.scores.max_with(other.scores),
            flagged: self.flagged | other.flagged,
        }
    }
}

impl fmt::Display for Detection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.flagged, f)
    }
}

/// Decides whether untrusted bytes are flagged: a [`Scorer`] with a
/// [`Threshold`] for each category.
///
/// A detector is immutable. Create one and share it between threads by
/// reference. It has the input contract of the scorer: any bytes of any
/// length are accepted, and a decision never fails, never panics, and
/// allocates no heap memory.
///
/// ```
/// use string2fingerprint::{Category, Detector};
///
/// let detector = Detector::new();
/// let detection = detector.detect(b"/files?name=..%2f..%2f..%2fetc%2fpasswd");
/// if detection.is_flagged() {
///     // Log or block the request, naming what was flagged.
///     println!("flagged: {detection}");
/// }
/// assert!(!detector.detect(b"/files?name=report-2026.pdf").is_flagged());
/// ```
#[derive(Clone, Debug)]
pub struct Detector {
    scorer: Scorer,
    thresholds: Thresholds,
}

impl Default for Detector {
    fn default() -> Self {
        Self::new()
    }
}

impl AsRef<Self> for Detector {
    fn as_ref(&self) -> &Self {
        self
    }
}

impl Detector {
    /// Creates the detector of [`Preset::Balanced`] on the preferred
    /// backend.
    ///
    /// Each call parses the model and allocates about 200 KiB. Create one
    /// detector and share it.
    #[must_use]
    pub fn new() -> Self {
        Self::with_preset(Preset::Balanced)
    }

    /// Creates the detector of `preset` on the preferred backend.
    #[must_use]
    pub fn with_preset(preset: Preset) -> Self {
        Self::from_parts(Scorer::with_embedded(preset.model()), preset.thresholds())
    }

    /// Creates a detector from a scorer and thresholds, such as a model and
    /// thresholds that a build script trained and calibrated.
    #[must_use]
    pub const fn from_parts(scorer: Scorer, thresholds: Thresholds) -> Self {
        Self { scorer, thresholds }
    }

    /// Returns a builder that starts from [`Preset::Balanced`].
    ///
    /// ```
    /// use string2fingerprint::detect::Preset;
    /// use string2fingerprint::{Category, Detector, Threshold};
    ///
    /// let detector = Detector::builder()
    ///     .preset(Preset::Blocking)
    ///     .category(Category::CommandInjection, Threshold::at(230))
    ///     .disable(Category::Malformed)
    ///     .build()?;
    /// assert_eq!(detector.thresholds().get(Category::Malformed), Threshold::NEVER);
    /// # Ok::<(), string2fingerprint::ScorerError>(())
    /// ```
    pub fn builder() -> DetectorBuilder {
        DetectorBuilder::new()
    }

    /// Returns the scorer.
    #[must_use]
    pub fn scorer(&self) -> &Scorer {
        &self.scorer
    }

    /// Returns the thresholds.
    #[must_use]
    pub const fn thresholds(&self) -> Thresholds {
        self.thresholds
    }

    /// Replaces the thresholds.
    #[must_use]
    pub const fn with_thresholds(mut self, thresholds: Thresholds) -> Self {
        self.thresholds = thresholds;
        self
    }

    /// Returns the scorer and the thresholds.
    #[must_use]
    pub fn into_parts(self) -> (Scorer, Thresholds) {
        (self.scorer, self.thresholds)
    }

    /// Decides scores that the scorer of this detector computed.
    #[must_use]
    pub fn decide(&self, scores: Scores) -> Detection {
        Detection::new(scores, &self.thresholds)
    }

    /// Scores `input` and decides the scores.
    #[must_use]
    pub fn detect(&self, input: &[u8]) -> Detection {
        self.decide(self.scorer.score(input))
    }

    /// Returns whether any category of `input` is flagged.
    #[must_use]
    pub fn is_flagged(&self, input: &[u8]) -> bool {
        self.detect(input).is_flagged()
    }

    /// Scores the fields of one message apart and decides their combined
    /// scores. See [`Scorer::score_fields`].
    #[must_use]
    pub fn detect_fields<I>(&self, fields: I) -> Detection
    where
        I: IntoIterator,
        I::Item: AsRef<[u8]>,
    {
        self.decide(self.scorer.score_fields(fields))
    }

    /// Decides each input and returns the detections in input order.
    #[must_use]
    pub fn detect_batch<I>(&self, inputs: I) -> Vec<Detection>
    where
        I: IntoIterator,
        I::Item: AsRef<[u8]>,
    {
        self.scorer
            .score_each(inputs)
            .map(|scores| self.decide(scores))
            .collect()
    }

    /// Decides each input on up to `threads` threads. See
    /// [`Scorer::score_parallel`].
    ///
    /// # Panics
    ///
    /// When the operating system cannot start a thread.
    #[must_use]
    pub fn detect_parallel<T>(&self, inputs: &[T], threads: usize) -> Vec<Detection>
    where
        T: AsRef<[u8]> + Sync,
    {
        self.scorer
            .score_parallel(inputs, threads)
            .into_iter()
            .map(|scores| self.decide(scores))
            .collect()
    }

    /// Reads `reader` to its end and decides its scores. See
    /// [`Scorer::score_reader`].
    ///
    /// # Errors
    ///
    /// Returns the first error of `reader` other than an interrupted read.
    pub fn detect_reader<R: io::Read>(&self, reader: R) -> io::Result<Detection> {
        self.scorer
            .score_reader(reader)
            .map(|scores| self.decide(scores))
    }

    /// Returns a stream that decides an input that arrives in parts with
    /// this detector. See [`DetectionStream`].
    #[must_use]
    pub fn stream(&self) -> DetectionStream<&Self> {
        DetectionStream::new(self)
    }
}

/// Lends the scorer of a detector to a [`ScoreStream`].
#[derive(Clone)]
struct ScorerOf<D>(D);

impl<D: AsRef<Detector>> AsRef<Scorer> for ScorerOf<D> {
    fn as_ref(&self) -> &Scorer {
        self.0.as_ref().scorer()
    }
}

/// Decides an input that arrives in parts, on top of a [`ScoreStream`].
///
/// [`detection`](Self::detection) decides the windows completed so far. A
/// category that it flags stays flagged, so a caller can stop reading as
/// soon as it is flagged. [`finish`](Self::finish) decides the whole input
/// as [`Detector::detect`] would. The stream borrows its detector through
/// `D`, such as `&Detector` or `Arc<Detector>`, and implements
/// [`io::Write`].
///
/// ```
/// use string2fingerprint::Detector;
///
/// let detector = Detector::new();
/// let mut stream = detector.stream();
/// for part in [&b"cmd=ping%20-c%201%20"[..], b"127.0.0.1%3Bcat%20/etc/passwd"] {
///     stream.update(part);
///     if stream.detection().is_flagged() {
///         break;
///     }
/// }
/// assert!(stream.finish().is_flagged());
/// ```
#[derive(Clone)]
pub struct DetectionStream<D: AsRef<Detector>> {
    stream: ScoreStream<ScorerOf<D>>,
}

/// Shows the progress and nothing derived from the input.
impl<D: AsRef<Detector>> fmt::Debug for DetectionStream<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DetectionStream")
            .field("len", &self.stream.len())
            .finish_non_exhaustive()
    }
}

impl<D: AsRef<Detector>> DetectionStream<D> {
    /// Creates a stream at the start of an input.
    pub fn new(detector: D) -> Self {
        Self {
            stream: ScoreStream::new(ScorerOf(detector)),
        }
    }

    /// Returns the detector.
    #[must_use]
    pub fn detector(&self) -> &Detector {
        self.stream.source().0.as_ref()
    }

    /// Consumes the next part of the input. See [`ScoreStream::update`].
    pub fn update(&mut self, part: &[u8]) {
        self.stream.update(part);
    }

    /// Decides the windows completed so far.
    #[must_use]
    pub fn detection(&self) -> Detection {
        self.detector().decide(self.stream.scores())
    }

    /// Ends the input, decides it, and starts a new input.
    pub fn finish(&mut self) -> Detection {
        let scores = self.stream.finish();
        self.detector().decide(scores)
    }

    /// Discards the input so far and starts a new input.
    pub fn reset(&mut self) {
        self.stream.reset();
    }

    /// Returns the number of bytes consumed since the input started.
    #[must_use]
    pub fn len(&self) -> u64 {
        self.stream.len()
    }

    /// Returns whether no byte was consumed since the input started.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.stream.is_empty()
    }

    /// Returns the detector and drops the input so far.
    pub fn into_detector(self) -> D {
        self.stream.into_scorer().0
    }
}

/// Every write consumes the whole buffer, and flushing does nothing.
impl<D: AsRef<Detector>> io::Write for DetectionStream<D> {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        self.update(buffer);
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Chooses the model, the backend, and the thresholds of a [`Detector`].
///
/// [`Detector::builder`] returns one, which starts from
/// [`Preset::Balanced`]. A later choice replaces an earlier one, and
/// [`preset`](Self::preset) replaces both the model and every threshold.
/// The model and backend choices are those of [`ScorerBuilder`].
#[derive(Clone, Debug)]
#[must_use]
pub struct DetectorBuilder {
    scorer: ScorerBuilder,
    thresholds: Thresholds,
}

impl Default for DetectorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl DetectorBuilder {
    /// Returns a builder for [`Preset::Balanced`] on the detected backend.
    pub fn new() -> Self {
        Self {
            scorer: ScorerBuilder::new().embedded(Preset::Balanced.model()),
            thresholds: Preset::Balanced.thresholds(),
        }
    }

    /// Uses the model and the thresholds of `preset`.
    pub fn preset(mut self, preset: Preset) -> Self {
        self.scorer = self.scorer.embedded(preset.model());
        self.thresholds = preset.thresholds();
        self
    }

    /// Loads an embedded model and keeps the thresholds.
    pub fn embedded(mut self, model: EmbeddedModel) -> Self {
        self.scorer = self.scorer.embedded(model);
        self
    }

    /// Runs `model` and keeps the thresholds.
    pub fn model(mut self, model: Model) -> Self {
        self.scorer = self.scorer.model(model);
        self
    }

    /// Parses `bytes` as the model. See [`ScorerBuilder::model_bytes`].
    pub fn model_bytes(mut self, bytes: &[u8]) -> Self {
        self.scorer = self.scorer.model_bytes(bytes);
        self
    }

    /// Selects `backend`. See [`ScorerBuilder::backend`].
    pub fn backend(mut self, backend: Backend) -> Self {
        self.scorer = self.scorer.backend(backend);
        self
    }

    /// Selects the backend of `kernel`. See [`ScorerBuilder::kernel`].
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.scorer = self.scorer.kernel(kernel);
        self
    }

    /// Replaces every threshold.
    pub fn thresholds(mut self, thresholds: Thresholds) -> Self {
        self.thresholds = thresholds;
        self
    }

    /// Sets the same threshold for every category.
    pub fn threshold(self, threshold: Threshold) -> Self {
        self.thresholds(Thresholds::uniform(threshold))
    }

    /// Sets the threshold of `category`.
    pub fn category(mut self, category: Category, threshold: Threshold) -> Self {
        self.thresholds = self.thresholds.with(category, threshold);
        self
    }

    /// Turns `category` off: it is never flagged.
    pub fn disable(self, category: Category) -> Self {
        self.category(category, Threshold::NEVER)
    }

    /// Turns off every category outside `categories` and keeps the
    /// thresholds of those in it.
    pub fn only(mut self, categories: impl Into<CategorySet>) -> Self {
        let keep = categories.into();
        for category in Category::ALL {
            if !keep.contains(category) {
                self.thresholds = self.thresholds.with(category, Threshold::NEVER);
            }
        }
        self
    }

    /// Creates the detector.
    ///
    /// # Errors
    ///
    /// Returns the [`ScorerError`] of [`ScorerBuilder::build`].
    pub fn build(self) -> Result<Detector, ScorerError> {
        Ok(Detector::from_parts(self.scorer.build()?, self.thresholds))
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write as _;
    use std::sync::Arc;

    use super::*;
    use crate::model::tests::random_model;
    use crate::testutil::Rng;

    fn detector(seed: u64) -> Detector {
        let scorer = Scorer::with_model(random_model(&mut Rng::new(seed), &[24]));
        Detector::from_parts(scorer, Thresholds::uniform(Threshold::at(140)))
    }

    fn inputs(seed: u64) -> Vec<Vec<u8>> {
        let mut rng = Rng::new(seed);
        [0, 1, 5, 64, 300, 1100, 17]
            .into_iter()
            .map(|len| {
                let mut input = vec![0u8; len];
                rng.fill(&mut input);
                input
            })
            .collect()
    }

    #[test]
    fn presets_name_their_models_and_thresholds() {
        for preset in Preset::ALL {
            assert_eq!(Preset::from_name(preset.name()), Some(preset));
            assert_eq!(preset.to_string(), preset.name());
            let detector = Detector::with_preset(preset);
            assert_eq!(detector.scorer().model(), &preset.model().load());
            assert_eq!(detector.thresholds(), preset.thresholds());
        }
        assert_eq!(Preset::from_name("Balanced"), None);
        let detector = Detector::default();
        assert_eq!(detector.thresholds(), Preset::Balanced.thresholds());
        assert_eq!(detector.scorer().model(), &Model::embedded());
        assert_eq!(
            Preset::Blocking.thresholds().get(Category::SqlInjection),
            Threshold::at(192)
        );
    }

    #[test]
    fn detections_follow_the_thresholds() {
        let scores = Scores::from_array([10, 200, 150, 0, 200, 0, 255, 99]);
        let thresholds = Thresholds::uniform(Threshold::at(150))
            .with(Category::TemplateInjection, Threshold::NEVER);
        let detection = Detection::new(scores, &thresholds);
        assert_eq!(detection.scores(), scores);
        assert_eq!(
            detection.flagged(),
            CategorySet::from(Category::Obfuscation)
                | Category::SqlInjection
                | Category::PathTraversal
        );
        assert!(detection.is_flagged());
        assert!(detection.contains(Category::PathTraversal));
        assert!(!detection.contains(Category::TemplateInjection));
        assert_eq!(detection.top(), Some((Category::Obfuscation, 200)));
        assert_eq!(detection.iter().count(), 3);
        assert_eq!(
            detection.to_string(),
            "obfuscation,sql_injection,path_traversal"
        );
        assert_eq!(Detection::NONE.to_string(), "benign");
        assert_eq!(Detection::NONE.top(), None);
        assert_eq!(Detection::default(), Detection::NONE);
        let other = Detection::new(Scores::from_array([255, 0, 0, 0, 0, 0, 0, 0]), &thresholds);
        assert_eq!(
            detection.max_with(other),
            Detection::new(scores.max_with(other.scores()), &thresholds)
        );
    }

    #[test]
    fn every_path_decides_like_detect() {
        let detector = detector(61);
        let inputs = inputs(62);
        let expected: Vec<Detection> = inputs.iter().map(|input| detector.detect(input)).collect();
        for (input, detection) in inputs.iter().zip(&expected) {
            assert_eq!(*detection, detector.decide(detector.scorer().score(input)));
            assert_eq!(detector.is_flagged(input), detection.is_flagged());
            assert_eq!(detector.detect_reader(&input[..]).unwrap(), *detection);
            let mut stream = detector.stream();
            for part in input.chunks(100) {
                stream.update(part);
                let so_far = stream.detection();
                assert_eq!(so_far.flagged() - detection.flagged(), CategorySet::EMPTY);
            }
            assert_eq!(stream.finish(), *detection);
        }
        assert_eq!(detector.detect_batch(&inputs), expected);
        assert_eq!(detector.detect_parallel(&inputs, 3), expected);
        let combined = expected
            .iter()
            .fold(Detection::NONE, |all, &one| all.max_with(one));
        assert_eq!(detector.detect_fields(&inputs), combined);
    }

    #[test]
    fn streams_share_their_detector() {
        let detector = Arc::new(detector(63));
        let input = vec![b'\''; 700];
        let mut stream = DetectionStream::new(Arc::clone(&detector));
        stream.write_all(&input).unwrap();
        stream.flush().unwrap();
        assert_eq!(stream.len(), 700);
        assert_eq!(format!("{stream:?}"), "DetectionStream { len: 700, .. }");
        assert_eq!(stream.finish(), detector.detect(&input));
        assert!(stream.is_empty());
        stream.update(&input);
        stream.reset();
        assert_eq!(stream.finish(), detector.detect(b""));
        assert_eq!(stream.detector().thresholds(), detector.thresholds());
        assert!(Arc::ptr_eq(&stream.into_detector(), &detector));
    }

    #[test]
    fn builder_sets_models_backends_and_thresholds() {
        let built = Detector::builder()
            .preset(Preset::Fast)
            .backend(Backend::Scalar)
            .threshold(Threshold::at(100))
            .category(Category::HeaderInjection, Threshold::at(250))
            .only(CategorySet::from(Category::HeaderInjection) | Category::SqlInjection)
            .build()
            .unwrap();
        assert_eq!(built.scorer().model(), &EmbeddedModel::Compact.load());
        assert_eq!(built.scorer().backend(), Backend::Scalar);
        let expected = Thresholds::uniform(Threshold::NEVER)
            .with(Category::HeaderInjection, Threshold::at(250))
            .with(Category::SqlInjection, Threshold::at(100));
        assert_eq!(built.thresholds(), expected);

        let model = random_model(&mut Rng::new(64), &[8]);
        let built = DetectorBuilder::default()
            .embedded(EmbeddedModel::Large)
            .model_bytes(&model.to_bytes())
            .kernel(Kernel::SCALAR)
            .disable(Category::Malformed)
            .build()
            .unwrap();
        assert_eq!(built.scorer().model(), &model);
        assert_eq!(
            built.thresholds(),
            Preset::Balanced
                .thresholds()
                .with(Category::Malformed, Threshold::NEVER)
        );
        let built = Detector::builder()
            .model(model.clone())
            .thresholds(expected)
            .build()
            .unwrap();
        assert_eq!(
            (built.scorer().model(), built.thresholds()),
            (&model, expected)
        );
        let (scorer, thresholds) = built
            .with_thresholds(Thresholds::uniform(Threshold::ALWAYS))
            .into_parts();
        assert_eq!(scorer.model(), &model);
        assert_eq!(thresholds, Thresholds::uniform(Threshold::ALWAYS));
        assert!(Detector::builder().model_bytes(b"").build().is_err());
    }
}
