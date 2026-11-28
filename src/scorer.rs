//! The scoring entry point.

use core::fmt;

use crate::backend::{UnsupportedBackend, Verified};
use crate::fingerprint::{Fingerprint, fingerprints};
use crate::kernel::{Kernel, selftest};
use crate::model::{EmbeddedModel, Logits, Model, ModelError, Trace};
use crate::{Backend, Scores};

/// Scores untrusted bytes with a [`Model`].
///
/// A scorer is immutable. Share one between threads by reference.
///
/// ```
/// use string2fingerprint::{Category, Scorer};
///
/// let scorer = Scorer::new();
/// let scores = scorer.score(b"/search?q=1%27%20OR%20%271%27=%271");
/// if scores.get(Category::SqlInjection) >= 192 {
///     // Act on the request, for example log or block it.
/// }
/// ```
#[derive(Clone, Debug)]
pub struct Scorer {
    model: Model,
    backend: Verified,
}

impl Default for Scorer {
    fn default() -> Self {
        Self::new()
    }
}

impl From<Model> for Scorer {
    /// Returns [`Scorer::with_model`].
    fn from(model: Model) -> Self {
        Self::with_model(model)
    }
}

impl From<EmbeddedModel> for Scorer {
    /// Returns [`Scorer::with_embedded`].
    fn from(model: EmbeddedModel) -> Self {
        Self::with_embedded(model)
    }
}

impl Scorer {
    /// Returns a builder that chooses the model and the backend, and
    /// reports a model that does not parse or a backend that the CPU lacks
    /// as one [`ScorerError`].
    ///
    /// ```
    /// use string2fingerprint::model::EmbeddedModel;
    /// use string2fingerprint::{Backend, Scorer};
    ///
    /// let scorer = Scorer::builder()
    ///     .embedded(EmbeddedModel::Compact)
    ///     .backend(Backend::Scalar)
    ///     .build()?;
    /// assert_eq!(scorer.backend(), Backend::Scalar);
    /// # Ok::<(), string2fingerprint::ScorerError>(())
    /// ```
    pub fn builder() -> ScorerBuilder {
        ScorerBuilder::new()
    }

    /// Parses a model in the format of [`crate::model`] and creates a
    /// scorer that runs it on the preferred backend.
    ///
    /// # Errors
    ///
    /// Returns the [`ModelError`] of [`Model::from_bytes`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ModelError> {
        Model::from_bytes(bytes).map(Self::with_model)
    }

    /// Creates a scorer that runs the embedded default model on the preferred
    /// backend that the running CPU supports. See [`Model::embedded`].
    ///
    /// Each call parses the model and allocates about 200 KiB. Create one
    /// scorer and share it instead of creating one for each input.
    #[must_use]
    pub fn new() -> Self {
        Self::with_model(Model::embedded())
    }

    /// Creates a scorer that runs an embedded model on the preferred backend
    /// that the running CPU supports. See [`EmbeddedModel`].
    #[must_use]
    pub fn with_embedded(model: EmbeddedModel) -> Self {
        Self::with_model(model.load())
    }

    /// Creates a scorer that runs `model` on the preferred backend that the
    /// running CPU supports. See [`Backend::detect`].
    #[must_use]
    pub fn with_model(model: Model) -> Self {
        Self {
            model,
            backend: Verified::detect(),
        }
    }

    /// Selects a specific backend.
    ///
    /// Scores do not depend on the backend. This is meant for tests and
    /// benchmarks, and for hosts that must avoid a particular instruction
    /// set.
    ///
    /// # Errors
    ///
    /// Returns [`UnsupportedBackend`] when the running CPU and operating
    /// system are not detected to support `backend`.
    pub fn with_backend(mut self, backend: Backend) -> Result<Self, UnsupportedBackend> {
        self.backend = Verified::new(backend)?;
        Ok(self)
    }

    /// Selects a specific backend without detecting its CPU features.
    ///
    /// This exists for backends that the operating system cannot report yet,
    /// so that [`Backend::is_supported`] is false for them on every CPU.
    /// Prefer [`Scorer::with_backend`] for every other backend.
    ///
    /// # Safety
    ///
    /// The running CPU must implement every CPU feature that the
    /// documentation of `backend` lists, and the operating system must have
    /// enabled them for this process. Otherwise scoring executes instructions
    /// that the CPU does not have, which is undefined behavior.
    #[must_use]
    #[expect(unsafe_code, reason = "the caller vouches for CPU features")]
    pub unsafe fn with_backend_unchecked(mut self, backend: Backend) -> Self {
        // SAFETY: the caller guarantees the CPU features of `backend`.
        self.backend = unsafe { Verified::new_unchecked(backend) };
        self
    }

    /// Selects the backend of `kernel`.
    ///
    /// A kernel proves its CPU features, so this needs no check. Use it to
    /// share one selection, including one made with
    /// [`Kernel::new_unchecked`], between scorers.
    #[must_use]
    pub fn with_kernel(mut self, kernel: Kernel) -> Self {
        self.backend = kernel.verified();
        self
    }

    /// Returns the backend in use.
    #[must_use]
    pub fn backend(&self) -> Backend {
        self.backend.get()
    }

    /// Returns the kernel in use, which runs on [`Scorer::backend`].
    #[must_use]
    pub fn kernel(&self) -> Kernel {
        Kernel::from_verified(self.backend)
    }

    /// Checks the kernel of the backend in use against the scalar kernel.
    ///
    /// Both kernels compute the sums of built-in matrices: random values at
    /// many widths, the extreme weights and activations, and sums that wrap.
    /// The result is whether every sum is equal. The check does not use the
    /// model. It allocates, and it takes a few milliseconds in an optimized
    /// build.
    ///
    /// A backend that detection selected passes, and its kernel is covered
    /// by the tests of this crate. The check is meant to follow
    /// [`Scorer::with_backend_unchecked`], where a CPU that implements
    /// another revision of an extension could otherwise return wrong scores
    /// silently. It does not make that call sound: a CPU without the
    /// features of the backend can fault or misbehave before the check
    /// returns. It also cannot show that every 32-bit lane of a vector
    /// kernel wraps instead of saturating, which no valid model depends on.
    #[must_use]
    pub fn kernel_self_test(&self) -> bool {
        selftest::run(self.backend)
    }

    /// Returns the model in use.
    #[must_use]
    pub fn model(&self) -> &Model {
        &self.model
    }

    /// Scores `input`.
    ///
    /// Any bytes of any length are accepted. The call never fails and never
    /// panics, runs in time linear in the input length, and allocates no heap
    /// memory. An input shorter than
    /// [`MIN_INPUT_LEN`](crate::fingerprint::MIN_INPUT_LEN) scores
    /// [`Scores::ZERO`]. A longer input scores, in each category, the highest
    /// score of any of its windows.
    #[must_use]
    pub fn score(&self, input: &[u8]) -> Scores {
        let mut windows = fingerprints(input);
        let mut scores = Scores::ZERO;
        while let Some(fingerprint) = windows.next_window() {
            scores = scores.max_with(self.score_fingerprint(fingerprint));
        }
        scores
    }

    /// Scores one fingerprint.
    #[must_use]
    pub fn score_fingerprint(&self, fingerprint: &Fingerprint) -> Scores {
        self.model.infer(self.backend, fingerprint)
    }

    /// Returns the logits of `input`: in each category, the highest logit
    /// of any of its windows.
    ///
    /// [`Logits::scores`] of the result equals [`Scorer::score`], because
    /// the logistic table never decreases. Logits resolve differences that
    /// scores round away near 0 and 255. The call has the contract of
    /// [`Scorer::score`]. An input without a window has the logit -128 in
    /// every category, which scores 0.
    #[must_use]
    pub fn logits(&self, input: &[u8]) -> Logits {
        let mut windows = fingerprints(input);
        let mut logits = Logits::NONE;
        while let Some(fingerprint) = windows.next_window() {
            logits = logits.max_with(self.logits_fingerprint(fingerprint));
        }
        logits
    }

    /// Returns the logits of one fingerprint.
    #[must_use]
    pub fn logits_fingerprint(&self, fingerprint: &Fingerprint) -> Logits {
        self.model.logits(self.kernel(), fingerprint)
    }

    /// Returns the values of every layer of the model for one fingerprint.
    #[must_use]
    pub fn trace_fingerprint(&self, fingerprint: &Fingerprint) -> Trace {
        self.model.trace(self.kernel(), fingerprint)
    }
}

/// The model that a [`ScorerBuilder`] loads.
#[derive(Clone)]
enum Source {
    Embedded(EmbeddedModel),
    Model(Model),
    Parsed(Result<Model, ModelError>),
}

/// The backend that a [`ScorerBuilder`] selects.
#[derive(Clone, Copy, Debug)]
enum Selection {
    Detect,
    Backend(Backend),
    Kernel(Kernel),
}

/// Chooses the model and the backend of a [`Scorer`].
///
/// [`Scorer::builder`] returns one. Without a choice, the builder loads the
/// embedded default model and detects the backend, like [`Scorer::new`]. A
/// later choice of the model or the backend replaces an earlier one. The
/// builder parses a model when it is given and checks the backend in
/// [`build`](Self::build), which reports the first problem.
#[derive(Clone)]
#[must_use]
pub struct ScorerBuilder {
    source: Source,
    selection: Selection,
}

/// Shows the choices and not the model parameters.
impl fmt::Debug for ScorerBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug = f.debug_struct("ScorerBuilder");
        match &self.source {
            Source::Embedded(model) => debug.field("embedded", model),
            Source::Model(model) | Source::Parsed(Ok(model)) => debug.field("model", model),
            Source::Parsed(Err(error)) => debug.field("model", error),
        };
        debug.field("backend", &self.selection).finish()
    }
}

impl Default for ScorerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ScorerBuilder {
    /// Returns a builder for the embedded default model on the detected
    /// backend.
    pub const fn new() -> Self {
        Self {
            source: Source::Embedded(EmbeddedModel::Default),
            selection: Selection::Detect,
        }
    }

    /// Loads an embedded model.
    pub fn embedded(mut self, model: EmbeddedModel) -> Self {
        self.source = Source::Embedded(model);
        self
    }

    /// Runs `model`.
    pub fn model(mut self, model: Model) -> Self {
        self.source = Source::Model(model);
        self
    }

    /// Parses `bytes` with [`Model::from_bytes`] and runs the model.
    /// [`build`](Self::build) reports an error of the parse.
    pub fn model_bytes(mut self, bytes: &[u8]) -> Self {
        self.source = Source::Parsed(Model::from_bytes(bytes));
        self
    }

    /// Detects the preferred backend, which is the default.
    pub fn detect_backend(mut self) -> Self {
        self.selection = Selection::Detect;
        self
    }

    /// Selects `backend`, which [`build`](Self::build) checks like
    /// [`Scorer::with_backend`].
    pub fn backend(mut self, backend: Backend) -> Self {
        self.selection = Selection::Backend(backend);
        self
    }

    /// Selects the backend of `kernel`, like [`Scorer::with_kernel`].
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.selection = Selection::Kernel(kernel);
        self
    }

    /// Creates the scorer.
    ///
    /// # Errors
    ///
    /// Returns [`ScorerError::Model`] when the bytes of
    /// [`model_bytes`](Self::model_bytes) are not a valid model, and
    /// otherwise [`ScorerError::Backend`] when the CPU is not detected to
    /// support the backend of [`backend`](Self::backend).
    pub fn build(self) -> Result<Scorer, ScorerError> {
        let model = match self.source {
            Source::Embedded(model) => model.load(),
            Source::Model(model) => model,
            Source::Parsed(parsed) => parsed?,
        };
        let kernel = match self.selection {
            Selection::Detect => Kernel::detect(),
            Selection::Backend(backend) => Kernel::new(backend)?,
            Selection::Kernel(kernel) => kernel,
        };
        Ok(Scorer::with_model(model).with_kernel(kernel))
    }
}

/// The reason that [`ScorerBuilder::build`] could not create a scorer.
///
/// The error never contains model bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ScorerError {
    /// The model bytes are not a valid model.
    Model(ModelError),
    /// The CPU is not detected to support the requested backend.
    Backend(UnsupportedBackend),
}

impl fmt::Display for ScorerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Model(error) => fmt::Display::fmt(error, f),
            Self::Backend(error) => fmt::Display::fmt(error, f),
        }
    }
}

impl core::error::Error for ScorerError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Model(error) => Some(error),
            Self::Backend(error) => Some(error),
        }
    }
}

impl From<ModelError> for ScorerError {
    fn from(error: ModelError) -> Self {
        Self::Model(error)
    }
}

impl From<UnsupportedBackend> for ScorerError {
    fn from(error: UnsupportedBackend) -> Self {
        Self::Backend(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fingerprint::BLOCK_LEN;
    use crate::model::tests::random_model;
    use crate::testutil::Rng;

    fn scorer(seed: u64) -> Scorer {
        Scorer::with_model(random_model(&mut Rng::new(seed), &[48, 24]))
    }

    #[test]
    fn default_backend_is_the_detected_one() {
        let scorer = scorer(1);
        assert_eq!(scorer.backend(), Backend::detect());
        assert_eq!(scorer.model().layer_sizes().len(), 3);
        let scalar = scorer.with_backend(Backend::Scalar).unwrap();
        assert_eq!(scalar.backend(), Backend::Scalar);
    }

    #[test]
    #[expect(unsafe_code, reason = "exercises the unchecked constructor")]
    fn unchecked_backend_matches_checked_backend() {
        let checked = scorer(9);
        let backend = checked.backend();
        // SAFETY: `backend` was detected on this CPU.
        let unchecked = unsafe { checked.clone().with_backend_unchecked(backend) };
        assert_eq!(unchecked.backend(), backend);
        let input = b"GET /a/../../etc/passwd?q=%27%20OR%201=1 HTTP/1.1";
        assert_eq!(unchecked.score(input), checked.score(input));
    }

    #[test]
    fn kernels_select_the_backend() {
        let scorer = scorer(10);
        assert_eq!(scorer.kernel(), Kernel::detect());
        let scalar = scorer.clone().with_kernel(Kernel::SCALAR);
        assert_eq!(scalar.backend(), Backend::Scalar);
        assert_eq!(scalar.kernel(), Kernel::SCALAR);
        let input = b"<svg onload=alert(1)>";
        assert_eq!(scalar.score(input), scorer.score(input));
    }

    #[test]
    fn logits_order_like_scores() {
        let scorer = scorer(11);
        let mut rng = Rng::new(12);
        for len in [0, 1, 2, 300, 3 * BLOCK_LEN + 9] {
            let mut input = vec![0u8; len];
            rng.fill(&mut input);
            let logits = scorer.logits(&input);
            assert_eq!(logits.scores(), scorer.score(&input), "length {len}");
            for fingerprint in fingerprints(&input) {
                let trace = scorer.trace_fingerprint(&fingerprint);
                assert_eq!(trace.logits(), scorer.logits_fingerprint(&fingerprint));
                assert_eq!(trace.scores(), scorer.score_fingerprint(&fingerprint));
            }
        }
        assert_eq!(scorer.logits(b""), Logits::NONE);
    }

    #[test]
    fn builder_defaults_to_the_default_model_on_the_detected_backend() {
        let scorer = Scorer::builder().build().unwrap();
        assert_eq!(scorer.model(), Scorer::new().model());
        assert_eq!(scorer.backend(), Backend::detect());
        assert_eq!(
            format!("{:?}", ScorerBuilder::default()),
            "ScorerBuilder { embedded: Default, backend: Detect }"
        );
    }

    #[test]
    fn builder_applies_the_last_choice() {
        let model = random_model(&mut Rng::new(13), &[8]);
        let bytes = model.to_bytes();
        let built = Scorer::builder()
            .embedded(EmbeddedModel::Large)
            .model_bytes(&bytes)
            .backend(Backend::Scalar)
            .build()
            .unwrap();
        assert_eq!(built.model(), &model);
        assert_eq!(built.backend(), Backend::Scalar);
        let built = Scorer::builder()
            .model(model.clone())
            .embedded(EmbeddedModel::Compact)
            .kernel(Kernel::SCALAR)
            .detect_backend()
            .build()
            .unwrap();
        assert_eq!(built.model(), &EmbeddedModel::Compact.load());
        assert_eq!(built.backend(), Backend::detect());
        assert_eq!(Scorer::from_bytes(&bytes).unwrap().model(), &model);
        assert_eq!(Scorer::from(model.clone()).model(), &model);
        assert_eq!(
            Scorer::from(EmbeddedModel::Strict).model(),
            &EmbeddedModel::Strict.load()
        );
    }

    #[test]
    fn builder_reports_the_model_before_the_backend() {
        use core::error::Error as _;

        let unsupported = Backend::ALL
            .iter()
            .copied()
            .find(|backend| !backend.is_supported())
            .expect("some backend belongs to another architecture");
        let error = Scorer::builder()
            .model_bytes(b"S2FP")
            .backend(unsupported)
            .build()
            .unwrap_err();
        assert_eq!(error, ScorerError::Model(ModelError::Truncated));
        assert_eq!(error.to_string(), ModelError::Truncated.to_string());
        assert!(error.source().is_some());
        assert!(format!("{:?}", Scorer::builder().model_bytes(b"")).contains("Truncated"));
        let error = Scorer::builder().backend(unsupported).build().unwrap_err();
        assert_eq!(error, ScorerError::Backend(UnsupportedBackend(unsupported)));
        assert_eq!(
            error.to_string(),
            UnsupportedBackend(unsupported).to_string()
        );
        assert_eq!(Scorer::from_bytes(b"").unwrap_err(), ModelError::Truncated);
    }

    #[test]
    fn inputs_without_a_window_score_zero() {
        let scorer = scorer(2);
        assert_eq!(scorer.score(b""), Scores::ZERO);
        for byte in 0..=255u8 {
            assert_eq!(scorer.score(&[byte]), Scores::ZERO);
        }
    }

    #[test]
    fn score_is_the_maximum_over_windows() {
        let scorer = scorer(3);
        let mut rng = Rng::new(4);
        for len in [
            2,
            100,
            BLOCK_LEN,
            2 * BLOCK_LEN,
            2 * BLOCK_LEN + 1,
            5 * BLOCK_LEN + 17,
        ] {
            let mut input = vec![0u8; len];
            rng.fill(&mut input);
            let expected = fingerprints(&input)
                .map(|fingerprint| scorer.score_fingerprint(&fingerprint))
                .fold(Scores::ZERO, Scores::max_with);
            assert_eq!(scorer.score(&input), expected, "length {len}");
        }
    }

    #[test]
    fn appending_input_never_lowers_a_completed_window() {
        let scorer = scorer(5);
        let mut rng = Rng::new(6);
        let mut input = vec![0u8; 6 * BLOCK_LEN];
        rng.fill(&mut input);
        // Once two more blocks follow, the windows over a prefix are final,
        // so the maximum over them can only grow.
        let mut previous = Scores::ZERO;
        for blocks in 2..=6 {
            let scores = scorer.score(&input[..blocks * BLOCK_LEN]);
            assert_eq!(scores.max_with(previous), scores, "{blocks} blocks");
            previous = scores;
        }
    }

    #[test]
    fn every_supported_backend_gives_the_same_scores() {
        let reference = scorer(7).with_backend(Backend::Scalar).unwrap();
        let mut rng = Rng::new(8);
        let mut input = vec![0u8; 3 * BLOCK_LEN + 5];
        rng.fill(&mut input);
        let expected = reference.score(&input);
        for &backend in Backend::PREFERENCE {
            match reference.clone().with_backend(backend) {
                Ok(scorer) => assert_eq!(scorer.score(&input), expected, "backend {backend}"),
                Err(error) => {
                    assert!(!backend.is_supported());
                    assert_eq!(error.backend(), backend);
                }
            }
        }
    }
}
