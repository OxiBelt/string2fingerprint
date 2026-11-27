//! The scoring entry point.

use crate::backend::{UnsupportedBackend, Verified};
use crate::fingerprint::{Fingerprint, fingerprints};
use crate::model::Model;
use crate::{Backend, Scores};

/// Scores untrusted bytes with a [`Model`].
///
/// A scorer is immutable. Share one between threads by reference.
///
/// ```
/// use string2fingerprint::{Category, Scorer};
///
/// # fn main() -> Result<(), string2fingerprint::model::ModelError> {
/// # let model_bytes = {
/// #     let mut bytes = b"S2FP\x01\x00\x01\x00\x00\x0c\x08\x00\x01\x00\x00\x00".to_vec();
/// #     bytes.resize(bytes.len() + 8 * (8 + 3072), 0);
/// #     let sum = bytes.iter().fold(0x811C_9DC5u32, |hash, &byte| {
/// #         (hash ^ u32::from(byte)).wrapping_mul(0x0100_0193)
/// #     });
/// #     bytes.extend(sum.to_le_bytes());
/// #     bytes
/// # };
/// let model = string2fingerprint::model::Model::from_bytes(&model_bytes)?;
/// let scorer = Scorer::with_model(model);
/// let scores = scorer.score(b"GET /index.html?id=1 HTTP/1.1");
/// let _sql = scores.get(Category::SqlInjection);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct Scorer {
    model: Model,
    backend: Verified,
}

impl Scorer {
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

    /// Returns the backend in use.
    #[must_use]
    pub fn backend(&self) -> Backend {
        self.backend.get()
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
