//! Kernel backends and their selection.

use core::fmt;

/// An implementation of the dot-product kernel.
///
/// Every backend returns bit-identical results, so the choice affects speed
/// only. [`Backend::detect`] picks the preferred backend that the running CPU
/// supports.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Backend {
    /// Portable safe Rust. Supported everywhere.
    Scalar,
}

impl Backend {
    /// The backends for the target architecture in order of preference,
    /// ending with [`Backend::Scalar`].
    pub const PREFERENCE: &'static [Self] = &[Self::Scalar];

    /// Returns the first backend in [`Backend::PREFERENCE`] that the running
    /// CPU and operating system are detected to support.
    #[must_use]
    pub fn detect() -> Self {
        Self::PREFERENCE
            .iter()
            .copied()
            .find(|backend| backend.is_supported())
            .unwrap_or(Self::Scalar)
    }

    /// Returns whether the running CPU and operating system are detected to
    /// support this backend.
    ///
    /// A backend for another architecture is never supported.
    #[must_use]
    pub fn is_supported(self) -> bool {
        match self {
            Self::Scalar => true,
        }
    }

    /// Returns a stable `snake_case` identifier, suitable for metric labels.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
        }
    }
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// The error returned when a requested [`Backend`] is not supported by the
/// running CPU and operating system.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnsupportedBackend(pub(crate) Backend);

impl UnsupportedBackend {
    /// Returns the backend that was requested.
    #[must_use]
    pub const fn backend(&self) -> Backend {
        self.0
    }
}

impl fmt::Display for UnsupportedBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "backend `{}` is not supported on this CPU", self.0)
    }
}

impl core::error::Error for UnsupportedBackend {}

/// A backend whose CPU features were established.
///
/// Holding a value of this type is the proof that target-specific kernels
/// rely on. It is created only after detection succeeds, or through an
/// `unsafe` constructor whose caller vouches for the CPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Verified(Backend);

impl Verified {
    /// The scalar backend, which needs no CPU feature.
    #[cfg_attr(not(test), expect(dead_code, reason = "used by tests and benchmarks"))]
    pub(crate) const SCALAR: Self = Self(Backend::Scalar);

    /// Returns the preferred supported backend.
    pub(crate) fn detect() -> Self {
        Self(Backend::detect())
    }

    /// Verifies that `backend` is supported.
    pub(crate) fn new(backend: Backend) -> Result<Self, UnsupportedBackend> {
        if backend.is_supported() {
            Ok(Self(backend))
        } else {
            Err(UnsupportedBackend(backend))
        }
    }

    /// Returns the verified backend.
    pub(crate) const fn get(self) -> Backend {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preference_ends_with_scalar_and_has_no_duplicates() {
        assert_eq!(Backend::PREFERENCE.last(), Some(&Backend::Scalar));
        for (index, backend) in Backend::PREFERENCE.iter().enumerate() {
            assert!(!Backend::PREFERENCE[..index].contains(backend));
        }
    }

    #[test]
    fn detect_returns_first_supported_preference() {
        let detected = Backend::detect();
        assert!(detected.is_supported());
        let position = Backend::PREFERENCE
            .iter()
            .position(|&backend| backend == detected)
            .expect("detected backend is listed");
        assert!(
            Backend::PREFERENCE[..position]
                .iter()
                .all(|backend| !backend.is_supported())
        );
        assert_eq!(Verified::detect().get(), detected);
    }

    #[test]
    fn scalar_is_always_supported() {
        assert!(Backend::Scalar.is_supported());
        assert_eq!(Verified::new(Backend::Scalar), Ok(Verified::SCALAR));
        assert_eq!(Backend::Scalar.to_string(), "scalar");
    }

    #[test]
    fn unsupported_error_names_the_backend() {
        let error = UnsupportedBackend(Backend::Scalar);
        assert_eq!(error.backend(), Backend::Scalar);
        assert_eq!(
            error.to_string(),
            "backend `scalar` is not supported on this CPU"
        );
    }
}
