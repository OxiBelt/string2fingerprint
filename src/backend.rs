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
    /// x86-64 with AVX-512 F, BW, and VNNI: 64 bytes per `VPDPBUSD`.
    Avx512Vnni,
    /// x86-64 with AVX2 and AVX-VNNI: 32 bytes per VEX-encoded `VPDPBUSD`.
    AvxVnni,
    /// x86-64 with AVX2: 32 bytes per `VPMADDUBSW` and `VPMADDWD` pair.
    Avx2,
}

impl Backend {
    /// The backends for the target architecture in order of preference,
    /// ending with [`Backend::Scalar`].
    pub const PREFERENCE: &'static [Self] = PREFERENCE;

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
        crate::kernel::supports(self)
    }

    /// Returns a stable `snake_case` identifier, suitable for metric labels.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Avx512Vnni => "avx512_vnni",
            Self::AvxVnni => "avx_vnni",
            Self::Avx2 => "avx2",
        }
    }
}

#[cfg(target_arch = "x86_64")]
const PREFERENCE: &[Backend] = &[
    Backend::Avx512Vnni,
    Backend::AvxVnni,
    Backend::Avx2,
    Backend::Scalar,
];

#[cfg(not(target_arch = "x86_64"))]
const PREFERENCE: &[Backend] = &[Backend::Scalar];

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

    /// Accepts `backend` without detection.
    ///
    /// # Safety
    ///
    /// The running CPU must implement every CPU feature of `backend`, and the
    /// operating system must have enabled them for this process.
    #[expect(unsafe_code, reason = "the caller vouches for CPU features")]
    pub(crate) const unsafe fn new_unchecked(backend: Backend) -> Self {
        Self(backend)
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
    fn names_are_unique() {
        let all = [
            Backend::Scalar,
            Backend::Avx512Vnni,
            Backend::AvxVnni,
            Backend::Avx2,
        ];
        for (index, backend) in all.iter().enumerate() {
            assert!(
                all[..index]
                    .iter()
                    .all(|other| other.name() != backend.name())
            );
        }
    }

    #[test]
    fn backends_of_other_architectures_are_unsupported() {
        let x86_64 = [Backend::Avx512Vnni, Backend::AvxVnni, Backend::Avx2];
        if !cfg!(target_arch = "x86_64") {
            for backend in x86_64 {
                assert!(!backend.is_supported());
                assert_eq!(Verified::new(backend), Err(UnsupportedBackend(backend)));
            }
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn x86_64_backends_imply_weaker_ones() {
        // AVX-512 implies AVX2 on every CPU. AVX-VNNI is detected together
        // with AVX2 by construction.
        if Backend::Avx512Vnni.is_supported() || Backend::AvxVnni.is_supported() {
            assert!(Backend::Avx2.is_supported());
        }
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
