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
    /// AArch64 with SVE and SVE I8MM: one vector length, 16 to 256 bytes, per
    /// `USDOT`.
    SveI8mm,
    /// AArch64 with NEON and I8MM: 16 bytes per `USDOT`.
    NeonI8mm,
    /// AArch64 with NEON and DotProd: 16 bytes per `SDOT`.
    NeonDotProd,
    /// AArch64 with NEON: 16 bytes per `SMULL`, `SMLAL2`, and `SADALP`.
    Neon,
    /// RISC-V 64 with RVV 1.0 and Zvdot4a8i. Reserved: the extension is not
    /// ratified, Linux cannot report it, and this crate has no kernel for it
    /// yet. It is never supported, and selecting it runs the scalar kernel.
    RvvZvdot4a8i,
    /// RISC-V 64 with RVV 1.0: `VLEN / 4` bytes per `vwmulsu.vv` and
    /// `vwadd.wv` pair.
    Rvv,
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
            Self::SveI8mm => "sve_i8mm",
            Self::NeonI8mm => "neon_i8mm",
            Self::NeonDotProd => "neon_dotprod",
            Self::Neon => "neon",
            Self::RvvZvdot4a8i => "rvv_zvdot4a8i",
            Self::Rvv => "rvv",
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

#[cfg(target_arch = "aarch64")]
const PREFERENCE: &[Backend] = &[
    Backend::SveI8mm,
    Backend::NeonI8mm,
    Backend::NeonDotProd,
    Backend::Neon,
    Backend::Scalar,
];

#[cfg(target_arch = "riscv64")]
const PREFERENCE: &[Backend] = &[Backend::RvvZvdot4a8i, Backend::Rvv, Backend::Scalar];

#[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "riscv64"
)))]
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
            Backend::SveI8mm,
            Backend::NeonI8mm,
            Backend::NeonDotProd,
            Backend::Neon,
            Backend::RvvZvdot4a8i,
            Backend::Rvv,
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
        let aarch64 = [
            Backend::SveI8mm,
            Backend::NeonI8mm,
            Backend::NeonDotProd,
            Backend::Neon,
        ];
        let riscv64 = [Backend::RvvZvdot4a8i, Backend::Rvv];
        let mut foreign = Vec::new();
        if !cfg!(target_arch = "x86_64") {
            foreign.extend(x86_64);
        }
        if !cfg!(target_arch = "aarch64") {
            foreign.extend(aarch64);
        }
        if !cfg!(target_arch = "riscv64") {
            foreign.extend(riscv64);
        }
        for backend in foreign {
            assert!(!backend.is_supported());
            assert_eq!(Verified::new(backend), Err(UnsupportedBackend(backend)));
        }
    }

    #[test]
    fn reserved_backends_are_never_supported() {
        assert!(!Backend::RvvZvdot4a8i.is_supported());
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
    #[cfg(target_arch = "aarch64")]
    fn aarch64_backends_imply_weaker_ones() {
        // Every AArch64 backend is detected together with NEON, and SVE I8MM
        // together with NEON I8MM, by construction.
        for backend in [Backend::SveI8mm, Backend::NeonI8mm, Backend::NeonDotProd] {
            if backend.is_supported() {
                assert!(Backend::Neon.is_supported());
            }
        }
        if Backend::SveI8mm.is_supported() {
            assert!(Backend::NeonI8mm.is_supported());
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
