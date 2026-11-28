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
    /// RISC-V 64 with RVV 1.0 and Zvdot4a8i: `VLEN` bytes per `vdot4asu.vv`.
    ///
    /// Experimental. The extension is not ratified and Linux cannot report
    /// it, so this backend is never detected. It runs only through
    /// [`Scorer::with_backend_unchecked`](crate::Scorer::with_backend_unchecked).
    ///
    /// The CPU features of this backend are the `V` extension version 1.0
    /// and the Zvdot4a8i draft as the Spike simulator implemented it under
    /// the name `zvdot4a` at revision `6e18887` of November 2026. In that
    /// draft `vdot4asu.vv vd, vs2, vs1` has the `OP-V` opcode with `funct3`
    /// `0b010` and `funct6` `0b101010`. It is legal without a mask for
    /// 32-bit elements and a register group of eight. For each element it
    /// multiplies each of the four signed bytes of `vs2` by the unsigned
    /// byte of `vs1` at the same position and adds the four products to the
    /// element of `vd`, modulo 2<sup>32</sup>. A CPU that implements another
    /// revision does not have these features. The kernel was executed on
    /// that simulator only. Run
    /// [`Scorer::kernel_self_test`](crate::Scorer::kernel_self_test) after
    /// selecting this backend.
    RvvZvdot4a8i,
    /// RISC-V 64 with RVV 1.0: `VLEN / 4` bytes per `vwmulsu.vv` and
    /// `vwadd.wv` pair.
    Rvv,
}

impl Backend {
    /// The backends for the target architecture in order of preference,
    /// ending with [`Backend::Scalar`].
    pub const PREFERENCE: &'static [Self] = PREFERENCE;

    /// Every backend of every architecture, starting with
    /// [`Backend::Scalar`], then the backends of x86-64, AArch64, and
    /// RISC-V 64, each in order of preference.
    pub const ALL: [Self; 10] = [
        Self::Scalar,
        Self::Avx512Vnni,
        Self::AvxVnni,
        Self::Avx2,
        Self::SveI8mm,
        Self::NeonI8mm,
        Self::NeonDotProd,
        Self::Neon,
        Self::RvvZvdot4a8i,
        Self::Rvv,
    ];

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

    /// Iterates over the backends of [`Backend::PREFERENCE`] that the
    /// running CPU and operating system are detected to support, in order
    /// of preference. The last one is [`Backend::Scalar`].
    pub fn supported() -> impl Iterator<Item = Self> {
        Self::PREFERENCE
            .iter()
            .copied()
            .filter(|backend| backend.is_supported())
    }

    /// Returns whether the running CPU and operating system are detected to
    /// support this backend.
    ///
    /// A backend for another architecture is never supported.
    #[must_use]
    pub fn is_supported(self) -> bool {
        crate::kernel::supports(self)
    }

    /// Returns the CPU features that the backend needs, all of which are
    /// detected before it is selected.
    ///
    /// The names are those of the `std` feature detection macros, except
    /// for features that the macros do not know: `svei8mm` is the Linux name
    /// of SVE I8MM in `AT_HWCAP2`, `v` the vector extension in `AT_HWCAP`,
    /// and `zvdot4a8i` the draft extension that no interface reports.
    /// [`Backend::Scalar`] needs none.
    #[must_use]
    pub const fn cpu_features(self) -> &'static [&'static str] {
        match self {
            Self::Scalar => &[],
            Self::Avx512Vnni => &["avx512f", "avx512bw", "avx512vnni"],
            Self::AvxVnni => &["avx2", "avxvnni"],
            Self::Avx2 => &["avx2"],
            Self::SveI8mm => &["neon", "fp16", "sve", "i8mm", "svei8mm"],
            Self::NeonI8mm => &["neon", "i8mm"],
            Self::NeonDotProd => &["neon", "dotprod"],
            Self::Neon => &["neon"],
            Self::RvvZvdot4a8i => &["v", "zvdot4a8i"],
            Self::Rvv => &["v"],
        }
    }

    /// Returns whether the backend implements an extension that is not
    /// ratified, which makes it impossible to detect. Only
    /// [`Backend::RvvZvdot4a8i`] is.
    #[must_use]
    pub const fn is_experimental(self) -> bool {
        matches!(self, Self::RvvZvdot4a8i)
    }

    /// Returns the backend whose [`name`](Self::name) is `name`.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|backend| backend.name() == name)
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Verified(Backend);

impl Verified {
    /// The scalar backend, which needs no CPU feature.
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
        let all = Backend::ALL;
        for (index, backend) in all.iter().enumerate() {
            assert!(
                all[..index]
                    .iter()
                    .all(|other| other.name() != backend.name())
            );
            assert_eq!(Backend::from_name(backend.name()), Some(*backend));
        }
        assert_eq!(Backend::from_name("AVX2"), None);
        assert_eq!(Backend::from_name(""), None);
    }

    #[test]
    fn every_preference_is_listed_once() {
        for backend in Backend::PREFERENCE {
            assert!(Backend::ALL.contains(backend));
        }
        let mut supported: Vec<Backend> = Backend::supported().collect();
        assert_eq!(supported.first(), Some(&Backend::detect()));
        assert_eq!(supported.pop(), Some(Backend::Scalar));
        for backend in Backend::ALL {
            assert_eq!(
                Backend::supported().any(|other| other == backend),
                backend.is_supported()
            );
        }
    }

    #[test]
    fn features_name_the_detected_extensions() {
        for backend in Backend::ALL {
            let features = backend.cpu_features();
            assert_eq!(features.is_empty(), backend == Backend::Scalar);
            assert_eq!(features.contains(&"zvdot4a8i"), backend.is_experimental());
        }
        assert!(!Backend::Rvv.is_experimental());
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
    fn backends_that_linux_cannot_report_are_never_detected() {
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
