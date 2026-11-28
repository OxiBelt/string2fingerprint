//! The matrix-vector kernel shared by every network layer.
//!
//! A kernel computes, for each row of a matrix of signed 8-bit weights, the
//! dot product of that row with a vector of unsigned 8-bit activations.
//! This is the only operation with CPU-specific implementations. Every
//! implementation returns the same 32-bit results as the scalar one, which
//! is the reference.
//!
//! A [`Kernel`] is a [`Backend`] whose CPU features were established, and
//! [`Weights`] a validated matrix. Together they run the kernel on matrices
//! of the caller, for example to test a backend or to build another network
//! on the same instructions:
//!
//! ```
//! use string2fingerprint::kernel::{Kernel, Weights};
//!
//! let weights = Weights::new(vec![1, 2, 3, -4, -5, -6], 3)?;
//! let mut sums = [0i32; 2];
//! Kernel::detect().matvec(&weights, &[10, 0, 255], &mut sums)?;
//! assert_eq!(sums, [10 + 3 * 255, -40 - 6 * 255]);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use core::fmt;

use crate::Backend;
use crate::backend::{UnsupportedBackend, Verified};

pub(crate) mod scalar;
pub(crate) mod selftest;

#[cfg(target_arch = "x86_64")]
#[path = "x86_64.rs"]
mod arch;

#[cfg(target_arch = "aarch64")]
#[path = "aarch64.rs"]
mod arch;

#[cfg(target_arch = "riscv64")]
#[path = "riscv64.rs"]
mod arch;

/// The target has no kernel of its own.
#[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "riscv64"
)))]
mod arch {
    use super::{Matrix, scalar};
    use crate::Backend;
    use crate::backend::Verified;

    pub(super) fn supports(_backend: Backend) -> bool {
        false
    }

    pub(super) fn matvec(_backend: Verified, matrix: &Matrix, input: &[u8], out: &mut [i32]) {
        scalar::matvec(matrix, input, out);
    }
}

/// A row-major matrix of signed 8-bit weights.
///
/// Every weight is in the range from -127 to 127. Kernels that bias the
/// activations to signed bytes rely on that range and on the stored row sums.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Matrix {
    weights: Box<[i8]>,
    row_sums: Box<[i32]>,
    cols: usize,
}

impl Matrix {
    /// Creates a matrix with `cols` columns from row-major `weights`.
    ///
    /// Returns `None` when `cols` is zero, when `weights` is not a whole
    /// number of rows, or when a weight is -128.
    pub(crate) fn new(weights: Box<[i8]>, cols: usize) -> Option<Self> {
        if cols == 0 || !weights.len().is_multiple_of(cols) || weights.contains(&i8::MIN) {
            return None;
        }
        // Kernels use the row sums modulo 2^32 only, so a row too wide for
        // an exact sum still gives the wrapping dot product.
        let row_sums = weights
            .chunks_exact(cols)
            .map(|row| {
                row.iter()
                    .fold(0i32, |sum, &weight| sum.wrapping_add(i32::from(weight)))
            })
            .collect();
        Some(Self {
            weights,
            row_sums,
            cols,
        })
    }

    /// Returns the number of rows.
    pub(crate) fn rows(&self) -> usize {
        self.row_sums.len()
    }

    /// Returns the number of columns, which is never zero.
    pub(crate) fn cols(&self) -> usize {
        self.cols
    }

    /// Returns the weights in row-major order.
    pub(crate) fn weights(&self) -> &[i8] {
        &self.weights
    }

    /// Returns the weights in row-major order.
    pub(crate) fn into_weights(self) -> Box<[i8]> {
        self.weights
    }

    /// Returns the sum of the weights of each row.
    #[cfg_attr(
        not(any(test, target_arch = "x86_64", target_arch = "aarch64")),
        expect(dead_code, reason = "used by kernels that bias the activations")
    )]
    pub(crate) fn row_sums(&self) -> &[i32] {
        &self.row_sums
    }

    /// Returns `sum(abs(weight)) * 255` for each row, which bounds the
    /// magnitude of the dot product of that row with any input.
    pub(crate) fn abs_dot_bounds(&self) -> impl ExactSizeIterator<Item = u64> + '_ {
        self.weights.chunks_exact(self.cols).map(|row| {
            row.iter()
                .map(|&weight| u64::from(weight.unsigned_abs()))
                .sum::<u64>()
                * 255
        })
    }
}

/// A backend whose CPU features were established, which runs the kernel.
///
/// Holding a kernel is the proof that its target-specific instructions can
/// execute. [`Kernel::new`] and [`Kernel::detect`] establish it by
/// detection, and [`Kernel::new_unchecked`] takes the word of the caller.
/// Every kernel returns the same sums for the same arguments.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Kernel(Verified);

impl Kernel {
    /// The scalar kernel, which needs no CPU feature.
    pub const SCALAR: Self = Self(Verified::SCALAR);

    /// Returns the kernel of [`Backend::detect`].
    #[must_use]
    pub fn detect() -> Self {
        Self(Verified::detect())
    }

    /// Returns the kernel of `backend`.
    ///
    /// # Errors
    ///
    /// Returns [`UnsupportedBackend`] when the running CPU and operating
    /// system are not detected to support `backend`.
    pub fn new(backend: Backend) -> Result<Self, UnsupportedBackend> {
        Verified::new(backend).map(Self)
    }

    /// Returns the kernel of `backend` without detecting its CPU features.
    ///
    /// This exists for backends that the operating system cannot report
    /// yet. Prefer [`Kernel::new`] for every other backend, and run
    /// [`Kernel::self_test`] before using the result.
    ///
    /// # Safety
    ///
    /// The running CPU must implement every CPU feature that the
    /// documentation of `backend` lists, and the operating system must have
    /// enabled them for this process. Otherwise the kernel executes
    /// instructions that the CPU does not have, which is undefined behavior.
    #[must_use]
    #[expect(unsafe_code, reason = "the caller vouches for CPU features")]
    pub const unsafe fn new_unchecked(backend: Backend) -> Self {
        // SAFETY: the caller guarantees the CPU features of `backend`.
        Self(unsafe { Verified::new_unchecked(backend) })
    }

    /// Returns the backend of the kernel.
    #[must_use]
    pub const fn backend(self) -> Backend {
        self.0.get()
    }

    /// Returns the proof that the scorer and the model rely on.
    pub(crate) const fn verified(self) -> Verified {
        self.0
    }

    /// Wraps a proof that the crate holds.
    pub(crate) const fn from_verified(verified: Verified) -> Self {
        Self(verified)
    }

    /// Writes the dot product of `input` with each row of `weights` to
    /// `out`.
    ///
    /// A sum wraps at 32 bits, so every backend returns the same values. A
    /// sum cannot wrap when 255 times the sum of the weight magnitudes of
    /// its row is at most `i32::MAX`, which [`Weights::dot_bounds`] gives.
    /// The call allocates nothing and takes time linear in the number of
    /// weights.
    ///
    /// # Errors
    ///
    /// Returns [`ShapeError`] and leaves `out` unchanged when `input` does
    /// not have [`Weights::cols`] elements or `out` does not have
    /// [`Weights::rows`] elements.
    pub fn matvec(
        self,
        weights: &Weights,
        input: &[u8],
        out: &mut [i32],
    ) -> Result<(), ShapeError> {
        if input.len() != weights.0.cols() || out.len() != weights.0.rows() {
            return Err(ShapeError(()));
        }
        matvec(self.0, &weights.0, input, out);
        Ok(())
    }

    /// Checks this kernel against the scalar kernel on built-in matrices.
    ///
    /// The result is whether every sum is equal. The check allocates, and
    /// it takes a few milliseconds in an optimized build. A kernel that
    /// detection selected passes. The check is meant to follow
    /// [`Kernel::new_unchecked`], where a CPU that implements another
    /// revision of an extension could otherwise return wrong sums silently.
    /// It does not make that call sound, and it cannot show that every
    /// 32-bit lane of a vector kernel wraps instead of saturating, which no
    /// valid model depends on.
    #[must_use]
    pub fn self_test(self) -> bool {
        selftest::run(self.0)
    }
}

impl Default for Kernel {
    /// Returns [`Kernel::detect`].
    fn default() -> Self {
        Self::detect()
    }
}

impl fmt::Display for Kernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.backend(), f)
    }
}

/// A validated row-major matrix of signed 8-bit weights for a [`Kernel`].
///
/// Every weight is in the range from -127 to 127, because kernels that bias
/// the activations to signed bytes rely on it. The matrix keeps the sum of
/// each row, so creating it reads every weight once.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Weights(Matrix);

impl Weights {
    /// Creates a matrix with `cols` columns from row-major `weights`.
    ///
    /// The number of rows is `weights.len() / cols`, and can be zero.
    ///
    /// # Errors
    ///
    /// Returns [`WeightsError`] when `cols` is zero, when `weights` is not a
    /// whole number of rows, or when a weight is -128.
    pub fn new(weights: Vec<i8>, cols: usize) -> Result<Self, WeightsError> {
        if cols == 0 {
            return Err(WeightsError::NoColumns);
        }
        if !weights.len().is_multiple_of(cols) {
            return Err(WeightsError::PartialRow);
        }
        Matrix::new(weights.into(), cols)
            .map(Self)
            .ok_or(WeightsError::WeightOutOfRange)
    }

    /// Returns the number of rows.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.0.rows()
    }

    /// Returns the number of columns, which is never zero.
    #[must_use]
    pub fn cols(&self) -> usize {
        self.0.cols()
    }

    /// Returns the weights in row-major order.
    #[must_use]
    pub fn as_slice(&self) -> &[i8] {
        self.0.weights()
    }

    /// Returns the weights of row `row`, or `None` when there is no such
    /// row.
    #[must_use]
    pub fn row(&self, row: usize) -> Option<&[i8]> {
        self.0.weights().chunks_exact(self.0.cols()).nth(row)
    }

    /// Returns 255 times the sum of the weight magnitudes of each row,
    /// which bounds the magnitude of its dot product with any input.
    pub fn dot_bounds(&self) -> impl ExactSizeIterator<Item = u64> + '_ {
        self.0.abs_dot_bounds()
    }

    /// Returns the weights in row-major order.
    #[must_use]
    pub fn into_vec(self) -> Vec<i8> {
        self.0.into_weights().into_vec()
    }
}

/// The reason that [`Weights::new`] refused a matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum WeightsError {
    /// The number of columns is zero.
    NoColumns,
    /// The number of weights is not a multiple of the number of columns.
    PartialRow,
    /// A weight is -128.
    WeightOutOfRange,
}

impl fmt::Display for WeightsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::NoColumns => "weight matrix has no columns",
            Self::PartialRow => "weights are not a whole number of rows",
            Self::WeightOutOfRange => "weight is -128",
        })
    }
}

impl core::error::Error for WeightsError {}

/// The error that [`Kernel::matvec`] returns when the lengths of its
/// arguments do not match the [`Weights`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ShapeError(());

impl fmt::Display for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("input or output length does not match the weight matrix")
    }
}

impl core::error::Error for ShapeError {}

/// Writes the dot product of `input` with each row of `matrix` to `out`.
///
/// Sums wrap at 32 bits, so every backend returns the same values for the
/// same arguments.
///
/// `input` must have [`Matrix::cols`] elements and `out` must have
/// [`Matrix::rows`] elements. Every caller in this crate guarantees both. A
/// call that breaks either is memory-safe and does not panic in release
/// builds, but the values in `out` are then unspecified and can differ
/// between backends.
pub(crate) fn matvec(backend: Verified, matrix: &Matrix, input: &[u8], out: &mut [i32]) {
    debug_assert_eq!(input.len(), matrix.cols());
    debug_assert_eq!(out.len(), matrix.rows());
    match backend.get() {
        Backend::Scalar => scalar::matvec(matrix, input, out),
        _ => arch::matvec(backend, matrix, input, out),
    }
}

/// Returns whether the running CPU and operating system are detected to
/// support `backend`.
pub(crate) fn supports(backend: Backend) -> bool {
    backend == Backend::Scalar || arch::supports(backend)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::testutil::Rng;

    /// Returns a matrix of weights drawn from the full permitted range.
    pub(crate) fn random_matrix(rng: &mut Rng, rows: usize, cols: usize) -> Matrix {
        let mut bytes = vec![0u8; rows * cols];
        rng.fill(&mut bytes);
        let weights = bytes
            .into_iter()
            .map(|byte| i8::from_ne_bytes([byte]).max(-127))
            .collect();
        Matrix::new(weights, cols).expect("weights are whole rows above -128")
    }

    #[test]
    fn matrix_rejects_invalid_shapes_and_weights() {
        assert!(Matrix::new(vec![1, 2, 3].into(), 0).is_none());
        assert!(Matrix::new(vec![1, 2, 3].into(), 2).is_none());
        assert!(Matrix::new(vec![1, -128, 3, 4].into(), 2).is_none());
        let matrix = Matrix::new(vec![1, -127, 3, 127].into(), 2).unwrap();
        assert_eq!((matrix.rows(), matrix.cols()), (2, 2));
        assert_eq!(matrix.row_sums(), &[-126, 130]);
        assert_eq!(
            matrix.abs_dot_bounds().collect::<Vec<_>>(),
            [128 * 255, 130 * 255]
        );
        let empty = Matrix::new(Vec::new().into(), 4).unwrap();
        assert_eq!((empty.rows(), empty.abs_dot_bounds().len()), (0, 0));
    }

    #[test]
    fn scalar_matvec_matches_definition() {
        let matrix = Matrix::new(vec![1, 2, 3, -4, -5, -6].into(), 3).unwrap();
        let mut out = [0i32; 2];
        matvec(Verified::SCALAR, &matrix, &[10, 0, 255], &mut out);
        assert_eq!(out, [10 + 765, -40 - 1530]);
    }

    /// Returns the backends that can run here, with the scalar one last.
    fn supported_backends() -> Vec<Verified> {
        Backend::PREFERENCE
            .iter()
            .filter_map(|&backend| Verified::new(backend).ok())
            .collect()
    }

    /// Compares every supported backend with the scalar kernel.
    fn assert_backends_agree(matrix: &Matrix, input: &[u8], context: &str) {
        let mut expected = vec![0i32; matrix.rows()];
        scalar::matvec(matrix, input, &mut expected);
        for backend in supported_backends() {
            // A sentinel shows whether a kernel skipped a row.
            let mut actual = vec![i32::MIN; matrix.rows()];
            matvec(backend, matrix, input, &mut actual);
            assert_eq!(actual, expected, "backend {}, {context}", backend.get());
        }
    }

    #[test]
    fn backends_agree_on_random_matrices_of_every_width() {
        let mut rng = Rng::new(21);
        // Every width up to a few vectors of the widest instruction set, then
        // widths around larger powers of two, then the fingerprint length.
        let widths = (1..=200).chain([
            255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049, 3071, 3072, 3073,
        ]);
        for cols in widths {
            for rows in [1, 3, 8] {
                let matrix = random_matrix(&mut rng, rows, cols);
                // Extra leading bytes vary the alignment of the input.
                let mut buffer = vec![0u8; cols + 3];
                rng.fill(&mut buffer);
                for offset in 0..4 {
                    let input = &buffer[offset..offset + cols - offset.min(cols - 1)];
                    let matrix = if input.len() == cols {
                        matrix.clone()
                    } else {
                        random_matrix(&mut rng, rows, input.len())
                    };
                    assert_backends_agree(&matrix, input, &format!("{rows}x{cols}+{offset}"));
                }
            }
        }
    }

    #[test]
    fn backends_agree_on_extreme_values() {
        for cols in [1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 3072] {
            for weight in [127i8, -127, 1, -1, 0] {
                let matrix = Matrix::new(vec![weight; 2 * cols].into(), cols).unwrap();
                for activation in [255u8, 128, 127, 1, 0] {
                    let input = vec![activation; cols];
                    let context = format!("{cols} columns of {weight} times {activation}");
                    assert_backends_agree(&matrix, &input, &context);
                }
            }
            // Alternating signs exercise pairwise saturation in kernels that
            // multiply into 16-bit lanes.
            let weights: Vec<i8> = (0..2 * cols)
                .map(|index| if index % 2 == 0 { 127 } else { -127 })
                .collect();
            let matrix = Matrix::new(weights.into(), cols).unwrap();
            let input: Vec<u8> = (0..cols)
                .map(|index| if index % 3 == 0 { 255 } else { 0 })
                .collect();
            assert_backends_agree(&matrix, &input, &format!("{cols} alternating columns"));
        }
    }

    #[test]
    fn backends_agree_when_sums_wrap() {
        // A model never has rows this wide, because validation bounds every
        // sum by `i32::MAX`. The kernel contract still defines the result:
        // the sum wraps at 32 bits, here several times over.
        for cols in [70_001, 140_017] {
            for weights in [[127i8, 127], [-127, -127], [127, -127]] {
                let row: Vec<i8> = (0..cols).map(|index| weights[index % 2]).collect();
                let matrix = Matrix::new(row.into(), cols).unwrap();
                let input = vec![255u8; cols];
                let context = format!("{cols} columns of {weights:?}");
                assert_backends_agree(&matrix, &input, &context);
            }
            let exact = 255 * 127 * cols as i64;
            assert!(exact > i64::from(i32::MAX));
            let matrix = Matrix::new(vec![127; cols].into(), cols).unwrap();
            let mut out = [0i32];
            matvec(Verified::SCALAR, &matrix, &vec![255; cols], &mut out);
            assert_eq!(i64::from(out[0]), i64::from(exact as i32));
        }
    }

    #[test]
    fn backends_are_reported() {
        // Run with `--nocapture` to record which kernels a test run executed.
        for &backend in Backend::PREFERENCE {
            let status = if supports(backend) {
                "executed"
            } else {
                "skipped, not supported"
            };
            eprintln!("kernel backend {backend}: {status}");
        }
    }

    #[test]
    fn public_kernels_match_the_scalar_kernel() {
        let mut rng = Rng::new(31);
        for (rows, cols) in [(0, 5), (1, 1), (3, 17), (8, 3072), (2, 70_001)] {
            let matrix = random_matrix(&mut rng, rows, cols);
            let weights = Weights::new(matrix.weights().to_vec(), cols).unwrap();
            assert_eq!((weights.rows(), weights.cols()), (rows, cols));
            assert_eq!(weights.as_slice(), matrix.weights());
            assert!(weights.dot_bounds().eq(matrix.abs_dot_bounds()));
            let mut input = vec![0u8; cols];
            rng.fill(&mut input);
            let mut expected = vec![0i32; rows];
            Kernel::SCALAR
                .matvec(&weights, &input, &mut expected)
                .unwrap();
            for backend in Backend::supported() {
                let kernel = Kernel::new(backend).unwrap();
                assert_eq!(kernel.backend(), backend);
                let mut actual = vec![i32::MIN; rows];
                kernel.matvec(&weights, &input, &mut actual).unwrap();
                assert_eq!(actual, expected, "backend {kernel}, {rows}x{cols}");
            }
        }
    }

    #[test]
    fn public_kernels_check_shapes() {
        let weights = Weights::new(vec![1, 2, 3, 4, 5, 6], 3).unwrap();
        assert_eq!(weights.row(1), Some(&[4, 5, 6][..]));
        assert_eq!(weights.row(2), None);
        assert_eq!(weights.row(usize::MAX), None);
        let kernel = Kernel::detect();
        let mut out = [7i32; 2];
        assert!(kernel.matvec(&weights, &[1, 2], &mut out).is_err());
        assert!(kernel.matvec(&weights, &[1, 2, 3, 4], &mut out).is_err());
        assert!(kernel.matvec(&weights, &[1, 2, 3], &mut out[..1]).is_err());
        assert_eq!(out, [7, 7]);
        assert_eq!(
            kernel.matvec(&weights, &[1, 2, 3], &mut []),
            Err(ShapeError(()))
        );
        kernel.matvec(&weights, &[1, 1, 1], &mut out).unwrap();
        assert_eq!(out, [6, 15]);
        assert_eq!(weights.clone().into_vec(), [1, 2, 3, 4, 5, 6]);
        assert!(!ShapeError(()).to_string().is_empty());
    }

    #[test]
    fn weights_reject_invalid_matrices() {
        assert_eq!(Weights::new(vec![1], 0), Err(WeightsError::NoColumns));
        assert_eq!(Weights::new(vec![], 0), Err(WeightsError::NoColumns));
        assert_eq!(
            Weights::new(vec![1, 2, 3], 2),
            Err(WeightsError::PartialRow)
        );
        assert_eq!(
            Weights::new(vec![1, -128], 2),
            Err(WeightsError::WeightOutOfRange)
        );
        assert_eq!(Weights::new(vec![], 4).unwrap().rows(), 0);
        for error in [
            WeightsError::NoColumns,
            WeightsError::PartialRow,
            WeightsError::WeightOutOfRange,
        ] {
            assert!(!error.to_string().is_empty());
        }
    }

    #[test]
    fn rows_whose_sum_wraps_are_accepted() {
        // 127 times this many columns exceeds `i32::MAX`, so the row sum
        // wraps. The kernels still agree on the wrapping dot product.
        let cols = 17_000_001;
        let weights = Weights::new(vec![127; cols], cols).unwrap();
        let input = vec![255u8; cols];
        let mut expected = [0i32];
        Kernel::SCALAR
            .matvec(&weights, &input, &mut expected)
            .unwrap();
        assert_eq!(
            i64::from(expected[0]),
            i64::from((127 * 255 * cols as i64) as i32)
        );
        for backend in Backend::supported() {
            let mut actual = [0i32];
            Kernel::new(backend)
                .unwrap()
                .matvec(&weights, &input, &mut actual)
                .unwrap();
            assert_eq!(actual, expected, "backend {backend}");
        }
    }

    #[test]
    fn kernels_are_created_only_for_supported_backends() {
        assert_eq!(Kernel::detect().backend(), Backend::detect());
        assert_eq!(Kernel::default(), Kernel::detect());
        assert_eq!(Kernel::SCALAR.backend(), Backend::Scalar);
        assert_eq!(Kernel::SCALAR.to_string(), "scalar");
        for &backend in Backend::ALL {
            match Kernel::new(backend) {
                Ok(kernel) => {
                    assert!(backend.is_supported());
                    assert!(kernel.self_test());
                }
                Err(error) => {
                    assert!(!backend.is_supported());
                    assert_eq!(error.backend(), backend);
                }
            }
        }
    }

    #[test]
    #[expect(unsafe_code, reason = "exercises the unchecked constructor")]
    fn unchecked_kernels_keep_their_backend() {
        let detected = Backend::detect();
        // SAFETY: `detected` was detected on this CPU.
        let kernel = unsafe { Kernel::new_unchecked(detected) };
        assert_eq!(kernel, Kernel::detect());
    }

    #[test]
    fn backends_agree_on_empty_matrices() {
        let matrix = Matrix::new(Vec::new().into(), 7).unwrap();
        assert_backends_agree(&matrix, &[1; 7], "no rows");
    }
}
