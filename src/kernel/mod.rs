//! The matrix-vector kernel shared by every network layer.
//!
//! A kernel computes, for each row of a [`Matrix`] of signed 8-bit weights,
//! the dot product of that row with a vector of unsigned 8-bit activations.
//! This is the only operation with CPU-specific implementations. Every
//! implementation returns the same 32-bit results as [`scalar`], which is the
//! reference.

use crate::Backend;
use crate::backend::Verified;

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
        let row_sums = weights
            .chunks_exact(cols)
            .map(|row| row.iter().map(|&weight| i32::from(weight)).sum())
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
    fn backends_agree_on_empty_matrices() {
        let matrix = Matrix::new(Vec::new().into(), 7).unwrap();
        assert_backends_agree(&matrix, &[1; 7], "no rows");
    }
}
