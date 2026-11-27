//! A self-test that compares a kernel with the scalar kernel.
//!
//! Unit tests cover every kernel that detection reports. This check exists
//! for a kernel that only a caller can vouch for, where a CPU that implements
//! another draft of an extension would otherwise return wrong sums silently.
//! It is ordinary code so that it also runs where the test harness cannot,
//! such as on a simulator without threads.
//!
//! The check has a known limit. No matrix here drives one 32-bit lane of a
//! vector accumulator beyond its range, because the width that this needs
//! grows with the number of lanes. A kernel whose lanes saturate instead of
//! wrapping would pass. A valid model cannot reach that case, because model
//! validation bounds the sum of the magnitudes of every row by `i32::MAX`.

use super::{Matrix, matvec, scalar};
use crate::backend::Verified;

/// Widths above the exhaustive range: around powers of two, and the
/// fingerprint length. They span several steps of every kernel with steps of
/// up to 2048 bytes.
const LARGE_WIDTHS: [usize; 16] = [
    511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049, 3071, 3072, 3073, 4095, 4096, 4097, 4099,
];

/// Every width up to this one is checked.
const EXHAUSTIVE_WIDTHS: usize = 260;

/// Widths for the checks of extreme values.
const EXTREME_WIDTHS: [usize; 12] = [1, 2, 3, 4, 5, 15, 16, 17, 63, 64, 65, 3072];

/// Widths whose sums wrap at 32 bits for the largest weight and activation.
/// The larger one spans more than two steps of 65,536 bytes, the longest
/// step of any kernel, which a RISC-V vector length of 65,536 bits gives.
const LONG_WIDTHS: [usize; 2] = [70_001, 140_017];

/// Returns whether the kernel of `backend` returns the sums of the scalar
/// kernel on the built-in matrices.
pub(crate) fn run(backend: Verified) -> bool {
    run_with(|matrix, input, out| matvec(backend, matrix, input, out))
}

/// A small deterministic generator (SplitMix64).
struct Generator(u64);

impl Generator {
    fn next_byte(&mut self) -> u8 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (value >> 56) as u8
    }

    fn activations(&mut self, len: usize) -> Vec<u8> {
        (0..len).map(|_| self.next_byte()).collect()
    }

    fn weights(&mut self, len: usize) -> Box<[i8]> {
        (0..len)
            .map(|_| i8::from_ne_bytes([self.next_byte()]).max(-127))
            .collect()
    }
}

/// Returns whether `kernel` and the scalar kernel agree on `matrix`.
fn agrees(kernel: &impl Fn(&Matrix, &[u8], &mut [i32]), matrix: &Matrix, input: &[u8]) -> bool {
    let mut expected = vec![0i32; matrix.rows()];
    scalar::matvec(matrix, input, &mut expected);
    // A sentinel shows whether a kernel skipped a row.
    let mut actual = vec![i32::MIN; matrix.rows()];
    kernel(matrix, input, &mut actual);
    actual == expected
}

/// Returns a matrix with `rows` rows that repeat `pattern` along each row.
fn patterned(pattern: &[i8], rows: usize, cols: usize) -> Option<Matrix> {
    let row = pattern.iter().copied().cycle().take(cols);
    let weights: Box<[i8]> = (0..rows).flat_map(|_| row.clone()).collect();
    Matrix::new(weights, cols)
}

/// Returns whether `kernel` returns the sums of the scalar kernel on the
/// built-in matrices. `kernel` has the contract of [`matvec`].
fn run_with(kernel: impl Fn(&Matrix, &[u8], &mut [i32])) -> bool {
    let mut generator = Generator(0x5EED_0F5E_1F7E_57ED);

    // Random values at every small width, so that every tail length of every
    // kernel occurs, and at larger widths.
    for cols in (1..=EXHAUSTIVE_WIDTHS).chain(LARGE_WIDTHS) {
        let Some(matrix) = Matrix::new(generator.weights(2 * cols), cols) else {
            return false;
        };
        // Extra leading bytes vary the alignment of the input.
        let buffer = generator.activations(cols + 3);
        for offset in 0..4 {
            if !agrees(&kernel, &matrix, &buffer[offset..offset + cols]) {
                return false;
            }
        }
    }

    // The largest products of either sign, and the activation values on both
    // sides of the sign bit, which a kernel must read as unsigned.
    for cols in EXTREME_WIDTHS {
        for pattern in [&[127i8][..], &[-127], &[1], &[-1], &[127, -127]] {
            let Some(matrix) = patterned(pattern, 2, cols) else {
                return false;
            };
            for activation in [255u8, 128, 127, 1] {
                if !agrees(&kernel, &matrix, &vec![activation; cols]) {
                    return false;
                }
            }
        }
    }

    for cols in LONG_WIDTHS {
        // Sums beyond 32 bits, which wrap.
        for pattern in [&[127i8][..], &[-127]] {
            let Some(matrix) = patterned(pattern, 1, cols) else {
                return false;
            };
            if !agrees(&kernel, &matrix, &vec![255; cols]) {
                return false;
            }
        }
        // Random values, so that a kernel with long steps must also place
        // every step after the first correctly.
        let Some(matrix) = Matrix::new(generator.weights(cols), cols) else {
            return false;
        };
        if !agrees(&kernel, &matrix, &generator.activations(cols)) {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Backend;

    #[test]
    fn supported_backends_pass() {
        for &backend in Backend::PREFERENCE {
            if let Ok(backend) = Verified::new(backend) {
                assert!(run(backend), "backend {}", backend.get());
            }
        }
    }

    /// The scalar kernel over `i64`, so that a test can break it in one
    /// place.
    fn broken(
        product: impl Fn(u8, i8) -> i64,
        finish: impl Fn(i64, usize) -> i32,
    ) -> impl Fn(&Matrix, &[u8], &mut [i32]) {
        move |matrix, input, out| {
            for (row, sum) in matrix.weights().chunks_exact(matrix.cols()).zip(out) {
                let exact = input.iter().zip(row).map(|(&x, &w)| product(x, w)).sum();
                *sum = finish(exact, row.len());
            }
        }
    }

    fn exact(x: u8, w: i8) -> i64 {
        i64::from(x) * i64::from(w)
    }

    fn wrap(sum: i64, _cols: usize) -> i32 {
        sum as i32
    }

    #[test]
    fn a_correct_kernel_passes() {
        assert!(run_with(broken(exact, wrap)));
    }

    #[test]
    fn wrong_kernels_fail() {
        // Activations read as signed bytes.
        assert!(!run_with(broken(
            |x, w| i64::from(i8::from_ne_bytes([x])) * i64::from(w),
            wrap
        )));
        // Weights read as unsigned bytes.
        assert!(!run_with(broken(
            |x, w| i64::from(x) * i64::from(w.cast_unsigned()),
            wrap
        )));
        // Products that saturate at 16 bits in pairs are approximated by a
        // clamp of each product.
        assert!(!run_with(broken(
            |x, w| exact(x, w).clamp(-32_768 / 2, 32_767 / 2),
            wrap
        )));
        // Sums that saturate instead of wrapping.
        assert!(!run_with(broken(exact, |sum, _| {
            sum.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
        })));
        // A result that is wrong for one width only, like a broken tail.
        for width in [1, 3, 7, 131, 259, 3072, 4099] {
            assert!(!run_with(broken(exact, |sum, cols| {
                wrap(sum, cols).wrapping_add(i32::from(cols == width))
            })));
        }
    }

    #[test]
    fn a_kernel_that_misplaces_a_later_step_fails() {
        // Models a kernel that reads every step after the first from the
        // wrong position, for each step length up to the longest one.
        for step in [16, 64, 256, 2048, 4096, 65_536] {
            assert!(!run_with(|matrix, input, out| {
                for (row, sum) in matrix.weights().chunks_exact(matrix.cols()).zip(out) {
                    let head = step.min(row.len());
                    let later = scalar::dot(&input[head / 4..], &row[head..]);
                    *sum = scalar::dot(&input[..head], &row[..head]).wrapping_add(later);
                }
            }));
        }
    }

    #[test]
    fn a_kernel_that_skips_a_row_or_a_column_fails() {
        assert!(!run_with(|matrix, input, out| {
            scalar::matvec(matrix, input, &mut out[..1]);
        }));
        assert!(!run_with(|matrix, input, out| {
            let cols = matrix.cols();
            for (row, sum) in matrix.weights().chunks_exact(cols).zip(out) {
                *sum = scalar::dot(&input[..cols - 1], row);
            }
        }));
        assert!(!run_with(|matrix, input, out| {
            for (row, sum) in matrix.weights().chunks_exact(matrix.cols()).zip(out) {
                *sum = scalar::dot(&input[1..], &row[1..]);
            }
        }));
    }
}
