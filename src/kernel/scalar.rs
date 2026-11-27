//! The portable kernel, and the reference for every other kernel.

use super::Matrix;

/// Returns the sum of `input[i] * weights[i]` over the common length of the
/// two slices, wrapping at 32 bits.
pub(crate) fn dot(input: &[u8], weights: &[i8]) -> i32 {
    input.iter().zip(weights).fold(0i32, |sum, (&x, &w)| {
        sum.wrapping_add(i32::from(x) * i32::from(w))
    })
}

/// See [`super::matvec`].
pub(crate) fn matvec(matrix: &Matrix, input: &[u8], out: &mut [i32]) {
    for (row, sum) in matrix.weights().chunks_exact(matrix.cols()).zip(out) {
        *sum = dot(input, row);
    }
}

#[cfg(test)]
mod tests {
    use super::dot;

    #[test]
    fn dot_handles_extremes_and_length_mismatch() {
        assert_eq!(dot(&[], &[]), 0);
        assert_eq!(dot(&[255, 255], &[127, -127]), 0);
        assert_eq!(dot(&[255; 4], &[-127; 4]), -4 * 255 * 127);
        assert_eq!(dot(&[1, 2, 3], &[1]), 1);
        assert_eq!(dot(&[1], &[1, 2, 3]), 1);
    }
}
