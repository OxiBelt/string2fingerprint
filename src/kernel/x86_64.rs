//! Kernels for x86-64: AVX-512 VNNI, AVX-VNNI, and AVX2.
//!
//! Unsafe code in this module has two purposes, and a safe implementation
//! cannot serve either: loading vectors through pointers, and calling a
//! function compiled for CPU features that the build does not enable
//! globally. Each use states the invariant it relies on.

use core::arch::x86_64::{
    __m256i, __mmask64, _mm_add_epi32, _mm_cvtsi128_si32, _mm_shuffle_epi32, _mm256_abs_epi8,
    _mm256_add_epi32, _mm256_castsi256_si128, _mm256_dpbusd_avx_epi32, _mm256_extracti128_si256,
    _mm256_loadu_si256, _mm256_madd_epi16, _mm256_maddubs_epi16, _mm256_set1_epi8,
    _mm256_set1_epi16, _mm256_setzero_si256, _mm256_sign_epi8, _mm256_xor_si256,
    _mm512_dpbusd_epi32, _mm512_loadu_si512, _mm512_maskz_loadu_epi8, _mm512_reduce_add_epi32,
    _mm512_setzero_si512,
};
use std::arch::is_x86_feature_detected;

use super::{Matrix, scalar};
use crate::Backend;
use crate::backend::Verified;

/// Returns whether the CPU features of an x86-64 `backend` are all detected.
///
/// Each arm lists exactly the features that the matching kernel enables.
/// Detection also confirms that the operating system saves the vector state.
pub(super) fn supports(backend: Backend) -> bool {
    match backend {
        Backend::Avx512Vnni => {
            is_x86_feature_detected!("avx512f")
                && is_x86_feature_detected!("avx512bw")
                && is_x86_feature_detected!("avx512vnni")
        }
        Backend::AvxVnni => is_x86_feature_detected!("avx2") && is_x86_feature_detected!("avxvnni"),
        Backend::Avx2 => is_x86_feature_detected!("avx2"),
        _ => false,
    }
}

/// See [`super::matvec`].
#[expect(
    unsafe_code,
    reason = "calls kernels compiled for CPU features that `Verified` established"
)]
pub(super) fn matvec(backend: Verified, matrix: &Matrix, input: &[u8], out: &mut [i32]) {
    match backend.get() {
        // SAFETY: a `Verified` backend exists only after `supports` returned
        // true for it, which detected `avx512f`, `avx512bw`, and
        // `avx512vnni`, or after a caller of its unsafe constructor vouched
        // for the same features.
        Backend::Avx512Vnni => unsafe { matvec_avx512_vnni(matrix, input, out) },
        // SAFETY: as above, for `avx2` and `avxvnni`.
        Backend::AvxVnni => unsafe { matvec_avx_vnni(matrix, input, out) },
        // SAFETY: as above, for `avx2`.
        Backend::Avx2 => unsafe { matvec_avx2(matrix, input, out) },
        _ => scalar::matvec(matrix, input, out),
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
fn matvec_avx512_vnni(matrix: &Matrix, input: &[u8], out: &mut [i32]) {
    for (row, sum) in matrix.weights().chunks_exact(matrix.cols()).zip(out) {
        *sum = dot_avx512_vnni(input, row);
    }
}

/// Returns the wrapping sum of `input[i] * weights[i]` over the common length
/// of the two slices, 64 bytes per `VPDPBUSD`.
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
#[inline]
#[expect(unsafe_code, reason = "loads vectors through pointers")]
fn dot_avx512_vnni(input: &[u8], weights: &[i8]) -> i32 {
    let len = input.len().min(weights.len());
    let (input, input_tail) = input[..len].as_chunks::<64>();
    let (weights, weight_tail) = weights[..len].as_chunks::<64>();
    let mut acc = _mm512_setzero_si512();
    for (x, w) in input.iter().zip(weights) {
        // SAFETY: `x` and `w` each refer to 64 initialized bytes, and an
        // unaligned load has no alignment requirement.
        let (x, w) = unsafe {
            (
                _mm512_loadu_si512(x.as_ptr().cast()),
                _mm512_loadu_si512(w.as_ptr().cast()),
            )
        };
        acc = _mm512_dpbusd_epi32(acc, x, w);
    }
    if !input_tail.is_empty() {
        // Both tails have the same length, from 1 to 63.
        let mask: __mmask64 = (1u64 << input_tail.len()) - 1;
        // SAFETY: a masked load accesses only the bytes that the mask
        // selects, which are the `input_tail.len()` initialized bytes of each
        // tail. Masked-off bytes read as zero and add nothing to the sum.
        let (x, w) = unsafe {
            (
                _mm512_maskz_loadu_epi8(mask, input_tail.as_ptr().cast()),
                _mm512_maskz_loadu_epi8(mask, weight_tail.as_ptr()),
            )
        };
        acc = _mm512_dpbusd_epi32(acc, x, w);
    }
    _mm512_reduce_add_epi32(acc)
}

#[target_feature(enable = "avx2,avxvnni")]
fn matvec_avx_vnni(matrix: &Matrix, input: &[u8], out: &mut [i32]) {
    for (row, sum) in matrix.weights().chunks_exact(matrix.cols()).zip(out) {
        *sum = dot_avx_vnni(input, row);
    }
}

/// Returns the wrapping sum of `input[i] * weights[i]` over the common length
/// of the two slices, 32 bytes per VEX-encoded `VPDPBUSD`.
#[target_feature(enable = "avx2,avxvnni")]
#[inline]
#[expect(unsafe_code, reason = "loads vectors through pointers")]
fn dot_avx_vnni(input: &[u8], weights: &[i8]) -> i32 {
    let len = input.len().min(weights.len());
    let (input, input_tail) = input[..len].as_chunks::<32>();
    let (weights, weight_tail) = weights[..len].as_chunks::<32>();
    let mut acc = _mm256_setzero_si256();
    for (x, w) in input.iter().zip(weights) {
        // SAFETY: `x` and `w` each refer to 32 initialized bytes, and an
        // unaligned load has no alignment requirement.
        let (x, w) = unsafe {
            (
                _mm256_loadu_si256(x.as_ptr().cast()),
                _mm256_loadu_si256(w.as_ptr().cast()),
            )
        };
        acc = _mm256_dpbusd_avx_epi32(acc, x, w);
    }
    sum_epi32(acc).wrapping_add(scalar::dot(input_tail, weight_tail))
}

#[target_feature(enable = "avx2")]
fn matvec_avx2(matrix: &Matrix, input: &[u8], out: &mut [i32]) {
    let rows = matrix.weights().chunks_exact(matrix.cols());
    for ((row, &row_sum), sum) in rows.zip(matrix.row_sums()).zip(out) {
        // `sum((x - 128) * w) + 128 * sum(w)` equals `sum(x * w)`.
        *sum = dot_biased_avx2(input, row).wrapping_add(row_sum.wrapping_mul(128));
    }
}

/// Returns the wrapping sum of `(input[i] - 128) * weights[i]` over the
/// common length of the two slices.
///
/// AVX2 has no instruction that multiplies unsigned by signed bytes without
/// saturating: `VPMADDUBSW` adds adjacent products into signed 16 bits, and
/// `255 * 127 * 2` does not fit. Biasing the activations to signed bytes
/// bounds each product by `128 * 127`, so a pair sums to at most 32512 in
/// magnitude and never saturates. This needs every weight above -128, which
/// [`Matrix`] guarantees.
#[target_feature(enable = "avx2")]
#[inline]
#[expect(unsafe_code, reason = "loads vectors through pointers")]
fn dot_biased_avx2(input: &[u8], weights: &[i8]) -> i32 {
    let len = input.len().min(weights.len());
    let (input, input_tail) = input[..len].as_chunks::<32>();
    let (weights, weight_tail) = weights[..len].as_chunks::<32>();
    let bias = _mm256_set1_epi8(i8::MIN);
    let ones = _mm256_set1_epi16(1);
    let mut acc = _mm256_setzero_si256();
    for (x, w) in input.iter().zip(weights) {
        // SAFETY: `x` and `w` each refer to 32 initialized bytes, and an
        // unaligned load has no alignment requirement.
        let (x, w) = unsafe {
            (
                _mm256_loadu_si256(x.as_ptr().cast()),
                _mm256_loadu_si256(w.as_ptr().cast()),
            )
        };
        // Flipping the top bit subtracts 128 and reinterprets as signed.
        let x = _mm256_xor_si256(x, bias);
        // `abs(x) * sign(w, x)` equals `x * w` per byte. The magnitude of
        // -128 is 128, which `VPMADDUBSW` reads as unsigned.
        let pairs = _mm256_maddubs_epi16(_mm256_abs_epi8(x), _mm256_sign_epi8(w, x));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(pairs, ones));
    }
    let tail = input_tail
        .iter()
        .zip(weight_tail)
        .fold(0i32, |sum, (&x, &w)| {
            sum.wrapping_add((i32::from(x) - 128) * i32::from(w))
        });
    sum_epi32(acc).wrapping_add(tail)
}

/// Returns the wrapping sum of the eight 32-bit lanes of `vector`.
#[target_feature(enable = "avx2")]
#[inline]
fn sum_epi32(vector: __m256i) -> i32 {
    let halves = _mm_add_epi32(
        _mm256_castsi256_si128(vector),
        _mm256_extracti128_si256::<1>(vector),
    );
    let pairs = _mm_add_epi32(halves, _mm_shuffle_epi32::<0b00_00_11_10>(halves));
    let total = _mm_add_epi32(pairs, _mm_shuffle_epi32::<0b00_00_00_01>(pairs));
    _mm_cvtsi128_si32(total)
}
