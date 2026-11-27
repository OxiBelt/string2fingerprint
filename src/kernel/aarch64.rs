//! Kernels for AArch64: SVE with I8MM, NEON with I8MM, NEON with DotProd, and
//! NEON.
//!
//! Unsafe code in this module has four purposes, and a safe implementation
//! cannot serve any of them: loading vectors through pointers, calling a
//! function compiled for CPU features that the build does not enable
//! globally, running `USDOT` and SVE instructions, which have no stable
//! intrinsics, through inline assembly, and declaring `getauxval`. Each use
//! states the invariant it relies on.

use core::arch::aarch64::{
    vaddvq_s32, vdotq_s32, vdupq_n_s32, vdupq_n_u8, veorq_u8, vget_low_s8, vld1q_s8, vld1q_u8,
    vmlal_high_s8, vmull_s8, vpadalq_s16, vreinterpretq_s8_u8,
};
use core::arch::asm;
use core::ffi::c_ulong;
use std::arch::is_aarch64_feature_detected;

use super::{Matrix, scalar};
use crate::Backend;
use crate::backend::Verified;

/// The `AT_HWCAP2` key of the auxiliary vector, from Linux
/// `include/uapi/linux/auxvec.h`.
const AT_HWCAP2: c_ulong = 26;

/// The `AT_HWCAP2` bit for SVE `USDOT`, from Linux
/// `arch/arm64/include/uapi/asm/hwcap.h`.
///
/// Linux `Documentation/arch/arm64/elf_hwcaps.rst` sets it for
/// `ID_AA64PFR0_EL1.SVE == 0b0001` and `ID_AA64ZFR0_EL1.I8MM == 0b0001`,
/// separately from `HWCAP2_I8MM` (bit 13) for the NEON instructions, which
/// is the only bit that `std` checks for `"i8mm"`.
const HWCAP2_SVEI8MM: c_ulong = 1 << 9;

#[expect(unsafe_code, reason = "declares a C library function")]
// SAFETY: glibc and musl define `getauxval` with this signature. It has no
// preconditions: it reads the auxiliary vector that the C library saved at
// startup and returns 0 for a key that is absent.
unsafe extern "C" {
    safe fn getauxval(key: c_ulong) -> c_ulong;
}

/// Returns whether the CPU features of an AArch64 `backend` are all detected.
///
/// Each arm lists exactly the features that the matching kernel enables,
/// including `neon` and `fp16`, which `sve` implies. Detection reads the
/// hardware capabilities that Linux reports, so it also confirms that the
/// kernel enabled the features for this process.
pub(super) fn supports(backend: Backend) -> bool {
    match backend {
        Backend::SveI8mm => {
            // The assembler accepts SVE `USDOT` under `sve` and `i8mm`, but
            // `i8mm` reports only the NEON instructions, so SVE `USDOT` needs
            // its own bit. Each check is kept even where Linux implies it.
            is_aarch64_feature_detected!("neon")
                && is_aarch64_feature_detected!("fp16")
                && is_aarch64_feature_detected!("sve")
                && is_aarch64_feature_detected!("i8mm")
                && getauxval(AT_HWCAP2) & HWCAP2_SVEI8MM != 0
        }
        Backend::NeonI8mm => {
            is_aarch64_feature_detected!("neon") && is_aarch64_feature_detected!("i8mm")
        }
        Backend::NeonDotProd => {
            is_aarch64_feature_detected!("neon") && is_aarch64_feature_detected!("dotprod")
        }
        Backend::Neon => is_aarch64_feature_detected!("neon"),
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
        // true for it, which detected `neon`, `fp16`, `sve`, `i8mm`, and SVE
        // `USDOT`, or after a caller of its unsafe constructor vouched for the
        // same features.
        Backend::SveI8mm => unsafe { matvec_sve_i8mm(matrix, input, out) },
        // SAFETY: as above, for `neon` and `i8mm`.
        Backend::NeonI8mm => unsafe { matvec_neon_i8mm(matrix, input, out) },
        // SAFETY: as above, for `neon` and `dotprod`.
        Backend::NeonDotProd => unsafe { matvec_neon_dotprod(matrix, input, out) },
        // SAFETY: as above, for `neon`.
        Backend::Neon => unsafe { matvec_neon(matrix, input, out) },
        _ => scalar::matvec(matrix, input, out),
    }
}

#[target_feature(enable = "sve,i8mm")]
fn matvec_sve_i8mm(matrix: &Matrix, input: &[u8], out: &mut [i32]) {
    for (row, sum) in matrix.weights().chunks_exact(matrix.cols()).zip(out) {
        *sum = dot_sve_i8mm(input, row);
    }
}

/// Returns the wrapping sum of `input[i] * weights[i]` over the common length
/// of the two slices, one vector length of bytes per SVE `USDOT`.
///
/// The loop is predicated by `WHILELO`, so it handles every vector length
/// from 128 to 2048 bits and every slice length, including zero, without a
/// separate tail.
#[target_feature(enable = "sve,i8mm")]
#[inline]
#[expect(
    unsafe_code,
    reason = "SVE instructions have no stable intrinsics or vector types"
)]
fn dot_sve_i8mm(input: &[u8], weights: &[i8]) -> i32 {
    let len = input.len().min(weights.len());
    let sum: i32;
    // SAFETY: the enclosing target features cover every instruction, and
    // `supports` confirmed SVE `USDOT` separately from NEON I8MM. Rust code
    // runs outside SME streaming mode, so the instructions use the SVE
    // vector length.
    //
    // Byte `k` of an iteration is active only when `index + k < len`, and
    // `len` is at most the length of either slice. Contiguous loads access
    // only active bytes, inactive bytes neither fault nor read memory, and
    // `/z` zeroes them so they add nothing to the sum. Byte loads have no
    // alignment requirement. `index` grows by the vector length and stays
    // below `len + 256`, so it cannot wrap.
    //
    // The block writes `z0` to `z2`, which alias `v0` to `v2`, `p0`, and
    // the condition flags, and declares all of them. It does not touch the
    // first-fault register, the stack, or the vector length.
    unsafe {
        asm!(
            "mov z0.s, #0",
            "whilelo p0.b, {index}, {len}",
            "b.none 3f",
            "2:",
            "ld1b {{ z1.b }}, p0/z, [{input}, {index}]",
            "ld1b {{ z2.b }}, p0/z, [{weights}, {index}]",
            "usdot z0.s, z1.b, z2.b",
            "incb {index}",
            "whilelo p0.b, {index}, {len}",
            "b.first 2b",
            "3:",
            // The low 32 bits of the 64-bit total are the wrapping sum of
            // the 32-bit lanes.
            "ptrue p0.s",
            "uaddv d0, p0, z0.s",
            "fmov {sum:w}, s0",
            input = in(reg) input.as_ptr(),
            weights = in(reg) weights.as_ptr(),
            len = in(reg) len,
            index = inout(reg) 0usize => _,
            sum = lateout(reg) sum,
            out("z0") _,
            out("z1") _,
            out("z2") _,
            out("p0") _,
            options(pure, readonly, nostack),
        );
    }
    sum
}

#[target_feature(enable = "neon,i8mm")]
fn matvec_neon_i8mm(matrix: &Matrix, input: &[u8], out: &mut [i32]) {
    for (row, sum) in matrix.weights().chunks_exact(matrix.cols()).zip(out) {
        *sum = dot_neon_i8mm(input, row);
    }
}

/// Returns the wrapping sum of `input[i] * weights[i]` over the common length
/// of the two slices, 16 bytes per NEON `USDOT`.
#[target_feature(enable = "neon,i8mm")]
#[inline]
#[expect(
    unsafe_code,
    reason = "loads vectors through pointers, and `USDOT` has no stable intrinsic"
)]
fn dot_neon_i8mm(input: &[u8], weights: &[i8]) -> i32 {
    let len = input.len().min(weights.len());
    let (input, input_tail) = input[..len].as_chunks::<16>();
    let (weights, weight_tail) = weights[..len].as_chunks::<16>();
    let mut acc = vdupq_n_s32(0);
    for (x, w) in input.iter().zip(weights) {
        // SAFETY: `x` and `w` each refer to 16 initialized bytes, and these
        // loads have no alignment requirement.
        let (x, w) = unsafe { (vld1q_u8(x.as_ptr()), vld1q_s8(w.as_ptr())) };
        // SAFETY: the enclosing target features cover `USDOT`, which reads
        // and writes only its three vector operands.
        unsafe {
            asm!(
                "usdot {acc:v}.4s, {x:v}.16b, {w:v}.16b",
                acc = inout(vreg) acc,
                x = in(vreg) x,
                w = in(vreg) w,
                options(pure, nomem, nostack, preserves_flags),
            );
        }
    }
    vaddvq_s32(acc).wrapping_add(scalar::dot(input_tail, weight_tail))
}

#[target_feature(enable = "neon,dotprod")]
fn matvec_neon_dotprod(matrix: &Matrix, input: &[u8], out: &mut [i32]) {
    let rows = matrix.weights().chunks_exact(matrix.cols());
    for ((row, &row_sum), sum) in rows.zip(matrix.row_sums()).zip(out) {
        // `sum((x - 128) * w) + 128 * sum(w)` equals `sum(x * w)`.
        *sum = dot_biased_neon_dotprod(input, row).wrapping_add(row_sum.wrapping_mul(128));
    }
}

/// Returns the wrapping sum of `(input[i] - 128) * weights[i]` over the
/// common length of the two slices, 16 bytes per `SDOT`.
///
/// `SDOT` multiplies signed bytes only, so the activations are biased to
/// signed bytes first.
#[target_feature(enable = "neon,dotprod")]
#[inline]
#[expect(unsafe_code, reason = "loads vectors through pointers")]
fn dot_biased_neon_dotprod(input: &[u8], weights: &[i8]) -> i32 {
    let len = input.len().min(weights.len());
    let (input, input_tail) = input[..len].as_chunks::<16>();
    let (weights, weight_tail) = weights[..len].as_chunks::<16>();
    let bias = vdupq_n_u8(0x80);
    let mut acc = vdupq_n_s32(0);
    for (x, w) in input.iter().zip(weights) {
        // SAFETY: `x` and `w` each refer to 16 initialized bytes, and these
        // loads have no alignment requirement.
        let (x, w) = unsafe { (vld1q_u8(x.as_ptr()), vld1q_s8(w.as_ptr())) };
        // Flipping the top bit subtracts 128 and reinterprets as signed.
        let x = vreinterpretq_s8_u8(veorq_u8(x, bias));
        acc = vdotq_s32(acc, x, w);
    }
    vaddvq_s32(acc).wrapping_add(dot_biased(input_tail, weight_tail))
}

#[target_feature(enable = "neon")]
fn matvec_neon(matrix: &Matrix, input: &[u8], out: &mut [i32]) {
    let rows = matrix.weights().chunks_exact(matrix.cols());
    for ((row, &row_sum), sum) in rows.zip(matrix.row_sums()).zip(out) {
        // `sum((x - 128) * w) + 128 * sum(w)` equals `sum(x * w)`.
        *sum = dot_biased_neon(input, row).wrapping_add(row_sum.wrapping_mul(128));
    }
}

/// Returns the wrapping sum of `(input[i] - 128) * weights[i]` over the
/// common length of the two slices, 16 bytes per `SMULL`, `SMLAL2`, and
/// `SADALP`.
///
/// NEON without DotProd multiplies bytes of one signedness only. Biasing the
/// activations to signed bytes bounds each product by `128 * 127`, so the two
/// products that `SMLAL2` adds into a signed 16-bit lane sum to at most 32512
/// in magnitude and never wrap. This needs every weight above -128, which
/// [`Matrix`] guarantees.
#[target_feature(enable = "neon")]
#[inline]
#[expect(unsafe_code, reason = "loads vectors through pointers")]
fn dot_biased_neon(input: &[u8], weights: &[i8]) -> i32 {
    let len = input.len().min(weights.len());
    let (input, input_tail) = input[..len].as_chunks::<16>();
    let (weights, weight_tail) = weights[..len].as_chunks::<16>();
    let bias = vdupq_n_u8(0x80);
    let mut acc = vdupq_n_s32(0);
    for (x, w) in input.iter().zip(weights) {
        // SAFETY: `x` and `w` each refer to 16 initialized bytes, and these
        // loads have no alignment requirement.
        let (x, w) = unsafe { (vld1q_u8(x.as_ptr()), vld1q_s8(w.as_ptr())) };
        // Flipping the top bit subtracts 128 and reinterprets as signed.
        let x = vreinterpretq_s8_u8(veorq_u8(x, bias));
        let pairs = vmlal_high_s8(vmull_s8(vget_low_s8(x), vget_low_s8(w)), x, w);
        acc = vpadalq_s16(acc, pairs);
    }
    vaddvq_s32(acc).wrapping_add(dot_biased(input_tail, weight_tail))
}

/// Returns the wrapping sum of `(input[i] - 128) * weights[i]` over the
/// common length of the two slices.
fn dot_biased(input: &[u8], weights: &[i8]) -> i32 {
    input.iter().zip(weights).fold(0i32, |sum, (&x, &w)| {
        sum.wrapping_add((i32::from(x) - 128) * i32::from(w))
    })
}
