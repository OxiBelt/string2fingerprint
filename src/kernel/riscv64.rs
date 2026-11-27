//! Kernels for RISC-V 64: RVV 1.0.
//!
//! The pinned toolchain has no stable `v` target feature, vector intrinsics,
//! or run-time feature detection for RISC-V. The kernel is therefore inline
//! assembly that enables `v` for the assembler only, and detection reads the
//! hardware capabilities from the auxiliary vector. A safe implementation can
//! do neither. Each unsafe use states the invariant it relies on.

use core::arch::asm;
use core::ffi::c_ulong;

use super::{Matrix, scalar};
use crate::Backend;
use crate::backend::Verified;

/// The `AT_HWCAP` key of the auxiliary vector, from `<elf.h>`.
const AT_HWCAP: c_ulong = 16;

/// `COMPAT_HWCAP_ISA_V`, `1 << ('V' - 'A')`, from the Linux header
/// `arch/riscv/include/uapi/asm/hwcap.h`.
const HWCAP_ISA_V: c_ulong = 1 << 21;

#[expect(unsafe_code, reason = "declares the C library function `getauxval`")]
// SAFETY: `getauxval` is provided by the C library that `std` links on Linux,
// with this signature. It takes any key, reads process-constant data, and
// returns zero for a key that is absent, so calling it is always sound.
unsafe extern "C" {
    safe fn getauxval(key: c_ulong) -> c_ulong;
}

/// Returns whether the CPU features of a RISC-V `backend` are detected.
///
/// Linux documents the `V` bit of `AT_HWCAP` as the way for a program to
/// learn whether it may execute version 1.0 vector instructions. The kernel
/// clears the bit when vector use is disabled for the process, and a thread
/// cannot turn vector use off again once it is on, so the answer holds for
/// the life of the process.
///
/// Linux has no interface that reports Zvdot4a8i, so that backend is never
/// detected.
pub(super) fn supports(backend: Backend) -> bool {
    match backend {
        Backend::Rvv => getauxval(AT_HWCAP) & HWCAP_ISA_V != 0,
        _ => false,
    }
}

/// See [`super::matvec`].
#[expect(
    unsafe_code,
    reason = "calls a kernel that needs CPU features that `Verified` established"
)]
pub(super) fn matvec(backend: Verified, matrix: &Matrix, input: &[u8], out: &mut [i32]) {
    match backend.get() {
        Backend::Rvv => {
            for (row, sum) in matrix.weights().chunks_exact(matrix.cols()).zip(out) {
                // SAFETY: a `Verified` backend exists only after `supports`
                // returned true for it, which detected the `V` extension, or
                // after a caller of its unsafe constructor vouched for it.
                *sum = unsafe { dot_rvv(input, row) };
            }
        }
        _ => scalar::matvec(matrix, input, out),
    }
}

/// Returns the wrapping sum of `input[i] * weights[i]` over the common length
/// of the two slices.
///
/// Each step loads up to `VLEN / 4` bytes, multiplies signed weights by
/// unsigned activations into 16 bits, where `127 * 255` fits, and adds the
/// sign-extended products to 32-bit lanes. The lanes and their final
/// reduction wrap at 32 bits, like the scalar kernel.
///
/// # Safety
///
/// The CPU must implement the `V` extension version 1.0, and the operating
/// system must allow this process to use it.
#[expect(unsafe_code, reason = "vector instructions through inline assembly")]
unsafe fn dot_rvv(input: &[u8], weights: &[i8]) -> i32 {
    let len = input.len().min(weights.len());
    let sum: i64;
    // SAFETY:
    // - The caller guarantees that `V` instructions may execute. `.option
    //   arch, +v` only lets the assembler encode them, and `.option pop`
    //   restores the assembler state.
    // - Each `vle8.v` reads `vl` bytes, and `vsetvli` sets `vl` to at most
    //   the remaining length in `a2`, which starts at `len`. The pointers
    //   advance by `vl`, so every read stays inside the first `len` bytes of
    //   each slice. Byte loads have no alignment requirement. With `len`
    //   zero the loop does not run and no load executes.
    // - `vsetvli zero, zero` keeps `vl` because `e8, m2` and `e16, m4` have
    //   the same ratio of element width to register grouping. The tail policy
    //   is undisturbed there, so a final short step leaves the accumulator
    //   lanes at and beyond `vl` intact.
    // - The block writes no memory and does not use the stack. It changes
    //   `a0` to `a2`, `t0`, the listed vector registers, and `vl` and `vtype`.
    //   The argument registers are declared as written, `clobber_abi("C")`
    //   covers `t0` and every vector register, and `vl` and `vtype` may
    //   change because `preserves_flags` is not set.
    unsafe {
        asm!(
            ".option push",
            ".option arch, +v",
            // Clear the 32-bit accumulator lanes in v16 to v23.
            "vsetvli t0, zero, e32, m8, ta, ma",
            "vmv.v.i v16, 0",
            "beqz a2, 3f",
            "2:",
            "vsetvli t0, a2, e8, m2, ta, ma",
            "vle8.v v2, (a0)",
            "vle8.v v4, (a1)",
            // v8 to v11: signed weights times unsigned activations, 16 bits.
            "vwmulsu.vv v8, v4, v2",
            "vsetvli zero, zero, e16, m4, tu, ma",
            "vwadd.wv v16, v16, v8",
            "add a0, a0, t0",
            "add a1, a1, t0",
            "sub a2, a2, t0",
            "bnez a2, 2b",
            "3:",
            "vsetvli t0, zero, e32, m8, ta, ma",
            "vmv.s.x v24, zero",
            "vredsum.vs v24, v16, v24",
            "vmv.x.s a2, v24",
            ".option pop",
            inout("a0") input.as_ptr() => _,
            inout("a1") weights.as_ptr() => _,
            inout("a2") len => sum,
            clobber_abi("C"),
            options(nostack, readonly),
        );
    }
    // `vmv.x.s` sign-extends the 32-bit lane to the register width.
    sum as i32
}
