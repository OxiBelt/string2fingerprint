//! Kernels for RISC-V 64: RVV 1.0, and RVV 1.0 with Zvdot4a8i.
//!
//! The pinned toolchain has no stable `v` target feature, vector intrinsics,
//! or run-time feature detection for RISC-V. The kernels are therefore inline
//! assembly that enables `v` for the assembler only, and detection reads the
//! hardware capabilities from the auxiliary vector. A safe implementation can
//! do neither. Each unsafe use states the invariant it relies on.
//!
//! Zvdot4a8i is not ratified. Its kernel follows the encoding and semantics
//! of the draft that the Spike simulator implements, which the documentation
//! of [`Backend::RvvZvdot4a8i`] states, and was executed only there. Linux
//! cannot report the extension, so the kernel is reachable only through the
//! `unsafe` opt-in of the scorer.

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
/// detected. It runs only for a caller that vouches for the CPU.
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
        Backend::RvvZvdot4a8i => {
            for (row, sum) in matrix.weights().chunks_exact(matrix.cols()).zip(out) {
                // SAFETY: detection never reports this backend, so a
                // `Verified` value for it exists only after a caller of the
                // unsafe constructor vouched for `V` and Zvdot4a8i.
                *sum = unsafe { dot_zvdot4a8i(input, row) };
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

/// `vdot4asu.vv v16, v8, v0`, unmasked, from the Zvdot4a8i draft: for each
/// 32-bit element, adds to `v16` the sum of the four products of a signed
/// byte of `v8` and the unsigned byte of `v0` at the same position.
///
/// The fields are `funct6 = 0b101010`, `vm = 1`, `vs2 = 8`, `vs1 = 0`,
/// `funct3 = 0b010`, `vd = 16`, and the `OP-V` opcode `0x57`. The assembler
/// of the pinned toolchain knows the mnemonic but refuses it as
/// experimental, so the word is emitted with `.insn`.
const VDOT4ASU_VV_V16_V8_V0: u32 =
    (0b10_1010 << 26) | (1 << 25) | (8 << 20) | (0b010 << 12) | (16 << 7) | 0x57;

const _: () = assert!(VDOT4ASU_VV_V16_V8_V0 == 0xAA80_2857);

/// Returns the wrapping sum of `input[i] * weights[i]` over the common length
/// of the two slices.
///
/// Each step loads up to `VLEN` bytes and adds, for every group of four
/// bytes, the four products of a signed weight and an unsigned activation to
/// a 32-bit lane with one `vdot4asu.vv`. The scalar kernel handles the last
/// one to three bytes. The lanes and their final reduction wrap at 32 bits,
/// like the scalar kernel.
///
/// # Safety
///
/// The CPU must implement the `V` extension version 1.0 and the Zvdot4a8i
/// extension with the encoding and semantics of [`VDOT4ASU_VV_V16_V8_V0`],
/// and the operating system must allow this process to use vector
/// instructions.
#[expect(unsafe_code, reason = "vector instructions through inline assembly")]
unsafe fn dot_zvdot4a8i(input: &[u8], weights: &[i8]) -> i32 {
    let len = input.len().min(weights.len());
    let (input, input_tail) = input[..len].as_chunks::<4>();
    let (weights, weight_tail) = weights[..len].as_chunks::<4>();
    let sum: i64;
    // SAFETY:
    // - The caller guarantees that `V` and Zvdot4a8i instructions may
    //   execute. `.option arch, +v` only lets the assembler encode the `V`
    //   instructions, and `.option pop` restores the assembler state.
    // - `a2` counts the remaining groups of four bytes and starts at the
    //   number of whole groups in each slice. The first `vsetvli` of a step
    //   sets `t0` to at most `a2` and at most `VLEN / 4`, the largest `vl`
    //   for `e32, m8`. `t1` is four times `t0`, so it is at most `VLEN`, the
    //   largest `vl` for `e8, m8`, and the second `vsetvli` sets `vl` to
    //   exactly `t1`. Each `vle8.v` therefore reads `4 * t0` bytes, the
    //   pointers advance by the same amount, and every read stays inside the
    //   whole groups of each slice. Byte loads have no alignment
    //   requirement. With no whole group the loop does not run and no load
    //   executes.
    // - The third `vsetvli` sets `vl` to exactly `t0`, which is at most the
    //   largest `vl` for `e32, m8`. The dot product then covers the `t0`
    //   elements that the loads filled. Its tail policy is undisturbed, so a
    //   final short step leaves the accumulator lanes at and beyond `vl`
    //   intact. The loaded registers `v0` to `v7` and `v8` to `v15` and the
    //   accumulator `v16` to `v23` are aligned register groups for `m8` and
    //   do not overlap. The instruction is unmasked, so `v0` is not read as
    //   a mask.
    // - The block writes no memory and does not use the stack. It changes
    //   `a0` to `a2`, `t0`, `t1`, the listed vector registers, and `vl` and
    //   `vtype`. The argument registers are declared as written,
    //   `clobber_abi("C")` covers the temporaries and every vector register,
    //   and `vl` and `vtype` may change because `preserves_flags` is not set.
    unsafe {
        asm!(
            ".option push",
            ".option arch, +v",
            // Clear the 32-bit accumulator lanes in v16 to v23.
            "vsetvli t0, zero, e32, m8, ta, ma",
            "vmv.v.i v16, 0",
            "beqz a2, 3f",
            "2:",
            // t0: groups in this step. t1: their bytes.
            "vsetvli t0, a2, e32, m8, ta, ma",
            "slli t1, t0, 2",
            "vsetvli zero, t1, e8, m8, ta, ma",
            "vle8.v v0, (a0)",
            "vle8.v v8, (a1)",
            "vsetvli zero, t0, e32, m8, tu, ma",
            // vdot4asu.vv v16, v8, v0
            ".insn 4, {vdot4asu}",
            "add a0, a0, t1",
            "add a1, a1, t1",
            "sub a2, a2, t0",
            "bnez a2, 2b",
            "3:",
            "vsetvli t0, zero, e32, m8, ta, ma",
            "vmv.s.x v24, zero",
            "vredsum.vs v24, v16, v24",
            "vmv.x.s a2, v24",
            ".option pop",
            vdot4asu = const VDOT4ASU_VV_V16_V8_V0,
            inout("a0") input.as_ptr() => _,
            inout("a1") weights.as_ptr() => _,
            inout("a2") input.len() => sum,
            clobber_abi("C"),
            options(nostack, readonly),
        );
    }
    // `vmv.x.s` sign-extends the 32-bit lane to the register width.
    (sum as i32).wrapping_add(scalar::dot(input_tail, weight_tail))
}
