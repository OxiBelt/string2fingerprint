//! Checks kernel backends against the scalar kernel on the running CPU.
//!
//! Usage: `kernel_self_test [--unchecked BACKEND]`.
//!
//! Without arguments, every backend that detection reports is checked.
//!
//! `--unchecked BACKEND` checks one backend, named as in `Backend::name`,
//! without detecting its CPU features. It exists for backends that the
//! operating system cannot report, such as `rvv_zvdot4a8i`. Running it on a
//! CPU that does not implement the features of the backend is undefined
//! behavior, so use it only on a CPU or a simulator that is known to
//! implement them.
//!
//! A backend is checked in two ways: the self-test of its kernel, and the
//! scores of the embedded model for built-in inputs, which must equal the
//! scores of the scalar backend. The program uses one thread and exits with
//! status 0 when every checked backend passes.

use std::env;
use std::ffi::OsString;
use std::process::ExitCode;

use string2fingerprint::{Backend, Scorer};

/// Input lengths around the block and window sizes of the fingerprint.
const LENGTHS: [usize; 12] = [0, 1, 2, 17, 100, 255, 256, 257, 511, 512, 513, 1500];

/// Returns `len` deterministic bytes that mix text with arbitrary bytes.
fn input(seed: u64, len: usize) -> Vec<u8> {
    const TEXT: &[u8] = b"GET /a/../b?q=%27%20or%201=1 <b onload=x> ${7*7} ;ls|cat \\x00\r\n";
    let mut state = seed;
    (0..len)
        .map(|index| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let random = state.to_be_bytes()[0];
            if random < 48 {
                state.to_be_bytes()[1]
            } else {
                TEXT[(index + usize::from(random)) % TEXT.len()]
            }
        })
        .collect()
}

/// Returns whether `scorer` passes both checks.
fn passes(scorer: &Scorer, reference: &Scorer) -> bool {
    if !scorer.kernel_self_test() {
        println!("{}: FAILED the kernel self-test", scorer.backend());
        return false;
    }
    for seed in 1..=4 {
        for len in LENGTHS {
            let input = input(seed, len);
            if scorer.score(&input) != reference.score(&input) {
                // The seed and length identify the input without echoing it.
                println!(
                    "{}: FAILED, scores differ for seed {seed} and length {len}",
                    scorer.backend()
                );
                return false;
            }
        }
    }
    println!("{}: passed", scorer.backend());
    true
}

/// Checks `backend` without detecting its CPU features.
///
/// # Safety
///
/// The running CPU must implement every CPU feature that the documentation
/// of `backend` lists, and the operating system must have enabled them for
/// this process.
#[expect(unsafe_code, reason = "the user vouches for CPU features")]
unsafe fn passes_unchecked(backend: Backend, reference: &Scorer) -> bool {
    // SAFETY: the caller guarantees the CPU features of `backend`.
    let scorer = unsafe { reference.clone().with_backend_unchecked(backend) };
    passes(&scorer, reference)
}

fn main() -> ExitCode {
    let arguments: Vec<OsString> = env::args_os().skip(1).collect();
    // An argument that is not UTF-8 names no backend and no option.
    let arguments: Vec<&str> = arguments
        .iter()
        .map(|argument| argument.to_str().unwrap_or(""))
        .collect();
    let reference = Scorer::new()
        .with_backend(Backend::Scalar)
        .expect("the scalar backend is always supported");
    let passed = match arguments.as_slice() {
        [] => {
            let mut passed = true;
            for &backend in Backend::PREFERENCE {
                match reference.clone().with_backend(backend) {
                    Ok(scorer) => passed &= passes(&scorer, &reference),
                    Err(_) => println!("{backend}: skipped, not detected"),
                }
            }
            passed
        }
        ["--unchecked", name] => {
            let Some(&backend) = Backend::PREFERENCE
                .iter()
                .find(|backend| backend.name() == *name)
            else {
                eprintln!("kernel_self_test: no backend `{name}` for this architecture");
                return ExitCode::from(2);
            };
            println!("{backend}: selected without detection");
            #[expect(unsafe_code, reason = "the user vouches for CPU features")]
            // SAFETY: `--unchecked` is the user's statement that the CPU
            // implements the features of the backend, as the usage text
            // requires.
            unsafe {
                passes_unchecked(backend, &reference)
            }
        }
        _ => {
            eprintln!("usage: kernel_self_test [--unchecked BACKEND]");
            return ExitCode::from(2);
        }
    };
    if passed {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
