# Contributing to string2fingerprint

Thanks for helping improve `string2fingerprint`. This project is a Linux-only
Rust library that scores untrusted text with SIMD-accelerated fingerprinting. Changes
that define how caller-controlled input is interpreted or how CPU-specific
instructions are selected require focused tests and independent review.

Use repository-relative paths in documentation, commands, issues, and pull
request notes. Run the commands in this guide from the repository root.

## Repository Layout

| Path | Purpose | Change here when |
| --- | --- | --- |
| `Cargo.toml`, `Cargo.lock`, `rust-toolchain.toml` | Package metadata, dependency resolution, lints, and toolchain. | The crate, toolchain, lint policy, or dependencies change. |
| `src/` | The Rust library. | Library code or crate-level documentation changes. |
| `docs/` | The design document. | A contract or the specified behavior changes. |
| `tests/downstream/` | A dependent crate that trains its model while it is compiled. | The `build` feature or the macros that embed a built model change. |
| `tests/docker/` | The image and script of the end-to-end check in a container. | A program, its options, or its output change, or the pinned toolchain moves. |
| `tests/scripts/` | Markdown checker and regression tests. | Documentation validation changes. |
| `.github/workflows/` | CI workflows. | Runners or CI checks change. |
| `README.md` | Project status and contributor entry points. | Setup, usage, or the project overview changes. |
| `SECURITY.md` | Security scope, requirements, and private reporting. | The threat model, reporting process, or scope changes. |

## Contribution Workflow

1. Identify the affected responsibility and the behavior or contract it owns.
2. Make the smallest reasonable change. Preserve documented behavior unless
   the contribution explicitly changes it.
3. Add or update regression tests when behavior changes. Exercise the normal
   entry point and the affected implementation paths.
4. Update documentation for user-visible behavior, compatibility, or changed
   development commands. Use Rust documentation for API contracts, the README
   for setup and usage, and `docs/` for technical specifications as they are
   introduced.
5. Run the relevant checks. Record the scope, results, and any checks that
   could not run. Clean up generated test data.

Use `rustup` and the toolchain pinned in `rust-toolchain.toml`:

```sh
cargo fmt --all --check
cargo build --all-targets --all-features --locked
cargo test --all-features --locked
cargo clippy --all-targets --all-features --locked -- -D warnings
cargo clippy --all-targets --locked -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc --all-features --no-deps --locked
```

The second Clippy run checks the build without optional features, such as
`train`. The crate in `tests/downstream/` checks the `build` feature from a
dependent crate, with a lock file of its own:

```sh
cargo test --locked --manifest-path tests/downstream/Cargo.toml
cargo clippy --all-targets --locked --manifest-path tests/downstream/Cargo.toml -- -D warnings
```

The end-to-end check builds the image of `tests/docker/Dockerfile` from the
working tree and runs the dependent crate and the examples in containers
without a network, as a user without privileges, on a read-only file system.
It needs Docker and Python 3. The Rust image provides the toolchain that
`rust-toolchain.toml` pins, and the script fails when the two differ, so move
both together:

```sh
tests/docker/e2e.sh
```

Use Python 3 for documentation checks and for the tests of the training
tools. The tests that need NumPy are skipped without it:

```sh
python3 -B -m unittest discover -s tests/scripts -p 'test_*.py'
python3 -B -m unittest discover -s tools/train -p 'test_*.py'
python3 -B tests/scripts/check_markdown_links.py --repo-root .
```

The link checker reads Git-tracked Markdown. Include new documents in the Git
index for a complete check. CI uses the runners' preinstalled `rustup` to
install the pinned toolchain. Native builds and tests use Ubuntu 26.04 on
x86-64 and Arm64, and `ubuntu-24.04-riscv` on RISC-V. The RISC-V job requires
access to the [RISE runner service](https://ci.riseproject.dev/docs/getting-started/configure).
These jobs validate the scaffold on the selected architectures.

For dependency additions, record the need, alternatives, affected features,
license and maintenance status, and relevant source or unsafe-code review.
Review the lockfile delta. Add automated dependency admission and release
checks as the relevant dependency or release surfaces develop. Describe public
compatibility changes and any migration steps alongside the affected contract.

## Continuous Integration

Each workflow in `.github/workflows/` has one purpose:

| Workflow | Checks | Toolchain |
| --- | --- | --- |
| `check.yml` | Documentation, formatting, Clippy, Rust documentation, and native builds and tests on x86-64, Arm64, and RISC-V. | Pinned |
| `toolchains.yml` | Builds and tests with the minimum supported version, stable, beta, and nightly, and each feature on its own. | Each named one |
| `targets.yml` | Clippy for other Linux targets, tests on big-endian s390x under QEMU and on 32-bit x86, and the refusal of other systems. | Pinned |
| `docker-e2e.yml` | `tests/docker/e2e.sh` on x86-64 and Arm64, and that both give the same scores and models. | Pinned, in the Rust image |
| `sanitizers.yml` | The tests and the kernel self-test under AddressSanitizer on x86-64 and Arm64. | A nightly of a pinned date |

`check.yml` gates every change on the pinned toolchain, so its results do not
depend on the day they ran. `toolchains.yml` also runs weekly, because new
releases change without a change to the repository. Its Clippy runs report
warnings as annotations without failing, and its nightly jobs never fail it,
because a new lint or a broken nightly release is fixed when the pinned
toolchain moves, not in an unrelated change. A failing stable build or test
fails it, and so does a failing beta, except on a pull request.

The minimum supported version is the `rust-version` in `Cargo.toml`. While it
equals the pinned toolchain, `check.yml` covers it and the MSRV job only checks
that. To reproduce a toolchain job locally, set `RUSTUP_TOOLCHAIN`, which takes
precedence over `rust-toolchain.toml`:

```sh
rustup toolchain install beta --profile minimal --component clippy
RUSTUP_TOOLCHAIN=beta cargo test --all-features --locked
cargo test --locked --features train
```

AddressSanitizer needs a nightly toolchain with `rust-src`, and checks the
memory accesses of every kernel that the CPU supports:

```sh
rustup toolchain install nightly-2026-11-28 --profile minimal --component rust-src
RUSTUP_TOOLCHAIN=nightly-2026-11-28 RUSTFLAGS=-Zsanitizer=address \
  cargo test -Zbuild-std --target x86_64-unknown-linux-gnu --all-features --locked --lib --tests
```

The s390x tests need `qemu-user`, `gcc-s390x-linux-gnu`, and
`libc6-dev-s390x-cross`, and the i686 tests need `gcc-multilib`. On Ubuntu,
the two sets of packages cannot be installed together:

```sh
rustup target add s390x-unknown-linux-gnu
CARGO_TARGET_S390X_UNKNOWN_LINUX_GNU_LINKER=s390x-linux-gnu-gcc \
CARGO_TARGET_S390X_UNKNOWN_LINUX_GNU_RUNNER='qemu-s390x -L /usr/s390x-linux-gnu' \
  cargo test --target s390x-unknown-linux-gnu --all-features --locked
```

## Commit Messages

Sign commits and use Conventional Commits in the form
`<type>(<scope>): <subject>`. Use `feat`, `fix`, `chore`, `docs`, `ci`,
`refactor`, `security`, `tests`, or `perf` for the type. Use the affected
responsibility as the scope and a short imperative subject. Format identifiers,
commands, paths, and literal values as inline code in titles and descriptions.

```text
docs(governance): document independent review requirements
ci(repository): validate tracked Markdown links
```

Add `Assisted-by: Codex <codex@openai.com>` to commits that used Codex. Use
`Assisted-model:` trailers for additional models that assisted, with the model
identity as reported by the host when available. Include a visible assistance
notice in assisted issues, pull requests, reviews, and comments. Keep private
report details and local-only evidence out of public messages.

## Rust Module Organization

Keep each module focused on one responsibility. Add a module when new code
owns a distinct responsibility and make its interfaces explicit. Keep
fingerprinting and input interpretation free of unrelated filesystem, network,
or process state. Isolate architecture-specific code and dispatch when those
are introduced. Avoid allocations, copies, and dynamic dispatch added solely
to cross a module boundary on a hot path.

## Tests and Temporary Data

Keep unit tests near the code they exercise and Cargo integration tests under
`tests/`. Use unique temporary directories for generated fixtures and clean
them up even when tests fail. Keep `target/`, machine-specific data, logs, and
temporary files out of commits.

As the input contract develops, test empty and short inputs, malformed cases,
lengths around relevant vector boundaries, and documented rejection or error
behavior. Compare results where implementations share a contract. For SIMD
changes, record the toolchain, target, CPU features, and paths actually
executed, including any paths skipped for missing capabilities. Run focused
checks on hardware that can execute affected instructions. Use available
sanitizers, Miri, and fuzzing where they exercise the affected code, and state
their coverage limits.

## Security Requirements

Treat caller-controlled input as untrusted. Changes to interpretation,
fingerprints, detection outcomes, or failure behavior must describe the
affected contract and include regression coverage. Bound input-driven work and
allocation according to the documented contract. Avoid input-triggered panics
and avoid including caller data or secrets in diagnostics.

Keep first-party unsafe Rust denied by default. A future exception must be
local to the code that needs it and explain why a safe implementation is
insufficient. Each unsafe operation requires a `SAFETY` comment. Establish the
full CPU-feature preconditions before executing target-specific instructions.
Document memory bounds, alignment, initialization, and lifetime assumptions as
applicable. Enable no specialized CPU features globally in default builds.
Define the behavior for unsupported capabilities with the relevant interface.

See the [Security Policy](SECURITY.md) for private reporting.

## Independent Review

Require approval from a named non-author person or a separate reviewing agent
for substantive changes to unsafe Rust or its allowlist, CPU-feature
detection or dispatch, untrusted-input handling, or fingerprint and detection
semantics. The reviewer must inspect the actual patch and affected callers,
check the safety or behavior argument, and identify any evidence gaps. Record
the reviewer and approval in the contribution record. Hold landing when an
independent reviewer is unavailable.

Use the agent host's native delegation lifecycle when assigning an agent.
Give the agent a bounded scope and ask for findings with file references,
checks run, remaining uncertainty, and the actual model identity if available.
The coordinating agent integrates the results and verifies the final change.

## Pull Request Checklist

- Commits are signed, follow the documented format, and include applicable
  assistance trailers.
- The affected area, intended behavior, and compatibility impact are clear.
- Behavior changes include regression coverage and any required documentation.
- Relevant checks were run, skipped checks are explained, and temporary data
  was removed.
- Security-sensitive changes describe untrusted inputs, failure behavior,
  affected safety preconditions, and remaining risks.
- Changes subject to independent review record a named reviewer and approval.
