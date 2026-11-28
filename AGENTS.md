# AGENTS.md

## Project Overview

`string2fingerprint` is a Linux-only Rust library that scores untrusted text
per category for how malformed it looks or whether it attempts to escape its
context. It fingerprints the input and runs a small 8-bit quantized network
whose dot-product kernel is selected at run time from the CPU features.
`docs/design.md` specifies the input contract, fingerprint, network, and
backend rules. Change that document together with the behavior it describes.

## Repository Structure

- `Cargo.toml` and `rust-toolchain.toml` hold package metadata, lints, and the
  pinned Rust toolchain.
- `src/` contains the library source.
- `models/` contains the embedded default model and its golden vectors.
- `examples/` contains command-line tools that score input, convert a
  training corpus into fingerprints, train a model with the `train` feature,
  and check the kernel backends.
- `tools/train/` contains the corpus generator and the training tools.
- `docs/` contains the design document.
- `tests/` contains Cargo integration tests, with their data in `tests/data/`.
- `tests/downstream/` is a crate of its own that depends on the library and
  trains its model in its build script with the `build` feature.
- `benches/` contains the dependency-free throughput benchmark.
- `tests/scripts/` contains the Markdown checker and its tests.
- `.github/workflows/` contains the CI workflows.
- `README.md` gives the project status and contributor entry points.
- `SECURITY.md` defines the security scope and private reporting process.

## Contributor Guidance

`CONTRIBUTING.md` is the source of truth for contributor workflow, security
requirements, pull request checks, and commit-message format. Use these
sections before making or reviewing changes:

- [Contribution Workflow](CONTRIBUTING.md#contribution-workflow)
- [Commit Messages](CONTRIBUTING.md#commit-messages)
- [Security Requirements](CONTRIBUTING.md#security-requirements)
- [Independent Review](CONTRIBUTING.md#independent-review)
- [Pull Request Checklist](CONTRIBUTING.md#pull-request-checklist)

For substantial work, use the agent host's native delegation lifecycle for
bounded tasks and independent review. Report each agent's role, actual model
identity when available, and the checks it ran. The coordinating agent owns
integration and verification.

## Agent Commit Message Guidance

Keep commit messages portable and repository-relevant. Use repository-relative
paths and describe the commands, results, and limits a reviewer can verify.
Leave host-specific paths, session identifiers, secrets, and private evidence
out of public contributions. Include the assistance trailers required by
[Commit Messages](CONTRIBUTING.md#commit-messages).

## Codex Security Advisory Guidance

Follow the [Security Policy](SECURITY.md) for validated, report-worthy findings
from Codex Security. Keep each finding in its own private report or advisory
until a fix or mitigation is verified and disclosure is authorized. Verify the
repository, affected revision, duplicate status, permissions, and visibility.
Obtain explicit approval for the exact payload before each submission or
publication, then read back the result. Request a CVE only when explicitly
authorized.
