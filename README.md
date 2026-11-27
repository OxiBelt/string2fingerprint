# string2fingerprint

`string2fingerprint` is a Linux-only Rust library that scores untrusted text,
such as the request fields seen by a web reverse proxy. It reduces the input
to fixed-size fingerprints, runs them through a small 8-bit quantized neural
network, and returns one score per category for how malformed the input looks
or whether it attempts to escape its context.

The dot-product kernel is selected at run time from the CPU features that are
present, in this order:

| Architecture | Order |
| --- | --- |
| x86-64 | AVX-512 VNNI, AVX-VNNI, AVX2, scalar |
| AArch64 | SVE with I8MM, NEON with I8MM, NEON with DotProd, NEON, scalar |
| RISC-V 64 | Zvdot4a8i, RVV 1.0, scalar |

The library is under construction and is not published. The
[design document](docs/design.md) specifies the input contract, categories,
fingerprint, network, and backend rules.

See [Contributing](CONTRIBUTING.md) for the toolchain, checks, and review process,
and the [Security Policy](SECURITY.md) for private vulnerability reporting.

Licensed under the [Apache License 2.0](LICENSE).
