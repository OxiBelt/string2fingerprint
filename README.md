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
| RISC-V 64 | Zvdot4a8i (reserved, see the design document), RVV 1.0, scalar |

## Usage

```rust
use string2fingerprint::{Category, Scorer};

// Create one scorer and share it. It is immutable and selects the kernel
// for the running CPU once.
let scorer = Scorer::new();

let scores = scorer.score(b"/search?q=1%27%20OR%20%271%27=%271");
if scores.get(Category::SqlInjection) >= 192 {
    // Act on the request, for example log or block it.
}
for (category, score) in scores.iter() {
    println!("{category}: {score}");
}
```

`Scorer::score` accepts any bytes of any length. It never fails or panics,
allocates no heap memory, and runs in time linear in the input length. A
score is an integer from 0 to 255. It is a model confidence, not a calibrated
probability, so choose thresholds by measuring on your own traffic.

The embedded default model was trained on synthetic data only and has not
been measured on real traffic. The [training tools](tools/train/README.md)
document the data, the measured results, and the limits. Load another model
with `Model::from_bytes` and `Scorer::with_model`.

To score lines of standard input from the command line, or to measure
throughput for each backend that the CPU supports:

```sh
cargo run --release --example score < lines.txt
cargo bench
```

## Status

The library is under construction and is not published. The
[design document](docs/design.md) specifies the input contract, categories,
fingerprint, network, and backend rules.

See [Contributing](CONTRIBUTING.md) for the toolchain, checks, and review process,
and the [Security Policy](SECURITY.md) for private vulnerability reporting.

Licensed under the [Apache License 2.0](LICENSE).
