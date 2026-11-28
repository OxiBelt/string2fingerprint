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
| RISC-V 64 | Zvdot4a8i (experimental and never detected, see the design document), RVV 1.0, scalar |

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

To decide with a threshold for each category, and to turn a category off:

```rust
use string2fingerprint::{Category, Scorer, Threshold, Thresholds};

let thresholds = Thresholds::uniform(Threshold::at(192))
    .with(Category::CommandInjection, Threshold::at(230))
    .with(Category::Malformed, Threshold::NEVER);

let flagged = Scorer::new().score(b"/files?name=..%2f..%2fetc%2fpasswd").flagged(&thresholds);
if !flagged.is_empty() {
    println!("flagged: {flagged}");
}
```

To choose those thresholds from labeled samples of your own traffic, record
their scores and labels in an `Evaluation` and calibrate each category, here
to the lowest threshold that flags at most 0.1% of the samples without its
label:

```rust
use string2fingerprint::eval::{Evaluation, Objective};
use string2fingerprint::{CategorySet, Scorer, Threshold, Thresholds};

let scorer = Scorer::new();
let samples: &[(&[u8], CategorySet)] = &[]; // Your labeled samples.
let mut evaluation = Evaluation::new();
for &(input, labels) in samples {
    evaluation.record(scorer.score(input), labels);
}
let thresholds = evaluation.thresholds(
    Objective::MaxFalsePositiveRate(0.001),
    Thresholds::uniform(Threshold::at(192)),
);
```

A category without negative samples keeps the fallback threshold. Calibrate
on samples that were not used to train the model, and check the result on
others.

`Scorer::score` accepts any bytes of any length. It never fails or panics,
allocates no heap memory, and runs in time linear in the input length. A
score is an integer from 0 to 255. It is a model confidence, not a calibrated
probability, so choose thresholds by measuring on your own traffic.

The crate embeds four models, which `EmbeddedModel` names:

| Model | Hidden layers | Size | Use |
| --- | --- | --- | --- |
| `compact` | 32 | 97 KiB | The fastest, where every microsecond counts. |
| `default` | 64, 64 | 198 KiB | The balance that `Scorer::new` uses. |
| `large` | 192, 96 | 597 KiB | The best separation, at about three times the cost. |
| `strict` | 64, 64 | 198 KiB | Flags less benign text and misses more attacks, for blocking. |

```rust
use string2fingerprint::Scorer;
use string2fingerprint::model::EmbeddedModel;

let scorer = Scorer::with_embedded(EmbeddedModel::Strict);
```

A program that uses only `Scorer::new` links only the default model. The
embedded models were trained on synthetic data only and have not been
measured on real traffic. The [training tools](tools/train/README.md)
document the data, the measured results of each model, and the limits. Load
another model with `Model::from_bytes` and `Scorer::with_model`.

To score lines of standard input from the command line, or to measure
throughput for each backend that the CPU supports:

```sh
cargo run --release --example score < lines.txt
cargo run --release --example score large < lines.txt
cargo bench
```

## Status

The library is under construction and is not published. The
[design document](docs/design.md) specifies the input contract, categories,
fingerprint, network, and backend rules.

See [Contributing](CONTRIBUTING.md) for the toolchain, checks, and review process,
and the [Security Policy](SECURITY.md) for private vulnerability reporting.

Licensed under the [Apache License 2.0](LICENSE).
