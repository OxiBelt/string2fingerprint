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
use string2fingerprint::{Category, CategorySet, Scorer};

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

## Training Your Own Model

The `train` feature adds `string2fingerprint::train`, which trains a model
on your own labeled samples without any other tool or dependency:

```toml
[dependencies]
string2fingerprint = { git = "https://github.com/OxiBelt/string2fingerprint", features = ["train"] }
```

```rust
use string2fingerprint::train::{Dataset, Sample, TrainConfig, Trainer};
use string2fingerprint::{Category, CategorySet, Scorer};

let mut data = Dataset::new();
// One benign sample on each line.
data.extend_from_lines(&std::fs::read("benign.txt")?, CategorySet::EMPTY)?;
// Labels, a tab, and a payload on each line, such as
// `sql_injection,obfuscation<TAB>1%2527 OR 1=1`.
data.extend_from_labeled_lines(&std::fs::read("attacks.tsv")?)?;
data.push(Sample::new(b"id=1' OR '1'='1", Category::SqlInjection))?;

// Hold some samples back to measure the model on text it did not see.
let (train, held_out) = data.split(0.1, 1);
let trained = Trainer::new(TrainConfig::default())?.train(&train)?;
std::fs::write("my.s2fp", trained.model().to_bytes())?;

let scorer = Scorer::with_model(trained.into_model());
let evaluation = held_out.evaluate(&scorer);
```

The `train` example trains from the command line, with every setting as an
option and a report of false positives and false negatives on held-out
samples, which it reads from files of their own:

```sh
cargo run --release --features train --example train -- \
  --benign benign.txt --labeled attacks.tsv --test-labeled held_out.tsv --out my.s2fp
```

A sample is the bytes of one input and the categories that apply to it, and
benign text has none. Training is deterministic: the same samples and
settings give the same model bytes on every supported host and with any
number of threads. `Model::from_bytes` loads the file again.

`TrainConfig::costs` weighs the two kinds of error. A higher cost of false
positives, or a higher weight of benign text, gives a model that flags less
benign text and misses more attacks, and a higher cost of false negatives
does the opposite. `Sample::weighted` does the same for single samples:

```rust
use string2fingerprint::train::Costs;

let mut config = TrainConfig::default();
// Wrong flags on benign text cost four times as much, and a missed
// command injection twice as much.
config.costs = Costs::UNIFORM
    .benign(4.0)
    .false_negative(Category::CommandInjection, 2.0);
```

`Trainer::fine_tune` continues the training of an existing model, such as an
embedded one, on your corrections: benign text that it flags and attacks that
it misses.

```rust
use string2fingerprint::model::Model;

let mut corrections = Dataset::new();
corrections.extend_from_lines(&std::fs::read("flagged_benign.txt")?, CategorySet::EMPTY)?;
// Samples whose scores must stay, above all attacks.
corrections.extend_from_corpus(&std::fs::read("replay.bin")?)?;

let base = Model::embedded();
let tuned = Trainer::new(TrainConfig::fine_tuning())?.fine_tune(&base, &corrections)?;
```

Benign corrections alone make the model miss attacks that look like them.
Tune together with samples whose scores must stay, and compare the tuned
model with the base on held-out samples. The training tools
[measured both cases](tools/train/README.md#correcting-a-model).

### Training While Compiling

The `build` feature trains a model in the build script of your crate, from
data files in the crate, and the crate embeds it. The model is trained again
whenever a data file changes, and the compiled crate contains no training
code. In `Cargo.toml`:

```toml
[dependencies]
string2fingerprint = { git = "https://github.com/OxiBelt/string2fingerprint" }

[build-dependencies]
string2fingerprint = { git = "https://github.com/OxiBelt/string2fingerprint", features = ["build"] }

# Cargo compiles build dependencies without optimization by default, which
# makes training about 20 times slower.
[profile.dev.package.string2fingerprint]
opt-level = 3

[profile.release.package.string2fingerprint]
opt-level = 3
```

The two dependencies are built apart with feature resolver 2 or later, the
default from edition 2021 on; a virtual workspace needs `resolver = "2"`.
Cargo reads profiles only from the root manifest of the workspace that is
built, so when your crate is a workspace member or a dependency of another
crate, the profile belongs in that root manifest.

In `build.rs`:

```rust
use string2fingerprint::build::{ModelBuild, Source};
use string2fingerprint::eval::Objective;
use string2fingerprint::{Threshold, Thresholds};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    ModelBuild::new("waf")
        .train_on(Source::benign_lines("data/benign.txt"))
        .train_on(Source::labeled_lines("data/attacks.tsv"))
        // Choose thresholds on a tenth of the samples, held out of training.
        .hold_out(0.1)
        .calibrate(
            Objective::MaxFalsePositiveRate(0.001),
            Thresholds::uniform(Threshold::at(192)),
        )
        .run()?;
    Ok(())
}
```

In the crate:

```rust
use string2fingerprint::{Scorer, Thresholds};

const THRESHOLDS: Thresholds = match string2fingerprint::include_thresholds!("waf") {
    Some(thresholds) => thresholds,
    None => panic!("the build script wrote valid thresholds"),
};

let model = string2fingerprint::include_model!("waf")?;
let flagged = Scorer::with_model(model).score(input).flagged(&THRESHOLDS);
```

`ModelBuild::run` returns the evaluation on the held-out samples, so that
the build script can fail the build when the model is not good enough. Set
`STRING2FINGERPRINT_MODEL_WAF` to the path of a model file, relative to the
package directory, to build with that model in place of training. The build
then prints a warning that names the file. `tests/downstream/` is a complete
example, and
the [design document](docs/design.md#training-while-compiling) specifies the
build.

A model is only as good as its data. The embedded models were trained on
600,000 generated samples, and a model trained on a few hundred samples knows
those samples. The [design document](docs/design.md#training) specifies the
procedure.

## Status

The library is under construction and is not published. The
[design document](docs/design.md) specifies the input contract, categories,
fingerprint, network, and backend rules.

See [Contributing](CONTRIBUTING.md) for the toolchain, checks, and review process,
and the [Security Policy](SECURITY.md) for private vulnerability reporting.

Licensed under the [Apache License 2.0](LICENSE).
