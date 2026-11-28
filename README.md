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

A `Detector` decides an input in one call. It pairs a model with a
threshold for each category and returns the scores with the categories they
flag:

```rust
use string2fingerprint::prelude::*;

// Create one detector and share it. It is immutable and selects the kernel
// for the running CPU once.
let detector = Detector::new();

let detection = detector.detect(b"/search?q=1%27%20OR%20%271%27=%271");
if detection.is_flagged() {
    // Act on the request, for example log or block it.
    println!("flagged: {detection}"); // "flagged: sql_injection"
}
```

A `Preset` pairs each embedded model with thresholds, and the builder
changes any of them:

| Preset | Model | Threshold | Attacks flagged | Benign flagged |
| --- | --- | --- | --- | --- |
| `fast` | `compact` | 128 | 97.2% | 1.24% |
| `balanced` | `default` | 128 | 98.1% | 1.19% |
| `thorough` | `large` | 128 | 98.5% | 1.04% |
| `strict` | `strict` | 192 | 96.7% | 0.29% |

The rates are the shares of attack and benign samples of the synthetic test
split that are flagged in any category, which the
[training tools](tools/train/README.md#every-model) record. Measure on your own traffic before you act on a decision.

```rust
use string2fingerprint::{Category, Detector, Preset, Threshold};

let detector = Detector::builder()
    .preset(Preset::Strict)
    .category(Category::CommandInjection, Threshold::at(230))
    .disable(Category::Malformed)
    .build()?;
```

A `Scorer` returns the scores without deciding them:

```rust
use string2fingerprint::{Category, Scorer};

let scorer = Scorer::new();
let scores = scorer.score(b"/search?q=1%27%20OR%20%271%27=%271");
if scores.get(Category::SqlInjection) >= 192 {
    // Act on the request.
}
for (category, score) in scores.iter() {
    println!("{category}: {score}");
}
```

Both score more than one buffer:

| `Scorer` | `Detector` | Scores |
| --- | --- | --- |
| `score_fields` | `detect_fields` | The fields of one message, each on its own, combined by the highest score. |
| `score_batch`, `score_parallel` | `detect_batch`, `detect_parallel` | Many inputs, on one thread or several. |
| `score_reader` | `detect_reader` | A reader to its end, with a buffer on the stack. Bound an untrusted reader with `Read::take`. |
| `stream` | `stream` | An input that arrives in parts, as a `ScoreStream` or a `DetectionStream`. |
| `windows` | | Each window of a long input, with the bytes it covers. |

A stream gives the same result as the whole input however the input is
split, and its scores so far never decrease, so a proxy can stop reading a
body as soon as it is flagged:

```rust
let mut stream = detector.stream();
for part in body_parts {
    stream.update(part);
    if stream.detection().is_flagged() {
        break;
    }
}
let detection = stream.finish();
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

### Model Versions

A model file records the version of the fingerprint layout it was trained on
and the versions of this crate that may load it, both in
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). The requirement
has the syntax and the rules of Cargo, such as `^0.1.0-dev` or
`>=0.3, <0.5`. `Model::from_bytes`, and every builder that takes model bytes,
refuses a model whose layout is not compatible with
`fingerprint::LAYOUT_VERSION` (`ModelError::IncompatibleLayout`) or whose
requirement `string2fingerprint::VERSION` does not match
(`ModelError::UnsupportedLibrary`), so a model never runs on fingerprints it
was not trained for. `Metadata::read` reads both versions without the checks:

```rust
use string2fingerprint::model::{Metadata, Model, ModelError};

let bytes = std::fs::read("route.s2fp")?;
match Model::from_bytes(&bytes) {
    Ok(model) => println!("layout {}", model.layout_version()),
    Err(ModelError::IncompatibleLayout | ModelError::UnsupportedLibrary) => {
        let metadata = Metadata::read(&bytes)?;
        eprintln!(
            "route.s2fp needs layout ^{} and library {}",
            metadata.layout_version(),
            metadata.library_requirement()
        );
    }
    Err(error) => return Err(error.into()),
}
```

A model that the crate trains requires `^` the version that trained it, and
`Model::with_library_requirement` states another range. The `model_info`
example prints the versions of model files and whether this crate loads them:

```sh
cargo run --release --example model_info -- route.s2fp default
```

A layout is compatible when its major number equals that of
`fingerprint::LAYOUT_VERSION` and it has no pre-release, since any change to
the fingerprint raises the major number. As in Cargo, a pre-release crate
version such as `0.2.0-rc.1` matches only a requirement that names a
pre-release of the same `0.2.0`, so a model saved for `^0.1.0` does not load
there. `Model::from_bytes_with_library_requirement`, or `model_info
--library-requirement REQ --out FILE MODEL`, states a new range for a model
whose layers still fit.

Files of format version 1, written before the versions were recorded, load
as layout 1.0.0 with the requirement `^0.1.0-dev`.

To score or decide lines of standard input from the command line, or to
measure throughput for each backend that the CPU supports:

```sh
cargo run --release --example score < lines.txt
cargo run --release --example score large < lines.txt
cargo run --release --example detect strict < lines.txt
cargo bench
```

### Low-Level Interface

Each stage of the pipeline is public, for callers that inspect a score,
build their own decision, or run the kernel elsewhere:

| Stage | Items |
| --- | --- |
| Symbols and n-grams | `fingerprint::lexical_symbol`, `shape_class`, `ngrams`, `Section` |
| Fingerprints | `fingerprint::Fingerprint`, `fingerprints`, `Fingerprinter`, `window_range` |
| Kernel and backends | `Backend`, `kernel::Kernel`, `kernel::Weights` |
| Network | `model::Model`, `LayerView`, `Logits`, `Trace` |
| Model versions | `model::Metadata`, `fingerprint::LAYOUT_VERSION`, `VERSION`, `version::Version`, `VersionReq` |

```rust
use string2fingerprint::fingerprint::fingerprints;
use string2fingerprint::kernel::Kernel;
use string2fingerprint::model::Model;

let model = Model::embedded();
let window = fingerprints(b"{{7*7}}").next().expect("the input has a window");
// The accumulators and activations of every layer, and the logits.
let trace = model.trace(Kernel::detect(), &window);
println!("{:?}", trace.logits());
```

`Scorer::builder` chooses the model and the backend and reports a model that
does not parse or a backend that the CPU lacks as one error.

### Asynchronous Interface

The `tokio` feature adds `string2fingerprint::tokio`, the same interface for
[Tokio](https://tokio.rs) 1.54.1 or later. Scoring is CPU-bound, about 1.6 ms
per 64 KiB with a SIMD backend, so these items run short inputs in the task
that polls them and longer ones on Tokio's blocking pool, where they do not
hold a runtime worker. Their results equal those of the synchronous items.

```toml
[dependencies]
string2fingerprint = { git = "https://github.com/OxiBelt/string2fingerprint", features = ["tokio"] }
# The runtime stays the application's own dependency.
tokio = { version = "1.54.1", features = ["rt-multi-thread", "macros", "net", "io-util"] }
```

| Synchronous | Asynchronous (`string2fingerprint::tokio`) |
| --- | --- |
| `Detector`, `Scorer` | `AsyncDetector`, `AsyncScorer`: shared handles whose methods of the same names return futures (`Offload`, `Parallel`) that own their input |
| `DetectionStream`, `ScoreStream` | `AsyncDetectionStream`, `AsyncScoreStream`: also Tokio `AsyncWrite`; long parts are scored on the blocking pool while the next arrives |
| `detect_reader`, `score_reader` on `io::Read` | the same names on Tokio `AsyncRead` |
| `fingerprint::Fingerprinter` | `tokio::fingerprint::FingerprintReader`: the windows of an `AsyncRead` |
| `Model::from_bytes` | `tokio::model::from_bytes`, `tokio::model::read` (a file) |

```rust
use string2fingerprint::tokio::AsyncDetector;
use string2fingerprint::Preset;
use tokio::io::AsyncReadExt;

// Create one detector and clone it into each task.
let detector = AsyncDetector::with_preset(Preset::Strict);

// A request field: up to 512 bytes are decided in this task.
let detection = detector.detect(b"/search?q=1%27%20OR%20%271%27=%271").await;

// A body read from a socket, bounded, scored on the blocking pool while it
// is read.
let detection = detector.detect_reader(body.take(1 << 20)).await?;

// A body that arrives in parts, with an early stop. Parts of at most 512
// bytes are decided as they arrive; a longer one goes to the blocking pool,
// and `stream.flush().await` waits for it.
let mut stream = detector.stream();
while let Some(part) = parts.next().await {
    stream.update(&part).await;
    if stream.detection().is_flagged() {
        break;
    }
}
let detection = stream.finish().await;
```

`AsyncDetector::with_inline_limit` sets the longest input that runs in the
polling task, 512 bytes by default; in a batch or the fields of a message,
each item counts as at least one window of 512 bytes. Dropping a future
before the pool started its work drops the work. Outside a Tokio runtime
every future runs in the task that polls it. The `detect_async` example decides lines of
standard input like `detect`:

```sh
cargo run --release --features tokio --example detect_async strict < lines.txt
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
`--library-requirement REQ` states the versions of the crate that may load
the model, and `ModelBuild::library_requirement` does the same in a build
script.

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
