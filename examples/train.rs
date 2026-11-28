//! Trains a model on labeled samples and reports how it scores them.
//!
//! Usage: `train [OPTION]... --out MODEL`, with the `train` feature.
//!
//! Training data, each option any number of times:
//!
//! - `--corpus FILE`: records in the format of `tools/train/dataset.py`.
//! - `--benign FILE`: one benign sample on each line.
//! - `--labeled FILE`: labels, a tab, and a payload on each line. Labels
//!   are category names separated by commas, or `benign`.
//!
//! Held-out data, to report false positives and false negatives on samples
//! that the model did not train on: `--test-corpus FILE`, `--test-benign
//! FILE`, `--test-labeled FILE`. Without any, the report is on the training
//! data.
//!
//! `--weight X` before a training data option gives the samples of the
//! files that follow that weight in the loss, in place of 1.
//!
//! `--base MODEL` continues the training of a model in place of starting
//! from random weights. `MODEL` is the name of an embedded model, such as
//! `default`, or the path of a model file. The settings then start from
//! `TrainConfig::fine_tuning`, and the report also shows the base model.
//! `--base` must come before every setting.
//!
//! Settings, with the defaults of `TrainConfig`: `--hidden N[,N]...` (or
//! `--hidden none`), `--epochs N`, `--quantized-epochs N`, `--batch-size N`,
//! `--learning-rate X`, `--weight-decay X`, `--input-dropout X`, `--seed N`,
//! `--threads N`.
//!
//! Costs of errors in the loss, 1 by default: `--benign-weight X`,
//! `--false-positive-cost CATEGORY=X`, `--false-negative-cost CATEGORY=X`.
//! `CATEGORY` is a category name or `all`.
//!
//! `--library-requirement REQ` states the versions of the library that may
//! load the model, such as `>=0.1.0-dev, <0.3`, in place of `^` the version
//! that trains it. The library that trains the model must match it.
//!
//! The model is written to `MODEL` in the format that `Model::from_bytes`
//! reads, with the layout version of the library and the library
//! requirement. Progress and the report go to standard error. Sample bytes are
//! never printed.
//!
//! `--help` prints the usage. The exit status is 2 for an invalid command
//! line or setting, and 1 when reading, training, or writing fails.

use std::env;
use std::fmt;
use std::fs;
use std::process::ExitCode;

use string2fingerprint::eval::Evaluation;
use string2fingerprint::model::{EmbeddedModel, Model};
use string2fingerprint::train::{Costs, Dataset, EpochReport, TrainConfig, Trainer};
use string2fingerprint::version::VersionReq;
use string2fingerprint::{Category, CategorySet, Scorer, Threshold};

const USAGE: &str =
    "usage: train [--corpus FILE] [--benign FILE] [--labeled FILE] [OPTION]... --out MODEL";

/// Why the program stopped.
enum Failure {
    /// The command line or a setting is invalid. Exit status 2.
    Usage(String),
    /// Reading, training, or writing failed. Exit status 1.
    Run(String),
}

impl Failure {
    fn usage(message: impl fmt::Display) -> Self {
        Self::Usage(message.to_string())
    }

    fn run(message: impl fmt::Display) -> Self {
        Self::Run(message.to_string())
    }
}

/// What a data file holds.
#[derive(Clone, Copy)]
enum Format {
    Corpus,
    Benign,
    Labeled,
}

/// A data file of the command line, read only after the whole command line
/// is valid.
struct Source {
    format: Format,
    path: String,
    weight: f32,
}

struct Options {
    train: Vec<Source>,
    test: Vec<Source>,
    config: TrainConfig,
    base: Option<String>,
    library_requirement: Option<VersionReq>,
    out: String,
}

/// Parses `value` as the value of `option`.
fn number<T: std::str::FromStr>(option: &str, value: &str) -> Result<T, Failure>
where
    T::Err: fmt::Display,
{
    value
        .parse()
        .map_err(|error| Failure::usage(format!("`{option} {value}`: {error}")))
}

/// Parses a weight or a cost, which must be finite and at least 0.
fn weight(option: &str, value: &str) -> Result<f32, Failure> {
    let weight: f32 = number(option, value)?;
    if weight.is_finite() && weight >= 0.0 {
        Ok(weight)
    } else {
        Err(Failure::usage(format!(
            "`{option} {value}`: not a finite number of at least 0"
        )))
    }
}

/// Parses `CATEGORY=X` and applies it with `one`, or with `all` for the
/// category `all`.
fn cost(
    costs: Costs,
    option: &str,
    value: &str,
    one: impl Fn(Costs, Category, f32) -> Costs,
    all: impl Fn(Costs, f32) -> Costs,
) -> Result<Costs, Failure> {
    let (name, cost) = value
        .split_once('=')
        .ok_or_else(|| Failure::usage(format!("`{option} {value}`: a cost is `CATEGORY=X`")))?;
    let cost = weight(option, cost).map_err(|_| {
        Failure::usage(format!(
            "`{option} {value}`: a cost is a finite number of at least 0"
        ))
    })?;
    if name == "all" {
        return Ok(all(costs, cost));
    }
    let category = Category::from_name(name)
        .ok_or_else(|| Failure::usage(format!("`{option} {value}`: unknown category `{name}`")))?;
    Ok(one(costs, category, cost))
}

/// Parses the command line. `None` asks for the usage.
fn parse(arguments: &[String]) -> Result<Option<Options>, Failure> {
    let mut train = Vec::new();
    let mut test = Vec::new();
    let mut config = TrainConfig::default();
    let mut base = None;
    let mut library_requirement = None;
    let mut out = None;
    let mut setting = None;
    let mut weight_option = None;
    let mut source_weight = 1.0f32;
    let mut rest = arguments.iter();
    while let Some(option) = rest.next() {
        let option = option.as_str();
        if option == "--help" {
            return Ok(None);
        }
        let value = rest
            .next()
            .ok_or_else(|| Failure::usage(format!("`{option}` needs a value")))?;
        let source = |list: &mut Vec<Source>, format, weight| {
            list.push(Source {
                format,
                path: value.clone(),
                weight,
            });
        };
        match option {
            "--weight" => {
                source_weight = weight(option, value)?;
                weight_option = Some(value.clone());
            }
            "--corpus" | "--benign" | "--labeled" => {
                let format = match option {
                    "--corpus" => Format::Corpus,
                    "--benign" => Format::Benign,
                    _ => Format::Labeled,
                };
                source(&mut train, format, source_weight);
                weight_option = None;
            }
            "--test-corpus" => source(&mut test, Format::Corpus, 1.0),
            "--test-benign" => source(&mut test, Format::Benign, 1.0),
            "--test-labeled" => source(&mut test, Format::Labeled, 1.0),
            "--out" if out.is_some() => return Err(Failure::usage("`--out` is given twice")),
            "--out" => out = Some(value.clone()),
            "--library-requirement" if library_requirement.is_some() => {
                return Err(Failure::usage("`--library-requirement` is given twice"));
            }
            "--library-requirement" => {
                let requirement: VersionReq = value
                    .parse()
                    .map_err(|error| Failure::usage(format!("`{option} {value}`: {error}")))?;
                if !requirement.matches(&string2fingerprint::VERSION) {
                    return Err(Failure::usage(format!(
                        "`{option} {requirement}` does not match this library, version {}",
                        string2fingerprint::VERSION
                    )));
                }
                library_requirement = Some(requirement);
            }
            "--base" if base.is_some() => return Err(Failure::usage("`--base` is given twice")),
            "--base" => {
                if let Some(setting) = setting {
                    return Err(Failure::usage(format!(
                        "`--base` must come before `{setting}`, because it replaces the settings"
                    )));
                }
                base = Some(value.clone());
                config = TrainConfig::fine_tuning();
            }
            _ => {
                match option {
                    "--hidden" if value == "none" => config.hidden.clear(),
                    "--hidden" => {
                        config.hidden = value
                            .split(',')
                            .map(str::parse)
                            .collect::<Result<_, _>>()
                            .map_err(|error| {
                                Failure::usage(format!("`{option} {value}`: {error}"))
                            })?;
                    }
                    "--epochs" => config.epochs = number(option, value)?,
                    "--quantized-epochs" => config.quantized_epochs = number(option, value)?,
                    "--batch-size" => config.batch_size = number(option, value)?,
                    "--learning-rate" => config.learning_rate = number(option, value)?,
                    "--weight-decay" => config.weight_decay = number(option, value)?,
                    "--input-dropout" => config.input_dropout = number(option, value)?,
                    "--seed" => config.seed = number(option, value)?,
                    "--threads" => config.threads = number(option, value)?,
                    "--benign-weight" => {
                        config.costs = config.costs.benign(weight(option, value)?);
                    }
                    "--false-positive-cost" => {
                        config.costs = cost(
                            config.costs,
                            option,
                            value,
                            Costs::false_positive,
                            Costs::false_positives,
                        )?;
                    }
                    "--false-negative-cost" => {
                        config.costs = cost(
                            config.costs,
                            option,
                            value,
                            Costs::false_negative,
                            Costs::false_negatives,
                        )?;
                    }
                    _ => return Err(Failure::usage(format!("unknown option `{option}`"))),
                }
                setting = Some(option.to_owned());
            }
        }
    }
    if let Some(value) = weight_option {
        return Err(Failure::usage(format!(
            "`--weight {value}` is not followed by a training data file"
        )));
    }
    let out = out.ok_or_else(|| Failure::usage("`--out MODEL` is required"))?;
    Ok(Some(Options {
        train,
        test,
        config,
        base,
        library_requirement,
        out,
    }))
}

/// Reads the samples of `sources` into one dataset.
fn load(sources: &[Source]) -> Result<Dataset, Failure> {
    let mut data = Dataset::new();
    for source in sources {
        let path = &source.path;
        let bytes = fs::read(path).map_err(|error| Failure::run(format!("{path}: {error}")))?;
        let mut samples = Dataset::new();
        match source.format {
            Format::Corpus => samples.extend_from_corpus(&bytes),
            Format::Benign => samples.extend_from_lines(&bytes, CategorySet::EMPTY),
            Format::Labeled => samples.extend_from_labeled_lines(&bytes),
        }
        .and_then(|_| samples.scale_weights(source.weight))
        .map_err(|error| Failure::run(format!("{path}: {error}")))?;
        data.extend_from_dataset(&samples);
    }
    Ok(data)
}

/// Loads `--base`: an embedded model by name, or a model file.
fn base_model(value: &str) -> Result<Model, Failure> {
    if let Some(embedded) = EmbeddedModel::from_name(value) {
        return Ok(embedded.load());
    }
    let bytes = fs::read(value).map_err(|error| {
        Failure::run(format!(
            "`--base {value}` is neither the name of an embedded model nor a readable file: {error}"
        ))
    })?;
    Model::from_bytes(&bytes).map_err(|error| Failure::run(format!("{value}: {error}")))
}

fn percent(rate: Option<f64>) -> String {
    rate.map_or_else(|| "-".to_owned(), |rate| format!("{:.2}%", 100.0 * rate))
}

fn report(name: &str, evaluation: &Evaluation) {
    eprintln!(
        "{name}: {} samples, {} benign",
        evaluation.samples(),
        evaluation.benign_samples()
    );
    let benign = evaluation.benign_negatives();
    eprintln!(
        "  category            positives     AUC  recall@128  benign@128  recall@192  benign@192"
    );
    for category in Category::ALL {
        let low = benign.confusion(category, Threshold::at(128));
        let high = benign.confusion(category, Threshold::at(192));
        eprintln!(
            "  {:<19} {:>9} {:>7} {:>11} {:>11} {:>11} {:>11}",
            category.name(),
            low.positives(),
            evaluation
                .auc(category)
                .map_or_else(|| "-".to_owned(), |auc| format!("{auc:.4}")),
            percent(low.recall()),
            percent(low.false_positive_rate()),
            percent(high.recall()),
            percent(high.false_positive_rate()),
        );
    }
    for threshold in [128, 192, 230] {
        let any = evaluation.any_confusion(Threshold::at(threshold));
        eprintln!(
            "  any category at {threshold}: {} of attack samples flagged, {} of benign samples flagged",
            percent(any.recall()),
            percent(any.false_positive_rate()),
        );
    }
}

fn run(options: Options) -> Result<(), Failure> {
    let Options {
        train,
        test,
        config,
        base,
        library_requirement,
        out,
    } = options;
    let trainer = Trainer::new(config).map_err(Failure::usage)?;
    let base = base.as_deref().map(base_model).transpose()?;
    let test_given = !test.is_empty();
    let (train, test) = (load(&train)?, load(&test)?);
    if test_given && test.is_empty() {
        return Err(Failure::run("the held-out data files hold no samples"));
    }
    eprintln!(
        "training on {} samples ({} windows, {} benign)",
        train.samples(),
        train.windows(),
        train.benign()
    );
    let (held_out, name) = if test_given {
        (&test, "held-out data")
    } else {
        (&train, "training data")
    };
    let progress = |epoch: &EpochReport| {
        eprintln!(
            "epoch {}/{}{}: loss {:.5}",
            epoch.epoch,
            epoch.epochs,
            if epoch.quantized { " (quantized)" } else { "" },
            epoch.loss
        );
    };
    let trained = match base {
        Some(base) => {
            let trained = trainer
                .fine_tune_with_progress(&base, &train, progress)
                .map_err(Failure::run)?;
            let scorer = Scorer::with_model(base);
            report(
                &format!("base model on {name}"),
                &held_out.evaluate(&scorer),
            );
            trained
        }
        None => trainer
            .train_with_progress(&train, progress)
            .map_err(Failure::run)?,
    };
    let (mean, largest) = trained.report().quantization_drift();
    eprintln!("quantization moves a score by {mean:.2} on average, {largest:.1} at most");

    let mut model = trained.into_model();
    if let Some(requirement) = library_requirement {
        // The requirement was checked against this library when it was
        // parsed.
        model = model
            .with_library_requirement(requirement)
            .map_err(Failure::run)?;
    }
    let bytes = model.to_bytes();
    fs::write(&out, &bytes).map_err(|error| Failure::run(format!("{out}: {error}")))?;
    eprintln!(
        "wrote {} bytes to {out}: layout {}, library {}",
        bytes.len(),
        model.layout_version(),
        model.library_requirement()
    );

    let scorer = Scorer::with_model(model);
    report(name, &held_out.evaluate(&scorer));
    Ok(())
}

fn main() -> ExitCode {
    let arguments: Result<Vec<String>, _> = env::args_os()
        .skip(1)
        .map(|argument| argument.into_string())
        .collect();
    let result = match arguments {
        Ok(arguments) => parse(&arguments),
        Err(_) => Err(Failure::usage("an argument is not valid UTF-8")),
    };
    let result = match result {
        Ok(Some(options)) => run(options),
        Ok(None) => {
            println!("{USAGE}");
            println!("Every option is described in examples/train.rs.");
            return ExitCode::SUCCESS;
        }
        Err(failure) => Err(failure),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(Failure::Usage(message)) => {
            eprintln!("train: {message}");
            eprintln!("{USAGE}");
            ExitCode::from(2)
        }
        Err(Failure::Run(message)) => {
            eprintln!("train: {message}");
            ExitCode::FAILURE
        }
    }
}
