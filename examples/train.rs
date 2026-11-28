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
//! Give `--base` before the settings that follow.
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
//! The model is written to `MODEL` in the format that `Model::from_bytes`
//! reads. Progress and the report go to standard error. Sample bytes are
//! never printed.

use std::env;
use std::error::Error;
use std::fs;
use std::process::ExitCode;

use string2fingerprint::eval::Evaluation;
use string2fingerprint::model::{EmbeddedModel, Model};
use string2fingerprint::train::{Costs, Dataset, EpochReport, TrainConfig, Trainer};
use string2fingerprint::{Category, CategorySet, Scorer, Threshold};

type Fallible<T> = Result<T, Box<dyn Error>>;

#[derive(Default)]
struct Arguments {
    train: Dataset,
    test: Dataset,
    config: TrainConfig,
    base: Option<Model>,
    out: Option<String>,
}

/// What a data file holds.
enum Format {
    Corpus,
    Benign,
    Labeled,
}

/// Adds the samples of the file at `path` to `data` with `weight`.
fn load(data: &mut Dataset, format: &Format, path: &str, weight: f32) -> Fallible<()> {
    let bytes = fs::read(path)?;
    let mut samples = Dataset::new();
    match format {
        Format::Corpus => samples.extend_from_corpus(&bytes)?,
        Format::Benign => samples.extend_from_lines(&bytes, CategorySet::EMPTY)?,
        Format::Labeled => samples.extend_from_labeled_lines(&bytes)?,
    };
    samples.scale_weights(weight)?;
    data.extend_from_dataset(&samples);
    Ok(())
}

/// Parses `CATEGORY=X` and applies it with `one`, or with `all` for the
/// category `all`.
fn cost(
    costs: Costs,
    value: &str,
    one: impl Fn(Costs, Category, f32) -> Costs,
    all: impl Fn(Costs, f32) -> Costs,
) -> Fallible<Costs> {
    let (name, cost) = value.split_once('=').ok_or("a cost is `CATEGORY=X`")?;
    let cost: f32 = cost.parse()?;
    if name == "all" {
        return Ok(all(costs, cost));
    }
    let category = Category::from_name(name).ok_or("unknown category")?;
    Ok(one(costs, category, cost))
}

fn parse(arguments: &[String]) -> Fallible<Arguments> {
    let mut parsed = Arguments::default();
    let mut weight = 1.0f32;
    let mut rest = arguments.iter();
    while let Some(option) = rest.next() {
        let value = rest
            .next()
            .ok_or_else(|| format!("`{option}` needs a value"))?;
        let config = &mut parsed.config;
        match option.as_str() {
            "--weight" => weight = value.parse()?,
            "--corpus" => load(&mut parsed.train, &Format::Corpus, value, weight)?,
            "--benign" => load(&mut parsed.train, &Format::Benign, value, weight)?,
            "--labeled" => load(&mut parsed.train, &Format::Labeled, value, weight)?,
            "--test-corpus" => load(&mut parsed.test, &Format::Corpus, value, 1.0)?,
            "--test-benign" => load(&mut parsed.test, &Format::Benign, value, 1.0)?,
            "--test-labeled" => load(&mut parsed.test, &Format::Labeled, value, 1.0)?,
            "--base" => {
                parsed.base = Some(match EmbeddedModel::from_name(value) {
                    Some(embedded) => embedded.load(),
                    None => Model::from_bytes(&fs::read(value)?)?,
                });
                *config = TrainConfig::fine_tuning();
            }
            "--hidden" if value == "none" => config.hidden.clear(),
            "--hidden" => {
                config.hidden = value.split(',').map(str::parse).collect::<Result<_, _>>()?;
            }
            "--epochs" => config.epochs = value.parse()?,
            "--quantized-epochs" => config.quantized_epochs = value.parse()?,
            "--batch-size" => config.batch_size = value.parse()?,
            "--learning-rate" => config.learning_rate = value.parse()?,
            "--weight-decay" => config.weight_decay = value.parse()?,
            "--input-dropout" => config.input_dropout = value.parse()?,
            "--seed" => config.seed = value.parse()?,
            "--threads" => config.threads = value.parse()?,
            "--benign-weight" => config.costs = config.costs.benign(value.parse()?),
            "--false-positive-cost" => {
                config.costs = cost(
                    config.costs,
                    value,
                    Costs::false_positive,
                    Costs::false_positives,
                )?;
            }
            "--false-negative-cost" => {
                config.costs = cost(
                    config.costs,
                    value,
                    Costs::false_negative,
                    Costs::false_negatives,
                )?;
            }
            "--out" => parsed.out = Some(value.clone()),
            _ => return Err(format!("unknown option `{option}`").into()),
        }
    }
    Ok(parsed)
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

fn run(arguments: &[String]) -> Fallible<()> {
    let Arguments {
        train,
        test,
        config,
        base,
        out,
    } = parse(arguments)?;
    let out = out.ok_or("`--out MODEL` is required")?;
    eprintln!(
        "training on {} samples ({} windows, {} benign)",
        train.samples(),
        train.windows(),
        train.benign()
    );
    let (held_out, name) = if test.is_empty() {
        (&train, "training data")
    } else {
        (&test, "held-out data")
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
    let trainer = Trainer::new(config)?;
    let trained = match base {
        Some(base) => {
            let trained = trainer.fine_tune_with_progress(&base, &train, progress)?;
            let scorer = Scorer::with_model(base);
            report(
                &format!("base model on {name}"),
                &held_out.evaluate(&scorer),
            );
            trained
        }
        None => trainer.train_with_progress(&train, progress)?,
    };
    let (mean, largest) = trained.report().quantization_drift();
    eprintln!("quantization moves a score by {mean:.2} on average, {largest:.1} at most");

    let bytes = trained.model().to_bytes();
    fs::write(&out, &bytes)?;
    eprintln!("wrote {} bytes to {out}", bytes.len());

    let scorer = Scorer::with_model(trained.into_model());
    report(name, &held_out.evaluate(&scorer));
    Ok(())
}

fn main() -> ExitCode {
    let arguments: Vec<String> = env::args().skip(1).collect();
    match run(&arguments) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("train: {error}");
            eprintln!(
                "usage: train [--corpus FILE] [--benign FILE] [--labeled FILE] [OPTION]... --out MODEL"
            );
            ExitCode::from(2)
        }
    }
}
