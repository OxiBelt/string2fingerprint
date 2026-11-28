//! Training of a model while a dependent crate is compiled.
//!
//! This module is available with the `build` feature, which includes the
//! `train` feature. A build script calls [`ModelBuild`] to train a model
//! from the data files of its crate, and the crate embeds the result with
//! [`include_model!`](crate::include_model). The model is then trained
//! again whenever a data file changes, and never at run time.
//!
//! Depend on the library twice: as a dependency for scoring, and as a build
//! dependency with the `build` feature for training. With feature resolver
//! 2 or later, the default from edition 2021 on, Cargo builds the two
//! separately, so the compiled crate contains no training code. A virtual
//! workspace needs `resolver = "2"` or later for that.
//!
//! ```toml
//! [dependencies]
//! string2fingerprint = { git = "https://github.com/OxiBelt/string2fingerprint" }
//!
//! [build-dependencies]
//! string2fingerprint = { git = "https://github.com/OxiBelt/string2fingerprint", features = ["build"] }
//!
//! # Cargo compiles build dependencies without optimization unless told
//! # otherwise, which makes training many times slower.
//! [profile.dev.package.string2fingerprint]
//! opt-level = 3
//!
//! [profile.release.package.string2fingerprint]
//! opt-level = 3
//! ```
//!
//! Cargo reads profiles only from the root manifest of the workspace that is
//! built. When the crate that trains is a member of a workspace or a
//! dependency of another crate, the profile belongs in that root manifest.
//! `[profile.dev.build-override] opt-level = 3` there optimizes every build
//! dependency instead.
//!
//! `build.rs`:
//!
//! ```no_run
//! use string2fingerprint::build::{ModelBuild, Source};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     ModelBuild::new("waf")
//!         .train_on(Source::benign_lines("data/benign.txt"))
//!         .train_on(Source::labeled_lines("data/attacks.tsv"))
//!         .run()?;
//!     Ok(())
//! }
//! ```
//!
//! `src/main.rs`:
//!
//! ```ignore
//! use string2fingerprint::Scorer;
//!
//! let model = string2fingerprint::include_model!("waf").expect("the build script validated the model");
//! let scorer = Scorer::with_model(model);
//! ```
//!
//! # What the build does
//!
//! [`ModelBuild::run`] reads the sources, trains, and writes
//! `<name>.s2fp` to the directory that Cargo gives the build script in
//! `OUT_DIR`. With [`ModelBuild::calibrate`] it also chooses a threshold
//! for each category on samples that training did not see, and writes
//! `<name>.thresholds` for
//! [`include_thresholds!`](crate::include_thresholds). It returns a
//! [`BuiltModel`], so that the build script can fail the build when the
//! model does not meet a requirement of its own.
//!
//! Training is deterministic, so the same data files give the same model
//! bytes on every build host and with any number of build jobs.
//!
//! `run` prints `cargo::rerun-if-changed` for every source and
//! `cargo::rerun-if-env-changed` for the variables below. Cargo then runs
//! the build script again only when one of them or the build script itself
//! changes, and no longer when any other file of the package changes. A
//! build script that reads other files must print the directive for them
//! itself.
//!
//! # Environment variables
//!
//! - `STRING2FINGERPRINT_MODEL_<NAME>` holds the path of a model file to
//!   use in place of training. `<NAME>` is the name of the model in upper
//!   case. A relative path is relative to the directory of the package. The
//!   file is validated and written to `OUT_DIR` like a trained model, and
//!   the build prints a warning that names it. The training sources are
//!   then not read, unless samples are [held out](ModelBuild::hold_out)
//!   from them. Use this to build with a model that was trained elsewhere,
//!   or without the data. An evaluation on held-out samples only means
//!   something when that model was not trained on them, such as a model
//!   that an earlier build of the same sources with the same seed trained.
//! - `STRING2FINGERPRINT_TRAIN_THREADS` holds the number of threads to
//!   train with. Without it, the build uses
//!   [`TrainConfig::threads`](crate::train::TrainConfig::threads) when
//!   that is not 0, and otherwise the number of jobs of the Cargo build.
//!
//! # Limits
//!
//! - The build host must be Linux, like every target of this library,
//!   because the build script links the library.
//! - Training takes build time in proportion to the data: about a second
//!   for each 10,000 samples with the default settings on four cores of
//!   an optimized build, and about 20 times that without the optimization
//!   above.
//! - Data files and the variables are trusted input. A build that trains on
//!   files from an untrusted source, or whose environment an untrusted
//!   source sets, lets that source choose what the model flags.
//! - A path must be UTF-8 without a line break, so that Cargo can read the
//!   directives that name it.

use std::env;
use std::ffi::OsString;
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::eval::{Evaluation, Objective};
use crate::model::{Model, ModelError};
use crate::train::{Dataset, DatasetError, TrainConfig, TrainError, TrainReport, Trainer};
use crate::{CategorySet, Scorer, Thresholds};

/// The prefix of the variable that names a model file to use in place of
/// training.
const MODEL_VARIABLE: &str = "STRING2FINGERPRINT_MODEL_";

/// The variable that holds the number of training threads.
const THREADS_VARIABLE: &str = "STRING2FINGERPRINT_TRAIN_THREADS";

/// The number of seconds of training after which the build warns that it
/// may be compiled without optimization.
const SLOW_SECONDS: u64 = 30;

/// The longest name of a model.
const MAX_NAME_LEN: usize = 64;

/// How the samples of a file are stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Format {
    /// One sample on each line, all with the same labels.
    Lines(CategorySet),
    /// Labels, a tab, and a payload on each line.
    LabeledLines,
    /// The corpus format of `tools/train/dataset.py`.
    Corpus,
}

/// A file of samples for [`ModelBuild`].
///
/// A relative path is relative to the directory of the package, which is
/// the working directory of a build script.
///
/// ```
/// use string2fingerprint::Category;
/// use string2fingerprint::build::Source;
///
/// let benign = Source::benign_lines("data/benign.txt");
/// let sql = Source::lines("data/sql.txt", Category::SqlInjection);
/// // Benign text that an earlier model flagged counts four times.
/// let corrections = Source::benign_lines("data/flagged.txt").weighted(4.0);
/// # let _ = (benign, sql, corrections);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Source {
    path: PathBuf,
    format: Format,
    weight: f32,
}

impl Source {
    fn new(path: impl AsRef<Path>, format: Format) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            format,
            weight: 1.0,
        }
    }

    /// A file with one benign sample on each line, as
    /// [`Dataset::extend_from_lines`] reads it.
    #[must_use]
    pub fn benign_lines(path: impl AsRef<Path>) -> Self {
        Self::new(path, Format::Lines(CategorySet::EMPTY))
    }

    /// A file with one sample on each line, all with the same `labels`, as
    /// [`Dataset::extend_from_lines`] reads it.
    #[must_use]
    pub fn lines(path: impl AsRef<Path>, labels: impl Into<CategorySet>) -> Self {
        Self::new(path, Format::Lines(labels.into()))
    }

    /// A file with labels, a tab, and a payload on each line, as
    /// [`Dataset::extend_from_labeled_lines`] reads it.
    #[must_use]
    pub fn labeled_lines(path: impl AsRef<Path>) -> Self {
        Self::new(path, Format::LabeledLines)
    }

    /// A corpus of `tools/train/dataset.py`, as
    /// [`Dataset::extend_from_corpus`] reads it.
    #[must_use]
    pub fn corpus(path: impl AsRef<Path>) -> Self {
        Self::new(path, Format::Corpus)
    }

    /// Sets the [weight](crate::train::Sample::weighted) of every sample of
    /// the file. The weight is 1 by default, and it has no effect on
    /// [validation](ModelBuild::validate_on).
    #[must_use]
    pub fn weighted(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    /// Returns the path of the file.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Reads the samples of the file.
    fn load(&self) -> Result<Dataset, BuildError> {
        let bytes = fs::read(&self.path).map_err(|error| BuildError::Io {
            path: self.path.clone(),
            error,
        })?;
        let mut data = Dataset::new();
        match self.format {
            Format::Lines(labels) => data.extend_from_lines(&bytes, labels),
            Format::LabeledLines => data.extend_from_labeled_lines(&bytes),
            Format::Corpus => data.extend_from_corpus(&bytes),
        }
        .and_then(|_| data.scale_weights(self.weight))
        .map_err(|error| BuildError::Dataset {
            path: self.path.clone(),
            error,
        })?;
        Ok(data)
    }
}

/// The reason [`ModelBuild::run`] did not write a model.
///
/// The error names files and never contains sample bytes.
#[derive(Debug)]
#[non_exhaustive]
pub enum BuildError {
    /// The name of the model is empty, longer than 64 bytes, or contains a
    /// byte other than a lowercase ASCII letter, a digit, or `_`.
    InvalidName,
    /// No output directory was set and `OUT_DIR` is not set, so the caller
    /// is not a build script.
    MissingOutDir,
    /// A setting is out of range. The value names the setting.
    InvalidSetting(&'static str),
    /// A file could not be read or written.
    Io {
        /// The file.
        path: PathBuf,
        /// The error of the operating system.
        error: io::Error,
    },
    /// A source does not hold samples in its format.
    Dataset {
        /// The file of the source.
        path: PathBuf,
        /// What is wrong with it.
        error: DatasetError,
    },
    /// The file named by `STRING2FINGERPRINT_MODEL_<NAME>` is not a model.
    Model {
        /// The file.
        path: PathBuf,
        /// What is wrong with it.
        error: ModelError,
    },
    /// Training failed.
    Train(TrainError),
    /// Thresholds were requested, and no sample was held out or given for
    /// validation.
    NoValidationSamples,
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidName => f.write_str(
                "model name must be 1 to 64 lowercase ASCII letters, digits, or underscores",
            ),
            Self::MissingOutDir => {
                f.write_str("no output directory: `OUT_DIR` is not set outside a build script")
            }
            Self::InvalidSetting(reason) => write!(f, "model build is invalid: {reason}"),
            Self::Io { path, error } => write!(f, "{}: {error}", path.display()),
            Self::Dataset { path, error } => write!(f, "{}: {error}", path.display()),
            Self::Model { path, error } => write!(f, "{}: {error}", path.display()),
            Self::Train(error) => write!(f, "{error}"),
            Self::NoValidationSamples => {
                f.write_str("thresholds need samples that are held out or given for validation")
            }
        }
    }
}

impl std::error::Error for BuildError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { error, .. } => Some(error),
            Self::Dataset { error, .. } => Some(error),
            Self::Model { error, .. } => Some(error),
            Self::Train(error) => Some(error),
            _ => None,
        }
    }
}

impl From<TrainError> for BuildError {
    fn from(error: TrainError) -> Self {
        Self::Train(error)
    }
}

/// Trains a model in a build script and writes it where
/// [`include_model!`](crate::include_model) finds it.
///
/// The [module documentation](self) shows the whole setup.
///
/// ```no_run
/// use string2fingerprint::build::{ModelBuild, Source};
/// use string2fingerprint::eval::Objective;
/// use string2fingerprint::train::{Costs, TrainConfig};
/// use string2fingerprint::{Threshold, Thresholds};
///
/// let mut config = TrainConfig::default();
/// config.costs = Costs::UNIFORM.benign(4.0);
///
/// let built = ModelBuild::new("waf")
///     .train_on(Source::benign_lines("data/benign.txt"))
///     .train_on(Source::labeled_lines("data/attacks.tsv"))
///     .config(config)
///     // Keep a tenth of the samples out of training, and choose on them
///     // the thresholds that flag at most 1 benign sample in 1000.
///     .hold_out(0.1)
///     .calibrate(
///         Objective::MaxFalsePositiveRate(0.001),
///         Thresholds::uniform(Threshold::at(192)),
///     )
///     .run()?;
///
/// // Fail the build when the model misses too many held-out attacks.
/// let evaluation = built.evaluation().expect("samples were held out");
/// let thresholds = built.thresholds().expect("thresholds were calibrated");
/// for (category, threshold) in thresholds.iter() {
///     let recall = evaluation.confusion(category, threshold).recall();
///     assert!(recall.is_none_or(|recall| recall >= 0.8), "{category}");
/// }
/// # Ok::<(), string2fingerprint::build::BuildError>(())
/// ```
#[derive(Clone, Debug)]
#[must_use = "a model build does nothing until `run` is called"]
pub struct ModelBuild {
    name: String,
    training: Vec<Source>,
    validation: Vec<Source>,
    base: Option<Model>,
    config: Option<TrainConfig>,
    held_out: f64,
    calibration: Option<(Objective, Thresholds)>,
    out_dir: Option<PathBuf>,
}

impl ModelBuild {
    /// Starts the build of the model `name`.
    ///
    /// The name is the one that [`include_model!`](crate::include_model)
    /// takes, and the model is written to `<name>.s2fp`. It consists of 1
    /// to 64 lowercase ASCII letters, digits, and `_`, so that each name
    /// has a variable of its own.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            training: Vec::new(),
            validation: Vec::new(),
            base: None,
            config: None,
            held_out: 0.0,
            calibration: None,
            out_dir: None,
        }
    }

    /// Adds a file of samples to train on.
    pub fn train_on(mut self, source: Source) -> Self {
        self.training.push(source);
        self
    }

    /// Adds a file of samples that the model is evaluated on and not
    /// trained on.
    pub fn validate_on(mut self, source: Source) -> Self {
        self.validation.push(source);
        self
    }

    /// Keeps about `fraction` of the training samples, from 0.0 to less
    /// than 1.0, out of training and evaluates the model on them.
    ///
    /// The samples are chosen by [`Dataset::split`] with the
    /// [seed](TrainConfig::seed) of the settings.
    pub fn hold_out(mut self, fraction: f64) -> Self {
        self.held_out = fraction;
        self
    }

    /// Sets the training settings.
    ///
    /// The default is [`TrainConfig::default`], or
    /// [`TrainConfig::fine_tuning`] with a [base model](Self::fine_tune).
    pub fn config(mut self, config: TrainConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Continues the training of `base`, as [`Trainer::fine_tune`] does,
    /// in place of training a new model.
    ///
    /// Read the documentation of [`Trainer::fine_tune`] first: tuning on
    /// benign text alone makes the model miss attacks that look like it.
    /// A build script that reads `base` from a file prints
    /// `cargo::rerun-if-changed` for that file itself.
    pub fn fine_tune(mut self, base: Model) -> Self {
        self.base = Some(base);
        self
    }

    /// Chooses a threshold for each category on the validation samples, as
    /// [`Evaluation::thresholds`] does, and writes them to
    /// `<name>.thresholds` for
    /// [`include_thresholds!`](crate::include_thresholds).
    ///
    /// A category for which `objective` is undefined, such as one without a
    /// validation sample, keeps its threshold from `fallback`. A rate or a
    /// recall outside 0.0 to 1.0, or a cost that is negative or not finite,
    /// is refused. The build needs samples from [`ModelBuild::hold_out`] or
    /// [`ModelBuild::validate_on`].
    ///
    /// Without this call, the build removes a thresholds file that an
    /// earlier build wrote, so that
    /// [`include_thresholds!`](crate::include_thresholds) cannot embed
    /// thresholds of another model.
    pub fn calibrate(mut self, objective: Objective, fallback: Thresholds) -> Self {
        self.calibration = Some((objective, fallback));
        self
    }

    /// Writes to `dir` in place of the directory in `OUT_DIR`.
    ///
    /// Use this to run a build outside a build script.
    pub fn out_dir(mut self, dir: impl AsRef<Path>) -> Self {
        self.out_dir = Some(dir.as_ref().to_path_buf());
        self
    }

    /// Trains the model and writes it.
    ///
    /// Prints the directives for Cargo that the
    /// [module documentation](self#what-the-build-does) lists, and a
    /// warning when training took long enough that the library was
    /// probably compiled without optimization.
    ///
    /// # Errors
    ///
    /// Returns a [`BuildError`] when the name or a setting is invalid, a
    /// file cannot be read or written, a source or a model file is
    /// malformed, training fails, or thresholds were requested without
    /// validation samples. Files are written last, so an error other than
    /// a failed write leaves the output directory as it was.
    pub fn run(self) -> Result<BuiltModel, BuildError> {
        let mut lines = Vec::new();
        let result = self.execute(&|name| env::var_os(name), &mut lines);
        for line in lines {
            println!("{line}");
        }
        result
    }

    /// Runs the build with the environment `variable` and collects in
    /// `lines` what the build script prints: directives for Cargo, and
    /// other lines, which Cargo shows only when asked to be very verbose.
    fn execute(
        self,
        variable: &dyn Fn(&str) -> Option<OsString>,
        lines: &mut Vec<String>,
    ) -> Result<BuiltModel, BuildError> {
        let name_is_valid = (1..=MAX_NAME_LEN).contains(&self.name.len())
            && self
                .name
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_');
        if !name_is_valid {
            return Err(BuildError::InvalidName);
        }
        if !(0.0..1.0).contains(&self.held_out) {
            return Err(BuildError::InvalidSetting(
                "the held-out fraction is not from 0.0 to less than 1.0",
            ));
        }
        if let Some((objective, _)) = self.calibration {
            if !objective_is_valid(objective) {
                return Err(BuildError::InvalidSetting(
                    "the objective has a rate outside 0.0 to 1.0 or an invalid cost",
                ));
            }
            if self.validation.is_empty() && self.held_out == 0.0 {
                return Err(BuildError::NoValidationSamples);
            }
        }

        let model_variable = model_variable(&self.name);
        let prebuilt = variable(&model_variable)
            .filter(|path| !path.is_empty())
            .map(PathBuf::from);
        // Every path is checked before any directive is printed.
        let mut watched = Vec::new();
        for path in self
            .training
            .iter()
            .chain(&self.validation)
            .map(Source::path)
            .chain(prebuilt.as_deref())
        {
            watched.push(directive_path(path)?);
        }
        lines.push(format!("cargo::rerun-if-env-changed={model_variable}"));
        lines.push(format!("cargo::rerun-if-env-changed={THREADS_VARIABLE}"));
        for path in watched {
            lines.push(format!("cargo::rerun-if-changed={path}"));
        }

        let out_dir = self
            .out_dir
            .clone()
            .or_else(|| variable("OUT_DIR").map(PathBuf::from))
            .ok_or(BuildError::MissingOutDir)?;

        let mut config = self.config.clone().unwrap_or_else(|| match self.base {
            Some(_) => TrainConfig::fine_tuning(),
            None => TrainConfig::default(),
        });
        config.threads = threads(config.threads, variable)?;
        // Settings are checked before any file is read.
        let trainer = Trainer::new(config)?;

        let mut validation = Dataset::new();
        for source in &self.validation {
            validation.extend_from_dataset(&source.load()?);
        }
        let mut training = Dataset::new();
        if prebuilt.is_none() || self.held_out > 0.0 {
            for source in &self.training {
                training.extend_from_dataset(&source.load()?);
            }
        }
        if self.held_out > 0.0 {
            let (rest, held_out) = training.split(self.held_out, trainer.config().seed);
            training = rest;
            validation.extend_from_dataset(&held_out);
        }
        if self.calibration.is_some() && validation.is_empty() {
            return Err(BuildError::NoValidationSamples);
        }

        let (model, report) = match &prebuilt {
            Some(path) => {
                let io = |error| BuildError::Io {
                    path: path.clone(),
                    error,
                };
                let model = Model::from_bytes(&fs::read(path).map_err(io)?).map_err(|error| {
                    BuildError::Model {
                        path: path.clone(),
                        error,
                    }
                })?;
                lines.push(format!(
                    "cargo::warning=the model `{}` is the file {} that `{model_variable}` names, \
                     not a model trained by this build",
                    self.name,
                    path.display()
                ));
                (model, None)
            }
            None => {
                let start = Instant::now();
                let trained = match &self.base {
                    Some(base) => trainer.fine_tune(base, &training),
                    None => trainer.train(&training),
                }?;
                let seconds = start.elapsed().as_secs();
                lines.push(format!(
                    "string2fingerprint: trained model `{}` on {} samples in {seconds} s, loss {:.5}",
                    self.name,
                    training.samples(),
                    trained.report().final_loss()
                ));
                if seconds >= SLOW_SECONDS {
                    lines.push(format!(
                        "cargo::warning=training the model `{}` took {seconds} s. Cargo compiles \
                         build dependencies without optimization by default: set `opt-level = 3` \
                         under `[profile.dev.package.string2fingerprint]` and \
                         `[profile.release.package.string2fingerprint]` in the root manifest of \
                         the workspace",
                        self.name
                    ));
                }
                let report = trained.report().clone();
                (trained.into_model(), Some(report))
            }
        };

        let scorer = Scorer::with_model(model);
        let evaluation = (!validation.is_empty()).then(|| validation.evaluate(&scorer));
        let thresholds = match (&self.calibration, &evaluation) {
            (Some((objective, fallback)), Some(evaluation)) => {
                Some(evaluation.thresholds(*objective, *fallback))
            }
            _ => None,
        };
        let model = scorer.model().clone();

        let write = |extension: &str, bytes: &[u8]| {
            let path = out_dir.join(format!("{}.{extension}", self.name));
            match fs::write(&path, bytes) {
                Ok(()) => Ok(path),
                Err(error) => Err(BuildError::Io { path, error }),
            }
        };
        let path = write(MODEL_EXTENSION, &model.to_bytes())?;
        let thresholds_path = match &thresholds {
            Some(thresholds) => Some(write(THRESHOLDS_EXTENSION, &thresholds.to_bytes())?),
            None => {
                // Cargo keeps `OUT_DIR` between builds, so the thresholds of
                // an earlier build would otherwise stay.
                let stale = out_dir.join(format!("{}.{THRESHOLDS_EXTENSION}", self.name));
                match fs::remove_file(&stale) {
                    Ok(()) => {}
                    Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                    Err(error) => return Err(BuildError::Io { path: stale, error }),
                }
                None
            }
        };

        Ok(BuiltModel {
            model,
            path,
            report,
            training_samples: if prebuilt.is_some() {
                0
            } else {
                training.samples()
            },
            evaluation,
            thresholds,
            thresholds_path,
        })
    }
}

/// The extension of a model file.
const MODEL_EXTENSION: &str = "s2fp";

/// The extension of a thresholds file.
const THRESHOLDS_EXTENSION: &str = "thresholds";

/// Returns the name of the variable that replaces the model `name`.
fn model_variable(name: &str) -> String {
    format!("{MODEL_VARIABLE}{}", name.to_ascii_uppercase())
}

/// Returns `path` as the text of a directive for Cargo, which must be UTF-8
/// and on one line.
fn directive_path(path: &Path) -> Result<&str, BuildError> {
    path.to_str()
        .filter(|text| !text.contains(['\n', '\r']))
        .ok_or(BuildError::InvalidSetting(
            "a path is not UTF-8 or contains a line break",
        ))
}

/// Returns whether [`Evaluation::thresholds`] can meet `objective` for a
/// category with samples.
fn objective_is_valid(objective: Objective) -> bool {
    match objective {
        Objective::MaxFalsePositiveRate(rate) | Objective::MinRecall(rate) => {
            (0.0..=1.0).contains(&rate)
        }
        Objective::MinCost {
            false_positive,
            false_negative,
        } => [false_positive, false_negative]
            .iter()
            .all(|cost| cost.is_finite() && *cost >= 0.0),
    }
}

/// Returns the number of threads to train with: the variable for it, then
/// the setting when it is not 0, then the number of jobs of the Cargo
/// build, then 0 for every thread of the host.
fn threads(
    setting: usize,
    variable: &dyn Fn(&str) -> Option<OsString>,
) -> Result<usize, BuildError> {
    let parse = |value: OsString| value.to_str().and_then(|text| text.trim().parse().ok());
    if let Some(value) = variable(THREADS_VARIABLE).filter(|value| !value.is_empty()) {
        return parse(value).filter(|&threads: &usize| threads > 0).ok_or(
            BuildError::InvalidSetting(
                "`STRING2FINGERPRINT_TRAIN_THREADS` is not a number above 0",
            ),
        );
    }
    if setting != 0 {
        return Ok(setting);
    }
    Ok(variable("NUM_JOBS").and_then(parse).unwrap_or(0))
}

/// The result of [`ModelBuild::run`].
#[derive(Clone, Debug)]
pub struct BuiltModel {
    model: Model,
    path: PathBuf,
    report: Option<TrainReport>,
    training_samples: usize,
    evaluation: Option<Evaluation>,
    thresholds: Option<Thresholds>,
    thresholds_path: Option<PathBuf>,
}

impl BuiltModel {
    /// Returns the model that was written.
    #[must_use]
    pub fn model(&self) -> &Model {
        &self.model
    }

    /// Returns the path of the model file.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns the report of the training run, or `None` when the model
    /// came from the file that `STRING2FINGERPRINT_MODEL_<NAME>` names.
    #[must_use]
    pub fn report(&self) -> Option<&TrainReport> {
        self.report.as_ref()
    }

    /// Returns the number of samples that the model was trained on, without
    /// the held-out ones. It is 0 when the model came from the file that
    /// `STRING2FINGERPRINT_MODEL_<NAME>` names.
    #[must_use]
    pub fn training_samples(&self) -> usize {
        self.training_samples
    }

    /// Returns the scores of the model on the validation samples, or `None`
    /// without such samples.
    #[must_use]
    pub fn evaluation(&self) -> Option<&Evaluation> {
        self.evaluation.as_ref()
    }

    /// Returns the thresholds that were written, or `None` when
    /// [`ModelBuild::calibrate`] was not called.
    #[must_use]
    pub fn thresholds(&self) -> Option<Thresholds> {
        self.thresholds
    }

    /// Returns the path of the thresholds file, or `None` when
    /// [`ModelBuild::calibrate`] was not called.
    #[must_use]
    pub fn thresholds_path(&self) -> Option<&Path> {
        self.thresholds_path.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::process;

    use super::*;
    use crate::model::EmbeddedModel;
    use crate::{Category, Threshold};

    /// A directory for the files of one test, removed when dropped.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(test: &str) -> Self {
            let dir = env::temp_dir().join(format!("s2fp-build-{}-{test}", process::id()));
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(dir.join("out")).unwrap();
            Self(dir)
        }

        fn out(&self) -> PathBuf {
            self.0.join("out")
        }

        fn file(&self, name: &str, contents: impl AsRef<[u8]>) -> PathBuf {
            let path = self.0.join(name);
            fs::write(&path, contents).unwrap();
            path
        }

        /// Writes `count` benign lines and `count` labeled lines and returns
        /// the two files.
        fn samples(&self, count: usize) -> (PathBuf, PathBuf) {
            let (mut benign, mut labeled) = (String::new(), String::from("# labels, payload\n"));
            for index in 0..count {
                benign += &format!("/items/{index}?page={}&sort=name\n", index % 7);
                labeled += &format!(
                    "sql_injection\t/items/{index}' UNION SELECT {index}, password FROM users--\n"
                );
            }
            (
                self.file("benign.txt", benign),
                self.file("attacks.tsv", labeled),
            )
        }

        fn written(&self) -> Vec<String> {
            let mut names: Vec<String> = fs::read_dir(self.out())
                .unwrap()
                .map(|entry| entry.unwrap().file_name().into_string().unwrap())
                .collect();
            names.sort();
            names
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn config() -> TrainConfig {
        TrainConfig {
            hidden: vec![6],
            epochs: 8,
            quantized_epochs: 2,
            batch_size: 16,
            learning_rate: 1e-2,
            ..TrainConfig::default()
        }
    }

    fn build(scratch: &Scratch, name: &str, samples: usize) -> ModelBuild {
        let (benign, labeled) = scratch.samples(samples);
        ModelBuild::new(name)
            .train_on(Source::benign_lines(benign))
            .train_on(Source::labeled_lines(labeled))
            .config(config())
            .out_dir(scratch.out())
    }

    type Variables<'a> = &'a [(&'a str, &'a str)];

    fn execute(
        build: ModelBuild,
        variables: Variables<'_>,
    ) -> (Result<BuiltModel, BuildError>, Vec<String>) {
        let variables: HashMap<&str, &str> = variables.iter().copied().collect();
        let mut lines = Vec::new();
        let result = build.execute(&|name| variables.get(name).map(OsString::from), &mut lines);
        (result, lines)
    }

    fn run(build: ModelBuild) -> BuiltModel {
        execute(build, &[]).0.unwrap()
    }

    #[test]
    fn a_build_writes_the_model_that_the_trainer_gives() {
        let scratch = Scratch::new("writes");
        let (benign, labeled) = scratch.samples(40);
        let (result, lines) = execute(build(&scratch, "waf", 40), &[("NUM_JOBS", "3")]);
        let built = result.unwrap();

        let mut data = Dataset::new();
        data.extend_from_lines(&fs::read(&benign).unwrap(), CategorySet::EMPTY)
            .unwrap();
        data.extend_from_labeled_lines(&fs::read(&labeled).unwrap())
            .unwrap();
        let trained = Trainer::new(config()).unwrap().train(&data).unwrap();
        assert_eq!(built.model(), trained.model());
        assert_eq!(built.report(), Some(trained.report()));
        assert_eq!(built.training_samples(), 80);
        assert!(built.evaluation().is_none());
        assert!(built.thresholds().is_none() && built.thresholds_path().is_none());

        assert_eq!(built.path(), scratch.out().join("waf.s2fp"));
        assert_eq!(scratch.written(), ["waf.s2fp"]);
        let bytes = fs::read(built.path()).unwrap();
        assert_eq!(bytes, built.model().to_bytes());
        assert_eq!(&Model::from_bytes(&bytes).unwrap(), built.model());

        let directives: Vec<&str> = lines
            .iter()
            .map(String::as_str)
            .filter(|line| line.starts_with("cargo::"))
            .collect();
        assert_eq!(
            directives,
            [
                "cargo::rerun-if-env-changed=STRING2FINGERPRINT_MODEL_WAF".to_owned(),
                "cargo::rerun-if-env-changed=STRING2FINGERPRINT_TRAIN_THREADS".to_owned(),
                format!("cargo::rerun-if-changed={}", benign.display()),
                format!("cargo::rerun-if-changed={}", labeled.display()),
            ]
        );
        assert_eq!(lines.len(), directives.len() + 1);
        assert!(lines[4].starts_with("string2fingerprint: trained model `waf` on 80 samples"));
    }

    #[test]
    fn the_output_directory_defaults_to_out_dir() {
        let scratch = Scratch::new("outdir");
        let mut build = build(&scratch, "m", 20);
        build.out_dir = None;
        let (result, _) = execute(build.clone(), &[]);
        assert!(matches!(result, Err(BuildError::MissingOutDir)));
        let out = scratch.out();
        let (result, _) = execute(build, &[("OUT_DIR", out.to_str().unwrap())]);
        assert_eq!(result.unwrap().path(), out.join("m.s2fp"));
    }

    #[test]
    fn held_out_and_validation_samples_are_not_trained_on() {
        let scratch = Scratch::new("heldout");
        let extra = scratch.file(
            "validation.tsv",
            "benign\t/items/900?page=1&sort=name\n\
             sql_injection\t/items/901' UNION SELECT 901, password FROM users--\n",
        );
        let fallback = Thresholds::uniform(Threshold::at(200));
        let objective = Objective::MinCost {
            false_positive: 1.0,
            false_negative: 1.0,
        };
        let built = run(build(&scratch, "waf", 40)
            .validate_on(Source::labeled_lines(&extra).weighted(7.0))
            .hold_out(0.25)
            .calibrate(objective, fallback));

        let evaluation = built.evaluation().unwrap();
        assert_eq!(evaluation.samples(), 20 + 2);
        assert_eq!(built.training_samples(), 60);

        // The thresholds are those of the evaluation.
        let thresholds = built.thresholds().unwrap();
        assert_eq!(thresholds, evaluation.thresholds(objective, fallback));
        assert_ne!(thresholds, fallback);
        assert_eq!(scratch.written(), ["waf.s2fp", "waf.thresholds"]);
        let path = built.thresholds_path().unwrap();
        assert_eq!(path, scratch.out().join("waf.thresholds"));
        let bytes: [u8; Thresholds::BYTES] = fs::read(path).unwrap().try_into().unwrap();
        assert_eq!(Thresholds::from_bytes(&bytes), Some(thresholds));

        // The model is the one that training on the rest alone gives.
        let (benign, labeled) = scratch.samples(40);
        let mut data = Dataset::new();
        data.extend_from_lines(&fs::read(benign).unwrap(), CategorySet::EMPTY)
            .unwrap();
        data.extend_from_labeled_lines(&fs::read(labeled).unwrap())
            .unwrap();
        let (rest, held_out) = data.split(0.25, config().seed);
        assert_eq!((rest.samples(), held_out.samples()), (60, 20));
        let trained = Trainer::new(config()).unwrap().train(&rest).unwrap();
        assert_eq!(built.model(), trained.model());
    }

    #[test]
    fn source_weights_reach_the_trainer() {
        let scratch = Scratch::new("weights");
        let (benign, labeled) = scratch.samples(30);
        let weighted = |weight: f32| {
            run(ModelBuild::new("w")
                .train_on(Source::benign_lines(&benign))
                .train_on(Source::labeled_lines(&labeled).weighted(weight))
                .validate_on(Source::labeled_lines(&labeled))
                .config(config())
                .out_dir(scratch.out()))
        };
        let recall = |built: &BuiltModel| {
            built
                .evaluation()
                .unwrap()
                .confusion(Category::SqlInjection, Threshold::at(128))
                .recall()
                .unwrap()
        };
        assert!(recall(&weighted(1.0)) > 0.9);
        // Attacks of weight 0 teach nothing.
        assert_eq!(recall(&weighted(0.0)), 0.0);
        assert_ne!(weighted(1.0).model(), weighted(2.0).model());

        let same_labels = run(ModelBuild::new("w")
            .train_on(Source::benign_lines(&benign))
            .train_on(Source::lines(&benign, Category::Malformed))
            .config(config())
            .out_dir(scratch.out()));
        assert_eq!(same_labels.training_samples(), 60);
    }

    #[test]
    fn a_model_file_replaces_training() {
        let scratch = Scratch::new("prebuilt");
        let compact = scratch.file("compact.s2fp", EmbeddedModel::Compact.bytes());
        let compact = compact.to_str().unwrap();
        let variables = [("STRING2FINGERPRINT_MODEL_MY_WAF2", compact)];

        // The training sources are not read, so they need not exist.
        let missing = ModelBuild::new("my_waf2")
            .train_on(Source::corpus(scratch.0.join("missing.bin")))
            .out_dir(scratch.out());
        let (result, lines) = execute(missing.clone(), &variables);
        let built = result.unwrap();
        assert_eq!(built.model(), &EmbeddedModel::Compact.load());
        assert!(built.report().is_none());
        assert_eq!(built.training_samples(), 0);
        assert_eq!(
            fs::read(scratch.out().join("my_waf2.s2fp")).unwrap(),
            EmbeddedModel::Compact.bytes()
        );
        assert!(lines.contains(&format!("cargo::rerun-if-changed={compact}")));
        assert!(
            lines.contains(&"cargo::rerun-if-env-changed=STRING2FINGERPRINT_MODEL_MY_WAF2".into())
        );
        // The replacement is never silent.
        assert!(lines.contains(&format!(
            "cargo::warning=the model `my_waf2` is the file {compact} that \
             `STRING2FINGERPRINT_MODEL_MY_WAF2` names, not a model trained by this build"
        )));

        // An empty variable is like none, and held-out samples need the
        // sources even with a model file.
        let (result, _) = execute(missing.clone(), &[("STRING2FINGERPRINT_MODEL_MY_WAF2", "")]);
        assert!(matches!(result, Err(BuildError::Io { .. })));
        let (result, _) = execute(missing.clone().hold_out(0.5), &variables);
        assert!(matches!(result, Err(BuildError::Io { .. })));

        // Validation and thresholds work with a model file.
        let (_, labeled) = scratch.samples(10);
        let fallback = Thresholds::uniform(Threshold::NEVER);
        let (result, _) = execute(
            missing
                .clone()
                .validate_on(Source::labeled_lines(labeled))
                .calibrate(Objective::MinRecall(1.0), fallback),
            &variables,
        );
        let built = result.unwrap();
        assert_eq!(built.evaluation().unwrap().samples(), 10);
        let thresholds = built.thresholds().unwrap();
        assert_ne!(thresholds.get(Category::SqlInjection), Threshold::NEVER);
        assert_eq!(thresholds.get(Category::Malformed), Threshold::NEVER);

        // A file that is not a model fails the build.
        let broken = scratch.file("broken.s2fp", b"S2FP");
        let variables = [("STRING2FINGERPRINT_MODEL_MY_WAF2", broken.to_str().unwrap())];
        let (result, _) = execute(missing, &variables);
        assert!(matches!(result, Err(BuildError::Model { path, .. }) if path == broken));
    }

    #[test]
    fn a_build_without_thresholds_removes_those_of_an_earlier_build() {
        let scratch = Scratch::new("stale");
        let fallback = Thresholds::uniform(Threshold::at(200));
        let calibrated = build(&scratch, "waf", 20)
            .hold_out(0.25)
            .calibrate(Objective::MinRecall(1.0), fallback);
        run(calibrated);
        assert_eq!(scratch.written(), ["waf.s2fp", "waf.thresholds"]);
        // Thresholds of other models stay.
        fs::write(scratch.out().join("other.thresholds"), [0; 16]).unwrap();

        let built = run(build(&scratch, "waf", 20));
        assert!(built.thresholds().is_none() && built.thresholds_path().is_none());
        assert_eq!(scratch.written(), ["other.thresholds", "waf.s2fp"]);
        // Nothing to remove is not an error either.
        run(build(&scratch, "waf", 20));
        assert_eq!(scratch.written(), ["other.thresholds", "waf.s2fp"]);
    }

    #[test]
    fn fine_tuning_continues_from_the_base_model() {
        let scratch = Scratch::new("finetune");
        let base = run(build(&scratch, "base", 40)).model().clone();
        let corrections = scratch.file(
            "corrections.txt",
            "write ' UNION SELECT between two queries\n".repeat(8),
        );
        let tune = ModelBuild::new("tuned")
            .train_on(Source::benign_lines(&corrections))
            .fine_tune(base.clone())
            .out_dir(scratch.out());
        let built = run(tune);
        let mut data = Dataset::new();
        data.extend_from_lines(&fs::read(corrections).unwrap(), CategorySet::EMPTY)
            .unwrap();
        let tuned = Trainer::new(TrainConfig::fine_tuning())
            .unwrap()
            .fine_tune(&base, &data)
            .unwrap();
        assert_eq!(built.model(), tuned.model());
        assert_ne!(built.model(), &base);
        assert_eq!(
            built.model().layer_sizes().collect::<Vec<_>>(),
            base.layer_sizes().collect::<Vec<_>>()
        );
    }

    #[test]
    fn threads_follow_the_variable_then_the_setting_then_the_jobs() {
        let resolve = |setting, variables: Variables<'_>| {
            let variables: HashMap<&str, &str> = variables.iter().copied().collect();
            threads(setting, &|name| variables.get(name).map(OsString::from)).ok()
        };
        assert_eq!(resolve(0, &[]), Some(0));
        assert_eq!(resolve(0, &[("NUM_JOBS", "6")]), Some(6));
        assert_eq!(resolve(0, &[("NUM_JOBS", "many")]), Some(0));
        assert_eq!(resolve(2, &[("NUM_JOBS", "6")]), Some(2));
        let both = [("NUM_JOBS", "6"), (THREADS_VARIABLE, " 5 ")];
        assert_eq!(resolve(2, &both), Some(5));
        assert_eq!(resolve(2, &[(THREADS_VARIABLE, "")]), Some(2));
        for invalid in ["0", "-1", "two", "1.5"] {
            assert_eq!(resolve(2, &[(THREADS_VARIABLE, invalid)]), None);
        }

        // The model does not depend on the number of threads.
        let scratch = Scratch::new("threads");
        let one = execute(build(&scratch, "t", 30), &[(THREADS_VARIABLE, "1")]);
        let five = execute(build(&scratch, "t", 30), &[(THREADS_VARIABLE, "5")]);
        assert_eq!(one.0.unwrap().model(), five.0.unwrap().model());
    }

    #[test]
    fn invalid_builds_are_refused_and_write_nothing() {
        let scratch = Scratch::new("invalid");
        let long = "n".repeat(MAX_NAME_LEN + 1);
        for name in [
            "",
            "a/b",
            "a.b",
            "..",
            "a b",
            "mod\u{e8}le",
            "Waf",
            "my-waf",
            long.as_str(),
        ] {
            let (result, lines) = execute(build(&scratch, name, 4), &[]);
            assert!(matches!(result, Err(BuildError::InvalidName)), "{name:?}");
            assert!(lines.is_empty());
        }
        let (result, _) = execute(build(&scratch, &long[1..], 20), &[]);
        assert!(result.is_ok());
        fs::remove_file(scratch.out().join(format!("{}.s2fp", &long[1..]))).unwrap();

        let refused = |build: ModelBuild, variables: Variables<'_>| {
            let error = execute(build, variables).0.unwrap_err();
            assert!(scratch.written().is_empty(), "{error}");
            error
        };
        for fraction in [-0.1, 1.0, 1.5, f64::NAN, f64::INFINITY] {
            let error = refused(build(&scratch, "m", 4).hold_out(fraction), &[]);
            assert!(matches!(error, BuildError::InvalidSetting(_)), "{fraction}");
        }
        let error = refused(build(&scratch, "m", 4), &[(THREADS_VARIABLE, "none")]);
        assert!(matches!(error, BuildError::InvalidSetting(_)));

        // An objective that no threshold can meet would keep every
        // fallback, so it is refused like another setting.
        let fallback = Thresholds::uniform(Threshold::at(128));
        for objective in [
            Objective::MinRecall(90.0),
            Objective::MaxFalsePositiveRate(-0.1),
            Objective::MaxFalsePositiveRate(f64::NAN),
            Objective::MinCost {
                false_positive: 1.0,
                false_negative: f64::INFINITY,
            },
        ] {
            let build = build(&scratch, "m", 4).hold_out(0.5);
            let error = refused(build.calibrate(objective, fallback), &[]);
            assert!(
                matches!(error, BuildError::InvalidSetting(_)),
                "{objective:?}"
            );
        }

        // A path that Cargo could not read in a directive, whether it names
        // a source or comes from the variable, is refused before any
        // directive is printed.
        use std::os::unix::ffi::OsStrExt;
        let not_utf8 = PathBuf::from(std::ffi::OsStr::from_bytes(b"data/benign\xff.txt"));
        for path in [
            PathBuf::from("data/a\ncargo::warning=injected.txt"),
            PathBuf::from("data/a\r.txt"),
            not_utf8,
        ] {
            let (result, lines) = execute(
                build(&scratch, "m", 4).train_on(Source::benign_lines(&path)),
                &[],
            );
            assert!(
                matches!(result, Err(BuildError::InvalidSetting(_))),
                "{path:?}"
            );
            assert!(lines.is_empty());
        }
        let (result, lines) = execute(
            build(&scratch, "m", 4),
            &[("STRING2FINGERPRINT_MODEL_M", "a\ncargo::warning=injected")],
        );
        assert!(matches!(result, Err(BuildError::InvalidSetting(_))));
        assert!(lines.is_empty());

        let mut invalid = config();
        invalid.epochs = 0;
        let error = refused(build(&scratch, "m", 4).config(invalid), &[]);
        assert!(matches!(
            error,
            BuildError::Train(TrainError::InvalidConfig(_))
        ));

        let fallback = Thresholds::uniform(Threshold::at(128));
        let error = refused(
            build(&scratch, "m", 4).calibrate(Objective::MinRecall(0.9), fallback),
            &[],
        );
        assert!(matches!(error, BuildError::NoValidationSamples));

        let empty = ModelBuild::new("m").out_dir(scratch.out());
        let error = refused(empty.clone(), &[]);
        assert!(matches!(error, BuildError::Train(TrainError::EmptyDataset)));

        let missing = scratch.0.join("missing.txt");
        let error = refused(empty.clone().train_on(Source::benign_lines(&missing)), &[]);
        assert!(matches!(&error, BuildError::Io { path, .. } if *path == missing));
        assert!(error.to_string().starts_with(missing.to_str().unwrap()));

        let malformed = scratch.file("malformed.tsv", "benign\tfine\nno tab here\n");
        let error = refused(
            empty.clone().train_on(Source::labeled_lines(&malformed)),
            &[],
        );
        assert!(matches!(
            &error,
            BuildError::Dataset { path, error: DatasetError::InvalidLine(2) } if *path == malformed
        ));
        let truncated = scratch.file("truncated.bin", [9, 0, 0, 0, 1, b'a']);
        let error = refused(empty.clone().validate_on(Source::corpus(&truncated)), &[]);
        assert!(matches!(
            error,
            BuildError::Dataset {
                error: DatasetError::TruncatedCorpus,
                ..
            }
        ));
        let (benign, _) = scratch.samples(4);
        for weight in [-1.0, f32::NAN, f32::INFINITY] {
            let source = Source::benign_lines(&benign).weighted(weight);
            let error = refused(empty.clone().train_on(source), &[]);
            assert!(matches!(
                error,
                BuildError::Dataset {
                    error: DatasetError::InvalidWeight,
                    ..
                }
            ));
        }

        // A directory in place of the output file fails the write.
        fs::create_dir(scratch.out().join("m.s2fp")).unwrap();
        let error = execute(build(&scratch, "m", 20), &[]).0.unwrap_err();
        assert!(
            matches!(&error, BuildError::Io { path, .. } if *path == scratch.out().join("m.s2fp"))
        );
    }

    #[test]
    fn errors_have_fixed_messages_and_sources() {
        use std::error::Error;

        let path = PathBuf::from("data/attacks.tsv");
        let io = BuildError::Io {
            path: path.clone(),
            error: io::Error::from(io::ErrorKind::NotFound),
        };
        assert!(io.to_string().starts_with("data/attacks.tsv: "));
        assert!(io.source().is_some());
        let dataset = BuildError::Dataset {
            path: path.clone(),
            error: DatasetError::InvalidLine(3),
        };
        assert_eq!(
            dataset.to_string(),
            "data/attacks.tsv: line 3 is not labels, a tab, and a payload"
        );
        assert!(dataset.source().is_some());
        let model = BuildError::Model {
            path,
            error: Model::from_bytes(b"").unwrap_err(),
        };
        assert!(model.to_string().starts_with("data/attacks.tsv: "));
        assert!(model.source().is_some());
        let train = BuildError::from(TrainError::EmptyDataset);
        assert_eq!(train.to_string(), "dataset has no window to train on");
        assert!(train.source().is_some());
        for (error, message) in [
            (
                BuildError::InvalidName,
                "model name must be 1 to 64 lowercase ASCII letters, digits, or underscores",
            ),
            (
                BuildError::MissingOutDir,
                "no output directory: `OUT_DIR` is not set outside a build script",
            ),
            (
                BuildError::InvalidSetting("reason"),
                "model build is invalid: reason",
            ),
            (
                BuildError::NoValidationSamples,
                "thresholds need samples that are held out or given for validation",
            ),
        ] {
            assert_eq!(error.to_string(), message);
            assert!(error.source().is_none());
        }
        assert_eq!(
            model_variable("my_waf_2"),
            "STRING2FINGERPRINT_MODEL_MY_WAF_2"
        );
        assert_eq!(Source::corpus("a/b.bin").path(), Path::new("a/b.bin"));
    }
}
