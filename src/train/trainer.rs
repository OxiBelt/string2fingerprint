//! The training loop.

use core::fmt;
use std::thread;

use super::dataset::Dataset;
use super::math::{Rng, cos, sigmoid};
use super::network::{self, Adam, Network, Pass, QuantizedWeights, SHARDS, Shard, batch_gradients};
use super::quantize::quantize;
use crate::Category;
use crate::backend::Verified;
use crate::fingerprint::FINGERPRINT_LEN;
use crate::model::{EmbeddedModel, MAX_HIDDEN, MAX_LAYERS, Model, ModelError};

/// The settings of a training run.
///
/// The defaults are those of the embedded default model.
/// [`TrainConfig::preset`] returns the settings of each embedded model.
///
/// ```
/// use string2fingerprint::train::TrainConfig;
///
/// let mut config = TrainConfig::default();
/// config.hidden = vec![32];
/// config.epochs = 8;
/// ```
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct TrainConfig {
    /// The number of units of each hidden layer, from the first. At most
    /// three layers of 1 to [`MAX_HIDDEN`] units. Without a hidden layer the
    /// model is linear.
    pub hidden: Vec<usize>,
    /// The number of passes over the data, at least 1.
    pub epochs: u32,
    /// The number of last epochs in which the forward pass rounds weights
    /// and activations as the quantized model does, so that the network
    /// adapts to quantization. At most `epochs`.
    pub quantized_epochs: u32,
    /// The number of windows whose gradients one step adds up, at least 1.
    pub batch_size: usize,
    /// The step size at the start. It falls to zero along a cosine.
    pub learning_rate: f32,
    /// The share of each weight that a step of size 1 removes. A step
    /// removes the share `learning_rate * weight_decay`, which must be
    /// below 1.
    pub weight_decay: f32,
    /// The share of features that are set to zero at random in each window,
    /// from 0.0 to less than 1.0. It makes the model rely on more than a few
    /// features.
    pub input_dropout: f32,
    /// The seed of the initial weights, the order of the windows, and the
    /// dropped features.
    pub seed: u64,
    /// The number of threads, or 0 for the number that the host offers. At
    /// most 8 are used. The trained model does not depend on it.
    pub threads: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            hidden: vec![64, 64],
            epochs: 16,
            quantized_epochs: 3,
            batch_size: 256,
            learning_rate: 2e-3,
            weight_decay: 1e-4,
            input_dropout: 0.1,
            seed: 1,
            threads: 0,
        }
    }
}

impl TrainConfig {
    /// Returns the settings with which the training tools in `tools/train/`
    /// build an embedded model.
    ///
    /// The settings give a model of the same shape. They give the embedded
    /// model itself only together with the data of those tools.
    #[must_use]
    pub fn preset(model: EmbeddedModel) -> Self {
        let hidden = match model {
            EmbeddedModel::Compact => vec![32],
            EmbeddedModel::Large => vec![192, 96],
            _ => vec![64, 64],
        };
        Self {
            hidden,
            ..Self::default()
        }
    }

    /// Returns why the settings are not usable, if they are not.
    fn validate(&self) -> Result<(), TrainError> {
        let invalid = |reason| Err(TrainError::InvalidConfig(reason));
        if self.hidden.len() >= MAX_LAYERS {
            return invalid("too many hidden layers");
        }
        if self
            .hidden
            .iter()
            .any(|units| !(1..=MAX_HIDDEN).contains(units))
        {
            return invalid("hidden layer size is out of range");
        }
        if self.epochs == 0 {
            return invalid("no epochs");
        }
        if self.quantized_epochs > self.epochs {
            return invalid("more quantized epochs than epochs");
        }
        if self.batch_size == 0 {
            return invalid("batch size is zero");
        }
        if !(self.learning_rate.is_finite() && self.learning_rate > 0.0) {
            return invalid("learning rate is not positive and finite");
        }
        if !(self.weight_decay.is_finite() && self.weight_decay >= 0.0) {
            return invalid("weight decay is negative or not finite");
        }
        if self.learning_rate * self.weight_decay >= 1.0 {
            return invalid("learning rate times weight decay is not below 1");
        }
        if !(0.0..1.0).contains(&self.input_dropout) {
            return invalid("input dropout is not within 0.0 to 1.0");
        }
        Ok(())
    }
}

/// The reason training did not produce a model.
///
/// The error never contains sample bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum TrainError {
    /// A setting is out of range. The value names the setting.
    InvalidConfig(&'static str),
    /// The dataset has no window.
    EmptyDataset,
    /// A parameter became infinite or undefined, or too large for the model
    /// format. A lower learning rate usually avoids this.
    Diverged,
    /// The quantized parameters are not a valid model.
    Model(ModelError),
}

impl fmt::Display for TrainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(reason) => write!(f, "training configuration is invalid: {reason}"),
            Self::EmptyDataset => f.write_str("dataset has no window to train on"),
            Self::Diverged => f.write_str("training diverged"),
            Self::Model(error) => write!(f, "trained parameters are not a valid model: {error}"),
        }
    }
}

impl core::error::Error for TrainError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Model(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ModelError> for TrainError {
    fn from(error: ModelError) -> Self {
        Self::Model(error)
    }
}

/// The progress of training after one epoch.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct EpochReport {
    /// The number of the epoch, from 1.
    pub epoch: u32,
    /// The number of epochs of the run.
    pub epochs: u32,
    /// Whether the epoch rounded weights and activations.
    pub quantized: bool,
    /// The mean loss of the epoch over every window and category: the
    /// binary cross-entropy of the forward pass that training ran, in nats.
    /// It includes the dropped features, and in a quantized epoch the
    /// rounding.
    pub loss: f64,
}

/// What a training run measured.
#[derive(Clone, Debug, PartialEq)]
pub struct TrainReport {
    epochs: Vec<EpochReport>,
    windows: usize,
    drift: (f64, f64),
}

impl TrainReport {
    /// Returns the report of every epoch, in order.
    #[must_use]
    pub fn epochs(&self) -> &[EpochReport] {
        &self.epochs
    }

    /// Returns the mean loss of the last epoch.
    #[must_use]
    pub fn final_loss(&self) -> f64 {
        self.epochs.last().map_or(f64::NAN, |epoch| epoch.loss)
    }

    /// Returns the number of windows that each epoch trained on.
    #[must_use]
    pub fn windows(&self) -> usize {
        self.windows
    }

    /// Returns the mean and the largest difference between a score of the
    /// quantized model and the score that the float network gives, over the
    /// first windows of the dataset, at most 4096.
    ///
    /// A mean of about one score step is usual. A much larger value says
    /// that the model lost accuracy when it was quantized, which more
    /// [quantized epochs](TrainConfig::quantized_epochs) reduce.
    #[must_use]
    pub fn quantization_drift(&self) -> (f64, f64) {
        self.drift
    }
}

/// The result of training: a model and the report of the run.
#[derive(Clone, Debug)]
pub struct Trained {
    model: Model,
    report: TrainReport,
}

impl Trained {
    /// Returns the quantized model.
    #[must_use]
    pub fn model(&self) -> &Model {
        &self.model
    }

    /// Returns the quantized model and drops the report.
    #[must_use]
    pub fn into_model(self) -> Model {
        self.model
    }

    /// Returns the report of the run.
    #[must_use]
    pub fn report(&self) -> &TrainReport {
        &self.report
    }
}

/// Trains models.
///
/// ```
/// use string2fingerprint::train::{Dataset, Sample, TrainConfig, Trainer};
/// use string2fingerprint::{Category, Scorer};
///
/// let mut data = Dataset::new();
/// for id in 0..40 {
///     let benign = format!("/items/{id}?page={}&sort=name", id % 7);
///     let attack = format!("/items/{id}' UNION SELECT {id}, password FROM users--");
///     data.push(Sample::benign(benign.as_bytes()))?;
///     data.push(Sample::new(attack.as_bytes(), Category::SqlInjection))?;
/// }
///
/// let mut config = TrainConfig::default();
/// config.hidden = vec![8];
/// config.epochs = 12;
/// config.batch_size = 16;
/// let trained = Trainer::new(config)?.train(&data)?;
///
/// let scorer = Scorer::with_model(trained.into_model());
/// let attack = scorer.score(b"/items/77' UNION SELECT 1, password FROM users--");
/// let benign = scorer.score(b"/items/77?page=3&sort=name");
/// assert!(attack.get(Category::SqlInjection) > benign.get(Category::SqlInjection));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// Training is deterministic: the same data, in the same order, with the
/// same settings gives the same model bytes, whatever the number of threads
/// and on every supported host. The arithmetic uses only operations that
/// IEEE 754 rounds exactly.
#[derive(Clone, Debug)]
pub struct Trainer {
    config: TrainConfig,
}

impl Trainer {
    /// Creates a trainer.
    ///
    /// # Errors
    ///
    /// Returns [`TrainError::InvalidConfig`] when a setting is out of the
    /// range that [`TrainConfig`] documents.
    pub fn new(config: TrainConfig) -> Result<Self, TrainError> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Returns the settings.
    #[must_use]
    pub fn config(&self) -> &TrainConfig {
        &self.config
    }

    /// Trains a model on `data` from random weights.
    ///
    /// # Errors
    ///
    /// Returns [`TrainError::EmptyDataset`] when `data` has no window, and
    /// [`TrainError::Diverged`] when the parameters grew without bound.
    pub fn train(&self, data: &Dataset) -> Result<Trained, TrainError> {
        self.train_with_progress(data, |_| {})
    }

    /// Trains like [`Trainer::train`] and calls `progress` after each
    /// epoch.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Trainer::train`].
    pub fn train_with_progress(
        &self,
        data: &Dataset,
        progress: impl FnMut(&EpochReport),
    ) -> Result<Trained, TrainError> {
        let mut dims = Vec::with_capacity(self.config.hidden.len() + 2);
        dims.push(FINGERPRINT_LEN);
        dims.extend(&self.config.hidden);
        dims.push(Category::COUNT);
        let mut rng = Rng::new(self.config.seed);
        let network = Network::random(&mut rng, &dims);
        self.run(network, data, &rng, progress)
    }

    /// Optimizes `network` on `data` and quantizes it.
    fn run(
        &self,
        mut network: Network,
        data: &Dataset,
        rng: &Rng,
        mut progress: impl FnMut(&EpochReport),
    ) -> Result<Trained, TrainError> {
        let config = &self.config;
        let windows = data.windows();
        if windows == 0 {
            return Err(TrainError::EmptyDataset);
        }
        let threads = match config.threads {
            0 => thread::available_parallelism().map_or(1, usize::from),
            threads => threads,
        }
        .min(SHARDS);
        let batch_size = config.batch_size.min(windows);
        let steps_per_epoch = windows.div_ceil(batch_size);
        let total_steps = u64::from(config.epochs) * steps_per_epoch as u64;

        let mut adam = Adam::new(&network.dims);
        let mut quantized_weights = QuantizedWeights::new(&network.dims);
        let mut shards: Vec<Shard> = (0..SHARDS).map(|_| Shard::new(&network.dims)).collect();
        let mut order: Vec<usize> = (0..windows).collect();
        let mut epochs = Vec::new();
        let mut step = 0u64;
        for epoch in 0..config.epochs {
            let quantized = epoch >= config.epochs - config.quantized_epochs;
            let mut epoch_rng = rng.fork(u64::from(epoch));
            epoch_rng.shuffle(&mut order);
            let mut loss = 0.0;
            for (batch, batch_windows) in order.chunks(batch_size).enumerate() {
                if quantized {
                    quantized_weights.update(&network);
                }
                let pass = Pass {
                    data,
                    windows: batch_windows,
                    position: batch * batch_size,
                    quantized,
                    dropout: config.input_dropout,
                    rng: &epoch_rng,
                    threads,
                };
                let forward_weights = quantized.then_some(&quantized_weights);
                let gradients = batch_gradients(&network, forward_weights, &pass, &mut shards);
                loss += network::loss(gradients);
                step += 1;
                let phase = core::f64::consts::PI * step as f64 / total_steps as f64;
                let rate = f64::from(config.learning_rate) * 0.5 * (1.0 + cos(phase));
                adam.step(
                    &mut network,
                    gradients,
                    rate as f32,
                    config.weight_decay,
                    threads,
                );
            }
            if !network.is_finite() {
                return Err(TrainError::Diverged);
            }
            let report = EpochReport {
                epoch: epoch + 1,
                epochs: config.epochs,
                quantized,
                loss: loss / (windows * Category::COUNT) as f64,
            };
            progress(&report);
            epochs.push(report);
        }

        let layers = quantize(&network).ok_or(TrainError::Diverged)?;
        let model = Model::from_layers(layers)?;
        let drift = quantization_drift(&network, &model, data);
        Ok(Trained {
            model,
            report: TrainReport {
                epochs,
                windows,
                drift,
            },
        })
    }
}

/// Returns the mean and the largest difference between the scores of
/// `model` and those of `network` over the first windows of `data`.
fn quantization_drift(network: &Network, model: &Model, data: &Dataset) -> (f64, f64) {
    const WINDOWS: usize = 4096;
    let windows = data.windows().min(WINDOWS);
    let (mut sum, mut largest) = (0.0f64, 0.0f64);
    for window in 0..windows {
        let logits = network.logits(data.window(window));
        let scores = model.infer(Verified::SCALAR, &data.fingerprint(window));
        for (&logit, &score) in logits.iter().zip(scores.as_array()) {
            let difference = (255.0 * sigmoid(f64::from(logit)) - f64::from(score)).abs();
            sum += difference;
            largest = largest.max(difference);
        }
    }
    (sum / (windows.max(1) * Category::COUNT) as f64, largest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::dataset::Sample;
    use crate::{CategorySet, Scorer};

    /// Benign and attack strings that differ in structure, with some
    /// variety, so that a small network separates them.
    fn dataset(count: usize) -> Dataset {
        let mut data = Dataset::new();
        let mut rng = Rng::new(11);
        let words = [
            "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
        ];
        for index in 0..count {
            let word = words[rng.below(words.len())];
            let number = rng.below(10_000);
            let benign = format!(
                "/api/v{}/{word}/{number}?sort={word}&page={}",
                index % 4,
                index % 9
            );
            data.push(Sample::benign(benign.as_bytes())).unwrap();
            let sql = format!("{word}=' OR {number}={number} UNION SELECT {word} FROM users--");
            data.push(Sample::new(sql.as_bytes(), Category::SqlInjection))
                .unwrap();
            let path = format!("/static/{word}/../../../../etc/passwd%00.{number}.png");
            data.push(Sample::new(path.as_bytes(), Category::PathTraversal))
                .unwrap();
        }
        data
    }

    fn config(hidden: &[usize]) -> TrainConfig {
        TrainConfig {
            hidden: hidden.to_vec(),
            epochs: 24,
            quantized_epochs: 4,
            batch_size: 16,
            learning_rate: 1e-2,
            threads: 1,
            ..TrainConfig::default()
        }
    }

    #[test]
    fn a_trained_model_separates_its_training_data() {
        let data = dataset(60);
        for hidden in [&[][..], &[12], &[10, 6]] {
            let trained = Trainer::new(config(hidden)).unwrap().train(&data).unwrap();
            let report = trained.report();
            assert_eq!(report.epochs().len(), 24);
            assert_eq!(report.windows(), data.windows());
            assert!(
                report.final_loss() < 0.5 * report.epochs()[0].loss,
                "hidden {hidden:?}"
            );
            assert!(report.epochs()[..20].iter().all(|epoch| !epoch.quantized));
            assert!(report.epochs()[20..].iter().all(|epoch| epoch.quantized));
            let (mean, largest) = report.quantization_drift();
            assert!(
                mean < 4.0 && largest >= mean,
                "hidden {hidden:?}: drift {mean}, {largest}"
            );

            let mut layers = hidden.to_vec();
            layers.push(Category::COUNT);
            assert_eq!(trained.model().layer_sizes().collect::<Vec<_>>(), layers);
            let scorer = Scorer::with_model(trained.into_model());
            let mut wrong = 0;
            for (labels, scores) in data.scores(&scorer) {
                wrong += usize::from(scores.at_least(128) != labels);
            }
            assert!(
                wrong * 20 <= data.samples(),
                "hidden {hidden:?}: {wrong} samples wrong"
            );
            assert!(
                !scorer
                    .score(b"/api/v9/hotel/31?sort=india&page=4")
                    .any_at_least(128)
            );
            let unseen = scorer.score(b"kilo=' OR 31=31 UNION SELECT lima FROM users--");
            assert_eq!(
                unseen.at_least(128),
                CategorySet::from(Category::SqlInjection)
            );
        }
    }

    #[test]
    fn training_is_deterministic_and_independent_of_threads() {
        let data = dataset(30);
        let train = |threads: usize, seed: u64| {
            let config = TrainConfig {
                threads,
                seed,
                ..config(&[9])
            };
            Trainer::new(config).unwrap().train(&data).unwrap()
        };
        let reference = train(1, 1);
        assert_eq!(train(1, 1).model(), reference.model());
        for threads in [0, 2, 5, 8, 200] {
            let other = train(threads, 1);
            assert_eq!(other.model(), reference.model(), "{threads} threads");
            assert_eq!(other.report(), reference.report(), "{threads} threads");
        }
        assert_ne!(train(1, 2).model(), reference.model());
    }

    /// The model bytes of a fixed run. Every supported architecture must
    /// give this value, because training uses only exactly rounded
    /// operations. A deliberate change to the trainer changes it.
    #[test]
    fn training_gives_the_same_model_on_every_host() {
        let data = dataset(20);
        let trained = Trainer::new(config(&[6, 5])).unwrap().train(&data).unwrap();
        let bytes = trained.model().to_bytes();
        let checksum = u32::from_le_bytes(bytes[bytes.len() - 4..].try_into().unwrap());
        assert_eq!(checksum, 0x8BA0_925D, "model checksum {checksum:#010x}");
    }

    #[test]
    fn progress_reports_every_epoch() {
        let data = dataset(10);
        let mut seen = Vec::new();
        let trained = Trainer::new(config(&[4]))
            .unwrap()
            .train_with_progress(&data, |report| seen.push(*report))
            .unwrap();
        assert_eq!(seen, trained.report().epochs());
        assert_eq!(
            seen.iter().map(|report| report.epoch).collect::<Vec<_>>(),
            (1..=24).collect::<Vec<_>>()
        );
        assert!(
            seen.iter()
                .all(|report| report.epochs == 24 && report.loss.is_finite())
        );
    }

    #[test]
    fn a_batch_larger_than_the_data_and_a_last_partial_batch_work() {
        let data = dataset(5);
        for batch_size in [1, 4, 7, 15, 16, 10_000] {
            let config = TrainConfig {
                batch_size,
                epochs: 2,
                quantized_epochs: 1,
                ..config(&[4])
            };
            let trained = Trainer::new(config).unwrap().train(&data).unwrap();
            assert!(
                trained.report().final_loss().is_finite(),
                "batch size {batch_size}"
            );
        }
    }

    #[test]
    fn invalid_settings_and_empty_data_are_rejected() {
        let invalid = |edit: &dyn Fn(&mut TrainConfig)| {
            let mut config = TrainConfig::default();
            edit(&mut config);
            matches!(Trainer::new(config), Err(TrainError::InvalidConfig(_)))
        };
        assert!(invalid(&|config| config.hidden = vec![8; MAX_LAYERS]));
        assert!(invalid(&|config| config.hidden = vec![0]));
        assert!(invalid(&|config| config.hidden = vec![MAX_HIDDEN + 1]));
        assert!(invalid(&|config| config.epochs = 0));
        assert!(invalid(
            &|config| config.quantized_epochs = config.epochs + 1
        ));
        assert!(invalid(&|config| config.batch_size = 0));
        for rate in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(invalid(&|config| config.learning_rate = rate));
        }
        for decay in [-1e-9, f32::NAN, f32::INFINITY] {
            assert!(invalid(&|config| config.weight_decay = decay));
        }
        // A step that removes all of a weight or more overshoots.
        assert!(invalid(&|config| config.weight_decay = 500.0));
        assert!(invalid(&|config| {
            config.learning_rate = 0.5;
            config.weight_decay = 2.0;
        }));
        assert!(!invalid(&|config| config.weight_decay = 499.0));
        for dropout in [-0.1, 1.0, f32::NAN] {
            assert!(invalid(&|config| config.input_dropout = dropout));
        }
        assert!(!invalid(
            &|config| config.hidden = vec![MAX_HIDDEN; MAX_LAYERS - 1]
        ));
        assert!(!invalid(&|config| config.hidden.clear()));
        assert!(!invalid(&|config| config.quantized_epochs = config.epochs));

        let trainer = Trainer::new(TrainConfig::default()).unwrap();
        assert_eq!(trainer.config(), &TrainConfig::default());
        assert_eq!(
            trainer.train(&Dataset::new()).err(),
            Some(TrainError::EmptyDataset)
        );
    }

    #[test]
    fn a_learning_rate_that_is_far_too_high_diverges_or_still_gives_a_valid_model() {
        let data = dataset(10);
        let config = TrainConfig {
            learning_rate: 1e30,
            weight_decay: 0.0,
            input_dropout: 0.0,
            ..config(&[4])
        };
        match Trainer::new(config).unwrap().train(&data) {
            Err(error) => assert_eq!(error, TrainError::Diverged),
            Ok(trained) => {
                let bytes = trained.model().to_bytes();
                assert_eq!(Model::from_bytes(&bytes).as_ref(), Ok(trained.model()));
            }
        }
    }

    #[test]
    fn presets_have_the_shapes_of_the_embedded_models() {
        for embedded in EmbeddedModel::ALL {
            let config = TrainConfig::preset(embedded);
            let mut sizes = config.hidden.clone();
            sizes.push(Category::COUNT);
            assert_eq!(
                embedded.load().layer_sizes().collect::<Vec<_>>(),
                sizes,
                "{embedded}"
            );
            assert!(Trainer::new(config).is_ok());
        }
    }

    #[test]
    fn errors_have_fixed_messages() {
        assert_eq!(
            TrainError::InvalidConfig("no epochs").to_string(),
            "training configuration is invalid: no epochs"
        );
        assert_eq!(
            TrainError::EmptyDataset.to_string(),
            "dataset has no window to train on"
        );
        assert_eq!(TrainError::Diverged.to_string(), "training diverged");
        let error = TrainError::from(ModelError::AccumulatorOverflow);
        assert_eq!(
            error.to_string(),
            "trained parameters are not a valid model: model layer can overflow its accumulator"
        );
        assert!(core::error::Error::source(&error).is_some());
        assert!(core::error::Error::source(&TrainError::Diverged).is_none());
    }
}
