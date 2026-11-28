//! The float network that training optimizes, and its gradients.
//!
//! The network has the shape of the quantized one: fully connected layers
//! whose hidden activations are clamped to the range from 0 to 1, which is
//! the range from 0 to 255 of the stored activations. The output layer
//! returns logits.
//!
//! A fingerprint has few features that are not zero, so the first layer is
//! stored by input and only the columns of those features are read.

use std::thread;

use super::dataset::Dataset;
use super::math::{Rng, round_ties_even, sigmoid, softplus};
use crate::{Category, CategorySet};

/// The number of parts into which the windows of a batch are divided.
///
/// Each part adds up its gradients on its own, and the parts are then added
/// in order. The result therefore does not depend on how many threads
/// computed the parts.
pub(super) const SHARDS: usize = 8;

/// Layer sizes and parameters.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Network {
    /// The number of features, then the number of outputs of each layer.
    pub(super) dims: Vec<usize>,
    /// The weights of each layer. The first layer is stored by input:
    /// the weight from input `i` to output `o` is at `i * outputs + o`. The
    /// other layers are stored by output, at `o * inputs + i`.
    pub(super) weights: Vec<Vec<f32>>,
    /// The bias of each output of each layer.
    pub(super) biases: Vec<Vec<f32>>,
}

impl Network {
    /// Creates a network with random weights and zero biases.
    pub(super) fn random(rng: &mut Rng, dims: &[usize]) -> Self {
        let mut weights = Vec::with_capacity(dims.len() - 1);
        let mut biases = Vec::with_capacity(dims.len() - 1);
        for pair in dims.windows(2) {
            let (inputs, outputs) = (pair[0], pair[1]);
            // He initialization, which suits the rectifier.
            let scale = (2.0 / inputs as f64).sqrt();
            weights.push(
                (0..inputs * outputs)
                    .map(|_| (rng.normal() * scale) as f32)
                    .collect(),
            );
            biases.push(vec![0.0; outputs]);
        }
        Self {
            dims: dims.to_vec(),
            weights,
            biases,
        }
    }

    /// Returns the number of layers.
    pub(super) fn layers(&self) -> usize {
        self.weights.len()
    }

    /// Returns whether every parameter is finite.
    pub(super) fn is_finite(&self) -> bool {
        self.weights
            .iter()
            .chain(&self.biases)
            .all(|tensor| tensor.iter().all(|value| value.is_finite()))
    }

    /// Returns the logits of one window, given as the indexes and values of
    /// its features that are not zero.
    pub(super) fn logits(&self, features: (&[u16], &[u8])) -> [f32; Category::COUNT] {
        let mut scratch = Scratch::new(&self.dims);
        scratch.inputs.extend(
            features
                .0
                .iter()
                .zip(features.1)
                .map(|(&index, &value)| (index, f32::from(value) / 255.0)),
        );
        let weights: Vec<&[f32]> = self.weights.iter().map(Vec::as_slice).collect();
        forward(&self.dims, &weights, &self.biases, false, &mut scratch);
        let mut logits = [0.0; Category::COUNT];
        logits.copy_from_slice(&scratch.pre[self.layers() - 1]);
        logits
    }
}

/// The weights that the forward pass uses in place of the real ones while
/// training adapts to quantization.
pub(super) struct QuantizedWeights {
    weights: Vec<Vec<f32>>,
    scales: Vec<f32>,
}

impl QuantizedWeights {
    pub(super) fn new(dims: &[usize]) -> Self {
        Self {
            weights: dims
                .windows(2)
                .map(|pair| vec![0.0; pair[0] * pair[1]])
                .collect(),
            scales: Vec::new(),
        }
    }

    /// Rounds the weights of each output to the 255 levels that they have
    /// after quantization.
    pub(super) fn update(&mut self, network: &Network) {
        for (layer, (rounded, weights)) in self.weights.iter_mut().zip(&network.weights).enumerate()
        {
            let (inputs, outputs) = (network.dims[layer], network.dims[layer + 1]);
            // Output `o` owns every element `o` of a group of `outputs`
            // elements in the first layer, and one group of `inputs`
            // elements in the others.
            let by_input = layer == 0;
            self.scales.clear();
            self.scales.resize(outputs, 0.0);
            if by_input {
                for column in weights.chunks_exact(outputs) {
                    for (scale, weight) in self.scales.iter_mut().zip(column) {
                        *scale = scale.max(weight.abs());
                    }
                }
            } else {
                for (scale, row) in self.scales.iter_mut().zip(weights.chunks_exact(inputs)) {
                    *scale = row.iter().fold(0.0, |max, weight| max.max(weight.abs()));
                }
            }
            for scale in &mut self.scales {
                *scale = scale.max(1e-12) / 127.0;
            }
            if by_input {
                for (rounded, column) in rounded
                    .chunks_exact_mut(outputs)
                    .zip(weights.chunks_exact(outputs))
                {
                    for ((rounded, &weight), &scale) in
                        rounded.iter_mut().zip(column).zip(&self.scales)
                    {
                        *rounded = round_ties_even(weight / scale) * scale;
                    }
                }
            } else {
                for ((rounded, row), &scale) in rounded
                    .chunks_exact_mut(inputs)
                    .zip(weights.chunks_exact(inputs))
                    .zip(&self.scales)
                {
                    for (rounded, &weight) in rounded.iter_mut().zip(row) {
                        *rounded = round_ties_even(weight / scale) * scale;
                    }
                }
            }
        }
    }

    fn as_slices(&self) -> Vec<&[f32]> {
        self.weights.iter().map(Vec::as_slice).collect()
    }
}

/// Buffers for the passes over one window.
struct Scratch {
    /// The features that are not zero and were not dropped, scaled to the
    /// range from 0 to 1.
    inputs: Vec<(u16, f32)>,
    /// The values of each layer before the clamp.
    pre: Vec<Vec<f32>>,
    /// The clamped values of each hidden layer.
    activations: Vec<Vec<f32>>,
    /// The gradient of the loss in the values of the current layer.
    delta: Vec<f32>,
    /// The gradient in the values of the layer before it.
    previous: Vec<f32>,
}

impl Scratch {
    fn new(dims: &[usize]) -> Self {
        let widest = dims[1..].iter().copied().max().unwrap_or(0);
        Self {
            inputs: Vec::new(),
            pre: dims[1..]
                .iter()
                .map(|&outputs| vec![0.0; outputs])
                .collect(),
            activations: dims[1..dims.len() - 1]
                .iter()
                .map(|&outputs| vec![0.0; outputs])
                .collect(),
            delta: Vec::with_capacity(widest),
            previous: Vec::with_capacity(widest),
        }
    }
}

/// Computes `scratch.pre` and `scratch.activations` from `scratch.inputs`.
fn forward(
    dims: &[usize],
    weights: &[&[f32]],
    biases: &[Vec<f32>],
    quantized: bool,
    scratch: &mut Scratch,
) {
    let outputs = dims[1];
    let first = &mut scratch.pre[0];
    first.copy_from_slice(&biases[0]);
    for &(index, value) in &scratch.inputs {
        let start = usize::from(index) * outputs;
        let column = &weights[0][start..start + outputs];
        for (sum, &weight) in first.iter_mut().zip(column) {
            *sum += weight * value;
        }
    }
    for layer in 1..weights.len() {
        let inputs = dims[layer];
        let (before, after) = scratch.pre.split_at_mut(layer);
        let activation = &mut scratch.activations[layer - 1];
        for (activation, &pre) in activation.iter_mut().zip(&before[layer - 1]) {
            // The clamp is the rectifier and the saturation of a stored
            // activation.
            let clamped = pre.clamp(0.0, 1.0);
            *activation = if quantized {
                round_ties_even(clamped * 255.0) / 255.0
            } else {
                clamped
            };
        }
        let rows = weights[layer].chunks_exact(inputs);
        for ((sum, row), &bias) in after[0].iter_mut().zip(rows).zip(&biases[layer]) {
            let mut dot = 0.0;
            for (&weight, &activation) in row.iter().zip(activation.iter()) {
                dot += weight * activation;
            }
            *sum = bias + dot;
        }
    }
}

/// The sum of the gradients of some windows, and of their losses.
#[derive(Clone, Debug)]
pub(super) struct Gradients {
    weights: Vec<Vec<f32>>,
    biases: Vec<Vec<f32>>,
    /// The sum of the losses of every category of every window.
    pub(super) loss: f64,
}

impl Gradients {
    pub(super) fn new(dims: &[usize]) -> Self {
        Self {
            weights: dims
                .windows(2)
                .map(|pair| vec![0.0; pair[0] * pair[1]])
                .collect(),
            biases: dims[1..]
                .iter()
                .map(|&outputs| vec![0.0; outputs])
                .collect(),
            loss: 0.0,
        }
    }

    fn clear(&mut self) {
        for tensor in self.weights.iter_mut().chain(&mut self.biases) {
            tensor.fill(0.0);
        }
        self.loss = 0.0;
    }

    #[cfg(test)]
    fn add(&mut self, other: &Self) {
        let tensors = self.weights.iter_mut().chain(&mut self.biases);
        for (tensor, other) in tensors.zip(other.weights.iter().chain(&other.biases)) {
            for (value, &other) in tensor.iter_mut().zip(other) {
                *value += other;
            }
        }
        self.loss += other.loss;
    }
}

/// The settings of one pass over a batch.
pub(super) struct Pass<'a> {
    pub(super) data: &'a Dataset,
    /// The windows of the batch.
    pub(super) windows: &'a [usize],
    /// The position of the first window of the batch in its epoch.
    pub(super) position: usize,
    /// Whether activations are rounded as the quantized network rounds
    /// them.
    pub(super) quantized: bool,
    /// The share of features that are set to zero at random.
    pub(super) dropout: f32,
    /// The generator of the epoch. Each window draws from its own stream.
    pub(super) rng: &'a Rng,
    pub(super) threads: usize,
}

/// One part of a batch: its gradients and its buffers.
pub(super) struct Shard {
    gradients: Gradients,
    scratch: Scratch,
}

impl Shard {
    pub(super) fn new(dims: &[usize]) -> Self {
        Self {
            gradients: Gradients::new(dims),
            scratch: Scratch::new(dims),
        }
    }
}

/// Computes the gradients of the mean loss of a batch.
///
/// Returns the shards that hold the gradients of the parts of the batch.
/// Their sum, in order, is the gradient of the batch.
///
/// The loss of a window is the sum over the categories of the binary
/// cross-entropy of the logit with the label. `forward_weights` are the
/// weights of the forward pass, which differ from those of `network` while
/// training adapts to quantization. The gradients then pass straight
/// through the rounding.
pub(super) fn batch_gradients<'a>(
    network: &Network,
    forward_weights: Option<&QuantizedWeights>,
    pass: &Pass<'_>,
    shards: &'a mut [Shard],
) -> &'a [Shard] {
    let weights = match forward_weights {
        Some(quantized) => quantized.as_slices(),
        None => network.weights.iter().map(Vec::as_slice).collect(),
    };
    let size = pass.windows.len().div_ceil(SHARDS).max(1);
    let used = pass.windows.len().div_ceil(size).max(1);
    let scale = 1.0 / pass.windows.len().max(1) as f32;
    let run = |index: usize, shard: &mut Shard| {
        shard.gradients.clear();
        let start = index * size;
        let end = (start + size).min(pass.windows.len());
        for offset in start..end {
            let mut rng = pass.rng.fork((pass.position + offset) as u64);
            let window = pass.windows[offset];
            accumulate(network, &weights, pass, window, scale, &mut rng, shard);
        }
    };

    let shards = &mut shards[..used];
    let group = used.div_ceil(pass.threads.clamp(1, used));
    in_parallel(shards.chunks_mut(group).enumerate(), |(number, shards)| {
        for (index, shard) in shards.iter_mut().enumerate() {
            run(number * group + index, shard);
        }
    });
    shards
}

/// Returns the sum of the losses of the windows of `shards`.
pub(super) fn loss(shards: &[Shard]) -> f64 {
    shards.iter().map(|shard| shard.gradients.loss).sum()
}

/// Runs `work` on every item, the first on this thread and each other on a
/// thread of its own.
fn in_parallel<T: Send>(items: impl Iterator<Item = T>, work: impl Fn(T) + Sync) {
    thread::scope(|scope| {
        let mut items = items;
        let own = items.next();
        for item in items {
            let work = &work;
            scope.spawn(move || work(item));
        }
        if let Some(item) = own {
            work(item);
        }
    });
}

/// Adds the gradients of one window to those of `shard`.
fn accumulate(
    network: &Network,
    weights: &[&[f32]],
    pass: &Pass<'_>,
    window: usize,
    scale: f32,
    rng: &mut Rng,
    shard: &mut Shard,
) {
    let Shard { gradients, scratch } = shard;
    let dims = &network.dims;
    let (indices, values) = pass.data.window(window);
    scratch.inputs.clear();
    for (&index, &value) in indices.iter().zip(values) {
        if pass.dropout > 0.0 && rng.uniform() < f64::from(pass.dropout) {
            continue;
        }
        scratch.inputs.push((index, f32::from(value) / 255.0));
    }
    forward(dims, weights, &network.biases, pass.quantized, scratch);

    let last = network.layers() - 1;
    let labels: CategorySet = pass.data.window_labels(window);
    scratch.delta.clear();
    for (category, &logit) in Category::ALL.into_iter().zip(&scratch.pre[last]) {
        let logit = f64::from(logit);
        let target = if labels.contains(category) { 1.0 } else { 0.0 };
        gradients.loss += softplus(logit) - target * logit;
        scratch
            .delta
            .push(flush((sigmoid(logit) - target) as f32 * scale));
    }

    for layer in (1..=last).rev() {
        let inputs = dims[layer];
        let activation = &scratch.activations[layer - 1];
        let rows = gradients.weights[layer].chunks_exact_mut(inputs);
        for (row, &delta) in rows.zip(&scratch.delta) {
            for (gradient, &activation) in row.iter_mut().zip(activation) {
                *gradient += delta * activation;
            }
        }
        for (gradient, &delta) in gradients.biases[layer].iter_mut().zip(&scratch.delta) {
            *gradient += delta;
        }
        scratch.previous.clear();
        scratch.previous.resize(inputs, 0.0);
        for (row, &delta) in weights[layer].chunks_exact(inputs).zip(&scratch.delta) {
            for (previous, &weight) in scratch.previous.iter_mut().zip(row) {
                *previous += weight * delta;
            }
        }
        // The clamp passes a gradient only where it did not saturate.
        for (previous, &pre) in scratch.previous.iter_mut().zip(&scratch.pre[layer - 1]) {
            if !(pre > 0.0 && pre < 1.0) {
                *previous = 0.0;
            }
        }
        core::mem::swap(&mut scratch.delta, &mut scratch.previous);
    }

    let outputs = dims[1];
    for (gradient, &delta) in gradients.biases[0].iter_mut().zip(&scratch.delta) {
        *gradient += delta;
    }
    for &(index, value) in &scratch.inputs {
        let start = usize::from(index) * outputs;
        let column = &mut gradients.weights[0][start..start + outputs];
        for (gradient, &delta) in column.iter_mut().zip(&scratch.delta) {
            *gradient += delta * value;
        }
    }
}

/// The Adam optimizer with decoupled weight decay.
pub(super) struct Adam {
    first: Gradients,
    second: Gradients,
    /// `BETA1` and `BETA2` to the power of the number of steps taken.
    powers: (f64, f64),
}

const BETA1: f32 = 0.9;
const BETA2: f32 = 0.999;
const EPSILON: f32 = 1e-8;

impl Adam {
    pub(super) fn new(dims: &[usize]) -> Self {
        Self {
            first: Gradients::new(dims),
            second: Gradients::new(dims),
            powers: (1.0, 1.0),
        }
    }

    /// Moves the parameters of `network` against the sum of the gradients
    /// of `shards`.
    ///
    /// `weight_decay` then shrinks every weight by the share `rate *
    /// weight_decay`. Biases do not decay.
    pub(super) fn step(
        &mut self,
        network: &mut Network,
        shards: &[Shard],
        rate: f32,
        weight_decay: f32,
        threads: usize,
    ) {
        self.powers.0 *= f64::from(BETA1);
        self.powers.1 *= f64::from(BETA2);
        let layers = network.layers();
        let parameters = network.weights.iter_mut().chain(&mut network.biases);
        let first = self.first.weights.iter_mut().chain(&mut self.first.biases);
        let second = self
            .second
            .weights
            .iter_mut()
            .chain(&mut self.second.biases);
        for (tensor, ((parameters, first), second)) in parameters.zip(first).zip(second).enumerate()
        {
            let update = Update {
                gradients: shards
                    .iter()
                    .map(|shard| {
                        let gradients = &shard.gradients;
                        if tensor < layers {
                            gradients.weights[tensor].as_slice()
                        } else {
                            gradients.biases[tensor - layers].as_slice()
                        }
                    })
                    .collect(),
                rate,
                decay: if tensor < layers {
                    rate * weight_decay
                } else {
                    0.0
                },
                first_correction: (1.0 / (1.0 - self.powers.0)) as f32,
                second_correction: (1.0 / (1.0 - self.powers.1)) as f32,
            };
            // Each element is updated on its own, so the division into
            // threads does not change the result. Small tensors are not
            // worth a thread.
            let len = parameters.len();
            let threads = if len < PARALLEL_LEN {
                1
            } else {
                threads.max(1)
            };
            let part = len.div_ceil(threads).next_multiple_of(BLOCK).max(BLOCK);
            let parts = parameters
                .chunks_mut(part)
                .zip(first.chunks_mut(part))
                .zip(second.chunks_mut(part))
                .enumerate();
            in_parallel(parts, |(number, ((parameters, first), second))| {
                update.apply(number * part, parameters, first, second);
            });
        }
    }
}

/// Returns zero for a magnitude below `1e-30`, and `value` otherwise.
#[inline]
fn flush(value: f32) -> f32 {
    if value.abs() < 1e-30 { 0.0 } else { value }
}

/// The number of elements that are summed and updated together.
const BLOCK: usize = 512;

/// The smallest tensor that is updated by several threads.
const PARALLEL_LEN: usize = 1 << 15;

/// One step of the optimizer for one tensor.
struct Update<'a> {
    /// The gradients of each shard for the whole tensor.
    gradients: Vec<&'a [f32]>,
    rate: f32,
    decay: f32,
    first_correction: f32,
    second_correction: f32,
}

impl Update<'_> {
    /// Updates the elements from `start` of the tensor.
    fn apply(&self, start: usize, parameters: &mut [f32], first: &mut [f32], second: &mut [f32]) {
        let mut sum = [0.0f32; BLOCK];
        let blocks = parameters
            .chunks_mut(BLOCK)
            .zip(first.chunks_mut(BLOCK))
            .zip(second.chunks_mut(BLOCK));
        for (block, ((parameters, first), second)) in blocks.enumerate() {
            let range = start + block * BLOCK..start + block * BLOCK + parameters.len();
            let sum = &mut sum[..parameters.len()];
            // Shards are added in order, whatever thread computed them.
            sum.fill(0.0);
            for gradients in &self.gradients {
                for (sum, &gradient) in sum.iter_mut().zip(&gradients[range.clone()]) {
                    *sum += gradient;
                }
            }
            for (((parameter, first), second), &gradient) in parameters
                .iter_mut()
                .zip(first.iter_mut())
                .zip(second.iter_mut())
                .zip(sum.iter())
            {
                // The averages of a weight whose features stay absent decay
                // towards zero. Values this small no longer move a weight,
                // and arithmetic on subnormal values is very slow.
                *first = flush(BETA1 * *first + (1.0 - BETA1) * gradient);
                *second = flush(BETA2 * *second + (1.0 - BETA2) * gradient * gradient);
                let direction = (*first * self.first_correction)
                    / ((*second * self.second_correction).sqrt() + EPSILON);
                *parameter -= self.rate * direction;
                *parameter -= self.decay * *parameter;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::dataset::Sample;

    fn dataset() -> Dataset {
        let mut data = Dataset::new();
        data.push(Sample::new(b"id=1' OR '1'='1' --", Category::SqlInjection))
            .unwrap();
        data.push(Sample::benign(b"GET /index.html HTTP/1.1"))
            .unwrap();
        let both = CategorySet::from(Category::PathTraversal) | Category::Obfuscation;
        data.push(Sample::new(b"..%252f..%252fetc%252fpasswd", both))
            .unwrap();
        data.push(Sample::benign(
            b"The quick brown fox jumps over the lazy dog.",
        ))
        .unwrap();
        data
    }

    /// The loss that `batch_gradients` differentiates, in 64-bit arithmetic.
    fn mean_loss(network: &Network, data: &Dataset) -> f64 {
        let mut loss = 0.0;
        for window in 0..data.windows() {
            let logits = network.logits(data.window(window));
            for (category, logit) in Category::ALL.into_iter().zip(logits) {
                let target = if data.window_labels(window).contains(category) {
                    1.0
                } else {
                    0.0
                };
                loss += softplus(f64::from(logit)) - target * f64::from(logit);
            }
        }
        loss / data.windows() as f64
    }

    fn gradients(network: &Network, data: &Dataset, threads: usize) -> Gradients {
        let windows: Vec<usize> = (0..data.windows()).collect();
        let mut shards: Vec<Shard> = (0..SHARDS).map(|_| Shard::new(&network.dims)).collect();
        let pass = Pass {
            data,
            windows: &windows,
            position: 0,
            quantized: false,
            dropout: 0.0,
            rng: &Rng::new(1),
            threads,
        };
        let shards = batch_gradients(network, None, &pass, &mut shards);
        let mut total = Gradients::new(&network.dims);
        for shard in shards {
            total.add(&shard.gradients);
        }
        assert_eq!(total.loss.to_bits(), loss(shards).to_bits());
        total
    }

    /// One shard that holds `gradients`.
    fn shard(dims: &[usize], gradients: Gradients) -> Vec<Shard> {
        let mut shard = Shard::new(dims);
        shard.gradients = gradients;
        vec![shard]
    }

    #[test]
    fn gradients_match_finite_differences() {
        let data = dataset();
        for hidden in [&[][..], &[5], &[6, 4]] {
            let mut dims = vec![crate::fingerprint::FINGERPRINT_LEN];
            dims.extend(hidden);
            dims.push(Category::COUNT);
            let mut rng = Rng::new(3);
            let mut network = Network::random(&mut rng, &dims);
            // Biases away from zero put some units on each side of the
            // clamp.
            for bias in network.biases.iter_mut().flatten() {
                *bias = (rng.uniform() * 0.6 - 0.1) as f32;
            }
            let analytic = gradients(&network, &data, 1);
            assert!(
                (analytic.loss / data.windows() as f64 - mean_loss(&network, &data)).abs() < 1e-5
            );

            let (features, _) = data.window(0);
            let outputs = dims[1];
            let mut checked = 0;
            for tensor in 0..2 * network.layers() {
                let layer = tensor % network.layers();
                let len = if tensor < network.layers() {
                    network.weights[layer].len()
                } else {
                    network.biases[layer].len()
                };
                for probe in 0..12 {
                    // In the first layer, probe weights of features that
                    // the data has, because the others have no gradient.
                    let index = if tensor == 0 {
                        usize::from(features[probe % features.len()]) * outputs + probe % outputs
                    } else {
                        rng.below(len)
                    };
                    let weight = tensor < network.layers();
                    let expected = if weight {
                        analytic.weights[layer][index]
                    } else {
                        analytic.biases[layer][index]
                    };
                    let original = if weight {
                        network.weights[layer][index]
                    } else {
                        network.biases[layer][index]
                    };
                    let step = 1e-3;
                    let mut moved = network.clone();
                    let mut set = |value: f32| {
                        if weight {
                            moved.weights[layer][index] = value;
                        } else {
                            moved.biases[layer][index] = value;
                        }
                        mean_loss(&moved, &data)
                    };
                    let numeric =
                        (set(original + step) - set(original - step)) / f64::from(2.0 * step);
                    let tolerance = 2e-3 + 0.02 * numeric.abs();
                    // A unit that crosses the clamp inside the step makes
                    // the difference quotient wrong, so a few probes may
                    // disagree.
                    if (numeric - f64::from(expected)).abs() <= tolerance {
                        checked += 1;
                    }
                }
            }
            let probes = 12 * 2 * network.layers();
            assert!(
                checked * 10 >= probes * 9,
                "hidden {hidden:?}: {checked} of {probes} probes agree"
            );
        }
    }

    #[test]
    fn gradients_do_not_depend_on_the_thread_count() {
        let mut data = Dataset::new();
        let mut rng = Rng::new(5);
        for index in 0..61 {
            let input: Vec<u8> = (0..20 + index).map(|_| rng.next_u64() as u8).collect();
            let labels = CategorySet::from_bits(rng.next_u64() as u8);
            data.push(Sample::new(&input, labels)).unwrap();
        }
        let dims = [crate::fingerprint::FINGERPRINT_LEN, 7, Category::COUNT];
        let network = Network::random(&mut rng, &dims);
        let reference = gradients(&network, &data, 1);
        for threads in [2, 3, 8, 64] {
            let other = gradients(&network, &data, threads);
            assert_eq!(other.weights, reference.weights, "{threads} threads");
            assert_eq!(other.biases, reference.biases, "{threads} threads");
            assert_eq!(
                other.loss.to_bits(),
                reference.loss.to_bits(),
                "{threads} threads"
            );
        }
    }

    #[test]
    fn quantized_weights_have_at_most_255_levels_for_each_output() {
        let dims = [crate::fingerprint::FINGERPRINT_LEN, 6, 5, Category::COUNT];
        let network = Network::random(&mut Rng::new(6), &dims);
        let mut quantized = QuantizedWeights::new(&dims);
        quantized.update(&network);
        for layer in 0..network.layers() {
            let (inputs, outputs) = (dims[layer], dims[layer + 1]);
            for output in 0..outputs {
                let element = |weights: &[f32], input: usize| {
                    if layer == 0 {
                        weights[input * outputs + output]
                    } else {
                        weights[output * inputs + input]
                    }
                };
                let max = (0..inputs)
                    .map(|input| element(&network.weights[layer], input).abs())
                    .fold(0.0, f32::max);
                let scale = max / 127.0;
                for input in 0..inputs {
                    let (real, rounded) = (
                        element(&network.weights[layer], input),
                        element(&quantized.weights[layer], input),
                    );
                    assert!((real - rounded).abs() <= scale * 0.5001);
                    let level = rounded / scale;
                    assert!((level - level.round()).abs() < 1e-3 && level.abs() <= 127.001);
                }
            }
        }
    }

    #[test]
    fn steps_do_not_depend_on_the_thread_count() {
        let data = dataset();
        let dims = [crate::fingerprint::FINGERPRINT_LEN, 16, Category::COUNT];
        let start = Network::random(&mut Rng::new(8), &dims);
        let windows: Vec<usize> = (0..data.windows()).collect();
        let stepped = |threads: usize| {
            let mut network = start.clone();
            let mut adam = Adam::new(&dims);
            let mut shards: Vec<Shard> = (0..SHARDS).map(|_| Shard::new(&dims)).collect();
            for _ in 0..3 {
                let pass = Pass {
                    data: &data,
                    windows: &windows,
                    position: 0,
                    quantized: false,
                    dropout: 0.2,
                    rng: &Rng::new(2),
                    threads,
                };
                let used = batch_gradients(&network, None, &pass, &mut shards);
                assert_eq!(used.len(), windows.len());
                adam.step(&mut network, used, 0.01, 0.1, threads);
            }
            network
        };
        let reference = stepped(1);
        assert_ne!(reference, start);
        for threads in [2, 3, 7, 100] {
            assert_eq!(stepped(threads), reference, "{threads} threads");
        }
    }

    #[test]
    fn steps_lower_the_loss_and_decay_only_weights() {
        let data = dataset();
        let dims = [crate::fingerprint::FINGERPRINT_LEN, 8, Category::COUNT];
        let mut network = Network::random(&mut Rng::new(7), &dims);
        let before = mean_loss(&network, &data);
        let mut adam = Adam::new(&dims);
        for _ in 0..40 {
            let step = shard(&dims, gradients(&network, &data, 1));
            adam.step(&mut network, &step, 0.01, 0.0, 1);
        }
        assert!(network.is_finite());
        assert!(mean_loss(&network, &data) < 0.5 * before);

        // With zero gradients, only the decay moves parameters.
        let mut decayed = network.clone();
        let zero = shard(&dims, Gradients::new(&dims));
        Adam::new(&dims).step(&mut decayed, &zero, 0.5, 0.2, 1);
        assert_eq!(decayed.biases, network.biases);
        for (decayed, original) in decayed
            .weights
            .iter()
            .flatten()
            .zip(network.weights.iter().flatten())
        {
            assert!((decayed - original * 0.9).abs() <= 1e-6 * original.abs().max(1.0));
        }
    }
}
