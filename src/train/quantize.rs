//! Conversion of a float network to the integer arithmetic of a model.

use super::network::Network;
use crate::fingerprint::FINGERPRINT_LEN;
use crate::model::{LayerParameters, MAX_SHIFT, Model};

/// The output layer stores logits in units of 1/16.
const LOGIT_SCALE: f64 = 16.0;

/// The largest multiplier that quantization produces, `2^30`.
const MULTIPLIER_LIMIT: f64 = (1u64 << 30) as f64;

/// Returns the parameters of the quantized layers of `network`, or `None`
/// when its values are too large to represent.
///
/// A stored activation is 255 times the float activation. Output `o` of a
/// layer has its own weight scale `s`, so that the accumulator
/// `sum(weight_q * activation_q) + bias_q` equals `255 / s` times the float
/// value. The multiplier and the shift then scale it by `s * unit / 255`,
/// where `unit` is 255 for a hidden layer and 16 for the logits.
pub(super) fn quantize(network: &Network) -> Option<Vec<LayerParameters>> {
    let layers = network.layers();
    (0..layers)
        .map(|layer| {
            let unit = if layer + 1 == layers {
                LOGIT_SCALE
            } else {
                255.0
            };
            quantize_layer(network, layer, unit)
        })
        .collect()
}

fn quantize_layer(network: &Network, layer: usize, unit: f64) -> Option<LayerParameters> {
    let (inputs, outputs) = (network.dims[layer], network.dims[layer + 1]);
    let stored = &network.weights[layer];
    // The first layer is stored by input.
    let weight = |output: usize, input: usize| -> f64 {
        f64::from(if layer == 0 {
            stored[input * outputs + output]
        } else {
            stored[output * inputs + input]
        })
    };

    let mut weights = Vec::with_capacity(outputs * inputs);
    let mut biases = Vec::with_capacity(outputs);
    let mut factors = Vec::with_capacity(outputs);
    for (output, &bias) in network.biases[layer].iter().enumerate() {
        let bias = f64::from(bias);
        let largest = (0..inputs).fold(0.0f64, |max, input| max.max(weight(output, input).abs()));
        // An output without usable weights keeps a scale that still
        // represents its bias within 31 bits.
        let scale = (largest / 127.0)
            .max(bias.abs() * 255.0 / MULTIPLIER_LIMIT)
            .max(1e-9);
        for input in 0..inputs {
            let level = (weight(output, input) / scale)
                .round_ties_even()
                .clamp(-127.0, 127.0);
            weights.push(level as i8);
        }
        // The scale bounds the magnitude by `2^30`.
        biases.push((bias * 255.0 / scale).round_ties_even() as i32);
        factors.push(scale * unit / 255.0);
    }

    // The largest shift for which every multiplier stays within `2^30`.
    // Doubling is exact, so this needs no logarithm.
    let largest = factors.iter().fold(0.0f64, |max, &factor| max.max(factor));
    if !largest.is_finite() {
        return None;
    }
    let mut shift = 0u8;
    let mut power = 1.0f64;
    while shift < MAX_SHIFT && largest * power * 2.0 <= MULTIPLIER_LIMIT {
        power *= 2.0;
        shift += 1;
    }
    if shift == 0 {
        return None;
    }
    let multipliers = factors
        .iter()
        .map(|&factor| (factor * power).round_ties_even() as i32)
        .collect();
    Some(LayerParameters {
        shift,
        biases,
        multipliers,
        weights,
    })
}

/// Returns the float network that computes what `model` computes, up to
/// the rounding of activations.
///
/// This inverts [`quantize`]: the multiplier and the shift of an output give
/// its weight scale, and the stored weights and bias are multiples of it.
pub(super) fn dequantize(model: &Model) -> Network {
    let layers: Vec<LayerParameters> = model.layers().collect();
    let mut dims = Vec::with_capacity(layers.len() + 1);
    dims.push(FINGERPRINT_LEN);
    dims.extend(layers.iter().map(|layer| layer.biases.len()));
    let mut weights = Vec::with_capacity(layers.len());
    let mut biases = Vec::with_capacity(layers.len());
    for (index, layer) in layers.iter().enumerate() {
        let (inputs, outputs) = (dims[index], dims[index + 1]);
        let unit = if index + 1 == layers.len() {
            LOGIT_SCALE
        } else {
            255.0
        };
        // The shift is at most 62, so the power is exact.
        let power = (1u64 << layer.shift) as f64;
        let scales: Vec<f64> = layer
            .multipliers
            .iter()
            .map(|&multiplier| f64::from(multiplier) / power * 255.0 / unit)
            .collect();
        let mut layer_weights = vec![0.0f32; inputs * outputs];
        for (output, row) in layer.weights.chunks_exact(inputs).enumerate() {
            for (input, &weight) in row.iter().enumerate() {
                // The first layer is stored by input.
                let position = if index == 0 {
                    input * outputs + output
                } else {
                    output * inputs + input
                };
                layer_weights[position] = (f64::from(weight) * scales[output]) as f32;
            }
        }
        weights.push(layer_weights);
        biases.push(
            layer
                .biases
                .iter()
                .zip(&scales)
                .map(|(&bias, &scale)| (f64::from(bias) * scale / 255.0) as f32)
                .collect(),
        );
    }
    Network {
        dims,
        weights,
        biases,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Verified;
    use crate::fingerprint::{FINGERPRINT_LEN, Fingerprint};
    use crate::model::Model;
    use crate::train::math::{Rng, sigmoid};
    use crate::{Category, testutil};

    fn features(fingerprint: &Fingerprint) -> (Vec<u16>, Vec<u8>) {
        fingerprint
            .as_array()
            .iter()
            .enumerate()
            .filter(|&(_, &value)| value != 0)
            .map(|(index, &value)| (index as u16, value))
            .unzip()
    }

    #[test]
    fn quantized_scores_follow_the_float_network() {
        let mut rng = Rng::new(1);
        let mut bytes = testutil::Rng::new(2);
        for hidden in [&[][..], &[24], &[32, 16], &[8, 8, 8]] {
            let mut dims = vec![FINGERPRINT_LEN];
            dims.extend(hidden);
            dims.push(Category::COUNT);
            let mut network = Network::random(&mut rng, &dims);
            for bias in network.biases.iter_mut().flatten() {
                *bias = (rng.uniform() - 0.5) as f32;
            }
            let model = Model::from_layers(quantize(&network).unwrap()).unwrap();
            assert_eq!(model.layer_sizes().collect::<Vec<_>>(), dims[1..]);

            let mut worst = 0.0f64;
            for _ in 0..50 {
                // A sparse fingerprint, like those of real windows.
                let mut array = [0u8; FINGERPRINT_LEN];
                for _ in 0..200 {
                    let mut pair = [0u8; 3];
                    bytes.fill(&mut pair);
                    let index =
                        usize::from(u16::from_le_bytes([pair[0], pair[1]])) % FINGERPRINT_LEN;
                    array[index] = pair[2];
                }
                let fingerprint = Fingerprint::from_array(array);
                let (indices, values) = features(&fingerprint);
                let logits = network.logits((&indices, &values));
                let scores = model.infer(Verified::SCALAR, &fingerprint);
                for (logit, &score) in logits.iter().zip(scores.as_array()) {
                    let expected = 255.0 * sigmoid(f64::from(*logit));
                    worst = worst.max((expected - f64::from(score)).abs());
                }
            }
            // Rounding of weights and activations moves a score by a few
            // steps, more with each layer.
            assert!(worst <= 12.0, "hidden {hidden:?}: worst difference {worst}");
        }
    }

    #[test]
    fn dequantized_models_score_like_the_model_and_quantize_back() {
        let mut rng = Rng::new(5);
        let mut bytes = testutil::Rng::new(6);
        for hidden in [&[][..], &[24], &[32, 16]] {
            let mut dims = vec![FINGERPRINT_LEN];
            dims.extend(hidden);
            dims.push(Category::COUNT);
            let mut network = Network::random(&mut rng, &dims);
            for bias in network.biases.iter_mut().flatten() {
                *bias = (rng.uniform() - 0.5) as f32;
            }
            let model = Model::from_layers(quantize(&network).unwrap()).unwrap();
            let restored = dequantize(&model);
            assert_eq!(restored.dims, dims);

            // Quantizing again gives the same weights. A multiplier or a
            // bias can move by one step, because the scale was rounded.
            let again = quantize(&restored).unwrap();
            for (first, second) in model.layers().zip(&again) {
                assert_eq!(first.shift, second.shift);
                assert_eq!(first.weights, second.weights);
                for (a, b) in first.multipliers.iter().zip(&second.multipliers) {
                    assert!((a - b).abs() <= 64, "multipliers {a} and {b}");
                }
                for (a, b) in first.biases.iter().zip(&second.biases) {
                    assert!((a - b).abs() <= 1, "biases {a} and {b}");
                }
            }

            let mut worst = 0.0f64;
            for _ in 0..30 {
                let mut array = [0u8; FINGERPRINT_LEN];
                for _ in 0..200 {
                    let mut pair = [0u8; 3];
                    bytes.fill(&mut pair);
                    let index =
                        usize::from(u16::from_le_bytes([pair[0], pair[1]])) % FINGERPRINT_LEN;
                    array[index] = pair[2];
                }
                let fingerprint = Fingerprint::from_array(array);
                let (indices, values) = features(&fingerprint);
                let logits = restored.logits((&indices, &values));
                let scores = model.infer(Verified::SCALAR, &fingerprint);
                for (logit, &score) in logits.iter().zip(scores.as_array()) {
                    let expected = 255.0 * sigmoid(f64::from(*logit));
                    worst = worst.max((expected - f64::from(score)).abs());
                }
            }
            // Only the rounding of activations and of the last shift is
            // left, which is less than the rounding of weights.
            assert!(worst <= 6.0, "hidden {hidden:?}: worst difference {worst}");
        }
    }

    #[test]
    fn a_layer_without_multipliers_dequantizes_to_zero() {
        let layer = LayerParameters {
            shift: 1,
            biases: vec![5; Category::COUNT],
            multipliers: vec![0; Category::COUNT],
            weights: vec![3; Category::COUNT * FINGERPRINT_LEN],
        };
        let network = dequantize(&Model::from_layers(vec![layer]).unwrap());
        assert!(network.weights[0].iter().all(|&weight| weight == 0.0));
        assert!(network.biases[0].iter().all(|&bias| bias == 0.0));
        assert!(quantize(&network).is_some());
    }

    #[test]
    fn extreme_parameters_are_represented_or_refused() {
        let dims = [FINGERPRINT_LEN, 4, Category::COUNT];
        let mut network = Network::random(&mut Rng::new(3), &dims);

        // Zero weights and a large bias: the bias sets the scale.
        network.weights[0].fill(0.0);
        network.biases[0] = vec![1000.0, -1000.0, 0.0, 1e-20];
        let layers = quantize(&network).unwrap();
        assert!(layers[0].weights.iter().all(|&weight| weight == 0));
        assert!(
            layers[0]
                .biases
                .iter()
                .all(|bias| bias.unsigned_abs() <= 1 << 30)
        );
        assert!(Model::from_layers(layers).is_ok());

        // Weights at the limits of the type still give a valid model.
        network.weights[0].fill(f32::MAX);
        assert!(quantize(&network).is_none());
        network.weights[0].fill(1e6);
        network.biases[0].fill(-1e6);
        let layers = quantize(&network).unwrap();
        assert!(layers[0].weights.iter().all(|&weight| weight == 127));
        assert!(Model::from_layers(layers).is_ok());

        network.weights[1][0] = f32::NAN;
        assert!(quantize(&network).is_none() || !network.is_finite());
        network.weights[1][0] = f32::INFINITY;
        assert!(quantize(&network).is_none());
    }

    #[test]
    fn multipliers_use_the_largest_shift() {
        let dims = [FINGERPRINT_LEN, Category::COUNT];
        let mut network = Network::random(&mut Rng::new(4), &dims);
        network.biases[0].fill(0.0);
        for (index, weight) in network.weights[0].iter_mut().enumerate() {
            // Output `o` has weights of magnitude `2^-o`.
            *weight = 1.0 / (1u32 << (index % Category::COUNT)) as f32;
        }
        let layers = quantize(&network).unwrap();
        let layer = &layers[0];
        // The largest factor is `16 / (127 * 255)`, which is `2^-11` times
        // about 1.01, so the shift is 40 and its multiplier is just above
        // `2^29`.
        assert_eq!(layer.shift, 40);
        assert!(layer.multipliers[0] > 1 << 29 && layer.multipliers[0] <= 1 << 30);
        for pair in layer.multipliers.windows(2) {
            assert!((pair[0] - 2 * pair[1]).abs() <= 1);
        }
        assert!(layer.weights.iter().all(|&weight| weight == 127));
    }
}
