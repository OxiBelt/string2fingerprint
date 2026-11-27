//! Quantized network models: the binary format, validation, and inference.
//!
//! A [`Model`] is a small fully connected network with unsigned 8-bit
//! activations and signed 8-bit weights. The design document in
//! `docs/design.md` specifies the arithmetic. This module adds the byte
//! format in which a trained model is stored.
//!
//! # Format
//!
//! All integers are little-endian.
//!
//! | Field | Size in bytes | Constraint |
//! | --- | --- | --- |
//! | magic | 4 | `S2FP` |
//! | version | 2 | [`FORMAT_VERSION`] |
//! | layer count `L` | 1 | 1 to [`MAX_LAYERS`] |
//! | reserved | 1 | zero |
//! | dimensions | `2 * (L + 1)` | first is the fingerprint length, last is the category count, others are 1 to [`MAX_HIDDEN`] |
//! | layers | see below | in order from input to output |
//! | checksum | 4 | FNV-1a 32 of all preceding bytes |
//!
//! A layer with `n` inputs and `m` outputs is stored as:
//!
//! | Field | Size in bytes | Constraint |
//! | --- | --- | --- |
//! | shift | 1 | 1 to [`MAX_SHIFT`] |
//! | reserved | 3 | zero |
//! | biases | `4 * m` | signed 32-bit |
//! | multipliers | `4 * m` | signed 32-bit, not negative |
//! | weights | `m * n` | signed 8-bit, row-major, not -128 |
//!
//! For every output, the magnitude of the bias plus 255 times the sum of the
//! weight magnitudes must not exceed `i32::MAX`. A model that passes
//! validation cannot overflow during inference for any input.
//!
//! The checksum detects accidental corruption. It is not an authenticity
//! check, so load models only from a trusted source.

use core::fmt;

use crate::backend::Verified;
use crate::fingerprint::{FINGERPRINT_LEN, Fingerprint};
use crate::kernel::{self, Matrix};
use crate::{Category, Scores};

/// The format version that this crate reads.
pub const FORMAT_VERSION: u16 = 1;

/// The largest number of layers in a model.
pub const MAX_LAYERS: usize = 4;

/// The largest dimension of a hidden layer.
pub const MAX_HIDDEN: usize = 256;

/// The largest shift of a layer.
pub const MAX_SHIFT: u8 = 62;

const MAGIC: [u8; 4] = *b"S2FP";

/// Maps an output-layer value `q`, stored at index `q + 128`, to the score
/// `round(255 / (1 + exp(-q / 16)))`.
const LOGISTIC: [u8; 256] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4,
    5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 9, 10, 10, 11, 11, 12, 13, 14, 14, 15, 16, 17, 18, 19, 20, 22,
    23, 24, 26, 27, 29, 30, 32, 34, 36, 38, 40, 42, 44, 47, 49, 51, 54, 57, 60, 62, 65, 69, 72, 75,
    78, 82, 85, 89, 93, 96, 100, 104, 108, 112, 116, 120, 124, 128, 131, 135, 139, 143, 147, 151,
    155, 159, 162, 166, 170, 173, 177, 180, 183, 186, 190, 193, 195, 198, 201, 204, 206, 208, 211,
    213, 215, 217, 219, 221, 223, 225, 226, 228, 229, 231, 232, 233, 235, 236, 237, 238, 239, 240,
    241, 241, 242, 243, 244, 244, 245, 245, 246, 247, 247, 248, 248, 248, 249, 249, 249, 250, 250,
    250, 251, 251, 251, 251, 252, 252, 252, 252, 252, 253, 253, 253, 253, 253, 253, 253, 253, 253,
    254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255,
];

/// The reason a byte string is not a valid model.
///
/// The error never contains model bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ModelError {
    /// The bytes end before the model does.
    Truncated,
    /// The bytes do not start with the model magic number.
    BadMagic,
    /// The format version is not [`FORMAT_VERSION`].
    UnsupportedVersion,
    /// The layer count or a layer dimension is out of range.
    InvalidDimensions,
    /// Bytes follow the checksum.
    TrailingData,
    /// The checksum does not match the content.
    ChecksumMismatch,
    /// A shift, multiplier, weight, or reserved field is out of range.
    InvalidParameter,
    /// A layer could overflow a 32-bit accumulator for some input.
    AccumulatorOverflow,
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Truncated => "model is truncated",
            Self::BadMagic => "model magic number is missing",
            Self::UnsupportedVersion => "model format version is not supported",
            Self::InvalidDimensions => "model dimensions are out of range",
            Self::TrailingData => "model is followed by unexpected data",
            Self::ChecksumMismatch => "model checksum does not match",
            Self::InvalidParameter => "model parameter is out of range",
            Self::AccumulatorOverflow => "model layer can overflow its accumulator",
        })
    }
}

impl core::error::Error for ModelError {}

/// One fully connected layer.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Layer {
    matrix: Matrix,
    bias: Box<[i32]>,
    multiplier: Box<[i32]>,
    shift: u8,
}

impl Layer {
    /// Validates the parameters of a layer with `cols` inputs.
    fn new(
        weights: Box<[i8]>,
        cols: usize,
        bias: Box<[i32]>,
        multiplier: Box<[i32]>,
        shift: u8,
    ) -> Result<Self, ModelError> {
        if !(1..=MAX_SHIFT).contains(&shift) || multiplier.iter().any(|&value| value < 0) {
            return Err(ModelError::InvalidParameter);
        }
        let matrix = Matrix::new(weights, cols).ok_or(ModelError::InvalidParameter)?;
        if matrix.rows() != bias.len() || matrix.rows() != multiplier.len() {
            return Err(ModelError::InvalidDimensions);
        }
        // The bound applies to each output on its own. A row bound is below
        // `2^15` per column and a bias magnitude is at most `2^31`, so the
        // 64-bit sum is exact.
        let overflows = matrix
            .abs_dot_bounds()
            .zip(&bias)
            .any(|(bound, &bias)| bound + u64::from(bias.unsigned_abs()) > i32::MAX as u64);
        if overflows {
            return Err(ModelError::AccumulatorOverflow);
        }
        Ok(Self {
            matrix,
            bias,
            multiplier,
            shift,
        })
    }

    /// Returns the scaled and rounded value of output `row`.
    ///
    /// Validation bounds `acc + bias` by `i32::MAX` in magnitude and the
    /// multiplier by `i32::MAX`, so the product is below `2^62`, and the
    /// rounding term is at most `2^61`. The sum fits in 64 bits.
    fn requantize(&self, row: usize, acc: i32) -> i64 {
        let sum = i64::from(acc) + i64::from(self.bias[row]);
        let rounding = 1i64 << (self.shift - 1);
        (sum * i64::from(self.multiplier[row]) + rounding) >> self.shift
    }
}

/// A validated quantized network.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Model {
    /// Never empty. Consecutive dimensions match, the first layer has
    /// [`FINGERPRINT_LEN`] inputs, and the last has [`Category::COUNT`]
    /// outputs.
    layers: Box<[Layer]>,
}

impl Model {
    /// Parses and validates a model.
    ///
    /// Parsing reads `bytes` once, allocates at most the size of `bytes`
    /// plus a small constant, and never panics.
    ///
    /// # Errors
    ///
    /// Returns the first [`ModelError`] found. See the [module
    /// documentation](self) for the format and its constraints.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ModelError> {
        let mut reader = Reader::new(bytes);
        if reader.take(MAGIC.len())? != MAGIC {
            return Err(ModelError::BadMagic);
        }
        if reader.u16()? != FORMAT_VERSION {
            return Err(ModelError::UnsupportedVersion);
        }
        let layer_count = usize::from(reader.u8()?);
        if reader.u8()? != 0 {
            return Err(ModelError::InvalidParameter);
        }
        if !(1..=MAX_LAYERS).contains(&layer_count) {
            return Err(ModelError::InvalidDimensions);
        }

        let mut dimensions = [0usize; MAX_LAYERS + 1];
        let dimensions = &mut dimensions[..=layer_count];
        for dimension in dimensions.iter_mut() {
            *dimension = usize::from(reader.u16()?);
        }
        let hidden = &dimensions[1..layer_count];
        if dimensions[0] != FINGERPRINT_LEN
            || dimensions[layer_count] != Category::COUNT
            || hidden
                .iter()
                .any(|dimension| !(1..=MAX_HIDDEN).contains(dimension))
        {
            return Err(ModelError::InvalidDimensions);
        }

        // Check the total length and the checksum before allocating, so that
        // allocation is bounded by the length of `bytes`.
        let body: usize = dimensions
            .windows(2)
            .map(|pair| 4 + pair[1] * (8 + pair[0]))
            .sum();
        let checksum_at = reader.position() + body;
        match bytes.len().cmp(&(checksum_at + 4)) {
            core::cmp::Ordering::Less => return Err(ModelError::Truncated),
            core::cmp::Ordering::Greater => return Err(ModelError::TrailingData),
            core::cmp::Ordering::Equal => {}
        }
        let (content, stored) = bytes.split_at(checksum_at);
        if stored != checksum(content).to_le_bytes() {
            return Err(ModelError::ChecksumMismatch);
        }

        let mut layers = Vec::with_capacity(layer_count);
        for pair in dimensions.windows(2) {
            let (cols, rows) = (pair[0], pair[1]);
            let shift = reader.u8()?;
            if reader.take(3)? != [0; 3] {
                return Err(ModelError::InvalidParameter);
            }
            let bias = reader.i32s(rows)?;
            let multiplier = reader.i32s(rows)?;
            let weights = reader
                .take(rows * cols)?
                .iter()
                .map(|&byte| i8::from_le_bytes([byte]))
                .collect();
            layers.push(Layer::new(weights, cols, bias, multiplier, shift)?);
        }
        Ok(Self {
            layers: layers.into(),
        })
    }

    /// Returns the number of outputs of each layer, from the first hidden
    /// layer to the output layer.
    pub fn layer_sizes(&self) -> impl ExactSizeIterator<Item = usize> + '_ {
        self.layers.iter().map(|layer| layer.matrix.rows())
    }

    /// Runs the network on one fingerprint.
    pub(crate) fn infer(&self, backend: Verified, fingerprint: &Fingerprint) -> Scores {
        let mut acc = [0i32; MAX_HIDDEN];
        let mut front = [0u8; MAX_HIDDEN];
        let mut back = [0u8; MAX_HIDDEN];
        let (mut source, mut target) = (&mut front, &mut back);
        let mut input: &[u8] = fingerprint.as_array();

        let Some((output, hidden)) = self.layers.split_last() else {
            return Scores::ZERO;
        };
        for layer in hidden {
            let rows = layer.matrix.rows();
            let acc = &mut acc[..rows];
            kernel::matvec(backend, &layer.matrix, input, acc);
            for (row, (activation, &acc)) in target[..rows].iter_mut().zip(acc.iter()).enumerate() {
                // The clamp is also the rectifier.
                *activation = layer.requantize(row, acc).clamp(0, 255) as u8;
            }
            core::mem::swap(&mut source, &mut target);
            input = &source[..rows];
        }

        let mut scores = [0u8; Category::COUNT];
        let acc = &mut acc[..Category::COUNT];
        kernel::matvec(backend, &output.matrix, input, acc);
        for (row, (score, &acc)) in scores.iter_mut().zip(acc.iter()).enumerate() {
            let value = output.requantize(row, acc).clamp(-128, 127);
            *score = LOGISTIC[(value + 128) as usize];
        }
        Scores::from_array(scores)
    }
}

/// Returns the FNV-1a 32 hash of `bytes`.
fn checksum(bytes: &[u8]) -> u32 {
    bytes.iter().fold(0x811C_9DC5, |hash, &byte| {
        (hash ^ u32::from(byte)).wrapping_mul(0x0100_0193)
    })
}

/// A bounds-checked cursor over model bytes.
struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn position(&self) -> usize {
        self.position
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], ModelError> {
        let end = self
            .position
            .checked_add(len)
            .ok_or(ModelError::Truncated)?;
        let taken = self
            .bytes
            .get(self.position..end)
            .ok_or(ModelError::Truncated)?;
        self.position = end;
        Ok(taken)
    }

    fn u8(&mut self) -> Result<u8, ModelError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, ModelError> {
        let bytes = self.take(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn i32s(&mut self, count: usize) -> Result<Box<[i32]>, ModelError> {
        let bytes = self.take(count.checked_mul(4).ok_or(ModelError::Truncated)?)?;
        let (chunks, _) = bytes.as_chunks::<4>();
        Ok(chunks
            .iter()
            .map(|&chunk| i32::from_le_bytes(chunk))
            .collect())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::testutil::Rng;

    /// The parameters of one layer in the order of the format.
    #[derive(Clone)]
    pub(crate) struct RawLayer {
        pub(crate) shift: u8,
        pub(crate) bias: Vec<i32>,
        pub(crate) multiplier: Vec<i32>,
        pub(crate) weights: Vec<i8>,
    }

    /// Encodes a model without validating it.
    pub(crate) fn encode(dimensions: &[usize], layers: &[RawLayer]) -> Vec<u8> {
        let mut bytes = MAGIC.to_vec();
        bytes.extend(FORMAT_VERSION.to_le_bytes());
        bytes.push(u8::try_from(layers.len()).unwrap());
        bytes.push(0);
        for &dimension in dimensions {
            bytes.extend(u16::try_from(dimension).unwrap().to_le_bytes());
        }
        for layer in layers {
            bytes.extend([layer.shift, 0, 0, 0]);
            for value in layer.bias.iter().chain(&layer.multiplier) {
                bytes.extend(value.to_le_bytes());
            }
            bytes.extend(layer.weights.iter().map(|weight| weight.to_le_bytes()[0]));
        }
        seal(&mut bytes);
        bytes
    }

    /// Appends the checksum of `bytes`.
    pub(crate) fn seal(bytes: &mut Vec<u8>) {
        let sum = checksum(bytes);
        bytes.extend(sum.to_le_bytes());
    }

    /// Replaces the checksum after `bytes` was edited.
    fn reseal(bytes: &mut Vec<u8>) {
        bytes.truncate(bytes.len() - 4);
        seal(bytes);
    }

    /// Returns random valid layers for `dimensions`. Weights span the full
    /// permitted range, and multipliers are sized so that activations and
    /// scores take varied values.
    pub(crate) fn random_layers(rng: &mut Rng, dimensions: &[usize]) -> Vec<RawLayer> {
        dimensions
            .windows(2)
            .map(|pair| {
                let (cols, rows) = (pair[0], pair[1]);
                let mut bytes = vec![0u8; rows * cols];
                rng.fill(&mut bytes);
                RawLayer {
                    shift: 20,
                    bias: (0..rows)
                        .map(|_| (rng.next_u32() % 20_001) as i32 - 10_000)
                        .collect(),
                    multiplier: (0..rows)
                        .map(|_| (rng.next_u32() % 4_000) as i32 + 12_000_000 / cols as i32)
                        .collect(),
                    weights: bytes
                        .into_iter()
                        .map(|byte| i8::from_le_bytes([byte]).max(-127))
                        .collect(),
                }
            })
            .collect()
    }

    /// Returns a random valid model with the given hidden dimensions.
    pub(crate) fn random_model(rng: &mut Rng, hidden: &[usize]) -> Model {
        let mut dimensions = vec![FINGERPRINT_LEN];
        dimensions.extend(hidden);
        dimensions.push(Category::COUNT);
        let bytes = encode(&dimensions, &random_layers(rng, &dimensions));
        Model::from_bytes(&bytes).expect("random model is valid")
    }

    /// Returns a fingerprint with arbitrary features.
    pub(crate) fn random_fingerprint(rng: &mut Rng) -> Fingerprint {
        let mut features = [0u8; FINGERPRINT_LEN];
        rng.fill(&mut features);
        Fingerprint::from_array(features)
    }

    /// The smallest valid model: one layer whose only nonzero parameters are
    /// given by `edit`.
    fn linear(edit: impl FnOnce(&mut RawLayer)) -> Vec<u8> {
        let mut layer = RawLayer {
            shift: 1,
            bias: vec![0; Category::COUNT],
            multiplier: vec![0; Category::COUNT],
            weights: vec![0; Category::COUNT * FINGERPRINT_LEN],
        };
        edit(&mut layer);
        encode(&[FINGERPRINT_LEN, Category::COUNT], &[layer])
    }

    fn infer(bytes: &[u8], fingerprint: &Fingerprint) -> Scores {
        Model::from_bytes(bytes)
            .expect("model is valid")
            .infer(Verified::SCALAR, fingerprint)
    }

    #[test]
    fn logistic_table_matches_its_definition() {
        for (index, &score) in LOGISTIC.iter().enumerate() {
            let value = (index as f64 - 128.0) / 16.0;
            let expected = (255.0 / (1.0 + (-value).exp())).round();
            assert_eq!(f64::from(score), expected, "index {index}");
        }
        assert_eq!((LOGISTIC[0], LOGISTIC[128], LOGISTIC[255]), (0, 128, 255));
        assert!(LOGISTIC.windows(2).all(|pair| pair[0] <= pair[1]));
    }

    #[test]
    fn checksum_matches_known_fnv1a_vectors() {
        assert_eq!(checksum(b""), 0x811C_9DC5);
        assert_eq!(checksum(b"a"), 0xE40C_292C);
        assert_eq!(checksum(b"foobar"), 0xBF9C_F968);
    }

    #[test]
    fn zero_model_scores_the_logistic_midpoint() {
        let scores = infer(&linear(|_| {}), &Fingerprint::ZERO);
        assert_eq!(scores, Scores::from_array([128; Category::COUNT]));
    }

    #[test]
    fn output_layer_applies_bias_weights_rounding_and_clamp() {
        let bytes = linear(|layer| {
            layer.shift = 2;
            layer.multiplier = vec![1; Category::COUNT];
            // (bias + 2) >> 2, rounding half up toward positive infinity.
            layer.bias = vec![0, 1, 2, -1, -2, -3, 100_000, -100_000];
        });
        let scores = infer(&bytes, &Fingerprint::ZERO);
        let expected =
            [0i64, 0, 1, 0, 0, -1, 127, -128].map(|value| LOGISTIC[(value + 128) as usize]);
        assert_eq!(scores, Scores::from_array(expected));

        let bytes = linear(|layer| {
            layer.multiplier = vec![2; Category::COUNT];
            layer.weights[0] = 1;
            layer.weights[FINGERPRINT_LEN + 1] = -1;
            layer.weights[2 * FINGERPRINT_LEN + 2] = 127;
        });
        let mut features = [0u8; FINGERPRINT_LEN];
        features[..3].copy_from_slice(&[16, 32, 255]);
        let scores = infer(&bytes, &Fingerprint::from_array(features));
        assert_eq!(scores.get(Category::Malformed), LOGISTIC[128 + 16]);
        assert_eq!(scores.get(Category::Obfuscation), LOGISTIC[128 - 32]);
        assert_eq!(scores.get(Category::SqlInjection), 255);
        assert_eq!(scores.get(Category::MarkupInjection), 128);
    }

    #[test]
    fn hidden_layers_rectify_and_saturate() {
        // Hidden unit 0 copies feature 0, unit 1 negates it, unit 2 scales it
        // past 255. The output layer reads one hidden unit per category.
        let hidden = RawLayer {
            shift: 1,
            bias: vec![0; 3],
            multiplier: vec![2; 3],
            weights: {
                let mut weights = vec![0i8; 3 * FINGERPRINT_LEN];
                weights[0] = 1;
                weights[FINGERPRINT_LEN] = -1;
                weights[2 * FINGERPRINT_LEN] = 100;
                weights
            },
        };
        let output = RawLayer {
            shift: 2,
            bias: vec![0; Category::COUNT],
            multiplier: vec![1; Category::COUNT],
            weights: {
                let mut weights = vec![0i8; Category::COUNT * 3];
                weights[0] = 1;
                weights[3 + 1] = 1;
                weights[6 + 2] = 1;
                weights
            },
        };
        let bytes = encode(&[FINGERPRINT_LEN, 3, Category::COUNT], &[hidden, output]);
        let mut features = [0u8; FINGERPRINT_LEN];
        features[0] = 40;
        let scores = infer(&bytes, &Fingerprint::from_array(features));
        assert_eq!(scores.get(Category::Malformed), LOGISTIC[128 + 10]);
        assert_eq!(scores.get(Category::Obfuscation), LOGISTIC[128]);
        assert_eq!(scores.get(Category::SqlInjection), LOGISTIC[128 + 64]);
    }

    #[test]
    fn extreme_valid_parameters_do_not_overflow() {
        // The largest accumulator magnitude that validation admits, the
        // largest multiplier, and both ends of the shift range.
        let max_dot = 255 * 127 * FINGERPRINT_LEN as i32;
        for shift in [1, MAX_SHIFT] {
            for sign in [1i32, -1] {
                let bytes = linear(|layer| {
                    layer.shift = shift;
                    layer.multiplier = vec![i32::MAX; Category::COUNT];
                    layer.bias = vec![sign * (i32::MAX - max_dot); Category::COUNT];
                    layer.weights.fill(if sign > 0 { 127 } else { -127 });
                });
                let scores = infer(&bytes, &Fingerprint::from_array([255; FINGERPRINT_LEN]));
                let expected = match (shift, sign) {
                    (1, 1) => 255,
                    (1, _) => 0,
                    // (2^31 - 1)^2 rounds to 1 after a shift by 62, and its
                    // negation rounds to -1.
                    (_, 1) => LOGISTIC[129],
                    _ => LOGISTIC[127],
                };
                assert_eq!(scores, Scores::from_array([expected; Category::COUNT]));
            }
        }
    }

    #[test]
    fn inference_is_deterministic_for_random_models() {
        let mut rng = Rng::new(5);
        for hidden in [&[][..], &[1], &[64], &[64, 64], &[256, 33, 7]] {
            let model = random_model(&mut rng, hidden);
            assert_eq!(model.layer_sizes().len(), hidden.len() + 1);
            assert_eq!(model.layer_sizes().last(), Some(Category::COUNT));
            let mut distinct = std::collections::HashSet::new();
            for _ in 0..16 {
                let fingerprint = random_fingerprint(&mut rng);
                let scores = model.infer(Verified::SCALAR, &fingerprint);
                assert_eq!(scores, model.infer(Verified::SCALAR, &fingerprint));
                distinct.insert(scores);
            }
            // Random models must produce varied scores, or tests that compare
            // backends through them would compare constants. A layer of one
            // unit can saturate the same way for every input.
            assert!(
                distinct.len() > 1 || hidden.contains(&1),
                "hidden {hidden:?}"
            );
        }
    }

    #[test]
    fn rejects_header_errors() {
        let valid = linear(|_| {});
        assert!(Model::from_bytes(&valid).is_ok());

        let mut bytes = valid.clone();
        bytes[0] = b'X';
        assert_eq!(Model::from_bytes(&bytes), Err(ModelError::BadMagic));

        let mut bytes = valid.clone();
        bytes[4] = 2;
        assert_eq!(
            Model::from_bytes(&bytes),
            Err(ModelError::UnsupportedVersion)
        );

        let mut bytes = valid.clone();
        bytes[7] = 1;
        assert_eq!(Model::from_bytes(&bytes), Err(ModelError::InvalidParameter));

        for layer_count in [0, MAX_LAYERS as u8 + 1, 255] {
            let mut bytes = valid.clone();
            bytes[6] = layer_count;
            assert_eq!(
                Model::from_bytes(&bytes),
                Err(ModelError::InvalidDimensions)
            );
        }
    }

    #[test]
    fn rejects_invalid_dimensions() {
        let layers = |dimensions: &[usize]| -> Vec<RawLayer> {
            dimensions
                .windows(2)
                .map(|pair| RawLayer {
                    shift: 1,
                    bias: vec![0; pair[1]],
                    multiplier: vec![0; pair[1]],
                    weights: vec![0; pair[0] * pair[1]],
                })
                .collect()
        };
        let cases: [&[usize]; 6] = [
            &[FINGERPRINT_LEN - 1, Category::COUNT],
            &[FINGERPRINT_LEN, Category::COUNT + 1],
            &[FINGERPRINT_LEN, 0, Category::COUNT],
            &[FINGERPRINT_LEN, MAX_HIDDEN + 1, Category::COUNT],
            &[FINGERPRINT_LEN, 4, 0, Category::COUNT],
            &[0, Category::COUNT],
        ];
        for dimensions in cases {
            let bytes = encode(dimensions, &layers(dimensions));
            assert_eq!(
                Model::from_bytes(&bytes),
                Err(ModelError::InvalidDimensions),
                "dimensions {dimensions:?}"
            );
        }
        let dimensions = [FINGERPRINT_LEN, MAX_HIDDEN, 1, MAX_HIDDEN, Category::COUNT];
        assert!(Model::from_bytes(&encode(&dimensions, &layers(&dimensions))).is_ok());
    }

    #[test]
    fn rejects_every_truncation_and_any_extension() {
        let mut rng = Rng::new(9);
        let dimensions = [FINGERPRINT_LEN, 3, Category::COUNT];
        let valid = encode(&dimensions, &random_layers(&mut rng, &dimensions));
        assert!(Model::from_bytes(&valid).is_ok());
        for len in 0..valid.len() {
            assert_eq!(
                Model::from_bytes(&valid[..len]),
                Err(ModelError::Truncated),
                "length {len}"
            );
        }
        let mut extended = valid.clone();
        extended.push(0);
        assert_eq!(Model::from_bytes(&extended), Err(ModelError::TrailingData));
    }

    #[test]
    fn rejects_any_single_bit_flip() {
        let mut rng = Rng::new(13);
        let dimensions = [FINGERPRINT_LEN, 2, Category::COUNT];
        let valid = encode(&dimensions, &random_layers(&mut rng, &dimensions));
        for index in 0..valid.len() {
            let mut bytes = valid.clone();
            bytes[index] ^= 1 << (index % 8);
            assert!(Model::from_bytes(&bytes).is_err(), "byte {index}");
        }
    }

    #[test]
    fn rejects_out_of_range_parameters() {
        let header = 4 + 2 + 1 + 1 + 4;
        for shift in [0, MAX_SHIFT + 1, 255] {
            let bytes = linear(|layer| layer.shift = shift);
            assert_eq!(Model::from_bytes(&bytes), Err(ModelError::InvalidParameter));
        }
        for reserved in 1..4 {
            let mut bytes = linear(|_| {});
            bytes[header + reserved] = 1;
            reseal(&mut bytes);
            assert_eq!(Model::from_bytes(&bytes), Err(ModelError::InvalidParameter));
        }
        let bytes = linear(|layer| layer.multiplier[3] = -1);
        assert_eq!(Model::from_bytes(&bytes), Err(ModelError::InvalidParameter));
        let bytes = linear(|layer| layer.weights[5] = -128);
        assert_eq!(Model::from_bytes(&bytes), Err(ModelError::InvalidParameter));
        let bytes = linear(|layer| layer.multiplier = vec![i32::MAX; Category::COUNT]);
        assert!(Model::from_bytes(&bytes).is_ok());
    }

    #[test]
    fn rejects_layers_that_could_overflow() {
        let max_dot = 255 * 127 * FINGERPRINT_LEN as i32;
        for sign in [1, -1] {
            let bytes = linear(|layer| {
                layer.weights[..FINGERPRINT_LEN].fill(127 * sign as i8);
                layer.bias[0] = sign * (i32::MAX - max_dot);
            });
            assert!(Model::from_bytes(&bytes).is_ok());
            let bytes = linear(|layer| {
                layer.weights[..FINGERPRINT_LEN].fill(-127 * sign as i8);
                layer.bias[0] = sign * (i32::MAX - max_dot + 1);
            });
            assert_eq!(
                Model::from_bytes(&bytes),
                Err(ModelError::AccumulatorOverflow)
            );
        }
        let bytes = linear(|layer| layer.bias[7] = i32::MIN);
        assert_eq!(
            Model::from_bytes(&bytes),
            Err(ModelError::AccumulatorOverflow)
        );
        // The bound applies to each output on its own: the largest weights
        // of one row and the largest bias of another row are compatible.
        let bytes = linear(|layer| {
            layer.weights[..FINGERPRINT_LEN].fill(127);
            layer.bias[1] = i32::MAX;
            layer.bias[2] = -i32::MAX;
        });
        assert!(Model::from_bytes(&bytes).is_ok());
    }

    #[test]
    fn arbitrary_bytes_never_panic() {
        let mut rng = Rng::new(17);
        let valid = linear(|_| {});
        for round in 0..2_000 {
            let mut bytes = vec![0u8; (rng.next_u32() % 64) as usize];
            rng.fill(&mut bytes);
            if round % 2 == 0 {
                // Keep a valid prefix so that parsing reaches later fields.
                let keep = (rng.next_u32() as usize % 14).min(bytes.len());
                bytes[..keep].copy_from_slice(&valid[..keep]);
            }
            let _ = Model::from_bytes(&bytes);
        }
    }

    #[test]
    fn errors_have_fixed_messages() {
        let errors = [
            ModelError::Truncated,
            ModelError::BadMagic,
            ModelError::UnsupportedVersion,
            ModelError::InvalidDimensions,
            ModelError::TrailingData,
            ModelError::ChecksumMismatch,
            ModelError::InvalidParameter,
            ModelError::AccumulatorOverflow,
        ];
        for (index, error) in errors.iter().enumerate() {
            let message = error.to_string();
            assert!(message.starts_with("model "));
            assert!(
                errors[..index]
                    .iter()
                    .all(|other| other.to_string() != message)
            );
        }
    }
}
