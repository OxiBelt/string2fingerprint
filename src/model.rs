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
//! | layout version length `a` | 1 | 1 to [`MAX_VERSION_LEN`](crate::version::MAX_VERSION_LEN) |
//! | layout version | `a` | a [`Version`] in ASCII |
//! | library requirement length `b` | 1 | 1 to [`MAX_REQUIREMENT_LEN`] |
//! | library requirement | `b` | a [`VersionReq`] in ASCII, in canonical form |
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
//!
//! # Compatibility
//!
//! The two versions are the [`Metadata`] of a model. The layout version is
//! the [`LAYOUT_VERSION`] of the fingerprint that the model was trained on,
//! and the library requirement names the versions of this crate that may
//! load the model. [`Model::from_bytes`] refuses a model unless
//! [`LAYOUT_VERSION`] matches the requirement `^layout` and
//! [`VERSION`](crate::VERSION) matches the library requirement, so that a
//! model is never used with a fingerprint or a network that it was not
//! trained for. [`Metadata::read`] reads the versions of any model without
//! checking them.
//!
//! ```
//! use string2fingerprint::fingerprint::LAYOUT_VERSION;
//! use string2fingerprint::model::{EmbeddedModel, Metadata, Model, ModelError};
//! use string2fingerprint::version::VersionReq;
//!
//! let bytes = EmbeddedModel::Compact.bytes();
//! let metadata = Metadata::read(bytes)?;
//! assert_eq!(metadata.layout_version(), &LAYOUT_VERSION);
//! assert!(metadata.library_requirement().matches(&string2fingerprint::VERSION));
//!
//! // A model for releases from 9.0 on does not load in this crate.
//! let future: VersionReq = ">=9.0.0".parse().expect("valid requirement");
//! let model = Model::from_bytes(bytes)?;
//! assert_eq!(model.with_library_requirement(future), Err(ModelError::UnsupportedLibrary));
//! # Ok::<(), ModelError>(())
//! ```
//!
//! Format version 1 had neither field. Such a model reads as layout 1.0.0
//! with the library requirement `^0.1.0-dev`, which is what every revision
//! of the crate that wrote format 1 extracted and would have written, and
//! [`Model::to_bytes`] writes it in format 2.
//!
//! # Inference
//!
//! [`Model::score`] runs the network on one fingerprint with a
//! [`Kernel`]. [`Model::logits`] stops before the logistic table and
//! returns the [`Logits`] of the output layer, and [`Model::trace`] also
//! keeps the accumulators and activations of every layer.
//!
//! ```
//! use string2fingerprint::fingerprint::fingerprints;
//! use string2fingerprint::kernel::Kernel;
//! use string2fingerprint::model::Model;
//!
//! let model = Model::embedded();
//! let window = fingerprints(b"1' OR '1'='1").next().expect("input has a window");
//! let trace = model.trace(Kernel::detect(), &window);
//! assert_eq!(trace.logits(), model.logits(Kernel::SCALAR, &window));
//! assert_eq!(trace.logits().scores(), model.score(Kernel::SCALAR, &window));
//! assert_eq!(trace.activations(0).map(<[u8]>::len), Some(64));
//! ```

use core::fmt;
use core::ops::Index;

use crate::backend::Verified;
use crate::fingerprint::{FINGERPRINT_LEN, Fingerprint, LAYOUT_VERSION};
use crate::kernel::{self, Kernel, Matrix};
use crate::version::{MAX_REQUIREMENT_LEN, Version, VersionReq};
use crate::{Category, Scores};

/// The format version that this crate writes. It also reads format
/// version 1, which has no [`Metadata`].
pub const FORMAT_VERSION: u16 = 2;

/// The format version without [`Metadata`].
const LEGACY_FORMAT_VERSION: u16 = 1;

/// The crate version that wrote every model of format version 1.
const LEGACY_LIBRARY_VERSION: Version = match Version::parse("0.1.0-dev") {
    Ok(version) => version,
    Err(_) => panic!("the legacy library version is valid"),
};

/// The largest number of layers in a model.
pub const MAX_LAYERS: usize = 4;

/// The largest dimension of a hidden layer.
pub const MAX_HIDDEN: usize = 256;

/// The largest shift of a layer.
pub const MAX_SHIFT: u8 = 62;

const MAGIC: [u8; 4] = *b"S2FP";

// The format stores the layer count in 8 bits and a dimension in 16 bits.
const _: () = assert!(MAX_LAYERS <= u8::MAX as usize);
const _: () = assert!(FINGERPRINT_LEN <= u16::MAX as usize);
const _: () = assert!(MAX_HIDDEN <= u16::MAX as usize);

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

/// Returns the score of an output-layer value, in units of 1/16:
/// `round(255 / (1 + exp(-value / 16)))`.
///
/// The value 0 scores 128, and the table rises from 0 at -128 to 255 at
/// 127.
#[must_use]
pub const fn score_of_logit(value: i8) -> u8 {
    LOGISTIC[(value as i16 + 128) as usize]
}

/// The output-layer values of one fingerprint, one for each [`Category`],
/// before the logistic table turns them into [`Scores`].
///
/// A value is the logit of the category in units of 1/16, clamped to the
/// range from -128 to 127, so 0 is the boundary at which a score is 128.
/// Logits keep the order of scores and resolve differences that the table
/// rounds away near 0 and 255.
///
/// There is no `Default`: the logits of an input without a window are
/// [`Logits::NONE`], which scores 0, while logits of 0 score 128.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Logits([i8; Category::COUNT]);

impl Logits {
    /// The lowest logit in every category, which scores 0. It is the logits
    /// of an input without a window, and the start of a fold with
    /// [`Logits::max_with`].
    pub const NONE: Self = Self([i8::MIN; Category::COUNT]);

    /// Creates logits from an array indexed by [`Category::index`].
    #[must_use]
    pub const fn from_array(values: [i8; Category::COUNT]) -> Self {
        Self(values)
    }

    /// Returns the logits as an array indexed by [`Category::index`].
    #[must_use]
    pub const fn as_array(&self) -> &[i8; Category::COUNT] {
        &self.0
    }

    /// Returns the logit of `category`.
    #[must_use]
    pub const fn get(&self, category: Category) -> i8 {
        self.0[category.index()]
    }

    /// Returns the logit of `category` as a float: the stored value divided
    /// by 16.
    #[must_use]
    pub fn value(&self, category: Category) -> f32 {
        f32::from(self.get(category)) / 16.0
    }

    /// Returns the scores of the logits through [`score_of_logit`].
    #[must_use]
    pub const fn scores(&self) -> Scores {
        let mut scores = [0u8; Category::COUNT];
        let mut index = 0;
        while index < Category::COUNT {
            scores[index] = score_of_logit(self.0[index]);
            index += 1;
        }
        Scores::from_array(scores)
    }

    /// Iterates over every category and its logit, ordered by discriminant.
    pub fn iter(&self) -> impl Iterator<Item = (Category, i8)> + '_ {
        Category::ALL.into_iter().zip(self.0)
    }

    /// Keeps, for each category, the higher of the two logits.
    #[must_use]
    pub fn max_with(mut self, other: Self) -> Self {
        for (value, other_value) in self.0.iter_mut().zip(other.0) {
            *value = (*value).max(other_value);
        }
        self
    }
}

impl Index<Category> for Logits {
    type Output = i8;

    fn index(&self, category: Category) -> &i8 {
        &self.0[category.index()]
    }
}

impl From<Logits> for Scores {
    fn from(logits: Logits) -> Self {
        logits.scores()
    }
}

/// The values of every layer of a model for one fingerprint, which
/// [`Model::trace`] records.
///
/// For each layer it keeps the accumulators, which are the biases plus the
/// dot products of the inputs with the weights, before the multiplier and
/// the shift. For each hidden layer it also keeps the activations that the
/// next layer reads. It holds about 5 KiB and no heap memory.
#[derive(Clone, PartialEq, Eq)]
pub struct Trace {
    layers: usize,
    sizes: [usize; MAX_LAYERS],
    accumulators: [[i32; MAX_HIDDEN]; MAX_LAYERS],
    activations: [[u8; MAX_HIDDEN]; MAX_LAYERS - 1],
    logits: Logits,
}

impl fmt::Debug for Trace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Trace")
            .field("layer_sizes", &&self.sizes[..self.layers])
            .field("logits", &self.logits)
            .finish_non_exhaustive()
    }
}

impl Trace {
    const EMPTY: Self = Self {
        layers: 0,
        sizes: [0; MAX_LAYERS],
        accumulators: [[0; MAX_HIDDEN]; MAX_LAYERS],
        activations: [[0; MAX_HIDDEN]; MAX_LAYERS - 1],
        logits: Logits([0; Category::COUNT]),
    };

    /// Returns the number of layers, including the output layer.
    #[must_use]
    pub const fn layer_count(&self) -> usize {
        self.layers
    }

    /// Returns the accumulator of each output of layer `layer`, or `None`
    /// when the model has no such layer.
    ///
    /// Validation bounds every accumulator by `i32::MAX` in magnitude.
    #[must_use]
    pub fn accumulators(&self, layer: usize) -> Option<&[i32]> {
        (layer < self.layers).then(|| &self.accumulators[layer][..self.sizes[layer]])
    }

    /// Returns the activations of hidden layer `layer`, or `None` for the
    /// output layer and beyond.
    #[must_use]
    pub fn activations(&self, layer: usize) -> Option<&[u8]> {
        (layer < self.layers.saturating_sub(1))
            .then(|| &self.activations[layer][..self.sizes[layer]])
    }

    /// Returns the logits of the output layer.
    #[must_use]
    pub const fn logits(&self) -> Logits {
        self.logits
    }

    /// Returns the scores of the output layer.
    #[must_use]
    pub const fn scores(&self) -> Scores {
        self.logits.scores()
    }
}

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
    /// The format version is neither 1 nor [`FORMAT_VERSION`].
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
    /// The layout version is not a valid [`Version`], or the library
    /// requirement is not a valid [`VersionReq`] in canonical form.
    InvalidMetadata,
    /// The model was trained on a fingerprint layout that
    /// [`LAYOUT_VERSION`] is not compatible with.
    IncompatibleLayout,
    /// [`VERSION`](crate::VERSION) does not match the library requirement
    /// of the model.
    UnsupportedLibrary,
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
            Self::InvalidMetadata => "model version metadata is not valid",
            Self::IncompatibleLayout => "model was trained on an incompatible fingerprint layout",
            Self::UnsupportedLibrary => "model does not support this library version",
        })
    }
}

impl core::error::Error for ModelError {}

/// The versions that a model states: the fingerprint layout it was trained
/// on, and the versions of this crate that may load it.
///
/// See [Compatibility](self#compatibility). [`Metadata::current`] is the
/// metadata of a model that this crate trains or builds from layers.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Metadata {
    layout_version: Version,
    library_requirement: VersionReq,
}

impl Metadata {
    /// Creates metadata from a layout version and a library requirement.
    ///
    /// The metadata is not checked here. [`Model::with_metadata`] checks
    /// it.
    #[must_use]
    pub fn new(layout_version: Version, library_requirement: VersionReq) -> Self {
        Self {
            layout_version,
            library_requirement,
        }
    }

    /// Returns the metadata of a model trained by this crate: the layout
    /// version [`LAYOUT_VERSION`], and the library requirement
    /// `^VERSION`, [`VersionReq::caret`] of [`VERSION`](crate::VERSION).
    #[must_use]
    pub fn current() -> Self {
        Self::new(LAYOUT_VERSION, VersionReq::caret(&crate::VERSION))
    }

    /// Returns the metadata that a model of format version 1 reads as.
    fn legacy() -> Self {
        Self::new(
            Version::new(1, 0, 0),
            VersionReq::caret(&LEGACY_LIBRARY_VERSION),
        )
    }

    /// Reads the metadata of the model in `bytes` without checking it.
    ///
    /// The header, the length, and the checksum are validated as by
    /// [`Model::from_bytes`], and the layers are not read. A caller can
    /// show the versions of a model that does not load, or choose a model
    /// file by them.
    ///
    /// # Errors
    ///
    /// Returns the first [`ModelError`] of the header, the length, the
    /// checksum, or the metadata.
    pub fn read(bytes: &[u8]) -> Result<Self, ModelError> {
        Header::read(bytes)?.metadata()
    }

    /// Returns the version of the fingerprint layout that the model was
    /// trained on.
    #[must_use]
    pub fn layout_version(&self) -> &Version {
        &self.layout_version
    }

    /// Returns the requirement on the versions of this crate that may load
    /// the model.
    #[must_use]
    pub fn library_requirement(&self) -> &VersionReq {
        &self.library_requirement
    }

    /// Returns whether [`LAYOUT_VERSION`] is compatible with the layout of
    /// the model: whether it matches the requirement `^layout`.
    #[must_use]
    pub fn is_layout_compatible(&self) -> bool {
        VersionReq::caret(&self.layout_version).matches(&LAYOUT_VERSION)
    }

    /// Returns whether [`VERSION`](crate::VERSION) matches the library
    /// requirement.
    #[must_use]
    pub fn supports_this_library(&self) -> bool {
        self.library_requirement.matches(&crate::VERSION)
    }

    /// Checks that this crate may use a model with this metadata.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::IncompatibleLayout`] unless the layout is
    /// compatible, and otherwise [`ModelError::UnsupportedLibrary`] unless
    /// this crate is supported.
    pub fn check(&self) -> Result<(), ModelError> {
        if !self.is_layout_compatible() {
            return Err(ModelError::IncompatibleLayout);
        }
        if !self.supports_this_library() {
            return Err(ModelError::UnsupportedLibrary);
        }
        Ok(())
    }

    /// Passes the encoding of the two versions, each after its length, to
    /// `sink`.
    fn encode(&self, mut sink: impl FnMut(&[u8])) {
        let layout = self.layout_version.as_str().as_bytes();
        // A version is at most `MAX_VERSION_LEN` bytes long.
        sink(&[layout.len() as u8]);
        sink(layout);
        let mut buffer = [0u8; MAX_REQUIREMENT_LEN];
        let requirement = self.library_requirement.write_canonical(&mut buffer);
        // A canonical requirement is at most `MAX_REQUIREMENT_LEN` bytes
        // long.
        sink(&[requirement.len() as u8]);
        sink(requirement);
    }

    /// Returns the length of [`Metadata::encode`].
    fn encoded_len(&self) -> usize {
        let mut buffer = [0u8; MAX_REQUIREMENT_LEN];
        2 + self.layout_version.as_str().len()
            + self.library_requirement.write_canonical(&mut buffer).len()
    }
}

impl Default for Metadata {
    /// Returns [`Metadata::current`].
    fn default() -> Self {
        Self::current()
    }
}

/// The parameters of one fully connected layer, as the format stores them.
///
/// A layer with `n` inputs has one bias and one multiplier for each of its
/// `m` outputs and `m * n` weights. [`Model::from_layers`] validates the
/// parameters, and [`Model::layers`] returns those of a model.
///
/// The fields are public so that a layer can be written as a literal. They
/// are the fields of format version [`FORMAT_VERSION`], and a format that
/// stores more for each layer would change this type.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct LayerParameters {
    /// The right shift that follows the multiplication, from 1 to
    /// [`MAX_SHIFT`].
    pub shift: u8,
    /// The bias of each output.
    pub biases: Vec<i32>,
    /// The multiplier of each output, which is not negative.
    pub multipliers: Vec<i32>,
    /// The weights in row-major order: all inputs of the first output, then
    /// of the second, and so on. No weight is -128.
    pub weights: Vec<i8>,
}

/// A borrowed view of the parameters of one layer of a [`Model`].
///
/// [`Model::layer`] returns it without copying, where [`Model::layers`]
/// copies the parameters into [`LayerParameters`]. `Debug` shows the
/// dimensions and the shift, not the parameters.
#[derive(Clone, Copy)]
pub struct LayerView<'a> {
    layer: &'a Layer,
}

impl fmt::Debug for LayerView<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LayerView")
            .field("inputs", &self.inputs())
            .field("outputs", &self.outputs())
            .field("shift", &self.shift())
            .finish_non_exhaustive()
    }
}

impl<'a> LayerView<'a> {
    /// Returns the number of inputs.
    #[must_use]
    pub fn inputs(&self) -> usize {
        self.layer.matrix.cols()
    }

    /// Returns the number of outputs.
    #[must_use]
    pub fn outputs(&self) -> usize {
        self.layer.matrix.rows()
    }

    /// Returns the right shift that follows the multiplication.
    #[must_use]
    pub fn shift(&self) -> u8 {
        self.layer.shift
    }

    /// Returns the bias of each output.
    #[must_use]
    pub fn biases(&self) -> &'a [i32] {
        &self.layer.bias
    }

    /// Returns the multiplier of each output.
    #[must_use]
    pub fn multipliers(&self) -> &'a [i32] {
        &self.layer.multiplier
    }

    /// Returns the weights in row-major order.
    #[must_use]
    pub fn weights(&self) -> &'a [i8] {
        self.layer.matrix.weights()
    }

    /// Returns the weights of output `row`, one for each input, or `None`
    /// when there is no such output.
    #[must_use]
    pub fn row(&self, row: usize) -> Option<&'a [i8]> {
        self.weights().chunks_exact(self.inputs()).nth(row)
    }

    /// Returns a copy of the parameters.
    #[must_use]
    pub fn to_parameters(&self) -> LayerParameters {
        LayerParameters {
            shift: self.layer.shift,
            biases: self.layer.bias.to_vec(),
            multipliers: self.layer.multiplier.to_vec(),
            weights: self.layer.matrix.weights().to_vec(),
        }
    }
}

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
///
/// `Debug` shows the layer sizes, the number of parameters, and the
/// [`Metadata`], not the parameters, and reads no weight.
#[derive(Clone, PartialEq, Eq)]
pub struct Model {
    /// Never empty. Consecutive dimensions match, the first layer has
    /// [`FINGERPRINT_LEN`] inputs, and the last has [`Category::COUNT`]
    /// outputs.
    layers: Box<[Layer]>,
    /// Passes [`Metadata::check`].
    metadata: Metadata,
}

impl fmt::Debug for Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sizes = self.layer_sizes();
        let mut layers = [0; MAX_LAYERS];
        let count = sizes.len();
        for (slot, size) in layers.iter_mut().zip(sizes) {
            *slot = size;
        }
        f.debug_struct("Model")
            .field("layer_sizes", &&layers[..count])
            .field("parameters", &self.parameter_count())
            .field(
                "layout_version",
                &format_args!("{}", self.metadata.layout_version),
            )
            .field(
                "library_requirement",
                &format_args!("{}", self.metadata.library_requirement),
            )
            .finish_non_exhaustive()
    }
}

/// The bytes of the default model that the crate embeds.
///
/// The model was trained on synthetic data with the tools in `tools/train/`,
/// whose documentation lists the procedure and the measured limits.
pub const DEFAULT_MODEL: &[u8] = include_bytes!("../models/default.s2fp");

static COMPACT_MODEL: &[u8] = include_bytes!("../models/compact.s2fp");
static LARGE_MODEL: &[u8] = include_bytes!("../models/large.s2fp");
static STRICT_MODEL: &[u8] = include_bytes!("../models/strict.s2fp");

/// A model that the crate embeds.
///
/// Every embedded model reads the same fingerprint and returns a score for
/// every [`Category`]. They differ in size, in the time a score takes, and
/// in the balance between flagged benign text and missed attacks. All were
/// trained on the same synthetic data with the tools in `tools/train/`,
/// whose documentation lists the presets and the measured results of each.
///
/// ```
/// use string2fingerprint::model::EmbeddedModel;
/// use string2fingerprint::Scorer;
///
/// let scorer = Scorer::with_embedded(EmbeddedModel::Compact);
/// assert_eq!(EmbeddedModel::from_name("compact"), Some(EmbeddedModel::Compact));
/// # let _ = scorer;
/// ```
///
/// A program that calls only [`Model::embedded`] or
/// [`Scorer::new`](crate::Scorer::new) links only the default model.
/// [`EmbeddedModel::bytes`] and the functions that call it link every
/// embedded model, about 1.1 MiB in total.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum EmbeddedModel {
    /// One hidden layer of 32 units, about 97 KiB. The fastest and the
    /// least accurate model.
    Compact,
    /// Two hidden layers of 64 units, about 198 KiB. [`Model::embedded`]
    /// returns it.
    Default,
    /// Hidden layers of 192 and 96 units, about 597 KiB. The best
    /// separation, at about three times the cost of the default model.
    Large,
    /// The shape of the default model, trained with benign samples weighted
    /// four times. It flags less benign text and misses more attacks, for
    /// hosts that block on a score.
    Strict,
}

impl EmbeddedModel {
    /// Every embedded model, from the smallest to the largest, then the
    /// specialized ones.
    pub const ALL: [Self; 4] = [Self::Compact, Self::Default, Self::Large, Self::Strict];

    /// Returns the stable lowercase name of the model.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Compact => "compact",
            Self::Default => "default",
            Self::Large => "large",
            Self::Strict => "strict",
        }
    }

    /// Returns the model whose [`name`](Self::name) is `name`.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|model| model.name() == name)
    }

    /// Returns the bytes of the model in the format of this module.
    #[must_use]
    pub fn bytes(self) -> &'static [u8] {
        match self {
            Self::Compact => COMPACT_MODEL,
            Self::Default => DEFAULT_MODEL,
            Self::Large => LARGE_MODEL,
            Self::Strict => STRICT_MODEL,
        }
    }

    /// Parses the model.
    ///
    /// Each call parses [`EmbeddedModel::bytes`] and allocates about their
    /// size.
    ///
    /// # Panics
    ///
    /// Never in a build that passed the crate's tests, for the reason that
    /// [`Model::embedded`] gives.
    #[must_use]
    pub fn load(self) -> Model {
        Model::from_bytes(self.bytes()).expect("every embedded model is valid")
    }
}

impl fmt::Display for EmbeddedModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl Model {
    /// Returns the default model that the crate embeds.
    ///
    /// Treat its scores as one signal among several. It was trained on
    /// synthetic data and has not been measured on real traffic.
    ///
    /// Each call parses [`DEFAULT_MODEL`] and allocates about 200 KiB.
    ///
    /// # Panics
    ///
    /// Never in a build that passed the crate's tests. The model bytes are
    /// compiled into the crate, so parsing them has the same result on every
    /// call, and a test checks that they are valid.
    #[must_use]
    pub fn embedded() -> Self {
        Self::from_bytes(DEFAULT_MODEL).expect("the embedded model is valid")
    }

    /// Parses and validates a model.
    ///
    /// Parsing reads `bytes` once, allocates at most the size of `bytes`
    /// plus a small constant, and never panics.
    ///
    /// # Errors
    ///
    /// Returns the first [`ModelError`] found: of the header, the length,
    /// the checksum, the [`Metadata`] and its [check](Metadata::check), and
    /// then the layers. See the [module documentation](self) for the format
    /// and its constraints.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ModelError> {
        let header = Header::read(bytes)?;
        let metadata = header.metadata()?;
        metadata.check()?;
        let layer_count = header.layer_count;
        let dimensions = &header.dimensions[..=layer_count];
        let mut reader = header.reader;
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
            metadata,
        })
    }

    /// Validates the parameters of every layer, from the first hidden layer
    /// to the output layer, and creates the model.
    ///
    /// The number of outputs of a layer is the length of its biases. The
    /// first layer has [`FINGERPRINT_LEN`] inputs, and every other layer has
    /// as many inputs as the layer before it has outputs. The constraints
    /// are those of the [format](self), so that
    /// `Model::from_layers(layers)?.to_bytes()` is a valid model file. The
    /// model has the [`Metadata::current`] of this crate, and
    /// [`Model::with_metadata`] replaces it.
    ///
    /// ```
    /// use string2fingerprint::Category;
    /// use string2fingerprint::fingerprint::FINGERPRINT_LEN;
    /// use string2fingerprint::model::{LayerParameters, Model};
    ///
    /// // A model with one layer and no evidence: every score is 128.
    /// let layer = LayerParameters {
    ///     shift: 1,
    ///     biases: vec![0; Category::COUNT],
    ///     multipliers: vec![0; Category::COUNT],
    ///     weights: vec![0; Category::COUNT * FINGERPRINT_LEN],
    /// };
    /// let model = Model::from_layers(vec![layer])?;
    /// assert_eq!(Model::from_bytes(&model.to_bytes())?, model);
    /// # Ok::<(), string2fingerprint::model::ModelError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidDimensions`] when the layer count, the
    /// number of outputs of any layer, or the length of its multipliers or
    /// weights does not fit, and otherwise the first [`ModelError`] that
    /// [`Model::from_bytes`] would return for the encoded parameters.
    pub fn from_layers(layers: Vec<LayerParameters>) -> Result<Self, ModelError> {
        let layer_count = layers.len();
        if !(1..=MAX_LAYERS).contains(&layer_count) {
            return Err(ModelError::InvalidDimensions);
        }
        // Every dimension is checked before any parameter, as
        // `Model::from_bytes` checks the header before the layers.
        let mut cols = FINGERPRINT_LEN;
        for (index, layer) in layers.iter().enumerate() {
            let rows = layer.biases.len();
            let rows_fit = if index + 1 == layer_count {
                rows == Category::COUNT
            } else {
                (1..=MAX_HIDDEN).contains(&rows)
            };
            if !rows_fit || layer.multipliers.len() != rows || layer.weights.len() != rows * cols {
                return Err(ModelError::InvalidDimensions);
            }
            cols = rows;
        }
        let mut cols = FINGERPRINT_LEN;
        let mut validated = Vec::with_capacity(layer_count);
        for layer in layers {
            let rows = layer.biases.len();
            validated.push(Layer::new(
                layer.weights.into(),
                cols,
                layer.biases.into(),
                layer.multipliers.into(),
                layer.shift,
            )?);
            cols = rows;
        }
        Ok(Self {
            layers: validated.into(),
            metadata: Metadata::current(),
        })
    }

    /// Returns the [`Metadata`] of the model.
    #[must_use]
    pub fn metadata(&self) -> &Metadata {
        &self.metadata
    }

    /// Returns the version of the fingerprint layout that the model was
    /// trained on.
    #[must_use]
    pub fn layout_version(&self) -> &Version {
        self.metadata.layout_version()
    }

    /// Returns the requirement on the versions of this crate that may load
    /// the model.
    #[must_use]
    pub fn library_requirement(&self) -> &VersionReq {
        self.metadata.library_requirement()
    }

    /// Replaces the [`Metadata`] of the model.
    ///
    /// Use it to state the versions of the crate that the model supports,
    /// or to keep the metadata of a model that was rebuilt from its layers.
    ///
    /// # Errors
    ///
    /// Returns the error of [`Metadata::check`], so that a model always
    /// loads in this crate.
    pub fn with_metadata(mut self, metadata: Metadata) -> Result<Self, ModelError> {
        metadata.check()?;
        self.metadata = metadata;
        Ok(self)
    }

    /// Replaces the library requirement of the model and keeps its layout
    /// version.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::UnsupportedLibrary`] when
    /// [`VERSION`](crate::VERSION) does not match `requirement`.
    pub fn with_library_requirement(self, requirement: VersionReq) -> Result<Self, ModelError> {
        let layout_version = self.metadata.layout_version;
        self.with_metadata(Metadata::new(layout_version, requirement))
    }

    /// Returns a copy of the parameters of every layer, from the first
    /// hidden layer to the output layer.
    ///
    /// [`Model::from_layers`] creates an equal model from them.
    pub fn layers(&self) -> impl ExactSizeIterator<Item = LayerParameters> + '_ {
        self.layer_views().map(|view| view.to_parameters())
    }

    /// Returns a view of layer `index`, counted from the first hidden layer,
    /// or `None` when there is no such layer.
    #[must_use]
    pub fn layer(&self, index: usize) -> Option<LayerView<'_>> {
        self.layers.get(index).map(|layer| LayerView { layer })
    }

    /// Iterates over a view of every layer, from the first hidden layer to
    /// the output layer.
    pub fn layer_views(&self) -> impl ExactSizeIterator<Item = LayerView<'_>> + '_ {
        self.layers.iter().map(|layer| LayerView { layer })
    }

    /// Returns the number of layers, including the output layer.
    #[must_use]
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Returns the number of weights, biases, and multipliers of every
    /// layer.
    #[must_use]
    pub fn parameter_count(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| layer.matrix.rows() * (layer.matrix.cols() + 2))
            .sum()
    }

    /// Returns the length of [`Model::to_bytes`] without encoding the
    /// model.
    #[must_use]
    pub fn encoded_len(&self) -> usize {
        let body: usize = self
            .layers
            .iter()
            .map(|layer| 4 + layer.matrix.rows() * (8 + layer.matrix.cols()))
            .sum();
        MAGIC.len() + 4 + self.metadata.encoded_len() + 2 * (self.layers.len() + 1) + body + 4
    }

    /// Returns the checksum that ends [`Model::to_bytes`], without
    /// allocating.
    ///
    /// Equal models have equal checksums, so the checksum can name the
    /// model in logs. It is not an authenticity check, and different models
    /// can share it.
    #[must_use]
    pub fn checksum(&self) -> u32 {
        let mut hash = FNV_OFFSET;
        self.encode_content(|bytes| hash = fnv1a(hash, bytes));
        hash
    }

    /// Returns the model in the [format](self) that [`Model::from_bytes`]
    /// reads.
    ///
    /// The format has one encoding of each model, so the bytes of a model
    /// parsed from format version [`FORMAT_VERSION`] equal the bytes it was
    /// parsed from. A model parsed from format version 1 is written in the
    /// current format.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.encoded_len());
        self.encode_content(|part| bytes.extend_from_slice(part));
        let sum = checksum(&bytes);
        bytes.extend(sum.to_le_bytes());
        bytes
    }

    /// Passes the encoding of the model up to its checksum to `sink`, in
    /// order.
    fn encode_content(&self, sink: impl FnMut(&[u8])) {
        self.encode_format(FORMAT_VERSION, sink);
    }

    /// Returns the checksum of the model in format version 1, which has
    /// no metadata. Tests that hold the checksum of a trained model use it,
    /// so that their value depends on the layers only.
    #[cfg(test)]
    pub(crate) fn legacy_checksum(&self) -> u32 {
        let mut hash = FNV_OFFSET;
        self.encode_format(LEGACY_FORMAT_VERSION, |bytes| hash = fnv1a(hash, bytes));
        hash
    }

    /// Passes the encoding of the model up to its checksum in
    /// `format_version` to `sink`, in order.
    fn encode_format(&self, format_version: u16, mut sink: impl FnMut(&[u8])) {
        sink(&MAGIC);
        sink(&format_version.to_le_bytes());
        // Validation bounds the layer count by `MAX_LAYERS` and every
        // dimension by `FINGERPRINT_LEN`, so both conversions are exact.
        sink(&[self.layers.len() as u8, 0]);
        if format_version != LEGACY_FORMAT_VERSION {
            self.metadata.encode(&mut sink);
        }
        sink(&(FINGERPRINT_LEN as u16).to_le_bytes());
        for layer in &self.layers {
            sink(&(layer.matrix.rows() as u16).to_le_bytes());
        }
        for layer in &self.layers {
            sink(&[layer.shift, 0, 0, 0]);
            for value in layer.bias.iter().chain(&layer.multiplier) {
                sink(&value.to_le_bytes());
            }
            let mut buffer = [0u8; 1024];
            for chunk in layer.matrix.weights().chunks(buffer.len()) {
                for (byte, &weight) in buffer.iter_mut().zip(chunk) {
                    *byte = weight.cast_unsigned();
                }
                sink(&buffer[..chunk.len()]);
            }
        }
    }

    /// Returns the number of outputs of each layer, from the first hidden
    /// layer to the output layer.
    pub fn layer_sizes(&self) -> impl ExactSizeIterator<Item = usize> + '_ {
        self.layers.iter().map(|layer| layer.matrix.rows())
    }

    /// Returns the scores of one fingerprint, computed with `kernel`.
    ///
    /// Scores do not depend on the kernel. The call never fails, allocates
    /// nothing, and takes time linear in [`Model::parameter_count`].
    #[must_use]
    pub fn score(&self, kernel: Kernel, fingerprint: &Fingerprint) -> Scores {
        self.infer(kernel.verified(), fingerprint)
    }

    /// Returns the logits of one fingerprint, computed with `kernel`, which
    /// [`Logits::scores`] turns into the scores of [`Model::score`].
    #[must_use]
    pub fn logits(&self, kernel: Kernel, fingerprint: &Fingerprint) -> Logits {
        self.forward(kernel.verified(), fingerprint, |_, _, _| {})
    }

    /// Returns the values of every layer for one fingerprint, computed with
    /// `kernel`.
    #[must_use]
    pub fn trace(&self, kernel: Kernel, fingerprint: &Fingerprint) -> Trace {
        let mut trace = Trace::EMPTY;
        trace.layers = self.layers.len();
        trace.logits = self.forward(
            kernel.verified(),
            fingerprint,
            |index, accumulators, activations| {
                let layer = &self.layers[index];
                trace.sizes[index] = accumulators.len();
                for ((stored, &acc), &bias) in trace.accumulators[index]
                    .iter_mut()
                    .zip(accumulators)
                    .zip(layer.bias.iter())
                {
                    // Validation bounds the sum by `i32::MAX` in magnitude.
                    *stored = acc.wrapping_add(bias);
                }
                if let Some(stored) = trace.activations.get_mut(index) {
                    stored[..activations.len()].copy_from_slice(activations);
                }
            },
        );
        trace
    }

    /// Runs the network on one fingerprint.
    pub(crate) fn infer(&self, backend: Verified, fingerprint: &Fingerprint) -> Scores {
        self.forward(backend, fingerprint, |_, _, _| {}).scores()
    }

    /// Runs the network on one fingerprint and returns the logits.
    ///
    /// After each layer it passes the index of the layer, the dot products
    /// of its outputs, and the activations of a hidden layer, which are
    /// empty for the output layer, to `observe`.
    #[inline(always)]
    fn forward(
        &self,
        backend: Verified,
        fingerprint: &Fingerprint,
        mut observe: impl FnMut(usize, &[i32], &[u8]),
    ) -> Logits {
        let mut acc = [0i32; MAX_HIDDEN];
        let mut front = [0u8; MAX_HIDDEN];
        let mut back = [0u8; MAX_HIDDEN];
        let (mut source, mut target) = (&mut front, &mut back);
        let mut input: &[u8] = fingerprint.as_array();

        let Some((output, hidden)) = self.layers.split_last() else {
            return Logits::NONE;
        };
        for (index, layer) in hidden.iter().enumerate() {
            let rows = layer.matrix.rows();
            let acc = &mut acc[..rows];
            kernel::matvec(backend, &layer.matrix, input, acc);
            for (row, (activation, &acc)) in target[..rows].iter_mut().zip(acc.iter()).enumerate() {
                // The clamp is also the rectifier.
                *activation = layer.requantize(row, acc).clamp(0, 255) as u8;
            }
            observe(index, acc, &target[..rows]);
            core::mem::swap(&mut source, &mut target);
            input = &source[..rows];
        }

        let mut logits = [0i8; Category::COUNT];
        let acc = &mut acc[..Category::COUNT];
        kernel::matvec(backend, &output.matrix, input, acc);
        for (row, (logit, &acc)) in logits.iter_mut().zip(acc.iter()).enumerate() {
            *logit = output.requantize(row, acc).clamp(-128, 127) as i8;
        }
        observe(hidden.len(), acc, &[]);
        Logits(logits)
    }
}

/// The fields of a model before its layers, with the length and the
/// checksum validated.
struct Header<'a> {
    format_version: u16,
    layer_count: usize,
    dimensions: [usize; MAX_LAYERS + 1],
    /// The text of the layout version, empty in format version 1.
    layout_version: &'a [u8],
    /// The text of the library requirement, empty in format version 1.
    library_requirement: &'a [u8],
    /// Positioned at the first layer.
    reader: Reader<'a>,
}

impl<'a> Header<'a> {
    /// Reads the header and checks the length and the checksum, which
    /// bounds every later allocation by the length of `bytes`.
    fn read(bytes: &'a [u8]) -> Result<Self, ModelError> {
        let mut reader = Reader::new(bytes);
        if reader.take(MAGIC.len())? != MAGIC {
            return Err(ModelError::BadMagic);
        }
        let format_version = reader.u16()?;
        if format_version != FORMAT_VERSION && format_version != LEGACY_FORMAT_VERSION {
            return Err(ModelError::UnsupportedVersion);
        }
        let layer_count = usize::from(reader.u8()?);
        if reader.u8()? != 0 {
            return Err(ModelError::InvalidParameter);
        }
        if !(1..=MAX_LAYERS).contains(&layer_count) {
            return Err(ModelError::InvalidDimensions);
        }
        let (mut layout_version, mut library_requirement): (&[u8], &[u8]) = (&[], &[]);
        if format_version != LEGACY_FORMAT_VERSION {
            let len = usize::from(reader.u8()?);
            layout_version = reader.take(len)?;
            let len = usize::from(reader.u8()?);
            library_requirement = reader.take(len)?;
        }

        let mut dimensions = [0usize; MAX_LAYERS + 1];
        for dimension in &mut dimensions[..=layer_count] {
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

        let body: usize = dimensions[..=layer_count]
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
        Ok(Self {
            format_version,
            layer_count,
            dimensions,
            layout_version,
            library_requirement,
            reader,
        })
    }

    /// Parses the metadata, without checking it.
    fn metadata(&self) -> Result<Metadata, ModelError> {
        if self.format_version == LEGACY_FORMAT_VERSION {
            return Ok(Metadata::legacy());
        }
        let text = |bytes| core::str::from_utf8(bytes).map_err(|_| ModelError::InvalidMetadata);
        let layout_version =
            Version::parse(text(self.layout_version)?).map_err(|_| ModelError::InvalidMetadata)?;
        let requirement = text(self.library_requirement)?;
        let library_requirement =
            VersionReq::parse(requirement).map_err(|_| ModelError::InvalidMetadata)?;
        let mut buffer = [0u8; MAX_REQUIREMENT_LEN];
        if library_requirement.write_canonical(&mut buffer) != requirement.as_bytes() {
            return Err(ModelError::InvalidMetadata);
        }
        Ok(Metadata::new(layout_version, library_requirement))
    }
}

/// The offset basis of FNV-1a 32.
const FNV_OFFSET: u32 = 0x811C_9DC5;

/// Continues the FNV-1a 32 hash `hash` over `bytes`.
fn fnv1a(hash: u32, bytes: &[u8]) -> u32 {
    bytes.iter().fold(hash, |hash, &byte| {
        (hash ^ u32::from(byte)).wrapping_mul(0x0100_0193)
    })
}

/// Returns the FNV-1a 32 hash of `bytes`.
fn checksum(bytes: &[u8]) -> u32 {
    fnv1a(FNV_OFFSET, bytes)
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
    use crate::Backend;
    use crate::testutil::Rng;

    /// The parameters of one layer in the order of the format.
    #[derive(Clone)]
    pub(crate) struct RawLayer {
        pub(crate) shift: u8,
        pub(crate) bias: Vec<i32>,
        pub(crate) multiplier: Vec<i32>,
        pub(crate) weights: Vec<i8>,
    }

    /// Encodes a model with [`Metadata::current`] without validating it.
    pub(crate) fn encode(dimensions: &[usize], layers: &[RawLayer]) -> Vec<u8> {
        let current = Metadata::current();
        encode_with(
            dimensions,
            layers,
            Some((
                current.layout_version().as_str(),
                &current.library_requirement().to_string(),
            )),
        )
    }

    /// Encodes a model with the texts of the layout version and the
    /// library requirement, or in format version 1 for `None`, without
    /// validating it.
    pub(crate) fn encode_with(
        dimensions: &[usize],
        layers: &[RawLayer],
        metadata: Option<(&str, &str)>,
    ) -> Vec<u8> {
        let mut bytes = MAGIC.to_vec();
        let format = if metadata.is_some() {
            FORMAT_VERSION
        } else {
            LEGACY_FORMAT_VERSION
        };
        bytes.extend(format.to_le_bytes());
        bytes.push(u8::try_from(layers.len()).unwrap());
        bytes.push(0);
        if let Some((layout, requirement)) = metadata {
            for text in [layout, requirement] {
                bytes.push(u8::try_from(text.len()).unwrap());
                bytes.extend(text.as_bytes());
            }
        }
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
    fn score_of_logit_reads_the_logistic_table() {
        for value in i8::MIN..=i8::MAX {
            assert_eq!(
                score_of_logit(value),
                LOGISTIC[(i16::from(value) + 128) as usize]
            );
        }
        assert_eq!(
            (score_of_logit(-128), score_of_logit(0), score_of_logit(127)),
            (0, 128, 255)
        );
        let logits = Logits::from_array([-128, -1, 0, 1, 16, 127, -16, 5]);
        assert_eq!(logits.get(Category::SqlInjection), 0);
        assert_eq!(logits[Category::Obfuscation], -1);
        assert_eq!(logits.value(Category::MarkupInjection), 1.0 / 16.0);
        assert_eq!(logits.value(Category::PathTraversal), 1.0);
        assert_eq!(logits.iter().count(), Category::COUNT);
        let scores = Scores::from(logits);
        for (category, logit) in logits.iter() {
            assert_eq!(scores.get(category), score_of_logit(logit));
        }
        let other = Logits::from_array([0; Category::COUNT]);
        assert_eq!(
            logits.max_with(other).as_array(),
            &[0, 0, 0, 1, 16, 127, 0, 5]
        );
    }

    /// Recomputes every layer from the parameters with plain integers.
    fn reference_trace(
        model: &Model,
        fingerprint: &Fingerprint,
    ) -> (Vec<Vec<i64>>, Vec<Vec<u8>>, Vec<i8>) {
        let mut input: Vec<u8> = fingerprint.as_array().to_vec();
        let (mut sums, mut activations, mut logits) = (Vec::new(), Vec::new(), Vec::new());
        let count = model.layer_count();
        for (index, layer) in model.layer_views().enumerate() {
            let mut layer_sums = Vec::new();
            let mut outputs = Vec::new();
            for row in 0..layer.outputs() {
                let dot: i64 = layer
                    .row(row)
                    .unwrap()
                    .iter()
                    .zip(&input)
                    .map(|(&weight, &x)| i64::from(weight) * i64::from(x))
                    .sum();
                let sum = dot + i64::from(layer.biases()[row]);
                layer_sums.push(sum);
                let shift = layer.shift();
                let value =
                    (sum * i64::from(layer.multipliers()[row]) + (1 << (shift - 1))) >> shift;
                outputs.push(value);
            }
            sums.push(layer_sums);
            if index + 1 == count {
                logits = outputs
                    .iter()
                    .map(|&value| value.clamp(-128, 127) as i8)
                    .collect();
            } else {
                input = outputs
                    .iter()
                    .map(|&value| value.clamp(0, 255) as u8)
                    .collect();
                activations.push(input.clone());
            }
        }
        (sums, activations, logits)
    }

    #[test]
    fn traces_record_every_layer() {
        let mut rng = Rng::new(41);
        for hidden in [&[][..], &[16], &[48, 24], &[7, 5, 3]] {
            let model = random_model(&mut rng, hidden);
            for _ in 0..4 {
                let fingerprint = random_fingerprint(&mut rng);
                let (sums, activations, logits) = reference_trace(&model, &fingerprint);
                for backend in Backend::supported() {
                    let kernel = Kernel::new(backend).unwrap();
                    let trace = model.trace(kernel, &fingerprint);
                    assert_eq!(trace.layer_count(), hidden.len() + 1);
                    for (layer, expected) in sums.iter().enumerate() {
                        let actual = trace.accumulators(layer).unwrap();
                        assert!(
                            actual
                                .iter()
                                .map(|&sum| i64::from(sum))
                                .eq(expected.iter().copied())
                        );
                    }
                    for (layer, expected) in activations.iter().enumerate() {
                        assert_eq!(trace.activations(layer), Some(&expected[..]));
                    }
                    assert_eq!(trace.activations(hidden.len()), None);
                    assert_eq!(trace.accumulators(hidden.len() + 1), None);
                    assert_eq!(trace.logits().as_array()[..], logits[..]);
                    assert_eq!(model.logits(kernel, &fingerprint), trace.logits());
                    assert_eq!(model.score(kernel, &fingerprint), trace.scores());
                    assert_eq!(trace.scores(), model.infer(Verified::SCALAR, &fingerprint));
                }
            }
        }
        let model = Model::embedded();
        let trace = model.trace(Kernel::SCALAR, &Fingerprint::ZERO);
        for index in [3, 4, usize::MAX] {
            assert_eq!(trace.accumulators(index), None);
            assert!(model.layer(index).is_none());
        }
        for index in [2, 3, usize::MAX] {
            assert_eq!(trace.activations(index), None);
        }
        assert_eq!(model.layer(0).unwrap().row(usize::MAX), None);
        assert_eq!(Trace::EMPTY.activations(0), None);
        assert_eq!(Logits::NONE.scores(), Scores::ZERO);

        assert_eq!(
            format!("{trace:?}"),
            format!(
                "Trace {{ layer_sizes: [64, 64, 8], logits: {:?}, .. }}",
                trace.logits()
            )
        );
    }

    #[test]
    fn debug_output_shows_the_shape_and_not_the_parameters() {
        let model = Model::embedded();
        let debug = format!("{model:?}");
        assert_eq!(
            debug,
            format!(
                "Model {{ layer_sizes: [64, 64, 8], parameters: {}, layout_version: {}, \
                 library_requirement: {}, .. }}",
                model.parameter_count(),
                model.layout_version(),
                model.library_requirement()
            )
        );
        assert_eq!(
            format!("{:?}", model.layer(1).unwrap()),
            format!(
                "LayerView {{ inputs: 64, outputs: 64, shift: {}, .. }}",
                model.layer(1).unwrap().shift()
            )
        );
    }

    #[test]
    fn model_information_matches_the_encoding() {
        let mut rng = Rng::new(43);
        let mut models: Vec<Model> = [&[][..], &[16], &[48, 24]]
            .iter()
            .map(|hidden| random_model(&mut rng, hidden))
            .collect();
        models.extend(EmbeddedModel::ALL.map(EmbeddedModel::load));
        for model in models {
            let bytes = model.to_bytes();
            assert_eq!(model.encoded_len(), bytes.len());
            assert_eq!(model.checksum().to_le_bytes(), bytes[bytes.len() - 4..]);
            assert_eq!(model.layer_count(), model.layer_sizes().len());
            let mut parameters = 0;
            for (index, (view, copy)) in model.layer_views().zip(model.layers()).enumerate() {
                assert_eq!(view.to_parameters(), copy);
                assert_eq!(model.layer(index).unwrap().weights(), &copy.weights[..]);
                assert_eq!(view.outputs(), copy.biases.len());
                assert_eq!(view.inputs() * view.outputs(), copy.weights.len());
                assert_eq!(view.multipliers(), &copy.multipliers[..]);
                assert_eq!(view.shift(), copy.shift);
                assert_eq!(view.row(0), Some(&copy.weights[..view.inputs()]));
                assert_eq!(view.row(view.outputs()), None);
                parameters += copy.weights.len() + 2 * copy.biases.len();
            }
            assert!(model.layer(model.layer_count()).is_none());
            assert_eq!(model.parameter_count(), parameters);
        }
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

        for version in [0, 3, 0x0102] {
            let mut bytes = valid.clone();
            bytes[4..6].copy_from_slice(&u16::to_le_bytes(version));
            assert_eq!(
                Model::from_bytes(&bytes),
                Err(ModelError::UnsupportedVersion)
            );
        }

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
        let header = 4 + 2 + 1 + 1 + Metadata::current().encoded_len() + 4;
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
                let keep = (rng.next_u32() as usize % 40).min(bytes.len());
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
            ModelError::InvalidMetadata,
            ModelError::IncompatibleLayout,
            ModelError::UnsupportedLibrary,
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

    fn parameters(layer: &RawLayer) -> LayerParameters {
        LayerParameters {
            shift: layer.shift,
            biases: layer.bias.clone(),
            multipliers: layer.multiplier.clone(),
            weights: layer.weights.clone(),
        }
    }

    #[test]
    fn to_bytes_reproduces_the_parsed_bytes() {
        let mut rng = Rng::new(31);
        for hidden in [&[][..], &[1], &[24], &[MAX_HIDDEN, 3], &[5, 6, 7]] {
            let mut dimensions = vec![FINGERPRINT_LEN];
            dimensions.extend(hidden);
            dimensions.push(Category::COUNT);
            let bytes = encode(&dimensions, &random_layers(&mut rng, &dimensions));
            let model = Model::from_bytes(&bytes).unwrap();
            assert_eq!(model.to_bytes(), bytes, "hidden {hidden:?}");
        }
        for embedded in EmbeddedModel::ALL {
            assert_eq!(embedded.load().to_bytes(), embedded.bytes(), "{embedded}");
        }
    }

    #[test]
    fn from_layers_matches_from_bytes() {
        let mut rng = Rng::new(32);
        for hidden in [&[][..], &[9], &[MAX_HIDDEN, 2], &[4, 3, 2]] {
            let mut dimensions = vec![FINGERPRINT_LEN];
            dimensions.extend(hidden);
            dimensions.push(Category::COUNT);
            let layers = random_layers(&mut rng, &dimensions);
            let parsed = Model::from_bytes(&encode(&dimensions, &layers)).unwrap();
            let built = Model::from_layers(layers.iter().map(parameters).collect()).unwrap();
            assert_eq!(built, parsed, "hidden {hidden:?}");
            assert_eq!(Model::from_layers(parsed.layers().collect()), Ok(parsed));
        }
        let embedded = Model::embedded();
        assert_eq!(embedded.layers().len(), 3);
        assert_eq!(
            Model::from_layers(embedded.layers().collect())
                .and_then(|model| model.with_metadata(embedded.metadata().clone())),
            Ok(embedded)
        );
    }

    #[test]
    fn from_layers_rejects_what_from_bytes_rejects() {
        let mut rng = Rng::new(33);
        let dimensions = [FINGERPRINT_LEN, 6, Category::COUNT];
        let valid: Vec<LayerParameters> = random_layers(&mut rng, &dimensions)
            .iter()
            .map(parameters)
            .collect();
        assert!(Model::from_layers(valid.clone()).is_ok());
        type Edit<'a> = &'a dyn Fn(&mut Vec<LayerParameters>);
        let edited = |edit: Edit<'_>| {
            let mut layers = valid.clone();
            edit(&mut layers);
            Model::from_layers(layers)
        };

        let dimension_edits: [Edit<'_>; 10] = [
            &|layers| layers.clear(),
            &|layers| {
                // A dimension error comes first, even after a parameter
                // error in an earlier layer.
                layers[0].shift = 0;
                layers[1].multipliers.push(0);
            },
            &|layers| {
                let hidden = LayerParameters {
                    shift: 1,
                    biases: vec![0; 6],
                    multipliers: vec![0; 6],
                    weights: vec![0; 36],
                };
                // Five layers: one more than the format allows.
                layers.splice(1..1, vec![hidden; 3]);
            },
            &|layers| {
                layers.pop();
            },
            &|layers| {
                layers.remove(0);
            },
            &|layers| layers[0].weights.push(0),
            &|layers| {
                layers[0].weights.pop();
            },
            &|layers| layers[1].multipliers.push(0),
            &|layers| {
                // A hidden layer without outputs.
                layers[0] = LayerParameters {
                    shift: 1,
                    ..LayerParameters::default()
                };
                layers[1].weights.clear();
            },
            &|layers| {
                // A hidden layer above the largest dimension.
                let rows = MAX_HIDDEN + 1;
                layers[0] = LayerParameters {
                    shift: 1,
                    biases: vec![0; rows],
                    multipliers: vec![0; rows],
                    weights: vec![0; rows * FINGERPRINT_LEN],
                };
                layers[1].weights = vec![0; Category::COUNT * rows];
            },
        ];
        for (index, edit) in dimension_edits.into_iter().enumerate() {
            assert_eq!(
                edited(edit),
                Err(ModelError::InvalidDimensions),
                "edit {index}"
            );
        }

        let parameter_edits: [Edit<'_>; 4] = [
            &|layers| layers[0].shift = 0,
            &|layers| layers[1].shift = MAX_SHIFT + 1,
            &|layers| layers[1].multipliers[0] = -1,
            &|layers| layers[0].weights[7] = i8::MIN,
        ];
        for (index, edit) in parameter_edits.into_iter().enumerate() {
            assert_eq!(
                edited(edit),
                Err(ModelError::InvalidParameter),
                "edit {index}"
            );
        }

        assert_eq!(
            edited(&|layers| {
                layers[0].weights.fill(127);
                layers[0].biases[0] = i32::MAX;
            }),
            Err(ModelError::AccumulatorOverflow)
        );
    }

    /// The smallest valid model with the given texts of the layout version
    /// and the library requirement.
    fn linear_with(layout: &str, requirement: &str) -> Vec<u8> {
        let layer = RawLayer {
            shift: 1,
            bias: vec![0; Category::COUNT],
            multiplier: vec![0; Category::COUNT],
            weights: vec![0; Category::COUNT * FINGERPRINT_LEN],
        };
        encode_with(
            &[FINGERPRINT_LEN, Category::COUNT],
            &[layer],
            Some((layout, requirement)),
        )
    }

    #[test]
    fn models_record_their_metadata() {
        let model = Model::from_bytes(&linear(|_| {})).unwrap();
        assert_eq!(model.metadata(), &Metadata::current());
        assert_eq!(model.layout_version(), &LAYOUT_VERSION);
        assert_eq!(
            model.library_requirement(),
            &VersionReq::caret(&crate::VERSION)
        );
        assert_eq!(Metadata::default(), Metadata::current());
        let bytes = model.to_bytes();
        assert_eq!(Metadata::read(&bytes), Ok(Metadata::current()));
        // The header holds both texts after their lengths.
        let layout = LAYOUT_VERSION.to_string();
        let requirement = VersionReq::caret(&crate::VERSION).to_string();
        let mut expected = vec![u8::try_from(layout.len()).unwrap()];
        expected.extend(layout.as_bytes());
        expected.push(u8::try_from(requirement.len()).unwrap());
        expected.extend(requirement.as_bytes());
        assert_eq!(&bytes[8..8 + expected.len()], expected);
        for embedded in EmbeddedModel::ALL {
            let metadata = Metadata::read(embedded.bytes()).unwrap();
            assert_eq!(metadata.layout_version(), &LAYOUT_VERSION, "{embedded}");
            assert_eq!(metadata.check(), Ok(()), "{embedded}");
            assert_eq!(embedded.load().metadata(), &metadata, "{embedded}");
        }
    }

    #[test]
    fn reads_format_version_1_as_layout_1_0_0() {
        let mut rng = Rng::new(51);
        let dimensions = [FINGERPRINT_LEN, 5, Category::COUNT];
        let layers = random_layers(&mut rng, &dimensions);
        let legacy = encode_with(&dimensions, &layers, None);
        assert_eq!(&legacy[4..6], &[1, 0]);
        let model = Model::from_bytes(&legacy).unwrap();
        assert_eq!(model.layout_version(), &Version::new(1, 0, 0));
        assert_eq!(model.library_requirement().to_string(), "^0.1.0-dev");
        assert_eq!(Metadata::read(&legacy).as_ref(), Ok(model.metadata()));
        assert_eq!(
            model.legacy_checksum().to_le_bytes(),
            legacy[legacy.len() - 4..]
        );
        // The model is written in the current format, with the same layers.
        let current = model.to_bytes();
        assert_eq!(&current[4..6], &FORMAT_VERSION.to_le_bytes());
        assert_eq!(Model::from_bytes(&current), Ok(model.clone()));
        let metadata_len = model.metadata().encoded_len();
        assert_eq!(
            current[8 + metadata_len..current.len() - 4],
            legacy[8..legacy.len() - 4]
        );
        // Every embedded model gives the same model from format 1.
        for embedded in EmbeddedModel::ALL {
            let model = embedded.load();
            let metadata_len = model.metadata().encoded_len();
            let mut bytes = embedded.bytes().to_vec();
            bytes.drain(8..8 + metadata_len);
            bytes[4..6].copy_from_slice(&LEGACY_FORMAT_VERSION.to_le_bytes());
            reseal(&mut bytes);
            assert_eq!(Model::from_bytes(&bytes).as_ref(), Ok(&model), "{embedded}");
        }
        // Format 1 has no metadata to be invalid, but its other errors stay.
        let mut truncated = legacy.clone();
        truncated.truncate(legacy.len() - 1);
        assert_eq!(Model::from_bytes(&truncated), Err(ModelError::Truncated));
    }

    #[test]
    fn rejects_invalid_metadata() {
        let requirement = VersionReq::caret(&crate::VERSION).to_string();
        assert!(Model::from_bytes(&linear_with("1.0.0", &requirement)).is_ok());
        for layout in ["", "1.0", "01.0.0", "1.0.0 ", "v1.0.0", "1.0.0-é"] {
            assert_eq!(
                Model::from_bytes(&linear_with(layout, &requirement)),
                Err(ModelError::InvalidMetadata),
                "{layout:?}"
            );
        }
        let long = format!("1.0.0+{}", "a".repeat(crate::version::MAX_VERSION_LEN));
        assert_eq!(
            Model::from_bytes(&linear_with(&long, &requirement)),
            Err(ModelError::InvalidMetadata)
        );
        // Valid requirements that are not in canonical form, and invalid ones.
        for requirement in [
            "",
            "0.1.0-dev",
            "^ 0.1.0-dev",
            " ^0.1.0-dev",
            ">=0.1.0-dev,<1",
            "^0.1.0-dev,",
            "<1 || >2",
        ] {
            assert_eq!(
                Model::from_bytes(&linear_with("1.0.0", requirement)),
                Err(ModelError::InvalidMetadata),
                "{requirement:?}"
            );
        }
        let mut bytes = linear_with("1.0.0", &requirement);
        bytes[9] = 0xFF;
        reseal(&mut bytes);
        assert_eq!(Model::from_bytes(&bytes), Err(ModelError::InvalidMetadata));
        assert_eq!(Metadata::read(&bytes), Err(ModelError::InvalidMetadata));
        // A length that runs past the end.
        let mut bytes = linear_with("1.0.0", &requirement);
        bytes[8] = 255;
        assert_eq!(
            Model::from_bytes(&bytes),
            Err(ModelError::InvalidDimensions)
        );
        assert_eq!(Model::from_bytes(&bytes[..12]), Err(ModelError::Truncated));
    }

    #[test]
    fn refuses_incompatible_layouts() {
        let requirement = VersionReq::caret(&crate::VERSION).to_string();
        for layout in ["1.0.0", "1.0.0+build.3", "1.0.0-rc.1", "1.0.0-alpha+x"] {
            let bytes = linear_with(layout, &requirement);
            let model =
                Model::from_bytes(&bytes).unwrap_or_else(|error| panic!("{layout}: {error}"));
            assert_eq!(model.layout_version().as_str(), layout);
            assert_eq!(model.to_bytes(), bytes, "{layout}");
        }
        for layout in [
            "0.9.0",
            "0.1.0",
            "1.0.1",
            "1.1.0",
            "2.0.0",
            "2.0.0-rc.1",
            "1.0.1-rc.1",
        ] {
            let bytes = linear_with(layout, &requirement);
            assert_eq!(
                Model::from_bytes(&bytes),
                Err(ModelError::IncompatibleLayout),
                "{layout}"
            );
            let metadata = Metadata::read(&bytes).unwrap();
            assert!(!metadata.is_layout_compatible(), "{layout}");
            assert_eq!(metadata.layout_version().as_str(), layout);
        }
        // The layout is checked before the library.
        assert_eq!(
            Model::from_bytes(&linear_with("2.0.0", ">=99")),
            Err(ModelError::IncompatibleLayout)
        );
    }

    #[test]
    fn refuses_unsupported_libraries() {
        let version = crate::VERSION;
        let next_major = if version.major() > 0 {
            Version::new(version.major() + 1, 0, 0)
        } else {
            Version::new(0, version.minor() + 1, 0)
        };
        let supported = [
            format!("^{version}"),
            format!("={version}"),
            format!(">={version}, <{next_major}"),
            format!("<={version}"),
        ];
        for requirement in &supported {
            let bytes = linear_with("1.0.0", requirement);
            let model =
                Model::from_bytes(&bytes).unwrap_or_else(|error| panic!("{requirement}: {error}"));
            assert_eq!(&model.library_requirement().to_string(), requirement);
            assert_eq!(model.to_bytes(), bytes);
        }
        let unsupported = [
            format!(">{version}"),
            format!("^{next_major}"),
            format!("<{version}"),
            "=0.0.0".to_owned(),
            ">=99.0.0".to_owned(),
        ];
        for requirement in &unsupported {
            let bytes = linear_with("1.0.0", requirement);
            assert_eq!(
                Model::from_bytes(&bytes),
                Err(ModelError::UnsupportedLibrary),
                "{requirement}"
            );
            let metadata = Metadata::read(&bytes).unwrap();
            assert!(!metadata.supports_this_library());
            assert!(metadata.is_layout_compatible());
            assert_eq!(metadata.check(), Err(ModelError::UnsupportedLibrary));
        }
    }

    #[test]
    fn with_metadata_keeps_models_loadable() {
        let model = Model::embedded();
        let wider: VersionReq = format!(">={}, <99", crate::VERSION).parse().unwrap();
        let widened = model
            .clone()
            .with_library_requirement(wider.clone())
            .unwrap();
        assert_eq!(widened.library_requirement(), &wider);
        assert_eq!(widened.layout_version(), model.layout_version());
        assert_ne!(widened, model);
        assert_eq!(
            widened.layers().collect::<Vec<_>>(),
            model.layers().collect::<Vec<_>>()
        );
        let bytes = widened.to_bytes();
        assert_eq!(widened.encoded_len(), bytes.len());
        assert_eq!(Model::from_bytes(&bytes).as_ref(), Ok(&widened));
        assert_eq!(
            model
                .clone()
                .with_library_requirement(">=99".parse().unwrap()),
            Err(ModelError::UnsupportedLibrary)
        );
        assert_eq!(
            model.clone().with_metadata(Metadata::new(
                Version::new(2, 0, 0),
                VersionReq::caret(&crate::VERSION)
            )),
            Err(ModelError::IncompatibleLayout)
        );
        let stated = Metadata::new("1.0.0+trained".parse().unwrap(), wider);
        let restated = model.with_metadata(stated.clone()).unwrap();
        assert_eq!(restated.metadata(), &stated);
        assert_eq!(Metadata::read(&restated.to_bytes()), Ok(stated));
    }

    /// Golden vectors written by `tools/train/train.py`: for each vector, a
    /// fingerprint followed by the scores that the training tools' integer
    /// simulation computed for it.
    fn golden(model: EmbeddedModel) -> &'static [u8] {
        match model {
            EmbeddedModel::Compact => include_bytes!("../models/compact.golden"),
            EmbeddedModel::Default => include_bytes!("../models/default.golden"),
            EmbeddedModel::Large => include_bytes!("../models/large.golden"),
            EmbeddedModel::Strict => include_bytes!("../models/strict.golden"),
        }
    }

    #[test]
    fn embedded_model_is_valid() {
        let model = Model::embedded();
        assert_eq!(model.layer_sizes().last(), Some(Category::COUNT));
        assert_eq!(Model::from_bytes(DEFAULT_MODEL), Ok(model.clone()));
        assert_eq!(EmbeddedModel::Default.load(), model);
    }

    #[test]
    fn every_embedded_model_is_valid_and_distinct() {
        let mut models: Vec<Model> = Vec::new();
        for embedded in EmbeddedModel::ALL {
            let model = embedded.load();
            assert_eq!(
                model.layer_sizes().last(),
                Some(Category::COUNT),
                "{embedded}"
            );
            assert!(models.iter().all(|other| *other != model), "{embedded}");
            models.push(model);
        }
        let sizes: Vec<usize> = EmbeddedModel::ALL
            .iter()
            .map(|model| model.bytes().len())
            .collect();
        assert!(sizes[0] < sizes[1] && sizes[1] < sizes[2], "{sizes:?}");
    }

    #[test]
    fn embedded_model_names_round_trip() {
        for model in EmbeddedModel::ALL {
            assert_eq!(EmbeddedModel::from_name(model.name()), Some(model));
            assert_eq!(model.to_string(), model.name());
        }
        assert_eq!(EmbeddedModel::from_name("Default"), None);
        assert_eq!(EmbeddedModel::from_name(""), None);
    }

    #[test]
    fn embedded_models_match_golden_vectors_on_every_backend() {
        for embedded in EmbeddedModel::ALL {
            let model = embedded.load();
            let (vectors, rest) =
                golden(embedded).as_chunks::<{ FINGERPRINT_LEN + Category::COUNT }>();
            assert!(rest.is_empty() && !vectors.is_empty(), "{embedded}");
            let backends = crate::Backend::PREFERENCE
                .iter()
                .filter_map(|&backend| Verified::new(backend).ok());
            for backend in backends {
                for (index, vector) in vectors.iter().enumerate() {
                    let (features, expected) = vector.split_at(FINGERPRINT_LEN);
                    let fingerprint = Fingerprint::from_array(features.try_into().unwrap());
                    let scores = model.infer(backend, &fingerprint);
                    assert_eq!(
                        scores.as_array().as_slice(),
                        expected,
                        "model {embedded}, golden vector {index}, backend {}",
                        backend.get()
                    );
                }
            }
        }
    }
}
