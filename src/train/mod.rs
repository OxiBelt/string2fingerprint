//! Training of models on labeled samples.
//!
//! This module is available with the `train` feature. It turns labeled
//! inputs into a [`Model`](crate::model::Model) that
//! [`Scorer::with_model`](crate::Scorer::with_model) runs:
//!
//! 1. Collect samples in a [`Dataset`]. A sample is the bytes of one input
//!    and the [categories](crate::Category) that apply to it. Benign text
//!    has no category.
//! 2. Train with a [`Trainer`], whose [`TrainConfig`] sets the shape of the
//!    network and the optimization.
//! 3. Keep the bytes of [`Model::to_bytes`](crate::model::Model::to_bytes),
//!    and load them with
//!    [`Model::from_bytes`](crate::model::Model::from_bytes).
//!
//! The trainer optimizes a float network with the shape of the quantized
//! one, adapts it to 8-bit weights and activations in its last epochs, and
//! then converts it to the integer arithmetic that the design document
//! specifies. It needs no dependency and no other tool.
//!
//! A model is only as good as its data. A few hundred samples give a model
//! that knows those samples. The embedded models were trained on 600,000
//! generated samples, and `tools/train/` documents that data.
//!
//! Training data is trusted input. A dataset and a trainer allocate memory
//! in proportion to the samples that they are given.

mod dataset;
mod math;
mod network;
mod quantize;
mod trainer;

pub use dataset::{Dataset, DatasetError, Sample, WINDOW_LEN};
pub use trainer::{EpochReport, TrainConfig, TrainError, TrainReport, Trained, Trainer};
