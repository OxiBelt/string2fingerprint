//! The items that most callers use, for a glob import.
//!
//! ```
//! use string2fingerprint::prelude::*;
//!
//! let detector = Detector::with_preset(Preset::Blocking);
//! let scores: Scores = detector.scorer().score(b"name=alice");
//! assert!(!detector.decide(scores).is_flagged());
//! ```

pub use crate::detect::{Detection, Detector, Preset};
pub use crate::model::EmbeddedModel;
pub use crate::{
    Backend, Category, CategorySet, ScoreStream, Scorer, Scores, Threshold, Thresholds,
};
