//! The items that most callers use, for a glob import.
//!
//! ```
//! use string2fingerprint::prelude::*;
//!
//! let detector = Detector::with_preset(Preset::Strict);
//! let scores: Scores = detector.scorer().score(b"name=alice");
//! assert!(!detector.decide(scores).is_flagged());
//! ```

pub use crate::detect::{Detection, DetectionStream, Detector, Preset};
pub use crate::model::EmbeddedModel;
#[cfg(feature = "tokio")]
pub use crate::tokio::{AsyncDetector, AsyncScorer};
pub use crate::{
    Backend, Category, CategorySet, ScoreStream, Scorer, ScorerError, Scores, Threshold, Thresholds,
};
