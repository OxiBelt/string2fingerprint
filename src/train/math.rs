//! Deterministic arithmetic for training.
//!
//! The functions of the platform's mathematics library can round the last
//! bit differently from one library or version to the next. Training calls
//! them millions of times, and one different bit can change a quantized
//! weight. The functions here use only addition, multiplication, division,
//! and square roots, which IEEE 754 rounds exactly, so the same inputs give
//! the same model on every host.

/// A small deterministic generator (SplitMix64).
#[derive(Clone, Debug)]
pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) const fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    /// Returns a generator for stream `stream` that is independent of the
    /// later output of this one.
    pub(crate) fn fork(&self, stream: u64) -> Self {
        let mut mixed = Self(self.0 ^ stream.wrapping_mul(0xD6E8_FEB8_6659_FD93));
        Self(mixed.next_u64())
    }

    /// Returns a value that is uniform in the range from 0 to 1, without 1.
    pub(crate) fn uniform(&mut self) -> f64 {
        // 53 random bits are exact in the significand.
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Returns a value below `bound`, which must not be zero.
    pub(crate) fn below(&mut self, bound: usize) -> usize {
        // The high half of the 128-bit product is below `bound`. Its bias is
        // at most `bound / 2^64`, which does not matter for shuffling.
        ((u128::from(self.next_u64()) * bound as u128) >> 64) as usize
    }

    /// Returns a value with the standard normal distribution.
    pub(crate) fn normal(&mut self) -> f64 {
        // Box-Muller. `1 - uniform` is above zero, so the logarithm is
        // finite.
        let radius = (-2.0 * ln(1.0 - self.uniform())).sqrt();
        radius * cos(core::f64::consts::TAU * self.uniform())
    }

    /// Shuffles `items` (Fisher-Yates).
    pub(crate) fn shuffle<T>(&mut self, items: &mut [T]) {
        for last in (1..items.len()).rev() {
            items.swap(last, self.below(last + 1));
        }
    }
}

/// The high bits of `ln(2)`. Its low 21 bits are zero, so that its product
/// with an integer below 2^21 in magnitude is exact.
const LN2_HIGH: f64 = 6.931_471_803_691_238e-1;

/// `ln(2) - LN2_HIGH`.
const LN2_LOW: f64 = 1.908_214_929_270_587_7e-10;

/// Returns `value * 2^exponent` for an exponent within the normal range.
fn scale_by_power_of_two(value: f64, exponent: i32) -> f64 {
    debug_assert!((-1022..=1023).contains(&exponent));
    value * f64::from_bits(((exponent + 1023) as u64) << 52)
}

/// Returns `e^x`, within a few units in the last place.
pub(crate) fn exp(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x > 709.0 {
        return f64::INFINITY;
    }
    if x < -745.0 {
        return 0.0;
    }
    // `x = k * ln(2) + r` with `|r| <= ln(2) / 2`.
    let k = (x * core::f64::consts::LOG2_E).round();
    let r = (x - k * LN2_HIGH) - k * LN2_LOW;
    // The Taylor series to `r^13`. The first omitted term is below `1e-17`.
    let mut sum = 1.0;
    let mut term = 13.0;
    while term >= 1.0 {
        sum = 1.0 + r * sum / term;
        term -= 1.0;
    }
    // Two steps keep each factor in the normal range, also when the result
    // is subnormal.
    let k = k as i32;
    scale_by_power_of_two(scale_by_power_of_two(sum, k / 2), k - k / 2)
}

/// Returns the natural logarithm of `x`, within a few units in the last
/// place. `x` must be positive and finite.
pub(crate) fn ln(x: f64) -> f64 {
    if x.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    if x == f64::INFINITY {
        return x;
    }
    // Make a subnormal value normal: `2^54` is exact.
    let (x, bias) = if x < f64::MIN_POSITIVE {
        (x * (1u64 << 54) as f64, -54)
    } else {
        (x, 0)
    };
    let bits = x.to_bits();
    let mut exponent = ((bits >> 52) & 0x7FF) as i32 - 1023 + bias;
    // The significand, in the range from 1 to 2.
    let mut significand = f64::from_bits((bits & ((1 << 52) - 1)) | (1023 << 52));
    if significand > core::f64::consts::SQRT_2 {
        significand *= 0.5;
        exponent += 1;
    }
    // `ln(m) = 2 * atanh(s)` with `s = (m - 1) / (m + 1)`, and `|s| < 0.172`.
    let s = (significand - 1.0) / (significand + 1.0);
    let square = s * s;
    // The series `sum(square^n / (2n + 1))` to `n = 13`. The first omitted
    // term is below `1e-22`.
    let mut sum = 0.0;
    let mut odd = 27.0;
    while odd >= 1.0 {
        sum = sum * square + 1.0 / odd;
        odd -= 2.0;
    }
    let exponent = f64::from(exponent);
    exponent * LN2_HIGH + (2.0 * s * sum + exponent * LN2_LOW)
}

/// Returns the cosine of `x` radians, with an absolute error below `1e-15`
/// for arguments of moderate size.
pub(crate) fn cos(x: f64) -> f64 {
    use core::f64::consts::{FRAC_PI_2, PI, TAU};
    if !x.is_finite() {
        return f64::NAN;
    }
    // The remainder is exact. Fold the period onto the first quarter.
    let x = x.abs() % TAU;
    let x = if x > PI { TAU - x } else { x };
    let (x, sign) = if x > FRAC_PI_2 {
        (PI - x, -1.0)
    } else {
        (x, 1.0)
    };
    // The Taylor series to `x^24`. The first omitted term is below `1e-21`.
    let square = x * x;
    let mut sum = 1.0;
    let mut even = 24.0;
    while even >= 2.0 {
        sum = 1.0 - square * sum / (even * (even - 1.0));
        even -= 2.0;
    }
    sign * sum
}

/// Returns `1 / (1 + e^-x)`.
pub(crate) fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + exp(-x))
}

/// Returns `ln(1 + e^x)`.
pub(crate) fn softplus(x: f64) -> f64 {
    x.max(0.0) + ln(1.0 + exp(-x.abs()))
}

/// Rounds a value of magnitude below `2^22` to the nearest integer, with
/// ties to the even one.
///
/// Adding `1.5 * 2^23` moves the value to where consecutive `f32` values
/// are one apart, so the addition itself rounds.
#[inline]
pub(crate) fn round_ties_even(x: f32) -> f32 {
    const MAGIC: f32 = 12_582_912.0;
    (x + MAGIC) - MAGIC
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(actual: f64, expected: f64, tolerance: f64) -> bool {
        (actual - expected).abs() <= tolerance * expected.abs().max(1.0)
    }

    #[test]
    fn exp_matches_the_platform_function() {
        let mut rng = Rng::new(1);
        for _ in 0..20_000 {
            let x = (rng.uniform() - 0.5) * 1400.0;
            let (actual, expected) = (exp(x), x.exp());
            assert!(
                (actual - expected).abs() <= 4e-15 * expected,
                "exp({x}) = {actual}, expected {expected}"
            );
        }
        for x in [
            0.0, 1.0, -1.0, 0.5, 1e-300, 709.0, -708.0, -745.0, 36.0, -36.0,
        ] {
            assert!(close(exp(x), x.exp(), 4e-15), "exp({x})");
        }
        assert_eq!(exp(0.0), 1.0);
        assert_eq!(exp(710.0), f64::INFINITY);
        assert_eq!(exp(-746.0), 0.0);
        assert!(exp(f64::NAN).is_nan());
        assert!(exp(-744.0) > 0.0);
    }

    #[test]
    fn ln_matches_the_platform_function() {
        let mut rng = Rng::new(2);
        for _ in 0..20_000 {
            let x = exp((rng.uniform() - 0.5) * 1400.0);
            assert!(close(ln(x), x.ln(), 4e-15), "ln({x}) = {}", ln(x));
        }
        for x in [
            1.0,
            2.0,
            0.5,
            1.5,
            f64::MIN_POSITIVE,
            5e-324,
            f64::MAX,
            1.0 + 1e-15,
        ] {
            assert!(close(ln(x), x.ln(), 4e-15), "ln({x}) = {}", ln(x));
        }
        assert_eq!(ln(1.0), 0.0);
        assert_eq!(ln(0.0), f64::NEG_INFINITY);
        assert_eq!(ln(f64::INFINITY), f64::INFINITY);
        assert!(ln(-1.0).is_nan());
        assert!(ln(f64::NAN).is_nan());
    }

    #[test]
    fn cos_matches_the_platform_function() {
        let mut rng = Rng::new(3);
        for _ in 0..20_000 {
            let x = (rng.uniform() - 0.5) * 100.0;
            assert!((cos(x) - x.cos()).abs() <= 2e-14, "cos({x}) = {}", cos(x));
        }
        assert_eq!(cos(0.0), 1.0);
        assert!((cos(core::f64::consts::PI) + 1.0).abs() <= 1e-15);
        assert!(cos(core::f64::consts::FRAC_PI_2).abs() <= 1e-15);
        assert!(cos(f64::INFINITY).is_nan());
        assert!(cos(f64::NAN).is_nan());
    }

    #[test]
    fn sigmoid_and_softplus_are_consistent() {
        assert_eq!(sigmoid(0.0), 0.5);
        assert_eq!(sigmoid(800.0), 1.0);
        assert_eq!(sigmoid(-800.0), 0.0);
        let mut rng = Rng::new(4);
        for _ in 0..2_000 {
            let x = (rng.uniform() - 0.5) * 80.0;
            assert!(close(sigmoid(x) + sigmoid(-x), 1.0, 1e-15));
            // The derivative of softplus is the sigmoid.
            let step = 1e-6;
            let slope = (softplus(x + step) - softplus(x - step)) / (2.0 * step);
            assert!((slope - sigmoid(x)).abs() <= 1e-8, "softplus slope at {x}");
            assert!(close(softplus(x), (1.0 + x.exp()).ln(), 1e-13));
        }
        assert_eq!(softplus(800.0), 800.0);
        assert_eq!(softplus(-800.0), 0.0);
    }

    #[test]
    fn round_ties_even_matches_the_standard_method() {
        let mut rng = Rng::new(5);
        for _ in 0..20_000 {
            let x = ((rng.uniform() - 0.5) * 600.0) as f32;
            assert_eq!(round_ties_even(x), x.round_ties_even(), "{x}");
        }
        for half in -300..300 {
            let x = half as f32 + 0.5;
            assert_eq!(round_ties_even(x), x.round_ties_even(), "{x}");
        }
        assert_eq!(round_ties_even(0.5), 0.0);
        assert_eq!(round_ties_even(1.5), 2.0);
        assert_eq!(round_ties_even(-2.5), -2.0);
    }

    #[test]
    fn generator_is_reproducible_and_spread() {
        let mut first = Rng::new(7);
        let mut second = Rng::new(7);
        assert_eq!(first.next_u64(), second.next_u64());
        assert_ne!(first.fork(1).next_u64(), first.fork(2).next_u64());
        assert_eq!(first.fork(3).next_u64(), second.fork(3).next_u64());

        let mut rng = Rng::new(8);
        let (mut sum, mut square_sum) = (0.0, 0.0);
        let count = 40_000;
        for _ in 0..count {
            let uniform = rng.uniform();
            assert!((0.0..1.0).contains(&uniform));
            assert!(rng.below(7) < 7);
            let normal = rng.normal();
            sum += normal;
            square_sum += normal * normal;
        }
        let mean = sum / f64::from(count);
        let variance = square_sum / f64::from(count) - mean * mean;
        assert!(mean.abs() < 0.02, "mean {mean}");
        assert!((variance - 1.0).abs() < 0.03, "variance {variance}");

        let mut items: Vec<usize> = (0..100).collect();
        rng.shuffle(&mut items);
        assert_ne!(items, (0..100).collect::<Vec<_>>());
        items.sort_unstable();
        assert_eq!(items, (0..100).collect::<Vec<_>>());
    }
}
