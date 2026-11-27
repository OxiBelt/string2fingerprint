//! Fixed-size fingerprints of untrusted bytes.
//!
//! A [`Fingerprint`] summarizes one window of the input as counts of symbol
//! n-grams. [`fingerprints`] yields the fingerprint of every window.
//! The extraction is portable integer code: it reads each input byte once,
//! allocates nothing, and gives the same result on every target.
//!
//! The design document in `docs/design.md` specifies the layout that this
//! module implements. A model is only meaningful for the exact layout it was
//! trained on, so a change to the tables and constants here needs every model
//! to be retrained.

use core::fmt;
use core::iter::FusedIterator;
use core::slice::Chunks;

/// The number of features in a [`Fingerprint`].
pub const FINGERPRINT_LEN: usize = 3072;

/// The number of input bytes in one block. A window is two adjacent blocks.
pub const BLOCK_LEN: usize = 256;

/// The shortest input that has a window. Shorter inputs contain no n-gram.
pub const MIN_INPUT_LEN: usize = 2;

/// First bucket and bucket count of lexical bigrams with a non-alphanumeric
/// symbol.
const LEX2: (usize, u32) = (0, 384);
/// First bucket and bucket count of lexical trigrams with a non-alphanumeric
/// symbol.
const LEX3: (usize, u32) = (384, 640);
/// First bucket and bucket count of all-alphanumeric lexical trigrams.
const WORD3: (usize, u32) = (1024, 256);
/// First bucket and bucket count of shape bigrams. Every bigram has a
/// bucket of its own.
const SHAPE2: (usize, u32) = (1280, SHAPE_CLASSES * SHAPE_CLASSES);
/// First bucket and bucket count of shape trigrams.
const SHAPE3: (usize, u32) = (2304, 768);

/// The bucket ranges in fingerprint order. They are adjacent and cover the
/// whole fingerprint.
const RANGES: [(usize, u32); 5] = [LEX2, LEX3, WORD3, SHAPE2, SHAPE3];

const _: () = {
    let mut next = 0;
    let mut index = 0;
    while index < RANGES.len() {
        assert!(RANGES[index].0 == next);
        next += RANGES[index].1 as usize;
        index += 1;
    }
    assert!(next == FINGERPRINT_LEN);
};

/// Window counts are capped at the last index of [`COMPRESS`].
const COUNT_CAP: u8 = 15;

/// Maps a capped window count to a feature, about `255 * log2(1 + n) / 4`.
const COMPRESS: [u8; COUNT_CAP as usize + 1] = [
    0, 64, 101, 128, 148, 165, 179, 191, 202, 212, 221, 229, 236, 243, 249, 255,
];

/// Shape classes below this value are ASCII letters and digits.
const ALNUM_CLASSES: u8 = 5;

/// Lexical symbols: ASCII with letter case folded, and UTF-8 roles for bytes
/// at or above `0x80`.
const LEX: [u8; 256] = build_lex();

/// The number of shape classes. A shape symbol fits in [`SHAPE_BITS`] bits.
const SHAPE_CLASSES: u32 = 32;

/// The number of bits of a shape symbol.
const SHAPE_BITS: u32 = 5;

const _: () = assert!(SHAPE_CLASSES == 1 << SHAPE_BITS);

/// Shape symbols: the class of a byte, below [`SHAPE_CLASSES`].
const SHAPE: [u8; 256] = build_shape();

/// Folds a byte at or above `0x80` into its UTF-8 role, numbered from zero.
///
/// The ranges separate the second bytes and lead bytes that decide whether a
/// sequence is overlong (`C0`, `C1`, `E0 80..9F`, `F0 80..8F`), a surrogate
/// (`ED A0..BF`), or beyond U+10FFFF (`F4 90..BF`, `F5..FF`). `EF` has a
/// role of its own because it leads the fullwidth forms of ASCII.
const fn utf8_role(byte: u8) -> u8 {
    match byte {
        0x80..=0x8F => 0,
        0x90..=0x9F => 1,
        0xA0..=0xBF => 2,
        0xC0..=0xC1 => 3,
        0xC2..=0xDF => 4,
        0xE0 => 5,
        0xE1..=0xEC | 0xEE => 6,
        0xED => 7,
        0xEF => 8,
        0xF0 => 9,
        0xF1..=0xF3 => 10,
        0xF4 => 11,
        _ => 12,
    }
}

const fn build_lex() -> [u8; 256] {
    let mut table = [0u8; 256];
    let mut index = 0;
    while index < 256 {
        let byte = index as u8;
        table[index] = if byte < 0x80 {
            byte.to_ascii_lowercase()
        } else {
            0x80 + utf8_role(byte)
        };
        index += 1;
    }
    table
}

const fn build_shape() -> [u8; 256] {
    // Classes 0 to 4 are alphanumeric, 5 to 10 are whitespace and controls,
    // 11 to 18 are groups of ASCII punctuation, 19 to 31 are UTF-8 roles.
    // Punctuation is grouped by its part in escape syntax, because lexical
    // symbols already keep every mark apart.
    let mut table = [0u8; 256];
    let mut index = 0;
    while index < 256 {
        let byte = index as u8;
        table[index] = match byte {
            b'a'..=b'f' => 0,
            b'g'..=b'z' => 1,
            b'A'..=b'F' => 2,
            b'G'..=b'Z' => 3,
            b'0'..=b'9' => 4,
            b' ' => 5,
            b'\t' => 6,
            b'\n' => 7,
            b'\r' => 8,
            0x00 => 9,
            0x01..=0x1F | 0x7F => 10,
            b'%' => 11,
            b'\\' => 12,
            b'&' => 13,
            b'#' => 14,
            b'+' => 15,
            b'"' | b'\'' | b'`' => 16,
            b'<' | b'>' | b'(' | b')' | b'[' | b']' | b'{' | b'}' => 17,
            0x80..=0xFF => 19 + utf8_role(byte),
            _ => 18,
        };
        assert!((table[index] as u32) < SHAPE_CLASSES);
        index += 1;
    }
    table
}

/// Scrambles an n-gram key so that its high bits depend on every key bit.
const fn mix(key: u32) -> u32 {
    let mut hash = key.wrapping_mul(0x9E37_79B1);
    hash ^= hash >> 15;
    hash = hash.wrapping_mul(0x85EB_CA77);
    hash ^= hash >> 13;
    hash
}

/// Maps a hash to a bucket in `range`, given as first bucket and count.
const fn bucket(hash: u32, range: (usize, u32)) -> usize {
    // The product is below `range.1 << 32`, so the result is below `range.1`.
    range.0 + ((hash as u64 * range.1 as u64) >> 32) as usize
}

/// Selects the two most recent shape symbols.
const SHAPE2_MASK: u32 = (1 << (2 * SHAPE_BITS)) - 1;

/// Selects the three most recent shape symbols.
const SHAPE3_MASK: u32 = (1 << (3 * SHAPE_BITS)) - 1;

/// The symbols that precede the next byte.
#[derive(Clone, Copy, Debug, Default)]
struct Rolling {
    /// Lexical symbols of the previous two bytes, most recent in bits 0 to 7.
    lex: u32,
    /// Shape symbols of the previous two bytes, most recent in the low
    /// [`SHAPE_BITS`] bits.
    shape: u32,
    /// Whether each of the previous two bytes is alphanumeric, most recent in
    /// bit 0.
    alnum: u32,
    /// The number of bytes consumed so far, saturating at 2.
    seen: u8,
}

/// Adds the n-grams that end in each byte of `block` to `histogram`.
fn count_block(state: &mut Rolling, block: &[u8], histogram: &mut [u8; FINGERPRINT_LEN]) {
    let mut bump = |index: usize| {
        // Every index passed here is below `FINGERPRINT_LEN`: `bucket` stays
        // inside its range, and a shape bigram is below the size of `SHAPE2`.
        if let Some(count) = histogram.get_mut(index) {
            *count = count.saturating_add(1);
        }
    };
    for &byte in block {
        let lex = u32::from(LEX[usize::from(byte)]);
        let shape = u32::from(SHAPE[usize::from(byte)]);
        let alnum = u32::from(shape < u32::from(ALNUM_CLASSES));

        let lex3 = ((state.lex << 8) | lex) & 0x00FF_FFFF;
        let shape3 = ((state.shape << SHAPE_BITS) | shape) & SHAPE3_MASK;
        let alnum3 = ((state.alnum << 1) | alnum) & 0b111;

        if state.seen >= 1 && alnum3 & 0b011 != 0b011 {
            bump(bucket(mix((lex3 & 0xFFFF) | 0x0100_0000), LEX2));
        }
        if state.seen >= 2 {
            let range = if alnum3 == 0b111 { WORD3 } else { LEX3 };
            bump(bucket(mix(lex3 | 0x0200_0000), range));
        }
        if state.seen >= 1 {
            // Below `SHAPE2.1`, which is the number of shape bigrams.
            bump(SHAPE2.0 + (shape3 & SHAPE2_MASK) as usize);
        }
        if state.seen >= 2 {
            bump(bucket(mix(shape3 | 0x0300_0000), SHAPE3));
        }

        state.lex = lex3 & 0xFFFF;
        state.shape = shape3 & SHAPE2_MASK;
        state.alnum = alnum3 & 0b011;
        state.seen = (state.seen + 1).min(2);
    }
}

/// A fixed-size summary of one window of input bytes.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Fingerprint([u8; FINGERPRINT_LEN]);

impl Fingerprint {
    /// The fingerprint of a window that contains no n-gram.
    pub const ZERO: Self = Self([0; FINGERPRINT_LEN]);

    /// Creates a fingerprint from its features.
    #[must_use]
    pub const fn from_array(features: [u8; FINGERPRINT_LEN]) -> Self {
        Self(features)
    }

    /// Returns the features.
    #[must_use]
    pub const fn as_array(&self) -> &[u8; FINGERPRINT_LEN] {
        &self.0
    }

    /// Sets the features from the histograms of one or two blocks.
    fn fill(&mut self, first: &[u8; FINGERPRINT_LEN], second: Option<&[u8; FINGERPRINT_LEN]>) {
        match second {
            Some(second) => {
                for ((feature, &a), &b) in self.0.iter_mut().zip(first).zip(second) {
                    let count = a.saturating_add(b).min(COUNT_CAP);
                    *feature = COMPRESS[usize::from(count)];
                }
            }
            None => {
                for (feature, &a) in self.0.iter_mut().zip(first) {
                    *feature = COMPRESS[usize::from(a.min(COUNT_CAP))];
                }
            }
        }
    }
}

impl Default for Fingerprint {
    fn default() -> Self {
        Self::ZERO
    }
}

/// Returns the number of windows of an input of `len` bytes.
#[must_use]
pub const fn window_count(len: usize) -> usize {
    if len < MIN_INPUT_LEN {
        0
    } else {
        let blocks = len.div_ceil(BLOCK_LEN);
        if blocks > 1 { blocks - 1 } else { 1 }
    }
}

/// Yields the [`Fingerprint`] of every window of `input`, in input order.
///
/// An input shorter than [`MIN_INPUT_LEN`] has no window. Otherwise the input
/// has [`window_count`] windows.
#[must_use]
pub fn fingerprints(input: &[u8]) -> Fingerprints<'_> {
    Fingerprints::new(input)
}

/// The iterator returned by [`fingerprints`].
///
/// It holds three fingerprint-sized buffers, about 9 KiB, and no heap memory.
#[derive(Clone)]
pub struct Fingerprints<'a> {
    blocks: Chunks<'a, u8>,
    remaining: usize,
    started: bool,
    state: Rolling,
    previous: [u8; FINGERPRINT_LEN],
    current: [u8; FINGERPRINT_LEN],
    output: Fingerprint,
}

/// Shows the number of windows left and nothing derived from the input, so
/// that logging an iterator cannot disclose input bytes.
impl fmt::Debug for Fingerprints<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Fingerprints")
            .field("remaining", &self.remaining)
            .finish_non_exhaustive()
    }
}

impl<'a> Fingerprints<'a> {
    fn new(input: &'a [u8]) -> Self {
        let remaining = window_count(input.len());
        let input = if remaining == 0 { &[][..] } else { input };
        Self {
            blocks: input.chunks(BLOCK_LEN),
            remaining,
            started: false,
            state: Rolling::default(),
            previous: [0; FINGERPRINT_LEN],
            current: [0; FINGERPRINT_LEN],
            output: Fingerprint::ZERO,
        }
    }

    /// Advances to the next window and returns its fingerprint by reference.
    ///
    /// This is [`Iterator::next`] without the copy.
    pub fn next_window(&mut self) -> Option<&Fingerprint> {
        if !self.started {
            self.started = true;
            let first = self.blocks.next()?;
            count_block(&mut self.state, first, &mut self.previous);
            let Some(second) = self.blocks.next() else {
                self.output.fill(&self.previous, None);
                self.remaining = 0;
                return Some(&self.output);
            };
            count_block(&mut self.state, second, &mut self.current);
        } else {
            let block = self.blocks.next()?;
            self.current.fill(0);
            count_block(&mut self.state, block, &mut self.current);
        }
        self.output.fill(&self.previous, Some(&self.current));
        core::mem::swap(&mut self.previous, &mut self.current);
        self.remaining = self.remaining.saturating_sub(1);
        Some(&self.output)
    }
}

impl Iterator for Fingerprints<'_> {
    type Item = Fingerprint;

    fn next(&mut self) -> Option<Fingerprint> {
        self.next_window().cloned()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for Fingerprints<'_> {}

impl FusedIterator for Fingerprints<'_> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::Rng;

    /// Recomputes every window from the specification without streaming:
    /// it finds each n-gram by position and assigns it to the block of its
    /// last byte.
    fn reference(input: &[u8]) -> Vec<Fingerprint> {
        if input.len() < MIN_INPUT_LEN {
            return Vec::new();
        }
        let block_count = input.len().div_ceil(BLOCK_LEN);
        let mut histograms = vec![[0u32; FINGERPRINT_LEN]; block_count];
        let lex = |index: usize| u32::from(LEX[usize::from(input[index])]);
        let shape = |index: usize| u32::from(SHAPE[usize::from(input[index])]);
        let alnum = |index: usize| SHAPE[usize::from(input[index])] < ALNUM_CLASSES;
        for end in 0..input.len() {
            let histogram = &mut histograms[end / BLOCK_LEN];
            if end >= 1 && !(alnum(end - 1) && alnum(end)) {
                let key = (lex(end - 1) << 8) | lex(end) | 0x0100_0000;
                histogram[bucket(mix(key), LEX2)] += 1;
            }
            if end >= 2 {
                let key = (lex(end - 2) << 16) | (lex(end - 1) << 8) | lex(end) | 0x0200_0000;
                let word = alnum(end - 2) && alnum(end - 1) && alnum(end);
                histogram[bucket(mix(key), if word { WORD3 } else { LEX3 })] += 1;
            }
            if end >= 1 {
                let bigram = (shape(end - 1) << SHAPE_BITS) | shape(end);
                histogram[SHAPE2.0 + bigram as usize] += 1;
            }
            if end >= 2 {
                let key = (shape(end - 2) << (2 * SHAPE_BITS))
                    | (shape(end - 1) << SHAPE_BITS)
                    | shape(end)
                    | 0x0300_0000;
                histogram[bucket(mix(key), SHAPE3)] += 1;
            }
        }
        let compress = |count: u32| COMPRESS[count.min(u32::from(COUNT_CAP)) as usize];
        let window = |blocks: &[[u32; FINGERPRINT_LEN]]| {
            let mut features = [0u8; FINGERPRINT_LEN];
            for (index, feature) in features.iter_mut().enumerate() {
                *feature = compress(blocks.iter().map(|block| block[index]).sum());
            }
            Fingerprint::from_array(features)
        };
        if block_count == 1 {
            vec![window(&histograms)]
        } else {
            histograms.windows(2).map(window).collect()
        }
    }

    fn single(input: &[u8]) -> Fingerprint {
        let mut windows = fingerprints(input);
        let fingerprint = windows.next().expect("input has a window");
        assert!(windows.next().is_none());
        fingerprint
    }

    /// Returns the number of n-grams counted in `range`, for a fingerprint
    /// whose counts are all below the cap.
    fn counted(fingerprint: &Fingerprint, range: (usize, u32)) -> usize {
        fingerprint.as_array()[range.0..range.0 + range.1 as usize]
            .iter()
            .map(|feature| COMPRESS.iter().position(|value| value == feature).unwrap())
            .sum()
    }

    /// Returns the n-gram counts of `input` in the order of [`RANGES`].
    fn counts(input: &[u8]) -> [usize; 5] {
        let fingerprint = single(input);
        RANGES.map(|range| counted(&fingerprint, range))
    }

    #[test]
    fn tables_cover_documented_classes() {
        assert_eq!(LEX[usize::from(b'Q')], b'q');
        assert_eq!(LEX[usize::from(b'q')], b'q');
        assert_eq!(LEX[usize::from(b'%')], b'%');
        assert_eq!(LEX[0x00], 0x00);
        assert!(
            LEX[0x80..]
                .iter()
                .all(|&symbol| (0x80..=0x8C).contains(&symbol))
        );
        assert!(LEX[..0x80].iter().all(|&symbol| symbol < 0x80));

        for class in 0..SHAPE_CLASSES {
            assert!(
                SHAPE.iter().any(|&symbol| u32::from(symbol) == class),
                "class {class} is unused"
            );
        }
        for byte in 0..=255u8 {
            let alnum = SHAPE[usize::from(byte)] < ALNUM_CLASSES;
            assert_eq!(alnum, byte.is_ascii_alphanumeric(), "byte {byte:#04x}");
        }
        // ASCII punctuation is grouped, and the marks of escape syntax have
        // classes of their own.
        for byte in (0..0x80u8).filter(u8::is_ascii_punctuation) {
            assert!((11..19).contains(&SHAPE[usize::from(byte)]));
        }
        for mark in *b"%\\&#+" {
            let class = SHAPE[usize::from(mark)];
            let members = SHAPE.iter().filter(|&&symbol| symbol == class).count();
            assert_eq!(members, 1, "class of byte {mark:#04x}");
        }
    }

    #[test]
    fn every_shape_bigram_has_its_own_bucket() {
        assert_eq!(SHAPE2.1, SHAPE_CLASSES * SHAPE_CLASSES);
        // One representative byte of each class.
        let representative = |class: u32| {
            (0..=255u8)
                .find(|&byte| u32::from(SHAPE[usize::from(byte)]) == class)
                .unwrap()
        };
        let mut seen = vec![false; SHAPE2.1 as usize];
        for first in 0..SHAPE_CLASSES {
            for second in 0..SHAPE_CLASSES {
                let fingerprint = single(&[representative(first), representative(second)]);
                let buckets = &fingerprint.as_array()[SHAPE2.0..SHAPE2.0 + SHAPE2.1 as usize];
                let index = buckets.iter().position(|&feature| feature != 0).unwrap();
                assert_eq!(counted(&fingerprint, SHAPE2), 1);
                assert!(!seen[index], "classes {first} and {second} share a bucket");
                seen[index] = true;
            }
        }
        assert!(seen.iter().all(|&used| used));
    }

    #[test]
    fn compress_is_strictly_increasing() {
        assert_eq!(COMPRESS[0], 0);
        assert_eq!(COMPRESS[usize::from(COUNT_CAP)], 255);
        assert!(COMPRESS.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn buckets_stay_in_their_ranges() {
        let mut rng = Rng::new(7);
        for range in RANGES {
            let end = range.0 + range.1 as usize;
            assert_eq!(bucket(0, range), range.0);
            assert_eq!(bucket(u32::MAX, range), end - 1);
            for _ in 0..10_000 {
                assert!((range.0..end).contains(&bucket(rng.next_u32(), range)));
            }
        }
    }

    #[test]
    fn short_inputs_have_no_window() {
        assert_eq!(window_count(0), 0);
        assert_eq!(window_count(1), 0);
        assert!(fingerprints(b"").next().is_none());
        assert!(fingerprints(b"'").next().is_none());
        assert_eq!(fingerprints(b"<").len(), 0);
    }

    #[test]
    fn window_count_matches_iteration_at_block_edges() {
        let lengths = [
            2, 3, 4, 255, 256, 257, 258, 511, 512, 513, 767, 768, 769, 1024, 1025, 4096,
        ];
        for len in lengths {
            let input = vec![b'/'; len];
            let windows = fingerprints(&input);
            assert_eq!(windows.len(), window_count(len), "length {len}");
            assert_eq!(windows.count(), window_count(len), "length {len}");
        }
        assert_eq!(window_count(256), 1);
        assert_eq!(window_count(257), 1);
        assert_eq!(window_count(512), 1);
        assert_eq!(window_count(513), 2);
    }

    #[test]
    fn debug_output_contains_no_input_bytes() {
        let mut windows = fingerprints(b"Authorization: Bearer SECRET");
        let before = format!("{windows:?}");
        assert_eq!(before, "Fingerprints { remaining: 1, .. }");
        windows.next_window();
        assert_eq!(format!("{windows:?}"), "Fingerprints { remaining: 0, .. }");
    }

    #[test]
    fn iterator_is_fused_and_reports_its_length() {
        let input = vec![b'%'; 3 * BLOCK_LEN + 1];
        let mut windows = fingerprints(&input);
        assert_eq!(windows.len(), 3);
        for remaining in (0..3).rev() {
            assert!(windows.next().is_some());
            assert_eq!(windows.len(), remaining);
        }
        assert!(windows.next().is_none());
        assert!(windows.next().is_none());
        assert!(windows.next_window().is_none());
    }

    #[test]
    fn ngrams_are_counted_in_their_ranges() {
        // Counts are in the order LEX2, LEX3, WORD3, SHAPE2, SHAPE3.
        // Alphanumeric bigrams have no lexical bucket.
        assert_eq!(counts(b"ab"), [0, 0, 0, 1, 0]);
        assert_eq!(counts(b"abc"), [0, 0, 1, 2, 1]);
        // "..", "./", and "../".
        assert_eq!(counts(b"../"), [2, 1, 0, 2, 1]);
        // "a=", "=1", "a=1", then "1&" and "=1&".
        assert_eq!(counts(b"a=1&"), [3, 2, 0, 3, 2]);
        assert_eq!(counts(b"abcde"), [0, 0, 3, 4, 3]);
    }

    #[test]
    fn letter_case_changes_only_shape_buckets() {
        let lower = single(b"union select");
        let mixed = single(b"UnIoN SeLeCt");
        assert_eq!(lower.as_array()[..SHAPE2.0], mixed.as_array()[..SHAPE2.0]);
        for range in [SHAPE2, SHAPE3] {
            let range = range.0..range.0 + range.1 as usize;
            assert_ne!(lower.as_array()[range.clone()], mixed.as_array()[range]);
        }
    }

    #[test]
    fn non_ascii_bytes_fold_to_utf8_roles() {
        // Different Hangul syllables whose bytes have the same UTF-8 roles.
        let first = single("가각간".as_bytes());
        let second = single("갇갈갉".as_bytes());
        assert_eq!(first, second);
        // An overlong encoding of '/' differs from a valid two-byte sequence.
        assert_ne!(single(b"a\xC0\xAFb"), single(b"a\xC3\xAFb"));
        // A surrogate differs from a valid three-byte sequence.
        assert_ne!(single(b"\xED\xA0\x80"), single(b"\xED\x9F\x80"));
        // A fullwidth form differs from another three-byte sequence.
        assert_ne!(single("\u{FF1C}".as_bytes()), single("\u{E01C}".as_bytes()));
    }

    #[test]
    fn repeated_ngrams_saturate_at_the_cap() {
        let fingerprint = single(&[b'%'; 2 * BLOCK_LEN]);
        let nonzero: Vec<u8> = fingerprint
            .as_array()
            .iter()
            .copied()
            .filter(|&feature| feature != 0)
            .collect();
        // One bucket in every range except the all-alphanumeric one.
        assert_eq!(nonzero, [255; 4]);
    }

    #[test]
    fn features_are_compressed_counts() {
        let mut rng = Rng::new(11);
        let mut input = vec![0u8; 700];
        rng.fill(&mut input);
        for fingerprint in fingerprints(&input) {
            assert!(
                fingerprint
                    .as_array()
                    .iter()
                    .all(|feature| COMPRESS.contains(feature))
            );
        }
    }

    #[test]
    fn streaming_matches_reference() {
        let mut rng = Rng::new(3);
        let mut lengths: Vec<usize> = (0..=20).collect();
        lengths.extend([
            254, 255, 256, 257, 258, 259, 260, 510, 511, 512, 513, 514, 515, 767, 768, 769, 770,
            1023, 1024, 1025, 1300, 2049,
        ]);
        for len in lengths {
            for style in 0..3 {
                let mut input = vec![0u8; len];
                rng.fill(&mut input);
                match style {
                    // Arbitrary bytes.
                    0 => {}
                    // Printable ASCII, which exercises every n-gram range.
                    1 => input
                        .iter_mut()
                        .for_each(|byte| *byte = 0x20 + *byte % 0x5F),
                    // A small alphabet, which exercises the count cap.
                    _ => input
                        .iter_mut()
                        .for_each(|byte| *byte = b"a./%'"[usize::from(*byte % 5)]),
                }
                let streamed: Vec<Fingerprint> = fingerprints(&input).collect();
                assert_eq!(streamed, reference(&input), "length {len}, style {style}");
            }
        }
    }

    #[test]
    fn a_payload_is_fully_inside_some_window() {
        let payload = b"' UNION SELECT password FROM users WHERE '1'='1' -- ";
        let alone = single(payload);
        let filler = b"The quick brown fox jumps over the lazy dog. ";
        for offset in (0..900).step_by(7) {
            let mut input: Vec<u8> = filler.iter().copied().cycle().take(offset).collect();
            input.extend_from_slice(payload);
            input.extend(filler.iter().copied().cycle().take(900 - offset));
            let covered = fingerprints(&input).any(|window| {
                window
                    .as_array()
                    .iter()
                    .zip(alone.as_array())
                    .all(|(have, need)| have >= need)
            });
            assert!(covered, "offset {offset}");
        }
    }

    #[test]
    fn layout_is_pinned() {
        // A model depends on the exact tables, hash, and bucket ranges. A
        // change here needs every model to be retrained.
        fn digest(fingerprint: &Fingerprint) -> u64 {
            fingerprint
                .as_array()
                .iter()
                .fold(0xCBF2_9CE4_8422_2325, |hash, &byte| {
                    (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01B3)
                })
        }
        let cases: [(&[u8], u64); 4] = [
            (b"../", PINNED[0]),
            (b"<script>alert(1)</script>", PINNED[1]),
            (b"id=1%27%20OR%20%271%27%3D%271", PINNED[2]),
            (b"\xC0\xAE\xC0\xAE/\x00etc/passwd", PINNED[3]),
        ];
        for (input, expected) in cases {
            assert_eq!(
                digest(&single(input)),
                expected,
                "input of {} bytes",
                input.len()
            );
        }
    }

    const PINNED: [u64; 4] = [
        0x7339884348943B08,
        0xBED0F82A454C2DD8,
        0x54F6EEFD99E6D2E9,
        0x6C79FEBB3C881605,
    ];
}
