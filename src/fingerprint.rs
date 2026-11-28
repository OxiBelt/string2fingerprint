//! Fixed-size fingerprints of untrusted bytes.
//!
//! A [`Fingerprint`] summarizes one window of the input as counts of symbol
//! n-grams. [`fingerprints`] yields the fingerprint of every window.
//! The extraction is portable integer code: it reads each input byte once,
//! allocates nothing, and gives the same result on every target.
//!
//! The parts of the extraction are public as well, for callers that inspect
//! or explain a fingerprint: [`lexical_symbol`] and [`shape_class`] map a
//! byte to its symbols, [`Section`] names the range of features that counts
//! one kind of n-gram, [`ngrams`] yields every n-gram of an input with the
//! feature it counts in, and [`window_range`] gives the input bytes of a
//! window.
//!
//! [`Fingerprinter`] extracts the same fingerprints from an input that
//! arrives in parts, such as a request body read from a socket.
//!
//! ```
//! use string2fingerprint::fingerprint::{self, Section};
//!
//! let input = b"../etc/passwd";
//! let window = fingerprint::fingerprints(input).next().expect("input has a window");
//! // Every n-gram that ends in the window raises the feature it counts in.
//! for ngram in fingerprint::ngrams(input) {
//!     assert!(window.as_array()[ngram.feature()] > 0);
//!     assert!(ngram.section().range().contains(&ngram.feature()));
//! }
//! assert_eq!(fingerprint::window_range(input.len(), 0), Some(0..input.len()));
//! assert!(window.section(Section::LexicalTrigram).iter().any(|&feature| feature > 0));
//! ```
//!
//! The design document in `docs/design.md` specifies the layout that this
//! module implements. A model is only meaningful for the exact layout it was
//! trained on, so a change to the tables and constants here needs every model
//! to be retrained.

use core::fmt;
use core::iter::FusedIterator;
use core::ops::Range;
use core::slice::Chunks;

/// The number of features in a [`Fingerprint`].
pub const FINGERPRINT_LEN: usize = 3072;

/// The number of input bytes in one block. A window is two adjacent blocks.
pub const BLOCK_LEN: usize = 256;

/// The shortest input that has a window. Shorter inputs contain no n-gram.
pub const MIN_INPUT_LEN: usize = 2;

/// The number of shape classes. [`shape_class`] returns a value below it.
pub const SHAPE_CLASS_COUNT: usize = SHAPE_CLASSES as usize;

/// The highest n-gram count that a feature distinguishes. Higher counts give
/// the feature of this count. See [`feature_value`].
pub const MAX_COUNT: usize = COUNT_CAP as usize;

/// The version of the fingerprint layout that this crate extracts.
///
/// The layout is everything that decides the features of a window: the
/// symbol tables, the sections and their n-grams, the hashes, the blocks and
/// windows, and the count table. A model records the layout version it was
/// trained on, and loads only where the layout is compatible. The version
/// follows Semantic Versioning:
///
/// - The major number changes when any input can get a different
///   fingerprint. Models of another major number are refused, because every
///   model must be retrained.
/// - The minor number changes when features are added that every earlier
///   model can ignore, because they were zero in every fingerprint that the
///   earlier layout extracted.
/// - The patch number changes when the layout is documented or implemented
///   differently but no fingerprint changes.
///
/// A model of layout `m` loads where this version matches the requirement
/// `^m`, as [`VersionReq::caret`](crate::version::VersionReq::caret) defines
/// it, so a model never loads with a layout older than its own.
pub const LAYOUT_VERSION: crate::version::Version = crate::version::Version::new(1, 0, 0);

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

/// A range of features in a [`Fingerprint`] that counts one kind of n-gram.
///
/// The sections are adjacent, in the order of [`Section::ALL`], and cover
/// the whole fingerprint. A lexical n-gram is one of [`lexical_symbol`]s and
/// a shape n-gram one of [`shape_class`]es. The design document gives the
/// reasons for the split.
///
/// ```
/// use string2fingerprint::fingerprint::{FINGERPRINT_LEN, Section};
///
/// assert_eq!(Section::ALL[0].range().start, 0);
/// assert_eq!(Section::ALL[4].range().end, FINGERPRINT_LEN);
/// assert_eq!(Section::of(1280), Some(Section::ShapeBigram));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum Section {
    /// Features 0 to 383: hashed lexical bigrams in which at least one
    /// symbol is not alphanumeric.
    LexicalBigram,
    /// Features 384 to 1023: hashed lexical trigrams in which at least one
    /// symbol is not alphanumeric.
    LexicalTrigram,
    /// Features 1024 to 1279: hashed lexical trigrams of alphanumeric
    /// symbols only.
    WordTrigram,
    /// Features 1280 to 2303: one feature for each shape bigram.
    ShapeBigram,
    /// Features 2304 to 3071: hashed shape trigrams.
    ShapeTrigram,
}

impl Section {
    /// Every section, in fingerprint order.
    ///
    /// The type is a slice, so that a section added with a new layout does
    /// not change it.
    pub const ALL: &'static [Self] = &[
        Self::LexicalBigram,
        Self::LexicalTrigram,
        Self::WordTrigram,
        Self::ShapeBigram,
        Self::ShapeTrigram,
    ];

    /// Returns the first feature and the number of features.
    const fn bounds(self) -> (usize, u32) {
        match self {
            Self::LexicalBigram => LEX2,
            Self::LexicalTrigram => LEX3,
            Self::WordTrigram => WORD3,
            Self::ShapeBigram => SHAPE2,
            Self::ShapeTrigram => SHAPE3,
        }
    }

    /// Returns the indexes of the features of this section.
    #[must_use]
    pub const fn range(self) -> Range<usize> {
        let (start, len) = self.bounds();
        start..start + len as usize
    }

    /// Returns the section that holds feature `index`, or `None` when the
    /// index is not below [`FINGERPRINT_LEN`].
    #[must_use]
    pub const fn of(index: usize) -> Option<Self> {
        let mut position = 0;
        while position < Self::ALL.len() {
            let section = Self::ALL[position];
            let (start, len) = section.bounds();
            if index >= start && index < start + len as usize {
                return Some(section);
            }
            position += 1;
        }
        None
    }

    /// Returns the number of symbols in an n-gram of this section, 2 or 3.
    #[must_use]
    pub const fn order(self) -> usize {
        match self {
            Self::LexicalBigram | Self::ShapeBigram => 2,
            Self::LexicalTrigram | Self::WordTrigram | Self::ShapeTrigram => 3,
        }
    }

    /// Returns whether the n-grams of this section are of shape classes
    /// rather than lexical symbols.
    #[must_use]
    pub const fn is_shape(self) -> bool {
        matches!(self, Self::ShapeBigram | Self::ShapeTrigram)
    }

    /// Returns whether several n-grams share a feature of this section.
    ///
    /// Every section except [`Section::ShapeBigram`] maps an n-gram to its
    /// feature with a fixed hash.
    #[must_use]
    pub const fn is_hashed(self) -> bool {
        !matches!(self, Self::ShapeBigram)
    }

    /// Returns a stable `snake_case` identifier.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::LexicalBigram => "lexical_bigram",
            Self::LexicalTrigram => "lexical_trigram",
            Self::WordTrigram => "word_trigram",
            Self::ShapeBigram => "shape_bigram",
            Self::ShapeTrigram => "shape_trigram",
        }
    }
}

impl fmt::Display for Section {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

const _: () = {
    let mut index = 0;
    while index < RANGES.len() {
        let (start, len) = Section::ALL[index].bounds();
        assert!(start == RANGES[index].0 && len == RANGES[index].1);
        index += 1;
    }
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

/// Returns the lexical symbol of `byte`.
///
/// An ASCII byte maps to itself with letter case folded, so `b'Q'` maps to
/// `b'q'`. A byte at or above `0x80` maps to `0x80` plus the number of its
/// role in UTF-8, from 0 to 12: three ranges of continuation bytes, then
/// lead bytes grouped so that overlong, surrogate, and out-of-range
/// sequences differ from valid ones.
#[must_use]
pub const fn lexical_symbol(byte: u8) -> u8 {
    LEX[byte as usize]
}

/// Returns the shape class of `byte`, below [`SHAPE_CLASS_COUNT`].
///
/// Classes 0 to 4 are lowercase hex letters, other lowercase letters,
/// uppercase hex letters, other uppercase letters, and digits. Classes 5 to
/// 10 are space, tab, line feed, carriage return, NUL, and other control
/// bytes. Classes 11 to 18 group ASCII punctuation by its part in escape
/// syntax, and classes 19 to 31 are the UTF-8 roles of [`lexical_symbol`].
#[must_use]
pub const fn shape_class(byte: u8) -> u8 {
    SHAPE[byte as usize]
}

/// Returns the feature of a window in which an n-gram occurs `count` times.
///
/// Counts above [`MAX_COUNT`] give the feature of [`MAX_COUNT`], which is
/// 255. The feature grows with the logarithm of the count.
#[must_use]
pub const fn feature_value(count: usize) -> u8 {
    let capped = if count > MAX_COUNT { MAX_COUNT } else { count };
    COMPRESS[capped]
}

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

/// Consumes one byte and passes the feature of each n-gram that ends in it
/// to `emit`, in the order of the sections.
///
/// Every feature passed to `emit` is below [`FINGERPRINT_LEN`]: `bucket`
/// stays inside its range, and a shape bigram is below the size of
/// `SHAPE2`.
#[inline(always)]
fn step(state: &mut Rolling, byte: u8, mut emit: impl FnMut(usize)) {
    let lex = u32::from(LEX[usize::from(byte)]);
    let shape = u32::from(SHAPE[usize::from(byte)]);
    let alnum = u32::from(shape < u32::from(ALNUM_CLASSES));

    let lex3 = ((state.lex << 8) | lex) & 0x00FF_FFFF;
    let shape3 = ((state.shape << SHAPE_BITS) | shape) & SHAPE3_MASK;
    let alnum3 = ((state.alnum << 1) | alnum) & 0b111;

    if state.seen >= 1 && alnum3 & 0b011 != 0b011 {
        emit(bucket(mix((lex3 & 0xFFFF) | 0x0100_0000), LEX2));
    }
    if state.seen >= 2 {
        let range = if alnum3 == 0b111 { WORD3 } else { LEX3 };
        emit(bucket(mix(lex3 | 0x0200_0000), range));
    }
    if state.seen >= 1 {
        // Below `SHAPE2.1`, which is the number of shape bigrams.
        emit(SHAPE2.0 + (shape3 & SHAPE2_MASK) as usize);
    }
    if state.seen >= 2 {
        emit(bucket(mix(shape3 | 0x0300_0000), SHAPE3));
    }

    state.lex = lex3 & 0xFFFF;
    state.shape = shape3 & SHAPE2_MASK;
    state.alnum = alnum3 & 0b011;
    state.seen = (state.seen + 1).min(2);
}

/// Adds the n-grams that end in each byte of `block` to `histogram`.
fn count_block(state: &mut Rolling, block: &[u8], histogram: &mut [u8; FINGERPRINT_LEN]) {
    for &byte in block {
        step(state, byte, |index| {
            // `step` passes only indexes below `FINGERPRINT_LEN`.
            if let Some(count) = histogram.get_mut(index) {
                *count = count.saturating_add(1);
            }
        });
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

    /// Creates a fingerprint from a slice of [`FINGERPRINT_LEN`] features,
    /// or returns `None` when the slice has another length.
    #[must_use]
    pub fn from_slice(features: &[u8]) -> Option<Self> {
        features.try_into().ok().map(Self)
    }

    /// Returns the features of `section`.
    #[must_use]
    pub fn section(&self, section: Section) -> &[u8] {
        &self.0[section.range()]
    }

    /// Iterates over the index and value of every feature that is not zero,
    /// in index order.
    pub fn nonzero(&self) -> impl Iterator<Item = (usize, u8)> + '_ {
        self.0
            .iter()
            .enumerate()
            .filter(|&(_, &feature)| feature != 0)
            .map(|(index, &feature)| (index, feature))
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

impl AsRef<[u8]> for Fingerprint {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl From<[u8; FINGERPRINT_LEN]> for Fingerprint {
    fn from(features: [u8; FINGERPRINT_LEN]) -> Self {
        Self(features)
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

/// Returns the input bytes that window `index` of an input of `len` bytes
/// covers, or `None` when the input has no such window.
///
/// Window `index` starts at block `index` and covers up to two blocks.
/// Consecutive windows share one block.
///
/// ```
/// use string2fingerprint::fingerprint::window_range;
///
/// assert_eq!(window_range(100, 0), Some(0..100));
/// assert_eq!(window_range(600, 1), Some(256..600));
/// assert_eq!(window_range(600, 2), None);
/// assert_eq!(window_range(1, 0), None);
/// ```
#[must_use]
pub const fn window_range(len: usize, index: usize) -> Option<Range<usize>> {
    if index >= window_count(len) {
        return None;
    }
    // The window exists, so its first block starts inside the input and
    // the product is below `len`.
    let start = index * BLOCK_LEN;
    let end = start.saturating_add(2 * BLOCK_LEN);
    Some(start..if end < len { end } else { len })
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

/// Extracts the fingerprints of an input that arrives in parts.
///
/// The windows are those that [`fingerprints`] yields for the whole input,
/// in the same order, however the input is split. A window is passed on as
/// soon as its second block is complete. The last window can only be known
/// at the end of the input, so [`Fingerprinter::finish`] returns it.
///
/// A fingerprinter holds three fingerprint-sized buffers, about 9 KiB, and
/// no heap memory. Box it or keep it in a long-lived place when that is too
/// much for a stack.
///
/// ```
/// use string2fingerprint::fingerprint::{Fingerprint, Fingerprinter, fingerprints};
///
/// let body = vec![b'a'; 1000];
/// let mut streamed: Vec<Fingerprint> = Vec::new();
/// let mut fingerprinter = Fingerprinter::new();
/// for part in body.chunks(100) {
///     fingerprinter.update(part, |window| streamed.push(window.clone()));
/// }
/// streamed.extend(fingerprinter.finish().cloned());
/// assert_eq!(streamed, fingerprints(&body).collect::<Vec<_>>());
/// ```
#[derive(Clone)]
pub struct Fingerprinter {
    state: Rolling,
    /// The histogram of the last complete block, when `has_previous`.
    previous: [u8; FINGERPRINT_LEN],
    /// The histogram of the block being filled.
    current: [u8; FINGERPRINT_LEN],
    output: Fingerprint,
    /// The number of bytes in the block being filled, below [`BLOCK_LEN`].
    filled: usize,
    /// Whether a block is complete.
    has_previous: bool,
    /// The number of bytes consumed since the input started.
    len: u64,
    /// The number of windows passed on since the input started.
    windows: u64,
}

/// Shows the counts and nothing derived from the input.
impl fmt::Debug for Fingerprinter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Fingerprinter")
            .field("len", &self.len)
            .field("windows", &self.windows)
            .finish_non_exhaustive()
    }
}

impl Default for Fingerprinter {
    fn default() -> Self {
        Self::new()
    }
}

impl Fingerprinter {
    /// Creates a fingerprinter at the start of an input.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            state: Rolling {
                lex: 0,
                shape: 0,
                alnum: 0,
                seen: 0,
            },
            previous: [0; FINGERPRINT_LEN],
            current: [0; FINGERPRINT_LEN],
            output: Fingerprint::ZERO,
            filled: 0,
            has_previous: false,
            len: 0,
            windows: 0,
        }
    }

    /// Consumes the next part of the input and passes the fingerprint of
    /// every window that it completes to `on_window`, in input order.
    ///
    /// A part of `n` bytes completes at most `n / 256 + 1` windows. The work
    /// is linear in the length of the part, and nothing is allocated.
    pub fn update(&mut self, part: &[u8], mut on_window: impl FnMut(&Fingerprint)) {
        let mut rest = part;
        while !rest.is_empty() {
            let take = (BLOCK_LEN - self.filled).min(rest.len());
            let (block, tail) = rest.split_at(take);
            count_block(&mut self.state, block, &mut self.current);
            self.filled += take;
            // A slice is shorter than `u64::MAX` bytes on every target.
            self.len = self.len.saturating_add(take as u64);
            rest = tail;
            if self.filled == BLOCK_LEN {
                if self.has_previous {
                    self.output.fill(&self.previous, Some(&self.current));
                    self.windows += 1;
                    on_window(&self.output);
                }
                core::mem::swap(&mut self.previous, &mut self.current);
                self.current.fill(0);
                self.has_previous = true;
                self.filled = 0;
            }
        }
    }

    /// Ends the input, returns the fingerprint of its last window if no
    /// call to [`update`](Self::update) passed it on, and starts a new
    /// input.
    ///
    /// An input that ends on a block boundary after its first window has
    /// no window left. An input shorter than [`MIN_INPUT_LEN`] has no
    /// window at all.
    pub fn finish(&mut self) -> Option<&Fingerprint> {
        let last = if self.filled > 0 {
            if self.has_previous {
                self.output.fill(&self.previous, Some(&self.current));
                true
            } else if self.len >= MIN_INPUT_LEN as u64 {
                self.output.fill(&self.current, None);
                true
            } else {
                false
            }
        } else if self.has_previous && self.windows == 0 {
            self.output.fill(&self.previous, None);
            true
        } else {
            false
        };
        self.reset();
        last.then_some(&self.output)
    }

    /// Discards the input so far and starts a new input.
    pub fn reset(&mut self) {
        self.state = Rolling::default();
        self.previous.fill(0);
        self.current.fill(0);
        self.filled = 0;
        self.has_previous = false;
        self.len = 0;
        self.windows = 0;
    }

    /// Returns the number of bytes consumed since the input started.
    #[must_use]
    pub const fn len(&self) -> u64 {
        self.len
    }

    /// Returns whether no byte was consumed since the input started.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the number of windows passed on since the input started.
    #[must_use]
    pub const fn windows(&self) -> u64 {
        self.windows
    }
}

/// One n-gram of an input and the feature that counts it.
///
/// [`ngrams`] yields them. The type holds positions and a feature index,
/// never input bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NGram {
    end: usize,
    feature: usize,
    section: Section,
}

impl NGram {
    /// Returns the input bytes of the n-gram.
    #[must_use]
    pub const fn range(&self) -> Range<usize> {
        self.end + 1 - self.section.order()..self.end + 1
    }

    /// Returns the index of the block that holds the last byte of the
    /// n-gram, which is the block that counts it.
    #[must_use]
    pub const fn block(&self) -> usize {
        self.end / BLOCK_LEN
    }

    /// Returns the index of the feature that counts the n-gram.
    #[must_use]
    pub const fn feature(&self) -> usize {
        self.feature
    }

    /// Returns the section of the feature.
    #[must_use]
    pub const fn section(&self) -> Section {
        self.section
    }
}

/// Yields every n-gram of `input` that a fingerprint counts, ordered by its
/// last byte and then by [`Section`].
///
/// A window counts each n-gram whose [`block`](NGram::block) it covers, so
/// the feature of a window is [`feature_value`] of the number of n-grams of
/// its blocks with that feature. An input shorter than [`MIN_INPUT_LEN`]
/// has no n-gram. The iterator allocates nothing.
#[must_use]
pub fn ngrams(input: &[u8]) -> NGrams<'_> {
    NGrams {
        input,
        position: 0,
        state: Rolling::default(),
        pending: [0; 4],
        pending_len: 0,
        pending_next: 0,
    }
}

/// The iterator returned by [`ngrams`].
#[derive(Clone)]
pub struct NGrams<'a> {
    input: &'a [u8],
    /// The index of the next byte to consume.
    position: usize,
    state: Rolling,
    /// The features of the n-grams that end in the byte before `position`.
    pending: [usize; 4],
    pending_len: u8,
    pending_next: u8,
}

/// Shows the position and nothing derived from the input.
impl fmt::Debug for NGrams<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NGrams")
            .field("position", &self.position)
            .finish_non_exhaustive()
    }
}

impl Iterator for NGrams<'_> {
    type Item = NGram;

    fn next(&mut self) -> Option<NGram> {
        while self.pending_next == self.pending_len {
            let &byte = self.input.get(self.position)?;
            self.position += 1;
            let (pending, len) = (&mut self.pending, &mut self.pending_len);
            *len = 0;
            step(&mut self.state, byte, |feature| {
                // `step` passes at most one feature for each section that
                // can end in a byte, which is four.
                if let Some(slot) = pending.get_mut(usize::from(*len)) {
                    *slot = feature;
                    *len += 1;
                }
            });
            self.pending_next = 0;
        }
        let feature = self.pending[usize::from(self.pending_next)];
        self.pending_next += 1;
        // `step` passes only features below `FINGERPRINT_LEN`, which all
        // have a section.
        let section = Section::of(feature)?;
        Some(NGram {
            end: self.position - 1,
            feature,
            section,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let pending = usize::from(self.pending_len - self.pending_next);
        let rest = (self.input.len() - self.position).checked_mul(4);
        (pending, rest.and_then(|rest| rest.checked_add(pending)))
    }
}

impl FusedIterator for NGrams<'_> {}

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
    fn sections_partition_the_fingerprint() {
        let mut next = 0;
        for (index, &section) in Section::ALL.iter().enumerate() {
            let range = section.range();
            assert_eq!(range.start, next);
            assert_eq!((range.start, range.len() as u32), RANGES[index]);
            next = range.end;
            assert_eq!(section.to_string(), section.name());
            assert_eq!(section.is_hashed(), section != Section::ShapeBigram);
        }
        assert_eq!(next, FINGERPRINT_LEN);
        for index in 0..FINGERPRINT_LEN {
            let section = Section::of(index).unwrap();
            assert!(section.range().contains(&index));
        }
        assert_eq!(Section::of(FINGERPRINT_LEN), None);
        assert_eq!(Section::of(usize::MAX), None);
    }

    #[test]
    fn public_tables_are_the_internal_ones() {
        for byte in 0..=255u8 {
            assert_eq!(lexical_symbol(byte), LEX[usize::from(byte)]);
            assert_eq!(shape_class(byte), SHAPE[usize::from(byte)]);
            assert!(usize::from(shape_class(byte)) < SHAPE_CLASS_COUNT);
        }
        for count in 0..40 {
            assert_eq!(
                feature_value(count),
                COMPRESS[count.min(usize::from(COUNT_CAP))]
            );
        }
        assert_eq!(feature_value(usize::MAX), 255);
        assert_eq!(feature_value(MAX_COUNT), 255);
    }

    #[test]
    fn window_ranges_cover_the_input_in_order() {
        for len in [
            0, 1, 2, 3, 255, 256, 257, 511, 512, 513, 767, 768, 769, 1025, 4096,
        ] {
            let count = window_count(len);
            for index in 0..count {
                let range = window_range(len, index).unwrap();
                assert_eq!(range.start, index * BLOCK_LEN, "length {len}");
                assert!(range.end <= len && range.len() <= 2 * BLOCK_LEN);
                if index + 1 == count {
                    assert_eq!(range.end, len);
                }
            }
            assert_eq!(window_range(len, count), None);
        }
        assert_eq!(window_range(usize::MAX, usize::MAX), None);
        let last = window_count(usize::MAX) - 1;
        assert_eq!(window_range(usize::MAX, last).unwrap().end, usize::MAX);
    }

    /// Rebuilds the fingerprint of every window from the n-grams.
    fn from_ngrams(input: &[u8]) -> Vec<Fingerprint> {
        let windows = window_count(input.len());
        let blocks = input.len().div_ceil(BLOCK_LEN);
        let mut histograms = vec![[0usize; FINGERPRINT_LEN]; blocks];
        for ngram in ngrams(input) {
            let range = ngram.range();
            assert_eq!(range.len(), ngram.section().order());
            assert!(range.end <= input.len());
            assert_eq!(ngram.block(), (range.end - 1) / BLOCK_LEN);
            assert_eq!(Section::of(ngram.feature()), Some(ngram.section()));
            histograms[ngram.block()][ngram.feature()] += 1;
        }
        (0..windows)
            .map(|index| {
                let mut features = [0u8; FINGERPRINT_LEN];
                let last = (index + 1).min(blocks - 1);
                for (feature, value) in features.iter_mut().enumerate() {
                    let count = (index..=last).map(|block| histograms[block][feature]).sum();
                    *value = feature_value(count);
                }
                Fingerprint::from_array(features)
            })
            .collect()
    }

    #[test]
    fn ngrams_rebuild_every_window() {
        let mut rng = Rng::new(17);
        for len in [0, 1, 2, 3, 7, 255, 256, 257, 512, 513, 800, 1300] {
            for style in 0..2 {
                let mut input = vec![0u8; len];
                rng.fill(&mut input);
                if style == 1 {
                    input
                        .iter_mut()
                        .for_each(|byte| *byte = b"ab1./%'<"[usize::from(*byte % 8)]);
                }
                let expected: Vec<Fingerprint> = fingerprints(&input).collect();
                assert_eq!(from_ngrams(&input), expected, "length {len}, style {style}");
            }
        }
    }

    #[test]
    fn ngrams_follow_the_documented_order() {
        let found: Vec<(core::ops::Range<usize>, Section)> = ngrams(b"a=1")
            .map(|ngram| (ngram.range(), ngram.section()))
            .collect();
        assert_eq!(
            found,
            [
                (0..2, Section::LexicalBigram),
                (0..2, Section::ShapeBigram),
                (1..3, Section::LexicalBigram),
                (0..3, Section::LexicalTrigram),
                (1..3, Section::ShapeBigram),
                (0..3, Section::ShapeTrigram),
            ]
        );
        assert_eq!(ngrams(b"").count(), 0);
        assert_eq!(ngrams(b"x").count(), 0);
        let mut iter = ngrams(b"abc");
        assert_eq!(iter.size_hint(), (0, Some(12)));
        assert_eq!(format!("{iter:?}"), "NGrams { position: 0, .. }");
        assert_eq!(iter.by_ref().count(), 4);
        assert!(iter.next().is_none());
    }

    #[test]
    fn fingerprint_views_agree() {
        let fingerprint = single(b"<img src=x onerror=alert(1)>");
        assert_eq!(
            Fingerprint::from_slice(fingerprint.as_ref()),
            Some(fingerprint.clone())
        );
        assert_eq!(Fingerprint::from_slice(&[0; FINGERPRINT_LEN - 1]), None);
        assert_eq!(Fingerprint::from(*fingerprint.as_array()), fingerprint);
        let total: usize = Section::ALL
            .iter()
            .map(|&section| fingerprint.section(section).len())
            .sum();
        assert_eq!(total, FINGERPRINT_LEN);
        let nonzero: Vec<(usize, u8)> = fingerprint.nonzero().collect();
        assert!(!nonzero.is_empty());
        assert!(
            nonzero
                .iter()
                .all(|&(index, value)| { value != 0 && fingerprint.as_array()[index] == value })
        );
        let zeros = fingerprint.as_array().iter().filter(|&&value| value == 0);
        assert_eq!(nonzero.len() + zeros.count(), FINGERPRINT_LEN);
        assert_eq!(Fingerprint::ZERO.nonzero().count(), 0);
    }

    /// Fingerprints `input` in the parts that `cuts` gives.
    fn in_parts(
        fingerprinter: &mut Fingerprinter,
        input: &[u8],
        cuts: &[usize],
    ) -> Vec<Fingerprint> {
        let mut windows = Vec::new();
        let mut start = 0;
        for &cut in cuts.iter().chain([&input.len()]) {
            let end = cut.clamp(start, input.len());
            let before = windows.len();
            fingerprinter.update(&input[start..end], |window| windows.push(window.clone()));
            assert!(windows.len() - before <= (end - start) / BLOCK_LEN + 1);
            assert_eq!(fingerprinter.windows(), windows.len() as u64);
            start = end;
        }
        assert_eq!(fingerprinter.len(), input.len() as u64);
        windows.extend(fingerprinter.finish().cloned());
        assert!(fingerprinter.is_empty());
        assert_eq!(fingerprinter.windows(), 0);
        windows
    }

    #[test]
    fn fingerprinter_matches_fingerprints_however_the_input_is_split() {
        let mut rng = Rng::new(23);
        let mut fingerprinter = Fingerprinter::new();
        let lengths = [
            0, 1, 2, 3, 100, 255, 256, 257, 511, 512, 513, 768, 769, 1024, 1500, 3000,
        ];
        for len in lengths {
            let mut input = vec![0u8; len];
            rng.fill(&mut input);
            input
                .iter_mut()
                .for_each(|byte| *byte = b"ab1./%'<\xC3"[usize::from(*byte % 9)]);
            let expected: Vec<Fingerprint> = fingerprints(&input).collect();
            // Whole, byte by byte, at every block edge, and at random cuts.
            let whole: [usize; 0] = [];
            let bytes: Vec<usize> = (1..len).collect();
            let edges: Vec<usize> = (1..=len / BLOCK_LEN).map(|n| n * BLOCK_LEN).collect();
            for cuts in [&whole[..], &bytes, &edges] {
                assert_eq!(
                    in_parts(&mut fingerprinter, &input, cuts),
                    expected,
                    "length {len}"
                );
            }
            for _ in 0..20 {
                let mut cuts: Vec<usize> = (0..rng.next_u32() % 8)
                    .map(|_| rng.next_u32() as usize % (len + 1))
                    .collect();
                cuts.sort_unstable();
                assert_eq!(
                    in_parts(&mut fingerprinter, &input, &cuts),
                    expected,
                    "length {len}, cuts {cuts:?}"
                );
            }
        }
    }

    #[test]
    fn fingerprinter_ignores_empty_parts_and_restarts_mid_block() {
        let mut rng = Rng::new(29);
        let mut input = vec![0u8; 900];
        rng.fill(&mut input);
        let expected: Vec<Fingerprint> = fingerprints(&input).collect();
        let mut fingerprinter = Fingerprinter::new();
        // An abandoned input that stops inside a block, then the real one
        // with empty parts between its parts.
        fingerprinter.update(&input[..300], |_| {});
        fingerprinter.reset();
        let mut windows = Vec::new();
        for part in input.chunks(100) {
            fingerprinter.update(&[], |_| unreachable!("an empty part completes no window"));
            fingerprinter.update(part, |window| windows.push(window.clone()));
        }
        windows.extend(fingerprinter.finish().cloned());
        assert_eq!(windows, expected);
        // An input finished inside a block leaves nothing behind.
        fingerprinter.update(&input[..700], |_| {});
        assert!(fingerprinter.finish().is_some());
        let mut again = Vec::new();
        fingerprinter.update(&input, |window| again.push(window.clone()));
        again.extend(fingerprinter.finish().cloned());
        assert_eq!(again, expected);
    }

    #[test]
    fn fingerprinter_reset_discards_the_input() {
        let mut fingerprinter = Fingerprinter::default();
        fingerprinter.update(&[b'%'; 700], |_| {});
        assert_eq!((fingerprinter.len(), fingerprinter.windows()), (700, 1));
        assert_eq!(
            format!("{fingerprinter:?}"),
            "Fingerprinter { len: 700, windows: 1, .. }"
        );
        fingerprinter.reset();
        assert!(fingerprinter.is_empty());
        let input = b"../../etc/passwd";
        fingerprinter.update(input, |_| unreachable!("one block completes no window"));
        assert_eq!(fingerprinter.finish(), fingerprints(input).next().as_ref());
        assert_eq!(fingerprinter.finish(), None);
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
