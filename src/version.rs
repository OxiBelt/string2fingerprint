//! Semantic versions and version requirements.
//!
//! A model file records two versions, so that a model is never used with a
//! fingerprint or a network that it was not trained for:
//!
//! - the [`LAYOUT_VERSION`](crate::fingerprint::LAYOUT_VERSION) of the
//!   fingerprint it was trained on, a [`Version`], and
//! - the versions of this crate that may load it, a [`VersionReq`] that
//!   [`VERSION`](crate::VERSION) must match.
//!
//! A [`Version`] has the form of [Semantic Versioning 2.0.0]: three numbers,
//! an optional pre-release, and optional build metadata, such as
//! `1.4.0-rc.1+build.5`. Versions are ordered by the precedence that the
//! specification defines.
//!
//! A [`VersionReq`] has the matching rules of version requirements in Cargo,
//! and a subset of their syntax: comparators separated by commas, which a
//! version must all match. A comparator is an operator (`=`, `>`, `>=`, `<`, `<=`,
//! `~`, or `^`) and a version whose minor and patch numbers may be left
//! out, a version without an operator, which means `^`, or a wildcard such
//! as `1.*`. `*` alone matches every version without a pre-release. A version
//! with a pre-release matches only when a comparator names the same major,
//! minor, and patch numbers with a pre-release of its own, so that
//! `>=1.0.0, <2.0.0` matches neither `1.5.0-rc.1` nor `2.0.0-rc.1`.
//!
//! Cargo also accepts a few spellings that it rewrites into those above:
//! `x` and `X` for `*`, an operator before a wildcard, `1.*.*`, and build
//! metadata in a comparator. They are refused here, so that every
//! requirement has one spelling that Cargo reads the same way. Like Cargo,
//! a requirement has at most [`MAX_COMPARATORS`] comparators.
//!
//! ```
//! use string2fingerprint::version::{Version, VersionReq};
//!
//! let requirement: VersionReq = ">=0.3, <0.5".parse()?;
//! assert!(requirement.matches(&"0.4.2".parse()?));
//! assert!(!requirement.matches(&"0.5.0".parse()?));
//! assert!(!requirement.matches(&"0.4.3-rc.1".parse()?));
//!
//! let caret = VersionReq::caret(&Version::new(0, 3, 1));
//! assert_eq!(caret.to_string(), "^0.3.1");
//! assert!(caret.matches(&Version::new(0, 3, 9)));
//! assert!(!caret.matches(&Version::new(0, 4, 0)));
//! # Ok::<(), string2fingerprint::version::VersionError>(())
//! ```
//!
//! Both types have one canonical text, which [`Display`](fmt::Display)
//! writes and which a model file stores. Parsing accepts spaces (`' '`,
//! not other whitespace) around operators and commas and a version without an operator, and writes them
//! in the canonical form: operators without a space, `^` written out, and
//! comparators separated by `, `.
//!
//! [Semantic Versioning 2.0.0]: https://semver.org/spec/v2.0.0.html

use core::cmp::Ordering;
use core::fmt;
use core::str::FromStr;

/// The longest text of a [`Version`], in bytes. It also bounds the
/// pre-release of each comparator of a [`VersionReq`].
pub const MAX_VERSION_LEN: usize = 64;

/// The longest canonical text of a [`VersionReq`], in bytes.
pub const MAX_REQUIREMENT_LEN: usize = 255;

/// The largest number of comparators in a [`VersionReq`], the limit of
/// Cargo.
pub const MAX_COMPARATORS: usize = 32;

// A model file stores the length of either text in one byte.
const _: () = assert!(MAX_VERSION_LEN <= u8::MAX as usize);
const _: () = assert!(MAX_REQUIREMENT_LEN <= u8::MAX as usize);

/// The reason a text is not a valid [`Version`] or [`VersionReq`].
///
/// The error never contains the text.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum VersionError {
    /// The text is empty or only spaces.
    Empty,
    /// A version is longer than [`MAX_VERSION_LEN`], or a requirement has
    /// more than [`MAX_COMPARATORS`] comparators, a canonical text longer
    /// than [`MAX_REQUIREMENT_LEN`], or a text longer than four times
    /// that.
    TooLong,
    /// A number does not fit in 64 bits.
    Overflow,
    /// The text does not follow the grammar, for example because a number
    /// has a leading zero, an identifier is empty, or a character is out of
    /// place.
    Syntax,
}

impl fmt::Display for VersionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Empty => "version is empty",
            Self::TooLong => "version is too long",
            Self::Overflow => "version number does not fit in 64 bits",
            Self::Syntax => "version is not valid semantic versioning syntax",
        })
    }
}

impl core::error::Error for VersionError {}

/// A version in the form of Semantic Versioning 2.0.0.
///
/// Comparison follows the precedence of the specification: the major,
/// minor, and patch numbers, then the pre-release, where a version without
/// a pre-release is the highest. Build metadata does not change precedence,
/// and [`Ord`] compares it as bytes only to order versions of equal
/// precedence, so that `Ord` agrees with `Eq`. [`Version::cmp_precedence`]
/// ignores it.
///
/// A version holds its text in place, so it is `Copy` and a constant can
/// hold one:
///
/// ```
/// use string2fingerprint::version::Version;
///
/// const RELEASE: Version = match Version::parse("2.1.0-rc.1+linux") {
///     Ok(version) => version,
///     Err(_) => panic!("the version is valid"),
/// };
/// assert_eq!((RELEASE.major(), RELEASE.minor(), RELEASE.patch()), (2, 1, 0));
/// assert_eq!((RELEASE.pre(), RELEASE.build()), ("rc.1", "linux"));
/// assert!(RELEASE < Version::new(2, 1, 0));
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Version {
    major: u64,
    minor: u64,
    patch: u64,
    /// The canonical text, followed by zero bytes.
    text: [u8; MAX_VERSION_LEN],
    len: u8,
    /// The pre-release is `text[pre_start..pre_end]`, empty when the two are
    /// equal. Build metadata follows a `+` at `pre_end`, when `pre_end` is
    /// below `len`.
    pre_start: u8,
    pre_end: u8,
}

impl Version {
    /// Creates the version `major.minor.patch`, without a pre-release or
    /// build metadata.
    #[must_use]
    pub const fn new(major: u64, minor: u64, patch: u64) -> Self {
        let mut text = [0u8; MAX_VERSION_LEN];
        // Three numbers of at most 20 digits and two dots fit in 64 bytes.
        let mut len = write_number(&mut text, 0, major);
        text[len] = b'.';
        len = write_number(&mut text, len + 1, minor);
        text[len] = b'.';
        len = write_number(&mut text, len + 1, patch);
        Self {
            major,
            minor,
            patch,
            text,
            len: len as u8,
            pre_start: len as u8,
            pre_end: len as u8,
        }
    }

    /// Parses a version in the form of Semantic Versioning 2.0.0.
    ///
    /// The text must be the whole version, without spaces, a leading `v`,
    /// or leading zeros in a number. It can be at most [`MAX_VERSION_LEN`]
    /// bytes long.
    ///
    /// # Errors
    ///
    /// Returns the [`VersionError`] that describes the first problem.
    pub const fn parse(text: &str) -> Result<Self, VersionError> {
        let bytes = text.as_bytes();
        if bytes.is_empty() {
            return Err(VersionError::Empty);
        }
        if bytes.len() > MAX_VERSION_LEN {
            return Err(VersionError::TooLong);
        }
        let mut at = 0;
        let major = match number(bytes, &mut at) {
            Ok(value) => value,
            Err(error) => return Err(error),
        };
        if !expect(bytes, &mut at, b'.') {
            return Err(VersionError::Syntax);
        }
        let minor = match number(bytes, &mut at) {
            Ok(value) => value,
            Err(error) => return Err(error),
        };
        if !expect(bytes, &mut at, b'.') {
            return Err(VersionError::Syntax);
        }
        let patch = match number(bytes, &mut at) {
            Ok(value) => value,
            Err(error) => return Err(error),
        };
        let mut pre_start = at;
        if expect(bytes, &mut at, b'-') {
            pre_start = at;
            if !identifiers(bytes, &mut at, true) {
                return Err(VersionError::Syntax);
            }
        }
        let pre_end = at;
        if expect(bytes, &mut at, b'+') && !identifiers(bytes, &mut at, false) {
            return Err(VersionError::Syntax);
        }
        if at != bytes.len() {
            return Err(VersionError::Syntax);
        }
        let mut text = [0u8; MAX_VERSION_LEN];
        let mut index = 0;
        while index < bytes.len() {
            text[index] = bytes[index];
            index += 1;
        }
        Ok(Self {
            major,
            minor,
            patch,
            text,
            len: bytes.len() as u8,
            pre_start: pre_start as u8,
            pre_end: pre_end as u8,
        })
    }

    /// Returns the major number.
    #[must_use]
    pub const fn major(&self) -> u64 {
        self.major
    }

    /// Returns the minor number.
    #[must_use]
    pub const fn minor(&self) -> u64 {
        self.minor
    }

    /// Returns the patch number.
    #[must_use]
    pub const fn patch(&self) -> u64 {
        self.patch
    }

    /// Returns the pre-release without its `-`, or the empty string when
    /// the version has none.
    #[must_use]
    pub fn pre(&self) -> &str {
        self.slice(usize::from(self.pre_start), usize::from(self.pre_end))
    }

    /// Returns the build metadata without its `+`, or the empty string when
    /// the version has none.
    #[must_use]
    pub fn build(&self) -> &str {
        let len = usize::from(self.len);
        let end = usize::from(self.pre_end);
        if end < len {
            self.slice(end + 1, len)
        } else {
            ""
        }
    }

    /// Returns whether the version has a pre-release.
    #[must_use]
    pub const fn is_prerelease(&self) -> bool {
        self.pre_start < self.pre_end
    }

    /// Returns the canonical text of the version, which
    /// [`Display`](fmt::Display) also writes.
    #[must_use]
    pub fn as_str(&self) -> &str {
        self.slice(0, usize::from(self.len))
    }

    /// Compares by the precedence of Semantic Versioning 2.0.0, which
    /// ignores build metadata.
    #[must_use]
    pub fn cmp_precedence(&self, other: &Self) -> Ordering {
        self.major
            .cmp(&other.major)
            .then(self.minor.cmp(&other.minor))
            .then(self.patch.cmp(&other.patch))
            .then_with(|| compare_pre(self.pre(), other.pre()))
    }

    fn slice(&self, start: usize, end: usize) -> &str {
        // Parsing accepts ASCII only, and `new` writes digits and dots.
        core::str::from_utf8(&self.text[start..end]).unwrap_or_default()
    }
}

impl PartialOrd for Version {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Version {
    fn cmp(&self, other: &Self) -> Ordering {
        self.cmp_precedence(other)
            .then_with(|| self.build().cmp(other.build()))
    }
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(self.as_str())
    }
}

impl fmt::Debug for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Version").field(&self.as_str()).finish()
    }
}

impl FromStr for Version {
    type Err = VersionError;

    fn from_str(text: &str) -> Result<Self, VersionError> {
        Self::parse(text)
    }
}

/// The operator of one comparator of a [`VersionReq`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Op {
    Exact,
    Greater,
    GreaterEq,
    Less,
    LessEq,
    Tilde,
    Caret,
    Wildcard,
}

impl Op {
    const fn prefix(self) -> &'static str {
        match self {
            Self::Exact => "=",
            Self::Greater => ">",
            Self::GreaterEq => ">=",
            Self::Less => "<",
            Self::LessEq => "<=",
            Self::Tilde => "~",
            Self::Caret => "^",
            Self::Wildcard => "",
        }
    }
}

/// One comparator of a [`VersionReq`]. Only a comparator with a patch
/// number can have a pre-release, and a wildcard has none.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Comparator {
    op: Op,
    major: u64,
    minor: Option<u64>,
    patch: Option<u64>,
    pre: [u8; MAX_VERSION_LEN],
    pre_len: u8,
}

impl Comparator {
    fn pre(&self) -> &str {
        // Parsing accepts ASCII only.
        core::str::from_utf8(&self.pre[..usize::from(self.pre_len)]).unwrap_or_default()
    }

    /// Matches `version` against the comparator alone, without the rule
    /// for pre-releases that [`VersionReq::matches`] adds.
    fn matches(&self, version: &Version) -> bool {
        match self.op {
            Op::Exact | Op::Wildcard => self.matches_exact(version),
            Op::Greater => self.matches_greater(version),
            Op::GreaterEq => self.matches_exact(version) || self.matches_greater(version),
            Op::Less => self.matches_less(version),
            Op::LessEq => self.matches_exact(version) || self.matches_less(version),
            Op::Tilde => self.matches_tilde(version),
            Op::Caret => self.matches_caret(version),
        }
    }

    fn matches_exact(&self, version: &Version) -> bool {
        version.major == self.major
            && self.minor.is_none_or(|minor| version.minor == minor)
            && self.patch.is_none_or(|patch| version.patch == patch)
            && version.pre() == self.pre()
    }

    fn matches_greater(&self, version: &Version) -> bool {
        if version.major != self.major {
            return version.major > self.major;
        }
        let Some(minor) = self.minor else {
            return false;
        };
        if version.minor != minor {
            return version.minor > minor;
        }
        let Some(patch) = self.patch else {
            return false;
        };
        if version.patch != patch {
            return version.patch > patch;
        }
        compare_pre(version.pre(), self.pre()) == Ordering::Greater
    }

    fn matches_less(&self, version: &Version) -> bool {
        if version.major != self.major {
            return version.major < self.major;
        }
        let Some(minor) = self.minor else {
            return false;
        };
        if version.minor != minor {
            return version.minor < minor;
        }
        let Some(patch) = self.patch else {
            return false;
        };
        if version.patch != patch {
            return version.patch < patch;
        }
        compare_pre(version.pre(), self.pre()) == Ordering::Less
    }

    fn matches_tilde(&self, version: &Version) -> bool {
        if version.major != self.major {
            return false;
        }
        if self.minor.is_some_and(|minor| version.minor != minor) {
            return false;
        }
        if let Some(patch) = self.patch
            && version.patch != patch
        {
            return version.patch > patch;
        }
        compare_pre(version.pre(), self.pre()) != Ordering::Less
    }

    fn matches_caret(&self, version: &Version) -> bool {
        if version.major != self.major {
            return false;
        }
        let Some(minor) = self.minor else {
            return true;
        };
        let Some(patch) = self.patch else {
            return if self.major > 0 {
                version.minor >= minor
            } else {
                version.minor == minor
            };
        };
        if self.major > 0 {
            if version.minor != minor {
                return version.minor > minor;
            }
            if version.patch != patch {
                return version.patch > patch;
            }
        } else if minor > 0 {
            if version.minor != minor {
                return false;
            }
            if version.patch != patch {
                return version.patch > patch;
            }
        } else if version.minor != minor || version.patch != patch {
            return false;
        }
        compare_pre(version.pre(), self.pre()) != Ordering::Less
    }

    /// Returns whether the comparator lets a pre-release of `version`
    /// match: it names the same major, minor, and patch numbers with a
    /// pre-release.
    fn admits_pre(&self, version: &Version) -> bool {
        self.major == version.major
            && self.minor == Some(version.minor)
            && self.patch == Some(version.patch)
            && self.pre_len > 0
    }
}

impl fmt::Display for Comparator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.op.prefix(), self.major)?;
        match self.minor {
            Some(minor) => write!(f, ".{minor}")?,
            None if self.op == Op::Wildcard => return f.write_str(".*"),
            None => return Ok(()),
        }
        match self.patch {
            Some(patch) => write!(f, ".{patch}")?,
            None if self.op == Op::Wildcard => return f.write_str(".*"),
            None => return Ok(()),
        }
        if self.pre_len > 0 {
            write!(f, "-{}", self.pre())?;
        }
        Ok(())
    }
}

/// A requirement on a [`Version`], in the syntax and with the matching
/// rules of Cargo.
///
/// The [module documentation](self) describes the syntax. A version
/// matches when it matches every comparator, and a version with a
/// pre-release matches only when a comparator also names its major, minor,
/// and patch numbers with a pre-release.
///
/// ```
/// use string2fingerprint::version::{Version, VersionReq};
///
/// let requirement: VersionReq = "~1.2".parse()?;
/// assert!(requirement.matches(&Version::new(1, 2, 7)));
/// assert!(!requirement.matches(&Version::new(1, 3, 0)));
///
/// let pre: VersionReq = ">=1.0.0-rc.1, <2".parse()?;
/// assert_eq!(pre.to_string(), ">=1.0.0-rc.1, <2");
/// assert!(pre.matches(&"1.0.0-rc.2".parse()?));
/// assert!(!pre.matches(&"1.1.0-rc.1".parse()?));
/// # Ok::<(), string2fingerprint::version::VersionError>(())
/// ```
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct VersionReq {
    /// Empty for `*`.
    comparators: Vec<Comparator>,
}

impl VersionReq {
    /// The requirement `*`, which every version without a pre-release
    /// matches.
    pub const STAR: Self = Self {
        comparators: Vec::new(),
    };

    /// Parses a requirement.
    ///
    /// # Errors
    ///
    /// Returns the [`VersionError`] that describes the first problem, and
    /// [`VersionError::TooLong`] when the text is longer than four times
    /// [`MAX_REQUIREMENT_LEN`] bytes, which bounds the work of parsing, or
    /// when the requirement has more than [`MAX_COMPARATORS`] comparators or
    /// a canonical text longer than [`MAX_REQUIREMENT_LEN`] bytes.
    pub fn parse(text: &str) -> Result<Self, VersionError> {
        let bytes = text.trim_matches(' ').as_bytes();
        if bytes.is_empty() {
            return Err(VersionError::Empty);
        }
        // Refused before parsing, so that the work does not grow with a
        // text far longer than any canonical one, whatever spaces it holds.
        if bytes.len() > 4 * MAX_REQUIREMENT_LEN {
            return Err(VersionError::TooLong);
        }
        if bytes == b"*" {
            return Ok(Self::STAR);
        }
        let mut comparators = Vec::new();
        let mut at = 0;
        loop {
            skip_spaces(bytes, &mut at);
            if comparators.len() == MAX_COMPARATORS {
                return Err(VersionError::TooLong);
            }
            comparators.push(comparator(bytes, &mut at)?);
            skip_spaces(bytes, &mut at);
            if at == bytes.len() {
                break;
            }
            if !expect(bytes, &mut at, b',') {
                return Err(VersionError::Syntax);
            }
        }
        let requirement = Self { comparators };
        if requirement.canonical_len() > MAX_REQUIREMENT_LEN {
            return Err(VersionError::TooLong);
        }
        Ok(requirement)
    }

    /// Returns the requirement `^version`, which matches the versions that
    /// Cargo treats as compatible with `version`: those not lower than
    /// `version` with the same numbers up to and including its leftmost
    /// nonzero one, or the same three numbers for `0.0.x`. A pre-release
    /// matches only when its numbers are those of `version` and `version`
    /// has a pre-release.
    ///
    /// Build metadata is left out, because a requirement cannot name it.
    #[must_use]
    pub fn caret(version: &Version) -> Self {
        let mut pre = [0u8; MAX_VERSION_LEN];
        let text = version.pre().as_bytes();
        pre[..text.len()].copy_from_slice(text);
        Self {
            comparators: vec![Comparator {
                op: Op::Caret,
                major: version.major,
                minor: Some(version.minor),
                patch: Some(version.patch),
                pre,
                // A pre-release is shorter than its version.
                pre_len: text.len() as u8,
            }],
        }
    }

    /// Returns whether `version` meets the requirement.
    #[must_use]
    pub fn matches(&self, version: &Version) -> bool {
        self.comparators
            .iter()
            .all(|comparator| comparator.matches(version))
            && (!version.is_prerelease()
                || self
                    .comparators
                    .iter()
                    .any(|comparator| comparator.admits_pre(version)))
    }

    /// Returns the number of comparators, 0 for `*`.
    #[must_use]
    pub fn len(&self) -> usize {
        self.comparators.len()
    }

    /// Returns whether the requirement is `*`.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.comparators.is_empty()
    }

    /// Writes the canonical text to `buffer` and returns it, without
    /// allocating.
    pub(crate) fn write_canonical<'a>(
        &self,
        buffer: &'a mut [u8; MAX_REQUIREMENT_LEN],
    ) -> &'a [u8] {
        struct Writer<'a> {
            buffer: &'a mut [u8; MAX_REQUIREMENT_LEN],
            len: usize,
        }
        impl fmt::Write for Writer<'_> {
            fn write_str(&mut self, text: &str) -> fmt::Result {
                let end = self.len + text.len();
                self.buffer
                    .get_mut(self.len..end)
                    .ok_or(fmt::Error)?
                    .copy_from_slice(text.as_bytes());
                self.len = end;
                Ok(())
            }
        }
        let mut writer = Writer { buffer, len: 0 };
        // Parsing and `caret` keep the canonical text within the buffer, so
        // the write cannot fail.
        let written = fmt::write(&mut writer, format_args!("{self}"));
        debug_assert!(written.is_ok(), "a canonical requirement fits its buffer");
        let len = writer.len;
        &buffer[..len]
    }

    fn canonical_len(&self) -> usize {
        struct Counter(usize);
        impl fmt::Write for Counter {
            fn write_str(&mut self, text: &str) -> fmt::Result {
                self.0 += text.len();
                Ok(())
            }
        }
        let mut counter = Counter(0);
        // Writing to a counter never fails.
        let _ = fmt::write(&mut counter, format_args!("{self}"));
        counter.0
    }
}

impl fmt::Display for VersionReq {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Some((first, rest)) = self.comparators.split_first() else {
            return f.write_str("*");
        };
        write!(f, "{first}")?;
        for comparator in rest {
            write!(f, ", {comparator}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for VersionReq {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("VersionReq")
            .field(&format_args!("{self}"))
            .finish()
    }
}

impl FromStr for VersionReq {
    type Err = VersionError;

    fn from_str(text: &str) -> Result<Self, VersionError> {
        Self::parse(text)
    }
}

impl From<&Version> for VersionReq {
    /// Returns [`VersionReq::caret`] of the version.
    fn from(version: &Version) -> Self {
        Self::caret(version)
    }
}

/// Parses one comparator at `at`, up to a space, a comma, or the end.
fn comparator(bytes: &[u8], at: &mut usize) -> Result<Comparator, VersionError> {
    let op = if expect(bytes, at, b'>') {
        if expect(bytes, at, b'=') {
            Some(Op::GreaterEq)
        } else {
            Some(Op::Greater)
        }
    } else if expect(bytes, at, b'<') {
        if expect(bytes, at, b'=') {
            Some(Op::LessEq)
        } else {
            Some(Op::Less)
        }
    } else if expect(bytes, at, b'=') {
        Some(Op::Exact)
    } else if expect(bytes, at, b'~') {
        Some(Op::Tilde)
    } else if expect(bytes, at, b'^') {
        Some(Op::Caret)
    } else {
        None
    };
    skip_spaces(bytes, at);
    let mut comparator = Comparator {
        op: op.unwrap_or(Op::Caret),
        major: number(bytes, at)?,
        minor: None,
        patch: None,
        pre: [0; MAX_VERSION_LEN],
        pre_len: 0,
    };
    // A wildcard stands for the minor or the patch number, and takes no
    // operator.
    let wildcard = |at: &mut usize| expect(bytes, at, b'*');
    if !expect(bytes, at, b'.') {
        return Ok(comparator);
    }
    if wildcard(at) {
        return wildcard_of(comparator, op);
    }
    comparator.minor = Some(number(bytes, at)?);
    if !expect(bytes, at, b'.') {
        return Ok(comparator);
    }
    if wildcard(at) {
        return wildcard_of(comparator, op);
    }
    comparator.patch = Some(number(bytes, at)?);
    if expect(bytes, at, b'-') {
        let start = *at;
        if !identifiers(bytes, at, true) {
            return Err(VersionError::Syntax);
        }
        let pre = &bytes[start..*at];
        if pre.len() > MAX_VERSION_LEN {
            return Err(VersionError::TooLong);
        }
        comparator.pre[..pre.len()].copy_from_slice(pre);
        comparator.pre_len = pre.len() as u8;
    }
    Ok(comparator)
}

fn wildcard_of(mut comparator: Comparator, op: Option<Op>) -> Result<Comparator, VersionError> {
    if op.is_some() {
        return Err(VersionError::Syntax);
    }
    comparator.op = Op::Wildcard;
    Ok(comparator)
}

fn skip_spaces(bytes: &[u8], at: &mut usize) {
    while *at < bytes.len() && bytes[*at] == b' ' {
        *at += 1;
    }
}

/// Consumes `byte` at `at` and returns whether it was there.
const fn expect(bytes: &[u8], at: &mut usize, byte: u8) -> bool {
    if *at < bytes.len() && bytes[*at] == byte {
        *at += 1;
        true
    } else {
        false
    }
}

/// Parses a number without a leading zero at `at`.
const fn number(bytes: &[u8], at: &mut usize) -> Result<u64, VersionError> {
    let start = *at;
    let mut value: u64 = 0;
    while *at < bytes.len() && bytes[*at].is_ascii_digit() {
        let digit = (bytes[*at] - b'0') as u64;
        value = match value.checked_mul(10) {
            Some(value) => match value.checked_add(digit) {
                Some(value) => value,
                None => return Err(VersionError::Overflow),
            },
            None => return Err(VersionError::Overflow),
        };
        *at += 1;
    }
    let len = *at - start;
    if len == 0 || (len > 1 && bytes[start] == b'0') {
        return Err(VersionError::Syntax);
    }
    Ok(value)
}

/// Consumes dot-separated identifiers of ASCII letters, digits, and `-` at
/// `at`, and returns whether each one was valid. In a pre-release
/// (`numeric_rule`), an identifier of digits has no leading zero.
const fn identifiers(bytes: &[u8], at: &mut usize, numeric_rule: bool) -> bool {
    loop {
        let start = *at;
        let mut digits = true;
        while *at < bytes.len() && (bytes[*at].is_ascii_alphanumeric() || bytes[*at] == b'-') {
            digits &= bytes[*at].is_ascii_digit();
            *at += 1;
        }
        let len = *at - start;
        if len == 0 || (numeric_rule && digits && len > 1 && bytes[start] == b'0') {
            return false;
        }
        if !expect(bytes, at, b'.') {
            return true;
        }
    }
}

/// Writes `value` in decimal to `text` at `at` and returns the end.
const fn write_number(text: &mut [u8; MAX_VERSION_LEN], at: usize, value: u64) -> usize {
    let mut digits = [0u8; 20];
    let mut count = 0;
    let mut rest = value;
    loop {
        digits[count] = b'0' + (rest % 10) as u8;
        count += 1;
        rest /= 10;
        if rest == 0 {
            break;
        }
    }
    let mut index = 0;
    while index < count {
        text[at + index] = digits[count - 1 - index];
        index += 1;
    }
    at + count
}

/// Compares two pre-releases by the precedence of Semantic Versioning
/// 2.0.0, where the empty pre-release is the highest.
fn compare_pre(left: &str, right: &str) -> Ordering {
    match (left.is_empty(), right.is_empty()) {
        (true, true) => return Ordering::Equal,
        (true, false) => return Ordering::Greater,
        (false, true) => return Ordering::Less,
        (false, false) => {}
    }
    let mut left = left.split('.');
    let mut right = right.split('.');
    loop {
        match (left.next(), right.next()) {
            (None, None) => return Ordering::Equal,
            (None, Some(_)) => return Ordering::Less,
            (Some(_), None) => return Ordering::Greater,
            (Some(left), Some(right)) => {
                let order = compare_identifier(left, right);
                if order != Ordering::Equal {
                    return order;
                }
            }
        }
    }
}

/// Compares two pre-release identifiers: numeric ones by value, below every
/// alphanumeric one, and alphanumeric ones in ASCII order.
fn compare_identifier(left: &str, right: &str) -> Ordering {
    let numeric = |text: &str| text.bytes().all(|byte| byte.is_ascii_digit());
    match (numeric(left), numeric(right)) {
        // Numeric identifiers of a pre-release have no leading zero, so the
        // longer one is the larger.
        (true, true) => left.len().cmp(&right.len()).then(left.cmp(right)),
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        (false, false) => left.cmp(right),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn version(text: &str) -> Version {
        text.parse()
            .unwrap_or_else(|error| panic!("{text:?}: {error}"))
    }

    fn requirement(text: &str) -> VersionReq {
        text.parse()
            .unwrap_or_else(|error| panic!("{text:?}: {error}"))
    }

    #[test]
    fn parses_versions_and_keeps_their_text() {
        for text in [
            "0.0.0",
            "1.2.3",
            "10.20.30",
            "1.0.0-alpha",
            "1.0.0-alpha.1",
            "1.0.0-0.3.7",
            "1.0.0-x.7.z.92",
            "1.0.0-x-y-z.--",
            "1.0.0-alpha+001",
            "1.0.0+20130313144700",
            "1.0.0-beta+exp.sha.5114f85",
            "1.0.0+21AF26D3----117B344092BD",
            "18446744073709551615.18446744073709551615.18446744073709551615",
        ] {
            let parsed = version(text);
            assert_eq!(parsed.to_string(), text);
            assert_eq!(parsed.as_str(), text);
        }
        let parsed = version("1.2.3-rc.1+build.7");
        assert_eq!((parsed.major(), parsed.minor(), parsed.patch()), (1, 2, 3));
        assert_eq!((parsed.pre(), parsed.build()), ("rc.1", "build.7"));
        assert!(parsed.is_prerelease());
        let plain = version("1.2.3+build");
        assert_eq!((plain.pre(), plain.build()), ("", "build"));
        assert!(!plain.is_prerelease());
    }

    #[test]
    fn rejects_invalid_versions() {
        for (text, error) in [
            ("", VersionError::Empty),
            ("1", VersionError::Syntax),
            ("1.2", VersionError::Syntax),
            ("1.2.3.4", VersionError::Syntax),
            ("01.2.3", VersionError::Syntax),
            ("1.02.3", VersionError::Syntax),
            ("1.2.03", VersionError::Syntax),
            ("v1.2.3", VersionError::Syntax),
            (" 1.2.3", VersionError::Syntax),
            ("1.2.3 ", VersionError::Syntax),
            ("1.2.3-", VersionError::Syntax),
            ("1.2.3-01", VersionError::Syntax),
            ("1.2.3-a..b", VersionError::Syntax),
            ("1.2.3-a_b", VersionError::Syntax),
            ("1.2.3+", VersionError::Syntax),
            ("1.2.3+a+b", VersionError::Syntax),
            ("1.2.3-é", VersionError::Syntax),
            ("1.-2.3", VersionError::Syntax),
            ("18446744073709551616.0.0", VersionError::Overflow),
        ] {
            assert_eq!(Version::parse(text), Err(error), "{text:?}");
        }
        // Leading zeros are allowed in build metadata, not in a pre-release.
        assert!(Version::parse("1.2.3+01").is_ok());
        assert!(Version::parse("1.2.3-0a").is_ok());
        let long = format!("1.2.3-{}", "a".repeat(MAX_VERSION_LEN - 6));
        assert_eq!(long.len(), MAX_VERSION_LEN);
        assert!(Version::parse(&long).is_ok());
        assert_eq!(
            Version::parse(&format!("{long}a")),
            Err(VersionError::TooLong)
        );
    }

    #[test]
    fn new_writes_the_numbers() {
        assert_eq!(Version::new(0, 0, 0), version("0.0.0"));
        assert_eq!(Version::new(1, 20, 300), version("1.20.300"));
        assert_eq!(
            Version::new(u64::MAX, u64::MAX, u64::MAX).to_string(),
            "18446744073709551615.18446744073709551615.18446744073709551615"
        );
    }

    #[test]
    fn orders_by_precedence() {
        // The order that Semantic Versioning 2.0.0 gives as an example.
        let ordered = [
            "1.0.0-alpha",
            "1.0.0-alpha.1",
            "1.0.0-alpha.beta",
            "1.0.0-beta",
            "1.0.0-beta.2",
            "1.0.0-beta.11",
            "1.0.0-rc.1",
            "1.0.0",
            "1.0.1",
            "1.1.0",
            "2.0.0",
            "10.0.0",
        ];
        for (index, left) in ordered.iter().enumerate() {
            for (other, right) in ordered.iter().enumerate() {
                assert_eq!(
                    version(left).cmp(&version(right)),
                    index.cmp(&other),
                    "{left} {right}"
                );
            }
        }
        let built = version("1.0.0+b");
        assert_eq!(built.cmp_precedence(&version("1.0.0+a")), Ordering::Equal);
        assert_eq!(built.cmp(&version("1.0.0+a")), Ordering::Greater);
        assert_ne!(built, version("1.0.0+a"));
        assert!(version("1.0.0-2") < version("1.0.0-10"));
        assert!(version("1.0.0-99999999999999999999") < version("1.0.0-a"));
    }

    #[test]
    fn versions_are_constants() {
        const PARSED: Version = match Version::parse("3.1.4-pi") {
            Ok(version) => version,
            Err(_) => panic!("valid"),
        };
        const NEW: Version = Version::new(3, 1, 4);
        assert!(PARSED < NEW);
        assert_eq!(format!("{PARSED:?}"), "Version(\"3.1.4-pi\")");
        assert_eq!(format!("{NEW:>7}"), "  3.1.4");
    }

    #[test]
    fn writes_requirements_in_canonical_form() {
        for (text, canonical) in [
            ("*", "*"),
            (" * ", "*"),
            ("1", "^1"),
            ("1.2", "^1.2"),
            ("1.2.3", "^1.2.3"),
            ("^1.2.3", "^1.2.3"),
            ("= 1.2.3-rc.1", "=1.2.3-rc.1"),
            (">= 0.3 , < 0.5", ">=0.3, <0.5"),
            (">=0.3,<0.5", ">=0.3, <0.5"),
            ("~1", "~1"),
            ("<= 2.0.0", "<=2.0.0"),
            (">1.0.0-0", ">1.0.0-0"),
            ("1.*", "1.*"),
            ("1.2.*", "1.2.*"),
        ] {
            let parsed = requirement(text);
            assert_eq!(parsed.to_string(), canonical, "{text:?}");
            assert_eq!(requirement(canonical), parsed, "{text:?}");
        }
        assert_eq!(requirement("*"), VersionReq::STAR);
        assert!(VersionReq::STAR.is_empty());
        assert_eq!(requirement(">=1, <2").len(), 2);
        assert_eq!(
            format!("{:?}", requirement(">=1, <2")),
            "VersionReq(>=1, <2)"
        );
    }

    #[test]
    fn rejects_invalid_requirements() {
        for (text, error) in [
            ("", VersionError::Empty),
            ("  ", VersionError::Empty),
            (",", VersionError::Syntax),
            ("1,", VersionError::Syntax),
            (",1", VersionError::Syntax),
            ("1 2", VersionError::Syntax),
            (">=1 <2", VersionError::Syntax),
            ("1||2", VersionError::Syntax),
            ("=>1", VersionError::Syntax),
            ("v1", VersionError::Syntax),
            ("01", VersionError::Syntax),
            ("1.2.3.4", VersionError::Syntax),
            ("1.2-rc", VersionError::Syntax),
            ("1.2.3+build", VersionError::Syntax),
            ("1.2.3-", VersionError::Syntax),
            ("1.2.3-01", VersionError::Syntax),
            (">=1.*", VersionError::Syntax),
            ("^1.2.*", VersionError::Syntax),
            ("1.*.3", VersionError::Syntax),
            ("*.1", VersionError::Syntax),
            ("*, 1", VersionError::Syntax),
            ("1, *", VersionError::Syntax),
            ("x", VersionError::Syntax),
            ("99999999999999999999", VersionError::Overflow),
        ] {
            assert_eq!(VersionReq::parse(text), Err(error), "{text:?}");
        }
        let most = vec!["^1"; MAX_COMPARATORS].join(", ");
        assert_eq!(requirement(&most).to_string(), most);
        assert_eq!(requirement(&most).len(), MAX_COMPARATORS);
        let many = vec!["1"; MAX_COMPARATORS + 1].join(",");
        assert_eq!(VersionReq::parse(&many), Err(VersionError::TooLong));
        // The canonical text writes `^`, a separator, and longer numbers.
        let wide = ["10000000"; 24].join(",");
        assert!(wide.len() <= MAX_REQUIREMENT_LEN);
        assert_eq!(VersionReq::parse(&wide), Err(VersionError::TooLong));
        let fits = ["^10000000"; 23].join(", ");
        assert!(fits.len() <= MAX_REQUIREMENT_LEN);
        assert_eq!(requirement(&fits).to_string(), fits);
        // Only spaces separate tokens, as in Cargo.
        for text in ["1\t,2", "\t*", "<2.0\n", "^\n3", "\r1"] {
            assert_eq!(
                VersionReq::parse(text),
                Err(VersionError::Syntax),
                "{text:?}"
            );
        }
        assert_eq!(
            VersionReq::parse(&format!(">={}1", " ".repeat(4 * MAX_REQUIREMENT_LEN))),
            Err(VersionError::TooLong)
        );
        let long_pre = format!("=1.0.0-{}", "a".repeat(MAX_VERSION_LEN + 1));
        assert_eq!(VersionReq::parse(&long_pre), Err(VersionError::TooLong));
        assert_eq!(
            VersionReq::parse(&"1,".repeat(MAX_REQUIREMENT_LEN * 3)),
            Err(VersionError::TooLong)
        );
    }

    /// The cases of the `semver` crate that Cargo uses, by operator.
    #[test]
    fn matches_like_cargo() {
        let cases: &[(&str, &[&str], &[&str])] = &[
            ("=1.0.0", &["1.0.0"], &["1.0.1", "0.9.9", "1.0.0-pre"]),
            ("=0.9", &["0.9.0", "0.9.5"], &["0.10.0", "0.9.5-pre"]),
            ("=1", &["1.0.0", "1.9.9"], &["2.0.0", "0.9.9"]),
            (">=1.0.0", &["1.0.0", "2.0.0"], &["0.1.0", "1.0.0-pre.2"]),
            (
                ">=2.1.0-alpha",
                &["2.1.0-alpha", "2.1.0-beta", "2.1.0", "3.0.0"],
                &["2.0.0", "2.1.1-alpha"],
            ),
            (">1.0.0", &["1.0.1", "2.0.0"], &["1.0.0", "1.0.1-pre"]),
            (">1", &["2.0.0"], &["1.9.9"]),
            (">1.2", &["1.3.0"], &["1.2.9"]),
            (
                "<1.0.0",
                &["0.1.0", "0.0.1"],
                &["1.0.0", "1.0.0-beta", "2.0.0", "0.9.0-alpha"],
            ),
            (
                "<=2.1.0-alpha2",
                &["2.1.0-alpha2", "2.1.0-alpha1", "2.0.0", "1.0.0"],
                &["2.1.0", "2.2.0-alpha1", "2.0.0-alpha2", "1.0.0-alpha2"],
            ),
            ("<1", &["0.9.9"], &["1.0.0"]),
            ("~1", &["1.0.0", "1.9.9"], &["2.0.0", "0.9.9"]),
            ("~1.2", &["1.2.0", "1.2.9"], &["1.3.0", "1.1.9"]),
            (
                "~1.2.2",
                &["1.2.2", "1.2.4"],
                &["1.2.1", "1.3.0", "1.2.3-beta"],
            ),
            (
                "~1.2.3-beta.2",
                &["1.2.3", "1.2.4", "1.2.3-beta.2", "1.2.3-beta.4"],
                &["1.3.0", "1.2.3-beta.1", "1.2.4-beta.2"],
            ),
            (
                "^1",
                &["1.1.2", "1.1.0", "1.2.1", "1.0.1"],
                &["0.9.1", "2.9.0", "0.1.4", "1.0.0-beta1"],
            ),
            (
                "^1.1",
                &["1.1.2", "1.1.0", "1.2.1"],
                &["0.9.1", "2.9.0", "1.0.1", "0.1.4"],
            ),
            (
                "^1.1.2",
                &["1.1.2", "1.1.4", "1.2.1"],
                &[
                    "0.9.1",
                    "2.9.0",
                    "1.1.1",
                    "0.0.1",
                    "1.1.2-alpha1",
                    "1.1.3-alpha1",
                    "2.9.0-alpha1",
                ],
            ),
            (
                "^0.1.2",
                &["0.1.2", "0.1.4"],
                &[
                    "0.9.1",
                    "2.9.0",
                    "1.1.1",
                    "0.0.1",
                    "0.1.2-beta",
                    "0.1.3-alpha",
                    "0.2.0-pre",
                ],
            ),
            (
                "^0.5.1-alpha3",
                &[
                    "0.5.1-alpha3",
                    "0.5.1-alpha4",
                    "0.5.1-beta",
                    "0.5.1",
                    "0.5.5",
                ],
                &[
                    "0.5.1-alpha1",
                    "0.5.2-alpha3",
                    "0.5.5-pre",
                    "0.5.0-pre",
                    "0.6.0",
                ],
            ),
            (
                "^0.0.2",
                &["0.0.2"],
                &["0.9.1", "2.9.0", "1.1.1", "0.0.1", "0.1.4"],
            ),
            (
                "^0.0",
                &["0.0.2", "0.0.0"],
                &["0.9.1", "2.9.0", "1.1.1", "0.1.4"],
            ),
            ("^0", &["0.9.1", "0.0.2", "0.0.0"], &["2.9.0", "1.1.1"]),
            (
                "^1.4.2-beta.5",
                &["1.4.2", "1.4.3", "1.4.2-beta.5", "1.4.2-beta.6", "1.4.2-c"],
                &[
                    "0.9.9",
                    "2.0.0",
                    "1.4.2-alpha",
                    "1.4.2-beta.4",
                    "1.4.3-beta.5",
                ],
            ),
            (
                "*",
                &["0.9.1", "2.9.0", "0.0.9", "1.0.1", "1.1.1"],
                &["1.0.0-pre"],
            ),
            (
                "1.*",
                &["1.2.0", "1.2.1", "1.1.1", "1.3.0"],
                &["0.0.9", "2.0.0", "1.2.0-pre"],
            ),
            (
                "1.2.*",
                &["1.2.0", "1.2.2", "1.2.4"],
                &["1.9.0", "1.0.9", "2.0.1", "0.1.3", "1.2.2-pre"],
            ),
            (
                ">=0.5.1-alpha3, <0.6",
                &[
                    "0.5.1-alpha3",
                    "0.5.1-alpha4",
                    "0.5.1-beta",
                    "0.5.1",
                    "0.5.5",
                ],
                &[
                    "0.5.1-alpha1",
                    "0.5.2-alpha3",
                    "0.5.5-pre",
                    "0.5.0-pre",
                    "0.6.0",
                    "0.6.0-pre",
                ],
            ),
            (
                ">=0.3, <0.5",
                &["0.3.0", "0.4.9"],
                &["0.2.9", "0.5.0", "0.5.0-dev", "0.4.0-dev"],
            ),
            (">=1.0.0, <1.0.0", &[], &["1.0.0"]),
        ];
        for &(text, accepted, refused) in cases {
            let parsed = requirement(text);
            for &candidate in accepted {
                assert!(parsed.matches(&version(candidate)), "{text} {candidate}");
            }
            for &candidate in refused {
                assert!(!parsed.matches(&version(candidate)), "{text} {candidate}");
            }
        }
        // Build metadata never changes a match.
        assert!(requirement("=1.0.0").matches(&version("1.0.0+build")));
    }

    #[test]
    fn caret_matches_its_version_and_compatible_ones() {
        for (text, accepted, refused) in [
            ("1.2.3", "1.9.0", "2.0.0"),
            ("0.2.3", "0.2.9", "0.3.0"),
            ("0.0.3", "0.0.3", "0.0.4"),
            ("0.1.0-dev", "0.1.0", "0.1.1-dev"),
            ("2.0.0-rc.1+build", "2.0.0-rc.2", "2.0.0-beta"),
        ] {
            let base = version(text);
            let caret = VersionReq::caret(&base);
            assert_eq!(caret, VersionReq::from(&base));
            assert!(caret.matches(&base), "{text}");
            assert!(caret.matches(&version(accepted)), "{text} {accepted}");
            assert!(!caret.matches(&version(refused)), "{text} {refused}");
            assert_eq!(requirement(&caret.to_string()), caret);
        }
        assert_eq!(
            VersionReq::caret(&version("0.1.0-dev+abc")).to_string(),
            "^0.1.0-dev"
        );
    }

    #[test]
    fn crate_and_layout_versions_are_valid() {
        assert_eq!(crate::VERSION.as_str(), env!("CARGO_PKG_VERSION"));
        assert_eq!(crate::fingerprint::LAYOUT_VERSION, version("1.0.0"));
        assert!(VersionReq::caret(&crate::VERSION).matches(&crate::VERSION));
    }

    #[test]
    fn errors_have_fixed_messages() {
        for error in [
            VersionError::Empty,
            VersionError::TooLong,
            VersionError::Overflow,
            VersionError::Syntax,
        ] {
            assert!(!error.to_string().is_empty());
        }
    }
}
