//! Sets of categories.

use core::fmt;
use core::ops::{BitAnd, BitOr, BitOrAssign, Sub};
use core::str::FromStr;

use crate::Category;

/// A set of [categories](Category).
///
/// A set names the categories that apply to an input: the labels of a
/// training or evaluation sample, or the categories whose score reached a
/// threshold. The empty set labels benign text.
///
/// ```
/// use string2fingerprint::{Category, CategorySet};
///
/// let labels = CategorySet::from(Category::SqlInjection) | Category::Obfuscation;
/// let flagged = CategorySet::from(Category::SqlInjection) | Category::Malformed;
///
/// // Flagged without the label: false positives. Labeled without a flag:
/// // false negatives.
/// assert_eq!(flagged - labels, Category::Malformed.into());
/// assert_eq!(labels - flagged, Category::Obfuscation.into());
/// ```
///
/// Bit `i` of [`CategorySet::bits`] is set when the category with
/// [`Category::index`] `i` is in the set. The bits are stable, and they are
/// the label byte of the corpus format of the training tools.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct CategorySet(u8);

const _: () = assert!(Category::COUNT == u8::BITS as usize);

impl CategorySet {
    /// The set without any category, which labels benign text.
    pub const EMPTY: Self = Self(0);

    /// The set of every category.
    pub const ALL: Self = Self(u8::MAX);

    /// Creates the set whose bit `i` says whether it holds the category
    /// with index `i`.
    #[must_use]
    pub const fn from_bits(bits: u8) -> Self {
        Self(bits)
    }

    /// Returns the set as one bit for each category, by index.
    #[must_use]
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// Creates the set that holds only `category`.
    #[must_use]
    pub const fn only(category: Category) -> Self {
        Self(1 << category.index())
    }

    /// Returns whether the set holds no category.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Returns the number of categories in the set.
    #[must_use]
    pub const fn len(self) -> usize {
        self.0.count_ones() as usize
    }

    /// Returns whether the set holds `category`.
    #[must_use]
    pub const fn contains(self, category: Category) -> bool {
        self.0 & (1 << category.index()) != 0
    }

    /// Returns the set with `category` added.
    #[must_use]
    pub const fn with(self, category: Category) -> Self {
        Self(self.0 | (1 << category.index()))
    }

    /// Returns the set with `category` removed.
    #[must_use]
    pub const fn without(self, category: Category) -> Self {
        Self(self.0 & !(1 << category.index()))
    }

    /// Returns the categories that are in either set.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Returns the categories that are in both sets.
    #[must_use]
    pub const fn intersection(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    /// Returns the categories of this set that are not in `other`.
    #[must_use]
    pub const fn difference(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }

    /// Iterates over the categories in the set, ordered by discriminant.
    pub fn iter(self) -> impl Iterator<Item = Category> {
        Category::ALL
            .into_iter()
            .filter(move |&category| self.contains(category))
    }
}

impl From<Category> for CategorySet {
    fn from(category: Category) -> Self {
        Self::only(category)
    }
}

impl FromIterator<Category> for CategorySet {
    fn from_iter<I: IntoIterator<Item = Category>>(categories: I) -> Self {
        categories.into_iter().fold(Self::EMPTY, Self::with)
    }
}

impl<T: Into<CategorySet>> BitOr<T> for CategorySet {
    type Output = Self;

    fn bitor(self, other: T) -> Self {
        self.union(other.into())
    }
}

impl<T: Into<CategorySet>> BitOrAssign<T> for CategorySet {
    fn bitor_assign(&mut self, other: T) {
        *self = self.union(other.into());
    }
}

impl<T: Into<CategorySet>> BitAnd<T> for CategorySet {
    type Output = Self;

    fn bitand(self, other: T) -> Self {
        self.intersection(other.into())
    }
}

impl<T: Into<CategorySet>> Sub<T> for CategorySet {
    type Output = Self;

    fn sub(self, other: T) -> Self {
        self.difference(other.into())
    }
}

/// Writes the names of the categories separated by commas, or `benign` for
/// the empty set. [`CategorySet::from_str`] reads the same form.
impl fmt::Display for CategorySet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            return f.write_str(BENIGN);
        }
        for (position, category) in self.iter().enumerate() {
            if position > 0 {
                f.write_str(",")?;
            }
            f.write_str(category.name())?;
        }
        Ok(())
    }
}

impl fmt::Debug for CategorySet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

/// The name of the empty set in text.
const BENIGN: &str = "benign";

/// The error of [`CategorySet::from_str`]: a name is not a category.
///
/// The error does not contain the text that was parsed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParseCategorySetError;

impl fmt::Display for ParseCategorySetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("text is not a list of category names")
    }
}

impl core::error::Error for ParseCategorySetError {}

/// Parses [category names](Category::name) separated by commas.
///
/// The empty string and `benign` are the empty set. Spaces and tabs around a
/// name are ignored, and a name can repeat.
///
/// ```
/// use string2fingerprint::{Category, CategorySet};
///
/// let labels: CategorySet = "sql_injection,obfuscation".parse().unwrap();
/// assert!(labels.contains(Category::SqlInjection));
/// assert_eq!(labels.len(), 2);
/// assert_eq!("benign".parse(), Ok(CategorySet::EMPTY));
/// assert!("sql".parse::<CategorySet>().is_err());
/// ```
impl FromStr for CategorySet {
    type Err = ParseCategorySetError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        fn trim(name: &str) -> &str {
            name.trim_matches([' ', '\t'])
        }
        let text = trim(text);
        if text.is_empty() || text == BENIGN {
            return Ok(Self::EMPTY);
        }
        text.split(',').try_fold(Self::EMPTY, |set, name| {
            Category::from_name(trim(name))
                .map(|category| set.with(category))
                .ok_or(ParseCategorySetError)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_and_all_bound_every_set() {
        assert!(CategorySet::EMPTY.is_empty());
        assert_eq!(CategorySet::EMPTY, CategorySet::default());
        assert_eq!(CategorySet::EMPTY.len(), 0);
        assert_eq!(CategorySet::ALL.len(), Category::COUNT);
        assert_eq!(CategorySet::ALL, Category::ALL.into_iter().collect());
        for category in Category::ALL {
            assert!(!CategorySet::EMPTY.contains(category));
            assert!(CategorySet::ALL.contains(category));
        }
    }

    #[test]
    fn bits_follow_category_indexes() {
        for category in Category::ALL {
            let set = CategorySet::only(category);
            assert_eq!(set.bits(), 1 << category.index());
            assert_eq!(set, CategorySet::from(category));
            assert_eq!(set.len(), 1);
            assert_eq!(set.iter().collect::<Vec<_>>(), [category]);
        }
        for bits in 0..=u8::MAX {
            assert_eq!(CategorySet::from_bits(bits).bits(), bits);
        }
    }

    #[test]
    fn with_and_without_change_one_category() {
        for bits in 0..=u8::MAX {
            let set = CategorySet::from_bits(bits);
            for category in Category::ALL {
                let added = set.with(category);
                let removed = set.without(category);
                assert!(added.contains(category));
                assert!(!removed.contains(category));
                assert_eq!(added.without(category), removed);
                for other in Category::ALL {
                    if other != category {
                        assert_eq!(added.contains(other), set.contains(other));
                        assert_eq!(removed.contains(other), set.contains(other));
                    }
                }
            }
        }
    }

    #[test]
    fn operators_match_the_set_operations() {
        let left = CategorySet::from(Category::Malformed) | Category::SqlInjection;
        let right = CategorySet::from(Category::SqlInjection) | Category::HeaderInjection;
        assert_eq!((left | right).len(), 3);
        assert_eq!(left & right, Category::SqlInjection.into());
        assert_eq!(left - right, Category::Malformed.into());
        assert_eq!(right - left, Category::HeaderInjection.into());
        assert_eq!(left - Category::Malformed, Category::SqlInjection.into());
        assert_eq!(left & Category::HeaderInjection, CategorySet::EMPTY);
        let mut grown = left;
        grown |= right;
        assert_eq!(grown, left.union(right));
        for a in 0..=u8::MAX {
            for b in [0, 1, 0x5A, 0xA5, 0xFF] {
                let (x, y) = (CategorySet::from_bits(a), CategorySet::from_bits(b));
                assert_eq!(x.union(y).bits(), a | b);
                assert_eq!(x.intersection(y).bits(), a & b);
                assert_eq!(x.difference(y).bits(), a & !b);
            }
        }
    }

    #[test]
    fn iteration_is_ordered_by_discriminant() {
        let set = CategorySet::from(Category::HeaderInjection)
            | Category::Malformed
            | Category::PathTraversal;
        assert_eq!(
            set.iter().collect::<Vec<_>>(),
            [
                Category::Malformed,
                Category::PathTraversal,
                Category::HeaderInjection
            ]
        );
    }

    #[test]
    fn text_round_trips_for_every_set() {
        for bits in 0..=u8::MAX {
            let set = CategorySet::from_bits(bits);
            assert_eq!(set.to_string().parse(), Ok(set));
        }
        assert_eq!(CategorySet::EMPTY.to_string(), "benign");
        assert_eq!(
            (CategorySet::from(Category::Obfuscation) | Category::SqlInjection).to_string(),
            "obfuscation,sql_injection"
        );
        assert_eq!(
            format!("{:?}", CategorySet::from(Category::Malformed)),
            "{Malformed}"
        );
    }

    #[test]
    fn parsing_accepts_blanks_and_repeats_and_rejects_other_text() {
        assert_eq!("".parse(), Ok(CategorySet::EMPTY));
        assert_eq!(" \t".parse(), Ok(CategorySet::EMPTY));
        assert_eq!(" benign ".parse(), Ok(CategorySet::EMPTY));
        assert_eq!(
            " malformed , malformed,sql_injection".parse(),
            Ok(CategorySet::from(Category::Malformed) | Category::SqlInjection)
        );
        for text in [
            ",",
            "malformed,",
            "benign,malformed",
            "Malformed",
            "sql injection",
            "malformed\n",
            "all",
        ] {
            assert_eq!(text.parse::<CategorySet>(), Err(ParseCategorySetError));
        }
        assert_eq!(
            ParseCategorySetError.to_string(),
            "text is not a list of category names"
        );
    }
}
