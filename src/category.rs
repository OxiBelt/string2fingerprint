//! Detection categories.

use core::fmt;

/// A kind of anomaly that a score describes.
///
/// Categories are independent. One input can score high in several of them,
/// for example a doubly percent-encoded directory traversal scores in both
/// [`Category::Obfuscation`] and [`Category::PathTraversal`].
///
/// The discriminants are stable. They index [`crate::Scores`] and the output
/// layer of a model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum Category {
    /// Structurally broken text: invalid UTF-8, embedded NUL or other control
    /// bytes, truncated or invalid percent-escapes, binary data.
    Malformed = 0,
    /// Layered or evasive encoding: repeated percent-encoding, `%u` escapes,
    /// overlong UTF-8, entity or backslash-escape runs that spell ASCII, and
    /// keyword splitting.
    Obfuscation = 1,
    /// An attempt to escape an SQL string or numeric context.
    SqlInjection = 2,
    /// An attempt to escape into HTML or XML markup, including cross-site
    /// scripting and external entity declarations.
    MarkupInjection = 3,
    /// An attempt to escape a directory, including encoded `../` sequences
    /// and references to sensitive absolute paths.
    PathTraversal = 4,
    /// An attempt to escape into a shell command.
    CommandInjection = 5,
    /// An attempt to escape into a template or expression language, including
    /// lookup expressions such as `${jndi:...}`.
    TemplateInjection = 6,
    /// An attempt to escape a header line with CR or LF, including response
    /// splitting.
    HeaderInjection = 7,
}

impl Category {
    /// The number of categories.
    pub const COUNT: usize = 8;

    /// Every category, ordered by discriminant.
    pub const ALL: [Self; Self::COUNT] = [
        Self::Malformed,
        Self::Obfuscation,
        Self::SqlInjection,
        Self::MarkupInjection,
        Self::PathTraversal,
        Self::CommandInjection,
        Self::TemplateInjection,
        Self::HeaderInjection,
    ];

    /// Returns the position of this category in [`Category::ALL`].
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Returns the category at `index` in [`Category::ALL`], if any.
    #[must_use]
    pub const fn from_index(index: usize) -> Option<Self> {
        if index < Self::COUNT {
            Some(Self::ALL[index])
        } else {
            None
        }
    }

    /// Returns a stable `snake_case` identifier, suitable for metric labels.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Malformed => "malformed",
            Self::Obfuscation => "obfuscation",
            Self::SqlInjection => "sql_injection",
            Self::MarkupInjection => "markup_injection",
            Self::PathTraversal => "path_traversal",
            Self::CommandInjection => "command_injection",
            Self::TemplateInjection => "template_injection",
            Self::HeaderInjection => "header_injection",
        }
    }

    /// Returns the category whose [`name`](Self::name) is `name`.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|category| category.name() == name)
    }
}

impl fmt::Display for Category {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::Category;

    #[test]
    fn all_is_ordered_by_discriminant() {
        for (index, category) in Category::ALL.into_iter().enumerate() {
            assert_eq!(category.index(), index);
            assert_eq!(Category::from_index(index), Some(category));
        }
    }

    #[test]
    fn from_index_rejects_out_of_range() {
        assert_eq!(Category::from_index(Category::COUNT), None);
        assert_eq!(Category::from_index(usize::MAX), None);
    }

    #[test]
    fn from_name_rejects_other_text() {
        for name in ["", "benign", "Malformed", "malformed ", "sql"] {
            assert_eq!(Category::from_name(name), None);
        }
    }

    #[test]
    fn names_are_unique_snake_case() {
        for (index, category) in Category::ALL.into_iter().enumerate() {
            let name = category.name();
            assert!(!name.is_empty());
            assert!(
                name.bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte == b'_')
            );
            assert_eq!(category.to_string(), name);
            assert_eq!(Category::from_name(name), Some(category));
            for other in &Category::ALL[..index] {
                assert_ne!(other.name(), name);
            }
        }
    }
}
