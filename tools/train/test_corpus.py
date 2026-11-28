"""Tests for the string2fingerprint synthetic corpus generator.

Run with:
    python3 -B -m unittest discover -s /home/user/work/corpus -p 'test_*.py'
"""

import io
import struct
import tempfile
import os
import unittest

import corpus


def collect(seed, n):
    return list(corpus.generate(seed, n))


def read_corpus(path):
    """Minimal reader mirroring write_corpus's format."""
    out = []
    header = struct.Struct("<IB")
    with open(path, "rb") as f:
        data = f.read()
    i = 0
    while i < len(data):
        length, mask = header.unpack_from(data, i)
        i += header.size
        payload = data[i:i + length]
        assert len(payload) == length, "truncated record"
        i += length
        out.append((payload, mask))
    return out


def is_valid_utf8(b):
    try:
        b.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def has_bad_control(b):
    return any(c < 0x20 and c not in (0x09, 0x0A, 0x0D) for c in b)


def has_invalid_pct_escape(b):
    hexd = b"0123456789abcdefABCDEF"
    for j in range(len(b)):
        if b[j] == 0x25:  # '%'
            h = b[j + 1:j + 3]
            if len(h) < 2 or h[0] not in hexd or h[1] not in hexd:
                return True
    return False


class TestDeterminism(unittest.TestCase):
    def test_same_seed_identical(self):
        self.assertEqual(collect(12345, 3000), collect(12345, 3000))

    def test_prefix_stability(self):
        # First n of a longer run match a shorter run (single RNG stream).
        short = collect(99, 500)
        longer = collect(99, 1500)
        self.assertEqual(short, longer[:500])

    def test_different_seeds_differ(self):
        a = collect(1, 2000)
        b = collect(2, 2000)
        self.assertNotEqual(a, b)
        # overlap should be small
        overlap = len(set(a) & set(b))
        self.assertLess(overlap, 200)


class TestBounds(unittest.TestCase):
    def test_length_bounds(self):
        for payload, _ in corpus.generate(5, 20000):
            self.assertGreaterEqual(len(payload), 1)
            self.assertLessEqual(len(payload), 512)

    def test_payload_is_bytes(self):
        for payload, mask in corpus.generate(6, 2000):
            self.assertIsInstance(payload, (bytes, bytearray))
            self.assertIsInstance(mask, int)

    def test_mask_range(self):
        for _, mask in corpus.generate(7, 20000):
            self.assertGreaterEqual(mask, 0)
            self.assertLessEqual(mask, 0xFF)

    def test_categories_shape(self):
        self.assertEqual(len(corpus.CATEGORIES), 8)
        self.assertEqual(len(set(corpus.CATEGORIES)), 8)

    def test_length_spread(self):
        lens = [len(p) for p, _ in corpus.generate(21, 20000)]
        # most between 8 and 300
        mid = sum(1 for n in lens if 8 <= n <= 300)
        self.assertGreater(mid / len(lens), 0.6)
        # a meaningful spread of short and longer samples
        self.assertGreater(sum(1 for n in lens if n < 16), 50)
        self.assertGreater(sum(1 for n in lens if n > 120), 50)


class TestBalance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples = collect(2024, 20000)

    def test_benign_attack_balance(self):
        benign = sum(1 for _, m in self.samples if m == 0)
        frac = benign / len(self.samples)
        self.assertGreater(frac, 0.40)
        self.assertLess(frac, 0.60)

    def test_multilabel_share(self):
        attacks = [m for _, m in self.samples if m != 0]
        multi = sum(1 for m in attacks if bin(m).count("1") >= 2)
        frac = multi / len(attacks)
        self.assertGreater(frac, 0.12)
        self.assertLess(frac, 0.45)

    def test_each_category_present_and_balanced(self):
        counts = [0] * 8
        for _, m in self.samples:
            for i in range(8):
                if m & (1 << i):
                    counts[i] += 1
        attack_total = sum(1 for _, m in self.samples if m != 0)
        for i, c in enumerate(counts):
            with self.subTest(cat=corpus.CATEGORIES[i]):
                # every category appears a nontrivial amount
                self.assertGreater(c, attack_total * 0.04,
                                   "%s too rare: %d" % (corpus.CATEGORIES[i], c))
                self.assertLess(c, attack_total * 0.55,
                                "%s too common: %d" % (corpus.CATEGORIES[i], c))

    def test_each_bit_alone_and_combined(self):
        alone = set()
        combined = set()
        for _, m in self.samples:
            if m == 0:
                continue
            bits = [i for i in range(8) if m & (1 << i)]
            if len(bits) == 1:
                alone.add(bits[0])
            else:
                for b in bits:
                    combined.add(b)
        for i in range(8):
            with self.subTest(cat=corpus.CATEGORIES[i]):
                self.assertIn(i, alone, "%s never alone" % corpus.CATEGORIES[i])
                self.assertIn(i, combined, "%s never combined" % corpus.CATEGORIES[i])


class TestLabelInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples = collect(555, 20000)

    def test_benign_valid_utf8_no_nul(self):
        for payload, mask in self.samples:
            if mask == 0:
                self.assertTrue(is_valid_utf8(payload),
                                "benign not utf-8: %r" % payload)
                self.assertNotIn(0, payload, "benign has NUL: %r" % payload)

    def test_malformed_only_criterion(self):
        checked = 0
        for payload, mask in self.samples:
            if mask == corpus.M_MALFORMED:
                checked += 1
                ok = (not is_valid_utf8(payload)
                      or has_bad_control(payload)
                      or has_invalid_pct_escape(payload))
                self.assertTrue(ok, "malformed-only not detectable: %r" % payload)
        self.assertGreater(checked, 50)

    def test_invalid_text_carries_the_malformed_bit(self):
        for payload, mask in self.samples:
            if not is_valid_utf8(payload) or has_bad_control(payload):
                self.assertTrue(mask & corpus.M_MALFORMED, "malformed bit missing: %r" % payload)

    def test_overlong_utf8_is_also_obfuscation(self):
        rng = random.Random(8)
        for _ in range(2000):
            payload, mask = corpus.gen_malformed(rng)
            if payload.startswith((b"\xc0\xae", b"\xc0\xaf", b"\xc0\xbc")):
                self.assertTrue(mask & corpus.M_OBFUSCATION, payload)

    def test_benign_no_control_except_whitespace(self):
        for payload, mask in self.samples:
            if mask == 0:
                self.assertFalse(has_bad_control(payload),
                                 "benign has control byte: %r" % payload)


class TestWriteCorpus(unittest.TestCase):
    def test_roundtrip(self):
        samples = collect(77, 3000)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.bin")
            n = corpus.write_corpus(path, samples)
            self.assertEqual(n, len(samples))
            read_back = read_corpus(path)
        self.assertEqual(read_back, samples)

    def test_count_return(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.bin")
            n = corpus.write_corpus(path, corpus.generate(3, 1234))
            self.assertEqual(n, 1234)

    def test_empty(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.bin")
            n = corpus.write_corpus(path, [])
            self.assertEqual(n, 0)
            self.assertEqual(read_corpus(path), [])


class TestGrammarDiversity(unittest.TestCase):
    def test_attacks_nearly_unique(self):
        # Compositional attack grammars should be almost collision-free.
        attacks = [p for p, m in corpus.generate(31, 20000) if m != 0]
        self.assertGreater(len(attacks), 5000)
        self.assertGreater(len(set(attacks)) / len(attacks), 0.97)

    def test_thousands_of_shapes_per_category(self):
        import collections
        by = collections.defaultdict(set)
        for p, m in corpus.generate(31, 40000):
            for i in range(8):
                if m & (1 << i):
                    by[i].add(p)
        for i in range(8):
            with self.subTest(cat=corpus.CATEGORIES[i]):
                self.assertGreater(len(by[i]), 2000,
                                   "%s: only %d shapes" % (corpus.CATEGORIES[i], len(by[i])))

    def test_overall_diversity(self):
        # Including repeated realistic benign phrases, still mostly distinct.
        seen = set(p for p, _ in corpus.generate(31, 10000))
        self.assertGreater(len(seen), 7000)


# --------------------------------------------------------------------------
# v2 guarantees
# --------------------------------------------------------------------------

import random  # noqa: E402
import re  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402

import benign  # noqa: E402


def _diff_span(a, b):
    """Length of the differing middle of a and b after removing common prefix/suffix."""
    p = 0
    while p < min(len(a), len(b)) and a[p] == b[p]:
        p += 1
    s = 0
    while s < min(len(a), len(b)) - p and a[-1 - s] == b[-1 - s]:
        s += 1
    return len(a) - p - s, len(b) - p - s


def _non_ascii_share(b):
    s = b.decode("utf-8", "replace")
    return sum(1 for ch in s if ord(ch) > 0x7F) / max(1, len(s))


class TestRecordsStream(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recs = list(corpus.generate_records(4242, 30000))

    def test_generate_matches_records(self):
        self.assertEqual([(r.payload, r.mask) for r in self.recs[:2000]], collect(4242, 2000))

    def test_every_benign_passes_final_check(self):
        for r in self.recs:
            if r.mask == 0:
                self.assertTrue(benign.benign_ok(r.payload), "benign failed check: %r" % r.payload[:200])

    def test_benign_share_of_pairs_and_hard_negatives(self):
        ben = [r for r in self.recs if r.mask == 0]
        special = sum(1 for r in ben if r.kind in ("clean_pair", "plain_pair", "hardneg"))
        frac = special / len(ben)
        self.assertGreater(frac, 0.20)
        self.assertLess(frac, 0.45)

    def test_corrupted_twin_differs_only_in_corruption(self):
        by_pid = defaultdict(dict)
        for r in self.recs:
            if r.kind in ("corrupt", "clean_pair"):
                by_pid[r.pair][r.kind] = r
        corrupt = [d for d in by_pid.values() if "corrupt" in d]
        both = [d for d in corrupt if "clean_pair" in d]
        self.assertGreater(len(corrupt), 500)
        self.assertGreater(len(both) / len(corrupt), 0.85)
        for d in both:
            clean, bad = d["clean_pair"], d["corrupt"]
            self.assertEqual(clean.mask, 0)
            self.assertEqual(bad.mask, corpus.M_MALFORMED)
            self.assertTrue(is_valid_utf8(clean.payload))
            self.assertTrue(benign.is_malformed(bad.payload))
            removed, added = _diff_span(clean.payload, bad.payload)
            self.assertLessEqual(removed, 3, (clean.payload, bad.payload))
            self.assertLessEqual(added, 16, (clean.payload, bad.payload))

    def test_corruption_kinds_are_varied(self):
        bad = [r.payload for r in self.recs if r.kind == "corrupt"]
        inv_utf8 = sum(1 for b in bad if not is_valid_utf8(b)) / len(bad)
        ctrl = sum(1 for b in bad if has_bad_control(b)) / len(bad)
        pct = sum(1 for b in bad if is_valid_utf8(b) and not has_bad_control(b)) / len(bad)
        self.assertGreater(inv_utf8, 0.3)
        self.assertGreater(ctrl, 0.1)
        self.assertGreater(pct, 0.05)  # only explainable by a damaged percent-escape

    def test_malformed_mostly_from_corruption(self):
        mal_only = [r for r in self.recs if r.mask == corpus.M_MALFORMED]
        frac = sum(1 for r in mal_only if r.kind == "corrupt") / len(mal_only)
        self.assertGreater(frac, 0.6)
        # corrupted samples come from realistic text: wide length spread
        lens = [len(r.payload) for r in mal_only if r.kind == "corrupt"]
        self.assertGreater(sum(1 for n in lens if n > 60) / len(lens), 0.4)

    def test_encoding_pairs(self):
        by_pid = defaultdict(dict)
        for r in self.recs:
            if r.kind in ("obf_pair", "plain_pair"):
                by_pid[r.pair][r.kind] = r
        pairs = [d for d in by_pid.values() if len(d) == 2]
        self.assertGreater(len(pairs), 200)
        # escape runs may have their backslashes doubled by a JSON or GraphQL carrier, or both
        layered = re.compile(rb"%25[0-9A-Fa-f]{2}|%u[0-9A-Fa-f]{4}|(&#[xX]?[0-9A-Fa-f]+;?){3}"
                             rb"|(\\{1,4}(x[0-9a-fA-F]{2}|u[0-9a-f]{4}|[0-7]{2,3}|u\{[0-9a-f]+\})){3}"
                             rb"|(\\{1,4}[0-9a-f]{2,6} ?){3}")
        for d in pairs:
            o, p = d["obf_pair"], d["plain_pair"]
            self.assertEqual(o.mask, corpus.M_OBFUSCATION)
            self.assertEqual(p.mask, 0)
            self.assertEqual(o.carrier, p.carrier)
            self.assertTrue(layered.search(o.payload), o.payload)
            self.assertFalse(re.search(rb"%25[0-9A-Fa-f]{2}", p.payload), p.payload)

    def test_carriers_shared_by_benign_and_attack(self):
        ben = {r.carrier for r in self.recs if r.kind == "benign"}
        atk = {r.carrier for r in self.recs if r.kind == "attack"}
        for c in ("bare", "query", "form", "cookie", "path", "prose", "json", "csv", "xml", "graphql",
                  "yaml", "html_attr", "header:ua", "header:referer", "header:xff", "header:cookie",
                  "header:auth", "header:disposition", "header:other"):
            with self.subTest(carrier=c):
                self.assertIn(c, ben)
                self.assertIn(c, atk)

    def test_no_double_encoding_without_obfuscation_bit(self):
        pat = re.compile(rb"%25[0-9A-Fa-f]{2}")
        for r in self.recs:
            if r.mask and not (r.mask & corpus.M_OBFUSCATION):
                self.assertFalse(pat.search(r.payload), r.payload[:200])

    def test_non_latin_benign_not_from_fixed_set(self):
        nl = [r.payload for r in self.recs if r.mask == 0 and _non_ascii_share(r.payload) > 0.3]
        self.assertGreater(len(nl), 1000)
        self.assertGreater(len(set(nl)) / len(nl), 0.97)
        cps = {ch for b in nl for ch in b.decode("utf-8") if ord(ch) > 0x7F}
        self.assertGreater(len(cps), 1500)

    def test_fullwidth_benign_and_obfuscation_both_present(self):
        cjk_punct = re.compile("[，。！？「」]")
        fw_letters = re.compile("[Ａ-Ｚａ-ｚ]{3}")
        ben = sum(1 for r in self.recs if r.mask == 0 and cjk_punct.search(r.payload.decode("utf-8")))
        obf = sum(1 for r in self.recs if r.mask & corpus.M_OBFUSCATION and is_valid_utf8(r.payload)
                  and fw_letters.search(r.payload.decode("utf-8")))
        self.assertGreater(ben, 300)
        self.assertGreater(obf, 20)


class TestBenignModule(unittest.TestCase):
    def test_vocabulary_large(self):
        self.assertGreaterEqual(len(set(benign.VOCAB)), 2000)

    def test_each_script_generated_not_fixed(self):
        rng = random.Random(9)
        for script in benign.SCRIPTS:
            with self.subTest(script=script):
                out = [benign.script_text(rng, script) for _ in range(300)]
                self.assertGreater(len(set(out)), 285)

    def test_english_text_varied(self):
        rng = random.Random(10)
        out = [benign.english_sentence(rng) for _ in range(2000)]
        self.assertGreater(len(set(out)), 1990)
        words = Counter(w.strip(".,;:!?\"'()").lower() for s in out for w in s.split())
        self.assertGreater(len(words), 1500)
        joined = "\n".join(out)
        for mark in (" -- ", "; ", "...", "%", "$", "&", "@", "#", "'"):
            self.assertIn(mark, joined)

    def test_corrupt_is_single_small_damage(self):
        rng = random.Random(11)
        for _ in range(3000):
            clean = benign.corruption_base(rng)
            if not benign.benign_ok(clean):
                continue
            bad = benign.corrupt(rng, clean)
            self.assertTrue(benign.is_malformed(bad))
            self.assertNotEqual(bad, clean)
            removed, added = _diff_span(clean, bad)
            self.assertLessEqual(removed, 3)
            self.assertLessEqual(added, 16)

    def test_corrupt_each_kind(self):
        rng = random.Random(12)
        base = "주문 확인 완료".encode("utf-8") + b"&q=caf%C3%A9+au+lait&page=2"
        for kind in benign.CORRUPTIONS:
            with self.subTest(kind=kind):
                bad = benign.corrupt(rng, base, kind)
                self.assertTrue(benign.is_malformed(bad))

    def test_short_query_strings_present(self):
        rng = random.Random(13)
        qs = [benign.query_string(rng) for _ in range(3000)]
        self.assertGreater(sum(1 for q in qs if len(q) <= 12), 100)

    def test_neutral_hosts(self):
        rng = random.Random(14)
        hosts = [corpus.rand_host(rng) for _ in range(3000)]
        self.assertGreater(len(set(hosts)), 2500)
        for h in hosts:
            for w in ("evil", "attacker", "exfil", "burp", "collab"):
                self.assertNotIn(w, h)


if __name__ == "__main__":
    unittest.main()
