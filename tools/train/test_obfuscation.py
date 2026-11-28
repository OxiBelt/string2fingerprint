"""Tests for the obfuscating encodings, the malformed generators, and the combiner."""

import random
import unittest

import benign
import corpus
import obfuscation

PAYLOADS = (
    "<script>alert(1)</script>",
    "' OR 1=1--",
    "../../etc/passwd",
    "; cat /etc/passwd",
    "{{7*7}}",
    "union select password from users",
)


class TechniqueTest(unittest.TestCase):
    def test_every_technique_changes_its_input_and_sets_the_label(self):
        rng = random.Random(1)
        for name in obfuscation.TECHNIQUES:
            for text in PAYLOADS:
                with self.subTest(technique=name, text=text):
                    data, mask = obfuscation.apply(name, text, rng)
                    self.assertNotEqual(data, text.encode("utf-8"))
                    self.assertTrue(mask & obfuscation.M_OBFUSCATION)
                    self.assertEqual(bool(mask & obfuscation.M_MALFORMED), name == "overlong")
                    self.assertEqual(benign.is_malformed(data) and not benign._valid_utf8(data), name == "overlong")

    def test_techniques_are_deterministic(self):
        for name in obfuscation.TECHNIQUES:
            with self.subTest(technique=name):
                first = [obfuscation.apply(name, text, random.Random(5)) for text in PAYLOADS]
                second = [obfuscation.apply(name, text, random.Random(5)) for text in PAYLOADS]
                self.assertEqual(first, second)

    def test_benign_check_rejects_obfuscated_payloads(self):
        rng = random.Random(2)
        for name in obfuscation.TECHNIQUES:
            if name == "overlong":
                continue  # not UTF-8, so never benign
            with self.subTest(technique=name):
                outputs = [obfuscation.apply(name, text, rng)[0] for text in PAYLOADS for _ in range(20)]
                passed = sum(1 for data in outputs if benign.benign_ok(data))
                self.assertLess(passed / len(outputs), 0.1)

    def test_layered_chains_add_a_percent_layer(self):
        rng = random.Random(3)
        for text in PAYLOADS:
            data, mask = obfuscation.layered(text, rng, 0)
            self.assertEqual(mask, obfuscation.M_OBFUSCATION)
            self.assertIn(b"%", data)
            data.decode("ascii")

    def test_obfuscation_samples_use_many_techniques(self):
        rng = random.Random(4)
        samples = [corpus.gen_obfuscation(rng)[0] for _ in range(3000)]
        self.assertGreater(len(set(samples)) / len(samples), 0.9)
        markers = (b"%25", b"%u", b"&#", b"\\x", b"\\u", b"\\u{", b"\xef\xbc", b"\xf0\x9d", b"\xe2\x80\x8b",
                   b"/**/", b"/*!", b"''", b"+AD", b"\xc0", b"'+'")
        for marker in markers:
            with self.subTest(marker=marker):
                self.assertTrue(any(marker in sample for sample in samples))


class MalformedTest(unittest.TestCase):
    def test_every_generated_sample_is_malformed(self):
        rng = random.Random(6)
        samples = [corpus.gen_malformed(rng) for _ in range(4000)]
        for data, mask in samples:
            # Overlong UTF-8 also carries obfuscation, as it does from `obfuscation.apply`.
            self.assertIn(mask, (corpus.M_MALFORMED, corpus.M_MALFORMED | corpus.M_OBFUSCATION))
            self.assertTrue(benign.is_malformed(data), data)
        self.assertGreater(len({data for data, _ in samples}) / len(samples), 0.85)

    def test_legacy_encodings_and_utf16_are_generated(self):
        rng = random.Random(7)
        samples = [corpus.gen_malformed(rng)[0] for _ in range(4000)]
        self.assertTrue(any(b"\x1b[" in s for s in samples))
        self.assertTrue(any(s.count(b"\x00") >= 3 for s in samples))
        self.assertTrue(any(s.startswith(magic) or magic in s for s in samples for magic in (b"PK\x03\x04", b"\x89PNG")))
        # legacy double-byte text: many high bytes and no valid UTF-8
        self.assertTrue(any(sum(1 for c in s if c >= 0x80) > 10 and not benign._valid_utf8(s) for s in samples))

    def test_every_corruption_kind_applies(self):
        rng = random.Random(8)
        for kind in benign.CORRUPTIONS:
            with self.subTest(kind=kind):
                done = 0
                for _ in range(300):
                    clean = rng.choice((benign.script_text(rng), benign.encoded_value(rng), benign.form_body(rng)))
                    out = benign._apply(rng, clean.encode("utf-8"), kind)
                    if out is not None and benign.is_malformed(out):
                        done += 1
                self.assertGreater(done, 20)


class CombinerTest(unittest.TestCase):
    def test_cross_category_payloads_have_two_labels(self):
        rng = random.Random(9)
        seen = set()
        for _ in range(500):
            data, mask = corpus.cross_category(rng)
            self.assertEqual(bin(mask).count("1"), 2, data)
            self.assertFalse(benign.benign_ok(data), data)
            seen.add(mask)
        self.assertGreaterEqual(len(seen), 8)

    def test_combine_adds_labels(self):
        rng = random.Random(10)
        for _ in range(2000):
            payload, mask = corpus.gen_sql(rng)
            data, combined = corpus.combine(rng, payload, mask)
            self.assertGreaterEqual(bin(combined).count("1"), 2, data)
            if combined & corpus.M_MALFORMED:
                self.assertTrue(benign.is_malformed(data), data)


if __name__ == "__main__":
    unittest.main()
