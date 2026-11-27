"""Tests for the benign formats and hard negatives in `formats.py`."""

import random
import unittest

import benign
import corpus
import formats


class FamilyTest(unittest.TestCase):
    def test_every_family_passes_the_benign_check(self):
        rng = random.Random(1)
        for name, fn, _ in formats.FAMILIES:
            with self.subTest(family=name):
                samples = [fn(rng).encode("utf-8") for _ in range(500)]
                passed = sum(1 for sample in samples if benign.benign_ok(sample))
                self.assertGreater(passed / len(samples), 0.85)
                self.assertGreater(len(set(samples)) / len(samples), 0.5)

    def test_families_are_registered_as_benign(self):
        names = {name for name, _, _ in benign.all_families()}
        for name, _, _ in formats.FAMILIES:
            self.assertIn(name, names)
        rng = random.Random(2)
        drawn = {benign.sample(rng)[1] for _ in range(20000)}
        self.assertEqual(drawn, names)

    def test_hard_negatives_are_benign_and_varied(self):
        rng = random.Random(3)
        samples = [formats.hard_negative(rng).encode("utf-8") for _ in range(4000)]
        passed = sum(1 for sample in samples if benign.benign_ok(sample))
        self.assertGreater(passed / len(samples), 0.97)
        self.assertGreater(len(set(samples)) / len(samples), 0.9)

    def test_corpus_hard_negatives_are_mostly_generated(self):
        rng = random.Random(4)
        samples = [corpus._hard_negative(rng) for _ in range(3000)]
        self.assertGreater(len(set(samples)) / len(samples), 0.6)


class BenignCheckTest(unittest.TestCase):
    def test_ordinary_relative_references_are_benign(self):
        for text in ("import { a } from '../lib/a';", "url(../../images/logo.png)", "see ../docs/setup.md for details",
                     "const x = require('../util');"):
            with self.subTest(text=text):
                self.assertTrue(benign.benign_ok(text.encode()))

    def test_traversals_stay_forbidden(self):
        for text in ("../etc/passwd", "../../../lib/a.js", "../../../../windows/win.ini", "src/../../../x",
                     "..\\..\\boot.ini", "../.ssh/id_rsa", "../config/secrets.yml", "/var/www/../../x"):
            with self.subTest(text=text):
                self.assertFalse(benign.benign_ok(text.encode()))

    def test_images_need_an_alt_and_no_handler(self):
        self.assertTrue(benign.benign_ok(b'<img src="/a.png" alt="a cat">'))
        self.assertFalse(benign.benign_ok(b'<img src=x onerror=alert(1)>'))
        self.assertFalse(benign.benign_ok(b'<img src="/a.png" alt="x" onload="go()">'))
        self.assertFalse(benign.benign_ok(b"<img src=x>"))


if __name__ == "__main__":
    unittest.main()
