"""Obfuscating encodings for the corpus.

Standard library only. Every transform takes text and a `random.Random`,
draws all of its randomness from it, and returns text or bytes, so output is
deterministic for a seed.

A transform here hides the characters or keywords of its input in a way
that ordinary clients do not produce: a second or third layer of
percent-encoding, `%u` escapes, overlong UTF-8, runs of numeric character
references or language escapes that spell ASCII, look-alike or decorated
characters, and keyword splitting. Each one adds the `obfuscation` label.
A single layer of percent-encoding, which every client applies, is not
here.

`obfuscate()` applies one transform, and `layered()` composes two or three
of them, so that the model sees chains such as an entity run that is then
percent-encoded twice.
"""

from __future__ import annotations

import base64
import random
from typing import Callable, Tuple

M_MALFORMED = 1 << 0
M_OBFUSCATION = 1 << 1

_UNRESERVED = frozenset(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")


# --------------------------------------------------------------------------
# Percent-encoding layers
# --------------------------------------------------------------------------

def pct_layer(data: bytes, rng: random.Random, every: bool = False) -> bytes:
    """One layer of percent-encoding, in random letter case.

    Unreserved bytes stay literal unless `every` is set, in which case a
    random share of them is escaped as well, as some evasions do.
    """
    upper = rng.random() < 0.7
    share = rng.choice((0.3, 0.6, 1.0)) if every else 0.0
    out = bytearray()
    for c in data:
        if c in _UNRESERVED and not (share and rng.random() < share):
            out.append(c)
        else:
            out += (b"%%%02X" if upper else b"%%%02x") % c
    return bytes(out)


def double_pct(text: str, rng: random.Random) -> bytes:
    first = pct_layer(text.encode("utf-8"), rng, every=rng.random() < 0.5)
    if b"%" not in first:
        first = b"".join(b"%%%02X" % c for c in text.encode("utf-8"))
    return pct_layer(first, rng)


def triple_pct(text: str, rng: random.Random) -> bytes:
    return pct_layer(double_pct(text, rng), rng)


def partial_double_pct(text: str, rng: random.Random) -> bytes:
    """Double-encode only some reserved bytes, single-encode the rest."""
    out = bytearray()
    forced = False
    data = text.encode("utf-8")
    reserved = [i for i, c in enumerate(data) if c not in _UNRESERVED]
    pick = rng.choice(reserved) if reserved else -1
    for i, c in enumerate(data):
        if c in _UNRESERVED and i != pick:
            out.append(c)
        elif i == pick or rng.random() < 0.5:
            out += b"%%25%02X" % c
            forced = True
        else:
            out += b"%%%02X" % c
    if not forced:
        out = bytearray(b"%25" + b"%02X" % data[0] + data[1:]) if data else bytearray(b"%252E")
    return bytes(out)


def u_encode(text: str, rng: random.Random) -> bytes:
    """IIS-style `%uXXXX` escapes for every or some characters."""
    share = rng.choice((1.0, 1.0, 0.6))
    out = []
    escaped = 0
    for ch in text:
        if rng.random() < share or (escaped == 0 and ch == text[-1]):
            out.append(("%%u%04X" if rng.random() < 0.8 else "%%u%04x") % ord(ch) if ord(ch) <= 0xFFFF else ch)
            escaped += 1
        else:
            out.append(ch)
    return "".join(out).encode("utf-8")


def overlong_utf8(text: str) -> bytes:
    """Overlong two-byte encoding of ASCII. The result is not valid UTF-8."""
    out = bytearray()
    for ch in text:
        c = ord(ch) & 0x7F
        out.append(0xC0 | (c >> 6))
        out.append(0x80 | (c & 0x3F))
    return bytes(out)


def overlong_pct(text: str, rng: random.Random) -> bytes:
    """Percent-escaped overlong UTF-8, such as `%c0%ae` for a dot."""
    width = rng.choice((2, 2, 3))
    upper = rng.random() < 0.5
    out = []
    for ch in text:
        c = ord(ch)
        if c > 0x7F or (ch.isalnum() and rng.random() < 0.5):
            out.append(ch)
            continue
        if width == 2:
            seq = (0xC0 | (c >> 6), 0x80 | (c & 0x3F))
        else:
            seq = (0xE0, 0x80 | (c >> 6), 0x80 | (c & 0x3F))
        out.append("".join(("%%%02X" if upper else "%%%02x") % b for b in seq))
    return "".join(out).encode("utf-8")


# --------------------------------------------------------------------------
# Escape runs
# --------------------------------------------------------------------------

def entity_run(text: str, rng: random.Random) -> str:
    """Numeric character references, in one or mixed bases, sometimes without `;`."""
    mode = rng.choice(("dec", "hex", "mixed"))
    pad = rng.choice((0, 0, 3, 5, 7))
    semi = rng.random() < 0.85
    upper_x = rng.random() < 0.3
    out = []
    for ch in text:
        o = ord(ch)
        base = mode if mode != "mixed" else rng.choice(("dec", "hex"))
        if base == "hex":
            digits = ("%X" if rng.random() < 0.5 else "%x") % o
            out.append("&#" + ("X" if upper_x else "x") + digits.rjust(pad, "0") + (";" if semi else ""))
        else:
            out.append("&#" + str(o).rjust(pad, "0") + (";" if semi else ""))
    return "".join(out)


def escape_run(text: str, rng: random.Random) -> str:
    """Language escapes that spell `text`: `\\x`, `\\u`, `\\u{}`, octal, or mixed."""
    style = rng.choice(("x", "x", "u", "u", "octal", "ubrace", "mixed"))
    out = []
    for ch in text:
        o = ord(ch)
        s = style if style != "mixed" else rng.choice(("x", "u", "octal"))
        if s == "x" and o <= 0xFF:
            out.append(("\\x%02x" if rng.random() < 0.8 else "\\x%02X") % o)
        elif s == "octal" and o <= 0xFF:
            out.append("\\%03o" % o)
        elif s == "ubrace":
            out.append("\\u{%x}" % o)
        else:
            out.append("\\u%04x" % o if o <= 0xFFFF else ch)
    return "".join(out)


def css_escape_run(text: str, rng: random.Random) -> str:
    """CSS hexadecimal escapes, such as `\\3c ` or `\\00003c`."""
    pad = rng.choice((0, 0, 6))
    out = []
    for ch in text:
        digits = "%x" % ord(ch)
        if pad:
            out.append("\\" + digits.rjust(pad, "0"))
        else:
            out.append("\\" + digits + " ")
    return "".join(out)


def utf7(text: str, rng: random.Random) -> str:
    """UTF-7 with the markup and punctuation characters shifted into base64."""
    out = []
    run = ""
    for ch in text + "\0":
        if ch != "\0" and not (ch.isalnum() and ord(ch) < 0x80) and ch not in " .,":
            run += ch
            continue
        if run:
            b64 = base64.b64encode(run.encode("utf-16-be")).decode("ascii").rstrip("=")
            out.append("+" + b64 + ("-" if rng.random() < 0.9 or ch.isalnum() else ""))
            run = ""
        if ch != "\0":
            out.append(ch)
    s = "".join(out)
    return s if "+" in s else "+AC0-" + s


# --------------------------------------------------------------------------
# Look-alike and decorated characters
# --------------------------------------------------------------------------

_FULLWIDTH_SPACE = "　"


def to_fullwidth(text: str, rng: random.Random) -> str:
    share = rng.choice((1.0, 1.0, 0.5))
    out = []
    for ch in text:
        o = ord(ch)
        if ch == " " and rng.random() < share:
            out.append(_FULLWIDTH_SPACE)
        elif 0x21 <= o <= 0x7E and rng.random() < share:
            out.append(chr(o - 0x20 + 0xFF00))
        else:
            out.append(ch)
    return "".join(out)


_HOMOGLYPHS = {
    "a": ("а", "ɑ"), "e": ("е",), "o": ("о", "ο"), "c": ("с", "ϲ"),
    "p": ("р", "ρ"), "x": ("х", "χ"), "y": ("у",), "i": ("і", "ı"),
    "s": ("ѕ",), "d": ("ԁ",), "j": ("ј",), "h": ("һ",), "l": ("ӏ",),
    "n": ("ո",), "r": ("г",), "t": ("т",), "A": ("А", "Α"), "E": ("Е", "Ε"),
    "O": ("О", "Ο"), "C": ("С",), "T": ("Т", "Τ"), "S": ("Ѕ",),
    "<": ("‹", "﹤", "˂", "❮"), ">": ("›", "﹥", "˃", "❯"),
    "'": ("′", "ʼ", "＇"), '"': ("″", "ʺ"), "/": ("⁄", "∕", "⧸"),
    "\\": ("∖", "﹨"), ".": ("․", "．"), "(": ("﹙",), ")": ("﹚",),
    "=": ("﹦",), ";": (";",), ":": ("ː", "﹕"),
}


def to_homoglyph(text: str, rng: random.Random) -> str:
    out = []
    changed = False
    for ch in text:
        alts = _HOMOGLYPHS.get(ch)
        if alts is not None and rng.random() < 0.6:
            out.append(rng.choice(alts))
            changed = True
        else:
            out.append(ch)
    if not changed:
        for i, ch in enumerate(out):
            if ch in _HOMOGLYPHS:
                out[i] = _HOMOGLYPHS[ch][0]
                break
    return "".join(out)


_MATH_BASES = (0x1D400, 0x1D41A, 0x1D5A0, 0x1D5BA, 0x1D670, 0x1D68A)  # bold, sans, monospace (upper, lower)


def to_math_alnum(text: str, rng: random.Random) -> str:
    """Mathematical alphanumeric letters, which compatibility normalization maps to ASCII."""
    style = rng.randrange(3)
    out = []
    for ch in text:
        if "A" <= ch <= "Z":
            out.append(chr(_MATH_BASES[2 * style] + ord(ch) - ord("A")))
        elif "a" <= ch <= "z":
            out.append(chr(_MATH_BASES[2 * style + 1] + ord(ch) - ord("a")))
        else:
            out.append(ch)
    return "".join(out)


_INVISIBLE = ("​", "‌", "‍", "⁠", "﻿", "­", "͏", "᠎")


def insert_invisible(text: str, rng: random.Random) -> str:
    """Zero-width or soft-hyphen characters between letters of a word."""
    out = []
    inserted = False
    for i, ch in enumerate(text):
        out.append(ch)
        if ch.isalpha() and i + 1 < len(text) and text[i + 1].isalpha() and rng.random() < 0.35:
            out.append(rng.choice(_INVISIBLE))
            inserted = True
    if not inserted and len(text) > 1:
        out.insert(1, rng.choice(_INVISIBLE))
    return "".join(out)


def combining_marks(text: str, rng: random.Random) -> str:
    """Combining marks after letters, which normalization can strip."""
    out = []
    for ch in text:
        out.append(ch)
        if ch.isalpha() and rng.random() < 0.4:
            out.append(chr(rng.choice((0x0300, 0x0301, 0x0302, 0x0308, 0x0327, 0x0336, 0x0338))))
    return "".join(out)


# --------------------------------------------------------------------------
# Keyword splitting
# --------------------------------------------------------------------------

def _split_words(text: str, rng: random.Random, seps: Tuple[str, ...], rate: float, alpha_only: bool) -> str:
    out = []
    split = False
    for i, ch in enumerate(text):
        out.append(ch)
        nxt = text[i + 1] if i + 1 < len(text) else ""
        ok = (ch.isalpha() and nxt.isalpha()) if alpha_only else (ch.isalnum() and nxt.isalnum())
        if ok and rng.random() < rate:
            out.append(rng.choice(seps))
            split = True
    if not split:
        for i in range(len(out) - 1):
            if out[i].isalpha() and out[i + 1].isalpha():
                out.insert(i + 1, seps[0])
                break
    return "".join(out)


def sql_comment_split(text: str, rng: random.Random) -> str:
    seps = ("/**/", "/**/", "/*x*/", "/*!*/", "/*%s*/" % rng.choice(("a", "foo", "1", "!50000")))
    return _split_words(text, rng, seps, 0.3, False)


def mysql_versioned(text: str, rng: random.Random) -> str:
    """Wrap SQL keywords in MySQL versioned comments, such as `/*!50000UNION*/`."""
    words = text.split(" ")
    out = []
    wrapped = False
    for w in words:
        if w.isalpha() and len(w) >= 3 and rng.random() < 0.6:
            out.append("/*!%s%s*/" % (rng.choice(("", "50000", "12345", "40100")), w))
            wrapped = True
        else:
            out.append(w)
    if not wrapped:
        out[0] = "/*!%s*/" % out[0]
    return rng.choice((" ", "/**/")).join(out)


def shell_quote_split(text: str, rng: random.Random) -> str:
    return _split_words(text, rng, ("''", '""', "\\", "$@", "${x}", "$()"), 0.35, True)


def js_string_split(text: str, rng: random.Random) -> str:
    """Spell identifiers as concatenated strings, such as `'al'+'ert'`."""
    q = rng.choice(("'", '"', "`"))
    out = []
    word = ""
    for ch in text + "\0":
        if ch.isalpha():
            word += ch
            continue
        if len(word) >= 4:
            cut = rng.randint(1, len(word) - 1)
            out.append(f"[{q}{word[:cut]}{q}+{q}{word[cut:]}{q}]" if rng.random() < 0.3 else f"{q}{word[:cut]}{q}+{q}{word[cut:]}{q}")
        else:
            out.append(word)
        word = ""
        if ch != "\0":
            out.append(ch)
    s = "".join(out)
    return s if s != text else f"{q}{text[:1]}{q}+{q}{text[1:]}{q}"


def case_and_null_comment(text: str, rng: random.Random) -> str:
    """Alternate case and split with empty comments, as filters that match one spelling miss."""
    s = "".join(c.upper() if rng.random() < 0.5 else c.lower() for c in text)
    return _split_words(s, rng, ("/**/", "/**/", "<!---->"), 0.2, True)


# --------------------------------------------------------------------------
# One transform, and chains of them
# --------------------------------------------------------------------------

# (name, weight)
_TEXT_TECHNIQUES: Tuple[Tuple[str, int], ...] = (
    ("entity", 3), ("escape", 3), ("css", 1), ("utf7", 1), ("fullwidth", 2), ("homoglyph", 2),
    ("math", 1), ("invisible", 2), ("combining", 1), ("sql_split", 2), ("mysql_versioned", 1),
    ("shell_split", 2), ("js_split", 1), ("case_comment", 1),
)
_BYTE_TECHNIQUES: Tuple[Tuple[str, int], ...] = (
    ("double_pct", 4), ("triple_pct", 1), ("partial_double", 2), ("u_encode", 2), ("overlong_pct", 1),
    ("overlong", 1),
)

TECHNIQUES = tuple(name for name, _ in _TEXT_TECHNIQUES + _BYTE_TECHNIQUES)


def _text_transform(name: str, text: str, rng: random.Random) -> str:
    return {
        "entity": entity_run, "escape": escape_run, "css": css_escape_run, "utf7": utf7,
        "fullwidth": to_fullwidth, "homoglyph": to_homoglyph, "math": to_math_alnum,
        "invisible": insert_invisible, "combining": combining_marks, "sql_split": sql_comment_split,
        "mysql_versioned": mysql_versioned, "shell_split": shell_quote_split, "js_split": js_string_split,
        "case_comment": case_and_null_comment,
    }[name](text, rng)


def apply(name: str, text: str, rng: random.Random) -> Tuple[bytes, int]:
    """Apply the technique `name`. Returns the bytes and the labels it adds.

    A technique that cannot change `text`, such as keyword splitting of a
    string without letters, falls back to a double percent-encoding, so that
    the `obfuscation` label is never set on unchanged text.
    """
    data, mask = _apply(name, text, rng)
    if data == text.encode("utf-8"):
        return double_pct(text, rng), M_OBFUSCATION
    return data, mask


def _apply(name: str, text: str, rng: random.Random) -> Tuple[bytes, int]:
    if name in ("double_pct", "triple_pct", "partial_double", "u_encode", "overlong_pct"):
        fn: Callable[[str, random.Random], bytes] = {
            "double_pct": double_pct, "triple_pct": triple_pct, "partial_double": partial_double_pct,
            "u_encode": u_encode, "overlong_pct": overlong_pct,
        }[name]
        return fn(text, rng), M_OBFUSCATION
    if name == "overlong":
        return overlong_utf8(text), M_OBFUSCATION | M_MALFORMED
    return _text_transform(name, text, rng).encode("utf-8"), M_OBFUSCATION


def choose(rng: random.Random, allow_malformed: bool = True) -> str:
    pool = _TEXT_TECHNIQUES + tuple(t for t in _BYTE_TECHNIQUES if allow_malformed or t[0] != "overlong")
    return rng.choices([n for n, _ in pool], [w for _, w in pool])[0]


def obfuscate(text: str, rng: random.Random, inner_mask: int) -> Tuple[bytes, int]:
    """Apply one obfuscation technique to `text`. Returns (bytes, mask)."""
    data, added = apply(choose(rng), text, rng)
    return data, inner_mask | added


def layered(text: str, rng: random.Random, inner_mask: int) -> Tuple[bytes, int]:
    """Apply a text technique, then one or two percent or `%u` layers."""
    name = rng.choices([n for n, _ in _TEXT_TECHNIQUES], [w for _, w in _TEXT_TECHNIQUES])[0]
    inner = _text_transform(name, text, rng)
    if inner == text:
        inner = entity_run(text, rng)
    data = pct_layer(inner.encode("utf-8"), rng)
    if b"%" not in data:
        data = b"".join(b"%%%02X" % c for c in inner.encode("utf-8"))
    if rng.random() < 0.4:
        data = pct_layer(data, rng)
    return data, inner_mask | M_OBFUSCATION
