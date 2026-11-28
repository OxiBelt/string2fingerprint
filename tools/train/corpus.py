#!/usr/bin/env python3
"""Synthetic training-corpus generator for `string2fingerprint`.

Standard library only, deterministic for a given seed. Produces multi-label
samples of untrusted web text (benign and attack) for training a small MLP
over hashed byte n-gram counts.

Label bitmask (bit i set => category i applies; benign == 0):

    0 malformed          structurally broken bytes
    1 obfuscation        layered/evasive encoding
    2 sql_injection
    3 markup_injection   XSS / HTML / XML injection
    4 path_traversal
    5 command_injection
    6 template_injection  SSTI / EL / jndi
    7 header_injection    CR/LF injection, response splitting

The grammars of the six injection categories live in `attacks.py`. Benign
text, the pools that benign and attack carriers share (parameter names,
cookie names, JSON keys, neutral hosts), corruption of benign samples, and the
final benign check live in `benign.py`. Most malformed samples are a
benign sample with one structural damage, emitted near its clean twin. Some
obfuscation samples are a benign value with layered encoding, emitted near
its plain or single-layer form. See `generate_records()`.

The corpus is training data for a detector. Samples are short synthetic
strings and are not working exploits for any particular system.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import struct
import sys
from collections import deque
from typing import Iterable, Iterator, List, NamedTuple, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attacks  # noqa: E402  (attack grammars)
import benign  # noqa: E402  (benign text, shared pools, corruption)

CATEGORIES: Tuple[str, ...] = (
    "malformed",
    "obfuscation",
    "sql_injection",
    "markup_injection",
    "path_traversal",
    "command_injection",
    "template_injection",
    "header_injection",
)

M_MALFORMED = 1 << 0
M_OBFUSCATION = 1 << 1
M_SQL = 1 << 2
M_MARKUP = 1 << 3
M_PATH = 1 << 4
M_CMD = 1 << 5
M_TEMPLATE = 1 << 6
M_HEADER = 1 << 7

MAX_LEN = 512


# --------------------------------------------------------------------------
# Low-level encoding helpers
# --------------------------------------------------------------------------

_UNRESERVED = frozenset(
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
)


def _as_bytes(x) -> bytes:
    if isinstance(x, bytes):
        return x
    if isinstance(x, bytearray):
        return bytes(x)
    return x.encode("utf-8", "surrogatepass")


def pct_encode(data, space_as_plus: bool = False, safe: bytes = b"") -> bytes:
    """Single-layer percent-encoding of a value."""
    b = _as_bytes(data)
    safeset = set(safe)
    out = bytearray()
    for c in b:
        if c in _UNRESERVED or c in safeset:
            out.append(c)
        elif space_as_plus and c == 0x20:
            out.append(0x2B)
        else:
            out += b"%%%02X" % c
    return bytes(out)


def pct_encode_all(data) -> bytes:
    """Percent-encode every byte (used to build further layers)."""
    b = _as_bytes(data)
    out = bytearray()
    for c in b:
        if c in _UNRESERVED:
            out.append(c)
        else:
            out += b"%%%02X" % c
    return bytes(out)


def valid_utf8(b: bytes) -> bool:
    try:
        b.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def has_low_control(b: bytes) -> bool:
    """True if any byte < 0x20 is present (any C0 control incl. tab/CR/LF)."""
    return any(c < 0x20 for c in b)


# --------------------------------------------------------------------------
# Obfuscation transforms (operate on a text string, return bytes + extra mask)
# --------------------------------------------------------------------------

_FULLWIDTH_SPACE = "\u3000"


def _to_fullwidth(s: str) -> str:
    out = []
    for ch in s:
        o = ord(ch)
        if ch == " ":
            out.append(_FULLWIDTH_SPACE)
        elif 0x21 <= o <= 0x7E:
            out.append(chr(o - 0x20 + 0xFF00))
        else:
            out.append(ch)
    return "".join(out)


_HOMO = {
    "a": "\u0430", "e": "\u0435", "o": "\u043E", "c": "\u0441",
    "p": "\u0440", "x": "\u0445", "y": "\u0443", "i": "\u0456",
    "s": "\u0455", "d": "\u0501", "j": "\u0458", "A": "\u0410",
    "E": "\u0415", "O": "\u041E", "C": "\u0421", "<": "\u2039",
    ">": "\u203A", "'": "\u2032", "/": "\u2044",
}


def _to_homoglyph(s: str, rng: random.Random) -> str:
    out = []
    for ch in s:
        alt = _HOMO.get(ch)
        if alt is not None and rng.random() < 0.7:
            out.append(alt)
        else:
            out.append(ch)
    return "".join(out)


def _entity_run(s: str, rng: random.Random) -> str:
    hexmode = rng.random() < 0.5
    pad = rng.random() < 0.3
    out = []
    for ch in s:
        o = ord(ch)
        if hexmode:
            if pad:
                out.append("&#x%04X;" % o)
            else:
                out.append("&#x%X;" % o)
        else:
            if pad:
                out.append("&#%05d;" % o)
            else:
                out.append("&#%d;" % o)
    return "".join(out)


def _escape_run(s: str, rng: random.Random) -> str:
    style = rng.choice(("x", "x", "u", "u", "octal"))
    out = []
    for ch in s:
        o = ord(ch)
        if style == "x":
            out.append("\\x%02x" % (o & 0xFF))
        elif style == "u":
            out.append("\\u%04x" % o)
        else:
            out.append("\\%o" % (o & 0xFF))
    return "".join(out)


def _u_encode(s: str) -> bytes:
    return "".join("%%u%04X" % ord(ch) for ch in s).encode("ascii")


def _overlong_utf8(s: str) -> bytes:
    """Overlong 2-byte encoding of ASCII (invalid; also malformed)."""
    out = bytearray()
    for ch in s:
        c = ord(ch) & 0x7F
        out.append(0xC0 | (c >> 6))
        out.append(0x80 | (c & 0x3F))
    return bytes(out)


def _sql_comment_split(s: str, rng: random.Random) -> str:
    sep = rng.choice(("/**/", "/**/", "/*x*/", "/*!*/"))
    out = []
    for i, ch in enumerate(s):
        out.append(ch)
        if ch.isalnum() and i + 1 < len(s) and s[i + 1].isalnum() and rng.random() < 0.3:
            out.append(sep)
    return "".join(out)


def _shell_quote_split(s: str, rng: random.Random) -> str:
    out = []
    for i, ch in enumerate(s):
        out.append(ch)
        if ch.isalpha() and i + 1 < len(s) and s[i + 1].isalpha() and rng.random() < 0.35:
            out.append(rng.choice(("''", '""', "\\")))
    return "".join(out)


def obfuscate(text: str, rng: random.Random, inner_mask: int) -> Tuple[bytes, int]:
    """Apply one obfuscation technique to `text`. Returns (bytes, mask)."""
    tech = rng.choice(
        (
            "double_pct", "double_pct", "triple_pct", "u_encode",
            "overlong", "entity", "escape", "escape",
            "fullwidth", "homoglyph", "sql_split", "shell_split",
        )
    )
    mask = M_OBFUSCATION | inner_mask
    if tech == "double_pct":
        return pct_encode_all(pct_encode_all(text)), mask
    if tech == "triple_pct":
        return pct_encode_all(pct_encode_all(pct_encode_all(text))), mask
    if tech == "u_encode":
        return _u_encode(text), mask
    if tech == "overlong":
        return _overlong_utf8(text), mask | M_MALFORMED
    if tech == "entity":
        return _entity_run(text, rng).encode("ascii"), mask
    if tech == "escape":
        return _escape_run(text, rng).encode("ascii"), mask
    if tech == "fullwidth":
        return _to_fullwidth(text).encode("utf-8"), mask
    if tech == "homoglyph":
        return _to_homoglyph(text, rng).encode("utf-8"), mask
    if tech == "sql_split":
        return _sql_comment_split(text, rng).encode("utf-8"), mask
    # shell_split
    return _shell_quote_split(text, rng).encode("utf-8"), mask


# --------------------------------------------------------------------------
# Attack grammars (see `attacks.py`)
# --------------------------------------------------------------------------

PARAM_NAMES = benign.PARAM_NAMES  # one pool shared by benign and attack carriers
rand_ident = attacks.rand_ident
rand_host = attacks.rand_host
gen_sql = attacks.gen_sql
gen_markup = attacks.gen_markup
gen_path = attacks.gen_path
gen_cmd = attacks.gen_cmd
gen_template = attacks.gen_template
gen_header = attacks.gen_header


def gen_malformed(rng: random.Random) -> Tuple[bytes, int]:
    flavor = rng.choice(
        ("trunc", "stray", "badlead", "badsecond", "nul", "c0",
         "badpct", "badpct", "binary", "overlong")
    )
    # optional benign-ish ascii context
    ctx = ""
    if rng.random() < 0.5:
        ctx = rng.choice(PARAM_NAMES) + "="
    out = bytearray(ctx.encode("ascii"))
    if flavor == "trunc":
        lead = rng.choice((0xE2, 0xE0, 0xF0, 0xC3, 0xED, 0xF4))
        out.append(lead)
        if rng.random() < 0.5:
            out.append(rng.choice((0x82, 0x9F, 0x80, 0xA0)))
    elif flavor == "stray":
        for _ in range(rng.randint(1, 4)):
            out.append(rng.randint(0x80, 0xBF))
    elif flavor == "badlead":
        out.append(rng.choice((0xC0, 0xC1, 0xF5, 0xF6, 0xF7, 0xFE, 0xFF)))
        out.append(rng.randint(0x80, 0xBF))
    elif flavor == "badsecond":
        out.append(rng.choice((0xE2, 0xF0, 0xC3)))
        out.append(rng.randint(0x00, 0x7F))  # invalid continuation
    elif flavor == "nul":
        out += rand_ident(rng, 2, 6).encode()
        out.append(0x00)
        out += rand_ident(rng, 2, 6).encode()
    elif flavor == "c0":
        out += rand_ident(rng, 2, 5).encode()
        out.append(rng.choice((0x01, 0x02, 0x07, 0x08, 0x0B, 0x0C, 0x1B, 0x1F)))
        out += rand_ident(rng, 2, 5).encode()
    elif flavor == "badpct":
        bad = rng.choice(
            ("%zz", "%gg", "%4", "%2", "%", "%%", "abc%", "%2g", "%x1",
             "a%", "foo%2", "%u", "val%ue%", "%2." )
        )
        out += bad.encode("ascii")
        if rng.random() < 0.4:
            out += rand_ident(rng, 2, 6).encode()
    elif flavor == "binary":
        n = rng.randint(4, 24)
        for _ in range(n):
            out.append(rng.randint(0, 255))
        # guarantee a disqualifying byte
        out[rng.randrange(len(ctx), len(out)) if len(out) > len(ctx) else -1] = rng.choice(
            (0x00, 0xFF, 0xC0, 0x1B, 0x85))
    else:  # overlong (invalid utf-8)
        out += _overlong_utf8(rng.choice(("..", "/", "<script>", "../etc")))
    b = bytes(out)
    # Make sure it is truly malformed by our test's criteria.
    if valid_utf8(b) and not any(c < 0x20 and c not in (0x09, 0x0A, 0x0D) for c in b):
        # fall back: append a stray continuation byte
        b = b + bytes((rng.randint(0x80, 0xBF),))
    return b, M_MALFORMED


def gen_obfuscation(rng: random.Random) -> Tuple[bytes, int]:
    """Obfuscation as the primary category: obfuscate some inner content."""
    r = rng.random()
    if r < 0.55:
        # obfuscate an attack payload -> attack bit + obfuscation
        inner_gen = rng.choice((gen_sql, gen_markup, gen_path, gen_cmd, gen_template))
        raw, inner_mask = inner_gen(rng)
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1")
        return obfuscate(text, rng, inner_mask)
    else:
        # obfuscate a benign-ish ASCII token -> obfuscation alone
        token = rng.choice(
            (rand_ident(rng, 4, 10), ".", "..", "/", "admin", "etc/passwd",
             "<a>", "SELECT", "cat", "' or", "script", rng.choice(PARAM_NAMES))
        )
        return obfuscate(token, rng, 0)


CAT_GEN = {
    M_SQL: gen_sql,
    M_MARKUP: gen_markup,
    M_PATH: gen_path,
    M_CMD: gen_cmd,
    M_TEMPLATE: gen_template,
    M_HEADER: gen_header,
    M_MALFORMED: gen_malformed,
    M_OBFUSCATION: gen_obfuscation,
}

_ATTACK_BITS = [
    M_MALFORMED, M_OBFUSCATION, M_SQL, M_MARKUP, M_PATH, M_CMD,
    M_TEMPLATE, M_HEADER,
]


# --------------------------------------------------------------------------
# Multi-label combiner
# --------------------------------------------------------------------------

def combine(rng: random.Random, payload: bytes, mask: int) -> Tuple[bytes, int]:
    options = ["append"]
    can_text = valid_utf8(payload) and not has_low_control(payload)
    if not (mask & M_MALFORMED):
        options += ["nul", "ctrl"]
        if can_text:
            options += ["obf", "obf", "overlong"]
    choice = rng.choice(options)
    if choice == "append":
        other_bit = rng.choice([b for b in _ATTACK_BITS if b not in (M_OBFUSCATION,) and not (mask & b)] or _ATTACK_BITS)
        raw, m2 = CAT_GEN[other_bit](rng)
        sep = rng.choice((" ", "&" + rng.choice(PARAM_NAMES) + "=", "; ", "|", "/"))
        return payload + sep.encode("latin-1", "ignore") + raw, mask | m2
    if choice == "nul":
        pos = rng.randint(0, len(payload))
        return payload[:pos] + b"\x00" + payload[pos:], mask | M_MALFORMED
    if choice == "ctrl":
        pos = rng.randint(0, len(payload))
        c = bytes((rng.choice((0x01, 0x07, 0x08, 0x0B, 0x1B)),))
        return payload[:pos] + c + payload[pos:], mask | M_MALFORMED
    if choice == "overlong":
        text = payload.decode("utf-8")
        return _overlong_utf8(text[: rng.randint(1, len(text))]) + payload, mask | M_OBFUSCATION | M_MALFORMED
    # obf: wrap whole payload with an obfuscation technique
    text = payload.decode("utf-8")
    return obfuscate(text, rng, mask)


# --------------------------------------------------------------------------
# Carrier embedding
# --------------------------------------------------------------------------

def _rand_params(rng: random.Random, n: int) -> List[Tuple[str, str]]:
    # Surrounding parameters drawn from the shared pools (same for benign and attack).
    return [(rng.choice(PARAM_NAMES), benign.query_value(rng)) for _ in range(n)]


_URLISH = frozenset(("query", "form", "cookie", "path", "header:referer"))
_MUST_ENCODE = frozenset(b"%&=#+;")


def _has_valid_escape(b: bytes) -> bool:
    hexd = b"0123456789abcdefABCDEF"
    return any(b[j] == 0x25 and b[j + 1] in hexd and b[j + 2] in hexd for j in range(len(b) - 2))


def embed_many(rng: random.Random, values: List[bytes], mask: int, is_benign: bool = False,
               allow_encode: bool = True) -> Tuple[List[bytes], str]:
    """Wrap each value in ONE shared carrier context (same surroundings for all).

    The carrier set, parameter names, cookie names, JSON keys, hosts and
    surrounding text are identical for benign values and attack payloads.
    Single-layer transport encoding never adds a label bit; it is skipped for
    malformed values (it would launder the bad bytes) and for values that
    already contain a %XX escape (re-encoding them would be double encoding).
    """
    malformed = bool(mask & M_MALFORMED)
    can_text = all(valid_utf8(v) and not has_low_control(v) for v in values)
    can_encode = allow_encode and not malformed and not any(_has_valid_escape(v) for v in values)

    carriers = ["bare", "query", "query", "form", "cookie", "header", "header", "path", "prose", "prose"]
    if can_text and not (mask & M_HEADER):
        carriers += ["json", "json"]
    carrier = rng.choice(carriers)
    hkind = rng.choice(("ua", "referer", "xff", "cookie")) if carrier == "header" else ""
    urlish = carrier in _URLISH or hkind == "referer"
    plus = rng.random() < 0.5 and carrier != "path"
    if urlish and can_encode:
        if is_benign:
            # a real client encodes reserved characters; raw spaces/non-ASCII sometimes survive in logs
            needs = any(c in _MUST_ENCODE for v in values for c in v)
            encode = needs or rng.random() < 0.7
        else:
            encode = rng.random() < 0.5
    else:
        encode = False

    def enc(v: bytes) -> bytes:
        return pct_encode(v, space_as_plus=plus) if encode else v

    name = "header:" + hkind if hkind else carrier
    outs: List[bytes] = []

    if carrier == "bare":
        return list(values), name

    if carrier in ("query", "form"):
        params = _rand_params(rng, rng.choices((0, 1, 2, 3, 4, 5), (20, 25, 22, 15, 10, 8))[0])
        key = rng.choice(PARAM_NAMES).encode()
        at = rng.randint(0, len(params))
        path = benign.url_path(rng, rng.randint(0, 3)).encode() if carrier == "query" else b""
        for v in values:
            pairs = [k.encode() + b"=" + pv.encode() for k, pv in params]
            pairs.insert(at, key + b"=" + enc(v))
            body = b"&".join(pairs)
            outs.append(path + b"?" + body if carrier == "query" else body)
        return outs, name

    if carrier == "cookie" or hkind == "cookie":
        cname = rng.choice(benign.COOKIE_NAMES).encode()
        extras = [b.encode() for b in benign.cookie_header(rng).split("; ")][:rng.randint(0, 3)]
        at = rng.randint(0, len(extras))
        for v in values:
            parts = list(extras)
            parts.insert(at, cname + b"=" + enc(v))
            outs.append(b"; ".join(parts))
        return outs, name

    if hkind == "ua":
        ua = benign.user_agent(rng).encode()
        mode = rng.random()
        for v in values:
            if mode < 0.5 and b")" in ua:
                i = ua.index(b")")
                outs.append(ua[:i] + b"; " + v + ua[i:])
            elif mode < 0.8:
                outs.append(ua + b" " + v)
            else:
                outs.append(v + b" " + ua)
        return outs, name

    if hkind == "referer":
        base = ("https://" + benign.neutral_host(rng) + benign.url_path(rng)).encode()
        key = rng.choice(PARAM_NAMES).encode()
        others = b"".join(b"&" + k.encode() + b"=" + pv.encode() for k, pv in _rand_params(rng, rng.randint(0, 2)))
        return [base + b"?" + key + b"=" + enc(v) + others for v in values], name

    if hkind == "xff":
        ips = [benign.ipv4(rng).encode() for _ in range(rng.randint(1, 3))]
        at = rng.randint(0, len(ips))
        for v in values:
            lst = list(ips)
            lst.insert(at, v)
            outs.append(b", ".join(lst))
        return outs, name

    if carrier == "path":
        segs = benign.url_path(rng, rng.randint(1, 4)).strip("/").split("/")
        at = rng.randint(0, len(segs))
        tail = ("?" + benign.query_string(rng)).encode() if rng.random() < 0.25 else b""
        for v in values:
            parts = [s.encode() for s in segs]
            parts.insert(at, enc(v))
            outs.append(b"/" + b"/".join(parts) + tail)
        return outs, name

    if carrier == "prose":
        text = benign.english_sentence(rng) if rng.random() < 0.75 else benign.script_text(rng)
        words = text.split(" ")
        at = rng.randint(0, len(words))
        pre, post = " ".join(words[:at]), " ".join(words[at:])
        wrap = rng.choice(((b"", b""), (b"", b""), (b'"', b'"'), (b"'", b"'"), (b"(", b")"), (b"[", b"]"), (b": ", b"")))
        for v in values:
            s = (pre.encode() + b" " if pre else b"") + wrap[0] + v + wrap[1] + (b" " + post.encode() if post else b"")
            outs.append(s)
        return outs, name

    # json: same document shape for every value; only the injected string differs
    key = rng.choice(benign.JSON_KEYS)
    ensure_ascii = rng.random() < 0.4
    others = [(rng.choice(benign.JSON_KEYS), rng.choice((
        rng.randint(0, 9999), True, False, None, benign.query_value(rng), benign.name(rng),
        round(rng.random() * 100, 2)))) for _ in range(rng.randint(0, 3))]
    at = rng.randint(0, len(others))
    nest = rng.choice((None, None, "data", "params", "payload", "user", "item"))
    seps = rng.choice(((",", ":"), (", ", ": ")))
    for v in values:
        items = list(others)
        items.insert(at, (key, v.decode("utf-8")))
        doc = dict(items)
        if key not in doc or doc[key] != v.decode("utf-8"):
            doc[key] = v.decode("utf-8")
        if nest:
            doc = {nest: doc}
        outs.append(json.dumps(doc, ensure_ascii=ensure_ascii, separators=seps).encode("utf-8"))
    return outs, "json"


# --------------------------------------------------------------------------
# Benign generators -> (bytes, 0)
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Hard negatives -> (bytes, 0)
# --------------------------------------------------------------------------

def _hard_negative(rng: random.Random) -> bytes:
    g = rng.randint(0, 15)
    if g == 0:  # prose with SQL words
        return rng.choice((
            "Please select one item from the list and drop me a line.",
            "Union members should order lunch where 1 of 5 options applies.",
            "We will update the table and delete the old draft tomorrow.",
            "Insert the key, then select your language from the menu.",
            "Drop by the office; the group meets where the project started.",
        )).encode()
    if g == 1:  # names with apostrophes
        return rng.choice((
            "O'Brien and d'Angelo met at O'Reilly's shop.",
            "Contact: Sinéad O'Connor, d'Artagnan, N'Golo.",
            "The order for O'Malley's café is ready.",
        )).encode("utf-8")
    if g == 2:  # contractions
        return rng.choice((
            "I can't believe it won't work; let's don't worry.",
            "They're sure it isn't ready, but we'll see what's next.",
            "You've got what's needed; it'd be fine if we're late.",
        )).encode()
    if g == 3:  # math / comparisons
        return rng.choice((
            "a < b and c > d, so x=1 and 1+1=2.",
            "50% off today, 100% satisfaction, price = $42.",
            "if x > 0 and y < 10 then z = x + y.",
            "ratio a/b where a=3, b=4, result < 1.",
        )).encode()
    if g == 4:  # quoted speech
        return rng.choice((
            'She said, "meet me where the river bends," and left.',
            'He wrote: "select the best option" in the margin.',
            '"Drop the act," they laughed, "and join the union."',
        )).encode()
    if g == 5:  # ordinary file paths
        return rng.choice((
            "./docs/readme.md", "/usr/local/bin/tool", "C:\\Users\\me\\file.txt",
            "src/main/java/App.java", "~/projects/app/index.html",
            "./build/output/app.log", "/var/data/report-2024.csv",
        )).encode()
    if g == 6:  # version strings with ..
        return rng.choice((
            "supports versions 1.2..3.4 of the spec",
            "range 10..20 inclusive, step 2",
            "see pages 5...12 for details",
            "v1.0.0..v2.0.0 changelog",
            "chapters 3..7 cover the basics",
        )).encode()
    if g == 7:  # shell-like words in prose
        return rng.choice((
            "I love cat pictures; ping me when you can.",
            "Who am I to judge? Just echo my thoughts back.",
            "Let's grep through the notes and tail the summary.",
            "The cat sat by the shell on the beach.",
        )).encode()
    if g == 8:  # currency / amounts
        return rng.choice((
            "Total: $100.00, tax $8.50, tip $15.",
            "Budget is $1,250 for Q3 and $980 for Q4.",
            "Refund of $42.99 processed on Nov 3.",
        )).encode()
    if g == 9:  # hashtags / css colors
        return rng.choice((
            "Loving the new look #design #ui color #fff on #000 background",
            "background:#1a2b3c; color:#fff; border:#ccc",
            "#throwback to #summer2024 vibes",
        )).encode()
    if g == 10:  # email headers quoted in single-line prose
        return rng.choice((
            "He forwarded it with Subject: Re: lunch and To: team@list.",
            "The header said From: ops and Date: Monday.",
            "Set the Reply-To: support line before sending.",
        )).encode()
    if g == 11:  # programming discussion naming tags/functions
        return rng.choice((
            "Use the script tag to load the bundle, then call init().",
            "The img element needs an alt attribute for accessibility.",
            "Wrap it in a div and style the body selector.",
            "The select element fires an onchange event in the browser.",
        )).encode()
    if g == 12:  # harmless inline formatting
        return rng.choice((
            "This is <b>bold</b> and this is <i>italic</i> text.",
            "Line one<br>line two<br>line three.",
            "Click <a>here</a> for the summary below.",
            "Use <code>x+1</code> in the example.",
        )).encode()
    if g == 13:  # regex snippets
        return rng.choice((
            "pattern: ^[a-z0-9._%+-]+@[a-z0-9.-]+\\.[a-z]{2,}$",
            "match /\\d{3}-\\d{4}/ for phone numbers",
            "use \\b(foo|bar)\\b to find whole words",
            "strip with s/\\s+/ /g before parsing",
        )).encode()
    if g == 14:  # semicolon / pipe separated benign lists
        return rng.choice((
            "apples; oranges; bananas; pears; grapes",
            "red | green | blue | yellow | orange",
            "name; email; phone; city; status",
            "step 1; step 2; step 3; done",
        )).encode()
    # g == 15: comparisons & relative refs mixing < > =
    return rng.choice((
        "Compare: 3 < 5, 10 > 2, 7 = 7, and 4 != 5.",
        "The function f(x) = x^2 grows when x > 1.",
        "Set width=100 height=50 and margin=0 in config.",
        "temperature >= 20C and humidity <= 60% today",
    )).encode()


class Record(NamedTuple):
    payload: bytes
    mask: int
    kind: str     # benign | hardneg | clean_pair | plain_pair | corrupt | obf_pair | attack
    pair: int     # id shared by the halves of a pair, else -1
    carrier: str  # carrier name, or "family:<name>" for standalone benign text


P_BENIGN_DRAW = 0.46     # pair halves add the rest of the benign mass (~50% overall)
P_HARDNEG = 0.20
P_BENIGN_CARRIER = 0.38
P_CORRUPT = 0.80         # share of malformed draws that are corruptions of a benign sample
P_ENC_PAIR = 0.45        # share of obfuscation draws that are benign encoding pairs
P_COMBINE = 0.25         # multi-label combiner rate for ordinary attack draws


def _benign_record(rng: random.Random) -> Tuple[bytes, str, str]:
    """One benign sample that passes benign.benign_ok(): (bytes, kind, carrier)."""
    for _ in range(64):
        r = rng.random()
        if r < P_HARDNEG:
            b, kind, car = _hard_negative(rng), "hardneg", "family:hardneg"
        elif r < P_HARDNEG + P_BENIGN_CARRIER:
            v = benign.benign_value(rng).encode("utf-8")
            (b,), car = embed_many(rng, [v], 0, is_benign=True)
            kind = "benign"
        else:
            b, fam = benign.sample(rng)
            kind, car = "benign", "family:" + fam
        if benign.benign_ok(b):
            return b, kind, car
    return b"ok", "benign", "family:fallback"


def _corruption_pair(rng: random.Random, pid: int) -> List[Record]:
    for _ in range(32):
        if rng.random() < 0.5:
            clean, _, car = _benign_record(rng)
        else:
            clean, car = benign.corruption_base(rng), "family:corruption_base"
            if not benign.benign_ok(clean):
                continue
        if len(clean) > MAX_LEN - 16:
            continue
        bad = benign.corrupt(rng, clean)
        if len(bad) <= MAX_LEN and benign.is_malformed(bad):
            out = [Record(bad, M_MALFORMED, "corrupt", pid, car)]
            if rng.random() < 0.95:
                out.append(Record(clean, 0, "clean_pair", pid, car))
            return out
    return []


def _encoding_pair(rng: random.Random, pid: int) -> List[Record]:
    """Benign value in three forms: plain, single-layer encoded (benign), and
    double/%u/entity-run/escape-run encoded (obfuscation bit only)."""
    for _ in range(16):
        text = benign.pair_source_value(rng)
        raw = text.encode("utf-8")
        single = pct_encode(raw, space_as_plus=rng.random() < 0.5)
        tech = rng.choice(("double", "double", "triple", "u", "entity", "escape"))
        if tech in ("double", "triple"):
            first = single if b"%" in single else b"".join(b"%%%02X" % c for c in raw)
            obf = pct_encode(first)
            if tech == "triple":
                obf = pct_encode(obf)
        elif tech == "u":
            obf = _u_encode(text)
        elif tech == "entity":
            obf = _entity_run(text, rng).encode("ascii")
        else:
            obf = _escape_run(text, rng).encode("ascii")
        (o_obf, o_single, o_plain), car = embed_many(rng, [obf, single, raw], M_OBFUSCATION, allow_encode=False)
        if len(o_obf) > MAX_LEN:
            continue
        out = [Record(o_obf, M_OBFUSCATION, "obf_pair", pid, car)]
        plain_ok = car in ("bare", "prose", "json", "header:ua") and not any(c in b"%&=#+;" for c in raw)
        prefer_plain = tech in ("entity", "escape")
        cands = [o_plain, o_single] if (prefer_plain and plain_ok) else [o_single, o_plain] if plain_ok else [o_single]
        for cand in cands:
            if len(cand) <= MAX_LEN and benign.benign_ok(cand):
                out.append(Record(cand, 0, "plain_pair", pid, car))
                break
        return out
    return []


def _draw(rng: random.Random, pid: int) -> List[Record]:
    if rng.random() < P_BENIGN_DRAW:
        b, kind, car = _benign_record(rng)
        return [Record(b, 0, kind, -1, car)]
    for _ in range(64):
        bit = rng.choice(_ATTACK_BITS)
        if bit == M_MALFORMED and rng.random() < P_CORRUPT:
            recs = _corruption_pair(rng, pid)
            if recs:
                return recs
            continue
        if bit == M_OBFUSCATION and rng.random() < P_ENC_PAIR:
            recs = _encoding_pair(rng, pid)
            if recs:
                return recs
            continue
        payload, mask = CAT_GEN[bit](rng)
        if rng.random() < P_COMBINE:
            payload, mask = combine(rng, payload, mask)
        (sample,), car = embed_many(rng, [payload], mask)
        if 1 <= len(sample) <= MAX_LEN:
            return [Record(sample, mask, "attack", -1, car)]
        # too long: regenerate so labels stay truthful
    b, kind, car = _benign_record(rng)
    return [Record(b, 0, kind, -1, car)]


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def generate_records(seed: int, count: int) -> Iterator[Record]:
    """Yield `count` deterministic Records (payload, mask, kind, pair id, carrier).

    The second half of a pair is queued and released a few samples later.
    """
    rng = random.Random(seed)
    pending: deque = deque()
    pid = 0
    emitted = 0
    while emitted < count:
        if pending and rng.random() < 0.4:
            yield pending.popleft()
            emitted += 1
            continue
        recs = _draw(rng, pid)
        pid += 1
        if not recs:
            continue
        yield recs[0]
        emitted += 1
        pending.extend(recs[1:])


def generate(seed: int, count: int) -> Iterator[Tuple[bytes, int]]:
    """Yield `count` deterministic (payload_bytes, label_mask) samples."""
    for rec in generate_records(seed, count):
        yield rec.payload, rec.mask


def write_corpus(path: str, samples: Iterable[Tuple[bytes, int]]) -> int:
    """Write records: u32 LE length, u8 mask, payload. Returns record count."""
    n = 0
    header = struct.Struct("<IB")
    with open(path, "wb") as f:
        for payload, mask in samples:
            if not isinstance(payload, (bytes, bytearray)):
                payload = _as_bytes(payload)
            f.write(header.pack(len(payload), mask & 0xFF))
            f.write(payload)
            n += 1
    return n


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="string2fingerprint corpus generator")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--count", type=int, default=400000)
    ap.add_argument("--out", type=str, default="corpus.bin")
    args = ap.parse_args(argv)
    n = write_corpus(args.out, generate(args.seed, args.count))
    print("wrote %d records to %s" % (n, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
