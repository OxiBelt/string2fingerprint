"""More benign text: markup, queries, commands, documents, and hard negatives.

Standard library only. Every generator takes a `random.Random`, draws all of
its randomness from it, and returns text, so output is deterministic for a
seed. `benign.py` registers the generators here as benign families, and
`corpus.py` draws its hard negatives from `hard_negative()`.

The families here are the ordinary forms of the syntax that attacks abuse:
safe HTML with links and images, parameterized SQL, command lines in
documentation, template placeholders, configuration documents, GraphQL,
and relative imports. Each sample must still pass `benign.benign_ok()`, so a
generator here never emits a breakout, an event handler, a command after a
separator, or any other form that the attack grammars produce.
"""

from __future__ import annotations

import random
from typing import Callable, List, Tuple

import benign

_LOWER = "abcdefghijklmnopqrstuvwxyz"


def _word(rng: random.Random) -> str:
    return rng.choice(benign.IDENT_WORDS)


def _words(rng: random.Random, lo: int = 1, hi: int = 5) -> str:
    return " ".join(rng.choice(benign.VOCAB) for _ in range(rng.randint(lo, hi)))


def _snake(rng: random.Random) -> str:
    return _word(rng) + rng.choice(("", "_" + _word(rng), "_id", "_at", "s"))


def _https(rng: random.Random) -> str:
    return "https://" + benign.neutral_host(rng) + benign.url_path(rng, rng.randint(0, 3))


# --------------------------------------------------------------------------
# Safe HTML
# --------------------------------------------------------------------------

_BLOCK_TAGS = ("p", "div", "section", "article", "li", "blockquote", "h2", "h3", "td", "figcaption", "dd")
_INLINE_TAGS = ("b", "i", "em", "strong", "span", "small", "code", "mark", "abbr", "sub", "sup", "u", "kbd")


def _attrs(rng: random.Random) -> str:
    out = []
    for _ in range(rng.randint(0, 2)):
        out.append(rng.choice((
            f'class="{_word(rng)}-{_word(rng)}"', f'id="{_word(rng)}{rng.randint(1, 99)}"',
            f'title="{_words(rng, 1, 3)}"', f'lang="{rng.choice(("en", "ko", "ja", "de", "fr"))}"',
            f'data-{_word(rng)}="{rng.randint(0, 999)}"', f'style="color:#{benign.rand_hex(rng, 6)};margin:{rng.randint(0, 20)}px"',
            f'dir="{rng.choice(("ltr", "rtl", "auto"))}"', f'aria-label="{_words(rng, 1, 3)}"',
        )))
    return "".join(" " + a for a in out)


def html_fragment(rng: random.Random) -> str:
    """A fragment of rich text as an editor would submit it."""
    parts = []
    for _ in range(rng.randint(1, 3)):
        kind = rng.randint(0, 7)
        text = benign.english_sentence(rng) if rng.random() < 0.7 else benign.script_text(rng)[:80]
        text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if kind == 0:
            words = text.split(" ")
            i = rng.randrange(len(words))
            tag = rng.choice(_INLINE_TAGS)
            words[i] = f"<{tag}>{words[i]}</{tag}>"
            tag = rng.choice(_BLOCK_TAGS)
            parts.append(f"<{tag}{_attrs(rng)}>{' '.join(words)}</{tag}>")
        elif kind == 1:
            extra = rng.choice(("", ' target="_blank"', ' rel="noopener"'))
            parts.append(f'<a href="{_https(rng)}"{extra}>{_words(rng, 1, 4)}</a>')
        elif kind == 2:
            src = rng.choice((_https(rng) + rng.choice((".png", ".jpg", ".webp")), "/images/" + benign.filename(rng)))
            extra = rng.choice(("", ' width="%d"' % rng.randint(16, 900), ' loading="lazy"'))
            parts.append(f'<img src="{src}" alt="{_words(rng, 1, 4)}"{extra}>')
        elif kind == 3:
            items = "".join(f"<li>{_words(rng, 1, 4)}</li>" for _ in range(rng.randint(2, 4)))
            tag = rng.choice(("ul", "ol"))
            parts.append(f"<{tag}{_attrs(rng)}>{items}</{tag}>")
        elif kind == 4:
            cells = "".join(f"<td>{rng.choice((_word(rng), str(rng.randint(0, 9999)), benign.money(rng)))}</td>" for _ in range(rng.randint(2, 4)))
            parts.append(f"<table><tr>{cells}</tr></table>")
        elif kind == 5:
            parts.append(f"<p>{text}<br>{_words(rng, 2, 6)}</p>")
        elif kind == 6:
            parts.append(f"<pre><code>{benign.code_line(rng).replace('<', '&lt;').replace('>', '&gt;')}</code></pre>")
        else:
            level = rng.randint(1, 4)
            parts.append(f"<h{level}>{benign.title_line(rng)[:60]}</h{level}>")
    return "".join(parts)


# --------------------------------------------------------------------------
# Parameterized SQL, as written in code and documentation
# --------------------------------------------------------------------------

def sql_query(rng: random.Random) -> str:
    t = _snake(rng) + rng.choice(("", "s"))
    cols = ", ".join(_snake(rng) for _ in range(rng.randint(1, 4)))
    ph = rng.choice(("?", "$1", ":id", "%s", "@id", ":" + _word(rng)))
    c = _snake(rng)
    kind = rng.randint(0, 9)
    if kind == 0:
        return f"SELECT {cols} FROM {t} WHERE {c} = {ph}"
    if kind == 1:
        return f"select {cols} from {t} where {c} = {ph} order by {_snake(rng)} {rng.choice(('asc', 'desc'))} limit {rng.randint(1, 100)}"
    if kind == 2:
        return f"INSERT INTO {t} ({cols}) VALUES ({', '.join([ph] * (cols.count(',') + 1))})"
    if kind == 3:
        return f"UPDATE {t} SET {c} = {ph} WHERE id = {rng.choice(('?', '$2', ':id'))}"
    if kind == 4:
        return f"DELETE FROM {t} WHERE {c} < NOW() - INTERVAL '{rng.randint(1, 90)} days'"
    if kind == 5:
        return f"SELECT COUNT(*) FROM {t} GROUP BY {c} HAVING COUNT(*) > {rng.randint(1, 50)}"
    if kind == 6:
        u = _snake(rng) + "s"
        return f"SELECT a.{c}, b.{_snake(rng)} FROM {t} a JOIN {u} b ON b.{t[:-1] if t.endswith('s') else t}_id = a.id"
    if kind == 7:
        return f"CREATE INDEX idx_{t}_{c} ON {t} ({c});"
    if kind == 8:
        return f"WHERE {c} LIKE {ph} AND {_snake(rng)} IS NOT NULL"
    return f"SELECT * FROM {t} WHERE {c} IN ({', '.join(str(rng.randint(1, 999)) for _ in range(rng.randint(2, 5)))})"


# --------------------------------------------------------------------------
# Command lines in documentation, without an injection separator
# --------------------------------------------------------------------------

def command_line(rng: random.Random) -> str:
    pkg = rng.choice(("requests", "numpy", "lodash", "react", "express", "serde", "tokio", "pytest", "flask", "axios"))
    ver = "%d.%d.%d" % (rng.randint(0, 9), rng.randint(0, 30), rng.randint(0, 20))
    branch = rng.choice(("main", "develop", "feature/" + _word(rng), "fix/" + _word(rng)))
    cmd = rng.choice((
        f"npm install {pkg}@{ver}", f"pip install {pkg}=={ver}", f"cargo add {pkg}", f"yarn add {pkg}",
        f"git checkout -b {branch}", f"git commit -m \"{_words(rng, 2, 5)}\"", f"git push origin {branch}",
        f"git log --oneline -{rng.randint(2, 20)}", f"docker run -p {rng.randint(3000, 9000)}:80 {_word(rng)}:{ver}",
        f"docker compose up -d {_word(rng)}", f"kubectl get pods -n {_word(rng)}", f"make {rng.choice(('build', 'test', 'release', 'clean'))}",
        f"python -m venv .venv", f"cargo test --release", f"go test ./...", f"terraform plan -out {_word(rng)}.plan",
        f"brew install {pkg}", f"apt-get install -y {_word(rng)}", f"mkdir -p {_word(rng)}/{_word(rng)}",
        f"tar -xzf {_word(rng)}-{ver}.tar.gz", f"ssh {_word(rng)}@{benign.neutral_host(rng)}",
        f"scp report.pdf {_word(rng)}@{benign.neutral_host(rng)}:", f"grep -rn \"{_word(rng)}\" src/",
        f"cd {_word(rng)} && npm run {rng.choice(('dev', 'build', 'lint'))}", f"export {_word(rng).upper()}_PORT={rng.randint(1000, 9999)}",
        f"rsync -av {_word(rng)}/ backup/", f"node {_word(rng)}.js --port {rng.randint(1000, 9999)}",
        f"ffmpeg -i {_word(rng)}.mov -vf scale={rng.choice((640, 1280))}:-1 out.mp4",
    ))
    return rng.choice((cmd, cmd, "$ " + cmd, f"Run `{cmd}` to {_words(rng, 2, 4)}.", f"Then type {cmd} and press Enter."))


# --------------------------------------------------------------------------
# Placeholders that templates and translations use without an expression
# --------------------------------------------------------------------------

def placeholder_text(rng: random.Random) -> str:
    name = _word(rng)
    ph = rng.choice((f"{{{name}}}", "%s", "%d", f"%({name})s", f":{name}", f"{{0}}", f"<{name}>", "{count}", f"[{name.upper()}]", "%1$s"))
    sentence = rng.choice((
        f"Hello {ph}, your order is ready.", f"{ph} items in your cart", f"Welcome back, {ph}!",
        f"You have {ph} new messages", f"Deleted {ph} of {{total}} files", f"Error: {ph} is required",
        f"{benign.script_text(rng)[:30]} {ph}",
    ))
    return sentence


# --------------------------------------------------------------------------
# Configuration documents and data formats
# --------------------------------------------------------------------------

def yaml_doc(rng: random.Random) -> str:
    lines = [f"{_snake(rng)}:"]
    for _ in range(rng.randint(1, 4)):
        v = rng.choice((str(rng.randint(0, 9999)), rng.choice(("true", "false", "null")), _word(rng),
                        f"\"{_words(rng, 1, 4)}\"", benign.neutral_host(rng), f"[{_word(rng)}, {_word(rng)}]"))
        lines.append(f"  {_snake(rng)}: {v}")
    return "\n".join(lines)


def toml_doc(rng: random.Random) -> str:
    out = [f"[{_word(rng)}]"]
    for _ in range(rng.randint(1, 4)):
        out.append(f"{_snake(rng)} = " + rng.choice((str(rng.randint(0, 9999)), f"\"{_word(rng)}\"", "true", f"[\"{_word(rng)}\", \"{_word(rng)}\"]")))
    return "\n".join(out)


def ini_line(rng: random.Random) -> str:
    return rng.choice((f"{_snake(rng)}={rng.randint(0, 9999)}", f"{_snake(rng)} = {_word(rng)}", f"[{_word(rng)}:{_word(rng)}]",
                       f"; {_words(rng, 2, 6)}", f"# {_words(rng, 2, 6)}", f"{_word(rng).upper()}_URL={_https(rng)}"))


def xml_doc(rng: random.Random) -> str:
    root = _word(rng)
    inner = "".join(f"<{t}>{rng.choice((_words(rng, 1, 3), str(rng.randint(0, 999))))}</{t}>"
                    for t in (_word(rng) for _ in range(rng.randint(1, 3))))
    attrs = rng.choice(("", f' id="{rng.randint(1, 999)}"', f' version="{rng.randint(1, 3)}.0"', f' xmlns="{_https(rng)}"'))
    return f"<{root}{attrs}>{inner}</{root}>"


def graphql_query(rng: random.Random) -> str:
    fields = " ".join(_snake(rng) for _ in range(rng.randint(1, 4)))
    arg = rng.choice((f"id: {rng.randint(1, 9999)}", f'id: "{benign.rand_alnum(rng, 8)}"', f"first: {rng.randint(1, 50)}",
                      f'name: "{_word(rng)}"', "id: $id"))
    op = rng.choice(("query", "query", "mutation", ""))
    name = rng.choice(("", " " + _word(rng).capitalize() + "Query"))
    return f"{op}{name} {{ {_word(rng)}({arg}) {{ {fields} }} }}".strip()


def relative_reference(rng: random.Random) -> str:
    """An ordinary relative import or asset path that climbs one or two levels."""
    up = "../" * rng.choice((1, 1, 2))
    mod = f"{_word(rng)}/{_word(rng)}" if rng.random() < 0.5 else _word(rng)
    asset = f"{_word(rng)}.{rng.choice(('png', 'svg', 'jpg', 'css', 'woff2'))}"
    return rng.choice((
        f"import {{ {_word(rng).capitalize()} }} from '{up}{mod}';", f"import {_word(rng)} from \"{up}{mod}\";",
        f"const {_word(rng)} = require('{up}{mod}');", f"background: url({up}images/{asset});",
        f"@import '{up}styles/{_word(rng)}.css';", f'<a href="{up}{_word(rng)}.html">{_words(rng, 1, 3)}</a>',
        f"from ..{_word(rng)} import {_word(rng)}", f"use super::{_word(rng)};", f"#include \"{up}{_word(rng)}.h\"",
        f"see {up}docs/{_word(rng)}.md for details",
    ))


def base64_blob(rng: random.Random) -> str:
    n = rng.choice((16, 24, 32, 44, 64, 88, 120))
    return rng.choice((benign.rand_b64(rng, n), benign.rand_b64(rng, n, url=True), "data:image/png;base64," + benign.rand_b64(rng, n)))


def jwt(rng: random.Random) -> str:
    return "eyJ" + benign.rand_b64(rng, rng.randint(15, 40), url=True) + ".eyJ" + benign.rand_b64(rng, rng.randint(30, 120), url=True) + "." + benign.rand_b64(rng, 43, url=True)


def uuid_or_hash(rng: random.Random) -> str:
    h = benign.rand_hex
    return rng.choice((f"{h(rng, 8)}-{h(rng, 4)}-4{h(rng, 3)}-a{h(rng, 3)}-{h(rng, 12)}", h(rng, 32), h(rng, 40), h(rng, 64),
                       "sha256:" + h(rng, 64), f"W/\"{h(rng, 16)}\""))


def windows_text(rng: random.Random) -> str:
    """Windows paths and commands as they appear in support text."""
    return rng.choice((
        f"C:\\Program Files\\{_word(rng).capitalize()}\\{_word(rng)}.exe", f"%APPDATA%\\{_word(rng).capitalize()}\\config.json",
        f"Open C:\\Users\\{_word(rng)}\\Documents\\{benign.filename(rng)} and save it.", f"D:\\{_word(rng)}\\{_word(rng)}_{rng.randint(1, 99)}.xlsx",
        f"HKEY_CURRENT_USER\\Software\\{_word(rng).capitalize()}", f"\\\\{_word(rng)}\\share\\{_word(rng)}\\{benign.filename(rng)}",
    ))


FAMILIES: List[Tuple[str, Callable[[random.Random], str], int]] = [
    ("html", html_fragment, 4),
    ("sql_query", sql_query, 2),
    ("command_line", command_line, 2),
    ("placeholder", placeholder_text, 1),
    ("yaml", yaml_doc, 1),
    ("toml", toml_doc, 1),
    ("ini", ini_line, 1),
    ("xml", xml_doc, 1),
    ("graphql", graphql_query, 1),
    ("relative_ref", relative_reference, 1),
    ("base64", base64_blob, 1),
    ("jwt", jwt, 1),
    ("uuid", uuid_or_hash, 1),
    ("windows_text", windows_text, 1),
]


# --------------------------------------------------------------------------
# Hard negatives: benign text that shares vocabulary or punctuation with attacks
# --------------------------------------------------------------------------

_SQL_WORDS = ("select", "drop", "union", "update", "insert", "delete", "where", "order", "join", "table", "from",
              "having", "group", "exec", "declare", "sleep", "benchmark", "like", "null", "into")
_SHELL_WORDS = ("cat", "ping", "echo", "grep", "tail", "shell", "kill", "touch", "sleep", "find", "curl", "id", "env",
                "nc", "bash", "dir", "type", "set", "net", "time")
_MARKUP_WORDS = ("script", "alert", "img", "onload", "iframe", "svg", "body", "style", "document", "cookie", "prompt",
                 "confirm", "frame", "object", "embed")
_PATH_WORDS = ("etc", "passwd", "root", "home", "proc", "system", "windows", "config", "boot", "hosts")
_TEMPLATE_WORDS = ("config", "class", "globals", "template", "freemarker", "velocity", "jinja", "runtime", "exec")
_HEADER_WORDS = ("Location", "Set-Cookie", "Content-Length", "Host", "Subject", "Bcc", "Refresh", "Content-Type")


def _sentence_with(rng: random.Random, words: Tuple[str, ...]) -> str:
    s = benign.english_sentence(rng).rstrip(".!?")
    tokens = s.split(" ")
    for _ in range(rng.randint(1, 3)):
        w = rng.choice(words)
        w = w.capitalize() if rng.random() < 0.3 else w
        tokens.insert(rng.randint(0, len(tokens)), w)
    return " ".join(tokens) + rng.choice((".", ".", "!", "?"))


def _apostrophe_name(rng: random.Random) -> str:
    pre = rng.choice(("O'", "D'", "d'", "L'", "N'", "Mc", "Mac", "De'", "dell'"))
    last = rng.choice(benign.LAST_NAMES)
    first = rng.choice(benign.FIRST_NAMES)
    return rng.choice((f"{first} {pre}{last}", f"{pre}{last}'s {rng.choice(benign.NOUNS)}", f"Mr. {pre}{last}",
                       f"{first} {pre}{last} and {rng.choice(benign.FIRST_NAMES)} {pre}{rng.choice(benign.LAST_NAMES)}"))


def _comparison(rng: random.Random) -> str:
    a, b = rng.randint(0, 99), rng.randint(0, 99)
    v = rng.choice(("x", "y", "n", "count", "total", "price", "age", "score"))
    return rng.choice((
        f"{v} < {a} and {v} > {b}", f"{a} <= {v} <= {b}", f"if {v} >= {a}: {v} = {b}", f"{v} != {a} || {v} == {b}",
        f"{a}% off when {v} > {b}", f"where {v} = {a} or {v} = {b}", f"{v}={a}&{rng.choice(('sort', 'page'))}={b}",
        f"ratio {a}/{b} = {a / max(b, 1):.2f}", f"between {a} and {b} inclusive", f"temperature >= {a}C and humidity <= {b}%",
    ))


def _quoted(rng: random.Random) -> str:
    q = rng.choice(('"', "'", "\u201c"))
    close = "\u201d" if q == "\u201c" else q
    quote = _sentence_with(rng, _SQL_WORDS + _SHELL_WORDS).rstrip(".!?")
    return rng.choice((f"She said, {q}{quote},{close} and left.", f"{q}{quote}{close} {rng.choice(('he wrote', 'they replied', 'reads the sign'))}.",
                       f"The note says {q}{quote}{close}"))


def _separators(rng: random.Random) -> str:
    sep = rng.choice(("; ", " | ", " & ", " && ", ", ", " / "))
    items = [rng.choice(benign.NOUNS) for _ in range(rng.randint(3, 6))]
    return sep.join(items)


def _code_with_operators(rng: random.Random) -> str:
    a, b = _word(rng), _word(rng)
    return rng.choice((
        f"if ({a} && !{b}) {{ return; }}", f"{a} || {b} || []", f"{a} = {b} ?? '{_word(rng)}';",
        f"cond ? '{_word(rng)}' : \"{_word(rng)}\"", f"[{a}, {b}].join('; ')", f"echo \"{_words(rng, 1, 3)}\"",
        f"`{a}` and `{b}` are {rng.choice(('equal', 'optional', 'deprecated'))}", f"user.name = '{_word(rng)}';",
        f"x = '{rng.choice(benign.LAST_NAMES)}' if ok else ''", f"{a}.replace('/', '\\\\')", f"query.where('{a}', '=', {rng.randint(1, 99)})",
    ))


def _header_prose(rng: random.Random) -> str:
    h = rng.choice(_HEADER_WORDS)
    return rng.choice((f"Set the {h} header before sending.", f"The {h}: value was missing in the reply.",
                       f"{h} is ignored by {_word(rng)} proxies.", f"Use {h} and {rng.choice(_HEADER_WORDS)} together."))


def _percent_prose(rng: random.Random) -> str:
    return rng.choice((f"{rng.randint(1, 100)}% of {rng.choice(benign.NOUNS)}s", f"growth of {rng.randint(1, 99)}.{rng.randint(0, 9)}%",
                       f"discount: {rng.randint(5, 70)}%!", f"{rng.randint(1, 9)}%-{rng.randint(10, 99)}% range"))


def _dots(rng: random.Random) -> str:
    a, b = rng.randint(1, 9), rng.randint(10, 30)
    return rng.choice((f"pages {a}..{b}", f"versions {a}.0..{a + 1}.0", f"wait... then {_word(rng)}", f"range({a}..={b})",
                       f"see chapter {a}...{b}", f"{_word(rng)}...{_word(rng)}"))


def _template_like(rng: random.Random) -> str:
    return rng.choice((f"price: ${rng.randint(1, 999)}.{rng.randint(0, 99):02d}", f"#{_word(rng)} #{_word(rng)}{rng.randint(1, 99)}",
                       f"{{\"{_word(rng)}\": {rng.randint(0, 99)}}}", f"<{_word(rng)}@{benign.neutral_host(rng)}>",
                       f"@{_word(rng)} mentioned you", f"<3 {_words(rng, 1, 3)}", f"[{_word(rng)}]({_https(rng)})"))


_HARD: Tuple[Tuple[Callable[[random.Random], str], int], ...] = (
    (lambda r: _sentence_with(r, _SQL_WORDS), 4),
    (lambda r: _sentence_with(r, _SHELL_WORDS), 3),
    (lambda r: _sentence_with(r, _MARKUP_WORDS), 3),
    (lambda r: _sentence_with(r, _PATH_WORDS), 2),
    (lambda r: _sentence_with(r, _TEMPLATE_WORDS), 2),
    (_apostrophe_name, 3), (_comparison, 3), (_quoted, 2), (_separators, 2), (_code_with_operators, 3),
    (_header_prose, 1), (_percent_prose, 1), (_dots, 1), (_template_like, 2),
    (html_fragment, 2), (sql_query, 2), (command_line, 2), (relative_reference, 1), (placeholder_text, 1),
)


def hard_negative(rng: random.Random) -> str:
    fns = [f for f, _ in _HARD]
    weights = [w for _, w in _HARD]
    return rng.choices(fns, weights)[0](rng)
