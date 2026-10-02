#!/usr/bin/env python3
"""Architecture ratchet: stop six measured cross-title costs from growing.

Why this exists
---------------
Some costs are paid by every title and keep growing because nothing stops them: title ids and
title-named modules inside shared code, blocking GPU syncs, raw `getenv` reads scattered through
hot code, a few source files that only ever get longer, and the shipping frontend depending on the
test namespace. None of them is a bug on the day it lands, so review waves each one through, and
the total only rises. This gate turns each into a per-file count that may go DOWN, never UP.

The rules (every count is per file, over tracked files only, with comments stripped)
--------------------------------------------------------------------------------------
  title-id       `PPSA#####` / `CUSA#####` in code under prosper/src and prosper/frontends.
                 String literals COUNT -- a title id in a string is still title-specific code.
                 Comments do not: a comment naming the capture a fact came from is fine.
  title-dir      a directory under prosper/src whose basename is a title slug. The row's value is
                 the total line count of the C/C++ sources in that directory, which may shrink but
                 not grow. Slugs are DERIVED from the per-title route folders in prosper/scripts/
                 (`gta5`, `sonic-frontiers-PPSA03831`, ...): the folder name, minus any trailing
                 title id, in its `-`, `_` and run-together spellings; plus any `ppsa#####` /
                 `cusa#####` basename.
  getenv         `getenv(` / `std::getenv(` / `secure_getenv(` call sites in prosper/src,
                 prosper/frontends and prosper/tests/fixtures/render_runner.h.
  blocking-sync  vkWaitForFences, vkQueueWaitIdle, vkDeviceWaitIdle,
                 VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, one row
                 per file per token, same roots as getenv.
  file-size      a C/C++ file under prosper/src, prosper/frontends or prosper/tests/fixtures longer
                 than FILE_SIZE_THRESHOLD lines may not exceed its baselined line cap.
  test-dep       `prosper::test::` in code under prosper/frontends -- the shipping app depending on
                 the test namespace.

Verdicts
--------
  new        a key not in the baseline                      -> FAIL (fix it; see the rule's hint)
  increase   a count above its baseline row                 -> FAIL
  decrease   a count below its baseline row                 -> FAIL until the row is lowered
  stale      a baseline row whose key no longer reproduces  -> FAIL until the row is deleted

The last two exist for the same reason as in tools/env/check_diag_gates.py: a baseline nobody is
forced to update stops describing the tree, and an unlowered row is headroom the next change can
quietly spend. `--update` applies exactly those two repairs -- it lowers and deletes rows and NEVER
raises or adds one -- so recording an improvement costs one command.

The two LINE-COUNT rules (title-dir, file-size) are caps rather than exact counts: any growth past
the cap fails, but a shrink only reports `decrease` once it is more than SHRINK_MARGIN below the
cap. That band is the concurrency tradeoff. With an exact count, every edit to a 16,000-line file
-- among the most frequently edited files in the tree -- would rewrite the same baseline row, so
two unrelated lanes would conflict on the baseline even though neither grew anything. With the
band, ordinary edits touch the baseline only when they actually move the cap. The price: up to
SHRINK_MARGIN of slack that a later change may re-grow into without failing. The cap itself never
rises unless a reviewer reads a baseline diff that raises it.

The count rules use exact values and per-file (per-token) rows, so two lanes conflict on the
baseline only when both change the same count in the same file -- and then they genuinely did.

Raising a row is the escape hatch, as in every ratchet: a one-line baseline edit a reviewer sees,
with a `# note` saying why. A new blocking-sync site in particular must name the guest-visible
result it delivers, in a comment beside it and in the PR.

Two modes: delta (what CI and the agent hook run) and full (baseline maintenance)
--------------------------------------------------------------------------------
FULL mode (no `--base`) is everything above: the whole tree against every row. It is the right
tool for maintaining the baseline and the wrong one for gating a change, because the tree moves
under every row. A PR that grows a capped file merges, the row is now stale, and from then on a
full check fails for every unrelated change until somebody repairs a row they never touched. That
happened on the day this gate was written (#4173 grew two capped files while it was in review).

DELTA mode (`--base REF`) answers only "did THIS change make anything worse?". It takes the merge
base of HEAD and REF (so a branch behind REF is not blamed for what REF changed since), and looks
only at the files that differ between that merge base and the working tree -- committed, staged,
unstaged, and untracked-but-not-ignored, so an agent is told before `git add -A` rather than after.
For every key those files touch it computes the count at the merge base and in the working tree,
and FAILS only when the count rose AND the working tree's baseline row does not cover the new
value. A file that did not exist at the merge base is compared against zero. So:

  * a pre-existing over-row count in a file this change did not touch never fails delta mode;
  * a rise inside a capped file's existing headroom passes, exactly as it does in full mode;
  * the escape hatch still works: raise the row, with its note, in the same change.

A title-dir key aggregates a directory, so a change to any file inside one compares the whole
directory's total. A moved file shows up as a NEW key under its new path: rename its row in the
same change. Decreases are reported as a notice (lower the row with `--update` when you can) and
never fail delta mode -- requiring them would recreate the concurrency problem full mode has.
Base contents are read with one `git cat-file --batch`; nothing outside the changed set is read.

What this CANNOT see -- read before quoting a clean run
-------------------------------------------------------
  * It is a lexer, not a compiler. `#if 0` blocks count as live; a macro that expands to `getenv`
    counts once, at its definition, not at each use. Calls through other wrappers (`SDL_getenv`,
    `_wgetenv`, `GetEnvironmentVariable`, the cached `PROSPER_ENV_*` macros) are not counted.
  * Comment stripping understands `//` (including backslash-continued), `/* */`, string and char
    literals with escapes, raw strings (`R"d(...)d"` with any encoding prefix) and C++14 digit
    separators (`1'000`). It does not understand trigraphs or `#include <a//b>`.
  * blocking-sync, getenv and test-dep ignore string-literal contents (a log message saying
    "vkWaitForFences failed" is not a sync); title-id reads them.
  * A vkWaitForFences reached through a dispatch-table member (`d.vkWaitForFences`) counts; one
    through a differently named `PFN_vkWaitForFences` variable does not.
  * Only C-family suffixes are scanned (SOURCE_SUFFIXES). Shaders, assembly and Python are not.
  * The slug list is only as complete as prosper/scripts/. A title with no route folder is not a
    slug, so `src/.../<that-title>/` would not be caught.

Exit status: 0 clean, 1 violations found, 2 could not evaluate (no git, empty scan, unparseable
baseline, a sanity floor tripped, a `--base` that is not a commit here or shares no merge base
with HEAD). 2 is never "clean": a scan that saw nothing proves nothing.

`--base REF` runs delta mode; `--selftest` runs hand-built positive and negative arms; `--list`
prints every finding with line numbers; `--emit-baseline` prints the rows for the current tree.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_BASELINE = HERE / "arch_ratchet_baseline.txt"

SOURCE_SUFFIXES = (
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".inl",
    ".ipp",
    ".m",
    ".mm",
)

SRC = "prosper/src/"
FRONTENDS = "prosper/frontends/"
FIXTURES = "prosper/tests/fixtures/"
RENDER_RUNNER = "prosper/tests/fixtures/render_runner.h"
SCRIPTS = "prosper/scripts/"

TITLE_ID_ROOTS = (SRC, FRONTENDS)
CALL_ROOTS = (SRC, FRONTENDS)
CALL_FILES = (RENDER_RUNNER,)
SIZE_ROOTS = (SRC, FRONTENDS, FIXTURES)
TEST_DEP_ROOTS = (FRONTENDS,)

# 5000 lines. On the head this gate was introduced against, the files above it are the ones
# docs/REFACTOR_PLAN_2026_09.md already names as split candidates. A lower threshold baselines many
# ordinary files and makes each a conflict surface; a higher one lets the next render_runner.h grow
# unwatched for a year.
FILE_SIZE_THRESHOLD = 5000
# A line cap reports `decrease` only when the file is more than 2% below it (~330 lines on a
# 16,500-line file). See the module docstring for the concurrency reasoning.
SHRINK_MARGIN = 0.02

TITLE_ID_RE = re.compile(r"(?:PPSA|CUSA)[0-9]{5}")
TITLE_DIR_ID_RE = re.compile(r"^(?:ppsa|cusa)[0-9]{5}$", re.IGNORECASE)
GETENV_RE = re.compile(r"\b(?:secure_getenv|getenv)\s*\(")
SYNC_TOKENS = (
    "vkWaitForFences",
    "vkQueueWaitIdle",
    "vkDeviceWaitIdle",
    "VK_PIPELINE_STAGE_ALL_COMMANDS_BIT",
    "VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT",
)
SYNC_RES = {token: re.compile(rf"\b{token}\b") for token in SYNC_TOKENS}
TEST_DEP_RE = re.compile(r"\bprosper\s*::\s*test\s*::")

RULES = ("title-id", "title-dir", "getenv", "blocking-sync", "file-size", "test-dep")
CAP_RULES = ("title-dir", "file-size")

# Sanity floors, far below today's tree. Tripping one means the scan is not seeing the tree (wrong
# --root, a broken lexer), not that the tree got clean overnight. If the tree really does get
# there, lower the floor in the same PR -- a reviewable event, as it should be.
MIN_SLUGS = 10
MIN_GETENV_SITES = 1

EXIT_OK, EXIT_VIOLATION, EXIT_UNEVALUATED = 0, 1, 2

RAW_PREFIXES = ("R", "u8R", "uR", "UR", "LR")
RAW_OPEN_RE = re.compile(r'"([^()\\\s"]{0,16})\(')

FIX_HINT = {
    "title-id": (
        "A title id in shared code. Derive the behaviour from what the guest presents (a "
        "descriptor, a packet, param.json) instead of naming the title. A COMMENT naming the "
        "capture a fact came from is fine; this rule ignores comments."
    ),
    "title-dir": (
        "A title-named module in shared code. Express the contract as a property of the guest's "
        "data, in the folder that owns that data. Existing ones may shrink, never grow."
    ),
    "getenv": (
        "A new raw getenv read. Read switches through src/diagnostics/env_cache.hpp (after "
        "tools/env/check_cached_env.py says the name may be cached), or pass configuration in."
    ),
    "blocking-sync": (
        "A new blocking GPU sync (CPU wait or ALL_COMMANDS barrier). A new site must name the "
        "guest-visible result it delivers -- in a comment beside it and in the PR -- and only "
        "then raise this row, in the same PR, where a reviewer sees it."
    ),
    "file-size": (
        "A file past its line cap. Split it (prosper/tools/refactor/split_file.py) or move code "
        "out; a cap rises only by a baseline edit a reviewer accepts."
    ),
    "test-dep": (
        "The shipping frontend referencing prosper::test::. Move what it needs into src/ or "
        "frontends/shared/ instead of reaching into the test namespace."
    ),
}


class EvaluationError(Exception):
    """The gate could not establish an answer; maps to exit status 2, never to a pass."""


def rule_of(key: str) -> str:
    """The rule name a baseline key belongs to (`getenv|path` -> `getenv`)."""
    return key.split("|", 1)[0]


# --------------------------------------------------------------------------------------------
# Lexing
# --------------------------------------------------------------------------------------------
def _blank(buf: list[str], start: int, end: int) -> None:
    for k in range(start, min(end, len(buf))):
        if buf[k] != "\n":
            buf[k] = " "


def _token_before(text: str, index: int) -> str:
    start = index
    while start > 0 and (text[start - 1].isalnum() or text[start - 1] in "_."):
        start -= 1
    return text[start:index]


def _quoted_end(text: str, start: int, quote: str) -> int:
    """Index one past the closing quote of the literal opening at `start` (or at the newline)."""
    j, n = start + 1, len(text)
    while j < n:
        ch = text[j]
        if ch == "\\":
            j += 2
            continue
        if ch == quote:
            return j + 1
        if ch == "\n":
            return j
        j += 1
    return n


def lex(text: str) -> tuple[str, str]:
    """Return (code, bare), both the same length as `text` with every newline kept.

    `code` has comments blanked and literals intact; `bare` additionally blanks the inside of
    every string and char literal. Positions map 1:1 to the source, so a reported line number is
    the real one.
    """
    code, bare = list(text), list(text)
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if ch == "/" and nxt == "/":
            j = i + 2
            while j < n:
                if text[j] == "\n":
                    k = j - 1
                    if k >= 0 and text[k] == "\r":
                        k -= 1
                    if k >= i + 2 and text[k] == "\\":
                        j += 1
                        continue
                    break
                j += 1
            _blank(code, i, j)
            _blank(bare, i, j)
            i = j
            continue
        if ch == "/" and nxt == "*":
            end = text.find("*/", i + 2)
            j = n if end < 0 else end + 2
            _blank(code, i, j)
            _blank(bare, i, j)
            i = j
            continue
        if ch == '"':
            if _token_before(text, i) in RAW_PREFIXES:
                m = RAW_OPEN_RE.match(text, i)
                if m:
                    close = ")" + m.group(1) + '"'
                    end = text.find(close, m.end())
                    j = n if end < 0 else end + len(close)
                    _blank(bare, i + 1, j - 1)
                    i = j
                    continue
            j = _quoted_end(text, i, '"')
            _blank(bare, i + 1, j - 1)
            i = j
            continue
        if ch == "'":
            token = _token_before(text, i)
            if token and token[0].isdigit():  # C++14 digit separator: 0x1'0000
                i += 1
                continue
            j = _quoted_end(text, i, "'")
            _blank(bare, i + 1, j - 1)
            i = j
            continue
        i += 1
    return "".join(code), "".join(bare)


def line_count(text: str) -> int:
    """Physical lines, as `wc -l` counts them plus an unterminated last line."""
    if not text:
        return 0
    return text.count("\n") + (0 if text.endswith("\n") else 1)


def _line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


# --------------------------------------------------------------------------------------------
# Scanning
# --------------------------------------------------------------------------------------------
@dataclass
class Finding:
    """One baseline-keyed measurement: a count (or a line total) plus where it came from."""

    key: str
    value: int
    lines: list[int] = field(default_factory=list)

    @property
    def rule(self) -> str:
        return rule_of(self.key)


def is_source(path: str) -> bool:
    return path.endswith(SOURCE_SUFFIXES)


def under(path: str, roots: Iterable[str]) -> bool:
    return any(path.startswith(root) for root in roots)


def derive_slugs(paths: Iterable[str]) -> set[str]:
    """Title slugs from the per-title route folders directly under prosper/scripts/."""
    folders = set()
    for path in paths:
        if path.startswith(SCRIPTS) and "/" in path[len(SCRIPTS) :]:
            folders.add(path[len(SCRIPTS) :].split("/", 1)[0])
    slugs = set()
    for folder in folders:
        base = re.sub(r"-(?:ppsa|cusa)[0-9]{5}$", "", folder.lower())
        if base:
            slugs.update({base, base.replace("-", "_"), base.replace("-", "")})
    return slugs


def is_title_dir(name: str, slugs: set[str]) -> bool:
    return name.lower() in slugs or bool(TITLE_DIR_ID_RE.match(name))


def _hits(regex: re.Pattern[str], view: str) -> list[int]:
    return [_line_of(view, m.start()) for m in regex.finditer(view)]


def scan(files: dict[str, str], slugs: set[str]) -> dict[str, Finding]:
    """Every finding for a tree given as {repo-relative posix path: text}."""
    found: dict[str, Finding] = {}

    def add(key: str, lines: list[int]) -> None:
        if lines:
            found[key] = Finding(key, len(lines), lines)

    dir_lines: dict[str, int] = {}
    for path in sorted(files):
        if not is_source(path):
            continue
        text = files[path]
        calls = under(path, CALL_ROOTS) or path in CALL_FILES
        if under(path, TITLE_ID_ROOTS + TEST_DEP_ROOTS) or calls:
            code, bare = lex(text)
            if under(path, TITLE_ID_ROOTS):
                add(f"title-id|{path}", _hits(TITLE_ID_RE, code))
            if calls:
                add(f"getenv|{path}", _hits(GETENV_RE, bare))
                for token, regex in SYNC_RES.items():
                    add(f"blocking-sync|{path}|{token}", _hits(regex, bare))
            if under(path, TEST_DEP_ROOTS):
                add(f"test-dep|{path}", _hits(TEST_DEP_RE, bare))
        if under(path, SIZE_ROOTS):
            lines = line_count(text)
            if lines > FILE_SIZE_THRESHOLD:
                found[f"file-size|{path}"] = Finding(f"file-size|{path}", lines)
        if path.startswith(SRC):
            parts = path.split("/")
            # parts[0:2] is prosper/src; every directory component below it, never the filename.
            for depth in range(2, len(parts) - 1):
                if is_title_dir(parts[depth], slugs):
                    directory = "/".join(parts[: depth + 1])
                    dir_lines[directory] = dir_lines.get(directory, 0) + line_count(text)
    for directory, total in dir_lines.items():
        key = f"title-dir|{directory}"
        found[key] = Finding(key, max(total, 1))
    return found


# --------------------------------------------------------------------------------------------
# Baseline
# --------------------------------------------------------------------------------------------
@dataclass
class Row:
    """One baseline row: key, the value it pins, and the free-text note after `#`."""

    key: str
    value: int
    note: str = ""


def parse_baseline(text: str) -> tuple[list[str], dict[str, Row]]:
    """(header comment lines, rows). Raises EvaluationError on anything it cannot read."""
    header: list[str] = []
    rows: dict[str, Row] = {}
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            if not rows:
                header.append(raw.rstrip())
            continue
        body, _sep, note = line.partition("#")
        fields = body.split()
        if len(fields) != 2 or not fields[1].isdigit():
            raise EvaluationError(f"baseline line {number}: expected `KEY VALUE  # note`: {raw!r}")
        key, value = fields[0], int(fields[1])
        if rule_of(key) not in RULES:
            raise EvaluationError(f"baseline line {number}: unknown rule in {key!r}")
        if key in rows:
            raise EvaluationError(f"baseline line {number}: duplicate key {key!r}")
        if value <= 0:
            raise EvaluationError(f"baseline line {number}: a zero row is a deleted row: {key!r}")
        rows[key] = Row(key, value, note.strip())
    while header and header[-1] == "":
        header.pop()
    return header, rows


def _row_order(key: str) -> tuple[int, str]:
    return RULES.index(rule_of(key)), key


def format_baseline(header: list[str], rows: Iterable[Row]) -> str:
    out = [*header, ""] if header else []
    for row in sorted(rows, key=lambda r: _row_order(r.key)):
        out.append(f"{row.key} {row.value}" + (f"  # {row.note}" if row.note else ""))
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------------------------
# Verdicts
# --------------------------------------------------------------------------------------------
@dataclass
class Problem:
    """One failure: what kind, which key, and the two numbers that disagree."""

    kind: str  # new | increase | decrease | stale
    key: str
    baseline: int
    current: int
    lines: list[int] = field(default_factory=list)
    merge_base: int | None = None  # delta mode only: the count at the merge base

    @property
    def rule(self) -> str:
        return rule_of(self.key)


def compare(found: dict[str, Finding], rows: dict[str, Row]) -> list[Problem]:
    problems: list[Problem] = []
    for key in sorted(found, key=_row_order):
        f = found[key]
        row = rows.get(key)
        if row is None:
            problems.append(Problem("new", key, 0, f.value, f.lines))
        elif f.value > row.value:
            problems.append(Problem("increase", key, row.value, f.value, f.lines))
        elif f.value < row.value:
            if f.rule in CAP_RULES and f.value >= row.value * (1 - SHRINK_MARGIN):
                continue
            problems.append(Problem("decrease", key, row.value, f.value, f.lines))
    for key in sorted(rows, key=_row_order):
        if key not in found:
            problems.append(Problem("stale", key, rows[key].value, 0))
    return problems


def delta_scope(changed: Iterable[str], slugs: set[str]) -> tuple[set[str], set[str]]:
    """(changed paths in a scanned root, title-dir directories those paths sit in)."""
    paths, dirs = set(), set()
    for path in changed:
        if not is_source(path):
            continue
        if under(path, TITLE_ID_ROOTS + CALL_ROOTS + SIZE_ROOTS) or path in CALL_FILES:
            paths.add(path)
        if path.startswith(SRC):
            parts = path.split("/")
            for depth in range(2, len(parts) - 1):
                if is_title_dir(parts[depth], slugs):
                    dirs.add("/".join(parts[: depth + 1]))
    return paths, dirs


def _key_in_scope(key: str, paths: set[str], dirs: set[str]) -> bool:
    target = key.split("|")[1]
    return target in dirs if rule_of(key) == "title-dir" else target in paths


def compare_delta(
    base_found: dict[str, Finding],
    head_found: dict[str, Finding],
    rows: dict[str, Row],
    paths: set[str],
    dirs: set[str],
) -> tuple[list[Problem], list[Problem]]:
    """(failures, notices) for the keys a change touches.

    A key fails only when its count ROSE between the merge base and the working tree AND the
    working tree's baseline row does not cover the new count. Rows elsewhere are not consulted,
    so a stale row in a file this change did not touch cannot fail it. Notices are decreases
    worth recording with `--update`; they never fail.
    """
    failures: list[Problem] = []
    notices: list[Problem] = []
    keys = {k for k in (*base_found, *head_found) if _key_in_scope(k, paths, dirs)}
    for key in sorted(keys, key=_row_order):
        was = base_found[key].value if key in base_found else 0
        head = head_found.get(key)
        now = head.value if head else 0
        lines = head.lines if head else []
        row = rows.get(key)
        if now > was:
            if row is None:
                failures.append(Problem("new", key, 0, now, lines, was))
            elif now > row.value:
                failures.append(Problem("increase", key, row.value, now, lines, was))
        elif now < was and row is not None and now < row.value:
            if rule_of(key) in CAP_RULES and now >= row.value * (1 - SHRINK_MARGIN):
                continue
            kind = "decrease" if now else "stale"
            notices.append(Problem(kind, key, row.value, now, lines, was))
    return failures, notices


def apply_repairs(rows: dict[str, Row], problems: list[Problem]) -> dict[str, Row]:
    """Lower `decrease` rows and delete `stale` ones. Never raises or adds a row."""
    out = {k: Row(r.key, r.value, r.note) for k, r in rows.items()}
    for p in problems:
        if p.kind == "decrease":
            out[p.key].value = p.current
        elif p.kind == "stale":
            del out[p.key]
    return out


def describe(p: Problem) -> str:
    where = ""
    if p.lines:
        shown = ", ".join(str(n) for n in p.lines[:12]) + (" ..." if len(p.lines) > 12 else "")
        where = f" (line {shown})"
    if p.merge_base is not None:
        where += f" [merge base: {p.merge_base}]"
    if p.kind == "new":
        return f"NEW       {p.key} = {p.current}{where}\n            {FIX_HINT[p.rule]}"
    if p.kind == "increase":
        return (
            f"INCREASE  {p.key}: baseline {p.baseline} -> now {p.current}{where}\n"
            f"            {FIX_HINT[p.rule]}"
        )
    if p.kind == "decrease":
        return (
            f"DECREASE  {p.key}: baseline {p.baseline} -> now {p.current}. Good -- now lower "
            f"the row to {p.current} (or run with --update) so the headroom cannot be re-spent."
        )
    return (
        f"STALE     {p.key}: baselined at {p.baseline}, no longer present. Delete the row "
        "(or run with --update)."
    )


# --------------------------------------------------------------------------------------------
# Tree access
# --------------------------------------------------------------------------------------------
def tracked_paths(root: Path) -> list[str]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True
        ).stdout.decode("utf-8", "replace")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise EvaluationError(f"could not list tracked files under {root}: {exc}") from exc
    return [p for p in out.split("\0") if p]


def load_tree(root: Path) -> tuple[dict[str, str], set[str]]:
    paths = tracked_paths(root)
    slugs = derive_slugs(paths)
    files: dict[str, str] = {}
    for path in paths:
        if not is_source(path):
            continue
        if not (under(path, TITLE_ID_ROOTS + CALL_ROOTS + SIZE_ROOTS) or path in CALL_FILES):
            continue
        try:
            files[path] = (root / path).read_bytes().decode("utf-8", "replace")
        except OSError:
            continue  # tracked but deleted in the working tree: its counts are gone, correctly
    return files, slugs


def _git(root: Path, *args: str, stdin: bytes | None = None) -> bytes:
    try:
        return subprocess.run(
            ["git", *args], cwd=root, input=stdin, capture_output=True, check=True
        ).stdout
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or b"").decode("utf-8", "replace").strip() or str(exc)
        raise EvaluationError(f"`git {' '.join(args[:3])}` failed under {root}: {detail}") from exc
    except OSError as exc:
        raise EvaluationError(f"could not run git under {root}: {exc}") from exc


def _nul_split(raw: bytes) -> list[str]:
    return [p for p in raw.decode("utf-8", "replace").split("\0") if p]


def resolve_merge_base(root: Path, ref: str) -> str:
    """The merge base of HEAD and `ref`, or EvaluationError (exit 2) when there is none."""
    try:
        sha = _git(root, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")
    except EvaluationError as exc:
        raise EvaluationError(f"--base {ref!r} is not a commit in {root} (fetch it?)") from exc
    try:
        return _git(root, "merge-base", "HEAD", sha.decode().strip()).decode().strip()
    except EvaluationError as exc:
        raise EvaluationError(f"HEAD and --base {ref!r} share no merge base") from exc


def read_blobs(root: Path, commit: str, paths: Iterable[str]) -> dict[str, str]:
    """{path: text} for every path that exists in `commit`, via ONE `git cat-file --batch`."""
    wanted = sorted(paths)
    if not wanted:
        return {}
    request = "".join(f"{commit}:{p}\n" for p in wanted).encode("utf-8")
    out = _git(root, "cat-file", "--batch", stdin=request)
    blobs: dict[str, str] = {}
    pos = 0
    for path in wanted:
        eol = out.index(b"\n", pos)
        header = out[pos:eol].split()
        pos = eol + 1
        if len(header) == 2 and header[1] == b"missing":
            continue
        if len(header) != 3:
            raise EvaluationError(f"unexpected `git cat-file --batch` header: {header!r}")
        size = int(header[2])
        if header[1] == b"blob":
            blobs[path] = out[pos : pos + size].decode("utf-8", "replace")
        pos += size + 1  # the content, then the newline cat-file appends after it
    return blobs


def run_delta(root: Path, baseline: Path, ref: str) -> int:
    """Delta mode: fail only on a count this change raised past its row. See the docstring."""
    if not baseline.is_file():
        raise EvaluationError(f"baseline {baseline} not found")
    _header, rows = parse_baseline(baseline.read_text(encoding="utf-8"))
    base = resolve_merge_base(root, ref)
    tracked = tracked_paths(root)
    untracked = _nul_split(_git(root, "ls-files", "--others", "--exclude-standard", "-z"))
    slugs = derive_slugs(tracked + untracked)
    if len(slugs) < MIN_SLUGS:
        raise EvaluationError(
            f"derived only {len(slugs)} title slug(s) from {SCRIPTS} (floor {MIN_SLUGS})"
        )
    changed = _nul_split(_git(root, "diff", "--name-only", "--no-renames", "-z", base, "--"))
    paths, dirs = delta_scope([*changed, *untracked], slugs)
    head_paths, base_paths = set(paths), set(paths)
    if dirs:
        prefixes = tuple(d + "/" for d in sorted(dirs))
        head_paths |= {p for p in (*tracked, *untracked) if p.startswith(prefixes)}
        listed = _git(root, "ls-tree", "-r", "--name-only", "-z", base, "--", *sorted(dirs))
        base_paths |= set(_nul_split(listed))
    head_files: dict[str, str] = {}
    for path in sorted(p for p in head_paths if is_source(p)):
        try:
            head_files[path] = (root / path).read_bytes().decode("utf-8", "replace")
        except OSError:
            continue  # deleted in the working tree: its counts are zero, correctly
    base_files = read_blobs(root, base, (p for p in base_paths if is_source(p)))
    failures, notices = compare_delta(
        scan(base_files, slugs), scan(head_files, slugs), rows, paths, dirs
    )
    print(
        f"delta vs merge base {base[:12]} ({ref}): {len(paths)} changed C/C++ file(s) in scope, "
        f"{len(dirs)} title dir(s)"
    )
    for p in notices:
        print("  notice: " + describe(p))
    if not failures:
        print("ok: no count this change touches rose past its baseline row")
        return EXIT_OK
    print(
        f"FAIL: this change raised {len(failures)} architecture-ratchet count(s):", file=sys.stderr
    )
    for p in failures:
        print("  " + describe(p), file=sys.stderr)
    print(
        "\nCounts here may go down, never up. If a rise is genuinely justified, raise or add the "
        f"row in {baseline.name} in this same change, with a `# note` saying why, so a reviewer "
        "sees it. A moved file is a NEW key: rename its row.",
        file=sys.stderr,
    )
    return EXIT_VIOLATION


def sanity(files: dict[str, str], slugs: set[str], found: dict[str, Finding]) -> None:
    for root in (SRC, FRONTENDS, FIXTURES):
        if not any(p.startswith(root) for p in files):
            raise EvaluationError(f"no tracked C/C++ sources under {root} -- wrong --root?")
    if len(slugs) < MIN_SLUGS:
        raise EvaluationError(
            f"derived only {len(slugs)} title slug(s) from {SCRIPTS} (floor {MIN_SLUGS}); the "
            "title-dir rule would be checking against nothing"
        )
    if sum(f.value for f in found.values() if f.rule == "getenv") < MIN_GETENV_SITES:
        raise EvaluationError("found no getenv call site at all -- the lexer is not seeing code")


def summary(found: dict[str, Finding]) -> str:
    parts = []
    for rule in RULES:
        rows = [f for f in found.values() if f.rule == rule]
        if rule in CAP_RULES:
            parts.append(f"{rule}={len(rows)} row(s)")
        else:
            parts.append(f"{rule}={sum(f.value for f in rows)} in {len(rows)} row(s)")
    return ", ".join(parts)


# --------------------------------------------------------------------------------------------
# Self-test: hand-built instances, independent of whatever the tree happens to contain
# --------------------------------------------------------------------------------------------
SELF_SLUGS = {"gta5", "sonic_frontiers", "sonicfrontiers", "sonic-frontiers"}
BIG = "int x;\n" * (FILE_SIZE_THRESHOLD + 1)

# One hand-written positive instance per rule. The tests and --selftest both use it.
POSITIVE_TREE = {
    "prosper/src/a.cpp": 'const char* t = "PPSA24651";\nint c = 0; // CUSA00001 in a comment\n',
    "prosper/src/gpu/gta5/c.cpp": "int a;\nint b;\n",
    "prosper/src/e.cpp": 'auto v = std::getenv("X");\nauto w = secure_getenv ("Y");\n',
    "prosper/src/f.cpp": (
        "vkWaitForFences(d, 1, &f, VK_TRUE, ~0ull);\n"
        "b.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;\n"
    ),
    "prosper/frontends/g.cpp": "prosper::test::RenderCtx ctx;\n",
    "prosper/tests/fixtures/big.h": BIG,
    RENDER_RUNNER: "vkQueueWaitIdle(q);\n",
}
POSITIVE_KEYS = {
    "title-id|prosper/src/a.cpp": 1,
    "title-dir|prosper/src/gpu/gta5": 2,
    "getenv|prosper/src/e.cpp": 2,
    "blocking-sync|prosper/src/f.cpp|vkWaitForFences": 1,
    "blocking-sync|prosper/src/f.cpp|VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT": 1,
    "test-dep|prosper/frontends/g.cpp": 1,
    "file-size|prosper/tests/fixtures/big.h": FILE_SIZE_THRESHOLD + 1,
    f"blocking-sync|{RENDER_RUNNER}|vkQueueWaitIdle": 1,
}
# Every rule's pattern written where it must NOT count: comments, string and raw-string literals,
# lookalike identifiers, non-scanned roots, a non-title directory.
NEGATIVE_TREE = {
    "prosper/src/n1.cpp": (
        '// getenv("A") vkWaitForFences PPSA24651 prosper::test::x\n'
        '/* getenv("B")\n vkDeviceWaitIdle */\n'
    ),
    "prosper/src/n2.cpp": (
        'log("vkWaitForFences failed // getenv(");\n'
        'auto r = R"x(getenv("C") */ vkQueueWaitIdle)x";\n'
    ),
    "prosper/src/n3.cpp": "int n = 0x1'0000; char q = '\"'; int m = 1'000;\n",
    "prosper/src/gpu/gtav/x.cpp": "int a;\n",
    "prosper/tests/host/t.cpp": 'auto v = getenv("X"); vkWaitForFences(); // PPSA24651\n',
    "prosper/src/my.cpp": 'my_getenv("x"); SDL_getenv("y"); PFN_vkWaitForFences p;\n',
    "prosper/frontends/s.cpp": 'puts("prosper::test::x");\n',
}


def scan_values(files: dict[str, str], slugs: set[str] = SELF_SLUGS) -> dict[str, int]:
    return {k: f.value for k, f in scan(files, slugs).items()}


def selftest() -> int:
    failures: list[str] = []

    def expect(label: str, got: object, want: object) -> None:
        if got != want:
            failures.append(f"{label}: got {got!r}, want {want!r}")

    expect("one positive instance per rule", scan_values(POSITIVE_TREE), POSITIVE_KEYS)
    expect("comments, literals and lookalikes are not code", scan_values(NEGATIVE_TREE), {})
    # The lexer must still see code AFTER each tricky construct -- one that swallowed the rest of
    # the file would pass the negative arm above vacuously.
    tail = {
        "prosper/src/t.cpp": "char q = '\"'; int n = 1'000;\n"
        'auto r = R"(x)"; /* c */ // d\n'
        'getenv("E");\n'
    }
    expect("code after literals is still seen", scan_values(tail), {"getenv|prosper/src/t.cpp": 1})
    cont = {"prosper/src/u.cpp": '// continued \\\ngetenv("F");\ngetenv("G");\n'}
    expect("backslash-continued comment", scan_values(cont), {"getenv|prosper/src/u.cpp": 1})

    found = scan(POSITIVE_TREE, SELF_SLUGS)
    rows = {k: Row(k, f.value) for k, f in found.items()}

    def kinds(tree: dict[str, str]) -> list[str]:
        return [p.kind for p in compare(scan(tree, SELF_SLUGS), rows)]

    expect("baseline equal to the tree is clean", kinds(POSITIVE_TREE), [])
    grown = dict(POSITIVE_TREE)
    grown["prosper/src/e.cpp"] += 'getenv("Z");\n'
    expect("increase fails", kinds(grown), ["increase"])
    shrunk = dict(POSITIVE_TREE, **{"prosper/src/e.cpp": 'auto v = std::getenv("X");\n'})
    expect("decrease fails until lowered", kinds(shrunk), ["decrease"])
    gone = {k: v for k, v in POSITIVE_TREE.items() if k != "prosper/frontends/g.cpp"}
    expect("stale row fails", kinds(gone), ["stale"])

    # Delta mode, on the same hand-built tree. The baseline is deliberately STALE for big.h (cap
    # below the file), which full mode reports and delta mode must not -- unless big.h changed.
    stale_rows = dict(rows)
    big = "file-size|prosper/tests/fixtures/big.h"
    stale_rows[big] = Row(big, FILE_SIZE_THRESHOLD - 1)

    def delta_kinds(head: dict[str, str], changed: list[str], rows_: dict[str, Row]) -> list[str]:
        paths, dirs = delta_scope(changed, SELF_SLUGS)
        fails, _notes = compare_delta(
            scan(POSITIVE_TREE, SELF_SLUGS), scan(head, SELF_SLUGS), rows_, paths, dirs
        )
        return [p.kind for p in fails]

    e_cpp = "prosper/src/e.cpp"
    full_kinds = [p.kind for p in compare(found, stale_rows)]
    expect("full mode sees the stale row", full_kinds, ["increase"])
    expect("delta: unrelated change passes", delta_kinds(POSITIVE_TREE, [e_cpp], stale_rows), [])
    expect("delta: a rise fails", delta_kinds(grown, [e_cpp], stale_rows), ["increase"])
    raised = dict(stale_rows, **{f"getenv|{e_cpp}": Row(f"getenv|{e_cpp}", 3)})
    expect("delta: a raised row covers it", delta_kinds(grown, [e_cpp], raised), [])
    z_cpp = "prosper/src/z.cpp"
    added = dict(POSITIVE_TREE, **{z_cpp: 'getenv("N");\n'})
    expect("delta: new file vs zero", delta_kinds(added, [z_cpp], stale_rows), ["new"])

    for message in failures:
        print(f"selftest: FAIL {message}", file=sys.stderr)
    print(f"selftest: 13 arms, {len(failures)} failed")
    return EXIT_VIOLATION if failures else EXIT_OK


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------
def run(root: Path, baseline: Path, mode: str) -> int:
    files, slugs = load_tree(root)
    found = scan(files, slugs)
    sanity(files, slugs, found)
    if mode == "emit":
        for key in sorted(found, key=_row_order):
            print(f"{key} {found[key].value}")
        return EXIT_OK
    if not baseline.is_file():
        raise EvaluationError(f"baseline {baseline} not found")
    header, rows = parse_baseline(baseline.read_text(encoding="utf-8"))
    print(f"scanned {len(files)} C/C++ file(s), {len(slugs)} title slug(s): {summary(found)}")
    if mode == "list":
        for key in sorted(found, key=_row_order):
            f = found[key]
            at = (" @ " + ",".join(map(str, f.lines))) if f.lines else ""
            print(f" {' ' if key in rows else '*'} {key} {f.value}{at}")
    problems = compare(found, rows)
    if mode == "update":
        repaired = apply_repairs(rows, problems)
        if repaired != rows:
            baseline.write_text(format_baseline(header, repaired.values()), encoding="utf-8")
            fixed = sum(p.kind in ("decrease", "stale") for p in problems)
            print(f"updated {baseline.name}: lowered or deleted {fixed} row(s)")
        rows = repaired
        problems = compare(found, rows)
    if not problems:
        print(f"ok: {len(rows)} baseline row(s), every count at or below its row")
        return EXIT_OK
    print(f"FAIL: {len(problems)} architecture-ratchet problem(s):", file=sys.stderr)
    for p in problems:
        print("  " + describe(p), file=sys.stderr)
    if any(p.kind in ("new", "increase") for p in problems):
        print(
            "\nCounts here may go down, never up. If a rise is genuinely justified, raise or add "
            f"the row in {baseline.name} with a `# note` saying why, so a reviewer sees it.",
            file=sys.stderr,
        )
    return EXIT_VIOLATION


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--root", default=str(HERE.parents[2]), help="checkout root")
    ap.add_argument("--baseline", default=str(DEFAULT_BASELINE), help="baseline file")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--selftest", action="store_true", help="run the hand-built arms")
    group.add_argument("--list", action="store_true", help="also print every finding (* = new)")
    group.add_argument(
        "--update",
        action="store_true",
        help="lower decreased rows and delete stale ones; never raises or adds a row",
    )
    group.add_argument(
        "--emit-baseline", action="store_true", help="print the rows for the current tree"
    )
    group.add_argument(
        "--base",
        metavar="REF",
        help="delta mode: fail only on counts raised since the merge base of HEAD and REF "
        "(e.g. origin/main, or a PR's base SHA)",
    )
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    mode = "list" if args.list else "update" if args.update else ""
    mode = "emit" if args.emit_baseline else mode
    try:
        if args.base is not None:
            return run_delta(Path(args.root).resolve(), Path(args.baseline), args.base)
        return run(Path(args.root).resolve(), Path(args.baseline), mode)
    except EvaluationError as exc:
        print(f"error: could not evaluate: {exc}", file=sys.stderr)
        return EXIT_UNEVALUATED


if __name__ == "__main__":
    sys.exit(main())
