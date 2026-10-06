#!/usr/bin/env python3
"""Every document says what kind it is, and every relative Markdown link resolves.

Two properties of the documentation that nobody can be relied on to keep by hand.

KIND. A reader of `prosper/docs/` needs to know whether a page is a design that describes how
something works today, a dated investigation whose numbers were true when measured, a title status
page, a PS5 reference fact, a plan, a process guide, or an archived record that must not be started
from. The folder used to be the only signal, and folders mix kinds (`gpu/` holds both the renderer
design and a dozen dated measurement passes). So every document under `prosper/docs/` opens with a
frontmatter block naming its kind and status:

    ---
    kind: investigation
    status: current
    ---

`spec/` and `adr/` carry their own, richer frontmatter, checked by `check_arch_docs.py`; this
checker skips them except to reject their kinds elsewhere. `AGENTS.md` folder maps carry none.

  kind      design | reference | investigation | status | process | plan | archive
  status    current | historical | superseded
  template  optional; `design` requires every DESIGN_SECTIONS heading below (the subsystem
            design-doc template)

A document in `archive/` is `kind: archive`, and an `archive` document lives in `archive/`; a
`superseded` document names where to go instead with `superseded-by: <path>`, and that path exists.

LINKS. Every relative link `[text](path)` in a tracked Markdown file, anywhere in the repository,
must point at a file or directory that exists. Code spans and fenced blocks are skipped (a C++
expression such as `vtbl[0](this)` is not a link); URLs, mail links and same-page anchors are not
checked; an anchor after a path is ignored and only the path is checked. A target made only of
dots, such as `(...)`, is an elided placeholder in prose, not a path, and is skipped. A path is
resolved component by component against the real directory listing, so a link resolves the same way
on every host: a wrong-case name, or a component ending in a dot or a space (which Win32 silently
strips, so `foo.md.` and `...` "exist" on Windows), is broken everywhere, as it is on CI's Linux
runner. The motivating case:
`tools/screenshots/shrink.py` re-encoded committed screenshots from PNG to WebP, and four links
across three documents kept pointing at the `.png` names, rendering as broken images with every
gate green.

WHAT IT CANNOT SEE: whether a kind is the right one, whether a `current` page is still true, or
whether a link points at the RIGHT existing file.

EXIT STATUS: 0 clean, 1 violation, 2 could not evaluate.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

EXIT_OK, EXIT_VIOLATION, EXIT_UNEVALUATED = 0, 1, 2

DOCS = Path("prosper/docs")
OWN_CHECKER = (DOCS / "spec", DOCS / "adr")
KINDS = ("design", "reference", "investigation", "status", "process", "plan", "archive")
STATUSES = ("current", "historical", "superseded")
DESIGN_SECTIONS = (
    "Scope",
    "Current state",
    "Decision",
    "Target design",
    "Interfaces",
    "Failure modes",
    "Tests",
    "Milestones",
    "Open questions",
    "Ruled out",
)
LINK_RE = re.compile(r"!?\[[^\]\n]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
FENCE_RE = re.compile(r"^(```|~~~).*?^\1", re.M | re.S)
CODE_SPAN_RE = re.compile(r"(`+)(?:(?!\1).)+?\1", re.S)
EXTERNAL_RE = re.compile(r"^(?:[a-zA-Z][a-zA-Z0-9+.-]*:|#|//)")
# `[Claude Code](...)` in prose elides a URL; three or more dots (or an ellipsis) are never a path.
PLACEHOLDER_RE = re.compile(r"^(?:\.{3,}|\u2026)$")
# Generated from the tracker issues by gen_progress_tracker.py: a broken link there is fixed in the
# issue it came from, never by editing this file, which the next regeneration would overwrite.
GENERATED = {Path("PROGRESS_TRACKER.md")}


class EvaluationError(Exception):
    """The gate could not establish an answer; exit 2, never a pass."""


def tracked_markdown(root: Path) -> list[Path]:
    proc = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", "--", "*.md"],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise EvaluationError(f"git ls-files failed: {proc.stderr.strip()}")
    return [root / p for p in proc.stdout.split("\0") if p]


def frontmatter(text: str) -> dict[str, str] | None:
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---\n", 4)
    if end < 0:
        return None
    fields = {}
    for line in text[4:end].splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip():
            fields[key.strip()] = value.strip()
    return fields


def under(rel: Path, base: Path) -> bool:
    return rel.parts[: len(base.parts)] == base.parts


def check_meta(root: Path, rel: Path, text: str) -> list[str]:
    if rel.name == "AGENTS.md" or any(under(rel, b) for b in OWN_CHECKER):
        return []
    fields = frontmatter(text.replace("\r\n", "\n"))
    where = rel.as_posix()
    if fields is None:
        return [f"{where}: no frontmatter; open with ---, `kind:` and `status:`, ---"]
    out = []
    kind, status = fields.get("kind"), fields.get("status")
    if kind not in KINDS:
        out.append(f"{where}: kind `{kind}` is not one of {', '.join(KINDS)}")
    if status not in STATUSES:
        out.append(f"{where}: status `{status}` is not one of {', '.join(STATUSES)}")
    in_archive = under(rel, DOCS / "archive")
    if (kind == "archive") != in_archive:
        out.append(f"{where}: `kind: archive` and the archive/ folder go together")
    if status == "superseded":
        succ = fields.get("superseded-by", "")
        if not succ:
            out.append(f"{where}: a superseded document names `superseded-by: <path>`")
        elif not (root / succ).exists():
            out.append(f"{where}: superseded-by {succ} does not exist")
    if fields.get("template") == "design":
        heads = set(re.findall(r"^##\s+(.+?)\s*$", text, re.M))
        missing = [s for s in DESIGN_SECTIONS if s not in heads]
        if missing:
            out.append(f"{where}: design template sections missing: {', '.join(missing)}")
    elif "template" in fields:
        out.append(f"{where}: unknown template `{fields['template']}`")
    return out


def resolves(base: Path, path: str, listings: dict[Path, set[str]]) -> bool:
    """Walk `path` from `base` one component at a time against the real directory listing.

    `Path.exists()` asks the host, and hosts disagree: Windows is case-insensitive and strips
    trailing dots and spaces from a component, so `...`, `B.md.` and `b.md` all "exist" there while
    CI's Linux runner reports them missing. Matching each name exactly against `os.listdir` gives
    one answer on every host: no listing holds `B.md.` or `b.md` when the file is `B.md`.
    """
    cur = base
    for part in path.replace("\\", "/").split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            cur = cur.parent
            continue
        names = listings.get(cur)
        if names is None:
            try:
                names = set(os.listdir(cur))
            except OSError:
                names = set()
            listings[cur] = names
        if part not in names:
            return False
        cur = cur / part
    return True


def check_links(
    root: Path, rel: Path, text: str, listings: dict[Path, set[str]] | None = None
) -> tuple[list[str], int]:
    """Return (problems, number of relative links checked)."""
    body = CODE_SPAN_RE.sub("", FENCE_RE.sub("", text))
    listings = {} if listings is None else listings
    out, checked = [], 0
    for m in LINK_RE.finditer(body):
        target = m.group(1)
        if EXTERNAL_RE.match(target) or PLACEHOLDER_RE.match(target):
            continue
        path = target.split("#", 1)[0].split("?", 1)[0]
        if not path:
            continue
        checked += 1
        if not resolves((root / rel).parent, path, listings):
            out.append(f"{rel.as_posix()}: broken link `{target}`")
    return out, checked


def evaluate(root: Path) -> list[str]:
    return evaluate_counted(root)[0]


def evaluate_counted(root: Path) -> tuple[list[str], int]:
    """Return (problems, relative links checked); the count makes a collapsed LINK_RE visible."""
    if not (root / DOCS).is_dir():
        raise EvaluationError(f"{DOCS} not found under {root}")
    files = tracked_markdown(root)
    if not files:
        raise EvaluationError("no tracked Markdown files; is --root a checkout?")
    problems, links, listings = [], 0, {}
    for path in files:
        rel = path.relative_to(root)
        text = path.read_text(encoding="utf-8", errors="replace")
        if under(rel, DOCS):
            problems += check_meta(root, rel, text)
        if rel not in GENERATED:
            found, n = check_links(root, rel, text, listings)
            problems += found
            links += n
    return problems, links


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=Path("."), help="repository root (default: .)")
    ap.add_argument("--github", action="store_true", help="emit GitHub annotations")
    args = ap.parse_args(argv)
    try:
        problems, links = evaluate_counted(args.root.resolve())
    except EvaluationError as exc:
        print(f"check_doc_meta: could not evaluate: {exc}", file=sys.stderr)
        return EXIT_UNEVALUATED
    for p in problems:
        print(f"::error::{p}" if args.github else p)
    if problems:
        print(f"check_doc_meta: {len(problems)} problem(s)", file=sys.stderr)
        return EXIT_VIOLATION
    print(
        f"check_doc_meta: every document has a kind, every relative link resolves "
        f"({links} relative links checked)"
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
