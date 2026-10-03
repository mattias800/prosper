#!/usr/bin/env python3
"""Fail a pull request whose body does not follow .github/pull_request_template.md.

The template is the project's PR contract (context, acceptance criteria, verification, checklist),
and nothing enforced it: a PR opened with a free-form body reads fine to a human and silently drops
the sections reviewers and the merge step rely on. This check is deliberately shallow -- it cannot
judge whether the prose is good, only whether the structure exists and was filled in.

WHAT IT ASSERTS

  * every `## Heading` in the template appears in the body (the template file is the source of
    truth, so editing the template changes the gate with no second list to keep in sync);
  * each section holds real content: after removing HTML comments, blank lines and empty
    checkboxes (`- [ ]`), at least one line must remain. The template's own placeholder comments
    therefore do not count as an answer.

Checklist items need not be ticked: ticking is the author's claim, and a box ticked to satisfy a
bot is worse than an honest unticked one. The section only has to exist and say something.

Usage:
    check_pr_body.py --body-file F [--template T]
    PR_BODY=... check_pr_body.py              # CI passes the body through the environment
    check_pr_body.py --selftest

Exit status: 0 compliant, 1 violations (listed), 2 could not evaluate.
"""

import argparse
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TEMPLATE = REPO_ROOT / ".github" / "pull_request_template.md"

HEADING = re.compile(r"^##[ \t]+(.+?)[ \t]*$", re.MULTILINE)
COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
EMPTY_ITEM = re.compile(r"^[-*][ \t]*(\[[ xX]?\])?[ \t]*$")


def sections(text):
    """Map each `## Heading` to the text under it, with HTML comments removed."""
    text = COMMENT.sub("", text.replace("\r\n", "\n"))
    marks = list(HEADING.finditer(text))
    out = {}
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        out[m.group(1).strip()] = text[m.end() : end]
    return out


def has_content(body):
    """True when at least one line is neither blank nor an empty list item / checkbox."""
    return any(line.strip() and not EMPTY_ITEM.match(line.strip()) for line in body.splitlines())


def violations(body, template):
    """Return a list of human-readable problems; empty means the body complies."""
    required = list(sections(template))
    if not required:
        raise ValueError("template has no '## ' headings; nothing to enforce")
    found = sections(body)
    problems = []
    for name in required:
        if name not in found:
            problems.append(f"missing section '## {name}'")
        elif not has_content(found[name]):
            problems.append(f"section '## {name}' is empty (placeholder comments do not count)")
    return problems


def selftest():
    """Violating AND compliant inputs, so a gate that stopped discriminating fails here."""
    tpl = "## A\n\n<!-- hint -->\n\n## B\n\n- [ ]\n"
    assert violations("## A\ntext\n## B\n- [ ] done\n", tpl) == [], "compliant body rejected"
    assert violations("## A\ntext\n", tpl) == ["missing section '## B'"], "missing not caught"
    assert len(violations("## A\n<!-- hint -->\n## B\n- [ ]\n", tpl)) == 2, "empty not caught"
    assert violations("", tpl), "empty body accepted"
    print("selftest ok")


def main(argv=None):
    """CLI entry point; returns the process exit status."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--body-file")
    ap.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        return 0
    try:
        if args.body_file:
            body = Path(args.body_file).read_text(encoding="utf-8")
        else:
            body = os.environ["PR_BODY"]
        template = Path(args.template).read_text(encoding="utf-8")
        problems = violations(body, template)
    except (KeyError, OSError, ValueError) as exc:
        print(f"could not evaluate: {exc!r} (pass --body-file or set PR_BODY)", file=sys.stderr)
        return 2
    for p in problems:
        print(f"::error::PR body: {p}")
    if problems:
        print(f"PR body does not follow {Path(args.template).name}: fill every section.")
        return 1
    print("PR body follows the template.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
