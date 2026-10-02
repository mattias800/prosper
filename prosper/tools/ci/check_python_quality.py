#!/usr/bin/env python3
"""Gate Python quality on what a PR CHANGES, never on the whole pre-existing tree.

The tree predates any Python lint config: measured 2026-10-02 against 211 tracked `.py` files,
ruff reports 1,520 findings and would reformat 209 files. Failing CI on that would either block every
PR or force a mass reformat that buries real diffs. So the gate is ratcheted:

  * ADDED .py file   -> must be ruff-clean, ruff-format-clean and carry a module docstring (D100).
  * MODIFIED .py file-> its ruff finding count must not rise against the base version. Formatting is
                        not demanded of a modified file, so a one-line fix stays a one-line diff.
  * ADDED non-test tool under prosper/tools/ -> the same PR must add or change a `test_*.py`
                        (or a file under prosper/tests/). Mirrors the contribution-shape rule: it asks
                        whether the author considered the test, a reviewer judges the rest.

EXIT STATUS IS THE VERDICT: non-zero on any violation. `--selftest` runs hand-built violating AND
compliant inputs through the same functions, so a gate that quietly stopped discriminating fails there
(a checker that cannot show it still fires is indistinguishable from one that was disabled).

Run from anywhere inside the checkout:
    python3 prosper/tools/ci/check_python_quality.py --base origin/main
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

EXCLUDED_PREFIXES = ("third_party/", "prosper/third_party/")
TOOL_PREFIX = "prosper/tools/"
TEST_TREE_PREFIX = "prosper/tests/"


def is_python(path):
    """True for a tracked Python source this gate owns."""
    return path.endswith(".py") and not path.startswith(EXCLUDED_PREFIXES)


def is_test_file(path):
    """True for a file that counts as a test for the presence rule."""
    name = path.rsplit("/", 1)[-1]
    return (name.startswith("test_") and name.endswith(".py")) or path.startswith(TEST_TREE_PREFIX)


def missing_test_violations(added, touched):
    """Added non-test tools need some test touched in the same change."""
    new_tools = [p for p in added if p.startswith(TOOL_PREFIX) and not is_test_file(p)]
    if not new_tools or any(is_test_file(p) for p in touched):
        return []
    return [f"{p}: new Python tool, but the PR adds or changes no test_*.py" for p in new_tools]


def ratchet_violations(path, base_count, head_count):
    """A modified file may keep its findings but never gain any."""
    if head_count > base_count:
        return [f"{path}: ruff findings rose {base_count} -> {head_count}"]
    return []


def ruff(args, stdin=None):
    """Run ruff from the dev environment; return the completed process."""
    return subprocess.run(
        [sys.executable, "-m", "ruff", *args],
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
    )


def count_findings(source, filename):
    """Number of ruff findings in `source`, as if it lived at `filename`."""
    proc = ruff(
        ["check", "--output-format", "json", "--stdin-filename", filename, "-"],
        stdin=source,
    )
    if proc.returncode not in (0, 1) or not proc.stdout.strip():
        raise RuntimeError(
            f"ruff could not evaluate {filename} (exit {proc.returncode}): {proc.stderr.strip()}"
        )
    try:
        findings = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ruff could not evaluate {filename}: invalid JSON") from exc
    if not isinstance(findings, list) or any(not isinstance(item, dict) for item in findings):
        raise RuntimeError(f"ruff could not evaluate {filename}: invalid finding list")
    return len(findings)


SAFE_REF = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._/@~^-]*")


def validate_ref(ref):
    """Reject anything that could be parsed by git as an option rather than a revision."""
    if not SAFE_REF.fullmatch(ref):
        raise ValueError(f"refusing base ref {ref!r}: must be a plain revision name")
    return ref


def git(*args):
    """Run git and return stdout, raising on failure."""
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def parse_changes(out):
    """Decode NUL-delimited changes, retaining rename origins and all touched test paths."""
    if out and not out.endswith("\0"):
        raise ValueError("incomplete git name-status output")
    fields = iter(out.split("\0")[:-1])
    added, modified, touched = [], {}, []
    try:
        for status in fields:
            old_path = path = next(fields)
            if status.startswith("R") and status[1:].isdigit():
                path = next(fields)
            elif status not in ("A", "M"):
                raise ValueError(f"unexpected git change status {status!r}")
            touched.append(path)
            if not is_python(path):
                continue
            if status == "A" or not is_python(old_path):
                added.append(path)
            else:
                modified[path] = old_path
    except StopIteration as exc:
        raise ValueError("incomplete git name-status output") from exc
    return added, modified, touched


def changed(base):
    """Return added Python paths, modified/renamed origins and all touched paths."""
    out = git(
        "diff", "--name-status", "-z", "--find-renames", "--diff-filter=AMR", f"{base}...HEAD"
    )
    return parse_changes(out)


def check_added(path):
    """Full ruff gate for a new file: lint (includes D100) and format."""
    problems = []
    lint = ruff(["check", "--output-format", "concise", path])
    if lint.returncode:
        problems.append(f"{path}: ruff check\n{lint.stdout.rstrip()}")
    fmt = ruff(["format", "--check", path])
    if fmt.returncode:
        problems.append(f"{path}: ruff format --check (run `ruff format {path}`)")
    return problems


def check_modified(path, base, base_path):
    """Ratchet for an existing file against its base version."""
    base_src = git("show", f"{base}:{base_path}")
    head_src = Path(path).read_text(encoding="utf-8")
    return ratchet_violations(path, count_findings(base_src, path), count_findings(head_src, path))


def run(base):
    """Evaluate the whole PR; return the list of violation strings."""
    merge_base = git("merge-base", validate_ref(base), "HEAD").strip()
    added, modified, touched = changed(merge_base)
    problems = missing_test_violations(added, touched)
    for path in added:
        problems += check_added(path)
    for path, base_path in modified.items():
        problems += check_modified(path, merge_base, base_path)
    print(f"checked {len(added)} added, {len(modified)} modified Python file(s)")
    return problems


def selftest():
    """Hand-built positive and negative cases through the same functions the gate uses."""
    failures = []

    def expect(label, got, want):
        if bool(got) != want:
            failures.append(f"{label}: got {got!r}, wanted violation={want}")

    expect(
        "tool without test",
        missing_test_violations(["prosper/tools/x/a.py"], ["prosper/tools/x/a.py"]),
        True,
    )
    expect(
        "tool with test",
        missing_test_violations(
            ["prosper/tools/x/a.py"], ["prosper/tools/x/a.py", "prosper/tools/x/test_a.py"]
        ),
        False,
    )
    expect(
        "test only",
        missing_test_violations(["prosper/tools/x/test_a.py"], ["prosper/tools/x/test_a.py"]),
        False,
    )
    expect("ratchet rises", ratchet_violations("a.py", 3, 4), True)
    expect("ratchet equal", ratchet_violations("a.py", 3, 3), False)
    expect("ratchet falls", ratchet_violations("a.py", 3, 0), False)
    expect("dirty source", count_findings("import os\n", "a.py"), True)
    expect("clean source", count_findings('"""Doc."""\n', "a.py"), False)
    expect("missing docstring", count_findings("x = 1\n", "a.py"), True)
    if failures:
        print("SELFTEST FAILED\n" + "\n".join(failures))
        return 1
    print("selftest ok")
    return 0


def main():
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default="origin/main")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    try:
        if args.selftest:
            return selftest()
        problems = run(args.base)
    except (RuntimeError, ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"COULD NOT EVALUATE: {exc}", file=sys.stderr)
        return 2
    if problems:
        print("\n".join(problems))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
