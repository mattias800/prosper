#!/usr/bin/env python3
"""Enforce: new Python tests use pytest conventions, not legacy uncollectable patterns.

Scans every test_*.py under prosper/ (in tests/, tools/, frontends/, etc.).
Compliant test files must define at least one test function (`test_*`) or non-unittest test class
(`Test*`) with test methods, and must NOT subclass `unittest.TestCase`. Files that are script-only
(executable scripts with no test functions or classes) or subclass `unittest.TestCase` violate
the policy.

Files in tools/ci/pytest_legacy_allowlist.txt predate the policy and are tolerated until migrated.
A listed file that has been converted or deleted must be removed from the list (the list only shrinks).

Usage: check_pytest_policy.py [--selftest]

The scanned root is always the checkout this script lives in; it takes no path argument.
"""

import ast
import sys
from pathlib import Path

# Directories to skip when scanning for test files
EXCLUDED_DIR_NAMES = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "node_modules",
    ".venv",
    "venv",
    "third_party",
}


def is_non_compliant(content: str, filename: str = "<input>") -> tuple[bool, str]:
    """Analyze Python source for pytest policy violations.

    Returns (violates, reason).
    A compliant file:
      - Does NOT subclass unittest.TestCase.
      - Defines at least one test_* function or Test* class with test methods.
    """
    try:
        tree = ast.parse(content, filename=filename)
    except Exception as exc:
        return True, f"syntax error parsing Python test: {exc}"

    tc_classes = []
    test_funcs = []
    test_classes = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            is_tc = False
            for base in node.bases:
                base_repr = ast.unparse(base)
                if "TestCase" in base_repr or "unittest" in base_repr:
                    is_tc = True
            if is_tc:
                tc_classes.append(node.name)
            elif node.name.startswith("Test"):
                test_classes.append(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test_"):
                test_funcs.append(node.name)

    if tc_classes:
        return True, f"subclasses unittest.TestCase ({', '.join(tc_classes)})"
    if not (test_funcs or test_classes):
        return True, "no test_* functions or Test* classes (script-only or uncollected)"
    return False, ""


def scan(root: Path) -> dict[str, str]:
    """Scan all test_*.py files under root.

    Returns dict of relative_path -> violation_reason for non-compliant files.
    """
    found = {}
    for p in root.rglob("test_*.py"):
        if any(part in EXCLUDED_DIR_NAMES for part in p.parts):
            continue
        rel = p.relative_to(root).as_posix()
        try:
            content = p.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            found[rel] = f"failed to read file: {exc}"
            continue
        violates, reason = is_non_compliant(content, filename=rel)
        if violates:
            found[rel] = reason
    return found


def check(found: dict[str, str] | set[str], allow: set[str]) -> list[str]:
    """Violations: new non-pytest test files, and allowlist entries that no longer apply."""
    found_keys = set(found.keys()) if isinstance(found, dict) else set(found)
    errs = []
    for f in sorted(found_keys - allow):
        reason = found[f] if isinstance(found, dict) else "violates pytest conventions"
        errs.append(
            f"{f}: {reason}; write it using pytest conventions (functions/asserts/fixtures). "
            f"See prosper/tests/AGENTS.md."
        )
    errs += [
        f"{f}: migrated or gone; remove it from pytest_legacy_allowlist.txt"
        for f in sorted(allow - found_keys)
    ]
    return errs


def selftest() -> None:
    """Hand-built violating and compliant inputs through is_non_compliant() and check()."""
    # Compliant pytest test function
    comp1 = "def test_something():\n    assert 1 == 1\n"
    violates, reason = is_non_compliant(comp1)
    assert not violates, f"expected compliant, got {reason}"

    # Compliant pytest test class
    comp2 = "class TestSomething:\n    def test_method(self):\n        pass\n"
    violates, reason = is_non_compliant(comp2)
    assert not violates, f"expected compliant, got {reason}"

    # Violating: subclasses unittest.TestCase
    viol_ut1 = (
        "import unittest\nclass MyTests(unittest.TestCase):\n    def test_a(self):\n        pass\n"
    )
    violates, reason = is_non_compliant(viol_ut1)
    assert violates and "unittest.TestCase" in reason

    # Violating: subclasses TestCase directly
    viol_ut2 = (
        "from unittest import TestCase\n"
        "class MyTests(TestCase):\n"
        "    def test_a(self):\n        pass\n"
    )
    violates, reason = is_non_compliant(viol_ut2)
    assert violates and "unittest.TestCase" in reason

    # Violating: script-only (no test_* funcs or Test* classes)
    viol_script = "import sys\nx = 1\nassert x == 1\n"
    violates, reason = is_non_compliant(viol_script)
    assert violates and "script-only" in reason

    # Check logic
    assert check({"prosper/tests/a.py": "script-only"}, set())
    assert check(set(), {"prosper/tests/a.py"})
    assert not check({"prosper/tests/a.py": "script-only"}, {"prosper/tests/a.py"})
    print("selftest ok")


def main() -> int:
    """CLI entry point."""
    if "--selftest" in sys.argv:
        selftest()
    root = Path(__file__).resolve().parents[2]
    lst = root / "tools/ci/pytest_legacy_allowlist.txt"
    allow = {
        line.strip()
        for line in lst.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    errs = check(scan(root), allow)
    print("\n".join(errs) or "pytest policy ok")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
