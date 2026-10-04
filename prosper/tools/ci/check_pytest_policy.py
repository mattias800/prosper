#!/usr/bin/env python3
"""Enforce: new Python tests use pytest conventions, not legacy uncollectable patterns.

Scans every test_*.py under prosper/ (in tests/, tools/, frontends/, etc.).
Compliant test files must define at least one eligible pytest test in discovery scope:
  - An eligible top-level module function whose name starts with `test_`
  - An eligible top-level class whose name starts with `Test` (without inheriting from
    unittest.TestCase or defining an __init__ constructor) containing at least one eligible
    method whose name starts with `test_`
Files that define no eligible test items (such as empty classes, hidden tests inside helpers,
or test methods inside non-Test classes) are rejected as script-only or uncollectable.
Files whose test classes subclass `unittest.TestCase` (including aliased imports like
`from unittest import TestCase as Base` or local base inheritance) are rejected as legacy
unittest suites.

Files in tools/ci/pytest_legacy_allowlist.txt predate the policy and are tolerated until migrated.
A listed file that has been converted or deleted must be removed from the list (the list only shrinks).
To prevent future PRs from introducing new legacy tests by simply adding them to the allowlist,
the gate enforces shrink-only semantics against an immutable git baseline (e.g. `--base origin/main`).

Initial rollout uses --bootstrap-allowlist with a valid base commit that predates the allowlist;
only already non-compliant test files in that immutable tree can receive exemptions.

Usage: check_pytest_policy.py [--base REF] [--bootstrap-allowlist] [--baseline-file PATH] [--selftest]
"""

from __future__ import annotations

import argparse
import ast
import subprocess
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

ALLOWLIST_REL_PATH = "prosper/tools/ci/pytest_legacy_allowlist.txt"


def _is_unittest_base(
    base: ast.AST,
    unittest_aliases: set[str],
    testcase_aliases: set[str],
    tc_class_names: set[str],
) -> bool:
    """Determine if a class base expression resolves to unittest.TestCase or a subclass."""
    if isinstance(base, ast.Name):
        if base.id in testcase_aliases or base.id in tc_class_names:
            return True
    elif isinstance(base, ast.Attribute):
        if isinstance(base.value, ast.Name) and base.value.id in unittest_aliases:
            if base.attr == "TestCase" or base.attr in tc_class_names:
                return True
        elif base.attr == "TestCase":
            return True
    return False


def is_non_compliant(content: str, filename: str = "<input>") -> tuple[bool, str]:
    """Analyze Python source for pytest policy violations in collection scope.

    Returns (violates, reason).
    A compliant file:
      - Does NOT subclass unittest.TestCase (direct, aliased, or inherited).
      - Defines at least one eligible top-level test_* function or Test* class
        (no __init__) with eligible test_* methods.
    """
    try:
        tree = ast.parse(content, filename=filename)
    except Exception as exc:
        return True, f"syntax error parsing Python test: {exc}"

    unittest_aliases: set[str] = set()
    testcase_aliases: set[str] = set()

    # Step 1: Collect import bindings for unittest and TestCase
    for stmt in tree.body:
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                if alias.name == "unittest":
                    unittest_aliases.add(alias.asname or alias.name)
        elif isinstance(stmt, ast.ImportFrom):
            if stmt.module == "unittest":
                for alias in stmt.names:
                    if alias.name == "TestCase":
                        testcase_aliases.add(alias.asname or alias.name)
            elif stmt.module and stmt.module.startswith("unittest."):
                for alias in stmt.names:
                    if alias.name == "TestCase":
                        testcase_aliases.add(alias.asname or alias.name)

    tc_classes: list[str] = []
    tc_class_names: set[str] = set()

    # Step 2: Track classes inheriting from unittest.TestCase (including local hierarchy)
    # We iterate over top-level class definitions
    top_level_classes = [stmt for stmt in tree.body if isinstance(stmt, ast.ClassDef)]

    changed = True
    while changed:
        changed = False
        for cls in top_level_classes:
            if cls.name in tc_class_names:
                continue
            for base in cls.bases:
                if _is_unittest_base(base, unittest_aliases, testcase_aliases, tc_class_names):
                    tc_class_names.add(cls.name)
                    tc_classes.append(cls.name)
                    changed = True
                    break

    if tc_classes:
        return True, f"subclasses unittest.TestCase ({', '.join(tc_classes)})"

    # Step 3: Find eligible top-level test functions (pytest collection convention)
    eligible_test_funcs: list[str] = []
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if stmt.name.startswith("test_"):
                eligible_test_funcs.append(stmt.name)

    # Step 4: Find eligible top-level Test* classes and their test methods
    # Pytest ignores Test* classes that define __init__ or inherit from unittest.TestCase
    eligible_test_classes: list[str] = []
    for cls in top_level_classes:
        if not cls.name.startswith("Test"):
            continue
        # Pytest discovery convention: classes with __init__ are not collected
        has_init = any(isinstance(m, ast.FunctionDef) and m.name == "__init__" for m in cls.body)
        if has_init:
            continue

        test_methods = [
            m.name
            for m in cls.body
            if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and m.name.startswith("test_")
        ]
        if test_methods:
            eligible_test_classes.append(cls.name)

    if not (eligible_test_funcs or eligible_test_classes):
        return (
            True,
            "no eligible test_* functions or Test* classes with test methods (script-only or uncollected)",
        )

    return False, ""


def scan(prosper_root: Path) -> dict[str, str]:
    """Scan all test_*.py files under prosper_root.

    Returns dict of relative_path (relative to prosper_root) -> violation_reason for non-compliant files.
    """
    found = {}
    for p in prosper_root.rglob("test_*.py"):
        if any(part in EXCLUDED_DIR_NAMES for part in p.parts):
            continue
        rel = p.relative_to(prosper_root).as_posix()
        try:
            content = p.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            found[rel] = f"failed to read file: {exc}"
            continue
        violates, reason = is_non_compliant(content, filename=rel)
        if violates:
            found[rel] = reason
    return found


def parse_allowlist(text: str) -> set[str]:
    """Parse allowlist lines, stripping comments and empty lines."""
    return {line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")}


def check(
    found: dict[str, str] | set[str],
    candidate_allow: set[str],
    baseline_allow: set[str] | None = None,
) -> list[str]:
    """Check violations against candidate allowlist and immutable shrink-only baseline.

    Violations:
      - Files in found that are not in candidate_allow.
      - Allowlist entries in candidate_allow that no longer violate (migrated or gone).
      - Allowlist entries added to candidate_allow that were not in baseline_allow (shrink-only).
    """
    found_keys = set(found.keys()) if isinstance(found, dict) else set(found)
    errs: list[str] = []

    # 1. Non-exempt violating test files
    for f in sorted(found_keys - candidate_allow):
        reason = found[f] if isinstance(found, dict) else "violates pytest conventions"
        errs.append(
            f"{f}: {reason}; write it using pytest conventions (functions/asserts/fixtures). "
            f"See prosper/tests/AGENTS.md."
        )

    # 2. Stale allowlist entries (migrated or deleted)
    for f in sorted(candidate_allow - found_keys):
        errs.append(f"{f}: migrated or gone; remove it from pytest_legacy_allowlist.txt")

    # 3. Shrink-only baseline enforcement: cannot introduce new exemptions
    if baseline_allow is not None:
        new_exemptions = candidate_allow - baseline_allow
        for f in sorted(new_exemptions):
            errs.append(
                f"{f}: newly added to pytest_legacy_allowlist.txt; the allowlist may only shrink. "
                f"New Python tests must follow pytest conventions."
            )

    return errs


def _git_text(repo_root: Path, *args: str) -> str:
    """Read Git metadata, refusing an unavailable baseline instead of disabling the gate."""
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args], capture_output=True, text=True, check=False
    )
    if result.returncode:
        raise ValueError(f"cannot read immutable pytest baseline: {result.stderr.strip()}")
    return result.stdout


def load_baseline_allowlist(
    repo_root: Path, base_ref: str | None = None, *, bootstrap: bool = False
) -> set[str] | None:
    """Read a resolved commit's allowlist, or explicitly bootstrap from its legacy sources."""
    if not base_ref:
        return None
    if base_ref.startswith("-"):
        raise ValueError("baseline ref must be a revision, not a Git option")
    commit = _git_text(
        repo_root, "rev-parse", "--verify", "--end-of-options", f"{base_ref}^{{commit}}"
    ).strip()
    if len(commit) not in (40, 64) or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("baseline did not resolve to one immutable commit id")

    # --full-tree keeps these repository-relative paths correct even when called inside prosper/.
    entry = _git_text(repo_root, "ls-tree", "--full-tree", "-z", commit, "--", ALLOWLIST_REL_PATH)
    if entry:
        return parse_allowlist(_git_text(repo_root, "show", f"{commit}:{ALLOWLIST_REL_PATH}"))
    if not bootstrap:
        raise ValueError(
            "baseline commit predates pytest_legacy_allowlist.txt; "
            "initial rollout requires explicit --bootstrap-allowlist"
        )

    # A bootstrap must not exempt a newly added script merely because the base has no allowlist.
    paths = _git_text(
        repo_root, "ls-tree", "-r", "--full-tree", "--name-only", "-z", commit, "--", "prosper"
    )
    allowed = set()
    for path in paths.split("\0"):
        if not path:
            continue
        source_path = Path(path)
        if not source_path.name.startswith("test_") or source_path.suffix != ".py":
            continue
        if any(part in EXCLUDED_DIR_NAMES for part in source_path.parts):
            continue
        source = _git_text(repo_root, "show", f"{commit}:{path}")
        if is_non_compliant(source, filename=path)[0]:
            allowed.add(source_path.relative_to("prosper").as_posix())
    return allowed


def selftest() -> None:
    """Hand-built violating and compliant inputs through is_non_compliant() and check()."""
    # Compliant pytest test function
    comp1 = "def test_something():\n    assert 1 == 1\n"
    violates, reason = is_non_compliant(comp1)
    assert not violates, f"expected compliant, got {reason}"

    # Compliant pytest test class with test method
    comp2 = "class TestSomething:\n    def test_method(self):\n        pass\n"
    violates, reason = is_non_compliant(comp2)
    assert not violates, f"expected compliant, got {reason}"

    # Negative control 1: Empty Test* class (no test methods)
    viol_empty_cls = "class TestEmpty:\n    pass\n"
    violates, reason = is_non_compliant(viol_empty_cls)
    assert violates and "no eligible test_* functions" in reason, (
        f"expected violation, got {reason}"
    )

    # Negative control 2: Hidden test_* function nested inside another function
    viol_nested = "def helper():\n    def test_hidden():\n        pass\n"
    violates, reason = is_non_compliant(viol_nested)
    assert violates and "no eligible test_* functions" in reason, (
        f"expected violation, got {reason}"
    )

    # Negative control 3: test method inside non-Test helper class
    viol_helper_cls = "class Helper:\n    def test_in_helper(self):\n        pass\n"
    violates, reason = is_non_compliant(viol_helper_cls)
    assert violates and "no eligible test_* functions" in reason, (
        f"expected violation, got {reason}"
    )

    # Negative control 4: Test* class with __init__ (pytest ignores this class)
    viol_init_cls = "class TestWithInit:\n    def __init__(self):\n        pass\n    def test_m(self):\n        pass\n"
    violates, reason = is_non_compliant(viol_init_cls)
    assert violates and "no eligible test_* functions" in reason, (
        f"expected violation, got {reason}"
    )

    # Violating: standard unittest.TestCase
    viol_ut1 = (
        "import unittest\nclass MyTests(unittest.TestCase):\n    def test_a(self):\n        pass\n"
    )
    violates, reason = is_non_compliant(viol_ut1)
    assert violates and "subclasses unittest.TestCase" in reason

    # Violating: direct TestCase import
    viol_ut2 = (
        "from unittest import TestCase\n"
        "class MyTests(TestCase):\n"
        "    def test_a(self):\n        pass\n"
    )
    violates, reason = is_non_compliant(viol_ut2)
    assert violates and "subclasses unittest.TestCase" in reason

    # Violating: aliased TestCase import (`from unittest import TestCase as Base`)
    viol_ut_alias = (
        "from unittest import TestCase as Base\n"
        "class TestLegacy(Base):\n"
        "    def test_m(self):\n        pass\n"
    )
    violates, reason = is_non_compliant(viol_ut_alias)
    assert violates and "subclasses unittest.TestCase" in reason, (
        f"expected aliased unittest violation, got {reason}"
    )

    # Violating: aliased module import (`import unittest as ut; class T(ut.TestCase)`)
    viol_ut_mod_alias = (
        "import unittest as ut\n"
        "class TestLegacy(ut.TestCase):\n"
        "    def test_m(self):\n        pass\n"
    )
    violates, reason = is_non_compliant(viol_ut_mod_alias)
    assert violates and "subclasses unittest.TestCase" in reason

    # Violating: local inheritance chain (`class BaseA(TestCase): ... class DerivedB(BaseA): ...`)
    viol_ut_chain = (
        "from unittest import TestCase\n"
        "class BaseFixture(TestCase):\n"
        "    pass\n"
        "class DerivedTests(BaseFixture):\n"
        "    def test_m(self):\n        pass\n"
    )
    violates, reason = is_non_compliant(viol_ut_chain)
    assert violates and "subclasses unittest.TestCase" in reason

    # Compliant: unrelated base named NotATestCase or similar
    comp_unrelated_base = (
        "class NotATestCase:\n"
        "    pass\n"
        "class TestValid(NotATestCase):\n"
        "    def test_ok(self):\n        pass\n"
    )
    violates, reason = is_non_compliant(comp_unrelated_base)
    assert not violates, f"expected compliant with unrelated base, got {reason}"

    # Violating: script-only (no test_* funcs or Test* classes)
    viol_script = "import sys\nx = 1\nassert x == 1\n"
    violates, reason = is_non_compliant(viol_script)
    assert violates and "script-only" in reason

    # Check logic: basic
    assert check({"tests/a.py": "script-only"}, set())
    assert check(set(), {"tests/a.py"})
    assert not check({"tests/a.py": "script-only"}, {"tests/a.py"})

    # Check logic: shrink-only baseline enforcement
    base_exemptions = {"tests/old_legacy.py"}
    cand_exemptions_grown = {"tests/old_legacy.py", "tests/new_bad.py"}
    cand_found = {"tests/old_legacy.py": "legacy", "tests/new_bad.py": "legacy"}
    errs = check(cand_found, cand_exemptions_grown, baseline_allow=base_exemptions)
    assert any("newly added to pytest_legacy_allowlist.txt" in e for e in errs)

    # Check logic: baseline shrink allowed
    cand_exemptions_shrunk = set()
    errs_shrunk = check(set(), cand_exemptions_shrunk, baseline_allow=base_exemptions)
    assert errs_shrunk == []

    print("selftest ok")


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--selftest", action="store_true", help="run internal self-tests and exit")
    baseline = parser.add_mutually_exclusive_group()
    baseline.add_argument(
        "--base", help="git base ref (e.g. origin/main) to enforce shrink-only allowlist"
    )
    baseline.add_argument("--baseline-file", help="path to immutable baseline allowlist file")
    parser.add_argument(
        "--bootstrap-allowlist",
        action="store_true",
        help="allow initial rollout, exempting only legacy sources already in --base",
    )
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return 0
    if args.bootstrap_allowlist and not args.base:
        parser.error("--bootstrap-allowlist requires --base")

    repo_root = Path(__file__).resolve().parents[2]
    lst_path = repo_root / "prosper/tools/ci/pytest_legacy_allowlist.txt"
    if not lst_path.is_file():
        # Fallback if run from prosper/ subdirectory
        lst_path = Path(__file__).resolve().parents[1] / "ci/pytest_legacy_allowlist.txt"

    candidate_allow = parse_allowlist(lst_path.read_text(encoding="utf-8"))

    try:
        baseline_allow = None
        if args.baseline_file:
            baseline_allow = parse_allowlist(Path(args.baseline_file).read_text(encoding="utf-8"))
        elif args.base:
            baseline_allow = load_baseline_allowlist(
                repo_root, args.base, bootstrap=args.bootstrap_allowlist
            )
    except (OSError, ValueError) as exc:
        print(f"pytest policy could not evaluate baseline: {exc}", file=sys.stderr)
        return 2

    prosper_root = repo_root / "prosper"
    if not prosper_root.is_dir():
        prosper_root = repo_root

    found = scan(prosper_root)
    errs = check(found, candidate_allow, baseline_allow=baseline_allow)
    print("\n".join(errs) or "pytest policy ok")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
