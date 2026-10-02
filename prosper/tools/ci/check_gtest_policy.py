#!/usr/bin/env python3
"""Enforce: new C++ tests use GoogleTest, not a hand-rolled main().

Scans tests/ and frontends/**/tests/ for .cpp files defining `int main(`. Files in
tools/ci/gtest_legacy_allowlist.txt predate the policy and are tolerated until migrated.
A listed file that no longer has main() must be removed from the list (the list only shrinks).

Usage: check_gtest_policy.py [--root DIR] [--selftest]
"""
import re
import sys
from pathlib import Path

MAIN_RE = re.compile(r'^\s*(?:extern "C"\s+)?int\s+main\s*\(', re.M)


def scan(root: Path):
    found = set()
    for base in [root / "tests", *root.glob("frontends/**/tests")]:
        if not base.is_dir():
            continue
        for p in base.rglob("*.cpp"):
            if MAIN_RE.search(p.read_text(encoding="utf-8", errors="replace")):
                found.add(p.relative_to(root).as_posix())
    return found


def check(found, allow):
    errs = [f"{f}: defines main(); write it with GoogleTest (TEST/TEST_F + prosper_add_gtest). "
            f"See tests/AGENTS.md." for f in sorted(found - allow)]
    errs += [f"{f}: migrated or gone; remove it from gtest_legacy_allowlist.txt"
             for f in sorted(allow - found)]
    return errs


def selftest():
    assert check({"tests/a.cpp"}, set())
    assert check(set(), {"tests/a.cpp"})
    assert not check({"tests/a.cpp"}, {"tests/a.cpp"})
    assert MAIN_RE.search("int main() {") and not MAIN_RE.search("// int main(")
    print("selftest ok")


def main():
    if "--selftest" in sys.argv:
        selftest()
    root = Path(sys.argv[sys.argv.index("--root") + 1]) if "--root" in sys.argv else Path(__file__).resolve().parents[2]
    lst = root / "tools/ci/gtest_legacy_allowlist.txt"
    allow = {l.strip() for l in lst.read_text().splitlines() if l.strip() and not l.startswith("#")}
    errs = check(scan(root), allow)
    print("\n".join(errs) or "gtest policy ok")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
