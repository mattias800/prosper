#!/usr/bin/env python3
"""Every knob parsed by diagnostics/env_numeric.hpp has an arm in the site test, and vice versa.

WHY THIS EXISTS. tests/diagnostics/test_env_numeric_sites.cpp cannot CALL the sites it asserts on:
nearly every one caches its read in a function-local static, so an arm that armed the variable at
runtime would go vacuous rather than red once the static was initialised (#2214, and
tools/env/check_cached_env.py is the gate for that hazard). Each arm therefore MIRRORS its site --
same helper, same fallback, same cap -- which pins the contract but not the wiring. The drift that
mirror invites is one-directional and silent: somebody converts a twentieth site, forgets the arm,
and the suite stays green while the new site's fallback is never checked by anything.

So this compares two sets over the whole tree:

    parsed  -- the PROSPER_* name in the FIRST argument of an env_u64_or_default / _capped call
    armed   -- the PROSPER_* names listed in the site test's table

and fails on either difference. A name in `parsed` and not in `armed` is a converted site with no
arm; a name in `armed` and not in `parsed` is an arm whose site was reverted or renamed, i.e. an
assertion about code that no longer exists.

It deliberately checks NAMES only, not fallbacks. Matching a fallback textually would mean parsing
`16ull * 1024ull` against `16 * 1024`, which is brittle enough that the gate would be regenerated
rather than read -- and a gate nobody trusts is worse than none (see diag_gate_baseline.txt's own
header on the same point). The fallback is what the arm itself asserts.

Run standalone against a checkout, or via ctest as env_numeric_arms.
"""
import re
import sys
from pathlib import Path

# EVERY name-taking entry point in diagnostics/env_numeric.hpp, not just the two the first revision
# knew about. That omission was not theoretical: env_u64_or_report's call site was the one site with
# bespoke arithmetic and a non-numeric fallback -- the site most likely to drift -- and it was the
# one site this gate could not see, while the success line below asserted full coverage (#3267 N2).
# Adding a helper to that header without adding it here re-opens exactly that hole.
CALL_RE = re.compile(
    r'env_(?:u64_or_(?:default(?:_auto)?(?:_capped)?|report)|tristate_or_default)\s*\(\s*\n?\s*"(PROSPER_[A-Z_0-9]+)"')
ARM_RE = re.compile(r'"(PROSPER_[A-Z_0-9]+)"')
SKIP_DIRS = {"build-linux", "build-windows", "third_party", ".git", "tmpdir"}
TEST = Path("tests/diagnostics/test_env_numeric_sites.cpp")
# The helper's own unit test constructs calls with names that are deliberately not call sites --
# it is testing the parser, not a knob -- so it is not a source of `parsed`.
EXEMPT_SOURCES = {Path("frontends/shared/tests/test_write_watch_policy.cpp")}


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    parsed: dict[str, list[str]] = {}
    for path in sorted(root.rglob("*")):
        if path.suffix not in (".cpp", ".hpp", ".h", ".cc"):
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        rel = path.relative_to(root)
        if rel == TEST or rel in EXEMPT_SOURCES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for name in CALL_RE.findall(text):
            parsed.setdefault(name, []).append(str(rel))

    test_path = root / TEST
    if not test_path.is_file():
        print(f"[env-arms] FAIL: {TEST} is missing")
        return 1
    body = test_path.read_text(encoding="utf-8", errors="replace")
    table = body[body.index("static const Site kSites[]"):body.index("int main()")]
    armed = set(ARM_RE.findall(table))

    missing = sorted(set(parsed) - armed)
    stale = sorted(armed - set(parsed))
    for name in missing:
        print(f"[env-arms] FAIL: {name} is parsed by env_numeric at "
              f"{', '.join(sorted(set(parsed[name])))} but has no arm in {TEST}")
    for name in stale:
        print(f"[env-arms] FAIL: {TEST} arms {name}, but no call site parses it with env_numeric "
              f"any more -- the arm asserts about code that is gone")
    if missing or stale:
        return 1
    # Print the PARSED count, not the armed one. They are equal here by construction -- both
    # differences were just checked -- but the claim being made is about coverage of the call sites,
    # and a success line should be phrased in the quantity it is asserting about. The first revision
    # printed the armed count and so read "27 ... every one armed" while 28 were parsed.
    print(f"[env-arms] ok: {len(parsed)} knob(s) parsed by env_numeric, every one armed "
          f"({len(armed)} arm name(s) in the table)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
