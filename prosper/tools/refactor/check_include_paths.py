#!/usr/bin/env python3
"""check_include_paths.py -- targets that compile project headers without the include root.

`prosper_core` publishes `src` as a PUBLIC include directory, so anything linking it can spell an
include `gpu/state/render_state.hpp` or `hle/kernel/hle_kernel.hpp` -- the depth-independent form
every move in tools/refactor/ normalises to. A handful of test targets do NOT link prosper_core:
they compile a header-only slice of the tree directly. Those inherit nothing, so they only ever
worked because the file they wanted sat at a path their own relative include happened to reach.

The moment a module moves into a folder, that spelling breaks -- and it breaks ONE target at a time,
at the bottom of a ten-minute build, which is how a mechanical restructure turns into an afternoon
of rebuild-and-guess. This finds all of them in one pass, before the first compile.

It is deliberately a REPORTER, not a fixer. What a target should see is a judgement -- a genuinely
standalone test that pulls in one header on purpose is different from one that wants the whole tree
-- and the repository has few enough of these that a person can decide each.

  python3 prosper/tools/refactor/check_include_paths.py            # exits 1 if any are found
"""

import pathlib
import re
import subprocess
import sys

PROJECT_PREFIXES = ("hle", "gpu", "host", "self", "loader")


def main() -> int:
    root = pathlib.Path(subprocess.run(["git", "rev-parse", "--show-toplevel"],
                                       capture_output=True, text=True, check=True).stdout.strip())
    cmake = root / "prosper" / "CMakeLists.txt"
    text = cmake.read_text()

    # A target may be declared more than once, under different `if()` arms -- test_game_compute is
    # declared twice with different configurations. Collect every declaration, or a fix applied to
    # the first one silently leaves the second broken.
    targets: dict[str, list[str]] = {}
    for m in re.finditer(r"add_executable\(\s*([A-Za-z0-9_]+)([^)]*)\)", text, re.S):
        targets.setdefault(m.group(1), []).append(m.group(2))

    linked = set(re.findall(r"target_link_libraries\(\s*([A-Za-z0-9_]+)[^)]*prosper_core", text, re.S))
    has_inc = set(re.findall(r"target_include_directories\(\s*([A-Za-z0-9_]+)[^)]*\bsrc\b", text, re.S))

    want = re.compile(r'#include\s+"(?:' + "|".join(PROJECT_PREFIXES) + r')/')
    findings: list[tuple[str, str]] = []
    for name, blocks in sorted(targets.items()):
        if name in linked or name in has_inc:
            continue
        for block in blocks:
            for rel in re.findall(r"([A-Za-z0-9_/\.\-]+\.(?:cpp|cc))", block):
                path = root / "prosper" / rel
                if path.exists() and want.search(path.read_text()):
                    findings.append((name, rel))
                    break

    findings = sorted(set(findings))
    for name, rel in findings:
        print(f"  [FAIL] {name} spells a project include but neither links prosper_core nor has "
              f"src on its include path  ({rel})")
    if findings:
        print(f"\n  {len(set(n for n, _ in findings))} target(s); add to prosper/CMakeLists.txt:")
        for name in sorted(set(n for n, _ in findings)):
            print(f"    target_include_directories({name} PRIVATE src)")
        return 1
    print(f"  [ok]   {len(targets)} target(s): every one that spells a project include can resolve it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
