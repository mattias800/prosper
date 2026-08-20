#!/usr/bin/env python3
"""check_include_paths.py -- targets that spell a project include they cannot resolve.

prosper has two include ROOTS. `prosper_core` publishes `src` as PUBLIC, so anything linking it can
spell `gpu/state/render_state.hpp` or `hle/kernel/hle_kernel.hpp`; `prosper_performance_capture`
publishes `frontends`, which is what makes `shared/live/live_renderer.hpp` resolve. Those
depth-independent forms are what tools/refactor/move_module.py normalises every include to, and they
are the reason a second move does not touch consumers at all.

A target that inherits neither root only ever worked because its own relative include happened to
reach the file. The moment a module moves into a folder that breaks -- ONE target at a time, at the
bottom of a ten-minute build, because the compiler reports the first and stops. This finds all of
them in one pass, before the first compile.

Two things it must do that the obvious version does not, and both were found the expensive way:

  TRANSITIVE LINKAGE, or it over-reports. `target_link_libraries(X PRIVATE prosper_live_renderer)`
  gives X the `src` root, because prosper_live_renderer links prosper_core PUBLIC. Checking only for
  a direct prosper_core mention flagged three targets that were already fine, and three redundant
  lines were added to CMakeLists.txt before a reviewer caught it.

  TRANSITIVE INCLUDES, or it under-reports -- the dangerous direction. A target's .cpp may include
  nothing but `render_runner.h`, and THAT header includes `shared/rtt/mrt_extent.hpp`. The include
  that needs a root is two files away from anything CMakeLists.txt names. Missing this is what let a
  move reach a full build before failing.

It reports; it does not edit. What a target should see is a judgement, and there are few enough of
these that a person can make it.

  python3 prosper/tools/refactor/check_include_paths.py      # exits 1 if any target cannot resolve
"""

import pathlib
import re
import subprocess
import sys

# Which include ROOT makes each canonical prefix resolve. Encoded here rather than in a reader's
# head: the wrong one produces a suggestion that looks right and does not work.
PREFIX_ROOT = {"hle": "src", "gpu": "src", "host": "src", "self": "src", "loader": "src",
               "shared": "frontends", "fixtures": "tests"}
ROOT_DIR = {"src": "src", "frontends": "frontends", "tests": "tests"}
# Libraries that publish a root PUBLIC, so every dependent inherits it.
PUBLISHERS = {"prosper_core": "src", "prosper_performance_capture": "frontends"}

INCLUDE_RE = re.compile(r'#include\s+"([^"<>]+)"')


def roots_needed(path: pathlib.Path, base: pathlib.Path,
                 cache: dict[pathlib.Path, frozenset[str]],
                 stack: frozenset[pathlib.Path] = frozenset()) -> frozenset[str]:
    """Include roots PATH needs, following project includes transitively.

    `stack` breaks include cycles, which real headers do form. A cycle returns the empty set for the
    revisited node rather than recursing -- its contribution is already being computed by the frame
    below, so nothing is lost.
    """
    if path in cache:
        return cache[path]
    if path in stack or not path.exists():
        return frozenset()
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return frozenset()

    found: set[str] = set()
    deeper = stack | {path}
    for spelling in INCLUDE_RE.findall(text):
        head = spelling.split("/", 1)[0]
        if head in PREFIX_ROOT and "/" in spelling:
            found.add(PREFIX_ROOT[head])
            target = base / ROOT_DIR[PREFIX_ROOT[head]] / spelling
            found |= roots_needed(target, base, cache, deeper)
            continue
        # A relative or same-directory include: follow it so a header two hops away is still seen.
        target = (path.parent / spelling)
        if target.exists():
            found |= roots_needed(target, base, cache, deeper)
    result = frozenset(found)
    cache[path] = result
    return result


def main() -> int:
    root = pathlib.Path(subprocess.run(["git", "rev-parse", "--show-toplevel"],
                                       capture_output=True, text=True, check=True).stdout.strip())
    base = root / "prosper"
    text = (base / "CMakeLists.txt").read_text()

    # A target may be declared more than once under different if() arms -- test_game_compute is.
    # Collect every declaration, or a fix applied to the first silently leaves the second broken.
    targets: dict[str, list[str]] = {}
    for m in re.finditer(r"add_executable\(\s*([A-Za-z0-9_]+)([^)]*)\)", text, re.S):
        targets.setdefault(m.group(1), []).append(m.group(2))

    # link graph, then propagate the published roots to a fixed point
    links: dict[str, set[str]] = {}
    for m in re.finditer(r"target_link_libraries\(\s*([A-Za-z0-9_]+)([^)]*)\)", text, re.S):
        deps = set(re.findall(r"[A-Za-z0-9_:]+", m.group(2))) - {"PRIVATE", "PUBLIC", "INTERFACE"}
        links.setdefault(m.group(1), set()).update(deps)

    # VISIBILITY IS LOAD-BEARING, and getting it wrong is silent in the direction that hurts.
    # `target_include_directories(X PRIVATE tests)` lets X compile against the tests root and gives
    # its dependents NOTHING. A closure that ignores the keyword propagates it anyway, and every
    # dependent then looks satisfied -- which is how test_shared_vulkan_device passed this check and
    # failed the build on `fixtures/render_runner.h`: it links prosper_live_renderer, whose `tests`
    # entry is PRIVATE (prosper/CMakeLists.txt:1755). So two sets per target: what it can USE, and
    # the strictly smaller set it EXPORTS.
    own: dict[str, set[str]] = {}        # usable when compiling this target
    exported: dict[str, set[str]] = {}   # inherited by anything linking it
    for m in re.finditer(r"target_include_directories\(\s*([A-Za-z0-9_]+)([^)]*)\)", text, re.S):
        visibility = "PRIVATE"
        for word in re.findall(r"[A-Za-z0-9_/\.\-]+", m.group(2)):
            if word in ("PRIVATE", "PUBLIC", "INTERFACE"):
                visibility = word
                continue
            if word in ROOT_DIR:
                own.setdefault(m.group(1), set()).add(word)
                if visibility in ("PUBLIC", "INTERFACE"):
                    exported.setdefault(m.group(1), set()).add(word)
    for lib, rootname in PUBLISHERS.items():
        exported.setdefault(lib, set()).add(rootname)

    changed = True
    while changed:      # a PUBLIC root travels through any number of PUBLIC links
        changed = False
        for name, deps in links.items():
            for dep in deps:
                gained = exported.get(dep, set()) - exported.get(name, set())
                # Only re-export what this target itself links PUBLIC-ly; CMake's own rule. Being
                # conservative here can only over-report, which is the safe direction.
                if gained and re.search(r"target_link_libraries\(\s*" + re.escape(name)
                                        + r"[^)]*PUBLIC[^)]*" + re.escape(dep), text, re.S):
                    exported.setdefault(name, set()).update(gained)
                    changed = True
    have: dict[str, set[str]] = {k: set(v) for k, v in own.items()}
    for name, deps in links.items():
        for dep in deps:
            have.setdefault(name, set()).update(exported.get(dep, set()))

    cache: dict[pathlib.Path, frozenset[str]] = {}
    findings: list[tuple[str, str, str]] = []
    for name, blocks in sorted(targets.items()):
        for block in blocks:
            for rel in re.findall(r"([A-Za-z0-9_/\.\-]+\.(?:cpp|cc))", block):
                path = base / rel
                if not path.exists():
                    continue
                for miss in sorted(roots_needed(path, base, cache) - have.get(name, set())):
                    findings.append((name, rel, miss))

    findings = sorted(set(findings))
    for name, rel, miss in findings:
        print(f"  [FAIL] {name} reaches a `{miss}`-rooted include but does not have {miss} on its "
              f"include path  ({rel})")
    if findings:
        need: dict[str, set[str]] = {}
        for name, _rel, miss in findings:
            need.setdefault(name, set()).add(miss)
        print(f"\n  {len(need)} target(s); add to prosper/CMakeLists.txt:")
        for name, roots in sorted(need.items()):
            print(f"    target_include_directories({name} PRIVATE {' '.join(sorted(roots))})")
        return 1
    print(f"  [ok]   {len(targets)} target(s): every project include reachable from each one "
          f"resolves, following includes and PUBLIC linkage transitively")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
