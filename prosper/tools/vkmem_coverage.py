#!/usr/bin/env python3
"""Find device-memory allocations the GPU memory budget cannot see.

`gpu_memory_budget` answers "how much device memory does prosper hold, per heap?" (#3533). Its
answer is only as complete as its wiring, and the wiring is a per-call-site convention that nothing
in the compiler enforces: a new `vkAllocateMemory` beside an existing one compiles, links, runs, and
is simply absent from the count.

The two ways that fails are not symmetric, which is why this is a gate rather than a comment:

  * an unwired ALLOCATION under-reports -- the line says prosper holds less than it does, and the
    number is quietly useless for the question it was built to answer;
  * an unwired FREE over-reports FOREVER -- those bytes are never given back, so the count climbs
    monotonically and reads as a leak. That is worse, because a rising number is exactly the kind of
    evidence somebody acts on, and the action is chasing a leak that does not exist.

So every call in prosper's shipped code goes through `prosper::gpu::allocate_device_memory` /
`free_device_memory` in `src/gpu/diagnostics/gpu_memory_budget_vk.hpp`.

WHICH FILES ARE "SHIPPED" IS DERIVED, NOT LISTED, and that is the whole point of this rewrite.
The first version of this checker named `tests/fixtures/render_runner.h` as the one tests/ file
compiled into the emulator. That was wrong: `render_runner.h:11-12` includes `gpu_detile_upload.h`
and `mapped_staging.h`, which carry eight more raw calls between them and are just as shipped --
the renderer's texture staging cache and its device-local detile output buffer. The instrument
therefore under-reported by an unknown amount while its PR claimed completeness.

The checker's own positive control had certified that list, and could not have caught it: the
planted probe went under `src/`, INSIDE the roots being scanned, so it tested whether the scanner
fires rather than whether the scanned set is the right set. That is the charter's positive-control
rule exactly -- a control drawn from the same source tests the DISCRIMINATOR, never the DOMAIN.

So this walks the include graph from every shipped translation unit and scans whatever it reaches,
wherever it lives. A header that a shipped `.cpp` can reach is shipped, by definition, and adding
one to the renderer can no longer quietly widen the blind spot.

Exit 0 when every reachable site is wired, 1 when one is not, 2 when the check could not run -- so
"could not evaluate" is never confused with "nothing found".
"""
import re
import sys
from pathlib import Path

# Translation units that become the emulator. Test `.cpp` files are NOT roots: a fixture reaches the
# scanned set only when shipped code actually includes it.
ROOT_DIRS = ["src", "frontends"]
ROOT_SUFFIXES = {".cpp", ".cc", ".cxx"}
HEADER_SUFFIXES = {".hpp", ".h", ".hxx", ".inl"}

# Where a quoted #include is resolved from, after the including file's own directory. These mirror
# the target_include_directories in CMakeLists.txt.
INCLUDE_ROOTS = ["src", "frontends", "tests", "."]

# The wrappers themselves are the one place the raw calls belong.
ALLOWED = {"src/gpu/diagnostics/gpu_memory_budget_vk.hpp"}

# Vendored third-party code is exempt, and this is a real reported cap rather than an oversight:
# the charter requires such libraries to be vendored VERBATIM, so rewriting their call sites is not
# available to us. Dear ImGui's Vulkan backend allocates its own font atlas and per-frame vertex and
# index buffers, and those bytes are invisible to the budget. The instrument's output says it counts
# prosper's own allocations; this is part of what that sentence excludes.
EXEMPT_PREFIXES = ("third_party/",)

RAW = re.compile(r"(?<![A-Za-z0-9_:])(vkAllocateMemory|vkFreeMemory)\s*\(")
QUOTED_INCLUDE = re.compile(r'^\s*#\s*include\s*"([^"]+)"')


def resolve(target: str, including: Path, root: Path):
    """Resolve a quoted #include the way the compiler would, or None if it is not ours."""
    candidate = (including.parent / target).resolve()
    if candidate.is_file():
        return candidate
    for base in INCLUDE_ROOTS:
        candidate = (root / base / target).resolve()
        if candidate.is_file():
            return candidate
    return None


def reachable_files(root: Path):
    """Every file the shipped translation units can reach through quoted includes."""
    queue = []
    for name in ROOT_DIRS:
        directory = root / name
        if not directory.is_dir():
            raise FileNotFoundError(f"expected a directory at {name}")
        queue += [p for p in directory.rglob("*") if p.suffix in ROOT_SUFFIXES]
    if not queue:
        raise RuntimeError("found no shipped translation units; the scan would be vacuously clean")

    seen = set()
    while queue:
        path = queue.pop()
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            text = resolved.read_text(errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            match = QUOTED_INCLUDE.match(line)
            if not match:
                continue
            target = resolve(match.group(1), resolved, root)
            if target and target.suffix in (HEADER_SUFFIXES | ROOT_SUFFIXES):
                queue.append(target)
    return seen


def scan(root: Path):
    """Yield (relative path, line number, symbol) for every raw call in reachable code."""
    for path in sorted(reachable_files(root)):
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            continue                                   # outside the tree; not ours to police
        if relative in ALLOWED or relative.startswith(EXEMPT_PREFIXES):
            continue
        for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            # A comment is not a call. Keep this crude rather than clever: a false positive costs one
            # line of reading, and a false negative is the whole failure mode.
            stripped = line.lstrip()
            if stripped.startswith("//") or stripped.startswith("*"):
                continue
            for match in RAW.finditer(line):
                yield relative, number, match.group(1)


def self_test() -> str:
    """Prove the walk reaches a header only a TRANSITIVE include can reach. Returns "" when sound.

    This is the control the first version of this checker did not have. That one planted its probe
    under `src/`, inside the roots it already scanned, so it could only ever confirm that the scanner
    fires -- the charter's rule that a control drawn from the same source tests the DISCRIMINATOR and
    never the DOMAIN. It certified a hand-written file list that was missing two shipped headers.

    So the probe here sits three includes deep, in a directory that is NOT a scanned root, reachable
    only by following the graph: shipped .cpp -> a header -> a nested header. If the walk ever stops
    being transitive, this fails and the checker reports that it could not evaluate -- rather than
    reporting a clean tree it did not actually look at.
    """
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        fake = Path(directory)
        (fake / "src").mkdir()
        (fake / "frontends").mkdir()
        (fake / "elsewhere").mkdir()
        (fake / "src" / "ship.cpp").write_text('#include "middle.h"\n')
        (fake / "src" / "middle.h").write_text('#include "../elsewhere/deep.h"\n')
        (fake / "elsewhere" / "deep.h").write_text("void f() { vkFreeMemory(d, m, nullptr); }\n")
        found = list(scan(fake))
        if not any(name.endswith("deep.h") for name, _, _ in found):
            return ("the include walk did not reach a header three levels deep, so a clean result "
                    "would be a statement about files it never opened")

        # ...and the negative half: a wired call must NOT be reported, or the gate cries wolf and
        # gets ignored, which is the same outcome as not having it.
        (fake / "elsewhere" / "deep.h").write_text(
            "void f() { prosper::gpu::free_device_memory(d, m); }\n")
        if list(scan(fake)):
            return "a correctly wired call was reported as a bypass"
    return ""


def main() -> int:
    root = Path(__file__).resolve().parent.parent      # .../prosper
    broken = self_test()
    if broken:
        print(f"vkmem_coverage: the checker itself is not sound: {broken}", file=sys.stderr)
        return 2
    try:
        findings = list(scan(root))
        reached = len(reachable_files(root))
    except Exception as error:                          # noqa: BLE001 - report, do not mask
        print(f"vkmem_coverage: could not run: {error}", file=sys.stderr)
        return 2

    if not findings:
        print(f"vkmem_coverage: {reached} files reachable from shipped translation units; "
              "every device-memory call in them goes through the counted wrappers")
        print("vkmem_coverage: third_party/ is exempt and its allocations are therefore NOT in the "
              "budget -- Dear ImGui's Vulkan backend allocates its own font atlas and per-frame "
              "buffers, and the charter requires it vendored verbatim")
        return 0

    print("vkmem_coverage: these bypass the GPU memory budget and make its numbers wrong:\n")
    for relative, number, symbol in findings:
        fix = "allocate_device_memory" if symbol == "vkAllocateMemory" else "free_device_memory"
        print(f"  {relative}:{number}: {symbol} -> use prosper::gpu::{fix}")
    print('\nInclude "gpu/diagnostics/gpu_memory_budget_vk.hpp" and call the wrapper. An unwired '
          "free is the worse half: it never gives its bytes back, so the count climbs and reads as "
          "a leak that is not there.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
