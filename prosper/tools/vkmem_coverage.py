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

# Where a quoted #include is resolved from, after the including file's own directory.
#
# These are READ OUT OF CMakeLists.txt rather than listed here. A hand-written list is the thing this
# checker already got wrong once, and a list of include roots fails the same way a list of files did:
# it looks right, it is never re-checked, and what it omits is dropped in silence. Reading the build
# file means adding a root to a target adds it here too.
#
# The fallbacks are only for a tree where that parse finds nothing.
FALLBACK_INCLUDE_ROOTS = ["src", "frontends", "tests", "."]


def include_roots(root: Path):
    """Every directory a target_include_directories line names, plus the repository root."""
    found = []
    build_file = root / "CMakeLists.txt"
    text = build_file.read_text(errors="replace") if build_file.is_file() else ""
    for match in re.finditer(r"target_include_directories\s*\(([^)]*)\)", text, re.S):
        for token in match.group(1).split():
            if token in ("PRIVATE", "PUBLIC", "INTERFACE", "SYSTEM", "BEFORE"):
                continue
            if token.startswith("$") or token.startswith("#"):
                continue
            candidate = token.strip('"')
            if (root / candidate).is_dir() and candidate not in found:
                found.append(candidate)
    for fallback in FALLBACK_INCLUDE_ROOTS:
        if fallback not in found and (root / fallback).is_dir():
            found.append(fallback)
    return found

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


def resolve(target: str, including: Path, root: Path, roots):
    """Resolve a quoted #include the way the compiler would, or None if it is not ours."""
    candidate = (including.parent / target).resolve()
    if candidate.is_file():
        return candidate
    for base in roots:
        candidate = (root / base / target).resolve()
        if candidate.is_file():
            return candidate
    return None


def reachable_files(root: Path, dropped=None):
    """Every file the shipped translation units can reach through quoted includes.

    `dropped` collects includes that did not resolve but whose basename exists somewhere in the tree.
    Those are edges this walk could not follow, and each one is a subtree it may never have opened --
    so they are reported and the checker refuses to answer, rather than calling a tree clean on the
    strength of a graph it knows is incomplete. This is the backstop that makes the root list above
    non-load-bearing: if the list is ever wrong, this says so instead of hiding it.
    """
    roots = include_roots(root)
    in_tree = {}
    for path in root.rglob("*"):
        if path.suffix in (HEADER_SUFFIXES | ROOT_SUFFIXES | {".inc"}):
            in_tree.setdefault(path.name, path)
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
            spelled = match.group(1)
            target = resolve(spelled, resolved, root, roots)
            if target is None:
                if dropped is not None and Path(spelled).name in in_tree:
                    dropped.add((relative_or_name(resolved, root), spelled))
                continue
            if target.suffix in (HEADER_SUFFIXES | ROOT_SUFFIXES | {".inc"}):
                queue.append(target)
    return seen


def relative_or_name(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def scan(root: Path, dropped=None):
    """Yield (relative path, line number, symbol) for every raw call in reachable code."""
    for path in sorted(reachable_files(root, dropped)):
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
    """Prove the walk does what the answer depends on. Returns "" when sound.

    Four properties, one per way this has actually been wrong or could be:

      1. it follows a TRANSITIVE chain, not just a shipped file's direct includes;
      2. it resolves an include that only a `target_include_directories` root can reach, which is the
         branch the previous control never executed at all -- its probe resolved through the
         including file's own directory every time, so dropping the entire root list still passed it;
      3. it REFUSES, rather than answering, when an include cannot be followed;
      4. it does not report a correctly wired call, because a gate that cries wolf gets ignored and
         that is the same outcome as having no gate.

    The first control this checker had tested none of these. It planted its probe under `src/`,
    inside the roots already scanned, so it could only confirm that the scanner fires -- the
    charter's rule that a control drawn from the same source tests the DISCRIMINATOR, never the
    DOMAIN. It certified a file list that was missing two shipped headers.
    """
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        fake = Path(directory)
        for name in ("src", "frontends", "elsewhere", "vendor_root"):
            (fake / name).mkdir()
        (fake / "CMakeLists.txt").write_text(
            "target_include_directories(thing PRIVATE src vendor_root)\n")
        (fake / "src" / "ship.cpp").write_text('#include "middle.h"\n')
        # A second shipped root, so dropping one from ROOT_DIRS fails here instead of silently
        # halving the scan. The reviewer of #3565 noted the previous control could not see that.
        (fake / "frontends" / "ship2.cpp").write_text('#include "../elsewhere/frontend_only.h"\n')
        (fake / "elsewhere" / "frontend_only.h").write_text(
            "void h() { vkFreeMemory(d, m, nullptr); }\n")
        (fake / "src" / "middle.h").write_text(
            '#include "../elsewhere/deep.h"\n#include "only_via_root.h"\n')
        (fake / "elsewhere" / "deep.h").write_text("void f() { vkFreeMemory(d, m, nullptr); }\n")
        # Reachable ONLY through the CMake-declared `vendor_root`: not beside its includer, and not
        # under any fallback root.
        (fake / "vendor_root" / "only_via_root.h").write_text(
            "void g() { vkAllocateMemory(d, &i, nullptr, &o); }\n")

        dropped = set()
        found = list(scan(fake, dropped))
        names = [name for name, _, _ in found]
        if dropped:
            return f"the walk could not follow an include it should have: {sorted(dropped)}"
        if not any(n.endswith("deep.h") for n in names):
            return ("the walk did not reach a header three levels deep, so a clean result would be "
                    "a statement about files it never opened")
        if not any(n.endswith("only_via_root.h") for n in names):
            return ("the walk did not resolve an include through a target_include_directories root, "
                    "so any header reachable only that way is invisible to it")
        if not any(n.endswith("frontend_only.h") for n in names):
            return ("the walk did not start from every shipped root, so a whole tree of translation "
                    "units is unscanned")

        # 3. an unfollowable include, whose basename DOES exist in the tree, must be announced.
        (fake / "src" / "middle.h").write_text('#include "deep.h"\n#include "only_via_root.h"\n')
        dropped = set()
        list(scan(fake, dropped))
        if not dropped:
            return ("an include that could not be followed was dropped in silence; a clean result "
                    "would then be a statement about a graph known to be incomplete")

        # 4. the negative half. EVERY raw call in the fake tree must be wired for this to mean
        # anything -- an earlier version left the frontend probe raw, so this arm reported a bypass
        # that was really its own leftover and the control failed on correct code.
        (fake / "src" / "middle.h").write_text('#include "../elsewhere/deep.h"\n')
        (fake / "elsewhere" / "deep.h").write_text(
            "void f() { prosper::gpu::free_device_memory(d, m); }\n")
        (fake / "elsewhere" / "frontend_only.h").write_text(
            "void h() { prosper::gpu::free_device_memory(d, m); }\n")
        dropped = set()
        if list(scan(fake, dropped)):
            return "a correctly wired call was reported as a bypass"
    return ""


def main() -> int:
    root = Path(__file__).resolve().parent.parent      # .../prosper
    dropped = set()
    try:
        # The checker proves itself before it is believed -- and inside the same guard, so a crash in
        # the control is "could not evaluate" rather than a traceback that reads as a tool bug.
        broken = self_test()
        if broken:
            print(f"vkmem_coverage: the checker itself is not sound: {broken}", file=sys.stderr)
            return 2
        findings = list(scan(root, dropped))
        reached = len(reachable_files(root))
    except Exception as error:                          # noqa: BLE001 - report, do not mask
        print(f"vkmem_coverage: could not run: {error}", file=sys.stderr)
        return 2

    if dropped:
        print("vkmem_coverage: could not follow these includes, so the graph is incomplete and a "
              "clean result would not mean anything:\n", file=sys.stderr)
        for where, spelled in sorted(dropped):
            print(f"  {where}: #include \"{spelled}\"", file=sys.stderr)
        print("\nAdd the directory to a target_include_directories line, or teach resolve() the "
              "form -- do not silence this.", file=sys.stderr)
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
