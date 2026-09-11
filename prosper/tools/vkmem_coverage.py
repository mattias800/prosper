#!/usr/bin/env python3
"""Find device-memory allocations the GPU memory budget cannot see.

`gpu_memory_budget` answers "how much device memory does prosper hold, per heap?" (#3533). Its
answer is only as complete as its wiring, and the wiring is a per-call-site convention that nothing
in the compiler enforces: a new `vkAllocateMemory` beside an existing one compiles, links, runs, and
is simply absent from the count.

The two ways that fails are not symmetric, which is why this check exists rather than a comment:

  * an unwired ALLOCATION under-reports -- the line says prosper holds less than it does, and the
    number is quietly useless for the question it was built to answer;
  * an unwired FREE over-reports FOREVER -- those bytes are never given back, so the count climbs
    monotonically and reads as a leak. That is worse, because a rising number is exactly the kind of
    evidence somebody acts on, and the action is chasing a leak that does not exist.

So: every call in prosper's shipped code goes through `prosper::gpu::allocate_device_memory` /
`free_device_memory` in `src/gpu/diagnostics/gpu_memory_budget_vk.hpp`.

TESTS AND STANDALONE TOOLS ARE DELIBERATELY EXEMPT. They create their own short-lived devices, are
not the emulator, and counting them would mix a fixture's staging buffer into a figure that claims
to describe a running title. `tests/fixtures/render_runner.h` is the exception to the exception --
despite its path it IS the shipped renderer backend (`src/gpu/execute/gpu_executor.cpp` and
`frontends/shared/live/live_renderer.cpp` both include it), so it is scanned.

Exit 0 when every shipped site is wired, 1 when one is not, 2 when the check could not run -- so
"could not evaluate" is never confused with "nothing found".
"""
import re
import sys
from pathlib import Path

# Shipped code: the emulator itself. Paths are repo-relative to the `prosper/` directory.
SCANNED_ROOTS = ["src", "frontends"]
# ...plus this one file, which lives under tests/ but is compiled into the shipped renderer.
SCANNED_EXTRA = ["tests/fixtures/render_runner.h"]
# The wrappers themselves are the one place the raw calls belong.
ALLOWED = {"src/gpu/diagnostics/gpu_memory_budget_vk.hpp"}

RAW = re.compile(r"(?<![A-Za-z0-9_:])(vkAllocateMemory|vkFreeMemory)\s*\(")
SOURCE_SUFFIXES = {".cpp", ".hpp", ".h", ".cc", ".cxx"}


def scan(root: Path):
    """Yield (relative path, line number, symbol) for every raw call in shipped code."""
    files = []
    for name in SCANNED_ROOTS:
        directory = root / name
        if not directory.is_dir():
            raise FileNotFoundError(f"expected a directory at {name}")
        files += [p for p in directory.rglob("*") if p.suffix in SOURCE_SUFFIXES]
    for name in SCANNED_EXTRA:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"expected a file at {name}")
        files.append(path)

    for path in sorted(set(files)):
        relative = path.relative_to(root).as_posix()
        if relative in ALLOWED:
            continue
        for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            # A declaration or a comment is not a call. Keep this crude rather than clever: a false
            # positive costs one line of reading, and a false negative is the whole failure mode.
            stripped = line.lstrip()
            if stripped.startswith("//") or stripped.startswith("*"):
                continue
            for match in RAW.finditer(line):
                yield relative, number, match.group(1)


def main() -> int:
    root = Path(__file__).resolve().parent.parent      # .../prosper
    try:
        findings = list(scan(root))
    except Exception as error:                          # noqa: BLE001 - report, do not mask
        print(f"vkmem_coverage: could not run: {error}", file=sys.stderr)
        return 2

    if not findings:
        print("vkmem_coverage: every shipped device-memory call goes through the counted wrappers")
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
