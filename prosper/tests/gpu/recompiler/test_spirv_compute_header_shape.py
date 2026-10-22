#!/usr/bin/env python3
"""Guard that SpirvCompute's implementation stays out of the header it is declared in.

rdna2_to_spirv_internal.hpp is compiled by eleven translation units. In #2739's pass 147 of
SpirvCompute's methods moved to src/gpu/recompiler/spirv/, taking the header from 5,390 lines to
3,278 -- roughly 23,000 fewer lines parsed per build.

Nothing stops it growing back. A method written inline in the class body compiles, passes every
test, and is invisible in review unless somebody is counting: the header gets a little larger with
each one, and the cost is spread across eleven translation units where nobody attributes it.

So this asserts the SHAPE rather than the size: no method body in the class may exceed a small
number of lines. The bound is on the individual body, not the file, because a file-size bound has to
be bumped for every legitimate addition and stops being read.

Deliberately textual. The point is to fail for whoever adds the next inline method, including on a
machine with no libclang and no compile database, and a brace-matching scan over one class is enough
to say how long each body is. It is not a parser and does not need to be -- it never has to be right
about WHICH method, only about whether one is too long.
"""
import pathlib
import re
import sys

HEADER = pathlib.Path(__file__).resolve().parents[3] / "src/gpu/recompiler/rdna2_spirv_compute.inl"
CLASS = "SpirvCompute"
LARGE = 40          # a body over this is implementation, not an accessor

# All bodies over LARGE now live in source files. Expand the class's member fragments so a new
# large body in any section is still counted, including without libclang or a compile database.
#
# This is a RATCHET. It may only go down. If those categories are taught to the tool, or a large
# method is written out-of-line by hand, lower it in the same change.
MAX_LARGE_BODIES = 0

fails = 0


def check(cond, label):
    global fails
    if not cond:
        print(f"  [FAIL] {label}")
        fails += 1
    else:
        print(f"  [ok  ] {label}")


def strip_noise(line: str) -> str:
    line = re.sub(r"//.*", "", line)
    line = re.sub(r'"(\\.|[^"\\])*"', '""', line)
    line = re.sub(r"'(\\.|[^'\\])*'", "''", line)
    return line


def class_span(lines):
    """[start, end) of the class body, by brace depth from its opening line."""
    start = next(i for i, l in enumerate(lines)
                 if re.match(r"\s*(struct|class)\s+" + CLASS + r"\b", l))
    depth = 0
    for i in range(start, len(lines)):
        code = strip_noise(lines[i])
        for ch in code:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return start, i
    raise AssertionError("class body never closes")


def method_bodies(lines, start, end):
    """(line, length) for every brace block that opens at class-member depth."""
    out = []
    depth = 0
    open_at = None
    for i in range(start, end + 1):
        for ch in strip_noise(lines[i]):
            if ch == "{":
                depth += 1
                if depth == 2 and open_at is None:      # 1 = class body, 2 = a member's body
                    open_at = i
            elif ch == "}":
                if depth == 2 and open_at is not None:
                    out.append((open_at + 1, i - open_at + 1))
                    open_at = None
                depth -= 1
    return out


def header_lines(path):
    """Read the class and its included member sections in their declaration order."""
    lines = []
    for line in path.read_text().splitlines():
        include = re.match(r'\s*#include "gpu/recompiler/([^"/]+\.inl)"', line)
        if include:
            lines.extend(header_lines(path.parent / include[1]))
        else:
            lines.append(line)
    return lines


def main() -> int:
    print("== test_spirv_compute_header_shape ==")
    if not HEADER.exists():
        print(f"  [FAIL] header not found: {HEADER}")
        return 1
    lines = header_lines(HEADER)
    start, end = class_span(lines)
    bodies = method_bodies(lines, start, end)

    # The scan must actually find bodies, or every assertion below passes vacuously -- the shape
    # this project records as a discriminator that cannot show its lever moved.
    check(len(bodies) > 20, f"the scan found {len(bodies)} member bodies (a zero here would make "
                            f"the bound below meaningless)")

    over = [(ln, n) for ln, n in bodies if n > LARGE]
    check(len(over) <= MAX_LARGE_BODIES,
          f"at most {MAX_LARGE_BODIES} method bodies over {LARGE} lines stay inline "
          f"(found {len(over)})")
    if len(over) > MAX_LARGE_BODIES:
        print("         New large inline bodies: "
              + ", ".join(f"line {ln} ({n}L)" for ln, n in over[MAX_LARGE_BODIES:][:5]))
        print("         Move it to src/gpu/recompiler/spirv/:")
        print("         Follow prosper/tools/refactor/README.md for a tool-driven move.")

    # Ratchet: if the count has DROPPED, the bound is stale and should be lowered with the change
    # that dropped it, or it silently stops guarding what it claims to.
    if len(over) < MAX_LARGE_BODIES:
        print(f"  [info ] only {len(over)} large bodies remain; lower MAX_LARGE_BODIES to that "
              f"in the change that moved them")

    longest = max((n for _, n in bodies), default=0)
    print(f"  [info ] {len(bodies)} inline bodies, longest {longest} line(s), "
          f"header {len(lines)} lines")
    print("== PASS ==" if not fails else f"== FAIL: {fails} ==")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
