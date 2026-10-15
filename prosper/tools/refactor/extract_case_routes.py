#!/usr/bin/env python3
"""extract_case_routes.py -- move a giant `case` body out of a switch into member-function routes.

`emit_alu` is one function whose `case Rdna2Format::X: { ... }` bodies run to a thousand lines
each, and almost every body is a sequence of top-level `if (shape) { ...; return true; }`
statements, each handling one shape of the instruction, followed by a generic tail. clangd's
ExtractFunction refuses all of them (each contains `return`), so the move has to be textual -- and
a textual move must prove it moved text and nothing else. This tool does exactly that move:

  * a PLAN names the first line of each top-level statement run (a "segment") and gives it a route
    name and an output file. Segments run from their first line to the line before the next plan
    entry; the last ends at the body's last line. Comments preceding an `if` belong to the segment
    of that `if` -- put the plan line on the comment.
  * every route is written as
        std::optional<bool> <Class>::<name>() {
        <segment, dedented>
            return std::nullopt;
        }
    so an unchanged `return X;` inside the segment still means "emit_alu returns X", and falling off
    the end means "not this shape, try the next route". The class is expected to expose the case's
    locals (b, rs, in, ok, ...) as members, so the moved text compiles unchanged.
  * a route named `-` is RAW: its dedented text is appended to the output file without a wrapper,
    for the tail that becomes ordinary code.
  * with --replace, the body lines are replaced in the source by one line of text, indented as
    the body was (--dedent + 4 spaces).

Checks, each of which can fail:
  1. TILING. Plan lines are strictly increasing, the first is the body's first line, all lie inside
     the body. So the segments cover the body exactly once.
  2. INDENT. Every non-blank line of the body is indented by at least --dedent spaces, so dedenting
     is reversible: re-indenting each segment reproduces the original lines byte for byte (checked).
  3. BALANCE. Each extracted segment has balanced braces/parentheses outside strings and comments,
     so a boundary never falls inside a statement.
  4. NO ESCAPE. No extracted segment has a `break`/`continue` at its own top level; those would
     change meaning once the segment is a function.
  5. READ-BACK. After writing, every segment's dedented text is found verbatim in its output file.
What this does NOT establish: that the result compiles (identifier visibility is the class's job)
or that behaviour is unchanged. The compiler and the corpus gate establish those.

  python3 prosper/tools/refactor/extract_case_routes.py --selftest
  python3 prosper/tools/refactor/extract_case_routes.py --file F --body 252:1478 --class C \\
      --plan plan.txt [--dedent 8] [--replace 'return lower_x(...);'] [--dry-run]

Plan syntax, one segment per line: `<first line> <route name or -> <output file>`; `#` comments.
"""

import argparse
import pathlib
import sys
import tempfile


def strip_code(text: str) -> str:
    """Remove comments and string/char literals so bracket counting sees only code."""
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            i = n if j < 0 else j + 2
        elif c in "\"'":
            j = i + 1
            while j < n and text[j] != c:
                j += 2 if text[j] == "\\" else 1
            i = j + 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def balanced(text: str) -> bool:
    code = strip_code(text)
    return all(code.count(a) == code.count(b) for a, b in ("{}", "()", "[]"))


def top_level_escape(lines: list[str], dedent: int) -> str | None:
    """A `break`/`continue` statement at the segment's own nesting depth 0."""
    depth = 0
    for line in lines:
        code = strip_code(line)
        stripped = code.strip()
        if depth == 0 and (stripped.startswith("break;") or stripped.startswith("continue;")):
            return line
        depth += code.count("{") - code.count("}")
    return None


def parse_plan(text: str):
    plan = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 3:
            raise SystemExit(f"bad plan line: {raw!r}")
        plan.append((int(parts[0]), parts[1], parts[2]))
    return plan


def render_route(cls: str, name: str, body: list[str]) -> str:
    if name == "-":
        return "".join(body)
    return (f"std::optional<bool> {cls}::{name}() {{\n" + "".join(body) +
            "    return std::nullopt;\n}\n\n")


def run(src: pathlib.Path, first: int, last: int, cls: str, plan, dedent: int,
        replace: str | None, dry_run: bool, root: pathlib.Path) -> int:
    lines = src.read_text().splitlines(keepends=True)
    if not (1 <= first <= last <= len(lines)):
        raise SystemExit("body range outside the file")
    starts = [p[0] for p in plan]
    if not plan or starts[0] != first or any(b <= a for a, b in zip(starts, starts[1:])) \
            or starts[-1] > last:
        raise SystemExit("plan does not tile the body: first entry must be the body's first line, "
                         "entries strictly increasing and inside the body")
    pad = " " * dedent
    for number in range(first, last + 1):
        line = lines[number - 1]
        if line.strip() and not line.startswith(pad):
            raise SystemExit(f"line {number} is indented less than {dedent}: {line!r}")
    segments = []
    for index, (start, name, out) in enumerate(plan):
        end = starts[index + 1] - 1 if index + 1 < len(plan) else last
        original = lines[start - 1:end]
        body = [ln[dedent:] if ln.strip() else ln for ln in original]
        if [(pad + ln) if ln.strip() else ln for ln in body] != original:
            raise SystemExit(f"segment {name} at {start}: dedent is not reversible")
        if name != "-":
            if not balanced("".join(original)):
                raise SystemExit(f"segment {name} at {start}-{end}: unbalanced brackets")
            escape = top_level_escape(original, dedent)
            if escape:
                raise SystemExit(f"segment {name} at {start}-{end}: top-level escape {escape!r}")
        segments.append((start, end, name, out, body))
    if sum(e - s + 1 for s, e, *_ in segments) != last - first + 1:
        raise SystemExit("segments do not cover the body exactly")
    if dry_run:
        for s, e, name, out, _ in segments:
            print(f"{s:6d}-{e:<6d} {e - s + 1:5d} {name:40s} {out}")
        return 0
    by_file: dict[str, list[str]] = {}
    for _, _, name, out, body in segments:
        by_file.setdefault(out, []).append(render_route(cls, name, body))
    for out, chunks in by_file.items():
        path = root / out
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write("".join(chunks))
    for s, _, name, out, body in segments:
        if "".join(body) not in (root / out).read_text():
            raise SystemExit(f"read-back failed for {name} in {out}")
    if replace is not None:
        new = lines[:first - 1] + [" " * (dedent + 4) + replace + "\n"] + lines[last:]
        src.write_text("".join(new))
    for _, _, name, _, _ in segments:
        if name != "-":
            print(f"    std::optional<bool> {name}();")
    for _, _, name, _, _ in segments:
        if name != "-":
            print(f"    if (auto routed = {name}()) return *routed;")
    print(f"[ok] {len(segments)} segments, {last - first + 1} lines moved", file=sys.stderr)
    return 0


SELFTEST_SOURCE = """\
bool f(int x) {
    switch (x) {
        case 1: {
            // first shape
            if (x == 1) {
                if (g("}")) return true;
            }
            if (x == 2) { return false; }
            int y = 3;
            return y == 3;
        }
        default: return false;
    }
}
"""


def selftest() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        src = root / "f.cpp"
        src.write_text(SELFTEST_SOURCE)
        plan = [(4, "route_one", "out.cpp"), (8, "route_two", "out.cpp"), (9, "-", "tail.cpp")]
        run(src, 4, 10, "C", plan, 8, "return lower(x);", False, root)
        out = (root / "out.cpp").read_text()
        assert "std::optional<bool> C::route_one() {\n    // first shape\n    if (x == 1) {" in out
        assert out.count("return std::nullopt;") == 2
        assert (root / "tail.cpp").read_text() == "    int y = 3;\n    return y == 3;\n"
        assert "            return lower(x);\n        }\n" in src.read_text()
        # Each check must be able to fail.
        src.write_text(SELFTEST_SOURCE)
        for bad, why in (([(5, "a", "o.cpp")], "tile"),
                         ([(4, "a", "o.cpp"), (6, "b", "o.cpp")], "unbalanced"),
                         ([(4, "a", "o.cpp"), (4, "b", "o.cpp")], "tile")):
            try:
                run(src, 4, 10, "C", bad, 8, None, True, root)
            except SystemExit as e:
                assert why in str(e), (why, str(e))
            else:
                raise AssertionError(f"plan {bad} was accepted")
        src.write_text(SELFTEST_SOURCE.replace("int y = 3;", "break;"))
        try:
            run(src, 4, 10, "C", [(4, "a", "o.cpp")], 8, None, True, root)
        except SystemExit as e:
            assert "escape" in str(e)
        else:
            raise AssertionError("top-level break accepted")
        src.write_text(SELFTEST_SOURCE)
        try:
            run(src, 4, 10, "C", [(4, "a", "o.cpp")], 16, None, True, root)
        except SystemExit as e:
            assert "indented less" in str(e)
        else:
            raise AssertionError("over-dedent accepted")
    print("[ok] extract_case_routes selftest")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--file")
    ap.add_argument("--body", help="FIRST:LAST lines of the case body (inclusive)")
    ap.add_argument("--class", dest="cls")
    ap.add_argument("--plan")
    ap.add_argument("--dedent", type=int, default=8)
    ap.add_argument("--replace")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--root", default=".", help="directory output paths are relative to")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not (args.file and args.body and args.cls and args.plan):
        ap.error("--file, --body, --class and --plan are required")
    first, last = (int(v) for v in args.body.split(":"))
    plan = parse_plan(pathlib.Path(args.plan).read_text())
    return run(pathlib.Path(args.file), first, last, args.cls, plan, args.dedent, args.replace,
               args.dry_run, pathlib.Path(args.root))


if __name__ == "__main__":
    sys.exit(main())
