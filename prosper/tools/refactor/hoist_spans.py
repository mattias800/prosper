#!/usr/bin/env python3
"""hoist_spans.py -- move exact line spans of ONE file to an earlier anchor in the SAME file,
and prove the move changed nothing but position.

This is the primitive the toolbox was missing for shrinking a giant function. `split_file.py`
moves regions BETWEEN files and `extract_function.py` drives clangd's ExtractFunction; neither
can do the one transformation a function full of local `struct`s needs, which is to lift those
declarations to namespace scope so that anything referring to them can then be extracted at all.
A local class is already closed over its enclosing function (C++ forbids it naming an automatic
variable), so hoisting one is a pure relocation -- but only if the text really is relocated
rather than retyped, which is what this tool establishes.

WHAT IT PROVES, and the proof is an INVERSE rather than a checksum: the output is read back from
disk, the decoration lines this tool inserted are removed from the positions it recorded, each
moved span is cut from the destination and re-inserted at its original index, and the result must
equal the original file byte for byte. A byte-identical round trip cannot hold if a line was
dropped, duplicated, reordered within a span, or silently reindented.

WHAT IT DOES NOT PROVE -- the same caveat `split_file.py` carries, and for the same reason:
byte accounting says no LINE was lost, never that the result COMPILES. A hoisted type that names
a function-local alias, or whose name collides at namespace scope, is a clean round trip and a
build error. Only the compiler establishes that, so build after every invocation.

Usage:
  hoist_spans.py --file F --anchor N --spans 100-120,300-315 [--header-file H] [--footer-file G]
  hoist_spans.py --selftest
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path


def parse_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for piece in text.split(','):
        piece = piece.strip()
        if not piece:
            continue
        lo, _, hi = piece.partition('-')
        spans.append((int(lo), int(hi or lo)))
    return spans


def dedent_line(line: str, n: int) -> str:
    """Strip exactly `n` leading spaces. A blank line keeps being blank; anything else that is
    not indented that far is an error, because silently leaving it would make the inverse
    ambiguous and the proof below meaningless."""
    if n == 0 or not line.strip():
        return line
    if not line.startswith(' ' * n):
        raise SystemExit(f"dedent {n}: line is not indented that far: {line!r}")
    return line[n:]


def plan(lines: list[str], anchor: int, spans: list[tuple[int, int]],
         header: list[str], footer: list[str], dedent: int = 0) -> list[str]:
    """Return the rewritten lines. `lines` is 0-based; anchor/spans are 1-based inclusive."""
    n = len(lines)
    spans = sorted(spans)
    prev_end = 0
    for lo, hi in spans:
        if lo < 1 or hi > n or lo > hi:
            raise SystemExit(f"span {lo}-{hi} out of range 1-{n}")
        if lo <= prev_end:
            raise SystemExit(f"span {lo}-{hi} overlaps the previous one")
        prev_end = hi
    if anchor < 1 or anchor > n:
        raise SystemExit(f"anchor {anchor} out of range 1-{n}")
    if anchor > spans[0][0]:
        raise SystemExit(f"anchor {anchor} must be at or before the first span line {spans[0][0]}")

    moved: list[str] = []
    for lo, hi in spans:
        moved.extend(dedent_line(l, dedent) for l in lines[lo - 1:hi])

    keep_before = lines[:anchor - 1]
    rest: list[str] = []
    cut = {i for lo, hi in spans for i in range(lo, hi + 1)}
    for i in range(anchor, n + 1):
        if i not in cut:
            rest.append(lines[i - 1])
    return keep_before + header + moved + footer + rest


def verify(original: list[str], produced: list[str], anchor: int,
           spans: list[tuple[int, int]], header: list[str], footer: list[str],
           dedent: int = 0) -> None:
    """Invert the move on `produced` and require the original back, byte for byte."""
    spans = sorted(spans)
    head_at = anchor - 1
    if produced[head_at:head_at + len(header)] != header:
        raise SystemExit("verify: header decoration is not where it was written")
    cursor = head_at + len(header)
    cut_spans: list[list[str]] = []
    pad = ' ' * dedent
    for lo, hi in spans:
        size = hi - lo + 1
        cut_spans.append([(pad + l if l.strip() else l)
                          for l in produced[cursor:cursor + size]])
        cursor += size
    if produced[cursor:cursor + len(footer)] != footer:
        raise SystemExit("verify: footer decoration is not where it was written")
    cursor += len(footer)
    remainder = produced[cursor:]

    rebuilt = list(produced[:head_at])
    tail_iter = iter(remainder)
    i = anchor
    n = len(original)
    cut = {}
    for (lo, hi), body in zip(spans, cut_spans):
        cut[lo] = (hi, body)
    while i <= n:
        if i in cut:
            hi, body = cut[i]
            rebuilt.extend(body)
            i = hi + 1
            continue
        try:
            rebuilt.append(next(tail_iter))
        except StopIteration:
            raise SystemExit("verify: ran out of remainder lines rebuilding the original")
        i += 1
    if next(tail_iter, None) is not None:
        raise SystemExit("verify: remainder had lines left over")
    if rebuilt != original:
        for k, (a, b) in enumerate(zip(rebuilt, original)):
            if a != b:
                raise SystemExit(
                    f"verify: round trip differs first at line {k + 1}\n"
                    f"  rebuilt : {a!r}\n  original: {b!r}")
        raise SystemExit(
            f"verify: round trip length differs ({len(rebuilt)} vs {len(original)})")


def selftest() -> int:
    original = [f"l{i}" for i in range(1, 21)]
    header = ["// == hoisted =="]
    footer = ["// == end =="]
    spans = [(5, 7), (12, 13)]
    out = plan(original, 3, spans, header, footer)
    expect = (["l1", "l2"] + header + ["l5", "l6", "l7", "l12", "l13"] + footer +
              ["l3", "l4", "l8", "l9", "l10", "l11"] +
              [f"l{i}" for i in range(14, 21)])
    assert out == expect, out
    verify(original, out, 3, spans, header, footer)

    # A corrupted output must be REJECTED -- a verifier that cannot fail is not a verifier.
    for mutation, label in (
        (lambda o: o[:4] + o[5:], "a dropped line"),
        (lambda o: o[:4] + ["x"] + o[5:], "a rewritten line"),
        (lambda o: o[:3] + [o[4], o[3]] + o[5:], "a reordered pair inside a span"),
    ):
        bad = mutation(list(out))
        try:
            verify(original, bad, 3, spans, header, footer)
        except SystemExit:
            pass
        else:
            print(f"selftest FAILED: {label} was accepted")
            return 1

    # No decoration, single span, anchor equal to the span start (a no-op move).
    out2 = plan(original, 5, [(5, 7)], [], [])
    assert out2 == original, out2
    verify(original, out2, 5, [(5, 7)], [], [])

    # Dedent: the moved lines lose exactly N leading spaces and the inverse puts them back.
    indented = ["a", "        x", "        y", "", "b"]
    out3 = plan(indented, 1, [(2, 4)], [], [], dedent=4)
    assert out3 == ["    x", "    y", "", "a", "b"], out3
    verify(indented, out3, 1, [(2, 4)], [], [], dedent=4)
    try:
        plan(["a", "  x"], 1, [(2, 2)], [], [], dedent=4)
    except SystemExit:
        pass
    else:
        print("selftest FAILED: dedent past a line's indentation was accepted")
        return 1

    # A real file round trip, so the on-disk path is exercised rather than only the lists.
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "f.txt"
        p.write_text("\n".join(original) + "\n")
        src = p.read_text().split("\n")
        if src and src[-1] == "":
            src.pop()
        produced = plan(src, 3, spans, header, footer)
        p.write_text("\n".join(produced) + "\n")
        back = p.read_text().split("\n")
        if back and back[-1] == "":
            back.pop()
        verify(src, back, 3, spans, header, footer)

    print("selftest OK (9 cases: move, inverse, 3 rejected mutations, no-op, dedent + its\n         rejected over-dedent, on-disk round trip)")
    return 0


def read_lines(path: Path) -> tuple[list[str], bool]:
    text = path.read_text()
    trailing_newline = text.endswith("\n")
    lines = text.split("\n")
    if trailing_newline:
        lines.pop()
    return lines, trailing_newline


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--file')
    ap.add_argument('--anchor', type=int,
                    help='insert the moved spans BEFORE this 1-based line of the original')
    ap.add_argument('--spans', help='comma-separated 1-based inclusive ranges, e.g. 100-120,300-315')
    ap.add_argument('--header-file', help='file whose lines are inserted before the moved spans')
    ap.add_argument('--footer-file', help='file whose lines are inserted after the moved spans')
    ap.add_argument('--dedent', type=int, default=0,
                    help='strip exactly N leading spaces from every moved line (the inverse check '
                         'puts them back, so the round trip still proves nothing else changed)')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if not (args.file and args.anchor and args.spans):
        ap.error("--file, --anchor and --spans are required without --selftest")

    path = Path(args.file)
    original, trailing_newline = read_lines(path)
    spans = parse_spans(args.spans)
    header: list[str] = []
    footer: list[str] = []
    if args.header_file:
        header, _ = read_lines(Path(args.header_file))
    if args.footer_file:
        footer, _ = read_lines(Path(args.footer_file))

    produced = plan(original, args.anchor, spans, header, footer, args.dedent)
    moved_lines = sum(hi - lo + 1 for lo, hi in spans)
    if args.dry_run:
        verify(original, produced, args.anchor, spans, header, footer, args.dedent)
        print(f"dry-run OK: {len(spans)} span(s), {moved_lines} line(s) would move to line "
              f"{args.anchor}; inverse round trip reproduces the original")
        return 0

    path.write_text("\n".join(produced) + ("\n" if trailing_newline else ""))
    back, _ = read_lines(path)
    verify(original, back, args.anchor, spans, header, footer, args.dedent)
    print(f"moved {len(spans)} span(s), {moved_lines} line(s) to line {args.anchor}; "
          f"re-read from disk and the inverse round trip reproduces the original")
    return 0


if __name__ == '__main__':
    sys.exit(main())
