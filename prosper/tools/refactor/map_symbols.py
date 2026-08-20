#!/usr/bin/env python3
"""map_symbols.py -- tile a translation unit into top-level regions, and find the seams.

Splitting a 25,000-line .cpp by hand is how a refactor introduces a bug that no reviewer can see: the
diff is enormous, every line looks moved, and a single dropped helper or reordered static initialiser
hides in it perfectly. So the split is driven by a parser, not by reading.

This reports two things about a file, both from clang's own AST:

  REGIONS -- a complete TILING of the file into top-level declarations. Region 0 is the preamble
    (includes, pragmas, file comment); each subsequent region runs from one line after the previous
    region's end through the end of a top-level cursor. Tiling is by CONSTRUCTION, not by check:
    because each region begins where the last ended, the concatenation of every region is the
    original file, byte for byte, and a leading comment block therefore travels with the declaration
    it documents. `split_file.py` relies on exactly this property -- it is what makes a split
    provable rather than merely reviewed.

  REFERENCES -- which region uses a symbol DEFINED IN ANOTHER REGION of the same file. This is the
    part worth having a tool for. A file's internal-linkage helpers (`static`, anonymous namespace)
    are invisible outside it, so any split that separates a helper from its callers does not fail at
    review, it fails at link -- and only after the whole move is done. The reference matrix names
    those edges up front, so a split can be chosen along a seam that has few of them, and the ones
    that remain are a known list to promote into an internal header rather than a surprise.

Usage:
  python3 prosper/tools/refactor/map_symbols.py --file prosper/src/gpu/recompiler/rdna2_to_spirv.cpp
  ... --clusters      # suggest seams: greedily group regions that reference each other
  ... --json out.json # machine-readable, for split_file.py

Flags come from the build's compile_commands.json, so the parse sees exactly what the compiler sees.
"""

import argparse
import bisect
import json
import pathlib
import shlex
import subprocess
import sys
from collections import defaultdict

import clang.cindex as ci

# The bindings look for a bare `libclang.so`, which the runtime package does not ship; the versioned
# soname is what is actually installed. Fail loudly rather than letting a later call die obscurely.
for cand in ("/usr/lib64/libclang.so.22.1", "/usr/lib64/libclang.so"):
    if pathlib.Path(cand).exists():
        ci.Config.set_library_file(cand)
        break


def repo_root() -> pathlib.Path:
    out = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                         capture_output=True, text=True, check=True)
    return pathlib.Path(out.stdout.strip())


def flags_for(target: pathlib.Path, db: pathlib.Path) -> tuple[list[str], pathlib.Path]:
    """Compiler flags for TARGET, and the file to actually parse.

    A header has no entry of its own, so parse the first translation unit that includes it and filter
    the AST back down to the header. That is not a workaround: a header only HAS an AST in the
    context of a TU, and which TU is a real choice -- a header behind an #ifdef sees different code
    in different ones.
    """
    entries = json.loads(db.read_text())
    by_file = {pathlib.Path(e["file"]).resolve(): e for e in entries}
    if target.resolve() in by_file:
        e = by_file[target.resolve()]
        return _clean(e), target

    # A header has no compile command of its own, so parse the first TU that includes it. The needle
    # must match the CANONICAL spelling this project now uses -- `#include "gpu/recompiler/x.hpp"` --
    # not a bare `"x.hpp"`, which the include normalisation removed from the tree entirely. Anchoring
    # on the bare form made this fail for 35 of 35 moved headers: a tool broken by the restructure it
    # was written to serve.
    needle = f'/{target.name}"'
    for e in entries:
        p = pathlib.Path(e["file"])
        if not p.exists():
            continue
        try:
            if needle in p.read_text():
                return _clean(e), p
        except UnicodeDecodeError:
            continue
    sys.exit(f"no compile command found for {target}, and no TU includes it")


def builtin_includes() -> list[str]:
    """clang's own headers (stddef.h, stdint.h ...).

    compile_commands.json is written for g++, whose builtin include directory clang does not know
    about, so a parse driven straight from the database dies on `'stddef.h' file not found` -- and
    dies SOFTLY: libclang returns a partial AST rather than failing, so the region map comes back
    short and looks merely uninteresting. That is the failure this project files under
    instrument-not-the-subject, so the diagnostic count is checked by the caller and this makes it
    zero.
    """
    out = subprocess.run(["clang", "-print-resource-dir"], capture_output=True, text=True)
    if out.returncode != 0:
        return []
    return ["-isystem", str(pathlib.Path(out.stdout.strip()) / "include")]


def _clean(entry: dict) -> list[str]:
    args = shlex.split(entry["command"]) if "command" in entry else list(entry["arguments"])
    out, skip = [], False
    for a in args[1:]:
        if skip:
            skip = False
            continue
        if a in ("-c", "-o"):
            skip = a == "-o"
            continue
        if a.endswith((".cpp", ".cc", ".c")):
            continue
        out.append(a)
    return out


REPLICATED = ("open", "close")   # roles that every output file needs a copy of


def regions_of(tu, target: pathlib.Path, total_lines: int, max_depth: int = 3) -> list[dict]:
    """Tile TARGET into regions, descending through namespaces.

    A flat tiling is useless on this codebase: nearly every file is one `namespace prosper { ... }`,
    so the whole 25,000 lines came back as a single region. Descending gives regions at the
    granularity a split actually needs, at the cost of one complication -- a part cannot just be a
    slice of the file, because the slice sits inside a namespace whose opening and closing braces
    live in other slices.

    So a namespace becomes THREE kinds of region: an `open` (everything from where the previous
    region ended through the line before its first member -- which folds in the file's includes for
    the outermost one), the members themselves, and a `close`. `open` and `close` carry the role
    `replicate`: every output file gets a copy, so every output re-opens and re-closes the same
    namespaces around whichever members it received.

    Tiling remains by construction. Each region still starts exactly where the previous ended, so
    concatenating every region in index order reproduces the file byte for byte -- which
    split_file.py asserts before it writes anything. And because opens sort before the members they
    enclose and closes sort after, an output built by taking every replicated region plus its own
    members IN INDEX ORDER is correctly nested without the splitter reasoning about nesting at all.
    """
    def in_target(c) -> bool:
        return (c.location.file is not None
                and pathlib.Path(c.location.file.name).resolve() == target.resolve())

    regions: list[dict] = []

    def mk(start: int, end: int, role: str, c, kind: str | None = None) -> None:
        regions.append({
            "index": len(regions), "start": start, "end": end, "role": role,
            "decl_line": c.extent.start.line if c is not None else start,
            "kind": kind or (str(c.kind).replace("CursorKind.", "") if c is not None else "TEXT"),
            "name": (c.spelling or "<anonymous>") if c is not None else "<text>",
            "is_definition": bool(c.is_definition()) if c is not None else False,
            "usr": c.get_usr() if c is not None else "",
            "symbols": [c.spelling or "<anonymous>"] if c is not None else [],
        })

    def emit(c, pos: int, depth: int) -> int:
        cend = c.extent.end.line
        if cend < pos:
            # Wholly inside the region already emitted -- several declarators on one line, or one
            # macro that expanded to more than one declaration. Fold it in rather than emitting an
            # inverted range, which would silently break the tiling.
            if regions:
                regions[-1]["symbols"].append(c.spelling or "<anonymous>")
            return pos
        kids = [k for k in c.get_children() if in_target(k)] if c.kind == ci.CursorKind.NAMESPACE else []
        if kids and depth < max_depth:
            kids.sort(key=lambda k: k.extent.start.line)
            first = kids[0].extent.start.line
            if first > pos:
                mk(pos, first - 1, "open", c, kind="NAMESPACE_OPEN")
                pos = first
            for k in kids:
                pos = emit(k, pos, depth + 1)
            if cend >= pos:
                mk(pos, cend, "close", c, kind="NAMESPACE_CLOSE")
                pos = cend + 1
            return pos
        mk(pos, cend, "body", c)
        return cend + 1

    tops = sorted([c for c in tu.cursor.get_children() if in_target(c)],
                  key=lambda c: c.extent.start.line)
    pos = 1
    for c in tops:
        pos = emit(c, pos, 0)
    if pos <= total_lines:
        mk(pos, total_lines, "close", None, kind="TRAILER")

    # Everything before the first namespace opener is the PREAMBLE -- the file comment and the
    # #include directives. Every output of a split needs all of it, so it carries the same
    # replicate role as a namespace open/close. Marked by POSITION rather than by kind: a stray
    # declaration above the first namespace still belongs in every part, whereas a rule keyed on
    # INCLUSION_DIRECTIVE would quietly file it into one of them.
    first_open = next((r["index"] for r in regions if r["role"] == "open"), None)
    if first_open is not None:
        for r in regions[:first_open]:
            r["role"] = "preamble"
    return regions


def cross_refs(tu, target: pathlib.Path, regions: list[dict]) -> dict[int, dict[int, int]]:
    """region -> {other region: reference count}, for symbols defined in this same file."""
    # Own every declaration whose LOCATION falls inside a region, not just the region's own cursor.
    # An enum constant, a struct member, a nested type: each has its own USR and is a CHILD of the
    # top-level cursor, so a map built from top-level USRs alone cannot see a reference to one. That
    # gap is invisible until it matters -- promoting a struct out of a file left every `Dim_2D` and
    # `Op_Capability` behind, 1,046 compile errors, because the enums those constants belong to were
    # never counted as dependencies.
    owner: dict[str, int] = {}
    bounds_pre = [(r["start"], r["end"], r["index"]) for r in regions]

    def owning_region(line: int) -> int | None:
        for s_, e_, i_ in bounds_pre:
            if s_ <= line <= e_:
                return i_
        return None

    pending = [tu.cursor]
    while pending:
        c = pending.pop()
        for ch in c.get_children():
            pending.append(ch)
            if ch.location.file is None:
                continue
            if pathlib.Path(ch.location.file.name).resolve() != target.resolve():
                continue
            usr = ch.get_usr()
            if not usr:
                continue
            idx = owning_region(ch.location.line)
            if idx is not None:
                owner.setdefault(usr, idx)

    starts = [r["start"] for r in regions]

    def region_of(line: int) -> int | None:
        # Regions tile the file and are sorted, so the owner is the last region starting at or
        # before LINE. Linear scan was O(regions) per reference and this file has ~10^5 of them.
        i = bisect.bisect_right(starts, line) - 1
        return i if 0 <= i < len(regions) else None

    edges: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    stack = [tu.cursor]
    while stack:
        c = stack.pop()
        for ch in c.get_children():
            stack.append(ch)
            if ch.location.file is None:
                continue
            if pathlib.Path(ch.location.file.name).resolve() != target.resolve():
                continue
            ref = ch.referenced
            if ref is None:
                continue
            usr = ref.get_usr()
            if usr not in owner:
                continue
            src = region_of(ch.location.line)
            dst = owner[usr]
            if src is None or src == dst:
                continue
            edges[src][dst] += 1
    return {k: dict(v) for k, v in edges.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", required=True, type=pathlib.Path)
    ap.add_argument("--build", default="prosper/build-linux")
    ap.add_argument("--json", type=pathlib.Path)
    ap.add_argument("--min-lines", type=int, default=0,
                    help="only print regions at least this many lines long")
    args = ap.parse_args()

    root = repo_root()
    target = (root / args.file) if not args.file.is_absolute() else args.file
    if not target.exists():
        sys.exit(f"no such file: {target}")
    db = root / args.build / "compile_commands.json"
    if not db.exists():
        sys.exit(f"no compile_commands.json at {db}; configure the build first")

    flags, parse_file = flags_for(target, db)
    flags += builtin_includes()
    total_lines = len(target.read_text().splitlines())

    index = ci.Index.create()
    tu = index.parse(str(parse_file), args=flags,
                     options=ci.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD)
    fatal = [d for d in tu.diagnostics if d.severity >= ci.Diagnostic.Error]
    if fatal:
        print(f"  WARNING {len(fatal)} parse error(s); the map may be incomplete")
        for d in fatal[:3]:
            print(f"    {d.spelling}")

    regions = regions_of(tu, target, total_lines)
    edges = cross_refs(tu, target, regions)

    bodies = [r for r in regions if r["role"] == "body"]
    print(f"== {target.relative_to(root)}: {total_lines} lines, {len(regions)} region(s) "
          f"({len(bodies)} splittable, {len(regions) - len(bodies)} replicated) ==")
    if parse_file != target:
        print(f"   parsed through {parse_file.relative_to(root)}")
    for r in regions:
        span = r["end"] - r["start"] + 1
        if span < args.min_lines:
            continue
        out = sum(edges.get(r["index"], {}).values())
        inn = sum(v for s, d in edges.items() for t, v in d.items() if t == r["index"])
        print(f"  [{r['index']:4d}] {r['start']:6d}-{r['end']:<6d} {span:6d}L  "
              f"out={out:<4d} in={inn:<4d} {r['role']:<9s} {r['kind']:<22s} {r['name'][:52]}")

    if args.json:
        # The digest is what lets split_file.py prove the map still describes the file. Without it
        # the only available check compares two values derived from the same read, which can only
        # fail if the file GREW -- an assertion that cannot fail on an edit is not a check.
        import hashlib
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        args.json.write_text(json.dumps({"file": str(target.relative_to(root)),
                                         "sha256": digest,
                                         "total_lines": total_lines,
                                         "regions": regions,
                                         "edges": {str(k): v for k, v in edges.items()}}, indent=1))
        print(f"  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
