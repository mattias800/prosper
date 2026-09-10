#!/usr/bin/env python3
"""survey_sizes.py -- rank the tree's large files by WHICH refactoring tool can act on each.

`map_symbols.py` answers "how is this one file shaped". That is the right question once a file has
been chosen, and the wrong one for choosing: it needs an invocation, a compile command and a reading
per file, and this tree has ninety candidates. Picking targets by `wc -l` is worse -- it ranks by
symptom. A 12,000-line file that is two hundred functions is an afternoon with `split_file.py`; a
12,000-line file that is ONE function is not, and no amount of splitting will make it one.

So this reports, for every file over a threshold, which of four states it is in:

  SPLIT     no single region dominates -> `split_file.py` applies today.
  EXTRACT   one region dominates, but it is not mostly lambdas -> clangd's ExtractFunction can act
            on it (see README's Ruled out: clangd does the capture analysis correctly).
  HAND      one region dominates AND it is mostly lambda bodies -> clangd REFUSES to extract from a
            lambda ("Don't extract from lambdas", ExtractFunction.cpp), so no tool in this repo or
            in clang can shrink it. Hand restructuring, or leave it alone.
  UNPARSED  a large part of the file is behind an inactive `#if` -> every AST-driven verdict about
            the file is answering about the other part only, so no AST-driven verdict is given.

The lambda share is the whole point of the third column. It is the datum that decides between "a
tool can do this" and "a tool cannot", and it was previously measured by hand for a single function.

TWO THINGS THIS TOOL MUST NOT DO, both of which it did do, and both of which failed SILENTLY --
producing a confident verdict about a different file rather than an error. They are the reason the
measurements below are taken the way they are rather than the obvious way:

  * MEASURE THE LAMBDA SHARE OVER THE WRONG CURSOR. Nearly every file here is one
    `namespace prosper { ... }`, so a file's only top-level cursor is the namespace -- and a
    namespace answers `is_definition()` with True. Walking top-level cursors to find the dominant
    region's cursor therefore always found the NAMESPACE, and unioned every lambda in the whole file
    into the dominant function's share. `testdata/dominant_no_lambda.cpp` is the hand-built instance:
    a dominant function containing no lambda at all, beside a small all-lambda sibling, which the
    old code reported at a 0.63 lambda share -- HAND, "no tool can act", for a function clangd
    extracts from happily. Descend to the region's OWN cursor (`region_cursor`).

  * INFER UNPARSED LINES FROM REGION KINDS. `map_symbols` tiles a file by cursor, so an inactive
    `#if` arm is not a region of its own: it is folded into whichever region follows it and wears
    that region's kind. Keying on TEXT/TRAILER therefore sees an arm only when it runs to the END of
    the file. `src/host/memory/guest_write_watch.cpp` is the tree's own counter-instance -- a
    481-line `#ifdef _WIN32` arm at lines 71-551, 19% of the file, reported as 0 unparsed lines with
    a dominant "symbol" of `sched.h`. Ask the PREPROCESSOR instead (`clang_getSkippedRanges`), which
    records every skipped arm wherever it sits, and sum them: hle_kernel_mem.cpp is 52.1% across
    seven spans, not the 51% its single trailing span showed.

PARSE FAILURES ARE REPORTED, NEVER SCORED. A libclang parse driven from a g++ compile_commands.json
fails SOFTLY -- it returns a partial AST rather than an error -- so a file that did not really parse
comes back with few regions and looks merely uninteresting. That is this project's instrument-not-the
-subject shape exactly, and a survey is where it would do the most damage, because the file would be
silently ranked as "nothing to do here". Any file whose parse emits an error is printed as PARSE-FAIL
with the first diagnostic and given no verdict.

Usage:
  python3 prosper/tools/refactor/survey_sizes.py                       # the default roots
  ... --min-lines 800 --roots src frontends                            # widen or narrow
  ... --json survey.json                                               # machine-readable
  ... --selftest                                                       # prove the classifier
"""
from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
import importlib.util
import json
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_TESTDATA = _HERE / "testdata"


def _load_map_symbols():
    """Import map_symbols.py as a module so its parser is shared rather than re-implemented."""
    spec = importlib.util.spec_from_file_location("map_symbols", _HERE / "map_symbols.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["map_symbols"] = mod
    spec.loader.exec_module(mod)
    return mod


# libclang is resolved LAZILY, on the first parse. `--selftest`'s arithmetic cases exercise only
# classify(), which is pure, so they must run on a machine that has no libclang at all -- that is
# what lets the classifier be a registered ctest case rather than a thing somebody remembers to run.
# Importing map_symbols at module scope would defeat it: that module imports clang.cindex on load.
MS = None
ci = None
_SKIPPED_RANGES = None


def _need_clang() -> None:
    global MS, ci, _SKIPPED_RANGES
    if MS is not None:
        return
    MS = _load_map_symbols()
    import clang.cindex as _ci
    ci = _ci
    # Built here, not at module scope: the `ranges` member is an array of cindex's own SourceRange,
    # so the struct cannot be declared until clang.cindex has been imported, and a ctypes
    # `_fields_` cannot be reassigned after the class exists.
    class _CXSourceRangeList(ctypes.Structure):
        _fields_ = [("count", ctypes.c_uint), ("ranges", ctypes.POINTER(ci.SourceRange))]

    lib = ci.conf.lib
    # The python bindings do not wrap clang_getSkippedRanges (checked on clang 21's cindex.py), so
    # reach the C entry point directly. argtypes are deliberately left unset: cindex's TranslationUnit
    # and File carry `_as_parameter_`, which ctypes uses for an unprototyped call, and declaring
    # them as c_void_p instead would bypass that. Only restype needs stating.
    try:
        lib.clang_getSkippedRanges.restype = ctypes.POINTER(_CXSourceRangeList)
    except AttributeError:
        # Fail CLOSED. Reporting 0 unparsed lines when the question cannot be asked is precisely the
        # silent-under-report this tool exists to avoid, and it would land as a confident verdict.
        sys.exit("libclang here has no clang_getSkippedRanges; cannot measure unparsed spans "
                 "and will not guess. Upgrade libclang (present since clang 3.7).")
    _SKIPPED_RANGES = lib.clang_getSkippedRanges


# A region must reach this share of the file before the file counts as dominated by it. Below it,
# splitting has somewhere to cut; at or above it, splitting only renames the problem.
DOMINANT_SHARE = 0.40
# ... and a dominant region must be at least this much of a lambda nest before clangd's refusal is
# what actually blocks extraction, rather than an incidental detail.
LAMBDA_SHARE = 0.50
# Skipped-arm lines this share of a file mean the AST is blind to a substantial part of it, and every
# AST-driven verdict is then answering about the other part only. A CONVENTION, not a measurement:
# the tree has two files above it (guest_write_watch.cpp at 19.7%, hle_kernel_mem.cpp at 52.1%) and
# the next is far below, so the band is wide and nothing here pins where in it the line belongs.
UNPARSED_SHARE = 0.15
# A file needs this many movable regions before a split has anywhere to cut. Deliberately small,
# and the first draft got this wrong in a way the tree caught: it demanded two regions of >=200 lines
# each, which classified `hle_kernel_mem.cpp` -- 7,637 lines whose biggest region is 180 (2%) -- as
# "nothing to do". A file of a hundred small functions is the EASIEST split there is, not the
# hardest; the thing that blocks splitting is one region dominating, which DOMINANT_SHARE already
# measures. Region COUNT is what a seam needs, not region size.
SPLITTABLE_REGIONS = 4
# Still reported, because a file whose regions are all tiny may want grouping by role rather than a
# straight cut -- but it no longer gates the verdict.
SPLITTABLE_REGION_LINES = 200


def merge_spans(spans: list[tuple[int, int]]) -> int:
    """Number of DISTINCT lines covered by SPANS. Union, never sum -- see lambda_lines."""
    if not spans:
        return 0
    spans = sorted(spans)
    merged = [list(spans[0])]
    for s, e in spans[1:]:
        if s <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return sum(e - s + 1 for s, e in merged)


def skipped_lines(tu, path: pathlib.Path) -> int:   # requires _need_clang()
    """Lines of PATH the PREPROCESSOR skipped -- the inactive arms of #if/#ifdef/#ifndef.

    Summed over every arm, and clipped to this file: a TU pulls in hundreds of headers whose own
    include guards skip, and those say nothing about the file being surveyed.
    """
    f = ci.File.from_name(tu, str(path))
    if f is None:
        return 0
    rl = _SKIPPED_RANGES(tu, f)
    if not rl:
        return 0
    spans: list[tuple[int, int]] = []
    for i in range(rl.contents.count):
        r = rl.contents.ranges[i]
        s, e = r.start, r.end
        if s.file is None or pathlib.Path(s.file.name).resolve() != path.resolve():
            continue
        spans.append((s.line, e.line))
    return merge_spans(spans)


def lambda_lines(cursor) -> int:   # requires _need_clang()
    """Number of DISTINCT lines inside lambda bodies under CURSOR.

    Union, not sum: lambdas nest here (a registration lambda containing a per-draw lambda), and
    adding their extents would count the inner one twice and can exceed the function's own length.
    """
    spans: list[tuple[int, int]] = []

    def walk(c) -> None:
        if c.kind == ci.CursorKind.LAMBDA_EXPR:
            e = c.extent
            if e.start.file is not None:
                spans.append((e.start.line, e.end.line))
            return                      # inner lambdas are already inside this span
        for k in c.get_children():
            walk(k)

    walk(cursor)
    return merge_spans(spans)


def region_cursor(tu, path: pathlib.Path, region: dict, max_depth: int = 3):
    """The cursor that PRODUCED this region, found by descending exactly as regions_of tiles.

    Not a top-level scan. `regions_of` descends through namespaces, so a region's cursor is usually
    a namespace MEMBER -- and the file's only top-level cursor is the namespace itself, which
    answers is_definition() with True and would match every region in the file. See the module
    docstring; `testdata/dominant_no_lambda.cpp` reddens if this goes back to a top-level scan.
    """
    want_start, want_end = region["decl_line"], region["end"]
    found = None

    def walk(c, depth: int) -> None:
        nonlocal found
        for k in c.get_children():
            if found is not None:
                return
            if k.location.file is None:
                continue
            if pathlib.Path(k.location.file.name).resolve() != path.resolve():
                continue
            if k.kind == ci.CursorKind.NAMESPACE and depth < max_depth:
                walk(k, depth + 1)
                continue
            if k.extent.start.line == want_start and k.extent.end.line == want_end:
                found = k
                return

    walk(tu.cursor, 0)
    return found


def classify(total: int, biggest: int, lam: int, regions: int, unparsed: int = 0) -> str:
    """The verdict, kept pure so --selftest can pin it without parsing anything.

    REGIONS is the file's MOVABLE region count -- regions a split could actually relocate, so the
    namespace open/close that every output part gets a copy of are not counted. See
    SPLITTABLE_REGIONS. UNPARSED is lines the preprocessor skipped, and it is checked FIRST: a
    verdict computed from half a file is not a weaker verdict, it is a statement about a different
    file.
    """
    if total <= 0:
        return "OK"
    if unparsed / total >= UNPARSED_SHARE:
        return "UNPARSED"
    share = biggest / total
    if share >= DOMINANT_SHARE:
        return "HAND" if lam / biggest >= LAMBDA_SHARE else "EXTRACT"
    return "SPLIT" if regions >= SPLITTABLE_REGIONS else "OK"


def measure(path: pathlib.Path, flags: list[str], parse_target: pathlib.Path) -> dict:
    """Parse and score one file. Split out from survey_one so fixtures can be scored with fixed
    flags, without a compile_commands.json."""
    _need_clang()
    rel = str(path)
    total = len(path.read_text(errors="replace").splitlines())
    index = ci.Index.create()
    try:
        tu = index.parse(str(parse_target), args=flags + MS.builtin_includes(),
                         options=ci.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD)
    except Exception as exc:                      # noqa: BLE001 - report, never score
        return {"file": rel, "lines": total, "verdict": "PARSE-FAIL", "why": repr(exc)}

    errs = [d for d in tu.diagnostics if d.severity >= ci.Diagnostic.Error]
    if errs:
        return {"file": rel, "lines": total, "verdict": "PARSE-FAIL",
                "why": f"{len(errs)} parse error(s); first: {errs[0].spelling}"}

    unparsed = skipped_lines(tu, path)
    regions = MS.regions_of(tu, path, total)
    # Movable regions only. An `open`/`close` is replicated into every output part, so it is not a
    # place a split can cut and must not count toward the file having a seam.
    sized = [(r["end"] - r["start"] + 1, r) for r in regions if r.get("role") not in MS.REPLICATED]
    if unparsed / max(total, 1) >= UNPARSED_SHARE:
        return {"file": rel, "lines": total, "regions": len(sized),
                "unparsed_lines": unparsed, "unparsed_share": round(unparsed / total, 3),
                "biggest": "<%d line(s) behind an inactive #if>" % unparsed,
                "biggest_lines": unparsed, "biggest_share": round(unparsed / total, 3),
                "lambda_lines": 0, "lambda_share_of_biggest": 0.0,
                "verdict": "UNPARSED"}
    if not sized:
        return {"file": rel, "lines": total, "verdict": "OK", "regions": 0,
                "unparsed_lines": unparsed}
    sized.sort(key=lambda t: -t[0])
    big_len, big = sized[0]

    # Lambda share is only meaningful for the dominant region, and finding its cursor means walking
    # the TU again; skip it when the file is not dominated anyway.
    lam, cursor_found = 0, True
    if big_len / total >= DOMINANT_SHARE:
        c = region_cursor(tu, path, big)
        if c is None:
            cursor_found = False
        else:
            lam = lambda_lines(c)
    if not cursor_found:
        # Never fall through to lam=0, which would read as EXTRACT -- the optimistic verdict -- on
        # no evidence at all.
        return {"file": rel, "lines": total, "regions": len(sized),
                "biggest": big["name"], "biggest_lines": big_len,
                "biggest_share": round(big_len / total, 3),
                "unparsed_lines": unparsed, "verdict": "NO-CURSOR",
                "why": "dominant region %d-%d matched no cursor" % (big["decl_line"], big["end"])}

    return {
        "file": rel, "lines": total, "regions": len(sized),
        "biggest": big["name"], "biggest_lines": big_len,
        "biggest_share": round(big_len / total, 3),
        "lambda_lines": lam,
        "lambda_share_of_biggest": round(lam / big_len, 3) if big_len else 0.0,
        "regions_over_%d" % SPLITTABLE_REGION_LINES:
            sum(1 for n, _ in sized if n >= SPLITTABLE_REGION_LINES),
        "unparsed_lines": unparsed,
        "verdict": classify(total, big_len, lam, len(sized), unparsed),
    }


def survey_one(path: pathlib.Path, db: pathlib.Path) -> dict:
    _need_clang()
    try:
        flags, parse_target = MS.flags_for(path, db)
    except SystemExit as exc:
        return {"file": str(path),
                "lines": len(path.read_text(errors="replace").splitlines()),
                "verdict": "PARSE-FAIL", "why": str(exc)}
    return measure(path, flags, parse_target)


def _arithmetic_selftest() -> int:
    """Pin the classifier on constructed cases -- no parsing, so it cannot pass for a parse reason."""
    cases = [
        # (total, biggest, lambda, movable_regions, expected)
        (10000, 9000, 8900, 1, "HAND"),      # live_renderer shape: one function, nearly all lambda
        (10000, 9000,   10, 1, "EXTRACT"),   # one function, lambda-free -> clangd can act
        (10000, 1000,    0, 8, "SPLIT"),     # many mid-sized regions -> split_file.py
        (10000, 1000,    0, 96, "SPLIT"),   # huge, all small regions -- the EASIEST split there is.
                                            # An earlier classifier called this OK, which was wrong.
        (10000,  300,    0, 2, "OK"),        # large-ish but only two regions -- no real seam
        (10000, 4000, 1999, 3, "EXTRACT"),   # just BELOW the lambda boundary
        (10000, 4000, 2000, 3, "HAND"),      # exactly AT it -- `>=` puts the boundary in HAND
        (   0,    0,    0, 0, "OK"),         # empty file must not divide by zero
    ]
    unparsed_cases = [
        # (total, biggest, lambda, regions, unparsed, expected)
        (7637,  180, 0, 246, 3978, "UNPARSED"),   # hle_kernel_mem's REAL shape: 52% behind #ifs,
                                                  # across SEVEN spans. Scored SPLIT before the
                                                  # skipped arms were counted, from the 48% the
                                                  # parser could see.
        (2513,  200, 0,  60,  495, "UNPARSED"),   # guest_write_watch: 19.7%, and every line of it
                                                  # MID-FILE. Reported 0 unparsed while the verdict
                                                  # keyed on region kinds.
        (10000, 1000, 0,  96, 1499, "SPLIT"),     # just below 15% -- ordinary scaffolding
        (10000, 1000, 0,  96, 1500, "UNPARSED"),  # exactly at it
        (10000, 9000, 8900, 1, 1600, "UNPARSED"), # unparsed WINS over HAND: a verdict from half a
                                                  # file is about a different file
    ]
    bad = 0
    for total, big, lam, nreg, want in cases:
        got = classify(total, big, lam, nreg)
        ok = got == want
        print(f"  [{'ok' if ok else 'FAIL'}] classify({total},{big},{lam},{nreg}) = {got}"
              f"{'' if ok else f' (want {want})'}")
        bad += not ok
    # The boundary pair above is the point, and the first draft of this selftest got it wrong in a
    # useful way: it asserted EXTRACT for 2000/4000 while the docstring said 0.50 is HAND. The
    # classifier was right and the expectation was not. Pin the direction explicitly so the next
    # reader cannot re-introduce that disagreement between comment and behaviour.
    if classify(10000, 4000, 2000, 3) != "HAND":
        print("  [FAIL] a lambda share exactly AT the threshold must be HAND")
        bad += 1
    if classify(10000, 4000, 1999, 3) != "EXTRACT":
        print("  [FAIL] a lambda share just BELOW the threshold must stay EXTRACT")
        bad += 1
    for total, big, lam, nreg, unp, want in unparsed_cases:
        got = classify(total, big, lam, nreg, unp)
        ok = got == want
        print(f"  [{'ok' if ok else 'FAIL'}] classify({total},{big},{lam},{nreg},unparsed={unp})"
              f" = {got}{'' if ok else f' (want {want})'}")
        bad += not ok
    return bad


def _parse_selftest() -> int:
    """Score the two fixtures, whose answers are known by construction.

    These are the cases the arithmetic above CANNOT reach: both past defects were in how the inputs
    to classify() were measured, not in classify() itself, so every arithmetic case passed happily
    while the tool reported the opposite verdict for a real file. Each fixture is built so that the
    defective measurement produces a DIFFERENT verdict, not merely a different number.
    """
    bad = 0
    flags = ["-std=c++20", "-xc++"]

    a = _TESTDATA / "dominant_no_lambda.cpp"
    r = measure(a, flags, a)
    lam_share = r.get("lambda_share_of_biggest")
    ok = r["verdict"] == "EXTRACT" and lam_share == 0.0
    print(f"  [{'ok' if ok else 'FAIL'}] {a.name}: verdict={r['verdict']} "
          f"biggest={r.get('biggest')!r} lambda_share={lam_share} "
          f"(want EXTRACT / 0.0 -- measuring over the namespace gives HAND)")
    if not ok:
        print(f"        full row: {r}")
    bad += not ok

    b = _TESTDATA / "midfile_inactive_arm.cpp"
    r = measure(b, flags, b)
    share = r.get("unparsed_share", 0.0)
    ok = r["verdict"] == "UNPARSED" and share >= UNPARSED_SHARE
    print(f"  [{'ok' if ok else 'FAIL'}] {b.name}: verdict={r['verdict']} "
          f"unparsed={r.get('unparsed_lines')} share={share} "
          f"(want UNPARSED -- inferring from region kinds gives 0 unparsed)")
    if not ok:
        print(f"        full row: {r}")
    bad += not ok
    return bad


def selftest() -> int:
    bad = _arithmetic_selftest()
    try:
        import clang.cindex  # noqa: F401
        have_clang = True
    except ImportError:
        have_clang = False
    if have_clang:
        print("  -- parse fixtures (libclang present) --")
        bad += _parse_selftest()
    else:
        # Say so rather than reporting a quieter PASS. These two checks are the ones that cover how
        # the classifier's INPUTS are measured, and both shipped defects lived there.
        print("  -- parse fixtures SKIPPED: no libclang here, so the two measurement checks")
        print("     (dominant-cursor, mid-file inactive arm) did NOT run. Arithmetic only.")
    print("== PASS ==" if not bad else f"== FAIL: {bad} ==")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=["src", "frontends", "tools", "tests"])
    ap.add_argument("--min-lines", type=int, default=1500)
    ap.add_argument("--build", default="prosper/build-linux")
    ap.add_argument("--json", type=pathlib.Path)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    _need_clang()
    root = MS.repo_root()
    db = root / args.build / "compile_commands.json"
    if not db.exists():
        db = root / "prosper" / args.build.replace("prosper/", "") / "compile_commands.json"
    if not db.exists():
        sys.exit(f"no compile_commands.json under {args.build}; configure cmake first")

    cands: list[pathlib.Path] = []
    for r in args.roots:
        base = root / "prosper" / r
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.suffix in (".cpp", ".hpp", ".h", ".cc") and p.is_file():
                if _TESTDATA in p.parents:
                    continue                      # fixtures are deliberately misshapen
                if len(p.read_text(errors="replace").splitlines()) >= args.min_lines:
                    cands.append(p)
    cands.sort()
    print(f"[survey] {len(cands)} file(s) at or above {args.min_lines} lines; "
          f"parsing with {args.jobs} job(s)\n")

    rows: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for res in ex.map(lambda p: survey_one(p, db), cands):
            rows.append(res)

    order = {"SPLIT": 0, "EXTRACT": 1, "UNPARSED": 2, "HAND": 3,
             "NO-CURSOR": 4, "PARSE-FAIL": 5, "OK": 6}
    print(f"{'verdict':<11}{'lines':>7}{'big':>7}{'share':>7}{'lam%':>6}{'unp%':>6}"
          f"  file / dominant symbol")
    print("-" * 104)
    for r in sorted(rows, key=lambda r: (order.get(r["verdict"], 9), -r.get("lines", 0))):
        if r["verdict"] in ("PARSE-FAIL", "NO-CURSOR"):
            print(f"{r['verdict']:<11}{r['lines']:>7}{'':>7}{'':>7}{'':>6}{'':>6}  {r['file']}\n"
                  f"{'':>44}  {r['why'][:90]}")
            continue
        rel = r["file"].split("prosper/", 1)[-1]
        unp = r.get("unparsed_lines", 0) / max(r.get("lines", 1), 1)
        print(f"{r['verdict']:<11}{r['lines']:>7}{r.get('biggest_lines', 0):>7}"
              f"{r.get('biggest_share', 0):>7.0%}{r.get('lambda_share_of_biggest', 0):>6.0%}"
              f"{unp:>6.0%}  {rel}\n{'':>44}  {r.get('biggest', '')}")

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    print("\n[survey] " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2))
        print(f"[survey] wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
