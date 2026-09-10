#!/usr/bin/env python3
"""survey_sizes.py -- rank the tree's large files by WHICH refactoring tool can act on each.

`map_symbols.py` answers "how is this one file shaped". That is the right question once a file has
been chosen, and the wrong one for choosing: it needs an invocation, a compile command and a reading
per file, and this tree has ninety candidates. Picking targets by `wc -l` is worse -- it ranks by
symptom. A 12,000-line file that is two hundred functions is an afternoon with `split_file.py`; a
12,000-line file that is ONE function is not, and no amount of splitting will make it one.

So this reports, for every file over a threshold, which of three states it is in:

  SPLIT    no single top-level region dominates -> `split_file.py` applies today.
  EXTRACT  one region dominates, but it is not mostly lambdas -> clangd's ExtractFunction can act on
           it (see README's Ruled out: clangd does the capture analysis correctly).
  HAND     one region dominates AND it is mostly lambda bodies -> clangd REFUSES to extract from a
           lambda ("Don't extract from lambdas", ExtractFunction.cpp), so no tool in this repo or in
           clang can shrink it. Hand restructuring, or leave it alone.

The lambda share is the whole point of the third column. It is the datum that decides between "a tool
can do this" and "a tool cannot", and it was previously measured by hand for a single function.

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
import importlib.util
import json
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent


def _load_map_symbols():
    """Import map_symbols.py as a module so its parser is shared rather than re-implemented."""
    spec = importlib.util.spec_from_file_location("map_symbols", _HERE / "map_symbols.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["map_symbols"] = mod
    spec.loader.exec_module(mod)
    return mod


# libclang is resolved LAZILY, on the first parse. `--selftest` exercises only classify(), which is
# pure arithmetic, so it must run on a machine that has no libclang at all -- that is what lets the
# classifier be a registered ctest case rather than a thing somebody remembers to run. Importing
# map_symbols at module scope would defeat it: that module imports clang.cindex on load.
MS = None
ci = None


def _need_clang() -> None:
    global MS, ci
    if MS is not None:
        return
    MS = _load_map_symbols()
    import clang.cindex as _ci
    ci = _ci


# A region must reach this share of the file before the file counts as dominated by it. Below it,
# splitting has somewhere to cut; at or above it, splitting only renames the problem.
DOMINANT_SHARE = 0.40
# ... and a dominant region must be at least this much of a lambda nest before clangd's refusal is
# what actually blocks extraction, rather than an incidental detail.
LAMBDA_SHARE = 0.50
# A text region this large means the parser never saw that code -- almost always the inactive side of
# an `#if defined(_WIN32)` platform arm. Below this it is ordinary scaffolding (includes, a namespace
# close); at or above it, the AST is blind to a substantial part of the file and every AST-driven
# verdict about that file is answering about the other part only.
UNPARSED_SHARE = 0.15
# A file needs this many top-level regions before a split has anywhere to cut. Deliberately small,
# and the first draft got this wrong in a way the tree caught: it demanded two regions of >=200 lines
# each, which classified `hle_kernel_mem.cpp` -- 7,637 lines whose biggest region is 180 (2%) -- as
# "nothing to do". A file of a hundred small functions is the EASIEST split there is, not the
# hardest; the thing that blocks splitting is one region dominating, which DOMINANT_SHARE already
# measures. Region COUNT is what a seam needs, not region size.
SPLITTABLE_REGIONS = 4
# Still reported, because a file whose regions are all tiny may want grouping by role rather than a
# straight cut -- but it no longer gates the verdict.
SPLITTABLE_REGION_LINES = 200


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
    if not spans:
        return 0
    spans.sort()
    merged = [list(spans[0])]
    for s, e in spans[1:]:
        if s <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return sum(e - s + 1 for s, e in merged)


def classify(total: int, biggest: int, lam: int, regions: int, unparsed: int = 0) -> str:
    """The verdict, kept pure so --selftest can pin it without parsing anything.

    REGIONS is the file's top-level region count, not a count of large ones -- see SPLITTABLE_REGIONS.
    UNPARSED is lines the AST never saw, and it is checked FIRST: a verdict computed from half a file
    is not a weaker verdict, it is a statement about a different file.
    """
    if total <= 0:
        return "OK"
    if unparsed / total >= UNPARSED_SHARE:
        return "UNPARSED"
    share = biggest / total
    if share >= DOMINANT_SHARE:
        return "HAND" if biggest > 0 and lam / biggest >= LAMBDA_SHARE else "EXTRACT"
    return "SPLIT" if regions >= SPLITTABLE_REGIONS else "OK"


def survey_one(path: pathlib.Path, db: pathlib.Path) -> dict:
    _need_clang()
    rel = str(path)
    total = len(path.read_text(errors="replace").splitlines())
    try:
        flags, parse_target = MS.flags_for(path, db)
    except SystemExit as exc:
        return {"file": rel, "lines": total, "verdict": "PARSE-FAIL", "why": str(exc)}
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

    regions = MS.regions_of(tu, path, total)
    # Regions the parser contributed no declaration to. `map_symbols` gives the outermost namespace's
    # trailing text the `close` role, so on a file whose second half sits behind an inactive `#if`
    # the ENTIRE unparsed arm arrives as one `close`/TRAILER region -- and this filter used to drop
    # it as scaffolding. On hle_kernel_mem.cpp that silently hid 3,909 lines, 51% of the file, and
    # scored what remained as though it were the whole thing.
    unparsed = sum(r["end"] - r["start"] + 1 for r in regions
                   if r.get("kind") in ("TEXT", "TRAILER")
                   and (r["end"] - r["start"] + 1) / max(total, 1) >= UNPARSED_SHARE)
    sized = [(r["end"] - r["start"] + 1, r) for r in regions if r.get("role") not in MS.REPLICATED]
    if not sized:
        return {"file": rel, "lines": total, "verdict": "OK", "regions": 0}
    if unparsed / max(total, 1) >= UNPARSED_SHARE:
        return {"file": rel, "lines": total, "regions": len(regions),
                "unparsed_lines": unparsed, "unparsed_share": round(unparsed / total, 3),
                "biggest": "<unparsed span>", "biggest_lines": unparsed,
                "biggest_share": round(unparsed / total, 3),
                "lambda_lines": 0, "lambda_share_of_biggest": 0.0,
                "verdict": "UNPARSED"}
    sized.sort(key=lambda t: -t[0])
    big_len, big = sized[0]

    # Lambda share is only meaningful for the dominant region, and finding its cursor means walking
    # the TU once more; skip it when the file is not dominated anyway.
    lam = 0
    if big_len / total >= DOMINANT_SHARE:
        for c in tu.cursor.get_children():
            if (c.location.file and pathlib.Path(c.location.file.name).resolve() == path.resolve()
                    and c.extent.start.line <= big["decl_line"] <= c.extent.end.line
                    and c.is_definition()):
                lam = lambda_lines(c)
                break

    big_regions = sum(1 for n, _ in sized if n >= SPLITTABLE_REGION_LINES)
    return {
        "file": rel, "lines": total, "regions": len(regions),
        "biggest": big["name"], "biggest_lines": big_len,
        "biggest_share": round(big_len / total, 3),
        "lambda_lines": lam,
        "lambda_share_of_biggest": round(lam / big_len, 3) if big_len else 0.0,
        "regions_over_%d" % SPLITTABLE_REGION_LINES: big_regions,
        "unparsed_lines": unparsed,
        "verdict": classify(total, big_len, lam, len(regions), unparsed),
    }


def selftest() -> int:
    """Pin the classifier on constructed cases -- no parsing, so it cannot pass for a parse reason."""
    cases = [
        # (total, biggest, lambda, big_regions, expected)
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
    # UNPARSED cases carry the fifth argument, so they are listed separately.
    unparsed_cases = [
        # (total, biggest, lambda, regions, unparsed, expected)
        (7637,  180, 0, 246, 3909, "UNPARSED"),   # hle_kernel_mem's REAL shape: 51% behind an #if.
                                                  # Scored SPLIT before the unparsed span was
                                                  # counted, from the 49% the parser could see.
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
        print(f"  [{'ok' if ok else 'FAIL'}] classify({total},{big},{lam},{nreg},unparsed={unp}) = {got}"
              f"{'' if ok else f' (want {want})'}")
        bad += not ok
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
                if len(p.read_text(errors="replace").splitlines()) >= args.min_lines:
                    cands.append(p)
    cands.sort()
    print(f"[survey] {len(cands)} file(s) at or above {args.min_lines} lines; parsing with {args.jobs} job(s)\n")

    rows: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for res in ex.map(lambda p: survey_one(p, db), cands):
            rows.append(res)

    rows.sort(key=lambda r: -r.get("lines", 0))
    order = {"SPLIT": 0, "EXTRACT": 1, "UNPARSED": 2, "HAND": 3, "PARSE-FAIL": 4, "OK": 5}
    print(f"{'verdict':<11}{'lines':>7}{'big':>7}{'share':>7}{'lam%':>6}  file / dominant symbol")
    print("-" * 100)
    for r in sorted(rows, key=lambda r: (order.get(r["verdict"], 9), -r.get("lines", 0))):
        if r["verdict"] == "PARSE-FAIL":
            print(f"{'PARSE-FAIL':<11}{r['lines']:>7}{'':>7}{'':>7}{'':>6}  {r['file']}\n"
                  f"{'':>38}  {r['why'][:90]}")
            continue
        rel = r["file"].split("prosper/", 1)[-1]
        print(f"{r['verdict']:<11}{r['lines']:>7}{r.get('biggest_lines', 0):>7}"
              f"{r.get('biggest_share', 0):>7.0%}{r.get('lambda_share_of_biggest', 0):>6.0%}  "
              f"{rel}\n{'':>38}  {r.get('biggest', '')}")

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
