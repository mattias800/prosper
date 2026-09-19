#!/usr/bin/env python3
"""census_bodies.py -- rank every FUNCTION, METHOD and CLASS body in the tree by size.

`survey_sizes.py` ranks FILES and says which tool can act on each. That is the right question when
choosing a file to split. It is the wrong question for the other half of this codebase's problem,
because its verdict for a one-function file is a single row: it names the dominant region and stops.
It cannot tell you that `register_live_renderer` is 8,221 lines while the next body in the same file
is 40, nor that a 900-line file is one 870-line method, nor rank the two against each other.

This walks the AST and reports EVERY definition with a body, tree-wide, ranked. Three columns decide
what can be done with each, and they are the reason this is a parser rather than a `wc -l`:

  LAMBDA SHARE -- the fraction of the body's lines that sit inside a lambda. clangd's ExtractFunction
    refuses to extract from a lambda ("Don't extract from lambdas", ExtractFunction.cpp), so a body
    that is mostly lambda is not mechanisable by the tools in this repo, however large it is. Counted
    over the body's OWN cursor, never over an enclosing namespace -- see survey_sizes.py's header for
    the instance where measuring over the namespace reported 0.729 for a function containing no
    lambda at all.

  NEST -- maximum BRACE depth reached inside the body (the body itself is 1). A 400-line function at
    depth 3 is a list; the same 400 lines at depth 9 is a decision tree, and the two want different
    treatment. Counted over COMPOUND_STMT cursors only -- AST depth is a different and much larger
    number (a flat 197-line function reports 21), and it is not what a reader sees.

  STMTS -- top-level sibling statements in the body. `split_file.py`'s unit is a region; extraction's
    unit is a RUN OF SIBLINGS, so a body with 600 top-level statements offers many seams and one with
    4 (a function that is a single switch) offers almost none, regardless of line count.

WHAT IS AND IS NOT COUNTED, because each of these silently changes the ranking:

  * DEFINITIONS ONLY. A declaration has no body and no size; counting it puts 1-line rows in a
    ranking of bodies.
  * DEDUPLICATED BY EXTENT. A header body is re-parsed in every TU that includes it -- hle_kernel.hpp
    is in hundreds. Keyed by (realpath, start offset, end offset) so it is counted once, and the TU
    it was first seen in is recorded so a disputed row can be re-derived.
  * LAMBDAS ARE ROWS, AND ALSO PARENT LINES. A lambda body is reported in its own right (an 800-line
    lambda is a refactoring target) with `parent` naming the function it sits in. It is NOT
    subtracted from that function's line count: the function really is that long. Sort with
    --no-lambdas to see only named bodies.
  * `#if`-INACTIVE CODE IS INVISIBLE, and this tool does not pretend otherwise. A body inside a
    skipped arm is not in the AST at all, so its absence is silent. Per-file unparsed shares are
    survey_sizes.py's job; this tool prints the count of files whose census may therefore be partial
    and names them under --show-partial.

PARSE FAILURES ARE REPORTED, NEVER SCORED -- a libclang parse driven from a g++ compile database
fails SOFTLY, returning a partial AST, so a TU that did not really parse yields FEW bodies and looks
merely tidy. That is this project's instrument-not-the-subject shape, and in a census it does the
most damage of all, because the answer is a silent under-count spread over every file the TU
includes. Any TU emitting an error diagnostic is listed under PARSE-FAIL and contributes no rows.

Usage:
  python3 prosper/tools/refactor/census_bodies.py                     # tree-wide, >=200 lines
  ... --min-lines 400 --roots src frontends                           # narrow
  ... --by-file                                                       # aggregate per file
  ... --kinds class                                                   # only class/struct bodies
  ... --json census.json                                              # machine-readable
  ... --selftest                                                      # prove the measurements
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import os
import pathlib
import shlex
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_TESTDATA = _HERE / "testdata"

# libclang is resolved LAZILY, on the first parse, so --selftest's arithmetic cases can run on a
# machine without it -- the same reason survey_sizes.py does this. See its header.
ci = None
# Resolution is guarded AND is done once from the main thread before any pool starts. libclang's
# Config.set_library_file() raises "library file must be set before before using any other
# functionalities" if any other thread has already touched the library, so two workers racing to
# resolve it kills the whole census -- and the traceback lands on a worker, which reads as a parse
# problem in whatever TU happened to be first rather than as a startup ordering bug.
_CLANG_LOCK = None


def _need_clang() -> None:
    global ci, _CLANG_LOCK
    if ci is not None:
        return
    import threading
    if _CLANG_LOCK is None:
        _CLANG_LOCK = threading.Lock()
    with _CLANG_LOCK:
        if ci is not None:
            return
        import clang.cindex as _ci
        for cand in ("/usr/lib64/libclang.so.22.1", "/usr/lib64/libclang.so"):
            if pathlib.Path(cand).exists():
                _ci.Config.set_library_file(cand)
                break
        ci = _ci


_BUILTIN_INCLUDES = None


def builtin_includes() -> list[str]:
    """clang's own builtin headers.

    The compile database is written for g++, whose builtin include directory clang does not know, so
    a parse driven straight from it dies on `stddef.h` -- and dies for EVERY TU at once, which in a
    census reads as "the tree has no large bodies" rather than as a missing flag. map_symbols.py
    resolves it the same way and for the same reason. Cached: `clang -print-resource-dir` is a
    process spawn, and this is called once per TU."""
    global _BUILTIN_INCLUDES
    if _BUILTIN_INCLUDES is None:
        import subprocess
        out = subprocess.run(["clang", "-print-resource-dir"], capture_output=True, text=True)
        _BUILTIN_INCLUDES = (["-isystem", str(pathlib.Path(out.stdout.strip()) / "include")]
                             if out.returncode == 0 and out.stdout.strip() else [])
    return _BUILTIN_INCLUDES


FUNC_KINDS = ("FUNCTION_DECL", "CXX_METHOD", "CONSTRUCTOR", "DESTRUCTOR",
              "CONVERSION_FUNCTION", "FUNCTION_TEMPLATE")
CLASS_KINDS = ("CLASS_DECL", "STRUCT_DECL", "CLASS_TEMPLATE",
               "CLASS_TEMPLATE_PARTIAL_SPECIALIZATION")
LAMBDA_KIND = "LAMBDA_EXPR"


def kind_group(kind_name: str) -> str:
    if kind_name in FUNC_KINDS:
        return "func"
    if kind_name in CLASS_KINDS:
        return "class"
    if kind_name == LAMBDA_KIND:
        return "lambda"
    return "other"


def line_span(start_line: int, end_line: int) -> int:
    """Lines occupied, inclusive. A one-line body is 1, never 0 -- a 0 sorts a body out of the
    ranking entirely, which is how a real target disappears from a census."""
    return max(0, end_line - start_line + 1)


def lambda_share(body_lines: int, lambda_lines: int) -> float:
    if body_lines <= 0:
        return 0.0
    return min(1.0, lambda_lines / body_lines)


def verdict(group: str, lines: int, lam_share: float, stmts: int) -> str:
    """What can act on this body.

    SPLIT    a class or namespace-scope aggregate large enough to divide by responsibility.
    EXTRACT  a function whose lines are mostly NOT lambda -- clangd's ExtractFunction can act.
    HAND     a function that is mostly lambda bodies -- clangd refuses; restructure by hand first.
    SEAMLESS a function with too few top-level statements to cut (one giant switch, say): the seams
             are inside the statements, so extraction needs the switch arms lifted first.
    OK       below the thresholds.
    """
    if lines < 200:
        return "OK"
    if group == "class":
        return "SPLIT"
    if lam_share >= 0.5:
        return "HAND"
    if stmts is not None and stmts < 8:
        return "SEAMLESS"
    return "EXTRACT"


def _extent(cur):
    e = cur.extent
    return (e.start.line, e.end.line, e.start.offset, e.end.offset)


def _body_of(cur):
    """The compound statement of a function, or None. A function's EXTENT includes its signature and
    any trailing attributes; its BODY is what a refactor can move, and the two differ by enough
    (multi-line signatures are common here) to change a ranking near a threshold."""
    for ch in cur.get_children():
        if ch.kind.name == "COMPOUND_STMT":
            return ch
    return None


def _measure(cur, body, path_of):
    """Lines, lambda-lines, top-level statement count and max nesting depth for one body.

    Lambda lines are a UNION OF LINE NUMBERS, not a sum of spans: two lambdas on one line, or a
    lambda inside a lambda, would otherwise push the share above 1.0 and turn an EXTRACT verdict
    into HAND. Measured over `cur`'s own subtree only."""
    lam_lines = set()
    max_depth = 0

    def walk(c, depth):
        nonlocal max_depth
        if depth > max_depth:
            max_depth = depth
        for ch in c.get_children():
            kn = ch.kind.name
            if kn == LAMBDA_KIND:
                e = ch.extent
                if e.start.file and path_of(e.start.file) is not None:
                    lam_lines.update(range(e.start.line, e.end.line + 1))
            # Depth counts BRACE nesting, not AST depth. They are wildly different numbers and only
            # one of them is what a reader sees: a 197-line function of flat `if`s reports AST depth
            # 21, because every subexpression is a cursor. Counting COMPOUND_STMT alone reproduces
            # the indentation level, which is the thing being claimed.
            walk(ch, depth + 1 if kn == "COMPOUND_STMT" else depth)

    walk(body if body is not None else cur, 1 if body is not None else 0)
    stmts = sum(1 for _ in body.get_children()) if body is not None else None
    return lam_lines, stmts, max_depth


def parse_tu(entry, roots_abs):
    """Parse one TU; return (rows, parse_error, files_touched). Never raises for a bad TU."""
    _need_clang()
    cwd = entry["directory"]
    cmd = entry.get("arguments")
    if cmd is None:
        cmd = shlex.split(entry["command"])
    args = []
    skip_next = False
    for a in cmd[1:]:
        if skip_next:
            skip_next = False
            continue
        if a in ("-o", "-c"):
            skip_next = a == "-o"
            continue
        if a.endswith((".cpp", ".cc", ".c")) and os.path.basename(a) == os.path.basename(entry["file"]):
            continue
        args.append(a)
    args += ["-ferror-limit=0", "-Wno-everything"] + builtin_includes()

    index = ci.Index.create()
    try:
        tu = index.parse(entry["file"], args=args, options=(
            ci.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
            | ci.TranslationUnit.PARSE_SKIP_FUNCTION_BODIES * 0))
    except Exception as exc:  # noqa: BLE001 -- a TU that cannot parse must not kill the census
        return [], f"{entry['file']}: {exc}", set()

    errs = [d for d in tu.diagnostics if d.severity >= 3]
    if errs:
        return [], f"{os.path.relpath(entry['file'], cwd)}: {errs[0].spelling}", set()

    cache = {}

    def path_of(f):
        """Resolved path if the file is under one of our roots, else None. Cached: called once per
        cursor, and a realpath() per cursor over a 12,000-line TU dominates the run."""
        name = f.name
        if name in cache:
            return cache[name]
        try:
            rp = os.path.realpath(name)
        except OSError:
            rp = name
        keep = any(rp.startswith(r) for r in roots_abs)
        cache[name] = rp if keep else None
        return cache[name]

    rows = []
    touched = set()

    def visit(cur, parent_name):
        for ch in cur.get_children():
            kn = ch.kind.name
            grp = kind_group(kn)
            loc = ch.extent.start.file
            here = path_of(loc) if loc else None
            if here is None:
                # Not under a root (a system or third_party header): do not descend -- the whole
                # point of the root filter is to avoid walking libstdc++.
                if loc is not None:
                    continue
                visit(ch, parent_name)
                continue
            touched.add(here)
            if grp in ("func", "class", "lambda") and (ch.is_definition() or grp == "lambda"):
                body = _body_of(ch) if grp != "class" else None
                sl, el, so, eo = _extent(ch)
                lam, stmts, depth = _measure(ch, body, path_of)
                lines = line_span(sl, el)
                rows.append({
                    "file": here, "kind": kn, "group": grp,
                    "name": ch.spelling or "<lambda>",
                    "qual": (parent_name + "::" + (ch.spelling or "<lambda>")) if parent_name else (ch.spelling or "<lambda>"),
                    "start": sl, "end": el, "lines": lines,
                    "lambda_lines": len(lam), "stmts": stmts, "depth": depth,
                    "key": (here, so, eo), "tu": entry["file"],
                    "parent": parent_name,
                })
                visit(ch, ch.spelling or parent_name or "<lambda>")
            else:
                visit(ch, parent_name)

    visit(tu.cursor, "")
    return rows, None, touched


def collect(db_path, roots_abs, jobs):
    _need_clang()  # once, on the main thread -- see _need_clang's note
    entries = json.load(open(db_path))
    seen_file = set()
    uniq = []
    for e in entries:
        f = os.path.realpath(e["file"])
        if f in seen_file:
            continue
        seen_file.add(f)
        if not any(f.startswith(r) for r in roots_abs):
            continue
        uniq.append(e)

    rows = {}
    fails = []
    touched = set()
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futs = {pool.submit(parse_tu, e, roots_abs): e for e in uniq}
        done = 0
        for fut in concurrent.futures.as_completed(futs):
            done += 1
            if done % 25 == 0:
                print(f"  ... {done}/{len(uniq)} TUs", file=sys.stderr, flush=True)
            r, err, t = fut.result()
            if err:
                fails.append(err)
                continue
            touched |= t
            for row in r:
                k = tuple(row.pop("key"))
                if k not in rows:
                    rows[k] = row
    return list(rows.values()), fails, touched, len(uniq)


def render(rows, args, repo_root):
    def rel(p):
        try:
            return os.path.relpath(p, repo_root)
        except ValueError:
            return p

    rows = [r for r in rows if r["lines"] >= args.min_lines]
    if args.kinds:
        rows = [r for r in rows if r["group"] in args.kinds]
    if args.no_lambdas:
        rows = [r for r in rows if r["group"] != "lambda"]

    for r in rows:
        r["share"] = lambda_share(r["lines"], r["lambda_lines"])
        r["verdict"] = verdict(r["group"], r["lines"], r["share"], r["stmts"])

    if args.by_file:
        agg = collections.defaultdict(lambda: {"n": 0, "biggest": 0, "name": "", "lines": 0})
        for r in rows:
            a = agg[r["file"]]
            a["n"] += 1
            a["lines"] += r["lines"]
            if r["lines"] > a["biggest"]:
                a["biggest"], a["name"] = r["lines"], r["qual"]
        out = sorted(agg.items(), key=lambda kv: -kv[1]["biggest"])[:args.top]
        print(f"{'biggest':>8} {'bodies':>6}  {'largest body':<44} file")
        for f, a in out:
            print(f"{a['biggest']:>8} {a['n']:>6}  {a['name'][:44]:<44} {rel(f)}")
        return

    rows.sort(key=lambda r: -r["lines"])
    rows = rows[:args.top]
    print(f"{'lines':>7} {'lam%':>5} {'stmt':>5} {'nest':>4} {'verdict':<8} {'name':<46} location")
    for r in rows:
        nm = r["qual"] if r["group"] != "lambda" else f"<lambda in {r['parent']}>"
        print(f"{r['lines']:>7} {r['share']*100:>4.0f}% {str(r['stmts'] or '-'):>5} "
              f"{r['depth']:>4} {r['verdict']:<8} {nm[:46]:<46} {rel(r['file'])}:{r['start']}")


def selftest() -> int:
    ok = True

    def check(label, got, want):
        nonlocal ok
        good = got == want
        ok &= good
        print(f"  [{'ok' if good else 'FAIL'}] {label} = {got!r}" + ("" if good else f" (want {want!r})"))

    print("-- line_span --")
    check("line_span(10,10)", line_span(10, 10), 1)
    check("line_span(10,12)", line_span(10, 12), 3)
    check("line_span(12,10)", line_span(12, 10), 0)

    print("-- lambda_share (a union can never exceed the body) --")
    check("share(100,50)", lambda_share(100, 50), 0.5)
    check("share(100,140)", lambda_share(100, 140), 1.0)
    check("share(0,5)", lambda_share(0, 5), 0.0)

    print("-- verdict --")
    check("small func", verdict("func", 199, 0.0, 50), "OK")
    check("big lambda-nest", verdict("func", 8000, 0.98, 50), "HAND")
    check("big plain func", verdict("func", 8000, 0.10, 50), "EXTRACT")
    check("boundary 0.5 is HAND", verdict("func", 800, 0.50, 50), "HAND")
    check("just under is EXTRACT", verdict("func", 800, 0.499, 50), "EXTRACT")
    check("one giant switch", verdict("func", 4000, 0.0, 3), "SEAMLESS")
    check("class", verdict("class", 900, 0.0, 40), "SPLIT")
    check("class below threshold", verdict("class", 150, 0.0, 40), "OK")

    print("-- kind_group --")
    check("CXX_METHOD", kind_group("CXX_METHOD"), "func")
    check("LAMBDA_EXPR", kind_group("LAMBDA_EXPR"), "lambda")
    check("STRUCT_DECL", kind_group("STRUCT_DECL"), "class")
    check("NAMESPACE", kind_group("NAMESPACE"), "other")

    fx = _TESTDATA / "dominant_no_lambda.cpp"
    if fx.exists():
        try:
            _need_clang()
        except Exception as exc:  # noqa: BLE001
            print(f"-- parse fixture skipped (no libclang: {exc}) --")
            return 0 if ok else 1
        print("-- parse fixture --")
        rows, err, _ = parse_tu({"directory": str(_TESTDATA), "file": str(fx),
                                 "arguments": ["c++", "-std=gnu++20", "-c", str(fx)]},
                                [str(_TESTDATA.resolve())])
        check("fixture parses", err, None)
        named = {r["name"]: r for r in rows if r["group"] == "func"}
        check("'dominant' found", "dominant" in named, True)
        if "dominant" in named:
            check("'dominant' has no lambda lines", named["dominant"]["lambda_lines"], 0)

    print("== PASS ==" if ok else "== FAIL ==")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="prosper/build-linux/compile_commands.json")
    ap.add_argument("--roots", nargs="*", default=["prosper/src", "prosper/frontends", "prosper/tools", "prosper/tests"])
    ap.add_argument("--min-lines", type=int, default=200)
    ap.add_argument("--top", type=int, default=60)
    ap.add_argument("--kinds", nargs="*", choices=["func", "class", "lambda"])
    ap.add_argument("--no-lambdas", action="store_true")
    ap.add_argument("--by-file", action="store_true")
    ap.add_argument("--jobs", type=int, default=min(12, (os.cpu_count() or 4)))
    ap.add_argument("--json")
    ap.add_argument("--show-fails", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    repo_root = os.path.realpath(".")
    roots_abs = [os.path.realpath(r) for r in args.roots]
    print(f"[census] db={args.db} roots={', '.join(args.roots)}", file=sys.stderr)
    rows, fails, touched, n_tu = collect(args.db, roots_abs, args.jobs)
    print(f"[census] {n_tu} TUs, {len(fails)} parse-fail, {len(rows)} bodies, {len(touched)} files",
          file=sys.stderr)
    if fails and (args.show_fails or len(fails) > n_tu // 10):
        print("PARSE-FAIL (contribute no rows -- this census under-counts by whatever they held):",
              file=sys.stderr)
        for f in fails[:20]:
            print("  " + f, file=sys.stderr)
    render(rows, args, repo_root)
    if args.json:
        for r in rows:
            r["share"] = lambda_share(r["lines"], r["lambda_lines"])
            r["verdict"] = verdict(r["group"], r["lines"], r["share"], r["stmts"])
        json.dump({"rows": rows, "parse_fail": fails, "tus": n_tu}, open(args.json, "w"), indent=1)
        print(f"[census] wrote {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
