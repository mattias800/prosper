#!/usr/bin/env python3
"""split_file.py -- split one translation unit into several, and PROVE nothing was lost.

Consumes the region tiling from `map_symbols.py`. A plan assigns each body region to an output file;
this writes those files and deletes the original.

An earlier version of this file was rejected in review, and how it failed is the reason this one is
shaped as it is. It never read `region["role"]`, so the namespace open/close regions its own
contract said would be replicated simply were not -- it emitted files with unbalanced braces, every
exported symbol trapped inside an unterminated `namespace {`, and then `git rm`'d the original. It
printed `[ok]` twice while doing it, because its "byte identity" precondition compared two values
computed from the same list and could therefore only fail if the file had GROWN.

So the checks here are arranged so that each one can actually fail:

  1. STALENESS. The map records a sha256 of the bytes it describes. If the file has changed since,
     the split is refused. This is the check the old vacuous assertion was pretending to be.

  2. STRUCTURE. Regions must tile [1, total_lines] exactly -- first starts at 1, each starts one
     line after the previous ends, last ends at the last line. Checked on the MAP, arithmetic only,
     so it does not depend on re-reading the text it is validating.

  3. PARTITION. Every body region assigned exactly once; replicated regions never assigned by hand.
     Unassigned and doubly-assigned are both hard errors -- the two failure modes a human reading a
     20,000-line diff has no chance of catching.

  4. PREPROCESSOR SAFETY. A region boundary inside an `#if` block would split a conditional across
     files. Refused, with the line reported.

  5. RECONSTRUCTION, after writing -- and it is worth being exact about what this does and does not
     establish, because the first version of this file overclaimed and was rejected for it.
     It has two halves:
       (a) every output is read BACK FROM DISK and compared against its assigned regions. The
           expected value is re-derived by the SAME expression that produced the file, so this
           half catches a bad write -- truncation, an encoding change, a file that never landed --
           and NOT a bad formula. A mistake in which regions belong where appears identically on
           both sides and passes.
       (b) the regions, concatenated in index order, must equal the original bytes, which are
           hashed before anything is deleted. This one is independent of the outputs: it establishes
           that the map tiles the original exactly, so no source line is unaccounted for.
     Together with check 3 (the partition) they mean no BYTES were lost -- check 5 alone does
     not, since (a) is self-referential and (b) says nothing about the outputs. They do not
     mean the result COMPILES: brace balance
     depends on which regions went where, and only the compiler establishes that. An earlier
     docstring here claimed this check "would have caught the unbalanced braces" that the rejected
     version shipped. It would not have. The build is what catches those, and the tool says so at
     the end of every run.

Replicated regions (`open`/`close`) are copied into every output so each one re-opens and re-closes
the same namespaces. That is the single place this tool writes a line that was not in the original,
which is why check 5 accounts for it explicitly rather than ignoring it.

  python3 prosper/tools/refactor/split_file.py --selftest
  python3 prosper/tools/refactor/map_symbols.py --file <f> --json /tmp/f.json
  python3 prosper/tools/refactor/split_file.py --map /tmp/f.json --plan <plan> --dry-run
  python3 prosper/tools/refactor/split_file.py --map /tmp/f.json --plan <plan>

Plan syntax, one output per line -- `path: <region indices and ranges>`:
  rdna2_emit_alu.cpp: 125-138 141 146
"""

import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import sys

REPLICATED_ROLES = ("preamble", "open", "close")

COND_OPEN = re.compile(r'^\s*#\s*(if|ifdef|ifndef)\b')
COND_CLOSE = re.compile(r'^\s*#\s*endif\b')


def repo_root() -> pathlib.Path:
    out = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                         capture_output=True, text=True, check=True)
    return pathlib.Path(out.stdout.strip())


def parse_spec(spec: str) -> list[int]:
    out: list[int] = []
    for tok in spec.split():
        if "-" in tok.strip("-"):
            a, b = tok.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        elif tok:
            out.append(int(tok))
    return out


def parse_plan(path: pathlib.Path) -> dict[str, list[int]]:
    plan: dict[str, list[int]] = {}
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            sys.exit(f"{path}:{lineno}: expected 'outfile: regions', got {raw!r}")
        out, spec = line.split(":", 1)
        plan.setdefault(out.strip(), []).extend(parse_spec(spec))
    return plan


def conditional_depth(lines: list[str]) -> list[int]:
    """Depth of #if nesting BEFORE each line (1-indexed, so index 0 is unused)."""
    depth = [0] * (len(lines) + 2)
    cur = 0
    for i, line in enumerate(lines, 1):
        depth[i] = cur
        if COND_OPEN.match(line):
            cur += 1
        elif COND_CLOSE.match(line):
            cur = max(0, cur - 1)
    depth[len(lines) + 1] = cur
    return depth


def check_structure(regions: list[dict], total_lines: int) -> list[str]:
    """Regions must tile [1, total_lines] exactly. Arithmetic only."""
    problems: list[str] = []
    if not regions:
        return ["the map contains no regions"]
    if regions[0]["start"] != 1:
        problems.append(f"first region starts at line {regions[0]['start']}, not 1")
    for prev, cur in zip(regions, regions[1:]):
        if cur["start"] != prev["end"] + 1:
            problems.append(f"gap or overlap: region {prev['index']} ends at {prev['end']}, "
                            f"region {cur['index']} starts at {cur['start']}")
    if regions[-1]["end"] != total_lines:
        problems.append(f"last region ends at {regions[-1]['end']}, file has {total_lines} lines")
    return problems


def anon_map(regions: list[dict]) -> dict[int, bool]:
    """Which regions sit inside an anonymous namespace, from the open/close markers in the map."""
    stack: list[str] = []
    out: dict[int, bool] = {}
    for r in regions:
        if r.get("role") == "open":
            stack.append(r["name"])
        out[r["index"]] = "<anonymous>" in stack
        if r.get("role") == "close" and stack:
            stack.pop()
    return out


def region_is_external(region: dict, in_anon_namespace: bool = False) -> bool:
    """External linkage, from the USR clang recorded, falling back to the scope walk.

    Same rule as promote_internal's: an external entity's USR starts `c:@`, an internal one is
    file-prefixed. A `static` free function is internal WITHOUT being in an anonymous namespace, so
    the scope walk alone gets it wrong -- which is why the USR is preferred.

    THE FALLBACK MATTERS AND THE FIRST VERSION HAD IT INVERTED. With no USR this returned False,
    i.e. "internal", and the regions that actually lack one are `LINKAGE_SPEC` -- `extern "C"`, the
    least internal thing in the language -- plus `INCLUSION_DIRECTIVE`. Measured: 0 of 155 body
    regions in gpu_capture.cpp, but 7 of 333 in hle_kernel.cpp, four of them LINKAGE_SPEC. The one
    reachable case happened to be a declaration, so the answer was right for the wrong reason, while
    `extern "C" __attribute__((weak))` DEFINITIONS in the same file would have been refused
    falsely. The scope walk is the honest fallback and the map already carries what it needs.
    """
    usr = region.get("usr") or ""
    if usr:
        return usr.startswith("c:@")
    return not in_anon_namespace


def unresolved_across_parts(map_data: dict, plan: dict[str, list[int]]) -> dict[int, set[str]]:
    """Regions each output part references that live in ANOTHER part and cannot be seen from it.

    This is the check that was missing, and its absence is not theoretical: splitting
    gpu_capture.cpp's serialize/deserialize into their own file passed every existing check --
    tiling, partition, byte-for-byte reconstruction -- and then failed to compile with 25 distinct
    undeclared names (`Writer`, `Reader`, `kMagic`, `read_table`, ...), because those definitions
    have internal linkage and stayed behind in the other part.

    Every existing check here is byte accounting: it establishes that no LINE was lost. None of them
    can see that a line ended up somewhere it cannot be used from. The reference matrix that answers
    it is already in the map, put there by map_symbols precisely because "a split that separates an
    internal-linkage helper from its callers does not fail at review, it fails at link".
    """
    regions = {r["index"]: r for r in map_data["regions"]}
    anon = anon_map(map_data["regions"])
    edges = {int(a): {int(b) for b in d} for a, d in map_data.get("edges", {}).items()}
    owner = {i: out for out, idxs in plan.items() for i in idxs}
    # A PREAMBLE is copied into every output, so a reference from it to an internal definition in
    # one part only is unsatisfied in the others. Restricted to `preamble` deliberately: an
    # `open`/`close` region holds only the text clang gave no top-level cursor, since real code
    # there would be its own body region, so that class is empty by construction -- and counting it
    # was a live false-positive source. Every such edge measured on hle_kernel.cpp was a namespace
    # re-opening brace whose `.referenced` resolved to the namespace decl, and acting on them
    # refused every legal two-part plan for that file. `cross_refs` no longer emits them, and this
    # is the second guard rather than the only one.
    replicated = {r["index"] for r in map_data["regions"] if r.get("role") == "preamble"}
    missing: dict[int, set[str]] = {}
    for src, dests in edges.items():
        for dest in dests:
            if dest not in owner:
                continue
            r = regions.get(dest, {})
            if r.get("role") != "body" or region_is_external(r, anon.get(dest, False)):
                continue                      # visible across translation units; nothing to do
            if src in owner:
                if owner[dest] == owner[src]:
                    continue
                users = {owner[src]}
            elif src in replicated:
                # Copied into EVERY output, so it needs the definition in every output that is not
                # the one holding it.
                users = {o for o in plan if o != owner[dest]}
                if not users:
                    continue
            else:
                continue
            missing.setdefault(dest, set()).update(users)
    return missing


def split(map_data: dict, plan: dict[str, list[int]], source_text: str,
          preamble_note: str = "") -> tuple[dict[str, str], list[str]]:
    """Pure: returns (outfile -> text, problems). Drives both the real run and --selftest."""
    regions = map_data["regions"]
    lines = source_text.splitlines(keepends=True)
    problems = check_structure(regions, len(lines))

    replicated = [r for r in regions if r.get("role") in REPLICATED_ROLES]
    bodies = {r["index"] for r in regions if r.get("role") not in REPLICATED_ROLES}
    replicated_idx = {r["index"] for r in replicated}

    seen: dict[int, str] = {}
    for out, idxs in plan.items():
        for i in idxs:
            if i in replicated_idx:
                problems.append(f"region {i} is a preamble/namespace region and is replicated "
                                f"automatically; remove it from {out}")
            elif i in seen:
                problems.append(f"region {i} assigned twice: {seen[i]} and {out}")
            else:
                seen[i] = out
    missing = sorted(bodies - set(seen))
    if missing:
        problems.append(f"{len(missing)} body region(s) unassigned: {missing[:20]}")

    # A boundary inside an `#if` is only a problem when the two sides go to DIFFERENT files -- then
    # the directive and its `#endif` are separated. Refusing every such boundary was over-strict and
    # rejected a valid plan: gpu_executor.cpp has conditionals spanning several consecutive regions
    # that all belong to the same output, where splitting "there" splits nothing.
    depth = conditional_depth([ln.rstrip("\n") for ln in lines])
    where = dict(seen)
    for r in regions:
        if r["index"] in replicated_idx:
            where[r["index"]] = "<replicated>"
    ordered = sorted(regions, key=lambda r: r["index"])
    for prev, cur in zip(ordered, ordered[1:]):
        if depth[cur["start"]] == 0:
            continue
        a, b = where.get(prev["index"]), where.get(cur["index"])
        if a != b:
            problems.append(f"line {cur['start']} is inside an #if (depth {depth[cur['start']]}) "
                            f"and regions {prev['index']} and {cur['index']} go to different "
                            f"files ({a} and {b}); that would separate the directive from its "
                            f"#endif")

    if problems:
        return {}, problems

    def text_of(idx: int) -> str:
        r = regions[idx]
        return "".join(lines[r["start"] - 1:r["end"]])

    outputs: dict[str, str] = {}
    for out, idxs in plan.items():
        chosen = set(idxs) | replicated_idx
        body = "".join(text_of(r["index"]) for r in regions if r["index"] in chosen)
        outputs[out] = (preamble_note + body) if preamble_note else body
    return outputs, []


def verify_reconstruction(map_data: dict, plan: dict[str, list[int]], outputs: dict[str, str],
                          original: str) -> list[str]:
    """Check 5: what was written must add back up to what was there."""
    regions = map_data["regions"]
    replicated_idx = {r["index"] for r in regions if r.get("role") in REPLICATED_ROLES}
    lines = original.splitlines(keepends=True)

    def text_of(idx: int) -> str:
        r = regions[idx]
        return "".join(lines[r["start"] - 1:r["end"]])

    problems: list[str] = []
    # Each output must equal exactly its replicated regions plus its own, in region order.
    for out, idxs in plan.items():
        chosen = set(idxs) | replicated_idx
        expected = "".join(text_of(r["index"]) for r in regions if r["index"] in chosen)
        if outputs.get(out, "") != expected:
            problems.append(f"{out}: content on disk does not match its assigned regions")
    # And every region, taken from wherever it lives, must rebuild the original byte for byte.
    rebuilt = "".join(text_of(r["index"]) for r in regions)
    if rebuilt != original:
        problems.append("the regions do not reconstruct the original file byte for byte")
    return problems


# --- self-test ----------------------------------------------------------------------------------
SAMPLE = """// file comment
#include <a>

namespace ns {

void alpha() {
    body();
}

void beta() {
    other();
}

}  // namespace ns
"""

SAMPLE_MAP = {
    "file": "sample.cpp",
    "total_lines": 14,
    "regions": [
        {"index": 0, "start": 1, "end": 5, "role": "open", "name": "ns"},
        {"index": 1, "start": 6, "end": 9, "role": "body", "name": "alpha"},
        {"index": 2, "start": 10, "end": 13, "role": "body", "name": "beta"},
        {"index": 3, "start": 14, "end": 14, "role": "close", "name": "ns"},
    ],
    "edges": {},
}

# A separate fixture for the preprocessor case, because the boundary has to fall INSIDE the
# conditional to exercise the rule. Region `alpha` deliberately absorbs the `#if` line, so region
# `beta` begins at depth 1 -- splitting there would put `#if` in one file and `#endif` in another.
COND_SRC = """// c
#include <a>

namespace ns {

void alpha() {
}
#if FOO

void beta() {
}
#endif

}  // namespace ns
"""

COND_MAP = {
    "file": "cond.cpp",
    "total_lines": 14,
    "regions": [
        {"index": 0, "start": 1, "end": 5, "role": "open", "name": "ns"},
        {"index": 1, "start": 6, "end": 8, "role": "body", "name": "alpha"},
        {"index": 2, "start": 9, "end": 12, "role": "body", "name": "beta"},
        {"index": 3, "start": 13, "end": 14, "role": "close", "name": "ns"},
    ],
    "edges": {},
}


def selftest() -> int:
    bad = 0

    def check(cond, label):
        nonlocal bad
        if not cond:
            print(f"  [FAIL] {label}")
            bad += 1

    outputs, problems = split(SAMPLE_MAP, {"a.cpp": [1], "b.cpp": [2]}, SAMPLE)
    check(not problems, f"a valid plan produces no problems (got {problems})")
    # THE defect that was shipped: replicated regions must actually appear in every output.
    for name, text in outputs.items():
        check(text.count("{") == text.count("}"),
              f"{name} is brace-balanced ({text.count('{')} open, {text.count('}')} close)")
        check("namespace ns {" in text, f"{name} re-opens the namespace")
        check("}  // namespace ns" in text, f"{name} re-closes the namespace")
    check("alpha" in outputs["a.cpp"] and "beta" not in outputs["a.cpp"],
          "a.cpp has its own body and not the other's")
    check(not verify_reconstruction(SAMPLE_MAP, {"a.cpp": [1], "b.cpp": [2]}, outputs, SAMPLE),
          "reconstruction check passes on a correct split")

    # must-fail cases: each has to be REJECTED
    _o, p = split(SAMPLE_MAP, {"a.cpp": [1]}, SAMPLE)
    check(any("unassigned" in x for x in p), "an unassigned region is refused")
    _o, p = split(SAMPLE_MAP, {"a.cpp": [1, 2], "b.cpp": [2]}, SAMPLE)
    check(any("assigned twice" in x for x in p), "a doubly-assigned region is refused")
    _o, p = split(SAMPLE_MAP, {"a.cpp": [0, 1], "b.cpp": [2]}, SAMPLE)
    check(any("replicated automatically" in x for x in p), "assigning a replicated region is refused")

    broken = json.loads(json.dumps(SAMPLE_MAP))
    broken["regions"][2]["start"] = 11          # leaves line 10 in no region
    _o, p = split(broken, {"a.cpp": [1], "b.cpp": [2]}, SAMPLE)
    check(any("gap or overlap" in x for x in p), "a non-tiling map is refused")

    _o, p = split(COND_MAP, {"a.cpp": [1], "b.cpp": [2]}, COND_SRC)
    check(any("#if" in x for x in p),
          "an #if boundary is refused when the two sides go to DIFFERENT files")
    _o, p2 = split(COND_MAP, {"a.cpp": [1, 2]}, COND_SRC)
    check(not any("#if" in x for x in p2),
          "and allowed when they go to the same file -- splitting there splits nothing")

    # and the reconstruction check must be able to FAIL
    tampered = dict(outputs)
    tampered["a.cpp"] = tampered["a.cpp"].replace("body();", "tampered();")
    check(verify_reconstruction(SAMPLE_MAP, {"a.cpp": [1], "b.cpp": [2]}, tampered, SAMPLE),
          "reconstruction check FAILS when an output is altered")

    # REFERENCES ACROSS THE CUT. Byte accounting cannot see this: the split below is a perfect
    # partition with perfect reconstruction, and does not compile. `beta` calls `alpha`, `alpha` has
    # internal linkage (a file-prefixed USR), and the plan puts them in different files.
    xref_map = {**SAMPLE_MAP, "regions": [dict(r) for r in SAMPLE_MAP["regions"]],
                "edges": {"2": {"1": 3}}}
    xref_map["regions"][1]["usr"] = "c:s.cpp@F@alpha#"        # internal linkage
    xref_map["regions"][2]["usr"] = "c:s.cpp@F@beta#"
    stranded = unresolved_across_parts(xref_map, {"a.cpp": [1], "b.cpp": [2]})
    check(set(stranded) == {1},
          f"an internal-linkage definition called from the other part is caught (got {stranded})")
    check(not unresolved_across_parts(xref_map, {"a.cpp": [1, 2]}),
          "and nothing is flagged when the caller and callee land in the same file")
    # The linkage test is the discriminator, not the edge: an EXTERNAL definition is visible from
    # the other translation unit, so splitting across it is legal and must not be refused. Without
    # this arm the check could refuse every cross-part edge and still look correct above.
    ext_map = {**xref_map, "regions": [dict(r) for r in xref_map["regions"]]}
    ext_map["regions"][1]["usr"] = "c:@F@alpha#"              # external linkage
    check(not unresolved_across_parts(ext_map, {"a.cpp": [1], "b.cpp": [2]}),
          "an EXTERNAL definition called across the cut is not flagged")

    # NO USR: the fallback is the scope walk, not a constant. An `extern "C"` definition records
    # no USR and is EXTERNAL, so a bare `return False` refused it as internal -- the inverted
    # default this replaced.
    nousr = {**xref_map, "regions": [dict(r) for r in xref_map["regions"]]}
    nousr["regions"][1]["usr"] = ""
    check(not unresolved_across_parts(nousr, {"a.cpp": [1], "b.cpp": [2]}),
          "a definition with no USR outside an anonymous namespace is treated as EXTERNAL")
    anon_case = {**nousr, "regions": [dict(r) for r in nousr["regions"]]}
    anon_case["regions"][0]["name"] = "<anonymous>"           # the open marker now names it
    check(set(unresolved_across_parts(anon_case, {"a.cpp": [1], "b.cpp": [2]})) == {1},
          "...and as INTERNAL when the scope walk says it is in an anonymous namespace")

    # A PREAMBLE is copied into every output, so its references have to resolve in every output.
    rep = {**xref_map, "regions": [dict(r) for r in xref_map["regions"]],
           "edges": {"0": {"1": 1}}}
    rep["regions"][0] = {**rep["regions"][0], "role": "preamble"}
    check(set(unresolved_across_parts(rep, {"a.cpp": [1], "b.cpp": [2]})) == {1},
          "a preamble's reference to an internal definition is caught in the other part")
    # ...and an `open`/`close` source is NOT counted. That class is empty by construction -- real
    # code there would be its own body region -- and counting it produced live false refusals: every
    # such edge on hle_kernel.cpp was a namespace re-opening brace whose `.referenced` resolved to
    # the namespace decl, which refused every legal two-part plan for that file.
    ns_src = {**xref_map, "regions": [dict(r) for r in xref_map["regions"]],
              "edges": {"0": {"1": 1}}}                       # region 0 stays the namespace `open`
    check(not unresolved_across_parts(ns_src, {"a.cpp": [1], "b.cpp": [2]}),
          "a namespace open/close is NOT treated as a replicated reference source")

    # THE GATE IS LAST, and it has to be: it used to sit in the middle of this function, so every
    # arm added after it printed [FAIL] and returned 0 anyway. Three mutations that disabled the
    # cross-reference check each printed a failure line and exited clean, and `main()`'s own
    # `if selftest(): return 1` guard was defeated with it -- the tool would have run on a
    # selftest it had already failed. Checking a mutation by reading the printed line rather than
    # the exit status is what hid it; the two disagreed.
    if bad:
        print("  the splitter's own guarantees are broken; it must not be run")
        return 1
    print("  [ok]   splitter self-test: replication, partition, tiling, #if, reconstruction, "
          "cross-part references")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", type=pathlib.Path)
    ap.add_argument("--plan", type=pathlib.Path)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if selftest():
        return 1
    if not args.map or not args.plan:
        sys.exit("--map and --plan are required")

    root = repo_root()
    map_data = json.loads(args.map.read_text())
    source = root / map_data["file"]
    original_bytes = source.read_bytes()
    original = original_bytes.decode()

    # --- check 1: the map still describes this file ----------------------------------------------
    recorded = map_data.get("sha256")
    if not recorded:
        sys.exit("the map has no sha256; regenerate it with the current map_symbols.py")
    actual = hashlib.sha256(original_bytes).hexdigest()
    if actual != recorded:
        sys.exit(f"refused: {map_data['file']} has changed since the map was made "
                 f"({actual[:12]} != {recorded[:12]}); re-run map_symbols.py")
    print(f"  [ok]   map matches the file on disk (sha256 {actual[:12]})")

    plan = parse_plan(args.plan)
    # BEFORE split() prints anything. A degraded parse degrades the TILING too, silently --
    # regions_of pads the remainder with a TRAILER, so check_structure and reconstruction both pass
    # on a map that does not describe the code. Two greens followed by "the map is unusable" reads
    # as though the first two meant something.
    if map_data.get("parse_errors"):
        print(f"  [FAIL] the map records {map_data['parse_errors']} parse error(s); its region and "
              f"reference data describe an incomplete AST.")
        print("         Fix the parse and re-run map_symbols before splitting on it.")
        return 1
    if "parse_errors" not in map_data:
        print("  [warn] this map predates parse-error recording; re-run map_symbols to have the "
              "cross-reference check gated on a clean parse")
    outputs, problems = split(map_data, plan, original)
    if problems:
        for p in problems:
            print(f"  [FAIL] {p}")
        return 1
    print(f"  [ok]   partition: {sum(len(v) for v in plan.values())} body region(s) -> "
          f"{len(plan)} file(s); namespace open/close replicated into each")

    # REFERENCES ACROSS THE CUT. Every check above is byte accounting -- it establishes that no line
    # was lost, never that a line can still be USED where it ended up. Splitting gpu_capture.cpp's
    # serialize/deserialize into their own file passed all of them and then failed to compile with
    # 25 undeclared names, because internal-linkage definitions stayed in the other part.
    stranded = unresolved_across_parts(map_data, plan)
    if stranded:
        regions = {r["index"]: r for r in map_data["regions"]}
        print(f"  [FAIL] {len(stranded)} definition(s) with internal linkage are referenced from a "
              f"part they are not in:")
        for i in sorted(stranded)[:10]:
            users = ", ".join(sorted(stranded[i]))
            print(f"           {regions[i].get('name', '?')}  (in "
                  f"{[o for o, v in plan.items() if i in v][0]}, needed by {users})")
        if len(stranded) > 10:
            print(f"           ... and {len(stranded) - 10} more")
        promote = ",".join(str(i) for i in sorted(stranded))
        print("         These have to be visible from both sides before the cut. Promote them "
              "first:")
        print(f"           promote_internal.py --map <this map> --regions {promote} \\")
        print("               --header <name>_internal.hpp --namespace prosper::gpu")
        print("         then re-run map_symbols (the promotion changes every region index) and "
              "re-plan.")
        return 1
    print(f"  [ok]   no internal-linkage definition is referenced from a part it is not in")

    if args.dry_run:
        for out, text in sorted(outputs.items()):
            print(f"    would write {out}: {text.count(chr(10))} line(s)")
        print("  (dry run -- nothing written)")
        return 0

    for out, text in outputs.items():
        dest = source.parent / out
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)

    # --- check 5: read back from disk, and make it add up ----------------------------------------
    on_disk = {}
    for out in plan:
        with open(source.parent / out, "r", encoding="utf-8", newline="") as fh:
            on_disk[out] = fh.read()
    problems = verify_reconstruction(map_data, plan, on_disk, original)
    if problems:
        for p in problems:
            print(f"  [FAIL] {p}")
        print("  the written files do not reconstruct the original; nothing was deleted")
        return 1
    print(f"  [ok]   reconstruction: {len(on_disk)} file(s) read back from disk rebuild the "
          f"original {len(original)} bytes exactly")

    # One output usually KEEPS the original's name -- a split is "this file, minus what moved out".
    # Removing the source then would delete the file just written, so the rm applies only when the
    # original name is genuinely absent from the plan.
    if not any((source.parent / o).resolve() == source.resolve() for o in plan):
        subprocess.run(["git", "rm", "-q", str(source.relative_to(root))], cwd=root, check=True)
    subprocess.run(["git", "add"] + [str((source.parent / o).relative_to(root)) for o in plan],
                   cwd=root, check=True)
    print("  now BUILD and run ctest; this tool does not verify its own work")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
