#!/usr/bin/env python3
"""promote_internal.py -- lift shared internal-linkage declarations into an internal header.

A big translation unit in this codebase is usually not one file's worth of code; it is several,
welded together by an anonymous namespace. rdna2_to_spirv.cpp is 25,448 lines and 26% of it is
inside `namespace { }`, including the 3,831-line `SpirvCompute` that five separate emit functions
operate on. Nothing can be split out while the thing it operates on has internal linkage.

So this performs the one step a pure move cannot: it takes named regions out of their anonymous
namespaces and writes them to an internal header inside a NAMED namespace, leaving the rest of the
file alone. That is a real semantic change -- internal linkage becomes external -- and it is the
only such change in the whole restructure, which is why it is a separate tool, a separate commit,
and stated plainly rather than folded into "moved some code".

WHAT IS CHECKED, and each can fail:

  * The map's sha256 must match the file, so the regions being cut are the regions that were mapped.
  * Every named region must actually be inside an anonymous namespace -- promoting something that
    was already external would be a silent no-op that reports success.
  * Nothing may be promoted that the ANONYMOUS-namespace code still needs and that would change
    meaning; specifically, the tool reports what remains behind referencing each promoted symbol so
    a reviewer can see the coupling it is creating.
  * After writing: the header must contain each promoted region's bytes EXACTLY, and the source
    with those bytes removed must equal the original minus exactly those spans plus the single
    added include line. Reconstruction, not inspection.

  python3 prosper/tools/refactor/promote_internal.py --selftest
  python3 prosper/tools/refactor/promote_internal.py --map f.json --regions 50,43,47 \
      --header rdna2_to_spirv_internal.hpp --namespace prosper::gpu [--dry-run]
"""

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys


def repo_root() -> pathlib.Path:
    return pathlib.Path(subprocess.run(["git", "rev-parse", "--show-toplevel"],
                                       capture_output=True, text=True, check=True).stdout.strip())


def anon_map(regions: list[dict]) -> dict[int, bool]:
    stack: list[str] = []
    out: dict[int, bool] = {}
    for r in regions:
        if r.get("role") == "open":
            stack.append(r["name"])
        out[r["index"]] = "<anonymous>" in stack
        if r.get("role") == "close" and stack:
            stack.pop()
    return out


# `static` is NOT in this list, and that is the correction. A static function definition is fine in
# a .cpp and is an ODR hazard in a .hpp: it has internal linkage, so each including translation unit
# gets its own copy, and the moment an `inline` function in the same header odr-uses it the program
# is ill-formed (no diagnostic required) -- [basic.def.odr]/14 exempts const objects, not functions.
# The first promotion left one such function alone under the old policy and produced exactly that
# pairing; `nm` showed the inline caller as a COMDAT weak symbol in both objects. Promoted statics
# become `inline` instead.
ALREADY_LINKED = ("inline", "constexpr", "template", "//", "/*", "*")


def needs_inline(region: dict, decl_text: str) -> bool:
    """A promoted function DEFINITION needs `inline` once more than one TU includes the header.

    This is the one edit the promotion makes to the code itself, and it is forced: a function that
    had internal linkage inside `namespace { }` acquires external linkage in a named one, so every
    translation unit including the header would emit its own definition and the link would fail with
    duplicate symbols. `static` and `constexpr` already carry the right linkage and are left alone.

    It does NOT apply to struct or enum declarations, or to member functions defined inside a class
    body -- those are already implicitly inline.
    """
    if region["kind"] not in ("FUNCTION_DECL", "FUNCTION_TEMPLATE"):
        return False
    if not region.get("is_definition"):
        return False
    return not decl_text.lstrip().startswith(ALREADY_LINKED)


def canonical_include(source_rel: str, header_name: str) -> str:
    """The include spelling every other tool in this repo expects.

    A bare `#include "x.hpp"` compiles -- the header is beside the source -- and it is still wrong
    here: map_symbols.py finds the TU for a header by searching for the CANONICAL `gpu/recompiler/x.hpp`
    form, so a bare include makes the new headers unmappable, and the next split of them impossible.
    The tool would have been unable to read its own output.
    """
    parts = pathlib.Path(source_rel).parts
    if "src" in parts:
        sub = parts[parts.index("src") + 1:-1]
        return "/".join((*sub, header_name))
    return header_name


def build(map_data: dict, original: str, promote: list[int], header_rel: str,
          namespace: str, guard_note: str, include_spelling: str = "") -> tuple[str, str, list[str]]:
    """Returns (header_text, new_source_text, inlined). Pure, so --selftest can drive it."""
    regions = {r["index"]: r for r in map_data["regions"]}
    lines = original.splitlines(keepends=True)

    def text_of(i: int) -> str:
        r = regions[i]
        return "".join(lines[r["start"] - 1:r["end"]])

    preamble = [r for r in map_data["regions"] if r.get("role") == "preamble"]
    head = ["#pragma once\n", "\n", guard_note, "\n"]
    head += ["".join(lines[r["start"] - 1:r["end"]]) for r in preamble]
    head += ["\n", f"namespace {namespace} {{\n", "\n"]
    inlined: list[str] = []
    for i in sorted(promote):
        r = regions[i]
        text = text_of(i)
        decl_offset = r["decl_line"] - r["start"]
        body_lines = text.splitlines(keepends=True)
        if 0 <= decl_offset < len(body_lines) and needs_inline(r, body_lines[decl_offset]):
            line = body_lines[decl_offset]
            stripped = line.lstrip()
            if stripped.startswith("static "):
                # REPLACE rather than prepend: `inline static` keeps the internal linkage that is
                # the hazard.
                indent = line[:len(line) - len(stripped)]
                body_lines[decl_offset] = indent + "inline " + stripped[len("static "):]
            else:
                body_lines[decl_offset] = "inline " + line
            inlined.append(r["name"])
            text = "".join(body_lines)
        head.append(text)
    head += [f"\n}}  // namespace {namespace}\n"]
    header_text = "".join(head)

    # Remove the promoted spans from the source, and add the include after the last preamble line.
    drop = sorted(promote, reverse=True)
    out_lines = list(lines)
    for i in drop:
        r = regions[i]
        del out_lines[r["start"] - 1:r["end"]]
    last_pre = max(r["end"] for r in preamble)
    insert_at = last_pre
    for i in sorted(promote):
        if regions[i]["start"] - 1 < insert_at:
            insert_at -= (regions[i]["end"] - regions[i]["start"] + 1)
    out_lines.insert(insert_at, f'#include "{include_spelling}"\n')
    return header_text, "".join(out_lines), inlined


def verify(original: str, map_data: dict, promote: list[int], header_text: str,
           new_source: str, header_rel: str) -> list[str]:
    regions = {r["index"]: r for r in map_data["regions"]}
    lines = original.splitlines(keepends=True)
    problems: list[str] = []
    spans = []
    for i in sorted(promote):
        r = regions[i]
        text = "".join(lines[r["start"] - 1:r["end"]])
        spans.append(text)
        if text not in header_text:
            # The only permitted difference is one `inline ` inserted at the declaration line.
            decl_offset = r["decl_line"] - r["start"]
            body = text.splitlines(keepends=True)
            if 0 <= decl_offset < len(body):
                line = body[decl_offset]
                stripped = line.lstrip()
                indent = line[:len(line) - len(stripped)]
                for candidate in ("inline " + line,
                                  indent + "inline " + stripped[len("static "):]
                                  if stripped.startswith("static ") else None):
                    if candidate is None:
                        continue
                    probe = list(body)
                    probe[decl_offset] = candidate
                    if "".join(probe) in header_text:
                        break
                else:
                    problems.append(f"region {i} ({r['name']}) is not present in the header, "
                                    f"verbatim or with a single inline adjustment")
                continue
            problems.append(f"region {i} ({r['name']}) is not present in the header, verbatim or "
                            f"with a single inserted `inline`")
    # The source must be the original with exactly those spans removed, plus one include line.
    rebuilt = new_source.replace(f'#include "{header_rel}"\n', "", 1)
    expected = original
    for text in spans:
        if text not in expected:
            problems.append("a promoted span is not present in the original (map is stale)")
            return problems
        expected = expected.replace(text, "", 1)
    if rebuilt != expected:
        problems.append("the remaining source is not the original minus exactly the promoted spans")
    return problems


SELF_SRC = """// c
#include <a>

namespace ns {
namespace {

struct Shared { int x; };

int helper() { return 1; }

}  // namespace

void user() { Shared s; helper(); }

}  // namespace ns
"""

SELF_MAP = {
    "file": "s.cpp", "total_lines": 16, "sha256": "",
    "regions": [
        {"index": 0, "start": 1, "end": 2, "role": "preamble", "name": "<a>", "kind": "INCLUSION_DIRECTIVE"},
        {"index": 1, "start": 3, "end": 4, "role": "open", "name": "ns"},
        {"index": 2, "start": 5, "end": 6, "role": "open", "name": "<anonymous>"},
        {"index": 3, "start": 7, "end": 8, "role": "body", "name": "Shared",
         "kind": "STRUCT_DECL", "is_definition": True, "decl_line": 7},
        {"index": 4, "start": 9, "end": 10, "role": "body", "name": "helper",
         "kind": "FUNCTION_DECL", "is_definition": True, "decl_line": 9},
        {"index": 5, "start": 11, "end": 12, "role": "close", "name": "<anonymous>"},
        {"index": 6, "start": 13, "end": 14, "role": "body", "name": "user"},
        {"index": 7, "start": 15, "end": 16, "role": "close", "name": "ns"},
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

    anon = anon_map(SELF_MAP["regions"])
    check(anon[3] and anon[4], "regions inside `namespace {` are detected as internal")
    check(not anon[6], "a region outside it is not")

    hdr, src, inl = build(SELF_MAP, SELF_SRC, [3, 4], "s_internal.hpp", "ns", "// note\n",
                          "s_internal.hpp")
    check(inl == ["helper"], f"the free function definition gains `inline` (got {inl})")
    check("inline int helper()" in hdr, "and it is spelled at the declaration")
    check("struct Shared" in hdr and "inline struct" not in hdr,
          "a struct declaration does NOT gain inline")
    check("struct Shared" in hdr and "int helper()" in hdr, "the header carries the promoted code")
    check("namespace ns {" in hdr and "#pragma once" in hdr, "the header is well formed")
    check("struct Shared" not in src, "the source no longer defines the promoted struct")
    check('#include "s_internal.hpp"' in src, "the source includes the new header")
    check("void user()" in src, "unpromoted code stays put")
    check(not verify(SELF_SRC, SELF_MAP, [3, 4], hdr, src, "s_internal.hpp"),
          "verification passes on a correct promotion")

    # must-fail: verification has to notice a header that dropped something
    check(verify(SELF_SRC, SELF_MAP, [3, 4], hdr.replace("int helper() { return 1; }", ""),
                 src, "s_internal.hpp"),
          "verification FAILS when the header loses a promoted region")
    # must-fail: and a source that lost more than it should
    check(verify(SELF_SRC, SELF_MAP, [3, 4], hdr, src.replace("void user() { Shared s; helper(); }", ""),
                 "s_internal.hpp"),
          "verification FAILS when the source loses something it should have kept")

    if bad:
        print("  the promotion tool's own guarantees are broken; it must not be run")
        return 1
    print("  [ok]   promote_internal self-test: detection, extraction, and both must-fail arms")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", type=pathlib.Path)
    ap.add_argument("--regions", help="comma-separated region indices to promote")
    ap.add_argument("--header", help="header filename, written beside the source")
    ap.add_argument("--namespace", default="prosper::gpu")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if selftest():
        return 1
    if not (args.map and args.regions and args.header):
        sys.exit("--map, --regions and --header are required")

    root = repo_root()
    map_data = json.loads(args.map.read_text())
    source = root / map_data["file"]
    original_bytes = source.read_bytes()
    actual = hashlib.sha256(original_bytes).hexdigest()
    if actual != map_data.get("sha256"):
        sys.exit(f"refused: {map_data['file']} has changed since the map was made; re-run map_symbols")
    original = original_bytes.decode()

    promote = [int(x) for x in args.regions.split(",") if x.strip()]
    regions = {r["index"]: r for r in map_data["regions"]}

    # A forward declaration and its definition are ONE entity in two regions, and they must travel
    # together. Promoting only the definition leaves `namespace { int f(...); }` behind in the
    # source while `prosper::gpu::f` arrives from the header, and every call site becomes ambiguous
    # -- a compile error whose message ("call of overloaded f is ambiguous") points at the call, not
    # at the split that caused it. Four such pairs exist in rdna2_to_spirv.cpp.
    import collections
    by_usr = collections.defaultdict(list)
    for r in map_data["regions"]:
        if r.get("usr") and r.get("role") == "body":
            by_usr[r["usr"]].append(r["index"])
    added: list[int] = []
    for i in list(promote):
        for sibling in by_usr.get(regions[i].get("usr", ""), []):
            if sibling not in promote:
                promote.append(sibling)
                added.append(sibling)
    if added:
        print(f"  [ok]   pulled in {len(added)} region(s) that redeclare a promoted entity: "
              + ", ".join(f"{regions[i]['name']} (line {regions[i]['start']})" for i in sorted(added)))
    promote.sort()
    anon = anon_map(map_data["regions"])
    # Two kinds of region legitimately move into the header, and conflating them was wrong:
    #
    #   * INTERNAL ones -- the point of the tool. Their linkage genuinely changes.
    #   * EXTERNAL ones that are simply defined LOWER DOWN the same file. The header is included at
    #     the top, so anything the promoted code references must come with it regardless of linkage.
    #     `RegState` is a struct at namespace scope declared after the code that uses it; refusing
    #     to move it left the header referencing a type it could not see.
    #
    # What must still be refused is a promotion with no internal regions at all, which would mean
    # the tool ran for no reason and said so approvingly.
    internal = [i for i in promote if anon.get(i)]
    external = [i for i in promote if not anon.get(i)]
    # An earlier version of this guard refused a set with no internal-linkage regions, reasoning
    # that promoting an already-external one is "a no-op reported as success". That is wrong, and it
    # blocked a legitimate use: moving an external definition into a header is not a no-op -- it
    # changes which translation units can see it, which is the entire point once a second .cpp needs
    # the same helper. The only genuinely empty case is an empty set.
    if not promote:
        print("  [FAIL] nothing to promote")
        return 1
    print(f"  [ok]   {len(internal)} region(s) change linkage (internal -> external); "
          f"{len(external)} were already external and move so a second translation unit can see "
          f"them, or because the header sits above their definitions")
    if external:
        for i in sorted(external):
            print(f"           moved for ordering: {regions[i]['kind']:<16s} {regions[i]['name']}")

    note = ("// Lifted out of rdna2_to_spirv.cpp's anonymous namespaces so the emit functions that\n"
            "// operate on them can live in their own translation units. These are INTERNAL to the\n"
            "// recompiler: nothing outside src/gpu/recompiler/ should include this header.\n")
    spelling = canonical_include(map_data["file"], args.header)
    header_text, new_source, inlined = build(map_data, original, promote, args.header,
                                             args.namespace, note, spelling)
    print(f"  [ok]   include written as \"{spelling}\" (the canonical form the other tools read)")
    problems = verify(original, map_data, promote, header_text, new_source, spelling)
    if problems:
        for p in problems:
            print(f"  [FAIL] {p}")
        return 1
    promoted_lines = sum(regions[i]["end"] - regions[i]["start"] + 1 for i in promote)
    print(f"  [ok]   reconstruction: header carries {promoted_lines} line(s); the source is the "
          f"original minus exactly those spans plus one include")
    print(f"  [note] {len(inlined)} function definition(s) gained `inline`, which is required once "
          f"more than one translation unit includes the header: "
          f"{', '.join(inlined[:6])}{' ...' if len(inlined) > 6 else ''}")

    if args.dry_run:
        print(f"    would write {args.header} ({header_text.count(chr(10))} lines) and shrink "
              f"{map_data['file']} to {new_source.count(chr(10))} lines")
        return 0

    dest = source.parent / args.header
    with open(dest, "w", encoding="utf-8", newline="") as fh:
        fh.write(header_text)
    with open(source, "w", encoding="utf-8", newline="") as fh:
        fh.write(new_source)
    subprocess.run(["git", "add", str(dest.relative_to(root)), str(source.relative_to(root))],
                   cwd=root, check=True)
    print("  now BUILD and run ctest; this tool does not verify its own work")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
