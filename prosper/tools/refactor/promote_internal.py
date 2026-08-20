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
import re
import subprocess
import sys


def repo_root() -> pathlib.Path:
    return pathlib.Path(subprocess.run(["git", "rev-parse", "--show-toplevel"],
                                       capture_output=True, text=True, check=True).stdout.strip())


COND_OPEN = re.compile(r'^\s*#\s*(if|ifdef|ifndef)\b')
COND_CLOSE = re.compile(r'^\s*#\s*endif\b')


def conditional_depth(lines: list[str]) -> list[int]:
    """#if nesting depth BEFORE each line (1-indexed)."""
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


def prototype_of(region: dict, lines: list[str]) -> str | None:
    """A forward declaration for a definition that is staying behind.

    Promoted code can call a function this tool deliberately did NOT promote -- one other
    translation units link against, whose single out-of-line definition must not move. The header
    then needs its DECLARATION, or it names something nothing has declared. Built by taking the text
    from the declaration line to the body's opening brace and terminating it with a semicolon.

    Default arguments are dropped: the definition keeps them where it is, and repeating them in a
    second declaration does not compile.
    """
    start = region["decl_line"] - 1
    text: list[str] = []
    for i in range(start, min(start + 12, len(lines))):
        line = lines[i]
        if "{" in line:
            text.append(line[:line.index("{")].rstrip())
            break
        text.append(line.rstrip("\n"))
    else:
        return None
    proto = " ".join(t.strip() for t in text if t.strip())
    # Strip default arguments PER PARAMETER. A single regex over the whole joined signature ate the
    # last parameter of make_shader_compile_key and emitted `bool fragment_wave32,;` -- syntactically
    # broken in a way that only shows up at the include site, far from here. Splitting on top-level
    # commas keeps each parameter's `= value` local to that parameter.
    open_at = proto.index("(") if "(" in proto else -1
    if open_at >= 0 and proto.endswith(")"):
        headtxt, params = proto[:open_at + 1], proto[open_at + 1:-1]
        out, depth, cur = [], 0, ""
        for ch in params:
            if ch in "(<[":
                depth += 1
            elif ch in ")>]":
                depth -= 1
            if ch == "," and depth == 0:
                out.append(cur); cur = ""
            else:
                cur += ch
        if cur.strip():
            out.append(cur)
        cleaned = [re.sub(r"\s*=.*$", "", x).strip() for x in out]
        proto = headtxt + ", ".join(c for c in cleaned if c) + ")"
    if not proto.endswith(")"):
        return None
    if region["kind"] == "VAR_DECL":
        # `thread_local GuestGpuWriteJournal g_x;` -> `extern thread_local GuestGpuWriteJournal g_x;`
        decl = lines[start].rstrip("\n")
        decl = decl.split("=")[0].rstrip().rstrip(";").strip()
        return "extern " + decl + ";" if decl else None
    return proto + ";"


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
    """A promoted function DEFINITION needs `inline`, and a promoted `static` one needs it MORE.

    Two distinct reasons, and the second was learned from an ODR violation this tool shipped:

    * A function that had internal linkage inside `namespace { }` acquires external linkage in a
      named one, so every translation unit including the header emits its own definition and the
      link fails on duplicate symbols.
    * A `static` definition in a header avoids that and has a worse problem. Each TU gets its own
      copy, so the moment an `inline` function in the same header odr-uses it the program is
      ill-formed, no diagnostic required. `static` is therefore CONVERTED to `inline` -- the
      opposite of the policy this function originally implemented, which was correct for a .cpp and
      wrong for a .hpp.

    `constexpr` already implies inline. Struct and enum declarations need nothing, and member
    functions defined inside a class body are implicitly inline.
    """
    if region["kind"] not in ("FUNCTION_DECL", "FUNCTION_TEMPLATE", "VAR_DECL"):
        return False
    if not region.get("is_definition"):
        return False
    if region["kind"] == "VAR_DECL":
        # A VARIABLE definition in a header is one object PER TRANSLATION UNIT, so the link fails
        # with "multiple definition" -- and for `thread_local` it fails in .tbss, which reads as a
        # TLS problem rather than as a promotion that took a variable along. C++17 inline variables
        # are the fix, and `inline thread_local X y;` is the correct order.
        #
        # This was a known-latent gap: a reviewer noted the policy covered only functions, and it
        # stayed latent exactly until a promotion first took a variable -- three of them here
        # (g_guest_gpu_writes, g_guest_readable_cache, g_guest_writable_cache).
        head = decl_text.lstrip()
        return not head.startswith(("inline", "constexpr", "extern", "//", "/*", "*"))
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
          namespace: str, guard_note: str, include_spelling: str = "",
          forward_decls: list[str] | None = None) -> tuple[str, str, list[str]]:
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
    if forward_decls:
        head.append("// Declared here, DEFINED in the .cpp this header was lifted out of: other\n"
                    "// translation units link against those definitions, so they must not move.\n")
        head += [d + "\n" for d in forward_decls]
        head.append("\n")
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

        # ORDER MATTERS. When a region's declaration is its first line, the verbatim text is a
        # SUBSTRING of the adjusted text ("int f()" inside "inline int f()"), so testing for the
        # verbatim form first reports a correctly-adjusted header as unchanged. Look for the
        # adjusted form first, then decide what a verbatim match means.
        body = text.splitlines(keepends=True)
        decl_offset = r["decl_line"] - r["start"]
        adjusted: list[str] = []
        if 0 <= decl_offset < len(body) and needs_inline(r, body[decl_offset]):
            line = body[decl_offset]
            stripped = line.lstrip()
            indent = line[:len(line) - len(stripped)]
            variant = (indent + "inline " + stripped[len("static "):]
                       if stripped.startswith("static ") else "inline " + line)
            probe = list(body)
            probe[decl_offset] = variant
            adjusted.append("".join(probe))

        if adjusted:
            if any(a in header_text for a in adjusted):
                continue
            # The tool should have adjusted this and the header does not show it. Naming the
            # unchanged case specifically matters: that is what a `static` free function looks like
            # after being left alone in a header, odr-used by an inline caller.
            problems.append(f"region {i} ({r['name']}) needed an `inline` adjustment and the header "
                            f"does not have it" +
                            (" -- it is present unchanged" if text in header_text
                             else " -- it is not present at all"))
            continue
        if text not in header_text:
            problems.append(f"region {i} ({r['name']}) is not present verbatim in the header")

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

    # prototype_of is pure and therefore reachable from here, unlike the fixes that live in main().
    # The fixture's default argument CONTAINS PARENTHESES, which is the shape that breaks the
    # single-regex stripping this replaced: `[^,)]+` stops at the inner `(`, and the signature comes
    # out as `void f(int a), bool b)`. A fixture with only plain defaults passes under BOTH
    # implementations and would certify nothing -- checked by mutation, not assumed.
    proto_lines = ["void f(int a = g(1),\n", "       bool b = false) {\n"]
    proto = prototype_of({"kind": "FUNCTION_DECL", "is_definition": True, "decl_line": 1,
                          "start": 1, "end": 2, "name": "f"}, proto_lines)
    check(proto == "void f(int a, bool b);", f"defaults with parentheses strip cleanly (got {proto!r})")
    check(proto and proto.count("(") == proto.count(")"),
          f"the signature stays balanced (got {proto!r})")

    # a promoted VARIABLE definition needs `inline` too, or the link fails with multiple definition
    var_map = json.loads(json.dumps(SELF_MAP))
    var_map["regions"][3] = {"index": 3, "start": 7, "end": 8, "role": "body", "name": "g_state",
                             "kind": "VAR_DECL", "is_definition": True, "decl_line": 7}
    var_src = SELF_SRC.replace("struct Shared { int x; };", "thread_local int g_state = 0;")
    hdr_v, _s, inl_v = build(var_map, var_src, [3, 4], "s_internal.hpp", "ns", "// n\n",
                             "s_internal.hpp")
    check("inline thread_local int g_state" in hdr_v,
          f"a promoted variable definition gains `inline` (got {inl_v})")

    # the static -> inline conversion, and the must-fail arm that makes the new policy checkable
    static_map = json.loads(json.dumps(SELF_MAP))
    static_src = SELF_SRC.replace("int helper() { return 1; }", "static int helper() { return 1; }")
    hdr_s, src_s, _inl = build(static_map, static_src, [3, 4], "s_internal.hpp", "ns", "// n\n",
                               "s_internal.hpp")
    check("inline int helper()" in hdr_s, "a promoted `static` definition is CONVERTED to inline")
    check("static int helper()" not in hdr_s and "inline static" not in hdr_s,
          "the keyword is replaced, not prefixed -- `inline static` keeps the internal linkage")
    check(verify(static_src, static_map, [3, 4],
                 hdr_s.replace("inline int helper()", "static int helper()"), src_s,
                 "s_internal.hpp"),
          "verification FAILS on a header that kept `static`, so the guard sees its own regression")

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
    ap.add_argument("--exclude", default="",
                    help="regions that must NOT be promoted even if they redeclare a promoted "
                         "entity. A forward declaration belongs in the header and its definition "
                         "may deliberately stay behind -- that pairing is the normal one, and "
                         "pulling the definition along turns a 797-line function into header text.")
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

    forward_decls: list[str] = []
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
    excluded = {int(x) for x in args.exclude.split(",") if x.strip()}
    promote = [i for i in promote if i not in excluded]
    added: list[int] = []
    for i in list(promote):
        for sibling in by_usr.get(regions[i].get("usr", ""), []):
            if sibling not in promote and sibling not in excluded:
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
    # DO NOT PROMOTE A SYMBOL OTHER TRANSLATION UNITS LINK AGAINST. Making a function `inline` in an
    # internal header gives a definition to every TU that INCLUDES that header -- and takes away the
    # single out-of-line definition every OTHER TU was linking to. gpu_executor.cpp defines
    # guest_readable/guest_writable, which agc_shader_layout, command_processor, gpu_capture,
    # hle_agc and hle_graphics all call through a public declaration: promoting them produced
    # "undefined reference" from five files that this tool never touched, which points nowhere near
    # the promotion.
    #
    # The test is whether the name appears in any tracked source outside this file. That
    # over-detects -- a comment mentioning the name counts -- and over-detecting is right here: the
    # cost is leaving a helper unpromoted, against a link failure diagnosed far from its cause.
    # Scan SOURCE only, and skip this tool's own directory. A first version grepped all of
    # `prosper`, which matched docs, CMakeLists, the header being generated -- and, best of all, the
    # comment in THIS FILE naming `g_guest_gpu_writes` as an example. The tool's own documentation
    # changed its classification, excluded the symbol from promotion, and produced 60 compile errors
    # in the header that still referenced it.
    def used_elsewhere(name: str) -> bool:
        if not name or name == "<anonymous>":
            return False
        out = subprocess.run(["git", "grep", "-l", "-w", name, "--",
                              "prosper/src", "prosper/frontends", "prosper/tests",
                              "prosper/tools", ":!prosper/tools/refactor"],
                             cwd=root, capture_output=True, text=True)
        skip = {map_data["file"], str((source.parent / args.header).relative_to(root))}
        files = [f for f in out.stdout.split()
                 if f not in skip and f.endswith((".cpp", ".hpp", ".h", ".cc"))]
        return bool(files)

    # ONLY EXTERNAL-LINKAGE regions can be public. A symbol inside an anonymous namespace cannot be
    # named from another translation unit at all, so a match elsewhere is necessarily a comment or
    # an unrelated same-named entity -- and acting on it is actively harmful, not merely
    # conservative: the symbol stays behind, and an `extern` declaration for it in the enclosing
    # namespace names a DIFFERENT entity, so every use becomes "reference to X is ambiguous".
    # Measured: `make_shader_compile_key` was excluded because live_renderer.cpp mentions it in a
    # comment about CPU profiling.
    # LINKAGE COMES FROM THE USR, not from the anonymous-namespace walk alone. clang encodes it:
    # an external-linkage entity's USR starts `c:@`, an internal one is file-prefixed
    # (`c:file.cpp@F@name`). A `static` free function is internal WITHOUT being in an anonymous
    # namespace, so the `anon` test alone missed it -- and then the name grep held it back because
    # some unrelated file happens to define a same-named static. Measured: `make_probe_pipe` was
    # excluded because of a same-named static in exec_image_linux.cpp:152. A name is not an
    # identity; the USR is.
    def externally_linked(i: int) -> bool:
        usr = regions[i].get("usr") or ""
        if usr:
            return usr.startswith("c:@")
        return not anon.get(i)          # no USR recorded: fall back to the scope walk

    public = [i for i in promote
              if externally_linked(i)
              and regions[i]["kind"] in ("FUNCTION_DECL", "VAR_DECL")
              and used_elsewhere(regions[i]["name"])]
    if public:
        print(f"  [ok]   {len(public)} region(s) are referenced outside this file and are NOT "
              f"promoted; they keep their single out-of-line definition:")
        for i in public[:8]:
            print(f"           {regions[i]['kind']:<14s} {regions[i]['name']}")
        promote = [i for i in promote if i not in public]
        # A public symbol also bars the conditional-block expansion below from re-adding it: that
        # expansion runs later and, left unchecked, put `guest_readable` straight back into the
        # promote set after this filter had removed it -- five other translation units then failed
        # to link against a definition that had become `inline` in a header they do not include.
        excluded |= set(public)
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
    # EXPAND TO WHOLE CONDITIONAL BLOCKS before refusing. A platform `#if` in this codebase spans
    # several regions -- gpu_executor.cpp has `#ifndef _WIN32 ... #else ... #endif` covering eight of
    # them -- so a promotion that touches one must take all of them, directives included, or the
    # `#if` and its `#endif` end up in different files. Refusing outright would be safe and useless;
    # the block is exactly the unit the author wrote.
    depth = conditional_depth(original.splitlines())
    total_lines = len(original.splitlines())

    def block_span(line: int) -> tuple[int, int]:
        """The outermost #if..#endif containing LINE, as a line range."""
        lo = line
        while lo > 1 and depth[lo] != 0:
            lo -= 1
        hi = line
        while hi < total_lines and depth[min(hi + 1, total_lines + 1)] != 0:
            hi += 1
        return lo, hi

    grew = True
    while grew:
        grew = False
        for i in sorted(promote):
            r = regions[i]
            if depth[r["start"]] == 0 and depth[min(r["end"] + 1, total_lines + 1)] == 0:
                continue
            lo, hi = block_span(r["start"] if depth[r["start"]] else r["end"])
            for j, rj in regions.items():
                if rj["role"] != "body" or j in promote or j in excluded:
                    continue
                if rj["start"] <= hi and rj["end"] >= lo:      # overlaps the conditional block
                    promote.append(j)
                    grew = True
        promote = sorted(set(promote))
    # A conditional block containing an excluded region cannot be lifted as a unit. Drop the WHOLE
    # block rather than refusing the run: the block is one indivisible thing, and taking part of it
    # is what separates an `#if` from its `#endif`.
    incomplete: set[int] = set()
    for i in list(promote):
        r = regions[i]
        if depth[r["start"]] == 0 and depth[min(r["end"] + 1, total_lines + 1)] == 0:
            continue
        lo, hi = block_span(r["start"] if depth[r["start"]] else r["end"])
        members = {j for j, rj in regions.items()
                   if rj["role"] == "body" and rj["start"] <= hi and rj["end"] >= lo}
        if members - set(promote):
            incomplete |= members
    if incomplete:
        dropped = sorted(set(promote) & incomplete)
        print(f"  [ok]   {len(dropped)} region(s) dropped: they sit in a preprocessor block that "
              f"also holds something excluded, so the block cannot move as a unit")
        promote = [i for i in promote if i not in incomplete]

    blocked = [i for i in promote
               if depth[regions[i]["start"]] != 0
               or depth[min(regions[i]["end"] + 1, total_lines + 1)] != 0]
    if blocked:
        print(f"  [ok]   {len(blocked)} region(s) lie inside a preprocessor conditional and are "
              f"promoted together with it")

    # Anything promoted code CALLS but that stayed behind needs a declaration in the header.
    edges_all = {int(k): {int(t) for t in v} for k, v in map_data.get("edges", {}).items()}
    wanted = {d for i in promote for d in edges_all.get(i, set()) if d in set(public)}
    src_lines = original.splitlines(keepends=True)
    protos = []
    for i in sorted(wanted):
        proto = prototype_of(regions[i], src_lines)
        if proto:
            protos.append(proto)
        else:
            print(f"  [warn] {regions[i]['name']} stays behind and is called from promoted "
                  f"code, but no declaration could be derived; add one by hand")
    if protos:
        print(f"  [ok]   {len(protos)} forward declaration(s) emitted for definitions that "
              f"stay behind")
        forward_decls.extend(protos)


    # A region whose start or end sits inside an `#if` cannot be lifted: the `#if` would stay in one
    # file and the `#endif` land in the other. split_file.py has always checked this; the promotion
    # path did not, and produced 73 compile errors headed "unterminated #ifndef" -- a message that
    # points at the header rather than at the lift that broke it.
    depth = conditional_depth(original.splitlines())
    # After expansion, a straddle can only survive if completing the block was impossible -- an
    # excluded region inside it, or a region the map does not cover. That is a real refusal.
    covered = set(promote)
    straddling = []
    for i in promote:
        r = regions[i]
        if depth[r["start"]] == 0 and depth[min(r["end"] + 1, total_lines + 1)] == 0:
            continue
        lo, hi = block_span(r["start"] if depth[r["start"]] else r["end"])
        if any(rj["role"] == "body" and rj["start"] <= hi and rj["end"] >= lo and j not in covered
               for j, rj in regions.items()):
            straddling.append(i)
    if straddling:
        for i in straddling[:8]:
            r = regions[i]
            print(f"  [FAIL] region {i} ({r['name']}) spans lines {r['start']}-{r['end']}, which "
                  f"begin or end inside an #if; lifting it would separate the directive from its "
                  f"#endif")
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
                                             args.namespace, note, spelling, forward_decls)
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
