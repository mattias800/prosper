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
  ... --clusters [N]  # suggest seams: greedily group regions that reference each other
                      #   into N parts (default 6), reporting the cut cost
  ... --json out.json # machine-readable, for split_file.py

Flags come from the build's compile_commands.json, so the parse sees exactly what the compiler sees.
"""

import argparse
import bisect
import json
import pathlib
import re
import shlex
import subprocess
import sys
from collections import defaultdict

# libclang is imported LAZILY. Everything that parses needs it, and `propose_clusters` -- which
# decides where a split should cut -- does not: it is arithmetic over the reference matrix. Keeping
# the import out of module scope is what lets `--selftest` pin that arithmetic on a machine with no
# libclang, which is the shape of this project's CI. `survey_sizes.py` resolves it the same way and
# for the same reason.
ci = None


def _need_clang() -> None:
    global ci
    if ci is not None:
        return
    import clang.cindex as _ci
    # The bindings look for a bare `libclang.so`, which the runtime package does not ship; the
    # versioned soname is what is actually installed. Fail loudly rather than letting a later call
    # die obscurely.
    for cand in ("/usr/lib64/libclang.so.22.1", "/usr/lib64/libclang.so"):
        if pathlib.Path(cand).exists():
            _ci.Config.set_library_file(cand)
            break
    ci = _ci                              # last: this is the guard the early return above tests


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

# A directive that CLOSES or UNDOES something, so it belongs to the declaration BEFORE it.
CLOSING_DIRECTIVE = re.compile(r"^\s*#\s*(endif|undef)\b")
# A directive that OPENS something, so it belongs to whatever follows. Absorbing past one of these
# would drag the next declaration's guard into the previous region.
OPENING_DIRECTIVE = re.compile(r"^\s*#\s*(if|ifdef|ifndef|define|include|else|elif|pragma)\b")


def absorb_trailing_directives(regions: list[dict], lines: list[str]) -> int:
    """Move a `#endif`/`#undef` that trails a declaration back into ITS region, not the next one.

    A cursor's extent stops at its last token, so a block written as

        #ifndef _WIN32
        ... declarations ...          <- the last one's cursor ends HERE
        #undef HELPER
        #endif
        void next_thing() {}          <- a different library's function

    leaves `#undef`/`#endif` prefixing the NEXT region, inseparable from `next_thing`. split_file.py
    then correctly refuses to put a guard and its `#endif` in different files, so no region-granular
    partition can cut between those two libraries at all. Measured on hle_service.cpp: the AvPlayer
    block's `#endif` sat in `s_gamepresets`' region, and every library in that file ends this way.

    RUN AS A POST-PASS, NOT DURING TILING, and that is the whole subtlety. Done inside the emitter
    this absorbs greedily, including a `#endif` whose natural home is the namespace CLOSE region that
    follows it -- which is replicated into every output, whereas a body region is not. Pulling it out
    of the close and into a body means the outputs that did not receive that body get the `#if`
    (replicated, in the open region) with no `#endif`. That is a strictly worse break than the one
    being fixed, and it is invisible until the compiler sees it. Absorption therefore happens only
    when the FOLLOWING region is another body, which needs the finished list.

    Tiling is preserved: whatever one region gains, the next loses. Returns how many it moved."""
    moved = 0
    for i, r in enumerate(regions[:-1]):
        nxt = regions[i + 1]
        # Absorb into a body, from anything except a namespace CLOSE. A close region's opening
        # directive lives in the matching OPEN region, which is replicated alongside it, so an
        # `#endif` there is already paired in every output -- pulling it into a body breaks exactly
        # that pairing. An OPEN region has no such claim: it spans the text before a namespace's
        # first member, and a leading `#endif` there closes something that came before it.
        if r["role"] != "body" or nxt["role"] == "close":
            continue
        last, j = r["end"], r["end"] + 1
        while j < nxt["end"]:            # never consume the next region whole
            line = lines[j - 1]
            if CLOSING_DIRECTIVE.match(line):
                last = j
            elif line.strip() == "":
                pass                     # blanks may sit between a block and its `#endif`
            elif OPENING_DIRECTIVE.match(line):
                break                    # belongs to what comes after
            else:
                break                    # a declaration or a comment: not ours
            j += 1
        if last > r["end"]:
            moved += last - r["end"]
            r["end"] = last
            nxt["start"] = last + 1
    return moved


def regions_of(tu, target: pathlib.Path, total_lines: int, lines: list[str] | None = None,
               max_depth: int = 3) -> list[dict]:
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
    if lines is not None:
        absorb_trailing_directives(regions, lines)

    first_open = next((r["index"] for r in regions if r["role"] == "open"), None)
    if first_open is not None:
        for r in regions[:first_open]:
            r["role"] = "preamble"
    return regions


def usrs_declared_outside(tu, target: pathlib.Path) -> set[str]:
    """USRs that some OTHER file in this translation unit also declares -- i.e. that a header
    carries.

    This is the difference between "links" and "compiles", and it is not the same question as
    linkage. A function defined in a .cpp inside `namespace prosper::gpu`, never declared in a
    header, has EXTERNAL linkage: clang's USR starts `c:@`, and the symbol is visible to the
    linker. Move its callers to a second .cpp and the program still links -- and does not compile,
    because no declaration is in scope.

    Measured: splitting rdna2_to_spirv.cpp per shader stage passed a linkage-based cross-part check
    and then failed on `safe_execz_branches` in three of the five outputs. Its sibling
    `safe_execz_branches_for_test` IS declared in rdna2_to_spirv.hpp; it is not. Nothing about the
    linkage of the two differs.

    So a definition may be left in another part only if some header declares it, which is exactly
    "this USR appears in a file other than the one being split".
    """
    out: set[str] = set()
    tgt = target.resolve()

    def walk(c) -> None:
        f = c.location.file
        if f is not None and pathlib.Path(f.name).resolve() != tgt:
            u = c.get_usr()
            if u:
                out.add(u)
        for k in c.get_children():
            walk(k)

    walk(tu.cursor)
    return out


# USRs libclang hands to entities that have no name to be referenced BY. `c:` is the one it gives a
# `static_assert`, an `extern "C"` linkage spec, and other anonymous cursors -- all of them, so they
# collide on a single key and the reference graph links every one of them to every other.
#
# The result is not noise, it is a confident wrong answer: on hle_service.cpp a `static_assert` about
# `VdecswConfig` appeared to be referenced 16 times from the libSceAvPlayer block, and split_file.py
# refused a correct partition because that "definition cannot be reached". An entity with no name
# cannot be referenced by name, so an edge to one is spurious by construction.
UNREFERENCEABLE_USRS = frozenset({"c:"})


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
            if not usr or usr in UNREFERENCEABLE_USRS:
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
            # A NAMESPACE cursor is not a reference to anything a split has to keep together. Every
            # re-opening `namespace {` and `namespace prosper {` resolves `.referenced` to the
            # namespace declaration, so the brace itself became an edge -- attributed to whichever
            # body region happened to hold the first cursor carrying that USR. Measured on
            # hle_kernel.cpp: 19 of 19 edges whose source is a replicated region were this, 13 of
            # them pointing at `fbsd_errno`, and acting on them refused every legal two-part plan
            # for the file. They are noise in every consumer, not only the splitter.
            if ch.kind == ci.CursorKind.NAMESPACE or ref.kind == ci.CursorKind.NAMESPACE:
                continue
            usr = ref.get_usr()
            if usr not in owner:
                continue
            src = region_of(ch.location.line)
            dst = owner[usr]
            if src is None or src == dst:
                continue
            edges[src][dst] += 1

    # TEMPLATES NEED A TEXTUAL FALLBACK, and this is not a nicety. A call to a function template
    # from a dependent context has no resolved `referenced` cursor, so the AST walk above cannot see
    # it -- the edge simply is not there. Twice in one file that produced a "complete" dependency
    # closure that did not compile, each time discovered only when the header referenced a name
    # nothing declared (`for_each_scalar_write`, then `must_fact_at`).
    #
    # So for template regions only, add an edge from any region whose text mentions the name as a
    # whole word. That over-connects -- a mention in a comment counts -- and over-connecting is the
    # safe direction here: a spurious edge drags an extra helper along, while a missing one produces
    # a broken split.
    lines = target.read_text(errors="ignore").splitlines(keepends=True)
    templates = [r for r in regions if r["kind"] == "FUNCTION_TEMPLATE" and r["name"]]
    for tmpl in templates:
        word = re.compile(r"\b" + re.escape(tmpl["name"]) + r"\b")
        for r in regions:
            if r["index"] == tmpl["index"] or r["role"] != "body":
                continue
            text = "".join(lines[r["start"] - 1:r["end"]])
            if word.search(text) and tmpl["index"] not in edges.get(r["index"], {}):
                edges[r["index"]][tmpl["index"]] += 1
    return {k: dict(v) for k, v in edges.items()}


def propose_clusters(regions, edges, parts: int, cap_lines: int | None = None) -> tuple[list[list[int]], dict]:
    """Group body regions into PARTS candidate output files, cutting where the references are thin.

    Greedy agglomeration over the reference matrix, heaviest edge first, which is what the file's
    REFERENCES section is for: an internal-linkage helper separated from its callers does not fail
    at review, it fails at link. Grouping on that matrix puts them on the same side by construction.

    The size cap is reported rather than hidden, because it is the part that lies. Two groups landing
    on the same line count means the CAP stopped the merge, not that a seam is there -- so the
    summary marks which boundaries are structural and which are the cap talking. Grouping by NAME
    does not find these: on gpu_executor.cpp a name-prefix pass put 183 of 273 regions in "other".

    CAP AND PART COUNT ARE SEPARATE KNOBS, and conflating them made the tool answer a question
    nobody asked. Deriving `cap = total // parts` asserts that every output should be the same size,
    which is true of a balanced partition and false of a SEMANTIC one: `hle_service.cpp` divides
    along the Sony library each block reimplements, and those libraries are genuinely unequal --
    libSceAvPlayer is an order of magnitude more code than libScePsml. Asking that file for 12 parts
    derived a 532-line cap, refused 583 merges the references wanted, and returned 96 groups with
    39% of references cut: a report about the cap, not about the file. Pass `cap_lines` to say how
    big an output may get and let the part count fall out of the structure.
    """
    body = {r["index"]: r for r in regions if r["role"] == "body"}
    size = {i: r["end"] - r["start"] + 1 for i, r in body.items()}
    total = sum(size.values())
    cap = max(1, cap_lines) if cap_lines else max(1, total // max(1, parts))

    w: dict[tuple[int, int], int] = {}
    for a, ds in edges.items():
        for b, n in ds.items():
            if a in body and b in body and a != b:
                key = (min(a, b), max(a, b))
                w[key] = w.get(key, 0) + n

    parent = {i: i for i in body}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    csize = dict(size)
    capped = 0
    for (a, b), _ in sorted(w.items(), key=lambda kv: -kv[1]):
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        if csize[ra] + csize[rb] > cap:
            capped += 1               # a merge the references wanted and the cap refused
            continue
        parent[rb] = ra
        csize[ra] += csize[rb]

    groups: dict[int, list[int]] = {}
    for i in body:
        groups.setdefault(find(i), []).append(i)

    # SECOND PASS: attach the tail. A cap-driven greedy leaves a long run of singletons -- 30 of 41
    # on gpu_capture.cpp -- and a proposal the caller then has to place by hand is not a proposal.
    # Merge the smallest group into whichever group it references most, ignoring the cap, until the
    # requested part count is reached. A region with no cross-references at all has nothing to
    # attach to and stays where it is; that is information, not a failure.
    def edge_between(g1, g2):
        return sum(w.get((min(a, b), max(a, b)), 0) for a in g1 for b in g2)

    # The cap binds here too, and that is the whole difference between a proposal and a lie. Without
    # it, "attach the smallest to whatever it references most" funnels every group into the largest
    # connected component: on this file it produced ONE 6,374-line group out of 6,515 and reported a
    # 0% cut -- a split that is not a split, wearing a perfect score. A 0% cut is a tell.
    # So attachment stops at the cap, the part count is whatever the structure and the cap allow,
    # and the caller is told when that is more than they asked for.
    while True:
        keys = sorted(groups, key=lambda k: sum(size[i] for i in groups[k]))
        if len(keys) <= parts:
            break
        moved = False
        for k in keys:
            ksz = sum(size[i] for i in groups[k])
            best, best_w = None, 0
            for other in keys:
                if other == k or sum(size[i] for i in groups[other]) + ksz > cap:
                    continue
                ww = edge_between(groups[k], groups[other])
                if ww > best_w:
                    best, best_w = other, ww
            if best is not None:
                groups[best] += groups[k]
                del groups[k]
                moved = True
                break
        if not moved:
            break            # nothing can grow without breaching the cap; report what there is

    out = sorted(groups.values(), key=lambda g: -sum(size[i] for i in g))
    gid = {i: n for n, g in enumerate(out) for i in g}
    cut = sum(n for (a, b), n in w.items() if gid[a] != gid[b])
    stats = {"cut": cut, "total_refs": sum(w.values()), "cap": cap, "capped_merges": capped,
             "singletons": sum(1 for g in out if len(g) == 1), "asked": parts,
             "cap_explicit": bool(cap_lines)}
    return out, stats


def print_clusters(regions, edges, parts: int, cap_lines: int | None = None) -> None:
    body = {r["index"]: r for r in regions if r["role"] == "body"}
    size = {i: r["end"] - r["start"] + 1 for i, r in body.items()}
    groups, st = propose_clusters(regions, edges, parts, cap_lines)
    share = st["cut"] / st["total_refs"] if st["total_refs"] else 0.0
    if st["cap_explicit"]:
        over = (f" -- cap {st['cap']}L was given explicitly, so the group COUNT is whatever the "
                f"structure yields" if len(groups) != st["asked"] else "")
    else:
        over = (f" -- you asked for {st['asked']}, and the reference structure plus that size cap do "
                f"not permit fewer (pass --cap-lines to set the cap independently)"
                if len(groups) > st["asked"] else "")
    print(f"\n== proposed seams: {len(groups)} group(s), target ~{st['cap']} lines each =={over}")
    print(f"   {st['cut']} of {st['total_refs']} references cross a boundary ({share:.0%}) -- that "
          f"is the promote-to-header list")
    print(f"   {st['capped_merges']} merge(s) the references wanted were refused by the size cap; "
          f"a group AT the cap is the cap talking, not a seam")
    print(f"   {st['singletons']} group(s) of one region\n")
    for n, g in enumerate(groups):
        lines = sum(size[i] for i in g)
        at_cap = " <- at the cap" if lines >= st["cap"] else ""
        tops = sorted(g, key=lambda i: -size[i])[:3]
        names = ", ".join(f"{body[i]['name']}({size[i]})" for i in tops)
        print(f"  [{n}] {len(g):>4} region(s) {lines:>6}L{at_cap}")
        print(f"       {names[:96]}")
        plan = " ".join(str(i) for i in sorted(g))
        # NOT truncated. This line exists to be copy-pasted into split_file.py; a silent cut at 300
        # characters drops every region past about the hundredth and yields a plan that is a valid
        # prefix of the right answer, which is the worst failure available here. split_file.py
        # catches an incomplete partition, but only after the caller has run it.
        print(f"       plan: {plan}")


def selftest() -> int:
    """Pin propose_clusters. Pure arithmetic over the reference matrix, so it runs without libclang.

    The clustering is the part that decides where a split cuts, and its failure mode is not a crash:
    it is a confident proposal that is wrong. Both arms below were live defects during development.
    """
    bad = 0

    def check(cond, label):
        nonlocal bad
        if not cond:
            print(f"  [FAIL] {label}")
            bad += 1

    # Two tight clusters, 100 lines each, joined by ONE weak edge. The seam is obvious by
    # construction, which is what makes it a usable oracle.
    regions = [{"index": 0, "start": 1, "end": 1, "role": "open", "name": "ns"}]
    line = 2
    for i in range(1, 5):
        regions.append({"index": i, "start": line, "end": line + 24, "role": "body",
                        "name": f"a{i}"})
        line += 25
    for i in range(5, 9):
        regions.append({"index": i, "start": line, "end": line + 24, "role": "body",
                        "name": f"b{i}"})
        line += 25
    edges = {1: {2: 50, 3: 50, 4: 50}, 2: {3: 50, 4: 50}, 3: {4: 50},
             5: {6: 50, 7: 50, 8: 50}, 6: {7: 50, 8: 50}, 7: {8: 50},
             4: {5: 1}}
    groups, st = propose_clusters(regions, edges, 2)
    check(len(groups) == 2, f"two clusters joined by one weak edge split in two (got {len(groups)})")
    check(all(len(g) == 4 for g in groups), f"and evenly (got {[len(g) for g in groups]})")
    check(st["cut"] == 1, f"cutting exactly the weak edge (got {st['cut']})")

    # THE CAP MUST BIND DURING ATTACHMENT. Without it, "attach the smallest group to whatever it
    # references most" funnels everything into the largest connected component: measured on
    # gpu_capture.cpp it produced ONE 6,374-line group out of 6,515 and reported a 0% cut -- a
    # split that is not a split, wearing a perfect score. A 0% cut over a non-trivial graph is a
    # tell, not a success, so it is asserted against directly.
    cap = st["cap"]
    check(all(sum(regions[i]["end"] - regions[i]["start"] + 1 for i in g) <= cap * 1.5
              for g in groups),
          "no group runs away past the size cap")
    # The fixture has to make the FIRST pass leave more groups than asked for, or the attachment
    # loop never runs and the arm certifies nothing -- checked by mutation, not assumed. Six regions
    # of 100 lines with parts=4 gives a cap of 150, so no two can merge (200 > 150): the greedy
    # leaves six, attachment is entered, and the cap is the only thing standing between six groups
    # and one.
    big = [{"index": 0, "start": 1, "end": 1, "role": "open", "name": "ns"}]
    ln = 2
    for i in range(1, 7):
        big.append({"index": i, "start": ln, "end": ln + 99, "role": "body", "name": f"c{i}"})
        ln += 100
    dense = {i: {j: 10 for j in range(1, 7) if j != i} for i in range(1, 7)}
    dgroups, dst = propose_clusters(big, dense, 4)
    check(len(dgroups) == 6,
          f"the size cap holds during attachment; without it everything funnels into the largest "
          f"connected component (got {len(dgroups)} group(s))")
    check(dst["cut"] > 0, "and a 0% cut is not manufactured by merging everything")

    # --cap-lines must OVERRIDE the derived cap, in both directions, on the same fixture. Testing
    # only the loosening direction would pass for an implementation that ignored the flag whenever
    # it was smaller than total/parts, which is the half a semantic split needs least.
    wide, wst = propose_clusters(big, dense, 4, cap_lines=650)
    check(len(wide) == 1 and wst["cap"] == 650,
          f"--cap-lines 650 lets the whole 600-line fixture become one group "
          f"(got {len(wide)} group(s), cap {wst['cap']})")
    narrow, nst = propose_clusters(big, dense, 1, cap_lines=100)
    check(len(narrow) == 6 and nst["cap"] == 100,
          f"--cap-lines 100 keeps six groups even though --clusters 1 would derive a 600-line cap "
          f"(got {len(narrow)} group(s), cap {nst['cap']})")
    check(wst["cap_explicit"] and not dst["cap_explicit"],
          "the report can tell an explicit cap from a derived one")

    # A region nothing references has nothing to attach to; it must survive rather than vanish.
    lone, _ = propose_clusters(regions, {1: {2: 5}}, 2)
    seen = [i for g in lone for i in g]
    # The SET and the COUNT, not just the count: a total of 8 is equally consistent with one region
    # duplicated and another dropped, which is precisely what a bad merge produces.
    check(sorted(seen) == list(range(1, 9)) and len(seen) == 8,
          f"every body region appears in exactly one group (got {sorted(seen)})")

    # absorb_trailing_directives moves a trailing `#endif`/`#undef` back into the region it closes.
    # Pure arithmetic over a region list, so it runs with no libclang. The NEGATIVE arms matter most:
    # absorbing one line too many drags the next declaration's own guard backwards, and absorbing
    # into a body region what belongs to a replicated namespace CLOSE leaves other outputs with an
    # `#if` and no `#endif` -- a strictly worse break than the one being fixed.
    def R(i, start, end, role="body"):
        return {"index": i, "start": start, "end": end, "role": role}

    # The AvPlayer shape: declaration, trailing #undef + #endif, then a different library.
    L = ["decl_a();", "#undef HELPER", "#endif", "void next() {}"]
    regs = [R(0, 1, 1), R(1, 2, 4)]
    check(absorb_trailing_directives(regs, L) == 2 and regs[0]["end"] == 3 and regs[1]["start"] == 4,
          f"absorbs a trailing #undef + #endif (got end={regs[0]['end']}, next start={regs[1]['start']})")

    # Blank line between the block and its #endif.
    L2 = ["decl_a();", "", "#endif", "void next() {}"]
    regs2 = [R(0, 1, 1), R(1, 2, 4)]
    absorb_trailing_directives(regs2, L2)
    check(regs2[0]["end"] == 3, f"absorbs across a blank line (got {regs2[0]['end']})")

    # STOPS at an opening directive: that #endif closes the NEXT declaration's conditional.
    L3 = ["decl_a();", "#ifdef X", "void next() {}", "#endif"]
    regs3 = [R(0, 1, 1), R(1, 2, 4)]
    check(absorb_trailing_directives(regs3, L3) == 0 and regs3[0]["end"] == 1,
          "STOPS at an opening directive")

    # STOPS at a declaration and at a comment (a leading comment documents what FOLLOWS it).
    L4 = ["decl_a();", "void next() {}", "#endif"]
    regs4 = [R(0, 1, 1), R(1, 2, 3)]
    check(absorb_trailing_directives(regs4, L4) == 0, "STOPS at a declaration")
    L5 = ["decl_a();", "// a comment about next()", "#endif"]
    regs5 = [R(0, 1, 1), R(1, 2, 3)]
    check(absorb_trailing_directives(regs5, L5) == 0, "STOPS at a comment")

    # THE REGRESSION ARM. The #endif belongs to the replicated namespace CLOSE that follows, not to
    # the body before it. Absorbing here is what left avplayer.cpp with an unterminated #ifndef.
    L6 = ["namespace {", "#ifndef _WIN32", "struct S {", "};", "#endif", "}"]
    regs6 = [R(0, 1, 2, "open"), R(1, 3, 4), R(2, 5, 6, "close")]
    check(absorb_trailing_directives(regs6, L6) == 0 and regs6[1]["end"] == 4,
          f"does NOT absorb into a body what belongs to a replicated close (got {regs6[1]['end']})")

    # Absorbing INTO a body FROM a replicated open is right: the open's job starts at `namespace {`,
    # and a leading `#endif` there closes a conditional that ended with the previous declaration.
    L8 = ["decl_a();", "#endif", "", "namespace {", "int x;"]
    regs8 = [R(0, 1, 1), R(1, 2, 4, "open"), R(2, 5, 5)]
    absorb_trailing_directives(regs8, L8)
    check(regs8[0]["end"] == 2 and regs8[1]["start"] == 3,
          f"absorbs from a replicated OPEN (got end={regs8[0]['end']}, open start={regs8[1]['start']})")

    # A run of closing directives between two bodies is absorbed whole.
    L7 = ["decl_a();", "#endif", "#endif", "void next() {}"]
    regs7 = [R(0, 1, 1), R(1, 2, 4)]
    absorb_trailing_directives(regs7, L7)
    check(regs7[0]["end"] == 3, f"absorbs a run of closing directives (got {regs7[0]['end']})")

    # An entity with no name cannot be referenced BY name, so the graph must never own a USR that
    # every anonymous cursor shares. Keyed on the exact value libclang produces, because the failure
    # is a confident wrong edge rather than a crash: it made split_file.py refuse a correct plan.
    check("c:" in UNREFERENCEABLE_USRS, "the shared anonymous USR is excluded from the graph")
    check(not ("c:@F@real_function" in UNREFERENCEABLE_USRS),
          "a real USR is not excluded by a prefix match")

    print("== PASS ==" if not bad else f"== FAIL: {bad} ==")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", type=pathlib.Path)
    ap.add_argument("--selftest", action="store_true",
                    help="pin the clustering arithmetic; needs no libclang and no build")
    ap.add_argument("--build", default="prosper/build-linux")
    ap.add_argument("--json", type=pathlib.Path)
    ap.add_argument("--clusters", nargs="?", type=int, const=6, default=None,
                    metavar="PARTS",
                    help="suggest seams: group regions that reference each other (default 6 parts)")
    ap.add_argument("--cap-lines", type=int, default=None, metavar="N",
                    help="largest output a group may reach, INDEPENDENT of --clusters. Without it "
                         "the cap is total/PARTS, which assumes every output should be the same "
                         "size -- false for a split along semantic lines, where the parts are "
                         "genuinely unequal")
    ap.add_argument("--min-lines", type=int, default=0,
                    help="only print regions at least this many lines long")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.file:
        sys.exit("--file is required")

    _need_clang()
    root = repo_root()
    target = (root / args.file) if not args.file.is_absolute() else args.file
    if not target.exists():
        sys.exit(f"no such file: {target}")
    db = root / args.build / "compile_commands.json"
    if not db.exists():
        sys.exit(f"no compile_commands.json at {db}; configure the build first")

    flags, parse_file = flags_for(target, db)
    flags += builtin_includes()
    source_lines = target.read_text().splitlines()
    total_lines = len(source_lines)

    index = ci.Index.create()
    tu = index.parse(str(parse_file), args=flags,
                     options=ci.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD)
    fatal = [d for d in tu.diagnostics if d.severity >= ci.Diagnostic.Error]
    if fatal:
        print(f"  WARNING {len(fatal)} parse error(s); the map may be incomplete")
        for d in fatal[:3]:
            print(f"    {d.spelling}")

    regions = regions_of(tu, target, total_lines, source_lines)
    edges = cross_refs(tu, target, regions)
    # Whether a header also declares each region's symbol. See usrs_declared_outside: this is what
    # decides if a definition can be left in another part, and linkage is not a substitute for it.
    declared = usrs_declared_outside(tu, target)
    for r in regions:
        r["declared_in_header"] = bool(r.get("usr") and r["usr"] in declared)

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

    if args.clusters is not None:
        print_clusters(regions, edges, args.clusters, args.cap_lines)

    if args.json:
        # The digest is what lets split_file.py prove the map still describes the file. Without it
        # the only available check compares two values derived from the same read, which can only
        # fail if the file GREW -- an assertion that cannot fail on an edit is not a check.
        import hashlib
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        # RECORD THE PARSE ERRORS IN THE MAP. The warning above goes to a terminal nobody keeps,
        # and the map outlives it -- so a consumer reading this file has no way to tell a complete
        # map from one built on a partial AST, and prints its own reassuring [ok] either way. A
        # libclang parse driven from a g++ database fails softly, so this is not a rare case.
        args.json.write_text(json.dumps({"file": str(target.relative_to(root)),
                                         "sha256": digest,
                                         "total_lines": total_lines,
                                         "parse_errors": len(fatal),
                                         "regions": regions,
                                         "edges": {str(k): v for k, v in edges.items()}}, indent=1))
        print(f"  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
