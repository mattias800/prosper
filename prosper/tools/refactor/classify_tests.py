#!/usr/bin/env python3
"""classify_tests.py -- propose a folder for each test, from what the test is built against.

prosper/tests held 228 files in one directory. Sorting them by hand is 228 judgement calls made from
filenames, and a filename is a weak signal: test_gpu_capture and test_gpu_dependency_graph share a
prefix and exercise different subsystems, while test_mb3_freelist and test_dmem_topology share none
and both exercise guest memory.

A test's `#include` lines are a far better signal, because they are what it is actually compiled
against, and every project header now carries its subsystem and folder in its canonical path
(`gpu/recompiler/rdna2_to_spirv.hpp`). So the includes name the destination directly.

WEIGHTING. Counting includes puts 76 of 228 tests in hle/dispatch -- not because they are about the
dispatcher, but because every HLE test includes dispatch.hpp to register a stub. A header appearing
in 76 tests separates nothing; one appearing in two says almost everything. Each include therefore
contributes log(N / how many tests include it), summed PER HEADER over the folder it belongs to.
Per header is the whole point: a test including six distinct pm4 headers is six pieces of evidence
for pm4, and an earlier version of this file keyed the evidence set on the (subsystem, folder) pair,
which collapsed those six into one and quietly turned the rule into "the rarest single folder wins".
That is a different algorithm, it is worse, and it read exactly the same from the outside.

It PROPOSES. A test whose two best folders are close has a real ambiguity, and the tool says so
rather than picking quietly. Where ALL of a test's evidence is ubiquitous, it reports the subsystem
rather than a folder, because that is what the evidence supports.

  python3 prosper/tools/refactor/classify_tests.py --selftest    # run this first
  python3 prosper/tools/refactor/classify_tests.py
  python3 prosper/tools/refactor/classify_tests.py --plan out.txt
"""

import argparse
import collections
import math
import pathlib
import re
import subprocess
import sys

PREFIXES = ("gpu", "hle", "host", "shared", "self", "loader")
# A folder-qualified project include: subsystem/folder/header.
INC = re.compile(r'#include\s+"(' + "|".join(PREFIXES) + r')/([^/"]+)/([^"]+)"')
# A subsystem that is still flat (src/self, src/loader) has no folder component. Classified at
# depth 1; an earlier version added these to the evidence set without adding them to the frequency
# table, so any test reaching this branch raised KeyError.
FLAT = re.compile(r'#include\s+"(' + "|".join(PREFIXES) + r')/([^/"]+)"')
BARE = re.compile(r'#include\s+"([^"/]+)"')
CANON = re.compile(r'#include\s+"(?:fixtures/)?([^"/]+)"')

RUNNERS = {"render_runner.h", "compute_runner.h", "image_compute_runner.h"}
NAMED_FIXTURES = RUNNERS | {"spirv_compute.h", "spirv_triangle.h", "synth_prx.h", "handmade_prx.h",
                            "at9_testvec.h", "test_scratch.h", "test_data.h"}


def evidence(text: str) -> collections.Counter:
    """(subsystem, folder, header) -> how many times this text includes it.

    Keyed on the HEADER, so distinct headers of one folder are distinct evidence.
    """
    out: collections.Counter = collections.Counter()
    for sub, folder, header in INC.findall(text):
        out[(sub, folder, header)] += 1
    for sub, header in FLAT.findall(text):
        out[(sub, "", header)] += 1
    return out


def is_fixture(name: str) -> bool:
    """A fixture is what tests are built ON, not a test OF anything."""
    return (name in NAMED_FIXTURES or name.endswith("_fixture.hpp")
            or (name.endswith((".h", ".hpp")) and not name.startswith("test_")))


def classify(texts: dict[str, str], depth: int = 2, rival_ratio: float = 0.85):
    """Returns (assignment, ambiguous, unclassified). Pure, so --selftest can drive it."""
    df: collections.Counter = collections.Counter()
    for name, text in texts.items():
        if is_fixture(name):
            continue
        df.update(evidence(text).keys())
    total = max(len(texts), 1)
    idf = {k: math.log(total / v) for k, v in df.items()}
    # Both conditions matter. The fraction is the real rule; the absolute floor stops a small
    # corpus from declaring everything ubiquitous, which on a four-file input made every test fall
    # back to its subsystem and looked exactly like a deliberate decision.
    ubiquitous = {k for k, v in df.items() if v > 0.20 * total and v >= 5}

    assignment: dict[str, str] = {}
    ambiguous: list[tuple[str, list[tuple[str, float]]]] = []
    unclassified: list[str] = []

    for name in sorted(texts):
        text = texts[name]
        stem = name.rsplit(".", 1)[0]
        if is_fixture(name):
            assignment[stem] = "fixtures"
            continue
        ev = evidence(text)
        if not ev:
            # No project include at all. If it drives one of the Vulkan/compute harnesses it is an
            # execution test, a real category. The include may be spelled bare (before this tree was
            # foldered) or `fixtures/...` (after), so both are matched -- an earlier version matched
            # only the bare form, which this project's own canonicalisation had already removed.
            if RUNNERS & set(CANON.findall(text)):
                assignment[stem] = "render"
            else:
                unclassified.append(name)
            continue

        score: dict[tuple[str, str], float] = collections.defaultdict(float)
        for key, times in ev.items():
            sub, folder, _header = key
            score[(sub, folder)] += idf[key] * times
        ranked = sorted(score.items(), key=lambda kv: -kv[1])
        (sub, folder), top = ranked[0]
        rivals = [(f"{s}/{f}" if f else s, round(n, 2)) for (s, f), n in ranked
                  if n >= top * rival_ratio]
        if len(rivals) > 1:
            ambiguous.append((name, rivals))
            subs = {s for (s, _f), n in ranked if n >= top * rival_ratio}
            assignment[stem] = sub if len(subs) == 1 else "misc"
            continue
        if all(k in ubiquitous for k in ev):
            assignment[stem] = sub
            continue
        assignment[stem] = f"{sub}/{folder}" if (depth >= 2 and folder) else sub
    return assignment, ambiguous, unclassified


# --- self-test ----------------------------------------------------------------------------------
# Every defect this file has had was silent: a non-recursive glob that reported 3 of 215 files as a
# clean result, an evidence set that collapsed six headers into one, a branch that raised KeyError,
# and a branch made dead by an include spelling this very restructure introduced. None of them could
# fail loudly, so none of them was noticed until a reviewer executed the tool. Hence: cases that
# assert the OUTCOME, and cases that assert what must NOT happen.

def _corpus(spec: dict[str, str], size: int = 20) -> dict[str, str]:
    """SPEC plus filler, so frequencies are realistic.

    Both of the first self-test cases failed on a four-file corpus for reasons that had nothing to do
    with the rule under test -- everything was "ubiquitous" at 20% of four, and log(1/1) is zero, so
    every score tied and every case came back ambiguous. A rule about relative frequency cannot be
    tested on a corpus too small to have frequencies.
    """
    files = dict(spec)
    for i in range(len(files), size):
        files[f"pad_x{i}.cpp"] = f'#include "hle/kernel/k{i}.hpp"\n'
    return files


PM4 = _corpus({
    # test_a includes THREE pm4 headers (each held by 5 files, so individually unremarkable) and one
    # timeline header (held by 2, so rarer). Per-header evidence gives pm4 3 x 1.386 = 4.16 against
    # timeline's 2.30 and pm4 wins. The collapsed keying this file used to have scores pm4 as a
    # single 1.386 term and picks TIMELINE -- so this arm distinguishes the two algorithms rather
    # than merely exercising one.
    "test_a.cpp": '#include "gpu/pm4/a.hpp"\n#include "gpu/pm4/b.hpp"\n'
                  '#include "gpu/pm4/c.hpp"\n#include "gpu/timeline/z.hpp"\n',
    **{f"pad_a{i}.cpp": '#include "gpu/pm4/a.hpp"\n' for i in range(4)},
    **{f"pad_b{i}.cpp": '#include "gpu/pm4/b.hpp"\n' for i in range(4)},
    **{f"pad_c{i}.cpp": '#include "gpu/pm4/c.hpp"\n' for i in range(4)},
    "pad_z.cpp": '#include "gpu/timeline/z.hpp"\n',
})

FLAT_CASE = _corpus({
    "test_flat.cpp": '#include "self/module.hpp"\n#include "self/other.hpp"\n'
                     '#include "loader/linker.hpp"\n',
    "pad_m.cpp": '#include "self/module.hpp"\n',
    "pad_o.cpp": '#include "self/other.hpp"\n',
    "pad_l.cpp": '#include "loader/linker.hpp"\n',
})

SELFTESTS = [
    ("per-header evidence beats a single rarer folder", PM4, {"test_a": "gpu/pm4"}),
    ("a flat subsystem classifies at depth 1 rather than raising KeyError",
     FLAT_CASE, {"test_flat": "self"}),
    ("a bare runner include is an execution test",
     _corpus({"test_r.cpp": '#include "render_runner.h"\n'}), {"test_r": "render"}),
    ("a canonicalised runner include is still an execution test",
     _corpus({"test_r2.cpp": '#include "fixtures/render_runner.h"\n'}), {"test_r2": "render"}),
    ("a fixture is never classified as a test of something",
     _corpus({"gta5_x_fixture.hpp": '#include "gpu/recompiler/a.hpp"\n',
              "compute_runner.h": '#include "gpu/execute/b.hpp"\n'}),
     {"gta5_x_fixture": "fixtures", "compute_runner": "fixtures"}),
]

NEGATIVE = [
    ("a ubiquitous header never decides a folder on its own",
     {f"test_{i}.cpp": '#include "hle/dispatch/dispatch.hpp"\n' for i in range(10)},
     lambda a: all(v == "hle" for v in a.values())),
    ("the rarest single folder does NOT win when the evidence is thin",
     PM4, lambda a: a.get("test_a") != "gpu/timeline"),
]


def selftest() -> int:
    bad = 0
    for name, texts, expected in SELFTESTS:
        got, _amb, _un = classify(texts)
        for stem, want in expected.items():
            if got.get(stem) != want:
                print(f"  [FAIL] {name}: {stem} -> {got.get(stem)!r}, expected {want!r}")
                bad += 1
    for name, texts, predicate in NEGATIVE:
        got, _amb, _un = classify(texts)
        if not predicate(got):
            print(f"  [FAIL] {name}: {got}")
            bad += 1
    if bad:
        print(f"  the classifier's own rules are broken; its output would be confident and wrong")
        return 1
    print(f"  [ok]   classifier self-test: {len(SELFTESTS)} placement case(s), "
          f"{len(NEGATIVE)} must-not case(s)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="prosper/tests")
    ap.add_argument("--plan", type=pathlib.Path)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if selftest():
        return 1

    root = pathlib.Path(subprocess.run(["git", "rev-parse", "--show-toplevel"],
                                       capture_output=True, text=True, check=True).stdout.strip())
    tdir = root / args.dir
    # RECURSIVE. A non-recursive glob over an already-foldered tree finds 21 of 215 files and
    # reports "3 classified" -- a confident, well-formed number that looks like a clean result. That
    # is the same shape as the os.listdir defect this restructure found in tools/re/hle_handler_map.py,
    # and it was live in this file at the same time.
    files = sorted(p for p in tdir.rglob("*")
                   if p.suffix in (".cpp", ".h", ".hpp") and p.is_file())
    if not files:
        sys.exit(f"refused: no test sources under {tdir}")
    texts = {p.name: p.read_text(errors="ignore") for p in files}
    if len(texts) != len(files):
        sys.exit(f"refused: {len(files) - len(texts)} duplicate basename(s) under {tdir}; the plan "
                 f"is keyed by module name and could not express them")

    assignment, ambiguous, unclassified = classify(texts, args.depth)

    counts = collections.Counter(assignment.values())
    print(f"== {len(files)} file(s) scanned: {len(assignment)} classified, "
          f"{len(unclassified)} unclassified, {len(ambiguous)} ambiguous ==")
    for folder, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {n:4d}  {folder}")
    if ambiguous:
        print("\n  ambiguous (within 15% of the winner; filed at the subsystem, or misc):")
        for name, rivals in ambiguous[:15]:
            print(f"    {name:<46s} {', '.join(f'{f}={n}' for f, n in rivals)}")
        if len(ambiguous) > 15:
            print(f"    ... and {len(ambiguous) - 15} more")
    if unclassified:
        print(f"\n  unclassified -- no project include at all ({len(unclassified)}):")
        for name in unclassified[:20]:
            print(f"    {name}")
        if len(unclassified) > 20:
            print(f"    ... and {len(unclassified) - 20} more")

    if args.plan:
        by_folder: dict[str, list[str]] = {}
        for mod, folder in sorted(assignment.items()):
            by_folder.setdefault(folder, []).append(mod)
        lines = ["# Generated by classify_tests.py from each test's own project includes.",
                 "# Review before applying: an `#include` names what a test is BUILT against, which",
                 "# is usually but not always what it is ABOUT.", ""]
        for folder, mods in sorted(by_folder.items()):
            lines.append(f"{folder}: {' '.join(mods)}")
        args.plan.write_text("\n".join(lines) + "\n")
        print(f"\n  wrote {args.plan}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
