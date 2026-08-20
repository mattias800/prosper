#!/usr/bin/env python3
"""classify_tests.py -- propose a folder for each test, from what the test actually includes.

prosper/tests holds 228 files in one directory. Sorting them by hand means 228 judgement calls made
from filenames, and a filename is a weak signal: test_gpu_capture and test_gpu_dependency_graph
share a prefix and exercise different subsystems, while test_mb3_freelist and test_dmem_topology
share none and both exercise guest memory.

A test's `#include` lines are a much better signal, because they are what the test is actually
built against. Every project header now carries its subsystem and folder in its canonical path
(`gpu/recompiler/rdna2_to_spirv.hpp`), so the includes name the destination directly. This reports
that mapping and emits a plan for move_module.py.

It PROPOSES. A test that includes three subsystems has a real ambiguity, and the tool says so rather
than picking quietly -- those are the ones worth a person's attention, and they are a short list.

  python3 prosper/tools/refactor/classify_tests.py                    # report
  python3 prosper/tools/refactor/classify_tests.py --plan out.txt     # emit a move plan
"""

import argparse
import collections
import pathlib
import re
import subprocess
import sys

PREFIXES = ("gpu", "hle", "host", "shared", "self", "loader")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="prosper/tests")
    ap.add_argument("--plan", type=pathlib.Path)
    ap.add_argument("--depth", type=int, default=2,
                    help="1 mirrors the subsystem (tests/gpu), 2 mirrors its folder too "
                         "(tests/gpu/recompiler)")
    args = ap.parse_args()

    root = pathlib.Path(subprocess.run(["git", "rev-parse", "--show-toplevel"],
                                       capture_output=True, text=True, check=True).stdout.strip())
    tdir = root / args.dir
    inc = re.compile(r'#include\s+"(' + "|".join(PREFIXES) + r')/([^/"]+)/[^"]+"')
    # Subsystems that are still flat have no folder component; classify those at depth 1.
    flat = re.compile(r'#include\s+"(' + "|".join(PREFIXES) + r')/([^/"]+)"')
    global INCLUDE_NAMES
    INCLUDE_NAMES = re.compile(r'#include\s+"([^"/]+)"')

    # --- weight each include by how RARE it is -------------------------------------------------
    # The first version of this counted includes directly, and put 76 of 228 tests in hle/dispatch --
    # not because they are about the dispatcher, but because every HLE test includes dispatch.hpp to
    # register a stub. A header that appears in 76 tests separates nothing; one that appears in two
    # says almost everything. So each include contributes log(N / how many tests include it), which
    # is the standard inverse-document-frequency weighting and collapses the ubiquitous headers to
    # near zero without needing a hand-maintained list of "boring" headers -- a list that would go
    # stale silently, and whose staleness would look exactly like a confident classification.
    import math
    files = sorted(tdir.glob("*.cpp")) + sorted(tdir.glob("*.h")) + sorted(tdir.glob("*.hpp"))
    texts = {f: f.read_text(errors="ignore") for f in files}
    df: collections.Counter = collections.Counter()
    for f in files:
        df.update(set(inc.findall(texts[f])))
    total = max(len(files), 1)
    idf = {k: math.log(total / v) for k, v in df.items()}

    assignment: dict[str, str] = {}
    ambiguous: list[tuple[str, list[tuple[str, int]]]] = []
    unclassified: list[str] = []

    # Fixtures and harnesses are not tests OF anything -- they are what tests are built on, and
    # every one of them would otherwise classify as whatever it happens to include most.
    FIXTURES = {"render_runner.h", "compute_runner.h", "image_compute_runner.h", "spirv_compute.h",
                "spirv_triangle.h", "synth_prx.h", "handmade_prx.h", "at9_testvec.h",
                "test_scratch.h"}
    RUNNERS = {"render_runner.h", "compute_runner.h", "image_compute_runner.h"}
    # A header included by more than a fifth of the tree separates nothing. idf already shrinks its
    # weight, but a test whose ONLY project include is such a header still gets classified by it --
    # and "every HLE test includes dispatch.hpp to register a stub" is exactly that case, which put
    # 54 unrelated tests in hle/dispatch. Where all the evidence is ubiquitous, the honest answer is
    # the subsystem, not the folder.
    UBIQUITOUS = {k for k, v in df.items() if v > 0.20 * total}

    for path in files:
        is_fixture = (path.name in FIXTURES or path.name.endswith("_fixture.hpp")
                      or (path.suffix in (".h", ".hpp") and not path.name.startswith("test_")))
        if is_fixture:
            assignment[path.stem] = "fixtures"
            continue
        seen = set(inc.findall(texts[path]))
        seen |= {(sub, "") for sub, _f in flat.findall(texts[path])}
        if not seen:
            # No project include at all. If it drives one of the Vulkan/compute harnesses it is an
            # execution test, which is a real category; otherwise it is genuinely unclassifiable
            # from includes and a person should look at it.
            if RUNNERS & set(INCLUDE_NAMES.findall(texts[path])):
                assignment[path.stem] = "render"
            else:
                unclassified.append(path.name)
            continue
        score: dict[tuple[str, str], float] = collections.defaultdict(float)
        for key in seen:
            score[key] += idf[key]
        ranked = sorted(score.items(), key=lambda kv: -kv[1])
        (sub, folder), top = ranked[0]
        # A rival is only a rival if it is genuinely close; exact float ties are rare once the
        # weights differ, so "within 15%" is what ambiguity means here.
        rivals = [(f"{s}/{f}", round(n, 2)) for (s, f), n in ranked if n >= top * 0.85]
        if len(rivals) > 1:
            ambiguous.append((path.name, rivals))
            subs = {s for (s, _f), n in ranked if n >= top * 0.85}
            assignment[path.stem] = (sub if len(subs) == 1 else "misc")
            continue
        if all(k in UBIQUITOUS for k in seen):
            assignment[path.stem] = sub
            continue
        assignment[path.stem] = (f"{sub}/{folder}" if (args.depth >= 2 and folder) else sub)

    counts = collections.Counter(assignment.values())
    print(f"== {len(assignment)} classified, {len(unclassified)} unclassified, "
          f"{len(ambiguous)} ambiguous ==")
    for folder, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {n:4d}  {folder}")
    if ambiguous:
        print(f"\n  ambiguous (tied between folders; assigned to the subsystem, or misc):")
        for name, rivals in ambiguous[:15]:
            print(f"    {name:<46s} {', '.join(f'{f}={n}' for f, n in rivals)}")
        if len(ambiguous) > 15:
            print(f"    ... and {len(ambiguous) - 15} more")
    if unclassified:
        print(f"\n  unclassified (no folder-qualified project include) -- {len(unclassified)}:")
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
