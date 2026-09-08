#!/usr/bin/env python3
"""Tally `shader_inspect --wave-reasons` over a directory of raw RDNA2 shader dumps.

Answers, for a whole title rather than for whatever a route happened to reach: how many fragment
shaders require the guest Wave64, what they require it for, and how many of those prosper admits
today at a host native wave32. On any NVIDIA host the entire supported subgroup range is 32..32, so
a shader requiring 64 that the renderer does not admit is DROPPED entirely -- on GTA V
that is most of the world's lighting (#3464).

Why a whole-database sweep rather than the runtime skip log: the runtime logs once per distinct
shader, so its count is a function of how far the route got. Two runtime counts taken this way, 43
and 86, differed by run length alone and read as a regression until they were matched by frame.

    python wave_reason_census.py <shader_inspect> <dir> [--stage-token PS] [--csv out.csv]

IMPORTANT -- read `unrecompiled` before quoting any total. shader_inspect has no resource table, so
a shader that samples a texture cannot be lowered and contributes NO reason data. Those are counted
and reported separately, never as "requires nothing". A census whose unrecompiled count is large is
a census of the shaders that happen not to touch memory, which is not a random sample of anything.
"""
import argparse
import collections
import concurrent.futures
import csv
import re
import subprocess
import sys
from pathlib import Path

END = re.compile(r"wave-reasons-end .*?recompiled=(\d+) table_dependent=(\d+) endpgm=(\d+)")
ROW = re.compile(r"wave-reasons required-subgroup-size=(\d+) reasons=(\S+) names=(\S+) "
                 r"reason-set-admissible=(\d+)")


class Result:
    __slots__ = ("path", "recompiled", "table_dependent", "endpgm",
                 "size", "reasons", "names", "admissible", "error")

    def __init__(self, path):
        self.path = path
        self.recompiled = self.table_dependent = self.endpgm = 0
        self.size = 0
        self.reasons = self.names = ""
        self.admissible = 0
        self.error = ""


def inspect(binary: str, path: Path) -> Result:
    r = Result(path)
    try:
        done = subprocess.run([binary, str(path), "--wave-reasons"],
                              capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as exc:
        r.error = str(exc)
        return r
    end = END.search(done.stdout)
    if not end:
        # The sentinel's whole purpose. Without it a dead process and a shader with no wave
        # requirement look identical, and the second is the answer that would be tallied.
        r.error = "no sentinel (exit %d)" % done.returncode
        return r
    r.recompiled, r.table_dependent, r.endpgm = (int(g) for g in end.groups())
    row = ROW.search(done.stdout)
    if row:
        r.size = int(row.group(1))
        r.reasons, r.names = row.group(2), row.group(3)
        r.admissible = int(row.group(4))
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("binary")
    ap.add_argument("directory")
    ap.add_argument("--stage-token", default="PS",
                    help="only files whose name contains _<TOKEN>_ (default PS)")
    ap.add_argument("--csv")
    ap.add_argument("--jobs", type=int, default=16)
    args = ap.parse_args()
    # Absolute up front: a relative tool path a shell resolves is not necessarily one
    # CreateProcess resolves on Windows. Caught here by the sentinel check rather than
    # producing a wrong number, but it still made the documented invocation fail.
    args.binary = str(Path(args.binary).resolve())

    root = Path(args.directory)
    token = "_%s_" % args.stage_token
    files = sorted(p for p in root.rglob("*.bin") if token in p.name)
    if not files:
        print("no files matching %s under %s" % (token, root), file=sys.stderr)
        return 2

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(lambda p: inspect(args.binary, p), files))

    errors = [r for r in results if r.error]
    unrecompiled = [r for r in results if not r.error and not r.recompiled]
    ok = [r for r in results if not r.error and r.recompiled]
    needs = [r for r in ok if r.size > 32]
    # The reason-set test is only the INNERMOST of the renderer's five conjuncts
    # (render_runner.h:7128-7133). The decisive one, `allow_native_fragment_vote_width`,
    # defaults false (:565) and comes from `title_id == "PPSA04263"` alone
    # (live_renderer.cpp:1216), and two more are properties of the HOST. A raw-dump sweep has
    # neither a title nor a host, so it reports the module property under its own name and
    # never calls anything admitted -- an earlier version did, and undercounted the loss
    # #3464 is about for every title but one.
    admissible = [r for r in needs if r.admissible]

    print("files=%d  errors=%d  unrecompiled(no resource table)=%d  analysed=%d"
          % (len(files), len(errors), len(unrecompiled), len(ok)))
    if not ok:
        print("\nNOTHING was analysed -- every shader needed a resource table this tool cannot")
        print("supply. No statement about wave width can be made from this run.")
    else:
        print("of the analysed: require>32=%d  of those reason-set admissible=%d"
              % (len(needs), len(admissible)))
        print("NOTE: admissible is NOT admitted. The renderer also requires the title to be on")
        print("the native-wave32 allowlist (currently PPSA04263 only) and the host to support")
        print("the width and features. For a per-title verdict use capture_wave_census.py.")
        by_reason = collections.Counter((r.reasons, r.names) for r in needs)
        if by_reason:
            print("\nreason sets among the %d that require a wide wave:" % len(needs))
            for (mask, names), n in by_reason.most_common():
                print("  %-8s %-40s %4d" % (mask, names, n))
        if needs:
            print("\nfirst wide-wave shaders by name:")
            for r in sorted(needs, key=lambda r: r.path.name)[:20]:
                print("  %-8s admissible=%d  %s"
                      % (r.reasons, r.admissible, r.path.name))
    # The caveat repeated at the END too: a reader who scrolls to the bottom for the numbers must
    # meet it there as well as at the top.
    if unrecompiled:
        print("\nCAVEAT: %d of %d shaders (%.1f%%) could not be lowered without a resource table"
              % (len(unrecompiled), len(files), 100.0 * len(unrecompiled) / len(files)))
        print("and contribute no reason data. The analysed set is the shaders that happen not to")
        print("need one, which is NOT a random sample -- a texture-sampling lighting shader is")
        print("exactly the kind that is missing. Treat every count above as a lower bound.")
    for r in errors[:10]:
        print("  error: %s: %s" % (r.path.name, r.error))

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["name", "recompiled", "table_dependent", "endpgm",
                        "required_subgroup_size", "reasons", "names",
                        "reason_set_admissible", "error"])
            for r in results:
                w.writerow([r.path.name, r.recompiled, r.table_dependent, r.endpgm,
                            r.size, r.reasons, r.names, r.admissible, r.error])
        print("\nwrote %s" % args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
