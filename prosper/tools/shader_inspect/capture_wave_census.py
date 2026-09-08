#!/usr/bin/env python3
"""Wave-width census over a captured frame, using the REAL resource table.

`wave_reason_census.py` sweeps a directory of raw RDNA2 dumps, and over GTA V's shader database
99.3% of them could not be lowered: shader_inspect has no descriptors and a real pixel shader
samples a texture. That census is therefore blind to precisely the lighting shaders #3464 is about.

gpu_replay has the real descriptors. This script drives it:

    gpu_replay --bundle F --bundle-extract-submit N cap.prgcap   # one submit, replayable
    gpu_replay --inspect-only cap.prgcap                         # draw list, with fs= identities
    gpu_replay --dump-shader ID:fs fs.spv cap.prgcap             # the REAL-TABLE compiled module
    shader_inspect fs.spv --wave-reasons                         # reads that module's own markers

so every reason it reports comes from a module compiled the way the renderer compiles it, and no
analysis is reimplemented anywhere.

    python capture_wave_census.py --gpu-replay G --shader-inspect S --capture cap.prgcap
    python capture_wave_census.py --gpu-replay G --shader-inspect S --bundle f.prgbundle

SCOPE, which bounds every number this prints: a capture is ONE FRAME. This answers "of the fragment
shaders this frame ran, how many need a wide wave and what for" -- not "how many does the title
have". That is the right question for #3464 (a frame with missing lighting is the evidence), but it
is a different question from the database sweep, and the two must not be quoted as one number.
"""
import argparse
import collections
import re
import subprocess
import sys
import tempfile
from pathlib import Path

DRAW = re.compile(r"^draw\[(\d+)\].*?\bfs=(\d+)/([0-9a-f]+)/", re.M)
DS_SUBMIT = re.compile(r"\bfirst=(\d+) last=(\d+)")
ROW = re.compile(r"wave-reasons required-subgroup-size=(\d+) reasons=(\S+) names=(\S+) "
                 r"native-wave32-admitted=(\d+)")
END = re.compile(r"wave-reasons-end .*?recompiled=(\d+)")


def run(cmd):
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True)


def discover_submit(gpu_replay, bundle):
    """The bundle's submit index is the capture's own ordinal, not 0 or 1.

    `--bundle-extract-submit` matches on `submit_index` and rejects 0, so on a single-submit bundle
    neither of the two numbers anyone tries first works: 0 is refused by the parser and 1 reports
    "bundle has no submit 1". `--bundle-ds-summary` prints the real ordinal as `first=`/`last=`.
    """
    done = run([gpu_replay, "--bundle", bundle, "--bundle-ds-summary"])
    found = {int(m.group(1)) for m in DS_SUBMIT.finditer(done.stdout + done.stderr)}
    if not found:
        return None
    return sorted(found)[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-replay", required=True)
    ap.add_argument("--shader-inspect", required=True)
    ap.add_argument("--capture", help="a .prgcap (one submit)")
    ap.add_argument("--bundle", help="a .prgbundle; one submit is extracted from it")
    ap.add_argument("--submit", type=int, help="submit index in the bundle (default: discovered)")
    ap.add_argument("--work", help="directory for intermediates (default: a temporary one)")
    args = ap.parse_args()
    # Resolved to absolute up front. A relative tool path that a shell finds is not
    # necessarily one CreateProcess finds on Windows, and the failure is a bare
    # FileNotFoundError from deep inside subprocess that names neither which path nor which
    # of the four child invocations produced it.
    args.gpu_replay = str(Path(args.gpu_replay).resolve())
    args.shader_inspect = str(Path(args.shader_inspect).resolve())
    for name in ("capture", "bundle"):
        if getattr(args, name):
            setattr(args, name, str(Path(getattr(args, name)).resolve()))
    if bool(args.capture) == bool(args.bundle):
        print("give exactly one of --capture or --bundle", file=sys.stderr)
        return 2

    tmp = None
    if args.work:
        work = Path(args.work)
        work.mkdir(parents=True, exist_ok=True)
    else:
        tmp = tempfile.TemporaryDirectory()
        work = Path(tmp.name)

    capture = args.capture
    if args.bundle:
        submit = args.submit or discover_submit(args.gpu_replay, args.bundle)
        if submit is None:
            print("could not discover a submit index; pass --submit", file=sys.stderr)
            return 2
        capture = work / "submit.prgcap"
        done = run([args.gpu_replay, "--bundle", args.bundle,
                    "--bundle-extract-submit", submit, capture])
        if not Path(capture).exists():
            print("extract failed:\n" + done.stdout + done.stderr, file=sys.stderr)
            return 2
        print("extracted submit %d" % submit)

    done = run([args.gpu_replay, "--inspect-only", capture])
    draws = DRAW.findall(done.stdout + done.stderr)
    if not draws:
        print("no draws found in %s" % capture, file=sys.stderr)
        print((done.stdout + done.stderr)[-2000:], file=sys.stderr)
        return 2

    # One dump per DISTINCT fragment shader. The `fs=<dwords>/<hash>/` identity on each draw line is
    # the renderer's own content key, so this counts shaders rather than draws -- a frame that draws
    # one shader 400 times must not read as 400 blocked shaders.
    first_draw = {}
    draw_count = collections.Counter()
    for draw_id, dwords, fs_hash in draws:
        draw_count[fs_hash] += 1
        first_draw.setdefault(fs_hash, (draw_id, int(dwords)))

    results = {}
    for fs_hash, (draw_id, dwords) in sorted(first_draw.items()):
        spv = work / ("fs_%s.spv" % fs_hash)
        run([args.gpu_replay, "--dump-shader", "%s:fs" % draw_id, spv, capture])
        if not spv.exists():
            results[fs_hash] = ("dump-failed", 0, "", "", 0)
            continue
        done = run([args.shader_inspect, spv, "--wave-reasons"])
        out = done.stdout + done.stderr
        if not END.search(out):
            # The sentinel's purpose: no line at all must never be tallied as "needs nothing".
            results[fs_hash] = ("no-sentinel", 0, "", "", 0)
            continue
        row = ROW.search(out)
        if not row:
            results[fs_hash] = ("no-census-row", 0, "", "", 0)
            continue
        results[fs_hash] = ("ok", int(row.group(1)), row.group(2), row.group(3),
                            int(row.group(4)))

    ok = {h: r for h, r in results.items() if r[0] == "ok"}
    bad = {h: r for h, r in results.items() if r[0] != "ok"}
    needs = {h: r for h, r in ok.items() if r[1] > 32}
    dropped = {h: r for h, r in needs.items() if not r[4]}

    print("draws=%d  distinct fragment shaders=%d  analysed=%d  unreadable=%d"
          % (len(draws), len(first_draw), len(ok), len(bad)))
    print("require>32=%d  admitted at native wave32=%d  DROPPED=%d"
          % (len(needs), len(needs) - len(dropped), len(dropped)))
    if needs:
        hist = collections.Counter((r[2], r[3]) for r in needs.values())
        print("\nreason sets among the %d requiring a wide wave:" % len(needs))
        for (mask, names), n in hist.most_common():
            print("  %-8s %-42s %3d shaders" % (mask, names, n))
    if dropped:
        print("\ndropped shaders (hash, reasons, draws using it):")
        for h, r in sorted(dropped.items(), key=lambda kv: -draw_count[kv[0]]):
            print("  %s  %-8s %-40s %3d draws" % (h[:16], r[2], r[3], draw_count[h]))
    if bad:
        print("\n%d shader(s) yielded no census row -- NOT counted as needing nothing:" % len(bad))
        for h, r in sorted(bad.items()):
            print("  %s  %s" % (h[:16], r[0]))
    print("\nSCOPE: one frame. This is what this frame RAN, not what the title contains.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
