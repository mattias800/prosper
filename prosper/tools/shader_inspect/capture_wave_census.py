#!/usr/bin/env python3
"""Wave-width census over a captured frame, using the REAL resource table.

`wave_reason_census.py` sweeps a directory of raw RDNA2 dumps, and over GTA V's shader database
99.3% of them could not be lowered: shader_inspect has no descriptors and a real pixel shader
samples a texture. That census is therefore blind to precisely the lighting shaders #3464 is about.

gpu_replay has the real descriptors. This script drives it:

    gpu_replay --bundle F --bundle-extract-submit N cap.prgcap        # one submit, replayable
    gpu_replay --inspect-only cap.prgcap                              # draws, with fs= identities
    gpu_replay --inspect-only --dump-shader ID:fs fs.spv cap.prgcap   # the REAL-TABLE module
    shader_inspect fs.spv --wave-reasons                              # reads that module's markers

so every reason it reports comes from a module compiled the way the renderer compiles it, and
nothing is reimplemented: this script only sequences those tools and tallies their output.

    python capture_wave_census.py --gpu-replay G --shader-inspect S --capture cap.prgcap
    python capture_wave_census.py --gpu-replay G --shader-inspect S --bundle f.prgbundle

WHAT "DROPPED" MEANS, spelled out because getting it wrong made the first version of this script
report the opposite of the truth. The renderer's gate is FIVE conjuncts
(`render_runner.h:7128-7133`); the reason-set equality at `:7140` is only the innermost. The
decisive one is `bd.allow_native_fragment_vote_width`, which defaults false (`:565`) and is set from
`title_id == "PPSA04263"` alone (`live_renderer.cpp:1216`, `:7591`) -- so **outside GTA V every
shader requiring more than 32 lanes is dropped, whatever its reason set**. Pass `--title` to say
which title the capture came from; without it this assumes NOT on the allowlist, which is true for
every title but one.

SCOPE: a capture is ONE FRAME (and, with --bundle, one SUBMIT of it). This answers "of the fragment
shaders this frame ran, how many need a wide wave and what for" -- not "how many does the title
have". That is the right question for #3464, but it is a different question from the database
sweep, and the two must not be quoted as one number.
"""
import argparse
import collections
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# The title whose reviewed route may run the narrow WaveAny class at native wave32. One string, in
# one place, mirroring live_renderer.cpp:1216.
NATIVE_VOTE_ALLOWLIST = ("PPSA04263",)

DRAW = re.compile(r"^draw\[(\d+)\].*?\bfs=(\d+)/([0-9a-f]+)/", re.M)
DS_SUBMIT = re.compile(r"\bfirst=(\d+) last=(\d+)")
BUNDLE_SUBMITS = re.compile(r"\bbundle v\d+ submits=(\d+)")
ROW = re.compile(r"wave-reasons required-subgroup-size=(\d+) reasons=(\S+) names=(\S+) "
                 r"reason-set-admissible=(\d+)")
END = re.compile(r"wave-reasons-end\b")
WALK = re.compile(r"\bspirv-walk=(\w+)")


def run(cmd):
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True)


def discover_submit(gpu_replay, bundle):
    """The bundle's submit index is the CAPTURE's ordinal, not 0 or 1.

    `--bundle-extract-submit` matches on `submit_index` and rejects 0, so on a single-submit bundle
    neither number anyone tries first works: 0 is refused by the parser and 1 reports "bundle has no
    submit 1". `--bundle-ds-summary` prints the real ordinal as `first=`/`last=`.

    Returns (submit, error). It REFUSES to guess when the bundle holds more than one submit: the
    `first=` fields are per depth-stencil identity and record where each identity was FIRST seen, so
    the maximum of them is not the largest, busiest or last submit -- it is an arbitrary one. A
    census of an arbitrary fraction of a frame that then prints "SCOPE: one frame" is worse than a
    refusal, so the caller is asked for --submit instead.
    """
    done = run([gpu_replay, "--bundle", bundle, "--bundle-ds-summary"])
    text = done.stdout + done.stderr
    count = BUNDLE_SUBMITS.search(text)
    ordinals = sorted({int(m.group(1)) for m in DS_SUBMIT.finditer(text)})
    if not ordinals:
        return None, "no submit ordinal found in --bundle-ds-summary output"
    if count and int(count.group(1)) != 1:
        return None, ("bundle holds %s submits and the ds-summary ordinals (%s) do not identify "
                      "which to census; pass --submit explicitly"
                      % (count.group(1), ",".join(str(o) for o in ordinals[:8])))
    if len(ordinals) > 1:
        return None, ("ambiguous submit ordinals %s; pass --submit explicitly"
                      % ",".join(str(o) for o in ordinals[:8]))
    return ordinals[0], None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-replay", required=True)
    ap.add_argument("--shader-inspect", required=True)
    ap.add_argument("--capture", help="a .prgcap (one submit)")
    ap.add_argument("--bundle", help="a .prgbundle; one submit is extracted from it")
    ap.add_argument("--submit", type=int, help="submit index in the bundle (default: discovered)")
    ap.add_argument("--title", default="",
                    help="title id the capture came from, e.g. PPSA04263. Decides whether the "
                         "native-wave32 allowlist applies; without it, assumed NOT on the "
                         "allowlist, which is true for every title but one.")
    ap.add_argument("--work", help="directory for intermediates (default: a temporary one)")
    args = ap.parse_args()
    # Resolved to absolute up front. A relative tool path that a shell finds is not necessarily one
    # CreateProcess finds on Windows, and the failure is a bare FileNotFoundError from deep inside
    # subprocess that names neither which path nor which of the child invocations produced it.
    args.gpu_replay = str(Path(args.gpu_replay).resolve())
    args.shader_inspect = str(Path(args.shader_inspect).resolve())
    for name in ("capture", "bundle"):
        if getattr(args, name):
            setattr(args, name, str(Path(getattr(args, name)).resolve()))
    if bool(args.capture) == bool(args.bundle):
        print("give exactly one of --capture or --bundle", file=sys.stderr)
        return 2

    allowlisted = args.title in NATIVE_VOTE_ALLOWLIST

    tmp = None
    if args.work:
        work = Path(args.work)
        work.mkdir(parents=True, exist_ok=True)
    else:
        tmp = tempfile.TemporaryDirectory()
        work = Path(tmp.name)

    capture = args.capture
    if args.bundle:
        submit = args.submit
        if submit is None:
            submit, error = discover_submit(args.gpu_replay, args.bundle)
            if submit is None:
                print("cannot choose a submit: %s" % error, file=sys.stderr)
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
        # --inspect-only alongside --dump-shader: the module is written from the same realization
        # either way (byte-identical output, verified), but the frame is not REPLAYED. That matters
        # beyond speed on the title this exists for -- GTA V has two compute programs that hang the
        # GPU into a driver recovery (#2542, #2690), and a census over N distinct shaders would
        # otherwise trigger them N times.
        run([args.gpu_replay, "--inspect-only", "--dump-shader", "%s:fs" % draw_id,
             spv, capture])
        if not spv.exists():
            results[fs_hash] = ("dump-failed", 0, "", "", 0)
            continue
        done = run([args.shader_inspect, spv, "--wave-reasons"])
        out = done.stdout + done.stderr
        if not END.search(out):
            # The sentinel's purpose: no line at all must never be tallied as "needs nothing".
            results[fs_hash] = ("no-sentinel", 0, "", "", 0)
            continue
        walk = WALK.search(out)
        if walk and walk.group(1) != "ok":
            # A truncated module reports size=0/reasons=absent, byte-identical to a genuine
            # wave-free shader. Counting it as analysed would put a module we could not read into
            # the population that says #3464 does not affect it.
            results[fs_hash] = ("spirv-" + walk.group(1), 0, "", "", 0)
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
    # Admitted only if the title is on the allowlist AND the reason set passes. Two more conjuncts
    # (host subgroup-size control, host subgroup features) are properties of the RUN and cannot be
    # decided here, so even a 1 is an upper bound on admission, never a guarantee.
    admitted = {h: r for h, r in needs.items() if allowlisted and r[4]}
    dropped = {h: r for h, r in needs.items() if h not in admitted}

    print("draws=%d  distinct fragment shaders=%d  analysed=%d  unreadable=%d"
          % (len(draws), len(first_draw), len(ok), len(bad)))
    print("title=%s  native-wave32 allowlist: %s"
          % (args.title or "(unspecified)",
             "YES" if allowlisted else "no -- every wide-wave shader below is dropped"))
    print("require>32=%d  possibly admitted=%d  DROPPED=%d"
          % (len(needs), len(admitted), len(dropped)))
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
        print("\n%d shader(s) yielded no usable census row -- NOT counted as needing nothing:"
              % len(bad))
        for h, r in sorted(bad.items()):
            print("  %s  %s" % (h[:16], r[0]))
    if admitted:
        print("\nNOTE: 'possibly admitted' is an UPPER bound. Admission also needs the host to "
              "support\nthe width and the module's subgroup features, which this tool cannot see.")
    print("\nSCOPE: one frame%s. This is what it RAN, not what the title contains."
          % (" (one submit of it)" if args.bundle else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
