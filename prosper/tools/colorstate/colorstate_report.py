#!/usr/bin/env python3
"""Summarise a PROSPER_COLORSTATETRACE log: do draws reach the scanout, and with
what colour state?

`PROSPER_COLORSTATETRACE=1|all|WxH` makes the live backend emit one raw-to-resolved
colour/depth record per matching draw (see `prosper/src/gpu/gpu_execute.hpp`). On a
4K title that is millions of lines, which is unreadable by hand. This tool answers
the questions those lines exist to answer:

  1. Do graphics draws write the *scanout* surface at all, or only offscreen targets?
  2. Are their colour writes enabled, or suppressed by the CB_COLOR_CONTROL.MODE the
     draw carries, or by a zero CB_TARGET_MASK / CB_SHADER_MASK?
  3. How does that change over the run -- e.g. between a phase that renders correctly
     and a phase that presents black?

Question 3 is the important one. A suppressed-draw count is only interpretable
against a phase where output is known good: on The Plucky Squire the suppressed
fraction is *higher* while the world renders correctly (88-95%) than during a black
phase, because CB_DISABLE is the engine's depth/shadow prepass scaling with scene
geometry. Reading a single phase in isolation invites exactly that false positive.

The scanout range is title- and run-specific. Take it from the guest's own memory
map (`Frame Buffer va range <lo> - <hi>`) and pass `--scanout-prefix`; there is no
safe default, so the tool refuses to guess.

Usage:
    colorstate_report.py LOG --scanout-prefix 0x9fc [--top N]
    colorstate_report.py --selftest
"""

import argparse
import re
import sys
from collections import Counter, defaultdict

# [color-state] es=0x.. ps=0x.. cb-control=P:HHHH mode=M target-mask=P:H shader-mask=P:H
RE_HEAD = re.compile(
    r"^\[color-state\] es=(0x[0-9a-f]+) ps=(0x[0-9a-f]+) "
    r"cb-control=(\d):([0-9a-f]+) mode=(\d+) "
    r"target-mask=(\d):([0-9a-f]+) shader-mask=(\d):([0-9a-f]+)")
# [color-state]   colorN=0xADDR WxH raw-format=.. resolved-format=.. resolved-cwm=H
RE_COLOR = re.compile(
    r"^\[color-state\]\s+color(\d)=(0x[0-9a-f]+) (\d+)x(\d+) "
    r"raw-format=(\d+) resolved-format=(\d+) resolved-cwm=([0-9a-f]+)")
# Unreal writes [YYYY.MM.DD-HH.MM.SS:mmm][frame] on every guest line; used only to
# attribute records to a guest-clock minute.
RE_TS = re.compile(r"^\[\d{4}\.\d\d\.\d\d-(\d\d)\.(\d\d)\.(\d\d)")

MODE_NAMES = {0: "DISABLE", 1: "NORMAL", 2: "ELIM_FAST_CLEAR", 3: "RESOLVE",
              6: "DCC_DECOMPRESS"}

# Modes whose colour writes resolve_pipeline_state suppresses (#1588): DISABLE, plus every
# colour-block metadata operation prosper does not model as its own pass. Only NORMAL blends
# the pixel shader's export; RESOLVE becomes a backend color0->color1 copy and DCC_DECOMPRESS
# keeps its helper-program handling, so neither is an ordinary shaded write either but both
# retain a non-zero mask here. Keep this in step with render_state.cpp.
SUPPRESSED_MODES = {0, 2, 4, 5, 7}


def parse(lines):
    """Yield one dict per colour-state draw record."""
    cur, ts = None, None
    for line in lines:
        m = RE_TS.match(line)
        if m:
            ts = f"{m.group(1)}:{m.group(2)}"
            continue
        m = RE_HEAD.match(line)
        if m:
            if cur:
                yield cur
            cur = {"ts": ts, "ps": m.group(2), "mode": int(m.group(5)),
                   "tmask_present": int(m.group(6)), "tmask": int(m.group(7), 16),
                   "smask_present": int(m.group(8)), "smask": int(m.group(9), 16),
                   "targets": []}
            continue
        m = RE_COLOR.match(line)
        if m and cur is not None:
            cur["targets"].append({"slot": int(m.group(1)), "addr": m.group(2),
                                   "w": int(m.group(3)), "h": int(m.group(4)),
                                   "cwm": int(m.group(7), 16)})
    if cur:
        yield cur


def effective_mask(rec):
    """An ABSENT mask means write-all; a PRESENT zero means write-nothing. That
    distinction is why the raw triple is retained rather than the resolved value."""
    t = rec["tmask"] if rec["tmask_present"] else 0xFF
    s = rec["smask"] if rec["smask_present"] else 0xFF
    return t & s & 0xFF


def report(records, scanout_prefix, top, out=sys.stdout):
    def hits_scanout(rec):
        return any(t["addr"].startswith(scanout_prefix) for t in rec["targets"])

    total = scan = 0
    combos = Counter()
    per_minute = defaultdict(Counter)
    addrs = Counter()

    for rec in records:
        total += 1
        if not hits_scanout(rec):
            continue
        scan += 1
        eff = effective_mask(rec)
        cwms = {t["cwm"] for t in rec["targets"]
                if t["addr"].startswith(scanout_prefix)}
        writes_colour = (rec["mode"] not in SUPPRESSED_MODES and eff != 0 and any(cwms))
        combos[(rec["mode"], eff, tuple(sorted(cwms)))] += 1
        per_minute[rec["ts"] or "??:??"][rec["mode"]] += 1
        for t in rec["targets"]:
            if t["addr"].startswith(scanout_prefix):
                addrs[(t["addr"], t["w"], t["h"], writes_colour)] += 1

    print(f"colour-state draw records: {total}", file=out)
    print(f"  writing scanout ({scanout_prefix}...): {scan}", file=out)
    print(f"  offscreen only:                       {total - scan}", file=out)
    if not scan:
        print("\nNO draw wrote the scanout. The render path never reaches the "
              "presented surface.", file=out)
        return
    print("\nscanout surfaces:", file=out)
    for (addr, w, h, vis), n in sorted(addrs.items()):
        print(f"  {addr} {w}x{h} writes-colour={str(vis):5s} draws={n}", file=out)
    print("\nscanout draws by mode / effective mask / resolved cwm:", file=out)
    for (mode, eff, cwms), n in combos.most_common(top):
        name = MODE_NAMES.get(mode, "UNKNOWN")
        cwm = ",".join(f"{c:x}" for c in cwms)
        note = ""
        if mode in SUPPRESSED_MODES and mode != 0:
            note = "   [colour-block metadata op -- colour writes suppressed]"
        elif mode == 0 or eff == 0 or not any(cwms):
            note = "   [colour writes suppressed]"
        print(f"  mode={mode} ({name:15s}) effective={eff:02x} cwm={cwm:<6s} "
              f"draws={n}{note}", file=out)
    print("\nscanout draws per guest-clock minute (compare a good phase against a "
          "bad one):", file=out)
    for minute in sorted(per_minute):
        c = per_minute[minute]
        tot = sum(c.values())
        supp = sum(n for m, n in c.items() if m in SUPPRESSED_MODES)
        modes = " ".join(f"mode{m}={n}" for m, n in sorted(c.items()))
        print(f"  {minute}  total={tot:7d}  suppressed={100.0 * supp / tot:5.1f}%  "
              f"{modes}", file=out)


SELFTEST = """\
[2026.07.31-16.09.04:786][992]LogSlate: Took 0.000306 second
[color-state] es=0x1 ps=0x2 cb-control=1:00cc0010 mode=1 target-mask=1:0000000f shader-mask=1:0000000f
[color-state]   color0=0x9fc2000000 3840x2160 raw-format=10 resolved-format=44 resolved-cwm=f
[color-state]   depth=1:1x1 raw-size=00000000 test=0 write=0 z=0x0/0x0 s=0x0/0x0 htile=0x0 stencil=0
[color-state] es=0x1 ps=0x3 cb-control=1:00cc0000 mode=0 target-mask=1:00000000 shader-mask=1:00000000
[color-state]   color0=0x9fc2000000 3840x2160 raw-format=10 resolved-format=44 resolved-cwm=0
[color-state] es=0x1 ps=0x4 cb-control=1:00cc0010 mode=1 target-mask=1:0000000f shader-mask=1:0000000f
[color-state]   color0=0x3005120000 1920x1080 raw-format=10 resolved-format=44 resolved-cwm=f
[color-state] es=0x1 ps=0x2 cb-control=1:00cc0020 mode=2 target-mask=1:0000000f shader-mask=1:0000000f
[color-state]   color0=0x9fc2000000 3840x2160 raw-format=10 resolved-format=44 resolved-cwm=f
"""


def selftest():
    import io
    recs = list(parse(SELFTEST.splitlines()))
    assert len(recs) == 4, recs
    assert recs[0]["mode"] == 1 and effective_mask(recs[0]) == 0x0F
    assert recs[1]["mode"] == 0 and effective_mask(recs[1]) == 0x00
    assert recs[2]["targets"][0]["addr"] == "0x3005120000"
    assert recs[0]["ts"] == "16:09"
    # #1588: a metadata mode writes no colour even when BOTH the effective mask and the
    # per-target resolved cwm say write-all. The record deliberately carries `resolved-cwm=f`,
    # which is the shape of a log captured BEFORE the suppression landed: that is the only input
    # on which the mode term of the verdict can matter, because a post-fix log already prints
    # cwm=0 and would be classified correctly by the old mask-only rule too. A record with
    # cwm=0 here would make the assertions below pass identically with or without the change.
    assert recs[3]["mode"] == 2 and effective_mask(recs[3]) == 0x0F
    assert {t["cwm"] for t in recs[3]["targets"]} == {0x0F}
    # An absent mask must mean write-all, not write-nothing.
    assert effective_mask({"tmask_present": 0, "tmask": 0,
                           "smask_present": 0, "smask": 0}) == 0xFF
    buf = io.StringIO()
    report(list(parse(SELFTEST.splitlines())), "0x9fc", 10, out=buf)
    text = buf.getvalue()
    assert "writing scanout (0x9fc...): 3" in text, text
    assert "offscreen only:                       1" in text, text
    assert "colour writes suppressed" in text, text
    assert "colour-block metadata op" in text, text
    # The offscreen-only draw must not be counted as reaching the scanout.
    assert "0x3005120000" not in text, text
    # Two of the three scanout draws (MODE=DISABLE and MODE=2) are suppressed; a report that
    # still counted only MODE=DISABLE would print 33.3%.
    assert "suppressed= 66.7%" in text, text
    # The writes-colour verdict itself, which is the term the mode set exists for. The MODE=2
    # record has a write-all cwm, so a mask-only rule calls it a colour writer and splits the
    # scanout surface 2-writing / 1-suppressed. Reading the mode instead gives 1 / 2.
    assert "writes-colour=False draws=2" in text, text
    assert "writes-colour=True  draws=1" in text, text
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", nargs="?", help="run log containing [color-state] lines")
    ap.add_argument("--scanout-prefix",
                    help="address prefix of the guest scanout, e.g. 0x9fc "
                         "(from the guest's 'Frame Buffer va range' line)")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return 0
    if not args.log or not args.scanout_prefix:
        ap.error("LOG and --scanout-prefix are required "
                 "(the scanout VA is title-specific; the tool will not guess)")
    with open(args.log, "r", errors="replace") as fh:
        report(parse(fh), args.scanout_prefix.lower(), args.top)
    return 0


if __name__ == "__main__":
    sys.exit(main())
