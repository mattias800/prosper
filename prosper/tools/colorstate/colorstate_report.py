#!/usr/bin/env python3
"""Summarise a PROSPER_COLORSTATETRACE log: do draws reach the scanout, and with
what colour state?

(Suppression is the guest mask, not CB_COLOR_CONTROL.MODE -- see #1724.)
`PROSPER_COLORSTATETRACE=1|all|WxH` makes the live backend emit one raw-to-resolved
colour/depth record per matching draw (see `prosper/src/gpu/gpu_execute.hpp`). On a
4K title that is millions of lines, which is unreadable by hand. This tool answers
the questions those lines exist to answer:

  1. Do graphics draws write the *scanout* surface at all, or only offscreen targets?
  2. Are their colour writes enabled, or suppressed by CB_COLOR_CONTROL.MODE=DISABLE
     or by a zero CB_TARGET_MASK / CB_SHADER_MASK?
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
    per_minute_suppressed = Counter()
    addrs = Counter()

    for rec in records:
        total += 1
        if not hits_scanout(rec):
            continue
        scan += 1
        eff = effective_mask(rec)
        cwms = {t["cwm"] for t in rec["targets"]
                if t["addr"].startswith(scanout_prefix)}
        # #1724: MODE does NOT gate colour writes. The renderer derives the write mask from
        # CB_TARGET_MASK & CB_SHADER_MASK alone, so a mode=0 draw with a non-zero mask DOES write.
        # Keying this on mode would make the tool contradict the renderer it exists to triage.
        writes_colour = eff != 0 and any(cwms)
        combos[(rec["mode"], eff, tuple(sorted(cwms)))] += 1
        per_minute[rec["ts"] or "??:??"][rec["mode"]] += 1
        if not writes_colour:
            per_minute_suppressed[rec["ts"] or "??:??"] += 1
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
        if eff == 0 or not any(cwms):
            note = "   [colour writes suppressed by the guest's mask]"
        elif mode not in (1, 3, 6):
            note = "   [mode not modelled -- runs as an ordinary colour draw]"
        print(f"  mode={mode} ({name:15s}) effective={eff:02x} cwm={cwm:<6s} "
              f"draws={n}{note}", file=out)
    print("\nscanout draws per guest-clock minute (compare a good phase against a "
          "bad one):", file=out)
    for minute in sorted(per_minute):
        c = per_minute[minute]
        tot = sum(c.values())
        # #1724: suppression is the guest's mask, NOT mode. Keying this on mode over-reported by
        # 8,326 draws on one measured Plucky Squire run. `mode0=` stays in the breakdown because
        # it is still worth seeing, but it is no longer labelled as suppression.
        supp = per_minute_suppressed.get(minute, 0)
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
"""


def selftest():
    import io
    recs = list(parse(SELFTEST.splitlines()))
    assert len(recs) == 3, recs
    assert recs[0]["mode"] == 1 and effective_mask(recs[0]) == 0x0F
    assert recs[1]["mode"] == 0 and effective_mask(recs[1]) == 0x00
    assert recs[2]["targets"][0]["addr"] == "0x3005120000"
    assert recs[0]["ts"] == "16:09"
    # An absent mask must mean write-all, not write-nothing.
    assert effective_mask({"tmask_present": 0, "tmask": 0,
                           "smask_present": 0, "smask": 0}) == 0xFF
    buf = io.StringIO()
    report(list(parse(SELFTEST.splitlines())), "0x9fc", 10, out=buf)
    text = buf.getvalue()
    assert "writing scanout (0x9fc...): 2" in text, text
    assert "offscreen only:                       1" in text, text
    assert "colour writes suppressed" in text, text
    # The offscreen-only draw must not be counted as reaching the scanout.
    assert "0x3005120000" not in text, text
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
