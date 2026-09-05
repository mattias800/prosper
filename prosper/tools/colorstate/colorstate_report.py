#!/usr/bin/env python3
"""Summarise a PROSPER_COLORSTATETRACE log: do draws reach the scanout, and with
what colour state?

(Suppression is the guest mask, not CB_COLOR_CONTROL.MODE -- see #1724.)
`PROSPER_COLORSTATETRACE=1|all|WxH` makes the live backend emit one raw-to-resolved
colour/depth record per matching draw (see `prosper/src/gpu/execute/gpu_execute.hpp`). On a
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
            cur = {"ts": ts, "es": m.group(1), "ps": m.group(2),
                   # The RAW CB_COLOR_CONTROL word, retained beside the decoded mode. A decode
                   # disagreeing with itself and a guest writing two different words are
                   # indistinguishable once the word is dropped -- which is instrument trap 265.
                   "cb_control_present": int(m.group(3)), "cb_control": int(m.group(4), 16),
                   "mode": int(m.group(5)),
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


def report(records, scanout_prefix, top, out=sys.stdout, by_program=False,
           by_pipeline=False):
    def hits_scanout(rec):
        return any(t["addr"].startswith(scanout_prefix) for t in rec["targets"])

    total = scan = 0
    combos = Counter()
    per_minute = defaultdict(Counter)
    per_minute_suppressed = Counter()
    addrs = Counter()
    # Per pixel-shader program, for --by-program. A flat scanout is written by SOME program, and
    # "which one" is the question a mode/mask census cannot answer -- two programs with identical
    # colour state are one row there and two rows here.
    per_program = defaultdict(lambda: {"draws": 0, "writing": 0,
                                       "modes": Counter(), "effs": Counter()})
    # Per (vertex, pixel, raw CB_COLOR_CONTROL word), for --by-pipeline. See trap 265: a per-PROGRAM
    # breakdown cannot express a distinction keyed on the pipeline, and reading one as though it
    # could produced exactly the wrong conclusion about CB_COLOR_CONTROL.MODE on #1706.
    per_pipeline = Counter()

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
        per_pipeline[(rec["es"], rec["ps"], rec["cb_control"], rec["mode"],
                      eff, writes_colour)] += 1
        entry = per_program[rec["ps"]]
        entry["draws"] += 1
        entry["writing"] += 1 if writes_colour else 0
        entry["modes"][rec["mode"]] += 1
        entry["effs"][eff] += 1

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
    if by_pipeline:
        # The axis --by-program cannot express: the (vertex, pixel) pair AND the raw register word.
        # If the raw words differ per pipeline, the guest programmed them and the decode is faithful;
        # if one word appears under both, the decode is disagreeing with itself.
        print("\nscanout draws by pipeline and RAW CB_COLOR_CONTROL (--by-pipeline):", file=out)
        for (es, ps, cbc, mode, eff, writes), n in per_pipeline.most_common(top):
            name = MODE_NAMES.get(mode, "UNKNOWN")
            print(f"  {n:>8}  es={es:<14} ps={ps:<14} cb-control=0x{cbc:08x} "
                  f"mode={mode} ({name}) effective={eff:02x} writes-colour={str(writes):5s}", file=out)
        words = {cbc for (_, _, cbc, _, _, _) in per_pipeline}
        # THE FALSIFIABLE GENERALISATION: is the raw register word determined by the PIPELINE?
        # This is the whole-population form of the manual result on #1706, and unlike the two checks
        # below it can come back either way -- a guest is free to write different CB_COLOR_CONTROL
        # values for the same (vertex, pixel) pair, and if it does, the pipeline is not the key and
        # any per-pipeline conclusion is unsafe.
        words_by_pipeline = defaultdict(set)
        for (es, ps, cbc, _, _, _) in per_pipeline:
            words_by_pipeline[(es, ps)].add(cbc)
        split = {k: v for k, v in words_by_pipeline.items() if len(v) > 1}
        print(f"  {len(words)} distinct raw CB_COLOR_CONTROL word(s) over "
              f"{len(words_by_pipeline)} (vertex, pixel) pair(s); "
              f"{len(split)} pair(s) carry MORE THAN ONE word", file=out)
        for (es, ps), v in sorted(split.items())[:top]:
            print(f"    SPLIT es={es} ps={ps} words=" +
                  ",".join(f"0x{w:08x}" for w in sorted(v)), file=out)
        if not split:
            print("    -> every pipeline carries exactly one word, so on THIS run the register "
                  "value is pipeline-determined. That is a property of the guest's command stream "
                  "and it is what a per-pipeline reading needs; it says nothing on its own about "
                  "whether the field is decoded correctly.", file=out)
        # A TRIPWIRE, NOT EVIDENCE, and labelled as such where it prints. `mode` is extracted from
        # `cb-control` by a pure function (`render_state.cpp`'s PM4_FIELD), so on any log this
        # emitter produced these agree BY CONSTRUCTION and this line cannot fail. It is here to catch
        # a log whose two fields disagree -- a hand-edited fixture, a future emitter that computes
        # the mode from something else, or a merged log from two builds. Reading agreement here as
        # evidence about the decode is the void-discriminator error this tool exists to help avoid.
        mismatched = [(es, ps, cbc, mode) for (es, ps, cbc, mode, _, _) in per_pipeline
                      if mode != ((cbc >> 4) & 0x7)]
        if mismatched:
            print(f"  {len(mismatched)} row(s) whose reported mode DISAGREES with bits [6:4] of "
                  f"their own raw word -- this log's two fields are inconsistent, so neither can be "
                  f"trusted:", file=out)
            for es, ps, cbc, mode in mismatched[:top]:
                print(f"    MISMATCH es={es} ps={ps} cb-control=0x{cbc:08x} "
                      f"reported mode={mode} but [6:4]={(cbc >> 4) & 0x7}", file=out)
        else:
            print("  reported mode agrees with bits [6:4] of the raw word on every row "
                  "(a tripwire on the log's own consistency -- these are the same value by "
                  "construction, so this CANNOT fail on a log prosper produced, and agreement "
                  "here is not evidence about the decode)", file=out)
    if by_program:
        # Which SHADER writes the presented surface. The mode/mask census above answers "with what
        # colour state"; it cannot answer "which draw", and when a scanout goes flat that is the
        # question -- a single fullscreen program covering the surface is one row here and is
        # indistinguishable from a thousand small draws in every other section of this report.
        # The program address is what PROSPER_SKIP_DRAW_PROGRAM takes, so a row here is directly
        # actionable as a one-program A/B rather than a process-wide lever.
        print("\nscanout draws by pixel-shader program (--by-program):", file=out)
        ranked = sorted(per_program.items(),
                        key=lambda kv: kv[1]["draws"], reverse=True)[:top]
        for ps, e in ranked:
            modes = ",".join(f"{MODE_NAMES.get(m, m)}={n}"
                             for m, n in sorted(e["modes"].items()))
            effs = ",".join(f"{v:02x}={n}" for v, n in sorted(e["effs"].items()))
            print(f"  ps={ps:<14} draws={e['draws']:<7d} writing-colour={e['writing']:<7d} "
                  f"modes[{modes}] effective[{effs}]", file=out)
        # A program showing several modes is the signal to re-key, not a conclusion. Say so where the
        # reader is looking at it, because the tempting reading -- "the decoded field is unreliable"
        # -- is the one that was wrong (trap 265).
        # Scans per_program, NOT `ranked`: a multi-mode program outside --top would otherwise get
        # neither a row nor the warning, and the DEFAULT --top is the silent case. --by-pipeline's
        # verdict already scans the whole population; this now matches it.
        multi = [ps for ps, e in per_program.items() if len(e["modes"]) > 1]
        if multi:
            shown = {ps for ps, _ in ranked}
            hidden = [ps for ps in multi if ps not in shown]
            print(f"  NOTE: {len(multi)} program(s) carry more than one mode" +
                  (f", {len(hidden)} of them BELOW --top and not listed above ({', '.join(hidden[:5])})"
                   if hidden else "") + ". That is NOT evidence the "
                  "decode is unreliable -- re-key with --by-pipeline before concluding anything, "
                  "since a guest writing two different CB_COLOR_CONTROL words per (vertex, pixel) "
                  "pair looks identical here (instrument trap 265).", file=out)


SELFTEST = """\
[2026.07.31-16.09.04:786][992]LogSlate: Took 0.000306 second
[color-state] es=0x1 ps=0x2 cb-control=1:00cc0010 mode=1 target-mask=1:0000000f shader-mask=1:0000000f
[color-state]   color0=0x9fc2000000 3840x2160 raw-format=10 resolved-format=44 resolved-cwm=f
[color-state]   depth=1:1x1 raw-size=00000000 test=0 write=0 z=0x0/0x0 s=0x0/0x0 htile=0x0 stencil=0
[color-state] es=0x1 ps=0x3 cb-control=1:00cc0000 mode=0 target-mask=1:00000000 shader-mask=1:00000000
[color-state]   color0=0x9fc2000000 3840x2160 raw-format=10 resolved-format=44 resolved-cwm=0
[color-state] es=0x1 ps=0x4 cb-control=1:00cc0010 mode=1 target-mask=1:0000000f shader-mask=1:0000000f
[color-state]   color0=0x3005120000 1920x1080 raw-format=10 resolved-format=44 resolved-cwm=f
[color-state] es=0x5 ps=0x2 cb-control=1:00cc0000 mode=0 target-mask=1:00000007 shader-mask=1:0000000f
[color-state]   color0=0x9fc2000000 3840x2160 raw-format=10 resolved-format=44 resolved-cwm=7
"""


def selftest():
    import io
    recs = list(parse(SELFTEST.splitlines()))
    assert len(recs) == 4, recs
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
    assert "writing scanout (0x9fc...): 3" in text, text
    assert "offscreen only:                       1" in text, text
    assert "colour writes suppressed" in text, text
    # The offscreen-only draw must not be counted as reaching the scanout.
    assert "0x3005120000" not in text, text
    # --by-program must name the shader that writes the scanout and separate it from the one whose
    # colour is suppressed. Without the section (or with the two programs merged) these fail: the
    # mode/mask census above cannot distinguish them, which is the reason the section exists.
    buf2 = io.StringIO()
    report(list(parse(SELFTEST.splitlines())), "0x9fc", 10, out=buf2, by_program=True)
    prog = buf2.getvalue()
    assert "scanout draws by pixel-shader program" in prog, prog
    assert "ps=0x2" in prog and "ps=0x3" in prog, prog
    # ps=0x2 writes colour; ps=0x3 is masked off. Same surface, same section, opposite verdicts.
    assert "ps=0x2            draws=2       writing-colour=2" in prog, prog
    assert "ps=0x3            draws=1       writing-colour=0" in prog, prog
    # The offscreen program must not appear in a SCANOUT breakdown.
    assert "ps=0x4" not in prog, prog
    # --by-pipeline must express the distinction --by-program structurally cannot: ps=0x2 appears
    # under TWO vertex shaders with TWO different raw CB_COLOR_CONTROL words, and reading only the
    # per-program view ("one program, two modes") is what produced the wrong conclusion this flag
    # exists to prevent (instrument trap 265).
    buf3 = io.StringIO()
    report(list(parse(SELFTEST.splitlines())), "0x9fc", 10, out=buf3, by_pipeline=True)
    pipe = buf3.getvalue()
    assert "scanout draws by pipeline and RAW CB_COLOR_CONTROL" in pipe, pipe
    assert "es=0x1" in pipe and "es=0x5" in pipe, pipe
    assert "cb-control=0x00cc0010" in pipe and "cb-control=0x00cc0000" in pipe, pipe
    # The verdict line is the point of the flag, and it must key on the RAW WORD: each distinct word
    # here decodes to exactly one mode, so the decode is faithful and "one program, two modes" is a
    # statement about the guest rather than about prosper.
    assert "0 pair(s) carry MORE THAN ONE word" in pipe, pipe
    assert "pipeline-determined" in pipe, pipe
    assert "agrees with bits [6:4]" in pipe, pipe

    # The two checks above must be able to FAIL, so drive each with a fixture that breaks it.
    #
    # (a) A log whose reported mode disagrees with bits [6:4] of its own raw word -- exactly the
    #     hand-built case that showed the previous verdict was unfalsifiable: two different words
    #     both reported as mode 0, i.e. every NORMAL draw mislabelled DISABLE. The old check said
    #     "the decode is faithful"; this one must say MISMATCH.
    BAD_MODE = SELFTEST.replace(
        "cb-control=1:00cc0010 mode=1 target-mask=1:0000000f shader-mask=1:0000000f\n"
        "[color-state]   color0=0x9fc2000000",
        "cb-control=1:00cc0010 mode=0 target-mask=1:0000000f shader-mask=1:0000000f\n"
        "[color-state]   color0=0x9fc2000000", 1)
    buf4 = io.StringIO()
    report(list(parse(BAD_MODE.splitlines())), "0x9fc", 10, out=buf4, by_pipeline=True)
    bad = buf4.getvalue()
    assert "MISMATCH" in bad and "reported mode=0 but [6:4]=1" in bad, bad
    assert "agrees with bits [6:4] of the raw word on every row" not in bad, bad

    # (b) A log where ONE (vertex, pixel) pair carries two different raw words -- the guest writing
    #     different colour state for the same pipeline. Then the pipeline is not the key and a
    #     per-pipeline reading is unsafe, which is the finding this line has to be able to report.
    SPLIT = SELFTEST + (
        "[color-state] es=0x5 ps=0x2 cb-control=1:00cc0010 mode=1 "
        "target-mask=1:00000007 shader-mask=1:0000000f\n"
        "[color-state]   color0=0x9fc2000000 3840x2160 raw-format=10 resolved-format=44 "
        "resolved-cwm=7\n")
    buf5 = io.StringIO()
    report(list(parse(SPLIT.splitlines())), "0x9fc", 10, out=buf5, by_pipeline=True)
    sp = buf5.getvalue()
    assert "1 pair(s) carry MORE THAN ONE word" in sp, sp
    assert "SPLIT es=0x5 ps=0x2 words=0x00cc0000,0x00cc0010" in sp, sp
    assert "pipeline-determined" not in sp, sp

    # (c) --by-program's NOTE must scan the whole population, not the --top slice. ps=0x2 carries two
    #     modes; at --top 1 it is ranked first here, so rank it out by giving another program more
    #     draws. The NOTE must still fire and must say the program is below --top.
    MANY = SELFTEST + "".join(
        "[color-state] es=0x9 ps=0x9 cb-control=1:00cc0010 mode=1 "
        "target-mask=1:0000000f shader-mask=1:0000000f\n"
        "[color-state]   color0=0x9fc2000000 3840x2160 raw-format=10 resolved-format=44 "
        "resolved-cwm=f\n" for _ in range(5))
    buf6 = io.StringIO()
    report(list(parse(MANY.splitlines())), "0x9fc", 1, out=buf6, by_program=True)
    many = buf6.getvalue()
    assert "ps=0x2" not in many.split("--by-program")[1].split("NOTE")[0], many
    assert "NOTE: 1 program(s) carry more than one mode" in many, many
    assert "BELOW --top and not listed above (0x2)" in many, many
    # And the per-program view must WARN rather than invite the wrong reading, since ps=0x2 now
    # carries two modes there.
    assert "instrument trap 265" in prog, prog
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
    ap.add_argument("--by-pipeline", action="store_true",
                    help="break scanout draws down by (vertex, pixel) pair AND the raw "
                         "CB_COLOR_CONTROL word -- the axis --by-program cannot express "
                         "(instrument trap 265)")
    ap.add_argument("--by-program", action="store_true",
                    help="also break scanout draws down by pixel-shader program address, which is "
                         "what PROSPER_SKIP_DRAW_PROGRAM takes")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return 0
    if not args.log or not args.scanout_prefix:
        ap.error("LOG and --scanout-prefix are required "
                 "(the scanout VA is title-specific; the tool will not guess)")
    with open(args.log, "r", errors="replace") as fh:
        report(parse(fh), args.scanout_prefix.lower(), args.top,
               by_program=args.by_program, by_pipeline=args.by_pipeline)
    return 0


if __name__ == "__main__":
    sys.exit(main())
