#!/usr/bin/env python3
"""Classify captured frames: is anything drawn, is any of it legible, and is it only a HUD?

Why this exists. `tools/screenshot` already writes `distinct_rgb_colors` and `nonblack_rgb_pixels`
to its JSONL manifest, over every pixel, in the same directory as the PNGs -- so "how many colours"
and "how much is lit" are NOT new here. What is new is `max` (the brightest channel anywhere), the
classification below, and running over any directory of frames including ones with no manifest.

Two ways of answering this by hand are wrong, and both were made here:

- **Downsampling before counting colours destroys thin UI.** A 4K frame resized to 160x90 loses white
  menu text on black entirely. Measured on Stray (`PPSA02101`), 2026-08-27: a frame containing START
  GAME / SETTINGS / CREDITS and a legible build stamp reported `colours=61 FLAT-BLACK mean=(0,0,0)`,
  and the title was one edit from being written up as not rendering.
- **"Fraction of non-black pixels" scores a flat WHITE clear as a perfect frame** -- 100% non-black,
  and every is-it-black test passes it.

  FLAT          nothing rendered, or a uniform fill: peak <= noise, or <= 8 colours covering >= 50%
                (a black clear, a white clear, a two-tone letterbox).
  SPARSE        nothing rises above near-black (peak < 48), whatever the coverage.
  UI-ON-BLACK   legible content covering under 2% -- a HUD, notice or small logo over a world that
                did not draw. THE diagnostic class: it means the guest is drawing and the WORLD is
                what is missing.
  LIT           something legible is drawn over more than 2% of the frame.

**`LIT` is not a promise that the frame is a game scene, and no threshold here can make it one.**
That is a measured limitation, not caution. On the committed corpus, UI and real scenes INTERLEAVE on
every statistic this tool computes:

    crisis-core-title.png            10.13% cover  4096+ colours   text + a glow on black
    stray-brightness-calibration.png 10.13% cover    228 colours   a settings menu on black
    messenger-title.png              12.41% cover     37 colours   real pixel art
    oregon-trail-gameloft-splash.png 16.03% cover   2159 colours   a flat logo on black
    blue-prince-title.png            21.83% cover  4096+ colours   real rendered 3D art

Coverage puts a logo above a scene; colour count puts pixel art below a logo. An earlier revision of
this tool drew the boundary at 10% coverage and claimed a "clean gap" in the corpus supported it --
the gap was an artifact of taking the minimum over frames a PREVIOUS revision had labelled as scenes,
which is the assumption under test. Deciding "is the world rendering?" above 2% needs eyes on the
frame; this tool narrows which frames need them.

The 2% bound on `UI-ON-BLACK` is where the class was independently validated, on evidence chosen by
someone other than its author: `sonic-frontiers-cyberspace-hud` (0.81%), `sonic-frontiers-autosave-
notice` (1.05%) and `metaphor-loading-mascot` (1.49%) are exactly the titles recorded as "world black
behind the HUD" (#2790) and "the background does not draw" (#2952).

CAVEAT, and it is not detectable from pixels: `--fps-overlay` burns prosper's OWN counter into the
PNG. A black frame carrying only that overlay classifies UI-ON-BLACK, which reads as "the guest is
drawing" when nothing of the sort happened. Do not classify overlay runs; `screenshot`'s manifest
records `run.assertions.fps_overlay` if you need to check after the fact.

Also not detected: a smooth gradient with no picture in it classifies LIT. prosper's seed-miss
gradient is exactly that shape, so a frame can score well here and still be a diagnostic fill.

SAMPLING: a 1/16 nearest-neighbour stride (every 4th pixel on both axes) at NATIVE resolution --
never a resize, which is the distinction the tool rests on. A stride still aliases: a black frame
ruled with 1px vertical lines at `x % 4 == 1`, covering a quarter of the screen, reports
`max=0 colours=1 FLAT` and is indistinguishable from a black clear. Pass `step=1` in-process if you
need it exact.

The colour count saturates at 4096 and the report marks a saturated count with a trailing `+`, so
two `4096+` rows are not comparable with each other.

Alpha is discarded rather than composited over black. That sounds like a divergence from
`capture_manifest.cpp`, which multiplies RGB by alpha -- measured on a real `screenshot` PNG it is
not, because `normalize_capture_rgba` forces alpha to 255 for composited and republished frames
(0 of 8,294,400 pixels differed). `CaptureSource::RawScanout` is the one path that leaves alpha
untouched, and no sample of it has been checked.

Always exits 0 on classification -- this reports, it does not gate. An unreadable file is reported on
its own line and does not abort the run or lose the tally.
"""
import sys, os, glob
try:
    from PIL import Image
except ImportError:
    sys.exit("frameclass: needs Pillow (python3 -m pip install --user pillow)")

COLOUR_CAP = 4096  # counting stops here; the report suffixes a saturated count with "+"

def classify(path, step=4, noise=8, bright_at=128, dim_at=48):
    return classify_image(Image.open(path).convert("RGB"), os.path.basename(path),
                          step, noise, bright_at, dim_at)

def classify_image(im, name, step=4, noise=8, bright_at=128, dim_at=48):
    w, h = im.size
    px = im.load()
    n = nonblack = bright = mx = 0
    cols = set()
    for y in range(0, h, step):
        for x in range(0, w, step):
            r, g, b = px[x, y]
            n += 1
            m = r if r > g else g
            if b > m: m = b
            if m > mx: mx = m
            if m > noise: nonblack += 1
            if m > bright_at: bright += 1
            if len(cols) < COLOUR_CAP: cols.add((r, g, b))
    share = nonblack / n if n else 0.0
    # A few colours only means "flat" when they COVER the frame -- but a letterbox is black bars plus
    # a uniform fill, which lands at 75-83%, so the bound cannot be 0.90. Gating on the colour count
    # alone instead swallowed a black frame carrying prosper's two-colour fps overlay and 6,120 bright
    # pixels, which is the opposite error.
    uniform_fill = len(cols) <= 8 and share >= 0.50
    if mx <= noise or uniform_fill:        kind = "FLAT"
    # Nothing in the frame rises above near-black. Keyed on the PEAK, not on a count above a fixed
    # brightness: `bendy-title.png` peaks at 127 over 93% of the frame and is a perfectly visible
    # picture, so a "pixels brighter than 128" test called it empty.
    elif mx < dim_at:                      kind = "SPARSE"
    # Legible content over almost nothing. This is the one diagnostic class, and 2% is where it was
    # independently validated -- see the module docstring.
    elif share < 0.02:                     kind = "UI-ON-BLACK"
    else:                                  kind = "LIT"
    return dict(name=name, max=mx, share=100 * share,
                bright=bright, colours=len(cols), kind=kind, w=w, h=h)

def selftest():
    """Every class from a hand-built frame.

    Deliberately NOT sampled from `assets/screenshots/`: the thresholds are tuned against that
    corpus, so a control drawn from it would confirm the tuning rather than the classifier. Each
    case below is constructed pixel by pixel, and each one pins a specific mistake this tool has
    already made once.
    """
    W, H = 640, 360
    def frame(fill=(0, 0, 0)):
        return Image.new("RGB", (W, H), fill)
    def box(im, fx, fy, fw, fh, col):
        for y in range(int(fy * H), int((fy + fh) * H)):
            for x in range(int(fx * W), int((fx + fw) * W)):
                im.putpixel((x, y), col)
        return im

    cases = []
    cases.append(("black clear", frame(), "FLAT"))
    cases.append(("white clear", frame((255, 255, 255)), "FLAT"))
    cases.append(("two-tone split", box(frame((20, 20, 20)), 0, 0, 1, .5, (200, 200, 200)), "FLAT"))
    # A 2.39:1 letterbox is black bars plus a uniform fill, which covers ~75% -- not the ~100% a
    # naive "few colours must cover the frame" bound assumes. Gating uniform_fill at 0.90 classified
    # this as a picture.
    cases.append(("letterbox bars", box(frame(), 0, .126, 1, .748, (140, 140, 140)), "FLAT"))
    # Legible content over almost nothing: the one diagnostic class.
    cases.append(("hud over black", box(frame(), .1, .4, .12, .04, (255, 255, 255)), "UI-ON-BLACK"))
    # prosper's own fps overlay is two colours on black and lands here by design -- see the caveat in
    # the module docstring. Asserted so the documented false positive cannot drift silently.
    cases.append(("fps overlay only", box(frame(), .01, .01, .07, .03, (255, 255, 255)), "UI-ON-BLACK"))
    # 8% coverage: a lit title screen. NOT the diagnostic class -- see the docstring on why coverage
    # cannot tell a title screen from a scene.
    cases.append(("lit title screen", box(frame(), 0, 0, 1, .08, (255, 240, 200)), "LIT"))
    # Peaks at 119, below any "bright pixel" bar, over the whole frame: still a picture.
    dim = frame()
    for y in range(H):
        for x in range(W):
            dim.putpixel((x, y), (x % 120, y % 120, (x + y) % 120))
    cases.append(("dim full-frame picture", dim, "LIT"))
    scene = frame()
    for y in range(H):
        for x in range(W):
            scene.putpixel((x, y), (x % 256, y % 256, (x * y) % 256))
    cases.append(("bright full-frame scene", scene, "LIT"))
    # Nothing rises above near-black. Coverage must NOT rescue these: a fifth of the frame at peak 15
    # is still nothing anyone can see, and a coverage-first ordering called it a picture.
    cases.append(("near-black fade, 5%", box(frame(), 0, 0, 1, .05, (15, 15, 15)), "SPARSE"))
    cases.append(("near-black fade, 20%", box(frame(), 0, 0, 1, .20, (15, 15, 15)), "SPARSE"))

    bad = 0
    for name, im, want in cases:
        got = classify_image(im, name)["kind"]
        ok = got == want
        bad += not ok
        print(f"  {'ok  ' if ok else 'FAIL'}  {name:<24} want={want:<12} got={got}")
    print(f"  {len(cases) - bad}/{len(cases)} passed")
    return 1 if bad else 0

def main(argv):
    if len(argv) >= 2 and argv[1] == "--selftest":
        return selftest()
    if len(argv) < 2:
        sys.exit("usage: frameclass.py <dir-of-pngs | file.png> [...]\n"
                 "       frameclass.py --selftest")
    paths = []
    for a in argv[1:]:
        if os.path.isdir(a):
            # .bmp too: F9 frame grabs write BMP, and CLAUDE.md makes F9 the first loop for a
            # graphical bug -- a directory of grabs must not report "nothing found".
            # Deduped by real path, NOT by globbing both cases: fnmatch normcases on Windows, so
            # "*.png" and "*.PNG" each match every file there and the tally silently doubles.
            found = []
            for ext in ("*.png", "*.bmp", "*.PNG", "*.BMP"):
                found.extend(glob.glob(os.path.join(a, ext)))
            seen = set()
            for f in sorted(found):
                key = os.path.normcase(os.path.realpath(f))
                if key not in seen:
                    seen.add(key)
                    paths.append(f)
        else:
            paths.append(a)
    if not paths:
        sys.exit("frameclass: no .png or .bmp found")
    tally = {}
    for p in paths:
        try:
            r = classify(p)
        except Exception as e:  # truncated PNG, not an image, unreadable -- report and continue
            print(f"{os.path.basename(p)[-28:]:>28}  UNREADABLE: {type(e).__name__}: {e}")
            tally["UNREADABLE"] = tally.get("UNREADABLE", 0) + 1
            continue
        tally[r["kind"]] = tally.get(r["kind"], 0) + 1
        print(f"{r['name'][-28:]:>28}  {r['w']}x{r['h']}  max={r['max']:>3}  "
              f"nonblack={r['share']:6.2f}%  bright={r['bright']:>7}  "
              f"colours={str(r['colours']) + ('+' if r['colours'] >= COLOUR_CAP else ''):>6}  "
              f"{r['kind']}")
    print("  " + "  ".join(f"{k}={v}" for k, v in sorted(tally.items())) + f"  total={len(paths)}")

if __name__ == "__main__":
    # Exit 0 for any completed classification -- this reports, it does not gate. Non-zero is
    # reserved for the tool itself failing: bad usage, no input, or a --selftest regression.
    sys.exit(main(sys.argv) or 0)
