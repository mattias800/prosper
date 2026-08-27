#!/usr/bin/env python3
"""Classify captured frames: is this picture CONTENT, UI over an absent world, or a flat clear?

Why this exists. `tools/screenshot` already writes `distinct_rgb_colors` and `nonblack_rgb_pixels`
to its JSONL manifest, at full resolution, in the same directory as the PNGs -- so "how many colours"
and "how much is lit" are NOT new here. What is new is `max` (the brightest channel anywhere),
`bright` (how much of it is actually legible), and the classification those two make possible:
separating **a HUD or menu drawn over a world that never rendered** from **a real scene**. That
distinction is the difference between rung 2 and rung 3, and no existing tool reports it.

Two ways of answering it by hand are wrong, and both were made here:

- **Downsampling before counting colours destroys thin UI.** A 4K frame resized to 160x90 loses white
  menu text on black entirely. Measured on Stray (`PPSA02101`), 2026-08-27: a frame containing START
  GAME / SETTINGS / CREDITS and a legible build stamp reported `colours=61 FLAT-BLACK mean=(0,0,0)`,
  and the title was one edit from being written up as not rendering.
- **"Fraction of non-black pixels" scores a flat WHITE clear as a perfect frame** -- 100% non-black,
  and every is-it-black test passes it.

The thresholds are tuned against the 128 committed frames in `assets/screenshots/`, which are
human-reviewed milestone evidence and therefore the closest thing to ground truth this repo has. That
corpus puts a clean gap where the boundary belongs: title screens, logos and EULAs cover 0.15-8.62%
of the frame, while the least-covered real scene covers 10.13%. An earlier revision of this tool drew
CONTENT at 10% and UI-ON-BLACK at 2%, which dropped all fourteen lit title screens into a 2-10% dead
band -- including five cited in COMPATIBILITY.md and BLOG.md as their title's milestone. That is the
same failure the tool exists to prevent, one threshold along.

  FLAT          nothing rendered, or a uniform fill: max <= noise, or <= 8 colours covering
                >= 90% of the frame (black clear, white clear, two-tone letterbox).
  UI-ON-BLACK   bright pixels exist and cover under 10% -- a HUD, menu, logo or title over a world
                that did not draw. THE class worth looking for: it means the guest is drawing and
                the WORLD is what is missing.
  SPARSE        some non-black content but nothing legible (bright == 0) -- a fade, or a frame so
                dark nothing in it can be read.
  CONTENT       a real picture.

CAVEAT, and it is not detectable from pixels: `--fps-overlay` burns prosper's OWN counter into the
PNG (opaque white on a translucent box). A black frame carrying only that overlay classifies
UI-ON-BLACK, which reads as "the guest is drawing its menu" when nothing of the sort happened. Do not
classify overlay runs; `screenshot`'s manifest records `run.assertions.fps_overlay` if you need to
check after the fact.

Also not detected: a smooth gradient with no picture in it classifies CONTENT. prosper's seed-miss
gradient is exactly that shape, so a frame can score well here and still be a diagnostic fill.

SAMPLING: a 1/16 nearest-neighbour stride (every 4th pixel on both axes) at NATIVE resolution --
never a resize, which is the distinction the tool rests on. A stride still aliases: a black frame
ruled with 1px vertical lines at `x % 4 == 1`, covering a quarter of the screen, reports
`max=0 colours=1 FLAT` and is indistinguishable from a black clear. Contrived, but real; pass
`step=1` in-process if you need it exact.

The colour count saturates at 4096 and the report marks a saturated count with a trailing `+`, so
two `4096+` rows are not comparable with each other.

Alpha is discarded rather than composited over black. That sounds like a divergence from
`capture_manifest.cpp:105-108`, which multiplies RGB by alpha -- measured on a real `screenshot` PNG
it is not, because `normalize_capture_rgba` forces alpha to 255 for composited and republished
frames (0 of 8,294,400 pixels differed). `CaptureSource::RawScanout` is the one path that leaves
alpha untouched, and no sample of it has been checked.

Always exits 0 on classification -- this reports, it does not gate. An unreadable file is reported on
its own line and does not abort the run or lose the tally.
"""
import sys, os, glob
try:
    from PIL import Image
except ImportError:
    sys.exit("frameclass: needs Pillow (python3 -m pip install --user pillow)")

COLOUR_CAP = 4096  # counting stops here; the report suffixes a saturated count with "+"

def classify(path, step=4, noise=8, bright_at=128):
    return classify_image(Image.open(path).convert("RGB"), os.path.basename(path), step, noise, bright_at)

def classify_image(im, name, step=4, noise=8, bright_at=128):
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
    # Coverage is tested before legibility on purpose: a well-covered frame is a picture even when it
    # is too dim for any pixel to count as bright. Testing `bright == 0` first classified
    # `bendy-title.png` -- 93.33% covered, 2,899 colours, max=127 -- as SPARSE.
    # A few colours only means "flat" when they COVER the frame. Gating on the colour count alone
    # made a black frame carrying prosper's two-colour fps overlay report FLAT despite 6,120 bright
    # pixels, because the colour test short-circuited ahead of UI-ON-BLACK.
    uniform_fill = len(cols) <= 8 and share >= 0.90
    if mx <= noise or uniform_fill:        kind = "FLAT"
    elif share >= 0.10:                    kind = "CONTENT"
    elif bright > 0:                       kind = "UI-ON-BLACK"
    else:                                  kind = "SPARSE"
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
    # A HUD or menu over a world that never drew -- the class the tool exists to find.
    cases.append(("menu text on black", box(frame(), .1, .4, .3, .04, (255, 255, 255)), "UI-ON-BLACK"))
    # 8% coverage: a lit title screen. Drawing UI-ON-BLACK at 2% put fourteen of these in a dead
    # band and reported them SPARSE, five of them cited as their title's milestone.
    cases.append(("lit title screen", box(frame(), 0, 0, 1, .08, (255, 240, 200)), "UI-ON-BLACK"))
    # prosper's own fps overlay is two colours on black. Gating FLAT on the colour count alone
    # swallowed it, and this frame reads UI-ON-BLACK by design -- see the caveat in the module
    # docstring. The assertion pins the documented false positive so it cannot change silently.
    cases.append(("fps overlay only", box(frame(), .01, .01, .07, .03, (255, 255, 255)), "UI-ON-BLACK"))
    # Covered but too dim for any pixel to count as bright: still a picture.
    dim = frame()
    for y in range(H):
        for x in range(W):
            dim.putpixel((x, y), (x % 120, y % 120, (x + y) % 120))
    cases.append(("dim full-frame picture", dim, "CONTENT"))
    scene = frame()
    for y in range(H):
        for x in range(W):
            scene.putpixel((x, y), (x % 256, y % 256, (x * y) % 256))
    cases.append(("bright full-frame scene", scene, "CONTENT"))
    # Non-black but nothing legible anywhere.
    cases.append(("near-black fade", box(frame(), 0, 0, 1, .05, (15, 15, 15)), "SPARSE"))

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
            for ext in ("*.png", "*.bmp", "*.PNG", "*.BMP"):
                paths.extend(sorted(glob.glob(os.path.join(a, ext))))
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
