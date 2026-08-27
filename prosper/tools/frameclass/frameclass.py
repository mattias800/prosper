#!/usr/bin/env python3
"""Classify captured frames: is this picture CONTENT, a flat clear, or UI on black?

Why this exists. `tools/screenshot` already reports how many frames are distinct, which answers
"is the picture changing". It does not answer "is there anything IN the picture", and the obvious
way to answer that by hand is wrong in a way that reads as success:

    im.resize((160, 90)); len(set(im.getdata()))     # <- destroys thin UI

A 4K frame downsampled to 160x90 loses white menu text on a black background entirely, so a
correctly-rendered main menu is reported as "flat black, 0 colours" and a title gets written up as
not rendering. That happened on Stray (PPSA02101) on 2026-08-27: the frames called flat black
contained START GAME / SETTINGS / CREDITS and a legible build stamp, at full brightness.

The other half is the opposite error: a frame where every pixel is non-black is not necessarily
content. A flat WHITE clear is 100% non-black and one colour, and any "fraction of non-black
pixels" test scores it as a perfect frame.

So classify on three numbers that disagree with each other, sampled at full resolution:

  max        the brightest channel anywhere. A frame with a 255 in it is not "black" whatever its
             mean says -- a mean is dominated by the background and says nothing about a HUD.
  nonblack%  the share of sampled pixels above the noise floor.
  colours    distinct colours, counted BEFORE any resize. Flat fills have one or two whatever
             their brightness.

  FLAT          max <= 8, or <= 8 distinct colours -- a clear, black or white or any other.
                (A real 4K picture has thousands; a clear with letterbox bars has two or three.)
  UI-ON-BLACK   bright pixels exist but cover under 2% -- a HUD or menu over an absent world.
                This is the case the naive method reports as "black", and it is usually the most
                interesting one: it means the guest is drawing and the WORLD is what is missing.
  SPARSE        some non-black content, nothing bright -- a fade, or a very dark scene.
  CONTENT       a real picture.

Exit status is 0 always; this is a reporting tool, not a gate. Compare runs by reading the table.
"""
import sys, os, glob
try:
    from PIL import Image
except ImportError:
    sys.exit("frameclass: needs Pillow (python3 -m pip install --user pillow)")

def classify(path, step=4, noise=8, bright_at=128):
    im = Image.open(path).convert("RGB")
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
            if len(cols) < 4096: cols.add((r, g, b))
    share = nonblack / n if n else 0.0
    if mx <= noise or len(cols) <= 8:      kind = "FLAT"
    elif share < 0.02 and bright > 0:      kind = "UI-ON-BLACK"
    elif share > 0.10:                     kind = "CONTENT"
    else:                                  kind = "SPARSE"
    return dict(name=os.path.basename(path), max=mx, share=100 * share,
                bright=bright, colours=len(cols), kind=kind, w=w, h=h)

def main(argv):
    if len(argv) < 2:
        sys.exit("usage: frameclass.py <dir-of-pngs | file.png> [...]")
    paths = []
    for a in argv[1:]:
        paths.extend(sorted(glob.glob(os.path.join(a, "*.png"))) if os.path.isdir(a) else [a])
    if not paths:
        sys.exit("frameclass: no PNGs found")
    tally = {}
    for p in paths:
        r = classify(p)
        tally[r["kind"]] = tally.get(r["kind"], 0) + 1
        print(f"{r['name'][-28:]:>28}  {r['w']}x{r['h']}  max={r['max']:>3}  "
              f"nonblack={r['share']:6.2f}%  bright={r['bright']:>7}  "
              f"colours={r['colours']:>5}  {r['kind']}")
    print("  " + "  ".join(f"{k}={v}" for k, v in sorted(tally.items())) + f"  total={len(paths)}")

if __name__ == "__main__":
    main(sys.argv)
