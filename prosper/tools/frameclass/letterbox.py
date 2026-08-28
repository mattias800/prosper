#!/usr/bin/env python3
"""letterbox — is this frame cinematic (bars top and bottom) or is the player driving?

Why this and not a motion metric: a cutscene moves the picture, so motion cannot separate "the
scene is playing" from "the player is walking". Cinematic BARS can, because a title draws them only
while it has taken control away. And they are geometric rather than chromatic, so they survive the
colour degradation that makes every other measurement on this title unreliable -- a black row is
black whatever the composite does to the middle of the frame.

Reports the bar height as a FRACTION of the frame so the answer is resolution-independent, and
prints the darkness it measured so a frame that is simply black all over cannot masquerade as a
cinematic one.

Two things to know before trusting an answer, both learned the hard way on PPSA17942:

1. **A collapsed WHITE present has no dark rows, so it reads as "no bars" — i.e. as gameplay.**
   The first version of this guarded only against darkness and reported 61 un-barred world frames
   on a run that never left its cutscene; every one was blown to RGB(255,255,255). Guarding one
   end of the range catches half the failure and INVERTS the other half. Hence `uniform()`.

2. **Absence of bars is not presence of gameplay.** A menu has no bars either. Pair this with a
   positive test — for that title, the field HUD it draws only while the player has control — and
   require both. On PPSA17942 the conjunction separates cleanly where neither half does:
   the world's colourful cutscene frames pass a HUD-colour test on their own, and its
   white-collapsed gameplay frames pass a bar test on their own.

And it only means anything on a title that actually bars its cutscenes. Confirm that before
reading a bar count as a cutscene count.
"""
import argparse, glob, os, sys

import numpy as np
from PIL import Image


def uniform(path, width=160, max_colours=3):
    """A collapsed present: essentially one colour, whether black OR blown to white.

    The white case is the one that bites. A frame blown to RGB(255,255,255) has no dark rows, so a
    bar detector reports "no bars" and the frame reads as GAMEPLAY when it carries no picture at
    all. Guarding only against darkness catches half the failure and inverts the other half.
    """
    img = Image.open(path).convert("RGB")
    h = max(1, round(img.height * width / img.width))
    arr = np.asarray(img.resize((width, h), Image.NEAREST)).reshape(-1, 3)
    return len(np.unique(arr, axis=0)) <= max_colours


def bars(path, dark=12, width=480):
    """(top_frac, bottom_frac, mid_mean) — how much of the frame is a dark bar, and how bright the
    middle is. A frame that is dark everywhere returns a low mid_mean and must be discarded by the
    caller: it is a collapsed present, not a cinematic."""
    img = Image.open(path).convert("L")
    h = max(1, round(img.height * width / img.width))
    arr = np.asarray(img.resize((width, h), Image.BILINEAR), dtype=np.float32)
    rowmax = arr.max(axis=1)                      # a bar is dark ACROSS the whole row
    top = 0
    while top < len(rowmax) and rowmax[top] <= dark:
        top += 1
    bot = 0
    while bot < len(rowmax) and rowmax[len(rowmax) - 1 - bot] <= dark:
        bot += 1
    mid = arr[top:len(arr) - bot] if top + bot < len(arr) else arr
    return top / len(rowmax), bot / len(rowmax), float(mid.mean()) if mid.size else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shots")
    ap.add_argument("--seconds-per-frame", type=float, default=1.0)
    ap.add_argument("--min-bar", type=float, default=0.04,
                    help="bar fraction on BOTH edges to call a frame cinematic")
    ap.add_argument("--min-mid", type=float, default=8.0,
                    help="minimum mean brightness between the bars; below this the frame is a "
                         "collapsed/black present and is reported as UNUSABLE rather than as "
                         "either answer")
    ap.add_argument("--show", action="store_true", help="print every frame, not just the summary")
    args = ap.parse_args()

    frames = sorted(glob.glob(os.path.join(args.shots, "*.png")))
    if not frames:
        raise SystemExit(f"letterbox: no PNGs in {args.shots}")

    open_frames, cine, unusable = [], 0, 0
    for i, path in enumerate(frames):
        t = i * args.seconds_per_frame
        top, bot, mid = bars(path)
        if uniform(path) or mid < args.min_mid:
            unusable += 1
            state = "unusable"
        elif top >= args.min_bar and bot >= args.min_bar:
            cine += 1
            state = "cinematic"
        else:
            open_frames.append((t, top, bot, mid, os.path.basename(path)))
            state = "OPEN-FRAME"
        if args.show:
            print(f"  t={t:6.0f}s  top={top:.3f} bot={bot:.3f} mid={mid:6.1f}  {state}")

    print(f"\n{len(frames)} frames: {cine} cinematic, {len(open_frames)} open-frame, "
          f"{unusable} unusable (too dark to judge)")
    if open_frames:
        print("\nframes with NO cinematic bars — the candidates for player control:")
        for t, top, bot, mid, name in open_frames[:40]:
            print(f"  t={t:6.0f}s  top={top:.3f} bot={bot:.3f} mid={mid:6.1f}  {name}")
        if len(open_frames) > 40:
            print(f"  ... and {len(open_frames) - 40} more")
    else:
        print("\nEvery usable frame is letterboxed. On a title that bars its cutscenes this is "
              "evidence\nthe run never left the scripted sequence — but confirm the title DOES bar "
              "its cutscenes\nbefore reading it that way, or the absence proves nothing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
