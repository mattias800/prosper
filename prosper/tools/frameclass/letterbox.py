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


def collapsed(path, width=320, min_sigma=12.0):
    """Is this a collapsed present rather than a picture — of ANY colour, flat or speckled?

    Measured by structure (luma standard deviation), not by colour count and not by brightness.
    Three failures forced that, all on PPSA17942 and all of them silent:

    - **Blown white.** A frame at RGB(255,255,255) has no dark rows, so a bar detector reports
      "no bars" and it reads as GAMEPLAY. Guarding only against darkness inverts this half.
    - **Flat blue with a magenta speckle.** 18 distinct colours across a full 8.3 MP frame, so a
      colour-count guard (this function's previous form, `<= 3`) passes it straight through. It
      was reported as un-barred — i.e. as a candidate for player control — on a run that never
      left its cutscene.
    - **A brightness floor cannot rescue either**, and on this title it is inverted: real
      HUD-over-dark gameplay frames measure mean 8.3-8.9 while the blue garbage measures
      29.7-31.9. Raising the floor removes the real frames first.

    Structure separates all three cleanly: sigma 4.1 on the blue collapse against 95+ on a real
    frame.
    """
    img = Image.open(path).convert("L")
    h = max(1, round(img.height * width / img.width))
    arr = np.asarray(img.resize((width, h), Image.BILINEAR), dtype=np.float32)
    return float(arr.std()) < min_sigma


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
    # An all-dark frame drives both scans to the far end, leaving no middle at all. Left alone the
    # two fractions sum to 2.0 and it reads as an extremely well-barred cinematic. A frame with no
    # middle is not a cinematic, it is nothing — so report NO measurable bars and a dark middle,
    # and let the caller's collapse guard reject it. Returning 1.0/1.0 instead would still satisfy
    # a `top >= x and bot >= x` test, which is the bug this replaced.
    if top + bot >= len(rowmax):
        return 0.0, 0.0, 0.0
    mid = arr[top:len(arr) - bot]
    return top / len(rowmax), bot / len(rowmax), float(mid.mean()) if mid.size else 0.0


def selftest():
    """Constructed cases, built pixel by pixel rather than sampled from a run.

    The folder's sibling does the same and its AGENTS.md says why: a detector tuned against real
    frames from one title can confirm itself. Each case below is a failure this tool actually had.
    """
    import tempfile
    ok = True

    def case(name, arr, want_collapsed=None, want_bars=None):
        nonlocal ok
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
            Image.fromarray(arr).save(fh.name)
            got_c = collapsed(fh.name)
            top, bot, mid = bars(fh.name)
        parts = []
        if want_collapsed is not None and got_c != want_collapsed:
            parts.append(f"collapsed={got_c} want {want_collapsed}")
        if want_bars is not None:
            got_b = top >= 0.04 and bot >= 0.04
            if got_b != want_bars:
                parts.append(f"bars={got_b} want {want_bars} (top={top:.3f} bot={bot:.3f})")
        print(f"  {'ok  ' if not parts else 'FAIL'} {name}"
              + (f"  -- {'; '.join(parts)}" if parts else ""))
        if parts:
            ok = False

    h, w = 216, 384
    rng = np.random.default_rng(11)

    picture = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    case("a noisy picture is not collapsed", picture, want_collapsed=False, want_bars=False)

    barred = picture.copy()
    barred[: int(h * 0.10)] = 0
    barred[-int(h * 0.10):] = 0
    case("a picture with bars reads as cinematic", barred, want_collapsed=False, want_bars=True)

    case("a blown-WHITE present is collapsed",
         np.full((h, w, 3), 255, np.uint8), want_collapsed=True)
    case("an all-black present is collapsed",
         np.zeros((h, w, 3), np.uint8), want_collapsed=True)

    # The case a colour-count guard misses. It must carry MORE than a handful of distinct colours
    # or it does not reproduce the real failure: PPSA17942's blue collapse holds 18 distinct
    # colours across a full 8.3 MP frame, which sails past a `<= 3` test. A two-colour speckle
    # would be caught by the very guard this case exists to prove insufficient.
    speckle = np.zeros((h, w, 3), np.uint8)
    speckle[:, :] = (0, 0, 255)
    for i in range(16):                       # 16 magenta shades + the blue ground = 17 colours
        ys, xs = rng.integers(0, h, 60), rng.integers(0, w, 60)
        speckle[ys, xs] = (255, 0, 255 - i * 3)
    assert len(np.unique(speckle.reshape(-1, 3), axis=0)) > 3, "case must exceed a colour-count guard"
    case("a flat blue SPECKLED collapse is collapsed", speckle, want_collapsed=True)

    # An all-dark frame drives both bar scans to the far end. It must not read as a superbly
    # barred cinematic -- assert on the BARS, or this case cannot see that regression.
    dark = (rng.random((h, w, 3)) * 6).astype(np.uint8)
    case("a noisy near-black frame is collapsed, not cinematic", dark,
         want_collapsed=True, want_bars=False)

    print("  == selftest PASS ==" if ok else "  == selftest FAIL ==")
    return 0 if ok else 1


def corner_structure(path, frac_w=0.22, frac_h=0.28, min_sat=0.02, min_sigma=15.0):
    """The POSITIVE half: is there HUD art in the two bottom corners?

    Absence of bars is not presence of gameplay — a menu has no bars either, and neither does a
    collapsed present. Some titles draw a persistent field HUD in the bottom corners (PPSA17942
    puts a circular minimap bottom-left and a party block bottom-right, neither of which it draws
    during a cutscene), and requiring BOTH halves separates cleanly where neither does alone.

    Keys on saturation AND structure, because either alone is fooled in the opposite direction:
    a colourful CUTSCENE frame is saturated in both corners (measured 0.67 on this title's village
    scene), and a flat saturated collapse scores 1.00 while containing nothing at all.
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size
    cw, ch = int(w * frac_w), int(h * frac_h)
    out = []
    for box in ((0, h - ch, cw, h), (w - cw, h - ch, w, h)):
        arr = np.asarray(img.crop(box), dtype=np.float32)
        mx, mn = arr.max(axis=2), arr.min(axis=2)
        sat = np.where(mx > 40, (mx - mn) / np.maximum(mx, 1.0), 0.0)
        sigma = float(np.asarray(img.crop(box).convert("L"), dtype=np.float32).std())
        out.append(float((sat > 0.25).mean()) >= min_sat and sigma > min_sigma)
    return all(out)


def hud_displacement(path_a, path_b, box, size=128):
    """Rigid translation between the same HUD region in two frames, by phase correlation.

    Why the HUD and not the world: on a title whose composite is broken, the world is the WORST
    place to look for motion — a frame flicking between rendered and collapsed swamps any real
    movement. The HUD is drawn correctly regardless, so a scrolling minimap is a locomotion signal
    the composite defect cannot destroy. On PPSA17942 this separated cleanly where a world-motion
    probe did not: displacement >= 2 px in 7 of 8 stick windows (median 28) against < 2 px in 8 of
    8 neutral windows (seven of them exactly 0).

    `box` is the region in source pixels; it is title-specific by nature — pass the minimap disc.
    """
    def region(p):
        im = Image.open(p).convert("L").crop(box).resize((size, size), Image.BILINEAR)
        return np.asarray(im, dtype=np.float32)
    a, b = region(path_a), region(path_b)
    fa = np.fft.fft2(a - a.mean())
    fb = np.fft.fft2(b - b.mean())
    r = fa * np.conj(fb)
    r /= (np.abs(r) + 1e-9)
    corr = np.fft.ifft2(r).real
    iy, ix = np.unravel_index(np.argmax(corr), corr.shape)
    dy = iy - size if iy > size // 2 else iy
    dx = ix - size if ix > size // 2 else ix
    return float(np.hypot(dx, dy))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shots", nargs="?")
    ap.add_argument("--seconds-per-frame", type=float, default=1.0)
    ap.add_argument("--min-bar", type=float, default=0.04,
                    help="bar fraction on BOTH edges to call a frame cinematic")
    ap.add_argument("--min-mid", type=float, default=8.0,
                    help="minimum mean brightness between the bars; below this the frame is a "
                         "collapsed present and is reported as UNUSABLE rather than as "
                         "either answer")
    ap.add_argument("--show", action="store_true", help="print every frame, not just the summary")
    ap.add_argument("--after", type=float, default=0.0,
                    help="ignore frames before this time. NOT cosmetic: a title's MENUS also have "
                         "no cinematic bars, and on a title whose menus are colourful they pass a "
                         "corner test too. Restricting to the phase you mean is part of the "
                         "measurement -- do it here, with the number stated, rather than by "
                         "trimming the output afterwards.")
    ap.add_argument("--require-hud-corners", action="store_true",
                    help="an un-barred frame counts as GAMEPLAY only if it also carries HUD art in "
                         "both bottom corners. Use on a title that draws a persistent field HUD "
                         "there; without it this tool reports absence of a cutscene, which a menu "
                         "satisfies too.")
    ap.add_argument("--selftest", action="store_true",
                    help="run the constructed cases; needs no captures")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    if not args.shots:
        raise SystemExit("letterbox: a shots directory is required (or --selftest)")
    frames = sorted(glob.glob(os.path.join(args.shots, "*.png")))
    if not frames:
        raise SystemExit(f"letterbox: no PNGs in {args.shots}")

    open_frames, cine, unusable = [], 0, 0
    for i, path in enumerate(frames):
        # The first capture happens one interval in, not at zero -- tools/screenshot's own manifest
        # records index 0 at elapsed_seconds == the interval. Reporting i*interval understates every
        # timestamp by one frame, which is how a 588 s span got published as 636 s.
        t = (i + 1) * args.seconds_per_frame
        if t < args.after:
            continue
        top, bot, mid = bars(path)
        if collapsed(path) or mid < args.min_mid:
            unusable += 1
            state = "unusable"
        elif top >= args.min_bar and bot >= args.min_bar:
            cine += 1
            state = "cinematic"
        elif args.require_hud_corners and not corner_structure(path):
            unusable += 1
            state = "open-no-hud"
        else:
            open_frames.append((t, top, bot, mid, os.path.basename(path)))
            state = "GAMEPLAY" if args.require_hud_corners else "OPEN-FRAME"
        if args.show:
            print(f"  t={t:6.0f}s  top={top:.3f} bot={bot:.3f} mid={mid:6.1f}  {state}")

    print(f"\n{len(frames)} frames: {cine} cinematic, {len(open_frames)} open-frame, "
          f"{unusable} unusable (collapsed — blown white, crushed black, or a flat speckle)")
    if open_frames:
        label = ("frames with the field HUD and no cinematic bars — player control:"
                 if args.require_hud_corners
                 else "frames with NO cinematic bars — candidates, but a menu qualifies too:")
        print("\n" + label)
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
