#!/usr/bin/env python3
"""classify_field — every published number for PPSA17942's field state, from the captures.

Committed because the alternative does not work: the figures in `docs/DRAGON_QUEST_STATUS.md` were
twice published from scratch analysis that nobody else could re-run, and twice they were wrong in
ways a re-run would have caught. Everything this title claims is derived here, with the thresholds
visible and a `--selftest` that pins them.

Three questions, three subcommands:

    field       how many frames are in the FIELD STATE, and over what span
    world       how much of the field renders a scene rather than collapsing
    locomotion  does the pad MOVE the player -- measured on the minimap, not the world

Why the marker is the HP bar and not "does it look like a HUD"
--------------------------------------------------------------
A generic "is there HUD art in the bottom corners" test fails on this title in both directions, and
both failures are real captures, not hypotheticals:

  * a torn-composite CUTSCENE whose bottom corners hold saturated, structured water passes it
    (this produced two false "gameplay" frames in runs 1 and 2, published as 1/1 before they were
    opened and found to be harbour cutscenes with no HUD at all);
  * a present collapsed to flat blue scores 1.00 saturated while containing nothing.

The party block's **HP bar** has neither problem. It is a fixed-size green UI element the title
draws only in the field, so its area fraction is essentially constant when present and zero when
not. Measured over all 950 frames of three runs:

    the 144 field frames      min = p25 = median = 0.0201, max 0.1871
    every other frame         max 0.0089, second highest 0.000321

`HP_BAR_MIN = 0.013` sits ~1.5x below the class and ~1.5x above everything else. Compare the
brightness floor it replaced, where 73 of 144 field frames sat within 1.0 of the threshold and the
nearest collapse was 0.26 away -- the frames are DARK precisely because they are HUD over an
unrendered world, so brightness was measuring the defect rather than the state.
"""
import argparse, glob, json, os, statistics, sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "tools", "frameclass"))

HP_BAR_MIN = 0.013
# The party block and the minimap disc, as fractions of the frame so they survive a resolution
# change. Derived by eye from a 3840x2160 capture and pinned by --selftest.
PARTY_BOX = (0.78, 0.77, 1.00, 1.00)
MINIMAP_BOX = (0.022, 0.722, 0.150, 0.949)
WORLD_BOX = (0.18, 0.05, 0.82, 0.72)


def _px(box, size):
    w, h = size
    return (int(w * box[0]), int(h * box[1]), int(w * box[2]), int(h * box[3]))


def hp_bar_fraction(path):
    """Area fraction of the party block that is the HP bar's green."""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img.crop(_px(PARTY_BOX, img.size))).astype(np.int16)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    return float(((g > 90) & (g - r > 25) & (g - b > 25)).mean())


def in_field(path):
    """Is the guest in the field state -- player control, HUD up, not a cutscene?

    One test, deliberately. Two others were tried and both removed:

    * A **collapse test first.** Unnecessary -- a frame carrying the HP bar is by definition not a
      collapse -- and its structure floor has only ~3 units of margin here, so it would have made
      the published counts depend on a threshold the marker does not need.
    * A **cinematic-bar veto.** Measured a no-op across all four runs (0/0/144/190 with and
      without), because a real cinematic's bars are deep enough to cover the party block, so the
      marker already rejects those frames. Worse than useless while it was tight: at 0.04 it
      rejected nine genuine "Pilchard Bay: Church" frames whose *world* had collapsed to black,
      reading unrendered darkness as letterboxing -- the same inversion that makes a brightness
      floor useless on this title. Its test case also passed for the wrong reason, since drawing
      30% bars over a synthetic frame erases the very HP bar the case depended on.

    So: the HP bar, and nothing else. It separates 0.0201 from 0.0089 across 1,370 frames.
    """
    return hp_bar_fraction(path) >= HP_BAR_MIN


def world_renders(path):
    """Does the WORLD carry a scene, as opposed to a HUD over a collapse?

    Deliberately three conditions rather than one: the composite fails in three different ways here
    (blown white, crushed black, flat blue speckle) and no single statistic catches all three.
    """
    img = Image.open(path).convert("RGB")
    box = _px(WORLD_BOX, img.size)
    region = img.crop(box)
    arr = np.asarray(region.resize((256, 144))).reshape(-1, 3)
    lum = arr.mean(axis=1)
    usable = float(((lum > 25) & (lum < 245)).mean())
    colours = len(np.unique(arr // 8, axis=0))
    sigma = float(np.asarray(region.convert("L"), dtype=np.float32).std())
    return usable > 0.30 and colours > 200 and sigma > 20


def load(shots):
    """[(elapsed_seconds, path)] from the run's own manifest.

    The manifest, never `index * interval`: tools/screenshot captures index 0 one interval in, so
    the ordinal form understates every timestamp by a frame. That is how a 588 s span was once
    published as 636 s.
    """
    manifests = glob.glob(os.path.join(shots, "*.jsonl"))
    if not manifests:
        raise SystemExit(f"classify_field: no .jsonl manifest in {shots}; it carries the "
                         f"elapsed_seconds this tool refuses to guess")
    times = {}
    for line in open(manifests[0], encoding="utf-8"):
        rec = json.loads(line)
        if rec.get("type") == "sample":
            times[os.path.basename(rec["png"])] = rec["elapsed_seconds"]
    out = []
    for path in sorted(glob.glob(os.path.join(shots, "*.png"))):
        t = times.get(os.path.basename(path))
        if t is not None:
            out.append((t, path))
    return out


def cmd_field(args):
    frames = [(t, p) for t, p in load(args.shots) if t >= args.after]
    field = [(t, p) for t, p in frames if in_field(p)]
    print(f"{len(frames)} frames at or after t={args.after:.0f}s")
    print(f"  field-state frames : {len(field)}")
    if field:
        print(f"  span               : t={field[0][0]:.1f}..{field[-1][0]:.1f}s "
              f"({field[-1][0] - field[0][0]:.0f}s)")
    return 0


def cmd_world(args):
    field = [(t, p) for t, p in load(args.shots) if t >= args.after and in_field(p)]
    if not field:
        print("no field-state frames")
        return 0
    rendered = [(t, p) for t, p in field if world_renders(p)]
    print(f"field-state frames        : {len(field)}")
    print(f"  with a rendered scene   : {len(rendered)} "
          f"({100 * len(rendered) / len(field):.0f}%)")
    best = run = 0
    times = [t for t, _ in field]
    ok = {t for t, _ in rendered}
    span_lo = span_hi = None
    lo = None
    for t in times:
        if t in ok:
            lo = t if run == 0 else lo
            run += 1
            if run > best:
                best, span_lo, span_hi = run, lo, t
        else:
            run = 0
    print(f"  longest STRICTLY consecutive rendered run: {best} samples"
          + (f" (t={span_lo:.0f}..{span_hi:.0f}s)" if span_lo is not None else ""))
    return 0


def minimap_change(path_a, path_b, box, size=128, disc=0.44):
    """How much the minimap's INTERIOR differs between two frames.

    Not phase correlation, which two earlier drafts used and which fails at both ends here:
    a large walk changes WHICH PART of the map is shown, so the two crops share almost no common
    structure and the correlation peaks at zero -- reporting a long walk as "did not move". Two of
    eight stick windows read exactly 0.0 px that way while their minimaps were plainly of different
    places.

    The disc mask is not cosmetic either. The minimap is a CIRCLE in a square crop, and the corners
    show the world behind it -- which on this title collapses to black in one frame and white in the
    next. Unmasked, that background flip alone scores 56 on a window whose map is pixel-identical,
    which is larger than most real movements.
    """
    S = size
    yy, xx = np.mgrid[0:S, 0:S]
    mask = ((yy - (S - 1) / 2) ** 2 + (xx - (S - 1) / 2) ** 2) <= (S * disc) ** 2

    def region(p):
        im = Image.open(p).convert("L").crop(box).resize((S, S), Image.BILINEAR)
        return np.asarray(im, dtype=np.float32)[mask]

    return float(np.mean(np.abs(region(path_a) - region(path_b))))


def cmd_locomotion(args):
    """Windows are given as start:end:label.

    `--guard` trims the end of each window. It does NOT change which neutral windows register
    movement -- swept 0 to 6 the counts are identical -- it changes the reported magnitudes, and it
    is kept only so a window cannot include the frame in which the stick was released. Stated
    because an earlier draft claimed the neutral result depended on it, which running the tool
    refutes.
    """
    field = [(t, p) for t, p in load(args.shots) if in_field(p)]
    print(f"field-state frames: {len(field)}\n")
    box = _px(MINIMAP_BOX, Image.open(field[0][1]).size)
    rows = []
    for spec in args.window:
        lo, hi, label = spec.split(":")
        lo, hi = float(lo), float(hi) - args.guard
        inside = [(t, p) for t, p in field if lo <= t <= hi]
        if len(inside) < 2:
            print(f"  {label:<10} {lo:6.0f}-{hi:<6.0f}  too few field frames ({len(inside)})")
            continue
        d = minimap_change(inside[0][1], inside[-1][1], box)
        rows.append((label, d))
        print(f"  {label:<10} {lo:6.0f}-{hi:<6.0f}  minimap change {d:7.2f}  ({len(inside)} frames)")
    if rows:
        print()
        for group in sorted({l for l, _ in rows}):
            vals = [d for l, d in rows if l == group]
            moved = sum(1 for v in vals if v >= args.moved)
            print(f"  {group:<10} {moved}/{len(vals)} windows changed >= {args.moved}, "
                  f"median {statistics.median(vals):.2f}, "
                  f"min {min(vals):.2f}, max {max(vals):.2f}")
    return 0


def selftest():
    """Pins the thresholds against constructed frames. The point is the MARGIN: each case sits
    clearly on its side, so a small retune does not silently flip the published counts."""
    import tempfile
    ok = True

    def check(cond, what):
        nonlocal ok
        print(f"  {'ok  ' if cond else 'FAIL'} {what}")
        ok = ok and bool(cond)

    h, w = 2160, 3840

    def save(arr):
        fh = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        Image.fromarray(arr).save(fh.name)
        return fh.name

    # A frame shaped like a real field frame. Two things this case must NOT do, both learned from
    # review: it must not be "black except the bar" (an empty frame trips the bar scan, so the case
    # would pass or fail for a reason unrelated to the marker), and the bar must be drawn at the
    # REAL element's measured size rather than whatever fills the box -- a bar at 0.07 area cannot
    # see a threshold retune to 0.03, which is inside the range a person might pick and which
    # collapses the published count from 144 to 5.
    rng0 = np.random.default_rng(3)
    frame = (rng0.random((h, w, 3)) * 30).astype(np.uint8)      # dim but structured world
    mx0, my0, mx1, my1 = _px(MINIMAP_BOX, (w, h))
    frame[my0:my1, mx0:mx1] = (120, 150, 90)                     # minimap disc, roughly
    x0, y0, x1, y1 = _px(PARTY_BOX, (w, h))
    # Size the bar so its area fraction lands on the REAL element's 0.0201, not merely above the
    # threshold. Then a retune to 0.03 -- which would silently take run3 from 144 frames to 5 --
    # reddens this case instead of passing it.
    target = 0.0201
    bw = int(round((x1 - x0) * 0.62))
    bh = max(1, int(round(target * (x1 - x0) * (y1 - y0) / bw)))
    frame[y0 + bh * 3: y0 + bh * 4, x0 + (x1 - x0 - bw) // 2: x0 + (x1 - x0 + bw) // 2] = (60, 200, 60)
    p = save(frame)
    frac = hp_bar_fraction(p)
    check(abs(frac - target) < 0.004,
          f"the constructed HP bar matches the real element's area ({frac:.4f} vs {target})")
    check(frac >= HP_BAR_MIN, f"...and clears the threshold ({frac:.4f} >= {HP_BAR_MIN})")
    check(in_field(p), "...and the frame classifies as field state")

    # PARTY_BOX must be pinned by a control NOT drawn inside it, or the case moves with the box and
    # can never fail. Draw the same bar at a FIXED screen position and require the box to find it.
    fixed = (rng0.random((h, w, 3)) * 30).astype(np.uint8)
    fixed[my0:my1, mx0:mx1] = (120, 150, 90)
    fx0, fy0 = int(w * 0.86), int(h * 0.86)
    fixed[fy0:fy0 + 22, fx0:fx0 + 300] = (60, 200, 60)
    check(hp_bar_fraction(save(fixed)) > 0,
          "PARTY_BOX covers the party block's real screen position (fixed control, not "
          "drawn from the box)")

    # The two real failure modes, neither of which may pass.
    blue = np.zeros((h, w, 3), np.uint8)
    blue[:, :] = (0, 0, 255)
    check(not in_field(save(blue)), "a flat blue collapse is not field state")
    check(not in_field(save(np.full((h, w, 3), 255, np.uint8))),
          "a blown-white collapse is not field state")

    # Saturated, structured water in the corners -- the generic corner test's false positive.
    water = np.zeros((h, w, 3), np.uint8)
    rng = np.random.default_rng(5)
    water[:, :, 2] = 120 + (rng.random((h, w)) * 90).astype(np.uint8)
    water[:, :, 1] = 40 + (rng.random((h, w)) * 40).astype(np.uint8)
    wp = save(water)
    check(hp_bar_fraction(wp) < HP_BAR_MIN,
          f"structured blue water carries no HP bar ({hp_bar_fraction(wp):.4f})")
    check(not in_field(wp), "...so a torn-composite cutscene is not field state")

    # Cinematic bars veto even with a HUD present.
    # A deep-barred cinematic must not classify as field. Note this passes because the bars cover
    # the party block, not because of a veto -- there is none. Recorded so nobody "restores" one.
    barred = frame.copy()
    barred[: int(h * 0.30)] = 0
    barred[-int(h * 0.30):] = 0
    check(not in_field(save(barred)), "a deep-barred cinematic is not field state")

    # The marker's LOWER side. Loosening the threshold must not admit the worst non-field frame
    # measured across all four runs (0.0089) -- otherwise a retune downward silently adds frames.
    faint = (rng0.random((h, w, 3)) * 30).astype(np.uint8)
    faint[my0:my1, mx0:mx1] = (120, 150, 90)
    fbw = int(round((x1 - x0) * 0.62))
    fbh = max(1, int(round(0.0089 * (x1 - x0) * (y1 - y0) / fbw)))
    faint[y0 + fbh * 3: y0 + fbh * 4,
          x0 + (x1 - x0 - fbw) // 2: x0 + (x1 - x0 + fbw) // 2] = (60, 200, 60)
    fp = save(faint)
    check(not in_field(fp),
          f"a frame at the worst measured NON-field value ({hp_bar_fraction(fp):.4f}) is rejected")

    # MINIMAP_BOX, pinned by a control at a FIXED screen position rather than one drawn from the
    # box -- the same trap PARTY_BOX had. A disc painted where the real minimap sits must register
    # as changed when it is replaced, and the box must actually cover it.
    mm_a = np.zeros((h, w, 3), np.uint8)
    mm_b = np.zeros((h, w, 3), np.uint8)
    cy, cx = int(h * 0.835), int(w * 0.086)          # real minimap centre, from a 4K capture
    ry, rx = np.mgrid[0:h, 0:w]
    disc_m = ((ry - cy) ** 2 + (rx - cx) ** 2) <= int(h * 0.113) ** 2
    mm_a[disc_m] = (200, 180, 120)
    mm_b[disc_m] = (40, 90, 60)
    box_m = _px(MINIMAP_BOX, (w, h))
    changed = minimap_change(save(mm_a), save(mm_b), box_m)
    check(changed > 15,
          f"MINIMAP_BOX covers the minimap's real screen position (change {changed:.1f})")

    # world_renders carries the 25% headline and nothing tested it.
    # Structured at a scale that SURVIVES the downscale. Two ways to get this wrong, both tried:
    # per-pixel noise averages to flat grey under the resize, and coarse smooth gradients keep too
    # few distinct colours. Real frames measure ~418 colours at sigma ~100, so the case is built
    # from randomised blocks big enough to survive 256x144 and contrasty enough to look like a
    # scene rather than a wash.
    block = 24
    bh_, bw_ = h // block + 1, w // block + 1
    tiles = (rng0.random((bh_, bw_, 3)) * 235 + 10).astype(np.uint8)
    scene = np.repeat(np.repeat(tiles, block, axis=0), block, axis=1)[:h, :w]
    check(world_renders(save(scene)), "a structured world region reads as rendered")
    check(not world_renders(save(np.full((h, w, 3), 255, np.uint8))),
          "a blown-white world does not")
    check(not world_renders(save(np.zeros((h, w, 3), np.uint8))), "a black world does not")
    flat_blue = np.zeros((h, w, 3), np.uint8); flat_blue[:, :] = (0, 0, 255)
    check(not world_renders(save(flat_blue)), "a flat blue world does not")

    print("  == selftest PASS ==" if ok else "  == selftest FAIL ==")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    for name, fn in (("field", cmd_field), ("world", cmd_world)):
        s = sub.add_parser(name)
        s.add_argument("shots")
        s.add_argument("--after", type=float, default=330.0)
        s.set_defaults(func=fn)
    s = sub.add_parser("locomotion")
    s.add_argument("shots")
    s.add_argument("--window", action="append", default=[], metavar="LO:HI:LABEL")
    s.add_argument("--guard", type=float, default=2.0,
                   help="seconds trimmed from each window's end; the character coasts briefly "
                        "after a release and without this the following neutral window inherits it")
    s.add_argument("--moved", type=float, default=15.0,
                   help="minimap change counting as movement; measured separation on PPSA17942 is "
                        "stick 22.4-37.3 against neutral 0.0-12.7")
    s.set_defaults(func=cmd_locomotion)
    s = sub.add_parser("selftest")
    s.set_defaults(func=lambda a: selftest())
    args = ap.parse_args()
    if not getattr(args, "func", None):
        ap.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
