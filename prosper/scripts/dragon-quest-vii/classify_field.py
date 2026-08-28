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
      collapse -- and its structure floor is not trustworthy here in any case: a genuine collapse
      in this very run (frame 188: three distinct colours, nothing rendered) measures sigma 13.14
      and sits OVER the 12.0 floor, while its neighbour 195 measures 11.56 and is caught. Do not
      describe the distance from a threshold to the nearest real frame as "margin" -- it is
      bounded by whichever collapse happens to sit just under the line.
    * A **cinematic-bar veto.** A no-op only ABOVE a bar threshold of ~0.052 (0 frames rejected at
      0.06 and 0.10, across all four runs), because a real cinematic's bars cover the party block
      and the marker already rejects those frames. At the 0.04 it actually shipped with it was
      worse than useless: it rejected nine genuine "Pilchard Bay: Church" frames whose *world* had
      collapsed to black, reading unrendered darkness as letterboxing -- 190 field frames became
      181. That is the same inversion that makes a brightness floor useless here. Its test case
      also passed for the wrong reason, since drawing 30% bars over a synthetic frame erases the
      very HP bar the case depended on.

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
    """Pins the thresholds and the boxes against constructed frames.

    Every case here exists because a review showed the previous version could not fail. Two traps
    are worth naming, because both are easy to walk back into:

    * **A control drawn FROM the box under test cannot pin that box.** The first version sized its
      HP bar as a fraction of PARTY_BOX, so the measured area fraction came out at the target for
      ANY box extent -- the case was arithmetically incapable of noticing a box change. Controls
      here are drawn at fixed PIXEL positions and sizes, and the expected fraction is asserted
      numerically, so moving or resizing a box moves the measurement off it.
    * **A control that also satisfies the mutation cannot pin the operation.** The minimap case
      compares a bright disc against a dark one, where a signed mean is as large as an absolute
      one -- so it could not see `abs` being removed. The case below changes equal areas up and
      down, which cancels under a signed mean.
    """
    import tempfile
    ok = True

    def check(cond, what):
        nonlocal ok
        print(f"  {'ok  ' if cond else 'FAIL'} {what}")
        ok = ok and bool(cond)

    h, w = 2160, 3840
    rng = np.random.default_rng(3)

    def save(arr):
        fh = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        Image.fromarray(arr).save(fh.name)
        return fh.name

    def field_frame(bar_px=None):
        """A frame shaped like a real one: dim structured world, minimap disc, and optionally an
        HP bar of an exact PIXEL size at a fixed screen position."""
        f = (rng.random((h, w, 3)) * 30).astype(np.uint8)
        cy, cx, r = int(h * 0.835), int(w * 0.086), int(h * 0.113)
        yy, xx = np.mgrid[0:h, 0:w]
        f[((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r] = (120, 150, 90)
        if bar_px:
            bw, bh = bar_px
            # Inside PARTY_BOX (x from 0.78w, y from 0.77h) with room for the widest bar
            # used below -- an overflowing bar is silently clipped and measures short.
            by, bx = int(h * 0.905), int(w * 0.810)
            f[by:by + bh, bx:bx + bw] = (60, 200, 60)
        return f

    # ---- the marker, pinned at the REAL element's measured area ------------------------------
    # PARTY_BOX spans 0.22w x 0.23h = 844 x 496 px at 4K. A bar of 522 x 20 px is 10440/418624 =
    # 0.02494 of it. Asserted numerically: change the box extent and this number moves.
    px = _px(PARTY_BOX, (w, h))
    box_area = (px[2] - px[0]) * (px[3] - px[1])
    # HARDCODED, not derived from PARTY_BOX. Deriving it reproduces the very circularity this case
    # exists to remove: widen the box and both the measurement and its expectation move together.
    # 522x20 px inside the box PARTY_BOX describes at 3840x2160 is 10440 / 418624 = 0.024939.
    frac_target = 0.024939
    p_field = save(field_frame(bar_px=(522, 20)))
    frac = hp_bar_fraction(p_field)
    check(abs(frac - frac_target) < 0.002,
          f"a fixed-size HP bar measures the area PARTY_BOX implies ({frac:.5f} vs "
          f"{frac_target:.5f}) -- moving or resizing the box moves this")
    check(in_field(p_field), "...and the frame classifies as field state")

    # Both sides of the threshold, built at the REAL measured values so a retune in either
    # direction reddens. 0.0201 is the field class; 0.008860 is the worst non-field frame across
    # all four runs.
    for name, target, want in (("the field class (0.0201)", 0.0201, True),
                               ("the worst NON-field value (0.008860)", 0.008860, False)):
        bw = 522
        # Round UP. Rounding down builds the case just under the value it names, which is how the
        # previous version left HP_BAR_MIN=0.00875 passing while it changed a published count.
        bh = max(1, -(-int(target * box_area) // bw))
        got = hp_bar_fraction(save(field_frame(bar_px=(bw, bh))))
        # Build at or ABOVE the named value, never under it, or the case cannot pin the boundary.
        check((got >= target * 0.99) and (in_field_frac(got) == want),
              f"a bar at {name} classifies as {'field' if want else 'not field'} "
              f"(measured {got:.5f})")

    # ---- collapses -----------------------------------------------------------------------
    check(not in_field(save(np.full((h, w, 3), 255, np.uint8))), "a blown-white collapse is not field")
    blue = np.zeros((h, w, 3), np.uint8); blue[:, :] = (0, 0, 255)
    check(not in_field(save(blue)), "a flat blue collapse is not field")

    # Saturated structured water in the corners -- the generic corner test's false positive.
    water = np.zeros((h, w, 3), np.uint8)
    water[:, :, 2] = 120 + (rng.random((h, w)) * 90).astype(np.uint8)
    water[:, :, 1] = 40 + (rng.random((h, w)) * 40).astype(np.uint8)
    check(not in_field(save(water)), "a torn-composite cutscene (structured water) is not field")

    # ---- world_renders: BOTH sides of all three thresholds --------------------------------
    block = 24
    tiles = (rng.random((h // block + 1, w // block + 1, 3)) * 235 + 10).astype(np.uint8)
    scene = np.repeat(np.repeat(tiles, block, axis=0), block, axis=1)[:h, :w]
    check(world_renders(save(scene)), "a structured world reads as rendered")
    check(not world_renders(save(np.full((h, w, 3), 255, np.uint8))), "a blown-white world does not")
    check(not world_renders(save(np.zeros((h, w, 3), np.uint8))), "a black world does not")
    check(not world_renders(save(blue)), "a flat blue world does not")
    # Just-too-few colours, and just-too-flat: these pin the two thresholds a global test misses.
    few = np.repeat(np.repeat((rng.random((6, 6, 3)) * 200 + 25).astype(np.uint8),
                              h // 6 + 1, axis=0), w // 6 + 1, axis=1)[:h, :w]
    check(not world_renders(save(few)), "a world of only a few large flat blocks does not")
    faintish = (128 + (rng.random((h, w, 3)) * 6 - 3)).astype(np.uint8)
    check(not world_renders(save(faintish)), "a nearly uniform mid-grey world does not")

    # BETWEEN the thresholds, so tightening either one reddens. A real frame is not saturated in
    # usable-range nor unlimited in colour: this one is ~0.6 usable with ~700 colours.
    mid = np.zeros((h, w, 3), np.uint8)
    mtiles = (rng.random((28, 28, 3)) * 170 + 40).astype(np.uint8)
    mid[:] = np.repeat(np.repeat(mtiles, h // 28 + 1, axis=0), w // 28 + 1, axis=1)[:h, :w]
    mid[: int(h * 0.35)] = 255                      # a blown band drags usable down to ~0.6
    u_mid = np.asarray(Image.fromarray(mid).crop(_px(WORLD_BOX, (w, h))).resize((256, 144)))
    lum_mid = u_mid.reshape(-1, 3).mean(axis=1)
    check(world_renders(save(mid)),
          f"a partly-blown world still reads as rendered "
          f"(usable {float(((lum_mid > 25) & (lum_mid < 245)).mean()):.2f}, "
          f"colours {len(np.unique(u_mid.reshape(-1, 3) // 8, axis=0))}) -- pins BOTH thresholds "
          f"from the loose side")

    # WORLD_BOX itself: a frame whose WORLD region is a scene and whose HUD corners are flat. If the
    # box moves onto the HUD, this stops reading as rendered.
    # HARDCODED region, NOT _px(WORLD_BOX, ...). Painting the scene into the box under test moves
    # the control with the box and the case can never fail -- the same circularity PARTY_BOX had.
    # These are the pixels WORLD_BOX describes at 3840x2160: x 691..3148, y 108..1555.
    split = np.zeros((h, w, 3), np.uint8)
    split[:] = (10, 10, 10)
    split[108:1555, 691:3148] = scene[108:1555, 691:3148]
    check(world_renders(save(split)),
          "a scene confined to the WORLD region reads as rendered (pins WORLD_BOX's position)")

    # ---- minimap_change: the disc mask AND the absolute value -----------------------------
    box_m = _px(MINIMAP_BOX, (w, h))
    base = field_frame()
    # (a) change only OUTSIDE the disc: the mask must reject it.
    outside = base.copy()
    outside[int(h * 0.72):int(h * 0.95), int(w * 0.135):int(w * 0.150)] = (255, 255, 255)
    check(minimap_change(save(base), save(outside), box_m) < 5,
          "a change outside the disc does not count as movement (the mask is load-bearing)")
    # (b) equal areas brighter and darker: a SIGNED mean cancels, an absolute one does not.
    signed = base.copy()
    cy, cx, r = int(h * 0.835), int(w * 0.086), int(h * 0.113)
    yy, xx = np.mgrid[0:h, 0:w]
    inside = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r
    upper = inside & (yy < cy)
    lower = inside & (yy >= cy)
    signed[upper] = (220, 220, 220)
    signed[lower] = (20, 20, 20)
    check(minimap_change(save(base), save(signed), box_m) > 15,
          "equal brightening and darkening still counts as change (pins the ABSOLUTE value)")

    # A FINE-GRAINED change: alternating 8 px stripes inside the disc. At the working resolution
    # this is plainly a change; downscale far enough and it averages to nothing, so this pins
    # `size` against being reduced.
    striped = base.copy()
    stripe = (yy // 8) % 2 == 0
    striped[inside & stripe] = (240, 240, 240)
    striped[inside & ~stripe] = (15, 15, 15)
    check(minimap_change(save(base), save(striped), box_m) > 15,
          "a fine-grained change is still seen (pins the sampling size against being coarsened)")

    print("  == selftest PASS ==" if ok else "  == selftest FAIL ==")
    return 0 if ok else 1


def in_field_frac(frac):
    """Whether an already-measured HP-bar fraction counts as the field state."""
    return frac >= HP_BAR_MIN


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
