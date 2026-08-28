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
from letterbox import bars, hud_displacement                     # noqa: E402

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

    Order matters. This deliberately does NOT consult a collapse test first: a frame carrying the
    HP bar is by definition not a collapse, and the structure floor that detects collapses has only
    ~3 units of margin against this title's field frames (min sigma 14.67 against a 12.0 floor,
    with the worst collapse at 11.71). Gating on it would make the published counts depend on that
    margin for no gain -- the HP bar already separates by ~1.5x on both sides. Cinematic bars are
    still a veto, because the title draws the HUD under them during some scripted sequences.
    """
    top, bot, _ = bars(path)
    if top >= 0.04 and bot >= 0.04:      # cinematic bars: the title took control away
        return False
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


def cmd_locomotion(args):
    """Windows are given as start:end:label. A guard band is subtracted from each window's end
    because the character coasts for a moment after the stick is released -- without it the two
    neutral windows that follow a release pick up that coast and the separation blurs. The band is
    an argument so it cannot hide inside the result.
    """
    field = [(t, p) for t, p in load(args.shots) if in_field(p)]
    print(f"field-state frames: {len(field)}\n")
    img0 = Image.open(field[0][1])
    box = _px(MINIMAP_BOX, img0.size)
    rows = []
    for spec in args.window:
        lo, hi, label = spec.split(":")
        lo, hi = float(lo), float(hi) - args.guard
        inside = [(t, p) for t, p in field if lo <= t <= hi]
        if len(inside) < 2:
            print(f"  {label:<10} {lo:6.0f}-{hi:<6.0f}  too few field frames ({len(inside)})")
            continue
        d = hud_displacement(inside[0][1], inside[-1][1], box)
        rows.append((label, d))
        print(f"  {label:<10} {lo:6.0f}-{hi:<6.0f}  displacement {d:6.1f} px  "
              f"({len(inside)} frames)")
    if rows:
        for group in sorted({l for l, _ in rows}):
            vals = [d for l, d in rows if l == group]
            moved = sum(1 for v in vals if v >= args.moved)
            print(f"\n  {group:<10} {moved}/{len(vals)} windows moved >= {args.moved} px, "
                  f"median {statistics.median(vals):.1f} px")
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

    # A frame shaped like a real field frame: a dim world, a minimap disc, and an HP bar of the
    # real element's size. Deliberately not "black except the bar" -- an empty frame trips the
    # cinematic-bar scan, which would make this case pass or fail for a reason unrelated to the
    # marker it is meant to pin.
    rng0 = np.random.default_rng(3)
    frame = (rng0.random((h, w, 3)) * 30).astype(np.uint8)      # dim but structured world
    mx0, my0, mx1, my1 = _px(MINIMAP_BOX, (w, h))
    frame[my0:my1, mx0:mx1] = (120, 150, 90)                     # minimap disc, roughly
    x0, y0, x1, y1 = _px(PARTY_BOX, (w, h))
    bar_h = int((y1 - y0) * 0.09)
    frame[y0 + bar_h * 3: y0 + bar_h * 4, x0 + 100: x1 - 60] = (60, 200, 60)
    p = save(frame)
    frac = hp_bar_fraction(p)
    check(frac >= HP_BAR_MIN, f"a drawn HP bar clears the threshold ({frac:.4f} >= {HP_BAR_MIN})")
    check(in_field(p), "...and the frame classifies as field state")

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
    barred = frame.copy()
    barred[: int(h * 0.10)] = 0
    barred[-int(h * 0.10):] = 0
    check(not in_field(save(barred)), "cinematic bars veto the field state even with the HUD up")

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
    s.add_argument("--moved", type=float, default=2.0)
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
