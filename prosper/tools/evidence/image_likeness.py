#!/usr/bin/env python3
"""Colour-aware image likeness against a reference (oracle) frame.

Why this exists (#2790): every metric this project reached for by default --
non-black percentage, distinct-colour count, near-white fraction, luminance
percentiles -- is ACHROMATIC. A frame can score 88% non-black, 3,676 colours and
"correct" luminance percentiles while containing none of the reference's colour at
all. That happened repeatedly on Sonic Frontiers: an atmosphere shader plus laser
effects scored well on every one of those while the level geometry was absent, and
the numbers rose as the picture got no closer.

What this reports instead:

  intersection   normalised 3D RGB histogram intersection, 0..1. The standard
                 bounded likeness measure: sum over bins of min(p, q). 1.0 means
                 identical colour distributions, 0.0 means disjoint palettes.
  hue bands      per-band coverage in both images, so the answer names WHAT is
                 missing ("oracle 38.1% green, ours 0.3% green") rather than only
                 saying the images differ.
  missing mass   how much of the reference's colour distribution has no counterpart
                 in the candidate. This is the number that goes to zero as a render
                 becomes correct, and it is not fooled by fog or bloom.

Deliberately NOT a pass/fail gate: it is a search and triage instrument. Use it to
rank candidate surfaces against a reference, and to say in one line what a frame is
missing. A human still judges whether a scene looks right.

Usage:
  image_likeness.py REFERENCE CANDIDATE [CANDIDATE ...]
  image_likeness.py --rank REFERENCE DIR      # rank every image in DIR by likeness
  image_likeness.py --grid REFERENCE CANDIDATE [--cells 16x9] [--threshold 32]
  image_likeness.py --regions REFERENCE CANDIDATE   # which region does it match?
  image_likeness.py --locate REFERENCE DIR          # rank surfaces by best region
"""
import sys, os, colorsys
import numpy as np
from PIL import Image

BINS = 8  # per channel -> 512 bins; coarse on purpose, robust to tonemap drift.

HUES = [("red", 345, 15), ("orange", 15, 45), ("yellow", 45, 70),
        ("green", 70, 165), ("cyan", 165, 195), ("blue", 195, 255),
        ("purple", 255, 285), ("magenta", 285, 345)]


def load(path, side=480):
    im = Image.open(path).convert("RGB")
    im.thumbnail((side, side))
    return np.asarray(im, dtype=np.uint8).reshape(-1, 3)


def hist(px):
    q = (px.astype(np.uint16) * BINS // 256).clip(0, BINS - 1)
    idx = q[:, 0] * BINS * BINS + q[:, 1] * BINS + q[:, 2]
    h = np.bincount(idx, minlength=BINS ** 3).astype(np.float64)
    return h / max(h.sum(), 1.0)


def hue_profile(px):
    """Fraction of reasonably saturated, reasonably lit pixels in each hue band.

    Unsaturated pixels (fog, grey, white) are excluded deliberately -- they are
    exactly what inflated the achromatic metrics, and they carry no hue evidence.
    """
    f = px.astype(np.float64) / 255.0
    mx, mn = f.max(1), f.min(1)
    v, c = mx, mx - mn
    s = np.divide(c, np.maximum(mx, 1e-9))
    keep = (s > 0.20) & (v > 0.10)
    out = {name: 0.0 for name, _, _ in HUES}
    out["_chromatic"] = float(keep.mean())
    if not keep.any():
        return out
    r, g, b = f[keep, 0], f[keep, 1], f[keep, 2]
    mxk, mnk = np.max(f[keep], 1), np.min(f[keep], 1)
    ck = np.maximum(mxk - mnk, 1e-9)
    h = np.zeros_like(ck)
    m = mxk == r; h[m] = ((g[m] - b[m]) / ck[m]) % 6
    m = mxk == g; h[m] = ((b[m] - r[m]) / ck[m]) + 2
    m = mxk == b; h[m] = ((r[m] - g[m]) / ck[m]) + 4
    deg = (h * 60.0) % 360.0
    total = len(deg)
    for name, lo, hi in HUES:
        sel = (deg >= lo) & (deg < hi) if lo < hi else ((deg >= lo) | (deg < hi))
        out[name] = float(sel.sum()) / total * out["_chromatic"]
    return out


def compare(ref_px, cand_px):
    hr, hc = hist(ref_px), hist(cand_px)
    inter = float(np.minimum(hr, hc).sum())
    missing = float(np.maximum(hr - hc, 0).sum())
    return inter, missing, hue_profile(ref_px), hue_profile(cand_px)


def report(ref_path, cand_path, ref_px=None):
    ref_px = load(ref_path) if ref_px is None else ref_px
    cand_px = load(cand_path)
    inter, missing, hr, hc = compare(ref_px, cand_px)
    print(f"\n{os.path.basename(cand_path)}")
    print(f"  intersection with reference : {inter:6.3f}   (1.0 = same palette)")
    print(f"  reference colour unmatched  : {missing:6.3f}   (0.0 = nothing missing)")
    print(f"  chromatic pixels  ref={hr['_chromatic']*100:5.1f}%  cand={hc['_chromatic']*100:5.1f}%")
    rows = [(n, hr[n] * 100, hc[n] * 100) for n, _, _ in HUES]
    rows = [r for r in rows if r[1] > 0.5 or r[2] > 0.5]
    if rows:
        print(f"  {'hue':<9} {'reference':>10} {'candidate':>10}   delta")
        for n, a, b in sorted(rows, key=lambda t: -t[1]):
            flag = "  <-- MISSING" if a > 3.0 and b < a * 0.25 else ""
            print(f"  {n:<9} {a:9.1f}% {b:9.1f}%   {b-a:+6.1f}{flag}")
    return inter


def grid_compare(ref_path, cand_path, cols=16, rows=9, threshold=32):
    """Spatial per-channel comparison on a resolution-independent CELLS x CELLS grid.

    The histogram measure above is GLOBAL: it scores identically whether the green is
    on the ground or in the sky. This adds the spatial half. Both images are box-
    filtered down to one cell per block, which makes the comparison independent of
    render resolution, dynamic-resolution scaling and letterboxing differences, then
    each cell's mean R, G and B are compared directly.

    The grid defaults to 16x9 so the cells are SQUARE on a 16:9 frame; a square grid
    on a widescreen image stretches every cell and smears vertical detail (the horizon
    line, in this title) across neighbours.

    `threshold` is the per-channel 0..255 tolerance for calling a cell matched. It is
    deliberately an argument rather than a constant: a tonemap or exposure difference
    shifts every cell a little, and what matters for "is the world there" is whether
    the SPATIAL colour layout agrees, not whether it agrees exactly.
    """
    a = np.asarray(Image.open(ref_path).convert("RGB").resize((cols, rows), Image.BOX),
                   dtype=np.int16)
    b = np.asarray(Image.open(cand_path).convert("RGB").resize((cols, rows), Image.BOX),
                   dtype=np.int16)
    d = np.abs(a - b)
    per_cell_max = d.max(axis=2)
    matched = per_cell_max <= threshold
    print(f"\n{os.path.basename(cand_path)}  [{cols}x{rows} grid, tol={threshold}]")
    print(f"  cells matched      : {matched.sum():3d}/{cols*rows}  "
          f"({100.0*matched.mean():5.1f}%)")
    for i, ch in enumerate("RGB"):
        print(f"  mean |delta| {ch}     : {d[:,:,i].mean():6.1f}   max {d[:,:,i].max():3d}")
    # Spatial map: which channel is most deficient in each cell, so the layout of the
    # failure is visible. '.' matched, r/g/b = candidate SHORT of reference on that
    # channel, R/G/B = candidate OVER.
    print("  map (lowercase = missing, uppercase = excess, '.' = within tolerance):")
    for y in range(rows):
        row = []
        for x in range(cols):
            if matched[y, x]:
                row.append('.')
            else:
                k = int(np.argmax(d[y, x]))
                ch = "RGB"[k]
                row.append(ch.lower() if b[y, x, k] < a[y, x, k] else ch)
        print("    " + "".join(row))
    return float(matched.mean())


REGIONS = [
    ("full",          0.00, 0.00, 1.00, 1.00),
    ("top half",      0.00, 0.00, 1.00, 0.50),
    ("bottom half",   0.00, 0.50, 1.00, 1.00),
    ("left half",     0.00, 0.00, 0.50, 1.00),
    ("right half",    0.50, 0.00, 1.00, 1.00),
    ("top third",     0.00, 0.00, 1.00, 0.33),
    ("middle third",  0.00, 0.33, 1.00, 0.67),
    ("bottom third",  0.00, 0.67, 1.00, 1.00),
    ("centre",        0.25, 0.25, 0.75, 0.75),
    ("top-left q",    0.00, 0.00, 0.50, 0.50),
    ("top-right q",   0.50, 0.00, 1.00, 0.50),
    ("bottom-left q", 0.00, 0.50, 0.50, 1.00),
    ("bottom-right q",0.50, 0.50, 1.00, 1.00),
]


def region_match(ref_path, cand_path, cols=64, rows=36, threshold=32, quiet=False):
    """Which REGION of the reference does this surface correspond to?

    An intermediate surface rarely holds the whole frame -- a shadow atlas, a
    half-resolution bloom plane or a G-buffer covering only part of the scene will
    score badly overall while matching one region well. Scoring regions separately
    turns the likeness measure into a locator: "this output corresponds to the
    oracle's lower half" is a far more useful sentence than "this output scores
    0.11". A finer grid than the 16x9 display map is used here because these scores
    are read numerically rather than as a picture.
    """
    a = np.asarray(Image.open(ref_path).convert("RGB").resize((cols, rows), Image.BOX),
                   dtype=np.int16)
    b = np.asarray(Image.open(cand_path).convert("RGB").resize((cols, rows), Image.BOX),
                   dtype=np.int16)
    d = np.abs(a - b).max(axis=2)
    out = []
    for name, x0, y0, x1, y1 in REGIONS:
        cx0, cx1 = int(x0 * cols), max(int(x1 * cols), int(x0 * cols) + 1)
        cy0, cy1 = int(y0 * rows), max(int(y1 * rows), int(y0 * rows) + 1)
        sub = d[cy0:cy1, cx0:cx1]
        out.append((float((sub <= threshold).mean()), name, float(sub.mean())))
    out.sort(reverse=True)
    if not quiet:
        print(f"\n{os.path.basename(cand_path)}  [region match, tol={threshold}]")
        for frac, name, mad in out[:6]:
            bar = "#" * int(frac * 40)
            print(f"  {name:<15} {frac*100:5.1f}% cells   mean|d|={mad:6.1f}  {bar}")
    return out[0]


def _take_opts(rest):
    """Pull shared options out of `rest`, in one place.

    Every mode parses options through this. An earlier version parsed --threshold in
    --grid only, so `--locate ... --threshold 80` was silently ignored and three runs
    at three tolerances returned byte-identical numbers -- which reads exactly like
    "the result is robust to tolerance". Unparsed flags must not be silent.
    """
    cols, rows, thr = 16, 9, 32
    known = {"--cells", "--threshold"}
    i = 0
    while i < len(rest):
        if rest[i] == "--cells":
            spec = rest[i + 1]; del rest[i:i + 2]
            if "x" in spec: cols, rows = (int(v) for v in spec.split("x"))
            else: cols = rows = int(spec)
        elif rest[i] == "--threshold":
            thr = int(rest[i + 1]); del rest[i:i + 2]
        elif rest[i].startswith("--") and rest[i] not in known:
            raise SystemExit(f"unknown option {rest[i]!r} -- refusing to run rather than "
                             f"silently ignore it")
        else:
            i += 1
    return cols, rows, thr


def main(argv):
    if len(argv) >= 3 and argv[0] == "--regions":
        rest = list(argv[1:]); _c, _r, thr = _take_opts(rest)
        ref = rest[0]
        for c in rest[1:]:
            region_match(ref, c, threshold=thr)
        return 0
    if len(argv) >= 3 and argv[0] == "--locate":
        rest = list(argv[1:]); _c, _r, thr = _take_opts(rest)
        ref, d = rest[0], rest[1]
        found = []
        for f in sorted(os.listdir(d)):
            fp = os.path.join(d, f)
            if not os.path.isfile(fp):
                continue
            try:
                frac, name, mad = region_match(ref, fp, threshold=thr, quiet=True)
                found.append((frac, name, mad, f))
            except Exception:
                pass
        print(f"=== best-matching region of {os.path.basename(ref)} per surface "
              f"(tol={thr}) ===")
        for frac, name, mad, f in sorted(found, reverse=True)[:20]:
            print(f"  {frac*100:5.1f}%  {name:<15} mean|d|={mad:6.1f}  {f}")
        return 0
    if len(argv) >= 3 and argv[0] == "--grid":
        rest = list(argv[1:]); cols, rows, thr = _take_opts(rest)
        ref = rest[0]
        for c in rest[1:]:
            grid_compare(ref, c, cols, rows, thr)
        return 0
    if len(argv) >= 3 and argv[0] == "--rank":
        ref, d = argv[1], argv[2]
        ref_px = load(ref)
        scored = []
        for f in sorted(os.listdir(d)):
            p = os.path.join(d, f)
            if not os.path.isfile(p):
                continue
            try:
                scored.append((report(ref, p, ref_px), f))
            except Exception:
                pass
        print("\n=== ranked by palette intersection ===")
        for s, f in sorted(scored, reverse=True):
            print(f"  {s:6.3f}  {f}")
        return 0
    if len(argv) < 2:
        print(__doc__); return 2
    ref = argv[0]
    ref_px = load(ref)
    for c in argv[1:]:
        report(ref, c, ref_px)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
