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

  palette        normalised 3D RGB histogram intersection, 0..1. The standard
                 bounded likeness measure: sum over bins of min(p, q). 1.0 means
                 identical colour distributions, 0.0 means disjoint palettes.
  hue recall     of the reference's CHROMATIC mass, how much the candidate
                 reproduces in the same hue band. This is the half that a grey
                 or monochrome candidate cannot fake.
  likeness       sqrt(palette * hue recall) -- the ranking score. A geometric
                 mean, so a zero on either factor collapses it.
  hue bands      per-band coverage in both images, so the answer names WHAT is
                 missing ("oracle 38.1% green, ours 0.3% green") rather than only
                 saying the images differ.

Why the score is a composite, and not the histogram intersection alone: an
independent review of the first version measured a plain grey GRADIENT scoring
0.502 against a real dark reference frame and ranking THIRD of eleven, above a
real menu capture at 0.397 -- and all-black beating a real frame. That is the
project's own recorded trap (a diagnostic gradient outscoring real content on a
colour metric) reproduced on the tool written to prevent it. A 512-bin RGB
histogram is dominated by whatever bins carry the most mass, which for most
frames is the dark and grey bins, so intersection alone is very nearly the
achromatic measure this file exists to replace. The hue factor is what makes the
score answer the question in the first paragraph. When the REFERENCE is itself
essentially achromatic the hue factor carries no information; the score then
falls back to the palette term and says so on the line, rather than silently
reporting a different quantity under the same name.

Deliberately NOT a pass/fail gate: it is a search and triage instrument. Use it to
rank candidate surfaces against a reference, and to say in one line what a frame is
missing. A human still judges whether a scene looks right.

Usage:
  image_likeness.py REFERENCE CANDIDATE [CANDIDATE ...]
  image_likeness.py --rank REFERENCE DIR      # rank every image in DIR by likeness
  image_likeness.py --grid REFERENCE CANDIDATE [--cells 16x9] [--threshold 32]
  image_likeness.py --regions REFERENCE CANDIDATE   # which region does it match?
  image_likeness.py --locate REFERENCE DIR          # rank surfaces by best region
  image_likeness.py --selftest                      # verify the measure's own claims
"""
import sys, os
import numpy as np
from PIL import Image

BINS = 8  # per channel -> 512 bins; coarse on purpose, robust to tonemap drift.

# Below this fraction of chromatic pixels the reference carries no usable hue
# evidence, and `hue recall` would be a ratio of two near-zero numbers.
MIN_REF_CHROMATIC = 0.02

HUES = [("red", 345, 15), ("orange", 15, 45), ("yellow", 45, 70),
        ("green", 70, 165), ("cyan", 165, 195), ("blue", 195, 255),
        ("purple", 255, 285), ("magenta", 285, 345)]


class ImageLikenessError(Exception):
    """Anything that makes a reported number meaningless. Never swallowed."""


def load(path, side=480):
    try:
        im = Image.open(path).convert("RGB")
        im.thumbnail((side, side))
        return np.asarray(im, dtype=np.uint8).reshape(-1, 3)
    except Exception as e:
        raise ImageLikenessError(f"cannot read {path}: {e}") from e


def load_full(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        raise ImageLikenessError(f"cannot read {path}: {e}") from e


def hist(px):
    q = (px.astype(np.uint16) * BINS // 256).clip(0, BINS - 1)
    idx = q[:, 0] * BINS * BINS + q[:, 1] * BINS + q[:, 2]
    h = np.bincount(idx, minlength=BINS ** 3).astype(np.float64)
    return h / max(h.sum(), 1.0)


def hue_profile(px):
    """Fraction of reasonably saturated, reasonably lit pixels in each hue band.

    Unsaturated pixels (fog, grey, white) are excluded deliberately -- they are
    exactly what inflated the achromatic metrics, and they carry no hue evidence.

    Each band's value is a fraction of ALL pixels, so the bands sum to
    `_chromatic`. That makes band values from two images directly comparable.
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
    """Return (palette, hue_recall_or_None, likeness, ref_hues, cand_hues).

    `hue_recall` is None exactly when the reference has too little chromatic mass
    for the ratio to mean anything; `likeness` then equals `palette`, and every
    caller is expected to say which of the two it printed.
    """
    hr, hc = hist(ref_px), hist(cand_px)
    palette = float(np.minimum(hr, hc).sum())
    # NOTE: sum(max(hr-hc,0)) is identically 1 - palette for two distributions that
    # each sum to 1. The first version printed it as "missing mass" beside the
    # intersection as though it were independent corroboration; it is the same
    # number written backwards, and a review caught it summing to 1.000000 on every
    # candidate. Do not reintroduce it as a second measure.
    ph, qh = hue_profile(ref_px), hue_profile(cand_px)
    ref_chromatic = ph["_chromatic"]
    if ref_chromatic < MIN_REF_CHROMATIC:
        return palette, None, palette, ph, qh
    hue_inter = sum(min(ph[n], qh[n]) for n, _, _ in HUES)
    hue_recall = hue_inter / ref_chromatic
    return palette, hue_recall, float(np.sqrt(palette * hue_recall)), ph, qh


def report(ref_path, cand_path, ref_px=None):
    ref_px = load(ref_path) if ref_px is None else ref_px
    cand_px = load(cand_path)
    palette, hue_recall, likeness, hr, hc = compare(ref_px, cand_px)
    print(f"\n{os.path.basename(cand_path)}")
    print(f"  palette intersection : {palette:6.3f}   (1.0 = same colour distribution)")
    if hue_recall is None:
        print(f"  hue recall           :    n/a   (reference is achromatic: "
              f"{hr['_chromatic']*100:.1f}% chromatic pixels)")
        print(f"  likeness             : {likeness:6.3f}   (PALETTE ONLY -- no hue evidence)")
    else:
        print(f"  hue recall           : {hue_recall:6.3f}   "
              f"(of the reference's chromatic mass)")
        print(f"  likeness             : {likeness:6.3f}   (sqrt of the two above)")
    print(f"  chromatic pixels  ref={hr['_chromatic']*100:5.1f}%  cand={hc['_chromatic']*100:5.1f}%")
    rows = [(n, hr[n] * 100, hc[n] * 100) for n, _, _ in HUES]
    rows = [r for r in rows if r[1] > 0.5 or r[2] > 0.5]
    if rows:
        print(f"  {'hue':<9} {'reference':>10} {'candidate':>10}   delta")
        for n, a, b in sorted(rows, key=lambda t: -t[1]):
            flag = "  <-- MISSING" if a > 3.0 and b < a * 0.25 else ""
            print(f"  {n:<9} {a:9.1f}% {b:9.1f}%   {b-a:+6.1f}{flag}")
    return likeness


def grid_compare(ref_path, cand_path, cols=16, rows=9, threshold=32):
    """Spatial per-channel comparison on a resolution-independent CELLS x CELLS grid.

    The histogram measure above is GLOBAL: it scores identically whether the green is
    on the ground or in the sky. This adds the spatial half. Both images are box-
    filtered down to one cell per block, which makes the comparison independent of
    render RESOLUTION and of dynamic-resolution scaling, then each cell's mean R, G
    and B are compared directly.

    It is NOT independent of letterboxing or pillarboxing, and an earlier version of
    this docstring claimed it was. Bars move content to different cells, so the cells
    stop corresponding: a review measured the same frame scoring 79.2% letterboxed
    and 68.1% pillarboxed against its unbarred self. Crop bars off before comparing.

    The grid defaults to 16x9 so the cells are SQUARE on a 16:9 frame; a square grid
    on a widescreen image stretches every cell and smears vertical detail (the horizon
    line, in this title) across neighbours.

    `threshold` is the per-channel 0..255 tolerance for calling a cell matched. It is
    deliberately an argument rather than a constant: a tonemap or exposure difference
    shifts every cell a little, and what matters for "is the world there" is whether
    the SPATIAL colour layout agrees, not whether it agrees exactly.
    """
    a = np.asarray(load_full(ref_path).resize((cols, rows), Image.BOX), dtype=np.int16)
    b = np.asarray(load_full(cand_path).resize((cols, rows), Image.BOX), dtype=np.int16)
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


# name, x0, y0, x1, y1 in fractions of the reference. `area` is derived, and is the
# tie-break: see the sort in region_match.
REGIONS = [
    ("full",          0.00, 0.00, 1.00, 1.00),
    ("top half",      0.00, 0.00, 1.00, 0.50),
    ("bottom half",   0.00, 0.50, 1.00, 1.00),
    ("left half",     0.00, 0.00, 0.50, 1.00),
    ("right half",    0.50, 0.00, 1.00, 1.00),
    ("top third",     0.00, 0.00, 1.00, 1.0 / 3.0),
    ("middle third",  0.00, 1.0 / 3.0, 1.00, 2.0 / 3.0),
    ("bottom third",  0.00, 2.0 / 3.0, 1.00, 1.00),
    ("centre",        0.25, 0.25, 0.75, 0.75),
    ("top-left q",    0.00, 0.00, 0.50, 0.50),
    ("top-right q",   0.50, 0.00, 1.00, 0.50),
    ("bottom-left q", 0.00, 0.50, 0.50, 1.00),
    ("bottom-right q",0.50, 0.50, 1.00, 1.00),
]


def region_match(ref_path, cand_path, cols=64, rows=36, threshold=32, quiet=False):
    """Which REGION of the reference does this whole surface correspond to?

    An intermediate surface rarely holds the whole frame -- a shadow atlas, a
    half-resolution bloom plane or a G-buffer covering only part of the scene will
    score badly overall while matching one region well. Scoring regions separately
    turns the likeness measure into a locator: "this output corresponds to the
    oracle's lower half" is a far more useful sentence than "this output scores
    0.11".

    The comparison CROPS THE REFERENCE and keeps the candidate whole. The first
    version resized both full images to one grid and then sliced the difference,
    which answers a different question -- "in which part of the frame do these two
    full images agree?" -- and gets the documented one wrong: a review fed it a PNG
    that was literally the oracle's bottom half and it reported "bottom third,
    75.0%", with "bottom half" fourth at 50%. Cropping the reference returns 100.0%
    on the correct region.

    A finer grid than the 16x9 display map is used here because these scores are
    read numerically rather than as a picture.
    """
    ref_im = load_full(ref_path)
    cand = np.asarray(load_full(cand_path).resize((cols, rows), Image.BOX), dtype=np.int16)
    W, H = ref_im.size
    out = []
    for name, x0, y0, x1, y1 in REGIONS:
        px0, py0 = int(round(x0 * W)), int(round(y0 * H))
        px1, py1 = max(int(round(x1 * W)), px0 + 1), max(int(round(y1 * H)), py0 + 1)
        crop = ref_im.crop((px0, py0, min(px1, W), min(py1, H)))
        a = np.asarray(crop.resize((cols, rows), Image.BOX), dtype=np.int16)
        d = np.abs(a - cand).max(axis=2)
        area = (x1 - x0) * (y1 - y0)
        out.append((float((d <= threshold).mean()), area, -float(d.mean()), name))
    # Tie-break on AREA, largest first. Sorting the old (frac, name, mad) tuple
    # broke frac ties alphabetically-descending, so comparing a frame against
    # ITSELF -- where every region scores 1.000 -- reported "top-right q" and did
    # not show "full" at all in the printed rows.
    out.sort(reverse=True)
    res = [(frac, name, -negmad) for frac, area, negmad, name in out]
    if not quiet:
        print(f"\n{os.path.basename(cand_path)}  [region match, tol={threshold}]")
        for frac, name, mad in res[:6]:
            bar = "#" * int(frac * 40)
            print(f"  {name:<15} {frac*100:5.1f}% cells   mean|d|={mad:6.1f}  {bar}")
    return res[0]


def _take_opts(rest, known):
    """Pull shared options out of `rest`, in one place.

    EVERY mode calls this, including --rank and the default mode. An earlier version
    parsed --threshold in --grid only, so `--locate ... --threshold 80` was silently
    ignored and three runs at three tolerances returned byte-identical numbers --
    which reads exactly like "the result is robust to tolerance". The repair was
    incomplete and a review caught the rest: --rank still read argv[1]/argv[2]
    directly and took no options at all (four --threshold values, one md5), while
    --locate and --regions accepted --cells and dropped it on the floor. "Accepted
    but unused" produces the identical symptom to "unknown and ignored", so a mode
    must consume its options through here AND pass them on.
    """
    cols, rows, thr = 16, 9, 32
    i = 0
    while i < len(rest):
        if rest[i].startswith("--") and rest[i] not in known:
            raise SystemExit(
                f"{rest[i]!r} is not accepted here -- refusing rather than ignoring it. "
                f"This mode takes: {' '.join(known) if known else '(no options)'}")
        if rest[i] == "--cells" and i + 1 < len(rest):
            spec = rest[i + 1]
            try:
                c, r = spec.lower().split("x")
                cols, rows = int(c), int(r)
            except Exception:
                raise SystemExit(f"--cells wants COLSxROWS, got {spec!r}")
            if cols < 1 or rows < 1:
                raise SystemExit(f"--cells must be positive, got {spec!r}")
            del rest[i:i + 2]
        elif rest[i] == "--threshold" and i + 1 < len(rest):
            try:
                thr = int(rest[i + 1])
            except Exception:
                raise SystemExit(f"--threshold wants an integer, got {rest[i + 1]!r}")
            if not 0 <= thr <= 255:
                raise SystemExit(f"--threshold must be 0..255, got {thr}")
            del rest[i:i + 2]
        elif rest[i].startswith("--"):
            raise SystemExit(f"option {rest[i]!r} is missing its value")
        else:
            i += 1
    return cols, rows, thr


def _need(rest, n, usage):
    if len(rest) < n:
        raise SystemExit(f"{usage}: need {n} path(s), got {len(rest)} after options")


def _scan_dir(d, fn):
    """Apply `fn` to every file in `d`, returning (results, failures).

    Failures are COUNTED and reported. The first version wrapped the whole per-file
    body including the reference load in `except Exception: pass`, so an unreadable
    reference, or a directory of corrupt files, printed a completed-looking header
    with zero rows and exited 0.
    """
    if not os.path.isdir(d):
        raise SystemExit(f"not a directory: {d}")
    results, failures = [], []
    for f in sorted(os.listdir(d)):
        p = os.path.join(d, f)
        if not os.path.isfile(p):
            continue
        try:
            results.append((fn(p), f))
        except ImageLikenessError as e:
            failures.append((f, str(e)))
    return results, failures


def _report_failures(failures, total_wanted):
    for f, msg in failures:
        print(f"  (skipped {f}: {msg})", file=sys.stderr)
    if not total_wanted:
        print("no readable images found -- nothing was compared", file=sys.stderr)
        return 2
    return 0


def selftest():
    """Assert the properties this file claims, on synthetic images.

    Every case here is one a review actually broke the first version with. They are
    cheap, they need no fixtures, and they are the difference between "the tool
    reports a number" and "the number means what the docstring says".
    """
    import tempfile
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}  {name}")
        ok = ok and bool(cond)

    with tempfile.TemporaryDirectory() as td:
        def save(name, im):
            p = os.path.join(td, name); im.save(p); return p

        # A colourful reference: green top half, red bottom half.
        W, H = 320, 180
        ref = Image.new("RGB", (W, H), (20, 200, 60))
        for y in range(H // 2, H):
            for x in range(W):
                ref.putpixel((x, y), (200, 40, 30))
        ref_p = save("ref.png", ref)

        grey = save("grey.png", Image.linear_gradient("L").convert("RGB").resize((W, H)))
        black = save("black.png", Image.new("RGB", (W, H), (0, 0, 0)))
        bottom = save("bottom.png", ref.crop((0, H // 2, W, H)))

        ref_px = load(ref_p)
        s_self = report_score(ref_p, ref_p, ref_px)
        s_grey = report_score(ref_p, grey, ref_px)
        s_black = report_score(ref_p, black, ref_px)
        check("identical scores 1.000", abs(s_self - 1.0) < 1e-6)
        check(f"grey gradient ({s_grey:.3f}) scores below 0.10", s_grey < 0.10)
        check(f"all-black ({s_black:.3f}) scores below 0.10", s_black < 0.10)
        check("gradient does not beat identity", s_grey < s_self)

        # The arm that actually discriminates the composite score from the plain
        # palette intersection. The saturated two-colour reference above CANNOT:
        # a grey gradient has no mass in its two bins, so palette alone already
        # scores it 0 and the check passes under either measure -- a control drawn
        # from the same source as the null. A REALISTIC dark frame is needed, and
        # with one the old measure is badly fooled: 0.740.
        yy = np.linspace(0, 1, H)[:, None]
        basev = (8 + 70 * yy) * np.ones((1, W))
        dark = np.stack([basev * 1.00, basev * 1.04, basev * 1.12], axis=2)
        dark[int(H * 0.55):, int(W * 0.15):int(W * 0.60)] = [30, 165, 55]
        dark[int(H * 0.20):int(H * 0.35), int(W * 0.60):] = [180, 60, 40]
        dark_p = save("dark.png", Image.fromarray(dark.clip(0, 255).astype(np.uint8)))
        gv = (np.linspace(6, 95, H)[:, None] * np.ones((1, W)))
        dgrad_p = save("dgrad.png", Image.fromarray(
            np.stack([gv, gv * 1.03, gv * 1.10], 2).clip(0, 255).astype(np.uint8)))
        dark_px = load(dark_p)
        pal, hue, like, _, _ = compare(dark_px, load(dgrad_p))
        check(f"palette ALONE is fooled by a matched grey gradient ({pal:.3f}) -- "
              f"this is why the score is a composite", pal > 0.50)
        check(f"likeness is not ({like:.3f}, hue recall {hue:.3f})", like < 0.10)

        frac, name, _ = region_match(ref_p, bottom, quiet=True)
        check(f"oracle's bottom half locates as 'bottom half' (got {name!r} "
              f"{frac*100:.1f}%)", name == "bottom half" and frac > 0.99)

        frac, name, _ = region_match(ref_p, ref_p, quiet=True)
        check(f"a frame against itself locates as 'full' (got {name!r})",
              name == "full" and frac > 0.99)

        # An option that is accepted must change something.
        a = grid_compare_score(ref_p, grey, 16, 9, 0)
        b = grid_compare_score(ref_p, grey, 16, 9, 255)
        check(f"--threshold changes --grid ({a:.3f} vs {b:.3f})", a != b)
        # --cells must CHANGE the answer, not merely be accepted. A control that
        # cannot fail is void (this one asserted True until it was rewritten).
        # Fine noise averages away on a coarse grid and survives on a fine one.
        rng = np.random.default_rng(7)
        noise = Image.fromarray(
            (np.asarray(ref, dtype=np.int16) +
             rng.integers(-90, 91, (H, W, 3))).clip(0, 255).astype(np.uint8))
        noise_p = save("noise.png", noise)
        coarse = grid_compare_score(ref_p, noise_p, 4, 3, 24)
        fine = grid_compare_score(ref_p, noise_p, 64, 36, 24)
        check(f"--cells changes the answer (4x3 {coarse:.3f} vs 64x36 {fine:.3f})",
              coarse != fine)
        r_coarse = region_match(ref_p, noise_p, cols=4, rows=3, threshold=24,
                                quiet=True)[0]
        r_fine = region_match(ref_p, noise_p, cols=64, rows=36, threshold=24,
                              quiet=True)[0]
        check(f"--cells reaches region_match ({r_coarse:.3f} vs {r_fine:.3f})",
              r_coarse != r_fine)

        # Options a mode does not USE must be refused, not accepted and dropped.
        for mode_args in (["--rank", ref_p, td, "--threshold", "40"],
                          ["--rank", ref_p, td, "--cells", "4x3"],
                          [ref_p, grey, "--threshold", "40"]):
            try:
                main(mode_args); refused = False
            except SystemExit:
                refused = True
            check(f"{' '.join(mode_args[:1] or ['default'])} refuses "
                  f"{mode_args[-2]}", refused)

    print("\nselftest: " + ("all passed" if ok else "FAILURES PRESENT"))
    return 0 if ok else 1


def report_score(ref_path, cand_path, ref_px=None):
    """The likeness score with no printing -- for selftest and ranking."""
    ref_px = load(ref_path) if ref_px is None else ref_px
    return compare(ref_px, load(cand_path))[2]


def grid_compare_score(ref_path, cand_path, cols, rows, threshold):
    a = np.asarray(load_full(ref_path).resize((cols, rows), Image.BOX), dtype=np.int16)
    b = np.asarray(load_full(cand_path).resize((cols, rows), Image.BOX), dtype=np.int16)
    return float((np.abs(a - b).max(axis=2) <= threshold).mean())


def main(argv):
    try:
        return _main(argv)
    except ImageLikenessError as e:
        print(f"image_likeness: {e}", file=sys.stderr)
        return 2


def _main(argv):
    if argv and argv[0] == "--selftest":
        return selftest()

    if argv and argv[0] in ("--regions", "--locate", "--grid", "--rank"):
        mode = argv[0]
        rest = list(argv[1:])
        # --rank scores by palette+hue, which has no grid and no tolerance, so it
        # accepts neither. Declaring the set per mode is what stops the
        # "accepted but unused" form of the original defect coming back.
        spatial = ("--cells", "--threshold")
        cols, rows, thr = _take_opts(rest, () if mode == "--rank" else spatial)
        if mode == "--rank":
            _need(rest, 2, "--rank REFERENCE DIR")
            ref, d = rest[0], rest[1]
            ref_px = load(ref)                     # loud if unreadable
            scored, failures = _scan_dir(d, lambda p: report(ref, p, ref_px))
            print("\n=== ranked by likeness (sqrt(palette * hue recall)) ===")
            for s, f in sorted(scored, key=lambda t: (-t[0], t[1])):
                print(f"  {s:6.3f}  {f}")
            return _report_failures(failures, len(scored))
        if mode == "--locate":
            _need(rest, 2, "--locate REFERENCE DIR")
            ref, d = rest[0], rest[1]
            load_full(ref)                         # loud if unreadable
            found, failures = _scan_dir(
                d, lambda p: region_match(ref, p, cols=max(cols, 8), rows=max(rows, 8),
                                          threshold=thr, quiet=True))
            print(f"=== best-matching region of {os.path.basename(ref)} per surface "
                  f"(tol={thr}, grid={max(cols,8)}x{max(rows,8)}) ===")
            for (frac, name, mad), f in sorted(found, key=lambda t: -t[0][0])[:20]:
                print(f"  {frac*100:5.1f}%  {name:<15} mean|d|={mad:6.1f}  {f}")
            return _report_failures(failures, len(found))
        if mode == "--grid":
            _need(rest, 2, "--grid REFERENCE CANDIDATE")
            for c in rest[1:]:
                grid_compare(rest[0], c, cols, rows, thr)
            return 0
        _need(rest, 2, "--regions REFERENCE CANDIDATE")
        for c in rest[1:]:
            region_match(rest[0], c, cols=max(cols, 8), rows=max(rows, 8), threshold=thr)
        return 0

    rest = list(argv)
    _take_opts(rest, ())  # default mode takes no options, but must still REFUSE them
    if len(rest) < 2:
        print(__doc__); return 2
    ref_px = load(rest[0])
    for c in rest[1:]:
        report(rest[0], c, ref_px)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
