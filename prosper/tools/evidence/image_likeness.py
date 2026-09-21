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


def structure(ref_path, cand_path, cols=16, rows=9):
    """Pearson correlation of the two images' downsampled luminance, clamped to 0..1.

    The second factor for references that carry no hue evidence. It is threshold
    free, and it is zero for exactly the images that fake a palette match: a flat
    fill and a near-flat frame have no luminance variance, so the correlation is
    undefined and is reported as 0 rather than as agreement.

    This exists because the first fix's `PALETTE ONLY` fallback reopened the very
    defect it was written for. Below the chromatic floor the score WAS the plain
    palette intersection, and on a real achromatic reference (a title screen) an
    all-black candidate scored 0.968 and ranked third of six, above a real capture
    at 0.942 -- the previous review's sentence, verbatim, one branch over. 32 of
    this repository's 190 committed screenshots sit under that floor, so it was
    not a corner.
    """
    a = np.asarray(load_full(ref_path).convert("L").resize((cols, rows), Image.BOX),
                   dtype=np.float64).ravel()
    b = np.asarray(load_full(cand_path).convert("L").resize((cols, rows), Image.BOX),
                   dtype=np.float64).ravel()
    if a.std() < 1e-6 or b.std() < 1e-6:
        return 0.0
    r = float(max(0.0, np.corrcoef(a, b)[0, 1]))
    # Correlation ALONE is affine invariant, and on a dark reference the palette
    # term is high precisely because both images are dark -- so nothing in the
    # product sees magnitude. Measured on the Little Nightmares III title screen:
    # the reference scaled by 0.01, a frame whose brightest pixel is 2/255 and
    # which is black to look at, scored r=0.781 and likeness 0.869 -- ABOVE a real
    # capture at 0.771. That is the gradient trap in its third form, so the
    # correlation is penalised by how far the best-fit gain departs from 1.
    slope = float(np.cov(a, b, bias=True)[0, 1] / max(a.var(), 1e-12))
    gain = 0.0 if slope <= 0.0 else min(slope, 1.0 / slope)
    return r * gain


def compare(ref_px, cand_px, struct=None):
    """Return (palette, second, likeness, ref_hues, cand_hues).

    `second` is the factor multiplied with the palette term. Which one it is
    depends on how much chromatic mass the REFERENCE carries, and the two are
    blended rather than switched so there is no cliff at the boundary: a review
    measured the switched version jumping from 0.978 at 1.9% chromatic to 0.000
    at 3.0%, which is a property of the instrument and not of the images.
    `struct` is the structure factor, which the caller supplies because it needs
    the file paths; when it is None only the hue term is used.
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
    hue_inter = sum(min(ph[n], qh[n]) for n, _, _ in HUES)
    hue_recall = hue_inter / ref_chromatic if ref_chromatic > 1e-9 else 0.0
    if struct is None:
        second = hue_recall
    else:
        # Weight hue by how much chromatic mass the reference actually has. At or
        # above the floor this is pure hue recall (the factor an independent review
        # measured as well founded, and which is left untouched); as the reference
        # goes achromatic the weight falls to zero and structure carries it.
        w = min(1.0, ref_chromatic / MIN_REF_CHROMATIC)
        second = w * hue_recall + (1.0 - w) * struct
    return palette, second, float(np.sqrt(palette * second)), ph, qh


def report(ref_path, cand_path, ref_px=None, quiet=False):
    """Score one candidate, printing the breakdown unless `quiet`.

    The selftest calls this with quiet=True rather than a private scoring helper.
    An earlier round had a separate `report_score()` that no mode ran, so its
    arms measured a quantity the tool never reported -- and it had already
    drifted (it omitted the structure factor).
    """
    ref_px = load(ref_path) if ref_px is None else ref_px
    cand_px = load(cand_path)
    chrom_probe = hue_profile(ref_px)["_chromatic"]
    # Only pay for the structure factor when the blend will actually use it.
    struct = (structure(ref_path, cand_path)
              if chrom_probe < MIN_REF_CHROMATIC else None)
    palette, second, likeness, hr, hc = compare(ref_px, cand_px, struct)
    chrom = hr["_chromatic"]
    w = min(1.0, chrom / MIN_REF_CHROMATIC)
    if quiet:
        return likeness
    print(f"\n{os.path.basename(cand_path)}")
    print(f"  palette intersection : {palette:6.3f}   (1.0 = same colour distribution)")
    hue_inter = sum(min(hr[n], hc[n]) for n, _, _ in HUES)
    hue_recall = hue_inter / chrom if chrom > 1e-9 else 0.0
    print(f"  hue recall           : {hue_recall:6.3f}   "
          f"(of the reference's chromatic mass)")
    if struct is not None:
        print(f"  structure            : {struct:6.3f}   (luminance correlation x gain)")
    if w >= 1.0:
        print(f"  likeness             : {likeness:6.3f}   = sqrt(palette x hue recall)")
    else:
        print(f"  likeness             : {likeness:6.3f}   = sqrt(palette x second), "
              f"second = {w:.2f}*hue + {1-w:.2f}*structure")
        print(f"       (reference is only {chrom*100:.1f}% chromatic, so hue evidence "
              f"is weighted down)")
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
        print(f"\n{os.path.basename(cand_path)}  [region match, tol={threshold}, "
              f"grid={cols}x{rows}]")
        for frac, name, mad in res[:6]:
            bar = "#" * int(frac * 40)
            print(f"  {name:<15} {frac*100:5.1f}% cells   mean|d|={mad:6.1f}  {bar}")
    return res[0]


def _take_opts(rest, known, cols_default=16, rows_default=9):
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
    cols, rows, thr = cols_default, rows_default, 32
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

    Every case here is one a review actually broke a previous version with. They
    are cheap, they need no fixtures, and they are the difference between "the
    tool reports a number" and "the number means what the docstring says".

    THE FIXTURE IS PART OF THE TEST. A second review found the region arm could
    not fail: the fixture was two flat colour bands, and the defect it was meant
    to catch -- comparing two full images stretched to one grid instead of
    cropping the reference -- is a no-op on flat bands, so reinstating it left the
    suite green while the ground truth went from 13 regions correct to 4. That is
    this project's same-source-control trap: the arm tested the discriminator, not
    the domain. The reference below is deliberately STRUCTURED (a vertical
    gradient plus four differently-coloured blocks at four distinct positions) so
    that the stretch and the crop cannot agree. Verified: with the legacy
    algorithm reinstated, bottom-half locates as "top-right q", left-half as
    "bottom-left q" and centre as "top-left q". Do not simplify this fixture.
    """
    import io, contextlib, tempfile
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}  {name}")
        ok = ok and bool(cond)

    with tempfile.TemporaryDirectory() as td:
        def save(name, im):
            p = os.path.join(td, name); im.save(p); return p

        W, H = 320, 180
        a = np.zeros((H, W, 3), np.float64)
        yy = np.linspace(0, 1, H)[:, None]
        a[:, :, 0] = 30 + 60 * yy; a[:, :, 1] = 40 + 40 * yy; a[:, :, 2] = 70 - 40 * yy
        a[int(H * .05):int(H * .22), int(W * .06):int(W * .30)] = [40, 190, 70]
        a[int(H * .40):int(H * .52), int(W * .55):int(W * .92)] = [210, 80, 40]
        a[int(H * .62):int(H * .80), int(W * .10):int(W * .40)] = [60, 90, 220]
        a[int(H * .85):, int(W * .70):] = [220, 210, 60]
        ref = Image.fromarray(a.clip(0, 255).astype(np.uint8))
        ref_p = save("ref.png", ref)

        grey = save("grey.png", Image.linear_gradient("L").convert("RGB").resize((W, H)))
        black = save("black.png", Image.new("RGB", (W, H), (0, 0, 0)))
        ref_px = load(ref_p)

        check("identical scores 1.000", abs(report(ref_p, ref_p, ref_px, quiet=True) - 1.0) < 1e-6)
        s_grey, s_black = report(ref_p, grey, ref_px, quiet=True), report(ref_p, black, ref_px, quiet=True)
        check(f"grey gradient ({s_grey:.3f}) scores below 0.10", s_grey < 0.10)
        check(f"all-black ({s_black:.3f}) scores below 0.10", s_black < 0.10)

        # Palette ALONE is fooled by a grey gradient matched to the reference's own
        # luminance range; the composite is not. This is the arm that separates the
        # two measures -- a saturated fixture cannot, because a gradient has no mass
        # in its bins and scores 0 under either.
        basev = (8 + 70 * yy) * np.ones((1, W))
        dark = np.stack([basev * 1.00, basev * 1.04, basev * 1.12], axis=2)
        dark[int(H * .55):, int(W * .15):int(W * .60)] = [30, 165, 55]
        dark[int(H * .20):int(H * .35), int(W * .60):] = [180, 60, 40]
        dark_p = save("dark.png", Image.fromarray(dark.clip(0, 255).astype(np.uint8)))
        gv = (np.linspace(6, 95, H)[:, None] * np.ones((1, W)))
        dgrad_p = save("dgrad.png", Image.fromarray(
            np.stack([gv, gv * 1.03, gv * 1.10], 2).clip(0, 255).astype(np.uint8)))
        pal, _, like, _, _ = compare(load(dark_p), load(dgrad_p), structure(dark_p, dgrad_p))
        check(f"palette ALONE is fooled by a matched grey gradient ({pal:.3f})", pal > 0.50)
        check(f"likeness is not ({like:.3f})", like < 0.10)

        # An ACHROMATIC reference: the first fix fell back to the bare palette term
        # here, which reopened the defect one branch over -- all-black outranked a
        # real capture on a real title screen. Structure carries the weight now.
        # THE FIXTURE MUST BE DARK. A third review found this arm could not fail
        # either: it used a BRIGHT ramp (20..210), against which all-black shares
        # only 6.1% of the palette mass, so reinstating the real defect -- below
        # the floor, likeness = the bare palette term -- still scored all-black
        # 0.061 and the arm passed. A real achromatic reference is a title screen:
        # mostly near-black, where all-black shares 90%+ of the palette and the
        # defect scores it 0.909. Measured on this fixture: HEAD gives all-black
        # 0.000 against a real capture at 0.937; the defect gives 0.909.
        gg = (3 + 9 * yy) * np.ones((1, W))
        mono = np.stack([gg, gg * 1.01, gg * 1.02], 2)
        mono[int(H * .34):int(H * .48), int(W * .22):int(W * .78)] = 120  # lit panel
        mono[int(H * .70):int(H * .74), int(W * .35):int(W * .65)] = 55   # dim prompt
        mono_p = save("mono.png", Image.fromarray(mono.clip(0, 255).astype(np.uint8)))
        mono_chrom = hue_profile(load(mono_p))["_chromatic"]
        near_p = save("near.png", Image.fromarray(
            (mono * 0.88 + 3).clip(0, 255).astype(np.uint8)))
        # T2: correlation alone is affine invariant, so a scaled-to-black copy
        # correlates perfectly while being black to look at. On the real LM3 title
        # screen it scored 0.869 -- above a real capture at 0.771.
        dim_p = save("dim.png", Image.fromarray(
            (mono * 0.01).clip(0, 255).astype(np.uint8)))
        s_real = report(mono_p, near_p, quiet=True)
        s_flat = report(mono_p, black, quiet=True)
        s_dim = report(mono_p, dim_p, quiet=True)
        dim_max = int(np.asarray(Image.open(dim_p)).max())
        check(f"achromatic reference is under the hue floor ({mono_chrom*100:.1f}% < "
              f"{MIN_REF_CHROMATIC*100:.0f}%)", mono_chrom < MIN_REF_CHROMATIC)
        check(f"...all-black ({s_flat:.3f}) loses to a real capture ({s_real:.3f})",
              s_flat < s_real and s_flat < 0.10)
        check(f"...and so does a scaled-to-black copy that CORRELATES perfectly "
              f"({s_dim:.3f}, brightest pixel {dim_max}/255)", s_dim < s_real * 0.5)

        # The blend across the chromatic floor must be CONTINUOUS. Nothing pinned
        # this, and a hard switch is the shape the round-2 defect had: two
        # references straddling the floor step 0.227 on the blend and 0.962 under
        # a switch. Build a pair either side of MIN_REF_CHROMATIC from the same
        # scene, so only the crossing differs.
        def tinted(frac):
            im = mono.copy()
            n = int(H * W * frac)
            flat = im.reshape(-1, 3)
            flat[:n] = [200, 40, 30]
            return Image.fromarray(im.clip(0, 255).astype(np.uint8))
        # Straddle the floor TIGHTLY. Two points far apart on the ramp measure the
        # ramp, not the boundary: at 0.5x and 1.5x the floor the blend itself steps
        # 0.661, which says nothing about continuity where the switch would be.
        lo_p = save("chrom_lo.png", tinted(MIN_REF_CHROMATIC * 0.95))
        hi_p = save("chrom_hi.png", tinted(MIN_REF_CHROMATIC * 1.05))
        lo_c = hue_profile(load(lo_p))["_chromatic"]
        hi_c = hue_profile(load(hi_p))["_chromatic"]
        step = abs(report(lo_p, near_p, quiet=True) - report(hi_p, near_p, quiet=True))
        # Honest claim: the blend REMOVES THE JUMP, it does not make the crossing
        # flat. second = (1-w)*struct, so likeness falls like sqrt(1-w) and the
        # approach steepens at the boundary by construction. Measured across the
        # floor: ~0.21 blended against ~0.94 under a hard switch.
        check(f"the hue/structure crossing is a ramp, not a jump "
              f"({lo_c*100:.2f}% -> {hi_c*100:.2f}% chromatic, step {step:.3f} "
              f"vs ~0.94 switched)",
              lo_c < MIN_REF_CHROMATIC <= hi_c and step < 0.40)

        # Region location, on the STRUCTURED fixture. Three shapes, not one.
        for rname, box in (("bottom half", (0, H // 2, W, H)),
                           ("left half", (0, 0, W // 2, H)),
                           ("centre", (int(W * .25), int(H * .25),
                                       int(W * .75), int(H * .75)))):
            cp = save(rname.replace(" ", "_") + ".png", ref.crop(box))
            frac, got, _ = region_match(ref_p, cp, quiet=True)
            check(f"the oracle's {rname} locates as {rname!r} (got {got!r} "
                  f"{frac*100:.1f}%)", got == rname and frac > 0.99)
        frac, got, _ = region_match(ref_p, ref_p, quiet=True)
        check(f"a frame against itself locates as 'full' (got {got!r})",
              got == "full" and frac > 0.99)
        # A genuine TIE: every region of a flat reference matches a flat candidate
        # at 100%, so only the tie-break decides. Cropping alone does not reach
        # this -- a self-comparison of a STRUCTURED frame has a unique winner, so
        # the arm above passes even with the tie-break reverted.
        flat_p = save("flat.png", Image.new("RGB", (W, H), (90, 92, 96)))
        flat2_p = save("flat2.png", Image.new("RGB", (W, H), (92, 94, 98)))
        tfrac, tgot, _ = region_match(flat_p, flat2_p, quiet=True)
        check(f"a tie across every region resolves to the largest, 'full' "
              f"(got {tgot!r} at {tfrac*100:.1f}%)", tgot == "full" and tfrac > 0.99)

        # Options must reach the code that uses them, THROUGH THE CLI. Calling the
        # internals directly would leave a broken call site green.
        def cli(args):
            buf, err = io.StringIO(), io.StringIO()
            try:
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
                    rc = main(args)
            except SystemExit as e:
                rc = e.code if isinstance(e.code, int) else 1
            return rc, buf.getvalue(), err.getvalue()

        _, o_a, _ = cli(["--grid", ref_p, grey, "--threshold", "0"])
        _, o_b, _ = cli(["--grid", ref_p, grey, "--threshold", "255"])
        check("--threshold reaches --grid via the CLI", o_a != o_b)
        # BOTH values must sit below the old max(cols, 8) clamp, or the clamp maps
        # them to different grids anyway and the arm passes with the defect present.
        _, o_c, _ = cli(["--regions", ref_p, dgrad_p, "--cells", "2x2"])
        _, o_d, _ = cli(["--regions", ref_p, dgrad_p, "--cells", "4x3"])
        check("--cells reaches --regions via the CLI (2x2 vs 4x3)", o_c != o_d)
        # --locate ECHOES its grid in the header, straight from the parsed values,
        # so comparing whole output is satisfied by the echo whether or not the
        # options reach region_match: a review reinstated the drop-the-options
        # defect in --locate alone and the suite stayed 27/27 green while all three
        # --cells values gave byte-identical answers. Compare the RESULT ROWS.
        def rows(text):
            return "\n".join(l for l in text.splitlines() if not l.startswith("==="))
        _, o_e, _ = cli(["--locate", ref_p, td, "--cells", "2x2"])
        _, o_f, _ = cli(["--locate", ref_p, td, "--cells", "4x3"])
        check("--cells reaches --locate via the CLI (2x2 vs 4x3)",
              rows(o_e) != rows(o_f))
        _, o_g, _ = cli(["--locate", ref_p, td, "--threshold", "1"])
        _, o_h, _ = cli(["--locate", ref_p, td, "--threshold", "200"])
        check("--threshold reaches --locate via the CLI (1 vs 200)",
              rows(o_g) != rows(o_h))

        # Options a mode does not USE must be refused, not accepted and dropped.
        for args in (["--rank", ref_p, td, "--threshold", "40"],
                     ["--rank", ref_p, td, "--cells", "4x3"],
                     [ref_p, grey, "--threshold", "40"],
                     ["--grid", ref_p, grey, "--bogus"]):
            rc, _, _ = cli(args)
            check(f"{args[0] if args[0].startswith('--') else 'default'} refuses "
                  f"{args[-2] if args[-1].isdigit() or 'x' in args[-1] else args[-1]}",
                  rc != 0)

        # Silent success: every one of these returned 0 before, and none is covered
        # by the arms above.
        bad = os.path.join(td, "bad"); os.makedirs(bad, exist_ok=True)
        open(os.path.join(bad, "x.png"), "w").write("not an image")
        for name, args in (("--grid with the candidate forgotten",
                            ["--grid", ref_p, "--threshold", "40"]),
                           ("--locate with an unreadable reference",
                            ["--locate", os.path.join(td, "nope.png"), td]),
                           ("--rank over an all-corrupt directory",
                            ["--rank", ref_p, bad]),
                           ("--locate over an all-corrupt directory",
                            ["--locate", ref_p, bad]),
                           ("--rank on a missing directory",
                            ["--rank", ref_p, os.path.join(td, "nodir")]),
                           ("a corrupt candidate in the default mode",
                            [ref_p, os.path.join(bad, "x.png")])):
            rc, _, _ = cli(args)
            check(f"{name} exits non-zero (got {rc})", rc != 0)

        # A directory with ONE good image and one corrupt one: the run succeeds, so
        # no exit code can report the skip. It must be named on stderr, or a
        # partially-read directory is silently reported as a complete ranking.
        mixed = os.path.join(td, "mixed"); os.makedirs(mixed, exist_ok=True)
        ref.save(os.path.join(mixed, "good.png"))
        open(os.path.join(mixed, "broken.png"), "w").write("not an image")
        rc, _, errtxt = cli(["--rank", ref_p, mixed])
        check(f"a skipped file in a mixed directory is NAMED on stderr (rc={rc})",
              "broken.png" in errtxt)

    print("\nselftest: " + ("all passed" if ok else "FAILURES PRESENT"))
    return 0 if ok else 1


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
        # The region modes read their scores numerically rather than as a picture,
        # so they default to a finer grid -- 64x36, which is what this file and
        # AGENTS.md have always documented. The CLI previously defaulted them to
        # 16x9 and then clamped any smaller --cells up to 8, so `--regions --cells
        # 2x2`, `4x2` and `8x8` returned byte-identical output with no grid printed,
        # and the documented default was not the one in force.
        fine = mode in ("--locate", "--regions")
        cols, rows, thr = _take_opts(rest, () if mode == "--rank" else spatial,
                                     64 if fine else 16, 36 if fine else 9)
        if mode == "--rank":
            _need(rest, 2, "--rank REFERENCE DIR")
            ref, d = rest[0], rest[1]
            ref_px = load(ref)                     # loud if unreadable
            scored, failures = _scan_dir(d, lambda p: report(ref, p, ref_px))
            print("\n=== ranked by likeness (sqrt(palette x hue/structure)) ===")
            zeros = 0
            for sc, f in sorted(scored, key=lambda t: (-t[0], t[1])):
                mark = "   (no likeness -- order below here is filename only)" \
                       if sc <= 0.0 and zeros == 0 else ""
                if sc <= 0.0:
                    zeros += 1
                print(f"  {sc:6.3f}  {f}{mark}")
            return _report_failures(failures, len(scored))
        if mode == "--locate":
            _need(rest, 2, "--locate REFERENCE DIR")
            ref, d = rest[0], rest[1]
            load_full(ref)                         # loud if unreadable
            found, failures = _scan_dir(
                d, lambda p: region_match(ref, p, cols=cols, rows=rows,
                                          threshold=thr, quiet=True))
            print(f"=== best-matching region of {os.path.basename(ref)} per surface "
                  f"(tol={thr}, grid={cols}x{rows}) ===")
            # Tie-break on mean|d| ascending, so a flat fill that matches the same
            # cell fraction cannot outrank a closer surface on filename order.
            for (frac, name, mad), f in sorted(found, key=lambda t: (-t[0][0], t[0][2], t[1]))[:20]:
                print(f"  {frac*100:5.1f}%  {name:<15} mean|d|={mad:6.1f}  {f}")
            return _report_failures(failures, len(found))
        if mode == "--grid":
            _need(rest, 2, "--grid REFERENCE CANDIDATE")
            for c in rest[1:]:
                grid_compare(rest[0], c, cols, rows, thr)
            return 0
        _need(rest, 2, "--regions REFERENCE CANDIDATE")
        for c in rest[1:]:
            region_match(rest[0], c, cols=cols, rows=rows, threshold=thr)
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
