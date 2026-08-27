#!/usr/bin/env python3
"""Is this screenshot actually the game's own pre-rendered picture?

Instrument trap 230. A title that blits a full-screen loading picture produces the most convincing
false progression evidence there is: it looks BETTER than the emulator could plausibly render, which
reads as success rather than as the warning it is. One such capture reached a merged PR, a BLOG.md
entry and COMPATIBILITY.md before a human recognised the artwork.

This decodes every same-resolution image asset in a dump and diffs it against the candidate. A real
match is unambiguous -- the true asset scored 0.02/255 while the next closest scored 52.77 -- so the
answer is not a judgement call.

    python3 prerender_check.py <screenshot> <dump-root> [--top N] [--threshold F] [--overlap P]

Exit 0 = no asset explains this frame (as far as THIS check can tell).
Exit 2 = an asset matches, either EXACTLY or well enough to dominate the frame. Not progression
         evidence without an explanation of what prosper actually rendered.
Exit 1 = the check could not run (missing Pillow, unreadable candidate, no comparable assets) --
         deliberately distinct from 0, because "could not check" must never read as "verified".

Two verdicts share exit 2 on purpose, because they need the same care:
  EXACT      mean abs diff below --threshold (default 1.0/255): the frame IS the stored picture.
  DOMINATED  at least --overlap percent of the INFORMATIVE pixels (default 75) within 8/255 of a
             stored picture. This is the case a mean-only test misses: a real loading screen usually
             has something drawn ON it -- a progress bar, a hint caption -- and a 2%-of-height bar
             is enough to lift the mean past any sane threshold while 98% of the frame is still the
             asset. There is deliberately no separate mean ceiling: an overlap
             this high already bounds the mean, so a ceiling could only add false negatives.

"Informative" is load-bearing and was added after review. Scoring every pixel makes the overlap
measure DARKNESS rather than identity: a black frame against a black asset scores 100%, and this
project renders plenty of near-black frames. A repo screenshot scored 88.77% against an unrelated
`pic1.dds` purely on shared blackness, and synthetic flat frames reach 100.00% against real assets.
So a pixel counts only when at least one of the two images has content there (`--floor`, default
16/255), and the fraction that qualifies is reported as COVERAGE. Below --min-coverage (default 25%)
the comparison carries no information and the asset is not scored at all -- a black frame matching a
black asset is "nothing was compared" wearing a verdict, which is the exact failure this whole tool
exists to prevent.

Scope, stated because a clean pass is easy to over-read: this finds assets stored as ordinary
images. It cannot see a pre-rendered movie frame or an asset in a container it cannot decode. A pass
means "no stored picture explains this frame", not "this frame is rendered 3D".
"""
import os
import sys


def iter_assets(root):
    exts = ('.dds', '.png', '.tga', '.bmp', '.jpg', '.jpeg')
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(exts):
                yield os.path.join(dirpath, name)


def main(argv):
    args = [a for a in argv[1:] if not a.startswith('--')]
    if len(args) < 2:
        print(__doc__)
        return 1
    candidate, dump_root = args[0], args[1]

    def opt(flag, default, cast):
        if flag in argv:
            try:
                return cast(argv[argv.index(flag) + 1])
            except (IndexError, ValueError):
                print("bad value for %s" % flag, file=sys.stderr)
                raise SystemExit(1)
        return default

    top = opt('--top', 5, int)
    threshold = opt('--threshold', 1.0, float)
    overlap = opt('--overlap', 75.0, float)
    floor = opt('--floor', 16, int)
    min_coverage = opt('--min-coverage', 25.0, float)
    structure_floor = opt('--structure-floor', 3.0, float)

    try:
        from PIL import Image
    except ImportError:
        print("prerender_check: Pillow is required (pip install pillow)", file=sys.stderr)
        return 1
    try:
        import numpy as np
    except ImportError:
        print("prerender_check: numpy is required (pip install numpy)", file=sys.stderr)
        return 1
    if not os.path.isdir(dump_root):
        print("prerender_check: not a directory: %s" % dump_root, file=sys.stderr)
        return 1
    try:
        want = Image.open(candidate).convert('RGB')
        wv = np.asarray(want, dtype=np.int16)
    except Exception as exc:                                    # noqa: BLE001
        print("prerender_check: cannot read %s: %s" % (candidate, exc), file=sys.stderr)
        return 1

    # A FEATURELESS frame identifies nothing, and this is the general form of the defect review
    # found at the dark end. Brightness is not information: an all-white frame against an all-white
    # `pic2.dds` scores mean 0.00 exactly as an all-black pair does, and many dumps ship both. What
    # a comparison needs is STRUCTURE. A capture with none is also not progression evidence in the
    # first place ("black or diagnostic-only captures are not progression evidence" -- CLAUDE.md).
    candidate_structure = float(np.asarray(want.convert('L'), dtype=np.float64).std())
    if candidate_structure < structure_floor:
        print("candidate is FEATURELESS (luma sd %.2f, below %.2f) -- a flat or near-flat frame.\n"
              "It cannot be matched against anything, at any brightness: an all-white frame matches\n"
              "an all-white asset exactly as an all-black pair does. An empty capture is also not\n"
              "progression evidence whatever any asset says. Not a pass."
              % (candidate_structure, structure_floor))
        return 1

    scored, skipped_size, undecodable, featureless_assets = [], 0, 0, 0
    for path in iter_assets(dump_root):
        try:
            asset = Image.open(path)
        except Exception:                                       # noqa: BLE001
            undecodable += 1
            continue
        if asset.size != want.size:
            skipped_size += 1
            continue
        try:
            av = np.asarray(asset.convert('RGB'), dtype=np.int16)
        except Exception:                                       # noqa: BLE001
            undecodable += 1
            continue
        # A featureless ASSET matches any frame of the same flat colour, so it can never support a
        # verdict either. Excluded here rather than filtered later, so it cannot reach any report.
        if float(np.asarray(asset.convert('L'), dtype=np.float64).std()) < structure_floor:
            featureless_assets += 1
            continue
        delta = np.abs(av - wv)                       # int16: no wraparound on a 0..255 difference
        per_pixel = delta.max(axis=2)
        mean = float(delta.mean())
        # A pixel is informative only if at least one side has content there. Without this the
        # overlap measures shared darkness and reports identity.
        informative = (av.max(axis=2) >= floor) | (wv.max(axis=2) >= floor)
        n_inf = int(informative.sum())
        coverage = n_inf / per_pixel.size * 100.0
        within8 = float((per_pixel[informative] <= 8).mean() * 100.0) if n_inf else 0.0
        scored.append((mean, within8, os.path.relpath(path, dump_root), coverage))

    if not scored:
        # No comparable asset is NOT a pass: nothing was actually compared.
        print("prerender_check: no comparable asset at this resolution (%dx%d); "
              "%d wrong-size, %d undecodable, %d featureless -- NOTHING WAS COMPARED" %
              (want.size[0], want.size[1], skipped_size, undecodable, featureless_assets),
              file=sys.stderr)
        return 1

    scored.sort()
    print("compared %d same-size asset(s) (%d wrong-size, %d undecodable, %d featureless)"
          % (len(scored), skipped_size, undecodable, featureless_assets))
    for mean, within8, rel, cov in scored[:top]:
        # Flag rows whose overlap the verdict must ignore, so a reader never sees a strong-looking
        # within8 sitting above a pass without knowing it was excluded.
        mark = '' if cov >= min_coverage else '  [overlap unusable: coverage < %.0f%%]' % min_coverage
        print("  mean=%7.2f/255  within8=%6.2f%%  coverage=%6.2f%%  %s%s"
              % (mean, within8, cov, rel, mark))

    # Every verdict is drawn ONLY from comparisons that carry information. Gating on the best
    # coverage anywhere in the dump is not enough -- an unrelated asset with content would license a
    # verdict about a black-vs-black pair, which is how the first attempt at this gate still reported
    # a pure-black frame as an EXACT MATCH.
    usable = [row for row in scored if row[3] >= min_coverage]

    # The two statistics need DIFFERENT gates, and conflating them hid a match.
    #
    # `mean` is a whole-frame average: darkness cannot corrupt it, because a dark region contributes
    # its (small) real difference rather than being scored as agreement. Two STRUCTURED images at
    # mean 0.00 are the same image whatever fraction of them is bright. So the exact verdict needs
    # only the structure gate.
    #
    # `within8` is a fraction over a SUBSET, so it is exactly the statistic a small or dark subset
    # can mislead -- hence the coverage floor, but on that verdict alone.
    #
    # Applying the coverage floor to both let the tool PASS a frame byte-identical to the game's own
    # artwork: `1/PIX/LEGAL_2.DDS` is white text on black, structured (luma sd 49.61) but only 4.09%
    # above the content floor, so at mean 0.00 it never reached `usable` and the tool reported "no
    # asset explains this frame". That class -- full-screen pre-rendered pictures on dark
    # backgrounds: legal notices, credits, title cards, dark loading art -- is precisely trap 230's.
    exact = min(scored, key=lambda row: row[0])
    if exact[0] < threshold:
        print("\nEXACT MATCH: this frame IS the game's own picture asset %s\n"
              "  mean abs diff %.2f/255, %.2f%% of informative pixels within 8/255 "
              "(coverage %.2f%%).\n"
              "  NOT progression evidence -- displaying a stored image needs no world rendering.\n"
              "  See instrument trap 230 in docs/GAME_COMPAT_ORCHESTRATION.md."
              % (exact[2], exact[0], exact[1], exact[3]))
        return 2

    if not usable:
        print("\nno asset carries enough content for an OVERLAP comparison (best coverage %.2f%%,\n"
              "below the %.2f%% bar), and none is an exact match. A near-empty frame and a\n"
              "near-empty asset agree on darkness, not identity -- this is not a pass."
              % (max(row[3] for row in scored), min_coverage))
        return 1

    # There is deliberately NO separate mean ceiling here. One was added and removed: an overlap of
    # >= 75% already bounds the mean at <= 69.75/255, so
    # any ceiling below that can only discard frames the bar would have flagged -- it cannot prevent
    # a false positive, only manufacture false negatives. The bound is
    #     mean <= 0.75*f*8 + 0.25*f*255 + (1-f)*15  =  f*69.75 + (1-f)*15
    # for informative fraction f -- increasing in f, so it maximises at FULL coverage, at 69.75. It
    # was first written here as 63.75, which dropped the matching pixels' own contribution: a pixel
    # counted as matching may differ by up to 8 per channel, not 0, and 0.75*8 = 6.00 is exactly the
    # gap. A hand-built frame (75% of pixels differing by exactly 8, 25% by 255) reaches mean 69.67
    # at overlap 75.00% -- above the figure the earlier comment said could not be reached, so the
    # tool refuted its own comment. Derive it rather than quoting it. Measured alongside it, 100 of
    # 233 structured assets
    # (42.9%) carrying a bright caption panel reach mean 40.5-59.25 while still exceeding the bar,
    # so a ceiling of 40 silently dropped nearly half of them. (An earlier "4 of 13" here came from
    # a probe that filtered to 1920x1080 and so excluded every 4K asset -- which is exactly where the
    # worst means live. Cite the bound; a count is a property of whoever's sample.)
    dominant = max(usable, key=lambda row: row[1])
    if dominant[1] >= overlap:
        print("\nDOMINATED: %.2f%% of this frame's INFORMATIVE pixels are within 8/255 of the\n"
              "  stored asset %s (coverage %.2f%%, mean abs diff %.2f/255 -- an overlay or caption\n"
              "  lifts the mean, which is why a mean-only test would have passed this).\n"
              "  The frame is mostly stored artwork. It is not progression evidence on its own:\n"
              "  say what prosper actually rendered here.\n"
              "  See instrument trap 230 in docs/GAME_COMPAT_ORCHESTRATION.md."
              % (dominant[1], dominant[2], dominant[3], dominant[0]))
        return 2

    print("\nno asset explains this frame (closest comparison %.2f/255; highest overlap %.2f%%\n"
          "with %s at %.2f%% coverage, under the %.2f%% dominance bar). That is not the same as\n"
          "proving it is rendered 3D; see this tool's scope note."
          % (exact[0], dominant[1], dominant[2], dominant[3], overlap))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
