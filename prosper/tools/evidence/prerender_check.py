#!/usr/bin/env python3
"""Is this screenshot actually the game's own pre-rendered picture?

Instrument trap 230. A title that blits a full-screen loading picture produces the most convincing
false progression evidence there is: it looks BETTER than the emulator could plausibly render, which
reads as success rather than as the warning it is. One such capture reached a merged PR, a BLOG.md
entry and COMPATIBILITY.md before a human recognised the artwork.

This decodes every same-resolution image asset in a dump and diffs it against the candidate. A real
match is unambiguous -- the true asset scored 0.02/255 while the next closest scored 52.77 -- so the
answer is not a judgement call.

    python3 prerender_check.py <screenshot> <dump-root> [--top N] [--threshold F]

Exit 0 = no asset matches (the frame is a genuine render, as far as THIS check can tell).
Exit 2 = an asset matches: the frame is the game's own artwork and is not progression evidence.
Exit 1 = the check could not run (missing Pillow, unreadable candidate, no comparable assets) --
         deliberately distinct from 0, because "could not check" must never read as "verified".

Scope, stated because a clean pass is easy to over-read: this finds FULL-SCREEN blits of assets
stored as ordinary images. It cannot see a pre-rendered movie frame, an asset stored in a container
it cannot decode, or artwork composited with other elements. A pass means "no stored picture
explains this frame", not "this frame is rendered 3D".
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

    try:
        from PIL import Image, ImageChops
    except ImportError:
        print("prerender_check: Pillow is required (pip install pillow)", file=sys.stderr)
        return 1
    if not os.path.isdir(dump_root):
        print("prerender_check: not a directory: %s" % dump_root, file=sys.stderr)
        return 1
    try:
        want = Image.open(candidate).convert('RGB')
    except Exception as exc:                                    # noqa: BLE001
        print("prerender_check: cannot read %s: %s" % (candidate, exc), file=sys.stderr)
        return 1

    scored, skipped_size, undecodable = [], 0, 0
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
            asset = asset.convert('RGB')
            hist = ImageChops.difference(asset, want).convert('L').histogram()
        except Exception:                                       # noqa: BLE001
            undecodable += 1
            continue
        total = sum(hist) or 1
        mean = sum(i * c for i, c in enumerate(hist)) / total
        within8 = sum(hist[:8]) / total * 100.0
        scored.append((mean, within8, os.path.relpath(path, dump_root)))

    if not scored:
        # No comparable asset is NOT a pass: nothing was actually compared.
        print("prerender_check: no asset matched this resolution (%dx%d); "
              "%d wrong-size, %d undecodable -- NOTHING WAS COMPARED" %
              (want.size[0], want.size[1], skipped_size, undecodable), file=sys.stderr)
        return 1

    scored.sort()
    print("compared %d same-size asset(s) (%d wrong-size, %d undecodable)"
          % (len(scored), skipped_size, undecodable))
    for mean, within8, rel in scored[:top]:
        print("  mean=%7.2f/255  within8=%6.2f%%  %s" % (mean, within8, rel))

    best_mean, best_within8, best_rel = scored[0]
    if best_mean < threshold:
        print("\nMATCH: this frame is the game's own picture asset %s\n"
              "  mean abs diff %.2f/255, %.2f%% of pixels within 8/255.\n"
              "  It is NOT progression evidence -- displaying a stored image needs no world "
              "rendering.\n  See instrument trap 230 in docs/GAME_COMPAT_ORCHESTRATION.md."
              % (best_rel, best_mean, best_within8))
        return 2
    print("\nno asset matches (closest %.2f/255, threshold %.2f) -- no STORED PICTURE explains this\n"
          "frame. That is not the same as proving it is rendered 3D; see this tool's scope note."
          % (best_mean, threshold))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
