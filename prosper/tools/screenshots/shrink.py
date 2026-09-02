#!/usr/bin/env python3
"""Shrink committed screenshots so the repository stays clonable.

WHY THIS EXISTS. Measured 2026-09-02 on a 621 MiB pack: `assets/screenshots` was 398 MiB of it and
every other image 73 MiB, against **18 MiB for all source, docs and tests in the entire history**.
Images were ~96% of the blob content. The cause was not that there are many screenshots (121) but
that each was a full 3840x2160 PNG averaging 2.5 MiB, the largest 20.6 MiB. At that rate the repo
reaches GitHub's 10 GB ceiling in about four years; at the size this tool produces, it does not
reach it in a working lifetime.

WHAT IT DOES. Downscales to a maximum width (default 1920 -- a progress blog does not need 4K) and
re-encodes to WebP, which is built for exactly this content. PNG is a poor fit for game captures:
it is lossless, so a photographic frame with lighting gradients cannot compress, which is why these
files are enormous. Quantising to a palette would keep the .png extension but visibly band those
same gradients, so the extension changes instead and callers are rewritten -- mechanical, and
verifiable by checking that every reference resolves afterwards.

WHAT IT DOES NOT DO. It cannot shrink the repository that already exists. Git keeps every blob it
has ever seen, so re-encoding ADDS the small versions while the large originals stay in history: a
full clone gets marginally bigger, not smaller. What improves -- immediately and a lot -- is every
SHALLOW or sparse clone, which fetches only the current tree: CI runs one per job. Reclaiming the
historical 398 MiB needs a history rewrite, which changes every commit SHA and breaks the commit
references in every existing pull request; that trade is deliberately not made here.

    shrink.py <path>...            re-encode in place (a directory is walked)
    shrink.py --check <path>...    report what would change; exit 1 if anything would
    shrink.py --max-width N        default 1920
    shrink.py --quality N          default 85
"""
import argparse
import os
import sys

try:
    from PIL import Image
except ImportError:
    sys.exit("shrink: Pillow is required (python3 -m pip install Pillow)")

SOURCE_SUFFIXES = (".png", ".bmp", ".jpg", ".jpeg")


def candidates(paths):
    for p in paths:
        if os.path.isdir(p):
            for root, _dirs, files in os.walk(p):
                for f in sorted(files):
                    if f.lower().endswith(SOURCE_SUFFIXES):
                        yield os.path.join(root, f)
        elif p.lower().endswith(SOURCE_SUFFIXES):
            yield p


def shrink_one(path, max_width, quality, dry_run):
    """Returns (before_bytes, after_bytes, out_path) or None when nothing to do."""
    before = os.path.getsize(path)
    with Image.open(path) as im:
        width, height = im.size
        # Never upscale: a capture already at or below the cap keeps its pixels, and only the
        # encoding changes.
        if width > max_width:
            new_height = max(1, round(height * max_width / width))
            resized = im.convert("RGB").resize((max_width, new_height), Image.LANCZOS)
        else:
            resized = im.convert("RGB")
        out_path = os.path.splitext(path)[0] + ".webp"
        if dry_run:
            # Encode to memory so --check reports the real number rather than an estimate.
            import io
            buf = io.BytesIO()
            resized.save(buf, "WEBP", quality=quality, method=6)
            return before, buf.tell(), out_path
        resized.save(out_path, "WEBP", quality=quality, method=6)
    after = os.path.getsize(out_path)
    if out_path != path:
        os.remove(path)
    return before, after, out_path


def main():
    ap = argparse.ArgumentParser(description="Shrink committed screenshots.")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--max-width", type=int, default=1920)
    ap.add_argument("--quality", type=int, default=85)
    ap.add_argument("--check", action="store_true",
                    help="report only; exit 1 if any file would change")
    args = ap.parse_args()

    total_before = total_after = 0
    changed = 0
    for path in candidates(args.paths):
        result = shrink_one(path, args.max_width, args.quality, args.check)
        if result is None:
            continue
        before, after, out_path = result
        total_before += before
        total_after += after
        changed += 1
        print(f"  {before/1048576:7.2f} -> {after/1048576:6.2f} MiB  "
              f"{os.path.basename(path)} -> {os.path.basename(out_path)}")

    if not changed:
        print("nothing to do")
        return 0
    saved = total_before - total_after
    print(f"\n{changed} file(s): {total_before/1048576:.1f} MiB -> {total_after/1048576:.1f} MiB "
          f"({saved/1048576:.1f} MiB saved, {total_after/total_before*100:.1f}% of original)")
    return 1 if args.check else 0


if __name__ == "__main__":
    sys.exit(main())
