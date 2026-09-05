#!/usr/bin/env python3
"""when_matched.py — where in a run does it match a guard's reviewed references, and when?

A content guard reports one verdict about one window. When it fails on SSIM alone -- "it renders and
does not match" -- that verdict cannot distinguish the two cases that matter:

  * the title still reaches the reviewed state, but at a different WALL-CLOCK TIME, because
    something upstream changed the boot's duration. The route's anchors then land elsewhere and the
    window looks at the wrong moment. Re-capturing the baseline here would be laundering: the
    references are still correct and the ROUTE is what drifted.
  * the title never reaches the reviewed state at all. That is content loss, and re-capturing the
    baseline would delete the evidence of it.

This tool separates them. It scores EVERY sample of a capture against the guard's stored
`structural_references` and prints where the best matches are, so "does anything in this run match,
and when?" is answered with a number instead of an argument.

It reads the `luma16x9` signatures the capture manifest already carries, so it needs no images and
no image library, and it scores with the same SSIM the checker uses.

usage:
    when_matched.py <snapshot-name> <capture.jsonl> [--snapshots <snapshots.json>] [--top N]

exit status:
    0   the sweep ran; the printed scores ARE the answer, including when nothing matches
    2   refused -- the snapshot, the manifest or the references could not be read, and no score
        below is a result
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def decode_luma(value):
    if len(value) != 288:
        raise ValueError("luma16x9 must be 288 hexadecimal characters")
    return tuple(bytes.fromhex(value))


def structural_similarity(left, right):
    """SSIM over two equally sized luminance signatures -- the checker's own formula.

    Kept byte-for-byte equivalent to `snapshot.py`'s `structural_similarity`, INCLUDING its clamp to
    [-1, 1]. The clamp is defensive rather than load-bearing (SSIM cannot leave that interval
    mathematically), but this tool's entire value is that it cannot disagree with the checker about
    what "matching" means, and a scorer that agrees only up to a clamp is a second oracle waiting to
    produce an unresolvable disagreement. Note `snaps.py` carries a copy WITHOUT the clamp; if these
    three ever need to diverge, they should stop being copies instead.
    """
    if len(left) != len(right) or not left:
        raise ValueError("SSIM inputs must have the same non-zero length")
    count = len(left)
    mean_left = sum(left) / count
    mean_right = sum(right) / count
    var_left = sum((v - mean_left) ** 2 for v in left) / count
    var_right = sum((v - mean_right) ** 2 for v in right) / count
    covariance = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right)) / count
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    denominator = (mean_left ** 2 + mean_right ** 2 + c1) * (var_left + var_right + c2)
    if denominator == 0:
        return 1.0
    return max(-1.0, min(1.0,
        ((2 * mean_left * mean_right + c1) * (2 * covariance + c2)) / denominator))


def coverage(record):
    """Fraction of non-black pixels, from whichever field this manifest carries.

    `snapshot.py`'s evidence file stores `nonblack_ratio`; a `tools/screenshot` manifest stores the
    raw `nonblack_rgb_pixels` count instead. Returning 0.0 for the second spelling would print a
    fully covered frame as empty -- an instrument failing in the direction of the conclusion, which
    is the one direction that must never be silent. None means "this manifest does not say".
    """
    if "nonblack_ratio" in record:
        return record["nonblack_ratio"]
    pixels = record.get("nonblack_rgb_pixels")
    width, height = record.get("width"), record.get("height")
    if pixels is None or not width or not height:
        return None
    return pixels / float(width * height)


def load_entry(snapshots_path, name):
    with open(snapshots_path) as handle:
        document = json.load(handle)
    entries = document if isinstance(document, list) else document.get("snapshots", document)
    for entry in entries:
        if isinstance(entry, dict) and entry.get("name") == name:
            return entry
    raise KeyError(f"no snapshot named {name!r} in {snapshots_path}")


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", help="snapshot entry whose references to score against")
    parser.add_argument("manifest", help="a run's capture.jsonl")
    parser.add_argument("--snapshots", default=os.path.join(HERE, "snapshots.json"))
    parser.add_argument("--top", type=int, default=10, help="how many best samples to list")
    args = parser.parse_args(argv)

    try:
        entry = load_entry(args.snapshots, args.name)
        references = [decode_luma(r["luma16x9"]) for r in entry.get("structural_references", [])]
        if not references:
            raise ValueError(f"{args.name} carries no structural_references to score against")
        samples = []
        with open(args.manifest) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("type") != "sample" or "luma16x9" not in record:
                    continue
                samples.append(record)
        if not samples:
            raise ValueError(f"{args.manifest} carries no samples with a luma16x9 signature")
    except Exception as error:                                   # noqa: BLE001 -- refusal is a contract
        print(f"when_matched.py: refused: {error}", file=sys.stderr)
        return 2

    floor = entry.get("min_structural_similarity")
    after = entry.get("capture_after_seconds")
    before = entry.get("capture_before_seconds")

    scored = []
    for record in samples:
        signature = decode_luma(record["luma16x9"])
        best = max(structural_similarity(signature, reference) for reference in references)
        scored.append((best, record))

    in_window = [(s, r) for s, r in scored
                 if after is not None and before is not None
                 and after <= r.get("elapsed_seconds", -1) <= before]

    print(f"{args.name}: {len(samples)} samples against {len(references)} reviewed references"
          f"{f', SSIM floor {floor}' if floor is not None else ''}")
    if after is not None and before is not None:
        print(f"guarded window: {after}..{before} s  ({len(in_window)} samples inside it)")

    ordered = sorted(scored, key=lambda pair: pair[0], reverse=True)
    print(f"\nbest {min(args.top, len(ordered))} samples anywhere in the run:")
    for best, record in ordered[:args.top]:
        mark = ""
        if floor is not None:
            mark = "  PASS" if best >= floor else ""
        inside = ""
        if after is not None and before is not None:
            inside = " [in window]" if after <= record.get("elapsed_seconds", -1) <= before else ""
        ratio = coverage(record)
        print("  t=%8.1fs  ssim=%.4f  colors=%6s  nonblack=%s%s%s"
              % (record.get("elapsed_seconds", 0.0), best,
                 record.get("distinct_rgb_colors"),
                 "n/a" if ratio is None else "%.4f" % ratio, mark, inside))

    if in_window:
        window_best = max(s for s, _ in in_window)
        print(f"\nbest inside the guarded window: {window_best:.4f}")
    run_best = ordered[0][0]
    print(f"best anywhere:                  {run_best:.4f}"
          f"  at t={ordered[0][1].get('elapsed_seconds', 0.0):.1f}s")
    if floor is not None:
        passing = [r for s, r in scored if s >= floor]
        print(f"samples at or above the floor anywhere in the run: {len(passing)}")
        if passing:
            print("  spanning t=%.1fs .. t=%.1fs" % (passing[0].get("elapsed_seconds", 0.0),
                                                     passing[-1].get("elapsed_seconds", 0.0)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
