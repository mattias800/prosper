#!/usr/bin/env python3
"""Aggregate `[compute-phase]` lines into a compute-side frame decomposition.

WHY THIS EXISTS
---------------
`PROSPER_COMPUTE_PHASE_TIMING=1` already emits a complete 17-timer decomposition of every compute
dispatch -- but it emits it *per dispatch*, one line each. A routed title produces tens of thousands
of those lines, so the instrument that knows where compute time goes has, in practice, been unreadable
at run scale. `PROSPER_RENDER_TIMING` is readable precisely because it rolls its phases up; this tool
gives the compute side the same shape, offline, without touching the emulator or perturbing a run.

Reads a run log (stdin or a path), parses `[compute-phase]` records, and prints:

  * the phase table -- ms, share of total, and mean ms/dispatch -- nested so each child sits under
    the parent whose interval contains it;
  * "unattributed" rows, i.e. parent time its own children do not explain (this is where an
    unmeasured cost hides, so it is printed rather than silently dropped);
  * the programs that cost the most, each with its dominant leaf phase.

`setup_ms` deliberately spans the image-binding loop as well as descriptor validation and buffer
binding, and that loop has no sub-timer on the `[compute-phase]` line -- so on an image-heavy title
most of `setup` lands in its "unattributed" row. Add `PROSPER_COMPUTE_IMAGE_TIMING=1` to the run and
this tool will also roll up the resulting `[compute-image]` records, which decompose exactly that
interval. Run both switches together whenever setup is the dominant phase.

Usage:
    compute_phase_report.py [LOG ...] [--top N] [--since-submit N] [--program 0xADDR] [--csv]
"""

import argparse
import re
import sys
from collections import defaultdict

RECORD = re.compile(r"\[compute-phase\]\s+(.*)")
IMAGE_RECORD = re.compile(r"\[compute-image\]\s+(.*)")
FIELD = re.compile(r"(\w+)=(0x[0-9a-fA-F]+|[-\d.]+)")

# `[compute-image]` decomposes one image binding inside the setup_ms interval. Several of its field
# names (cache_ms, prepare_ms) collide with `[compute-phase]` fields that mean something else
# entirely, so the two record types are parsed separately and never merged into one dictionary.
IMAGE_PHASES = [
    ("query (driver caps)", "query_ms"),
    ("import (renderer RTT)", "import_ms"),
    ("cache lookup + validate", "cache_ms"),
    ("staging alloc", "staging_ms"),
    ("prepare upload", "prepare_ms"),
    ("image allocation", "allocation_ms"),
    ("view", "view_ms"),
    ("sampler", "sampler_ms"),
]

# (label, key, parent) -- parent is the phase whose measured interval contains this one.
# `total_ms` is the root; the top-level five partition it by construction in execute_item().
PHASES = [
    ("setup",              "setup_ms",              "total_ms"),
    ("  validate",         "setup_validate_ms",     "setup_ms"),
    ("  buffers",          "setup_buffers_ms",      "setup_ms"),
    ("pipeline",           "pipeline_ms",           "total_ms"),
    ("dispatch (GPU)",     "dispatch_ms",           "total_ms"),
    ("writeback",          "writeback_ms",          "total_ms"),
    ("  prepare",          "writeback_prepare_ms",  "writeback_ms"),
    ("  buffers",          "writeback_buffers_ms",  "writeback_ms"),
    ("  images",           "writeback_images_ms",   "writeback_ms"),
    ("    map",            "map_ms",                "writeback_images_ms"),
    ("    prepare",        "prepare_ms",            "writeback_images_ms"),
    ("    watch",          "watch_ms",              "writeback_images_ms"),
    ("    pack",           "pack_ms",               "writeback_images_ms"),
    ("    layout",         "layout_ms",             "writeback_images_ms"),
    ("    notify",         "notify_ms",             "writeback_images_ms"),
    ("    cache",          "cache_ms",              "writeback_images_ms"),
    ("  publish",          "writeback_publish_ms",  "writeback_ms"),
    ("cleanup",            "cleanup_ms",            "total_ms"),
]
# Leaf phases are the ones an optimisation can actually target; a parent is just their sum.
LEAVES = [k for (_, k, _) in PHASES if k not in {p for (_, _, p) in PHASES}]


def _fields(text):
    out = {}
    for key, value in FIELD.findall(text):
        out[key] = int(value, 16) if value.startswith("0x") else float(value)
    return out


def parse(streams, since_submit, program):
    records, images = [], []
    for stream in streams:
        for line in stream:
            match = RECORD.search(line)
            if match:
                fields = _fields(match.group(1))
                if "total_ms" not in fields:
                    continue
                if since_submit is not None and fields.get("submit", 0) < since_submit:
                    continue
                if program is not None and fields.get("code") != program:
                    continue
                records.append(fields)
                continue
            match = IMAGE_RECORD.search(line)
            if match:
                fields = _fields(match.group(1))
                # `ms` is this binding's whole interval; without it the record cannot be attributed.
                if "ms" not in fields:
                    continue
                if program is not None and fields.get("code") != program:
                    continue
                images.append(fields)
    return records, images


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="*", help="run logs (default: stdin)")
    ap.add_argument("--top", type=int, default=12, help="programs to list (default 12)")
    ap.add_argument("--since-submit", type=int, default=None,
                    help="ignore dispatches before this submit number (skips boot/warmup)")
    ap.add_argument("--program", type=lambda s: int(s, 0), default=None,
                    help="restrict to one program code address")
    ap.add_argument("--csv", action="store_true", help="emit the phase table as CSV")
    args = ap.parse_args()

    streams = []
    if args.logs:
        streams = [open(p, "r", errors="replace") for p in args.logs]
    else:
        streams = [sys.stdin]
    records, images = parse(streams, args.since_submit, args.program)

    if not records:
        print("no [compute-phase] records found "
              "(run with PROSPER_COMPUTE_PHASE_TIMING=1)", file=sys.stderr)
        return 1

    # A dispatch that fails leaves execute_item() through an early break, so phase_dispatch and
    # phase_writeback are never advanced past phase_start. Its `dispatch_ms` is then computed
    # backwards (and prints NEGATIVE) while `cleanup_ms` swallows the whole dispatch. Those records
    # carry a valid `total_ms` and nothing else, so they are reported separately and never summed
    # into the phase table -- mixing them in silently inverts it. See #1732.
    failed = [r for r in records if r.get("ok", 1) == 0]
    records = [r for r in records if r.get("ok", 1) != 0]
    if not records:
        print(f"all {len(failed)} dispatches in this log failed; no phase decomposition is "
              f"available (their sub-timers are not meaningful)", file=sys.stderr)
        return 1

    totals = defaultdict(float)
    for rec in records:
        for _, key, _ in PHASES:
            totals[key] += rec.get(key, 0.0)
        totals["total_ms"] += rec["total_ms"]

    n = len(records)
    grand = totals["total_ms"]
    submits = {int(r["submit"]) for r in records if "submit" in r}
    programs = {int(r["code"]) for r in records if "code" in r}

    # Self-check: every phase of a successful dispatch is a forward interval, so a negative total
    # means the records are not what this tool assumes. Say so loudly rather than printing a table
    # that looks authoritative -- a decomposition with an impossible row in it is worse than none.
    negative = [k for (_, k, _) in PHASES if totals[k] < -1e-6]
    if negative:
        print(f"WARNING: negative time in {', '.join(negative)} on ok=1 records. "
              f"The phase boundaries are not all being set; this table is NOT trustworthy.",
              file=sys.stderr)

    if args.csv:
        print("phase,key,ms,pct,ms_per_dispatch")
        for label, key, _ in PHASES:
            print(f"{label.strip()},{key},{totals[key]:.3f},"
                  f"{100.0 * totals[key] / grand if grand else 0:.2f},{totals[key] / n:.4f}")
        return 0

    print(f"compute decomposition: {n} succeeded dispatches, {len(submits)} submits, "
          f"{len(programs)} programs, {grand:.0f} ms total")
    print(f"  mean {grand / n:.3f} ms/dispatch"
          + (f", {n / len(submits):.1f} dispatches/submit" if submits else ""))
    if failed:
        failed_ms = sum(r["total_ms"] for r in failed)
        share = 100.0 * len(failed) / (len(failed) + n)
        print(f"  EXCLUDED: {len(failed)} FAILED dispatches ({share:.0f}% of all), "
              f"{failed_ms:.0f} ms wall. Their phase split is not recoverable -- a failed dispatch "
              f"leaves execute_item early, so its dispatch_ms prints negative and cleanup_ms "
              f"absorbs the whole record. Only the totals above are decomposed.")
        by_program = defaultdict(int)
        for r in failed:
            by_program[int(r.get("code", 0))] += 1
        worst = sorted(by_program.items(), key=lambda kv: -kv[1])[:5]
        print("  worst failing programs: "
              + ", ".join(f"0x{c:x} ({k})" for c, k in worst))
    print()
    print(f"  {'phase':<20}{'ms':>12}{'% total':>10}{'ms/disp':>12}")
    print("  " + "-" * 54)

    # Walk the tree so each parent is followed by its children and then by the part of its interval
    # the children do not explain. An unattributed remainder is a real measurement, not noise: it is
    # time inside the parent interval that no sub-timer covers, and it is exactly where an unmeasured
    # cost would sit -- so it is printed rather than silently dropped.
    children = defaultdict(list)
    label_of = {}
    for label, key, parent in PHASES:
        children[parent].append(key)
        label_of[key] = label

    def row(label, ms, depth):
        pct = 100.0 * ms / grand if grand else 0.0
        print(f"  {'  ' * depth + label:<20}{ms:>12.1f}{pct:>9.1f}%{ms / n:>12.3f}")

    def walk(key, depth):
        for child in children.get(key, []):
            row(label_of[child].strip(), totals[child], depth)
            walk(child, depth + 1)
        kids = children.get(key)
        if kids:
            rest = totals[key] - sum(totals[k] for k in kids)
            if grand and abs(rest) / grand > 0.005:
                row("unattributed", rest, depth)

    walk("total_ms", 0)

    print("  " + "-" * 54)
    print(f"  {'total':<20}{grand:>12.1f}{100.0:>9.1f}%{grand / n:>12.3f}")

    # The image-binding loop sits inside setup_ms and has no `[compute-phase]` sub-timer, so on an
    # image-heavy title it IS setup's unattributed row. Break it out when the run carried
    # PROSPER_COMPUTE_IMAGE_TIMING; say so explicitly when it did not, because a missing section
    # here reads far too easily as "there was nothing there".
    print()
    if images:
        image_total = sum(i["ms"] for i in images)
        # `[compute-image]` carries no ok flag, so these records span FAILED dispatches as well --
        # a dispatch that fails has usually already bound its images. Denominating them against the
        # succeeded-only total would therefore overstate their share (and can exceed 100%). Use the
        # wall time of every dispatch, and label it, so the two sections are not silently on
        # different bases.
        base = grand + sum(r["total_ms"] for r in failed)
        pct = lambda ms: (100.0 * ms / base) if base else 0.0
        print(f"  setup image bindings: {len(images)} bindings, {image_total:.0f} ms "
              f"({pct(image_total):.1f}% of ALL dispatch wall, succeeded + failed)")
        print(f"  {'sub-phase':<26}{'ms':>12}{'% all':>10}{'ms/binding':>13}")
        print("  " + "-" * 61)
        attributed = 0.0
        for label, key in IMAGE_PHASES:
            ms = sum(i.get(key, 0.0) for i in images)
            attributed += ms
            print(f"  {label:<26}{ms:>12.1f}{pct(ms):>9.1f}%{ms / len(images):>13.3f}")
        rest = image_total - attributed
        print(f"  {'unattributed':<26}{rest:>12.1f}{pct(rest):>9.1f}%{rest / len(images):>13.3f}")
        print("  " + "-" * 61)
        guest_bytes = sum(i.get("guest", 0.0) for i in images)
        staging_bytes = sum(i.get("staging", 0.0) for i in images)
        imported = sum(1 for i in images if i.get("imported", 0.0))
        print(f"  guest bytes bound {guest_bytes / (1 << 30):.2f} GiB, "
              f"staged {staging_bytes / (1 << 30):.2f} GiB, "
              f"{imported} bindings imported from the renderer")
    else:
        print("  setup image bindings: no [compute-image] records in this log.")
        print("  Re-run with PROSPER_COMPUTE_IMAGE_TIMING=1 to decompose setup's unattributed row;")
        print("  its absence here does NOT mean image binding was free.")

    # Which program to attack, and which leaf inside it. Ranking programs by *total* cost (not mean)
    # is deliberate: a cheap kernel dispatched thousands of times outranks an expensive rare one.
    per_program = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(int)
    for rec in records:
        code = int(rec.get("code", 0))
        counts[code] += 1
        for _, key, _ in PHASES:
            per_program[code][key] += rec.get(key, 0.0)
        per_program[code]["total_ms"] += rec["total_ms"]

    ranked = sorted(per_program.items(), key=lambda kv: -kv[1]["total_ms"])[:args.top]
    print()
    print(f"  top {len(ranked)} programs by total cost")
    print(f"  {'code':<16}{'n':>8}{'ms':>12}{'% total':>9}{'mean_ms':>10}  dominant leaf")
    print("  " + "-" * 76)
    for code, phase in ranked:
        dominant = max(LEAVES, key=lambda k: phase[k])
        # Use the timer key, not the display label: "buffers" appears under both setup and
        # writeback, and naming the wrong one sends the reader to the wrong code path.
        label = dominant[:-3]
        print(f"  0x{code:<14x}{counts[code]:>8}{phase['total_ms']:>12.1f}"
              f"{100.0 * phase['total_ms'] / grand if grand else 0:>8.1f}%"
              f"{phase['total_ms'] / counts[code]:>10.3f}  {label} "
              f"({100.0 * phase[dominant] / phase['total_ms'] if phase['total_ms'] else 0:.0f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
