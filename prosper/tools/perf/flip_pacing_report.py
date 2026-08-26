#!/usr/bin/env python3
"""Aggregate `[ev] GpuFlip t=<seconds>` lines into a guest frame-pacing report.

WHY THIS EXISTS
---------------
Guest frame-rate hunts on Windows dev boxes have been fps-counter-driven: an average says
"32 fps" but not WHY. The flip timestamps (PROSPER_EVLOG=1) carry the guest's production
cadence directly, and the INTERVAL DISTRIBUTION names the limiter class:

  * a tight spike at one period -- a hard pacer (a vsync, a fixed timer);
  * clustering at timer-tick multiples (~15.6/31.25 ms on Windows) -- a wait in the
    production path resolving on the OS tick, regardless of the requested timeout;
  * a wide spread -- work-bound production (profile the fold, don't hunt waits).

The report also splits the timeline into windows so a cinematic-to-gameplay phase change
does not smear one phase's numbers over another.

Usage:
    flip_pacing_report.py [LOG ...] [--window-s S] [--tick-ms]
"""

import argparse
import collections
import re
import statistics
import sys

FLIP = re.compile(r"\[ev\] GpuFlip t=([0-9.]+)")
# The Win32 timer tick on dev boxes is 15.625 ms; quantized waits land on its multiples.
TICK_MS = 15.625


def parse_flips(paths):
    stamps = []
    for path in paths:
        if path == "-":
            handle = sys.stdin
            for line in handle:
                m = FLIP.search(line)
                if m:
                    stamps.append(float(m.group(1)))
        else:
            with open(path, encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    m = FLIP.search(line)
                    if m:
                        stamps.append(float(m.group(1)))
    stamps.sort()
    return stamps


def classify(ms, tick):
    """Name the bucket an interval falls into, for the quantization verdict."""
    nearest = max(1, round(ms / tick))
    error = abs(ms - nearest * tick)
    if error <= 0.15 * tick:
        return f"{nearest} tick(s)"
    return "between ticks"


def report(stamps, window_s, tick):
    if len(stamps) < 2:
        print("fewer than 2 flips recorded -- nothing to pace")
        return
    intervals = [ms for ms in ((b - a) * 1000.0 for a, b in zip(stamps, stamps[1:]))
                 if ms > 0.5]
    if not intervals:
        print("no usable intervals")
        return

    overall = 1000.0 / statistics.mean(intervals)
    print(f"flips={len(stamps)} intervals={len(intervals)} "
          f"mean period {statistics.mean(intervals):.2f} ms "
          f"-> {overall:.1f} fps average")

    quantized = sum(1 for ms in intervals if classify(ms, tick) != "between ticks")
    print(f"tick-aligned intervals (within 15% of a {tick:.3f} ms tick multiple): "
          f"{quantized} ({100 * quantized / len(intervals):.1f}%)")

    hist = collections.Counter(round(ms) for ms in intervals)
    print("interval histogram (ms: count, top buckets):")
    for bucket, count in sorted(hist.items()):
        if count >= max(2, len(intervals) // 200):
            print(f"  {bucket:4d} ms x{count}")

    # Windowed view: a phase change (cinematic -> gameplay) must not smear two regimes.
    window_flips = max(1, int(window_s * (1000.0 / statistics.mean(intervals))))
    print(f"windows of ~{window_flips} flips:")
    start = 0
    while start < len(intervals):
        chunk = intervals[start:start + window_flips]
        mean_ms = statistics.mean(chunk)
        quantized = sum(1 for ms in chunk if classify(ms, tick) != "between ticks")
        print(f"  flips {start:5d}-{start + len(chunk):5d}: "
              f"{1000.0 / mean_ms:6.1f} fps  mean {mean_ms:6.2f} ms  "
              f"tick-aligned {100 * quantized / len(chunk):5.1f}%")
        start += window_flips

    slowest = sorted(range(len(intervals)), key=lambda i: -intervals[i])[:5]
    print("slowest intervals (ms):",
          ", ".join(f"{intervals[i]:.1f} @ flip {i}" for i in slowest))


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate [ev] GpuFlip timestamps into a guest frame-pacing report.")
    parser.add_argument("logs", nargs="*", help="run logs (PROSPER_EVLOG=1); '-' = stdin")
    parser.add_argument("--window-s", type=float, default=10.0,
                        help="window length in seconds for the phase split (default 10)")
    parser.add_argument("--tick-ms", type=float, default=TICK_MS,
                        help="the OS timer tick to test quantization against (default 15.625)")
    args = parser.parse_args()

    paths = args.logs if args.logs else ["-"]
    stamps = parse_flips(paths)
    report(stamps, args.window_s, args.tick_ms)
    return 0


if __name__ == "__main__":
    sys.exit(main())
