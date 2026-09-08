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

# Match the timestamp wherever it sits on the line. The emitter also prints handle/bufidx/
# mode/fliparg, and requiring t= to sit adjacent to the tag is what let a format mismatch go
# unnoticed: this tool read zero flips from every real run for as long as it existed, and
# reported that as "nothing to pace" (#3452). `.` excludes newline without DOTALL, so a match
# cannot run past the end of one line.
FLIP = re.compile(r"\[ev\] GpuFlip\b.*?\bt=([0-9.]+)")
# Any line announcing a flip at all. Used only to tell 'this run had no flips' apart from
# 'this tool could not read the flips this run recorded' -- the distinction #3452 turned on.
FLIP_TAG = re.compile(r"\[ev\] GpuFlip\b")
# The Win32 timer tick on dev boxes is 15.625 ms; quantized waits land on its multiples.
TICK_MS = 15.625


def parse_flips(paths):
    """Return one (path, timestamps, untimed) timeline PER SOURCE. Never pooled.

    `untimed` exists so a parse failure cannot be reported as an empty run: a log full of flips
    the parser cannot read, and a log with no flips, are different facts, and this tool printed
    the same sentence for both until #3452.

    Timelines are kept separate because the timestamp epoch is PER PROCESS -- every run starts
    near zero. Pooling and sorting two logs interleaves them into intervals neither run
    contained: a 60 fps and a 30 fps capture merge into a plausible-looking distribution with a
    fabricated sub-tick histogram, which this report's own wording then calls work-bound
    production. Comparing two runs is exactly the A/B this tool is for, so the wrong answer was
    reachable by its intended use.
    """
    timelines = []
    for path in paths:
        stamps = []
        untimed = 0
        opened = None if path == "-" else open(path, encoding="utf-8", errors="replace")
        handle = sys.stdin if opened is None else opened
        try:
            for line in handle:
                m = FLIP.search(line)
                if m:
                    try:
                        stamps.append(float(m.group(1)))
                    except ValueError:
                        # A malformed number is an unreadable flip, not a missing one.
                        untimed += 1
                elif FLIP_TAG.search(line):
                    untimed += 1
        finally:
            if opened is not None:
                opened.close()
        # Detect concatenated runs BEFORE sorting. The per-source split cannot see them --
        # `cat a.log b.log | tool` is one stream -- and sorting is what destroys the evidence:
        # the epoch is per process, so a second run restarts near zero and steps BACKWARDS in
        # file order. Sorted, that is indistinguishable from one slow run, which is the
        # fabricated distribution this tool must never produce.
        restarts = sum(1 for a, b in zip(stamps, stamps[1:]) if b < a)
        stamps.sort()
        timelines.append((path, stamps, untimed, restarts))
    return timelines


def classify(ms, tick):
    """Name the bucket an interval falls into, for the quantization verdict."""
    nearest = max(1, round(ms / tick))
    error = abs(ms - nearest * tick)
    if error <= 0.15 * tick:
        return f"{nearest} tick(s)"
    return "between ticks"


def report(stamps, window_s, tick, untimed=0, restarts=0):
    # Report the unreadable population WHENEVER it exists, not only when nothing parsed. Two
    # readable flips among thousands of unreadable ones produced a confident distribution over
    # 0.06% of the data and said nothing about the rest -- #3452's own failure, one branch lower.
    if restarts:
        print(f"WARNING: the timestamps step backwards {restarts} time(s), so this input holds"
              f" more than one run (the epoch is per process). Intervals across a restart are not"
              f" real. Pass each run as its own argument instead of concatenating them.")
    if untimed and stamps:
        share = 100.0 * untimed / (untimed + len(stamps))
        print(f"WARNING: {untimed} of {untimed + len(stamps)} flip line(s) ({share:.1f}%) carried"
              f" no readable t= timestamp and are NOT in the figures below."
              f" Treat this distribution as describing only the {len(stamps)} that parsed.")
    if len(stamps) < 2:
        if untimed:
            # Name the TOOL as the failure. Saying 'nothing to pace' about a log holding
            # thousands of flip lines is how #3452 stayed invisible for as long as it did:
            # the sentence described the run, and the run was fine.
            print(f'{untimed} flip line(s) found, none with a readable t= timestamp: this log'
                  f' cannot be paced. The emitter and this parser disagree on the line format,'
                  f' or the run predates the timestamp added in #3452.')
            return
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
    timelines = parse_flips(paths)
    # One report per source. Several logs are several runs, each with its own epoch, so they
    # are never merged -- see parse_flips. A header only when there is more than one, so the
    # single-log output a reader already knows is unchanged.
    for index, (path, stamps, untimed, restarts) in enumerate(timelines):
        if len(timelines) > 1:
            if index:
                print("")
            print(f"=== {path} ===")
        report(stamps, args.window_s, args.tick_ms, untimed, restarts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
