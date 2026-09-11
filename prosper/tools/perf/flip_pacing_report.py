#!/usr/bin/env python3
"""Aggregate `[ev] GpuFlip t=<seconds>` lines into a guest frame-pacing report.

WHY THIS EXISTS
---------------
Guest frame-rate hunts on Windows dev boxes have been fps-counter-driven: an average says
"32 fps" but not WHY. The flip timestamps (PROSPER_EVLOG=1) carry the guest's production
cadence directly, and the INTERVAL DISTRIBUTION names the limiter class:

  * a tight spike at one period -- a hard pacer (a vsync, a fixed timer);
  * clustering at timer-tick multiples (~15.6/31.25 ms on Windows) WELL ABOVE the chance
    baseline -- a wait in the production path resolving on the OS tick, regardless of the
    requested timeout. Read the excess over chance, never the raw share: the acceptance
    band covers 30% of the real line, so intervals with no relationship to the tick score
    30% and a reported 50% is 1.7x chance, not 'half the frames are tick-bound' (#3454);
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

# WHERE A BACKWARDS STEP STOPS BEING EMISSION JITTER AND STARTS BEING A RUN BOUNDARY.
# The detector below counts timestamps that step backwards in file order. Two very different
# facts produce one, and the report used to print the same sentence -- "this input holds more
# than one run" -- for both (#3462):
#
#   * a RESTART. `evlog_seconds()` (hle_graphics.cpp) is a function-local static initialised on
#     its own first call, and the only callers are the two flip emitters, so every run's first
#     flip line is t~=0.000000 and a second run in the same stream steps back by roughly the
#     FIRST run's whole elapsed time -- seconds to minutes for anything worth pacing.
#   * a RACE. `evlog_seconds()` is evaluated as an argument and the stream lock is taken
#     afterwards, so two emitting threads can interleave. That window is the gap between reading
#     the clock and taking the lock inside one fprintf: microseconds, and at most one scheduler
#     quantum if the emitting thread is preempted inside it.
#
# TICK_MS is a generous ceiling for the second (a preemption cannot resolve faster than the
# timer tick it waits on) and one second is far below the first. BETWEEN them this tool says it
# cannot tell -- because it cannot, and a single threshold would only move the boundary and go
# on printing one of the two sentences for a fact that is neither.
RACE_MS = TICK_MS
RESTART_MS = 1000.0

# Half-width of the acceptance band around each tick multiple, as a fraction of the tick.
# It was a bare 0.15 inside classify() while the report's own sentence said '15%' in a
# separate literal -- two places to change and one of them silent. Named here so the
# predicate, the printed band and the chance baseline below cannot drift apart.
BAND = 0.15

# THE NULL. The band spans +/-BAND of every tick, i.e. 2*BAND of each tick period, so an
# interval bearing no relationship to the tick still lands inside it 2*BAND of the time --
# 30.0% at BAND=0.15. Without this printed next to the share, the figure is uninterpretable:
# one GTA V run (#3450) produced windows at 96.5%, 60.5%, 50.1% and 9.3%, and only the first
# is a strong finding while the LAST is evidence AGAINST tick-bound production -- which read
# as 'no finding' rather than as the signal it is (#3454).
CHANCE = min(1.0, 2.0 * BAND)


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
        # Keep the MAGNITUDES, not just how many. A sub-microsecond race and a 100-second run
        # boundary are different facts, and a count cannot tell them apart (#3462).
        backsteps = [(a - b) * 1000.0 for a, b in zip(stamps, stamps[1:]) if b < a]
        stamps.sort()
        timelines.append((path, stamps, untimed, backsteps))
    return timelines


def classify(ms, tick, band=BAND):
    """Name the bucket an interval falls into, for the quantization verdict."""
    nearest = max(1, round(ms / tick))
    error = abs(ms - nearest * tick)
    if error <= band * tick:
        return f"{nearest} tick(s)"
    return "between ticks"


def alignment_power(intervals, tick, band=BAND):
    """The fraction of intervals for which "aligned" is arithmetically REACHABLE.

    classify() computes nearest = max(1, round(ms / tick)), so an interval shorter than the
    bottom edge of the k=1 band -- (1 - band) * tick, 13.28 ms at the defaults -- is measured
    against ONE tick however small it is, and its error can never fall inside the band. Such an
    interval scores 'between ticks' by construction, not by evidence.

    That matters because the sub-chance verdict below reads as a finding. Without this term a
    run at 156 fps (6.4 ms intervals) scores 0.0% and gets told it is ANTI-aligned -- a
    confident negative on a population where no other answer was arithmetically possible, which
    is a new instrument lie in the fix for an old one. Caught in review of #3454; the GTA V run
    that motivated the issue is unaffected (its windows mean 24-134 ms, all above the floor).
    """
    if not intervals:
        return 0.0
    floor = (1.0 - band) * tick
    return sum(1 for ms in intervals if ms >= floor) / len(intervals)


def alignment_verdict(share, chance=None, power=1.0):
    """Turn an observed tick-aligned SHARE (0..1) into a ratio against chance and a verdict.

    Separate from the printing so the thresholds can be asserted directly.

    THE THRESHOLDS BELOW ARE A JUDGEMENT CALL, NOT A DERIVED RESULT -- 0.75 / 1.25 / 2.0 are
    round numbers chosen to be coarse, because a share against a flat null does not support a
    sharper claim. They are stated here rather than left to look principled. A real test would
    need the sample size too: distinguishing 1.25x from chance at n intervals needs roughly
    n > 150 before the binomial spread is narrower than the gap. Reported as a ratio precisely
    so a reader can apply their own bar.

    `power` is the fraction of the population for which alignment is REACHABLE at all (see
    alignment_power). Below half, no verdict about the tick is honest and none is given.
    """
    chance = CHANCE if chance is None else chance
    if chance <= 0:
        # Unreachable from report() -- CHANCE is a positive constant -- but exercised directly by
        # the self-test, and the guard is what stops a future caller dividing by a widened band.
        return None, 'no chance baseline is defined for this band'
    if power < 0.5:
        return share / chance, (
            'NO POWER -- %.0f%% of intervals are shorter than the band around one tick, where '
            'alignment is arithmetically impossible, so this share is a property of the frame '
            'rate rather than evidence about the tick' % (100.0 * (1.0 - power)))
    # THE NULL SCALES WITH POWER, and the first version of this gate missed that. Only `power`
    # of the population can score aligned at all; the rest contribute a structural zero. So the
    # expected share for a tick-unrelated run is power * CHANCE, not CHANCE. Measuring a mixed-
    # rate run against the full 30% understates its ratio by 1/power -- up to 2x at the gate --
    # and can print 'BELOW chance / evidence AGAINST' on a share that is in fact above its own
    # null. That is the same fabricated negative the power term was added to remove, surviving
    # in the partial case; and a mixed-rate run is precisely the phase change the windowed view
    # exists for, so it is the common case rather than a corner. Found in re-review of #3454.
    ratio = share / (chance * power)
    if ratio < 0.75:
        return ratio, ('BELOW chance -- production here is anti-aligned, which is evidence AGAINST a tick-bound wait, not an absence of evidence')
    if ratio < 1.25:
        return ratio, 'at chance -- no evidence of tick quantisation'
    if ratio < 2.0:
        return ratio, 'modestly above chance -- weak, do not call this tick-bound on its own'
    return ratio, 'well above chance -- real quantisation'


def describe_backsteps(backsteps):
    """Name what a set of backwards timestamp steps IS, from the largest one's magnitude.

    Separate from the printing so the classification can be asserted directly, and returned as
    (kind, sentence) so a caller can branch on the kind rather than on the prose.

    The detector is near-total for its intended case -- concatenating any two real logs always
    produces a backwards step at the seam, because every run's first flip is t~=0 -- so the
    question was never whether it fires but what it means when it does. See RACE_MS/RESTART_MS
    for where the two thresholds come from and why the middle band is left unnamed (#3462).
    """
    if not backsteps:
        return None, ""
    worst = max(backsteps)
    count = len(backsteps)
    if worst >= RESTART_MS:
        return "restart", (
            f"WARNING: the timestamps step backwards {count} time(s), the largest by"
            f" {worst / 1000.0:.3f} s, so this input holds more than one run (the epoch is per"
            f" process, so a second run restarts near zero). Intervals across a restart are not"
            f" real. Pass each run as its own argument instead of concatenating them.")
    if worst <= RACE_MS:
        return "race", (
            f"WARNING: the timestamps step backwards {count} time(s), the largest by only"
            f" {worst:.3f} ms. That is FAR too small to be a run boundary -- a second run steps"
            f" back by the first run's whole duration -- so it is out-of-order EMISSION: the"
            f" clock is read before the stream lock is taken, so two emitting threads can"
            f" interleave. The input is one run; {count} interval(s) below are not real. The"
            f" remedy is the emitter's ordering, not splitting the input.")
    return "unclear", (
        f"WARNING: the timestamps step backwards {count} time(s), the largest by {worst:.1f} ms."
        f" That is too large for emission jitter (at most one scheduler quantum, ~{RACE_MS:.3f} ms)"
        f" and too small for a run boundary (a whole run's duration, at least"
        f" {RESTART_MS / 1000.0:.0f} s), so this tool cannot say which it is. Read the log around"
        f" the step before trusting any interval that crosses it.")


def report(stamps, window_s, tick, untimed=0, backsteps=()):
    # Report the unreadable population WHENEVER it exists, not only when nothing parsed. Two
    # readable flips among thousands of unreadable ones produced a confident distribution over
    # 0.06% of the data and said nothing about the rest -- #3452's own failure, one branch lower.
    _, sentence = describe_backsteps(backsteps)
    if sentence:
        print(sentence)
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
    overall_share = quantized / len(intervals)
    power = alignment_power(intervals, tick)
    ratio, verdict = alignment_verdict(overall_share, power=power)
    print(f"tick-aligned intervals (within +/-{BAND * 100:.0f}% of a {tick:.3f} ms tick "
          f"multiple): {quantized} ({100 * overall_share:.1f}%)")
    # Name the baseline actually used. Printing 30.0% beside a ratio computed against a
    # power-scaled null would be a third way for this line to mislead.
    effective = CHANCE * power if power >= 0.5 else CHANCE
    # Below the gate the baseline is deliberately NOT scaled, so the parenthetical must not claim
    # it was -- the numbers were consistent, only the words were wrong, which is this line's own
    # failure mode one more time (re-review of #3454).
    scaled = ("" if power >= 0.999 or power < 0.5
              else f" (scaled by {100 * power:.0f}% reachable)")
    print(f"  chance baseline {100 * effective:.1f}%{scaled} -- {ratio:.2f}x chance: {verdict}")

    hist = collections.Counter(round(ms) for ms in intervals)
    print("interval histogram (ms: count, top buckets):")
    for bucket, count in sorted(hist.items()):
        if count >= max(2, len(intervals) // 200):
            print(f"  {bucket:4d} ms x{count}")

    # Windowed view: a phase change (cinematic -> gameplay) must not smear two regimes.
    window_flips = max(1, int(window_s * (1000.0 / statistics.mean(intervals))))
    print(f"windows of ~{window_flips} flips:")
    start = 0
    window_sizes = []   # (share, interval count) -- the count decides if it may vote below
    while start < len(intervals):
        chunk = intervals[start:start + window_flips]
        mean_ms = statistics.mean(chunk)
        quantized = sum(1 for ms in chunk if classify(ms, tick) != "between ticks")
        share = quantized / len(chunk)
        window_sizes.append((share, len(chunk)))
        w_power = alignment_power(chunk, tick)
        w_ratio, _ = alignment_verdict(share, power=w_power)
        # Mark the sub-chance windows explicitly. A low share reading as 'no finding' is
        # exactly how the 9.3% window's information was thrown away (#3454) -- but only
        # where the window could have scored otherwise, or the mark is automatic.
        note = "no power" if w_power < 0.5 else (
            "BELOW chance" if w_ratio is not None and w_ratio < 0.75 else "")
        print(f"  flips {start:5d}-{start + len(chunk):5d}: "
              f"{1000.0 / mean_ms:6.1f} fps  mean {mean_ms:6.2f} ms  "
              f"tick-aligned {100 * share:5.1f}% ({w_ratio:.2f}x chance) {note}")
        start += window_flips

    # The whole-run share is a mean over windows that can disagree by an order of magnitude,
    # which is the very smearing the windowed view exists to prevent -- so say so rather than
    # letting the summary line quietly reintroduce it (#3454).
    # A trailing window can hold a handful of intervals, whose share is quantised into
    # huge steps -- 3 intervals can only score 0/33/67/100%. Comparing that against a full
    # window manufactures a disagreement, so short windows do not vote (#3454 review).
    full = [share for share, n in window_sizes if n >= max(8, window_flips // 4)]
    if len(full) >= 2:
        lo, hi = min(full), max(full)
        if hi - lo >= 0.25:
            print(f"  NOTE: windows disagree by {100 * lo:.1f}%..{100 * hi:.1f}% tick-aligned,"
                  f" so the whole-run {100 * overall_share:.1f}% above is an average over"
                  f" regimes that differ. Read the windows, not the summary.")

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
    for index, (path, stamps, untimed, backsteps) in enumerate(timelines):
        if len(timelines) > 1:
            if index:
                print("")
            print(f"=== {path} ===")
        report(stamps, args.window_s, args.tick_ms, untimed, backsteps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
