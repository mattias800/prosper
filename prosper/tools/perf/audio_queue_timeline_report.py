#!/usr/bin/env python3
"""Report SDL input queue occupancy from PROSPER_AUDIO_QUEUE_TIMELINE logs.

Usage: audio_queue_timeline_report.py LOG [--min-episode-us N] [--active-window-ms N]

A zero queue means no unconverted input bytes remain. Converted audio can still be
playing downstream, even when the zero is bracketed by positive queue samples.
These episodes are INPUT QUEUE GAPS, not measured consumption shortfalls or hardware
XRUNs. Use PROSPER_AUDIO_DEMAND=1 with audio_delivery_report.py for SDL consumption.

The historical activity filter retains episodes bracketed by positive samples,
shorter than --active-window-ms, and not adjacent to a sampling discontinuity.
It is a heuristic queue filter, not proof of activity or idleness. All excluded
samples remain visible. Durations estimate one median sampling interval per sample;
zero between samples is unobserved. Arrival records supply cadence context only:
their relative gap clock cannot be aligned to the sampler's absolute t_us clock.
"""
import re
import sys

LINE = re.compile(
    r"\[audio-queue\]\s+t_us=(\d+)\s+port=(\d+)\s+queued=(\d+)\s+grain=(\d+)")
# The arrival emitter carries no absolute timestamp -- only the gap since the previous arrival on
# that port -- so arrival times are RECONSTRUCTED by accumulating gaps per port. That is why the
# activity gate is expressed as a maximum inter-arrival gap rather than as an absolute window: the
# gap is the field that is actually measured, and using it directly avoids pinning a reconstructed
# clock to the sampler's independent one.
DBG = re.compile(
    r"\[audio-dbg\]\s+port=(\d+)\s+gap=([0-9.]+)ms")


def parse(text):
    """-> ({port: [(t_us, queued, grain), ...]}, {port: [arrival_gap_us, ...]}) in file order."""
    ports, gaps = {}, {}
    for line in text.splitlines():
        m = LINE.search(line)
        if m:
            t_us, port, queued, grain = (int(g) for g in m.groups())
            ports.setdefault(port, []).append((t_us, queued, grain))
            continue
        d = DBG.search(line)
        if d:
            gaps.setdefault(int(d.group(1)), []).append(int(round(float(d.group(2)) * 1000)))
    return ports, gaps


def arrival_context(gaps):
    """CONTEXT ONLY -> {port: (count, median_gap_us)} from the [audio-dbg] emitter.

    Not used to gate anything; see the module docstring for why the two clocks cannot be aligned.
    """
    out = {}
    for port, gs in gaps.items():
        live = [g for g in gs if g > 0]      # the first arrival's gap is reported as 0
        out[port] = (len(gs), median(live))
    return out


def classify_dry_episodes(samples, dry_eps, interval_us, max_dip_us):
    """Return (bracketed input-queue episodes, excluded sample count).

    Bracketing, duration and continuity are heuristic filters. None establish a
    consumer shortage or prove that an excluded interval was intentional silence.
    """
    if not dry_eps:
        return [], 0
    first_audio = next((t_us for t_us, q, _ in samples if q > 0), None)
    last_audio = next((t_us for t_us, q, _ in reversed(samples) if q > 0), None)
    under, idle = [], 0
    for ep in dry_eps:
        start_us, end_us, n, abuts_gap = ep
        dur_us = n * interval_us
        fed_before = first_audio is not None and first_audio < start_us
        fed_after = last_audio is not None and last_audio > end_us
        if fed_before and fed_after and dur_us <= max_dip_us and not abuts_gap:
            under.append(ep)
        else:
            idle += n
    return under, idle


def episodes(samples, is_dry, max_gap_us=None):
    """Maximal runs of consecutive dry samples -> [(start_us, end_us, n_samples, abuts_gap)].

    `end_us` is the timestamp of the LAST dry sample, so a single-sample episode has zero
    span. Reporting the span as (end - start) would call every one-sample episode 0 us and
    round the shortest real underruns away, so the caller is given the sample count too and
    the duration is charged one sampling interval per sample.

    `max_gap_us` splits a run at a DISCONTINUITY in the series, and without it this function
    welds together two dry stretches that are not adjacent in time. The sampler emits nothing
    for a closed port, and a port is normally dry on both sides of its own lifecycle boundary --
    so a close/reopen produced one bracketed "underrun" spanning the gap, with the shutdown and
    start-up idleness that arms 6b/6c exist to catch merged into it. Constructed and measured:
    fed -> 20 dry -> 30 s of no samples -> 20 dry -> fed reported ONE 40 ms underrun and zero
    idle samples. Lost samples do the same thing more quietly. Found in review of #3070.

    An episode that abuts such a gap is flagged, because continuity of feeding cannot be
    established across an interval nobody observed -- see classify_dry_episodes.
    """
    out = []
    run = None
    prev_t = None
    for t_us, queued, grain in samples:
        broke = (max_gap_us is not None and prev_t is not None
                 and t_us - prev_t > max_gap_us)
        if broke and run is not None:
            run[3] = True                     # the run ends at a discontinuity
            out.append(tuple(run))
            run = None
        if is_dry(queued, grain):
            if run is None:
                run = [t_us, t_us, 0, bool(broke)]   # ...and may START at one
            run[1] = t_us
            run[2] += 1
        elif run is not None:
            out.append(tuple(run))
            run = None
        prev_t = t_us
    if run is not None:
        out.append(tuple(run))
    return out


def median(xs):
    s = sorted(xs)
    return s[len(s) // 2] if s else 0


def interval_us(samples):
    """The sampler's actual measured interval: the median inter-sample delta on one port.

    Measured rather than assumed. A run whose sampler was itself quantized (the #3013 trap the
    sampler's own comment records) would otherwise have its episode durations scaled by a
    nominal interval it never achieved.
    """
    if len(samples) < 2:
        return 0
    deltas = [samples[i + 1][0] - samples[i][0] for i in range(len(samples) - 1)]
    return median(deltas)


def report(ports, gaps=None, min_episode_us=0, active_window_us=200000):
    if not ports:
        print("no [audio-queue] samples found -- was PROSPER_AUDIO_QUEUE_TIMELINE set?")
        return 1
    ctx = arrival_context(gaps or {})
    for port in sorted(ports):
        samples = ports[port]
        qs = [q for _, q, _ in samples]
        grain = median([g for _, _, g in samples])
        iv = interval_us(samples)
        span_s = (samples[-1][0] - samples[0][0]) / 1e6 if len(samples) > 1 else 0.0

        # COVERAGE, printed because the alternative is a silent one. Samples x interval against
        # the wall-clock span says how much of the window was actually observed; a port that
        # closed and reopened, or a run that lost output, shows up here as a shortfall instead
        # of as a mysteriously long episode. Well over 100% means the sampler ran faster than
        # its own median, which is ordinary jitter.
        # (n - 1) intervals span n samples, not n. With n the figure is off by one interval,
        # which is invisible on a long run and reads 200% on a two-sample port. Review of #3070.
        coverage = ((len(samples) - 1) * iv / (samples[-1][0] - samples[0][0]) * 100.0
                    if len(samples) > 1 and samples[-1][0] > samples[0][0] else 0.0)
        print(f"port {port}: {len(samples)} samples over {span_s:.1f} s"
              f"  sampling interval {iv} us (median)"
              f"  coverage {coverage:.0f}%")
        if port in ctx:
            n, g = ctx[port]
            print(f"  [audio-dbg] context: {n} arrivals, median gap {g / 1000.0:.2f} ms"
                  f"  (context only -- not used to gate anything)")
        if grain:
            print(f"  grain {grain} B   queue min {min(qs)} B  median {median(qs)} B"
                  f" = {median(qs) / grain:.2f} grains  max {max(qs)} B")
        else:
            print(f"  grain UNKNOWN (0)   queue min {min(qs)} B  median {median(qs)} B")

        # 4x the measured interval: a couple of missed samples is jitter, a larger hole is a
        # discontinuity -- a closed port, or lost output. Expressed in units of the MEASURED
        # interval so it scales with whatever the sampler actually achieved.
        gap_us = iv * 4 if iv else None
        all_dry = episodes(samples, lambda q, g: q == 0, gap_us)
        dry, n_idle = classify_dry_episodes(samples, all_dry, iv, active_window_us)
        thin = episodes(samples, lambda q, g: g and q < g, gap_us)

        def total_us(eps):
            return sum(e[2] * iv for e in eps)

        kept = [e for e in dry if e[2] * iv >= min_episode_us]
        print(f"  EXCLUDED (outside queue-gap filter): {n_idle} samples,"
              f" {n_idle * iv / 1000.0:.1f} ms"
              f", {100.0 * n_idle / len(samples):.2f}% of samples"
              f"  [excluded below; max dip {active_window_us // 1000} ms]")
        print(f"  INPUT QUEUE GAP (bracketed): {len(kept)} episodes"
              + (f" (of {len(dry)}; {len(dry) - len(kept)} shorter than"
                 f" {min_episode_us} us)" if min_episode_us else "")
              + f", {total_us(kept) / 1000:.1f} ms total"
              f", {100.0 * sum(e[2] for e in kept) / len(samples):.2f}% of samples")
        if kept:
            worst = max(kept, key=lambda e: e[2])
            print(f"    longest {worst[2] * iv} us at t={worst[0]} us")
        print(f"  THIN (under one grain, ungated): {len(thin)} episodes,"
              f" {total_us(thin) / 1000:.1f} ms total,"
              f" {100.0 * sum(e[2] for e in thin) / len(samples):.2f}% of samples")

        print(f"  verdict: {len(kept)} retained input queue gaps; "
              "SDL consumption shortfalls and hardware XRUNs are unknown")
    return 0


def main(argv):
    """Parse arguments strictly. Two silent defects this replaces:

    the usage line documented `--min-episode-us N` (a SPACE) while only the `=N` form was read, so
    the documented spelling was accepted and ignored; and an unrecognised `--flag` was dropped
    without a word. The emitter this reads from states the opposite convention for its own
    variables -- a malformed value disables the trigger rather than firing at an unintended
    moment -- so a typo here must cost the run, never the measurement. Review of #3070.
    """
    paths, min_ep, window_us = [], 0, 200000
    it = iter(argv[1:])
    for a in it:
        if not a.startswith("--"):
            paths.append(a)
            continue
        name, eq, val = a.partition("=")
        if name not in ("--min-episode-us", "--active-window-ms"):
            print(f"error: unknown option {name!r}", file=sys.stderr)
            return 2
        if not eq:                       # the space-separated form the usage line documents
            val = next(it, "")
        try:
            n = int(val)
        except ValueError:
            print(f"error: {name} needs an integer, got {val!r}", file=sys.stderr)
            return 2
        if n < 0:
            print(f"error: {name} cannot be negative", file=sys.stderr)
            return 2
        if name == "--active-window-ms" and n == 0:
            # 0 makes every dry episode longer than the bound, i.e. it disables underrun
            # detection while still printing a report. That is the one value whose effect a
            # reader would not predict from the name, so it is refused rather than echoed.
            print("error: --active-window-ms 0 would classify every dry stretch as idle, "
                  "disabling queue-gap detection entirely", file=sys.stderr)
            return 2
        if name == "--min-episode-us":
            min_ep = n
        else:
            window_us = n * 1000
    if not paths:
        print(__doc__)
        return 2
    with open(paths[0], "r", encoding="utf-8", errors="replace") as fh:
        ports, gaps = parse(fh.read())
    return report(ports, gaps, min_ep, window_us)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
