#!/usr/bin/env python3
"""Read a PROSPER_AUDIO_QUEUE_TIMELINE log and report how close each audio port came to dry.

The metric this exists for is the ZERO-CROSSING EPISODE, not the mean and not the sample
percentage. A device queue that empties for 3 ms and refills is an audible underrun, and it is
invisible to both of the obvious summaries: the one-second delivery average stays at ~100% of real
time (that is exactly how #3016 hid), and a percentage-of-samples figure buries a burst of
contiguous dry samples in a large denominator. So an episode -- a maximal run of consecutive
samples at zero on one port -- is counted once, and its duration is reported.

  usage: audio_queue_timeline_report.py <log> [--min-episode-us N] [--active-window-ms N]
         both `--opt N` and `--opt=N` are accepted; an unknown option is an error

Input lines come from the sampler in audio_sink_sdl3.cpp:

  [audio-queue] t_us=123456 port=17 queued=2048 grain=2048

`queued` and `grain` are both GUEST-format bytes (the sampler uses SDL_GetAudioStreamQueued, not
Available, for exactly that reason), so grain-relative thresholds here need no format flags.

## An empty queue is not an underrun, and the timeline alone cannot tell the difference

A port that is OPEN but not being fed reads `queued=0` forever, and that is silence, not
starvation. A discriminator that cannot tell the two apart would decide a pacer A/B on whichever
arm happened to idle more.

Worth recording precisely, because the first run analysed here is a case where the confound was
suspected and turned out NOT to be the explanation. That run reported a median queue depth of
0.00 grains and 53% of samples dry on Blasphemous 2, which looked like exactly this artefact. It
was not: with the gate in place only **6 samples** out of 588,705 reclassify as idle, so the 53%
is real underrunning. (The other half of that first reading was mine, not the tool's -- I read the
longest episode's `40248 us` as 40 seconds. It is 40 ms, which is an underrun, not silence.)

So the gate exists on principle rather than on that evidence, and the evidence it was built to
explain away survived it. Both halves are stated because a plausible instrument artefact is a very
comfortable explanation for an inconvenient measurement, and this one would have retired a real
defect.

So a dry sample counts as an underrun only when the port was actively streaming across it. The
gate is taken from the sampler's OWN series and needs no second source:

    a dry EPISODE is an UNDERRUN if audio was seen anywhere before it, audio is seen anywhere
    after it, it is no longer than --active-window-ms, and it does not abut a gap in the
    sample series. Otherwise it is not counted.

The first two conditions are GLOBAL, not a local window. An earlier version of this line said
"within --active-window-ms on BOTH sides", which is not what the code does and disagrees with it
on real input -- the window is a maximum DIP DURATION. Found in review of #3070, and worth the
space because a docstring that describes a rule the code does not implement is the most citable
kind of wrong.

Deliberately not cross-correlated with the per-arrival `[audio-dbg]` emitter, which would be the
obvious richer gate. That emitter logs only the GAP since the previous arrival, so arrival times
have to be reconstructed by accumulation and their origin is the first arrival, while the
sampler's `t_us` origin is sink construction. The two clocks cannot be aligned from the log, and a
gate resting on a mis-aligned clock would be exactly the kind of instrument this tool exists to
avoid. When [audio-dbg] lines are present they are used only for CONTEXT -- arrival count and
median cadence -- and are labelled as such.

Both figures are always printed. A tool that silently drops samples reports an improvement that is
really a filter, so IDLE time is shown next to underrun time rather than subtracted out of sight.
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
    """Split dry EPISODES into underruns and idleness, using only this port's own series.

    A dry episode is an UNDERRUN when all four hold:

      * audio was seen BEFORE it   -- otherwise the port had not started feeding yet;
      * audio is seen AFTER it     -- otherwise the port stopped, and the tail is silence;
      * it is no longer than max_dip_us -- otherwise it is silence between two fed stretches,
        not a dip in a stream. This is the condition that matters, and the reason the first two
        are not sufficient: a 5000 ms gap between two songs IS bracketed by audio on both sides,
        so bracketing alone would call it one enormous underrun;
      * it does not abut a DISCONTINUITY in the sample series. The bracketing samples are then
        on the far side of an interval nobody observed, so "the port was being fed across this"
        is exactly what the data cannot say. Counting it would attribute unobserved time to a
        defect -- and the common case, a port closing and reopening, is idleness.

    Per EPISODE rather than per sample, deliberately. A per-sample version of the same rule
    silently turns into a pure duration cap -- the samples at the edges of a long silence fail
    the far-side test and get excluded too -- so the bracketing half of the rule stops meaning
    anything while still appearing in the code. Measured while writing the tests for it.

    Returns (underrun_episodes, idle_sample_count).
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
        coverage = (len(samples) * iv / (samples[-1][0] - samples[0][0]) * 100.0
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
        print(f"  NOT COUNTED (idle, or unobservable): {n_idle} samples,"
              f" {n_idle * iv / 1000.0:.1f} ms"
              f", {100.0 * n_idle / len(samples):.2f}% of samples"
              f"  [excluded below; max dip {active_window_us // 1000} ms]")
        print(f"  UNDERRUN (dry while streaming): {len(kept)} episodes"
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

        # The verdict names the UNDERRUN episode count, because that is the quantity an A/B between
        # two pacers can actually separate; a mean cannot, and this tool exists because of that.
        if not kept:
            # Two different facts, and the old wording conflated them: "no underrun found" is
            # not "the queue never emptied" when dry stretches were excluded by the gate.
            print("  verdict: the queue never emptied" if not n_idle else
                  "  verdict: no underrun -- every dry stretch was idle, over the dip bound,\n"
                  "           or across a gap in sampling (see NOT COUNTED above)")
        elif len(kept) < 5:
            print(f"  verdict: {len(kept)} isolated underruns")
        else:
            print(f"  verdict: STARVED -- {len(kept)} underruns")
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
