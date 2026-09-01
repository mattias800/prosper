#!/usr/bin/env python3
"""Aggregate `[audio-dbg]` lines into an audio-delivery health report.

WHY THIS EXISTS
---------------
Audio underrun hunts on Windows dev boxes have been ear-driven: "it stutters like a small
buffer", then a cycle of pacing changes and listen tests. `PROSPER_AUDIO_DEBUG=1` makes the
SDL sink log every guest grain delivery (arrival gap, frames, queued bytes); this tool turns
that log into the numbers that name the failure mode, offline, without listening:

  * delivery cadence -- mean/median/p99/max inter-arrival gap per port, and the fraction of
    gaps beyond one and two grain periods;
  * queue health -- mean/min queued bytes, and the share of arrivals that found the queue
    below one grain and below a quarter grain -- a THIN CUSHION, which for a pacer that hands
    over one grain at a time is the normal steady state rather than a fault -- plus the share
    that arrived to a COMPLETELY EMPTY queue, which is the starvation case;
  * effective delivery rate vs the device rate -- a persistent deficit is a CLOCK DRIFT
    between the guest's budgeted audio clock and the device crystal; no cushion survives it,
    and the fix is drift compensation, not deeper buffering;
  * burst clustering -- the fraction of arrivals that carry more than one grain of audio
    after a gap of more than two grain periods. This shape is CONSISTENT WITH a quantized
    mixer wake, but the cadence alone cannot prove it: a producer simply delivering audio
    below real time also arrives in clusters, spaced by production rather than by a wait
    (#3080 -- the verdict named the wrong cause on a title where a `PROSPER_TIMEDWAIT_CENSUS=1`
    measurement, in the same log, showed the guest's only timed wait resolving at x1.02, i.e.
    not quantized at all; the -55% delivery-rate deficit already explained the clustering). So
    if `[timedwait Ns]` census lines are present in the same input, their ratio for the
    dominant wait primitive SETTLES which cause applies; if they are absent, the report says so
    and names the census as the discriminator instead of asserting a mechanism it cannot see.

WHAT THIS TOOL DOES NOT SEE
---------------------------
`[audio-dbg]` is emitted per accepted `output()` call from the guest. Grains the guest never
delivers (a mixer that skipped a period) appear only as a longer gap. Content-level defects
(a mixer that produced silence or repeated samples) are invisible here; the queue level and
the delivery rate are most of the report's world -- the one exception is the wait-primitive
census below, read from the SAME log when the run captured both diagnostics together.

Usage:
    audio_delivery_report.py [LOG ...] [--port P] [--device-hz H] [--channels C] [--bytes-per-frame N]
"""

import argparse
import re
import sys
from collections import defaultdict

RECORD = re.compile(
    r"\[audio-dbg\] port=(\d+) gap=([0-9.]+)ms frames=(\d+) queued_before=(\d+)"
    r"(?: \(grain=(\d+)\))?")

# `report_timedwait_census()` (src/hle/kernel/timedwait_census.hpp) prints one of these per
# primitive every 5 s when PROSPER_TIMEDWAIT_CENSUS=1. It is a DIRECT measurement of how coarse
# a wait primitive actually is -- requested vs actual, as a ratio -- which is exactly the
# question the burst-clustering verdict below cannot answer from cadence alone. A primitive with
# no requested interval (an absolute-clock deadline, e.g. cond_timedwait) prints "?" instead of a
# ratio and is skipped: no ratio, no evidence either way.
TIMEDWAIT_RECORD = re.compile(
    r"\[timedwait \S+\]\s+(\S+)\s+calls=(\d+)\s+requested=\s*(?:[0-9.]+\s*ms|\?)"
    r"\s+actual=\s*[0-9.]+\s*ms(?:\s+x([0-9.]+))?")

# A ratio comfortably above 1.0 means the wait genuinely overshot what the guest asked for (the
# "coarse tick" the burst verdict describes); a ratio near 1.0 means the wait resolved close to
# on time and cannot be the cause of clustered arrivals. #3013 measured ~x2.9 for a primitive that
# WAS the cause; #3080 measured x1.02 for one that was not, on a title whose clustering was fully
# explained by a -55% delivery-rate deficit instead. The threshold sits well clear of both.
QUANTIZED_RATIO_THRESHOLD = 1.3

# The device consumes `device_hz * channels * bytes_per_frame` bytes per second. The guest's
# delivery rate against THAT number is the clock-drift measurement; a persistent deficit
# drains any cushion at the deficit rate no matter how deep the buffer is.


def parse_streams(paths):
    for path in paths:
        if path == "-":
            for line in sys.stdin:
                yield path, line
        else:
            with open(path, encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    yield path, line


def analyze(paths, port_filter, device_hz, channels, bytes_per_frame):
    per_port = {}
    # name -> {"calls": int, "ratio_calls_sum": float}; mean ratio = ratio_calls_sum / calls.
    # Weighted by calls (not averaged line-to-line) so a primitive with many more calls in one
    # 5 s window is not diluted by a quiet window reporting the same primitive.
    census = {}
    for path, line in parse_streams(paths):
        m = RECORD.search(line)
        if not m:
            tm = TIMEDWAIT_RECORD.search(line)
            if tm and tm.group(3):
                name = tm.group(1)
                calls = int(tm.group(2))
                ratio = float(tm.group(3))
                if calls > 0:
                    slot = census.setdefault(name, {"calls": 0, "ratio_calls_sum": 0.0})
                    slot["calls"] += calls
                    slot["ratio_calls_sum"] += ratio * calls
            continue
        port = int(m.group(1))
        if port_filter is not None and port != port_filter:
            continue
        gap_ms = float(m.group(2))
        frames = int(m.group(3))
        queued = int(m.group(4))
        # The emitter states the grain size in BYTES on every line. Prefer it over inferring the
        # grain from a frame count: frames[0] takes the FIRST record, so one atypical arrival at
        # the start of a log silently rescales every queue threshold below -- and it fails in the
        # reassuring direction, zeroing the thin-cushion and starvation rows rather than
        # inflating them. Optional in the regex so older logs still parse.
        grain_bytes_logged = int(m.group(5)) if m.group(5) else 0
        slot = per_port.setdefault(port, {
            "gaps": [], "frames": [], "queued": [], "sources": defaultdict(int),
            "grain_bytes": 0,
        })
        if grain_bytes_logged:
            slot["grain_bytes"] = grain_bytes_logged
        slot["gaps"].append(gap_ms)
        slot["frames"].append(frames)
        slot["queued"].append(queued)
    return per_port, census


def dominant_census(census):
    """The most-called wait primitive's mean overshoot ratio, or None if no census was present.

    Global to the process, not per port -- the census counts every timed wait regardless of which
    audio port (if any) it paces -- so this is the best available evidence for a burst-clustering
    verdict on any port, not a per-port measurement. Most-called rather than worst-ratio: on a
    title that uses exactly one primitive (the common case -- see #3080, where usleep was the
    ONLY one that ever appeared), that is the same primitive either way, and on a title using
    several, the one actually driving the pacing is more likely the one called the most.
    """
    if not census:
        return None
    name = max(census, key=lambda k: census[k]["calls"])
    calls = census[name]["calls"]
    if calls <= 0:
        return None
    return {"name": name, "calls": calls, "ratio": census[name]["ratio_calls_sum"] / calls}


def percentile(sorted_values, fraction):
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, int(fraction * len(sorted_values)))
    return sorted_values[index]


def report_port(port, slot, device_hz, channels, bytes_per_frame, census=None):
    gaps = sorted(slot["gaps"])
    queued = sorted(slot["queued"])
    frames_total = sum(slot["frames"])
    calls = len(slot["gaps"])
    span_s = sum(gaps) / 1000.0
    # The emitter logs TWO independent facts per record -- frames=N and (grain=B) -- so each is used
    # for what it is, and neither needs the CLI flags to be right:
    #
    #   * grain_frames, for grain_ms and the gap-beyond-N-grains rows, is the MEDIAN frame count. The
    #     median rather than frames[0] because one atypical first arrival otherwise rescales every
    #     cadence row -- review measured that fabricating "gaps beyond 2 grains: 100.0%" and a false
    #     quantized-mixer verdict.
    #   * grain_bytes, for the queue rows, is the LOGGED byte size.
    #
    # An earlier revision derived grain_frames FROM grain_bytes, which reintroduced the flag
    # dependency it was meant to remove: an s16 title read with the default f32 flags reported
    # grain=128 frames from a correct 1024 B. Two logged facts, used directly, have no such coupling.
    if slot["frames"]:
        ordered = sorted(slot["frames"])
        grain_frames = ordered[len(ordered) // 2]
    else:
        grain_frames = 0
    grain_ms = grain_frames / device_hz * 1000.0 if device_hz else 0.0

    print(f"port {port}: calls={calls} span={span_s:.1f}s grain={grain_frames} frames"
          f" ({grain_ms:.2f} ms)")

    if not gaps:
        print("  no inter-arrival gaps recorded -- nothing to analyze")
        return

    mean_gap = sum(gaps) / len(gaps)
    print(f"  delivery cadence: mean {mean_gap:.2f} ms  median {percentile(gaps, 0.5):.2f} ms"
          f"  p99 {percentile(gaps, 0.99):.2f} ms  max {gaps[-1]:.2f} ms")

    one_grain = [g for g in gaps if g > grain_ms]
    two_grain = [g for g in gaps if g > 2 * grain_ms]
    if grain_ms > 0:
        print(f"  gaps beyond 1 grain ({grain_ms:.2f} ms): {len(one_grain)}"
              f" ({100 * len(one_grain) / len(gaps):.1f}%)")
        print(f"  gaps beyond 2 grains:             {len(two_grain)}"
              f" ({100 * len(two_grain) / len(gaps):.1f}%)")

    if queued:
        mean_q = sum(queued) / len(queued)
        # A GRAIN, not a frame. These thresholds were bytes_per_frame * channels -- one FRAME, 8 bytes
        # for f32 stereo -- so "below 1 grain" tested q < 8 and "below 1/4 grain" tested q < 2. Both
        # were really asking whether the queue was EMPTY, under labels promising something far weaker,
        # which is the direction that gets quoted: a run reported as 1.1% "below a quarter grain" was
        # 1.1% completely dry.
        #
        # The logged byte size needs no flags to be correct. The MEDIAN-frame inference above is the
        # fallback for logs predating the (grain=N) field, and is the only place bytes_per_frame and
        # channels are consulted.
        grain_bytes = slot.get("grain_bytes") or (grain_frames * bytes_per_frame * channels)
        under_one = sum(1 for q in queued if q < grain_bytes) if grain_bytes else 0
        under_quarter = sum(1 for q in queued if q < grain_bytes // 4) if grain_bytes else 0
        empty = sum(1 for q in queued if q == 0)
        print(f"  queue at arrival: mean {mean_q:.0f} B  min {queued[0]} B")
        # NOT labelled starvation. One grain of cushion is the normal steady state for a pacer
        # handing over a grain at a time, so "below one grain" describes most healthy arrivals --
        # the fixture added with this change prints 100% of them beside 0% empty. The old wording,
        # "the device starved between deliveries", was true only while the threshold was
        # accidentally testing queued == 0; widening it 256x to a real grain left the words
        # describing a condition they no longer match. Starvation is the EMPTY row.
        print(f"  arrivals with under 1 grain buffered: {under_one}"
              f" ({100 * under_one / len(queued):.1f}%) -- thin cushion")
        print(f"  arrivals below 1/4 grain:          {under_quarter}"
              f" ({100 * under_quarter / len(queued):.1f}%) -- very thin")
        print(f"  arrivals with an EMPTY queue:         {empty}"
              f" ({100 * empty / len(queued):.1f}%) -- STARVED: nothing left to play")

    if span_s > 0 and device_hz:
        delivered_hz = frames_total / span_s
        device_hz_total = device_hz
        drift_pct = 100.0 * (delivered_hz - device_hz_total) / device_hz_total
        print(f"  effective delivery: {delivered_hz:.0f} frames/s vs device {device_hz_total}"
              f" -> drift {drift_pct:+.2f}%")
        if drift_pct < -0.3:
            print("  VERDICT: the guest's audio clock runs below the device rate -- a clock"
                  " deficit, not a buffering defect. No cushion survives it; fix the guest"
                  " audio clock or compensate the drift in the sink.")
        elif drift_pct > 0.3:
            print("  VERDICT: the guest's audio clock runs above the device rate -- the"
                  " cushion grows until the depth cap; drift compensation should slow the"
                  " delivery.")
        else:
            print("  VERDICT: delivery matches the device rate -- a stutter at this rate is"
                  " a burst/quantization problem, not a clock deficit. See the cadence and"
                  " burst rows above.")

    if grain_ms > 0:
        bursts = sum(1 for g in gaps if g > 2 * grain_ms)
        if bursts > len(gaps) * 0.2 and mean_gap > 1.5 * grain_ms:
            # This cadence shape (clustering beyond two grain periods) is what a quantized mixer
            # wake looks like -- but it is also what a producer simply running below real time
            # looks like, since its arrivals are spaced by production rather than by a wait. #3080:
            # this exact code path asserted the wait-primitive cause on a title where the census
            # measured the opposite (x1.02, i.e. not quantized) and a -55% delivery-rate deficit
            # already explained the clustering. So the verdict is only as strong as the evidence
            # available for THIS run: earned when a census is present, a named hypothesis when not.
            if census is not None and census["ratio"] is not None:
                if census["ratio"] < QUANTIZED_RATIO_THRESHOLD:
                    print("  VERDICT: the inter-arrival cadence clusters beyond two grain periods,"
                          f" but the timedwait census RULES OUT a quantized wake -- {census['name']}"
                          f" resolves at x{census['ratio']:.2f} ({census['calls']} calls), not a"
                          " coarse tick. The clustering has some other cause -- see the delivery-rate"
                          " verdict above if it fired -- and the wait primitive is not it.")
                else:
                    print("  VERDICT: the inter-arrival cadence clusters beyond two grain periods,"
                          f" and the timedwait census CONFIRMS a coarse wait -- {census['name']}"
                          f" resolves at x{census['ratio']:.2f} ({census['calls']} calls) against its"
                          " requested interval. The mixer's wake is quantized, not a buffering"
                          " defect. Fix the wait primitive's resolution.")
            else:
                print("  VERDICT: the inter-arrival cadence clusters beyond two grain periods --"
                      " consistent with EITHER a quantized mixer wake (a timed wait resolving on a"
                      " coarse tick) OR a producer delivering audio below real time (which also"
                      " arrives in clusters). This cadence alone cannot tell them apart. Re-run with"
                      " PROSPER_TIMEDWAIT_CENSUS=1 captured into the same log and check the dominant"
                      " wait primitive's ratio: near 1.0 rules out quantization, well above it"
                      " confirms the wait is the cause.")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate [audio-dbg] lines into an audio-delivery health report.")
    parser.add_argument("logs", nargs="*", help="run logs (PROSPER_AUDIO_DEBUG=1); '-' = stdin")
    parser.add_argument("--port", type=int, default=None,
                        help="analyze one port (default: every port with records)")
    parser.add_argument("--device-hz", type=int, default=48000,
                        help="the playback device's sample rate (default 48000)")
    parser.add_argument("--channels", type=int, default=2, help="output channels (default 2)")
    parser.add_argument("--bytes-per-frame", type=int, default=4,
                        help="bytes per sample per channel as delivered (4 = f32, 2 = s16; default 4);"
                             " only consulted for logs predating the emitter (grain=N) field")
    args = parser.parse_args()

    paths = args.logs if args.logs else ["-"]
    per_port, census = analyze(paths, args.port, args.device_hz, args.channels, args.bytes_per_frame)
    if not per_port:
        print("no [audio-dbg] records found (run with PROSPER_AUDIO_DEBUG=1)")
        return 1
    census_summary = dominant_census(census)
    for port in sorted(per_port):
        report_port(port, per_port[port], args.device_hz, args.channels, args.bytes_per_frame,
                    census_summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
