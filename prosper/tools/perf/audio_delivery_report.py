#!/usr/bin/env python3
"""Aggregate `[audio-dbg]` lines into an audio-delivery health report.

Reports delivery cadence, input queue occupancy and rate relative to --device-hz.
An empty SDL input queue is not proof of a playback underrun: converted audio may
already be playing downstream. Rate differences do not identify their own cause.
Optional PROSPER_AUDIO_DEMAND=1 records measure consumption requests per stream and
phase. Their additional input bytes are approximate under resampling and are not
hardware XRUNs or a count of silence inserted. Counters are cumulative per open;
only the last snapshot is used, never the sum of periodic reports. Use one process
run per log file. Content correctness and physical backend XRUNs need other evidence.

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

DEMAND = re.compile(
    r"\[audio-demand\] t_us=(\d+) port=(\d+) generation=(\d+) phase=(startup|active|paused) "
    r"calls=(\d+) requested_bytes=(\d+) shortfall_calls=(\d+) additional_bytes=(\d+) "
    r"first_ns=(\d+) last_ns=(\d+)")
DEMAND_OPEN = re.compile(r"\[audio-demand-open\] port=(\d+) generation=(\d+) (.*)")

# Process-wide wait context; no producer thread/caller identity is present.
TIMEDWAIT_RECORD = re.compile(
    r"\[timedwait \S+\]\s+(\S+)\s+calls=(\d+)\s+requested=\s*(?:[0-9.]+\s*ms|\?)"
    r"\s+actual=\s*[0-9.]+\s*ms(?:\s+x([0-9.]+))?")
QUANTIZED_RATIO_THRESHOLD = 1.3


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
    demand, formats = {}, {}
    for path, line in parse_streams(paths):
        dm = DEMAND.search(line)
        om = DEMAND_OPEN.search(line)
        if dm:
            t, port, generation = map(int, dm.groups()[:3])
            if port_filter is None or port == port_filter:
                key = (path, port, generation, dm.group(4))
                values = tuple(map(int, dm.groups()[4:]))
                prior = demand.get(key)
                # Reject reset/concatenated runs; otherwise the last row could hide
                # an earlier shortfall and falsely report a clean run.
                valid = (not prior or (prior[2] and t >= prior[0] and
                         all(x >= y for x, y in zip(values[:4], prior[1][:4]))))
                valid = valid and values[2] <= values[0] and values[5] >= values[4]
                if prior and prior[1][0]:
                    valid = valid and values[4] == prior[1][4] and values[5] >= prior[1][5]
                demand[key] = (t, values, valid)
        elif om:
            formats[(path, int(om.group(1)), int(om.group(2)))] = om.group(3)
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
    return per_port, census, demand, formats


def dominant_census(census):
    """Most-called process-wide wait primitive and its call-weighted mean ratio.

    These records do not identify the audio producer or establish its wake cadence.
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
        # describing a condition they no longer match. Even the EMPTY row cannot establish downstream starvation.
        print(f"  arrivals with under 1 grain buffered: {under_one}"
              f" ({100 * under_one / len(queued):.1f}%) -- thin cushion")
        print(f"  arrivals below 1/4 grain:          {under_quarter}"
              f" ({100 * under_quarter / len(queued):.1f}%) -- very thin")
        print(f"  arrivals with an EMPTY queue:         {empty}"
              f" ({100 * empty / len(queued):.1f}%) -- input empty; downstream playback unknown")

    if span_s > 0 and device_hz:
        # The emitter logs gap=0.00ms for a port's FIRST arrival -- there is no previous arrival
        # on that port to measure from -- so `gaps` holds only (calls - 1) REAL inter-arrival
        # intervals, and span_s = sum(gaps) covers exactly those: the time from the first arrival
        # to the last. frames_total, though, sums frames from all `calls` arrivals, including the
        # first one, whose own accumulation time was never captured by any logged gap (it may
        # have been building up for a full grain period before this log even starts). Dividing
        # the full frame total by that (calls - 1)-interval span overcounts by one arrival's
        # worth of frames, inflating delivered_hz by a factor of calls / (calls - 1) -- roughly
        # +1/(calls - 1) at the calls this tool typically sees -- issue #3061. Measured example:
        # 11 records of 100 frames at an exact 1000 Hz real-time cadence have a TRUE drift of
        # 0.00%, but frames_total/span_s reports 1100 frames / 1.0 s = 1100 vs device 1000, i.e.
        # +10.00% -- comfortably past the +-0.3% threshold below, so a real-time log at small N
        # was reported as running the audio clock fast.
        #
        # Fix: drop the first arrival's frames from the numerator to match what span_s already
        # spans, rather than stretching span_s by an imputed extra grain period -- the two give
        # the same answer only when every arrival is the same size, and dropping the frame needs
        # no such assumption. Gate on the emitter's own zero-gap sentinel (not just "index 0")
        # so a log excerpt that starts mid-stream -- where the first logged gap is a real,
        # already-elapsed interval, not the emitter's "no previous arrival" zero -- is left
        # unadjusted rather than having a genuine arrival's frames silently discarded.
        frames_in_span = frames_total
        if slot["gaps"][0] == 0.0:
            frames_in_span = frames_total - slot["frames"][0]
        delivered_hz = frames_in_span / span_s
        device_hz_total = device_hz
        drift_pct = 100.0 * (delivered_hz - device_hz_total) / device_hz_total
        print(f"  effective delivery: {delivered_hz:.0f} frames/s vs device {device_hz_total}"
              f" -> drift {drift_pct:+.2f}%")
        if drift_pct < -0.3:
            print("  VERDICT: measured delivery deficit relative to the configured rate;"
                  " investigate guest production, scheduling and pacing to establish the cause.")
        elif drift_pct > 0.3:
            print("  VERDICT: measured delivery exceeds the configured rate;"
                  " check the selected rate and production/pacing before changing buffering.")
        else:
            print("  VERDICT: delivery matches the device rate on average;"
                  " consumption continuity and audible correctness remain unmeasured by arrivals.")

    if grain_ms > 0:
        bursts = sum(1 for g in gaps if g > 2 * grain_ms)
        if bursts > len(gaps) * 0.2 and mean_gap > 1.5 * grain_ms:
            print("  Cadence clusters beyond two grain periods: late production and delayed wakes"
                  " can both cause this; arrivals cannot tell them apart.")
            if census is not None and census["ratio"] is not None:
                timing = ("near requested duration" if census["ratio"] < QUANTIZED_RATIO_THRESHOLD
                          else "longer than requested duration")
                print(f"  Process wait context: {census['name']} x{census['ratio']:.2f}"
                      f" ({census['calls']} calls), {timing}.")
                print("  This aggregate does not identify the audio producer's thread/caller;"
                      " it neither proves nor excludes delayed mixer wakes.")
            else:
                print("  PROSPER_TIMEDWAIT_CENSUS=1 can add process wait context;"
                      " correlate producer thread/caller and scheduling before assigning a cause.")


def report_demand(demand, formats):
    if not demand:
        print("SDL consumption demand: unavailable (run with PROSPER_AUDIO_DEMAND=1)")
        return
    print("SDL consumption demand: cumulative input-format requests; not hardware XRUNs")
    for (path, port, generation, phase), (t, values, valid) in sorted(demand.items()):
        calls, requested, shortfalls, additional, first, last = values
        print(f"  source={path} port={port} generation={generation} phase={phase} snapshot_us={t}")
        print("    " + formats.get((path, port, generation), "input/device format context unavailable"))
        if not valid:
            print("    INVALID counter sequence: use one process run per log file")
        elif not calls:
            print("    UNOBSERVED: no consumption callbacks in this phase")
        else:
            print(f"    calls={calls} requested_bytes={requested} shortfall_calls={shortfalls}"
                  f" additional_bytes={additional} first_ns={first} last_ns={last}")
            print("    Additional bytes are approximate under resampling; no silence duration inferred.")


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
    per_port, census, demand, formats = analyze(paths, args.port, args.device_hz, args.channels, args.bytes_per_frame)
    if not per_port:
        print("no [audio-dbg] records found (run with PROSPER_AUDIO_DEBUG=1)")
        if not demand:
            report_demand(demand, formats)
            return 1
    census_summary = dominant_census(census)
    for port in sorted(per_port):
        report_port(port, per_port[port], args.device_hz, args.channels, args.bytes_per_frame,
                    census_summary)
    report_demand(demand, formats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
