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
    below one grain (the device starved between deliveries) and below a quarter grain
    (audible underrun territory);
  * effective delivery rate vs the device rate -- a persistent deficit is a CLOCK DRIFT
    between the guest's budgeted audio clock and the device crystal; no cushion survives it,
    and the fix is drift compensation, not deeper buffering;
  * burst clustering -- the fraction of arrivals that carry more than one grain of audio
    after a gap of more than two grain periods, which names a quantized mixer wake.

WHAT THIS TOOL DOES NOT SEE
---------------------------
`[audio-dbg]` is emitted per accepted `output()` call from the guest. Grains the guest never
delivers (a mixer that skipped a period) appear only as a longer gap. Content-level defects
(a mixer that produced silence or repeated samples) are invisible here; the queue level and
the delivery rate are the report's whole world.

Usage:
    audio_delivery_report.py [LOG ...] [--port P] [--device-hz H] [--channels C] [--bytes-per-frame N]
"""

import argparse
import re
import sys
from collections import defaultdict

RECORD = re.compile(
    r"\[audio-dbg\] port=(\d+) gap=([0-9.]+)ms frames=(\d+) queued_before=(\d+)")

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
    for path, line in parse_streams(paths):
        m = RECORD.search(line)
        if not m:
            continue
        port = int(m.group(1))
        if port_filter is not None and port != port_filter:
            continue
        gap_ms = float(m.group(2))
        frames = int(m.group(3))
        queued = int(m.group(4))
        slot = per_port.setdefault(port, {
            "gaps": [], "frames": [], "queued": [], "sources": defaultdict(int),
        })
        slot["gaps"].append(gap_ms)
        slot["frames"].append(frames)
        slot["queued"].append(queued)
    return per_port


def percentile(sorted_values, fraction):
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, int(fraction * len(sorted_values)))
    return sorted_values[index]


def report_port(port, slot, device_hz, channels, bytes_per_frame):
    gaps = sorted(slot["gaps"])
    queued = sorted(slot["queued"])
    frames_total = sum(slot["frames"])
    calls = len(slot["gaps"])
    span_s = sum(gaps) / 1000.0
    grain_frames = slot["frames"][0] if slot["frames"] else 0
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
        under_one = sum(1 for q in queued if q < bytes_per_frame * channels)
        under_quarter = sum(1 for q in queued if q < bytes_per_frame * channels // 4)
        print(f"  queue at arrival: mean {mean_q:.0f} B  min {queued[0]} B")
        print(f"  arrivals below 1 grain of buffer:  {under_one}"
              f" ({100 * under_one / len(queued):.1f}%) -- the device starved between"
              f" deliveries")
        print(f"  arrivals below 1/4 grain:          {under_quarter}"
              f" ({100 * under_quarter / len(queued):.1f}%) -- audible underrun territory")

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
            print("  VERDICT: the inter-arrival cadence clusters beyond two grain periods --"
                  " the mixer's wake is quantized (a timed wait resolving on a coarse tick),"
                  " not a buffering defect. Fix the wait primitive's resolution.")


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
                        help="bytes per frame per channel-set as delivered (default 4)")
    args = parser.parse_args()

    paths = args.logs if args.logs else ["-"]
    per_port = analyze(paths, args.port, args.device_hz, args.channels, args.bytes_per_frame)
    if not per_port:
        print("no [audio-dbg] records found (run with PROSPER_AUDIO_DEBUG=1)")
        return 1
    for port in sorted(per_port):
        report_port(port, per_port[port], args.device_hz, args.channels, args.bytes_per_frame)
    return 0


if __name__ == "__main__":
    sys.exit(main())
