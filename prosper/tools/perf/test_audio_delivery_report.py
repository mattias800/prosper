#!/usr/bin/env python3
"""Self-test for audio_delivery_report.py.

The report's whole job is to name the failure mode (clock deficit vs quantized mixer wake
vs device starvation) from a delivery log. Each case here is a failure mode the report must
classify, plus the traps: a port filter that must exclude other ports, and a zero-queue
all-fast log that must NOT be called a clock deficit.

Run: python3 tools/perf/test_audio_delivery_report.py
"""

import subprocess
import sys
from pathlib import Path

TOOL = Path(__file__).resolve().parent / "audio_delivery_report.py"


def dbg(port=17, gap=5.33, frames=256, queued=1024, bpf=4, channels=2):
    # grain DERIVED from frames rather than hardcoded. It was a literal (grain=1024) while every
    # case used 256 frames of f32 stereo -- a real grain of 2048 B -- so the fixture contradicted its
    # own arithmetic. Invisible while the parser discarded the field, and it inverted the
    # half-a-grain arm the moment the parser started reading it. Derived, it cannot drift again.
    grain = frames * bpf * channels
    return (f"[audio-dbg] port={port} gap={gap:.2f}ms frames={frames}"
            f" queued_before={queued} (grain={grain})\n")


def run(log_text, extra=None):
    proc = subprocess.run(
        [sys.executable, str(TOOL), "-", *(extra or [])],
        input=log_text, capture_output=True, text=True)
    return proc.stdout + proc.stderr, proc.returncode


def main():
    failures = []

    def expect(text, needle, what):
        if needle not in text:
            failures.append(f"{what}: expected {needle!r} in the report")

    def reject(text, needle, what):
        if needle in text:
            failures.append(f"{what}: expected {needle!r} to be absent")

    # 0. The queue thresholds mean GRAINS, and this arm exists because they did not.
    #
    #    They were bytes_per_frame * channels -- one FRAME, 8 bytes for f32 stereo -- so "below 1
    #    grain" tested q < 8 and "below 1/4 grain" tested q < 2. Every case below passed anyway,
    #    because they only ever assert that a LABEL is present, never that the number under it is
    #    right. A run reporting "1.1% below a quarter grain" was really 1.1% completely dry, and that
    #    figure was quoted into an issue before anyone noticed.
    #
    #    grain here is 256 frames * 4 bytes * 2 channels = 2048 B. A queue of 1024 B is half a
    #    grain: below one grain, above a quarter. Under the old arithmetic it was above BOTH
    #    thresholds, so this arm fails on the bug and passes on the fix.
    log = "".join(dbg(gap=5.33, frames=256, queued=1024) for _ in range(200))
    out, _ = run(log, ["--bytes-per-frame", "4", "--channels", "2"])
    expect(out, "arrivals with under 1 grain buffered: 200", "case 0: half a grain is below one grain")
    expect(out, "arrivals below 1/4 grain:          0", "case 0: half a grain is NOT below a quarter")

    #    And an empty queue must register as empty, on its own row rather than only as "below".
    log = "".join(dbg(gap=5.33, frames=256, queued=0) for _ in range(200))
    out, _ = run(log, ["--bytes-per-frame", "4", "--channels", "2"])
    expect(out, "arrivals with an EMPTY queue:         200", "case 0: an empty queue is reported empty")

    # 1. Real-time delivery, deep queue: no defect. The verdict must NOT name a clock
    #    deficit (a plausible wrong call: an average at the device rate with jitter).
    log = "".join(dbg(gap=5.33, queued=3072) for _ in range(200))
    out, _ = run(log)
    expect(out, "delivery matches the device rate", "case 1 verdict")

    # 2. Clock deficit: every call arrives 1/3 late with an empty queue. The verdict must
    #    name the clock deficit -- this is exactly the Blasphemous 2 signature.
    log = "".join(dbg(gap=15.7, queued=64) for _ in range(300))
    out, _ = run(log)
    expect(out, "clock deficit", "case 2 verdict")
    # A NUMBER, not the row label. This assertion used to be `expect(out, "the device starved
    # between deliveries")` -- a string printed on every run regardless of the count, so review
    # confirmed that hardwiring under_one = 0 left it green. queued=64 is under one 2048 B grain
    # but not empty, so this pins the thin-cushion count AND that it is not miscounted as
    # starvation, which is the distinction the label fix in this change is about.
    expect(out, "arrivals with under 1 grain buffered: 300", "case 2: all 300 are a thin cushion")
    expect(out, "arrivals with an EMPTY queue:         0", "case 2: a thin cushion is NOT starvation")

    # 3. Quantized mixer wake: the combined delivery averages real-time (256 frames per
    #    5.33 ms) but the arrivals cluster -- 1 ms after a tick, then 9.66 ms of silence.
    #    The verdict must name the burst/quantization problem, not a clock deficit.
    log = "".join(dbg(gap=1.0, queued=4096) + dbg(gap=9.66, queued=4096)
                  for _ in range(300))
    out, _ = run(log)
    expect(out, "delivery matches the device rate", "case 3 drift verdict")
    expect(out, "burst/quantization problem", "case 3 burst verdict")
    reject(out, "the guest's audio clock runs below the device rate", "case 3 must not miscall a clock deficit")

    # 4. Port filter: records for another port must not leak into the report.
    log = ("".join(dbg(port=17, gap=15.7, queued=64) for _ in range(100))
           + "".join(dbg(port=18, gap=99.0, queued=0) for _ in range(100)))
    out, _ = run(log, ["--port", "17"])
    reject(out, "99.00", "case 4 port filter")
    expect(out, "port 17:", "case 4 port header")

    # 5. Empty input: a clean refusal, not a traceback.
    out, _ = run("")
    if "no [audio-dbg] records" not in out:
        failures.append(f"case 5: expected the no-records refusal, got: {out!r}")

    if failures:
        print("FAILURES:")
        for failure in failures:
            print(" -", failure)
        return 1
    print("audio_delivery_report self-test: all cases pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
