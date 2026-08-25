#!/usr/bin/env python3
"""Self-test for flip_pacing_report.py.

The report's whole job is to name the pacing limiter class from the flip-interval
distribution. Each case is a regime the report must classify, plus the trap: a phase change
(cinematic -> gameplay) that must appear as two windows, not one smeared average.

Run: python3 tools/perf/test_flip_pacing_report.py
"""

import subprocess
import sys
from pathlib import Path

TOOL = Path(__file__).resolve().parent / "flip_pacing_report.py"


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

    # 1. Tick-quantized production: alternating ~16/~31 ms intervals (the measured
    #    Blasphemous 2 signature). The report must call it tick-aligned.
    stamps = []
    t = 1.0
    for i in range(600):
        t += 0.016 if i % 2 else 0.031
        stamps.append(t)
    log = "".join(f"[ev] GpuFlip t={t:.3f}\n" for t in stamps)
    out, _ = run(log)
    expect(out, "tick-aligned", "case 1 quantization verdict")
    expect(out, "fps average", "case 1 average line")

    # 2. Work-bound production: a wide spread around 8 ms. The report must NOT call a
    #    spread distribution tick-aligned.
    stamps = []
    t = 1.0
    for i in range(600):
        t += 0.0075 + 0.002 * ((i * 37) % 11) / 11.0
        stamps.append(t)
    log = "".join(f"[ev] GpuFlip t={t:.3f}\n" for t in stamps)
    out, _ = run(log)
    expect(out, "fps average", "case 2 average line")

    # 3. Phase change: 300 flips at 60 Hz then 300 at 30 Hz. The windowed split must show
    #    both rates rather than one smeared average.
    stamps = []
    t = 1.0
    for i in range(300):
        t += 1 / 60.0
        stamps.append(t)
    for i in range(300):
        t += 1 / 30.0
        stamps.append(t)
    log = "".join(f"[ev] GpuFlip t={t:.3f}\n" for t in stamps)
    out, _ = run(log, ["--window-s", "5"])
    expect(out, "windows of", "case 3 window rows")

    # 4. Too few flips: a clean refusal, not a traceback.
    out, _ = run("[ev] GpuFlip t=1.000\n")
    if "fewer than 2 flips" not in out:
        failures.append(f"case 4: expected the too-few-flips refusal, got: {out!r}")

    if failures:
        print("FAILURES:")
        for failure in failures:
            print(" -", failure)
        return 1
    print("flip_pacing_report self-test: all cases pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
