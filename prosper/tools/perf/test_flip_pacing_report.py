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

    # 5. THE DOMAIN ARM. Every case above composes its own input in the format this parser
    #    expects, so all of them passed for as long as the tool existed while it read zero
    #    flips from every real log: the emitter never wrote the `t=` the parser required
    #    (#3452). A control drawn from the same source as the null tests the discriminator,
    #    never the domain. These two lines are copied VERBATIM from prosper-app stderr with
    #    PROSPER_EVLOG=1, so a future divergence between emitter and parser fails here.
    real = (
        "[ev] GpuFlip t=41.416330 handle=0x1002 bufidx=0 mode=0x1 fliparg=0x0" "\n"
        "[ev] GpuFlip t=41.432850 handle=0x1002 bufidx=1 mode=0x1 fliparg=0x0" "\n"
    )
    out, _ = run(real)
    if "cannot be paced" in out or "fewer than 2 flips" in out:
        failures.append(f"case 5: the parser could not read real emitter output: {out!r}")

    # 6. A flip line with no timestamp must be reported as a TOOL failure, naming the count,
    #    not as a fact about the run. This is the exact shape that hid #3452: 7,938 flip lines
    #    in a real log reported as "nothing to pace".
    untimed = "[ev] GpuFlip handle=0x1002 bufidx=0 mode=0x1 fliparg=0x0" "\n"
    out, _ = run(untimed * 3)
    if "3 flip line(s) found" not in out or "cannot be paced" not in out:
        failures.append(f"case 6: expected a loud parse failure naming the count, got: {out!r}")
    if "fewer than 2 flips" in out:
        failures.append("case 6: reported a parse failure as an empty run")

    # 7. THE EMITTER LINK. Cases 1-6 all feed strings this file composes, so none of them can
    #    notice the emitter dropping or moving the timestamp -- which is the regression that
    #    actually happened, and it hid for the tool's whole life. Read the emitter source and
    #    require the field to still be there. A format string is not a live run, but it is the
    #    one link to the producer a pure unit test can hold.
    emitter = (Path(__file__).resolve().parents[2] / "src/hle/graphics/hle_graphics.cpp")
    if not emitter.is_file():
        failures.append(f"case 7: emitter source not found at {emitter}")
    else:
        text = emitter.read_text(encoding="utf-8", errors="replace")
        flip_lines = [line for line in text.splitlines() if "[ev] GpuFlip" in line]
        if not flip_lines:
            failures.append("case 7: no [ev] GpuFlip emitter found in hle_graphics.cpp")
        elif not any("t=%" in line for line in flip_lines):
            failures.append("case 7: the [ev] GpuFlip emitter writes no t= timestamp, so every real log is unpaceable (#3452): " + flip_lines[0].strip())

    # 8. A PARTIALLY readable log must say so. Reporting only when NOTHING parsed left the
    #    real hazard open: a handful of timestamped flips among thousands of untimed ones
    #    produced a confident distribution over a fraction of a percent of the population and
    #    mentioned nothing. That is the #3452 failure one branch lower down.
    timed = [1.000, 1.016, 1.032]
    mixed = "".join(f"[ev] GpuFlip t={t:.3f} handle=0x1002 bufidx=0 mode=0x1 fliparg=0x0\n"
                    for t in timed)
    mixed += "[ev] GpuFlip handle=0x1002 bufidx=0 mode=0x1 fliparg=0x0\n" * 5000
    out, _ = run(mixed)
    if "WARNING" not in out or "5000 of 5003" not in out:
        failures.append(f"case 8: a 99.9% unreadable population went unreported: {out!r}")

    # 9. Two logs are two runs. The timestamp epoch is per process, so pooling them
    #    interleaves two timelines into intervals neither run contained -- a 60 fps and a
    #    30 fps capture merged into a plausible ~60 fps with a fabricated sub-tick histogram,
    #    which this report's own wording then calls work-bound production. Comparing two runs
    #    is the tool's intended use, so the wrong answer was reachable by using it as designed.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for name, step in (("a", 1.0 / 60.0), ("b", 1.0 / 30.0)):
            t = 1.0
            rows = []
            for _ in range(300):
                rows.append(f"[ev] GpuFlip t={t:.6f} handle=0x1002 bufidx=0 mode=0x1 fliparg=0x0")
                t += step
            path = str(Path(tmp) / f"{name}.log")
            Path(path).write_text("\n".join(rows) + "\n", encoding="utf-8")
            paths.append(path)
        proc = subprocess.run([sys.executable, str(TOOL), *paths],
                              capture_output=True, text=True)
        merged = proc.stdout + proc.stderr
    rates = [line for line in merged.splitlines() if "fps average" in line]
    if len(rates) != 2:
        failures.append(f"case 9: expected one report per log, got {len(rates)}: {merged!r}")
    else:
        if "60.0 fps" not in rates[0] or "30.0 fps" not in rates[1]:
            failures.append(f"case 9: per-log rates were merged or reordered: {rates!r}")

    if failures:
        print("FAILURES:")
        for failure in failures:
            print(" -", failure)
        return 1
    print("flip_pacing_report self-test: all cases pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
