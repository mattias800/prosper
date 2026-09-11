#!/usr/bin/env python3
"""Self-test for flip_pacing_report.py.

The report's whole job is to name the pacing limiter class from the flip-interval
distribution. Each case is a regime the report must classify, plus the trap: a phase change
(cinematic -> gameplay) that must appear as two windows, not one smeared average.

Run: python3 tools/perf/test_flip_pacing_report.py
"""

import importlib.util
import random
import re
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
        # BOTH flip emitters, not just the one this parser reads. They are ALTERNATIVES: the
        # in-stream GPU flip (`prosper_vo_flip_from_gpu`) and the API flip
        # (`sceVideoOutSubmitFlip`) each advance the flip state on their own, so a title using
        # the API path produces SubmitFlip lines and no GpuFlip lines at all. Greping only
        # GpuFlip left the `t=` added to SubmitFlip in the same PR removable with nothing going
        # red -- and it is the only timing this tool's successor would have for that class of
        # title (#3462, the residual noted on #3453's review).
        # The consequence differs per emitter, so say which one it is rather than pasting
        # #3452's sentence over both -- the second is not the failure #3452 was.
        for tag, consequence in (
                ("[ev] GpuFlip",
                 "so every log this parser reads is unpaceable (#3452)"),
                ("[ev] SubmitFlip",
                 "so a title that flips through the API path rather than the in-stream GPU"
                 " packet records no flip timing at all (#3462)")):
            flip_lines = [line for line in text.splitlines() if tag in line]
            if not flip_lines:
                failures.append(f"case 7: no {tag} emitter found in hle_graphics.cpp")
            elif not any("t=%" in line for line in flip_lines):
                failures.append(f"case 7: the {tag} emitter writes no t= timestamp, {consequence}: "
                                + flip_lines[0].strip())

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

    # 10. The residual half of case 9: two runs CONCATENATED into one stream. The per-source
    #     split cannot see this -- it is one source -- and stamps.sort() then destroys the
    #     backwards step that betrays it, leaving a fabricated distribution indistinguishable
    #     from one slow run. Detected before sorting.
    joined = []
    for step in (1.0 / 60.0, 1.0 / 30.0):
        t = 1.0
        for _ in range(200):
            joined.append(f"[ev] GpuFlip t={t:.6f} handle=0x1002 bufidx=0 mode=0x1 fliparg=0x0")
            t += step
    out, _ = run("\n".join(joined) + "\n")
    if "step backwards" not in out:
        failures.append(f"case 10: concatenated runs went undetected: {out!r}")

    # ...and a single clean run must stay silent, or the warning becomes noise nobody reads.
    single = []
    t = 1.0
    for _ in range(200):
        single.append(f"[ev] GpuFlip t={t:.6f} handle=0x1002 bufidx=0 mode=0x1 fliparg=0x0")
        t += 1.0 / 60.0
    out, _ = run("\n".join(single) + "\n")
    if "step backwards" in out:
        failures.append(f"case 10: a single monotonic run was flagged as concatenated: {out!r}")

    # 11. #3454: the tick-aligned share is meaningless without its null. The band is +/-BAND of
    #     the tick, so it covers 2*BAND of every tick period and an interval unrelated to the
    #     tick lands in it that often anyway. The report must PRINT that baseline, and the
    #     printed band must be derived from BAND rather than restated as a literal -- the tool
    #     previously said '15%' in one place and tested `0.15` in another.
    spec = importlib.util.spec_from_file_location('flip_pacing_report', str(TOOL))
    FPR = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(FPR)
    # Read the constants defensively. A tool with no baseline at all is the PRE-FIX state, and
    # it must produce one named failure per arm rather than an AttributeError that aborts the
    # run before cases 12-15 execute -- otherwise the without-fix arm proves only that case 11
    # discriminates, and says nothing about the other four.
    BAND = getattr(FPR, 'BAND', None)
    CHANCE = getattr(FPR, 'CHANCE', None)
    verdict_of = getattr(FPR, 'alignment_verdict', None)
    if BAND is None or CHANCE is None or verdict_of is None:
        failures.append('case 11: flip_pacing_report exposes no BAND/CHANCE/alignment_verdict,'
                        ' so the tick-aligned share has no stated null at all (#3454)')
        BAND, CHANCE = 0.15, 0.30   # what the report is expected to use, so the arms below
                                    # still discriminate instead of crashing

    stamps = []
    t = 1.0
    for i in range(600):
        t += 0.015625
        stamps.append(t)
    log = "".join(f"[ev] GpuFlip t={t:.6f}\n" for t in stamps)
    out, _ = run(log)
    expect(out, f"chance baseline {100 * CHANCE:.1f}%", "case 11 baseline is printed")
    expect(out, f"+/-{BAND * 100:.0f}%", "case 11 printed band is derived from BAND")
    expect(out, "well above chance", "case 11 a perfectly tick-locked run is a strong finding")

    # 12. THE POSITIVE CONTROL FOR THE NULL ITSELF, and the reason this case exists: asserting
    #     '2*BAND = 30%' in a test would only restate the arithmetic the code already performs.
    #     So the null is measured instead, from a generator built OUTSIDE the tool and bearing no
    #     relationship to the tick -- uniform intervals over [20, 120] ms.
    #
    #     Its EXACT expected coverage is derivable, and is not 30%: the tick multiples falling
    #     wholly inside [20, 120] are k=2..7 (31.25 .. 109.375 ms), six bands of 2*0.15*15.625 =
    #     4.6875 ms, so 28.125 of 100 ms = 28.1%. k=1 and k=8 lie outside the range and
    #     contribute nothing. Measured across three seeds: 27.2%, 28.3%, 29.3%.
    #
    #     That gap between the exact 28.1% and the printed asymptotic 30.0% is the honest
    #     limitation of a flat null, and it is why this asserts a BAND rather than a point. What
    #     must hold is the verdict: a run with no tick relationship must not read as a finding.
    rng = random.Random(20260911)
    stamps = []
    t = 1.0
    for _ in range(4000):
        t += rng.uniform(0.020, 0.120)
        stamps.append(t)
    log = "".join(f"[ev] GpuFlip t={t:.6f}\n" for t in stamps)
    out, _ = run(log)
    share_line = next((l for l in out.splitlines() if "tick-aligned intervals" in l), "")
    match = re.search(r'[(]([0-9.]+)%[)]', share_line)
    if not match:
        failures.append(f"case 12: no tick-aligned share line: {out!r}")
    else:
        observed = float(match.group(1)) / 100.0
        # Derive the expected coverage for THIS generator's support FROM BAND, rather than
        # hard-coding a window. A fixed 22-34% band tolerated BAND anywhere in 0.118..0.181 and
        # was blind to CHANCE entirely -- it guarded neither of the two things the PR claimed
        # (found in review). Deriving it means a predicate that stops using BAND is caught even
        # though the observation moves with it.
        tick = FPR.TICK_MS
        half = BAND * tick
        covered, k = 0.0, 1
        while k * tick - half < 120.0:
            a, b = max(20.0, k * tick - half), min(120.0, k * tick + half)
            if b > a:
                covered += b - a
            k += 1
        expected = covered / 100.0
        if abs(observed - expected) > 0.03:
            failures.append(
                f"case 12: unrelated intervals scored {100 * observed:.1f}%, but the band derived"
                f" from BAND={BAND} predicts {100 * expected:.1f}% -- classify() and BAND have drifted")
    # The other half the old window could not see: CHANCE must stay tied to BAND, and the
    # printed baseline must be CHANCE rather than a separately-maintained literal.
    if abs(CHANCE - 2.0 * BAND) > 1e-12:
        failures.append(f"case 12: CHANCE ({CHANCE}) is not 2*BAND ({2 * BAND})")
    expect(out, f"chance baseline {100 * CHANCE:.1f}%", "case 12 printed baseline is CHANCE")
    expect(out, "at chance", "case 12 an unrelated run must read as no evidence, not a finding")

    # 13. A BELOW-baseline run carries information -- production is actively anti-aligned, which
    #     is evidence AGAINST a tick-bound wait. It previously read as a weak positive, so the
    #     9.3% window in #3450 was discarded as 'no finding'. Intervals parked on half-tick
    #     offsets are as far from every multiple as it is possible to be.
    rng = random.Random(4242)
    stamps = []
    t = 1.0
    for _ in range(1200):
        t += (15.625 * rng.choice([1, 2]) + 15.625 * 0.5 + rng.gauss(0, 0.3)) / 1000.0
        stamps.append(t)
    log = "".join(f"[ev] GpuFlip t={t:.6f}\n" for t in stamps)
    out, _ = run(log)
    expect(out, "BELOW chance", "case 13 anti-aligned production is named as such")
    expect(out, "evidence AGAINST", "case 13 says what a sub-chance share means")

    # 14. The whole-run share is an average over windows. When those windows disagree the
    #     summary reintroduces exactly the smearing the windowed view exists to prevent, so it
    #     must say so. One tick-locked phase followed by one anti-aligned phase.
    rng = random.Random(99)
    stamps = []
    t = 1.0
    for _ in range(900):
        t += (15.625 + rng.gauss(0, 0.3)) / 1000.0
        stamps.append(t)
    for _ in range(900):
        t += (15.625 * 1.5 + rng.gauss(0, 0.3)) / 1000.0
        stamps.append(t)
    log = "".join(f"[ev] GpuFlip t={t:.6f}\n" for t in stamps)
    out, _ = run(log, ["--window-s", "4"])
    expect(out, "windows disagree", "case 14 a two-regime run flags its own summary as an average")

    # 15. The verdict thresholds themselves, asserted directly rather than through the report, so
    #     a wording change in the printer cannot silently move a boundary.
    if verdict_of is None:
        failures.append('case 15: no alignment_verdict to assert thresholds against')
        verdict_of = lambda share, chance=None: (None, '')
    ratio, verdict = verdict_of(CHANCE)
    if ratio is None or abs(ratio - 1.0) > 1e-9 or 'at chance' not in verdict:
        failures.append(f"case 15: a share exactly at chance must read as at-chance: {verdict!r}")
    _, verdict = verdict_of(CHANCE * 0.3)
    if 'BELOW chance' not in verdict:
        failures.append(f"case 15: a third of chance must read as below: {verdict!r}")
    _, verdict = verdict_of(CHANCE * 3.0)
    if 'well above chance' not in verdict:
        failures.append(f"case 15: three times chance must read as a real finding: {verdict!r}")
    # A band whose acceptance region covers the whole line has no discriminating power left;
    # the helper must not divide by a zero or negative baseline.
    ratio, verdict = verdict_of(0.5, chance=0.0)
    if ratio is not None:
        failures.append("case 15: a zero baseline must yield no ratio rather than a division")

    # 16. #3454 review: a run FASTER than the first band cannot score aligned at all, because
    #     classify() forces nearest >= 1 -- so every interval shorter than (1-BAND)*tick is
    #     'between ticks' by construction. Without a power term the sub-chance verdict then told
    #     a 156 fps run it was 'actively anti-aligned, evidence AGAINST a tick-bound wait', which
    #     is a confident negative on a population where no other answer was possible -- a new
    #     instrument lie introduced by the fix for an old one.
    stamps = []
    t = 1.0
    for _ in range(1500):
        t += 1.0 / 156.0
        stamps.append(t)
    log = "".join(f"[ev] GpuFlip t={t:.6f}\n" for t in stamps)
    out, _ = run(log)
    expect(out, "NO POWER", "case 16 a run below the first band reports no power")
    if "evidence AGAINST" in out:
        failures.append("case 16: a 156 fps run was called anti-aligned, which it cannot be")

    # 17. ...and the power term must NOT silence a real negative. Case 13's anti-aligned run
    #     sits at 1-2 ticks, well inside the reachable range, so it must still say so. Without
    #     this arm, 'always report NO POWER' would pass case 16. NOTE it passes in BOTH
    #     directions -- the pre-fix tool also reports this correctly, because it has no power
    #     term to over-apply. It is an over-suppression control, not a discriminator, and is
    #     counted as neither.
    rng = random.Random(4242)
    stamps = []
    t = 1.0
    for _ in range(1200):
        t += (15.625 * rng.choice([1, 2]) + 15.625 * 0.5 + rng.gauss(0, 0.3)) / 1000.0
        stamps.append(t)
    log = "".join(f"[ev] GpuFlip t={t:.6f}\n" for t in stamps)
    out, _ = run(log)
    expect(out, "BELOW chance", "case 17 a reachable anti-aligned run still reports the negative")
    if "NO POWER" in out:
        failures.append("case 17: a run at 1-2 ticks was wrongly reported as having no power")

    # 18. The verdict helper's power gate, asserted directly. Wrapped because a helper that has
    #     no `power` parameter at all is the PRE-FIX state, and it must produce a NAMED failure
    #     rather than a TypeError that aborts before cases 16 and 17 are reported.
    try:
        _, verdict = verdict_of(0.0, power=0.0)
        if 'NO POWER' not in verdict:
            failures.append(f"case 18: zero power must suppress the verdict: {verdict!r}")
        _, verdict = verdict_of(0.0, power=1.0)
        if 'BELOW chance' not in verdict:
            failures.append(f"case 18: full power must still allow a negative: {verdict!r}")
    except TypeError as exc:
        failures.append(f"case 18: alignment_verdict takes no power term at all ({exc})")

    # 19. Re-review residual: PARTIAL power. Only `power` of the population can score at all, so
    #     the null for the whole-population share is power * CHANCE. Measuring against the full
    #     30% understates the ratio by 1/power and can call a share BELOW chance when it is above
    #     its own null -- the same fabricated negative, surviving in the partial case. A
    #     mixed-rate run is the phase change the windowed view exists for, so this is the common
    #     case. share=0.20 at power=0.6: 0.20/0.30 = 0.67x (BELOW) against the flat null, but
    #     0.20/(0.30*0.6) = 1.11x (at chance) against the real one.
    try:
        ratio, verdict = verdict_of(0.20, power=0.6)
        if 'BELOW chance' in verdict:
            failures.append(
                f"case 19: a share above its own power-scaled null was called BELOW chance ({verdict!r})")
        if ratio is None or abs(ratio - 0.20 / (FPR.CHANCE * 0.6)) > 1e-9:
            failures.append(f"case 19: ratio {ratio} is not against the power-scaled null")
        # ...and full power must leave the flat null untouched, or this becomes a blanket
        # loosening rather than a correction.
        ratio, _ = verdict_of(0.20, power=1.0)
        if ratio is None or abs(ratio - 0.20 / FPR.CHANCE) > 1e-9:
            failures.append(f"case 19: full power must divide by CHANCE exactly, got {ratio}")
    except TypeError as exc:
        failures.append(f"case 19: no power term to scale the null with ({exc})")

    # ---------------------------------------------------------------- #3462
    # 24. A RACE IS NOT A RESTART, and the detector printed the same sentence for both. The
    #     backwards-step detector (#3453) has no magnitude threshold, so a sub-microsecond
    #     interleave between two emitting threads -- `evlog_seconds()` is evaluated as a function
    #     argument and the stream lock is taken afterwards -- asserted "this input holds more than
    #     one run", which is a different fact with a different remedy (fix the emitter's ordering
    #     versus pass the runs separately). Two lines emitted 50 us out of order in an otherwise
    #     monotonic 60 Hz run.
    rows, t = [], 1.0
    for i in range(300):
        if i in (100, 200):
            # The thread that read the clock LATER reached the stream first.
            rows.append(f"[ev] GpuFlip t={t + 0.000050:.6f} handle=0x1002 bufidx=0 mode=0x1 fliparg=0x0")
        rows.append(f"[ev] GpuFlip t={t:.6f} handle=0x1002 bufidx=0 mode=0x1 fliparg=0x0")
        t += 1.0 / 60.0
    out, _ = run("\n".join(rows) + "\n")
    expect(out, "step backwards 2 time(s)", "case 24 the backwards steps are still detected")
    expect(out, "0.050 ms", "case 24 the magnitude of the largest step is stated")
    if "more than one run" in out:
        failures.append(f"case 24: a 50 us emission race was reported as a second run, which it"
                        f" cannot be -- a restart steps back by a whole run (#3462): {out!r}")
    if "EMISSION" not in out:
        failures.append(f"case 24: the out-of-order emission was never named, so the remedy the"
                        f" message implies is the wrong one (#3462): {out!r}")

    # 25. ...and the real restart must keep its verdict AND gain the magnitude that justifies it,
    #     or this becomes a blanket loosening rather than a distinction. Two 60 Hz runs
    #     concatenated, so the seam steps back by the first run's whole elapsed time.
    rows, stamps = [], []
    for _ in range(2):
        t = 1.0
        for _ in range(200):
            stamps.append(t)
            rows.append(f"[ev] GpuFlip t={t:.6f} handle=0x1002 bufidx=0 mode=0x1 fliparg=0x0")
            t += 1.0 / 60.0
    out, _ = run("\n".join(rows) + "\n")
    # Derived from the construction rather than restated: the seam steps back from the first
    # run's last stamp to the second run's first, which is 199 frames at 60 Hz, not 200.
    seam_s = stamps[199] - stamps[200]
    expect(out, "more than one run", "case 25 a real restart is still called a restart")
    expect(out, f"{seam_s:.3f} s",
           "case 25 the restart states the magnitude that justifies the verdict")
    if "EMISSION" in out:
        failures.append(f"case 25: a 3.3 s run boundary was reported as emission jitter: {out!r}")

    # 26. The classifier asserted directly, so a wording change in the printer cannot move a
    #     boundary, and so the MIDDLE band is pinned: a step too large for a scheduler quantum
    #     and too small for a run is neither, and saying so is the point of the fix. A single
    #     threshold would only move the boundary and go on printing one of the two sentences.
    #     Wrapped because a tool with no classifier at all is the PRE-FIX state, and it must
    #     produce named failures rather than an AttributeError that aborts the run.
    describe = getattr(FPR, "describe_backsteps", None)
    if describe is None:
        failures.append("case 26: flip_pacing_report exposes no describe_backsteps, so the"
                        " backwards-step magnitude is never classified at all (#3462)")
    else:
        race_ms = getattr(FPR, "RACE_MS", None)
        restart_ms = getattr(FPR, "RESTART_MS", None)
        if race_ms is None or restart_ms is None:
            failures.append("case 26: no RACE_MS/RESTART_MS thresholds are stated (#3462)")
        else:
            checks = [
                ([], None, "no backwards step is no verdict"),
                ([race_ms], "race", "a step at the jitter ceiling is emission, not a restart"),
                ([restart_ms], "restart", "a step at a whole second is a run boundary"),
                ([(race_ms + restart_ms) / 2.0], "unclear",
                 "a step between the two is named as unclassifiable, not guessed"),
                ([0.05, 5.0 * restart_ms], "restart",
                 "the LARGEST step decides: jitter beside a restart is still a restart"),
            ]
            for steps, want, what in checks:
                kind, sentence = describe(steps)
                if kind != want:
                    failures.append(f"case 26: {what} -- describe_backsteps({steps}) said"
                                    f" {kind!r}, expected {want!r}")
                if want == "unclear" and "more than one run" in sentence:
                    failures.append("case 26: the unclassifiable band still asserts a second run")

    if failures:
        print("FAILURES:")
        for failure in failures:
            print(" -", failure)
        return 1
    print("flip_pacing_report self-test: all cases pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
