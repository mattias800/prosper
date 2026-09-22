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
    # ...and the HEADLINE must be marked void with it. Deriving the rate from the wall-clock span
    # (#3560) is right for one run and meaningless for several: sorted, the span covers roughly
    # one run's timeline while the flips come from all of them, so the rate over-counts by about
    # the number of runs. Saying the intervals are not real while printing a confident rate above
    # them would be the corrected instrument lying in a new place.
    if "NOT meaningful" not in out:
        failures.append(f"case 10: the rate over a concatenated span was printed without a"
                        f" caveat, so the restart warning and the headline disagree: {out!r}")

    # ...and a single clean run must stay silent, or the warning becomes noise nobody reads.
    single = []
    t = 1.0
    for _ in range(200):
        single.append(f"[ev] GpuFlip t={t:.6f} handle=0x1002 bufidx=0 mode=0x1 fliparg=0x0")
        t += 1.0 / 60.0
    out, _ = run("\n".join(single) + "\n")
    if "step backwards" in out:
        failures.append(f"case 10: a single monotonic run was flagged as concatenated: {out!r}")
    if "NOT meaningful" in out:
        failures.append(f"case 10: a single clean run's rate was voided, so the caveat is"
                        f" automatic and carries no information: {out!r}")

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

    # ---------------------------------------------------------------- #3560
    # A HELPER FOR THE ARMS BELOW. Builds a timeline holding exactly `n_long` long intervals and
    # `n_burst` sub-0.5 ms ones, spread evenly, over exactly `span_s` seconds -- so the run's TRUE
    # rate is (n_long + n_burst) / span_s, computed here from the construction and never from the
    # tool. The burst fraction is the free variable, which is the point: it is what the tool was
    # silently filtering out of its own headline.
    def burst_log(n_long, n_burst, span_s, burst_ms=0.2, t0=1.0):
        long_ms = (span_s * 1000.0 - n_burst * burst_ms) / n_long
        ivals = []
        for i in range(n_long):
            ivals.append(long_ms)
            if n_burst and (i * n_burst) // n_long != ((i + 1) * n_burst) // n_long:
                ivals.append(burst_ms)
        t = t0
        rows = [f"[ev] GpuFlip t={t:.6f} handle=0x1002 bufidx=0 mode=0x1 fliparg=0x0"]
        for ms in ivals:
            t += ms / 1000.0
            rows.append(f"[ev] GpuFlip t={t:.6f} handle=0x1002 bufidx=0 mode=0x1 fliparg=0x0")
        return "\n".join(rows) + "\n"

    def headline(out):
        m = re.search(r"-> ([0-9.]+) fps average", out)
        return None if m is None else float(m.group(1))

    # 20. THE ARM FOR #3560. The headline rate was 1000/mean of the RETAINED intervals, and the
    #     retained fraction varies per run -- so two runs' headline figures were means over
    #     different populations while reading as the runs' rates. Measured on the #3379 A/B: 44.1
    #     vs 59.2 fps reported against a wall-clock truth of 57.8 vs 97.9, a 1.34x ratio where the
    #     truth is 1.65x, which made a working pacing fix look like it had missed by a quarter.
    #
    #     So the arm is NOT 'a rate is printed' -- that passes either way. Two runs with the SAME
    #     true rate and DIFFERENT burst fractions must report the same figure. Both carry 1200
    #     intervals over 20.000 s, i.e. 60.0 flips/s by construction; one drops 10% of them to the
    #     filter and the other 50%. Pre-fix they report 54.1 and 30.2.
    sparse = burst_log(n_long=1080, n_burst=120, span_s=20.0)   # 10% burst
    dense = burst_log(n_long=600, n_burst=600, span_s=20.0)     # 50% burst
    out_sparse, _ = run(sparse)
    out_dense, _ = run(dense)
    r_sparse, r_dense = headline(out_sparse), headline(out_dense)
    if r_sparse is None or r_dense is None:
        failures.append(f"case 20: no headline rate in one of the reports: "
                        f"{out_sparse!r} / {out_dense!r}")
    else:
        for name, rate in (("10%-burst", r_sparse), ("50%-burst", r_dense)):
            if abs(rate - 60.0) > 0.6:
                failures.append(
                    f"case 20: the {name} run is 1200 flips over 20.000 s = 60.0 flips/s by"
                    f" construction, and the report says {rate} fps -- the headline is a mean over"
                    f" the filtered population, not the run's rate (#3560)")
        if abs(r_sparse - r_dense) > 0.5:
            failures.append(
                f"case 20: two runs at the same true rate reported {r_sparse} and {r_dense} fps"
                f" because they retained different fractions of their intervals, so the figures"
                f" this tool exists to compare are not comparable (#3560)")

    # 21. ...and the filtered population must be STATED wherever it is used, or the mean printed
    #     beside the headline silently describes a different population from the headline itself.
    #     The retention is a fraction, not just a count: `intervals=12318` next to `flips=16028`
    #     never told anyone the two lines disagreed on purpose.
    expect(out_dense, "600 of 1200 retained", "case 21 the retained count is related to the total")
    expect(out_dense, "50.0%", "case 21 the retention is stated as a fraction")
    if "NOT the run's period" not in out_dense:
        failures.append(f"case 21: the retained mean is printed without saying it is not the run's"
                        f" period, so 1000/mean still reads as the rate: {out_dense!r}")

    # 22. THE SAME DEFECT ONE LEVEL DOWN, which is where this file's history says a correction
    #     tends to leave it (instrument trap 275: three corrections each carried the previous
    #     defect forward in a narrower form). Every window row printed 1000/mean of its own
    #     retained subset, so a bursty phase and a genuinely slower phase produced identical rows;
    #     and the windows were cut over the FILTERED list while their ordinals were labelled
    #     'flips'. This run holds two 10 s phases, both at 60 flips/s: one clean, one paired. All
    #     four windows must read ~60 fps. Pre-fix they read 60.0, 60.0, 45.1, 30.2.
    clean = burst_log(n_long=600, n_burst=0, span_s=10.0, t0=1.0)
    paired = burst_log(n_long=300, n_burst=300, span_s=10.0, t0=11.0)
    # Drop `paired`'s first line: it would duplicate the seam rather than continue the timeline.
    two_phase = clean + "\n".join(paired.strip().splitlines()[1:]) + "\n"
    out, _ = run(two_phase, ["--window-s", "5"])
    rows = re.findall(r"flips\s+(\d+)-\s*(\d+):\s+([0-9.]+) fps", out)
    if len(rows) < 4:
        failures.append(f"case 22: expected four window rows, got {rows!r}: {out!r}")
    else:
        for start, end, fps in rows:
            if abs(float(fps) - 60.0) > 3.0:
                failures.append(
                    f"case 22: window {start}-{end} holds 5 s of a 60 flips/s phase and reports"
                    f" {fps} fps -- the window rate is 1000/mean of its retained subset (#3560)")
        # The ordinals say 'flips', so the last one must reach the last interval. Cut over the
        # filtered list it stops at the retained count, which on this run is 900 of 1200.
        if int(rows[-1][1]) != 1200:
            failures.append(
                f"case 22: the window ordinals are labelled 'flips' but the last one ends at"
                f" {rows[-1][1]} of 1200 intervals, so they index the filtered list (#3560)")

    # 23. Degenerate spans, now that a rate is computed from one.
    #     (a) An all-burst run has no distribution to report and a perfectly well-defined rate:
    #         199 intervals of 0.1 ms is 10000 flips/s. Pre-fix it printed 'no usable intervals'
    #         and no rate at all, which is the filter deciding the run did not happen.
    all_burst = burst_log(n_long=199, n_burst=0, span_s=0.0199)
    out, code = run(all_burst)
    rate = headline(out)
    if rate is None or abs(rate - 10000.0) > 100.0:
        failures.append(f"case 23a: an all-burst run reported no usable rate ({rate}): {out!r}")
    if "no population" not in out:
        failures.append(f"case 23a: the empty distribution population went unnamed: {out!r}")
    #     (b) ...and a timeline with no extent must refuse rather than divide by its own zero.
    #         NOTE this half passes in BOTH directions -- the pre-fix tool never divided, because
    #         it never computed a span. It is a guard against the fix, not a discriminator, and is
    #         counted as neither.
    out, code = run("[ev] GpuFlip t=1.000000\n" * 5)
    if code != 0 or "Traceback" in out:
        failures.append(f"case 23b: identical timestamps crashed the report (rc={code}): {out!r}")
    if headline(out) is not None:
        failures.append(f"case 23b: a rate was printed for a timeline with no extent: {out!r}")
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
    rows_race = list(rows)
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

    # 25b. #3560 AND #3462 MEET HERE, and the textual merge of them was clean and broken.
    #      #3560 added a "<-- NOT meaningful" marker on the headline rate, keyed on the
    #      `restarts` parameter; #3462 renamed that parameter to `backsteps`. The merge left a
    #      line naming a variable that no longer exists -- a NameError on every input reaching
    #      it -- and no arm noticed, because each branch's own suite passed on its own base.
    #      So the marker is pinned here to the CLASSIFICATION, in BOTH directions:
    #
    #        * a RESTART voids the rate: several runs have no single span to divide by.
    #        * a RACE does NOT: the input is ONE run whose emission interleaved, its span is
    #          real, and voiding it would be a fresh false negative -- the opposite failure
    #          to the one #3560 exists to fix, arrived at from the other side.
    #
    #      Case 25's `out` (a restart) is reused and case 24's rows are re-run, so this arm
    #      cannot drift from the constructions it judges.
    if "Traceback" in out:
        failures.append(f"case 25b: the restart report crashed: {out!r}")
    if "NOT meaningful" not in out:
        failures.append(f"case 25b: a concatenated input's headline rate was printed with no"
                        f" caveat -- there is no single span to divide by (#3560): {out!r}")
    out_race, _ = run("\n".join(rows_race) + "\n")
    if "Traceback" in out_race:
        failures.append(f"case 25b: the race report crashed -- the #3560 marker is reading a"
                        f" parameter #3462 renamed: {out_race!r}")
    if "NOT meaningful" in out_race:
        failures.append(f"case 25b: a 50 us emission race voided the headline rate, but the"
                        f" input is ONE run and its wall-clock span is real (#3462):"
                        f" {out_race!r}")

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

    # ---------------------------------------------------------------- #3581
    # Three of #3562's six behaviour changes were GREEN IN BOTH DIRECTIONS when it merged:
    # reverting them changed the report and reddened nothing, so three `(#3560)` comments went on
    # asserting fixes that no arm held down (#3581). None of the three can produce a wrong number
    # on its own, which is why it was not blocking -- but this file's history is corrections that
    # each carried the previous defect forward in a narrower form (instrument trap 275/277), and
    # an unpinned claim in a tool whose subject is "a number that reads as a measurement and is
    # not one" is the one thing that must not be left lying around here.
    #
    # Cases 27-29 are those three arms, one per unpinned line. Each block names the exact
    # pre-#3562 expression it is measured against, so the without-fix run stays reproducible.
    #
    # A SECOND BUILDER, because `burst_log` above spreads its bursts evenly at a chosen fraction
    # and these arms need the interval VALUES placed deliberately -- a tick-locked phase beside an
    # anti-aligned one, and one long interval at a known ordinal. The caller writes the interval
    # list, so every expected number below is derived from the construction and never read back
    # out of the tool.
    def interval_log(ivals, t0=1.0):
        t = t0
        rows = [f"[ev] GpuFlip t={t:.6f} handle=0x1002 bufidx=0 mode=0x1 fliparg=0x0"]
        for ms in ivals:
            t += ms / 1000.0
            rows.append(f"[ev] GpuFlip t={t:.6f} handle=0x1002 bufidx=0 mode=0x1 fliparg=0x0")
        return "\n".join(rows) + "\n"

    WINDOW_ROW = re.compile(r"flips\s+(\d+)-\s*(\d+):\s+([0-9.]+) fps\s+mean\s+[0-9.]+ ms"
                            r"\s+tick-aligned\s+([0-9.]+)%")
    RETAINED = re.compile(r"\[mean over (\d+)/(\d+)\]")

    def window_rows(text):
        """(start, end, fps, share, retained) per window row.

        The retained count is the bracketed one where anything was dropped and the row's own
        width where nothing was -- the report prints the bracket only when the two differ. Rows
        whose whole population was filtered carry no statistic and are not rows for this purpose.
        """
        found = []
        for line in text.splitlines():
            m = WINDOW_ROW.search(line)
            if not m:
                continue
            r = RETAINED.search(line)
            start, end = int(m.group(1)), int(m.group(2))
            found.append((start, end, float(m.group(3)), float(m.group(4)) / 100.0,
                          int(r.group(1)) if r else end - start))
        return found

    def window_width(text):
        m = re.search(r"windows of ~(\d+) flips", text)
        return None if m is None else int(m.group(1))

    # 27. THE WINDOW LENGTH IS CUT OVER THE RAW TIMELINE:
    #
    #       window_flips = max(1, int(window_s * 1000.0 * len(raw) / span_ms))
    #
    #     The pre-#3562 line was `int(window_s * (1000.0 / statistics.mean(intervals)))`, i.e.
    #     window_s times the rate of the RETAINED set. That rate is the true rate multiplied by
    #     the retained fraction, so every window comes out short by exactly that fraction: on a
    #     run that drops half its intervals to the burst filter, a `--window-s 5` row covers 2.5 s
    #     of wall clock while the header above it says the reader got the period they asked for.
    #     The windowed view exists to locate a phase change in TIME, so a window length that
    #     shrinks with a run's burstiness misplaces the boundary it is read for -- silently,
    #     because nothing in the output is denominated in seconds.
    #
    #     CASE 22 CANNOT SEE THIS, which is why this needs its own arm. Case 22 checks each row's
    #     RATE (its own flips over its own span -- right at any window length) and the last row's
    #     ordinal (the full 1200 whatever the step -- also right at any window length). The cut
    #     can be halved with both still passing, and measured, it is.
    #
    #     Case 20's `dense` run is reused rather than rebuilt, so this arm cannot drift from the
    #     construction it judges: 1200 intervals over exactly 20.000 s, half of them sub-0.5 ms.
    #     At --window-s 5 that is four windows of 300 flips, each covering 5.000 s. Pre-fix it is
    #     eight windows of 150, each covering 2.500 s.
    span_s, n_raw, window_s = 20.0, 1200, 5.0
    want_width = int(window_s * n_raw / span_s)     # 300 flips = 5 s at 60 flips/s
    want_rows = int(span_s / window_s)              # 4 windows over a 20 s run
    out, _ = run(dense, ["--window-s", str(window_s)])
    # The regime, stated rather than assumed: half of this run is burst, which is the whole reason
    # the two forms differ at all. If that ever stops holding, the arm below measures nothing.
    expect(out, "600 of 1200 retained", "case 27 the run is half burst, or this arm is void")
    width = window_width(out)
    if width != want_width:
        failures.append(
            f"case 27: {n_raw} intervals over {span_s:.3f} s is {n_raw / span_s:.0f} flips/s, so a"
            f" {window_s:.0f} s window is {want_width} flips and the report says {width} -- the"
            f" window length is cut over the retained set, so it is short by the retained"
            f" fraction (#3560)")
    rows = window_rows(out)
    if len(rows) != want_rows:
        failures.append(
            f"case 27: a {span_s:.0f} s run at --window-s {window_s:.0f} must split into"
            f" {want_rows} windows and produced {len(rows)} -- the window length is not"
            f" {window_s:.0f} s of wall clock (#3560)")
    for start, end, fps, _share, _retained in rows:
        # The row prints its flips and its rate, so the wall clock it covers is flips/rate -- the
        # quantity --window-s names, and the one nothing else in this file asserts.
        covered = (end - start) / fps
        if abs(covered - window_s) > 0.15:
            failures.append(
                f"case 27: window {start}-{end} reports {end - start} flips at {fps} fps, so it"
                f" covers {covered:.3f} s of wall clock where --window-s asked for"
                f" {window_s:.3f} s (#3560)")

    # 28. THE WINDOW-VOTE BAR IS A QUARTER OF THE LARGEST WINDOW'S RETAINED COUNT:
    #
    #       cap = max((n for _, n in window_sizes), default=0)
    #       full = [share for share, n in window_sizes if n >= max(8, cap // 4)]
    #
    #     The pre-#3562 predicate was `n >= max(8, window_flips // 4)`. `n` is a RETAINED count
    #     and `window_flips` is a RAW one, so as soon as the retained fraction falls below a
    #     quarter the bar sits above every window at once: `full` comes out empty, and the
    #     "windows disagree ... read the windows, not the summary" note is silenced ENTIRELY.
    #     That fails in the worse direction twice over -- the note disappears on the burstiest
    #     runs, which are where a phase change is likeliest to be hiding, and its absence reads
    #     as windows that agree rather than as a threshold none of them could clear.
    #
    #     Two equal phases, one locked to the tick and one parked on the half-tick offset, each
    #     600 real intervals with five 0.2 ms bursts behind every one -- a 16.7% retained
    #     fraction, comfortably under the quarter where the two bars cross. --window-s is half the
    #     constructed span, so the cut lands on the seam and the two windows ARE the two phases:
    #     100.0% against 0.0% tick-aligned, 600 retained of 3600 raw each. The correct bar is
    #     max(8, 600 // 4) = 150 and both vote; the pre-fix bar is max(8, 3600 // 4) = 900 and
    #     neither does.
    bursts_per_long, groups = 5, 600
    ivals = []
    for long_ms in (FPR.TICK_MS, FPR.TICK_MS * 1.5):
        for _ in range(groups):
            ivals.append(long_ms)
            ivals.extend([0.2] * bursts_per_long)
    out, _ = run(interval_log(ivals), ["--window-s", f"{sum(ivals) / 2000.0:.6f}"])
    width = window_width(out)
    rows = window_rows(out)
    voters = [r for r in rows if r[4] >= 8]
    # THE PRECONDITIONS, asserted apart from the conclusion so a construction that drifts out of
    # the regime fails under its own name instead of quietly making the arm pass. The regime is:
    # at least two windows large enough to vote, disagreeing by more than the note's 25-point
    # threshold, every one of them below a quarter of the RAW window width.
    if width is None or len(voters) < 2:
        failures.append(f"case 28: expected at least two windows with a votable retained"
                        f" population, got {rows!r}: {out!r}")
    else:
        shares = [r[3] for r in voters]
        over = [r[4] for r in voters if r[4] >= width // 4]
        if max(shares) - min(shares) < 0.25:
            failures.append(f"case 28: the two phases were built to disagree by 100 points and the"
                            f" windows report {shares!r} -- this run is no longer the regime the"
                            f" arm is about")
        elif over:
            failures.append(f"case 28: a window retained {over!r} intervals of a {width}-flip raw"
                            f" window and so clears the pre-fix bar of {width // 4} on its own,"
                            f" which means this run cannot tell the two bars apart")
        elif "windows disagree" not in out:
            failures.append(
                f"case 28: two windows at {100 * min(shares):.1f}% and {100 * max(shares):.1f}%"
                f" tick-aligned printed no disagreement note, because each retained"
                f" {voters[0][4]} intervals and the bar was a quarter of the {width}-flip RAW"
                f" window rather than of the largest window's retained count -- so the note is"
                f" silenced exactly on the bursty runs a phase change is likeliest to hide in"
                f" (#3560): {out!r}")

    # 28b. ...and the 8-interval FLOOR must survive, or case 28 is equally satisfied by dropping
    #      the bar altogether, which would reintroduce the defect the bar was added for (#3454
    #      review): a trailing sliver's share is quantised into huge steps -- 3 intervals can only
    #      score 0/33/67/100% -- so comparing one against a full window manufactures a
    #      disagreement out of the quantisation. This is a CONTROL, not a discriminator for
    #      #3581: it passes in both directions of all three mutations above, and is counted as
    #      neither. It discriminates only the floor.
    #
    #      The tail has to land in the gap BETWEEN the two terms or the control is void, which is
    #      the first thing asserted below: three windows of 20 anti-aligned intervals (0% each,
    #      so cap // 4 = 5) and a 6-interval tail of tick-locked ones (100%). Six clears 5 and is
    #      under 8, so the floor is the only term excluding it -- with the floor the three full
    #      windows agree and nothing prints, without it the tail votes and 0%..100% does. A
    #      3-interval tail would be excluded by `cap // 4` alone and would assert nothing.
    #
    #      The window length carries a deliberate +0.4 flip: the report TRUNCATES window_flips, so
    #      an exactly-20.0 request can land on 19 through float error and put three anti-aligned
    #      intervals into the tail, which is a different regime rather than a failure.
    ivals = [FPR.TICK_MS * 1.5] * 60 + [FPR.TICK_MS] * 6
    window_s = (20 + 0.4) * (sum(ivals) / 1000.0) / len(ivals)
    out, _ = run(interval_log(ivals), ["--window-s", f"{window_s:.6f}"])
    rows = window_rows(out)
    slivers = [r for r in rows if r[4] < 8]
    full_rows = [r[3] for r in rows if r[4] >= 8]
    if not slivers or not full_rows:
        failures.append(f"case 28b: this run was built to leave a sub-8-interval tail beside full"
                        f" windows and produced {rows!r}, so the control is void: {out!r}")
    elif max(r[3] for r in slivers) - min(full_rows) < 0.25:
        failures.append(f"case 28b: the tail scores {[r[3] for r in slivers]!r} against full"
                        f" windows at {full_rows!r}, so there is no manufactured disagreement for"
                        f" the floor to suppress and the control claims nothing")
    elif "windows disagree" in out:
        failures.append(f"case 28b: a {slivers[-1][1] - slivers[-1][0]}-interval tail voted"
                        f" against full windows that agree with each other, which is the"
                        f" quantisation the 8-interval floor exists to keep out (#3454 review):"
                        f" {out!r}")

    # 29. THE SLOWEST-INTERVAL ORDINALS ARE OVER THE RAW LIST:
    #
    #       slowest = sorted(range(len(raw)), key=lambda i: -raw[i])[:5]
    #       ... f"{raw[i]:.1f} @ flip {i}"
    #
    #     The pre-#3562 pair indexed `intervals`, the filtered list, under a label reading "flip".
    #     This line's whole job is to point a reader back into the log at the moment the run
    #     stalled, and a filtered index is off by every burst that preceded it -- so the tool
    #     names a flip that is not the one it measured, and the reader goes and looks at it. It is
    #     the only output here that is a POINTER rather than a statistic, which is why being
    #     quietly wrong costs more than a wrong number would.
    #
    #     The issue's worked example, constructed exactly: 112 dropped bursts interleaved among
    #     the first 300 real intervals, then one 500 ms stall. Its raw ordinal is 412 and its
    #     index in the filtered list is 300, so the two forms print `@ flip 412` and `@ flip 300`
    #     for the same interval.
    normal_ms, burst_ms, stall_ms = 16.0, 0.2, 500.0
    ivals = []
    for _ in range(112):
        ivals.extend([normal_ms, burst_ms])
    ivals.extend([normal_ms] * 188)
    stall_at = len(ivals)
    ivals.append(stall_ms)
    ivals.extend([normal_ms] * 100)
    filtered_at = sum(1 for ms in ivals[:stall_at] if ms > 0.5)
    if (stall_at, filtered_at) != (412, 300):
        failures.append(f"case 29: the construction drifted -- the stall sits at raw ordinal"
                        f" {stall_at} and filtered index {filtered_at}, expected 412 and 300")
    out, _ = run(interval_log(ivals))
    named = re.findall(r"([0-9.]+) @ flip (\d+)", out)
    if not named:
        failures.append(f"case 29: no slowest-interval line at all: {out!r}")
    # THE GENERAL INVARIANT first: whatever the tool names, `flip i` must index the flip timeline,
    # so raw[i] has to be the value printed beside it. That catches the defect on all five entries
    # rather than only on the stall, and no filtered index can satisfy it once anything is dropped.
    for value, ordinal in named:
        i = int(ordinal)
        if i >= len(ivals) or abs(ivals[i] - float(value)) > 0.06:
            failures.append(
                f"case 29: the report names {value} ms '@ flip {ordinal}', but flip {ordinal} of"
                f" this run is {ivals[i] if i < len(ivals) else 'past the end'} ms -- the ordinals"
                f" index the filtered list while the label says flip, so they point at the wrong"
                f" line of the log (#3560)")
    # ...and the stall itself, named with both numbers, because this is the one a reader follows.
    stall = [int(o) for v, o in named if abs(float(v) - stall_ms) < 0.06]
    if not stall:
        failures.append(f"case 29: the {stall_ms:.0f} ms stall is not among the slowest intervals"
                        f" at all: {out!r}")
    elif stall[0] != stall_at:
        failures.append(
            f"case 29: the {stall_ms:.0f} ms stall is flip {stall_at} and the report says flip"
            f" {stall[0]}, which is its index in the filtered list after the"
            f" {stall_at - filtered_at} earlier bursts were dropped -- a pointer into the log that"
            f" does not point at the thing (#3560)")

    # 30. THE API FLIP PATH (#3564). `sceVideoOutSubmitFlip` advances the flip state exactly as
    #     the in-stream packet does, and this parser read only `[ev] GpuFlip` -- so a title
    #     flipping through the API reported "fewer than 2 flips -- nothing to pace" about a
    #     paceable run. The positive instance is built HERE, not by the emitter the null came
    #     from: 121 SubmitFlip lines at exactly 60 Hz, in the emitter's field order.
    api = "".join(
        f"[ev] SubmitFlip t={1.0 + i / 60.0:.6f} handle=0x1001 bufidx={i % 2} flipmode=0x1"
        f" fl013arg=0x{i:x}\n" for i in range(121))
    out, _ = run(api)
    if "fewer than 2 flips" in out or "cannot be paced" in out:
        failures.append(f"case 30: a SubmitFlip-only run was reported as unpaceable: {out!r}")
    m = re.search(r"flips=(\d+) over [0-9.]+ s -> ([0-9.]+) fps", out)
    if not m or int(m.group(1)) != 121 or abs(float(m.group(2)) - 60.0) > 0.5:
        failures.append(f"case 30: expected 121 flips at 60.0 fps from the API path, got: {out!r}")
    if "SubmitFlip=121" not in out:
        failures.append(f"case 30: the report does not say the timeline came from SubmitFlip: {out!r}")
    #     Mixed, in the shape the corpus actually holds: two API blank flips at boot, then GPU
    #     flips. These five lines are verbatim prosper stderr (PROSPER_EVLOG=1, PPSA05684), the
    #     domain arm for the new tag the way case 5 is for GpuFlip.
    mixed_real = (
        "[ev] SubmitFlip t=0.000000 handle=0x1001 bufidx=-1 flipmode=0x1 fl013arg=0xffffffffffffffff\n"
        "[ev] SubmitFlip t=0.005005 handle=0x1001 bufidx=-1 flipmode=0x1 fl013arg=0x0\n"
        "[ev] GpuFlip t=0.129640 handle=0x1001 bufidx=0 mode=0x1 fliparg=0x0\n"
        "[ev] GpuFlip t=0.183581 handle=0x1001 bufidx=1 mode=0x1 fliparg=0x1\n"
        "[ev] GpuFlip t=0.202183 handle=0x1001 bufidx=2 mode=0x1 fliparg=0x2\n")
    out, _ = run(mixed_real)
    if "flips=5 " not in out:
        failures.append(f"case 30: a mixed run must pool both emitters into 5 flips: {out!r}")
    if "GpuFlip=3 SubmitFlip=2" not in out:
        failures.append(f"case 30: mixed run must name both emitter counts: {out!r}")
    #     A SubmitFlip the emulator REJECTED (bufidx outside [-1, 15]) is printed before the
    #     validation and never flips, so it must not become an interval.
    #     Both edges of the emulator's accepted range [-1, 15] are flips; 16 is not.
    edges = ("[ev] SubmitFlip t=1.000000 handle=0x1001 bufidx=-1 flipmode=0x1 fl013arg=0x0\n"
             "[ev] SubmitFlip t=1.016000 handle=0x1001 bufidx=15 flipmode=0x1 fl013arg=0x1\n")
    out, _ = run(edges)
    if "flips=2 " not in out or "rejected" in out:
        failures.append(f"case 30: bufidx -1 and 15 must both be accepted flips: {out!r}")
    rejected = api + "[ev] SubmitFlip t=9.000000 handle=0x1001 bufidx=16 flipmode=0x1 fl013arg=0x0\n"
    out, _ = run(rejected)
    if "flips=121 " not in out or "1 SubmitFlip call(s) with an out-of-range bufidx" not in out:
        failures.append(f"case 30: a rejected SubmitFlip was counted as a flip or not named: {out!r}")
    #     A GpuFlip-only log must print exactly what it printed before: no emitter line.
    out, _ = run(real)
    if "flip emitters:" in out:
        failures.append(f"case 30: a GpuFlip-only log grew an emitter line: {out!r}")

    if failures:
        print("FAILURES:")
        for failure in failures:
            print(" -", failure)
        return 1
    print("flip_pacing_report self-test: all cases pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
