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

    # 0b. The (grain=N) field must actually be USED, and this arm exists because review showed the
    #     headline change of this commit had no arm that would redden if reverted: every other fixture
    #     logs a grain equal to frames * bpf * channels, so honouring the field and inferring it give
    #     identical answers, and deleting the parse left the whole suite green.
    #
    #     A first attempt used an atypical FIRST record. That stopped discriminating the moment the
    #     inference fallback became a median, which is robust to exactly that -- good for the tool,
    #     vacuous for the arm. So this uses the case review actually named: an s16 title read with the
    #     default f32 flags. Every record is 256 frames and the emitter logs the truth,
    #     256 * 2 * 2 = 1024 B, while inferring with bpf=4 gives 2048 B. Queues sit at 1500 B -- ABOVE
    #     the real grain, BELOW the inferred one -- so honouring the field reports 0 thin-cushion
    #     arrivals and inferring reports 200. Note the direction: inferring invents a cushion problem
    #     that is not there.
    log = "".join(dbg(frames=256, queued=1500, bpf=2, channels=2) for _ in range(200))
    out, _ = run(log)
    expect(out, "arrivals with under 1 grain buffered: 0",
           "case 0b: the logged grain is honoured over the flag-derived inference")
    expect(out, "grain=256 frames",
           "case 0b: and the header reports the same grain the rows used")

    # 0c. The cadence grain is the MEDIAN frame count, not the first record's.
    #
    #     One atypical first arrival otherwise rescales every gap-beyond-N-grains row. Here the first
    #     record is 64 frames (1.33 ms at 48 kHz) and the rest are 256 (5.33 ms), with gaps at a
    #     steady 5.33 ms. Taking frames[0] makes the reference grain 1.33 ms, so every one of those
    #     normal gaps counts as "beyond 2 grains" and the verdict flips to a quantized mixer wake --
    #     review measured exactly that, 100.0%, on a healthy log. The median reads 256 and the rows
    #     stay at 0.
    log = dbg(frames=64, gap=5.33, queued=3072) + "".join(
        dbg(frames=256, gap=5.33, queued=3072) for _ in range(200))
    out, _ = run(log)
    expect(out, "gaps beyond 2 grains:             0",
           "case 0c: an atypical first record does not rescale the cadence rows")
    expect(out, "grain=256 frames", "case 0c: the header reports the typical grain")

    # 0d. The LEGACY path -- a log with no (grain=N) field -- must still compute a grain from the
    #     flags. Review found that deleting the inference branch left the whole suite green, which is
    #     the same shape as R1 one level down, and on the path the --bytes-per-frame help text points
    #     users at. So: an old-format log, and the thresholds must still be grain-relative.
    log = "".join(
        "[audio-dbg] port=17 gap=5.33ms frames=256 queued_before=1024\n" for _ in range(200))
    out, _ = run(log)
    expect(out, "grain=256 frames", "case 0d: a log without the grain field still reports a grain")
    expect(out, "arrivals with under 1 grain buffered: 200",
           "case 0d: and the inference is grain-relative, not frame-relative")

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

    # 6/7/8 (#3080). The bottom "mixer's wake is quantized" verdict fires on clustered arrivals
    # regardless of WHY they cluster, and case 3 above never reaches it (its 9.66 ms gap sits
    # under the 10.67 ms two-grain threshold). This is a reconstruction of the actual Blasphemous 2
    # numbers from #3080: a steady ~11.85 ms gap against a 5.33 ms grain -- 100% of arrivals beyond
    # two grains -- driven by a guest producing audio at ~45% of real time (drift ~-55%, matching
    # the issue's -55.29%). Before the fix this printed "the mixer's wake is quantized ... Fix the
    # wait primitive's resolution" unconditionally, which is the exact wrong verdict #3080 reports:
    # the title's ONLY timed wait (usleep) measured x1.02 via PROSPER_TIMEDWAIT_CENSUS, i.e. not
    # quantized, and the clock deficit already explains the clustering.
    deficit_log = "".join(dbg(gap=11.85, queued=64) for _ in range(488))

    # 6. No census in the log: the verdict must hedge rather than assert a mechanism it has no
    #    evidence for, and must name the census as the way to settle it.
    out, _ = run(deficit_log)
    expect(out, "cannot tell them apart", "case 6: no census -> hedged, not asserted")
    expect(out, "PROSPER_TIMEDWAIT_CENSUS=1", "case 6: names the discriminator")
    reject(out, "Fix the wait primitive's resolution",
           "case 6: must not issue the imperative with no evidence for it")

    # 7. A census IS present and shows the guest's only wait primitive resolving close to exact
    #    (x1.02, the #3080 figure) -- the verdict must say quantization is RULED OUT, not assert it.
    census_low = ("[timedwait 5s] usleep           calls=488       requested= 20.143 ms"
                  "  actual= 20.503 ms  x1.02\n")
    out, _ = run(deficit_log + census_low)
    expect(out, "RULES OUT a quantized wake", "case 7: census rules out quantization")
    expect(out, "x1.02", "case 7: reports the measured ratio")
    reject(out, "Fix the wait primitive's resolution",
           "case 7: must not issue the imperative when the census contradicts it")

    # 8. A census IS present and shows a genuinely coarse wait (x2.90, #3013's figure for the
    #    primitive that really was the cause elsewhere) -- here the verdict is earned, and this is
    #    the one case where "fix the wait primitive" should still appear.
    census_high = ("[timedwait 5s] sem_timedwait    calls=488       requested=  5.330 ms"
                   "  actual= 15.457 ms  x2.90\n")
    out, _ = run(deficit_log + census_high)
    expect(out, "CONFIRMS a coarse wait", "case 8: census confirms quantization")
    expect(out, "x2.90", "case 8: reports the measured ratio")
    expect(out, "Fix the wait primitive's resolution",
           "case 8: the imperative IS earned when the census supports it")

    # 9. Drift bias hand-check (#3061). The `[audio-dbg]` emitter logs gap=0.00ms for a port's
    #    FIRST arrival -- no previous arrival to measure from -- so `span_s = sum(gaps)` covers
    #    only the REAL inter-arrival intervals while `frames_total` sums frames from every
    #    arrival, including that first, unmeasured one. `frames_total / span_s` overcounts by one
    #    arrival's worth of frames.
    #
    #    Chosen so the TRUE rate is exact and the bias is exact too: 11 records of 100 frames at
    #    a 1000 Hz device, delivered at EXACTLY real-time cadence (100.00 ms per 100-frame grain
    #    == 100 frames/s * 1000 == 1000 Hz). By construction the true delivered rate is exactly
    #    the device rate:
    #
    #      true_drift = (delivered_hz - device_hz) / device_hz
    #                 = (frames_after_first / real_span_s - device_hz) / device_hz
    #                 = ((10 * 100) / 1.0 - 1000) / 1000 = 0 / 1000 = 0.00%
    #
    #    The pre-fix formula divides ALL 11 records' frames by the same 1.0 s real span:
    #
    #      biased_hz   = frames_total / span_s = (11 * 100) / 1.0 = 1100
    #      biased_drift = (1100 - 1000) / 1000 = +10.00%    (== calls / (calls - 1) - 1 = 1/10)
    #
    #    +10.00% is comfortably past the +-0.3% threshold, so the bug does not just misreport a
    #    number -- it flips the verdict to the false "runs above the device rate" the issue names.
    log = dbg(gap=0.00, frames=100, queued=100000, bpf=1, channels=1)
    log += "".join(
        dbg(gap=100.00, frames=100, queued=100000, bpf=1, channels=1) for _ in range(10))
    out, _ = run(log, ["--device-hz", "1000"])
    expect(out, "-> drift +0.00%",
           "case 9: an exact real-time 11-record log must report the true 0.00% drift, not +10.00%")
    expect(out, "delivery matches the device rate",
           "case 9: the true rate must read as matching, not as a clock excess")
    reject(out, "runs above the device rate",
           "case 9: the pre-fix +1/(N-1) bias (+10.00% at N=11) must not survive")

    if failures:
        print("FAILURES:")
        for failure in failures:
            print(" -", failure)
        return 1
    print("audio_delivery_report self-test: all cases pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
