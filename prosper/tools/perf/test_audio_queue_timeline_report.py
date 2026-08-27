#!/usr/bin/env python3
"""Self-test for audio_queue_timeline_report.py.

Each arm is written so it FAILS if the property under test is broken -- checked by mutation, not
by inspection. The properties that matter are the ones a mean cannot see, because the whole
reason this tool exists is that the one-second delivery average stayed at 100% of real time while
the queue emptied in every gap (#3016).
"""
import io
import re
import subprocess
import sys
import tempfile
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "audio_queue_timeline_report.py")

fails = 0


def run(log, *extra):
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(log)
        path = fh.name
    try:
        p = subprocess.run([sys.executable, TOOL, path, *extra],
                           capture_output=True, text=True, encoding="utf-8")
        return p.stdout + p.stderr, p.returncode
    finally:
        os.unlink(path)


def expect(out, needle, what):
    global fails
    if needle not in out:
        fails += 1
        print(f" [FAIL] {what}: expected {needle!r} in the report")
        print("        got:\n" + "\n".join("        " + l for l in out.splitlines()))


def expect_rc(rc, want, what):
    """Exit-code assertion that actually COUNTS.

    Several arms in this file printed a [FAIL] line without touching `fails`, so the suite printed
    the failure and then exited 0 -- and ctest reads only the exit code. Review of #3070 measured
    it: with strict argument parsing removed from the tool entirely, the self-test printed both
    FAIL lines and still reported all cases pass. A helper rather than scattered `fails += 1`
    lines, so the next exit-code arm cannot reintroduce it.
    """
    global fails
    if rc != want:
        fails += 1
        print(f" [FAIL] {what}: expected exit {want}, got {rc}")


def expect_not(out, needle, what):
    global fails
    if needle in out:
        fails += 1
        print(f" [FAIL] {what}: did NOT expect {needle!r} in the report")
        print("        got:\n" + "\n".join("        " + l for l in out.splitlines()))


def sample(t_us, queued, port=17, grain=2048):
    return f"INFO: [audio-queue] t_us={t_us} port={port} queued={queued} grain={grain}\n"


def main():
    # 1. A healthy run: fed continuously, queue never empties.
    log = "".join(sample(i * 1000, 6144) for i in range(2000))
    out, rc = run(log)
    expect(out, "= 3.00 grains", "healthy: the cushion is reported in grains")
    expect(out, "the queue never emptied", "healthy: no starvation claimed")
    expect(out, "UNDERRUN (dry while streaming): 0 episodes", "healthy: zero underruns")
    expect_rc(rc, 0, "healthy: a clean log exits 0")

    # 2. THE CONFOUND ARM, and the reason the gate exists. A port that is OPEN but never fed reads
    #    queued=0 for its whole life. Before the gate this tool called that STARVED, and on a real
    #    600 s run it reported 53% of samples dry with a 40 SECOND "episode" -- which would have
    #    decided a pacer A/B on whichever arm happened to idle more.
    log = "".join(sample(i * 1000, 0) for i in range(2000))
    out, _ = run(log)
    expect(out, "NOT COUNTED (idle, or unobservable): 2000 samples",
           "confound: a never-fed port is IDLE for every sample")
    expect(out, "UNDERRUN (dry while streaming): 0 episodes",
           "confound: ...and reports ZERO underruns, not 2000")
    expect_not(out, "STARVED", "confound: a silent port is never called STARVED")
    # 3. Idle time is REPORTED, not silently dropped. A filter that hides what it removed reports
    #    an improvement that is really a threshold.
    expect(out, "100.00% of samples", "confound: the idle fraction is stated")

    # 4. THE ARM THIS TOOL EXISTS FOR. One contiguous dry burst of 8 samples, with the port fed on
    #    both sides, is ONE underrun episode of 8 ms -- not 8 episodes, and not idle.
    log = "".join(sample(i * 1000, 0 if 500 <= i < 508 else 6144) for i in range(2000))
    out, _ = run(log)
    expect(out, "UNDERRUN (dry while streaming): 1 episodes",
           "burst: 8 contiguous dry samples bracketed by audio are ONE underrun")
    expect(out, "8.0 ms total", "burst: charged one sampling interval per dry sample")
    expect(out, "longest 8000 us at t=500000 us", "burst: the longest episode is located")
    expect(out, "NOT COUNTED (idle, or unobservable): 0 samples",
           "burst: a bracketed dip is not idle")

    # 5. A LONG silence BETWEEN two fed stretches -- bracketed by audio on both sides, and still
    #    not an underrun. This is the arm that shows bracketing alone is insufficient: a 5 s gap
    #    between two songs satisfies "audio before" and "audio after" and would be reported as one
    #    enormous underrun without the duration condition.
    log = "".join(sample(i * 1000, 0 if 200 <= i < 5200 else 6144) for i in range(5400))
    out, _ = run(log)
    expect(out, "UNDERRUN (dry while streaming): 0 episodes",
           "long silence: a 5 s bracketed gap is not an underrun")
    expect(out, "NOT COUNTED (idle, or unobservable): 5000 samples",
           "long silence: ...it is idle, all 5000 samples of it")
    expect(out, "no underrun -- every dry stretch was idle",
           "long silence: the verdict says no underrun, not \"never emptied\"")

    # 6. --active-window-ms is the max-dip boundary, and must be shown to bite in BOTH directions
    #    on the SAME input. An 8 ms dip is an underrun at the 200 ms default and idleness at 2 ms.
    #    A knob that cannot be shown to move the answer is a knob whose default nobody can justify.
    log8 = "".join(sample(i * 1000, 0 if 500 <= i < 508 else 6144) for i in range(2000))
    out_big, _ = run(log8)
    expect(out_big, "UNDERRUN (dry while streaming): 1 episodes",
           "max dip: an 8 ms dip counts at the 200 ms default")
    out_small, _ = run(log8, "--active-window-ms=2")
    expect(out_small, "UNDERRUN (dry while streaming): 0 episodes",
           "max dip: the same 8 ms dip is idleness at a 2 ms bound")
    expect(out_small, "NOT COUNTED (idle, or unobservable): 8 samples",
           "max dip: ...and the eight samples are accounted for, not dropped")
    expect(out_small, "max dip 2 ms", "max dip: the bound in force is stated in the output")

    # 6b. LEADING dry: the port is open and dry BEFORE its first audio ever arrives. That is
    #     start-up, not starvation, and it is short enough to pass the duration bound -- so the
    #     "audio was seen before it" condition is the only thing that classifies it. Added because
    #     mutating that condition away reddened nothing: the gate had an untested third of itself.
    log_lead = ("".join(sample(i * 1000, 0) for i in range(60)) +
                "".join(sample((60 + i) * 1000, 6144) for i in range(500)))
    out, _ = run(log_lead)
    expect(out, "UNDERRUN (dry while streaming): 0 episodes",
           "leading dry: a port dry before its first audio is not an underrun")
    expect(out, "NOT COUNTED (idle, or unobservable): 60 samples",
           "leading dry: ...it is start-up idleness")

    # 6c. TRAILING dry: the port stops being fed and stays dry to the end of the log. Symmetric to
    #     6b, and it must be the "audio seen after it" condition doing the work -- again short
    #     enough that the duration bound cannot be what excludes it.
    log_tail = ("".join(sample(i * 1000, 6144) for i in range(500)) +
                "".join(sample((500 + i) * 1000, 0) for i in range(60)))
    out, _ = run(log_tail)
    expect(out, "UNDERRUN (dry while streaming): 0 episodes",
           "trailing dry: a port that stops feeding is not underrunning")
    expect(out, "NOT COUNTED (idle, or unobservable): 60 samples",
           "trailing dry: ...it is shutdown idleness")

    # 6d. THE WELDING CASE, and the reason episodes() takes a gap threshold. The sampler emits
    #     nothing for a closed port, and a port is normally dry on BOTH sides of its own lifecycle
    #     boundary. Without a discontinuity split these two 20-sample dry stretches, 30 s apart in
    #     wall clock, merge into one "underrun" of 40 ms that is bracketed by audio and passes the
    #     duration bound -- so 6b's start-up idleness and 6c's shutdown idleness, which those arms
    #     verify at the two ENDS of the log, are defeated at every interior open/close. Constructed
    #     by review of #3070, which measured exactly this: 1 episode, 40 ms, 0 idle.
    log_weld = ("".join(sample(i * 1000, 6144) for i in range(100)) +
                "".join(sample((100 + i) * 1000, 0) for i in range(20)) +
                "".join(sample(30_120_000 + i * 1000, 0) for i in range(20)) +
                "".join(sample(30_140_000 + i * 1000, 6144) for i in range(100)))
    out, _ = run(log_weld)
    expect(out, "UNDERRUN (dry while streaming): 0 episodes",
           "welding: two dry stretches across a sampling gap are NOT one underrun")
    expect(out, "NOT COUNTED (idle, or unobservable): 40 samples",
           "welding: both stretches are accounted for as unobservable, not dropped")

    # 6e. And the gap must be VISIBLE, not merely handled. Coverage -- samples x interval against
    #     the wall-clock span -- is what turns a silent hole into a number a reader can act on.
    #     Without it the report above would look like a clean 30 s window.
    expect(out, "coverage 1%", "welding: the sampling gap shows as a coverage shortfall")
    out_full, _ = run("".join(sample(i * 1000, 6144) for i in range(2000)))
    expect(out_full, "coverage 100%", "coverage reads ~100% on a gapless run")

    # 6f. The split must fire on a DISCONTINUITY, not on ordinary jitter. A single missed sample
    #     inside a dry stretch is still one underrun -- a threshold that split on any irregularity
    #     would inflate the episode count, which is the quantity an A/B compares.
    # Samples at 0..99 ms fed, 100..104 dry, a 3 ms hole (under the 4 ms threshold), 107..111
    # dry, then fed again at 112 ms with NO further hole -- the first draft of this fixture
    # left a second 9 ms hole before the audio resumed, which legitimately split the episode
    # and made the arm fail for a reason that had nothing to do with what it tests.
    log_jitter = ("".join(sample(i * 1000, 6144) for i in range(100)) +
                  "".join(sample((100 + i) * 1000, 0) for i in range(5)) +
                  "".join(sample(107_000 + i * 1000, 0) for i in range(5)) +
                  "".join(sample(112_000 + i * 1000, 6144) for i in range(100)))
    out_j, _ = run(log_jitter)
    expect(out_j, "UNDERRUN (dry while streaming): 1 episodes",
           "jitter: a 2 ms hiccup inside a dry stretch does not split the episode")

    # 6g. Argument parsing is STRICT, and both spellings work. The usage line documented
    #     `--min-episode-us N` while main() read only the `=N` form, so the documented spelling
    #     was accepted and silently ignored -- and an unknown `--flag` was dropped without a word.
    #     The emitter this reads from states the opposite rule for its own variables (a malformed
    #     value disables the trigger rather than firing at an unintended moment).
    log_sp = "".join(sample(i * 1000, 0 if 500 <= i < 508 else 6144) for i in range(2000))
    out_sp, rc_sp = run(log_sp, "--active-window-ms", "2")
    expect(out_sp, "max dip 2 ms", "args: the SPACE-separated form is honoured, not ignored")
    expect(out_sp, "UNDERRUN (dry while streaming): 0 episodes",
           "args: ...and it actually changes the answer")
    out_eq, _ = run(log_sp, "--active-window-ms=2")
    expect(out_eq, "max dip 2 ms", "args: the = form still works")
    _, rc_bad = run(log_sp, "--activ-window-ms=2")
    expect_rc(rc_bad, 2,
              "args: an unknown option must exit non-zero, else a typo runs the default arm")
    _, rc_nonint = run(log_sp, "--active-window-ms", "soon")
    expect_rc(rc_nonint, 2, "args: a non-integer value must exit non-zero")

    # 6h. The interval is the MEDIAN, not the mean, and this arm exists because the fix above
    #      raised its stakes: `iv` now sets the segmentation threshold (gap_us = iv * 4), so a
    #      mean would change the episode COUNT and not merely the reported milliseconds. No other
    #      arm feeds a non-uniform interval. Review of #3070 asked for exactly this one.
    #
    #      1 ms for 400 samples, then one 300 ms stall, then 1 ms again. Median = 1000 us; mean is
    #      dragged to ~1374 us, which is above gap_us/4 and would resegment the series.
    log_mm = ("".join(sample(i * 1000, 6144) for i in range(400)) +
              sample(700_000, 6144) +
              "".join(sample(701_000 + i * 1000, 6144) for i in range(400)))
    out_mm, _ = run(log_mm)
    expect(out_mm, "sampling interval 1000 us",
           "interval: one long stall does not drag the interval off the median")

    # 6h-bis. The estimator arm above pins the MEDIAN; this pins the CONSEQUENCE its comment
    #          claims -- that a mean would resegment the series. A dry stretch broken by one ~5 ms
    #          hole discriminates: 5000 us is above 4x1000 (the median threshold, so it splits and
    #          the episode is not counted) and below 4x1376 (a mean threshold, which would NOT
    #          split and would report one underrun). Review of #3070 supplied the construction.
    log_reseg = ("".join(sample(i * 1000, 6144) for i in range(400)) +
                 sample(700_000, 6144) +                              # the stall that moves a mean
                 "".join(sample(701_000 + i * 1000, 6144) for i in range(100)) +
                 "".join(sample(801_000 + i * 1000, 0) for i in range(5)) +
                 "".join(sample(811_000 + i * 1000, 0) for i in range(5)) +   # 5 ms hole
                 "".join(sample(816_000 + i * 1000, 6144) for i in range(100)))
    out_rs, _ = run(log_reseg)
    expect(out_rs, "UNDERRUN (dry while streaming): 0 episodes",
           "resegmentation: a 5 ms hole splits under the median threshold, so neither half counts")

    # 6i. grain=0. The emitter can log it before a port's format is known, and the header divides
    #      by it to report the cushion in grains. The branch must survive and must still measure
    #      dryness, which is independent of the grain.
    log_g0 = "".join(f"[audio-queue] t_us={i * 1000} port=17 queued=0 grain=0\n" for i in range(300))
    out_g0, rc_g0 = run(log_g0)
    expect(out_g0, "grain UNKNOWN (0)", "grain 0: reported as unknown rather than dividing by it")
    expect_rc(rc_g0, 0, "grain 0: still produces a report")
    expect(out_g0, "NOT COUNTED", "grain 0: dryness is still classified without a grain")

    # 6j. Coverage counts INTERVALS, not samples: n samples span (n-1) intervals. A two-sample
    #      port read 200% before this was fixed, which is the kind of figure that discredits the
    #      whole header.
    out_cov, _ = run(sample(0, 6144) + sample(1000, 6144))
    expect(out_cov, "coverage 100%", "coverage: two samples one interval apart read 100%, not 200%")

    # 6k. --active-window-ms 0 would classify every dry stretch as idle -- i.e. silently disable
    #      the thing the tool measures -- so it is refused rather than echoed back.
    _, rc_zero = run("".join(sample(i * 1000, 0) for i in range(10)), "--active-window-ms", "0")
    expect_rc(rc_zero, 2, "a zero active window is refused, not silently accepted")

    # 7. Separated bursts are separate episodes -- the count is what an A/B compares.
    log = "".join(sample(i * 1000, 0 if i in (100, 101, 300, 700, 701, 702) else 6144)
                  for i in range(2000))
    out, _ = run(log)
    expect(out, "UNDERRUN (dry while streaming): 3 episodes",
           "three separated bursts are three underruns")
    # Pin the STARVED threshold from BOTH sides -- a one-sided assertion cannot tell a correct
    # threshold from one that fires always or never.
    expect_not(out, "STARVED", "three underruns is below the STARVED threshold")
    expect(out, "3 isolated underruns", "...and is reported as isolated instead")
    log5 = "".join(sample(i * 1000, 0 if i in (100, 300, 500, 700, 900) else 6144)
                   for i in range(2000))
    out5, _ = run(log5)
    expect(out5, "STARVED -- 5 underruns", "five underruns crosses into STARVED")

    # 8. --min-episode-us drops only the short ones and must SAY how many it dropped.
    out, _ = run(log, "--min-episode-us=2000")
    expect(out, "UNDERRUN (dry while streaming): 2 episodes",
           "filter keeps the two multi-sample episodes")
    expect(out, "1 shorter than 2000 us", "filter reports what it dropped")

    # 9. The sampling interval is MEASURED, not assumed. A log sampled at 15.6 ms -- the #3013 trap
    #    the sampler's own comment records -- must scale durations by 15600, not by 1000.
    log = "".join(sample(i * 15600, 0 if 50 <= i < 54 else 6144) for i in range(200))
    out, _ = run(log)
    expect(out, "sampling interval 15600 us", "the interval is measured from the timestamps")
    expect(out, "62.4 ms total", "episode duration uses the MEASURED interval, not a nominal 1 ms")

    # 10. THIN (under one grain) is a weaker condition than dry, left UNGATED on purpose, and must
    #     not be conflated with it: half a grain has not emptied.
    log = "".join(sample(i * 1000, 1024) for i in range(500))
    out, _ = run(log)
    expect(out, "UNDERRUN (dry while streaming): 0 episodes", "half a grain is not an underrun")
    expect(out, "THIN (under one grain, ungated): 1 episodes", "...but it is thin, as one episode")
    expect(out, "verdict: the queue never emptied",
           "and the verdict keys on dry, not thin -- with nothing excluded, it may say so")

    # 11. Two ports are reported independently. Blasphemous 2 plays through two, and interleaving
    #     them into one series would fabricate episodes neither port had.
    log = "".join(sample(i * 1000, 6144, port=17) + sample(i * 1000, 0, port=18)
                  for i in range(300))
    out, _ = run(log)
    expect(out, "port 17:", "two ports: the first is reported")
    expect(out, "port 18:", "two ports: the second is reported")
    expect(out, "NOT COUNTED (idle, or unobservable): 300 samples",
           "two ports: the never-fed one is idle, not merged into the fed one")

    # 12. [audio-dbg] context is parsed and LABELLED as context. The docstring explains why it
    #     cannot gate anything (two unalignable clocks), so the output must not imply that it does.
    log = ("".join(sample(i * 1000, 6144) for i in range(500)) +
           "".join("[audio-dbg] port=17 gap=5.33ms frames=256 queued_before=1024\n"
                   for _ in range(90)))
    out, _ = run(log)
    expect(out, "90 arrivals, median gap 5.33 ms", "arrival context is parsed")
    expect(out, "not used to gate anything", "...and is labelled as context only")

    # 13. An empty or unrelated log must say so and exit non-zero. A silent clean report of nothing
    #     reads as "no underruns".
    out, rc = run("some other log entirely\n")
    expect(out, "no [audio-queue] samples found", "an unrelated log is reported as no data")
    expect_rc(rc, 1, "empty log: must exit non-zero, since exit 0 reads as a clean run")

    if fails:
        print(f"audio_queue_timeline_report self-test: {fails} FAILURES")
        return 1
    print("audio_queue_timeline_report self-test: all cases pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
