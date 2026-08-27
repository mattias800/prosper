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
    if rc != 0:
        print(f" [FAIL] healthy: exit {rc}, expected 0")

    # 2. THE CONFOUND ARM, and the reason the gate exists. A port that is OPEN but never fed reads
    #    queued=0 for its whole life. Before the gate this tool called that STARVED, and on a real
    #    600 s run it reported 53% of samples dry with a 40 SECOND "episode" -- which would have
    #    decided a pacer A/B on whichever arm happened to idle more.
    log = "".join(sample(i * 1000, 0) for i in range(2000))
    out, _ = run(log)
    expect(out, "IDLE (port open, not being fed): 2000 samples",
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
    expect(out, "IDLE (port open, not being fed): 0 samples",
           "burst: a bracketed dip is not idle")

    # 5. A LONG silence BETWEEN two fed stretches -- bracketed by audio on both sides, and still
    #    not an underrun. This is the arm that shows bracketing alone is insufficient: a 5 s gap
    #    between two songs satisfies "audio before" and "audio after" and would be reported as one
    #    enormous underrun without the duration condition.
    log = "".join(sample(i * 1000, 0 if 200 <= i < 5200 else 6144) for i in range(5400))
    out, _ = run(log)
    expect(out, "UNDERRUN (dry while streaming): 0 episodes",
           "long silence: a 5 s bracketed gap is not an underrun")
    expect(out, "IDLE (port open, not being fed): 5000 samples",
           "long silence: ...it is idle, all 5000 samples of it")
    expect(out, "the queue never emptied", "long silence: and the verdict says so")

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
    expect(out_small, "IDLE (port open, not being fed): 8 samples",
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
    expect(out, "IDLE (port open, not being fed): 60 samples",
           "leading dry: ...it is start-up idleness")

    # 6c. TRAILING dry: the port stops being fed and stays dry to the end of the log. Symmetric to
    #     6b, and it must be the "audio seen after it" condition doing the work -- again short
    #     enough that the duration bound cannot be what excludes it.
    log_tail = ("".join(sample(i * 1000, 6144) for i in range(500)) +
                "".join(sample((500 + i) * 1000, 0) for i in range(60)))
    out, _ = run(log_tail)
    expect(out, "UNDERRUN (dry while streaming): 0 episodes",
           "trailing dry: a port that stops feeding is not underrunning")
    expect(out, "IDLE (port open, not being fed): 60 samples",
           "trailing dry: ...it is shutdown idleness")

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
    expect(out, "the queue never emptied", "and the verdict keys on dry, not thin")

    # 11. Two ports are reported independently. Blasphemous 2 plays through two, and interleaving
    #     them into one series would fabricate episodes neither port had.
    log = "".join(sample(i * 1000, 6144, port=17) + sample(i * 1000, 0, port=18)
                  for i in range(300))
    out, _ = run(log)
    expect(out, "port 17:", "two ports: the first is reported")
    expect(out, "port 18:", "two ports: the second is reported")
    expect(out, "IDLE (port open, not being fed): 300 samples",
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
    if rc == 0:
        print(" [FAIL] empty log: exit 0, which reads as a clean run")
        return 1

    if fails:
        print(f"audio_queue_timeline_report self-test: {fails} FAILURES")
        return 1
    print("audio_queue_timeline_report self-test: all cases pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
