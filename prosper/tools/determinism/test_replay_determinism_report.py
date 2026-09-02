#!/usr/bin/env python3
"""Regression for replay_determinism_report.py's VERDICT machine (#2945).

The census half of that tool is arithmetic. The half worth pinning is the third verdict: a campaign
whose subject returned one hash and whose control never failed reports **UNDECIDED**, not
DETERMINISTIC. #2945's rate drifts machine-wide over minutes, so a quiet window and a repaired
renderer produce identical numbers, and the whole reason the control is in the campaign is to keep
those two apart. Delete the control check and every arm below except one still passes -- so that
one arm is the test, and it is written as a MUTATION: it re-runs the evaluator with the check
removed and fails if the mutant still reports UNDECIDED.

Pure and platform-independent: no Vulkan, no GPU, no capture. It asserts on the reading of a
campaign, which is where this issue's two published misreadings happened.
"""

import importlib.util
import io
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = importlib.util.spec_from_file_location(
    "prosper_replay_determinism_report", os.path.join(HERE, "replay_determinism_report.py"))
RPT = importlib.util.module_from_spec(SPEC)
# Register before exec: @dataclass resolves its own module out of sys.modules, so a
# module loaded from a path and never registered raises there rather than importing.
sys.modules[SPEC.name] = RPT
SPEC.loader.exec_module(RPT)

fails = 0


def check(cond, msg):
    global fails
    if not cond:
        print(f"FAIL: {msg}")
        fails += 1
    else:
        print(f"ok: {msg}")


HEADER = "epoch,round,cond,peers,gpu_pct,role,arm,value,rc,ms\n"


def csv(rows):
    return HEADER + "".join(",".join(str(f) for f in r) + "\n" for r in rows)


def verdict(text):
    return RPT.evaluate(RPT.parse(io.StringIO(text)))


def subject(round_, value, epoch=None, arm="host/full", rc=0, cond="unloaded", peers=0, gpu=8):
    return [epoch if epoch is not None else 1000 + round_, round_, cond, peers, gpu,
            "subject", arm, value, rc, 1500]


def control(round_, value, epoch=None, rc=0, cond="unloaded", peers=0, gpu=8):
    return [epoch if epoch is not None else 1000 + round_, round_, cond, peers, gpu,
            "control", "host/vkprobe", value, rc, 900]


def main():
    # --- The class this whole campaign exists to detect, built BY HAND -------------------------
    # A fixture that cannot express a varying campaign would make every clean arm below vacuous, so
    # the first arm constructs one and requires the tool to name it.
    varying = csv([subject(1, "aaaa"), subject(2, "bbbb"), subject(3, "aaaa")])
    r = verdict(varying)
    check(r.verdict == RPT.NONDETERMINISTIC, "two distinct hashes on one arm -> NONDETERMINISTIC")
    check(any("2 distinct" in reason for reason in r.reasons),
          "...and the reason names how many distinct hashes over how many replays")

    # A varying arm outranks a silent control: the campaign is decisive without it.
    r = verdict(csv([subject(1, "aaaa"), control(1, "pass"),
                     subject(2, "bbbb"), control(2, "pass")]))
    check(r.verdict == RPT.NONDETERMINISTIC,
          "a varying subject is decisive even when the control never fired")

    # --- The three-way verdict -----------------------------------------------------------------
    stable_fired = csv([subject(1, "aaaa"), control(1, "pass"),
                        subject(2, "aaaa"), control(2, "fail:3/20"),
                        subject(3, "aaaa"), control(3, "pass")])
    r = verdict(stable_fired)
    check(r.verdict == RPT.DETERMINISTIC,
          "one hash + a control that failed at least once -> DETERMINISTIC")

    stable_quiet = csv([subject(1, "aaaa"), control(1, "pass"),
                        subject(2, "aaaa"), control(2, "pass"),
                        subject(3, "aaaa"), control(3, "pass")])
    r = verdict(stable_quiet)
    check(r.verdict == RPT.UNDECIDED,
          "one hash + a control that never failed -> UNDECIDED, not DETERMINISTIC")
    check(any("no instance observed" in reason for reason in r.reasons),
          "...and it says what to quote instead")

    r = verdict(csv([subject(1, "aaaa"), subject(2, "aaaa"), subject(3, "aaaa")]))
    check(r.verdict == RPT.UNDECIDED, "one hash and NO control rows at all -> UNDECIDED")

    # A control that could not run has measured nothing. Reading its non-zero exit as a failure of
    # the class would manufacture DETERMINISTIC out of a broken instrument.
    r = verdict(csv([subject(1, "aaaa"), control(1, "unavailable", rc=2),
                     subject(2, "aaaa"), control(2, "unavailable", rc=2)]))
    check(r.verdict == RPT.UNDECIDED, "a control that could not RUN does not count as fired")
    check(any("could not run" in reason for reason in r.reasons),
          "...and the report says the control was broken")

    # A replay that did not finish is not a sample of what the finished ones measured.
    r = verdict(csv([subject(1, "aaaa"), control(1, "fail:1/20"),
                     subject(2, "none", rc=124), control(2, "fail:1/20"),
                     subject(3, "aaaa"), control(3, "pass")]))
    check(r.verdict == RPT.UNDECIDED, "a non-zero replay exit -> UNDECIDED even with a live control")
    check(any("exited non-zero" in reason for reason in r.reasons),
          "...and the reason names the unfinished replays rather than a hash count")
    # ...and the timeout's empty value is NOT counted as a second hash, which would report a
    # timeout as the very defect the campaign is looking for.
    check(not any("distinct" in reason for reason in r.reasons),
          "an unfinished replay is not counted as a distinct hash")

    # Per-arm, not pooled: two arms each stable at their OWN hash is not two distinct hashes.
    r = verdict(csv([subject(1, "aaaa", arm="host/full"), subject(1, "bbbb", arm="host/draw42"),
                     subject(2, "aaaa", arm="host/full"), subject(2, "bbbb", arm="host/draw42"),
                     control(1, "fail:2/20")]))
    check(r.verdict == RPT.DETERMINISTIC,
          "distinct arms are counted separately, not pooled into one census")

    # --- The mutation arm: remove the control check and this test must go red -------------------
    original = RPT.control_failed
    try:
        RPT.control_failed = lambda row: row.role == "control"   # "any control row counts as fired"
        mutant = verdict(stable_quiet)
    finally:
        RPT.control_failed = original
    check(mutant.verdict == RPT.DETERMINISTIC,
          "MUTATION: with the control check defeated the quiet campaign reports DETERMINISTIC")
    check(verdict(stable_quiet).verdict == RPT.UNDECIDED,
          "...and the unmutated evaluator still reports UNDECIDED, so the arm is not vacuous")

    # --- Parsing: a malformed row is an error, never a silently dropped sample ------------------
    try:
        verdict(HEADER + "1,1,unloaded,0,8,subject,host/full,aaaa\n")
        check(False, "a short row raises instead of being dropped")
    except ValueError as exc:
        check("expected 10 fields" in str(exc), "a short row raises and says what was expected")
    try:
        verdict(HEADER + "notanumber,1,unloaded,0,8,subject,host/full,aaaa,0,10\n")
        check(False, "a non-numeric epoch raises")
    except ValueError as exc:
        check("line" in str(exc), "a non-numeric field raises with the line number")

    # --- Span and exit codes -------------------------------------------------------------------
    r = verdict(csv([subject(1, "aaaa", epoch=1000), control(1, "fail:1/20", epoch=1000),
                     subject(2, "aaaa", epoch=8200), control(2, "pass", epoch=8200)]))
    check(r.span_s == 7200, "the wall-clock span is reported, because n without a span is one window")
    check("7200s" in RPT.format_report(r), "...and appears in the formatted report")

    # A campaign labelled `selfload` whose GPU never got busy has not tested the load condition.
    # The report cannot decide that for the reader, but it must not hide it either.
    r = verdict(csv([subject(1, "aaaa", cond="selfload", gpu=6),
                     control(1, "pass", cond="selfload", gpu=6)]))
    text = RPT.format_report(r)
    check("GPU busy: mean 6.0%" in text,
          "the report prints the GPU utilisation each condition actually reached")

    tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".determinism_test.csv")
    for text, expected in ((varying, 1), (stable_fired, 0), (stable_quiet, 2)):
        with open(tmp, "w") as handle:
            handle.write(text)
        try:
            rc = RPT.main([tmp, "--exit-code"])
        finally:
            os.unlink(tmp)
        check(rc == expected, f"--exit-code returns {expected} for that campaign")

    if fails:
        print(f"\n{fails} FAILED")
        return 1
    print("\n== PASS ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
