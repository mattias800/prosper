#!/usr/bin/env python3
"""The hle_calls value-capture positive control, one assertion per state (#2053).

Twelve tools under prosper/tools ship a test_<name>.py; hle_calls shipped none -- and it carries a
nine-state verdict machine that decides whether a reader should trust an entire measurement. A
refactor that silently broke one arm would have been caught only if a human happened to notice the
header field in a future run.

The verdict logic was reachable only from inside gdb (`import gdb` at module scope), so #2053 lifted
it into hle_calls_control.py. This asserts on that same function -- the gdb script forwards to it, so
a break here is a break there. A copied table would test itself.

The nine rows were verified BY EXECUTION during #2035's review, against a throwaway C++ fixture under
real gdb 17.2. They are known-correct rather than aspirational; this commits them.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hle_calls_control import positive_control, CONTROL, CONTROL_LOGIN, CONTROL_NO_EVENT

fails = 0


def check(cond, label):
    global fails
    if cond:
        print("  [ok]   %s" % label)
    else:
        print("  [FAIL] %s" % label)
        fails += 1


def token(mode, values_enabled, counts, values):
    return positive_control(mode, values_enabled, counts, values)[0]


def main():
    print("== test_hle_calls ==")
    N = CONTROL_NO_EVENT
    L = CONTROL_LOGIN

    # --values off: the counts are still good, so this must NOT read as a failure.
    check(token("launch", False, {}, {}) == "unchecked(values-off)", "values-off is unchecked, not a violation")

    # The two ways to have no control at all, which must stay distinguishable: never armed vs armed
    # and never entered. Collapsing them would tell a reader to widen --filter when the filter was
    # already right.
    check(token("launch", True, {}, {}) == "not-in-filter", "control absent from counts is not-in-filter")
    check(token("launch", True, {CONTROL: 0}, {}) == "not-called", "control armed but unentered is not-called")

    # A value the contract cannot produce means the CAPTURE is wrong, not the guest.
    check(token("attach", True, {CONTROL: 3}, {CONTROL: {L: 1, 0x1: 1, N: 1}}).startswith("VIOLATED(unexpected-ret-0x1)"),
          "an impossible return value violates, naming the value")

    # LOGIN is once per PROCESS, so two cannot both be real.
    check(token("launch", True, {CONTROL: 4}, {CONTROL: {L: 2, N: 2}}) == "VIOLATED(login-x2)",
          "two LOGINs violate")

    # The good case.
    check(token("launch", True, {CONTROL: 3}, {CONTROL: {L: 1, N: 2}}) == "ok", "one LOGIN in launch mode is ok")

    # attach with no LOGIN is EXPECTED -- this is the row an earlier revision got wrong, telling a
    # reader of a perfectly good attach capture to throw it away.
    check(token("attach", True, {CONTROL: 5}, {CONTROL: {N: 5}}) == "absent(attach:login-consumed-pre-window)",
          "attach with no LOGIN is absent(), not a violation")

    # launch, no LOGIN, and NOT every return captured: a miss and a loss are indistinguishable, so
    # the instrument must refuse to pick one.
    check(token("launch", True, {CONTROL: 6}, {CONTROL: {N: 5}}) == "VOID(5-returns-for-6-calls)",
          "launch with a lost return is VOID, not VIOLATED")

    # launch, no LOGIN, every return accounted for: now it IS a violation.
    check(token("launch", True, {CONTROL: 6}, {CONTROL: {N: 6}}) == "VIOLATED(launch-window-no-login)",
          "launch with all returns captured and no LOGIN violates")

    # The VOID/VIOLATED boundary is the single most consequential branch here -- it is the difference
    # between "distrust this run" and "this run cannot say". Pin it at exactly one lost return.
    check(token("launch", True, {CONTROL: 6}, {CONTROL: {N: 6}}) != token("launch", True, {CONTROL: 6}, {CONTROL: {N: 5}}),
          "one lost return is what separates VIOLATED from VOID")

    # The display cap and the verdict read DIFFERENT structures: truncation lives in the print loop,
    # the verdict reads the full dict. A value pushed out of a displayed top-6 must still violate --
    # exactly the separation a later refactor collapses by accident.
    buried = {L: 1}
    for i in range(1, 8):
        buried[0x8096F000 + i] = 100 - i          # seven values, all outranking the stray below
    buried[0x1] = 1                                # rank 9: never displayed
    check(token("attach", True, {CONTROL: sum(buried.values())}, {CONTROL: buried}) ==
          "VIOLATED(unexpected-ret-0x1)",
          "a value ranked out of the displayed top-6 still violates")

    # ===== the benign cause of the loudest verdict (#2054) ====================================
    # hle_service.cpp delivers the LOGIN only `if (a0 && delivered.exchange(1) == 0)`. A guest that
    # polls sceUserServiceGetEvent(nullptr) consumes no LOGIN and gets NO_EVENT forever, with no
    # defect anywhere -- and the tool told it to distrust a perfectly good value set. That is the
    # same failure shape #2035 was opened to fix, relocated.
    launch_all_noevent = ("launch", True, {CONTROL: 6}, {CONTROL: {N: 6}})

    # Not recorded -> unchanged verdict, but the note must SAY the benign case is unexcluded.
    tok_unknown, note_unknown = positive_control(*launch_all_noevent, control_eligible_calls=None)
    check(tok_unknown == "VIOLATED(launch-window-no-login)", "no a0 data keeps the old verdict")
    check("cannot be excluded" in note_unknown, "no a0 data says the benign case is unexcluded")

    # Every call passed NULL -> a distinct, NON-ACCUSATORY state. This is the whole point.
    tok_null, note_null = positive_control(*launch_all_noevent, control_eligible_calls=0)
    check(tok_null == "absent(control-never-eligible)", "all-null out-pointers is absent(), not VIOLATED")
    check("NOT a defect" in note_null, "the all-null verdict says plainly it is not a defect")

    # At least one eligible call -> a genuine violation, and the note must say the benign case is
    # EXCLUDED rather than repeating the hedge. A verdict that hedges when it need not is as useless
    # as one that accuses when it should not.
    tok_real, note_real = positive_control(*launch_all_noevent, control_eligible_calls=6)
    check(tok_real == "VIOLATED(launch-window-no-login)", "an eligible call keeps the violation")
    check("excluded" in note_real and "cannot be excluded" not in note_real,
          "with a0 recorded the violation states the benign case is EXCLUDED")

    # Eligibility must not override the VOID split: a lost return still makes miss and loss
    # indistinguishable, whatever the pointer was. Kills placing the new state after the VOID check.
    check(positive_control("launch", True, {CONTROL: 6}, {CONTROL: {N: 5}}, control_eligible_calls=6)[0]
          == "VOID(5-returns-for-6-calls)", "an eligible call does not turn a lost return into a violation")
    # ...and the all-null state DOES take precedence over VOID, because it asks a prior question:
    # not "can I tell a miss from a loss" but "was there anything to miss".
    check(positive_control("launch", True, {CONTROL: 6}, {CONTROL: {N: 5}}, control_eligible_calls=0)[0]
          == "absent(control-never-eligible)", "all-null precedes the VOID split")

    # Eligibility is irrelevant once a LOGIN was actually seen.
    check(positive_control("launch", True, {CONTROL: 3}, {CONTROL: {L: 1, N: 2}}, control_eligible_calls=0)[0]
          == "ok", "a captured LOGIN outranks any eligibility count")

    # A note must accompany every non-ok verdict: a bare token tells a reader nothing to act on.
    for mode, ve, c, v in (("launch", False, {}, {}), ("launch", True, {}, {}),
                           ("launch", True, {CONTROL: 0}, {}),
                           ("attach", True, {CONTROL: 5}, {CONTROL: {N: 5}}),
                           ("launch", True, {CONTROL: 6}, {CONTROL: {N: 5}}),
                           ("launch", True, {CONTROL: 6}, {CONTROL: {N: 6}})):
        tok, note = positive_control(mode, ve, c, v)
        check(bool(note.strip()), "verdict %s carries an actionable note" % tok)
    check(positive_control("launch", True, {CONTROL: 3}, {CONTROL: {L: 1, N: 2}})[1] == "",
          "the ok verdict carries no note")

    print("== FAILURES: %d ==" % fails if fails else "== all checks passed ==")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
