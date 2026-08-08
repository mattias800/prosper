#!/usr/bin/env python3
"""The `hle_calls` value-capture positive control, lifted out of the gdb script (#2053).

`hle_calls_gdb.py` does `import gdb` at module scope, so nothing in it can be imported outside a
gdb process and no ordinary test could reach any of it. The verdict machine needs nothing from gdb:
it is a pure function of (mode, values_enabled, counts, values). Lifting it here lets the whole
nine-state table be tested with no gdb dependency and no fixture process, and leaves the gdb script
importing the same code the test asserts on -- which is the only arrangement where the test means
anything.

The docstring on `positive_control` is the contract; it moved here verbatim.
"""

CONTROL = "s_user_getevent"
CONTROL_LOGIN = 0
CONTROL_NO_EVENT = 0x80960007


def positive_control(mode, values_enabled, counts, values,
                     control=CONTROL, control_login=CONTROL_LOGIN,
                     control_no_event=CONTROL_NO_EVENT,
                     control_eligible_calls=None):
    """Verdict on this run's value capture -- `(token, note)`, note "" when nothing is wrong.

    The control is `s_user_getevent`, whose contract is derivable from the source before any run:
    exactly one LOGIN (`0x0`) per PROCESS, `0x80960007` forever after. **The expected observation is
    therefore mode-dependent, and that is the whole point of computing it here** -- an earlier
    revision of this tool documented the launch-mode expectation as if it held everywhere, which told
    a reader of a perfectly good attach capture to throw it away.

      launch  the window opens at the first instruction, so the single LOGIN falls inside it:
              exactly one `0x0` is required, and its absence is a signal.
      attach  init consumed the LOGIN before gdb could attach, so a correct capture has none. Zero or
              one is normal; only more than one is impossible.

    Absence of the control is not a violation, and the two ways to have none are reported apart:
    `not-in-filter` (never armed) and `not-called` (armed, never entered). Either way the run carries
    no value-capture control at all.

    A missing LOGIN in launch mode is only VIOLATED when every one of the control's calls was
    captured. If any return was lost, a miss and a loss are indistinguishable and the verdict is VOID
    -- an instrument that cannot tell those apart must not pick one.

    `control_eligible_calls` is how many of the control's calls passed a NON-NULL out-pointer, or
    None when the caller did not record it (#2054). It exists because the loudest verdict this
    machine prints has a BENIGN cause it could not otherwise distinguish: hle_service.cpp delivers
    the LOGIN only `if (a0 && delivered.exchange(1) == 0)`, so a guest that polls
    sceUserServiceGetEvent(nullptr) consumes no LOGIN and gets NO_EVENT forever, with no defect
    anywhere -- and the tool would tell it to throw away a perfectly good value set. Zero eligible
    calls is therefore a distinct, non-accusatory state rather than a violation.
    """
    if not values_enabled:
        return ("unchecked(values-off)",
                "no return values were recorded, so this run has no value-capture control; "
                "the call counts are unaffected. Re-run with --values to check the mechanism.")
    if control not in counts:
        return ("not-in-filter",
                "%s was never armed (--filter excluded it, or this binary has no such "
                "handler), so nothing in this run checks the value-capture mechanism. Widen "
                "--filter to include it before believing a surprising value." % control)
    calls = counts[control]
    if calls == 0:
        return ("not-called",
                "%s was armed but never entered in this window, so nothing in this run checks "
                "the value-capture mechanism." % control)

    seen = values.get(control, {})
    captured = sum(seen.values())
    logins = seen.get(control_login, 0)
    unexpected = sorted(v for v in seen if v not in (control_login, control_no_event))

    if unexpected:
        return ("VIOLATED(unexpected-ret-%#x)" % unexpected[0],
                "%s can only answer 0x0 or %#x, so a captured %#x means the capture itself is "
                "wrong (or the handler changed). Distrust every value in this run."
                % (control, control_no_event, unexpected[0]))
    if logins > 1:
        return ("VIOLATED(login-x%d)" % logins,
                "%s delivers LOGIN once per process, so %d captured 0x0 returns cannot all be "
                "real. Distrust every value in this run." % (control, logins))
    if logins == 1:
        return ("ok", "")
    if mode != "launch":
        return ("absent(attach:login-consumed-pre-window)",
                "expected in attach mode and NOT a defect: init consumed the once-per-process "
                "LOGIN long before gdb could attach. Do not discard this run's values over it -- "
                "but note that nothing in it independently confirms the mechanism either; only "
                "--launch can.")
    # Before deciding a missing LOGIN is anyone's fault, ask whether the control was ever ELIGIBLE
    # to deliver one (#2054). This is checked ahead of the VOID/VIOLATED split because it is not a
    # weaker form of either: those two ask "can I tell a miss from a loss", and this asks whether
    # there was anything to miss.
    if control_eligible_calls == 0:
        return ("absent(control-never-eligible)",
                "every one of %s's %d calls passed a NULL out-pointer, and the LOGIN is delivered "
                "only when that pointer is non-null -- so no LOGIN was consumed and none could be "
                "captured. NOT a defect and not a reason to distrust this run's values, but nothing "
                "in it confirms the value-capture mechanism either." % (control, calls))
    if captured != calls:
        return ("VOID(%d-returns-for-%d-calls)" % (captured, calls),
                "the launch window covered init yet captured no LOGIN -- but only %d returns were "
                "recorded for %s's %d calls, so a lost value and a wrong one are indistinguishable "
                "here. This run neither confirms nor refutes the mechanism."
                % (captured, control, calls))
    if control_eligible_calls is None:
        return ("VIOLATED(launch-window-no-login)",
                "the window started at the first instruction and captured all %d %s returns, none "
                "of them the once-per-process LOGIN. Distrust every value in this run -- UNLESS the "
                "guest only ever passed a null event pointer, which returns NO_EVENT without "
                "consuming the LOGIN. This run did not record the out-pointer, so that benign case "
                "cannot be excluded here (#2054)." % (calls, control))
    return ("VIOLATED(launch-window-no-login)",
            "the window started at the first instruction and captured all %d %s returns, none of "
            "them the once-per-process LOGIN, and %d of those calls passed a NON-NULL out-pointer "
            "and were therefore eligible to receive it. The benign null-pointer explanation is "
            "excluded. Distrust every value in this run."
            % (calls, control, control_eligible_calls))
