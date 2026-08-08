"""gdb-side of `hle_calls` — count every HLE handler entry over a bounded window.

Loaded by `hle_calls.py` via `gdb -batch -x`. Reads its configuration from the
environment so the driver can stay a plain subprocess call:

    HLE_CALLS_SYMS    path to a "<hex-addr> <name>" table (one per line)
    HLE_CALLS_CLOCK   symbol whose entries bound the window (default prosper::k_usleep)
    HLE_CALLS_TICKS   how many clock entries to run for (default 400)
    HLE_CALLS_MODE    "attach" (default) or "launch" — launch arms every breakpoint
                      BEFORE the guest's entry point runs, so the window covers init
    HLE_CALLS_VALUES  "1" to record each handler's return values, not only its count
    HLE_CALLS_ORDER   how many leading calls to report in call order (default 24)
    HLE_CALLS_INFERIOR_TTY  launch mode only: where the INFERIOR's stdin/stdout/stderr
                      go. The driver holds gdb's own stdout on a pipe, and without this
                      the launched emulator's entire run log would inherit that pipe and
                      accumulate in the driver's memory. gdb opens the path; it does not
                      create it, so the driver does that first.

Every handler gets a Python breakpoint whose `stop()` counts the hit and returns
False, so the inferior is never actually stopped for the user. The count is kept
in this script rather than read back from `Breakpoint.hit_count`: relying on
hit_count for a non-stopping breakpoint silently reported zero for every handler
in the first version of this tool, which reads exactly like "the guest called
nothing" and is the trap this file exists to avoid.

Two things make a launch-mode result checkable rather than merely plausible, and
both are printed unconditionally:

  * `first-calls` lists the leading calls **in call order**. A window that opened
    during init leads with allocator/module-setup handlers; a window that opened
    late leads with the frame-loop pollers. That distinction is what separates
    "init made no interesting call" from "this run never saw init" — the two read
    identically in a histogram.
  * `finish-failures` counts return-value captures that could not be armed or
    decoded — but it is NOT the whole story, so each row also prints
    `(captured N/M)` whenever fewer values were recorded than calls counted.
    That second form is the one to trust: a handler that leaves its frame
    without returning normally (longjmp, thread teardown) loses its capture
    without incrementing `finish-failures`.
  * `window=complete|SHORT` says whether the clock really reached `--ticks`.
    `run`/`continue` also return on a signal gdb stops for, which would
    otherwise print a truncated histogram that reads as a finished window.
  * `positive-control=` states the verdict on the value-capture mechanism
    itself, so the reader never has to derive it — and, critically, so the rule
    is applied **per mode**. See `_positive_control()` below for the whole state
    machine and why an absent LOGIN is normal in attach mode and a defect signal
    in launch mode.

Two limits of `--values` worth knowing before believing a number:

  * It records the return **register**, never an out-struct. A handler that
    returns 0 while writing wrong bytes through a pointer argument is invisible.
  * On a handler that longjmps back into its own caller, gdb's finish breakpoint
    sits at an address the landing pad reuses, so it can capture the *landing*
    value as if it were a return. Measured on gdb 17.2. Non-returning paths are
    therefore not merely uncounted, they can be mis-counted; `(captured N/M)` is
    what flags the surrounding rows as suspect.

`handle SIGSEGV … pass` below applies to attach mode too, and materially improves
it: prosper uses SIGSEGV as a working mechanism (write watches, fault handler),
and gdb's default is to stop, so an attach window previously truncated at the
first guest fault.

GDB's manual says a `Breakpoint.stop()` implementation "should not ... alter, add
or delete any breakpoint", and `--values` adds one there. It works on gdb 17.2 —
verified across recursion and multiple threads — but a future gdb misbehaving
here is a tool bug, not a new emulator defect.
"""

import os
import sys
import gdb

# gdb runs this file with `-x`, which does NOT put its directory on sys.path -- so the sibling import
# below has to be made reachable explicitly (#2053).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hle_calls_control as _control


def _load_syms(path):
    out = []
    with open(path) as handle:
        for line in handle:
            parts = line.split()
            if len(parts) != 2:
                continue
            out.append((int(parts[0], 16), parts[1]))
    return out


SYMS = _load_syms(os.environ["HLE_CALLS_SYMS"])
CLOCK = os.environ.get("HLE_CALLS_CLOCK", "prosper::k_usleep")
TICKS = int(os.environ.get("HLE_CALLS_TICKS", "400"))
MODE = os.environ.get("HLE_CALLS_MODE", "attach")
VALUES = os.environ.get("HLE_CALLS_VALUES", "") == "1"
ORDER_N = int(os.environ.get("HLE_CALLS_ORDER", "24"))
INFERIOR_TTY = os.environ.get("HLE_CALLS_INFERIOR_TTY", "")

# The built-in control for the value-capture mechanism, and the two values it can answer.
# src/hle/hle_service.cpp `HLE(s_user_getevent)`: a function-local `static` flag makes the
# initial LOGIN event (return 0) a once-per-PROCESS delivery, and every later call answers
# SCE_USER_SERVICE_ERROR_NO_EVENT. Everything in _positive_control() follows from that.
CONTROL = "s_user_getevent"
CONTROL_LOGIN = 0
CONTROL_NO_EVENT = 0x80960007

counts = {}
# How many of CONTROL's calls passed a NON-NULL out-pointer (#2054). The LOGIN is delivered only
# `if (a0 && ...)`, so a guest polling with nullptr consumes none and gets NO_EVENT forever with no
# defect anywhere -- and the loudest verdict this tool prints could not tell that from a real one.
# Counted for the CONTROL handler only: it is one register read on one handler, not on every call.
control_eligible = 0
values = {}                     # name -> {return value: count}
order = []                      # leading calls, in call order
state = {"ticks": 0, "calls": 0, "finish_failures": 0, "exited": False}


class _Finish(gdb.FinishBreakpoint):
    """One-shot: record what the handler returned, then get out of the way."""

    def __init__(self, frame, name):
        super().__init__(frame, internal=True)
        self.hle_name = name
        self.silent = True

    def stop(self):
        try:
            result = int(self.return_value) & 0xFFFFFFFFFFFFFFFF
        except Exception:                       # no debug info for the return type
            state["finish_failures"] += 1
            return False
        values.setdefault(self.hle_name, {})
        values[self.hle_name][result] = values[self.hle_name].get(result, 0) + 1
        return False

    def out_of_scope(self):
        # The handler's frame vanished without a normal return (longjmp, thread
        # teardown). Deliberately NOT counted into finish_failures: gdb calls this
        # more than once per breakpoint on some versions, so it is not a reliable
        # counter. The per-row `captured N/M` below is what makes such a loss
        # visible, and it needs no cooperation from gdb at all.
        pass


class _Counter(gdb.Breakpoint):
    def __init__(self, addr, name):
        super().__init__("*%#x" % addr, gdb.BP_BREAKPOINT)
        self.silent = True
        self.hle_name = name
        counts[name] = 0

    def stop(self):
        global control_eligible
        counts[self.hle_name] += 1
        state["calls"] += 1
        if self.hle_name == CONTROL:
            # a0 is the first integer argument, so $rdi under SysV -- the same frame the finish
            # breakpoint below already uses. Failure to read it must not disturb the count: an
            # unreadable register leaves the eligibility unknown for that call, which is strictly
            # better than guessing either way.
            try:
                if int(gdb.parse_and_eval("$rdi")) & 0xFFFFFFFFFFFFFFFF:
                    control_eligible += 1
            except Exception:
                pass
        if len(order) < ORDER_N:
            order.append((state["calls"], self.hle_name))
        if VALUES:
            try:
                _Finish(gdb.newest_frame(), self.hle_name)
            except Exception:
                state["finish_failures"] += 1
        return False


class _Clock(gdb.Breakpoint):
    def __init__(self, spec):
        super().__init__(spec, gdb.BP_BREAKPOINT)
        self.silent = True

    def stop(self):
        state["ticks"] += 1
        return state["ticks"] >= TICKS


def _on_exited(event):
    state["exited"] = True


def _positive_control():
    """Forwards to the lifted, gdb-free verdict machine (#2053).

    The logic lives in `hle_calls_control.py` so a test can reach it: this file does `import gdb` at
    module scope, so nothing here is importable outside a gdb process and the nine-state table had no
    coverage at all. The test asserts on the SAME function this calls -- a copy would test itself.
    """
    return _control.positive_control(MODE, VALUES, counts, values,
                                     CONTROL, CONTROL_LOGIN, CONTROL_NO_EVENT,
                                     control_eligible)



gdb.execute("set pagination off")
gdb.execute("set confirm off")
# A guest fault is the emulator's business, not this tool's: pass every signal
# straight through so a launch-mode run cannot be stopped short by one.
gdb.execute("handle SIGSEGV SIGBUS SIGILL SIGFPE nostop noprint pass")
gdb.events.exited.connect(_on_exited)

armed = 0
for addr, name in SYMS:
    try:
        _Counter(addr, name)
        armed += 1
    except gdb.error as exc:  # a symbol the running binary does not actually have
        print("hle_calls: skip %s (%s)" % (name, exc))

try:
    _Clock(CLOCK)
except gdb.error as exc:
    raise SystemExit("hle_calls: cannot set the clock breakpoint on %s: %s" % (CLOCK, exc))

print("hle_calls: armed %d handler breakpoints; window = %d entries of %s (mode=%s values=%s)"
      % (armed, TICKS, CLOCK, MODE, "on" if VALUES else "off"))

if MODE == "launch":
    if INFERIOR_TTY:
        # Without this the inferior inherits gdb's stdout/stderr, which the driver holds on a
        # pipe until the window closes: the emulator's ENTIRE run log would then sit in the
        # driver's memory (a PROSPER_GFXLOG/PROSPER_DBG log reaches 1.5 GB, and exhausting
        # memory on this box takes every concurrent lane's shell down with it). gdb's own
        # stdout stays on the pipe — the result block is a few hundred bytes.
        # `set inferior-tty` OPENS this path and does not create it, so the driver creates and
        # truncates it first — otherwise gdb aborts at `run` with "No such file or directory"
        # before a single breakpoint fires. It also redirects the inferior's stdin. The path is
        # interpolated unquoted, so the driver refuses whitespace (gdb itself handles spaces
        # fine — see the note there; the restriction is this driver's, not gdb's).
        gdb.execute("set inferior-tty %s" % INFERIOR_TTY)
    # Every breakpoint above is already in place, so the window opens at the
    # process's first instruction — which is the whole point of this mode.
    gdb.execute("run")
else:
    gdb.execute("continue")

rows = sorted(((c, n) for n, c in counts.items() if c), reverse=True)
print("HLE_CALLS_BEGIN")
# `window=` is the difference between "the guest made no more calls" and "gdb stopped early".
# `run`/`continue` also return on a signal gdb stops for (SIGABRT is not in the pass list above,
# and these titles do abort), which otherwise prints a truncated histogram that looks complete.
control, control_note = _positive_control()
print("clock=%s entries=%d/%d window=%s positive-control=%s armed=%d mode=%s calls=%d "
      "finish-failures=%d exited=%d"
      % (CLOCK, state["ticks"], TICKS, "complete" if state["ticks"] >= TICKS else "SHORT",
         control, armed, MODE, state["calls"], state["finish_failures"],
         1 if state["exited"] else 0))
# Only a verdict of `ok` needs no explanation. Every other state carries the one sentence that
# says whether the run is usable, so the reader does not have to have the README open to know.
if control_note:
    print("positive-control-note: " + control_note)
if order:
    print("first-calls: " + "  ".join("%d:%s" % (n, name) for n, name in order))
for count, name in rows:
    line = "%8d  %s" % (count, name)
    if VALUES:
        seen = values.get(name, {})
        captured = sum(seen.values())
        if seen:
            top = sorted(seen.items(), key=lambda kv: -kv[1])[:6]
            line += "   ret " + ", ".join("%#x x%d" % (v, c) for v, c in top)
            if len(seen) > len(top):
                line += ", +%d more" % (len(seen) - len(top))
        else:
            line += "   ret (none captured)"
        # A handler that leaves its frame without returning normally (longjmp, thread teardown)
        # loses its capture WITHOUT incrementing finish-failures, so the header alone cannot say
        # a value set is complete. Only this per-row comparison can.
        if captured != count:
            line += "   (captured %d/%d)" % (captured, count)
    print(line)
print("distinct=%d total=%d" % (len(rows), sum(c for c, _ in rows)))
print("HLE_CALLS_END")

if MODE == "launch":
    if not state["exited"]:
        gdb.execute("kill")
else:
    gdb.execute("detach")
gdb.execute("quit")
