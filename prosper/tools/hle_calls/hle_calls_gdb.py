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
import gdb


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

counts = {}
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
        counts[self.hle_name] += 1
        state["calls"] += 1
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
        # `set inferior-tty` takes the rest of the line as the path and does not create it;
        # the driver rejects whitespace and creates the file.
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
print("clock=%s entries=%d/%d window=%s armed=%d mode=%s calls=%d finish-failures=%d exited=%d"
      % (CLOCK, state["ticks"], TICKS, "complete" if state["ticks"] >= TICKS else "SHORT",
         armed, MODE, state["calls"], state["finish_failures"],
         1 if state["exited"] else 0))
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
