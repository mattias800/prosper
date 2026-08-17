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
    decoded, and every one of them now names its own reason on the
    `finish-failure-reasons:` line — a bare count is what let #2075 stay
    undiagnosed. It is still NOT the whole story, so each row also prints
    `(captured N/M)` whenever fewer values were recorded than calls counted.
    That second form is the one to trust: a handler that leaves its frame
    without returning normally (longjmp, thread teardown) loses its capture
    without incrementing `finish-failures`.
  * `value-source=dwarf:N,rax:M` says which read answered. Both are the same
    number when both are available; only DWARF is independent of this tool's
    "the HLE ABI returns in %rax" assumption, and a disagreement between them
    is reported loudly on its own line. `dwarf:0` is normal, not degraded:
    prosper's default build type is Release, which carries no debug info.
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
import re
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

RET_MASK = 0xFFFFFFFFFFFFFFFF

counts = {}
# How many of CONTROL's calls passed a NON-NULL out-pointer (#2054). The LOGIN is delivered only
# `if (a0 && ...)`, so a guest polling with nullptr consumes none and gets NO_EVENT forever with no
# defect anywhere -- and the loudest verdict this tool prints could not tell that from a real one.
# Counted for the CONTROL handler only: it is one register read on one handler, not on every call.
control_eligible = 0
values = {}                     # name -> {return value: count}
order = []                      # leading calls, in call order
state = {"ticks": 0, "calls": 0, "finish_failures": 0, "exited": False,
         # Where each captured value came from, and whether the two sources ever disagreed.
         # `rax` is not a degraded mode -- see _Finish.stop() -- but a reader is entitled to know
         # which one answered, because only `dwarf` is independent of this tool's ABI assumption.
         "src_dwarf": 0, "src_rax": 0, "src_mismatch": 0}
# Distinct reasons a capture produced NO value at all, with counts. `finish-failures=N` on its own
# cost this feature four days of being void with nobody able to say why (#2075): the header said how
# many captures failed and nothing said what the failure WAS. One line of text is the whole fix.
finish_failure_reasons = {}


REASON_CAP = 32


def _reason(kind, exc):
    # gdb's messages carry the failing ADDRESS ("Cannot access memory at address 0x…"), which differs
    # per call: left alone, every failure would be its own "distinct" reason and this table would grow
    # once per call over a window of thousands. Fold the address out, and cap the table below.
    return re.sub(r"0x[0-9a-fA-F]+", "0x<addr>", "%s: %s: %s" % (kind, type(exc).__name__, exc))[:200]


def _note_reason(text):
    if text in finish_failure_reasons:
        finish_failure_reasons[text] += 1
    elif len(finish_failure_reasons) < REASON_CAP:
        finish_failure_reasons[text] = 1
    else:
        overflow = "(+reasons beyond the first %d distinct)" % REASON_CAP
        finish_failure_reasons[overflow] = finish_failure_reasons.get(overflow, 0) + 1


class _Finish(gdb.FinishBreakpoint):
    """One-shot: record what the handler returned, then get out of the way."""

    def __init__(self, frame, name):
        super().__init__(frame, internal=True)
        self.hle_name = name
        self.silent = True

    def stop(self):
        """Read the return value from TWO independent sources, because one of them is optional.

        `gdb.FinishBreakpoint.return_value` is derived from the function's **DWARF return type**, so
        it exists only on a build that carries debug info. prosper's default build type is `Release`
        (`-O3 -DNDEBUG`, no `-g`, see prosper/CMakeLists.txt) -- on such a binary gdb has no type for
        the handler, `return_value` is `None`, and `int(None)` raised once per call: that is #2075,
        where `finish-failures` equalled the call count exactly and every row read `(captured 0/N)`.
        Measured one-variable A/B on ONE binary (`objcopy --strip-debug` of a RelWithDebInfo
        `boot_trace`, same gdb 17.2, same title, same window): with `.debug_*` present
        `finish-failures=0` and the control reported `ok`; with it removed, 4 calls / 4 failures /
        `VOID(0-returns-for-4-calls)`. The FinishBreakpoint itself constructed fine in both arms, so
        the "gdb cannot unwind to the guest caller" hypothesis in #2075 is falsified, not fixed.

        `%rax` needs no debug info and is exactly right for what this tool arms: the driver
        enumerates only functions with the six-`unsigned long` HLE signature, and every one of those
        is written through an `HLE(...)` macro that returns `uint64_t` -- an integer return, i.e.
        `%rax` under SysV. The finish breakpoint stops at the return address in the caller, before
        any instruction there has executed, so `%rax` still holds the value the handler returned.
        `CONFIDENCE: HIGH` -- and it is checked rather than asserted: when both sources answer, a
        disagreement is counted and reported, which is what would fire if a handler ever returned
        something the ABI does not put in `%rax` (a struct, a float).

        DWARF is preferred when present because it is ABI-aware for ANY return type, so a debug build
        keeps behaving exactly as it did before this change.
        """
        why = []
        dwarf = None
        try:
            rv = self.return_value
            if rv is None:
                why.append("return_value: None (this binary has no DWARF return type for the "
                           "handler -- a Release build; %rax below is the answer)")
            else:
                dwarf = int(rv) & RET_MASK
        except Exception as exc:
            why.append(_reason("return_value", exc))
        rax = None
        try:
            rax = int(gdb.newest_frame().read_register("rax")) & RET_MASK
        except Exception as exc:
            why.append(_reason("rax", exc))

        result = dwarf if dwarf is not None else rax
        if result is None:
            # Only now is the capture lost -- and the reader gets to see why, per distinct reason.
            state["finish_failures"] += 1
            for text in why:
                _note_reason(text)
            return False
        if dwarf is not None and rax is not None and dwarf != rax:
            state["src_mismatch"] += 1
        state["src_dwarf" if dwarf is not None else "src_rax"] += 1
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
            except Exception as exc:
                # The other half of #2075's diagnosis problem: this site and the decode site above
                # increment ONE counter, so "finish-failures == calls" could not distinguish "never
                # armed" from "armed and could not be decoded". Both now say which they were.
                _note_reason(_reason("FinishBreakpoint()", exc))
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
# `value-source=` says which of the two reads answered (see _Finish.stop()). It is printed only with
# --values, where it is never noise: `dwarf:0,rax:N` is the normal shape on prosper's default Release
# build, and it is the field that would have named #2075's cause on sight.
source = ""
if VALUES:
    source = " value-source=dwarf:%d,rax:%d" % (state["src_dwarf"], state["src_rax"])
print("clock=%s entries=%d/%d window=%s positive-control=%s armed=%d mode=%s calls=%d "
      "finish-failures=%d exited=%d%s"
      % (CLOCK, state["ticks"], TICKS, "complete" if state["ticks"] >= TICKS else "SHORT",
         control, armed, MODE, state["calls"], state["finish_failures"],
         1 if state["exited"] else 0, source))
# Only a verdict of `ok` needs no explanation. Every other state carries the one sentence that
# says whether the run is usable, so the reader does not have to have the README open to know.
if control_note:
    print("positive-control-note: " + control_note)
# A failure count with no reason is what made #2075 a four-day investigation instead of a one-line
# diagnosis. Distinct reasons, most frequent first, capped so a pathological run cannot flood.
if finish_failure_reasons:
    ranked = sorted(finish_failure_reasons.items(), key=lambda kv: -kv[1])
    shown = ranked[:4]
    line = "finish-failure-reasons: " + "; ".join("%dx %s" % (n, t) for t, n in shown)
    if len(ranked) > len(shown):
        line += "; +%d more distinct" % (len(ranked) - len(shown))
    print(line)
# The check on this tool's own ABI assumption: both sources read the same call, so they can only
# disagree if the handler's real return does not live in %rax. Loud, because every %rax-sourced value
# in the run would then be suspect -- including on builds where DWARF is absent and nothing can check.
if state["src_mismatch"]:
    print("value-source-MISMATCH: %d captures where the DWARF return value and %%rax disagreed. "
          "Every rax-sourced value in this run is suspect: a handler whose return does not live in "
          "%%rax (a struct or float return) breaks this tool's ABI assumption." % state["src_mismatch"])
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
