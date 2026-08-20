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
    HLE_CALLS_OUT_BYTES  0 = off. Otherwise snapshot this many bytes at a0/a1 before
                      and after each call, and report what CHANGED -- the half a
                      return value cannot reach (#2045)
    HLE_CALLS_INFERIOR_TTY  launch mode only: where the INFERIOR's stdin/stdout/stderr
                      go. The driver holds gdb's own stdout on a pipe, and without this
                      the launched emulator's entire run log would inherit that pipe and
                      accumulate in the driver's memory. gdb opens the path; it does not
                      create it, so the driver does that first.
    HLE_CALLS_ELF_TYPE   the target's ELF `e_type` (2 = ET_EXEC, 3 = ET_DYN/PIE)
    HLE_CALLS_ELF_ENTRY  its link-time entry point -- one half of the load-bias
                      subtraction below (#2605)

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
  * `elf=`, `load-bias=` and `symbol-check=` say where the breakpoints actually
    went. On an `ET_EXEC` target the bias is `0x0` and the check is a formality;
    on a PIE it is the whole difference between a measurement and a window that
    armed nothing, so it is derived from the live process and then verified
    against gdb's own symbol table before a single breakpoint is inserted
    (`_resolve_load_bias` / `_symbol_landing`, #2605).
  * `positive-control=` states the verdict on the value-capture mechanism
    itself, so the reader never has to derive it — and, critically, so the rule
    is applied **per mode**. See `_positive_control()` below for the whole state
    machine and why an absent LOGIN is normal in attach mode and a defect signal
    in launch mode.

Two limits of `--values` worth knowing before believing a number:

  * It records the return **register**, never an out-struct. A handler that
    returns 0 while writing wrong bytes through a pointer argument is invisible
    to it — that is what `--out-bytes` answers (see `_sample_out_args` below),
    and it is a separate observation with its own five-state accounting.
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


def _abort(text):
    """Stop the run with a named cause, rather than letting it produce a window that armed nothing.

    The driver prints an `HLE_CALLS_ABORT:` line INSTEAD of the raw gdb transcript, so whatever is
    said here is what a reader sees. Exit code 3 so a caller can tell "this tool refused" from "gdb
    died" (1) -- both differ from a clean run's 0.
    """
    print("HLE_CALLS_ABORT: " + text)
    sys.stdout.flush()
    gdb.execute("quit 3")


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
# 0 = off. When non-zero, snapshot this many bytes at a0/a1 before and after every call (#2045).
OUT_BYTES = int(os.environ.get("HLE_CALLS_OUT_BYTES", "0"))
INFERIOR_TTY = os.environ.get("HLE_CALLS_INFERIOR_TTY", "")
# The target's linkage, from the driver's own read of the ELF header. ET_EXEC (2) means `nm`'s
# addresses are already runtime addresses; ET_DYN (3) means they are offsets and the bias below has
# to come from the live process. 0 = the driver did not recognise the file as an ELF.
ELF_TYPE = int(os.environ.get("HLE_CALLS_ELF_TYPE", "2"))
ELF_ENTRY = int(os.environ.get("HLE_CALLS_ELF_ENTRY", "0"))
ET_EXEC = 2
ET_DYN = 3
AT_ENTRY = 9                    # auxv key: the entry point the kernel actually jumped to
# How many enumerated symbols to verify land exactly on a function start before arming anything.
# A sample, not the whole table: this is one gdb command per symbol and the table reaches ~750.
SYMBOL_CHECK_N = 8

# The built-in control for the value-capture mechanism, and the two values it can answer.
# src/hle/service/hle_service.cpp `HLE(s_user_getevent)`: a function-local `static` flag makes the
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


# --- out-struct sampling (--out-bytes, #2045) ---------------------------------------------------
# The recurring bug shape on this codebase is "success returned, out-struct never written":
# sceSystemServiceGetStatus aliased to the wrong handler, sceSystemServiceGetDisplaySafeAreaInfo
# leaving `ratio` 0, sceNpEntitlementAccessGetSkuFlag unregistered with the out pointer untouched,
# sceNpTrophy2GetTrophyInfoArray handing back a garbage count. EVERY one of them returns 0, so a
# return-value census cannot see any of them -- the bound a value census puts on a candidate reaches
# exactly as far as the return register.
#
# So: snapshot N bytes at whichever of a0/a1 is a readable pointer, on entry and again at return,
# and report the difference. The before/after form is what makes the result self-validating, which
# is the property #2075 showed this tool otherwise lacked: "the handler wrote nothing" and "this
# tool never looked" are DIFFERENT counters here, and both are printed.
OUT_ARGS = (("a0", "rdi"), ("a1", "rsi"))       # SysV integer argument registers 0 and 1
OUT_SHAPE_CAP = 32
# A pointer is never in page 0. Anything below this is an ordinary small argument -- a handle, a
# size, an enum -- and saying so is not the same statement as "the pointer could not be read".
OUT_MIN_POINTER = 0x1000
out_stats = {}                  # (handler, "a0"|"a1") -> counters + observed diff shapes


def _out_stats(name, argname):
    key = (name, argname)
    if key not in out_stats:
        out_stats[key] = {"calls": 0, "noreg": 0, "null": 0, "small": 0, "unreadable": 0,
                          "read": 0, "changed": 0, "same_zero": 0, "same_nonzero": 0,
                          "lost": 0, "shapes": {}}
    return out_stats[key]


def _out_shape(before, after):
    """`@6 00->01` — the changed offsets only, which is what a reader compares against the source."""
    diffs = [(i, before[i], after[i]) for i in range(len(before)) if before[i] != after[i]]
    parts = ["@%d %02x->%02x" % d for d in diffs[:8]]
    if len(diffs) > 8:
        parts.append("+%d more bytes" % (len(diffs) - 8))
    return " ".join(parts)


def _sample_out_args(name):
    """Entry half of the diff: returns [(stats, addr, before_bytes)] for each sampled argument."""
    snaps = []
    try:
        frame = gdb.newest_frame()
        inferior = gdb.selected_inferior()
    except Exception:
        return snaps
    for argname, reg in OUT_ARGS:
        st = _out_stats(name, argname)
        st["calls"] += 1
        try:
            value = int(frame.read_register(reg)) & RET_MASK
        except Exception:
            st["noreg"] += 1
            continue
        if value == 0:
            st["null"] += 1
            continue
        if value < OUT_MIN_POINTER:
            st["small"] += 1
            continue
        try:
            before = bytes(inferior.read_memory(value, OUT_BYTES))
        except Exception:
            # The distinction this whole feature turns on: nothing was READ here. It is NOT
            # evidence that nothing was written.
            st["unreadable"] += 1
            continue
        st["read"] += 1
        snaps.append((st, value, before))
    return snaps


class _Finish(gdb.FinishBreakpoint):
    """One-shot: record what the handler returned and what it wrote, then get out of the way."""

    def __init__(self, frame, name):
        super().__init__(frame, internal=True)
        self.hle_name = name
        self.silent = True
        self.out_snaps = []             # filled by _Counter.stop() right after construction

    def stop(self):
        # Out-bytes first, and unconditionally: the return-value capture has a failure path that
        # returns early, and an out-struct observation must not be lost to it.
        self._capture_out_bytes()
        if VALUES:
            self._capture_return()
        return False

    def _capture_out_bytes(self):
        try:
            inferior = gdb.selected_inferior()
        except Exception:
            for st, _, _ in self.out_snaps:
                st["lost"] += 1
            return
        for st, addr, before in self.out_snaps:
            try:
                after = bytes(inferior.read_memory(addr, len(before)))
            except Exception:
                # Readable on entry, unreadable now (the guest freed or unmapped it). Neither
                # "wrote" nor "did not write" -- a third state, and it says so.
                st["lost"] += 1
                continue
            if after != before:
                st["changed"] += 1
                shape = _out_shape(before, after)
                if shape in st["shapes"] or len(st["shapes"]) < OUT_SHAPE_CAP:
                    st["shapes"][shape] = st["shapes"].get(shape, 0) + 1
                else:
                    st["shapes"]["(+shapes beyond the first %d)" % OUT_SHAPE_CAP] = \
                        st["shapes"].get("(+shapes beyond the first %d)" % OUT_SHAPE_CAP, 0) + 1
            elif any(before):
                # Ambiguous, and the ONE case this instrument cannot resolve: the bytes already held
                # a value, so a handler that wrote exactly that is indistinguishable from one that
                # wrote nothing. Kept apart from same-zero, where "nothing was written" is the far
                # likelier reading.
                st["same_nonzero"] += 1
            else:
                st["same_zero"] += 1
        # Accounted for. Clearing them keeps a later out_of_scope() from counting the same samples
        # a second time as lost.
        self.out_snaps = []

    def _capture_return(self):
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
            return
        if dwarf is not None and rax is not None and dwarf != rax:
            state["src_mismatch"] += 1
        state["src_dwarf" if dwarf is not None else "src_rax"] += 1
        values.setdefault(self.hle_name, {})
        values[self.hle_name][result] = values[self.hle_name].get(result, 0) + 1

    def out_of_scope(self):
        # The handler's frame vanished without a normal return (longjmp, thread
        # teardown). Deliberately NOT counted into finish_failures: gdb calls this
        # more than once per breakpoint on some versions, so it is not a reliable
        # counter. The per-row `captured N/M` below is what makes such a loss
        # visible, and it needs no cooperation from gdb at all.
        #
        # An out-struct sample IS counted lost here, and for the opposite reason: nothing else can
        # see it. `(captured N/M)` derives from the value histogram, so with `--values` off there is
        # no other trace of a sample that never got its second half. gdb calling this twice would
        # over-count `lost` -- an over-report of "this run could not tell", never of "it wrote".
        for st, _, _ in self.out_snaps:
            st["lost"] += 1
        self.out_snaps = []


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
        # Sample BEFORE arming the finish breakpoint: the entry snapshot is a fact about this call
        # whether or not the second half can be armed, and a dropped one is then counted as `lost`
        # rather than silently vanishing.
        snaps = _sample_out_args(self.hle_name) if OUT_BYTES else []
        if VALUES or OUT_BYTES:
            try:
                finish = _Finish(gdb.newest_frame(), self.hle_name)
                finish.out_snaps = snaps
            except Exception as exc:
                # The other half of #2075's diagnosis problem: this site and the decode site above
                # increment ONE counter, so "finish-failures == calls" could not distinguish "never
                # armed" from "armed and could not be decoded". Both now say which they were.
                _note_reason(_reason("FinishBreakpoint()", exc))
                state["finish_failures"] += 1
                for st, _, _ in snaps:
                    st["lost"] += 1
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


# --- load bias (#2605) --------------------------------------------------------------------------
# Every address in SYMS is a LINK-TIME address, because that is what `nm` reports. On an ET_EXEC
# target those are the runtime addresses and there is nothing to do. On a PIE (`ET_DYN`) they are
# offsets from a base the kernel picks per run, and arming them unrelocated puts every breakpoint in
# unmapped memory -- which is not a wrong measurement, it is no measurement, and it reads like one
# (#2593 refused such a target outright; this replaces the refusal).
#
# The base is NOT guessed. gdb disables randomization by default, so `0x555555554000` is what a
# launch under gdb happens to produce on x86-64 Linux today -- and an ordinary attach to a running
# process measured `0x5634df2d1000`. A constant that is right until it silently is not is worse than
# a refusal, so the bias is read from the process and then checked against a second, independent
# source before anything is armed.


def _auxv_entry(pid):
    """The entry point the KERNEL jumped to, from `/proc/<pid>/auxv` -- or None.

    `AT_ENTRY` is the runtime address of the ELF's `e_entry`, so their difference is the load bias by
    definition, for a PIE and a non-PIE alike. This is the same quantity gdb derives for itself
    (see `_gdb_entry_point`), obtained a different way, which is what makes the two a cross-check
    rather than one value read twice.

    Deliberately NOT "the first mapping in `/proc/<pid>/maps`", which is the obvious reading and is
    wrong on the very target this tool exists for: prosper maps the GUEST image low, so a live
    `boot_trace`'s first three map lines are `410000000-...` and the executable's own text does not
    appear until far down the file (measured 2026-08-17 on a PIE build; the real bias that run was
    `0x55b927796000`). A maps-based derivation therefore has to match the executable by path -- with
    `(deleted)` suffixes, symlinks and containers to get right -- while `AT_ENTRY` is one unambiguous
    word the kernel wrote.
    """
    try:
        with open("/proc/%d/auxv" % pid, "rb") as handle:
            data = handle.read()
    except OSError:
        return None
    width = 8                   # x86-64: two 8-byte words per auxv entry
    for off in range(0, len(data) - 2 * width + 1, 2 * width):
        key = int.from_bytes(data[off:off + width], "little")
        if key == AT_ENTRY:
            return int.from_bytes(data[off + width:off + 2 * width], "little")
    return None


def _gdb_entry_point():
    """gdb's own view of the running entry point, from `info files` -- or None.

    Measured on gdb 17.2: before the inferior exists this reports the file's link-time entry, and
    once the main executable has been relocated it reports the runtime one. So it answers the same
    question as `AT_ENTRY` through gdb's displacement logic instead of the kernel's auxv.
    """
    try:
        text = gdb.execute("info files", to_string=True)
    except gdb.error:
        return None
    match = re.search(r"Entry point:\s*(0x[0-9a-fA-F]+)", text)
    return int(match.group(1), 16) if match else None


def _symbol_landing(bias, sample_n=SYMBOL_CHECK_N):
    """Do the biased addresses actually land on the functions they name? `(exact, named, total, misses)`

    This is the check that makes a wrong bias fail LOUDLY instead of producing an empty histogram.
    It asks gdb -- whose symbol table is relocated independently of the arithmetic above -- what
    lives at each address this run is about to arm. A correct bias yields `<name> in section .text`;
    a wrong one yields `No symbol matches` (unmapped) or `<other> + 19` (mid-function). Both are
    caught, and the caller refuses the run.

    `exact` (the address is a function START) is the criterion, not the name: identical-code folding
    can put two handlers at one address, so gdb naming the other one there is not an error. `named`
    is reported alongside it for the reader, never used to refuse.
    """
    total = min(sample_n, len(SYMS))
    if not total:
        return (0, 0, 0, [])
    # Spread across the table rather than taking the first N: a bias that is wrong only for part of
    # the image (a mismatched --binary, a rebuilt target) shows up at the far end.
    picks = [i * len(SYMS) // total for i in range(total)]
    exact = named = 0
    misses = []
    for i in picks:
        addr, name = SYMS[i]
        try:
            text = gdb.execute("info symbol %#x" % (addr + bias), to_string=True).strip()
        except gdb.error as exc:
            text = "info symbol failed: %s" % exc
        head = text.split(" in section ")[0]
        if " in section " in text and not re.search(r"\+ \d+$", head):
            exact += 1
            if name in text:
                named += 1
        else:
            misses.append("%s -> %#x -> %s" % (name, addr + bias, text.splitlines()[0][:120]))
    return (exact, named, total, misses)


def _resolve_load_bias():
    """`(bias, note)` -- and in launch mode this is what first brings the process into existence.

    The hard half of #2605 is that `--launch` arms every breakpoint BEFORE the process runs, which is
    the entire point of the mode (it is the only way to cover init), so there is no base to read yet.
    `starti` resolves that without giving anything up: it stops at the FIRST instruction of the
    process, which on a dynamically linked binary is the loader's `_start` -- measured on gdb 17.2,
    the stop is at `_start () from /lib64/ld-linux-x86-64.so.2` with the executable already mapped
    (`/proc/<pid>/maps` shows it) and gdb's symbols already relocated. Nothing of the program has run,
    so the window still opens where it did before: no instruction of init is lost.

    Rejected alternatives, so they are not re-derived:
      * assuming gdb's disabled-randomization base -- see the block comment above;
      * arming at the entry point, stopping there, then arming the rest -- that DOES lose the
        instructions before the entry point, and `starti` makes it unnecessary;
      * letting gdb relocate symbol-spec breakpoints (`break 'prosper::s_user_getevent'`) instead of
        addresses -- gdb would handle the bias, but the specs are ambiguous for exactly the symbols
        this tool arms: every handler is `static` (nm reports `t`), anonymous-namespace names are
        not unique across translation units under GCC's `_GLOBAL__N_1` mangling, and a name matching
        several locations silently changes what a row counts. Addresses select one function; the
        enumeration already picked which one.
    """
    if ELF_TYPE != ET_DYN:
        return (0, "ET_EXEC: link-time addresses are runtime addresses")
    if MODE == "launch":
        try:
            gdb.execute("starti")
        except gdb.error as exc:
            _abort("this target is a PIE (ET_DYN), so its load bias can only be read from the "
                   "running process, and `starti` -- which stops at the first instruction, before "
                   "anything of the program has run -- failed: %s. `starti` needs gdb 8.1 or newer. "
                   "Either use a newer gdb, or build the target non-PIE "
                   "(cmake -DCMAKE_EXE_LINKER_FLAGS=-no-pie)." % exc)
    try:
        pid = gdb.selected_inferior().pid
    except Exception:
        pid = 0
    if not pid:
        _abort("this target is a PIE (ET_DYN) and there is no live inferior to read a load bias "
               "from, so every breakpoint would land in unmapped memory. Refusing rather than "
               "reporting a window that armed nothing.")
    kernel = _auxv_entry(pid)
    debugger = _gdb_entry_point()
    if kernel is None and debugger is None:
        _abort("this target is a PIE (ET_DYN) and neither /proc/%d/auxv nor gdb's own `info files` "
               "would give the running entry point, so the load bias cannot be derived. Refusing "
               "rather than reporting a window that armed nothing." % pid)
    if kernel is not None and debugger is not None and kernel != debugger:
        # Two independent readings of one quantity. If they disagree, one of them is about a
        # different image than the other, and picking either would be a guess.
        _abort("the running entry point disagrees between its two sources: /proc/%d/auxv AT_ENTRY "
               "is %#x and gdb's `info files` says %#x. One of them is describing a different image "
               "(a mismatched --binary, or an exec this tool did not follow). Refusing rather than "
               "picking one." % (pid, kernel, debugger))
    entry = kernel if kernel is not None else debugger
    bias = entry - ELF_ENTRY
    return (bias, "ET_DYN: %#x = running entry %#x - link-time entry %#x, from %s"
            % (bias, entry, ELF_ENTRY,
               "/proc/<pid>/auxv AT_ENTRY (gdb agrees)" if kernel is not None and debugger is not None
               else "/proc/<pid>/auxv AT_ENTRY" if kernel is not None else "gdb `info files`"))


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

if MODE == "launch" and INFERIOR_TTY:
    # Before `starti`/`run`, because it takes effect when the inferior is created. Without it the
    # inferior inherits gdb's stdout/stderr, which the driver holds on a pipe until the window
    # closes: the emulator's ENTIRE run log would then sit in the driver's memory (a
    # PROSPER_GFXLOG/PROSPER_DBG log reaches 1.5 GB, and exhausting memory on this box takes every
    # concurrent lane's shell down with it). gdb's own stdout stays on the pipe — the result block
    # is a few hundred bytes.
    # `set inferior-tty` OPENS this path and does not create it, so the driver creates and truncates
    # it first — otherwise gdb aborts at `run` with "No such file or directory" before a single
    # breakpoint fires. It also redirects the inferior's stdin. The path is interpolated unquoted,
    # so the driver refuses whitespace (gdb itself handles spaces fine — see the note there; the
    # restriction is this driver's, not gdb's).
    gdb.execute("set inferior-tty %s" % INFERIOR_TTY)

# On a PIE this starts the inferior (stopped at its first instruction) so the base can be read; on
# an ET_EXEC target it costs nothing and returns 0.
LOAD_BIAS, bias_note = _resolve_load_bias()
LAUNCH_STARTED = MODE == "launch" and ELF_TYPE == ET_DYN
exact, named, checked, misses = _symbol_landing(LOAD_BIAS)
print("hle_calls: load bias %#x — %s" % (LOAD_BIAS, bias_note))
if checked and exact != checked:
    # The bias is arithmetic; this is the observation. gdb's symbol table is relocated by gdb's own
    # logic, so it is an independent answer to "is this address the function we think it is" — and
    # it is the check that turns a wrong bias into a named refusal instead of a silent empty window.
    _abort("%d of %d sampled handler addresses do not land on a function start at load bias %#x, so "
           "arming them would measure nothing. The binary passed to --binary is not the one this "
           "process is running, or its load bias could not be derived. Misses: %s"
           % (checked - exact, checked, LOAD_BIAS, "; ".join(misses[:4])))
print("hle_calls: symbol check %d/%d sampled addresses land exactly on a function start "
      "(%d also match by name)" % (exact, checked, named))

armed = 0
for addr, name in SYMS:
    try:
        _Counter(addr + LOAD_BIAS, name)
        armed += 1
    except gdb.error as exc:  # a symbol the running binary does not actually have
        print("hle_calls: skip %s (%s)" % (name, exc))

try:
    _Clock(CLOCK)
except gdb.error as exc:
    raise SystemExit("hle_calls: cannot set the clock breakpoint on %s: %s" % (CLOCK, exc))

print("hle_calls: armed %d handler breakpoints at load bias %#x; window = %d entries of %s "
      "(mode=%s values=%s)" % (armed, LOAD_BIAS, TICKS, CLOCK, MODE, "on" if VALUES else "off"))

if MODE == "launch":
    # Every breakpoint above is already in place, so the window opens at the process's first
    # instruction — which is the whole point of this mode. On a PIE the process is already stopped
    # AT that first instruction (see _resolve_load_bias), so `continue` opens the same window `run`
    # does; nothing of init has executed either way.
    gdb.execute("continue" if LAUNCH_STARTED else "run")
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
if OUT_BYTES:
    source += " out-bytes=%d" % OUT_BYTES
# `elf=`/`load-bias=`/`symbol-check=` are in the header rather than only in the arming line above
# because the driver forwards only this block to stdout. A reader must be able to tell WHERE the
# breakpoints went from the result itself: on a PIE that is the difference between a histogram and a
# window that armed nothing, and the pair of them is also what says the tool ran at all (#2605).
print("clock=%s entries=%d/%d window=%s positive-control=%s armed=%d mode=%s calls=%d "
      "finish-failures=%d exited=%d elf=%s load-bias=%#x symbol-check=%d/%d%s"
      % (CLOCK, state["ticks"], TICKS, "complete" if state["ticks"] >= TICKS else "SHORT",
         control, armed, MODE, state["calls"], state["finish_failures"],
         1 if state["exited"] else 0,
         "ET_DYN" if ELF_TYPE == ET_DYN else "ET_EXEC" if ELF_TYPE == ET_EXEC else "unknown",
         LOAD_BIAS, exact, checked, source))
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
    if OUT_BYTES:
        # One line per sampled argument, then the diff shapes under it. Every counter is printed,
        # including the zeros: "changed=0 same-zero=12" (the handler wrote nothing into a buffer
        # that was zero) and "read=0 unreadable=12" (this tool never saw the bytes) are the two
        # readings this feature exists to keep apart, and a field that vanishes when zero would
        # make the second look like the first.
        printed = False
        for argname, _ in OUT_ARGS:
            st = out_stats.get((name, argname))
            if not st or (st["read"] + st["unreadable"] + st["small"] + st["noreg"]) == 0:
                continue        # NULL on every call: reported by the fallback line below instead
            printed = True
            print("            out %s[%d] calls=%d read=%d changed=%d same-zero=%d same-nonzero=%d "
                  "null=%d small=%d unreadable=%d lost=%d%s"
                  % (argname, OUT_BYTES, st["calls"], st["read"], st["changed"], st["same_zero"],
                     st["same_nonzero"], st["null"], st["small"], st["unreadable"], st["lost"],
                     " noreg=%d" % st["noreg"] if st["noreg"] else ""))
            ranked = sorted(st["shapes"].items(), key=lambda kv: -kv[1])
            for shape, n in ranked[:3]:
                print("              %s  x%d" % (shape, n))
            if len(ranked) > 3:
                print("              +%d more diff shapes" % (len(ranked) - 3))
        if not printed and any((name, a) in out_stats for a, _ in OUT_ARGS):
            print("            out: a0 and a1 were NULL on every call — nothing to sample")
print("distinct=%d total=%d" % (len(rows), sum(c for c, _ in rows)))
print("HLE_CALLS_END")

if MODE == "launch":
    if not state["exited"]:
        gdb.execute("kill")
else:
    gdb.execute("detach")
gdb.execute("quit")
