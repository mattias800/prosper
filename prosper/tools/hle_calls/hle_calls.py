#!/usr/bin/env python3
"""hle_calls — "which Sony functions is the guest calling *right now*?"

Returns a per-function call histogram covering **every** HLE handler in the
binary, over a bounded window — either on a live prosper process (`--pid`) or on
one this tool starts itself (`--launch`), and optionally with each handler's
**return values** rather than only its call count (`--values`).

Why this is not `PROSPER_SVCLOG=1`: `svc_log` is opt-in per handler. On the
revision this tool was written against, roughly 1,035 handler registrations were served
by 94 `svc_log` call sites, so most of the implemented Sony surface is invisible
to the svc log. "The stall shows only sceUserServiceGetEvent in the
log" therefore bounds the *instrumented* surface, not the guest's real traffic —
and that gap is exactly what you need closed when a title goes quiet and you
want to know what it is polling. `PROSPER_PROGRESS_UNIMPL` has the complementary
gap: it counts only the handlers that do NOT exist.

This tool needs no emulator rebuild and no gating env var: it enumerates the
handler symbols out of the prosper binary with `nm` (they all share the six
`unsigned long` HLE signature) and counts them with non-stopping gdb Python
breakpoints.

Usage
-----
    python3 tools/hle_calls/hle_calls.py --pid $(pgrep -x boot_trace) \\
        --binary build-linux/boot_trace --ticks 400

    # only the libSce service layer, not kernel/graphics/file
    ... --filter '^s_'

    # cover INIT: start the process under gdb with every breakpoint already
    # armed, and record what each handler answered
    python3 tools/hle_calls/hle_calls.py --ticks 30 --values --filter '^s_' \\
        --launch build-linux/boot_trace <DUMP_ROOT>/PPSA05325-app0

Reading the result
------------------
The header line reports the window actually observed:

    clock=prosper::k_usleep entries=400/400 window=complete \\
        positive-control=absent(attach:login-consumed-pre-window) armed=750 mode=attach \\
        calls=21362 finish-failures=0 exited=0 elf=ET_EXEC load-bias=0x0 symbol-check=8/8

An **empty histogram with a non-zero `entries`** is a real measurement: the
guest entered no HLE handler while the clock advanced 400 times. An empty
histogram with `entries=0` means the window never opened and the run says
nothing — treat it as void, not as a negative. `window=SHORT` means gdb stopped
before the clock reached `--ticks` (a signal it stops for, or the inferior
exiting), so the histogram is truncated rather than finished.

Include a handler you already know fires as your own positive control before
believing a surprising zero (`sceUserServiceGetEvent` is a good one on almost
any title: engines poll it every frame).

Init coverage, and how to check it
----------------------------------
`--pid` cannot see init: the process is already past it by the time anything can
attach, and arming hundreds of breakpoints takes longer still. `--launch` closes
that — gdb loads the binary, every breakpoint goes in, and only then does the
process run, so the window starts at the first instruction.

"Covered init" is not something to assume, so the result carries its own check:
`first-calls` lists the leading calls **in call order**. A window that really
covered init leads with allocator / module-setup handlers; one that opened late
leads with the frame-loop pollers (`pad_read_state`, `s_user_getevent`). A
histogram alone cannot tell those apart — that is exactly how an earlier attach
arm looked like a negative when it was void.

Return values, and its positive control
---------------------------------------
A count says *which* function ran, not whether prosper answered it correctly, and
a registered-but-mismodelled call is invisible to every absence check. `--values`
adds a per-handler histogram of return values, so a wrong constant is visible:

       36  s_user_getevent   ret 0x80960007 x35, 0x0 x1

That line is also the built-in positive control for the mechanism — but **what it
should show depends on the mode.** `s_user_getevent` delivers the initial LOGIN
event exactly once *per process* (returning 0) and reports NO_EVENT
(`0x80960007`) forever after, and that "once per process" is a function-local
`static` in `src/hle/service/hle_service.cpp`, not a per-window property:

  * under `--launch` the window opens at the first instruction, so the single
    `0x0` falls inside it — a correct capture shows exactly one, and its absence
    is a real signal;
  * under `--pid` init consumed that `0x0` before gdb could attach, so a correct
    capture shows `0x80960007` and **no `0x0` at all**. That is the normal
    observation on any settled title; do not discard such a run.

More than one `0x0` is impossible in either mode. Rather than leave any of this
to be re-derived, the header states the verdict itself in `positive-control=`:

    ok                                    the mechanism captured the LOGIN
    absent(attach:login-consumed-pre-window)  normal for --pid; no control either way
    VIOLATED(...)                         distrust every value in this run
    VOID(...)                             returns were lost; neither confirmed nor refuted
    not-in-filter / not-called            no control in this run at all
    unchecked(values-off)                 --values is off, so there is nothing to check

Anything other than `ok` also prints a one-line `positive-control-note:` saying
whether the run is usable.

Completeness is per row, not per run. `finish-failures` in the header counts
captures that could not be **armed or decoded**, and it does not see the more
common loss: a handler that leaves its frame without returning normally (longjmp,
thread teardown) loses its value silently. So each row prints `(captured N/M)`
whenever fewer values were recorded than calls counted — that is the field to
read before calling a value set complete. Every failure also names itself on a
`finish-failure-reasons:` line; a bare count is what let #2075 stay undiagnosed.

`value-source=dwarf:N,rax:M` says which of the two reads answered: gdb's
DWARF-typed `return_value`, or the SysV integer return register. `dwarf:0` is
normal rather than degraded — prosper's default build type is `Release`, which
carries no debug info, and that is precisely the configuration on which this
feature used to record nothing at all (#2075). When both answer they must agree,
and a disagreement prints a loud `value-source-MISMATCH:` line.

One limit to know before believing a number: on a handler that longjmps back
into its own caller, gdb's finish breakpoint can capture the *landing* value as
if it were a return (measured on gdb 17.2), so a `(captured N/M)` row's other
values are suspect rather than merely fewer.

`--values` costs a second trap per call, so pair it with `--filter`.

Out-structs — the half a return value cannot reach
--------------------------------------------------
`--values` records the return **register**. The recurring defect on this codebase
is the one that register cannot see: *success returned, out-struct never
written*. `sceSystemServiceGetStatus` aliased to the wrong handler,
`sceSystemServiceGetDisplaySafeAreaInfo` leaving `ratio` at 0,
`sceNpEntitlementAccessGetSkuFlag` unregistered with its out pointer untouched,
`sceNpTrophy2GetTrophyInfoArray` handing back a garbage count — every one of them
returns `0`, and every one is invisible in a value census.

`--out-bytes N` snapshots N bytes at whichever of a0/a1 is a readable pointer,
**before and after** each call, and reports what changed:

    5  s_syss_getstatus   ret 0x0 x5
            out a0[12] calls=5 read=5 changed=5 same-zero=0 same-nonzero=0 \\
                null=0 small=0 unreadable=0 lost=0
              @5 00->01 @6 00->01  x5

The before/after form is what makes it self-validating, and every counter is
printed including the zeros: `changed=0 same-zero=12` (the handler wrote nothing
into a buffer that was zero) and `read=0 unreadable=12` (this tool never saw the
bytes) are the two readings that must never be confused, and a field that
disappeared when zero would let the second read as the first.

`same-nonzero` is the one genuinely ambiguous state: the bytes already held a
value, so a handler that wrote exactly that is indistinguishable from one that
wrote nothing — a guest polling into a buffer it does not clear produces it.
`lost` means the entry snapshot had no second half (the frame left without
returning, or the memory went away): neither reading, and counted apart from
both.

The launched process's own output
--------------------------------
`--launch` passes this process's environment straight through, so set `PROSPER_*`
switches on the hle_calls command line itself — which makes it easy to launch an
emulator that prints a 1.5 GB run log. That output does **not** come back through
this tool: gdb's own stdout is on a pipe (the result block is tiny), and the
inferior's stdin/stdout/stderr are redirected to `--inferior-log`, which defaults
to `/dev/null`. Pass a real path to keep the run log:

    ... --inferior-log ~/work/boot.log --launch build-linux/boot_trace <DUMP_ROOT>/...

The file is created and truncated before gdb starts. The inferior's *stdin* is
redirected too, so a launched program that reads stdin sees EOF. `--inferior-log`
applies to `--launch` only; passing it with `--pid` is an error, because there is
no inferior whose output this tool controls.

PIE targets, and where the breakpoints went
-------------------------------------------
`nm` reports link-time addresses. On an `ET_EXEC` binary those are the runtime
addresses; on a PIE they are offsets from a base the kernel picks per run, so the
tool reads that base out of the live process and adds it before arming — under
`--pid` from the attached process, and under `--launch` after a `starti` that
stops at the first instruction, before anything of the program has run, so no
init coverage is lost. Ubuntu's gcc defaults to `-pie` and Fedora's does not, so
both shapes occur for the same source (#2605).

The header reports the outcome — `elf=ET_DYN load-bias=0x555555554000
symbol-check=8/8`. The bias is `running entry - link-time entry`, read from
`/proc/<pid>/auxv` and cross-checked against gdb's own `info files`; it is never
assumed, because gdb disables randomization and its `0x555555554000` is right
until a plain `--pid` attach (measured: `0x5634df2d1000`) makes it silently
wrong. `symbol-check` is the observation behind it: gdb was asked what lives at a
sample of the addresses about to be armed, and each named a function *start*. A
wrong bias — or a `--binary` that is not the image the process is running —
fails that and the run is refused by name rather than measuring nothing.

Requirements: Linux, `gdb` with Python, `nm`, and ptrace attach permitted
(`kernel.yama.ptrace_scope=0`, for `--pid`; `--launch` runs its own child). The
prosper binary must not be stripped — the handler enumeration below reads the
symbol table. Debug info is not required, `--values` included.
"""

import argparse
import os
import re
import signal
import subprocess
import sys
import tempfile

HLE_SIGNATURE = ("(unsigned long, unsigned long, unsigned long, "
                 "unsigned long, unsigned long, unsigned long)")
# --out-bytes ceiling. Two reads of this width per call per argument, and the whole snapshot is held
# per in-flight call; a struct worth inspecting here is tens of bytes, not kilobytes.
OUT_BYTES_MAX = 256
# ELF `e_type`. ET_EXEC's link-time addresses are its runtime addresses; ET_DYN's are offsets from a
# base the kernel picks per run, which is the whole of #2605.
ET_EXEC = 2
ET_DYN = 3
HERE = os.path.dirname(os.path.abspath(__file__))


def enumerate_handlers(binary, name_filter):
    """Every `prosper::`-scoped function with the six-arg HLE signature."""
    nm = subprocess.run(["nm", binary], capture_output=True, text=True)
    if nm.returncode != 0:
        sys.exit("hle_calls: nm failed on %s: %s" % (binary, nm.stderr.strip()))
    mangled = []
    for line in nm.stdout.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[1] in ("t", "T"):
            mangled.append((parts[0], parts[2]))
    if not mangled:
        sys.exit("hle_calls: no text symbols in %s (stripped build?)" % binary)

    demangle = subprocess.run(["c++filt"], input="\n".join(m[1] for m in mangled),
                              capture_output=True, text=True)
    demangled = demangle.stdout.splitlines()
    if len(demangled) != len(mangled):
        sys.exit("hle_calls: c++filt returned %d names for %d symbols"
                 % (len(demangled), len(mangled)))

    pattern = re.compile(name_filter) if name_filter else None
    seen, out = set(), []
    for (addr, _), pretty in zip(mangled, demangled):
        if HLE_SIGNATURE not in pretty or "prosper::" not in pretty:
            continue
        if "[clone .cold]" in pretty:
            continue
        short = pretty.split(HLE_SIGNATURE)[0]
        short = short.replace("prosper::", "").replace("(anonymous namespace)::", "")
        # A lambda/thunk whose *parameter list* merely contains the signature is
        # not itself a handler; a real handler's name ends right at the '('.
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", short):
            continue
        if pattern and not pattern.search(short):
            continue
        if short in seen:
            continue
        seen.add(short)
        out.append((addr, short))
    return out


def read_elf_geometry(binary):
    """`(e_type, e_entry)` -- what decides whether a link-time address is a runtime address (#2605).

    This tool arms breakpoints at the RAW addresses `nm` reports, which are link-time addresses. In a
    non-PIE executable (`ET_EXEC`) they are the runtime addresses too. Under a toolchain that
    defaults to `-pie` (Ubuntu's gcc does; Fedora's does not) the same addresses are offsets from a
    load base chosen at run time, and arming them unrelocated puts every breakpoint in unmapped
    memory:

        Cannot insert breakpoint 3.
        Cannot access memory at address 0x1149

    That is what a GitHub `ubuntu-24.04` runner produced from the same fixture that passes on Fedora.
    #2593 replaced the spew with a refusal; this function replaces the refusal with the load bias,
    which the gdb side derives from the LIVE process (`hle_calls_gdb.py`, `_resolve_load_bias`) --
    `e_entry` here is one of the two halves of that subtraction, the other being the kernel's own
    `AT_ENTRY`. Nothing is guessed: in particular the base gdb picks with randomization disabled is
    never assumed, because it is right until it silently is not (a real attach measured
    `0x5634df2d1000`, nowhere near gdb's `0x555555554000`).

    Returns `(None, None)` for anything that is not an ELF, and lets the later steps produce their
    own message -- `nm` is next and says so plainly.
    """
    try:
        with open(binary, "rb") as handle:
            header = handle.read(64)
    except OSError as exc:
        sys.exit("hle_calls: cannot read %s: %s" % (binary, exc))
    if len(header) < 32 or header[:4] != b"\x7fELF":
        return (None, None)
    order = "little" if header[5] == 1 else "big"
    e_type = int.from_bytes(header[16:18], order)
    # e_entry sits at 0x18 in both classes; only its width differs (ELF32 4 bytes, ELF64 8).
    width = 8 if header[4] == 2 else 4
    if len(header) < 0x18 + width:
        return (e_type, None)
    e_entry = int.from_bytes(header[0x18:0x18 + width], order)
    return (e_type, e_entry)


def _prepare_inferior_log(path):
    """Absolute path for gdb's `set inferior-tty`, with the file ready to be opened.

    gdb *opens* this path for the inferior and does not create it — a missing file aborts
    the run at `run` with "No such file or directory", before a single breakpoint fires —
    so create it here. Truncate a regular file so a run's log is that run's; leave anything
    else (`/dev/null`, a tty, a fifo) exactly as it is.
    """
    resolved = os.path.abspath(path or os.devnull)
    # Refuse a whitespace path. NOT because gdb would mangle it — measured on gdb 17.2,
    # `set inferior-tty` is an add_setshow_filename_cmd and keeps the whole path, spaces
    # included (`show inferior-tty` echoes it back intact), and it tilde-expands too. The
    # reason is this driver: it interpolates the path into a gdb command line and does not
    # quote it, and gdb's own quoting rules for filename settings are not something to rely
    # on silently. Rejecting is a one-line conservatism with a clear message; the day a real
    # path needs a space, quote it here and delete this check rather than trusting the
    # interpolation.
    if any(c.isspace() for c in resolved):
        sys.exit("hle_calls: --inferior-log path must not contain whitespace (this driver "
                 "interpolates it into a gdb command line unquoted): %s" % resolved)
    # A DIRECTORY fails opaquely otherwise (#2054): the isfile() guard below skips creation, gdb
    # then fails inside `run` with a message that never names --inferior-log, and the reader is left
    # debugging the launch. Named here, where the cause is known.
    if os.path.isdir(resolved):
        sys.exit("hle_calls: --inferior-log %s is a directory; pass a file path (gdb opens this "
                 "path for the inferior's stdio and will fail inside `run` with a message that "
                 "does not mention --inferior-log)" % resolved)
    try:
        if not os.path.exists(resolved) or os.path.isfile(resolved):
            with open(resolved, "w"):
                pass
    except OSError as exc:
        sys.exit("hle_calls: cannot open --inferior-log %s: %s" % (resolved, exc))
    return resolved


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pid", type=int, help="live prosper process (mutually exclusive with --launch)")
    ap.add_argument("--launch", nargs=argparse.REMAINDER,
                    help="start this program under gdb instead of attaching, with every "
                         "breakpoint already armed, so the window covers init. Must be last: "
                         "everything after it is the program and its arguments.")
    ap.add_argument("--values", action="store_true",
                    help="also record each handler's return values (costs a second trap per "
                         "call — pair it with --filter)")
    ap.add_argument("--out-bytes", type=int, default=0, metavar="N",
                    help="also snapshot N bytes at whichever of a0/a1 is a readable pointer, "
                         "before and after each call, and report what CHANGED. This is the half a "
                         "return-value census cannot reach: the recurring bug shape here is "
                         "'success returned, out-struct never written', which returns 0 every time. "
                         "Costs two memory reads per call per argument — pair it with --filter.")
    ap.add_argument("--order", type=int, default=24,
                    help="how many leading calls to report in call order (default 24); this is "
                         "what shows whether the window covered init")
    ap.add_argument("--binary", help="the prosper executable (default: /proc/<pid>/exe, "
                                     "or the --launch program)")
    ap.add_argument("--ticks", type=int, default=400,
                    help="how many clock entries the window lasts (default 400)")
    ap.add_argument("--clock", default="prosper::k_usleep",
                    help="symbol whose entries bound the window (default prosper::k_usleep)")
    ap.add_argument("--filter", default=None,
                    help="regex; keep only handlers whose short name matches")
    ap.add_argument("--inferior-log", default=None,
                    help="--launch only: where the LAUNCHED program's stdin/stdout/stderr go "
                         "(default /dev/null). Its output is deliberately not returned through "
                         "this tool — an emulator run log reaches gigabytes and would otherwise "
                         "sit in this process's memory. The file is created and truncated first.")
    ap.add_argument("--gdb", default="gdb")
    ap.add_argument("--timeout", type=int, default=600, help="seconds (default 600)")
    args = ap.parse_args()

    # `--launch` with nothing after it parses as an empty REMAINDER, which is falsy — check that case
    # first so it gets its own message instead of the generic "pass exactly one" one.
    if args.launch is not None and not args.launch:
        sys.exit("hle_calls: --launch needs a program to run")
    if (args.pid is None) == (not args.launch):
        sys.exit("hle_calls: pass exactly one of --pid <pid> or --launch <program> [args...]")
    # Refuse rather than silently ignore: in --pid mode the emulator is not this tool's child,
    # it already owns its stdout/stderr, and accepting the flag would imply we redirected them.
    # A refusal rather than a clamp: a run that quietly sampled a different width than the one asked
    # for would put an unfalsifiable number in front of a reader.
    if args.out_bytes < 0 or args.out_bytes > OUT_BYTES_MAX:
        sys.exit("hle_calls: --out-bytes must be between 0 (off) and %d; got %d"
                 % (OUT_BYTES_MAX, args.out_bytes))
    if not args.launch and args.inferior_log is not None:
        sys.exit("hle_calls: --inferior-log applies to --launch only — a --pid target owns its "
                 "own output and this tool cannot redirect it")

    if args.launch:
        binary = args.binary or args.launch[0]
    else:
        binary = args.binary or os.path.realpath("/proc/%d/exe" % args.pid)
    if not os.path.exists(binary):
        sys.exit("hle_calls: no such binary: %s" % binary)
    # Whether `nm`'s addresses are runtime addresses, and what the gdb side needs to find out if they
    # are not. See read_elf_geometry().
    elf_type, elf_entry = read_elf_geometry(binary)
    if elf_type == ET_DYN and elf_entry is None:
        sys.exit("hle_calls: %s is ET_DYN but its ELF header carries no entry point, so the load "
                 "bias cannot be derived. Build the target non-PIE "
                 "(cmake -DCMAKE_EXE_LINKER_FLAGS=-no-pie) and re-run." % binary)

    handlers = enumerate_handlers(binary, args.filter)
    if not handlers:
        sys.exit("hle_calls: found no HLE handlers in %s "
                 "(wrong binary, stripped build, or too narrow a --filter)" % binary)
    print("hle_calls: %d handler symbols in %s (%s)"
          % (len(handlers), binary,
             "ET_DYN -- the load bias is read from the live process before arming"
             if elf_type == ET_DYN else
             "ET_EXEC -- link-time addresses are runtime addresses" if elf_type == ET_EXEC else
             "not an ELF header this tool recognises"),
          file=sys.stderr)

    with tempfile.NamedTemporaryFile("w", suffix=".syms", delete=False) as table:
        for addr, name in handlers:
            table.write("%s %s\n" % (addr, name))
        syms_path = table.name

    env = dict(os.environ)
    if args.launch:
        env["HLE_CALLS_INFERIOR_TTY"] = _prepare_inferior_log(args.inferior_log)
    else:
        # This process's environment is passed straight through, so drop any inherited copy
        # rather than let it look like attach mode redirects something. It does not.
        env.pop("HLE_CALLS_INFERIOR_TTY", None)
    env["HLE_CALLS_SYMS"] = syms_path
    env["HLE_CALLS_CLOCK"] = args.clock
    env["HLE_CALLS_TICKS"] = str(args.ticks)
    env["HLE_CALLS_MODE"] = "launch" if args.launch else "attach"
    env["HLE_CALLS_VALUES"] = "1" if args.values else "0"
    env["HLE_CALLS_ORDER"] = str(args.order)
    env["HLE_CALLS_OUT_BYTES"] = str(args.out_bytes)
    # The gdb side needs both halves of the bias subtraction: this file's link-time entry point, and
    # the running process's real one. It reads the second itself; only the first is knowable here.
    env["HLE_CALLS_ELF_TYPE"] = str(elf_type if elf_type is not None else 0)
    env["HLE_CALLS_ELF_ENTRY"] = str(elf_entry if elf_entry is not None else 0)
    script = os.path.join(HERE, "hle_calls_gdb.py")
    if args.launch:
        # `--args` must come last; the gdb script issues the `run` itself, after
        # every breakpoint is in place.
        cmd = [args.gdb, "-batch", "-x", script, "--args"] + list(args.launch)
    else:
        cmd = [args.gdb, "-p", str(args.pid), "-batch", "-x", script]
    # These pipes hold GDB's own output only, and that is deliberate: the result block is a few
    # hundred bytes, while a launched emulator's run log reaches gigabytes and would otherwise be
    # inherited straight onto this pipe and buffered here until the window closed. The inferior's
    # stdin/stdout/stderr go to --inferior-log instead (HLE_CALLS_INFERIOR_TTY above).
    # Popen rather than subprocess.run so a timeout can kill the whole tree. Note what actually
    # reaps the launched emulator: gdb sets PTRACE_O_EXITKILL on a process it STARTED, so the
    # inferior dies when gdb dies even though it sits in its own process group (measured:
    # gdb pid==pgrp, inferior pid==its own pgrp, both in gdb's session). The killpg below is the
    # belt to that braces and also covers anything else gdb spawned. It does NOT apply in --pid
    # mode, correctly: there the emulator is not gdb's child and must survive a detach.
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, start_new_session=True)

    def _kill_tree():
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            proc.kill()
        try:
            proc.communicate()
        except Exception:
            pass
        try:
            os.unlink(syms_path)
        except OSError:
            pass

    try:
        out, err = proc.communicate(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        _kill_tree()
        sys.exit("hle_calls: gdb timed out after %ds — result is void, not negative"
                 % args.timeout)
    except BaseException:
        # KeyboardInterrupt included, and it is the likely way a human ends a long run.
        # start_new_session detaches gdb from the terminal, so Ctrl-C does NOT reach it:
        # without this the tool exits and leaves gdb plus the emulator reparented to init —
        # exactly the orphan the session/group handling exists to prevent.
        _kill_tree()
        raise
    run = subprocess.CompletedProcess(cmd, proc.returncode, out, err)
    os.unlink(syms_path)

    body, keep, notes, aborts = [], False, [], []
    for line in run.stdout.splitlines():
        if line == "HLE_CALLS_BEGIN":
            keep = True
            continue
        if line == "HLE_CALLS_END":
            keep = False
            continue
        if keep:
            body.append(line)
        elif line.startswith("HLE_CALLS_ABORT:"):
            aborts.append(line)
        elif line.startswith("hle_calls:"):
            # What the gdb side resolved and armed, forwarded rather than swallowed. Only the
            # result block reaches stdout, so before this these lines — the load bias it derived,
            # the symbols it could not arm — were visible ONLY when the run produced nothing at all,
            # i.e. exactly when it was too late to be diagnostic.
            notes.append(line)
    for line in notes:
        print(line, file=sys.stderr)
    if not body:
        # An abort names its own cause, so print it INSTEAD of the raw gdb transcript: the transcript
        # is what a reader had to search before, and "VOID, not a negative" is true but not useful
        # when the tool already knows precisely what was wrong.
        if aborts:
            sys.stderr.write(run.stderr)
            sys.exit("\n".join(aborts))
        sys.stderr.write(run.stdout)
        sys.stderr.write(run.stderr)
        sys.exit("hle_calls: gdb produced no result block — VOID, not a negative")
    print("\n".join(body))
    return 0


if __name__ == "__main__":
    sys.exit(main())
