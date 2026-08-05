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

    clock=prosper::k_usleep entries=400 armed=750 mode=attach calls=21362 \\
        finish-failures=0 exited=0

An **empty histogram with a non-zero `entries`** is a real measurement: the
guest entered no HLE handler while the clock advanced 400 times. An empty
histogram with `entries=0` means the window never opened and the run says
nothing — treat it as void, not as a negative.

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

That line is also the built-in positive control for the mechanism. `s_user_getevent`
delivers the initial LOGIN event exactly once (returning 0) and then reports
NO_EVENT (`0x80960007`) forever, so a correct capture must show exactly one `0x0`
against many `0x80960007`. If it does not, distrust every other value in the run.
`finish-failures` counts returns that could not be captured at all; a non-zero
value there means the value set is incomplete, not that those calls returned
nothing.

`--values` costs a second trap per call, so pair it with `--filter`.

Requirements: Linux, `gdb` with Python, `nm`, and ptrace attach permitted
(`kernel.yama.ptrace_scope=0`). The prosper binary must not be stripped.
`--launch` passes this process's environment straight through, so set
`PROSPER_*` switches on the hle_calls command line itself.
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
    ap.add_argument("--gdb", default="gdb")
    ap.add_argument("--timeout", type=int, default=600, help="seconds (default 600)")
    args = ap.parse_args()

    if (args.pid is None) == (not args.launch):
        sys.exit("hle_calls: pass exactly one of --pid <pid> or --launch <program> [args...]")
    if args.launch is not None and not args.launch:
        sys.exit("hle_calls: --launch needs a program to run")

    if args.launch:
        binary = args.binary or args.launch[0]
    else:
        binary = args.binary or os.path.realpath("/proc/%d/exe" % args.pid)
    if not os.path.exists(binary):
        sys.exit("hle_calls: no such binary: %s" % binary)

    handlers = enumerate_handlers(binary, args.filter)
    if not handlers:
        sys.exit("hle_calls: found no HLE handlers in %s "
                 "(wrong binary, stripped build, or too narrow a --filter)" % binary)
    print("hle_calls: %d handler symbols in %s" % (len(handlers), binary), file=sys.stderr)

    with tempfile.NamedTemporaryFile("w", suffix=".syms", delete=False) as table:
        for addr, name in handlers:
            table.write("%s %s\n" % (addr, name))
        syms_path = table.name

    env = dict(os.environ)
    env["HLE_CALLS_SYMS"] = syms_path
    env["HLE_CALLS_CLOCK"] = args.clock
    env["HLE_CALLS_TICKS"] = str(args.ticks)
    env["HLE_CALLS_MODE"] = "launch" if args.launch else "attach"
    env["HLE_CALLS_VALUES"] = "1" if args.values else "0"
    env["HLE_CALLS_ORDER"] = str(args.order)
    script = os.path.join(HERE, "hle_calls_gdb.py")
    if args.launch:
        # `--args` must come last; the gdb script issues the `run` itself, after
        # every breakpoint is in place.
        cmd = [args.gdb, "-batch", "-x", script, "--args"] + list(args.launch)
    else:
        cmd = [args.gdb, "-p", str(args.pid), "-batch", "-x", script]
    # Own a process group so a timeout takes the launched emulator down with gdb.
    # Killing gdb alone would orphan the inferior it started, and an orphaned
    # emulator holds the GPU and the dump against every other lane on the box.
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, start_new_session=True)
    try:
        out, err = proc.communicate(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            proc.kill()
        proc.communicate()
        os.unlink(syms_path)
        sys.exit("hle_calls: gdb timed out after %ds — result is void, not negative"
                 % args.timeout)
    run = subprocess.CompletedProcess(cmd, proc.returncode, out, err)
    os.unlink(syms_path)

    body, keep = [], False
    for line in run.stdout.splitlines():
        if line == "HLE_CALLS_BEGIN":
            keep = True
            continue
        if line == "HLE_CALLS_END":
            keep = False
            continue
        if keep:
            body.append(line)
    if not body:
        sys.stderr.write(run.stdout)
        sys.stderr.write(run.stderr)
        sys.exit("hle_calls: gdb produced no result block — VOID, not a negative")
    print("\n".join(body))
    return 0


if __name__ == "__main__":
    sys.exit(main())
