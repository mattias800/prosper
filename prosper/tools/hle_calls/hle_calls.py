#!/usr/bin/env python3
"""hle_calls — "which Sony functions is the guest calling *right now*?"

Attaches to a live prosper process and returns a per-function call histogram
covering **every** HLE handler in the binary, over a bounded window.

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

Reading the result
------------------
The header line reports the window actually observed:

    clock=prosper::k_usleep entries=400 armed=750

An **empty histogram with a non-zero `entries`** is a real measurement: the
guest entered no HLE handler while the clock advanced 400 times. An empty
histogram with `entries=0` means the window never opened and the run says
nothing — treat it as void, not as a negative.

Include a handler you already know fires as your own positive control before
believing a surprising zero (`sceUserServiceGetEvent` is a good one on almost
any title: engines poll it every frame).

Requirements: Linux, `gdb` with Python, `nm`, and ptrace attach permitted
(`kernel.yama.ptrace_scope=0`). The prosper binary must not be stripped.
"""

import argparse
import os
import re
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
    ap.add_argument("--pid", type=int, required=True, help="live prosper process")
    ap.add_argument("--binary", help="the prosper executable (default: /proc/<pid>/exe)")
    ap.add_argument("--ticks", type=int, default=400,
                    help="how many clock entries the window lasts (default 400)")
    ap.add_argument("--clock", default="prosper::k_usleep",
                    help="symbol whose entries bound the window (default prosper::k_usleep)")
    ap.add_argument("--filter", default=None,
                    help="regex; keep only handlers whose short name matches")
    ap.add_argument("--gdb", default="gdb")
    ap.add_argument("--timeout", type=int, default=600, help="seconds (default 600)")
    args = ap.parse_args()

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
    cmd = [args.gdb, "-p", str(args.pid), "-batch",
           "-x", os.path.join(HERE, "hle_calls_gdb.py")]
    try:
        run = subprocess.run(cmd, env=env, capture_output=True, text=True,
                             timeout=args.timeout)
    except subprocess.TimeoutExpired:
        os.unlink(syms_path)
        sys.exit("hle_calls: gdb timed out after %ds — result is void, not negative"
                 % args.timeout)
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
