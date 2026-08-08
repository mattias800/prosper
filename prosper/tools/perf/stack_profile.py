#!/usr/bin/env python3
"""stack_profile — name the code location where each thread is blocked.

LINUX ONLY, like its sibling `wait_profile.py`. It drives `gdb -p` and checks /proc/<pid>, so
the Windows and macOS lanes cannot run it at all. Stated in the first line because these two
tools are now the standard way to answer "why is this title stalled" in this project, and half
the lanes can run neither -- a reader on those platforms should learn that here rather than
from an empty report.

`wait_profile.py` answers *how much* each thread waits, cheaply, by reading /proc.
It cannot answer *where*, and the gap is not a detail: a mutex wait and a condition-variable
wait are the SAME `/proc` state, the same `wchan` (`futex_do_wait`) and the same syscall
(202). Measured on a hand-built control with three threads blocked in three known
functions, `/proc` reported two of them identically; only a stack could tell them apart.

So this tool periodically attaches gdb, walks every thread's stack, and aggregates the
result by call site. It is the expensive half of the pair — run it at a low rate alongside
`wait_profile`, not instead of it.

    tools/perf/stack_profile.py --pid 12345 --samples 12 --interval 5

WHAT THIS COSTS, AND WHY THE TOOL INSISTS ON TELLING YOU
--------------------------------------------------------
Every sample STOPS the process (ptrace). On a 4-thread toy that is ~85-90 ms; on a large
binary with many threads it is longer. That is a real perturbation of the thing you are
measuring, so the report always prints the total stopped time and its share of wall clock.
If you are chasing a frame-rate problem, a tool that silently steals 20% of the wall clock
will invent one. Read that line before reading anything else.

THE FAILURE MODE THAT MADE THIS TOOL PRINT SO LOUDLY
-----------------------------------------------------
When gdb cannot attach it prints an error and produces NO stacks -- which is
character-for-character what a process with nothing blocked produces. During development
this tool reported "no blocked threads" for a process that had four, because the attach was
failing and stderr had been discarded. So:

  * a sample that yields no stacks is counted as FAILED, never as "nothing was blocked";
  * the failed count is printed even when it is zero, so its absence is never ambiguous;
  * if EVERY sample failed the tool exits non-zero and says so, rather than printing an
    empty report that reads like a clean result.

A tool that cannot distinguish "I did not measure" from "there was nothing to measure" is
not a negative result, it is a void one.

RUN THIS WHERE THE TARGET RUNS
-------------------------------
gdb must be on the same side of the container boundary as the process. Measured both ways
on this box:

    process on host      + gdb on host       -> works
    process in distrobox + gdb in distrobox  -> works
    process on host      + gdb in distrobox  -> ptrace: Operation not permitted

So for a `prosper-app` launched inside distrobox, run this inside the same container; for a
host process, run it on the host. The rule is "same namespace", not "always the host" -- and
getting that backwards costs nothing but confusion, because per the section above the denial
produces an empty profile that reads like a healthy process. The tool detects this specific
error and names it.
"""

from __future__ import annotations

import argparse
import collections
import os
import re
import shutil
import subprocess
import sys
import time

# gdb frames that are plumbing rather than an answer. A stack whose top frame is
# `futex_wait` tells you nothing you did not already know from /proc; the first frame
# BELOW this set is the one that names the lock.
PLUMBING = re.compile(
    r"^_*("
    r"futex\w*|\w*futex_\w+|"
    r"pthread_cond_\w+|pthread_mutex_\w+|pthread_rwlock_\w+|pthread_join\w*|"
    r"lll_\w+|"
    r"sem_(wait|timedwait)\w*|new_sem_wait\w*|"
    r"clock_nanosleep\w*|nanosleep|usleep|sleep|"
    r"poll|ppoll|select|pselect|epoll_wait|epoll_pwait|read|recv|recvmsg|"
    r"syscall\w*|internal_syscall\w*|syscall_cancel\w*|internal_syscall_cancel\w*|"
    r"libc_\w*(read|poll|select|wait)\w*|GI_\w+"
    r")$"
)

# glibc reports versioned symbols as `pthread_mutex_lock@@GLIBC_2.2.5`. Matching against
# the raw string silently fails every `$`-anchored pattern above, which does not produce an
# error -- it produces a plausible-looking answer naming a libc internal instead of the
# caller. Caught only because the control had known-correct answers to disagree with.
VERSION_SUFFIX = re.compile(r"@+[A-Za-z0-9_.]+$")


def strip_symbol(fn: str) -> str:
    return VERSION_SUFFIX.sub("", fn).strip()

FRAME_RE = re.compile(
    r"^#(?P<n>\d+)\s+"
    r"(?:0x[0-9a-fA-F]+\s+in\s+)?"
    r"(?P<fn>[^(]+?)\s*\("
)
THREAD_RE = re.compile(
    r'^Thread\s+\d+\s+\(Thread\s+\S+\s+\(LWP\s+(?P<lwp>\d+)\)\s*(?:"(?P<name>[^"]*)")?'
)


def gdb_stacks(pid: int, gdb: str, timeout: float):
    """One sample. Returns (threads, stopped_seconds, error_or_None).

    `threads` maps lwp -> (thread_name, [frame function names, outermost last]).
    An empty mapping with error=None is still reported as a failure by the caller: see
    the module docstring on why "no stacks" must never be read as "nothing blocked".
    """
    cmd = [
        gdb, "-p", str(pid), "-batch", "-nx",
        "-iex", "set auto-solib-add off",   # do not pay for shared-library symbols
        "-iex", "set confirm off",
        "-iex", "set pagination off",
        "-ex", "thread apply all bt",
    ]
    started = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {}, time.monotonic() - started, f"gdb exceeded {timeout:g}s"
    except FileNotFoundError:
        return {}, 0.0, f"{gdb} not found"
    stopped = time.monotonic() - started
    blob = proc.stdout + proc.stderr

    if "Operation not permitted" in blob:
        return {}, stopped, (
            "ptrace: Operation not permitted -- run gdb on the SAME side of the container "
            "boundary as the target (a container cannot ptrace a host process; "
            "in-container gdb against an in-container prosper-app works)"
        )
    if "No such process" in blob or "not being run" in blob:
        return {}, stopped, f"pid {pid} is gone"

    threads, lwp, name, frames = {}, None, None, []
    for line in blob.splitlines():
        m = THREAD_RE.match(line)
        if m:
            if lwp is not None:
                threads[lwp] = (name, frames)
            lwp, name, frames = int(m.group("lwp")), (m.group("name") or "?"), []
            continue
        f = FRAME_RE.match(line.strip())
        if f and lwp is not None:
            fn = strip_symbol(f.group("fn"))
            if fn and not fn.startswith("0x"):
                frames.append(fn)
    if lwp is not None:
        threads[lwp] = (name, frames)

    if not threads:
        detail = next((l for l in blob.splitlines() if "rror" in l or "ptrace" in l), "")
        return {}, stopped, f"gdb produced no stacks{': ' + detail.strip() if detail else ''}"
    return threads, stopped, None


def blocking_site(frames, depth: int):
    """The first non-plumbing frame, plus `depth-1` callers beneath it.

    Returns None when a thread's whole stack is plumbing -- that is a real outcome
    (a pure library wait with no symbolized caller), reported as such rather than
    silently bucketed with something else.
    """
    for i, fn in enumerate(frames):
        if not PLUMBING.match(fn):
            return " <- ".join(frames[i:i + depth])
    return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Name where each thread is blocked, by periodic gdb stack sampling.")
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--samples", type=int, default=10, help="number of stack samples")
    ap.add_argument("--interval", type=float, default=5.0,
                    help="seconds between samples (keep high; each one stops the process)")
    ap.add_argument("--depth", type=int, default=3, help="caller frames per reported site")
    ap.add_argument("--top", type=int, default=3, help="sites listed per thread")
    ap.add_argument("--gdb", default=os.environ.get("PROSPER_GDB", "gdb"))
    ap.add_argument("--gdb-timeout", type=float, default=60.0)
    args = ap.parse_args()

    if shutil.which(args.gdb) is None:
        print(f"error: {args.gdb} not found. Install gdb, or pass --gdb.", file=sys.stderr)
        return 2
    if not os.path.isdir(f"/proc/{args.pid}"):
        print(f"error: no process {args.pid}", file=sys.stderr)
        return 2

    # thread key -> Counter of blocking sites; plus how many samples saw the thread at all.
    sites = collections.defaultdict(collections.Counter)
    seen = collections.Counter()
    ok = failed = 0
    errors = collections.Counter()
    stopped_total = 0.0

    wall_start = time.monotonic()
    for i in range(args.samples):
        threads, stopped, err = gdb_stacks(args.pid, args.gdb, args.gdb_timeout)
        stopped_total += stopped
        if err or not threads:
            failed += 1
            errors[err or "gdb produced no stacks"] += 1
        else:
            ok += 1
            for lwp, (name, frames) in threads.items():
                key = f"{name}/{lwp}"
                seen[key] += 1
                site = blocking_site(frames, args.depth)
                sites[key][site if site else "(no symbolized caller -- pure library wait)"] += 1
        if i + 1 < args.samples:
            time.sleep(args.interval)
    wall = time.monotonic() - wall_start

    print(f"\nstack_profile: pid {args.pid}")
    print(f"  samples: {ok} succeeded, {failed} FAILED  (of {args.samples} attempted)")
    for msg, n in errors.most_common():
        print(f"    failure x{n}: {msg}")
    # Perturbation is printed before any finding, because it bounds how much of the
    # finding the tool itself caused.
    share = (stopped_total / wall * 100.0) if wall > 0 else 0.0
    print(f"  process stopped by this tool: {stopped_total:.2f}s of {wall:.2f}s wall "
          f"({share:.1f}% -- this is perturbation, not a measurement of the program)")
    if share > 10.0:
        print("    WARNING: over 10% of wall clock was spent stopped. Raise --interval "
              "before trusting any timing conclusion drawn alongside this run.")

    if ok == 0:
        print("\n  NO SAMPLES SUCCEEDED -- this is not a result. Nothing below is a finding.")
        return 1

    print(f"\n  {'thread':<28} {'seen':>5}  blocking site (top {args.top}, by sample count)")
    print(f"  {'-' * 28} {'-' * 5}  {'-' * 60}")
    for key in sorted(seen, key=lambda k: -seen[k]):
        first = True
        for site, n in sites[key].most_common(args.top):
            pct = n / seen[key] * 100.0
            label = f"{key:<28} {seen[key]:>5}" if first else f"{'':<28} {'':>5}"
            print(f"  {label}  {pct:5.1f}%  {site}")
            first = False
    print(f"\n  Shares are per thread: n/(samples in which THAT thread was seen), so a "
          f"thread\n  that started late is not diluted by samples it could not appear in.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
