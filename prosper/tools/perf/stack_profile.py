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

# An UNRESOLVED frame. `set auto-solib-add off` (which this tool passes, to avoid paying for
# shared-library symbols on every attach) makes gdb print `??` for any frame in a library it
# did not load. `??` is not an answer and must never be reported as the blocking site -- but it
# is also not plumbing to be silently swallowed, because a stack that is ENTIRELY `??` means the
# tool has told you nothing and should say so. Both cases are handled in blocking_site().
#
# Found by review, not by the control: the control ran where gdb resolved libc, so its frames
# came back named (`pthread_cond_wait@@GLIBC_2.3.2`) and `??` never appeared. The first live
# prosper run printed `?? <- ??` as a blocking site for four threads and I read past it.
UNRESOLVED = re.compile(r"^\?+$")

# Sentinel: distinguishes "nothing in this stack resolved" from "this stack is all plumbing".
# They look the same in a report and mean opposite things -- the first is a tool failure for
# that thread, the second is a real finding about the program.
UNRESOLVED_ONLY = "(no symbolized frame -- gdb resolved nothing in this stack)"

# A thread whose stack produced NO frames at all. Distinct again: `UNRESOLVED_ONLY` means gdb
# walked the stack and could not name the frames; this means nothing was parsed for the thread.
# Both are tool failures for that thread and neither is a finding about the program -- which is
# the whole reason they are not allowed to share the "pure library wait" label.
NO_FRAMES = "(no frames parsed for this thread -- tool failure, not a finding)"

# C++ wait wrappers. These are plumbing exactly like the C ones, but they are TEMPLATED --
# `std::condition_variable::__wait_until_impl<std::chrono::duration<long, std::ratio<1l, ...> > >`
# -- so an anchored `$` pattern over the whole symbol cannot match them. Matched by prefix.
#
# Also from review. The first live run reported
#   GfxFlipThread  100.0%  std::__condvar::wait_until <- std::condition_variable::__wait_until_impl
# as a finding, when it is the wait machinery itself: the tool named the thing it exists to skip.
CXX_PLUMBING_PREFIXES = (
    "std::__condvar::",
    "std::condition_variable::",
    "std::condition_variable_any::",
    "std::mutex::lock",
    "std::mutex::try_lock",
    "std::recursive_mutex::lock",
    "std::timed_mutex::",
    "std::shared_mutex::",
    "std::unique_lock",
    "std::lock_guard",
    "std::this_thread::sleep_for",
    "std::this_thread::sleep_until",
    "std::this_thread::yield",
    "std::thread::join",
    "std::__atomic_futex_unaligned",
    "__gthread_cond_",
    "__gthread_mutex_",
)

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


def is_plumbing(fn: str) -> bool:
    """True for wait machinery, in C or C++ spelling."""
    return bool(PLUMBING.match(fn)) or fn.startswith(CXX_PLUMBING_PREFIXES)


def blocking_site(frames, depth: int):
    """The first frame that is neither plumbing nor unresolved, plus `depth-1` callers.

    Three distinct outcomes, kept distinct because they call for different actions:
      * a named site            -> the answer
      * None                    -> the whole stack is plumbing (a pure library wait)
      * UNRESOLVED_ONLY         -> every frame is `??`; the tool measured nothing here and
                                   the caller must say so rather than print `?? <- ??`

    `??` frames are skipped when they sit ABOVE a named frame, because with
    `auto-solib-add off` they are precisely the library plumbing we mean to skip. They are
    only fatal to the answer when nothing below them resolved.
    """
    for i, fn in enumerate(frames):
        if is_plumbing(fn) or UNRESOLVED.match(fn):
            continue
        # `<-` reads as "called by", so the displayed chain must stay CONTIGUOUS. An earlier
        # version filtered `??` out of the middle, turning ["answer", "??", "caller2"] into
        # "answer <- caller2" -- which asserts an adjacency that does not exist. Truncate at
        # the first unresolved caller instead and mark the cut, so the gap is visible.
        chain, truncated = [], False
        for f in frames[i:]:
            if UNRESOLVED.match(f):
                truncated = True
                break
            chain.append(f)
            if len(chain) == depth:
                break
        site = " <- ".join(chain)
        if truncated and len(chain) < depth:
            site += " <- ..."      # callers below exist but did not resolve
        return site
    # No named site. Three different reasons, and only one of them is a statement about the
    # PROGRAM -- so they must not share a label. An earlier version collapsed all three into
    # "pure library wait", which claims a finding in the two cases that support none.
    if not frames:
        return NO_FRAMES
    if any(UNRESOLVED.match(f) for f in frames):
        return UNRESOLVED_ONLY
    return None            # genuinely all plumbing: a real finding about the program


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
    # UPPER BOUND, not a measurement: this is gdb's whole wall time -- process spawn, attach,
    # symbol work, unwind, detach -- and the target is only actually stopped for part of it.
    # Labelled as a bound because a tool that argues perturbation numbers must be trustworthy
    # cannot then quote one that overstates itself without saying which way it errs. Erring
    # high is the safe direction: it can cause an unnecessary --interval increase, never a
    # falsely-clean run.
    print(f"  perturbation UPPER BOUND: gdb held {stopped_total:.2f}s of {wall:.2f}s wall "
          f"({share:.1f}%) -- includes gdb startup/symbol/detach, so the true stopped time is "
          f"lower. Not a measurement of the program.")
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
