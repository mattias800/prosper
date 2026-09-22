#!/usr/bin/env python3
"""lock_holder -- who HOLDS the lock everyone is waiting on, and where did they take it?

LINUX ONLY. It reads /proc/<pid>/task/*/syscall for the futex address and uses process_vm_readv
to read the owner word, neither of which exists on Windows or macOS. See
docs/THREAD_WAIT_PROFILING.md section 5c for what a port must preserve.

This is the third question in a stall investigation, and the one that ends it:

    wait_profile   how much is each thread blocked, and on which lock ADDRESS   (cheap, /proc)
    stack_profile  where in the code does the waiter block                      (gdb)
    lock_holder    who OWNS that lock, and what are they doing with it          (this)

"Thread A waits on lock X" is a symptom. "Thread B holds X inside foo() every frame" is a fix.
Without the third answer an investigation stops at a futex address, which names nothing.

HOW THE OWNER IS RECOVERED
--------------------------
glibc passes `&mutex->__data.__lock` as the futex uaddr, and `pthread_mutex_t` on x86-64 begins:

    offset 0   int          __lock       <- the futex word, i.e. the uaddr we observe
    offset 4   unsigned int __count
    offset 8   int          __owner      <- the OWNER'S TID, for error-checking/robust/PI mutexes
    offset 12  unsigned int __nusers

So a single 4-byte read at `uaddr + 8` names the holding thread -- no debugger, no symbols, no
guessing. The read goes through process_vm_readv (/proc/<pid>/mem), which needs the same ptrace
permission as a debugger attach but does NOT stop the target: unlike stack_profile, this tool is
non-invasive and can run at wait_profile's sampling rate.

WHEN __owner IS ZERO, AND WHY THAT IS REPORTED RATHER THAN HIDDEN
-----------------------------------------------------------------
An earlier version of this comment asserted that a PTHREAD_MUTEX_NORMAL does not record its owner.
**That is wrong on current glibc**, measured directly by reading the struct out of a control that
holds one of each kind:

    errck: __lock=2 __count=0 __owner=991900 __nusers=1 __kind=2   (PTHREAD_MUTEX_ERRORCHECK)
    plain: __lock=2 __count=0 __owner=991900 __nusers=1 __kind=0   (PTHREAD_MUTEX_NORMAL)

`__kind=0` and the owner is recorded anyway. So the ordinary `std::mutex` case -- by far the most
common one in this codebase -- IS directly answerable, which is the good news and was nearly
documented away.

The tool still does not ASSUME it, because that measurement covers one glibc on one machine and
this address may not be a mutex at all:

  * a live TID   -> the holder is NAMED, directly and cheaply. Done.
  * 0            -> ownership not recorded by this lock (older glibc, other libc, or not a mutex).
  * anything else-> not a live thread here, so not an owner. Reported as unresolved.

Reporting these differently is the whole point. A tool that silently fell back to a guess would
present it with the same confidence as a read, and the two are not the same evidence.

CONDITION VARIABLES ARE NOT MUTEXES
------------------------------------
For a condvar wait, the uaddr is inside `pthread_cond_t`, whose layout has no `__owner` at +8 --
the value there is part of the internal futex/sequence state. This tool cannot tell the two apart
from the address alone, so a "plausible" TID read from a condvar would be pure noise. Guard: an
owner is only reported when the value is a LIVE TID in this process. A stale or nonsense value is
reported as unresolved rather than as a thread.

A NEGATIVE NEEDS COVERAGE (#3401)
---------------------------------
"No thread was in a futex wait" is only a finding about threads whose syscall state was READ.
The first version turned every OSError into the same None a readable non-futex sample returns,
so a run denied /proc access printed `owner-word read failures: 0` -- true, the reader was never
reached -- beside a confident "this process is not blocked on a lock", and exited 0. Every
thread-sample is now one of: readable futex wait, readable non-futex, vanished mid-sweep (the
thread exited; a race, not a failure), or unreadable (with its errno). Zero readable samples is
UNAVAILABLE and exits 2; a partial run states what it could not see. And even a fully readable
run supports only "none observed in these samples", which is how the negative is worded.
"""

from __future__ import annotations

import argparse
import collections
import ctypes
import ctypes.util
import errno
import os
import struct
import sys
import time

PTHREAD_MUTEX_OWNER_OFFSET = 8      # x86-64 glibc; see module docstring
NR_FUTEX = 202                      # x86-64
# The /proc mount this reads. A constant so the self-test can build a process tree by hand.
PROC_ROOT = "/proc"
# A task file that stops existing between listing the tasks and reading it means the thread
# EXITED -- an ordinary race, and not a gap in coverage the way a permission denial is.
VANISHED = (errno.ENOENT, errno.ESRCH)


def read_checked(path):
    """(text, None) on success, (None, errno-name) on failure -- the failure is kept, not erased."""
    try:
        with open(path) as f:
            return f.read().strip(), None
    except OSError as e:
        return None, (errno.errorcode.get(e.errno, str(e.errno)) if e.errno else type(e).__name__)


def read(path):
    return read_checked(path)[0]


def thread_ids_checked(pid):
    """(tids, None) or ([], errno-name). An empty list with no error is a process with no tasks."""
    try:
        return sorted(int(t) for t in os.listdir(f"{PROC_ROOT}/{pid}/task")), None
    except OSError as e:
        return [], (errno.errorcode.get(e.errno, str(e.errno)) if e.errno else type(e).__name__)


def thread_ids(pid):
    return thread_ids_checked(pid)[0]


def thread_name(pid, tid):
    stat = read(f"{PROC_ROOT}/{pid}/task/{tid}/stat")
    if not stat or "(" not in stat:
        return None
    return stat[stat.index("(") + 1:stat.rindex(")")]


_libc = None


def read_remote(pid, addr, size):
    """process_vm_readv without stopping the target. None on failure (unmapped/permission)."""
    global _libc
    if _libc is None:
        _libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)

    class IOVec(ctypes.Structure):
        _fields_ = [("iov_base", ctypes.c_void_p), ("iov_len", ctypes.c_size_t)]

    buf = (ctypes.c_char * size)()
    local = IOVec(ctypes.cast(buf, ctypes.c_void_p), size)
    remote = IOVec(ctypes.c_void_p(addr), size)
    n = _libc.process_vm_readv(ctypes.c_int(pid), ctypes.byref(local), ctypes.c_ulong(1),
                               ctypes.byref(remote), ctypes.c_ulong(1), ctypes.c_ulong(0))
    return bytes(buf) if n == size else None


def futex_wait_target(pid, tid):
    """(uaddr, "futex") for a thread in a futex wait, else (None, status).

    status is "not-futex" (READ, and not a futex wait), "vanished" (the thread exited between
    enumeration and this read), "unreadable:<ERRNO>", or "malformed" (read, but not parseable).
    Only the first two are observations; the rest are missing coverage and must not be counted
    as a non-futex sample -- which is what made #3401's negative false."""
    text, err = read_checked(f"{PROC_ROOT}/{pid}/task/{tid}/syscall")
    if err is not None:
        if err in (errno.errorcode[e] for e in VANISHED):
            return None, "vanished"
        return None, f"unreadable:{err}"
    sc = text.split()
    if not sc:
        return None, "malformed"
    if sc[0] != str(NR_FUTEX):
        # "running", "-1 <sp> <pc>" (blocked outside a syscall), or another syscall number.
        return None, "not-futex"
    if len(sc) < 2:
        return None, "malformed"
    try:
        return int(sc[1], 16), "futex"
    except ValueError:
        return None, "malformed"


def owner_of(pid, uaddr, live_tids):
    """The TID recorded as owning the mutex whose futex word is at `uaddr`.

    Returns (tid, None) when an owner is recorded and is a live thread,
            (None, reason) otherwise -- never a guess.
    """
    raw = read_remote(pid, uaddr + PTHREAD_MUTEX_OWNER_OFFSET, 4)
    if raw is None:
        return None, "could not read the owner word (unmapped, or no ptrace permission)"
    owner = struct.unpack("<i", raw)[0]
    if owner == 0:
        return None, ("__owner is 0 -- this lock does not track ownership "
                      "(PTHREAD_MUTEX_NORMAL / plain std::mutex), or it is not a mutex at all")
    if owner not in live_tids:
        return None, (f"__owner reads {owner}, which is not a live thread here -- "
                      f"the address is probably not a pthread_mutex_t (a condvar's word at +8 is "
                      f"internal state, not a TID)")
    return owner, None


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Name the thread holding each contended lock, and how long it holds it.")
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--interval", type=float, default=0.05)
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()

    if not os.path.isdir(f"{PROC_ROOT}/{args.pid}"):
        print(f"error: no process {args.pid}", file=sys.stderr)
        return 2

    # uaddr -> Counter of waiter names ; uaddr -> Counter of owner labels
    waiters = collections.defaultdict(collections.Counter)
    owners = collections.defaultdict(collections.Counter)
    unresolved = collections.defaultdict(collections.Counter)
    contended_samples = collections.Counter()
    samples_taken = 0
    read_failures = 0
    # Coverage, kept separately from the owner-word reads it used to be confused with.
    observed = 0                          # thread-samples whose syscall state was READ
    vanished = 0                          # thread exited mid-sweep: a race, not a gap
    unreadable = collections.Counter()    # reason -> thread-samples we could NOT see
    enum_failures = collections.Counter() # reason -> sweeps whose task list could not be read
    stopped = ""

    for _ in range(args.samples):
        tids, enum_err = thread_ids_checked(args.pid)
        if enum_err is not None:
            if enum_err in (errno.errorcode[e] for e in VANISHED):
                stopped = "target exited"
                break
            enum_failures[enum_err] += 1
            time.sleep(args.interval)
            continue
        if not tids:
            stopped = "target has no threads (exited)"
            break
        live = set(tids)
        samples_taken += 1
        blocked_on = {}
        for tid in tids:
            u, status = futex_wait_target(args.pid, tid)
            if status == "futex":
                observed += 1
                blocked_on.setdefault(u, []).append(tid)
            elif status == "not-futex":
                observed += 1
            elif status == "vanished":
                vanished += 1
            else:
                unreadable[status] += 1

        for uaddr, ts in blocked_on.items():
            contended_samples[uaddr] += 1
            for tid in ts:
                waiters[uaddr][f"{thread_name(args.pid, tid) or '?'}/{tid}"] += 1
            owner, why = owner_of(args.pid, uaddr, live)
            if owner is not None:
                owners[uaddr][f"{thread_name(args.pid, owner) or '?'}/{owner}"] += 1
            else:
                unresolved[uaddr][why] += 1
                if "could not read" in why:
                    read_failures += 1
        time.sleep(args.interval)

    print(f"\nlock_holder: pid {args.pid}")
    short = ""
    if samples_taken != args.samples:
        missed_sweeps = sum(enum_failures.values())
        short = f"  ({stopped})" if stopped else (
            f"  ({missed_sweeps} sweep(s) could not read the task list)" if missed_sweeps else "")
    print(f"  samples: {samples_taken} of {args.samples} attempted{short}")
    if enum_failures:
        print("  task-list read failures: " + ", ".join(
            f"{why} x{n}" for why, n in enum_failures.most_common())
            + "  <-- those sweeps saw no thread at all")
    missing = sum(unreadable.values())
    print(f"  syscall coverage: {observed} thread-sample(s) read, {missing} unreadable"
          + (" (" + ", ".join(f"{why} x{n}" for why, n in unreadable.most_common()) + ")"
             if missing else "")
          + f", {vanished} vanished mid-sweep")
    print(f"  owner-word read failures: {read_failures}"
          f"{'  <-- check ptrace permission; every owner below would be missing' if read_failures else ''}")
    if observed == 0:
        # Nothing was SEEN, so nothing can be concluded -- in either direction. Exit 2, the
        # "could not evaluate" status, so a script cannot read this as a clean negative.
        print("\n  UNAVAILABLE: no thread's syscall state could be read in any sample, so this run"
              "\n  says NOTHING about lock waits. Fix access (same user? ptrace_scope? container?)"
              "\n  and re-run; do not record this as 'not blocked on a lock'.\n")
        return 2
    if not contended_samples:
        print(f"\n  No futex wait was observed in {observed} readable thread-sample(s) over"
              f" {samples_taken} sweep(s).\n  That is an observation about THESE samples, not"
              f" proof the process never blocks on a lock;\n  a wait shorter than --interval can"
              f" fall between sweeps.")
        if missing:
            print(f"  PARTIAL: {missing} thread-sample(s) could not be read, and a waiter among"
                  f" them would be invisible here.")
        print()
        return 0
    if missing:
        print(f"\n  PARTIAL: {missing} thread-sample(s) could not be read; a lock whose waiters were"
              f" all among them is missing below.")

    print(f"\n  {'lock (futex uaddr)':<20} {'contended':>9}  holder / waiters")
    print(f"  {'-'*20} {'-'*9}  {'-'*52}")
    for uaddr, n in contended_samples.most_common(args.top):
        pct = n / samples_taken * 100.0
        print(f"  0x{uaddr:<18x} {pct:>8.1f}%")
        for label, c in owners[uaddr].most_common(3):
            print(f"  {'':<20} {'':>9}    HELD BY  {label}  ({c/n*100:.0f}% of its contended samples)")
        for why, c in unresolved[uaddr].most_common(2):
            print(f"  {'':<20} {'':>9}    holder UNRESOLVED ({c}x): {why}")
        for label, c in waiters[uaddr].most_common(4):
            print(f"  {'':<20} {'':>9}    waiting  {label}  ({c}x)")
    print(f"\n  Shares are of the {samples_taken} samples taken. A lock contended in a high share "
          f"of samples\n  with ONE consistent holder is the thing to fix; many holders means the "
          f"lock is hot,\n  not held.\n")
    print("  Next: run stack_profile against the HOLDER's tid to see where it takes the lock.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
