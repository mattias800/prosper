#!/usr/bin/env python3
# waitgraph.py — build a wait-for graph for a TOTALLY deadlocked guest.
#
# When every guest thread is parked, `PROSPER_APP_GUEST_DUMP_PATH` tells you WHERE each thread
# stopped and `PROSPER_SYNC_RING` tells you what happened before they stopped. Neither, alone, tells
# you WHO everyone is waiting for — and with ~76 threads that is not a question you want to answer by
# reading a 70-line snapshot next to a 200,000-line event log by hand.
#
# This joins the two. For each parked thread T waiting on object O it draws an edge to the thread
# that most recently woke O (signal / broadcast / interrupt). Then:
#
#   * a ROOT is a parked thread whose object NOTHING has ever woken. Nothing in the process has ever
#     been the producer for it, so no amount of waiting will help. This is usually the real defect.
#   * a CYCLE is a classic lock-ordering deadlock: A waits for B waits for A.
#
# A forest with roots and no cycle means the failure is starvation / a lost or never-sent wakeup,
# NOT a lock-ordering bug — which is worth knowing before hunting for one.
#
# Both inputs may be the same file: `prosper-app`'s timed dump appends the thread snapshot and the
# sync ring together, which is the intended way to use this.
#
# Usage:
#     tools/re/waitgraph.py <dump-file> [<dump-file> ...]
#
# Produce the input with:
#     PROSPER_SYNC_RING=4000000 PROSPER_APP_GUEST_DUMP_MS=110000 \
#     PROSPER_APP_GUEST_DUMP_PATH=/tmp/dump.txt  prosper-app <game>
#
# Caveat worth reading before drawing conclusions: only the CONDITION-slot paths record wake events.
# A thread parked in an `address` (raw futex) wait therefore always looks like a root, because no
# producer was ever recorded for its object — not because none exists. Those are reported separately
# for exactly that reason; do not read them as findings.

import collections
import re
import sys

THREAD_RE = re.compile(
    r"^\[thread-trace\] tid=(\d+) pthread=(0x[0-9a-f]+).*?"
    r"waits=(\w[\w-]*)@(0x[0-9a-f]+)\(source=(0x[0-9a-f]+)\)")
EVENT_RE = re.compile(
    r"^\[sync-trace\] seq=(\d+) kind=(\S+)\s+tid=(\d+) pthread=(0x[0-9a-f]+) "
    r"object=(0x[0-9a-f]+) source=(0x[0-9a-f]+) value=(\d+)")

WAKE_KINDS = ("signal", "broadcast", "interrupt", "futex-wake", "guest-wake")


def main(paths):
    threads = {}                        # pthread -> (kind, object, source); last snapshot wins
    last_wake = {}                      # object -> (seq, kind, pthread)
    events_on = collections.Counter()
    sources_on = collections.defaultdict(set)   # object -> guest condvars seen behind that slot

    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                match = THREAD_RE.match(line)
                if match:
                    threads[match.group(2)] = (match.group(3), match.group(4), match.group(5))
                    continue
                match = EVENT_RE.match(line)
                if match:
                    obj = match.group(5)
                    events_on[obj] += 1
                    sources_on[obj].add(match.group(6))
                    if match.group(2) in WAKE_KINDS:
                        last_wake[obj] = (int(match.group(1)), match.group(2), match.group(4))

    if not threads:
        print("no [thread-trace] lines found — is PROSPER_APP_GUEST_DUMP_PATH set?")
        return 1
    print("parked threads: %d   objects with recorded events: %d" % (len(threads), len(events_on)))

    roots, unrecorded, edges = [], [], {}
    for pthread, (kind, obj, source) in sorted(threads.items(), key=lambda kv: int(kv[0], 16)):
        if obj in last_wake:
            edges[pthread] = last_wake[obj][2]
        elif kind == "address" and events_on[obj] == 0:
            unrecorded.append((pthread, kind, obj, source))   # no events at all; not a finding
        else:
            roots.append((pthread, kind, obj, source, events_on[obj]))

    print("\n=== ROOTS: parked on an object nothing has ever woken ===")
    if not roots:
        print("  (none)")
    for pthread, kind, obj, source, count in roots:
        print("  pthread=%-6s %-10s object=%s guest-source=%s events-ever=%d"
              % (pthread, kind, obj, source, count))

    if unrecorded:
        print("\n=== not evidence: raw `address` waits, whose wakes this ring does not record ===")
        for pthread, kind, obj, source in unrecorded:
            print("  pthread=%-6s %-10s object=%s" % (pthread, kind, obj))

    # A condition slot is a recycled entry in a fixed 4096-slot array (sync_futex.cpp: a retired
    # slot returns to 0 and a DIFFERENT guest condvar may then claim it). So one object address can
    # cover two unrelated condvars, and an edge drawn across that boundary joins a wake on one to a
    # wait on the other. The event record carries the guest source, so the run can say whether it
    # actually happened here instead of leaving the reader to assume it did not.
    reused = sorted(obj for obj in sources_on if len(sources_on[obj]) > 1)
    tainted = sorted(p for p in edges if threads[p][1] in reused)
    if reused:
        print("\n=== WARNING: %d slot(s) held more than one guest condvar in this run ===" % len(reused))
        for obj in reused:
            print("  object=%s  sources=%s" % (obj, " ".join(sorted(sources_on[obj]))))
        print("  Edges through these objects may pair a wake on one condvar with a wait on another.")
        print("  %s" % ("%d edge(s) below rest on a reused slot: %s" % (len(tainted), " ".join(tainted))
                        if tainted else "No edge below rests on one, so the graph is unaffected."))

    print("\n=== wait-for edges (thread -> thread that last woke its object) ===")
    fanin = collections.Counter()
    self_served = 0
    for pthread in sorted(edges, key=lambda p: int(p, 16)):
        target = edges[pthread]
        if target == pthread:
            self_served += 1
            continue
        fanin[target] += 1
        note = "" if target in threads else "   <-- NOT in the live thread set (exited?)"
        print("  %-6s -> %-6s%s" % (pthread, target, note))
    if self_served:
        print("  (%d threads whose last waker was themselves — omitted)" % self_served)

    if fanin:
        print("\n=== most-awaited threads ===")
        for target, count in fanin.most_common(5):
            live = "parked" if target in threads else "NOT in the live thread set"
            print("  %-6s awaited by %d thread(s)  [%s]" % (target, count, live))

    print("\n=== cycles ===")
    seen, found = set(), []
    for start in edges:
        order, cursor = {}, start
        chain = []
        while cursor in edges and cursor not in order:
            order[cursor] = len(chain)
            chain.append(cursor)
            cursor = edges[cursor]
        if cursor in order:
            cycle = tuple(chain[order[cursor]:])
            key = tuple(sorted(cycle))
            if key not in seen:
                seen.add(key)
                found.append(cycle)
    for cycle in found:
        print("  " + " -> ".join(cycle) + " -> " + cycle[0])
    if not found:
        print("  (none — the graph terminates at the roots above, so this is starvation or a")
        print("   never-sent wakeup rather than a lock-ordering deadlock)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
