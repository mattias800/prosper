"""gdb-side of `hle_calls` — count every HLE handler entry over a bounded window.

Loaded by `hle_calls.py` via `gdb -batch -x`. Reads its configuration from the
environment so the driver can stay a plain subprocess call:

    HLE_CALLS_SYMS    path to a "<hex-addr> <name>" table (one per line)
    HLE_CALLS_CLOCK   symbol whose entries bound the window (default prosper::k_usleep)
    HLE_CALLS_TICKS   how many clock entries to run for (default 400)

Every handler gets a Python breakpoint whose `stop()` counts the hit and returns
False, so the inferior is never actually stopped for the user. The count is kept
in this script rather than read back from `Breakpoint.hit_count`: relying on
hit_count for a non-stopping breakpoint silently reported zero for every handler
in the first version of this tool, which reads exactly like "the guest called
nothing" and is the trap this file exists to avoid.
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

counts = {}
state = {"ticks": 0}


class _Counter(gdb.Breakpoint):
    def __init__(self, addr, name):
        super().__init__("*%#x" % addr, gdb.BP_BREAKPOINT)
        self.silent = True
        self.hle_name = name
        counts[name] = 0

    def stop(self):
        counts[self.hle_name] += 1
        return False


class _Clock(gdb.Breakpoint):
    def __init__(self, spec):
        super().__init__(spec, gdb.BP_BREAKPOINT)
        self.silent = True

    def stop(self):
        state["ticks"] += 1
        return state["ticks"] >= TICKS


gdb.execute("set pagination off")
gdb.execute("set confirm off")

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

print("hle_calls: armed %d handler breakpoints; window = %d entries of %s"
      % (armed, TICKS, CLOCK))
gdb.execute("continue")

rows = sorted(((c, n) for n, c in counts.items() if c), reverse=True)
print("HLE_CALLS_BEGIN")
print("clock=%s entries=%d armed=%d" % (CLOCK, state["ticks"], armed))
for count, name in rows:
    print("%8d  %s" % (count, name))
print("distinct=%d total=%d" % (len(rows), sum(c for c, _ in rows)))
print("HLE_CALLS_END")

gdb.execute("detach")
gdb.execute("quit")
