#!/usr/bin/env python3
"""guest_stacks — where is every PARKED guest fiber, in a live prosper process?

A fiber-based title (Naughty Dog's NdJob, for one) does not keep its interesting call stacks on
host threads. When a job waits, its fiber is parked: the worker thread switches to another fiber
and goes back to scanning for work, so `gdb`'s `thread apply all bt` shows every worker in the same
idle loop and nothing else. The stack that says what the title is actually waiting for is sitting in
a guest memory region that no thread currently points at, and gdb cannot unwind it anyway (no
symbols, no CFI, and the guest's own frame layout).

This walks the guest stack regions of a running prosper directly through `/proc/<pid>/mem` and
reports every value in them that looks like a pointer into the guest module, as `eboot+0x…`. That is
deliberately cruder than an unwinder and much more useful here: a return address, a spilled function
pointer and a stale frame all show up, and the ORDER (descending address = outermost first) is
enough to read the call chain by eye against `tools/re/edis.py`.

    guest_stacks.py <pid> [--match 0x5a9ce0[,0x5ab700-0x5ab800]] [--base 0x410000000]

    --match   print only regions containing at least one of these module offsets (a bare offset, or
              a `lo-hi` range). Without it every region with guest pointers is printed, which on a
              real title is tens of thousands of lines.
    --base    guest module base (default 0x410000000 -- check a known address first, e.g. take one
              `[hwbp] rip=eboot+0x…` line and the raw RIP beside it).

WHAT THIS DOES NOT DO, stated because the obvious reading is the wrong one: it does not know where a
fiber's stack pointer is, so it cannot tell a LIVE frame from stale bytes left below it by an older,
deeper call. A long run of guest pointers with a multi-KiB gap under it is usually the live frame
set, but that is a heuristic. Confirm a frame you intend to rely on with a hardware breakpoint on
the instruction after the call site (`PROSPER_HWBP`) -- a count that stops one short of the others
is the evidence; this tool is how you find the address worth arming.
"""
import argparse
import re
import struct
import sys

GUEST_STACK_LO = 0x1800000000
GUEST_STACK_HI = 0x1810000000


def parse_match(spec):
    out = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.append((int(lo, 0), int(hi, 0)))
        else:
            v = int(part, 0)
            out.append((v, v + 1))
    return out


def regions(pid, lo, hi):
    found = []
    try:
        fh = open("/proc/%d/maps" % pid)
    except OSError as exc:
        # A bad pid and a permission problem are different answers, and a traceback is neither.
        print("cannot read /proc/%d/maps: %s" % (pid, exc), file=sys.stderr)
        return None
    with fh:
        for line in fh:
            m = re.match(r"([0-9a-f]+)-([0-9a-f]+) (\S+)", line)
            if not m:
                continue
            a, b, perm = int(m.group(1), 16), int(m.group(2), 16), m.group(3)
            if lo <= a < hi and "r" in perm:
                found.append((a, b))
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pid", type=int)
    ap.add_argument("--match", default="")
    ap.add_argument("--base", default="0x410000000")
    ap.add_argument("--span", default="0x3700000",
                    help="module size; a value outside [base, base+span) is not reported")
    ap.add_argument("--lo", default=hex(GUEST_STACK_LO))
    ap.add_argument("--hi", default=hex(GUEST_STACK_HI))
    args = ap.parse_args()

    base, span = int(args.base, 0), int(args.span, 0)
    want = parse_match(args.match)
    regs = regions(args.pid, int(args.lo, 0), int(args.hi, 0))
    if regs is None:
        return 2
    if not regs:
        print("no readable guest stack regions in [%s, %s) -- wrong pid, or the guest has not "
              "allocated its fiber stacks yet" % (args.lo, args.hi), file=sys.stderr)
        return 2

    printed = 0
    with open("/proc/%d/mem" % args.pid, "rb", 0) as mem:
        for a, b in regs:
            try:
                mem.seek(a)
                data = mem.read(b - a)
            except OSError:
                continue                      # a region can go away under us; that is not an error
            n = len(data) // 8
            vals = struct.unpack("<%dQ" % n, data[:n * 8])
            ptrs = [(a + i * 8, v - base) for i, v in enumerate(vals) if base <= v < base + span]
            if not ptrs:
                continue
            if want and not any(lo <= o < hi for _, o in ptrs for lo, hi in want):
                continue
            print("=== REGION 0x%x-0x%x  %d guest pointer(s) ===" % (a, b, len(ptrs)))
            for addr, off in sorted(ptrs, reverse=True):
                print("   0x%x : eboot+0x%x" % (addr, off))
            printed += 1
    if not printed:
        print("no region matched", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
