# gdb -p PID -batch -x probe232z.py — issue #232:
# 1) read the parked FAsyncLoadingThread2 (ALT) thread's regs/stack to recover `this`,
#    then dump async-loading counters + package-state histogram.
# 2) sample the GameThread (thread 1) 40x for a guest-RA histogram of its tick phase.
import gdb, struct, time, threading
from collections import Counter

gdb.execute("set pagination off")
gdb.execute("set confirm off")
inf = gdb.selected_inferior()

def rd(addr, n=8):
    try:
        return int.from_bytes(bytes(inf.read_memory(addr, n)), "little")
    except gdb.error:
        return None

def hx(v):
    return hex(v) if v is not None else "??"

def sym(v):
    if v is None:
        return "??"
    if 0x400000000 <= v < 0x406700000:
        return "eboot+0x%x" % (v - 0x400000000)
    if 0x600000000 <= v < 0x700000000:
        return "stub+0x%x" % (v - 0x600000000)
    return hex(v)

# --- part 1: ALT thread ---
alt = None
for t in inf.threads():
    if t.name and t.name.startswith("FAsyncLoading"):
        alt = t
        break
cands = []
if alt:
    alt.switch()
    print("ALT lwp=%d" % alt.ptid[1])
    for r in ("rbx", "r12", "r13", "r14", "r15", "rbp", "rdi", "rsi"):
        v = int(gdb.parse_and_eval("$" + r)) & 0xffffffffffffffff
        print("  %s=0x%x" % (r, v))
        if 0x10000000 < v < 0x7000000000:
            cands.append(v)
    sp = int(gdb.parse_and_eval("$sp")) & 0xffffffffffffffff
    try:
        mem = bytes(inf.read_memory(sp, 2048))
        ras = []
        for off in range(0, len(mem) - 7, 8):
            v = struct.unpack_from("<Q", mem, off)[0]
            if 0x400000000 <= v < 0x406700000:
                ras.append("sp+0x%x:%s" % (off, sym(v)))
            elif 0x10000000 < v < 0x7000000000:
                cands.append(v)
        print("  ALT guest RAs:", " ".join(ras[:14]))
    except gdb.error:
        pass

def plausible_loader(v):
    # zen loader `this`: +0x1b0/+0x1b4 small counters, +0x58 small, +0xe0 slots>=0 small,
    # +0x10c <= +0xe0, +0xd8 pointer-ish or 0, +0x800660 small (evq count), +0x5d0 small
    c1 = rd(v + 0x1b0, 4); c2 = rd(v + 0x1b4, 4)
    if c1 is None or c2 is None or c1 > 100000 or c2 > 100000:
        return False
    n = rd(v + 0x58, 4); slots = rd(v + 0xe0, 4); fr = rd(v + 0x10c, 4)
    if n is None or slots is None or fr is None or n > 1000000 or slots > 1000000 or fr > slots:
        return False
    qn = rd(v + 0x800660, 4)
    if qn is None or qn > 64:
        return False
    return True

seen = set()
loader = None
for v in cands:
    for base in (v, v - 0x10, v - 0x20):
        if base in seen or base < 0:
            continue
        seen.add(base)
        if plausible_loader(base):
            loader = base
            print("LOADER CANDIDATE this=0x%x (from 0x%x)" % (base, v))
            break
    if loader:
        break

if loader:
    print("QueuedPackagesCounter +0x1b0 =", hx(rd(loader + 0x1b0, 4)))
    print("ExistingAsyncPkgCtr   +0x1b4 =", hx(rd(loader + 0x1b4, 4)))
    print("QueuedPackages.Num    +0x58  =", hx(rd(loader + 0x58, 4)))
    print("pendingIoBytes        +0x5d8 =", hx(rd(loader + 0x5d8)))
    print("heapNum               +0x5d0 =", hx(rd(loader + 0x5d0, 4)))
    print("ExternalReadQueue     +0x138 =", hx(rd(loader + 0x138)), "->", hx(rd(rd(loader + 0x138) or 0)))
    slots = rd(loader + 0xe0, 4) or 0
    free = rd(loader + 0x10c, 4) or 0
    data = rd(loader + 0xd8)
    print("pkg map: data=%s slots=%d free=%d" % (hx(data), slots, free))
    if data and 0 < slots < 50000:
        states = Counter()
        sample_pkgs = {}
        for i in range(slots):
            e = data + 24 * i
            pid = rd(e)
            pkg = rd(e + 8)
            if not pkg or pkg < 0x10000000 or pkg > 0x7000000000 or not pid:
                continue
            st = rd(pkg + 0xe0, 1)
            if st is None:
                continue
            states[st] += 1
            if st not in sample_pkgs:
                sample_pkgs[st] = (pid, pkg)
        print("pkg state histogram:", dict(states))
        for st, (pid, pkg) in sorted(sample_pkgs.items()):
            print("  st=%d sample pkg=0x%x id=0x%016x" % (st, pkg, pid))
else:
    print("no loader candidate found")

# --- part 2: GameThread RA histogram ---
hist = Counter()
for i in range(40):
    for t in inf.threads():
        if t.num == 1:
            t.switch()
            break
    try:
        pc = int(gdb.parse_and_eval("$pc")) & 0xffffffffffffffff
        sp = int(gdb.parse_and_eval("$sp")) & 0xffffffffffffffff
        top = (sp + 0x1000) & ~0xfff
        first = None
        try:
            mem = bytes(inf.read_memory(sp, min(top - sp, 1024)))
            for off in range(0, len(mem) - 7, 8):
                v = struct.unpack_from("<Q", mem, off)[0]
                if 0x400000000 <= v < 0x406700000:
                    first = v
                    break
        except gdb.error:
            pass
        if 0x400000000 <= pc < 0x406700000:
            hist[("pc", pc & ~0x3f)] += 1
        elif first:
            hist[("ra", first)] += 1
        else:
            hist[("host", 0)] += 1
    except gdb.error:
        pass
    def stopper():
        time.sleep(0.05)
        gdb.post_event(lambda: gdb.execute("interrupt"))
    threading.Thread(target=stopper, daemon=True).start()
    try:
        gdb.execute("continue")
    except gdb.error:
        break
print("GT sample histogram:")
for (k, v), c in hist.most_common(12):
    print("  %2d x %s %s" % (c, k, sym(v) if v else "-"))
gdb.execute("detach")
