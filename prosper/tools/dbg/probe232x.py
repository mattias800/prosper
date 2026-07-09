# gdb -p PID -batch -x probe232x.py — issue #232: catch ProcessLoadedPackages (0x259f440)
# on the flushing GameThread, capture this, dump loader state + all async packages.
import gdb, struct, time

gdb.execute("set pagination off")
gdb.execute("set confirm off")
inf = gdb.selected_inferior()
EB = 0x400000000

def rd(addr, n=8):
    try:
        return int.from_bytes(bytes(inf.read_memory(addr, n)), "little")
    except gdb.error:
        return None

def hx(v):
    return hex(v) if v is not None else "??"

bp = gdb.Breakpoint("*0x%x" % (EB + 0x259f440))
t0 = time.time()
gdb.execute("continue")
dt = time.time() - t0
thr = gdb.selected_thread()
this = int(gdb.parse_and_eval("$rdi"))
a1 = int(gdb.parse_and_eval("$esi")) & 0xffffffff
a2 = int(gdb.parse_and_eval("$rdx"))
a3 = int(gdb.parse_and_eval("$rcx"))
a4 = int(gdb.parse_and_eval("$r8"))
print("HIT PLP after %.4fs lwp=%d name=%s this=0x%x esi=0x%x rdx=0x%x rcx=0x%x r8=0x%x" %
      (dt, thr.ptid[1], thr.name or "?", this, a1, a2, a3, a4))
bp.delete()

print("== loader `this` raw 0x000..0x240 ==")
for base in range(0, 0x240, 0x20):
    print("  +0x%03x: %s" % (base, " ".join(hx(rd(this + base + i)) for i in range(0, 0x20, 8))))
print("counters: +0x1b0=%s +0x1b4=%s +0x5d0=%s +0x5d8=%s +0x800678=%s" % (
    hx(rd(this + 0x1b0, 4)), hx(rd(this + 0x1b4, 4)), hx(rd(this + 0x5d0, 4)),
    hx(rd(this + 0x5d8)), hx(rd(this + 0x800678, 4))))
print("QueuedPackages num +0x58 =", hx(rd(this + 0x58, 4)))
print("ExternalReadQueue +0x138 =", hx(rd(this + 0x138)), "->", hx(rd(rd(this + 0x138) or 0)))
print("DeferredDeleteQ +0xa8 =", hx(rd(this + 0xa8)), "->", hx(rd(rd(this + 0xa8) or 0)))
# LoadedPackagesToProcess-ish arrays seen in PLP: +0x78/+0x80 and +0x88/+0x90, +0xf0/+0xf8
for off in (0x78, 0x88, 0xf0):
    print("arr @+0x%x: data=%s num=%s" % (off, hx(rd(this + off)), hx(rd(this + off + 8, 4))))

# AsyncPackageLookup at +0xd8 (zen layout). Verify plausibility first.
slots = rd(this + 0xe0, 4) or 0
free = rd(this + 0x10c, 4) or 0
data = rd(this + 0xd8)
print("pkg map: data=%s slots=%d free=%d" % (hx(data), slots, free))
if data and 0 < slots < 50000:
    states = {}
    stuck = []
    for i in range(slots):
        e = data + 24 * i
        pid = rd(e)
        pkg = rd(e + 8)
        if not pkg or pkg < 0x10000000 or pkg > 0x7000000000 or not pid or pid == 0xffffffffffffffff:
            continue
        st = rd(pkg + 0xe0, 1)
        if st is None:
            continue
        states[st] = states.get(st, 0) + 1
        if len(stuck) < 24:
            stuck.append((pid, pkg, st))
    print("state histogram:", {k: v for k, v in sorted(states.items())})
    for pid, pkg, st in stuck:
        print("  pkg=0x%x id=0x%016x st=%d | +0x160(size)=%s +0x16c(prio)=%s +0x140=%s +0x118(extN)=%s" % (
            pkg, pid, st, hx(rd(pkg + 0x160)), hx(rd(pkg + 0x16c, 4)), hx(rd(pkg + 0x140, 4)),
            hx(rd(pkg + 0x118, 4))))
        print("    raw+0x00:", " ".join(hx(rd(pkg + o)) for o in range(0, 0x40, 8)))
        print("    raw+0xc0:", " ".join(hx(rd(pkg + o)) for o in range(0xc0, 0x100, 8)))
gdb.execute("detach")
