# gdb -p PID -batch -x flow5.py — DOLL loading-progression diagnosis (diag/doll-loading-progression):
# the game ticks its per-frame state fn eboot+0x5044740 (`this` + vtable call *0x288) forever on the
# loading screen. Identify the ticked object (RTTI typename), the *0x288 target, diff the object's
# fields across ~300 frames (what changes vs what is parked), and histogram the GameThread's RIP/RA
# between frames to find the poll site.
import gdb, time
from collections import Counter

EBOOT = 0x400000000
ENTRY = EBOOT + 0x5044740

gdb.execute("set pagination off")
gdb.execute("set confirm off")
inf = gdb.selected_inferior()

def rd(addr, n):
    try:
        return bytes(inf.read_memory(addr, n))
    except gdb.error:
        return None

def q(addr):
    b = rd(addr, 8)
    return int.from_bytes(b, "little") if b else None

def sym(v):
    if v is None: return "??"
    if EBOOT <= v < EBOOT + 0x6700000: return "eboot+0x%x" % (v - EBOOT)
    if 0x500000000 <= v < 0x600000000: return "prx+0x%x" % (v - 0x500000000)
    if 0x600000000 <= v < 0x700000000: return "stub+0x%x" % (v - 0x600000000)
    return hex(v)

def typename(vt):
    if not vt: return "?"
    ti = q(vt - 8)
    if not ti: return "?"
    nm = q(ti + 8)
    if not nm: return "?"
    s = rd(nm, 160)
    if not s: return "?"
    return s.split(b"\0")[0].decode("latin1", "replace")

def snap(obj, nwords):
    out = []
    for i in range(nwords):
        out.append(q(obj + i * 8))
    return out

bp = gdb.Breakpoint("*0x%x" % ENTRY)
gdb.execute("continue")
rdi = int(gdb.parse_and_eval("$rdi")) & 0xffffffffffffffff
vt = q(rdi)
print("tick this=0x%x vtable=%s type=%s" % (rdi, sym(vt), typename(vt)))
if vt:
    tgt = q(vt + 0x288)
    print("vtable*0x288 target = %s" % sym(tgt))
snap1 = snap(rdi, 0x30)

# let ~300 frames pass with the bp disabled
bp.enabled = False
gdb.execute("continue &")
time.sleep(5.0)
gdb.execute("interrupt")
time.sleep(0.5)
snap2 = snap(rdi, 0x30)
print("--- object field diff over ~5s (offset: then -> now) ---")
for i in range(0x30):
    a, b = snap1[i], snap2[i]
    tag = "SAME " if a == b else "CHANG"
    print("  +0x%03x %s %-18s -> %-18s %s" % (i * 8, tag,
          hex(a) if a is not None else "??", hex(b) if b is not None else "??",
          ("(" + typename(a) + ")") if (tag == "SAME " and a and EBOOT < a < EBOOT + 0x9000000 and (a % 8 == 0)) else ""))

# GameThread RIP/RA histogram: 80 quick samples
print("--- GameThread sample histogram (80 samples) ---")
hist = Counter()
rahist = Counter()
t1 = None
for t in inf.threads():
    if t.ptid[1] == inf.pid:  # main thread lwp == pid
        t1 = t
        break
if t1 is None:
    t1 = inf.threads()[-1]
for i in range(80):
    gdb.execute("continue &")
    time.sleep(0.04)
    gdb.execute("interrupt")
    time.sleep(0.02)
    try:
        t1.switch()
        pc = int(gdb.parse_and_eval("$pc")) & 0xffffffffffffffff
        sp = int(gdb.parse_and_eval("$sp")) & 0xffffffffffffffff
        hist[sym(pc)] += 1
        mem = rd(sp, 0x300)
        if mem:
            ras = []
            for off in range(0, len(mem) - 7, 8):
                v = int.from_bytes(mem[off:off+8], "little")
                if EBOOT <= v < EBOOT + 0x6700000:
                    ras.append(sym(v))
                if len(ras) >= 3:
                    break
            rahist[",".join(ras)] += 1
    except gdb.error as e:
        print("sample err %s" % e)
for k, v in hist.most_common(12):
    print("  pc %-24s x%d" % (k, v))
print("--- top stack-RA triples ---")
for k, v in rahist.most_common(12):
    print("  ras [%s] x%d" % (k, v))
print("FLOW5-DONE")
