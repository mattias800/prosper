# gdb -p PID -batch -x state232.py — issue #232 session 5: post-savedata steady-state survey.
# 1) gate bytes (eboot+0x9803460/+0x9803470) + doubles at +0x9603030..48
# 2) per-named-worker RIP + a short guest-RA stack classification
# 3) GameThread (thread 1) RIP samples x20
import gdb, time
from collections import Counter

EBOOT = 0x400000000
gdb.execute("set pagination off")
gdb.execute("set confirm off")
inf = gdb.selected_inferior()

def rd(addr, n=8):
    try:
        return bytes(inf.read_memory(addr, n))
    except gdb.error:
        return None

def sym(v):
    if 0x400000000 <= v < 0x406700000: return "eboot+0x%x" % (v - EBOOT)
    if 0x500000000 <= v < 0x600000000: return "prx+0x%x" % (v - 0x500000000)
    if 0x600000000 <= v < 0x700000000: return "stub+0x%x" % (v - 0x600000000)
    return hex(v)

for a in (EBOOT + 0x9803460, EBOOT + 0x9803470):
    b = rd(a, 16)
    print("gate 0x%x = %s" % (a, b.hex() if b else "??"))
for a in range(EBOOT + 0x9603030, EBOOT + 0x9603050, 8):
    b = rd(a, 8)
    print("dbl  0x%x = %s" % (a, b.hex() if b else "??"))

WANT = ("DollLevelPreloa", "SaveLoadUpdate", "DLCDataUpdate", "ShareUpdate",
        "LevelStreaming", "FAsyncLoading", "RenderThread", "RHIThread", "IoDispatcher", "IoService")
for t in inf.threads():
    nm = t.name or ""
    if not any(nm.startswith(w) for w in WANT):
        continue
    t.switch()
    pc = int(gdb.parse_and_eval("$pc")) & 0xffffffffffffffff
    sp = int(gdb.parse_and_eval("$sp")) & 0xffffffffffffffff
    ras = []
    mem = rd(sp, 0x400)
    if mem:
        for off in range(0, len(mem) - 7, 8):
            v = int.from_bytes(mem[off:off+8], "little")
            if 0x400000000 <= v < 0x406700000:
                ras.append("eboot+0x%x" % (v - EBOOT))
            if len(ras) >= 6:
                break
    print("thr %-16s lwp=%d pc=%s stack-ras=%s" % (nm, t.ptid[1], sym(pc), ras))

# GameThread samples
main = None
for t in inf.threads():
    if t.num == 1:
        main = t
        break
hist = Counter()
if main:
    for i in range(20):
        gdb.execute("continue &", to_string=True)
        time.sleep(0.15)
        gdb.execute("interrupt", to_string=True)
        time.sleep(0.05)
        try:
            main.switch()
            pc = int(gdb.parse_and_eval("$pc")) & 0xffffffffffffffff
            hist[sym(pc)] += 1
        except gdb.error:
            pass
print("GameThread RIP histogram:")
for k, c in hist.most_common(10):
    print("  %3d %s" % (c, k))
