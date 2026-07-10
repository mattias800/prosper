# gdb -p PID -batch -x flow6.py — DOLL loading-progression: who pumps the per-frame unimplemented
# calls? Break on the host dispatcher prosper_on_unimpl, capture the import index (rdi) + the guest
# RA chain off the stack for ~40 hits, and histogram them. The two dominant chains are the
# per-frame sceNetCtlCheckCallback and sceErrorDialogUpdateStatus pumps — their callers identify
# the engine/game subsystem that is waiting on network-state delivery.
import gdb
from collections import Counter

EBOOT = 0x400000000
gdb.execute("set pagination off")
gdb.execute("set confirm off")
inf = gdb.selected_inferior()

def rd(addr, n):
    try:
        return bytes(inf.read_memory(addr, n))
    except gdb.error:
        return None

def sym(v):
    if EBOOT <= v < EBOOT + 0x6700000: return "eboot+0x%x" % (v - EBOOT)
    if 0x600000000 <= v < 0x700000000: return "stub+0x%x" % (v - 0x600000000)
    return hex(v)

bp = gdb.Breakpoint("prosper_on_unimpl")
chains = Counter()
for i in range(40):
    gdb.execute("continue")
    try:
        idx = int(gdb.parse_and_eval("$rdi")) & 0xffffffff
        sp = int(gdb.parse_and_eval("$sp")) & 0xffffffffffffffff
        mem = rd(sp, 0x1200)
        ras = []
        if mem:
            for off in range(0, len(mem) - 7, 8):
                v = int.from_bytes(mem[off:off+8], "little")
                if EBOOT <= v < EBOOT + 0x6700000:
                    ras.append(sym(v))
                if len(ras) >= 8:
                    break
        chains["idx=%d [%s]" % (idx, " <- ".join(ras))] += 1
    except gdb.error as e:
        print("hit err %s" % e)
print("--- unimpl-call chains over 40 hits ---")
for k, v in chains.most_common(20):
    print("  x%-3d %s" % (v, k))
print("FLOW6-DONE")
