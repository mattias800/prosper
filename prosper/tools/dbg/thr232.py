# gdb -p PID -batch -x thr232.py — list all threads: name, pc, top guest RAs from stack scan
import gdb

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
    if 0x500000000 <= v < 0x600000000: return "prx+0x%x" % (v - 0x500000000)
    return hex(v)

for t in inf.threads():
    try:
        t.switch()
        pc = int(gdb.parse_and_eval("$pc")) & 0xffffffffffffffff
        sp = int(gdb.parse_and_eval("$sp")) & 0xffffffffffffffff
        ras = []
        mem = rd(sp, 0x600)
        if mem:
            for off in range(0, len(mem) - 7, 8):
                v = int.from_bytes(mem[off:off+8], "little")
                if EBOOT <= v < EBOOT + 0x6700000:
                    ras.append("eboot+0x%x" % (v - EBOOT))
                if len(ras) >= 5:
                    break
        print("thr %-18s lwp=%-8d pc=%-22s ras=%s" % (t.name or "?", t.ptid[1], sym(pc), ",".join(ras)))
    except gdb.error as e:
        print("thr ?? err %s" % e)
print("THR232-DONE")
