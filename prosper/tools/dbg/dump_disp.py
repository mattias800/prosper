# gdb -p PID -batch -x dump_disp.py — dump the IoDispatcher state recorded in /root/disp.txt
import gdb
gdb.execute("set pagination off")
d = int(open("/root/disp.txt").read().strip(), 16)
inf = gdb.selected_inferior()
def rd(addr, n=8):
    try:
        return int.from_bytes(bytes(inf.read_memory(addr, n)), "little")
    except gdb.error:
        return None
def hx(v):
    return hex(v) if v is not None else "unreadable"
print("disp=0x%x" % d)
print("  +0x00..0x40:", " ".join(hx(rd(d + o)) for o in range(0, 0x48, 8)))
print("  +0x30 flushGate(u32) =", hx(rd(d + 0x30, 4)))
print("  +0x178 directInFlight(u32) =", hx(rd(d + 0x178, 4)))
batch = rd(d + 0x20)
print("  curBatch(+0x20) =", hx(batch))
if batch:
    print("    batch +0x08 head =", hx(rd(batch + 8)), " +0x10 tail =", hx(rd(batch + 0x10)),
          " +0x18 =", hx(rd(batch + 0x18)), " +0x20 pending =", hx(rd(batch + 0x20)))
    print("    batch +0x68 flag =", hx(rd(batch + 0x68, 1)), " +0x69 flag =", hx(rd(batch + 0x69, 1)))
    h = rd(batch + 8)
    hops = 0
    while h and hops < 8:
        print("      req 0x%x: +0x0=%s +0x8(id32)=%s +0x18(size)=%s +0x28(node)=%s +0x90=%s +0xb1=%s +0xb2=%s" % (
            h, hx(rd(h)), hx(rd(h + 8, 4)), hx(rd(h + 0x18)), hx(rd(h + 0x28)), hx(rd(h + 0x90)),
            hx(rd(h + 0xb1, 1)), hx(rd(h + 0xb2, 1))))
        h = rd(h)
        hops += 1
pool = rd(d + 8)
print("  pool(+0x8) =", hx(pool))
if pool:
    print("    pool +0x18 =", hx(rd(pool + 0x18)), " +0x20 head =", hx(rd(pool + 0x20)))
    h = rd(pool + 0x20)
    hops = 0
    while h and hops < 96:
        if hops < 5:
            print("      node 0x%x next=%s buf=%s" % (h, hx(rd(h)), hx(rd(h + 8))))
        h = rd(h)
        hops += 1
    print("    free nodes walked:", hops)
print("  retry list: +0x190 cnt =", hx(rd(d + 0x190)), " +0x198 head =", hx(rd(d + 0x198)), " +0x1a0 tail =", hx(rd(d + 0x1a0)))
rh = rd(d + 0x198)
hops = 0
while rh and hops < 8:
    print("    retry req 0x%x next=%s +0xc(flags32)=%s +0xb2=%s" % (rh, hx(rd(rh)), hx(rd(rh + 0xc, 4)), hx(rd(rh + 0xb2, 1))))
    rh = rd(rh)
    hops += 1
gdb.execute("detach")
