# gdb -p PID -batch -x disp232.py
# Issue #232: capture the IoDispatcher object from a pool-pop hit (0x23d5a20, rdi=disp+0x8),
# run to the stall, then dump the dispatcher/batch/pool state that gates the final flush.
import gdb, struct, time

gdb.execute("set pagination off")
gdb.execute("handle SIGSEGV SIGBUS SIGILL nostop pass noprint")
gdb.execute("handle SIG34 SIG35 SIG36 SIG37 SIG38 nostop pass noprint")

DISP = [0]

class PopBp(gdb.Breakpoint):
    def stop(self):
        try:
            rdi = int(gdb.parse_and_eval("$rdi"))
            DISP[0] = rdi - 8
        except gdb.error:
            return False
        return True   # stop once

bp = PopBp("*0x4023d5a20")
gdb.execute("continue")
print("[disp232] pool-pop hit: disp=0x%x" % DISP[0])
bp.delete()

inf = gdb.selected_inferior()
def rd(addr, n=8):
    try:
        return int.from_bytes(bytes(inf.read_memory(addr, n)), "little")
    except gdb.error:
        return None

# let it run to the stall (the caller kills us on time budget); sample every 20 s
for round in range(3):
    gdb.execute("continue &")
    time.sleep(20)
    gdb.execute("interrupt")
    time.sleep(1)
    d = DISP[0]
    print("=== sample %d disp=0x%x ===" % (round, d))
    batch = rd(d + 0x20)
    print("  disp+0x20 curBatch = %s" % hex(batch or 0))
    print("  disp+0x30 (flush gate, skip if >1) = %s" % hex(rd(d + 0x30, 4) or 0))
    print("  disp+0x178 directInFlight = %s" % hex(rd(d + 0x178, 4) or 0))
    pool = rd(d + 0x8)
    if pool:
        print("  pool obj = 0x%x count(+0x18)=%s head(+0x20)=%s" % (pool, hex(rd(pool+0x18) or 0), hex(rd(pool+0x20) or 0)))
        h = rd(pool + 0x20); hops = 0
        while h and hops < 64:
            b = rd(h + 8)
            if hops < 6: print("    node 0x%x buf=0x%x" % (h, b or 0))
            h = rd(h); hops += 1
        print("    free nodes: %d%s" % (hops, "+" if hops >= 64 else ""))
    if batch:
        print("  batch+0x08 reqHead = %s" % hex(rd(batch + 0x8) or 0))
        print("  batch+0x10 reqTail = %s" % hex(rd(batch + 0x10) or 0))
        print("  batch+0x20 pendingBytes = %s" % hex(rd(batch + 0x20) or 0))
        print("  batch+0x69 flag = %s  batch+0x68 flag = %s" % (hex(rd(batch+0x69,1) or 0), hex(rd(batch+0x68,1) or 0)))
    # retry queue (dispatcher requeues failed appends via 0x23d6300 into r14 queue obj)
    print("  disp+0x00..0x40:", " ".join(hex(rd(d + o) or 0) for o in range(0, 0x40, 8)))
    print("  disp+0x180 lockish:", hex(rd(d+0x180) or 0), "cnt+0x190:", hex(rd(d+0x190) or 0), "retryHead+0x198:", hex(rd(d+0x198) or 0), "retryTail+0x1a0:", hex(rd(d+0x1a0) or 0))
gdb.execute("detach")
