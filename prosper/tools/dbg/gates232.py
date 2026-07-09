# gdb -p PID -batch -x gates232.py — issue #232: the eboot+0x5044740 state machine's .data gates.
# 1) read the gate bytes (eboot+0x9803460 / +0x9803470) and the doubles at +0x9603030..48
# 2) arm hardware WRITE watchpoints on both gates for 60 s; report every writer (RIP + thread)
import gdb, time

EBOOT = 0x400000000
G0 = EBOOT + 0x9803460
G1 = EBOOT + 0x9803470

gdb.execute("set pagination off")
gdb.execute("set confirm off")
inf = gdb.selected_inferior()

def rd(addr, n=8):
    try:
        return bytes(inf.read_memory(addr, n))
    except gdb.error:
        return None

for a in (G0, G1):
    b = rd(a, 16)
    print("gate 0x%x = %s" % (a, b.hex() if b else "??"))
for a in range(EBOOT + 0x9603030, EBOOT + 0x9603050, 8):
    b = rd(a, 8)
    print("dbl  0x%x = %s" % (a, b.hex() if b else "??"))

class W(gdb.Breakpoint):
    def __init__(self, addr):
        super().__init__("*(char[16]*)0x%x" % addr, gdb.BP_WATCHPOINT, gdb.WP_WRITE)
        self.addr = addr
    def stop(self):
        pc = int(gdb.parse_and_eval("$pc")) & 0xffffffffffffffff
        t = gdb.selected_thread()
        off = pc - EBOOT
        print("WRITE gate 0x%x from pc=0x%x (eboot+0x%x) thread=%s lwp=%s"
              % (self.addr, pc, off, t.name, t.ptid[1]))
        return False  # keep running

W(G0)
W(G1)
print("watching gates for 60s ...")
gdb.execute("continue &")
time.sleep(60)
gdb.execute("interrupt")
time.sleep(1)
for a in (G0, G1):
    b = rd(a, 16)
    print("gate 0x%x = %s (end)" % (a, b.hex() if b else "??"))
