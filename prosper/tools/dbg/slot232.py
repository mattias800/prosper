# gdb -p PID -batch -x slot232.py — at the DOLL plateau, catch the RenderThread wait loop
# (eboot+0x221d6c2: mov 0x30(%rbx),%rax ; cmpq $0,(%rax)) and dump the polled slot + context.
# Also catch the caller's virtual poll at eboot+0x3bf485c to identify the viewport object.
import gdb, time, struct

EBOOT = 0x400000000
inf = gdb.selected_inferior()

def rd64(addr):
    try:
        return struct.unpack("<Q", bytes(inf.read_memory(addr, 8)))[0]
    except gdb.error:
        return None

class LoopBP(gdb.Breakpoint):
    def __init__(self):
        super().__init__("*0x%x" % (EBOOT + 0x221d6c2))
        self.n = 0
    def stop(self):
        try:
            self.n += 1
            if self.n > 3: return False
            rbx = int(gdb.parse_and_eval("$rbx"))
            r12 = int(gdb.parse_and_eval("$r12"))
            r13 = int(gdb.parse_and_eval("$r13"))
            slotp = rd64(rbx + 0x30)
            slotv = rd64(slotp) if slotp else None
            ctx28 = rd64(rbx + 0x28)
            t0 = rd64(r12)      # timeout[0]
            t1 = rd64(r12 + 8)  # timeout[1]
            scale = rd64(r13)
            print("LOOP hit#%d rbx=0x%x slotp=0x%x slotv=%s ctx28=0x%x timeouts=(0x%x,0x%x) scale=0x%x"
                  % (self.n, rbx, slotp or 0, hex(slotv) if slotv is not None else "?", ctx28 or 0,
                     t0 or 0, t1 or 0, scale or 0), flush=True)
            # dump the wait object
            for off in range(0, 0x40, 8):
                print("   [rbx+0x%02x] = 0x%x" % (off, rd64(rbx + off) or 0), flush=True)
        except Exception as e:
            print("looperr %s" % e, flush=True)
        return False

class VirtBP(gdb.Breakpoint):
    def __init__(self):
        super().__init__("*0x%x" % (EBOOT + 0x3bf485c))
        self.n = 0
    def stop(self):
        try:
            self.n += 1
            if self.n > 3: return False
            rdi = int(gdb.parse_and_eval("$rdi"))
            r13 = int(gdb.parse_and_eval("$r13"))
            vt = rd64(rdi)
            fn = rd64(vt + 0x338) if vt else None
            print("VIRT hit#%d obj=0x%x vtable=0x%x(eboot+0x%x) fn+0x338=0x%x(eboot+0x%x) r13=0x%x"
                  % (self.n, rdi, vt or 0, (vt - EBOOT) if vt else 0,
                     fn or 0, (fn - EBOOT) if fn else 0, r13), flush=True)
            for off in range(0, 0x60, 8):
                print("   [r13+0x%02x] = 0x%x" % (off, rd64(r13 + off) or 0), flush=True)
        except Exception as e:
            print("virterr %s" % e, flush=True)
        return False

gdb.execute("set pagination off")
gdb.execute("handle SIGSEGV nostop pass noprint")
LoopBP()
VirtBP()
gdb.execute("continue &")
time.sleep(25)
gdb.execute("interrupt")
time.sleep(2)
print("=== slot trace done ===", flush=True)
gdb.execute("detach")
