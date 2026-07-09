# gdb -p PID -batch -x slot232.py — at the DOLL plateau, catch the RenderThread wait loop
# (eboot+0x221d6c2: mov 0x30(%rbx),%rax ; cmpq $0,(%rax)) and dump the polled slot + context.
# Synchronous continue/stop iterations (a python sleep would block gdb's event loop).
# Wrap the gdb invocation in `timeout` — if the breakpoints never hit, gdb blocks in continue.
import gdb, struct

EBOOT = 0x400000000
inf = gdb.selected_inferior()

def rd64(addr):
    try:
        return struct.unpack("<Q", bytes(inf.read_memory(addr, 8)))[0]
    except gdb.error:
        return None

gdb.execute("set pagination off")
gdb.execute("handle SIGSEGV nostop pass noprint")

# --- the wait loop: dump the polled slot ---
bp1 = gdb.Breakpoint("*0x%x" % (EBOOT + 0x221d6c2))
for i in range(3):
    gdb.execute("continue")
    rbx = int(gdb.parse_and_eval("$rbx"))
    r12 = int(gdb.parse_and_eval("$r12"))
    r13 = int(gdb.parse_and_eval("$r13"))
    slotp = rd64(rbx + 0x30)
    slotv = rd64(slotp) if slotp else None
    print("LOOP hit#%d rbx=0x%x slotp=0x%x slotv=%s timeouts=(0x%x,0x%x) scale=0x%x"
          % (i, rbx, slotp or 0, hex(slotv) if slotv is not None else "?",
             rd64(r12) or 0, rd64(r12 + 8) or 0, rd64(r13) or 0), flush=True)
    for off in range(0, 0x48, 8):
        print("   [rbx+0x%02x] = 0x%x" % (off, rd64(rbx + off) or 0), flush=True)
bp1.delete()

# --- the caller's virtual poll: identify the viewport object ---
bp2 = gdb.Breakpoint("*0x%x" % (EBOOT + 0x3bf485c))
for i in range(2):
    gdb.execute("continue")
    rdi = int(gdb.parse_and_eval("$rdi"))
    r13 = int(gdb.parse_and_eval("$r13"))
    vt = rd64(rdi)
    fn = rd64(vt + 0x338) if vt else None
    print("VIRT hit#%d obj=0x%x vtable=eboot+0x%x fn(+0x338)=eboot+0x%x r13=0x%x"
          % (i, rdi, (vt - EBOOT) if vt else 0, (fn - EBOOT) if fn else 0, r13), flush=True)
    for off in range(0, 0x70, 8):
        print("   [r13+0x%02x] = 0x%x" % (off, rd64(r13 + off) or 0), flush=True)
bp2.delete()
print("=== slot trace done ===", flush=True)
gdb.execute("detach")
