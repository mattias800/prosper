# gdb -p PID -batch -x w1k232.py — break on the w1KFAHVqpaU PLT (0x669a220) and its wrapper
# (0x58df3f0), dump register + stack args, and probe candidate Dcb pointers for a PM4 stream
# (type-3 header top bits 0b11). Runs a bounded number of hits (wrap in `timeout`).
import gdb, struct

EBOOT = 0x400000000
inf = gdb.selected_inferior()

def rd(addr, n=8):
    try:
        return int.from_bytes(bytes(inf.read_memory(addr, n)), "little")
    except gdb.error:
        return None

def looks_pm4(addr):
    v = rd(addr, 4)
    if v is None: return "unmapped"
    return "PM4" if (v & 0xC0000000) == 0xC0000000 else "0x%08x" % v

gdb.execute("set pagination off")
gdb.execute("handle SIGSEGV nostop pass noprint")

bp = gdb.Breakpoint("*0x%x" % (EBOOT + 0x220aace))   # the call site (regs are the caller's)
for i in range(4):
    gdb.execute("continue")
    regs = {r: int(gdb.parse_and_eval("$" + r)) for r in ("rdi","rsi","rdx","rcx","r8","r9","rsp","r12","rax")}
    print("CALLSITE hit#%d rdi=0x%x rsi=0x%x rdx=0x%x rcx=0x%x r8=0x%x r9=0x%x r12=0x%x rax=0x%x"
          % (i, regs["rdi"],regs["rsi"],regs["rdx"],regs["rcx"],regs["r8"],regs["r9"],regs["r12"],regs["rax"]),
          flush=True)
    sp = regs["rsp"]
    for k in range(6):
        a = rd(sp + k*8)
        print("   [rsp+0x%02x] = 0x%x  -> pm4? %s" % (k*8, a or 0, looks_pm4(a) if a else "n/a"), flush=True)
    # the caller built (rax,r12): buffer entry base = rax + r12; fields +0 base, +0x10 dwords
    base = rd(regs["rax"] + regs["r12"])
    dw = rd(regs["rax"] + regs["r12"] + 0x10, 4)
    print("   buffer_entry: base=0x%x dwords=0x%x  base->pm4? %s"
          % (base or 0, dw or 0, looks_pm4(base) if base else "n/a"), flush=True)
    if base:
        # dump the first few dwords
        for k in range(4):
            print("     [base+0x%02x]=0x%08x" % (k*4, rd(base + k*4, 4) or 0), flush=True)
print("=== w1k probe done ===", flush=True)
gdb.execute("detach")
