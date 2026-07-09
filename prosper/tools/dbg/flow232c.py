# gdb -p PID -batch -x flow232c.py — issue #232 session 6:
# 1) at eboot+0x5044740 entry: rbp-chain backtrace (guest frames) + raw stack RA scan
# 2) RTTI typeinfo names for this (vtable 0x408a46708) and media obj (vtable 0x4090757c0)
import gdb

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
    return hex(v)

def typename(vt):
    # Itanium ABI: vtable[-1] = typeinfo ptr; typeinfo+8 = name ptr
    ti = q(vt - 8)
    if not ti: return "?"
    nm = q(ti + 8)
    if not nm: return "?"
    s = rd(nm, 120)
    if not s: return "?"
    return s.split(b"\0")[0].decode("latin1", "replace")

bp = gdb.Breakpoint("*0x%x" % ENTRY)
gdb.execute("continue")
rdi = int(gdb.parse_and_eval("$rdi")) & 0xffffffffffffffff
rsp = int(gdb.parse_and_eval("$sp")) & 0xffffffffffffffff
rbp = int(gdb.parse_and_eval("$rbp")) & 0xffffffffffffffff
print("this=0x%x vtable=%s type=%s" % (rdi, sym(q(rdi)), typename(q(rdi))))
o800 = q(rdi + 0x800)
if o800:
    print("media obj type=%s" % typename(q(o800)))
# return address = *rsp at entry
print("caller RA = %s" % sym(q(rsp)))
# rbp chain
fp = rbp
for d in range(16):
    ra = q(fp + 8)
    nfp = q(fp)
    if ra is None or nfp is None: break
    print("frame %2d fp=0x%x ra=%s" % (d, fp, sym(ra)))
    if not (EBOOT <= ra < EBOOT + 0x6700000):
        # keep walking anyway a couple frames
        pass
    if nfp <= fp or nfp - fp > 0x100000: break
    fp = nfp
# raw scan of return-address-looking values on the stack
mem = rd(rsp, 0x800)
if mem:
    out = []
    for off in range(0, len(mem) - 7, 8):
        v = int.from_bytes(mem[off:off+8], "little")
        if EBOOT <= v < EBOOT + 0x6700000:
            out.append("sp+0x%x:%s" % (off, sym(v)))
    print("stack text-ptrs:", " ".join(out[:40]))
print("FLOW232C-DONE")
