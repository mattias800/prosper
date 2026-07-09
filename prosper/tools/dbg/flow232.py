# gdb -p PID -batch -x flow232.py — issue #232 session 6: characterize the GameThread
# flow-advance vtable call `call *0x288(%rax)` at eboot+0x5044775.
# rdi = object (this+0x7c0 of the frame/flow fn's this=rsi), rax = vtable.
# For N hits: dump rdi/rsi objects, the +0x288 slot target, and the returned al.
import gdb

EBOOT = 0x400000000
SITE = EBOOT + 0x5044775
RET = SITE + 6  # call *0x288(%rax) = ff 90 88 02 00 00

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

def dump_obj(tag, base, size):
    obj = rd(base, size)
    if not obj:
        print("  %s @0x%x unreadable" % (tag, base))
        return
    for off in range(0, size, 0x20):
        row = obj[off:off+0x20]
        qs = ["%016x" % int.from_bytes(row[k:k+8], "little") for k in range(0, len(row) & ~7, 8)]
        print("  %s+0x%03x: %s" % (tag, off, " ".join(qs)))
    for off in range(0, size & ~7, 8):
        v = int.from_bytes(obj[off:off+8], "little")
        if EBOOT <= v < EBOOT + 0x6700000:
            print("  %s+0x%03x -> %s (text)" % (tag, off, sym(v)))

print("site bytes @ %s: %s" % (sym(SITE), (rd(SITE, 8) or b"").hex()))

bp = gdb.Breakpoint("*0x%x" % SITE)
seen = set()
rets = {}
for i in range(8):
    gdb.execute("continue")
    t = gdb.selected_thread()
    rdi = int(gdb.parse_and_eval("$rdi")) & 0xffffffffffffffff
    rsi = int(gdb.parse_and_eval("$rsi")) & 0xffffffffffffffff
    vt = q(rdi)
    tgt = q(vt + 0x288) if vt else None
    print("HIT %d thread=%s lwp=%s obj(rdi)=0x%x this(rsi)=0x%x vtable=%s target=%s" %
          (i, t.name, t.ptid[1], rdi, rsi, sym(vt), sym(tgt)))
    if rdi not in seen:
        seen.add(rdi)
        dump_obj("obj", rdi, 0x140)
    if i == 0:
        dump_obj("this", rsi, 0x140)
        # also dump around this+0x7c0
        dump_obj("this7c0", rsi + 0x780, 0x80)
    # get the return value at the return address, same thread
    tb = gdb.Breakpoint("*0x%x" % RET, temporary=True)
    tb.thread = t.global_num
    bp.enabled = False
    gdb.execute("continue")
    r2 = int(gdb.parse_and_eval("$rax")) & 0xff
    rets[r2] = rets.get(r2, 0) + 1
    print("  -> returned al=0x%x" % r2)
    bp.enabled = True
print("return-value histogram:", rets)
print("FLOW232-DONE")
