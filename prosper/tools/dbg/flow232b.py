# gdb -p PID -batch -x flow232b.py — issue #232 session 6: probe fn eboot+0x5044740 entry.
# Reads this (rdi), this+0x7c0 (movie obj), this+0x800 (media obj), then dumps this.
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

def dump_obj(tag, base, size):
    obj = rd(base, size)
    if not obj:
        print("  %s @0x%x unreadable" % (tag, base))
        return
    for off in range(0, size, 0x20):
        row = obj[off:off+0x20]
        qs = ["%016x" % int.from_bytes(row[k:k+8], "little") for k in range(0, len(row) & ~7, 8)]
        print("  %s+0x%03x: %s" % (tag, off, " ".join(qs)))

bp = gdb.Breakpoint("*0x%x" % ENTRY)
for i in range(3):
    gdb.execute("continue")
    t = gdb.selected_thread()
    rdi = int(gdb.parse_and_eval("$rdi")) & 0xffffffffffffffff
    o7c0 = q(rdi + 0x7c0)
    o800 = q(rdi + 0x800)
    print("HIT %d thread=%s lwp=%s this=0x%x  [+0x7c0]=0x%x [+0x800]=0x%x" %
          (i, t.name, t.ptid[1], rdi, o7c0 or 0, o800 or 0))
    if i == 0:
        print("this vtable = %s" % sym(q(rdi)))
        dump_obj("this", rdi, 0x200)
        dump_obj("this+0x780", rdi + 0x780, 0xa0)
        dump_obj("this+0xc00", rdi + 0xc00, 0x60)
        # sub-objects if present
        if o7c0:
            print("movieobj vtable = %s" % sym(q(o7c0)))
            dump_obj("movie", o7c0, 0x80)
        if o800:
            print("mediaobj vtable = %s" % sym(q(o800)))
            dump_obj("media", o800, 0x80)
print("FLOW232B-DONE")
