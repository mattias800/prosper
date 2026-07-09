# gdb -p PID -batch -x probe232y.py — issue #232: dump thread1's remaining stack (to the top
# of its mapping) qword by qword; classify guest addresses.
import gdb, struct

gdb.execute("set pagination off")
gdb.execute("set confirm off")
inf = gdb.selected_inferior()

def sym(v):
    if 0x400000000 <= v < 0x406700000:
        return "eboot+0x%x" % (v - 0x400000000)
    if 0x600000000 <= v < 0x700000000:
        return "stub+0x%x" % (v - 0x600000000)
    return None

for t in inf.threads():
    if t.num == 1:
        t.switch()
        break
pc = int(gdb.parse_and_eval("$pc")) & 0xffffffffffffffff
sp = int(gdb.parse_and_eval("$sp")) & 0xffffffffffffffff
top = (sp + 0x1000) & ~0xfff
print("thread1 pc=0x%x sp=0x%x top=0x%x" % (pc, sp, top))
try:
    mem = bytes(inf.read_memory(sp, top - sp))
except gdb.error:
    # fall back: byte at a time pages
    mem = b""
    a = sp
    while a < top:
        try:
            mem += bytes(inf.read_memory(a, 8))
        except gdb.error:
            break
        a += 8
for off in range(0, len(mem) - 7, 8):
    v = struct.unpack_from("<Q", mem, off)[0]
    s = sym(v)
    print("  sp+0x%03x: 0x%016x%s" % (off, v, (" " + s) if s else ""))
gdb.execute("detach")
