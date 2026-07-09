# gdb boot_trace core -batch -x corescan.py — scan guest .bss/data for the UE4 fatal message
import gdb
gdb.execute("set pagination off")
inf = gdb.selected_inferior()
# guest eboot rw data: vaddr 0x8828000..0x9a1cf30 (+base 0x400000000)
regions = [(0x408828000, 0x409a1cf30 - 0x8828000 + 0x8828000)]
pats = ["Assertion failed", "Fatal error", "LowLevelFatalError", "Ran out", "out of memory",
        "Unable to", "Failed to", "GPU has hung"]
def scan(base, size):
    step = 1 << 20
    off = 0
    while off < size:
        n = min(step, size - off)
        try:
            mem = bytes(inf.read_memory(base + off, n))
        except gdb.error:
            off += n
            continue
        for p in pats:
            for enc in (p.encode(), p.encode("utf-16-le")):
                i = 0
                while True:
                    i = mem.find(enc, i)
                    if i < 0:
                        break
                    ctx = mem[max(0, i - 32): i + 512]
                    try:
                        if enc == p.encode():
                            s = ctx.decode("ascii", "replace")
                        else:
                            s = ctx.decode("utf-16-le", "replace")
                    except Exception:
                        s = repr(ctx[:200])
                    s = "".join(c if 31 < ord(c) < 127 else "." for c in s)
                    print("HIT 0x%x [%s]: %s" % (base + off + i, p, s[:360]))
                    i += len(enc)
        off += n
base = 0x408828000
size = 0x9a1cf30 - 0x8828000
scan(base, size)
print("scan done")
