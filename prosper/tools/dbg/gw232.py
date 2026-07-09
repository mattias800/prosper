# gdb -p PID -batch -x gw232.py — dump per-thread host frame + guest return addresses (safe reads).
import gdb, struct

inf = gdb.selected_inferior()

def scan(thr):
    thr.switch()
    try:
        fr = gdb.newest_frame()
        pc = fr.pc()
    except gdb.error:
        return
    # host frames (short)
    names = []
    f = fr
    for _ in range(6):
        if f is None: break
        try:
            n = f.name()
        except gdb.error:
            n = None
        names.append(n or hex(f.pc()))
        try:
            f = f.older()
        except gdb.error:
            break
    print("  host: " + " <- ".join(names))
    # guest RAs on the stack
    try:
        sp = int(gdb.parse_and_eval("$sp"))
    except gdb.error:
        return
    out = []
    step = 4096
    lo = sp - 128
    hi = sp + 8192
    a = lo
    while a < hi:
        n = min(step, hi - a)
        try:
            mem = bytes(inf.read_memory(a, n))
        except gdb.error:
            a += n
            continue
        for off in range(0, len(mem) - 7, 8):
            v = struct.unpack_from("<Q", mem, off)[0]
            if 0x400000000 <= v < 0x406700000:
                out.append("eboot+0x%x" % (v - 0x400000000))
            elif 0x600000000 <= v < 0x610000000:
                out.append("prx+0x%x" % (v - 0x600000000))
        a += n
    print("  guest: " + " ".join(out[:24]))

gdb.execute("set pagination off")
for t in inf.threads():
    print("== thread lwp=%d name=%s ==" % (t.ptid[1], t.name or ""))
    try:
        scan(t)
    except Exception as e:
        print("  err: %s" % e)
