# gdb -p PID -batch -x probe232w.py — issue #232: sample ALL threads' RIPs at the stall;
# print guest pc / guest return addresses to pin the busy-spin loop.
import gdb, struct, time, threading

gdb.execute("set pagination off")
gdb.execute("set confirm off")
inf = gdb.selected_inferior()

def sym(v):
    if v is None:
        return "??"
    if 0x400000000 <= v < 0x406700000:
        return "eboot+0x%x" % (v - 0x400000000)
    if 0x600000000 <= v < 0x700000000:
        return "prx+0x%x" % (v - 0x600000000)
    return None

def sample(rnd):
    for t in inf.threads():
        try:
            t.switch()
            pc = int(gdb.parse_and_eval("$pc")) & 0xffffffffffffffff
            sp = int(gdb.parse_and_eval("$sp")) & 0xffffffffffffffff
        except gdb.error:
            continue
        pcs = sym(pc)
        ras = []
        try:
            mem = bytes(inf.read_memory(sp, 3072))
            for off in range(0, len(mem) - 7, 8):
                v = struct.unpack_from("<Q", mem, off)[0]
                s = sym(v)
                if s:
                    ras.append(s)
                if len(ras) >= 8:
                    break
        except gdb.error:
            pass
        # only print threads with guest presence
        if pcs or ras:
            print("r%d lwp=%d name=%s pc=%s | %s" % (rnd, t.ptid[1], t.name or "?", pcs or hex(pc), " ".join(ras)), flush=True)

for i in range(6):
    sample(i)
    done = threading.Event()
    def stopper():
        time.sleep(0.25)
        gdb.post_event(lambda: gdb.execute("interrupt"))
    threading.Thread(target=stopper, daemon=True).start()
    try:
        gdb.execute("continue")
    except gdb.error as e:
        print("cont err", e)
        break
gdb.execute("detach")
