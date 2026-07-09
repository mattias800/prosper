# gdb -p PID -batch -x scan_disp.py
# Locate the IoDispatcher backend by scanning guest memory for pointers to the stalled batch
# (batch = last H896 tag, written to /root/batch.txt by the runner), then dump the flush gate.
import gdb
gdb.execute("set pagination off")
batch = int(open("/root/batch.txt").read().strip(), 16)
inf = gdb.selected_inferior()
pid = gdb.selected_inferior().pid
def rd(addr, n=8):
    try:
        return int.from_bytes(bytes(inf.read_memory(addr, n)), "little")
    except gdb.error:
        return None
def hx(v):
    return hex(v) if v is not None else "??"

# collect writable mappings in the guest heap ranges
regions = []
for line in open("/proc/%d/maps" % pid):
    parts = line.split()
    if "rw" not in parts[1]:
        continue
    lo, hi = (int(x, 16) for x in parts[0].split("-"))
    if hi - lo > (1 << 32):   # skip giant reservations
        continue
    # guest arenas: 0x10xxxxxxxx..0x14xxxxxxxx and 0x2xxxxxxxxx
    if 0x1000000000 <= lo < 0x4000000000 or 0x400000000 <= lo < 0x500000000:
        regions.append((lo, hi))
print("scanning %d regions for 0x%x" % (len(regions), batch))
hits = []
needle = batch.to_bytes(8, "little")
for lo, hi in regions:
    off = lo
    while off < hi:
        n = min(1 << 22, hi - off)
        try:
            mem = bytes(inf.read_memory(off, n))
        except gdb.error:
            off += n
            continue
        i = 0
        while True:
            i = mem.find(needle, i)
            if i < 0:
                break
            if (off + i) % 8 == 0:
                hits.append(off + i)
            i += 1
        off += n
print("hits:", [hex(h) for h in hits[:40]])
# For each hit, try interpreting it as disp+0x20 or disp+0x28 and sanity-check the neighborhood.
for h in hits[:40]:
    for base_off in (0x20, 0x28):
        d = h - base_off
        gate = rd(d + 0x30, 4)
        inflight_head = rd(d + 0x28)
        cur = rd(d + 0x20)
        direct = rd(d + 0x178, 4)
        if gate is None or cur is None:
            continue
        # plausible: gate small-ish, cur/head look like heap ptrs
        if gate < 0x100000 and (cur is None or cur < 0x8000000000):
            print("cand disp=0x%x (hit as +0x%x): +0x20 cur=%s +0x28 head=%s +0x30 gate=%s +0x178 direct=%s" % (
                d, base_off, hx(cur), hx(inflight_head), hx(gate), hx(direct)))
            if inflight_head and inflight_head < 0x8000000000:
                nodep = inflight_head
                for k in range(4):
                    if not nodep or nodep > 0x8000000000:
                        break
                    print("    inflight[%d]=0x%x next=%s free68=%s reqHead+8=%s" % (
                        k, nodep, hx(rd(nodep)), hx(rd(nodep + 0x68, 2)), hx(rd(nodep + 8))))
                    nodep = rd(nodep)
# also dump the batch object itself
print("batch 0x%x:" % batch)
for o in range(0, 0x78, 8):
    print("  +0x%02x = %s" % (o, hx(rd(batch + o))))
gdb.execute("detach")
