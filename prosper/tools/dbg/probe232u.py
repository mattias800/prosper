# gdb -p PID -batch -x probe232u.py — issue #232: at the FlushAsyncLoading stall, capture
# FAsyncLoadingThread2 state (UE 4.27 zenloader). ProcessAsyncLoading=eboot+0x25b2980 spins
# INSIDE its do-while, so break on in-loop addresses:
#   0x25b2ac4 loop-condition head (rbx=this, r15=ThreadState)
#   0x25b29e0 CreateAsyncPackagesFromQueue callsite (rbx=this)
#   0x25acf40 CreateAsyncPackagesFromQueue entry (rdi=this)
# Layout: QueuedPackagesCounter=this+0x1b0, ExistingAsyncPackagesCounter=this+0x1b4,
# QueuedPackages TArray={data +0x50,num +0x58}, AsyncPackageLookup TMap elements=+0xd8
# (24-byte {u64 PackageId, FAsyncPackage2*, i32 hashnext}), slots num=+0xe0, free=+0x10c,
# ExternalReadQueue head=+0x138, DeferredDeleteQueue=+0xa8, pkg state byte=pkg+0xe0,
# pkg ExternalReadDependencies={data pkg+0x110, num pkg+0x118}.
import gdb, struct, time

gdb.execute("set pagination off")
gdb.execute("set confirm off")
inf = gdb.selected_inferior()
EB = 0x400000000

def rd(addr, n=8):
    try:
        return int.from_bytes(bytes(inf.read_memory(addr, n)), "little")
    except gdb.error:
        return None

def hx(v):
    return hex(v) if v is not None else "??"

hit = {"this": 0, "ts": 0, "where": ""}

class B1(gdb.Breakpoint):  # loop head: rbx=this r15=ThreadState
    def stop(self):
        hit["this"] = int(gdb.parse_and_eval("$rbx"))
        hit["ts"] = int(gdb.parse_and_eval("$r15"))
        hit["where"] = "loophead"
        return True

class B2(gdb.Breakpoint):  # capq entry: rdi=this rsi=ThreadState
    def stop(self):
        hit["this"] = int(gdb.parse_and_eval("$rdi"))
        hit["ts"] = int(gdb.parse_and_eval("$rsi"))
        hit["where"] = "capq"
        return True

class B3(gdb.Breakpoint):  # PAL entry: rdi=this rsi=ThreadState
    def stop(self):
        hit["this"] = int(gdb.parse_and_eval("$rdi"))
        hit["ts"] = int(gdb.parse_and_eval("$rsi"))
        hit["where"] = "entry"
        return True

b1 = B1("*0x%x" % (EB + 0x25b2ac4))
b2 = B2("*0x%x" % (EB + 0x25acf40))
b3 = B3("*0x%x" % (EB + 0x25b2980))
t0 = time.time()
gdb.execute("continue")
dt = time.time() - t0
thr = gdb.selected_thread()
this = hit["this"]
ts = hit["ts"]
print("HIT %s after %.3fs on lwp=%d name=%s this=0x%x tstate=0x%x" %
      (hit["where"], dt, thr.ptid[1], thr.name or "?", this, ts))
b1.delete(); b2.delete(); b3.delete()

qnum = rd(this + 0x58, 4)
print("QueuedPackagesCounter   [this+0x1b0] =", hx(rd(this + 0x1b0, 4)))
print("ExistingAsyncPkgCounter [this+0x1b4] =", hx(rd(this + 0x1b4, 4)))
print("ctr [this+0x140] =", hx(rd(this + 0x140, 4)), " heapnum [this+0x5d0] =", hx(rd(this + 0x5d0, 4)))
print("suspend [this+0x19] =", hx(rd(this + 0x19, 1)), " [this+0x18] =", hx(rd(this + 0x18, 1)))
print("QueuedPackages num [this+0x58] =", hx(qnum), " data =", hx(rd(this + 0x50)))
erq = rd(this + 0x138)
print("ExternalReadQueue [this+0x138] =", hx(erq), "-> next", hx(rd(erq or 0)))
ddq = rd(this + 0xa8)
print("DeferredDeleteQ   [this+0xa8]  =", hx(ddq), "-> next", hx(rd(ddq or 0)))
print("evq arr [this+0x800658] =", hx(rd(this + 0x800658)), "cnt =", hx(rd(this + 0x800660, 4)))
print("ThreadState +0x18 defrees =", hx(rd(ts + 0x18, 4)), " +0x30 curnode =", hx(rd(ts + 0x30)),
      " +0x39 usetl =", hx(rd(ts + 0x39, 1)))

arr = rd(this + 0x800658)
cnt = rd(this + 0x800660, 4) or 0
if arr and cnt and cnt < 16:
    for i in range(cnt):
        q = rd(arr + 8 * i)
        if not q:
            continue
        print("  evq[%d]=0x%x zenaphore=%s commit(+0x8)=%s pop(+0x10)=%s ring[pop&mask]=%s" % (
            i, q, hx(rd(q)), hx(rd(q + 8)), hx(rd(q + 0x10)),
            hx(rd(q + 0x18 + 8 * ((rd(q + 0x10) or 0) & 0x7ffff)))))

# AsyncPackageLookup: enumerate all packages + state bytes
slots = rd(this + 0xe0, 4) or 0
free = rd(this + 0x10c, 4) or 0
data = rd(this + 0xd8)
print("AsyncPackageLookup: slots=%d free=%d live=%d data=%s" % (slots, free, slots - free, hx(data)))
if data and 0 < slots < 20000:
    states = {}
    stuck = []
    for i in range(slots):
        e = data + 24 * i
        pid = rd(e)
        pkg = rd(e + 8)
        if not pkg or pkg < 0x10000000 or pkg > 0x7000000000 or not pid:
            continue
        st = rd(pkg + 0xe0, 1)
        if st is None:
            continue
        states[st] = states.get(st, 0) + 1
        if st != 11 and len(stuck) < 40:
            stuck.append((pid, pkg, st))
    print("state histogram:", {k: v for k, v in sorted(states.items())})
    for pid, pkg, st in stuck:
        print("  pkg=0x%x id=0x%016x state=%d extreads num=%s idx=%s refc(+0x140)=%s desc0x10=%s" % (
            pkg, pid, st, hx(rd(pkg + 0x118, 4)), hx(rd(pkg + 0xd8, 4)),
            hx(rd(pkg + 0x140, 4)), hx(rd(rd(pkg + 0x10) or 0))))
        # dump some raw package words for layout discovery
        print("    raw:", " ".join(hx(rd(pkg + o)) for o in range(0xd0, 0x120, 8)))
gdb.execute("detach")
