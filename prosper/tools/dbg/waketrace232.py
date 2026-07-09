# gdb -p PID -batch -x waketrace232.py — at the DOLL plateau, trace the wake chain:
# every k_cond_timedwait entry (thread, cond slot, timeout-from-now), every cond signal/broadcast,
# and every AGC submit. Non-stop logging; detaches after DURATION seconds.
import gdb, time, struct

DURATION = 150

def tname():
    t = gdb.selected_thread()
    return "%s(lwp %d)" % (t.name or "?", t.ptid[1])

def guest_ra():
    try:
        sp = int(gdb.parse_and_eval("$sp"))
        inf = gdb.selected_inferior()
        mem = bytes(inf.read_memory(sp, 8))
        v = struct.unpack("<Q", mem)[0]
        if 0x400000000 <= v < 0x406700000:
            return "eboot+0x%x" % (v - 0x400000000)
        if 0x600000000 <= v < 0x610000000:
            return "prx+0x%x" % (v - 0x600000000)
        return hex(v)
    except gdb.error:
        return "?"

t0 = time.time()

class TimedWaitBP(gdb.Breakpoint):
    def stop(self):
        try:
            a0 = int(gdb.parse_and_eval("$rdi"))
            a2 = int(gdb.parse_and_eval("$rdx"))
            rem = -1.0
            if a2:
                inf = gdb.selected_inferior()
                mem = bytes(inf.read_memory(a2, 16))
                sec, nsec = struct.unpack("<qq", mem)
                rem = sec + nsec / 1e9 - time.time()
            print("[%7.2f] TIMEDWAIT %s cond_slot=0x%x timeout_in=%.3fs ra=%s"
                  % (time.time() - t0, tname(), a0, rem, guest_ra()), flush=True)
        except Exception as e:
            print("bp err", e, flush=True)
        return False

class SigBP(gdb.Breakpoint):
    def __init__(self, spec, label):
        super().__init__(spec)
        self.label = label
    def stop(self):
        try:
            a0 = int(gdb.parse_and_eval("$rdi"))
            print("[%7.2f] %s %s a0=0x%x ra=%s"
                  % (time.time() - t0, self.label, tname(), a0, guest_ra()), flush=True)
        except Exception as e:
            print("bp err", e, flush=True)
        return False

gdb.execute("set pagination off")
gdb.execute("handle SIGSEGV nostop pass noprint")
gdb.execute("handle SIGUSR1 nostop pass noprint")
gdb.execute("handle SIGUSR2 nostop pass noprint")

TimedWaitBP("prosper::k_cond_timedwait")
SigBP("prosper::k_cond_signal", "SIGNAL")
SigBP("prosper::k_cond_broadcast", "BCAST")
SigBP("prosper::agc_driver_submit_dcb", "SUBMIT")
SigBP("prosper::k_cond_wait", "CONDWAIT")

gdb.execute("continue &")
time.sleep(DURATION)
print("=== waketrace done ===", flush=True)
gdb.execute("interrupt")
time.sleep(2)
gdb.execute("detach")
