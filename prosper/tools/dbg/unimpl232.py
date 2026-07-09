# gdb -p PID -batch -x unimpl232.py — count prosper_on_unimpl hits (import index) per thread for 60 s.
import gdb, time, struct, collections

DURATION = 60
hits = collections.Counter()

class UBP(gdb.Breakpoint):
    def stop(self):
        try:
            idx = int(gdb.parse_and_eval("$rdi")) & 0xffffffff
            t = gdb.selected_thread()
            hits[(t.name or "?", idx)] += 1
        except Exception:
            pass
        return False

gdb.execute("set pagination off")
gdb.execute("handle SIGSEGV nostop pass noprint")
UBP("prosper_on_unimpl")
gdb.execute("continue &")
time.sleep(DURATION)
gdb.execute("interrupt")
time.sleep(2)
for (name, idx), n in hits.most_common(40):
    print("UNIMPL idx=%d thread=%s count=%d" % (idx, name, n), flush=True)
print("=== unimpl trace done ===", flush=True)
gdb.execute("detach")
