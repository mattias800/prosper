# gdb -p PID -batch -x find_disp.py — break on the pool pop once, record disp to /root/disp.txt
import gdb
gdb.execute("set pagination off")
gdb.execute("handle SIGSEGV SIGBUS SIGILL nostop pass noprint")
for s in range(34, 39):
    gdb.execute("handle SIG%d nostop pass noprint" % s)
gdb.execute("tbreak *0x4023d5a20")
gdb.execute("continue")
try:
    rdi = int(gdb.parse_and_eval("$rdi"))
    open("/root/disp.txt", "w").write(hex(rdi - 8))
    gdb.write("[find_disp] disp=%s\n" % hex(rdi - 8))
except gdb.error as e:
    gdb.write("[find_disp] failed: %s\n" % e)
gdb.execute("detach")
