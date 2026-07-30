#!/bin/bash
WT="${PROSPER_REPO_ROOT:?set to your checkout root}/.claude/worktrees/agent-aa1d2a853141bdab4/prosper"
cd "$WT" || exit 1
export PROSPER_GUEST_FS=1 PROSPER_NULL_PAGE=1 PROSPER_RENDER=1 PROSPER_GFXLOG=1 PROSPER_FILELOG=1 PROSPER_EVLOG=1 PROSPER_SC2LOG=1
./build-linux/boot_trace /root/PPSA17942-app0 > /root/gwalk.log 2>&1 &
PID=$!
echo "pid=$PID"
last=0; same=0; el=0
while kill -0 $PID 2>/dev/null; do
  sleep 8; el=$((el+8))
  cur=$(grep -c GpuFlip /root/gwalk.log 2>/dev/null)
  echo "t=$el flips=$cur"
  if [ "$cur" -gt 100 ] && [ "$cur" -eq "$last" ]; then same=$((same+1)); else same=0; fi
  last=$cur
  if [ "$same" -ge 2 ]; then echo FROZEN; break; fi
  if [ "$el" -ge 300 ]; then echo giveup; break; fi
done
kill -0 $PID 2>/dev/null || { echo died; exit 1; }
cat > /tmp/gw.gdb << "GEOF"
set pagination off
handle all nostop pass noprint
define gs
  printf "  --host--\n"
  frame 5
  printf "  --guest RAs on stack--\n"
  set $p = $sp - 256
  set $e = $sp + 4096
  while $p < $e
    set $v = *(unsigned long long*)$p
    if ($v >= 0x400000000 && $v < 0x406700000)
      printf "    eboot+0x%llx\n", $v - 0x400000000
    end
    set $p = $p + 8
  end
end
thread apply all gs
GEOF
gdb -p $PID -batch -x /tmp/gw.gdb > /root/gwalk_bt.txt 2>&1
kill -9 $PID 2>/dev/null
echo GWALK-DONE
# who signaled conds last (SC2LOG) before freeze:
echo "=== last 25 cond signals ==="
grep "COND.signal\|COND.broadcast" /root/gwalk.log | tail -25
