#!/bin/bash
# Run DOLL; when flips FREEZE (the poll wall), attach and deep-walk the poll object.
WT=/mnt/c/Users/matti/repos/ps5ys/.claude/worktrees/agent-aa1d2a853141bdab4/prosper
cd "$WT" || exit 1
export PROSPER_GUEST_FS=1 PROSPER_NULL_PAGE=1 PROSPER_RENDER=1 PROSPER_GFXLOG=1 PROSPER_FILELOG=1 PROSPER_EVLOG=1
./build-linux/boot_trace /root/PPSA17942-app0 > /root/deep.log 2>&1 &
PID=$!
echo "pid=$PID"
last=0; same=0; el=0
while kill -0 $PID 2>/dev/null; do
  sleep 8; el=$((el+8))
  cur=$(grep -c GpuFlip /root/deep.log 2>/dev/null)
  echo "t=$el flips=$cur"
  if [ "$cur" -gt 100 ] && [ "$cur" -eq "$last" ]; then same=$((same+1)); else same=0; fi
  last=$cur
  if [ "$same" -ge 2 ]; then echo "FROZEN at $cur"; break; fi
  if [ "$el" -ge 400 ]; then echo "gave up waiting for freeze"; break; fi
done
if ! kill -0 $PID 2>/dev/null; then echo "died"; tail -20 /root/deep.log; exit 1; fi
gdb -p $PID -batch -x "$WT/tools/dbg/poll_deep.gdb" > /root/poll_deep.txt 2>&1
kill -9 $PID 2>/dev/null
echo DEEP-DONE
