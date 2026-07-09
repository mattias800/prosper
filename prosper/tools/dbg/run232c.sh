#!/bin/bash
# Issue #232: boot DOLL to the park point (flips stop ~175), then take N spaced gdb python samples.
WT=/mnt/c/Users/matti/repos/ps5ys/.claude/worktrees/agent-af9660f2dbcb26e7d/prosper
cd "$WT" || exit 1
export PROSPER_GUEST_FS=1 PROSPER_NULL_PAGE=1 PROSPER_RENDER=1 PROSPER_GFXLOG=1 PROSPER_FILELOG=1 PROSPER_EVLOG=1
./build-linux/boot_trace /root/PPSA17942-app0 > /root/d232.log 2>&1 &
PID=$!
echo "pid=$PID"
# wait until flips plateau (no change across 2 consecutive 10s windows, flips>100)
last=0; same=0; el=0
while kill -0 $PID 2>/dev/null; do
  sleep 10; el=$((el+10))
  cur=$(grep -c GpuFlip /root/d232.log 2>/dev/null)
  echo "t=$el flips=$cur"
  if [ "$cur" -gt 100 ] && [ "$cur" -eq "$last" ]; then same=$((same+1)); else same=0; fi
  last=$cur
  if [ "$same" -ge 2 ]; then echo PLATEAU; break; fi
  if [ "$el" -ge 240 ]; then echo giveup; break; fi
done
kill -0 $PID 2>/dev/null || { echo DIED; tail -20 /root/d232.log; exit 1; }
for i in 1 2 3; do
  gdb -p $PID -batch -x "$WT/tools/dbg/gw232.py" > /root/gw232_$i.txt 2>&1
  echo "sample $i lines=$(wc -l < /root/gw232_$i.txt)"
  sleep 12
done
kill -9 $PID 2>/dev/null
echo RUN232C-DONE
