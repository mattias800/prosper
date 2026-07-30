#!/bin/bash
# Issue #232: boot to the FlushAsyncLoading stall, then probe FAsyncLoadingThread2 state.
WT="${PROSPER_REPO_ROOT:?set to your checkout root}/.claude/worktrees/agent-a9e9a0302ab3fc359/prosper"
cd "$WT" || exit 1
export PROSPER_GUEST_FS=1 PROSPER_NULL_PAGE=1 PROSPER_RENDER=1 PROSPER_GFXLOG=1 PROSPER_FILELOG=1 PROSPER_EVLOG=1
PID=
for try in 1 2 3 4 5 6; do
  ./build-linux/boot_trace /root/PPSA17942-app0 > /root/d232u.log 2>&1 &
  PID=$!
  echo "try=$try pid=$PID"
  sleep 45
  kill -0 $PID 2>/dev/null && break
  echo "DIED (try $try)"
  PID=
done
[ -n "$PID" ] || { echo "ALL TRIES DIED"; exit 1; }
# wait for the stall: reads frozen for 30 s and > 2000
prev=-1
for i in $(seq 1 12); do
  cur=$(grep -ac "read-submit id" /root/d232u.log)
  echo "t+$((45+i*30)) reads=$cur"
  if [ "$cur" = "$prev" ] && [ "$cur" -gt 2000 ]; then echo STALLED; break; fi
  prev=$cur
  sleep 30
  kill -0 $PID 2>/dev/null || { echo DIED-WAITING; exit 1; }
done
echo "$PID" > /root/pid232.txt
timeout 180 gdb -p $PID -batch -x "$WT/tools/dbg/probe232u.py" 2>&1 \
  | grep -v "^\[New\|libthread_db\|debuginfod\|Debuginfod\|warning:\|^\[Thread\|^Downloading" | tail -80
kill -9 $PID 2>/dev/null
echo RUN232U-DONE
