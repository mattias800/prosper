#!/bin/bash
# Issue #232: boot DOLL to the FlushAsyncLoading stall and LEAVE IT RUNNING.
# PID -> /root/pid232.txt; log -> /root/d232u.log. Kill with: kill -9 $(cat /root/pid232.txt)
WT=/mnt/c/Users/matti/repos/ps5ys/.claude/worktrees/agent-a9e9a0302ab3fc359/prosper
cd "$WT" || exit 1
export PROSPER_GUEST_FS=1 PROSPER_NULL_PAGE=1 PROSPER_RENDER=1 PROSPER_GFXLOG=1 PROSPER_FILELOG=1 PROSPER_EVLOG=1
# kill any previous instance we started
[ -f /root/pid232.txt ] && kill -9 "$(cat /root/pid232.txt)" 2>/dev/null
sleep 1
PID=
for try in 1 2 3 4 5 6; do
  setsid ./build-linux/boot_trace /root/PPSA17942-app0 > /root/d232u.log 2>&1 &
  PID=$!
  echo "try=$try pid=$PID"
  sleep 45
  kill -0 $PID 2>/dev/null && break
  echo "DIED (try $try)"
  PID=
done
[ -n "$PID" ] || { echo "ALL TRIES DIED"; exit 1; }
echo "$PID" > /root/pid232.txt
prev=-1
for i in $(seq 1 12); do
  cur=$(grep -ac "read-submit id" /root/d232u.log)
  echo "t+$((45+i*30)) reads=$cur"
  if [ "$cur" = "$prev" ] && [ "$cur" -gt 2000 ]; then echo READY; exit 0; fi
  prev=$cur
  sleep 30
  kill -0 $PID 2>/dev/null || { echo DIED-WAITING; exit 1; }
done
echo READY-MAYBE
