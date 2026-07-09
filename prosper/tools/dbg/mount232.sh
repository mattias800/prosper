#!/bin/bash
# Issue #232: boot DOLL and live-catch the savedata mount-wrapper callers with gdb.
WT=/mnt/c/Users/matti/repos/ps5ys/.claude/worktrees/agent-a2d51084866d8d045/prosper
cd "$WT" || exit 1
export PROSPER_GUEST_FS=1 PROSPER_NULL_PAGE=1 PROSPER_RENDER=1 PROSPER_GFXLOG=1 PROSPER_FILELOG=1 PROSPER_SVCLOG=1
pkill -9 boot_trace; sleep 1
PID=
for try in 1 2 3; do
  setsid ./build-linux/boot_trace /root/PPSA17942-app0 > /root/svc3.log 2>&1 &
  PID=$!
  echo "try=$try pid=$PID"
  sleep 22
  kill -0 $PID 2>/dev/null && break
  PID=
done
[ -n "$PID" ] || { echo ALL-DIED; exit 1; }
timeout 150 gdb -p $PID -batch -x "$WT/tools/dbg/mount232.gdb" > /root/mount232.out 2>&1
echo "=== gdb hits ==="
grep -aE "MOUNTWRAP|DIRSEARCH|^0x" /root/mount232.out | head -60
kill -9 $PID 2>/dev/null
echo MOUNT232-DONE
