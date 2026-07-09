#!/bin/bash
# Issue #232: strace the GameThread (main tid) in steady state + finer gdb probing of the tick loop.
STEADY_T=${1:-60}
WT=/mnt/c/Users/matti/repos/ps5ys/.claude/worktrees/agent-a36ca69a115e7c61a/prosper
cd "$WT" || exit 1
export PROSPER_GUEST_FS=1 PROSPER_NULL_PAGE=1 PROSPER_RENDER=1 PROSPER_GFXLOG=1 PROSPER_FILELOG=1 PROSPER_EVLOG=1
PID=
for try in 1 2 3 4 5 6 7 8 9 10 11 12; do
  ./build-linux/boot_trace /root/PPSA17942-app0 > /root/d232k.log 2>&1 &
  PID=$!
  echo "try=$try pid=$PID"
  el=0
  while kill -0 $PID 2>/dev/null && [ $el -lt $STEADY_T ]; do
    sleep 15; el=$((el+15))
    f=$(grep -c GpuFlip /root/d232k.log 2>/dev/null)
    echo "t=$el flips=$f"
  done
  kill -0 $PID 2>/dev/null && break
  echo "DIED (try $try): $(grep -a 'WORKER-THREAD FAULT' /root/d232k.log | head -1)"
  PID=
done
[ -n "$PID" ] || { echo "ALL TRIES DIED"; exit 1; }
echo "=== strace -c main tid 6s ==="
timeout 6 strace -c -p $PID 2>&1 | tail -25
echo "=== strace raw main tid 150 lines ==="
timeout 4 strace -p $PID -tt 2>&1 | head -150
kill -9 $PID 2>/dev/null
echo "flips=$(grep -c GpuFlip /root/d232k.log)"
echo RUN232K-DONE
