#!/bin/bash
# Issue #232 post-fence: robust per-thread sampler (python walker, error-safe).
# Boots DOLL, waits STEADY_T, then runs gw232.py 3x spaced 12 s.
STEADY_T=${1:-150}
WT=/mnt/c/Users/matti/repos/ps5ys/.claude/worktrees/agent-a36ca69a115e7c61a/prosper
cd "$WT" || exit 1
export PROSPER_GUEST_FS=1 PROSPER_NULL_PAGE=1 PROSPER_RENDER=1 PROSPER_GFXLOG=1 PROSPER_FILELOG=1 PROSPER_EVLOG=1
PID=
for try in 1 2 3 4 5 6 7 8 9 10 11 12; do
  ./build-linux/boot_trace /root/PPSA17942-app0 > /root/d232j.log 2>&1 &
  PID=$!
  echo "try=$try pid=$PID"
  el=0
  while kill -0 $PID 2>/dev/null && [ $el -lt $STEADY_T ]; do
    sleep 15; el=$((el+15))
    f=$(grep -c GpuFlip /root/d232j.log 2>/dev/null)
    echo "t=$el flips=$f"
  done
  kill -0 $PID 2>/dev/null && break
  echo "DIED (try $try): $(grep -a 'WORKER-THREAD FAULT' /root/d232j.log | head -1)"
  cp /root/d232j.log /root/d232j_died_$try.log
  PID=
done
[ -n "$PID" ] || { echo "ALL TRIES DIED"; exit 1; }
for i in 1 2 3; do
  echo "=== SAMPLE $i ==="
  gdb -p $PID -batch -x "$WT/tools/dbg/gw232.py" > /root/gwj_$i.txt 2>&1
  echo "lines: $(wc -l < /root/gwj_$i.txt)"
  sleep 12
done
kill -9 $PID 2>/dev/null
echo "=== thread1 (GameThread) all samples ==="
for i in 1 2 3; do
  echo "--- sample $i ---"
  grep -A3 "lwp=$PID " /root/gwj_$i.txt
done
echo "=== flips at end: $(grep -c GpuFlip /root/d232j.log) ==="
echo RUN232J-DONE
