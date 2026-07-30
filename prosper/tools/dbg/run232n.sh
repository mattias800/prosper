#!/bin/bash
# Issue #232: live dispatcher-state capture at the stall.
WT="${PROSPER_REPO_ROOT:?set to your checkout root}/.claude/worktrees/agent-a36ca69a115e7c61a/prosper"
cd "$WT" || exit 1
export PROSPER_GUEST_FS=1 PROSPER_NULL_PAGE=1 PROSPER_RENDER=1 PROSPER_GFXLOG=1 PROSPER_FILELOG=1 PROSPER_EVLOG=1
PID=
for try in 1 2 3 4 5 6 7 8; do
  ./build-linux/boot_trace /root/PPSA17942-app0 > /root/d232n.log 2>&1 &
  PID=$!
  echo "try=$try pid=$PID"
  sleep 20
  kill -0 $PID 2>/dev/null && break
  echo "DIED (try $try) flips=$(grep -ac GpuFlip /root/d232n.log)"
  PID=
done
[ -n "$PID" ] || { echo "ALL TRIES DIED"; exit 1; }
# attach now (reads still flowing), capture disp on first pool pop, then sample through the stall
timeout 240 gdb -p $PID -batch -x "$WT/tools/dbg/disp232.py" 2>&1 | grep -v "^\[New\|^\[Thread\|libthread_db" | tail -80
echo "flips=$(grep -ac GpuFlip /root/d232n.log)"
echo "last-read: $(grep -a "read-submit id" /root/d232n.log | tail -1 | cut -c1-150)"
kill -9 $PID 2>/dev/null
echo RUN232N-DONE
