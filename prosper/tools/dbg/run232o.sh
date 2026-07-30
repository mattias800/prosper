#!/bin/bash
# Issue #232: two-phase live capture — find disp early, dump dispatcher state at the stall.
WT="${PROSPER_REPO_ROOT:?set to your checkout root}/.claude/worktrees/agent-a36ca69a115e7c61a/prosper"
cd "$WT" || exit 1
export PROSPER_GUEST_FS=1 PROSPER_NULL_PAGE=1 PROSPER_RENDER=1 PROSPER_GFXLOG=1 PROSPER_FILELOG=1 PROSPER_EVLOG=1
PID=
for try in 1 2 3 4 5 6 7 8; do
  ./build-linux/boot_trace /root/PPSA17942-app0 > /root/d232o.log 2>&1 &
  PID=$!
  echo "try=$try pid=$PID"
  sleep 16
  kill -0 $PID 2>/dev/null && break
  echo "DIED (try $try) flips=$(grep -ac GpuFlip /root/d232o.log)"
  PID=
done
[ -n "$PID" ] || { echo "ALL TRIES DIED"; exit 1; }
rm -f /root/disp.txt
echo "=== phase 1: find disp (t=16s, reads flowing) ==="
timeout 150 gdb -p $PID -batch -x "$WT/tools/dbg/find_disp.py" 2>&1 | tail -6
test -f /root/disp.txt || { echo "NO DISP CAPTURED"; kill -9 $PID; exit 1; }
echo "disp=$(cat /root/disp.txt)"
# survive check + wait for the stall: reads stop; give it until t=150s
sleep 120
kill -0 $PID 2>/dev/null || { echo "DIED during wait"; tail -3 /root/d232o.log | cut -c1-140; exit 1; }
lastread1=$(grep -ac "read-submit id" /root/d232o.log)
sleep 20
lastread2=$(grep -ac "read-submit id" /root/d232o.log)
echo "reads t-20s=$lastread1 now=$lastread2 (equal => stalled)"
echo "=== phase 2: dump dispatcher state ==="
timeout 90 gdb -p $PID -batch -x "$WT/tools/dbg/dump_disp.py" 2>&1 | grep -v "^\[New\|libthread_db\|debuginfod\|Debuginfod\|warning:\|^0x00007\|clock_nanosleep\|^\[Thread\|answered N"
kill -9 $PID 2>/dev/null
echo RUN232O-DONE
