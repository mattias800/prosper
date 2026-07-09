#!/bin/bash
# Issue #232: with ORDERED ptr-tag completions, does the load stream past the old wall into draws?
WT=/mnt/c/Users/matti/repos/ps5ys/.claude/worktrees/agent-a36ca69a115e7c61a/prosper
cd "$WT" || exit 1
export PROSPER_GUEST_FS=1 PROSPER_NULL_PAGE=1 PROSPER_RENDER=1 PROSPER_GFXLOG=1 PROSPER_FILELOG=1 PROSPER_EVLOG=1
PID=
for try in 1 2 3 4 5 6 7 8; do
  ./build-linux/boot_trace /root/PPSA17942-app0 > /root/d232r.log 2>&1 &
  PID=$!
  echo "try=$try pid=$PID"
  el=0
  while kill -0 $PID 2>/dev/null && [ $el -lt 300 ]; do
    sleep 20; el=$((el+20))
    f=$(grep -ac GpuFlip /root/d232r.log)
    r=$(grep -ac "read-submit id" /root/d232r.log)
    d=$(grep -ac "kind=DrawIndex" /root/d232r.log)
    c=$(grep -ac "DcbSetCxRegistersIndirect" /root/d232r.log)
    v=$(grep -ac "draws: [1-9]" /root/d232r.log)
    echo "t=$el flips=$f reads=$r drawpkts=$d cxind=$c drawsub=$v"
  done
  kill -0 $PID 2>/dev/null && break
  echo "DIED (try $try): $(grep -a 'WORKER-THREAD FAULT' /root/d232r.log | head -1 | cut -c1-110)"
  PID=
done
[ -n "$PID" ] || { echo "ALL TRIES DIED"; exit 1; }
kill -9 $PID 2>/dev/null
echo "=== final ==="
echo "reads=$(grep -ac 'read-submit id' /root/d232r.log) drawpkts=$(grep -ac 'kind=DrawIndex' /root/d232r.log)"
grep -a "read-submit id" /root/d232r.log | tail -2 | cut -c1-150
grep -a "draws: [1-9]" /root/d232r.log | tail -3 | cut -c1-150
echo RUN232R-DONE
