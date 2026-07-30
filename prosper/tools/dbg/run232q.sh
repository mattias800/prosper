#!/bin/bash
# Issue #232: does pumping user events (dispatcher wakes) unstick the final APR batch?
WT="${PROSPER_REPO_ROOT:?set to your checkout root}/.claude/worktrees/agent-a36ca69a115e7c61a/prosper"
cd "$WT" || exit 1
export PROSPER_GUEST_FS=1 PROSPER_NULL_PAGE=1 PROSPER_RENDER=1 PROSPER_GFXLOG=1 PROSPER_FILELOG=1 PROSPER_EVLOG=1 PROSPER_PUMP_USEREV=1
PID=
for try in 1 2 3 4 5 6 7 8; do
  ./build-linux/boot_trace /root/PPSA17942-app0 > /root/d232q.log 2>&1 &
  PID=$!
  echo "try=$try pid=$PID"
  el=0
  while kill -0 $PID 2>/dev/null && [ $el -lt 240 ]; do
    sleep 20; el=$((el+20))
    f=$(grep -ac GpuFlip /root/d232q.log)
    r=$(grep -ac "read-submit id" /root/d232q.log)
    d=$(grep -ac "kind=DrawIndex" /root/d232q.log)
    c=$(grep -ac "DcbSetCxRegistersIndirect" /root/d232q.log)
    echo "t=$el flips=$f reads=$r drawpkts=$d cxind=$c"
  done
  kill -0 $PID 2>/dev/null && break
  echo "DIED (try $try): $(grep -a 'WORKER-THREAD FAULT' /root/d232q.log | head -1 | cut -c1-110)"
  PID=
done
[ -n "$PID" ] || { echo "ALL TRIES DIED"; exit 1; }
kill -9 $PID 2>/dev/null
echo "=== final ==="
echo "reads=$(grep -ac 'read-submit id' /root/d232q.log) draws=$(grep -ac 'kind=DrawIndex' /root/d232q.log)"
grep -a "read-submit id" /root/d232q.log | tail -2 | cut -c1-150
echo RUN232Q-DONE
