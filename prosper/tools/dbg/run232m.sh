#!/bin/bash
# Issue #232: capture the APR channel tail (AMPRLOG) at the moment async loading goes silent.
WT="${PROSPER_REPO_ROOT:?set to your checkout root}/.claude/worktrees/agent-a36ca69a115e7c61a/prosper"
cd "$WT" || exit 1
export PROSPER_GUEST_FS=1 PROSPER_NULL_PAGE=1 PROSPER_RENDER=1 PROSPER_GFXLOG=1 PROSPER_FILELOG=1 PROSPER_EVLOG=1 PROSPER_AMPRLOG=1
PID=
for try in 1 2 3 4 5 6 7 8; do
  ./build-linux/boot_trace /root/PPSA17942-app0 > /root/d232m.log 2>&1 &
  PID=$!
  echo "try=$try pid=$PID"
  el=0
  while kill -0 $PID 2>/dev/null && [ $el -lt 120 ]; do
    sleep 15; el=$((el+15))
    echo "t=$el flips=$(grep -ac GpuFlip /root/d232m.log)"
  done
  kill -0 $PID 2>/dev/null && break
  echo "DIED (try $try)"
  PID=
done
[ -n "$PID" ] || { echo "ALL TRIES DIED"; exit 1; }
kill -9 $PID 2>/dev/null
echo "=== last read-submits ==="
grep -a "read-submit id\|read-direct id" /root/d232m.log | tail -6 | cut -c1-180
echo "=== last 30 amprlog/apr lines ==="
grep -an "amprlog\|\[apr\]" /root/d232m.log | tail -40 | cut -c1-190
echo "=== last AprPtrTagComplete / AprTagComplete ==="
grep -an "AprPtrTagComplete\|AprTagComplete" /root/d232m.log | tail -6 | cut -c1-160
echo "=== ASoW submits total / after last completion ==="
grep -ac "ASoW5WE-UPo(Submit)" /root/d232m.log
echo RUN232M-DONE
