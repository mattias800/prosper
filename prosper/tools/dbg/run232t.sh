#!/bin/bash
# Issue #232: long soak — is the post-fix loader a hard wall or a crawl?
WT="${PROSPER_REPO_ROOT:?set to your checkout root}/.claude/worktrees/agent-a36ca69a115e7c61a/prosper"
cd "$WT" || exit 1
export PROSPER_GUEST_FS=1 PROSPER_NULL_PAGE=1 PROSPER_RENDER=1 PROSPER_GFXLOG=1 PROSPER_FILELOG=1 PROSPER_EVLOG=1
PID=
for try in 1 2 3 4 5 6; do
  ./build-linux/boot_trace /root/PPSA17942-app0 > /root/d232t.log 2>&1 &
  PID=$!
  echo "try=$try pid=$PID"
  sleep 45
  kill -0 $PID 2>/dev/null && break
  echo "DIED (try $try)"
  PID=
done
[ -n "$PID" ] || { echo "ALL TRIES DIED"; exit 1; }
for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
  r=$(grep -ac "read-submit id" /root/d232t.log)
  d=$(grep -ac "kind=DrawIndex" /root/d232t.log)
  o=$(grep -a "read-submit id" /root/d232t.log | tail -1 | grep -o "off=0x[0-9a-f]*")
  f=$(grep -ac GpuFlip /root/d232t.log)
  echo "t=$((45+i*60)) flips=$f reads=$r lastoff=$o drawpkts=$d"
  sleep 60
  kill -0 $PID 2>/dev/null || { echo DIED; break; }
done
kill -9 $PID 2>/dev/null
echo RUN232T-DONE
