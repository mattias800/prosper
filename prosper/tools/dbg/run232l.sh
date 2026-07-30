#!/bin/bash
# Issue #232/#241: stability A/B — with the exact-length w1K fold, how many of 6 boots survive 90 s?
WT="${PROSPER_REPO_ROOT:?set to your checkout root}/.claude/worktrees/agent-a36ca69a115e7c61a/prosper"
cd "$WT" || exit 1
export PROSPER_GUEST_FS=1 PROSPER_NULL_PAGE=1 PROSPER_RENDER=1 PROSPER_GFXLOG=1 PROSPER_FILELOG=1 PROSPER_EVLOG=1
ok=0
for try in 1 2 3 4 5 6; do
  ./build-linux/boot_trace /root/PPSA17942-app0 > /root/d232l_$try.log 2>&1 &
  PID=$!
  el=0; alive=1
  while [ $el -lt 90 ]; do
    sleep 15; el=$((el+15))
    kill -0 $PID 2>/dev/null || { alive=0; break; }
  done
  f=$(grep -c GpuFlip /root/d232l_$try.log)
  w=$(grep -c "SubmitDcbFinal" /root/d232l_$try.log)
  if [ $alive -eq 1 ]; then
    ok=$((ok+1))
    echo "try=$try SURVIVED flips=$f finalsubmits=$w"
    grep -a "SubmitDcbFinal #" /root/d232l_$try.log | head -3 | cut -c1-160
    grep -ac "arg9 count unavailable" /root/d232l_$try.log
  else
    echo "try=$try DIED flips=$f finalsubmits=$w: $(grep -a 'WORKER-THREAD FAULT' /root/d232l_$try.log | head -1 | cut -c1-120)"
    tail -2 /root/d232l_$try.log | cut -c1-140
  fi
  kill -9 $PID 2>/dev/null
  wait $PID 2>/dev/null
done
echo "SURVIVED $ok of 6"
echo RUN232L-DONE
