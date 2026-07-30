#!/bin/bash
# Issue #232: does sceSystemServiceGetDisplaySafeAreaInfo=1.0 produce scene draws?
WT="${PROSPER_REPO_ROOT:?set to your checkout root}/.claude/worktrees/agent-a9e9a0302ab3fc359/prosper"
cd "$WT" || exit 1
export PROSPER_GUEST_FS=1 PROSPER_NULL_PAGE=1 PROSPER_RENDER=1 PROSPER_GFXLOG=1 PROSPER_FILELOG=1 PROSPER_EVLOG=1
[ -f /root/pid232.txt ] && kill -9 "$(cat /root/pid232.txt)" 2>/dev/null
sleep 1
PID=
for try in 1 2 3 4 5 6; do
  setsid ./build-linux/boot_trace /root/PPSA17942-app0 > /root/d232safe.log 2>&1 &
  PID=$!
  echo "try=$try pid=$PID"
  sleep 45
  kill -0 $PID 2>/dev/null && break
  echo "DIED (try $try)"
  PID=
done
[ -n "$PID" ] || { echo "ALL TRIES DIED"; exit 1; }
echo "$PID" > /root/pid232.txt
for i in $(seq 1 10); do
  f=$(grep -ac GpuFlip /root/d232safe.log)
  d=$(grep -ac "kind=DrawIndex" /root/d232safe.log)
  v=$(grep -a "draws: [1-9]" /root/d232safe.log | tail -1 | grep -oE "draws: [0-9]+")
  echo "t+$((45+i*30)) flips=$f drawpkts=$d lastdraws='$v'"
  sleep 30
  kill -0 $PID 2>/dev/null || { echo DIED; break; }
done
echo "safe-area calls: $(grep -ac 'GetDisplaySafeArea\|1n37q1Bvc5Y' /root/d232safe.log)"
kill -9 $PID 2>/dev/null
echo RUN232SAFE-DONE
