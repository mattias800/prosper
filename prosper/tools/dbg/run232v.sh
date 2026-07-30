#!/bin/bash
# Issue #232: A/B — deny all .usm movie files; does DOLL advance past the (suspected)
# opening-cutscene gate into real scene draws?
WT="${PROSPER_REPO_ROOT:?set to your checkout root}/.claude/worktrees/agent-a9e9a0302ab3fc359/prosper"
cd "$WT" || exit 1
export PROSPER_GUEST_FS=1 PROSPER_NULL_PAGE=1 PROSPER_RENDER=1 PROSPER_GFXLOG=1 PROSPER_FILELOG=1 PROSPER_EVLOG=1
export PROSPER_DENY_SUBSTR=.usm
[ -f /root/pid232.txt ] && kill -9 "$(cat /root/pid232.txt)" 2>/dev/null
sleep 1
PID=
for try in 1 2 3 4 5 6; do
  setsid ./build-linux/boot_trace /root/PPSA17942-app0 > /root/d232v.log 2>&1 &
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
  r=$(grep -ac "read-submit id" /root/d232v.log)
  d=$(grep -ac "kind=DrawIndex" /root/d232v.log)
  di=$(grep -ac "DrawIndexAuto" /root/d232v.log)
  f=$(grep -ac GpuFlip /root/d232v.log)
  dn=$(grep -ac "DENIED" /root/d232v.log)
  v=$(grep -ac "draws: [1-9]" /root/d232v.log)
  echo "t+$((45+i*30)) flips=$f reads=$r denied=$dn drawpkts=$d drawauto=$di drawsub=$v"
  sleep 30
  kill -0 $PID 2>/dev/null || { echo DIED; break; }
done
echo RUN232V-DONE
