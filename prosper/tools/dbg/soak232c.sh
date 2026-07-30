#!/bin/bash
# Issue #232 session 6: boot DOLL to steady state, leave running. PID -> /root/pid232s6.txt.
# NO pkill (other agents run boot_trace on this WSL); kill only our own recorded pid.
WT="${PROSPER_REPO_ROOT:?set to your checkout root}/.claude/worktrees/agent-ac66b5fec5918f7aa/prosper"
LOG=/root/${1:-s6}.log
cd "$WT" || exit 1
export PROSPER_GUEST_FS=1 PROSPER_NULL_PAGE=1 PROSPER_RENDER=1 PROSPER_GFXLOG=1 PROSPER_FILELOG=1 PROSPER_SVCLOG=1
[ -f /root/pid232s6.txt ] && kill -9 "$(cat /root/pid232s6.txt)" 2>/dev/null
sleep 1
rm -f "$LOG"
# run under a different comm name: other agents pkill -9 boot_trace on this shared WSL
cp -f ./build-linux/boot_trace /root/bt232
PID=
for try in 1 2 3 4 5 6; do
  setsid /root/bt232 /root/PPSA17942-app0 > "$LOG" 2>&1 &
  PID=$!
  echo "try=$try pid=$PID"
  sleep 45
  kill -0 $PID 2>/dev/null && break
  echo "DIED (try $try)"
  PID=
done
[ -n "$PID" ] || { echo "ALL TRIES DIED"; tail -20 "$LOG"; exit 1; }
echo "$PID" > /root/pid232s6.txt
for i in $(seq 1 ${2:-5}); do
  sleep 30
  kill -0 $PID 2>/dev/null || { echo DIED-SOAK; tail -20 "$LOG"; exit 1; }
  echo "t+$((45+i*30)) flips=$(grep -ac GpuFlip "$LOG") draws=$(grep -ac "kind=DrawIndex" "$LOG")"
done
echo SOAK-READY pid=$PID
