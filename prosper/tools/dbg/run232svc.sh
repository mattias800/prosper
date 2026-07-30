#!/bin/bash
# Issue #232 session 5: boot DOLL, capture the unimplemented-service call log + draw evidence.
# Usage: run232svc.sh <logname> [extra soak iterations]
WT="${PROSPER_REPO_ROOT:?set to your checkout root}/.claude/worktrees/agent-a2d51084866d8d045/prosper"
LOG=/root/${1:-d232svc}.log
ITERS=${2:-8}
cd "$WT" || exit 1
export PROSPER_GUEST_FS=1 PROSPER_NULL_PAGE=1 PROSPER_RENDER=1 PROSPER_GFXLOG=1 PROSPER_FILELOG=1 PROSPER_EVLOG=1 PROSPER_SVCLOG=1
[ -f /root/pid232.txt ] && kill -9 "$(cat /root/pid232.txt)" 2>/dev/null
sleep 1
PID=
for try in 1 2 3 4 5 6; do
  setsid ./build-linux/boot_trace /root/PPSA17942-app0 > "$LOG" 2>&1 &
  PID=$!
  echo "try=$try pid=$PID"
  sleep 45
  kill -0 $PID 2>/dev/null && break
  echo "DIED (try $try)"
  PID=
done
[ -n "$PID" ] || { echo "ALL TRIES DIED"; tail -30 "$LOG"; exit 1; }
echo "$PID" > /root/pid232.txt
for i in $(seq 1 $ITERS); do
  f=$(grep -ac GpuFlip "$LOG")
  d=$(grep -ac "kind=DrawIndex" "$LOG")
  echo "t+$((45+i*30)) flips=$f drawpkts=$d"
  sleep 30
  kill -0 $PID 2>/dev/null || { echo DIED; break; }
done
echo "=== unimplemented (svc libs) ==="
grep -a "unimplemented:" "$LOG" | grep -aE "PlayGo|SaveData|Trophy|Share|NpUniversal|NpWebApi|VoiceQoS|AppContent|SystemService"
echo "=== svc HLE calls logged ==="
grep -a "^\[svc\]" "$LOG" | head -120
echo "=== gates (eboot+0x9803460 / +0x9803470) ==="
if [ -f /root/pid232.txt ] && kill -0 "$(cat /root/pid232.txt)" 2>/dev/null; then
  gdb -p "$(cat /root/pid232.txt)" -batch -ex "set pagination off" \
      -ex "x/16bx 0x409803460" -ex "x/16bx 0x409803470" 2>/dev/null | tail -6
fi
echo "=== draw evidence ==="
grep -ac "kind=DrawIndex" "$LOG"
grep -a "draws: [1-9]" "$LOG" | tail -3
kill -9 $PID 2>/dev/null
echo RUN232SVC-DONE
