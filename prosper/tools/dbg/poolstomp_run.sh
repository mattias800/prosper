#!/bin/bash
# #312 poolstomp multi-run harness. Variables live in this FILE (bash reads them), so the
# claude wsl.exe cmdline-stripping of $VAR does not apply. Usage: poolstomp_run.sh <first> <count> [extra env...]
WT=/mnt/c/Users/matti/repos/ps5ys/.claude/worktrees/agent-a5d8efe035816c6c7/prosper
cd "$WT" || exit 1
FIRST=${1:-1}; COUNT=${2:-1}; shift 2 2>/dev/null
SCRIPT="15:cross;20:start;25:cross;30:start;35:cross;40:cross;45:start;50:cross;60:cross;70:start;80:cross;90:cross;100:cross;110:cross"
for n in $(seq "$FIRST" $((FIRST + COUNT - 1))); do
  LOG=/root/doll312_p${n}.log
  timeout 130 env PROSPER_GUEST_FS=1 PROSPER_NULL_PAGE=1 PROSPER_PROGRESS=10 \
    "PROSPER_PAD_SCRIPT=${SCRIPT}" "$@" ./build-linux/boot_trace /root/PPSA17942-app0 > "$LOG" 2>&1
  RC=$?
  POOL=$(grep -c "POOLSHIFT" "$LOG")
  CRASH=$(grep -oE "Canary was 0x[0-9a-f]+|unrecognized block [0-9a-fx]+" "$LOG" | head -1)
  SHIFT=$(grep -c "20015f00" "$LOG")
  SHIFTRIP=$(grep -oE "rip=0x[0-9a-f]+ \(image\+0x[0-9a-f]+\)" "$LOG" | head -1)
  PROG=$(grep "\[progress\]" "$LOG" | tail -1)
  echo "=== run ${n} rc=${RC} POOLSHIFT=${POOL} 20015f00=${SHIFT} crash=[${CRASH}] ${SHIFTRIP}"
  echo "    ${PROG}"
done
