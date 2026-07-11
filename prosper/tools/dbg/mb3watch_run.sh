#!/bin/bash
# #312 MB3WATCH capture harness. Arms a per-thread HW write-watch on the MB3 per-thread pool cache
# head slot (base+0x20) the instant getspecific/setspecific hands the guest that base — before any
# store — to catch the off-by-one corruptor in the act. Vars live in this file (safe from cmdline
# stripping). Usage: mb3watch_run.sh <first> <count> [extra env VAR=VAL ...]
WT=/mnt/c/Users/matti/repos/ps5ys/.claude/worktrees/agent-a48ff9e03b626cc45/prosper
cd "$WT" || exit 1
FIRST=${1:-1}; COUNT=${2:-1}; shift 2 2>/dev/null
SCRIPT="15:cross;20:start;25:cross;30:start;35:cross;40:cross;45:start;50:cross;60:cross;70:start;80:cross;90:cross;100:cross;110:cross;120:cross;130:cross"
pkill -9 boot_trace 2>/dev/null; sleep 1
for n in $(seq "$FIRST" $((FIRST + COUNT - 1))); do
  LOG=/root/doll312_mb3_${n}.log
  timeout 150 env PROSPER_GUEST_FS=1 PROSPER_NULL_PAGE=1 PROSPER_PROGRESS=10 \
    PROSPER_MB3WATCH=1 \
    "PROSPER_PAD_SCRIPT=${SCRIPT}" "$@" ./build-linux/boot_trace /root/PPSA17942-app0 > "$LOG" 2>&1
  RC=$?
  ARMED=$(grep -c "\[mb3watch\] ARMED" "$LOG")
  HITS=$(grep -c "\[mb3watch\] WRITE" "$LOG")
  CORRUPT=$(grep -c "POOLSHIFT CORRUPTOR" "$LOG")
  CRASH=$(grep -oE "Canary was 0x[0-9a-f]+|unrecognized block [0-9a-fx]+|WORKER-THREAD FAULT[^\n]*" "$LOG" | head -1)
  PROG=$(grep "\[progress\]" "$LOG" | tail -1 | grep -oE "t=[0-9.]+s|draws_cum=[0-9]+|flips=[0-9]+" | tr '\n' ' ')
  echo "=== run ${n} rc=${RC} armed=${ARMED} writes=${HITS} CORRUPTOR=${CORRUPT} crash=[${CRASH}]"
  echo "    ${PROG}"
  if [ "$CORRUPT" -gt 0 ]; then
    echo "    --- CORRUPTOR CAPTURE ---"
    grep -A6 "POOLSHIFT CORRUPTOR" "$LOG" | head -30
  fi
done
