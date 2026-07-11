#!/bin/bash
# #312 menu-drive repro + inline summary. Usage: run312.sh <tag> [extra env as VAR=VAL ...]
# Log: /root/doll312x_<tag>.log
TAG=$1; shift
WT=/mnt/c/Users/matti/repos/ps5ys/.claude/worktrees/agent-a8ffd958941d2a852/prosper
LOG=/root/doll312x_${TAG}.log
cd "$WT" || exit 1
timeout 170 env PROSPER_GUEST_FS=1 PROSPER_NULL_PAGE=1 PROSPER_FILELOG=1 PROSPER_PROGRESS=5 \
  PROSPER_PROGRESS_UNIMPL=1 \
  "PROSPER_PAD_SCRIPT=15:cross;20:start;25:cross;30:start;35:cross;40:cross;45:start;50:cross;60:cross;70:start;80:cross;90:cross;100:cross;110:cross" \
  "$@" ./build-linux/boot_trace /root/PPSA17942-app0 > "$LOG" 2>&1
RC=$?
echo "=== ${TAG} rc=${RC} ==="
FATALS=$(grep -cE "MallocBinned3 Corruption|unrecognized block" "$LOG")
WFAULT=$(grep -c "WORKER-THREAD FAULT" "$LOG")
LIVE=$(grep -c "SUSPECT-REL1-LIVE" "$LOG")
NOINIT=$(grep -c "SUSPECT-REL1-NOINIT" "$LOG")
VIOL=$(grep -c "NOT satisfied" "$LOG")
echo "fatals=${FATALS} workerfaults=${WFAULT} LIVE=${LIVE} NOINIT=${NOINIT} waitviol=${VIOL}"
grep -nE "MallocBinned3 Corruption|unrecognized block|WORKER-THREAD FAULT" "$LOG" | head -4
grep "\[progress\]" "$LOG" | tail -1
