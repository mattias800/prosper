#!/bin/bash
# #312: test whether SYNCHRONOUS fence writes (PROSPER_EOP_WRITE_SYNC=1, no pend deferral) + the WAF
# guard close the gate. Usage: waf_sync.sh <first> <N> <timeout_s> [SYNC] [GUARD]
WT="${PROSPER_REPO_ROOT:?set to your checkout root}/.claude/worktrees/agent-a8f10a7598f626404/prosper"
cd "$WT" || exit 1
FIRST=${1:-1}; N=${2:-3}; TMO=${3:-120}; SYNC=${4:-1}; GUARD=${5:-1}
SCRIPT="15:cross;20:start;25:cross;30:start;35:cross;40:cross;45:start;50:cross;60:cross;70:start;80:cross;90:cross;100:cross;110:cross;120:cross;130:cross"
pkill -9 boot_trace 2>/dev/null; sleep 1
fatal=0; clean=0
for i in $(seq 0 $((N-1))); do
  n=$((FIRST+i))
  log=/root/doll312_sync_s${SYNC}_g${GUARD}_${n}.log
  timeout "$TMO" env PROSPER_GUEST_FS=1 PROSPER_NULL_PAGE=1 PROSPER_PROGRESS=20 \
    PROSPER_EOP_WRITE_SYNC=$SYNC PROSPER_REL1_WAF_GUARD=$GUARD "PROSPER_PAD_SCRIPT=${SCRIPT}" \
    ./build-linux/boot_trace /root/PPSA17942-app0 > "$log" 2>&1
  rc=$?
  f=$(grep -cE "MallocBinned3 Corruption|unrecognized block|WORKER-THREAD FAULT|Canary was 0x" "$log")
  can=$(grep -oE "Canary was 0x[0-9a-f]+|unrecognized block [0-9a-f]+|WORKER-THREAD FAULT[^\"]*" "$log" | head -1)
  prog=$(grep "\[progress\]" "$log" | tail -1 | grep -oE "t=[0-9.]+s|draws_cum=[0-9]+|flips=[0-9]+" | tr '\n' ' ')
  if [ "$f" -gt 0 ]; then fatal=$((fatal+1)); tag=FATAL; else clean=$((clean+1)); tag=clean; fi
  echo "  [sync=$SYNC g=$GUARD n=$n] rc=$rc $tag fatals=$f [$can] | $prog"
done
echo "=== SYNC=$SYNC GUARD=$GUARD : FATAL=${fatal}/${N} CLEAN=${clean}/${N} ==="
