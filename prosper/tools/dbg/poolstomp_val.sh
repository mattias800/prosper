#!/bin/bash
# #312 free-canary guard validation: N guard-ON runs, longer timeout, check for fatal/wedge + how far
# DOLL gets. Vars live in the file (safe from cmdline stripping). Usage: poolstomp_val.sh <N> <timeout_s>
WT=/mnt/c/Users/matti/repos/ps5ys/.claude/worktrees/agent-a5d8efe035816c6c7/prosper
cd "$WT" || exit 1
N=${1:-4}; TMO=${2:-135}; GUARD=${3:-1}
SCRIPT="15:cross;20:start;25:cross;30:start;35:cross;40:cross;45:start;50:cross;60:cross;70:start;80:cross;90:cross;100:cross;110:cross;120:cross;130:cross"
pkill -9 boot_trace 2>/dev/null; sleep 1
fatal=0; clean=0; oom=0
for n in $(seq 1 "$N"); do
  log=/root/doll312_val_g${GUARD}_${n}.log
  timeout "$TMO" env PROSPER_GUEST_FS=1 PROSPER_NULL_PAGE=1 PROSPER_PROGRESS=20 \
    PROSPER_FREECANARY_GUARD=$GUARD "PROSPER_PAD_SCRIPT=${SCRIPT}" \
    ./build-linux/boot_trace /root/PPSA17942-app0 > "$log" 2>&1
  rc=$?
  f=$(grep -cE "MallocBinned3 Corruption|unrecognized block|WORKER-THREAD FAULT" "$log")
  sh=$(grep -c "20015f00" "$log")
  prog=$(grep "\[progress\]" "$log" | tail -1 | grep -oE "t=[0-9.]+s|draws_cum=[0-9]+|flips=[0-9]+")
  if [ "$f" -gt 0 ]; then fatal=$((fatal+1)); elif [ "$rc" = "137" ]; then oom=$((oom+1)); else clean=$((clean+1)); fi
  echo "  [val n=$n] rc=$rc fatal=$f 20015f00=$sh | $(echo $prog | tr '\n' ' ')"
done
echo "=== GUARD-ON VALIDATION: FATAL=${fatal}/${N} CLEAN=${clean}/${N} OOM=${oom}/${N} ==="
