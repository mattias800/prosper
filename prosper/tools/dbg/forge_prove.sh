#!/bin/bash
# #312 forge-guard proof: N synchronous menu-drive runs (no MB3WATCH, undistorted timing) with the
# REL1-FORGE guard (default ON unless GUARD=0 passed). Tallies fatal/clean/oom + DOLL progression.
# Usage: forge_prove.sh <N> <timeout_s> [GUARD]   (GUARD=0 to A/B the baseline)
WT="${PROSPER_REPO_ROOT:?set to your checkout root}/.claude/worktrees/agent-a48ff9e03b626cc45/prosper"
cd "$WT" || exit 1
N=${1:-8}; TMO=${2:-150}; GUARD=${3:-1}
SCRIPT="15:cross;20:start;25:cross;30:start;35:cross;40:cross;45:start;50:cross;60:cross;70:start;80:cross;90:cross;100:cross;110:cross;120:cross;130:cross"
pkill -9 boot_trace 2>/dev/null; sleep 1
fatal=0; clean=0; oom=0
for n in $(seq 1 "$N"); do
  log=/root/doll312_forge_g${GUARD}_${n}.log
  timeout "$TMO" env PROSPER_GUEST_FS=1 PROSPER_NULL_PAGE=1 PROSPER_PROGRESS=20 \
    PROSPER_REL1_FORGE_GUARD=$GUARD "PROSPER_PAD_SCRIPT=${SCRIPT}" \
    ./build-linux/boot_trace /root/PPSA17942-app0 > "$log" 2>&1
  rc=$?
  f=$(grep -cE "MallocBinned3 Corruption|unrecognized block|WORKER-THREAD FAULT" "$log")
  sh=$(grep -c "20015f00" "$log")
  prog=$(grep "\[progress\]" "$log" | tail -1 | grep -oE "t=[0-9.]+s|draws_cum=[0-9]+|flips=[0-9]+" | tr '\n' ' ')
  if [ "$f" -gt 0 ]; then fatal=$((fatal+1)); tag=FATAL;
  elif [ "$rc" = "137" ]; then oom=$((oom+1)); tag=OOM;
  else clean=$((clean+1)); tag=clean; fi
  echo "  [g=$GUARD n=$n] rc=$rc $tag fatals=$f 20015f00=$sh | $prog"
done
echo "=== GUARD=$GUARD : FATAL=${fatal}/${N} CLEAN=${clean}/${N} OOM=${oom}/${N} ==="
