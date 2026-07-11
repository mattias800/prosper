#!/bin/bash
# #312 free-canary-guard A/B. Runs N baseline (guard=0) + N guard-on (guard=1) boots, timeout-bounded,
# tallies MallocBinned3 fatals per arm. All logic in this FILE so claude's wsl.exe $VAR-stripping
# does not apply. Usage: poolstomp_ab.sh <N> <timeout_s>
WT=/mnt/c/Users/matti/repos/ps5ys/.claude/worktrees/agent-a5d8efe035816c6c7/prosper
cd "$WT" || exit 1
N=${1:-4}; TMO=${2:-110}
SCRIPT="15:cross;20:start;25:cross;30:start;35:cross;40:cross;45:start;50:cross;60:cross;70:start;80:cross;90:cross;100:cross;110:cross"
pkill -9 boot_trace 2>/dev/null; sleep 1
run_arm() {
  local guard=$1 tag=$2 fatal=0 clean=0 shift=0
  for n in $(seq 1 "$N"); do
    local log=/root/doll312_ab_${tag}_${n}.log
    timeout "$TMO" env PROSPER_GUEST_FS=1 PROSPER_NULL_PAGE=1 PROSPER_PROGRESS=20 \
      PROSPER_FREECANARY_GUARD=$guard "PROSPER_PAD_SCRIPT=${SCRIPT}" \
      ./build-linux/boot_trace /root/PPSA17942-app0 > "$log" 2>&1
    local rc=$?
    local f=$(grep -cE "MallocBinned3 Corruption|unrecognized block" "$log")
    local sk=$(grep -cE "FREECANARY" "$log")
    local sh=$(grep -c "20015f00" "$log")
    local prog=$(grep "\[progress\]" "$log" | tail -1 | grep -oE "t=[0-9.]+s|draws_cum=[0-9]+")
    if [ "$f" -gt 0 ]; then fatal=$((fatal+1)); else clean=$((clean+1)); fi
    echo "  [$tag n=$n] rc=$rc fatal_lines=$f freecanary_skips=$sk 20015f00=$sh | $(echo $prog | tr '\n' ' ')"
  done
  echo "=== ARM $tag (guard=$guard): FATAL=${fatal}/${N} CLEAN=${clean}/${N} ==="
}
echo "########## BASELINE (guard=0) ##########"
run_arm 0 base
echo "########## GUARD-ON (guard=1) ##########"
run_arm 1 guard
