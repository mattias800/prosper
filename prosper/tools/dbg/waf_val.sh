#!/bin/bash
# #312 CLOSE validation — label free-state WAF guard. N synchronous menu-drive runs, undistorted
# timing (no MB3WATCH). Tallies ALL fatal signatures (Canary / unrecognized block / worker-fault /
# 0x20015f00) + WAF suppression activity + DOLL progression. Vars live in-file (cmdline-strip safe).
# Usage: waf_val.sh <first> <N> <timeout_s> [GUARD]   (GUARD=0 for the session-10 A/B baseline)
WT=/mnt/c/Users/matti/repos/ps5ys/.claude/worktrees/agent-a8f10a7598f626404/prosper
cd "$WT" || exit 1
FIRST=${1:-1}; N=${2:-4}; TMO=${3:-120}; GUARD=${4:-1}
SCRIPT="15:cross;20:start;25:cross;30:start;35:cross;40:cross;45:start;50:cross;60:cross;70:start;80:cross;90:cross;100:cross;110:cross;120:cross;130:cross"
pkill -9 boot_trace 2>/dev/null; sleep 1
fatal=0; clean=0; oom=0
for i in $(seq 0 $((N-1))); do
  n=$((FIRST+i))
  log=/root/doll312_waf_g${GUARD}_${n}.log
  timeout "$TMO" env PROSPER_GUEST_FS=1 PROSPER_NULL_PAGE=1 PROSPER_PROGRESS=20 \
    PROSPER_REL1_WAF_GUARD=$GUARD "PROSPER_PAD_SCRIPT=${SCRIPT}" \
    ./build-linux/boot_trace /root/PPSA17942-app0 > "$log" 2>&1
  rc=$?
  f=$(grep -cE "MallocBinned3 Corruption|unrecognized block|WORKER-THREAD FAULT|Canary was 0x" "$log")
  canary=$(grep -oE "Canary was 0x[0-9a-f]+" "$log" | head -1)
  sh=$(grep -c "20015f00" "$log")
  waf=$(grep -c "REL-WAF-SUPPRESS" "$log")
  prog=$(grep "\[progress\]" "$log" | tail -1 | grep -oE "t=[0-9.]+s|draws_cum=[0-9]+|flips=[0-9]+" | tr '\n' ' ')
  if [ "$f" -gt 0 ]; then fatal=$((fatal+1)); tag=FATAL;
  elif [ "$rc" = "137" ]; then oom=$((oom+1)); tag=OOM;
  else clean=$((clean+1)); tag=clean; fi
  echo "  [g=$GUARD n=$n] rc=$rc $tag fatals=$f canary=[$canary] 20015f00=$sh waf_suppress=$waf | $prog"
done
echo "=== WAF GUARD=$GUARD : FATAL=${fatal}/${N} CLEAN=${clean}/${N} OOM=${oom}/${N} ==="
