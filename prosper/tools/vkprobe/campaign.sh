#!/usr/bin/env bash
# campaign.sh -- run vkprobe repeatedly over a long wall-clock window and summarise by RUN.
#
# WHY THIS EXISTS. #2945's failure rate drifts machine-wide over minutes: the identical command
# measured 0/20 in one ten-minute window and 12/12 in the next with nothing changed. A single
# vkprobe invocation therefore samples the WINDOW as much as the subject, and an A/B measured as
# "arm A for a while, then arm B" is void rather than negative. Two rules follow, and this script
# exists to make both of them the default:
#
#   1. Pair the arms INSIDE one process (`--vs-b/--fs-b`), so both modules meet the same window.
#   2. Spread the runs over a long window, because the failure is per-PROCESS in shape -- more
#      iterations inside one process is not more samples of the thing that varies.
#
# Usage:  campaign.sh <vkprobe> <workdir> <rounds> <sleep-seconds> [-- extra vkprobe args...]
#
# Every round runs each configuration below once. Output is one CSV line per run on stdout:
#   epoch,round,config,arm,plain_empty,indexed_empty,iterations
# so `awk -F, '{...}'` gives run-level rates directly. Read the RUN counts, not the iteration
# counts: quote "k failing runs of n", which is the unit the README's figures use.
set -u
probe=${1:?vkprobe binary}; work=${2:?work directory}; rounds=${3:?rounds}; gap=${4:?sleep seconds}
shift 4; [ "${1:-}" = "--" ] && shift
cd "$work" || exit 2

emit() {   # <round> <config> <log-file>
    local round=$1 config=$2 log=$3 epoch; epoch=$(date +%s)
    # Arms are labelled "a: "/"b: " in a paired run and unprefixed otherwise.
    awk -v e="$epoch" -v r="$round" -v c="$config" '
        /non-indexed: covered/ { arm = ($2 ~ /^[ab]:$/) ? $2 : "-:"; sub(/:$/, "", arm)
                                 plain[arm] = $(NF-2); iters[arm] = $NF }
        /^\[vkprobe\] (a: |b: )?indexed:/ { arm = ($2 ~ /^[ab]:$/) ? $2 : "-:"; sub(/:$/, "", arm)
                                 idx[arm] = $(NF-2); iters[arm] = $NF }
        END { for (a in iters) printf "%s,%s,%s,%s,%s,%s,%s\n", e, r, c, a, plain[a], idx[a], iters[a] }
    ' "$log"
}

for round in $(seq 1 "$rounds"); do
    # Configuration 1 -- THE VERDICT: prosper's own generated module against a hand-written module
    # doing the same storage-buffer loads, in one process.
    timeout 120 "$probe" --vs prosper_vs.spv --fs prosper_fs.spv \
                         --vs-b hand_vs.spv  --fs-b hand_fs.spv \
                         --iterations 20 "$@" > c1.log 2>&1
    emit "$round" prosper-vs-hand c1.log
    # Configuration 2 -- THE CHARACTERISATION: a module whose position never touches memory,
    # against one that reports the vertex indices it was handed.
    timeout 120 "$probe" --vs nossbo_vs.spv --fs hand_fs.spv \
                         --vs-b idxrb_vs.spv --fs-b hand_fs.spv \
                         --readback-dwords 16:3 --iterations 20 "$@" > c2.log 2>&1
    emit "$round" nossbo-vs-readback c2.log
    grep -hE 'readback \[' c2.log >> readback-patterns.log
    sleep "$gap"
done
