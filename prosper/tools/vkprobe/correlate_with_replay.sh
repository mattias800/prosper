#!/usr/bin/env bash
# correlate_with_replay.sh -- run the CONTROL and the SUBJECT alternately and see whether they go
# bad together.
#
# `vkprobe` reproduces #2945's class with no prosper code in the process; `gpu_replay <capsule>
# --draw 42` reproduces it through prosper's whole backend. Both are intermittent, and both drift
# with the same machine-wide rhythm -- which is exactly the situation in which two separate
# measurement sessions cannot be compared. Interleaving them inside one loop turns "they both fail
# sometimes" into a per-round paired observation: if the control is bad on the rounds the subject is
# bad and good on the rounds it is good, they go bad in the same WINDOWS -- which transfers the
# control's verdict to those windows and no further. It does not by itself establish a shared
# MECHANISM: a common environmental driver (GPU load from another process) produces exactly this
# association without any shared defect.
#
# It lives beside vkprobe rather than in tools/gpu_replay because the QUESTION is the control's.
# Note the boundary this folder's AGENTS.md draws is about linkage: this script starts gpu_replay as
# a child process and reads its stdout, which is not the same thing as a control that links prosper.
#
# Usage:  correlate_with_replay.sh <vkprobe> <gpu_replay> <capsule> <draw> <workdir> <rounds> <gap>
#
# One CSV line per round on stdout:
#   epoch,round,prosper_module_indexed_empty,hand_module_indexed_empty,replay_hash,replay_verdict
# where replay_verdict is good/bad/other against the two hashes #2945 records for draw 42.
set -u
probe=${1:?vkprobe}; replay=${2:?gpu_replay}; capsule=${3:?capsule}; draw=${4:?draw}
work=${5:?workdir}; rounds=${6:?rounds}; gap=${7:?gap}
cd "$work" || exit 2
GOOD=9068fcf09de07383    # the half-screen composite triangle rendered
BAD=a5e7b61cbf984383     # the scanout is untouched: the draw drew nothing

for round in $(seq 1 "$rounds"); do
    timeout 120 "$probe" --vs prosper_vs.spv --fs prosper_fs.spv \
                         --vs-b hand_vs.spv --fs-b hand_fs.spv --iterations 20 > cr_vk.log 2>&1
    a=$(awk '/^\[vkprobe\] a: indexed:/ {print $(NF-2)}' cr_vk.log)
    b=$(awk '/^\[vkprobe\] b: indexed:/ {print $(NF-2)}' cr_vk.log)
    timeout 180 "$replay" "$capsule" --draw "$draw" cr_out.bmp > cr_replay.log 2>&1
    h=$(grep -oE 'draw='"$draw"' bytes=[0-9]+ hash=[0-9a-f]+' cr_replay.log | grep -oE 'hash=[0-9a-f]+' | tail -1)
    h=${h#hash=}
    case "$h" in "$GOOD") v=good ;; "$BAD") v=bad ;; "") v=no-hash ;; *) v=other ;; esac
    printf '%s,%s,%s,%s,%s,%s\n' "$(date +%s)" "$round" "${a:-?}" "${b:-?}" "${h:-none}" "$v"
    sleep "$gap"
done
