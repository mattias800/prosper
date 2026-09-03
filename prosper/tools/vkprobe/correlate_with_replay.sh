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
#                                  [untouched-scanout-hash]
#
# One CSV line per round on stdout:
#   epoch,round,prosper_module_indexed_empty,hand_module_indexed_empty,replay_hash,replay_verdict
# where replay_verdict is bad/rendered/no-hash.
#
# THE VERDICT IS KEYED ON THE FAILURE, NOT ON THE SUCCESS, and that is not a stylistic choice.
# "The draw drew nothing" hashes to the capsule's own RTT seed for the scanout it writes, so that
# value is a property of the CAPTURE and survives every renderer change. The successful hash is a
# property of the RENDERER and moves whenever the picture legitimately improves: the value this
# script shipped with, 9068fcf09de07383, was BALAN's draw 42 on 2026-08-23 and by 2026-09-02 the
# same draw on the same capsule rendered ccc433ff6d980383. Keyed the other way round, every round
# of a perfectly healthy campaign reads as `other` and the correlation silently becomes a
# correlation with nothing.
#
# The key is DERIVED FROM THE CAPSULE at startup rather than hardcoded, because this script is
# itself the proof that a hardcoded hash rots. "The draw drew nothing" leaves the scanout exactly as
# the capsule seeded it, so the failure hash is the seed's hash -- and the capsule carries both the
# draw's target address and that seed's hash:
#   draw[42] ... target=0000009fc0000000 ...
#   rtt-seed addr=0000009fc0000000 ... hash=a5e7b61cbf984383
# Pass an explicit value as $8 to override the derivation; a derivation that fails says so loudly
# and falls back to the recorded BALAN s3537 value rather than to silence.
set -u
probe=${1:?vkprobe}; replay=${2:?gpu_replay}; capsule=${3:?capsule}; draw=${4:?draw}
work=${5:?workdir}; rounds=${6:?rounds}; gap=${7:?gap}
BAD=${8:-}
cd "$work" || exit 2

if [ -z "$BAD" ]; then
    inspect=$("$replay" "$capsule" --inspect-only /dev/null 2>/dev/null)
    target=$(printf '%s\n' "$inspect" | grep -oE "^draw\[$draw\] .* target=[0-9a-f]+" \
             | grep -oE 'target=[0-9a-f]+' | head -1)
    target=${target#target=}
    if [ -n "$target" ]; then
        BAD=$(printf '%s\n' "$inspect" | grep -oE "^rtt-seed addr=$target .* hash=[0-9a-f]+" \
              | grep -oE 'hash=[0-9a-f]+' | head -1)
        BAD=${BAD#hash=}
    fi
    if [ -n "$BAD" ]; then
        echo "correlate: draw $draw writes target $target, whose capsule seed hashes to $BAD;" \
             "that is the 'drew nothing' value for this capsule" >&2
    else
        BAD=a5e7b61cbf984383
        echo "correlate: could NOT derive the failure hash from this capsule (no draw[$draw] target" \
             "or no matching rtt-seed line); falling back to BALAN s3537's $BAD, which is almost" \
             "certainly wrong for any other capture -- pass the right one as argument 8" >&2
    fi
fi

for round in $(seq 1 "$rounds"); do
    timeout 120 "$probe" --vs prosper_vs.spv --fs prosper_fs.spv \
                         --vs-b hand_vs.spv --fs-b hand_fs.spv --iterations 20 > cr_vk.log 2>&1
    a=$(awk '/^\[vkprobe\] a: indexed:/ {print $(NF-2)}' cr_vk.log)
    b=$(awk '/^\[vkprobe\] b: indexed:/ {print $(NF-2)}' cr_vk.log)
    timeout 180 "$replay" "$capsule" --draw "$draw" cr_out.bmp > cr_replay.log 2>&1
    h=$(grep -oE 'draw='"$draw"' bytes=[0-9]+ hash=[0-9a-f]+' cr_replay.log | grep -oE 'hash=[0-9a-f]+' | tail -1)
    h=${h#hash=}
    case "$h" in "$BAD") v=bad ;; "") v=no-hash ;; *) v=rendered ;; esac
    printf '%s,%s,%s,%s,%s,%s\n' "$(date +%s)" "$round" "${a:-?}" "${b:-?}" "${h:-none}" "$v"
    sleep "$gap"
done
