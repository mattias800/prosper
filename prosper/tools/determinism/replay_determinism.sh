#!/usr/bin/env bash
# replay_determinism.sh -- run a renderer-determinism campaign and write the CSV
# `replay_determinism_report.py` reads. #2945.
#
# THE QUESTION. Replay one frozen capture many times with one binary and count how many distinct
# output hashes come back. One means the renderer was deterministic across those runs; more than one
# is #2945.
#
# WHY IT IS A SCRIPT AND NOT A `for` LOOP. Three rules, each learned by a campaign that got the
# answer wrong without them (docs/GRAPHICS.md, "Renderer determinism"):
#
#   1. A CONTROL RUNS BESIDE THE SUBJECT, IN THE SAME ROUND. The rate drifts machine-wide over
#      minutes -- the identical command measured 0 of 20 in one window and 12 of 12 in the next --
#      so a clean campaign and a quiet window are the same numbers. `tools/vkprobe` reproduces the
#      class with no prosper code in the process; it is here so the report can answer UNDECIDED
#      instead of "fixed". Read its README before quoting it: a clean control run proves nothing on
#      its own, which is exactly why it is used only to detect the window, never to clear anything.
#   2. THE ARMS ALTERNATE ORDER. Anything measured as "arm A for a while, then arm B" is void here.
#   3. LOAD IS A CONDITION, AND SO ARE PEERS. Concurrent GPU work from another process induces the
#      failing regime on demand (sufficient, not necessary). `--load N` runs N extra full replays
#      during every second block; the peer count is recorded on every row either way, because
#      another lane's `gpu_replay`, `vkprobe` or `ctest` is exactly the load that matters and
#      `pgrep -x prosper-app` cannot see any of them.
#
# It also does NOT use `gpu_replay --warmup-repeats`: that renders each repeat on top of the
# previous one's persistent targets, so its hashes drift by construction (GRAPHICS.md). Every
# sample here is a separate process.
#
# Usage:
#   replay_determinism.sh --replay <gpu_replay> --capture <file.prgcap|.prgbundle> --out <csv>
#                         [--rounds N] [--gap S] [--block N] [--load N] [--label NAME]
#                         [--arm 'LABEL=extra gpu_replay args'] ...
#                         [--control <vkprobe> --shaders <dir>] [--control-iterations N]
#                         [--work <dir>] [--peer-wait S]
#
# `--control-iterations` trades sensitivity against cost, and the trade is not the obvious one: the
# control's failure is per-PROCESS in shape (a run is bad or it is not; inside a bad one most
# iterations fail), so more ROUNDS beats more iterations per round. Raise it only to make each
# control process live longer and so overlap more of a short bad window.
#
# With no --arm, one arm named `full` replays the whole capture. Repeat --arm for more, e.g.
#   --arm 'full=' --arm 'draw42=--draw 42'
#
# CSV (tidy, one row per observation):  epoch,round,cond,peers,gpu_pct,role,arm,value,rc,ms
#
# `gpu_pct` is amdgpu's `gpu_busy_percent` sampled at the moment the row is written, or -1
# where the kernel does not expose it. It is here so the LOAD condition measures its own
# premise: a block labelled `selfload` whose GPU utilisation never rose has not tested the
# load condition, and no other column in this file would say so.
set -u

replay=; capture=; out=; rounds=60; gap=2; block=10; load=0; label=local
control=; shaders=; work=; peer_wait=300; control_iterations=20
arms=()
while [ $# -gt 0 ]; do
    case "$1" in
        --replay)    replay=$2; shift 2 ;;
        --capture)   capture=$2; shift 2 ;;
        --out)       out=$2; shift 2 ;;
        --rounds)    rounds=$2; shift 2 ;;
        --gap)       gap=$2; shift 2 ;;
        --block)     block=$2; shift 2 ;;
        --load)      load=$2; shift 2 ;;
        --label)     label=$2; shift 2 ;;
        --arm)       arms+=("$2"); shift 2 ;;
        --control)   control=$2; shift 2 ;;
        --shaders)   shaders=$2; shift 2 ;;
        --work)      work=$2; shift 2 ;;
        --peer-wait) peer_wait=$2; shift 2 ;;
        --control-iterations) control_iterations=$2; shift 2 ;;
        -h|--help)   sed -n '2,45p' "$0"; exit 0 ;;
        *) echo "replay_determinism.sh: unknown argument: $1" >&2; exit 2 ;;
    esac
done
[ -n "$replay" ] && [ -n "$capture" ] && [ -n "$out" ] || {
    echo "replay_determinism.sh: --replay, --capture and --out are required" >&2; exit 2; }
[ -x "$replay" ] || { echo "replay_determinism.sh: not executable: $replay" >&2; exit 2; }
[ -s "$capture" ] || { echo "replay_determinism.sh: missing or empty capture: $capture" >&2; exit 2; }
# A capsule is a positional argument to gpu_replay; a bundle needs --bundle, and passing one
# positionally is rejected in a way that reads like a bad file rather than a bad invocation.
case "$capture" in
    *.prgbundle) capture_flag=--bundle ;;
    *)           capture_flag= ;;
esac
[ ${#arms[@]} -gt 0 ] || arms=("full=")
work=${work:-$(mktemp -d)}
mkdir -p "$work" || exit 2

# The control's shaders are SPIR-V assembly in the tree; assemble them once. A missing spirv-as is
# not a reason to run without a control -- it is a reason for every control row to say it could not
# run, which the report reads as UNDECIDED rather than as a clean window.
# An already-assembled module in the work directory is used as-is, so a host without spirv-tools
# can still run the control against modules assembled once inside the container.
control_ready=0
if [ -n "$control" ] && [ -x "$control" ]; then
    control_ready=1
    for s in no_ssbo_vs index_readback_vs minimal_green_fs; do
        if [ ! -f "$work/$s.spv" ]; then
            if [ -n "$shaders" ] && command -v spirv-as >/dev/null 2>&1; then
                spirv-as --target-env vulkan1.1 "$shaders/$s.spvasm" -o "$work/$s.spv" 2>/dev/null \
                    || control_ready=0
            else
                control_ready=0
            fi
        fi
    done
fi
[ -n "$control" ] && [ "$control_ready" != 1 ] && \
    echo "replay_determinism.sh: control unavailable; every control row will say so and the" \
         "report will read the campaign as UNDECIDED rather than clean" >&2

gpu_pct() {
    local f
    for f in /sys/class/drm/card*/device/gpu_busy_percent; do
        [ -r "$f" ] && { cat "$f" 2>/dev/null || echo -1; return; }
    done
    echo -1
}

# Peer GPU consumers, deliberately EXCLUDING `gpu_replay` and `vkprobe` -- this campaign's own
# tools, which `tools/gpu_busy.sh` counts and which would therefore make the campaign see itself as
# a peer and wait forever. That exclusion is the whole reason this is not a call to gpu_busy.sh, and
# it has a consequence the reader must be told: **another lane's gpu_replay, vkprobe or ctest is
# invisible here**, and so is a sibling campaign of this very script. The `gpu_pct` column is the
# ground truth about load; this count is only about the frontends whose captures must not be
# contaminated.
#
# A pgrep that ERRORS is not an answer, and reporting it as "no peers" is the trap-222 shape
# gpu_busy.sh exists to prevent -- a guard whose failure looks like its success. Fail closed: an
# unusable count reports a peer, so the campaign waits rather than trampling one.
peers() {
    local t=0 c rc
    for n in prosper-app boot_trace screenshot screenshot_snap; do
        c=$(pgrep -c -x "$n" 2>/dev/null); rc=$?
        if [ "$rc" -eq 1 ]; then
            c=0                                  # no match is an answer
        elif [ "$rc" -ne 0 ]; then
            echo "replay_determinism.sh: pgrep failed on '$n' (rc=$rc); assuming a peer is up" >&2
            echo 1
            return 0
        fi
        t=$((t + c))
    done
    echo "$t"
}

STOP=$work/.load_stop
PIDS=$work/.load_pids
start_load() {
    [ "$load" -gt 0 ] || return 0
    [ -e "$STOP" ] && return 0
    : > "$STOP"; : > "$PIDS"
    for i in $(seq 1 "$load"); do
        ( while [ -e "$STOP" ]; do
              # shellcheck disable=SC2086 -- $capture_flag is empty for a capsule by design
              timeout 600 "$replay" $capture_flag "$capture" "$work/load_$i.bmp" >/dev/null 2>&1
          done ) &
        echo $! >> "$PIDS"
    done
}
stop_load() {
    [ -e "$STOP" ] || return 0
    rm -f "$STOP"
    [ -f "$PIDS" ] && { while read -r p; do wait "$p" 2>/dev/null; done < "$PIDS"; rm -f "$PIDS"; }
}
# A TERM handler that does not exit makes the script IGNORE the signal, which has already produced
# three copies of a campaign running at once against one CSV. Exit explicitly.
trap 'stop_load' EXIT
trap 'stop_load; exit 143' TERM
trap 'stop_load; exit 130' INT

emit() { printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' "$(date +%s)" "$1" "$2" "$3" "$(gpu_pct)" \
             "$4" "$5" "$6" "$7" "$8" >> "$out"; }

run_subject() {   # <round> <cond> <peers> <arm-spec>
    local round=$1 cond=$2 p=$3 spec=$4
    local name=${spec%%=*} extra=${spec#*=}
    local log=$work/replay_${round}_$name.log img=$work/out_${round}_$name.bmp
    local t0 t1 rc h
    t0=$(date +%s%N)
    # shellcheck disable=SC2086 -- $extra and $capture_flag are deliberate argument lists
    timeout 900 "$replay" $capture_flag "$capture" $extra "$img" > "$log" 2>&1; rc=$?
    t1=$(date +%s%N)
    # A bundle prints one `bundle-submit=<id> ... hash=<h>` line per submit and NO final `output=`
    # line, so a bundle replay recorded by the output line alone has no value at all -- and #2945's
    # own headline measurement is per-submit ("submit 3537 gave 5 distinct hashes, submit 3540 gave
    # 1"). Join the per-submit hashes in submit order and record that: a change in ANY submit is a
    # distinct value, and the value names which submit moved.
    h=$(grep -E '^\[gpureplay\] bundle-submit=[0-9]+ .* hash=[0-9a-f]+$' "$log" \
        | sed -E 's/^\[gpureplay\] bundle-submit=([0-9]+) .* hash=([0-9a-f]+)$/\1:\2/' \
        | paste -sd'|' -)
    if [ -z "$h" ]; then
        h=$(grep -oE '^\[gpureplay\] output=[0-9x]+ target=[0-9a-f]+ draw=[0-9]+ bytes=[0-9]+ hash=[0-9a-f]+' \
            "$log" | grep -oE 'hash=[0-9a-f]+' | tail -1)
        h=${h#hash=}
    fi
    emit "$round" "$cond" "$p" subject "$label/$name" "${h:-none}" "$rc" "$(( (t1 - t0) / 1000000 ))"
    rm -f "$img"
}

run_control() {   # <round> <cond> <peers>
    local round=$1 cond=$2 p=$3
    [ -n "$control" ] || return 0
    if [ "$control_ready" != 1 ]; then
        emit "$round" "$cond" "$p" control "$label/vkprobe" "unavailable" 2 0
        return 0
    fi
    local log=$work/control_$round.log t0 t1 rc line ok total
    t0=$(date +%s%N)
    # --indices 3,4,5, NEVER the identity sequence: with 0,1,2 a readback of [1,2,3] cannot
    # distinguish a correct index fetch from the shader having been handed the ordinals, which made
    # the first campaign on this issue score 91.6% where the diagnostic indices score 1.2%.
    timeout 300 "$control" --vs "$work/no_ssbo_vs.spv" --fs "$work/minimal_green_fs.spv" \
        --vs-b "$work/index_readback_vs.spv" --fs-b "$work/minimal_green_fs.spv" \
        --indices 3,4,5 --readback-dwords 16:8 --iterations "$control_iterations" > "$log" 2>&1
    rc=$?
    t1=$(date +%s%N)
    # ALL of them, not `head -1`. vkprobe prints one readback line PER PATTERN (vkprobe.cpp, the
    # `for (const auto& [pattern, count] : tally.indexed_words)` loop), from a map, so the lines
    # arrive in lexicographic pattern order. Reading the first line alone makes `total` the count
    # for one pattern instead of the iteration count, and -- because the correct pattern
    # `[--,--,--,4,5,6,--,--]` sorts BEFORE a wrong one such as `[1,2,3,--,...]` -- a round whose
    # only failures sort after it scores `pass`. That error is conservative (a missed fire pushes
    # the verdict toward UNDECIDED, never toward DETERMINISTIC), which is exactly why it survived a
    # whole campaign unnoticed: it can only ever weaken a result, so nothing ever looks wrong.
    line=$(grep -E '^\[vkprobe\] b: indexed +vertex-index readback' "$log")
    # `rc` in the CSV means "could this control run at all", NOT vkprobe's own exit status.
    # vkprobe exits 1 whenever any iteration covered no pixels, and with the diagnostic indices
    # 3,4,5 the indexed arm reads outside the record buffer and covers nothing BY DESIGN -- so its
    # exit 1 is the normal outcome here. Reporting that as a broken instrument would make every
    # campaign UNDECIDED for a reason that is not true. Only exit 2 (vkprobe's setup-error code)
    # and a missing readback line mean the control did not run.
    if [ -z "$line" ] || [ "$rc" -eq 2 ]; then
        emit "$round" "$cond" "$p" control "$label/vkprobe" "did-not-run:rc=$rc" 2 \
             "$(( (t1 - t0) / 1000000 ))"
        return 0
    fi
    ok=$(printf '%s\n' "$line" | grep -oE '\[--,--,--,4,5,6,--,--\] x[0-9]+' | grep -oE '[0-9]+$' \
         | awk '{ n += $1 } END { if (NR > 0) print n }')
    # Sum the per-pattern counts with awk rather than `paste | bc`: bc is not always installed, and
    # an empty total would compare unequal to ok and report a control FAILURE on every round --
    # turning "the control never fired" into "the control always fired", which is exactly the
    # inversion this campaign exists to prevent. A total that does not parse is did-not-run.
    total=$(printf '%s\n' "$line" | grep -oE 'x[0-9]+' | sed 's/x//' \
            | awk '{ n += $1 } END { if (NR > 0) print n }')
    ok=${ok:-0}
    if [ -z "$total" ] || [ "$total" -le 0 ] 2>/dev/null; then
        emit "$round" "$cond" "$p" control "$label/vkprobe" "did-not-run:unparsed-readback" 2 \
             "$(( (t1 - t0) / 1000000 ))"
        return 0
    fi
    if [ "$ok" = "$total" ]; then
        emit "$round" "$cond" "$p" control "$label/vkprobe" pass 0 "$(( (t1 - t0) / 1000000 ))"
    else
        emit "$round" "$cond" "$p" control "$label/vkprobe" "fail:$ok/$total" 0 \
             "$(( (t1 - t0) / 1000000 ))"
    fi
}

[ -s "$out" ] || printf 'epoch,round,cond,peers,gpu_pct,role,arm,value,rc,ms\n' > "$out"

for round in $(seq 1 "$rounds"); do
    # `no-selfload` means exactly what it says: THIS campaign started no loader of its own for this
    # block. It does NOT mean the GPU was idle -- a sibling campaign, another lane's ctest, or any
    # gpu_replay on the box is invisible to `peers()` above. Read `gpu_pct`, never the label, when
    # you want to know what the load actually was.
    blk=$(( (round - 1) / block ))
    cond=no-selfload
    if [ "$load" -gt 0 ] && [ $(( blk % 2 )) -eq 1 ]; then cond=selfload; fi

    # Wait a peer out, bounded, so another lane's capture is not contaminated -- then proceed and
    # say so in the row rather than stalling the campaign forever.
    w=0
    while [ "$(peers)" -gt 0 ] && [ "$w" -lt "$peer_wait" ]; do
        stop_load; sleep 15; w=$((w + 15))
    done
    p=$(peers)
    if [ "$cond" = selfload ]; then
        if [ "$p" -gt 0 ]; then cond=peerload; else start_load; fi
    fi
    [ "$p" -gt 0 ] && [ "$cond" = no-selfload ] && cond=no-selfload-peer

    if [ $(( round % 2 )) -eq 1 ]; then
        run_control "$round" "$cond" "$p"
        for spec in "${arms[@]}"; do run_subject "$round" "$cond" "$p" "$spec"; done
    else
        for spec in "${arms[@]}"; do run_subject "$round" "$cond" "$p" "$spec"; done
        run_control "$round" "$cond" "$p"
    fi
    sleep "$gap"
done
stop_load
