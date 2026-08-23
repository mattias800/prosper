#!/bin/sh
# Answer "is a peer lane mid-run on the shared GPU?" with an exit code that means what it says.
#
# WHY THIS EXISTS: the inline spelling this replaces — `pgrep -x prosper-app screenshot boot_trace`
# — is a usage error (`pgrep -x` takes exactly ONE pattern; instrument trap 222). It exits 2 having
# looked at nothing, and the two common wrappers around it both read as a clean box: stderr
# suppressed it prints nothing, and `... || echo "GPU free"` prints *GPU free* because the command
# failed. A guard whose failure looks like its success is worse than no guard, so briefs cite this
# tested command instead of an inline line nobody runs in isolation (#2948).
#
# Usage: gpu_busy.sh [-q] [extra-name ...]
#   -q           exit status only, no output
#   extra-name   further exact process names to count
#
# Exit codes follow pgrep conventions so loops read naturally
# (`while gpu_busy.sh -q; do sleep 30; done` waits for a free box — while spins on success,
# i.e. while the box is BUSY; `until` would spin while it is FREE):
#   0  consumers running (busy)
#   1  none running (free)
#   2  tool or usage error — NEVER answers "free" from a broken tool
#
# Counted by default: the five GPU consumers this repo runs, INCLUDING `screenshot_snap` —
# snapshot.py executes a copy of the frontend under that name, so a peer two full boots into a
# `verify` is invisible to a check that only lists built names (see GAME_COMPAT_ORCHESTRATION.md,
# "`pgrep -x screenshot` does not see a snapshot run"). Names are matched against comm, which the
# kernel truncates to 15 characters.

set -u

quiet=0
names="prosper-app boot_trace gpu_replay screenshot screenshot_snap"

case "${1-}" in
    -h|--help)
        printf 'Usage: gpu_busy.sh [-q] [extra-name ...]\n'
        printf 'Exit 0 = consumers running, 1 = free, 2 = tool/usage error.\n'
        exit 0 ;;
esac

while [ $# -gt 0 ]; do
    case "$1" in
        -q) quiet=1 ;;
        -*) printf 'gpu_busy.sh: unknown option: %s\n' "$1" >&2; exit 2 ;;
        *)  names="$names $1" ;;
    esac
    shift
done

command -v pgrep >/dev/null 2>&1 || { echo "gpu_busy.sh: pgrep not found" >&2; exit 2; }

total=0
for n in $names; do
    c=$(pgrep -c -x "$n" 2>/dev/null)
    rc=$?
    if [ "$rc" -eq 1 ]; then
        c=0                      # no match is an answer, not an error
    elif [ "$rc" -ne 0 ]; then
        echo "gpu_busy.sh: pgrep failed on '$n' (rc=$rc)" >&2
        exit 2
    fi
    total=$((total + c))
    if [ "$c" -gt 0 ] && [ "$quiet" -eq 0 ]; then
        printf '%s: %s\n' "$n" "$c"
    fi
done

if [ "$total" -gt 0 ]; then
    [ "$quiet" -eq 1 ] || printf 'GPU busy: %s consumer(s)\n' "$total"
    exit 0
fi
[ "$quiet" -eq 1 ] || echo "GPU free"
exit 1
