#!/bin/bash
# Block for N seconds (default 120) while background work runs, then print progress.
# A foreground wait that costs one line of output instead of a poll loop.
N=${1:-120}
LOG=${2:-/var/tmp/claude-1000/build2.log}
command sleep "$N"
echo "targets built: $(grep -c 'Built target' "$LOG" 2>/dev/null)"
tail -1 "$LOG" 2>/dev/null
pgrep -f 'cmake --build [b]uild-linux' >/dev/null && echo STILL_BUILDING || echo NO_BUILD_RUNNING
