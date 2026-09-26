#!/bin/bash
# Wait for THIS worktree's cmake build to finish, then run ctest and print the tail.
#
# The pattern is bracketed (`[b]uild-linux`) for two reasons, and both have bitten already:
# a bare `pgrep -f 'cmake --build'` matches this script's OWN command line, so the loop never
# exits; and it matches the OTHER agent's concurrent build of a different tree, so it would wait
# on work that is not ours. The bracket makes the pattern not match its own literal text, and
# naming the build directory scopes it to this worktree.
LOG=${1:-/var/tmp/claude-1000/build2.log}
DEADLINE=$(( $(date +%s) + 7200 ))
while pgrep -f 'cmake --build [b]uild-linux' >/dev/null; do
  if [ "$(date +%s)" -gt "$DEADLINE" ]; then echo "WAITER_TIMED_OUT"; exit 2; fi
  sleep 20
done
echo "=== build tail ==="
tail -3 "$LOG"
if grep -q 'error:' "$LOG"; then
  echo "BUILD_HAD_ERRORS"
  grep -m5 'error:' "$LOG"
  exit 1
fi
distrobox enter ps5ys -- /var/home/mattias800/repos/ps5ys/.claude/worktrees/agent-a854aa8ac88dedcc5/.lane/test.sh 2>&1 | tail -8
