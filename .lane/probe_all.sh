#!/bin/bash
# Probe candidate blocks ONE AT A TIME: cut into a parameterless probe function, compile, and let
# the compiler name every free variable. One at a time on purpose -- cutting two blocks together
# makes a variable that the first one declares look free in the second.
W=/var/home/mattias800/repos/ps5ys/.claude/worktrees/agent-a854aa8ac88dedcc5/prosper
PROBE=$HOME/lane-rrh-probe
PRISTINE=$HOME/lane-rrh-dryrun/prosper/tests/fixtures/render_runner.h
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR=$W/build-linux/tmpdir
mkdir -p "$TMPDIR"
cat > "$TMPDIR/syntax_check.cpp" <<'EOF'
#include "fixtures/render_runner.h"
int main() { return 0; }
EOF

probe() {
  cp "$PRISTINE" "$PROBE/prosper/tests/fixtures/render_runner.h"
  ( cd "$PROBE" && python3 .lane/probe_free.py "$1" "$2" "$3" --whole ) 2>&1 | tail -1
  /usr/bin/g++ -I"$PROBE/prosper/tests" -I"$W/tests" -I"$W/frontends" -I"$W/src" \
      -I"$W/third_party/libatrac9/src" -O0 -std=gnu++20 -fsyntax-only \
      "$TMPDIR/syntax_check.cpp" 2>&1 \
    | grep -oE "‘[a-zA-Z_][a-zA-Z0-9_]*’ was not declared" | sort -u | tr '\n' ' '
  echo
  echo "---"
}

probe '    if (ds_active) {' 0 0
probe '    if (seed_rgba) {' 0 0
probe '    if (use_color1 && effective_seed1) {' 0 0
