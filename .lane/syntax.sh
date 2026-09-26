#!/bin/bash
# Fast syntax gate: compile a TU that includes render_runner.h from $1 (a tests root override).
# Not a substitute for the full build -- it proves the header parses and its templates that get
# instantiated here type-check, nothing about link or behaviour.
W=/var/home/mattias800/repos/ps5ys/.claude/worktrees/agent-a854aa8ac88dedcc5/prosper
OVERRIDE=${1:-$W/tests}
OUT=$W/build-linux/tmpdir
mkdir -p "$OUT"
cat > "$OUT/syntax_check.cpp" <<'EOF'
#include "fixtures/render_runner.h"
int main() { return 0; }
EOF
export TMPDIR=$OUT
/usr/bin/g++ -I"$OVERRIDE" -I"$W/tests" -I"$W/frontends" -I"$W/src" \
    -I"$W/third_party/libatrac9/src" -O0 -std=gnu++20 -fsyntax-only "$OUT/syntax_check.cpp"
echo "SYNTAX_EXIT=$?"
