#!/bin/bash
cd /var/home/mattias800/repos/ps5ys/.claude/worktrees/agent-a854aa8ac88dedcc5/prosper/build-linux
export TMPDIR=$PWD/tmpdir
ctest --no-tests=error -R "^${1}$" --output-on-failure
echo "CTEST_EXIT=$?"
