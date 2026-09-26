#!/bin/bash
set -e
cd /var/home/mattias800/repos/ps5ys/.claude/worktrees/agent-a854aa8ac88dedcc5/prosper
export TMPDIR=$PWD/build-linux/tmpdir
cmake --build build-linux -j3
