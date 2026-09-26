#!/bin/bash
set -e
cd /var/home/mattias800/repos/ps5ys/.claude/worktrees/agent-a854aa8ac88dedcc5/prosper
mkdir -p build-linux/tmpdir
export TMPDIR=$PWD/build-linux/tmpdir
cmake -S . -B build-linux -DCMAKE_BUILD_TYPE=RelWithDebInfo -DGAME_DUMP=/var/mnt/prosper/testdata/PPSA24651-app0
