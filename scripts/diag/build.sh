#!/bin/bash
# Build prosper in this agent worktree (run inside WSL Ubuntu-24.04 as root).
set -e
WT="${PROSPER_REPO_ROOT:?set to your checkout root}/.claude/worktrees/agent-aa922b2bc59375888"
cd "$WT/prosper"
if [ ! -f build-linux/CMakeCache.txt ]; then
  cmake -S . -B build-linux -DCMAKE_BUILD_TYPE=Release -DGAME_DUMP="${PROSPER_REPO_ROOT:?set to your checkout root}/PPSA24651-app0"
fi
cmake --build build-linux -j16 2>&1 | tail -5
echo BUILD_OK
