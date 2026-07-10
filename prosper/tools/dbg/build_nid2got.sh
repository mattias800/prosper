#!/bin/bash
WT=/mnt/c/Users/matti/repos/ps5ys/.claude/worktrees/agent-ac66b5fec5918f7aa/prosper
cd "$WT" || exit 1
g++ -O1 -std=c++20 -I src tools/dbg/nid2got.cpp src/self/*.cpp -o /root/nid2got 2>&1 | head -20
ls -la /root/nid2got
