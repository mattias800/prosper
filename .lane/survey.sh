#!/bin/bash
cd /var/home/mattias800/repos/ps5ys/.claude/worktrees/agent-a854aa8ac88dedcc5/prosper
export TMPDIR=$PWD/build-linux/tmpdir
export PYTHONDONTWRITEBYTECODE=1
python3 tools/refactor/survey_sizes.py --roots tests --min-lines 2500 "$@"
