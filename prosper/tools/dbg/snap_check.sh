#!/bin/bash
cd "${PROSPER_REPO_ROOT:?set to your checkout root}/.claude/worktrees/agent-a48ff9e03b626cc45/prosper" || exit 1
pkill -9 boot_trace 2>/dev/null; sleep 1
python3 tools/snapshot/snapshot.py check
echo "SNAPSHOT_RC=$?"
