---
name: start-task
description: Prepare an authorized prosper task using current instructions, issue ownership, an own worktree, and relevant verification.
---

# Start a task

Apply the steps relevant to the assigned task. This skill does not authorize publishing, pushing,
merging, or running workloads; preserve the task's existing authority and resource coordination.

1. **Read and verify the charter.** Read `CLAUDE.md`, `LOCAL.md` if present, and `CONTRIBUTING.md`; fetch in your own worktree, then run `python3 prosper/tools/session_start.py` from its checkout root. Inspect intentional differences; never reset another worktree to silence a warning.
2. **Check ownership for issue work.** Read the assigned tracker and current comments. Re-verify the defect on current `main`; check `in-progress`, live claims and matching remote branches before claiming new implementation work. A review or documentation task does not require claiming unrelated issues.
3. **Claim when appropriate and authorized.** Follow `CLAUDE.md`'s claim protocol; re-read comments for a race. Push an owned branch early only within the task's publishing scope.
4. **Own worktree, never the shared checkout:**
   ```bash
   git worktree add .claude/worktrees/<slug> -b <branch> origin/main
   ```
   Use the exact requested revision for a pinned review. Do not `git stash`; use `prosper/tools/wt_stash.py` or a temporary commit in your own worktree.
5. **Read before hypothesising.** The applicable nested `AGENTS.md`, the relevant status doc, and its `## Ruled out` section. Game work starts with `prosper/docs/GAME_COMPAT_ORCHESTRATION.md`.
6. **Plan to the scope.** The PR template headings can help; a small change needs only its problem, behavior, and relevant verification.
7. **Relevant baseline.** Reuse applicable author evidence. For behavior changes, run only needed focused checks in an owned build when execution is authorized and coordinated; use `ctest --test-dir <BUILD_DIR> --no-tests=error` and report counts and failures. Read-only and documentation tasks need no configure/build baseline. Follow `LOCAL.md` and the task's rules for `TMPDIR`, build and artifact locations.
8. **Regressions come with behavior changes.** Add or extend a meaningful test that fails without the fix; document nonbehavioral scope and verification limits (`prosper/tests/AGENTS.md`).
9. **Finish within authority.** Merge only when the user or task explicitly authorizes it and the charter's review and verification requirements pass. Remove your own worktree after completion and delete a remote branch only when authorized; preserve other owners' work.
