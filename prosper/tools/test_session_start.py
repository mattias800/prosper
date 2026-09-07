#!/usr/bin/env python3
"""Exercise startup context against real temporary Git repositories and the shipped hook."""
import json
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "prosper/tools/session_start.py"
HOOK = json.loads((ROOT / ".claude/settings.json").read_text())["hooks"]["SessionStart"][0]["hooks"][0]
SPEC = importlib.util.spec_from_file_location("session_start", TOOL)
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


class StartupTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="session start ")
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name) / "checkout with spaces"
        self.remote = Path(self.tmp.name) / "origin.git"
        self.repo.mkdir()
        self.git("init", "-q", "-b", "main")
        self.git("config", "user.name", "Test")
        self.git("config", "user.email", "test@example.invalid")
        for name in ("CLAUDE.md", "AGENTS.md", "CONTRIBUTING.md"):
            (self.repo / name).write_text("Current instruction\n")
        dest = self.repo / "prosper/tools/session_start.py"
        dest.parent.mkdir(parents=True)
        shutil.copyfile(TOOL, dest)
        self.commit("initial")
        self.git("clone", "--bare", "-q", str(self.repo), str(self.remote))
        self.git("remote", "add", "origin", str(self.remote))
        self.git("fetch", "-q", "origin")

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.repo), *args],
                                       stderr=subprocess.PIPE, text=True).strip()

    def commit(self, message):
        self.git("add", ".")
        self.git("commit", "-qm", message)

    def check(self, expected, *args, repo=None):
        proc = subprocess.run([sys.executable, str(TOOL), "--repo", str(repo or self.repo), *args],
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, expected, proc.stdout + proc.stderr)
        if expected:
            self.assertNotIn("OK:", proc.stdout)
        return proc.stdout

    def hook(self, repo=None):
        # Execute the actual exec-form hook argv, with the documented project-dir substitution.
        # No shell is involved: project paths containing spaces stay one argument.
        project = str(repo or self.repo)
        argv = [HOOK["command"]] + [x.replace("${CLAUDE_PROJECT_DIR}", project) for x in HOOK["args"]]
        proc = subprocess.run(argv, input="{}", capture_output=True, text=True,
                              timeout=HOOK["timeout"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        output = json.loads(proc.stdout)["hookSpecificOutput"]
        self.assertEqual(output["hookEventName"], "SessionStart")
        return output["additionalContext"]

    def advance_remote(self):
        old = self.git("rev-parse", "HEAD")
        (self.repo / "CLAUDE.md").write_text("New instruction\n")
        self.commit("new charter")
        self.git("push", "-q", "origin", "main")
        self.git("reset", "--hard", old)
        # A push updates tracking refs. Reset that ref to reproduce a session that never fetched.
        self.git("update-ref", "refs/remotes/origin/main", old)

    def test_current_and_hook_positive_control(self):
        self.assertIn("OK:", self.check(0))
        self.assertIn("OK:", self.hook())

    def test_dirty_and_staged_instructions_reach_hook_context(self):
        (self.repo / "CLAUDE.md").write_text("Local instruction\n")
        self.assertIn("CLAUDE.md: reference=differs; local edits=yes", self.check(1))
        self.assertIn("WARNING: CLAUDE.md", self.hook())
        self.git("add", "CLAUDE.md")
        self.assertIn("local edits=yes", self.check(1))

    def test_missing_working_charter(self):
        (self.repo / "CLAUDE.md").unlink()
        self.assertIn("CLAUDE.md: reference=differs", self.check(1))

    def test_stale_tracking_ref_is_not_current(self):
        self.advance_remote()
        self.assertIn("tracking ref is stale", self.check(1))
        self.assertIn("tracking ref is stale", self.hook())

    def test_fetched_new_charter_and_divergence(self):
        self.advance_remote()
        self.git("fetch", "-q", "origin")
        self.assertIn("ahead=0 behind=1", self.check(1))
        (self.repo / "feature.txt").write_text("Feature\n")
        self.commit("feature")
        self.assertIn("ahead=1 behind=1", self.check(1))

    def test_feature_branch_and_detached_head(self):
        self.git("switch", "-qc", "feature")
        (self.repo / "feature.txt").write_text("Feature\n")
        self.commit("feature")
        self.assertIn("branch=feature", self.check(0))
        self.git("checkout", "-q", "--detach", "origin/main")
        self.assertIn("branch=(detached)", self.check(0))

    def test_linked_worktree_checks_its_own_instructions(self):
        linked = Path(self.tmp.name) / "linked worktree"
        self.git("worktree", "add", "-q", "--detach", str(linked), "HEAD")
        self.check(0, repo=linked)
        (linked / "CLAUDE.md").write_text("Changed in linked tree\n")
        self.assertIn("WARNING: CLAUDE.md", self.hook(repo=linked))
        self.check(0)

    def test_missing_reference_and_offline_are_unverified(self):
        self.assertIn("UNVERIFIED:", self.check(2, "--offline"))
        self.git("update-ref", "-d", "refs/remotes/origin/main")
        self.assertIn("UNVERIFIED:", self.check(2))
        self.assertIn("UNVERIFIED:", self.hook())

    def test_unreachable_remote_is_unverified(self):
        self.git("remote", "set-url", "origin", str(Path(self.tmp.name) / "absent.git"))
        self.assertIn("UNVERIFIED:", self.check(2))
        self.assertIn("UNVERIFIED:", self.hook())

    def test_line_endings_are_not_instruction_changes(self):
        self.git("config", "core.autocrlf", "true")
        for name in ("CLAUDE.md", "AGENTS.md", "CONTRIBUTING.md"):
            (self.repo / name).write_bytes(b"Current instruction\r\n")
        self.check(0)

    def test_git_timeout_and_unavailable_are_unverified(self):
        for error in (subprocess.TimeoutExpired("git", 5), FileNotFoundError("git")):
            with self.subTest(error=type(error).__name__), patch.object(
                    PROBE.subprocess, "run", side_effect=error):
                status, report = PROBE.inspect(self.repo)
                self.assertEqual(status, 2)
                self.assertIn("UNVERIFIED:", report)
                self.assertNotIn("OK:", report)

    def test_whole_check_deadline_leaves_time_for_hook_output(self):
        with patch.object(PROBE.time, "monotonic", side_effect=[0, 13]), patch.object(
                PROBE.subprocess, "run") as run:
            status, report = PROBE.inspect(self.repo)
            self.assertEqual(status, 2)
            self.assertIn("deadline exceeded", report)
            run.assert_not_called()


    def test_git_path_output_is_never_used_as_a_filesystem_path(self):
        # An MSYS2/Cygwin git answers path queries in POSIX form (/c/Users/...), which native
        # Windows Python reads back as the drive-relative \c\Users\... and which that same git
        # then rejects. A check that round-trips its own root through git's path flavour loses
        # the repository on exactly the pairing the Windows CI job runs (#3443).
        #
        # The poisoned answer is built BY HAND rather than taken from the git on this host,
        # whose flavour happens to survive the trip: a control drawn from the same source as
        # the null it validates cannot express the case being tested for.
        real = PROBE.subprocess.run
        consulted = []

        def poisoned(argv, **kwargs):
            if "--show-toplevel" in argv:
                consulted.append(argv)
                return subprocess.CompletedProcess(argv, 0, b"/c/nonexistent/checkout\n", b"")
            return real(argv, **kwargs)

        with patch.object(PROBE.subprocess, "run", side_effect=poisoned):
            status, report = PROBE.inspect(self.repo)
        # The damage first, then the contract that prevents it: a check that consults the
        # poisoned answer reports UNVERIFIED on a checkout that is perfectly current.
        self.assertEqual(status, 0, report)
        self.assertIn("OK:", report)
        self.assertEqual(consulted, [], "the worktree root must not come from git path output")

    def test_root_is_found_from_a_subdirectory(self):
        # The relative derivation that replaced --show-toplevel still has to climb: this is the
        # only arm where the cdup is non-empty, so it is what proves the join is right.
        self.assertIn("OK:", self.check(0, repo=self.repo / "prosper/tools"))


if __name__ == "__main__":
    unittest.main()
