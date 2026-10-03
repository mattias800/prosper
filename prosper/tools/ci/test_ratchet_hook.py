"""Tests for ratchet_hook: command parsing, and every verdict against a throwaway git checkout."""

import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ratchet_hook as hook  # noqa: E402

REPO_ROOT = HERE.parents[2]
NEEDS_GIT = unittest.skipUnless(shutil.which("git"), "git is not on PATH")


class Parsing(unittest.TestCase):
    """Which commands are gated, and where they run."""

    def test_commit_and_push_are_gated(self):
        self.assertEqual([("commit", None)], hook.gated_git_invocations("git commit -m x"))
        self.assertEqual([("push", None)], hook.gated_git_invocations("git push -u origin b"))

    def test_other_git_commands_are_not(self):
        for command in ("git status", "git log --oneline", "git commit-tree HEAD^{tree}", "ls"):
            self.assertEqual([], hook.gated_git_invocations(command), command)

    def test_a_commit_message_is_not_a_subcommand(self):
        # `git log --grep commit` names commit only as an option value after the subcommand.
        self.assertEqual([], hook.gated_git_invocations("git log --grep commit"))

    def test_cd_and_dash_c_set_the_directory(self):
        self.assertEqual([("commit", "a")], hook.gated_git_invocations("cd a && git commit -m x"))
        self.assertEqual([("push", "b")], hook.gated_git_invocations("git -C b push"))
        self.assertEqual(
            [("commit", os.path.join("a", "b"))],
            hook.gated_git_invocations("cd a; git -C b commit"),
        )

    def test_env_prefix_and_chain(self):
        found = hook.gated_git_invocations("GIT_EDITOR=true git commit && git push")
        self.assertEqual([("commit", None), ("push", None)], found)

    def test_powershell_keeps_windows_paths(self):
        found = hook.gated_git_invocations(
            'Set-Location "C:\\w\\tree"; git commit -m x', posix=False
        )
        self.assertEqual([("commit", "C:\\w\\tree")], found)
        found = hook.gated_git_invocations("& git.exe -C C:\\w\\tree push", posix=False)
        self.assertEqual([("push", "C:\\w\\tree")], found)


@NEEDS_GIT
class Verdicts(unittest.TestCase):
    """main() against a real git checkout with a stand-in checker."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ratchet_hook_")
        self.git("init", "-q")
        self.git("commit", "-q", "--allow-empty", "-m", "base")
        self.git("update-ref", "refs/remotes/origin/main", "HEAD")
        self.checker = Path(self.tmp, hook.CHECKER)
        self.argv_log = Path(self.tmp, "checker_argv.json")
        # The hook trusts only the project's own origin/main; the throwaway repo IS the project.
        self._old_project = os.environ.get("CLAUDE_PROJECT_DIR")
        os.environ["CLAUDE_PROJECT_DIR"] = self.tmp

    def tearDown(self):
        if self._old_project is None:
            os.environ.pop("CLAUDE_PROJECT_DIR", None)
        else:
            os.environ["CLAUDE_PROJECT_DIR"] = self._old_project
        shutil.rmtree(self.tmp, ignore_errors=True)

    def git(self, *args):
        cmd = ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args]
        subprocess.run(cmd, cwd=self.tmp, check=True, capture_output=True)

    def stand_in(self, status, output, log):
        return textwrap.dedent(
            f"""\
            import json, sys
            assert "--root" in sys.argv, sys.argv
            with open({str(log)!r}, "w") as f:
                json.dump(sys.argv[1:], f)
            sys.stdout.write({output!r})
            sys.exit({status})
            """
        )

    def install_checker(self, status, output="checker said so\n"):
        """Commit a stand-in checker and point origin/main at it: the only copy the hook runs."""
        self.checker.parent.mkdir(parents=True, exist_ok=True)
        self.checker.write_text(self.stand_in(status, output, self.argv_log))
        self.git("add", hook.CHECKER)
        self.git("commit", "-q", "-m", "checker")
        self.git("update-ref", "refs/remotes/origin/main", "HEAD")

    def checker_argv(self):
        if not self.argv_log.exists():
            return None
        return json.loads(self.argv_log.read_text())

    def run_hook(self, command, cwd=None, tool="Bash"):
        payload = {"tool_name": tool, "tool_input": {"command": command}, "cwd": cwd or self.tmp}
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            status = hook.main(json.dumps(payload))
        return status, out.getvalue(), err.getvalue()

    def test_non_gated_command_runs_nothing(self):
        self.install_checker(1)
        self.assertEqual((0, "", ""), self.run_hook("git status"))

    def test_missing_checker_fails_open_visibly(self):
        status, out, _ = self.run_hook("git commit -m x")
        self.assertEqual(0, status)
        self.assertIn("ratchet UNVERIFIED", json.loads(out)["systemMessage"])

    def test_clean_checker_allows_silently(self):
        self.install_checker(0)
        self.assertEqual((0, "", ""), self.run_hook("git push"))

    def test_checker_runs_in_delta_mode_against_origin_main(self):
        self.install_checker(0)
        self.run_hook("git commit -m x")
        argv = self.checker_argv()
        self.assertIsNotNone(argv, "the checker never ran")
        self.assertEqual(["--base", "refs/remotes/origin/main"], argv[argv.index("--base") :][:2])

    def test_missing_origin_main_fails_open_without_running_the_checker(self):
        self.install_checker(0)
        self.git("update-ref", "-d", "refs/remotes/origin/main")
        status, out, _ = self.run_hook("git push")
        self.assertEqual(0, status)
        message = json.loads(out)["systemMessage"]
        self.assertIn("ratchet UNVERIFIED", message)
        self.assertIn("origin/main", message)
        self.assertIsNone(self.checker_argv())

    def test_violation_warns_but_does_not_block(self):
        self.assertFalse(hook.BLOCK_ON_VIOLATION, "the rollout is warn-only")
        self.install_checker(1, "INCREASE file-size|x 10 -> 11\n")
        status, out, err = self.run_hook("git commit -m x")
        self.assertEqual(0, status)
        message = json.loads(out)["systemMessage"]
        self.assertIn("ratchet WARNING", message)
        self.assertIn("INCREASE file-size|x 10 -> 11", message)
        self.assertIn("arch_ratchet_baseline.txt", message)  # says how to fix it
        self.assertIn("INCREASE file-size|x 10 -> 11", err)

    def test_blocking_switch_turns_a_violation_into_exit_2(self):
        self.install_checker(1, "INCREASE file-size|x 10 -> 11\n")
        original = hook.BLOCK_ON_VIOLATION
        hook.BLOCK_ON_VIOLATION = True
        try:
            status, out, err = self.run_hook("git commit -m x")
        finally:
            hook.BLOCK_ON_VIOLATION = original
        self.assertEqual(2, status)
        self.assertEqual("", out)
        self.assertIn("INCREASE file-size|x 10 -> 11", err)

    def test_long_findings_are_truncated_in_the_message(self):
        self.install_checker(1, "x" * (hook.SYSTEM_MESSAGE_LIMIT * 2))
        _status, out, err = self.run_hook("git commit -m x")
        self.assertLess(len(json.loads(out)["systemMessage"]), hook.SYSTEM_MESSAGE_LIMIT + 100)
        self.assertIn("truncated", err)

    def test_a_working_tree_checker_is_never_executed(self):
        # A PR branch or a dirty tree may carry its own check_arch_ratchet.py; the hook runs before
        # the permission prompt, so only the copy reviewed onto origin/main may execute.
        self.install_checker(0)
        planted = Path(self.tmp, "planted.json")
        self.checker.write_text(self.stand_in(0, "", planted))
        self.run_hook("git commit -m x")
        self.assertFalse(planted.exists(), "the working-tree checker was executed")
        self.assertIsNotNone(self.checker_argv(), "the origin/main checker did not run")

    def test_a_tag_named_origin_main_cannot_replace_the_checker(self):
        # git resolves `origin/main` via refs/tags/ before refs/remotes/; a fetched fork tag of that
        # name must not swap in its own checker.
        self.install_checker(0)
        planted = Path(self.tmp, "planted.json")
        self.checker.write_text(self.stand_in(0, "", planted))
        self.git("add", hook.CHECKER)
        self.git("commit", "-q", "-m", "fork checker")
        self.git("tag", "origin/main")
        self.git("checkout", "-q", "HEAD~1")
        self.run_hook("git commit -m x")
        self.assertFalse(planted.exists(), "a tag named origin/main supplied the checker")
        self.assertIsNotNone(
            self.checker_argv(), "the refs/remotes/origin/main checker did not run"
        )

    def test_another_repository_runs_nothing(self):
        # A scratch clone of a contributor's branch has its own origin/main and its own checker.
        other = tempfile.mkdtemp(prefix="ratchet_other_")
        self.addCleanup(shutil.rmtree, other, ignore_errors=True)
        marker = Path(other, "ran.json")

        def git(*args):
            cmd = ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args]
            subprocess.run(cmd, cwd=other, check=True, capture_output=True)

        git("init", "-q")
        Path(other, hook.CHECKER).parent.mkdir(parents=True)
        Path(other, hook.CHECKER).write_text(self.stand_in(0, "", marker))
        git("add", hook.CHECKER)
        git("commit", "-q", "-m", "foreign checker")
        git("update-ref", "refs/remotes/origin/main", "HEAD")
        self.install_checker(0)
        status, out, _ = self.run_hook(f"git -C {Path(other).as_posix()} push")
        self.assertEqual(0, status)
        self.assertIn("not this project's repository", json.loads(out)["systemMessage"])
        self.assertFalse(marker.exists(), "a foreign repository's checker was executed")

    def test_no_project_dir_runs_nothing(self):
        self.install_checker(0)
        os.environ.pop("CLAUDE_PROJECT_DIR", None)
        status, out, _ = self.run_hook("git commit -m x")
        self.assertEqual(0, status)
        self.assertIn("CLAUDE_PROJECT_DIR", json.loads(out)["systemMessage"])
        self.assertIsNone(self.checker_argv())

    def test_could_not_evaluate_fails_open(self):
        self.install_checker(2)
        status, out, _ = self.run_hook("git commit -m x")
        self.assertEqual(0, status)
        self.assertIn("exited 2", json.loads(out)["systemMessage"])

    def test_directory_comes_from_dash_c(self):
        self.install_checker(1)
        outside = tempfile.mkdtemp(prefix="ratchet_hook_outside_")
        try:
            status, out, _ = self.run_hook(
                f"git -C {Path(self.tmp).as_posix()} commit -m x", cwd=outside
            )
        finally:
            shutil.rmtree(outside, ignore_errors=True)
        self.assertEqual(0, status)
        self.assertIn("ratchet WARNING", json.loads(out)["systemMessage"])

    def test_not_a_checkout_fails_open(self):
        outside = tempfile.mkdtemp(prefix="ratchet_hook_outside_")
        try:
            status, out, _ = self.run_hook("git commit -m x", cwd=outside)
        finally:
            shutil.rmtree(outside, ignore_errors=True)
        self.assertEqual(0, status)
        self.assertIn("ratchet UNVERIFIED", out)

    def test_end_to_end_as_a_process(self):
        self.install_checker(1)
        payload = json.dumps(
            {"tool_name": "Bash", "tool_input": {"command": "git push"}, "cwd": self.tmp}
        )
        proc = subprocess.run(
            [sys.executable, str(HERE / "ratchet_hook.py")],
            input=payload,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn("checker said so", json.loads(proc.stdout)["systemMessage"])
        self.assertIn("checker said so", proc.stderr)


class Registration(unittest.TestCase):
    """The settings entry must be able to block once BLOCK_ON_VIOLATION is flipped.

    A `||` interpreter chain would swallow exit 2, so this holds even while the rollout is
    warn-only -- otherwise flipping the switch would silently do nothing.
    """

    def test_settings_register_the_hook_for_both_shells(self):
        settings = json.loads((REPO_ROOT / ".claude" / "settings.json").read_text())
        entries = settings["hooks"]["PreToolUse"]
        hooks = [h for e in entries if e.get("matcher") == "Bash|PowerShell" for h in e["hooks"]]
        commands = [h["command"] for h in hooks if "ratchet_hook.py" in h["command"]]
        self.assertEqual(1, len(commands))
        self.assertNotIn("|| python", commands[0])
        self.assertIn('[ "$r" -eq 2 ] && exit 2', commands[0])


if __name__ == "__main__":
    unittest.main()
