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
        subprocess.run(["git", "init", "-q", self.tmp], check=True)
        self.checker = Path(self.tmp, hook.CHECKER)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def install_checker(self, status, output="checker said so\n"):
        self.checker.parent.mkdir(parents=True, exist_ok=True)
        self.checker.write_text(
            textwrap.dedent(
                f"""\
                import sys
                assert "--root" in sys.argv, sys.argv
                sys.stdout.write({output!r})
                sys.exit({status})
                """
            )
        )

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

    def test_violation_blocks_with_checker_output(self):
        self.install_checker(1, "INCREASE file-size|x 10 -> 11\n")
        status, out, err = self.run_hook("git commit -m x")
        self.assertEqual(2, status)
        self.assertEqual("", out)
        self.assertIn("INCREASE file-size|x 10 -> 11", err)

    def test_could_not_evaluate_fails_open(self):
        self.install_checker(2)
        status, out, _ = self.run_hook("git commit -m x")
        self.assertEqual(0, status)
        self.assertIn("exited 2", json.loads(out)["systemMessage"])

    def test_directory_comes_from_dash_c(self):
        self.install_checker(1)
        outside = tempfile.mkdtemp(prefix="ratchet_hook_outside_")
        try:
            status, _, _ = self.run_hook(
                f"git -C {Path(self.tmp).as_posix()} commit -m x", cwd=outside
            )
        finally:
            shutil.rmtree(outside, ignore_errors=True)
        self.assertEqual(2, status)

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
        self.assertEqual(2, proc.returncode, proc.stderr)
        self.assertIn("checker said so", proc.stderr)


class Registration(unittest.TestCase):
    """The settings entry must be able to block: a `||` interpreter chain would swallow exit 2."""

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
