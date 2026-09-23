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
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "prosper/tools/session_start.py"
HOOK = json.loads((ROOT / ".claude/settings.json").read_text())["hooks"]["SessionStart"][0]["hooks"][0]
SPEC = importlib.util.spec_from_file_location("session_start", TOOL)
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def hook_command(project):
    """The configured hook, ready to run, with the documented project-dir substitution.

    Both hook forms are accepted so this keeps exercising whatever is configured rather than a
    remembered shape. Exec form (``command`` plus ``args``) is a real argv and involves no shell.
    Shell form (``command`` alone) is one string handed to the platform shell -- cmd.exe on
    Windows, ``sh`` elsewhere -- which is what lets a single committed string try more than one
    interpreter name, since no name exists on every platform (#3540). Quoted paths keep a project
    directory containing spaces in one argument under either form.
    """
    substitute = lambda text: text.replace("${CLAUDE_PROJECT_DIR}", project)
    if "args" in HOOK:
        return [substitute(HOOK["command"])] + [substitute(x) for x in HOOK["args"]], False
    return substitute(HOOK["command"]), True


def interpreters(command):
    """The interpreter names the configured hook tries, in the order it tries them."""
    names = []
    text = command if isinstance(command, str) else " ".join(command)
    for segment in text.split("||"):
        head = segment.strip().split(maxsplit=1)
        if head and head[0] != "echo" and head[0] not in names:
            names.append(head[0])
    return names


def diagnosis(command, proc):
    """Say which interpreter could not be started, not merely that a number was not zero."""
    tried = ", ".join(name + " -> " + (shutil.which(name) or "NOT FOUND")
                      for name in interpreters(command))
    return "\n".join([
        "The SessionStart hook configured in .claude/settings.json exited "
        + str(proc.returncode) + " and produced no session context.",
        "  configured: " + (command if isinstance(command, str) else " ".join(command)),
        "  interpreters this host resolves: " + tried,
        "  hook stderr: " + (proc.stderr.strip() or "(none)"),
        "A stock python.org install on Windows provides python.exe only, and the name python3 "
        "there resolves to the Microsoft Store App Execution Alias stub, which exits 9009 without "
        "running anything (#3540). A failure here is a hook configuration problem on this host, "
        "not a logic failure in session_start.py.",
    ])


# The arms below hold the configured hook to Claude Code's own budget, HOOK["timeout"]. That budget
# is for the hook's WORK, and on a freshly built CI runner it is not the work that gets measured the
# first time. Measured on twelve replicas of the Windows MinGW job (#3583): the first hook of the run
# took 0.86-11.7 s where every later one took 0.13-0.76 s, and ALL of the excess elapsed before
# session_start.py ran its first statement -- while cmd.exe, the py launcher and the interpreter
# were started for the first time after a full build. The check itself, git work included, took
# about 0.6 s even inside the 11.7 s call. Six CI failures in 300 runs were exactly that first
# launch overrunning the cap; none was ever one of the five later hooks.
#
# So the launch chain is started once per class, outside every measured call, before any arm times
# it. The warm-up runs the configured command verbatim; only the directory differs, and it is not a
# repository, so the check reports UNVERIFIED at its first git call and does no other work.
WARMUP_CAP = 300  # seconds. Generous on purpose: it bounds a hang, it is not a performance claim.


def warm_hook_chain():
    """Start the configured hook once and return how long that took, in seconds."""
    with tempfile.TemporaryDirectory(prefix="session start warm-up ") as empty:
        command, shell = hook_command(empty)
        # A ceiling stops git discovering a repository ABOVE the temporary directory, which would
        # turn the warm-up into a real check of some unrelated checkout (with a network ls-remote).
        env = dict(os.environ, GIT_CEILING_DIRECTORIES=str(Path(empty).parent))
        start = time.monotonic()
        try:
            subprocess.run(command, shell=shell, input="{}", capture_output=True, text=True,
                           timeout=WARMUP_CAP, env=env)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Starting the configured SessionStart hook once, against an empty "
                               "directory, did not finish within " + str(WARMUP_CAP) + " s. That "
                               "launch does no git work to speak of, so this host cannot start the "
                               "hook's interpreter chain at all (#3583).") from exc
        return time.monotonic() - start


def overrun(command, exc, elapsed, warmup):
    """Say how long the hook really took and what it printed, not merely that a cap was hit."""
    output = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode(errors="replace")
    return "\n".join([
        "The SessionStart hook did not finish inside its configured " + str(HOOK["timeout"])
        + " s budget.",
        "  configured: " + (command if isinstance(command, str) else " ".join(command)),
        # run() kills only the process it started -- the shell -- and then reads until the pipes
        # close, which is when the interpreter that shell started has exited. So this figure is the
        # hook's real duration, not the cap. A CTest entry far longer than the cap is that wait.
        "  subprocess.run returned after %.1f s; the untimed warm-up launch took %.1f s." % (
            elapsed, warmup),
        "  stdout: " + (output.strip() or "(none)"),
        "A verdict on stdout means the hook was slow, not wrong; no stdout means it never got as "
        "far as printing. The check itself bounds its git work at 12 s, so a duration well past "
        "that was spent starting the interpreter, not checking (#3583).",
    ])


class StartupTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.warmup_seconds = warm_hook_chain()

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
        # Execute what Claude Code executes: the command exactly as configured, in the form it is
        # configured in, with the documented project-dir substitution.
        command, shell = hook_command(str(repo or self.repo))
        start = time.monotonic()
        try:
            proc = subprocess.run(command, shell=shell, input="{}", capture_output=True, text=True,
                                  timeout=HOOK["timeout"])
        except subprocess.TimeoutExpired as exc:
            self.fail(overrun(command, exc, time.monotonic() - start, self.warmup_seconds))
        self.assertEqual(proc.returncode, 0, diagnosis(command, proc))
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

    def test_hook_without_any_interpreter_still_reports_that_it_did_not_run(self):
        # Discoverability is the point of the hook, not a nicety: its job is to tell a session that
        # the charter it just loaded may be stale, so a hook that cannot start its interpreter has
        # to say so in session context. Measured on Windows 2026-09-11 with the previous exec-form
        # `python3` configuration: the session carried no session-start line and no error of any
        # kind -- an unarmed instrument reading as a confident silence (#3540).
        #
        # PATH is emptied rather than an interpreter renamed. The case under test is a host that
        # has none of the configured names, and a control built from the interpreters this host
        # does have could not express it.
        bare = Path(self.tmp.name) / "no interpreters here"
        bare.mkdir()
        command, shell = hook_command(str(self.repo))
        proc = subprocess.run(command, shell=shell, input="{}", capture_output=True, text=True,
                              timeout=HOOK["timeout"], env=dict(os.environ, PATH=str(bare)))
        # Not an assertion about the exit status: the contract is that something reaches stdout,
        # which is what SessionStart turns into context.
        self.assertIn("session-start", proc.stdout, proc.stderr)
        self.assertIn("UNVERIFIED", proc.stdout, proc.stderr)
        # Self-invalidating control: an interpreter reached despite the emptied PATH would make
        # this arm vacuous, and reporting the real check's verdict is how that shows up.
        self.assertNotIn("OK:", proc.stdout)

    def test_hook_overrun_names_its_duration_and_output(self):
        # The next overrun has to be a diagnosis rather than a cross-check against main (#3583):
        # the report must carry how long the hook really ran and whatever it printed, because
        # those two facts are what separate "slow to start" from "hung" and from "wrong". Built by
        # hand, since a real overrun is exactly what this file no longer produces on purpose.
        verdict = json.dumps({"hookSpecificOutput": {"additionalContext": "OK: current"}})
        for stdout in (verdict, verdict.encode(), None):
            with self.subTest(stdout=type(stdout).__name__):
                exc = subprocess.TimeoutExpired("hook", HOOK["timeout"], output=stdout)
                report = overrun("hook", exc, 47.3, 0.5)
                self.assertIn("returned after 47.3 s", report)
                self.assertIn("warm-up launch took 0.5 s", report)
                self.assertIn("OK: current" if stdout else "stdout: (none)", report)

    def test_root_is_found_from_a_subdirectory(self):
        # The relative derivation that replaced --show-toplevel still has to climb: this is the
        # only arm where the cdup is non-empty, so it is what proves the join is right.
        self.assertIn("OK:", self.check(0, repo=self.repo / "prosper/tools"))


if __name__ == "__main__":
    unittest.main()
