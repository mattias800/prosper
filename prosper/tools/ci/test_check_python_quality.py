"""Tests for check_python_quality: pure rules plus a real throwaway-repo run of the whole gate."""

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import check_python_quality as cpq  # noqa: E402

REPO_ROOT = HERE.parents[2]
NEEDS_RUFF = unittest.skipUnless(
    importlib.util.find_spec("ruff") is not None,
    "ruff not installed (uv sync --group dev); the python-quality CI job always runs this",
)
CLEAN = '"""Purpose."""\n\n\ndef f():\n    """Return one."""\n    return 1\n'


class PureRules(unittest.TestCase):
    """Rules that need no filesystem."""

    def test_is_python_excludes_third_party(self):
        self.assertTrue(cpq.is_python("prosper/tools/a.py"))
        self.assertFalse(cpq.is_python("prosper/third_party/x/a.py"))
        self.assertFalse(cpq.is_python("a.cpp"))

    def test_tool_without_test_is_flagged_and_with_test_is_not(self):
        tool = "prosper/tools/x/a.py"
        self.assertTrue(cpq.missing_test_violations([tool], [tool]))
        self.assertEqual(
            [], cpq.missing_test_violations([tool], [tool, "prosper/tools/x/test_a.py"])
        )
        self.assertEqual([], cpq.missing_test_violations([tool], [tool, "prosper/tests/host/t.py"]))

    def test_only_python_test_names_or_test_tree_changes_satisfy_presence(self):
        tool = "prosper/tools/x/a.py"
        self.assertTrue(cpq.missing_test_violations([tool], [tool, "prosper/tools/test_bypass.md"]))
        self.assertTrue(cpq.is_test_file("prosper/tools/test_a.py"))
        self.assertTrue(cpq.is_test_file("prosper/tests/host/test_tool.cpp"))

    def test_validate_ref_rejects_option_like_input(self):
        for bad in ("--output=x", "-x", "", "a b", "a;b"):
            with self.assertRaises(ValueError, msg=bad):
                cpq.validate_ref(bad)
        for good in ("origin/main", "HEAD~1", "abc123", "refs/heads/a-b"):
            self.assertEqual(good, cpq.validate_ref(good))

    def test_ratchet_reason_names_both_counts(self):
        (msg,) = cpq.ratchet_violations("a.py", 2, 5)
        self.assertIn("2 -> 5", msg)
        self.assertEqual([], cpq.ratchet_violations("a.py", 5, 5))

    def test_ruff_errors_and_invalid_reports_are_not_zero_findings(self):
        for code, stdout in ((2, "[]"), (0, ""), (1, "not JSON"), (0, "{}"), (0, '["x"]')):
            with self.subTest(code=code, stdout=stdout):
                result = subprocess.CompletedProcess([], code, stdout, "tool error")
                with mock.patch.object(cpq, "ruff", return_value=result):
                    with self.assertRaisesRegex(RuntimeError, "could not evaluate"):
                        cpq.count_findings("x = 1\n", "a.py")

    def test_valid_ruff_reports_are_counted(self):
        for code, stdout, count in ((0, "[]", 0), (1, "[{}, {}]", 2)):
            result = subprocess.CompletedProcess([], code, stdout, "")
            with mock.patch.object(cpq, "ruff", return_value=result):
                self.assertEqual(count, cpq.count_findings("x = 1\n", "a.py"))

    def test_nul_records_preserve_paths_rename_origins_and_non_python_tests(self):
        quoted = "prosper/tools/test_caf\u00e9\t.py"
        records = f"A\0{quoted}\0R090\0old.py\0new.py\0M\0prosper/tests/test.cpp\0"
        added, modified, touched = cpq.parse_changes(records)
        self.assertEqual([quoted], added)
        self.assertEqual({"new.py": "old.py"}, modified)
        self.assertEqual([quoted, "new.py", "prosper/tests/test.cpp"], touched)

    def test_incomplete_or_unknown_git_records_are_refused(self):
        for records in ("A\0", "R100\0old.py\0", "A\0a.py", "X\0a.py\0"):
            with self.subTest(records=records):
                with self.assertRaises(ValueError):
                    cpq.parse_changes(records)

    def test_cli_reports_unevaluated_tool_failure(self):
        with mock.patch.object(sys, "argv", ["quality", "--base", "main"]):
            with mock.patch.object(cpq, "run", side_effect=RuntimeError("tool failed")):
                self.assertEqual(2, cpq.main())

    @NEEDS_RUFF
    def test_selftest_passes(self):
        self.assertEqual(0, cpq.selftest())


@NEEDS_RUFF
class EndToEnd(unittest.TestCase):
    """Positive control: a clean PR passes; hand-built bad PRs each fail for their own reason."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.repo = Path(self._tmp.name)
        self._old = Path.cwd()
        os.chdir(self.repo)
        self.addCleanup(os.chdir, self._old)
        (self.repo / "pyproject.toml").write_text(
            (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"), encoding="utf-8"
        )
        self.git("init", "-q", "-b", "main")
        self.git("config", "user.email", "t@example.invalid")
        self.git("config", "user.name", "t")
        (self.repo / "prosper/tools/x").mkdir(parents=True)
        (self.repo / "prosper/tools/x/old.py").write_text(
            "import importlib.util\nx = 1\n", encoding="utf-8"
        )
        self.commit("base")
        self.git("checkout", "-q", "-b", "pr")

    def git(self, *args):
        subprocess.run(["git", *args], check=True, capture_output=True)

    def commit(self, msg):
        self.git("add", "-A")
        self.git("commit", "-q", "-m", msg)

    def write(self, rel, text):
        (self.repo / rel).write_text(text, encoding="utf-8")

    def verdict(self):
        return cpq.run("main")

    def test_clean_added_tool_with_test_passes(self):
        self.write("prosper/tools/x/new.py", CLEAN)
        self.write("prosper/tools/x/test_new.py", CLEAN)
        self.commit("pr")
        self.assertEqual([], self.verdict())

    def test_added_tool_without_test_fails(self):
        self.write("prosper/tools/x/new.py", CLEAN)
        self.commit("pr")
        self.assertTrue(any("no test_*.py" in p for p in self.verdict()))

    def test_added_file_without_docstring_fails(self):
        self.write("prosper/tools/x/test_new.py", "x = 1\n")
        self.commit("pr")
        self.assertTrue(any("D100" in p for p in self.verdict()))

    def test_added_unformatted_file_fails(self):
        self.write("prosper/tools/x/test_new.py", '"""Doc."""\nx = {  "a":1 }\n')
        self.commit("pr")
        self.assertTrue(any("format --check" in p for p in self.verdict()))

    def test_renamed_file_is_ratcheted_against_its_old_path(self):
        self.git("mv", "prosper/tools/x/old.py", "prosper/tools/x/moved.py")
        self.commit("pure move")
        self.assertEqual([], self.verdict())
        self.write("prosper/tools/x/moved.py", "import importlib.util\nimport sys\nx = 1\n")
        self.commit("move gains a finding")
        self.assertTrue(any("moved.py: ruff findings rose" in p for p in self.verdict()))

    def test_gate_and_selftest_give_the_same_answer_from_a_subdirectory(self):
        # ctest runs from the build tree; ruff must still find the root config (D100 included).
        self.write("prosper/tools/x/new.py", "x = 1\n")
        self.write("prosper/tools/x/test_new.py", CLEAN)
        self.commit("pr")
        os.chdir(self.repo / "prosper/tools/x")
        self.assertTrue(any("D100" in p for p in self.verdict()))
        self.assertEqual(0, cpq.selftest())

    def test_modified_file_may_keep_findings_but_not_add_them(self):
        self.write("prosper/tools/x/old.py", "import importlib.util\nx = 2\n")
        self.commit("keep")
        self.assertEqual([], self.verdict())
        self.write("prosper/tools/x/old.py", "import importlib.util\nimport sys\nx = 2\n")
        self.commit("worse")
        self.assertTrue(any("findings rose" in p for p in self.verdict()))

    def test_non_python_test_tree_file_satisfies_the_presence_rule(self):
        self.write("prosper/tools/x/new.py", CLEAN)
        (self.repo / "prosper/tests/host").mkdir(parents=True)
        self.write("prosper/tests/host/test_tool.cpp", "// Presence fixture; no execution.\n")
        self.commit("tool with a test-tree change")
        self.assertEqual([], self.verdict())

    def test_non_python_test_basename_outside_test_tree_does_not_bypass_presence(self):
        self.write("prosper/tools/x/new.py", CLEAN)
        self.write("prosper/tools/x/test_bypass.md", "# Metadata is not a Python test.\n")
        self.commit("tool with misleading metadata name")
        self.assertTrue(any("no test_*.py" in p for p in self.verdict()))

    def test_git_quoted_python_filename_is_checked(self):
        self.write("prosper/tools/x/test_caf\u00e9.py", "x = 1\n")
        self.commit("quoted path")
        self.assertTrue(any("D100" in p for p in self.verdict()))

    def test_non_ascii_edit_reaches_ruff_intact(self):
        # A MODIFIED file's base and head sources reach ruff on stdin. With text=True and no
        # encoding, Python encoded them with the host code page (cp1252 on Windows), which cannot
        # represent CJK text, so the check crashed on a compliant edit. U+4E2D is in no Western
        # code page, which is what makes this arm fail without the fix.
        self.write("prosper/tools/x/old.py", '"""CJK text."""\n\nX = "\u4e2d"\n')
        self.commit("non-ascii edit")
        self.assertEqual([], self.verdict())

    def test_renamed_python_retains_baseline_and_rejects_new_findings(self):
        self.git("mv", "prosper/tools/x/old.py", "prosper/tools/x/moved.py")
        self.commit("rename")
        self.assertEqual([], self.verdict())
        self.write("prosper/tools/x/moved.py", "import importlib.util\nimport sys\nx = 1\n")
        self.commit("renamed file gets worse")
        self.assertTrue(any("findings rose" in p for p in self.verdict()))


if __name__ == "__main__":
    unittest.main()
