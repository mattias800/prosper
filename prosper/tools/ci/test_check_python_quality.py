"""Tests for check_python_quality: pure rules plus a real throwaway-repo run of the whole gate."""

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

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

    def test_ratchet_reason_names_both_counts(self):
        (msg,) = cpq.ratchet_violations("a.py", 2, 5)
        self.assertIn("2 -> 5", msg)
        self.assertEqual([], cpq.ratchet_violations("a.py", 5, 5))

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

    def test_modified_file_may_keep_findings_but_not_add_them(self):
        self.write("prosper/tools/x/old.py", "import importlib.util\nx = 2\n")
        self.commit("keep")
        self.assertEqual([], self.verdict())
        self.write("prosper/tools/x/old.py", "import importlib.util\nimport sys\nx = 2\n")
        self.commit("worse")
        self.assertTrue(any("findings rose" in p for p in self.verdict()))


if __name__ == "__main__":
    unittest.main()
