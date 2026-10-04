"""Pin pytest discovery and prevent failed/option-shaped baselines from weakening the gate."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import check_pytest_policy as cpp  # noqa: E402


def test_compliant_function():
    code = "def test_example():\n    assert 1 == 1\n"
    violates, reason = cpp.is_non_compliant(code)
    assert not violates
    assert reason == ""


def test_compliant_async_function():
    code = "async def test_async():\n    assert True\n"
    violates, reason = cpp.is_non_compliant(code)
    assert not violates
    assert reason == ""


def test_compliant_class_with_methods():
    code = "class TestExample:\n    def test_one(self):\n        assert True\n"
    violates, reason = cpp.is_non_compliant(code)
    assert not violates
    assert reason == ""


def test_compliant_class_with_unrelated_base():
    code = (
        "class NotATestCase:\n"
        "    pass\n"
        "class TestValid(NotATestCase):\n"
        "    def test_ok(self):\n"
        "        pass\n"
    )
    violates, reason = cpp.is_non_compliant(code)
    assert not violates
    assert reason == ""


def test_violating_empty_test_class():
    code = "class TestEmpty:\n    pass\n"
    violates, reason = cpp.is_non_compliant(code)
    assert violates
    assert "no eligible test_* functions" in reason


def test_violating_nested_test_function():
    code = "def helper():\n    def test_hidden():\n        pass\n"
    violates, reason = cpp.is_non_compliant(code)
    assert violates
    assert "no eligible test_* functions" in reason


def test_violating_test_method_in_non_test_class():
    code = "class Helper:\n    def test_in_helper(self):\n        pass\n"
    violates, reason = cpp.is_non_compliant(code)
    assert violates
    assert "no eligible test_* functions" in reason


def test_violating_test_class_with_init():
    code = (
        "class TestIgnoredByPytest:\n"
        "    def __init__(self):\n"
        "        pass\n"
        "    def test_m(self):\n"
        "        pass\n"
    )
    violates, reason = cpp.is_non_compliant(code)
    assert violates
    assert "no eligible test_* functions" in reason


def test_violating_unittest_subclass_attribute():
    code = (
        "import unittest\n"
        "class MySuite(unittest.TestCase):\n"
        "    def test_run(self):\n        pass\n"
    )
    violates, reason = cpp.is_non_compliant(code)
    assert violates
    assert "subclasses unittest.TestCase" in reason
    assert "MySuite" in reason


def test_violating_unittest_subclass_name():
    code = (
        "from unittest import TestCase\n"
        "class MySuite(TestCase):\n"
        "    def test_run(self):\n        pass\n"
    )
    violates, reason = cpp.is_non_compliant(code)
    assert violates
    assert "subclasses unittest.TestCase" in reason


def test_violating_unittest_subclass_aliased_import():
    code = (
        "from unittest import TestCase as Base\n"
        "class TestLegacy(Base):\n"
        "    def test_m(self):\n        pass\n"
    )
    violates, reason = cpp.is_non_compliant(code)
    assert violates
    assert "subclasses unittest.TestCase" in reason
    assert "TestLegacy" in reason


def test_violating_unittest_subclass_aliased_module():
    code = (
        "import unittest as ut\n"
        "class TestLegacy(ut.TestCase):\n"
        "    def test_m(self):\n        pass\n"
    )
    violates, reason = cpp.is_non_compliant(code)
    assert violates
    assert "subclasses unittest.TestCase" in reason


def test_violating_unittest_subclass_hierarchy():
    code = (
        "import unittest\n"
        "class BaseFixture(unittest.TestCase):\n"
        "    pass\n"
        "class DerivedTests(BaseFixture):\n"
        "    def test_m(self):\n        pass\n"
    )
    violates, reason = cpp.is_non_compliant(code)
    assert violates
    assert "subclasses unittest.TestCase" in reason
    assert "BaseFixture" in reason or "DerivedTests" in reason


def test_violating_script_only():
    code = "import os\nx = 10\nassert x == 10\n"
    violates, reason = cpp.is_non_compliant(code)
    assert violates
    assert "no eligible test_* functions" in reason


def test_syntax_error_reported():
    code = "def invalid_syntax(:"
    violates, reason = cpp.is_non_compliant(code)
    assert violates
    assert "syntax error" in reason


def test_check_new_violation():
    found = {"prosper/tests/new_bad_test.py": "script-only"}
    allow = set()
    errs = cpp.check(found, allow)
    assert len(errs) == 1
    assert "new_bad_test.py" in errs[0]
    assert "pytest conventions" in errs[0]


def test_check_stale_allowlist_entry():
    found = set()
    allow = {"prosper/tests/migrated_test.py"}
    errs = cpp.check(found, allow)
    assert len(errs) == 1
    assert "migrated or gone" in errs[0]
    assert "pytest_legacy_allowlist.txt" in errs[0]


def test_check_compliant_with_allowlist():
    found = {"prosper/tests/legacy_test.py": "subclasses unittest.TestCase"}
    allow = {"prosper/tests/legacy_test.py"}
    errs = cpp.check(found, allow)
    assert errs == []


def test_check_shrink_only_baseline_rejects_addition():
    baseline = {"prosper/tests/old_legacy.py"}
    candidate = {"prosper/tests/old_legacy.py", "prosper/tests/new_bad.py"}
    found = {
        "prosper/tests/old_legacy.py": "subclasses unittest.TestCase",
        "prosper/tests/new_bad.py": "subclasses unittest.TestCase",
    }
    errs = cpp.check(found, candidate, baseline_allow=baseline)
    assert any("newly added to pytest_legacy_allowlist.txt" in e for e in errs)
    assert any("new_bad.py" in e for e in errs)


def test_check_shrink_only_baseline_accepts_removal():
    baseline = {"prosper/tests/old_legacy.py"}
    candidate = set()
    found = set()
    errs = cpp.check(found, candidate, baseline_allow=baseline)
    assert errs == []


def test_scan_tree_and_ignore_dirs(tmp_path: Path):
    # Compliant file
    (tmp_path / "test_good.py").write_text("def test_ok(): assert True\n", encoding="utf-8")
    # Violating file
    (tmp_path / "test_bad.py").write_text("x = 1\n", encoding="utf-8")
    # File in excluded dir
    venv_dir = tmp_path / ".venv"
    venv_dir.mkdir()
    (venv_dir / "test_venv.py").write_text("x = 1\n", encoding="utf-8")

    found = cpp.scan(tmp_path)
    assert "test_bad.py" in found
    assert "test_good.py" not in found
    assert ".venv/test_venv.py" not in found


def _fixture_env() -> dict[str, str]:
    """Keep temporary Git controls independent of user config and injected Git settings."""
    env = {
        key: os.environ[key]
        for key in ("PATH", "SYSTEMROOT", "SystemRoot", "LANG", "LC_ALL", "TMP", "TEMP")
        if key in os.environ
    }
    env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1", PYTHONDONTWRITEBYTECODE="1")
    return env


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", f"core.hooksPath={os.devnull}", *args],
        cwd=repo,
        env=_fixture_env(),
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _isolated_repo(tmp_path: Path, *, with_allowlist: bool = True) -> tuple[Path, Path, str]:
    """Exercise the real CLI in an owned repository, never the developer's index or hooks."""
    repo = tmp_path / "repo"
    checker = repo / "prosper/tools/ci/check_pytest_policy.py"
    checker.parent.mkdir(parents=True)
    checker.write_bytes(Path(cpp.__file__).read_bytes())
    tests = repo / "prosper/tests"
    tests.mkdir()
    (tests / "test_old.py").write_text("assert True\n", encoding="utf-8")
    if with_allowlist:
        (checker.parent / "pytest_legacy_allowlist.txt").write_text(
            "tests/test_old.py\n", encoding="utf-8"
        )
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(
        repo,
        "-c",
        "user.name=Policy control",
        "-c",
        "user.email=policy@example.invalid",
        "commit",
        "-qm",
        "immutable control baseline",
    )
    return repo, checker, _git(repo, "rev-parse", "HEAD").strip()


def _checker(repo: Path, checker: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-B", str(checker), *args],
        cwd=repo,
        env=_fixture_env(),
        capture_output=True,
        text=True,
        check=False,
    )


def test_real_git_baseline_rejects_allowlist_growth(tmp_path: Path):
    repo, checker, commit = _isolated_repo(tmp_path)
    assert _checker(repo, checker, "--base", commit).returncode == 0
    (repo / "prosper/tests/test_added.py").write_text("assert False\n", encoding="utf-8")
    (checker.parent / "pytest_legacy_allowlist.txt").write_text(
        "tests/test_old.py\ntests/test_added.py\n", encoding="utf-8"
    )
    result = _checker(repo, checker, "--base", commit)
    assert result.returncode == 1
    assert "test_added.py: newly added" in result.stdout


def test_invalid_baseline_cannot_report_policy_success(tmp_path: Path):
    repo, checker, _ = _isolated_repo(tmp_path)
    result = _checker(repo, checker, "--base", "refs/heads/missing-policy-control")
    assert result.returncode == 2
    assert "could not evaluate baseline" in result.stderr
    assert "pytest policy ok" not in result.stdout


def test_option_shaped_base_cannot_write_git_output(tmp_path: Path):
    repo, checker, _ = _isolated_repo(tmp_path)
    prefix = tmp_path / "output-sink"
    sink = Path(str(prefix) + ":prosper/tools/ci/pytest_legacy_allowlist.txt")
    if os.name != "nt":  # Windows cannot create the colon-bearing output prefix.
        sink.parent.mkdir(parents=True)
    result = _checker(repo, checker, "--base=--output=" + str(prefix))
    assert result.returncode == 2
    assert "not a Git option" in result.stderr
    assert not sink.exists()


def test_explicit_bootstrap_grandfathers_only_existing_legacy_files(tmp_path: Path):
    repo, checker, commit = _isolated_repo(tmp_path, with_allowlist=False)
    allowlist = checker.parent / "pytest_legacy_allowlist.txt"
    allowlist.write_text("tests/test_old.py\n", encoding="utf-8")
    undeclared = _checker(repo, checker, "--base", commit)
    assert undeclared.returncode == 2
    assert "requires explicit --bootstrap-allowlist" in undeclared.stderr
    assert _checker(repo, checker, "--base", commit, "--bootstrap-allowlist").returncode == 0
    assert cpp.load_baseline_allowlist(repo / "prosper", commit, bootstrap=True) == {
        "tests/test_old.py"
    }
    (repo / "prosper/tests/test_added.py").write_text("assert False\n", encoding="utf-8")
    allowlist.write_text("tests/test_old.py\ntests/test_added.py\n", encoding="utf-8")
    result = _checker(repo, checker, "--base", commit, "--bootstrap-allowlist")
    assert result.returncode == 1
    assert "test_added.py: newly added" in result.stdout


def test_bootstrap_cannot_excuse_an_unavailable_commit(tmp_path: Path):
    repo, checker, _ = _isolated_repo(tmp_path)
    result = _checker(
        repo, checker, "--base", "refs/heads/missing-policy-control", "--bootstrap-allowlist"
    )
    assert result.returncode == 2
    assert "could not evaluate baseline" in result.stderr
    assert "pytest policy ok" not in result.stdout


def test_explicit_external_baseline_is_a_read_only_input(tmp_path: Path):
    repo, checker, _ = _isolated_repo(tmp_path)
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("tests/test_old.py\n", encoding="utf-8")
    original = baseline.read_bytes()
    assert _checker(repo, checker, "--baseline-file", str(baseline)).returncode == 0
    assert baseline.read_bytes() == original
    conflicting = _checker(repo, checker, "--baseline-file", str(baseline), "--base", "missing")
    assert conflicting.returncode == 2
