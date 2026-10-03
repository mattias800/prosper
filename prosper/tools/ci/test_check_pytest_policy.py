"""Unit tests for check_pytest_policy.py."""

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


def test_compliant_class():
    code = "class TestExample:\n    def test_one(self):\n        assert True\n"
    violates, reason = cpp.is_non_compliant(code)
    assert not violates
    assert reason == ""


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


def test_violating_script_only():
    code = "import os\nx = 10\nassert x == 10\n"
    violates, reason = cpp.is_non_compliant(code)
    assert violates
    assert "no test_* functions or Test* classes" in reason


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


def test_scan_tree_and_ignore_dirs(tmp_path):
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
