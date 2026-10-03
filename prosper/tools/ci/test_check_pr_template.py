#!/usr/bin/env python3
"""Tests for check_pr_template.py (exit code is truth)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import check_pr_template as cpt  # noqa: E402


def test_selftest_passes():
    """Selftest positive and negative controls pass cleanly."""
    assert cpt.selftest() == 0


def test_strip_html_comments():
    text = "Hello <!-- comment here --> world <!-- multiline\n comment --> end"
    assert cpt.strip_html_comments(text).strip() == "Hello  world  end"


def test_event_json_valid_pull_request(tmp_path: Path):
    event_path = tmp_path / "event.json"
    valid_body = """## Context
Some context.
## Higher Goal
Some goal.
## Acceptance Criteria
- [x] Done
## Out of Scope
None
## Summary of Changes
Changed a thing
## Verification
- Tests added: test_foo
- Red without the fix: observed red
- Commands run + results: ctest passed
- Could not verify: none
## Checklist
- [x] All good
"""
    event_path.write_text(json.dumps({"pull_request": {"body": valid_body}}), encoding="utf-8")
    problems = cpt.check_event_file(event_path)
    assert problems == []


def test_event_json_non_pr_event_skips_cleanly(tmp_path: Path):
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps({"push": {"ref": "refs/heads/main"}}), encoding="utf-8")
    problems = cpt.check_event_file(event_path)
    assert problems == []


def test_event_json_missing_file_reports_error():
    problems = cpt.check_event_file(Path("non_existent_event_file_12345.json"))
    assert any("not found" in p for p in problems)


if __name__ == "__main__":
    try:
        import pytest

        sys.exit(pytest.main([__file__]))
    except ImportError:
        # Fallback runner when running outside uv/pytest environment
        import inspect

        funcs = [
            obj
            for name, obj in list(globals().items())
            if name.startswith("test_") and inspect.isfunction(obj)
        ]
        failed = 0
        for f in funcs:
            try:
                # Provide tmp_path if test expects it
                sig = inspect.signature(f)
                if "tmp_path" in sig.parameters:
                    import tempfile

                    with tempfile.TemporaryDirectory() as td:
                        f(Path(td))
                else:
                    f()
                print(f"ok   {f.__name__}")
            except Exception as e:
                print(f"FAIL {f.__name__}: {e}", file=sys.stderr)
                failed += 1
        print(f"{len(funcs)} tests, {failed} failed")
        sys.exit(1 if failed else 0)
