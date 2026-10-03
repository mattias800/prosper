#!/usr/bin/env python3
"""Tests for check_pr_template.py (exit code is truth)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import check_pr_template as cpt  # noqa: E402


class CheckPrTemplateTests(unittest.TestCase):
    def test_selftest_passes(self):
        """Selftest positive and negative controls pass cleanly."""
        self.assertEqual(cpt.selftest(), 0)

    def test_strip_html_comments(self):
        text = "Hello <!-- comment here --> world <!-- multiline\n comment --> end"
        self.assertEqual(cpt.strip_html_comments(text).strip(), "Hello  world  end")

    def test_event_json_valid_pull_request(self):
        with tempfile.TemporaryDirectory() as td:
            event_path = Path(td) / "event.json"
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
            event_path.write_text(
                json.dumps({"pull_request": {"body": valid_body}}), encoding="utf-8"
            )
            problems = cpt.check_event_file(event_path)
            self.assertEqual(problems, [])

    def test_event_json_non_pr_event_skips_cleanly(self):
        with tempfile.TemporaryDirectory() as td:
            event_path = Path(td) / "event.json"
            event_path.write_text(
                json.dumps({"push": {"ref": "refs/heads/main"}}), encoding="utf-8"
            )
            problems = cpt.check_event_file(event_path)
            self.assertEqual(problems, [])

    def test_event_json_missing_file_reports_error(self):
        problems = cpt.check_event_file(Path("non_existent_event_file_12345.json"))
        self.assertTrue(any("not found" in p for p in problems))


if __name__ == "__main__":
    unittest.main()
