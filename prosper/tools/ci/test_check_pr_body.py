"""Tests for check_pr_body: pure rules, the real template, and the exit-code contract."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import check_pr_body as cpb  # noqa: E402

TEMPLATE = cpb.DEFAULT_TEMPLATE.read_text(encoding="utf-8")


def filled(template):
    """A body made from the template with every section given one real line."""
    return "\n".join(
        f"## {name}\n\nSomething real for {name}.\n" for name in cpb.sections(template)
    )


class Rules(unittest.TestCase):
    def test_selftest(self):
        cpb.selftest()

    def test_filled_template_passes(self):
        # Positive control: a gate that refuses everything would satisfy every refusal arm below.
        self.assertEqual(cpb.violations(filled(TEMPLATE), TEMPLATE), [])

    def test_unfilled_template_is_rejected(self):
        # The template's Checklist ships with real items, so it alone passes unfilled; every
        # section that is only placeholder comments or an empty checkbox must be flagged.
        problems = cpb.violations(TEMPLATE, TEMPLATE)
        empty = "is empty (placeholder comments do not count)"
        self.assertIn(f"section '## Context' {empty}", problems)
        self.assertIn(f"section '## Acceptance Criteria' {empty}", problems)
        self.assertTrue(all("is empty" in p for p in problems))
        self.assertGreaterEqual(len(problems), 5)

    def test_free_form_body_names_each_missing_section(self):
        problems = cpb.violations("Fixes the build.", TEMPLATE)
        self.assertIn("missing section '## Verification'", problems)
        self.assertTrue(all(p.startswith("missing") for p in problems))

    def test_unticked_checklist_is_fine(self):
        body = filled(TEMPLATE).replace("Something real for Checklist.", "- [ ] not yet")
        self.assertEqual(cpb.violations(body, TEMPLATE), [])

    def test_crlf_body_passes(self):
        body = filled(TEMPLATE).replace("\n", "\r\n")
        self.assertEqual(cpb.violations(body, TEMPLATE), [])

    def test_template_without_headings_is_an_error(self):
        with self.assertRaises(ValueError):
            cpb.violations("x", "no headings here")

    def test_exit_codes(self):
        with tempfile.TemporaryDirectory() as d:
            good = Path(d, "good.md")
            good.write_text(filled(TEMPLATE), encoding="utf-8")
            bad = Path(d, "bad.md")
            bad.write_text("nope", encoding="utf-8")
            self.assertEqual(cpb.main(["--body-file", str(good)]), 0)
            self.assertEqual(cpb.main(["--body-file", str(bad)]), 1)
            self.assertEqual(cpb.main(["--body-file", str(Path(d, "absent.md"))]), 2)
        with mock.patch.dict(os.environ, clear=False):
            os.environ.pop("PR_BODY", None)
            self.assertEqual(cpb.main([]), 2)


if __name__ == "__main__":
    unittest.main()
