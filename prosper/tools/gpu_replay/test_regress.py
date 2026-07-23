#!/usr/bin/env python3
# test_regress.py (#1258) — unit tests for the pure logic of the capsule-hash regression gate
# (regress.py): output-hash extraction from gpu_replay's log and the baseline-vs-current classifier.
# gpu_replay itself is not invoked here (that path is exercised manually against a local corpus).
import importlib.util
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = importlib.util.spec_from_file_location("prosper_regress", os.path.join(HERE, "regress.py"))
REGRESS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REGRESS)


class ParseOutputHashTests(unittest.TestCase):
    def test_takes_the_output_line_hash(self):
        log = (
            "[gpureplay] rev=abc submit=1000 3840x2160 draws=46 ... resource-data=present\n"
            "  VS CB b=3 ... hash=cf3e4df9be3d0fd1 ...\n"       # a resource hash must NOT win
            "[gpureplay] output=3840x2160 bytes=33177600 hash=670576577254c2e8 expected_hash=0\n"
        )
        self.assertEqual(REGRESS.parse_output_hash(log), "670576577254c2e8")

    def test_last_output_line_wins(self):
        log = ("[gpureplay] output=8x8 hash=1111111111111111\n"
               "[gpureplay] output=8x8 hash=2222222222222222\n")
        self.assertEqual(REGRESS.parse_output_hash(log), "2222222222222222")

    def test_none_when_absent(self):
        self.assertIsNone(REGRESS.parse_output_hash("[gpureplay] rev=abc inspect only, no render\n"))

    def test_lowercased(self):
        self.assertEqual(REGRESS.parse_output_hash("output=1x1 hash=ABCDEF0123456789"),
                         "abcdef0123456789")


class ClassifyCheckTests(unittest.TestCase):
    def test_all_match(self):
        cur = {"a.prgcap": "dead", "b.prgcap": "beef"}
        reg, err, new, missing = REGRESS.classify_check(cur, dict(cur))
        self.assertEqual((reg, err, new, missing), ([], [], [], []))

    def test_regression_detected(self):
        reg, err, new, missing = REGRESS.classify_check(
            {"a.prgcap": "NEWHASH"}, {"a.prgcap": "oldhash"})
        self.assertEqual(reg, [("a.prgcap", "oldhash", "NEWHASH")])
        self.assertEqual((err, new, missing), ([], [], []))

    def test_error_capsule_is_a_failure_not_silently_ok(self):
        reg, err, new, missing = REGRESS.classify_check({"a.prgcap": None}, {"a.prgcap": "x"})
        self.assertEqual(err, ["a.prgcap"])
        self.assertEqual(reg, [])  # a None hash must not be miscompared as a regression

    def test_new_and_missing_are_informational(self):
        reg, err, new, missing = REGRESS.classify_check(
            {"b.prgcap": "hb"}, {"a.prgcap": "ha"})
        self.assertEqual(reg, [])
        self.assertEqual(err, [])
        self.assertEqual(new, ["b.prgcap"])       # present now, not baselined
        self.assertEqual(missing, ["a.prgcap"])   # baselined, absent from corpus


if __name__ == "__main__":
    unittest.main()
