#!/usr/bin/env python3
"""Tests for compare_runs.py. Synthetic manifests, no title, no GPU.

Each refusal arm builds a pair that matches EVERYWHERE except one condition, so a refusal can only be
attributed to that condition. The matching pair is also asserted to be accepted: without it, a tool
that refuses everything would pass every refusal arm.
"""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_runs as cr  # noqa: E402


def run_record(**over):
    record = {
        "type": "run", "schema": 1, "title": "PPSA00000", "build_revision": "a" * 40,
        "input_route": "route.pad", "capture_mode": "wall_seconds", "seconds": 60.0, "every": 1,
        "requested": 60, "warmup_ms": 0, "warmup_submits": 0, "render_every": "",
        "render_every_for_ms": "", "render_scale": "", "render_target_dim": "",
        "render_resource_dim": "", "assertions": {"fps_overlay": False},
    }
    record.update(over)
    return record


def conditions_record(**over):
    record = {
        "type": "conditions", "schema": 1, "harness": "tools/screenshot",
        "present_path": "forced_readback", "os": "linux", "flip_pace_fps": "",
        "gpu_known": True, "gpu_vendor_id": 4098, "gpu_device_id": 29631,
        "gpu_driver_version": 99, "gpu_api_version": 4202496, "gpu_device_type": 2,
    }
    record.update(over)
    return record


def summary_record(**over):
    record = {
        "type": "summary", "schema": 1, "status": "ok", "frame_rate_measured": True,
        "typical_fps_measured": True, "typical_fps": 60.0, "distinct_fps": 59.0,
        "low_1pct_fps": 50.0, "interval_p99_ms": 20.0, "interval_p95_ms": 17.5,
        "active_fraction": 0.98,
    }
    record.update(over)
    return record


def manifest(run=None, conditions=None, summary=None):
    records = {"run": run_record(**(run or {})),
               "conditions": conditions_record(**(conditions or {})),
               "summary": summary_record(**(summary or {}))}
    return records


class Refusals(unittest.TestCase):
    def test_identical_conditions_are_comparable(self):
        self.assertEqual(cr.refusals(manifest(), manifest()), [])

    def test_build_revision_and_measured_fields_may_differ(self):
        candidate = manifest(run={"build_revision": "b" * 40}, summary={"typical_fps": 30.0})
        self.assertEqual(cr.refusals(manifest(), candidate), [])

    def test_each_condition_alone_refuses(self):
        cases = [
            ("run", {"title": "PPSA11111"}, "run.title"),
            ("run", {"input_route": "other.pad"}, "run.input_route"),
            ("run", {"seconds": 30.0}, "run.seconds"),
            ("run", {"render_scale": "2"}, "run.render_scale"),
            ("run", {"render_every": "4"}, "run.render_every"),
            ("run", {"assertions": {"fps_overlay": True}}, "run.fps_overlay"),
            ("conditions", {"harness": "prosper-app"}, "conditions.harness"),
            ("conditions", {"present_path": "gpu_present"}, "conditions.present_path"),
            ("conditions", {"os": "windows"}, "conditions.os"),
            ("conditions", {"flip_pace_fps": "60"}, "conditions.flip_pace_fps"),
            ("conditions", {"gpu_vendor_id": 4318}, "conditions.gpu_vendor_id"),
            ("conditions", {"gpu_device_id": 1}, "conditions.gpu_device_id"),
            ("conditions", {"gpu_driver_version": 100}, "conditions.gpu_driver_version"),
            ("conditions", {"gpu_api_version": 1}, "conditions.gpu_api_version"),
            ("conditions", {"gpu_device_type": 1}, "conditions.gpu_device_type"),
        ]
        for record, change, expected in cases:
            with self.subTest(expected):
                candidate = manifest(**{record: change})
                reasons = cr.refusals(manifest(), candidate)
                self.assertTrue(any(expected in r for r in reasons), reasons)

    def test_a_manifest_without_conditions_is_refused_and_says_why(self):
        old = manifest()
        del old["conditions"]
        reasons = cr.refusals(manifest(), old)
        self.assertTrue(any("predates the `conditions` record" in r for r in reasons), reasons)

    def test_unknown_gpu_is_refused_even_when_both_are_unknown(self):
        unknown = {"gpu_known": False, "gpu_vendor_id": None, "gpu_device_id": None,
                   "gpu_driver_version": None, "gpu_api_version": None, "gpu_device_type": None}
        reasons = cr.refusals(manifest(conditions=unknown), manifest(conditions=unknown))
        self.assertTrue(any("device is unknown" in r for r in reasons), reasons)

    def test_failed_or_empty_runs_are_refused(self):
        for change, expected in [
            ({"status": "GUEST-FAULT"}, "not 'ok'"),
            ({"frame_rate_measured": False}, "no frames were published"),
            ({"typical_fps_measured": False}, "fewer than two distinct frames"),
        ]:
            with self.subTest(expected):
                reasons = cr.refusals(manifest(), manifest(summary=change))
                self.assertTrue(any(expected in r for r in reasons), reasons)

    def test_a_mixed_window_is_refused(self):
        reasons = cr.refusals(manifest(), manifest(summary={"active_fraction": 0.62}))
        self.assertTrue(any("mixed two regimes" in r for r in reasons), reasons)

    def test_active_fraction_at_the_floor_is_accepted(self):
        self.assertEqual(cr.refusals(manifest(), manifest(summary={"active_fraction": 0.90})), [])


class Comparison(unittest.TestCase):
    def crossed(self, **summary):
        _, crossed = cr.compare(manifest(), manifest(summary=summary), 3.0, 5.0)
        return crossed

    def test_no_change_crosses_nothing(self):
        self.assertEqual(self.crossed(), [])

    def test_typical_drop_over_threshold_is_a_regression(self):
        self.assertEqual(len(self.crossed(typical_fps=58.0)), 1)   # -3.33%
        self.assertEqual(self.crossed(typical_fps=58.5), [])       # -2.5%

    def test_low_drop_over_threshold_is_a_regression(self):
        self.assertEqual(len(self.crossed(low_1pct_fps=47.0)), 1)  # -6%
        self.assertEqual(self.crossed(low_1pct_fps=48.0), [])      # -4%

    def test_an_improvement_is_not_a_regression(self):
        self.assertEqual(self.crossed(typical_fps=75.0, low_1pct_fps=70.0), [])

    def test_a_tail_that_is_null_on_one_side_is_not_compared(self):
        rows, crossed = cr.compare(manifest(), manifest(summary={"low_1pct_fps": None}), 3.0, 5.0)
        low = next(r for r in rows if r["key"] == "low_1pct_fps")
        self.assertIsNone(low["delta"])
        self.assertEqual(crossed, [])

    def test_p99_is_lower_is_better(self):
        rows, _ = cr.compare(manifest(), manifest(summary={"interval_p99_ms": 40.0}), 3.0, 5.0)
        p99 = next(r for r in rows if r["key"] == "interval_p99_ms")
        self.assertTrue(p99["delta"][2], "a longer p99 is worse")


class Cli(unittest.TestCase):
    def write(self, directory, name, records):
        path = os.path.join(directory, name)
        with open(path, "w", encoding="utf-8") as handle:
            for record in records.values():
                handle.write(json.dumps(record) + "\n")
        return path

    def run_main(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cr.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_exit_codes_and_that_a_refusal_prints_no_delta(self):
        with tempfile.TemporaryDirectory() as d:
            base = self.write(d, "base.jsonl", manifest())
            same = self.write(d, "same.jsonl", manifest())
            slow = self.write(d, "slow.jsonl", manifest(summary={"typical_fps": 40.0}))
            other_gpu = self.write(d, "gpu.jsonl", manifest(conditions={"gpu_device_id": 7}))

            code, out, _ = self.run_main(base, same)
            self.assertEqual(code, 0)
            self.assertIn("comparable", out)

            code, out, _ = self.run_main(base, slow)
            self.assertEqual(code, 1)
            self.assertIn("REGRESSION", out)

            code, out, err = self.run_main(base, other_gpu)
            self.assertEqual(code, 2)
            self.assertIn("REFUSED", err)
            self.assertNotIn("typical fps", out, "a refused pair must print no delta")

    def test_an_unreadable_or_malformed_manifest_is_a_refusal_not_a_pass(self):
        with tempfile.TemporaryDirectory() as d:
            base = self.write(d, "base.jsonl", manifest())
            bad = os.path.join(d, "bad.jsonl")
            with open(bad, "w", encoding="utf-8") as handle:
                handle.write("{not json\n")
            self.assertEqual(self.run_main(base, bad)[0], 2)
            self.assertEqual(self.run_main(base, os.path.join(d, "missing.jsonl"))[0], 2)

    def test_the_last_record_of_each_type_wins(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "twice.jsonl")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(summary_record(typical_fps=10.0)) + "\n")
                handle.write(json.dumps(summary_record(typical_fps=60.0)) + "\n")
            self.assertEqual(cr.load_manifest(path)["summary"]["typical_fps"], 60.0)


if __name__ == "__main__":
    unittest.main()
