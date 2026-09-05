#!/usr/bin/env python3
"""A zero-exit recorder must not certify an empty, failed, or unrelated decode."""
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import doctor
import renderdoc_inspect


class SampleVerdictTests(unittest.TestCase):
    def verdict(self, text, record_rc=0, decode_rc=0):
        return doctor.sample_verdict(
            {"returncode": record_rc}, {"returncode": decode_rc, "stdout": text},
            ["sched:sched_switch", "sched:sched_wakeup"])

    def test_both_events_required(self):
        result = self.verdict("sched:sched_switch:\nsched:sched_wakeup:\nsched:sched_switch:\n")
        self.assertEqual(result["status"], "READY")
        self.assertEqual(result["samples"], {"sched:sched_switch": 2, "sched:sched_wakeup": 1})
        self.assertEqual(self.verdict("sched:sched_switch:\n")["status"], "UNAVAILABLE")

    def test_empty_success_is_unavailable(self):
        self.assertEqual(self.verdict("")["status"], "UNAVAILABLE")

    def test_error_mentions_are_not_samples(self):
        self.assertEqual(self.verdict(
            "event syntax error: sched:sched_switch\nmissing sched:sched_wakeup\n"
        )["status"], "UNAVAILABLE")

    def test_failed_commands_do_not_certify_stale_samples(self):
        text = "sched:sched_switch:\nsched:sched_wakeup:\n"
        for rec, dec in ((1, 0), (0, 1), (None, 0)):
            self.assertEqual(self.verdict(text, rec, dec)["status"], "UNAVAILABLE")

    def test_user_and_kernel_events_stay_distinct(self):
        self.assertEqual(doctor.events_in("cpu-clock:u:\ncpu-clock:k:\n"),
                         {"cpu-clock:u": 1, "cpu-clock:k": 1})


class ReplayWrapperTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=".")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.capture = self.root / "control.rdc"
        self.capture.write_bytes(b"fixture, not a real RDC")
        self.output = self.root / "replay"

    def invoke(self, report=None, returncode=0):
        def replay(*args, **kwargs):
            if report is not None:
                (self.output / "replay.json").write_text(report)
            return subprocess.CompletedProcess(args[0], returncode, "", "")

        argv = ["renderdoc_inspect.py", str(self.capture), "--output", str(self.output)]
        with patch("sys.argv", argv), patch("shutil.which", return_value="qrenderdoc"), \
                patch("subprocess.run", side_effect=replay) as process, \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            try:
                status = renderdoc_inspect.main()
            except SystemExit as exc:
                status = exc.code
        return status, process.call_count

    def test_success_requires_report(self):
        self.assertEqual(self.invoke(json.dumps({"status": "REPLAYED"})), (0, 1))

    def test_empty_success_fails(self):
        self.assertEqual(self.invoke(), (1, 1))

    def test_failed_process_cannot_certify_report(self):
        self.assertEqual(self.invoke(json.dumps({"status": "REPLAYED"}), 1), (1, 1))

    def test_malformed_report_fails(self):
        self.assertEqual(self.invoke("not JSON"), (1, 1))

    def test_failed_report_fails(self):
        self.assertEqual(self.invoke(json.dumps({"status": "FAILED"})), (1, 1))

    def test_stale_output_is_rejected_before_replay(self):
        self.output.mkdir()
        (self.output / "replay.json").write_text('{"status":"REPLAYED"}')
        self.assertEqual(self.invoke(), (2, 0))

    def test_missing_capture_is_rejected_before_replay(self):
        self.capture.unlink()
        self.assertEqual(self.invoke(), (2, 0))


class IndexControlTests(unittest.TestCase):
    def check(self, data, draws=5, copies=1):
        buffers = [SimpleNamespace(resourceId=i, length=len(data)) for i in range(copies)]
        return renderdoc_inspect.index_control_matches([None] * draws, buffers,
                                                       lambda *_: data)

    def expected(self):
        # Independent byte construction, not the production struct.pack expression.
        return b"".join(i.to_bytes(4, "little") for i in range(6)) + b"\xff" * 1000

    def test_exact_contents_pass(self):
        self.assertEqual(self.check(self.expected()), ["0"])

    def test_zero_buffer_is_not_a_control(self):
        with self.assertRaisesRegex(RuntimeError, "exact SSBO matches=0"):
            self.check(bytes(1024))

    def test_untouched_sentinels_are_checked(self):
        with self.assertRaisesRegex(RuntimeError, "exact SSBO matches=0"):
            self.check(self.expected()[:-1] + b"\x00")

    def test_draw_count_is_checked(self):
        with self.assertRaisesRegex(RuntimeError, "draws=4"):
            self.check(self.expected(), draws=4)

    def test_ambiguous_buffers_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "exact SSBO matches=2"):
            self.check(self.expected(), copies=2)


if __name__ == "__main__":
    unittest.main()
