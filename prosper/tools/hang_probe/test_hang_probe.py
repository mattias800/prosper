#!/usr/bin/env python3
import importlib.util
import os
import subprocess
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = importlib.util.spec_from_file_location("prosper_hang_probe", os.path.join(HERE, "hang_probe.py"))
HANG_PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HANG_PROBE)


class ClassificationTests(unittest.TestCase):
    def test_active_gpu_native_frame_is_running(self):
        text = ("0x0000000000442f85 in prosper::gpu::build_stage_table(prosper::gpu::GpuState const&) ()\n"
                "guest-bt: loaded\n#0 0x442f85 in ??? ()\n")
        self.assertEqual(HANG_PROBE.classify(text),
                         ("RUNNING", text.splitlines()[0]))

    def test_active_live_frontend_native_frame_is_running(self):
        text = ("0x0000000000652319 in prosper::frontend::register_live_renderer()::{lambda()#1}::operator()() const ()\n"
                "guest-bt: loaded\n#0 0x652319 in ??? ()\n")
        self.assertEqual(HANG_PROBE.classify(text),
                         ("RUNNING", text.splitlines()[0]))

    def test_active_libc_native_frame_is_running(self):
        text = "0x00007f00 in _int_free_chunk () from /lib64/libc.so.6\n#0 0x7f00 in ??? ()\n"
        self.assertEqual(HANG_PROBE.classify(text),
                         ("RUNNING", text.splitlines()[0]))

    def test_active_unsymbolicated_guest_frame_is_running(self):
        text = "0x0000000410f4bb02 in ?? ()\n#0 0x410f4bb02 in ??? ()\n"
        self.assertEqual(HANG_PROBE.classify(text),
                         ("RUNNING", text.splitlines()[0]))

    def test_native_blocking_frame_remains_unknown(self):
        text = "0x00007f00 in pthread_cond_wait@@GLIBC_2.3.2 ()\n#0 0x7f00 in ??? ()\n"
        self.assertEqual(HANG_PROBE.classify(text),
                         ("UNKNOWN", text.splitlines()[0]))

    def test_guest_wait_takes_precedence_over_lower_running_frame(self):
        text = "#0 k_sema_wait\n#1 prosper::gpu::execute_ordered\n"
        self.assertEqual(HANG_PROBE.classify(text), ("BLOCKED", "#0 k_sema_wait"))

    def test_unknown_without_native_frame_has_empty_evidence(self):
        self.assertEqual(HANG_PROBE.classify("#0 0x7f00 in ??? ()\n"), ("UNKNOWN", ""))


class GuestBtResultTests(unittest.TestCase):
    @mock.patch.object(HANG_PROBE.subprocess, "run")
    def test_failed_guest_bt_is_unknown_with_stderr(self, run):
        run.return_value = subprocess.CompletedProcess(
            ["guest_bt"], 1, stdout="", stderr="ptrace: Operation not permitted\n")
        self.assertEqual(HANG_PROBE.run_guest_bt(["guest_bt"], 5),
                         ("UNKNOWN", "(guest_bt failed, exit 1: ptrace: Operation not permitted)", ""))

    @mock.patch.object(HANG_PROBE.subprocess, "run")
    def test_successful_guest_bt_is_classified(self, run):
        run.return_value = subprocess.CompletedProcess(
            ["guest_bt"], 0, stdout="#0 k_sema_wait\n", stderr="")
        self.assertEqual(HANG_PROBE.run_guest_bt(["guest_bt"], 5),
                         ("BLOCKED", "#0 k_sema_wait", "#0 k_sema_wait\n"))


class ExitCodeTests(unittest.TestCase):
    def counts(self, **updates):
        counts = {"BLOCKED": 0, "RUNNING": 0, "DEAD": 0, "UNKNOWN": 0}
        counts.update(updates)
        return counts

    def test_clean_classified_runs_succeed(self):
        self.assertEqual(HANG_PROBE.result_exit_code(self.counts(RUNNING=3)), 0)

    def test_reproduced_hang_is_bisect_failure(self):
        self.assertEqual(HANG_PROBE.result_exit_code(self.counts(BLOCKED=1)), 1)

    def test_unknown_or_dead_runs_are_inconclusive(self):
        self.assertEqual(HANG_PROBE.result_exit_code(self.counts(UNKNOWN=1)), 2)
        self.assertEqual(HANG_PROBE.result_exit_code(self.counts(DEAD=1)), 2)


if __name__ == "__main__":
    unittest.main()
