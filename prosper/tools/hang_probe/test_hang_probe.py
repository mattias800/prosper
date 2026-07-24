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
