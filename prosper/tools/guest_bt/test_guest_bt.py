#!/usr/bin/env python3
import importlib.util
import os
import subprocess
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = importlib.util.spec_from_file_location("prosper_guest_bt", os.path.join(HERE, "guest_bt.py"))
GUEST_BT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GUEST_BT)


class GdbStatusTests(unittest.TestCase):
    @mock.patch.object(GUEST_BT.subprocess, "run")
    def test_propagates_gdb_failure_status(self, run):
        run.return_value = subprocess.CompletedProcess(["gdb"], 7)
        self.assertEqual(GUEST_BT.run_gdb("gdb", 123, "plugin.py", "guest-bt 1", {}), 7)

    @mock.patch.object(GUEST_BT.subprocess, "run")
    def test_success_returns_zero(self, run):
        run.return_value = subprocess.CompletedProcess(["gdb"], 0)
        self.assertEqual(GUEST_BT.run_gdb("gdb", 123, "plugin.py", "guest-bt 1", {}), 0)


if __name__ == "__main__":
    unittest.main()
