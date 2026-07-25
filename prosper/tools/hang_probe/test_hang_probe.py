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
        text = ("guest-bt-native: 0x0000000000442f85 in prosper::gpu::build_stage_table\n"
                "guest-bt: loaded\n#0 0x442f85 in ??? ()\n")
        self.assertEqual(HANG_PROBE.classify(text),
                         ("RUNNING", "0x0000000000442f85 in prosper::gpu::build_stage_table"))

    def test_active_live_frontend_native_frame_is_running(self):
        text = ("guest-bt-native: 0x0000000000652319 in prosper::frontend::register_live_renderer()::{lambda()#1}::operator()() const\n"
                "guest-bt: loaded\n#0 0x652319 in ??? ()\n")
        self.assertEqual(HANG_PROBE.classify(text),
                         ("RUNNING", text.splitlines()[0].removeprefix("guest-bt-native: ")))

    def test_active_libc_native_frame_is_running(self):
        text = "guest-bt-native: 0x00007f00 in _int_free_chunk\n#0 0x7f00 in ??? ()\n"
        self.assertEqual(HANG_PROBE.classify(text),
                         ("RUNNING", "0x00007f00 in _int_free_chunk"))

    def test_active_unsymbolicated_guest_frame_is_running(self):
        text = "guest-bt-native: 0x0000000410f4bb02 in ??\n#0 0x410f4bb02 in ??? ()\n"
        self.assertEqual(HANG_PROBE.classify(text),
                         ("RUNNING", "0x0000000410f4bb02 in ??"))

    def test_native_blocking_frame_remains_unknown(self):
        text = ("guest-bt-native: 0x00007f00 in pthread_cond_wait@@GLIBC_2.3.2\n"
                "#0 0x7f00 in ??? ()\n#1 prosper::gpu::present\n")
        self.assertEqual(HANG_PROBE.classify(text),
                         ("UNKNOWN", "0x00007f00 in pthread_cond_wait@@GLIBC_2.3.2"))

    def test_decorated_glibc_waits_remain_unknown(self):
        for name in ("__GI___clock_nanosleep", "__GI___poll", "__GI___select"):
            with self.subTest(name=name):
                frame = f"0x00007f00 in {name}"
                text = f"guest-bt-native: {frame}\n#1 prosper::gpu::execute_ordered\n"
                self.assertEqual(HANG_PROBE.classify(text), ("UNKNOWN", frame))

    def test_native_blocking_operations_remain_unknown(self):
        names = ("__futex_abstimed_wait_common64", "pthread_cond_timedwait",
                 "pthread_mutex_lock", "pthread_rwlock_rdlock", "sem_clockwait")
        for name in names:
            with self.subTest(name=name):
                frame = f"0x00007f00 in {name}"
                self.assertEqual(HANG_PROBE.classify(f"guest-bt-native: {frame}\n"),
                                 ("UNKNOWN", frame))

    def test_active_native_operations_are_running(self):
        names = ("futex_wake", "pthread_cond_signal", "pthread_mutex_unlock",
                 "pthread_mutex_trylock", "pthread_rwlock_unlock",
                 "prosper::gpu::select_shader")
        for name in names:
            with self.subTest(name=name):
                frame = f"0x0000000000442f85 in {name}"
                self.assertEqual(HANG_PROBE.classify(f"guest-bt-native: {frame}\n"),
                                 ("RUNNING", frame))

    def test_attach_frame_before_thread_switch_is_ignored(self):
        text = ("0x00007f00 in prosper::k_sema_wait()\n#1 prosper::gpu::present\n"
                "guest-bt-native: 0x0000000410f4bb02 in ??\n#0 0x410f4bb02 in ??? ()\n")
        self.assertEqual(HANG_PROBE.classify(text),
                         ("RUNNING", "0x0000000410f4bb02 in ??"))

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
