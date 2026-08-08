#!/usr/bin/env python3
"""Tests for stack_profile's frame classification.

These run everywhere: they exercise the parsing and classification against RECORDED gdb
output, so they need no gdb, no ptrace and no live process. That matters more than
convenience -- an end-to-end test needing ptrace would skip silently on any machine
without it, and a silently skipped test is indistinguishable from a passing one.

The fixture below is verbatim gdb 17.2 output for a purpose-built control whose three
threads block in three known functions. It is the input that broke the first version of
the tool, kept because the failure was invisible without a known-correct answer to
disagree with: the tool reported `__internal_syscall_cancel`, which is not wrong-looking.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import stack_profile as sp  # noqa: E402


# Verbatim `gdb -p <pid> -batch -ex 'thread apply all bt'`, glibc 2.4x, gdb 17.2.
REAL_GDB_OUTPUT = """
Thread 4 (Thread 0x7faee0f5b6c0 (LWP 910931) "known"):
#0  0x00007faee0fda312 in __syscall_cancel_arch () from /lib64/libc.so.6
#1  0x00007faee0fce52c in __internal_syscall_cancel () from /lib64/libc.so.6
#2  0x00007faee0fd4b1a in __futex_abstimed_wait_common () from /lib64/libc.so.6
#3  0x00007faee0fd1a51 in pthread_mutex_lock@@GLIBC_2.2.5 () from /lib64/libc.so.6
#4  0x00000000004004e4 in marlow_blocks_on_mutex () at known.c:10
#5  0x0000000000400560 in t_mutex (a=0x0) at known.c:14

Thread 3 (Thread 0x7faee075a6c0 (LWP 910932) "known"):
#0  0x00007faee0fda312 in __syscall_cancel_arch () from /lib64/libc.so.6
#1  0x00007faee0fce52c in __internal_syscall_cancel () from /lib64/libc.so.6
#2  0x00007faee0fce877 in __futex_abstimed_wait_cancelable64 () from /lib64/libc.so.6
#3  0x00007faee0fd10cc in pthread_cond_wait@@GLIBC_2.3.2 () from /lib64/libc.so.6
#4  0x000000000040050e in marlow_blocks_on_condvar () at known.c:11

Thread 2 (Thread 0x7faee0f596c0 (LWP 910933) "known"):
#0  0x00007faee0fda312 in __syscall_cancel_arch () from /lib64/libc.so.6
#1  0x00007faee0fce52c in __internal_syscall_cancel () from /lib64/libc.so.6
#2  0x00007faee0f8b8e7 in clock_nanosleep@@GLIBC_2.2.5 () from /lib64/libc.so.6
#3  0x00007faee0f90d63 in sleep () from /lib64/libc.so.6
#4  0x000000000040051f in marlow_blocks_on_sleep () at known.c:12
"""


class TestSymbolStripping(unittest.TestCase):
    def test_version_suffix_removed(self):
        # The bug: `$`-anchored patterns never matched a versioned symbol, so glibc
        # frames were classified as application code.
        self.assertEqual(sp.strip_symbol("pthread_cond_wait@@GLIBC_2.3.2"), "pthread_cond_wait")
        self.assertEqual(sp.strip_symbol("clock_nanosleep@GLIBC_2.2.5"), "clock_nanosleep")

    def test_unversioned_symbol_untouched(self):
        self.assertEqual(sp.strip_symbol("marlow_blocks_on_mutex"), "marlow_blocks_on_mutex")

    def test_versioned_plumbing_is_recognised_as_plumbing(self):
        # Directly guards the regression: this is the assertion that fails if the `$`
        # anchor is reintroduced without stripping.
        for sym in ("pthread_cond_wait@@GLIBC_2.3.2", "pthread_mutex_lock@@GLIBC_2.2.5",
                    "clock_nanosleep@@GLIBC_2.2.5"):
            self.assertIsNotNone(sp.PLUMBING.match(sp.strip_symbol(sym)),
                                 f"{sym} must classify as plumbing")


class TestBlockingSite(unittest.TestCase):
    """The tool's actual job: skip the wait machinery, name the caller."""

    def _sites(self, depth=1):
        threads, _, err = self._parse()
        self.assertIsNone(err)
        return {lwp: sp.blocking_site(frames, depth) for lwp, (_, frames) in threads.items()}

    def _parse(self):
        # Exercise the real parser by faking one gdb invocation.
        import subprocess
        real = subprocess.run

        class Fake:
            stdout, stderr, returncode = REAL_GDB_OUTPUT, "", 0

        subprocess.run = lambda *a, **k: Fake()
        try:
            return sp.gdb_stacks(1, "gdb", 10.0)
        finally:
            subprocess.run = real

    def test_all_three_known_sites_are_named(self):
        sites = set(self._sites().values())
        self.assertIn("marlow_blocks_on_mutex", sites)
        self.assertIn("marlow_blocks_on_condvar", sites)
        self.assertIn("marlow_blocks_on_sleep", sites)

    def test_mutex_and_condvar_are_distinguished(self):
        # /proc cannot do this: both threads are state S, wchan futex_do_wait, syscall 202.
        # Being able to separate them is the entire reason this tool exists.
        sites = self._sites()
        self.assertNotEqual(sites[910931], sites[910932])

    def test_no_libc_internal_is_ever_reported_as_the_site(self):
        # The original failure mode: a plausible-looking libc frame in place of the answer.
        for lwp, site in self._sites().items():
            self.assertFalse(site.startswith("__"),
                             f"lwp {lwp} reported libc internal {site!r} as the blocking site")

    def test_depth_includes_callers(self):
        self.assertEqual(self._sites(depth=2)[910931],
                         "marlow_blocks_on_mutex <- t_mutex")

    def test_thread_identity_is_parsed(self):
        threads, _, err = self._parse()
        self.assertIsNone(err)
        self.assertEqual(len(threads), 3)
        self.assertEqual(threads[910932][0], "known")


class TestFailureIsNotAResult(unittest.TestCase):
    """A sample that yields nothing must be a FAILURE, never 'nothing was blocked'."""

    def _run_with(self, stdout, stderr=""):
        import subprocess
        real = subprocess.run

        class Fake:
            returncode = 0

        Fake.stdout, Fake.stderr = stdout, stderr
        subprocess.run = lambda *a, **k: Fake()
        try:
            return sp.gdb_stacks(1, "gdb", 10.0)
        finally:
            subprocess.run = real

    def test_empty_output_reports_an_error(self):
        threads, _, err = self._run_with("")
        self.assertEqual(threads, {})
        self.assertIsNotNone(err, "empty gdb output must not be reported as a clean sample")

    def test_container_ptrace_denial_is_named_specifically(self):
        _, _, err = self._run_with("", "ptrace: Operation not permitted.")
        self.assertIsNotNone(err)
        # The message must state the *siting* rule, because the denial is otherwise silent:
        # no stacks is byte-identical to a process with nothing blocked.
        self.assertIn("container", err,
                      "the ptrace denial must tell the user which side to run gdb on")

    def test_dead_process_is_named(self):
        _, _, err = self._run_with("", "ptrace: No such process.")
        self.assertIsNotNone(err)
        self.assertIn("gone", err)


if __name__ == "__main__":
    unittest.main(verbosity=2)
