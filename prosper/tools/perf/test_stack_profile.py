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


class TestReviewFindings(unittest.TestCase):
    """Three cases the hand-built control could not produce, found by review of a live run.

    The control ran where gdb resolved libc and where no C++ wait wrappers appeared, so all
    three were structurally inexpressible in it -- the exact "positive control drawn from the
    same source as the null" shape CLAUDE.md warns about. The frame NAMES below are verbatim
    from stack_profile's own output on a live prosper-app (Blue Prince, 87 threads); the
    surrounding gdb line formatting is reconstructed, which is faithful because the classifier
    only ever sees the names.
    """

    def test_unresolved_frames_are_never_the_reported_site(self):
        # `set auto-solib-add off` yields `??` for unloaded libraries. The live run printed
        # `?? <- ??` as the blocking site for four threads.
        self.assertEqual(sp.blocking_site(["??", "??", "prosper::disk_worker_loop"], 2),
                         "prosper::disk_worker_loop")

    def test_stack_of_only_unresolved_frames_says_so(self):
        # Distinct from "all plumbing": here the TOOL failed for this thread, and reporting it
        # as a finding about the program would be a fabrication.
        self.assertEqual(sp.blocking_site(["??", "??", "??"], 3), sp.UNRESOLVED_ONLY)
        self.assertIsNotNone(sp.UNRESOLVED_ONLY)

    def test_cxx_condition_variable_wrappers_are_plumbing(self):
        # The live run reported this chain as GfxFlipThread's blocking SITE. It is the wait
        # machinery itself -- the tool named the thing it exists to skip.
        frames = [
            "std::__condvar::wait_until",
            "std::condition_variable::__wait_until_impl<std::chrono::duration<long, std::ratio<1l, 1000000000l> > >",
            "std::condition_variable::wait_until<std::chrono::duration<long, std::ratio<1l, 1000000000l> > >",
            "prosper::gfx_flip_wait_for_work",
        ]
        self.assertEqual(sp.blocking_site(frames, 1), "prosper::gfx_flip_wait_for_work")

    def test_cxx_sleep_wrappers_are_plumbing(self):
        frames = [
            "std::this_thread::sleep_for<long, std::ratio<1l, 1000000000l> >",
            "std::this_thread::sleep_until<std::chrono::_V2::steady_clock, std::chrono::duration<long, std::ratio<1l, 1000000000l> > >",
            "prosper::fmod_audio_out_tick",
        ]
        self.assertEqual(sp.blocking_site(frames, 1), "prosper::fmod_audio_out_tick")

    def test_gthread_wrappers_are_plumbing(self):
        # `__gthread_cond_wait` is the one that reads as already-covered but is not: the `_*`
        # prefix eats the underscores, and then `pthread_cond_\w+` needs a literal "pthread".
        for sym in ("__gthread_cond_wait", "__gthread_mutex_lock"):
            self.assertTrue(sp.is_plumbing(sym), f"{sym} must classify as plumbing")

    def test_a_real_application_frame_is_still_not_plumbing(self):
        # The paired negative: the widened filter must not swallow the answers.
        for sym in ("prosper::test::readback_persistent_color_target",
                    "prosper::k_wait_on_address",
                    "std::vector<unsigned char, std::allocator<unsigned char> >::assign<unsigned char const*, void>"):
            self.assertFalse(sp.is_plumbing(sym), f"{sym} must NOT be swallowed as plumbing")


class TestOutcomesThatAreNotFindings(unittest.TestCase):
    """Second review pass: three ways to have no named site, only ONE of which is a finding.

    Collapsing them shares a label that asserts something about the program in the two cases
    that support nothing. This is the tool's own central thesis -- "I did not measure" must
    never render as "there was nothing to measure" -- applied one level below where the first
    pass stopped.
    """

    def test_all_plumbing_is_a_real_finding_and_keeps_the_None_outcome(self):
        # This one IS a statement about the program: every frame resolved, and all were waits.
        self.assertIsNone(sp.blocking_site(["pthread_cond_wait", "sem_wait"], 2))

    def test_plumbing_plus_unresolved_is_not_a_pure_library_wait(self):
        # Frames genuinely did not resolve, so "pure library wait" is one possibility of several.
        self.assertEqual(sp.blocking_site(["pthread_cond_wait", "??", "??"], 2),
                         sp.UNRESOLVED_ONLY)

    def test_empty_stack_has_its_own_outcome(self):
        self.assertEqual(sp.blocking_site([], 2), sp.NO_FRAMES)

    def test_the_three_outcomes_are_all_distinct(self):
        outcomes = {
            sp.blocking_site(["pthread_cond_wait", "sem_wait"], 2),   # all plumbing -> None
            sp.blocking_site(["pthread_cond_wait", "??"], 2),          # -> UNRESOLVED_ONLY
            sp.blocking_site([], 2),                                   # -> NO_FRAMES
        }
        self.assertEqual(len(outcomes), 3, "the three no-site outcomes must not share a label")


class TestChainContiguity(unittest.TestCase):
    """`<-` reads as 'called by', so an elided middle frame asserts a false adjacency."""

    def test_unresolved_middle_frame_truncates_rather_than_eliding(self):
        site = sp.blocking_site(["real_answer", "??", "caller2"], 3)
        self.assertNotIn("real_answer <- caller2", site,
                         "eliding `??` claims real_answer is called by caller2, which is false")
        self.assertTrue(site.startswith("real_answer"))
        self.assertIn("...", site, "the truncation must be visible")

    def test_contiguous_named_callers_are_still_joined(self):
        self.assertEqual(sp.blocking_site(["a", "b", "c"], 3), "a <- b <- c")

    def test_depth_reached_before_the_gap_needs_no_marker(self):
        # Truncation by --depth is not a gap in the data, so it must not be marked as one.
        self.assertEqual(sp.blocking_site(["a", "b", "??"], 2), "a <- b")


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
