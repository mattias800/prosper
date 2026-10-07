#!/usr/bin/env python3
"""Parser, selection and sampling-loop controls for the /proc scheduler probe.

The probe's loop is driven against a synthetic /proc tree through `collect`'s `proc_root`
parameter, so the identity-change and unreadable-process terminals are exercised rather than
described. Nothing here reads the real /proc or needs a running game.

The loop runs on a virtual clock (`SampleClock`), and every change to the synthetic tree happens
inside the collector's own pacing sleep, on the collector's thread, before a chosen sample. So no
arm depends on wall-clock timing or on a concurrent rename being atomic to a reader. The earlier
fixture used timer threads and `Path.replace`, and in CI's parallel test run it failed on macOS (the
timer missed the window) and on Windows (the rename raced the reader) (#4142).
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import schedstat_probe
from schedstat_probe import collect, read_thread, select_threads, stat_fields

HZ = 200.0
STEP_NS = round(1e9 / HZ)  # The collector's own tick, computed the way `collect` computes it.


class SampleClock:
    """Stands in for the `time` module inside `schedstat_probe` during one `collect` call.

    `monotonic_ns` advances by a fixed small read cost per call, so read brackets are ordered
    the way real ones are. `sleep` advances virtual time and then runs every action that has
    come due, on the caller's thread. Because the collector sleeps up to each tick before it
    reads, an action registered with `before_sample(k)` runs after sample k-1 has been read and
    before sample k is: samples 0..k-1 see the old tree, sample k onward sees the new one.

    It exposes only `monotonic_ns` and `sleep`. Any other use of `time` by the collector fails
    with AttributeError instead of silently reading the real clock.
    """

    READ_COST_NS = 1_000  # Far below STEP_NS, so a read never crosses into the next tick.
    CALL_BUDGET = 1_000_000  # A loop that stops advancing fails instead of spinning forever.

    def __init__(self):
        self.now = 10**12  # Arbitrary; nothing may assume the clock starts at zero.
        self.start = None  # The first reading, which is `collect`'s own start time.
        self.calls = 0
        self.pending = []

    def monotonic_ns(self):
        self.calls += 1
        if self.calls > self.CALL_BUDGET:
            raise RuntimeError('collector stopped advancing the virtual clock')
        value = self.now
        if self.start is None:
            self.start = value
        self.now += self.READ_COST_NS
        return value

    def sleep(self, seconds):
        if seconds < 0:
            raise ValueError('negative sleep')
        self.now += round(seconds * 1e9)
        due = [entry for entry in self.pending if self.start + entry[0] * STEP_NS <= self.now]
        self.pending = [entry for entry in self.pending if entry not in due]
        for _, action in sorted(due, key=lambda entry: entry[0]):
            action()

    def before_sample(self, index, action):
        # Sample k's tick is start + k * STEP_NS, where start is `collect`'s first clock read.
        # Sample 0 is read before any sleep, so it cannot be preceded.
        if index < 1:
            raise ValueError('an action can only precede sample 1 or later')
        self.pending.append((index, action))


def stat_text(pid, comm, state, starttime):
    # /proc stat after the parenthesized comm: state is field 0 and starttime field 19.
    return f'{pid} ({comm}) {state} ' + ' '.join(['0'] * 18) + f' {starttime}\n'


def make_thread(task_root, tid, comm, *, state='S', starttime=4242,
                runtime=1000, runnable=200, switches=3, wchan='futex_wait'):
    task = task_root / str(tid)
    task.mkdir(parents=True)
    (task / 'comm').write_text(comm + '\n')
    (task / 'stat').write_text(stat_text(tid, comm, state, starttime))
    (task / 'schedstat').write_text(f'{runtime} {runnable} {switches}\n')
    (task / 'wchan').write_text(wchan)
    return task


class StatFieldsTests(unittest.TestCase):
    def test_parenthesized_command_with_spaces_and_parens_is_skipped(self):
        # The comm itself contains ") S " followed by more text, so a parser that split at the
        # first ')' instead of the last would read state 'S' and a shifted starttime.
        raw = stat_text(7, 'a) S 1 (b c', 'R', 12345)
        self.assertEqual(stat_fields(raw), ('R', 12345))

    def test_malformed_and_short_stat_refuse(self):
        with self.assertRaisesRegex(ValueError, 'malformed'):
            stat_fields('7 no-close-paren R 0 0')
        with self.assertRaisesRegex(ValueError, 'short'):
            stat_fields('7 (name) R 0 0 0')


class ThreadSelectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / '1234'
        (self.root / 'task').mkdir(parents=True)

    def test_read_thread_reports_counters_for_the_wanted_name_only(self):
        make_thread(self.root / 'task', 11, 'prosper-app', runtime=7, runnable=5, switches=2)
        row = read_thread(self.root, 11, 'prosper-app')
        self.assertEqual((row['tid'], row['starttime'], row['state']), (11, 4242, 'S'))
        self.assertEqual((row['runtime_ns'], row['runnable_ns'], row['switches']), (7, 5, 2))
        self.assertEqual(row['wchan'], 'futex_wait')
        # A TID that has been recycled under a different comm is not silently sampled.
        self.assertIsNone(read_thread(self.root, 11, 'other-app'))

    def test_running_thread_reports_no_wchan(self):
        make_thread(self.root / 'task', 12, 'prosper-app', state='R', wchan='stale')
        self.assertEqual(read_thread(self.root, 12, 'prosper-app')['wchan'], '')

    def test_short_schedstat_refuses_rather_than_reporting_partial_counters(self):
        task = make_thread(self.root / 'task', 13, 'prosper-app')
        (task / 'schedstat').write_text('1000 200\n')
        with self.assertRaisesRegex(ValueError, 'short /proc schedstat'):
            read_thread(self.root, 13, 'prosper-app')

    def test_select_threads_matches_by_comm_and_skips_unreadable_entries(self):
        make_thread(self.root / 'task', 21, 'prosper-app')
        make_thread(self.root / 'task', 9, 'prosper-app')
        make_thread(self.root / 'task', 22, 'gpu-worker')
        (self.root / 'task' / '23').mkdir()  # exited between listdir and read: no comm file
        self.assertEqual(select_threads(self.root, 'prosper-app'), [9, 21])


class CollectLoopTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.proc = Path(self.tmp.name)
        self.root = self.proc / '1234'
        (self.root / 'task').mkdir(parents=True)
        self.root.joinpath('stat').write_text(stat_text(1234, 'prosper-app', 'S', 99))
        make_thread(self.root / 'task', 11, 'prosper-app')
        make_thread(self.root / 'task', 12, 'gpu-worker')
        self.clock = SampleClock()
        patcher = mock.patch.object(schedstat_probe, 'time', self.clock)
        patcher.start()
        self.addCleanup(patcher.stop)

    def collect(self, seconds):
        return collect(1234, 'prosper-app', seconds, HZ, proc_root=self.proc)

    def test_completed_run_records_read_brackets_for_the_selected_threads(self):
        report = self.collect(0.1)
        self.assertEqual(report['terminal'], 'completed')
        self.assertEqual(report['candidate_tids'], [11])
        # 0.1 s at 200 Hz is 20 ticks, and the stop check runs before each sleep, so the loop
        # also takes the sample whose tick is the stop time itself: exactly 21 on the virtual
        # clock. A loop that stopped pacing takes tens of thousands; one that stopped early fewer.
        self.assertEqual(report['sample_count'], 21)
        self.assertEqual(report['overruns'], 0)
        self.assertEqual(report['matched_thread_rows'], report['sample_count'])
        self.assertEqual(report['read_failures'], 0)
        previous_after = None
        for row in report['samples']:
            self.assertLessEqual(row['before_ns'], row['after_ns'])
            if previous_after is not None:
                self.assertGreaterEqual(row['before_ns'], previous_after)
            previous_after = row['after_ns']
            self.assertEqual([t['tid'] for t in row['threads']], [11])

    def test_process_identity_change_stops_the_run_instead_of_mixing_two_processes(self):
        # The same PID now names a process with a different starttime: a PID reuse.
        self.clock.before_sample(5, lambda: self.root.joinpath('stat').write_text(
            stat_text(1234, 'prosper-app', 'S', 100)))
        report = self.collect(5.0)
        self.assertEqual(report['terminal'], 'process identity changed')
        # Exactly the five samples taken before the swap, and none from the new process.
        self.assertEqual(report['sample_count'], 5)
        self.assertEqual(report['matched_thread_rows'], 5)

    def test_unreadable_process_stops_the_run(self):
        self.clock.before_sample(5, self.root.joinpath('stat').unlink)
        report = self.collect(5.0)
        self.assertEqual(report['terminal'], 'process exited or became unreadable')
        self.assertEqual(report['sample_count'], 5)

    def test_thread_read_failure_is_counted_not_swallowed_into_a_zero_sample(self):
        task = self.root / 'task' / '11'
        self.clock.before_sample(5, lambda: (task / 'schedstat').write_text('1000 200\n'))
        report = self.collect(0.2)
        self.assertEqual(report['terminal'], 'completed')
        # Samples 0-4 read the thread; every sample from index 5 on fails its one thread read,
        # and each failure is counted and leaves that sample with no row rather than a zero one.
        self.assertGreater(report['sample_count'], 5)
        self.assertEqual(report['read_failures'], report['sample_count'] - 5)
        self.assertEqual(report['matched_thread_rows'], 5)
        self.assertEqual([len(row['threads']) for row in report['samples']],
                         [1] * 5 + [0] * (report['sample_count'] - 5))

    def test_an_overrun_resynchronises_the_tick_instead_of_bursting_to_catch_up(self):
        # Sample 5 is read three ticks late (the host stalled). That is one overrun, and the loop
        # restarts its schedule from the late read; a loop that kept the old schedule would take
        # the next samples back-to-back without sleeping and count each of them as an overrun.
        def stall():
            self.clock.now += 3 * STEP_NS

        self.clock.before_sample(5, stall)
        report = self.collect(0.2)
        self.assertEqual(report['terminal'], 'completed')
        self.assertEqual(report['overruns'], 1)
        gaps = [later['before_ns'] - earlier['after_ns']
                for earlier, later in zip(report['samples'], report['samples'][1:])]
        # The late sample becomes the new schedule origin: the next read follows at once (its tick
        # is that late read), and from there the loop sleeps a full tick again instead of bursting.
        self.assertLess(gaps[5], STEP_NS // 2)
        self.assertGreater(gaps[6], STEP_NS // 2)

    def test_sampling_rate_that_rounds_the_interval_to_zero_refuses(self):
        with self.assertRaisesRegex(ValueError, 'rounded to zero'):
            collect(1234, 'prosper-app', 0.1, 2e9, proc_root=self.proc)


if __name__ == '__main__':
    unittest.main()
