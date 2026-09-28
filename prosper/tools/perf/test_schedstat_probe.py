#!/usr/bin/env python3
"""Parser, selection and sampling-loop controls for the /proc scheduler probe.

The probe's loop is driven against a synthetic /proc tree through `collect`'s `proc_root`
parameter, so the identity-change and unreadable-process terminals are exercised rather than
described. Nothing here reads the real /proc or needs a running game.
"""

import tempfile
import threading
import time
import unittest
from pathlib import Path

from schedstat_probe import collect, read_thread, select_threads, stat_fields


def replace_atomically(path, text):
    # A plain write_text truncates first, so a concurrent reader can observe an empty file and
    # the probe would report "unreadable" where the test means "identity changed". Real /proc
    # reads never see a torn file; rename gives the synthetic tree the same property.
    temporary = path.with_name(path.name + '.new')
    temporary.write_text(text)
    temporary.replace(path)


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
        raw = stat_text(7, 'name with (spaces)', 'R', 12345)
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

    def after(self, delay, action):
        thread = threading.Thread(target=lambda: (time.sleep(delay), action()), daemon=True)
        thread.start()
        self.addCleanup(thread.join)

    def test_completed_run_records_read_brackets_for_the_selected_threads(self):
        report = collect(1234, 'prosper-app', 0.1, 200.0, proc_root=self.proc)
        self.assertEqual(report['terminal'], 'completed')
        self.assertEqual(report['candidate_tids'], [11])
        self.assertGreater(report['sample_count'], 0)
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
        self.after(0.05, lambda: replace_atomically(
            self.root / 'stat', stat_text(1234, 'prosper-app', 'S', 100)))
        report = collect(1234, 'prosper-app', 5.0, 200.0, proc_root=self.proc)
        self.assertEqual(report['terminal'], 'process identity changed')

    def test_unreadable_process_stops_the_run(self):
        self.after(0.05, lambda: self.root.joinpath('stat').unlink())
        report = collect(1234, 'prosper-app', 5.0, 200.0, proc_root=self.proc)
        self.assertEqual(report['terminal'], 'process exited or became unreadable')

    def test_thread_read_failure_is_counted_not_swallowed_into_a_zero_sample(self):
        task = self.root / 'task' / '11'
        self.after(0.02, lambda: replace_atomically(task / 'schedstat', '1000 200\n'))
        report = collect(1234, 'prosper-app', 0.2, 200.0, proc_root=self.proc)
        self.assertEqual(report['terminal'], 'completed')
        self.assertGreater(report['read_failures'], 0)
        self.assertLess(report['matched_thread_rows'], report['sample_count'])

    def test_sampling_rate_that_rounds_the_interval_to_zero_refuses(self):
        with self.assertRaisesRegex(ValueError, 'rounded to zero'):
            collect(1234, 'prosper-app', 0.1, 2e9, proc_root=self.proc)


if __name__ == '__main__':
    unittest.main()
