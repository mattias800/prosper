#!/usr/bin/env python3
"""Boundary and refusal controls for perf_f8_gap's interval join."""

import json
import tempfile
import unittest
from pathlib import Path

from perf_f8_gap import iter_samples, load_spans, summarize, zone


TRIGGER = 100_000_000_000


def write_capture(path, spans=((0, 1_000_000_000), (2_000_000_000, 3_000_000_000)),
                  dropped=0):
    records = [{'type': 'header', 'trigger_monotonic_ns': TRIGGER}]
    records.extend({'type': 'renderer', 'span_start_t_ns': start, 't_ns': end}
                   for start, end in spans)
    records.append({'type': 'footer', 'complete': True,
                    'renderer_records': len(spans), 'renderer_dropped': dropped,
                    'compute_dropped': 0})
    path.write_text(''.join(json.dumps(r) + '\n' for r in records))


def flat(pid, tid, time, symbol='work', period=10, dso='/tmp/prosper-app'):
    return f'prosper-app {pid}/{tid} {time}: {period} cpu-clock:u: 1234 {symbol} ({dso})\n'


class PerfF8GapTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.f8 = Path(self.tmp.name) / 'capture.prperf'
        self.perf = Path(self.tmp.name) / 'perf.txt'
        write_capture(self.f8)

    def test_full_window_separates_span_gap_and_other_cpu_samples(self):
        self.perf.write_text(''.join((flat(42, 43, '100.100000000', 'renderer'),
                                      flat(42, 44, '101.500000000', 'worker'),
                                      flat(42, 43, '102.500000000', 'renderer'),
                                      flat(42, 43, '102.900000000', 'renderer'))))
        result = summarize(self.f8, self.perf)
        self.assertEqual(result['pid'], 42)
        self.assertEqual(result['f8_spans'], 2)
        self.assertEqual(result['zones']['renderer-span']['samples'], 3)
        self.assertEqual(result['zones']['between-spans']['samples'], 1)
        self.assertEqual(result['zones']['between-spans']['sampled_cpu_clock_ms'], 0.00001)
        self.assertEqual(result['zones']['between-spans']['top_tids'], [('44', 1)])

    def test_partial_overlap_and_wrong_clock_refuse_instead_of_reporting_percentages(self):
        for lines in ((flat(42, 43, '102.400000000'), flat(42, 43, '102.900000000')),
                      (flat(42, 43, '200.100000000'), flat(42, 44, '201.500000000'),
                       flat(42, 43, '202.900000000'))):
            with self.subTest(lines=lines):
                self.perf.write_text(''.join(lines))
                with self.assertRaisesRegex(ValueError, 'do not cover the F8 detail interval'):
                    summarize(self.f8, self.perf)

    def test_dropped_or_overlapping_f8_records_refuse(self):
        write_capture(self.f8, dropped=1)
        with self.assertRaisesRegex(ValueError, 'dropped detail records'):
            load_spans(self.f8)
        write_capture(self.f8, spans=((0, 2_100_000_000), (2_000_000_000, 3_000_000_000)))
        with self.assertRaisesRegex(ValueError, 'nonoverlapping renderer spans'):
            load_spans(self.f8)

    def test_mixed_process_export_refuses(self):
        self.perf.write_text(''.join((flat(42, 43, '100.100000000'),
                                      flat(99, 99, '101.500000000'),
                                      flat(42, 43, '102.900000000'))))
        with self.assertRaisesRegex(ValueError, 'multiple processes'):
            summarize(self.f8, self.perf)

    def test_load_spans_accepts_touching_renderer_spans(self):
        # Two submits that abut leave a zero-length gap. That is an observation about the
        # capture, not a malformed record, and `schedstat_f8.join`'s null-coverage branch exists
        # precisely to handle it — so tightening this guard from `>` to `>=` would both discard a
        # valid capture and strand that branch. The decision is pinned here rather than only
        # asserted in a comment.
        #
        # Scope, deliberately: this asserts `load_spans`, NOT `summarize`. A capture whose spans
        # all touch is still refused further down for having no `between-spans` samples, which is
        # correct — there is no gap to compare against. `schedstat_f8.join` is the consumer that
        # uses these spans.
        write_capture(self.f8, spans=((0, 1_000_000_000), (1_000_000_000, 2_000_000_000)))
        _, spans = load_spans(self.f8)
        self.assertEqual(spans, [(TRIGGER, TRIGGER + 1_000_000_000),
                                 (TRIGGER + 1_000_000_000, TRIGGER + 2_000_000_000)])
        with self.assertRaisesRegex(ValueError, 'lacks samples in one of the two comparison'):
            self.perf.write_text(''.join((flat(42, 43, '100.100000000'),
                                          flat(42, 43, '101.900000000'))))
            summarize(self.f8, self.perf)

    def test_missing_renderer_start_refuses(self):
        records = [json.loads(line) for line in self.f8.read_text().splitlines()]
        del records[1]['span_start_t_ns']
        self.f8.write_text(''.join(json.dumps(r) + '\n' for r in records))
        with self.assertRaisesRegex(ValueError, 'lacks two nonnegative integer endpoints'):
            load_spans(self.f8)

    def test_call_graph_export_preserves_leaf_and_missing_stack(self):
        self.perf.write_text('[RAGE] RenderTh 42/43 100.000000001: 100 cpu-clock:u:\n'
                             '\t7f00 draw+0x1 (/tmp/prosper-app)\n\n'
                             'worker 42/44 101.5: 101 cpu-clock:u:\n\n')
        samples = list(iter_samples(self.perf))
        self.assertEqual([sample['time_ns'] for sample in samples],
                         [100_000_000_001, 101_500_000_000])
        self.assertEqual(samples[0]['frames'][0]['symbol'], 'draw+0x1')
        self.assertEqual(samples[1]['frames'], [])

    def test_recording_of_another_frontend_refuses_instead_of_reporting_no_app_work(self):
        # Under `tools/screenshot` or `boot_trace` every sample would otherwise fall into
        # `no_named_app_frame`, leaving three empty `top_app_frames` that read as "the
        # application did nothing" rather than "this reader was given the wrong binary name".
        lines = ''.join((flat(42, 43, '100.100000000', dso='/opt/screenshot'),
                         flat(42, 44, '101.500000000', dso='/opt/screenshot'),
                         flat(42, 43, '102.900000000', dso='/opt/screenshot')))
        self.perf.write_text(lines)
        with self.assertRaisesRegex(ValueError, "no sample in any zone carries a named frame"):
            summarize(self.f8, self.perf)
        named = summarize(self.f8, self.perf, 'screenshot')
        self.assertEqual(named['app_binary'], 'screenshot')
        self.assertEqual(named['zones']['renderer-span']['top_app_frames'], [('work', 2)])
        self.assertEqual(named['zones']['renderer-span']['no_named_app_frame_samples'], 0)

    def test_zone_edges_include_start_and_exclude_end(self):
        spans = [(100, 200), (300, 400)]
        self.assertEqual([zone(t, spans, [100, 300]) for t in
                          (99, 100, 199, 200, 299, 300, 399, 400)],
                         ['outside', 'renderer-span', 'renderer-span', 'between-spans',
                          'between-spans', 'renderer-span', 'renderer-span', 'outside'])


if __name__ == '__main__':
    unittest.main()
