#!/usr/bin/env python3
"""Discriminating scheduler/F8 boundary, identity and zero-measurement controls.

Each case here is written so it fails when the behaviour it names is removed, not merely when
the join crashes. The bridging control in particular puts the dropped observation INTERIOR to
one span: a dropped observation that straddles a zone boundary is refused by the zone check
anyway, so it would pass with the anti-bridging guard deleted.
"""
import copy
import unittest

from schedstat_f8 import ZONE_FIELDS, interval, join


def sample(before, after, runtime, runnable, *, tid=7, starttime=11):
    return {'before_ns': before, 'after_ns': after, 'threads': [{
        'tid': tid, 'starttime': starttime, 'state': 'S', 'wchan': 'futex',
        'runtime_ns': runtime, 'runnable_ns': runnable}]}


def report_of(*samples, tids=(7,)):
    return {'candidate_tids': list(tids), 'samples': list(samples)}


SPANS = [(0, 10), (20, 30)]
BASE = report_of(sample(2, 2, 0, 0), sample(8, 8, 3, 1),
                 sample(12, 12, 3, 1), sample(18, 18, 4, 2),
                 sample(22, 22, 4, 2), sample(28, 28, 7, 2))


class IntervalTests(unittest.TestCase):
    def test_interval_maps_each_time_to_one_zone_and_refuses_outside(self):
        starts = [0, 20]
        self.assertEqual([interval(t, SPANS, starts) for t in (0, 9, 10, 19, 20, 29)],
                         [('renderer', 0), ('renderer', 0), ('gap', 0), ('gap', 0),
                          ('renderer', 1), ('renderer', 1)])
        self.assertIsNone(interval(30, SPANS, starts))
        self.assertIsNone(interval(-1, SPANS, starts))


class JoinAttributionTests(unittest.TestCase):
    def test_complete_brackets_are_attributed_to_their_own_zone(self):
        zones = join(BASE, SPANS, 7)
        self.assertEqual(zones['renderer']['segments'], 2)
        self.assertEqual(zones['renderer']['runtime_ns'], 6)
        self.assertEqual(zones['renderer']['coverage'], 0.6)
        self.assertEqual(zones['gap']['segments'], 1)
        self.assertEqual(zones['gap']['sleep_or_unknown_ns'], 4)

    def test_bracket_crossing_a_span_boundary_is_not_attributed(self):
        crossing = copy.deepcopy(BASE)
        crossing['samples'][1]['after_ns'] = 11  # midpoint remains inside renderer.
        self.assertEqual(join(crossing, SPANS, 7)['renderer']['segments'], 1)

    def test_a_dropped_observation_interior_to_one_span_is_never_bridged(self):
        # Three observations inside renderer span 0. Without the adjacency guard, dropping the
        # middle one leaves a pair that is entirely inside that span and would be counted as a
        # segment spanning an unobserved interval.
        dense = report_of(sample(2, 2, 0, 0), sample(4, 4, 1, 0), sample(8, 8, 3, 1),
                          sample(22, 22, 4, 2), sample(24, 24, 5, 2))
        self.assertEqual(join(dense, SPANS, 7)['renderer']['segments'], 3)
        missing = copy.deepcopy(dense)
        missing['samples'][1]['threads'] = []
        self.assertEqual(join(missing, SPANS, 7)['renderer']['segments'], 1)


class MeasuredZeroTests(unittest.TestCase):
    """A zone with nothing attributable must report zeros, never absent keys.

    `tools/perf/AGENTS.md`: "Keep missing measurements distinct from measured zeros." An empty
    gap zone is the expected shape whenever renderer spans cover the sampled window, so a
    consumer reading `segments` must not get a KeyError out of an ordinary capture.
    """

    def test_zone_with_no_attributable_segment_reports_explicit_zeros(self):
        covered = report_of(sample(2, 2, 0, 0), sample(4, 4, 1, 0),
                            sample(22, 22, 1, 0), sample(24, 24, 2, 0))
        gap = join(covered, SPANS, 7)['gap']
        for field in ZONE_FIELDS:
            self.assertIn(field, gap)
            self.assertEqual(gap[field], 0, field)
        self.assertEqual(gap['expected_wall_ns'], 10)
        self.assertEqual(gap['coverage'], 0.0)  # measured: the gap existed and nothing landed in it.
        self.assertEqual(gap['wchan_snapshots'], [])

    def test_every_zone_always_carries_the_full_field_set(self):
        zones = join(BASE, SPANS, 7)
        for name, row in zones.items():
            self.assertEqual(set(ZONE_FIELDS) - set(row), set(), name)


class ZeroLengthGapTests(unittest.TestCase):
    """A zero-length gap is unmeasurable, not a measured zero, and must not divide by zero."""

    def test_touching_spans_report_null_gap_coverage_instead_of_raising(self):
        touching = [(0, 10), (10, 20)]
        zones = join(report_of(sample(2, 2, 0, 0), sample(8, 8, 3, 1)), touching, 7)
        self.assertEqual(zones['gap']['expected_wall_ns'], 0)
        self.assertIsNone(zones['gap']['coverage'])
        self.assertEqual(zones['gap']['segments'], 0)
        self.assertEqual(zones['renderer']['segments'], 1)
        self.assertEqual(zones['renderer']['coverage'], 6 / 20)

    def test_single_span_reports_null_gap_coverage_instead_of_raising(self):
        zones = join(report_of(sample(2, 2, 0, 0), sample(8, 8, 3, 1)), [(0, 10)], 7)
        self.assertEqual(zones['gap']['expected_wall_ns'], 0)
        self.assertIsNone(zones['gap']['coverage'])
        self.assertEqual(zones['renderer']['segments'], 1)


class RefusalTests(unittest.TestCase):
    def rejects(self, report, spans, tid, reason):
        with self.assertRaisesRegex(ValueError, reason):
            join(report, spans, tid)

    def test_changed_thread_identity_refuses(self):
        changed = copy.deepcopy(BASE)
        changed['samples'][-1]['threads'][0]['starttime'] = 12
        self.rejects(changed, SPANS, 7, 'changed identity')

    def test_tid_absent_at_sampler_start_refuses(self):
        self.rejects(BASE, SPANS, 8, 'not present')

    def test_overlapping_renderer_spans_refuse(self):
        self.rejects(BASE, [(0, 12), (10, 30)], 7, 'nonoverlapping')

    def test_empty_or_inverted_spans_refuse(self):
        self.rejects(BASE, [], 7, 'nonempty')
        self.rejects(BASE, [(10, 10), (20, 30)], 7, 'positive')

    def test_overlapping_sampler_read_brackets_refuse(self):
        overlapping = report_of(sample(2, 6, 0, 0), sample(4, 8, 3, 1))
        self.rejects(overlapping, SPANS, 7, 'overlapping sampler read brackets')

    def test_reversed_read_bracket_refuses(self):
        self.rejects(report_of(sample(6, 2, 0, 0), sample(8, 8, 1, 0)), SPANS, 7,
                     'reversed read bracket')

    def test_regressed_scheduler_counter_refuses(self):
        # Same starttime, so the identity check passes and the counter check is what must fire.
        regressed = report_of(sample(2, 2, 5, 5), sample(8, 8, 4, 5))
        self.rejects(regressed, SPANS, 7, 'counter regressed')
        self.rejects(report_of(sample(2, 2, 5, 5), sample(8, 8, 5, 4)), SPANS, 7,
                     'counter regressed')

    def test_duplicate_selected_tid_in_one_sample_refuses(self):
        duplicated = copy.deepcopy(BASE)
        duplicated['samples'][0]['threads'].append(
            dict(duplicated['samples'][0]['threads'][0]))
        self.rejects(duplicated, SPANS, 7, 'duplicate selected TID')

    def test_fewer_than_two_observations_refuses(self):
        self.rejects(report_of(sample(2, 2, 0, 0)), SPANS, 7, 'fewer than two observations')


if __name__ == '__main__':
    unittest.main()
