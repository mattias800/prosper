#!/usr/bin/env python3
"""Failure arms for the A/B identity guard and shape-weighted local comparison."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from compare_capture_pair import PairError, comparable_manifests, load_run, shape_weighted_estimates


def manifest(value):
    return {
        'title': 'fixture', 'dump': '/guest/fixture', 'duration': 240,
        'frozen_binary_sha256': 'binary-sha',
        'route_identity': {'sha256': 'route-sha'},
        'driver_identity': {'sha256': 'driver-sha'},
        'capture_driver': {'sha256_before': 'harness-sha', 'sha256_after': 'harness-sha'},
        'measurement': {'native_perf': False, 'f8_after_ms': 180000,
                        'throughput_claim': False},
        'env': {'PROSPER_FRAME_RESOURCE_RESERVE': value, 'PROSPER_RENDER': '1',
                'TMPDIR': '/state/' + value, 'PROSPER_PAD_SCRIPT': '@/route/' + value},
    }


def renderer(draws, total, build):
    return {'type': 'renderer', 'draws': draws, 'callbacks': 1, 'texture_bytes': 0,
            'total_ms': total, 'build_resources_ms': build,
            'frontend_buffer_ms': 0.5, 'backend_ms': 0.5, 'gpu_wait_ms': 0.1}


class ComparePairTest(unittest.TestCase):
    def test_only_one_balanced_value_may_change(self):
        control, candidate = manifest('0'), manifest('1')
        self.assertEqual(comparable_manifests(
            control, candidate, 'PROSPER_FRAME_RESOURCE_RESERVE'),
            {'name': 'PROSPER_FRAME_RESOURCE_RESERVE', 'control': '0', 'candidate': '1'})
        # The first real experiment carried this flag in only the control arm.
        unbalanced = copy.deepcopy(candidate)
        del unbalanced['env']['PROSPER_FRAME_RESOURCE_RESERVE']
        with self.assertRaisesRegex(PairError, 'key order or membership differs'):
            comparable_manifests(control, unbalanced, 'PROSPER_FRAME_RESOURCE_RESERVE')
        unbalanced = copy.deepcopy(candidate)
        unbalanced['env']['PROSPER_RENDER'] = '0'
        with self.assertRaisesRegex(PairError, 'expected only'):
            comparable_manifests(control, unbalanced, 'PROSPER_FRAME_RESOURCE_RESERVE')
        unbalanced = copy.deepcopy(candidate)
        del unbalanced['env']['TMPDIR']
        with self.assertRaisesRegex(PairError, 'key order or membership differs'):
            comparable_manifests(control, unbalanced, 'PROSPER_FRAME_RESOURCE_RESERVE')
        unbalanced = copy.deepcopy(candidate)
        unbalanced['env'] = dict(reversed(list(unbalanced['env'].items())))
        with self.assertRaisesRegex(PairError, 'key order or membership differs'):
            comparable_manifests(control, unbalanced, 'PROSPER_FRAME_RESOURCE_RESERVE')
        unbalanced = manifest('enabled')
        with self.assertRaisesRegex(PairError, 'different byte lengths'):
            comparable_manifests(control, unbalanced, 'PROSPER_FRAME_RESOURCE_RESERVE')

    def test_identity_and_complete_harness_are_required(self):
        control, candidate = manifest('0'), manifest('1')
        candidate['driver_identity']['sha256'] = 'other-driver'
        with self.assertRaisesRegex(PairError, 'driver SHA'):
            comparable_manifests(control, candidate, 'PROSPER_FRAME_RESOURCE_RESERVE')
        control, candidate = manifest('0'), manifest('1')
        control['frozen_binary_sha256'] = candidate['frozen_binary_sha256'] = ''
        with self.assertRaisesRegex(PairError, 'frozen_binary_sha256'):
            comparable_manifests(control, candidate, 'PROSPER_FRAME_RESOURCE_RESERVE')
        control, candidate = manifest('0'), manifest('1')
        candidate['frozen_binary_sha256'] = 'other-binary'
        with self.assertRaisesRegex(PairError, 'frozen_binary_sha256'):
            comparable_manifests(control, candidate, 'PROSPER_FRAME_RESOURCE_RESERVE')
        control, candidate = manifest('0'), manifest('1')
        candidate['route_identity']['sha256'] = 'other-route'
        with self.assertRaisesRegex(PairError, 'route SHA'):
            comparable_manifests(control, candidate, 'PROSPER_FRAME_RESOURCE_RESERVE')
        control, candidate = manifest('0'), manifest('1')
        candidate['capture_driver'] = {'sha256_before': 'other-harness',
                                       'sha256_after': 'other-harness'}
        with self.assertRaisesRegex(PairError, 'capture driver SHA'):
            comparable_manifests(control, candidate, 'PROSPER_FRAME_RESOURCE_RESERVE')
        control, candidate = manifest('0'), manifest('1')
        candidate['measurement']['f8_after_ms'] = 170000
        with self.assertRaisesRegex(PairError, 'measurement settings differ'):
            comparable_manifests(control, candidate, 'PROSPER_FRAME_RESOURCE_RESERVE')
        control, candidate = manifest('0'), manifest('1')
        candidate['capture_driver']['sha256_after'] = 'changed-during-run'
        with self.assertRaisesRegex(PairError, 'during a run'):
            comparable_manifests(control, candidate, 'PROSPER_FRAME_RESOURCE_RESERVE')

    def test_shape_medians_use_matched_counts_not_raw_totals(self):
        control = [renderer(1, 10, 2), renderer(1, 12, 4),
                   renderer(2, 1, 1), renderer(2, 3, 3),
                   renderer(9, 99, 99)]  # unmatched shape must not affect the estimate
        candidate = [renderer(1, 8, 1), renderer(1, 10, 3),
                     renderer(2, 2, 1), renderer(2, 4, 3)]
        result = shape_weighted_estimates(control, candidate, min_records=2)
        self.assertEqual((result['shape_count'], result['matched_records_per_arm']), (2, 4))
        self.assertEqual(result['fields']['total_ms']['control_mean_of_shape_medians_ms'], 6.5)
        self.assertEqual(result['fields']['total_ms']['candidate_mean_of_shape_medians_ms'], 6.0)
        self.assertEqual(result['fields']['total_ms']['candidate_minus_control_ms'], -0.5)
        with self.assertRaisesRegex(PairError, 'no recurring renderer shapes'):
            shape_weighted_estimates(control, candidate, min_records=3)

    def test_missing_peer_or_validity_audit_refuses_before_capture(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / 'run.json').write_text(json.dumps(manifest('0')))
            result = {'returncode': 124, 'validity_errors': [], 'peers_after': []}
            (directory / 'child-result.json').write_text(json.dumps(result))
            (directory / 'peer-samples.json').write_text(json.dumps([
                {'peers': [], 'frozen_pids': [123]}]))
            del result['validity_errors']
            (directory / 'child-result.json').write_text(json.dumps(result))
            with self.assertRaisesRegex(PairError, 'omits their audit'):
                load_run(directory)
            result['validity_errors'] = []
            (directory / 'child-result.json').write_text(json.dumps(result))
            (directory / 'peer-samples.json').write_text(json.dumps([{}]))
            with self.assertRaisesRegex(PairError, 'peer activity or peer-scan errors'):
                load_run(directory)
            (directory / 'peer-samples.json').write_text(json.dumps([
                {'peers': [{'comm': 'ninja'}], 'frozen_pids': [123]}]))
            with self.assertRaisesRegex(PairError, 'peer activity or peer-scan errors'):
                load_run(directory)
            (directory / 'peer-samples.json').write_text(json.dumps([
                {'peers': [], 'frozen_pids': [123]}]))
            (directory / 'peer-samples.json').write_text(json.dumps([
                {'peers': [], 'frozen_pids': [123],
                 'frozen_pid_error': 'duplicate frozen executable copies'}]))
            with self.assertRaisesRegex(PairError, 'peer activity or peer-scan errors'):
                load_run(directory)
            (directory / 'peer-samples.json').write_text(json.dumps([
                {'peers': [], 'frozen_pids': [123]}]))
            del result['peers_after']
            (directory / 'child-result.json').write_text(json.dumps(result))
            with self.assertRaisesRegex(PairError, 'omits their audit'):
                load_run(directory)

    def test_capture_count_and_truncation_refuse(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / 'run.json').write_text(json.dumps(manifest('0')))
            (directory / 'child-result.json').write_text(json.dumps(
                {'returncode': 124, 'validity_errors': [], 'peers_after': []}))
            (directory / 'peer-samples.json').write_text(json.dumps([
                {'peers': [], 'frozen_pids': [123]}]))
            with self.assertRaisesRegex(PairError, 'exactly one F8 capture'):
                load_run(directory)
            (directory / 'first.prperf').write_text('fixture')
            (directory / 'second.prperf').write_text('fixture')
            with self.assertRaisesRegex(PairError, 'exactly one F8 capture'):
                load_run(directory)
            (directory / 'second.prperf').unlink()
            with mock.patch('compare_capture_pair.load_capture', return_value=['records']):
                with mock.patch('compare_capture_pair.summarize', return_value={
                    'truncation': 'detail truncated', 'counts': {'renderer': 1}}):
                    with self.assertRaisesRegex(PairError, 'truncated or incomplete'):
                        load_run(directory)
                with mock.patch('compare_capture_pair.summarize', return_value={
                    'truncation': 'detail not truncated', 'counts': {'renderer': 0}}):
                    with self.assertRaisesRegex(PairError, 'no renderer records'):
                        load_run(directory)
                with mock.patch('compare_capture_pair.summarize', return_value={
                    'truncation': 'detail not truncated', 'counts': {'renderer': 2}}):
                    _, records, _, samples = load_run(directory)
                    self.assertEqual((records, samples), (['records'], 1))


if __name__ == '__main__':
    unittest.main()
