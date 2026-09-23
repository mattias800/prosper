#!/usr/bin/env python3
"""Compare two routed F8 windows only after their A/B identities pass preflight.

Shape-weighted means of per-shape medians describe local recorded renderer work. They are neither
equal-input proof nor simulation FPS, and the result does not compare delivery.
"""

import argparse
import collections
import json
from pathlib import Path
import statistics
import sys

from performance_capture_report import load_capture, summarize


ISOLATION_ENV = frozenset({
    'TMPDIR', 'PROSPER_SAVE0', 'PROSPER_GRAPHICS_PIPELINE_CACHE_PATH',
    'XDG_CACHE_HOME', 'MESA_SHADER_CACHE_DIR', 'PROSPER_CAPTURE_DIR',
    'PROSPER_PAD_SCRIPT',
})
FIELDS = ('build_resources_ms', 'frontend_buffer_ms', 'total_ms',
          'backend_ms', 'gpu_wait_ms')


class PairError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise PairError(message)


def comparable_manifests(control, candidate, delta):
    """Refuse an unbalanced environment, including a control-only flag."""
    for key in ('title', 'dump', 'duration', 'frozen_binary_sha256'):
        require(control.get(key) == candidate.get(key) and control.get(key),
                f'{key} differs or is missing')
    for key in ('route_identity', 'driver_identity', 'capture_driver', 'measurement'):
        require(key in control and key in candidate, f'{key} is missing')
    require(control['route_identity'].get('sha256') and
            control['route_identity'].get('sha256') ==
            candidate['route_identity'].get('sha256'), 'route SHA differs or is missing')
    require(control['driver_identity'].get('sha256') and
            control['driver_identity'].get('sha256') ==
            candidate['driver_identity'].get('sha256'), 'driver SHA differs or is missing')
    for side in (control, candidate):
        driver = side['capture_driver']
        require(driver.get('sha256_before') == driver.get('sha256_after') and
                driver.get('sha256_before'), 'capture driver changed during a run')
        require(side['measurement'].get('native_perf') is False,
                'native perf overlap is unsupported by this comparison')
        require(side['measurement'].get('f8_after_ms') is not None,
                'F8 anchor is missing')
    require(control['capture_driver']['sha256_before'] ==
            candidate['capture_driver']['sha256_before'], 'capture driver SHA differs')
    require(control['measurement'] == candidate['measurement'],
            'measurement settings differ')

    require(list(control['env']) == list(candidate['env']),
            'environment key order or membership differs')
    left = {key: value for key, value in control['env'].items()
            if key not in ISOLATION_ENV}
    right = {key: value for key, value in candidate['env'].items()
             if key not in ISOLATION_ENV}
    require(left.keys() == right.keys(),
            'environment key sets differ; use the same named switch in both arms')
    require(delta in left, f'declared delta {delta} is absent')
    differences = {key for key in left if left[key] != right[key]}
    require(differences == {delta},
            f'environment values differ at {sorted(differences)}, expected only {delta}')
    require(len(left[delta].encode()) == len(right[delta].encode()),
            'declared delta values have different byte lengths')
    return {'name': delta, 'control': left[delta], 'candidate': right[delta]}


def shape_weighted_estimates(control, candidate, min_records=20):
    """Mean of per-shape medians, weighted by each pair's smaller record count."""
    def groups(records):
        result = collections.defaultdict(list)
        for row in records:
            if row.get('type') != 'renderer':
                continue
            shape = tuple(row[key] for key in ('draws', 'callbacks', 'texture_bytes'))
            for key in FIELDS:
                require(key in row, f'renderer record lacks {key}')
            result[shape].append(row)
        return result

    left, right = groups(control), groups(candidate)
    matched = sorted(set(left) & set(right))
    matched = [shape for shape in matched
               if min(len(left[shape]), len(right[shape])) >= min_records]
    require(matched, 'no recurring renderer shapes meet the count threshold')
    matched_count = sum(min(len(left[shape]), len(right[shape])) for shape in matched)
    fields = {}
    for field in FIELDS:
        left_total = sum(min(len(left[shape]), len(right[shape])) *
                         statistics.median(row[field] for row in left[shape])
                         for shape in matched)
        right_total = sum(min(len(left[shape]), len(right[shape])) *
                          statistics.median(row[field] for row in right[shape])
                          for shape in matched)
        fields[field] = {'control_mean_of_shape_medians_ms': left_total / matched_count,
                         'candidate_mean_of_shape_medians_ms': right_total / matched_count,
                         'candidate_minus_control_ms':
                         (right_total - left_total) / matched_count,
                         'relative_percent': 100 * (right_total / left_total - 1)
                         if left_total else None}
    control_count = sum(map(len, left.values()))
    candidate_count = sum(map(len, right.values()))
    shapes = []
    for draws, callbacks, texture_bytes in matched:
        shape = (draws, callbacks, texture_bytes)
        shapes.append({
            'draws': draws, 'callbacks': callbacks, 'texture_bytes': texture_bytes,
            'control_records': len(left[shape]), 'candidate_records': len(right[shape]),
            'control_total_median_ms': statistics.median(
                row['total_ms'] for row in left[shape]),
            'candidate_total_median_ms': statistics.median(
                row['total_ms'] for row in right[shape]),
            'control_build_median_ms': statistics.median(
                row['build_resources_ms'] for row in left[shape]),
            'candidate_build_median_ms': statistics.median(
                row['build_resources_ms'] for row in right[shape]),
        })
    shapes.sort(key=lambda row: min(row['control_records'], row['candidate_records']) *
                row['control_total_median_ms'], reverse=True)
    return {'shape_count': len(matched),
            'matched_records_per_arm': matched_count,
            'control_renderer_records': control_count,
            'candidate_renderer_records': candidate_count,
            'matched_fraction_control': matched_count / control_count,
            'matched_fraction_candidate': matched_count / candidate_count,
            'fields': fields,
            'shapes': shapes,
            'interpretation': 'local mean of shape medians; not equal inputs or FPS'}


def load_run(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'run.json').read_text())
    result = json.loads((directory / 'child-result.json').read_text())
    require(result.get('returncode') in (0, 124), 'guest failed or timed out abnormally')
    require(result.get('validity_errors') == [],
            'run has validity errors or omits their audit')
    require(result.get('peers_after') == [],
            'run ended with peers or omits their audit')
    samples = json.loads((directory / 'peer-samples.json').read_text())
    require(samples and all(row.get('peers') == [] and
                            'frozen_pids' in row and
                            not row.get('frozen_pid_error')
                            for row in samples), 'run has peer activity or peer-scan errors')
    captures = list(directory.glob('*.prperf'))
    require(len(captures) == 1, 'expected exactly one F8 capture')
    records = load_capture(captures[0])
    summary = summarize(records)
    require(summary.get('truncation') == 'detail not truncated',
            'F8 detail is truncated or incomplete')
    require(summary['counts'].get('renderer', 0) > 0, 'F8 has no renderer records')
    return manifest, records, summary, len(samples)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('control_dir', type=Path)
    parser.add_argument('candidate_dir', type=Path)
    parser.add_argument('--delta', required=True, help='the one environment value allowed to differ')
    parser.add_argument('--min-records', type=int, default=20)
    args = parser.parse_args(argv)
    try:
        require(args.min_records > 0, '--min-records must be positive')
        left, left_records, left_summary, left_samples = load_run(args.control_dir)
        right, right_records, right_summary, right_samples = load_run(args.candidate_dir)
        delta = comparable_manifests(left, right, args.delta)
        result = shape_weighted_estimates(left_records, right_records, args.min_records)
        result['delta'] = delta
        result['binary_sha256'] = left['frozen_binary_sha256']
        result['route_sha256'] = left['route_identity']['sha256']
        result['f8_anchor_ms'] = left['measurement']['f8_after_ms']
        result['peer_samples'] = [left_samples, right_samples]
        result['detail_seconds'] = [left_summary['detail_seconds'],
                                    right_summary['detail_seconds']]
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(f'compare_capture_pair: REFUSED: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
