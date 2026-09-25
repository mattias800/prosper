#!/usr/bin/env python3
"""Join sampled /proc schedstat deltas to complete F8 renderer/gap intervals.

Only pairs of reads wholly within one interval contribute. The unobserved edge
fraction remains explicit; scheduler runtime, runnable wait, and the signed
remainder are observations of one thread, not a partition of whole-game FPS.
"""
import argparse
import bisect
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perf_f8_gap import load_spans


def interval(t, spans, starts):
    i = bisect.bisect_right(starts, t) - 1
    if i < 0 or t >= spans[-1][1]:
        return None
    if t < spans[i][1]:
        return ('renderer', i)
    return ('gap', i)


def join(report, spans, tid):
    if not spans or any(a >= b for a, b in spans) or any(
            right[0] < left[1] for left, right in zip(spans, spans[1:])):
        raise ValueError('renderer spans must be nonempty, positive, and nonoverlapping')
    if tid not in report['candidate_tids']:
        raise ValueError('selected TID was not present when the sampler started')
    starts = [a for a, _ in spans]
    wanted = []
    for index, row in enumerate(report['samples']):
        if row['after_ns'] < row['before_ns']:
            raise ValueError('reversed read bracket')
        selected = [t for t in row['threads'] if t['tid'] == tid]
        if len(selected) > 1:
            raise ValueError('duplicate selected TID in one sample')
        if selected:
            wanted.append((index, row['before_ns'], row['after_ns'], selected[0]))
    if len(wanted) < 2:
        raise ValueError('selected TID has fewer than two observations')
    identities = {t['starttime'] for _, _, _, t in wanted}
    if len(identities) != 1:
        raise ValueError('selected TID changed identity')
    result = {k: collections.Counter() for k in ('renderer', 'gap')}
    wchans = {k: collections.Counter() for k in ('renderer', 'gap')}
    for (a_index, a_before, a_after, ta), (b_index, b_before, b_after, tb) in zip(wanted, wanted[1:]):
        if b_index != a_index + 1:
            continue  # Never bridge a failed or missing selected-thread observation.
        if b_before < a_after:
            raise ValueError('overlapping sampler read brackets')
        ia = interval(a_before, spans, starts)
        # A counter may change at any point during either /proc read. Count a
        # segment only when both complete read brackets belong to one F8 zone.
        if (ia is None or ia != interval(a_after, spans, starts) or
                ia != interval(b_before, spans, starts) or
                ia != interval(b_after, spans, starts)):
            continue
        a = (a_before + a_after) // 2
        b = (b_before + b_after) // 2
        zone = ia[0]
        runtime = tb['runtime_ns'] - ta['runtime_ns']
        runnable = tb['runnable_ns'] - ta['runnable_ns']
        if runtime < 0 or runnable < 0:
            raise ValueError('scheduler counter regressed for one thread identity')
        row = result[zone]
        row['segments'] += 1
        row['wall_ns'] += b - a
        row['runtime_ns'] += runtime
        row['runnable_ns'] += runnable
        row['sleep_or_unknown_ns'] += b - a - runtime - runnable
        row['max_bracket_ns'] = max(row['max_bracket_ns'], a_after - a_before,
                                    b_after - b_before)
        wchans[zone][ta['state'] + ':' + ta['wchan']] += 1
    expected = {'renderer': sum(b - a for a, b in spans),
                'gap': sum(b[0] - a[1] for a, b in zip(spans, spans[1:]))}
    return {zone: {**result[zone], 'expected_wall_ns': expected[zone],
                   'coverage': result[zone]['wall_ns'] / expected[zone],
                   'wchan_snapshots': wchans[zone].most_common()}
            for zone in result}


def selftest():
    spans = [(0, 10), (20, 30)]
    assert interval(5, spans, [0, 20]) == ('renderer', 0)
    assert interval(15, spans, [0, 20]) == ('gap', 0)
    assert interval(30, spans, [0, 20]) is None
    def sample(t, runtime, runnable):
        return {'before_ns': t, 'after_ns': t,
                'threads': [{'tid': 7, 'starttime': 11, 'state': 'S',
                             'wchan': 'futex', 'runtime_ns': runtime,
                             'runnable_ns': runnable}]}
    report = {'candidate_tids': [7], 'samples': [sample(2, 0, 0), sample(8, 3, 1),
                          sample(12, 3, 1), sample(18, 4, 2),
                          sample(22, 4, 2), sample(28, 7, 2)]}
    result = join(report, spans, 7)
    assert result['renderer']['segments'] == 2
    assert result['renderer']['runtime_ns'] == 6
    assert result['renderer']['coverage'] == 12 / 20
    assert result['gap']['segments'] == 1
    assert result['gap']['sleep_or_unknown_ns'] == 4
    bad = {'candidate_tids': [7], 'samples': [sample(8, 0, 0), sample(12, 0, 0)]}
    assert join(bad, spans, 7)['renderer'].get('segments', 0) == 0
    # The second sample's midpoint is inside the renderer span, but its
    # counter read crosses the boundary and cannot be attributed to either side.
    crossing = {'candidate_tids': [7], 'samples': [sample(2, 0, 0),
                            {**sample(8, 3, 1), 'after_ns': 11}]}
    assert join(crossing, spans, 7)['renderer'].get('segments', 0) == 0
    print('sched/F8 interval and cross-boundary controls passed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('f8', nargs='?', type=Path)
    parser.add_argument('scheduler', nargs='?', type=Path)
    parser.add_argument('--tid', type=int)
    parser.add_argument('--pid', type=int)
    parser.add_argument('--selftest', action='store_true')
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    if not args.f8 or not args.scheduler or not args.tid or not args.pid:
        parser.error('F8 capture, scheduler JSON, --pid, and --tid are required')
    _, spans = load_spans(args.f8)
    report = json.loads(args.scheduler.read_text())
    if report['terminal'] != 'completed' or report['overruns']:
        raise ValueError('scheduler probe incomplete or overran its requested interval')
    if report['pid'] != args.pid:
        raise ValueError('scheduler process ID differs from the requested perf process')
    print(json.dumps({'pid': report['pid'], 'tid': args.tid,
                      'read_failures': report['read_failures'],
                      'zones': join(report, spans, args.tid)}, indent=2))


if __name__ == '__main__':
    main()
