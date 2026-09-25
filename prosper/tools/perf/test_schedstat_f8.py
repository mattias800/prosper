#!/usr/bin/env python3
"""Discriminating scheduler/F8 boundary and identity controls."""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schedstat_f8 import join
from schedstat_probe import stat_fields


def sample(before, after, runtime, runnable, *, tid=7, starttime=11):
    return {'before_ns': before, 'after_ns': after, 'threads': [{
        'tid': tid, 'starttime': starttime, 'state': 'S', 'wchan': 'futex',
        'runtime_ns': runtime, 'runnable_ns': runnable}]}


def rejects(report, spans, tid, reason):
    try:
        join(report, spans, tid)
    except ValueError as error:
        assert reason in str(error), error
    else:
        raise AssertionError(f'expected refusal: {reason}')


def main():
    # The process stat parser must ignore a parenthesized command containing spaces.
    fields = 'R ' + ' '.join(['0'] * 18 + ['12345'])
    assert stat_fields('7 (name with spaces) ' + fields) == ('R', 12345)
    spans = [(0, 10), (20, 30)]
    report = {'candidate_tids': [7], 'samples': [
        sample(2, 2, 0, 0), sample(8, 8, 3, 1),
        sample(12, 12, 3, 1), sample(18, 18, 4, 2),
        sample(22, 22, 4, 2), sample(28, 28, 7, 2)]}
    zones = join(report, spans, 7)
    assert zones['renderer']['segments'] == 2
    assert zones['renderer']['runtime_ns'] == 6
    assert zones['renderer']['coverage'] == 0.6
    assert zones['gap']['segments'] == 1
    assert zones['gap']['sleep_or_unknown_ns'] == 4

    crossing = copy.deepcopy(report)
    crossing['samples'][1]['after_ns'] = 11  # midpoint remains inside renderer.
    assert join(crossing, spans, 7)['renderer']['segments'] == 1

    missing = copy.deepcopy(report)
    missing['samples'][1]['threads'] = []
    assert join(missing, spans, 7)['renderer']['segments'] == 1

    changed = copy.deepcopy(report)
    changed['samples'][-1]['threads'][0]['starttime'] = 12
    rejects(changed, spans, 7, 'changed identity')
    rejects(report, spans, 8, 'not present')
    rejects(report, [(0, 12), (10, 30)], 7, 'nonoverlapping')
    print('scheduler/F8 bracket, missing-read, identity, and span controls passed')


if __name__ == '__main__':
    main()
