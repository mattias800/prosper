#!/usr/bin/env python3
"""Census and VERDICT for a renderer-determinism campaign (#2945).

The measurement this reads is trivial -- replay one frozen capture many times and count how many
distinct output hashes come back. The reason it needs a tool is the *reading*, and this file exists
because two campaigns on #2945 were read wrongly in opposite directions:

  * A clean campaign was read as "fixed". It cannot be, on its own. #2945's failure rate drifts
    machine-wide over minutes -- the identical command measured 0 of 20 failures in one ten-minute
    window and 12 of 12 in the next with nothing changed -- so a campaign that happens to sit in a
    quiet window produces exactly the same numbers as a campaign on a repaired renderer.
  * A varying campaign was read as "prosper is nondeterministic". It may be the apparatus:
    `gpu_replay --warmup-repeats` renders each repeat on top of the previous one's persistent
    targets, so its hashes drift by construction.

So a campaign carries a CONTROL beside the subject: a bare-Vulkan program with no prosper code in
the process, which reproduces the same class. The control is not there to be believed on its own --
its own README says a clean run proves nothing -- it is there so a clean SUBJECT can be told apart
from a quiet WINDOW. That gives three verdicts rather than two, and the third is the one that keeps
being skipped:

  NONDETERMINISTIC  a subject arm returned more than one distinct hash. The campaign is decisive.
  DETERMINISTIC     every subject arm returned one hash AND the control failed at least once, so
                    the campaign demonstrably sampled a window in which this class is expressible.
  UNDECIDED         every subject arm returned one hash and the control never failed either. This
                    is NOT a negative result. Quote it as "no instance observed", never as "fixed".

Input is the CSV written by `replay_determinism.sh` (see its header comment). Rows are tidy:

    epoch,round,cond,peers,gpu_pct,role,arm,value,rc,ms

`role` is `subject` or `control`; `arm` names the environment and what was replayed; `value` is the
output hash for a subject and `pass` / `fail:<detail>` for a control.
"""

from __future__ import annotations

import argparse
import collections
import csv
import sys
from dataclasses import dataclass, field

FIELDS = ["epoch", "round", "cond", "peers", "gpu_pct", "role", "arm", "value", "rc",
          "ms"]

NONDETERMINISTIC = "NONDETERMINISTIC"
DETERMINISTIC = "DETERMINISTIC"
UNDECIDED = "UNDECIDED"


@dataclass
class Row:
    epoch: int
    round: int
    cond: str
    peers: int
    gpu_pct: int
    role: str
    arm: str
    value: str
    rc: int
    ms: int


@dataclass
class Report:
    rows: list = field(default_factory=list)
    verdict: str = UNDECIDED
    reasons: list = field(default_factory=list)

    @property
    def subjects(self):
        return [r for r in self.rows if r.role == "subject"]

    @property
    def controls(self):
        return [r for r in self.rows if r.role == "control"]

    @property
    def span_s(self) -> int:
        if not self.rows:
            return 0
        return max(r.epoch for r in self.rows) - min(r.epoch for r in self.rows)


def parse(handle) -> list:
    """Read the tidy CSV. A malformed row is an error, never a silently dropped sample."""
    rows = []
    reader = csv.reader(handle)
    for lineno, raw in enumerate(reader, start=1):
        if not raw or raw[0].startswith("#"):
            continue
        if raw == FIELDS:                      # header
            continue
        if len(raw) != len(FIELDS):
            raise ValueError(f"line {lineno}: expected {len(FIELDS)} fields, got {len(raw)}: {raw}")
        try:
            rows.append(Row(epoch=int(raw[0]), round=int(raw[1]), cond=raw[2], peers=int(raw[3]),
                            gpu_pct=int(raw[4]), role=raw[5], arm=raw[6], value=raw[7],
                            rc=int(raw[8]), ms=int(raw[9])))
        except ValueError as exc:
            raise ValueError(f"line {lineno}: {exc}") from exc
    return rows


def control_failed(row: Row) -> bool:
    """A control row reports a failure by any value that is not exactly `pass`.

    `rc` is deliberately NOT part of this. A control that could not run has measured nothing, and
    reading its non-zero exit as a failure of the class would manufacture a DETERMINISTIC verdict
    out of a broken instrument -- the exact inversion this whole file exists to prevent.
    """
    return row.role == "control" and row.rc == 0 and row.value != "pass"


def control_broken(row: Row) -> bool:
    return row.role == "control" and row.rc != 0


NON_ANSWERS = ("", "none")


def subject_answered(row: Row) -> bool:
    """Did this replay actually produce a hash?

    The runner writes `none` when it found no hash line in the replay's output, and it writes that
    with the replay's own exit status -- which is frequently 0, because a tool can exit cleanly
    having printed something the parser did not recognise. That is not a hypothetical: the runner
    recorded `none` for EVERY bundle replay until it learned that `--bundle` prints per-submit lines
    and no `output=` line at all.

    Counting `none` as a hash is the exact failure this whole file exists to prevent, one level in.
    A campaign of 469 rows of `none` has one distinct value per arm and no varying arm, so a reader
    that filters only on `rc` calls it DETERMINISTIC -- reporting a campaign that measured NOTHING
    as the strongest possible result. The producer bug is fixable and was fixed; the reader is the
    durable artefact, and it has to refuse.
    """
    return row.role == "subject" and row.rc == 0 and row.value not in NON_ANSWERS


def evaluate(rows: list) -> Report:
    report = Report(rows=rows)
    if not rows:
        report.verdict = UNDECIDED
        report.reasons.append("no rows")
        return report

    subjects = [r for r in rows if r.role == "subject"]
    if not subjects:
        report.verdict = UNDECIDED
        report.reasons.append("no subject rows: nothing was replayed")
        return report

    # Three kinds of subject row, and only the first is a sample. A replay that did not finish has
    # no output to compare; a replay that finished without producing a hash measured nothing at all.
    # Counting either as a value would turn a timeout into NONDETERMINISTIC and a campaign of
    # non-answers into DETERMINISTIC.
    answered = [r for r in subjects if subject_answered(r)]
    failed_replays = [r for r in subjects if r.rc != 0]
    # Derived from the SAME predicate as `answered`, not from an independent re-test of the
    # value: two spellings of one rule can disagree, and the disagreement would be silent.
    non_answers = [r for r in subjects if r.rc == 0 and not subject_answered(r)]
    per_arm = collections.defaultdict(collections.Counter)
    for row in answered:
        per_arm[row.arm][row.value] += 1

    varying = {arm: c for arm, c in per_arm.items() if len(c) > 1}

    if varying:
        report.verdict = NONDETERMINISTIC
        for arm, counter in sorted(varying.items()):
            report.reasons.append(
                f"arm {arm}: {len(counter)} distinct hashes over {sum(counter.values())} replays")
        return report

    if non_answers:
        report.verdict = UNDECIDED
        report.reasons.append(
            f"{len(non_answers)} of {len(subjects)} replays exited 0 but recorded no hash "
            f"({'/'.join(sorted({r.value or '<empty>' for r in non_answers}))}); a campaign that "
            "recorded nothing has one value per arm and no varying arm, which is not a result")
        return report

    if failed_replays:
        report.verdict = UNDECIDED
        report.reasons.append(
            f"{len(failed_replays)} of {len(subjects)} replays exited non-zero; a run that did not "
            "finish is not a sample of what the finished ones measured")
        return report

    silent_arms = sorted({r.arm for r in subjects} - set(per_arm))
    if silent_arms:
        report.verdict = UNDECIDED
        report.reasons.append(
            f"arm(s) {', '.join(silent_arms)} produced no usable replay at all")
        return report

    controls = [r for r in rows if r.role == "control"]
    fired = [r for r in controls if control_failed(r)]
    broken = [r for r in controls if control_broken(r)]
    if not controls:
        report.verdict = UNDECIDED
        report.reasons.append(
            "no control rows: a clean subject cannot be told apart from a quiet window")
        return report
    if not fired:
        report.verdict = UNDECIDED
        report.reasons.append(
            f"the control never failed in {len(controls)} rounds, so this campaign has not shown "
            "that it sampled a window in which the class is expressible; quote 'no instance "
            "observed', not 'deterministic'")
        if broken:
            report.reasons.append(f"...and {len(broken)} control round(s) could not run at all")
        return report

    report.verdict = DETERMINISTIC
    report.reasons.append(
        f"the control failed in {len(fired)} of {len(controls)} rounds while every subject arm "
        "returned one hash, so the campaign met the failing regime and the subject survived it")
    return report


def format_report(report: Report) -> str:
    out = []
    span = report.span_s
    rounds = len({r.round for r in report.rows})
    out.append(f"rounds={rounds}  observations={len(report.rows)}  "
               f"wall-clock span={span}s ({span / 3600:.2f} h)")
    if span and rounds > 1:
        out.append(f"  (a span shorter than the drift period samples ONE window, not {rounds})")
    conds = collections.Counter(r.cond for r in report.rows)
    out.append(f"conditions: {dict(conds)}")
    peers = sum(1 for r in report.rows if r.peers > 0)
    out.append(f"observations taken while a peer GPU consumer was up: {peers}")

    # A "loaded" arm that did not actually load the GPU has not tested the load condition, and
    # nothing else in the CSV would say so.
    for cond in sorted({r.cond for r in report.rows}):
        pcts = [r.gpu_pct for r in report.rows if r.cond == cond and r.gpu_pct >= 0]
        if pcts:
            out.append(f"  [{cond}] GPU busy: mean {sum(pcts) / len(pcts):.1f}%  "
                       f"max {max(pcts)}%  over {len(pcts)} samples")
        else:
            out.append(f"  [{cond}] GPU busy: not sampled on this platform")

    per_arm = collections.defaultdict(collections.Counter)
    unusable = collections.defaultdict(collections.Counter)
    for row in report.subjects:
        if subject_answered(row):
            per_arm[row.arm][row.value] += 1
        else:
            unusable[row.arm][row.value if row.rc == 0 else f"rc={row.rc}"] += 1
    for arm in sorted(set(per_arm) | set(unusable)):
        counter = per_arm[arm]
        out.append(f"\nsubject {arm}: {len(counter)} distinct over {sum(counter.values())} "
                   f"USABLE replays")
        for value, n in counter.most_common():
            out.append(f"    {value} x{n}")
        for value, n in unusable[arm].most_common():
            out.append(f"    [not a sample] {value} x{n}")

    controls = report.controls
    if controls:
        fired = [r for r in controls if control_failed(r)]
        broken = [r for r in controls if control_broken(r)]
        out.append(f"\ncontrol: {len(fired)} failing of {len(controls)} rounds"
                   f"{f', {len(broken)} could not run' if broken else ''}")
        for row in fired[:10]:
            out.append(f"    round={row.round} cond={row.cond} arm={row.arm} {row.value}")

    out.append(f"\nVERDICT: {report.verdict}")
    for reason in report.reasons:
        out.append(f"  - {reason}")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv", help="campaign CSV, or - for stdin")
    ap.add_argument("--exit-code", action="store_true",
                    help="exit 1 on NONDETERMINISTIC, 2 on UNDECIDED, 0 on DETERMINISTIC")
    args = ap.parse_args(argv)
    handle = sys.stdin if args.csv == "-" else open(args.csv, newline="")
    try:
        rows = parse(handle)
    finally:
        if handle is not sys.stdin:
            handle.close()
    report = evaluate(rows)
    print(format_report(report))
    if not args.exit_code:
        return 0
    return {DETERMINISTIC: 0, NONDETERMINISTIC: 1, UNDECIDED: 2}[report.verdict]


if __name__ == "__main__":
    sys.exit(main())
