#!/usr/bin/env python3
"""Compare two `tools/screenshot` capture manifests, and REFUSE when the runs are not comparable.

WHY THIS EXISTS
---------------
A framerate is a statement about a tuple -- (title, route, build, GPU, driver, present path, pacing)
-- and a figure quoted without its tuple cannot be set against another one. This project has been
wrong that way repeatedly: a readback-path rate against a GPU-present rate, a `fifo` run against an
`immediate` one, a rate over a menu-heavy window against one over gameplay. The charter lists those as
traps to remember; this tool makes them impossible to commit by accident, because it will not print a
delta for two runs whose conditions differ.

WHAT IT DOES
------------
    compare_runs.py BASELINE.jsonl CANDIDATE.jsonl

Reads the `run`, `conditions` and `summary` records of each manifest. If every condition that decides
comparability matches, it prints per-field deltas and exits 1 when a regression threshold is crossed.
If any condition differs or is missing in either run it prints each reason and exits 2 WITHOUT printing
a delta. Only the build revision and the measured fields may differ.

EXIT CODES
    0  comparable, no regression
    1  comparable, a threshold was crossed
    2  refused: the runs are not comparable, or a manifest could not be read

A refusal is never softened into a warning. "Could not compare" and "compared and fine" must not share
an exit code, for the same reason ctest's "no tests ran" and "all passed" must not (see the charter).

THE THRESHOLDS ARE PROPOSED, NOT MEASURED
-----------------------------------------
3% on the typical rate and 5% on the 1% low are starting points, not noise-calibrated limits: run-to-run
variance on this project has not been characterised per title. Repeat the BASELINE against itself to
learn a title's noise before trusting a threshold on it, and tune with --max-*-drop-pct.

WHAT THIS DOES NOT DO
---------------------
It does not measure anything, and it never reads or writes anything but the two manifests. It cannot
tell whether a scene was rendered correctly (a framerate is a statement about time, never pixels) and
it does not look at image content.
"""

import argparse
import json
import sys

# Conditions that must be IDENTICAL, and present, in both runs. Each is a way two rates have been
# wrongly compared in this project's history. (record, key) pairs.
MATCH_KEYS = [
    ("run", "title"),
    ("run", "input_route"),
    ("run", "capture_mode"),
    ("run", "seconds"),
    ("run", "every"),
    ("run", "requested"),
    ("run", "warmup_ms"),
    ("run", "warmup_submits"),
    ("run", "render_every"),
    ("run", "render_every_for_ms"),
    ("run", "render_scale"),
    ("run", "render_target_dim"),
    ("run", "render_resource_dim"),
    ("conditions", "harness"),
    ("conditions", "present_path"),
    ("conditions", "os"),
    ("conditions", "flip_pace_fps"),
    ("conditions", "gpu_vendor_id"),
    ("conditions", "gpu_device_id"),
    ("conditions", "gpu_driver_version"),
    ("conditions", "gpu_api_version"),
    ("conditions", "gpu_device_type"),
]

# The burned-in annotation costs time per frame, so a run with it and a run without are not the same
# measurement. It lives inside the run record's `assertions` object.
ASSERTION_MATCH_KEYS = ["fps_overlay"]

# A window that mixed two regimes (a menu and gameplay) yields a rate that describes neither; the
# counter's own documentation says to narrow the window rather than quote it. Below this share of the
# window producing frames at roughly the typical rate, the pair is refused.
MIN_ACTIVE_FRACTION = 0.90

# (summary key, label, higher_is_better, unit)
METRICS = [
    ("typical_fps", "typical fps", True, "fps"),
    ("distinct_fps", "run-average distinct fps", True, "fps"),
    ("low_1pct_fps", "1% low (1/p99)", True, "fps"),
    ("interval_p99_ms", "p99 frame time", False, "ms"),
    ("interval_p95_ms", "p95 frame time", False, "ms"),
    ("active_fraction", "active fraction", True, ""),
]


class ManifestError(Exception):
    """A manifest that cannot be read at all (as opposed to one that is merely not comparable)."""


def load_manifest(path):
    """Return {'run': ..., 'conditions': ..., 'summary': ...}, keeping the LAST record of each type."""
    records = {}
    try:
        with open(path, encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ManifestError(f"{path}:{number}: not valid JSON ({error.msg})") from error
                if isinstance(record, dict) and record.get("type") in (
                    "run",
                    "conditions",
                    "summary",
                ):
                    records[record["type"]] = record
    except OSError as error:
        raise ManifestError(f"{path}: cannot be read ({error.strerror})") from error
    return records


def _get(records, record, key):
    entry = records.get(record)
    if entry is None:
        return None
    if record == "run" and key in ASSERTION_MATCH_KEYS:
        return (entry.get("assertions") or {}).get(key)
    return entry.get(key)


def refusals(baseline, candidate):
    """Every reason the pair is not comparable. Empty means comparable."""
    reasons = []
    for label, records in (("baseline", baseline), ("candidate", candidate)):
        for record in ("run", "conditions", "summary"):
            if record not in records:
                hint = (
                    " (the manifest predates the `conditions` record; re-run it)"
                    if record == "conditions"
                    else ""
                )
                reasons.append(f"{label}: no `{record}` record{hint}")
    if reasons:
        return reasons

    for record, key in MATCH_KEYS + [("run", k) for k in ASSERTION_MATCH_KEYS]:
        a, b = _get(baseline, record, key), _get(candidate, record, key)
        if a is None or b is None:
            side = "baseline" if a is None else "candidate"
            reasons.append(f"{record}.{key}: missing in {side}")
        elif a != b:
            reasons.append(f"{record}.{key}: baseline {a!r} != candidate {b!r}")

    for label, records in (("baseline", baseline), ("candidate", candidate)):
        conditions, summary = records["conditions"], records["summary"]
        if conditions.get("gpu_known") is not True:
            reasons.append(f"{label}: the renderer never selected a GPU, so the device is unknown")
        if summary.get("status") != "ok":
            reasons.append(f"{label}: run status is {summary.get('status')!r}, not 'ok'")
        if summary.get("frame_rate_measured") is not True:
            reasons.append(f"{label}: no frames were published, so there is no rate")
        elif summary.get("typical_fps_measured") is not True:
            reasons.append(f"{label}: fewer than two distinct frames, so there is no typical rate")
        else:
            active = summary.get("active_fraction")
            if not isinstance(active, (int, float)) or active < MIN_ACTIVE_FRACTION:
                reasons.append(
                    f"{label}: active fraction {active!r} is below {MIN_ACTIVE_FRACTION:.2f}; the window "
                    f"mixed two regimes, so its rate describes neither. Narrow the route and re-run"
                )
    return reasons


def _delta(metric, baseline_value, candidate_value, higher_is_better):
    """(absolute, percent, regressed_direction) or None when either side is absent."""
    if baseline_value is None or candidate_value is None:
        return None
    absolute = candidate_value - baseline_value
    percent = (absolute / baseline_value * 100.0) if baseline_value else None
    worse = (absolute < 0) if higher_is_better else (absolute > 0)
    return absolute, percent, worse


def compare(baseline, candidate, max_typical_drop_pct, max_low_drop_pct):
    """Per-metric rows plus the list of crossed thresholds. Caller must have checked refusals()."""
    rows, crossed = [], []
    for key, label, higher_better, unit in METRICS:
        a = baseline["summary"].get(key)
        b = candidate["summary"].get(key)
        delta = _delta(key, a, b, higher_better)
        rows.append(
            {
                "key": key,
                "label": label,
                "unit": unit,
                "baseline": a,
                "candidate": b,
                "delta": delta,
            }
        )
        if delta is None or delta[1] is None:
            continue
        drop = -delta[1]  # positive when the candidate is lower
        if key == "typical_fps" and drop > max_typical_drop_pct:
            crossed.append(f"typical fps down {drop:.1f}% (limit {max_typical_drop_pct:g}%)")
        if key == "low_1pct_fps" and drop > max_low_drop_pct:
            crossed.append(f"1% low down {drop:.1f}% (limit {max_low_drop_pct:g}%)")
    return rows, crossed


def _fmt(value, unit):
    if value is None:
        return "--"
    return f"{value:.3f} {unit}".rstrip()


def render(rows, baseline, candidate):
    lines = []
    revision_a = baseline["run"].get("build_revision", "?")
    revision_b = candidate["run"].get("build_revision", "?")
    lines.append("comparable: same title, route, GPU, driver, present path and pacing")
    if revision_a == revision_b:
        lines.append(
            f"note: both runs are build {revision_a[:12]}; this is a noise measurement, "
            f"not a before/after"
        )
    else:
        lines.append(f"baseline build {revision_a[:12]}  ->  candidate build {revision_b[:12]}")
    lines.append(f"{'metric':<28}{'baseline':>16}{'candidate':>16}{'delta':>18}")
    for row in rows:
        delta = row["delta"]
        if delta is None:
            text = "n/a (not measured)"
        else:
            absolute, percent, _ = delta
            text = f"{absolute:+.3f}" + (f" ({percent:+.1f}%)" if percent is not None else "")
        lines.append(
            f"{row['label']:<28}{_fmt(row['baseline'], row['unit']):>16}"
            f"{_fmt(row['candidate'], row['unit']):>16}{text:>18}"
        )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare two tools/screenshot manifests; refuse when the runs are not comparable."
    )
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument(
        "--max-typical-drop-pct",
        type=float,
        default=3.0,
        help="regression when typical fps falls by more than this (proposed, not "
        "noise-calibrated; default 3)",
    )
    parser.add_argument(
        "--max-low-drop-pct",
        type=float,
        default=5.0,
        help="regression when the 1%% low falls by more than this (proposed, not "
        "noise-calibrated; default 5)",
    )
    parser.add_argument("--json", action="store_true", help="print one JSON object instead of text")
    args = parser.parse_args(argv)

    try:
        baseline = load_manifest(args.baseline)
        candidate = load_manifest(args.candidate)
    except ManifestError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2

    reasons = refusals(baseline, candidate)
    if reasons:
        if args.json:
            print(json.dumps({"comparable": False, "refusals": reasons}, indent=2))
        else:
            print(
                "REFUSED: these runs are not comparable, so no delta is reported:", file=sys.stderr
            )
            for reason in reasons:
                print(f"  - {reason}", file=sys.stderr)
        return 2

    rows, crossed = compare(baseline, candidate, args.max_typical_drop_pct, args.max_low_drop_pct)
    if args.json:
        print(
            json.dumps(
                {
                    "comparable": True,
                    "regressions": crossed,
                    "rows": [
                        {k: r[k] for k in ("key", "baseline", "candidate")}
                        | {
                            "delta": None
                            if r["delta"] is None
                            else {"absolute": r["delta"][0], "percent": r["delta"][1]}
                        }
                        for r in rows
                    ],
                },
                indent=2,
            )
        )
    else:
        print(render(rows, baseline, candidate))
        for item in crossed:
            print(f"REGRESSION: {item}")
    return 1 if crossed else 0


if __name__ == "__main__":
    sys.exit(main())
