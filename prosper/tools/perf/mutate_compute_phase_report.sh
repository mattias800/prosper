#!/bin/bash
# Per-check mutation coverage for compute_phase_report.py.
#
# WHY THIS EXISTS — and the rule it makes mechanical
# --------------------------------------------------
# `test_compute_phase_report.py` passing proves the checks ran. Mutating a mechanism away and watching
# the suite go red proves the suite is not blind. Neither proves that *the check you wrote for that
# mechanism* works — a survivor masked by three red siblings is invisible at suite granularity.
#
#   A mutation that turns the suite red proves the suite is not blind.
#   It does NOT prove the check you wrote for that mutation works.
#
# So every row below names the mutation AND the check that must kill it, and the run fails if any
# mutation is killed by a different check, survives, or aborts the suite. This caught two real defects
# in its own first run: a fixture whose two records cancelled (so the negative-time check proved
# nothing) and a mutation aimed at the reporting filter rather than the exclusion filter. Both read as
# healthy coverage at suite granularity.
#
# It does NOT close the ladder, and this file shipped with the next rung in it: a mutation BROADER than
# the defect it stood for (returning on any alias where the original returned only on all) let an
# unrelated check claim the kill. So a row is only as good as the fidelity of its mutation, and a
# reviewer should read the mutation text, not just the green line.
#
# Safe by construction: it copies the tool and its test to a scratch directory and mutates the COPY,
# so a crash cannot leave the tracked file modified.
#
# Usage: tools/perf/mutate_compute_phase_report.sh
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=$(mktemp -d "${TMPDIR:-/var/tmp}/computeperf-mutate-XXXXXX") || exit 2
trap 'rm -rf "$WORK"' EXIT
cp "$HERE/compute_phase_report.py" "$HERE/test_compute_phase_report.py" "$WORK/" || exit 2
TOOL="$WORK/compute_phase_report.py"
PRISTINE="$WORK/pristine.py"
cp "$TOOL" "$PRISTINE"

run_mutation() { # $1=name  $2=check that must kill it  $3=old text  $4=new text
  cp "$PRISTINE" "$TOOL"
  if ! python3 - "$TOOL" "$3" "$4" <<'PY'
import sys
p, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(p).read()
assert old in s, "anchor not found"
open(p, "w").write(s.replace(old, new, 1))
PY
  then printf '  %-38s ANCHOR MISSING (mutation not applied)\n' "$1"; return 1; fi

  out=$(python3 "$WORK/test_compute_phase_report.py" 2>&1)
  failed=$(echo "$out" | grep '^  FAIL' | sed 's/^  FAIL //' | awk '{$NF=""; print}' | cut -c1-55)
  # "aborted" means the SUITE stopped, not that a tool traceback appears inside a FAIL detail — a
  # check whose job is to catch a traceback necessarily quotes one when it fails.
  if ! echo "$out" | grep -qE 'all cases passed|case\(s\) failed'; then
    printf '  %-38s SUITE ABORTED (masks later checks)\n' "$1"; return 1
  fi
  # `--` before the pattern: a check name may legitimately start with "--" (e.g. "--csv carries the
  # stale-model banner"), which grep would otherwise parse as options and reject, printing a usage
  # error and reporting a SURVIVED that is really a harness fault.
  if echo "$failed" | grep -qF -- "$2"; then
    printf '  %-38s killed by: %s\n' "$1" "$2"
  else
    printf '  %-38s *** SURVIVED *** expected "%s", got: %s\n' \
           "$1" "$2" "$(echo "$failed" | tr '\n' ';')"
    return 1
  fi
}

bad=0
echo "per-check mutation coverage for compute_phase_report.py"
run_mutation "suppress --since-submit images" "no image share survives the filter" \
  '    images_suppressed_by_filter = bool(images) and args.since_submit is not None' \
  '    images_suppressed_by_filter = False' || bad=1
run_mutation "exclude alias bindings" "alias ms does not land in image unattributed" \
  '    images = [i for i in images if not i.get("alias", 0.0)]' \
  '    images = list(images)' || bad=1
run_mutation "alias per-binding denominator" "ms/binding uses the real-binding denominator" \
  '    images = [i for i in images if not i.get("alias", 0.0)]' \
  '    images = list(images)' || bad=1
# The mutation must have the DEFECT'S SHAPE. The historical bug returned only when EVERY record was
# an alias; a broader "return on any alias" mutation kills five checks, so the per-check assertion is
# then satisfied by a check that could not have caught the original defect.
run_mutation "no early return on all-alias" "all-alias log still prints top programs" \
  '              f"(no work, no sub-timers); nothing to decompose.")' \
  '              f"(no work, no sub-timers); nothing to decompose.")
        return 0' || bad=1
run_mutation "parent-relative unattributed" "small parent keeps its unattributed remainder" \
  '            if parent_ms and abs(rest) / parent_ms > 0.01:' \
  '            if grand and abs(rest) / grand > 0.005:' || bad=1
# Two separate filters exist: one builds `failed` (reporting), one rebuilds `records` (exclusion).
# Mutating the first only removes the EXCLUDED line; the bad record still never reaches the table.
run_mutation "report ok=0 dispatches" "failed dispatch still reported" \
  '    failed = [r for r in records if r.get("ok", 1) == 0 or "ok" not in r]' \
  '    failed = []' || bad=1
run_mutation "exclude ok=0 from phase table" "no negative time reaches the table" \
  '    records = [r for r in records if r.get("ok", 1) != 0 and "ok" in r]' \
  '    records = list(records)' || bad=1
run_mutation "stale-model banner on stdout" "stale model banner survives redirection" \
  '        print(f"  !! NOT TRUSTWORTHY: {warning}. Update the report model before using this table.")' \
  '        pass' || bad=1
run_mutation "missing-file message" "missing file is a message, not a traceback" \
  '        except OSError as error:' '        except ZeroDivisionError as error:' || bad=1
run_mutation "stale-model banner in --csv" "--csv carries the stale-model banner" \
  '            print(f"# NOT TRUSTWORTHY,{warning}")' '            pass' || bad=1
run_mutation "nest storage cache in prepare" \
  "storage cache nested without negative residual" \
  '    if not _storage_image(image):' '    if True:' || bad=1
run_mutation "prefer stable shader hash" \
  "binding rollup prefers stable hash over run-local code" \
  '            if "hash" in image:' '            if False and "hash" in image:' || bad=1
run_mutation "retain binding in rollup key" \
  "binding rollup keeps distinct bindings separate" \
  '            key = (identity, binding, image_class)' \
  '            key = (identity, None, image_class)' || bad=1
run_mutation "retain class in rollup key" \
  "binding rollup keeps image classes separate" \
  '            key = (identity, binding, image_class)' \
  '            key = (identity, binding, None)' || bad=1
run_mutation "tally upload-skipped independently" \
  "persistent and upload-skipped tallies are independent" \
  '                group["upload_yes"] += bool(image["upload_skipped"])' \
  '                group["upload_yes"] += bool(image["persistent"])' || bad=1
run_mutation "retain bounded address variants" \
  "binding rollup retains a bounded address set" \
  '            if "addr" in image:' '            if False and "addr" in image:' || bad=1
run_mutation "warn on impossible image model" \
  "impossible image model warns on stdout and stderr" \
  '    if broken_storage_nest or broken_image_root or negative_image_timers:' \
  '    if False:' || bad=1

cp "$PRISTINE" "$TOOL"
if python3 "$WORK/test_compute_phase_report.py" >/dev/null 2>&1; then
  echo "unmutated copy: suite green"
else
  echo "unmutated copy: SUITE NOT GREEN — the harness or the test is broken"; bad=1
fi
exit $bad
