#!/bin/bash
# Per-check mutation coverage for the pure compute-timing selector policy.
#
# A green selector test proves its fixtures ran. These mutations prove that the exact checks named
# below detect the dangerous failure shapes: accepting the wrong stable program, calling a
# zero-match run valid, publishing both the explicit and destructor summaries, ignoring a raw
# cache-eligibility gate, or declaring a transfer comparison valid when its gate was never
# observed. The tracked source is never edited; each mutation builds a scratch copy.
set -u

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=$(mktemp -d "${TMPDIR:-/var/tmp}/compute-timing-selector-mutate-XXXXXX") || exit 2
trap 'rm -rf "$WORK"' EXIT
cp "$ROOT/frontends/shared/compute_timing_selector.hpp" \
   "$ROOT/frontends/shared/compute_transfer_gate_census.hpp" \
   "$ROOT/tests/test_compute_timing_selector.cpp" "$WORK/" || exit 2
TIMING_HEADER="$WORK/compute_timing_selector.hpp"
TRANSFER_HEADER="$WORK/compute_transfer_gate_census.hpp"
TIMING_PRISTINE="$WORK/compute_timing_selector.pristine.hpp"
TRANSFER_PRISTINE="$WORK/compute_transfer_gate_census.pristine.hpp"
cp "$TIMING_HEADER" "$TIMING_PRISTINE"
cp "$TRANSFER_HEADER" "$TRANSFER_PRISTINE"
CXX_BIN=${CXX:-c++}

build_and_run() {
  "$CXX_BIN" -std=c++20 -Wall -Wextra -Werror -I"$WORK" \
    "$WORK/test_compute_timing_selector.cpp" -o "$WORK/test-selector" || return 2
  "$WORK/test-selector" 2>&1
}

restore_headers() {
  cp "$TIMING_PRISTINE" "$TIMING_HEADER"
  cp "$TRANSFER_PRISTINE" "$TRANSFER_HEADER"
}

run_mutation() { # $1=name $2=exact check $3=header $4=old $5=new
  restore_headers
  python3 - "$3" "$4" "$5" <<'PY' || return 1
import sys
path, old, new = sys.argv[1:]
with open(path, encoding="utf-8") as stream:
    source = stream.read()
if source.count(old) != 1:
    raise SystemExit(f"anchor count is {source.count(old)}, expected exactly one")
with open(path, "w", encoding="utf-8") as stream:
    stream.write(source.replace(old, new, 1))
PY
  output=$(build_and_run)
  status=$?
  if [ "$status" -eq 2 ]; then
    printf '  %-34s BUILD FAILED (mutation not observed)\n' "$1"
    return 1
  fi
  failed=$(printf '%s\n' "$output" | sed -n 's/^  FAIL //p')
  if [ "$status" -eq 0 ]; then
    printf '  %-34s *** SURVIVED ***\n' "$1"
    return 1
  fi
  if [ "$failed" = "$2" ]; then
    printf '  %-34s killed by: %s\n' "$1" "$2"
    return 0
  fi
  printf '  %-34s WRONG KILL expected "%s", got: %s\n' "$1" "$2" "$failed"
  return 1
}

bad=0
echo "per-check mutation coverage for compute timing selector"
run_mutation "accept mismatched stable hash" "stable hash mismatches never select" \
  "$TIMING_HEADER" \
  '(!selector.hash_requested || program_hash == selector.hash);' \
  '(!selector.hash_requested || (static_cast<void>(program_hash), true));' || bad=1
run_mutation "accept a zero-match run" "zero-match summary is apparatus-invalid" \
  "$TIMING_HEADER" \
  'return counters.matched == 0;' \
  'return counters.matched != 0;' || bad=1
run_mutation "allow duplicate summaries" \
  "explicit report and destructor fallback emit exactly once" \
  "$TIMING_HEADER" \
  'if (counters.summary_reported) return false;' \
  'if (false) return false;' || bad=1
run_mutation "ignore renderer ownership gate" \
  "renderer ownership blocks storage cache candidate" \
  "$TRANSFER_HEADER" \
  'return !inputs.renderer_owned && inputs.dcc_cache_safe &&' \
  'return inputs.dcc_cache_safe &&' || bad=1
run_mutation "accept unobserved storage gates" \
  "one-sided storage gate observation is apparatus-invalid" \
  "$TRANSFER_HEADER" \
  '            counters.consumer_storage_gate_observations == 0);' \
  '            false);' || bad=1

restore_headers
if build_and_run >/dev/null; then
  echo "unmutated copy: suite green"
else
  echo "unmutated copy: SUITE NOT GREEN"; bad=1
fi
exit "$bad"
