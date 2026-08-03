#!/bin/bash
# Per-check mutation coverage for the pure compute-timing selector policy.
#
# A green selector test proves its fixtures ran. These mutations prove that the exact checks named
# below detect the three dangerous failure shapes: accepting the wrong stable program, calling a
# zero-match run valid, and publishing both the explicit and destructor summaries. The tracked
# source is never edited; each mutation builds a scratch copy.
set -u

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=$(mktemp -d "${TMPDIR:-/var/tmp}/compute-timing-selector-mutate-XXXXXX") || exit 2
trap 'rm -rf "$WORK"' EXIT
cp "$ROOT/frontends/shared/compute_timing_selector.hpp" \
   "$ROOT/tests/test_compute_timing_selector.cpp" "$WORK/" || exit 2
HEADER="$WORK/compute_timing_selector.hpp"
PRISTINE="$WORK/pristine.hpp"
cp "$HEADER" "$PRISTINE"
CXX_BIN=${CXX:-c++}

build_and_run() {
  "$CXX_BIN" -std=c++20 -Wall -Wextra -Werror -I"$WORK" \
    "$WORK/test_compute_timing_selector.cpp" -o "$WORK/test-selector" || return 2
  "$WORK/test-selector" 2>&1
}

run_mutation() { # $1=name $2=exact check $3=old $4=new
  cp "$PRISTINE" "$HEADER"
  python3 - "$HEADER" "$3" "$4" <<'PY' || return 1
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
  '(!selector.hash_requested || program_hash == selector.hash);' \
  '(!selector.hash_requested || (static_cast<void>(program_hash), true));' || bad=1
run_mutation "accept a zero-match run" "zero-match summary is apparatus-invalid" \
  'return counters.matched == 0;' \
  'return counters.matched != 0;' || bad=1
run_mutation "allow duplicate summaries" \
  "explicit report and destructor fallback emit exactly once" \
  'if (counters.summary_reported) return false;' \
  'if (false) return false;' || bad=1

cp "$PRISTINE" "$HEADER"
if build_and_run >/dev/null; then
  echo "unmutated copy: suite green"
else
  echo "unmutated copy: SUITE NOT GREEN"; bad=1
fi
exit "$bad"
