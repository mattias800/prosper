#!/bin/bash
# Per-check mutation coverage for the behavior-neutral compute-authority census.
#
# The mutation has the defect's exact shape: only an overlapping raw buffer is mislabeled as a
# proven GPU consumer. It must be killed by the named raw-overlap positive control and no sibling.
# The tracked source is never edited; the script builds a scratch copy.
set -u

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=$(mktemp -d "${TMPDIR:-/var/tmp}/compute-authority-mutate-XXXXXX") || exit 2
trap 'rm -rf "$WORK"' EXIT
cp "$ROOT/frontends/shared/compute_authority_census.hpp" \
   "$ROOT/tests/test_compute_authority_census.cpp" "$WORK/" || exit 2
HEADER="$WORK/compute_authority_census.hpp"
CXX_BIN=${CXX:-c++}

python3 - "$HEADER" <<'PY' || exit 1
import sys

path = sys.argv[1]
old = """        case ShadowComputeAuthorityConsumerKind::RawBuffer:
            return ShadowComputeAuthorityAccess::GuestMirror;"""
new = """        case ShadowComputeAuthorityConsumerKind::RawBuffer:
            return ShadowComputeAuthorityAccess::ProvenGpu;"""
with open(path, encoding="utf-8") as stream:
    source = stream.read()
if source.count(old) != 1:
    raise SystemExit(f"mutation anchor count is {source.count(old)}, expected exactly one")
with open(path, "w", encoding="utf-8") as stream:
    stream.write(source.replace(old, new, 1))
PY

if ! "$CXX_BIN" -std=c++20 -Wall -Wextra -Werror -I"$WORK" \
    "$WORK/test_compute_authority_census.cpp" -o "$WORK/test-authority"
then
    echo "raw overlap mutation: BUILD FAILED (mutation not observed)"
    exit 1
fi

output=$("$WORK/test-authority" 2>&1)
status=$?
failed=$(printf '%s\n' "$output" | sed -n 's/^  FAIL //p')
expected="overlapping raw-buffer consumer forces guest materialization"
if [ "$status" -eq 0 ]; then
    echo "raw overlap mutation: *** SURVIVED ***"
    exit 1
fi
if [ "$failed" != "$expected" ]; then
    printf 'raw overlap mutation: WRONG KILL expected "%s", got: %s\n' \
        "$expected" "$failed"
    exit 1
fi
printf 'raw overlap mutation: killed by: %s\n' "$expected"

cp "$ROOT/frontends/shared/compute_authority_census.hpp" "$HEADER"
if "$CXX_BIN" -std=c++20 -Wall -Wextra -Werror -I"$WORK" \
       "$WORK/test_compute_authority_census.cpp" -o "$WORK/test-authority" && \
   "$WORK/test-authority" >/dev/null
then
    echo "unmutated copy: suite green"
else
    echo "unmutated copy: SUITE NOT GREEN"
    exit 1
fi
