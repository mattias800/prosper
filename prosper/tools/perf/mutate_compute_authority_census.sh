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
   "$ROOT/frontends/shared/compute_authority_live_census.hpp" \
   "$ROOT/frontends/shared/compute_timing_selector.hpp" \
   "$ROOT/tests/test_compute_authority_census.cpp" "$WORK/" || exit 2
HEADER="$WORK/compute_authority_census.hpp"
LIVE_HEADER="$WORK/compute_authority_live_census.hpp"
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

# The live wrapper's unconditional submit close is an ordered visibility boundary, not cleanup
# bookkeeping. Misattributing it as capture has the historical defect's shape: pending authority is
# still closed, so a weak pending=0-only test would stay green. The exact named attribution check
# must be the sole kill.
cp "$ROOT/frontends/shared/compute_authority_census.hpp" "$HEADER"
cp "$ROOT/frontends/shared/compute_authority_live_census.hpp" "$LIVE_HEADER"
python3 - "$LIVE_HEADER" <<'PY' || exit 1
import sys

path = sys.argv[1]
old = """        const ShadowComputeAuthorityTransition transition = submit_.observe(
            ShadowComputeAuthorityConsumerKind::SubmitEnd);"""
new = """        const ShadowComputeAuthorityTransition transition = submit_.observe(
            ShadowComputeAuthorityConsumerKind::Capture);"""
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
    echo "submit-end attribution mutation: BUILD FAILED (mutation not observed)"
    exit 1
fi

output=$("$WORK/test-authority" 2>&1)
status=$?
failed=$(printf '%s\n' "$output" | sed -n 's/^  FAIL //p')
expected="live submit end unconditionally closes pending authority with exact attribution"
if [ "$status" -eq 0 ]; then
    echo "submit-end attribution mutation: *** SURVIVED ***"
    exit 1
fi
if [ "$failed" != "$expected" ]; then
    printf 'submit-end attribution mutation: WRONG KILL expected "%s", got: %s\n' \
        "$expected" "$failed"
    exit 1
fi
printf 'submit-end attribution mutation: killed by: %s\n' "$expected"

# A hash match is not evidence that the proposed deferred-authority lever ever moved. Remove both
# retained/admitted requirements while keeping exact selection intact; only the zero-output fixture
# may kill this defect-shaped false-positive verdict.
cp "$ROOT/frontends/shared/compute_authority_census.hpp" "$HEADER"
cp "$ROOT/frontends/shared/compute_authority_live_census.hpp" "$LIVE_HEADER"
python3 - "$LIVE_HEADER" <<'PY' || exit 1
import sys

path = sys.argv[1]
old = """            counters_.retained_storage_outputs != 0 &&
            counters_.authority.admitted_results != 0 &&"""
new = """            true &&
            true &&"""
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
    echo "zero-authority-lever mutation: BUILD FAILED (mutation not observed)"
    exit 1
fi

output=$("$WORK/test-authority" 2>&1)
status=$?
failed=$(printf '%s\n' "$output" | sed -n 's/^  FAIL //p')
expected="a matched producer with no admitted retained result is apparatus-invalid"
if [ "$status" -eq 0 ]; then
    echo "zero-authority-lever mutation: *** SURVIVED ***"
    exit 1
fi
if [ "$failed" != "$expected" ]; then
    printf 'zero-authority-lever mutation: WRONG KILL expected "%s", got: %s\n' \
        "$expected" "$failed"
    exit 1
fi
printf 'zero-authority-lever mutation: killed by: %s\n' "$expected"

# The full-submit draw probe exists specifically because the first exact realized draw did not name
# Syberia's hot atlas. Recreate the old probe's defect shape by dropping the retained range after
# its first draw while leaving the epoch otherwise healthy; only the later-draw canary may kill it.
cp "$ROOT/frontends/shared/compute_authority_census.hpp" "$HEADER"
cp "$ROOT/frontends/shared/compute_authority_live_census.hpp" "$LIVE_HEADER"
python3 - "$HEADER" <<'PY' || exit 1
import sys

path = sys.argv[1]
old = """        current_draw_overlapping_ranges_ = 0;
        return ShadowComputeAuthoritySubmitDrawProbeAction::DrawCompleted;
    }

    constexpr ShadowComputeAuthoritySubmitDrawProbeAction end_submit"""
new = """        current_draw_overlapping_ranges_ = 0;
        if (draws_in_epoch_ == 1)
            pending_range_ = ShadowComputeAuthorityRange::unknown();
        return ShadowComputeAuthoritySubmitDrawProbeAction::DrawCompleted;
    }

    constexpr ShadowComputeAuthoritySubmitDrawProbeAction end_submit"""
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
    echo "later-draw retention mutation: BUILD FAILED (mutation not observed)"
    exit 1
fi

output=$("$WORK/test-authority" 2>&1)
status=$?
failed=$(printf '%s\n' "$output" | sed -n 's/^  FAIL //p')
expected="full-submit draw probe retains atlas through a later draw until submit end"
if [ "$status" -eq 0 ]; then
    echo "later-draw retention mutation: *** SURVIVED ***"
    exit 1
fi
if [ "$failed" != "$expected" ]; then
    printf 'later-draw retention mutation: WRONG KILL expected "%s", got: %s\n' \
        "$expected" "$failed"
    exit 1
fi
printf 'later-draw retention mutation: killed by: %s\n' "$expected"

# A resource row is provisional until DrawResourceEnd proves the draw realized. Recreate the review
# defect by promoting a staged overlap from an unrealized draw into the epoch totals; only the exact
# unrealized-overlap check may kill it.
cp "$ROOT/frontends/shared/compute_authority_census.hpp" "$HEADER"
cp "$ROOT/frontends/shared/compute_authority_live_census.hpp" "$LIVE_HEADER"
python3 - "$HEADER" <<'PY' || exit 1
import sys

path = sys.argv[1]
old = """        } else {
            counters_.unrealized_resource_observations = shadow_compute_authority_add("""
new = """        } else {
            if (current_draw_overlapping_ranges_ != 0) saw_overlap_ = true;
            counters_.overlapping_ranges = shadow_compute_authority_add(
                counters_.overlapping_ranges, current_draw_overlapping_ranges_);
            counters_.unrealized_resource_observations = shadow_compute_authority_add("""
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
    echo "unrealized-overlap mutation: BUILD FAILED (mutation not observed)"
    exit 1
fi

output=$("$WORK/test-authority" 2>&1)
status=$?
failed=$(printf '%s\n' "$output" | sed -n 's/^  FAIL //p')
expected="full-submit draw probe never promotes overlap from an unrealized draw"
if [ "$status" -eq 0 ]; then
    echo "unrealized-overlap mutation: *** SURVIVED ***"
    exit 1
fi
if [ "$failed" != "$expected" ]; then
    printf 'unrealized-overlap mutation: WRONG KILL expected "%s", got: %s\n' \
        "$expected" "$failed"
    exit 1
fi
printf 'unrealized-overlap mutation: killed by: %s\n' "$expected"

# An unknown resource range can conceal the selected atlas. Remove only that fail-closed validity
# term; the exact invalid-footprint fixture must be the sole kill.
cp "$ROOT/frontends/shared/compute_authority_census.hpp" "$HEADER"
cp "$ROOT/frontends/shared/compute_authority_live_census.hpp" "$LIVE_HEADER"
python3 - "$HEADER" <<'PY' || exit 1
import sys

path = sys.argv[1]
old = """        return counters_.invalid_ranges == 0 &&
               counters_.unrealized_resource_observations == 0 &&"""
new = """        return counters_.unrealized_resource_observations == 0 &&"""
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
    echo "invalid-footprint mutation: BUILD FAILED (mutation not observed)"
    exit 1
fi

output=$("$WORK/test-authority" 2>&1)
status=$?
failed=$(printf '%s\n' "$output" | sed -n 's/^  FAIL //p')
expected="full-submit draw probe rejects an invalid resource footprint"
if [ "$status" -eq 0 ]; then
    echo "invalid-footprint mutation: *** SURVIVED ***"
    exit 1
fi
if [ "$failed" != "$expected" ]; then
    printf 'invalid-footprint mutation: WRONG KILL expected "%s", got: %s\n' \
        "$expected" "$failed"
    exit 1
fi
printf 'invalid-footprint mutation: killed by: %s\n' "$expected"

cp "$ROOT/frontends/shared/compute_authority_census.hpp" "$HEADER"
cp "$ROOT/frontends/shared/compute_authority_live_census.hpp" "$LIVE_HEADER"
if "$CXX_BIN" -std=c++20 -Wall -Wextra -Werror -I"$WORK" \
       "$WORK/test_compute_authority_census.cpp" -o "$WORK/test-authority" && \
   "$WORK/test-authority" >/dev/null
then
    echo "unmutated copy: suite green"
else
    echo "unmutated copy: SUITE NOT GREEN"
    exit 1
fi
