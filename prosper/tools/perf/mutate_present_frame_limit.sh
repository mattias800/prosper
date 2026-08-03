#!/usr/bin/env bash
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK=$(mktemp -d "${TMPDIR:-/var/tmp}/present-frame-limit-mutate-XXXXXX") || exit 2
trap 'rm -rf "$WORK"' EXIT

cp "$ROOT/frontends/prosper-app/present_policy.hpp" \
   "$ROOT/frontends/prosper-app/test_present_policy.cpp" "$WORK/" || exit 2

python3 - "$WORK/present_policy.hpp" <<'PY' || exit 1
import sys

path = sys.argv[1]
old = """        (void)source;
        ++count_;"""
new = """        if (source == PresentedFrameSource::GpuCpuFallback) {
            ++count_;
            return;
        }
        ++count_;"""
with open(path, encoding="utf-8") as stream:
    source = stream.read()
if source.count(old) != 1:
    raise SystemExit(f"mutation anchor count is {source.count(old)}, expected exactly one")
with open(path, "w", encoding="utf-8") as stream:
    stream.write(source.replace(old, new, 1))
PY

CXX_BIN=${CXX:-c++}
VULKAN_CFLAGS=$(pkg-config --cflags vulkan 2>/dev/null || true)
if ! "$CXX_BIN" -std=c++20 -Wall -Wextra -Werror $VULKAN_CFLAGS -I"$WORK" \
        "$WORK/test_present_policy.cpp" -o "$WORK/test-present-policy"; then
    echo "CPU-fallback frame-limit mutation: BUILD FAILED (mutation not observed)"
    exit 1
fi

output=$("$WORK/test-present-policy" 2>&1)
status=$?
failed=$(printf '%s\n' "$output" | sed -n 's/^FAIL //p')
expected="GPU CPU fallback honors the same presented-frame limit"
if [ "$status" -eq 0 ]; then
    echo "CPU-fallback frame-limit mutation: *** SURVIVED ***"
    exit 1
fi
if [ "$failed" != "$expected" ]; then
    printf 'CPU-fallback frame-limit mutation: WRONG KILL expected "%s", got: %s\n' \
        "$expected" "$failed"
    exit 1
fi
printf 'CPU-fallback frame-limit mutation: killed by: %s\n' "$expected"
