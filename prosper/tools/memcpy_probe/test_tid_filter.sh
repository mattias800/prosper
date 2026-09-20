#!/usr/bin/env bash
# Native controls for the opt-in TID filter.  Builds only the shim/control, never a game.
set -euo pipefail

root=$(cd -- "$(dirname -- "$0")" && pwd)
scratch_base="$root/../../build-memcpy-probe-tid-tests"
mkdir -p "$scratch_base"
work=$(mktemp -d "$scratch_base/run.XXXXXX")
trap 'rm -rf -- "$work"' EXIT
export TMPDIR="$work"

cc -shared -fPIC -O2 -Wall -Wextra -Werror -o "$work/probe.so" "$root/memcpy_probe.c" -ldl -lpthread
cc -O2 -Wall -Wextra -Werror -o "$work/control" "$root/probe_control.c" -ldl -lpthread

run_default="$work/default"
mkdir "$run_default"
LD_PRELOAD="$work/probe.so" PROSPER_MEMCPY_PROBE_DIR="$run_default" "$work/control" > "$run_default/control.txt"
default_report=$(find "$run_default" -name 'memcpy-sites-*.txt' -print -quit)
grep -qx 'total_calls=420000 total_bytes=4272640000 total_cycles=[0-9]* overflow_calls=0' "$default_report"
grep -qx '# return_address cycles calls bytes dso dso_offset nearest_symbol' "$default_report"

run_arm() {
    local index=$1 dir="$work/tid$1"
    mkdir "$dir"
    LD_PRELOAD="$work/probe.so" PROSPER_MEMCPY_PROBE_DIR="$dir" \
        PROSPER_MEMCPY_PROBE_TID_FILE="$dir/tid" "$work/control" --threaded "$dir/tid" "$index" > "$dir/control.txt"
    local report selected observed
    report=$(find "$dir" -name 'memcpy-sites-*.txt' -print -quit)
    selected=$(sed -n 's/.* selected=\([0-9][0-9]*\) .*/\1/p' "$dir/control.txt")
    observed=$(sed -n 's/.* observed_tid=\([0-9][0-9]*\) .*/\1/p' "$report")
    test "$selected" = "$observed"
    grep -q "target_tid=$selected .*total_calls=50000 .*overflow_calls=0" "$report"
    awk -v tid="$selected" '$1 ~ /^[0-9]+$/ { if ($1 != tid) exit 1; calls += $4 } END { exit calls == 50000 ? 0 : 1 }' "$report"
    awk '$1 ~ /^[0-9]+$/ { print $7 }' "$report" | sort > "$dir/offsets.txt"
}

run_arm 0
run_arm 1
cmp "$work/tid0/offsets.txt" "$work/tid1/offsets.txt"

check_empty_report() {
    local report=$1 prefix=$2
    grep -q "^$prefix" "$report"
    grep -q ' total_calls=0 total_bytes=0 total_cycles=0 overflow_calls=0$' "$report"
    ! awk '$1 ~ /^[0-9]+$/ { found = 1 } END { exit found ? 0 : 1 }' "$report"
}

for invalid in bad 0 123junk; do
    run_invalid="$work/invalid-direct-$invalid"
    mkdir "$run_invalid"
    LD_PRELOAD="$work/probe.so" PROSPER_MEMCPY_PROBE_DIR="$run_invalid" \
        PROSPER_MEMCPY_PROBE_TID="$invalid" "$work/control" > "$run_invalid/control.txt"
    invalid_report=$(find "$run_invalid" -name 'memcpy-sites-*.txt' -print -quit)
    check_empty_report "$invalid_report" 'probe_mode=invalid-tid-filter target_tid=0 observed_tid=0 '
done

check_invalid_file() {
    local name=$1 contents=$2
    local dir="$work/invalid-file-$name"
    mkdir "$dir"
    printf '%s' "$contents" > "$dir/tid"
    LD_PRELOAD="$work/probe.so" PROSPER_MEMCPY_PROBE_DIR="$dir" \
        PROSPER_MEMCPY_PROBE_TID_FILE="$dir/tid" "$work/control" > "$dir/control.txt"
    local report
    report=$(find "$dir" -name 'memcpy-sites-*.txt' -print -quit)
    check_empty_report "$report" 'probe_mode=tid-file-wait target_tid=0 observed_tid=0 '
    grep -Eq ' invalid_tid_file_reads=[1-9][0-9]* ' "$report"
}

check_invalid_file junk $'123\njunk\n'
check_invalid_file truncated "$(printf '%070d\n' 123)"

run_invalid_nul_file="$work/invalid-file-nul"
mkdir "$run_invalid_nul_file"
printf '123\0junk\n' > "$run_invalid_nul_file/tid"
LD_PRELOAD="$work/probe.so" PROSPER_MEMCPY_PROBE_DIR="$run_invalid_nul_file" \
    PROSPER_MEMCPY_PROBE_TID_FILE="$run_invalid_nul_file/tid" "$work/control" > "$run_invalid_nul_file/control.txt"
invalid_nul_report=$(find "$run_invalid_nul_file" -name 'memcpy-sites-*.txt' -print -quit)
check_empty_report "$invalid_nul_report" 'probe_mode=tid-file-wait target_tid=0 observed_tid=0 '
grep -Eq ' invalid_tid_file_reads=[1-9][0-9]* ' "$invalid_nul_report"

echo "memcpy_probe TID-filter controls: PASS"
