#!/usr/bin/env bash
set -euo pipefail
tool_dir=$(cd "$(dirname "$0")" && pwd)
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT

c++ -std=c++17 -O2 -fPIC -shared -fno-optimize-sibling-calls \
    "$tool_dir/new_probe.cpp" -ldl -pthread -o "$scratch/new_probe.so"
c++ -std=c++17 -O2 -g -no-pie -rdynamic -fno-inline -fno-ipa-icf \
    "$tool_dir/probe_control.cpp" -pthread -o "$scratch/control"

LD_PRELOAD="$scratch/new_probe.so" PROSPER_NEW_PROBE_DIR="$scratch" \
    "$scratch/control" > "$scratch/control.txt"
report=$(find "$scratch" -maxdepth 1 -name 'new-sites-*.tsv' -print -quit)
test -n "$report"
python3 "$tool_dir/verify_report.py" "$scratch/control.txt" "$report" --elf EXEC

# The app harness commonly ends through `timeout`/SIGTERM, bypassing exit
# destructors. Require a report from a still-running control before killing it.
mkdir "$scratch/periodic"
LD_PRELOAD="$scratch/new_probe.so" PROSPER_NEW_PROBE_DIR="$scratch/periodic" \
    "$scratch/control" sleep > "$scratch/periodic-control.txt" &
control_pid=$!
sleep 2
periodic="$scratch/periodic/new-sites-$control_pid.tsv"
kill -0 "$control_pid"
test -s "$periodic"
python3 "$tool_dir/verify_report.py" "$scratch/periodic-control.txt" "$periodic" --elf EXEC
kill -TERM "$control_pid"
wait "$control_pid" || test "$?" -eq 143
printf '%s\n' 'PASS: complete content exists before SIGTERM'

# PIE must use the load-relative offset, and a path containing spaces must
# remain a single TSV field rather than shifting every later column.
mkdir "$scratch/with spaces"
c++ -std=c++17 -O2 -g -fPIE -pie -rdynamic -fno-inline -fno-ipa-icf \
    "$tool_dir/probe_control.cpp" -pthread -o "$scratch/with spaces/control pie"
LD_PRELOAD="$scratch/new_probe.so" PROSPER_NEW_PROBE_DIR="$scratch/with spaces" \
    "$scratch/with spaces/control pie" > "$scratch/pie-control.txt"
pie_report=$(find "$scratch/with spaces" -maxdepth 1 -name 'new-sites-*.tsv' -print -quit)
python3 "$tool_dir/verify_report.py" "$scratch/pie-control.txt" "$pie_report" \
    --elf DYN --space-path

# A literal tab in the executable path must be escaped inside its TSV field.
tab_dir="$scratch/with"$'\t'"tab"
mkdir "$tab_dir"
cp "$scratch/with spaces/control pie" "$tab_dir/control"
LD_PRELOAD="$scratch/new_probe.so" PROSPER_NEW_PROBE_DIR="$tab_dir" \
    "$tab_dir/control" > "$scratch/tab-control.txt"
tab_report=$(find "$tab_dir" -maxdepth 1 -name 'new-sites-*.tsv' -print -quit)
python3 "$tool_dir/verify_report.py" "$scratch/tab-control.txt" "$tab_report" \
    --elf DYN --tab-path

# A fork without exec inherits the parent's counters and loses its reporter
# thread. The child must refuse attribution rather than publish those counts.
mkdir "$scratch/fork"
LD_PRELOAD="$scratch/new_probe.so" PROSPER_NEW_PROBE_DIR="$scratch/fork" \
    "$scratch/control" fork > "$scratch/fork-control.txt" 2> "$scratch/fork-stderr.txt"
python3 - "$scratch/fork-control.txt" "$scratch/fork-stderr.txt" "$scratch/fork" <<'PY'
import re
import sys
from pathlib import Path
control = Path(sys.argv[1]).read_text()
match = re.search(r"^fork_child_pid=(\d+)$", control, re.MULTILINE)
assert match, control
child_report = Path(sys.argv[3]) / f"new-sites-{match[1]}.tsv"
assert child_report.read_text().startswith("# refused=forked-child ")
assert "REFUSED forked child without exec" in Path(sys.argv[2]).read_text()
print("PASS: forked child refuses inherited attribution")
PY

# A missing output directory must be a visible failure, not a silent zero-row
# result that could be mistaken for a run with no allocations.
LD_PRELOAD="$scratch/new_probe.so" PROSPER_NEW_PROBE_DIR="$scratch/no such dir" \
    "$scratch/control" > "$scratch/failure-control.txt" 2> "$scratch/failure-stderr.txt"
python3 - "$scratch/failure-stderr.txt" <<'PY'
import sys
from pathlib import Path
message = Path(sys.argv[1]).read_text()
assert "REPORT FAILED at open" in message, message
print("PASS: report-write failure is visible")
PY

# Force saturation with the same code path, then ensure the report cannot
# silently claim all calls were attributed to the surviving entries.
c++ -std=c++17 -O2 -fPIC -shared -fno-optimize-sibling-calls \
    -DPROSPER_NEW_PROBE_TEST_SLOTS=2 "$tool_dir/new_probe.cpp" \
    -ldl -pthread -o "$scratch/new_probe_tiny.so"
mkdir "$scratch/overflow"
LD_PRELOAD="$scratch/new_probe_tiny.so" PROSPER_NEW_PROBE_DIR="$scratch/overflow" \
    "$scratch/control" > "$scratch/overflow-control.txt"
overflow_report=$(find "$scratch/overflow" -maxdepth 1 -name 'new-sites-*.tsv' -print -quit)
python3 - "$overflow_report" <<'PY'
import re
import sys
header = open(sys.argv[1], encoding="utf-8").readline()
calls = re.search(r"\boverflow_calls=(\d+)", header)
size = re.search(r"\boverflow_bytes=(\d+)", header)
assert calls and int(calls[1]) > 0, header
assert size and int(size[1]) > 0, header
print("PASS: bounded table exposes overflow calls and bytes")
PY
