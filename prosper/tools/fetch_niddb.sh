#!/usr/bin/env bash
# Fetch the NID symbol-name list (idc/ps4libdoc) used to name imports in traces.
# Not redistributed with prosper; run this once. Output: prosper/data/known_names.txt
# The NidDb loads it automatically if present (else it falls back to a built-in list).
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$here/data"
if command -v gh >/dev/null 2>&1; then
  gh api repos/idc/ps4libdoc/contents/known_names.txt -H "Accept: application/vnd.github.raw" > "$here/data/known_names.txt"
else
  curl -fsSL https://raw.githubusercontent.com/idc/ps4libdoc/master/known_names.txt -o "$here/data/known_names.txt"
fi
echo "wrote $here/data/known_names.txt ($(wc -l < "$here/data/known_names.txt") names)"
