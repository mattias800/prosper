#!/usr/bin/env bash
# Print the highest glibc symbol version required by a packaged tree -- the oldest distribution the
# artifact can run on.
#
# This is the one portability property bundling CANNOT fix: no AppImage ships libc, so every binary
# and every bundled library keeps the versioned glibc symbols it was linked against, and the machine
# that BUILT the artifact sets the floor for everyone who runs it. Reported as a plain `2.NN` on
# stdout so BUILD.txt can state it and verify-linux-app.sh can gate on it.
#
# Non-numeric GLIBC_ABI_* tags (e.g. GLIBC_ABI_GNU2_TLS) are printed to stderr. They are a floor too,
# but not one that sorts against `2.NN`, so callers gate on them separately rather than folding them
# into the number.
#
# Usage: glibc-floor.sh <dir-or-file>
set -euo pipefail

target="${1:-}"
[ -n "$target" ] || { echo "glibc-floor.sh: pass a directory or file" >&2; exit 2; }

scan() {
    # `objdump -T` on a non-ELF file is an error, not a finding: ignore it rather than abort the scan.
    objdump -T "$1" 2>/dev/null || true
}

files=()
if [ -d "$target" ]; then
    while IFS= read -r -d '' f; do files+=("$f"); done < <(
        find "$target" -type f \( -name 'prosper-app' -o -name '*.so' -o -name '*.so.*' \) -print0)
else
    files=("$target")
fi

[ "${#files[@]}" -gt 0 ] || { echo "glibc-floor.sh: nothing to scan in $target" >&2; exit 1; }

all=""
for f in "${files[@]}"; do
    all+="$(scan "$f")"$'\n'
done

# Numeric floor. sort -V orders 2.9 below 2.34 correctly, which a lexical sort does not.
# `|| true` because pipefail would otherwise turn "this tree needs no versioned glibc symbol" -- a
# legitimate result reported below as `unknown` -- into a hard abort on grep's exit 1.
floor="$( { printf '%s' "$all" \
    | grep -oE 'GLIBC_[0-9]+\.[0-9]+(\.[0-9]+)?' \
    | sed 's/^GLIBC_//' \
    | sort -V -u \
    | tail -1; } || true)"

# grep exits 1 when a tree happens to need no versioned glibc symbol at all; that is a valid result,
# not a failure, so this is deliberately not part of an && chain.
abi_tags="$(printf '%s' "$all" | grep -oE 'GLIBC_ABI_[A-Za-z0-9_]+' | sort -u || true)"
if [ -n "$abi_tags" ]; then
    printf 'glibc-floor: non-numeric ABI tags required: %s\n' "$(echo "$abi_tags" | tr '\n' ' ')" >&2
fi

if [ -z "$floor" ]; then
    echo "unknown"
    exit 0
fi
echo "$floor"
