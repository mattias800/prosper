#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "usage: $0 DUMP_DIR [NEW_RUN_DIR]" >&2
}

if (( $# < 1 || $# > 2 )); then
    usage
    exit 2
fi

script_dir="$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)"
repo_root="$(CDPATH='' cd -- "$script_dir/../.." && pwd)"
dump_dir="$1"
app="${PROSPER_APP:-$repo_root/build-linux/prosper-app}"

if [[ ! -d "$dump_dir" ]]; then
    echo "Plucky Squire dump directory does not exist: $dump_dir" >&2
    exit 2
fi
if [[ ! -x "$app" ]]; then
    echo "prosper-app is not executable: $app" >&2
    exit 2
fi
app="$(CDPATH='' cd -- "$(dirname -- "$app")" && pwd)/$(basename -- "$app")"

if (( $# == 2 )); then
    run_dir="$2"
    if ! mkdir -- "$run_dir"; then
        echo "NEW_RUN_DIR must not already exist: $run_dir" >&2
        exit 2
    fi
else
    user_state_root="${XDG_STATE_HOME:-${HOME:?HOME must be set}/.local/state}"
    artifact_root="${PROSPER_ARTIFACT_ROOT:-$user_state_root/prosper-plucky}"
    mkdir -p -- "$artifact_root"
    run_dir="$(mktemp -d "$artifact_root/first-gameplay.XXXXXX")"
fi

mkdir -- "$run_dir/save0" "$run_dir/savedata"
run_dir="$(CDPATH='' cd -- "$run_dir" && pwd)"
dump_dir="$(CDPATH='' cd -- "$dump_dir" && pwd)"

export PROSPER_GUEST_FS=1
export PROSPER_GUEST_ARGS="${PROSPER_GUEST_ARGS:--force-gfx-direct}"
export PROSPER_RENDER=1
export PROSPER_RENDER_EVERY=1
export PROSPER_RENDER_SCALE=1
export PROSPER_PAD_SCRIPT=@scripts/plucky-squire/reach-first-gameplay.pad
export PROSPER_CAPTURE_TITLE=PPSA15319
export PROSPER_SAVE0="$run_dir/save0"
export PROSPER_SAVEDATA_DIR="$run_dir/savedata"

printf 'Plucky Squire fresh route\n'
printf '  run directory: %s\n' "$run_dir"
printf '  SAVE0 file-mount root: %s\n' "$PROSPER_SAVE0"
printf '  SaveDataMemory root: %s\n' "$PROSPER_SAVEDATA_DIR"

cd "$repo_root"
exec "$app" --dump "$dump_dir"
