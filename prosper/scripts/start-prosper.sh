#!/usr/bin/env bash
# Launch a title with the native Linux prosper-app frontend.
#
# This is the Linux counterpart of scripts/start-prosper.ps1 and ships at the root of the release
# tarball. It sets the two environment switches the frame loop currently needs (there is no settings
# UI yet) and then execs the packaged binary.
#
#   ./start-prosper.sh ~/ps5/PPSA24651-app0
#   ./start-prosper.sh --test-pattern --frames 120
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Beside the script in a release tree (usr/bin/prosper-app), or beside it directly when someone has
# copied the binary next to the launcher.
app=""
for candidate in "$here/usr/bin/prosper-app" "$here/prosper-app"; do
    if [ -x "$candidate" ]; then app="$candidate"; break; fi
done
[ -n "$app" ] || {
    echo "start-prosper.sh: prosper-app not found beside this script" >&2
    exit 1
}

dump=""
savedata_dir="$here/savedata"
guest_args="-force-gfx-direct"
record=""
record_axis="flip"
frames=0
present_mode="fifo"
test_pattern=0

usage() {
    cat >&2 <<'EOF'
Usage: start-prosper.sh [options] <app0-directory>
       start-prosper.sh --test-pattern [--frames N]

  --savedata-dir <dir>   where guest saves live (default: ./savedata beside this script)
  --guest-args <args>    guest argv (default: -force-gfx-direct; pass '' to send none)
  --present-mode <mode>  fifo (default), mailbox, or immediate
  --frames <n>           present N frames then exit
  --record <path>        record controller/keyboard input as a replay route
  --record-axis <axis>   flip (default) or pad-read; requires --record
  --test-pattern         render the built-in pattern instead of booting a title
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --savedata-dir)  savedata_dir="$2"; shift 2 ;;
        --guest-args)    guest_args="$2";   shift 2 ;;
        # Lower-cased to match start-prosper.ps1, whose [ValidateSet] is case-insensitive and
        # which passes .ToLowerInvariant() to the app: `--present-mode FIFO` must not be an error
        # on Linux when `-PresentMode FIFO` works on Windows.
        --present-mode)  present_mode="$(printf '%s' "$2" | tr '[:upper:]' '[:lower:]')"; shift 2 ;;
        --frames)        frames="$2";       shift 2 ;;
        --record)        record="$2";       shift 2 ;;
        --record-axis)   record_axis="$(printf '%s' "$2" | tr '[:upper:]' '[:lower:]')"; shift 2 ;;
        --test-pattern)  test_pattern=1;    shift ;;
        -h|--help)       usage; exit 0 ;;
        -*) echo "start-prosper.sh: unknown option: $1" >&2; usage; exit 2 ;;
        *)  if [ -n "$dump" ]; then
                echo "start-prosper.sh: pass one app0 directory, not several ('$dump' then '$1')" >&2
                exit 2
            fi
            dump="$1"; shift ;;
    esac
done

case "$present_mode" in
    fifo|mailbox|immediate) ;;
    *) echo "start-prosper.sh: --present-mode must be fifo, mailbox or immediate" >&2; exit 2 ;;
esac
case "$record_axis" in
    flip|pad-read) ;;
    *) echo "start-prosper.sh: --record-axis must be flip or pad-read" >&2; exit 2 ;;
esac
if [ -z "$record" ] && [ "$record_axis" != "flip" ]; then
    echo "start-prosper.sh: --record-axis requires --record" >&2
    exit 2
fi

args=()
if [ "$test_pattern" -eq 1 ]; then
    args+=(--test-pattern)
else
    [ -n "$dump" ] || { echo "start-prosper.sh: pass an app0 directory, or use --test-pattern" >&2
                        usage; exit 2; }
    [ -d "$dump" ] || { echo "start-prosper.sh: not a directory: $dump" >&2; exit 2; }
    dump="$(cd "$dump" && pwd)"
    mkdir -p "$savedata_dir"
    savedata_dir="$(cd "$savedata_dir" && pwd)"
    # PROSPER_GUEST_FS is not set: it is read at guest_tls.cpp:46 only under
    # `#ifdef __APPLE__`. On Linux (`:58`) guest TLS is ON by default and the variable
    # that exists is the opt-OUT PROSPER_NO_GUEST_FS (#2098).
    export PROSPER_GUEST_ARGS="$guest_args"
    export PROSPER_SAVEDATA_DIR="$savedata_dir"
    args+=(--dump "$dump")
fi

# `cond && cmd` as a statement would abort the script under `set -e` whenever the condition is
# simply false, which for every one of these is the DEFAULT. They stay as `if` blocks.
case "$frames" in
    ''|*[!0-9]*) echo "start-prosper.sh: --frames takes a number" >&2; exit 2 ;;
esac
if [ "$frames" -gt 0 ]; then
    args+=(--frames "$frames")
fi
if [ "$present_mode" != "fifo" ]; then
    args+=(--present-mode "$present_mode")
fi
if [ -n "$record" ]; then
    mkdir -p "$(dirname "$record")"
    args+=(--record "$record")
    if [ "$record_axis" != "flip" ]; then
        args+=(--record-axis "$record_axis")
    fi
fi

exec "$app" "${args[@]}"
