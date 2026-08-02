#!/usr/bin/env bash
# Gate the Linux release artifacts produced by package-linux-app.sh.
#
# This runs on every pull request, not only on tags, because the failure this guards against is
# silent: a release archive whose binary cannot start is indistinguishable from a good one until a
# user downloads it. Everything here is asserted against the artifacts themselves -- extracted from
# the archive, never from the build tree.
#
# The runtime arms use `--list-games`, the one prosper-app path that is documented headless (no
# window, no Vulkan, no guest), so the packaged binary really is executed rather than merely
# inspected. Three arms are run, not one: a missing directory (exit 2), an empty directory (exit 1),
# and a directory holding one title (exit 0, one record on stdout). A single arm would pass just as
# well if the binary always failed the same way, so the arms exist to make that indistinguishable
# case impossible.
#
# Usage: verify-linux-app.sh --out-dir <dir> [--max-glibc <2.NN>]
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

out_dir=""
# The runner this is built on (see the Linux App Package CI job). Raising this is a deliberate act:
# it drops every user whose distribution ships an older glibc, and nothing else in the pipeline can
# see that happen.
max_glibc="2.39"

while [ $# -gt 0 ]; do
    case "$1" in
        --out-dir)    out_dir="$2";    shift 2 ;;
        --max-glibc)  max_glibc="$2";  shift 2 ;;
        *) echo "verify-linux-app.sh: unknown argument: $1" >&2; exit 2 ;;
    esac
done

[ -n "$out_dir" ] || { echo "verify-linux-app.sh: --out-dir is required" >&2; exit 2; }
out_dir="$(cd "$out_dir" && pwd)"

fail() { echo "verify: FAIL: $*" >&2; exit 1; }
ok()   { echo "verify: ok: $*"; }

tarball="$out_dir/prosper-linux-x86_64.tar.gz"
appimage="$out_dir/prosper-linux-x86_64.AppImage"

for f in "$tarball" "$tarball.sha256" "$appimage" "$appimage.sha256"; do
    [ -f "$f" ] || fail "missing artifact: $(basename "$f")"
done
ok "all four artifacts present"

# --- checksums -------------------------------------------------------------------------------
( cd "$out_dir" && sha256sum -c prosper-linux-x86_64.tar.gz.sha256 >/dev/null ) \
    || fail "tarball sha256 does not match"
( cd "$out_dir" && sha256sum -c prosper-linux-x86_64.AppImage.sha256 >/dev/null ) \
    || fail "AppImage sha256 does not match"
ok "both sha256 files verify against their archives"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

# --- fixtures for the runtime arms -----------------------------------------------------------
# A title root is a directory holding eboot.bin (frontends/prosper-app/game_path.hpp), so an empty
# file of that name is enough to make the library scan report exactly one title.
mkdir -p "$work/games-empty" "$work/games-one/FAKE00001-app0"
: > "$work/games-one/FAKE00001-app0/eboot.bin"

smoke() {
    # $1 = label, rest = the command that runs prosper-app
    local label="$1"; shift
    local rc out

    rc=0; out="$("$@" --list-games --games-dir "$work/does-not-exist" 2>&1)" || rc=$?
    [ "$rc" -eq 2 ] || fail "$label: missing games dir should exit 2, got $rc ($out)"
    grep -q 'not a directory' <<<"$out" || fail "$label: missing games dir did not say so: $out"

    rc=0; out="$("$@" --list-games --games-dir "$work/games-empty" 2>&1)" || rc=$?
    [ "$rc" -eq 1 ] || fail "$label: empty games dir should exit 1, got $rc ($out)"
    grep -q '0 title(s)' <<<"$out" || fail "$label: empty games dir did not report 0 titles: $out"

    rc=0; out="$("$@" --list-games --games-dir "$work/games-one" 2>&1)" || rc=$?
    [ "$rc" -eq 0 ] || fail "$label: one-title games dir should exit 0, got $rc ($out)"
    grep -q '1 title(s)' <<<"$out" || fail "$label: one-title games dir did not report 1 title: $out"
    grep -q 'FAKE00001-app0' <<<"$out" || fail "$label: the title record is missing: $out"

    ok "$label: ran headless and distinguished all three library states (exit 2 / 1 / 0)"
}

# --- tarball ---------------------------------------------------------------------------------
tar -xzf "$tarball" -C "$work" || fail "tarball does not extract"
tree="$work/prosper-linux-x86_64"
[ -d "$tree" ] || fail "tarball did not extract to prosper-linux-x86_64/"
bin="$tree/usr/bin/prosper-app"
[ -x "$bin" ] || fail "no executable at usr/bin/prosper-app"
file "$bin" | grep -q 'ELF 64-bit.*x86-64' || fail "usr/bin/prosper-app is not an x86-64 ELF"
for extra in README.md BUILD.txt start-prosper.sh AppRun; do
    [ -e "$tree/$extra" ] || fail "tarball is missing $extra"
done
[ -x "$tree/start-prosper.sh" ] || fail "start-prosper.sh is not executable"
ok "tarball extracts with the binary, launcher, README, BUILD.txt and AppRun"

# The binary must find its bundled libraries by RUNPATH alone. Anything else means the tree only
# works where it was built.
runpath="$(objdump -p "$bin" | awk '/RUNPATH|RPATH/ {print $2; exit}')"
[ "$runpath" = '$ORIGIN/../lib' ] || fail "RUNPATH is '$runpath', expected \$ORIGIN/../lib"
ok "RUNPATH is \$ORIGIN/../lib"

# --- the dependency assertions ---------------------------------------------------------------
# Deliberately unset LD_LIBRARY_PATH: inheriting the build environment's would let a system copy
# satisfy a library the archive is supposed to carry, and the check would pass for the wrong reason.
deps="$(env -u LD_LIBRARY_PATH ldd "$bin")"
if grep -q 'not found' <<<"$deps"; then
    echo "$deps" >&2
    fail "the packaged binary has unresolved libraries"
fi
ok "ldd resolves every dependency with no 'not found'"

# The whole reason this format exists: the FFmpeg/libva sonames differ on every distribution, so
# they must come from usr/lib inside the archive and never from the host.
for soname in libavcodec libavformat libavutil libswresample libswscale libva; do
    line="$(grep -E "^[[:space:]]*${soname}\.so" <<<"$deps" || true)"
    [ -n "$line" ] || fail "$soname is not a dependency at all -- did the build lose FFmpeg/VA-API?"
    # Match the extracted tree, not a `usr/lib` spelling: RUNPATH is $ORIGIN/../lib, so ldd prints
    # the path it actually walked -- `.../usr/bin/../lib/libavcodec.so.NN`. What matters is only
    # that the resolved file lies inside the archive rather than on the host.
    grep -q "$tree/" <<<"$line" \
        || fail "$soname resolves outside the archive, so it was not bundled: $line"
done
ok "FFmpeg and libva resolve to the bundled copies in usr/lib"

# SDL3 is built static (SDL_STATIC in prosper/CMakeLists.txt). If it ever turns up as a shared
# object the archive has acquired a runtime dependency it does not declare -- the same assertion the
# Windows job makes about MinGW/SDL DLLs.
# An `A && fail` statement here would abort the script under `set -e` in the GOOD case, because grep
# exits 1 when it finds nothing. The passing path must be the one that returns zero.
if grep -qi 'libSDL3' <<<"$deps"; then
    fail "prosper-app unexpectedly links SDL3 dynamically"
fi
ok "SDL3 is statically linked (no libSDL3 dependency)"

# --- glibc floor -------------------------------------------------------------------------------
floor_err="$work/glibc.err"
floor="$("$here/glibc-floor.sh" "$tree" 2>"$floor_err")"
[ "$floor" != "unknown" ] || fail "could not determine the glibc floor"
highest="$(printf '%s\n%s\n' "$floor" "$max_glibc" | sort -V | tail -1)"
[ "$highest" = "$max_glibc" ] \
    || fail "glibc floor is $floor, above the declared maximum $max_glibc -- this artifact will not
        start on any distribution older than $floor. Either build on an older runner or raise
        --max-glibc deliberately."
if [ -s "$floor_err" ]; then
    cat "$floor_err" >&2
    fail "the tree requires non-numeric glibc ABI tags, which raise the floor beyond $floor"
fi
ok "glibc floor is $floor (declared maximum $max_glibc)"

# --- runtime ------------------------------------------------------------------------------------
smoke "tarball" "$bin"

# --- AppImage -----------------------------------------------------------------------------------
[ -x "$appimage" ] || fail "the AppImage is not executable"
file "$appimage" | grep -q 'ELF 64-bit' || fail "the AppImage is not an ELF image"
# CI containers have no FUSE; this is the supported way to run an AppImage without it.
export APPIMAGE_EXTRACT_AND_RUN=1
smoke "AppImage" "$appimage"

echo
echo "verify: all Linux release artifact checks passed"
