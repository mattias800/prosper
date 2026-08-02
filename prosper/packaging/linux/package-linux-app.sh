#!/usr/bin/env bash
# Package a built prosper-app into the two Linux desktop release artifacts:
#
#   prosper-linux-x86_64.AppImage   single self-contained file, the desktop default
#   prosper-linux-x86_64.tar.gz     the same tree, for hosts where the AppImage runtime cannot run
#
# WHY BUNDLING IS NOT OPTIONAL. prosper-app links FFmpeg (AJM audio decode) and libva (AvPlayer
# video). Those sonames differ on every distribution -- Ubuntu 24.04 ships libavcodec.so.60 /
# libavutil.so.58 while Fedora 43 ships libavcodec.so.62 / libavutil.so.60 -- so a bare binary
# built here does not start anywhere else. linuxdeploy copies the closure into usr/lib and rewrites
# RUNPATH to $ORIGIN/../lib, which is what makes one build run across distributions. SDL3 is already
# static (SDL_STATIC in the top-level CMakeLists), so it is not part of that closure.
#
# WHAT BUNDLING DOES NOT FIX: glibc. No AppImage bundles libc, so the glibc of the machine that runs
# THIS script is the floor for everyone who runs the artifact. Build it on the oldest distribution
# you intend to support; verify-linux-app.sh asserts the resulting floor against a declared maximum
# so the floor cannot rise unnoticed.
#
# THE HOST MATTERS BEYOND glibc. What gets bundled is whatever the host's FFmpeg drags in, and that
# is a property of the distribution, not of prosper. Ubuntu 24.04 -- the pinned CI runner, and the
# only configuration this is verified on -- pulls 116 libraries. Fedora 43 pulls 231, including
# Samba, Kerberos, ICU and OpenCL, and the result segfaults in an ELF initializer before main().
# See README.md "Ruled out". Running this on another distribution may produce an archive that does
# not start, which is exactly why verify-linux-app.sh executes the packaged binary.
#
# Usage:
#   package-linux-app.sh --build-dir <dir> --out-dir <dir> [--tools-dir <dir>] [--build-info <text>]
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$here/../../.." && pwd)"

build_dir=""
out_dir=""
tools_dir=""
build_info=""

while [ $# -gt 0 ]; do
    case "$1" in
        --*) [ $# -ge 2 ] || { echo "package-linux-app.sh: $1 needs a value" >&2; exit 2; } ;;&
        --build-dir)  build_dir="$2"; shift 2 ;;
        --out-dir)    out_dir="$2";   shift 2 ;;
        --tools-dir)  tools_dir="$2"; shift 2 ;;
        --build-info) build_info="$2"; shift 2 ;;
        *) echo "package-linux-app.sh: unknown argument: $1" >&2; exit 2 ;;
    esac
done

[ -n "$build_dir" ] || { echo "package-linux-app.sh: --build-dir is required" >&2; exit 2; }
[ -n "$out_dir" ]   || { echo "package-linux-app.sh: --out-dir is required" >&2; exit 2; }

app_binary="$build_dir/prosper-app"
[ -x "$app_binary" ] || { echo "package-linux-app.sh: no executable at $app_binary" >&2; exit 1; }

mkdir -p "$out_dir"
out_dir="$(cd "$out_dir" && pwd)"
tools_dir="${tools_dir:-$out_dir/.tools}"
mkdir -p "$tools_dir"
tools_dir="$(cd "$tools_dir" && pwd)"

# Pinned by tag AND by content hash. `continuous` is a moving target that would silently change what
# a release was built with; a tag alone still lets the asset be re-uploaded. Both tools were run
# against this exact prosper tree before these hashes were recorded.
LINUXDEPLOY_URL="https://github.com/linuxdeploy/linuxdeploy/releases/download/1-alpha-20251107-1/linuxdeploy-x86_64.AppImage"
LINUXDEPLOY_SHA256="c20cd71e3a4e3b80c3483cef793cda3f4e990aca14014d23c544ca3ce1270b4d"
APPIMAGETOOL_URL="https://github.com/AppImage/appimagetool/releases/download/1.9.1/appimagetool-x86_64.AppImage"
APPIMAGETOOL_SHA256="ed4ce84f0d9caff66f50bcca6ff6f35aae54ce8135408b3fa33abfc3cb384eb0"

fetch_tool() {
    local url="$1" want="$2" dest="$3"
    if [ -f "$dest" ] && printf '%s  %s\n' "$want" "$dest" | sha256sum -c --status; then
        echo "package: reusing $(basename "$dest")"
        return 0
    fi
    echo "package: fetching $(basename "$dest")"
    curl -fsSL --retry 3 --retry-delay 2 -o "$dest" "$url"
    # Refuse to run a tool whose content is not the reviewed one, rather than warn and continue.
    if ! printf '%s  %s\n' "$want" "$dest" | sha256sum -c --status; then
        echo "package: sha256 mismatch for $url" >&2
        echo "  expected $want" >&2
        echo "  actual   $(sha256sum "$dest" | cut -d' ' -f1)" >&2
        exit 1
    fi
    chmod +x "$dest"
}

fetch_tool "$LINUXDEPLOY_URL"   "$LINUXDEPLOY_SHA256"   "$tools_dir/linuxdeploy-x86_64.AppImage"
fetch_tool "$APPIMAGETOOL_URL"  "$APPIMAGETOOL_SHA256"  "$tools_dir/appimagetool-x86_64.AppImage"

# Both tools are themselves AppImages. CI containers have no FUSE, so run them extracted.
export APPIMAGE_EXTRACT_AND_RUN=1

# linuxdeploy strips every library it bundles using the binutils `strip` inside its own AppImage.
# That copy is older than the toolchains some hosts build with and aborts on a section it does not
# know -- on Fedora 43 every bundled library fails with `unknown type [0x13] section .relr.dyn`, and
# the whole run exits non-zero. Distribution libraries arrive stripped anyway, so the saving is
# small and not worth a packaging step that breaks on the build host's age. prosper-app itself is
# deliberately left unstripped so a user's crash report still carries symbols.
export NO_STRIP=1

appdir="$out_dir/AppDir"
rm -rf "$appdir"
mkdir -p "$appdir/usr/bin"
cp "$app_binary" "$appdir/usr/bin/prosper-app"
chmod +x "$appdir/usr/bin/prosper-app"

echo "package: bundling the shared-library closure"
"$tools_dir/linuxdeploy-x86_64.AppImage" \
    --appdir "$appdir" \
    -e "$appdir/usr/bin/prosper-app" \
    -d "$here/prosper.desktop" \
    -i "$here/prosper.png"

# linuxdeploy owns everything under usr/; these three are the release's own additions and sit at the
# AppDir root so they are visible both to someone who extracts the tarball and to
# `--appimage-extract`.
cp "$repo_root/prosper/docs/LINUX_RELEASE.md" "$appdir/README.md"
cp "$repo_root/prosper/scripts/start-prosper.sh" "$appdir/start-prosper.sh"
chmod +x "$appdir/start-prosper.sh"
# Computed into a variable first: inside `echo "... $(...)"` the substitution's failure is masked by
# echo's own zero status, and BUILD.txt would silently record an empty floor.
floor="$("$here/glibc-floor.sh" "$appdir")" \
    || { echo "package-linux-app.sh: could not measure the glibc floor" >&2; exit 1; }
{
    echo "prosper Linux x86_64"
    if [ -n "$build_info" ]; then printf '%s\n' "$build_info"; fi
    echo "glibc floor: $floor"
} > "$appdir/BUILD.txt"

echo "package: building the AppImage"
( cd "$out_dir" && "$tools_dir/appimagetool-x86_64.AppImage" AppDir prosper-linux-x86_64.AppImage )

echo "package: building the tarball"
# Staged through a directory named for the archive so the tarball extracts into one self-named
# folder rather than scattering AppDir/ into the user's cwd.
stage="$out_dir/.stage"
rm -rf "$stage"
mkdir -p "$stage"
cp -a "$appdir" "$stage/prosper-linux-x86_64"
tar -czf "$out_dir/prosper-linux-x86_64.tar.gz" -C "$stage" prosper-linux-x86_64
rm -rf "$stage"

( cd "$out_dir" && sha256sum prosper-linux-x86_64.AppImage > prosper-linux-x86_64.AppImage.sha256 )
( cd "$out_dir" && sha256sum prosper-linux-x86_64.tar.gz   > prosper-linux-x86_64.tar.gz.sha256 )

echo "package: done"
ls -la "$out_dir"/prosper-linux-x86_64.*
