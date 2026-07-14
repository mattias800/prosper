#!/usr/bin/env bash
# fetch-macos-vulkan.sh — download a UNIVERSAL (x86_64+arm64) MoltenVK for the macOS harness app.
#
# prosper on macOS is built x86_64 (it runs the PS5 guest's x86-64 code natively under Rosetta 2), so
# it needs an x86_64 Vulkan driver. Homebrew's molten-vk/vulkan-loader are arm64-only and cannot link
# into the x86_64 binary. The prebuilt KhronosGroup/MoltenVK release ships a universal dylib, which is
# exactly what we need. This drops it into <repo>/.macos-vulkan/ (gitignored); pass its path to CMake:
#
#   cmake -S prosper -B prosper/build-mac-app -G Ninja \
#     -DCMAKE_OSX_ARCHITECTURES=x86_64 -DPROSPER_APP=ON -DPROSPER_AUDIO_SDL3=ON -DPROSPER_PAD_SDL3=ON \
#     -DPROSPER_MACOS_MOLTENVK="$PWD/.macos-vulkan/lib/libMoltenVK.dylib"
#   cmake --build prosper/build-mac-app --target prosper-app
#   PROSPER_VULKAN_LIB="$PWD/.macos-vulkan/lib/libMoltenVK.dylib" \
#     ./prosper/build-mac-app/prosper-app --test-pattern       # smoke test (no game needed)
#
# The LunarG Vulkan SDK for macOS is an equivalent (also universal) source if you prefer it.
set -euo pipefail

VER="${1:-v1.4.1}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="$REPO_ROOT/.macos-vulkan"
TAR_URL="https://github.com/KhronosGroup/MoltenVK/releases/download/${VER}/MoltenVK-macos.tar"

echo "==> Fetching MoltenVK ${VER} (universal) into ${DEST}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
curl -fsSL -o "$tmp/mvk.tar" "$TAR_URL"
tar xf "$tmp/mvk.tar" -C "$tmp"

SRC="$tmp/MoltenVK/MoltenVK/dynamic/dylib/macOS/libMoltenVK.dylib"
mkdir -p "$DEST/lib" "$DEST/share/vulkan/icd.d"
cp "$SRC" "$DEST/lib/"
cat > "$DEST/share/vulkan/icd.d/MoltenVK_icd.json" <<JSON
{
  "file_format_version": "1.0.0",
  "ICD": { "library_path": "../../../lib/libMoltenVK.dylib", "api_version": "1.4.0", "is_portability_driver": true }
}
JSON

echo "==> Installed $(lipo -archs "$DEST/lib/libMoltenVK.dylib" 2>/dev/null) libMoltenVK.dylib"
echo "==> Configure with: -DPROSPER_MACOS_MOLTENVK=$DEST/lib/libMoltenVK.dylib"
