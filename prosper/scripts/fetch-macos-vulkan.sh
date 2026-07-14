#!/usr/bin/env bash
# fetch-macos-vulkan.sh — obtain a UNIVERSAL (x86_64+arm64) MoltenVK for the macOS harness app.
#
# prosper on macOS is built x86_64 (it runs the PS5 guest's x86-64 code natively under Rosetta 2), so
# it needs an x86_64 Vulkan driver. Homebrew's molten-vk/vulkan-loader are arm64-only and cannot link
# into the x86_64 binary. This drops a universal MoltenVK into <repo>/.macos-vulkan/ (gitignored) and
# points CMake at it with -DPROSPER_MACOS_MOLTENVK=.../.macos-vulkan/lib/libMoltenVK.dylib.
#
# TWO sources (see docs/PORTING.md "macOS harness app"):
#   (default) Khronos prebuilt release — small, fast; FINE FOR BUILDING/CI (build-only), but its
#             x86_64 build has a SPIRV-Cross bug (#693) that fails to convert some guest shaders to
#             MSL, so it does NOT render real content.
#   --lunarg  LunarG Vulkan SDK's MoltenVK — a good universal build whose SPIRV-Cross converts the
#             guest shaders. USE THIS TO ACTUALLY RENDER. ~370 MB download, headless-installed to a
#             local dir (no changes to your system beyond <repo>/.macos-vulkan and a temp dir).
#
# Usage:  bash prosper/scripts/fetch-macos-vulkan.sh            # Khronos release (build/CI)
#         bash prosper/scripts/fetch-macos-vulkan.sh --lunarg   # LunarG SDK MoltenVK (rendering)
set -euo pipefail

MODE="khronos"; [ "${1:-}" = "--lunarg" ] && MODE="lunarg"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="$REPO_ROOT/.macos-vulkan"
mkdir -p "$DEST/lib" "$DEST/share/vulkan/icd.d"
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT

if [ "$MODE" = "lunarg" ]; then
  VER="$(curl -fsSL https://vulkan.lunarg.com/sdk/latest/mac.txt | tr -d '[:space:]')"
  echo "==> Fetching LunarG Vulkan SDK ${VER} for macOS (universal MoltenVK; good SPIRV-Cross)"
  curl -fsSL -o "$tmp/sdk.zip" "https://sdk.lunarg.com/sdk/download/${VER}/mac/vulkansdk-macos-${VER}.zip"
  ( cd "$tmp" && unzip -q sdk.zip )
  APP="$(find "$tmp" -maxdepth 2 -name 'vulkansdk-macOS-*' -path '*/MacOS/*' -type f | head -1)"
  chmod +x "$APP"
  echo "==> Headless-installing the SDK to $tmp/VulkanSDK ..."
  "$APP" --root "$tmp/VulkanSDK" --accept-licenses --accept-obligations --default-answer --confirm-command install >/dev/null 2>&1
  SRC="$(find "$tmp/VulkanSDK/macOS/lib" -name libMoltenVK.dylib | head -1)"
else
  VER="${2:-v1.4.1}"
  echo "==> Fetching Khronos MoltenVK ${VER} (universal; build/CI only — see #693)"
  curl -fsSL -o "$tmp/mvk.tar" "https://github.com/KhronosGroup/MoltenVK/releases/download/${VER}/MoltenVK-macos.tar"
  tar xf "$tmp/mvk.tar" -C "$tmp"
  SRC="$tmp/MoltenVK/MoltenVK/dynamic/dylib/macOS/libMoltenVK.dylib"
fi

cp "$SRC" "$DEST/lib/libMoltenVK.dylib"
cat > "$DEST/share/vulkan/icd.d/MoltenVK_icd.json" <<JSON
{
  "file_format_version": "1.0.0",
  "ICD": { "library_path": "../../../lib/libMoltenVK.dylib", "api_version": "1.4.0", "is_portability_driver": true }
}
JSON

echo "==> Installed $(lipo -archs "$DEST/lib/libMoltenVK.dylib" 2>/dev/null) libMoltenVK.dylib  [$MODE]"
echo "==> Configure with: -DPROSPER_MACOS_MOLTENVK=$DEST/lib/libMoltenVK.dylib"
[ "$MODE" = "khronos" ] && echo "==> NOTE: for real rendering use --lunarg (the Khronos x86_64 build can't convert some shaders, #693)"
