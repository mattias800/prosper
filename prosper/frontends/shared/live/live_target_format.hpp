// live_target_format.hpp — the one place a LiveTargetPixelFormat is mapped onto anything else.
//
// The same defect has now cost two titles a whole render layer. Dead Cells (#773) lost an HDR
// lighting target because a renderer-owned surface was reuploaded as RGBA8; Syberia: Remastered
// loses its entire 3D menu scene because the compute write-back notifier described an
// R11G11B10_FLOAT write as VK_FORMAT_R8G8B8A8_UNORM, failed the mirror-identity check, and returned
// before republishing the RTT entry — so every later consumer sampled guest bytes prosper never
// wrote. Both were a two-way ternary over a three-member enum, whose `else` silently reports an
// unhandled member as RGBA8.
//
// Every mapping below is a `switch` with NO `default:` label. Together with `-Werror=switch` on the
// prosper_live_renderer target, adding a LiveTargetPixelFormat member becomes a BUILD FAILURE at
// each site that has to learn about it, instead of a silent RGBA8 fallback that only surfaces as a
// black frame in one game months later. The trailing `return` after each switch is the fail-closed
// answer for an out-of-range value cast into the enum; it is unreachable for a real enumerator and
// deliberately does not suppress -Wswitch the way a `default:` label would.
#pragma once

#include <cstdint>

#include <vulkan/vulkan.h>

#include "gpu/execute/gpu_execute.hpp"

namespace prosper::frontend {

// How one texel of the renderer's CPU snapshot is laid out. Conversion loops that implement a
// subset of the layouts branch on THIS rather than on "is it RGBA8, else assume FP16", so a new
// pixel format must be classified explicitly before it can reach a decoder that cannot read it.
enum class LiveTargetSourceLayout : uint8_t {
    Unorm8x4,
    Float16x4,
    PackedR11G11B10,
    Unorm8x1,
    Uint32x1,
    Float32x1,
    Unorm8x2,
};

constexpr LiveTargetSourceLayout live_target_source_layout(
    prosper::gpu::LiveTargetPixelFormat format) {
    switch (format) {
        case prosper::gpu::LiveTargetPixelFormat::Rgba8Unorm:
            return LiveTargetSourceLayout::Unorm8x4;
        case prosper::gpu::LiveTargetPixelFormat::Rgba16Float:
            return LiveTargetSourceLayout::Float16x4;
        case prosper::gpu::LiveTargetPixelFormat::R11G11B10Float:
            return LiveTargetSourceLayout::PackedR11G11B10;
        case prosper::gpu::LiveTargetPixelFormat::R8Unorm:
            return LiveTargetSourceLayout::Unorm8x1;
        case prosper::gpu::LiveTargetPixelFormat::R32Uint:
            return LiveTargetSourceLayout::Uint32x1;
        case prosper::gpu::LiveTargetPixelFormat::R32Float:
            return LiveTargetSourceLayout::Float32x1;
        case prosper::gpu::LiveTargetPixelFormat::Rg8Unorm:
            return LiveTargetSourceLayout::Unorm8x2;
    }
    return LiveTargetSourceLayout::Unorm8x4;
}

// The exact backend VkFormat a live target of this pixel format is rendered, read back, sampled and
// cached as. This is the identity the RTT mirror check compares, so an approximate answer is worse
// than no answer: it silently invalidates a target that was correctly written.
constexpr VkFormat live_target_pixel_format_vk(prosper::gpu::LiveTargetPixelFormat format) {
    switch (format) {
        case prosper::gpu::LiveTargetPixelFormat::Rgba8Unorm:
            return VK_FORMAT_R8G8B8A8_UNORM;
        case prosper::gpu::LiveTargetPixelFormat::Rgba16Float:
            return VK_FORMAT_R16G16B16A16_SFLOAT;
        case prosper::gpu::LiveTargetPixelFormat::R11G11B10Float:
            return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
        case prosper::gpu::LiveTargetPixelFormat::R8Unorm:
            return VK_FORMAT_R8_UNORM;
        case prosper::gpu::LiveTargetPixelFormat::R32Uint:
            return VK_FORMAT_R32_UINT;
        case prosper::gpu::LiveTargetPixelFormat::R32Float:
            return VK_FORMAT_R32_SFLOAT;
        case prosper::gpu::LiveTargetPixelFormat::Rg8Unorm:
            return VK_FORMAT_R8G8_UNORM;
    }
    return VK_FORMAT_UNDEFINED;
}

// Bytes per texel in the renderer's snapshot/backing for this format. Zero means "unknown", which
// every caller must treat as a decline rather than as a size.
constexpr uint32_t live_target_pixel_format_bytes(prosper::gpu::LiveTargetPixelFormat format) {
    switch (format) {
        case prosper::gpu::LiveTargetPixelFormat::Rgba8Unorm:      return 4u;
        case prosper::gpu::LiveTargetPixelFormat::Rgba16Float:     return 8u;
        case prosper::gpu::LiveTargetPixelFormat::R11G11B10Float:  return 4u;
        case prosper::gpu::LiveTargetPixelFormat::R8Unorm:         return 1u;
        case prosper::gpu::LiveTargetPixelFormat::R32Uint:         return 4u;
        case prosper::gpu::LiveTargetPixelFormat::R32Float:        return 4u;
        case prosper::gpu::LiveTargetPixelFormat::Rg8Unorm:        return 2u;
    }
    return 0u;
}

// Diagnostic label. Trace output that mislabels a format sends the next investigation down the
// wrong path, so this is exhaustive for the same reason the numeric mappings are.
constexpr const char* live_target_pixel_format_name(prosper::gpu::LiveTargetPixelFormat format) {
    switch (format) {
        case prosper::gpu::LiveTargetPixelFormat::Rgba8Unorm:      return "rgba8";
        case prosper::gpu::LiveTargetPixelFormat::Rgba16Float:     return "rgba16f";
        case prosper::gpu::LiveTargetPixelFormat::R11G11B10Float:  return "r11g11b10";
        case prosper::gpu::LiveTargetPixelFormat::R8Unorm:         return "r8";
        case prosper::gpu::LiveTargetPixelFormat::R32Uint:         return "r32ui";
        case prosper::gpu::LiveTargetPixelFormat::R32Float:        return "r32f";
        case prosper::gpu::LiveTargetPixelFormat::Rg8Unorm:        return "rg8";
    }
    return "unknown";
}

// Reverse direction. VkFormat is an open enumeration with hundreds of members, so this cannot be a
// -Wswitch-checked mapping; it instead fails closed on every format the live-target contract does
// not name. A caller that ignored the result and used the out-parameter anyway would reintroduce
// exactly the RGBA8-fallback bug, so the value is left untouched on failure.
constexpr bool live_target_pixel_format_from_vk(VkFormat vk_format,
                                                prosper::gpu::LiveTargetPixelFormat& format) {
    if (vk_format == VK_FORMAT_R8G8B8A8_UNORM) {
        format = prosper::gpu::LiveTargetPixelFormat::Rgba8Unorm;
        return true;
    }
    if (vk_format == VK_FORMAT_R16G16B16A16_SFLOAT) {
        format = prosper::gpu::LiveTargetPixelFormat::Rgba16Float;
        return true;
    }
    if (vk_format == VK_FORMAT_B10G11R11_UFLOAT_PACK32) {
        format = prosper::gpu::LiveTargetPixelFormat::R11G11B10Float;
        return true;
    }
    if (vk_format == VK_FORMAT_R8_UNORM) {
        format = prosper::gpu::LiveTargetPixelFormat::R8Unorm;
        return true;
    }
    if (vk_format == VK_FORMAT_R32_UINT) {
        format = prosper::gpu::LiveTargetPixelFormat::R32Uint;
        return true;
    }
    if (vk_format == VK_FORMAT_R32_SFLOAT) {
        format = prosper::gpu::LiveTargetPixelFormat::R32Float;
        return true;
    }
    if (vk_format == VK_FORMAT_R8G8_UNORM) {
        format = prosper::gpu::LiveTargetPixelFormat::Rg8Unorm;
        return true;
    }
    return false;
}

} // namespace prosper::frontend
