// backend_color_formats.h -- the host Vulkan format the renderer gives each guest colour/texture
// format, and the byte and numeric-class arithmetic that follows from it. Moved verbatim out of
// render_runner.h (which includes this inside the same namespace) so the format tables can be read
// and extended in one place.
#pragma once

#include "gpu/resources/shader_resources.hpp"   // SpirvImageNumericClass
#include <vulkan/vulkan.h>
#include <algorithm>
#include <cstdint>

namespace prosper::test {

// Block-compressed sampled formats the backend carries natively: guest BCn blocks are copied
// straight into the staging buffer instead of being decoded to RGBA8 on the CPU. These are
// SAMPLED-texture formats only -- never a colour target, storage image or blit destination.
// Returns the bytes of one 4x4 block, or 0 for every non-block format.
inline uint32_t backend_block_compressed_bytes(VkFormat format) {
    switch (format) {
        case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:
        case VK_FORMAT_BC4_UNORM_BLOCK:
            return 8u;
        case VK_FORMAT_BC2_UNORM_BLOCK:
        case VK_FORMAT_BC3_UNORM_BLOCK:
        case VK_FORMAT_BC5_UNORM_BLOCK:
        case VK_FORMAT_BC6H_UFLOAT_BLOCK:
        case VK_FORMAT_BC7_UNORM_BLOCK:
            return 16u;
        default:
            return 0u;
    }
}

inline VkFormat backend_color_format(VkFormat format) {
    if (backend_block_compressed_bytes(format) != 0u)
        return format;
    if (format == VK_FORMAT_R16_SFLOAT ||
        format == VK_FORMAT_R16G16_SFLOAT ||
        format == VK_FORMAT_R16G16B16A16_SFLOAT ||
        format == VK_FORMAT_B10G11R11_UFLOAT_PACK32)
        return format;
    if (format == VK_FORMAT_R8_UNORM)
        return VK_FORMAT_R8_UNORM;
    if (format == VK_FORMAT_R8_UINT)
        return VK_FORMAT_R8_UINT;
    if (format == VK_FORMAT_R8G8B8A8_UINT)
        return VK_FORMAT_R8G8B8A8_UINT;
    if (format == VK_FORMAT_R8G8_UNORM)
        return VK_FORMAT_R8G8_UNORM;
    if (format == VK_FORMAT_R32_UINT)
        return VK_FORMAT_R32_UINT;
    if (format == VK_FORMAT_R32G32B32A32_UINT)
        return VK_FORMAT_R32G32B32A32_UINT;
    if (format == VK_FORMAT_R32G32B32A32_SFLOAT)
        return VK_FORMAT_R32G32B32A32_SFLOAT;
    if (format == VK_FORMAT_R32_SFLOAT)
        return VK_FORMAT_R32_SFLOAT;
    return VK_FORMAT_R8G8B8A8_UNORM;
}

// Bytes per TEXEL. A block-compressed format has no per-texel size, so it answers 0: every caller
// that can see a BC texture must size it with backend_texture_bytes() instead, and a caller that
// was missed fails closed (its `!bpp` guard) rather than over- or under-running a buffer.
inline uint32_t backend_color_bytes_per_pixel(VkFormat format) {
    format = backend_color_format(format);
    if (backend_block_compressed_bytes(format) != 0u) return 0u;
    if (format == VK_FORMAT_R32G32B32A32_UINT ||
        format == VK_FORMAT_R32G32B32A32_SFLOAT) return 16u;
    if (format == VK_FORMAT_R16G16B16A16_SFLOAT) return 8u;
    if (format == VK_FORMAT_R16G16_SFLOAT) return 4u;
    if (format == VK_FORMAT_R16_SFLOAT) return 2u;
    if (format == VK_FORMAT_R8_UNORM || format == VK_FORMAT_R8_UINT) return 1u;
    if (format == VK_FORMAT_R8G8_UNORM) return 2u;
    return 4u;
}

// Bytes of a width x height x depth x layers texture at one mip level, block-granular for a BC
// format (ceil(w/4) * ceil(h/4) blocks per slice) and texel-granular otherwise. This is exactly the
// tightly packed buffer layout vkCmdCopyBufferToImage reads with bufferRowLength = 0 and
// bufferImageHeight = 0. Saturates to UINT64_MAX on overflow so a caller's size check refuses it.
inline uint64_t backend_texture_bytes(VkFormat format, uint32_t width, uint32_t height,
                                      uint32_t depth = 1u, uint32_t layers = 1u) {
    const uint32_t block = backend_block_compressed_bytes(backend_color_format(format));
    uint64_t result = block ? block : backend_color_bytes_per_pixel(format);
    const uint64_t factors[4] = {
        block ? (uint64_t(width) + 3u) / 4u : uint64_t(width),
        block ? (uint64_t(height) + 3u) / 4u : uint64_t(height),
        std::max<uint64_t>(depth, 1u), std::max<uint64_t>(layers, 1u)};
    for (const uint64_t factor : factors) {
        if (factor && result > UINT64_MAX / factor) return UINT64_MAX;
        result *= factor;
    }
    return result;
}

// Bytes of levels [0, levels) of a 2D chain, each level tightly packed at max(w>>L,1) x max(h>>L,1)
// and concatenated level 0 first: the staging layout of FrameResource::uploaded_mip_levels.
inline uint64_t backend_texture_chain_bytes(VkFormat format, uint32_t width, uint32_t height,
                                            uint32_t levels) {
    uint64_t total = 0;
    for (uint32_t level = 0; level < std::max(levels, 1u); ++level) {
        const uint64_t bytes = backend_texture_bytes(
            format, std::max(width >> level, 1u), std::max(height >> level, 1u));
        if (bytes == UINT64_MAX || total > UINT64_MAX - bytes) return UINT64_MAX;
        total += bytes;
    }
    return total;
}

inline prosper::gpu::SpirvImageNumericClass backend_image_numeric_class(VkFormat format) {
    using NumericClass = prosper::gpu::SpirvImageNumericClass;
    if (backend_block_compressed_bytes(backend_color_format(format)) != 0u)
        return NumericClass::Float;
    switch (backend_color_format(format)) {
        case VK_FORMAT_R8_UINT:
        case VK_FORMAT_R8G8B8A8_UINT:
        case VK_FORMAT_R32_UINT:
        case VK_FORMAT_R32G32B32A32_UINT:
            return NumericClass::Uint;
        case VK_FORMAT_R8_UNORM:
        case VK_FORMAT_R8G8_UNORM:
        case VK_FORMAT_R8G8B8A8_UNORM:
        case VK_FORMAT_R16_SFLOAT:
        case VK_FORMAT_R16G16_SFLOAT:
        case VK_FORMAT_R16G16B16A16_SFLOAT:
        case VK_FORMAT_R32G32B32A32_SFLOAT:
        case VK_FORMAT_R32_SFLOAT:
        case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
            return NumericClass::Float;
        default:
            return NumericClass::Unknown;
    }
}
}  // namespace prosper::test
