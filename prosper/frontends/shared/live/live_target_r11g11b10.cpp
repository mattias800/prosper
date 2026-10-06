// pack_live_target_r11g11b10: repack a live render-target snapshot into the R11G11B10 words a
// compute storage image reads. Declared in live_compute.hpp; split out of live_compute.cpp unchanged.
#include "shared/live/live_compute.hpp"
#include "shared/live/live_target_format.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace prosper::frontend {

bool pack_live_target_r11g11b10(const prosper::gpu::LiveTargetSnapshot& snapshot,
                                uint8_t* packed, size_t packed_size) {
    if (!snapshot.width || !snapshot.height || !snapshot.pixels) return false;
    const uint64_t texels = static_cast<uint64_t>(snapshot.width) * snapshot.height;
    const uint32_t source_bytes =
        prosper::frontend::live_target_pixel_format_bytes(snapshot.format);
    if (!source_bytes) return false;
    const prosper::frontend::LiveTargetSourceLayout layout =
        prosper::frontend::live_target_source_layout(snapshot.format);
    if (texels > SIZE_MAX / source_bytes || texels > SIZE_MAX / sizeof(uint32_t) ||
        snapshot.pixels->size() != static_cast<size_t>(texels) * source_bytes ||
        !packed || packed_size != static_cast<size_t>(texels) * sizeof(uint32_t))
        return false;
    if (layout == prosper::frontend::LiveTargetSourceLayout::PackedR11G11B10) {
        std::memcpy(packed, snapshot.pixels->data(), packed_size);
        return true;
    }
    if (layout == prosper::frontend::LiveTargetSourceLayout::Unorm8x1 ||
        layout == prosper::frontend::LiveTargetSourceLayout::Unorm8x2 ||
        layout == prosper::frontend::LiveTargetSourceLayout::Uint32x1 ||
        layout == prosper::frontend::LiveTargetSourceLayout::Float32x1)
        return false;
    for (size_t t = 0; t < static_cast<size_t>(texels); ++t) {
        float rgb[3]{};
        if (layout == prosper::frontend::LiveTargetSourceLayout::Float16x4) {
            for (uint32_t c = 0; c < 3; ++c) {
                uint16_t half = 0;
                std::memcpy(&half, snapshot.pixels->data() + t * 8 + c * 2, sizeof(half));
                rgb[c] = prosper::gpu::half_to_float(half);
            }
        } else {
            for (uint32_t c = 0; c < 3; ++c)
                rgb[c] = (*snapshot.pixels)[t * 4 + c] / 255.0f;
        }
        const uint32_t word = static_cast<uint32_t>(prosper::gpu::float_to_f11(rgb[0])) |
                              (static_cast<uint32_t>(prosper::gpu::float_to_f11(rgb[1])) << 11) |
                              (static_cast<uint32_t>(prosper::gpu::float_to_f10(rgb[2])) << 22);
        std::memcpy(packed + t * sizeof(word), &word, sizeof(word));
    }
    return true;
}

} // namespace prosper::frontend
