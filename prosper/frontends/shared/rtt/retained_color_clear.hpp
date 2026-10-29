// A retained 2D clear must match this pass and outlive changes to the RTT cache.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace prosper::frontend {

template <typename Surface, typename Format>
bool retained_color_slot_matches(const Surface* surface, bool load_existing, uint32_t width,
                                 uint32_t height, Format format, uint32_t volume_depth) {
    return load_existing && surface && !volume_depth && !surface->volume_depth &&
           surface->w == width && surface->h == height && surface->format == format;
}

// Call after resource preparation, which can drain guest writes and erase retained surfaces.
// The returned pointer borrows only the caller's storage, never a cache node. A volume's dirty
// metadata can describe a partial colour-plane write, so it cannot supply a uniform 2D clear.
template <typename Surface, typename Format>
const float* retained_color_slot_clear(const Surface* surface, bool load_existing, uint32_t width,
                                       uint32_t height, Format format, uint32_t volume_depth,
                                       std::array<float, 4>& storage) {
    if (!retained_color_slot_matches(surface, load_existing, width, height, format, volume_depth) ||
        !surface->has_uniform_color)
        return nullptr;
    storage = surface->uniform_color;
    return storage.data();
}

// CPU seeds retain their immutable pixel owner through rendering and pass diagnostics. Holding
// this owner preserves lifetime only; the caller still selects after the guest-write drain.
template <typename Surface, typename Format>
std::shared_ptr<const std::vector<uint8_t>>
retained_color_slot_seed(const Surface* surface, bool load_existing, uint32_t width,
                         uint32_t height, Format format, uint32_t volume_depth, size_t bytes) {
    if (!retained_color_slot_matches(surface, load_existing, width, height, format, volume_depth) ||
        !surface->rgba || surface->rgba->size() != bytes)
        return {};
    return surface->rgba;
}

}   // namespace prosper::frontend
