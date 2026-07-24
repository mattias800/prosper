#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace prosper::frontend {

// Copy a renderer-owned target into a sampled-resource buffer, nearest-scaling when the view extent
// differs. The RTT format owns the byte stride: consumers such as 1D views may have been provisionally
// sized as RGBA8 even when the cached producer is RGBA16F, so this helper resizes before every write.
inline bool inject_rtt_pixels(std::vector<uint8_t>& dst, uint32_t dst_w, uint32_t dst_h,
                              const std::vector<uint8_t>& src, uint32_t src_w, uint32_t src_h,
                              uint32_t bytes_per_pixel) {
    if (!dst_w || !dst_h || !src_w || !src_h || !bytes_per_pixel) return false;
    const uint64_t dst_texels = static_cast<uint64_t>(dst_w) * dst_h;
    const uint64_t src_texels = static_cast<uint64_t>(src_w) * src_h;
    if (dst_texels > std::numeric_limits<size_t>::max() / bytes_per_pixel ||
        src_texels > std::numeric_limits<size_t>::max() / bytes_per_pixel)
        return false;
    const size_t dst_bytes = static_cast<size_t>(dst_texels) * bytes_per_pixel;
    const size_t src_bytes = static_cast<size_t>(src_texels) * bytes_per_pixel;
    if (src.size() != src_bytes) return false;
    if (src_w == dst_w && src_h == dst_h) {
        dst = src;
        return true;
    }
    dst.resize(dst_bytes);
    for (uint32_t y = 0; y < dst_h; ++y) {
        const uint32_t sy = static_cast<uint32_t>(static_cast<uint64_t>(y) * src_h / dst_h);
        for (uint32_t x = 0; x < dst_w; ++x) {
            const uint32_t sx = static_cast<uint32_t>(static_cast<uint64_t>(x) * src_w / dst_w);
            const size_t si = (static_cast<size_t>(sy) * src_w + sx) * bytes_per_pixel;
            const size_t di = (static_cast<size_t>(y) * dst_w + x) * bytes_per_pixel;
            std::memcpy(dst.data() + di, src.data() + si, bytes_per_pixel);
        }
    }
    return true;
}

} // namespace prosper::frontend
