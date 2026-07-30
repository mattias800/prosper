#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace prosper::frontend {

// An immutable CPU RTT snapshot can back a FrameResource directly only when the consumer uses the
// producer's exact extent and byte layout. Scaled views still need their owned nearest-neighbor copy;
// malformed snapshots stay on the guest-decode fallback.
inline bool exact_rtt_snapshot_borrowable(uint32_t dst_w, uint32_t dst_h,
                                          uint32_t src_w, uint32_t src_h,
                                          uint32_t bytes_per_pixel,
                                          size_t source_bytes) {
    if (!dst_w || !dst_h || dst_w != src_w || dst_h != src_h || !bytes_per_pixel)
        return false;
    const uint64_t texels = static_cast<uint64_t>(src_w) * src_h;
    if (texels > std::numeric_limits<size_t>::max() / bytes_per_pixel) return false;
    return source_bytes == static_cast<size_t>(texels) * bytes_per_pixel;
}

namespace detail {

template <size_t BytesPerPixel>
inline void inject_scaled_rtt_rows(uint8_t* dst, uint32_t dst_w, uint32_t dst_h,
                                   const uint8_t* src, uint32_t src_w, uint32_t src_h) {
    const size_t dst_row_bytes = static_cast<size_t>(dst_w) * BytesPerPixel;
    uint32_t previous_sy = UINT32_MAX;

    // Render scaling is normally an integer factor (Astro Bot is 3840x2160 over a 1920x1080
    // retained target). Expand one source row and bulk-copy its vertical repetitions. This avoids
    // millions of tiny variable-sized memcpy calls and integer divisions in the old per-texel loop.
    const uint32_t integer_x_scale = dst_w >= src_w && dst_w % src_w == 0
        ? dst_w / src_w : 0;
    for (uint32_t y = 0; y < dst_h; ++y) {
        const uint32_t sy = static_cast<uint32_t>(static_cast<uint64_t>(y) * src_h / dst_h);
        uint8_t* dst_row = dst + static_cast<size_t>(y) * dst_row_bytes;
        if (y && sy == previous_sy) {
            std::memcpy(dst_row, dst_row - dst_row_bytes, dst_row_bytes);
            continue;
        }

        const uint8_t* src_row = src + static_cast<size_t>(sy) * src_w * BytesPerPixel;
        if (src_w == dst_w) {
            std::memcpy(dst_row, src_row, dst_row_bytes);
        } else if (integer_x_scale) {
            for (uint32_t sx = 0; sx < src_w; ++sx) {
                const uint8_t* pixel = src_row + static_cast<size_t>(sx) * BytesPerPixel;
                for (uint32_t repeat = 0; repeat < integer_x_scale; ++repeat)
                    std::memcpy(dst_row +
                                    (static_cast<size_t>(sx) * integer_x_scale + repeat) *
                                        BytesPerPixel,
                                pixel, BytesPerPixel);
            }
        } else {
            // Precomputing x offsets once would require a scratch allocation. The uncommon
            // non-integer path instead uses a division per output texel, but still specializes the
            // copy width so the compiler emits a scalar load/store rather than a libc call.
            for (uint32_t x = 0; x < dst_w; ++x) {
                const uint32_t sx = static_cast<uint32_t>(
                    static_cast<uint64_t>(x) * src_w / dst_w);
                std::memcpy(dst_row + static_cast<size_t>(x) * BytesPerPixel,
                            src_row + static_cast<size_t>(sx) * BytesPerPixel,
                            BytesPerPixel);
            }
        }
        previous_sy = sy;
    }
}

} // namespace detail

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
    switch (bytes_per_pixel) {
        case 1: detail::inject_scaled_rtt_rows<1>(dst.data(), dst_w, dst_h,
                                                  src.data(), src_w, src_h); break;
        case 2: detail::inject_scaled_rtt_rows<2>(dst.data(), dst_w, dst_h,
                                                  src.data(), src_w, src_h); break;
        case 4: detail::inject_scaled_rtt_rows<4>(dst.data(), dst_w, dst_h,
                                                  src.data(), src_w, src_h); break;
        case 8: detail::inject_scaled_rtt_rows<8>(dst.data(), dst_w, dst_h,
                                                  src.data(), src_w, src_h); break;
        case 16: detail::inject_scaled_rtt_rows<16>(dst.data(), dst_w, dst_h,
                                                    src.data(), src_w, src_h); break;
        default:
            for (uint32_t y = 0; y < dst_h; ++y) {
                const uint32_t sy = static_cast<uint32_t>(
                    static_cast<uint64_t>(y) * src_h / dst_h);
                for (uint32_t x = 0; x < dst_w; ++x) {
                    const uint32_t sx = static_cast<uint32_t>(
                        static_cast<uint64_t>(x) * src_w / dst_w);
                    const size_t si = (static_cast<size_t>(sy) * src_w + sx) * bytes_per_pixel;
                    const size_t di = (static_cast<size_t>(y) * dst_w + x) * bytes_per_pixel;
                    std::memcpy(dst.data() + di, src.data() + si, bytes_per_pixel);
                }
            }
            break;
    }
    return true;
}

} // namespace prosper::frontend
