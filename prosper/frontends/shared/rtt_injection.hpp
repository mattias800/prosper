#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <unordered_map>
#include <utility>
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

// One immutable producer snapshot is commonly bound through the same scaled view by many draws and
// bindings in a submit. Materialize each (snapshot identity, source shape, destination shape,
// bytes-per-pixel) only once. The retained source owner prevents allocator address reuse from aliasing
// a live key; the retained output lets FrameResource copies and the backend's raw-pointer upload key
// share the bytes.
class RttInjectionCache {
public:
    std::shared_ptr<const std::vector<uint8_t>> materialize(
        std::shared_ptr<const std::vector<uint8_t>> source,
        uint32_t dst_w, uint32_t dst_h, uint32_t src_w, uint32_t src_h,
        uint32_t bytes_per_pixel) {
        if (!source) return {};
        if (exact_rtt_snapshot_borrowable(
                dst_w, dst_h, src_w, src_h, bytes_per_pixel, source->size()))
            return source;

        const Key key{source.get(), dst_w, dst_h, src_w, src_h, bytes_per_pixel};
        auto found = entries_.find(key);
        if (found != entries_.end()) return found->second.pixels;

        auto mutable_pixels = std::make_shared<std::vector<uint8_t>>();
        if (!inject_rtt_pixels(*mutable_pixels, dst_w, dst_h, *source,
                               src_w, src_h, bytes_per_pixel))
            return {};
        std::shared_ptr<const std::vector<uint8_t>> pixels = std::move(mutable_pixels);
        entries_.emplace(key, Entry{std::move(source), pixels});
        return pixels;
    }

    size_t size() const { return entries_.size(); }

private:
    struct Key {
        const std::vector<uint8_t>* source = nullptr;
        uint32_t dst_w = 0, dst_h = 0, src_w = 0, src_h = 0, bytes_per_pixel = 0;

        bool operator==(const Key& other) const {
            return source == other.source && dst_w == other.dst_w && dst_h == other.dst_h &&
                src_w == other.src_w && src_h == other.src_h &&
                bytes_per_pixel == other.bytes_per_pixel;
        }
    };

    struct KeyHash {
        size_t operator()(const Key& key) const {
            size_t hash = std::hash<const void*>{}(key.source);
            auto mix = [&](uint32_t value) {
                hash ^= std::hash<uint32_t>{}(value) + 0x9e3779b9u +
                    (hash << 6u) + (hash >> 2u);
            };
            mix(key.dst_w); mix(key.dst_h); mix(key.src_w); mix(key.src_h);
            mix(key.bytes_per_pixel);
            return hash;
        }
    };

    struct Entry {
        std::shared_ptr<const std::vector<uint8_t>> source;
        std::shared_ptr<const std::vector<uint8_t>> pixels;
    };

    std::unordered_map<Key, Entry, KeyHash> entries_;
};

} // namespace prosper::frontend
