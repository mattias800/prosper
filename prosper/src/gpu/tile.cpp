// tile.cpp — see tile.hpp. GFX10 SW_4KB_S de-swizzle for 32-bpp surfaces.
#include "tile.hpp"
#include <cstring>

namespace prosper::gpu {

bool tile_mode_is_tiled(uint32_t tile_mode) {
    return tile_mode == (uint32_t)TileMode::Sw4KbS;
}

size_t tiled_surface_bytes(uint32_t width, uint32_t height, uint32_t tile_mode, uint32_t pitch) {
    if (!tile_mode_is_tiled(tile_mode)) return (size_t)width * height * 4;
    uint32_t pw = pitch ? pitch : width;
    uint32_t tiles_per_row = (pw + 31) / 32, tile_rows = (height + 31) / 32;   // pad H up to whole tiles
    return (size_t)tiles_per_row * tile_rows * (32 * 32) * 4;
}

namespace {
// Map linear (x,y) -> the texel's linear INDEX in the tiled surface for SW_4KB_S: 32x32 micro-tiles laid
// out row-major over the padded pitch, Morton order within a tile with Y in the low bit of each pair.
inline uint64_t sw4kb_s_index(uint32_t x, uint32_t y, uint32_t tiles_per_row) {
    constexpr uint32_t T = 32, TB = 5;                 // tile size, log2(tile)
    uint32_t tx = x >> TB, ty = y >> TB, ix = x & (T - 1), iy = y & (T - 1);
    uint32_t morton = 0;
    for (uint32_t b = 0; b < TB; b++) {
        morton |= ((iy >> b) & 1u) << (2 * b);
        morton |= ((ix >> b) & 1u) << (2 * b + 1);
    }
    return (uint64_t)(ty * tiles_per_row + tx) * (T * T) + morton;
}

// The single tiled<->linear walk both public functions share: per texel, the tiled offset (Morton) and
// the linear offset, moving 4 bytes in the chosen direction so the index math can never diverge between
// tile and detile. `tiled_bytes` bounds the tiled side; an out-of-range tiled texel reads as zero when
// detiling and is skipped when tiling (tile_surface pre-zeroes the whole padded destination).
template <bool ToTiled>
void sw4kb_s_copy(uint8_t* dst, const uint8_t* src, uint32_t width, uint32_t height, uint32_t pitch,
                  size_t tiled_bytes) {
    uint32_t pw = pitch ? pitch : width;
    uint32_t tiles_per_row = (pw + 31) / 32;
    for (uint32_t y = 0; y < height; y++)
        for (uint32_t x = 0; x < width; x++) {
            uint64_t t = sw4kb_s_index(x, y, tiles_per_row) * 4;   // tiled offset
            size_t   l = ((size_t)y * width + x) * 4;              // linear offset
            if (ToTiled) { if (t + 4 <= tiled_bytes) std::memcpy(dst + t, src + l, 4); }
            else         { if (t + 4 <= tiled_bytes) std::memcpy(dst + l, src + t, 4);
                           else                      std::memset(dst + l, 0, 4); }
        }
}
} // namespace

void detile_surface(uint8_t* dst, const uint8_t* src, uint32_t width, uint32_t height,
                    uint32_t tile_mode, uint32_t pitch) {
    if (!tile_mode_is_tiled(tile_mode)) { std::memcpy(dst, src, (size_t)width * height * 4); return; }
    sw4kb_s_copy<false>(dst, src, width, height, pitch, tiled_surface_bytes(width, height, tile_mode, pitch));
}

std::vector<uint8_t> detile_surface(const std::vector<uint8_t>& src, uint32_t width, uint32_t height,
                                    uint32_t tile_mode, uint32_t pitch) {
    std::vector<uint8_t> out((size_t)width * height * 4, 0);
    if (src.empty()) return out;
    // Enforce the "src holds at least tiled_surface_bytes" precondition here instead of trusting the
    // caller: the pointer overload's bounds guard is computed from the DIMENSIONS (padded height), so a
    // naturally-sized width*height*4 vector would be read past its heap allocation. Short input is
    // zero-padded — missing tail texels detile as zero, matching the pointer overload's OOB policy.
    const size_t need = tiled_surface_bytes(width, height, tile_mode, pitch);
    if (src.size() < need) {
        std::vector<uint8_t> padded(need, 0);
        std::memcpy(padded.data(), src.data(), src.size());
        detile_surface(out.data(), padded.data(), width, height, tile_mode, pitch);
    } else {
        detile_surface(out.data(), src.data(), width, height, tile_mode, pitch);
    }
    return out;
}

void tile_surface(uint8_t* dst, const uint8_t* src, uint32_t width, uint32_t height,
                  uint32_t tile_mode, uint32_t pitch) {
    if (!tile_mode_is_tiled(tile_mode)) { std::memcpy(dst, src, (size_t)width * height * 4); return; }
    const size_t dst_bytes = tiled_surface_bytes(width, height, tile_mode, pitch);
    std::memset(dst, 0, dst_bytes);   // padding texels (rows beyond height) stay zero
    sw4kb_s_copy<true>(dst, src, width, height, pitch, dst_bytes);
}

} // namespace prosper::gpu
