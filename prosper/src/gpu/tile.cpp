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
} // namespace

void detile_surface(uint8_t* dst, const uint8_t* src, uint32_t width, uint32_t height,
                    uint32_t tile_mode, uint32_t pitch) {
    if (!tile_mode_is_tiled(tile_mode)) { std::memcpy(dst, src, (size_t)width * height * 4); return; }
    uint32_t pw = pitch ? pitch : width;
    uint32_t tiles_per_row = (pw + 31) / 32;
    const size_t src_bytes = tiled_surface_bytes(width, height, tile_mode, pitch);
    for (uint32_t y = 0; y < height; y++)
        for (uint32_t x = 0; x < width; x++) {
            uint64_t s = sw4kb_s_index(x, y, tiles_per_row) * 4;
            uint8_t* d = dst + ((size_t)y * width + x) * 4;
            if (s + 4 <= src_bytes) std::memcpy(d, src + s, 4);
            else                    std::memset(d, 0, 4);
        }
}

std::vector<uint8_t> detile_surface(const std::vector<uint8_t>& src, uint32_t width, uint32_t height,
                                    uint32_t tile_mode, uint32_t pitch) {
    std::vector<uint8_t> out((size_t)width * height * 4, 0);
    if (!src.empty()) detile_surface(out.data(), src.data(), width, height, tile_mode, pitch);
    return out;
}

void tile_surface(uint8_t* dst, const uint8_t* src, uint32_t width, uint32_t height,
                  uint32_t tile_mode, uint32_t pitch) {
    if (!tile_mode_is_tiled(tile_mode)) { std::memcpy(dst, src, (size_t)width * height * 4); return; }
    uint32_t pw = pitch ? pitch : width;
    uint32_t tiles_per_row = (pw + 31) / 32;
    const size_t dst_bytes = tiled_surface_bytes(width, height, tile_mode, pitch);
    std::memset(dst, 0, dst_bytes);
    for (uint32_t y = 0; y < height; y++)
        for (uint32_t x = 0; x < width; x++) {
            uint64_t s = sw4kb_s_index(x, y, tiles_per_row) * 4;    // tiled position
            const uint8_t* srcpx = src + ((size_t)y * width + x) * 4;  // linear source
            if (s + 4 <= dst_bytes) std::memcpy(dst + s, srcpx, 4);
        }
}

} // namespace prosper::gpu
