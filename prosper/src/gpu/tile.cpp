// tile.cpp — see tile.hpp. GFX10 SW_4KB_S de-swizzle, generalized over the element size (#119).
#include "tile.hpp"
#include <cstring>
#include <algorithm>

namespace prosper::gpu {

bool tile_mode_is_tiled(uint32_t tile_mode) {
    return tile_mode == (uint32_t)TileMode::Sw4KbS;
}

namespace {
// 4KB standard-swizzle micro-tile geometry for bpe-byte elements: a tile is a FIXED 4096 bytes,
// so it holds 4096/bpe elements, arranged 2^bx wide x 2^by tall with bx >= by (wide-before-tall):
//   1 B -> 64x64,  2 B -> 64x32,  4 B -> 32x32,  8 B -> 32x16,  16 B -> 16x16.
// The old code hardcoded 4 bytes/element (32x32 geometry) everywhere, so any 8/16-bpp surface
// detiled with the wrong tile dimensions — every texel mis-ordered (#119).
// CONFIDENCE: HIGH for 4 B (live pixel-verified) and 16 B (BC3, llvmpipe re-tile verified);
// MED for the rest (standard 4KB-tile arithmetic; verify vs a real surface via PROSPER_DUMP_RAWTILE).
inline void sw4kb_dims(uint32_t bpe, uint32_t& bx, uint32_t& by) {
    uint32_t bits = 0; while ((4096u >> bits) > bpe) bits++;   // log2(4096/bpe), bpe a power of two
    bx = (bits + 1) / 2; by = bits / 2;
}
// GFX10 SW_4KB_S element order within the tile. The order is BYTES-PER-ELEMENT-DEPENDENT (AMD's
// standard-swizzle SW_PATTERN genuinely differs per bpp) — two orders are pixel-verified against The
// Messenger's live surfaces, and both share the shape "a low block of L X-bits then L Y-bits, then
// interleaved (y,x) Morton pairs above":
//
//   32-bpp (32x32 tile, bx=by=5): L=2 -> [x0,x1, y0,y1, y2,x2, y3,x3, y4,x4]           #118 (title art)
//    8-bpp (64x64 tile, bx=by=6): L=4 -> [x0,x1,x2,x3, y0,y1,y2,y3, y4,x4, y5,x5]       (R8 font atlas)
//
// The previous code applied the 32-bpp L=2 pattern at EVERY bpe. That serrated edges at 32-bpp until
// #118 fixed the low nibble, but at 8-bpp (the game's ONLY R8 surface — the 2048x1024 caption font
// atlas) the L=2 pattern is badly wrong: it scrambles every 64x64 tile into an unreadable weave, so
// the intro caption text renders as invisible/garbage. The 8-bpp order below is pixel-verified: it
// resolves the atlas into a clean grid of readable Latin + CJK glyphs. CONFIDENCE: HIGH for bx=by=5
// (32-bpp) and bx=by=6 (8-bpp) — both live-verified; the L=2 fallback covers 16-bpp/BC-block geometries
// (bx=by=4, bx=5/by=4) which round-trip but have no game-observed instance to pixel-verify yet.
inline uint32_t sw4kb_morton(uint32_t ix, uint32_t iy, uint32_t bx, uint32_t by) {
    uint32_t m = 0;
    // Low-block half-width L: 4 for the 64x64 (8-bpp) tile, else 2 (the #118-verified 32-bpp nibble).
    const uint32_t L = (bx == 6 && by == 6) ? 4u : 2u;
    uint32_t bit = 0;
    for (uint32_t b = 0; b < L && b < bx; b++) m |= ((ix >> b) & 1u) << (bit++);   // low X bits
    for (uint32_t b = 0; b < L && b < by; b++) m |= ((iy >> b) & 1u) << (bit++);   // low Y bits
    for (uint32_t b = L; b < by; b++) {                                            // (y,x) Morton pairs
        m |= ((iy >> b) & 1u) << (bit++);
        if (b < bx) m |= ((ix >> b) & 1u) << (bit++);
    }
    for (uint32_t b = by; b < bx; b++) m |= ((ix >> b) & 1u) << (bit++);           // wider-dim extra X
    return m;
}
// Element (x,y) -> its linear element INDEX in the tiled surface: tiles laid out row-major over the
// padded pitch, sw4kb_morton within a tile.
inline uint64_t sw4kb_index(uint32_t x, uint32_t y, uint32_t tiles_per_row, uint32_t bx, uint32_t by) {
    uint32_t tw = 1u << bx, th = 1u << by;
    uint32_t tx = x >> bx, ty = y >> by, ix = x & (tw - 1), iy = y & (th - 1);
    return (uint64_t)(ty * tiles_per_row + tx) * ((uint64_t)tw * th) + sw4kb_morton(ix, iy, bx, by);
}

// The single tiled<->linear walk every public function shares: per element, the tiled offset
// (Morton) and the linear offset, moving `bpe` bytes in the chosen direction so the index math can
// never diverge between tile and detile. `tiled_bytes` bounds the tiled side; an out-of-range tiled
// element reads as zero when detiling and is skipped when tiling (tile pre-zeroes the destination).
template <bool ToTiled>
void sw4kb_copy(uint8_t* dst, const uint8_t* src, uint32_t ew, uint32_t eh, uint32_t pitch,
                uint32_t bpe, size_t tiled_bytes) {
    uint32_t bx = 0, by = 0; sw4kb_dims(bpe, bx, by);
    uint32_t pw = pitch ? pitch : ew;
    uint32_t tiles_per_row = (pw + (1u << bx) - 1) >> bx;
    for (uint32_t y = 0; y < eh; y++)
        for (uint32_t x = 0; x < ew; x++) {
            uint64_t t = sw4kb_index(x, y, tiles_per_row, bx, by) * bpe;   // tiled offset
            size_t   l = ((size_t)y * ew + x) * bpe;                       // linear offset
            if (ToTiled) { if (t + bpe <= tiled_bytes) std::memcpy(dst + t, src + l, bpe); }
            else         { if (t + bpe <= tiled_bytes) std::memcpy(dst + l, src + t, bpe);
                           else                        std::memset(dst + l, 0, bpe); }
        }
}

size_t sw4kb_tiled_bytes(uint32_t ew, uint32_t eh, uint32_t pitch, uint32_t bpe) {
    uint32_t bx = 0, by = 0; sw4kb_dims(bpe, bx, by);
    uint32_t pw = pitch ? pitch : ew;
    uint32_t tiles_per_row = (pw + (1u << bx) - 1) >> bx;
    uint32_t tile_rows     = (eh + (1u << by) - 1) >> by;   // pad up to whole tiles
    return (size_t)tiles_per_row * tile_rows * 4096;        // a 4KB tile is 4096 bytes at ANY bpe
}
} // namespace

size_t tiled_surface_bytes(uint32_t width, uint32_t height, uint32_t tile_mode, uint32_t pitch,
                           uint32_t bytes_per_texel) {
    if (!tile_mode_is_tiled(tile_mode)) return (size_t)width * height * bytes_per_texel;
    return sw4kb_tiled_bytes(width, height, pitch, bytes_per_texel);
}

void detile_surface(uint8_t* dst, const uint8_t* src, uint32_t width, uint32_t height,
                    uint32_t tile_mode, uint32_t pitch, uint32_t bytes_per_texel) {
    if (!tile_mode_is_tiled(tile_mode)) { std::memcpy(dst, src, (size_t)width * height * bytes_per_texel); return; }
    sw4kb_copy<false>(dst, src, width, height, pitch, bytes_per_texel,
                      sw4kb_tiled_bytes(width, height, pitch, bytes_per_texel));
}

std::vector<uint8_t> detile_surface(const std::vector<uint8_t>& src, uint32_t width, uint32_t height,
                                    uint32_t tile_mode, uint32_t pitch, uint32_t bytes_per_texel) {
    std::vector<uint8_t> out((size_t)width * height * bytes_per_texel, 0);
    if (src.empty()) return out;
    // Enforce the "src holds at least tiled_surface_bytes" precondition here instead of trusting the
    // caller: the pointer overload's bounds guard is computed from the DIMENSIONS (padded height), so a
    // naturally-sized width*height*bpt vector would be read past its heap allocation. Short input is
    // zero-padded — missing tail texels detile as zero, matching the pointer overload's OOB policy.
    const size_t need = tiled_surface_bytes(width, height, tile_mode, pitch, bytes_per_texel);
    if (src.size() < need) {
        std::vector<uint8_t> padded(need, 0);
        std::memcpy(padded.data(), src.data(), src.size());
        detile_surface(out.data(), padded.data(), width, height, tile_mode, pitch, bytes_per_texel);
    } else {
        detile_surface(out.data(), src.data(), width, height, tile_mode, pitch, bytes_per_texel);
    }
    return out;
}

void tile_surface(uint8_t* dst, const uint8_t* src, uint32_t width, uint32_t height,
                  uint32_t tile_mode, uint32_t pitch, uint32_t bytes_per_texel) {
    if (!tile_mode_is_tiled(tile_mode)) { std::memcpy(dst, src, (size_t)width * height * bytes_per_texel); return; }
    const size_t dst_bytes = tiled_surface_bytes(width, height, tile_mode, pitch, bytes_per_texel);
    std::memset(dst, 0, dst_bytes);   // padding texels (rows beyond height) stay zero
    sw4kb_copy<true>(dst, src, width, height, pitch, bytes_per_texel, dst_bytes);
}

size_t tiled_elements_bytes(uint32_t ew, uint32_t eh, uint32_t bpe, uint32_t tile_mode) {
    if (!tile_mode_is_tiled(tile_mode) || bpe == 0) return (size_t)ew * eh * bpe;
    return sw4kb_tiled_bytes(ew, eh, /*pitch*/0, bpe);
}

void detile_elements(uint8_t* dst, const uint8_t* src, size_t src_bytes,
                     uint32_t ew, uint32_t eh, uint32_t bpe, uint32_t tile_mode) {
    if (!tile_mode_is_tiled(tile_mode) || bpe == 0) {
        size_t n = std::min(src_bytes, (size_t)ew * eh * bpe);
        std::memcpy(dst, src, n);
        if (n < (size_t)ew * eh * bpe) std::memset(dst + n, 0, (size_t)ew * eh * bpe - n);
        return;
    }
    sw4kb_copy<false>(dst, src, ew, eh, /*pitch*/0, bpe, src_bytes);
}

} // namespace prosper::gpu
