// tile.cpp — see tile.hpp. GFX10 SW_4KB_S de-swizzle, generalized over the element size (#119).
// GFX10 SW_64KB_S / SW_64KB_R_X de-swizzle from the AMD addrlib swizzle-pattern tables (#288).
#include "tile.hpp"
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <vector>

namespace prosper::gpu {

bool tile_mode_is_tiled(uint32_t tile_mode) {
    return tile_mode == (uint32_t)TileMode::Sw4KbS ||
           tile_mode == (uint32_t)TileMode::Sw64KbS ||
           tile_mode == (uint32_t)TileMode::Sw64KbRX;
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

// ---------------------------------------------------------------------------------------------------
// GFX10 64KB swizzles: SW_64KB_S (tile_mode 9) and SW_64KB_R_X (tile_mode 27) — issue #288.
//
// Source of truth: AMD addrlib gfx10 (Mesa src/amd/addrlib/src/gfx10/gfx10SwizzlePattern.h +
// core/addrlib.cpp ComputeOffsetFromSwizzlePattern). A 64KB block holds 2^16 bytes; each of the 16
// byte-offset bits within the block is the XOR of a set of ELEMENT-coordinate bits given by an
// (x-mask, y-mask) pair below (addrlib's ADDR_BIT_SETTING with the z/slice and s/sample masks dropped:
// this detiler only handles 2D single-sample surfaces, where z == 0 and s == 0 make those terms
// vanish). Blocks are laid out row-major over the padded pitch (addrlib
// ComputeSurfaceAddrFromCoordMacroTiled: addr = blkIdx * 64K + patternOffset(x, y)).
//
// Cross-validation of the extraction pipeline: the same addrlib tables reproduce prosper's two
// PIXEL-VERIFIED SW_4KB_S orders exactly (GFX10_SW_4K_S nibble expansion at 4 bpe ==
// [x0 x1 y0 y1 y2 x2 y3 x3 y4 x4] == sw4kb_morton L=2; at 1 bpe == the verified L=4 font-atlas
// order), and SW_64KB_S decodes a live-captured DOLL 1024x512 BC1 material texture into a coherent
// sprite atlas (docs/GFX10_SW_64KB_TILING.md). CONFIDENCE: HIGH for SW_64KB_S (addrlib equation +
// live-texture validation). SW_64KB_R_X additionally XORs pipe bits into offset bits 8..8+log2(pipes)-1;
// the pattern depends on the GPU's pipe count, which for PS5 is fixed hardware but not publicly
// documented — see sw64kb_rx_pipes_log2 below. CONFIDENCE: MED for R_X (equation exact per addrlib,
// pipe-count parameter selected empirically).
struct PatBit { uint16_t x, y; };

// SW_64K_S: one 16-bit pattern per element size (identical for every pipe count / RB+ in addrlib).
static const PatBit kSw64kS[5][16] = {
    { {0x0001,0x0000}, {0x0002,0x0000}, {0x0004,0x0000}, {0x0008,0x0000}, {0x0000,0x0001}, {0x0000,0x0002}, {0x0000,0x0004}, {0x0000,0x0008}, {0x0000,0x0010}, {0x0010,0x0000}, {0x0000,0x0020}, {0x0020,0x0000}, {0x0000,0x0040}, {0x0040,0x0000}, {0x0000,0x0080}, {0x0080,0x0000} },  // 1 bpe
    { {0x0000,0x0000}, {0x0001,0x0000}, {0x0002,0x0000}, {0x0004,0x0000}, {0x0000,0x0001}, {0x0000,0x0002}, {0x0000,0x0004}, {0x0008,0x0000}, {0x0000,0x0008}, {0x0010,0x0000}, {0x0000,0x0010}, {0x0020,0x0000}, {0x0000,0x0020}, {0x0040,0x0000}, {0x0000,0x0040}, {0x0080,0x0000} },  // 2 bpe
    { {0x0000,0x0000}, {0x0000,0x0000}, {0x0001,0x0000}, {0x0002,0x0000}, {0x0000,0x0001}, {0x0000,0x0002}, {0x0000,0x0004}, {0x0004,0x0000}, {0x0000,0x0008}, {0x0008,0x0000}, {0x0000,0x0010}, {0x0010,0x0000}, {0x0000,0x0020}, {0x0020,0x0000}, {0x0000,0x0040}, {0x0040,0x0000} },  // 4 bpe
    { {0x0000,0x0000}, {0x0000,0x0000}, {0x0000,0x0000}, {0x0001,0x0000}, {0x0000,0x0001}, {0x0000,0x0002}, {0x0002,0x0000}, {0x0004,0x0000}, {0x0000,0x0004}, {0x0008,0x0000}, {0x0000,0x0008}, {0x0010,0x0000}, {0x0000,0x0010}, {0x0020,0x0000}, {0x0000,0x0020}, {0x0040,0x0000} },  // 8 bpe
    { {0x0000,0x0000}, {0x0000,0x0000}, {0x0000,0x0000}, {0x0000,0x0000}, {0x0000,0x0001}, {0x0000,0x0002}, {0x0001,0x0000}, {0x0002,0x0000}, {0x0000,0x0004}, {0x0004,0x0000}, {0x0000,0x0008}, {0x0008,0x0000}, {0x0000,0x0010}, {0x0010,0x0000}, {0x0000,0x0020}, {0x0020,0x0000} },  // 16 bpe
};

// SW_64K_R_X (1 fragment): the pattern depends on the pipe count; [pipesLog2][elemLog2][bit].
// From GFX10_SW_64K_R_X_1xaa_PATINFO (Navi1x) with z-terms dropped (2D, slice 0).
static const PatBit kSw64kRX[7][5][16] = {
  { // 1 pipe
    { {0x0001,0x0000}, {0x0002,0x0000}, {0x0004,0x0000}, {0x0000,0x0002}, {0x0000,0x0001}, {0x0000,0x0004}, {0x0008,0x0000}, {0x0000,0x0008}, {0x0000,0x0010}, {0x0010,0x0000}, {0x0000,0x0020}, {0x0020,0x0000}, {0x0000,0x0040}, {0x0040,0x0000}, {0x0000,0x0080}, {0x0080,0x0000} },  // 1 bpe
    { {0x0000,0x0000}, {0x0001,0x0000}, {0x0002,0x0000}, {0x0004,0x0000}, {0x0000,0x0001}, {0x0000,0x0002}, {0x0000,0x0004}, {0x0008,0x0000}, {0x0000,0x0008}, {0x0010,0x0000}, {0x0000,0x0010}, {0x0020,0x0000}, {0x0000,0x0020}, {0x0040,0x0000}, {0x0000,0x0040}, {0x0080,0x0000} },  // 2 bpe
    { {0x0000,0x0000}, {0x0000,0x0000}, {0x0001,0x0000}, {0x0002,0x0000}, {0x0000,0x0001}, {0x0000,0x0002}, {0x0000,0x0004}, {0x0004,0x0000}, {0x0000,0x0008}, {0x0008,0x0000}, {0x0000,0x0010}, {0x0010,0x0000}, {0x0000,0x0020}, {0x0020,0x0000}, {0x0000,0x0040}, {0x0040,0x0000} },  // 4 bpe
    { {0x0000,0x0000}, {0x0000,0x0000}, {0x0000,0x0000}, {0x0001,0x0000}, {0x0000,0x0001}, {0x0002,0x0000}, {0x0004,0x0000}, {0x0000,0x0002}, {0x0000,0x0004}, {0x0008,0x0000}, {0x0000,0x0008}, {0x0010,0x0000}, {0x0000,0x0010}, {0x0020,0x0000}, {0x0000,0x0020}, {0x0040,0x0000} },  // 8 bpe
    { {0x0000,0x0000}, {0x0000,0x0000}, {0x0000,0x0000}, {0x0000,0x0000}, {0x0001,0x0000}, {0x0000,0x0001}, {0x0002,0x0000}, {0x0000,0x0002}, {0x0000,0x0004}, {0x0004,0x0000}, {0x0000,0x0008}, {0x0008,0x0000}, {0x0000,0x0010}, {0x0010,0x0000}, {0x0000,0x0020}, {0x0020,0x0000} },  // 16 bpe
  },
  { // 2 pipes
    { {0x0001,0x0000}, {0x0002,0x0000}, {0x0004,0x0000}, {0x0000,0x0002}, {0x0000,0x0001}, {0x0000,0x0004}, {0x0008,0x0000}, {0x0000,0x0010}, {0x0008,0x0008}, {0x0010,0x0000}, {0x0000,0x0020}, {0x0020,0x0000}, {0x0000,0x0040}, {0x0040,0x0000}, {0x0000,0x0080}, {0x0080,0x0000} },  // 1 bpe
    { {0x0000,0x0000}, {0x0001,0x0000}, {0x0002,0x0000}, {0x0004,0x0000}, {0x0000,0x0001}, {0x0000,0x0002}, {0x0000,0x0004}, {0x0008,0x0000}, {0x0008,0x0008}, {0x0010,0x0000}, {0x0000,0x0010}, {0x0020,0x0000}, {0x0000,0x0020}, {0x0040,0x0000}, {0x0000,0x0040}, {0x0080,0x0000} },  // 2 bpe
    { {0x0000,0x0000}, {0x0000,0x0000}, {0x0001,0x0000}, {0x0002,0x0000}, {0x0000,0x0001}, {0x0000,0x0002}, {0x0000,0x0004}, {0x0004,0x0000}, {0x0008,0x0008}, {0x0008,0x0000}, {0x0000,0x0010}, {0x0010,0x0000}, {0x0000,0x0020}, {0x0020,0x0000}, {0x0000,0x0040}, {0x0040,0x0000} },  // 4 bpe
    { {0x0000,0x0000}, {0x0000,0x0000}, {0x0000,0x0000}, {0x0001,0x0000}, {0x0000,0x0001}, {0x0002,0x0000}, {0x0004,0x0000}, {0x0000,0x0002}, {0x0008,0x0008}, {0x0008,0x0000}, {0x0000,0x0004}, {0x0010,0x0000}, {0x0000,0x0010}, {0x0020,0x0000}, {0x0000,0x0020}, {0x0040,0x0000} },  // 8 bpe
    { {0x0000,0x0000}, {0x0000,0x0000}, {0x0000,0x0000}, {0x0000,0x0000}, {0x0001,0x0000}, {0x0000,0x0001}, {0x0002,0x0000}, {0x0000,0x0002}, {0x0008,0x0008}, {0x0004,0x0000}, {0x0000,0x0004}, {0x0008,0x0000}, {0x0000,0x0010}, {0x0010,0x0000}, {0x0000,0x0020}, {0x0020,0x0000} },  // 16 bpe
  },
  { // 4 pipes
    { {0x0001,0x0000}, {0x0002,0x0000}, {0x0004,0x0000}, {0x0000,0x0002}, {0x0000,0x0001}, {0x0000,0x0004}, {0x0008,0x0000}, {0x0000,0x0010}, {0x0008,0x0008}, {0x0010,0x0010}, {0x0000,0x0020}, {0x0020,0x0000}, {0x0000,0x0040}, {0x0040,0x0000}, {0x0000,0x0080}, {0x0080,0x0000} },  // 1 bpe
    { {0x0000,0x0000}, {0x0001,0x0000}, {0x0002,0x0000}, {0x0004,0x0000}, {0x0000,0x0001}, {0x0000,0x0002}, {0x0000,0x0004}, {0x0008,0x0000}, {0x0008,0x0008}, {0x0010,0x0010}, {0x0000,0x0010}, {0x0020,0x0000}, {0x0000,0x0020}, {0x0040,0x0000}, {0x0000,0x0040}, {0x0080,0x0000} },  // 2 bpe
    { {0x0000,0x0000}, {0x0000,0x0000}, {0x0001,0x0000}, {0x0002,0x0000}, {0x0000,0x0001}, {0x0000,0x0002}, {0x0000,0x0004}, {0x0004,0x0000}, {0x0008,0x0008}, {0x0010,0x0010}, {0x0000,0x0008}, {0x0010,0x0000}, {0x0000,0x0020}, {0x0020,0x0000}, {0x0000,0x0040}, {0x0040,0x0000} },  // 4 bpe
    { {0x0000,0x0000}, {0x0000,0x0000}, {0x0000,0x0000}, {0x0001,0x0000}, {0x0000,0x0001}, {0x0002,0x0000}, {0x0004,0x0000}, {0x0000,0x0002}, {0x0008,0x0008}, {0x0010,0x0010}, {0x0000,0x0004}, {0x0008,0x0000}, {0x0000,0x0010}, {0x0020,0x0000}, {0x0000,0x0020}, {0x0040,0x0000} },  // 8 bpe
    { {0x0000,0x0000}, {0x0000,0x0000}, {0x0000,0x0000}, {0x0000,0x0000}, {0x0001,0x0000}, {0x0000,0x0001}, {0x0002,0x0000}, {0x0000,0x0002}, {0x0008,0x0008}, {0x0010,0x0010}, {0x0000,0x0004}, {0x0004,0x0000}, {0x0000,0x0008}, {0x0010,0x0000}, {0x0000,0x0020}, {0x0020,0x0000} },  // 16 bpe
  },
  { // 8 pipes
    { {0x0001,0x0000}, {0x0002,0x0000}, {0x0004,0x0000}, {0x0000,0x0002}, {0x0000,0x0001}, {0x0000,0x0004}, {0x0008,0x0000}, {0x0000,0x0010}, {0x0008,0x0008}, {0x0010,0x0010}, {0x0020,0x0020}, {0x0020,0x0000}, {0x0000,0x0040}, {0x0040,0x0000}, {0x0000,0x0080}, {0x0080,0x0000} },  // 1 bpe
    { {0x0000,0x0000}, {0x0001,0x0000}, {0x0002,0x0000}, {0x0004,0x0000}, {0x0000,0x0001}, {0x0000,0x0002}, {0x0000,0x0004}, {0x0008,0x0000}, {0x0008,0x0008}, {0x0010,0x0010}, {0x0020,0x0020}, {0x0010,0x0000}, {0x0000,0x0020}, {0x0040,0x0000}, {0x0000,0x0040}, {0x0080,0x0000} },  // 2 bpe
    { {0x0000,0x0000}, {0x0000,0x0000}, {0x0001,0x0000}, {0x0002,0x0000}, {0x0000,0x0001}, {0x0000,0x0002}, {0x0000,0x0004}, {0x0004,0x0000}, {0x0008,0x0008}, {0x0010,0x0010}, {0x0020,0x0020}, {0x0008,0x0000}, {0x0000,0x0010}, {0x0020,0x0000}, {0x0000,0x0040}, {0x0040,0x0000} },  // 4 bpe
    { {0x0000,0x0000}, {0x0000,0x0000}, {0x0000,0x0000}, {0x0001,0x0000}, {0x0000,0x0001}, {0x0002,0x0000}, {0x0004,0x0000}, {0x0000,0x0002}, {0x0008,0x0008}, {0x0010,0x0010}, {0x0020,0x0020}, {0x0008,0x0000}, {0x0000,0x0004}, {0x0010,0x0000}, {0x0000,0x0020}, {0x0040,0x0000} },  // 8 bpe
    { {0x0000,0x0000}, {0x0000,0x0000}, {0x0000,0x0000}, {0x0000,0x0000}, {0x0001,0x0000}, {0x0000,0x0001}, {0x0002,0x0000}, {0x0000,0x0002}, {0x0008,0x0008}, {0x0010,0x0010}, {0x0020,0x0020}, {0x0004,0x0000}, {0x0000,0x0004}, {0x0008,0x0000}, {0x0000,0x0010}, {0x0020,0x0000} },  // 16 bpe
  },
  { // 16 pipes
    { {0x0001,0x0000}, {0x0002,0x0000}, {0x0004,0x0000}, {0x0000,0x0002}, {0x0000,0x0001}, {0x0000,0x0004}, {0x0008,0x0000}, {0x0000,0x0010}, {0x0008,0x0008}, {0x0010,0x0010}, {0x0040,0x0020}, {0x0020,0x0040}, {0x0000,0x0040}, {0x0040,0x0000}, {0x0000,0x0080}, {0x0080,0x0000} },  // 1 bpe
    { {0x0000,0x0000}, {0x0001,0x0000}, {0x0002,0x0000}, {0x0004,0x0000}, {0x0000,0x0001}, {0x0000,0x0002}, {0x0000,0x0004}, {0x0008,0x0000}, {0x0008,0x0008}, {0x0010,0x0010}, {0x0040,0x0020}, {0x0020,0x0040}, {0x0000,0x0010}, {0x0040,0x0000}, {0x0000,0x0040}, {0x0080,0x0000} },  // 2 bpe
    { {0x0000,0x0000}, {0x0000,0x0000}, {0x0001,0x0000}, {0x0002,0x0000}, {0x0000,0x0001}, {0x0000,0x0002}, {0x0000,0x0004}, {0x0004,0x0000}, {0x0008,0x0008}, {0x0010,0x0010}, {0x0040,0x0020}, {0x0020,0x0040}, {0x0000,0x0008}, {0x0010,0x0000}, {0x0000,0x0040}, {0x0040,0x0000} },  // 4 bpe
    { {0x0000,0x0000}, {0x0000,0x0000}, {0x0000,0x0000}, {0x0001,0x0000}, {0x0000,0x0001}, {0x0002,0x0000}, {0x0004,0x0000}, {0x0000,0x0002}, {0x0008,0x0008}, {0x0010,0x0010}, {0x0040,0x0020}, {0x0020,0x0040}, {0x0000,0x0004}, {0x0008,0x0000}, {0x0000,0x0010}, {0x0040,0x0000} },  // 8 bpe
    { {0x0000,0x0000}, {0x0000,0x0000}, {0x0000,0x0000}, {0x0000,0x0000}, {0x0001,0x0000}, {0x0000,0x0001}, {0x0002,0x0000}, {0x0000,0x0002}, {0x0008,0x0008}, {0x0010,0x0010}, {0x0040,0x0020}, {0x0020,0x0040}, {0x0000,0x0004}, {0x0004,0x0000}, {0x0000,0x0008}, {0x0010,0x0000} },  // 16 bpe
  },
  { // 32 pipes
    { {0x0001,0x0000}, {0x0002,0x0000}, {0x0004,0x0000}, {0x0000,0x0002}, {0x0000,0x0001}, {0x0000,0x0004}, {0x0008,0x0000}, {0x0000,0x0010}, {0x0008,0x0008}, {0x0010,0x0010}, {0x0080,0x0020}, {0x0020,0x0080}, {0x0040,0x0040}, {0x0040,0x0000}, {0x0000,0x0080}, {0x0080,0x0000} },  // 1 bpe
    { {0x0000,0x0000}, {0x0001,0x0000}, {0x0002,0x0000}, {0x0004,0x0000}, {0x0000,0x0001}, {0x0000,0x0002}, {0x0000,0x0004}, {0x0008,0x0000}, {0x0008,0x0008}, {0x0010,0x0010}, {0x0080,0x0020}, {0x0020,0x0080}, {0x0040,0x0040}, {0x0010,0x0000}, {0x0000,0x0040}, {0x0080,0x0000} },  // 2 bpe
    { {0x0000,0x0000}, {0x0000,0x0000}, {0x0001,0x0000}, {0x0002,0x0000}, {0x0000,0x0001}, {0x0000,0x0002}, {0x0000,0x0004}, {0x0004,0x0000}, {0x0008,0x0008}, {0x0010,0x0010}, {0x0080,0x0020}, {0x0020,0x0080}, {0x0040,0x0040}, {0x0008,0x0000}, {0x0000,0x0010}, {0x0040,0x0000} },  // 4 bpe
    { {0x0000,0x0000}, {0x0000,0x0000}, {0x0000,0x0000}, {0x0001,0x0000}, {0x0000,0x0001}, {0x0002,0x0000}, {0x0004,0x0000}, {0x0000,0x0002}, {0x0008,0x0008}, {0x0010,0x0010}, {0x0080,0x0020}, {0x0020,0x0080}, {0x0040,0x0040}, {0x0000,0x0004}, {0x0008,0x0000}, {0x0000,0x0010} },  // 8 bpe
    { {0x0000,0x0000}, {0x0000,0x0000}, {0x0000,0x0000}, {0x0000,0x0000}, {0x0001,0x0000}, {0x0000,0x0001}, {0x0002,0x0000}, {0x0000,0x0002}, {0x0008,0x0008}, {0x0010,0x0010}, {0x0080,0x0020}, {0x0020,0x0080}, {0x0040,0x0044}, {0x0004,0x0000}, {0x0000,0x0008}, {0x0010,0x0000} },  // 16 bpe
  },
  { // 64 pipes
    { {0x0001,0x0000}, {0x0002,0x0000}, {0x0004,0x0000}, {0x0000,0x0002}, {0x0000,0x0001}, {0x0000,0x0004}, {0x0008,0x0000}, {0x0000,0x0010}, {0x0008,0x0008}, {0x0010,0x0010}, {0x0100,0x0020}, {0x0020,0x0100}, {0x0080,0x0040}, {0x0040,0x0080}, {0x0000,0x0080}, {0x0080,0x0000} },  // 1 bpe
    { {0x0000,0x0000}, {0x0001,0x0000}, {0x0002,0x0000}, {0x0004,0x0000}, {0x0000,0x0001}, {0x0000,0x0002}, {0x0000,0x0004}, {0x0008,0x0000}, {0x0008,0x0008}, {0x0010,0x0010}, {0x0100,0x0020}, {0x0020,0x0100}, {0x0080,0x0040}, {0x0040,0x0080}, {0x0000,0x0010}, {0x0080,0x0000} },  // 2 bpe
    { {0x0000,0x0000}, {0x0000,0x0000}, {0x0001,0x0000}, {0x0002,0x0000}, {0x0000,0x0001}, {0x0000,0x0002}, {0x0000,0x0004}, {0x0004,0x0000}, {0x0008,0x0008}, {0x0010,0x0010}, {0x0100,0x0020}, {0x0020,0x0100}, {0x0080,0x0040}, {0x0040,0x0080}, {0x0000,0x0008}, {0x0010,0x0000} },  // 4 bpe
    { {0x0000,0x0000}, {0x0000,0x0000}, {0x0000,0x0000}, {0x0001,0x0000}, {0x0000,0x0001}, {0x0002,0x0000}, {0x0004,0x0000}, {0x0000,0x0002}, {0x0008,0x0008}, {0x0010,0x0010}, {0x0100,0x0020}, {0x0020,0x0100}, {0x0080,0x0044}, {0x0040,0x0080}, {0x0000,0x0008}, {0x0010,0x0000} },  // 8 bpe
    { {0x0000,0x0000}, {0x0000,0x0000}, {0x0000,0x0000}, {0x0000,0x0000}, {0x0001,0x0000}, {0x0000,0x0001}, {0x0002,0x0000}, {0x0000,0x0002}, {0x0008,0x0008}, {0x0010,0x0010}, {0x0100,0x0020}, {0x0020,0x0100}, {0x0080,0x0044}, {0x0044,0x0080}, {0x0000,0x0008}, {0x0010,0x0000} },  // 16 bpe
  },
};

// elemLog2 for a power-of-two bpe in [1,16]; returns UINT32_MAX for sizes the tables don't cover.
inline uint32_t sw64kb_elem_log2(uint32_t bpe) {
    switch (bpe) { case 1: return 0; case 2: return 1; case 4: return 2; case 8: return 3; case 16: return 4; }
    return UINT32_MAX;
}

// 64KB block dims in elements (addrlib Block256_2d scaled x16 in each axis: 8 extra bits, 4 to each).
inline void sw64kb_dims(uint32_t elem_log2, uint32_t& bw, uint32_t& bh) {
    static const uint16_t w[5] = {256, 256, 128, 128, 64}, h[5] = {256, 128, 128, 64, 64};
    bw = w[elem_log2]; bh = h[elem_log2];
}

// PS5's pipe count for the R_X pattern. Not publicly documented for Oberon; default 16 pipes by
// hardware analogy (Oberon is Navi10-class: 36-40 CU, 256-bit GDDR6, 64 ROPs / 16 RBs, and
// Navi10's GB_ADDR_CONFIG is 16 pipes). Captured mode-27 surfaces are all speckle/checkerboard
// content on which every pipe variant scores identically (TV, mip-consistency, autocorrelation) —
// re-pin via PROSPER_RX_PIPES=<n> A/B when a smooth authored mode-27 surface appears.
// CONFIDENCE: MED (equation exact per addrlib; only this parameter is analogy-based).
inline uint32_t sw64kb_rx_pipes_log2() {
    static int cached = -1;
    if (cached < 0) {
        cached = 4;                                            // 16 pipes
        if (const char* e = std::getenv("PROSPER_RX_PIPES")) {
            int n = atoi(e);
            int lg = 0; while ((1 << lg) < n && lg < 6) lg++;
            if (n >= 1 && (1 << lg) == n) cached = lg;
        }
    }
    return (uint32_t)cached;
}

inline const PatBit* sw64kb_pattern(uint32_t tile_mode, uint32_t elem_log2) {
    if (tile_mode == (uint32_t)TileMode::Sw64KbRX) return kSw64kRX[sw64kb_rx_pipes_log2()][elem_log2];
    return kSw64kS[elem_log2];
}

size_t sw64kb_tiled_bytes(uint32_t ew, uint32_t eh, uint32_t pitch, uint32_t bpe) {
    uint32_t el = sw64kb_elem_log2(bpe);
    if (el == UINT32_MAX) return (size_t)ew * eh * bpe;        // unsupported bpe -> linear size
    uint32_t bw = 0, bh = 0; sw64kb_dims(el, bw, bh);
    uint32_t pw = pitch ? pitch : ew;
    uint32_t blocks_per_row = (pw + bw - 1) / bw;
    uint32_t block_rows     = (eh + bh - 1) / bh;
    return (size_t)blocks_per_row * block_rows * 65536;
}

// The 64KB tiled<->linear walk. The pattern offset is XOR-separable in x and y (each offset bit is
// parity(x&xm)^parity(y&ym)), so precompute fx[] / fy[] per coordinate and each element's in-block
// offset is fx[x]^fy[y] — exact per addrlib (patterns are evaluated on GLOBAL element coords; every
// mask stays within the coordinate range of one block for the pipe counts shipped here, but global
// coords keep even wider masks correct).
template <bool ToTiled>
void sw64kb_copy(uint8_t* dst, const uint8_t* src, uint32_t ew, uint32_t eh, uint32_t pitch,
                 uint32_t bpe, size_t tiled_bytes, uint32_t tile_mode) {
    uint32_t el = sw64kb_elem_log2(bpe);
    if (el == UINT32_MAX) {                                    // unsupported bpe -> straight copy
        size_t n = (size_t)ew * eh * bpe;
        if (ToTiled) std::memcpy(dst, src, std::min(n, tiled_bytes));
        else         std::memcpy(dst, src, std::min(n, tiled_bytes));
        return;
    }
    uint32_t bw = 0, bh = 0; sw64kb_dims(el, bw, bh);
    const PatBit* pat = sw64kb_pattern(tile_mode, el);
    uint32_t pw = pitch ? pitch : ew;
    uint32_t blocks_per_row = (pw + bw - 1) / bw;
    std::vector<uint16_t> fx(ew), fy(eh);
    for (uint32_t x = 0; x < ew; x++) {
        uint32_t v = 0;
        for (uint32_t i = el; i < 16; i++) v |= (uint32_t)(__builtin_popcount(x & pat[i].x) & 1) << i;
        fx[x] = (uint16_t)v;
    }
    for (uint32_t y = 0; y < eh; y++) {
        uint32_t v = 0;
        for (uint32_t i = el; i < 16; i++) v |= (uint32_t)(__builtin_popcount(y & pat[i].y) & 1) << i;
        fy[y] = (uint16_t)v;
    }
    for (uint32_t y = 0; y < eh; y++) {
        uint64_t brow = (uint64_t)(y / bh) * blocks_per_row;
        for (uint32_t x = 0; x < ew; x++) {
            uint64_t t = ((brow + x / bw) << 16) | (uint32_t)(fx[x] ^ fy[y]);
            size_t   l = ((size_t)y * ew + x) * bpe;
            if (ToTiled) { if (t + bpe <= tiled_bytes) std::memcpy(dst + t, src + l, bpe); }
            else         { if (t + bpe <= tiled_bytes) std::memcpy(dst + l, src + t, bpe);
                           else                        std::memset(dst + l, 0, bpe); }
        }
    }
}

inline bool is_64kb_mode(uint32_t tile_mode) {
    return tile_mode == (uint32_t)TileMode::Sw64KbS || tile_mode == (uint32_t)TileMode::Sw64KbRX;
}
} // namespace

size_t tiled_surface_bytes(uint32_t width, uint32_t height, uint32_t tile_mode, uint32_t pitch,
                           uint32_t bytes_per_texel) {
    if (!tile_mode_is_tiled(tile_mode)) return (size_t)width * height * bytes_per_texel;
    if (is_64kb_mode(tile_mode)) return sw64kb_tiled_bytes(width, height, pitch, bytes_per_texel);
    return sw4kb_tiled_bytes(width, height, pitch, bytes_per_texel);
}

void detile_surface(uint8_t* dst, const uint8_t* src, uint32_t width, uint32_t height,
                    uint32_t tile_mode, uint32_t pitch, uint32_t bytes_per_texel) {
    if (!tile_mode_is_tiled(tile_mode)) { std::memcpy(dst, src, (size_t)width * height * bytes_per_texel); return; }
    if (is_64kb_mode(tile_mode)) {
        sw64kb_copy<false>(dst, src, width, height, pitch, bytes_per_texel,
                           sw64kb_tiled_bytes(width, height, pitch, bytes_per_texel), tile_mode);
        return;
    }
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
    if (is_64kb_mode(tile_mode)) {
        sw64kb_copy<true>(dst, src, width, height, pitch, bytes_per_texel, dst_bytes, tile_mode);
        return;
    }
    sw4kb_copy<true>(dst, src, width, height, pitch, bytes_per_texel, dst_bytes);
}

size_t tiled_elements_bytes(uint32_t ew, uint32_t eh, uint32_t bpe, uint32_t tile_mode) {
    if (!tile_mode_is_tiled(tile_mode) || bpe == 0) return (size_t)ew * eh * bpe;
    if (is_64kb_mode(tile_mode)) return sw64kb_tiled_bytes(ew, eh, /*pitch*/0, bpe);
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
    if (is_64kb_mode(tile_mode)) {
        sw64kb_copy<false>(dst, src, ew, eh, /*pitch*/0, bpe, src_bytes, tile_mode);
        return;
    }
    sw4kb_copy<false>(dst, src, ew, eh, /*pitch*/0, bpe, src_bytes);
}

} // namespace prosper::gpu
