// tile.cpp — see tile.hpp. GFX10 SW_4KB_S de-swizzle, generalized over the element size (#119).
// GFX10 SW_64KB_S / SW_64KB_R_X de-swizzle from the AMD addrlib swizzle-pattern tables (#288).
#include "tile.hpp"
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <mutex>
#include <set>
#include <vector>

namespace prosper::gpu {

bool tile_mode_is_tiled(uint32_t tile_mode) {
    return tile_mode == (uint32_t)TileMode::Sw4KbS ||
           tile_mode == (uint32_t)TileMode::Sw64KbS ||
           tile_mode == (uint32_t)TileMode::Sw64KbRX;
}

// One-time diagnostic when a NON-ZERO (tiled) tile_mode is not one of the swizzles we de-swizzle
// (Sw4KbS=5 / Sw64KbS=9 / Sw64KbRX=27): the caller then copies the surface VERBATIM as if linear, so
// it samples as a scrambled swizzle-weave. Other GFX10 modes a PS5 T# can legally carry — SW_256B_*,
// SW_4KB_D/*_X, SW_64KB_S_T/D_T, the SW_64KB_Z/D depth/displayable families, SW_VAR_* — all land here.
// Log once per unrecognized mode under PROSPER_GFXLOG so this silent linear-fallback is observable
// instead of masquerading as a correct linear copy (#383). No-op for mode 0 (genuinely linear).
static void warn_unhandled_tile_mode(uint32_t tile_mode, uint32_t w, uint32_t h) {
    if (tile_mode == 0 || tile_mode_is_tiled(tile_mode) || !getenv("PROSPER_GFXLOG")) return;
    static std::set<uint32_t> seen;
    static std::mutex mu;
    std::lock_guard<std::mutex> lk(mu);
    if (seen.insert(tile_mode).second)
        fprintf(stderr, "[tile] UNHANDLED GFX10 tile_mode=%u (%ux%u) -> copied as LINEAR; surface will "
                        "sample SCRAMBLED (only Sw4KbS=5/Sw64KbS=9/Sw64KbRX=27 are de-swizzled)\n",
                        tile_mode, w, h);
}

namespace {
// One addrlib swizzle-pattern entry: the (x-mask, y-mask) coordinate bits an offset bit is built from.
// Defined here (and the SW_64K_S table forward-declared) so sw4kb_morton below can derive the 4KB order
// from the low bits of the SAME authoritative 64KB pattern (#379); both are defined together lower down.
struct PatBit { uint16_t x, y; };
extern const PatBit kSw64kS[5][16];

// 4KB standard-swizzle micro-tile geometry for bpe-byte elements: a tile is a FIXED 4096 bytes,
// so it holds 4096/bpe elements, arranged 2^bx wide x 2^by tall with bx >= by (wide-before-tall):
//   1 B -> 64x64,  2 B -> 64x32,  4 B -> 32x32,  8 B -> 32x16,  16 B -> 16x16.
// The old code hardcoded 4 bytes/element (32x32 geometry) everywhere, so any 8/16-bpp surface
// detiled with the wrong tile dimensions — every texel mis-ordered (#119).
// CONFIDENCE: HIGH — sw4kb_dims is pure 4KB-tile arithmetic and sw4kb_morton now derives the within-
// tile order from the authoritative addrlib SW_64KB_S table (#379), which reproduces the 1 B / 4 B
// pixel-verified orders exactly, so all five element sizes share one validated source of truth.
inline void sw4kb_dims(uint32_t bpe, uint32_t& bx, uint32_t& by) {
    uint32_t bits = 0; while ((4096u >> bits) > bpe) bits++;   // log2(4096/bpe), bpe a power of two
    bx = (bits + 1) / 2; by = bits / 2;
}
// GFX10 SW_4KB_S element order within the tile. The order is BYTES-PER-ELEMENT-DEPENDENT (AMD's
// standard-swizzle SW_PATTERN genuinely differs per bpp). The 4KB order is exactly the LOW-BIT
// TRUNCATION of the authoritative addrlib SW_64KB_S pattern (docs/GFX10_SW_64KB_TILING.md), so we
// derive it from the SAME in-file kSw64kS table used by the 64KB detiler rather than an ad-hoc
// generator — one source of truth, correct at every element size.
//
// kSw64kS[elem_log2][k] gives, for byte-offset bit k of the 64KB block, the single element-coordinate
// bit (x-mask or y-mask) it carries. The first elem_log2 entries are the within-element byte bits
// ({0,0}); the next (bx+by) entries are this 4KB tile's element-index bits, low->high. So element
// index bit e reads pattern entry [elem_log2 + e], whose set x/y mask names the coordinate bit that
// lands at output bit e. This reproduces all five ground-truth orders (both derivations in #379 agree,
// and the two pixel-verified cases — 32-bpp #118 and 8-bpp R8 font atlas — come out identical):
//   1 B  (64x64): x0 x1 x2 x3 y0 y1 y2 y3 y4 x4 y5 x5
//   2 B  (64x32): x0 x1 x2 y0 y1 y2 x3 y3 x4 y4 x5
//   4 B  (32x32): x0 x1 y0 y1 y2 x2 y3 x3 y4 x4
//   8 B  (32x16): x0 y0 y1 x1 x2 y2 x3 y3 x4
//   16 B (16x16): y0 y1 x0 x1 y2 x2 y3 x3
// The prior L-generator was correct only at 1 B and 4 B; it swapped the low X/Y pairs at 2/8/16 B,
// scrambling any SW_4KB_S BC1/BC4 (8 B) / BC2/3/5/6/7 (16 B) / R16 (2 B) surface into a coherent weave
// (the #118/#102 failure class at those bpe). #379.
inline uint32_t sw4kb_morton(uint32_t ix, uint32_t iy, uint32_t bx, uint32_t by) {
    const uint32_t bits = bx + by;            // log2(elements per 4KB tile) = log2(4096/bpe)
    const uint32_t elem_log2 = 12u - bits;    // 4096 == 2^12, bpe == 2^elem_log2 -> bits == 12 - elem_log2
    uint32_t m = 0;
    for (uint32_t e = 0; e < bits; e++) {
        const PatBit& pb = kSw64kS[elem_log2][elem_log2 + e];
        if (pb.x) { uint32_t b = 0; while (!((pb.x >> b) & 1u)) b++; m |= ((ix >> b) & 1u) << e; }
        else if (pb.y) { uint32_t b = 0; while (!((pb.y >> b) & 1u)) b++; m |= ((iy >> b) & 1u) << e; }
        // pb == {0,0} cannot occur in a standard-swizzle element-addressing bit; if it ever did, that
        // output bit stays 0 (a benign degenerate) rather than looping forever on a zero mask.
    }
    return m;
}

struct Sw4kbLookup {
    uint32_t bx = 0, by = 0, tw = 0, th = 0;
    std::vector<uint16_t> byte_offsets;
};

Sw4kbLookup make_sw4kb_lookup(uint32_t bpe) {
    Sw4kbLookup lookup;
    sw4kb_dims(bpe, lookup.bx, lookup.by);
    lookup.tw = 1u << lookup.bx;
    lookup.th = 1u << lookup.by;
    lookup.byte_offsets.resize(static_cast<size_t>(lookup.tw) * lookup.th);
    for (uint32_t y = 0; y < lookup.th; ++y)
        for (uint32_t x = 0; x < lookup.tw; ++x)
            lookup.byte_offsets[static_cast<size_t>(y) * lookup.tw + x] =
                static_cast<uint16_t>(sw4kb_morton(x, y, lookup.bx, lookup.by) * bpe);
    return lookup;
}

const Sw4kbLookup& sw4kb_lookup(uint32_t bpe) {
    static const Sw4kbLookup b1 = make_sw4kb_lookup(1);
    static const Sw4kbLookup b2 = make_sw4kb_lookup(2);
    static const Sw4kbLookup b4 = make_sw4kb_lookup(4);
    static const Sw4kbLookup b8 = make_sw4kb_lookup(8);
    static const Sw4kbLookup b16 = make_sw4kb_lookup(16);
    switch (bpe) {
        case 1: return b1;
        case 2: return b2;
        case 4: return b4;
        case 8: return b8;
        case 16: return b16;
        default: return b1;
    }
}

// The single tiled<->linear walk every public function shares: per element, the tiled offset
// (Morton) and the linear offset, moving `bpe` bytes in the chosen direction so the index math can
// never diverge between tile and detile. `tiled_bytes` bounds the tiled side; an out-of-range tiled
// element reads as zero when detiling and is skipped when tiling (tile pre-zeroes the destination).
template <bool ToTiled>
void sw4kb_copy(uint8_t* dst, const uint8_t* src, uint32_t ew, uint32_t eh, uint32_t pitch,
                uint32_t bpe, size_t tiled_bytes) {
    const Sw4kbLookup& lookup = sw4kb_lookup(bpe);
    uint32_t pw = pitch ? pitch : ew;
    uint32_t tiles_per_row = (pw + lookup.tw - 1) / lookup.tw;
    uint32_t surface_tile_rows = (eh + lookup.th - 1) / lookup.th;
    uint32_t surface_tile_cols = (ew + lookup.tw - 1) / lookup.tw;
    for (uint32_t ty = 0; ty < surface_tile_rows; ++ty) {
        const uint32_t rows = std::min(lookup.th, eh - ty * lookup.th);
        for (uint32_t tx = 0; tx < surface_tile_cols; ++tx) {
            const uint32_t columns = std::min(lookup.tw, ew - tx * lookup.tw);
            const size_t tile_base = static_cast<size_t>(ty * tiles_per_row + tx) * 4096;
            for (uint32_t iy = 0; iy < rows; ++iy) {
                const size_t linear_base =
                    (static_cast<size_t>(ty * lookup.th + iy) * ew + tx * lookup.tw) * bpe;
                const uint16_t* offsets =
                    lookup.byte_offsets.data() + static_cast<size_t>(iy) * lookup.tw;
                for (uint32_t ix = 0; ix < columns; ++ix) {
                    const size_t tiled = tile_base + offsets[ix];
                    const size_t linear = linear_base + static_cast<size_t>(ix) * bpe;
                    if (ToTiled) {
                        if (tiled + bpe <= tiled_bytes) std::memcpy(dst + tiled, src + linear, bpe);
                    } else if (tiled + bpe <= tiled_bytes) {
                        std::memcpy(dst + linear, src + tiled, bpe);
                    } else {
                        std::memset(dst + linear, 0, bpe);
                    }
                }
            }
        }
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
// SW_64K_S: one 16-bit pattern per element size (identical for every pipe count / RB+ in addrlib).
// (PatBit is declared up top; sw4kb_morton truncates this table for the 4KB order — #379.)
const PatBit kSw64kS[5][16] = {
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

// For the PS5/default 16-pipe GFX10_SW_64K_R_X_1xaa pattern, AddrLib nibble2[74]
// contributes z3,z2,z1,z0 to byte-offset bits 8..11 respectively. The X/Y portions are the
// existing kSw64kRX[4] table above. Keeping this separate makes the previously 2D-only table's
// intentional z==0 projection explicit.
constexpr uint16_t kSw64kbRXVolumeZ[16] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0x0008, 0x0004, 0x0002, 0x0001, 0, 0, 0, 0
};

size_t sw64kb_volume_bytes(uint32_t width, uint32_t height, uint32_t depth, uint32_t bpe) {
    const uint32_t el = sw64kb_elem_log2(bpe);
    if (el == UINT32_MAX) return 0;
    const size_t slice = sw64kb_tiled_bytes(width, height, 0, bpe);
    if (!slice || depth > SIZE_MAX / slice) return 0;
    return slice * depth;
}

template <bool ToTiled>
bool sw64kb_volume_copy(uint8_t* dst, const uint8_t* src, size_t tiled_bytes,
                        uint32_t width, uint32_t height, uint32_t depth, uint32_t bpe) {
    const uint32_t el = sw64kb_elem_log2(bpe);
    if (el == UINT32_MAX || sw64kb_rx_pipes_log2() != 4) return false;
    uint32_t bw = 0, bh = 0;
    sw64kb_dims(el, bw, bh);
    const uint64_t blocks_x = (static_cast<uint64_t>(width) + bw - 1) / bw;
    const uint64_t blocks_y = (static_cast<uint64_t>(height) + bh - 1) / bh;
    const PatBit* pat = kSw64kRX[4][el];
    std::vector<uint16_t> fx(width), fy(height), fz(depth);
    for (uint32_t x = 0; x < width; x++) {
        uint32_t v = 0;
        for (uint32_t i = el; i < 16; i++)
            v |= static_cast<uint32_t>(__builtin_popcount(x & pat[i].x) & 1) << i;
        fx[x] = static_cast<uint16_t>(v);
    }
    for (uint32_t y = 0; y < height; y++) {
        uint32_t v = 0;
        for (uint32_t i = el; i < 16; i++)
            v |= static_cast<uint32_t>(__builtin_popcount(y & pat[i].y) & 1) << i;
        fy[y] = static_cast<uint16_t>(v);
    }
    for (uint32_t z = 0; z < depth; z++) {
        uint32_t v = 0;
        for (uint32_t i = el; i < 16; i++)
            v |= static_cast<uint32_t>(__builtin_popcount(z & kSw64kbRXVolumeZ[i]) & 1) << i;
        fz[z] = static_cast<uint16_t>(v);
    }
    for (uint32_t z = 0; z < depth; z++) {
        const uint64_t slab = static_cast<uint64_t>(z) * blocks_y * blocks_x;
        for (uint32_t y = 0; y < height; y++) {
            const uint64_t row = slab + static_cast<uint64_t>(y / bh) * blocks_x;
            for (uint32_t x = 0; x < width; x++) {
                const uint64_t block = row + x / bw;
                const uint64_t tiled = (block << 16) | static_cast<uint32_t>(fx[x] ^ fy[y] ^ fz[z]);
                const size_t linear =
                    ((static_cast<size_t>(z) * height + y) * width + x) * bpe;
                if (ToTiled) {
                    if (tiled + bpe <= tiled_bytes) std::memcpy(dst + tiled, src + linear, bpe);
                } else if (tiled + bpe <= tiled_bytes) {
                    std::memcpy(dst + linear, src + tiled, bpe);
                } else {
                    std::memset(dst + linear, 0, bpe);
                }
            }
        }
    }
    return true;
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
    if (!tile_mode_is_tiled(tile_mode)) { warn_unhandled_tile_mode(tile_mode, width, height);
                                          std::memcpy(dst, src, (size_t)width * height * bytes_per_texel); return; }
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
        if (bpe != 0) warn_unhandled_tile_mode(tile_mode, ew, eh);   // bpe==0 is a caller error, not a mode gap
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

bool tile_mode_supports_volume(uint32_t tile_mode) {
    return tile_mode == (uint32_t)TileMode::Linear ||
           (tile_mode == (uint32_t)TileMode::Sw64KbRX && sw64kb_rx_pipes_log2() == 4);
}

size_t tiled_volume_bytes(uint32_t width, uint32_t height, uint32_t depth,
                          uint32_t tile_mode, uint32_t bytes_per_texel) {
    if (!width || !height || !depth || !bytes_per_texel) return 0;
    if (tile_mode == (uint32_t)TileMode::Linear) {
        const uint64_t texels = static_cast<uint64_t>(width) * height * depth;
        if (texels > SIZE_MAX / bytes_per_texel) return 0;
        return static_cast<size_t>(texels * bytes_per_texel);
    }
    if (!tile_mode_supports_volume(tile_mode)) return 0;
    return sw64kb_volume_bytes(width, height, depth, bytes_per_texel);
}

bool detile_volume(uint8_t* dst, const uint8_t* src, size_t src_bytes,
                   uint32_t width, uint32_t height, uint32_t depth,
                   uint32_t tile_mode, uint32_t bytes_per_texel) {
    const size_t linear_bytes = static_cast<size_t>(width) * height * depth * bytes_per_texel;
    if (tile_mode == (uint32_t)TileMode::Linear) {
        if (src_bytes < linear_bytes) return false;
        std::memcpy(dst, src, linear_bytes);
        return true;
    }
    if (!tile_mode_supports_volume(tile_mode)) return false;
    return sw64kb_volume_copy<false>(dst, src, src_bytes, width, height, depth, bytes_per_texel);
}

bool tile_volume(uint8_t* dst, size_t dst_bytes, const uint8_t* src,
                 uint32_t width, uint32_t height, uint32_t depth,
                 uint32_t tile_mode, uint32_t bytes_per_texel) {
    const size_t need = tiled_volume_bytes(width, height, depth, tile_mode, bytes_per_texel);
    if (!need || dst_bytes < need) return false;
    if (tile_mode == (uint32_t)TileMode::Linear) {
        std::memcpy(dst, src, need);
        return true;
    }
    std::memset(dst, 0, need);
    return sw64kb_volume_copy<true>(dst, src, need, width, height, depth, bytes_per_texel);
}

} // namespace prosper::gpu
