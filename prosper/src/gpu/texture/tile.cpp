// tile.cpp — see tile.hpp. GFX10 SW_4KB_S de-swizzle, generalized over the element size (#119).
// GFX10 SW_64KB_S / SW_64KB_R_X de-swizzle from the AMD addrlib swizzle-pattern tables (#288).
#include "gpu/texture/tile.hpp"
#include <array>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <limits>
#include <unordered_map>
#include <functional>
#include <mutex>
#include <set>
#include <thread>
#include <system_error>
#include <vector>

#if (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define PROSPER_HAVE_TARGET_AVX2 1
#endif

namespace prosper::gpu {

size_t linear_sampled_row_pitch(uint32_t width, uint32_t bytes_per_texel) {
    if (!width || !bytes_per_texel) return 0;
    const uint64_t tight = static_cast<uint64_t>(width) * bytes_per_texel;
    const uint64_t aligned = (tight + 255u) & ~uint64_t{255u};
    return aligned > std::numeric_limits<size_t>::max() ? 0 : static_cast<size_t>(aligned);
}

size_t linear_sampled_surface_bytes(uint32_t width, uint32_t height, uint32_t bytes_per_texel) {
    const size_t pitch = linear_sampled_row_pitch(width, bytes_per_texel);
    if (!pitch || !height || pitch > std::numeric_limits<size_t>::max() / height) return 0;
    return pitch * height;
}

bool tile_mode_is_tiled(uint32_t tile_mode) {
    return tile_mode == (uint32_t)TileMode::Sw256BS ||
           tile_mode == (uint32_t)TileMode::Sw4KbS ||
           tile_mode == (uint32_t)TileMode::Sw64KbS ||
           tile_mode == (uint32_t)TileMode::Sw64KbZX ||
           tile_mode == (uint32_t)TileMode::Sw64KbRX;
}

uint32_t videoout_scanout_tile_mode(uint32_t videoout_tiling_mode, uint32_t bytes_per_texel) {
    // Only the 32-bpp display formats are established here; anything else stays linear rather than
    // de-swizzling with an unverified block geometry. See the header for the evidence.
    if (videoout_tiling_mode != kVideoOutTilingModeTile || bytes_per_texel != 4)
        return (uint32_t)TileMode::Linear;
    return (uint32_t)TileMode::Sw64KbRX;
}

size_t gfx10_dcc_metadata_bytes(uint32_t width, uint32_t height, uint32_t depth,
                                uint32_t tile_mode, uint32_t bytes_per_texel,
                                bool pipe_aligned) {
    if (tile_mode != (uint32_t)TileMode::Sw64KbRX || !width || !height || !depth)
        return 0;
    uint32_t elem_log2 = 0;
    switch (bytes_per_texel) {
        case 1: elem_log2 = 0; break;
        case 2: elem_log2 = 1; break;
        case 4: elem_log2 = 2; break;
        case 8: elem_log2 = 3; break;
        case 16: elem_log2 = 4; break;
        default: return 0;
    }

    // AddrLib GetMetaBlkSize(Gfx10DataColor): PS5's 16-pipe R_X configuration resolves to a
    // 2^12-byte metadata block for both pipe-aligned and unaligned single-sample surfaces. A color
    // compression block is 2^8 bytes and one DCC metadata element is one byte.
    (void)pipe_aligned;
    constexpr uint64_t meta_block_bytes = 1u << 12;
    const uint32_t meta_bits_log2 = 12u + 8u - elem_log2;
    const uint64_t meta_width = 1ull << ((meta_bits_log2 + 1u) / 2u);
    const uint64_t meta_height = 1ull << (meta_bits_log2 / 2u);
    const uint64_t blocks_w = (static_cast<uint64_t>(width) + meta_width - 1u) / meta_width;
    const uint64_t blocks_h = (static_cast<uint64_t>(height) + meta_height - 1u) / meta_height;
    if (blocks_w > std::numeric_limits<uint64_t>::max() / blocks_h)
        return 0;
    const uint64_t blocks_per_slice = blocks_w * blocks_h;
    if (blocks_per_slice > std::numeric_limits<uint64_t>::max() / depth)
        return 0;
    const uint64_t blocks = blocks_per_slice * depth;
    if (blocks > std::numeric_limits<uint64_t>::max() / meta_block_bytes)
        return 0;
    const uint64_t bytes = blocks * meta_block_bytes;
    return bytes <= std::numeric_limits<size_t>::max() ? static_cast<size_t>(bytes) : 0;
}

bool gfx10_dcc_fast_clear_rgba8(uint8_t* dst, size_t texel_count,
                                const uint8_t* metadata, size_t metadata_bytes,
                                uint32_t num_components, bool alpha_is_on_msb,
                                uint8_t* clear_code) {
    if (!metadata || !metadata_bytes ||
        (num_components != 3 && num_components != 4) || (!dst && texel_count))
        return false;
    const uint8_t code = metadata[0];
    if (code != 0x00 && code != 0x40 && code != 0x80 && code != 0xc0)
        return false;
    if (!std::all_of(metadata + 1, metadata + metadata_bytes,
                     [=](uint8_t value) { return value == code; }))
        return false;

    const uint8_t color = (code == 0x80 || code == 0xc0) ? 255 : 0;
    uint8_t pixel[4] = {color, color, color, 255};
    if (num_components == 4) {
        const uint8_t alpha = (code == 0x40 || code == 0xc0) ? 255 : 0;
        const uint32_t alpha_component = alpha_is_on_msb ? 3u : 0u;
        std::fill(pixel, pixel + 4, color);
        pixel[alpha_component] = alpha;
    }
    for (size_t i = 0; i < texel_count; ++i)
        std::memcpy(dst + i * 4, pixel, sizeof(pixel));
    if (clear_code) *clear_code = code;
    return true;
}

// One-time diagnostic when a NON-ZERO (tiled) tile_mode is not one of the swizzles we de-swizzle
// (Sw256BS=1 / Sw4KbS=5 / Sw64KbS=9 / Sw64KbZX=24 / Sw64KbRX=27): the caller then
// copies the surface VERBATIM as if linear, so it samples as a scrambled swizzle-weave. Other GFX10
// modes a PS5 T# can legally carry — the remaining SW_256B_* variants, SW_4KB_D/*_X,
// SW_64KB_S_T/D_T, the SW_64KB_Z/D depth/displayable families, SW_VAR_* — all land here.
// Log once per unrecognized mode under PROSPER_GFXLOG so this silent linear-fallback is observable
// instead of masquerading as a correct linear copy (#383). No-op for mode 0 (genuinely linear).
static void warn_unhandled_tile_mode(uint32_t tile_mode, uint32_t w, uint32_t h) {
    if (tile_mode == 0 || tile_mode_is_tiled(tile_mode) || !getenv("PROSPER_GFXLOG")) return;
    static std::set<uint32_t> seen;
    static std::mutex mu;
    std::lock_guard<std::mutex> lk(mu);
    if (seen.insert(tile_mode).second)
        fprintf(stderr, "[tile] UNHANDLED GFX10 tile_mode=%u (%ux%u) -> copied as LINEAR; surface will "
                        "sample SCRAMBLED (only Sw256BS=1/Sw4KbS=5/Sw64KbS=9/Sw64KbZX=24/Sw64KbRX=27 are de-swizzled)\n",
                        tile_mode, w, h);
}

namespace {
// One addrlib swizzle-pattern entry: the (x-mask, y-mask) coordinate bits an offset bit is built from.
// Defined here (and the SW_64K_S table forward-declared) so sw4kb_morton below can derive the 4KB order
// from the low bits of the SAME authoritative 64KB pattern (#379); both are defined together lower down.
struct PatBit { uint16_t x, y; };
struct PatBitMsaa { uint16_t x, y; uint8_t sample; };
extern const PatBit kSw64kS[5][16];

// GFX10 SW_256B_S is the micro-tiled standard swizzle used for small textures. AMD AddrLib's
// GFX10_SW_256_S pattern is exactly the first eight byte-address bits of the corresponding
// SW_64KB_S pattern below. A 256-byte block therefore has these element dimensions:
//   1 B -> 16x16, 2 B -> 16x8, 4 B -> 8x8, 8 B -> 8x4, 16 B -> 4x4.
// Blocks are row-major over a block-aligned pitch. The 16-byte case is especially important for
// BC2/3/5/6/7: its within-block order is y0,y1,x0,x1, not linear x0,x1,y0,y1.
inline void sw256_dims(uint32_t bpe, uint32_t& bx, uint32_t& by) {
    const uint32_t bits = 8u - static_cast<uint32_t>(__builtin_ctz(bpe));
    bx = (bits + 1u) / 2u;
    by = bits / 2u;
}

struct Sw256Lookup {
    uint32_t tw = 0, th = 0;
    std::vector<uint8_t> byte_offsets;
};

Sw256Lookup make_sw256_lookup(uint32_t bpe) {
    Sw256Lookup lookup;
    uint32_t bx = 0, by = 0;
    sw256_dims(bpe, bx, by);
    lookup.tw = 1u << bx;
    lookup.th = 1u << by;
    lookup.byte_offsets.resize(static_cast<size_t>(lookup.tw) * lookup.th);
    const uint32_t elem_log2 = static_cast<uint32_t>(__builtin_ctz(bpe));
    for (uint32_t y = 0; y < lookup.th; ++y) {
        for (uint32_t x = 0; x < lookup.tw; ++x) {
            uint32_t byte_offset = 0;
            for (uint32_t bit = elem_log2; bit < 8u; ++bit) {
                const PatBit& pattern = kSw64kS[elem_log2][bit];
                const uint32_t value = pattern.x
                    ? static_cast<uint32_t>(__builtin_popcount(x & pattern.x) & 1)
                    : static_cast<uint32_t>(__builtin_popcount(y & pattern.y) & 1);
                byte_offset |= value << bit;
            }
            lookup.byte_offsets[static_cast<size_t>(y) * lookup.tw + x] =
                static_cast<uint8_t>(byte_offset);
        }
    }
    return lookup;
}

const Sw256Lookup& sw256_lookup(uint32_t bpe) {
    static const Sw256Lookup b1 = make_sw256_lookup(1);
    static const Sw256Lookup b2 = make_sw256_lookup(2);
    static const Sw256Lookup b4 = make_sw256_lookup(4);
    static const Sw256Lookup b8 = make_sw256_lookup(8);
    static const Sw256Lookup b16 = make_sw256_lookup(16);
    switch (bpe) {
        case 1: return b1;
        case 2: return b2;
        case 4: return b4;
        case 8: return b8;
        case 16: return b16;
        default: return b1;
    }
}

template <bool ToTiled>
void sw256_copy(uint8_t* dst, const uint8_t* src, uint32_t ew, uint32_t eh, uint32_t pitch,
                uint32_t bpe, size_t tiled_bytes, size_t tiled_origin = 0) {
    const Sw256Lookup& lookup = sw256_lookup(bpe);
    const uint32_t padded_width = pitch ? pitch : ew;
    const uint32_t blocks_per_row = (padded_width + lookup.tw - 1u) / lookup.tw;
    const uint32_t block_rows = (eh + lookup.th - 1u) / lookup.th;
    const uint32_t block_cols = (ew + lookup.tw - 1u) / lookup.tw;
    for (uint32_t by = 0; by < block_rows; ++by) {
        const uint32_t rows = std::min(lookup.th, eh - by * lookup.th);
        for (uint32_t bx = 0; bx < block_cols; ++bx) {
            const uint32_t columns = std::min(lookup.tw, ew - bx * lookup.tw);
            const size_t block_base = static_cast<size_t>(by * blocks_per_row + bx) * 256u;
            for (uint32_t y = 0; y < rows; ++y) {
                const size_t linear_base =
                    (static_cast<size_t>(by * lookup.th + y) * ew + bx * lookup.tw) * bpe;
                const uint8_t* offsets =
                    lookup.byte_offsets.data() + static_cast<size_t>(y) * lookup.tw;
                for (uint32_t x = 0; x < columns; ++x) {
                    const size_t tiled = tiled_origin + block_base + offsets[x];
                    const size_t linear = linear_base + static_cast<size_t>(x) * bpe;
                    if (ToTiled) {
                        if (tiled + bpe <= tiled_bytes)
                            std::memcpy(dst + tiled, src + linear, bpe);
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

size_t sw256_tiled_bytes(uint32_t ew, uint32_t eh, uint32_t pitch, uint32_t bpe) {
    uint32_t bx = 0, by = 0;
    sw256_dims(bpe, bx, by);
    const uint32_t padded_width = pitch ? pitch : ew;
    const uint32_t blocks_per_row = (padded_width + (1u << bx) - 1u) >> bx;
    const uint32_t block_rows = (eh + (1u << by) - 1u) >> by;
    return static_cast<size_t>(blocks_per_row) * block_rows * 256u;
}

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
                uint32_t bpe, size_t tiled_bytes, size_t tiled_origin = 0) {
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
                    const size_t tiled = tiled_origin + tile_base + offsets[ix];
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

// SW_64K_Z_X, 1 fragment, 16 pipes. Expanded directly from AMD AddrLib's
// GFX10_SW_64K_Z_X_1xaa_PATINFO rows {12,9,10,11,7}/{74}/{1,30,31,32,33}, with z=0
// for a 2D surface. Mode 24 first appeared live as Astro Bot's R32 compute destination (#825).
// Like R_X, bits 8..11 rotate the 64KB block across pipes; unlike R_X, the low byte follows
// Z/Morton ordering. Oberon uses the same fixed 16-pipe selection as the existing mode-27 path.
static const PatBit kSw64kZX[5][16] = {
    { {0x0001,0x0000}, {0x0000,0x0001}, {0x0002,0x0000}, {0x0000,0x0002}, {0x0004,0x0000}, {0x0000,0x0004}, {0x0008,0x0000}, {0x0000,0x0010}, {0x0008,0x0008}, {0x0010,0x0010}, {0x0040,0x0020}, {0x0020,0x0040}, {0x0000,0x0040}, {0x0040,0x0000}, {0x0000,0x0080}, {0x0080,0x0000} },  // 1 bpe
    { {0x0000,0x0000}, {0x0001,0x0000}, {0x0000,0x0001}, {0x0002,0x0000}, {0x0000,0x0002}, {0x0004,0x0000}, {0x0000,0x0004}, {0x0008,0x0000}, {0x0008,0x0008}, {0x0010,0x0010}, {0x0040,0x0020}, {0x0020,0x0040}, {0x0000,0x0010}, {0x0040,0x0000}, {0x0000,0x0040}, {0x0080,0x0000} },  // 2 bpe
    { {0x0000,0x0000}, {0x0000,0x0000}, {0x0001,0x0000}, {0x0000,0x0001}, {0x0002,0x0000}, {0x0000,0x0002}, {0x0004,0x0000}, {0x0000,0x0004}, {0x0008,0x0008}, {0x0010,0x0010}, {0x0040,0x0020}, {0x0020,0x0040}, {0x0000,0x0008}, {0x0010,0x0000}, {0x0000,0x0040}, {0x0040,0x0000} },  // 4 bpe
    { {0x0000,0x0000}, {0x0000,0x0000}, {0x0000,0x0000}, {0x0001,0x0000}, {0x0000,0x0001}, {0x0002,0x0000}, {0x0000,0x0002}, {0x0004,0x0000}, {0x0008,0x0008}, {0x0010,0x0010}, {0x0040,0x0020}, {0x0020,0x0040}, {0x0000,0x0004}, {0x0008,0x0000}, {0x0000,0x0010}, {0x0040,0x0000} },  // 8 bpe
    { {0x0000,0x0000}, {0x0000,0x0000}, {0x0000,0x0000}, {0x0000,0x0000}, {0x0001,0x0000}, {0x0000,0x0001}, {0x0002,0x0000}, {0x0000,0x0002}, {0x0008,0x0008}, {0x0010,0x0010}, {0x0040,0x0020}, {0x0020,0x0040}, {0x0000,0x0004}, {0x0004,0x0000}, {0x0000,0x0008}, {0x0010,0x0000} },  // 16 bpe
};

// SW_64K_Z_X, 4 fragments, 16 pipes. Expanded from AMD AddrLib's
// GFX10_SW_64K_Z_X_4xaa_PATINFO rows for the five supported element sizes. Unlike the 1xaa table,
// S0/S1 are real address inputs: dropping them aliases all four samples. Z/slice remains zero for a
// non-array 2D_MSAA resource. The pipe-count choice matches the established Oberon mode-24 path.
// Source: AMD ROCr AddrLib gfx10SwizzlePattern.h (MIT), PATINFO rows {18..22, 74/95/96, 31/74..76}.
static const PatBitMsaa kSw64kZX4x[5][16] = {
    { {0,0,1}, {0,0,2}, {0x0001,0,0}, {0,0x0001,0}, {0x0002,0,0}, {0,0x0002,0}, {0x0004,0,0}, {0,0x0004,0},
      {0x0008,0x0008,0}, {0x0010,0x0010,0}, {0x0040,0x0020,0}, {0x0020,0x0040,0},
      {0,0x0008,0}, {0x0010,0,0}, {0,0x0040,0}, {0x0040,0,0} }, // 1 B
    { {0,0,0}, {0,0,1}, {0,0,2}, {0x0001,0,0}, {0,0x0001,0}, {0x0002,0,0}, {0,0x0002,0}, {0x0004,0,0},
      {0x0008,0x0008,0}, {0x0010,0x0010,0}, {0x0040,0x0020,0}, {0x0020,0x0040,0},
      {0x0008,0,0}, {0,0x0010,0}, {0x0040,0,0}, {0,0x0004 | 0x0040,0} }, // 2 B
    { {0,0,0}, {0,0,0}, {0,0,1}, {0,0,2}, {0x0001,0,0}, {0,0x0001,0}, {0x0002,0,0}, {0,0x0002,0},
      {0x0008,0x0008,0}, {0x0010,0x0010,0}, {0x0040,0x0020,0}, {0x0020,0x0040,0},
      {0,0x0008,0}, {0x0010,0,0}, {0,0x0004 | 0x0040,0}, {0x0004,0x0080,0} }, // 4 B
    { {0,0,0}, {0,0,0}, {0,0,0}, {0,0,1}, {0,0,2}, {0x0001,0,0}, {0,0x0001,0}, {0x0002,0,0},
      {0x0008,0x0008,0}, {0x0010,0x0010,0}, {0x0040,0x0020 | 0x0004,0}, {0x0020,0x0040,0},
      {0,0x0008,0}, {0x0010,0,0}, {0,0x0002 | 0x0040,0}, {0x0004,0x0080,0} }, // 8 B
    { {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}, {0,0,1}, {0,0,2}, {0x0001,0,0}, {0,0x0001,0},
      {0x0008,0x0008,0}, {0x0010,0x0010,0}, {0x0040,0x0020 | 0x0004,0}, {0x0022,0x0040,0},
      {0,0x0008,0}, {0x0010,0,0}, {0,0x0002 | 0x0040,0}, {0x0004,0x0080,0} }, // 16 B
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
    if (tile_mode == (uint32_t)TileMode::Sw64KbZX) return kSw64kZX[elem_log2];
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

// Row-parallelism for the tiled<->linear walk (#1177). The walk is row-independent: for detile each
// output row writes a disjoint linear span and only reads (read-only) tiled bytes; for tile the layout
// is a bijection so every (y,x) writes a distinct tiled offset. So splitting the row range across
// threads is race-free. Uses per-call std::thread (not a pool): this gates on large surfaces where the
// spawn cost is amortized by the 4K detile work, and it avoids any pool-reuse synchronization hazard.
// Small surfaces and PROSPER_DETILE_SINGLE_THREADED / PROSPER_DETILE_THREADS=1 stay single-threaded;
// detiling is memory-bandwidth-bound, so the auto thread count is capped at 16. Astro Bot's 4K
// surfaces improve from 8 to 16 workers on a 16-core Strix Halo, while 32 regresses.
inline unsigned detile_row_threads(size_t work_bytes, uint32_t eh) {
    static const int env = [] {
        const char* s = getenv("PROSPER_DETILE_THREADS");
        return s ? atoi(s) : -1;   // -1 = auto
    }();
    static const bool single = getenv("PROSPER_DETILE_SINGLE_THREADED") != nullptr;
    if (single || env == 1) return 1u;
    if (work_bytes < (size_t)512 * 1024) return 1u;            // small surface: spawn not worth it
    const unsigned hw = std::thread::hardware_concurrency();
    const unsigned want = env > 1 ? std::min((unsigned)env, 32u)   // clamp an over-large override
                                  : std::min(hw ? hw : 4u, 16u);   // memory-bound -> cap at 16
    const unsigned by_rows = eh / 32u;                             // keep >= 32 rows per thread
    return std::max(1u, std::min(want, by_rows));
}

// body(row_begin, row_end) must not throw: it runs on worker threads that are only join()ed, so an
// escaping exception would std::terminate. The only caller is the memcpy/memset detile loop (noexcept).
template <class Body>
inline void parallel_rows(uint32_t eh, unsigned nthreads, Body&& body) {
    if (nthreads <= 1 || eh == 0) { if (eh) body(0u, eh); return; }
    const uint32_t chunk = (eh + nthreads - 1) / nthreads;
    // Auto-joining workers keep a partially-created set safe when the OS refuses another thread.
    // Any ranges that could not get a worker are completed synchronously below.
    std::vector<std::jthread> workers;
    workers.reserve(nthreads - 1);
    unsigned next_worker = 1;
    try {
        for (; next_worker < nthreads; ++next_worker) {
            const uint32_t begin = next_worker * chunk;
            const uint32_t end = std::min(eh, begin + chunk);
            if (begin >= end) break;
            workers.emplace_back([&body, begin, end] { body(begin, end); });
        }
    } catch (const std::system_error&) {
        // Fall through: ranges that did not get a worker run synchronously below.
    }
    body(0u, std::min(eh, chunk));
    for (; next_worker < nthreads; ++next_worker) {
        const uint32_t begin = next_worker * chunk;
        const uint32_t end = std::min(eh, begin + chunk);
        if (begin >= end) break;
        body(begin, end);
    }
}

inline void copy_swizzled_element(uint8_t* dst, const uint8_t* src, uint32_t bpe) {
    switch (bpe) {
        case 1: *dst = *src; break;
        case 2: std::memcpy(dst, src, 2); break;
        case 4: std::memcpy(dst, src, 4); break;
        case 8: std::memcpy(dst, src, 8); break;
        case 16: std::memcpy(dst, src, 16); break;
        default: std::memcpy(dst, src, bpe); break;
    }
}

inline void zero_swizzled_element(uint8_t* dst, uint32_t bpe) {
    switch (bpe) {
        case 1: *dst = 0; break;
        case 2: std::memset(dst, 0, 2); break;
        case 4: std::memset(dst, 0, 4); break;
        case 8: std::memset(dst, 0, 8); break;
        case 16: std::memset(dst, 0, 16); break;
        default: std::memset(dst, 0, bpe); break;
    }
}

#if defined(PROSPER_HAVE_TARGET_AVX2)
// A complete 64KB block has no per-element bounds failures. For the dominant element sizes, gather
// precomputed swizzle offsets and write one contiguous linear span at a time. The low x bit maps to
// the first address bit above elemLog2 in every supported pattern, so aligned 2/4-byte texel pairs
// are adjacent in tiled memory and can share one 4/8-byte gather lane. Astro Bot moves about 3.7 GiB
// of 8-byte, 2.5 GiB of 4-byte, and 580 MiB of 2-byte elements through this path in 120 frames.
__attribute__((target("avx2")))
void detile_full_block_row_avx2(uint8_t* dst, const uint8_t* tiled_block,
                                const uint16_t* x_offsets, uint16_t y_offset,
                                uint32_t columns, uint32_t bpe) {
    uint32_t x = 0;
    const __m128i y = _mm_set1_epi16(static_cast<int16_t>(y_offset));
    static const bool paired_gathers =
        std::getenv("PROSPER_NO_PAIRED_AVX2_DETILE") == nullptr;
    const __m128i even_offsets = _mm_setr_epi8(
        0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1);
    if (bpe == 2 && paired_gathers) {
        for (; x + 8 <= columns; x += 8) {
            const __m128i offsets16 = _mm_xor_si128(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(x_offsets + x)), y);
            const __m128i pair_offsets16 = _mm_shuffle_epi8(offsets16, even_offsets);
            const __m128i pair_offsets32 = _mm_cvtepu16_epi32(pair_offsets16);
            const __m128i values = _mm_i32gather_epi32(
                reinterpret_cast<const int*>(tiled_block), pair_offsets32, 1);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + static_cast<size_t>(x) * 2),
                             values);
        }
    } else if (bpe == 4 && paired_gathers) {
        for (; x + 8 <= columns; x += 8) {
            const __m128i offsets16 = _mm_xor_si128(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(x_offsets + x)), y);
            const __m128i pair_offsets16 = _mm_shuffle_epi8(offsets16, even_offsets);
            const __m256i pair_offsets64 = _mm256_cvtepu16_epi64(pair_offsets16);
            const __m256i values = _mm256_i64gather_epi64(
                reinterpret_cast<const long long*>(tiled_block), pair_offsets64, 1);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + static_cast<size_t>(x) * 4),
                                values);
        }
    } else if (bpe == 4) {
        for (; x + 8 <= columns; x += 8) {
            const __m128i offsets16 = _mm_xor_si128(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(x_offsets + x)), y);
            const __m256i offsets32 = _mm256_cvtepu16_epi32(offsets16);
            const __m256i values = _mm256_i32gather_epi32(
                reinterpret_cast<const int*>(tiled_block), offsets32, 1);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + static_cast<size_t>(x) * 4),
                                values);
        }
    } else if (bpe == 8) {
        for (; x + 4 <= columns; x += 4) {
            const __m128i offsets16 = _mm_xor_si128(
                _mm_loadl_epi64(reinterpret_cast<const __m128i*>(x_offsets + x)), y);
            const __m256i offsets64 = _mm256_cvtepu16_epi64(offsets16);
            const __m256i values = _mm256_i64gather_epi64(
                reinterpret_cast<const long long*>(tiled_block), offsets64, 1);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + static_cast<size_t>(x) * 8),
                                values);
        }
    }
    for (; x < columns; ++x)
        copy_swizzled_element(dst + static_cast<size_t>(x) * bpe,
                              tiled_block + (x_offsets[x] ^ y_offset), bpe);
}

__attribute__((target("avx2")))
void tile_full_block_row_avx2(uint8_t* tiled_block, const uint8_t* linear_src,
                              const uint16_t* x_offsets, uint16_t y_offset,
                              uint32_t columns, uint32_t bpe) {
    uint32_t x = 0;
    const __m128i y = _mm_set1_epi16(static_cast<int16_t>(y_offset));
    static const bool paired_tiles =
        std::getenv("PROSPER_NO_PAIRED_AVX2_TILE") == nullptr;
    const __m128i even_offsets = _mm_setr_epi8(
        0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1);
    if (bpe == 2 && paired_tiles) {
        for (; x + 8 <= columns; x += 8) {
            const __m128i offsets16 = _mm_xor_si128(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(x_offsets + x)), y);
            const __m128i pair_offsets16 = _mm_shuffle_epi8(offsets16, even_offsets);
            const uint16_t off0 = static_cast<uint16_t>(_mm_extract_epi16(pair_offsets16, 0));
            const uint16_t off1 = static_cast<uint16_t>(_mm_extract_epi16(pair_offsets16, 1));
            const uint16_t off2 = static_cast<uint16_t>(_mm_extract_epi16(pair_offsets16, 2));
            const uint16_t off3 = static_cast<uint16_t>(_mm_extract_epi16(pair_offsets16, 3));
            const uint32_t* src32 =
                reinterpret_cast<const uint32_t*>(linear_src + static_cast<size_t>(x) * 2);
            *reinterpret_cast<uint32_t*>(tiled_block + off0) = src32[0];
            *reinterpret_cast<uint32_t*>(tiled_block + off1) = src32[1];
            *reinterpret_cast<uint32_t*>(tiled_block + off2) = src32[2];
            *reinterpret_cast<uint32_t*>(tiled_block + off3) = src32[3];
        }
    } else if (bpe == 4 && paired_tiles) {
        for (; x + 8 <= columns; x += 8) {
            const __m128i offsets16 = _mm_xor_si128(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(x_offsets + x)), y);
            const __m128i pair_offsets16 = _mm_shuffle_epi8(offsets16, even_offsets);
            const uint16_t off0 = static_cast<uint16_t>(_mm_extract_epi16(pair_offsets16, 0));
            const uint16_t off1 = static_cast<uint16_t>(_mm_extract_epi16(pair_offsets16, 1));
            const uint16_t off2 = static_cast<uint16_t>(_mm_extract_epi16(pair_offsets16, 2));
            const uint16_t off3 = static_cast<uint16_t>(_mm_extract_epi16(pair_offsets16, 3));
            const uint64_t* src64 =
                reinterpret_cast<const uint64_t*>(linear_src + static_cast<size_t>(x) * 4);
            *reinterpret_cast<uint64_t*>(tiled_block + off0) = src64[0];
            *reinterpret_cast<uint64_t*>(tiled_block + off1) = src64[1];
            *reinterpret_cast<uint64_t*>(tiled_block + off2) = src64[2];
            *reinterpret_cast<uint64_t*>(tiled_block + off3) = src64[3];
        }
    } else if (bpe == 8) {
        for (; x + 4 <= columns; x += 4) {
            const __m128i offsets16 = _mm_xor_si128(
                _mm_loadl_epi64(reinterpret_cast<const __m128i*>(x_offsets + x)), y);
            const uint16_t off0 = static_cast<uint16_t>(_mm_extract_epi16(offsets16, 0));
            const uint16_t off1 = static_cast<uint16_t>(_mm_extract_epi16(offsets16, 2));
            const __m128i val0 = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(linear_src + static_cast<size_t>(x) * 8));
            const __m128i val1 = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(linear_src + static_cast<size_t>(x) * 8 + 16));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(tiled_block + off0), val0);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(tiled_block + off1), val1);
        }
    } else if (bpe == 16) {
        for (; x + 2 <= columns; x += 2) {
            const __m128i offsets16 = _mm_xor_si128(
                _mm_cvtsi32_si128(*reinterpret_cast<const int32_t*>(x_offsets + x)), y);
            const uint16_t off0 = static_cast<uint16_t>(_mm_extract_epi16(offsets16, 0));
            const uint16_t off1 = static_cast<uint16_t>(_mm_extract_epi16(offsets16, 1));
            const __m128i val0 = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(linear_src + static_cast<size_t>(x) * 16));
            const __m128i val1 = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(linear_src + static_cast<size_t>(x) * 16 + 16));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(tiled_block + off0), val0);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(tiled_block + off1), val1);
        }
    }
    for (; x < columns; ++x)
        copy_swizzled_element(tiled_block + (x_offsets[x] ^ y_offset),
                              linear_src + static_cast<size_t>(x) * bpe, bpe);
}

#endif

// The 64KB tiled<->linear walk. The pattern offset is XOR-separable in x and y (each offset bit is
// parity(x&xm)^parity(y&ym)), so precompute fx[] / fy[] per coordinate and each element's in-block
// offset is fx[x]^fy[y] — exact per addrlib (patterns are evaluated on GLOBAL element coords; every
// mask stays within the coordinate range of one block for the pipe counts shipped here, but global
// coords keep even wider masks correct).
template <bool ToTiled>
void sw64kb_copy(uint8_t* dst, const uint8_t* src, uint32_t ew, uint32_t eh, uint32_t pitch,
                 uint32_t bpe, size_t tiled_bytes, uint32_t tile_mode,
                 size_t tiled_origin = 0, bool allow_avx2 = true) {
    uint32_t el = sw64kb_elem_log2(bpe);
    if (el == UINT32_MAX) {
        const size_t n = (size_t)ew * eh * bpe;
        std::memcpy(dst, src, std::min(n, tiled_bytes));
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
    // Detiling in linear row order revisits every 64KB source block once per output row. A 4K
    // RGBA16F surface has thirty-four block columns, so that walk cycles through more than 2 MiB
    // before returning to the next few bytes of any block. Finish one 64KB block at a time instead:
    // source reads then remain cache-local while each output row still receives a contiguous span.
    // Split on whole block rows so workers write disjoint output rows and cannot false-share. The
    // same order also lowers the CPU cost of guest writeback; either direction can be restored to
    // the older row-major walk independently for diagnosis.
    static const bool row_major_detile = std::getenv("PROSPER_DETILE_ROW_MAJOR") != nullptr;
    static const bool row_major_tile = std::getenv("PROSPER_TILE_ROW_MAJOR") != nullptr;
    if ((!ToTiled && !row_major_detile) || (ToTiled && !row_major_tile)) {
#if defined(PROSPER_HAVE_TARGET_AVX2)
        static const bool cpu_has_avx2 = __builtin_cpu_supports("avx2");
        const bool use_avx2_gather = allow_avx2 && cpu_has_avx2 &&
            std::getenv("PROSPER_NO_AVX2_DETILE") == nullptr;
        const bool use_avx2_tile = allow_avx2 && cpu_has_avx2 &&
            std::getenv("PROSPER_NO_AVX2_TILE") == nullptr;
#endif
            const uint32_t surface_block_rows = (eh + bh - 1) / bh;
            const uint32_t surface_block_cols = (ew + bw - 1) / bw;
            static const bool paired_copy =
                std::getenv("PROSPER_NO_PAIRED_TILE_COPY") == nullptr;
            auto process_block_rows = [&](uint32_t block_row0, uint32_t block_row1) {
                for (uint32_t block_y = block_row0; block_y < block_row1; ++block_y) {
                    const uint32_t y0 = block_y * bh;
                    const uint32_t rows = std::min(bh, eh - y0);
                    for (uint32_t block_x = 0; block_x < surface_block_cols; ++block_x) {
                        const uint32_t x0 = block_x * bw;
                        const uint32_t columns = std::min(bw, ew - x0);
                        const uint64_t block =
                            static_cast<uint64_t>(block_y) * blocks_per_row + block_x;
                        const uint64_t block_base = tiled_origin + (block << 16);
                        const bool full_block = block_base <= tiled_bytes &&
                            tiled_bytes - static_cast<size_t>(block_base) >= 65536u;
                        for (uint32_t iy = 0; iy < rows; ++iy) {
                            const uint32_t y = y0 + iy;
                            const uint16_t y_offset = fy[y];
                            uint8_t* linear_dst = nullptr;
                            const uint8_t* linear_src = nullptr;
                            if constexpr (!ToTiled)
                                linear_dst = dst + (static_cast<size_t>(y) * ew + x0) * bpe;
                            else
                                linear_src = src + (static_cast<size_t>(y) * ew + x0) * bpe;
#if defined(PROSPER_HAVE_TARGET_AVX2)
                            if constexpr (!ToTiled) {
                                if (full_block && (bpe == 2 || bpe == 4 || bpe == 8)) {
                                    if (use_avx2_gather) {
                                        detile_full_block_row_avx2(linear_dst, src + block_base,
                                                                  fx.data() + x0, y_offset,
                                                                  columns, bpe);
                                        continue;
                                    }
                                }
                            } else {
                                if (full_block && (bpe == 2 || bpe == 4 || bpe == 8 || bpe == 16)) {
                                    if (use_avx2_tile) {
                                        tile_full_block_row_avx2(dst + block_base, linear_src,
                                                                fx.data() + x0, y_offset,
                                                                columns, bpe);
                                        continue;
                                    }
                                }
                            }
#endif
                            for (uint32_t ix = 0; ix < columns;) {
                                const uint64_t tiled = block_base + (fx[x0 + ix] ^ y_offset);
                                // Up through 8-byte elements, x0 is the first swizzle bit above the
                                // byte-within-element bits. Since block x origins and ix are even,
                                // the next texel immediately follows this one in both layouts.
                                uint32_t run = paired_copy && bpe <= 8 && ix + 1 < columns
                                    ? 2u : 1u;
                                uint32_t bytes = run * bpe;
                                // A bounded source can end between otherwise-adjacent texels. Keep
                                // the first valid element instead of rejecting/zeroing the whole pair;
                                // the next loop iteration handles the truncated neighbor separately.
                                if (!full_block && run == 2 &&
                                    (tiled > tiled_bytes || bytes > tiled_bytes - tiled)) {
                                    run = 1;
                                    bytes = bpe;
                                }
                                if constexpr (ToTiled) {
                                    if (full_block ||
                                        (tiled <= tiled_bytes && bytes <= tiled_bytes - tiled))
                                        copy_swizzled_element(dst + tiled,
                                                              linear_src + static_cast<size_t>(ix) * bpe,
                                                              bytes);
                                } else {
                                    if (full_block ||
                                        (tiled <= tiled_bytes && bytes <= tiled_bytes - tiled))
                                        copy_swizzled_element(
                                            linear_dst + static_cast<size_t>(ix) * bpe,
                                            src + tiled, bytes);
                                    else
                                        zero_swizzled_element(
                                            linear_dst + static_cast<size_t>(ix) * bpe, bytes);
                                }
                                ix += run;
                            }
                        }
                    }
                }
            };
            const unsigned threads = std::min(
                detile_row_threads((size_t)ew * eh * bpe, eh), surface_block_rows);
            parallel_rows(surface_block_rows, threads, process_block_rows);
            return;
    }
    auto process_rows = [&](uint32_t y0, uint32_t y1) {
        for (uint32_t y = y0; y < y1; y++) {
            uint64_t brow = (uint64_t)(y / bh) * blocks_per_row;
            for (uint32_t x = 0; x < ew; x++) {
                uint64_t t = tiled_origin +
                             (((brow + x / bw) << 16) | (uint32_t)(fx[x] ^ fy[y]));
                size_t   l = ((size_t)y * ew + x) * bpe;
                if (ToTiled) {
                    if (t + bpe <= tiled_bytes)
                        copy_swizzled_element(dst + t, src + l, bpe);
                } else {
                    if (t + bpe <= tiled_bytes)
                        copy_swizzled_element(dst + l, src + t, bpe);
                    else
                        zero_swizzled_element(dst + l, bpe);
                }
            }
        }
    };
    parallel_rows(eh, detile_row_threads((size_t)ew * eh * bpe, eh), process_rows);
}

inline bool is_64kb_mode(uint32_t tile_mode) {
    return tile_mode == (uint32_t)TileMode::Sw64KbS ||
           tile_mode == (uint32_t)TileMode::Sw64KbZX ||
           tile_mode == (uint32_t)TileMode::Sw64KbRX;
}

// A packed tail level is addressed by GLOBAL element coordinates inside one shared macroblock.
// Its AddrLib mipTailOffset is metadata, not a linear pointer delta: adding it to the ordinary
// within-level swizzle produces a self-consistent but physically wrong layout. These walks apply
// mipTailCoordX/Y before evaluating the same authoritative pattern as the full-surface paths.
template <bool ToTiled>
void sw4kb_level_copy(uint8_t* dst, const uint8_t* src, size_t tiled_bytes,
                      uint32_t ew, uint32_t eh, uint32_t bpe,
                      uint32_t tail_x, uint32_t tail_y) {
    const Sw4kbLookup& lookup = sw4kb_lookup(bpe);
    for (uint32_t y = 0; y < eh; ++y) {
        const uint32_t gy = tail_y + y;
        for (uint32_t x = 0; x < ew; ++x) {
            const uint32_t gx = tail_x + x;
            // Thin GFX10 tails occupy exactly one block. Coordinates outside it indicate a bad
            // layout descriptor; the bounds check below turns those accesses into zero/no-op.
            const size_t block = static_cast<size_t>(gy / lookup.th) + gx / lookup.tw;
            const size_t tiled = block * 4096u +
                lookup.byte_offsets[static_cast<size_t>(gy % lookup.th) * lookup.tw +
                                    (gx % lookup.tw)];
            const size_t linear = (static_cast<size_t>(y) * ew + x) * bpe;
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

template <bool ToTiled>
void sw64kb_level_copy(uint8_t* dst, const uint8_t* src, size_t tiled_bytes,
                       uint32_t ew, uint32_t eh, uint32_t bpe, uint32_t tile_mode,
                       uint32_t tail_x, uint32_t tail_y) {
    const uint32_t el = sw64kb_elem_log2(bpe);
    if (el == UINT32_MAX) return;
    uint32_t bw = 0, bh = 0;
    sw64kb_dims(el, bw, bh);
    const PatBit* pat = sw64kb_pattern(tile_mode, el);
    for (uint32_t y = 0; y < eh; ++y) {
        const uint32_t gy = tail_y + y;
        for (uint32_t x = 0; x < ew; ++x) {
            const uint32_t gx = tail_x + x;
            uint32_t within = 0;
            for (uint32_t i = el; i < 16; ++i) {
                const uint32_t bit = (__builtin_popcount(gx & pat[i].x) ^
                                      __builtin_popcount(gy & pat[i].y)) & 1u;
                within |= bit << i;
            }
            const size_t block = static_cast<size_t>(gy / bh) + gx / bw;
            const size_t tiled = block * 65536u + within;
            const size_t linear = (static_cast<size_t>(y) * ew + x) * bpe;
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

// For the PS5/default 16-pipe GFX10_SW_64K_R_X_1xaa pattern, AddrLib nibble2[74]
// contributes z3,z2,z1,z0 to byte-offset bits 8..11 respectively. The X/Y portions are the
// existing kSw64kRX[4] table above. Keeping this separate makes the previously 2D-only table's
// intentional z==0 projection explicit.
constexpr uint16_t kSw64kbRXVolumeZ[16] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0x0008, 0x0004, 0x0002, 0x0001, 0, 0, 0, 0
};

size_t sw64kb_rx_volume_bytes(uint32_t width, uint32_t height, uint32_t depth, uint32_t bpe) {
    const uint32_t el = sw64kb_elem_log2(bpe);
    if (el == UINT32_MAX) return 0;
    const size_t slice = sw64kb_tiled_bytes(width, height, 0, bpe);
    if (!slice || depth > SIZE_MAX / slice) return 0;
    return slice * depth;
}

template <bool ToTiled>
bool sw64kb_rx_volume_copy(uint8_t* dst, const uint8_t* src, size_t tiled_bytes,
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

struct PatBit3 {
    uint16_t x, y, z;
};

// AMD AddrLib's authoritative GFX10_SW_64K_S3 patterns for one-pipe standard 3D
// resources. Unlike R_X's view-as-2D form above, S3 packs Z into a true 64 KiB
// macroblock. Each row is indexed by log2(bytes-per-element), and each entry maps
// one byte-address bit to an XOR of element-coordinate bits.
constexpr PatBit3 kSw64kbS3[5][16] = {
    { // 1 B: 64x32x32
        {1,0,0}, {2,0,0}, {0,0,1}, {0,1,0}, {0,0,2}, {0,2,0}, {4,0,0}, {0,0,4},
        {0,4,0}, {8,0,0}, {0,0,8}, {0,8,0}, {16,0,0}, {0,0,16}, {0,16,0}, {32,0,0},
    },
    { // 2 B: 32x32x32
        {0,0,0}, {1,0,0}, {0,0,1}, {0,1,0}, {0,0,2}, {0,2,0}, {2,0,0}, {0,0,4},
        {0,4,0}, {4,0,0}, {0,0,8}, {0,8,0}, {8,0,0}, {0,0,16}, {0,16,0}, {16,0,0},
    },
    { // 4 B: 32x32x16
        {0,0,0}, {0,0,0}, {1,0,0}, {0,1,0}, {0,0,1}, {0,2,0}, {2,0,0}, {0,0,2},
        {0,4,0}, {4,0,0}, {0,0,4}, {0,8,0}, {8,0,0}, {0,0,8}, {0,16,0}, {16,0,0},
    },
    { // 8 B: 32x16x16 (Plucky Squire's 32-cubed RGBA16F lighting volume)
        {0,0,0}, {0,0,0}, {0,0,0}, {1,0,0}, {0,0,1}, {0,1,0}, {2,0,0}, {0,0,2},
        {0,2,0}, {4,0,0}, {0,0,4}, {0,4,0}, {8,0,0}, {0,0,8}, {0,8,0}, {16,0,0},
    },
    { // 16 B: 16x16x16
        {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}, {0,0,1}, {0,1,0}, {1,0,0}, {0,0,2},
        {0,2,0}, {2,0,0}, {0,0,4}, {0,4,0}, {4,0,0}, {0,0,8}, {0,8,0}, {8,0,0},
    },
};

constexpr uint32_t kSw64kbS3Dims[5][3] = {
    {64, 32, 32}, {32, 32, 32}, {32, 32, 16}, {32, 16, 16}, {16, 16, 16},
};

// GFX10 standard 3D swizzles are NESTED: SW_4K_S3 addresses 4 KiB using the LOW 12 bits of the very
// same per-element pattern SW_64K_S3 uses for 16, with block dimensions equal to whatever those 12
// bits can reach. The table above is therefore REUSED rather than transcribed a second time; a copy
// would be five rows of coordinate masks obliged to stay in lockstep forever, and nothing would
// notice if they stopped.
//
// Two checks that the nesting is right, neither of which is a round-trip tautology (a tile/detile
// round trip is self-consistent for any invented swizzle and proves nothing about the hardware):
//
//  1. Every element size lands on exactly 4096 bytes from its low 12 bits -- 16x16x16x1B,
//     8x16x16x2B, 8x16x8x4B, 8x8x8x8B, 4x8x8x16B. That is not automatic: a wrong bit count misses
//     4096 on all five rows at once.
//  2. Sonic Racing: CrossWorlds (PPSA08804) binds a 16x16x16 2-byte grading LUT and the GUEST
//     declares size=8192. This pattern predicts ceil(16/8) * ceil(16/16) * ceil(16/16) = 2 blocks
//     * 4096 = 8192 bytes. The guest's own size field agrees with the derivation. (#2229)
constexpr uint32_t kSw4kbS3Bits = 12;
constexpr uint32_t kSw4kbS3Dims[5][3] = {
    {16, 16, 16}, {8, 16, 16}, {8, 16, 8}, {8, 8, 8}, {4, 8, 8},
};

// The dims above are DERIVED DATA sitting next to the derivation they come from, which is a hazard:
// the pattern could change and the table not follow, with nothing to notice. The literal table is
// kept anyway -- a reader of a tiling file needs the block geometry at a glance, and a loop
// accumulating coordinate masks does not provide that -- so the drift is closed by checking it
// instead of by deleting it.
//
// Compile-time rather than a test arm, because a static_assert cannot be skipped, cannot be run on
// the wrong build directory, and reports at the point of the mistake. If the pattern changes and the
// table does not, this file stops compiling.
constexpr uint32_t sw4kb_s3_derived_dim(uint32_t el, uint32_t axis) {
    uint32_t highest = 0;
    for (uint32_t i = el; i < kSw4kbS3Bits; ++i) {
        const uint32_t mask = axis == 0 ? kSw64kbS3[el][i].x
                            : axis == 1 ? kSw64kbS3[el][i].y
                                        : kSw64kbS3[el][i].z;
        if (mask > highest) highest = mask;
    }
    // The masks are single coordinate bits, so the highest one addresses [0, 2*mask); an axis no bit
    // references spans exactly one texel.
    return highest ? highest * 2u : 1u;
}
static_assert(sw4kb_s3_derived_dim(0, 0) == kSw4kbS3Dims[0][0] &&
              sw4kb_s3_derived_dim(0, 1) == kSw4kbS3Dims[0][1] &&
              sw4kb_s3_derived_dim(0, 2) == kSw4kbS3Dims[0][2] &&
              sw4kb_s3_derived_dim(1, 0) == kSw4kbS3Dims[1][0] &&
              sw4kb_s3_derived_dim(1, 1) == kSw4kbS3Dims[1][1] &&
              sw4kb_s3_derived_dim(1, 2) == kSw4kbS3Dims[1][2] &&
              sw4kb_s3_derived_dim(2, 0) == kSw4kbS3Dims[2][0] &&
              sw4kb_s3_derived_dim(2, 1) == kSw4kbS3Dims[2][1] &&
              sw4kb_s3_derived_dim(2, 2) == kSw4kbS3Dims[2][2] &&
              sw4kb_s3_derived_dim(3, 0) == kSw4kbS3Dims[3][0] &&
              sw4kb_s3_derived_dim(3, 1) == kSw4kbS3Dims[3][1] &&
              sw4kb_s3_derived_dim(3, 2) == kSw4kbS3Dims[3][2] &&
              sw4kb_s3_derived_dim(4, 0) == kSw4kbS3Dims[4][0] &&
              sw4kb_s3_derived_dim(4, 1) == kSw4kbS3Dims[4][1] &&
              sw4kb_s3_derived_dim(4, 2) == kSw4kbS3Dims[4][2],
              "kSw4kbS3Dims no longer matches what the low 12 bits of kSw64kbS3 can address");
// And that each derived block is exactly 4 KiB -- the property that fails on a wrong bit count.
static_assert(kSw4kbS3Dims[0][0] * kSw4kbS3Dims[0][1] * kSw4kbS3Dims[0][2] * 1u == 4096 &&
              kSw4kbS3Dims[1][0] * kSw4kbS3Dims[1][1] * kSw4kbS3Dims[1][2] * 2u == 4096 &&
              kSw4kbS3Dims[2][0] * kSw4kbS3Dims[2][1] * kSw4kbS3Dims[2][2] * 4u == 4096 &&
              kSw4kbS3Dims[3][0] * kSw4kbS3Dims[3][1] * kSw4kbS3Dims[3][2] * 8u == 4096 &&
              kSw4kbS3Dims[4][0] * kSw4kbS3Dims[4][1] * kSw4kbS3Dims[4][2] * 16u == 4096,
              "a SW_4KB_S3 block no longer holds exactly 4096 bytes");

// Shared by both standard-3D block sizes. `bits` is log2 of the block, so 16 -> 64 KiB (S3) and
// 12 -> 4 KiB (4K_S3); `dims` is the matching block geometry.
size_t s3_volume_bytes(uint32_t width, uint32_t height, uint32_t depth, uint32_t bpe,
                       uint32_t bits, const uint32_t (*dims)[3]) {
    const uint32_t el = sw64kb_elem_log2(bpe);
    if (el == UINT32_MAX || !width || !height || !depth) return 0;
    const uint64_t block_bytes = 1ull << bits;
    const uint64_t blocks_x = (static_cast<uint64_t>(width) + dims[el][0] - 1) / dims[el][0];
    const uint64_t blocks_y = (static_cast<uint64_t>(height) + dims[el][1] - 1) / dims[el][1];
    const uint64_t blocks_z = (static_cast<uint64_t>(depth) + dims[el][2] - 1) / dims[el][2];
    if (blocks_x > SIZE_MAX / block_bytes ||
        blocks_y > SIZE_MAX / (blocks_x * block_bytes) ||
        blocks_z > SIZE_MAX / (blocks_x * blocks_y * block_bytes)) return 0;
    return static_cast<size_t>(blocks_x * blocks_y * blocks_z * block_bytes);
}

size_t sw64kb_s3_volume_bytes(uint32_t width, uint32_t height, uint32_t depth, uint32_t bpe) {
    return s3_volume_bytes(width, height, depth, bpe, 16, kSw64kbS3Dims);
}

size_t sw4kb_s3_volume_bytes(uint32_t width, uint32_t height, uint32_t depth, uint32_t bpe) {
    return s3_volume_bytes(width, height, depth, bpe, kSw4kbS3Bits, kSw4kbS3Dims);
}

// `bits` selects the block size: 16 for SW_64K_S3, 12 for SW_4K_S3. Because 4K_S3's pattern is the
// low 12 bits of the same table, the only differences are the loop bound on the pattern bits, the
// block geometry, and the shift that concatenates block index with in-block offset.
template <bool ToTiled>
bool s3_volume_copy(uint8_t* dst, const uint8_t* src, size_t tiled_bytes,
                    uint32_t width, uint32_t height, uint32_t depth, uint32_t bpe,
                    uint32_t bits, const uint32_t (*dims)[3]) {
    const uint32_t el = sw64kb_elem_log2(bpe);
    if (el == UINT32_MAX) return false;
    const uint32_t bw = dims[el][0];
    const uint32_t bh = dims[el][1];
    const uint32_t bd = dims[el][2];
    const uint64_t blocks_x = (static_cast<uint64_t>(width) + bw - 1) / bw;
    const uint64_t blocks_y = (static_cast<uint64_t>(height) + bh - 1) / bh;
    const PatBit3* pat = kSw64kbS3[el];
    std::vector<uint16_t> fx(width), fy(height), fz(depth);
    for (uint32_t x = 0; x < width; ++x) {
        uint32_t v = 0;
        for (uint32_t i = el; i < bits; ++i)
            v |= static_cast<uint32_t>(__builtin_popcount(x & pat[i].x) & 1) << i;
        fx[x] = static_cast<uint16_t>(v);
    }
    for (uint32_t y = 0; y < height; ++y) {
        uint32_t v = 0;
        for (uint32_t i = el; i < bits; ++i)
            v |= static_cast<uint32_t>(__builtin_popcount(y & pat[i].y) & 1) << i;
        fy[y] = static_cast<uint16_t>(v);
    }
    for (uint32_t z = 0; z < depth; ++z) {
        uint32_t v = 0;
        for (uint32_t i = el; i < bits; ++i)
            v |= static_cast<uint32_t>(__builtin_popcount(z & pat[i].z) & 1) << i;
        fz[z] = static_cast<uint16_t>(v);
    }
    for (uint32_t z = 0; z < depth; ++z) {
        const uint64_t block_slab = static_cast<uint64_t>(z / bd) * blocks_y * blocks_x;
        for (uint32_t y = 0; y < height; ++y) {
            const uint64_t block_row = block_slab + static_cast<uint64_t>(y / bh) * blocks_x;
            for (uint32_t x = 0; x < width; ++x) {
                const uint64_t block = block_row + x / bw;
                const uint64_t tiled = (block << bits) | (fx[x] ^ fy[y] ^ fz[z]);
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

template <bool ToTiled>
bool sw64kb_s3_volume_copy(uint8_t* dst, const uint8_t* src, size_t tiled_bytes,
                           uint32_t width, uint32_t height, uint32_t depth, uint32_t bpe) {
    return s3_volume_copy<ToTiled>(dst, src, tiled_bytes, width, height, depth, bpe,
                                   16, kSw64kbS3Dims);
}

template <bool ToTiled>
bool sw4kb_s3_volume_copy(uint8_t* dst, const uint8_t* src, size_t tiled_bytes,
                          uint32_t width, uint32_t height, uint32_t depth, uint32_t bpe) {
    return s3_volume_copy<ToTiled>(dst, src, tiled_bytes, width, height, depth, bpe,
                                   kSw4kbS3Bits, kSw4kbS3Dims);
}
} // namespace

size_t tiled_surface_bytes(uint32_t width, uint32_t height, uint32_t tile_mode, uint32_t pitch,
                           uint32_t bytes_per_texel) {
    if (!tile_mode_is_tiled(tile_mode)) return (size_t)width * height * bytes_per_texel;
    if (tile_mode == (uint32_t)TileMode::Sw256BS)
        return sw256_tiled_bytes(width, height, pitch, bytes_per_texel);
    if (is_64kb_mode(tile_mode)) return sw64kb_tiled_bytes(width, height, pitch, bytes_per_texel);
    return sw4kb_tiled_bytes(width, height, pitch, bytes_per_texel);
}


// DIAGNOSTIC (PROSPER_TILECENSUS=1): which tiling work actually runs, by geometry.
//
// #3149 measured that this file's block copier is ~22% of CPU during Stray's boot while the GPU sits
// at 4.5%, but no profiler here can name the CALL SITE: at -O3 the callers inline away (folded stacks
// resolve to `~unique_ptr`) and DWARF unwinding collapses through the guest JIT to `main+0x...`. A
// counter at the leaf answers the question the profiler cannot -- not which line calls it, but which
// SURFACE it is called on and how often, which is what identifies the workload.
//
// Keyed on (direction, w, h, bytes-per-element, tile_mode) and dumped at exit, so a 3840x2160x8
// surface running once per frame is immediately distinguishable from a small one running constantly.
namespace {
// `who` is part of the KEY, not the value.  Holding it as a value field made attribution
// last-writer-wins: one geometry reached from two call sites reported whichever ran last, which
// on #3149 printed the outer `compute` scope for a row whose work is really `smpl-upl`.  Keying on
// it splits that row instead, so a geometry reached from several sites is visible as several rows
// and the percentages stay attributable.  Tags and ops are string literals, so pointer identity is
// the intended comparison.
struct TileCensusKey {
    const char* op; const char* who; uint32_t w, h, bpe, mode;
    bool operator==(const TileCensusKey& o) const {
        return op == o.op && who == o.who && w == o.w && h == o.h && bpe == o.bpe &&
               mode == o.mode;
    }
};
struct TileCensusHash {
    size_t operator()(const TileCensusKey& k) const {
        // A fixed-width 64-bit accumulator rather than shifts off `size_t`: the previous form
        // shifted by 33, which is UB rather than a wrap wherever `size_t` is 32-bit.  Nothing here
        // needs the spread those shifts were reaching for.
        uint64_t h = std::hash<const void*>{}(k.op);
        h = h * 1099511628211ull ^ std::hash<const void*>{}(k.who);
        h = h * 1099511628211ull ^ k.w;
        h = h * 1099511628211ull ^ k.h;
        h = h * 1099511628211ull ^ k.bpe;
        h = h * 1099511628211ull ^ k.mode;
        return static_cast<size_t>(h);
    }
};
struct TileCensusValue { uint64_t calls = 0, bytes = 0; };
std::mutex g_tile_census_mx;
std::unordered_map<TileCensusKey, TileCensusValue, TileCensusHash> g_tile_census;

thread_local const char* g_tile_census_tag = "?";

bool tile_census_enabled() {
    static const bool on = std::getenv("PROSPER_TILECENSUS") != nullptr;
    return on;
}

void tile_census_report() {
    std::lock_guard<std::mutex> lk(g_tile_census_mx);
    std::vector<std::pair<TileCensusKey, TileCensusValue>> rows(
        g_tile_census.begin(), g_tile_census.end());
    std::sort(rows.begin(), rows.end(),
              [](const auto& a, const auto& b) { return a.second.bytes > b.second.bytes; });
    uint64_t total_bytes = 0, total_calls = 0;
    for (const auto& r : rows) { total_bytes += r.second.bytes; total_calls += r.second.calls; }
    std::fprintf(stderr, "[tilecensus] %zu geometries, %llu calls, %.1f MiB moved\n",
                 rows.size(), (unsigned long long)total_calls,
                 double(total_bytes) / (1024.0 * 1024.0));
    for (size_t i = 0; i < rows.size() && i < 14; ++i)
        std::fprintf(stderr,
                     "[tilecensus]   %-9s %-20s %5ux%-5u bpe=%u mode=%u calls=%llu %.1f MiB (%.1f%%)\n",
                     rows[i].first.who, rows[i].first.op, rows[i].first.w, rows[i].first.h, rows[i].first.bpe,
                     rows[i].first.mode, (unsigned long long)rows[i].second.calls,
                     double(rows[i].second.bytes) / (1024.0 * 1024.0),
                     total_bytes ? 100.0 * double(rows[i].second.bytes) / double(total_bytes) : 0.0);
}

// Row count for a volume, saturating instead of wrapping: this only names a census bucket, so a
// clamped value groups honestly while an overflowed one silently merges two unrelated geometries.
// Note what saturating does NOT fix: the row's `bytes` weight is derived from the same clamped
// value, and the report ranks by bytes, so a saturated row would sort high on a fabricated weight.
// Unreachable for any real surface -- a volume would need >2^32 rows -- and left visible rather
// than dropped, because a census that silently discards a geometry cannot show its own invalidity.
uint32_t census_rows(uint32_t height, uint32_t depth) {
    const uint64_t rows = static_cast<uint64_t>(height) * depth;
    return rows > 0xffffffffull ? 0xffffffffu : static_cast<uint32_t>(rows);
}

void tile_census_note(const char* op, uint32_t w, uint32_t h, uint32_t bpe, uint32_t mode) {
    if (!tile_census_enabled()) return;
    bool due = false;
    {
        std::lock_guard<std::mutex> lk(g_tile_census_mx);
        auto& v = g_tile_census[TileCensusKey{op, g_tile_census_tag, w, h, bpe, mode}];
        ++v.calls;
        v.bytes += uint64_t(w) * h * bpe;
        // Dumped PERIODICALLY, not at exit: a bounded capture run is killed with SIGTERM and every
        // frontend here exits via _exit, so an atexit report would never print -- the same trap that
        // hid the dmem write trace's output on #3146. A periodic dump always produces evidence.
        // Threshold on BYTES as well as calls: this workload is a few enormous copies (a 4K FP16
        // surface is 66 MiB), so a call-count threshold alone never fires and the instrument stays
        // silent while moving gigabytes -- which is indistinguishable from "nothing happened".
        static uint64_t since = 0, since_bytes = 0;
        ++since;
        since_bytes += uint64_t(w) * h * bpe;
        if (since >= 20000 || since_bytes >= (1ull << 30)) {
            since = 0; since_bytes = 0; due = true;
        }
    }
    if (due) tile_census_report();
}
} // namespace

void detile_surface(uint8_t* dst, const uint8_t* src, uint32_t width, uint32_t height,
                    uint32_t tile_mode, uint32_t pitch, uint32_t bytes_per_texel) {
    tile_census_note("detile_surface", width, height, bytes_per_texel, tile_mode);
    if (!tile_mode_is_tiled(tile_mode)) { warn_unhandled_tile_mode(tile_mode, width, height);
                                          std::memcpy(dst, src, (size_t)width * height * bytes_per_texel); return; }
    if (tile_mode == (uint32_t)TileMode::Sw256BS) {
        sw256_copy<false>(dst, src, width, height, pitch, bytes_per_texel,
                          sw256_tiled_bytes(width, height, pitch, bytes_per_texel));
        return;
    }
    if (is_64kb_mode(tile_mode)) {
        sw64kb_copy<false>(dst, src, width, height, pitch, bytes_per_texel,
                           sw64kb_tiled_bytes(width, height, pitch, bytes_per_texel), tile_mode);
        return;
    }
    sw4kb_copy<false>(dst, src, width, height, pitch, bytes_per_texel,
                      sw4kb_tiled_bytes(width, height, pitch, bytes_per_texel));
}

bool detile_writes_whole_destination(uint32_t tile_mode, uint32_t bytes_per_texel) {
    // Untiled: a straight memcpy of exactly width*height*bytes_per_texel.
    if (!tile_mode_is_tiled(tile_mode)) return true;
    // sw256_copy and sw4kb_copy iterate every (x, y) in the surface and either copy the element or
    // memset it to zero, for any bpe their lookup accepts -- there is no early-out that writes less.
    if (tile_mode == (uint32_t)TileMode::Sw256BS) return true;
    // sw64kb_copy has one: an element size its pattern tables do not cover falls back to
    // `memcpy(dst, src, min(n, tiled_bytes))`, which can stop short of the destination.
    if (is_64kb_mode(tile_mode)) return sw64kb_elem_log2(bytes_per_texel) != UINT32_MAX;
    return true;
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
                  uint32_t tile_mode, uint32_t pitch, uint32_t bytes_per_texel,
                  bool allow_avx2) {
    tile_census_note("tile_surface", width, height, bytes_per_texel, tile_mode);
    if (!tile_mode_is_tiled(tile_mode)) { std::memcpy(dst, src, (size_t)width * height * bytes_per_texel); return; }
    const size_t dst_bytes = tiled_surface_bytes(width, height, tile_mode, pitch, bytes_per_texel);
    // Zero only what the copy will NOT write. The zero-fill exists for padding texels -- the ones
    // in blocks that extend past `width`/`height` -- but the copy overwrites every in-bounds texel,
    // so zeroing the whole surface repeats that work. On a 3840x2160 target this memset alone is
    // 33 MiB per writeback, and this file's block copier already measures ~22% of CPU (#3149).
    //
    // 64 KiB modes lay blocks out row-major (block = (y/bh)*blocks_x + (x/bw), 64 KiB each), so
    // when `width` is block-aligned there is no column padding and every block row below
    // height/bh is fully covered. Those rows need no zeroing; the tail from the first partially
    // covered row onward still does. Any other mode, an unaligned width, or a non-zero pitch keeps
    // the whole-surface fill, which is always correct.
    size_t zero_from = 0;
    if (is_64kb_mode(tile_mode) && pitch == 0) {
        const uint32_t el = sw64kb_elem_log2(bytes_per_texel);
        if (el != UINT32_MAX) {
            uint32_t bw = 0, bh = 0;
            sw64kb_dims(el, bw, bh);
            if (bw && bh && width % bw == 0) {
                const size_t blocks_x = width / bw;
                const size_t covered_rows = height / bh;   // block rows entirely within `height`
                const size_t covered = covered_rows * blocks_x * 65536u;
                if (covered <= dst_bytes) zero_from = covered;
            }
        }
    }
    std::memset(dst + zero_from, 0, dst_bytes - zero_from);
    if (tile_mode == (uint32_t)TileMode::Sw256BS) {
        sw256_copy<true>(dst, src, width, height, pitch, bytes_per_texel, dst_bytes);
        return;
    }
    if (is_64kb_mode(tile_mode)) {
        sw64kb_copy<true>(dst, src, width, height, pitch, bytes_per_texel, dst_bytes, tile_mode,
                          0, allow_avx2);
        return;
    }
    sw4kb_copy<true>(dst, src, width, height, pitch, bytes_per_texel, dst_bytes);
}

size_t tiled_msaa_surface_bytes(uint32_t width, uint32_t height, uint32_t tile_mode,
                                uint32_t bytes_per_texel, uint32_t sample_count) {
    if (!width || !height || sample_count != 4u ||
        tile_mode != static_cast<uint32_t>(TileMode::Sw64KbZX))
        return 0;
    const uint32_t element_log2 = sw64kb_elem_log2(bytes_per_texel);
    if (element_log2 == UINT32_MAX || element_log2 + 2u >= 16u) return 0;
    // AMD AddrLib ComputeThinBlockDimension: log2(elements) = log2(64 KiB) - log2(B/element)
    // - log2(samples). Four samples use the width-precedent square/square-ish split.
    const uint32_t element_bits = 16u - element_log2 - 2u;
    const uint32_t width_bits = (element_bits + 1u) / 2u;
    const uint32_t block_width = 1u << width_bits;
    const uint32_t block_height = 1u << (element_bits - width_bits);
    const uint64_t blocks_x = (static_cast<uint64_t>(width) + block_width - 1u) / block_width;
    const uint64_t blocks_y = (static_cast<uint64_t>(height) + block_height - 1u) / block_height;
    if (blocks_x && blocks_y > SIZE_MAX / blocks_x / 65536u) return 0;
    return static_cast<size_t>(blocks_x * blocks_y * 65536u);
}

namespace {
template <bool ToTiled>
bool sw64kb_zx_4xaa_copy(uint8_t* dst, size_t tiled_bytes, const uint8_t* src,
                         uint32_t width, uint32_t height, uint32_t bytes_per_texel) {
    const uint32_t element_log2 = sw64kb_elem_log2(bytes_per_texel);
    if (!dst || !src || element_log2 == UINT32_MAX) return false;
    const uint32_t element_bits = 16u - element_log2 - 2u;
    const uint32_t width_bits = (element_bits + 1u) / 2u;
    const uint32_t block_width = 1u << width_bits;
    const uint32_t block_height = 1u << (element_bits - width_bits);
    const uint64_t blocks_x = (static_cast<uint64_t>(width) + block_width - 1u) / block_width;
    const PatBitMsaa* pattern = kSw64kZX4x[element_log2];

    std::vector<uint16_t> x_offsets(width), y_offsets(height);
    std::array<uint16_t, 4> sample_offsets{};
    for (uint32_t x = 0; x < width; ++x)
        for (uint32_t bit = element_log2; bit < 16u; ++bit)
            x_offsets[x] |= static_cast<uint16_t>(
                (__builtin_popcount(x & pattern[bit].x) & 1u) << bit);
    for (uint32_t y = 0; y < height; ++y)
        for (uint32_t bit = element_log2; bit < 16u; ++bit)
            y_offsets[y] |= static_cast<uint16_t>(
                (__builtin_popcount(y & pattern[bit].y) & 1u) << bit);
    for (uint32_t sample = 0; sample < 4u; ++sample)
        for (uint32_t bit = element_log2; bit < 16u; ++bit)
            sample_offsets[sample] |= static_cast<uint16_t>(
                (__builtin_popcount(sample & pattern[bit].sample) & 1u) << bit);

    const size_t plane_texels = static_cast<size_t>(width) * height;
    for (uint32_t sample = 0; sample < 4u; ++sample) {
        for (uint32_t y = 0; y < height; ++y) {
            const uint64_t block_row = static_cast<uint64_t>(y / block_height) * blocks_x;
            for (uint32_t x = 0; x < width; ++x) {
                const uint64_t block = block_row + x / block_width;
                const uint64_t tiled = (block << 16u) |
                    (x_offsets[x] ^ y_offsets[y] ^ sample_offsets[sample]);
                const size_t linear =
                    (static_cast<size_t>(sample) * plane_texels +
                     static_cast<size_t>(y) * width + x) * bytes_per_texel;
                if (tiled > tiled_bytes || bytes_per_texel > tiled_bytes - tiled) {
                    if constexpr (!ToTiled) std::memset(dst + linear, 0, bytes_per_texel);
                    continue;
                }
                if constexpr (ToTiled)
                    std::memcpy(dst + tiled, src + linear, bytes_per_texel);
                else
                    std::memcpy(dst + linear, src + tiled, bytes_per_texel);
            }
        }
    }
    return true;
}
} // namespace

bool detile_msaa_surface(uint8_t* dst, const uint8_t* src, size_t src_bytes,
                         uint32_t width, uint32_t height, uint32_t tile_mode,
                         uint32_t bytes_per_texel, uint32_t sample_count) {
    const size_t expected = tiled_msaa_surface_bytes(
        width, height, tile_mode, bytes_per_texel, sample_count);
    if (!expected || src_bytes < expected) return false;
    return sw64kb_zx_4xaa_copy<false>(dst, src_bytes, src, width, height, bytes_per_texel);
}

bool tile_msaa_surface(uint8_t* dst, size_t dst_bytes, const uint8_t* src,
                       uint32_t width, uint32_t height, uint32_t tile_mode,
                       uint32_t bytes_per_texel, uint32_t sample_count) {
    const size_t expected = tiled_msaa_surface_bytes(
        width, height, tile_mode, bytes_per_texel, sample_count);
    if (!expected || dst_bytes < expected) return false;
    std::memset(dst, 0, expected);
    return sw64kb_zx_4xaa_copy<true>(dst, expected, src, width, height, bytes_per_texel);
}

size_t gfx10_htile_msaa_metadata_bytes(uint32_t width, uint32_t height,
                                       uint32_t tile_mode, uint32_t sample_count,
                                       bool pipe_aligned) {
    if (!width || !height || sample_count != 4u || !pipe_aligned ||
        tile_mode != static_cast<uint32_t>(TileMode::Sw64KbZX))
        return 0;
    // AMD AddrLib HwlComputeHtileInfo/GetMetaBlkSize, 16-pipe GFX10 configuration:
    //   ROCm/ROCR-Runtime d614ea8bbd73, src/image/addrlib/src/gfx10/gfx10addrlib.cpp
    // Pipe-aligned thin Z_X uses a 2^15-byte meta block. Its 2^19 covered pixels split
    // width-precedent into 1024x512. HTILE sizing is independent of the depth sample count, but
    // this API deliberately retains the observed 4xaa gate rather than claiming broader support.
    constexpr uint64_t meta_width = 1024u;
    constexpr uint64_t meta_height = 512u;
    constexpr uint64_t meta_block_bytes = 1u << 15;
    const uint64_t blocks_x = (static_cast<uint64_t>(width) + meta_width - 1u) / meta_width;
    const uint64_t blocks_y = (static_cast<uint64_t>(height) + meta_height - 1u) / meta_height;
    if (blocks_x && blocks_y > SIZE_MAX / blocks_x / meta_block_bytes) return 0;
    return static_cast<size_t>(blocks_x * blocks_y * meta_block_bytes);
}

bool gfx10_htile_metadata_is_decompressed(const uint8_t* metadata, size_t metadata_bytes,
                                           size_t expected_bytes, uint32_t* uniform_value) {
    if (uniform_value) *uniform_value = 0;
    if (!metadata || !expected_bytes || metadata_bytes != expected_bytes ||
        (metadata_bytes % sizeof(uint32_t)) != 0)
        return false;
    uint32_t first = 0;
    std::memcpy(&first, metadata, sizeof(first));
    // PAL Gfx9Htile::GetInitialValue(), c5e800072a32, gfx9MaskRam.cpp:
    // - depth-only:    ZMax=0x3fff, ZMin=0, ZMask=0xf (no Z compression)
    // - depth+stencil: full Z range, SMem=3 (no stencil compression), SR0/SR1 unknown,
    //                  ZMask=0xf (no Z compression)
    constexpr uint32_t depth_only_decompressed = 0xfffc000fu;
    constexpr uint32_t depth_stencil_decompressed = 0xfffff3ffu;
    if (first != depth_only_decompressed && first != depth_stencil_decompressed) return false;
    for (size_t offset = sizeof(uint32_t); offset < metadata_bytes;
         offset += sizeof(uint32_t)) {
        uint32_t value = 0;
        std::memcpy(&value, metadata + offset, sizeof(value));
        if (value != first) return false;
    }
    if (uniform_value) *uniform_value = first;
    return true;
}

Gfx10HtileMsaaSource gfx10_htile_msaa_source(
    const uint8_t* metadata, size_t metadata_bytes,
    uint32_t width, uint32_t height, uint32_t tile_mode,
    uint32_t bytes_per_texel, uint32_t sample_count, bool pipe_aligned) {
    if (bytes_per_texel != sizeof(float)) return Gfx10HtileMsaaSource::Unsupported;
    const size_t expected = gfx10_htile_msaa_metadata_bytes(
        width, height, tile_mode, sample_count, pipe_aligned);
    if (!metadata || !expected || metadata_bytes != expected)
        return Gfx10HtileMsaaSource::Unsupported;

    // PAL Gfx9Htile::GetClearValue(depth=0), depth-only layout:
    //   (ZMax << 18) | (ZMin << 4) | ZMask = 0.
    // A full uniform plane therefore describes every 8x8 block and all samples as +0.0 without
    // consulting base memory. Do not generalize this to nonzero encodings here: inverse depth
    // quantization and the depth+stencil layout require their own proven contract.
    if (std::all_of(metadata, metadata + metadata_bytes,
                    [](uint8_t value) { return value == 0u; }))
        return Gfx10HtileMsaaSource::DepthZeroFastClear;
    if (gfx10_htile_metadata_is_decompressed(
            metadata, metadata_bytes, expected))
        return Gfx10HtileMsaaSource::UncompressedBase;
    return Gfx10HtileMsaaSource::Unsupported;
}

bool materialize_gfx10_htile_msaa_surface(
    uint8_t* dst, size_t dst_bytes,
    const uint8_t* tiled_base, size_t tiled_base_bytes,
    const uint8_t* metadata, size_t metadata_bytes,
    uint32_t width, uint32_t height, uint32_t tile_mode,
    uint32_t bytes_per_texel, uint32_t sample_count, bool pipe_aligned,
    Gfx10HtileMsaaSource* source) {
    if (source) *source = Gfx10HtileMsaaSource::Unsupported;
    if (!dst || !width || !height || !sample_count || !bytes_per_texel)
        return false;
    size_t linear_bytes = width;
    if (linear_bytes > SIZE_MAX / height) return false;
    linear_bytes *= height;
    if (linear_bytes > SIZE_MAX / sample_count) return false;
    linear_bytes *= sample_count;
    if (linear_bytes > SIZE_MAX / bytes_per_texel) return false;
    linear_bytes *= bytes_per_texel;
    if (dst_bytes < linear_bytes) return false;

    const Gfx10HtileMsaaSource selected = gfx10_htile_msaa_source(
        metadata, metadata_bytes, width, height, tile_mode,
        bytes_per_texel, sample_count, pipe_aligned);
    if (selected == Gfx10HtileMsaaSource::DepthZeroFastClear) {
        // IEEE-754 +0.0 is all-zero bits. Base may be null, short, or poison: fast-clear metadata is
        // authoritative and the stale tiled allocation must not be observed.
        std::memset(dst, 0, linear_bytes);
    } else if (selected == Gfx10HtileMsaaSource::UncompressedBase) {
        const size_t expected_base = tiled_msaa_surface_bytes(
            width, height, tile_mode, bytes_per_texel, sample_count);
        if (!tiled_base || !expected_base || tiled_base_bytes < expected_base ||
            !detile_msaa_surface(dst, tiled_base, tiled_base_bytes, width, height,
                                tile_mode, bytes_per_texel, sample_count))
            return false;
    } else {
        return false;
    }
    if (source) *source = selected;
    return true;
}

size_t tiled_elements_bytes(uint32_t ew, uint32_t eh, uint32_t bpe, uint32_t tile_mode) {
    if (!tile_mode_is_tiled(tile_mode) || bpe == 0) return (size_t)ew * eh * bpe;
    if (tile_mode == (uint32_t)TileMode::Sw256BS)
        return sw256_tiled_bytes(ew, eh, /*pitch*/0, bpe);
    if (is_64kb_mode(tile_mode)) return sw64kb_tiled_bytes(ew, eh, /*pitch*/0, bpe);
    return sw4kb_tiled_bytes(ew, eh, /*pitch*/0, bpe);
}

TiledMipLevelLayout tiled_mip_level_layout(uint32_t ew, uint32_t eh, uint32_t bpe,
                                           uint32_t tile_mode, uint32_t max_mip,
                                           uint32_t mip_level) {
    TiledMipLevelLayout result;
    if (!ew || !eh || !bpe || mip_level > max_mip || max_mip >= 16)
        return result;

    if (tile_mode == 0) {
        // Official GFX10 AddrLib HwlComputeSurfaceInfoLinear: ADDR_SW_LINEAR aligns every mip's
        // pitch to 256 bytes and stores a multi-level chain in reverse order (smallest level first).
        // pMipInfo[i].offset is therefore the sum of aligned levels max_mip..i+1.
        if (bpe > 256 || (256 % bpe) != 0) return result;
        const uint32_t pitch_align = 256 / bpe;
        size_t offset = 0;
        for (uint32_t level = max_mip; level > mip_level; --level) {
            const uint32_t width = std::max(ew >> level, 1u);
            const uint32_t height = std::max(eh >> level, 1u);
            const uint32_t pitch = ((width + pitch_align - 1) / pitch_align) * pitch_align;
            const size_t level_bytes = static_cast<size_t>(pitch) * height * bpe;
            if (offset > SIZE_MAX - level_bytes) return {};
            offset += level_bytes;
        }
        result.byte_offset = offset;
        result.supported = true;
        return result;
    }
    if (!tile_mode_is_tiled(tile_mode)) return result;

    if (tile_mode == (uint32_t)TileMode::Sw256BS) {
        if (bpe != 1 && bpe != 2 && bpe != 4 && bpe != 8 && bpe != 16)
            return result;
        uint32_t bx = 0, by = 0;
        sw256_dims(bpe, bx, by);
        const uint32_t block_width = 1u << bx;
        const uint32_t block_height = 1u << by;
        // AddrLib ComputeSurfaceInfoMicroTiled stores a mip chain in reverse order, smallest first,
        // with each level independently aligned to its 256-byte block. Unlike macro-tiled modes it
        // has no shared mip tail.
        size_t offset = 0;
        for (uint32_t level = max_mip; level > mip_level; --level) {
            const uint32_t width = std::max(ew >> level, 1u);
            const uint32_t height = std::max(eh >> level, 1u);
            const uint32_t pitch = (width + block_width - 1u) / block_width * block_width;
            const uint32_t padded_height =
                (height + block_height - 1u) / block_height * block_height;
            const size_t level_bytes = static_cast<size_t>(pitch) * padded_height * bpe;
            if (offset > SIZE_MAX - level_bytes) return {};
            offset += level_bytes;
        }
        result.byte_offset = offset;
        result.supported = true;
        return result;
    }

    uint32_t block_width = 0, block_height = 0, block_log2 = 0;
    if (is_64kb_mode(tile_mode)) {
        const uint32_t elem_log2 = sw64kb_elem_log2(bpe);
        if (elem_log2 == UINT32_MAX) return result;
        sw64kb_dims(elem_log2, block_width, block_height);
        block_log2 = 16;
    } else {
        if (bpe != 1 && bpe != 2 && bpe != 4 && bpe != 8 && bpe != 16)
            return result;
        uint32_t bx = 0, by = 0;
        sw4kb_dims(bpe, bx, by);
        block_width = 1u << bx;
        block_height = 1u << by;
        block_log2 = 12;
    }
    result.supported = true;
    if (max_mip == 0) return result;

    const uint32_t num_levels = max_mip + 1;
    const uint32_t tail_width = block_width >> 1;   // AddrLib GetMipTailDim, thin GFX10 resource
    const uint32_t tail_height = block_height;
    const uint32_t max_tail_levels = block_log2 <= 11
        ? 1u + (1u << (block_log2 - 9u)) : block_log2 - 4u;
    uint32_t first_tail = num_levels;
    for (uint32_t level = 0; level < num_levels; ++level) {
        const uint32_t width = std::max(ew >> level, 1u);
        const uint32_t height = std::max(eh >> level, 1u);
        if (width <= tail_width && height <= tail_height &&
            num_levels - level <= max_tail_levels) {
            first_tail = level;
            break;
        }
    }
    if (mip_level >= first_tail) {
        // GFX10 AddrLib ComputeSurfaceInfo: tail levels use an addressable byte origin inside the
        // allocation's first macroblock. The alternating bit extraction below converts that byte
        // origin to the equivalent element coordinate; keeping both makes the layout independently
        // testable and documents why adding byte_offset to the ordinary within-level swizzle works.
        const uint32_t m = max_tail_levels - 1u - (mip_level - first_tail);
        const uint32_t mip_offset = m > 6u ? (16u << m) : (m << 8u);
        uint32_t mip_x = ((mip_offset >> 9)  & 1u) |
                         ((mip_offset >> 10) & 2u) |
                         ((mip_offset >> 11) & 4u) |
                         ((mip_offset >> 12) & 8u) |
                         ((mip_offset >> 13) & 16u) |
                         ((mip_offset >> 14) & 32u);
        uint32_t mip_y = ((mip_offset >> 8)  & 1u) |
                         ((mip_offset >> 9)  & 2u) |
                         ((mip_offset >> 10) & 4u) |
                         ((mip_offset >> 11) & 8u) |
                         ((mip_offset >> 12) & 16u) |
                         ((mip_offset >> 13) & 32u);
        const uint32_t elem_log2 = static_cast<uint32_t>(__builtin_ctz(bpe));
        if (block_log2 & 1u) {
            std::swap(mip_x, mip_y);
            if (elem_log2 & 1u) {
                mip_y = (mip_y << 1u) | (mip_x & 1u);
                mip_x >>= 1u;
            }
        }
        result.byte_offset = mip_offset;
        // mip_x/mip_y count whole 256-BYTE blocks, so converting them to elements needs that block's
        // element extent, which depends ONLY on the element size (AddrLib Block256_2d): 1B=16x16,
        // 2B=16x8, 4B=8x8, 8B=8x4, 16B=4x4. The previous `block_width >> 4` derived the multiplier
        // from the macroblock instead, which is right only when the macroblock happens to be 16x the
        // 256-byte block (the 64 KiB modes) and four times too small for every 4 KiB mode. A 4 KiB
        // 32-bpp macroblock is 32x32 elements, so it produced tail_x=4 where the level really starts
        // at element 16 -- i.e. byte 0x80 instead of the byte_offset 0x800 reported alongside it, so
        // the two origins in this same struct disagreed and packed-tail levels decoded foreign texels.
        // The invariant now asserted by the tile tests: the element origin must address exactly
        // byte_offset under the surface's own swizzle.
        static constexpr uint32_t kBlock256Width[5]  = {16, 16, 8, 8, 4};
        static constexpr uint32_t kBlock256Height[5] = {16,  8, 8, 4, 4};
        if (elem_log2 >= 5) return {};
        result.tail_x = mip_x * kBlock256Width[elem_log2];
        result.tail_y = mip_y * kBlock256Height[elem_log2];
        result.tail_block_bytes = 1u << block_log2;
        result.in_tail = true;
        return result;
    }

    const size_t block_bytes = size_t{1} << block_log2;
    size_t offset = first_tail == num_levels ? 0 : block_bytes;
    for (int level = static_cast<int>(first_tail) - 1; level >= 0; --level) {
        if (static_cast<uint32_t>(level) == mip_level) {
            result.byte_offset = offset;
            return result;
        }
        const uint32_t width = std::max(ew >> level, 1u);
        const uint32_t height = std::max(eh >> level, 1u);
        const uint32_t pitch = (width + block_width - 1) / block_width * block_width;
        const uint32_t padded_height =
            (height + block_height - 1) / block_height * block_height;
        const size_t level_bytes = static_cast<size_t>(pitch) * padded_height * bpe;
        if (offset > SIZE_MAX - level_bytes) return {};
        offset += level_bytes;
    }
    return {};
}

size_t tiled_mip_chain_bytes(uint32_t ew, uint32_t eh, uint32_t bpe,
                             uint32_t tile_mode, uint32_t max_mip) {
    const TiledMipLevelLayout level_zero =
        tiled_mip_level_layout(ew, eh, bpe, tile_mode, max_mip, 0);
    if (!level_zero.supported) return 0;
    // Small allocations can place every level, including level zero, in one shared macroblock.
    // Their complete per-slice chain is exactly that tail block; rejecting it made valid small cube
    // maps impossible to stride even though every individual level had a proven tail coordinate.
    if (level_zero.in_tail) return level_zero.tail_block_bytes;
    const size_t level_zero_bytes = tile_mode == 0
        ? linear_sampled_surface_bytes(ew, eh, bpe)
        : tiled_surface_bytes(ew, eh, tile_mode, 0, bpe);
    if (!level_zero_bytes || level_zero.byte_offset > SIZE_MAX - level_zero_bytes) return 0;
    return level_zero.byte_offset + level_zero_bytes;
}

size_t tiled_mip_level_offset(uint32_t ew, uint32_t eh, uint32_t bpe, uint32_t tile_mode,
                              uint32_t max_mip, uint32_t mip_level) {
    const TiledMipLevelLayout layout = tiled_mip_level_layout(
        ew, eh, bpe, tile_mode, max_mip, mip_level);
    return layout.supported && !layout.in_tail ? layout.byte_offset : 0;
}

void detile_elements(uint8_t* dst, const uint8_t* src, size_t src_bytes,
                     uint32_t ew, uint32_t eh, uint32_t bpe, uint32_t tile_mode) {
    tile_census_note("detile_elements", ew, eh, bpe, tile_mode);
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
    if (tile_mode == (uint32_t)TileMode::Sw256BS) {
        sw256_copy<false>(dst, src, ew, eh, /*pitch*/0, bpe, src_bytes);
        return;
    }
    sw4kb_copy<false>(dst, src, ew, eh, /*pitch*/0, bpe, src_bytes);
}

void detile_elements_level(uint8_t* dst, const uint8_t* src, size_t src_bytes,
                           uint32_t ew, uint32_t eh, uint32_t bpe, uint32_t tile_mode,
                           uint32_t tail_x, uint32_t tail_y) {
    tile_census_note("detile_elements_level", ew, eh, bpe, tile_mode);
    const size_t linear_bytes = static_cast<size_t>(ew) * eh * bpe;
    if (!bpe) {
        if (linear_bytes) std::memset(dst, 0, linear_bytes);
        return;
    }
    if (!tile_mode_is_tiled(tile_mode)) {
        const size_t n = std::min(linear_bytes, src_bytes);
        std::memcpy(dst, src, n);
        if (n < linear_bytes) std::memset(dst + n, 0, linear_bytes - n);
        return;
    }
    if (is_64kb_mode(tile_mode)) {
        sw64kb_level_copy<false>(dst, src, src_bytes, ew, eh, bpe, tile_mode,
                                 tail_x, tail_y);
    } else if (tile_mode == (uint32_t)TileMode::Sw256BS) {
        // SW_256B_S mip levels never share a tail. The selected resource base already includes the
        // reverse-chain level offset, so level-local addressing starts at block zero.
        sw256_copy<false>(dst, src, ew, eh, /*pitch*/0, bpe, src_bytes);
    } else {
        sw4kb_level_copy<false>(dst, src, src_bytes, ew, eh, bpe, tail_x, tail_y);
    }
}

void detile_surface_level(uint8_t* dst, const uint8_t* src, size_t src_bytes,
                          uint32_t width, uint32_t height, uint32_t tile_mode,
                          uint32_t bytes_per_texel, uint32_t tail_x, uint32_t tail_y) {
    tile_census_note("detile_surface_level", width, height, bytes_per_texel, tile_mode);
    detile_elements_level(dst, src, src_bytes, width, height, bytes_per_texel,
                          tile_mode, tail_x, tail_y);
}

void tile_surface_level(uint8_t* dst, size_t dst_bytes, const uint8_t* src,
                        uint32_t width, uint32_t height, uint32_t tile_mode,
                        uint32_t bytes_per_texel, uint32_t tail_x, uint32_t tail_y) {
    tile_census_note("tile_surface_level", width, height, bytes_per_texel, tile_mode);
    if (!bytes_per_texel) return;
    const size_t linear_bytes = static_cast<size_t>(width) * height * bytes_per_texel;
    if (!tile_mode_is_tiled(tile_mode)) {
        std::memcpy(dst, src, std::min(linear_bytes, dst_bytes));
        return;
    }
    if (is_64kb_mode(tile_mode)) {
        sw64kb_level_copy<true>(dst, src, dst_bytes, width, height, bytes_per_texel,
                                tile_mode, tail_x, tail_y);
    } else if (tile_mode == (uint32_t)TileMode::Sw256BS) {
        sw256_copy<true>(dst, src, width, height, /*pitch*/0, bytes_per_texel, dst_bytes);
    } else {
        sw4kb_level_copy<true>(dst, src, dst_bytes, width, height, bytes_per_texel,
                               tail_x, tail_y);
    }
}

bool tile_mode_supports_volume(uint32_t tile_mode) {
    return tile_mode == (uint32_t)TileMode::Linear ||
           tile_mode == (uint32_t)TileMode::Sw4KbS ||
           tile_mode == (uint32_t)TileMode::Sw64KbS ||
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
    if (tile_mode == (uint32_t)TileMode::Sw4KbS)
        return sw4kb_s3_volume_bytes(width, height, depth, bytes_per_texel);
    return tile_mode == (uint32_t)TileMode::Sw64KbS
               ? sw64kb_s3_volume_bytes(width, height, depth, bytes_per_texel)
               : sw64kb_rx_volume_bytes(width, height, depth, bytes_per_texel);
}

bool detile_volume(uint8_t* dst, const uint8_t* src, size_t src_bytes,
                   uint32_t width, uint32_t height, uint32_t depth,
                   uint32_t tile_mode, uint32_t bytes_per_texel) {
    // Saturate rather than wrap: `height * depth` is uint32 arithmetic, and a large
    // volume would otherwise alias into another row of the census.
    tile_census_note("detile_volume", width, census_rows(height, depth), bytes_per_texel,
                     tile_mode);
    const size_t linear_bytes = static_cast<size_t>(width) * height * depth * bytes_per_texel;
    if (tile_mode == (uint32_t)TileMode::Linear) {
        if (src_bytes < linear_bytes) return false;
        std::memcpy(dst, src, linear_bytes);
        return true;
    }
    if (!tile_mode_supports_volume(tile_mode)) return false;
    if (tile_mode == (uint32_t)TileMode::Sw4KbS)
        return sw4kb_s3_volume_copy<false>(dst, src, src_bytes, width, height, depth,
                                        bytes_per_texel);
    return tile_mode == (uint32_t)TileMode::Sw64KbS
               ? sw64kb_s3_volume_copy<false>(dst, src, src_bytes, width, height, depth,
                                               bytes_per_texel)
               : sw64kb_rx_volume_copy<false>(dst, src, src_bytes, width, height, depth,
                                               bytes_per_texel);
}

bool tile_volume(uint8_t* dst, size_t dst_bytes, const uint8_t* src,
                 uint32_t width, uint32_t height, uint32_t depth,
                 uint32_t tile_mode, uint32_t bytes_per_texel) {
    // Saturate rather than wrap: `height * depth` is uint32 arithmetic, and a large
    // volume would otherwise alias into another row of the census.
    tile_census_note("tile_volume", width, census_rows(height, depth), bytes_per_texel,
                     tile_mode);
    const size_t need = tiled_volume_bytes(width, height, depth, tile_mode, bytes_per_texel);
    if (!need || dst_bytes < need) return false;
    if (tile_mode == (uint32_t)TileMode::Linear) {
        std::memcpy(dst, src, need);
        return true;
    }
    std::memset(dst, 0, need);
    if (tile_mode == (uint32_t)TileMode::Sw4KbS)
        return sw4kb_s3_volume_copy<true>(dst, src, need, width, height, depth,
                                        bytes_per_texel);
    return tile_mode == (uint32_t)TileMode::Sw64KbS
               ? sw64kb_s3_volume_copy<true>(dst, src, need, width, height, depth,
                                              bytes_per_texel)
               : sw64kb_rx_volume_copy<true>(dst, src, need, width, height, depth,
                                              bytes_per_texel);
}


TileCensusScope::TileCensusScope(const char* who) : prev(g_tile_census_tag) {
    g_tile_census_tag = who;
}
TileCensusScope::~TileCensusScope() { g_tile_census_tag = prev; }

} // namespace prosper::gpu
