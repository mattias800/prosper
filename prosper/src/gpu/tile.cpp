// tile.cpp — see tile.hpp. GFX10 SW_4KB_S de-swizzle, generalized over the element size (#119).
// GFX10 SW_64KB_S / SW_64KB_R_X de-swizzle from the AMD addrlib swizzle-pattern tables (#288).
#include "tile.hpp"
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <limits>
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

#endif

// The 64KB tiled<->linear walk. The pattern offset is XOR-separable in x and y (each offset bit is
// parity(x&xm)^parity(y&ym)), so precompute fx[] / fy[] per coordinate and each element's in-block
// offset is fx[x]^fy[y] — exact per addrlib (patterns are evaluated on GLOBAL element coords; every
// mask stays within the coordinate range of one block for the pipe counts shipped here, but global
// coords keep even wider masks correct).
template <bool ToTiled>
void sw64kb_copy(uint8_t* dst, const uint8_t* src, uint32_t ew, uint32_t eh, uint32_t pitch,
                 uint32_t bpe, size_t tiled_bytes, uint32_t tile_mode,
                 size_t tiled_origin = 0) {
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
        static const bool use_avx2_gather =
            std::getenv("PROSPER_NO_AVX2_DETILE") == nullptr &&
            __builtin_cpu_supports("avx2");
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
                            uint8_t* linear = nullptr;
                            if constexpr (!ToTiled)
                                linear = dst + (static_cast<size_t>(y) * ew + x0) * bpe;
#if defined(PROSPER_HAVE_TARGET_AVX2)
                            if constexpr (!ToTiled) {
                                if (full_block && (bpe == 2 || bpe == 4 || bpe == 8)) {
                                    if (use_avx2_gather) {
                                        detile_full_block_row_avx2(linear, src + block_base,
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
                                                              src + (static_cast<size_t>(y) * ew +
                                                                     x0 + ix) * bpe,
                                                              bytes);
                                } else {
                                    if (full_block ||
                                        (tiled <= tiled_bytes && bytes <= tiled_bytes - tiled))
                                        copy_swizzled_element(
                                            linear + static_cast<size_t>(ix) * bpe,
                                            src + tiled, bytes);
                                    else
                                        zero_swizzled_element(
                                            linear + static_cast<size_t>(ix) * bpe, bytes);
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
    if (tile_mode == (uint32_t)TileMode::Sw256BS)
        return sw256_tiled_bytes(width, height, pitch, bytes_per_texel);
    if (is_64kb_mode(tile_mode)) return sw64kb_tiled_bytes(width, height, pitch, bytes_per_texel);
    return sw4kb_tiled_bytes(width, height, pitch, bytes_per_texel);
}

void detile_surface(uint8_t* dst, const uint8_t* src, uint32_t width, uint32_t height,
                    uint32_t tile_mode, uint32_t pitch, uint32_t bytes_per_texel) {
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
    if (tile_mode == (uint32_t)TileMode::Sw256BS) {
        sw256_copy<true>(dst, src, width, height, pitch, bytes_per_texel, dst_bytes);
        return;
    }
    if (is_64kb_mode(tile_mode)) {
        sw64kb_copy<true>(dst, src, width, height, pitch, bytes_per_texel, dst_bytes, tile_mode);
        return;
    }
    sw4kb_copy<true>(dst, src, width, height, pitch, bytes_per_texel, dst_bytes);
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
        result.tail_x = mip_x * (block_width >> 4u);   // Block256_2d element dimensions
        result.tail_y = mip_y * (block_height >> 4u);
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

size_t tiled_mip_level_offset(uint32_t ew, uint32_t eh, uint32_t bpe, uint32_t tile_mode,
                              uint32_t max_mip, uint32_t mip_level) {
    const TiledMipLevelLayout layout = tiled_mip_level_layout(
        ew, eh, bpe, tile_mode, max_mip, mip_level);
    return layout.supported && !layout.in_tail ? layout.byte_offset : 0;
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
    if (tile_mode == (uint32_t)TileMode::Sw256BS) {
        sw256_copy<false>(dst, src, ew, eh, /*pitch*/0, bpe, src_bytes);
        return;
    }
    sw4kb_copy<false>(dst, src, ew, eh, /*pitch*/0, bpe, src_bytes);
}

void detile_elements_level(uint8_t* dst, const uint8_t* src, size_t src_bytes,
                           uint32_t ew, uint32_t eh, uint32_t bpe, uint32_t tile_mode,
                           uint32_t tail_x, uint32_t tail_y) {
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
    detile_elements_level(dst, src, src_bytes, width, height, bytes_per_texel,
                          tile_mode, tail_x, tail_y);
}

void tile_surface_level(uint8_t* dst, size_t dst_bytes, const uint8_t* src,
                        uint32_t width, uint32_t height, uint32_t tile_mode,
                        uint32_t bytes_per_texel, uint32_t tail_x, uint32_t tail_y) {
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
