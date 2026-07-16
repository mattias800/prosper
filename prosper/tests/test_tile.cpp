// test_tile — the GPU surface de-swizzle (tile.hpp). Verifies the SW_4KB_S detile is the exact inverse of
// the tile (round-trip identity for arbitrary sizes), that linear mode is a passthrough, and that the
// tiled layout is genuinely a permutation (no texel dropped/duplicated).
#include "../src/gpu/tile.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

// A linear image where each texel encodes its (x,y) so any misplacement is detectable.
static std::vector<uint8_t> make_ref(uint32_t w, uint32_t h) {
    std::vector<uint8_t> v((size_t)w * h * 4);
    for (uint32_t y = 0; y < h; y++)
        for (uint32_t x = 0; x < w; x++) {
            uint8_t* p = &v[((size_t)y * w + x) * 4];
            p[0] = (uint8_t)x; p[1] = (uint8_t)(x >> 8);
            p[2] = (uint8_t)y; p[3] = (uint8_t)(y >> 8);
        }
    return v;
}

static bool roundtrip_ok(uint32_t w, uint32_t h) {
    auto ref = make_ref(w, h);
    // The tiled buffer is padded up to whole 32-texel tile rows (e.g. 1080 -> 1088).
    std::vector<uint8_t> tiled(tiled_surface_bytes(w, h, (uint32_t)TileMode::Sw4KbS), 0);
    tile_surface(tiled.data(), ref.data(), w, h, (uint32_t)TileMode::Sw4KbS);
    std::vector<uint8_t> back((size_t)w * h * 4, 0);
    detile_surface(back.data(), tiled.data(), w, h, (uint32_t)TileMode::Sw4KbS);
    return back == ref;
}

// Independent, deliberately slow SW_4KB_S oracle. These bit orders are the low-bit GFX10 addrlib
// patterns documented by the hardware-derived golden tests below. This does not share production's
// lookup table or indexing code, so it catches a fast-path error that a tile/detile round trip cannot.
struct RefOrder {
    const uint8_t* bits;
    uint32_t count, tile_w, tile_h;
};

static RefOrder reference_order(uint32_t bpe) {
    // 0..6 = x bit; 0x80..0x86 = y bit.
    static const uint8_t b1[]  = {0,1,2,3, 0x80,0x81,0x82,0x83,0x84, 4,0x85,5};
    static const uint8_t b2[]  = {0,1,2, 0x80,0x81,0x82, 3,0x83,4,0x84,5};
    static const uint8_t b4[]  = {0,1, 0x80,0x81,0x82, 2,0x83,3,0x84,4};
    static const uint8_t b8[]  = {0,0x80,0x81,1,2,0x82,3,0x83,4};
    static const uint8_t b16[] = {0x80,0x81,0,1,0x82,2,0x83,3};
    switch (bpe) {
        case 1:  return {b1,  12, 64, 64};
        case 2:  return {b2,  11, 64, 32};
        case 4:  return {b4,  10, 32, 32};
        case 8:  return {b8,   9, 32, 16};
        case 16: return {b16,  8, 16, 16};
        default: return {nullptr, 0, 0, 0};
    }
}

static uint32_t reference_element_index(uint32_t x, uint32_t y, const RefOrder& order) {
    uint32_t index = 0;
    for (uint32_t out_bit = 0; out_bit < order.count; ++out_bit) {
        const uint8_t source = order.bits[out_bit];
        const uint32_t bit = source & 0x7fu;
        const uint32_t value = source & 0x80u ? ((y >> bit) & 1u) : ((x >> bit) & 1u);
        index |= value << out_bit;
    }
    return index;
}

template <bool ToTiled>
static void reference_sw4kb_copy(uint8_t* dst, const uint8_t* src, uint32_t w, uint32_t h,
                                 uint32_t pitch, uint32_t bpe, size_t tiled_bytes) {
    const RefOrder order = reference_order(bpe);
    const uint32_t tiles_per_row = ((pitch ? pitch : w) + order.tile_w - 1) / order.tile_w;
    for (uint32_t y = 0; y < h; ++y) for (uint32_t x = 0; x < w; ++x) {
        const uint32_t tx = x / order.tile_w, ty = y / order.tile_h;
        const uint32_t ix = x % order.tile_w, iy = y % order.tile_h;
        const size_t tiled = static_cast<size_t>(ty * tiles_per_row + tx) * 4096 +
                             reference_element_index(ix, iy, order) * bpe;
        const size_t linear = (static_cast<size_t>(y) * w + x) * bpe;
        if (ToTiled) {
            if (tiled + bpe <= tiled_bytes) std::memcpy(dst + tiled, src + linear, bpe);
        } else if (tiled + bpe <= tiled_bytes) {
            std::memcpy(dst + linear, src + tiled, bpe);
        } else {
            std::memset(dst + linear, 0, bpe);
        }
    }
}

int main() {
    printf("== test_tile ==\n");

    CHECK(tile_mode_is_tiled((uint32_t)TileMode::Sw4KbS), "tile_mode 5 (SW_4KB_S) is tiled");
    CHECK(!tile_mode_is_tiled((uint32_t)TileMode::Linear), "tile_mode 0 (linear) is not tiled");

    // Linear mode: detile is a straight copy.
    {
        auto ref = make_ref(37, 19);
        auto out = detile_surface(ref, 37, 19, (uint32_t)TileMode::Linear);
        CHECK(out == ref, "linear detile is a passthrough copy");
    }

    // Round-trip identity: detile(tile(x)) == x, across tile-aligned and non-aligned sizes.
    CHECK(roundtrip_ok(32, 32),     "round-trip 32x32 (one tile)");
    CHECK(roundtrip_ok(64, 64),     "round-trip 64x64");
    CHECK(roundtrip_ok(1920, 1080), "round-trip 1920x1080 (the game's render target)");
    CHECK(roundtrip_ok(96, 64),     "round-trip 96x64 (non-square)");

    // The vector convenience overload must be safe for a NATURALLY-sized (width*height*4, unpadded)
    // tiled input: it zero-pads internally instead of reading past the vector (heap OOB). The visible
    // texels must still match; only the bottom (padded) tile-row region may detile as zero.
    {
        const uint32_t w = 1920, h = 1080;
        auto ref = make_ref(w, h);
        std::vector<uint8_t> tiled(tiled_surface_bytes(w, h, (uint32_t)TileMode::Sw4KbS), 0);
        tile_surface(tiled.data(), ref.data(), w, h, (uint32_t)TileMode::Sw4KbS);
        std::vector<uint8_t> truncated(tiled.begin(), tiled.begin() + (size_t)w * h * 4);
        auto full = detile_surface(tiled, w, h, (uint32_t)TileMode::Sw4KbS);
        auto part = detile_surface(truncated, w, h, (uint32_t)TileMode::Sw4KbS);
        CHECK(full == ref, "vector overload detiles a fully-padded input to the reference");
        bool top_matches = std::equal(part.begin(), part.begin() + (size_t)w * 1024 * 4, ref.begin());
        CHECK(top_matches, "vector overload on an UNPADDED input: rows above the last tile-row intact");
    }

    // The tiled layout must be a permutation: every source index used exactly once (no drops/dupes) for a
    // tile-aligned surface.
    {
        const uint32_t w = 64, h = 32, n = w * h;
        auto ref = make_ref(w, h);
        std::vector<uint8_t> tiled((size_t)n * 4, 0);
        tile_surface(tiled.data(), ref.data(), w, h, (uint32_t)TileMode::Sw4KbS);
        // Decode each tiled texel's (x,y) and mark it seen.
        std::vector<int> seen(n, 0);
        bool inrange = true;
        for (uint32_t i = 0; i < n; i++) {
            const uint8_t* p = &tiled[(size_t)i * 4];
            uint32_t x = p[0] | (p[1] << 8), y = p[2] | (p[3] << 8);
            if (x >= w || y >= h) { inrange = false; break; }
            seen[y * w + x]++;
        }
        bool perm = inrange;
        for (uint32_t i = 0; i < n && perm; i++) perm = (seen[i] == 1);
        CHECK(perm, "SW_4KB_S tiling is a bijective permutation (no texel dropped or duplicated)");
    }

    // --- Per-bpp geometry (#119): a 4KB tile is a FIXED 4096 bytes, so its texel dims depend on
    // bytes_per_texel (1 B -> 64x64, 2 B -> 64x32, 4 B -> 32x32). The old code hardcoded 4 B.
    {
        const uint32_t M = (uint32_t)TileMode::Sw4KbS;
        CHECK(tiled_surface_bytes(1920, 1080, M) == (size_t)60 * 34 * 4096,
              "32-bpp tile geometry unchanged (1920x1080 -> 60x34 tiles)");
        CHECK(tiled_surface_bytes(100, 100, M, 0, 1) == (size_t)2 * 2 * 4096,
              "8-bpp: 100x100 -> 2x2 tiles of 64x64 (NOT the 32-bpp 4x4)");
        CHECK(tiled_surface_bytes(100, 100, M, 0, 2) == (size_t)2 * 4 * 4096,
              "16-bpp: 100x100 -> 2x4 tiles of 64x32");
        CHECK(tiled_elements_bytes(20, 20, 16, M) == (size_t)2 * 2 * 4096,
              "16-byte elements (BC3 blocks): 20x20 -> 2x2 tiles of 16x16 (unchanged)");
        CHECK(tiled_elements_bytes(40, 20, 8, M) == (size_t)2 * 2 * 4096,
              "8-byte elements (BC1 blocks): 40x20 -> 2x2 tiles of 32x16 (was square-approximated)");
    }

    // Round-trip identity per element size (the shared indexer makes tile/detile exact inverses;
    // this proves the per-bpp geometry is self-consistent incl. non-square 64x32 / 32x16 tiles).
    {
        const uint32_t M = (uint32_t)TileMode::Sw4KbS;
        auto rt_bpe = [&](uint32_t w, uint32_t h, uint32_t bpe) -> bool {
            std::vector<uint8_t> ref((size_t)w * h * bpe);
            for (size_t i = 0; i < ref.size(); i++) ref[i] = (uint8_t)((i * 2654435761u) >> 13);
            std::vector<uint8_t> tiled(tiled_surface_bytes(w, h, M, 0, bpe), 0);
            tile_surface(tiled.data(), ref.data(), w, h, M, 0, bpe);
            std::vector<uint8_t> back((size_t)w * h * bpe, 0);
            detile_surface(back.data(), tiled.data(), w, h, M, 0, bpe);
            return back == ref;
        };
        CHECK(rt_bpe(200, 130, 1), "round-trip 200x130 @ 1 B/texel (64x64 tiles, unaligned dims)");
        CHECK(rt_bpe(130, 70, 2),  "round-trip 130x70 @ 2 B/texel (64x32 tiles)");
        CHECK(rt_bpe(96, 64, 4),   "round-trip 96x64 @ 4 B/texel (explicit-bpt path == default)");
        CHECK(rt_bpe(70, 40, 8),   "round-trip 70x40 @ 8 B/elem (32x16 tiles)");
        CHECK(rt_bpe(40, 40, 16),  "round-trip 40x40 @ 16 B/elem (16x16 tiles)");
    }

    // Compare the optimized tile-oriented implementation directly with the independent per-element
    // oracle. Cover every element size, partial edge tiles, padded pitches, and bounded/truncated reads.
    {
        const uint32_t M = (uint32_t)TileMode::Sw4KbS;
        struct Case { uint32_t w, h, pitch, bpe; };
        const Case cases[] = {
            {131, 70, 193, 1}, {131, 70, 191, 2}, {67, 45, 96, 4},
            {70, 41, 97, 8}, {41, 39, 55, 16},
        };
        bool tile_matches = true, detile_matches = true, truncated_matches = true;
        for (const Case& c : cases) {
            std::vector<uint8_t> linear((size_t)c.w * c.h * c.bpe);
            for (size_t i = 0; i < linear.size(); ++i)
                linear[i] = (uint8_t)((i * 11400714819323198485ull) >> 37);
            const size_t bytes = tiled_surface_bytes(c.w, c.h, M, c.pitch, c.bpe);
            std::vector<uint8_t> expected(bytes, 0), actual(bytes, 0);
            reference_sw4kb_copy<true>(expected.data(), linear.data(), c.w, c.h,
                                       c.pitch, c.bpe, bytes);
            tile_surface(actual.data(), linear.data(), c.w, c.h, M, c.pitch, c.bpe);
            tile_matches &= actual == expected;

            std::vector<uint8_t> expected_linear(linear.size(), 0), actual_linear(linear.size(), 0);
            reference_sw4kb_copy<false>(expected_linear.data(), expected.data(), c.w, c.h,
                                        c.pitch, c.bpe, bytes);
            detile_surface(actual_linear.data(), expected.data(), c.w, c.h, M, c.pitch, c.bpe);
            detile_matches &= actual_linear == expected_linear;

            // detile_elements exposes an explicit source bound. Use pitch=width, as that API does.
            const size_t unpitched_bytes = tiled_elements_bytes(c.w, c.h, c.bpe, M);
            std::vector<uint8_t> unpitched(unpitched_bytes, 0);
            reference_sw4kb_copy<true>(unpitched.data(), linear.data(), c.w, c.h,
                                       0, c.bpe, unpitched_bytes);
            const size_t available = unpitched_bytes > 123 ? unpitched_bytes - 123 : unpitched_bytes / 2;
            std::fill(expected_linear.begin(), expected_linear.end(), 0);
            std::fill(actual_linear.begin(), actual_linear.end(), 0);
            reference_sw4kb_copy<false>(expected_linear.data(), unpitched.data(), c.w, c.h,
                                        0, c.bpe, available);
            detile_elements(actual_linear.data(), unpitched.data(), available,
                            c.w, c.h, c.bpe, M);
            truncated_matches &= actual_linear == expected_linear;
        }
        CHECK(tile_matches, "optimized SW_4KB_S tile matches independent addrlib-order oracle");
        CHECK(detile_matches, "optimized SW_4KB_S detile matches oracle for padded pitches/edge tiles");
        CHECK(truncated_matches, "optimized bounded detile matches oracle for truncated tiled buffers");
    }

    // Golden positions for the GFX10 SW_4KB_S element order (#118): the lowest four element bits are a
    // 4x4 sub-block [x0, x1, y0, y1], so (1,0) -> element 1 (x0), (2,0) -> element 2 (x1), (0,1) ->
    // element 4 (y0), (0,2) -> element 8 (y1). The previous order asserted (0,1)->1 / (1,0)->2 (plain
    // y-low Morton), which serrated every edge of every tiled texture (pixel-verified WRONG vs The
    // Messenger's title art). The 8-bpp cross-tile step is one whole 4KB tile.
    {
        const uint32_t M = (uint32_t)TileMode::Sw4KbS;
        std::vector<uint8_t> lin32((size_t)64 * 32 * 4, 0);
        for (uint32_t y = 0; y < 32; y++) for (uint32_t x = 0; x < 64; x++)
            lin32[((size_t)y * 64 + x) * 4] = (uint8_t)(x ^ (y << 4) ^ 0x5A);
        std::vector<uint8_t> t32(tiled_surface_bytes(64, 32, M), 0);
        tile_surface(t32.data(), lin32.data(), 64, 32, M);
        CHECK(t32[1 * 4] == lin32[((size_t)0 * 64 + 1) * 4], "32-bpp golden: texel (1,0) at element 1 (x0 low)");
        CHECK(t32[2 * 4] == lin32[((size_t)0 * 64 + 2) * 4], "32-bpp golden: texel (2,0) at element 2 (x1)");
        CHECK(t32[4 * 4] == lin32[((size_t)1 * 64 + 0) * 4], "32-bpp golden: texel (0,1) at element 4 (y0 at bit 2)");
        CHECK(t32[8 * 4] == lin32[((size_t)2 * 64 + 0) * 4], "32-bpp golden: texel (0,2) at element 8 (y1 at bit 3)");
        CHECK(t32[4096]  == lin32[((size_t)0 * 64 + 32) * 4], "32-bpp golden: texel (32,0) starts tile 1 (byte 4096)");

        // 8-bpp uses a DIFFERENT within-tile order than 32-bpp (the 64x64 tile's SW_4KB_S order is
        // pixel-verified against the game's 2048x1024 R8 caption-font atlas): the low EIGHT element bits
        // are a 16x16 sub-block [x0,x1,x2,x3, y0,y1,y2,y3] (four low X bits, then four low Y bits), then
        // (y,x) Morton pairs [y4,x4, y5,x5]. So (1,0)->1 (x0), (4,0)->4 (x2), (8,0)->8 (x3),
        // (0,1)->16 (y0 at bit 4), (0,2)->32 (y1), (16,0)->512 (x4 at bit 9). The OLD code shared the
        // 32-bpp [x0,x1,y0,y1] low nibble here, which scrambled every 64x64 tile into a weave.
        std::vector<uint8_t> lin8((size_t)128 * 64, 0);
        for (size_t i = 0; i < lin8.size(); i++) lin8[i] = (uint8_t)(i * 31 + 7);
        std::vector<uint8_t> t8(tiled_surface_bytes(128, 64, M, 0, 1), 0);
        tile_surface(t8.data(), lin8.data(), 128, 64, M, 0, 1);
        CHECK(t8[1]    == lin8[(size_t)0 * 128 + 1],  "8-bpp golden: texel (1,0) at element 1 (x0)");
        CHECK(t8[4]    == lin8[(size_t)0 * 128 + 4],  "8-bpp golden: texel (4,0) at element 4 (x2)");
        CHECK(t8[8]    == lin8[(size_t)0 * 128 + 8],  "8-bpp golden: texel (8,0) at element 8 (x3)");
        CHECK(t8[16]   == lin8[(size_t)1 * 128 + 0],  "8-bpp golden: texel (0,1) at element 16 (y0 at bit 4)");
        CHECK(t8[32]   == lin8[(size_t)2 * 128 + 0],  "8-bpp golden: texel (0,2) at element 32 (y1 at bit 5)");
        CHECK(t8[512]  == lin8[(size_t)0 * 128 + 16], "8-bpp golden: texel (16,0) at element 512 (x4 at bit 9)");
        CHECK(t8[4096] == lin8[(size_t)0 * 128 + 64], "8-bpp golden: texel (64,0) starts tile 1 (byte 4096, 64-wide tile)");
    }

    // #379: golden within-tile order for the remaining element sizes 2/8/16 B — derived from the low
    // (bx+by) bits of the authoritative kSw64kS pattern (docs/GFX10_SW_64KB_TILING.md: the 4KB order is
    // the low-bit truncation of the 64KB order). The old L-generator was correct only at 1 B / 4 B; at
    // 2/8/16 B it swapped the low X/Y pairs, scrambling every SW_4KB_S R16 / BC1/BC4 / BC2/3/5/6/7
    // surface into a coherent-looking weave. Each element stores x in byte0 and y in byte1, so a checked
    // element uniquely identifies its source texel and a mis-order cannot pass on a coincidental byte.
    {
        const uint32_t M = (uint32_t)TileMode::Sw4KbS;
        auto fill = [](std::vector<uint8_t>& lin, uint32_t w, uint32_t h, uint32_t bpt) {
            for (uint32_t y = 0; y < h; y++) for (uint32_t x = 0; x < w; x++) {
                lin[((size_t)y * w + x) * bpt + 0] = (uint8_t)x;
                lin[((size_t)y * w + x) * bpt + 1] = (uint8_t)y;
            }
        };
        // The check: element `idx` (its byte0/byte1) equals source texel (x,y)'s byte0/byte1.
        #define GOLD(t, lin, w, bpt, idx, tx, ty, msg) \
            CHECK((t)[(size_t)(idx)*(bpt)+0] == (lin)[((size_t)(ty)*(w)+(tx))*(bpt)+0] && \
                  (t)[(size_t)(idx)*(bpt)+1] == (lin)[((size_t)(ty)*(w)+(tx))*(bpt)+1], msg)

        // 2 B/element (64x32 tile): order x0 x1 x2 y0 y1 y2 x3 y3 x4 y4 x5.
        std::vector<uint8_t> lin2((size_t)128 * 64 * 2, 0); fill(lin2, 128, 64, 2);
        std::vector<uint8_t> t2(tiled_surface_bytes(128, 64, M, 0, 2), 0);
        tile_surface(t2.data(), lin2.data(), 128, 64, M, 0, 2);
        GOLD(t2, lin2, 128, 2, 1,  1, 0, "2 B golden: texel (1,0) -> element 1 (x0 at bit 0)");
        GOLD(t2, lin2, 128, 2, 2,  2, 0, "2 B golden: texel (2,0) -> element 2 (x1 at bit 1)");
        GOLD(t2, lin2, 128, 2, 4,  4, 0, "2 B golden: texel (4,0) -> element 4 (x2 at bit 2)");
        GOLD(t2, lin2, 128, 2, 8,  0, 1, "2 B golden: texel (0,1) -> element 8 (y0 at bit 3)");
        GOLD(t2, lin2, 128, 2, 16, 0, 2, "2 B golden: texel (0,2) -> element 16 (y1 at bit 4)");

        // 8 B/element (32x16 tile, BC1/BC4 blocks): order x0 y0 y1 x1 x2 y2 x3 y3 x4.
        std::vector<uint8_t> lin8b((size_t)64 * 32 * 8, 0); fill(lin8b, 64, 32, 8);
        std::vector<uint8_t> t8b(tiled_surface_bytes(64, 32, M, 0, 8), 0);
        tile_surface(t8b.data(), lin8b.data(), 64, 32, M, 0, 8);
        GOLD(t8b, lin8b, 64, 8, 1,  1, 0, "8 B golden: texel (1,0) -> element 1 (x0 at bit 0)");
        GOLD(t8b, lin8b, 64, 8, 2,  0, 1, "8 B golden: texel (0,1) -> element 2 (y0 at bit 1)");
        GOLD(t8b, lin8b, 64, 8, 4,  0, 2, "8 B golden: texel (0,2) -> element 4 (y1 at bit 2)");
        GOLD(t8b, lin8b, 64, 8, 8,  2, 0, "8 B golden: texel (2,0) -> element 8 (x1 at bit 3)");
        GOLD(t8b, lin8b, 64, 8, 16, 4, 0, "8 B golden: texel (4,0) -> element 16 (x2 at bit 4)");

        // 16 B/element (16x16 tile, BC2/3/5/6/7 blocks): order y0 y1 x0 x1 y2 x2 y3 x3 — Y pair FIRST.
        // The old code emitted the X pair first, swapping every non-(0,0) block in the micro-tile.
        std::vector<uint8_t> lin16((size_t)32 * 32 * 16, 0); fill(lin16, 32, 32, 16);
        std::vector<uint8_t> t16(tiled_surface_bytes(32, 32, M, 0, 16), 0);
        tile_surface(t16.data(), lin16.data(), 32, 32, M, 0, 16);
        GOLD(t16, lin16, 32, 16, 1,  0, 1, "16 B golden: texel (0,1) -> element 1 (y0 at bit 0)");
        GOLD(t16, lin16, 32, 16, 2,  0, 2, "16 B golden: texel (0,2) -> element 2 (y1 at bit 1)");
        GOLD(t16, lin16, 32, 16, 4,  1, 0, "16 B golden: texel (1,0) -> element 4 (x0 at bit 2)");
        GOLD(t16, lin16, 32, 16, 8,  2, 0, "16 B golden: texel (2,0) -> element 8 (x1 at bit 3)");
        GOLD(t16, lin16, 32, 16, 16, 0, 4, "16 B golden: texel (0,4) -> element 16 (y2 at bit 4)");
        #undef GOLD
    }

    // --- GFX10 64KB modes (#288): SW_64KB_S (tile_mode 9) and SW_64KB_R_X (tile_mode 27). ---
    CHECK(tile_mode_is_tiled((uint32_t)TileMode::Sw64KbS),  "tile_mode 9 (SW_64KB_S) is tiled");
    CHECK(tile_mode_is_tiled((uint32_t)TileMode::Sw64KbRX), "tile_mode 27 (SW_64KB_R_X) is tiled");

    // Tiled footprint: a 64KB block holds 65536 bytes; its element dims depend on bpe
    // (1 B -> 256x256, 2 B -> 256x128, 4 B -> 128x128, 8 B -> 128x64, 16 B -> 64x64).
    {
        const uint32_t M = (uint32_t)TileMode::Sw64KbS;
        CHECK(tiled_elements_bytes(128, 64, 8, M) == 65536,
              "8-byte elements (BC1 blocks): 128x64 (a 512x256 BC1 texture) = exactly one 64KB block");
        CHECK(tiled_surface_bytes(3840, 2160, M, 0, 4) == (size_t)30 * 17 * 65536,
              "32-bpp: 3840x2160 -> 30x17 blocks of 128x128");
        CHECK(tiled_elements_bytes(256, 128, 16, M) == (size_t)4 * 2 * 65536,
              "16-byte elements (BC7 blocks): 256x128 -> 4x2 blocks of 64x64");
    }

    // Round-trip identity for both 64KB modes at every element size (incl. block-unaligned dims).
    {
        auto rt64 = [&](uint32_t M, uint32_t w, uint32_t h, uint32_t bpe) -> bool {
            std::vector<uint8_t> ref((size_t)w * h * bpe);
            for (size_t i = 0; i < ref.size(); i++) ref[i] = (uint8_t)((i * 2654435761u) >> 11);
            std::vector<uint8_t> tiled(tiled_surface_bytes(w, h, M, 0, bpe), 0);
            tile_surface(tiled.data(), ref.data(), w, h, M, 0, bpe);
            std::vector<uint8_t> back((size_t)w * h * bpe, 0);
            detile_surface(back.data(), tiled.data(), w, h, M, 0, bpe);
            return back == ref;
        };
        const uint32_t S = (uint32_t)TileMode::Sw64KbS, R = (uint32_t)TileMode::Sw64KbRX;
        CHECK(rt64(S, 256, 256, 1),  "SW_64KB_S round-trip 256x256 @ 1 B (one block)");
        CHECK(rt64(S, 300, 140, 2),  "SW_64KB_S round-trip 300x140 @ 2 B (unaligned)");
        CHECK(rt64(S, 500, 300, 4),  "SW_64KB_S round-trip 500x300 @ 4 B (unaligned)");
        CHECK(rt64(S, 128, 64, 8),   "SW_64KB_S round-trip 128x64 @ 8 B (one block, BC1 grid)");
        CHECK(rt64(S, 130, 70, 16),  "SW_64KB_S round-trip 130x70 @ 16 B (unaligned BC7 grid)");
        CHECK(rt64(R, 260, 130, 1),  "SW_64KB_R_X round-trip 260x130 @ 1 B");
        CHECK(rt64(R, 300, 140, 2),  "SW_64KB_R_X round-trip 300x140 @ 2 B");
        CHECK(rt64(R, 500, 300, 4),  "SW_64KB_R_X round-trip 500x300 @ 4 B");
        CHECK(rt64(R, 130, 70, 8),   "SW_64KB_R_X round-trip 130x70 @ 8 B");
        CHECK(rt64(R, 70, 70, 16),   "SW_64KB_R_X round-trip 70x70 @ 16 B");
    }

    // Golden positions from the addrlib GFX10_SW_64K_S pattern (gfx10SwizzlePattern.h). At 8 bpe the
    // 16 offset bits are [x0 | y0 y1 x1 x2 | y2 x3 y3 x4 | y4 x5 y5 x6] above the 3 byte bits, i.e.
    // element (1,0) -> byte 8, (0,1) -> 16, (0,2) -> 32, (2,0) -> 64... — the SW_4KB_S order continued
    // with (y,x) Morton pairs. The last element of an aligned 128x64 grid lands at the block's last
    // 8 bytes (the pattern is a bijection over the block).
    {
        const uint32_t M = (uint32_t)TileMode::Sw64KbS;
        const uint32_t ew = 128, eh = 64, bpe = 8;
        std::vector<uint8_t> ref((size_t)ew * eh * bpe);
        for (size_t i = 0; i < ref.size(); i++) ref[i] = (uint8_t)((i >> 3) * 73 + i);
        std::vector<uint8_t> tiled(65536, 0);
        tile_surface(tiled.data(), ref.data(), ew, eh, M, 0, bpe);
        auto at = [&](uint32_t x, uint32_t y) { return &ref[((size_t)y * ew + x) * bpe]; };
        CHECK(std::memcmp(&tiled[8],     at(1, 0), bpe) == 0, "64KB_S 8B golden: element (1,0) at byte 8 (x0 -> bit 3)");
        CHECK(std::memcmp(&tiled[16],    at(0, 1), bpe) == 0, "64KB_S 8B golden: element (0,1) at byte 16 (y0 -> bit 4)");
        CHECK(std::memcmp(&tiled[32],    at(0, 2), bpe) == 0, "64KB_S 8B golden: element (0,2) at byte 32 (y1 -> bit 5)");
        CHECK(std::memcmp(&tiled[64],    at(2, 0), bpe) == 0, "64KB_S 8B golden: element (2,0) at byte 64 (x1 -> bit 6)");
        CHECK(std::memcmp(&tiled[256],   at(0, 4), bpe) == 0, "64KB_S 8B golden: element (0,4) at byte 256 (y2 -> bit 8)");
        CHECK(std::memcmp(&tiled[512],   at(8, 0), bpe) == 0, "64KB_S 8B golden: element (8,0) at byte 512 (x3 -> bit 9)");
        CHECK(std::memcmp(&tiled[32768], at(64, 0), bpe) == 0, "64KB_S 8B golden: element (64,0) at byte 32768 (x6 -> bit 15)");
        CHECK(std::memcmp(&tiled[65536 - 8], at(127, 63), bpe) == 0, "64KB_S 8B golden: element (127,63) is the block's last 8 bytes");
    }

    // Golden positions for the GFX10 SW_64K_S pattern at 16 bpe (BC7/BC6/BC5 blocks) — the element
    // size DOLL's loading-banner artwork uses (2688x448 / 1344x672 / 3400x1400 BC7 atlases, #304). The
    // 16 offset bits above the 4 byte bits are [y0 y1 x0 x1 | y2 x2 y3 x3 | y4 x4 y5 x5] (kSw64kS[4]),
    // i.e. element (0,1) -> byte 16 (y0 -> bit 4), (1,0) -> byte 64 (x0 -> bit 6), (2,0) -> byte 128,
    // (0,2) -> byte 32, (0,4) -> byte 256, (4,0) -> byte 512, and (63,63) -> the block's last 16 bytes.
    // A 64KB block at 16 bpe is 64x64 elements (a 256x256-texel BC7 region). This pins the 16-B within-
    // block order that the offline #304 decode validated (the 2688x448 BC7 detiled coherently).
    {
        const uint32_t M = (uint32_t)TileMode::Sw64KbS;
        const uint32_t ew = 64, eh = 64, bpe = 16;
        std::vector<uint8_t> ref((size_t)ew * eh * bpe);
        for (size_t i = 0; i < ref.size(); i++) ref[i] = (uint8_t)((i >> 4) * 97 + i);
        std::vector<uint8_t> tiled(65536, 0);
        tile_surface(tiled.data(), ref.data(), ew, eh, M, 0, bpe);
        auto at = [&](uint32_t x, uint32_t y) { return &ref[((size_t)y * ew + x) * bpe]; };
        CHECK(std::memcmp(&tiled[16],    at(0, 1),  bpe) == 0, "64KB_S 16B golden: element (0,1) at byte 16 (y0 -> bit 4)");
        CHECK(std::memcmp(&tiled[32],    at(0, 2),  bpe) == 0, "64KB_S 16B golden: element (0,2) at byte 32 (y1 -> bit 5)");
        CHECK(std::memcmp(&tiled[64],    at(1, 0),  bpe) == 0, "64KB_S 16B golden: element (1,0) at byte 64 (x0 -> bit 6)");
        CHECK(std::memcmp(&tiled[128],   at(2, 0),  bpe) == 0, "64KB_S 16B golden: element (2,0) at byte 128 (x1 -> bit 7)");
        CHECK(std::memcmp(&tiled[256],   at(0, 4),  bpe) == 0, "64KB_S 16B golden: element (0,4) at byte 256 (y2 -> bit 8)");
        CHECK(std::memcmp(&tiled[512],   at(4, 0),  bpe) == 0, "64KB_S 16B golden: element (4,0) at byte 512 (x2 -> bit 9)");
        CHECK(std::memcmp(&tiled[65536 - 16], at(63, 63), bpe) == 0, "64KB_S 16B golden: element (63,63) is the block's last 16 bytes");
    }

    // Block-unaligned MULTI-BLOCK-ROW addressing at 16 bpe — the exact case issue #304 flagged as
    // untested: a 2688-wide BC7 is 672 blocks wide, and 672 is NOT a multiple of the 64-block macro
    // width, so blocks_per_row = ceil(672/64) = 11 (a padded pitch). A wrong blocks_per_row (using the
    // unpadded 672/64 = 10, or the byte pitch) shifts every block row by one macro block and produces
    // the 1-px column stripes reported in #304. Here a 140-wide (block-unaligned: ceil(140/64) = 3),
    // 70-tall (2 block rows) grid pins that the first element of block column 1 / block row 1 land at
    // exactly one / blocks_per_row 64KB blocks. Offline this addressing decoded the banner coherently;
    // the stripes were NOT here (they are a degenerate-vertex geometry bug, tracked under #273).
    {
        const uint32_t M = (uint32_t)TileMode::Sw64KbS;
        const uint32_t ew = 140, eh = 70, bpe = 16;      // blocks_per_row=ceil(140/64)=3, block_rows=2
        std::vector<uint8_t> ref((size_t)ew * eh * bpe);
        for (size_t i = 0; i < ref.size(); i++) ref[i] = (uint8_t)((i >> 4) * 131 + i);
        const size_t tbytes = tiled_elements_bytes(ew, eh, bpe, M);
        CHECK(tbytes == (size_t)3 * 2 * 65536, "64KB_S 16B: 140x70 -> 3x2 blocks (blocks_per_row padded to 3)");
        std::vector<uint8_t> tiled(tbytes, 0);
        tile_surface(tiled.data(), ref.data(), ew, eh, M, 0, bpe);
        auto at = [&](uint32_t x, uint32_t y) { return &ref[((size_t)y * ew + x) * bpe]; };
        CHECK(std::memcmp(&tiled[65536],           at(64, 0), bpe) == 0, "64KB_S 16B multiblock: element (64,0) starts block 1 (byte 65536)");
        CHECK(std::memcmp(&tiled[(size_t)2 * 65536], at(128, 0), bpe) == 0, "64KB_S 16B multiblock: element (128,0) starts block 2 (byte 131072)");
        CHECK(std::memcmp(&tiled[(size_t)3 * 65536], at(0, 64), bpe) == 0, "64KB_S 16B multiblock: element (0,64) starts block row 1 (block 3 = blocks_per_row)");
        CHECK(std::memcmp(&tiled[(size_t)3 * 65536 + 64], at(1, 64), bpe) == 0, "64KB_S 16B multiblock: element (1,64) at block 3 (block row 1) + within-block (1,0)=byte 64");
        // Full round-trip over the block-unaligned surface (no texel dropped/duplicated across blocks).
        std::vector<uint8_t> back((size_t)ew * eh * bpe, 0);
        detile_elements(back.data(), tiled.data(), tbytes, ew, eh, bpe, M);
        CHECK(back == ref, "64KB_S 16B multiblock: detile is the exact inverse over 140x70 (block-unaligned)");
    }

    // SW_64KB_R_X pipe-XOR golden (default 16 pipes, 4 bpe): offset bits 8..11 are the pipe bits
    // x3^y3, x4^y4, x6^y5, x5^y6, then bits 12..15 are y3 x4 y6 x6 — so element (8,0) sets only
    // bit 8 (byte 256), (0,8) sets bits 8+12 (byte 4352), (8,8) cancels the bit-8 XOR leaving bit
    // 12 (byte 4096), and (16,0) sets bits 9+13 (byte 8704). This pins the pipe-rotation wiring.
    if (!getenv("PROSPER_RX_PIPES")) {
        const uint32_t M = (uint32_t)TileMode::Sw64KbRX;
        const uint32_t ew = 128, eh = 128, bpe = 4;
        std::vector<uint8_t> ref((size_t)ew * eh * bpe);
        for (size_t i = 0; i < ref.size(); i++) ref[i] = (uint8_t)((i >> 2) * 151 + i);
        std::vector<uint8_t> tiled(65536, 0);
        tile_surface(tiled.data(), ref.data(), ew, eh, M, 0, bpe);
        auto at = [&](uint32_t x, uint32_t y) { return &ref[((size_t)y * ew + x) * bpe]; };
        CHECK(std::memcmp(&tiled[256],  at(8, 0),  bpe) == 0, "64KB_R_X 4B golden: element (8,0) at byte 256 (x3 -> bit 8)");
        CHECK(std::memcmp(&tiled[4352], at(0, 8),  bpe) == 0, "64KB_R_X 4B golden: element (0,8) at byte 4352 (y3 -> bits 8+12)");
        CHECK(std::memcmp(&tiled[4096], at(8, 8),  bpe) == 0, "64KB_R_X 4B golden: element (8,8) at byte 4096 (pipe XOR cancels)");
        CHECK(std::memcmp(&tiled[8704], at(16, 0), bpe) == 0, "64KB_R_X 4B golden: element (16,0) at byte 8704 (x4 -> bits 9+13)");
    }

    // GFX10 thin/view-as-2D 3D SW_64KB_R_X. Each Z slice owns a padded 2D block grid, while Z
    // participates in the pipe-XOR inside that slice; z0..z3 map to offset bits 11..8.
    if (!getenv("PROSPER_RX_PIPES")) {
        const uint32_t M = (uint32_t)TileMode::Sw64KbRX;
        CHECK(tile_mode_supports_volume(M), "SW_64KB_R_X has an explicit 3D pipe-XOR pattern");
        CHECK(gfx10_dcc_metadata_bytes(32, 32, 32, M, 4, true) == 128 * 1024,
              "32x32x32 RGBA8 R_X DCC occupies one 4 KiB metadata block per slice");
        CHECK(gfx10_dcc_metadata_bytes(513, 512, 1, M, 4, true) == 8 * 1024,
              "R_X DCC sizing rounds a 4-B image to 512x512 metadata-block coverage");
        CHECK(gfx10_dcc_metadata_bytes(32, 32, 32, (uint32_t)TileMode::Sw64KbS,
                                       4, true) == 0,
              "DCC sizing rejects a swizzle outside the supported R_X contract");
        std::vector<uint8_t> clear_pixels(8, 0xaa);
        const std::vector<uint8_t> clear_0001(4096, 0x40);
        uint8_t clear_code = 0xff;
        CHECK(gfx10_dcc_fast_clear_rgba8(clear_pixels.data(), 2,
                                         clear_0001.data(), clear_0001.size(),
                                         4, true, &clear_code) && clear_code == 0x40 &&
              clear_pixels == std::vector<uint8_t>({0, 0, 0, 255, 0, 0, 0, 255}),
              "uniform DCC_CLEAR_0001 materializes color zero and MSB alpha one");
        const std::vector<uint8_t> clear_0000(16, 0x00);
        const std::vector<uint8_t> clear_1111(16, 0xc0);
        CHECK(gfx10_dcc_fast_clear_rgba8(clear_pixels.data(), 1,
                                         clear_0000.data(), clear_0000.size(), 4, true) &&
              clear_pixels[0] == 0 && clear_pixels[1] == 0 &&
              clear_pixels[2] == 0 && clear_pixels[3] == 0 &&
              gfx10_dcc_fast_clear_rgba8(clear_pixels.data(), 1,
                                         clear_1111.data(), clear_1111.size(), 4, true) &&
              clear_pixels[0] == 255 && clear_pixels[1] == 255 &&
              clear_pixels[2] == 255 && clear_pixels[3] == 255,
              "uniform DCC_CLEAR_0000 and DCC_CLEAR_1111 materialize all-zero/all-one");
        const std::vector<uint8_t> clear_1110(16, 0x80);
        CHECK(gfx10_dcc_fast_clear_rgba8(clear_pixels.data(), 1,
                                         clear_1110.data(), clear_1110.size(),
                                         4, false) &&
              clear_pixels[0] == 0 && clear_pixels[1] == 255 &&
              clear_pixels[2] == 255 && clear_pixels[3] == 255,
              "uniform DCC_CLEAR_1110 places alpha zero in the LSB component");
        CHECK(gfx10_dcc_fast_clear_rgba8(clear_pixels.data(), 1,
                                         clear_0000.data(), clear_0000.size(),
                                         3, true) &&
              clear_pixels[0] == 0 && clear_pixels[1] == 0 &&
              clear_pixels[2] == 0 && clear_pixels[3] == 255 &&
              gfx10_dcc_fast_clear_rgba8(clear_pixels.data(), 1,
                                         clear_1110.data(), clear_1110.size(),
                                         3, false) &&
              clear_pixels[0] == 255 && clear_pixels[1] == 255 &&
              clear_pixels[2] == 255 && clear_pixels[3] == 255,
              "three-component DCC clears materialize RGB with sampled alpha one");
        std::vector<uint8_t> mixed_clear(16, 0); mixed_clear.back() = 0x40;
        const std::vector<uint8_t> uncompressed(16, 0xff);
        CHECK(!gfx10_dcc_fast_clear_rgba8(clear_pixels.data(), 1,
                                          mixed_clear.data(), mixed_clear.size(), 4, true) &&
              !gfx10_dcc_fast_clear_rgba8(clear_pixels.data(), 1,
                                          uncompressed.data(), uncompressed.size(), 4, true) &&
              !gfx10_dcc_fast_clear_rgba8(clear_pixels.data(), 1,
                                          clear_0001.data(), clear_0001.size(), 2, true),
              "mixed, uncompressed, and unvalidated component layouts remain unsupported");
        CHECK(tiled_volume_bytes(120, 68, 32, M, 8) == (size_t)1 * 2 * 32 * 65536,
              "120x68x32 @ 8 B uses 1x2 padded 2D blocks per Z slice");
        auto rt_volume = [&](uint32_t bpe) {
            static const uint32_t dims[5][2] = {
                {256,256}, {256,128}, {128,128}, {128,64}, {64,64}
            };
            uint32_t el = 0; while ((1u << el) < bpe) el++;
            const uint32_t w = dims[el][0] + 3, h = dims[el][1] + 2, d = 5;
            std::vector<uint8_t> ref((size_t)w * h * d * bpe);
            for (size_t i = 0; i < ref.size(); i++) ref[i] = (uint8_t)((i * 2246822519u) >> 13);
            const size_t bytes = tiled_volume_bytes(w, h, d, M, bpe);
            std::vector<uint8_t> tiled(bytes, 0), back(ref.size(), 0);
            return tile_volume(tiled.data(), tiled.size(), ref.data(), w, h, d, M, bpe) &&
                   detile_volume(back.data(), tiled.data(), tiled.size(), w, h, d, M, bpe) &&
                   back == ref;
        };
        CHECK(rt_volume(1) && rt_volume(2) && rt_volume(4) && rt_volume(8) && rt_volume(16),
              "SW_64KB_R_X 3D round-trip is exact at every supported texel size");

        const uint32_t w = 128, h = 64, d = 16, bpe = 8;
        std::vector<uint8_t> ref((size_t)w * h * d * bpe);
        for (size_t i = 0; i < ref.size(); i++) ref[i] = (uint8_t)((i >> 3) * 43 + i);
        std::vector<uint8_t> tiled((size_t)d * 65536, 0);
        CHECK(tile_volume(tiled.data(), tiled.size(), ref.data(), w, h, d, M, bpe),
              "one 128x64 8-B block per volume slice tiles successfully");
        auto at3 = [&](uint32_t x, uint32_t y, uint32_t z) {
            return &ref[(((size_t)z * h + y) * w + x) * bpe];
        };
        CHECK(std::memcmp(&tiled[(size_t)1 * 65536 + 2048], at3(0,0,1), bpe) == 0,
              "64KB_R_X 3D golden: z0 maps to byte-offset bit 11");
        CHECK(std::memcmp(&tiled[(size_t)2 * 65536 + 1024], at3(0,0,2), bpe) == 0,
              "64KB_R_X 3D golden: z1 maps to byte-offset bit 10");
        CHECK(std::memcmp(&tiled[(size_t)4 * 65536 + 512], at3(0,0,4), bpe) == 0,
              "64KB_R_X 3D golden: z2 maps to byte-offset bit 9");
        CHECK(std::memcmp(&tiled[(size_t)8 * 65536 + 256], at3(0,0,8), bpe) == 0,
              "64KB_R_X 3D golden: z3 maps to byte-offset bit 8");
    }

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
