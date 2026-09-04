// test_tile — the GPU surface de-swizzle (tile.hpp). Verifies the standard-swizzle detilers are exact
// inverses of tiling (round-trip identity for arbitrary sizes), that linear mode is a passthrough, and
// that tiled layouts are genuine permutations (no texel dropped/duplicated).
#include "gpu/texture/tile.hpp"
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

    CHECK(tile_mode_is_tiled((uint32_t)TileMode::Sw256BS), "tile_mode 1 (SW_256B_S) is tiled");
    CHECK(tile_mode_is_tiled((uint32_t)TileMode::Sw4KbS), "tile_mode 5 (SW_4KB_S) is tiled");
    CHECK(!tile_mode_is_tiled((uint32_t)TileMode::Linear), "tile_mode 0 (linear) is not tiled");
    CHECK(linear_sampled_row_pitch(1920, 1) == 2048 &&
          linear_sampled_surface_bytes(1920, 1080, 1) == 2048u * 1080u,
          "linear R8 sampled images align each row to 256 bytes");
    CHECK(linear_sampled_row_pitch(3840, 1) == 3840 &&
          linear_sampled_surface_bytes(3840, 2160, 1) == 3840u * 2160u,
          "already-aligned linear sampled rows retain their tight pitch");

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

    // GFX10 SW_256B_S: small textures use 256-byte micro-tiles. The pattern is not a tight row-major
    // image: at 16 bytes/element (BC3 blocks), y0/y1 occupy byte-address bits 4/5 and x0/x1 bits 6/7.
    // Plucky Squire's 144x144 controller glyph is exactly a 36x36 grid of these BC3 blocks.
    {
        const uint32_t M = (uint32_t)TileMode::Sw256BS;
        CHECK(tiled_elements_bytes(36, 36, 16, M) == 20736,
              "SW_256B_S BC3: 36x36 blocks occupy 9x9 256-byte micro-tiles");
        bool roundtrips = true;
        struct Case { uint32_t w, h, bpe; };
        const Case cases[] = {
            {19, 18, 1}, {19, 11, 2}, {11, 10, 4}, {11, 7, 8}, {7, 7, 16},
        };
        for (const Case& c : cases) {
            std::vector<uint8_t> ref((size_t)c.w * c.h * c.bpe);
            for (size_t i = 0; i < ref.size(); ++i)
                ref[i] = (uint8_t)((i * 2246822519u) >> 17);
            const size_t bytes = tiled_elements_bytes(c.w, c.h, c.bpe, M);
            std::vector<uint8_t> tiled(bytes, 0), back(ref.size(), 0);
            tile_surface(tiled.data(), ref.data(), c.w, c.h, M, 0, c.bpe);
            detile_elements(back.data(), tiled.data(), tiled.size(), c.w, c.h, c.bpe, M);
            roundtrips &= back == ref;
        }
        CHECK(roundtrips, "SW_256B_S round-trip is exact at every supported element size");

        const uint32_t w = 36, h = 36, bpe = 16;
        std::vector<uint8_t> ref((size_t)w * h * bpe);
        for (size_t i = 0; i < ref.size(); ++i) ref[i] = (uint8_t)((i >> 4) * 73 + i);
        std::vector<uint8_t> tiled(tiled_elements_bytes(w, h, bpe, M), 0);
        tile_surface(tiled.data(), ref.data(), w, h, M, 0, bpe);
        auto at = [&](uint32_t x, uint32_t y) { return &ref[((size_t)y * w + x) * bpe]; };
        CHECK(std::memcmp(&tiled[16],  at(0, 1), bpe) == 0,
              "SW_256B_S 16B golden: element (0,1) at byte 16 (y0 -> bit 4)");
        CHECK(std::memcmp(&tiled[32],  at(0, 2), bpe) == 0,
              "SW_256B_S 16B golden: element (0,2) at byte 32 (y1 -> bit 5)");
        CHECK(std::memcmp(&tiled[64],  at(1, 0), bpe) == 0,
              "SW_256B_S 16B golden: element (1,0) at byte 64 (x0 -> bit 6)");
        CHECK(std::memcmp(&tiled[128], at(2, 0), bpe) == 0,
              "SW_256B_S 16B golden: element (2,0) at byte 128 (x1 -> bit 7)");
        CHECK(std::memcmp(&tiled[256], at(4, 0), bpe) == 0,
              "SW_256B_S 16B golden: element (4,0) starts the next block");
        CHECK(std::memcmp(&tiled[(size_t)9 * 256], at(0, 4), bpe) == 0,
              "SW_256B_S 16B golden: element (0,4) starts block row one");

        const TiledMipLevelLayout mip0 = tiled_mip_level_layout(36, 36, 16, M, 2, 0);
        const TiledMipLevelLayout mip1 = tiled_mip_level_layout(36, 36, 16, M, 2, 1);
        const TiledMipLevelLayout mip2 = tiled_mip_level_layout(36, 36, 16, M, 2, 2);
        CHECK(mip0.supported && !mip0.in_tail && mip0.byte_offset == 8704 &&
              mip1.supported && mip1.byte_offset == 2304 &&
              mip2.supported && mip2.byte_offset == 0,
              "SW_256B_S mip chain is reverse-ordered and independently block-aligned");
    }

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

    // --- GFX10 64KB modes (#288/#825): S (9), Z_X (24), and R_X (27). ---
    CHECK(tile_mode_is_tiled((uint32_t)TileMode::Sw64KbS),  "tile_mode 9 (SW_64KB_S) is tiled");
    CHECK(tile_mode_is_tiled((uint32_t)TileMode::Sw64KbZX), "tile_mode 24 (SW_64KB_Z_X) is tiled");
    CHECK(tile_mode_is_tiled((uint32_t)TileMode::Sw64KbRX), "tile_mode 27 (SW_64KB_R_X) is tiled");

    // GFX10 thin 2D mip chains are tail-first, then smallest-to-largest outside the tail. Evergate's
    // 2048x1152 RGBA8, 12-level SW_4KB_S background therefore starts mip 0 after a 4KB tail and
    // padded mips 6..1, not at the descriptor base. These offsets are direct AddrLib arithmetic.
    {
        const uint32_t M = (uint32_t)TileMode::Sw4KbS;
        CHECK(tiled_mip_level_offset(2048, 1152, 4, M, 11, 0) == 3186688,
              "SW_4KB_S mip0 follows the tail and reversed smaller levels");
        CHECK(tiled_mip_level_offset(2048, 1152, 4, M, 11, 1) == 827392,
              "SW_4KB_S mip1 offset uses the same tail-first chain");
        CHECK(tiled_mip_chain_bytes(256, 256, 8, (uint32_t)TileMode::Sw64KbRX, 8) == 720896,
              "RGBA16F 256-square R_X array slice includes its complete tail-first mip chain");
        CHECK(tiled_mip_chain_bytes(256, 256, 8, (uint32_t)TileMode::Sw64KbS, 8) == 720896,
              "RGBA16F 256-square S array slice has the matching complete mip-chain stride");
        CHECK(tiled_mip_chain_bytes(32, 32, 8, (uint32_t)TileMode::Sw64KbRX, 5) == 65536,
              "small RGBA16F cube-face chain wholly contained by the tail occupies one block");
        CHECK(tiled_mip_level_offset(2048, 1152, 4, M, 11, 6) == 4096,
              "last non-tail mip immediately follows the shared 4KB tail");
        CHECK(tiled_mip_level_offset(2048, 1152, 4, M, 0, 0) == 0,
              "single-level surfaces remain based at byte zero");

        const TiledMipLevelLayout tail7 =
            tiled_mip_level_layout(2048, 1152, 4, M, 11, 7);
        CHECK(tail7.supported && tail7.in_tail && tail7.byte_offset == 2048 &&
                  tail7.tail_x == 16 && tail7.tail_y == 0 && tail7.tail_block_bytes == 4096,
              "first SW_4KB_S tail level exposes AddrLib byte and element origins");
        const TiledMipLevelLayout tail11 =
            tiled_mip_level_layout(2048, 1152, 4, M, 11, 11);
        CHECK(tail11.supported && tail11.in_tail && tail11.byte_offset == 768 &&
                  tail11.tail_x == 8 && tail11.tail_y == 8,
              "last allocated tail level retains its sparse max-tail slot");

        // Pack level 7 into its shared block using the equivalent global tail coordinate, then read
        // it through the byte-origin API. A write must alter only that level and preserve every
        // sentinel byte belonging to its siblings.
        std::vector<uint8_t> full_linear((size_t)32 * 32 * 4, 0);
        std::vector<uint8_t> level_linear((size_t)16 * 9 * 4, 0);
        for (uint32_t y = 0; y < 9; ++y) for (uint32_t x = 0; x < 16; ++x) {
            const uint32_t value = 0x81000000u | (y << 8) | x;
            std::memcpy(level_linear.data() + ((size_t)y * 16 + x) * 4, &value, 4);
            std::memcpy(full_linear.data() +
                            ((size_t)(tail7.tail_y + y) * 32 + tail7.tail_x + x) * 4,
                        &value, 4);
        }
        std::vector<uint8_t> shared_tail(4096, 0);
        tile_surface(shared_tail.data(), full_linear.data(), 32, 32, M, 0, 4);
        std::vector<uint8_t> extracted(level_linear.size(), 0);
        detile_surface_level(extracted.data(), shared_tail.data(), shared_tail.size(),
                             16, 9, M, 4, tail7.tail_x, tail7.tail_y);
        CHECK(extracted == level_linear,
              "tail byte origin detiles the same texels as the documented global coordinate");
        std::vector<uint8_t> rewritten = shared_tail;
        std::vector<uint8_t> replacement(level_linear.size(), 0x5a);
        tile_surface_level(rewritten.data(), rewritten.size(), replacement.data(),
                           16, 9, M, 4, tail7.tail_x, tail7.tail_y);
        std::vector<uint8_t> roundtrip(replacement.size(), 0);
        detile_surface_level(roundtrip.data(), rewritten.data(), rewritten.size(),
                             16, 9, M, 4, tail7.tail_x, tail7.tail_y);
        CHECK(roundtrip == replacement,
              "tail level write/read round-trip preserves the selected mip");
        size_t changed = 0;
        for (size_t i = 0; i < rewritten.size(); ++i) changed += rewritten[i] != shared_tail[i];
        CHECK(changed <= replacement.size(),
              "tail writeback does not clear or overwrite sibling padding bytes");
    }

    // A packed-tail level reports TWO origins for the same texels: byte_offset and (tail_x, tail_y).
    // They must agree, i.e. the element origin has to land on byte_offset under the surface's own
    // swizzle. The round-trip above cannot see a disagreement because it both writes and reads at
    // (tail_x, tail_y); only cross-checking against byte_offset pins the real placement. This caught
    // 4 KiB tail levels resolving to a quarter of their true element origin (#320: PPSA02664's
    // mipped 4x4 SpriteMask sprites decoded foreign, fully transparent texels, so every mask
    // fragment failed its alpha test and a "visible outside mask" fill covered the whole frame).
    {
        struct Case { uint32_t mode, bpe, w, h, max_mip; };
        const Case cases[] = {
            {(uint32_t)TileMode::Sw4KbS,   4,    4,    4,  2},   // PPSA02664 mask sprite
            {(uint32_t)TileMode::Sw4KbS,   4, 2048, 1152, 11},
            {(uint32_t)TileMode::Sw4KbS,   8,  256,  256,  8},
            {(uint32_t)TileMode::Sw4KbS,  16,  256,  256,  8},
            {(uint32_t)TileMode::Sw64KbS,  4, 1920, 1080,  7},
            {(uint32_t)TileMode::Sw64KbS,  8,  512,  512,  9},
        };
        size_t checked = 0, disagreed = 0;
        for (const Case& c : cases) {
            // Element extent of one whole block for this mode/bpe, found through the public API.
            const size_t block_bytes = tiled_elements_bytes(1, 1, c.bpe, c.mode);
            uint32_t bw = 0, bh = 0;
            for (uint32_t w = 1; w <= 1024 && !bw; w <<= 1)
                for (uint32_t h = 1; h <= 1024; h <<= 1)
                    if ((size_t)w * h * c.bpe == block_bytes &&
                        tiled_elements_bytes(w, h, c.bpe, c.mode) == block_bytes) { bw = w; bh = h; break; }
            if (!bw || !bh) continue;
            // Stamp every element of one block with its own coordinate, then tile it.
            std::vector<uint8_t> linear(block_bytes, 0), tiled(block_bytes, 0);
            for (uint32_t y = 0; y < bh; ++y) for (uint32_t x = 0; x < bw; ++x) {
                const uint32_t stamp = (y << 16) | x;
                std::memcpy(linear.data() + ((size_t)y * bw + x) * c.bpe, &stamp, 4);
            }
            tile_surface(tiled.data(), linear.data(), bw, bh, c.mode, 0, c.bpe);
            for (uint32_t level = 0; level <= c.max_mip; ++level) {
                const TiledMipLevelLayout l =
                    tiled_mip_level_layout(c.w, c.h, c.bpe, c.mode, c.max_mip, level);
                if (!l.supported || !l.in_tail) continue;
                if (l.byte_offset + 4 > tiled.size()) continue;
                uint32_t stamp = 0;
                std::memcpy(&stamp, tiled.data() + l.byte_offset, 4);
                ++checked;
                if ((stamp & 0xffffu) != l.tail_x || (stamp >> 16) != l.tail_y) ++disagreed;
            }
        }
        CHECK(checked >= 20, "tail-origin cross-check exercised the packed-tail levels");
        CHECK(disagreed == 0,
              "every packed-tail element origin addresses exactly its reported byte_offset");
    }

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

    // Official GFX10 AddrLib linear mip chains are also smallest-first. Astro's 480x270 RGBA32F
    // post resource has eight levels: each pitch is 256-byte aligned (64 four-byte elements).
    {
        const TiledMipLevelLayout linear3 = tiled_mip_level_layout(480, 270, 4, 0, 7, 3);
        const TiledMipLevelLayout linear5 = tiled_mip_level_layout(480, 270, 4, 0, 7, 5);
        CHECK(linear3.supported && !linear3.in_tail && linear3.byte_offset == 7680,
              "Astro linear mip3 follows smaller aligned levels exactly");
        CHECK(linear5.supported && !linear5.in_tail && linear5.byte_offset == 1536,
              "Astro linear mip5 follows levels 7 and 6 exactly");
    }

    // Astro Bot's 1920x1080 post chain selects levels 5/6 of an eight-level 64KB_R_X allocation.
    // Both are packed into the first 64 KiB macroblock, at the exact coordinates below.
    {
        const uint32_t M = (uint32_t)TileMode::Sw64KbRX;
        const TiledMipLevelLayout tail5 =
            tiled_mip_level_layout(1920, 1080, 4, M, 7, 5);
        const TiledMipLevelLayout tail6 =
            tiled_mip_level_layout(1920, 1080, 4, M, 7, 6);
        CHECK(tail5.supported && tail5.in_tail && tail5.byte_offset == 32768 &&
                  tail5.tail_x == 64 && tail5.tail_y == 0 &&
                  tail5.tail_block_bytes == 65536,
              "Astro 60x33 level-5 R_X tail coordinate is exact");
        CHECK(tail6.supported && tail6.in_tail && tail6.byte_offset == 16384 &&
                  tail6.tail_x == 0 && tail6.tail_y == 64,
              "Astro 30x16 level-6 R_X tail coordinate is exact");

        std::vector<uint8_t> block_linear((size_t)128 * 128 * 4, 0);
        std::vector<uint8_t> mip_linear((size_t)60 * 33 * 4, 0);
        for (uint32_t y = 0; y < 33; ++y) for (uint32_t x = 0; x < 60; ++x) {
            const uint32_t value = 0xa5000000u | (y << 10) | x;
            std::memcpy(mip_linear.data() + ((size_t)y * 60 + x) * 4, &value, 4);
            std::memcpy(block_linear.data() +
                            ((size_t)(tail5.tail_y + y) * 128 + tail5.tail_x + x) * 4,
                        &value, 4);
        }
        std::vector<uint8_t> tiled(65536, 0), extracted(mip_linear.size(), 0);
        tile_surface(tiled.data(), block_linear.data(), 128, 128, M, 0, 4);
        detile_surface_level(extracted.data(), tiled.data(), tiled.size(), 60, 33, M, 4,
                             tail5.tail_x, tail5.tail_y);
        CHECK(extracted == mip_linear,
              "Astro R_X tail extraction matches the full-block swizzle exactly");
    }

    // Round-trip identity for all three 64KB modes at every element size (incl. block-unaligned dims).
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
        const uint32_t S = (uint32_t)TileMode::Sw64KbS;
        const uint32_t Z = (uint32_t)TileMode::Sw64KbZX;
        const uint32_t R = (uint32_t)TileMode::Sw64KbRX;
        CHECK(rt64(S, 256, 256, 1),  "SW_64KB_S round-trip 256x256 @ 1 B (one block)");
        CHECK(rt64(S, 300, 140, 2),  "SW_64KB_S round-trip 300x140 @ 2 B (unaligned)");
        CHECK(rt64(S, 500, 300, 4),  "SW_64KB_S round-trip 500x300 @ 4 B (unaligned)");
        CHECK(rt64(S, 128, 64, 8),   "SW_64KB_S round-trip 128x64 @ 8 B (one block, BC1 grid)");
        CHECK(rt64(S, 130, 70, 16),  "SW_64KB_S round-trip 130x70 @ 16 B (unaligned BC7 grid)");
        CHECK(rt64(Z, 260, 260, 1),  "SW_64KB_Z_X round-trip 260x260 @ 1 B");
        CHECK(rt64(Z, 300, 140, 2),  "SW_64KB_Z_X round-trip 300x140 @ 2 B");
        CHECK(rt64(Z, 500, 300, 4),  "SW_64KB_Z_X round-trip 500x300 @ 4 B");
        CHECK(rt64(Z, 130, 70, 8),   "SW_64KB_Z_X round-trip 130x70 @ 8 B");
        CHECK(rt64(Z, 70, 70, 16),   "SW_64KB_Z_X round-trip 70x70 @ 16 B");
        CHECK(rt64(R, 260, 130, 1),  "SW_64KB_R_X round-trip 260x130 @ 1 B");
        CHECK(rt64(R, 300, 140, 2),  "SW_64KB_R_X round-trip 300x140 @ 2 B");
        CHECK(rt64(R, 500, 300, 4),  "SW_64KB_R_X round-trip 500x300 @ 4 B");
        CHECK(rt64(R, 130, 70, 8),   "SW_64KB_R_X round-trip 130x70 @ 8 B");
        CHECK(rt64(R, 70, 70, 16),   "SW_64KB_R_X round-trip 70x70 @ 16 B");
        // Large 64KB-mode surfaces (>= 512 KiB) take the row-parallel detile path (#1177) — these span
        // many thread chunks, so round-trip identity proves the threaded sw64kb_copy walk is byte-for-byte
        // equal to the single-threaded result. The gpu_surface_detile_threaded CTest forces
        // PROSPER_DETILE_THREADS so this runs multi-threaded regardless of hardware_concurrency, and the
        // default gpu_surface_detile run covers the single-threaded control (< the thread gate on a
        // 1-core CI, or via PROSPER_DETILE_SINGLE_THREADED).
        CHECK(rt64(R, 2048, 2048, 4), "SW_64KB_R_X round-trip 2048x2048 @ 4 B (row-parallel detile)");
        CHECK(rt64(R, 3840, 2160, 4), "SW_64KB_R_X round-trip 3840x2160 @ 4 B (row-parallel, native 4K)");
        CHECK(rt64(S, 3840, 2160, 4), "SW_64KB_S round-trip 3840x2160 @ 4 B (row-parallel, native 4K)");
    }

    // Bounded SW_64KB reads may end between a pair of adjacent texels. The paired fast path must
    // preserve the first complete texel and zero only the truncated neighbor, exactly like the
    // original per-element walk. A cutoff after texel (0,0) exercises this edge for every paired
    // element size.
    {
        const uint32_t M = (uint32_t)TileMode::Sw64KbS;
        bool split_pair_matches = true;
        for (uint32_t bpe : {1u, 2u, 4u, 8u}) {
            const uint32_t ew = 128, eh = 64;
            std::vector<uint8_t> linear((size_t)ew * eh * bpe);
            for (size_t i = 0; i < linear.size(); ++i)
                linear[i] = static_cast<uint8_t>(0x31u + i * 17u);
            std::vector<uint8_t> tiled(tiled_elements_bytes(ew, eh, bpe, M), 0);
            tile_surface(tiled.data(), linear.data(), ew, eh, M, 0, bpe);

            std::vector<uint8_t> actual(linear.size(), 0xcc);
            detile_elements(actual.data(), tiled.data(), bpe, ew, eh, bpe, M);
            split_pair_matches &=
                std::memcmp(actual.data(), linear.data(), bpe) == 0 &&
                std::all_of(actual.begin() + bpe, actual.end(),
                            [](uint8_t value) { return value == 0; });
        }
        CHECK(split_pair_matches,
              "64KB bounded detile preserves a complete texel when its paired neighbor is truncated");
    }

    // AMD AddrLib GFX10_SW_64K_Z_X_1xaa golden, 16 pipes, 4 bpe. The selected pattern is
    // nibble01[10] + nibble2[74] + nibble3[31]: low Morton pairs, four pipe-XOR bits, then
    // y3/x4/y6/x6. These independent positions pin both the Z order and the pipe rotation used
    // by Astro Bot's live mode-24 compute surfaces; a tile/detile round-trip alone cannot do that.
    {
        const uint32_t M = (uint32_t)TileMode::Sw64KbZX;
        const uint32_t ew = 128, eh = 128, bpe = 4;
        std::vector<uint8_t> ref((size_t)ew * eh * bpe);
        for (size_t i = 0; i < ref.size(); i++) ref[i] = (uint8_t)((i >> 2) * 181 + i);
        std::vector<uint8_t> tiled(65536, 0);
        tile_surface(tiled.data(), ref.data(), ew, eh, M, 0, bpe);
        auto at = [&](uint32_t x, uint32_t y) { return &ref[((size_t)y * ew + x) * bpe]; };
        CHECK(std::memcmp(&tiled[4],      at(1, 0),  bpe) == 0, "64KB_Z_X 4B golden: element (1,0) at byte 4");
        CHECK(std::memcmp(&tiled[8],      at(0, 1),  bpe) == 0, "64KB_Z_X 4B golden: element (0,1) at byte 8");
        CHECK(std::memcmp(&tiled[256],    at(8, 0),  bpe) == 0, "64KB_Z_X 4B golden: element (8,0) at byte 256");
        CHECK(std::memcmp(&tiled[4352],   at(0, 8),  bpe) == 0, "64KB_Z_X 4B golden: element (0,8) at byte 4352");
        CHECK(std::memcmp(&tiled[8704],   at(16, 0), bpe) == 0, "64KB_Z_X 4B golden: element (16,0) at byte 8704");
        CHECK(std::memcmp(&tiled[512],    at(0, 16), bpe) == 0, "64KB_Z_X 4B golden: element (0,16) at byte 512");
        CHECK(std::memcmp(&tiled[2048],   at(32, 0), bpe) == 0, "64KB_Z_X 4B golden: element (32,0) at byte 2048");
        CHECK(std::memcmp(&tiled[1024],   at(0, 32), bpe) == 0, "64KB_Z_X 4B golden: element (0,32) at byte 1024");
        CHECK(std::memcmp(&tiled[33792],  at(64, 0), bpe) == 0, "64KB_Z_X 4B golden: element (64,0) at byte 33792");
        CHECK(std::memcmp(&tiled[18432],  at(0, 64), bpe) == 0, "64KB_Z_X 4B golden: element (0,64) at byte 18432");
    }

    // AMD AddrLib GFX10_SW_64K_Z_X_4xaa golden, 16 pipes, 4 B/sample. S0/S1 occupy
    // byte-offset bits 2/3, followed by x0/y0 at bits 4/5. The four values of one pixel therefore
    // occupy bytes 0,4,8,12, while the next x/y texels begin at 16/32. This independently pins the
    // sample lever; an ordinary 1xaa detiler aliases sample 0 and cannot satisfy these positions.
    {
        const uint32_t M = (uint32_t)TileMode::Sw64KbZX;
        const uint32_t w = 64, h = 64, bpe = 4, samples = 4;
        const size_t plane = static_cast<size_t>(w) * h;
        std::vector<uint32_t> linear(plane * samples);
        for (uint32_t s = 0; s < samples; ++s)
            for (uint32_t y = 0; y < h; ++y)
                for (uint32_t x = 0; x < w; ++x)
                    linear[static_cast<size_t>(s) * plane + static_cast<size_t>(y) * w + x] =
                        0xa5000000u | (s << 20) | (y << 10) | x;
        const size_t bytes = tiled_msaa_surface_bytes(w, h, M, bpe, samples);
        CHECK(bytes == 65536u, "64KB_Z_X 4xaa 4B: one 64x64x4 block is exactly 64 KiB");
        std::vector<uint8_t> tiled(bytes, 0);
        CHECK(tile_msaa_surface(tiled.data(), tiled.size(),
                                  reinterpret_cast<const uint8_t*>(linear.data()),
                                  w, h, M, bpe, samples),
              "64KB_Z_X 4xaa surface tiles through the explicit sample-bit pattern");
        auto word_at = [&](size_t byte) {
            uint32_t value = 0; std::memcpy(&value, tiled.data() + byte, sizeof(value)); return value;
        };
        CHECK(word_at(0) == linear[0 * plane] && word_at(4) == linear[1 * plane] &&
                  word_at(8) == linear[2 * plane] && word_at(12) == linear[3 * plane],
              "64KB_Z_X 4xaa golden: sample 0/1/2/3 map to byte 0/4/8/12");
        CHECK(word_at(16) == linear[1] && word_at(32) == linear[w],
              "64KB_Z_X 4xaa golden: x1/y1 follow the two sample bits at byte 16/32");
        CHECK(word_at(256) == linear[8] && word_at(4352) == linear[8 * w],
              "64KB_Z_X 4xaa golden: pipe-rotated x8/y8 positions match AddrLib");
        std::vector<uint32_t> back(linear.size(), 0);
        CHECK(detile_msaa_surface(reinterpret_cast<uint8_t*>(back.data()), tiled.data(),
                                    tiled.size(), w, h, M, bpe, samples) && back == linear,
              "64KB_Z_X 4xaa detile restores four distinct sample planes exactly");
    }

    {
        const uint32_t M = (uint32_t)TileMode::Sw64KbZX;
        bool all_sizes_round_trip = true;
        for (uint32_t bpe : {1u, 2u, 4u, 8u, 16u}) {
            const uint32_t w = 137, h = 73, samples = 4;
            const size_t linear_bytes = static_cast<size_t>(w) * h * samples * bpe;
            std::vector<uint8_t> linear(linear_bytes), back(linear_bytes, 0);
            for (size_t i = 0; i < linear.size(); ++i)
                linear[i] = static_cast<uint8_t>(i * 109u + bpe * 17u);
            const size_t bytes = tiled_msaa_surface_bytes(w, h, M, bpe, samples);
            std::vector<uint8_t> tiled(bytes, 0);
            all_sizes_round_trip &= bytes != 0 &&
                tile_msaa_surface(tiled.data(), tiled.size(), linear.data(),
                                  w, h, M, bpe, samples) &&
                detile_msaa_surface(back.data(), tiled.data(), tiled.size(),
                                    w, h, M, bpe, samples) && back == linear;
        }
        CHECK(all_sizes_round_trip,
              "64KB_Z_X 4xaa round-trip is exact at every supported element size and across blocks");
        std::vector<uint8_t> short_src(
            tiled_msaa_surface_bytes(64, 64, M, 4, 4) - 1u, 0);
        std::vector<uint8_t> untouched(64u * 64u * 4u * 4u, 0x5au);
        CHECK(!detile_msaa_surface(untouched.data(), short_src.data(), short_src.size(),
                                     64, 64, M, 4, 4) &&
                  std::all_of(untouched.begin(), untouched.end(), [](uint8_t v) { return v == 0x5au; }),
              "truncated MSAA allocation is rejected before any sample plane is detiled");
        CHECK(tiled_msaa_surface_bytes(64, 64, M, 4, 2) == 0 &&
                  tiled_msaa_surface_bytes(64, 64, (uint32_t)TileMode::Sw64KbS, 4, 4) == 0,
              "unsupported MSAA counts and swizzle modes remain fail-visible");
    }

    // AMD AddrLib's 16-pipe GFX10 HTILE shape and PAL's exact decompressed initialization
    // constants. This is the metadata contract that can authorize reading an ordinary R32_FLOAT
    // base allocation for Asterix's 2D_MSAA IMAGE_LOAD; a uniform-but-nearby value is not evidence.
    {
        const uint32_t M = (uint32_t)TileMode::Sw64KbZX;
        const size_t expected = gfx10_htile_msaa_metadata_bytes(1920, 1080, M, 4, true);
        CHECK(expected == 196608u,
              "1920x1080 16-pipe SW_64KB_Z_X HTILE occupies six 32 KiB metadata blocks");
        CHECK(gfx10_htile_msaa_metadata_bytes(1920, 1080, M, 2, true) == 0 &&
                  gfx10_htile_msaa_metadata_bytes(1920, 1080, M, 4, false) == 0 &&
                  gfx10_htile_msaa_metadata_bytes(
                      1920, 1080, (uint32_t)TileMode::Sw64KbRX, 4, true) == 0,
              "HTILE sizing remains fail-closed outside the exact 4xaa pipe-aligned Z_X contract");

        std::vector<uint32_t> htile(expected / sizeof(uint32_t), 0xfffc000fu);
        uint32_t uniform = 0;
        CHECK(gfx10_htile_metadata_is_decompressed(
                  reinterpret_cast<const uint8_t*>(htile.data()), expected, expected, &uniform) &&
                  uniform == 0xfffc000fu,
              "uniform depth-only PAL HTILE initialization proves an uncompressed base");
        std::fill(htile.begin(), htile.end(), 0xfffff3ffu);
        CHECK(gfx10_htile_metadata_is_decompressed(
                  reinterpret_cast<const uint8_t*>(htile.data()), expected, expected, &uniform) &&
                  uniform == 0xfffff3ffu,
              "uniform depth+stencil PAL HTILE initialization proves an uncompressed base");

        std::fill(htile.begin(), htile.end(), 0xfffc000eu);
        CHECK(!gfx10_htile_metadata_is_decompressed(
                  reinterpret_cast<const uint8_t*>(htile.data()), expected, expected),
              "one-bit-near PAL HTILE state cannot authorize reading base texels");
        std::fill(htile.begin(), htile.end(), 0xfffc000fu);
        htile[17] = 0xfffff3ffu;
        CHECK(!gfx10_htile_metadata_is_decompressed(
                  reinterpret_cast<const uint8_t*>(htile.data()), expected, expected),
              "nonuniform HTILE remains compressed or ambiguous and is rejected");
        htile[17] = 0xfffc000fu;
        CHECK(!gfx10_htile_metadata_is_decompressed(
                  reinterpret_cast<const uint8_t*>(htile.data()), expected - 4u, expected),
              "short HTILE metadata cannot prove the complete surface decompressed");

        // PAL GetClearValue(+0.0), depth-only, is exactly a uniform zero dword. Metadata owns the
        // result in that state: poison/NaN base bytes must never leak into any of the four layers.
        const uint32_t clear_w = 64, clear_h = 64;
        const size_t clear_meta_bytes = gfx10_htile_msaa_metadata_bytes(
            clear_w, clear_h, M, 4, true);
        const size_t clear_base_bytes = tiled_msaa_surface_bytes(
            clear_w, clear_h, M, 4, 4);
        std::vector<uint32_t> zero_htile(clear_meta_bytes / sizeof(uint32_t), 0u);
        std::vector<uint32_t> poison_base(clear_base_bytes / sizeof(uint32_t), 0x7fc00001u);
        std::vector<uint32_t> clear_planes(
            static_cast<size_t>(clear_w) * clear_h * 4u, 0x7fc00001u);
        Gfx10HtileMsaaSource clear_source = Gfx10HtileMsaaSource::Unsupported;
        CHECK(materialize_gfx10_htile_msaa_surface(
                  reinterpret_cast<uint8_t*>(clear_planes.data()),
                  clear_planes.size() * sizeof(uint32_t),
                  reinterpret_cast<const uint8_t*>(poison_base.data()),
                  poison_base.size() * sizeof(uint32_t),
                  reinterpret_cast<const uint8_t*>(zero_htile.data()),
                  zero_htile.size() * sizeof(uint32_t),
                  clear_w, clear_h, M, 4, 4, true, &clear_source) &&
                  clear_source == Gfx10HtileMsaaSource::DepthZeroFastClear &&
                  std::all_of(clear_planes.begin(), clear_planes.end(),
                              [](uint32_t bits) { return bits == 0u; }),
              "uniform zero HTILE materializes exact +0.0 in every texel and all four layers "
              "without reading poison base data");

        auto rejects_zero_clear = [&](const std::vector<uint32_t>& metadata,
                                      size_t metadata_bytes, uint32_t tile_mode,
                                      uint32_t bytes_per_texel, uint32_t samples) {
            std::fill(clear_planes.begin(), clear_planes.end(), 0x5a5a5a5au);
            clear_source = Gfx10HtileMsaaSource::UncompressedBase;
            return !materialize_gfx10_htile_msaa_surface(
                       reinterpret_cast<uint8_t*>(clear_planes.data()),
                       clear_planes.size() * sizeof(uint32_t),
                       reinterpret_cast<const uint8_t*>(poison_base.data()),
                       poison_base.size() * sizeof(uint32_t),
                       reinterpret_cast<const uint8_t*>(metadata.data()), metadata_bytes,
                       clear_w, clear_h, tile_mode, bytes_per_texel, samples, true,
                       &clear_source) &&
                   clear_source == Gfx10HtileMsaaSource::Unsupported &&
                   std::all_of(clear_planes.begin(), clear_planes.end(),
                               [](uint32_t bits) { return bits == 0x5a5a5a5au; });
        };
        std::vector<uint32_t> nonuniform_zero_htile = zero_htile;
        nonuniform_zero_htile[17] = 1u;
        CHECK(rejects_zero_clear(nonuniform_zero_htile, clear_meta_bytes, M, 4, 4),
              "one nonzero HTILE dword rejects without touching output");
        std::vector<uint32_t> unsupported_uniform_htile = zero_htile;
        std::fill(unsupported_uniform_htile.begin(), unsupported_uniform_htile.end(), 1u);
        CHECK(rejects_zero_clear(unsupported_uniform_htile, clear_meta_bytes, M, 4, 4),
              "unsupported nonzero uniform HTILE remains fail-visible");
        CHECK(rejects_zero_clear(zero_htile, clear_meta_bytes - 4u, M, 4, 4) &&
                  rejects_zero_clear(zero_htile, clear_meta_bytes, M, 4, 2) &&
                  rejects_zero_clear(zero_htile, clear_meta_bytes,
                                     (uint32_t)TileMode::Sw64KbS, 4, 4) &&
                  rejects_zero_clear(zero_htile, clear_meta_bytes, M, 8, 4),
              "zero-clear materialization rejects short footprint, count, swizzle, and element size");
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

    // GFX10 standard thick 3D SW_64KB_S. AddrLib's S3 pattern interleaves X/Y/Z inside one
    // 64 KiB macroblock instead of allocating a separate 2D block for every logical slice.
    {
        const uint32_t M = (uint32_t)TileMode::Sw64KbS;
        CHECK(tile_mode_supports_volume(M), "SW_64KB_S has an explicit standard 3D pattern");
        CHECK(tiled_volume_bytes(0, 32, 32, M, 8) == 0,
              "zero-width SW_64KB_S3 volumes are rejected without forming a block grid");
        CHECK(tiled_volume_bytes(32, 32, 32, M, 8) == 4 * 65536,
              "Plucky 32-cubed RGBA16F volume uses 1x2x2 S3 macroblocks");

        const uint32_t w = 32, h = 32, d = 32, bpe = 8;
        std::vector<uint8_t> ref((size_t)w * h * d * bpe);
        for (size_t i = 0; i < ref.size(); ++i) ref[i] = (uint8_t)((i >> 3) * 59 + i);
        std::vector<uint8_t> tiled(tiled_volume_bytes(w, h, d, M, bpe), 0);
        CHECK(tile_volume(tiled.data(), tiled.size(), ref.data(), w, h, d, M, bpe),
              "Plucky 32-cubed RGBA16F volume tiles successfully");
        auto at3 = [&](uint32_t x, uint32_t y, uint32_t z) {
            return &ref[(((size_t)z * h + y) * w + x) * bpe];
        };
        CHECK(std::memcmp(&tiled[8], at3(1,0,0), bpe) == 0,
              "64KB_S3 8B golden: x0 maps to byte-address bit 3");
        CHECK(std::memcmp(&tiled[16], at3(0,0,1), bpe) == 0,
              "64KB_S3 8B golden: z0 maps to byte-address bit 4");
        CHECK(std::memcmp(&tiled[32], at3(0,1,0), bpe) == 0,
              "64KB_S3 8B golden: y0 maps to byte-address bit 5");
        CHECK(std::memcmp(&tiled[32768], at3(16,0,0), bpe) == 0,
              "64KB_S3 8B golden: x4 maps to byte-address bit 15");
        CHECK(std::memcmp(&tiled[65536], at3(0,16,0), bpe) == 0,
              "64KB_S3 8B golden: y4 starts the next macroblock");
        CHECK(std::memcmp(&tiled[2 * 65536], at3(0,0,16), bpe) == 0,
              "64KB_S3 8B golden: z4 starts the next two-block slab");

        auto rt_volume = [&](uint32_t bytes) {
            static const uint32_t dims[5][3] = {
                {64,32,32}, {32,32,32}, {32,32,16}, {32,16,16}, {16,16,16}
            };
            uint32_t el = 0; while ((1u << el) < bytes) ++el;
            const uint32_t rw = dims[el][0] + 1;
            const uint32_t rh = dims[el][1] + 1;
            const uint32_t rd = dims[el][2] + 1;
            std::vector<uint8_t> linear((size_t)rw * rh * rd * bytes);
            for (size_t i = 0; i < linear.size(); ++i)
                linear[i] = (uint8_t)((i * 3266489917u) >> 17);
            std::vector<uint8_t> swizzled(tiled_volume_bytes(rw, rh, rd, M, bytes), 0);
            std::vector<uint8_t> back(linear.size(), 0);
            return tile_volume(swizzled.data(), swizzled.size(), linear.data(), rw, rh, rd,
                               M, bytes) &&
                   detile_volume(back.data(), swizzled.data(), swizzled.size(), rw, rh, rd,
                                 M, bytes) && back == linear;
        };
        CHECK(rt_volume(1) && rt_volume(2) && rt_volume(4) && rt_volume(8) && rt_volume(16),
              "SW_64KB_S3 round-trip is exact for every supported texel size");
    }

    // GFX10 standard thick 3D SW_4KB_S (#2229). Its pattern is the LOW 12 BITS of the S3 table
    // above, so these arms exist to pin the nesting rather than to re-test the pattern: if the bit
    // count or the derived block dims drift, the golden addresses below move and the byte totals
    // stop landing on 4096.
    //
    // A round trip cannot establish any of this -- it is self-consistent for an arbitrary invented
    // swizzle -- so the load-bearing arms are the byte-address goldens and the two size checks.
    {
        const uint32_t M = (uint32_t)TileMode::Sw4KbS;
        CHECK(tile_mode_supports_volume(M), "SW_4KB_S has a standard 3D pattern (#2229)");

        // Every element size must fill exactly one 4 KiB block at its own derived dimensions. A
        // wrong bit count misses 4096 on all five at once, which is what makes this a real check.
        CHECK(tiled_volume_bytes(16, 16, 16, M, 1) == 4096 &&
              tiled_volume_bytes(8, 16, 16, M, 2) == 4096 &&
              tiled_volume_bytes(8, 16,  8, M, 4) == 4096 &&
              tiled_volume_bytes(8,  8,  8, M, 8) == 4096 &&
              tiled_volume_bytes(4,  8,  8, M, 16) == 4096,
              "each SW_4KB_S3 texel size fills exactly one 4 KiB block at its derived dimensions");
        CHECK(tiled_volume_bytes(0, 16, 16, M, 2) == 0,
              "zero-width SW_4KB_S3 volumes are rejected without forming a block grid");

        // The guest-corroborated case: Sonic Racing: CrossWorlds binds a 16x16x16 2-byte grading
        // LUT and the GUEST declares size=8192. Two 8x16x16 blocks is 8192, so the derivation
        // agrees with the title's own size field rather than only with itself.
        CHECK(tiled_volume_bytes(16, 16, 16, M, 2) == 8192,
              "PPSA08804's 16-cubed 2-byte LUT is 2 blocks = 8192 bytes, matching its declared size");

        const uint32_t w = 16, h = 16, d = 16, bpe = 2;
        std::vector<uint8_t> ref((size_t)w * h * d * bpe);
        for (size_t i = 0; i < ref.size(); ++i) ref[i] = (uint8_t)((i >> 1) * 31 + i);
        std::vector<uint8_t> tiled(tiled_volume_bytes(w, h, d, M, bpe), 0);
        CHECK(tile_volume(tiled.data(), tiled.size(), ref.data(), w, h, d, M, bpe),
              "PPSA08804's 16-cubed LUT tiles successfully");
        auto at3 = [&](uint32_t x, uint32_t y, uint32_t z) {
            return &ref[(((size_t)z * h + y) * w + x) * bpe];
        };
        // 2 B row, bits 1..11: x0->1, z0->2, y0->3, z1->4, y1->5, x1->6, z2->7, y2->8, x2->9,
        // z3->10, y3->11. Block geometry is 8x16x16, so x=8 opens the next macroblock.
        CHECK(std::memcmp(&tiled[2],   at3(1,0,0), bpe) == 0,
              "4KB_S3 2B golden: x0 maps to byte-address bit 1");
        CHECK(std::memcmp(&tiled[4],   at3(0,0,1), bpe) == 0,
              "4KB_S3 2B golden: z0 maps to byte-address bit 2");
        CHECK(std::memcmp(&tiled[8],   at3(0,1,0), bpe) == 0,
              "4KB_S3 2B golden: y0 maps to byte-address bit 3");
        CHECK(std::memcmp(&tiled[64],  at3(2,0,0), bpe) == 0,
              "4KB_S3 2B golden: x1 maps to byte-address bit 6");
        CHECK(std::memcmp(&tiled[512], at3(4,0,0), bpe) == 0,
              "4KB_S3 2B golden: x2 maps to byte-address bit 9");
        CHECK(std::memcmp(&tiled[2048], at3(0,8,0), bpe) == 0,
              "4KB_S3 2B golden: y3 maps to the top in-block bit 11");
        CHECK(std::memcmp(&tiled[4096], at3(8,0,0), bpe) == 0,
              "4KB_S3 2B golden: x=8 starts the second macroblock, not a bit inside the first");

        auto rt4k = [&](uint32_t bytes) {
            static const uint32_t dims[5][3] = {
                {16,16,16}, {8,16,16}, {8,16,8}, {8,8,8}, {4,8,8}
            };
            uint32_t el = 0; while ((1u << el) < bytes) ++el;
            const uint32_t rw = dims[el][0] + 1;
            const uint32_t rh = dims[el][1] + 1;
            const uint32_t rd = dims[el][2] + 1;
            std::vector<uint8_t> linear((size_t)rw * rh * rd * bytes);
            for (size_t i = 0; i < linear.size(); ++i)
                linear[i] = (uint8_t)((i * 2654435761u) >> 19);
            std::vector<uint8_t> swizzled(tiled_volume_bytes(rw, rh, rd, M, bytes), 0);
            std::vector<uint8_t> back(linear.size(), 0);
            return tile_volume(swizzled.data(), swizzled.size(), linear.data(), rw, rh, rd,
                               M, bytes) &&
                   detile_volume(back.data(), swizzled.data(), swizzled.size(), rw, rh, rd,
                                 M, bytes) && back == linear;
        };
        CHECK(rt4k(1) && rt4k(2) && rt4k(4) && rt4k(8) && rt4k(16),
              "SW_4KB_S3 round-trip is exact for every texel size (weak on its own; see above)");

        // States the block invariant directly: the address map covers one block exactly. Kept for
        // what it SAYS, not for extra detection power -- and that distinction was measured, not
        // assumed, because it was first added on a justification that turned out to be false.
        //
        // The claim was "the round trip cannot see aliasing, so this arm catches what it misses."
        // It can. Collapsing the z contribution (`fz[z] = 0`) reddens the goldens, BOTH round trips
        // and this arm together. The reason is arithmetic rather than incidental: one block holds
        // exactly 4096/bpe elements writing bpe bytes each -- 4096 bytes into 4096 bytes, for every
        // element size -- so a byte written twice forces a byte written never. **Aliasing and holes
        // are the same defect seen from two sides.** The round trip detects the aliasing (a collision
        // loses data); this arm detects the hole. For a single block they are equivalent.
        //
        // So do not add arms on the theory that this one is load-bearing where the others are not; it
        // is a readable statement of an invariant that the round trip already enforces obliquely.
        //
        // Mechanically: tile_volume memsets the destination to zero first, so a byte still zero
        // afterwards is a byte nothing wrote. The source therefore contains no zero bytes. No offset
        // arithmetic is duplicated here, which is the point -- a test that re-implements the map
        // agrees with itself whatever the map does.
        auto covers_block_bijectively = [&](uint32_t bpe, uint32_t bw, uint32_t bh, uint32_t bd) {
            const size_t elems = (size_t)bw * bh * bd;
            std::vector<uint8_t> linear(elems * bpe);
            for (size_t i = 0; i < linear.size(); ++i) linear[i] = (uint8_t)(i % 255u + 1u);  // never 0
            const size_t bytes = tiled_volume_bytes(bw, bh, bd, M, bpe);
            if (bytes != 4096 || elems * bpe != 4096) return false;
            std::vector<uint8_t> tiled(bytes, 0);
            if (!tile_volume(tiled.data(), tiled.size(), linear.data(), bw, bh, bd, M, bpe))
                return false;
            for (uint8_t b : tiled) if (b == 0) return false;   // an unwritten byte => a collision
            return true;
        };
        CHECK(covers_block_bijectively(1, 16, 16, 16) &&
              covers_block_bijectively(2,  8, 16, 16) &&
              covers_block_bijectively(4,  8, 16,  8) &&
              covers_block_bijectively(8,  8,  8,  8) &&
              covers_block_bijectively(16, 4,  8,  8),
              "SW_4KB_S3 maps one block bijectively for every texel size -- full coverage, no aliasing");
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

    {
        // Sonic Frontiers' Cyber Space scene-width kernels read one resource through
        // IMAGE_LOAD_MIP: 2048x2048, R32G32_FLOAT (8 B/texel), tile_mode 27, TWELVE levels
        // (#2818/#2790). Before any of that can be materialised, every level has to be locatable,
        // so pin the placement the uploader would have to trust: all 12 supported, tail levels
        // sharing one block, non-tail levels strictly disjoint and inside the chain.
        const uint32_t ew = 2048, eh = 2048, bpe = 8, tm = 27, max_mip = 11;
        const size_t chain = tiled_mip_chain_bytes(ew, eh, bpe, tm, max_mip);
        CHECK(chain != 0, "Frontiers' 12-level 2048x2048 R32G32F chain is modeled");
        bool all_ok = true, disjoint = true, in_chain = true;
        uint32_t tail_levels = 0;
        std::vector<std::pair<size_t, size_t>> spans;   // [begin, end) per non-tail level
        for (uint32_t level = 0; level <= max_mip; ++level) {
            const TiledMipLevelLayout l = tiled_mip_level_layout(ew, eh, bpe, tm, max_mip, level);
            if (!l.supported) { all_ok = false; break; }
            if (l.in_tail) { tail_levels++; continue; }   // tail levels share the first block
            const uint32_t lw = std::max(ew >> level, 1u), lh = std::max(eh >> level, 1u);
            const size_t bytes = (size_t)lw * lh * bpe;
            if (l.byte_offset + bytes > chain) in_chain = false;
            spans.emplace_back(l.byte_offset, l.byte_offset + bytes);
        }
        // Pairwise, NOT against a running end: a tiled chain stores its levels smallest-first, so
        // byte_offset DEcreases as the level index rises and an ascending-order check reports a
        // false overlap on a correct layout. (It did -- that was this test's first version.)
        for (size_t a = 0; a < spans.size() && disjoint; ++a)
            for (size_t b = a + 1; b < spans.size(); ++b)
                if (spans[a].first < spans[b].second && spans[b].first < spans[a].second) {
                    disjoint = false; break;
                }
        CHECK(all_ok, "every one of Frontiers' 12 mip levels reports a supported placement");
        CHECK(in_chain, "no Frontiers mip level is placed past the end of its own chain");
        CHECK(disjoint, "Frontiers' non-tail mip levels occupy disjoint byte ranges");
        CHECK(tail_levels > 0, "the small Frontiers levels are packed into the shared mip tail");
    }

    // #3149 targeted zero-fill: tile_surface now zeroes only what the copy will not write.
    // The case that separates it from the old whole-surface memset is a width that IS block
    // aligned with a height that is NOT -- 3840x2160 at 4 bytes/texel in Sw64KbRX, where blocks
    // are 128x128, so 3840%128 == 0 while 2160%128 == 112. Built by hand rather than drawn from
    // the generators above: a padding byte left stale is invisible to a round-trip check, so the
    // sentinel sweep is the half that can actually fail.
    {
        // All three 64 KiB modes, not just Sw64KbRX: the targeted zero-fill's guard keys on
        // is_64kb_mode(), so a mode-specific block geometry regression would otherwise be
        // invisible here. Flagged in review of #3149.
        const uint32_t modes[] = {(uint32_t)TileMode::Sw64KbS, (uint32_t)TileMode::Sw64KbZX,
                                  (uint32_t)TileMode::Sw64KbRX};
        for (const uint32_t M : modes) {
        const uint32_t W = 3840, H = 2160, BPE = 4;
        std::vector<uint8_t> lin((size_t)W * H * BPE);
        for (size_t i = 0; i < lin.size(); i++) {
            uint8_t v = (uint8_t)(i * 7 + 1);
            if (v == 0xAB) v = 0x11;          // 0xAB must stay exclusive to the sentinel
            lin[i] = v;
        }
        const size_t tb = tiled_surface_bytes(W, H, M, 0, BPE);
        std::vector<uint8_t> tiled(tb, 0xAB), back((size_t)W * H * BPE, 0);
        tile_surface(tiled.data(), lin.data(), W, H, M, 0, BPE);
        detile_surface(back.data(), tiled.data(), W, H, M, 0, BPE);
        size_t residual = 0;
        for (size_t i = 0; i < tb; i++) if (tiled[i] == 0xAB) residual++;
        CHECK(std::memcmp(lin.data(), back.data(), lin.size()) == 0,
              "4K block-aligned-width / unaligned-height tile+detile round-trips exactly");
        CHECK(residual == 0,
              "every tiled byte is written by the copy or zeroed -- no stale padding survives");
        }
    }

    // #3284 AVX2 vectorized tile_surface: verify that tile_full_block_row_avx2 produces byte-for-byte
    // identical tiled outputs to the unvectorized scalar path across element sizes (2, 4, 8, 16 BPE)
    // on 1080p surfaces.
    {
        const uint32_t modes[] = {(uint32_t)TileMode::Sw64KbS, (uint32_t)TileMode::Sw64KbZX,
                                  (uint32_t)TileMode::Sw64KbRX};
        for (const uint32_t M : modes) {
            for (const uint32_t bpe : {2u, 4u, 8u, 16u}) {
                const uint32_t W = 1920, H = 1080;
                std::vector<uint8_t> lin((size_t)W * H * bpe);
                for (size_t i = 0; i < lin.size(); i++)
                    lin[i] = static_cast<uint8_t>((i * 13u + 7u) ^ (i >> 8));
                const size_t tb = tiled_surface_bytes(W, H, M, 0, bpe);
                std::vector<uint8_t> tiled_avx2(tb, 0);
                std::vector<uint8_t> tiled_scalar(tb, 0);
                tile_surface(tiled_avx2.data(), lin.data(), W, H, M, 0, bpe, /*allow_avx2=*/true);
                tile_surface(tiled_scalar.data(), lin.data(), W, H, M, 0, bpe, /*allow_avx2=*/false);
                CHECK(std::memcmp(tiled_avx2.data(), tiled_scalar.data(), tb) == 0,
                      "AVX2 tile_surface matches scalar tile_surface byte-for-byte");

                // Mutation arm: verify the comparison is strictly discriminating by mutating
                // a byte in the tiled buffer and asserting a mismatch against the scalar baseline.
                tiled_avx2[tb / 2 + 13] ^= 0xa5;
                CHECK(std::memcmp(tiled_avx2.data(), tiled_scalar.data(), tb) != 0,
                      "mutation arm: corrupted tile buffer fails equality against scalar baseline");
            }
        }
    }

    // ---------------------------------------------------------------------------------------------
    // DESTINATION COVERAGE. The frontend materializer hands detile_surface a POOLED buffer that
    // still holds the previous surface's bytes (frontends/shared/live/decode_scratch.hpp), so it
    // relies on `detile_writes_whole_destination` to say when it may skip pre-zeroing.
    // A false positive there is a silent wrong picture, so assert it by filling the destination
    // with poison and requiring that none of it survives.
    {
        const uint32_t modes[] = {
            (uint32_t)TileMode::Linear,  (uint32_t)TileMode::Sw256BS, (uint32_t)TileMode::Sw4KbS,
            (uint32_t)TileMode::Sw64KbS, (uint32_t)TileMode::Sw64KbZX, (uint32_t)TileMode::Sw64KbRX,
        };
        // Block-aligned and deliberately ragged extents: partial blocks are where a copier that
        // bounds itself by the tiled source would leave a hole.
        const uint32_t sizes[][2] = {{256, 128}, {129, 67}, {1920, 1080}, {3840, 2176}, {7, 3}};
        bool any_true = false, any_false = false;
        for (const uint32_t M : modes) {
            for (const uint32_t bpe : {1u, 2u, 4u, 8u, 16u}) {
                for (const auto& wh : sizes) {
                    const uint32_t W = wh[0], H = wh[1];
                    const size_t dst_bytes = (size_t)W * H * bpe;
                    if (dst_bytes > (96u << 20)) continue;   // keep the sweep's wall time bounded
                    // Only the coverage claim is under test, so give the detiler a COMPLETE source:
                    // a short one legitimately zero-fills, which is a different contract.
                    const size_t tb = std::max<size_t>(
                        tiled_surface_bytes(W, H, M, 0, bpe), dst_bytes);
                    // The poison byte is excluded from the source, so a surviving one is
                    // unambiguous -- it can only be a destination byte nothing wrote.
                    std::vector<uint8_t> src(tb);
                    for (size_t i = 0; i < tb; i++) {
                        uint8_t v = (uint8_t)((i * 31u + 5u) ^ (i >> 7));
                        src[i] = (v == 0xDB) ? 0x00 : v;
                    }
                    const bool claims_full = detile_writes_whole_destination(M, bpe);
                    (claims_full ? any_true : any_false) = true;
                    if (!claims_full) continue;
                    // BOTH entry points, because the predicate in tile.hpp answers for both and the
                    // materializer uses both -- the BC branch of live_renderer.cpp calls
                    // `detile_elements`, every other converted branch calls `detile_surface`. They
                    // route to the same three copiers today, so testing one appeared to cover the
                    // other; a fast path added to either would falsify the predicate for the call
                    // sites that use it with nothing turning red.
                    for (int entry = 0; entry < 2; entry++) {
                        std::vector<uint8_t> dst(dst_bytes, 0xDB);   // poison
                        const char* who = entry ? "detile_elements" : "detile_surface";
                        if (entry)
                            detile_elements(dst.data(), src.data(), tb, W, H, bpe, M);
                        else
                            detile_surface(dst.data(), src.data(), W, H, M, 0, bpe);
                        size_t survivors = 0;
                        for (size_t i = 0; i < dst_bytes; i++) survivors += (dst[i] == 0xDB);
                        if (survivors) {
                            printf("  [FAIL] %s left %zu poison bytes for mode=%u bpe=%u "
                                   "%ux%u while claiming whole-destination coverage\n",
                                   who, survivors, M, bpe, W, H);
                            fails++;
                        }
                    }
                }
            }
        }
        CHECK(any_true, "coverage predicate accepts the shapes the materializer actually uses "
                        "(swept through detile_surface AND detile_elements)");
        // Mutation arm by construction, not by exit code alone: the predicate must be able to say
        // NO, or "claims_full" above is vacuous and the poison sweep tested nothing.
        CHECK(!detile_writes_whole_destination((uint32_t)TileMode::Sw64KbRX, 6u),
              "coverage predicate refuses a 64KB element size its pattern tables do not cover");
        CHECK(detile_writes_whole_destination((uint32_t)TileMode::Sw64KbRX, 8u),
              "coverage predicate accepts a 64KB element size the tables do cover");
        (void)any_false;
    }

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
