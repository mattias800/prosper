// test_tile — the GPU surface de-swizzle (tile.hpp). Verifies the SW_4KB_S detile is the exact inverse of
// the tile (round-trip identity for arbitrary sizes), that linear mode is a passthrough, and that the
// tiled layout is genuinely a permutation (no texel dropped/duplicated).
#include "../src/gpu/tile.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdint>
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

    // Golden positions: the 32-bpp order must be BYTE-IDENTICAL to the shipped, pixel-verified
    // layout (y0 in the low Morton bit: (0,1) -> element 1, (1,0) -> element 2), and the 8-bpp
    // cross-tile step must be one whole 4KB tile.
    {
        const uint32_t M = (uint32_t)TileMode::Sw4KbS;
        std::vector<uint8_t> lin32((size_t)64 * 32 * 4, 0);
        for (uint32_t y = 0; y < 32; y++) for (uint32_t x = 0; x < 64; x++)
            lin32[((size_t)y * 64 + x) * 4] = (uint8_t)(x ^ (y << 4) ^ 0x5A);
        std::vector<uint8_t> t32(tiled_surface_bytes(64, 32, M), 0);
        tile_surface(t32.data(), lin32.data(), 64, 32, M);
        CHECK(t32[1 * 4] == lin32[((size_t)1 * 64 + 0) * 4], "32-bpp golden: texel (0,1) at element 1 (y0 low)");
        CHECK(t32[2 * 4] == lin32[((size_t)0 * 64 + 1) * 4], "32-bpp golden: texel (1,0) at element 2");
        CHECK(t32[4096]  == lin32[((size_t)0 * 64 + 32) * 4], "32-bpp golden: texel (32,0) starts tile 1 (byte 4096)");

        std::vector<uint8_t> lin8((size_t)128 * 64, 0);
        for (size_t i = 0; i < lin8.size(); i++) lin8[i] = (uint8_t)(i * 31 + 7);
        std::vector<uint8_t> t8(tiled_surface_bytes(128, 64, M, 0, 1), 0);
        tile_surface(t8.data(), lin8.data(), 128, 64, M, 0, 1);
        CHECK(t8[1] == lin8[(size_t)1 * 128 + 0], "8-bpp golden: texel (0,1) at element 1 (same Morton order)");
        CHECK(t8[4096] == lin8[(size_t)0 * 128 + 64], "8-bpp golden: texel (64,0) starts tile 1 (byte 4096, 64-wide tile)");
    }

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
