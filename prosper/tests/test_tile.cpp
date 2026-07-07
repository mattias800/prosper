// test_tile — the GPU surface de-swizzle (tile.hpp). Verifies the SW_4KB_S detile is the exact inverse of
// the tile (round-trip identity for arbitrary sizes), that linear mode is a passthrough, and that the
// tiled layout is genuinely a permutation (no texel dropped/duplicated).
#include "../src/gpu/tile.hpp"
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

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
