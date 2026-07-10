// test_index_size_detect — the unannounced-32-bit index-buffer fingerprint (#304).
//
// DOLL's UE4 Slate/UMG quad index buffers are 32-bit, but the title never announces the index size
// (no sceAgcDcbSetIndexSize, no VGT_INDEX_TYPE register), so index_type defaults to 16-bit and each
// 32-bit index is misread as two 16-bit ones — collapsing the banner quad to a degenerate triangle
// (the #304 "flat wedge"). index_buffer_is_unannounced_32bit() recovers the real size from the
// buffer bytes. These vectors are the exact ones captured live from the DOLL boot.
#include "../src/gpu/gpu_execute.hpp"
#include <cstdio>
#include <cstdint>
#include <vector>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

// Run the fingerprint over a 32-bit index vector: p32 reads the dwords, p16 aliases the same bytes.
static bool detect32(const std::vector<uint32_t>& i32, uint32_t n) {
    const uint16_t* p16 = reinterpret_cast<const uint16_t*>(i32.data());
    return index_buffer_is_unannounced_32bit(p16, i32.data(), n);
}
// Run it over a 16-bit index vector: p16 reads the words, p32 aliases the same bytes (what the misread
// would produce). Pad so p32 can read n dwords (2n words).
static bool detect16(std::vector<uint16_t> i16, uint32_t n) {
    i16.resize(i16.size() + 2 * n, 0);   // headroom for the 32-bit aliased read
    const uint32_t* p32 = reinterpret_cast<const uint32_t*>(i16.data());
    return index_buffer_is_unannounced_32bit(i16.data(), p32, n);
}

int main() {
    printf("== test_index_size_detect ==\n");

    // The real banner quad, stored 32-bit: [0,1,2,2,1,3] (two triangles over 4 verts). Detected 32-bit.
    CHECK(detect32({0, 1, 2, 2, 1, 3}, 6), "banner 32-bit quad [0,1,2,2,1,3] -> 32-bit");
    // Another live 32-bit quad captured: [2,1,3,0,1,2].
    CHECK(detect32({2, 1, 3, 0, 1, 2}, 6), "32-bit quad [2,1,3,0,1,2] -> 32-bit");

    // Genuine 16-bit buffers must NOT be reinterpreted.
    CHECK(!detect16({0, 1, 2, 0, 2, 3, 3, 2, 4, 3, 4, 5}, 12), "DOLL scene 16-bit mesh -> stays 16-bit");
    CHECK(!detect16({0, 4, 5}, 3),                              "DOLL fullscreen 16-bit tri [0,4,5] -> stays 16-bit");
    CHECK(!detect16({0, 1, 2, 2, 3, 0}, 6),                     "Messenger 16-bit quad [0,1,2,2,3,0] -> stays 16-bit");

    // Degenerate guards: an all-zero buffer (any_nonzero==false) and a 1-index draw (no odd word) stay 16-bit.
    CHECK(!detect32({0, 0, 0, 0}, 4), "all-zero 32-bit buffer -> not detected (needs real data)");
    CHECK(!detect32({7}, 1),          "single-index draw -> not detected (no odd word to test)");

    // A 32-bit buffer whose HIGH halves are non-zero (a real large mesh, indices >= 0x10000) is not a
    // small-index quad -> reject (the fingerprint requires all values < 0x10000).
    CHECK(!detect32({0x10005, 1, 2}, 3), "32-bit indices >= 0x10000 -> not the quad fingerprint");

    printf(fails ? "FAILED (%d)\n" : "PASSED\n", fails);
    return fails ? 1 : 0;
}
