// test_index_size_detect — the unannounced-32-bit index-buffer fingerprint (#304).
//
// DOLL's UE4 Slate/UMG quad index buffers are 32-bit, but the title never announces the index size
// (no sceAgcDcbSetIndexSize, no VGT_INDEX_TYPE register), so index_type defaults to 16-bit and each
// 32-bit index is misread as two 16-bit ones — collapsing the banner quad to a degenerate triangle
// (the #304 "flat wedge"). index_buffer_is_unannounced_32bit() recovers the real size from the
// buffer bytes. These vectors are the exact ones captured live from the DOLL boot.
#include "gpu/execute/gpu_execute.hpp"
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

    // ---------------------------------------------------------------------------------------
    // #304 part two: the same unannounced 32-bit buffer with indices AT OR ABOVE 0x10000.
    //
    // Tomb Raider I-III Remastered (PPSA16901) draws its level from one ~775,000-vertex pool, so a
    // draw's indices sit in a 64 KiB window above zero and the repeated high half is a NON-ZERO
    // constant, which the fingerprint above cannot match (its last assertion pins that).
    //
    // The two pointers address DIFFERENT memory here, and that is not an artifact of the test: for a
    // DrawIndexOffset the 16-bit address is index_base + offset*2 while the 32-bit one is
    // index_base + offset*4, so they diverge. Every vector below is a live capture from a boot to
    // Croft Manor (2026-08-26), paired as the executor sees them.
    printf("-- part two: constant non-zero high half --\n");

    auto detect_high = [](std::vector<uint16_t> i16, std::vector<uint32_t> i32, uint32_t n) {
        i16.resize(i16.size() + 2 * n, 0);
        i32.resize(i32.size() + n, 0);
        return index_buffer_is_unannounced_32bit_high(i16.data(), i32.data(), n);
    };

    // Odd parity carries the high half (the 16-bit address is 4-byte aligned).
    CHECK(detect_high({33161, 3, 33160, 3, 33159, 3, 33162, 3},
                      {428291, 428290, 428289, 428292, 428293, 428294, 428295, 428296}, 8),
          "TR world draw, odd-parity high half -> 32-bit");
    // Even parity carries it (the 16-bit address is 2 mod 4). 55,677 of the frame's draws look like
    // this against 21,871 of the shape above, so checking one parity misses most of the corruption.
    CHECK(detect_high({2, 41911, 2, 41916, 2, 41915, 2, 41913},
                      {301685, 301684, 301683, 301686, 301687, 301688, 301689, 301690}, 8),
          "TR world draw, even-parity high half -> 32-bit");

    // Genuine 16-bit buffers from the SAME frame must be left alone. Each is rejected by a
    // different clause, which is why all three are kept.
    CHECK(!detect_high({36, 37, 41, 36, 41, 40, 37, 38},
                       {5308496, 5242965, 5505109, 5701718, 5308496, 5242965, 5505109, 5701718}, 8),
          "TR character quad -> stays 16-bit (32-bit span too wide)");
    CHECK(!detect_high({4436, 4435, 4434, 4436, 4434, 4437, 4436, 4437},
                       {3256607121u, 3280918884u, 3242189160u, 3256607121u,
                        3280918884u, 3242189160u, 3256607121u, 3280918884u}, 8),
          "TR 16-bit mesh -> stays 16-bit (32-bit values past the plausibility cap)");
    CHECK(!detect_high({0, 1, 2, 3, 4, 2, 0, 5},
                       {65536, 196610, 131076, 327680, 65536, 196610, 131076, 327680}, 8),
          "16-bit list whose aliased dwords are small-ish -> stays 16-bit (no constant parity)");

    // Mutation arms: take the confirmed even-parity case and break exactly one clause each. Without
    // these the positive above cannot distinguish "the fingerprint matched" from "the function
    // returns true for anything shaped roughly like this".
    CHECK(!detect_high({2, 41911, 7, 41916, 2, 41915, 9, 41913},
                       {301685, 301684, 301683, 301686, 301687, 301688, 301689, 301690}, 8),
          "mutation: parity no longer constant -> rejected");
    CHECK(!detect_high({2, 41911, 2, 41916, 2, 41915, 2, 41913},
                       {301685, 999999, 301683, 301686, 301687, 301688, 301689, 301690}, 8),
          "mutation: 32-bit span exceeds one 64 KiB window -> rejected");
    CHECK(!detect_high({2, 41911, 2, 41916, 2, 41915, 2, 41913},
                       {301685, 301685, 301685, 301685, 301685, 301685, 301685, 301685}, 8),
          "mutation: 32-bit reading has one repeated value -> rejected");
    CHECK(!detect_high({0, 41911, 0, 41916, 0, 41915, 0, 41913},
                       {301685, 301684, 301683, 301686, 301687, 301688, 301689, 301690}, 8),
          "mutation: zero high half is the other detector's case -> rejected here");
    CHECK(!detect_high({2, 41911, 2, 41916},
                       {301685, 301684, 301683, 301686}, 4),
          "mutation: too few samples for 'constant parity' to mean anything -> rejected");

    // The two detectors must not overlap: what the zero-high-half one accepts, this one declines,
    // so the call site's ordering can never change an existing verdict.
    CHECK(!detect_high({0, 0, 1, 0, 2, 0, 17, 0}, {0, 1, 2, 17, 0, 1, 2, 17}, 8),
          "DOLL zero-high-half quad -> declined here, still owned by the original detector");

    printf(fails ? "FAILED (%d)\n" : "PASSED\n", fails);
    return fails ? 1 : 0;
}
