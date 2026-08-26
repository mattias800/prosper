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
    // constant, which the fingerprint above cannot match (its last assertion pins that boundary).
    //
    // TWO CONFIGURATIONS, and both are covered here because they are not equivalent:
    //   * DISJOINT -- a DrawIndexOffset makes the 16-bit and 32-bit addresses different memory
    //     (index_base + offset*2 against index_base + offset*4).
    //   * ALIASED  -- with no DrawIndexOffset the caller passes THE SAME POINTER twice, so the two
    //     readings are the same bytes. This is the dangerous one: a 16-bit period-2 buffer is then
    //     byte-identical to a clustered 32-bit list, and only the vertex-range bound separates them.
    printf("-- part two: constant non-zero high half --\n");

    // Disjoint: independent arrays, as a DrawIndexOffset produces.
    auto detect_hi = [](std::vector<uint16_t> i16, std::vector<uint32_t> i32, uint32_t n,
                        uint32_t vb) {
        i16.resize(i16.size() + 2 * n, 0);
        i32.resize(i32.size() + n, 0);
        return index_buffer_is_unannounced_32bit_high(i16.data(), i32.data(), n, vb);
    };
    // Aliased: one buffer, read both ways -- the real no-offset call shape.
    auto detect_alias = [](std::vector<uint16_t> i16, uint32_t n, uint32_t vb) {
        i16.resize(i16.size() + 2 * n, 0);
        return index_buffer_is_unannounced_32bit_high(
            i16.data(), reinterpret_cast<const uint32_t*>(i16.data()), n, vb);
    };
    // Build the 16-bit image of a clustered 32-bit list: low half varying, high half constant.
    auto pack32 = [](uint32_t base, uint32_t count) {
        std::vector<uint16_t> v;
        for (uint32_t i = 0; i < count; i++) {
            v.push_back((uint16_t)((base + i) & 0xFFFF));
            v.push_back((uint16_t)((base + i) >> 16));
        }
        return v;
    };

    const uint32_t kTrPool = 775111;   // Tomb Raider's real level pool, size/stride

    // Live captures from the Croft Manor boot, in the disjoint shape the executor sees them.
    CHECK(detect_hi({33161, 3, 33160, 3, 33159, 3, 33162, 3},
                    {428291, 428290, 428289, 428292, 428293, 428294, 428295, 428296}, 8, kTrPool),
          "TR world draw, odd-parity high half -> 32-bit");
    CHECK(detect_hi({2, 41911, 2, 41916, 2, 41915, 2, 41913},
                    {301685, 301684, 301683, 301686, 301687, 301688, 301689, 301690}, 8, kTrPool),
          "TR world draw, even-parity high half -> 32-bit");
    // The same thing in the ALIASED shape, which nothing covered before.
    CHECK(detect_alias(pack32(428289, 16), 16, kTrPool),
          "TR-shaped 32-bit list, aliased pointers -> 32-bit");

    // --- The false positives that rejected the first version of this detector. ---
    // Each is a GENUINE 16-bit buffer with a period-2 pattern, byte-identical to a clustered 32-bit
    // list. The byte fingerprint alone accepts all of them; the vertex-range bound is what does not.
    // These were constructed by hand against the shipped header, outside whatever produced the
    // positives above, and each was confirmed to be accepted before the bound existed.
    {
        std::vector<uint16_t> spokes;        // LINE_LIST, 64 spokes to hub vertex 7
        for (uint16_t i = 0; i < 64; i++) { spokes.push_back(100 + i); spokes.push_back(7); }
        CHECK(!detect_alias(spokes, 64, 200),
              "16-bit line list radiating from hub vertex 7 -> stays 16-bit");
        std::vector<uint16_t> cone;          // TRIANGLE_STRIP cone, apex vertex 12
        for (uint16_t i = 0; i < 64; i++) { cone.push_back(200 + i); cone.push_back(12); }
        CHECK(!detect_alias(cone, 64, 1000),
              "16-bit triangle-strip cone with apex vertex 12 -> stays 16-bit");
        // Only the FIRST 128 words are period-2; the sample cap must not let that speak for the rest.
        std::vector<uint16_t> head_only = spokes;
        for (uint16_t i = 0; i < 2872; i++) head_only.push_back((uint16_t)(500 + (i % 97)));
        CHECK(!detect_alias(head_only, 3000, 4000),
              "16-bit mesh whose first 128 words happen to be period-2 -> stays 16-bit");
        // And the reason each is rejected is the BOUND, not the pattern: given a pool large enough to
        // hold the implied indices they would be accepted. This is the honest statement of the limit
        // and it is asserted rather than left in a comment.
        CHECK(detect_alias(spokes, 64, 1u << 20),
              "KNOWN LIMIT: with a pool that large the pattern IS accepted -- the bound is the\n"
          "           discriminator, and this arm is EXPECTED to flip if the detector is tightened");
    }

    // Genuine 16-bit buffers from the same frame must be left alone.
    CHECK(!detect_hi({36, 37, 41, 36, 41, 40, 37, 38},
                     {5308496, 5242965, 5505109, 5701718, 5308496, 5242965, 5505109, 5701718}, 8, kTrPool),
          "TR character quad -> stays 16-bit (32-bit values past the pool)");
    CHECK(!detect_hi({4436, 4435, 4434, 4436, 4434, 4437, 4436, 4437},
                     {3256607121u, 3280918884u, 3242189160u, 3256607121u,
                      3280918884u, 3242189160u, 3256607121u, 3280918884u}, 8, kTrPool),
          "TR 16-bit mesh -> stays 16-bit (32-bit values past the pool)");
    CHECK(!detect_hi({0, 1, 2, 3, 4, 2, 0, 5},
                     {65536, 196610, 131076, 327680, 65536, 196610, 131076, 327680}, 8, kTrPool),
          "16-bit list whose aliased dwords are in range -> stays 16-bit (span exceeds one window)");

    // Mutation arms: take the confirmed even-parity case and break exactly one clause each. Without
    // these the positives cannot distinguish "the fingerprint matched" from "this returns true for
    // anything roughly this shape".
    CHECK(!detect_hi({2, 41911, 7, 41916, 2, 41915, 9, 41913},
                     {301685, 301684, 301683, 301686, 301687, 301688, 301689, 301690}, 8, kTrPool),
          "mutation: parity no longer constant -> rejected");
    // The bound must be ABOVE the injected outlier, or the vertex-range clause rejects the arm first
    // and the span clause is never reached -- which is exactly what happened here: deleting the span
    // clause entirely left the whole suite green. A mutation arm that another clause short-circuits
    // tests nothing.
    CHECK(!detect_hi({2, 41911, 2, 41916, 2, 41915, 2, 41913},
                     {301685, 999999, 301683, 301686, 301687, 301688, 301689, 301690}, 8, 2000000),
          "mutation: 32-bit span exceeds one 64 KiB window -> rejected");
    CHECK(!detect_hi({2, 41911, 2, 41916, 2, 41915, 2, 41913},
                     {301685, 301685, 301685, 301685, 301685, 301685, 301685, 301685}, 8, kTrPool),
          "mutation: 32-bit reading has one repeated value -> rejected");
    CHECK(!detect_hi({0, 41911, 0, 41916, 0, 41915, 0, 41913},
                     {301685, 301684, 301683, 301686, 301687, 301688, 301689, 301690}, 8, kTrPool),
          "mutation: zero high half is the other detector's case -> rejected here");
    CHECK(!detect_hi({2, 41911, 2, 41916}, {301685, 301684, 301683, 301686}, 4, kTrPool),
          "mutation: too few samples for 'constant parity' to mean anything -> rejected");
    // The absolute ceiling, independent of the caller's bound: a constant of 400 implies indices near
    // 26.2M, which no real mesh reaches. Without kMaxPlausibleIndex a large enough bound accepts it.
    {
        std::vector<uint16_t> huge;
        for (uint16_t i = 0; i < 64; i++) { huge.push_back(1000 + i); huge.push_back(400); }
        CHECK(!detect_alias(huge, 64, 1u << 30),
              "mutation: implied indices past the absolute ceiling -> rejected even with a huge bound");
    }
    CHECK(!detect_hi({2, 41911, 2, 41916, 2, 41915, 2, 41913},
                     {301685, 301684, 301683, 301686, 301687, 301688, 301689, 301690}, 8, 301686),
          "mutation: one index at or past the vertex bound -> rejected");
    CHECK(!detect_hi({2, 41911, 2, 41916, 2, 41915, 2, 41913},
                     {301685, 301684, 301683, 301686, 301687, 301688, 301689, 301690}, 8, 0),
          "mutation: no vertex bound available -> declines rather than guesses");

    // R4: the 64-sample cap is load-bearing, and this is the arm that says so. A real draw whose
    // indices straddle a 64 KiB boundary carries TWO high halves; the cap means only the first 64
    // entries are examined, so the run stays classifiable. Scanning the whole buffer instead would see
    // both constants, fail the parity clause, and reject a genuine 32-bit buffer -- so changing
    // min(n,64) to n must break this arm.
    {
        std::vector<uint16_t> straddle;
        for (uint16_t i = 0; i < 64; i++) { straddle.push_back(100 + i); straddle.push_back(2); }
        for (uint16_t i = 0; i < 128; i++) { straddle.push_back(200 + i); straddle.push_back(3); }
        CHECK(detect_alias(straddle, 192, 300000),
              "32-bit run straddling a 64 KiB boundary -> still 32-bit (this is why the cap exists)");
    }

    // The two detectors must not overlap, so the call site's ordering can never change an existing
    // verdict.
    CHECK(!detect_hi({0, 0, 1, 0, 2, 0, 17, 0}, {0, 1, 2, 17, 0, 1, 2, 17}, 8, kTrPool),
          "DOLL zero-high-half quad -> declined here, still owned by the original detector");

    printf(fails ? "FAILED (%d)\n" : "PASSED\n", fails);
    return fails ? 1 : 0;
}
