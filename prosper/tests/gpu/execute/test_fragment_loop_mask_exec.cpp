// A scalar pair becomes a loop-carried predicate. Two outer iterations visit disjoint lane
// parities; each inner iteration finishes lanes after 1..4 visits. The accumulated predicate
// must retain lanes that already finished the inner loop AND the previous outer iteration.
// Both loops terminate independently of the accumulator being tested.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "fixtures/render_runner.h"
#include "fixtures/spirv_triangle.h"
#include <cstdio>
#include <iterator>
#include <vector>

using namespace prosper::gpu;
static constexpr uint32_t program[] = {
    0xbed40380u, 0xbed50380u, 0xbed8047eu, 0xd7650007u, 0x000100c1u, 0xd7660007u,
    0x00020ec1u, 0x36120e83u, 0x4a121281u, 0x36160e81u, 0x7e140280u, 0xbf880019u,
    0x7e100280u, 0xd4c2005cu, 0x0002150bu, 0xbefe045cu, 0xbed6047eu, 0xbf88000du,
    0xd4c6005cu, 0x00010109u, 0x8ac47e54u, 0xbeea045cu, 0x88d4445cu, 0xbefe045cu,
    0xbf860006u, 0x4a101081u, 0x7d8c1308u, 0x8afe6a5cu, 0x8aea5c54u, 0x88d46a5cu,
    0xbf89fff2u, 0xbefe0458u, 0x4a141481u, 0xd4c1006au, 0x0001050au, 0xbefe046au,
    0xbf89ffe6u, 0xbefe0458u, 0x87ea5854u, 0xd5010000u, 0x01a9e480u, 0x7e0202f2u,
    0x7e040280u, 0x7e0602f2u, 0xf800080fu, 0x03020100u, 0xbf810000u,
};

static constexpr uint32_t zero_trip[] = {
    0xbed40380u, 0xbed503c1u, 0xbed8047eu, 0xd7650007u, 0x000100c1u, 0xd7660007u,
    0x00020ec1u, 0x36120e83u, 0x4a121281u, 0x36160e81u, 0x7e140282u, 0xbf880019u,
    0x7e100280u, 0xd4c2005cu, 0x0002150bu, 0xbefe045cu, 0xbed6047eu, 0xbf88000du,
    0xd4c6005cu, 0x00010109u, 0x8ac47e54u, 0xbeea045cu, 0x88d4445cu, 0xbefe045cu,
    0xbf860006u, 0x4a101081u, 0x7d8c1308u, 0x8afe6a5cu, 0x8aea5c54u, 0x88d46a5cu,
    0xbf89fff2u, 0xbefe0458u, 0x4a141481u, 0xd4c1006au, 0x0001050au, 0xbefe046au,
    0xbf89ffe6u, 0xbefe0458u, 0x87ea5854u, 0xd501000cu, 0x01a90280u, 0xd4c6006au,
    0x00014107u, 0xd5010002u, 0x01a9e480u, 0xd501000du, 0x01a90280u, 0x7d841b0cu,
    0xd5010000u, 0x01a9e480u, 0x7e0202f2u, 0x7e0602f2u, 0xf800080fu, 0x03020100u,
    0xbf810000u,
};

int main() {
    int failures = 0;
    const auto check = [&](bool ok, const char* message) {
        std::printf("[%s] %s\n", ok ? "ok" : "FAIL", message);
        if (!ok) ++failures;
    };
    const auto fragment = recompile_fragment(program, std::size(program), nullptr, nullptr,
        UINT32_MAX, nullptr, false, {RecompileDiagnosticStage::Fragment, 0xa2790001});
    check(!fragment.empty(), "defined scalar accumulator compiles through nested loops");
    if (fragment.empty()) return 1;
    check(fragment_spirv_required_subgroup_size(fragment) == 64,
          "scalar-bit projection requires exact fragment wave64");
    bool population_scan = false;
    for (size_t i = 5; i < fragment.size();) {
        const uint32_t words = fragment[i] >> 16;
        if (!words || words > fragment.size() - i) return 1;
        // OpGroupNonUniformIAdd is 349. The only count in this fixture is MBCNT(-1):
        // its answer is a physical lane position, never a scan of the active population.
        if ((fragment[i] & 0xffffu) == 349) population_scan = true;
        i += words;
    }
    check(!population_scan, "all-ones lane-ID calculation does not count active invocations");

    const auto& ctx = prosper::test::render_vk_ctx();
    const bool supported = ctx.subgroup_size_control && ctx.min_subgroup_size <= 64 &&
        ctx.max_subgroup_size >= 64 &&
        (ctx.required_subgroup_size_stages & VK_SHADER_STAGE_FRAGMENT_BIT) &&
        (ctx.subgroup_stages & VK_SHADER_STAGE_FRAGMENT_BIT) &&
        (ctx.subgroup_operations & VK_SUBGROUP_FEATURE_VOTE_BIT) &&
        (ctx.subgroup_operations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT);
    if (!supported) {
        std::puts("SKIP: device cannot execute the required fragment wave64 contract");
        return failures ? 1 : 77;
    }
    const std::vector<uint32_t> vertex(std::begin(kTriVertSpv), std::end(kTriVertSpv));
    const uint32_t yellow[] = {0x7e0002f2, 0x7e0202f2, 0x7e040280, 0x7e0602f2,
                              0xf800180f, 0x03020100, 0xbf810000};
    const auto reference_fragment = recompile_fragment(yellow, std::size(yellow));
    const auto expected = prosper::test::render_triangle_rgba(vertex, reference_fragment, 64, 64);
    const auto actual = prosper::test::render_triangle_rgba(vertex, fragment, 64, 64);
    check(expected.size() == 64 * 64 * 4, "reference triangle rendered");
    unsigned covered = 0, mismatches = 0;
    for (size_t i = 0; i + 3 < expected.size(); i += 4) {
        if (expected[i] > 200 && expected[i + 1] > 200 && expected[i + 2] < 20) ++covered;
    }
    check(covered > 256, "reference contains a substantial yellow triangle");
    if (actual.size() == expected.size()) {
        for (size_t i = 0; i < actual.size(); ++i) if (actual[i] != expected[i]) ++mismatches;
    }
    check(actual.size() == expected.size() && mismatches == 0,
          "both loop backedges preserve every completed lane's accumulator bit");
    const auto zero_fragment = recompile_fragment(zero_trip, std::size(zero_trip), nullptr,
        nullptr, UINT32_MAX, nullptr, false, {RecompileDiagnosticStage::Fragment, 0xa2790002});
    check(!zero_fragment.empty(), "zero-trip accumulator compiles");
    const auto zero_pixels = zero_fragment.empty() ? std::vector<uint8_t>{}
        : prosper::test::render_triangle_rgba(vertex, zero_fragment, 64, 64);
    unsigned low_lanes = 0, high_lanes = 0, bad_seed = 0;
    if (zero_pixels.size() == expected.size()) {
        for (size_t i = 0; i + 3 < expected.size(); i += 4) {
            if (expected[i] <= 200 || expected[i + 1] <= 200) continue;
            if (zero_pixels[i + 2] > 200) ++high_lanes;
            else if (zero_pixels[i + 2] < 20) ++low_lanes;
            if (zero_pixels[i] < 200 || zero_pixels[i + 1] < 200) ++bad_seed;
        }
    }
    check(low_lanes > 0 && high_lanes > 0,
          "independent lane-id witness exercised covered lanes in both physical halves");
    check(zero_pixels.size() == expected.size() && bad_seed == 0,
          "zero inner iterations preserve distinct loaded low and high mask words");
    std::printf("low_lanes=%u high_lanes=%u bad_seed=%u\n", low_lanes, high_lanes, bad_seed);
    std::printf("covered=%u mismatched_bytes=%u\n", covered, mismatches);
    return failures ? 1 : 0;
}
