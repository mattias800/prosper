// The wording of the two Wave64 lines a person reads to answer "is this my GPU's fault?".
//
// Formatter tests only: they pin what the lines SAY. That the renderer prints the host line at
// start-up, with this device's real facts, is test_wave64_host_announcement's job.
#include "diagnostics/perf/wave64_refusal.hpp"
#include <gtest/gtest.h>
#include <cstddef>
#include <iterator>
#include <string>

using namespace prosper::diagnostics::perf;

namespace {
bool has(const std::string& text, const char* needle) {
    return text.find(needle) != std::string::npos;
}

FragmentWave64Host host(uint32_t min_size, uint32_t max_size, bool size_control = true,
                        bool fragment_required_size = true, bool fragment_subgroups = true) {
    return {size_control, fragment_required_size, fragment_subgroups, min_size, max_size};
}
} // namespace

TEST(Wave64HostLine, A64LaneDeviceIsReportedNativeWithoutAWarning) {
    const std::string line = fragment_wave64_host_line(host(32, 64));
    EXPECT_TRUE(has(line, "[render] guest Wave64 fragment programs: NATIVE")) << line;
    EXPECT_TRUE(has(line, "range 32..64")) << line;
    EXPECT_FALSE(has(line, "WARNING")) << line;
    EXPECT_EQ(line.back(), '\n');
}

TEST(Wave64HostLine, A32LaneDeviceIsWarnedInPlainWords) {
    const std::string line = fragment_wave64_host_line(host(32, 32));
    EXPECT_TRUE(has(line, "[render] WARNING: this GPU cannot run PS5 Wave64 fragment programs"))
        << line;
    EXPECT_TRUE(has(line, "fragment subgroups 32..32")) << line;
    // What a player will see, and where to look for which shaders.
    EXPECT_TRUE(has(line, "draws DROPPED")) << line;
    EXPECT_TRUE(has(line, "geometry or lighting")) << line;
    EXPECT_TRUE(has(line, "[wave64-unsupported] refusal=fragment/subgroup-contract")) << line;
    EXPECT_FALSE(has(line, "NATIVE")) << line;
    EXPECT_EQ(line.back(), '\n');
}

TEST(Wave64HostLine, AWideRangeIsNotEnoughWithoutEveryOtherFact) {
    // 64 lanes in range, but the fragment stage cannot be pinned to them (or cannot use subgroup
    // operations at all): the renderer's admission test refuses, so this must warn too.
    for (const FragmentWave64Host& partial :
         {host(32, 64, /*size_control=*/false),
          host(32, 64, true, /*fragment_required_size=*/false),
          host(32, 64, true, true, /*fragment_subgroups=*/false), host(8, 8), host(128, 128)}) {
        const std::string line = fragment_wave64_host_line(partial);
        EXPECT_FALSE(partial.native());
        EXPECT_TRUE(has(line, "WARNING")) << line;
    }
    // The warning names which of the facts is missing.
    EXPECT_TRUE(has(fragment_wave64_host_line(host(32, 64, false)), "exact-size-control=0"));
    EXPECT_TRUE(has(fragment_wave64_host_line(host(32, 64, true, false)), "fragment-exact-size=0"));
    EXPECT_TRUE(
        has(fragment_wave64_host_line(host(32, 64, true, true, false)), "fragment-subgroup-ops=0"));
}

TEST(Wave64RefusalLine, ARecompileRefusalSaysTheHostWasNotConsulted) {
    for (Wave64Refusal site : {Wave64Refusal::FragmentRecompile, Wave64Refusal::ComputeRecompile}) {
        const std::string line = wave64_refusal_line(site, 0x4112ac0100, 0, UINT32_MAX, 0, 0, {});
        EXPECT_TRUE(has(line, "host-subgroups=not-consulted")) << line;
        EXPECT_TRUE(has(line, "wave-reasons=not-consulted")) << line;
        EXPECT_TRUE(has(line, "NOT a host limit")) << line;
        EXPECT_FALSE(has(line, "unavailable")) << line;
    }
    const std::string fragment = wave64_refusal_line(Wave64Refusal::FragmentRecompile, 0x4112ac0100,
                                                     0, UINT32_MAX, 0, 0, {});
    EXPECT_TRUE(has(fragment, "[wave64-unsupported] stage=fragment program=0x4112ac0100 "))
        << "the prefix the census tools key on is unchanged: " << fragment;
    EXPECT_TRUE(has(fragment, "refusal=fragment/recompile")) << fragment;
}

TEST(Wave64RefusalLine, ASubgroupRefusalStillNamesTheHostRange) {
    const std::string known =
        wave64_refusal_line(Wave64Refusal::FragmentSubgroup, 0x4071, 0x40710001, 2, 32, 32, {});
    EXPECT_TRUE(has(known, "refusal=fragment/subgroup-contract")) << known;
    EXPECT_TRUE(has(known, "host-subgroups=32..32")) << known;
    EXPECT_TRUE(has(known, "wave-reasons=0x2")) << known;
    EXPECT_FALSE(has(known, "not-consulted")) << known;
    // A subgroup refusal whose host range really is missing keeps the old word: there the host
    // WAS the question, and the answer is absent.
    const std::string missing =
        wave64_refusal_line(Wave64Refusal::ComputeSubgroup, 0x4072, 0, UINT32_MAX, 0, 0, {});
    EXPECT_TRUE(has(missing, "host-subgroups=unavailable")) << missing;
    EXPECT_FALSE(has(missing, "not-consulted")) << missing;
}

// ADR 0028's route= field. The vocabulary is named before the routes exist, so the names are pinned
// here: a census tool keys on them.
TEST(Wave64RouteField, TheVocabularyIsFixedAndReservedNamesExist) {
    const char* const expected[] = {"native",  "proven-width-independent", "workgroup-exchange",
                                    "n-lanes", "fragment-promoted",        "owned-wave", "refused"};
    ASSERT_EQ(kWave64RouteCount, std::size(expected));
    for (size_t i = 0; i < kWave64RouteCount; ++i)
        EXPECT_STREQ(wave64_route_name(static_cast<Wave64Route>(i)), expected[i]);
    EXPECT_STREQ(wave64_route_name(Wave64Route::Count), "unknown");
}

TEST(Wave64RouteField, EveryRefusalLineSaysRouteRefused) {
    for (size_t i = 0; i < kWave64RefusalCount; ++i) {
        const Wave64Refusal site = static_cast<Wave64Refusal>(i);
        const std::string line = wave64_refusal_line(site, 0x4071, 0x40710001, 2, 32, 32, {});
        EXPECT_TRUE(has(line, "route=refused guest-wave=64 ")) << line;
        const std::string refusal =
            std::string("refusal=") + kWave64RefusalNames[i] + " route=refused ";
        EXPECT_TRUE(has(line, refusal.c_str())) << "route= follows refusal=: " << line;
    }
}

TEST(Wave64RouteField, AnAdmittedProgramIsNotTaggedUnsupported) {
    const std::string proven =
        wave64_route_line(Wave64Route::ProvenWidthIndependent, false, 0x4071, 0x40710001);
    EXPECT_TRUE(has(proven, "[wave64-route] stage=fragment program=0x4071 identity=0x40710001 "
                            "route=proven-width-independent guest-wave=64\n"))
        << proven;
    EXPECT_FALSE(has(proven, "unsupported")) << proven;
    const std::string native = wave64_route_line(Wave64Route::Native, true, 0x4072, 0);
    EXPECT_TRUE(has(native, "stage=compute ")) << native;
    EXPECT_TRUE(has(native, "route=native ")) << native;
}

TEST(Wave64RouteField, ARouteThatDoesNotExistYetPrintsNothing) {
    for (Wave64Route reserved : {Wave64Route::WorkgroupExchange, Wave64Route::NLanes,
                                 Wave64Route::FragmentPromoted, Wave64Route::OwnedWave,
                                 Wave64Route::Refused})
        EXPECT_TRUE(wave64_route_line(reserved, false, 1, 2).empty())
            << wave64_route_name(reserved);
}
