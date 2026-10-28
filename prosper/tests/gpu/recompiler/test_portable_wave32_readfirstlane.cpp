// A refused Wave32 scalar publication needs a real first-active-lane broadcast, not lane-local DATA.
// Missing launch/spill inputs remain Hard even beside an independently admitted unloaded predicate.
#include "bpermute_spirv_oracle.hpp"
#include "fixtures/arbitrary_placeholder.hpp"
#include "fixtures/checked_compute_launch.hpp"
#include "fixtures/compute_runner.h"
#include "gpu/recompiler/rdna2_arbitrary_placeholder.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace prosper::gpu;
namespace fixture = prosper::test::arbitrary_placeholder;
struct Shape {
    uint32_t local = 64, first = 5, destination = 4;
    bool admitted_predicate = true;
};
ComputeShaderConfig config(Shape shape) {
    auto result = fixture::config();
    result.local_x = shape.local;
    result.wave_size = 32;
    result.native_subgroup_size = 0;
    prosper::test::complete_compute_launch(result);
    return result;
}
fixture::Words program(Shape shape) {
    fixture::Words words;
    if (shape.admitted_predicate) {
        words = fixture::scalar_relay();
        words.pop_back();   // retain the live predicate and its real store, then continue
    }
    const fixture::Words suffix{
        0x7d860080u | shape.first,   // VCC_LO = (first <= local_id.x)
        0xbefe036au,   // EXEC_LO = VCC_LO
        0x7e000500u | (shape.destination << 17u),   // sDST = readfirstlane(v0)
        0xbefe03c1u,   // restore full EXEC_LO
        0x7e020200u | shape.destination,   // v1 = scalar result, including EXEC-off recipients
        0xe0702000u,
        0x80020100u,
        0xbf810000u,
    };
    words.insert(words.end(), suffix.begin(), suffix.end());
    return words;
}
struct Compilation {
    fixture::Words module;
    std::string reason;
};
Compilation compile(const fixture::Words& words, Shape shape) {
    auto resources = fixture::table();
    resources.resources[0].size = shape.local * sizeof(uint32_t);
    static std::atomic<uint64_t> next{0x5039320000ull};
    const auto address = next.fetch_add(0x100);
    auto module = recompile_compute(words.data(), words.size(), &resources, config(shape),
                                    {RecompileDiagnosticStage::Compute, address});
    return {std::move(module), last_terminal_reject_reason(address)};
}
fixture::Words expected(Shape shape) {
    fixture::Words result(shape.local);
    for (uint32_t lane = 0; lane < shape.local; ++lane) {
        const uint32_t base = lane / 32 * 32, end = std::min(base + 32, shape.local);
        result[lane] = shape.first < end ? std::max(base, shape.first) : base;
    }
    return result;
}
void interpreter(Shape shape) {
    SCOPED_TRACE(shape.local);
    SCOPED_TRACE(shape.first);
    SCOPED_TRACE(shape.destination);
    const auto result = compile(program(shape), shape);
    ASSERT_FALSE(result.module.empty()) << result.reason;
    ASSERT_GT(arbitrary_placeholder_reads(result.module).fresh_mask, 0u);
    EXPECT_FALSE(compute_spirv_uses_group_non_uniform(result.module));
    bpermute_oracle::Interpreter vm(result.module);
    vm.native_subgroup_width = 64;   // portable guest Wave32s share a wider physical host domain
    vm.supplied_push_constants = config(shape).user_sgprs;
    const auto actual = vm.run(shape.local, fixture::Words(shape.local, 0xdeadbeefu));
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    EXPECT_EQ(actual, expected(shape));
}
fixture::Words execute(const fixture::Words& words, Shape shape) {
    const auto result = compile(words, shape);
    EXPECT_FALSE(result.module.empty()) << result.reason;
    if (result.module.empty()) return {};
    bpermute_oracle::Interpreter vm(result.module);
    vm.native_subgroup_width = 64;
    vm.supplied_push_constants = config(shape).user_sgprs;
    auto actual = vm.run(shape.local, fixture::Words(shape.local, 0xdeadbeefu));
    EXPECT_TRUE(vm.error.empty()) << vm.error;
    return actual;
}
fixture::Words mask_reader(bool implicit, uint32_t mask) {
    return implicit ? fixture::Words{0x7e040281u, (1u << 25u) | (1u << 17u) | (2u << 9u) | 128u}
                    : fixture::Words{0xd5010001u, 128u | (129u << 9u) | (mask << 18u)};
}
void replace_numeric_read(fixture::Words& words, const fixture::Words& replacement) {
    words.erase(words.end() - 4);
    words.insert(words.end() - 3, replacement.begin(), replacement.end());
}
}   // namespace

TEST(PortableWave32Readfirstlane, EveryReplicaReceivesItsGuestWavesFirstActiveValue) {
    for (uint32_t local : {16u, 32u, 48u, 64u, 128u})
        for (uint32_t first : {0u, 5u, 31u, 32u, 37u, 63u, 64u})
            interpreter({.local = local, .first = first});
}
TEST(PortableWave32Readfirstlane, NumericVccWordsAreRealScalarDestinations) {
    for (uint32_t destination : {106u, 107u})
        for (uint32_t first : {5u, 32u, 64u})
            interpreter({.first = first, .destination = destination});
}
TEST(PortableWave32Readfirstlane, GuestWaveBoundariesAndEmptyExecExecuteOnTheGpu) {
    for (uint32_t local : {16u, 32u, 48u, 64u, 128u})
        for (uint32_t first : {5u, 37u, 64u}) {
            const Shape shape{.local = local, .first = first};
            SCOPED_TRACE(local);
            SCOPED_TRACE(first);
            const auto result = compile(program(shape), shape);
            ASSERT_FALSE(result.module.empty()) << result.reason;
            std::vector<uint32_t> actual;
            prosper::test::run_compute(result.module, {0.0f}, local, 1, {},
                                       fixture::Words(local, 0xdeadbeefu), &actual, local, nullptr,
                                       nullptr, nullptr, 0, nullptr, config(shape).user_sgprs);
            EXPECT_EQ(actual, expected(shape));
        }
}
TEST(PortableWave32Readfirstlane, UnknownLaunchAndMissingSpillStayRefused) {
    const Shape shape;
    auto launch = config(shape);
    launch.dispatch_launch = {};
    const auto words = program(shape);
    const auto resources = fixture::table();
    EXPECT_TRUE(recompile_compute(words.data(), words.size(), &resources, launch).empty());
    const auto spilled = compile(fixture::with_missing_spill_root(words, 60), shape);
    EXPECT_TRUE(spilled.module.empty());
    EXPECT_NE(spilled.reason.find("origin=spill"), std::string::npos) << spilled.reason;
}

TEST(PortableWave32Readfirstlane, NumericVccPublicationDoesNotInventAMaskDomain) {
    const Shape shape{.destination = 106};
    for (bool implicit : {false, true}) {
        auto words = program(shape);
        replace_numeric_read(words, mask_reader(implicit, 106));
        const auto result = compile(words, shape);
        EXPECT_TRUE(result.module.empty()) << implicit;
        EXPECT_NE(result.reason.find("missing-wave32-vcc-mask"), std::string::npos)
            << result.reason;
    }
}

TEST(PortableWave32Readfirstlane, HighWordDestinationPreservesTheIndependentLowMask) {
    const Shape shape{.destination = 107};
    auto words = program(shape);
    replace_numeric_read(words, mask_reader(true, 106));
    fixture::Words want(shape.local);
    for (uint32_t lane = 0; lane < shape.local; ++lane) want[lane] = lane >= shape.first;
    EXPECT_EQ(execute(words, shape), want);
}

TEST(PortableWave32Readfirstlane, DestinationPreservesAnAdjacentSingleWordSavedMask) {
    const Shape shape{.destination = 21};
    auto words = program(shape);
    words.insert(words.end() - 8, {0x7d8800f9u, 0x06869490u});   // s20 = (16 > local_id.x)
    replace_numeric_read(words, mask_reader(false, 20));
    fixture::Words want(shape.local);
    for (uint32_t lane = 0; lane < shape.local; ++lane) want[lane] = lane < 16;
    EXPECT_EQ(execute(words, shape), want);
}

TEST(PortableWave32Readfirstlane, ASingleWordWriteDoesNotResurrectAnOverlappingPairMask) {
    const Shape shape{.destination = 21};
    auto words = program(shape);
    words.insert(words.end() - 8, {0xbe94047eu});   // saved B64 EXEC at s[20:21]
    replace_numeric_read(words, mask_reader(false, 20));
    EXPECT_TRUE(compile(words, shape).module.empty());
}

TEST(PortableWave32Readfirstlane, AnUntrackedSlotIsNotAnOrdinaryBroadcastSource) {
    const Shape shape;
    auto words = program(shape);
    words.insert(words.begin() + 8, {0xd7610002u, 128u | (128u << 9u)});   // v2[0] = 0
    words[10] = 0x7e080502u;   // READFIRST s4,v2 under EXEC that starts at lane5
    EXPECT_TRUE(compile(words, shape).module.empty());
}
