// A real exact64 dispatch must preserve the initialized low half and arbitrary old high half.
#include <cstdint>
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "fixtures/arbitrary_placeholder.hpp"
#include "fixtures/render_runner.h"
#include "shared/live/live_compute.hpp"
#include <gtest/gtest.h>
#include <memory>

namespace fixture = prosper::test::arbitrary_placeholder;
namespace live = prosper::frontend;
using namespace prosper::gpu;

namespace {
void execute(const fixture::Words& code, const fixture::Words& expected, uint64_t address,
             const ComputeShaderConfig& launch = fixture::config()) {
    ASSERT_TRUE(prosper::test::render_vk_ctx().ok);
    const auto shared = shared_vulkan_context();
    ASSERT_TRUE(shared.valid());
    if (select_native_compute_subgroup_size(shared, launch, false, false) != 64)
        GTEST_SKIP() << "enabled exact64 subgroup contract unavailable";
    live::execute_live_compute_items({});
    const auto device = live::live_compute_storage_image_device();
    ASSERT_TRUE(device.initialized);
    ASSERT_TRUE(device.adopted);
    fixture::Words output(64, 0xdeadbeefu);
    auto resources = fixture::table(reinterpret_cast<uint64_t>(output.data()));
    ComputeItem item;
    item.code_addr = address;
    item.spirv = recompile_compute(code.data(), code.size(), &resources, launch,
                                   {RecompileDiagnosticStage::Compute, item.code_addr});
    ASSERT_FALSE(item.spirv.empty()) << last_terminal_reject_reason(item.code_addr);
    item.required_subgroup_size = 64;
    item.resources = std::make_shared<ShaderResourceTable>(resources);
    item.user_sgprs = launch.user_sgprs;
    item.recompile_config_available = true;
    item.recompile_config = launch;
    item.launch.local_x = item.launch.threads_x = 64;
    item.launch.local_y = item.launch.local_z = item.launch.threads_y = item.launch.threads_z = 1;
    item.launch.groups_x = item.launch.groups_y = item.launch.groups_z = 1;
    const auto cpu = live::live_compute_cpu_fill_dispatches();
    ASSERT_TRUE(live::execute_live_compute_items({item}));
    EXPECT_EQ(live::live_compute_cpu_fill_dispatches(), cpu);
    EXPECT_EQ(output, expected);
}
} // namespace

TEST(ArbitraryPlaceholderGpu, TheActualWaveKeepsBothHalvesOfTheRetainedValue) {
    execute(fixture::narrowed_writer(), fixture::expected(true), 0x5497b00200ull);
}

TEST(ArbitraryPlaceholderGpu, CmpxKeepsExactlyOneLogicalLaneActive) {
    execute(fixture::cmpx_predicate(), fixture::cmpx_expected(), 0x5497b00600ull);
}

TEST(ArbitraryPlaceholderGpu, TheScalarPairLoopKeepsTheCurrentValueInBothHalves) {
    execute(fixture::scalar_loop_relay(), fixture::Words(64, 22), 0x5497b00700ull);
}

TEST(ArbitraryPlaceholderGpu, MissingReadfirstPublicationsStayHardInStraightAndLoopPrograms) {
    for (bool loop : {false, true}) {
        SCOPED_TRACE(loop ? "loop" : "straight");
        const auto words =
            loop ? fixture::readfirst_loop_relay() : fixture::readfirst_straight_relay();
        const auto resources = fixture::table();
        const uint64_t address = 0x5497b00c00ull + loop;
        const auto module =
            recompile_compute(words.data(), words.size(), &resources, fixture::config(),
                              {RecompileDiagnosticStage::Compute, address});
        EXPECT_TRUE(module.empty());
        if (module.empty()) {
            EXPECT_NE(
                last_terminal_reject_reason(address).find("origin=unmodelled-scalar-publication"),
                std::string::npos);
            continue;
        }
        // Mutation control: any admitted module must still broadcast the first active lane.
        execute(words, loop ? fixture::Words(64, 7) : fixture::cmpx_expected(), address);
    }
}

TEST(ArbitraryPlaceholderGpu, AnUnloadedDataReadAfterThePredicateStillStoresEveryLane) {
    execute(fixture::data_after_predicate(), fixture::Words(64, 0), 0x5497b01000ull);
}

TEST(ArbitraryPlaceholderGpu, AnUnloadedReadlaneResultPublishesTheSameValueAcrossTheWave) {
    execute(fixture::readlane_relay(), fixture::expected(false), 0x5497b01100ull);
}

TEST(ArbitraryPlaceholderGpu, AnUnloadedImplicitScalarCompareKeepsTheSameWaveBranch) {
    execute(fixture::implicit_compare_after_predicate(), fixture::Words(64, 22), 0x5497b01300ull);
}

TEST(ArbitraryPlaceholderGpu, AnActuallyWrittenUnloadedSpillKeepsTheSameWordAcrossTheWave) {
    execute(fixture::written_unloaded_spill(), fixture::Words(64, 0), 0x5497b01600ull);
}

TEST(ArbitraryPlaceholderGpu, AProvidedAddtidStoreKeepsItsRealLdsData) {
    auto launch = fixture::config();
    launch.user_sgprs[0] = 13;
    launch.lds_bytes = 256;
    execute(fixture::addtid_after_writelane(false), fixture::Words(64, 7), 0x5497b02200ull, launch);
}

TEST(ArbitraryPlaceholderGpu, ARelativeBankCannotHideALostProvidedCandidate) {
    const auto words = fixture::relative_after_writelane();
    const auto resources = fixture::table();
    auto launch = fixture::config();
    launch.user_sgprs[0] = 13;
    launch.user_sgprs[4] = 1;
    constexpr uint64_t address = 0x5497b03300ull;
    const auto module = recompile_compute(words.data(), words.size(), &resources, launch,
                                          {RecompileDiagnosticStage::Compute, address});
    EXPECT_TRUE(module.empty());
    if (module.empty()) {
        EXPECT_NE(last_terminal_reject_reason(address).find("origin=lost-definition word=v6"),
                  std::string::npos);
    } else {
        auto expected = fixture::Words(64, 7);
        expected[0] = 13;
        execute(words, expected, address, launch);
    }
}

TEST(ArbitraryPlaceholderGpu, AProvidedRelativeBankRetainsTheRuntimeSelectedValue) {
    auto launch = fixture::config();
    launch.user_sgprs[4] = 1;
    execute(fixture::relative_after_writelane(false), fixture::Words(64, 7), 0x5497b03400ull,
            launch);
}

TEST(ArbitraryPlaceholderGpu, AnUnloadedRelativeCandidateKeepsItsConsistentValue) {
    auto launch = fixture::config();
    launch.user_sgprs[4] = 1;
    execute(fixture::relative_after_writelane(false, false), fixture::Words(64, 0), 0x5497b03500ull,
            launch);
}

TEST(ArbitraryPlaceholderGpu, AnEarlyAddtidStoreCannotReplaceAWritelaneValueWithZero) {
    const auto words = fixture::addtid_after_writelane();
    const auto resources = fixture::table();
    auto launch = fixture::config();
    launch.user_sgprs[0] = 13;
    launch.lds_bytes = 256;
    constexpr uint64_t address = 0x5497b02300ull;
    const auto module = recompile_compute(words.data(), words.size(), &resources, launch,
                                          {RecompileDiagnosticStage::Compute, address});
    EXPECT_TRUE(module.empty());
    if (module.empty()) {
        EXPECT_NE(last_terminal_reject_reason(address).find("origin=lost-definition word=v6"),
                  std::string::npos);
    } else {
        // An unsafe guard-removal mutation must be detected by an actual exact64 dispatch.
        fixture::Words expected(64, 7);
        expected[0] = 13;
        execute(words, expected, address, launch);
    }
}

TEST(ArbitraryPlaceholderGpu, APartialWriteKeepsTheProvidedInactiveLanes) {
    execute(fixture::partial_old_after_writelane(false), fixture::partial_old_expected(),
            0x5497b02500ull);
}

TEST(ArbitraryPlaceholderGpu, APartialWriteCannotInventTheLostInactiveDestination) {
    const auto words = fixture::partial_old_after_writelane();
    const auto resources = fixture::table();
    auto launch = fixture::config();
    launch.user_sgprs[0] = 13;
    constexpr uint64_t address = 0x5497b02600ull;
    const auto module = recompile_compute(words.data(), words.size(), &resources, launch,
                                          {RecompileDiagnosticStage::Compute, address});
    EXPECT_TRUE(module.empty());
    if (module.empty()) {
        EXPECT_NE(last_terminal_reject_reason(address).find("origin=lost-definition word=v6"),
                  std::string::npos);
    } else {
        execute(words, fixture::partial_old_expected(), address, launch);
    }
}
