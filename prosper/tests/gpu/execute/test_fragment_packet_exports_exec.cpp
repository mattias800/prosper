#include "fixtures/compute_runner.h"
#include "fixtures/fragment_packet_exports_fixture.hpp"
#include <gtest/gtest.h>

namespace {
using namespace prosper::gpu;
namespace f = prosper::test::fragment_packet_exports;
struct Execution {
    FragmentPacketWaveBatch batch;
    FragmentPacketWaveResult result;
    std::vector<uint32_t> words;
    bool unsupported = false;
};
Execution execute(uint32_t family, bool missing = false, uint32_t en = 15) {
    Execution result;
    prosper::test::ComputeOwnedDispatch owner;
    owner.prepare = [&](const prosper::test::ComputeEnabledContract& enabled,
                        prosper::test::ComputeOwnedPlan& plan) {
        if (!enabled.shader_int64_enabled) {
            result.unsupported = true;
            return false;
        }
        std::vector<FragmentResourcePacket> waves;
        for (uint32_t wave = 0; wave < 3; ++wave) {
            auto input = family >= 8 && family <= 14 ? f::pending_write(family - 8, true, wave)
                         : family == 15              ? f::pending_join(true, en != 0, wave)
                         : family == 1               ? f::scratch(wave, true)
                         : family == 2               ? f::multiple()
                         : family == 3               ? f::compressed(en)
                         : family == 4               ? f::missing_active(missing && wave == 2)
                         : family == 5               ? f::peer(missing && wave == 2)
                         : family == 6               ? f::wqm(missing && wave == 2)
                         : family == 7               ? f::previous_destination(missing && wave == 2)
                                                     : f::scratch(wave);
            input.device = {enabled.device_identity, enabled.shader_int64_enabled,
                            enabled.rgba32_sfloat_sampled};
            waves.push_back(std::move(input));
        }
        const auto kernel = std::make_shared<const FragmentPacketKernel>(
            recompile_fragment_packet_kernel(waves[0]));
        result.batch = pack_fragment_packet_waves(kernel, waves, f::placements(*kernel));
        if (!result.batch.rejection.empty()) return false;
        plan.spirv = kernel->program.packet.spirv;
        plan.input.resize(result.batch.input_words.size());
        std::memcpy(plan.input.data(), result.batch.input_words.data(), plan.input.size() * 4);
        plan.output_words = static_cast<uint32_t>(result.batch.output_words.size());
        plan.initial_output = result.batch.output_words;
        plan.readonly_words = {result.batch.authority, &result.batch.authority->words()};
        plan.wave_count = 3;
        return true;
    };
    const auto raw = prosper::test::run_compute({}, {}, 64, 0, {}, {}, nullptr, 64, nullptr,
                                                nullptr, nullptr, 0, &owner);
    result.words.resize(raw.size());
    if (!raw.empty()) std::memcpy(result.words.data(), raw.data(), raw.size() * 4);
    result.result = decode_fragment_packet_waves(result.batch, result.words,
                                                 owner.completion_and_host_availability,
                                                 owner.enabled.device_identity);
    std::printf("[architectural-exp-execution] family=%u missing=%u en=%u workgroups=%u "
                "attempts=%u physical_subgroup=%u\n",
                family, missing, en, owner.plan.wave_count, owner.dispatch_attempts,
                prosper::test::default_compute_subgroup_properties().size);
    if (!result.unsupported) EXPECT_EQ(owner.dispatch_attempts, 1u);
    return result;
}
TEST(FragmentPacketExportExecution, DistinctScratchWavesHaveObservedZeroAndAbsentInactive) {
    const auto actual = execute(0);
    if (actual.unsupported) GTEST_SKIP() << "No queried+enabled Int64 owner";
    ASSERT_TRUE(actual.result.rejection.empty()) << actual.result.rejection;
    ASSERT_EQ(actual.result.architectural_exports.size(), 3u);
    EXPECT_TRUE(actual.result.exports.empty());
    for (uint32_t wave = 0; wave < 3; ++wave) {
        const auto expected = f::scratch_records(wave);
        const auto offset = actual.batch.placements[wave].output_base + 128;
        EXPECT_TRUE(std::equal(expected.begin(), expected.end(), actual.words.begin() + offset));
        for (uint32_t lane = 0; lane < 64; ++lane) {
            const auto& output = actual.result.architectural_exports[wave][lane];
            EXPECT_EQ(output.events[0].source_words[0].has_value(), f::active(lane));
            EXPECT_EQ(output.commit_eligible, f::active(lane) && lane != 40);
            if (f::active(lane)) EXPECT_EQ(output.colors[0][0]->bits, f::value(wave));
        }
    }
}
TEST(FragmentPacketExportExecution, InactiveControlsStillCompleteAllWorkgroups) {
    const auto actual = execute(1);
    if (actual.unsupported) GTEST_SKIP() << "No queried+enabled Int64 owner";
    ASSERT_TRUE(actual.result.rejection.empty()) << actual.result.rejection;
    ASSERT_EQ(actual.result.architectural_exports.size(), 3u);
    for (uint32_t wave = 0; wave < 3; ++wave) {
        const auto expected = f::scratch_records(wave, true);
        EXPECT_TRUE(
            std::equal(expected.begin(), expected.end(),
                       actual.words.begin() + actual.batch.placements[wave].output_base + 128));
        for (const auto& lane : actual.result.architectural_exports[wave]) {
            EXPECT_FALSE(lane.commit_eligible);
            EXPECT_TRUE(lane.terminal_done);
            EXPECT_FALSE(lane.events[0].source_words[0].has_value());
        }
    }
}
TEST(FragmentPacketExportExecution, OriginalChannelsNullAndLastVmSurviveLaterExecChange) {
    const auto actual = execute(2);
    if (actual.unsupported) GTEST_SKIP() << "No queried+enabled Int64 owner";
    ASSERT_TRUE(actual.result.rejection.empty()) << actual.result.rejection;
    ASSERT_EQ(actual.result.architectural_exports.size(), 3u);
    for (const auto& wave : actual.result.architectural_exports)
        for (uint32_t lane = 0; lane < 64; ++lane) {
            const auto& out = wave[lane];
            ASSERT_EQ(out.events.size(), 4u);
            EXPECT_TRUE(out.events.back().exec);
            EXPECT_EQ(out.valid_mask, f::active(lane));
            EXPECT_EQ(out.commit_eligible, f::active(lane) && lane != 40);
            EXPECT_EQ(out.colors[2][0]->bits, 0x80010000u + lane);
            EXPECT_EQ(out.colors[2][1]->bits, 0x80020000u + lane);
            EXPECT_EQ(out.colors[2][2].has_value(), f::active(lane));
            if (f::active(lane)) EXPECT_EQ(out.colors[2][2]->bits, 0x80030000u + lane);
            EXPECT_FALSE(out.colors[2][3].has_value());
        }
}
TEST(FragmentPacketExportExecution, CompressedRgbaUsesOriginalPackedSourcePair) {
    for (uint32_t en : {3u, 12u, 15u}) {
        const auto actual = execute(3, false, en);
        if (actual.unsupported) GTEST_SKIP() << "No queried+enabled Int64 owner";
        ASSERT_TRUE(actual.result.rejection.empty()) << actual.result.rejection;
        ASSERT_EQ(actual.result.architectural_exports.size(), 3u);
        for (const auto& wave : actual.result.architectural_exports)
            for (uint32_t lane = 0; lane < 64; ++lane)
                for (uint32_t channel = 0; channel < 4; ++channel) {
                    const auto& component = wave[lane].colors[3][channel];
                    const bool observed = f::active(lane) && (en & (1u << channel));
                    EXPECT_EQ(component.has_value(), observed);
                    if (observed) {
                        const uint32_t packed = (channel < 2 ? 0x12348000u : 0x7fff0000u) + lane;
                        EXPECT_EQ(component->bits, (packed >> ((channel & 1u) * 16)) & 0xffffu);
                        EXPECT_EQ(component->width, 16u);
                    }
                }
    }
}
TEST(FragmentPacketExportExecution, RealWaitProtectsOriginalPayloadAgainstEveryExecWriter) {
    for (uint32_t family = 0; family < 7; ++family) {
        const auto actual = execute(8 + family);
        if (actual.unsupported) GTEST_SKIP() << "No queried+enabled Int64 owner";
        ASSERT_TRUE(actual.result.rejection.empty()) << actual.result.rejection;
        ASSERT_EQ(actual.result.architectural_exports.size(), 3u);
        for (uint32_t wave = 0; wave < 3; ++wave) {
            const auto expected = f::pending_records(family, wave);
            const auto offset = actual.batch.placements[wave].output_base + 128;
            ASSERT_GE(actual.words.size(), offset + expected.size());
            EXPECT_TRUE(
                std::equal(expected.begin(), expected.end(), actual.words.begin() + offset));
            for (uint32_t lane = 0; lane < 64; ++lane) {
                const auto& events = actual.result.architectural_exports[wave][lane].events;
                ASSERT_EQ(events.size(), 2u);
                EXPECT_EQ(events[0].exec, f::active(lane));
                if (f::active(lane)) EXPECT_EQ(*events[0].source_words[0], 0x87000000u + lane);
                const bool on = family == 3   ? f::active(lane) && lane != 40
                                : family == 4 ? lane < 4 || (lane >= 28 && lane < 36) ||
                                                    (lane >= 40 && lane < 44) || lane >= 60
                                              : f::active(lane);
                EXPECT_EQ(events[1].exec, on);
                EXPECT_EQ(events[1].source_words[0].has_value(), on);
                if (on)
                    EXPECT_EQ(*events[1].source_words[0],
                              family == 0 ? f::value(wave) : 0x87000000u + lane);
            }
        }
    }
}
TEST(FragmentPacketExportExecution, BothConditionalWaitPathsPreservePreOverwritePayload) {
    for (uint32_t scc : {0u, 1u}) {
        const auto actual = execute(15, false, scc);
        if (actual.unsupported) GTEST_SKIP() << "No queried+enabled Int64 owner";
        ASSERT_TRUE(actual.result.rejection.empty()) << actual.result.rejection;
        ASSERT_EQ(actual.result.architectural_exports.size(), 3u);
        for (uint32_t wave = 0; wave < 3; ++wave)
            for (uint32_t lane = 0; lane < 64; ++lane) {
                const auto& output = actual.result.architectural_exports[wave][lane];
                ASSERT_EQ(output.events.size(), 2u);
                EXPECT_EQ(output.events[0].exec, f::active(lane));
                if (f::active(lane)) {
                    EXPECT_EQ(*output.events[0].source_words[0], 0x87000000u + lane);
                    EXPECT_EQ(output.colors[0][0]->bits, f::value(wave));
                }
            }
    }
}
TEST(FragmentPacketExportExecution, ActiveUnavailableLateWordCannotHideBehindVmOrHostMask) {
    for (bool missing : {false, true}) {
        const auto actual = execute(4, missing);
        if (actual.unsupported) GTEST_SKIP() << "No queried+enabled Int64 owner";
        if (missing) {
            EXPECT_EQ(actual.result.rejection, "packet-runtime-vgpr-read-before-definition");
            EXPECT_EQ(actual.result.wave, 2u);
            EXPECT_EQ(actual.result.lane, 40u);
            EXPECT_EQ(actual.result.pc, 0u);
            EXPECT_TRUE(actual.result.architectural_exports.empty());
        } else {
            ASSERT_TRUE(actual.result.rejection.empty()) << actual.result.rejection;
            EXPECT_EQ(actual.result.architectural_exports[2][40].colors[0][0]->bits, 40u);
            EXPECT_FALSE(actual.result.architectural_exports[2][40].commit_eligible);
        }
    }
}
TEST(FragmentPacketExportExecution, InactivePeerAndReactivatedWordsRetainOriginalReadAuthority) {
    for (uint32_t family : {5u, 6u, 7u})
        for (bool missing : {false, true}) {
            const auto actual = execute(family, missing);
            if (actual.unsupported) GTEST_SKIP() << "No queried+enabled Int64 owner";
            if (missing) {
                EXPECT_EQ(actual.result.rejection, "packet-runtime-vgpr-read-before-definition");
                EXPECT_EQ(actual.result.wave, 2u);
                EXPECT_EQ(actual.result.pc, family == 5 ? 0u : family == 6 ? 1u : 3u);
                EXPECT_TRUE(actual.result.architectural_exports.empty());
            } else {
                ASSERT_TRUE(actual.result.rejection.empty()) << actual.result.rejection;
                const uint32_t expected = family == 5   ? 0x87000028u
                                          : family == 6 ? 0x7700003fu
                                                        : 0x3f800000u;
                EXPECT_EQ(actual.result.architectural_exports[2][63].colors[0][0]->bits, expected);
            }
        }
}
} // namespace
