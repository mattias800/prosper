// Real workgroup synchronization proof for saved raw words; the CPU VM cannot prove barriers.
#include "fixtures/compute_runner.h"
#include "fixtures/fragment_packet_raw_masks_fixture.hpp"
#include <gtest/gtest.h>
#include <cstring>

TEST(FragmentPacketRawMasksExec, BothCurrentHalvesOverwriteTransferAndLogicalPositions) {
    using namespace prosper::gpu;
    namespace f = prosper::test::fragment_raw_masks;
    const auto queried = prosper::test::default_compute_subgroup_properties();
    ASSERT_GT(queried.size, 0u) << "no runtime cannot pass";
    reset_float_controls_support_for_test();
    uint32_t attempts = 0;
    auto cases = f::cases();
    const auto joins = f::joins();
    cases.insert(cases.end(), joins.begin(), joins.end());
    cases.push_back(f::reused_producer());
    cases.push_back(f::wqm_saved());
    cases.push_back(f::saveexec_scc(false));
    cases.push_back(f::saveexec_scc(true));
    cases.push_back(f::scalar_data_add());
    cases.push_back(f::not_scc(false));
    cases.push_back(f::not_scc(true));
    for (const auto& c : cases) {
        SCOPED_TRACE(c.name);
        const auto p = recompile_fragment_packet(c.packet);
        ASSERT_FALSE(p.spirv.empty()) << p.rejection;
        prosper::test::ComputeOwnedDispatch owner;
        owner.prepare = [&](const auto&, auto& plan) {
            plan.spirv = p.spirv;
            plan.input.resize(p.input_words.size());
            std::memcpy(plan.input.data(), p.input_words.data(), p.input_words.size() * 4);
            plan.output_words = static_cast<uint32_t>(p.output_words.size());
            return true;
        };
        ++attempts;
        const auto floats = prosper::test::run_compute({}, {}, 64, 0, {}, {}, nullptr, 64, nullptr,
                                                       nullptr, nullptr, 0, &owner);
        ASSERT_TRUE(owner.completion_and_host_availability);
        std::vector<uint32_t> words(floats.size());
        if (!floats.empty()) std::memcpy(words.data(), floats.data(), words.size() * 4);
        const auto result =
            decode_fragment_packet(p, words, owner.completion_and_host_availability);
        EXPECT_TRUE(result.rejection.empty()) << result.rejection;
        EXPECT_EQ(result.exports, c.expected) << "all64 independent raw EXP words";
    }
    EXPECT_EQ(attempts, f::kCases + 11);
    std::fprintf(stderr,
                 "[raw-mask-gpu] queried_physical_subgroup_size=%u attempts=%u expected=%u\n",
                 queried.size, attempts, f::kCases + 11);
}
TEST(FragmentPacketRawMasksExec, CachedMaskMaterializationIsWorkgroupIndependent) {
    using namespace prosper::gpu;
    namespace f = prosper::test::fragment_raw_masks;
    const auto cases = f::distinct_wave_cases();
    FragmentPacketWaveBatch batch;
    prosper::test::ComputeOwnedDispatch owner;
    owner.prepare = [&](const auto& enabled, auto& plan) {
        std::vector<FragmentResourcePacket> waves;
        for (const auto& c : cases) {
            auto wave = f::wave_input(c);
            wave.device = {enabled.device_identity, enabled.shader_int64_enabled,
                           enabled.rgba32_sfloat_sampled};
            waves.push_back(std::move(wave));
        }
        const auto code = std::make_shared<const FragmentPacketKernel>(
            recompile_fragment_packet_kernel(waves[0]));
        batch = pack_fragment_packet_waves(
            code, waves, prosper::test::fragment_packet_wave::placements(*code, 3));
        if (!batch.rejection.empty()) return false;
        plan.spirv = code->program.packet.spirv;
        plan.input.resize(batch.input_words.size());
        std::memcpy(plan.input.data(), batch.input_words.data(), plan.input.size() * 4);
        plan.output_words = static_cast<uint32_t>(batch.output_words.size());
        plan.initial_output = batch.output_words;
        plan.readonly_words = {batch.authority, &batch.authority->words()};
        plan.wave_count = 3;
        return true;
    };
    const auto floats = prosper::test::run_compute({}, {}, 64, 0, {}, {}, nullptr, 64, nullptr,
                                                   nullptr, nullptr, 0, &owner);
    ASSERT_TRUE(owner.completion_and_host_availability);
    ASSERT_EQ(owner.dispatch_attempts, 1u);
    std::vector<uint32_t> words(floats.size());
    if (!floats.empty()) std::memcpy(words.data(), floats.data(), words.size() * 4);
    const auto result = decode_fragment_packet_waves(
        batch, words, owner.completion_and_host_availability, owner.enabled.device_identity);
    ASSERT_TRUE(result.rejection.empty()) << result.rejection;
    ASSERT_EQ(result.exports.size(), 3u);
    for (uint32_t wave = 0; wave < 3; ++wave) EXPECT_EQ(result.exports[wave], cases[wave].expected);
    std::fprintf(
        stderr, "[raw-mask-gpu] queried_physical_subgroup_size=%u attempts=%u workgroups=3\n",
        prosper::test::default_compute_subgroup_properties().size, owner.dispatch_attempts);
}
