#include "fixtures/compute_runner.h"
#include "fixtures/fragment_packet_mask_entry_fixture.hpp"
#include <gtest/gtest.h>
#include <cstring>

TEST(FragmentPacketMaskEntryExec, OriginalSourceStateDefinitionsAndAbsentUnusedInputs) {
    using namespace prosper::gpu;
    namespace f = prosper::test::fragment_mask_entry;
    const auto queried = prosper::test::default_compute_subgroup_properties();
    ASSERT_GT(queried.size, 0u) << "unavailable runtime must never pass";
    reset_float_controls_support_for_test();
    uint32_t attempts = 0;
    const auto cases = f::cases();
    ASSERT_EQ(cases.size(), f::kCases);
    for (const auto& c : cases) {
        SCOPED_TRACE(c.name);
        const auto program = recompile_fragment_packet(c.packet);
        ASSERT_FALSE(program.spirv.empty()) << program.rejection;
        prosper::test::ComputeOwnedDispatch owner;
        owner.prepare = [&](const auto&, auto& plan) {
            plan.spirv = program.spirv;
            plan.input.resize(program.input_words.size());
            std::memcpy(plan.input.data(), program.input_words.data(),
                        program.input_words.size() * 4);
            plan.output_words = static_cast<uint32_t>(program.output_words.size());
            return true;
        };
        ++attempts;
        const auto floats = prosper::test::run_compute({}, {}, 64, 0, {}, {}, nullptr, 64, nullptr,
                                                       nullptr, nullptr, 0, &owner);
        ASSERT_TRUE(owner.completion_and_host_availability);
        ASSERT_EQ(owner.dispatch_attempts, 1u);
        std::vector<uint32_t> words(floats.size());
        if (!words.empty()) std::memcpy(words.data(), floats.data(), words.size() * 4);
        const auto result =
            decode_fragment_packet(program, words, owner.completion_and_host_availability);
        ASSERT_TRUE(result.rejection.empty()) << result.rejection;
        EXPECT_EQ(result.exports, c.expected) << "all64 independent original raw sinks";
    }
    EXPECT_EQ(attempts, f::kCases);
    std::fprintf(stderr,
                 "[mask-entry-gpu] queried_physical_subgroup_size=%u attempts=%u expected=%u\n",
                 queried.size, attempts, f::kCases);
}
TEST(FragmentPacketMaskEntryExec, CachedAvailabilityProfileAndIndependentWaveData) {
    using namespace prosper::gpu;
    namespace f = prosper::test::fragment_mask_entry;
    const auto c = f::cases().front();
    FragmentPacketWaveBatch batch;
    prosper::test::ComputeOwnedDispatch owner;
    owner.prepare = [&](const auto& enabled, auto& plan) {
        auto prototype = f::wave_input(c);
        prototype.device = {enabled.device_identity, enabled.shader_int64_enabled,
                            enabled.rgba32_sfloat_sampled};
        auto opposite = prototype;
        opposite.invocation.exec_mask = 0;
        opposite.invocation.vcc_mask = ~prototype.invocation.vcc_mask;
        opposite.invocation.scc = !prototype.invocation.scc;
        const auto kernel = std::make_shared<const FragmentPacketKernel>(
            recompile_fragment_packet_kernel(prototype));
        batch = pack_fragment_packet_waves(
            kernel, std::vector<FragmentResourcePacket>{prototype, opposite, prototype});
        if (!batch.rejection.empty()) return false;
        plan.spirv = kernel->program.packet.spirv;
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
    if (!words.empty()) std::memcpy(words.data(), floats.data(), words.size() * 4);
    const auto result = decode_fragment_packet_waves(
        batch, words, owner.completion_and_host_availability, owner.enabled.device_identity);
    ASSERT_TRUE(result.rejection.empty()) << result.rejection;
    ASSERT_EQ(result.exports.size(), 3u);
    EXPECT_EQ(result.exports[0], c.expected);
    EXPECT_EQ(result.exports[1], f::fd::expected(0, 0x42230011u));
    EXPECT_EQ(result.exports[2], c.expected);
    std::fprintf(
        stderr, "[mask-entry-gpu] queried_physical_subgroup_size=%u attempts=%u workgroups=3\n",
        prosper::test::default_compute_subgroup_properties().size, owner.dispatch_attempts);
}
