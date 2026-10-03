#include "fixtures/compute_runner.h"
#include "fixtures/fragment_packet_definedness_fixture.hpp"
#include <gtest/gtest.h>
#include <cstring>

namespace {
using namespace prosper::gpu;
namespace f = prosper::test::fragment_definedness;
namespace r = prosper::test::fragment_resource_packet;
void input_plan(const FragmentPacketProgram& p, prosper::test::ComputeOwnedPlan& plan) {
    plan.spirv = p.spirv;
    plan.input.resize(p.input_words.size());
    std::memcpy(plan.input.data(), p.input_words.data(), plan.input.size() * sizeof(uint32_t));
    plan.output_words = static_cast<uint32_t>(p.output_words.size());
}
std::vector<uint32_t> dispatch(prosper::test::ComputeOwnedDispatch& owner, uint32_t& attempts) {
    ++attempts;
    const auto floats = prosper::test::run_compute({}, {}, 64, 0, {}, {}, nullptr, 64,
                                                   nullptr, nullptr, nullptr, 0, &owner);
    std::vector<uint32_t> words(floats.size());
    if (!floats.empty()) std::memcpy(words.data(), floats.data(), words.size() * sizeof(uint32_t));
    EXPECT_TRUE(owner.completion_and_host_availability) << "same-owner complete synchronized readback";
    return words;
}
}

TEST(FragmentPacketDefinednessExec, ActualMaskBranchPeerAndInactiveRawOutput) {
    const auto queried = prosper::test::default_compute_subgroup_properties();
    std::fprintf(stderr, "[vgpr-definedness-gpu] queried_physical_subgroup_size=%u expected_dispatch_attempts=16\n", queried.size);
    ASSERT_GT(queried.size, 0u) << "no runtime cannot pass";
    reset_float_controls_support_for_test();
    uint32_t attempts = 0;
    for (const auto& c : f::cases()) {
        SCOPED_TRACE(c.name);
        const auto p = recompile_fragment_packet(c.packet);
        ASSERT_FALSE(p.spirv.empty()) << p.rejection;
        prosper::test::ComputeOwnedDispatch owner;
        owner.prepare = [&](const auto&, auto& plan) { input_plan(p, plan); return true; };
        const auto words = dispatch(owner, attempts);
        const auto result = decode_fragment_packet(p, words, owner.completion_and_host_availability);
        EXPECT_EQ(result.exports, c.expected) << "all64 independent raw EXP records";
        if (c.failure_kind) {
            EXPECT_FALSE(result.rejection.empty()); EXPECT_TRUE(result.exports.empty());
            EXPECT_EQ(result.lane, c.failure_lane); EXPECT_EQ(result.pc, c.failure_pc);
            EXPECT_EQ(result.reg, c.failure_reg); EXPECT_EQ(result.kind, c.failure_kind);
        } else EXPECT_TRUE(result.rejection.empty()) << result.rejection;
    }
    EXPECT_EQ(attempts, 16u);
    std::fprintf(stderr, "[vgpr-definedness-gpu] integer_dispatch_attempts=%u expected=16\n", attempts);
}

TEST(FragmentPacketDefinednessExec, ActualOwnedResourceWriterAndPartialEntryChain) {
    const auto queried = prosper::test::default_compute_subgroup_properties();
    std::fprintf(stderr, "[vgpr-definedness-gpu] queried_physical_subgroup_size=%u expected_dispatch_attempts=5\n", queried.size);
    ASSERT_GT(queried.size, 0u) << "no runtime cannot pass";
    uint32_t attempts = 0;
    const auto execute = [&](auto input, const auto& expected, bool absent_lod) {
        FragmentResourcePacketProgram p;
        prosper::test::ComputeOwnedDispatch owner;
        owner.prepare = [&](const auto& enabled, auto& plan) {
            input.device = {enabled.device_identity, enabled.shader_int64_enabled, enabled.rgba32_sfloat_sampled};
            p = recompile_fragment_resource_packet(input);
            if (p.packet.spirv.empty()) return false;
            input_plan(p.packet, plan);
            for (const auto& image : p.images) {
                std::vector<prosper::test::ComputeSampledMip> mips;
                for (const auto& mip : image.mips) mips.push_back({mip.width, mip.height, mip.texels});
                plan.images.push_back(std::move(mips));
            }
            return true;
        };
        const auto words = dispatch(owner, attempts);
        EXPECT_FALSE(p.packet.spirv.empty()) << p.packet.rejection;
        const auto result = decode_fragment_resource_packet(p, words, owner.completion_and_host_availability,
                                                           owner.enabled.device_identity);
        if (absent_lod) {
            EXPECT_FALSE(result.rejection.empty()); EXPECT_TRUE(result.exports.empty());
            EXPECT_EQ(result.failure, FragmentPacketRuntimeFailure::UndefinedVgpr);
            EXPECT_EQ(result.pc, input.images[0].pc); EXPECT_EQ(result.vgpr, 12u);
        } else EXPECT_TRUE(result.rejection.empty()) << result.rejection;
        EXPECT_EQ(result.exports, expected) << "actual P1/P2/FP/resource/wide/peer all64 raw sink";
    };
    for (bool lod : {false, true}) for (bool inactive : {false, true}) {
        const auto input = f::resource_chain(lod, inactive);
        execute(input, r::expected_chain(input, lod), false);
    }
    execute(f::resource_missing_lod(), std::vector<uint32_t>{}, true);
    EXPECT_EQ(attempts, 5u);
    std::fprintf(stderr, "[vgpr-definedness-gpu] resource_dispatch_attempts=%u expected=5\n", attempts);
}
