#include "fixtures/compute_runner.h"
#include "fixtures/fragment_special_f32_fixture.hpp"
#include <cstdio>
#include <cstring>
#include <gtest/gtest.h>

namespace {
using namespace prosper::gpu;
namespace fixture = prosper::test::fragment_special_f32;
class FragmentSpecialF32Execution : public testing::Test {
protected:
    uint32_t dispatches = 0;
    bool unavailable = false;
    void SetUp() override {
        if (!prosper::test::default_compute_subgroup_properties().size)
            GTEST_SKIP() << "No Vulkan device; no execution credited";
    }
    void TearDown() override {
        RecordProperty("owned_dispatch_attempts", dispatches);
        const auto* info = testing::UnitTest::GetInstance()->current_test_info();
        std::printf("[fragment-special-f32-execution] case=%s.%s owned_dispatch_attempts=%u\n",
                    info->test_suite_name(), info->name(), dispatches);
    }
    FragmentResourcePacketResult execute(FragmentResourcePacket input) {
        FragmentResourcePacketProgram program;
        prosper::test::ComputeOwnedDispatch owner;
        owner.prepare = [&](const prosper::test::ComputeEnabledContract& enabled,
                            prosper::test::ComputeOwnedPlan& plan) {
            if (!enabled.shader_int64_enabled) {
                unavailable = true;
                return false;
            }
            input.device = {enabled.device_identity, enabled.shader_int64_enabled,
                            enabled.rgba32_sfloat_sampled};
            program = recompile_fragment_resource_packet(
                input, {RecompileDiagnosticStage::Fragment, 0x4224});
            if (program.packet.spirv.empty()) return false;
            plan.spirv = program.packet.spirv;
            plan.input.resize(program.packet.input_words.size());
            std::memcpy(plan.input.data(), program.packet.input_words.data(),
                        plan.input.size() * 4);
            plan.output_words = static_cast<uint32_t>(program.packet.output_words.size());
            return true;
        };
        const auto floats = prosper::test::run_compute({}, {}, 64, 0, {}, {}, nullptr, 64, nullptr,
                                                       nullptr, nullptr, 0, &owner);
        ++dispatches;
        if (unavailable) return {};
        EXPECT_FALSE(program.packet.spirv.empty()) << program.packet.rejection;
        std::vector<uint32_t> raw(floats.size());
        if (!floats.empty()) std::memcpy(raw.data(), floats.data(), raw.size() * 4);
        EXPECT_EQ(raw.size(), program.packet.output_words.size());
        EXPECT_TRUE(owner.completion_and_host_availability);
        return decode_fragment_resource_packet(program, raw, owner.completion_and_host_availability,
                                               owner.enabled.device_identity);
    }
    void expect(const FragmentResourcePacket& p, uint32_t op) {
        const auto result = execute(p);
        if (unavailable) return;
        ASSERT_TRUE(result.rejection.empty()) << result.rejection;
        const auto wanted = fixture::expected(p, op);
        ASSERT_EQ(result.exports.size(), wanted.size());
        for (size_t i = 0; i < wanted.size(); ++i)
            EXPECT_EQ(result.exports[i], wanted[i]) << "word=" << i;
    }
};
} // namespace

TEST_F(FragmentSpecialF32Execution, AllExponentsOriginalUnaryAndRawPeerSinks) {
    for (uint32_t op : {0x2au, 0x2eu, 0x33u})
        for (uint32_t round = 0; round < 4; ++round)
            for (uint32_t batch = 0; batch < 4; ++batch) {
                std::array<uint32_t, 64> words{};
                for (uint32_t lane = 0; lane < 64; ++lane) {
                    const uint32_t e = std::min(254u, 1 + batch * 64 + lane);
                    constexpr uint32_t frac[]{1, 0x155555, 0x3fffff, 0x7fffff};
                    words[lane] = (e << 23) | frac[lane % 4];
                    if (op == 0x2a && lane & 1) words[lane] |= 0x80000000;
                }
                SCOPED_TRACE("op=" + std::to_string(op) + " round=" + std::to_string(round) +
                             " batch=" + std::to_string(batch));
                expect(fixture::packet(op, words, uint8_t(0x30 | round)), op);
                if (unavailable)
                    GTEST_SKIP()
                        << "Queried AND enabled shaderInt64 unavailable; no execution credited";
            }
}
TEST_F(FragmentSpecialF32Execution, FlushSignedZeroInfinityAndInactiveReadlane) {
    for (uint32_t op : {0x2au, 0x2eu, 0x33u})
        for (uint32_t denorm = 0; denorm < 4; ++denorm)
            for (uint32_t round = 0; round < 4; ++round) {
                const auto mode = uint8_t((denorm << 4) | round);
                SCOPED_TRACE("op=" + std::to_string(op) + " mode=" + std::to_string(mode));
                expect(fixture::packet(op, fixture::rails(op), mode, round == 3), op);
                if (unavailable)
                    GTEST_SKIP()
                        << "Queried AND enabled shaderInt64 unavailable; no execution credited";
            }
}
TEST_F(FragmentSpecialF32Execution, StickyHighLaneFailurePublishesNothing) {
    for (uint32_t op : {0x2au, 0x2eu, 0x33u}) {
        auto inputs = fixture::rails();
        inputs[63] = 0x7fc00001;
        auto p = fixture::packet(op, inputs);
        p.invocation.guest_code.insert(p.invocation.guest_code.end() - 1,
                                       {0x7e000000u | (9u << 17) | (0x33u << 9) | 256u});
        const auto result = execute(p);
        if (unavailable)
            GTEST_SKIP() << "Queried AND enabled shaderInt64 unavailable; no execution credited";
        EXPECT_TRUE(result.exports.empty());
        EXPECT_EQ(result.lane, 63u);
        EXPECT_EQ(result.pc, 0u);
        EXPECT_EQ(result.failure, FragmentPacketRuntimeFailure::SpecialNanOrNegativeRoot);
        EXPECT_EQ(result.rejection,
                  "packet-runtime-special-f32-nan-or-negative-root-unimplemented");
    }
    const auto result = execute(fixture::high_pc_failure());
    if (unavailable) GTEST_SKIP() << "Queried AND enabled shaderInt64 unavailable";
    EXPECT_TRUE(result.exports.empty());
    EXPECT_EQ(result.lane, 63u);
    EXPECT_EQ(result.pc, 257u);
    EXPECT_EQ(result.failure, FragmentPacketRuntimeFailure::SpecialNanOrNegativeRoot);
}
