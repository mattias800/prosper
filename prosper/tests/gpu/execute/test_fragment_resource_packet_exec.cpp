// Execute actual ORIGINAL owned logical64 guest resource/interpolation/numeric instructions.
// Same-device queried+enabled capabilities feed the compiler BEFORE dispatch; actual synchronized
// raw status readback feeds the PRODUCTION all-or-nothing consumer, never a test-only success path.
#include "fixtures/compute_runner.h"
#include "fixtures/fragment_resource_packet_fixture.hpp"
#include <cstdio>
#include <cstring>
#include <gtest/gtest.h>

namespace {
using namespace prosper::gpu;
namespace fixture = prosper::test::fragment_resource_packet;
int dispatches = 0;
bool unsupported_owner = false;
void check(bool value, const std::string& name) {
    EXPECT_TRUE(value) << name;
}
class FragmentResourcePacketExecution : public testing::Test {
    void SetUp() override {
        unsupported_owner = false; dispatches = 0;
        if (!prosper::test::default_compute_subgroup_properties().size)
            GTEST_SKIP() << "No Vulkan compute device; no execution credited";
    }
    void TearDown() override { RecordProperty("owned_dispatch_attempts", dispatches); }
};
struct Execution {
    FragmentResourcePacketProgram program;
    FragmentResourcePacketResult result;
    std::vector<uint32_t> raw;
};
Execution execute(FragmentResourcePacket input, const std::string& name) {
    Execution result;
    prosper::test::ComputeOwnedDispatch owner;
    owner.prepare = [&](const prosper::test::ComputeEnabledContract& enabled, prosper::test::ComputeOwnedPlan& plan) {
        if (!enabled.shader_int64_enabled || (!input.images.empty() && !enabled.rgba32_sfloat_sampled)) {
            unsupported_owner = true; return false;
        }
        input.device = {enabled.device_identity, enabled.shader_int64_enabled, enabled.rgba32_sfloat_sampled};
        result.program = recompile_fragment_resource_packet(input, {RecompileDiagnosticStage::Fragment, 0x4192});
        if (result.program.packet.spirv.empty()) return false;
        plan.spirv = result.program.packet.spirv;
        plan.input.resize(result.program.packet.input_words.size());
        std::memcpy(plan.input.data(), result.program.packet.input_words.data(), plan.input.size() * 4);
        plan.output_words = static_cast<uint32_t>(result.program.packet.output_words.size());
        for (const auto& image : result.program.images) {
            std::vector<prosper::test::ComputeSampledMip> mips;
            for (const auto& mip : image.mips) mips.push_back({mip.width, mip.height, mip.texels});
            plan.images.push_back(std::move(mips));
        }
        return true;
    };
    const auto floats = prosper::test::run_compute({}, {}, 64, 0, {}, {}, nullptr, 64,
                                                  nullptr, nullptr, nullptr, 0, &owner);
    ++dispatches;
    if (unsupported_owner) return result;
    check(!result.program.packet.spirv.empty(), name + " exact enabled-owner compiler: " + result.program.packet.rejection);
    result.raw.resize(floats.size());
    if (!floats.empty()) std::memcpy(result.raw.data(), floats.data(), floats.size() * 4);
    check(result.raw.size() == result.program.packet.output_words.size() && owner.completion_and_host_availability,
          name + " complete same-owner synchronized raw readback");
    result.result = decode_fragment_resource_packet(result.program, result.raw,
        owner.completion_and_host_availability, owner.enabled.device_identity);
    return result;
}
void expect(const Execution& actual, const std::vector<uint32_t>& wanted, const std::string& name) {
    check(actual.result.rejection.empty(), name + " production transaction: " + actual.result.rejection);
    check(actual.result.exports == wanted, name + " exact all logical64 raw sinks");
    if (actual.result.exports.size() == wanted.size())
        for (size_t word = 0; word < wanted.size(); ++word)
            check(actual.result.exports[word] == wanted[word], name + " word=" + std::to_string(word));
}
std::vector<uint32_t> numeric(uint32_t raw) {
    std::vector<uint32_t> words(64 * 12, 0);
    for (uint32_t lane = 0; lane < 64; ++lane) {
        const uint32_t header[]{1, 1, 1, 0, 1, 0, 1, 1};
        std::copy(std::begin(header), std::end(header), words.begin() + lane * 12);
        words[lane * 12 + 8] = raw;
    }
    return words;
}
} // namespace
TEST_F(FragmentResourcePacketExecution, OriginalOwnedResourceChains) {
    for (bool lod : {false, true}) for (bool inactive : {false, true}) {
        std::vector<uint32_t> baseline_spirv, baseline_words;
        for (uint32_t variant : {0u, 1u}) {
            const auto input = fixture::chain(lod, inactive, variant);
            const auto name = "chain_lod" + std::to_string(lod) + "_inactive40_" + std::to_string(inactive) + "_resource" + std::to_string(variant);
            const auto actual = execute(input, name);
            if (unsupported_owner) GTEST_SKIP() << "Queried+enabled Int64/RGBA32F owner unavailable; no execution credited";
            expect(actual, fixture::expected_chain(input, lod, variant), name);
            if (!variant) { baseline_spirv = actual.program.packet.spirv; baseline_words = actual.result.exports; }
            else {
                check(actual.program.packet.spirv == baseline_spirv, name + " resource bytes not baked into instruction module");
                check(actual.result.exports != baseline_words, name + " actual changed buffer/image changes live raw sinks");
            }
        }
    }
}
TEST_F(FragmentResourcePacketExecution, LargerRectangularExplicitMip) {
    std::vector<uint32_t> baseline_spirv, baseline_words;
    for (uint32_t variant : {0u, 1u}) {
        const auto actual = execute(fixture::rectangular_chain(variant), "rectangular_mip4_" + std::to_string(variant));
        if (unsupported_owner) GTEST_SKIP() << "Queried+enabled Int64/RGBA32F owner unavailable";
        expect(actual, fixture::expected_rectangular_chain(variant), "128x64 six-mip raw word execution");
        if (!variant) { baseline_spirv = actual.program.packet.spirv; baseline_words = actual.result.exports; }
        else {
            EXPECT_EQ(actual.program.packet.spirv, baseline_spirv);
            EXPECT_NE(actual.result.exports, baseline_words);
        }
    }
}
TEST_F(FragmentResourcePacketExecution, IntegerArithmeticPreservesActualGuestModes) {
    for (uint32_t denorm = 0; denorm < 4; ++denorm) for (uint32_t round = 0; round < 4; ++round) {
        const auto mode = static_cast<uint8_t>((denorm << 4) | round);
        const auto suffix = std::to_string(mode);
        const auto halfway = execute(fixture::arithmetic(3, 0x3f800000, 0x33800000, mode), "halfway_" + suffix);
        if (unsupported_owner) GTEST_SKIP() << "Queried+enabled shaderInt64 unavailable";
        expect(halfway,
               numeric(round == 1 ? 0x3f800001 : 0x3f800000), "halfway_" + suffix);
        expect(execute(fixture::arithmetic(3, 0x3f800000, 0xab800000, mode), "distant_subtract_" + suffix),
               numeric(round == 2 || round == 3 ? 0x3f7fffff : 0x3f800000), "distant_subtract_" + suffix);
        expect(execute(fixture::arithmetic(8, 0x00800000, 0x3f000000, mode), "output_denorm_" + suffix),
               numeric(denorm & 2 ? 0x00400000 : 0), "output_denorm_" + suffix);
        expect(execute(fixture::arithmetic(3, 1, 1, mode), "input_denorm_" + suffix),
               numeric(denorm == 3 ? 2 : 0), "input_denorm_" + suffix);
    }
}
TEST_F(FragmentResourcePacketExecution, FailedHighLaneSuppressesEveryExport) {
    auto bad = fixture::chain(); bad.invocation.vgprs[0].words[63] = 0x7fc00001;
    auto result = execute(bad, "failure_lane63");
    if (unsupported_owner) GTEST_SKIP() << "Queried+enabled Int64/RGBA32F owner unavailable";
    check(result.result.exports.empty() && result.result.failure == FragmentPacketRuntimeFailure::InterpolationNotExact && result.result.lane == 63,
          "one high-half failure suppresses ALL exports after all workers complete");
    bad = fixture::chain(); for (auto& [reg, value] : bad.invocation.sgprs) if (reg == 16) value ^= 1;
    result = execute(bad, "failure_original_m0");
    check(result.result.exports.empty() && result.result.failure == FragmentPacketRuntimeFailure::M0Mismatch,
          "actual reaching original M0 identity required");
}
