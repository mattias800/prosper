// Real multiple-workgroup original-program execution through the existing enabled-owner runner.
// CPU WorkgroupId simulation is not GPU barrier evidence. No raster/attachment admission claimed.
#include "fixtures/compute_runner.h"
#include "fixtures/fragment_packet_wave_fixture.hpp"
#include <gtest/gtest.h>

namespace {
using namespace prosper::gpu;
namespace fixture = prosper::test::fragment_packet_wave;
struct Execution {
    FragmentPacketWaveBatch batch;
    FragmentPacketWaveResult result;
    std::vector<uint32_t> raw;
    bool unsupported = false;
};
Execution execute(uint32_t fault = 0, bool image = false) {
    Execution result;
    if (!prosper::test::default_compute_subgroup_properties().size) {
        result.unsupported = true;
        return result;
    }
    prosper::test::ComputeOwnedDispatch owner;
    owner.prepare = [&](const prosper::test::ComputeEnabledContract& enabled,
                        prosper::test::ComputeOwnedPlan& plan) {
        if (!enabled.shader_int64_enabled || (image && !enabled.rgba32_sfloat_sampled)) {
            result.unsupported = true;
            return false;
        }
        std::vector<FragmentResourcePacket> waves;
        for (uint32_t wave = 0; wave < 3; ++wave) {
            auto input = image        ? fixture::resource::chain()
                         : fault == 3 ? fixture::entry_m0_packet(wave)
                         : fault == 5 ? fixture::scalar_exec_packet(wave)
                                      : fixture::packet(wave);
            input.device = {enabled.device_identity, enabled.shader_int64_enabled,
                            enabled.rgba32_sfloat_sampled};
            waves.push_back(std::move(input));
        }
        if (image) waves[1].buffers[0].words[1] = fixture::resource::bits(0.5f);
        const auto kernel = std::make_shared<const FragmentPacketKernel>(
            recompile_fragment_packet_kernel(waves[0]));
        if (fault == 1)
            for (auto& column : waves[2].invocation.vgprs)
                if (column.reg == 8) column.available_mask &= ~(uint64_t{1} << 40);
        result.batch = pack_fragment_packet_waves(kernel, waves, fixture::placements(*kernel, 3));
        if (!result.batch.rejection.empty()) return false;
        if (fault == 2) result.batch.input_words[8] = result.batch.placements[0].input_base;
        if (fault == 4)
            result.batch.input_words[result.batch.placements[2].input_base + 2 +
                                     kernel->layout.scalar_available_offsets.back()] = 0;
        if (fault == 6)
            result.batch.input_words[5 + 2 * 2] = result.batch.placements[0].output_base;
        plan.spirv = kernel->program.packet.spirv;
        plan.input.resize(result.batch.input_words.size());
        std::memcpy(plan.input.data(), result.batch.input_words.data(), plan.input.size() * 4);
        plan.output_words = static_cast<uint32_t>(result.batch.output_words.size());
        plan.initial_output = result.batch.output_words;
        // Alias the immutable typed authority owner, not a mutable upload-vector reconstruction.
        plan.readonly_words = {result.batch.authority, &result.batch.authority->words()};
        plan.wave_count = 3;
        for (const auto& binding : result.batch.images) {
            std::vector<prosper::test::ComputeSampledMip> mips;
            for (const auto& mip : binding.mips)
                mips.push_back({mip.width, mip.height, mip.texels});
            plan.images.push_back(std::move(mips));
        }
        return true;
    };
    const auto raw = prosper::test::run_compute({}, {}, 64, 0, {}, {}, nullptr, 64, nullptr,
                                                nullptr, nullptr, 0, &owner);
    result.raw.resize(raw.size());
    if (!raw.empty()) std::memcpy(result.raw.data(), raw.data(), raw.size() * 4);
    result.result = decode_fragment_packet_waves(result.batch, result.raw,
                                                 owner.completion_and_host_availability,
                                                 owner.enabled.device_identity);
    std::printf("[fragment-packet-wave-execution] workgroups=%u attempted_dispatches=%u "
                "physical_subgroup=%u fault=%u image=%u\n",
                owner.dispatch_attempts ? owner.plan.wave_count : 0, owner.dispatch_attempts,
                prosper::test::default_compute_subgroup_properties().size, fault, image);
    EXPECT_EQ(owner.dispatch_attempts, 1u);
    return result;
}
TEST(FragmentPacketWaveExecution, OneKernelDistinctNoncontiguousWaves) {
    const auto actual = execute();
    if (actual.unsupported) GTEST_SKIP() << "No queried+enabled Int64 owner";
    ASSERT_TRUE(actual.result.rejection.empty()) << actual.result.rejection;
    ASSERT_EQ(actual.result.exports.size(), 3u);
    for (uint32_t wave = 0; wave < 3; ++wave)
        EXPECT_EQ(actual.result.exports[wave], fixture::expected(wave));
}
TEST(FragmentPacketWaveExecution, LateInvalidPeerBlocksAllEarlierExports) {
    const auto actual = execute(1);
    if (actual.unsupported) GTEST_SKIP() << "No queried+enabled Int64 owner";
    EXPECT_EQ(actual.result.rejection, "packet-runtime-vgpr-read-before-definition");
    EXPECT_EQ(actual.result.wave, 2u);
    EXPECT_EQ(actual.result.pc, 7u);
    EXPECT_TRUE(actual.result.exports.empty());
}
TEST(FragmentPacketWaveExecution, WrongInputOrdinalNeverAliasesAnotherWave) {
    const auto actual = execute(2);
    if (actual.unsupported) GTEST_SKIP() << "No queried+enabled Int64 owner";
    EXPECT_EQ(actual.result.rejection, "packet-wave-metadata-or-completion-invalid");
    EXPECT_TRUE(actual.result.exports.empty());
}
TEST(FragmentPacketWaveExecution, SuppliedEntryM0IsPerWaveData) {
    const auto actual = execute(3);
    if (actual.unsupported) GTEST_SKIP() << "No queried+enabled Int64 owner";
    ASSERT_TRUE(actual.result.rejection.empty()) << actual.result.rejection;
    ASSERT_EQ(actual.result.exports.size(), 3u);
    for (uint32_t wave = 0; wave < 3; ++wave)
        EXPECT_EQ(actual.result.exports[wave], fixture::expected(wave));
}
TEST(FragmentPacketWaveExecution, DynamicScalarPairRestoresBothExecHalves) {
    const auto actual = execute(5);
    if (actual.unsupported) GTEST_SKIP() << "No queried+enabled Int64 owner";
    ASSERT_TRUE(actual.result.rejection.empty()) << actual.result.rejection;
    ASSERT_EQ(actual.result.exports.size(), 3u);
    for (uint32_t wave = 0; wave < 3; ++wave)
        EXPECT_EQ(actual.result.exports[wave], fixture::expected(wave));
}
TEST(FragmentPacketWaveExecution, CorruptedOutputRoutingNeverWritesAliasedWave) {
    const auto actual = execute(6);
    if (actual.unsupported) GTEST_SKIP() << "No queried+enabled Int64 owner";
    EXPECT_EQ(actual.result.rejection, "packet-wave-metadata-or-completion-invalid");
    EXPECT_EQ(actual.result.wave, 2u);
    EXPECT_TRUE(actual.result.exports.empty());
    ASSERT_EQ(actual.raw.size(), actual.batch.output_words.size());
    const auto base = actual.batch.placements[2].output_base;
    const auto span = actual.batch.kernel->layout.output_words + kPacketWaveOutputPrefix;
    EXPECT_TRUE(std::equal(actual.raw.begin() + base, actual.raw.begin() + base + span,
                           actual.batch.output_words.begin() + base))
        << "invalid WG must not store any guest EXP/status/provenance word";
    for (uint32_t wave = 0; wave < 2; ++wave) {
        const auto expected = fixture::expected(wave);
        EXPECT_TRUE(std::equal(expected.begin(), expected.end(),
                               actual.raw.begin() + actual.batch.placements[wave].output_base +
                                   kPacketWaveOutputPrefix))
            << "safe valid WG retains its independent raw sinks";
    }
}
TEST(FragmentPacketWaveExecution, MissingScalarPresenceNeverLeavesBarrierParticipants) {
    const auto actual = execute(4);
    if (actual.unsupported) GTEST_SKIP() << "No queried+enabled Int64 owner";
    EXPECT_EQ(actual.result.rejection, "packet-wave-metadata-or-completion-invalid");
    EXPECT_EQ(actual.result.wave, 2u);
    EXPECT_TRUE(actual.result.exports.empty());
}
TEST(FragmentPacketWaveExecution, SharedImagesDistinctBufferWords) {
    const auto actual = execute(0, true);
    if (actual.unsupported) GTEST_SKIP() << "No queried+enabled Int64/RGBA32F owner";
    ASSERT_TRUE(actual.result.rejection.empty()) << actual.result.rejection;
    ASSERT_EQ(actual.result.exports.size(), 3u);
    auto wanted = fixture::resource::expected_chain(fixture::resource::chain(), false);
    EXPECT_EQ(actual.result.exports[0], wanted);
    EXPECT_EQ(actual.result.exports[2], wanted);
    for (uint32_t lane = 0; lane < 64; ++lane)
        wanted[lane * 36 + 11] = fixture::resource::bits(0.75f);
    EXPECT_EQ(actual.result.exports[1], wanted);
}
}   // namespace
