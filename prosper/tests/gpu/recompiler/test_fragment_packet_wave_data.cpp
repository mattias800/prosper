// Cache/data boundary: one actual SOURCE module, independently executed workgroup ordinals,
// different scalar/primitive/resource facts and all-wave transactional production publication.
#include "fixtures/fragment_packet_wave_fixture.hpp"
#include "bpermute_spirv_oracle.hpp"
#include <gtest/gtest.h>
#include <cstdlib>
#include <filesystem>
#include <fstream>

namespace {
using namespace prosper::gpu;
namespace fixture = prosper::test::fragment_packet_wave;
FragmentPacketKernel compile_kernel(const FragmentResourcePacket& input, const char* scenario) {
    auto compiled = recompile_fragment_packet_kernel(input);
    // Diagnostic CPU-test-only retention at actual compile success. No default files, no changed
    // lowering, and no assumption that factory representatives equal these actual test modules.
    const char* root = std::getenv("PROSPER_PACKET_WAVE_SPV_DIRECTORY");
    if (compiled.program.packet.spirv.empty() || !root || !*root) return compiled;
    const auto* test = ::testing::UnitTest::GetInstance()->current_test_info();
    EXPECT_NE(test, nullptr);
    if (!test) return compiled;
    const auto name = std::string(test->name()) + "_" + scenario;
    std::error_code error;
    std::filesystem::create_directories(root, error);
    EXPECT_FALSE(error) << name << ": SOURCE directory: " << error.message();
    if (error) return compiled;
    std::ofstream file(std::filesystem::path(root) / (name + ".spv"), std::ios::binary);
    const auto& words = compiled.program.packet.spirv;
    file.write(reinterpret_cast<const char*>(words.data()),
               static_cast<std::streamsize>(words.size() * sizeof(uint32_t)));
    file.close();
    EXPECT_TRUE(bool(file)) << name << ": actual cached-kernel SOURCE retention";
    return compiled;
}
std::shared_ptr<const FragmentPacketKernel> kernel() {
    return std::make_shared<const FragmentPacketKernel>(compile_kernel(fixture::packet(), "base"));
}
std::vector<FragmentResourcePacket> inputs() {
    return {fixture::packet(0), fixture::packet(1), fixture::packet(2)};
}
std::vector<uint32_t> execute(const FragmentPacketWaveBatch& batch,
                              const std::vector<uint32_t>* authority_override = nullptr,
                              bool supply_authority = true, uint32_t only_wave = UINT32_MAX) {
    auto words = batch.output_words;
    // Sequential independent Workgroup storage scopes; NOT a model of GPU inter-workgroup order.
    for (uint32_t wave : {2u, 0u, 1u}) {
        if (only_wave != UINT32_MAX && only_wave != wave) continue;
        bpermute_oracle::Interpreter vm(batch.kernel->program.packet.spirv);
        for (uint32_t slot = 0; slot < batch.images.size(); ++slot)
            for (const auto& mip : batch.images[slot].mips)
                vm.sampled_images[16 + slot].push_back({mip.width, mip.height, mip.texels});
        std::map<uint32_t, std::vector<uint32_t>> buffers{{0, batch.input_words}, {1, words}};
        if (supply_authority)
            buffers.emplace(2, authority_override ? *authority_override : batch.authority->words());
        words = vm.run_buffers(64, buffers, 1, wave);
        EXPECT_TRUE(vm.error.empty()) << vm.error;
        if (!vm.error.empty()) return {};
    }
    return words;
}
TEST(FragmentPacketWaveData, SharedImageBindingIsDataNotCachedPrototype) {
    auto original = fixture::resource::chain();
    const auto code =
        std::make_shared<const FragmentPacketKernel>(compile_kernel(original, "shared_image"));
    ASSERT_FALSE(code->program.packet.spirv.empty());
    auto waves = std::vector<FragmentResourcePacket>{original, original, original};
    waves[1].buffers[0].words[1] = fixture::resource::bits(0.5f);
    auto batch = pack_fragment_packet_waves(code, waves, fixture::placements(*code, 3));
    ASSERT_TRUE(batch.rejection.empty());
    auto decoded =
        decode_fragment_packet_waves(batch, execute(batch), true, original.device.device_identity);
    ASSERT_EQ(decoded.exports.size(), 3u) << decoded.rejection;
    auto expected = fixture::resource::expected_chain(original, false);
    EXPECT_EQ(decoded.exports[0], expected);
    EXPECT_EQ(decoded.exports[2], expected);
    for (uint32_t lane = 0; lane < 64; ++lane)
        expected[lane * 36 + 11] = fixture::resource::bits(0.75f);
    EXPECT_EQ(decoded.exports[1], expected);
    waves[2].images[0].mips[0].texels[0][0] ^= 1;
    EXPECT_EQ(pack_fragment_packet_waves(code, waves).rejection,
              "packet-wave-independent-image-binding-unimplemented");
}
TEST(FragmentPacketWaveData, OneCachedProgramDistinctWavesAndBases) {
    const auto code = kernel();
    ASSERT_FALSE(code->program.packet.spirv.empty());
    const auto waves = inputs();
    const auto places = fixture::placements(*code, 3);
    const auto batch = pack_fragment_packet_waves(code, waves, places);
    ASSERT_TRUE(batch.rejection.empty()) << batch.rejection;
    const auto decoded =
        decode_fragment_packet_waves(batch, execute(batch), true, waves[0].device.device_identity);
    ASSERT_TRUE(decoded.rejection.empty()) << decoded.rejection;
    ASSERT_EQ(decoded.exports.size(), 3u);
    for (uint32_t wave = 0; wave < 3; ++wave)
        EXPECT_EQ(decoded.exports[wave], fixture::expected(wave));
    EXPECT_TRUE(code->program.packet.input_words.empty())
        << "cached code retains no prototype launch bytes";
    EXPECT_EQ(batch.kernel.get(), code.get());
    const auto other = compile_kernel(waves[2], "different_wave");
    EXPECT_EQ(other.program.packet.spirv, code->program.packet.spirv)
        << "values/EXEC/availability/M0 must not enter code";
    auto changed_mode = waves[0];
    changed_mode.invocation.float_mode.value = 0x31;
    changed_mode.launch_rsrc1.value =
        (changed_mode.launch_rsrc1.value & ~(255u << 12)) | (0x31u << 12);
    const auto mode_kernel = compile_kernel(changed_mode, "changed_mode");
    ASSERT_FALSE(mode_kernel.program.packet.spirv.empty());
    EXPECT_NE(mode_kernel.program.packet.spirv, code->program.packet.spirv)
        << "an actual code-generation mode must not share the old cached module";
}
TEST(FragmentPacketWaveData, MissingInactivePeerInLateWaveBlocksWholeBatch) {
    const auto code = kernel();
    auto waves = inputs();
    for (auto& column : waves[2].invocation.vgprs)
        if (column.reg == 8) column.available_mask &= ~(uint64_t{1} << 40);
    const auto batch = pack_fragment_packet_waves(code, waves, fixture::placements(*code, 3));
    ASSERT_TRUE(batch.rejection.empty());
    const auto decoded =
        decode_fragment_packet_waves(batch, execute(batch), true, waves[0].device.device_identity);
    EXPECT_TRUE(decoded.exports.empty());
    EXPECT_EQ(decoded.wave, 2u);
    EXPECT_EQ(decoded.rejection, "packet-runtime-vgpr-read-before-definition");
    EXPECT_EQ(decoded.pc, 7u);
}
TEST(FragmentPacketWaveData, GenuineDynamicEntryM0AndLateDescriptorMismatch) {
    std::vector<FragmentResourcePacket> waves;
    for (uint32_t wave = 0; wave < 3; ++wave) waves.push_back(fixture::entry_m0_packet(wave));
    const auto code =
        std::make_shared<const FragmentPacketKernel>(compile_kernel(waves[0], "entry_m0"));
    ASSERT_FALSE(code->program.packet.spirv.empty());
    auto batch = pack_fragment_packet_waves(code, waves, fixture::placements(*code, 3));
    ASSERT_TRUE(batch.rejection.empty()) << batch.rejection;
    auto decoded =
        decode_fragment_packet_waves(batch, execute(batch), true, waves[0].device.device_identity);
    ASSERT_EQ(decoded.exports.size(), 3u) << decoded.rejection;
    for (uint32_t wave = 0; wave < 3; ++wave)
        EXPECT_EQ(decoded.exports[wave], fixture::expected(wave));
    waves[2].invocation.sgprs.back().second ^=
        1;   // actual s3 no longer equals separately owned V#
    batch = pack_fragment_packet_waves(code, waves, fixture::placements(*code, 3));
    ASSERT_TRUE(batch.rejection.empty());
    decoded =
        decode_fragment_packet_waves(batch, execute(batch), true, waves[0].device.device_identity);
    EXPECT_TRUE(decoded.exports.empty());
    EXPECT_EQ(decoded.wave, 2u);
    EXPECT_EQ(decoded.pc, 0u);
    EXPECT_EQ(decoded.rejection, "packet-runtime-descriptor-mismatch");
    waves[2] = fixture::entry_m0_packet(2);
    batch = pack_fragment_packet_waves(code, waves, fixture::placements(*code, 3));
    batch.input_words[batch.placements[2].input_base + 2 + code->layout.entry_m0_available_offset] =
        0;
    decoded =
        decode_fragment_packet_waves(batch, execute(batch), true, waves[0].device.device_identity);
    EXPECT_TRUE(decoded.exports.empty());
    EXPECT_EQ(decoded.wave, 2u);
    EXPECT_EQ(decoded.rejection, "packet-wave-metadata-or-completion-invalid");
}
TEST(FragmentPacketWaveData, DynamicScalarPairRestoresExecInBothLogicalHalves) {
    std::vector<FragmentResourcePacket> waves;
    for (uint32_t wave = 0; wave < 3; ++wave) waves.push_back(fixture::scalar_exec_packet(wave));
    const auto code =
        std::make_shared<const FragmentPacketKernel>(compile_kernel(waves[0], "scalar_exec"));
    ASSERT_FALSE(code->program.packet.spirv.empty());
    const auto other = compile_kernel(waves[2], "different_wave");
    EXPECT_EQ(other.program.packet.spirv, code->program.packet.spirv);
    const auto batch = pack_fragment_packet_waves(code, waves, fixture::placements(*code, 3));
    ASSERT_TRUE(batch.rejection.empty()) << batch.rejection;
    const auto decoded =
        decode_fragment_packet_waves(batch, execute(batch), true, waves[0].device.device_identity);
    ASSERT_EQ(decoded.exports.size(), 3u) << decoded.rejection;
    for (uint32_t wave = 0; wave < 3; ++wave)
        EXPECT_EQ(decoded.exports[wave], fixture::expected(wave));
}
TEST(FragmentPacketWaveData, CorruptedOutputRoutingCannotWriteAnotherOwnedRegion) {
    const auto code = kernel();
    const auto waves = inputs();
    auto batch = pack_fragment_packet_waves(code, waves, fixture::placements(*code, 3));
    ASSERT_TRUE(batch.rejection.empty());
    const auto positive =
        decode_fragment_packet_waves(batch, execute(batch), true, waves[0].device.device_identity);
    ASSERT_EQ(positive.exports.size(), 3u) << positive.rejection;
    for (uint32_t wave = 0; wave < 3; ++wave)
        EXPECT_EQ(positive.exports[wave], fixture::expected(wave));
    // Fault only mutable binding0. The private binding2 placement owner stays unchanged.
    batch.input_words[5 + 2 * 2] = batch.placements[0].output_base;
    EXPECT_EQ(execute(batch, nullptr, true, 2), batch.output_words)
        << "rejected WG must leave the ENTIRE output unchanged, including the alias target";
    const auto raw = execute(batch);
    ASSERT_EQ(raw.size(), batch.output_words.size());
    const auto base = batch.placements[2].output_base;
    const auto span = code->layout.output_words + kPacketWaveOutputPrefix;
    EXPECT_TRUE(std::equal(raw.begin() + base, raw.begin() + base + span,
                           batch.output_words.begin() + base));
    const auto refused =
        decode_fragment_packet_waves(batch, raw, true, waves[0].device.device_identity);
    EXPECT_TRUE(refused.exports.empty());
    EXPECT_EQ(refused.wave, 2u);
    EXPECT_EQ(refused.rejection, "packet-wave-metadata-or-completion-invalid");
}
TEST(FragmentPacketWaveData, MissingTruncatedOrMismatchedAuthorityNeverGrantsOwnership) {
    const auto code = kernel();
    const auto waves = inputs();
    const auto batch = pack_fragment_packet_waves(code, waves, fixture::placements(*code, 3));
    ASSERT_TRUE(batch.rejection.empty());
    const auto assert_untouched = [&](const std::vector<uint32_t>& raw) {
        EXPECT_EQ(raw, batch.output_words)
            << "no guest output before authority extent and identity";
        EXPECT_EQ(decode_fragment_packet_waves(batch, raw, true, waves[0].device.device_identity)
                      .rejection,
                  "packet-wave-metadata-or-completion-invalid");
    };
    assert_untouched(execute(batch, nullptr, false));   // actual binding2 absent, ArrayLength0
    for (const auto length : {0u, 1u, 5u, 6u, 11u}) {
        auto truncated = batch.authority->words();
        truncated.resize(length);
        assert_untouched(execute(batch, &truncated));
    }
    for (uint32_t word = 0; word < kPacketWaveAuthorityHeaderWords; ++word) {
        auto mismatched = batch.authority->words();
        mismatched[word] ^= 1;
        assert_untouched(execute(batch, &mismatched));
    }
    // These are typed SOURCE transport-envelope controls, not mutable private authority APIs.
    // Consumer separately requires the retained exact kernel/placements/extents, not public copies.
    const auto good = execute(batch);
    auto changed = batch;
    changed.authority.reset();
    EXPECT_EQ(decode_fragment_packet_waves(changed, good, true, waves[0].device.device_identity)
                  .rejection,
              "packet-wave-ownership-unproved");
    changed = batch;
    changed.placements[2].output_base = changed.placements[0].output_base;
    EXPECT_EQ(decode_fragment_packet_waves(changed, good, true, waves[0].device.device_identity)
                  .rejection,
              "packet-wave-ownership-unproved");
    changed = batch;
    changed.kernel = std::make_shared<const FragmentPacketKernel>(*code);
    EXPECT_EQ(decode_fragment_packet_waves(changed, good, true, waves[0].device.device_identity)
                  .rejection,
              "packet-wave-ownership-unproved");
    changed = batch;
    changed.input_words.push_back(0);
    EXPECT_EQ(decode_fragment_packet_waves(changed, good, true, waves[0].device.device_identity)
                  .rejection,
              "packet-wave-ownership-unproved");
}
TEST(FragmentPacketWaveData, UnownedInBoundsPaddingIsNotOutputAuthority) {
    const auto code = kernel();
    const auto waves = inputs();
    auto batch = pack_fragment_packet_waves(code, waves, fixture::padding_placements(*code));
    ASSERT_TRUE(batch.rejection.empty()) << batch.rejection;
    const auto positive =
        decode_fragment_packet_waves(batch, execute(batch), true, waves[0].device.device_identity);
    ASSERT_EQ(positive.exports.size(), 3u) << positive.rejection;
    for (uint32_t wave = 0; wave < 3; ++wave)
        EXPECT_EQ(positive.exports[wave], fixture::expected(wave));
    const auto span = code->layout.output_words + kPacketWaveOutputPrefix;
    const auto padding = 11 + span;
    ASSERT_LE(padding + span, batch.placements[1].output_base);
    batch.input_words[9] = padding;   // mutable wave2 route; private authority/extents unchanged
    EXPECT_EQ(execute(batch, nullptr, true, 2), batch.output_words)
        << "isolated refused WG leaves the complete output, including safe padding, untouched";
    const auto raw = execute(batch);
    ASSERT_EQ(raw.size(), batch.output_words.size());
    for (uint32_t base : {padding, batch.placements[2].output_base})
        EXPECT_TRUE(std::equal(raw.begin() + base, raw.begin() + base + span,
                               batch.output_words.begin() + base));
    const auto refused =
        decode_fragment_packet_waves(batch, raw, true, waves[0].device.device_identity);
    EXPECT_TRUE(refused.exports.empty());
    EXPECT_EQ(refused.wave, 2u);
    EXPECT_EQ(refused.rejection, "packet-wave-metadata-or-completion-invalid");
    for (uint32_t wave = 0; wave < 2; ++wave) {
        const auto expected = fixture::expected(wave);
        EXPECT_TRUE(
            std::equal(expected.begin(), expected.end(),
                       raw.begin() + batch.placements[wave].output_base + kPacketWaveOutputPrefix));
    }
}
TEST(FragmentPacketWaveData, EveryWaveKnownnessProfileAndOriginalIdentity) {
    const auto code = kernel();
    auto waves = inputs();
    const auto reject = [&](const char* reason) {
        EXPECT_EQ(pack_fragment_packet_waves(code, waves).rejection, reason);
    };
    waves[2].invocation.sgprs.pop_back();
    reject("packet-wave-sgpr-presence-unavailable");
    waves = inputs();
    waves[2].invocation.guest_code.back() = 0xbf800000u;
    reject("packet-wave-original-code-association-mismatch");
    waves = inputs();
    waves[2].invocation.float_mode.value ^= 1;
    reject("packet-wave-code-generation-profile-mismatch");
    waves = inputs();
    waves[2].launch_rsrc1.available = false;
    waves[2].launch_rsrc1.value = 0;
    reject("packet-f32-launch-rsrc1-unavailable");
    waves = inputs();
    waves[2].buffers[0].pc += 1;
    reject("packet-buffer-readpoint-unavailable");
    waves = inputs();
    waves[2].invocation.slots_available[63] = false;
    reject("packet-wave-invocation-state-unavailable");
    waves = inputs();
    ASSERT_TRUE(pack_fragment_packet_waves(code, waves).rejection.empty());
}
TEST(FragmentPacketWaveData, InvalidBasesTagsAndCompletionCannotPublish) {
    const auto code = kernel();
    const auto waves = inputs();
    auto places = fixture::placements(*code, 3);
    places[2].input_base = places[0].input_base;
    EXPECT_EQ(pack_fragment_packet_waves(code, waves, places).rejection,
              "packet-wave-base-overlap");
    places = fixture::placements(*code, 3);
    places[2].output_base = UINT32_MAX;
    EXPECT_EQ(pack_fragment_packet_waves(code, waves, places).rejection,
              "packet-wave-base-range-invalid");
    auto batch = pack_fragment_packet_waves(code, waves, fixture::placements(*code, 3));
    batch.input_words[8] = batch.placements[0].input_base;   // wrong wave2 input ordinal tag
    auto decoded =
        decode_fragment_packet_waves(batch, execute(batch), true, waves[0].device.device_identity);
    EXPECT_TRUE(decoded.exports.empty());
    EXPECT_EQ(decoded.rejection, "packet-wave-metadata-or-completion-invalid");
    batch = pack_fragment_packet_waves(code, waves, fixture::placements(*code, 3));
    batch.input_words[batch.placements[2].input_base + 2 +
                      code->layout.scalar_available_offsets.back()] = 0;
    decoded =
        decode_fragment_packet_waves(batch, execute(batch), true, waves[0].device.device_identity);
    EXPECT_TRUE(decoded.exports.empty());
    EXPECT_EQ(decoded.wave, 2u);
    EXPECT_EQ(decoded.rejection, "packet-wave-metadata-or-completion-invalid");
    batch = pack_fragment_packet_waves(code, waves, fixture::placements(*code, 3));
    const auto good = execute(batch);
    EXPECT_TRUE(decode_fragment_packet_waves(batch, good, false, waves[0].device.device_identity)
                    .exports.empty());
    EXPECT_EQ(decode_fragment_packet_waves(batch, good, true, waves[0].device.device_identity + 1)
                  .rejection,
              "packet-wave-completion-or-device-unproved");
    auto bad = good;
    bad[0] ^= 1;
    EXPECT_EQ(
        decode_fragment_packet_waves(batch, bad, true, waves[0].device.device_identity).rejection,
        "packet-wave-output-guard-corrupted");
}
}   // namespace
