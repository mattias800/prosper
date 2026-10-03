// Cache/data boundary: one actual SOURCE module, independently executed workgroup ordinals,
// different scalar/primitive/resource facts and all-wave transactional production publication.
#include "fixtures/fragment_packet_wave_fixture.hpp"
#include "bpermute_spirv_oracle.hpp"
#include <gtest/gtest.h>

namespace {
using namespace prosper::gpu;
namespace fixture = prosper::test::fragment_packet_wave;
std::shared_ptr<const FragmentPacketKernel> kernel() {
    return std::make_shared<const FragmentPacketKernel>(
        recompile_fragment_packet_kernel(fixture::packet()));
}
std::vector<FragmentResourcePacket> inputs() {
    return {fixture::packet(0), fixture::packet(1), fixture::packet(2)};
}
std::vector<uint32_t> execute(const FragmentPacketWaveBatch& batch) {
    auto words = batch.output_words;
    // Sequential independent Workgroup storage scopes; NOT a model of GPU inter-workgroup order.
    for (uint32_t wave : {2u, 0u, 1u}) {
        bpermute_oracle::Interpreter vm;
        vm.parse(batch.kernel->program.packet.spirv);
        for (uint32_t slot = 0; slot < batch.images.size(); ++slot)
            for (const auto& mip : batch.images[slot].mips)
                vm.sampled_images[16 + slot].push_back({mip.width, mip.height, mip.texels});
        words = vm.run_packet(batch.input_words, words, wave);
        EXPECT_TRUE(vm.error.empty()) << vm.error;
        if (!vm.error.empty()) return {};
    }
    return words;
}
TEST(FragmentPacketWaveData, SharedImageBindingIsDataNotCachedPrototype) {
    auto original = fixture::resource::chain();
    const auto code =
        std::make_shared<const FragmentPacketKernel>(recompile_fragment_packet_kernel(original));
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
    const auto other = recompile_fragment_packet_kernel(waves[2]);
    EXPECT_EQ(other.program.packet.spirv, code->program.packet.spirv)
        << "values/EXEC/availability/M0 must not enter code";
    auto changed_mode = waves[0];
    changed_mode.invocation.float_mode.value = 0x31;
    changed_mode.launch_rsrc1.value =
        (changed_mode.launch_rsrc1.value & ~(255u << 12)) | (0x31u << 12);
    const auto mode_kernel = recompile_fragment_packet_kernel(changed_mode);
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
        std::make_shared<const FragmentPacketKernel>(recompile_fragment_packet_kernel(waves[0]));
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
