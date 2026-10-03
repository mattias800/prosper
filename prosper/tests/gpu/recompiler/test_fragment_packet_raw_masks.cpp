// A scalar half overwrite must preserve the other actual 32 mask bits. A Bool alias/zero bank
// cannot supply descriptor arithmetic or a later EXEC transfer. Evaluate the original SOURCE sinks.
#include "bpermute_spirv_oracle.hpp"
#include "fixtures/fragment_packet_raw_masks_fixture.hpp"
#include <gtest/gtest.h>
#include <cstdlib>
#include <filesystem>
#include <fstream>

namespace {
using namespace prosper::gpu;
namespace f = prosper::test::fragment_raw_masks;
void retain(const FragmentPacketProgram& p, const std::string& name) {
    const char* root = std::getenv("PROSPER_RAW_MASK_SPV_DIRECTORY");
    if (!root || !*root) return;
    std::filesystem::create_directories(root);
    std::ofstream file(std::filesystem::path(root) / (name + ".spv"), std::ios::binary);
    file.write(reinterpret_cast<const char*>(p.spirv.data()),
               static_cast<std::streamsize>(p.spirv.size() * sizeof(uint32_t)));
    file.close();
    EXPECT_TRUE(bool(file)) << "actual CPU SOURCE retention";
}
void evaluate(const f::Case& c) {
    SCOPED_TRACE(c.name);
    const auto p = recompile_fragment_packet(c.packet);
    ASSERT_FALSE(p.spirv.empty()) << p.rejection;
    bpermute_oracle::Interpreter vm(p.spirv);
    const auto words = vm.run_packet(p.input_words, p.output_words);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    const auto result = decode_fragment_packet(p, words, true);
    EXPECT_TRUE(result.rejection.empty()) << result.rejection;
    EXPECT_EQ(result.exports, c.expected) << "independent all64 physical words/EXEC/raw EXP";
    retain(p, c.name);
}
}   // namespace
TEST(FragmentPacketRawMasks, BothCurrentHalvesOverwriteTransferAndLogicalPositions) {
    reset_float_controls_support_for_test();
    const auto cases = f::cases();
    ASSERT_EQ(cases.size(), f::kCases);
    for (const auto& c : cases) evaluate(c);
    std::fprintf(stderr, "[raw-mask-source] evaluations=%zu expected=%u\n", cases.size(),
                 f::kCases);
}
TEST(FragmentPacketRawMasks, BothConditionalPredecessorsSupplyCurrentPhysicalWords) {
    for (const auto& c : f::joins()) evaluate(c);
}
TEST(FragmentPacketRawMasks, ReusedEventsAndEarlierWqmServicePublishActualWords) {
    evaluate(f::reused_producer());
    evaluate(f::wqm_saved());
}
TEST(FragmentPacketRawMasks, SaveexecMaterializesOldMaskButSccTestsNewExec) {
    evaluate(f::saveexec_scc(false));
    evaluate(f::saveexec_scc(true));
}
TEST(FragmentPacketRawMasks, PhysicalWordsFeedScalarDataWrappingArithmetic) {
    evaluate(f::scalar_data_add());
}
TEST(FragmentPacketRawMasks, NotSccTestsTheCompleteCurrentInvertedMask) {
    evaluate(f::not_scc(false));
    evaluate(f::not_scc(true));
}
TEST(FragmentPacketRawMasks, CompleteWriterDoesNotResurrectEntryWordsAndAbsenceStillRefuses) {
    auto c = f::overwritten(f::asymmetric, true, true);
    std::erase_if(c.packet.sgprs,
                  [](const auto& word) { return word.first == 20 || word.first == 21; });
    c.name = "saved_mask_no_old_entry_pair";
    evaluate(c);
    for (uint32_t absent : {30u, 31u}) {
        auto input = c.packet;
        std::erase_if(input.sgprs, [&](const auto& word) { return word.first == absent; });
        const auto p = recompile_fragment_packet(input);
        EXPECT_TRUE(p.spirv.empty());
        EXPECT_EQ(p.rejection, "packet-sgpr-read-before-definition");
    }
}
TEST(FragmentPacketRawMasks, OneCachedKernelIndependentMasksAndCompletionOrder) {
    const auto cases = f::distinct_wave_cases();
    std::vector<FragmentResourcePacket> waves;
    for (const auto& c : cases) waves.push_back(f::wave_input(c));
    const auto code =
        std::make_shared<const FragmentPacketKernel>(recompile_fragment_packet_kernel(waves[0]));
    ASSERT_FALSE(code->program.packet.spirv.empty()) << code->program.packet.rejection;
    const auto other = recompile_fragment_packet_kernel(waves[2]);
    EXPECT_EQ(other.program.packet.spirv, code->program.packet.spirv)
        << "different masks/scalar values are DATA, not compiler specialization";
    retain(code->program.packet, "cached_mask_kernel_prototype");
    retain(other.program.packet, "cached_mask_kernel_distinct_wave");
    const auto batch = pack_fragment_packet_waves(
        code, waves, prosper::test::fragment_packet_wave::placements(*code, 3));
    ASSERT_TRUE(batch.rejection.empty()) << batch.rejection;
    auto words = batch.output_words;
    for (uint32_t wave : {2u, 0u, 1u}) {
        bpermute_oracle::Interpreter vm(code->program.packet.spirv);
        words = vm.run_buffers(
            64, {{0, batch.input_words}, {1, words}, {2, batch.authority->words()}}, 1, wave);
        ASSERT_TRUE(vm.error.empty()) << vm.error;
    }
    const auto result =
        decode_fragment_packet_waves(batch, words, true, waves[0].device.device_identity);
    ASSERT_TRUE(result.rejection.empty()) << result.rejection;
    ASSERT_EQ(result.exports.size(), 3u);
    for (uint32_t wave = 0; wave < 3; ++wave) EXPECT_EQ(result.exports[wave], cases[wave].expected);
}
