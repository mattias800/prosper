// The mechanical CFG split must retain exact-PC descriptor use, region bounds and entry-mask
// queries. These test the exported helpers directly, including matched unsafe uses.
#include "gpu/recompiler/rdna2_cfg_use_proofs.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <vector>
#include <unordered_set>

using namespace prosper::gpu;
namespace {
Rdna2Inst move(uint32_t pc, int dst, Operand source) {
    Rdna2Inst in{};
    in.fmt = Rdna2Format::SOP1;
    in.opcode = kSop1OpcodeMovB32;
    in.pc = pc;
    in.len_dwords = 1;
    in.dst = {OperandKind::SGPR, dst};
    in.src[0] = source;
    in.n_src = 1;
    return in;
}
TEST(CfgUseProofs, EntryM0SaveUsesHalfOpenBoundsAndTheFirstMatchingSave) {
    const std::vector<Rdna2Inst> code{move(4, 7, {OperandKind::Special, 124}),
                                      move(8, 9, {OperandKind::Special, 124})};
    EXPECT_EQ(entry_m0_save_in_range(code, 4, 8), 7);
    EXPECT_EQ(entry_m0_save_in_range(code, 5, 9), 9);
    EXPECT_EQ(entry_m0_save_in_range(code, 4, 9), 7);
    EXPECT_EQ(entry_m0_save_in_range(code, 8, 8), -1);
    EXPECT_EQ(entry_m0_save_in_range({}, 0, 9), -1);
}
TEST(CfgUseProofs, EntryM0SaveDoesNotAcceptOtherSourcesWidthsOrMaskDestinations) {
    auto good = move(4, 7, {OperandKind::Special, 124});
    EXPECT_EQ(entry_m0_save_in_range({good}, 0, 5), 7);
    auto wrong = good;
    wrong.src[0].kind = OperandKind::SGPR;
    EXPECT_EQ(entry_m0_save_in_range({wrong}, 0, 5), -1);
    wrong = good;
    wrong.src[0].value = 123;
    EXPECT_EQ(entry_m0_save_in_range({wrong}, 0, 5), -1);
    wrong = good;
    wrong.opcode = kSop1OpcodeMovB64;
    EXPECT_EQ(entry_m0_save_in_range({wrong}, 0, 5), -1);
    wrong = good;
    wrong.dst.value = 106;
    EXPECT_EQ(entry_m0_save_in_range({wrong}, 0, 5), -1);
}
TEST(CfgUseProofs, SavedMaskCopiesRetainEntryOwnershipAndRegionBounds) {
    const std::vector<Rdna2Inst> code{move(4, 30, {OperandKind::SGPR, 20})};
    RegState plain;
    EXPECT_FALSE(has_unpersisted_b32_mask_lifetime(code, 4, 5, plain));
    RegState mask;
    mask.sreg_bool[20] = 1;
    EXPECT_TRUE(has_unpersisted_b32_mask_lifetime(code, 4, 5, mask));
    EXPECT_FALSE(has_unpersisted_b32_mask_lifetime(code, 0, 4, mask));
    EXPECT_FALSE(has_unpersisted_b32_mask_lifetime(code, 5, 9, mask));
    RegState narrow;
    narrow.sreg_bool_b32.insert(20);
    EXPECT_TRUE(has_unpersisted_b32_mask_lifetime(code, 4, 5, narrow));
}
TEST(CfgUseProofs, OnlyArchitecturalLowMaskWordsSeedTheUnpersistedCopyQuery) {
    RegState entry;
    for (int source : {106, 126})
        EXPECT_TRUE(has_unpersisted_b32_mask_lifetime({move(4, 30, {OperandKind::Special, source})},
                                                      4, 5, entry));
    for (int source : {107, 127})
        EXPECT_FALSE(has_unpersisted_b32_mask_lifetime(
            {move(4, 30, {OperandKind::Special, source})}, 4, 5, entry));
    auto write_exec = move(4, 126, {OperandKind::InlineInt, 0});
    EXPECT_TRUE(has_unpersisted_b32_mask_lifetime({write_exec}, 4, 5, entry));
}
std::vector<Rdna2Inst> descriptor_program() {
    Rdna2Inst load{};
    load.fmt = Rdna2Format::SMEM;
    load.opcode = kSmemOpcodeLoadDwordX2;
    load.pc = 0;
    load.len_dwords = 2;
    load.dst = {OperandKind::SGPR, 16};
    load.src[0] = {OperandKind::SGPR, 4};
    load.src[1] = {OperandKind::SGPR, 8};
    load.n_src = 2;
    Rdna2Inst fetch{};
    fetch.fmt = Rdna2Format::MUBUF;
    fetch.opcode = 0;
    fetch.pc = 2;
    fetch.len_dwords = 2;
    fetch.src[0] = {OperandKind::VGPR, 0};
    fetch.src[1] = {OperandKind::SGPR, 16};
    fetch.src[2] = {OperandKind::InlineInt, 0};
    fetch.n_src = 3;
    Rdna2Inst end{};
    end.pc = 4;
    end.len_dwords = 1;
    end.fmt = Rdna2Format::SOPP;
    end.opcode = 0x01;
    end.is_end = true;
    return {load, fetch, end};
}
ShaderResourceTable exact_resource() {
    ShaderResourceTable table;
    ShaderResource resource{};
    resource.cls = ResourceClass::VertexBuffer;
    resource.fetch_pc = 2;
    table.resources.push_back(resource);
    return table;
}
TEST(CfgUseProofs, DescriptorFragmentNeedsAnExactKeylessBufferConsumer) {
    const auto code = descriptor_program();
    for (uint32_t wave : {32u, 64u}) {
        auto table = exact_resource();
        EXPECT_EQ(proven_smem_x2_descriptor_fragment_loads(code, &table, wave),
                  (std::unordered_set<uint32_t>{0}));
        EXPECT_TRUE(proven_smem_x2_descriptor_fragment_loads(code, nullptr, wave).empty());
        table.resources[0].fetch_pc = 3;
        EXPECT_TRUE(proven_smem_x2_descriptor_fragment_loads(code, &table, wave).empty());
        table = exact_resource();
        table.resources[0].sgpr_base = 16;
        EXPECT_TRUE(proven_smem_x2_descriptor_fragment_loads(code, &table, wave).empty());
        table = exact_resource();
        table.resources[0].cls = ResourceClass::Texture;
        EXPECT_TRUE(proven_smem_x2_descriptor_fragment_loads(code, &table, wave).empty());
    }
}
TEST(CfgUseProofs, DescriptorConsumerDoesNotHideAnOrdinaryScalarObservation) {
    auto code = descriptor_program();
    const auto table = exact_resource();
    EXPECT_FALSE(proven_smem_x2_descriptor_fragment_loads(code, &table, 64).empty());
    code.insert(code.begin() + 2, move(4, 30, {OperandKind::SGPR, 16}));
    code.back().pc = 5;
    EXPECT_TRUE(proven_smem_x2_descriptor_fragment_loads(code, &table, 64).empty());
    code[2] = move(4, 30, {OperandKind::SGPR, 18});
    EXPECT_FALSE(proven_smem_x2_descriptor_fragment_loads(code, &table, 64).empty());
}
} // namespace
